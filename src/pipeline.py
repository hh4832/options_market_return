import argparse
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import shutil
import subprocess
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from .config import BINS, Config
from .data_loader import load_finlab
from .validation import validate_options, validate_prices, DataQualityError
from .signals import build_signals, rolling_percentile, percentile_bins
from .outcomes import compute_outcomes
from .statistics import descriptive, hac_contrast
from .fdr import apply_fdr, fdr_diagnostics
from .annual_analysis import CELL_KEYS, annual_cells, annual_summary
from .robustness import neighbor_diagnostics, classify
from .diagnostics import data_reports, write_summary
from .observations import generate_observations, write_observations
from .visualization import make_plots

ROOT=Path(__file__).resolve().parents[1]
STAGES=['Loading data','Validating data','Building signals','Computing rolling percentiles',
        'Computing outcomes','Running descriptive statistics','Running HAC tests','Running Family FDR',
        'Running Global FDR','Running monotonicity diagnostics','Running extreme-value diagnostics',
        'Running annual stability','Running neighbor robustness','Generating diagnostics',
        'Generating observations','Saving outputs','Archiving to Drive']

def git_info(root=ROOT):
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    try:
        commit=git('rev-parse','HEAD')
    except subprocess.CalledProcessError:
        commit='UNCOMMITTED'
    return dict(commit=commit,branch=git('branch','--show-current'),dirty=bool(git('status','--porcelain')))

def dependency_versions():
    names=['numpy','pandas','scipy','statsmodels','matplotlib','tqdm','finlab']
    result={}
    for name in names:
        try: result[name]=version(name)
        except PackageNotFoundError: result[name]='NOT_INSTALLED'
    return result

def archive_to_drive(run, root):
    root=Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Drive root does not exist; mount/verify: {root}')
    dest=root/run.name
    # Atomic reservation forbids overwrite, including simultaneous executions.
    dest.mkdir(exist_ok=False)
    for item in run.iterdir():
        if item.is_dir(): shutil.copytree(item,dest/item.name)
        else: shutil.copy2(item,dest/item.name)
    return dest

def run_research(config=Config(), loader=load_finlab, output_root=None, archive=True, formal=True):
    info=git_info()
    if formal and (info['commit']=='UNCOMMITTED' or info['dirty']):
        raise RuntimeError('Formal research requires a committed clean checkout')
    if formal and platform.python_version_tuple()[:2] != ('3','11'):
        raise RuntimeError('Formal research requires Python 3.11')
    stamp=datetime.now(ZoneInfo(config.timezone)).strftime('%Y%m%d_%H%M%S')
    tag=info['commit'][:12] if formal else 'SYNTHETIC_VALIDATION'
    root=Path(output_root) if output_root is not None else ROOT/'outputs'
    root.mkdir(parents=True,exist_ok=True)
    run=root/f'{stamp}_{tag}'; run.mkdir(exist_ok=False)
    diagnostics=run/'diagnostics'; diagnostics.mkdir()
    observations=run/'observations'; observations.mkdir()
    metadata=dict(execution_timestamp=datetime.now(ZoneInfo(config.timezone)).isoformat(),
                  timezone=config.timezone,git=info,python_version=platform.python_version(),
                  dependency_versions=dependency_versions(),research_config=config.metadata(),
                  formal=formal,synthetic=not formal,status='RUNNING')
    def save_metadata():
        (run/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        (run/'run_info.txt').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    save_metadata()
    bar=tqdm(total=len(STAGES),desc='Research pipeline')
    def stage(name):
        bar.set_description(name); tqdm.write(name); bar.update(1)
    try:
        stage('Loading data')
        options,prices,provenance,available=loader()
        metadata.update(price_provenance=provenance,available_adjusted_keys=available)
        # Keep raw quality evidence even when validation raises.
        for name,frame in [('options',options),('prices',prices)]:
            frame.to_csv(diagnostics/f'{name}_validation_input.csv')
        stage('Validating data')
        options=validate_options(options); prices=validate_prices(prices,provenance,config.adjustment_rtol)
        metadata['dataset_date_ranges']={name:dict(start=str(frame.index.min()),end=str(frame.index.max()),rows=len(frame)) for name,frame in [('options',options),('0050',prices)]}
        coverage,missing=data_reports(options,prices)
        # Never use signals off the authoritative stock-price trading calendar.
        calendar=prices.index
        common=options.index.intersection(calendar)
        if len(common)<min(config.rolling_windows):
            raise DataQualityError('Insufficient shared trading history')
        stage('Building signals')
        signal_options=options.reindex(calendar[calendar>=options.index.min()])
        signals,ratio_check,zero_check=build_signals(signal_options,config)
        signal_diag=[]; percentile_data={}; rolling_diag=[]
        for (source,name,k),s in signals.items():
            signal_diag.append(dict(source=source,signal_name=name,change_days=k,n=len(s),valid_count=s.notna().sum(),
                                    missing_count=s.isna().sum(),mean=s.mean(),std=s.std(),min=s.min(),max=s.max(),
                                    q05=s.quantile(.05),q50=s.quantile(.5),q95=s.quantile(.95)))
        stage('Computing rolling percentiles')
        for identity,s in tqdm(signals.items(),desc='Signal definitions',leave=False):
            # Original option calendar is reindexed to full stock calendar IN shared range.
            # Missing option days consume a slot and invalidate full-window ranks.
            s=s.reindex(calendar[calendar>=options.index.min()])
            for window in config.rolling_windows:
                rank=rolling_percentile(s,window)
                bucket=percentile_bins(rank)
                percentile_data[(*identity,window)]=(rank,bucket)
                rolling_diag.append(dict(source=identity[0],signal_name=identity[1],change_days=identity[2],rolling_window=window,
                                         warm_up_loss=min(window-1,len(s)),missing_window_loss=max(0,int(rank.isna().sum())-min(window-1,len(s))),
                                         first_valid_date=rank.first_valid_index(),valid_count=rank.notna().sum(),total_count=len(rank)))
        stage('Computing outcomes')
        outcomes,alignment=compute_outcomes(prices,calendar,config.horizons)
        outcome_diag=pd.DataFrame([dict(horizon=h,valid_count=outcomes[h].notna().sum(),missing_count=outcomes[h].isna().sum(),first_valid_date=outcomes[h].first_valid_index(),last_valid_date=outcomes[h].last_valid_index()) for h in config.horizons])
        jobs=[]; rows=[]; counts=[]
        stage('Running descriptive statistics')
        for (source,name,k,window),(rank,bucket) in tqdm(percentile_data.items(),desc='Descriptive groups',leave=False):
            bucket=bucket.reindex(calendar)
            for h in config.horizons:
                y=outcomes[h]; eligible=bucket.notna() & y.notna()
                unconditional=y[eligible].mean()
                for b in BINS:
                    selected=eligible & (bucket==b)
                    identity=dict(source=source,signal_name=name,change_days=k,rolling_window=window,percentile_bin=b,horizon=h)
                    stats=descriptive(y[selected])
                    rows.append(dict(**identity,**stats,unconditional_n=int(eligible.sum()),unconditional_mean_return=unconditional,
                                     mean_excess_vs_unconditional=stats['mean_return']-unconditional))
                    jobs.append((y,selected,eligible,identity))
                    counts.append(dict(**identity,rank_count=int((bucket==b).sum()),usable_count=int(selected.sum()),
                                       bucket_fraction=float(selected.sum()/eligible.sum()) if eligible.sum() else np.nan))
        stage('Running HAC tests')
        hac_diag=[]
        for row,(y,selected,eligible,identity) in tqdm(zip(rows,jobs),total=len(rows),desc='HAC cells',leave=False):
            for prefix,right in [('zero',None),('unconditional',eligible)]:
                test=hac_contrast(y,selected,right,lag=identity['horizon'],min_sample=config.min_sample)
                row[prefix+'_hac_t']=test['hac_t']; row[prefix+'_p_value']=test['p_value']
                row[prefix+'_standard_error']=test['standard_error']
                hac_diag.append(dict(**identity,test=prefix,**test))
        results=pd.DataFrame(rows)
        stage('Running Family FDR'); results=apply_fdr(results,family=True)
        stage('Running Global FDR'); results=apply_fdr(results,family=False)
        stage('Running monotonicity diagnostics')
        trends=[]
        for identity,g in results.groupby(['source','signal_name','change_days','rolling_window','horizon'],observed=True):
            g=g.set_index('percentile_bin').reindex(BINS)
            enough=bool((g.n>=config.min_sample).all())
            if enough and g.mean_return.nunique()>1:
                rho,p=spearmanr(np.arange(7),g.mean_return)
                erho,ep=spearmanr(np.arange(7),g.mean_excess_vs_unconditional)
            else: rho=p=erho=ep=np.nan
            trends.append(dict(zip(['source','signal_name','change_days','rolling_window','horizon'],identity),
                               spearman_rho=rho,spearman_p_value=p,excess_spearman_rho=erho,excess_spearman_p_value=ep,
                               effect_direction='INCREASING' if rho>0 else 'DECREASING' if rho<0 else 'UNDETERMINED',
                               sample_sufficiency=enough,monotonicity=bool(np.isfinite(p) and p<.05)))
        trend=pd.DataFrame(trends)
        stage('Running extreme-value diagnostics')
        extreme_rows=[]; extreme_hac=[]
        contrasts=[('HIGH_UNCONDITIONAL','>95',None),('LOW_UNCONDITIONAL','<5',None),
                   ('HIGH_MIDDLE','>95','40–60'),('LOW_MIDDLE','<5','40–60'),('LOW_HIGH','<5','>95')]
        for (source,name,k,window),(_,bucket) in tqdm(percentile_data.items(),desc='Extreme contrasts',leave=False):
            bucket=bucket.reindex(calendar)
            for h in config.horizons:
                y=outcomes[h];eligible=bucket.notna() & y.notna()
                high=y[eligible & (bucket=='>95')].mean();low=y[eligible & (bucket=='<5')].mean()
                benchmark=y[eligible].mean()
                for contrast,left_bin,right_bin in contrasts:
                    left=eligible & (bucket==left_bin);right=eligible if right_bin is None else eligible & (bucket==right_bin)
                    test=hac_contrast(y,left,right,lag=h,min_sample=config.min_sample)
                    identity=dict(source=source,signal_name=name,change_days=k,rolling_window=window,horizon=h,contrast=contrast)
                    extreme_rows.append(dict(**identity,**test,contrast_p_value=test['p_value'],extreme_high_effect=high-benchmark,
                                             extreme_low_effect=low-benchmark,extreme_spread=low-high))
                    extreme_hac.append(dict(**identity,percentile_bin=left_bin,test='extreme_'+contrast,**test))
        extremes=pd.DataFrame(extreme_rows)
        # Each predeclared contrast has its own family/global universe.
        extremes=pd.concat([apply_fdr(apply_fdr(g.reset_index(drop=True),('contrast',),True),('contrast',),False)
                            for _,g in extremes.groupby('contrast')],ignore_index=True)
        stage('Running annual stability')
        annual_rows=[]
        for (source,name,k,window),(_,bucket) in tqdm(percentile_data.items(),desc='Annual groups',leave=False):
            bucket=bucket.reindex(calendar)
            for h in config.horizons:
                annual_rows.extend(annual_cells(outcomes[h],bucket,dict(source=source,signal_name=name,change_days=k,rolling_window=window,horizon=h),config))
        annual=pd.DataFrame(annual_rows)
        if annual.empty:
            raise DataQualityError('No eligible annual cells')
        annual_stats=annual_summary(annual,config)
        results=results.merge(annual_stats,on=CELL_KEYS,how='left')
        # Empty cells are explicitly non-stable/non-concentrated, never truthy NaN.
        for col in ('annual_stability','year_concentration'):
            results[col]=results[col].fillna(False).astype(bool)
        stage('Running neighbor robustness')
        neighbor=neighbor_diagnostics(results,config)
        results=classify(results.merge(neighbor,on=CELL_KEYS),config)
        results=results.merge(trend[['source','signal_name','change_days','rolling_window','horizon','monotonicity']],on=['source','signal_name','change_days','rolling_window','horizon'])
        results['extreme_effect']=results.percentile_bin.isin(['<5','>95']) & results.unconditional_significant_fdr_family_05
        stage('Generating diagnostics')
        fdrs=[fdr_diagnostics(results)]
        fdrs.extend(fdr_diagnostics(g,('contrast',),f'extreme_{c}') for c,g in extremes.groupby('contrast'))
        tables=dict(data_coverage=coverage,missing_data_report=missing,date_alignment=alignment,
                    signal_diagnostics=pd.DataFrame(signal_diag),percentile_bucket_counts=pd.DataFrame(counts),
                    rolling_window_coverage=pd.DataFrame(rolling_diag),outcome_coverage=outcome_diag,
                    hac_diagnostics=pd.DataFrame(hac_diag+extreme_hac),fdr_diagnostics=pd.concat(fdrs,ignore_index=True),
                    annual_contribution=annual_stats,parameter_neighbor_diagnostics=neighbor,
                    ratio_cross_validation=ratio_check,zero_denominators=zero_check,
                    annual_sample_counts=annual[CELL_KEYS+['year','n']])
        for name,table in tables.items(): table.to_csv(diagnostics/f'{name}.csv',index=False)
        write_summary(diagnostics,tables)
        stage('Generating observations')
        observed=generate_observations(results,trend,extremes)
        write_observations(observations,observed)
        stage('Saving outputs')
        for name,frame in [('research_results',results),('annual_results',annual),('annual_summary',annual_stats),('monotonicity_results',trend),('extreme_results',extremes)]:
            frame.to_csv(run/f'{name}.csv',index=False)
        data_dir=run/'reproducibility_data';data_dir.mkdir()
        options.to_csv(data_dir/'options.csv');prices.to_csv(data_dir/'0050_prices.csv')
        pd.DataFrame({f'{a}/{b}':v for (a,b,k),v in signals.items()}).to_csv(data_dir/'signals.csv')
        pd.DataFrame({f'{a}/{b}/W{w}':v[0] for (a,b,k,w),v in percentile_data.items()}).to_csv(data_dir/'percentiles.csv')
        outcomes.to_csv(data_dir/'outcomes.csv')
        (run/'config.json').write_text(json.dumps(config.metadata(),ensure_ascii=False,indent=2),encoding='utf-8')
        if config.plots:
            make_plots(results,run/'figures',lambda groups:tqdm(groups,desc='Heatmaps / response curves'))
        metadata.update(status='SUCCESS',research_decision='修改後再測',ratio_mismatch_rows=len(ratio_check),
                        planned_main_cells=len(results),archive_status='PENDING' if archive else 'NOT_REQUESTED')
        save_metadata()
        stage('Archiving to Drive')
        if archive:
            metadata['archive_status']='COPYING';save_metadata()
            dest=archive_to_drive(run,config.drive_output_root)
            metadata.update(archive_status='SUCCESS',drive_archive=str(dest));save_metadata()
            shutil.copy2(run/'metadata.json',dest/'metadata.json');shutil.copy2(run/'run_info.txt',dest/'run_info.txt')
        return run
    except Exception as exc:
        # Never persist token-bearing exception payloads in failure reports.
        safe=str(exc)
        import os
        for name in ('FINLAB_API_TOKEN','GITHUB_TOKEN'):
            token=os.environ.get(name)
            if token: safe=safe.replace(token,'[REDACTED]')
        metadata.update(status=getattr(exc, 'status', 'FAILED'),error_type=type(exc).__name__,error=safe)
        save_metadata()
        (diagnostics/'failure.json').write_text(json.dumps({'error_type':type(exc).__name__,'error':safe},ensure_ascii=False,indent=2),encoding='utf-8')
        (diagnostics/'diagnostics_summary.md').write_text(f'# FAILED\n{type(exc).__name__}: {safe}\n不得將本次 run 視為有效研究結果。',encoding='utf-8')
        if archive and Path(config.drive_output_root).is_dir() and not (Path(config.drive_output_root)/run.name).exists():
            try: archive_to_drive(run,config.drive_output_root)
            except Exception: pass  # primary error retained; local evidence remains
        raise RuntimeError(f'Pipeline failed; diagnostics: {run}; {type(exc).__name__}: {safe}') from None
    finally:
        bar.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--no-archive',action='store_true')
    args=parser.parse_args()
    run=run_research(Config(plots=not args.no_plots),archive=not args.no_archive)
    print(f'Output: {run}')

if __name__=='__main__': main()
