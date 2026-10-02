import numpy as np
import pandas as pd
from .config import LIMITATIONS

def generate_observations(results, trend, extremes):
    rows=[]
    def emit(kind, identity, text):
        rows.append(dict(observation_type=kind,identity=identity,observation=text))
    for _,r in trend.iterrows():
        identity=f'{r.source}/{r.signal_name}/W{r.rolling_window}/C{r.horizon}'
        emit('MONOTONICITY',identity,f'七個 bin 均值 Spearman rho={r.spearman_rho:.4g}, descriptive p={r.spearman_p_value:.4g}, direction={r.effect_direction}; 不是線性或正式 HAC trend 證據。')
    for _,r in results.iterrows():
        identity=f'{r.source}/{r.signal_name}/W{r.rolling_window}/{r.percentile_bin}/C{r.horizon}'
        if r.unconditional_significant_raw_05:
            emit('HAC_AND_FDR',identity,f'excess={r.mean_excess_vs_unconditional:.6g}, HAC p={r.unconditional_p_value:.4g}, Family q={r.unconditional_p_value_fdr_family:.4g}, Global q={r.unconditional_p_value_fdr_global:.4g}; evidence={r.evidence_level}')
        if r.unconditional_significant_fdr_family_05 and not r.unconditional_significant_fdr_global_05:
            emit('FAMILY_ONLY',identity,'Family BH 通過、Global BH 未通過。')
        if r.annual_stability:
            emit('ANNUAL_STABILITY',identity,f'年度方向比例 positive={r.positive_year_ratio:.3g}, negative={r.negative_year_ratio:.3g}; eligible years={r.valid_year_count}; annual/full-sample sign agreement={r.annual_direction_agrees}')
        if r.year_concentration:
            emit('YEAR_CONCENTRATION',identity,f'top year={r.top_contributing_year}, top1={r.top_1_year_contribution:.3g}, top3={r.top_3_year_contribution:.3g} absolute annual excess mass。')
        if r.parameter_island:
            emit('PARAMETER_ISLAND',identity,f'鄰近方向一致率={r.neighbor_direction_consistency:.3g}, Family-significant neighbors={r.neighbor_family_significant_count}; 不視為強證據。')
        if r.percentile_bin in ('<5','>95') and not r.sample_sufficiency:
            emit('EXTREME_SAMPLE',identity,f'extreme bin n={r.n}, 低於預先設定最低樣本數。')
    for identity,g in results.groupby(['source','signal_name','percentile_bin'],observed=True):
        for axis in ('rolling_window','horizon'):
            effects=g.groupby(axis).mean_excess_vs_unconditional.mean().dropna()
            signs=np.sign(effects)
            emit('DIRECTION_'+axis.upper(),str(identity),f'各 {axis} 平均 cell effect 方向：'+','.join(f'{k}:{int(v):+d}' for k,v in signs.items())+'；僅描述，不合併成獨立證據。')
    for identity,g in results.groupby(['source','signal_name','rolling_window','horizon'],observed=True):
        extreme=g[g.percentile_bin.isin(['<5','>95'])]
        middle=g[~g.percentile_bin.isin(['<5','>95'])]
        if extreme.unconditional_significant_fdr_family_05.any() and not middle.unconditional_significant_fdr_family_05.any():
            emit('EXTREME_ONLY',str(identity),'僅極端 bins 通過 main comparison Family BH；其他 bins 未通過，不代表其他 bins effect=0。')
    for _,r in extremes[extremes.contrast_significant_fdr_family_05].iterrows():
        emit('EXTREME_CONTRAST',f'{r.source}/{r.signal_name}/W{r.rolling_window}/C{r.horizon}/{r.contrast}',f'effect={r.effect:.6g}, HAC p={r.contrast_p_value:.4g}, Family q={r.contrast_p_value_fdr_family:.4g}, Global q={r.contrast_p_value_fdr_global:.4g}, n={r.left_n}/{r.right_n}')
    return pd.DataFrame(rows,columns=['observation_type','identity','observation'])

def write_observations(path, frame):
    frame.to_csv(path/'observation_summary.csv',index=False)
    text='# Observations\n\n'+LIMITATIONS+'\n\n目前研究決策：修改後再測。尚未完成樣本外與可交易性驗證。\n\n'
    text+='\n'.join(f'- [{r.observation_type}] {r.identity}: {r.observation}' for r in frame.itertuples())
    (path/'observation_summary.md').write_text(text,encoding='utf-8')
