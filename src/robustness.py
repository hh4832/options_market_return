import numpy as np
import pandas as pd
from .config import BINS
from .annual_analysis import CELL_KEYS

def neighbor_diagnostics(results, config):
    lookup = results.set_index(CELL_KEYS)
    rows=[]
    for _,row in results.iterrows():
        key=[row[k] for k in CELL_KEYS]
        neighbors=[]; axis_info={}
        for axis, sequence in [('rolling_window',config.rolling_windows),('percentile_bin',BINS),('horizon',config.horizons)]:
            idx=CELL_KEYS.index(axis); pos=sequence.index(key[idx]); items=[]
            for offset in (-1,1):
                if 0<=pos+offset<len(sequence):
                    nk=key.copy(); nk[idx]=sequence[pos+offset]
                    if tuple(nk) in lookup.index:
                        candidate=lookup.loc[tuple(nk)]
                        if np.isfinite(candidate.mean_excess_vs_unconditional) and candidate.n>=config.min_sample:
                            items.append(candidate); neighbors.append(candidate)
            axis_info[axis+'_neighbor_count']=len(items)
            axis_info[axis+'_direction_consistency']=float(np.mean([np.sign(n.mean_excess_vs_unconditional)==np.sign(row.mean_excess_vs_unconditional) for n in items])) if items else np.nan
        direction=float(np.mean([np.sign(n.mean_excess_vs_unconditional)==np.sign(row.mean_excess_vs_unconditional) for n in neighbors])) if neighbors else np.nan
        # Objective definition: same sign and effect magnitude between .5x and 2x.
        effect=float(np.mean([np.sign(n.mean_excess_vs_unconditional)==np.sign(row.mean_excess_vs_unconditional) and .5*abs(row.mean_excess_vs_unconditional)<=abs(n.mean_excess_vs_unconditional)<=2*abs(row.mean_excess_vs_unconditional) for n in neighbors])) if neighbors else np.nan
        significant_neighbors=sum(bool(n.unconditional_significant_fdr_family_05) for n in neighbors)
        significant=bool(row.unconditional_significant_raw_05)
        island=bool(significant and (not neighbors or direction<config.neighbor_threshold or significant_neighbors==0))
        rows.append(dict(zip(CELL_KEYS,key),neighbor_count=len(neighbors),neighbor_direction_consistency=direction,
                         neighbor_effect_consistency=effect,neighbor_family_significant_count=significant_neighbors,
                         parameter_island=island,flag='PARAMETER_ISLAND' if island else '',**axis_info))
    return pd.DataFrame(rows)

def classify(results, config):
    result=results.copy()
    result['sample_sufficiency']=result.n>=config.min_sample
    result['annual_direction_agrees']=np.sign(result.median_annual_effect)==np.sign(result.mean_excess_vs_unconditional)
    result['robustness_warning']=(~result.sample_sufficiency | ~result.annual_stability |
                                  ~result.annual_direction_agrees | result.year_concentration | result.parameter_island |
                                  (result.neighbor_direction_consistency<config.neighbor_threshold) |
                                  result.neighbor_direction_consistency.isna())
    result['evidence_level']=np.select([
        result.unconditional_significant_fdr_global_05 & ~result.robustness_warning,
        result.unconditional_significant_fdr_global_05,
        result.unconditional_significant_fdr_family_05,
        result.unconditional_significant_raw_05],['A','GLOBAL_WITH_WARNINGS','B','C'],default='D')
    return result
