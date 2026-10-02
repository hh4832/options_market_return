import numpy as np
import pandas as pd

def bh(values):
    p = np.asarray(values, dtype=float)
    valid = np.isfinite(p)
    if ((p[valid]<0)|(p[valid]>1)).any():
        raise ValueError('Invalid p-values')
    q = np.full(p.shape, np.nan)
    pv = p[valid]
    if len(pv):
        order = np.argsort(pv)
        sorted_q = np.minimum.accumulate((pv[order]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1]
        inv = np.empty(len(pv)); inv[order] = np.minimum(sorted_q,1)
        q[valid] = inv
    return q

def apply_fdr(frame, prefixes=('zero','unconditional'), family=True):
    result = frame.copy()
    for prefix in prefixes:
        p = prefix+'_p_value'
        suffix = 'family' if family else 'global'
        qcol = p+'_fdr_'+suffix
        result[qcol] = np.nan
        groups = result.groupby(['source','signal_name']).groups.values() if family else [result.index]
        for ids in groups:
            result.loc[ids,qcol] = bh(result.loc[ids,p])
        result[prefix+'_significant_fdr_'+suffix+'_05'] = result[qcol] < .05
        result[prefix+'_significant_raw_05'] = result[p] < .05
    return result

def fdr_diagnostics(frame, prefixes=('zero','unconditional'), universe='main'):
    rows=[]
    for prefix in prefixes:
        for (source, signal), group in frame.groupby(['source','signal_name']):
            rows.append(dict(universe=universe, test=prefix, scope='family',source=source,signal_name=signal,
                             planned_tests=len(group),valid_tests=group[prefix+'_p_value'].notna().sum(),
                             discoveries=group[prefix+'_significant_fdr_family_05'].sum()))
        rows.append(dict(universe=universe,test=prefix,scope='global',source='ALL',signal_name='ALL',
                         planned_tests=len(frame),valid_tests=frame[prefix+'_p_value'].notna().sum(),
                         discoveries=frame[prefix+'_significant_fdr_global_05'].sum()))
    return pd.DataFrame(rows)
