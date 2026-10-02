import numpy as np
import pandas as pd
from .statistics import descriptive

CELL_KEYS = ['source','signal_name','change_days','rolling_window','percentile_bin','horizon']

def annual_cells(y, buckets, identity, config):
    from .config import BINS
    rows=[]
    eligible = buckets.notna() & y.notna()
    for year in sorted(y.index[eligible].year.unique()):
        universe = eligible & (y.index.year == year)
        benchmark = y[universe].mean()
        for bucket in BINS:
            x = y[universe & (buckets == bucket)]
            stats = descriptive(x)
            rows.append(dict(**identity, year=int(year),percentile_bin=bucket,**stats,
                             unconditional_mean_return=benchmark,
                             excess_return=stats['mean_return']-benchmark))
    return rows

def annual_summary(annual, config):
    rows=[]
    for identity, group in annual.groupby(CELL_KEYS, observed=True):
        valid = group[group.n >= config.min_sample].copy()
        all_nonempty = group[group.n > 0]
        mass = (all_nonempty.n*all_nonempty.excess_return).abs()
        total = mass.sum()
        weights = mass/total if total>0 else mass*0
        order = weights.sort_values(ascending=False)
        pos = (valid.excess_return>0).sum(); neg=(valid.excess_return<0).sum()
        count=len(valid)
        top1=float(order.iloc[0]) if len(order) else np.nan
        rows.append(dict(zip(CELL_KEYS,identity),valid_year_count=count,
                         positive_year_count=int(pos),negative_year_count=int(neg),
                         positive_year_ratio=pos/count if count else np.nan,
                         negative_year_ratio=neg/count if count else np.nan,
                         median_annual_effect=valid.excess_return.median(),annual_effect_std=valid.excess_return.std(),
                         best_year=int(valid.loc[valid.excess_return.idxmax(),'year']) if count else None,
                         worst_year=int(valid.loc[valid.excess_return.idxmin(),'year']) if count else None,
                         top_contributing_year=int(group.loc[order.index[0],'year']) if len(order) else None,
                         top_1_year_contribution=top1,top_3_year_contribution=float(order.head(3).sum()) if len(order) else np.nan,
                         annual_stability=bool(count>=config.min_years and max(pos,neg)/count>=config.annual_direction_threshold),
                         year_concentration=bool(top1>config.concentration_threshold)))
    return pd.DataFrame(rows)
