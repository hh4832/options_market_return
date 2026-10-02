import numpy as np
import pandas as pd

def compute_outcomes(prices, signal_index, horizons):
    calendar = prices.index
    # Mapping strictly AFTER t, including option dates before 0050 inception.
    pos = calendar.searchsorted(signal_index, side='right')
    valid_entry = (pos < len(calendar)) & (signal_index >= calendar[0])
    entry_dates = np.full(len(signal_index), np.datetime64('NaT'), dtype='datetime64[ns]')
    entry_dates[valid_entry] = calendar.values[pos[valid_entry]]
    result, alignment = pd.DataFrame(index=signal_index), []
    for h in horizons:
        endpoint = pos + h - 1
        valid = valid_entry & (endpoint < len(calendar))
        returns = np.full(len(signal_index), np.nan)
        out_dates = np.full(len(signal_index), np.datetime64('NaT'), dtype='datetime64[ns]')
        for i in np.flatnonzero(valid):
            # Internal missing price excludes this path rather than skipping dates.
            path = prices.iloc[pos[i]:endpoint[i]+1][['adj_open','adj_close']]
            if not path.isna().any().any():
                returns[i] = prices.adj_close.iloc[endpoint[i]] / prices.adj_open.iloc[pos[i]] - 1
            out_dates[i] = calendar.values[endpoint[i]]
        result[h] = returns
        alignment.extend(dict(signal_date=t, entry_date=e, outcome_date=o, horizon=h,
                              return_value=r, eligible=bool(pd.notna(r)))
                         for t,e,o,r in zip(signal_index,entry_dates,out_dates,returns))
    return result, pd.DataFrame(alignment)
