import numpy as np
import pandas as pd
from .config import BINS
from .data_loader import OPTION_COLUMNS

def safe_divide(a, b):
    return a.div(b.where(b != 0))

def build_signals(options, config):
    signals, ratio_checks, zero_checks = {}, [], []
    for source, (put_col, call_col, ratio_col) in OPTION_COLUMNS.items():
        put, call = options[put_col], options[call_col]
        bases = {'PUT': put, 'CALL': call, 'PCR': safe_divide(put, call),
                 'CALLSHARE': safe_divide(call, put + call)}
        computed = bases['PCR']
        supplied = options[ratio_col] / 100
        for date in options.index:
            if pd.notna(computed.loc[date]) and pd.notna(supplied.loc[date]):
                diff = abs(computed.loc[date] - supplied.loc[date])
                if diff > config.ratio_tolerance:
                    ratio_checks.append(dict(source=source, date=date, calculated=computed.loc[date],
                                             supplied=supplied.loc[date], absolute_difference=diff))
        for name, denominator in [('PCR', call), ('CALLSHARE', put + call)]:
            for date in denominator.index[denominator == 0]:
                zero_checks.append(dict(source=source, signal_name=name, date=date, reason='ZERO_DENOMINATOR'))
        for name, values in bases.items():
            signals[(source, name + '_LEVEL', 0)] = values
            for k in config.change_days:
                signals[(source, f'{name}_CHANGE_{k}D', k)] = values - values.shift(k)
    return signals, pd.DataFrame(ratio_checks, columns=['source','date','calculated','supplied','absolute_difference']), pd.DataFrame(zero_checks, columns=['source','signal_name','date','reason'])

def rolling_percentile(series, window):
    return series.rolling(window, min_periods=window).apply(
        lambda x: 100 * (np.count_nonzero(x < x[-1]) + .5 * np.count_nonzero(x == x[-1])) / len(x), raw=True)

def percentile_bins(rank):
    if ((rank.dropna() < 0) | (rank.dropna() > 100)).any():
        raise ValueError('Percentile outside [0,100]')
    codes = np.select([rank < 5, rank < 20, rank < 40, rank < 60, rank < 80, rank <= 95, rank <= 100], range(7), default=-1)
    return pd.Series(pd.Categorical.from_codes(codes, categories=BINS, ordered=True), index=rank.index)
