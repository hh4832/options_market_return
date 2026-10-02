import numpy as np
import pandas as pd

class DataQualityError(ValueError):
    pass

def normalize(frame, name):
    frame = frame.copy()
    # Date labels must represent daily observations, not numeric timestamps.
    if pd.api.types.is_numeric_dtype(frame.index.dtype):
        raise DataQualityError(f'{name}: numeric date index')
    try:
        index = pd.to_datetime(frame.index, errors='raise')
    except (ValueError, TypeError) as exc:
        raise DataQualityError(f'{name}: malformed dates') from exc
    if index.isna().any() or index.tz is not None or (index != index.normalize()).any():
        raise DataQualityError(f'{name}: invalid daily date index')
    if index.has_duplicates:
        raise DataQualityError(f'{name}: duplicate dates')
    if frame.columns.has_duplicates:
        raise DataQualityError(f'{name}: duplicate columns')
    if not index.is_monotonic_increasing:
        raise DataQualityError(f'{name}: unsorted dates')
    frame.index = index
    try:
        frame = frame.apply(pd.to_numeric, errors='raise').astype(float)
    except (TypeError, ValueError) as exc:
        raise DataQualityError(f'{name}: nonnumeric values') from exc
    if np.isinf(frame.to_numpy()).any():
        raise DataQualityError(f'{name}: non-finite values')
    if frame.empty or frame.isna().all().any():
        raise DataQualityError(f'{name}: empty dataset/column')
    return frame

def validate_options(frame):
    frame = normalize(frame, 'options')
    if (frame < 0).any().any():
        raise DataQualityError('Negative option counts/ratios')
    return frame

def validate_prices(frame, provenance, rtol=1e-5):
    from .config import KEYS
    if any(provenance.get(k) != KEYS[k] for k in ('adj_open', 'adj_close', 'raw_open', 'raw_close')):
        raise DataQualityError('Adjusted-price provenance not confirmed')
    frame = normalize(frame, '0050')
    if set(frame.columns) != {'adj_open', 'adj_close', 'raw_open', 'raw_close'}:
        raise DataQualityError('Missing price fields')
    if (frame <= 0).any().any():
        raise DataQualityError('Non-positive price')
    # Pre-listing all-NaN rows may be trimmed; internal dates are retained.
    observed = frame.notna().any(axis=1)
    frame = frame.loc[observed[observed].index[0]:observed[observed].index[-1]]
    fo = frame.adj_open / frame.raw_open
    fc = frame.adj_close / frame.raw_close
    comparable = fo.notna() & fc.notna()
    if not np.allclose(fo[comparable], fc[comparable], rtol=rtol, atol=1e-10):
        raise DataQualityError('Open/close adjustment factors disagree')
    if not comparable.any():
        raise DataQualityError('No comparable adjusted/raw prices')
    return frame
