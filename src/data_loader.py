import os
import pandas as pd
from .config import KEYS
from .validation import DataQualityError

class CredentialUnavailable(RuntimeError):
    status = 'DATA_VALIDATION_BLOCKED_BY_CREDENTIAL'

OPTION_COLUMNS = {
    'VOLUME': ('賣權成交量', '買權成交量', '買賣權成交量比率%'),
    'OPEN_INTEREST': ('賣權未平倉量', '買權未平倉量', '買賣權未平倉量比率%'),
}

def _search_diagnostics(data, key):
    """Return search results for metadata only; never gate required dataset loading."""
    try:
        matches = data.search(key)
    except Exception:
        return []
    if isinstance(matches, pd.Series):
        return matches.astype(str).tolist()
    try:
        return [str(value) for value in matches]
    except TypeError:
        return [str(matches)]

def _get_required_dataset(data, key):
    """Load a required FinLab dataset without any fallback."""
    try:
        return data.get(key)
    except Exception as exc:
        raise DataQualityError(f'Failed to load required dataset {key}; no raw fallback') from exc

def load_finlab():
    import finlab
    from finlab import data
    token = os.environ.get('FINLAB_API_TOKEN')
    if not token:
        raise CredentialUnavailable('DATA_VALIDATION_BLOCKED_BY_CREDENTIAL: set FINLAB_API_TOKEN through Colab Secret/environment')
    finlab.login(token)

    # data.search() is diagnostic metadata only. In particular, a pandas Series
    # checks membership against its index, so it must not be used as an
    # availability gate for dataset keys.
    available = {
        key: _search_diagnostics(data, key)
        for key in (KEYS['adj_open'], KEYS['adj_close'])
    }

    options = pd.DataFrame(_get_required_dataset(data, KEYS['options']))
    required = [col for group in OPTION_COLUMNS.values() for col in group]
    if not set(required).issubset(options.columns):
        raise DataQualityError(f'Option schema changed; expected {required}; got {list(options.columns)}')
    options = options[required]

    series = {}
    for name in ('adj_open', 'adj_close', 'raw_open', 'raw_close'):
        key = KEYS[name]
        dataset = _get_required_dataset(data, key)
        if '0050' not in dataset.columns:
            raise DataQualityError(f'0050 missing in {key}')
        series[name] = dataset['0050']

    # Union preserves missing dates; no dropna trading-calendar compression.
    prices = pd.DataFrame(series)
    return options, prices, dict(KEYS), available
