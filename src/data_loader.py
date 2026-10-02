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

def load_finlab():
    import finlab
    from finlab import data
    token = os.environ.get('FINLAB_API_TOKEN')
    if not token:
        raise CredentialUnavailable('DATA_VALIDATION_BLOCKED_BY_CREDENTIAL: set FINLAB_API_TOKEN through Colab Secret/environment')
    finlab.login(token)
    available = {}
    for key in (KEYS['adj_open'], KEYS['adj_close']):
        matches = data.search(key)
        available[key] = list(matches)
        if key not in matches:
            raise DataQualityError(f'Adjusted price key unavailable: {key}; no raw fallback')
    options = pd.DataFrame(data.get(KEYS['options']))
    required = [col for group in OPTION_COLUMNS.values() for col in group]
    if not set(required).issubset(options.columns):
        raise DataQualityError(f'Option schema changed; expected {required}; got {list(options.columns)}')
    options = options[required]
    series = {}
    for name in ('adj_open', 'adj_close', 'raw_open', 'raw_close'):
        dataset = data.get(KEYS[name])
        if '0050' not in dataset.columns:
            raise DataQualityError(f'0050 missing in {KEYS[name]}')
        series[name] = dataset['0050']
    # Union preserves missing dates; no dropna trading-calendar compression.
    prices = pd.DataFrame(series)
    return options, prices, dict(KEYS), available
