from dataclasses import asdict, dataclass

BINS = ('<5', '5–20', '20–40', '40–60', '60–80', '80–95', '>95')
KEYS = dict(options='tw_option_put_call_ratio', adj_open='etl:adj_open',
            adj_close='etl:adj_close', raw_open='price:開盤價', raw_close='price:收盤價')
PRICE_DOCUMENTATION = 'https://finlab.finance/blog/custom-backtest-price-series'
LIMITATIONS = '''反對者觀點：Put/Call OI 不等同 bearish/bullish，包含 protective put、short put、covered call、spread、market-maker hedging、expiry effects 與結構變化。PCR 與 CallShare 是高度相關 transformation，不能算兩份独立證據。參數與 horizons 高度相關；BH-FDR 不能消除 data snooping。七個 bin means 的 Spearman 是描述性單調性檢查，不是獨立樣本的正式市場推論，也不是線性證據。未做樣本外驗證、成本、滑價、流動性與交易策略測試，無法判定可交易性或因果。'''.replace('独', '獨')

@dataclass(frozen=True)
class Config:
    rolling_windows: tuple = (60, 120, 252, 504, 756)
    horizons: tuple = (1, 2, 3, 5, 10, 20)
    change_days: tuple = (1, 3, 5)
    timezone: str = 'Asia/Taipei'
    drive_output_root: str = '/content/drive/MyDrive/00Quant_Research/options_market_return'
    min_sample: int = 30
    min_years: int = 5
    annual_direction_threshold: float = 0.7
    concentration_threshold: float = 0.5
    neighbor_threshold: float = 0.6
    alpha: float = 0.05
    ratio_tolerance: float = 0.000051  # provider rounds percent to .01
    adjustment_rtol: float = 1e-5
    plots: bool = True

    def __post_init__(self):
        for values in (self.rolling_windows, self.horizons, self.change_days):
            if not values or tuple(sorted(set(values))) != values or min(values) < 1:
                raise ValueError('Parameter sequences must be unique, ascending positive integers')
        if self.min_sample < 2 or self.min_years < 1:
            raise ValueError('Invalid sample thresholds')

    def metadata(self):
        return dict(**asdict(self), keys=KEYS, adjusted_price_documentation=PRICE_DOCUMENTATION,
                    bins=['[0,5)', '[5,20)', '[20,40)', '[40,60)', '[60,80)', '[80,95]', '(95,100]'],
                    percentile='100*(count(past < current)+0.5*count(past == current))/window; includes t; full window only',
                    changes='x[t]-x[t-k] for all signals; raw counts in contracts, PCR in ratio units, CallShare in fraction units',
                    zero_handling='zero denominator => NaN, never fill',
                    outcome='adjusted close[t+h]/adjusted open[t+1]-1; full price trading calendar',
                    benchmark='same source/signal/window/horizon eligible dates',
                    hac='two-sided normal Wald; Bartlett/Newey-West lag=h; full-calendar influence function, missing dates zero influence, never zero returns; finite sample T/(T-1)',
                    fdr='BH; family=source × signal definition; zero/excess separate; extremes separate contrast universes',
                    annual_contribution='abs(n_year*annual_excess)/sum(abs(n_year*annual_excess)); annual eligible benchmark',
                    evidence='excess-comparison inference; Global significant with warnings => GLOBAL_WITH_WARNINGS, not Level A')
