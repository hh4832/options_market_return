import pandas as pd

def data_reports(options, prices):
    coverage=[]; missing=[]
    for name, frame in [('options',options),('0050',prices)]:
        for col in frame:
            s=frame[col]; good=s.dropna()
            coverage.append(dict(dataset=name,column=col,n=len(s),valid_count=len(good),missing_count=s.isna().sum(),
                                 first_valid_date=good.index.min(),last_valid_date=good.index.max(),
                                 duplicate_dates=frame.index.duplicated().sum(),malformed_dates=0,non_finite_values=0))
            missing.extend(dict(dataset=name,column=col,date=d,reason='MISSING',status='RECORDED_NOT_FILLED') for d in s.index[s.isna()])
    # Explicitly record option-only / price-only dates in shared listing range.
    start,end=prices.index.min(),prices.index.max()
    for d in options.index[(options.index>=start)&(options.index<=end)].difference(prices.index):
        missing.append(dict(dataset='alignment',column='0050',date=d,reason='OPTION_DATE_NOT_PRICE_CALENDAR',status='EXCLUDED'))
    for d in prices.index.difference(options.index):
        missing.append(dict(dataset='alignment',column='options',date=d,reason='PRICE_DATE_NOT_OPTION_CALENDAR',status='NO_SIGNAL'))
    return pd.DataFrame(coverage),pd.DataFrame(missing,columns=['dataset','column','date','reason','status'])

def write_summary(path, tables):
    lines=['# Diagnostics','所有數字由本次 run 計算；missing 未填補。']
    for name,table in tables.items():
        lines.append(f'- {name}: {len(table)} rows')
    hac=tables['hac_diagnostics']
    lines.append(f'- HAC non-OK tests: {int((hac.status != "OK").sum())}')
    fdr=tables['fdr_diagnostics']
    for _,r in fdr[fdr.scope=='global'].iterrows():
        lines.append(f'- {r.universe}/{r.test}: planned={r.planned_tests}, valid={r.valid_tests}, Global BH discoveries={r.discoveries}')
    (path/'diagnostics_summary.md').write_text('\n'.join(lines),encoding='utf-8')
