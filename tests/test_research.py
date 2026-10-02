import numpy as np
import pandas as pd
import pytest
from src.config import Config, BINS, KEYS
from src.data_loader import OPTION_COLUMNS, load_finlab, CredentialUnavailable
from src.signals import build_signals, rolling_percentile, percentile_bins
from src.outcomes import compute_outcomes
from src.validation import validate_prices, normalize, DataQualityError
from src.statistics import hac_contrast
from src.fdr import bh, apply_fdr
from src.annual_analysis import annual_cells, annual_summary
from src.robustness import neighbor_diagnostics
from src.pipeline import archive_to_drive

def options():
    idx=pd.bdate_range('2020-01-01',periods=12)
    f=pd.DataFrame(index=idx)
    for _,(p,c,q) in OPTION_COLUMNS.items():
        f[p]=np.arange(12.)+2; f[c]=np.arange(12.)+1; f[q]=f[p]/f[c]*100
    return f

@pytest.mark.parametrize('source', OPTION_COLUMNS)
def test_signal_formulas(source):
    f=options(); signals,checks,_=build_signals(f,Config())
    assert len(signals)==32 and checks.empty
    p,c,_=OPTION_COLUMNS[source]
    bases={'PUT':f[p],'CALL':f[c],'PCR':f[p]/f[c],'CALLSHARE':f[c]/(f[p]+f[c])}
    for name,x in bases.items():
        pd.testing.assert_series_equal(signals[source,name+'_LEVEL',0],x)
        for k in (1,3,5):
            pd.testing.assert_series_equal(signals[source,f'{name}_CHANGE_{k}D',k],x-x.shift(k))

def test_zero_denominator():
    f=options();p,c,_=OPTION_COLUMNS['VOLUME'];f.loc[f.index[0],[p,c]]=0
    s,_,z=build_signals(f,Config())
    assert pd.isna(s['VOLUME','PCR_LEVEL',0].iloc[0])
    assert pd.isna(s['VOLUME','CALLSHARE_LEVEL',0].iloc[0]) and len(z)==2

def test_percentile_no_lookahead():
    x=pd.Series([1.,2.,3.,4.,5.,6.]);a=rolling_percentile(x,3)
    assert a.iloc[:2].isna().all() and a.iloc[2]==pytest.approx(100*2.5/3)
    x.iloc[4:]=999
    pd.testing.assert_series_equal(a.iloc[:4],rolling_percentile(x,3).iloc[:4])
    x.iloc[2]=np.nan
    assert rolling_percentile(x,3).iloc[2:5].isna().all()

def test_boundaries():
    x=pd.Series([0,4.99,5,20,40,60,80,95,95.01,100,np.nan])
    assert list(percentile_bins(x).iloc[:10])==[BINS[i] for i in [0,0,1,2,3,4,5,5,6,6]]
    assert pd.isna(percentile_bins(x).iloc[-1])

def prices():
    idx=pd.bdate_range('2020-01-02',periods=30)
    f=pd.DataFrame({'raw_open':np.arange(30)+100.,'raw_close':np.arange(30)+101.},index=idx)
    f['adj_open']=f.raw_open*2;f['adj_close']=f.raw_close*2
    return f

@pytest.mark.parametrize('h',[1,2,3,5,10,20])
def test_outcome_mapping(h):
    f=prices();y,a=compute_outcomes(f,f.index,(h,))
    assert y[h].iloc[0]==pytest.approx(f.adj_close.iloc[h]/f.adj_open.iloc[1]-1)
    assert a.entry_date.iloc[0]==f.index[1] and a.outcome_date.iloc[0]==f.index[h]
    assert y[h].iloc[-h:].isna().all()

def test_adjusted_validation_missing():
    f=prices();validate_prices(f,KEYS)
    bad=f.copy();bad.iloc[2,bad.columns.get_loc('adj_open')]*=1.1
    with pytest.raises(DataQualityError):validate_prices(bad,KEYS)
    f.iloc[2,f.columns.get_loc('adj_close')]=np.nan
    y,_=compute_outcomes(f,f.index,(3,));assert pd.isna(y[3].iloc[0])
    with pytest.raises(DataQualityError):normalize(pd.concat([f,f.iloc[:1]]),'duplicate')
    with pytest.raises(DataQualityError):normalize(pd.DataFrame({'x':[1]},index=['bad']),'malformed')

@pytest.mark.parametrize('lag',[1,2,3,5,10,20])
def test_hac_lags(lag):
    from statsmodels.stats.sandwich_covariance import cov_hac
    import statsmodels.api as sm
    y=np.random.default_rng(42).normal(size=100)
    a=hac_contrast(y,np.ones(100,bool),lag=lag,min_sample=2)
    expected=np.sqrt(cov_hac(sm.OLS(y,np.ones((100,1))).fit(),nlags=lag)[0,0])
    assert a['lag']==lag and a['standard_error']==pytest.approx(expected)

def test_bh_grouping():
    np.testing.assert_allclose(bh([.01,.04,.03]),[.03,.04,.04])
    f=pd.DataFrame({'source':['VOLUME']*2+['OPEN_INTEREST']*2,'signal_name':['PCR_LEVEL']*4,'zero_p_value':[.01,.04,.5,.9],'unconditional_p_value':[.1,.2,.3,.4]})
    a=apply_fdr(apply_fdr(f),family=False)
    np.testing.assert_allclose(a.zero_p_value_fdr_family,[.02,.04,.9,.9])
    np.testing.assert_allclose(a.zero_p_value_fdr_global,bh(f.zero_p_value))
    np.testing.assert_allclose(a.unconditional_p_value_fdr_global,bh(f.unconditional_p_value))

def test_annual_grouping():
    idx=pd.to_datetime(['2019-12-31','2020-01-02','2020-01-03']);y=pd.Series([.1,.2,.3],index=idx)
    b=pd.Series(pd.Categorical([BINS[0]]*3,categories=BINS),index=idx)
    identity=dict(source='VOLUME',signal_name='PCR_LEVEL',change_days=0,rolling_window=60,horizon=1)
    a=pd.DataFrame(annual_cells(y,b,identity,Config(min_sample=2,min_years=1)))
    assert a[a.percentile_bin==BINS[0]].n.tolist()==[1,2]
    assert not annual_summary(a,Config(min_sample=2,min_years=1)).empty

def test_neighbor_island():
    rows=[]
    for w,e,p in [(60,.01,False),(120,.02,True),(252,-.01,False)]:
        rows.append(dict(source='VOLUME',signal_name='PCR_LEVEL',change_days=0,rolling_window=w,percentile_bin=BINS[0],horizon=1,n=100,mean_excess_vs_unconditional=e,unconditional_significant_raw_05=p,unconditional_significant_fdr_family_05=p))
    n=neighbor_diagnostics(pd.DataFrame(rows),Config(rolling_windows=(60,120,252),horizons=(1,)))
    assert n.iloc[1].neighbor_direction_consistency==.5 and n.iloc[1].parameter_island

def test_credential_block(monkeypatch):
    monkeypatch.delenv('FINLAB_API_TOKEN',raising=False)
    with pytest.raises(CredentialUnavailable) as e:load_finlab()
    assert e.value.status=='DATA_VALIDATION_BLOCKED_BY_CREDENTIAL'

def test_archive_no_overwrite(tmp_path):
    run=tmp_path/'run';run.mkdir();(run/'result').write_text('ok')
    drive=tmp_path/'drive';drive.mkdir();archive_to_drive(run,drive)
    with pytest.raises(FileExistsError):archive_to_drive(run,drive)

def test_calendar_missing_consumes_change_slot():
    f=options();idx=f.index;f=f.drop(idx[3]).reindex(idx)
    s,_,_=build_signals(f,Config())
    assert pd.isna(s['VOLUME','PUT_CHANGE_1D',1].loc[idx[4]])

def test_hac_overlap_covariance():
    y=np.random.default_rng(10).normal(size=100)
    mask=np.arange(100)%2==0
    a=hac_contrast(y,mask,np.ones(100,bool),lag=3,min_sample=2)
    assert a['effect']==pytest.approx(y[mask].mean()-y.mean())
    identical=hac_contrast(y,mask,mask,lag=3,min_sample=2)
    assert identical['effect']==0 and identical['status']=='DEGENERATE_VARIANCE'
