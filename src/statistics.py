import numpy as np
from scipy.stats import norm

def descriptive(values):
    x = np.asarray(values.dropna(), dtype=float)
    n = len(x)
    return dict(n=n, win_count=int((x>0).sum()), loss_count=int((x<0).sum()),
                zero_count=int((x==0).sum()), win_rate=float((x>0).mean()) if n else np.nan,
                mean_return=float(x.mean()) if n else np.nan,
                median_return=float(np.median(x)) if n else np.nan,
                std_return=float(x.std(ddof=1)) if n>1 else np.nan,
                min_return=float(x.min()) if n else np.nan,
                max_return=float(x.max()) if n else np.nan)

def hac_contrast(y, left, right=None, lag=1, min_sample=30):
    """Ratio-of-means influence contrast on full trading-calendar grid.

    Missing/non-selected observations have zero influence, never zero return.
    Supports conditional-zero, conditional-unconditional and disjoint-bin contrasts.
    """
    y = np.asarray(y, dtype=float)
    left = np.asarray(left, dtype=bool) & np.isfinite(y)
    right = None if right is None else np.asarray(right, dtype=bool) & np.isfinite(y)
    nl, nr = int(left.sum()), None if right is None else int(right.sum())
    out = dict(hac_t=np.nan, p_value=np.nan, standard_error=np.nan, effect=np.nan,
               lag=lag, left_n=nl, right_n=nr, status='INSUFFICIENT_OBSERVATIONS')
    if nl == 0 or (right is not None and nr == 0):
        return out
    ml = y[left].mean()
    mr = 0 if right is None else y[right].mean()
    out['effect'] = ml-mr
    if nl < min_sample or (nr is not None and nr < min_sample):
        return out
    active = left if right is None else left | right
    positions = np.flatnonzero(active)
    lo, hi = positions[0], positions[-1]+1
    y, left = y[lo:hi], left[lo:hi]
    right = None if right is None else right[lo:hi]
    T = len(y)
    if T <= lag+1:
        return out
    influence = np.zeros(T)
    influence[left] += (y[left]-ml)/nl
    if right is not None:
        influence[right] -= (y[right]-mr)/nr
    variance = float(influence @ influence)
    for k in range(1, min(lag,T-1)+1):
        variance += 2*(1-k/(lag+1))*float(influence[k:] @ influence[:-k])
    variance *= T/(T-1)
    if not np.isfinite(variance) or variance <= 1e-24:
        out['status'] = 'DEGENERATE_VARIANCE'
        return out
    se = np.sqrt(variance)
    t = (ml-mr)/se
    out.update(hac_t=float(t), p_value=float(2*norm.sf(abs(t))), standard_error=float(se), status='OK')
    return out
