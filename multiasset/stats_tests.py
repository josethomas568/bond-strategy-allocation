"""Statistical tests: information coefficients and bootstrap confidence intervals for Sharpe differences."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ic_series(score, fwd):
    """Monthly cross-sectional Spearman correlation between a signal at t and the next month's excess return."""
    out = {}
    for t in score.index:
        if t not in fwd.index:
            continue
        a, b = score.loc[t], fwd.loc[t]
        m = a.notna() & b.notna()
        if m.sum() >= 5 and a[m].nunique() > 1:
            out[t] = spearmanr(a[m], b[m]).statistic
    return pd.Series(out)


def ic_summary(ic):
    ic = ic.dropna()
    return {"mean_ic": ic.mean(), "t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)), "hit_rate": (ic > 0).mean(),
            "months": len(ic)}


def stationary_bootstrap_idx(n, n_out, block, rng):
    """Politis-Romano stationary bootstrap indices (mean block length `block`)."""
    idx = np.empty(n_out, dtype=int)
    idx[0] = rng.integers(n)
    p = 1.0 / block
    jumps = rng.random(n_out) < p
    starts = rng.integers(n, size=n_out)
    for i in range(1, n_out):
        idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
    return idx


def sharpe(x, cash):
    ex = x - cash
    return ex.mean() / x.std() * np.sqrt(12)


def sharpe_diff_ci(a, b, cash, n_boot=5000, block=6, seed=3):
    """Bootstrap the Sharpe ratio difference a - b, resampling months jointly (same indices for both)."""
    a, b, cash = (np.asarray(v, float) for v in (a, b, cash))
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(n_boot)
    for k in range(n_boot):
        i = stationary_bootstrap_idx(n, n, block, rng)
        d[k] = sharpe(a[i], cash[i]) - sharpe(b[i], cash[i])
    point = sharpe(a, cash) - sharpe(b, cash)
    return {"diff": point, "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
            "p_one_sided": float((d <= 0).mean())}
