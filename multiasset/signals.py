"""
Tactical signals, all stamped at month-end t and built only from data up to t.

- Time-series momentum (TSMOM): sign of each asset's trailing 12-month return in excess of cash.
- Carry: trailing 12-month distribution yield minus the cash (T-bill ETF) yield, ranked across assets.
  For gold and commodity ETFs this is a weak proxy: their distributions are T-bill interest at best,
  so their carry is roughly zero minus cash (they do not observe the futures roll yield).
- Combined: the average of the two scores.

Every signal becomes a score s in [-1, 1] per asset, and the overlay tilts a strategic base portfolio
by the same rule: weight_i proportional to base_i * (1 + TILT * s_i), fully invested.
"""
import numpy as np
import pandas as pd

from .data import ASSETS, CASH

TILT = 0.5


def excess_return(D, months):
    px = D["month_px"]
    r = px / px.shift(months) - 1
    return r[ASSETS].sub(r[CASH], axis=0)


def cs_z(x):
    """Cross-sectional z-score each month (across the 10 assets)."""
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def tsmom(D):
    return np.sign(excess_return(D, 12))


def carry_raw(D):
    y = D["yield"]
    return y[ASSETS].sub(y[CASH], axis=0)


def carry(D):
    return cs_z(carry_raw(D)).clip(-1, 1)


def combined(D):
    return (tsmom(D) + carry(D)) / 2


def tilt(base, score):
    """base: weights (dates x assets); score: same shape in [-1, 1]. Fully invested, long only."""
    s = score.reindex(index=base.index, columns=base.columns).fillna(0.0)
    w = base * (1 + TILT * s)
    return w.div(w.sum(axis=1), axis=0)


def next_month_excess(D):
    """Label for month-end t: the asset's return over month t+1 minus cash. Used only for evaluation and ML training."""
    m = D["monthly"]
    return m[ASSETS].sub(m[CASH], axis=0).shift(-1)
