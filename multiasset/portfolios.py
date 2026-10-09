"""
Strategic portfolio rules. Each takes information available at a month-end and returns
long-only weights over the 10 assets that sum to 1.

Estimation inputs come from `estimate()`: a Ledoit-Wolf shrunk covariance of daily returns
over the trailing 36 months (minimum 12), annualized, and trailing mean returns.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .data import ASSETS

LOOKBACK_MONTHS, MIN_MONTHS = 36, 12
MAX_W_MINVAR = 0.40          # cap per asset for minimum variance (otherwise it piles into one bond fund)
MAX_W_MV = 0.30              # cap per asset for mean-variance
RISK_AVERSION = 5.0          # mean-variance trade-off between expected return and variance
SHRINK_MU = 0.5              # how far trailing means are pulled toward their cross-sectional average


def estimate(daily, t):
    start = t - pd.DateOffset(months=LOOKBACK_MONTHS)
    win = daily.loc[(daily.index > start) & (daily.index <= t), ASSETS].dropna()
    if len(win) < MIN_MONTHS * 19:
        raise ValueError(f"not enough history at {t.date()}")
    cov = pd.DataFrame(LedoitWolf().fit(win.values).covariance_ * 252, index=ASSETS, columns=ASSETS)
    mu = win.mean() * 252
    return cov, mu


def _solve(obj, n, ub, x0=None):
    x0 = np.full(n, 1 / n) if x0 is None else x0
    res = minimize(obj, x0, method="SLSQP", bounds=[(0, ub)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def sixty_forty(cov=None, mu=None):
    w = pd.Series(0.0, index=ASSETS)
    w["SPY"], w["IEF"] = 0.6, 0.4
    return w


def equal_weight(cov=None, mu=None):
    return pd.Series(1 / len(ASSETS), index=ASSETS)


def min_variance(cov, mu=None):
    S = cov.values
    w = _solve(lambda w: w @ S @ w, len(S), MAX_W_MINVAR)
    return pd.Series(w, index=ASSETS)


def risk_parity(cov, mu=None):
    """Equal risk contribution, via the convex formulation of Spinu (2013): minimize
    0.5 y'Sy - sum(log y) over y > 0, then w = y / sum(y)."""
    S = cov.values
    n = len(S)
    res = minimize(lambda y: 0.5 * y @ S @ y - np.log(y).sum(), np.full(n, 1.0),
                   jac=lambda y: S @ y - 1 / y, method="L-BFGS-B", bounds=[(1e-8, None)] * n,
                   options={"maxiter": 5000, "gtol": 1e-12})
    w = res.x / res.x.sum()
    return pd.Series(w, index=ASSETS)


def mean_variance(cov, mu):
    """Long-only, max 30% per asset; expected returns are trailing means shrunk halfway to their average."""
    m = (1 - SHRINK_MU) * mu + SHRINK_MU * mu.mean()
    S, m = cov.values, m.values
    w = _solve(lambda w: -(w @ m) + 0.5 * RISK_AVERSION * (w @ S @ w), len(S), MAX_W_MV)
    return pd.Series(w, index=ASSETS)


def risk_contributions(w, cov):
    S = cov.values
    rc = w.values * (S @ w.values)
    return pd.Series(rc / rc.sum(), index=w.index)


STRATEGIC = {
    "60/40": sixty_forty,
    "Equal weight": equal_weight,
    "Minimum variance": min_variance,
    "Risk parity": risk_parity,
    "Mean-variance": mean_variance,
}
