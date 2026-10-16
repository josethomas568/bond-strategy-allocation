"""
Getmansky-Lo-Makarov (2004) de-smoothing of appraisal-based or illiquid returns.

Model: the reported return is a moving average of true economic returns,
    R_obs_t = theta_0 R_t + theta_1 R_{t-1} + ... + theta_k R_{t-k},  theta_j >= 0, sum(theta) = 1.
So reported returns have the true mean but a variance of xi * var(true), with xi = sum(theta_j^2) <= 1,
and they are autocorrelated. Estimation follows GLM: fit an MA(k) to the demeaned reported returns by
maximum likelihood, R_obs_t - mu = eps_t + psi_1 eps_{t-1} + ... (psi_0 = 1), and normalise
theta_j = psi_j / sum(psi). True returns are recovered by inverting the filter:
    R_t = (R_obs_t - theta_1 R_{t-1} - ... - theta_k R_{t-k}) / theta_0,
started from the sample mean.
"""
import warnings

import numpy as np
import pandas as pd


def fit_glm(r, k=2):
    """Returns dict with theta (array), xi, psi, and whether the GLM constraints (theta >= 0) hold."""
    from statsmodels.tsa.arima.model import ARIMA
    x = np.asarray(r, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(x - x.mean(), order=(0, 0, k), trend="n").fit()
    psi = np.r_[1.0, res.params[:k]]
    theta = psi / psi.sum()
    return {"theta": theta, "xi": float((theta ** 2).sum()), "psi": psi, "valid": bool((theta >= 0).all()),
            "aic": float(res.aic), "k": k}


def unsmooth(r, theta):
    r = pd.Series(r, dtype=float)
    k = len(theta) - 1
    mu = r.mean()
    out = np.empty(len(r))
    hist = [mu] * k                              # pre-sample true returns set to the mean
    for i, v in enumerate(r.values):
        lagged = sum(theta[j] * hist[-j] for j in range(1, k + 1)) if k else 0.0
        out[i] = (v - lagged) / theta[0]
        hist.append(out[i])
    return pd.Series(out, index=r.index, name=f"{r.name} (de-smoothed)" if r.name else "de-smoothed")


def autocorr(r, lag=1):
    return pd.Series(r).autocorr(lag)


def beta(r, m):
    common = r.index.intersection(m.index)
    a, b = r.loc[common], m.loc[common]
    return float(np.cov(a, b)[0, 1] / b.var())


def dimson_beta(r, m, lags=2):
    """Sum of slopes on the market's current and lagged returns (Dimson 1979): a check on smoothed betas."""
    common = r.index.intersection(m.index)
    X = pd.concat({f"m{j}": m.shift(j) for j in range(lags + 1)}, axis=1).loc[common].dropna()
    y = r.loc[X.index]
    A = np.c_[np.ones(len(X)), X.values]
    coef = np.linalg.lstsq(A, y.values, rcond=None)[0]
    return float(coef[1:].sum())


def max_drawdown(r):
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def summary(obs, true, market, periods_per_year):
    ann = np.sqrt(periods_per_year)
    return {
        "reported_vol": obs.std() * ann, "true_vol": true.std() * ann, "vol_ratio": true.std() / obs.std(),
        "reported_ac1": autocorr(obs), "true_ac1": autocorr(true),
        "reported_beta": beta(obs, market), "true_beta": beta(true, market),
        "reported_dimson_beta": dimson_beta(obs, market), "true_dimson_beta": dimson_beta(true, market),
        "reported_max_drawdown": max_drawdown(obs), "true_max_drawdown": max_drawdown(true),
        "mean_return_reported": obs.mean() * periods_per_year, "mean_return_true": true.mean() * periods_per_year,
    }
