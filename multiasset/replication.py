"""
Hedge fund replication with the multi-asset ETF universe (10 ETFs + T-bills).

Two estimators of a strategy index's exposures, both walk-forward:

* Rolling returns-based style analysis (Sharpe 1992). Each month, regress the index's returns over
  the trailing 36 months on the ETF and T-bill returns with weights >= 0 that sum to 1. The fitted
  weights are the replicating portfolio.
* Kalman filter. Exposures follow a random walk, beta_t = beta_{t-1} + eta_t, and the index's excess
  return is r_t - rf_t = x_t' beta_t + e_t, with x_t the ETFs' excess returns. Observation and state
  noise are estimated by maximum likelihood on the first 36 months only and then fixed. Two versions:
    - unconstrained: exposures may be negative or sum above 1 (a "paper" replication; it needs shorting)
    - constrained: after every update the estimate is projected onto {beta >= 0, sum(beta) <= 1}, the
      remainder held in T-bills, so the portfolio is long-only and fully funded like the regression's.

Timing: EDHEC publishes a month's index return during the following month, so the weights held in
month t+1 use index data only through month t-1 (a one-month reporting lag). ETF data is cut at the
same point. Out-of-sample evaluation compares the replica's return in t+1 to the index's.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import ASSETS, CASH

FACTORS = ASSETS + [CASH]
WINDOW = 36          # months of history for each regression and for the Kalman parameter estimates
LAG = 1              # months between an index return and the decision that can use it


# ------------------------------------------------------------------------------------------- RBSA
def style_weights(y, X):
    """min ||y - X w||^2 subject to w >= 0, sum(w) = 1. X columns are FACTORS (cash included)."""
    Xv, yv = np.asarray(X, float), np.asarray(y, float)
    n = Xv.shape[1]
    G, c = Xv.T @ Xv, Xv.T @ yv
    res = minimize(lambda w: w @ G @ w - 2 * c @ w, np.full(n, 1 / n), jac=lambda w: 2 * (G @ w - c),
                   method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   options={"maxiter": 1000, "ftol": 1e-14})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def rolling_style(y, F, dates, window=WINDOW, lag=LAG):
    """Weights decided at each date in `dates`, from the `window` months ending `lag` months earlier."""
    W, r2 = [], []
    for t in dates:
        end = t - pd.offsets.MonthEnd(lag)
        idx = y.loc[:end].index[-window:]
        if len(idx) < window:
            raise ValueError(f"not enough history before {t.date()}")
        w = style_weights(y.loc[idx], F.loc[idx, FACTORS])
        fit = F.loc[idx, FACTORS].values @ w
        r2.append(1 - ((y.loc[idx] - fit) ** 2).sum() / ((y.loc[idx] - y.loc[idx].mean()) ** 2).sum())
        W.append(pd.Series(w, index=FACTORS, name=t))
    return pd.DataFrame(W), pd.Series(r2, index=dates, name="in_sample_r2")


# ------------------------------------------------------------------------------------------- Kalman
def project_long_only(b):
    """Euclidean projection onto {b >= 0, sum(b) <= 1}."""
    p = np.maximum(b, 0)
    if p.sum() <= 1:
        return p
    u = np.sort(b)[::-1]                       # projection onto the simplex (Duchi et al. 2008)
    css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(u) + 1) > (css - 1))[0][-1]
    tau = (css[k] - 1) / (k + 1)
    return np.maximum(b - tau, 0)


def kalman_path(y, X, R, Q, b0, P0, constrained=False):
    """Run the filter; returns filtered exposures (T x k) and the log-likelihood of the innovations."""
    T, k = X.shape
    b, P = b0.copy(), P0.copy()
    out, ll = np.empty((T, k)), 0.0
    Qm = np.eye(k) * Q
    for t in range(T):
        P = P + Qm                                   # predict (random walk)
        x = X[t]
        S = x @ P @ x + R
        e = y[t] - x @ b
        K = P @ x / S
        b = b + K * e
        P = P - np.outer(K, x @ P)
        P = (P + P.T) / 2
        if constrained:
            b = project_long_only(b)
        ll += -0.5 * (np.log(2 * np.pi * S) + e * e / S)
        out[t] = b
    return out, ll


def fit_kalman_params(y, X, b0):
    """MLE of observation variance R and state variance Q on a training window (diffuse-ish start)."""
    k = X.shape[1]
    P0 = np.eye(k) * 0.25

    def nll(theta):
        R, Q = np.exp(theta)
        return -kalman_path(y, X, R, Q, b0, P0)[1]
    r0 = np.var(y - X @ b0)
    best = min((minimize(nll, [np.log(r0), np.log(q0)], method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400}) for q0 in (1e-6, 1e-4, 1e-3)),
               key=lambda r: r.fun)
    R, Q = np.exp(best.x)
    return float(R), float(Q)


def kalman_weights(y, F, dates, window=WINDOW, lag=LAG):
    """Walk-forward Kalman exposures. Parameters come from the first `window` months only; the filter
    then runs forward month by month. Returns (unconstrained weights, constrained weights, params)."""
    ex_y = (y - F[CASH]).dropna()
    ex_X = F[ASSETS].sub(F[CASH], axis=0).loc[ex_y.index]
    train = ex_y.index[:window]
    b0 = style_weights(y.loc[train], F.loc[train, FACTORS])[:-1]          # ETF part of the regression fit
    R, Q = fit_kalman_params(ex_y.loc[train].values, ex_X.loc[train].values, b0)
    P0 = np.eye(len(ASSETS)) * 0.25
    paths = {}
    for name, cons in (("unconstrained", False), ("constrained", True)):
        B, _ = kalman_path(ex_y.values, ex_X.values, R, Q, b0, P0, constrained=cons)
        paths[name] = pd.DataFrame(B, index=ex_y.index, columns=ASSETS)
    out = {}
    for name, B in paths.items():
        rows = []
        for t in dates:
            b = B.loc[:t - pd.offsets.MonthEnd(lag)].iloc[-1]
            w = pd.Series(0.0, index=FACTORS, name=t)
            w[ASSETS] = b.values
            w[CASH] = 1 - b.sum()
            rows.append(w)
        out[name] = pd.DataFrame(rows)
    return out["unconstrained"], out["constrained"], {"R": R, "Q": Q, "train_start": train[0], "train_end": train[-1]}


# ------------------------------------------------------------------------------------------- evaluation
def replica_returns(W, F):
    """Gross replica return earned in the month after each decision date."""
    nxt = {t: F.index[F.index.get_loc(t) + 1] for t in W.index if F.index.get_loc(t) + 1 < len(F.index)}
    return pd.Series({nxt[t]: float(W.loc[t, FACTORS] @ F.loc[nxt[t], FACTORS]) for t in nxt})


def tracking(rep, idx, cash):
    common = rep.index.intersection(idx.index)
    a, b, c = rep.loc[common], idx.loc[common], cash.loc[common]
    d = a - b
    n = len(common)
    ann = lambda r: (1 + r).prod() ** (12 / n) - 1
    beta = np.cov(b, a)[0, 1] / a.var()
    return {"months": n, "start": common[0].date(), "end": common[-1].date(),
            "tracking_error": d.std() * np.sqrt(12), "correlation": a.corr(b),
            "oos_r2": 1 - (d ** 2).sum() / ((b - b.mean()) ** 2).sum(),
            "index_return": ann(b), "replica_return": ann(a), "return_gap": ann(a) - ann(b),
            "index_vol": b.std() * np.sqrt(12), "replica_vol": a.std() * np.sqrt(12),
            "index_sharpe": (b - c).mean() / b.std() * np.sqrt(12),
            "replica_sharpe": (a - c).mean() / a.std() * np.sqrt(12),
            "beta_index_on_replica": beta}
