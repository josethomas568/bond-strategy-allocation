"""
Walk-forward backtest engine.

Weights decided at month-end t (from data up to t) are held through month t+1.
Each month-end the portfolio trades from its drifted weights back to the new target;
cost = sum_i cost_i * |target_i - drifted_i|. A leverage factor L scales exposures;
the borrowed amount (L-1) pays cash (BIL) plus a financing spread.
"""
import numpy as np
import pandas as pd

from .data import ASSETS, CASH, COST

FIN_SPREAD = 0.005 / 12   # borrowing cost over T-bills for levered portfolios, per month


def run(weights, monthly, leverage=None, cost_mult=1.0, name=""):
    """weights: DataFrame (decision month-ends x ASSETS). Returns a DataFrame of monthly results
    stamped at the month in which the return was earned."""
    dates = weights.index
    out = []
    prev_drift = None
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        w = weights.loc[t, ASSETS].values
        L = 1.0 if leverage is None else float(leverage.loc[t])
        e = L * w
        trade = np.abs(e - (prev_drift if prev_drift is not None else np.zeros_like(e)))
        cost = float((trade * COST[ASSETS].values).sum()) * cost_mult
        r = monthly.loc[t1, ASSETS].values
        rc = monthly.loc[t1, CASH]
        gross = float(e @ r) - (L - 1) * (rc + FIN_SPREAD)
        net = gross - cost
        prev_drift = e * (1 + r) / (1 + gross)
        out.append({"date": t1, "gross": gross, "cost": cost, "net": net, "turnover": trade.sum() / 2,
                    "cash": rc, "leverage": L})
    df = pd.DataFrame(out).set_index("date")
    df.attrs["name"] = name
    return df


def stats(r, cash):
    """r, cash: monthly return Series on the same dates."""
    n = len(r)
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    ex = r - cash
    vol = r.std() * np.sqrt(12)
    downside = r[r < 0]
    return {
        "annual_return": wealth.iloc[-1] ** (12 / n) - 1,
        "volatility": vol,
        "sharpe": ex.mean() * 12 / vol,
        "max_drawdown": dd.min(),
        "worst_month": r.min(),
        "worst_12m": (1 + r).rolling(12).apply(np.prod, raw=True).min() - 1,
        "pct_months_up": (r > 0).mean(),
        "calmar": (wealth.iloc[-1] ** (12 / n) - 1) / abs(dd.min()),
        "sortino": ex.mean() * 12 / (np.sqrt((np.minimum(r - cash, 0) ** 2).mean()) * np.sqrt(12)),
        "months": n,
    }
