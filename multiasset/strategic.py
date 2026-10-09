"""Walk-forward strategic portfolios, including risk parity levered to the 60/40's expected risk."""
import numpy as np
import pandas as pd

from . import backtest
from .data import ASSETS, FIRST_DECISION
from .portfolios import STRATEGIC, estimate

MAX_LEVERAGE = 3.0


def decision_dates(D):
    m = D["monthly"].index
    complete = m[m <= D["last_date"]]          # drop a partial current month
    return complete[complete >= pd.Timestamp(FIRST_DECISION)]


def build(D):
    dates = decision_dates(D)
    W = {k: [] for k in STRATEGIC}
    lev, covs = [], {}
    for t in dates:
        cov, mu = estimate(D["daily"], t)
        covs[t] = cov
        for k, fn in STRATEGIC.items():
            W[k].append(fn(cov, mu).rename(t))
        rp = W["Risk parity"][-1].values
        s60 = W["60/40"][-1].values
        vol_rp, vol_60 = np.sqrt(rp @ cov.values @ rp), np.sqrt(s60 @ cov.values @ s60)
        lev.append(min(vol_60 / vol_rp, MAX_LEVERAGE))
    W = {k: pd.DataFrame(v) for k, v in W.items()}
    W["Risk parity (levered)"] = W["Risk parity"].copy()
    leverage = pd.Series(lev, index=dates)
    return W, leverage, covs


def run_all(D, W, leverage, cost_mult=1.0):
    res = {}
    for k, w in W.items():
        res[k] = backtest.run(w, D["monthly"], leverage=leverage if k == "Risk parity (levered)" else None,
                              cost_mult=cost_mult, name=k)
    return res


def summary(res):
    rows = []
    for k, r in res.items():
        s = backtest.stats(r["net"], r["cash"])
        s.update({"strategy": k, "turnover_per_year": r["turnover"].mean() * 12,
                  "cost_per_year": r["cost"].mean() * 12, "avg_leverage": r["leverage"].mean()})
        rows.append(s)
    return pd.DataFrame(rows).set_index("strategy")
