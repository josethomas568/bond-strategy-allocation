"""
Machine-learning tactical signal: gradient boosting on a panel of (month, asset) rows.

Target    : next-month excess return divided by the asset's trailing 12-month volatility.
Features  : momentum over 1/3/6/12 months, 12-1 momentum, trend (price vs 10-month average),
            5-year return (value-style reversal), volatility (3m, 12m), carry and its 12-month change,
            cross-sectional ranks of momentum and carry, two market-wide variables
            (U.S. stock 12-month excess return, cash yield) and the asset class.
Walk-forward: refit every January on all rows whose label is known, minus a one-month embargo.
Tuning    : purged K-fold cross-validation inside each training set (López de Prado, 2018):
            training rows whose label month falls inside a validation block, or within an embargo
            of 3 months after it, are removed. Score = mean monthly rank correlation (IC) between
            prediction and outcome, the same quantity the overlay uses.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import ASSETS, CASH, UNIVERSE
from .signals import carry_raw, cs_z, excess_return, next_month_excess

GRID = [dict(max_depth=d, learning_rate=lr, max_iter=it, min_samples_leaf=40, l2_regularization=1.0)
        for d, lr, it in itertools.product([2, 3], [0.03, 0.1], [100, 300])]
K_FOLDS, EMBARGO_MONTHS = 5, 3
FIRST_OOS = "2013-01-31"
SEED = 11


def features(D):
    px, daily = D["month_px"], D["daily"]
    f = {}
    for h in (1, 3, 6, 12):
        f[f"mom_{h}m"] = excess_return(D, h)
    f["mom_12_1"] = excess_return(D, 12) - excess_return(D, 1)
    f["trend_10m"] = px[ASSETS] / px[ASSETS].rolling(10).mean() - 1
    f["ret_60m"] = excess_return(D, 60)
    vol_d = daily[ASSETS]
    f["vol_3m"] = vol_d.rolling(63).std().resample("ME").last() * np.sqrt(252)
    f["vol_12m"] = vol_d.rolling(252).std().resample("ME").last() * np.sqrt(252)
    c = carry_raw(D)
    f["carry"] = c
    f["carry_chg_12m"] = c - c.shift(12)
    f["mom_rank"] = excess_return(D, 12).rank(axis=1, pct=True)
    f["carry_rank"] = c.rank(axis=1, pct=True)
    f["carry_z"] = cs_z(c)
    mkt12 = excess_return(D, 12)["SPY"]
    cash_y = D["yield"][CASH]
    rows = []
    for name, df in f.items():
        rows.append(df.stack(future_stack=True).rename(name))
    X = pd.concat(rows, axis=1)
    X.index.names = ["date", "asset"]
    X["mkt_mom_12m"] = X.index.get_level_values("date").map(mkt12)
    X["cash_yield"] = X.index.get_level_values("date").map(cash_y)
    classes = {k: i for i, k in enumerate(sorted({v[0] for v in UNIVERSE.values()}))}
    X["asset_class"] = X.index.get_level_values("asset").map(lambda a: classes[UNIVERSE[a][0]])
    y = (next_month_excess(D) / (f["vol_12m"] / np.sqrt(12))).stack(future_stack=True)
    y.index.names = ["date", "asset"]
    return X, y.reindex(X.index)


def mean_ic(pred, y):
    """Average across months of the cross-sectional Spearman correlation."""
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    ics = [spearmanr(g.p, g.y).statistic for _, g in df.groupby(level="date") if len(g) >= 5]
    return float(np.nanmean(ics)) if ics else np.nan


def purged_kfold(dates, k=K_FOLDS, embargo=EMBARGO_MONTHS):
    """Yield (train_dates, val_dates) over contiguous time blocks. Rows are stamped at the decision month t
    and their label is month t+1, so a training row overlaps validation if t+1 is in the block;
    the embargo also removes training rows in the `embargo` months right after the block."""
    u = np.array(sorted(set(dates)))
    blocks = np.array_split(u, k)
    for b in blocks:
        lo, hi = b[0], b[-1]
        lab = u + pd.offsets.MonthEnd(1)
        overlap = (lab >= lo) & (u <= hi)                       # label or features date inside the block
        after = (u > hi) & (u <= hi + pd.offsets.MonthEnd(embargo))
        keep = ~overlap & ~after
        yield u[keep], b


class _Model:
    """Gradient boosting that ignores features with no data in the training window
    (the 5-year return only exists from 2010 on)."""

    def __init__(self, params):
        self.params = params

    def fit(self, X, y):
        self.cols = [c for c in X.columns if X[c].notna().any()]
        Xc = X[self.cols]
        self.m = HistGradientBoostingRegressor(random_state=SEED,
                                               categorical_features=[Xc.columns.get_loc("asset_class")],
                                               **self.params).fit(Xc, y)
        return self

    def predict(self, X):
        return self.m.predict(X[self.cols])


def fit_model(X, y, params):
    return _Model(params).fit(X, y)


def tune(X, y):
    d = X.index.get_level_values("date")
    scores = []
    for params in GRID:
        ics = []
        for tr, va in purged_kfold(d):
            mtr, mva = d.isin(tr), d.isin(va)
            m = fit_model(X[mtr], y[mtr], params)
            ics.append(mean_ic(pd.Series(m.predict(X[mva]), index=X.index[mva]), y[mva]))
        scores.append(np.nanmean(ics))
    best = int(np.nanargmax(scores))
    return GRID[best], scores[best], scores


def walk_forward(D, X=None, y=None):
    if X is None:
        X, y = features(D)
    ok = y.notna()
    d_all = X.index.get_level_values("date")
    dates = sorted(set(d_all[d_all >= pd.Timestamp(FIRST_OOS)]))
    preds, log = [], []
    current, model = None, None
    for t in dates:
        if t.month == 1 or model is None:
            # labels known at t are those for decision months <= t-1; one more month of embargo
            cutoff = t - pd.offsets.MonthEnd(2)
            tr = ok & (d_all <= cutoff) & (d_all >= pd.Timestamp("2008-04-30"))
            params, cv_ic, _ = tune(X[tr], y[tr])
            model = fit_model(X[tr], y[tr], params)
            log.append({"refit": t, "train_rows": int(tr.sum()), "train_end": cutoff, "cv_ic": cv_ic, **params})
        rows = d_all == t
        preds.append(pd.Series(model.predict(X[rows]), index=X.index[rows]))
    pred = pd.concat(preds)
    return pred, pd.DataFrame(log), X, y


def ml_score(pred):
    p = pred.unstack("asset")[ASSETS]
    return cs_z(p).clip(-1, 1)
