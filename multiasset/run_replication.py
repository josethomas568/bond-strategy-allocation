"""
Hedge fund and private-market replication with the multi-asset ETF universe.

    python -m multiasset.run_replication        # -> results/replication/  (about 1 minute)

1. Replicate EDHEC hedge fund indices with the 10 ETFs + T-bills: rolling style analysis and a
   Kalman filter (unconstrained and long-only). Out-of-sample tracking error and correlation.
2. De-smooth a private-market index (Getmansky-Lo-Makarov) to estimate its true volatility and
   equity beta, then replicate the de-smoothed returns with the same ETFs.
3. Backtest the long-only replication portfolios through the existing engine, with trading costs
   at 1x and 2x.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

from . import backtest, data, desmooth, hedgefunds, replication as rp

OUT = "results/replication"
PRIVATE_K = 2            # GLM smoothing lags (sensitivity for 1-3 is reported)
PRIVATE_MIN_HISTORY = 60  # periods before the first walk-forward de-smoothing estimate (monthly stand-in)


def factor_returns(D):
    F = D["monthly"][rp.FACTORS].dropna()
    return F[F.index <= D["last_date"].to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(
        0 if D["last_date"].is_month_end else 1)]


def with_terminal_row(W, months):
    """The engine earns month t+1's return on weights decided at t and stops one row before the end;
    append the next month (weights unused) so the last decision is also held for a month."""
    nxt = months[months > W.index[-1]]
    if len(nxt) == 0:
        return W
    return pd.concat([W, W.iloc[[-1]].set_axis([nxt[0]])])


def replicate_hf(F, H):
    rows, weights, r2s, params, reps = [], {}, {}, {}, {}
    for s in H.columns:
        y = H[s].dropna()
        common = y.index.intersection(F.index)
        if len(common) < rp.WINDOW + 24:
            continue
        y = y.loc[common]
        dates = common[rp.WINDOW + rp.LAG - 1: -1]
        Wr, r2 = rp.rolling_style(y, F, dates)
        Wu, Wc, prm = rp.kalman_weights(y, F, dates)
        for name, W in [("Rolling style analysis", Wr), ("Kalman (unconstrained)", Wu), ("Kalman (long-only)", Wc)]:
            rep = rp.replica_returns(W, F)
            t = rp.tracking(rep, y, F[data.CASH])
            t.update({"strategy": s, "method": name, "avg_in_sample_r2": r2.mean() if name.startswith("Rolling") else np.nan,
                      "avg_gross_exposure": W[data.ASSETS].abs().sum(axis=1).mean(),
                      "avg_cash_weight": W[data.CASH].mean()})
            rows.append(t)
            weights[(s, name)] = W
            reps[(s, name)] = rep
        r2s[s], params[s] = r2, prm
    return pd.DataFrame(rows), weights, r2s, params, reps


def backtest_replicas(D, F, H, weights, targets):
    rows, nets = [], {}
    for s in targets:
        for name in ("Rolling style analysis", "Kalman (long-only)"):
            W = with_terminal_row(weights[(s, name)], F.index)
            for mult in (1.0, 2.0):
                r = backtest.run(W, D["monthly"], assets=rp.FACTORS, cost_mult=mult, name=f"{s} | {name}")
                y = H[s].reindex(r.index).dropna()
                r = r.loc[y.index]
                st = backtest.stats(r["net"], r["cash"])
                tr = rp.tracking(r["net"], y, r["cash"])
                rows.append({"strategy": s, "method": name, "cost_multiple": mult,
                             "net_return": st["annual_return"], "volatility": st["volatility"],
                             "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"],
                             "turnover_per_year": r["turnover"].mean() * 12, "cost_per_year": r["cost"].mean() * 12,
                             "index_return": tr["index_return"], "index_sharpe": tr["index_sharpe"],
                             "index_max_drawdown": desmooth.max_drawdown(y),
                             "net_tracking_error": tr["tracking_error"], "net_correlation": tr["correlation"],
                             "start": r.index[0].date(), "end": r.index[-1].date(), "months": len(r)})
                if mult == 1.0:
                    nets[(s, name)] = r["net"]
    return pd.DataFrame(rows), nets


def glm_table(H):
    rows = []
    for s in H.columns:
        x = H[s].dropna()
        x = x[x != 0] if (x == 0).mean() > 0.02 else x          # Short Selling has zero-filled months at the end
        f = desmooth.fit_glm(x, 2)
        tr = desmooth.unsmooth(x, f["theta"])
        rows.append({"strategy": s, "theta0": f["theta"][0], "theta1": f["theta"][1], "theta2": f["theta"][2],
                     "smoothing_index_xi": f["xi"], "valid": f["valid"], "reported_ac1": desmooth.autocorr(x),
                     "true_ac1": desmooth.autocorr(tr), "reported_vol": x.std() * np.sqrt(12),
                     "true_vol": tr.std() * np.sqrt(12), "months": len(x)})
    return pd.DataFrame(rows).set_index("strategy").sort_values("theta0")


def private_study(D, F):
    obs, label, stand_in = hedgefunds.load_private()
    quarterly = (obs.index.to_series().diff().dt.days.median() > 45)
    ppy = 4 if quarterly else 12
    if quarterly:
        Fq = (1 + F).resample("QE").prod() - 1
        Fq = Fq[Fq.index.isin((1 + F).resample("QE").count().query("SPY == 3").index)]
        mkt, Fx, window, lag = Fq["SPY"], Fq, 20, 1
        min_hist = 24
    else:
        mkt, Fx, window, lag = F["SPY"], F, rp.WINDOW, rp.LAG
        min_hist = PRIVATE_MIN_HISTORY

    # full-sample GLM fit, with sensitivity to the number of lags
    sens = []
    for k in (1, 2, 3):
        f = desmooth.fit_glm(obs, k)
        tr = desmooth.unsmooth(obs, f["theta"])
        row = {"k": k, "aic": f["aic"], "valid": f["valid"], "xi": f["xi"],
               **{f"theta{j}": f["theta"][j] for j in range(k + 1)},
               **desmooth.summary(obs, tr, mkt, ppy)}
        sens.append(row)
    sens = pd.DataFrame(sens).set_index("k")
    fit = desmooth.fit_glm(obs, PRIVATE_K)
    true = desmooth.unsmooth(obs, fit["theta"])

    # walk-forward: at each decision date, re-estimate theta on reported data available then,
    # de-smooth, and fit the style regression to the de-smoothed returns
    common = obs.index.intersection(Fx.index)
    first = max(rp.WINDOW if not quarterly else window, 0)
    dates = [t for t in common[:-1]
             if len(obs.loc[:t - (pd.offsets.QuarterEnd(lag) if quarterly else pd.offsets.MonthEnd(lag))]) >= min_hist
             and len(common[common <= t]) >= first + lag]
    W, thetas = [], []
    for t in dates:
        end = t - (pd.offsets.QuarterEnd(lag) if quarterly else pd.offsets.MonthEnd(lag))
        hist = obs.loc[:end]
        f = desmooth.fit_glm(hist, PRIVATE_K)
        th = f["theta"] if f["valid"] else np.r_[1.0, np.zeros(PRIVATE_K)]
        tr_t = desmooth.unsmooth(hist, th)
        idx = tr_t.index.intersection(Fx.index)[-window:]
        w = rp.style_weights(tr_t.loc[idx], Fx.loc[idx, rp.FACTORS])
        W.append(pd.Series(w, index=rp.FACTORS, name=t))
        thetas.append(pd.Series(th, index=[f"theta{j}" for j in range(PRIVATE_K + 1)], name=t))
    W, thetas = pd.DataFrame(W), pd.DataFrame(thetas)

    rep = rp.replica_returns(W, Fx)
    scale = 12 / ppy
    def trk(target):
        t = rp.tracking(rep, target, Fx[data.CASH])
        if quarterly:   # rp.tracking annualizes monthly; rescale for quarters
            t["tracking_error"] /= np.sqrt(scale)
            for key in ("index_vol", "replica_vol"):
                t[key] /= np.sqrt(scale)
            for key in ("index_sharpe", "replica_sharpe"):
                t[key] /= np.sqrt(scale)
            n = t["months"]
            for key, ser in (("index_return", target.loc[rep.index.intersection(target.index)]),
                             ("replica_return", rep.loc[rep.index.intersection(target.index)])):
                t[key] = (1 + ser).prod() ** (ppy / n) - 1
            t["return_gap"] = t["replica_return"] - t["index_return"]
        return t
    track = pd.DataFrame({"vs de-smoothed returns": trk(true), "vs reported returns": trk(obs)}).T

    # backtest the replica through the engine (monthly), quarterly weights carried forward each month
    if quarterly:
        months = D["monthly"].index[(D["monthly"].index >= W.index[0]) & (D["monthly"].index <= F.index[-1])]
        Wm = W.reindex(months, method="ffill")
    else:
        Wm = W
    Wm = with_terminal_row(Wm, D["monthly"].index[D["monthly"].index <= F.index[-1]])
    bt_rows, bt_net = [], None
    for mult in (1.0, 2.0):
        r = backtest.run(Wm, D["monthly"], assets=rp.FACTORS, cost_mult=mult, name="private replica")
        st = backtest.stats(r["net"], r["cash"])
        bt_rows.append({"cost_multiple": mult, "net_return": st["annual_return"], "volatility": st["volatility"],
                        "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"],
                        "turnover_per_year": r["turnover"].mean() * 12, "cost_per_year": r["cost"].mean() * 12,
                        "start": r.index[0].date(), "end": r.index[-1].date(), "months": len(r)})
        if mult == 1.0:
            bt_net = r["net"]
    return {"label": label, "stand_in": stand_in, "periods_per_year": ppy, "obs": obs, "true": true, "fit": fit,
            "sensitivity": sens, "weights": W, "thetas": thetas, "tracking": track,
            "backtest": pd.DataFrame(bt_rows), "replica_net": bt_net}


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    D = data.load()
    F = factor_returns(D)
    H = hedgefunds.load_edhec()

    trk, weights, r2s, params, reps = replicate_hf(F, H)
    bt, nets = backtest_replicas(D, F, H, weights, hedgefunds.TARGETS)
    glm = glm_table(H)
    P = private_study(D, F)

    # 60/40 over the same window, for context
    from . import strategic
    W6 = {"60/40": strategic.build(D)[0]["60/40"]}
    r60 = backtest.run(W6["60/40"], D["monthly"], name="60/40")
    win = bt[["start", "end"]].iloc[0]
    r60 = r60.loc[str(win["start"]):str(win["end"])]
    s60 = backtest.stats(r60["net"], r60["cash"])

    trk.to_csv(f"{OUT}/hf_tracking_out_of_sample.csv", index=False)
    bt.to_csv(f"{OUT}/hf_replica_backtests.csv", index=False)
    glm.to_csv(f"{OUT}/glm_smoothing_edhec.csv")
    for (s, name), W in weights.items():
        if s in hedgefunds.TARGETS:
            slug = s.lower().replace("/", "-").replace(" ", "-")
            mslug = {"Rolling style analysis": "rbsa", "Kalman (unconstrained)": "kalman-free",
                     "Kalman (long-only)": "kalman-long-only"}[name]
            W.to_csv(f"{OUT}/weights_{slug}_{mslug}.csv")
    pd.DataFrame({f"{s} | {m}": v for (s, m), v in nets.items()}).to_csv(f"{OUT}/hf_replica_monthly_net.csv")
    P["sensitivity"].to_csv(f"{OUT}/private_desmoothing_sensitivity.csv")
    P["tracking"].to_csv(f"{OUT}/private_replica_tracking.csv")
    P["backtest"].to_csv(f"{OUT}/private_replica_backtest.csv", index=False)
    P["weights"].to_csv(f"{OUT}/private_replica_weights.csv")
    pd.DataFrame({"reported": P["obs"], "desmoothed": P["true"]}).to_csv(f"{OUT}/private_returns_reported_vs_desmoothed.csv")
    meta = {"edhec_source": H.attrs["source"], "edhec_end": str(H.index[-1].date()),
            "etf_price_file": D["source"], "factor_start": str(F.index[0].date()),
            "window_months": rp.WINDOW, "reporting_lag_months": rp.LAG,
            "kalman_params": {s: {k: (str(v.date()) if hasattr(v, "date") else v) for k, v in p.items()}
                              for s, p in params.items() if s in hedgefunds.TARGETS},
            "private_series": P["label"], "private_is_stand_in": P["stand_in"],
            "private_theta": P["fit"]["theta"].round(4).tolist(), "private_k": PRIVATE_K,
            "sixty_forty_same_window": {"net_return": s60["annual_return"], "sharpe": s60["sharpe"],
                                        "max_drawdown": s60["max_drawdown"]}}
    json.dump(meta, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)
    return {"trk": trk, "bt": bt, "glm": glm, "P": P, "meta": meta, "weights": weights}


if __name__ == "__main__":
    out = main()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t = out["trk"]
    print(t[t.strategy.isin(hedgefunds.TARGETS)][["strategy", "method", "tracking_error", "correlation", "oos_r2",
                                                   "index_return", "replica_return"]].round(3).to_string(index=False))
    print(out["bt"].round(3).to_string(index=False))
    print(out["glm"].round(3).to_string())
    P = out["P"]
    print(P["label"])
    print(P["sensitivity"].round(3).T.to_string())
    print(P["tracking"].round(3).to_string())
    print(P["backtest"].round(3).to_string(index=False))
    print(json.dumps(out["meta"], indent=1, default=str))
