"""Tactical overlays on the risk-parity base: rules vs. machine learning, judged on the same months."""
import pandas as pd

from . import backtest, ml, signals, stats_tests

BASE = "Risk parity"


def run(D, W, ml_out=None):
    base = W[BASE]
    fwd = signals.next_month_excess(D)
    pred, log, X, y = ml_out if ml_out is not None else ml.walk_forward(D)
    scores = {
        "Momentum (TSMOM)": signals.tsmom(D),
        "Carry": signals.carry(D),
        "Momentum + carry": signals.combined(D),
        "Gradient boosting (ML)": ml.ml_score(pred),
    }
    oos_start = pd.Timestamp(ml.FIRST_OOS)
    weights = {BASE + " (no overlay)": base}
    for k, s in scores.items():
        b = base if k != "Gradient boosting (ML)" else base.loc[base.index >= oos_start]
        weights[k] = signals.tilt(b, s)
    res = {k: backtest.run(w, D["monthly"], name=k) for k, w in weights.items()}

    # evaluation on the common out-of-sample window (months the ML model never trained on)
    common = res["Gradient boosting (ML)"].index
    rows, tests, ics = [], [], []
    for k, r in res.items():
        rc = r.loc[common]
        s = backtest.stats(rc["net"], rc["cash"])
        full = backtest.stats(r["net"], r["cash"]) if k != "Gradient boosting (ML)" else s
        rows.append({"overlay": k, **{f"oos_{a}": b for a, b in s.items()},
                     "full_sharpe": full["sharpe"], "full_annual_return": full["annual_return"],
                     "full_max_drawdown": full["max_drawdown"], "turnover_per_year": rc["turnover"].mean() * 12,
                     "cost_per_year": rc["cost"].mean() * 12})
    table = pd.DataFrame(rows).set_index("overlay")

    base_r = res[BASE + " (no overlay)"].loc[common]
    for k in scores:
        r = res[k].loc[common]
        t = stats_tests.sharpe_diff_ci(r["net"], base_r["net"], r["cash"])
        tests.append({"comparison": f"{k} vs no overlay", **t})
    rule_best = max([k for k in scores if k != "Gradient boosting (ML)"], key=lambda k: table.loc[k, "oos_sharpe"])
    t = stats_tests.sharpe_diff_ci(res["Gradient boosting (ML)"]["net"], res[rule_best].loc[common]["net"],
                                   res[rule_best].loc[common]["cash"])
    tests.append({"comparison": f"Gradient boosting (ML) vs {rule_best}", **t})

    ml_raw = pred.unstack("asset")
    raw = {"Momentum (TSMOM)": signals.excess_return(D, 12), "Carry": signals.carry_raw(D),
           "Momentum + carry": signals.combined(D), "Gradient boosting (ML)": ml_raw}
    for k, s in raw.items():
        ic_oos = stats_tests.ic_series(s.loc[s.index.isin(common - pd.offsets.MonthEnd(1))], fwd)
        ics.append({"signal": k, **stats_tests.ic_summary(ic_oos)})
    return {"results": res, "table": table, "tests": pd.DataFrame(tests).set_index("comparison"),
            "ic": pd.DataFrame(ics).set_index("signal"), "ml_log": log, "pred": pred, "weights": weights,
            "best_rule": rule_best, "common": common}
