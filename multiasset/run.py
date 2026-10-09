"""
Run the whole multi-asset study and write results/multiasset/*.

    python -m multiasset.run            # about 2 minutes
"""
import json
import os

import numpy as np
import pandas as pd

from . import backtest, data, ml, montecarlo, strategic, tactical
from .portfolios import risk_contributions

OUT = "results/multiasset"


def main(price_file=None):
    os.makedirs(OUT, exist_ok=True)
    D = data.load(price_file)
    W, L, covs = strategic.build(D)
    S = strategic.run_all(D, W, L)
    S2 = strategic.run_all(D, W, L, cost_mult=2.0)
    strat = strategic.summary(S)
    strat["sharpe_at_2x_costs"] = strategic.summary(S2)["sharpe"]

    ml_out = ml.walk_forward(D)
    T = tactical.run(D, W, ml_out)

    # Monte Carlo 1: full history, strategic portfolios + rule-based overlays
    full = {k: v["net"] for k, v in S.items()}
    for k in ("Momentum (TSMOM)", "Carry", "Momentum + carry"):
        full[f"Risk parity + {k.split(' (')[0].lower()}"] = T["results"][k]["net"]
    R = pd.DataFrame(full)
    cash = S["60/40"]["cash"]
    mc, ann, mdd = montecarlo.simulate(R, cash)
    # sensitivity: every strategy keeps its risk but earns only half its historical return above cash
    R_half = R - 0.5 * R.sub(cash, axis=0).mean()
    mc_half, _, _ = montecarlo.simulate(R_half, cash)
    # Monte Carlo 2: 2013 onward, adding the ML overlay (it only has out-of-sample returns from 2013)
    common = T["common"]
    R_ml = R.loc[common].copy()
    R_ml["Risk parity + gradient boosting"] = T["results"]["Gradient boosting (ML)"]["net"]
    mc_ml, _, _ = montecarlo.simulate(R_ml, cash.loc[common])

    # current allocations and risk contributions at the last decision date
    t_last = W["60/40"].index[-1]
    current = pd.DataFrame({k: w.loc[t_last] for k, w in W.items() if k != "Risk parity (levered)"})
    current["Risk parity + momentum"] = T["weights"]["Momentum (TSMOM)"].loc[t_last]
    rc = pd.DataFrame({k: risk_contributions(current[k], covs[t_last]) for k in current})

    # write everything
    strat.to_csv(f"{OUT}/strategic_backtest.csv")
    T["table"].to_csv(f"{OUT}/tactical_overlays.csv")
    T["tests"].to_csv(f"{OUT}/sharpe_difference_tests.csv")
    T["ic"].to_csv(f"{OUT}/signal_information_coefficients.csv")
    T["ml_log"].to_csv(f"{OUT}/ml_walk_forward_log.csv", index=False)
    mc.to_csv(f"{OUT}/monte_carlo_10y.csv")
    mc_half.to_csv(f"{OUT}/monte_carlo_10y_half_premium.csv")
    mc_ml.to_csv(f"{OUT}/monte_carlo_10y_2013_onward_with_ml.csv")
    current.to_csv(f"{OUT}/current_weights.csv")
    rc.to_csv(f"{OUT}/current_risk_contributions.csv")
    pd.DataFrame({k: v["net"] for k, v in {**S, **T["results"]}.items()}).to_csv(f"{OUT}/monthly_net_returns.csv")
    pd.concat({k: w for k, w in W.items()}, names=["strategy", "date"]).to_csv(f"{OUT}/strategic_weights.csv")
    meta = {"data_file": D["source"], "last_price_date": str(D["last_date"].date()),
            "backtest_start": str(R.index[0].date()), "backtest_end": str(R.index[-1].date()),
            "months": len(R), "ml_oos_start": str(common[0].date()), "ml_oos_months": len(common),
            "best_rule": T["best_rule"], "monte_carlo_paths": montecarlo.N_PATHS,
            "decision_date_for_current_weights": str(t_last.date())}
    json.dump(meta, open(f"{OUT}/run_info.json", "w"), indent=1)
    return {"D": D, "W": W, "L": L, "S": S, "strat": strat, "T": T, "mc": mc, "mc_half": mc_half,
            "mc_ml": mc_ml, "ann": ann, "mdd": mdd, "R": R, "current": current, "rc": rc, "meta": meta}


if __name__ == "__main__":
    out = main()
    pd.set_option("display.width", 250)
    print(json.dumps(out["meta"], indent=1))
    print(out["strat"].round(3).to_string())
    print(out["T"]["table"].round(3).to_string())
    print(out["mc"].round(3).to_string())
    print(out["mc_half"][["median_annual_return", "p5_annual_return", "prob_loss_10y", "prob_beat_cash"]].round(3).to_string())
    print(out["mc_ml"][["median_annual_return", "p5_annual_return", "prob_drawdown_over_20"]].round(3).to_string())
