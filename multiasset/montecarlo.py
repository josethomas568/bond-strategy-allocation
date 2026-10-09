"""
Monte Carlo of 10-year outcomes.

Monthly net returns of all strategies are resampled jointly with a stationary block bootstrap
(mean block 12 months): every simulated month takes the same historical month for every strategy,
so cross-strategy correlation and crisis clustering are preserved and the strategies face the same
simulated markets. Cash is resampled alongside.

Limitation: the bootstrap can only replay the history it is given (here a period in which
U.S. stocks returned far more than their long-run average), so it describes the spread of outcomes
around that history, not a forecast.
"""
import numpy as np
import pandas as pd

from .stats_tests import stationary_bootstrap_idx

N_PATHS, HORIZON, BLOCK, SEED = 10_000, 120, 12, 2026
START_VALUE = 100_000


def simulate(returns, cash, n_paths=N_PATHS, horizon=HORIZON, block=BLOCK, seed=SEED):
    """returns: DataFrame (months x strategies) of monthly net returns; cash: Series on the same months."""
    R = returns.values
    c = cash.values
    rng = np.random.default_rng(seed)
    n, k = R.shape
    ann = np.empty((n_paths, k))
    mdd = np.empty((n_paths, k))
    beat_cash = np.empty((n_paths, k), dtype=bool)
    for p in range(n_paths):
        idx = stationary_bootstrap_idx(n, horizon, block, rng)
        path = R[idx]
        wealth = np.cumprod(1 + path, axis=0)
        ann[p] = wealth[-1] ** (12 / horizon) - 1
        mdd[p] = (wealth / np.maximum.accumulate(wealth, axis=0) - 1).min(axis=0)
        beat_cash[p] = wealth[-1] > np.prod(1 + c[idx])
    rows = []
    for j, name in enumerate(returns.columns):
        a, d = ann[:, j], mdd[:, j]
        end = START_VALUE * (1 + a) ** (horizon / 12)
        rows.append({
            "strategy": name,
            "median_annual_return": np.median(a),
            "p5_annual_return": np.percentile(a, 5),
            "p95_annual_return": np.percentile(a, 95),
            "prob_loss_10y": (a < 0).mean(),
            "prob_beat_cash": beat_cash[:, j].mean(),
            "median_max_drawdown": np.median(d),
            "p5_max_drawdown": np.percentile(d, 5),
            "prob_drawdown_over_20": (d < -0.20).mean(),
            "end_value_p5": np.percentile(end, 5),
            "end_value_median": np.median(end),
            "end_value_p95": np.percentile(end, 95),
        })
    return pd.DataFrame(rows).set_index("strategy"), ann, mdd
