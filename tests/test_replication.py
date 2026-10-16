"""Tests for the hedge fund / private-market replication module.

    python -m pytest -q tests/

Tests marked `needs_prices` use the Yahoo price file in data/ (gitignored; create it with
python -m multiasset.fetch_data) and are skipped without it.
"""
import glob

import numpy as np
import pandas as pd
import pytest

from multiasset import backtest, data, desmooth, hedgefunds, replication as rp

needs_prices = pytest.mark.skipif(not glob.glob("data/multiasset_prices_*.csv"), reason="price file not present")


@pytest.fixture(scope="module")
def D():
    return data.load()


@pytest.fixture(scope="module")
def F(D):
    from multiasset.run_replication import factor_returns
    return factor_returns(D)


# ------------------------------------------------------------------ engine
@needs_prices
def test_engine_default_unchanged(D):
    from multiasset import strategic
    W, L, _ = strategic.build(D)
    a = backtest.run(W["Risk parity"], D["monthly"])
    b = backtest.run(W["Risk parity"], D["monthly"], assets=data.ASSETS)
    pd.testing.assert_frame_equal(a, b)


@needs_prices
def test_engine_holds_cash(D, F):
    W = pd.DataFrame(0.0, index=F.index[:24], columns=rp.FACTORS)
    W[data.CASH] = 1.0
    r = backtest.run(W, D["monthly"], assets=rp.FACTORS)
    assert np.allclose(r["gross"], D["monthly"].loc[r.index, data.CASH])
    assert r["cost"].iloc[1:].abs().max() < 1e-12            # no trading after the first month


# ------------------------------------------------------------------ estimators on simulated data
def test_style_weights_recover_known_mix():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0.005, 0.04, (60, len(rp.FACTORS))), columns=rp.FACTORS)
    w = np.zeros(len(rp.FACTORS)); w[[0, 3, 10]] = [0.3, 0.2, 0.5]
    est = rp.style_weights(X.values @ w, X)
    assert np.abs(est - w).max() < 1e-4


def test_projection_is_feasible_and_idempotent():
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = rng.normal(0, 0.6, 10)
        p = rp.project_long_only(b)
        assert (p >= -1e-12).all() and p.sum() <= 1 + 1e-9
        assert np.allclose(rp.project_long_only(p), p)


def test_kalman_follows_drifting_exposure():
    rng = np.random.default_rng(2)
    T, k = 240, 3
    X = rng.normal(0, 0.04, (T, k))
    beta = np.cumsum(rng.normal(0, 0.05, (T, k)), axis=0) + 0.3
    y = (X * beta).sum(1) + rng.normal(0, 0.003, T)
    R, Q = rp.fit_kalman_params(y[:120], X[:120], beta[0])
    B, _ = rp.kalman_path(y, X, R, Q, beta[0], np.eye(k) * 0.25)
    static = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.abs(B[120:] - beta[120:]).mean() < 0.5 * np.abs(static - beta[120:]).mean()


def test_glm_recovers_smoothing_and_inverts():
    rng = np.random.default_rng(7)
    theta = np.array([0.5, 0.3, 0.2])
    true = pd.Series(rng.normal(0.007, 0.04, 600))
    obs = pd.Series(np.convolve(true, theta)[:600])
    f = desmooth.fit_glm(obs, 2)
    assert f["valid"] and np.abs(f["theta"] - theta).max() < 0.06
    rec = desmooth.unsmooth(obs, theta)
    assert (rec - true).iloc[30:].abs().max() < 1e-6
    assert desmooth.unsmooth(obs, theta).mean() == pytest.approx(obs.mean(), abs=2e-4)


# ------------------------------------------------------------------ no look-ahead
@needs_prices
def test_weights_do_not_use_future_data(F):
    H = hedgefunds.load_edhec()
    y = H["Long/Short Equity"].loc[F.index[0]:].dropna()
    common = y.index.intersection(F.index)
    y, Fc = y.loc[common], F.loc[common]
    cut = pd.Timestamp("2015-06-30")
    dates = [t for t in common[rp.WINDOW + rp.LAG - 1:] if t <= cut]
    full_r, _ = rp.rolling_style(y, Fc, dates)
    trunc_r, _ = rp.rolling_style(y.loc[:cut], Fc.loc[:cut], dates)
    pd.testing.assert_frame_equal(full_r, trunc_r)
    _, full_k, _ = rp.kalman_weights(y, Fc, dates)
    _, trunc_k, _ = rp.kalman_weights(y.loc[:cut], Fc.loc[:cut], dates)
    pd.testing.assert_frame_equal(full_k, trunc_k)


# ------------------------------------------------------------------ data loaders
def test_bundled_edhec():
    H = hedgefunds.load_edhec()
    assert H.shape == (293, 13)
    assert H.index[0] == pd.Timestamp("1997-01-31") and H.index[-1] == pd.Timestamp("2021-05-31")
    assert H.loc["1997-01-31", "Global Macro"] == pytest.approx(0.0573)


def test_cdli_file_is_parsed(tmp_path, monkeypatch):
    (tmp_path / "data" / "private").mkdir(parents=True)
    pd.DataFrame({"Quarter End": ["9/30/2004", "12/31/2004", "3/31/2005"],
                  "Total Return": ["2.10%", "2.50%", "-0.40%"]}).to_csv(tmp_path / "data/private/cdli_returns.csv", index=False)
    monkeypatch.chdir(tmp_path)
    s, label, stand_in = hedgefunds.load_private()
    assert not stand_in and "Cliffwater" in label
    assert list(s.round(4)) == [0.021, 0.025, -0.004]
    assert s.index[1] == pd.Timestamp("2004-12-31")


@needs_prices
def test_private_study_runs_quarterly(D, F, monkeypatch):
    """A synthetic appraisal-smoothed credit index (true = 0.6 HYG + 0.4 BIL per quarter) must come out
    quarterly, with smoothing found, volatility raised and the replica tilted to HYG."""
    from multiasset import run_replication as rr
    Fq = (1 + F).resample("QE").prod() - 1
    true = 0.6 * Fq["HYG"] + 0.4 * Fq["BIL"]
    obs = pd.Series(np.convolve(true.values, [0.5, 0.3, 0.2])[:len(true)], index=true.index).iloc[3:]
    monkeypatch.setattr(hedgefunds, "load_private", lambda: (obs.rename("synthetic"), "synthetic quarterly", False))
    P = rr.private_study(D, F)
    assert P["periods_per_year"] == 4
    assert P["fit"]["theta"][0] < 0.8 and P["sensitivity"].loc[2, "vol_ratio"] > 1.2
    assert P["weights"]["HYG"].mean() > 0.3
