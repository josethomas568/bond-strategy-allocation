# Bond strategy allocation → global multi-asset allocation

Three studies built on public ETF data:

1. **[Global multi-asset allocation](#global-multi-asset-allocation)** — strategic portfolios, tactical signals (rules vs. machine learning), Monte Carlo of 10-year outcomes, and a one-page client note.
2. **[Hedge fund and private-market replication](#hedge-fund-and-private-market-replication)** — replicating EDHEC hedge fund indices with the same ETFs (rolling style analysis vs. Kalman filter), de-smoothing a private-market index, and backtesting the replicas with trading costs.
3. **[Bond strategy one-pager](#bond-strategy-one-pager)** — what a 10% or 20% bond or credit allocation does to a 60/40, with an investor PDF and a live Excel backup.

---

## Global multi-asset allocation

**Client note:** [`output/client-note_global-multi-asset_2026-10-09.pdf`](output/client-note_global-multi-asset_2026-10-09.pdf)

### Universe

| Asset class | ETF | | Asset class | ETF |
|---|---|---|---|---|
| U.S. stocks | SPY | | High-yield credit | HYG |
| Developed ex-U.S. stocks | EFA | | Inflation-protected Treasuries | TIP |
| Emerging-market stocks | EEM | | Commodities | DBC |
| U.S. Treasuries (7–10 yr) | IEF | | Gold | GLD |
| Investment-grade credit | LQD | | U.S. REITs | VNQ |

Cash: BIL (1–3 month T-bills). Daily dividend-adjusted prices from Yahoo Finance, January 2005 – October 9, 2026.

### Design

- **Walk-forward, monthly.** At each month-end the portfolio is set using only data up to that close, held for a month, then rebalanced. Backtest: May 2008 – September 2026 (221 months), so the 2008 crisis, 2020 and 2022 are all inside it.
- **Estimation.** Ledoit-Wolf shrunk covariance of daily returns over the trailing 36 months (minimum 12, so the test can start in 2008).
- **Transaction costs.** Charged on every trade from drifted to target weights, 2–10 bps one-way per ETF (SPY/IEF 2, TIP/GLD 3, EFA/LQD/VNQ 5, EEM/HYG/DBC 10). All results are also run at 2× costs.
- **Strategic portfolios.**
  - **60/40:** 60% SPY / 40% IEF.
  - **Equal weight:** 1/N.
  - **Minimum variance:** long-only, at most 40% in any one ETF.
  - **Risk parity:** equal risk contribution (Spinu's convex formulation).
  - **Risk parity (levered):** scaled to the 60/40's ex-ante volatility, with leverage capped at 3×. It borrows at T-bills plus 0.5%.
  - **Constrained mean-variance:** long-only, at most 30% per ETF, risk aversion 5. Trailing means are shrunk 50% toward their cross-sectional average.
- **Tactical overlay.** Every signal tilts the same risk-parity base the same way: weight ∝ base × (1 + 0.5 × score), with the score in [−1, 1] and the portfolio fully invested.
  - **Time-series momentum:** the sign of each asset's 12-month return in excess of cash.
  - **Carry:** trailing 12-month distribution yield minus the T-bill yield, cross-sectional z-score. Dividends are recovered from the step in Yahoo's adjustment factor; the rebuilt annual totals match Yahoo's dividend history to the cent.
  - **Momentum + carry:** the average of the two scores.
  - **Gradient boosting (ML):** `HistGradientBoostingRegressor` on a (month × asset) panel.
    - Features: 17 inputs covering momentum at 1/3/6/12 months, trend, the 5-year return, volatility, carry and its change, cross-sectional ranks, market-wide variables and asset class.
    - Target: next-month excess return divided by trailing volatility.
    - Training: retrained every January on past data with a one-month embargo.
    - Tuning: a small grid chosen by **purged K-fold cross-validation** (5 time blocks; rows whose label overlaps a validation block are removed, plus a 3-month embargo), scored by the same rank correlation the overlay uses.
    - Out-of-sample predictions start in January 2013. All overlays are compared on those same 164 months.
- **Monte Carlo.** A joint stationary block bootstrap of monthly net returns: 10,000 paths of 120 months, mean block length 12, the same resampled months for every strategy. There is a sensitivity case in which every strategy earns only half its historical return above cash.

### Results

**Strategic portfolios, May 2008 – September 2026, after costs**

| Strategy | Return/yr | Volatility | Sharpe | Max drawdown | Turnover/yr | Sharpe at 2× costs |
|---|---|---|---|---|---|---|
| 60/40 | 8.3% | 9.7% | 0.75 | −27.6% | 0.14 | 0.75 |
| Equal weight | 5.5% | 10.4% | 0.45 | −32.6% | 0.17 | 0.44 |
| Minimum variance | 3.4% | 5.8% | 0.39 | −14.1% | 0.26 | 0.38 |
| Risk parity | 4.6% | 7.2% | 0.49 | −17.1% | 0.17 | 0.49 |
| Mean-variance | 6.2% | 8.8% | 0.59 | −22.7% | 1.17 | 0.57 |
| Risk parity (levered) | 6.5% | 13.0% | 0.46 | −32.5% | 0.38 | 0.45 |

- **60/40:** the plain 60/40 had the best risk-adjusted return. This sample is dominated by a long U.S. equity bull market, which rewards the portfolio with the most SPY.
- **Diversified portfolios:** they cut drawdowns sharply but gave up return.
- **Levered risk parity:** it did not reach the 60/40's risk-adjusted return. Its 13% realized volatility is mostly 2008 (31% annualized from mid-2008 to mid-2009), when leveraged credit, REITs and commodities fell together. It also lost 22% in 2022, against 17% for the 60/40.

**Tactical overlays on risk parity, out-of-sample window February 2013 – September 2026**

| Overlay on risk parity | Return/yr | Sharpe | Max drawdown | Turnover/yr | Full-period Sharpe (2008–) |
|---|---|---|---|---|---|
| No overlay | 4.2% | 0.41 | −15.5% | 0.13 | 0.49 |
| Momentum (TSMOM) | 4.5% | 0.46 | −15.9% | 1.01 | 0.56 |
| Carry | 4.1% | 0.39 | −16.8% | 0.21 | 0.49 |
| Momentum + carry | 4.4% | 0.44 | −16.2% | 0.54 | 0.54 |
| Gradient boosting (ML) | 3.9% | 0.35 | −16.9% | 1.44 | — |

| Signal (predicting next-month excess returns) | Mean rank IC | t-stat | Months IC > 0 |
|---|---|---|---|
| Momentum (TSMOM) | +0.085 | +2.29 | 56% |
| Carry | +0.006 | +0.22 | 51% |
| Momentum + carry | +0.002 | +0.07 | 51% |
| Gradient boosting (ML) | −0.023 | −0.72 | 52% |

| Comparison (block bootstrap, 5,000 draws) | Sharpe difference | 95% CI | One-sided p |
|---|---|---|---|
| Momentum vs. no overlay | +0.06 | −0.08 to +0.20 | 0.21 |
| Carry vs. no overlay | −0.02 | −0.12 to +0.09 | 0.62 |
| Momentum + carry vs. no overlay | +0.03 | −0.04 to +0.12 | 0.22 |
| ML vs. no overlay | −0.05 | −0.15 to +0.05 | 0.84 |
| ML vs. momentum | −0.11 | −0.28 to +0.05 | 0.91 |

**The ML signal did not beat the simple rules.**
- **ML:** its out-of-sample rank correlation with next-month returns was slightly negative, and its overlay had a lower Sharpe than both the momentum rule and the unadjusted base. The model design, features and tuning grid were fixed before any out-of-sample result was looked at, and were not changed afterward.
- **Momentum:** the only signal with a measurable edge (IC t-stat 2.3). Even that does not survive a correction for testing four signals, and its Sharpe improvement is not statistically significant.
- **Carry:** added nothing here. For gold and commodity ETFs, distribution yield is a poor proxy for carry because the futures roll yield is not observed.

**Monte Carlo, 10 years**

| Strategy | Median return/yr | 5th–95th pct | P(10-yr loss) | P(drawdown > 20%) | Median, half premium | P(loss), half premium |
|---|---|---|---|---|---|---|
| 60/40 | 8.6% | 3.5% to 12.8% | 0.4% | 56% | 4.8% | 5.5% |
| Equal weight | 5.6% | 0.3% to 10.7% | 4.3% | 48% | 3.2% | 15.1% |
| Minimum variance | 3.5% | 0.4% to 6.4% | 3.3% | 8% | 2.3% | 10.3% |
| Risk parity | 4.7% | 1.0% to 8.3% | 2.0% | 16% | 2.9% | 9.8% |
| Mean-variance | 6.2% | 2.1% to 10.6% | 0.6% | 38% | 3.5% | 7.4% |
| Risk parity (levered) | 6.6% | −0.3% to 13.6% | 5.6% | 67% | 3.5% | 19.4% |
| Risk parity + momentum | 5.0% | 1.7% to 8.3% | 0.7% | 7% | 3.0% | 6.6% |
| Risk parity + carry | 4.8% | 1.1% to 8.2% | 1.7% | 17% | 2.9% | 9.0% |
| Risk parity + momentum + carry | 4.9% | 1.5% to 8.2% | 1.0% | 10% | 3.0% | 7.3% |

The bootstrap replays 2008–2026, which was unusually kind to U.S. stocks, so the half-premium columns are the more cautious reading. A second run on 2013 onward that includes the ML overlay is in `results/multiasset/monte_carlo_10y_2013_onward_with_ml.csv`.

### Checks

- **No look-ahead:** after cutting the data at June 30, 2016 and rerunning, every strategic weight, leverage factor, signal, ML feature and ML prediction up to that date was identical to the full run.
- **Purged folds:** no training row's label month falls inside its validation block.
- **Risk parity:** risk contributions are equal (10.0% each).
- **60/40 return:** recomputed directly from SPY and IEF monthly returns, 8.33% a year gross, matching the engine.

### Run

```bash
pip install -r requirements.txt
python -m multiasset.fetch_data     # optional: refresh prices (needs yfinance and access to Yahoo Finance)
python -m multiasset.run            # backtests, overlays, ML walk-forward, Monte Carlo -> results/multiasset/ (~2 min)
python -m multiasset.client_note    # one-page client note -> output/
```

| File | What it does |
|---|---|
| `multiasset/data.py` | Universe, prices, monthly returns, dividends and trailing yields |
| `multiasset/portfolios.py` | 60/40, equal weight, minimum variance, risk parity, constrained mean-variance |
| `multiasset/backtest.py` | Walk-forward engine: drifted weights, per-ETF trading costs, leverage and financing |
| `multiasset/strategic.py` | Monthly strategic decisions, including levered risk parity |
| `multiasset/signals.py` | Momentum, carry and the tilt mechanism shared by all overlays |
| `multiasset/ml.py` | Feature panel, purged K-fold CV, annual walk-forward gradient boosting |
| `multiasset/stats_tests.py` | Information coefficients, stationary bootstrap, Sharpe-difference intervals |
| `multiasset/tactical.py` | Runs every overlay on the same base and the same out-of-sample months |
| `multiasset/montecarlo.py` | Joint block-bootstrap of 10-year outcomes |
| `multiasset/client_note.py` | Plain-language one-pager; every sentence's numbers come from `results/multiasset/` |

---

## Hedge fund and private-market replication

Can the same 10 ETFs plus T-bills reproduce hedge fund strategy returns? And what do private-market returns look like once the appraisal smoothing is removed? The module reuses the multi-asset universe, price file and backtest engine.

### Data

- **Hedge funds:** [EDHEC-Risk alternative indices](https://climateimpact.edhec.edu/all-downloads-hedge-funds-indices), 13 monthly strategy indices, net of fees.
  - The copy used here is the one EDHEC licenses to the R package PerformanceAnalytics, covering January 1997 to May 2021. `multiasset/fetch_hedgefunds.py` downloads it from GitHub and checks it against an independent copy: all 199 overlapping months agree exactly.
  - It is saved at `multiasset/reference_data/edhec_alternative_indices.csv`.
  - To extend the study past May 2021, save a current EDHEC download as `data/edhec_*.csv`. It is used automatically.
- **Overlap with the ETFs:** June 2007 (BIL's first full month) to May 2021, 168 months. After the 36-month estimation window and the reporting lag, 131 months are out of sample (July 2010 – May 2021).
- **Private markets:** the [Cliffwater Direct Lending Index](https://www.cliffwaterdirectlendingindex.com/about), quarterly.
  - Its quarterly history needs a free registration, so it isn't bundled. Save it as `data/private/cdli*.csv` or `.xlsx`.
  - Until that file exists, the private-markets section runs on **EDHEC Distressed Securities** as a stand-in. Distressed Securities is the most autocorrelated credit-like index available here. The results below are for that stand-in.

### Method

- **Rolling style analysis** (Sharpe, 1992). Each month, the strategy's last 36 monthly returns are regressed on the 10 ETFs and T-bills. Weights are constrained to be ≥ 0 and sum to 1, so the fitted weights are a long-only, fully funded portfolio.
- **Kalman filter.** Exposures follow a random walk. The strategy's excess return is the ETFs' excess returns times today's exposures, plus noise.
  - The observation and state noise are estimated by maximum likelihood on the first 36 months only and then fixed. The filter then runs forward one month at a time.
  - **Unconstrained:** exposures may be short or sum above 1. This is a paper replication only.
  - **Long-only:** after each update the exposures are projected onto {≥ 0, sum ≤ 1}, with the rest in T-bills. This version is tradable, like the regression portfolio.
- **No look-ahead.** EDHEC publishes a month's return during the following month, so the weights held in month t+1 use data only through month t−1. A test cuts the data in June 2015 and confirms every weight up to then is unchanged.
- **De-smoothing** (Getmansky, Lo & Makarov, 2004).
  - Reported returns are modeled as a weighted average of the current and last two months' true returns, with weights θ₀ + θ₁ + θ₂ = 1.
  - θ is estimated by maximum likelihood as an MA(2) and normalised. Inverting the filter gives the true returns. Volatility scales by 1/√(Σθ²); the mean is unchanged.
  - For the replication, θ is re-estimated each month on the data available at the time. The style regression is then fitted to those de-smoothed returns.
- **Backtest.** The long-only replicas run through `multiasset/backtest.py`, with monthly rebalancing and the same per-ETF trading costs (BIL 1 bp), at 1× and 2× costs.
  - The engine gained an optional `assets` argument so a portfolio can hold T-bills explicitly. With the default, every earlier result is unchanged to 1e-16.

### Results

**Out-of-sample tracking, gross of costs, July 2010 – May 2021 (131 months)**

| Strategy | Method | Tracking error | Correlation | Out-of-sample R² | Index return/yr | Replica return/yr | Avg. T-bills |
|---|---|---|---|---|---|---|---|
| Long/short equity | Rolling style analysis | 2.7% | 0.92 | 0.84 | 6.9% | 5.7% | 53% |
| Long/short equity | Kalman, unconstrained | 3.0% | 0.90 | 0.80 | 6.9% | 6.3% | 64% |
| Long/short equity | Kalman, long-only | 3.2% | 0.88 | 0.77 | 6.9% | 6.9% | 39% |
| Global macro | Rolling style analysis | 2.9% | 0.67 | 0.44 | 4.0% | 2.8% | 63% |
| Global macro | Kalman, unconstrained | 3.1% | 0.65 | 0.35 | 4.0% | 2.9% | 67% |
| Global macro | Kalman, long-only | 3.2% | 0.66 | 0.30 | 4.0% | 3.6% | 47% |
| Event driven | Rolling style analysis | 3.8% | 0.84 | 0.70 | 6.3% | 5.6% | 52% |
| Event driven | Kalman, unconstrained | 3.6% | 0.86 | 0.72 | 6.3% | 4.2% | 80% |
| Event driven | Kalman, long-only | 4.2% | 0.81 | 0.63 | 6.3% | 6.9% | 6% |

- **Long/short equity** is mostly market exposure. On average about 24% SPY, 9% EFA and 4% EEM, with half in T-bills; the replica's correlation with the index is 0.92.
- **Event driven** adds high-yield credit to the same equity mix: 18% SPY and 12% HYG in the regression.
- **Global macro** is hard to replicate with long-only ETFs (correlation 0.67). Its returns come from timing and shorting that a long-only portfolio cannot reproduce.
- **The Kalman filter did not beat the simple rolling regression** on tracking error for long/short equity or global macro. For event driven, the likelihood estimate set the exposure drift to zero, so its filter treats exposures as fixed.
- **Long-only Kalman replicas** came closest to the indices' returns because they held less cash. That came at the price of higher tracking error.
- **Across all 13 EDHEC indices,** out-of-sample correlation runs from 0.92 (long/short equity) to 0.21 (short selling, which a long-only replica cannot reproduce). The full table is in `results/replication/hf_tracking_out_of_sample.csv`.

**Replication portfolios backtested with trading costs, July 2010 – May 2021**

| Strategy | Method | Net return/yr | Volatility | Sharpe | Max drawdown | Turnover/yr | Cost/yr | Sharpe at 2× costs | Index return/yr | Index Sharpe | Index max drawdown | Net tracking error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Long/short equity | Rolling style analysis | 5.7% | 5.9% | 0.90 | −10.2% | 0.42 | 4 bp | 0.89 | 6.9% | 0.96 | −10.9% | 2.7% |
| Long/short equity | Kalman, long-only | 6.8% | 6.5% | 0.98 | −9.7% | 0.78 | 7 bp | 0.97 | 6.9% | 0.96 | −10.9% | 3.2% |
| Global macro | Rolling style analysis | 2.8% | 3.0% | 0.80 | −3.6% | 0.59 | 4 bp | 0.79 | 4.0% | 0.92 | −4.5% | 2.9% |
| Global macro | Kalman, long-only | 3.5% | 4.0% | 0.79 | −4.4% | 0.70 | 6 bp | 0.78 | 4.0% | 0.92 | −4.5% | 3.2% |
| Event driven | Rolling style analysis | 5.5% | 5.2% | 0.98 | −8.3% | 0.67 | 7 bp | 0.96 | 6.3% | 0.86 | −15.1% | 3.8% |
| Event driven | Kalman, long-only | 6.8% | 6.6% | 0.98 | −10.2% | 0.34 | 3 bp | 0.98 | 6.3% | 0.86 | −15.1% | 4.2% |

- **Costs:** trading costs are 3–7 bp a year, so 2× costs barely move the result. The replicas' turnover is low because the exposures change slowly.
- **Fees:** the indices are net of hedge fund fees. Here the replicas matched or came within about 1.2% a year of the index returns, at Sharpe ratios of 0.79–0.98 against the indices' 0.86–0.96.
- **Context:** a plain 60/40 earned 11.2% a year at a 1.40 Sharpe over the same months. This decade favoured simple equity beta over every hedge fund style.

**How smoothed are hedge fund returns? GLM fit on each EDHEC index, 1997–2021**

| Strategy | θ₀ | θ₁ | θ₂ | Smoothing index Σθ² | Reported 1-month autocorr. | After de-smoothing | Reported vol | True vol | Constraints hold |
|---|---|---|---|---|---|---|---|---|---|
| Convertible arbitrage | 0.59 | 0.30 | 0.11 | 0.45 | 0.50 | 0.01 | 5.8% | 8.6% | yes |
| Fixed income arbitrage | 0.59 | 0.31 | 0.09 | 0.46 | 0.48 | −0.00 | 4.0% | 5.9% | yes |
| Distressed securities | 0.63 | 0.25 | 0.11 | 0.48 | 0.44 | 0.02 | 6.3% | 9.0% | yes |
| Relative value | 0.67 | 0.24 | 0.08 | 0.52 | 0.38 | 0.01 | 4.1% | 5.7% | yes |
| Event driven | 0.74 | 0.19 | 0.07 | 0.59 | 0.28 | 0.01 | 6.6% | 8.5% | yes |
| Long/short equity | 0.80 | 0.14 | 0.05 | 0.67 | 0.20 | 0.00 | 7.2% | 8.9% | yes |
| Global macro | 0.93 | 0.06 | 0.01 | 0.87 | 0.07 | 0.00 | 5.1% | 5.4% | yes |
| CTA global | 1.06 | −0.02 | −0.04 | 1.13 | −0.01 | 0.01 | 7.9% | 7.4% | no: not smoothed |

Eight of the 13 rows are shown; the rest are in `results/replication/glm_smoothing_edhec.csv`. Smoothing tracks illiquidity, as GLM found: the arbitrage and distressed strategies hold hard-to-price securities and are the most smoothed. Futures-based CTA and macro funds aren't smoothed at all.

**Private markets: stand-in series (EDHEC Distressed Securities, monthly, 1997–2021)**

| | Reported | De-smoothed (GLM, k = 2) |
|---|---|---|
| Volatility | 6.3% | 9.0% (×1.43) |
| 1-month autocorrelation | 0.44 | 0.02 |
| Beta to SPY | 0.32 | 0.49 |
| Beta to SPY with 2 lags (Dimson) | 0.53 | 0.54 |
| Max drawdown | −22.9% | −24.6% |
| Mean return | 8.2%/yr | 8.2%/yr |

- **Reported risk understates true risk.** The reported series hides about 30% of the true volatility and a third of the equity beta.
- **The Dimson check agrees.** Adding lagged market returns to the reported series gives a beta of 0.53, matching the de-smoothed beta. So the extra beta is the market's effect arriving late, not an artefact of the method.
- **The lag choice matters.** With one lag the volatility multiple is 1.25×; with three lags it is 1.52×. Two lags has the lowest AIC (`private_desmoothing_sensitivity.csv`).
- **The de-smoothed returns are easier to replicate.** On average the replica held 19% SPY, 11% EFA, 15% HYG, 8% LQD, 6% DBC and 30% T-bills. Against the de-smoothed series, its out-of-sample correlation is 0.77 (R² 0.59) and the index-on-replica beta is 0.99. Against the reported returns, correlation is 0.74, R² 0.35 and beta 0.63, because the reported series reacts to markets late and only partly.
- **The replica has no smoothing to flatter it.** Its Sharpe of 0.87 sits between the reported index's 0.89, which smoothing inflates, and the de-smoothed index's 0.60. Backtested with costs, it earned 6.5% a year at 7.2% volatility (Sharpe 0.85; 0.84 at 2× costs).

### Run

```bash
pip install -r requirements.txt
python -m multiasset.run_replication     # -> results/replication/  (~40 s)
python -m pytest -q                      # 10 tests: engine compatibility, estimators on simulated data, no look-ahead, CDLI loader
python -m multiasset.fetch_hedgefunds    # optional: re-download and re-verify the EDHEC indices
```

When `data/private/cdli*.csv` (or `.xlsx`) is present, `run_replication` uses it automatically. It runs quarterly, with a 20-quarter regression window, a one-quarter reporting lag and SPY compounded to quarters.

| File | What it does |
|---|---|
| `multiasset/hedgefunds.py` | EDHEC and CDLI loaders; the stand-in rule |
| `multiasset/replication.py` | Constrained style regression, Kalman filter (unconstrained and projected long-only), out-of-sample tracking statistics |
| `multiasset/desmooth.py` | GLM estimation, inversion, Dimson beta, before/after statistics |
| `multiasset/run_replication.py` | Runs the three studies and the cost backtests, and writes `results/replication/` |
| `multiasset/fetch_hedgefunds.py` | Downloads and cross-checks the EDHEC indices |
| `tests/test_replication.py` | The tests above |

---

## Bond strategy one-pager

Tests what a 10% or 20% allocation to a bond or credit strategy does to a 60% stocks / 40% bonds portfolio, using public ETFs as stand-ins, and writes an investor one-pager (PDF) plus a live Excel backup.

| Strategy | Stand-in ETF |
|---|---|
| Long-term Treasuries | TLT |
| Investment-grade credit | LQD |
| High-yield credit | HYG |
| Inflation-protected bonds | TIP |

Base portfolio: SPY (stocks) and AGG (bonds), rebalanced monthly. Cash for the Sharpe ratio: BIL. Period: June 2007 onward, so all three stress windows are covered (2008, 2020, 2022).

### Run

```bash
pip install -r requirements.txt
python fetch_prices.py      # optional: refresh data (needs yfinance and internet access to Yahoo Finance)
python make_onepager.py     # writes output/bond-strategy-allocation_<date>.pdf and .xlsx
```

`make_onepager.py` recalculates the workbook with LibreOffice when it is installed and checks 153 workbook figures against the Python analysis before writing the PDF. Without LibreOffice, Excel calculates the workbook when the file is opened.

| File | What it does |
|---|---|
| `analysis.py` | Returns, monthly-rebalanced portfolios, drawdowns, rolling volatility, correlations, stress windows |
| `build_workbook.py` | Excel backup: prices are the only data; every other number is a formula |
| `build_pdf.py` | One-page PDF; the takeaways are written from the numbers, so they update with new data |
| `make_onepager.py` | Runs everything and verifies the workbook against the analysis |
| `fetch_prices.py` | Downloads fresh ETF prices from Yahoo Finance |

Edit the yellow cells on the workbook's Assumptions sheet (base weights, allocation sizes, stress-window dates), or the constants at the top of `analysis.py` (`STRATEGIES`, `ALLOCS`, `STRESS`) to change the PDF.

---

*Past results and simulations do not predict future returns. Educational project, not investment advice.*
