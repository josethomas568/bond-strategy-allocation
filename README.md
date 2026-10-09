# Bond strategy allocation → global multi-asset allocation

Two studies built on public ETF data:

1. **[Global multi-asset allocation](#global-multi-asset-allocation)** — strategic portfolios, tactical signals (rules vs. machine learning), Monte Carlo of 10-year outcomes, and a one-page client note.
2. **[Bond strategy one-pager](#bond-strategy-one-pager)** — what a 10% or 20% bond or credit allocation does to a 60/40, with an investor PDF and a live Excel backup.

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
