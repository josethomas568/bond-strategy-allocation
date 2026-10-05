# Bond strategy allocation one-pager

Tests what a 10% or 20% allocation to a bond or credit strategy does to a 60% stocks / 40% bonds portfolio, using public ETFs as stand-ins, and writes an investor one-pager (PDF) plus a live Excel backup.

| Strategy | Stand-in ETF |
|---|---|
| Long-term Treasuries | TLT |
| Investment-grade credit | LQD |
| High-yield credit | HYG |
| Inflation-protected bonds | TIP |

Base portfolio: SPY (stocks) and AGG (bonds), rebalanced monthly. Cash for the Sharpe ratio: BIL. Period: June 2007 onward, so all three stress windows are covered (2008, 2020, 2022).

## Run

```bash
pip install pandas numpy openpyxl reportlab matplotlib
python fetch_prices.py      # optional: refresh data (needs yfinance and internet access to Yahoo Finance)
python make_onepager.py     # writes output/bond-strategy-allocation_<date>.pdf and .xlsx
```

`make_onepager.py` recalculates the workbook with LibreOffice when it is installed and checks 153 workbook figures against the Python analysis before writing the PDF. Without LibreOffice, Excel calculates the workbook when the file is opened.

## Files

| File | What it does |
|---|---|
| `analysis.py` | Returns, monthly-rebalanced portfolios, drawdowns, rolling volatility, correlations, stress windows |
| `build_workbook.py` | Excel backup: prices are the only data; every other number is a formula |
| `build_pdf.py` | One-page PDF; the takeaways are written from the numbers, so they update with new data |
| `make_onepager.py` | Runs everything and verifies the workbook against the analysis |
| `fetch_prices.py` | Downloads fresh ETF prices from Yahoo Finance |

## Changing the scenarios

Edit the yellow cells on the workbook's Assumptions sheet (base weights, allocation sizes, stress-window dates), or the constants at the top of `analysis.py` (`STRATEGIES`, `ALLOCS`, `STRESS`) to change the PDF.
