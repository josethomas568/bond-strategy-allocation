"""
Refresh the multi-asset price file from Yahoo Finance (needs yfinance and internet access).

    pip install yfinance
    python -m multiasset.fetch_data     # writes data/multiasset_prices_yahoo_<today>.csv

Columns: ticker, date, close, adj_close. Dividends are recovered from the two price columns,
so no separate dividend file is needed.
"""
import os
from datetime import date

import pandas as pd
import yfinance as yf

from .data import ASSETS, CASH

TICKERS = ASSETS + [CASH, "SHY", "TLT", "AGG"]

raw = yf.download(TICKERS, start="2005-01-01", auto_adjust=False, actions=False, progress=False, group_by="ticker")
rows = []
for t in TICKERS:
    d = raw[t][["Close", "Adj Close"]].dropna().reset_index()
    d.columns = ["date", "close", "adj_close"]
    d.insert(0, "ticker", t)
    rows.append(d)
out = pd.concat(rows)
out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
os.makedirs("data", exist_ok=True)
path = f"data/multiasset_prices_yahoo_{date.today():%Y-%m-%d}.csv"
out.to_csv(path, index=False)
print(f"wrote {path}: {len(out):,} rows, last date {out.date.max()}")
