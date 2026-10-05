"""
Refresh the input data: daily dividend-adjusted closes from Yahoo Finance.

    pip install yfinance
    python fetch_prices.py          # writes data/etf_prices_yahoo_<today>.csv
    python make_onepager.py         # uses the newest file in data/

The output has the same columns as the file the first one-pager was built from:
ticker, date, close, adj_close.
"""
import os
from datetime import date

import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "AGG", "BIL", "TLT", "LQD", "HYG", "TIP", "IEF", "JNK", "MBB", "EMB", "SHY", "BKLN"]

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
path = f"data/etf_prices_yahoo_{date.today():%Y-%m-%d}.csv"
out.to_csv(path, index=False)
print(f"wrote {path}: {len(out):,} rows, {out.ticker.nunique()} tickers, last date {out.date.max()}")
