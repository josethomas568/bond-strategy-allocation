"""
Data layer for the global multi-asset framework.

Everything is indexed by month-end. A value stamped at month-end t uses only data
available at the close of t. Returns stamped at t are the returns earned during month t
(from the close of t-1 to the close of t).
"""
import glob

import numpy as np
import pandas as pd

UNIVERSE = {  # ticker: (asset class, plain-language name)
    "SPY": ("Equity", "U.S. stocks"),
    "EFA": ("Equity", "Developed-market stocks"),
    "EEM": ("Equity", "Emerging-market stocks"),
    "IEF": ("Rates", "U.S. Treasuries (7-10 yr)"),
    "LQD": ("Credit", "Investment-grade corporate bonds"),
    "HYG": ("Credit", "High-yield corporate bonds"),
    "TIP": ("Rates", "Inflation-protected Treasuries"),
    "DBC": ("Real assets", "Broad commodities"),
    "GLD": ("Real assets", "Gold"),
    "VNQ": ("Real assets", "U.S. real estate (REITs)"),
}
ASSETS = list(UNIVERSE)
CASH = "BIL"

# One-way trading cost per unit of turnover (bid-ask spread plus a little market impact), in decimals.
COST = pd.Series({"SPY": 2, "EFA": 5, "EEM": 10, "IEF": 2, "LQD": 5, "HYG": 10, "TIP": 3,
                  "DBC": 10, "GLD": 3, "VNQ": 5}, dtype=float) / 1e4
FIRST_DECISION = "2008-04-30"   # first month-end with 12 months of history for every asset and the cash yield


def load(path=None):
    path = path or sorted(glob.glob("data/multiasset_prices_*.csv"))[-1]
    df = pd.read_csv(path, parse_dates=["date"]).sort_values(["ticker", "date"])
    px = df.pivot(index="date", columns="ticker", values="adj_close")
    close = df.pivot(index="date", columns="ticker", values="close")
    cols = ASSETS + [CASH]
    px, close = px[cols], close[cols]

    # Dividends are recovered from the step in the adjustment factor on each ex-date:
    # factor_{t-1} = factor_t * (1 - D_t / close_{t-1})  =>  D_t = close_{t-1} * (1 - factor_{t-1} / factor_t)
    f = px / close
    div = (close.shift() * (1 - f.shift() / f)).where(lambda x: x > 1e-6, 0.0).fillna(0.0)

    daily = px.pct_change()
    me = px.resample("ME").last()
    month_ret = me.pct_change()
    # trailing 12-month distribution yield at each month-end (income paid over the last year / price today)
    div12 = div.resample("ME").sum().rolling(12, min_periods=12).sum()
    yld = div12 / close.resample("ME").last()
    return {"daily": daily, "monthly": month_ret, "yield": yld, "month_px": me, "source": path,
            "last_date": px.index.max()}
