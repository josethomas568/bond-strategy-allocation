"""
Hedge fund and private-market index data.

* EDHEC-Risk alternative indices (monthly, 13 strategies). The bundled copy runs Jan 1997 - May 2021
  (see fetch_hedgefunds.py). A newer EDHEC download saved as data/edhec_*.csv is used instead when present.
* Private-market index (quarterly). Cliffwater Direct Lending Index (CDLI) total returns, saved by the
  user as data/private/cdli*.csv or .xlsx (free registration at cliffwaterdirectlendingindex.com).
  Until that file exists, the de-smoothing study runs on a clearly labelled stand-in: EDHEC Distressed
  Securities, the most autocorrelated credit-like series available here.

Returns are decimals, stamped at the month-end (or quarter-end) of the period they were earned.
"""
import glob
import os
import re

import pandas as pd

REF = os.path.join(os.path.dirname(__file__), "reference_data", "edhec_alternative_indices.csv")
TARGETS = ["Long/Short Equity", "Global Macro", "Event Driven"]


def _parse_edhec_download(path):
    """EDHEC's own download: a date column (dd/mm/yyyy) and one column per index, values like '1.19%'."""
    df = pd.read_csv(path, sep=None, engine="python")
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.set_index(date_col)
    df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace("%", "").str.strip(), errors="coerce"))
    if df.abs().stack().median() > 0.2:          # printed in percent
        df = df / 100
    df.index = df.index + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    return df.rename(columns=lambda c: c.strip().replace("Funds of Funds", "Funds Of Funds"))


def load_edhec():
    """Monthly EDHEC indices. Prefers the newest data/edhec_*.csv; falls back to the bundled 1997-2021 copy."""
    user = sorted(glob.glob("data/edhec_*.csv"))
    if user:
        df = _parse_edhec_download(user[-1])
        df.attrs["source"] = user[-1]
    else:
        df = pd.read_csv(REF, parse_dates=["date"], index_col="date")
        df.attrs["source"] = "EDHEC-Risk alternative indices via R PerformanceAnalytics (bundled copy)"
    return df


def _find_cols(df):
    date_col = next(c for c in df.columns if re.search(r"date|quarter|period|as of", str(c), re.I))
    ret_col = next(c for c in df.columns if c != date_col and re.search(r"total|return|cdli", str(c), re.I))
    return date_col, ret_col


def load_private():
    """Quarterly private-market index returns. Returns (Series, label, is_stand_in)."""
    files = sorted(glob.glob("data/private/cdli*.csv") + glob.glob("data/private/cdli*.xlsx"))
    if files:
        path = files[-1]
        df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
        date_col, ret_col = _find_cols(df)
        s = pd.Series(pd.to_numeric(df[ret_col].astype(str).str.replace("%", ""), errors="coerce").values,
                      index=pd.to_datetime(df[date_col]) + pd.offsets.QuarterEnd(0)).dropna().sort_index()
        if s.abs().median() > 0.2:
            s = s / 100
        s.name = "Cliffwater Direct Lending Index"
        return s, "Cliffwater Direct Lending Index (quarterly)", False
    s = load_edhec()["Distressed Securities"].copy()
    s.name = "EDHEC Distressed Securities"
    return s, "EDHEC Distressed Securities (monthly) - stand-in until the CDLI file is added", True
