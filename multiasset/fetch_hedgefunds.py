"""
Fetch the EDHEC-Risk alternative (hedge fund style) indices and write them as a plain CSV.

    python -m multiasset.fetch_hedgefunds        # -> multiasset/reference_data/edhec_alternative_indices.csv

Source: the `edhec` dataset distributed, with EDHEC-Risk's permission, in the R package
PerformanceAnalytics (github.com/braverock/PerformanceAnalytics, data/edhec.rda). 13 strategy
indices, monthly returns January 1997 - May 2021. The file is an R xts object; the month-end
dates are read from its `index` attribute.

The download is cross-checked against an independent copy published by the R-Finance
PortfolioAnalytics project (monthly returns to July 2013, printed to 0.01%): every
overlapping value must agree to within rounding, or the script stops.

For later months, download the current series from EDHEC (climateimpact.edhec.edu, free
registration) and save it as data/edhec_*.csv; hedgefunds.load() prefers that file.
"""
import io
import os
import urllib.request

import numpy as np
import pandas as pd

RDA_URL = "https://raw.githubusercontent.com/braverock/PerformanceAnalytics/master/data/edhec.rda"
CHECK_URL = ("https://raw.githubusercontent.com/R-Finance/PortfolioAnalytics/master/"
             "sandbox/symposium2013/data/EDHEC-index-history.csv")
OUT = os.path.join(os.path.dirname(__file__), "reference_data", "edhec_alternative_indices.csv")


def _xts_index(obj):
    """Walk an R object's attribute pairlist and return the 'index' attribute (seconds since epoch)."""
    attrs = obj.attributes
    while attrs is not None and getattr(attrs, "info", None) is not None and attrs.info.type.name == "LIST":
        tag = attrs.tag
        name = tag.value.value.decode() if tag is not None and hasattr(tag.value, "value") else None
        if name == "index":
            return np.asarray(attrs.value[0].value)
        attrs = attrs.value[1] if len(attrs.value) > 1 else None
    raise ValueError("no xts index attribute found")


def fetch():
    import rdata   # pip install rdata

    raw = urllib.request.urlopen(RDA_URL, timeout=60).read()
    parsed = rdata.parser.parse_data(raw)
    arr = rdata.conversion.convert(parsed, default_encoding="utf-8")["edhec"]
    dates = pd.to_datetime(_xts_index(parsed.object.value[0]), unit="s").normalize()
    df = pd.DataFrame(np.asarray(arr), index=dates, columns=[str(c) for c in arr.coords["dim_1"].values])
    df.index.name = "date"
    df = df.rename(columns={"Funds of Funds": "Funds Of Funds"})

    chk = pd.read_csv(io.BytesIO(urllib.request.urlopen(CHECK_URL, timeout=60).read()), sep=";")
    chk["date"] = pd.to_datetime(chk["date"], format="%d/%m/%Y")
    chk = chk.set_index("date").apply(lambda s: s.str.rstrip("%").astype(float) / 100)
    common = df.index.intersection(chk.index)
    diff = (df.loc[common, chk.columns] - chk.loc[common]).abs().max().max()
    if diff > 0.00006:
        raise SystemExit(f"EDHEC download disagrees with the reference copy by {diff:.5f}; not writing")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, float_format="%.4f")
    print(f"wrote {OUT}: {len(df)} months {df.index[0]:%Y-%m} to {df.index[-1]:%Y-%m}, {df.shape[1]} indices; "
          f"matches the reference copy on {len(common)} overlapping months (max diff {diff:.6f})")
    return df


if __name__ == "__main__":
    fetch()
