"""
Generate the investor one-pager (PDF) and its Excel backup in one step.

    python make_onepager.py [path/to/prices.csv]

1. analysis.run()           -> statistics, allocation scenarios, stress tests
2. build_workbook.build()   -> live Excel workbook (prices in, formulas for everything else)
3. LibreOffice recalculation, then every workbook figure is checked against the Python figure
4. build_pdf.build()        -> one-page PDF whose text and tables come from the same numbers
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import date

import openpyxl
import pandas as pd

import analysis
import build_pdf
import build_workbook

RECALC = os.environ.get("RECALC", "/mnt/skills/public/xlsx/scripts/recalc.py")


def recalc(path):
    if not (shutil.which("soffice") or shutil.which("libreoffice")) or not os.path.exists(RECALC):
        print("LibreOffice recalculation not available: Excel will calculate the workbook when it is opened.")
        return False
    out = subprocess.run([sys.executable, RECALC, path, "600"], capture_output=True, text=True)
    res = json.loads(out.stdout)
    if res.get("status") != "success" or res.get("total_errors"):
        raise RuntimeError(f"workbook formulas failed: {res}")
    print(f"Workbook recalculated: {res['total_formulas']:,} formulas, 0 errors")
    return True


def verify(R, path, rows):
    """Every figure the PDF shows must equal the workbook's own formula result."""
    wb = openpyxl.load_workbook(path, data_only=True)
    a, s, p, c = wb["Allocation scenarios"], wb["Stress tests"], wb["Strategy profiles"], wb["Correlations"]
    checks = []
    sc = R["scenarios"].set_index("portfolio")
    for port, r in rows["AROW"].items():
        for col, key in [(8, "annual_return"), (9, "volatility"), (10, "sharpe"), (12, "max_drawdown")]:
            checks.append((f"{port} {key}", a.cell(r, col).value, sc.loc[port, key]))
    st = R["stress"].set_index("portfolio")
    for name, r in rows["SROW"].items():
        for j, k in enumerate(rows["ks"]):
            checks.append((f"stress {name} {k}", s.cell(r, 2 + j).value, st.loc[name, f"{k}_return"]))
    pr = R["strategies"].set_index("ticker")
    for t, r in rows["PROW"].items():
        for col, key in [(4, "annual_return"), (5, "volatility"), (6, "sharpe"), (7, "max_drawdown"),
                         (10, "corr_to_stocks"), (11, "corr_to_bonds"), (12, "corr_to_6040"),
                         (13, "rollvol_median"), (14, "rollvol_max"), (16, "rollvol_latest")]:
            checks.append((f"{t} {key}", p.cell(r, col).value, pr.loc[t, key]))
    for i, t in enumerate(analysis.STRATEGIES):
        for j, k in enumerate(rows["ks"]):
            checks.append((f"stress corr {t} {k}", c.cell(rows["COR_STRESS_ROW0"] + i, 2 + j).value,
                           R["corr_stress"].loc[t, k]))
    bad = [(n, x, y) for n, x, y in checks if x is None or abs(float(x) - float(y)) > 1e-9]
    if bad:
        raise RuntimeError(f"{len(bad)} workbook figures differ from the analysis, e.g. {bad[:3]}")
    print(f"Verified {len(checks)} workbook figures against the analysis (all match to 1e-9)")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    R = analysis.run(src)
    os.makedirs("output", exist_ok=True)
    stem = f"output/bond-strategy-allocation_{date.today():%Y-%m-%d}"
    xlsx, pdf = f"{stem}.xlsx", f"{stem}.pdf"
    rows = build_workbook.build(R, xlsx)
    if recalc(xlsx):
        verify(R, xlsx, rows)
    bottom = build_pdf.build(R, pdf)
    print(f"Wrote {pdf} (bottom margin {bottom:.0f}pt) and {xlsx}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
