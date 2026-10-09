"""
One-page client note in plain language, built from results/multiasset/*.csv.

    python -m multiasset.client_note
"""
import io
import json
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

RES = "results/multiasset"
FONT_DIR = "/usr/share/fonts/truetype/crosextra"
for name, file in [("Body", "Carlito-Regular"), ("Body-Bold", "Carlito-Bold"), ("Body-Italic", "Carlito-Italic"),
                   ("Head", "Caladea-Bold")]:
    pdfmetrics.registerFont(TTFont(name, f"{FONT_DIR}/{file}.ttf"))
registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
for fp in ("Carlito-Regular", "Carlito-Bold"):
    font_manager.fontManager.addfont(f"{FONT_DIR}/{fp}.ttf")

INK, INK2, MUTED = colors.HexColor("#15202b"), colors.HexColor("#4a5562"), colors.HexColor("#7b8590")
NAVY, RULE, TINT = colors.HexColor("#1f3a5f"), colors.HexColor("#d5dbe2"), colors.HexColor("#f2f5f8")
ACCENT = "#2a78d6"

body = ParagraphStyle("b", fontName="Body", fontSize=9.2, leading=11.9, textColor=INK)
small = ParagraphStyle("s", fontName="Body", fontSize=7.1, leading=8.9, textColor=INK2)
cell = ParagraphStyle("c", fontName="Body", fontSize=7.9, leading=9.4, textColor=INK)
hcell = ParagraphStyle("h", fontName="Body-Bold", fontSize=7.2, leading=8.6, textColor=INK2)

# plain-language names and one-line descriptions, in display order
SHOW = [
    ("60/40", "60/40", "60% U.S. stocks, 40% U.S. Treasuries"),
    ("Equal weight", "Equal weight", "10% in each of the ten investments"),
    ("Minimum variance", "Steadiest mix", "the mix with the smallest expected swings"),
    ("Risk parity", "Risk-balanced", "each investment adds the same amount of risk"),
    ("Risk parity + momentum", "Risk-balanced + trend", "risk-balanced, leaning toward what has been rising"),
    ("Mean-variance", "Return-seeking", "tilts toward recent winners, max 30% in any one"),
    ("Risk parity (levered)", "Risk-balanced, borrowed", "risk-balanced, borrowing to match the 60/40's risk"),
]


def P(x, d=0):
    s = f"{abs(x) * 100:.{d}f}%"
    return ("−" + s) if x < 0 and round(abs(x) * 100, d) != 0 else s


def load():
    st = pd.read_csv(f"{RES}/strategic_backtest.csv", index_col=0)
    tac = pd.read_csv(f"{RES}/tactical_overlays.csv", index_col=0)
    tests = pd.read_csv(f"{RES}/sharpe_difference_tests.csv", index_col=0)
    ic = pd.read_csv(f"{RES}/signal_information_coefficients.csv", index_col=0)
    mc = pd.read_csv(f"{RES}/monte_carlo_10y.csv", index_col=0)
    mch = pd.read_csv(f"{RES}/monte_carlo_10y_half_premium.csv", index_col=0)
    meta = json.load(open(f"{RES}/run_info.json"))
    # one table of full-period track records, including the momentum overlay
    rec = st[["annual_return", "max_drawdown", "sharpe"]].copy()
    m = tac.loc["Momentum (TSMOM)"]
    rec.loc["Risk parity + momentum"] = [m["full_annual_return"], m["full_max_drawdown"], m["full_sharpe"]]
    return st, tac, tests, ic, mc, mch, meta, rec


def findings(st, tac, tests, ic, mc, mch, rec):
    best = rec["sharpe"].idxmax()
    others = rec.drop(index=["60/40", "Risk parity + momentum"])["annual_return"]
    f1 = (f"<b>The plain 60/40 mix gave the most return for the risk taken</b>, earning {P(rec.loc['60/40', 'annual_return'], 1)} "
          f"a year. The broader mixes earned {P(others.min(), 1)} to {P(others.max(), 1)}, because U.S. stocks, the 60/40's main "
          f"holding, had an exceptional run.") if best == "60/40" else (
          f"<b>{best} gave the most return for the risk taken.</b>")
    f2 = (f"<b>Spreading money more widely cut the worst falls.</b> The risk-balanced mix fell at most "
          f"{P(-rec.loc['Risk parity', 'max_drawdown'])} and the steadiest mix {P(-rec.loc['Minimum variance', 'max_drawdown'])}, "
          f"against {P(-rec.loc['60/40', 'max_drawdown'])} for the 60/40.")
    t = tests.loc["Momentum (TSMOM) vs no overlay"]
    f3 = (f"<b>A simple trend rule helped a little.</b> Leaning toward investments that had beaten cash over the past year lifted the "
          f"risk-balanced mix from {P(rec.loc['Risk parity', 'annual_return'], 1)} to {P(rec.loc['Risk parity + momentum', 'annual_return'], 1)} "
          f"a year and cut its chance of a 20% fall over 10 years from {P(mc.loc['Risk parity', 'prob_drawdown_over_20'])} to "
          f"{P(mc.loc['Risk parity + momentum', 'prob_drawdown_over_20'])}. The gain is small enough that it could be luck.")
    ml_vs = tests.loc[[i for i in tests.index if i.startswith("Gradient boosting (ML) vs") and "no overlay" not in i][0]]
    beat = tac.loc["Gradient boosting (ML)", "oos_sharpe"] > tac.loc["Risk parity (no overlay)", "oos_sharpe"]
    f4 = (f"<b>Our machine-learning model did not help.</b> On {tac.loc['Gradient boosting (ML)', 'oos_months']:.0f} months it "
          f"never saw in training (2013 to 2026), it did worse than the simple trend rule"
          + ("" if beat else " and worse than making no changes at all")
          + ". Its forecasts were no better than chance, so we would not use it.") if ml_vs["diff"] < 0 else (
          "<b>Our machine-learning model edged out the simple rules</b>, but not by enough to rule out luck.")
    mr = pd.read_csv(f"{RES}/monthly_net_returns.csv", index_col=0, parse_dates=True)
    lev = mr["Risk parity (levered)"]
    w = (1 + lev).cumprod()
    trough = (w / w.cummax() - 1).idxmin()
    y22 = {k: (1 + mr[k].loc["2022"]).prod() - 1 for k in ("Risk parity (levered)", "60/40")}
    f5 = (f"<b>Borrowing to boost the risk-balanced mix backfired.</b> It fell as much as "
          f"{P(-rec.loc['Risk parity (levered)', 'max_drawdown'])} in {trough:%Y}, and lost {P(-y22['Risk parity (levered)'])} in 2022 "
          f"(the 60/40 lost {P(-y22['60/40'])}), when bonds and stocks fell together.")
    return [f1, f2, f3, f4, f5]


def chart(mc, w_in, h_in):
    plt.rcParams.update({"font.family": "Carlito", "font.size": 7.5})
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=300)
    rows = [k for k, _, _ in SHOW]
    names = {k: n for k, n, _ in SHOW}
    for i, k in enumerate(rows):
        lo, med, hi = (mc.loc[k, c] / 1000 for c in ("end_value_p5", "end_value_median", "end_value_p95"))
        ax.plot([lo, hi], [i, i], color=ACCENT, lw=5.5, alpha=0.28, solid_capstyle="round", zorder=2)
        ax.scatter([med], [i], s=26, color=ACCENT, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(med, i - 0.2, f"${med:,.0f}k", ha="center", va="bottom", fontsize=6.7, color="#15202b", fontweight="bold")
        ax.text(lo - 4, i, f"${lo:,.0f}k", ha="right", va="center", fontsize=6.3, color="#4a5562")
        ax.text(hi + 4, i, f"${hi:,.0f}k", ha="left", va="center", fontsize=6.3, color="#4a5562")
    ax.axvline(100, color="#9aa3ad", lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(100, -0.62, " $100k invested", ha="left", va="center", fontsize=6.3, color="#7b8590")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([names[k] for k in rows], fontsize=7.8)
    ax.set_ylim(len(rows) - 0.3, -0.75)
    lo_all = mc.loc[rows, "end_value_p5"].min() / 1000
    hi_all = mc.loc[rows, "end_value_p95"].max() / 1000
    ax.set_xlim(min(80, lo_all - 25), hi_all + 30)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}k"))
    ax.grid(axis="x", color="#e6e9ed", lw=0.6, zorder=0)
    ax.tick_params(axis="x", length=0, colors="#7b8590", labelsize=6.8)
    ax.tick_params(axis="y", length=0, labelcolor="#15202b", labelsize=7.8)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.99, bottom=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


def table(rec, mc):
    head = ["Approach", "What it does", "Return<br/>per year", "Worst<br/>fall", "Return per<br/>unit of risk",
            "Chance of a 20%+<br/>fall in 10 years"]
    data = [[Paragraph(h, hcell) for h in head]]
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ALIGN", (2, 1), (-1, -1), "RIGHT"), ("FONTNAME", (2, 1), (-1, -1), "Body"),
             ("FONTSIZE", (2, 1), (-1, -1), 8), ("TEXTCOLOR", (2, 1), (-1, -1), INK),
             ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE), ("BACKGROUND", (0, 1), (-1, 1), TINT)]
    for k, name, desc in SHOW:
        data.append([Paragraph(f"<b>{name}</b>", cell), Paragraph(desc, cell), P(rec.loc[k, "annual_return"], 1),
                     P(rec.loc[k, "max_drawdown"], 0), f"{rec.loc[k, 'sharpe']:.2f}",
                     P(mc.loc[k, "prob_drawdown_over_20"], 0)])
    t = Table(data, colWidths=[92, 196, 46, 40, 56, 76])
    t.setStyle(TableStyle(style))
    return t


def build(path):
    st, tac, tests, ic, mc, mch, meta, rec = load()
    W, H = letter
    M = 40
    cw = W - 2 * M
    c = canvas.Canvas(path, pagesize=letter)
    c.setTitle("Building a global portfolio: what 18 years of evidence shows")
    c.setAuthor("Jose Thomas")
    end = pd.Timestamp(meta["backtest_end"])
    start = pd.Timestamp(meta["backtest_start"])
    years = meta["months"] / 12

    c.setFillColor(NAVY)
    c.rect(0, H - 8, W, 8, stroke=0, fill=1)
    y = H - 33
    c.setFont("Body-Bold", 7.5)
    c.setFillColor(MUTED)
    c.drawString(M, y, f"CLIENT NOTE  ·  {date.today():%B %Y}".upper())
    c.drawRightString(W - M, y, f"TESTED {start:%B %Y} – {end:%B %Y}".upper())
    y -= 25
    c.setFont("Head", 20)
    c.setFillColor(INK)
    c.drawString(M, y, f"Building a global portfolio: what {years:.0f} years of evidence shows")
    deck = Paragraph(
        "We tested six ways to combine ten global investments (U.S., developed and emerging-market stocks; Treasuries, "
        "corporate and inflation-protected bonds; commodities, gold and property) as if run month by month since 2008, "
        "using only what was known at the time and paying trading costs. We then asked whether market signals or a "
        "machine-learning model could improve on them, and simulated 10,000 possible decades.",
        ParagraphStyle("d", parent=body, fontSize=9.8, leading=12.8, textColor=INK2))
    _, h = deck.wrap(cw, 100)
    y -= 6 + h
    deck.drawOn(c, M, y)

    y -= 16
    c.setFont("Body-Bold", 8)
    c.setFillColor(NAVY)
    c.drawString(M, y, "WHAT WE FOUND")
    y -= 3
    for txt in findings(st, tac, tests, ic, mc, mch, rec):
        p = Paragraph(txt, body)
        _, h = p.wrap(cw - 13, 200)
        y -= h + 3
        c.setFillColor(NAVY)
        c.circle(M + 3, y + h - 4.4, 1.7, stroke=0, fill=1)
        p.drawOn(c, M + 12, y)

    y -= 17
    c.setFont("Body-Bold", 8)
    c.setFillColor(NAVY)
    c.drawString(M, y, "WHAT $100,000 COULD BECOME OVER 10 YEARS")
    c.setFont("Body", 7.3)
    c.setFillColor(INK2)
    c.drawString(M, y - 11, "Dot: middle outcome. Bar: 90 of 100 simulated decades fall inside it. "
                            "Simulated by replaying blocks of the 2008–2026 record in random order.")
    ch = 190
    img = chart(mc, cw / 72, ch / 72)
    y -= 15 + ch
    c.drawImage(img, M, y, width=cw, height=ch)

    y -= 16
    c.setFont("Body-Bold", 8)
    c.setFillColor(NAVY)
    c.drawString(M, y, f"TRACK RECORD, {start:%B %Y} TO {end:%B %Y}, AFTER TRADING COSTS".upper())
    t = table(rec, mc)
    tw, th = t.wrap(0, 0)
    y -= 5 + th
    t.drawOn(c, M, y)

    y -= 13
    half = mch.loc["60/40"]
    what = Paragraph(
        "<b>What this means for you.</b> Clever signals mattered far less than the basic mix. The right mix depends mostly "
        "on how large a temporary fall you can sit through without changing course. If a fall of "
        f"{P(-rec.loc['60/40', 'max_drawdown'])} would not change your plans, the 60/40 rewarded patience. If it would, the "
        f"risk-balanced mix with a trend tilt gave up about {abs(rec.loc['60/40', 'annual_return'] - rec.loc['Risk parity + momentum', 'annual_return'])*100:.0f} "
        f"points of return a year for a worst fall of {P(-rec.loc['Risk parity + momentum', 'max_drawdown'])}. "
        "The next decade may be less generous than the last: if every approach earns only half its past return above cash, "
        f"the 60/40's middle outcome drops to {P(half['median_annual_return'], 1)} a year, with a {P(half['prob_loss_10y'])} "
        "chance of ending the decade below where it started.", body)
    _, h = what.wrap(cw, 200)
    y -= h
    what.drawOn(c, M, y)

    y -= 10
    notes = (
        "<b>How it was tested.</b> Each month-end, every approach set its mix from up to three years of daily prices "
        "(ten exchange-traded funds: SPY, EFA, EEM, IEF, LQD, HYG, TIP, DBC, GLD, VNQ; cash = BIL), then held it for a month. "
        "Trading costs of 0.02% to 0.10% per trade are deducted; doubling them changes no conclusion. The machine-learning model "
        "(gradient boosting) was retrained every January on past data only, with a gap between training and testing periods "
        "so it could not peek at the answers. <i>Return per unit of risk</i> is the Sharpe ratio. Data: Yahoo Finance, "
        f"dividends reinvested, through {pd.Timestamp(meta['last_price_date']):%B} {pd.Timestamp(meta['last_price_date']).day}, {pd.Timestamp(meta['last_price_date']).year}. "
        "<b>Important.</b> Past results and simulations do not predict future returns, and the period tested was unusually "
        "good for U.S. stocks. This note is for discussion and is not a recommendation for any individual.")
    p = Paragraph(notes, small)
    _, h = p.wrap(cw, 200)
    y -= h
    p.drawOn(c, M, y)
    if y < 26:
        raise RuntimeError(f"client note overflowed the page (bottom at {y:.0f}pt)")
    c.setFont("Body", 6.5)
    c.setFillColor(MUTED)
    c.drawString(M, 20, f"Generated automatically from the backtest results on {date.today():%Y-%m-%d}.")
    c.drawRightString(W - M, 20, "Page 1 of 1")
    c.showPage()
    c.save()
    return y


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    out = f"output/client-note_global-multi-asset_{date.today():%Y-%m-%d}.pdf"
    print("bottom", build(out), out)
