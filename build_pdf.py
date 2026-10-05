"""One-page investor PDF, written for a non-technical reader. All text is generated from the numbers."""
import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from analysis import STRATEGIES, STRESS

FONT_DIR = "/usr/share/fonts/truetype/crosextra"
for name, file in [("Body", "Carlito-Regular"), ("Body-Bold", "Carlito-Bold"), ("Body-Italic", "Carlito-Italic"),
                   ("Head", "Caladea-Bold"), ("Head-Reg", "Caladea-Regular")]:
    pdfmetrics.registerFont(TTFont(name, f"{FONT_DIR}/{file}.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402
registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
for fp in ("Carlito-Regular", "Carlito-Bold"):
    font_manager.fontManager.addfont(f"{FONT_DIR}/{fp}.ttf")

INK, INK2, MUTED = colors.HexColor("#15202b"), colors.HexColor("#4a5562"), colors.HexColor("#7b8590")
NAVY, RULE, TINT = colors.HexColor("#1f3a5f"), colors.HexColor("#d5dbe2"), colors.HexColor("#f2f5f8")
SERIES = {"60/40": "#5b6470", "TLT": "#2a78d6", "LQD": "#eb6834", "HYG": "#1baf7a", "TIP": "#eda100"}

body = ParagraphStyle("b", fontName="Body", fontSize=9.4, leading=12.2, textColor=INK)
small = ParagraphStyle("s", fontName="Body", fontSize=7.3, leading=9.2, textColor=INK2)
cell = ParagraphStyle("c", fontName="Body", fontSize=8, leading=9.6, textColor=INK)
cellb = ParagraphStyle("cb", parent=cell, fontName="Body-Bold")
hcell = ParagraphStyle("h", fontName="Body-Bold", fontSize=7.4, leading=8.8, textColor=INK2)


def pct(x, d=0, sign=False):
    s = f"{abs(x)*100:.{d}f}%"
    if x < 0 and round(abs(x) * 100, d) != 0:
        return "−" + s
    return ("+" + s) if sign and x > 0 else s


def loss(x):
    """'a 36% loss' style figure for prose (x is a negative return)."""
    return f"{abs(x)*100:.0f}%"


def _v(name, singular, plural):
    """Verb agreement: 'Long-term Treasuries move' vs 'High-yield credit moves'."""
    return plural if name.split()[-1].endswith("s") else singular


def takeaways(R):
    sc = R["scenarios"].set_index("portfolio")
    st = R["stress"].set_index("portfolio")
    prof = R["strategies"].set_index("ticker")
    ks = list(STRESS)
    base = st.loc["60/40"]
    imp = {t: {k: st.loc[f"20% {t}", f"{k}_return"] - base[f"{k}_return"] for k in ks} for t in STRATEGIES}

    crash = max(STRATEGIES, key=lambda t: imp[t]["2008"] + imp[t]["2020"])
    allthree = [t for t in STRATEGIES if all(imp[t][k] > 0.0005 for k in ks)]
    steady = max([t for t in allthree if t != crash] or [t for t in STRATEGIES if t != crash],
                 key=lambda t: min(imp[t].values()))
    weak = min(STRATEGIES, key=lambda t: sc.loc[f"20% {t}", "mdd_change"])

    b = {k: base[f"{k}_return"] for k in ks}
    x = {t: {k: st.loc[f"20% {t}", f"{k}_return"] for k in ks} for t in STRATEGIES}
    s0 = sc.loc["60/40", "sharpe"]

    n1 = STRATEGIES[crash][0]
    t1 = (f"<b>{n1} protected best when stocks crashed.</b> A 20% slice cut the 2008 loss from "
          f"{loss(b['2008'])} to {loss(x[crash]['2008'])} and the 2020 loss from {loss(b['2020'])} to {loss(x[crash]['2020'])}.")
    if imp[crash]["2022"] < 0:
        t1 += (f" {_v(n1, 'It', 'They')} made 2022 worse: a {loss(x[crash]['2022'])} loss instead of {loss(b['2022'])}, "
               f"because rising interest rates pushed stocks and bonds down together.")

    n2 = STRATEGIES[steady][0]
    only = "the only strategy that" if allthree == [steady] else "a strategy that"
    t2 = (f"<b>{n2} {_v(n2, 'was', 'were')} {only} softened all three crises:</b> losses of {loss(x[steady]['2008'])}, "
          f"{loss(x[steady]['2020'])} and {loss(x[steady]['2022'])} instead of {loss(b['2008'])}, {loss(b['2020'])} "
          f"and {loss(b['2022'])}. Return per unit of risk rose from {s0:.2f} to {sc.loc[f'20% {steady}', 'sharpe']:.2f}.")

    n3 = STRATEGIES[weak][0]
    sw = sc.loc[f"20% {weak}", "sharpe"]
    t3 = (f"<b>{n3} {_v(n3, 'moves', 'move')} with stocks</b> (correlation {prof.loc[weak, 'corr_to_6040']:.2f} with the 60/40). "
          f"A 20% slice barely changed the worst loss ({loss(sc.loc['60/40', 'max_drawdown'])} to "
          f"{loss(sc.loc[f'20% {weak}', 'max_drawdown'])}) and "
          + (f"lowered return per unit of risk ({s0:.2f} to {sw:.2f})." if sw < s0 - 0.005 else "left return per unit of risk about the same.")
          + " In a portfolio it acts more like extra stocks than extra protection.")

    r0 = sc.loc["60/40", "annual_return"]
    blends = sc.drop(index="60/40")["annual_return"]
    spy = prof.loc["SPY", "annual_return"]
    if (blends < r0).all():
        t4 = (f"<b>The trade-off:</b> every blend earned a little less per year than the plain 60/40 "
              f"({pct(blends.min(), 1)} to {pct(blends.max(), 1)}, against {pct(r0, 1)}), because U.S. stocks returned "
              f"{pct(spy, 1)} a year over this period. The strategies bought smaller losses, not higher returns.")
    else:
        t4 = (f"<b>Returns:</b> blends earned {pct(blends.min(), 1)} to {pct(blends.max(), 1)} a year, "
              f"against {pct(r0, 1)} for the plain 60/40.")
    return [t1, t2, t3, t4]


LEGEND = [("60/40", "60/40 as it is today")] + [(t, f"+20% {v[0]}") for t, v in STRATEGIES.items()]


def chart(R, w_in, h_in):
    """Horizontal grouped bars: one group per crisis, one bar per portfolio, value at each bar's tip."""
    st = R["stress"].set_index("portfolio")
    ks = list(STRESS)
    rows = ["60/40"] + [f"20% {t}" for t in STRATEGIES]
    names = {"2008": "2008 financial crisis", "2020": "2020 COVID crash", "2022": "2022 rate hikes"}
    plt.rcParams.update({"font.family": "Carlito", "font.size": 7.5})
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=300)
    gap, y, yticks = 2.2, 0.0, []
    lo = min(st.loc[rows, [f"{k}_return" for k in ks]].min()) * 100
    for g, k in enumerate(ks):
        ax.text(lo * 1.16, y - 1.15, names[k], ha="left", va="center", fontsize=7.6, fontweight="bold", color="#15202b")
        for p in rows:
            key = "60/40" if p == "60/40" else p.split()[-1]
            v = st.loc[p, f"{k}_return"] * 100
            ax.barh(y, v, height=0.78, color=SERIES[key], zorder=3)
            ax.text(v - 0.5, y, f"{v:.0f}%".replace("-", "\u2212"), ha="right", va="center", fontsize=6.6,
                    color="#15202b", fontweight="bold" if p == "60/40" else "normal")
            y += 1
        y += gap
    ax.set_ylim(y - gap - 0.4, -1.6)
    ax.set_xlim(lo * 1.16, 0.5)
    ax.set_yticks([])
    ax.axvline(0, color="#9aa3ad", lw=0.8, zorder=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%".replace("-", "\u2212")))
    ax.grid(axis="x", color="#e6e9ed", lw=0.6, zorder=0)
    ax.tick_params(axis="both", length=0, colors="#7b8590", labelsize=6.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.07)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


def draw_legend(c, x, y, width):
    """Legend drawn in the PDF (text in ink, colored swatch beside it), wrapping to new lines."""
    c.setFont("Body", 7.2)
    cx, cy = x, y
    for key, lab in LEGEND:
        w = 10 + c.stringWidth(lab, "Body", 7.2) + 10
        if cx + w > x + width:
            cx, cy = x, cy - 10
        c.setFillColor(colors.HexColor(SERIES[key]))
        c.rect(cx, cy - 1, 6.5, 6.5, stroke=0, fill=1)
        c.setFillColor(INK2)
        c.drawString(cx + 9.5, cy, lab)
        cx += w
    return cy


def main_table(R):
    sc = R["scenarios"].set_index("portfolio")
    st = R["stress"].set_index("portfolio")
    ks = list(STRESS)
    head = ["Portfolio", "Return\nper year", "Ups and downs\n(volatility)", "Worst loss\n(max drawdown)",
            "Return per unit\nof risk (Sharpe)", *[f"{k}\ncrisis" for k in ks]]
    data = [[Paragraph(h.replace("\n", "<br/>"), hcell) for h in head]]
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ALIGN", (1, 1), (-1, -1), "RIGHT"), ("FONTNAME", (1, 1), (-1, -1), "Body"),
             ("FONTSIZE", (1, 1), (-1, -1), 8), ("TEXTCOLOR", (1, 1), (-1, -1), INK),
             ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("LINEBEFORE", (5, 0), (5, -1), 0.5, RULE)]
    for i, p in enumerate(sc.index):
        r = i + 1
        if p == "60/40":
            lab = Paragraph("<b>60/40 as it is today</b>", cell)
            style += [("BACKGROUND", (0, r), (-1, r), TINT), ("FONTNAME", (1, r), (-1, r), "Body-Bold")]
        else:
            t = p.split()[-1]
            lab = Paragraph(f"<font color='{SERIES[t]}'>■</font> + {p.split()[0]} {STRATEGIES[t][0]} <font color='#7b8590'>({t})</font>", cell)
        if p.startswith("20%"):
            style.append(("LINEBELOW", (0, r), (-1, r), 0.4, RULE))
        data.append([lab, pct(sc.loc[p, "annual_return"], 1), pct(sc.loc[p, "volatility"], 1),
                     pct(sc.loc[p, "max_drawdown"], 1), f"{sc.loc[p, 'sharpe']:.2f}",
                     *[pct(st.loc[p, f"{k}_return"], 1) for k in ks]])
    t = Table(data, colWidths=[176, 46, 58, 60, 64, 44, 44, 44])
    t.setStyle(TableStyle(style))
    return t


def profile_table(R):
    pr = R["strategies"].set_index("ticker")
    head = ["Strategy on its own", "Return<br/>per year", "Worst<br/>loss", "Moves with<br/>the 60/40?"]
    data = [[Paragraph(h, hcell) for h in head]]

    def mood(c):
        return "Opposite" if c < -0.1 else "Unrelated" if c < 0.2 else "Somewhat" if c < 0.5 else "Closely"
    for t in STRATEGIES:
        c = pr.loc[t, "corr_to_6040"]
        data.append([Paragraph(f"<font color='{SERIES[t]}'>■</font> <b>{STRATEGIES[t][0]}</b><br/>"
                               f"<font color='#7b8590' size='6.8'>{t}: {STRATEGIES[t][1]}</font>", cell),
                     pct(pr.loc[t, "annual_return"], 1), pct(pr.loc[t, "max_drawdown"], 0),
                     Paragraph(f"{mood(c)} <font color='#7b8590'>({c:+.2f})</font>".replace("-", "−"), cell)])
    t = Table(data, colWidths=[124, 36, 32, 58])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("ALIGN", (1, 1), (2, -1), "RIGHT"), ("FONTNAME", (1, 1), (-1, -1), "Body"),
                           ("FONTSIZE", (1, 1), (-1, -1), 8), ("TEXTCOLOR", (1, 1), (-1, -1), INK),
                           ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def build(R, path):
    W, H = letter
    M = 40
    cw = W - 2 * M
    m = R["meta"]
    c = canvas.Canvas(path, pagesize=letter)
    c.setTitle("Adding a bond strategy to a 60/40 portfolio")
    c.setAuthor("Jose Thomas")
    c.setSubject("Investor one-pager: bond and credit strategies as a 10% or 20% allocation")

    # header band
    c.setFillColor(NAVY)
    c.rect(0, H - 8, W, 8, stroke=0, fill=1)
    y = H - 34
    c.setFont("Body-Bold", 7.5)
    c.setFillColor(MUTED)
    c.drawString(M, y, f"INVESTOR BRIEF  ·  {date.today():%B %Y}".upper())
    c.drawRightString(W - M, y, f"DATA: JUNE 2007 – {m['end']:%B %Y}".upper())
    y -= 26
    c.setFont("Head", 21)
    c.setFillColor(INK)
    c.drawString(M, y, "Adding a bond strategy to a 60/40 portfolio")
    y -= 6
    deck = Paragraph(
        "We tested four bond and credit strategies as a 10% or 20% slice of a typical portfolio of 60% stocks and "
        f"40% bonds, over {m['years']:.0f} years that include three market crises. Each strategy is represented by "
        "a public exchange-traded fund, so every number can be checked.",
        ParagraphStyle("d", parent=body, fontSize=10.2, leading=13.4, textColor=INK2))
    _, h = deck.wrap(cw, 100)
    y -= h
    deck.drawOn(c, M, y)

    # short answer
    y -= 15
    c.setFont("Body-Bold", 8)
    c.setFillColor(NAVY)
    c.drawString(M, y, "THE SHORT ANSWER")
    y -= 4
    for txt in takeaways(R):
        p = Paragraph(txt, body)
        _, h = p.wrap(cw - 14, 200)
        y -= h + 3
        c.setFillColor(NAVY)
        c.circle(M + 3, y + h - 4.6, 1.8, stroke=0, fill=1)
        p.drawOn(c, M + 12, y)

    # chart + profile table
    y -= 17
    top = y
    c.setFont("Body-Bold", 8)
    c.setFillColor(NAVY)
    c.drawString(M, top, "IN A CRISIS: HOW MUCH THE PORTFOLIO LOST")
    pt = profile_table(R)
    pw, ph = pt.wrap(0, 0)
    c.drawString(W - M - pw, top, "EACH STRATEGY ON ITS OWN")
    chart_w = cw - pw - 24
    ly = draw_legend(c, M, top - 13, chart_w)
    chart_h = 160
    img = chart(R, chart_w / 72, chart_h / 72)
    c.drawImage(img, M, ly - 6 - chart_h, width=chart_w, height=chart_h)
    pt.drawOn(c, W - M - pw, top - 8 - ph)
    cap = Paragraph("Fall in value from the stock market's high to its low in each crisis "
                    "(for 2022, January to October). Each blend puts 20% in the strategy.", small)
    _, chh = cap.wrap(chart_w, 50)
    y = min(ly - 6 - chart_h, top - 8 - ph) - 3
    cap.drawOn(c, M, y - chh)
    y -= chh + 16

    # main table
    c.setFont("Body-Bold", 8)
    c.setFillColor(NAVY)
    c.drawString(M, y, f"THE FULL RECORD, JUNE 2007 TO {m['end']:%B %Y}".upper())
    mt = main_table(R)
    mw, mh = mt.wrap(0, 0)
    y -= 6 + mh
    mt.drawOn(c, M, y)

    # how to read + method
    y -= 14
    notes = [
        "<b>How to read this.</b> <i>Return per year</i> is the average yearly growth, with income reinvested. "
        "<i>Ups and downs</i> measures how much the value swings in a typical year; lower is calmer. "
        "<i>Worst loss</i> is the largest fall from a previous high before recovering. "
        "<i>Return per unit of risk</i> (the Sharpe ratio) is the return earned above cash for each unit of ups and downs; higher is better. "
        "<i>Moves with the 60/40</i> describes the correlation of daily returns: +1 means always moving together, −1 always opposite.",
        "<b>How it was built.</b> Stocks are the SPDR S&amp;P 500 ETF (SPY); bonds are the iShares Core U.S. Aggregate Bond ETF (AGG); "
        "cash is the SPDR 1-3 Month T-Bill ETF (BIL). A 10% slice is funded proportionally, so the portfolio becomes 54% stocks, "
        "36% bonds and 10% strategy. Portfolios are rebalanced monthly. Daily dividend-adjusted prices from Yahoo Finance, "
        "checked against an independent copy. Crisis windows: Oct 9 2007 – Mar 9 2009, Feb 19 – Mar 23 2020, Jan 3 – Oct 12 2022. "
        "Full calculations are in the accompanying Excel workbook.",
        "<b>Important.</b> Past results do not predict future returns. ETFs are stand-ins and differ from any actual strategy's "
        "holdings, fees and trading. In 2008 some bond ETFs briefly traded below the value of their holdings, which makes their crisis "
        "losses look larger than the underlying bonds'. This brief is for discussion and is not investment advice.",
    ]
    for txt in notes:
        p = Paragraph(txt, small)
        _, h = p.wrap(cw, 200)
        y -= h
        p.drawOn(c, M, y)
        y -= 4
    if y < 24:
        raise RuntimeError(f"one-pager overflowed the page (bottom at {y:.0f}pt)")
    c.setFont("Body", 6.5)
    c.setFillColor(MUTED)
    c.drawString(M, 22, f"Generated automatically on {date.today():%Y-%m-%d} from {m['days']:,} trading days of data.")
    c.drawRightString(W - M, 22, "Page 1 of 1")
    c.showPage()
    c.save()
    return y
