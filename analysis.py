"""
Strategy allocation analysis using public ETF return series as stand-ins.

Inputs : data/etf_prices_*.csv  (ticker, date, close, adj_close; dividend-adjusted daily closes)
Outputs: a dict of tidy DataFrames consumed by make_onepager.py (PDF + Excel).

Conventions
- Returns are daily total returns from adjusted closes.
- Base portfolio: 60% SPY / 40% AGG. A strategy allocation of x is funded pro rata,
  so a 10% blend is 54% SPY / 36% AGG / 10% strategy. All portfolios rebalance to
  target weights at each month-end and drift with the market in between.
- Sharpe ratio: annualized mean daily return in excess of T-bills (BIL), divided by
  annualized volatility (sqrt(252)).
- Max drawdown: largest peak-to-trough fall in the daily value of the portfolio.
"""
import glob
import numpy as np
import pandas as pd

STRATEGIES = {  # ticker: (plain-language name, short description for the reader)
    "TLT": ("Long-term Treasuries", "U.S. government bonds, 20+ years to maturity"),
    "LQD": ("Investment-grade credit", "bonds of large, highly rated U.S. companies"),
    "HYG": ("High-yield credit", "bonds of lower-rated U.S. companies"),
    "TIP": ("Inflation-protected bonds", "U.S. Treasuries indexed to inflation (TIPS)"),
}
EQUITY, BONDS, CASH = "SPY", "AGG", "BIL"
ALLOCS = (0.0, 0.10, 0.20)
START = "2007-06-01"  # first month with every series, including T-bills (BIL from 2007-05-30)

STRESS = {  # name: (start close, end close, what happened)
    "2008": ("2007-10-09", "2009-03-09", "Global financial crisis: stock market peak to trough"),
    "2020": ("2020-02-19", "2020-03-23", "COVID crash: stock market peak to trough"),
    "2022": ("2022-01-03", "2022-10-12", "Inflation and rate hikes: stocks and bonds fell together"),
}
ROLL = 63  # trading days in the rolling-volatility window (about 3 months)


def load_prices(path=None):
    path = path or sorted(glob.glob("data/etf_prices_*.csv"))[-1]
    df = pd.read_csv(path, parse_dates=["date"])
    px = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    need = [EQUITY, BONDS, CASH, *STRATEGIES]
    px = px.loc[px.index >= pd.Timestamp(START) - pd.Timedelta(days=7), need]
    px = px.loc[px[need].notna().all(axis=1)]
    return px, path


def portfolio_returns(rets, weights):
    """Daily returns of a portfolio rebalanced to `weights` at each month-end."""
    w_target = pd.Series(weights)
    rets = rets[w_target.index]
    month = rets.index.to_period("M")
    out = np.empty(len(rets))
    w = w_target.values.copy()
    for i, (r, m) in enumerate(zip(rets.values, month)):
        if i > 0 and m != month[i - 1]:
            w = w_target.values.copy()          # rebalance at the start of each new month
        port = float(w @ r)
        out[i] = port
        w = w * (1 + r) / (1 + port)            # drift
    return pd.Series(out, index=rets.index)


def drawdown(r):
    v = (1 + r).cumprod()
    return v / v.cummax() - 1


def stats(r, rf):
    ex = r - rf
    vol = r.std() * np.sqrt(252)
    years = len(r) / 252
    dd = drawdown(r)
    trough = dd.idxmin()
    peak = (1 + r).cumprod().loc[:trough].idxmax()
    return {
        "annual_return": (1 + r).prod() ** (1 / years) - 1,
        "volatility": vol,
        "sharpe": ex.mean() * 252 / vol,
        "max_drawdown": dd.min(),
        "mdd_peak": peak.date(),
        "mdd_trough": trough.date(),
    }


def window_return(r, start, end):
    """Cumulative return from the close on `start` to the close on `end`."""
    sub = r.loc[(r.index > pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
    v = np.concatenate([[1.0], (1 + sub.values).cumprod()])   # include the starting value as a peak
    return v[-1] - 1, (v / np.maximum.accumulate(v) - 1).min()


def run(path=None):
    px, src = load_prices(path)
    rets = px.pct_change().dropna()
    rets = rets.loc[START:]
    rf = rets[CASH]

    base_w = {EQUITY: 0.6, BONDS: 0.4}
    ports = {"60/40": portfolio_returns(rets, base_w)}
    for t in STRATEGIES:
        for a in ALLOCS[1:]:
            w = {EQUITY: 0.6 * (1 - a), BONDS: 0.4 * (1 - a), t: a}
            ports[f"{int(a*100)}% {t}"] = portfolio_returns(rets, w)
    ports = pd.DataFrame(ports)

    # 1) strategy-level statistics
    rows = []
    for t, (name, desc) in STRATEGIES.items():
        s = stats(rets[t], rf)
        rv = rets[t].rolling(ROLL).std() * np.sqrt(252)
        rows.append({"ticker": t, "strategy": name, "what_it_holds": desc, **s,
                     "corr_to_stocks": rets[t].corr(rets[EQUITY]),
                     "corr_to_bonds": rets[t].corr(rets[BONDS]),
                     "corr_to_6040": rets[t].corr(ports["60/40"]),
                     "rollvol_median": rv.median(), "rollvol_max": rv.max(),
                     "rollvol_max_date": rv.idxmax().date(), "rollvol_latest": rv.iloc[-1]})
    for t, name in ((EQUITY, "U.S. stocks (S&P 500)"), (BONDS, "Core U.S. bonds")):
        s = stats(rets[t], rf)
        rv = rets[t].rolling(ROLL).std() * np.sqrt(252)
        rows.append({"ticker": t, "strategy": name, "what_it_holds": "60/40 building block", **s,
                     "corr_to_stocks": rets[t].corr(rets[EQUITY]), "corr_to_bonds": rets[t].corr(rets[BONDS]),
                     "corr_to_6040": rets[t].corr(ports["60/40"]),
                     "rollvol_median": rv.median(), "rollvol_max": rv.max(),
                     "rollvol_max_date": rv.idxmax().date(), "rollvol_latest": rv.iloc[-1]})
    strat = pd.DataFrame(rows)

    corr = rets[[EQUITY, BONDS, *STRATEGIES]].corr()
    # correlation in each stress window (does diversification hold when it matters?)
    corr_stress = {}
    for k, (a, b, _) in STRESS.items():
        sub = rets.loc[(rets.index > a) & (rets.index <= b)]
        corr_stress[k] = sub[list(STRATEGIES)].corrwith(sub[EQUITY])
    corr_stress = pd.DataFrame(corr_stress)

    # 2) allocation scenarios
    sc = []
    base = stats(ports["60/40"], rf)
    for col in ports:
        s = stats(ports[col], rf)
        t = col.split()[-1] if col != "60/40" else None
        a = 0 if col == "60/40" else int(col.split("%")[0]) / 100
        sc.append({"portfolio": col, "strategy": STRATEGIES[t][0] if t else "None (base case)", "ticker": t or "",
                   "allocation": a, "weight_SPY": 0.6 * (1 - a), "weight_AGG": 0.4 * (1 - a), **s,
                   "sharpe_change": s["sharpe"] - base["sharpe"],
                   "mdd_change": s["max_drawdown"] - base["max_drawdown"]})
    scen = pd.DataFrame(sc)

    # 3) stress tests
    st = []
    for col in ports:
        row = {"portfolio": col}
        for k, (a, b, _) in STRESS.items():
            row[f"{k}_return"], row[f"{k}_max_dd"] = window_return(ports[col], a, b)
        st.append(row)
    for t in [EQUITY, BONDS, *STRATEGIES]:
        row = {"portfolio": t}
        for k, (a, b, _) in STRESS.items():
            row[f"{k}_return"], row[f"{k}_max_dd"] = window_return(rets[t], a, b)
        st.append(row)
    stress = pd.DataFrame(st)

    # time series for charts / workbook
    rollvol = (rets[[EQUITY, BONDS, *STRATEGIES]].rolling(ROLL).std() * np.sqrt(252)).dropna()
    dds = ports.apply(drawdown)
    growth = (1 + ports).cumprod()
    monthly_px = px.loc[START:].resample("ME").last()

    base_day = px.index[px.index < pd.Timestamp(START)][-1]
    prices_daily = px.loc[base_day:]
    meta = {"source_file": src, "start": rets.index[0].date(), "end": rets.index[-1].date(),
            "days": len(rets), "years": len(rets) / 252}
    return dict(meta=meta, strategies=strat, corr=corr, corr_stress=corr_stress, scenarios=scen,
                stress=stress, rollvol=rollvol, drawdowns=dds, growth=growth, daily_returns=rets,
                port_returns=ports, monthly_prices=monthly_px, prices_daily=prices_daily)


if __name__ == "__main__":
    R = run()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(R["meta"])
    print(R["strategies"].drop(columns="what_it_holds").round(3).to_string())
    print(R["corr"].round(2))
    print(R["corr_stress"].round(2))
    print(R["scenarios"].drop(columns=["weight_SPY", "weight_AGG"]).round(3).to_string())
    print(R["stress"].round(3).to_string())
