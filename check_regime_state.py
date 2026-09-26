# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_regime_state.py
=====================
The session-16 walk-forward found the book's edge concentrates in ~Dec-Apr
(calendar slices S2, S5) and is weak Aug-Dec / Apr-Aug (S1, S4, S6).  Only 2
seasonal cycles -- could be a real market-STATE effect (the flow trigger is a
continuation signal, so it needs a trending tape), or 2-sample luck.

This re-cuts the deployed-spec book P&L by CONTEMPORANEOUS, lookahead-free
market-state metrics (all as-of the PRIOR session close, i.e. what a morning
throttle would know) instead of by calendar month:

  er_1d       SPY prior-day intraday Kaufman efficiency ratio |C-O| / sum|dC|
  er_10d      SPY 10-day directional-persistence ratio |ret_10d| / sum|daily ret|
  rv_20d      SPY 20-day realized vol (annualised), prior day
  drv_5d      5-day change in rv_20d  (vol rising vs falling)
  absret_1d   |SPY prior-day close-to-close return|
  gap         |SPY today's open / prior close - 1|
  gex_sign    SPY prior-day net_gex sign  (POSITIVE = pinned / NEGATIVE = trend)
  vix / dvix  prior-day VIX close and 5-day change   (if the endpoint reaches back)
  month

Then:
  1. quintile P&L by each metric (IS/OOS + the 6 slices)
  2. does the metric EXPLAIN the calendar?  -- P&L per slice WITHIN the
     favourable vs unfavourable metric bucket.  If the favourable-bucket P&L is
     flat across S1..S6, the "seasonality" was really this metric.
  3. metric mean by slice -- is the metric systematically higher in S2/S5?
  4. a size-throttle sim (full size in the favourable bucket, half otherwise)
     on the blended book -- does it cut slice variance without killing the mean?

Usage:  python check_regime_state.py  [--tickers ...]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

from check_config_walkforward import _flow_for, _slice_idx, SLICE_EDGES

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _spy_daily():
    d = pl.read_parquet(f"{HIST}/SPY.parquet", columns=["start_time", "open", "high", "low", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 960)].sort_values(["date", "mod"])
    rows = []
    for dt, g in d.groupby("date"):
        c = g["close"].to_numpy(float)
        o = g["open"].to_numpy(float)[0]
        er = abs(c[-1] - o) / max(np.sum(np.abs(np.diff(c))), 1e-9)
        rows.append((dt, o, c[-1], er))
    df = pd.DataFrame(rows, columns=["date", "open", "close", "er_1d"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["rv_20d"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["drv_5d"] = df["rv_20d"] - df["rv_20d"].shift(5)
    df["absret_1d"] = df["ret"].abs()
    df["gap"] = (df["open"] / df["close"].shift(1) - 1).abs()
    # 10-day directional persistence
    r10 = df["close"] / df["close"].shift(10) - 1
    sumabs = df["ret"].abs().rolling(10).sum()
    df["er_10d"] = r10.abs() / sumabs.replace(0, np.nan)
    # everything except `gap` is shifted to prior session (lookahead-free)
    for c in ("er_1d", "rv_20d", "drv_5d", "absret_1d", "er_10d"):
        df[c] = df[c].shift(1)
    return df


def _spy_gex_sign():
    g = pl.read_parquet(f"{HIST}/GEXSPY.parquet").to_pandas()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")
    col = "net_gex_prior" if "net_gex_prior" in g.columns else "net_gex"
    return {d.date(): ("POSITIVE" if v > 0 else "NEGATIVE") for d, v in zip(g["date"], g[col])}


def _vix():
    try:
        import requests
        from dotenv import load_dotenv
        load_dotenv()
        h = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json"}
        r = requests.get("https://api.unusualwhales.com/api/stock/VIX/volatility/realized",
                         headers=h, params={"timeframe": "2Y"}, timeout=20)
        rows = r.json().get("data", [])
        s = pd.Series({pd.Timestamp(x["date"]).date(): float(x["price"]) for x in rows if x.get("price")})
        s = s.sort_index()
        return s.shift(1), (s - s.shift(5)).shift(1)   # prior-day level, prior 5d change
    except Exception as e:
        print(f"  (VIX unavailable: {e})")
        return pd.Series(dtype=float), pd.Series(dtype=float)


def _agg(pnls):
    if len(pnls) < 8:
        return f"n={len(pnls):>4} thin"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    sd = np.std([np.mean(b) for b in sl if len(b) >= 5]) * 100
    slc = " ".join(f"S{j+1}{np.mean(b) * 100:+.0f}" if len(b) >= 5 else f"S{j+1}··" for j, b in enumerate(sl))
    return (f"n={len(v):>4}  avg {v.mean() * 100:>+6.1f}%  IS {np.mean(i) * 100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o) * 100 if o else float('nan'):>+6.1f}%  win {np.mean(v > 0):.2f}  "
            f"sliceSD {sd:4.1f}  [{slc}]")


def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 40:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def run(a):
    import directional_flow_backtester as D
    from amt_profile import amt_open_map, amt_ok
    from config import RULES
    from check_adx_dmi import _intraday_adx, _intra_asof
    from macro_calendar import is_macro_am_day
    from check_open_delay import _rule_trades

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]
    tickers = sorted({r["ticker"] for r in rules})
    flow_all = _flow_for(D, tickers)

    spy = _spy_daily().set_index("date")
    gexsign = _spy_gex_sign()
    vix, dvix = _vix()

    TR = []
    for tk in tickers:
        tk_rules = [r for r in rules if r["ticker"] == tk]
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        amt = amt_open_map(tk) if any(r.get("amt_open") for r in tk_rules) else {}
        ema_stacks = {int(r["ema_confirm"]): D.load_ema_stack(HIST, tk, int(r["ema_confirm"]))
                      for r in tk_rules if r.get("ema_confirm")}
        dmi_tfs = {int(r["dmi_confirm"].get("tf", 15)) for r in tk_rules if r.get("dmi_confirm")}
        iadx = {tf: _intraday_adx(tk, tf, close_only=True) for tf in dmi_tfs}
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {dd: g for dd, g in tb.groupby("date")}
        for r in tk_rules:
            for d, p, m in _rule_trades(D, r, flow_all, gex, vol, trd, amp, reg_src, amt, amt_ok,
                                        ema_stacks, bbd, bbc, iadx, _intra_asof, is_macro_am_day):
                TR.append((r["name"], d, p))

    df = pd.DataFrame(TR, columns=["rule", "date", "pnl"])
    df["d"] = pd.to_datetime(df["date"])
    df = df.merge(spy, left_on="d", right_index=True, how="left")
    df["gex_sign"] = df["date"].map(lambda x: gexsign.get(x))
    df["vix"] = df["date"].map(lambda x: vix.get(pd.Timestamp(x).date()) if len(vix) else np.nan)
    df["dvix"] = df["date"].map(lambda x: dvix.get(pd.Timestamp(x).date()) if len(dvix) else np.nan)
    df["month"] = df["d"].dt.month
    df["slice"] = df["date"].map(_slice_idx)

    METRICS = ["er_1d", "er_10d", "rv_20d", "drv_5d", "absret_1d", "gap", "vix", "dvix"]
    print("=" * 116)
    print(f"  BOOK P&L by CONTEMPORANEOUS MARKET-STATE   ({len(df)} trades)   split {SPLIT}")
    print("=" * 116)
    print(f"\n  baseline: {_agg(list(zip(df['date'], df['pnl'])))}")

    print("\n  -- date-clustered spearman(metric, trade P&L) --")
    for mc in METRICS + ["month"]:
        sub = df.dropna(subset=[mc])
        if len(sub) < 100:
            print(f"    {mc:10}  (n={len(sub)} thin)"); continue
        rho_all = _spear(sub[mc], sub["pnl"])
        rho_is = _spear(sub[sub.date < SPLIT][mc], sub[sub.date < SPLIT]["pnl"])
        rho_oos = _spear(sub[sub.date >= SPLIT][mc], sub[sub.date >= SPLIT]["pnl"])
        print(f"    {mc:10}  rho ALL {rho_all:+.3f}   IS {rho_is:+.3f}   OOS {rho_oos:+.3f}   n {len(sub)}")

    print("\n  -- quintile P&L by metric --")
    for mc in METRICS:
        sub = df.dropna(subset=[mc]).copy()
        if len(sub) < 150:
            continue
        try:
            sub["q"] = pd.qcut(sub[mc], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        cells = []
        for q in sorted(sub["q"].dropna().unique()):
            b = sub[sub["q"] == q]
            cells.append(f"Q{int(q)+1} {b['pnl'].mean()*100:+5.1f}%(w{(b['pnl']>0).mean():.2f})")
        print(f"    {mc:10}  " + "  ".join(cells))
    # gex_sign
    for sgn in ("POSITIVE", "NEGATIVE"):
        b = df[df.gex_sign == sgn]
        if len(b) >= 50:
            print(f"    gex={sgn:9}  {_agg(list(zip(b['date'], b['pnl'])))}")

    # ---- does the top metric EXPLAIN the calendar? ----
    print("\n" + "=" * 116)
    print("  DOES A METRIC EXPLAIN THE SLICE PATTERN?  (P&L per slice within favourable vs unfavourable bucket)")
    print("=" * 116)
    for mc, hi_is_good in (("vix", True), ("er_1d", True), ("er_10d", True), ("rv_20d", True), ("absret_1d", True)):
        sub = df.dropna(subset=[mc]).copy()
        if len(sub) < 200:
            continue
        med = sub[mc].median()
        fav = sub[sub[mc] >= med] if hi_is_good else sub[sub[mc] < med]
        unf = sub[sub[mc] < med] if hi_is_good else sub[sub[mc] >= med]
        print(f"\n  {mc}  (favourable = {'high' if hi_is_good else 'low'}, split at median {med:.3f})")
        print(f"    favourable   {_agg(list(zip(fav['date'], fav['pnl'])))}")
        print(f"    unfavourable {_agg(list(zip(unf['date'], unf['pnl'])))}")
        mby = sub.groupby("slice")[mc].mean()
        print(f"    metric mean by slice: " + "  ".join(f"S{int(k)+1} {v:.3f}" for k, v in mby.items() if pd.notna(k)))

    # ---- is VIX / vol a WITHIN-RULE lever, or just book composition? ----
    print("\n" + "=" * 116)
    print("  WITHIN-RULE: P&L by VIX tercile and by prior-day gamma sign  (does the state add signal INSIDE a rule?)")
    print("=" * 116)
    big = df["rule"].value_counts()
    for rn in big[big >= 120].index:
        g = df[df["rule"] == rn]
        gv = g.dropna(subset=["vix"])
        line = f"    {rn:22} "
        if len(gv) >= 90:
            lo, hi = gv["vix"].quantile(0.33), gv["vix"].quantile(0.67)
            for lab, m in (("VIXlo", gv.vix <= lo), ("VIXmid", (gv.vix > lo) & (gv.vix < hi)), ("VIXhi", gv.vix >= hi)):
                b = gv[m]
                line += f"{lab} {b['pnl'].mean()*100:+5.1f}%(n{len(b)}) "
        for sgn in ("POSITIVE", "NEGATIVE"):
            b = g[g.gex_sign == sgn]
            if len(b) >= 20:
                line += f" gex{sgn[:3]} {b['pnl'].mean()*100:+5.1f}%(n{len(b)})"
        print(line)

    # ---- throttle sim ----
    print("\n" + "=" * 116)
    print("  SIZE-THROTTLE SIM   (full size when metric favourable, half otherwise)")
    print("=" * 116)
    for mc, hi_is_good, cut in (("vix", True, "p33"), ("er_1d", True, "median"), ("er_10d", True, "median"),
                                ("rv_20d", True, "p33"), ("absret_1d", True, "median")):
        sub = df.dropna(subset=[mc]).copy()
        if len(sub) < 200:
            continue
        thr = sub[mc].median() if cut == "median" else sub[mc].quantile(0.33)
        fav_mask = (sub[mc] >= thr) if hi_is_good else (sub[mc] < thr)
        base = list(zip(sub["date"], sub["pnl"]))
        thr_pnl = list(zip(sub["date"], np.where(fav_mask, sub["pnl"], sub["pnl"] * 0.5)))
        print(f"\n  {mc} (throttle to 0.5x when {'below' if hi_is_good else 'above'} {cut} {thr:.3f})")
        print(f"    base      {_agg(base)}")
        print(f"    throttled {_agg(thr_pnl)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    a = ap.parse_args()
    run(a)
