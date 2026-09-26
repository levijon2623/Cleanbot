# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27.0", "polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "python-dotenv"]
# ///
"""
check_market_tide.py
====================

Market Tide = market-wide net options premium (net_call_premium - net_put_premium,
cumulative on the day, ask-lift minus bid-hit).  The per-TICKER version of this
had zero forward predictive power (session 11 netprem diagnostic) -- does the
MARKET-WIDE aggregate behave differently as a macro risk-on/off regime?

  --build           pull /market/market-tide for the window -> _tide_cache/
  --test predict    tide level / slope  ->  SPY & QQQ forward 30/60-min return,
                    vs the drift-free unconditional-range baseline
  --test rules      split the index CALL rules' option P&L by the tide state at
                    the trigger minute -- does "tide bullish" gate improve them?

Usage:
  python check_market_tide.py --build 2024-08-20 2026-08-21
  python check_market_tide.py --test predict --split 2025-08-21
  python check_market_tide.py --test rules --split 2025-08-21
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import polars as pl
import httpx
from dotenv import load_dotenv

load_dotenv()
API = "https://api.unusualwhales.com/api"
HDRS = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json",
        "User-Agent": "cleanbot/1.0"}
HIST = "historical"
CACHE = "_tide_cache/market_tide.parquet"


def _trading_days(a: dt.date, b: dt.date):
    d, out = a, []
    while d <= b:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def build(a):
    os.makedirs("_tide_cache", exist_ok=True)
    lo, hi = sorted(dt.date.fromisoformat(x) for x in a.dates)
    have = set()
    if os.path.exists(CACHE):
        have = set(pd.read_parquet(CACHE, columns=["date"])["date"].astype(str))
    rows = []
    days = [d for d in _trading_days(lo, hi) if d.isoformat() not in have]
    for i, d in enumerate(days, 1):
        try:
            r = httpx.get(f"{API}/market/market-tide", headers=HDRS,
                          params={"date": d.isoformat()}, timeout=30)
            data = r.json().get("data", []) if r.status_code == 200 else []
        except httpx.HTTPError:
            data = []
        for x in data:
            rows.append((x["date"], x["timestamp"],
                         float(x["net_call_premium"]), float(x["net_put_premium"]),
                         float(x["net_volume"])))
        if i % 50 == 0:
            print(f"  {i}/{len(days)}")
        time.sleep(0.15)
    if not rows:
        print("nothing new"); return
    new = pd.DataFrame(rows, columns=["date", "timestamp", "ncp", "npp", "nvol"])
    if os.path.exists(CACHE):
        new = pd.concat([pd.read_parquet(CACHE), new], ignore_index=True)
    new = new.drop_duplicates("timestamp").sort_values("timestamp")
    new.to_parquet(CACHE, index=False)
    print(f"  {len(new)} rows -> {CACHE}  ({new['date'].min()} .. {new['date'].max()})")


def _load_tide():
    df = pd.read_parquet(CACHE)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = df["ts"].dt.date
    df["mod"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[(df["mod"] >= 570) & (df["mod"] <= 960)].sort_values("ts")
    df["net_prem"] = df["ncp"] - df["npp"]          # cumulative, per day
    # de-trend the ABSOLUTE scale (market-wide premium grew over 2y): z-score the
    # day's net_prem path vs a trailing 60-session pool of same-minute values
    df["np_slope"] = df.groupby("date")["net_prem"].diff(3)     # last 15 min
    return df


def _spy_qqq_1m():
    out = {}
    for tk in ("SPY", "QQQ"):
        p = f"{HIST}/{tk}.parquet"
        if not os.path.exists(p):
            continue
        d = pl.read_parquet(p).to_pandas()
        d.columns = [c.lower() for c in d.columns]
        et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        mo = et.dt.hour * 60 + et.dt.minute
        m = (mo >= 570) & (mo <= 960)
        g = pd.DataFrame({"ts": et[m].values, "mod": mo[m].values,
                          "c": d["close"][m].astype(float).values,
                          "h": d["high"][m].astype(float).values,
                          "l": d["low"][m].astype(float).values})
        g["ts"] = pd.to_datetime(g["ts"]); g["date"] = g["ts"].dt.date
        out[tk] = g.sort_values("ts").reset_index(drop=True)
    return out


def test_predict(a):
    tide = _load_tide()
    split = pd.Timestamp(a.split).date() if a.split else None
    px = _spy_qqq_1m()
    for tk, g in px.items():
        gd = {d: x for d, x in g.groupby("date")}
        rows = []
        for d, td in tide.groupby("date"):
            day = gd.get(d)
            if day is None:
                continue
            dpx = day.set_index("mod")
            up = (dpx["h"].max() - dpx["c"].iloc[0]) / dpx["c"].iloc[0]
            dn = (dpx["c"].iloc[0] - dpx["l"].min()) / dpx["c"].iloc[0]
            for _, r in td.iterrows():
                m = int(r["mod"])
                pm = dpx.index[dpx.index <= m]
                if len(pm) == 0:
                    continue
                p0 = dpx.loc[pm[-1], "c"]
                for h in (30, 60):
                    fm = dpx.index[dpx.index <= m + h]
                    if len(fm) == 0 or fm[-1] - m < h * 0.6:
                        continue
                    fret = dpx.loc[fm[-1], "c"] / p0 - 1
                    rows.append((d, m, r["net_prem"], r["np_slope"], h, fret))
        R = pd.DataFrame(rows, columns=["date", "mod", "level", "slope", "h", "fret"])
        R = R[R["mod"] <= 900]
        print(f"\n  {tk}  tide -> forward return   (n={len(R)})")
        for h in (30, 60):
            s = R[R["h"] == h]
            for lbl, sub in ([("IS", s[s.date < split]), ("OOS", s[s.date >= split])] if split
                             else [("ALL", s)]):
                if len(sub) < 100:
                    continue
                sl = _spear(sub["level"], sub["fret"])
                ss = _spear(sub["slope"], sub["fret"])
                print(f"     +{h}m  {lbl:4} n={len(sub):>5}  sp(level) {sl:+.3f}   sp(slope) {ss:+.3f}")


def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def test_rules(a):
    import directional_flow_backtester as D
    from config import RULES
    tide = _load_tide()
    # tide state per (date, mod-bucket): 'bull' if net_prem > 0 AND rising, 'bear' if < 0 AND falling, else 'neutral'
    tide["state"] = np.where((tide["net_prem"] > 0) & (tide["np_slope"] > 0), "bull",
                    np.where((tide["net_prem"] < 0) & (tide["np_slope"] < 0), "bear", "neutral"))
    tstate = {}
    for _, r in tide.iterrows():
        tstate.setdefault(r["date"], []).append((int(r["mod"]), r["state"]))
    for d in tstate:
        tstate[d].sort()

    def state_at(date, ts):
        arr = tstate.get(date)
        if not arr:
            return None
        m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
        prev = [s for mm, s in arr if mm <= m]
        return prev[-1] if prev else None

    split = pd.Timestamp(a.split).date()
    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date
    idx_rules = [r for r in RULES if r.get("enabled", True) and r["direction"] == "CALL"
                 and r["ticker"] in ("SPY", "QQQ", "IWM")]
    for r in idx_rules:
        tk = r["ticker"]
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        trigs = D.triggers_for(flow, tk)
        if not trigs:
            tkf, _ = D._screen_build_one("lake/silver/option-contracts-1m", tk)
            trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
        D.annotate_flow_pct(trigs, 60)
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        by = {"ALL": [], "bull": [], "neutral": [], "bear": []}
        for t, thr in D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src):
            st = state_at(t["date"], t["ts"])
            for _, _, _, dd, pnl in D.simulate_trigger(t, "CALL", r.get("dte", [0, 1]), r.get("time_stop_mins"),
                                                       bbd, bbc, only=(thr, float(r["target_roe"]), float(r["rr"]))):
                by["ALL"].append((dd, pnl))
                if st:
                    by[st].append((dd, pnl))
        print(f"\n  {r['name']}   split by MARKET TIDE at trigger")
        for k in ("ALL", "bull", "neutral", "bear"):
            v = by[k]
            if len(v) < 15:
                print(f"    {k:8} n={len(v):>4} (thin)"); continue
            ii = [p for d, p in v if d < split]; oo = [p for d, p in v if d >= split]
            print(f"    {k:8} n={len(v):>4}  IS {np.mean(ii)*100 if ii else float('nan'):>+6.1f}%  "
                  f"OOS {np.mean(oo)*100 if oo else float('nan'):>+6.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("dates", nargs="*")
    ap.add_argument("--test", choices=("predict", "rules"))
    ap.add_argument("--split", default="2025-08-21")
    a = ap.parse_args()
    if a.build:
        build(a)
    elif a.test == "predict":
        test_predict(a)
    elif a.test == "rules":
        test_rules(a)
    else:
        ap.print_help()
