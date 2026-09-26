# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
triage_smallcap_uoa.py
=======================
Phase-2 triage screen over the 90-day small/mid-cap backfill (2026-04-01 ->
2026-08-21). For each ticker in smallcap_universe.json, scores whether it
produces tradeable FLOW-TRIGGER signal worth promoting to a full ~2yr backfill:

  density   avg per-minute net-prem records / day (EMA-crossover needs enough
            intraday points -- <~30/day is too sparse for the cum-flow EMA)
  n_trig    EMA(5) cum-flow crossovers over the 90d (same triggers_for logic
            the live bot uses)
  hit       directional hit-rate: CALL trigger -> underlying up over next 30m /
            PUT -> down.  0.50 = coin flip
  edge      mean SIGNED 30m forward return in the trigger's direction (bps)

90 days is too short to walk-forward validate -- this is hypothesis-generating.
Output: smallcap_triage_shortlist.json (names passing density+count+edge) +
a ranked table.

Usage:
  python triage_smallcap_uoa.py
  python triage_smallcap_uoa.py --min-density 30 --min-trig 20 --min-hit 0.52
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
UNIVERSE = "smallcap_universe.json"
OUT = "smallcap_triage_shortlist.json"


def _naive(s):
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert("America/New_York").dt.tz_localize(None)


def _triggers(tk):
    """(date, ts, dir, abs_flow) EMA(5) cum-flow crossovers -- mirrors
    directional_flow_backtester.triggers_for on a single ticker's netprem."""
    p = f"{HIST}/NETPREM{tk}.parquet"
    if not os.path.exists(p):
        return None, 0.0
    df = pl.read_parquet(p).to_pandas()
    if "net_premium" not in df.columns or df.empty:
        return None, 0.0
    df["minute_et"] = _naive(df["minute_et"])
    df["date"] = df["minute_et"].dt.date
    df = df.sort_values("minute_et")
    n_days = df["date"].nunique()
    density = len(df) / max(n_days, 1)
    df["cum"] = df.groupby("date")["net_premium"].cumsum()
    out = []
    for d, g in df.groupby("date"):
        g = g.sort_values("minute_et")
        cum = g["cum"].values.astype(float)
        if len(cum) < 6:
            continue
        ema = pd.Series(cum).ewm(span=5, adjust=False).mean().values
        mt = g["minute_et"].values
        hrs = g["minute_et"].dt.hour.values
        for i in range(1, len(cum)):
            bull = cum[i - 1] <= ema[i - 1] and cum[i] > ema[i]
            bear = cum[i - 1] >= ema[i - 1] and cum[i] < ema[i]
            if (bull or bear) and 9 <= hrs[i] < 15:
                out.append((d, mt[i], "CALL" if bull else "PUT", abs(cum[i])))
    return out, density


def _fwd_returns(tk, trigs):
    """hit-rate + mean signed 30m forward return for a ticker's triggers."""
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p) or not trigs:
        return None
    d = pl.read_parquet(p).to_pandas()
    d.columns = [c.lower() for c in d.columns]
    if "start_time" not in d.columns or "close" not in d.columns:
        return None
    et = _naive(d["start_time"])
    mo = et.dt.hour * 60 + et.dt.minute
    m = (mo >= 570) & (mo <= 960)
    px = pd.DataFrame({"ts": et[m].values, "mod": mo[m].values, "c": d["close"][m].astype(float).values})
    px["date"] = pd.to_datetime(px["ts"]).dt.date
    by_date = {dd: g.set_index("mod")["c"] for dd, g in px.groupby("date")}
    sret, hits, n = [], 0, 0
    for dd, ts, dirn, _ in trigs:
        s = by_date.get(dd)
        if s is None:
            continue
        mod = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
        pre = s.index[s.index <= mod]
        post = s.index[s.index <= mod + 30]
        if len(pre) == 0 or len(post) == 0 or post[-1] - mod < 18:
            continue
        r = s.loc[post[-1]] / s.loc[pre[-1]] - 1
        sr = r if dirn == "CALL" else -r
        sret.append(sr)
        hits += sr > 0
        n += 1
    if n < 5:
        return None
    return {"n": n, "hit": hits / n, "edge_bps": float(np.mean(sret)) * 1e4}


def run(a):
    with open(UNIVERSE) as f:
        tickers = json.load(f)["tickers"]
    rows = []
    for i, tk in enumerate(tickers, 1):
        trigs, density = _triggers(tk)
        if trigs is None:
            continue
        fr = _fwd_returns(tk, trigs)
        rows.append({"ticker": tk, "density": round(density, 1), "n_trig": len(trigs),
                     "hit": None if not fr else round(fr["hit"], 3),
                     "edge_bps": None if not fr else round(fr["edge_bps"], 1),
                     "n_scored": 0 if not fr else fr["n"]})
        if i % 200 == 0:
            print(f"  {i}/{len(tickers)}")
    df = pd.DataFrame(rows)
    df.to_parquet("_smallcap_triage_all.parquet", index=False)

    keep = df[(df["density"] >= a.min_density) & (df["n_trig"] >= a.min_trig)
              & df["hit"].notna() & (df["hit"] >= a.min_hit) & (df["edge_bps"] > 0)].copy()
    keep["score"] = keep["n_trig"] * keep["edge_bps"] * (keep["hit"] - 0.5)
    keep = keep.sort_values("score", ascending=False)

    print(f"\n  scanned {len(df)} tickers with data")
    print(f"  {(df['density'] >= a.min_density).sum()} pass density >= {a.min_density}/day")
    print(f"  {len(keep)} pass ALL (density, n_trig >= {a.min_trig}, hit >= {a.min_hit}, edge > 0)")
    print(f"\n  top 40 by score (n_trig x edge_bps x (hit-0.5)):")
    print(f"  {'ticker':8}{'density':>9}{'n_trig':>8}{'hit':>7}{'edge_bps':>10}{'n_sc':>7}")
    for _, r in keep.head(40).iterrows():
        print(f"  {r['ticker']:8}{r['density']:>9.1f}{r['n_trig']:>8}{r['hit']:>7.3f}"
              f"{r['edge_bps']:>10.1f}{r['n_scored']:>7}")

    with open(OUT, "w") as f:
        json.dump({"tickers": keep["ticker"].tolist(), "n": len(keep),
                    "criteria": {"min_density": a.min_density, "min_trig": a.min_trig,
                                 "min_hit": a.min_hit}}, f, indent=2)
    print(f"\n  wrote {len(keep)} -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-density", type=float, default=30.0)
    ap.add_argument("--min-trig", type=int, default=20)
    ap.add_argument("--min-hit", type=float, default=0.52)
    a = ap.parse_args()
    run(a)
