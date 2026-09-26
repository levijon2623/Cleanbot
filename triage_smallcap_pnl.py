# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
triage_smallcap_pnl.py
=======================
Phase-2b triage: the metric that actually predicted success for the mega-cap
rules -- simulated 0DTE/1DTE option-BRACKET P&L (fees included), not raw
directional hit-rate. Zero API cost: the silver option-contract bars are
already on disk for every ticker (silver is full-market), so this just points
directional_flow_backtester._option_paths / _bracket_pnl at the 90-day
small/mid-cap flow triggers.

  1) one lazy pass over the 99 in-window silver day-partitions, filtered to the
     candidate tickers (density-passers from _smallcap_triage_all.parquet),
     DTE 0-8, |moneyness| <= 4%  ->  _smallcap_pnl_cache/bars.parquet
  2) per ticker: EMA(5) cum-flow crossovers on the 90d netprem (same triggers_for
     logic), trailing-45d percentile gate (annotate_flow_pct), then _option_paths
     + _bracket_pnl at a fixed recipe  ->  mean option expectancy + win rate
  3) rank by expectancy; write smallcap_pnl_shortlist.json

90 days, no IS/OOS -- still hypothesis-generating, but on the RIGHT axis.

Usage:
  python triage_smallcap_pnl.py
  python triage_smallcap_pnl.py --pct 80 --tr 1.0 --rr 1.0 --min-trades 15
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
SILVER = "lake/silver/option-contracts-1m"
CACHE_DIR = "_smallcap_pnl_cache"
BARS_CACHE = f"{CACHE_DIR}/bars.parquet"
WIN_LO, WIN_HI = "2026-04-01", "2026-08-21"
OUT = "smallcap_pnl_shortlist.json"


def _naive(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)


def _triggers(tk):
    """[{date, ts, dir, abs_flow}] EMA(5) cum-flow crossovers -- mirrors
    directional_flow_backtester.triggers_for on one ticker's netprem."""
    p = f"{HIST}/NETPREM{tk}.parquet"
    if not os.path.exists(p):
        return [], 0.0
    df = pl.read_parquet(p).to_pandas()
    if "net_premium" not in df.columns or df.empty:
        return [], 0.0
    df["minute_et"] = _naive(df["minute_et"])
    df["date"] = df["minute_et"].dt.date
    df = df.sort_values("minute_et")
    density = len(df) / max(df["date"].nunique(), 1)
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
                out.append({"date": d, "ts": pd.Timestamp(mt[i]),
                            "dir": "CALL" if bull else "PUT", "abs_flow": float(abs(cum[i]))})
    return out, density


def _extract_bars(candidates):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(BARS_CACHE):
        return pd.read_parquet(BARS_CACHE)
    parts = [p for p in sorted(glob.glob(f"{SILVER}/date=*/bars.parquet"))
             if WIN_LO <= p.split("date=")[1][:10] <= WIN_HI]
    want = set(candidates)
    frames = []
    for i, p in enumerate(parts, 1):
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol").is_in(want))
              .with_columns(((pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days()).alias("dte"))
              .filter((pl.col("dte") >= 0) & (pl.col("dte") <= 8))
              .filter((pl.col("strike") - pl.col("underlying_close")).abs() / pl.col("underlying_close") <= 0.04)
              .select(["underlying_symbol", "option_chain_id", "option_type", "strike", "expiry",
                       "minute_et", "high", "low", "close", "bid_close", "ask_close", "underlying_close"]))
        frames.append(lf.collect().to_pandas())
        if i % 20 == 0:
            print(f"  bars {i}/{len(parts)}")
    bars = pd.concat(frames, ignore_index=True)
    bars["minute_et"] = _naive(bars["minute_et"])
    bars["date"] = bars["minute_et"].dt.date
    bars["expiry"] = pd.to_datetime(bars["expiry"]).dt.date
    bars.to_parquet(BARS_CACHE, index=False)
    print(f"  cached {len(bars):,} rows -> {BARS_CACHE}")
    return bars


def run(a):
    import directional_flow_backtester as D

    allsc = pd.read_parquet("_smallcap_triage_all.parquet")
    cands = allsc[allsc["density"] >= a.min_density]["ticker"].tolist()
    print(f"  {len(cands)} density-passing candidates")
    bars = _extract_bars(cands)
    bars_by_tk = {tk: g for tk, g in bars.groupby("underlying_symbol")}

    rows = []
    for j, tk in enumerate(cands, 1):
        trigs, density = _triggers(tk)
        if not trigs:
            continue
        D.annotate_flow_pct(trigs, 45)
        tb = bars_by_tk.get(tk)
        if tb is None or tb.empty:
            continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        pnls = []
        for t in trigs:
            thr = t.get("thr")
            if not thr or a.pct not in thr or t["abs_flow"] < thr[a.pct]:
                continue
            for p in D._option_paths(t, t["dir"], [0, 1], bbd, bbc):
                pnls.append(D._bracket_pnl(*p, a.tr, a.rr, None, 15 * 60 + 55))
        n = len(pnls)
        rows.append({"ticker": tk, "density": round(density, 1), "n_trig": len(trigs),
                     "n_trades": n,
                     "exp_pct": round(float(np.mean(pnls)) * 100, 1) if n else None,
                     "win": round(float(np.mean([x > 0 for x in pnls])), 3) if n else None})
        if j % 100 == 0:
            print(f"  {j}/{len(cands)}")

    df = pd.DataFrame(rows)
    df.to_parquet(f"{CACHE_DIR}/pnl_all.parquet", index=False)
    keep = df[(df["n_trades"] >= a.min_trades) & df["exp_pct"].notna() & (df["exp_pct"] > 0)].copy()
    keep = keep.sort_values("exp_pct", ascending=False)

    print(f"\n  {len(df)} tickers simulated, {(df['n_trades'] >= a.min_trades).sum()} with >= {a.min_trades} option trades")
    print(f"  {len(keep)} positive expectancy   (recipe: tr={a.tr} rr={a.rr} pct={a.pct}, 15:55 flatten, fees in)")
    if len(df[df['n_trades'] >= a.min_trades]):
        pool = df[df['n_trades'] >= a.min_trades]
        print(f"  pooled expectancy across all {len(pool)} tradeable names: {pool['exp_pct'].mean():+.2f}%  "
              f"(median {pool['exp_pct'].median():+.2f}%)")
    print(f"\n  top 40 by expectancy:")
    print(f"  {'ticker':8}{'density':>9}{'n_trig':>8}{'n_trd':>7}{'exp%':>9}{'win':>7}")
    for _, r in keep.head(40).iterrows():
        print(f"  {r['ticker']:8}{r['density']:>9.1f}{r['n_trig']:>8}{r['n_trades']:>7}"
              f"{r['exp_pct']:>9.1f}{r['win']:>7.3f}")

    with open(OUT, "w") as f:
        json.dump({"tickers": keep["ticker"].tolist(), "n": len(keep),
                    "recipe": {"tr": a.tr, "rr": a.rr, "pct": a.pct, "min_trades": a.min_trades}}, f, indent=2)
    print(f"\n  wrote {len(keep)} -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-density", type=float, default=30.0)
    ap.add_argument("--pct", type=int, default=80, choices=[50, 65, 80, 90, 95])
    ap.add_argument("--tr", type=float, default=1.0)
    ap.add_argument("--rr", type=float, default=1.0)
    ap.add_argument("--min-trades", type=int, default=15)
    a = ap.parse_args()
    run(a)
