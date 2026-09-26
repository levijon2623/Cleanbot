# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27.0", "polars>=1.0.0", "python-dotenv"]
# ///
"""
build_smallcap_universe.py
============================
Phase 1 of the small/mid-cap UOA expansion: cheap ticker SELECTION, no history
backfill. /screener/stocks only covers the top 500 most-active names (verified
2026-09-04 -- offset=500 returns empty), so it can't reach small/mid-caps on its
own; this instead classifies the full silver-layer universe (~4,850 optionable
tickers, already on disk, zero API cost to enumerate) one /stock/{t}/info call
at a time (cheap: marketcap + issue_type + has_options, ~1 req/ticker, no
per-day fan-out) and ranks survivors by EXISTING LOCAL option volume (silver,
free) to pick the most liquid --limit of them.

  --classify   run the ~4,850 /stock/{t}/info calls -> _universe_cache/info.json
               (incremental: skips tickers already classified)
  --rank       filter (Common Stock, has_options, marketcap in [--cap-min,
               --cap-max], not already in config.WATCHLIST) + rank by trailing
               20-day avg local option volume -> smallcap_universe.json

Usage:
  python build_smallcap_universe.py --classify
  python build_smallcap_universe.py --rank --limit 1500 --cap-min 3e8 --cap-max 1e10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import httpx
import polars as pl
from dotenv import load_dotenv

load_dotenv()
API = "https://api.unusualwhales.com/api"
HDRS = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json",
        "User-Agent": "cleanbot/1.0"}
SILVER = "lake/silver/option-contracts-1m"
CACHE_DIR = "_universe_cache"
INFO_CACHE = f"{CACHE_DIR}/info.json"
OUT_PATH = "smallcap_universe.json"


def _full_ticker_list():
    """Every underlying_symbol that's ever appeared in the silver bars -- scan a
    handful of recent day-partitions (union) rather than all ~500 for speed."""
    parts = sorted(glob.glob(f"{SILVER}/date=*/bars.parquet"))[-15:]
    tickers = set()
    for p in parts:
        tickers |= set(pl.scan_parquet(p).select("underlying_symbol").unique()
                        .collect()["underlying_symbol"].to_list())
    return sorted(tickers)


def classify(a):
    os.makedirs(CACHE_DIR, exist_ok=True)
    have = {}
    if os.path.exists(INFO_CACHE):
        with open(INFO_CACHE) as f:
            have = json.load(f)
    tickers = _full_ticker_list()
    todo = [t for t in tickers if t not in have]
    print(f"  universe: {len(tickers)} tickers, {len(have)} already classified, {len(todo)} to fetch")
    for i, t in enumerate(todo, 1):
        try:
            r = httpx.get(f"{API}/stock/{t}/info", headers=HDRS, timeout=15)
            row = r.json().get("data", {}) if r.status_code == 200 else {}
        except httpx.HTTPError:
            row = {}
        have[t] = {"marketcap": row.get("marketcap"), "issue_type": row.get("issue_type"),
                   "has_options": row.get("has_options"), "marketcap_size": row.get("marketcap_size"),
                   "sector": row.get("sector")}
        if i % 100 == 0:
            print(f"    {i}/{len(todo)}")
            with open(INFO_CACHE, "w") as f:
                json.dump(have, f)
        time.sleep(0.2)
    with open(INFO_CACHE, "w") as f:
        json.dump(have, f)
    print(f"  wrote {len(have)} tickers -> {INFO_CACHE}")


def _local_liquidity(tickers):
    """Trailing 20-day avg option volume per ticker, from LOCAL silver (free)."""
    parts = sorted(glob.glob(f"{SILVER}/date=*/bars.parquet"))[-20:]
    want = set(tickers)
    frames = []
    for p in parts:
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol").is_in(want))
              .group_by("underlying_symbol")
              .agg((pl.col("ask_volume").sum() + pl.col("bid_volume").sum()).alias("vol")))
        frames.append(lf.collect())
    if not frames:
        return {}
    tot = pl.concat(frames).group_by("underlying_symbol").agg(pl.col("vol").mean().alias("avg_vol"))
    return dict(zip(tot["underlying_symbol"].to_list(), tot["avg_vol"].to_list()))


def rank(a):
    with open(INFO_CACHE) as f:
        info = json.load(f)
    from config import WATCHLIST
    already = set() if a.keep_watchlist else set(WATCHLIST) | {"SPY", "QQQ", "IWM", "AVGO", "GLD"}
    out_path = a.out or OUT_PATH

    cands = []
    for t, row in info.items():
        if t in already:
            continue
        if row.get("issue_type") != "Common Stock" or not row.get("has_options"):
            continue
        mc = row.get("marketcap")
        try:
            mc = float(mc)
        except (TypeError, ValueError):
            continue
        if not (a.cap_min <= mc <= a.cap_max):
            continue
        cands.append(t)
    print(f"  {len(cands)} candidates pass Common-Stock/has_options/marketcap [{a.cap_min:,.0f}, {a.cap_max:,.0f}]")

    liq = _local_liquidity(cands)
    cands = [t for t in cands if liq.get(t, 0) > 0]
    cands.sort(key=lambda t: liq[t], reverse=True)
    chosen = cands[:a.limit]
    print(f"  {len(cands)} have measurable local option volume; keeping top {len(chosen)}")
    print(f"  liquidity range kept: {liq[chosen[0]]:,.0f} (most liquid) .. {liq[chosen[-1]]:,.0f} (least)")

    with open(out_path, "w") as f:
        json.dump({"tickers": chosen, "cap_min": a.cap_min, "cap_max": a.cap_max,
                    "n": len(chosen)}, f, indent=2)
    print(f"  wrote {len(chosen)} tickers -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--cap-min", type=float, default=3e8)
    ap.add_argument("--cap-max", type=float, default=1e10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-watchlist", action="store_true")
    a = ap.parse_args()
    if a.classify:
        classify(a)
    elif a.rank:
        rank(a)
    else:
        ap.print_help()
