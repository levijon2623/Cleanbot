# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_scanner.py
================

Tests the UW "multi-signal scanner" idea (their Sept-2026 dev email): flag a
ticker when 2+ of these fire the same session --
  1. FLOW ALERT   : a contract with day premium > $250k AND day volume > prior OI
  2. OI GROWTH    : the ticker's total OI increased for >= --oi-days consecutive sessions
  3. DARK POOL    : dark-pool volume > --dp-pct of the 30d avg stock volume  [phase 2, needs a backfill]

Phase 1 here: signals 1 + 2 only, derived from the silver-tape OI snapshots
(_oi_cache/ from check_oi_conviction; DTE 7-60 / |mny|<=25% -- a proxy).
Does a 2-signal day predict the forward return, vs the drift-free unconditional
range baseline (the control that busted magnet theory)?

Usage:
  python check_scanner.py --tickers NVDA AVGO MU META TSLA --split 2025-08-21
  python check_scanner.py --oi-days 3 --fwd 10 --split 2025-08-21
"""
from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

OI_CACHE = "_oi_cache"
HIST = "historical"
COHORT = ["NVDA", "AVGO", "MU", "META", "TSLA"]


def _daily_signals(tk: str, oi_days: int) -> pd.DataFrame | None:
    fp = os.path.join(OI_CACHE, f"{tk}.parquet")
    if not os.path.exists(fp):
        return None
    s = pd.read_parquet(fp).sort_values(["option_chain_id", "date"])
    s["prev_oi"] = s.groupby("option_chain_id")["open_interest"].shift(1)
    s["is_alert"] = (s["day_prem"] > 250_000) & (s["day_vol"] > s["prev_oi"].fillna(1e18))
    g = s.groupby("date").agg(alerts=("is_alert", "sum"),
                              total_oi=("open_interest", "sum")).reset_index()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date").reset_index(drop=True)
    g["flow_alert"] = g["alerts"] > 0
    inc = g["total_oi"].diff() > 0
    run = inc * (inc.groupby((~inc).cumsum()).cumcount() + 1)
    g["oi_growth"] = run >= oi_days
    g["n_signals"] = g["flow_alert"].astype(int) + g["oi_growth"].astype(int)
    return g


def _underlying_daily(tk: str):
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    df = pl.read_parquet(p).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York")
    mo = et.dt.hour * 60 + et.dt.minute
    g = df.assign(d=et.dt.date)[(mo >= 570) & (mo <= 960)].groupby("d").agg(
        close=("close", "last"), high=("high", "max"), low=("low", "min"), open=("open", "first"))
    g.index = pd.to_datetime(g.index)
    return g.sort_index()


def run(args):
    rows = []
    for tk in args.tickers:
        sig = _daily_signals(tk, args.oi_days)
        ud = _underlying_daily(tk)
        if sig is None or ud is None:
            print(f"  {tk}: missing _oi_cache or historical parquet -- skipped")
            continue
        ud = ud.reindex(sorted(set(ud.index) | set(sig["date"]))).ffill()
        c = ud["close"]
        fwd = c.shift(-args.fwd) / c - 1
        # forward up/down range from close, for the drift-free baseline
        roll_hi = ud["high"].shift(-1).rolling(args.fwd).max().shift(-(args.fwd - 1))
        roll_lo = ud["low"].shift(-1).rolling(args.fwd).min().shift(-(args.fwd - 1))
        up_rng = roll_hi / c - 1
        dn_rng = roll_lo / c - 1
        m = sig.set_index("date")
        for d in m.index:
            if d not in c.index or not np.isfinite(fwd.get(d, np.nan)):
                continue
            rows.append(dict(tk=tk, date=d, n_signals=int(m.loc[d, "n_signals"]),
                             flow_alert=bool(m.loc[d, "flow_alert"]), oi_growth=bool(m.loc[d, "oi_growth"]),
                             fwd=float(fwd[d]), up=float(up_rng.get(d, np.nan)), dn=float(dn_rng.get(d, np.nan))))
    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit("no data")
    split = pd.Timestamp(args.split) if args.split else None

    print("=" * 92)
    print(f"  MULTI-SIGNAL SCANNER   cohort={','.join(args.tickers)}   fwd={args.fwd}d   OI-growth>={args.oi_days}d")
    print(f"  n={len(R)} ticker-days" + (f"   IS<{split.date()}<=OOS" if split else ""))
    print("=" * 92)
    base = R["fwd"].mean() * 100

    def _blk(name, sub):
        if len(sub) < 15:
            print(f"     {name:22} n={len(sub):>4}  (thin)"); return
        for lbl, s in ([("IS", sub[sub.date < split]), ("OOS", sub[sub.date >= split])] if split
                       else [("ALL", sub)]):
            if len(s) < 10:
                continue
            fr = s["fwd"].mean() * 100
            wr = (s["fwd"] > 0).mean() * 100
            # is |fwd| bigger than a random day? (breakout tell)
            ab = s["fwd"].abs().mean() * 100
            print(f"     {name:22} {lbl:4} n={len(s):>4}  fwdRet {fr:>+6.2f}%  win {wr:>4.0f}%  |fwd| {ab:>4.2f}%")

    print(f"\n  overall mean fwd return: {base:+.2f}%   (baseline for all cuts)\n")
    _blk("0 signals", R[R.n_signals == 0])
    _blk("1 signal", R[R.n_signals == 1])
    _blk("2 signals (SCANNER)", R[R.n_signals == 2])
    print()
    _blk("flow_alert only", R[R.flow_alert & ~R.oi_growth])
    _blk("oi_growth only", R[R.oi_growth & ~R.flow_alert])
    _blk("BOTH", R[R.flow_alert & R.oi_growth])
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=COHORT)
    ap.add_argument("--oi-days", type=int, default=3)
    ap.add_argument("--fwd", type=int, default=10)
    ap.add_argument("--split", help="YYYY-MM-DD IS/OOS split")
    a = ap.parse_args()
    a.tickers = [t.upper() for t in a.tickers]
    run(a)
