# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "pandas>=2.0.0", "numpy>=1.26.0"]
# ///
"""
check_flow_drift.py
===================

directional_flow_backtester.py gates entries on a FIXED dollar `min_flow`
($3.5M, $7.5M, ...). Over a 2-year span the dollar size of "normal" options
flow drifts -- underlying price, IV, and the secular rise in options volume all
inflate `(ask_vol - bid_vol) * vwap * 100`. If it drifts, a fixed threshold
means something different in 2024 than in 2026, and any single value the grid
picks is wrong for half the window.

This measures it, per ticker, straight off the same signal stream the backtest
uses (build_flow_1m -> triggers_for). For each ticker it reports the
distribution of trigger `abs_flow` by calendar quarter, a first-vs-last drift
factor, and -- the number that actually matters -- what PERCENTILE each enabled
config `min_flow` sits at in the first quarter vs the last. If a $3.5M bar was
the 80th percentile early and the 45th percentile late, the threshold has
silently loosened to let ~2x more signals through.

Usage:
    python check_flow_drift.py
    python check_flow_drift.py --tickers SPY NVDA MU
    python check_flow_drift.py --monthly          # month-by-month p90 series
    python check_flow_drift.py --lake <silver option-contracts-1m dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WATCHLIST
from directional_flow_backtester import build_flow_1m, triggers_for, _naive, DEFAULT_LAKE

PCTS = (10, 25, 50, 75, 90, 99)


def trigger_frame(flow, ticker):
    trigs = triggers_for(flow, ticker)
    if not trigs:
        return pd.DataFrame(columns=["date", "abs_flow", "dir"])
    df = pd.DataFrame(trigs)[["date", "abs_flow", "dir"]]
    df["date"] = pd.to_datetime(df["date"])
    df["q"] = df["date"].dt.to_period("Q")
    df["m"] = df["date"].dt.to_period("M")
    return df


def pctile_of(arr, v):
    """percentile rank (0-100) of value v within arr; higher = stricter gate."""
    if len(arr) == 0 or v is None:
        return None
    return float((arr < v).mean() * 100.0)


def enabled_cells(ticker):
    out = []
    for regime in ("POSITIVE_GEX", "NEGATIVE_GEX"):
        for direction in ("CALL", "PUT"):
            c = WATCHLIST[ticker].get(regime, {}).get(direction, {})
            if c.get("enabled") and c.get("min_flow"):
                out.append((regime.split("_")[0], direction, float(c["min_flow"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=list(WATCHLIST))
    ap.add_argument("--lake", default=DEFAULT_LAKE)
    ap.add_argument("--monthly", action="store_true", help="also print the month-by-month p90 series")
    args = ap.parse_args()

    try:
        flow = build_flow_1m(args.lake)
    except Exception as e:
        sys.exit(f"could not load the flow cache (is --build-bars still running?): {e}")
    flow["minute_et"] = _naive(flow["minute_et"])
    flow["date"] = flow["minute_et"].dt.date
    span_lo, span_hi = min(flow["date"]), max(flow["date"])
    print(f"flow cache: {span_lo} -> {span_hi}  ({flow['date'].nunique()} trading days)\n")

    summary = []
    for tk in [t.upper() for t in args.tickers]:
        df = trigger_frame(flow, tk)
        if df.empty or df["q"].nunique() < 2:
            print(f"{tk}: too few triggers ({len(df)}) to assess drift\n")
            continue

        quarters = sorted(df["q"].unique())
        tbl = pd.DataFrame(
            {str(q): np.percentile(df.loc[df["q"] == q, "abs_flow"], PCTS) / 1e6 for q in quarters},
            index=[f"p{p}" for p in PCTS],
        )
        first_q, last_q = quarters[0], quarters[-1]
        a_first = df.loc[df["q"] == first_q, "abs_flow"].values
        a_last = df.loc[df["q"] == last_q, "abs_flow"].values
        p90_first = np.percentile(a_first, 90)
        p90_last = np.percentile(a_last, 90)
        drift = p90_last / p90_first if p90_first else float("nan")

        print("=" * 78)
        print(f"  {tk}   {len(df)} triggers   |   trigger abs_flow by quarter ($M)")
        print("=" * 78)
        with pd.option_context("display.float_format", lambda x: f"{x:8.2f}"):
            print(tbl.to_string())
        print(f"\n  p90 drift  {first_q} -> {last_q}:  ${p90_first/1e6:.1f}M -> ${p90_last/1e6:.1f}M   "
              f"= {drift:.2f}x")

        cells = enabled_cells(tk)
        if cells:
            print("\n  config min_flow -> percentile rank of the trigger distribution:")
            print(f"    {'cell':<16}{'min_flow':>10}{'  %ile '+str(first_q):>16}{'  %ile '+str(last_q):>16}   note")
            for regime, direction, mf in cells:
                pf = pctile_of(a_first, mf)
                pl_ = pctile_of(a_last, mf)
                note = ""
                if pf is not None and pl_ is not None:
                    d = pf - pl_
                    if d >= 12:
                        note = f"LOOSENED ~{d:.0f}pp (lets through more)"
                    elif d <= -12:
                        note = f"tightened ~{-d:.0f}pp"
                    else:
                        note = "stable"
                print(f"    {regime+'/'+direction:<16}{mf/1e6:>8.1f}M{('' if pf is None else f'{pf:6.0f}%'):>16}"
                      f"{('' if pl_ is None else f'{pl_:6.0f}%'):>16}   {note}")
        else:
            print("\n  (no enabled config cells with a min_flow)")

        if args.monthly:
            months = sorted(df["m"].unique())
            p90m = [np.percentile(df.loc[df["m"] == m, "abs_flow"], 90) / 1e6 for m in months]
            print("\n  monthly p90 ($M):")
            for m, v in zip(months, p90m):
                bar = "#" * int(round(v / max(p90m) * 40))
                print(f"    {m}  {v:7.1f}  {bar}")
        print()
        summary.append((tk, len(df), p90_first / 1e6, p90_last / 1e6, drift))

    if summary:
        print("=" * 78)
        print("  SUMMARY   (p90 of trigger abs_flow, first quarter -> last quarter)")
        print("=" * 78)
        print(f"  {'ticker':<8}{'triggers':>10}{'p90 first':>12}{'p90 last':>12}{'drift':>9}   verdict")
        for tk, n, pf, pl_, dr in summary:
            v = ("~flat" if 0.8 <= dr <= 1.25 else
                 f"{'UP' if dr > 1 else 'DOWN'} {dr:.1f}x -- fixed $ threshold is non-stationary")
            print(f"  {tk:<8}{n:>10}{pf:>11.1f}M{pl_:>11.1f}M{dr:>8.2f}x   {v}")
        print("\n  drift within 0.8-1.25x  -> fixed min_flow is fine, leave it.")
        print("  drift outside that      -> switch the grid to a trailing-percentile threshold.")


if __name__ == "__main__":
    main()
