# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_cost_by_premium.py
========================
WHAT DOES IT COST TO TRADE A CHEAP OPTION? -- SPREAD AND COMMISSION BY PREMIUM.

WHY THIS BLOCKS THE FLOOR QUESTION
    check_strike_offset found ATM beats OTM+1 and OTM+2 on the current book, but
    the $0.50 entry floor removed 543 and 1,058 OTM candidates -- selectively
    from the CHEAP end, which is exactly where the convexity argument lives. The
    obvious next move is to lower the floor and re-run. That would be wrong
    today, because the fill model cannot price what it would let in:

      1  CUSHION_CAP is per-TICKER (SPY 1.50 ... GLD 0.40). It does not vary
         with premium, so a $0.30 contract is charged the same spreads as a
         $5.00 one -- in ROE terms a wildly different cost.
      2  COMMISSION_PCT is a FLAT 1.5% of premium, independent of contract
         count. Real commission is PER CONTRACT, and the allocator targets fixed
         premium (qty = target / per_contract_cost), so a $0.50 contract buys
         10x the contracts of a $5.00 one and pays 10x the commission on the
         same capital. The flat model undercharges cheap options by roughly the
         ratio of the premiums.
      3  The 27% entry-fill rate is pooled, not per-premium, and a wider
         relative spread fills worse.

    Lower the floor without fixing these and the sim will report that cheap OTM
    options are wonderful, for the same reason the retired 1.5-spread cushion
    reported that AVGO was hopeless: the fill assumption IS the result
    (METHODOLOGY 1a).

WHAT THIS MEASURES
    Straight off the bronze tape, for dte<=1 prints in the nine book tickers:
    the median spread in dollars and as a share of mid, per premium bucket, plus
    UW's own side tags. Then the round-trip cost in ROE terms under a
    per-contract commission, against what the deployed flat 1.5% charges.
    No modelling, no fitting -- these are quoted spreads on real prints.

Usage:
  python check_cost_by_premium.py
  python check_cost_by_premium.py --commission 0.65
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

BRONZE = "lake/bronze/full-tape"
TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "META", "AVGO", "SMH", "GLD", "MSFT"]
EDGES = [0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 1e9]
LABEL = ["$0.10-0.25", "$0.25-0.50", "$0.50-1.00", "$1.00-2.00",
         "$2.00-5.00", "$5.00+"]


def load(days):
    out = []
    for d in days:
        p = os.path.join(BRONZE, f"{d}.parquet")
        try:
            x = (pl.scan_parquet(p)
                 .filter(pl.col("underlying_symbol").is_in(TICKERS))
                 .select("underlying_symbol", "executed_at", "expiry", "price",
                         "nbbo_bid", "nbbo_ask", "size", "tags")
                 .collect().to_pandas())
        except Exception as e:
            print(f"    {d}: {type(e).__name__}")
            continue
        t = pd.to_datetime(x["executed_at"], utc=True).dt.tz_convert("America/New_York")
        x["mod"] = t.dt.hour * 60 + t.dt.minute
        # subtracting two .dt.date Series gives OBJECT dtype, and .dt.days then
        # fails -- normalise both to datetime64 first
        exp = pd.to_datetime(x["expiry"], errors="coerce").dt.normalize()
        day = t.dt.tz_localize(None).dt.normalize()
        x["dte"] = (exp - day).dt.days
        out.append(x[(x["dte"] <= 1) & (x["mod"].between(570, 955))])
        print(f"    {d}: {len(out[-1]):,} dte<=1 RTH prints", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commission", type=float, default=0.65,
                    help="$ per contract per side")
    a = ap.parse_args()

    days = sorted(os.path.basename(p)[:-8]
                  for p in glob.glob(os.path.join(BRONZE, "*.parquet")))
    if not days:
        raise SystemExit(f"  no bronze tape in {BRONZE}")
    print(f"  {len(days)} sessions on disk: {days[0]} .. {days[-1]}")
    df = load(days)
    for c in ("price", "nbbo_bid", "nbbo_ask", "size"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["nbbo_ask"] > df["nbbo_bid"]) & (df["nbbo_bid"] > 0)]
    df["mid"] = (df["nbbo_bid"] + df["nbbo_ask"]) / 2
    df["spread"] = df["nbbo_ask"] - df["nbbo_bid"]
    df["rel"] = df["spread"] / df["mid"]
    df["bucket"] = pd.cut(df["mid"], EDGES, labels=LABEL, right=False)
    tg = df["tags"].astype(str)
    df["at_ask"] = tg.str.contains("ask_side", na=False)
    df["at_bid"] = tg.str.contains("bid_side", na=False)

    print(f"\n  {len(df):,} prints with a two-sided NBBO\n")
    print(f"{'='*104}")
    print(f"  1. QUOTED SPREAD BY PREMIUM  (measured, not modelled)")
    print(f"{'='*104}")
    print(f"  {'bucket':>12} {'prints':>10} {'med mid':>9} {'med spread':>11} "
          f"{'spread % of mid':>16} {'p75 %':>8}")
    for b in LABEL:
        g = df[df["bucket"] == b]
        if len(g) < 500:
            continue
        print(f"  {b:>12} {len(g):>10,} {g['mid'].median():>9.2f} "
              f"{g['spread'].median():>11.2f} {g['rel'].median()*100:>15.1f}% "
              f"{g['rel'].quantile(.75)*100:>7.1f}%")

    print(f"\n{'='*104}")
    print(f"  2. ROUND-TRIP COST IN ROE TERMS  (what the trade must overcome)")
    print(f"{'='*104}")
    print(f"  Commission ${a.commission:.2f}/contract/side. The allocator targets")
    print(f"  FIXED PREMIUM, so a cheaper contract buys proportionally more of")
    print(f"  them -- commission per DOLLAR DEPLOYED therefore scales as 1/premium.")
    print(f"\n  {'bucket':>12} {'med mid':>9} {'spread cost':>12} {'commission':>11} "
          f"{'TOTAL':>8} {'deployed 1.5%':>14} {'under/over':>12}")
    for b in LABEL:
        g = df[df["bucket"] == b]
        if len(g) < 500:
            continue
        mid = g["mid"].median()
        # one spread round-trip is the honest floor for a marketable exit plus a
        # non-marketable entry; CUSHION_CAP says the exit alone runs 0.40-1.50
        sp_cost = g["rel"].median()
        comm = (2 * a.commission) / (mid * 100)
        tot = sp_cost + comm
        print(f"  {b:>12} {mid:>9.2f} {sp_cost*100:>11.1f}% {comm*100:>10.2f}% "
              f"{tot*100:>7.1f}% {1.5:>13.1f}% "
              f"{('UNDER by ' + f'{(tot-0.015)*100:.1f}pp') if tot > 0.015 else ('over by ' + f'{(0.015-tot)*100:.1f}pp'):>12}")

    print(f"\n{'='*104}")
    print(f"  3. WHERE THE $0.50 FLOOR SITS")
    print(f"{'='*104}")
    below = df[df["mid"] < 0.50]
    above = df[df["mid"] >= 0.50]
    print(f"  prints below $0.50: {len(below):,} ({len(below)/len(df)*100:.1f}%)   "
          f"median spread {below['rel'].median()*100:.1f}% of mid")
    print(f"  prints at/above   : {len(above):,} ({len(above)/len(df)*100:.1f}%)   "
          f"median spread {above['rel'].median()*100:.1f}% of mid")
    r = below["rel"].median() / above["rel"].median()
    print(f"  -> sub-floor contracts quote {r:.1f}x the relative spread")
    c_lo = (2 * a.commission) / (below["mid"].median() * 100)
    c_hi = (2 * a.commission) / (above["mid"].median() * 100)
    print(f"  -> and pay {c_lo/c_hi:.1f}x the commission per dollar deployed")
    print(f"  -> combined drag below the floor "
          f"{(below['rel'].median()+c_lo)*100:.1f}% vs "
          f"{(above['rel'].median()+c_hi)*100:.1f}% above it")

    print(f"\n  WHAT THIS MEANS FOR LOWERING THE FLOOR")
    print(f"  The deployed flat 1.5% is a single number standing in for a cost")
    print(f"  that varies several-fold across this range. Any test that lowers")
    print(f"  the floor while charging 1.5% is measuring the assumption, not the")
    print(f"  strikes -- which is the same error the retired 1.5-SPREAD exit")
    print(f"  cushion made in the other direction.")


if __name__ == "__main__":
    main()
