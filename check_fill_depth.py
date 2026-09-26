# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_fill_depth.py
===================
CALIBRATE sim_core.FILL_MODELS AGAINST REAL NBBO DEPTH -- complete sessions.

SUPERSEDES check_fill_calibration.py, which hit three limits that mattered:
  * it paged the per-contract API and every busy contract truncated at exactly
    20,000 prints, biasing the sample to whichever end of the day the endpoint
    returns first (all 38,000 IWM prints came back under $0.30 -- i.e. late-day
    decay -- while the bot had entered IWM at $0.62 that morning);
  * it could only reach contracts the bot had already traded;
  * it could not window by time of day.
Reading BRONZE removes all three: every print of every session, with the entry
window selectable.

WHAT IS BEING CALIBRATED, AND WHY IT MATTERS BEYOND ONE RULE
    sim_core prices a `bot` exit at   bid - spread x (0.5 win / 1.5 lose).
    Those cushions were never validated -- there was no depth in the tape until
    2026-09-01. They sit under every backtest in this project, including the
    conclusion that AVGO is a good signal whose edge is eaten by friction
    (mid-vs-bot swing of -558pp OOS). If the real cushion is near zero, that
    verdict and many others are understated.

🚨 COVERAGE: depth exists from 2026-09-01 ONLY, and is NOT retroactive
   (2026-08-21 serves 40 columns; 09-01..09-11 serve 47; 09-15 serves 49).
   So this calibrates the MODEL on ~10 sessions and the corrected cushion is
   then applied to history. It is NOT a re-pricing of the backtest.

THREE BIASES THAT REMAIN, none of which bronze fixes
  1. SURVIVORSHIP. The tape shows orders that FILLED. A seller resting at the
     bid who never got hit leaves no print, so the observed cushion is biased
     toward zero by construction. This bounds the cushion; it does not prove it.
  2. INFERRED SIDE. Direction is read from the print's position in the spread,
     which misclassifies trades arriving between quote updates.
  3. NOT OUR ORDER. These are other participants, many of them market makers
     whose fills we cannot replicate.
  Consequently the honest output is a RANGE, and the recommendation below is
  deliberately more conservative than the median the tape shows.

Usage:
  python check_fill_depth.py
  python check_fill_depth.py --tickers IWM QQQ SPY --entry-window 570 900
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

BRONZE = "lake/bronze/full-tape"
BOOK = ["SPY", "QQQ", "IWM", "NVDA", "META", "AVGO", "SMH", "GLD", "MSFT"]


def load(dates, tickers, max_dte, lo_mod, hi_mod, min_mid):
    frames = []
    for p in dates:
        d = os.path.basename(p)[:-8]
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol").is_in(tickers))
              .select("underlying_symbol", "executed_at", "price", "size",
                      "nbbo_bid", "nbbo_ask", "nbbo_bid_size", "nbbo_ask_size",
                      "expiry", "option_type", "strike"))
        df = lf.collect()
        if df.is_empty():
            continue
        df = df.with_columns(
            (pl.col("executed_at").dt.convert_time_zone("America/New_York")
             .dt.hour().cast(pl.Int32) * 60
             + pl.col("executed_at").dt.convert_time_zone("America/New_York")
             .dt.minute().cast(pl.Int32)).alias("mod"),
            (pl.col("expiry").cast(pl.Date)
             - pl.lit(pd.Timestamp(d).date())).dt.total_days().alias("dte"),
            pl.col("nbbo_bid").cast(pl.Float64),
            pl.col("nbbo_ask").cast(pl.Float64),
            pl.col("price").cast(pl.Float64),
        ).filter(
            (pl.col("mod") >= lo_mod) & (pl.col("mod") <= hi_mod)
            & (pl.col("dte") >= 0) & (pl.col("dte") <= max_dte)
            & (pl.col("nbbo_ask") > pl.col("nbbo_bid"))
            & (pl.col("nbbo_bid") > 0)
        )
        if df.is_empty():
            continue
        df = df.with_columns(
            ((pl.col("nbbo_bid") + pl.col("nbbo_ask")) / 2).alias("mid")
        ).filter(pl.col("mid") >= min_mid).with_columns(
            (pl.col("nbbo_ask") - pl.col("nbbo_bid")).alias("spread"),
            pl.lit(d).alias("date"),
        ).with_columns(
            (pl.col("spread") / pl.col("mid") * 100).alias("spread_pct"),
            ((pl.col("price") - pl.col("nbbo_bid")) / pl.col("spread")).alias("pos"),
        )
        frames.append(df)
        print(f"    {d}: {df.height:>9,} prints", flush=True)
    return pl.concat(frames) if frames else pl.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=BOOK)
    ap.add_argument("--max-dte", type=int, default=1)
    ap.add_argument("--entry-window", nargs=2, type=int, default=[570, 960],
                    help="minute-of-day range (570=09:30, 900=15:00, 960=16:00)")
    ap.add_argument("--min-mid", type=float, default=0.30)
    ap.add_argument("--max-size", type=int, default=5,
                    help="only prints this small are comparable to our 1-2 lots")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(BRONZE, "*.parquet")))
    if not files:
        print("  no bronze"); return
    print(f"  scanning {len(files)} sessions, tickers={' '.join(a.tickers)}, "
          f"dte<={a.max_dte}, mod {a.entry_window[0]}-{a.entry_window[1]}, "
          f"mid>=${a.min_mid}")
    T = load(files, a.tickers, a.max_dte, a.entry_window[0], a.entry_window[1],
             a.min_mid)
    if T.is_empty():
        print("  nothing"); return
    df = T.to_pandas()
    df.to_parquet("_fill_depth.parquet", index=False)
    print(f"\n  {len(df):,} prints across {df['date'].nunique()} sessions")

    print(f"\n{'='*96}")
    print(f"  1. SPREAD IN THE TRADEABLE RANGE  (% of mid)")
    print(f"{'='*96}")
    print(f"  {'ticker':7} {'prints':>9} {'p25':>7} {'median':>8} {'p75':>7} "
          f"{'p90':>7} {'med $':>7}")
    for tk, g in df.groupby("underlying_symbol"):
        print(f"  {tk:7} {len(g):>9,} {g['spread_pct'].quantile(.25):>7.2f} "
              f"{g['spread_pct'].median():>8.2f} {g['spread_pct'].quantile(.75):>7.2f} "
              f"{g['spread_pct'].quantile(.90):>7.2f} {g['spread'].median():>7.2f}")

    print(f"\n{'='*96}")
    print(f"  2. RESTING DEPTH vs OUR 1-2 LOT ORDER")
    print(f"{'='*96}")
    D = df.dropna(subset=["nbbo_bid_size"])
    print(f"  {'ticker':7} {'prints':>9} {'bid p10':>8} {'bid med':>8} "
          f"{'ask med':>8} {'>=2 lots':>9} {'>=10':>7}")
    for tk, g in D.groupby("underlying_symbol"):
        bs = pd.to_numeric(g["nbbo_bid_size"], errors="coerce")
        ks = pd.to_numeric(g["nbbo_ask_size"], errors="coerce")
        print(f"  {tk:7} {len(g):>9,} {bs.quantile(.10):>8.0f} {bs.median():>8.0f} "
              f"{ks.median():>8.0f} {(bs>=2).mean()*100:>8.0f}% "
              f"{(bs>=10).mean()*100:>6.0f}%")

    print(f"\n{'='*96}")
    print(f"  3. THE CUSHION -- small sells, (bid - price) / spread")
    print(f"     0 = filled at the bid.  sim_core charges 0.5 (win) / 1.5 (lose).")
    print(f"{'='*96}")
    s = df[(df["size"] <= a.max_size) & (df["pos"] < 0.5)].copy()
    s["cushion"] = (s["nbbo_bid"] - s["price"]) / s["spread"]
    print(f"  {'ticker':7} {'n':>8} {'median':>8} {'p75':>7} {'p90':>7} {'p99':>7} "
          f"{'worse than bid':>15}")
    for tk, g in s.groupby("underlying_symbol"):
        if len(g) < 100:
            continue
        print(f"  {tk:7} {len(g):>8,} {g['cushion'].median():>8.2f} "
              f"{g['cushion'].quantile(.75):>7.2f} {g['cushion'].quantile(.90):>7.2f} "
              f"{g['cushion'].quantile(.99):>7.2f} {(g['cushion']>0).mean()*100:>14.1f}%")
    if len(s) >= 100:
        print(f"  {'ALL':7} {len(s):>8,} {s['cushion'].median():>8.2f} "
              f"{s['cushion'].quantile(.75):>7.2f} {s['cushion'].quantile(.90):>7.2f} "
              f"{s['cushion'].quantile(.99):>7.2f} {(s['cushion']>0).mean()*100:>14.1f}%")

    print(f"\n{'='*96}")
    print(f"  4. DOES IT DEPEND ON TIME OF DAY?  (entries stop at 15:00)")
    print(f"{'='*96}")
    print(f"  {'window':14} {'n':>9} {'med spread%':>12} {'med cushion':>12} "
          f"{'bid depth med':>14}")
    for lo, hi, lbl in ((570, 660, "09:30-11:00"), (660, 780, "11:00-13:00"),
                        (780, 900, "13:00-15:00"), (900, 960, "15:00-16:00")):
        g = df[(df["mod"] >= lo) & (df["mod"] < hi)]
        gs = s[(s["mod"] >= lo) & (s["mod"] < hi)]
        if g.empty:
            continue
        bs = pd.to_numeric(g["nbbo_bid_size"], errors="coerce")
        print(f"  {lbl:14} {len(g):>9,} {g['spread_pct'].median():>12.2f} "
              f"{(gs['cushion'].median() if len(gs) else np.nan):>12.2f} "
              f"{bs.median():>14.0f}")

    print(f"\n  WHAT TO DO WITH THIS")
    med = s["cushion"].median() if len(s) else np.nan
    p90 = s["cushion"].quantile(.90) if len(s) else np.nan
    print(f"  tape median cushion {med:+.2f}, p90 {p90:+.2f}, against a model that")
    print(f"  charges 1.5 on losing exits.")
    print(f"  Do NOT simply set the cushion to the median: survivorship means")
    print(f"  unfilled resting orders never print, so the tape CANNOT show the cost")
    print(f"  of not getting filled. A defensible revision is the p90, which still")
    print(f"  prices most of the observable adverse cases, with the existing 1.5")
    print(f"  retained as the pessimistic bound in FILL_MODELS rather than deleted.")
    print(f"  Any change must be re-run across the whole book before it is believed.")


if __name__ == "__main__":
    main()
