# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
measure_cushion_cap.py
======================
Derive sim_core.CUSHION_CAP for a ticker from the bronze tape.

The cap is HALF the mean within-minute range of the NBBO bid, in spreads: the
bid cannot cost you more in timing than it actually moved, and a trade arriving
at a random instant sits on average about half the range from the minute close.
See the CUSHION_CAP docstring in sim_core for why the FILL cost (0.028 spreads)
is not the right number and why the raw dispersion must not be added to it.

Only valid on dates carrying NBBO depth -- 2026-09-01 onward. Run this before
adding any new ticker to CUSHION_CAP; a missing entry means `botcap` silently
falls back to the uncapped 1.5 and the rule is NOT re-priced (which is how the
AMZN row in the first disabled-rule sweep came back as a spurious +0.0).

Usage:
  python measure_cushion_cap.py AMZN
  python measure_cushion_cap.py AMZN LULU TSLA
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import polars as pl

BRONZE = "lake/bronze/full-tape"


def main():
    tickers = [t.upper() for t in sys.argv[1:]]
    if not tickers:
        sys.exit("  usage: measure_cushion_cap.py TICKER [TICKER ...]")
    files = sorted(glob.glob(os.path.join(BRONZE, "*.parquet")))
    if not files:
        sys.exit("  no bronze partitions")

    frames = []
    for f in files:
        d = os.path.basename(f)[:-8]
        df = (pl.scan_parquet(f)
              .filter(pl.col("underlying_symbol").is_in(tickers))
              .select("underlying_symbol", "executed_at", "nbbo_bid", "nbbo_ask",
                      "expiry", "strike", "option_type", "size", "price")
              .collect())
        if df.is_empty():
            continue
        df = df.with_columns(
            pl.col("nbbo_bid").cast(pl.Float64),
            pl.col("nbbo_ask").cast(pl.Float64),
            pl.col("price").cast(pl.Float64),
            (pl.col("expiry").cast(pl.Date)
             - pl.lit(__import__("pandas").Timestamp(d).date())).dt.total_days().alias("dte"),
        ).filter(
            (pl.col("nbbo_ask") > pl.col("nbbo_bid")) & (pl.col("nbbo_bid") > 0)
            & (pl.col("dte") >= 0) & (pl.col("dte") <= 1)
        ).with_columns(
            ((pl.col("nbbo_bid") + pl.col("nbbo_ask")) / 2).alias("mid")
        ).filter(pl.col("mid") >= 0.30)
        if df.is_empty():
            continue
        frames.append(df.with_columns(
            pl.col("executed_at").dt.truncate("1m").alias("minute"),
            (pl.col("nbbo_ask") - pl.col("nbbo_bid")).alias("spread"),
            (pl.col("underlying_symbol") + "|" + pl.col("expiry").cast(pl.Utf8) + "|"
             + pl.col("strike").cast(pl.Utf8) + "|"
             + pl.col("option_type")).alias("con"),
        ))
        print(f"    {d}: {df.height:>9,} prints", flush=True)

    if not frames:
        print("  no prints for those tickers in the depth-bearing range")
        return
    T = pl.concat(frames)
    G = (T.group_by(["underlying_symbol", "con", "minute"])
         .agg(pl.col("nbbo_bid").max().alias("hi"),
              pl.col("nbbo_bid").min().alias("lo"),
              pl.col("spread").median().alias("sp"),
              pl.len().alias("n"))
         .filter((pl.col("n") >= 5) & (pl.col("sp") > 0))
         .with_columns(((pl.col("hi") - pl.col("lo")) / pl.col("sp")).alias("rng_sp"))
         ).to_pandas()
    P = T.to_pandas()

    print(f"\n  {'ticker':8} {'prints':>10} {'spread%':>9} {'med $':>7} "
          f"{'con-mins':>9} {'mean rng':>9} {'CAP':>6}")
    for tk in tickers:
        p = P[P["underlying_symbol"] == tk]
        g = G[G["underlying_symbol"] == tk]
        if p.empty or len(g) < 200:
            print(f"  {tk:8} {len(p):>10,} -- too few contract-minutes "
                  f"({len(g)}) to set a cap")
            continue
        sp_pct = (p["spread"] / p["mid"] * 100).median()
        cap = min(1.5, g["rng_sp"].mean() / 2.0)
        print(f"  {tk:8} {len(p):>10,} {sp_pct:>8.2f}% {p['spread'].median():>7.2f} "
              f"{len(g):>9,} {g['rng_sp'].mean():>9.2f} {cap:>6.2f}")
    print(f"\n  add to sim_core.CUSHION_CAP, then re-run check_fill_impact.")
    print(f"  the cap is floored at nothing and capped at 1.50 -- it may only")
    print(f"  ever LOWER the charge relative to `bot`.")


if __name__ == "__main__":
    main()
