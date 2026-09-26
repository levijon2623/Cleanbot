# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
build_iv_surface.py
===================
Builds the ATM IMPLIED-VOL TERM STRUCTURE for the book's tickers from the silver
option lake, once, so the IVR / term-structure / vanna-x-dIV tests can all read
the same cache instead of each re-scanning 125 GB.

Source: lake/silver/option-contracts-1m/date=*/bars.parquet -- per-contract,
per-minute `iv_close` across the FULL expiry ladder (0,3,4,5,...,42,56,70,91 DTE
on a typical day).  501 sessions, 2024-08-20 .. 2026-08-21.

  _ivs_cache/{T}.parquet     (date, mod15, dte, iv, n)
      ATM implied vol = volume-unweighted mean `iv_close` over every contract
      (call AND put) within +/-`--band` of spot, per 15-minute bucket per expiry.
      Calls and puts are averaged together deliberately: at the same strike they
      differ only by put-call-parity noise and the borrow/dividend term, and
      averaging halves the quote noise on the thin back months.

Sanity filters, all of which matter on this data:
  * iv_close in (0.01, 5.0)   -- the lake carries a few 0.0 and 1e3 rows
  * >= `--min-n` contracts in the ATM band, else the bucket is dropped; a
    single stale quote on a back-month strike is otherwise the whole "IV"
  * underlying_close > 0

Nothing here is a test.  See `check_ivr_termstructure.py` for the hypotheses.

Usage:
  python build_iv_surface.py                 # all book tickers, resumable
  python build_iv_surface.py --tickers SPY QQQ --force
"""
from __future__ import annotations

import argparse
import glob
import os

import polars as pl

SILVER = "lake/silver/option-contracts-1m"
CACHE = "_ivs_cache"
BOOK = ["AVGO", "GLD", "IWM", "META", "MSFT", "NVDA", "QQQ", "SMH", "SPY"]


def build(tickers, band: float, min_n: int, force: bool):
    os.makedirs(CACHE, exist_ok=True)
    todo = [t for t in tickers
            if force or not os.path.exists(os.path.join(CACHE, f"{t}.parquet"))]
    if not todo:
        print("  all tickers cached; --force to rebuild")
        return
    parts = sorted(glob.glob(f"{SILVER}/date=*/bars.parquet"))
    print(f"  {len(parts)} silver dates -> {len(todo)} tickers: {' '.join(todo)}")

    frames = {t: [] for t in todo}
    for i, p in enumerate(parts, 1):
        lf = (
            pl.scan_parquet(p)
            .filter(
                pl.col("underlying_symbol").is_in(todo)
                & pl.col("iv_close").is_not_null()
                & (pl.col("iv_close") > 0.01) & (pl.col("iv_close") < 5.0)
                & (pl.col("underlying_close") > 0)
                & (((pl.col("strike") - pl.col("underlying_close")).abs()
                    / pl.col("underlying_close")) <= band)
            )
            .with_columns(
                dte=(pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days(),
                # CAST BEFORE MULTIPLYING. polars `dt.hour()` is Int8, so
                # `hour * 60` silently OVERFLOWS (09:30 -> 28, not 570) and the
                # whole column wraps into -128..127. Caught 2026-09-12.
                mod15=((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)) // 15) * 15,
                date=pl.col("minute_et").dt.date(),
            )
            .filter(pl.col("dte") >= 0)
            .group_by("underlying_symbol", "date", "mod15", "dte")
            .agg(iv=pl.col("iv_close").mean(), n=pl.len())
            .filter(pl.col("n") >= min_n)
        )
        df = lf.collect()
        for t in todo:
            sub = df.filter(pl.col("underlying_symbol") == t).drop("underlying_symbol")
            if sub.height:
                frames[t].append(sub)
        if i % 50 == 0:
            print(f"    {i}/{len(parts)}", flush=True)

    for t in todo:
        if not frames[t]:
            print(f"  {t}: NO DATA")
            continue
        out = pl.concat(frames[t]).sort("date", "mod15", "dte")
        fp = os.path.join(CACHE, f"{t}.parquet")
        out.write_parquet(fp)
        d = out["date"]
        print(f"  {t}: {out.height:>7} rows  {d.min()}..{d.max()}  "
              f"{out['date'].n_unique()} sessions  dte {out['dte'].min()}-{out['dte'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=BOOK)
    ap.add_argument("--band", type=float, default=0.015,
                    help="ATM moneyness band, +/- fraction of spot")
    ap.add_argument("--min-n", type=int, default=2,
                    help="min contracts in the band for a bucket to count")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.tickers, a.band, a.min_n, a.force)
