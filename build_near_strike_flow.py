"""
build_near_strike_flow.py
=========================
Extracts the near-the-money, nearest-two-expiry slice of the silver lake for
SPY / QQQ / IWM into _nsf_cache/, one pair of files per session.

This is plumbing for check_strike_crossing_flow.py -- it measures nothing.

WHAT IS KEPT
    opt/{date}.parquet   every contract-minute on the two nearest listed
                         expiries (dte_rank 0 and 1) with a strike within 2% of
                         the day's spot range. Side-split volume, multi-leg
                         volume, NBBO close, delta, OI.
    spot/{date}.parquet  one row per ticker-minute: spot reconstructed as the
                         MEDIAN underlying_close over every row of that ticker in
                         that minute, all expiries, and how many rows voted.

🚨 dte_rank, NOT "0DTE". IWM did not have a daily expiry for the whole window,
so on some sessions its nearest expiry is days away. Labelling rank 0 as "0DTE"
would silently mix same-day and multi-day contracts under one name. Both the
rank and the true calendar `dte_days` are stored; the study filters on the
latter.

🚨 SPOT IS RECONSTRUCTED, SO IT IS CHECKED. underlying_close on an option row is
the underlying's price at that contract's last print, which can be stale on a
quiet contract. The median over hundreds of rows per minute should be robust to
that, but "should" is not a measurement: check_strike_crossing_flow.py compares
it against historical/{T}.parquet (an independent 1-minute OHLC feed) before
anything else runs, per METHODOLOGY 6d.

Usage:  python build_near_strike_flow.py [--start 2023-10-12] [--end 2026-09-18]
        (skips sessions already cached; --force rebuilds)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import time

import polars as pl

LAKE = os.path.join("lake", "silver", "option-contracts-1m")
OUT = "_nsf_cache"
TICKERS = ["SPY", "QQQ", "IWM"]
BAND = 0.02            # keep strikes within 2% beyond the day's spot range

KEEP = ["underlying_symbol", "option_type", "strike", "expiry", "minute_et",
        "volume", "trade_count", "premium", "ask_volume", "bid_volume",
        "mid_volume", "no_side_volume", "multi_volume", "bid_close",
        "ask_close", "underlying_close", "delta_close", "open_interest"]


def build_day(path: str, day: dt.date) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = (pl.scan_parquet(path)
           .filter(pl.col("underlying_symbol").is_in(TICKERS))
           .select(KEEP)
           .collect())
    if raw.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    spot = (raw.group_by("underlying_symbol", "minute_et")
            .agg(pl.col("underlying_close").median().alias("spot"),
                 pl.len().alias("n_rows"))
            .sort("underlying_symbol", "minute_et"))

    # nearest two listed expiries on/after the session date, per ticker
    exps = (raw.filter(pl.col("expiry") >= day)
            .select("underlying_symbol", "expiry").unique()
            .sort("underlying_symbol", "expiry")
            .with_columns(pl.col("expiry").rank("dense")
                          .over("underlying_symbol").cast(pl.Int8)
                          .sub(1).alias("dte_rank"))
            .filter(pl.col("dte_rank") <= 1))

    rng = spot.group_by("underlying_symbol").agg(
        (pl.col("spot").min() * (1 - BAND)).alias("lo"),
        (pl.col("spot").max() * (1 + BAND)).alias("hi"))

    opt = (raw.join(exps, on=["underlying_symbol", "expiry"])
           .join(rng, on="underlying_symbol")
           .filter(pl.col("strike").is_between(pl.col("lo"), pl.col("hi")))
           .drop("lo", "hi")
           .with_columns((pl.col("expiry") - pl.lit(day)).dt.total_days()
                         .cast(pl.Int16).alias("dte_days"))
           .rename({"underlying_symbol": "ticker"}))
    return opt, spot.rename({"underlying_symbol": "ticker"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-10-12")
    ap.add_argument("--end", default="2099-01-01")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.join(OUT, "opt"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "spot"), exist_ok=True)

    days = sorted(glob.glob(os.path.join(LAKE, "date=*", "bars.parquet")))
    t0, n = time.time(), 0
    for p in days:
        d = dt.date.fromisoformat(p.split("date=")[1][:10])
        if not (a.start <= d.isoformat() <= a.end):
            continue
        fo = os.path.join(OUT, "opt", f"{d}.parquet")
        fs = os.path.join(OUT, "spot", f"{d}.parquet")
        if not a.force and os.path.exists(fo) and os.path.exists(fs):
            continue
        opt, spot = build_day(p, d)
        if opt.is_empty():
            print(f"  {d}  no rows for {TICKERS}")
            continue
        opt.write_parquet(fo)
        spot.write_parquet(fs)
        n += 1
        if n % 50 == 0:
            print(f"  {n} sessions  {time.time() - t0:.0f}s")
    print(f"done: {n} sessions built in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
