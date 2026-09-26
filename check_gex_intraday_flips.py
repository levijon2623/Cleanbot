# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0"]
# ///
"""
check_gex_intraday_flips.py  --  Stage 2 of the GEX plan.

The daily backtest stamps every minute of a trading day with ONE GEX regime
(the sign of that day's net GEX). This script asks, from the recorded 1-minute
spot GEX: how often does the regime a 9:30 ET "morning poll" would see actually
FLIP sign later the same session, and when it flips, is it a brief wobble or
does it stay flipped into the close?

    rare + transient  ->  the daily label is a fine approximation. Stage 2 done;
                          keep the morning-poll design in bot_runner.
    common / sticky   ->  rebuild directional_flow_backtester to map the
                          per-minute spot GEX onto each signal instead of one
                          sign per day (and the live bot should re-poll intraday).

Input: lake/silver/spot-exposures-1m/  (built by
`uw_options_data_lake.py spot-gex-build`).  Field: gamma_per_one_percent_move_oi
(same field the live WS `gex:` stream and _net_gex_from_row use).

A sign change is only counted as a flip when |gamma| at that minute exceeds
--min-frac of the day's peak |gamma| -- a flip while gamma is ~0 is noise, the
same "fragile" filter check_gex_lookahead.py applies to daily data.

Usage:
    python check_gex_intraday_flips.py
    python check_gex_intraday_flips.py SPY MU TSLA
    python check_gex_intraday_flips.py --lake lake --min-frac 0.10
    python check_gex_intraday_flips.py --all-hours          # don't clip to RTH
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

GAMMA = "gamma_per_one_percent_move_oi"
RTH_OPEN_MOD = 9 * 60 + 30     # 09:30 ET, minutes past local midnight
RTH_CLOSE_MOD = 16 * 60        # 16:00 ET


def load_spot(lake: Path, tickers: list[str] | None, all_hours: bool) -> pl.DataFrame:
    root = lake / "silver" / "spot-exposures-1m"
    if not root.is_dir():
        sys.exit(f"not found: {root}  (run `uw_options_data_lake.py spot-gex-build` first)")
    # date comes from minute_et below; the hive date= partition column would just
    # collide, so don't materialise it.
    lf = pl.scan_parquet((root / "**" / "*.parquet").as_posix(), hive_partitioning=False)
    cols = lf.collect_schema().names()
    if GAMMA not in cols:
        sys.exit(f"{GAMMA} not in the spot dataset; columns: {cols}")
    lf = lf.select("ticker", "minute_et", pl.col(GAMMA).alias("g"))
    if tickers:
        lf = lf.filter(pl.col("ticker").is_in([t.upper() for t in tickers]))
    lf = lf.filter(pl.col("g").is_not_null()).with_columns(
        pl.col("minute_et").dt.date().alias("date"),
        (pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
         + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"),
    )
    if not all_hours:
        lf = lf.filter((pl.col("mod") >= RTH_OPEN_MOD) & (pl.col("mod") <= RTH_CLOSE_MOD))
    return lf.sort("ticker", "date", "minute_et").collect()


def per_day_flips(df: pl.DataFrame, min_frac: float) -> pl.DataFrame:
    g = ["ticker", "date"]
    df = df.with_columns(
        pl.col("g").first().over(g).alias("g0"),
        pl.col("g").abs().max().over(g).alias("gpeak"),
        pl.col("minute_et").first().over(g).alias("t0"),
    ).with_columns(
        pl.col("g").sign().alias("s"),
        pl.col("g0").sign().alias("s0"),
    ).with_columns(
        (
            (pl.col("s") != 0)
            & (pl.col("s") != pl.col("s0"))
            & (pl.col("g").abs() > min_frac * pl.col("gpeak"))
        ).alias("opp")
    )
    day = df.group_by(g).agg(
        pl.len().alias("mins"),
        pl.col("s0").first(),
        pl.col("s").last().alias("s_end"),
        pl.col("opp").any().alias("flipped"),
        pl.col("opp").sum().alias("opp_mins"),
        (pl.col("minute_et").filter(pl.col("opp")).first() - pl.col("t0").first())
        .dt.total_minutes().alias("ttf_min"),
    )
    return day.with_columns((pl.col("s_end") != pl.col("s0")).alias("ended_opposite"))


def summarize(day: pl.DataFrame) -> pl.DataFrame:
    return day.group_by("ticker").agg(
        pl.len().alias("days"),
        (100 * pl.col("flipped").mean()).round(1).alias("flip_%"),
        (100 * pl.col("ended_opposite").mean()).round(1).alias("end_opp_%"),
        (100 * pl.col("opp_mins").sum() / pl.col("mins").sum()).round(1).alias("opp_min_%"),
        pl.col("ttf_min").filter(pl.col("flipped")).median().round(0).alias("med_ttf_min"),
        (100 * (pl.col("s0") > 0).mean()).round(0).alias("morning_POS_%"),
    ).sort("ticker")


def verdict(row: dict) -> str:
    if row["flip_%"] < 5:
        return "daily label OK (flips rare)"
    if (row["end_opp_%"] or 0) >= 10 or (row["opp_min_%"] or 0) >= 15:
        return "INTRADAY GEX NEEDED (flips common & sticky)"
    return "borderline (flips happen but mostly transient)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", help="tickers to check (default: all in the dataset)")
    ap.add_argument("--lake", type=Path, default=Path("lake"))
    ap.add_argument("--min-frac", type=float, default=0.10,
                    help="ignore sign flips while |gamma| < this * the day's peak |gamma| (default 0.10)")
    ap.add_argument("--all-hours", action="store_true", help="do not clip to 09:30-16:00 ET")
    args = ap.parse_args()

    df = load_spot(args.lake, args.tickers or None, args.all_hours)
    if df.is_empty():
        sys.exit("no spot-GEX rows matched.")
    day = per_day_flips(df, args.min_frac)
    summ = summarize(day)

    span = df.select(pl.col("date").min().alias("lo"), pl.col("date").max().alias("hi")).row(0)
    print("=" * 78)
    print(f"  INTRADAY GEX FLIPS   field={GAMMA}   {span[0]} -> {span[1]}")
    print(f"  RTH {'clipped 09:30-16:00 ET' if not args.all_hours else 'OFF (all hours)'}"
          f"   noise filter: |gamma| > {args.min_frac:.0%} of daily peak")
    print("=" * 78)
    print("  flip_%      = trading days where the morning-poll GEX sign flips at least once")
    print("  end_opp_%   = days that CLOSE on the opposite sign to the morning poll")
    print("  opp_min_%   = share of RTH minutes sitting on the opposite sign")
    print("  med_ttf_min = median minutes from the open to the first flip (flip days only)")
    print()
    hdr = f"  {'ticker':<7}{'days':>6}{'flip_%':>9}{'end_opp_%':>11}{'opp_min_%':>11}{'med_ttf_min':>13}{'morn_POS_%':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in summ.iter_rows(named=True):
        print(f"  {row['ticker']:<7}{row['days']:>6}{row['flip_%']:>9}{row['end_opp_%']:>11}"
              f"{row['opp_min_%']:>11}{str(row['med_ttf_min']):>13}{str(row['morning_POS_%']):>12}"
              f"   {verdict(row)}")

    print()
    worst = summ.select(pl.col("flip_%").max()).item()
    sticky = summ.filter(pl.col("end_opp_%") >= 10)["ticker"].to_list()
    if worst < 5:
        print("  OVERALL: every ticker flips < 5% of days. The daily GEX label is a sound")
        print("           approximation -- keep the morning-poll design, Stage 2 is closed.")
    elif sticky:
        print(f"  OVERALL: {', '.join(sticky)} close on the opposite regime >=10% of days.")
        print("           Rebuild directional_flow_backtester with the per-minute spot GEX")
        print("           (join gamma sign at signal-time, not one sign per day), and have")
        print("           bot_runner re-poll GEX intraday rather than once at the open.")
    else:
        print("  OVERALL: flips occur but are mostly transient (intraday reverts by close).")
        print("           A per-signal spot-GEX check is worth adding as a FILTER; a full")
        print("           intraday-regime backtest rebuild is probably not warranted yet.")


if __name__ == "__main__":
    main()
