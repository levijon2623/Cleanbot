"""
flow_units_calibration.py
=========================

Does the backtest's Silver-Lake flow reconstruction match Unusual Whales'
live net-prem-ticks feed? If yes, the config.py `min_flow` thresholds transfer
to live as-is. If it's a stable multiple, scale them. If it's all over the
place, UW's endpoint filters trades differently and the thresholds need
re-fitting against UW data.

For one or more tickers on one trading day it computes, minute by minute:
  - LAKE : sum over contracts of (ask_volume - bid_volume) * vwap * 100     (call +, put -)
           = exactly what true_options_simulator.get_lake_flow does
  - UW   : net_call_premium - net_put_premium from /net-prem-ticks?date=

...then reports the end-of-day totals, their ratio, the minute-level
correlation, and a verdict.

Run on the box that has the lake. Reads UW_API_KEY from .env.

Usage:
    python flow_units_calibration.py NVDA --date 2026-08-21
    python flow_units_calibration.py SPY NVDA AAPL --date 2026-08-20
    python flow_units_calibration.py NVDA --date 2026-08-21 --lake /path/to/lake/silver/option-contracts-1m
"""

import os
import sys
import argparse
import datetime as dt
from statistics import fmean, pstdev
from dotenv import load_dotenv
import requests

DEFAULT_LAKE = "lake/silver/option-contracts-1m"
UW_URL = "https://api.unusualwhales.com/api/stock/{t}/net-prem-ticks"


def fmt(n):
    return f"{n:,.0f}"


# ---------------------------------------------------------------------------
def lake_minute_flow(lake_dir, ticker, date_str):
    """{ 'HH:MM' (ET) : signed net premium } from the Silver bars for that date."""
    try:
        import polars as pl
    except ImportError:
        sys.exit("polars not installed - needed to read the lake.")

    part = os.path.join(lake_dir, f"date={date_str}", "bars.parquet")
    if not os.path.exists(part):
        # fall back to hive-scan of the whole tree
        part = lake_dir

    lf = pl.scan_parquet(part, hive_partitioning=True).filter(
        pl.col("underlying_symbol") == ticker
    )
    signed = (
        pl.when(pl.col("option_type") == "call")
        .then((pl.col("ask_volume") - pl.col("bid_volume")) * pl.col("vwap") * 100)
        .otherwise(-(pl.col("ask_volume") - pl.col("bid_volume")) * pl.col("vwap") * 100)
    )
    g = (
        lf.with_columns(signed.alias("f"))
        .with_columns(pl.col("minute_et").dt.strftime("%Y-%m-%d").alias("d"),
                      pl.col("minute_et").dt.strftime("%H:%M").alias("m"))
        .filter(pl.col("d") == date_str)
        .group_by("m")
        .agg(pl.col("f").sum())
        .sort("m")
        .collect()
    )
    return {row["m"]: float(row["f"]) for row in g.iter_rows(named=True)}


def uw_minute_flow(ticker, date_str, headers):
    """{ 'HH:MM' (ET) : net_call_premium - net_put_premium } from net-prem-ticks."""
    r = requests.get(UW_URL.format(t=ticker.upper()), headers=headers,
                     params={"date": date_str}, timeout=20)
    if r.status_code != 200:
        print(f"  {ticker}: UW HTTP {r.status_code} - {r.text[:200]}")
        return {}
    data = r.json().get("data", [])
    out = {}
    for tick in data:
        ts = tick.get("tape_time") or tick.get("time") or ""
        try:
            t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            et = t.astimezone(dt.timezone(dt.timedelta(hours=-4)))  # rough ET; minute bucket only
            key = et.strftime("%H:%M")
        except Exception:
            continue
        val = float(tick.get("net_call_premium", 0) or 0) - float(tick.get("net_put_premium", 0) or 0)
        out[key] = out.get(key, 0.0) + val
    return out


# ---------------------------------------------------------------------------
def compare(ticker, lake, uw):
    keys = sorted(set(lake) | set(uw))
    if not keys:
        print(f"  {ticker}: no data on either side.\n")
        return None

    lake_tot = sum(lake.values())
    uw_tot = sum(uw.values())

    # paired minutes only, for correlation + ratio
    paired = [(lake[k], uw[k]) for k in keys if k in lake and k in uw
              and abs(lake[k]) > 1 and abs(uw[k]) > 1]
    ratios = [u / l for l, u in paired if l != 0]

    corr = float("nan")
    if len(paired) > 3:
        ls = [p[0] for p in paired]
        us = [p[1] for p in paired]
        lm, um = fmean(ls), fmean(us)
        num = sum((a - lm) * (b - um) for a, b in paired)
        den = (sum((a - lm) ** 2 for a in ls) ** 0.5) * (sum((b - um) ** 2 for b in us) ** 0.5)
        corr = num / den if den else float("nan")

    print(f"=== {ticker}  ({len(keys)} minutes, {len(paired)} paired) ===")
    print(f"  EOD total  LAKE : {fmt(lake_tot)}")
    print(f"  EOD total  UW   : {fmt(uw_tot)}")
    eod_ratio = uw_tot / lake_tot if lake_tot else float("nan")
    print(f"  EOD ratio UW/LAKE .... {eod_ratio:.3f}")
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        q1, q3 = ratios[len(ratios) // 4], ratios[3 * len(ratios) // 4]
        print(f"  per-minute ratio ..... median {med:.3f}  IQR [{q1:.2f}, {q3:.2f}]")
    print(f"  minute correlation ... {corr:.3f}")

    verdict = "NEEDS REFIT"
    if abs(eod_ratio - 1.0) < 0.15 and corr > 0.8:
        verdict = "MATCH  -> thresholds transfer as-is"
    elif ratios and (q3 - q1) < 0.5 * abs(med) and corr > 0.7:
        verdict = f"STABLE FACTOR ~{eod_ratio:.2f}  -> multiply config min_flow by {1/eod_ratio:.2f}"
    print(f"  >>> {verdict}\n")
    return {"ticker": ticker, "lake": lake_tot, "uw": uw_tot, "ratio": eod_ratio,
            "corr": corr, "verdict": verdict}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (must exist in the lake)")
    ap.add_argument("--lake", default=DEFAULT_LAKE, help=f"silver option-contracts-1m dir (default {DEFAULT_LAKE})")
    args = ap.parse_args()

    load_dotenv()
    key = os.getenv("UW_API_KEY")
    if not key:
        sys.exit("UW_API_KEY not in .env")
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    if not os.path.exists(args.lake):
        sys.exit(f"lake dir not found: {args.lake}  (pass --lake)")

    rows = []
    for tk in args.tickers:
        tk = tk.upper()
        print(f"\n--- {tk} {args.date} ---")
        lake = lake_minute_flow(args.lake, tk, args.date)
        uw = uw_minute_flow(tk, args.date, headers)
        print(f"  lake minutes: {len(lake)}   uw minutes: {len(uw)}")
        r = compare(tk, lake, uw)
        if r:
            rows.append(r)

    print("#" * 70)
    print("  SUMMARY")
    print("#" * 70)
    for r in rows:
        print(f"  {r['ticker']:<6}  UW/LAKE {r['ratio']:.3f}  corr {r['corr']:.2f}  | {r['verdict']}")
    print()
    print("  Run this for 3-5 different days before trusting the verdict -- one day")
    print("  can be dominated by a single large print.")


if __name__ == "__main__":
    main()
