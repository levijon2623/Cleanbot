# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "httpx>=0.27.0", "python-dotenv", "tzdata"]
# ///
"""
backfill_presample.py
=====================
Extends the lake back to the API's history floor, 2023-10-12, one day at a time.

WHY A DRIVER AND NOT `uw_options_data_lake build <range>`
    That command loops the WHOLE range into bronze first, and `silver-build`
    is a separate second pass that does not delete bronze behind it. Bronze runs
    ~1.8 GB/day, so 214 days would want ~385 GB against 285 GB free -- it would
    fill the disk and die partway. This driver runs the established cycle
    per day instead: download -> bronze -> silver -> DELETE BRONZE.

ORDER OF WORK, and it matters
    Phase 1 (cheap, minutes): netprem + OHLC + daily GEX for the pre-sample
    range. `net-prem-ticks` is the TRIGGER series -- without it the backtest
    cannot run at all -- so it lands first and the expensive phase can be
    interrupted without wasting it.
    Phase 2 (expensive, hours): the full option tape, newest-first so that any
    interruption still leaves a CONTIGUOUS block butted against the existing
    2024-08-20 lake rather than an island.

SAFETY
    * resumable -- a day whose silver partition exists is skipped, so re-running
      after an interruption costs nothing
    * stops cleanly if free disk falls below --min-free-gb
    * bronze for a day is deleted as soon as that day's silver is written
    * 2025-04-04 and 2025-09-29 are KNOWN CORRUPT AT SOURCE (they are the two
      holes in the existing 2024-08-20..2026-08-21 lake). --retry-corrupt tries
      them once more; expect them to fail again.

Usage:
  python backfill_presample.py --phase 1              # trigger/underlying data
  python backfill_presample.py --phase 2 --confirm    # the tape (overnight)
  python backfill_presample.py --status
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

FLOOR = dt.date(2023, 10, 12)          # measured against the live API 2026-09-12
LAKE_START = dt.date(2024, 8, 20)      # first day already held
TICKERS = ["SPY", "QQQ", "IWM", "META", "NVDA", "MSFT", "SMH", "AVGO", "GLD"]
CORRUPT = [dt.date(2025, 4, 4), dt.date(2025, 9, 29)]


def _free_gb() -> float:
    return shutil.disk_usage(".").free / 1e9


def _days():
    from uw_options_data_lake import trading_days
    return trading_days(FLOOR, LAKE_START - dt.timedelta(days=1))


def status():
    from uw_options_data_lake import silver_partition_path, DEFAULT_LAKE
    d = _days()
    have = [x for x in d if silver_partition_path(DEFAULT_LAKE, x).exists()]
    miss = [x for x in d if x not in set(have)]
    print(f"  pre-sample window {FLOOR} .. {LAKE_START - dt.timedelta(days=1)}")
    print(f"  sessions in window : {len(d)}")
    print(f"  silver present     : {len(have)}")
    print(f"  remaining          : {len(miss)}")
    if miss:
        print(f"  next up (newest-first): {miss[-1]}  ... oldest {miss[0]}")
        print(f"  est. disk needed   : {len(miss)*0.187:.0f} GB   free now: {_free_gb():.0f} GB")
    bronze = Path("lake/bronze/full-tape")
    if bronze.is_dir():
        n = list(bronze.glob("*.parquet"))
        if n:
            print(f"  !! {len(n)} bronze file(s) left behind "
                  f"({sum(p.stat().st_size for p in n)/1e9:.1f} GB) -- "
                  f"normally deleted after silver")


def phase1(confirm: bool):
    """Trigger + underlying series. Cheap API calls, no bulk download."""
    py = [sys.executable, "uw_options_data_lake.py"]
    lo, hi = FLOOR.isoformat(), (LAKE_START - dt.timedelta(days=1)).isoformat()
    jobs = [
        ("netprem-build", [lo, hi, "--tickers", *TICKERS] + (["--confirm"] if confirm else [])),
        ("ohlc-build",    [lo, hi, "--tickers", *TICKERS] + (["--confirm"] if confirm else [])),
        ("gex-daily-build", ["--tickers", *TICKERS, "--timeframe", "10Y"]),
    ]
    for cmd, args in jobs:
        print(f"\n=== {cmd} {' '.join(args[:2])} ===", flush=True)
        r = subprocess.run(py + [cmd] + args)
        if r.returncode != 0:
            print(f"  !! {cmd} exited {r.returncode} -- stopping phase 1")
            return False
    return True


def phase2(min_free: float, limit: int | None, retry_corrupt: bool):
    """The option tape, newest-first, one day at a time, bronze deleted as we go."""
    from uw_options_data_lake import (Client, build_one, build_silver_one,
                                      silver_partition_path, bronze_path,
                                      DEFAULT_LAKE, NoDataForDate, FatalError)
    from dotenv import load_dotenv
    load_dotenv(encoding="utf-8-sig")
    key = os.getenv("UW_API_KEY")
    if not key:
        print("  UW_API_KEY missing"); return
    client = Client(key)

    todo = [d for d in _days() if not silver_partition_path(DEFAULT_LAKE, d).exists()]
    todo.sort(reverse=True)                     # newest-first: contiguous with the lake
    if retry_corrupt:
        todo = [d for d in CORRUPT
                if not silver_partition_path(DEFAULT_LAKE, d).exists()] + todo
    if limit:
        todo = todo[:limit]
    print(f"  {len(todo)} sessions to fetch, newest-first. free={_free_gb():.0f} GB", flush=True)

    ok = fail = 0
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        if _free_gb() < min_free:
            print(f"  STOPPING: free disk {_free_gb():.0f} GB < {min_free} GB floor")
            break
        try:
            build_one(client, d, DEFAULT_LAKE)
            res = build_silver_one(DEFAULT_LAKE, d)
            bp = bronze_path(DEFAULT_LAKE, d)
            if bp.exists():
                bp.unlink()                     # the established cycle
            if res is None and not silver_partition_path(DEFAULT_LAKE, d).exists():
                print(f"  {d} no silver produced"); fail += 1
            else:
                ok += 1
        except (NoDataForDate, FatalError) as e:
            print(f"  {d} SKIP ({type(e).__name__}: {str(e)[:90]})"); fail += 1
            bp = bronze_path(DEFAULT_LAKE, d)
            if bp.exists():
                bp.unlink()
        except Exception as e:
            print(f"  {d} ERROR ({type(e).__name__}: {str(e)[:90]})"); fail += 1
        if i % 5 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] ok={ok} fail={fail} "
                  f"free={_free_gb():.0f}GB  {el/60:.0f}m elapsed  "
                  f"eta {el/max(1,i)*(len(todo)-i)/3600:.1f}h", flush=True)
    print(f"\n  done: {ok} sessions added, {fail} failed/skipped, free={_free_gb():.0f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N sessions (use a small value to smoke-test)")
    ap.add_argument("--retry-corrupt", action="store_true")
    a = ap.parse_args()
    os.chdir(Path(__file__).parent)
    if a.status or not a.phase:
        status(); return
    if a.phase == 1:
        phase1(a.confirm)
    else:
        phase2(a.min_free_gb, a.limit, a.retry_corrupt)


if __name__ == "__main__":
    main()
