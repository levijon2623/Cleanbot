# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_presample_ready.py
========================
Prerequisites 2-4 of PRESAMPLE_PLAN.md. Run this BEFORE the plan is signed.

THIS SCRIPT DELIBERATELY COMPUTES NO TRIGGERS, NO TRADES AND NO P&L.
    It only asks whether the DATA is present and structurally comparable to the
    2024-08-20+ lake. Nothing it prints reveals whether the book made money in
    the pre-sample window, so running it does not burn the holdout. Any check
    that would require simulating a trade belongs in the scored pass, after
    sign-off -- do not add one here.

  P2  coverage      every session has netprem + OHLC + GEX + a silver partition
  P3  0DTE mix      share of sessions offering a same-day expiry per ticker,
                    pre-sample vs current lake. The rules are 0/1DTE; SPY/QQQ/IWM
                    had dailies through this window, but META/NVDA depend on a
                    weekly landing on the trade day. A large drop would mean the
                    book is STRUCTURALLY different there, not merely unluckier,
                    and T1 would be measuring a different strategy.
  P4  IV integrity  the lake's `iv_close` is corrupt for dte==0 from ~2024-10 to
                    ~2025-12 (see check_ivr_termstructure.valid_0dte). Establish
                    whether the pre-sample window is clean before any
                    IV-dependent number is quoted from it.

Usage:  python check_presample_ready.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import polars as pl

from check_config_walkforward import PRESAMPLE_EDGES, SLICE_EDGES, _pre_slice_idx
from uw_options_data_lake import trading_days, silver_partition_path, DEFAULT_LAKE

HIST = "historical"
TK = ["SPY", "QQQ", "IWM", "META", "NVDA"]
LO, HI = PRESAMPLE_EDGES[0], SLICE_EDGES[0] - pd.Timedelta(days=1).to_pytimedelta()


def _dates(path, col_candidates=("tape_time", "start_time")):
    """Set of calendar dates present in a series file.

    TZ HANDLING IS NOT OPTIONAL HERE. `NETPREM*`/OHLC store a UTC INSTANT
    (`tape_time`/`start_time`) which must be converted to ET before taking the
    date. `GEX*` stores an already-local `datetime.date` -- running that through
    `to_datetime(utc=True).tz_convert("America/New_York")` reads midnight UTC and
    hands back the PREVIOUS day, which is exactly what made GEX look like it
    covered 166 of 214 pre-sample sessions when it actually covers all 214.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    col = next((c for c in col_candidates if c in df.columns), None)
    if col is None:
        return set()
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        t = pd.to_datetime(s, utc=True).dt.tz_convert("America/New_York")
        return set(t.dt.date)
    return set(pd.to_datetime(s).dt.date)          # already-local date column


def p2_coverage(days):
    print(f"\n{'='*96}\n  P2  COVERAGE  {LO} .. {HI}  ({len(days)} sessions)\n{'='*96}")
    silver = [d for d in days if silver_partition_path(DEFAULT_LAKE, d).exists()]
    print(f"  silver partitions : {len(silver)}/{len(days)}"
          + ("" if len(silver) == len(days) else "   <- backfill still running"))
    gex = _dates(f"{HIST}/GEXSPY.parquet", ("date", "start_time"))
    rows = []
    for tk in TK:
        npd = _dates(f"{HIST}/NETPREM{tk}.parquet") or set()
        ohl = _dates(f"{HIST}/{tk}.parquet") or set()
        g = _dates(f"{HIST}/GEX{tk}.parquet", ("date", "start_time")) or set()
        rows.append((tk, len([d for d in days if d in npd]),
                     len([d for d in days if d in ohl]),
                     len([d for d in days if d in g])))
    print(f"  {'ticker':7} {'netprem':>9} {'ohlc':>7} {'gex':>7}   (of "
          f"{len(days)} sessions)")
    bad = []
    for tk, a, b, c in rows:
        flag = "" if min(a, b) == len(days) else "  <- GAP"
        if flag:
            bad.append(tk)
        print(f"  {tk:7} {a:>9} {b:>7} {c:>7}{flag}")
    if not bad:
        print("  -> netprem and OHLC complete for every rule ticker.")
    # slice spread
    cnt = {}
    for d in days:
        k = _pre_slice_idx(d)
        cnt[k] = cnt.get(k, 0) + 1
    print("  pre-sample slices : "
          + "  ".join(f"P{k+1}={v}" for k, v in sorted(cnt.items()) if k is not None))
    return len(silver) == len(days) and not bad


def _expiry_share(dates, tk):
    """Share of sessions on which a same-day (0DTE) contract exists for `tk`."""
    hit = tot = 0
    for d in dates:
        p = silver_partition_path(DEFAULT_LAKE, d)
        if not p.exists():
            continue
        tot += 1
        n = (pl.scan_parquet(p)
             .filter((pl.col("underlying_symbol") == tk)
                     & (pl.col("expiry") == pl.col("minute_et").dt.date()))
             .select(pl.len()).collect().item())
        hit += int(n > 0)
    return hit, tot


def p3_zero_dte(days):
    print(f"\n{'='*96}\n  P3  0DTE AVAILABILITY -- is the book structurally the "
          f"same there?\n{'='*96}")
    cur = trading_days(SLICE_EDGES[0], SLICE_EDGES[-1])
    pre = [d for d in days if silver_partition_path(DEFAULT_LAKE, d).exists()]
    if len(pre) < 20:
        print(f"  only {len(pre)} pre-sample partitions present -- re-run when "
              f"the backfill finishes")
        return None
    step = max(1, len(cur) // 60)
    curs = cur[::step]
    print(f"  sampled {len(pre)} pre-sample vs {len(curs)} current sessions")
    print(f"  {'ticker':7} {'pre-sample':>12} {'current':>10}   verdict")
    for tk in TK:
        a, at = _expiry_share(pre, tk)
        b, bt = _expiry_share(curs, tk)
        pa = a / at * 100 if at else float("nan")
        pb = b / bt * 100 if bt else float("nan")
        d = pa - pb
        v = ("comparable" if abs(d) < 10 else
             "LOWER pre-sample" if d < 0 else "higher pre-sample")
        print(f"  {tk:7} {pa:>11.1f}% {pb:>9.1f}%   {v} ({d:+.1f}pp)")
    print("  -> a drop >10pp means the rule cannot pick the same contract there;")
    print("     T1 would then be scoring a different strategy, not the same one.")


def p4_iv(days):
    print(f"\n{'='*96}\n  P4  IV INTEGRITY (dte==0) -- is the pre-sample window "
          f"clean?\n{'='*96}")
    pre = [d for d in days if silver_partition_path(DEFAULT_LAKE, d).exists()]
    if not pre:
        print("  no pre-sample partitions yet")
        return
    step = max(1, len(pre) // 14)
    print(f"  {'date':12} {'iv dte0':>9} {'iv dte1-3':>11}  ratio   verdict")
    bad = 0
    for d in pre[::step]:
        p = silver_partition_path(DEFAULT_LAKE, d)
        lf = (pl.scan_parquet(p)
              .filter((pl.col("underlying_symbol") == "SPY")
                      & pl.col("iv_close").is_not_null() & (pl.col("iv_close") > 0)
                      & (((pl.col("strike") - pl.col("underlying_close")).abs()
                          / pl.col("underlying_close")) <= 0.015))
              .with_columns(dte=(pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days())
              .select("dte", "iv_close").collect().to_pandas())
        z = lf[lf["dte"] == 0]["iv_close"]
        r = lf[lf["dte"].between(1, 3)]["iv_close"]
        if not len(z) or not len(r):
            print(f"  {str(d):12} {'--':>9} {'--':>11}   (no 0DTE that session)")
            continue
        ratio = z.median() / r.median()
        ok = ratio >= 0.5
        bad += (not ok)
        print(f"  {str(d):12} {z.median():>9.3f} {r.median():>11.3f}  "
              f"{ratio:>5.2f}   {'clean' if ok else 'CORRUPT'}")
    print(f"  -> {'window looks CLEAN' if not bad else f'{bad} sampled sessions CORRUPT'}"
          f"  (corruption elsewhere runs ~2024-10..2025-12)")


def main():
    days = trading_days(LO, HI)
    print(f"PRE-SAMPLE READINESS -- no triggers, no trades, no P&L computed.")
    ok = p2_coverage(days)
    p3_zero_dte(days)
    p4_iv(days)
    print(f"\n{'='*96}")
    print(f"  P1 slice edges: PRESAMPLE_EDGES added, S1..S6 unchanged, "
          f"labels P1..P3 via slice_label()  [done]")
    print(f"  P2 coverage   : {'READY' if ok else 'INCOMPLETE — see gaps above'}")
    print("  Sign PRESAMPLE_PLAN.md before anything in this window is scored.")


if __name__ == "__main__":
    main()
