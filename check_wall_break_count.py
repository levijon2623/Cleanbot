# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_wall_break_count.py
=========================
HOW MANY EVENTS ARE THERE? -- power check BEFORE a wall-break study.

THE HYPOTHESIS (live IWM read, 2026-09-22)
    Price tests a call wall, it holds as resistance, volume does NOT fade across
    repeated tests, and then it breaks and squeezes. Mechanism: below the wall
    dealers are long gamma and suppress the move; above it the hedge flips and
    they buy into strength.

WHY COUNT FIRST
    check_flow_spike pre-committed five criteria to a hypothesis that turned out
    to have 93,837 events and no effect; check_multiplier spent a full build on
    something that came down to one session. The cheap question -- "is there
    enough here to measure?" -- was not asked first in either case. It is asked
    first here, and nothing else is measured until it is answered.

🚨 THE STANDING INSTRUCTION THIS RUNS AGAINST
    METHODOLOGY 7: "the edge is INFORMATIONAL, not MECHANICAL ... GEX
    conditioning was IS/OOS-inconsistent. Stop building tests that assume dealer
    hedging is the mechanism." Walls-as-magnets is already busted
    (check_magnet, check_gamma_walls), prior-day POC went 0/48 cells, and
    check_wpoc_gate closed at |gain| < 0.6bp on n=81,727. What is NEW here is
    the level's FAILURE conditioned on volume, not the level holding -- but the
    prior is heavily against, and this file exists to find out whether it is
    even worth arguing about.

DEFINITIONS -- explicit, because every one of them is a knob
    touch    a bar whose HIGH reaches within `tol` bp of the call wall while its
             CLOSE is still below it (the mirror for the put wall)
    session  touches are collapsed if within `gap` minutes of each other, so one
             long consolidation is ONE test, not forty
    break    a later bar CLOSES beyond the wall by more than `tol`
    walls come from _level_cache/walls_*.parquet -- profiled at 09:35 and frozen
    ON PURPOSE (build_level_tape), so they are knowable before any touch.

Usage:
  python check_wall_break_count.py
  python check_wall_break_count.py --tol 10 --gap 5
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

RTH0, RTH1 = 570, 955


def bars(tk):
    df = (pl.scan_parquet(f"historical/{tk}.parquet")
          .select("date", "minute_et", "high", "low", "close", "volume")
          .collect().to_pandas())
    t = pd.to_datetime(df["minute_et"])
    df["mod"] = t.dt.hour * 60 + t.dt.minute
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["mod"] >= RTH0) & (df["mod"] <= RTH1)]
    return {d: g.sort_values("mod") for d, g in df.groupby("date")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=5.0,
                    help="bp: how close a high must come to count as a touch")
    ap.add_argument("--gap", type=int, default=10,
                    help="minutes: touches closer than this are one test")
    # 🚨 A BREAK MUST GO SOMEWHERE AND STAY. The first pass called it a break if
    # a close ever cleared the wall by --tol, which on a $290 name is ~15c --
    # and returned an 89% break rate, rising to 95% with more tests. That is not
    # a squeeze, it is price nicking a level it was already sitting on, and more
    # tests simply means more minutes near it. Same failure as check_flow_spike's
    # "99% reclaim, median 1 minute".
    ap.add_argument("--min-ext", type=float, default=25.0,
                    help="bp beyond the wall a close must reach to be a break")
    ap.add_argument("--hold", type=int, default=5,
                    help="consecutive closes that must stay beyond it")
    a = ap.parse_args()

    fs = sorted(glob.glob("_level_cache/walls_*dte0-3.parquet"),
                key=os.path.getsize, reverse=True)
    if not fs:
        print("  no walls cache -- run build_level_tape.py"); return
    W = pl.scan_parquet(fs[0]).collect().to_pandas()
    W["date"] = pd.to_datetime(W["date"]).dt.date
    print(f"  walls: {fs[0]}")
    print(f"         {len(W):,} ticker-days, {W['date'].nunique()} sessions, "
          f"{sorted(W['ticker'].unique())}")
    print(f"  touch = high within {a.tol:.0f}bp of the wall, close still below")
    print(f"  tests collapsed within {a.gap}m\n")

    rows = []
    for tk, g in W.groupby("ticker"):
        B = bars(tk)
        for _i, r in g.iterrows():
            d = r["date"]
            bb = B.get(d)
            if bb is None or len(bb) < 60 or not np.isfinite(r.get("cw", np.nan)):
                continue
            for side, lvl in (("call", r["cw"]), ("put", r["pw"])):
                if not np.isfinite(lvl) or lvl <= 0:
                    continue
                band = lvl * a.tol / 1e4
                ext = lvl * a.min_ext / 1e4
                if side == "call":
                    hit = bb[(bb["high"] >= lvl - band) & (bb["close"] < lvl)]
                    beyond = (bb["close"] > lvl + ext).to_numpy()
                else:
                    hit = bb[(bb["low"] <= lvl + band) & (bb["close"] > lvl)]
                    beyond = (bb["close"] < lvl - ext).to_numpy()
                if hit.empty:
                    continue
                mods = sorted(hit["mod"].tolist())
                tests, last = [], -999
                for m in mods:
                    if m - last > a.gap:
                        tests.append(m)
                    last = m
                # a break is --hold CONSECUTIVE closes at least --min-ext beyond
                # the wall, and it must begin after the first test
                bmods = bb["mod"].to_numpy()
                first_brk, run = None, 0
                for i in range(len(bmods)):
                    if bmods[i] <= tests[0]:
                        run = 0
                        continue
                    run = run + 1 if beyond[i] else 0
                    if run >= a.hold:
                        first_brk = int(bmods[i - a.hold + 1])
                        break
                # how far did it actually run, from the wall, after the tests?
                after = bb[bb["mod"] > tests[0]]
                if len(after):
                    mfe = ((after["high"].max() - lvl) / lvl * 1e4 if side == "call"
                           else (lvl - after["low"].min()) / lvl * 1e4)
                else:
                    mfe = np.nan
                rows.append(dict(ticker=tk, date=d, side=side, level=lvl,
                                 spot=r["spot"], gamma_sign=r["gamma_sign"],
                                 n_tests=len(tests), first_test=tests[0],
                                 broke=first_brk is not None,
                                 brk_mod=first_brk, mfe_bp=mfe))
        print(f"    {tk} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  no touches at all"); return
    R.to_parquet("_wall_events.parquet", index=False)

    print(f"\n{'='*92}")
    print(f"  1. THE FUNNEL")
    print(f"{'='*92}")
    print(f"  {'':22} {'call wall':>12} {'put wall':>12} {'both':>10}")
    c, p = R[R["side"] == "call"], R[R["side"] == "put"]
    def row(lab, fn):
        print(f"  {lab:22} {fn(c):>12,} {fn(p):>12,} {fn(R):>10,}")
    row("ticker-days touched", len)
    row("  >=2 separate tests", lambda x: (x["n_tests"] >= 2).sum())
    row("  >=3 separate tests", lambda x: (x["n_tests"] >= 3).sum())
    row("of >=2, then BROKE", lambda x: ((x["n_tests"] >= 2) & x["broke"]).sum())
    row("of >=2, held", lambda x: ((x["n_tests"] >= 2) & ~x["broke"]).sum())

    print(f"\n{'='*92}")
    print(f"  2. BREAK RATE BY TEST COUNT  (does testing more predict breaking?)")
    print(f"{'='*92}")
    print(f"  {'tests':>6} {'n':>7} {'broke':>8} {'rate':>8}")
    for n in (1, 2, 3, 4):
        sel = R[R["n_tests"] == n] if n < 4 else R[R["n_tests"] >= 4]
        if len(sel) < 5:
            continue
        print(f"  {('>=4' if n == 4 else n):>6} {len(sel):>7,} "
              f"{sel['broke'].sum():>8,} {sel['broke'].mean()*100:>7.1f}%")

    print(f"\n{'='*92}")
    print(f"  3. THE MECHANISM'S OWN PREDICTION -- break rate by gamma sign")
    print(f"     negative gamma should amplify; if + and - match, the mechanism")
    print(f"     is wrong even if the pattern is real.")
    print(f"{'='*92}")
    m = R[R["n_tests"] >= 2]
    print(f"  {'gamma':>8} {'n':>7} {'broke':>8} {'rate':>8}")
    for s, gg in m.groupby("gamma_sign"):
        print(f"  {s:>8} {len(gg):>7,} {gg['broke'].sum():>8,} "
              f"{gg['broke'].mean()*100:>7.1f}%")

    print(f"\n{'='*92}")
    print(f"  3b. HOW FAR DID IT RUN?  (max excursion beyond the wall, bp)")
    print(f"      A 'break' that goes 25bp and stops is not the event described.")
    print(f"{'='*92}")
    print(f"  {'group':22} {'n':>7} {'p50':>8} {'p75':>8} {'p90':>8}")
    for lab, sel in (("broke", m[m["broke"]]), ("held", m[~m["broke"]])):
        v = sel["mfe_bp"].dropna()
        if len(v):
            print(f"  {lab:22} {len(v):>7,} {v.quantile(.5):>8.0f} "
                  f"{v.quantile(.75):>8.0f} {v.quantile(.9):>8.0f}")

    print(f"\n{'='*92}")
    print(f"  4. POWER")
    print(f"{'='*92}")
    n2 = int((R['n_tests'] >= 2).sum())
    nb = int(((R['n_tests'] >= 2) & R['broke']).sum())
    print(f"  events with >=2 tests:            {n2:>6,}")
    print(f"  of which broke:                   {nb:>6,}")
    print(f"  per ticker per year (approx):     "
          f"{n2 / max(W['ticker'].nunique(), 1) / (W['date'].nunique() / 252):>6.0f}")
    print(f"\n  A volume conditioner splits these again -- roughly in half if it")
    print(f"  is a median cut. Judge whether the smaller half is still worth")
    print(f"  measuring BEFORE anything else is built.")


if __name__ == "__main__":
    main()
