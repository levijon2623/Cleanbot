# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_structures.py
===================
Scores the four option structures from `build_structure_tape.py`, each against
ITS OWN random-entry-minute control, under both fill bounds.

TWO SEPARATE QUESTIONS, and they must not be conflated
    1. Does the SIGNAL work in this instrument?   real  vs  its own null
    2. Does the instrument MAKE MONEY?            real  vs  zero
    check_step1_redo already showed these can disagree violently: 87 of 150
    naked cells beat their null while only 2 made money. A structure that
    beats its control but still loses is a real finding about the signal and
    a non-finding about trading it.

WHY EACH STRUCTURE NEEDS ITS OWN NULL
    The four have wildly different variance and capital bases (naked ~$0.92 at
    risk, diagonal ~$4.17). A low-variance instrument will show a smaller loss
    than a high-variance one for reasons that have nothing to do with the
    signal. Comparing each to its own random-minute control removes exactly
    that, which is why the comparison is never structure-vs-structure on the
    raw mean.

THE FILL BOUNDS DECIDE WHAT COUNTS
    `mid` prices every leg at mid (optimistic); `cross` pays the ask on every
    buy and hits the bid on every sell, both ends (pessimistic). The truth is
    between, but a gap that exists only at `mid` is a statement about the fill
    model, not the market -- and on multi-leg structures that gap is doubled.
    Conclusions are read off `cross`.

INFERENCE
    Day-block bootstrap resampling whole SESSIONS, with real and null drawn
    from the SAME resampled days so the pairing is preserved. One position per
    ticker-day, so n counts days.

Usage:  python check_structures.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import SLICE_EDGES

PRE_LO = pd.Timestamp("2023-10-12").date()
DEPLOY_LO = SLICE_EDGES[0]
SPLIT = pd.Timestamp("2025-08-21").date()
STRUCTURES = ("naked", "debit_vert", "diagonal", "credit_vert")


def window(d):
    if d < DEPLOY_LO:
        return "PRE"
    return "IS" if d < SPLIT else "OOS"


def boot_gap(real, null, n=4000, seed=71):
    """Day-block bootstrap of mean(real) - mean(null).

    Real and null are resampled from the SAME day draw: they share the calendar
    exactly (every null entry sits on a day that produced a real trigger), so
    resampling them independently would break the pairing and overstate
    precision.
    """
    if real.empty or null.empty:
        return np.nan, np.nan, np.nan
    R = {d: g["pnl"].to_numpy() for d, g in real.groupby("date")}
    N = {d: g["pnl"].to_numpy() for d, g in null.groupby("date")}
    days = sorted(set(R) & set(N))
    if not days:
        return np.nan, np.nan, np.nan
    obs = real["pnl"].mean() - null["pnl"].mean()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = [days[k] for k in rng.choice(len(days), len(days), replace=True)]
        a = np.concatenate([R[d] for d in pick])
        b = np.concatenate([N[d] for d in pick])
        out.append(a.mean() - b.mean())
    return obs, *np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", default="_structure_tape.parquet")
    a = ap.parse_args()
    df = pd.read_parquet(a.tape)
    df["win"] = df["date"].map(window)

    print(f"STRUCTURE COMPARISON -- each against its own random-minute control")
    print(f"  {df['date'].nunique()} dates, {len(df):,} priced positions")
    print(f"  one position per ticker-day, held to EOD. n counts DAYS.\n")

    ntests = 0
    for fill in ("cross", "mid"):
        tag = ("PESSIMISTIC BOUND -- conclusions are read off this"
               if fill == "cross" else "OPTIMISTIC BOUND -- context only")
        print("=" * 112)
        print(f"  FILL = {fill.upper()}   ({tag})")
        print("=" * 112)
        print(f"  {'structure':13} {'n':>5} {'REAL':>8} {'NULL':>8} {'GAP':>8} "
              f"{'95% CI':>18}  {'PRE':>8} {'IS':>8} {'OOS':>8}")
        for st in STRUCTURES:
            g = df[(df["structure"] == st) & (df["fill"] == fill)]
            real = g[g["kind"] == "real"]
            null = g[g["kind"] == "null"]
            if real.empty:
                print(f"  {st:13} (none)")
                continue
            obs, c1, c2 = boot_gap(real, null)
            ntests += 1
            sig = np.isfinite(c1) and (c1 > 0 or c2 < 0)
            w = {k: real[real["win"] == k]["pnl"].mean() for k in ("PRE", "IS", "OOS")}
            print(f"  {st:13} {len(real):>5} {real['pnl'].mean()*100:>+7.1f}% "
                  f"{null['pnl'].mean()*100:>+7.1f}% {obs*100:>+7.1f}% "
                  f"[{c1*100:>+6.1f},{c2*100:>+6.1f}]{'*' if sig else ' '}  "
                  + " ".join(f"{w[k]*100:>+7.1f}%" for k in ("PRE", "IS", "OOS")))
        print()

    print("=" * 112)
    print("  HOW TO READ THIS")
    print("=" * 112)
    c = df[(df["fill"] == "cross") & (df["kind"] == "real")]
    prof = {st: c[c["structure"] == st]["pnl"].mean() for st in STRUCTURES
            if (c["structure"] == st).any()}
    pos = [s for s, v in prof.items() if v > 0]
    print(f"  structures PROFITABLE at the pessimistic bound: "
          f"{pos if pos else 'NONE'}")
    print(f"  Beating the null means the SIGNAL works in that instrument.")
    print(f"  Making money is a separate and stricter claim -- check the REAL")
    print(f"  column, not the GAP. On the naked long the two already disagree:")
    print(f"  87 of 150 grid cells beat their null and only 2 were profitable.")
    print(f"\n  {ntests} tests ({len(STRUCTURES)} structures x 2 fills) -> "
          f"~{ntests*0.05:.1f} false positives expected at 95%.")
    print(f"  NOTE: no holdout remains (PRESAMPLE_PLAN.md was spent 2026-09-12),")
    print(f"  so nothing here can be promoted to a deployment decision. A")
    print(f"  structure that looks good is a candidate for FORWARD paper")
    print(f"  testing, not for capital.")


if __name__ == "__main__":
    main()
