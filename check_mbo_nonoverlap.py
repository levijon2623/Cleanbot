# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_mbo_nonoverlap.py
=======================
THE DECIDING TEST for `tick_sgn`.

`check_mbo_horizon` showed tick_sgn at 120m clearing every pre-committed
criterion on RTY: +51.89bp, IS +67.92 / OOS +27.74, day-block p=0.003, and it
beat the time-of-day control. That is 5x the 9bp perp cost bar.

**I do not trust it, for a reason internal to my own test.** The day-block
bootstrap resampled days and then shuffled outcomes WITHIN each day. That
destroys the autocorrelation of overlapping forward windows -- real 120m returns
at adjacent minutes overlap ~99%, shuffled ones do not. A null that is easier to
beat than reality UNDERSTATES p. The criterion was written before seeing the
result, which makes it honest, not correct.

THE FIX: USE ONLY NON-OVERLAPPING WINDOWS
-----------------------------------------
RTH is ~390 minutes, so a 120m horizon admits exactly 3 disjoint windows per day
(09:30->11:30, 11:30->13:30, 13:30->15:30). Across 43 RTY days that is ~129
genuinely independent observations rather than 16,812 overlapping ones.

Too few for deciles, so the split is the MEDIAN. A smaller, cleaner test beats a
large contaminated one.

PHASES: the choice of where to start the first window is arbitrary, so every
offset is run as its OWN internally-non-overlapping sample. If the effect is
real it should appear in most phases; if it is an artefact of one particular
alignment it will scatter. Phases are NOT pooled -- pooling would reintroduce
the overlap this test exists to remove.

WHAT WOULD CONVINCE ME
  * the median-split gap stays materially positive on phase 0
  * it holds the same sign in most phases
  * IS and OOS agree
  * it clears the 9bp perp bar on independent data
If the effect collapses here, the overlap was carrying it and `tick_sgn` joins
the rest of the MBO thread.

Usage:  python check_mbo_nonoverlap.py
        python check_mbo_nonoverlap.py --feature imb --horizon 60
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

CACHE = "_mbo_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
PERP_BAR = 9.0


def load(sym, h):
    import polars as pl
    p = os.path.join(CACHE, f"{sym}.parquet")
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d.sort_values(["date", "mod"]).reset_index(drop=True)
    d[f"fwd{h}"] = d.groupby("date")["mid"].shift(-h) / d["mid"] - 1.0
    return d


def median_gap(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 20:
        return np.nan, 0
    med = np.median(x)
    hi, lo = y[x > med], y[x <= med]
    if len(hi) < 8 or len(lo) < 8:
        return np.nan, 0
    return float(np.mean(hi) - np.mean(lo)), len(x)


def run(a):
    rng = np.random.default_rng(23)
    h, feat = a.horizon, a.feature
    syms = a.symbols or [f[:-8] for f in os.listdir(CACHE) if f.endswith(".parquet")]

    for sym in sorted(syms):
        d = load(sym, h)
        if d is None:
            continue
        col = f"fwd{h}"
        print("\n" + "=" * 100)
        print(f"  {sym}   feature={feat}   horizon={h}m   NON-OVERLAPPING WINDOWS ONLY")
        print(f"  RTH {RTH_HI - RTH_LO} min -> {(RTH_HI - RTH_LO)//h} disjoint windows/day")
        print("=" * 100)
        print(f"  {'phase':>6} {'n':>5} {'days':>5} {'gap(bp)':>9} {'IS':>8} {'OOS':>8} "
              f"{'hi_mean':>9} {'lo_mean':>9}  clears9bp")
        gaps = []
        for off in range(0, h, max(h // 6, 1)):
            mods = list(range(RTH_LO + off, RTH_HI - h + 1, h))
            s = d[d["mod"].isin(mods)]
            s = s[np.isfinite(s[col])]
            if s.empty:
                continue
            g, n = median_gap(s[feat].to_numpy(float), s[col].to_numpy(float))
            if not np.isfinite(g):
                continue
            gi, _ = median_gap(s[s.date < SPLIT][feat].to_numpy(float),
                               s[s.date < SPLIT][col].to_numpy(float))
            go, _ = median_gap(s[s.date >= SPLIT][feat].to_numpy(float),
                               s[s.date >= SPLIT][col].to_numpy(float))
            x = s[feat].to_numpy(float); y = s[col].to_numpy(float)
            med = np.median(x)
            gaps.append(g)
            print(f"  {off:>6} {n:>5} {s['date'].nunique():>5} {g*1e4:>+9.2f} "
                  f"{gi*1e4 if np.isfinite(gi) else np.nan:>+8.2f} "
                  f"{go*1e4 if np.isfinite(go) else np.nan:>+8.2f} "
                  f"{np.mean(y[x>med])*1e4:>+9.2f} {np.mean(y[x<=med])*1e4:>+9.2f}  "
                  f"{'YES' if abs(g)*1e4 > PERP_BAR else 'no'}")
        if gaps:
            gaps = np.array(gaps)
            pos = (gaps > 0).mean()
            print(f"\n  across {len(gaps)} phases: mean {gaps.mean()*1e4:>+7.2f}bp   "
                  f"median {np.median(gaps)*1e4:>+7.2f}bp   "
                  f"sd {gaps.std()*1e4:>6.2f}bp   {pos*100:.0f}% positive")
            # day-block bootstrap on phase 0, resampling DAYS (no within-day shuffle)
            mods0 = list(range(RTH_LO, RTH_HI - h + 1, h))
            s0 = d[d["mod"].isin(mods0)]
            s0 = s0[np.isfinite(s0[col])]
            obs, _ = median_gap(s0[feat].to_numpy(float), s0[col].to_numpy(float))
            days = s0["date"].unique()
            by = {k: g for k, g in s0.groupby("date")}
            hits = 0
            for _ in range(a.boot):
                pick = rng.choice(days, size=len(days), replace=True)
                ss = pd.concat([by[k] for k in pick], ignore_index=True)
                # permute the FEATURE across days -- breaks the link without
                # touching the outcome series' own structure
                ss[feat] = rng.permutation(ss[feat].to_numpy())
                g2, _ = median_gap(ss[feat].to_numpy(float), ss[col].to_numpy(float))
                if np.isfinite(g2) and abs(g2) >= abs(obs):
                    hits += 1
            print(f"  phase-0 day-block bootstrap: observed {obs*1e4:+.2f}bp   "
                  f"p={hits/a.boot:.3f}   (n={len(s0)} independent windows)")

    print("\n" + "=" * 100)
    print("  If the effect collapses versus the overlapping estimate, the overlap was")
    print("  carrying it. A 52bp effect that survives ~129 independent observations is")
    print("  real; one that only exists across 16,812 overlapping windows is not.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature", default="tick_sgn")
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=3000)
    run(ap.parse_args())
