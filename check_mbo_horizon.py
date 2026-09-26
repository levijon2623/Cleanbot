# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scipy"]
# ///
"""
check_mbo_horizon.py
====================
Does the MBO directional effect keep growing past 60 minutes -- far enough to
clear a real execution cost?

`check_mbo_directional` stopped at 60m and found `imb` RISING with horizon:
    1m +0.36   5m +1.19   15m +2.28   30m +2.38   60m +4.00 bp
An effect that grows might eventually clear a cost floor. Two floors matter:

    HL perp taker   0.00045/side  ->  9.0 bp ROUND TRIP  (check_spy_perp_overnight:51)
    CME futures     ~1 tick + commission -> ~1.2 bp round trip

This extends to 2h, 4h and EOD. **"Keep extending the horizon until something
clears the cost" is exactly the search that manufactured half the false
positives this week**, so the bar is fixed here before anything is run.

==================  PRE-COMMITTED CRITERIA  ================================
  H1  decile spread must EXCEED the cost floor it is being judged against
      (both bars reported; 9.0bp is the one that matters for the perp idea)
  H2  IS and OOS agree in SIGN at that horizon
  H3  DAY-BLOCK bootstrap p < 0.05. At long horizons the effective sample is
      DAYS, NOT MINUTES -- a 4h forward return at minute t overlaps 239 minutes
      with t+1, and at EOD every minute of a day shares ONE endpoint. Resampling
      minutes would treat 16,812 overlapping observations as independent and
      overstate significance by an enormous factor. We resample DAYS.
  H4  *** THE CONTROL THAT MATTERS *** -- see below.

H4: LONGER HORIZONS SECRETLY SAMPLE EARLIER IN THE DAY
------------------------------------------------------
A 4h forward return only exists for minutes before ~12:00 ET; a 2h return only
before ~14:00. So each longer horizon silently restricts the sample to EARLIER
minutes. If the 4h effect is really a morning effect, it will look like horizon
growth and is not.

H4 therefore recomputes the 60m effect on the SAME restricted minute subset. If
60m-on-morning-minutes is just as large as 4h-on-morning-minutes, the growth is
time-of-day, not horizon. This is the same logic as the A5 baseline in
check_mbo_flow_interaction: derive what the number would be under the null
mechanism, and require the observed effect to beat THAT, not zero.

EOD is a VARIABLE horizon (long in the morning, short in the afternoon), not a
fixed one -- reported separately and never compared like-for-like.

Usage:  python check_mbo_horizon.py
        python check_mbo_horizon.py --symbols RTY_c_0 --boot 4000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

CACHE = "_mbo_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
SIGNED = ["imb", "absorb_sgn", "tick_sgn", "hhi_diff", "age_diff"]
HORIZONS = [30, 60, 120, 240]
PERP_BAR = 9.0      # bp, HL taker round trip
FUT_BAR = 1.2       # bp, CME ~1 tick + commission


def load(sym):
    import polars as pl
    p = os.path.join(CACHE, f"{sym}.parquet")
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d.sort_values(["date", "mod"]).reset_index(drop=True)
    for h in HORIZONS:
        d[f"fwd{h}"] = d.groupby("date")["mid"].shift(-h) / d["mid"] - 1.0
    # EOD: variable horizon, to the session's last observed mid
    last = d.groupby("date")["mid"].transform("last")
    d["fwdEOD"] = last / d["mid"] - 1.0
    return d


def decile_gap(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 200 or np.nanstd(x) == 0:
        return np.nan, 0
    lo, hi = np.nanquantile(x, 0.1), np.nanquantile(x, 0.9)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.nan, 0
    top, bot = y[x >= hi], y[x <= lo]
    if len(top) < 30 or len(bot) < 30:
        return np.nan, 0
    return float(np.mean(top) - np.mean(bot)), min(len(top), len(bot))


def day_block_p(d, feat, col, boot, rng):
    """Resample DAYS with replacement. The honest null when forward windows
    overlap: effective n is days, not minutes."""
    obs, _ = decile_gap(d[feat].to_numpy(float), d[col].to_numpy(float))
    if not np.isfinite(obs):
        return np.nan
    days = d["date"].unique()
    by = {k: g for k, g in d.groupby("date")}
    hits = 0
    for _ in range(boot):
        pick = rng.choice(days, size=len(days), replace=True)
        s = pd.concat([by[k] for k in pick], ignore_index=True)
        # break the feature-outcome link while keeping day structure
        s[col] = s.groupby("date")[col].transform(lambda v: v.sample(frac=1, random_state=None).to_numpy())
        g, _ = decile_gap(s[feat].to_numpy(float), s[col].to_numpy(float))
        if np.isfinite(g) and abs(g) >= abs(obs):
            hits += 1
    return hits / boot


def run(a):
    rng = np.random.default_rng(19)
    syms = a.symbols or [f[:-8] for f in os.listdir(CACHE) if f.endswith(".parquet")]
    for sym in sorted(syms):
        d = load(sym)
        if d is None or d.empty:
            continue
        print("\n" + "=" * 116)
        print(f"  {sym}   {len(d):,} minutes, {d['date'].nunique()} days")
        print(f"  bars: PERP {PERP_BAR:.1f}bp round trip   FUTURES {FUT_BAR:.1f}bp")
        print("=" * 116)
        print(f"  {'feature':11} {'horiz':>6} {'gap(bp)':>9} {'IS':>8} {'OOS':>8} "
              f"{'H4 60m/same':>12} {'daysN':>6} {'blockp':>7}  H1p H1f H2 H3 H4")
        for f in SIGNED:
            for h in list(HORIZONS) + ["EOD"]:
                col = f"fwd{h}"
                if col not in d:
                    continue
                sub = d[np.isfinite(d[col])]
                if len(sub) < 400:
                    continue
                g, nd = decile_gap(sub[f].to_numpy(float), sub[col].to_numpy(float))
                if not np.isfinite(g):
                    continue
                gi, _ = decile_gap(sub[sub.date < SPLIT][f].to_numpy(float),
                                   sub[sub.date < SPLIT][col].to_numpy(float))
                go, _ = decile_gap(sub[sub.date >= SPLIT][f].to_numpy(float),
                                   sub[sub.date >= SPLIT][col].to_numpy(float))
                # H4: the 60m effect on the SAME (time-restricted) rows
                h4v, _ = decile_gap(sub[f].to_numpy(float), sub["fwd60"].to_numpy(float)) \
                    if "fwd60" in sub else (np.nan, 0)
                pv = day_block_p(sub, f, col, a.boot, rng) if abs(g) * 1e4 > 1.0 else np.nan
                h1p = abs(g) * 1e4 > PERP_BAR
                h1f = abs(g) * 1e4 > FUT_BAR
                h2 = np.isfinite(gi) and np.isfinite(go) and np.sign(gi) == np.sign(go)
                h3 = np.isfinite(pv) and pv < 0.05
                h4 = np.isfinite(h4v) and abs(g) > abs(h4v) * 1.25
                print(f"  {f:11} {str(h)+('m' if h != 'EOD' else ''):>6} "
                      f"{g*1e4:>+9.2f} {gi*1e4 if np.isfinite(gi) else np.nan:>+8.2f} "
                      f"{go*1e4 if np.isfinite(go) else np.nan:>+8.2f} "
                      f"{h4v*1e4 if np.isfinite(h4v) else np.nan:>+12.2f} "
                      f"{sub['date'].nunique():>6} {pv:>7.3f}  "
                      f"{'Y' if h1p else '.'}   {'Y' if h1f else '.'}   "
                      f"{'Y' if h2 else '.'}  {'Y' if h3 else '.'}  {'Y' if h4 else '.'}")
            print()

    print("=" * 116)
    print("  H1p = clears the 9bp PERP bar   H1f = clears the 1.2bp FUTURES bar")
    print("  H4  = the effect EXCEEDS the 60m effect measured on the SAME time-restricted")
    print("        rows by >25%. A '.' here means the apparent horizon growth is really")
    print("        a TIME-OF-DAY effect: longer horizons only exist for earlier minutes.")
    print("  EOD is a VARIABLE horizon and is not comparable to the fixed ones.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=1500)
    run(ap.parse_args())
