# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scipy"]
# ///
"""
check_mbo_directional.py
========================
THE BIGGER QUESTION: does CME order-book microstructure carry DIRECTIONAL edge
on the future itself, at ANY horizon?

`check_mbo_gate.py` asked a narrow question -- do MBO features separate winners
from losers in 94 option trades -- and returned a clean null. Two explanations
were possible and I had been asserting the first without testing it:

  (a) the microsecond -> 30-minute COMPRESSION destroyed a real signal, or
  (b) the features are simply empty.

This separates them. Same features, same data, but scored against FORWARD
RETURN OF THE FUTURE at 1 / 5 / 15 / 30 / 60 minutes. If they predict at 1-5m
and decay to nothing by 30-60m, (a) is confirmed: the signal is real but lives
at a horizon we cannot trade. If they predict at NO horizon, (b) is confirmed
and I should stop blaming the horizon.

It is also a far more powerful test: ~21,500 minute observations instead of 94
trades.

SIGNED FEATURES ONLY -- and why the last test could never have answered this
---------------------------------------------------------------------------
hhi / absorb / tickchase / age_s / qdepth are MAGNITUDES. They cannot predict
direction, only volatility or activity. The directional versions are:

  imb        queue imbalance (bid_qty - ask_qty)/(total) at the touch
  absorb_sgn (ask-side hidden fills - bid-side)/total  -- + = buyers taking dark size
  tick_sgn   (ask-side re-pegs - bid-side)/total       -- + = sellers chasing up
  hhi_diff   HHI(bid) - HHI(ask)   -- which side has the committed wall
  age_diff   median age(bid) - median age(ask)  -- which side is more patient

*** POSITIVE CONTROL: `imb` ***
Queue imbalance is one of the most robustly documented short-horizon signals in
the microstructure literature. **If our reconstruction cannot reproduce it at
1-5 minutes, the book is wrong and every null in this file is uninterpretable.**
It is included precisely so a null has to survive a working-pipeline check
first. Treat a flat `imb` as a BUG REPORT, not a finding.

OVERLAPPING WINDOWS: a +60m return at minute t shares 59 minutes with t+1, so
|Spearman| significance is overstated. Read the DECILE SPREAD and the IS/OOS
agreement, not the p-value -- same convention as check_aggressor_imbalance.

Usage:  python check_mbo_directional.py
        python check_mbo_directional.py --symbols RTY_c_0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

CACHE = "_mbo_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
HORIZONS = [1, 5, 15, 30, 60]
SIGNED = ["imb", "absorb_sgn", "tick_sgn", "hhi_diff", "age_diff"]
CONTROL = "imb"


def load(sym):
    import polars as pl
    p = os.path.join(CACHE, f"{sym}.parquet")
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d.sort_values(["date", "mod"]).reset_index(drop=True)
    # forward returns WITHIN the session (no overnight gap contamination)
    for h in HORIZONS:
        d[f"fwd{h}"] = (d.groupby("date")["mid"].shift(-h) / d["mid"] - 1.0)
    return d


def decile_spread(x, y, q=10):
    """mean(y | top decile of x) - mean(y | bottom decile). Robust to outliers
    and to the monotonicity a Spearman assumes."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 200 or np.nanstd(x) == 0:
        return np.nan, 0
    try:
        lo, hi = np.nanquantile(x, 0.1), np.nanquantile(x, 0.9)
    except Exception:
        return np.nan, 0
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.nan, 0
    top, bot = y[x >= hi], y[x <= lo]
    if len(top) < 30 or len(bot) < 30:
        return np.nan, 0
    return float(np.mean(top) - np.mean(bot)), min(len(top), len(bot))


def run(a):
    from scipy.stats import spearmanr
    syms = a.symbols or [f[:-8] for f in os.listdir(CACHE) if f.endswith(".parquet")]
    for sym in sorted(syms):
        d = load(sym)
        if d is None or d.empty:
            print(f"  ! no cache for {sym}"); continue
        print("\n" + "=" * 104)
        print(f"  {sym}   {len(d):,} minute observations, {d['date'].nunique()} days")
        print("  decile spread = mean fwd return in the TOP feature decile minus the BOTTOM, in bp")
        print("=" * 104)
        print(f"  {'feature':12} {'horizon':>8} {'spearman':>10} {'D10-D1(bp)':>12} "
              f"{'IS(bp)':>9} {'OOS(bp)':>9}  {'n/dec':>7}")
        for f in SIGNED:
            if f not in d:
                continue
            tag = "  <-- POSITIVE CONTROL" if f == CONTROL else ""
            for h in HORIZONS:
                x = d[f].to_numpy(float)
                y = d[f"fwd{h}"].to_numpy(float)
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 300:
                    continue
                rho = spearmanr(x[m], y[m]).statistic
                sp, nd = decile_spread(x, y)
                isk = (d["date"] < SPLIT).to_numpy()
                sp_is, _ = decile_spread(x[isk], y[isk])
                sp_oos, _ = decile_spread(x[~isk], y[~isk])
                print(f"  {f:12} {h:>7}m {rho:>+10.4f} {sp*1e4 if np.isfinite(sp) else np.nan:>+12.2f} "
                      f"{sp_is*1e4 if np.isfinite(sp_is) else np.nan:>+9.2f} "
                      f"{sp_oos*1e4 if np.isfinite(sp_oos) else np.nan:>+9.2f} {nd:>7}"
                      + (tag if h == HORIZONS[0] else ""))
            print()

    print("=" * 104)
    print("  HOW TO READ THIS")
    print("=" * 104)
    print("  1. CHECK `imb` FIRST. If queue imbalance shows no 1-5m edge with the")
    print("     correct sign in both instruments, the book reconstruction is broken")
    print("     and nothing else on this page means anything.")
    print("  2. If `imb` works and the others decay from 1m to 60m, the COMPRESSION")
    print("     hypothesis is confirmed: real signal, wrong horizon for us.")
    print("  3. If `imb` works and the others are flat at EVERY horizon, those")
    print("     features are empty and the horizon was never the problem.")
    print("  Overlapping windows inflate |spearman|; judge on decile spread and")
    print("  IS/OOS agreement.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", default=None)
    run(ap.parse_args())
