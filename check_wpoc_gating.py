# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_wpoc_gating.py
====================
RETRACTION AND REPLACEMENT for the A5 control in `check_wpoc_gate`.

`check_wpoc_gate` inherited the A5 baseline from `check_mbo_flow_interaction`,
which is algebraically degenerate at a balanced direction mix -- see the header
of `check_mbo_gating.py` for the full derivation. In one line:

    gap - base2m = (1 - 2p) * (M_al - M_op),   p = P(aligned)

and p ~= 0.5 by construction here too, so the statistic vanishes regardless of
the truth, and the `gap > 1.5*base` gate is unpassable in both branches. The
wPOC verdict must therefore be re-derived on a statistic that can actually move.

Everything else about `check_wpoc_gate` stands and is reused unchanged:
  * the wPOC is built from the PRIOR 5 sessions only (the `.shift(1)` fix for
    the lookahead in the proposal's reference code)
  * triggers are THINNED to >= h minutes apart within a day, so the forward
    windows scored here are genuinely non-overlapping

REPLACEMENT STATISTICS (same as check_mbo_gating)
    gain_al = E[d*r | aligned] - E[d*r]      what gating would actually add
    gamma   = 1/4 [ E(r|+,+) - E(r|+,-) - E(r|-,+) + E(r|-,-) ]   true 2x2 interaction

F1 = sign(spot - wPOC) is the regime feature; F2 proximity is reported as a
subgroup of F1 (near vs far by within-ticker distance rank), since proximity is
unsigned and cannot align with a direction on its own.

Inference is a day-block bootstrap over (ticker, date) blocks, 95% percentile CI.

Usage:  python check_wpoc_gating.py
        python check_wpoc_gating.py --tickers IWM QQQ --boot 3000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_mbo_gating import cells
from check_wpoc_gate import HORIZONS, SPLIT, build_panel, thin


def boot_ci(d, s, r, key, stat, boot, rng):
    by = {}
    for i, k in enumerate(key):
        by.setdefault(k, []).append(i)
    ks = list(by.keys())
    blocks = {k: np.array(v) for k, v in by.items()}
    vals = []
    for _ in range(boot):
        pick = rng.choice(len(ks), size=len(ks), replace=True)
        ii = np.concatenate([blocks[ks[j]] for j in pick])
        if (s[ii] == d[ii]).sum() < 10 or (s[ii] != d[ii]).sum() < 10:
            continue
        v = cells(d[ii], s[ii], r[ii])[stat]
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < boot // 4:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def prep(S, col):
    s = S["f1"].to_numpy(float)
    d = S["dir"].to_numpy(float)
    r = S[col].to_numpy(float)
    k = (S["ticker"].astype(str) + "|" + S["date"].astype(str)).to_numpy()
    dt = S["date"].to_numpy()
    m = np.isfinite(r) & np.isfinite(s) & (s != 0)
    return d[m], s[m], r[m], k[m], dt[m]


def block(label, S, col, boot, rng):
    if len(S) < 120:
        return
    d, s, r, k, dt = prep(S, col)
    if len(d) < 100 or (s == d).sum() < 25 or (s != d).sum() < 25:
        return
    c = cells(d, s, r)
    lo1, hi1 = boot_ci(d, s, r, k, "gain_al", boot, rng)
    lo2, hi2 = boot_ci(d, s, r, k, "gamma", boot, rng)

    def half(m):
        if m.sum() < 60 or (s[m] == d[m]).sum() < 15 or (s[m] != d[m]).sum() < 15:
            return np.nan
        return cells(d[m], s[m], r[m])["gain_al"]
    gi = half(dt < SPLIT)
    go = half(dt >= SPLIT)
    s1 = "*" if np.isfinite(lo1) and lo1 * hi1 > 0 else " "
    s2 = "*" if np.isfinite(lo2) and lo2 * hi2 > 0 else " "
    print(f"  {label:16} {len(d):>6} {c['p']:>6.3f} {c['m_all']*1e4:>+8.2f} "
          f"{c['M_al']*1e4:>+8.2f} {c['gain_al']*1e4:>+8.2f}{s1}"
          f"[{lo1*1e4:>+7.2f},{hi1*1e4:>+7.2f}] {c['gamma']*1e4:>+8.2f}{s2}"
          f"[{lo2*1e4:>+7.2f},{hi2*1e4:>+7.2f}] "
          f"{gi*1e4 if np.isfinite(gi) else np.nan:>+7.2f} "
          f"{go*1e4 if np.isfinite(go) else np.nan:>+7.2f}")


def run(a):
    import directional_flow_backtester as D
    from config import RULES
    rng = np.random.default_rng(31)
    tickers = a.tickers or sorted({r["ticker"] for r in RULES if r.get("enabled", True)})

    parts = []
    for tk in tickers:
        P = build_panel(D, tk)
        if P is None or P.empty:
            print(f"  ! {tk}: no panel")
            continue
        parts.append(P)
        print(f"  {tk:5} {len(P):>6} triggers  {P['date'].nunique():>4} days")
    if not parts:
        return
    A = pd.concat(parts, ignore_index=True)

    print("\n" + "=" * 124)
    print("  F1 = sign(spot - wPOC).  gain_al = gating the flow trigger on F1"
          " agreement vs not gating.")
    print("  NON-OVERLAPPING triggers only (>= h minutes apart within a day).")
    print("=" * 124)
    print(f"  {'cell':16} {'n':>6} {'p(al)':>6} {'ungated':>8} {'gate_al':>8} "
          f"{'gain_al':>8} {'[95% CI]':>18} {'gamma':>8} {'[95% CI]':>18} "
          f"{'IS':>7} {'OOS':>7}")
    for h in HORIZONS:
        col = f"fwd{h}"
        S = thin(A, h)
        S = S[np.isfinite(S[col])]
        block(f"all {h}m", S, col, a.boot, rng)
        # F2 proximity as a subgroup of F1 (proximity is unsigned on its own)
        block(f"  near {h}m", S[S["dist_r"] <= 0.33], col, a.boot, rng)
        block(f"  far  {h}m", S[S["dist_r"] >= 0.67], col, a.boot, rng)
        print()

    print("=" * 124)
    print("  * = day-block bootstrap 95% CI excludes zero.")
    print("  These are UNDERLYING returns. The bot trades 0DTE options on them, so a")
    print("  gain here only matters if it survives the option's spread and decay --")
    print("  a separate, harder bar than any bp figure in this table.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    run(ap.parse_args())
