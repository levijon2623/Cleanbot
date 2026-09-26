# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_core_book.py
==================
IS THE PROPOSED REMOVAL OF {MSFT CHOP PUT, SMH, AVGO, GLD} JUSTIFIED, AND ON
WHAT GROUNDS?  The four are NOT equally defensible and this separates them.

THE DISTINCTION THAT MATTERS
    check_rule_selection_wf established an A PRIORI LIQUIDITY SCREEN -- keep
    rules whose MEDIAN ENTRY SPREAD is under 5% -- which uses NO performance
    data, carries ZERO selection bias, and was implementable on day one.  It
    drops exactly AVGO (6.7%), GLD (6.2%) and SMH (9.2%).  Those three removals
    need no further argument: a rule you can state in advance cannot be overfit.

    MSFT CHOP PUT PASSES THAT SCREEN (3.7%).  Dropping it rests on its realised
    P&L, which is selection on the outcome -- the exact trap that makes the
    "hindsight five" +30.5% OOS an upper bound rather than a forecast.  So it
    gets tested separately and held to a higher standard here.

BOOKS SCORED (all through sim_core: sequential fills, per-rule deployed exits,
fill="bot", so every book is on one basis)
    all9        every enabled rule
    spread6     a priori spread screen -- drops AVGO/GLD/SMH
    core5       spread6 minus MSFT CHOP PUT  <- the proposed working set
    nomsft8     all9 minus MSFT only, to isolate MSFT's contribution

Reported per book: n, all/IS/OOS mean, win, total and OOS-total P&L, 6-slice
coverage, plus a DAY-BLOCK bootstrap of the core5-minus-spread6 OOS difference.
If that interval spans zero, dropping MSFT is not distinguishable from noise
and the honest label is "removed on judgement", not "removed on evidence".

Usage:
  python check_core_book.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
#: dropped by the A PRIORI median-entry-spread screen (>5%). No performance data.
WIDE_SPREAD = ["AVGO HIVOL PUT", "GLD amp1 CALL", "SMH LOWVOL PUT"]
#: passes the spread screen; removal rests on realised P&L (see docstring).
JUDGEMENT = ["MSFT CHOP PUT"]


def trades() -> dict[str, list[tuple]]:
    """{rule name: [(date, pnl)]} for every enabled rule, one basis."""
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    out = {}
    for r in [x for x in RULES if x.get("enabled", True)]:
        cand = sim_core.build_candidates(D, r)
        if not cand:
            continue
        out[r["name"]] = sim_core.walk(cand, sim_core.policy_for(r, TRAIL_PCT),
                                       sim_core.eod_mod(r), fill="bot")
    return out


def _rows(tr, keep):
    return sorted([(d, p) for n, v in tr.items() if n in keep for d, p in v])


def _stat(rows, label):
    v = np.array([p for _, p in rows], float)
    if not len(v):
        print(f"  {label:12} (empty)")
        return {}
    i = np.array([p for d, p in rows if d < SPLIT], float)
    o = np.array([p for d, p in rows if d >= SPLIT], float)
    sl = [[] for _ in range(6)]
    for d, p in rows:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    pop = sum(1 for b in sl if len(b) >= 3)
    sstr = " ".join(f"S{j+1}{np.mean(b)*100:+.0f}" if len(b) >= 3 else f"S{j+1}··"
                    for j, b in enumerate(sl))
    print(f"  {label:12} n={len(v):>4}  all {v.mean()*100:>+6.1f}%  "
          f"IS {i.mean()*100 if len(i) else np.nan:>+6.1f}%  "
          f"OOS {o.mean()*100 if len(o) else np.nan:>+6.1f}%  "
          f"win {(v>0).mean():.2f}  tot {v.sum():>+7.2f}  "
          f"OOStot {o.sum() if len(o) else np.nan:>+7.2f}  pop {pop}/6  [{sstr}]")
    return dict(oos=o.mean() if len(o) else np.nan, oostot=o.sum() if len(o) else np.nan)


def _boot(a_rows, b_rows, n=4000, seed=31):
    """Day-block bootstrap of mean(A) - mean(B) on OOS, resampling whole days.

    Both books are resampled with the SAME day draw, because they share most of
    their trading days -- treating them as independent samples would overstate
    the precision of a difference between two overlapping books.
    """
    A = {}
    for d, p in a_rows:
        if d >= SPLIT:
            A.setdefault(d, []).append(p)
    B = {}
    for d, p in b_rows:
        if d >= SPLIT:
            B.setdefault(d, []).append(p)
    days = sorted(set(A) | set(B))
    if not days:
        return np.nan, np.nan, np.nan
    obs = (np.mean([p for d in A for p in A[d]])
           - np.mean([p for d in B for p in B[d]]))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = [days[k] for k in rng.choice(len(days), len(days), replace=True)]
        a = [p for d in pick for p in A.get(d, [])]
        b = [p for d in pick for p in B.get(d, [])]
        if a and b:
            out.append(np.mean(a) - np.mean(b))
    return obs, *np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    tr = trades()
    allr = set(tr)
    spread6 = allr - set(WIDE_SPREAD)
    core5 = spread6 - set(JUDGEMENT)
    nomsft8 = allr - set(JUDGEMENT)

    print("  Per-rule, on the deployed basis (sequential, fill=bot):")
    for n in sorted(tr):
        tag = ("  <- wide spread, A PRIORI drop" if n in WIDE_SPREAD else
               "  <- passes spread screen; removal is a JUDGEMENT call"
               if n in JUDGEMENT else "")
        _stat(tr[n], n.replace(" ", "")[:12])
        if tag:
            print(f"               {n}{tag}")

    print(f"\n{'='*118}\n  BOOKS\n{'='*118}")
    b_all = _rows(tr, allr)
    b_s6 = _rows(tr, spread6)
    b_c5 = _rows(tr, core5)
    b_n8 = _rows(tr, nomsft8)
    _stat(b_all, "all9")
    _stat(b_s6, "spread6")
    _stat(b_c5, "core5")
    _stat(b_n8, "nomsft8")
    print(f"\n  spread6 drops (a priori, no performance data): {sorted(WIDE_SPREAD)}")
    print(f"  core5 additionally drops:                      {sorted(JUDGEMENT)}")

    print(f"\n{'='*118}\n  IS DROPPING MSFT DISTINGUISHABLE FROM NOISE?\n{'='*118}")
    for lab, x, y in (("core5 - spread6", b_c5, b_s6),
                      ("nomsft8 - all9", b_n8, b_all),
                      ("spread6 - all9 (the a priori screen)", b_s6, b_all)):
        obs, c1, c2 = _boot(x, y, a.boot)
        star = "  *" if np.isfinite(c1) and (c1 > 0 or c2 < 0) else ""
        print(f"    OOS {lab:36} {obs*100:>+7.1f}pp   "
              f"day-block 95% CI [{c1*100:>+7.1f}, {c2*100:>+7.1f}]pp{star}")
    print("\n  A removal whose interval spans zero is 'removed on judgement',")
    print("  not 'removed on evidence'. Both can be correct decisions -- but the")
    print("  label has to be honest, because only the a priori screen is immune")
    print("  to the selection bias that makes the hindsight book look like +30.5%.")


if __name__ == "__main__":
    main()
