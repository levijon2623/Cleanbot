# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_trail_regression.py
=========================
Regression check on the trailing-exit deployment, NOT a new search.

check_book_now showed LULU (-30.8% OOS), MSFT CHOP PUT (-10.3%) and AMZN
afternoon PUT (-1.6%) negative under the new trailing exit. But that compared
TRAIL-on-realistic-fills against OLD-BRACKET-on-MID-fills, which is not
like-for-like: realistic fills alone cost ~10pp book-wide.

The decision-relevant question is narrow: **for each rule, holding the fill
model fixed at REALISTIC, is the static bracket or the trail better?**

  * trail better            -> keep TRAIL_PCT
  * static better, both +   -> set trail_pct: 0 (exit-compatibility problem)
  * negative under BOTH     -> the rule is the problem, not the exit -> disable

Usage:  python check_trail_regression.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
import sim_core

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _st(pnls):
    if not pnls:
        return dict(n=0, all=np.nan, is_=np.nan, oos=np.nan, win=np.nan, tot=0.0)
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    return dict(n=len(v), all=v.mean(), is_=np.mean(i) if i else np.nan,
                oos=np.mean(o) if o else np.nan, win=(v > 0).mean(), tot=v.sum())


def run(a):
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    rules = [r for r in RULES if r.get("enabled", True)]
    out = {}
    tot_static, tot_trail = [], []

    for r in rules:
        # sim_core: one builder, one simulator, live exit cushion MODELLED.
        # This matters here more than anywhere else -- the cushion charges 1.5x
        # spread on a losing exit vs 0.5x on a winning one, and the trail exits
        # at a loss far more often than the static bracket does. Running this
        # comparison WITHOUT the cushion (as it was before 2026-09-10) is
        # systematically biased IN FAVOUR OF THE TRAIL, and this script's output
        # is what sets `trail_pct` per rule in config.
        cand = sim_core.build_candidates(D, r)
        if not cand:
            continue
        em = sim_core.eod_mod(r)
        tr_, rr_ = float(r["target_roe"]), float(r["rr"])
        static = dict(name="static", kind="fixed", tp=tr_,
                      stop=(tr_ / rr_ if tr_ / rr_ < 1.0 else None))
        trail = dict(name="trail", kind="trail", tp=None,
                     trail=float(r.get("trail_pct", TRAIL_PCT) or 0.50), stop=None)
        s = sim_core.walk(cand, static, em, realistic=True, cushion=True)
        t_ = sim_core.walk(cand, trail, em, realistic=True, cushion=True)
        out[r["name"]] = (_st(s), _st(t_))
        tot_static += s
        tot_trail += t_

    print("=" * 120)
    print("  TRAILING-EXIT REGRESSION CHECK — REALISTIC FILLS ON BOTH SIDES (like-for-like)")
    print("=" * 120)
    print(f"\n  {'rule':24} {'STATIC bracket':>28}   {'TRAIL 50%':>28}   verdict")
    print(f"  {'':24} {'n    all     OOS   win':>28}   {'n    all     OOS   win':>28}")
    acts = []
    for rn, (s, t) in sorted(out.items(), key=lambda x: x[1][1]["oos"]):
        both_neg = s["oos"] < 0 and t["oos"] < 0
        if both_neg:
            verdict, act = "RULE is bad (both -)", ("disable?", rn)
        elif t["oos"] >= s["oos"]:
            verdict, act = "trail better -> keep", None
        else:
            verdict, act = "STATIC better -> trail_pct 0", ("trail_pct 0", rn)
        if act:
            acts.append(act)
        print(f"  {rn:24} {s['n']:>4} {s['all']*100:>+6.1f}% {s['oos']*100:>+7.1f}% {s['win']:>5.2f}   "
              f"{t['n']:>4} {t['all']*100:>+6.1f}% {t['oos']*100:>+7.1f}% {t['win']:>5.2f}   {verdict}")

    S, T = _st(tot_static), _st(tot_trail)
    print(f"\n  {'BLENDED':24} {S['n']:>4} {S['all']*100:>+6.1f}% {S['oos']*100:>+7.1f}% {S['win']:>5.2f}   "
          f"{T['n']:>4} {T['all']*100:>+6.1f}% {T['oos']*100:>+7.1f}% {T['win']:>5.2f}")
    print(f"  {'total P&L':24} {'':>4} {S['tot']:>+13.2f} {'':>11}   {'':>4} {T['tot']:>+13.2f}")

    # what the book looks like if we act on the verdicts
    fixed = []
    for rn, (s, t) in out.items():
        if any(rn == x[1] and x[0] == "disable?" for x in acts):
            continue
        fixed += (tot_static and [] )  # placeholder, recomputed below
    print("\n  -- if we applied the verdicts (per-rule best of the two, dropping both-negative) --")
    keep_oos, keep_n, keep_tot = [], 0, 0.0
    for rn, (s, t) in out.items():
        if s["oos"] < 0 and t["oos"] < 0:
            continue
        best = t if t["oos"] >= s["oos"] else s
        keep_oos.append(best["oos"] * best["n"])
        keep_n += best["n"]
        keep_tot += best["tot"]
    print(f"    kept rules: {sum(1 for _rn,(s,t) in out.items() if not (s['oos']<0 and t['oos']<0))}"
          f"/{len(out)}   n={keep_n}   OOS-weighted ~{sum(keep_oos)/max(keep_n,1)*100:+.1f}%   "
          f"total {keep_tot:+.2f}")
    print("\n  ACTIONS SUGGESTED:")
    for kind, rn in acts:
        print(f"    {rn:24} -> {kind}")
    if not acts:
        print("    none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.parse_args()
    run(a)
