# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_exit_bracketology.py
==========================
THE WHOLE EXIT FAMILY, RE-SCORED UNDER THE CORRECTED FILL MODEL.

WHY REVISIT
    Three separate exit decisions in this book trace back to a fill model that
    has since been retired, and all three were made BEFORE 2026-09-16:
      * check_exit_sweep (09-08) and check_exit_walkforward (09-10) chose the
        pure trail over fixed brackets
      * check_giveback (09-14) killed the ROE-armed give-back family
      * check_strike_selection (09-07) put GLD on strike_offset 1
    Every one predates CUSHION_CAP, FILL_COST and the 27% entry-fill
    measurement, and check_giveback additionally ran its OWN `_sim`/`_walk`
    rather than sim_core -- a second simulator, which is the drift METHODOLOGY 1
    is about.

    The cushion change is not a uniform shift, which is why the re-run can move
    results either way. The retired rule keyed on PROFITABILITY (0.5x if the
    exit was green, 1.5x if not), so a give-back -- which fires green by
    construction -- got a systematic 3x discount for being profitable. The
    current rule keys on the TAG: ADVERSE_TAGS = ("stop","trail","give"), so a
    give-back and a trail now pay the SAME per-ticker cap (GLD 0.40 ... SPY
    1.50). Give-backs lose their discount; wide-spread names gain relief.

🚨 THIS IS A SEARCH, SO THE HEADLINE IS THE WALK-FORWARD, NOT THE TOTALS
    18 policies scored on one sample will always produce a winner. METHODOLOGY 5
    is explicit: walk-forward the SELECTION, not just the strategy -- pick the
    policy using only prior slices, score it on the next, chain. That is exactly
    the test that killed the give-back family last time (+4.5% chained vs +8.9%
    deployed, the winner changing every slice), and it is the only number here
    that means anything. Raw totals are printed because they are informative
    about MECHANISM, and are labelled as the search they are.

PRE-COMMITTED CRITERIA -- fixed before the first run
    E1  chained walk-forward selection beats always-trail50 on total ROE
    E2  the selection is STABLE: one policy wins >= 4 of the 6 slices
    E3  the best single policy beats trail50 on >= 6 of 9 rules
    E4  it also beats trail50 on DAY-LEVEL median ROE, not just totals

    E2 is the criterion the give-back family failed last time and the one most
    likely to fail again -- a cost-model correction shifts levels, it does not
    obviously make an unstable parameter choice stable. The honest prior is
    that this fails again, better measured.

THE DEAD ZONE THIS IS ALL ABOUT
    trail50's stop only clears entry once peak ROE exceeds +100%:
        peak   0%  -> stop -50%      peak +100% -> stop   0%
        peak +50%  -> stop -25%      peak +200% -> stop +50%
    Every trade peaking between 0% and +100% exits at a loss BY CONSTRUCTION.
    An armed give-back locks part of the GAIN instead of a fraction of the
    PRICE, which is the mechanism that would close that hole -- if it survives.

Usage:
  python check_exit_bracketology.py --paper
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()


def policies():
    """Name -> policy dict, in sim_core.simulate's own vocabulary.

    `arm`/`give` are only consulted when `trail` is set (sim_core.py:290-299),
    and the give level supersedes the trail level whenever it is higher. So a
    give-back policy is a trail with a floor that arms at `arm` ROE and keeps
    (1 - give) of the peak gain.
    """
    P = {}
    for t in (0.25, 0.50, 0.75):
        P[f"trail{int(t*100)}"] = dict(kind="trail", trail=t)
    for tp, st in ((1.0, 0.50), (1.0, 0.67), (1.5, 0.50), (2.0, 0.67)):
        P[f"tp{int(tp*100)}/sl{int(st*100)}"] = dict(kind="bracket", tp=tp, stop=st)
    P["tp100+trail50"] = dict(kind="both", tp=1.0, trail=0.50)
    P["tp200+trail50"] = dict(kind="both", tp=2.0, trail=0.50)
    for arm in (0.50, 1.00, 1.50):
        for give in (0.25, 0.50, 0.75):
            P[f"arm{int(arm*100)}/give{int(give*100)}"] = dict(
                kind="give", trail=0.50, arm=arm, give=give)
    return P


def stats(res):
    if not res:
        return dict(n=0, tot=0.0, med=np.nan, daymed=np.nan, loss50=np.nan,
                    win=np.nan)
    v = np.array([p for _d, p in res], float) * 100
    d = {}
    for dt, p in res:
        d.setdefault(dt, []).append(p * 100)
    return dict(n=len(v), tot=float(v.sum()), med=float(np.median(v)),
                daymed=float(np.median([np.median(x) for x in d.values()])),
                loss50=float((v <= -50).mean() * 100),
                win=float((v > 0).mean() * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--slices", type=int, default=6)
    a = ap.parse_args()

    import directional_flow_backtester as D
    POL = policies()
    print(f"  {len(POL)} exit policies, fill={a.fill}, per-ticker CUSHION_CAP\n")

    # build candidates ONCE per rule; every policy replays the same paths
    book, dates = {}, []
    for rule in sim_core.research_rules(include_paper=a.paper):
        cand = sim_core.build_candidates(D, rule)
        if not cand:
            continue
        book[rule["name"]] = (rule, cand)
        dates += [d for d, _m, _p in cand]
        print(f"    {rule['name']:24} {len(cand):>5} candidates", flush=True)
    if not book:
        print("  nothing"); return
    dates = sorted(set(dates))
    print(f"\n  {sum(len(c) for _r, c in book.values()):,} candidates, "
          f"{len(dates)} sessions ({dates[0]} .. {dates[-1]})\n")

    # policy -> [(date, pnl)] pooled across rules
    res = {p: [] for p in POL}
    per_rule = {p: {} for p in POL}
    for nm, (rule, cand) in book.items():
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(rule["ticker"])
        for pn, pol in POL.items():
            r = sim_core.walk(cand, pol, eod, fill=a.fill, cush_cap=cap)
            res[pn] += r
            per_rule[pn][nm] = stats(r)
        print(f"    {nm} walked", flush=True)

    base = "trail50"
    print(f"\n{'='*104}")
    print(f"  1. THE SEARCH (all policies, whole sample) -- NOT the verdict")
    print(f"{'='*104}")
    print(f"  {'policy':20} {'n':>6} {'total':>10} {'medROE':>9} {'dayMed':>9} "
          f"{'loss50':>8} {'win':>7} {'vs trail50':>11}")
    S = {p: stats(res[p]) for p in POL}
    b = S[base]["tot"]
    for p, s in sorted(S.items(), key=lambda kv: -kv[1]["tot"]):
        mark = "  <- deployed" if p == base else ""
        print(f"  {p:20} {s['n']:>6} {s['tot']:>+10.0f} {s['med']:>+9.1f} "
              f"{s['daymed']:>+9.1f} {s['loss50']:>7.1f}% {s['win']:>6.1f}% "
              f"{s['tot']-b:>+11.0f}{mark}")

    # ---- THE VERDICT: walk-forward the SELECTION -------------------------
    print(f"\n{'='*104}")
    print(f"  2. WALK-FORWARD OF THE POLICY CHOICE (E1/E2) -- the verdict")
    print(f"{'='*104}")
    edges = np.array_split(np.array(dates), a.slices)
    chained, picks = [], []
    for k in range(1, len(edges)):
        prior = set(np.concatenate(edges[:k]).tolist())
        cur = set(edges[k].tolist())
        scored = {p: sum(pn for d, pn in res[p] if d in prior) for p in POL}
        pick = max(scored, key=scored.get)
        got = [pn for d, pn in res[pick] if d in cur]
        chained += got
        picks.append(pick)
        dep = [pn for d, pn in res[base] if d in cur]
        print(f"  slice {k+1}/{len(edges)}  {str(edges[k][0])} .. {str(edges[k][-1])}"
              f"   picked {pick:20} -> {sum(got)*100:>+8.0f}   "
              f"trail50 {sum(dep)*100:>+8.0f}")
    ch = sum(chained) * 100
    dp = sum(pn for d, pn in res[base]
             if d in set(np.concatenate(edges[1:]).tolist())) * 100
    stable = max(set(picks), key=picks.count)
    nstab = picks.count(stable)
    print(f"\n  chained selection {ch:>+9.0f}   always-trail50 {dp:>+9.0f}   "
          f"delta {ch-dp:>+9.0f}")
    print(f"  most-picked policy: {stable} ({nstab}/{len(picks)} slices)")
    print(f"  E1 chained beats trail50   {'PASS' if ch > dp else 'FAIL'}")
    print(f"  E2 selection stable (>=4)  {'PASS' if nstab >= 4 else 'FAIL'}"
          f"  -- picks: {picks}")

    # ---- E3 / E4 on the best single policy -------------------------------
    best = max(S, key=lambda p: S[p]["tot"])
    print(f"\n{'='*104}")
    print(f"  3. BEST SINGLE POLICY '{best}' vs trail50, PER RULE (E3)")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'trail50 med':>12} {f'{best} med':>16} {'delta':>9}")
    winr = 0
    for nm in book:
        x, y = per_rule[base][nm], per_rule[best][nm]
        if not np.isfinite(x["med"]) or not np.isfinite(y["med"]):
            continue
        winr += int(y["med"] > x["med"])
        print(f"  {nm:24} {x['med']:>+12.1f} {y['med']:>+16.1f} "
              f"{y['med']-x['med']:>+9.1f}")
    print(f"\n  E3 best beats trail50 on {winr}/{len(book)} rules  "
          f"{'PASS' if winr >= 6 else 'FAIL'}")
    d1, d2 = S[base]["daymed"], S[best]["daymed"]
    print(f"  E4 day-level median  trail50 {d1:+.1f}  {best} {d2:+.1f}  "
          f"{'PASS' if d2 > d1 else 'FAIL'}")

    print(f"\n  HOW TO READ IT")
    print(f"  Block 1 is a search over {len(POL)} policies and will always show a")
    print(f"  winner; treat it as evidence about MECHANISM, not as a result.")
    print(f"  Block 2 is the only number that survives the search: if the chained")
    print(f"  selection cannot beat simply always running trail50, the ranking in")
    print(f"  block 1 is noise being read as signal -- which is exactly how the")
    print(f"  give-back family died the first time.")


if __name__ == "__main__":
    main()
