# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_trades_per_day.py
=======================
HOW MANY TRADES A DAY DOES THIS BOOK NEED?  -- resolving a contradiction.

THE CONTRADICTION
    check_scalp_policies: a core leg restricted to ONE trade per day (`cap=1`)
    was worth +124.5pp OOS against the deployed book's +2649.8pp -- under 5% of
    the return. I read that as "the first trigger of the day is a bad entry".
    check_skip_anchor then tested that directly, with the day set held fixed,
    and found NO trigger-ordinal effect at all: anchoring on the 2nd/3rd/4th/5th
    trigger moved the matched-day total by -34 / -326 / -52 / +86pp, every CI
    straddling zero. If the first trigger were the problem, dropping it would
    help. It does not.
    So `cap=1` is not weak because of WHICH trade it takes. It is weak because
    of HOW MANY. This measures that directly.

WHAT IS VARIED, AND WHAT IS NOT
    Only `cap` -- the per-day trade ceiling. The policy, the fill model, the
    sequential guard and the trigger set are all untouched, so the arms differ
    in exactly one thing.
    Both are reported because they answer different questions:
      RAW        cap=n on every day it trades -- what you would actually earn
      MATCHED    cap=n vs uncapped ON THE SAME DAYS -- the marginal value of
                 the 2nd, 3rd, nth trade, with day selection held fixed
    A capped arm trades on the SAME days as the uncapped one (a cap never
    removes a day, unlike `skip`), so here the two are the same day set and the
    comparison is clean by construction -- worth stating, since the skip test
    needed an explicit control for exactly this.

WHAT WOULD FALSIFY "THE BOOK NEEDS VOLUME OF TRADES"
    If return were roughly proportional to trade count, cap=1 taking ~half the
    trades would return ~half the total. It returns ~5%. That is strongly
    CONVEX, and convexity means the marginal trade is worth more than the
    average one -- the opposite of diminishing returns, and the thing to check.

Usage:
  python check_trades_per_day.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()


def part(r, oos):
    return [(d, p) for d, p in r if ((d >= SPLIT) if oos else (d < SPLIT))]


def tot(r):
    return sum(p for _, p in r) * 100


def per_day(res):
    d = {}
    for dt, p in res:
        d[dt] = d.get(dt, 0.0) + p
    return d


def boot(a_by, b_by, dates, n, rng):
    dates = list(dates)
    if not dates:
        return (np.nan, np.nan)
    diff = np.array([(a_by.get(d, 0.0) - b_by.get(d, 0.0)) for d in dates], float) * 100
    out = np.empty(n)
    idx = np.arange(len(dates))
    for i in range(n):
        out[i] = diff[rng.choice(idx, size=len(idx), replace=True)].sum()
    return tuple(np.percentile(out, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", nargs="*", type=int, default=[1, 2, 3, 4, 6])
    ap.add_argument("--fill", default="bot")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=29)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = sim_core.research_rules()
    arms = {c: [] for c in a.caps}
    arms[None] = []
    per_rule = {c: {} for c in list(a.caps) + [None]}
    ntd = []                       # per (rule, day) trade counts, uncapped

    for rule in rules:
        tk = rule["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        cand = sim_core.build_candidates(D, rule, trigs=trigs)
        if not cand:
            continue
        eod_m = sim_core.eod_mod(rule)
        pol = sim_core.policy_for(rule)
        for c in list(a.caps) + [None]:
            r = sim_core.walk(cand, pol, eod_m, fill=a.fill, cap=c)
            arms[c] += r
            per_rule[c][rule["name"]] = r
        full = sim_core.walk(cand, pol, eod_m, fill=a.fill)
        cnt = {}
        for d, _ in full:
            cnt[d] = cnt.get(d, 0) + 1
        ntd += [(rule["name"], d, n) for d, n in cnt.items()]
        print(f"  {rule['name']} done", flush=True)

    print(f"\n{'='*100}")
    print(f"  RETURN vs PER-DAY TRADE CEILING   (only `cap` varies; fill={a.fill})")
    print(f"{'='*100}")
    print(f"  {'cap':10} {'IS trd':>7} {'IS tot%':>10} {'OOS trd':>8} {'OOS days':>9} "
          f"{'OOS tot%':>10} {'share of full':>14}")
    full_oos = tot(part(arms[None], True))
    for c in list(a.caps) + [None]:
        i, o = part(arms[c], False), part(arms[c], True)
        lbl = "uncapped" if c is None else f"cap={c}"
        sh = (tot(o) / full_oos * 100) if full_oos else np.nan
        print(f"  {lbl:10} {len(i):>7} {tot(i):>+10.1f} {len(o):>8} "
              f"{len({d for d,_ in o}):>9} {tot(o):>+10.1f} {sh:>13.1f}%")

    print(f"\n  MARGINAL VALUE OF LIFTING THE CAP  (OOS, same day set by construction)")
    print(f"  {'step':18} {'delta pp':>10} {'95% CI (day bootstrap)':>26}")
    base = per_day(part(arms[None], True))
    dates = sorted(base)
    prev = None
    for c in list(a.caps) + [None]:
        cur = per_day(part(arms[c], True))
        if prev is not None:
            lo, hi = boot(cur, prev, dates, a.boot, rng)
            d = (sum(cur.get(x, 0.0) for x in dates) -
                 sum(prev.get(x, 0.0) for x in dates)) * 100
            sig = "  <--" if (lo > 0 or hi < 0) else ""
            print(f"  {prev_lbl:>7} -> {('uncapped' if c is None else f'cap={c}'):<8} "
                  f"{d:>10.1f} [{lo:>+9.1f},{hi:>+9.1f}]{sig}")
        prev = cur
        prev_lbl = "uncapped" if c is None else f"cap={c}"

    N = pd.DataFrame(ntd, columns=["rule", "date", "n"])
    Noos = N[N["date"] >= SPLIT]
    print(f"\n  HOW MANY TRADES A (rule, day) ACTUALLY PRODUCES, uncapped -- OOS")
    vc = Noos["n"].value_counts().sort_index()
    for k, v in vc.items():
        print(f"    {k} trade(s): {v:>4} rule-days  ({v/len(Noos)*100:>4.1f}%)")
    print(f"    mean {Noos['n'].mean():.2f} per rule-day")

    print(f"\n  PER-RULE OOS TOTALS")
    names = sorted(per_rule[None])
    print(f"  {'cap':10} " + " ".join(f"{n.split()[0]:>9}" for n in names))
    for c in list(a.caps) + [None]:
        lbl = "uncapped" if c is None else f"cap={c}"
        cells = [f"{tot(part(per_rule[c].get(n, []), True)):>+9.1f}" for n in names]
        print(f"  {lbl:10} " + " ".join(cells))

    print(f"\n  HOW TO READ IT")
    print(f"  'share of full' against the trade counts is the whole story. If cap=1")
    print(f"  takes half the trades for 5% of the return, the book is CONVEX in trades")
    print(f"  per day and the later trades of a day carry it -- which is NOT the same")
    print(f"  claim as 'later TRIGGERS are better entries' (check_skip_anchor tested")
    print(f"  that and found nothing). The difference is that a cap truncates a day")
    print(f"  mid-sequence, while skip shifts where the sequence starts.")


if __name__ == "__main__":
    main()
