# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_skip_anchor.py
====================
ANCHOR THE BOOK ON THE 2nd / 3rd / 4th TRIGGER, THEN TAKE EVERY GUARDED ORDER.

WHERE THIS CAME FROM
    check_scalp_policies decomposed a core+sleeve arm three ways and all three
    agreed the core leg -- entries restricted to the FIRST trigger of the day --
    was worth +124.5pp OOS against the deployed book's +2649.8pp. Under 5% of
    the return, on roughly half the trades. So the question is not whether to
    hold a core, it is WHICH TRIGGER the book should start on.
    `sim_core.walk(skip=n)` stays FLAT through the first n eligible triggers and
    then takes everything the sequential guard allows -- exactly "anchor on the
    (n+1)th trigger", with no other change to the policy.

=====================  THE CONFOUND, AND THE CONTROL  =====================
Raising `skip` silently CHANGES THE DAY SET. A day with two triggers contributes
nothing at skip=2, so the surviving days are increasingly the HIGH-TRIGGER, high
-activity days. Comparing skip=3 against skip=0 on their own day sets therefore
conflates two different claims:
    (a) later triggers are better than earlier ones          <- the question
    (b) busy days are better than quiet ones                 <- not the question
So every comparison here is ALSO run on MATCHED DAYS: skip=0 restricted to the
exact dates that skip=n was able to trade. That holds the day set fixed and
isolates the trigger-ordinal effect. The raw column is kept beside it, because
the difference between the two IS the day-selection effect and is worth seeing.

POWER: the book has ~46 OOS days. Differences are bootstrapped over DAYS, not
trades (METHODOLOGY 7). Expect wide intervals and read the ORDERING of the arms
more than any single total.

Usage:
  python check_skip_anchor.py
  python check_skip_anchor.py --max-skip 5 --fill mid
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()


def per_day(res):
    """[(date, pnl)] -> {date: summed pnl}. The day is the unit of inference."""
    d = {}
    for dt, p in res:
        d[dt] = d.get(dt, 0.0) + p
    return d


def boot_diff(a_by_day, b_by_day, dates, n, rng):
    """Day-clustered bootstrap of sum(a) - sum(b) over a fixed date set."""
    dates = list(dates)
    if not dates:
        return (np.nan, np.nan)
    A = np.array([a_by_day.get(d, 0.0) for d in dates], float)
    B = np.array([b_by_day.get(d, 0.0) for d in dates], float)
    diff = (A - B) * 100
    out = np.empty(n)
    idx = np.arange(len(dates))
    for i in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        out[i] = diff[s].sum()
    return tuple(np.percentile(out, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-skip", type=int, default=4)
    ap.add_argument("--fill", default="bot")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = sim_core.research_rules()
    SKIPS = list(range(0, a.max_skip + 1))
    res = {s: [] for s in SKIPS}
    per_rule = {s: {} for s in SKIPS}

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
        pol = sim_core.policy_for(rule)     # NOT the raw rule -- see sim_core:96
        for s in SKIPS:
            r = sim_core.walk(cand, pol, eod_m, fill=a.fill, skip=s)
            res[s] += r
            per_rule[s][rule["name"]] = r
        print(f"  {rule['name']} done", flush=True)

    def part(r, oos):
        return [(d, p) for d, p in r if ((d >= SPLIT) if oos else (d < SPLIT))]

    print(f"\n{'='*104}")
    print(f"  ANCHORING THE BOOK ON A LATER TRIGGER   (policy unchanged, fill={a.fill})")
    print(f"{'='*104}")
    print(f"  {'anchor':22} {'IS trd':>7} {'IS tot%':>10} {'OOS trd':>8} {'OOS days':>9} "
          f"{'OOS tot%':>10} {'OOS/trade':>10}")
    for s in SKIPS:
        i, o = part(res[s], False), part(res[s], True)
        it = sum(p for _, p in i) * 100
        ot = sum(p for _, p in o) * 100
        od = len({d for d, _ in o})
        lbl = "1st (baseline)" if s == 0 else f"{s+1}th".replace("2th", "2nd").replace("3th", "3rd")
        print(f"  skip={s}  {lbl:14} {len(i):>7} {it:>+10.1f} {len(o):>8} {od:>9} "
              f"{ot:>+10.1f} {(ot/len(o) if o else np.nan):>+10.2f}")

    print(f"\n  MATCHED-DAY COMPARISON vs baseline  (OOS only)")
    print(f"  Holds the DAY SET fixed, so this is the trigger-ordinal effect alone.")
    print(f"  {'anchor':14} {'days':>6} {'skip=n':>10} {'skip=0 same days':>18} "
          f"{'delta':>10} {'95% CI (day boot)':>24}")
    base_o = per_day(part(res[0], True))
    for s in SKIPS[1:]:
        o_s = part(res[s], True)
        dset = sorted({d for d, _ in o_s})
        if not dset:
            continue
        a_by = per_day(o_s)
        tot_s = sum(a_by.get(d, 0.0) for d in dset) * 100
        tot_0 = sum(base_o.get(d, 0.0) for d in dset) * 100
        lo, hi = boot_diff(a_by, base_o, dset, a.boot, rng)
        sig = "  <--" if (lo > 0 or hi < 0) else ""
        print(f"  skip={s:<9} {len(dset):>6} {tot_s:>+10.1f} {tot_0:>+18.1f} "
              f"{tot_s-tot_0:>+10.1f} [{lo:>+8.1f},{hi:>+8.1f}]{sig}")

    print(f"\n  DAY-SELECTION EFFECT (what the raw table would have credited to 'later")
    print(f"  triggers' but is really 'busier days')")
    for s in SKIPS[1:]:
        o_s = part(res[s], True)
        dset = sorted({d for d, _ in o_s})
        if not dset:
            continue
        tot_0_all = sum(base_o.values()) * 100
        tot_0_sub = sum(base_o.get(d, 0.0) for d in dset) * 100
        print(f"    skip={s}: baseline on ALL {len(base_o)} days {tot_0_all:+.1f}  vs "
              f"on its {len(dset)} surviving days {tot_0_sub:+.1f}  "
              f"-> {tot_0_sub - tot_0_all:+.1f}pp is day selection")

    print(f"\n  PER-RULE OOS TOTALS")
    names = sorted(per_rule[0])
    print(f"  {'anchor':14} " + " ".join(f"{n.split()[0]:>9}" for n in names))
    for s in SKIPS:
        cells = []
        for n in names:
            o = part(per_rule[s].get(n, []), True)
            cells.append(f"{sum(p for _, p in o)*100:>+9.1f}")
        print(f"  skip={s:<9} " + " ".join(cells))

    print(f"\n  HOW TO READ IT")
    print(f"  The MATCHED-DAY delta is the answer to the question asked. The raw")
    print(f"  table above it is inflated by day selection in an amount the third")
    print(f"  block quantifies. If matched-day deltas are flat while raw totals")
    print(f"  fall, later triggers are neither better nor worse -- the book simply")
    print(f"  loses the days that never produced enough triggers to anchor on.")


if __name__ == "__main__":
    main()
