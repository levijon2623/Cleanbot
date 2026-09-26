# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_sample_invariance.py
==========================
DIAGNOSTIC, not a hypothesis test. Answers one question:

  How much of our validation sample is an artefact of the POSITION POLICY,
  and is the policy-INVARIANT core big enough to validate gates on?

THE PROBLEM
-----------
`bot_runner.py:1288` holds one position per ticker. That guard decides which
triggers become trades. The EXIT POLICY decides when the ticker frees up, which
decides which SUBSEQUENT triggers become trades. So two exit policies do not
merely score the same trades differently -- they produce DIFFERENT TRADE
POPULATIONS. Measured 2026-09-09 on the same rules and triggers: deployed trail
-> 310 trades, trail+arm20/g25 -> 603 trades.

Every rejection on 2026-09-09 was on SAMPLE SIZE, not expectancy: CVX 1/6
slices; the AMT variants failed V3 everywhere (13-30 trades per cell);
arm50/BE 3/6 slices; the nine re-tested candidates were all blocked by slice
coverage. Slice coverage is a function of the guard, not of the rule. We have
been rejecting rules on a criterion downstream of a risk-control choice.

THE INVARIANT CORE
------------------
The FIRST qualifying trigger of each ticker-day is taken under EVERY exit policy
and EVERY position limit -- the ticker is flat at the open (EOD flatten), so
trigger #1 is always taken. It is naturally one-observation-per-ticker-day, so
it also sidesteps day-clustering. That makes it the right unit for the question
"does this gate select better triggers?", which is separate from "what will the
bot earn?" (which is policy-dependent, and correctly so).

WHAT THIS PRINTS
----------------
Per rule, and for the book:
  as-screened   every matched trigger scored independently (the old, inflated unit)
  sequential    the deployed one-position-per-ticker guard, under two very
                different exit policies -- the SPREAD between them is the
                policy artefact, measured
  first-trig    one observation per ticker-day: the invariant core
plus slice coverage for each, because that is the criterion doing the rejecting.

No pass/fail. This decides whether a first-trigger validation framework is worth
building, before building it.

Usage:  python check_sample_invariance.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
from check_exit_walkforward import _eod_mod
from check_giveback import _build, _sim

SPLIT = pd.Timestamp("2025-08-21").date()


def _walk_seq(cand, pol, eod_m, cushion=True):
    cur, busy, out = None, -1, []
    for d, m, path in cand:
        if d != cur:
            cur, busy = d, -1
        if m < busy:
            continue
        pnl, xm, _tag = _sim(path, pol, eod_m, True, cushion)
        out.append((d, pnl))
        busy = xm
    return out


def _all_trigs(cand, pol, eod_m, cushion=True):
    """as-screened: every trigger scored independently (no guard at all)."""
    return [(d, _sim(p, pol, eod_m, True, cushion)[0]) for d, m, p in cand]


def _first_trig(cand, pol, eod_m, cushion=True):
    """the invariant core: the FIRST trigger of each day only."""
    seen, out = set(), []
    for d, m, path in cand:
        if d in seen:
            continue
        seen.add(d)
        out.append((d, _sim(path, pol, eod_m, True, cushion)[0]))
    return out


def _walk_cap(cand, pol, eod_m, cap, skip=0, after=None, cushion=True):
    """The deployed guard PLUS a max-trades-per-ticker-day cap, an ordinal skip,
    and/or a time cutoff. Exit policy is held FIXED so only the entry admission
    rule varies.

      cap    max trades per ticker-day (None = deployed, unlimited)
      skip   ignore the first `skip` ACTIONABLE triggers of each ticker-day.
             "Actionable" = we were flat and could have taken it; triggers that
             arrive while a position is open are already blocked by the guard and
             do NOT consume the skip budget.
      after  no entry before this minute-of-day (the temporal analogue of skip)

    MECHANICAL NOTE, and it matters: skipping is NOT "start at trade 2 of the
    current sequence". When we skip we stay FLAT, so the next trigger is eligible
    IMMEDIATELY rather than having to wait for a position to close. skip=1
    therefore builds a genuinely different sequence -- generally entering EARLIER
    than the deployed run's 2nd trade, not later.
    """
    cur, busy, ntd, nsk, out = None, -1, 0, 0, []
    for d, m, path in cand:
        if d != cur:
            cur, busy, ntd, nsk = d, -1, 0, 0
        if m < busy:
            continue                      # position already open -- guard blocks
        if after is not None and m < after:
            continue                      # temporal cutoff: never actionable
        if nsk < skip:
            nsk += 1
            continue                      # ordinal skip: stay flat, next is eligible now
        if cap is not None and ntd >= cap:
            continue
        pnl, xm, _t = _sim(path, pol, eod_m, True, cushion)
        out.append((d, pnl))
        busy = xm
        ntd += 1
    return out


def _cov(pnls):
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    pop = [np.mean(b) for b in sl if len(b) >= 3]
    return len(pop), sum(1 for x in pop if x > 0)


def _line(lbl, pnls):
    if not pnls:
        return f"    {lbl:16} n=    0"
    v = np.array([p for _, p in pnls], float)
    i = np.array([p for d, p in pnls if d < SPLIT], float)
    o = np.array([p for d, p in pnls if d >= SPLIT], float)
    nd = len({d for d, _ in pnls})
    npop, nposs = _cov(pnls)
    return (f"    {lbl:16} n={len(v):>5} d={nd:>4} t/d {len(v)/max(nd,1):>4.1f} "
            f"IS {i.mean()*100 if len(i) else float('nan'):>+7.1f}% "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+7.1f}% "
            f"slices {nposs}/{npop}")


def run(a):
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    import sim_core

    rules = [r for r in RULES if r.get("enabled", True)]
    # The reference row must be the ACTUAL deployed exit per rule -- META/NVDA
    # carry trail_pct 0, so a book-wide trail50 would mislabel itself "deployed".
    # `fast` stays uniform on purpose: it is the contrast, not the baseline.
    fast = dict(name="fast", kind="trail", trail=TRAIL_PCT or 0.50, arm=0.20, give=0.25)

    tot = {k: [] for k in ("screen", "seq_trail", "seq_fast", "first")}
    built = {}   # candidates are expensive to build -- do it ONCE, reuse for the grid
    print("=" * 118)
    print("  SAMPLE INVARIANCE -- how much of the sample is a POSITION-POLICY artefact?")
    print("  seq(trail) vs seq(fast-exit): same rules, same triggers, DIFFERENT populations.")
    print("  first-trig: one obs per ticker-day -- taken under every policy, so INVARIANT.")
    print("=" * 118)
    for r in rules:
        c = _build(D, r, TRAIL_PCT)
        if not c:
            continue
        em = _eod_mod(r)
        trail = sim_core.policy_for(r, TRAIL_PCT)
        built[r["name"]] = (r, c, em)
        s = _all_trigs(c, trail, em)
        q1 = _walk_seq(c, trail, em)
        q2 = _walk_seq(c, fast, em)
        f = _first_trig(c, trail, em)
        tot["screen"] += s; tot["seq_trail"] += q1; tot["seq_fast"] += q2; tot["first"] += f
        print(f"\n  {r['name']}")
        print(_line("as-screened", s))
        print(_line(f"seq ({trail['name']})", q1))
        print(_line("seq (fast exit)", q2))
        print(_line("first-trig", f))

    print("\n" + "=" * 118)
    print("  BOOK TOTAL")
    print("=" * 118)
    for k, lbl in (("screen", "as-screened"), ("seq_trail", "seq (deployed)"),
                   ("seq_fast", "seq (fast exit)"), ("first", "first-trig")):
        print(_line(lbl, tot[k]))
    # ---- clean isolation: hold the EXIT fixed, vary only entry ADMISSION ----
    print("\n" + "=" * 118)
    print("  ADMISSION SWEEP -- exit policy held FIXED at the deployed trail50.")
    print("  Only which triggers are ALLOWED to become trades varies.")
    print("  CAVEAT: skip=0/cap=1 is the EARLIEST trigger of the day, and later-session > open")
    print("  is already established -- so the ordinal rows carry an hour bias, not just a count.")
    print("  The `after` rows below are the TEMPORAL version of the same idea, unconfounded.")
    print("=" * 118)
    print(f"  {'ordinal (skip first N, then cap)':34}")
    grid = {}
    for skip in (0, 1, 2):
        for cap in (1, 2, 3, None):
            rows = []
            for nm, (r, c, em) in built.items():
                rows += _walk_cap(c, trail, em, cap, skip=skip)
            grid[(skip, cap)] = rows
            print(_line(f"skip={skip} cap={cap if cap else 'none'}", rows))
        print()

    print(f"  {'temporal (no entry before ET), cap=3':34}")
    for after in (9 * 60 + 35, 10 * 60, 10 * 60 + 30, 11 * 60):
        rows = []
        for nm, (r, c, em) in built.items():
            rows += _walk_cap(c, trail, em, 3, after=after)
        print(_line(f"after {after//60}:{after%60:02d} cap=3", rows))

    # hour profile: is "first trigger" really just "early trigger"?
    print("\n  hour-of-day profile of the FIRST actionable trigger vs all others:")
    fh, oh = [], []
    for nm, (r, c, em) in built.items():
        seen = set()
        for d, m, _p in c:
            (fh if d not in seen else oh).append(m // 60)
            seen.add(d)
    import collections
    cf, co = collections.Counter(fh), collections.Counter(oh)
    hrs = sorted(set(cf) | set(co))
    print("    hour   " + " ".join(f"{h:>5}" for h in hrs))
    print("    first  " + " ".join(f"{cf.get(h,0):>5}" for h in hrs))
    print("    later  " + " ".join(f"{co.get(h,0):>5}" for h in hrs))

    # ---- DECOMPOSITION: is it "the 1st signal is bad" or "thin days are bad"? ----
    # skip=1 removes TWO things at once: (a) the first trade of every day, and
    # (b) every day that only ever had ONE actionable trigger. (b) would be a
    # day-quality effect wearing an ordinal costume. Conditioning on days with
    # >= k actionable triggers removes (b) and isolates (a).
    print("\n" + "=" * 118)
    print("  DECOMPOSITION -- ordinal position WITHIN days of equal depth")
    print("  (removes the 'skip=1 also deletes single-trigger days' composition effect)")
    print("=" * 118)
    for min_depth in (1, 2, 3):
        seqs = []
        for nm, (r, c, em) in built.items():
            byday = {}
            cur, busy = None, -1
            for d, m, path in c:
                if d != cur:
                    cur, busy = d, -1
                if m < busy:
                    continue
                pnl, xm, _t = _sim(path, trail, em, True, True)
                byday.setdefault(d, []).append(pnl)
                busy = xm
            for d, lst in byday.items():
                if len(lst) >= min_depth:
                    seqs.append((d, lst))
        print(f"\n  days with >= {min_depth} actionable trigger(s):  "
              f"{len(seqs)} ticker-days, {sum(len(l) for _, l in seqs)} trades")
        for pos in range(0, 4):
            rows = [(d, l[pos]) for d, l in seqs if len(l) > pos]
            lbl = f"trade #{pos+1}" if pos < 3 else "trade #4+"
            if pos == 3:
                rows = [(d, p) for d, l in seqs for p in l[3:]]
            print(_line(lbl, rows))

    # ---- WALK-FORWARD the skip choice (the arbiter that killed the give-back) ----
    print("\n" + "=" * 118)
    print("  WALK-FORWARD THE ADMISSION CHOICE -- setting picked ONLY on data before each")
    print("  slice, scored on that slice, chained. 16 cells were searched; this is the test")
    print("  that says whether we would have PICKED the winner in time.")
    print("=" * 118)
    wf_sel, wf_dep = [], []
    dep_rows = grid[(0, None)]
    for k in range(1, 6):
        picks = {}
        for key, rows in grid.items():
            prior = [p for d, p in rows if _slice_idx(d) is not None and _slice_idx(d) < k]
            if len(prior) >= 20:
                picks[key] = float(np.mean(prior))
        if not picks:
            continue
        best = max(picks, key=picks.get)
        cur = [p for d, p in grid[best] if _slice_idx(d) == k]
        dep = [p for d, p in dep_rows if _slice_idx(d) == k]
        if not cur:
            continue
        print(f"  S{k+1}  picked skip={best[0]} cap={best[1] if best[1] else 'none'}"
              f"   scored {np.mean(cur)*100:>+7.1f}%   deployed {np.mean(dep)*100 if dep else float('nan'):>+7.1f}%")
        wf_sel += cur
        wf_dep += dep
    if wf_sel:
        print(f"\n  chained: selected {np.mean(wf_sel)*100:>+7.1f}% (n={len(wf_sel)})"
              f"   vs deployed {np.mean(wf_dep)*100:>+7.1f}% (n={len(wf_dep)})"
              f"   -> {'SELECTION WINS' if np.mean(wf_sel) > np.mean(wf_dep) else 'deployed wins'}")

    a1, a2 = len(tot["seq_trail"]), len(tot["seq_fast"])
    print(f"\n  POLICY ARTEFACT: the deployed guard yields {a1} trades under the trail and "
          f"{a2} under a faster exit\n  ({(a2/max(a1,1)-1)*100:+.0f}% sample change from the EXIT choice alone, "
          f"on identical rules and triggers).")
    print(f"  INVARIANT CORE: {len(tot['first'])} observations, "
          f"{len({d for d,_ in tot['first']})} ticker-days -- unchanged by any exit or position policy.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run(ap.parse_args())
