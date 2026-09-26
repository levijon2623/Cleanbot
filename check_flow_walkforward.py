# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_flow_walkforward.py
=========================
SHOULD THE FLOW GATE GO UP? WALK-FORWARD THE SELECTION, NOT THE STRATEGY.

WHERE THIS COMES FROM
    check_flow_threshold answered the question that was asked -- a LOOSER gate
    fails on all five criteria, marginal band -11.2 ROE/trade against the
    deployed band's +9.9 -- and left a louder observation behind. The book total
    rises monotonically with the percentile, and at p90 the two halves agree to
    within 0.5 ROE/trade (IS +20.7, OOS +20.2) while totalling MORE than the
    deployed mix on 41% fewer trades (+4,721 / 231 vs +3,785 / 392).

    That is an argmax over seven values read off one table. METHODOLOGY 5 exists
    because exactly that move is how check_exit_bracketology produced a policy
    that looked best in every historical sort and then LOST the chained
    walk-forward to plain trail50, +3,334 against +4,374. The selection is the
    thing that has to be walked forward, not the strategy.

THE PROTOCOL
    Six sequential ~4-month calendar slices, S1..S6 (check_config_walkforward's
    canonical edges -- not re-derived here). For each evaluated slice k:
        pick each rule's percentile using slices 1..k-1 ONLY
        trade slice k with that pick
    Chained over S2..S6, so five evaluated slices and no forward information at
    any point. A rule with fewer than --min-train trades in the training window
    keeps its DEPLOYED percentile, because switching a live gate on four
    observations is not a decision anyone would make.

    The sequential guard resets per day, so a slice's trades are independent of
    every other slice's. That is what lets one walk per (rule, percentile) be
    bucketed by date afterwards instead of re-walking 7 x 9 x 5 times.

THE COMPARATORS -- the point is that three of them can beat it
    DEPLOYED   config.py's percentiles, same slices. The incumbent.
    WALK-FWD   the protocol above.
    FIXED p90  what you would do having read check_flow_threshold's table.
               CONTAMINATED BY CONSTRUCTION: p90 was chosen by looking at the
               whole sample including these slices. It is here as the thing to
               beat honestly, not as a fair competitor.
    PLACEBO    a percentile drawn at random per (rule, slice). METHODOLOGY 7.
    ORACLE     the best percentile for each slice chosen WITH knowledge of it.
               Unachievable; it bounds how much selection could ever be worth.

PRE-COMMITTED CRITERIA -- fixed before the first run
    W1  walk-forward total > deployed total, chained over S2..S6
    W2  walk-forward > placebo
    W3  walk-forward wins in >= 3 of the 5 evaluated slices
    W4  the selection is STABLE -- the median rule changes its percentile in
        fewer than half of its selection opportunities. A choice that flips
        every slice is noise being fitted, whatever it totals.

RESULT -- 2026-09-20, post spot-fix. THE ANSWER DEPENDS ON WHICH BOOK, AND THE
SPLIT IS NOT A POST-HOC CARVE-OUT: research_rules() defaults to the core five,
and PAPER_ONLY was set on 2026-09-12 on an A PRIORI median-entry-spread screen
(AVGO 6.7%, GLD 6.2%, SMH 9.2% vs 1.1-4.0%) that uses no performance data.

    CORE FIVE (the default set) -- ALL FOUR PASS
        DEPLOYED   +4,444 / 197      WALK-FWD  +6,361 / 241   (+43%)
        FIXED p90  +4,879 / 121      PLACEBO   +1,422 / 373
        ORACLE     +9,364 / 282      -> walk-forward captures 39% of headroom
        W1 PASS  W2 PASS  W3 4/5  W4 40% change rate PASS

    FULL BOOK (--paper) -- W4 FAILS
        DEPLOYED   +4,077 / 321      WALK-FWD  +4,881 / 392   (+20%)
        FIXED p90  +4,648 / 193      ORACLE   +10,863 / 375   (12% of headroom)
        W4 60% change rate FAIL -- GLD flips all 5 slices, AVGO 4 of 5.
    The wide-spread paper rules are where the instability lives, which is the
    same property that put them in PAPER_ONLY. Consistent, not convenient.

    🚨 THE PICKS ARE HETEROGENEOUS, WHICH KILLS "JUST RAISE EVERYTHING TO p90":
        SPY  dep p90 -> p90 p20 p20 p20 p20     wants LOOSER
        IWM  dep p80 -> p90 p95 p90 p90 p90     wants TIGHTER
        QQQ  dep p95 -> p65 p65 p90 p90 p90
        NVDA dep p80 -> p20 p20 p35 p35 p50     wants LOOSER
    check_flow_threshold's monotone book-level gradient is an AGGREGATE. Per
    rule the optimum goes both ways, and on the core five FIXED p90 (+4,879)
    now loses to adaptive (+6,361) -- the opposite of the full book, where they
    were level. Whatever is being learned is per-rule, not a global level.

    🚨🚨 REJECTED ON THE SELECTION-CRITERION SENSITIVITY. Re-running the core
    five with --select per_trade -- an equally defensible objective, and the
    capital-neutral one -- REVERSES the verdict:
            select on      WALK-FWD    vs DEPLOYED +4,444    W1
            total            +6,361        +1,917           PASS
            per_trade        +3,482          -962           FAIL
    The picks do not survive the swap either (NVDA p20/p20/p35/p35/p50 becomes
    p20/p50/p80/p80/p80; QQQ p65/p65 becomes p20/p90). Two defensible
    objectives, opposite signs, same data and same protocol -- that is a coin
    flip with a scorecard attached, not a finding. Had per_trade been the
    default in this file, the first run would have reported a clean failure
    with identical confidence. ADAPTIVE PER-RULE SELECTION IS NOT DEPLOYABLE.

    WHAT SURVIVES: FIXED p90 is IDENTICAL in both tables (+4,879 / 121) because
    it performs no selection, and it beats DEPLOYED (+4,444 / 197) on 39% fewer
    trades under either objective. It remains contaminated -- p90 was read off
    check_flow_threshold's full-sample sweep -- so the only clean test left is
    the PRE-SAMPLE HOLDOUT (2023-10-12 .. 2024-08-19, P1-P3), which
    PRESAMPLE_PLAN.md protects and which nothing in this line of work has
    touched. One test, one spend. Not run; the holdout is not mine to spend.

Usage:
  python check_flow_walkforward.py            # core five -- the canonical run
  python check_flow_walkforward.py --paper
  python check_flow_walkforward.py --select per_trade
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

GRID = [20, 35, 50, 65, 80, 90, 95]
SEED = 20260920


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--min-train", type=int, default=10,
                    help="training trades required before a rule may switch")
    ap.add_argument("--select", choices=("total", "per_trade"), default="total")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _slice_idx, SLICE_EDGES
    from check_config_walkforward import _flow_for
    D.GRID_MIN_FLOW_PCT = GRID

    NS = len(SLICE_EDGES) - 1
    book = {}          # rule -> {P -> [(slice_idx, pnl)]}
    dep = {}
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, d0 = rule["ticker"], int(rule.get("min_flow_pct") or 0)
        if not d0:
            continue
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        tmap = {(t["date"], pd.Timestamp(t["ts"]).hour * 60
                 + pd.Timestamp(t["ts"]).minute): t
                for t in trigs if t["dir"] == rule["direction"]}
        loose = {**rule, "min_flow_pct": GRID[0]}
        cand = sim_core.build_candidates(D, loose, trigs=trigs)
        if not cand:
            continue
        ct = [tmap.get((d, m)) for d, m, _p in cand]
        pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        book[rule["name"]], dep[rule["name"]] = {}, d0
        for P in GRID:
            keep = [c for c, t in zip(cand, ct)
                    if t is not None and t.get("thr")
                    and t["abs_flow"] >= t["thr"][P]]
            rows = sim_core.walk(keep, pol, eod, fill=a.fill, cush_cap=cap)
            book[rule["name"]][P] = [(_slice_idx(d), p * 100) for d, p in rows
                                     if _slice_idx(d) is not None]
        print(f"    {rule['name']} done  (deployed p{d0})", flush=True)

    if not book:
        print("  nothing"); return

    def score(nm, P, slices):
        v = [p for s, p in book[nm][P] if s in slices]
        if a.select == "per_trade":
            return (np.mean(v) if v else -1e9), len(v)
        return (float(np.sum(v)) if v else -1e9), len(v)

    rng = np.random.default_rng(SEED)
    res = {k: [0.0] * NS for k in
           ("DEPLOYED", "WALK-FWD", "FIXED p90", "PLACEBO", "ORACLE")}
    ntr = {k: [0] * NS for k in res}
    picks = {nm: {} for nm in book}
    switches = {nm: [0, 0] for nm in book}   # [changes, opportunities]

    for k in range(1, NS):                   # evaluate S2..S6
        train = set(range(0, k))
        for nm in book:
            # --- the walk-forward pick: prior slices only ---
            best, bn = None, 0
            for P in GRID:
                s, n = score(nm, P, train)
                if n >= a.min_train and (best is None or s > best[1]):
                    best, bn = (P, s), n
            pick = best[0] if best else dep[nm]
            prev = picks[nm].get(k - 1, dep[nm])
            switches[nm][1] += 1
            switches[nm][0] += int(pick != prev)
            picks[nm][k] = pick
            # --- everyone scores the SAME evaluated slice ---
            for lab, P in (("DEPLOYED", dep[nm]), ("WALK-FWD", pick),
                           ("FIXED p90", 90),
                           ("PLACEBO", int(rng.choice(GRID)))):
                v = [p for s, p in book[nm][P] if s == k]
                res[lab][k] += float(np.sum(v)); ntr[lab][k] += len(v)
            orc = max(GRID, key=lambda P: sum(p for s, p in book[nm][P]
                                              if s == k))
            v = [p for s, p in book[nm][orc] if s == k]
            res["ORACLE"][k] += float(np.sum(v)); ntr["ORACLE"][k] += len(v)

    ev = list(range(1, NS))
    print(f"\n{'='*104}")
    print(f"  1. CHAINED WALK-FORWARD -- select on prior slices, trade the next")
    print(f"     select on {a.select}, min {a.min_train} training trades to "
          f"switch, {'full book' if a.paper else 'core five'}")
    print(f"{'='*104}")
    print(f"  {'model':14} " + "".join(f"{'S'+str(i+1):>12}" for i in ev)
          + f"{'TOTAL':>12}{'trades':>9}")
    for lab in ("DEPLOYED", "WALK-FWD", "FIXED p90", "PLACEBO", "ORACLE"):
        tot = sum(res[lab][i] for i in ev)
        n = sum(ntr[lab][i] for i in ev)
        print(f"  {lab:14} " + "".join(f"{res[lab][i]:>+12.0f}" for i in ev)
              + f"{tot:>+12.0f}{n:>9}")

    wf = sum(res["WALK-FWD"][i] for i in ev)
    dp = sum(res["DEPLOYED"][i] for i in ev)
    pl = sum(res["PLACEBO"][i] for i in ev)
    wins = sum(1 for i in ev if res["WALK-FWD"][i] > res["DEPLOYED"][i])

    print(f"\n{'='*104}")
    print(f"  2. W4 -- IS THE SELECTION STABLE, OR IS IT CHASING?")
    print(f"     A pick that flips every slice is noise being fitted, whatever "
          f"it totals.")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'dep':>5} " +
          "".join(f"{'->S'+str(i+1):>7}" for i in ev) + f"{'changes':>9}")
    rates = []
    for nm in book:
        ch, op = switches[nm]
        rates.append(ch / max(op, 1))
        print(f"  {nm:24} {'p'+str(dep[nm]):>5} " +
              "".join(f"{'p'+str(picks[nm].get(i, dep[nm])):>7}" for i in ev)
              + f"{ch}/{op:>7}")
    med = float(np.median(rates)) if rates else 1.0
    print(f"\n  median change rate {med:.0%}")

    print(f"\n{'='*104}")
    print(f"  SCORECARD  (pre-committed)")
    print(f"{'='*104}")
    m = lambda ok: "PASS" if ok else "FAIL"
    print(f"  W1  walk-forward > deployed   {wf:>+9.0f} vs {dp:>+9.0f}   "
          f"{m(wf > dp)}")
    print(f"  W2  walk-forward > placebo    {wf:>+9.0f} vs {pl:>+9.0f}   "
          f"{m(wf > pl)}")
    print(f"  W3  slices won                {wins:>9}/{len(ev)}"
          f"{'':>13}{m(wins >= 3)}")
    print(f"  W4  selection stable (<50%)   {med:>8.0%}{'':>14}   {m(med < 0.5)}")
    print(f"\n  ORACLE is the ceiling on what ANY selection rule could have "
          f"earned here.")
    print(f"  If WALK-FWD sits near DEPLOYED and far from ORACLE, the "
          f"percentile is not")
    print(f"  learnable from four months of history, whatever the full-sample "
          f"table showed.")


if __name__ == "__main__":
    main()
