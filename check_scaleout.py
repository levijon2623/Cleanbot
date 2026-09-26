# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_scaleout.py
=================
SEQUENTIAL PROFIT TAKING (scaling out): hold N contracts, sell tranches at
rising ROE levels, leave a runner.

WHY THIS IS STRUCTURALLY DIFFERENT FROM EVERY EXIT ALREADY REJECTED
    Every exit tested so far fails the same way: it leaves early, captures more
    of the peak on trades that turn, and forfeits the trades that don't -- and
    the second effect dominates because the tail IS the book. Scaling out never
    fully exits: the runner keeps unlimited upside. So the usual failure mode
    does not obviously apply.

WHAT ARITHMETIC ALREADY SETTLES, before any backtest
    A trade's scaled return is  sum_k w_k * pnl_k  over the tranches, so by
    linearity of expectation the MEAN is the weighted mean of the component
    exits' means. A blend therefore CANNOT beat its best component on
    expectancy. Since check_exit_sweep found trail50 (+15.3%) beats
    tp100/slnone (+12.4%) and EOD badly, any blend of TPs with a runner is
    bounded above by pure trail50.

    So this script is NOT looking for a higher mean -- that is precluded. It
    measures the three things linearity does NOT cover:
      1. VARIANCE. Lower dispersion at similar mean is a real benefit: it is
         what lets a book carry more capital at the same risk. That is a
         portfolio-construction gain, not an edge, and is reported as such.
      2. THE SEQUENTIAL GUARD. Partial exits do NOT free the ticker -- the
         position is open until the last tranche closes -- so the guard admits
         a DIFFERENT set of trades than a full exit would. check_signal_exits
         showed a faster exit inflated n from 105 to 416 and diluted the book;
         scaling out pushes the other way and that is not captured by any
         weighted average.
      3. FRICTION. Each tranche crosses the spread separately. Modelled here by
         charging every tranche the same cushioned exit fill sim_core applies.

SCHEDULES (fractions must sum to 1; the last entry is the runner)
    trim_light   20% @ +100%, 20% @ +200%, 60% runner on the trail
    trim_even    40% @ +100%, 40% @ +200%, 20% runner
    trim_heavy   40% @  +50%, 40% @ +100%, 20% runner
    half_at_100  50% @ +100%, 50% runner
    (reference: pure trail50 = 100% runner, no trimming)

Usage:
  python check_scaleout.py --rule "IWM HIVOL CALL"
  python check_scaleout.py --tickers SPY QQQ IWM
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import SLICE_EDGES
import sim_core

PRE_LO = pd.Timestamp("2023-10-12").date()
DEPLOY_LO = SLICE_EDGES[0]
SPLIT = pd.Timestamp("2025-08-21").date()

SCHEDULES = {
    "pure trail50":  [],
    "half_at_100":   [(0.50, 1.00)],
    "trim_light":    [(0.20, 1.00), (0.20, 2.00)],
    "trim_even":     [(0.40, 1.00), (0.40, 2.00)],
    "trim_heavy":    [(0.40, 0.50), (0.40, 1.00)],
}


def scaled_pnl(path, eod_m, tranches, trail=0.50):
    """-> (blended pnl per unit capital, exit minute of the LAST tranche).

    Tranches fill at their ROE level on the BID; the runner then follows the
    deployed 50% trail from the running peak. Every tranche pays the cushioned
    exit fill and commission that sim_core charges, so trimming is not made
    artificially cheap.
    """
    import directional_flow_backtester as D
    e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
    if e_mid <= 0:
        return None
    n = len(cl)
    if n < 2:
        return None
    entry = min(round(e_mid + 0.01, 2), round(e_ask, 2))
    if entry <= 0:
        return None
    ref = e_mid

    def fill(i, lvl_hint, level_triggered):
        px = bid[i] if bid[i] > 0 else min(lvl_hint, cl[i])
        if level_triggered:
            px = min(px, lvl_hint)
        if ask is not None and bid[i] > 0 and ask[i] > bid[i]:
            sp = ask[i] - bid[i]
            px = max(0.01, px - sp * (0.5 if px > entry else 1.5))
        return (px - entry) / entry - D.COMMISSION_PCT

    pending = list(tranches)
    booked, done_w = 0.0, 0.0
    peak = ref
    for i in range(n):
        if mods[i] >= eod_m:
            return booked + (1.0 - done_w) * fill(i, cl[i], False), int(mods[i])
        # tranche fills first -- a level touched intrabar is taken
        for w, roe in list(pending):
            lvl = entry * (1.0 + roe)
            if bid[i] >= lvl:
                booked += w * fill(i, lvl, False)
                done_w += w
                pending.remove((w, roe))
        if done_w < 1.0 - 1e-9:
            t = peak * (1 - trail)
            if lo[i] <= t:
                return booked + (1.0 - done_w) * fill(i, t, True), int(mods[i])
        else:
            return booked, int(mods[i])
        peak = max(peak, cl[i])
    return booked + (1.0 - done_w) * fill(n - 1, cl[-1], False), int(mods[-1])


def summarise(rows, label, base=None):
    if not rows:
        print(f"    {label:14} (none)")
        return None
    d = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows], float)
    w = []
    for lo_, hi_ in ((PRE_LO, DEPLOY_LO), (DEPLOY_LO, SPLIT), (SPLIT, SLICE_EDGES[-1])):
        m = (d >= lo_) & (d < hi_)
        w.append(p[m].mean() * 100 if m.sum() else np.nan)
    sharpe = p.mean() / p.std() if p.std() > 0 else np.nan
    delta = f"{(p.mean()-base)*100:>+6.1f}pp" if base is not None else "   --  "
    print(f"    {label:14} n={len(p):>5}  mean {p.mean()*100:>+7.1f}% {delta}  "
          f"sd {p.std()*100:>6.1f}  m/sd {sharpe:>6.3f}  win {(p>0).mean():.2f}  "
          f"PRE {w[0]:>+6.1f} IS {w[1]:>+6.1f} OOS {w[2]:>+6.1f}")
    return p.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", nargs="*", default=None)
    ap.add_argument("--tickers", nargs="*", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES, TRAIL_PCT

    if a.rule:
        jobs = [next(r for r in RULES if r["name"] == n) for n in a.rule]
    else:
        jobs = [{"name": f"{tk} {dr}", "ticker": tk, "direction": dr,
                 "dte": [0, 1], "min_flow_pct": a.pct,
                 "target_roe": 1.0, "rr": 1.0}
                for tk in a.tickers for dr in a.dirs]

    seq = {k: [] for k in SCHEDULES}
    fixed = {k: [] for k in SCHEDULES}
    for rule in jobs:
        tk = rule["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        cand = sim_core.build_candidates(D, rule, trigs=trigs, since=None)
        if not cand:
            continue
        em = sim_core.eod_mod(rule)
        for name, tr in SCHEDULES.items():
            busy, cur = -1, None
            for d_, m_, path in cand:
                if d_ != cur:
                    cur, busy = d_, -1
                r = scaled_pnl(path, em, tr, TRAIL_PCT or 0.50)
                if r is None:
                    continue
                fixed[name].append((d_, r[0]))
                if m_ < busy:
                    continue
                seq[name].append((d_, r[0]))
                busy = r[1]
        print(f"  {rule['name']} done", flush=True)

    for label, store in (("SEQUENTIAL (the live guard)", seq),
                         ("AS-SCREENED (sample held fixed)", fixed)):
        print(f"\n{'='*118}\n  SCALE-OUT SCHEDULES -- {label}\n{'='*118}")
        base = None
        for name in SCHEDULES:
            m = summarise(store[name], name, base)
            if name == "pure trail50":
                base = m
    print(f"\n  MEAN cannot beat the best component -- a blend is a weighted")
    print(f"  average (see the docstring). Judge these on sd and m/sd: lower")
    print(f"  dispersion at a similar mean is a CAPITAL-EFFICIENCY gain, which")
    print(f"  is a different and more modest claim than an edge.")
    print(f"  Compare n between the two blocks: partial exits hold the ticker")
    print(f"  LONGER, so the guard admits fewer trades -- the one effect a")
    print(f"  weighted average cannot show.")


if __name__ == "__main__":
    main()
