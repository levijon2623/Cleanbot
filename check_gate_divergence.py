# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_gate_divergence.py
========================
DOES THE LIVE FLOW GATE MEAN THE SAME THING AS THE BACKTESTED ONE?

THE SUSPICION
    `_live_flow_threshold` is a percentile of the bot's OWN crossover history,
    built by `_log_flow_trigger`. But that call sits below
    bot_runner.py:1708 -- `if ticker in self.active_snipes: continue` -- so a
    crossover occurring while a position is open is NEVER LOGGED. Research does
    the opposite: `annotate_flow_pct` runs over `triggers_for()`, every
    crossover, no guard applied.

    So the live gate is a percentile of a SUBSET and the backtested gate is a
    percentile of the WHOLE. Same name, same code path downstream, different
    population. That is the shape of the two divergences already found this
    week (`strike_offset` honoured live but not in research; a stale entry spot
    in build_candidates), and it is worth knowing the size before touching a
    live gate.

TWO DIFFERENCES, NOT ONE -- SO MEASURE THEM SEPARATELY
    A. AGGREGATION. Research takes the percentile over EVERY crossover in the
       trailing window. Live takes ONE VALUE PER DAY (that day's MEDIAN
       crossover flow), deliberately -- bot_runner:44 says a choppy session
       would otherwise flood the sample with near-duplicates, and claims the
       two agree "within ~5%". That claim has never been re-checked here.
    B. POPULATION. All crossovers vs crossovers-outside-holds. The new one.

    Reported as a 2x2 so A and B can be read apart. Collapsing them would let a
    known, deliberate 5% difference hide an unknown one.

WHAT ACTUALLY MATTERS
    Not the threshold gap in dollars -- the FLIPS. A gate that differs by 8% but
    never changes a trigger's verdict costs nothing. Section 3 counts triggers
    whose pass/fail changes, and section 4 walks the book under each gate so the
    difference is priced in ROE rather than in percent.

HOLD WINDOWS come from sim_core.walk(picks_out=...) -- the entry and exit minute
of every trade the sequential guard actually took, which is exactly the interval
during which live would have been blind.

RESULT -- 2026-09-20. THE DIVERGENCE IS REAL AND THE P&L SIGN DEPENDS ON WHICH
BOOK, WHICH IS NOT THE ANSWER THE SUSPICION PREDICTED.

    Live is blind to 0.7%-4.8% of crossovers (IWM worst, SPY least) -- far less
    than the 94% guard-refusal rate might suggest, because holds are a small
    share of session MINUTES even when they refuse most signals.

    Threshold gap vs research (median, and 90th pct of |gap|):
        rule    live/cross      all/daily     live/daily
        QQQ       -1.1%±11      -14.1%±26      -17.1%±29
        GLD       -3.8%±16      -10.5%±30      -18.6%±34
        IWM       -7.6%±22       -5.7%±14      -12.6%±29
        SPY       +0.0%± 7       -8.3%±17      -10.2%±20
    Verdicts flip on 2.2%-5.4% of triggers under the live gate.

    🚨 bot_runner.py:44 CLAIMS THE DAILY-MEDIAN AGGREGATION AGREES WITH THE
    PER-CROSSOVER PERCENTILE "within ~5%". It does not, for 4 of 9 rules:
    QQQ -14.1%, GLD -10.5%, SPY -8.3%, IWM -5.7%. That comment should be
    corrected whatever else is decided.

    Priced in ROE -- and this is where the prediction failed:
                        all/cross   live/cross   all/daily   live/daily
        CORE FIVE          +4,053      +3,907     +5,114       +4,391
        PAPER FOUR           -267      ...                     -1,864
        BOOK TOTAL         +3,785      +3,048     +4,152       +2,527
    Research scores all/cross; the bot runs live/daily. Book-wide that is
    -1,258 ROE (-33%), but it is NOT evenly spread: the CORE FIVE are +338
    BETTER under the live gate, and the entire deficit is the paper rules
    (GLD alone -1,147, MSFT -456).

    Mechanism, and it is coherent: crossovers during a hold carry HIGH
    cumulative flow (a hold begins after a big one, on an active day), so
    excluding them LOWERS the percentile -> a LOOSER live gate -> more trades.
    check_flow_threshold already showed looser is worse. It shows up as IWM
    105 -> 128 trades and +787 -> +337.

    SO: fixing the population difference is a CORRECTNESS fix -- live and
    research must compute the same quantity or every backtest scores a strategy
    the bot does not run -- but it is NOT a profit fix, and on the core five the
    current (wrong) gate is slightly ahead. Decide it on correctness grounds.

    UNMEASURED: live defaults to a 90d window, research to 60d; this ran at each
    rule's own setting. The 20-day FLOW_HISTORY_MIN_DAYS warmup and the static-
    JSON fallback before it are not modelled either.
    NOT ACTED ON: all/daily beating all/cross on the core five (+5,114 vs
    +4,053) is an in-sample observation on one sample -- the exact shape that
    check_flow_walkforward just rejected. Flagged, not banked.

Usage:
  python check_gate_divergence.py --paper
  python check_gate_divergence.py --window 90     # the LIVE default
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

GRID = [50, 65, 80, 90, 95]


def rolling_thr(trigs, mask, window_days, pcts, daily_median=False):
    """For each trigger, the Pth percentile of the trailing window, using only
    triggers where `mask` is True and only days STRICTLY BEFORE.

    Mirrors directional_flow_backtester.annotate_flow_pct exactly -- same
    stable sort, same >=30-sample floor, same 'a full window of history must
    have elapsed' rule -- and adds the mask plus the live daily-median option.
    """
    n = len(trigs)
    out = [None] * n
    if not n:
        return out
    order = np.argsort([t["ts"] for t in trigs], kind="stable")
    day_ns = np.array([pd.Timestamp(trigs[i]["ts"]).normalize().value
                       for i in order])
    flows = np.array([trigs[i]["abs_flow"] for i in order], float)
    keep = np.array([bool(mask[i]) for i in order])
    win = int(window_days) * 86_400_000_000_000
    first = day_ns[0]
    for pos, i in enumerate(order):
        cur = day_ns[pos]
        if cur - first < win:
            continue
        lo = int(np.searchsorted(day_ns, cur - win, side="left"))
        hi = int(np.searchsorted(day_ns, cur, side="left"))
        sl = slice(lo, hi)
        h, d, k = flows[sl], day_ns[sl], keep[sl]
        h, d = h[k], d[k]
        if daily_median:
            # one value per day = that day's MEDIAN crossover flow
            if len(d) == 0:
                continue
            ud = np.unique(d)
            h = np.array([np.median(h[d == u]) for u in ud])
        if len(h) < 30:
            continue
        out[i] = {P: float(np.percentile(h, P)) for P in pcts}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--window", type=int, default=None,
                    help="trailing days; default = each rule's own setting")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rows, pnl_rows = [], []
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dep = rule["ticker"], int(rule.get("min_flow_pct") or 0)
        if not dep:
            continue
        win = a.window or int(rule.get("flow_window_days", 60))
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, win)
        tmap = {(t["date"], pd.Timestamp(t["ts"]).hour * 60
                 + pd.Timestamp(t["ts"]).minute): i
                for i, t in enumerate(trigs)}

        # --- the book as deployed, to recover the hold windows ---
        cand = sim_core.build_candidates(D, rule)
        if not cand:
            continue
        pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        picks = []
        sim_core.walk(cand, pol, eod, fill=a.fill, cush_cap=cap,
                      picks_out=picks)
        holds = {}
        for ci, xm in picks:
            d, m, _p = cand[ci]
            holds.setdefault(d, []).append((int(m), int(xm)))

        # A crossover is INVISIBLE to the live bot if it lands inside a hold.
        held = []
        for t in trigs:
            m = pd.Timestamp(t["ts"]).hour * 60 + pd.Timestamp(t["ts"]).minute
            held.append(any(e <= m <= x for e, x in holds.get(t["date"], [])))
        held = np.array(held)
        vis = ~held

        thr = {
            ("all", "cross"): rolling_thr(trigs, ~held | held, win, GRID),
            ("live", "cross"): rolling_thr(trigs, vis, win, GRID),
            ("all", "daily"): rolling_thr(trigs, ~held | held, win, GRID, True),
            ("live", "daily"): rolling_thr(trigs, vis, win, GRID, True),
        }

        # --- how much do the thresholds differ, and does the verdict flip? ---
        idx = [i for i, t in enumerate(trigs)
               if t["dir"] == rule["direction"] and t["date"] >= sim_core.DEPLOYED_START]
        for (pop, agg), series in thr.items():
            gaps, flips, n = [], 0, 0
            base = thr[("all", "cross")]
            for i in idx:
                b, s = base[i], series[i]
                if not b or not s:
                    continue
                n += 1
                gaps.append((s[dep] - b[dep]) / b[dep] * 100)
                f = trigs[i]["abs_flow"]
                if (f >= s[dep]) != (f >= b[dep]):
                    flips += 1
            if n:
                rows.append(dict(rule=rule["name"], pop=pop, agg=agg, n=n,
                                 gap=float(np.median(gaps)),
                                 gap90=float(np.percentile(np.abs(gaps), 90)),
                                 flip=flips / n * 100,
                                 held=float(held[idx].mean() * 100)))

        # --- price the difference: walk the book under each gate ---
        loose = {**rule, "min_flow_pct": 50}
        lc = sim_core.build_candidates(D, loose, trigs=trigs)
        ct = [tmap.get((d, m)) for d, m, _p in lc]
        for (pop, agg), series in thr.items():
            keep = []
            for c, ti in zip(lc, ct):
                if ti is None or not series[ti]:
                    continue
                if trigs[ti]["abs_flow"] >= series[ti][dep]:
                    keep.append(c)
            r = sim_core.walk(keep, pol, eod, fill=a.fill, cush_cap=cap)
            pnl_rows.append(dict(rule=rule["name"], pop=pop, agg=agg,
                                 n=len(r), tot=sum(p for _d, p in r) * 100))
        print(f"    {rule['name']} done  (p{dep}, {win}d, "
              f"{held[idx].mean()*100:.0f}% of crossovers inside a hold)",
              flush=True)

    R, P = pd.DataFrame(rows), pd.DataFrame(pnl_rows)
    if R.empty:
        print("  nothing"); return

    print(f"\n{'='*104}")
    print(f"  1. HOW MUCH OF THE CROSSOVER HISTORY IS LIVE ACTUALLY BLIND TO?")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'% of crossovers inside a hold':>34}")
    for nm, g in R.groupby("rule", sort=False):
        print(f"  {nm:24} {g['held'].iloc[0]:>33.1f}%")

    print(f"\n{'='*104}")
    print(f"  2. THRESHOLD GAP vs the research gate (all crossovers, "
          f"per-crossover)")
    print(f"     median % difference in the p-gate, and the 90th pct of |gap|")
    print(f"{'='*104}")
    print(f"  {'rule':24} " + "".join(
        f"{p+'/'+a_:>16}" for p, a_ in
        (("live", "cross"), ("all", "daily"), ("live", "daily"))))
    for nm, g in R.groupby("rule", sort=False):
        cells = ""
        for pop, agg in (("live", "cross"), ("all", "daily"), ("live", "daily")):
            r = g[(g["pop"] == pop) & (g["agg"] == agg)]
            cells += (f"{r['gap'].iloc[0]:>+9.1f}%±{r['gap90'].iloc[0]:>4.0f}"
                      if len(r) else f"{'--':>16}")
        print(f"  {nm:24} {cells}")
    print(f"\n  'all/daily' isolates the KNOWN aggregation difference "
          f"(bot_runner:44 claims ~5%).")
    print(f"  'live/cross' isolates the POPULATION difference -- the new one.")

    print(f"\n{'='*104}")
    print(f"  3. THE NUMBER THAT MATTERS -- % of triggers whose VERDICT flips")
    print(f"{'='*104}")
    print(f"  {'rule':24} " + "".join(
        f"{p+'/'+a_:>16}" for p, a_ in
        (("live", "cross"), ("all", "daily"), ("live", "daily"))))
    for nm, g in R.groupby("rule", sort=False):
        cells = ""
        for pop, agg in (("live", "cross"), ("all", "daily"), ("live", "daily")):
            r = g[(g["pop"] == pop) & (g["agg"] == agg)]
            cells += f"{r['flip'].iloc[0]:>15.1f}%" if len(r) else f"{'--':>16}"
        print(f"  {nm:24} {cells}")

    print(f"\n{'='*104}")
    print(f"  4. PRICED IN ROE -- the book walked under each gate")
    print(f"{'='*104}")
    print(f"  {'rule':24} " + "".join(f"{p+'/'+a_:>15}" for p, a_ in
          (("all", "cross"), ("live", "cross"), ("all", "daily"),
           ("live", "daily"))))
    for nm, g in P.groupby("rule", sort=False):
        cells = ""
        for pop, agg in (("all", "cross"), ("live", "cross"),
                         ("all", "daily"), ("live", "daily")):
            r = g[(g["pop"] == pop) & (g["agg"] == agg)]
            cells += (f"{r['tot'].iloc[0]:>+10.0f}/{int(r['n'].iloc[0]):<4d}"
                      if len(r) else f"{'--':>15}")
        print(f"  {nm:24} {cells}")
    tot = P.groupby(["pop", "agg"]).agg(tot=("tot", "sum"), n=("n", "sum"))
    print(f"  {'BOOK TOTAL':24} " + "".join(
        f"{tot.loc[(p, a_), 'tot']:>+10.0f}/{int(tot.loc[(p, a_), 'n']):<4d}"
        for p, a_ in (("all", "cross"), ("live", "cross"),
                      ("all", "daily"), ("live", "daily"))))
    base = tot.loc[("all", "cross"), "tot"]
    print(f"\n  Research scores the book as 'all/cross'. The bot runs "
          f"'live/daily'.")
    print(f"  Difference: {tot.loc[('live','daily'),'tot'] - base:+.0f} ROE "
          f"({(tot.loc[('live','daily'),'tot'] - base)/abs(base)*100:+.1f}%)")


if __name__ == "__main__":
    main()
