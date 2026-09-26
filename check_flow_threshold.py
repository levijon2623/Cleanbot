# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_flow_threshold.py
=======================
CAN THE REGIME GATES CARRY A LOOSER FLOW GATE?

THE QUESTION
    The book was built trigger-first: EMA(5) crossover of cumulative net
    premium, then `min_flow_pct` (a trailing-60d percentile of the trigger's own
    |cumulative flow|), then the regime / hour / amp / amt gates on top. The
    deployed percentiles are high -- 65, 80, 90, even 95 for QQQ -- which is a
    hard ration on trade count. If the gates downstream are doing real work,
    they should be able to hold up a LOWER flow bar and buy back trades.

    So: sweep `min_flow_pct` from 95 down to 20 and ask whether the trades that
    a looser gate ADDS are worth taking.

🚨 THREE WAYS THIS QUESTION ANSWERS ITSELF IF YOU LET IT
    1  MORE TRADES AT POSITIVE EXPECTANCY RAISES GROSS P&L AUTOMATICALLY.
       Total ROE is not the test. The test is the expectancy of the MARGINAL
       BAND -- the trades sitting between the loose threshold and the deployed
       one -- scored on their own. If those are flat, the extra gross is just
       more capital, and section 3 prices that.

    2  `abs_flow` IS PARTLY A CLOCK. Cumulative flow grows through the session
       by construction, so a low percentile preferentially admits EARLY
       triggers. Lowering the gate can therefore look like a flow finding when
       it is really an hour finding, and every rule already has an `hours` gate
       that was tuned separately. Section 4 reports the marginal band's hour mix
       against the deployed band's, and re-scores it WITHIN hour.

    3  THE SEQUENTIAL GUARD MEANS ADDED TRADES DISPLACE EXISTING ONES.
       One position per ticker: a new low-flow trigger at 10:05 takes the slot
       that a high-flow trigger at 11:20 would have used. That is the same
       capital-displacement effect that sank check_reentry, and it means the
       loose book is NOT the strict book plus extras. Section 3 walks both
       sequentially so the displacement is priced, and reports how many deployed
       trades survive.

PRE-COMMITTED CRITERIA -- fixed before the first run
    T1  the marginal band has positive STANDALONE expectancy
    T2  the marginal band agrees in sign across IS and OOS
    T3  the sequentially-walked book improves in total ROE at the looser gate
    T4  >= 6 of 9 rules improve
    T5  the marginal band's edge is not merely an hour shift -- it survives
        restricted to the hours the deployed band already trades

RESULT -- 2026-09-20, RE-BASELINED after the spot fix (sim_core.py, the `bbd`
sort). The pre-fix run reached the same verdict on a stale strike; these are the
numbers to cite.
    THE GATES CANNOT CARRY A LOOSER FLOW BAR. All five criteria FAIL.

        band                        n   ROE/trade   win%
        deployed (p65-p95)      2,575        +9.9   43.0
        marginal (p20->dep)     8,848       -11.2   25.9     IS -9.6 / OOS -12.6

    Book total, sequentially walked, is monotonic in the percentile:
        p20      p35      p50      p65      p80      p90      p95
    -13,080  -10,880   -4,484     -269   +3,994   +4,721   +1,653
       1630     1371     1086      748      425      231      130

    NOT an hour artifact. `abs_flow` is partly a clock -- the marginal band is
    29% pre-11am against the deployed band's 10% -- but re-weighting the
    marginal band to the deployed hour mix moved it -11.2 -> -11.2. Exactly
    zero, the same as pre-fix. The flow gate works AS A FLOW GATE.

    🚨 THE GRADIENT GOT STRONGER, NOT WEAKER, ON THE FIX:
        pctile    p20    p35    p50    p65    p80    p90    p95
        IS/tr    -6.6   -8.4   -6.9   -0.4   +4.7  +20.7   +0.7
        OOS/tr   -9.2   -7.6   -1.5   -0.4  +14.3  +20.2  +23.7
    At p90 the halves now agree to within 0.5 ROE/trade (+20.7 vs +20.2), where
    pre-fix they read +13.7 vs +31.4. Two independently-fitted halves landing on
    the same number is much harder to dismiss as an argmax than one big cell was.
    The p90 book totals MORE than the deployed mix (+4,721 on 231 trades vs
    +3,785 on 392) -- i.e. more money on 41% fewer trades.

    🚨 STILL NOT A LICENCE TO RAISE THE DEPLOYED THRESHOLDS. This is an argmax
    over seven values read off one table, which is the selection METHODOLOGY 5
    forbids, however well the halves agree. The honest version picks each rule's
    percentile on IS ALONE and scores it on OOS, and it has not been run.
    NOTE ALSO: GLD amp1 CALL is negative at EVERY percentile (-7,149 at p20 to
    -649 at p90) and drags the book total wherever it appears. It is in
    PAPER_ONLY. Re-read this table with --paper omitted before concluding
    anything about the core five.

Usage:
  python check_flow_threshold.py --paper
  python check_flow_threshold.py --paper --loose 35
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import sim_core

GRID = [20, 35, 50, 65, 80, 90, 95]
SPLIT = pd.Timestamp("2025-08-21").date()


def mod_of(ts):
    t = pd.Timestamp(ts)
    return int(t.hour) * 60 + int(t.minute)


def ex(v):
    """Total and per-trade expectancy of a list of ROE percentages."""
    if not len(v):
        return 0.0, np.nan, np.nan, 0
    a = np.asarray(v, float)
    return float(a.sum()), float(a.mean()), float((a > 0).mean() * 100), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--loose", type=int, default=20,
                    help="the looser percentile to compare against deployed")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    # The deployed grid stops at 50. Research needs to look BELOW the floor the
    # book was built on, so extend it here. bot_runner never imports this module
    # and config.py is untouched -- annotate_flow_pct just reads this global.
    D.GRID_MIN_FLOW_PCT = GRID

    rows, per_rule = [], {}
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dep = rule["ticker"], int(rule.get("min_flow_pct") or 0)
        if not dep:
            continue                         # flow_abs rules have no percentile
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        tmap = {(t["date"], mod_of(t["ts"])): t for t in trigs
                if t["dir"] == rule["direction"]}

        loose = {**rule, "min_flow_pct": GRID[0]}
        cand = sim_core.build_candidates(D, loose, trigs=trigs)
        if not cand:
            continue
        ct = [tmap.get((d, m)) for d, m, _p in cand]
        pol = sim_core.policy_for(rule)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)

        def keep(P):
            out = []
            for c, t in zip(cand, ct):
                if t is None or not t.get("thr"):
                    continue
                if t["abs_flow"] >= t["thr"][P]:
                    out.append(c)
            return out

        # --- standalone: every candidate scored on its own, no guard ---
        solo = {}
        for c, t in zip(cand, ct):
            if t is None or not t.get("thr"):
                continue
            pnl, _xm, _tg = sim_core.simulate(c[2], pol, eod, fill=a.fill,
                                              cush_cap=cap)
            solo[(c[0], c[1])] = dict(pnl=pnl * 100, hour=c[1] // 60,
                                      flow=t["abs_flow"], thr=t["thr"])

        # --- sequential: the book as it would actually run at each P ---
        seq = {}
        for P in GRID:
            k = keep(P)
            r = sim_core.walk(k, pol, eod, fill=a.fill, cush_cap=cap)
            seq[P] = [(d, p * 100) for d, p in r]

        per_rule[rule["name"]] = dict(seq=seq, solo=solo, dep=dep, cand=cand,
                                      ct=ct, rule=rule)
        for P in GRID:
            tot, mean, win, n = ex([p for _d, p in seq[P]])
            i_ = [p for d, p in seq[P] if d <= SPLIT]
            o_ = [p for d, p in seq[P] if d > SPLIT]
            rows.append(dict(rule=rule["name"], P=P, dep=dep, n=n, tot=tot,
                             mean=mean, win=win,
                             is_n=len(i_), is_tot=float(np.sum(i_)),
                             oos_n=len(o_), oos_tot=float(np.sum(o_))))
        print(f"    {rule['name']} done  (deployed p{dep})", flush=True)

    if not per_rule:
        print("  nothing"); return
    R = pd.DataFrame(rows)

    print(f"\n{'='*104}")
    print(f"  1. THE SWEEP -- sequentially walked, each rule on its own "
          f"deployed exit")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'dep':>4} " +
          "".join(f"{'p'+str(P):>11}" for P in GRID))
    for nm, g in R.groupby("rule", sort=False):
        g = g.set_index("P")
        cells = "".join(f"{g.loc[P,'tot']:>+7.0f}/{int(g.loc[P,'n']):<3d}"
                        for P in GRID)
        print(f"  {nm:24} {int(g['dep'].iloc[0]):>4} {cells}")
    tot = R.groupby("P").agg(tot=("tot", "sum"), n=("n", "sum"))
    print(f"  {'BOOK TOTAL':24} {'':>4} " +
          "".join(f"{tot.loc[P,'tot']:>+7.0f}/{int(tot.loc[P,'n']):<3d}"
                  for P in GRID))
    print(f"\n  cells are  total ROE / trade count.")

    print(f"\n{'='*104}")
    print(f"  1b. THE SAME SWEEP, SPLIT IS / OOS AT {SPLIT}")
    print(f"      The book total above rises monotonically into p90, which "
          f"invites 'raise the gate'.")
    print(f"      That is a best-of-7 pick on one sample -- METHODOLOGY 5. The "
          f"only thing that")
    print(f"      distinguishes a gradient from an argmin is whether the two "
          f"halves agree.")
    print(f"{'='*104}")
    print(f"  {'pctile':>8} {'IS total':>11} {'IS/tr':>8} {'OOS total':>11} "
          f"{'OOS/tr':>8} {'n':>7}")
    agg = R.groupby("P").agg(is_tot=("is_tot", "sum"), is_n=("is_n", "sum"),
                             oos_tot=("oos_tot", "sum"),
                             oos_n=("oos_n", "sum"), n=("n", "sum"))
    for P in GRID:
        r_ = agg.loc[P]
        print(f"  {'p'+str(P):>8} {r_['is_tot']:>+11.0f} "
              f"{r_['is_tot']/max(r_['is_n'],1):>+8.1f} "
              f"{r_['oos_tot']:>+11.0f} "
              f"{r_['oos_tot']/max(r_['oos_n'],1):>+8.1f} {int(r_['n']):>7}")

    print(f"\n{'='*104}")
    print(f"  2. T1/T2 -- THE MARGINAL BAND, SCORED STANDALONE")
    print(f"     trades a p{a.loose} gate ADDS that the deployed gate refuses.")
    print(f"     No sequential guard here: this is the raw signal, not the book.")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'band':>10} {'n':>5} {'total':>9} {'per tr':>8} "
          f"{'win':>7} {'IS/tr':>8} {'OOS/tr':>8}")
    mall, mis, moos = [], [], []
    dall = []
    for nm, v in per_rule.items():
        marg, dpl = [], []
        for (d, m), s in v["solo"].items():
            hi, lo = s["thr"][v["dep"]], s["thr"][a.loose]
            if s["flow"] >= hi:
                dpl.append((d, s["pnl"]))
            elif s["flow"] >= lo:
                marg.append((d, s["pnl"]))
        v["marg"], v["dpl"] = marg, dpl
        mall += [p for _d, p in marg]
        dall += [p for _d, p in dpl]
        i_ = [p for d, p in marg if d <= SPLIT]
        o_ = [p for d, p in marg if d > SPLIT]
        mis += i_; moos += o_
        t, mn, w, n = ex([p for _d, p in marg])
        print(f"  {nm:24} {'p'+str(a.loose)+'-p'+str(v['dep']):>10} {n:>5} "
              f"{t:>+9.0f} {mn:>+8.1f} {w:>6.1f}% "
              f"{np.mean(i_) if i_ else np.nan:>+8.1f} "
              f"{np.mean(o_) if o_ else np.nan:>+8.1f}")
    t, mn, w, n = ex(mall)
    td, mnd, wd, nd = ex(dall)
    print(f"  {'-'*100}")
    print(f"  {'MARGINAL BAND':24} {'':>10} {n:>5} {t:>+9.0f} {mn:>+8.1f} "
          f"{w:>6.1f}% {np.mean(mis) if mis else np.nan:>+8.1f} "
          f"{np.mean(moos) if moos else np.nan:>+8.1f}")
    print(f"  {'DEPLOYED BAND':24} {'':>10} {nd:>5} {td:>+9.0f} {mnd:>+8.1f} "
          f"{wd:>6.1f}%")
    print(f"\n  T1 marginal expectancy > 0        "
          f"{'PASS' if mn > 0 else 'FAIL'}   ({mn:+.1f} ROE/trade)")
    sgn = (np.mean(mis) > 0) == (np.mean(moos) > 0) if (mis and moos) else False
    print(f"  T2 IS and OOS agree in sign       {'PASS' if sgn else 'FAIL'}")

    print(f"\n{'='*104}")
    print(f"  3. T3/T4 -- THE BOOK, WALKED SEQUENTIALLY (displacement priced in)")
    print(f"{'='*104}")
    dep_tot = sum(R[(R['rule'] == nm) & (R['P'] == v['dep'])]['tot'].iloc[0]
                  for nm, v in per_rule.items())
    dep_n = sum(int(R[(R['rule'] == nm) & (R['P'] == v['dep'])]['n'].iloc[0])
                for nm, v in per_rule.items())
    lo_tot = float(tot.loc[a.loose, "tot"]); lo_n = int(tot.loc[a.loose, "n"])
    print(f"  {'book':26} {'trades':>8} {'total ROE':>11} {'per trade':>11}")
    print(f"  {'deployed percentiles':26} {dep_n:>8} {dep_tot:>+11.0f} "
          f"{dep_tot/max(dep_n,1):>+11.1f}")
    print(f"  {'all rules at p'+str(a.loose):26} {lo_n:>8} {lo_tot:>+11.0f} "
          f"{lo_tot/max(lo_n,1):>+11.1f}")
    nimp = 0
    for nm, v in per_rule.items():
        d_ = R[(R['rule'] == nm) & (R['P'] == v['dep'])]['tot'].iloc[0]
        l_ = R[(R['rule'] == nm) & (R['P'] == a.loose)]['tot'].iloc[0]
        nimp += int(l_ > d_)
    print(f"\n  T3 looser book totals more       "
          f"{'PASS' if lo_tot > dep_tot else 'FAIL'}")
    print(f"  T4 rules improved                 {nimp}/{len(per_rule)}   "
          f"{'PASS' if nimp >= 6 else 'FAIL'}")
    print(f"\n  Per-trade is the honest column: {lo_n} trades against "
          f"{dep_n} is {lo_n/max(dep_n,1):.2f}x the capital.")

    print(f"\n{'='*104}")
    print(f"  4. T5 -- IS THE MARGINAL BAND JUST AN EARLIER CLOCK?")
    print(f"     Cumulative flow grows through the session, so a low percentile")
    print(f"     admits early triggers by construction.")
    print(f"{'='*104}")
    hm = Counter(); hd = Counter()
    for v in per_rule.values():
        for (d, m), s in v["solo"].items():
            hi, lo = s["thr"][v["dep"]], s["thr"][a.loose]
            if s["flow"] >= hi:
                hd[s["hour"]] += 1
            elif s["flow"] >= lo:
                hm[s["hour"]] += 1
    print(f"  {'hour':>6} {'deployed':>12} {'marginal':>12} "
          f"{'marg ROE/tr':>13} {'dep ROE/tr':>12}")
    within = []
    for h in sorted(set(hd) | set(hm)):
        mv, dv = [], []
        for v in per_rule.values():
            for (d, m), s in v["solo"].items():
                if s["hour"] != h:
                    continue
                hi, lo = s["thr"][v["dep"]], s["thr"][a.loose]
                if s["flow"] >= hi:
                    dv.append(s["pnl"])
                elif s["flow"] >= lo:
                    mv.append(s["pnl"])
        if mv and dv:
            within.append((len(mv), np.mean(mv)))
        print(f"  {h:>6} {hd[h]:>7} {hd[h]/max(sum(hd.values()),1)*100:>4.0f}% "
              f"{hm[h]:>7} {hm[h]/max(sum(hm.values()),1)*100:>4.0f}% "
              f"{np.mean(mv) if mv else np.nan:>+13.1f} "
              f"{np.mean(dv) if dv else np.nan:>+12.1f}")
    wgt = (sum(n * mu for n, mu in within) / sum(n for n, _ in within)
           if within else np.nan)
    print(f"\n  Marginal expectancy re-weighted to the DEPLOYED band's hour mix:"
          f" {wgt:+.1f} ROE/trade")
    print(f"  (raw marginal was {mn:+.1f}. A large gap means the flow gate was "
          f"acting as an hour gate.)")
    print(f"  T5 survives the hour control      "
          f"{'PASS' if np.isfinite(wgt) and wgt > 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
