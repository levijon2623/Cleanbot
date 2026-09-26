# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_crossing_quality.py
=========================
NOT EVERY EMA CROSSOVER IS THE SAME EVENT. DOES THE DIFFERENCE PAY?

THE MECHANICS
    ema[i] = a*cum[i] + (1-a)*ema[i-1],  a = 2/(5+1)
    gap[i] = cum[i] - ema[i] = (1-a)*(cum[i] - ema[i-1])
    A bullish cross needs gap[i-1] <= 0 and gap[i] > 0, which reduces to

        Δcum  >  |gap[i-1]|

    The bar is set by how far below the EMA cum ALREADY was. When cum has been
    hugging the EMA that gap is ~0 and an arbitrarily small Δcum crosses it.
    So one trigger name covers two events: flow ARRIVING (cum surges through a
    wide gap) and the EMA CONVERGING onto a flat cum. Passive vs aggressive.

MEASURED ON IWM -- 49,612 crossovers, 717 sessions, 69.2/day (matching the
"~70 crossovers/day/ticker" noted at bot_runner.py:42):
        Δcum vs a normal minute    p10 0.57x   p50 2.30x   p90 12.2x   p99 67.8x
        22% of crossovers move LESS than a typical minute, median Δcum $36k
        25% move >5x,                                      median Δcum $562k
    A 15x difference in the flow doing the crossing, and the trigger treats
    them identically.

WHY NOTHING ALREADY CATCHES THIS
    `min_flow_pct` gates on |cum_flow| -- a LEVEL, orthogonal to HOW the cross
    happened. A day with plenty of accumulated conviction and a $36k
    EMA-convergence twitch passes the gate today. Distinct from
    check_flow_accel (sub-minute acceleration AROUND a crossover),
    check_flow_spike (unconditional spikes -- null), and check_flow_threshold
    (the level itself -- load-bearing). Crossing quality has never been tested.

PRE-COMMITTED: the primary test is the 5x threshold, chosen BEFORE the run
because it is the split the IWM diagnostic used, not because it won. The other
thresholds exist for the MONOTONICITY criterion, not to be argmaxed.

🚨 THREE GUARDS, EACH FOR A WAY TODAY ALREADY WENT WRONG
    MAGNITUDE FLOOR   check_flow_spike passed 4 of 5 pre-committed criteria on
                      +0.34bp because they were sign tests. C1 is in ROE per
                      trade with an explicit floor.
    RANDOM-SPLIT      check_vwap's hypothesis beat baseline by +0.7/trade and
    PLACEBO           sat at the 63rd percentile of a random subset of the same
                      size. Any filter that trades less must beat that.
    MONOTONICITY      a real effect should STRENGTHEN as the cut tightens. One
                      bright cell among four thresholds is arithmetic.
    And the MIRROR (passive-only) is reported: if aggressive and passive are
    both near baseline, the variable carries nothing -- the diagnostic that
    called check_vwap correctly.

PRE-COMMITTED CRITERIA
    C1  5x variant beats baseline by >= 5.0 ROE per trade
    C2  and its TOTAL does not fall below baseline
    C3  IS and OOS both improve on their own baseline half
    C4  beats a random split of equal size (100 draws, 95th percentile)
    C5  >= 6 of 9 rules improve
    C6  per-trade is monotone non-decreasing across 1x -> 2x -> 5x -> 10x

Usage:
  python check_crossing_quality.py --paper
  python check_crossing_quality.py --paper --win 60
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
SEED = 20260921
FLOOR = 5.0
CUTS = [1.0, 2.0, 5.0, 10.0]
PRIMARY = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--win", type=int, default=30,
                    help="trailing window for the median |Δcum|, current minute EXCLUDED")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rows, per_rule = [], {}
    diag = []
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk = rule["ticker"]
        f = _flow_for(D, [tk])
        if f.empty:
            continue
        g = f[f["underlying_symbol"] == tk].copy()
        ts = pd.to_datetime(g["minute_et"])
        g["date"] = ts.dt.date
        g["mod"] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)

        # (date, mod) -> ratio of |Δcum| to the trailing median |Δcum|
        ratio = {}
        for d, x in g.groupby("date"):
            x = x.sort_values("mod")
            cum = x["cum_flow"].to_numpy(float)
            mods = x["mod"].to_numpy(int)
            if len(cum) < a.win + 5:
                continue
            dcum = np.abs(np.diff(cum, prepend=cum[0]))
            med = (pd.Series(dcum).rolling(a.win, min_periods=10)
                   .median().shift(1).to_numpy())        # causal
            for i in range(len(cum)):
                if np.isfinite(med[i]) and med[i] > 0:
                    ratio[(d, int(mods[i]))] = float(dcum[i] / med[i])

        cand = sim_core.build_candidates(D, rule)
        if not cand:
            continue
        rr = [ratio.get((d, m)) for d, m, _p in cand]
        known = [x for x in rr if x is not None]
        diag.append(dict(rule=rule["name"], n=len(cand),
                         known=len(known) / max(len(cand), 1) * 100,
                         med=float(np.median(known)) if known else np.nan,
                         passive=float(np.mean([x <= 1 for x in known]) * 100)
                         if known else np.nan))

        pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        variants = {"baseline (all)": lambda x: True}
        for c in CUTS:
            variants[f"aggressive >={c:g}x"] = (lambda x, c=c:
                                                x is not None and x >= c)
        variants["MIRROR passive <1x"] = lambda x: x is not None and x < 1.0

        per_rule[rule["name"]] = {}
        for name, fn in variants.items():
            keep = [c for c, x in zip(cand, rr) if fn(x)]
            r = sim_core.walk(keep, pol, eod, fill=a.fill, cush_cap=cap)
            recs = [dict(rule=rule["name"], date=d, pnl=p * 100) for d, p in r]
            per_rule[rule["name"]][name] = recs
            rows += [dict(variant=name, **x) for x in recs]
        print(f"    {rule['name']} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  nothing"); return
    R["half"] = np.where(R["date"] <= SPLIT, "IS", "OOS")
    R.to_parquet("_crossing_quality.parquet", index=False)
    names = list(dict.fromkeys(R["variant"]))
    BASE = names[0]
    PRIM = f"aggressive >={PRIMARY:g}x"

    print(f"\n{'='*104}")
    print(f"  0. WHAT THE GATE-PASSING TRIGGERS LOOK LIKE")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'candidates':>11} {'ratio known':>12} "
          f"{'median ratio':>13} {'passive <1x':>12}")
    for x in diag:
        print(f"  {x['rule']:24} {x['n']:>11} {x['known']:>11.0f}% "
              f"{x['med']:>13.1f} {x['passive']:>11.1f}%")

    print(f"\n{'='*104}")
    print(f"  1. THE BOOK UNDER EACH CROSSING-QUALITY CUT")
    print(f"{'='*104}")
    print(f"  {'variant':22} {'n':>6} {'total':>10} {'per trade':>11} "
          f"{'win':>7} {'IS/tr':>9} {'OOS/tr':>9}")
    S = {}
    for nm in names:
        q = R[R["variant"] == nm]
        S[nm] = dict(n=len(q), tot=q["pnl"].sum(), per=q["pnl"].mean(),
                     is_=q[q["half"] == "IS"]["pnl"].mean(),
                     oos=q[q["half"] == "OOS"]["pnl"].mean())
        print(f"  {nm:22} {S[nm]['n']:>6} {S[nm]['tot']:>+10.0f} "
              f"{S[nm]['per']:>+11.1f} {(q['pnl']>0).mean()*100:>6.1f}% "
              f"{S[nm]['is_']:>+9.1f} {S[nm]['oos']:>+9.1f}")

    print(f"\n{'='*104}")
    print(f"  2. C4 -- vs A RANDOM SPLIT OF THE SAME SIZE")
    print(f"{'='*104}")
    b = R[R["variant"] == BASE]["pnl"].to_numpy()
    rng = np.random.default_rng(SEED)
    print(f"  {'variant':22} {'n':>6} {'per trade':>11} {'rand p50':>10} "
          f"{'rand p95':>10} {'pctile':>8}")
    pct = {}
    for nm in names[1:]:
        n = min(S[nm]["n"], len(b))
        if n < 10:
            continue
        dr = np.array([rng.choice(b, size=n, replace=False).mean()
                       for _ in range(100)])
        pct[nm] = (dr < S[nm]["per"]).mean() * 100
        print(f"  {nm:22} {S[nm]['n']:>6} {S[nm]['per']:>+11.1f} "
              f"{np.percentile(dr,50):>+10.1f} {np.percentile(dr,95):>+10.1f} "
              f"{pct[nm]:>7.0f}%")

    print(f"\n{'='*104}")
    print(f"  3. C5 -- PER RULE  ({PRIM} vs baseline, total ROE)")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'baseline':>12} {'aggressive':>12} {'delta':>9} "
          f"{'n base':>8} {'n agg':>7}")
    nimp = 0
    for nm, v in per_rule.items():
        tb = sum(x["pnl"] for x in v[BASE])
        ta = sum(x["pnl"] for x in v[PRIM])
        nimp += int(ta > tb)
        print(f"  {nm:24} {tb:>+12.0f} {ta:>+12.0f} {ta-tb:>+9.0f} "
              f"{len(v[BASE]):>8} {len(v[PRIM]):>7}")

    print(f"\n{'='*104}")
    print(f"  SCORECARD  (pre-committed; primary cut = {PRIMARY:g}x)")
    print(f"{'='*104}")
    m = lambda ok: "PASS" if ok else "FAIL"
    d = S[PRIM]["per"] - S[BASE]["per"]
    mono = [S[f"aggressive >={c:g}x"]["per"] for c in CUTS]
    is_ok = S[PRIM]["is_"] > S[BASE]["is_"] and S[PRIM]["oos"] > S[BASE]["oos"]
    print(f"  C1  per-trade beats baseline by >={FLOOR:.0f}   {d:>+7.1f}   "
          f"{m(d >= FLOOR)}")
    print(f"  C2  total not below baseline        "
          f"{S[PRIM]['tot'] - S[BASE]['tot']:>+7.0f}   "
          f"{m(S[PRIM]['tot'] >= S[BASE]['tot'])}")
    print(f"  C3  BOTH halves improve      IS {S[PRIM]['is_']:>+6.1f} vs "
          f"{S[BASE]['is_']:>+6.1f} / OOS {S[PRIM]['oos']:>+6.1f} vs "
          f"{S[BASE]['oos']:>+6.1f}   {m(is_ok)}")
    print(f"  C4  beats random-split p95          "
          f"{pct.get(PRIM, 0):>6.0f}%   {m(pct.get(PRIM, 0) >= 95)}")
    print(f"  C5  rules improved                    {nimp:>3}/{len(per_rule)}  "
          f"  {m(nimp >= 6)}")
    print(f"  C6  monotone across cuts   " +
          " -> ".join(f"{x:+.1f}" for x in mono) +
          f"   {m(all(mono[i] <= mono[i+1] + 1e-9 for i in range(len(mono)-1)))}")
    mir = S["MIRROR passive <1x"]
    print(f"\n  MIRROR (passive <1x): {mir['per']:+.1f}/trade on {mir['n']} "
          f"trades vs baseline {S[BASE]['per']:+.1f}.")
    print(f"  If aggressive AND passive both sit on baseline, crossing quality")
    print(f"  carries nothing -- the check that called check_vwap correctly.")


if __name__ == "__main__":
    main()
