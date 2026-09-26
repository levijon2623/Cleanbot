# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_flow_align.py
===================
IS "CUMULATIVE FLOW ALIGNMENT" A loss50 SKIP RULE ON THE BASELINE BOOK?

THE FINDING THIS CHASES
    check_ema_trigger split EMA-cross entries on whether cumulative net flow
    agreed with the trade: aligned 107 trades, loss50 15.0%, OOS +1964; opposed
    615 trades, loss50 28.1%, OOS -4332. Same trigger, same exit, sign of the
    tide the only difference. This asks whether the same split works on the
    WHALE SPIKE triggers the book actually trades.

🚨 THE CONFOUND THAT COULD FAKE THE WHOLE RESULT
    The trigger is an EMA(5) CROSSOVER of cumulative net premium, not a level.
    A CALL fires when cum-flow crosses ABOVE ITS OWN EMA -- which says nothing
    about cum-flow's SIGN. In the live ledger all 6 CALL entries fired on
    NEGATIVE cum-flow (opposed) and 23 of 25 PUTs on negative cum-flow
    (aligned). If that holds historically, "aligned vs opposed" is largely
    "PUT vs CALL", and the call and put rules already differ for reasons that
    have nothing to do with the tide.
    So the split is reported PER RULE first. Only if alignment varies WITHIN a
    rule is the pooled comparison meaningful, and the within-rule figures are
    the ones to believe.

PRECEDENT: check_skip_hunt tested six loss50 skip features and all six rejected,
    with spearman(loss50, hit50) = +0.664 -- skipping the losers skips the
    winners too. This is the seventh candidate and inherits that prior.

BINS
    aligned        sign(cum_flow at entry) matches the trade direction
    opposed        it does not
    strong         aligned AND |cum_flow| in the top quartile of its own
                   TRAILING 2-hour distribution (causal: the current minute is
                   excluded from the window it is judged against)

Usage:
  python check_flow_align.py --paper
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()


def flow_state(D, tk, win=120, q=0.75):
    """{(date, mod): (cum_flow, is_top_quartile_of_trailing_2h)}"""
    from check_config_walkforward import _flow_for
    f = _flow_for(D, [tk])
    if f.empty:
        return {}
    g = f[f["underlying_symbol"] == tk].copy()
    ts = pd.to_datetime(g["minute_et"])
    g["date"] = ts.dt.date
    g["mod"] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)
    out = {}
    for d, gd in g.sort_values("mod").groupby("date"):
        cum = gd["cum_flow"].astype(float)
        # .shift(1): the current minute must not enter its own percentile
        thr = cum.abs().rolling(win, min_periods=30).quantile(q).shift(1)
        for m, c, t in zip(gd["mod"].to_numpy(int), cum.to_numpy(float),
                           thr.to_numpy(float)):
            out[(d, int(m))] = (float(c), bool(np.isfinite(t) and abs(c) >= t))
    return out


def agg(rows, label):
    if not rows:
        return f"  {label:28} (none)"
    v = np.array([r["pnl"] for r in rows], float)
    o = [r["pnl"] for r in rows if r["date"] >= SPLIT]
    dset = {r["date"] for r in rows}
    return (f"  {label:28} {len(v):>6} {len(dset):>6} "
            f"{(v <= -50).mean()*100:>7.1f}% {(v > 0).mean()*100:>6.1f}% "
            f"{np.median(v):>+9.1f} {(np.sum(o) if o else np.nan):>+10.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--seed", type=int, default=53)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D

    rows = []
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dirn = rule["ticker"], rule["direction"]
        fs = flow_state(D, tk)
        if not fs:
            continue
        cand = sim_core.build_candidates(D, rule)
        if not cand:
            continue
        pol = sim_core.policy_for(rule)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        want = 1 if dirn == "CALL" else -1
        for d, m, path in cand:
            st = fs.get((d, int(m)))
            if st is None:
                continue
            cum, strong = st
            pnl, xm, tag = sim_core.simulate(path, pol, eod, fill=a.fill,
                                             cush_cap=cap)
            rows.append(dict(rule=rule["name"], dir=dirn, date=d,
                             pnl=pnl * 100, cum=cum,
                             aligned=bool(np.sign(cum) == want),
                             strong=bool(np.sign(cum) == want and strong)))
        print(f"    {rule['name']} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  nothing"); return
    R.to_parquet("_flow_align.parquet", index=False)

    print(f"\n{'='*100}")
    print(f"  1. IS ALIGNMENT EVEN VARIABLE WITHIN A RULE?  (the confound check)")
    print(f"{'='*100}")
    print(f"  {'rule':24} {'dir':5} {'n':>6} {'aligned':>9} {'opposed':>9} "
          f"{'strong':>8}")
    for nm, g in R.groupby("rule"):
        print(f"  {nm:24} {g['dir'].iloc[0]:5} {len(g):>6} "
              f"{g['aligned'].mean()*100:>8.1f}% {(~g['aligned']).mean()*100:>8.1f}% "
              f"{g['strong'].mean()*100:>7.1f}%")
    near = [nm for nm, g in R.groupby("rule")
            if g["aligned"].mean() > 0.95 or g["aligned"].mean() < 0.05]
    print(f"\n  rules that are ~always one way: {near if near else 'none'}")
    print(f"  -> those contribute NOTHING to a within-rule comparison; a pooled")
    print(f"     split across them would be comparing rules, not flow states.")

    print(f"\n{'='*100}")
    print(f"  2. POOLED (confounded by direction -- shown for completeness)")
    print(f"{'='*100}")
    print(f"  {'bin':28} {'n':>6} {'days':>6} {'loss50':>8} {'win':>7} "
          f"{'med ROE':>9} {'OOS tot':>10}")
    recs = R.to_dict("records")
    print(agg(recs, "ALL"))
    print(agg([r for r in recs if r["aligned"]], "aligned"))
    print(agg([r for r in recs if not r["aligned"]], "opposed"))
    print(agg([r for r in recs if r["strong"]], "strongly aligned"))

    print(f"\n{'='*100}")
    print(f"  3. WITHIN DIRECTION (removes the CALL/PUT confound)")
    print(f"{'='*100}")
    for dirn in ("CALL", "PUT"):
        sub = [r for r in recs if r["dir"] == dirn]
        if not sub:
            continue
        print(f"\n  --- {dirn} rules ---")
        print(f"  {'bin':28} {'n':>6} {'days':>6} {'loss50':>8} {'win':>7} "
              f"{'med ROE':>9} {'OOS tot':>10}")
        print(agg([r for r in sub if r["aligned"]], "aligned"))
        print(agg([r for r in sub if not r["aligned"]], "opposed"))
        print(agg([r for r in sub if r["strong"]], "strongly aligned"))

    print(f"\n{'='*100}")
    print(f"  4. WITHIN RULE -- the version to believe")
    print(f"{'='*100}")
    print(f"  {'rule':24} {'aligned n':>10} {'loss50':>8} {'opposed n':>10} "
          f"{'loss50':>8} {'delta':>8}")
    deltas = []
    for nm, g in R.groupby("rule"):
        aa = g[g["aligned"]]
        oo = g[~g["aligned"]]
        if len(aa) < 15 or len(oo) < 15:
            print(f"  {nm:24} {len(aa):>10} {'--':>8} {len(oo):>10} "
                  f"{'--':>8}  too few")
            continue
        la = (aa["pnl"] <= -50).mean() * 100
        lo = (oo["pnl"] <= -50).mean() * 100
        deltas.append(la - lo)
        print(f"  {nm:24} {len(aa):>10} {la:>7.1f}% {len(oo):>10} "
              f"{lo:>7.1f}% {la-lo:>+7.1f}")
    if deltas:
        print(f"\n  rules where aligned has the LOWER loss50: "
              f"{sum(1 for x in deltas if x < 0)}/{len(deltas)}")
        print(f"  mean within-rule delta {np.mean(deltas):+.1f}pp "
              f"(negative = alignment helps)")

    print(f"\n  HOW TO READ IT")
    print(f"  Block 1 decides whether blocks 2-4 mean anything. If a rule is")
    print(f"  ~always aligned or ~always opposed, its trades cannot speak to the")
    print(f"  question and the pooled split is a rule comparison in disguise.")
    print(f"  Block 4 is the unconfounded test: does alignment separate loss50")
    print(f"  INSIDE a rule, and does it do so for most rules or just one?")


if __name__ == "__main__":
    main()
