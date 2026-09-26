# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_gate_lift.py
==================
HOW MUCH OF EACH RULE'S RETURN IS SUPPLIED BY ITS GATES?

    lift = deployed return  -  best unconditioned base for that ticker/direction

The book was built in a specific order, and that order has a statistical
consequence. First the least-bad flow thresholds were chosen per ticker and
direction; then regime / hour / AMT / ema / dmi conditioners were developed
until those candidates turned net-positive. `check_step1_redo` has since shown
the starting set was UNIFORMLY negative -- only 2 of 150 unconditioned cells
have a positive mean. So the gates are not refining an edge that was already
there; they are supplying the entire edge.

That makes `lift` a fragility metric rather than a quality metric. A rule whose
gates must manufacture +60pp is absorbing far more selection pressure than one
that needs +10pp, and it has correspondingly more room to be an artifact of the
window the gates were chosen in. This is not a claim that high lift is WRONG --
it is a claim about how much evidence would be required to believe it.

The base is the BEST of that ticker/direction's five threshold cells, which is
generous to the rule: it assumes the threshold was chosen perfectly. Lift
measured against the mean cell would be larger still.

Reads `_step1_grid.parquet` (check_step1_redo) and scores the deployed rules
through sim_core on the same basis. Nothing here is a hypothesis test; it is a
descriptive decomposition of numbers already computed.

Usage:  python check_gate_lift.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sim_core


def deployed_returns():
    """{rule name: (n, all-period mean)} on the deployed basis."""
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    out = {}
    for r in [x for x in RULES if x.get("enabled", True)]:
        cand = sim_core.build_candidates(D, r)
        if not cand:
            continue
        rows = sim_core.walk(cand, sim_core.policy_for(r, TRAIL_PCT),
                             sim_core.eod_mod(r), fill="bot")
        if rows:
            p = np.array([x[1] for x in rows], float)
            out[r["name"]] = (r["ticker"], r["direction"], len(p), p.mean())
    return out


def main():
    grid = pd.read_parquet("_step1_grid.parquet")
    best = (grid.sort_values("all", ascending=False)
                .groupby(["ticker", "dir"], as_index=False).first())
    bmap = {(r.ticker, r.dir): (r.pct, r.all, r.n) for r in best.itertuples()}

    dep = deployed_returns()
    rows = []
    for name, (tk, d, n, m) in dep.items():
        b = bmap.get((tk, d))
        if b is None:
            continue
        rows.append(dict(rule=name, ticker=tk, dir=d, n_dep=n, dep=m,
                         base_pct=b[0], base=b[1], n_base=b[2], lift=m - b[1]))
    df = pd.DataFrame(rows).sort_values("lift", ascending=False)

    print("  GATE LIFT -- how much of each deployed rule's return comes from its")
    print("  conditioners rather than from the trigger it is built on.\n")
    print(f"  {'rule':22} {'n':>4} {'deployed':>9} | {'best base':>10} {'n':>5} "
          f"{'thr':>4} | {'LIFT':>8}")
    print("  " + "-" * 78)
    for r in df.itertuples():
        print(f"  {r.rule:22} {r.n_dep:>4} {r.dep*100:>+8.1f}% | "
              f"{r.base*100:>+9.1f}% {r.n_base:>5} {'p'+str(r.base_pct):>4} | "
              f"{r.lift*100:>+7.1f}pp")

    print(f"\n  median lift {df['lift'].median()*100:+.1f}pp   "
          f"max {df['lift'].max()*100:+.1f}pp ({df.iloc[0]['rule']})")
    print(f"  rules whose base is already positive: "
          f"{int((df['base'] > 0).sum())} of {len(df)}")
    print(f"  rules whose ENTIRE return is lift (base <= 0): "
          f"{int((df['base'] <= 0).sum())} of {len(df)}")

    print("\n  HOW TO READ THIS")
    print("  Lift is not a defect on its own -- conditioning on regime is a real")
    print("  thing to do. It is a measure of how much the rule is ASSERTING, and")
    print("  therefore how much evidence it needs. A +60pp lift built on n=13")
    print("  deployed trades is asserting a great deal on very little.")
    print("  Cross-check against the pre-sample: the holdout failed hardest on")
    print("  exactly the SPY+QQQ cut (-39.7%, 1 winner in 14), and those carry")
    print("  two of the three largest lifts here.")


if __name__ == "__main__":
    main()
