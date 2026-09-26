# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_rule_ablation.py
======================
Take a deployed rule apart one component at a time and attribute its return to
each: the flow threshold, the exit policy, and each gate separately.

WHY THIS EXISTS -- and a correction it was written to make
    `check_gate_lift.py` reported NVDA LOWVOL PUT at -7.3pp "negative lift",
    i.e. its conditioners appearing to make it WORSE than unconditioned. That
    number conflates three different things, because it compared the deployed
    rule against the BEST unconditioned cell for that ticker/direction:

      * threshold -- NVDA deploys min_flow_pct 80, but its best base cell was
        p50. The p80 base is -11.2%, not -6.3%.
      * exit      -- NVDA sets `trail_pct: 0`, so it runs the STATIC bracket
        (tp=target_roe, stop=target_roe/rr), while every grid cell ran the
        default 50% trail. Different instrument entirely.
      * gates     -- hours and regime, the only part "lift" was meant to measure.

    Against the matched p80 cell the gap is -2.4pp, not -7.3pp. `lift` remains
    useful as a rough fragility signal but it is NOT a clean gate measurement
    for any rule that deploys a non-default threshold or exit. This script is.

METHOD
    A ladder from the bare trigger to the deployed rule, adding ONE component
    per rung, then a leave-one-out pass that removes each gate from the full
    rule. Both directions are reported because they disagree whenever gates
    interact, and that disagreement is itself informative.

    Everything is scored through sim_core on the deployed basis: sequential
    fills, fill="bot", $0.50 floor, per-window means so decay stays visible.

THIS IS DIAGNOSIS, NOT SEARCH
    There is no holdout left -- the pre-sample was spent on 2026-09-12 and the
    OOS window has been examined for weeks. So a variant that scores better
    here CANNOT be promoted on that basis; it would be selection on the
    outcome. The output says which component is responsible for a rule's
    behaviour, not which variant to deploy.

Usage:
  python check_rule_ablation.py --rule "NVDA LOWVOL PUT"
  python check_rule_ablation.py --rule "META LOWVOL PUT" --pcts 50 65 80 90 95
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd

from check_config_walkforward import SLICE_EDGES
import sim_core

PRE_LO = pd.Timestamp("2023-10-12").date()
DEPLOY_LO = SLICE_EDGES[0]
SPLIT = pd.Timestamp("2025-08-21").date()
#: rule keys that act as GATES (as opposed to instrument/exit choices)
GATE_KEYS = ("hours", "regime", "amt_open", "ema_confirm", "dmi_confirm",
             "vol_overlay", "flow_zscore", "dex_pct_max", "skip_macro_am",
             "eod_flatten")


def score(rows):
    if not rows:
        return None
    d = np.array([x[0] for x in rows])
    p = np.array([x[1] for x in rows], float)
    w = {"n": len(p), "all": p.mean(), "win": float((p > 0).mean())}
    for lbl, lo, hi in (("pre", PRE_LO, DEPLOY_LO), ("is", DEPLOY_LO, SPLIT),
                        ("oos", SPLIT, SLICE_EDGES[-1])):
        m = (d >= lo) & (d < hi)
        w[lbl] = p[m].mean() if m.sum() else np.nan
    return w


def run_variant(D, rule, trigs, since=None):
    from config import TRAIL_PCT
    cand = sim_core.build_candidates(D, rule, trigs=trigs, since=since)
    if not cand:
        return None
    return score(sim_core.walk(cand, sim_core.policy_for(rule, TRAIL_PCT),
                               sim_core.eod_mod(rule), fill="bot"))


def line(lbl, s, width=46):
    if not s:
        print(f"  {lbl:{width}} (no trades)")
        return
    print(f"  {lbl:{width}} n={s['n']:>4}  all {s['all']*100:>+7.1f}%  "
          f"PRE {s['pre']*100:>+7.1f}%  IS {s['is']*100:>+7.1f}%  "
          f"OOS {s['oos']*100:>+7.1f}%  win {s['win']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True)
    ap.add_argument("--pcts", nargs="*", type=int, default=[50, 65, 80, 90, 95])
    a = ap.parse_args()

    import directional_flow_backtester as D
    from config import RULES
    dep = next((r for r in RULES if r["name"] == a.rule), None)
    if dep is None:
        raise SystemExit(f"no rule named {a.rule!r}")
    tk = dep["ticker"]

    flow = __import__("check_config_walkforward").__dict__["_flow_for"](D, [tk])
    if flow.empty:
        raise SystemExit(f"no flow series for {tk}")
    trigs = D.triggers_for(flow, tk)
    D.annotate_flow_pct(trigs, dep.get("flow_window_days", 60))

    gates = {k: dep[k] for k in GATE_KEYS if k in dep and dep[k] not in (None, [])}
    print(f"RULE ABLATION -- {a.rule}")
    print(f"  deployed: min_flow_pct={dep.get('min_flow_pct')}  "
          f"trail_pct={dep.get('trail_pct')}  target_roe={dep.get('target_roe')}  "
          f"rr={dep.get('rr')}")
    print(f"  gates: {gates}\n")

    bare = {"name": "bare", "ticker": tk, "direction": dep["direction"],
            "dte": dep.get("dte", [0, 1]), "min_flow_pct": dep.get("min_flow_pct"),
            "target_roe": dep.get("target_roe", 1.0), "rr": dep.get("rr", 1.0)}

    # ---- 1. threshold sweep on the BARE trigger, default trail exit
    print(f"{'='*118}\n  1. BARE TRIGGER by flow threshold (default {sim_core.__name__} "
          f"trail exit, no gates)\n{'='*118}")
    for p in a.pcts:
        r = dict(bare, min_flow_pct=p)
        tag = "   <- deployed threshold" if p == dep.get("min_flow_pct") else ""
        line(f"p{p}" + tag, run_variant(D, r, trigs))

    # ---- 2. the ladder: add one component at a time
    print(f"\n{'='*118}\n  2. LADDER -- one component added per rung\n{'='*118}")
    cur = dict(bare)
    prev = run_variant(D, cur, trigs)
    line(f"bare @ p{dep.get('min_flow_pct')} + trail exit", prev)
    steps = []
    if dep.get("trail_pct") is not None:
        steps.append(("exit -> trail_pct=%s" % dep["trail_pct"],
                      {"trail_pct": dep["trail_pct"]}))
    for k, v in gates.items():
        steps.append((f"+ gate {k}={v}", {k: v}))
    for lbl, patch in steps:
        cur = dict(cur, **patch)
        s = run_variant(D, cur, trigs)
        delta = (s["all"] - prev["all"]) * 100 if (s and prev) else float("nan")
        line(f"{lbl}", s)
        print(f"  {'':46} ^ delta {delta:+.1f}pp")
        prev = s
    print(f"\n  (final rung should equal the deployed rule)")
    line("DEPLOYED (config, verbatim)", run_variant(D, dep, trigs))

    # ---- 3. leave-one-out from the full rule
    print(f"\n{'='*118}\n  3. LEAVE-ONE-OUT -- remove ONE component from the "
          f"deployed rule\n{'='*118}")
    full = run_variant(D, dep, trigs)
    line("deployed (reference)", full)
    print()
    for k in list(gates) + ["trail_pct"]:
        r = copy.deepcopy(dep)
        if k == "trail_pct":
            r["trail_pct"] = None                 # fall through to config default
            lbl = "without static bracket (use default trail)"
        else:
            r.pop(k, None)
            lbl = f"without gate {k}"
        s = run_variant(D, r, trigs)
        d = (s["all"] - full["all"]) * 100 if (s and full) else float("nan")
        line(lbl, s)
        print(f"  {'':46} ^ removing it changes all-period by {d:+.1f}pp"
              + ("   <- component is COSTING the rule" if d > 0 else ""))

    print(f"\n  Read column-wise, not just the all-period mean: a component that")
    print(f"  helps IS and hurts PRE/OOS is the signature of a gate fitted to the")
    print(f"  window it was chosen in. No variant here is deployable evidence --")
    print(f"  there is no holdout left to confirm one.")


if __name__ == "__main__":
    main()
