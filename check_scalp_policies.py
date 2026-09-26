# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_scalp_policies.py
=======================
SCALP BRACKETS, AND A CORE POSITION WITH A SCALPING SLEEVE.

TWO IDEAS, TESTED TOGETHER BECAUSE THEY SHARE THE SAME MACHINERY
  1. SCALP BRACKETS. The volume-exit run produced 2.7x the trades -- which looks
     a lot like scalping. So ask it directly: what does a tight bracket do?
        [TP +20%, SL -10%]   and   [TP +60%, SL -20%]
     `walk`'s sequential guard sets `busy` to the EXIT minute, so a bracket that
     closes early frees the slot and the next trigger is taken automatically.
     Re-entry is not an extra feature here; it is what the guard already does.
  2. CORE + SCALP SLEEVE. Buy 5, hold 3 to the deployed exit, cycle 2 on the
     scalp leg, reload at the next trigger. The core is NOT blocked by the
     sleeve's re-entries, which is the whole point: exposure to a sustained run
     is never fully given up.

WHY THE SLEEVE MIGHT SUCCEED WHERE THE FULL-POSITION VERSION FAILED
    Every previous exit family permanently cut the day's exposure, and the one
    intervention that ever helped (+5.2pp, leg-in vertical) was the one that did
    not. A full-position signal exit turned out to be a cut after all: it
    interrupts the sustained run that carries the book. A sleeve keeps 60% of
    that run untouched by construction, so it can only ever give up 40% of the
    upside -- while capturing the scalp on the other 40%.

ACCOUNTING, stated because it is the easy thing to get wrong
    Every arm is expressed as RETURN ON ONE UNIT of the original position, so
    all arms are comparable to the baseline's 1.0 unit:
        core+scalp total = 0.6 x sum(core trade returns)
                         + 0.4 x sum(sleeve trade returns)
    The sleeve doing more trades does NOT inflate its weight; it means those
    trades are each on 40% of a unit. This is per-unit-of-premium accounting,
    not a claim about margin or buying power.

🚨 POLICY MAPPING -- the bug this script was written around
    `simulate` reads pol['tp'] / ['stop'] / ['trail']. The RULE dict carries
    `trail_pct` / `target_roe` / `rr` and NONE of those keys, so passing a rule
    straight in silently yields tp=stop=trail=None -- hold-to-EOD, with the
    deployed brackets never applied. `sim_core.policy_for(rule)` is the mapping.
    check_volume_exit.py shipped with exactly this bug on 2026-09-15 and
    reported an EOD-vs-EOD comparison as "vs the deployed policy".

Usage:
  python check_scalp_policies.py
  python check_scalp_policies.py --core 0.6 --fill mid
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core
from check_volume_exit import und_bars, build_sig

SPLIT = pd.Timestamp("2025-08-21").date()


def agg(res, weight=1.0):
    if not res:
        return dict(n=0, days=0, tot=0.0, mean=np.nan)
    df = pd.DataFrame(res, columns=["date", "pnl"])
    return dict(n=len(df), days=df["date"].nunique(),
                tot=float(df["pnl"].sum() * 100 * weight),
                mean=float(df["pnl"].mean() * 100))


def split(res, weight=1.0):
    return (agg([r for r in res if r[0] < SPLIT], weight),
            agg([r for r in res if r[0] >= SPLIT], weight))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default="bot")
    ap.add_argument("--core", type=float, default=0.6, help="core share (3 of 5)")
    ap.add_argument("--thresh", type=float, default=0.75)
    ap.add_argument("--min-prior", type=int, default=3)
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = sim_core.research_rules()
    SCALP = [("scalp 20/10", {"tp": 0.20, "stop": 0.10}),
             ("scalp 60/20", {"tp": 0.60, "stop": 0.20})]

    acc = {}                      # arm -> list of (date, weighted pnl)
    per_rule = {}
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
        pol = sim_core.policy_for(rule)
        up = rule["direction"] == "CALL"
        sigs = build_sig(cand, und_bars(tk), up, a.thresh, a.min_prior)

        arms = {}
        arms["BASELINE (deployed)"] = (sim_core.walk(cand, pol, eod_m, fill=a.fill), 1.0)
        for lbl, sp in SCALP:
            arms[lbl] = (sim_core.walk(cand, sp, eod_m, fill=a.fill), 1.0)

        # ---- core + sleeve. cap=1 makes the core the FIRST trigger of the day,
        # held on the deployed exit; the sleeve cycles independently.
        core = sim_core.walk(cand, pol, eod_m, fill=a.fill, cap=1)
        sleeve_sig = sim_core.walk(cand, pol, eod_m, fill=a.fill, sigs=sigs)
        arms[f"core{int(a.core*100)} + sleeve:signal"] = (core, a.core)
        arms[f"core{int(a.core*100)} + sleeve:signal_B"] = (sleeve_sig, 1 - a.core)
        for lbl, sp in SCALP:
            arms[f"core{int(a.core*100)} + sleeve:{lbl}"] = (core, a.core)
            arms[f"core{int(a.core*100)} + sleeve:{lbl}_B"] = (
                sim_core.walk(cand, sp, eod_m, fill=a.fill), 1 - a.core)

        for k, (res, w) in arms.items():
            acc.setdefault(k, []).extend([(d, p * w) for d, p in res])
            per_rule.setdefault(k, {}).setdefault(rule["name"], []).extend(
                [(d, p * w) for d, p in res])
        print(f"  {rule['name']} done", flush=True)

    # fold the paired *_B legs into their partner
    merged, counts = {}, {}
    for k, v in acc.items():
        base_k = k[:-2] if k.endswith("_B") else k
        merged.setdefault(base_k, []).extend(v)
        counts.setdefault(base_k, 0)
        counts[base_k] += len(v)

    print(f"\n{'='*96}")
    print(f"  SCALP BRACKETS AND CORE+SLEEVE   (fill={a.fill}, core={a.core:.0%})")
    print(f"  Totals are RETURN ON ONE UNIT of the original position.")
    print(f"{'='*96}")
    print(f"  {'arm':34} {'IS trades':>10} {'IS tot%':>10} {'OOS trades':>11} "
          f"{'OOS tot%':>10} {'OOS/trade':>10}")
    base_oos = None
    order = ["BASELINE (deployed)", "scalp 20/10", "scalp 60/20"]
    order += [k for k in sorted(merged) if k not in order]
    for k in order:
        if k not in merged:
            continue
        i, o = split(merged[k])
        if k == "BASELINE (deployed)":
            base_oos = o["tot"]
        d = f"{o['tot'] - base_oos:+.1f}" if base_oos is not None else ""
        print(f"  {k:34} {i['n']:>10} {i['tot']:>+10.1f} {o['n']:>11} "
              f"{o['tot']:>+10.1f} {o['mean']:>+10.2f}   {d}")

    print(f"\n  PER-RULE OOS TOTALS")
    names = sorted({n for v in per_rule.values() for n in v})
    print(f"  {'arm':34} " + " ".join(f"{n.split()[0]:>9}" for n in names))
    for k in order:
        if k.endswith("_B") or k not in per_rule:
            continue
        cells = []
        for n in names:
            rows = per_rule[k].get(n, [])
            partner = per_rule.get(k + "_B", {}).get(n, [])
            _, o = split(rows + partner)
            cells.append(f"{o['tot']:>+9.1f}")
        print(f"  {k:34} " + " ".join(cells))

    print(f"\n  HOW TO READ IT")
    print(f"  A scalp bracket with no trail CANNOT catch a sustained run: +20% is")
    print(f"  the ceiling on every winner it takes. The question is whether many")
    print(f"  capped winners beat few uncapped ones in THIS book, where the totals")
    print(f"  are carried by a handful of large moves.")
    print(f"  For core+sleeve, compare against BASELINE: the core alone is 60% of")
    print(f"  the baseline's exposure, so the sleeve has to earn back the 40% it")
    print(f"  gave up before anything above zero is a real gain.")


if __name__ == "__main__":
    main()
