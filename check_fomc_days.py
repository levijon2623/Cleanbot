# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_fomc_days.py
==================
HOW BIG IS THE "TINY SAMPLE" THAT LETS THE BOOK TRADE FOMC DAYS?

THE DECISION THIS RE-EXAMINES
    `skip_macro_am` suppresses IWM/QQQ HIVOL CALL on CPI/PCE/NFP mornings.
    FOMC is deliberately NOT gated -- macro_calendar records the reason:
    "FOMC's 2pm IV crush is real but the bot navigates it fine (net positive,
    tiny sample)". `is_fomc_day` exists but no rule calls it; the only consumer
    is ml_feature_scan, as a feature.
    That decision therefore rests entirely on how tiny "tiny" is, which the note
    does not say. This says it.

WHY IT IS WORTH RE-ASKING NOW
    Two things changed since session 16 (2026-09-06): the calendar extended, and
    the fill model was recalibrated against real NBBO depth. Both move the
    numbers the original call was made on.

WHAT TO EXPECT, STATED FIRST SO THE RESULT CANNOT BE READ INTO
    IWM HIVOL CALL trades ~64 days in the whole history. There are 16 FOMC days
    in the deployed window. The intersection is small by construction, so this
    will almost certainly be UNDERPOWERED -- the useful output is the sample
    size and the direction, not a verdict. A rule that has seen two FOMC days
    has no business claiming it "navigates them fine" in either direction.

Usage:
  python check_fomc_days.py
  python check_fomc_days.py --fill botcap --paper
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core
from macro_calendar import FOMC, is_macro_am_day

SPLIT = pd.Timestamp("2025-08-21").date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    fomc = {pd.Timestamp(d).date() for d in FOMC}
    print(f"  {len(fomc)} FOMC dates on the calendar, fill={a.fill}\n")
    print(f"  {'rule':24} {'FOMC n':>7} {'FOMC days':>10} {'FOMC tot%':>10} "
          f"{'other n':>8} {'other %/trade':>14} {'FOMC %/trade':>13}")

    tot_f, tot_o = [], []
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk = rule["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        cand = sim_core.build_candidates(D, rule, trigs=trigs)
        if not cand:
            continue
        pol = sim_core.policy_for(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        res = sim_core.walk(cand, pol, sim_core.eod_mod(rule), fill=a.fill,
                            cush_cap=cap)
        f = [(d, p) for d, p in res if d in fomc]
        o = [(d, p) for d, p in res if d not in fomc]
        tot_f += f
        tot_o += o
        fm = np.mean([p for _, p in f]) * 100 if f else np.nan
        om = np.mean([p for _, p in o]) * 100 if o else np.nan
        print(f"  {rule['name']:24} {len(f):>7} {len({d for d,_ in f}):>10} "
              f"{sum(p for _,p in f)*100:>+10.1f} {len(o):>8} "
              f"{om:>+14.2f} {fm:>+13.2f}")

    print(f"\n  {'POOLED':24} {len(tot_f):>7} {len({d for d,_ in tot_f}):>10} "
          f"{sum(p for _,p in tot_f)*100:>+10.1f} {len(tot_o):>8} "
          f"{np.mean([p for _,p in tot_o])*100:>+14.2f} "
          f"{(np.mean([p for _,p in tot_f])*100 if tot_f else np.nan):>+13.2f}")

    # the comparison that actually matters: is FOMC worse than macro-AM, which
    # IS gated? If it is not, the asymmetry in the config needs a reason.
    print(f"\n  FOR CONTEXT -- the days that ARE gated (CPI/PCE/NFP mornings)")
    am = [(d, p) for d, p in tot_o if is_macro_am_day(d)]
    plain = [(d, p) for d, p in tot_o if not is_macro_am_day(d)]
    for lbl, s in (("FOMC (NOT gated)", tot_f), ("macro-AM (gated on 2 rules)", am),
                   ("ordinary days", plain)):
        if not s:
            continue
        v = np.array([p for _, p in s]) * 100
        print(f"    {lbl:28} n={len(v):>4}  days={len({d for d,_ in s}):>3}  "
              f"mean {v.mean():>+7.2f}%  median {np.median(v):>+7.2f}%  "
              f"win {(v>0).mean()*100:>3.0f}%")

    print(f"\n  HOW TO READ IT")
    print(f"  Compare the SAMPLE SIZES first. A rule with a handful of FOMC days")
    print(f"  cannot support 'navigates them fine', and equally cannot support")
    print(f"  gating them -- the honest position on a sample that small is that")
    print(f"  the question is open, not that the current setting is validated.")
    print(f"  METHODOLOGY 7: power is set by DAYS, and the day counts here are")
    print(f"  the whole story.")


if __name__ == "__main__":
    main()
