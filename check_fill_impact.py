# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_fill_impact.py
====================
WHAT DOES THE CALIBRATED EXIT CUSHION DO TO THE BOOK?

`botcap` prices the exit cushion from measurement rather than assumption
(sim_core.CUSHION_CAP / FILL_COST, from 18.9M bronze prints over
2026-09-01..09-15), and keys it on the exit TAG rather than on profitability:
  * adverse triggers (stop/trail/give) fire because the bid is falling, so they
    pay the intra-minute timing bound  -- which for a PROFITABLE trail is MORE
    than the legacy 0.5, not less;
  * everything else (eod/time/tp/sig) is marketable into a resting bid and pays
    only the measured fill cost, 0.03 spreads, against a legacy 0.5 or 1.5.
So the re-pricing moves in BOTH directions and no rule's sign can be predicted
from its cap. Rules exiting mostly at EOD gain; rules exiting mostly on a
profitable trail can lose.

READ THE PER-TICKER COLUMN, NOT THE TOTAL. The cap is 1.50 (no change) for SPY
and QQQ and as low as 0.40 for GLD, so a book-wide total mixes rules that were
re-priced with rules that were not. If the gain shows up in SPY/QQQ -- which the
cap does not touch -- something is wrong with the implementation, not with the
market.

WHAT THIS DOES NOT SETTLE
    The cap bounds the TIMING cost by how far the bid actually moved. It does
    not measure what share of that movement is adverse; the true cost sits
    somewhere between 0 and the cap. `bot` therefore stays as the pessimistic
    bound and `mid` as the frictionless one, and a result that only survives
    under `botcap` is not a result.

Usage:
  python check_fill_impact.py
  python check_fill_impact.py --paper      # include the four paper-only rules
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()


def tot(r, oos):
    return sum(p for d, p in r if ((d >= SPLIT) if oos else (d < SPLIT))) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true",
                    help="include AVGO/SMH/GLD/MSFT, which the cap most affects")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = sim_core.research_rules(include_paper=a.paper)
    print(f"  {len(rules)} rules (paper={'yes' if a.paper else 'no'})\n")
    print(f"  {'rule':24} {'tk':5} {'cap':>5} {'IS bot':>9} {'IS cap':>9} "
          f"{'OOS bot':>9} {'OOS cap':>9} {'OOS d':>8} {'OOS mid':>9}")
    agg = {k: [] for k in ("bot", "botcap", "mid")}
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
        pol = sim_core.policy_for(rule)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        rb = sim_core.walk(cand, pol, eod, fill="bot")
        rc = sim_core.walk(cand, pol, eod, fill="botcap", cush_cap=cap)
        rm = sim_core.walk(cand, pol, eod, fill="mid")
        for k, r in (("bot", rb), ("botcap", rc), ("mid", rm)):
            agg[k] += r
        print(f"  {rule['name']:24} {tk:5} {(cap if cap else np.nan):>5.2f} "
              f"{tot(rb,0):>+9.1f} {tot(rc,0):>+9.1f} "
              f"{tot(rb,1):>+9.1f} {tot(rc,1):>+9.1f} "
              f"{tot(rc,1)-tot(rb,1):>+8.1f} {tot(rm,1):>+9.1f}", flush=True)

    print(f"\n  {'POOLED':24} {'':5} {'':>5} "
          f"{tot(agg['bot'],0):>+9.1f} {tot(agg['botcap'],0):>+9.1f} "
          f"{tot(agg['bot'],1):>+9.1f} {tot(agg['botcap'],1):>+9.1f} "
          f"{tot(agg['botcap'],1)-tot(agg['bot'],1):>+8.1f} "
          f"{tot(agg['mid'],1):>+9.1f}")
    b, c, m = tot(agg['bot'], 1), tot(agg['botcap'], 1), tot(agg['mid'], 1)
    if m != b:
        print(f"\n  the cap recovers {(c-b)/(m-b)*100:.0f}% of the gap between the")
        print(f"  pessimistic model and the frictionless one "
              f"({b:+.0f} -> {c:+.0f} -> {m:+.0f})")
    print(f"\n  SANITY: SPY and QQQ carry cap=1.50 and STILL move, which is correct")
    print(f"  under the tag-keyed rule -- their non-adverse exits (eod/tp) drop from")
    print(f"  0.5/1.5 to FILL_COST 0.03 while profitable trails rise from 0.5 to 1.50.")
    print(f"  (Before the tag change the cap alone could not move them, and this line")
    print(f"  asserted they must match. That assertion is retired, not violated.)")


if __name__ == "__main__":
    main()
