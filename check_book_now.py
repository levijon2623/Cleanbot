# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_book_now.py
=================
"What is the book worth, exactly as it is configured right now?"

Reads config.RULES as deployed (including the new trailing exit) and scores it
on the fill model the bot actually runs:

  * SEQUENTIAL fills  -- one position per ticker (bot_runner.py:1288). The
    as-screened backtester counts every trigger independently, which inflated
    the book ~10x in trade count and ~2.5x in OOS expectancy.
  * TRAILING exit     -- config.TRAIL_PCT / per-rule trail_pct, peak tracked on
    the bid, no take-profit while trailing.
  * MID and REALISTIC fills -- realistic = enter at ASK, exit at BID, which is
    close to what bot_runner actually does (marketable limit at bid - cushion).

Everything here already nets config COMMISSION_PCT (1.5% of premium round-trip).

Usage:  python check_book_now.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
import sim_core

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _line(lbl, pnls, width=24):
    if not pnls:
        return f"    {lbl:{width}} (none)"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    c = m = 0
    for _, p in sorted(pnls):
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    npop = sum(1 for b in sl if len(b) >= 4)
    slc = " ".join(f"S{j+1}{np.mean(b)*100:+.0f}" if len(b) >= 4 else f"S{j+1}··"
                   for j, b in enumerate(sl))
    return (f"    {lbl:{width}} n={len(v):>4}  {v.mean()*100:>+6.1f}%  "
            f"IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  "
            f"win {(v>0).mean():.2f}  maxLL {m:>2}  tot {v.sum():>+7.2f}  "
            f"pop {npop}/6  [{slc}]")


def run(a):
    import directional_flow_backtester as D
    from config import (RULES, TRAIL_PCT, SIZING_TARGET_PREMIUM_PCT,
                        MAX_PORTFOLIO_RISK_PCT)

    rules = [r for r in RULES if r.get("enabled", True)]
    if getattr(a, "rules", None):
        rules = [r for r in rules if r["name"] in a.rules]
    if getattr(a, "drop", None):
        rules = [r for r in rules if r["name"] not in a.drop]
    per_rule, mid_book, real_book = {}, [], []

    for r in rules:
        # ONE builder, ONE simulator -- see sim_core's header. `rl` now carries
        # the live exit cushion (bid - 1.5x spread on losses, 0.5x on gains),
        # which this script did NOT model before 2026-09-10 and which is worth
        # ~10pp/trade book-wide. `m_` stays mid-fill for the historical contrast.
        cand = sim_core.build_candidates(D, r)
        if not cand:
            continue
        em = sim_core.eod_mod(r)
        pol = sim_core.policy_for(r, TRAIL_PCT)
        tp_ = float(r.get("trail_pct", TRAIL_PCT) or 0.0)
        # `bot` mirrors bot_runner exactly: enter min(mid+0.01, ask), exit
        # bid - spread*(0.5|1.5). NOT ask-in -- the bot never pays the ask, and
        # charging it invents ~half a spread per trade (worth +2.7pp book-wide,
        # far more on wide-quote names). `mid` is kept as the frictionless
        # contrast. See check_fill_sensitivity.py for the full band.
        m_ = sim_core.walk(cand, pol, em, fill="mid")
        rl = sim_core.walk(cand, pol, em, fill="bot")
        per_rule[r["name"]] = (m_, rl, tp_)
        mid_book += m_
        real_book += rl

    print("=" * 122)
    print(f"  THE BOOK AS CONFIGURED NOW   TRAIL_PCT={TRAIL_PCT}   sequential fills   split {SPLIT}")
    print("=" * 122)

    print("\n  -- per rule (REALISTIC fills: enter ask, exit bid) --")
    for rn, (m_, rl, tp_) in sorted(per_rule.items(), key=lambda x: -np.mean(
            [p for d, p in x[1][1] if d >= SPLIT] or [-9])):
        print(_line(f"{rn}", rl))

    print("\n  -- BLENDED BOOK --")
    print(_line("MID fills", mid_book, 24))
    print(_line("REALISTIC fills", real_book, 24))

    v = np.array([p for _, p in real_book])
    o = np.array([p for d, p in real_book if d >= SPLIT])
    days = len({d for d, _ in real_book})
    yrs = (max(d for d, _ in real_book) - min(d for d, _ in real_book)).days / 365.25
    print("\n  -- what that means, roughly --")
    print(f"    {len(v)} trades over {days} trading days / {yrs:.1f} years  "
          f"= {len(v)/max(yrs,1e-9):.0f} trades/yr ({len(v)/max(days,1):.2f} per active day)")
    print(f"    net expectancy per trade (realistic, after {0.015*100:.1f}% round-trip cost):")
    print(f"       all {v.mean()*100:+.2f}%   IS {np.mean([p for d,p in real_book if d<SPLIT])*100:+.2f}%   "
          f"OOS {o.mean()*100:+.2f}%")
    for eq in (25_000, 50_000):
        prem = eq * SIZING_TARGET_PREMIUM_PCT
        print(f"    at ${eq:,} equity and {SIZING_TARGET_PREMIUM_PCT*100:.0f}% premium/trade "
              f"(~${prem:,.0f} at risk per trade):")
        print(f"       OOS ${prem*o.mean():+,.0f}/trade  ->  ${prem*o.mean()*len(v)/max(yrs,1e-9):+,.0f}/yr "
              f"({prem*o.mean()*len(v)/max(yrs,1e-9)/eq*100:+.1f}% of equity/yr, un-compounded)")
    print(f"\n    caveat: OOS n={len(o)} trades. Sizing is flat-premium so the % of equity "
          f"scales with SIZING_TARGET_PREMIUM_PCT, not with the win rate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="*", default=None,
                    help="score ONLY these rules. NOTE: hand-picking rules that already "
                         "tested well is SELECTION ON THE OUTCOME and the resulting book "
                         "number is optimistically biased. See the caveat printed below.")
    ap.add_argument("--drop", nargs="*", default=None, help="exclude these rules")
    a = ap.parse_args()
    run(a)
