# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_strike_offset.py
======================
ATM vs 1-STRIKE-OTM ON THE CURRENT BOOK, UNDER THE *CORRECTED* FILL MODEL.

WHY THIS IS NOT A REPEAT OF check_strike_selection
    That study (2026-09-07) already tested ATM / OTM+1 / OTM+2 / moneyness-based
    picks, split by VIX and vol regime. But it predates every fill correction --
    it references no sim_core, no botcap, no CUSHION_CAP, no FILL_COST, and it
    scored TP/SL brackets rather than the deployed trail50. The corrections
    landed 2026-09-16, and all three bear on OTM strikes ASYMMETRICALLY and in
    OPPOSITE directions:

      1  EXIT CUSHION. OTM options carry wider relative spreads, so the retired
         flat 1.5-spreads charge hit them hardest -- on the wide names it was
         2.4-3.8x the bid's measured movement. CUSHION_CAP (0.40-1.50, keyed on
         the exit tag) removes most of that penalty. FAVOURS OTM.
      2  ENTRY FILL. Only 27% of paper entries would actually have filled at the
         live 10s timeout, not the 100% assumed, and a wider spread fills worse
         still. PENALISES OTM.
      3  The $0.50 entry floor rejects more OTM contracts outright, and the
         reject rate is itself a result.

    Net direction is genuinely unknown, which is what makes the re-run worth the
    compute rather than a foregone conclusion.

🚨 THE OFFSET PICKER IS VERIFIED AGAINST THE DEPLOYED ONE AT k=0
    A second strike picker is exactly the drifting copy METHODOLOGY 1 is about.
    So the variant is asserted to return the IDENTICAL contract to
    D.pick_contract at offset 0 before any result is believed; if it does not,
    the run aborts. Everything else -- gates, entry floor, exit policy, fill
    model, cushion cap -- is the deployed path untouched, so the only thing
    varying between arms is the strike.

WHAT TO EXPECT, STATED FIRST
    OTM is cheaper, so the same underlying move is a larger ROE -- a $2 SPY move
    on a 1-strike-OTM 0DTE can clear the +100% a 50% trail needs, where the same
    move on ATM is roughly +50% and dies in the dead zone. Against that: lower
    delta means more trades expire worthless, the spread is wider, and the $0.50
    floor bites. The interesting cell is loss50 RISING while median ROE and the
    right tail improve -- that is the convexity trade, and whether it is worth
    taking depends on the tail, not the win rate (METHODOLOGY 7: win rate is not
    the objective).

Usage:
  python check_strike_offset.py --paper
  python check_strike_offset.py --paper --offsets 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()


def make_picker(D, k):
    """pick_contract, but k strikes further OUT of the money.

    Mirrors the deployed picker's filtering exactly (type, expiry/dte, the
    3-minute quote-staleness window, last quote per contract) and then steps k
    strikes away from spot instead of taking the nearest. At k=0 it must return
    what the deployed picker returns -- asserted below.
    """
    def pick(day_bars, ts, direction, target_dte, spot):
        at = day_bars[(day_bars["minute_et"] <= ts)
                      & (day_bars["minute_et"] >= ts - pd.Timedelta(minutes=3))]
        at = at[at["option_type"] == ("call" if direction == "CALL" else "put")]
        if at.empty:
            return None
        entry_date = pd.Timestamp(ts).date()
        at = at.assign(dte=(at["expiry"] - entry_date).map(lambda x: x.days))
        at = at[at["dte"] == target_dte]
        if at.empty:
            return None
        at = at.sort_values("minute_et").groupby("option_chain_id").last().reset_index()
        at = at.sort_values("strike").reset_index(drop=True)
        i = int((at["strike"] - spot).abs().argmin())
        j = i + k if direction == "CALL" else i - k      # OTM = up for calls
        if j < 0 or j >= len(at):
            return None
        return at.iloc[j]["option_chain_id"]
    return pick


def agg(rows, label):
    if not rows:
        return f"  {label:22} {'(none)':>6}"
    v = np.array([r["pnl"] for r in rows], float)
    i = [r["pnl"] for r in rows if r["date"] < SPLIT]
    o = [r["pnl"] for r in rows if r["date"] >= SPLIT]
    win = v[v > 0]
    return (f"  {label:22} {len(v):>6} {len({r['date'] for r in rows}):>5} "
            f"{(v <= -50).mean()*100:>7.1f}% {(v > 0).mean()*100:>6.1f}% "
            f"{np.median(v):>+8.1f} {(np.mean(win) if len(win) else 0):>+9.1f} "
            f"{np.percentile(v, 95):>+9.1f} "
            f"{(np.sum(i) if i else 0):>+9.0f} {(np.sum(o) if o else 0):>+9.0f}")


HDR = (f"  {'arm':22} {'n':>6} {'days':>5} {'loss50':>8} {'win':>7} "
       f"{'medROE':>8} {'avg win':>9} {'p95':>9} {'IS':>9} {'OOS':>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    import directional_flow_backtester as D
    orig = D.pick_contract

    # ---- verify the variant reproduces the deployed picker at k=0 ----------
    probe = sim_core.research_rules(include_paper=a.paper)[0]
    chk = {"same": 0, "diff": 0}
    D.pick_contract = make_picker(D, 0)
    c0 = sim_core.build_candidates(D, probe)
    D.pick_contract = orig
    c1 = sim_core.build_candidates(D, probe)
    for x, y in zip(c0, c1):
        chk["same" if (x[0], x[1]) == (y[0], y[1]) else "diff"] += 1
    if len(c0) != len(c1) or chk["diff"]:
        D.pick_contract = orig
        raise SystemExit(f"  offset picker does NOT reproduce the deployed one "
                         f"at k=0 ({len(c0)} vs {len(c1)} candidates, "
                         f"{chk['diff']} mismatched) -- aborting rather than "
                         f"reporting a strike effect that is really a picker bug")
    print(f"  picker verified: k=0 reproduces D.pick_contract exactly "
          f"({len(c0)} candidates on {probe['name']})\n")

    out = {}
    try:
        for k in a.offsets:
            D.pick_contract = make_picker(D, k)
            rows, rejected = [], 0
            for rule in sim_core.research_rules(include_paper=a.paper):
                tk = rule["ticker"]
                cand = sim_core.build_candidates(D, rule)
                pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
                cap = sim_core.CUSHION_CAP.get(tk)
                for d, m, path in cand:
                    pnl, xm, tag = sim_core.simulate(path, pol, eod,
                                                     fill=a.fill, cush_cap=cap)
                    rows.append(dict(rule=rule["name"], ticker=tk, date=d,
                                     pnl=pnl * 100, entry=path[0], tag=tag))
            out[k] = rows
            print(f"    offset +{k}: {len(rows)} trades", flush=True)
    finally:
        D.pick_contract = orig          # never leave the module patched

    base = out.get(0, [])
    n0 = len(base)
    print(f"\n{'='*118}")
    print(f"  1. HEADLINE -- same rules, same gates, same exit, only the strike moves")
    print(f"{'='*118}")
    print(HDR)
    for k in a.offsets:
        lbl = "ATM (deployed)" if k == 0 else f"OTM +{k} strike"
        print(agg(out[k], lbl))
    print(f"\n  The $0.50 entry floor rejects more of the cheaper OTM contracts:")
    for k in a.offsets:
        print(f"    offset +{k}: {len(out[k]):>5} candidates survive the floor"
              f"{'' if k == 0 else f'  ({len(out[k])-n0:+d} vs ATM)'}")
        med = np.median([r["entry"] for r in out[k]]) if out[k] else np.nan
        print(f"                median entry premium ${med:.2f}")

    print(f"\n{'='*118}")
    print(f"  2. PER RULE -- median ROE by arm (C5: does it hold across the book?)")
    print(f"{'='*118}")
    names = sorted({r["rule"] for r in base})
    print(f"  {'rule':24}" + "".join(f"{('ATM' if k==0 else f'OTM+{k}'):>12}"
                                     for k in a.offsets) + f"{'best':>10}")
    wins = {k: 0 for k in a.offsets}
    for nm in names:
        meds = {}
        for k in a.offsets:
            v = [r["pnl"] for r in out[k] if r["rule"] == nm]
            meds[k] = np.median(v) if len(v) >= 10 else np.nan
        good = {k: v for k, v in meds.items() if np.isfinite(v)}
        b = max(good, key=good.get) if good else None
        if b is not None:
            wins[b] += 1
        print(f"  {nm:24}" + "".join(f"{meds[k]:>+12.1f}" if np.isfinite(meds[k])
                                     else f"{'--':>12}" for k in a.offsets)
              + f"{('ATM' if b == 0 else f'OTM+{b}') if b is not None else '--':>10}")
    print(f"\n  rules won: " + "  ".join(
        f"{'ATM' if k == 0 else f'OTM+{k}'} {wins[k]}" for k in a.offsets))

    print(f"\n{'='*118}")
    print(f"  3. PAIRED BY TRADE -- the same trigger, both strikes")
    print(f"{'='*118}")
    print(f"  Matching on (rule, date) so day composition and trigger timing are")
    print(f"  identical between arms; only the contract differs.")
    for k in a.offsets:
        if k == 0:
            continue
        A = {(r["rule"], r["date"]): r["pnl"] for r in base}
        B = {(r["rule"], r["date"]): r["pnl"] for r in out[k]}
        keys = sorted(set(A) & set(B))
        if len(keys) < 30:
            print(f"  OTM+{k}: only {len(keys)} matched pairs")
            continue
        d = np.array([B[x] - A[x] for x in keys])
        print(f"  OTM+{k}: {len(keys)} paired trades   mean {d.mean():+.1f}pp   "
              f"median {np.median(d):+.1f}pp   better {100*(d>0).mean():.0f}%   "
              f"total {d.sum():+.0f}")

    print(f"\n  HOW TO READ IT")
    print(f"  Win rate is NOT the objective here -- the book's edge is a right")
    print(f"  tail. loss50 rising while avg-win and p95 rise faster is the")
    print(f"  convexity trade working. Block 3 is the cleanest comparison: the")
    print(f"  same trigger on the same day, differing only in strike.")


if __name__ == "__main__":
    main()
