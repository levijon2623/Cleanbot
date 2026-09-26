# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_entry_fill_wf.py
======================
RE-RUN THE BOOK TAKING ONLY THE ENTRIES THAT WOULD ACTUALLY HAVE FILLED.

THE LAST UNMEASURED LEG OF THE FILL MODEL
    The exit side was calibrated from NBBO depth (sim_core.CUSHION_CAP) and came
    out far cheaper than assumed. The entry side is the opposite: every backtest
    fills `min(mid + 0.01, ask)` unconditionally, but on a wide market that is a
    RESTING buy below the offer, and it only fills if the offer comes down to it.
    check_entry_fill measured 57% at the live 10s timeout against an assumed
    100%, on 30 paper trades. This runs the same question over every candidate.

THE SPLIT THAT DECIDES WHO IS AFFECTED
        entry_limit = min(round(mid + 0.01, 2), round(ask, 2))
    On a $0.01-wide market mid+0.01 EXCEEDS the ask, so min() clamps the limit TO
    the ask: the order is MARKETABLE and fills at once. The sitting-duck case
    only arises when the spread is wide enough that mid+0.01 lands below the ask.
    So the first table is the marketable share per ticker, and it is expected to
    exonerate the tight-spread book (SPY/QQQ 1.15-1.21%, IWM 2.15%, NVDA 1.48%)
    and hit the wide one (AVGO 5.39%, SMH 12.86%).

MODELLING CHOICES, each of which could be argued
  * A non-marketable entry is taken only if the per-minute ask reaches the limit
    within `--window` minutes; otherwise the candidate is DROPPED.
  * A dropped entry leaves us FLAT, so `walk`'s sequential guard makes the next
    trigger eligible -- which is what really happens when an order does not fill.
  * The fill is priced at the limit, and timed at the ORIGINAL minute. Modelling
    the later fill minute would shift the whole path and is a bigger change than
    this first pass should make. It therefore FLATTERS the filled set slightly.

🚨 GRANULARITY, and it cuts both ways
   The ask series is per-MINUTE (silver `ask_close`), so an offer that dipped to
   our limit mid-minute and recovered is invisible -- understating fills. But we
   also ignore queue position, which overstates them. The paper-trade test at
   true 10-second resolution gave 57%; treat any minute-level number here as the
   optimistic end of a range whose realistic value is lower.

Usage:
  python check_entry_fill_wf.py
  python check_entry_fill_wf.py --window 5 --paper
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()


def classify(cand, window):
    """-> (keep_mask, marketable_mask, wait_minutes) per candidate."""
    keep, mkt, waits = [], [], []
    for d, m, path in cand:
        e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
        if e_mid <= 0 or e_ask <= 0:
            keep.append(False); mkt.append(False); waits.append(np.nan)
            continue
        lim = min(round(e_mid + 0.01, 2), round(e_ask, 2))
        if lim >= round(e_ask, 2) - 1e-9:
            # the limit IS the ask -- marketable, fills immediately
            keep.append(True); mkt.append(True); waits.append(0.0)
            continue
        a = np.asarray(ask, float)
        mm = np.asarray(mods, int)
        j0 = int(np.searchsorted(mm, m))
        j1 = int(np.searchsorted(mm, m + window, side="right"))
        seg = a[j0:j1]
        hit = np.where(np.isfinite(seg) & (seg <= lim + 1e-9))[0]
        if hit.size:
            keep.append(True); mkt.append(False)
            waits.append(float(mm[j0 + hit[0]] - m))
        else:
            keep.append(False); mkt.append(False); waits.append(np.nan)
    return np.array(keep), np.array(mkt), np.array(waits, float)


def tot(r, oos):
    return sum(p for d, p in r if ((d >= SPLIT) if oos else (d < SPLIT))) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=5,
                    help="minutes the resting order is allowed to wait")
    ap.add_argument("--fill", default="bot")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = sim_core.research_rules(include_paper=a.paper)
    print(f"  {len(rules)} rules, wait window {a.window}m, fill={a.fill}\n")
    print(f"  {'rule':24} {'tk':6} {'cands':>7} {'marketable':>11} "
          f"{'rest-fill':>10} {'DROPPED':>8} {'med wait':>9}")
    per, allrows = {}, []
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
        keep, mkt, waits = classify(cand, a.window)
        n = len(cand)
        rest = keep & ~mkt
        w = waits[rest]
        print(f"  {rule['name']:24} {tk:6} {n:>7} {mkt.mean()*100:>10.0f}% "
              f"{rest.sum()/max(n-mkt.sum(),1)*100:>9.0f}% "
              f"{(~keep).mean()*100:>7.0f}% "
              f"{(np.median(w) if w.size else np.nan):>8.1f}m", flush=True)
        pol = sim_core.policy_for(rule)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        full = sim_core.walk(cand, pol, eod, fill=a.fill, cush_cap=cap)
        filt = sim_core.walk([c for c, k in zip(cand, keep) if k], pol, eod,
                             fill=a.fill, cush_cap=cap)
        per[rule["name"]] = (full, filt, mkt.mean(), (~keep).mean())
        allrows.append((rule["name"], tk, n, mkt.mean(), (~keep).mean()))

    print(f"\n{'='*96}")
    print(f"  BOOK WITH AND WITHOUT THE UNFILLABLE ENTRIES")
    print(f"{'='*96}")
    print(f"  {'rule':24} {'IS all':>9} {'IS fill':>9} {'OOS all':>9} "
          f"{'OOS fill':>9} {'OOS d':>8} {'trades':>13}")
    A = B = []
    ta = tb = []
    for name, (full, filt, mk, dr) in per.items():
        print(f"  {name:24} {tot(full,0):>+9.1f} {tot(filt,0):>+9.1f} "
              f"{tot(full,1):>+9.1f} {tot(filt,1):>+9.1f} "
              f"{tot(filt,1)-tot(full,1):>+8.1f} "
              f"{len(full):>6} ->{len(filt):>5}")
        ta = ta + full
        tb = tb + filt
    print(f"\n  {'POOLED':24} {tot(ta,0):>+9.1f} {tot(tb,0):>+9.1f} "
          f"{tot(ta,1):>+9.1f} {tot(tb,1):>+9.1f} "
          f"{tot(tb,1)-tot(ta,1):>+8.1f} {len(ta):>6} ->{len(tb):>5}")

    print(f"\n  HOW TO READ IT")
    print(f"  A high 'marketable' share means the ticker is largely immune: the")
    print(f"  limit clamps to the ask and fills at once. It does NOT mean the two")
    print(f"  book columns match -- dropping even one candidate frees the")
    print(f"  sequential slot, so the guard takes DIFFERENT downstream trades.")
    print(f"  SPY keeps all 15 trades and still moves. Read directions, not sizes.")
    print(f"  For the rest, 'DROPPED' is the share of the backtest's trades the")
    print(f"  live bot would never have got. If the OOS delta is NEGATIVE, the")
    print(f"  dropped entries were winners -- adverse selection, the mechanism")
    print(f"  check_entry_fill found on the paper trades.")


if __name__ == "__main__":
    main()
