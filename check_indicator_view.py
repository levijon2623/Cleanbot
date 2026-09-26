# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_indicator_view.py
=======================
THE TRIGGER AS A 0DTE INDICATOR, not a bot: no mechanical exit, just "here is a
setup" -- and the question becomes whether a human with a simple bracket could
trade it.

WHY FIRST-TOUCH AND NOT MEAN RETURN
    Drop the exit policy and mean return is undefined -- it depends entirely on
    the discretion you are trying to evaluate. What IS well defined is FIRST
    TOUCH: from the entry, does the price reach +target before -stop? That is
    exactly the experience of a trader holding a bracket, it needs no exit
    tuning, and it captures the PATH ORDER that MFE/MAE throw away.

    MFE +80.4% / MAE -60.5% (check_peak_profit) says both extremes get reached.
    It cannot say which comes first, and for an indicator product that is the
    entire question.

WHAT THE PRIOR WORK DID AND DID NOT SETTLE
    check_signal_quality concluded "do NOT build a human-advisory product on
    this trigger" -- MFE/MAE 1.00-1.02, hit 49.4-50.2%. But that was measured on
    the UNDERLYING. On the OPTION the ratio is 1.33, because a long option's
    upside is unbounded and its downside floored at -100%. That convexity is
    available to any entry, so it is not edge -- but it does change the
    arithmetic a bracket faces, and it was never measured.

THE CONTROL IS THE POINT
    Every cell is also computed on `offday` entries -- same ticker, same
    minute-of-day, a nearby date on which the bot would NOT have entered (no
    threshold-passing trigger within +/-30min). If real and control hit their
    targets at the same rate, the indicator is telling you nothing a clock
    could not. Same construction as check_day_level_null, which is the only
    control here that is not blind to day selection.

Usage:
  python check_indicator_view.py --tickers SPY QQQ IWM
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import SLICE_EDGES
from check_day_level_null import build_controls
import sim_core

PRE_LO = pd.Timestamp("2023-10-12").date()
DEPLOY_LO = SLICE_EDGES[0]
SPLIT = pd.Timestamp("2025-08-21").date()
#: (target, stop) as fractions of the entry, both positive
BRACKETS = [(0.25, 0.25), (0.25, 0.50), (0.50, 0.25), (0.50, 0.50),
            (1.00, 0.50), (1.00, 1.00)]


def first_touch(path, eod_m, target, stop):
    """-> +1 target first, -1 stop first, 0 neither by EOD; or None.

    Entry is the `bot` fill (min(mid+0.01, ask)) and the levels are struck off
    it, because that is the price a trader would actually have paid. Touch is
    tested on the BID for the target (what you could sell into) and on the LOW
    for the stop (what would have taken you out intrabar) -- the pessimistic
    pairing, so the result is not flattered by quoting both sides at the mid.
    """
    e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
    if e_mid <= 0:
        return None
    entry = min(round(e_mid + 0.01, 2), round(e_ask, 2))
    if entry <= 0:
        return None
    up, dn = entry * (1 + target), entry * (1 - stop)
    n = len(cl)

    def _eod(i):
        """Realised return if neither level was touched -- the position is
        CLOSED AT THE BID, not written off as zero P&L. Scoring 'neither' as
        0.0 inflated the +100/-100 bracket to EV +23.0% by treating 77% of
        trades, most of them near-total losses on a 0DTE, as costless."""
        px = bid[i] if bid[i] > 0 else cl[i]
        if ask is not None and bid[i] > 0 and ask[i] > bid[i]:
            sp = ask[i] - bid[i]
            px = max(0.01, px - sp * (0.5 if px > entry else 1.5))
        return (px - entry) / entry

    for i in range(n):
        if mods[i] >= eod_m:
            return 0, _eod(i)
        if lo[i] <= dn:
            return -1, -stop
        if bid[i] >= up:
            return 1, target
    return 0, _eod(n - 1)


def tally(rows, target, stop):
    if not rows:
        return None
    v = np.array([r[1][0] for r in rows], int)
    r_ = np.array([r[1][1] for r in rows], float)
    n = len(v)
    pw, pl = (v == 1).mean(), (v == -1).mean()
    return dict(n=n, pw=pw, pl=pl, pn=(v == 0).mean(), ev=r_.mean())


def line(lbl, t, width=10):
    if t is None:
        print(f"    {lbl:{width}} (none)")
        return
    print(f"    {lbl:{width}} n={t['n']:>6}  target-first {t['pw']*100:>5.1f}%  "
          f"stop-first {t['pl']*100:>5.1f}%  neither {t['pn']*100:>5.1f}%  "
          f"EV {t['ev']*100:>+6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--seed", type=int, default=91)
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    rng = np.random.default_rng(a.seed)

    paths = {"real": [], "offday": []}
    for tk in a.tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        _, offday, _ = build_controls(trigs, a.pct, 30, 20, rng)
        for direction in a.dirs:
            rule = {"name": f"{tk} {direction}", "ticker": tk,
                    "direction": direction, "dte": [0, 1],
                    "min_flow_pct": a.pct, "target_roe": 1.0, "rr": 1.0}
            em = sim_core.eod_mod(rule)
            for arm, src in (("real", trigs), ("offday", offday)):
                if not src:
                    continue
                for d_, m_, p in sim_core.build_candidates(
                        D, rule, trigs=src, since=None):
                    paths[arm].append((d_, em, p))
        print(f"  {tk} done", flush=True)

    print(f"\n{'='*104}")
    print(f"  THE TRIGGER AS A 0DTE INDICATOR -- first touch of a bracket")
    print(f"  (p{a.pct}, {' '.join(a.tickers)}; no mechanical exit, entry at the "
          f"bot fill)")
    print(f"{'='*104}")
    for target, stop in BRACKETS:
        print(f"\n  target +{int(target*100)}%  /  stop -{int(stop*100)}%")
        res = {}
        for arm in ("real", "offday"):
            rows = []
            for d_, em, p in paths[arm]:
                r = first_touch(p, em, target, stop)
                if r is not None:
                    rows.append((d_, r))
            res[arm] = (tally(rows, target, stop), rows)
            line(arm, res[arm][0])
        ra, oa = res["real"][0], res["offday"][0]
        if ra and oa:
            print(f"    {'GAP':10} target-first {(ra['pw']-oa['pw'])*100:>+5.1f}pp"
                  f"   EV {(ra['ev']-oa['ev'])*100:>+6.1f}pp")
        # window stability on the real arm
        rows = res["real"][1]
        if rows:
            d = np.array([r[0] for r in rows])
            v = np.array([r[1][0] for r in rows], int)
            rr = np.array([r[1][1] for r in rows], float)
            parts = []
            for lbl, lo_, hi_ in (("PRE", PRE_LO, DEPLOY_LO),
                                  ("IS", DEPLOY_LO, SPLIT),
                                  ("OOS", SPLIT, SLICE_EDGES[-1])):
                m = (d >= lo_) & (d < hi_)
                if m.sum():
                    ev = rr[m].mean()
                    parts.append(f"{lbl} {ev*100:+.1f}%")
            print(f"    {'by window':10} " + "   ".join(parts))

    print(f"\n{'='*104}")
    print(f"  EV here is the BRACKET's expectancy only -- it ignores commission,")
    print(f"  the 'neither' tail (closed at EOD, usually near zero), and any")
    print(f"  discretion. Read the GAP row first: if real and offday hit their")
    print(f"  targets at the same rate, the indicator adds nothing a clock could")
    print(f"  not, whatever the absolute EV looks like.")


if __name__ == "__main__":
    main()

