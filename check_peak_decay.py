# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_peak_decay.py
===================
HOW MUCH OF THE PEAK IS ACTUALLY REACHABLE? -- the ceiling on exit engineering.

THE QUESTION THIS ANSWERS
    A live IWM trade peaked +364% and the 50% trail booked +98%. Capturing 27%
    of the peak feels inefficient -- but "efficient" presumes MFE is a target,
    and it is not: no causal rule can sell AT the peak, because the peak is only
    identifiable afterwards. Every exit needs LAG to recognise a turn.

    So the honest ceiling is not MFE. It is "what was the option still worth k
    minutes after its peak", for the smallest k a real detector could achieve.
    If value decays slowly, a better turn-detector is worth building. If the
    peak is a spike, then no exit reaches it and the deployed trail is already
    near the achievable frontier -- which would close the exit-engineering
    question rather than leaving it open.

WHY THIS IS NOT ANOTHER EXIT BACKTEST
    It tests no rule and fits no parameter. It measures a property of the price
    PATH: post-peak decay. That property bounds every exit rule that could ever
    be written, including ones nobody has thought of yet.

ALSO REPORTED
    * time-to-peak, and how long the position sits within 10% of its peak
      (the window a detector actually has to act in)
    * the share of the peak the DEPLOYED 50% trail captures, as the benchmark
      any new rule has to beat
    * how quickly the 50% retracement that triggers the trail arrives

Everything is measured on the BID -- the side an exit actually fills against --
and capped at the rule's EOD flatten.

Usage:
  python check_peak_decay.py --tickers SPY QQQ IWM
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import SLICE_EDGES
import sim_core

LAGS = (1, 2, 3, 5, 10, 15, 30)


def rows_for(cand, eod_m):
    """Per candidate: peak, time-to-peak, value at peak+k, trail capture."""
    out = []
    for d, m, path in cand:
        e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
        if e_mid <= 0:
            continue
        n = int(np.searchsorted(mods, eod_m, side="right"))
        if n < 5:
            continue
        b = np.asarray(bid[:n], float)
        mm = np.asarray(mods[:n], int)
        if b.max() <= 0:
            continue
        i = int(np.argmax(b))
        peak = b[i]
        rec = {"date": d, "peak_roe": peak / e_mid - 1.0,
               "ttp": int(mm[i] - m), "budget": int(eod_m - m)}
        # value retained k minutes after the peak
        for k in LAGS:
            j = int(np.searchsorted(mm, mm[i] + k))
            rec[f"lag{k}"] = (b[min(j, n - 1)] / peak) if peak > 0 else np.nan
        # minutes spent within 10% of the peak, from the peak onward
        near = b[i:] >= peak * 0.90
        rec["near_mins"] = int(np.sum(near))
        # when does the 50% retracement (the trail's trigger) arrive?
        below = np.where(b[i:] <= peak * 0.50)[0]
        rec["t_to_50"] = int(below[0]) if len(below) else -1
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rows = []
    for tk in a.tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        for direction in a.dirs:
            rule = {"name": f"{tk} {direction}", "ticker": tk,
                    "direction": direction, "dte": [0, 1],
                    "min_flow_pct": a.pct, "target_roe": 1.0, "rr": 1.0}
            cand = sim_core.build_candidates(D, rule, trigs=trigs, since=None)
            if cand:
                rows += rows_for(cand, sim_core.eod_mod(rule))
        print(f"  {tk} done", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("  nothing")
        return
    df.to_parquet("_peak_decay.parquet", index=False)

    print(f"\n{'='*100}")
    print(f"  POST-PEAK DECAY -- the ceiling on any exit rule "
          f"(p{a.pct}, {' '.join(a.tickers)}, n={len(df):,})")
    print(f"{'='*100}")
    print(f"  mean peak ROE {df['peak_roe'].mean()*100:+.1f}%   "
          f"median time-to-peak {df['ttp'].median():.0f}m   "
          f"median budget {df['budget'].median():.0f}m")
    print(f"\n  SHARE OF THE PEAK STILL AVAILABLE k MINUTES AFTER IT:")
    print(f"  {'lag':>5}  {'mean':>8}  {'median':>8}   (1.00 = sold exactly at the peak)")
    for k in LAGS:
        c = df[f"lag{k}"].dropna()
        print(f"  {k:>4}m  {c.mean():>8.3f}  {c.median():>8.3f}")

    print(f"\n  HOW LONG THE POSITION STAYS WITHIN 10% OF ITS PEAK:")
    q = np.percentile(df["near_mins"], [25, 50, 75, 90])
    print(f"    p25 {q[0]:.0f}m   median {q[1]:.0f}m   p75 {q[2]:.0f}m   p90 {q[3]:.0f}m")
    print(f"    -> this is the entire window a detector has to act in")

    hit = df[df["t_to_50"] >= 0]["t_to_50"]
    print(f"\n  TIME FROM PEAK TO THE 50% RETRACEMENT (the deployed trail's trigger):")
    print(f"    reached on {len(hit)/len(df)*100:.0f}% of trades   "
          f"median {np.median(hit):.0f}m   p25 {np.percentile(hit,25):.0f}m")

    print(f"\n  BY PEAK SIZE -- do the big winners decay differently?")
    print(f"  {'peak bucket':16} {'n':>6} {'lag1':>7} {'lag5':>7} {'lag15':>7} "
          f"{'near10%':>8}")
    for lbl, lo, hi in (("<+50%", -9, 0.50), ("+50..100%", 0.50, 1.00),
                        ("+100..200%", 1.00, 2.00), (">+200%", 2.00, 99)):
        g = df[(df["peak_roe"] >= lo) & (df["peak_roe"] < hi)]
        if len(g) < 20:
            continue
        print(f"  {lbl:16} {len(g):>6} {g['lag1'].mean():>7.3f} "
              f"{g['lag5'].mean():>7.3f} {g['lag15'].mean():>7.3f} "
              f"{g['near_mins'].median():>7.0f}m")

    print(f"\n  HOW TO READ IT")
    print(f"  A causal exit cannot sell at the peak -- it needs lag to see a turn.")
    print(f"  The lag column IS the ceiling: an exit that recognises the turn in")
    print(f"  k minutes cannot capture more than the lag-k share, before costs.")
    print(f"  Compare that ceiling with what the deployed trail already gets")
    print(f"  before concluding the exit is the thing worth engineering.")


if __name__ == "__main__":
    main()
