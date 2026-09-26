# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_run_coverage.py
=====================
WHAT FRACTION OF LARGE SPY RUNS DOES THE DEPLOYED TRIGGER ALREADY CATCH?

THE QUESTION BEHIND IT
    Every entry study in this project so far has tested PRECISION -- given a
    trigger, is the outcome better? Eight features have been tried that way and
    all eight failed, with the standing result that whatever marks losers marks
    winners too. Nobody has tested RECALL: of the moves worth catching, how many
    does the trigger never see?
    That is the cheap question to answer before building anything that mines
    large runs for their flow antecedents. If the trigger already fires ahead of
    most runs, such a study would mostly re-find the whale spike. If it fires
    ahead of very few, there is real headroom.

🚨 RECALL ALONE IS MEANINGLESS -- IT IS REPORTED AGAINST ITS OWN BASE RATE
    A trigger that fired every minute would "precede" 100% of runs. So every
    recall figure here is shown next to the share of ALL minutes that carry a
    trigger, and the ratio of the two. Lift ~1.0 means the trigger's apparent
    coverage is exactly what firing that often buys you at random, and the
    coverage number is an artefact of frequency. This is the control that can
    kill the result, and it is computed in the same table rather than left to
    the reader.

🚨 RUNS ARE DEFINED CAUSALLY AND NON-OVERLAPPING
    A run STARTS at minute m if SPY's close rises >= the threshold within the
    next WINDOW minutes. Overlapping starts are collapsed -- during a $2 climb
    every minute has a qualifying forward move, and counting them all would
    inflate the denominator by an order of magnitude and flatter any feature
    that persists. After a run is recorded the scan jumps to the minute of its
    peak. Nothing peeks beyond the window, and a start is only allowed where the
    whole window fits inside RTH.

THRESHOLDS ARE IN DOLLARS *AND* IN ATR
    $1.00 is not a fixed quantity: SPY was 437 in Oct 2023 and 763 now, so the
    same dollar is 0.23% then and 0.13% now. Worse, an unnormalised threshold
    selects high-volatility SESSIONS, and any "similarity" found among the runs
    is then a restatement of that selection. Both are reported.

    Note also what the EXIT can monetise. A 0DTE ATM call at ~50 delta priced
    near $2 gains roughly $0.50 on a $1.00 underlying move -- about +25% ROE --
    while a 50% trail only clears entry above +100% peak. A $1.00 run is inside
    the dead zone: predicted perfectly, still a loss. The larger thresholds are
    the ones that matter for deployment.

Usage:
  python check_run_coverage.py
  python check_run_coverage.py --window 30 --lookback 10
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

import sim_core

RTH0, RTH1 = 570, 955          # 09:30 - 15:55 ET


def spy_bars(since):
    df = (pl.scan_parquet("historical/SPY.parquet")
          .select("date", "minute_et", "close", "high", "low", "volume")
          .collect().to_pandas())
    t = pd.to_datetime(df["minute_et"])
    df["mod"] = t.dt.hour * 60 + t.dt.minute
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["mod"] >= RTH0) & (df["mod"] <= RTH1) & (df["date"] >= since)]
    return df.sort_values(["date", "mod"]).reset_index(drop=True)


def day_atr(df, n=14):
    """Prior-day ATR per session, so today's threshold uses no future bar."""
    d = (df.groupby("date")
           .agg(hi=("high", "max"), lo=("low", "min"), cl=("close", "last")))
    tr = pd.concat([d["hi"] - d["lo"],
                    (d["hi"] - d["cl"].shift()).abs(),
                    (d["lo"] - d["cl"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=5).mean().shift(1)


def find_runs(closes, mods, thr, window):
    """-> [(start_mod, peak_mod, size)] non-overlapping, causal."""
    out, i, n = [], 0, len(closes)
    while i < n:
        if mods[i] + window > RTH1:
            break
        j = min(i + window, n - 1)
        fwd = closes[i + 1:j + 1]
        if len(fwd) == 0:
            break
        k = int(np.argmax(fwd))
        move = fwd[k] - closes[i]
        if move >= thr:
            out.append((int(mods[i]), int(mods[i + 1 + k]), float(move)))
            i = i + 1 + k            # jump past the peak: no overlap
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30,
                    help="minutes a run has to develop")
    ap.add_argument("--lookback", type=int, default=10,
                    help="a trigger 'precedes' a run if it fired within this many "
                         "minutes before the start")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    since = sim_core.DEPLOYED_START
    bars = spy_bars(since)
    atr = day_atr(bars)
    days = sorted(bars["date"].unique())
    print(f"  SPY {len(bars):,} RTH bars over {len(days)} sessions "
          f"({days[0]} .. {days[-1]})")

    # ---- triggers: raw signal, and the deployed rule with all its gates -----
    flow = _flow_for(D, ["SPY"])
    trigs = D.triggers_for(flow, "SPY")
    raw = {}
    for t in trigs:
        if t["date"] < since or t["dir"] != "CALL":
            continue
        ts = pd.Timestamp(t["ts"])
        raw.setdefault(t["date"], set()).add(ts.hour * 60 + ts.minute)

    rule = next((r for r in sim_core.research_rules(include_paper=True)
                 if r["ticker"] == "SPY" and r["direction"] == "CALL"), None)
    dep = {}
    if rule is not None:
        for d, m, _p in sim_core.build_candidates(D, rule):
            dep.setdefault(d, set()).add(int(m))
    n_raw = sum(len(v) for v in raw.values())
    n_dep = sum(len(v) for v in dep.values())
    print(f"  CALL triggers: raw signal {n_raw:,}   "
          f"deployed rule '{rule['name'] if rule else '--'}' {n_dep:,}\n")

    # BASE RATE, COMPUTED NOT ESTIMATED.
    # The obvious formula -- triggers * (lookback+1) / minutes -- assumes the
    # windows never overlap. They do, heavily: 20,315 raw triggers * 11 minutes
    # exceeds the 196,733 minutes available, so the estimate saturated at 100%
    # and produced the impossible reading of 64.9% recall against a 100% base.
    # So build the actual covered-minute SET per day and measure its share.
    def covered_share(trig_map):
        cov = elig = 0
        for d, g in bars.groupby("date"):
            mods = g["mod"].to_numpy(int)
            mods = mods[mods + a.window <= RTH1]
            if not len(mods):
                continue
            T = trig_map.get(d, set())
            elig += len(mods)
            if T:
                cov += sum(1 for m in mods
                           if any(x in T for x in range(m - a.lookback, m + 1)))
        return cov / max(elig, 1), elig

    br_raw, tot_min = covered_share(raw)
    br_dep, _ = covered_share(dep)
    print(f"  minutes within {a.lookback}m after a trigger: "
          f"raw {br_raw*100:.1f}%   deployed {br_dep*100:.2f}%   "
          f"(of {tot_min:,} eligible)")

    print(f"  window {a.window}m   a trigger counts if it fired within "
          f"{a.lookback}m before the run starts\n")
    hdr = (f"  {'threshold':>14} {'runs':>6} {'days':>5} {'per day':>8}  "
           f"{'RAW recall':>11} {'base':>7} {'lift':>6}  "
           f"{'DEPLOYED recall':>16} {'base':>7} {'lift':>6}")
    print(hdr)

    def coverage(thr_fn, label):
        runs, hit_raw, hit_dep, rdays = 0, 0, 0, set()
        for d, g in bars.groupby("date"):
            thr = thr_fn(d)
            if thr is None or not np.isfinite(thr):
                continue
            ev = find_runs(g["close"].to_numpy(float), g["mod"].to_numpy(int),
                           thr, a.window)
            if not ev:
                continue
            rdays.add(d)
            R, Dp = raw.get(d, set()), dep.get(d, set())
            for s, _pk, _mv in ev:
                runs += 1
                w = range(s - a.lookback, s + 1)
                hit_raw += any(m in R for m in w)
                hit_dep += any(m in Dp for m in w)
        if not runs:
            print(f"  {label:>14} {'no runs':>6}")
            return
        rr, rd = hit_raw / runs, hit_dep / runs
        print(f"  {label:>14} {runs:>6} {len(rdays):>5} "
              f"{runs/max(len(rdays),1):>8.1f}  "
              f"{rr*100:>10.1f}% {br_raw*100:>6.1f}% {rr/max(br_raw,1e-9):>6.2f}  "
              f"{rd*100:>15.1f}% {br_dep*100:>6.1f}% {rd/max(br_dep,1e-9):>6.2f}")

    print("  --- fixed dollar thresholds ---")
    for thr in (1.0, 2.0, 3.0, 4.0):
        coverage(lambda d, t=thr: t, f"${thr:.2f}")

    print("\n  --- prior-day ATR-normalised (size-free) ---")
    for f in (0.25, 0.50, 0.75):
        coverage(lambda d, f=f: (atr.get(d) * f
                                 if d in atr.index and np.isfinite(atr.get(d))
                                 else None), f"{f:.2f} ATR")

    print(f"\n  HOW TO READ IT")
    print(f"  LIFT is the column that matters. 1.0 means the trigger precedes")
    print(f"  runs exactly as often as firing that frequently would by chance --")
    print(f"  the recall is bought with frequency, not information. Well above")
    print(f"  1.0 means the signal genuinely leads the move.")
    print(f"  LOW recall with HIGH lift is the interesting case: the trigger is")
    print(f"  right when it fires but blind to most runs, which is exactly the")
    print(f"  headroom a flow-antecedent study would be looking for.")
    print(f"  HIGH recall at lift ~1.0 means a mining study would rediscover")
    print(f"  the whale spike and call it a finding.")


if __name__ == "__main__":
    main()
