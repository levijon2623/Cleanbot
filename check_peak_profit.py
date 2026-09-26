# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_peak_profit.py
====================
PEAK PROFIT (MFE) AND WORST DRAWDOWN (MAE) ON THE OPTION ITSELF, which is
exit-independent -- so it looks at the trigger with the exit policy removed.

WHY THIS IS NOT ALREADY DONE
    check_signal_quality measured MFE/MAE on the UNDERLYING and found 1.00-1.02
    against a null of 1.00 -- no standalone directional edge. That does NOT
    settle the option, because convexity and theta mean the option's excursion
    is not a linear image of the underlying's: a move that is unremarkable in
    the stock can double a 0DTE contract, and a flat tape bleeds the option
    while leaving the stock's MFE untouched.

WHY THE NULL IS MANDATORY HERE, NOT OPTIONAL
    MFE is monotone in TIME BUDGET -- the longer you hold, the higher the
    running maximum, with no skill involved. The flow trigger fires earlier in
    the day more often than a uniform draw would, so its trades get more budget
    and a higher peak FOR FREE. This is exactly the confound that inflated
    check_step1_redo's headline (see check_policy_matrix).

    Control: every real trigger is re-timed to a RANDOM MINUTE WITHIN ITS OWN
    HOUR, and the same contract-selection and path logic is re-run. Same day,
    same hour, same budget to within an hour, same instrument -- only the
    signal's precise timing is destroyed. The difference between the real peak
    and the control peak is the part the trigger can claim.

WHAT IS REPORTED
    MFE at the BID (what could actually be sold into) and at the MID, MAE, the
    MFE/MAE ratio, time-to-peak, and the share of trades whose peak ever clears
    +25 / +50 / +100%. That last one is directly actionable: sim_core.policy_for
    notes a 50% trail only rises above the entry once peak ROE exceeds +100%, so
    the share clearing +100% is an upper bound on how often the deployed trail
    can exit green at all.

Usage:
  python check_peak_profit.py --tickers SPY QQQ IWM --pct 65
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


def excursions(cand, eod_m):
    """[(date, mod, mfe_bid, mfe_mid, mae_bid, mins_to_peak, budget)] per candidate.

    Everything is measured from the entry MID (the bracket is priced off the
    mid) and capped at the EOD flatten, so no excursion is credited to minutes
    the bot would never have held through.
    """
    out = []
    for d, m, path in cand:
        e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
        if e_mid <= 0:
            continue
        k = int(np.searchsorted(mods, eod_m, side="right"))
        if k < 2:
            continue
        b = np.asarray(bid[:k], float)
        a = np.asarray(ask[:k], float)
        mid = np.where((b > 0) & (a > 0), (b + a) / 2.0, np.asarray(cl[:k], float))
        mm = np.asarray(mods[:k], int)
        i = int(np.argmax(b))
        out.append((d, m,
                    b.max() / e_mid - 1.0,
                    mid.max() / e_mid - 1.0,
                    b.min() / e_mid - 1.0,
                    int(mm[i] - m),
                    int(eod_m - m)))
    return out


def retime(trigs, rng):
    """Copy the triggers, moving each to a random minute WITHIN ITS OWN HOUR.

    Keeps date, hour and therefore the time budget to within an hour; destroys
    the signal's precise timing. The trigger dicts are deep-copied so the real
    list is never mutated -- a subtle way to silently null out the real arm.
    """
    out = []
    for t in trigs:
        s = copy.deepcopy(t)
        ts = pd.Timestamp(t["ts"])
        newmin = int(rng.integers(0, 60))
        s["ts"] = ts.replace(minute=newmin)
        out.append(s)
    return out


def summarise(rows, label):
    if not rows:
        print(f"  {label:22} (none)")
        return None
    d = np.array([r[0] for r in rows])
    mfe_b = np.array([r[2] for r in rows])
    mfe_m = np.array([r[3] for r in rows])
    mae_b = np.array([r[4] for r in rows])
    ttp = np.array([r[5] for r in rows], float)
    bud = np.array([r[6] for r in rows], float)
    ratio = abs(mfe_b.mean() / mae_b.mean()) if abs(mae_b.mean()) > 1e-9 else np.nan
    print(f"  {label:22} n={len(rows):>5}  MFEbid {mfe_b.mean()*100:>+7.1f}%  "
          f"MFEmid {mfe_m.mean()*100:>+7.1f}%  MAE {mae_b.mean()*100:>+7.1f}%  "
          f"ratio {ratio:>4.2f}  peak@{ttp.mean():>5.0f}m  budget {bud.mean():>5.0f}m")
    return dict(mfe_b=mfe_b, mfe_m=mfe_m, mae_b=mae_b, dates=d, ttp=ttp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--seed", type=int, default=53)
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    rng = np.random.default_rng(a.seed)

    real, null = [], []
    for tk in a.tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        shifted = retime(trigs, rng)
        D.annotate_flow_pct(shifted, 60)
        for direction in a.dirs:
            rule = {"name": f"{tk} {direction}", "ticker": tk,
                    "direction": direction, "dte": [0, 1],
                    "min_flow_pct": a.pct, "target_roe": 1.0, "rr": 1.0}
            em = sim_core.eod_mod(rule)
            for src, bucket in ((trigs, real), (shifted, null)):
                cand = sim_core.build_candidates(D, rule, trigs=src, since=None)
                if cand:
                    bucket += excursions(cand, em)
        print(f"  {tk} done", flush=True)

    print(f"\n{'='*118}")
    print(f"  PEAK PROFIT / DRAWDOWN ON THE OPTION  (p{a.pct}, "
          f"{' '.join(a.tickers)}, exit-independent)")
    print(f"{'='*118}")
    R = summarise(real, "REAL triggers")
    N = summarise(null, "same-hour control")
    if not R or not N:
        return

    print(f"\n  DIFFERENCE (real - control), the part the trigger can claim:")
    for k, lbl in (("mfe_b", "peak profit @bid"), ("mae_b", "worst drawdown")):
        print(f"    {lbl:20} {(R[k].mean() - N[k].mean())*100:>+6.2f}pp")

    print(f"\n  SHARE OF TRADES WHOSE PEAK EVER CLEARS:")
    print(f"  {'threshold':12} {'REAL':>8} {'control':>9}   note")
    for th, note in ((0.25, ""), (0.50, ""),
                     (1.00, "<- a 50% trail cannot exit GREEN below this")):
        print(f"  {'+'+str(int(th*100))+'%':12} {(R['mfe_b']>=th).mean()*100:>7.1f}% "
              f"{(N['mfe_b']>=th).mean()*100:>8.1f}%   {note}")

    print(f"\n  BY WINDOW (real, peak profit @bid):")
    for lbl, lo, hi in (("PRE", PRE_LO, DEPLOY_LO), ("IS", DEPLOY_LO, SPLIT),
                        ("OOS", SPLIT, SLICE_EDGES[-1])):
        m = (R["dates"] >= lo) & (R["dates"] < hi)
        if m.sum():
            print(f"    {lbl:4} n={int(m.sum()):>5}  "
                  f"MFE {R['mfe_b'][m].mean()*100:>+6.1f}%  "
                  f"MAE {R['mae_b'][m].mean()*100:>+6.1f}%")

    # ---- ordinal split: is the known "first trigger is the worst" effect an
    # ENTRY problem or an EXIT problem? MFE is exit-independent, so if trade #1
    # has the same peak as #2/#3 then its badness is the exit failing to capture
    # a peak that was there; if its peak is genuinely lower, the entry is worse.
    print(f"\n{'='*118}")
    print(f"  PEAK PROFIT BY WITHIN-DAY TRIGGER ORDINAL")
    print(f"  (memory: trade #1 is systematically the worst -- OOS -12.0%, n=187,")
    print(f"   and the effect is ORDINAL not temporal. MFE says WHY.)")
    print(f"{'='*118}")
    dfr = pd.DataFrame(real, columns=["date", "mod", "mfe_b", "mfe_m",
                                      "mae_b", "ttp", "budget"])
    dfr["rank"] = dfr.sort_values("mod").groupby("date").cumcount() + 1
    dfr["bucket"] = np.where(dfr["rank"] >= 4, "4+", dfr["rank"].astype(str))
    print(f"  {'trigger #':10} {'n':>6} {'MFEbid':>9} {'MAE':>9} {'ratio':>7} "
          f"{'budget':>8}  {'>=+100%':>8}")
    for b in ["1", "2", "3", "4+"]:
        g = dfr[dfr["bucket"] == b]
        if g.empty:
            continue
        r = abs(g["mfe_b"].mean() / g["mae_b"].mean()) if abs(g["mae_b"].mean()) > 1e-9 else np.nan
        print(f"  {b:10} {len(g):>6} {g['mfe_b'].mean()*100:>+8.1f}% "
              f"{g['mae_b'].mean()*100:>+8.1f}% {r:>7.2f} "
              f"{g['budget'].mean():>7.0f}m {(g['mfe_b']>=1.0).mean()*100:>7.1f}%")
    g1 = dfr[dfr["bucket"] == "1"]
    rest = dfr[dfr["bucket"] != "1"]
    if len(g1) and len(rest):
        d_mfe = (g1["mfe_b"].mean() - rest["mfe_b"].mean()) * 100
        d_bud = g1["budget"].mean() - rest["budget"].mean()
        print(f"\n  trade #1 vs the rest:  MFE {d_mfe:+.1f}pp   "
              f"budget {d_bud:+.0f}min")
        print(f"  If MFE is COMPARABLE, trade #1's known -12% OOS is an EXIT")
        print(f"  failure -- the peak was there and was not captured. If MFE is")
        print(f"  materially LOWER, the entry itself is worse and skipping it is")
        print(f"  justified on entry quality. Note budget differs, so read the")
        print(f"  MFE gap against the budget gap, not on its own.")

    print(f"\n  READ IT AS AN UPPER BOUND. MFE is what a PERFECT exit would have")
    print(f"  captured -- it is unattainable by construction, and the control")
    print(f"  shows how much of it a random entry in the same hour also gets.")


if __name__ == "__main__":
    main()
