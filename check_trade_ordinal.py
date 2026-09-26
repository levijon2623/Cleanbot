# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_trade_ordinal.py
======================
IS THE 2nd TRADE OF A DAY REALLY WORTH 22x THE 1st -- OR IS IT THREE OUTLIERS?

WHERE THIS COMES FROM
    check_trades_per_day: cap=1 returns +124.5pp OOS (4.7% of the book), cap=2
    returns +2908.6pp (109.8%). The whole difference is the SECOND trade of a
    day -- 25 of them OOS -- worth +2784.1pp with a 95% CI of [+453, +6334].
    That interval is nearly an order of magnitude wide, which is what a total
    dominated by a few observations looks like. Two things are therefore needed
    before the cap=2 finding is worth anything.

1. A PROPERLY POWERED ORDINAL TEST -- WITHIN-DAY PAIRING
    Comparing two whole books (cap=1 vs cap=2) differences out almost none of
    the noise: both arms contain the same first trades, and every day-level
    shock lands in both. The paired form asks the question directly --
        for each rule-day with >= 2 trades:   pnl(2nd) - pnl(1st)
    -- which cancels the day effect exactly, because both trades share the same
    ticker, the same session, the same regime and the same volatility. What is
    left is the ordinal. Bootstrapped over RULE-DAYS (METHODOLOGY 7: the day is
    the unit, never the trade).

2. AN OUTLIER AUTOPSY
    A mean is not evidence when n=25. Reported: every 2nd-trade return sorted,
    the share of the total carried by the top 1/3/5, the total with those
    removed, the median, the positive rate, and a leave-one-DAY-out jackknife of
    the whole cap=1 -> cap=2 delta. If dropping one day moves the delta by more
    than the delta itself, the finding is one day wearing a p-value.

    A trimmed result that survives is a real, if smaller, effect. One that
    collapses to zero means the book's edge is a lottery ticket and cap=2 is not
    a policy but a bet on catching the same ticket again.

Usage:
  python check_trade_ordinal.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()


def boot_mean(v, groups, n, rng):
    """Cluster bootstrap of a mean, resampling GROUPS (rule-days)."""
    g = pd.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in g}
    out = np.empty(n)
    for i in range(n):
        s = np.concatenate([idx[k] for k in rng.choice(g, size=len(g), replace=True)])
        out[i] = v[s].mean()
    return tuple(np.percentile(out, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default="bot")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rows = []
    for rule in sim_core.research_rules():
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
        res = sim_core.walk(cand, pol, sim_core.eod_mod(rule), fill=a.fill,
                            with_tags=True)
        seen = {}
        for d, p, tag in res:
            seen[d] = seen.get(d, 0) + 1
            rows.append(dict(rule=rule["name"], date=d, ordinal=seen[d],
                             pnl=p, tag=tag))
        print(f"  {rule['name']} done", flush=True)

    T = pd.DataFrame(rows)
    if T.empty:
        print("  nothing"); return
    T["oos"] = T["date"] >= SPLIT
    T["rd"] = T["rule"] + "|" + T["date"].astype(str)
    T.to_parquet("_trade_ordinal.parquet", index=False)

    print(f"\n{'='*94}")
    print(f"  TRADE RETURN BY WITHIN-DAY ORDINAL   (n={len(T):,}, "
          f"{T['date'].nunique()} dates)")
    print(f"{'='*94}")
    for lbl, sub in (("IS", T[~T["oos"]]), ("OOS", T[T["oos"]])):
        print(f"\n  {lbl}")
        print(f"    {'ordinal':9} {'n':>5} {'mean %':>9} {'median %':>9} "
              f"{'win rate':>9} {'total pp':>10}")
        for o in sorted(sub["ordinal"].unique()):
            g = sub[sub["ordinal"] == o]
            if len(g) < 3:
                continue
            print(f"    {o:<9} {len(g):>5} {g['pnl'].mean()*100:>+9.2f} "
                  f"{g['pnl'].median()*100:>+9.2f} {(g['pnl']>0).mean()*100:>8.0f}% "
                  f"{g['pnl'].sum()*100:>+10.1f}")

    print(f"\n{'='*94}")
    print(f"  1. WITHIN-DAY PAIRED TEST -- same rule, same day, ordinal n vs n-1")
    print(f"{'='*94}")
    print(f"  {'pair':14} {'rule-days':>10} {'mean diff pp':>14} {'95% CI':>22} "
          f"{'median':>9}")
    for lo, hi in ((1, 2), (2, 3)):
        for lbl, sub in (("IS", T[~T["oos"]]), ("OOS", T[T["oos"]])):
            piv = sub.pivot_table(index="rd", columns="ordinal", values="pnl")
            if lo not in piv.columns or hi not in piv.columns:
                continue
            p = piv[[lo, hi]].dropna()
            if len(p) < 5:
                continue
            d = (p[hi] - p[lo]).to_numpy(float) * 100
            ci = boot_mean(d, p.index.to_numpy(), a.boot, rng)
            sig = "  <--" if (ci[0] > 0 or ci[1] < 0) else ""
            print(f"  {lbl} {lo}->{hi:<9} {len(p):>10} {d.mean():>+14.2f} "
                  f"[{ci[0]:>+9.2f},{ci[1]:>+9.2f}] {np.median(d):>+9.2f}{sig}")
    print(f"  -> the pair cancels the day: same ticker, session, regime and vol.")
    print(f"     A null here with a huge cap=1-vs-cap=2 gap means the gap is NOT")
    print(f"     about ordinal quality.")

    print(f"\n{'='*94}")
    print(f"  2. OUTLIER AUTOPSY -- the OOS 2nd trades that carry the cap=2 finding")
    print(f"{'='*94}")
    o2 = T[(T["oos"]) & (T["ordinal"] == 2)].sort_values("pnl", ascending=False)
    o1 = T[(T["oos"]) & (T["ordinal"] == 1)]
    tot2 = o2["pnl"].sum() * 100
    print(f"  n={len(o2)} second trades, total {tot2:+.1f}pp "
          f"(vs {len(o1)} first trades, total {o1['pnl'].sum()*100:+.1f}pp)")
    print(f"\n  every 2nd trade, largest first:")
    for i, r in enumerate(o2.itertuples(), 1):
        print(f"    {i:>3}. {r.rule:22} {str(r.date)}  {r.pnl*100:>+9.1f}%  ({r.tag})")
    print(f"\n  CONCENTRATION")
    v = o2["pnl"].to_numpy(float) * 100
    for k in (1, 2, 3, 5):
        if k <= len(v):
            print(f"    top {k}: {v[:k].sum():>+9.1f}pp "
                  f"({v[:k].sum()/tot2*100 if tot2 else np.nan:>5.1f}% of total)   "
                  f"remainder {v[k:].sum():>+9.1f}pp")
    print(f"    median {np.median(v):+.1f}%   win rate {(v>0).mean()*100:.0f}%   "
          f"mean {v.mean():+.1f}%")
    if len(v) > 4:
        tr = np.sort(v)[2:-2]
        print(f"    10%-trimmed mean (drop 2 each tail): {tr.mean():+.1f}% "
              f"-> implied total {tr.mean()*len(v):+.1f}pp")

    print(f"\n  LEAVE-ONE-DAY-OUT JACKKNIFE of the cap=1 -> cap=2 delta")
    days = sorted(set(o2["date"]))
    deltas = []
    for d in days:
        deltas.append(o2[o2["date"] != d]["pnl"].sum() * 100)
    deltas = np.array(deltas)
    worst = days[int(np.argmin(deltas))]
    print(f"    full delta {tot2:+.1f}pp")
    print(f"    dropping one day: min {deltas.min():+.1f}  max {deltas.max():+.1f}  "
          f"spread {deltas.max()-deltas.min():.1f}pp")
    print(f"    most influential day {worst} -- removing it costs "
          f"{tot2 - deltas.min():.1f}pp ({(tot2-deltas.min())/tot2*100:.0f}% of the total)")

    print(f"\n  HOW TO READ IT")
    print(f"  If section 1 is null while cap=1 vs cap=2 is enormous, the second")
    print(f"  trade is not a BETTER trade -- it is an EXTRA one, and the book simply")
    print(f"  needs more than one shot per day.")
    print(f"  If section 2 shows one or two days carrying most of the total, cap=2 is")
    print(f"  not a policy conclusion, it is a bet that the same lottery ticket")
    print(f"  reappears out of sample.")


if __name__ == "__main__":
    main()
