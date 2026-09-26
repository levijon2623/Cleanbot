# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_contra_filter.py
======================
BULL-SHARE AS A FILTER ON THE BOOK THE BOT ALREADY TRADES.

WHY A FILTER AND NOT A TRIGGER
    check_contra_trigger ran the contrarian bull-share signal as a standalone
    entry and it failed every criterion: 0/8 held-out tickers positive, 0/6
    slices, OOS -36,176, and a share-SHUFFLED placebo that scored slightly
    BETTER (-29.4 vs -30.2 median ROE). It also fired 6,696 times in 151
    sessions -- roughly 44 trades a day -- because stripping the rules' gates to
    test the signal cleanly also stripped what makes the book work.

    That run isolated where the edge actually lives: the GATES, not the trigger.
    The deployed whale spike through the same permissive rule loses -234k
    in-sample across 48,639 trades; with its regime/AMT/flow-percentile gates it
    is the profitable book. So the only remaining form worth testing keeps every
    gate intact and asks a much smaller question: among the ~2,400 trades the
    book ALREADY takes, does the bull share at entry separate them?

    Precedent and prior: check_flow_align asked exactly this shape of question
    of cumulative-flow alignment and it failed (6/9 rules, mean +0.1pp). This is
    the tenth feature through this gate.

THE HYPOTHESIS, DIRECTIONAL AND PRE-STATED
    From check_run_antecedent, heavy aggressive CALL premium preceded DOWN
    moves. So the filter is CONTRARIAN:
        a CALL trade is BETTER when the bull share at entry is LOW
        a PUT  trade is BETTER when the bull share at entry is HIGH
    Stating the direction in advance matters -- with three bins and two
    directions, a post-hoc reading of whichever cell looks good is guaranteed a
    winner.

🚨 WHY THE SAMPLE IS THE WHOLE BOOK
    The 30s cache was built from the 151 days that CARRY the book's candidates,
    so every deployed candidate has 30s flow at its entry minute. This is not a
    subset -- it is the same 2,396 trades, with one extra column.

CAUSAL BINNING
    The share is cut on a TRAILING 60-minute tercile with the current minute
    excluded (shift 1). Cutting on the pooled sample median is the look-ahead
    that manufactured the "fading the bull" gradient and then inverted when it
    was measured honestly (METHODOLOGY 7 trap list).

Usage:
  python check_contra_filter.py --paper
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

import sim_core

CACHE = "_flow30_cache"
SPLIT = pd.Timestamp("2025-08-21").date()


def share_map():
    """-> {(date, ticker, mod): tercile} 0 = lowest bull share, 2 = highest."""
    f = sorted(glob.glob(os.path.join(CACHE, "date=*", "flow30.parquet")))
    if not f:
        raise SystemExit(f"  no 30s buckets in {CACHE}")
    cols = ("ca", "cb", "pa", "pb")
    df = pl.concat([pl.read_parquet(p).with_columns(
        [pl.col(c).cast(pl.Float64) for c in cols]) for p in f]).to_pandas()
    df["bull"] = df["ca"] + df["pb"]
    df["bear"] = df["cb"] + df["pa"]
    g = (df.groupby(["underlying_symbol", "date", "mod"])[["bull", "bear"]]
           .sum().reset_index())
    g["date"] = pd.to_datetime(g["date"]).dt.date
    out, raw = {}, {}
    for (tk, d), x in g.groupby(["underlying_symbol", "date"]):
        x = x.sort_values("mod")
        tot = (x["bull"] + x["bear"]).replace(0, np.nan)
        sh = x["bull"] / tot
        lo = sh.rolling(60, min_periods=20).quantile(1 / 3).shift(1)
        hi = sh.rolling(60, min_periods=20).quantile(2 / 3).shift(1)
        for m, s, l, u in zip(x["mod"], sh, lo, hi):
            if not np.isfinite(s) or not np.isfinite(l) or not np.isfinite(u):
                continue
            out[(d, tk, int(m))] = 0 if s <= l else (2 if s >= u else 1)
            raw[(d, tk, int(m))] = float(s)
    return out, raw


def agg(rows, label):
    if not rows:
        return f"  {label:26} {'(none)':>6}"
    v = np.array([r["pnl"] for r in rows], float)
    o = [r["pnl"] for r in rows if r["date"] >= SPLIT]
    return (f"  {label:26} {len(v):>6} {len({r['date'] for r in rows}):>6} "
            f"{(v <= -50).mean()*100:>7.1f}% {(v > 0).mean()*100:>6.1f}% "
            f"{np.median(v):>+9.1f} {(np.sum(o) if o else 0):>+10.0f}")


HDR = (f"  {'bin':26} {'n':>6} {'days':>6} {'loss50':>8} {'win':>7} "
       f"{'medROE':>9} {'OOS':>10}")
NAME = {0: "share LOW (bottom 3rd)", 1: "share MID", 2: "share HIGH (top 3rd)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--seed", type=int, default=53)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D

    SH, RAW = share_map()
    print(f"  30s share map: {len(SH):,} ticker-minutes\n")

    rows, missing = [], 0
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dirn = rule["ticker"], rule["direction"]
        cand = sim_core.build_candidates(D, rule)
        if not cand:
            continue
        pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        for d, m, path in cand:
            b = SH.get((d, tk, int(m)))
            if b is None:
                missing += 1
                continue
            pnl, xm, tag = sim_core.simulate(path, pol, eod, fill=a.fill,
                                             cush_cap=cap)
            # HYPOTHESIS BIN: contrarian. A CALL wants a LOW bull share, a PUT
            # wants a HIGH one. `good` is what the pre-stated hypothesis says
            # should be the better half.
            good = (b == 0) if dirn == "CALL" else (b == 2)
            bad = (b == 2) if dirn == "CALL" else (b == 0)
            rows.append(dict(rule=rule["name"], ticker=tk, dir=dirn, date=d,
                             pnl=pnl * 100, bin=b, good=good, bad=bad,
                             share=RAW.get((d, tk, int(m)))))
        print(f"    {rule['name']} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  nothing"); return
    R.to_parquet("_contra_filter.parquet", index=False)
    R["placebo"] = R.groupby(["ticker", "date"])["bin"].transform(
        lambda s: s.to_numpy()[rng.permutation(len(s))])
    recs = R.to_dict("records")
    print(f"\n  {len(R):,} book trades matched to a 30s share "
          f"({missing} had no 30s bucket at entry)\n")

    print(f"{'='*96}")
    print(f"  1. BY SHARE TERCILE, WITHIN DIRECTION (the hypothesis is directional)")
    print(f"{'='*96}")
    for dirn in ("CALL", "PUT"):
        sub = [r for r in recs if r["dir"] == dirn]
        if not sub:
            continue
        want = 0 if dirn == "CALL" else 2
        print(f"\n  --- {dirn} (hypothesis: better when share is "
              f"{'LOW' if dirn == 'CALL' else 'HIGH'}) ---")
        print(HDR)
        for b in (0, 1, 2):
            mark = "  <- predicted better" if b == want else ""
            print(agg([r for r in sub if r["bin"] == b], NAME[b]) + mark)

    print(f"\n{'='*96}")
    print(f"  2. THE FILTER ITSELF -- skip the predicted-bad half")
    print(f"{'='*96}")
    print(HDR)
    print(agg(recs, "ALL (book as deployed)"))
    print(agg([r for r in recs if not r["bad"]], "SKIP predicted-bad"))
    print(agg([r for r in recs if r["good"]], "ONLY predicted-good"))
    print(f"\n  --- placebo: same bins, shuffled within ticker-day ---")
    print(HDR)
    for b in (0, 1, 2):
        print(agg([r for r in recs if r["placebo"] == b], f"placebo {NAME[b]}"))

    print(f"\n{'='*96}")
    print(f"  3. WITHIN RULE -- one rule or the book? (C5)")
    print(f"{'='*96}")
    print(f"  {'rule':24} {'good n':>7} {'loss50':>8} {'bad n':>7} {'loss50':>8} "
          f"{'delta':>8}")
    deltas = []
    for nm, g in R.groupby("rule"):
        gd, bd = g[g["good"]], g[g["bad"]]
        if len(gd) < 15 or len(bd) < 15:
            print(f"  {nm:24} {len(gd):>7} {'--':>8} {len(bd):>7} {'--':>8}  too few")
            continue
        lg = (gd["pnl"] <= -50).mean() * 100
        lb = (bd["pnl"] <= -50).mean() * 100
        deltas.append(lg - lb)
        print(f"  {nm:24} {len(gd):>7} {lg:>7.1f}% {len(bd):>7} {lb:>7.1f}% "
              f"{lg-lb:>+8.1f}")
    if deltas:
        print(f"\n  rules where the predicted-good half has LOWER loss50: "
              f"{sum(1 for x in deltas if x < 0)}/{len(deltas)}")
        print(f"  mean within-rule delta {np.mean(deltas):+.1f}pp "
              f"(negative = the hypothesis holds)")

    print(f"\n{'='*96}")
    print(f"  4. PAIRED WITHIN DAY -- removes day composition")
    print(f"{'='*96}")
    dl = []
    for (tk, d), g in R.groupby(["ticker", "date"]):
        gd, bd = g[g["good"]], g[g["bad"]]
        if len(gd) and len(bd):
            dl.append(gd["pnl"].median() - bd["pnl"].median())
    if len(dl) >= 10:
        dl = np.array(dl)
        print(f"  {len(dl)} paired ticker-days   mean delta medROE {dl.mean():>+8.1f}"
              f"   share favouring the hypothesis {(dl > 0).mean()*100:.0f}%")
    else:
        print(f"  only {len(dl)} paired ticker-days -- the bins rarely co-occur,")
        print(f"  so block 2 cannot be separated from day composition.")

    print(f"\n  HOW TO READ IT")
    print(f"  Block 2 is the deployable question: does skipping the predicted-bad")
    print(f"  half beat taking everything? The placebo underneath it is the")
    print(f"  control -- if shuffled bins separate the book as well, the share is")
    print(f"  not doing the work. Block 3 decides whether any effect is the book")
    print(f"  or a single rule, and block 4 whether it survives day composition.")


if __name__ == "__main__":
    main()
