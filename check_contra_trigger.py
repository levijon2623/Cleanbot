# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_contra_trigger.py
=======================
THE CONTRARIAN BULL-SHARE SIGNAL AS AN ACTUAL TRIGGER, SCORED ON P&L.

WHAT IS BEING TESTED
    check_run_antecedent found, on SPY, that inside high-premium minutes the
    BULLISH SHARE of aggressive option premium points the wrong way, monotonically
    across five buckets:

        bull share  37%  ->  up-down  +2.15pp
                    45%  ->           +3.18pp
                    50%  ->           -1.22pp
                    55%  ->           -2.52pp
                    63%  ->           -3.79pp

    Heavy call buying precedes DOWN moves. It survived a trailing-move control
    (5/5 buckets) and an activity control (spearman with premium +0.030). So:
        premium hot AND bull share LOW   -> buy CALLS
        premium hot AND bull share HIGH  -> buy PUTS

🚨 PREDICTING A MOVE IS NOT THE SAME AS MAKING MONEY, AND THIS IS WHERE THE
    LAST EIGHT FEATURES DIED. Every one of them sorted outcomes plausibly and
    then failed on P&L, because whatever marks losers marks winners too
    (spearman(loss50, hit50) = +0.664). Run onset says nothing about whether a
    50% trail can monetise the move. Only simulate() can answer that.

🚨 SPY IS THE DISCOVERY SET AND IS REPORTED SEPARATELY.
    The pattern was found on SPY. Scoring it on SPY and calling that evidence is
    circular. All nine cached tickers are run; SPY is quarantined into its own
    line and the verdict rests on the EIGHT it was never fitted to. That is a
    genuine cross-sectional holdout, not a re-slice of the same data.

GATES ARE DELIBERATELY REMOVED
    The deployed SPY rule is CHOP-only with amt_open, flow_zscore and
    min_flow_pct on top -- and SPY is CHOP on just 18 of the 151 cached days.
    Testing through it would measure the gates, not the signal, on a sample too
    small to read. So every ticker runs a PERMISSIVE rule (hours 9-14, dte 0/1,
    flow_abs 0 so the matcher's threshold is a no-op) and the contract pick,
    the $0.50 entry floor and the exit all stay exactly as deployed.

PRE-COMMITTED CRITERIA (METHODOLOGY 4) -- fixed before the first run
    C1  day-level IS median ROE > 0 on the 8 held-out tickers
    C2  ... OOS total > 0
    C3  >= 5 of 6 calendar slices populated AND positive
    C4  OOS beats the p95 of the SHUFFLED-SHARE placebo, matched on n
    C5  >= 5 of the 8 held-out tickers have positive median ROE
    C6  beats the deployed whale-spike trigger run through the SAME permissive
        rule on the SAME days -- otherwise it is a worse way to do what the
        book already does

    C5 and C6 are the ones this is most likely to fail. Neither is negotiable
    afterwards.

Usage:
  python check_contra_trigger.py
  python check_contra_trigger.py --prem-q 0.90 --share-q 0.20 --cooldown 30
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
TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "META", "AVGO", "SMH", "GLD", "MSFT"]
SPLIT = pd.Timestamp("2025-08-21").date()


def load30():
    f = sorted(glob.glob(os.path.join(CACHE, "date=*", "flow30.parquet")))
    if not f:
        raise SystemExit(f"  no 30s buckets in {CACHE}")
    cols = ("ca", "cb", "pa", "pb")
    parts = [pl.read_parquet(p).with_columns(
        [pl.col(c).cast(pl.Float64) for c in cols]) for p in f]
    df = pl.concat(parts).to_pandas()
    df["bull"] = df["ca"] + df["pb"]
    df["bear"] = df["cb"] + df["pa"]
    g = (df.groupby(["underlying_symbol", "date", "mod"])[["bull", "bear"]]
           .sum().reset_index())
    g["date"] = pd.to_datetime(g["date"]).dt.date
    return g


def make_trigs(g, tk, prem_q, share_q, cooldown, rng=None):
    """Contrarian triggers in D.triggers_for's schema. Causal throughout.

    Thresholds are TRAILING quantiles with the current minute excluded
    (shift(1)), so a minute is never compared against a distribution it is
    itself in -- the look-ahead that inverted the 'fading the bull' gradient
    when it was cut at the sample median instead.
    """
    out = []
    for d, x in g[g["underlying_symbol"] == tk].groupby("date"):
        x = x.sort_values("mod").copy()
        prem = x["bull"] + x["bear"]
        share = (x["bull"] / prem.replace(0, np.nan))
        if rng is not None:                      # PLACEBO: break the pairing
            share = pd.Series(share.to_numpy()[rng.permutation(len(share))],
                              index=share.index)
        pt = prem.rolling(60, min_periods=20).quantile(prem_q).shift(1)
        lo = share.rolling(60, min_periods=20).quantile(share_q).shift(1)
        hi = share.rolling(60, min_periods=20).quantile(1 - share_q).shift(1)
        hot = prem >= pt
        last = {"CALL": -10**9, "PUT": -10**9}
        for m, h, s, l, u, pv in zip(x["mod"], hot, share, lo, hi, prem):
            if not h or not np.isfinite(s):
                continue
            dirn = ("CALL" if (np.isfinite(l) and s <= l)
                    else "PUT" if (np.isfinite(u) and s >= u) else None)
            if dirn is None or m - last[dirn] < cooldown:
                continue
            last[dirn] = m
            ts = pd.Timestamp(d) + pd.Timedelta(minutes=int(m))
            out.append({"date": d, "ts": ts, "hour": int(m // 60),
                        "dir": dirn, "abs_flow": float(pv)})
    return out


def permissive(tk, dirn):
    return dict(name=f"{tk} CONTRA {dirn}", ticker=tk, direction=dirn,
                hours=[9, 10, 11, 12, 13, 14], dte=[0, 1],
                flow_abs=0.0, trail_pct=0.50, enabled=True)


def run(D, tk, dirn, trigs, days):
    rule = permissive(tk, dirn)
    t = [x for x in trigs if x["dir"] == dirn]
    if not t:
        return []
    try:
        cand = sim_core.build_candidates(D, rule, trigs=t)
    except Exception as e:
        print(f"    {tk} {dirn}: {type(e).__name__}: {e}")
        return []
    pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
    cap = sim_core.CUSHION_CAP.get(tk)
    rows = []
    for d, m, path in cand:
        if d not in days:
            continue
        pnl, xm, tag = sim_core.simulate(path, pol, eod, fill="botcap",
                                         cush_cap=cap)
        rows.append(dict(ticker=tk, dir=dirn, date=d, pnl=pnl * 100, tag=tag))
    return rows


def agg(rows, label):
    if not rows:
        return f"  {label:24} {'(none)':>7}"
    v = np.array([r["pnl"] for r in rows], float)
    i = [r["pnl"] for r in rows if r["date"] < SPLIT]
    o = [r["pnl"] for r in rows if r["date"] >= SPLIT]
    dset = {r["date"] for r in rows}
    return (f"  {label:24} {len(v):>7} {len(dset):>6} "
            f"{(v <= -50).mean()*100:>7.1f}% {(v > 0).mean()*100:>6.1f}% "
            f"{np.median(v):>+9.1f} {(np.sum(i) if i else 0):>+10.0f} "
            f"{(np.sum(o) if o else 0):>+10.0f}")


HDR = (f"  {'group':24} {'n':>7} {'days':>6} {'loss50':>8} {'win':>7} "
       f"{'medROE':>9} {'IS':>10} {'OOS':>10}")


def day_med(rows):
    if not rows:
        return np.nan
    d = {}
    for r in rows:
        d.setdefault(r["date"], []).append(r["pnl"])
    return float(np.median([np.median(v) for v in d.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prem-q", type=float, default=0.90)
    ap.add_argument("--share-q", type=float, default=0.20)
    ap.add_argument("--cooldown", type=int, default=30)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D

    g = load30()
    days = set(g["date"].unique())
    print(f"  30s cache: {len(days)} sessions, {g['underlying_symbol'].nunique()} tickers")
    print(f"  trigger: premium >= trailing p{a.prem_q*100:.0f}, bull share in the "
          f"extreme {a.share_q*100:.0f}% tail, {a.cooldown}m cooldown\n")

    real, plac, base = [], [], []
    for tk in TICKERS:
        t = make_trigs(g, tk, a.prem_q, a.share_q, a.cooldown)
        p = make_trigs(g, tk, a.prem_q, a.share_q, a.cooldown, rng=rng)
        for dirn in ("CALL", "PUT"):
            real += run(D, tk, dirn, t, days)
            plac += run(D, tk, dirn, p, days)
        # BASELINE: the deployed whale-spike trigger, same permissive rule,
        # same days. C6 is against this, not against zero.
        try:
            from check_config_walkforward import _flow_for
            fl = _flow_for(D, [tk])
            bt = D.triggers_for(fl, tk)
            for dirn in ("CALL", "PUT"):
                base += run(D, tk, dirn, bt, days)
        except Exception as e:
            print(f"    {tk} baseline: {type(e).__name__}")
        print(f"    {tk} done", flush=True)

    R = pd.DataFrame(real)
    if R.empty:
        print("  no candidates survived"); return
    R.to_parquet("_contra_trigger.parquet", index=False)
    held = R[R["ticker"] != "SPY"].to_dict("records")
    spy = R[R["ticker"] == "SPY"].to_dict("records")
    P = [r for r in plac if r["ticker"] != "SPY"]
    B = [r for r in base if r["ticker"] != "SPY"]

    print(f"\n{'='*104}")
    print(f"  1. HEADLINE -- the 8 held-out tickers are the verdict")
    print(f"{'='*104}")
    print(HDR)
    print(agg(held, "CONTRA (8 held-out)"))
    print(agg(B, "  baseline whale spike"))
    print(agg(P, "  placebo (share shuffled)"))
    print(agg(spy, "SPY (discovery -- circular)"))

    print(f"\n{'='*104}")
    print(f"  2. BY DIRECTION  (the signal is contrarian; both sides should work)")
    print(f"{'='*104}")
    print(HDR)
    for dirn in ("CALL", "PUT"):
        print(agg([r for r in held if r["dir"] == dirn], f"held-out {dirn}"))

    print(f"\n{'='*104}")
    print(f"  3. C5 -- PER TICKER (>= 5 of 8 held-out positive)")
    print(f"{'='*104}")
    print(f"  {'ticker':8} {'n':>6} {'days':>6} {'loss50':>8} {'medROE':>9} {'OOS':>10}")
    c5 = 0
    for tk in TICKERS:
        s = [r for r in held if r["ticker"] == tk]
        if not s:
            continue
        v = np.array([r["pnl"] for r in s], float)
        o = [r["pnl"] for r in s if r["date"] >= SPLIT]
        c5 += int(np.median(v) > 0)
        print(f"  {tk:8} {len(v):>6} {len({r['date'] for r in s}):>6} "
              f"{(v<=-50).mean()*100:>7.1f}% {np.median(v):>+9.1f} "
              f"{(np.sum(o) if o else 0):>+10.0f}")
    n_tk = len({r['ticker'] for r in held})
    print(f"\n  C5: {c5}/{n_tk} held-out tickers with positive median ROE")

    print(f"\n{'='*104}")
    print(f"  4. C3 -- CALENDAR SLICES")
    print(f"{'='*104}")
    H = pd.DataFrame(held)
    H["q"] = pd.PeriodIndex(pd.to_datetime(H["date"]), freq="Q")
    qs = sorted(H["q"].unique())[-6:]
    pos = 0
    print(f"  {'slice':10} {'n':>6} {'days':>6} {'total':>10} {'medROE':>9}")
    for q in qs:
        s = H[H["q"] == q]
        tot = s["pnl"].sum()
        pos += int(tot > 0)
        print(f"  {str(q):10} {len(s):>6} {s['date'].nunique():>6} "
              f"{tot:>+10.0f} {s['pnl'].median():>+9.1f}")
    print(f"\n  C3: {pos}/{len(qs)} slices positive")

    print(f"\n{'='*104}")
    print(f"  5. C4 -- OOS vs THE SHUFFLED-SHARE PLACEBO")
    print(f"{'='*104}")
    ro = [r["pnl"] for r in held if r["date"] >= SPLIT]
    po = [r["pnl"] for r in P if r["date"] >= SPLIT]
    obs, n = float(np.sum(ro)), len(ro)
    if n and len(po) > n:
        draws = np.array([np.sum(rng.choice(po, n, replace=False))
                          for _ in range(a.boot)])
        p95 = float(np.percentile(draws, 95))
        print(f"  CONTRA OOS {obs:>+10.0f} on n={n}")
        print(f"  placebo    p50 {np.percentile(draws,50):>+10.0f}   p95 {p95:>+10.0f}")
        print(f"  C4: {'PASS' if obs > p95 else 'FAIL'} "
              f"(percentile {(draws < obs).mean()*100:.1f})")
    else:
        print(f"  too few for a bootstrap (real n={n}, placebo n={len(po)})")

    print(f"\n{'='*104}")
    print(f"  SCORECARD  (held-out 8 only)")
    print(f"{'='*104}")
    dm = day_med([r for r in held if r["date"] < SPLIT])
    oos = np.sum([r["pnl"] for r in held if r["date"] >= SPLIT]) if held else 0
    bo = np.sum([r["pnl"] for r in B if r["date"] >= SPLIT]) if B else 0
    print(f"  C1 day-level IS median ROE > 0   {dm:>+9.1f}  "
          f"{'PASS' if dm > 0 else 'FAIL'}")
    print(f"  C2 OOS total > 0                 {oos:>+9.0f}  "
          f"{'PASS' if oos > 0 else 'FAIL'}")
    print(f"  C6 beats baseline whale spike    {oos:>+9.0f} vs {bo:>+9.0f}  "
          f"{'PASS' if oos > bo else 'FAIL'}")
    print(f"  C3/C4/C5 -- see blocks above")


if __name__ == "__main__":
    main()
