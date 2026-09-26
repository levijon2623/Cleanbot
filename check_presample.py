# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "httpx>=0.27.0", "python-dotenv"]
# ///
"""
check_presample.py
==================
THE PRE-SAMPLE HOLDOUT TEST. Runs the six pre-registered tests in
PRESAMPLE_PLAN.md sections 3 exactly once, on 2023-10-12 .. 2024-08-19.

READ PRESAMPLE_PLAN.md BEFORE TOUCHING THIS FILE. That document was written and
signed before any pre-sample data was scored, and this script is its executable
form. Adding a test here, changing a threshold, or re-cutting a result after
seeing it converts the only untouched data this book will ever have into
another in-sample search. There is no more history behind 2023-10-12 -- the API
403s past it -- so this cannot be redone.

THE SIX TESTS
  T1a  core5 profitable pre-sample                CI lower bound > 0
  T1b  index-only (SPY/QQQ/IWM) profitable        CI lower bound > 0   <- clean read
  T2   a priori spread screen still helps         spread6 - all9 > 0
  T3   the four excluded rules still bad          each < 0
  T4   VIX-regime conditioning holds              high - low > 0, same sign
  T5   sampling-policy invariance holds           sequential vs screened, same sign

T1 is split because readiness check P3 found META/NVDA offer a same-day expiry
on only ~20% of pre-sample sessions (Friday-only weeklies) vs 30.2% today, so
those two rules fill a structurally different contract there. T1b is the read
that is free of that confound.

EVERYTHING IS SCORED AS DEPLOYED: sim_core sequential fills, per-rule
`policy_for` exits, fill="bot", $0.50 entry floor. No parameter is refit.
Inference is a day-block bootstrap resampling whole sessions.

Usage:  python check_presample.py            # refuses to run until 214/214
        python check_presample.py --force    # only for a documented exception
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from check_config_walkforward import (PRESAMPLE_EDGES, SLICE_EDGES,
                                      _pre_slice_idx, _slice_idx)
import sim_core

PRE_LO, PRE_HI = PRESAMPLE_EDGES[0], SLICE_EDGES[0]      # [lo, hi)
SPLIT = pd.Timestamp("2025-08-21").date()
INDEX_RULES = ["SPY CHOP CALL", "QQQ HIVOL CALL", "IWM HIVOL CALL"]
WIDE_SPREAD = ["AVGO HIVOL PUT", "GLD amp1 CALL", "SMH LOWVOL PUT"]
JUDGEMENT = ["MSFT CHOP PUT"]


# ----------------------------------------------------------------- data
def book(pop="seq"):
    """{rule: DataFrame(date, pnl, dte)} over the FULL lake, scored as deployed."""
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    out = {}
    for r in [x for x in RULES if x.get("enabled", True)]:
        meta: list[dict] = []
        # since=None is REQUIRED here and nowhere else. sim_core.DEPLOYED_START
        # otherwise filters the entire pre-sample window out, which would make
        # this script silently score nothing. This is the one sanctioned caller.
        cand = sim_core.build_candidates(D, r, meta_out=meta, since=None)
        if not cand:
            continue
        pol, em = sim_core.policy_for(r, TRAIL_PCT), sim_core.eod_mod(r)
        if pop == "seq":
            picks: list[tuple] = []
            rows = sim_core.walk(cand, pol, em, fill="bot", picks_out=picks)
            dte = [meta[ci]["dte"] for ci, _ in picks]
        else:
            rows = [(d, sim_core.simulate(p, pol, em, fill="bot")[0])
                    for d, m, p in cand]
            dte = [m["dte"] for m in meta]
        out[r["name"]] = pd.DataFrame(
            {"date": [d for d, _ in rows], "pnl": [p for _, p in rows], "dte": dte})
    return out


def _win(df, lo, hi):
    return df[(df["date"] >= lo) & (df["date"] < hi)]


def rows_for(bk, keep, lo, hi):
    fr = [_win(v, lo, hi) for k, v in bk.items() if k in keep]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=["date", "pnl", "dte"])


# ----------------------------------------------------------------- stats
def boot(a: pd.DataFrame, b: pd.DataFrame | None = None, n=4000, seed=91):
    """Day-block bootstrap. mean(a) if b is None, else mean(a)-mean(b).

    Whole SESSIONS are resampled, and when comparing two books the SAME day
    draw is applied to both -- they share most of their calendar, so treating
    them as independent would overstate the precision of the difference.
    """
    if a.empty or (b is not None and b.empty):
        return (np.nan, np.nan, np.nan)
    A = {d: g["pnl"].to_numpy() for d, g in a.groupby("date")}
    B = {d: g["pnl"].to_numpy() for d, g in b.groupby("date")} if b is not None else None
    days = sorted(set(A) | (set(B) if B else set()))
    obs = a["pnl"].mean() - (b["pnl"].mean() if b is not None else 0.0)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = [days[k] for k in rng.choice(len(days), len(days), replace=True)]
        xa = np.concatenate([A[d] for d in pick if d in A]) if any(d in A for d in pick) else None
        if xa is None or not len(xa):
            continue
        if B is None:
            out.append(xa.mean())
        else:
            xb = np.concatenate([B[d] for d in pick if d in B]) if any(d in B for d in pick) else None
            if xb is not None and len(xb):
                out.append(xa.mean() - xb.mean())
    if not out:
        return (obs, np.nan, np.nan)
    return (obs, *np.percentile(out, [2.5, 97.5]))


def line(lbl, df, width=30):
    if df.empty:
        print(f"  {lbl:{width}} (no trades)")
        return
    v = df["pnl"].to_numpy()
    sl = {}
    for d in df["date"]:
        k = _pre_slice_idx(d)
        if k is not None:
            sl[k] = sl.get(k, 0) + 1
    cov = " ".join(f"P{k+1}={sl.get(k,0)}" for k in range(3))
    print(f"  {lbl:{width}} n={len(v):>4}  mean {v.mean()*100:>+7.1f}%  "
          f"win {(v>0).mean():.2f}  tot {v.sum():>+7.2f}  days {df['date'].nunique():>3}  [{cov}]")


# ----------------------------------------------------------------- vix
def vix_frame():
    import httpx
    from dotenv import load_dotenv
    load_dotenv(encoding="utf-8-sig")
    h = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json"}
    r = httpx.get("https://api.unusualwhales.com/api/stock/VIX/volatility/realized",
                  headers=h, params={"timeframe": "5Y"}, timeout=30)
    rows = r.json().get("data", [])
    s = pd.Series({pd.Timestamp(x["date"]).date(): float(x["price"])
                   for x in rows if x.get("price")}).sort_index()
    return s


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from uw_options_data_lake import trading_days, silver_partition_path, DEFAULT_LAKE
    days = trading_days(PRE_LO, PRE_HI - pd.Timedelta(days=1).to_pytimedelta())
    have = sum(1 for d in days if silver_partition_path(DEFAULT_LAKE, d).exists())
    print(f"PRE-SAMPLE HOLDOUT  {PRE_LO} .. {PRE_HI - pd.Timedelta(days=1).to_pytimedelta()}")
    print(f"  silver partitions {have}/{len(days)}")
    if have < len(days) and not a.force:
        print("\n  REFUSING TO SCORE a partially-backfilled window (PRESAMPLE_PLAN 7).")
        print("  Scoring a subset spends the holdout on an arbitrary slice of it.")
        print("  Wait for the backfill, or pass --force and document why.")
        return

    # THE DERIVED CACHES MUST COVER THE WINDOW TOO.
    # The silver lake being complete is NOT sufficient. `_ticker_bars` reads the
    # prebuilt `opt_bars_atm.parquet`, not the lake, and the first run of this
    # script silently produced ZERO pre-sample trades for all nine rules --
    # which printed as "T1a FAIL" rather than as an error. A plumbing failure
    # that renders as a verdict is the worst possible outcome on a one-shot
    # holdout, so it is now a hard stop.
    import pandas as _pd
    bars = "opt_bars_atm.parquet"
    if not os.path.exists(bars):
        print(f"\n  ABORT: {bars} missing. Run: "
              f"python directional_flow_backtester.py --build-bars")
        return
    bmin = _pd.read_parquet(bars, columns=["date"])["date"].min()
    if bmin > PRE_LO:
        print(f"\n  ABORT: {bars} starts {bmin}, after the pre-sample window "
              f"opens {PRE_LO}.")
        print(f"  It was built before the backfill and would yield ZERO "
              f"pre-sample trades,")
        print(f"  which would print as a FAIL rather than an error. Rebuild:")
        print(f"      python directional_flow_backtester.py --build-bars")
        return
    print(f"  bars cache covers from {bmin}  [ok]")
    print()

    bk = book("seq")
    scr = book("screened")
    allr = set(bk)
    core5 = allr - set(WIDE_SPREAD) - set(JUDGEMENT)
    spread6 = allr - set(WIDE_SPREAD)

    # --- context first: what does this window look like?
    print("=" * 100)
    print("  CONTEXT -- read this BEFORE the verdicts (PRESAMPLE_PLAN 5)")
    print("=" * 100)
    vix = vix_frame()
    for lbl, lo, hi in (("pre-sample", PRE_LO, PRE_HI),
                        ("in-sample ", SLICE_EDGES[0], SPLIT),
                        ("out-of-sam", SPLIT, SLICE_EDGES[-1])):
        w = vix[(vix.index >= lo) & (vix.index < hi)]
        if len(w):
            print(f"  VIX {lbl}: median {w.median():5.1f}   "
                  f"share >=18 {100*(w >= 18).mean():5.1f}%   n={len(w)}")
    print("  (deployed overlay: VIX>=18 earns +24%/trade vs +7% below, so a LOW-VIX")
    print("   window is EXPECTED to read weaker. That is not a refutation.)\n")

    print("  0DTE share of fills, pre-sample vs in/out-of-sample:")
    for r in sorted(bk):
        p, c = _win(bk[r], PRE_LO, PRE_HI), _win(bk[r], SLICE_EDGES[0], SLICE_EDGES[-1])
        fp = f"{100*(p['dte']==0).mean():.0f}%" if len(p) else "--"
        fc = f"{100*(c['dte']==0).mean():.0f}%" if len(c) else "--"
        flag = ""
        if len(p) and len(c) and abs((p["dte"] == 0).mean() - (c["dte"] == 0).mean()) > 0.15:
            flag = "   <- STRUCTURAL SHIFT"
        print(f"    {r:22} pre {fp:>5}  (n={len(p):>3})   now {fc:>5}  (n={len(c):>4}){flag}")

    # --- the tests
    print("\n" + "=" * 100)
    print("  THE SIX PRE-REGISTERED TESTS")
    print("=" * 100)
    verdict = {}

    pre_core = rows_for(bk, core5, PRE_LO, PRE_HI)
    pre_idx = rows_for(bk, INDEX_RULES, PRE_LO, PRE_HI)
    # IWM only gained full daily expiries at ~2024-04-18 (the P2/P3 boundary),
    # so 30 P2 sessions carry a different contract structure. P3 is the window
    # where all three index rules match today exactly. SPY+QQQ are 100% in every
    # slice and are the third, structurally-clean-throughout cut. See
    # PRESAMPLE_PLAN.md, "Readiness finding: it is IWM, not META/NVDA".
    P3_LO = PRESAMPLE_EDGES[2]
    pre_idx_p3 = rows_for(bk, INDEX_RULES, P3_LO, PRE_HI)
    pre_sq = rows_for(bk, ["SPY CHOP CALL", "QQQ HIVOL CALL"], PRE_LO, PRE_HI)
    for tag, lbl, df in (("T1a", "core5 (5 rules)", pre_core),
                         ("T1b", "index-only SPY/QQQ/IWM", pre_idx),
                         ("T1b-P3", "index-only, P3 only (structure matches today)", pre_idx_p3),
                         ("T1b-SQ", "SPY+QQQ only (100% comparable throughout)", pre_sq)):
        o, c1, c2 = boot(df)
        ok = np.isfinite(c1) and c1 > 0
        verdict[tag] = ok
        ctx = (core5 if tag == "T1a" else
               ["SPY CHOP CALL", "QQQ HIVOL CALL"] if tag == "T1b-SQ" else INDEX_RULES)
        print(f"\n  {tag}  {lbl}")
        line("      pre-sample", df)
        line("      in-sample (context)", rows_for(bk, ctx, SLICE_EDGES[0], SPLIT))
        line("      OOS (context)", rows_for(bk, ctx, SPLIT, SLICE_EDGES[-1]))
        print(f"      mean {o*100:+.1f}%  day-block 95% CI [{c1*100:+.1f}, {c2*100:+.1f}]"
              f"   -> {'PASS' if ok else 'FAIL'}")

    o, c1, c2 = boot(rows_for(bk, spread6, PRE_LO, PRE_HI), rows_for(bk, allr, PRE_LO, PRE_HI))
    verdict["T2"] = np.isfinite(o) and o > 0
    print(f"\n  T2  spread6 - all9 = {o*100:+.1f}pp  CI [{c1*100:+.1f}, {c2*100:+.1f}]"
          f"   -> {'PASS' if verdict['T2'] else 'FAIL'}")

    print("\n  T3  the four excluded rules, pre-sample")
    t3 = True
    for r in WIDE_SPREAD + JUDGEMENT:
        d = _win(bk.get(r, pd.DataFrame(columns=["date", "pnl", "dte"])), PRE_LO, PRE_HI)
        m = d["pnl"].mean() if len(d) else np.nan
        bad = np.isfinite(m) and m < 0
        t3 &= bool(bad) or not np.isfinite(m)
        line(f"      {r}", d)
        print(f"      -> {'still negative' if bad else 'POSITIVE pre-sample' if np.isfinite(m) else 'no trades'}")
    verdict["T3"] = t3

    pv = vix.reindex(sorted(set(pre_core["date"]))).shift(1) if len(pre_core) else pd.Series(dtype=float)
    if len(pre_core) and pv.notna().any():
        med = vix[(vix.index >= PRE_LO) & (vix.index < PRE_HI)].median()
        hi_d = {d for d in pre_core["date"] if pd.notna(vix.get(d, np.nan)) and vix.get(d) >= med}
        hi = pre_core[pre_core["date"].isin(hi_d)]
        lo = pre_core[~pre_core["date"].isin(hi_d)]
        o, c1, c2 = boot(hi, lo)
        verdict["T4"] = np.isfinite(o) and o > 0
        print(f"\n  T4  VIX>=window-median minus below, pre-sample")
        line("      high VIX", hi)
        line("      low VIX", lo)
        print(f"      diff {o*100:+.1f}pp  CI [{c1*100:+.1f}, {c2*100:+.1f}]"
              f"   -> {'PASS' if verdict['T4'] else 'FAIL'}")
    else:
        verdict["T4"] = None
        print("\n  T4  no VIX overlap -- SKIPPED")

    s_seq = rows_for(bk, core5, PRE_LO, PRE_HI)["pnl"].mean()
    s_scr = rows_for(scr, core5, PRE_LO, PRE_HI)["pnl"].mean()
    verdict["T5"] = np.isfinite(s_seq) and np.isfinite(s_scr) and np.sign(s_seq) == np.sign(s_scr)
    print(f"\n  T5  sequential {s_seq*100:+.1f}%  vs as-screened {s_scr*100:+.1f}%"
          f"   -> {'PASS (same sign)' if verdict['T5'] else 'FAIL (sign differs)'}")

    print("\n" + "=" * 100)
    print("  VERDICTS  " + "   ".join(
        f"{k}={'PASS' if v else 'skip' if v is None else 'FAIL'}" for k, v in verdict.items()))
    print("  6 tests at 95% -> ~0.3 false positives expected. T1a/T1b are nested "
          "and count as ONE finding.")
    print("  Where T1a and T1b disagree, T1b governs the conclusion about the EDGE")
    print("  and T1a the conclusion about the book AS CONFIGURED (PRESAMPLE_PLAN 3).")
    print("=" * 100)


if __name__ == "__main__":
    main()
