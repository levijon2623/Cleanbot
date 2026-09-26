# /// script
# requires-python = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_flow_divergence.py
========================
PRICE MAKES A NEW HIGH, FLOW DOES NOT. IS THE BREAKOUT FAKE?

THE HYPOTHESIS
    If price makes a new session high while cumulative net premium makes a
    LOWER high, aggressive buyers are lifting the ask into a passive seller who
    is absorbing them. The breakout has no conviction behind it and fails.
    Mirrored for new lows. This is the classic CVD-divergence read.

WHY THIS ONE IS TESTABLE WHEN UNDERLYING CVD IS NOT
    `cum_flow` is built as (ask_volume - bid_volume) x vwap x 100, cumulated
    intraday -- which IS a cumulative volume delta, on the OPTIONS tape. Unlike
    the underlying's, it is COMPLETE (check_tick_fidelity measured the Webull
    TICK stream at 42% of volume with 55% of ticks unclassified, ~17% usable and
    biased against busy minutes) and it exists for the full history. So the
    divergence claim can be measured rather than shipped on faith.

🚨 THE BASE RATE IS THE WHOLE PROBLEM
    New highs cluster in sessions that are trending up. The unconditional return
    after a new high is therefore NOT zero, and "divergent new highs are
    followed by weakness" could be true of ALL new highs. The comparison that
    means anything is CONFIRMED vs DIVERGENT, not divergent vs zero -- the same
    trap that made check_trough_anatomy's 81% recovery rate the real finding.

🚨 AND NEW HIGHS OVERLAP
    A session making new highs makes many of them, minutes apart, and their
    forward windows are the same move counted repeatedly (METHODOLOGY 2a).
    --min-gap enforces spacing; section 1 reports how much is lost.

SIGN CONVENTION
    Everything is signed TOWARD THE BREAKOUT: a new high keeps its sign, a new
    low is negated. So positive = the breakout continued, negative = it failed.
    Highs and lows then pool.

PRE-COMMITTED CRITERIA -- with a magnitude floor, because sign tests are not
criteria (check_flow_spike passed 4 of 5 on +0.34bp)
    D1  divergent breakouts have a NEGATIVE median forward return at +30m
    D2  confirmed - divergent >= 5bp at +30m
    D3  divergent is below an hour-matched placebo by >= 5bp
    D4  IS and OOS agree in sign on the D2 contrast
    D5  D2 survives --min-gap 15

Usage:
  python check_flow_divergence.py
  python check_flow_divergence.py --min-gap 15
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

import sim_core

RTH0, RTH1 = 570, 955
SPLIT = pd.Timestamp("2025-08-21").date()
HOR = [15, 30, 60]
SEED = 20260922


def bars(tk):
    df = (pl.scan_parquet(f"historical/{tk}.parquet")
          .select("date", "minute_et", "high", "low", "close")
          .collect().to_pandas())
    t = pd.to_datetime(df["minute_et"])
    df["mod"] = t.dt.hour * 60 + t.dt.minute
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["mod"] >= RTH0) & (df["mod"] <= RTH1)]
    return {d: g.sort_values("mod").set_index("mod") for d, g in df.groupby("date")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--start", type=int, default=600,
                    help="minute-of-day before which breakouts are ignored")
    ap.add_argument("--min-gap", type=int, default=0)
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    tks = a.tickers or sorted({r["ticker"] for r in
                               sim_core.research_rules(include_paper=True)})
    rng = np.random.default_rng(SEED)
    rows = []
    for tk in tks:
        f = _flow_for(D, [tk])
        if f.empty:
            continue
        g = f[f["underlying_symbol"] == tk].copy()
        ts = pd.to_datetime(g["minute_et"])
        g["date"] = ts.dt.date
        g["mod"] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)
        B = bars(tk)
        for d, x in g.groupby("date"):
            if d < sim_core.DEPLOYED_START:
                continue
            bb = B.get(d)
            if bb is None or len(bb) < 120:
                continue
            x = x.sort_values("mod")
            cum = dict(zip(x["mod"].astype(int), x["cum_flow"].astype(float)))
            mods = [int(m) for m in bb.index]
            hi = lo = None
            cmax = cmin = None
            events = []
            for m in mods:
                c = cum.get(m)
                if c is None:
                    continue
                h, l_ = float(bb.loc[m, "high"]), float(bb.loc[m, "low"])
                for side in ("high", "low"):
                    if side == "high":
                        new = hi is None or h > hi
                        conf = cmax is None or c > cmax
                    else:
                        new = lo is None or l_ < lo
                        conf = cmin is None or c < cmin
                    if new and m >= a.start and hi is not None:
                        events.append((m, side, bool(conf)))
                hi = h if hi is None else max(hi, h)
                lo = l_ if lo is None else min(lo, l_)
                cmax = c if cmax is None else max(cmax, c)
                cmin = c if cmin is None else min(cmin, c)

            emods = [e[0] for e in events]
            for k, (m, side, conf) in enumerate(events):
                nxt = [q for q in emods if q > m]
                gap = (nxt[0] - m) if nxt else 10_000
                px = float(bb.loc[m, "close"])
                rec = dict(ticker=tk, date=d, mod=m, hour=m // 60, side=side,
                           confirmed=conf, gap=gap,
                           half=("IS" if d <= SPLIT else "OOS"))
                sgn = 1.0 if side == "high" else -1.0
                for hh in HOR:
                    mh = m + hh
                    rec[f"f{hh}"] = (sgn * (float(bb.loc[mh, "close"]) - px)
                                     / px * 1e4) if mh in bb.index else np.nan
                # hour-matched placebo, same session, same sign
                pool = [q for q in mods if q // 60 == m // 60
                        and abs(q - m) >= 30 and (q + 30) in bb.index]
                if pool:
                    pk = rng.choice(pool, size=min(20, len(pool)), replace=False)
                    rec["p30"] = float(np.median(
                        [sgn * (float(bb.loc[int(q) + 30, "close"])
                                - float(bb.loc[int(q), "close"]))
                         / float(bb.loc[int(q), "close"]) * 1e4 for q in pk]))
                rows.append(rec)
        print(f"    {tk} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  no events"); return
    R.to_parquet("_flow_divergence.parquet", index=False)

    print(f"\n{'='*100}")
    print(f"  1. WHAT GOT SELECTED  (breakouts after {a.start//60:02d}:"
          f"{a.start%60:02d})")
    print(f"{'='*100}")
    print(f"  {'ticker':8} {'events':>8} {'days':>6} {'per day':>8} "
          f"{'confirmed':>10} {'med gap':>8} {'<=15m':>7}")
    for tk, gg in R.groupby("ticker", sort=False):
        nd = gg["date"].nunique()
        gp = gg["gap"][gg["gap"] < 10_000]
        print(f"  {tk:8} {len(gg):>8,} {nd:>6} {len(gg)/max(nd,1):>8.1f} "
              f"{gg['confirmed'].mean()*100:>9.1f}% "
              f"{(gp.median() if len(gp) else np.nan):>8.0f} "
              f"{(gg['gap']<=15).mean()*100:>6.0f}%")
    print(f"  {'ALL':8} {len(R):>8,} {R['date'].nunique():>6} "
          f"{'':>8} {R['confirmed'].mean()*100:>9.1f}%")

    if a.min_gap:
        b4 = len(R)
        R = R[R["gap"] > a.min_gap]
        print(f"\n  --min-gap {a.min_gap}: kept {len(R):,} of {b4:,} "
              f"({len(R)/max(b4,1)*100:.0f}%)")

    C, V = R[R["confirmed"]], R[~R["confirmed"]]
    print(f"\n{'='*100}")
    print(f"  2. FORWARD RETURN, SIGNED TOWARD THE BREAKOUT (median bp)")
    print(f"     positive = the breakout continued; negative = it failed")
    print(f"{'='*100}")
    print(f"  {'group':26} {'n':>8} " + "".join(f"{'+'+str(h)+'m':>10}"
                                                for h in HOR))
    for lab, gg in (("CONFIRMED (flow agrees)", C),
                    ("DIVERGENT (flow lags)", V)):
        print(f"  {lab:26} {len(gg):>8,} " +
              "".join(f"{gg[f'f{h}'].median():>10.2f}" for h in HOR))
    print(f"  {'placebo (hour-matched)':26} {len(R):>8,} " +
          "".join(f"{R['p30'].median():>10.2f}" if h == 30 else f"{'--':>10}"
                  for h in HOR))
    print(f"  {'-'*96}")
    print(f"  {'CONTRAST conf - div':26} {'':>8} " +
          "".join(f"{C[f'f{h}'].median() - V[f'f{h}'].median():>+10.2f}"
                  for h in HOR))

    print(f"\n{'='*100}")
    print(f"  3. D4 -- BOTH HALVES")
    print(f"{'='*100}")
    print(f"  {'group':26} {'half':>5} {'n':>8} " +
          "".join(f"{'+'+str(h)+'m':>10}" for h in HOR))
    for lab, gg in (("CONFIRMED", C), ("DIVERGENT", V)):
        for hf in ("IS", "OOS"):
            q = gg[gg["half"] == hf]
            if len(q):
                print(f"  {lab:26} {hf:>5} {len(q):>8,} " +
                      "".join(f"{q[f'f{h}'].median():>10.2f}" for h in HOR))

    print(f"\n{'='*100}")
    print(f"  4. BY SIDE  (is this symmetric, or only one direction?)")
    print(f"{'='*100}")
    print(f"  {'side':10} {'group':12} {'n':>8} " +
          "".join(f"{'+'+str(h)+'m':>10}" for h in HOR))
    for sd in ("high", "low"):
        for lab, gg in (("confirmed", C), ("divergent", V)):
            q = gg[gg["side"] == sd]
            if len(q):
                print(f"  {sd:10} {lab:12} {len(q):>8,} " +
                      "".join(f"{q[f'f{h}'].median():>10.2f}" for h in HOR))

    FLOOR = 5.0
    d30 = C["f30"].median() - V["f30"].median()
    v30 = V["f30"].median()
    p30 = R["p30"].median()
    isd = (C[C["half"] == "IS"]["f30"].median()
           - V[V["half"] == "IS"]["f30"].median())
    osd = (C[C["half"] == "OOS"]["f30"].median()
           - V[V["half"] == "OOS"]["f30"].median())
    m = lambda ok: "PASS" if ok else "FAIL"
    print(f"\n{'='*100}")
    print(f"  SCORECARD  (pre-committed, floor {FLOOR:.0f}bp)")
    print(f"{'='*100}")
    print(f"  D1  divergent +30m < 0            {v30:>+8.2f}bp   {m(v30 < 0)}")
    print(f"  D2  confirmed - divergent >= {FLOOR:.0f}bp  {d30:>+8.2f}bp   "
          f"{m(d30 >= FLOOR)}")
    print(f"  D3  divergent below placebo by {FLOOR:.0f}bp "
          f"{v30 - p30:>+8.2f}bp   {m(p30 - v30 >= FLOOR)}")
    print(f"  D4  both halves agree in sign     IS {isd:>+6.2f} / "
          f"OOS {osd:>+6.2f}   {m((isd > 0) == (osd > 0))}")
    print(f"  D5  re-run with --min-gap 15")


if __name__ == "__main__":
    main()
