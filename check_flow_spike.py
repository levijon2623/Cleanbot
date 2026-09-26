# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_flow_spike.py
===================
DOES A COUNTER-DIRECTIONAL FLOW SPIKE MARK A LOCAL EXTREME?

THE OBSERVATION (live, 2026-09-21, SPY)
    Cumulative net premium jumped ~3.3M in one minute -- 6x the median minute --
    WHILE PRICE WAS STILL FALLING. Price undershot ~7bp further, reclaimed the
    spike level, and continued. A discretionary long off that bottom made +35%.

    One trade proves nothing. What makes it worth measuring is that the bot
    COULD NOT HAVE SEEN IT: SPY's cum flow sat at 47% of its p90 gate at the
    time, comfortably refused, and the spike itself was only 5.1% of that gate.

🚨 THE GATE IS A LEVEL. THIS IS A CHANGE. THEY ARE DIFFERENT STATISTICS.
    `min_flow_pct` asks "has enough conviction accumulated TODAY"
    (|cum_flow| >= percentile). This asks "did a lot of conviction arrive RIGHT
    NOW" (|Δcum_flow| >> normal). A spike can be 6x the median minute while the
    level is nowhere near p90. check_flow_threshold established the LEVEL is
    load-bearing in both halves; nothing has tested the CHANGE on its own.

WHAT IS ALREADY KNOWN, SO THIS IS NOT RE-RUN BLIND
    check_flow_accel     sub-minute acceleration AT the trigger        null
    check_trough_anatomy flow not confirming a new adverse extreme     0.500
    check_flow_threshold the LEVEL gate                                load-bearing
    Both nulls were conditioned on a crossover that had ALREADY fired and a
    level gate that had ALREADY passed. This is a standalone event, on days the
    gate never opens.

🚨 FOUR TRAPS, EACH WITH ITS CONTROL
    1  LOOK-AHEAD IN THE THRESHOLD. "6x the median minute" is only meaningful if
       the median is trailing and EXCLUDES the current minute. Defining the
       spike off a whole-day or whole-sample statistic is the look-ahead that
       inverted "fading the bull" (METHODOLOGY 7).
    2  IT MIGHT JUST BE SPIKES, NOT COUNTER-DIRECTIONAL ONES. The decisive
       control is SAME-direction spikes of identical size. If both reverse, the
       counter-ness carries nothing and this is just "big flow precedes moves".
    3  IT MIGHT JUST BE THE CLOCK, OR VOLATILITY. A placebo drawn from the SAME
       SESSION and SAME HOUR, >=30m clear, signed the same way, says what price
       does in any comparable window.
    4  CLUSTERED SPIKES DOUBLE-COUNT. Spikes arrive in bursts, and overlapping
       forward windows measure the same move repeatedly (METHODOLOGY 2a, and
       the spacing problem check_arrival had to add --min-gap for). Section 1
       reports spacing; --min-gap enforces it.

SAMPLE
    2024-08-20 onward (sim_core.DEPLOYED_START), IS/OOS split 2025-08-21.
    The 2023-10-12..2024-08-19 PRE-SAMPLE HOLDOUT is deliberately NOT touched --
    PRESAMPLE_PLAN.md governs it and it is not mine to spend. Note this runs on
    EVERY trading day, not the 151 candidate days, so power is far better than
    most studies here.

PRE-COMMITTED CRITERIA -- fixed before the first run
    S1  counter-directional spikes: positive median forward return toward the
        spike at +15m
    S2  counter beats SAME-direction spikes (the contrast that matters)
    S3  counter beats the hour-matched placebo
    S4  the sign holds in BOTH the IS and OOS halves
    S5  it survives --min-gap 30 (no double-counted clusters)

RESULT -- 2026-09-21. NULL, and unusually well powered: 93,837 spikes over 503
sessions and 9 tickers. Power is not the excuse here.

    Spikes are COMMON, not rare: 21 per ticker per day at 6x the median minute,
    and 70-89% are followed by another within 30 minutes.

    Forward return, signed toward the spike (median bp):
                            +5m    +15m    +30m    +60m
        COUNTER            0.31    0.34    0.28    0.18
        SAME              -0.14   -0.38   -0.41   -0.42
        placebo             --     0.30     --      --
    COUNTER BARELY BEATS THE HOUR-MATCHED PLACEBO -- +0.34 against +0.30, a
    margin of 0.04bp. S3 "passes" on noise. With --min-gap 30 (19% kept) it is
    +0.41 against +0.29. A counter-directional spike is, to measurement, an
    ordinary minute.

    🚨 AND IN THE STRATUM THAT ACTUALLY MATTERS IT REVERSES. The live IWM event
    was 58.6x the median minute. In the 50x+ bucket, counter-directional is
    NEGATIVE: -0.24 at +15m pooled, -0.36 with the gap filter, -1.32 at +30m.
    The hypothesis fails hardest exactly where the observation came from.
    No monotonic relationship with magnitude either -- 6-10x +0.56, 10-20x
    0.00, 20-50x +0.40, 50x+ -0.24. That is what 32 cells of noise looks like.

    IWM is not special: +0.38 / +0.23 / 0.00 / -0.11, decaying to nothing.

    THE ONE CONSISTENT THING IS THE OTHER SIDE. SAME-direction spikes -- flow
    chasing a move already underway -- are NEGATIVE at every horizon, in BOTH
    halves, and the effect STRENGTHENS when clustering is removed (-1.24 at
    +15m, -1.93 at +30m with --min-gap 30; IS -0.85/-1.68, OOS -1.69/-2.36).
    Chasing is punished. It is still ~2bp, which does not pay for a round trip,
    but it is the only cell in the table that behaves like a signal.

    OPERATIONALLY: overshoot median 19.8bp, p90 73.1bp. The live example's ~7bp
    was SHALLOWER than typical, not typical -- a SPY-only first pass said 10bp
    and that understated it.

    CONSEQUENCE FOR THE BOOK: the level gate is not leaving anything on the
    table by ignoring the change. `min_flow_pct` can stay a level.

    🚨 THE PRE-COMMITTED CRITERIA PASSED 4 OF 5 ON THIS. S1-S4 were SIGN tests
    with no magnitude floor, so a +0.34bp result -- one cent on a $285 name --
    cleared them. S1b (a 5bp floor) was added after the first run and is the
    only one that fails. Sign tests are not criteria; see METHODOLOGY 7.

Usage:
  python check_flow_spike.py
  python check_flow_spike.py --k 4 --min-gap 30
  python check_flow_spike.py --tickers SPY QQQ IWM
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import check_reentry as RE          # verified RTH bar loader
import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
HORIZONS = [5, 15, 30, 60]
SEED = 20260921


def bp(a, b):
    return (a - b) / b * 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--k", type=float, default=6.0,
                    help="spike = |d1| >= k x trailing median |d1|")
    ap.add_argument("--win", type=int, default=30,
                    help="trailing window for the median, current minute EXCLUDED")
    ap.add_argument("--pdir", type=int, default=5,
                    help="minutes of price used to define the prevailing direction")
    ap.add_argument("--min-gap", type=int, default=0,
                    help="drop a spike if another lands within N minutes after it")
    ap.add_argument("--min-abs", type=float, default=2.5e5,
                    help="floor on |d1| so a quiet denominator cannot manufacture a spike")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = sim_core.research_rules(include_paper=True)
    tks = sorted({r["ticker"] for r in rules})
    if a.tickers:
        tks = [t for t in a.tickers if t in tks] or a.tickers

    rows = []
    rng = np.random.default_rng(SEED)
    for tk in tks:
        f = _flow_for(D, [tk])
        if f.empty:
            continue
        g = f[f["underlying_symbol"] == tk].copy()
        ts = pd.to_datetime(g["minute_et"])
        g["date"] = ts.dt.date
        g["mod"] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)
        B = RE.bars(tk)

        for d, x in g.groupby("date"):
            if d < sim_core.DEPLOYED_START:
                continue                      # pre-sample holdout, untouched
            bars = B.get(d)
            if bars is None or len(bars) < 60:
                continue
            x = x.sort_values("mod")
            cum = x["cum_flow"].to_numpy(float)
            mods = x["mod"].to_numpy(int)
            if len(cum) < a.win + 20:
                continue
            d1 = np.diff(cum, prepend=cum[0])
            # trailing median |d1|, CURRENT MINUTE EXCLUDED -- trap 1
            s = pd.Series(np.abs(d1))
            trail = s.rolling(a.win, min_periods=10).median().shift(1).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.abs(d1) / trail
            hit = np.where(np.isfinite(ratio) & (ratio >= a.k)
                           & (np.abs(d1) >= a.min_abs))[0]
            spikes = [int(mods[i]) for i in hit]

            for i in hit:
                m = int(mods[i])
                if m not in bars.index:
                    continue
                sdir = 1 if d1[i] > 0 else -1
                m0 = m - a.pdir
                if m0 not in bars.index:
                    continue
                p_now = float(bars.loc[m, "close"])
                p_then = float(bars.loc[m0, "close"])
                if p_now == p_then:
                    continue
                pdir = 1 if p_now > p_then else -1
                nxt = [q for q in spikes if q > m]
                rec = dict(ticker=tk, date=d, mod=m, hour=m // 60,
                           sdir=sdir, counter=bool(pdir != sdir),
                           ratio=float(ratio[i]), d1=float(d1[i]),
                           cum=float(cum[i]),
                           gap=(nxt[0] - m) if nxt else 10_000,
                           half=("IS" if d <= SPLIT else "OOS"))
                # forward return, signed TOWARD the spike direction
                for h in HORIZONS:
                    mh = m + h
                    rec[f"f{h}"] = (sdir * bp(float(bars.loc[mh, "close"]), p_now)
                                    if mh in bars.index else np.nan)
                # overshoot: furthest price runs AGAINST the spike before +60
                w = [q for q in range(m + 1, m + 61) if q in bars.index]
                if w:
                    ext = (min(float(bars.loc[q, "low"]) for q in w) if sdir > 0
                           else max(float(bars.loc[q, "high"]) for q in w))
                    rec["overshoot"] = -sdir * bp(ext, p_now)
                    lvl = [q for q in w
                           if (float(bars.loc[q, "high"]) >= p_now if sdir > 0
                               else float(bars.loc[q, "low"]) <= p_now)]
                    rec["reclaim_min"] = (lvl[0] - m) if lvl else np.nan
                # PLACEBO: same session, same hour, >=30m clear, same sign
                pool = [int(q) for q in bars.index
                        if q // 60 == m // 60 and abs(q - m) >= 30
                        and (q + 15) in bars.index]
                if pool:
                    pk = rng.choice(pool, size=min(20, len(pool)), replace=False)
                    rec["p15"] = float(np.median(
                        [sdir * bp(float(bars.loc[int(q) + 15, "close"]),
                                   float(bars.loc[int(q), "close"])) for q in pk]))
                rows.append(rec)
        print(f"    {tk} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  no spikes found"); return
    R.to_parquet("_flow_spike.parquet", index=False)

    print(f"\n{'='*104}")
    print(f"  1. WHAT GOT SELECTED  (spike = |Δcum| >= {a.k}x trailing "
          f"{a.win}m median, floor ${a.min_abs/1e3:.0f}k)")
    print(f"{'='*104}")
    print(f"  {'ticker':8} {'spikes':>8} {'days':>6} {'per day':>8} "
          f"{'counter':>9} {'med gap':>9} {'<=30m':>7} {'med ratio':>10}")
    for tk, gg in R.groupby("ticker", sort=False):
        nd = gg["date"].nunique()
        gp = gg["gap"][gg["gap"] < 10_000]
        print(f"  {tk:8} {len(gg):>8,} {nd:>6} {len(gg)/max(nd,1):>8.1f} "
              f"{gg['counter'].mean()*100:>8.1f}% "
              f"{(gp.median() if len(gp) else np.nan):>9.0f} "
              f"{(gg['gap'] <= 30).mean()*100:>6.0f}% {gg['ratio'].median():>10.1f}")
    print(f"  {'ALL':8} {len(R):>8,} {R['date'].nunique():>6} "
          f"{len(R)/max(R['date'].nunique(),1):>8.1f} "
          f"{R['counter'].mean()*100:>8.1f}%")

    if a.min_gap:
        before = len(R)
        R = R[R["gap"] > a.min_gap]
        print(f"\n  --min-gap {a.min_gap}: kept {len(R):,} of {before:,} "
              f"({len(R)/max(before,1)*100:.0f}%)")

    C, S = R[R["counter"]], R[~R["counter"]]

    print(f"\n{'='*104}")
    print(f"  2. FORWARD RETURN, SIGNED TOWARD THE SPIKE  (median bp)")
    print(f"     COUNTER = flow spiked against the prevailing {a.pdir}m price move")
    print(f"{'='*104}")
    print(f"  {'group':22} {'n':>7} " + "".join(f"{'+'+str(h)+'m':>10}"
                                                for h in HORIZONS))
    for lab, gg in (("COUNTER-directional", C), ("SAME-direction", S),
                    ("placebo (hour-matched)", R)):
        col = (lambda h: f"p{h}") if lab.startswith("placebo") else (lambda h: f"f{h}")
        if lab.startswith("placebo"):
            cells = "".join(f"{R['p15'].median():>10.2f}" if h == 15
                            else f"{'--':>10}" for h in HORIZONS)
        else:
            cells = "".join(f"{gg[col(h)].median():>10.2f}" for h in HORIZONS)
        print(f"  {lab:22} {len(gg):>7,} {cells}")
    print(f"\n  {'CONTRAST counter - same':22} {'':>7} " +
          "".join(f"{C[f'f{h}'].median() - S[f'f{h}'].median():>+10.2f}"
                  for h in HORIZONS))

    print(f"\n{'='*104}")
    print(f"  3. S4 -- DOES THE SIGN HOLD IN BOTH HALVES?")
    print(f"{'='*104}")
    print(f"  {'group':22} {'half':>5} {'n':>7} " +
          "".join(f"{'+'+str(h)+'m':>10}" for h in HORIZONS))
    for lab, gg in (("COUNTER", C), ("SAME", S)):
        for hf in ("IS", "OOS"):
            q = gg[gg["half"] == hf]
            if not len(q):
                continue
            print(f"  {lab:22} {hf:>5} {len(q):>7,} " +
                  "".join(f"{q[f'f{h}'].median():>10.2f}" for h in HORIZONS))

    print(f"\n{'='*104}")
    print(f"  4. THE OPERATIONAL NUMBERS -- overshoot and reclaim")
    print(f"     'how much further does it run against you, and when does it "
          f"come back?'")
    print(f"{'='*104}")
    print(f"  {'group':22} {'n':>7} {'overshoot p50':>14} {'p90':>8} "
          f"{'reclaims':>9} {'med min':>9}")
    for lab, gg in (("COUNTER", C), ("SAME", S)):
        o = gg["overshoot"].dropna()
        rc = gg["reclaim_min"]
        print(f"  {lab:22} {len(gg):>7,} {o.median():>13.1f}bp "
              f"{o.quantile(0.9):>7.1f} {rc.notna().mean()*100:>8.0f}% "
              f"{rc.median():>9.0f}")
    print(f"\n  The live SPY example overshot ~7bp before reversing.")

    print(f"\n{'='*104}")
    print(f"  5. BY TICKER -- counter-directional, median bp")
    print(f"     IWM matters here: its median minute is ~50k against SPY's")
    print(f"     ~656k, so the same dollar spike is a far bigger relative event.")
    print(f"{'='*104}")
    print(f"  {'ticker':8} {'n':>7} {'med ratio':>10} " +
          "".join(f"{'+'+str(h)+'m':>10}" for h in HORIZONS))
    for tk, gg in C.groupby("ticker", sort=False):
        print(f"  {tk:8} {len(gg):>7,} {gg['ratio'].median():>10.1f} " +
              "".join(f"{gg[f'f{h}'].median():>10.2f}" for h in HORIZONS))

    print(f"\n{'='*104}")
    print(f"  6. BY SPIKE MAGNITUDE -- does any effect live in the TAIL?")
    print(f"     The live IWM event was 58.6x the median minute. Pooling that")
    print(f"     with 6x spikes would bury it. Stratifying on the DEFINING")
    print(f"     variable is legitimate; note it is also 4 more comparisons per")
    print(f"     row, so read a lone standout bucket as a thread, not a finding")
    print(f"     (METHODOLOGY 7, 'count your tests before you read the stars').")
    print(f"{'='*104}")
    BUCKETS = [(6, 10), (10, 20), (20, 50), (50, 1e9)]
    print(f"  {'bucket':12} {'group':9} {'n':>7} " +
          "".join(f"{'+'+str(h)+'m':>10}" for h in HORIZONS))
    for lo, hi in BUCKETS:
        for lab, gg in (("counter", C), ("same", S)):
            q = gg[(gg["ratio"] >= lo) & (gg["ratio"] < hi)]
            if len(q) < 30:
                print(f"  {f'{lo}-{hi:g}x':12} {lab:9} {len(q):>7,}   "
                      f"(too few to read)")
                continue
            print(f"  {f'{lo}-{hi:g}x':12} {lab:9} {len(q):>7,} " +
                  "".join(f"{q[f'f{h}'].median():>10.2f}" for h in HORIZONS))

    print(f"\n{'='*104}")
    print(f"  SCORECARD  (pre-committed)")
    print(f"{'='*104}")
    m = lambda ok: "PASS" if ok else "FAIL"
    c15, s15 = C["f15"].median(), S["f15"].median()
    p15 = R["p15"].median()
    isv = C[C["half"] == "IS"]["f15"].median()
    osv = C[C["half"] == "OOS"]["f15"].median()
    # 🚨 A MAGNITUDE FLOOR, ADDED AFTER THE FIRST RUN AND SAID SO.
    # S1-S4 as originally written were SIGN tests. SPY passed all four on
    # +0.33bp -- about one cent on a $285 underlying, against a ~3% option
    # round trip. A criterion that returns PASS on a number indistinguishable
    # from zero is not a criterion. FLOOR is the smallest move that could pay
    # for itself; it is reported alongside, NOT substituted for the original
    # verdicts, so the goalposts stay visible rather than moving.
    FLOOR = 5.0                    # bp, ~half a 1-strike move on these names
    print(f"  S1  counter +15m > 0          {c15:>+8.2f}bp{'':>10}{m(c15 > 0)}")
    print(f"  S1b ... and > {FLOOR:.0f}bp floor    {c15:>+8.2f}bp{'':>10}"
          f"{m(c15 > FLOOR)}   <-- the one that would matter")
    print(f"  S2  counter > same            {c15:>+8.2f} vs {s15:>+7.2f}   "
          f"{m(c15 > s15)}")
    print(f"  S3  counter > placebo         {c15:>+8.2f} vs {p15:>+7.2f}   "
          f"{m(c15 > p15)}")
    print(f"  S4  both halves agree in sign  IS {isv:>+7.2f} / OOS {osv:>+7.2f} "
          f"  {m((isv > 0) == (osv > 0))}")
    print(f"  S5  run again with --min-gap 30 to check clustering")


if __name__ == "__main__":
    main()
