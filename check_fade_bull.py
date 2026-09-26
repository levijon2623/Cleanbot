# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_fade_bull.py
==================
"FADING THE BULL": do PUTs pay MORE when bought into a BULLISH cumulative tide?

THE OBSERVATION THIS CHASES
    check_flow_align rejected cumulative-flow alignment as a universal skip rule
    (6/9 rules, mean +0.1pp) but left a direction asymmetry behind:

        CALL aligned  n=588  win 47.6%  medROE  -3.5  OOS +19285
        CALL opposed  n=260  win 38.1%  medROE -20.9  OOS   -788
        PUT  aligned  n=723  win 43.6%  medROE  -5.4  OOS   +563
        PUT  opposed  n=825  win 52.6%  medROE  +2.7  OOS  +4358

    CALLs want the tide with them; PUTs want it AGAINST them. A put bought into
    a bullish tide is the trade being isolated here. The proposed mechanism is
    the equity skew -- "elevator down, choppy base": crowded-long tape grinds up
    and falls in one move, so the payoff to a put is convex in how one-sided the
    bullish flow has become.

🚨 WHAT WOULD MAKE THIS FAKE, IN DESCENDING ORDER OF LIKELIHOOD

    1  LOOK-AHEAD IN "STRONG". Splitting the fade bin at the sample median of
       cum-flow gave win 63.9% vs 41.3% -- but that median is computed from
       trades the strategy would not have seen. Strength here is therefore a
       TRAILING 2-hour quantile with the current minute excluded (shift(1)),
       reusing check_flow_align.flow_state rather than a second copy (METHODOLOGY 1).

    2  TICKER COMPOSITION. META is 345 of the 825 fade trades (42%) and has by
       far the lowest loss50, so a pooled win rate is substantially a statement
       about META. Reported per rule, and the per-rule table is the one to believe.

    3  DAY COMPOSITION. The fade bin spans 54 days and the with-tide bin 46. A
       pooled gap can be entirely which days each bin drew. Paired within day:
       a day counts only if it holds BOTH.

    4  IT IS A DAY PROPERTY, NOT AN ENTRY PROPERTY. "Today's tape is bullish"
       and "the tide was bullish at this minute" are different claims with
       different strategies attached. PLACEBO A re-reads the tide at a RANDOM
       OTHER MINUTE of the same day: if it sorts outcomes just as well, the
       signal is the day, and entry timing adds nothing.

    5  NOTHING THERE AT ALL. PLACEBO B shuffles the tide across days within each
       rule, destroying any real relation while preserving both marginals. It
       calibrates what this statistic reads on noise.

    6  IT IS JUST "FADE THE TIDE", NOT AN EQUITY SKEW. If the mirror trade --
       CALLs bought into a BEARISH tide -- also works, the mechanism is not the
       skew and the story is wrong even if the numbers are right. The CALL
       mirror is run alongside and is a genuine discriminator, because the
       stated mechanism PREDICTS it should fail.

PRE-COMMITTED CRITERIA (METHODOLOGY 4) -- fixed before the first run
    C1  day-level IS median ROE of FADE STRONG > 0
    C2  ... OOS total > 0
    C3  >= 5 of 6 calendar slices populated AND positive
    C4  OOS beats p95 of PLACEBO B (tide shuffled across days), matched on n
    C5  >= 4 of 5 PUT rules show FADE STRONG beating WITH-TIDE on median ROE
    C6  the CALL mirror does NOT pass C1+C2 -- required by the stated mechanism

    C5 is the one this is most likely to fail, and it is not negotiable
    afterwards: a 42%-META effect is a META effect.

Usage:
  python check_fade_bull.py --paper
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core
from check_flow_align import flow_state          # reuse; do not re-derive

SPLIT = pd.Timestamp("2025-08-21").date()
HDR = (f"  {'bin':26} {'n':>6} {'days':>6} {'loss50':>8} {'win':>7} "
       f"{'medROE':>9} {'IS':>10} {'OOS':>10}")


def agg(rows, label):
    if not rows:
        return f"  {label:26} {'(none)':>6}"
    v = np.array([r["pnl"] for r in rows], float)
    i = [r["pnl"] for r in rows if r["date"] < SPLIT]
    o = [r["pnl"] for r in rows if r["date"] >= SPLIT]
    return (f"  {label:26} {len(v):>6} {len({r['date'] for r in rows}):>6} "
            f"{(v <= -50).mean()*100:>7.1f}% {(v > 0).mean()*100:>6.1f}% "
            f"{np.median(v):>+9.1f} {(np.sum(i) if i else 0):>+10.0f} "
            f"{(np.sum(o) if o else 0):>+10.0f}")


def day_median_roe(rows):
    """Day-level statistic -- power is set by DAYS, not trades (METHODOLOGY 2)."""
    if not rows:
        return np.nan
    d = {}
    for r in rows:
        d.setdefault(r["date"], []).append(r["pnl"])
    return float(np.median([np.median(v) for v in d.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D

    rows = []
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dirn = rule["ticker"], rule["direction"]
        fs = flow_state(D, tk)
        if not fs:
            continue
        cand = sim_core.build_candidates(D, rule)
        if not cand:
            continue
        pol = sim_core.policy_for(rule)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        # the tide a PUT wants against it is BULLISH; a CALL's mirror is BEARISH
        fade_sign = 1.0 if dirn == "PUT" else -1.0
        by_day = {}
        for (dd, mm), (c, s) in fs.items():
            by_day.setdefault(dd, []).append((mm, c, s))
        for d, m, path in cand:
            st = fs.get((d, int(m)))
            if st is None:
                continue
            cum, strong = st
            pnl, xm, tag = sim_core.simulate(path, pol, eod, fill=a.fill,
                                             cush_cap=cap)
            fade = (np.sign(cum) == fade_sign)
            # PLACEBO A: the same day's tide read at a random OTHER minute
            alt = [x for x in by_day.get(d, []) if x[0] != int(m)]
            if alt:
                _, ac, as_ = alt[rng.integers(len(alt))]
                pa = "STRONG" if (np.sign(ac) == fade_sign and as_) else (
                    "WEAK" if np.sign(ac) == fade_sign else "WITH")
            else:
                pa = None
            rows.append(dict(rule=rule["name"], dir=dirn, date=d, pnl=pnl * 100,
                             cum=cum, fade=fade, strong=bool(fade and strong),
                             bin=("STRONG" if (fade and strong)
                                  else "WEAK" if fade else "WITH"),
                             pa=pa))
        print(f"    {rule['name']} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  nothing"); return
    R.to_parquet("_fade_bull.parquet", index=False)

    P = R[R["dir"] == "PUT"]
    C = R[R["dir"] == "CALL"]
    recs = P.to_dict("records")

    print(f"\n{'='*104}")
    print(f"  1. THE STRATEGY  (PUT rules; 'fade' = bought into a BULLISH tide)")
    print(f"{'='*104}")
    print(HDR)
    print(agg(recs, "ALL PUTs"))
    print(agg([r for r in recs if r["bin"] == "WITH"], "with bear tide"))
    print(agg([r for r in recs if r["bin"] == "WEAK"], "fade, weak tide"))
    print(agg([r for r in recs if r["bin"] == "STRONG"], "FADE STRONG"))
    print(f"\n  --- PLACEBO A: tide re-read at a random OTHER minute, same day ---")
    print(HDR)
    for b in ("WITH", "WEAK", "STRONG"):
        print(agg([r for r in recs if r["pa"] == b], f"placebo {b}"))
    print(f"\n  If placebo STRONG matches the real STRONG, the signal is the DAY")
    print(f"  being bullish, not the tide at the entry minute.")

    print(f"\n{'='*104}")
    print(f"  2. PER RULE -- C5 lives here (pooled is 42% META)")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'strong n':>9} {'medROE':>9} {'with n':>8} "
          f"{'medROE':>9} {'delta':>9} {'pass':>6}")
    c5 = 0
    tested = 0
    for nm, g in P.groupby("rule"):
        s = g[g["bin"] == "STRONG"]
        w = g[g["bin"] == "WITH"]
        if len(s) < 10 or len(w) < 10:
            print(f"  {nm:24} {len(s):>9} {'--':>9} {len(w):>8} {'--':>9} "
                  f"{'':>9} {'n/a':>6}")
            continue
        tested += 1
        ms, mw = s["pnl"].median(), w["pnl"].median()
        ok = ms > mw
        c5 += int(ok)
        print(f"  {nm:24} {len(s):>9} {ms:>+9.1f} {len(w):>8} {mw:>+9.1f} "
              f"{ms-mw:>+9.1f} {'YES' if ok else 'no':>6}")
    print(f"\n  C5: {c5}/{tested} rules favour FADE STRONG "
          f"(needs >= 4 of 5)")

    print(f"\n{'='*104}")
    print(f"  3. PAIRED WITHIN DAY -- a day counts only if it holds both bins")
    print(f"{'='*104}")
    dl = []
    for d, g in P.groupby("date"):
        s, w = g[g["bin"] == "STRONG"], g[g["bin"] == "WITH"]
        if len(s) and len(w):
            dl.append(s["pnl"].median() - w["pnl"].median())
    if dl:
        dl = np.array(dl)
        print(f"  {len(dl)} paired days   mean delta medROE {dl.mean():>+8.1f}   "
              f"days favouring fade {(dl > 0).mean()*100:.0f}%")
    else:
        print(f"  no days hold both bins -- the split is BETWEEN days, which")
        print(f"  means block 1 cannot be separated from day composition.")

    print(f"\n{'='*104}")
    print(f"  4. CALENDAR SLICES  (C3)")
    print(f"{'='*104}")
    st = P[P["bin"] == "STRONG"].copy()
    if len(st):
        st["q"] = pd.PeriodIndex(pd.to_datetime(st["date"]), freq="Q")
        qs = sorted(st["q"].unique())[-6:]
        print(f"  {'slice':10} {'n':>6} {'days':>6} {'total':>10} {'medROE':>9}")
        pos = 0
        for q in qs:
            g = st[st["q"] == q]
            tot = g["pnl"].sum()
            pos += int(tot > 0)
            print(f"  {str(q):10} {len(g):>6} {g['date'].nunique():>6} "
                  f"{tot:>+10.0f} {g['pnl'].median():>+9.1f}")
        print(f"\n  C3: {pos}/{len(qs)} slices positive (needs >= 5 of 6)")

    print(f"\n{'='*104}")
    print(f"  5. C4 -- BOOTSTRAP NULL (PLACEBO B: tide shuffled across days)")
    print(f"{'='*104}")
    real = [r["pnl"] for r in recs
            if r["bin"] == "STRONG" and r["date"] >= SPLIT]
    obs = float(np.sum(real)) if real else 0.0
    n = len(real)
    pool = [r["pnl"] for r in recs if r["date"] >= SPLIT]
    if n and len(pool) > n:
        draws = np.array([np.sum(rng.choice(pool, n, replace=False))
                          for _ in range(a.boot)])
        p95 = float(np.percentile(draws, 95))
        print(f"  FADE STRONG OOS total {obs:>+10.0f} on n={n}")
        print(f"  null p50 {np.percentile(draws,50):>+10.0f}   "
              f"p95 {p95:>+10.0f}")
        print(f"  C4: {'PASS' if obs > p95 else 'FAIL'}  "
              f"(percentile {(draws < obs).mean()*100:.1f})")
    else:
        print(f"  too few OOS trades to bootstrap (n={n})")

    print(f"\n{'='*104}")
    print(f"  6. C6 -- THE CALL MIRROR  (mechanism says this must FAIL)")
    print(f"{'='*104}")
    cr = C.to_dict("records")
    print(HDR)
    print(agg([r for r in cr if r["bin"] == "WITH"], "CALL with bull tide"))
    print(agg([r for r in cr if r["bin"] == "STRONG"], "CALL FADE STRONG (bear)"))
    print(f"\n  'Elevator down, choppy base' predicts the put side works and the")
    print(f"  call side does not. If both work it is just 'fade the tide' and")
    print(f"  the equity-skew story is wrong even where the numbers are right.")

    print(f"\n{'='*104}")
    print(f"  SCORECARD")
    print(f"{'='*104}")
    sr = [r for r in recs if r["bin"] == "STRONG"]
    d_is = day_median_roe([r for r in sr if r["date"] < SPLIT])
    oos = np.sum([r["pnl"] for r in sr if r["date"] >= SPLIT]) if sr else 0
    print(f"  C1 day-level IS median ROE > 0      {d_is:>+8.1f}   "
          f"{'PASS' if d_is > 0 else 'FAIL'}")
    print(f"  C2 OOS total > 0                    {oos:>+8.0f}   "
          f"{'PASS' if oos > 0 else 'FAIL'}")
    print(f"  C3/C4/C5/C6 -- see blocks above")


if __name__ == "__main__":
    main()
