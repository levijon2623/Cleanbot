# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_flow_accel.py
===================
DOES THE WHALE ACCELERATE OR DIE INSIDE THE TRIGGER CANDLE?

THE QUESTION
    A 1-minute aggregate hides the second derivative. Two trigger candles with
    identical net premium are different animals if one front-loads its flow and
    fades while the other builds into the close of the minute -- the first is a
    print that already happened, the second is a participant still working an
    order. This splits the entry candle into H1 (0-29s) and H2 (30-59s) and asks
    whether the shape predicts adverse selection.

WHY THE SERIES IS REBUILT FROM TAPE
    UW serves net-prem-ticks at 1m only -- every granularity spelling returns the
    identical 405 rows. backfill_flow30.py rebuilds 30s buckets from the trade
    tape using UW's OWN ask_side/bid_side labels, which reconciles at 0.959-1.000
    against their 1m series. See that module for why the labels, and not a
    price-vs-quote inference, are the input.

THE MINUTE MEASURED IS THE TRIGGER CANDLE, AND THE FEATURE IS CAUSAL
    build_candidates returns the minute of the TRIGGER timestamp `ts`: the entry
    bid/ask is read from bars at `minute_et <= ts` and the P&L path from bars
    strictly after it (sim_core.py:500-517). So minute m is the candle whose flow
    fired the trigger and at whose close the bot enters -- both halves are
    complete before the fill, and nothing here peeks. Bucket alignment is
    confirmed empirically, not assumed: the tape series correlates with UW's at
    lag 0 and collapses to ~0.03 at +/-1 minute.

    DEPLOYABILITY CAVEAT, stated up front so a positive result is not oversold:
    H1/H2 is available at the same instant as the trigger, so it adds no latency
    -- but UW does not serve sub-minute flow, and this series exists only because
    it was rebuilt from the full tape after the fact. Trading on it would require
    a live tape feed and real-time aggressor classification the bot does not have.
    A finding here is a reason to build that, not something switch-on deployable.

🚨 THE TRIGGER CANDLE IS NOT A SPIKE -- WHICH CHANGES WHAT TO MEASURE
    Measured on 1,199 candidates: the trigger minute sits at the 53.6th
    percentile of its own day's premium, with only 9% above the 90th and SPY at
    the 34.5th -- BELOW median. The join is not at fault; the same run shows
    |tape-uw| at those very minutes with a median ratio of 0.000, which is the
    strongest possible confirmation that the (date, ticker, minute) join is
    right. The premise was at fault, "whale spike" included.

    The trigger is an EMA(5) crossover of CUMULATIVE intraday net premium, gated
    on abs(latest_cumulative_flow). A smoothed running total can cross a lagging
    EMA on a completely ordinary minute, so there is frequently no whale in the
    trigger candle to accelerate or die. What the candle IS, is directional:
    68.7% of its aggressive premium runs one way against 50% for a two-sided
    minute. The trigger selects persistence, not bursts.

TWO MEASURES, AND WHICH ONE IS FAITHFUL  (--measure)
    net    H2_net - H1_net, signed, in the trigger's direction. The trigger
           watches CUMULATIVE net premium, so per-minute net is its first
           derivative and this is the second -- the actual acceleration of the
           line the rule is built on. Mechanically faithful; the default.
    gross  calls-lifted + puts-hit for a CALL (ca+pb), the mirror for a PUT.
           Urgency and participation in the trade's direction, ignoring the
           other side. A different question, not a worse one, so both are kept.
    Each bucket stores the four components, so neither needs the tape again.

🚨 THE FIDELITY FILTER, AND WHY IT IS PER-MINUTE
    The tape and UW disagree on a handful of minutes per session -- same-second,
    same-size call+put synthetics and multi-strike packages, which UW excludes
    from a directional net premium. One 09:41 combo booked $54.5M against their
    $2.3M. Drop those minutes on QQQ 2026-09-15 and the remaining 381 correlate
    at 1.000, so a DAY-level gate would discard near-exact tape over minutes this
    test never reads. A candidate is therefore kept on whether ITS OWN minute
    reconciles.
    This selects on fidelity, not on outcome: which minutes are triggers was
    fixed by UW's clean series inside the backtester, so a phantom tape spike
    cannot manufacture a trigger. The residual risk is that a combo landing
    inside a trigger minute drops that trade; the count is reported so the
    reader can judge whether it is plausibly random rather than take it on faith.

PLACEBO SUBJECT (METHODOLOGY 7)
    Every table carries a placebo: the same trades binned by the H1/H2 shape of a
    RANDOM OTHER minute in the same ticker-day. It has the same marginal
    distribution and the same day composition, so if it separates loss50 as well
    as the real thing, the shape is not about the trigger.

BINS (fixed ratios -- causal, nothing calibrated on the sample)
    DECELERATING   H2 < H1
    ACCELERATING   H1 <= H2 < 2*H1
    VIOLENT        H2 >= 2*H1

Usage:
  python check_flow_accel.py --paper
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
CACHE = "_flow30_cache"


# ------------------------------------------------------------------ loading
#: Columns are cast on READ, not trusted from disk. UW returns whole-dollar
#: minutes on some sessions and fractional on others, so parquet lands uw_net as
#: Int64 for one day and Float64 for the next and the concat across 151 days
#: dies on a schema mismatch. Casting here also keeps the loader working against
#: files written by any past version of the extractor.
def _num(df: pl.DataFrame, cols) -> pl.DataFrame:
    have = [c for c in cols if c in df.columns]
    return df.with_columns([pl.col(c).cast(pl.Float64) for c in have])


def load_30s():
    """-> {(date, ticker): {mod: (h1_bull, h2_bull, h1_bear, h2_bear, net)}}"""
    f = sorted(glob.glob(os.path.join(CACHE, "date=*", "flow30.parquet")))
    if not f:
        raise SystemExit(f"  no 30s buckets in {CACHE} -- run backfill_flow30.py")
    cols = ("ca", "cb", "pa", "pb", "midp", "net", "gross", "n")
    df = pl.concat([_num(pl.read_parquet(p), cols) for p in f]).to_pandas()
    miss = [c for c in ("ca", "cb", "pa", "pb") if c not in df.columns]
    if miss:
        raise SystemExit(f"  {CACHE} predates the component schema (missing "
                         f"{miss}) -- re-run backfill_flow30.py --run")
    # gross in each direction, and signed net oriented to each direction
    df["g_bull"] = df["ca"] + df["pb"]
    df["g_bear"] = df["cb"] + df["pa"]
    df["n_bull"] = df["net"]           # net is already +bullish
    df["n_bear"] = -df["net"]
    out = {}
    for (d, tk), g in df.groupby(["date", "underlying_symbol"]):
        m = {}
        for mod, gm in g.groupby("mod"):
            h1 = gm[gm["half"] == 0]
            h2 = gm[gm["half"] == 1]
            m[int(mod)] = dict(
                g_bull=(float(h1["g_bull"].sum()), float(h2["g_bull"].sum())),
                g_bear=(float(h1["g_bear"].sum()), float(h2["g_bear"].sum())),
                n_bull=(float(h1["n_bull"].sum()), float(h2["n_bull"].sum())),
                n_bear=(float(h1["n_bear"].sum()), float(h2["n_bear"].sum())),
                net=float(h1["net"].sum() + h2["net"].sum()))
        out[(d, tk)] = m
    return out, len(f)


def load_uw():
    """-> {(date, ticker): (per-minute uw_net dict, p90 of |uw_net|)}"""
    f = sorted(glob.glob(os.path.join(CACHE, "date=*", "uw1m.parquet")))
    if not f:
        return {}
    df = pl.concat([_num(pl.read_parquet(p), ("uw_net", "mod")) for p in f]
                   ).to_pandas()
    out = {}
    for (d, tk), g in df.groupby(["date", "underlying_symbol"]):
        v = g["uw_net"].astype(float)
        out[(d, tk)] = (dict(zip(g["mod"].astype(int), v)),
                        float(np.nanpercentile(np.abs(v), 90)) if len(v) else np.nan)
    return out


def classify(h1, h2):
    """Fixed thresholds -- nothing calibrated on the sample, so no look-ahead.

    Signed `net` can go negative, where a ratio is meaningless, so the rule is
    stated on the DIFFERENCE first and only uses the 2x ratio from a positive
    base. On non-negative `gross` this reduces exactly to H2<H1 / H2<2*H1 / rest,
    so the two measures stay comparable.
    """
    if h1 == 0 and h2 == 0:
        return None
    if h2 < h1:
        return "DECELERATING"
    if h1 <= 0:                       # turning up off a flat or adverse base
        return "VIOLENT" if h2 > 0 else "ACCELERATING"
    return "VIOLENT" if h2 >= 2 * h1 else "ACCELERATING"


def agg(rows, label):
    if not rows:
        return f"  {label:22} {'(none)':>6}"
    v = np.array([r["pnl"] for r in rows], float)
    o = [r["pnl"] for r in rows if r["date"] >= SPLIT]
    return (f"  {label:22} {len(v):>6} {len({r['date'] for r in rows}):>6} "
            f"{(v <= -50).mean()*100:>7.1f}% {(v > 0).mean()*100:>6.1f}% "
            f"{np.median(v):>+9.1f} {(np.sum(o) if o else np.nan):>+10.0f}")


HDR = (f"  {'bin':22} {'n':>6} {'days':>6} {'loss50':>8} {'win':>7} "
       f"{'med ROE':>9} {'OOS tot':>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--tol", type=float, default=1.0,
                    help="minute kept if |tape-uw| <= tol * p90(|uw|) that day")
    ap.add_argument("--measure", choices=("net", "gross"), default="net",
                    help="net = second derivative of the line the trigger "
                         "watches (faithful); gross = directional participation")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D

    S, nday = load_30s()
    U = load_uw()
    print(f"  loaded {nday} extracted sessions, {len(S)} ticker-days, "
          f"uw1m for {len(U)}")
    print(f"  measure = {a.measure.upper()}"
          f"{'  (H2-H1 of the cumulative line the trigger watches)' if a.measure == 'net' else '  (directional participation)'}\n")

    rows = []
    drop = dict(no_tape=0, no_minute=0, fidelity=0, degenerate=0)
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dirn = rule["ticker"], rule["direction"]
        cand = sim_core.build_candidates(D, rule)
        if not cand:
            continue
        pol = sim_core.policy_for(rule)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        for d, m, path in cand:
            key = (d.isoformat(), tk)
            mm = S.get(key)
            if mm is None:
                drop["no_tape"] += 1
                continue
            rec = mm.get(int(m))
            if rec is None:
                drop["no_minute"] += 1
                continue

            uw = U.get(key)
            if uw is not None and np.isfinite(uw[1]) and uw[1] > 0:
                un = uw[0].get(int(m))
                if un is None or abs(rec["net"] - un) > a.tol * uw[1]:
                    drop["fidelity"] += 1
                    continue

            fld = f"{'n' if a.measure == 'net' else 'g'}_" \
                  f"{'bull' if dirn == 'CALL' else 'bear'}"
            h1, h2 = rec[fld]
            b = classify(h1, h2)
            if b is None:
                drop["degenerate"] += 1
                continue

            # placebo: the shape of a RANDOM OTHER minute, same ticker-day
            alt = [x for x in mm if x != int(m) and 570 <= x <= 955]
            pb = None
            if alt:
                pb = classify(*mm[int(rng.choice(alt))][fld])

            pnl, xm, tag = sim_core.simulate(path, pol, eod, fill=a.fill,
                                             cush_cap=cap)
            rows.append(dict(rule=rule["name"], dir=dirn, date=d, mod=int(m),
                             pnl=pnl * 100, bin=b, placebo=pb,
                             h1=h1, h2=h2, tot=h1 + h2,
                             ratio=(h2 / h1 if h1 > 0 else np.inf)))
        print(f"    {rule['name']} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  no candidates survived -- is the backfill still running?")
        return
    R.to_parquet("_flow_accel.parquet", index=False)

    tot_drop = sum(drop.values())
    print(f"\n  kept {len(R):,} candidates; dropped {tot_drop:,} "
          f"({drop['no_tape']:,} day not extracted, {drop['no_minute']:,} no "
          f"bucket, {drop['fidelity']:,} minute failed fidelity, "
          f"{drop['degenerate']:,} no directional flow)")

    print(f"\n{'='*92}")
    print(f"  1. BASE RATES -- is the split even informative?")
    print(f"{'='*92}")
    print(f"  {'bin':22} {'share':>8} {'placebo share':>15} "
          f"{'med $ in candle':>17}")
    for b in ("DECELERATING", "ACCELERATING", "VIOLENT"):
        g = R[R["bin"] == b]
        ps = (R["placebo"] == b).mean() * 100
        print(f"  {b:22} {len(g)/len(R)*100:>7.1f}% {ps:>14.1f}% "
              f"${g['tot'].median()/1e3:>15,.0f}k")
    print(f"\n  If the real and placebo shares match, the trigger candle is not")
    print(f"  shaped differently from an ordinary minute, and any outcome split")
    print(f"  below has to come from the trades, not from the shape being rare.")

    print(f"\n{'='*92}")
    print(f"  2. OUTCOME BY SHAPE  (pooled -- see block 4 for the honest version)")
    print(f"{'='*92}")
    recs = R.to_dict("records")
    print(HDR)
    print(agg(recs, "ALL"))
    for b in ("DECELERATING", "ACCELERATING", "VIOLENT"):
        print(agg([r for r in recs if r["bin"] == b], b))
    print(f"\n  --- PLACEBO (same trades, shape of a random other minute) ---")
    print(HDR)
    for b in ("DECELERATING", "ACCELERATING", "VIOLENT"):
        print(agg([r for r in recs if r["placebo"] == b], f"placebo {b[:12]}"))

    print(f"\n{'='*92}")
    print(f"  3. WITHIN DIRECTION")
    print(f"{'='*92}")
    for dirn in ("CALL", "PUT"):
        sub = [r for r in recs if r["dir"] == dirn]
        if not sub:
            continue
        print(f"\n  --- {dirn} ---")
        print(HDR)
        for b in ("DECELERATING", "ACCELERATING", "VIOLENT"):
            print(agg([r for r in sub if r["bin"] == b], b))

    print(f"\n{'='*92}")
    print(f"  4. PAIRED WITHIN DAY -- removes day composition")
    print(f"{'='*92}")
    print(f"  A day contributes only if it holds BOTH shapes, so the comparison")
    print(f"  is never between a good day and a bad one.")
    print(f"  {'contrast':34} {'days':>6} {'mean delta loss50':>19} {'wins':>8}")
    for x, y in (("VIOLENT", "DECELERATING"), ("ACCELERATING", "DECELERATING"),
                 ("VIOLENT", "ACCELERATING")):
        ds, dl = [], []
        for d, g in R.groupby("date"):
            gx, gy = g[g["bin"] == x], g[g["bin"] == y]
            if len(gx) < 1 or len(gy) < 1:
                continue
            dl.append((gx["pnl"] <= -50).mean() * 100
                      - (gy["pnl"] <= -50).mean() * 100)
            ds.append(d)
        if len(dl) >= 10:
            dl = np.array(dl)
            print(f"  {x[:12]:>12} vs {y[:12]:<18} {len(dl):>6} "
                  f"{dl.mean():>+18.1f} {(dl < 0).mean()*100:>7.0f}%")
        else:
            print(f"  {x[:12]:>12} vs {y[:12]:<18} {len(dl):>6} "
                  f"{'too few paired days':>19}")

    print(f"\n{'='*92}")
    print(f"  5. WITHIN RULE -- does it hold in most rules or one?")
    print(f"{'='*92}")
    print(f"  {'rule':24} {'decel n':>8} {'loss50':>8} {'accel+viol n':>13} "
          f"{'loss50':>8} {'delta':>8}")
    deltas = []
    for nm, g in R.groupby("rule"):
        dd = g[g["bin"] == "DECELERATING"]
        uu = g[g["bin"] != "DECELERATING"]
        if len(dd) < 15 or len(uu) < 15:
            print(f"  {nm:24} {len(dd):>8} {'--':>8} {len(uu):>13} "
                  f"{'--':>8}  too few")
            continue
        ld = (dd["pnl"] <= -50).mean() * 100
        lu = (uu["pnl"] <= -50).mean() * 100
        deltas.append(lu - ld)
        print(f"  {nm:24} {len(dd):>8} {ld:>7.1f}% {len(uu):>13} "
              f"{lu:>7.1f}% {lu-ld:>+7.1f}")
    if deltas:
        print(f"\n  rules where acceleration has the LOWER loss50: "
              f"{sum(1 for x in deltas if x < 0)}/{len(deltas)}")
        print(f"  mean within-rule delta {np.mean(deltas):+.1f}pp "
              f"(negative = acceleration helps)")

    print(f"\n  HOW TO READ IT")
    print(f"  Block 2's placebo is the control: if a random minute's shape sorts")
    print(f"  outcomes as well as the trigger minute's, there is nothing here.")
    print(f"  Block 4 is the version that survives day composition, and block 5")
    print(f"  decides whether any effect is a book-wide mechanic or one rule.")


if __name__ == "__main__":
    main()
