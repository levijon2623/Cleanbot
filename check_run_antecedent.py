# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_run_antecedent.py
=======================
DOES 30-SECOND DIRECTIONAL FLOW LEAD A LARGE SPY RUN?

WHY THESE FEATURES AND NOT FLOW GENERALLY
    check_run_coverage settled the prior question. The deployed rule precedes
    only ~1.2% of $2 SPY runs, so the headroom is enormous -- but the RAW
    cumulative-flow crossover, ungated, covers 74% of all minutes and precedes
    just 62% of run starts. Lift 0.84, and 0.84 again hour-matched. That family
    of signal demonstrably does not know where a run begins.
    So mining "flow before runs" and landing on anything crossover-shaped would
    be rediscovering a feature already measured at lift 0.84. The features below
    are deliberately the ones the trigger has NEVER seen: the sub-minute split,
    the two sides separately, and the direction ratio. All come from the four
    components backfill_flow30 stores per half-minute.

🚨 THE TRAP THIS DESIGN EXISTS TO AVOID
    Selecting on runs and looking backwards always finds similarities -- there
    are hundreds of runs and every feature has a distribution. The only thing
    that separates a signal from a shadow is the BASE RATE: of all the times the
    antecedent occurred, how often did a run follow? Every row below reports
    P(run | feature) against P(run) on the same minutes, and their ratio. A lift
    near 1.0 means the feature marks runs exactly as often as marking minutes at
    random would.

🚨 THE SAMPLE IS NOT A RANDOM SET OF SESSIONS
    The 30s cache covers the 151 days on which the BOOK had a candidate -- days
    where some rule fired, which correlates with activity. Base rates are
    therefore computed within the SAME day set, never against all 511 sessions,
    so the comparison stays internally valid even though the days are selected.

CAUSALITY
    Every feature uses minute m and earlier only. Percentile features use a
    trailing window with the current minute EXCLUDED (shift 1), the same
    construction check_flow_align uses. The label looks forward from m; the
    features never do. Run detection is imported from check_run_coverage rather
    than rewritten (METHODOLOGY 1).

COUNT THE TESTS: 4 features x 3 cuts x 2 thresholds = 24 cells. At 95% that is
    ~1 false positive by construction, so a lone strong cell is a thread, not a
    finding. The placebo column is what tells them apart.

Usage:
  python check_run_antecedent.py
  python check_run_antecedent.py --thr 2.0 --window 30 --horizon 5
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

import check_run_coverage as C

CACHE = "_flow30_cache"
TICK = "SPY"


def load_flow30():
    """-> DataFrame[date, mod, h1_bull, h2_bull, h1_bear, h2_bear] for SPY."""
    f = sorted(glob.glob(os.path.join(CACHE, "date=*", "flow30.parquet")))
    if not f:
        raise SystemExit(f"  no 30s buckets in {CACHE} -- run backfill_flow30.py")
    cols = ("ca", "cb", "pa", "pb", "n")
    parts = []
    for p in f:
        d = pl.read_parquet(p)
        d = d.with_columns([pl.col(c).cast(pl.Float64)
                            for c in cols if c in d.columns])
        parts.append(d.filter(pl.col("underlying_symbol") == TICK))
    df = pl.concat(parts).to_pandas()
    if df.empty:
        raise SystemExit(f"  no {TICK} rows in the 30s cache")
    df["bull"] = df["ca"] + df["pb"]
    df["bear"] = df["cb"] + df["pa"]
    w = df.pivot_table(index=["date", "mod"], columns="half",
                       values=["bull", "bear"], aggfunc="sum").fillna(0.0)
    w.columns = [f"h{int(h)+1}_{v}" for v, h in w.columns]
    w = w.reset_index()
    w["date"] = pd.to_datetime(w["date"]).dt.date
    return w


def build_features(w):
    """Causal, per day. Nothing here may see minute m+1."""
    out = []
    for d, g in w.sort_values("mod").groupby("date"):
        g = g.copy()
        bull = g["h1_bull"] + g["h2_bull"]
        bear = g["h1_bear"] + g["h2_bear"]
        tot = (bull + bear).replace(0, np.nan)

        # 1. BURST: this minute's bullish premium against its own trailing
        #    30-minute median. shift(1) keeps the current minute out of the
        #    window it is being judged against.
        med = bull.rolling(30, min_periods=10).median().shift(1)
        g["burst"] = bull / med.replace(0, np.nan)

        # 2. IMBALANCE: how one-sided the minute is. The 1m trigger sees only
        #    the signed residue, never the ratio.
        g["imbalance"] = bull / tot

        # 3. ACCEL: the sub-minute second derivative -- does the bullish side
        #    build into the close of the minute or fade? Invisible at 1m.
        g["accel"] = (g["h2_bull"] - g["h1_bull"]) / tot

        # 4. H2 SHARE: the back half's share of the minute's bullish premium.
        g["h2_share"] = g["h2_bull"] / bull.replace(0, np.nan)

        out.append(g)
    return pd.concat(out, ignore_index=True)


def label_runs(bars, days, thr, window, horizon):
    """-> {(date, mod): True} for minutes where a run starts within horizon."""
    lab, n = {}, 0
    for d, g in bars[bars["date"].isin(days)].groupby("date"):
        ev = C.find_runs(g["close"].to_numpy(float), g["mod"].to_numpy(int),
                         thr, window)
        n += len(ev)
        for s, _pk, _mv in ev:
            for m in range(s - horizon, s + 1):
                lab[(d, m)] = True
    return lab, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=5,
                    help="feature at m counts as leading a run starting in [m, m+h]")
    ap.add_argument("--seed", type=int, default=53)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    w = load_flow30()
    F = build_features(w)
    days = sorted(F["date"].unique())
    bars = C.spy_bars(min(days))
    bars = bars[bars["date"].isin(days)]
    F = F[F["mod"] + a.window <= C.RTH1]
    print(f"  {TICK}: {len(F):,} minutes of 30s flow over {len(days)} sessions "
          f"({days[0]} .. {days[-1]})")

    # PLACEBO: the same feature value drawn from a RANDOM OTHER MINUTE of the
    # same day. Same marginal distribution, same day composition, no relation
    # to what happens next. If it lifts as well as the real thing, nothing here
    # is about timing.
    for col in ("burst", "imbalance", "accel", "h2_share"):
        F[f"pl_{col}"] = F.groupby("date")[col].transform(
            lambda s: s.to_numpy()[rng.permutation(len(s))])

    feats = ["burst", "imbalance", "accel", "h2_share"]
    for thr in a.thr:
        lab, nruns = label_runs(bars, days, thr, a.window, a.horizon)
        y = np.array([(d, m) in lab for d, m in zip(F["date"], F["mod"])])
        base = y.mean()
        print(f"\n{'='*94}")
        print(f"  ${thr:.2f} runs / {a.window}m window   {nruns} runs   "
              f"P(run starts within {a.horizon}m) = {base*100:.2f}% of minutes")
        print(f"{'='*94}")
        print(f"  {'feature':12} {'cut':>10} {'n':>7} {'P(run|f)':>10} "
              f"{'lift':>7} {'placebo lift':>14}")
        for col in feats:
            v = F[col].to_numpy(float)
            pv = F[f"pl_{col}"].to_numpy(float)
            ok = np.isfinite(v)
            for q in (0.80, 0.90, 0.95):
                t = np.nanquantile(v[ok], q)
                m = ok & (v >= t)
                if m.sum() < 50:
                    continue
                lift = (y[m].mean() / base) if base else np.nan
                pt = np.nanquantile(pv[np.isfinite(pv)], q)
                pm = np.isfinite(pv) & (pv >= pt)
                plift = (y[pm].mean() / base) if base else np.nan
                print(f"  {col:12} {'top ' + f'{int((1-q)*100)}%':>10} "
                      f"{m.sum():>7,} {y[m].mean()*100:>9.2f}% "
                      f"{lift:>7.2f} {plift:>14.2f}")

    print(f"\n  HOW TO READ IT")
    print(f"  LIFT 1.0 = the feature marks run-onset minutes exactly as often as")
    print(f"  marking minutes at random. The benchmark to beat is the existing")
    print(f"  signal's 0.84 on this same task, and anything under ~1.2 is not")
    print(f"  worth a trigger. PLACEBO LIFT should sit at 1.0; if it tracks the")
    print(f"  real column, the feature is picking up day composition, not timing.")
    print(f"  24 cells were computed -- expect ~1 to look good by chance.")


if __name__ == "__main__":
    main()
