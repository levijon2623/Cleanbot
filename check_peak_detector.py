# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_peak_detector.py
======================
VOLUME + BAR RANGE + RSI TOGETHER -- is the combination anything like reliable?

WHAT IS ALREADY KNOWN, SEPARATELY
    Ranked against the OTHER running-extreme bars of the same move
    (check_peak_bar, placebos ~0.50):
        volume vs move mean   0.709   and STRENGTHENS with peak size -> 0.750
        effort/result         0.653
        bar range vs ATR      0.624   stable across peak sizes
        RSI(14) extension     0.591   but DECAYS to 0.524 (null) on >+200%
        RSI 3-bar change      0.414   i.e. momentum fading into the peak
    Each leans the right way. None is a detector on its own.

THE QUESTION A MEAN RANK CANNOT ANSWER
    "The peak bar is usually a high-volume bar" is not "a high-volume bar is
    usually the peak". There are ~7 new-extreme bars per move and only one peak,
    so the base rate is low and precision is what matters. This scores EVERY
    running-extreme bar, flags which was the peak, and measures precision and
    recall directly -- the same framing check_volume_agree used to show the best
    volume rule reaches 14% precision against a 5% base rate.

🚨 WHY COMBINING MIGHT BUY NOTHING
    Volume and bar range are mechanically linked: a wider bar trades more. If
    the features are redundant, an AND of three conditions just fires less often
    at the same precision -- fewer catches for no gain. So the correlation matrix
    is printed FIRST and the combinations are read against it.

NO SEARCHING. The rule forms are fixed in advance (top-quartile / top-decile on
    each subset). Thresholds are not tuned to the answer, and no rule is added
    after seeing the table -- with 5 features and a free threshold this would
    otherwise be a machine for manufacturing a detector.

Usage:
  python check_peak_detector.py --tickers SPY QQQ IWM
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_peak_levels import load_levels, peak_rows
from check_peak_bar import bars_of, feats_at, midrank

FEATS = ("volx", "effort", "rngx", "rsi_ext", "rsi_d3")
LAB = {"volx": "volume", "effort": "effort/result", "rngx": "bar range",
       "rsi_ext": "RSI level", "rsi_d3": "RSI 3-bar chg"}
#: rsi_d3 ranked LOW at peaks (momentum fading), so its useful direction is
#: inverted relative to the others. Flipped once here rather than in every rule.
INVERT = {"rsi_d3"}


def build(df, bars, DAY):
    rows = []
    for r in df.itertuples():
        b = bars.get(r.ticker, {}).get(r.date)
        lv = DAY.get(r.ticker, {}).get(r.date)
        if b is None or lv is None:
            continue
        atr = lv["atr"]
        if not (np.isfinite(atr) and atr > 0):
            continue
        up = r.dir == "CALL"
        mods = b["mod"]
        s = int(np.searchsorted(mods, r.entry, side="left"))
        e = int(np.searchsorted(mods, r.eod, side="right"))
        if e - s < 10:
            continue
        seg = b["hi"][s:e] if up else b["lo"][s:e]
        j = int(np.argmax(seg) if up else np.argmin(seg))
        k = s + j
        idx, cur = [], (-np.inf if up else np.inf)
        for t in range(s, k + 1):
            x = b["hi"][t] if up else b["lo"][t]
            if (up and x > cur) or (not up and x < cur):
                cur = x
                idx.append(t)
        if len(idx) < 6:
            continue
        F = [feats_at(b, t, s, e - 1, atr, up) for t in idx]
        for i, t in enumerate(idx):
            rec = dict(date=r.date, ticker=r.ticker, peak_roe=r.peak_roe,
                       is_peak=(t == k))
            for f in FEATS:
                pool = np.array([x.get(f, np.nan) for q, x in enumerate(F) if q != i],
                                float)
                pool = pool[np.isfinite(pool)]
                v = F[i].get(f, np.nan)
                if np.isfinite(v) and pool.size >= 4:
                    rk = midrank(pool, v)
                    rec[f] = (1.0 - rk) if f in INVERT else rk
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="+", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    a = ap.parse_args()

    df = peak_rows(a.tickers, a.dirs, a.pct)
    DAY, _ = load_levels(a.tickers)
    bars = {tk: bars_of(tk) for tk in a.tickers}
    F = build(df, bars, DAY).dropna(subset=list(FEATS))
    if F.empty:
        print("  nothing"); return
    F.to_parquet("_peak_detector.parquet", index=False)
    base = F["is_peak"].mean()
    print(f"\n{'='*94}")
    print(f"  {len(F):,} running-extreme bars, {int(F['is_peak'].sum()):,} of them "
          f"peaks  ->  BASE RATE {base*100:.1f}%")
    print(f"  ranks are within-move; rsi_d3 inverted so high = 'more peak-like'")
    print(f"{'='*94}")

    print(f"\n  1. ARE THEY INDEPENDENT?  (rank correlation across all extreme bars)")
    print(f"  {'':16}" + "".join(f"{LAB[f]:>15}" for f in FEATS))
    C = F[list(FEATS)].corr(method="spearman")
    for f in FEATS:
        print(f"  {LAB[f]:16}" + "".join(f"{C.loc[f,g]:>15.2f}" for g in FEATS))
    print(f"  -> pairs above ~0.6 are one signal twice; an AND of those cannot")
    print(f"     add information, it can only fire less often.")

    print(f"\n  2. SINGLE FEATURES")
    print(f"  {'rule':28} {'fires':>8} {'catches':>8} {'precision':>10} "
          f"{'recall':>8} {'lift':>6}")
    rules = []
    for f in FEATS:
        for q, lbl in ((0.75, "top quartile"), (0.90, "top decile")):
            rules.append((f"{LAB[f]} {lbl}", F[f] >= q))
    for lbl, m in rules:
        n = int(m.sum())
        if not n:
            continue
        c = int((m & F["is_peak"]).sum())
        p = c / n
        print(f"  {lbl:28} {n:>8,} {c:>8,} {p*100:>9.1f}% "
              f"{c/F['is_peak'].sum()*100:>7.1f}% {p/base:>5.2f}x")

    print(f"\n  3. COMBINATIONS (pre-specified, not searched)")
    print(f"  {'rule':28} {'fires':>8} {'catches':>8} {'precision':>10} "
          f"{'recall':>8} {'lift':>6}")
    combos = [
        ("vol+range, both top qtr", (F.volx >= .75) & (F.rngx >= .75)),
        ("vol+RSI lvl, both top qtr", (F.volx >= .75) & (F.rsi_ext >= .75)),
        ("vol+RSI chg, both top qtr", (F.volx >= .75) & (F.rsi_d3 >= .75)),
        ("all three, top qtr", (F.volx >= .75) & (F.rngx >= .75) & (F.rsi_ext >= .75)),
        ("all three, top decile", (F.volx >= .90) & (F.rngx >= .90) & (F.rsi_ext >= .90)),
        ("vol+range+RSIchg, top qtr",
         (F.volx >= .75) & (F.rngx >= .75) & (F.rsi_d3 >= .75)),
        ("mean of 5 >= 0.75", F[list(FEATS)].mean(axis=1) >= .75),
        ("mean of 5 >= 0.85", F[list(FEATS)].mean(axis=1) >= .85),
    ]
    for lbl, m in combos:
        n = int(m.sum())
        if not n:
            print(f"  {lbl:28} {'never fires':>8}")
            continue
        c = int((m & F["is_peak"]).sum())
        p = c / n
        print(f"  {lbl:28} {n:>8,} {c:>8,} {p*100:>9.1f}% "
              f"{c/F['is_peak'].sum()*100:>7.1f}% {p/base:>5.2f}x")

    print(f"\n  4. THE CEILING -- best achievable if you could pick perfectly")
    for lbl, sub in (("all moves", F), (">+200% peaks", F[F.peak_roe >= 2.0])):
        if sub.empty:
            continue
        per = sub.groupby(level=0).size() if False else None
        n_moves = len(sub) / max(sub["is_peak"].sum(), 1)
        print(f"    {lbl:16} {len(sub):>8,} bars, {int(sub['is_peak'].sum()):>6,} "
              f"peaks -> {sub['is_peak'].mean()*100:.1f}% base, "
              f"~{n_moves:.1f} extremes per peak")

    print(f"\n  HOW TO READ IT")
    print(f"  'lift' is precision over the base rate. A rule at 2x precision still")
    print(f"  misses most peaks and fires mostly on non-peaks; the question is")
    print(f"  whether that is worth acting on, and check_volume_exit already says")
    print(f"  a signal exit LOSES once re-entry is priced (-2029pp OOS), because")
    print(f"  interrupting a run costs more than the extra trades recover.")
    print(f"  Judge 'reliable' against that, not against 0.50.")


if __name__ == "__main__":
    main()
