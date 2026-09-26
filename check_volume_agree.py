# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_volume_agree.py
=====================
HOW CLEANLY DO THE THREE VOLUME MEASURES CONVERGE AT THE PEAK?

Three independent datasets each said the peak minute is a high-volume minute:
    ETF 1m tape   (check_peak_bar)   rank 0.709
    MBO fill_vol  (check_peak_mbo)   rank 0.653
    MBO trade VAP (check_peak_vap)   rank 0.655
Separate agreement on the MEAN is not the same as agreeing TRADE BY TRADE. Three
noisy measures of the same underlying quantity will all average high while
disagreeing on which individual minute was the peak. That distinction is the
whole difference between "volume marks the peak" and "volume could detect it".

So all three ranks are recomputed HERE, in one pass, on the same rows -- no
joining of three parquet files on a float key -- and then:
  A  pairwise correlation of the three ranks
  B  CONCORDANCE: how often do all three agree the peak bar is high-volume?
  C  the detector question: PRECISION. If a rule fires when all three exceed a
     threshold, how many of the move's NON-peak new-extreme minutes also fire?
     A mean rank of 0.71 says the peak is usually a high-volume bar. It does NOT
     say a high-volume bar is usually the peak, and only C can tell them apart.

Restricted to the MBO days (RTY 43 + NQ 12), since two of the three measures
exist only there. METHODOLOGY 7: the honest n is DAYS.

Usage:
  python check_volume_agree.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

from check_peak_levels import load_levels, peak_rows
from check_peak_bar import bars_of, boot, midrank

PROXY = {"IWM": "RTY_c_0", "QQQ": "NQ_c_0"}


def _load(cache, col):
    out = {}
    for tk, sym in PROXY.items():
        fp = os.path.join(cache, f"{sym}.parquet")
        if not os.path.exists(fp):
            continue
        d = pl.read_parquet(fp).to_pandas()
        d["date"] = pd.to_datetime(d["date"]).dt.date
        out[tk] = {dt: g.set_index("mod")[col] for dt, g in d.groupby("date")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=19)
    a = ap.parse_args()
    rng_ = np.random.default_rng(a.seed)

    mbo = _load("_mbo_cache", "fill_vol")
    vap = _load("_vap_cache", "tvol")
    if not mbo or not vap:
        print("  need _mbo_cache and _vap_cache"); return
    tickers = list(PROXY)
    df = peak_rows(["SPY", "QQQ", "IWM"], ["CALL", "PUT"], a.pct)
    df = df[df["ticker"].isin(tickers)]
    DAY, _ = load_levels(tickers)
    bars = {tk: bars_of(tk) for tk in tickers}

    rows, fire = [], []
    for r in df.itertuples():
        b = bars.get(r.ticker, {}).get(r.date)
        M = mbo.get(r.ticker, {}).get(r.date)
        V = vap.get(r.ticker, {}).get(r.date)
        if b is None or M is None or V is None:
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
        ext_mods = [int(mods[t]) for t in idx]
        if not all(m in M.index and m in V.index for m in ext_mods):
            continue
        pk_mod = int(mods[k])
        mv = b["v"][s:e]
        mvm = np.nanmean(mv) if mv.size else np.nan
        if not (np.isfinite(mvm) and mvm > 0):
            continue

        # the three series, evaluated at the SAME set of new-extreme minutes
        tape = np.array([b["v"][t] / mvm for t in idx], float)
        fill = np.array([float(M.at[m]) for m in ext_mods], float)
        tvol = np.array([float(V.at[m]) for m in ext_mods], float)
        pk_i = ext_mods.index(pk_mod)

        def rank_of(arr, i):
            pool = np.delete(arr, i)
            pool = pool[np.isfinite(pool)]
            return midrank(pool, arr[i]) if (pool.size >= 4 and np.isfinite(arr[i])) else np.nan

        rec = dict(ticker=r.ticker, date=r.date, dir=r.dir, peak_roe=r.peak_roe,
                   n_ext=len(idx),
                   tape=rank_of(tape, pk_i), fill=rank_of(fill, pk_i),
                   tvol=rank_of(tvol, pk_i))
        rows.append(rec)

        # ---- for PRECISION: rank EVERY new-extreme minute, flag which is the peak
        for i, m in enumerate(ext_mods):
            fire.append(dict(date=r.date, is_peak=(i == pk_i),
                             tape=rank_of(tape, i), fill=rank_of(fill, i),
                             tvol=rank_of(tvol, i)))

    A = pd.DataFrame(rows).dropna(subset=["tape", "fill", "tvol"])
    F = pd.DataFrame(fire).dropna(subset=["tape", "fill", "tvol"])
    if A.empty:
        print("  nothing scored"); return
    A.to_parquet("_volume_agree.parquet", index=False)

    print(f"\n{'='*92}")
    print(f"  DO THE THREE VOLUME MEASURES AGREE AT THE PEAK?")
    print(f"  n={len(A):,} trades on {A['date'].nunique()} days")
    print(f"{'='*92}")

    print(f"\n  MEAN RANK AT THE PEAK (0.5 = indistinguishable)")
    for c, lbl in (("tape", "ETF 1m tape volume"), ("fill", "MBO fill volume"),
                   ("tvol", "MBO trade volume (VAP)")):
        v = A[c].to_numpy(float)
        ci = boot(v, A["date"].to_numpy(), a.boot, rng_)
        print(f"    {lbl:26} {v.mean():.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")

    print(f"\n  A. PAIRWISE CORRELATION OF THE RANKS (trade by trade)")
    for x, y in (("tape", "fill"), ("tape", "tvol"), ("fill", "tvol")):
        p = float(np.corrcoef(A[x], A[y])[0, 1])
        sp = float(A[x].corr(A[y], method="spearman"))
        print(f"    {x:5} vs {y:5}   pearson {p:+.3f}   spearman {sp:+.3f}")
    print(f"    -> MBO fill vs VAP trade volume measure nearly the same thing and")
    print(f"       should be high. The informative pair is ETF tape vs either MBO")
    print(f"       one: different instrument, different venue, different vendor.")

    print(f"\n  B. CONCORDANCE -- how often do all three point the same way?")
    for thr in (0.5, 0.6, 0.7, 0.8):
        hi = (A[["tape", "fill", "tvol"]] > thr)
        n3 = (hi.sum(axis=1) == 3).mean() * 100
        n2 = (hi.sum(axis=1) >= 2).mean() * 100
        n0 = (hi.sum(axis=1) == 0).mean() * 100
        print(f"    rank > {thr:.1f}:  all three {n3:>5.1f}%   "
              f"at least two {n2:>5.1f}%   none {n0:>5.1f}%")
    exp3 = ((1 - 0.5) ** 3) * 100
    print(f"    (if the three were independent coin flips, 'all three > 0.5' "
          f"would be {exp3:.1f}%)")

    print(f"\n  C. PRECISION -- the detector question")
    print(f"     Every new-extreme minute of every move is ranked, not just the peak.")
    print(f"     n={len(F):,} candidate minutes, of which {F['is_peak'].sum():,} "
          f"({F['is_peak'].mean()*100:.1f}%) are the actual peak.")
    print(f"     {'rule':34} {'fires':>8} {'catches':>8} {'precision':>10} {'recall':>8}")
    base = F["is_peak"].mean()
    rules = [
        ("tape > 0.70", F["tape"] > 0.70),
        ("all three > 0.50", (F["tape"] > .5) & (F["fill"] > .5) & (F["tvol"] > .5)),
        ("all three > 0.70", (F["tape"] > .7) & (F["fill"] > .7) & (F["tvol"] > .7)),
        ("all three > 0.80", (F["tape"] > .8) & (F["fill"] > .8) & (F["tvol"] > .8)),
        ("mean of three > 0.75", F[["tape", "fill", "tvol"]].mean(axis=1) > 0.75),
    ]
    for lbl, mask in rules:
        n = int(mask.sum())
        if n == 0:
            continue
        caught = int((mask & F["is_peak"]).sum())
        prec = caught / n
        rec = caught / F["is_peak"].sum()
        print(f"     {lbl:34} {n:>8,} {caught:>8,} {prec*100:>9.1f}% {rec*100:>7.1f}%")
    print(f"     {'(base rate -- fire on every extreme)':34} {len(F):>8,} "
          f"{int(F['is_peak'].sum()):>8,} {base*100:>9.1f}% {100.0:>7.1f}%")

    print(f"\n  HOW TO READ IT")
    print(f"  B says whether the three SEE the same minute. C says whether that is")
    print(f"  worth anything: precision must beat the base rate by enough to pay for")
    print(f"  exiting early on every false alarm. A rank of 0.71 at the peak is")
    print(f"  compatible with a precision barely above chance, because there are")
    print(f"  many more non-peak extremes than peaks.")


if __name__ == "__main__":
    main()
