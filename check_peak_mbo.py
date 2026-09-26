# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_peak_mbo.py
=================
WHAT WAS THE ORDER BOOK DOING AT THE PEAK MINUTE?  -- a PILOT, not a test.

check_peak_bar showed the peak bar carries more volume and more volume PER UNIT
OF RANGE than the earlier new-high bars of the same move (0.71 / 0.65 rank,
placebo 0.49). That is the absorption SHAPE inferred from 1m OHLCV. The book
itself would say whether size was actually stacking and being eaten -- whether
depth replenished at the touch while price failed to advance.

=========================  READ THE POWER FIRST  ==========================
  RTY.c.0  43 days   -> the IWM proxy
  NQ.c.0   12 days   -> the QQQ proxy
  SPY has no MBO at all (that would be ES, never pulled).
METHODOLOGY 7: POWER IS SET BY DAYS. 43 days and 12 days are pilot-scale. The
day-clustered CI below will be wide and it is not lying when it is. 12 days is
underpowered to the point where it can only produce false negatives --
check_mbo_flow_interaction says so in its own docstring about this same sample.
Nothing here can promote or retire a rule. It can only say whether a book-level
tell is worth BUYING DATA to investigate ($179/mo for a live feed).

TWO INSTRUMENT CAVEATS, both structural
  1. The peaks are ETF OPTION peaks; the book is the FUTURES book. RTY and IWM
     track closely but they are different instruments with different
     participants. A tell that shows up here still has to survive on the ETF.
  2. `absorb` in the cache is the HIDDEN-SIZE definition (fill volume hitting
     order ids never seen added). build_mbo_features.py:209 records it as
     measuring ~0 in practice, so a null on that column says nothing about
     absorption in the sense meant here. `qdepth`, `age_s`, `hhi` and
     `fill_vol` are the columns that carry the replenishment story.

DESIGN -- identical to check_peak_bar, so the two are directly comparable
  The peak minute is ranked against the OTHER RUNNING-EXTREME minutes of the
  same move: every one of them set a new high (low); the peak is just the last.
  PLACEBO: an earlier running-extreme minute scored as if it were the peak. It
  must return 0.50, and it is re-read in every subset (METHODOLOGY 7).

Usage:
  python check_peak_mbo.py
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
MBO_FEATS = ("qdepth", "age_s", "hhi", "n_orders_touch", "spread_ticks",
             "imb", "fill_vol", "tickchase", "absorb", "absorb_sgn",
             "mod_price", "mod_size")
MLAB = {"qdepth": "depth at touch", "age_s": "resting order age (s)",
        "hhi": "queue concentration", "n_orders_touch": "orders at touch",
        "spread_ticks": "spread (ticks)", "imb": "book imbalance",
        "fill_vol": "fill volume", "tickchase": "price-changing MODIFYs",
        "absorb": "hidden-fill share (known ~0)", "absorb_sgn": "hidden fill, signed",
        "mod_price": "re-peg count", "mod_size": "size-change count"}


def load_mbo():
    out = {}
    for tk, sym in PROXY.items():
        fp = f"_mbo_cache/{sym}.parquet"
        if not os.path.exists(fp):
            continue
        d = pl.read_parquet(fp).to_pandas()
        d["date"] = pd.to_datetime(d["date"]).dt.date
        out[tk] = {dt: g.set_index("mod") for dt, g in d.groupby("date")}
        print(f"  {tk} <- {sym}: {d['date'].nunique()} days, {len(d):,} minute rows")
    return out


def run(df, bars, DAY, mbo, rng_):
    rows = []
    for r in df.itertuples():
        M = mbo.get(r.ticker, {}).get(r.date)
        b = bars.get(r.ticker, {}).get(r.date)
        lv = DAY.get(r.ticker, {}).get(r.date)
        if M is None or b is None or lv is None:
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
        run_mods, cur = [], (-np.inf if up else np.inf)
        for t in range(s, k + 1):
            x = b["hi"][t] if up else b["lo"][t]
            if (up and x > cur) or (not up and x < cur):
                cur = x
                run_mods.append(int(mods[t]))
        if len(run_mods) < 6:
            continue
        pk_mod = int(mods[k])
        have = [m for m in run_mods if m in M.index]
        if pk_mod not in M.index or len(have) < 6:
            continue
        prior = [m for m in have if m != pk_mod]
        if len(prior) < 5:
            continue
        pi = int(rng_.integers(0, len(prior)))
        plc_mod = prior[pi]
        plc_pool = [m for q, m in enumerate(prior) if q != pi]

        rec = dict(ticker=r.ticker, dir=r.dir, date=r.date, peak_roe=r.peak_roe,
                   nrun=len(have))
        for f in MBO_FEATS:
            if f not in M.columns:
                continue
            pv = M.at[pk_mod, f] if pk_mod in M.index else np.nan
            pool = pd.to_numeric(M.loc[prior, f], errors="coerce").dropna().to_numpy()
            if np.isfinite(pv) and pool.size >= 4:
                rec[f] = midrank(pool, pv)
            qv = M.at[plc_mod, f] if plc_mod in M.index else np.nan
            qpool = pd.to_numeric(M.loc[plc_pool, f], errors="coerce").dropna().to_numpy()
            if np.isfinite(qv) and qpool.size >= 4:
                rec[f + "_plc"] = midrank(qpool, qv)
        rows.append(rec)
    return pd.DataFrame(rows)


def report(A, nboot, rng_, title, minn=40):
    if A.empty or len(A) < minn:
        print(f"\n  {title}: too few rows ({len(A)})")
        return
    print(f"\n  {title}   (n={len(A):,}, {A['date'].nunique()} DAYS -- this is the "
          f"real sample size)")
    print(f"  {'feature':30} {'n':>6} {'REAL pct':>9} {'95% CI':>17} "
          f"{'PLACEBO':>9} {'95% CI':>17}")
    for f in MBO_FEATS:
        if f not in A.columns:
            continue
        g = A.dropna(subset=[f])
        if len(g) < minn:
            continue
        v = g[f].to_numpy(float)
        ci = boot(v, g["date"].to_numpy(), nboot, rng_)
        pc = f + "_plc"
        if pc in A.columns and A[pc].notna().sum() >= minn:
            p = A.dropna(subset=[pc])
            pv = p[pc].to_numpy(float)
            pci = boot(pv, p["date"].to_numpy(), nboot, rng_)
        else:
            pv, pci = np.array([np.nan]), (np.nan, np.nan)
        flag = "  <--" if (ci[0] > 0.5 or ci[1] < 0.5) else ""
        bad = " !PLC" if (np.isfinite(pci[0]) and (pci[0] > 0.5 or pci[1] < 0.5)) else ""
        print(f"  {MLAB.get(f, f):30} {len(g):>6} {v.mean():>9.3f} "
              f"[{ci[0]:>6.3f},{ci[1]:>6.3f}] {np.nanmean(pv):>9.3f} "
              f"[{pci[0]:>6.3f},{pci[1]:>6.3f}]{flag}{bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng_ = np.random.default_rng(a.seed)

    tickers = list(PROXY)
    mbo = load_mbo()
    if not mbo:
        print("  no MBO cache"); return
    df = peak_rows(["SPY", "QQQ", "IWM"], a.dirs, a.pct)
    df = df[df["ticker"].isin(tickers)]
    DAY, _ = load_levels(tickers)
    bars = {tk: bars_of(tk) for tk in tickers}
    A = run(df, bars, DAY, mbo, rng_)
    if A.empty:
        print("  nothing scored -- no peak minutes landed on MBO days"); return
    A.to_parquet("_peak_mbo.parquet", index=False)

    print(f"\n{'='*100}")
    print(f"  ORDER BOOK AT THE PEAK MINUTE vs THE MOVE'S EARLIER NEW EXTREMES")
    print(f"  PILOT ONLY -- {A['date'].nunique()} MBO days total. Wide CIs are honest.")
    print(f"{'='*100}")
    report(A, a.boot, rng_, "ALL (both proxies pooled)")
    for tk, g in A.groupby("ticker"):
        report(g, a.boot, rng_, f"{tk} -> {PROXY[tk]}")
    big = A[A["peak_roe"] >= 1.0]
    report(big, a.boot, rng_, "LARGE PEAKS (ROE >= +100%)")

    print(f"\n  HOW TO READ IT")
    print(f"  The honest n is DAYS, printed in each header, not the row count.")
    print(f"  'depth at touch' and 'resting order age' are the replenishment story:")
    print(f"  if size is stacking and sitting while price stalls, depth ranks HIGH at")
    print(f"  the peak. 'fill volume' should echo check_peak_bar's volume result if")
    print(f"  the futures book and the ETF tape agree.")
    print(f"  'hidden-fill share' is known to measure ~0 (build_mbo_features.py:209),")
    print(f"  so a null there is a property of the FEATURE, not of absorption.")
    print(f"  A '!PLC' mark voids that row. Nothing here can move a deployed rule --")
    print(f"  at 43 and 12 days it decides only whether a live feed is worth buying.")


if __name__ == "__main__":
    main()
