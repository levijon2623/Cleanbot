# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_peak_bar.py
=================
COULD YOU HAVE TOLD, AT THE TIME, THAT THIS NEW HIGH WAS THE LAST ONE?

check_peak_anatomy found the only property that scales with peak size is VOLUME
(0.85x -> 2.09x day-mean from the smallest to the largest peaks; 61st -> 91st
percentile of the day). But volume is elevated throughout a strong move, so
"the peak bar is a high-volume bar" is not yet a detector. This tests it.

THE CONTROL THAT MAKES IT A REAL QUESTION
    Comparing the peak bar against ALL other bars is mechanical: the peak bar
    holds the move's highest high, so anything correlated with making a new high
    (range, and through range, volume) is elevated by construction.
    So the comparison is against the OTHER RUNNING-HIGH BARS of the same move --
    the bars that each set a new running max on the way up (running LOWS for
    puts). Every one of them made a new extreme; the peak bar is simply the last.
    The question becomes exactly the tradeable one: is the FINAL new high
    distinguishable, at the time, from the earlier new highs?

WHAT ABSORPTION LOOKS LIKE IN 1-MINUTE OHLCV
    True absorption needs volume AT PRICE inside the bar -- whether size is
    stacking at the top of the candle while price fails to advance. `historical/`
    carries 1m OHLCV only, so that is NOT available for SPY/QQQ/IWM and is not
    faked here. What OHLCV does support is the effort-vs-result family:
      effort   volume per unit of range -- high volume, little progress
      clspos   where the bar closed within its own range (low = rejected)
      wick     the share of the bar above max(open, close) -- the rejected tail
      volx     volume against the move's own mean
    Real book data exists only for the futures proxies on 43 (RTY) / 12 (NQ)
    days -- see check_peak_mbo.py. That is the pilot; this is the full sample.

PLACEBO (METHODOLOGY 7: placebo SUBJECT, not just a control arm)
    A randomly chosen EARLIER running-high bar, scored through the identical
    pipeline as if it had been the peak. It must return 0.50. Four builds of
    check_peak_levels were saved by exactly this, so it is not optional -- and
    it is re-read inside every subset, because a placebo calibrated in the
    pooled table is not calibrated in a cut of it.

Usage:
  python check_peak_bar.py --tickers SPY QQQ IWM
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

from check_peak_levels import load_levels, peak_rows

RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
FEATS = ("volx", "effort", "clspos", "wick", "rngx", "vjump", "rsi_ext", "rsi_d3")
FLAB = {"volx": "volume vs move mean", "effort": "effort/result (vol per range)",
        "clspos": "close position in bar", "wick": "rejected tail share",
        "rngx": "bar range vs ATR", "vjump": "volume vs prior bar",
        "rsi_ext": "RSI(14) 1m extension", "rsi_d3": "RSI 3-bar change"}


def _rsi(c, n=14):
    """Wilder RSI over an array of closes; NaN through the warmup."""
    out = np.full(c.size, np.nan)
    if c.size < n + 1:
        return out
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, d.size):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        out[i + 1] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def bars_of(tk):
    d = pl.read_parquet(f"historical/{tk}.parquet",
                        columns=["start_time", "open", "high", "low", "close", "volume"]).to_pandas()
    et = (pd.to_datetime(d["start_time"], utc=True)
          .dt.tz_convert("America/New_York").dt.tz_localize(None))
    d["date"] = et.dt.date
    d["mod"] = (et.dt.hour * 60 + et.dt.minute).astype(int)
    d = d[(d["mod"] >= RTH_LO) & (d["mod"] <= RTH_HI)]
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    out = {}
    for dt, g in d.sort_values("mod").groupby("date"):
        cl = g["close"].to_numpy(float)
        out[dt] = dict(mod=g["mod"].to_numpy(np.int32), o=g["open"].to_numpy(float),
                       hi=g["high"].to_numpy(float), lo=g["low"].to_numpy(float),
                       cl=cl, v=g["volume"].to_numpy(float), rsi=_rsi(cl, 14))
    return out


def feats_at(b, k, lo_i, hi_i, atr, up):
    """Features of bar k, measured against the move's own bars [lo_i, hi_i]."""
    hi, lo, cl, o, v = b["hi"][k], b["lo"][k], b["cl"][k], b["o"][k], b["v"][k]
    rng = hi - lo
    mv = b["v"][lo_i:hi_i + 1]
    mvm = np.nanmean(mv) if mv.size else np.nan
    body_top = max(o, cl)
    body_bot = min(o, cl)
    out = {
        "volx": (v / mvm) if mvm and np.isfinite(mvm) and mvm > 0 else np.nan,
        "rngx": rng / atr if atr > 0 else np.nan,
        # effort vs result: size traded per unit of price travelled. High =
        # a lot of volume bought very little movement = the absorption shape.
        "effort": (v / (rng / atr)) / mvm if (rng > 0 and mvm and mvm > 0 and atr > 0) else np.nan,
        # where it closed in its own range; for a CALL peak a LOW value is
        # rejection, so flip puts to keep "high = more rejection" in one direction
        "clspos": ((hi - cl) / rng) if rng > 0 else np.nan,
        "wick": ((hi - body_top) / rng) if rng > 0 else np.nan,
        "vjump": (v / b["v"][k - 1]) if k > 0 and b["v"][k - 1] > 0 else np.nan,
    }
    # RSI, mirrored so "higher = more extended IN THE TRADE'S DIRECTION". A call
    # peak should be overbought; a put trough oversold. Pooling them raw averages
    # the signal away -- check_peak_anatomy first reported a 46.6 median doing
    # exactly that, which is the midpoint of 65.5 and 34.6, not a finding.
    r = b["rsi"]
    rv = r[k] if k < r.size else np.nan
    out["rsi_ext"] = (rv if up else (100.0 - rv)) if np.isfinite(rv) else np.nan
    # and the CHANGE over 3 bars -- a turn is a roll-over, not just a level
    rp = r[k - 3] if k >= 3 else np.nan
    if np.isfinite(rv) and np.isfinite(rp):
        out["rsi_d3"] = (rv - rp) if up else (rp - rv)
    else:
        out["rsi_d3"] = np.nan
    if not up:                       # mirror the two directional shapes
        out["clspos"] = ((cl - lo) / rng) if rng > 0 else np.nan
        out["wick"] = ((body_bot - lo) / rng) if rng > 0 else np.nan
    return out


def run(df, bars, DAY, rng_):
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
        k = s + j                                   # THE peak bar
        # running-extreme bars on the way up (or down) to the peak
        run_idx = []
        cur = -np.inf if up else np.inf
        for t in range(s, k + 1):
            x = b["hi"][t] if up else b["lo"][t]
            if (up and x > cur) or (not up and x < cur):
                cur = x
                run_idx.append(t)
        if len(run_idx) < 6:
            continue                                # need a distribution to rank in
        prior = [t for t in run_idx if t != k]
        pk = feats_at(b, k, s, e - 1, atr, up)
        pri = [feats_at(b, t, s, e - 1, atr, up) for t in prior]
        # PLACEBO: one of the earlier new-high bars, treated as if it were the peak
        pi = int(rng_.integers(0, len(prior)))
        plc = pri[pi]
        plc_pool = [x for q, x in enumerate(pri) if q != pi]

        rec = dict(ticker=r.ticker, dir=r.dir, date=r.date, peak_roe=r.peak_roe,
                   nrun=len(run_idx))
        for f in FEATS:
            pv = pk.get(f, np.nan)
            pool = np.array([x.get(f, np.nan) for x in pri], float)
            pool = pool[np.isfinite(pool)]
            if np.isfinite(pv) and pool.size >= 4:
                rec[f] = midrank(pool, pv)
            qv = plc.get(f, np.nan)
            qpool = np.array([x.get(f, np.nan) for x in plc_pool], float)
            qpool = qpool[np.isfinite(qpool)]
            if np.isfinite(qv) and qpool.size >= 4:
                rec[f + "_plc"] = midrank(qpool, qv)
        rows.append(rec)
    return pd.DataFrame(rows)


def midrank(pool, v):
    """Rank of `v` within `pool`, splitting TIES down the middle.

    A strict `(pool < v).mean()` biases discrete features downward: a value tied
    with half the pool scores 0 for those ties instead of 0.5. On low-cardinality
    columns (spread in ticks, orders at touch, depth) that dragged the PLACEBO to
    0.28-0.43 and voided the rows -- which is how the bug was found. Mid-ranks
    return 0.5 for a random member of any pool, ties or not.
    """
    return float(((pool < v).mean() + (pool <= v).mean()) / 2.0)


def boot(vals, days, n, rng_):
    uniq = pd.unique(days)
    idx = {d: np.where(days == d)[0] for d in uniq}
    out = np.empty(n)
    for i in range(n):
        sel = np.concatenate([idx[d] for d in rng_.choice(uniq, size=len(uniq), replace=True)])
        out[i] = vals[sel].mean()
    return np.percentile(out, [2.5, 97.5])


def report(A, nboot, rng_, title):
    if A.empty or len(A) < 50:
        print(f"\n  {title}: too few rows ({len(A)})")
        return
    print(f"\n  {title}   (n={len(A):,}, {A['date'].nunique()} days)")
    print(f"  {'feature':30} {'n':>6} {'REAL pct':>9} {'95% CI':>17} "
          f"{'PLACEBO':>9} {'95% CI':>17}")
    for f in FEATS:
        g = A.dropna(subset=[f])
        if len(g) < 50:
            continue
        v = g[f].to_numpy(float)
        ci = boot(v, g["date"].to_numpy(), nboot, rng_)
        p = A.dropna(subset=[f + "_plc"])
        pv = p[f + "_plc"].to_numpy(float)
        pci = boot(pv, p["date"].to_numpy(), nboot, rng_)
        flag = "  <--" if (ci[0] > 0.5 or ci[1] < 0.5) else ""
        bad = " !PLC" if (pci[0] > 0.5 or pci[1] < 0.5) else ""
        print(f"  {FLAB[f]:30} {len(g):>6} {v.mean():>9.3f} "
              f"[{ci[0]:>6.3f},{ci[1]:>6.3f}] {pv.mean():>9.3f} "
              f"[{pci[0]:>6.3f},{pci[1]:>6.3f}]{flag}{bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="+", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng_ = np.random.default_rng(a.seed)

    df = peak_rows(a.tickers, a.dirs, a.pct)
    if df.empty:
        print("  no trades"); return
    DAY, _ = load_levels(a.tickers)
    bars = {tk: bars_of(tk) for tk in a.tickers}
    A = run(df, bars, DAY, rng_)
    if A.empty:
        print("  nothing scored"); return
    A.to_parquet("_peak_bar.parquet", index=False)

    print(f"\n{'='*98}")
    print(f"  IS THE FINAL NEW HIGH DISTINGUISHABLE FROM THE EARLIER ONES?")
    print(f"  pct = the peak bar's rank among the move's OTHER running-extreme bars.")
    print(f"  0.5 = indistinguishable. PLACEBO = an earlier new-high bar scored the")
    print(f"  same way; it must sit at 0.50 or the column is not readable.")
    print(f"{'='*98}")
    report(A, a.boot, rng_, "ALL TRADES")
    for lbl, lo, hi in (("+100..200%", 1.0, 2.0), (">+200%", 2.0, 99)):
        report(A[(A["peak_roe"] >= lo) & (A["peak_roe"] < hi)], a.boot, rng_,
               f"PEAK ROE {lbl}")
    for dr in sorted(A["dir"].unique()):
        report(A[A["dir"] == dr], a.boot, rng_, f"{dr} ONLY")

    print(f"\n  HOW TO READ IT")
    print(f"  'effort/result' is the absorption proxy: volume per unit of range. If the")
    print(f"  peak bar ranks high there, a lot of size bought very little progress --")
    print(f"  the shape of size being absorbed. 'rejected tail share' is the wick above")
    print(f"  the body (below it, for puts).")
    print(f"  A '!PLC' mark means the placebo missed 0.50 and that ROW IS NOT READABLE,")
    print(f"  whatever the real column says. Check it per subset, not just pooled.")
    print(f"  NOTE: volume AT PRICE inside the bar is not in 1m OHLCV, so true")
    print(f"  absorption is only approximated here. check_peak_mbo.py has the book.")


if __name__ == "__main__":
    main()
