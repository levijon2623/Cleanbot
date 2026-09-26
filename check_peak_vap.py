# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_peak_vap.py
=================
AT THE PEAK MINUTE, WHERE IN THE CANDLE DID THE VOLUME TRADE -- AND WAS ANY OF
IT HIDDEN?

THE HYPOTHESIS BEING TESTED, stated as the user put it
    "if volume is climbing at the upper end of the candle, but price isn't
     moving, that would indicate absorption"
    check_peak_bar could only see the SHAPE of that (volume per unit of range
    ranks 0.65 at the peak vs a 0.49 placebo). It could not see WHERE in the bar
    the volume sat, because 1m OHLCV does not carry it. build_vap.py rebuilds
    volume-at-price from raw MBO, so this can finally be asked directly:
        top25     share of the minute's volume in the TOP quarter of its range
        vap_pos   volume-weighted position of trades within the range
        buy_top   buy-aggressor share of that top-quarter volume  <-- the crux:
                  heavy BUYING at the top that fails to lift price IS absorption
        hid_top   share of top-quarter volume that was never displayed
                  (the iceberg test, per-order, not the level turnover ratio)

WHAT SEPARATES THE TWO ICEBERG NUMBERS
    ice_ratio / ice_max are LEVEL TURNOVER: executed volume over the largest
    depth ever shown at that price. A level that displays 20 and trades 200 has
    a ratio of 10 with nothing hidden at all, if ten honest orders replenished
    it. Reported for context, never as an iceberg claim.
    hid_share / hid_top ARE the iceberg measure: an execution larger than the
    displayed size OF THAT ORDER, so the excess was never on the book.
    (Same definition as build_mbo_features.absorb, which fires on 27.9% of RTY
    minutes -- that feature is NOT the dud its own stale docstring claims.)

=========================  READ THE POWER FIRST  ==========================
  RTY 44 days -> IWM proxy     NQ 14 days -> QQQ proxy     SPY: no MBO at all
METHODOLOGY 7: power is set by DAYS. This is pilot scale and cannot promote or
retire anything. It decides only whether volume-at-price is worth paying for.
The book is FUTURES; the peaks are ETF OPTION peaks. A tell here still has to
survive on the ETF.

DESIGN -- identical to check_peak_bar / check_peak_mbo so all three compare
    The peak minute is ranked against the move's OTHER RUNNING-EXTREME minutes:
    each of them set a new high (low); the peak is simply the last one. Ranks
    use MID-RANKS -- a strict `<` scores ties as 0 and dragged the placebos to
    0.28-0.43 on the first MBO run, voiding exactly the discrete columns that
    mattered. PLACEBO: an earlier running-extreme minute scored as if it were
    the peak; it must return 0.50, and it is re-read in every subset.

Usage:
  python check_peak_vap.py
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
CACHE = "_vap_cache"
FEATS = ("top25", "vap_pos", "buy_top", "hid_top", "hid_share", "sell_bot",
         "bot25", "vol_hhi", "ice_ratio", "ice_max", "tvol", "signed_vol")
LAB = {"top25": "volume in TOP quarter", "vap_pos": "volume position in range",
       "buy_top": "buy-aggressor share, top qtr", "hid_top": "HIDDEN share, top qtr",
       "hid_share": "HIDDEN share, whole bar", "sell_bot": "sell-aggr share, bottom qtr",
       "bot25": "volume in BOTTOM quarter", "vol_hhi": "volume concentration",
       "ice_ratio": "level turnover (not iceberg)", "ice_max": "worst-level turnover",
       "tvol": "total volume", "signed_vol": "signed volume (buy-sell)"}


def load_vap():
    out = {}
    for tk, sym in PROXY.items():
        fp = os.path.join(CACHE, f"{sym}.parquet")
        if not os.path.exists(fp):
            continue
        d = pl.read_parquet(fp).to_pandas()
        d["date"] = pd.to_datetime(d["date"]).dt.date
        out[tk] = {dt: g.set_index("mod") for dt, g in d.groupby("date")}
        print(f"  {tk} <- {sym}: {d['date'].nunique()} days, {len(d):,} minute rows")
    return out


def run(df, bars, DAY, vap, rng_):
    rows = []
    for r in df.itertuples():
        V = vap.get(r.ticker, {}).get(r.date)
        b = bars.get(r.ticker, {}).get(r.date)
        if V is None or b is None or DAY.get(r.ticker, {}).get(r.date) is None:
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
        pk_mod = int(mods[k])
        have = [m for m in run_mods if m in V.index]
        if pk_mod not in V.index or len(have) < 6:
            continue
        prior = [m for m in have if m != pk_mod]
        if len(prior) < 5:
            continue
        pi = int(rng_.integers(0, len(prior)))
        plc_mod = prior[pi]
        plc_pool = [m for q, m in enumerate(prior) if q != pi]

        rec = dict(ticker=r.ticker, dir=r.dir, date=r.date, peak_roe=r.peak_roe)
        for f in FEATS:
            if f not in V.columns:
                continue
            # For a PUT the move is downward, so the absorbing side is the
            # BOTTOM of the candle. Mirror the two directional columns so
            # "high rank = more absorption of our move" reads one way for both.
            src = f
            if not up:
                src = {"top25": "bot25", "bot25": "top25",
                       "buy_top": "sell_bot", "sell_bot": "buy_top",
                       "hid_top": "hid_bot"}.get(f, f)
                if src not in V.columns:
                    continue
            pv = V.at[pk_mod, src] if pk_mod in V.index else np.nan
            if f == "vap_pos" and not up and np.isfinite(pv):
                pv = 1.0 - pv
            pool = pd.to_numeric(V.loc[prior, src], errors="coerce").dropna().to_numpy()
            if not up and f == "vap_pos":
                pool = 1.0 - pool
            if np.isfinite(pv) and pool.size >= 4:
                rec[f] = midrank(pool, pv)
            qv = V.at[plc_mod, src] if plc_mod in V.index else np.nan
            if f == "vap_pos" and not up and np.isfinite(qv):
                qv = 1.0 - qv
            qpool = pd.to_numeric(V.loc[plc_pool, src], errors="coerce").dropna().to_numpy()
            if not up and f == "vap_pos":
                qpool = 1.0 - qpool
            if np.isfinite(qv) and qpool.size >= 4:
                rec[f + "_plc"] = midrank(qpool, qv)
        rows.append(rec)
    return pd.DataFrame(rows)


def report(A, nboot, rng_, title, minn=40):
    if A.empty or len(A) < minn:
        print(f"\n  {title}: too few rows ({len(A)})")
        return
    print(f"\n  {title}   (n={len(A):,}, {A['date'].nunique()} DAYS)")
    print(f"  {'feature':32} {'n':>6} {'REAL':>8} {'95% CI':>17} "
          f"{'PLACEBO':>8} {'95% CI':>17}")
    for f in FEATS:
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
        print(f"  {LAB.get(f, f):32} {len(g):>6} {v.mean():>8.3f} "
              f"[{ci[0]:>6.3f},{ci[1]:>6.3f}] {np.nanmean(pv):>8.3f} "
              f"[{pci[0]:>6.3f},{pci[1]:>6.3f}]{flag}{bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    rng_ = np.random.default_rng(a.seed)

    vap = load_vap()
    if not vap:
        print("  no VAP cache -- run build_vap.py first"); return
    tickers = list(PROXY)
    df = peak_rows(["SPY", "QQQ", "IWM"], ["CALL", "PUT"], a.pct)
    df = df[df["ticker"].isin(tickers)]
    DAY, _ = load_levels(tickers)
    bars = {tk: bars_of(tk) for tk in tickers}
    A = run(df, bars, DAY, vap, rng_)
    if A.empty:
        print("  nothing scored"); return
    A.to_parquet("_peak_vap.parquet", index=False)

    print(f"\n{'='*100}")
    print(f"  VOLUME AT PRICE AT THE PEAK MINUTE vs THE MOVE'S EARLIER NEW EXTREMES")
    print(f"  Directional columns are MIRRORED for puts, so 'top quarter' always")
    print(f"  means the leading edge of OUR move. PILOT: {A['date'].nunique()} days.")
    print(f"{'='*100}")
    report(A, a.boot, rng_, "ALL")
    for tk, g in A.groupby("ticker"):
        report(g, a.boot, rng_, f"{tk} -> {PROXY[tk]}")
    report(A[A["peak_roe"] >= 1.0], a.boot, rng_, "LARGE PEAKS (ROE >= +100%)")

    print(f"\n  HOW TO READ IT")
    print(f"  ABSORPTION, as hypothesised, needs BOTH: volume piled at the leading")
    print(f"  edge of the candle ('volume in TOP quarter' high) AND that volume being")
    print(f"  aggressive in OUR direction ('buy-aggressor share, top qtr' high) while")
    print(f"  price fails to advance. One without the other is just a busy minute.")
    print(f"  'HIDDEN share' is the iceberg leg: was the size doing the absorbing")
    print(f"  never displayed? 'level turnover' is NOT an iceberg number -- honest")
    print(f"  replenishment produces large ratios with nothing hidden.")
    print(f"  A '!PLC' mark voids that row.")


if __name__ == "__main__":
    main()
