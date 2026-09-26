# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_hour_drift.py
===================
Adjudicates the hour-10 question at the UNDERLYING level, where option
mechanics (theta, TP-capture, time budget) cannot reach.

check_hour10 left a contradiction: on the deployed rules hour 10 is
spectacular and IS/OOS-stable (+31.7/+33.6, 6/7 rules, day-bootstrap 100th
pct), but it loses its lead under a matched time stop and shows nothing in the
broad 12-ticker trigger population. Both failing tests are about the OPTION.
The market-structure claim -- opening imbalance clears and institutional/VWAP
algos engage ~10:00, so flow then carries more information -- is a claim about
the STOCK. So test the stock.

Per entry hour, over all triggers (>= trailing-60d p50) on 12 tickers:

  A. DIRECTION-ADJUSTED UNDERLYING DRIFT to the close
       sgn * (close/spot_at_trigger - 1), sgn = +1 CALL / -1 PUT.
     Pure delta signal. No premium, no theta, no take-profit.
  B. MATCHED-HORIZON DRIFT (+30 / +60 / +120 min)
     The same control that broke the option result, applied to the stock. If
     hour 10 leads at a FIXED horizon, the effect is real and structural.
  C. FLOW INTENSITY by hour, and each hour vs the PRIOR hour
     Is 10:00 actually where the size shows up? (the mechanism the claim needs)
  D. ATM IV DRIFT by hour  (from _atm_iv_cache)
     Vega is the other thing bundled into option P&L: if IV systematically
     rises after 10:00 entries and falls after 14:00 entries, part of the "hour
     edge" is a vega tailwind, not direction.

All stats are DAY-LEVEL (one observation per ticker-date-hour) so intraday
clustering cannot inflate a bucket.

Usage:  python check_hour_drift.py [--tickers ...]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
CLOSE_MOD = 15 * 60 + 55
TICKERS = ["AMZN", "AVGO", "GLD", "IWM", "META", "MSFT", "NVDA", "QQQ",
           "SPY", "TSLA", "AAPL", "GOOGL"]


def _spot_by_day(tk):
    import polars as pl
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return {}
    d = pl.read_parquet(p, columns=["start_time", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 960)].sort_values(["date", "mod"])
    return {dt: (g["mod"].to_numpy(), g["close"].to_numpy(float)) for dt, g in d.groupby("date")}


def _at(arr, m):
    mm, vv = arr
    i = int(np.searchsorted(mm, m, side="right")) - 1
    return float(vv[i]) if i >= 0 else np.nan


def _iv_by_day(tk):
    import polars as pl
    p = f"_atm_iv_cache/{tk}.parquet"
    if not os.path.exists(p):
        return {}
    d = pl.read_parquet(p).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    out = {}
    for dt, g in d.groupby("date"):
        g = g.sort_values("mod15")
        out[dt] = (g["mod15"].to_numpy(), g["iv_close"].to_numpy(float))
    return out


def _cell(sub, col):
    v = sub[col].dropna()
    if len(v) < 25:
        return f"{'':>22}"
    i = sub[sub.date < SPLIT][col].dropna()
    o = sub[sub.date >= SPLIT][col].dropna()
    return (f"{v.mean()*1e4:>+6.1f}bp(IS{i.mean()*1e4 if len(i) else float('nan'):>+5.1f}"
            f"/OOS{o.mean()*1e4 if len(o) else float('nan'):>+5.1f})")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    tickers = [t.upper() for t in (a.tickers or TICKERS)]
    rows = []
    for tk in tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        spot = _spot_by_day(tk)
        ivd = _iv_by_day(tk)
        if not spot:
            continue
        for t in trigs:
            thr = t.get("thr")
            if not thr or t["abs_flow"] < thr.get(50, 1e99):
                continue
            d, ts = t["date"], t["ts"]
            arr = spot.get(d)
            if arr is None:
                continue
            m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
            s0 = _at(arr, m)
            if not np.isfinite(s0) or s0 <= 0:
                continue
            sgn = 1.0 if t["dir"] == "CALL" else -1.0
            rec = dict(ticker=tk, date=d, hour=m // 60, mod=m, sgn=sgn,
                       flow=float(t["abs_flow"]))
            rec["d_close"] = sgn * (_at(arr, CLOSE_MOD) / s0 - 1.0)
            for H in (30, 60, 120):
                if m + H <= CLOSE_MOD:
                    rec[f"d{H}"] = sgn * (_at(arr, m + H) / s0 - 1.0)
                else:
                    rec[f"d{H}"] = np.nan
            iv = ivd.get(d)
            if iv is not None:
                b0 = m - (m % 15)
                i0 = _at(iv, b0)
                if np.isfinite(i0) and i0 > 0:
                    ic = _at(iv, CLOSE_MOD - (CLOSE_MOD % 15))
                    i60 = _at(iv, (m + 60) - ((m + 60) % 15)) if m + 60 <= CLOSE_MOD else np.nan
                    rec["iv0"] = i0
                    rec["div_close"] = (ic / i0 - 1.0) if np.isfinite(ic) else np.nan
                    rec["div_60"] = (i60 / i0 - 1.0) if np.isfinite(i60) else np.nan
            rows.append(rec)

    R = pd.DataFrame(rows)
    if R.empty:
        print("no triggers"); return
    # de-cluster: one observation per ticker-date-hour
    num = [c for c in ("d_close", "d30", "d60", "d120", "div_close", "div_60", "flow", "iv0")
           if c in R.columns]
    G = R.groupby(["ticker", "date", "hour"])[num].mean().reset_index()
    G["date"] = pd.to_datetime(G["date"]).dt.date

    print("=" * 118)
    print(f"  ENTRY-HOUR / UNDERLYING-DRIFT   {len(R)} triggers -> {len(G)} ticker-day-hours   "
          f"{len(tickers)} tickers   split {SPLIT}")
    print("=" * 118)

    print("\n  -- A/B. DIRECTION-ADJUSTED UNDERLYING DRIFT by entry hour --")
    print(f"    {'hr':>3} {'n':>5}  {'to close':>22}  {'+30m':>22}  {'+60m':>22}  {'+120m':>22}")
    for h in sorted(G["hour"].unique()):
        s = G[G["hour"] == h]
        if len(s) < 30:
            continue
        print(f"    {int(h):>3} {len(s):>5}  {_cell(s,'d_close'):>22}  {_cell(s,'d30'):>22}  "
              f"{_cell(s,'d60'):>22}  {_cell(s,'d120'):>22}")
    print("    (matched horizons +30/+60/+120 are the control: same opportunity for every hour)")

    print("\n  -- per-ticker: which hour has the best +60m drift? --")
    tally = {}
    for tk, s in G.groupby("ticker"):
        best, bv = None, -9
        for h in sorted(s["hour"].unique()):
            g = s[(s["hour"] == h)]["d60"].dropna()
            if len(g) < 25:
                continue
            if g.mean() > bv:
                best, bv = int(h), g.mean()
        if best is not None:
            tally[tk] = (best, bv * 1e4)
    for tk, (h, v) in sorted(tally.items()):
        print(f"    {tk:6} best hour {h}  ({v:+.1f}bp)")
    from collections import Counter
    c = Counter(h for h, _ in tally.values())
    print(f"    -> hour histogram across tickers: {dict(sorted(c.items()))}")

    print("\n  -- C. FLOW INTENSITY by hour (per-ticker z-scored), and vs the PRIOR hour --")
    G["fz"] = G.groupby("ticker")["flow"].transform(lambda x: (x - x.mean()) / (x.std() or 1))
    prev = None
    for h in sorted(G["hour"].unique()):
        s = G[G["hour"] == h]
        if len(s) < 30:
            continue
        mu = s["fz"].mean()
        rel = f"{(s['flow'].mean()/prev - 1)*100:>+6.1f}% vs prior hr" if prev else " " * 20
        print(f"    hour {int(h):>2}  n={len(s):>5}  flow z {mu:>+5.2f}  "
              f"mean ${s['flow'].mean()/1e6:>6.2f}M  {rel}")
        prev = s["flow"].mean()

    if "div_close" in G.columns:
        print("\n  -- D. ATM IV DRIFT by hour (the vega confound) --")
        print(f"    {'hr':>3} {'n':>5}  {'IV chg to close':>22}  {'IV chg +60m':>22}  {'IV level':>9}")
        for h in sorted(G["hour"].unique()):
            s = G[G["hour"] == h].dropna(subset=["div_close"])
            if len(s) < 25:
                continue
            print(f"    {int(h):>3} {len(s):>5}  {_cell(s,'div_close'):>22}  {_cell(s,'div_60'):>22}  "
                  f"{s['iv0'].mean():>8.3f}")
        print("    (IV chg in bp of RELATIVE IV change; a long option is LONG vega, so"
              " positive = tailwind)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    a = ap.parse_args()
    run(a)
