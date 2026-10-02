# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27.0", "polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "python-dotenv"]
# ///
"""
check_rr_skew.py
================

Does the 25-delta RISK-REVERSAL SKEW (put_IV - call_IV at ~30 DTE) carry any
forward signal for the rule tickers?  A rising RR = fresh downside-hedging demand
= fear; a flattening / inverting RR = risk-on / call demand.

Data: /stock/{t}/historical-risk-reversal-skew?delta=25&expiry=E -- returns the
RR time series for ONE monthly expiry.  We stitch: for each trading day, take the
RR of whichever monthly expiry is closest to --target-dte (within a window).
UW only serves this back to ~2025-09, so there is ~1 YEAR of data -- NO IS/OOS
split is possible; treat everything here as hypothesis-generating, not validated.

Per ticker:
  spearman(RR level, fwd Nd underlying return)   -- is fear priced-in bullish/bearish?
  spearman(RR change 5d, fwd Nd return)          -- does steepening lead a drop?
  forward-return by RR-level tercile, vs the drift-free up/down-range baseline

Usage:
  python check_rr_skew.py --tickers META MSFT NVDA SPY QQQ IWM AVGO GLD AMZN TSLA
  python check_rr_skew.py --fwd 10 --target-dte 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd
import polars as pl
import httpx
from dotenv import load_dotenv

load_dotenv()
API = "https://api.unusualwhales.com/api"
HDRS = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json",
        "User-Agent": "webullrg/1.0"}
HIST = "historical"
CACHE = "_rr_cache"
RULE_TICKERS = ["META", "MSFT", "NVDA", "SPY", "QQQ", "IWM", "AVGO", "GLD", "AMZN", "TSLA"]


def _third_fridays(start: dt.date, end: dt.date):
    out, y, m = [], start.year, start.month
    while dt.date(y, m, 1) <= end:
        d = dt.date(y, m, 1)
        d += dt.timedelta(days=(4 - d.weekday()) % 7)     # first Friday
        d += dt.timedelta(days=14)                        # third Friday
        if start <= d <= end:
            out.append(d)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _rr_series(tk: str, target_dte: int, force=False) -> pd.Series:
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{tk}_dte{target_dte}.parquet")
    if os.path.exists(fp) and not force:
        return pd.read_parquet(fp)["rr"]
    exps = _third_fridays(dt.date(2025, 8, 1), dt.date.today() + dt.timedelta(days=60))
    frames = []
    for e in exps:
        try:
            r = httpx.get(f"{API}/stock/{tk}/historical-risk-reversal-skew",
                          headers=HDRS, params={"delta": "25", "expiry": e.isoformat()}, timeout=30)
            rows = r.json().get("data", []) if r.status_code == 200 else []
        except httpx.HTTPError:
            rows = []
        for x in rows:
            d = dt.date.fromisoformat(x["date"])
            frames.append((d, e, (e - d).days, float(x["risk_reversal"])))
    if not frames:
        return pd.Series(dtype=float)
    df = pd.DataFrame(frames, columns=["date", "expiry", "dte", "rr"])
    df = df[(df["dte"] >= 7) & (df["dte"] <= 75)]
    df["gap"] = (df["dte"] - target_dte).abs()
    df = df.sort_values(["date", "gap"]).drop_duplicates("date", keep="first")
    s = df.set_index("date")["rr"].sort_index()
    s.index = pd.to_datetime(s.index)
    pd.DataFrame({"rr": s}).to_parquet(fp)
    return s


def _underlying(tk):
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    df = pl.read_parquet(p).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York")
    mo = et.dt.hour * 60 + et.dt.minute
    g = df.assign(d=et.dt.date)[(mo >= 570) & (mo <= 960)].groupby("d").agg(
        close=("close", "last"), high=("high", "max"), low=("low", "min"))
    g.index = pd.to_datetime(g.index)
    return g.sort_index()


def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def run(args):
    print("=" * 96)
    print(f"  RISK-REVERSAL SKEW (25d, ~{args.target_dte} DTE)  ->  fwd {args.fwd}d underlying return")
    print(f"  ~1yr of data (2025-09+), NO IS/OOS -- hypothesis-generating only")
    print("=" * 96)
    print(f"  {'ticker':7}{'n':>5}{'RR now':>9}{'RR mean':>9}{'sp(RR,fwd)':>12}"
          f"{'sp(dRR5,fwd)':>13}{'Q3-Q1 fwd% (RR hi-lo)':>22}{'vs base':>9}")
    for tk in args.tickers:
        rr = _rr_series(tk, args.target_dte, force=args.rebuild)
        ud = _underlying(tk)
        if rr.empty or ud is None:
            print(f"  {tk:7}  no data"); continue
        idx = sorted(set(rr.index) & set(ud.index))
        d = pd.DataFrame(index=idx)
        d["rr"] = rr.reindex(idx)
        c = ud["close"].reindex(idx)
        d["fwd"] = c.shift(-args.fwd) / c - 1
        d["drr5"] = d["rr"].diff(5)
        d = d.dropna(subset=["rr", "fwd"])
        if len(d) < 40:
            print(f"  {tk:7} n={len(d):>3}  (thin)"); continue
        q = pd.qcut(d["rr"].rank(method="first"), 3, labels=False)
        q3 = d["fwd"][q == 2].mean() * 100
        q1 = d["fwd"][q == 0].mean() * 100
        base = d["fwd"].mean() * 100
        print(f"  {tk:7}{len(d):>5}{d['rr'].iloc[-1]:>+9.3f}{d['rr'].mean():>+9.3f}"
              f"{_spear(d['rr'], d['fwd']):>+12.2f}{_spear(d['drr5'], d['fwd']):>+13.2f}"
              f"{q3 - q1:>+18.2f}pp{q3 - base:>+7.1f}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=RULE_TICKERS)
    ap.add_argument("--fwd", type=int, default=10)
    ap.add_argument("--target-dte", type=int, default=30)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    a.tickers = [t.upper() for t in a.tickers]
    run(a)
