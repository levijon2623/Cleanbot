# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27.0", "numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0", "python-dotenv"]
# ///
"""
check_cor1m.py
===============
Test the "popular" claim with the REAL Cboe 1-Month Implied Correlation Index
(COR1M, user-supplied CSV, 2024-08-20..2026-08-21, 511 days):

    "COR1M trending < 8  =>  bearish mega-cap"

COR1M range in-sample: min 3.44, p25 10.1, median 13.3, p75 18.8, max 49.9.
So "< 8" is roughly the bottom ~15-20% of readings (and Aug-2026 is living there).

Tests, IS/OOS @ 2025-08-21, targets = fwd QQQ return and fwd cap-wtd mega-basket
return (NVDA AAPL MSFT GOOGL AMZN META AVGO TSLA):
  - spearman(COR1M level, fwd ret)            -- does the level lead?
  - spearman(COR1M 5d change, fwd ret)        -- the "trending" part
  - event study: fwd ret after COR1M < 8  vs  >= 8  vs  all
  - "trending < 8": COR1M < 8 AND 5d change < 0
  - CONTROL: same vs VIX and vs SPY 30d IV -- does COR1M add anything beyond index vol?

Usage:
  python check_cor1m.py
  python check_cor1m.py --fwd 5 10 20 30 --low 8
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl
import httpx
from dotenv import load_dotenv

load_dotenv()
API = "https://api.unusualwhales.com/api"
HDRS = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json",
        "User-Agent": "cleanbot/1.0"}
HIST = "historical"
CSV = "CBOE 1-Month Implied Correlation Historical Data.csv"
BASKET = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "TSLA"]
SPLIT = pd.Timestamp("2025-08-21")


def _daily_close(tk):
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).to_pandas()
    d.columns = [c.lower() for c in d.columns]
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York")
    mo = et.dt.hour * 60 + et.dt.minute
    m = (mo >= 570) & (mo <= 960)
    s = pd.DataFrame({"date": et[m].dt.date.values, "c": d["close"][m].astype(float).values})
    return s.groupby("date")["c"].last()


def _vix():
    """VIX close from /stock/VIX/volatility/realized 'price' (~1yr) + full-history
    VIXY (ETF proxy, /ohlc/1d) as a longer but noisier stand-in."""
    out = {}
    try:
        r = httpx.get(f"{API}/stock/VIX/volatility/realized", headers=HDRS, timeout=30)
        rows = r.json().get("data", []) if r.status_code == 200 else []
        for x in rows:
            out[pd.to_datetime(x["date"]).date()] = float(x["price"])
    except (httpx.HTTPError, KeyError, ValueError):
        pass
    if out:
        s = pd.Series(out).sort_index()
        return s, "VIX"
    return None, None


def _spy_iv():
    try:
        r = httpx.get(f"{API}/stock/SPY/volatility/realized", headers=HDRS, timeout=30)
        rows = r.json().get("data", []) if r.status_code == 200 else []
    except httpx.HTTPError:
        rows = []
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d["iv"] = pd.to_numeric(d["implied_volatility"], errors="coerce")
    return d.groupby("date")["iv"].last()


def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 25:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def run(a):
    cor = pd.read_csv(CSV)
    cor["date"] = pd.to_datetime(cor["Date"]).dt.date
    cor["cor1m"] = pd.to_numeric(cor["Price"], errors="coerce")
    cor = cor.set_index("date")["cor1m"].sort_index()
    cor.index = pd.to_datetime(cor.index)
    print(f"  COR1M: {len(cor)} days  {cor.index.min().date()}..{cor.index.max().date()}  "
          f"min {cor.min():.1f}  p25 {cor.quantile(.25):.1f}  med {cor.median():.1f}  "
          f"p75 {cor.quantile(.75):.1f}  max {cor.max():.1f}")

    caps_src = {}
    import json
    with open("_universe_cache/info.json") as f:
        info = json.load(f)
    caps = {t: float(info[t]["marketcap"]) for t in BASKET if info.get(t, {}).get("marketcap")}
    w = pd.Series(caps) / sum(caps.values())

    closes = {t: _daily_close(t) for t in ["QQQ"] + BASKET}
    idx = cor.index
    for t, s in closes.items():
        if s is None:
            print(f"  WARN no history for {t}")
            continue
        s.index = pd.to_datetime(s.index)
        idx = idx.intersection(s.index)
    qqq = closes["QQQ"].reindex(idx)
    basket = (pd.DataFrame({t: closes[t].reindex(idx) for t in w.index})
              .pct_change().mul(w).sum(axis=1).add(1).cumprod())

    df = pd.DataFrame({"cor1m": cor.reindex(idx), "qqq": qqq, "basket": basket})
    df["dcor5"] = df["cor1m"].diff(5)
    vix, vsrc = _vix()
    if vix is not None:
        vix.index = pd.to_datetime(vix.index)
        df["vix"] = vix.reindex(idx)
        print(f"  VIX control: {vsrc}, {df['vix'].notna().sum()} days")
    spy_iv = _spy_iv()
    if spy_iv is not None:
        spy_iv.index = pd.to_datetime(spy_iv.index)
        df["spy_iv"] = spy_iv.reindex(idx)
        print(f"  SPY-IV control: {df['spy_iv'].notna().sum()} days (~1yr)")
    print(f"  {len(df)} aligned trading days\n")

    for tgt, nm in [("qqq", "QQQ"), ("basket", "mega-basket")]:
        print(f"  ===  forward {nm} return  ===")
        print(f"  {'':13}" + "".join(f"{h}d".rjust(9) for h in a.fwd))
        for lbl, sub in [("FULL", df), ("IS", df[df.index < SPLIT]), ("OOS", df[df.index >= SPLIT])]:
            def sprow(col):
                return "".join(f"{_spear(sub[col], sub[tgt].shift(-h) / sub[tgt] - 1):>+9.3f}" for h in a.fwd)
            print(f"    {lbl:4} sp(COR1M) " + sprow("cor1m"))
            print(f"    {lbl:4} sp(dCOR5) " + sprow("dcor5"))
            if "vix" in sub:
                print(f"    {lbl:4} sp(VIX)   " + sprow("vix"))
            if "spy_iv" in sub and sub["spy_iv"].notna().sum() > 25:
                print(f"    {lbl:4} sp(SPYiv) " + sprow("spy_iv"))
        print()

    print(f"  ===  EVENT STUDY: mean forward QQQ return by COR1M state  ===")
    print(f"  {'state':22}{'n':>5}" + "".join(f"{h}d".rjust(9) for h in a.fwd))
    lo = a.low
    p20 = df["cor1m"].quantile(.2)
    states = [
        (f"COR1M < {lo}", df["cor1m"] < lo),
        (f"COR1M < {lo} & falling", (df["cor1m"] < lo) & (df["dcor5"] < 0)),
        (f"COR1M >= {lo}", df["cor1m"] >= lo),
        (f"bottom quintile (<{p20:.1f})", df["cor1m"] <= p20),
        ("top quintile", df["cor1m"] >= df["cor1m"].quantile(.8)),
        ("ALL", pd.Series(True, index=df.index)),
    ]
    for lbl, mask in states:
        row = "".join(f"{((df['qqq'].shift(-h) / df['qqq'] - 1)[mask].mean()*100):>+8.2f}%" for h in a.fwd)
        print(f"  {lbl:22}{int(mask.sum()):>5}{row}")

    print(f"\n  ===  same event study, mega-BASKET  ===")
    print(f"  {'state':22}{'n':>5}" + "".join(f"{h}d".rjust(9) for h in a.fwd))
    for lbl, mask in states:
        row = "".join(f"{((df['basket'].shift(-h) / df['basket'] - 1)[mask].mean()*100):>+8.2f}%" for h in a.fwd)
        print(f"  {lbl:22}{int(mask.sum()):>5}{row}")

    df.to_parquet("_cor1m.parquet")
    print(f"\n  wrote _cor1m.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fwd", nargs="+", type=int, default=[5, 10, 20, 30])
    ap.add_argument("--low", type=float, default=8.0)
    a = ap.parse_args()
    run(a)
