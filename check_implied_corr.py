# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27.0", "numpy>=1.26.0", "pandas>=2.0.0", "python-dotenv"]
# ///
"""
check_implied_corr.py
======================
"Popular" claim: implied correlation trending very low  =>  bearish mega-cap
(a narrow / dispersion-crowded market that's fragile to a correlated selloff).

UW has no COR1M endpoint, but /stock/{t}/volatility/realized gives ~1yr of daily
30-day ATM implied vol + close price per ticker.  Build an implied-correlation
PROXY from the index leg vs a cap-weighted mega-cap single-stock leg:

    rho_proxy = ( sigma_index / sum_i w_i * sigma_i ) ** 2        (Cboe-style)

low rho_proxy = stocks dispersing (moving independently);  high (-> 1) = moving
together.  Then test whether the LEVEL or the 5d CHANGE of rho_proxy predicts
forward QQQ / mega-basket returns.

Only ~1yr of data (2025-09+), single-ish regime -- hypothesis check, not validated.

Usage:
  python check_implied_corr.py
  python check_implied_corr.py --fwd 5 10 20
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import httpx
from dotenv import load_dotenv

load_dotenv()
API = "https://api.unusualwhales.com/api"
HDRS = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json",
        "User-Agent": "cleanbot/1.0"}

IDX = ["SPY", "QQQ"]
BASKET = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "NFLX", "PLTR"]


def _ivseries(tk):
    r = httpx.get(f"{API}/stock/{tk}/volatility/realized", headers=HDRS, timeout=30)
    rows = r.json().get("data", []) if r.status_code == 200 else []
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"])
    d["iv"] = pd.to_numeric(d["implied_volatility"], errors="coerce")
    d["px"] = pd.to_numeric(d["price"], errors="coerce")
    return d.set_index("date")[["iv", "px"]].sort_index()


def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def run(a):
    import json
    with open("_universe_cache/info.json") as f:
        info = json.load(f)
    caps = {t: float(info[t]["marketcap"]) for t in BASKET if info.get(t, {}).get("marketcap")}
    w = pd.Series(caps) / sum(caps.values())
    print(f"  basket weights: " + ", ".join(f"{t} {w[t]*100:.0f}%" for t in w.index))

    iv = {}
    for tk in IDX + BASKET:
        s = _ivseries(tk)
        if s is not None:
            iv[tk] = s
    idx_common = None
    for tk in iv:
        idx_common = iv[tk].index if idx_common is None else idx_common.intersection(iv[tk].index)
    print(f"  {len(idx_common)} common trading days  ({idx_common.min().date()} .. {idx_common.max().date()})")

    stock_iv = pd.DataFrame({t: iv[t]["iv"].reindex(idx_common) for t in w.index})
    wavg_iv = (stock_iv * w).sum(axis=1)                     # cap-weighted single-stock IV
    basket_px = (pd.DataFrame({t: iv[t]["px"].reindex(idx_common) for t in w.index})
                 .pct_change().mul(w).sum(axis=1).add(1).cumprod())   # cap-weighted basket index

    out = pd.DataFrame(index=idx_common)
    for leg in IDX:
        rho = (iv[leg]["iv"].reindex(idx_common) / wavg_iv) ** 2
        out[f"rho_{leg}"] = rho
        out[f"drho_{leg}"] = rho.diff(5)
    out["spy_iv"] = iv["SPY"]["iv"].reindex(idx_common)     # control: index vol alone
    out["basket_iv"] = wavg_iv                               # control: single-stock vol alone
    out["qqq_px"] = iv["QQQ"]["px"].reindex(idx_common)
    out["basket_px"] = basket_px

    print(f"\n  rho_proxy distribution (SPY leg):")
    q = out["rho_SPY"].quantile([0, .1, .25, .5, .75, .9, 1])
    print("   " + "  ".join(f"p{int(k*100)}={v:.3f}" for k, v in q.items()))
    print(f"  rho_proxy distribution (QQQ leg):")
    q = out["rho_QQQ"].quantile([0, .1, .25, .5, .75, .9, 1])
    print("   " + "  ".join(f"p{int(k*100)}={v:.3f}" for k, v in q.items()))

    for tgt, nm in [("qqq_px", "QQQ"), ("basket_px", "mega-basket")]:
        c = out[tgt]
        print(f"\n  ===  forward {nm} return -- spearman vs each predictor  ===")
        print(f"  {'fwd':>5} | {'rho_SPY':>9}{'rho_QQQ':>9} | {'spy_iv':>9}{'basket_iv':>10}"
              f" | {'drho_SPY':>9}  (does rho beat spy_iv alone?)")
        for h in a.fwd:
            fr = c.shift(-h) / c - 1
            row = out.assign(fr=fr).dropna(subset=["fr"])
            vals = {k: _spear(row[k], row["fr"]) for k in
                    ("rho_SPY", "rho_QQQ", "spy_iv", "basket_iv", "drho_SPY")}
            print(f"  {h:>4}d | {vals['rho_SPY']:>+9.3f}{vals['rho_QQQ']:>+9.3f} | "
                  f"{vals['spy_iv']:>+9.3f}{vals['basket_iv']:>+10.3f} | {vals['drho_SPY']:>+9.3f}")

    # bottom-quintile forward return: rho vs its vol controls
    print(f"\n  ===  bottom-quintile forward return (the 'low reading = bearish' test)  ===")
    for tgt, nm in [("qqq_px", "QQQ"), ("basket_px", "mega-basket")]:
        c = out[tgt]
        print(f"\n  {nm}:  {'fwd':>4}  {'lowQ rho_SPY':>13}{'lowQ spy_iv':>12}{'lowQ basket_iv':>15}{'all':>9}")
        for h in a.fwd:
            fr = (c.shift(-h) / c - 1)
            r = out.assign(fr=fr).dropna(subset=["fr"])
            def lq(k):
                return r["fr"][r[k] <= r[k].quantile(.2)].mean() * 100
            print(f"  {'':>7}{h:>4}d {lq('rho_SPY'):>+12.2f}%{lq('spy_iv'):>+11.2f}%"
                  f"{lq('basket_iv'):>+14.2f}%{r['fr'].mean()*100:>+8.2f}%")

    # monthly view -- is the signal broad or one episode?
    print(f"\n  ===  monthly mean rho_SPY  +  that month's fwd-20d QQQ return  ===")
    m = out.copy()
    m["fwd20"] = m["qqq_px"].shift(-20) / m["qqq_px"] - 1
    g = m.groupby(m.index.to_period("M")).agg(rho=("rho_SPY", "mean"), fwd20=("fwd20", "mean"), n=("rho_SPY", "size"))
    for per, r in g.iterrows():
        bar = "#" * int(max(0, r["rho"]) * 60)
        print(f"    {per}  rho {r['rho']:.3f} {bar:<28}  fwd20 QQQ {r['fwd20']*100:>+6.2f}%  (n={int(r['n'])})")

    out.to_parquet("_implied_corr.parquet")
    print(f"\n  wrote _implied_corr.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fwd", nargs="+", type=int, default=[5, 10, 20, 30])
    a = ap.parse_args()
    run(a)
