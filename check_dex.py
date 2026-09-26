# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_dex.py
============
Options DELTA EXPOSURE (net_dex = call_delta + put_delta from /greek-exposure,
already on disk in historical/GEX{T}.parquet for 28 tickers back to ~2022).
GEX (gamma) is a deployed regime lever; DEX (dealer directional positioning) has
never been tested here.

  --test predict   prior-day net_dex features -> forward 1/5/10d underlying return
                   (IS/OOS @ 2025-08-21, with net_gex as the control -- does DEX
                   add anything beyond the gamma regime we already use?)
  --test rules     split each deployed rule's option P&L by prior-day DEX state
                   at the trigger date (like the GEX / AMT open-location gates)
  --weekly         resample net_dex Fri->Fri, forward-WEEK return

Features (all PRIOR-day close, lookahead-free):
  dex_sign  sign(net_dex)          + = dealers net long delta
  dex_pct   trailing-252d %ile     extreme directional positioning
  dex_d5    5d change              positioning drift / directional flow
  dex_ovg   net_dex / |net_gex|    directional vs convex balance

Usage:
  python check_dex.py --test predict
  python check_dex.py --test predict --weekly
  python check_dex.py --test rules
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD = 15 * 60 + 55
TK = ["SPY", "QQQ", "IWM", "NVDA", "META", "MSFT", "AMZN", "AAPL", "TSLA", "AVGO", "GLD", "SMH"]


def _dex(tk):
    p = f"{HIST}/GEX{tk}.parquet"
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).to_pandas()
    if "net_dex" not in d.columns:
        return None
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    d = d.sort_values("date").set_index("date")
    return d[["net_dex", "net_gex"]].astype(float)


def _px(tk):
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).to_pandas()
    d.columns = [c.lower() for c in d.columns]
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York")
    mo = et.dt.hour * 60 + et.dt.minute
    m = (mo >= 570) & (mo <= 960)
    g = pd.DataFrame({"d": et[m].dt.date.values, "c": d["close"][m].astype(float).values})
    s = g.groupby("d")["c"].last()
    s.index = pd.to_datetime(s.index)
    return s


def _feat(dx, weekly):
    df = dx.copy()
    if weekly:
        df = df.resample("W-FRI").last().dropna()
    df["dex_sign"] = np.sign(df["net_dex"])
    df["dex_pct"] = df["net_dex"].rolling(252 if not weekly else 52, min_periods=20).apply(
        lambda w: (w.iloc[-1] > w[:-1]).mean() if len(w) > 1 else np.nan)
    df["dex_d5"] = df["net_dex"].diff(5 if not weekly else 4)
    df["dex_ovg"] = df["net_dex"] / df["net_gex"].abs().clip(lower=1)
    df["gex_sign"] = np.sign(df["net_gex"])
    df["gex_pct"] = df["net_gex"].rolling(252 if not weekly else 52, min_periods=20).apply(
        lambda w: (w.iloc[-1] > w[:-1]).mean() if len(w) > 1 else np.nan)
    return df.shift(1)                       # everything known only from PRIOR close


def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def test_predict(a):
    hs = [1, 5, 10] if not a.weekly else [1, 2, 4]
    unit = "week" if a.weekly else "day"
    print("=" * 100)
    print(f"  net_dex features -> forward {unit} return   (IS/OOS @ {SPLIT}{'  [WEEKLY]' if a.weekly else ''})")
    print("=" * 100)
    pool = []
    for tk in a.tickers:
        dx, px = _dex(tk), _px(tk)
        if dx is None or px is None:
            print(f"  {tk}: no data"); continue
        f = _feat(dx, a.weekly)
        px = px.resample("W-FRI").last().dropna() if a.weekly else px
        j = f.join(px.rename("c"), how="inner").dropna(subset=["c"])
        for h in hs:
            j[f"fr{h}"] = j["c"].shift(-h) / j["c"] - 1
        j["tk"] = tk
        pool.append(j)
    if not pool:
        return
    P = pd.concat(pool)
    for lbl, sub in (("POOLED IS", P[P.index.map(lambda x: x.date() < SPLIT)]),
                     ("POOLED OOS", P[P.index.map(lambda x: x.date() >= SPLIT)])):
        print(f"\n  {lbl}  (n={len(sub)})")
        print(f"    {'':10}" + "".join(f"fwd{h}{unit[0]}".rjust(10) for h in hs))
        for feat in ("dex_sign", "dex_pct", "dex_d5", "dex_ovg", "gex_sign", "gex_pct"):
            row = "".join(f"{_spear(sub[feat], sub[f'fr{h}']):>+10.3f}" for h in hs)
            print(f"    {feat:10}{row}")
    # decile spread on the best-looking feature (dex_pct), OOS
    print(f"\n  OOS forward-{hs[-1]}{unit[0]} return by net_dex-percentile decile (pooled):")
    o = P[P.index.map(lambda x: x.date() >= SPLIT)].dropna(subset=["dex_pct", f"fr{hs[-1]}"])
    o["dq"] = pd.qcut(o["dex_pct"].rank(method="first"), 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    for q, g in o.groupby("dq", observed=True):
        print(f"    {q:8}  n={len(g):>4}  mean {g[f'fr{hs[-1]}'].mean()*100:>+6.2f}%")


def test_rules(a):
    import directional_flow_backtester as D
    from config import RULES
    try:
        from amt_profile import amt_open_map, amt_ok
    except Exception:
        amt_open_map = amt_ok = None

    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date
    rules = [r for r in RULES if r.get("enabled", True)]

    print("=" * 96)
    print("  deployed rules' option P&L split by PRIOR-DAY DEX state at the trigger date")
    print("=" * 96)
    for r in rules:
        tk = r["ticker"]; direction = r["direction"].upper()
        dx = _dex(tk)
        if dx is None:
            print(f"\n  {r['name']}: no GEX{tk}.parquet"); continue
        f = _feat(dx, False)
        dsign = {d.date(): v for d, v in f["dex_sign"].items()}
        dpct = {d.date(): v for d, v in f["dex_pct"].items()}
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        ema_stack = D.load_ema_stack(HIST, tk, int(r["ema_confirm"])) if r.get("ema_confirm") else None
        amt = amt_open_map(tk) if (r.get("amt_open") and amt_open_map) else {}
        trigs = D.triggers_for(flow, tk)
        if not trigs:
            tkf, _ = D._screen_build_one("lake/silver/option-contracts-1m", tk)
            trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
        D.annotate_flow_pct(trigs, int(r.get("flow_window_days") or 60))
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        base_tr, rr = float(r["target_roe"]), float(r["rr"])
        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
        buckets = {"ALL": [], "dex+": [], "dex-": [], "dex hi(>.8)": [], "dex lo(<.2)": []}
        for t, thr in matched:
            d, ts = t["date"], t["ts"]
            if r.get("amt_open") and amt_ok and not amt_ok(r["amt_open"], amt.get(d)):
                continue
            if ema_stack is not None:
                st = D.ema_state_at(ema_stack, ts)
                if st is not None and st != ("BULL" if direction == "CALL" else "BEAR"):
                    continue
            sg, pc = dsign.get(d), dpct.get(d)
            for p in D._option_paths(t, direction, r.get("dte", [0, 1]), bbd, bbc):
                pnl = D._bracket_pnl(*p, base_tr, rr, None, EOD)
                buckets["ALL"].append((d, pnl))
                if sg == 1:
                    buckets["dex+"].append((d, pnl))
                elif sg == -1:
                    buckets["dex-"].append((d, pnl))
                if pc is not None and pc > 0.8:
                    buckets["dex hi(>.8)"].append((d, pnl))
                elif pc is not None and pc < 0.2:
                    buckets["dex lo(<.2)"].append((d, pnl))
        print(f"\n  {r['name']}  ({tk} {direction})")
        for k, v in buckets.items():
            if len(v) < 12:
                print(f"    {k:14} n={len(v):>4}  (thin)"); continue
            i = [p for d, p in v if d < SPLIT]; o = [p for d, p in v if d >= SPLIT]
            print(f"    {k:14} n={len(v):>4}  IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
                  f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  win {np.mean([x>0 for _,x in v]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", choices=("predict", "rules"), required=True)
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--tickers", nargs="+", default=TK)
    a = ap.parse_args()
    a.tickers = [t.upper() for t in a.tickers]
    (test_predict if a.test == "predict" else test_rules)(a)
