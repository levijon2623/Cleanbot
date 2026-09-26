# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_netprem_predictive.py
===========================

Same gate question as check_aggressor_imbalance.py, but on UW's SERVER-SIDE
per-minute flow (`historical/NETPREM{T}.parquet` from `netprem-build`) instead of
the silver-tape reconstruction.  If UW's own trade classification (sweep
detection, complex-order handling, full chain, off-tape prints) carries forward
predictive power that our de-cumulated side counters didn't, it shows up here.

Constructions, per RTH minute, from UW's native fields:

  prem   - net_premium = net_call_premium - net_put_premium
           == THE EXACT QUANTITY THE LIVE BOT CONSUMES.  This run is the honest
           test of the bot's actual signal against forward underlying return.
  delta  - net_delta = UW's server-side share-equivalent dealer hedge
  cvol   - net_volume = net_call_volume - net_put_volume  (net aggressor contracts)
  ratio  - net_volume / (call_volume + put_volume), bounded [-1, 1] (drift-resistant)
  pskew  - net_premium / (|net_call_premium| + |net_put_premium|), bounded premium tilt

prem/delta/cvol scored as day-cumulative (`__cum`) and trailing-15m (`__imp15`);
ratio/pskew as trailing-15m and trailing-60m (`__r15` / `__r60`).

Forward underlying return (from `historical/{T}.parquet` 1-min close) at
+15/+30/+60m and to-the-close, same day.  Per construction x mode x horizon:
spearman, D10-D1 decile spread (bps), decile monotonicity (of 9 steps), and
top/bottom-decile directional hit-rate.  --split runs it IS vs OOS.

Overlapping minute samples -> |spearman| significance is overstated; use it to
rank, trust the decile spread + IS/OOS agreement.

Usage:
  python check_netprem_predictive.py SPY 2024-08-20 2026-08-21 --range
  python check_netprem_predictive.py SPY,QQQ,IWM,NVDA 2024-08-20 2026-08-21 --range --split 2025-08-21
  python check_netprem_predictive.py SPY 2024-08-20 2026-08-21 --range --hours 10-15
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
HORIZONS = [15, 30, 60, "close"]
SIGCOLS = ["prem__cum", "prem__imp15", "delta__cum", "delta__imp15",
           "cvol__cum", "cvol__imp15", "ratio__r15", "ratio__r60",
           "pskew__r15", "pskew__r60"]


def _load(tk: str, lo: date, hi: date) -> pd.DataFrame:
    """Per-minute netprem fields joined to the underlying 1-min close, RTH only."""
    np_path = f"{HIST}/NETPREM{tk}.parquet"
    px_path = f"{HIST}/{tk}.parquet"
    npf = (pl.read_parquet(np_path)
           .select("minute_et", "date",
                   "net_call_premium", "net_put_premium", "net_call_volume",
                   "net_put_volume", "call_volume", "put_volume", "net_delta",
                   "net_premium", "net_volume")
           .filter((pl.col("date") >= lo) & (pl.col("date") <= hi)))
    px = (pl.read_parquet(px_path)
          .select("minute_et", pl.col("close").alias("spot"))
          .unique(subset=["minute_et"], keep="last"))
    df = npf.join(px, on="minute_et", how="left").sort("minute_et")
    d = df.to_pandas()
    d["mod"] = d["minute_et"].dt.hour * 60 + d["minute_et"].dt.minute
    d = d[(d["mod"] >= 9 * 60 + 30) & (d["mod"] <= 16 * 60)]
    return d


def _signals(day: pd.DataFrame) -> pd.DataFrame:
    d = day.sort_values("mod").reset_index(drop=True)
    full = pd.RangeIndex(d["mod"].iloc[0], d["mod"].iloc[-1] + 1)
    d = d.set_index("mod").reindex(full)
    d["spot"] = d["spot"].ffill().bfill()
    for c in ("net_premium", "net_delta", "net_volume", "call_volume", "put_volume",
              "net_call_premium", "net_put_premium"):
        d[c] = d[c].fillna(0.0)

    out = pd.DataFrame(index=d.index)
    out["spot"] = d["spot"]
    for name, col in (("prem", "net_premium"), ("delta", "net_delta"), ("cvol", "net_volume")):
        out[f"{name}__cum"] = d[col].cumsum()
        out[f"{name}__imp15"] = d[col].rolling(15, min_periods=5).sum()

    gross_v = (d["call_volume"] + d["put_volume"]).rolling(15, min_periods=5).sum()
    net_v15 = d["net_volume"].rolling(15, min_periods=5).sum()
    gross_v60 = (d["call_volume"] + d["put_volume"]).rolling(60, min_periods=15).sum()
    net_v60 = d["net_volume"].rolling(60, min_periods=15).sum()
    out["ratio__r15"] = np.where(gross_v > 0, net_v15 / gross_v, np.nan)
    out["ratio__r60"] = np.where(gross_v60 > 0, net_v60 / gross_v60, np.nan)

    gp15 = (d["net_call_premium"].abs() + d["net_put_premium"].abs()).rolling(15, min_periods=5).sum()
    npm15 = d["net_premium"].rolling(15, min_periods=5).sum()
    gp60 = (d["net_call_premium"].abs() + d["net_put_premium"].abs()).rolling(60, min_periods=15).sum()
    npm60 = d["net_premium"].rolling(60, min_periods=15).sum()
    out["pskew__r15"] = np.where(gp15 > 0, npm15 / gp15, np.nan)
    out["pskew__r60"] = np.where(gp60 > 0, npm60 / gp60, np.nan)

    for h in (15, 30, 60):
        out[f"fwd{h}"] = out["spot"].shift(-h) / out["spot"] - 1.0
    out["fwdclose"] = out["spot"].iloc[-1] / out["spot"] - 1.0
    out["mod"] = out.index
    return out


def assemble(mn: pd.DataFrame, lo_h: int, hi_h: int) -> pd.DataFrame:
    rows = []
    for _, day in mn.groupby("date"):
        if day["mod"].nunique() < 60:
            continue
        s = _signals(day)
        s["date"] = day["date"].iloc[0]
        rows.append(s)
    alls = pd.concat(rows, ignore_index=True)
    alls["date"] = pd.to_datetime(alls["date"]).dt.date
    return alls[(alls["mod"] >= lo_h * 60) & (alls["mod"] <= hi_h * 60 + 59)]


def _eval(df: pd.DataFrame, sig: str, hz) -> dict | None:
    fcol = "fwdclose" if hz == "close" else f"fwd{hz}"
    d = df[[sig, fcol]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200:
        return None
    x, y = d[sig].to_numpy(), d[fcol].to_numpy()
    if np.std(x) == 0:
        return None
    sp = pd.Series(x).rank().corr(pd.Series(y).rank())
    q = pd.qcut(pd.Series(x).rank(method="first"), 10, labels=False).to_numpy()
    means = pd.Series(y).groupby(q).mean()
    if len(means) < 10:
        return None
    spread = means.iloc[-1] - means.iloc[0]
    mono = int(np.sum(np.diff(means.to_numpy()) > 0))
    top, bot = y[q == 9], y[q == 0]
    return {"n": len(d), "sp": sp, "spread_bps": spread * 1e4, "mono": mono,
            "hit_top": float(np.mean(top > 0)), "hit_bot": float(np.mean(bot < 0))}


def report(tk: str, alls: pd.DataFrame, split: date | None):
    print("=" * 100)
    print(f"  {tk}   UW net-prem-ticks -> forward underlying return"
          + (f"   (IS < {split} <= OOS)" if split else ""))
    print("=" * 100)
    slices = ([("IS", alls[alls["date"] < split]), ("OOS", alls[alls["date"] >= split])]
              if split is not None else [("ALL", alls)])
    for hz in HORIZONS:
        label = "close" if hz == "close" else f"+{hz}m"
        print(f"\n  ---- forward horizon {label} " + "-" * 58)
        print(f"     {'construction':15}{'slice':6}{'n':>8}{'spearman':>10}"
              f"{'D10-D1(bps)':>13}{'mono':>6}{'hitTop':>8}{'hitBot':>8}")
        for sig in SIGCOLS:
            for sl_name, sl in slices:
                r = _eval(sl, sig, hz)
                if r is None:
                    print(f"     {sig:15}{sl_name:6}{'--':>8}")
                    continue
                print(f"     {sig:15}{sl_name:6}{r['n']:>8}{r['sp']:>+10.3f}"
                      f"{r['spread_bps']:>+13.1f}{r['mono']:>4}/9{r['hit_top']:>8.2f}{r['hit_bot']:>8.2f}")

    name, sl = slices[-1]
    print(f"\n  ---- ranking @ +30m [{name}] " + "-" * 50)
    scored = [(s, _eval(sl, s, 30)) for s in SIGCOLS]
    scored = [(s, r) for s, r in scored if r]
    scored.sort(key=lambda kv: abs(kv[1]["sp"]), reverse=True)
    for sig, r in scored:
        v = ("DIRECTIONAL" if abs(r["sp"]) >= 0.05 and r["mono"] in (0, 1, 8, 9)
             else "weak" if abs(r["sp"]) >= 0.03 else "noise")
        print(f"     {sig:15} spearman {r['sp']:+.3f}   D10-D1 {r['spread_bps']:+.1f}bps"
              f"   mono {r['mono']}/9   -> {v}")
    print()


def run(args, tk: str, lo: date, hi: date):
    try:
        mn = _load(tk, lo, hi)
    except FileNotFoundError as e:
        print(f"  {tk}: missing file ({e})")
        return
    if mn.empty:
        print(f"  {tk}: no rows in range")
        return
    lo_h, hi_h = 9, 16
    if args.hours:
        a, b = args.hours.split("-")
        lo_h, hi_h = int(a), int(b)
    alls = assemble(mn, lo_h, hi_h)
    report(tk, alls, date.fromisoformat(args.split) if args.split else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker")
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--range", action="store_true")
    ap.add_argument("--split", help="YYYY-MM-DD: before = IS, rest = OOS")
    ap.add_argument("--hours", help="restrict signal minutes to HH-HH ET, e.g. 10-15")
    a = ap.parse_args()
    lo, hi = sorted(date.fromisoformat(x) for x in a.dates[:2]) if a.range else (
        date.fromisoformat(a.dates[0]), date.fromisoformat(a.dates[-1]))
    for t in [x.strip().upper() for x in a.ticker.split(",") if x.strip()]:
        run(a, t, lo, hi)
