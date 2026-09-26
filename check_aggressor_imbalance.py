# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_aggressor_imbalance.py
============================

Gate question, before building any entry logic:  does options aggressor-imbalance
(volume lifting the ask minus volume hitting the bid) predict the underlying's
NEXT move -- and which *construction* of it predicts best?

The live bot already uses one construction: premium-weighted net flow,
  Sigma (ask_vol - bid_vol) * vwap * 100        (calls +, puts -)
cumulative on the day, EMA(5) crossover.  That is construction "prem" below and
serves as the baseline.  We test four more:

  prem   - premium-weighted net aggressor flow          (baseline / current bot)
  delta  - Sigma (ask_vol - bid_vol) * delta_close * 100
           = the share-equivalent hedge the dealer must put on. Mechanically this
           is the thing that actually pushes spot, so it should predict better.
  ratio  - signed aggressor contracts / gross contracts, bounded [-1, 1].
           Drift-resistant -- check_flow_drift.py showed the $ magnitude of "big
           flow" moves 2-23x over 2y; a bounded ratio should be far more stationary.
  d0     - "delta" restricted to 0DTE contracts -- the most hedging-urgent flow.
  otm    - "delta" restricted to OTM contracts (dir-aware) -- strips hedging /
           covered-call noise, leaves directional-conviction buying.

Each is scored two ways:
  cum    - value cumulative from the open (a "where do we stand" level)
  imp15  - trailing 15-minute sum (an impulse / acceleration)
(ratio uses trailing-15m and trailing-60m windows instead of cum.)

Forward underlying return measured at +15m, +30m, +60m, and to-the-close, all
within the same day.  Reported per construction x mode x horizon:
  spearman  - rank corr(signal_t, fwd_ret)                   overlapping, see note
  D10-D1    - mean fwd return of the top signal decile minus the bottom decile
  mono      - of the 9 decile-to-decile steps, how many increase (9/9 = perfectly
              monotone -> a usable continuous signal)
  hit       - P(fwd_ret > 0 | top decile) and P(fwd_ret < 0 | bottom decile)

--split does the whole thing IS vs OOS; a construction whose D10-D1 spread and
sign of spearman survive OOS is the one worth turning into an entry rule.

NOTE the minute samples overlap (a 60m horizon shares 59 minutes with its
neighbour), so |spearman| significance is overstated -- treat it as a ranking
device, not a p-value.  The decile spread + IS/OOS agreement is the real test.

Usage:
  python check_aggressor_imbalance.py SPY 2024-08-20 2026-08-21 --range
  python check_aggressor_imbalance.py SPY,QQQ,IWM,NVDA 2024-08-20 2026-08-21 --range --split 2025-08-21
  python check_aggressor_imbalance.py SPY 2024-08-20 2026-08-21 --range --hours 10-14
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

LAKE = "lake/silver/option-contracts-1m"
CACHE_DIR = "_imb_cache"
RTH_OPEN, RTH_CLOSE = 9 * 60 + 30, 16 * 60
HORIZONS = [15, 30, 60, "close"]
CONSTRUCTIONS = ["prem", "delta", "ratio", "d0", "otm"]


def _silver_dates() -> list[date]:
    out = []
    for p in glob.glob(os.path.join(LAKE, "date=*")):
        try:
            out.append(date.fromisoformat(os.path.basename(p).split("=", 1)[1]))
        except ValueError:
            pass
    return sorted(out)


def build_minute(ticker: str, days: list[date], force=False) -> pd.DataFrame:
    """Per-RTH-minute aggregates for one ticker over `days`.  Cached to parquet.

    columns: date, mod (minute-of-day ET), spot,
             prem_nf, delta_nf, signed_ct, gross_ct, d0_delta_nf, otm_delta_nf
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{ticker}_{days[0]}_{days[-1]}.parquet")
    if os.path.exists(cache) and not force:
        return pd.read_parquet(cache)

    frames = []
    for i, d in enumerate(days, 1):
        p = os.path.join(LAKE, f"date={d.isoformat()}", "bars.parquet")
        if not os.path.exists(p):
            continue
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol") == ticker)
              .select("option_type", "strike", "expiry", "minute_et",
                      "vwap", "delta_close", "ask_volume", "bid_volume",
                      "underlying_close"))
        df = lf.collect()
        if df.is_empty():
            continue
        df = df.with_columns(
            (pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
             + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"),
            (pl.col("expiry").cast(pl.Date) - pl.lit(d)).dt.total_days().alias("dte"),
        ).filter((pl.col("mod") >= RTH_OPEN) & (pl.col("mod") <= RTH_CLOSE))
        if df.is_empty():
            continue
        is_call = pl.col("option_type") == "call"
        net_ct = (pl.col("ask_volume") - pl.col("bid_volume")).cast(pl.Float64)
        dir_sign = pl.when(is_call).then(1.0).otherwise(-1.0)
        # OTM, direction-aware: call strike above spot, put strike below spot
        otm = ((is_call & (pl.col("strike") > pl.col("underlying_close")))
               | (~is_call & (pl.col("strike") < pl.col("underlying_close"))))
        df = df.with_columns(
            (net_ct * dir_sign * pl.col("vwap") * 100).alias("_prem"),
            (net_ct * pl.col("delta_close").fill_null(0.0) * 100).alias("_delta"),
            (net_ct * dir_sign).alias("_signed_ct"),
            (pl.col("ask_volume") + pl.col("bid_volume")).cast(pl.Float64).alias("_gross_ct"),
            pl.when(pl.col("dte") == 0)
              .then(net_ct * pl.col("delta_close").fill_null(0.0) * 100)
              .otherwise(0.0).alias("_d0"),
            pl.when(otm)
              .then(net_ct * pl.col("delta_close").fill_null(0.0) * 100)
              .otherwise(0.0).alias("_otm"),
        )
        agg = df.group_by("mod").agg(
            pl.col("underlying_close").median().alias("spot"),
            pl.col("_prem").sum().alias("prem_nf"),
            pl.col("_delta").sum().alias("delta_nf"),
            pl.col("_signed_ct").sum().alias("signed_ct"),
            pl.col("_gross_ct").sum().alias("gross_ct"),
            pl.col("_d0").sum().alias("d0_delta_nf"),
            pl.col("_otm").sum().alias("otm_delta_nf"),
        ).sort("mod").with_columns(pl.lit(d).alias("date"))
        frames.append(agg.to_pandas())
        if i % 50 == 0:
            print(f"  {ticker}: {i}/{len(days)} days")

    if not frames:
        out = pd.DataFrame()
    else:
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.date
    out.to_parquet(cache, index=False)
    return out


def _signals(day: pd.DataFrame) -> pd.DataFrame:
    """Given one day's per-minute rows (sorted by mod), attach every
    construction x mode signal column."""
    d = day.sort_values("mod").reset_index(drop=True)
    # reindex to a dense 1-min grid so trailing windows are true minutes
    full = pd.RangeIndex(d["mod"].iloc[0], d["mod"].iloc[-1] + 1)
    d = d.set_index("mod").reindex(full)
    d["spot"] = d["spot"].ffill()
    for c in ("prem_nf", "delta_nf", "signed_ct", "gross_ct", "d0_delta_nf", "otm_delta_nf"):
        d[c] = d[c].fillna(0.0)

    out = pd.DataFrame(index=d.index)
    out["spot"] = d["spot"]
    for name, col in (("prem", "prem_nf"), ("delta", "delta_nf"),
                      ("d0", "d0_delta_nf"), ("otm", "otm_delta_nf")):
        out[f"{name}__cum"] = d[col].cumsum()
        out[f"{name}__imp15"] = d[col].rolling(15, min_periods=5).sum()
    num15 = d["signed_ct"].rolling(15, min_periods=5).sum()
    den15 = d["gross_ct"].rolling(15, min_periods=5).sum()
    num60 = d["signed_ct"].rolling(60, min_periods=15).sum()
    den60 = d["gross_ct"].rolling(60, min_periods=15).sum()
    out["ratio__r15"] = np.where(den15 > 0, num15 / den15, np.nan)
    out["ratio__r60"] = np.where(den60 > 0, num60 / den60, np.nan)

    for h in (15, 30, 60):
        out[f"fwd{h}"] = out["spot"].shift(-h) / out["spot"] - 1.0
    out["fwdclose"] = out["spot"].iloc[-1] / out["spot"] - 1.0
    out["mod"] = out.index
    return out


SIGCOLS = ["prem__cum", "prem__imp15", "delta__cum", "delta__imp15",
           "d0__cum", "d0__imp15", "otm__cum", "otm__imp15",
           "ratio__r15", "ratio__r60"]


def assemble(mn: pd.DataFrame, lo_h: int, hi_h: int) -> pd.DataFrame:
    rows = []
    for _, day in mn.groupby("date"):
        s = _signals(day)
        s["date"] = day["date"].iloc[0]
        rows.append(s)
    alls = pd.concat(rows, ignore_index=True)
    alls = alls[(alls["mod"] >= lo_h * 60) & (alls["mod"] <= hi_h * 60 + 59)]
    return alls


def _eval(df: pd.DataFrame, sig: str, hz) -> dict | None:
    fcol = "fwdclose" if hz == "close" else f"fwd{hz}"
    d = df[[sig, fcol]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200:
        return None
    x = d[sig].to_numpy()
    y = d[fcol].to_numpy()
    if np.std(x) == 0:
        return None
    sp = pd.Series(x).rank().corr(pd.Series(y).rank())
    # deciles by signal
    q = pd.qcut(pd.Series(x).rank(method="first"), 10, labels=False)
    means = pd.Series(y).groupby(q.values).mean()
    if len(means) < 10:
        return None
    spread = means.iloc[-1] - means.iloc[0]
    mono = int(np.sum(np.diff(means.to_numpy()) > 0))
    top = y[q.values == 9]
    bot = y[q.values == 0]
    hit_top = float(np.mean(top > 0)) if len(top) else np.nan
    hit_bot = float(np.mean(bot < 0)) if len(bot) else np.nan
    return {"n": len(d), "sp": sp, "spread_bps": spread * 1e4,
            "mono": mono, "hit_top": hit_top, "hit_bot": hit_bot}


def report(tk: str, alls: pd.DataFrame, split: date | None):
    print("=" * 100)
    print(f"  {tk}   aggressor-imbalance -> forward underlying return"
          + (f"   (IS < {split} <= OOS)" if split else ""))
    print("=" * 100)
    slices = [("ALL", alls)]
    if split is not None:
        slices = [("IS", alls[alls["date"] < split]), ("OOS", alls[alls["date"] >= split])]

    for hz in HORIZONS:
        label = "close" if hz == "close" else f"+{hz}m"
        print(f"\n  ---- forward horizon {label} " + "-" * 60)
        print(f"     {'construction':16}{'slice':6}{'n':>8}{'spearman':>10}"
              f"{'D10-D1(bps)':>13}{'mono':>6}{'hitTop':>8}{'hitBot':>8}")
        for sig in SIGCOLS:
            for sl_name, sl in slices:
                r = _eval(sl, sig, hz)
                if r is None:
                    print(f"     {sig:16}{sl_name:6}{'--':>8}")
                    continue
                print(f"     {sig:16}{sl_name:6}{r['n']:>8}{r['sp']:>+10.3f}"
                      f"{r['spread_bps']:>+13.1f}{r['mono']:>4}/9"
                      f"{r['hit_top']:>8.2f}{r['hit_bot']:>8.2f}")

    # headline: rank constructions by OOS (or ALL) |spearman| + spread at +30m
    print(f"\n  ---- ranking @ +30m " + "-" * 60)
    key_slice = slices[-1]
    name, sl = key_slice
    scored = []
    for sig in SIGCOLS:
        r = _eval(sl, sig, 30)
        if r:
            scored.append((sig, r))
    scored.sort(key=lambda kv: abs(kv[1]["sp"]), reverse=True)
    for sig, r in scored:
        verdict = ("directional signal" if abs(r["sp"]) >= 0.05 and r["mono"] in (0, 1, 8, 9)
                   else "weak" if abs(r["sp"]) >= 0.03 else "noise")
        print(f"     {sig:16}[{name}]  spearman {r['sp']:+.3f}   D10-D1 {r['spread_bps']:+.1f}bps"
              f"   mono {r['mono']}/9   -> {verdict}")
    print()


def run(args, tk: str, days: list[date]):
    mn = build_minute(tk, days, force=args.rebuild)
    if mn.empty:
        print(f"  {tk}: no silver data in range")
        return
    lo_h, hi_h = 9, 15
    if args.hours:
        a, b = args.hours.split("-")
        lo_h, hi_h = int(a), int(b)
    alls = assemble(mn, lo_h, hi_h)
    split = date.fromisoformat(args.split) if args.split else None
    report(tk, alls, split)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker", help="one ticker or comma list")
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--range", action="store_true")
    ap.add_argument("--split", help="YYYY-MM-DD: entries before are IS, rest OOS")
    ap.add_argument("--hours", help="restrict signal minutes to HH-HH ET (entry hour), e.g. 10-14")
    ap.add_argument("--rebuild", action="store_true", help="force per-minute cache rebuild")
    a = ap.parse_args()
    tickers = [t.strip().upper() for t in a.ticker.split(",") if t.strip()]
    if a.range and len(a.dates) == 2:
        lo, hi = sorted(date.fromisoformat(x) for x in a.dates)
        dd = [d for d in _silver_dates() if lo <= d <= hi]
    else:
        dd = sorted(date.fromisoformat(x) for x in a.dates)
    if not dd:
        raise SystemExit("no silver dates in range")
    for t in tickers:
        run(a, t, dd)
