# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_markup_regime.py
======================

Can we tell, on day D, whether a momentum name (NVDA and its cohort) is entering
/ inside a sustained MARKUP -- the "grind up, dips bought" state the Mar-Aug 2025
NVDA backtest was fit on -- vs the other ~80% of the time when a CALL bias bleeds?

Target (per ticker, per day, causal -> uses only D and earlier for features):
    markup[D] = 1  iff  the 20-trading-day forward return is strong  AND  the
                        worst drawdown inside that window stays shallow.
    --markup-mode pct (default): "strong" = forward return in this ticker's top
        --ret-pctile; "shallow" = forward max-drawdown in the better --dd-pctile.
    --markup-mode abs: forward return >= --ret-min and drawdown >= -(--dd-max).

Features (all daily, all causal), from the historical/ parquets:
  price   -- trailing momentum, distance from highs, pullback depth, ATR
             compression, new-high frequency, up/down volume
  gex     -- net_gex/dex/charm/vanna z-scores + slopes (historical/GEX{T}.parquet,
             back to 2022 -- the vanna/charm melt-up mechanics, dealers-short-gamma
             squeeze fuel)
  flow    -- 5-day cumulative net call premium / net premium / net_delta, and the
             call aggressor ratio (historical/NETPREM{T}.parquet, 2yr only)

Pools days across --tickers (default the semi/megacap-momentum cohort) so there
are enough independent ~20d windows to run an IS/OOS split.  Per feature it
reports: Spearman vs the 20d forward return, and P(markup) in the top vs bottom
feature decile, IS and OOS.  Then a simple composite score (count of features in
their bullish tercile) and the forward-return / markup-rate by score.

NOTE overlapping 20d windows inflate significance hugely -- the IS/OOS agreement
and the cross-ticker pooling are the real tests, not any single p-value.

Usage:
  python check_markup_regime.py --split 2025-08-21
  python check_markup_regime.py --tickers NVDA --split 2025-08-21
  python check_markup_regime.py --split 2025-08-21 --markup-mode abs --ret-min 0.08 --dd-max 0.05
"""
from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
COHORT = ["NVDA", "AVGO", "MU", "META", "TSLA"]


# ---------------------------------------------------------------- loaders
def load_daily_ohlc(tk: str) -> pd.DataFrame | None:
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    df = pl.read_parquet(p).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    if "start_time" not in df or "close" not in df:
        return None
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York")
    m = et.dt.hour * 60 + et.dt.minute
    df = df.assign(d=et.dt.date)[(m >= 570) & (m <= 960)]
    g = df.groupby("d").agg(open=("open", "first"), high=("high", "max"),
                            low=("low", "min"), close=("close", "last"),
                            volume=("volume", "sum")).reset_index()
    g["d"] = pd.to_datetime(g["d"])
    return g.sort_values("d").reset_index(drop=True)


def load_gex_daily(tk: str) -> pd.DataFrame | None:
    p = f"{HIST}/GEX{tk}.parquet"
    if not os.path.exists(p):
        return None
    df = pl.read_parquet(p).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df:
        return None
    df["d"] = pd.to_datetime(df["date"])
    keep = ["d"] + [c for c in ("net_gex", "net_dex", "net_charm", "net_vanna",
                                "call_gamma", "put_gamma") if c in df.columns]
    return df[keep].sort_values("d").reset_index(drop=True)


def load_netprem_daily(tk: str) -> pd.DataFrame | None:
    p = f"{HIST}/NETPREM{tk}.parquet"
    if not os.path.exists(p):
        return None
    lf = pl.scan_parquet(p).select(
        pl.col("date"),
        pl.col("net_call_premium").cast(pl.Float64),
        pl.col("net_premium").cast(pl.Float64),
        pl.col("net_delta").cast(pl.Float64),
        pl.col("call_volume").cast(pl.Float64),
        pl.col("put_volume").cast(pl.Float64),
        (pl.col("call_volume") - pl.col("put_volume")).cast(pl.Float64).alias("cmp_vol"),
        pl.col("call_volume_ask_side").cast(pl.Float64),
        pl.col("call_volume_bid_side").cast(pl.Float64),
    )
    g = (lf.group_by("date").agg(
            pl.col("net_call_premium").sum(),
            pl.col("net_premium").sum(),
            pl.col("net_delta").sum(),
            pl.col("call_volume_ask_side").sum(),
            pl.col("call_volume_bid_side").sum(),
        ).sort("date").collect().to_pandas())
    g["d"] = pd.to_datetime(g["date"])
    g["call_aggr_ratio"] = (g["call_volume_ask_side"]
                            / (g["call_volume_ask_side"] + g["call_volume_bid_side"]).replace(0, np.nan))
    return g[["d", "net_call_premium", "net_premium", "net_delta", "call_aggr_ratio"]]


# ---------------------------------------------------------------- features
def _z(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return (s - m) / sd.replace(0, np.nan)


def build_features(tk: str) -> pd.DataFrame | None:
    px = load_daily_ohlc(tk)
    if px is None or len(px) < 120:
        return None
    d = px.copy()
    c, h, lo, v = d["close"], d["high"], d["low"], d["volume"]

    d["ret_5d"] = c.pct_change(5)
    d["ret_20d"] = c.pct_change(20)
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    d["above_sma20"] = c / sma20 - 1
    d["above_sma50"] = c / sma50 - 1
    d["sma20_slope"] = sma20 / sma20.shift(10) - 1
    d["dist_from_high"] = c / c.rolling(60).max() - 1
    d["pullback_10d"] = c.rolling(10).min() / c.rolling(10).max() - 1
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr14, atr50 = tr.rolling(14).mean(), tr.rolling(50).mean()
    d["atr_compression"] = atr14 / atr50 - 1               # <0 = coiled
    bbw = (c.rolling(20).std() * 4) / sma20
    d["bbw_pctile"] = bbw.rolling(120, min_periods=40).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    d["newhigh_20_cnt"] = (c >= c.rolling(20).max()).rolling(10).sum()
    upv = v.where(c > c.shift(), 0.0).rolling(10).sum()
    dnv = v.where(c < c.shift(), 0.0).rolling(10).sum()
    d["up_vol_ratio"] = upv / dnv.replace(0, np.nan)

    gx = load_gex_daily(tk)
    if gx is not None:
        d = d.merge(gx, on="d", how="left")
        for col in ("net_gex", "net_dex", "net_charm", "net_vanna"):
            if col in d:
                d[col] = d[col].ffill(limit=3)
                d[f"{col}_z"] = _z(d[col], 60)
                d[f"{col}_slope"] = d[col] - d[col].shift(5)
        if {"call_gamma", "put_gamma"} <= set(d.columns):
            d["cg_pg_ratio"] = d["call_gamma"].ffill(limit=3) / d["put_gamma"].abs().ffill(limit=3).replace(0, np.nan)

    npm = load_netprem_daily(tk)
    if npm is not None:
        d = d.merge(npm, on="d", how="left")
        d["ncp_5d_z"] = _z(d["net_call_premium"].rolling(5).sum(), 60)
        d["np_5d_z"] = _z(d["net_premium"].rolling(5).sum(), 60)
        d["ndelta_5d_z"] = _z(d["net_delta"].rolling(5).sum(), 60)
        d["call_aggr_5d"] = d["call_aggr_ratio"].rolling(5).mean()

    d["ticker"] = tk
    return d


FEATURES = [
    "ret_5d", "ret_20d", "above_sma20", "above_sma50", "sma20_slope",
    "dist_from_high", "pullback_10d", "atr_compression", "bbw_pctile",
    "newhigh_20_cnt", "up_vol_ratio",
    "net_gex_z", "net_gex_slope", "net_dex_z", "net_charm_z", "net_vanna_z",
    "cg_pg_ratio", "ncp_5d_z", "np_5d_z", "ndelta_5d_z", "call_aggr_5d",
]
# sign of the "bullish" direction for each feature (for the composite score)
BULL_DIR = {f: 1 for f in FEATURES}
BULL_DIR.update({"pullback_10d": 1, "dist_from_high": 1, "atr_compression": -1})


# ---------------------------------------------------------------- target
def add_target(d: pd.DataFrame, fwd: int, mode: str, ret_p, dd_p, ret_min, dd_max) -> pd.DataFrame:
    c, lo = d["close"], d["low"]
    fwd_ret = c.shift(-fwd) / c - 1
    roll_min = lo.shift(-1).rolling(fwd, min_periods=fwd).min().shift(-(fwd - 1))
    fwd_dd = roll_min / c - 1
    d = d.assign(fwd_ret=fwd_ret, fwd_dd=fwd_dd)
    ok = fwd_ret.notna() & fwd_dd.notna()
    if mode == "abs":
        d["markup"] = (ok & (fwd_ret >= ret_min) & (fwd_dd >= -dd_max)).astype(float)
    else:
        rt = fwd_ret[ok].quantile(ret_p / 100.0)
        dt = fwd_dd[ok].quantile(1 - dd_p / 100.0)   # dd_p% best -> the (1-dd_p) quantile of dd
        d["markup"] = (ok & (fwd_ret >= rt) & (fwd_dd >= dt)).astype(float)
    d.loc[~ok, "markup"] = np.nan
    return d


# ---------------------------------------------------------------- report
def _decile_markup(sub: pd.DataFrame, feat: str) -> tuple | None:
    x = sub[[feat, "markup", "fwd_ret"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 200:
        return None
    q = pd.qcut(x[feat].rank(method="first"), 10, labels=False)
    mk = x["markup"].groupby(q).mean()
    fr = x["fwd_ret"].groupby(q).mean()
    if len(mk) < 10:
        return None
    sp = x[feat].rank().corr(x["fwd_ret"].rank())
    mono = int(np.sum(np.diff(mk.to_numpy()) > 0))
    return (len(x), sp, mk.iloc[0], mk.iloc[-1], fr.iloc[0] * 100, fr.iloc[-1] * 100, mono)


def run(args):
    frames = []
    for tk in args.tickers:
        d = build_features(tk)
        if d is None:
            print(f"  {tk}: insufficient data -- skipped")
            continue
        d = add_target(d, args.fwd, args.markup_mode, args.ret_pctile, args.dd_pctile,
                       args.ret_min, args.dd_max)
        frames.append(d)
    if not frames:
        raise SystemExit("no data")
    P = pd.concat(frames, ignore_index=True)
    P = P[P["d"] >= pd.Timestamp("2022-01-01")]
    split = pd.Timestamp(args.split) if args.split else None

    base = P["markup"].mean()
    print("=" * 100)
    print(f"  MARKUP REGIME   cohort={','.join(args.tickers)}   fwd={args.fwd}d   mode={args.markup_mode}")
    print(f"  base markup rate: {base*100:.1f}%   (n={P['markup'].notna().sum()} labelled days"
          + (f", IS<{split.date()}<=OOS)" if split else ")"))
    print("=" * 100)

    slices = ([("IS", P[P["d"] < split]), ("OOS", P[P["d"] >= split])]
              if split is not None else [("ALL", P)])

    print(f"\n  {'feature':16}{'slice':5}{'n':>7}{'spearman':>10}"
          f"{'P(mk)|D1':>10}{'P(mk)|D10':>11}{'fwdRet D1':>11}{'D10':>8}{'mono':>6}")
    print("  " + "-" * 92)
    scored = {}
    for feat in FEATURES:
        if feat not in P.columns:
            continue
        row_ok = True
        for sl_name, sl in slices:
            r = _decile_markup(sl, feat)
            if r is None:
                print(f"  {feat:16}{sl_name:5}{'--':>7}")
                row_ok = False
                continue
            n, sp, p1, p10, f1, f10, mono = r
            scored.setdefault(feat, {})[sl_name] = (sp, p10 - p1, mono)
            print(f"  {feat:16}{sl_name:5}{n:>7}{sp:>+10.3f}{p1:>10.2f}{p10:>11.2f}"
                  f"{f1:>+11.1f}{f10:>+8.1f}{mono:>4}/9")
        if row_ok:
            print()

    # features whose top-decile markup lift AND spearman sign agree IS & OOS
    if split is not None:
        good = []
        for f, sd in scored.items():
            if "IS" in sd and "OOS" in sd:
                (spi, lifti, moni), (spo, lifto, mono) = sd["IS"], sd["OOS"]
                if np.sign(spi) == np.sign(spo) and lifti > 0.03 and lifto > 0.03:
                    good.append((f, spi, spo, lifti, lifto))
        print("\n  ---- features that separate markup IS AND OOS (|lift|>3pp, same spearman sign) ----")
        if not good:
            print("     (none)")
        for f, spi, spo, li, lo in sorted(good, key=lambda x: -(x[3] + x[4])):
            print(f"     {f:16} spearman {spi:+.3f}/{spo:+.3f}   markup lift {li*100:+.0f}pp/{lo*100:+.0f}pp")
        comp_feats = [g[0] for g in good] or [f for f in FEATURES if f in P.columns]
    else:
        comp_feats = [f for f in FEATURES if f in P.columns]

    # composite: count of comp_feats in their bullish tercile, per row
    print(f"\n  ---- composite regime score ({len(comp_feats)} features: "
          f"{', '.join(comp_feats)}) ----")
    for f in comp_feats:
        P[f"_b_{f}"] = np.nan
    for tk in args.tickers:
        mtk = P["ticker"] == tk
        for f in comp_feats:
            s = P.loc[mtk, f]
            if BULL_DIR.get(f, 1) >= 0:
                thr = s.quantile(2 / 3)
                P.loc[mtk, f"_b_{f}"] = (s >= thr).astype(float)
            else:
                thr = s.quantile(1 / 3)
                P.loc[mtk, f"_b_{f}"] = (s <= thr).astype(float)
    P["score"] = P[[f"_b_{f}" for f in comp_feats]].sum(axis=1, min_count=1)
    nf = len(comp_feats)
    P["score_bkt"] = pd.cut(P["score"], bins=[-0.1, nf*0.33, nf*0.66, nf + 0.1],
                            labels=["low", "mid", "high"])
    slices2 = ([("IS", P[P["d"] < split]), ("OOS", P[P["d"] >= split])]
               if split is not None else [("ALL", P)])
    print(f"     {'slice':6}{'bucket':8}{'n':>7}{'markup%':>10}{'fwdRet mean':>13}{'fwdRet med':>12}")
    for sl_name, sl in slices2:
        for bkt in ["low", "mid", "high"]:
            x = sl[sl["score_bkt"] == bkt]
            x = x[x["markup"].notna()]
            if len(x) < 30:
                continue
            print(f"     {sl_name:5}{bkt:8}{len(x):>7}{x['markup'].mean()*100:>9.1f}%"
                  f"{x['fwd_ret'].mean()*100:>+12.1f}%{x['fwd_ret'].median()*100:>+11.1f}%")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=COHORT)
    ap.add_argument("--fwd", type=int, default=20, help="forward horizon in trading days (default 20)")
    ap.add_argument("--markup-mode", choices=("pct", "abs"), default="pct")
    ap.add_argument("--ret-pctile", type=float, default=70, help="pct mode: forward return must exceed this ticker-percentile")
    ap.add_argument("--dd-pctile", type=float, default=50, help="pct mode: forward drawdown must be in the best this-percent")
    ap.add_argument("--ret-min", type=float, default=0.06, help="abs mode: min forward return")
    ap.add_argument("--dd-max", type=float, default=0.06, help="abs mode: max forward drawdown (as a positive frac)")
    ap.add_argument("--split", help="YYYY-MM-DD IS/OOS split")
    args = ap.parse_args()
    args.tickers = [t.upper() for t in args.tickers]
    run(args)
