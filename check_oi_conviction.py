# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_oi_conviction.py
======================

Phase 2 of the markup question, but a different hypothesis: not dealer-hedging
mechanics off real-time flow, but INSTITUTIONAL POSITIONING / CONVICTION that
leads price by days-to-weeks.  Does the open-interest structure + unusual options
activity on day D predict the SIGN and SIZE of the underlying's move over the
next N trading days?

Built from the silver tape (EOD per-contract snapshot: OI, day volume, ask/bid
volume, premium, IV, delta), filtered to DTE 7-60 and |moneyness| <= 25%.

Daily features (per underlying, causal):
  oi_wdelta_chg  -- 5d change in the OI-weighted net delta of the whole chain
                    (aggregate directional positioning drift)
  d_call_oi_otm  -- 5d sum of OI change in OTM calls (+3% .. +20%)      [accumulation up]
  d_put_oi_otm   -- 5d sum of OI change in OTM puts  (-20% .. -3%)      [accumulation down]
  otm_cp_oi      -- OTM call OI / OTM put OI  (standing skew of positioning)
  uoa_call/put   -- 5d premium-weighted count of contracts where day volume
                    > 1.5x prior OI  AND  ask-side ratio > 0.6  AND  OTM
                    (fresh, aggressive, directional opens)
  uoa_net        -- uoa_call - uoa_put
  otm_call_ivsh  -- 5d change in mean IV of OTM calls (call-skew steepening)
  oi_concn       -- Herfindahl of |OI change| across strikes (targeted vs diffuse)

Targets, per day:
  fwd_ret   -- signed N-day forward return   (does the signal call direction?)
  fwd_abs   -- |N-day forward return|        (does |signal| call an imminent breakout?)

Pools the cohort, IS/OOS split.  Per feature: Spearman vs fwd_ret and vs fwd_abs,
plus the forward-return spread between the top and bottom signal decile.

Usage:
  python check_oi_conviction.py --build --tickers NVDA AVGO MU META TSLA
  python check_oi_conviction.py --split 2025-08-21
  python check_oi_conviction.py --split 2025-08-21 --fwd 10
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
HIST = "historical"
CACHE = "_oi_cache"
COHORT = ["NVDA", "AVGO", "MU", "META", "TSLA"]
DTE_LO, DTE_HI = 7, 60
MNY = 0.25


def _silver_parts():
    return sorted(glob.glob(os.path.join(LAKE, "date=*", "bars.parquet")))


def build_oi_snapshot(tk: str, force=False) -> pd.DataFrame:
    """One EOD row per (date, contract) for `tk`: strike, expiry, dte, moneyness,
    oi, day_vol, ask_vol, bid_vol, premium, iv, delta, spot.  Cached."""
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{tk}.parquet")
    if os.path.exists(fp) and not force:
        return pd.read_parquet(fp)
    parts = _silver_parts()
    frames = []
    for i, p in enumerate(parts, 1):
        d = date.fromisoformat(os.path.basename(os.path.dirname(p)).split("=", 1)[1])
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol") == tk)
              .select("option_chain_id", "option_type", "strike", "expiry", "minute_et",
                      "open_interest", "volume", "ask_volume", "bid_volume", "premium",
                      "iv_close", "delta_close", "underlying_close"))
        df = lf.collect()
        if df.is_empty():
            continue
        # EOD: last bar per contract for OI/IV/delta/spot; day totals for volume/premium
        df = df.with_columns(
            (pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
             + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("_m"))
        eod = (df.sort("_m").group_by("option_chain_id").agg(
                    pl.col("option_type").first(),
                    pl.col("strike").first().cast(pl.Float64),
                    pl.col("expiry").first(),
                    pl.col("open_interest").last().cast(pl.Float64),
                    pl.col("iv_close").last(),
                    pl.col("delta_close").last(),
                    pl.col("underlying_close").last().cast(pl.Float64),
                    pl.col("volume").sum().cast(pl.Float64).alias("day_vol"),
                    pl.col("ask_volume").sum().cast(pl.Float64).alias("ask_vol"),
                    pl.col("bid_volume").sum().cast(pl.Float64).alias("bid_vol"),
                    pl.col("premium").sum().cast(pl.Float64).alias("day_prem"),
              ))
        spot = float(eod.select(pl.col("underlying_close").median()).item() or 0.0)
        if spot <= 0:
            continue
        eod = eod.with_columns(
            ((pl.col("expiry").cast(pl.Date) - pl.lit(d)).dt.total_days()).alias("dte"),
            ((pl.col("strike") - spot) / spot).alias("mny"),
            pl.lit(d).alias("date"),
        ).filter((pl.col("dte") >= DTE_LO) & (pl.col("dte") <= DTE_HI)
                 & (pl.col("mny").abs() <= MNY))
        if not eod.is_empty():
            frames.append(eod.to_pandas())
        if i % 50 == 0:
            print(f"  {tk}: {i}/{len(parts)} partitions")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(fp, index=False)
    print(f"  {tk}: {len(out):,} contract-days -> {fp}")
    return out


def _z(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return (s - m) / sd.replace(0, np.nan)


def _daily_underlying_close(tk):
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    df = pl.read_parquet(p).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York")
    m = et.dt.hour * 60 + et.dt.minute
    g = df.assign(d=et.dt.date)[(m >= 570) & (m <= 960)].groupby("d").agg(
        close=("close", "last"), low=("low", "min"), high=("high", "max"))
    g.index = pd.to_datetime(g.index)
    return g.sort_index()


def build_features(tk: str, fwd: int) -> pd.DataFrame | None:
    snap = build_oi_snapshot(tk)
    if snap is None or snap.empty:
        return None
    snap = snap.sort_values(["option_chain_id", "date"])
    snap["d_oi"] = snap.groupby("option_chain_id")["open_interest"].diff()
    snap["ask_ratio"] = snap["ask_vol"] / (snap["ask_vol"] + snap["bid_vol"]).replace(0, np.nan)
    is_call = snap["option_type"] == "call"
    otm_call = is_call & (snap["mny"].between(0.03, MNY))
    otm_put = (~is_call) & (snap["mny"].between(-MNY, -0.03))
    fresh_aggr = (snap["day_vol"] > 1.5 * snap["open_interest"].clip(lower=1)) & (snap["ask_ratio"] > 0.6)

    snap["_oc"] = otm_call.astype(int)
    snap["_op"] = otm_put.astype(int)
    snap["_fa"] = fresh_aggr.astype(int)
    rows = []
    for d, g in snap.groupby("date"):
        oi_wdelta = float((g["open_interest"] * g["delta_close"]).sum())
        gc, gp = g[g["_oc"] == 1], g[g["_op"] == 1]
        adoi = g.groupby("strike")["d_oi"].sum().abs()
        acoi = gc.groupby("strike")["d_oi"].sum().abs()
        apoi = gp.groupby("strike")["d_oi"].sum().abs()
        def _herf(s):
            return float(((s / s.sum()) ** 2).sum()) if s.sum() > 0 else np.nan
        # signed concentration: + if the day's |OI change| is call-dominated, - if put
        call_doi_abs = float(gc["d_oi"].abs().sum())
        put_doi_abs = float(gp["d_oi"].abs().sum())
        cp_tilt = ((call_doi_abs - put_doi_abs) / (call_doi_abs + put_doi_abs)
                   if (call_doi_abs + put_doi_abs) > 0 else 0.0)
        rows.append({
            "date": d,
            "oi_wdelta": oi_wdelta,
            "d_call_oi_otm_1d": float(gc["d_oi"].sum()),
            "d_put_oi_otm_1d": float(gp["d_oi"].sum()),
            "otm_cp_oi": float(gc["open_interest"].sum() / max(gp["open_interest"].sum(), 1)),
            "uoa_call_1d": float(g[(g["_oc"] == 1) & (g["_fa"] == 1)]["day_prem"].sum()),
            "uoa_put_1d": float(g[(g["_op"] == 1) & (g["_fa"] == 1)]["day_prem"].sum()),
            "otm_call_iv": float(gc["iv_close"].mean()) if len(gc) else np.nan,
            "oi_concn": _herf(adoi),
            "call_concn": _herf(acoi),
            "put_concn": _herf(apoi),
            "concn_dir_1d": (_herf(adoi) or 0.0) * cp_tilt,
            "n_strikes": int((adoi > 0).sum()),
            "day_opt_vol": float(g["day_vol"].sum()),
            "day_opt_prem": float(g["day_prem"].sum()),
        })
    f = pd.DataFrame(rows).sort_values("date").set_index("date")
    f["concn_dir"] = f["concn_dir_1d"].rolling(3).mean()
    # per-ticker de-trended versions -- raw counts grow over 2yr and differ by name
    for col in ("n_strikes", "oi_concn", "day_opt_vol", "day_opt_prem"):
        f[f"{col}_z"] = _z(f[col], 60)

    f["oi_wdelta_chg"] = f["oi_wdelta"].diff(5)
    f["d_call_oi_otm"] = f["d_call_oi_otm_1d"].rolling(5).sum()
    f["d_put_oi_otm"] = f["d_put_oi_otm_1d"].rolling(5).sum()
    f["uoa_call"] = f["uoa_call_1d"].rolling(5).sum()
    f["uoa_put"] = f["uoa_put_1d"].rolling(5).sum()
    f["uoa_net"] = f["uoa_call"] - f["uoa_put"]
    f["otm_call_ivsh"] = f["otm_call_iv"].diff(5)
    for col in ("uoa_call", "uoa_put", "uoa_net"):
        f[f"{col}_z"] = _z(f[col], 60)
    # froth ratio: call UOA premium as a share of total OTM UOA premium (level-free)
    f["uoa_call_share"] = f["uoa_call"] / (f["uoa_call"] + f["uoa_put"]).replace(0, np.nan)

    px = _daily_underlying_close(tk)
    if px is None:
        return None
    f = f.join(px, how="left")
    f["close"] = f["close"].ffill()
    f["fwd_ret"] = f["close"].shift(-fwd) / f["close"] - 1
    f["fwd_abs"] = f["fwd_ret"].abs()
    f["ticker"] = tk
    return f.reset_index()


FEATURES = ["uoa_net", "uoa_net_z", "uoa_call_z", "uoa_put_z", "uoa_call_share",
            "oi_concn", "oi_concn_z", "n_strikes", "n_strikes_z"]


def _eval(sub, feat, tgt):
    x = sub[[feat, tgt]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 150:
        return None
    sp = x[feat].rank().corr(x[tgt].rank())
    q = pd.qcut(x[feat].rank(method="first"), 5, labels=False)
    m = x[tgt].groupby(q).mean()
    return len(x), sp, (m.iloc[-1] - m.iloc[0]) * 100, m.iloc[-1] * 100, m.iloc[0] * 100


def run(args):
    frames = []
    for tk in args.tickers:
        f = build_features(tk, args.fwd)
        if f is not None:
            frames.append(f)
    if not frames:
        raise SystemExit("no data -- run --build first")
    P = pd.concat(frames, ignore_index=True)
    split = pd.Timestamp(args.split) if args.split else None
    slices = ([("IS", P[P["date"] < split]), ("OOS", P[P["date"] >= split])]
              if split else [("ALL", P)])

    print("=" * 100)
    print(f"  OI / UOA CONVICTION -> {args.fwd}d forward move   cohort={','.join(args.tickers)}")
    print(f"  n={P['fwd_ret'].notna().sum()} labelled days" + (f"   IS<{split.date()}<=OOS" if split else ""))
    print("=" * 100)

    for tgt, lab in (("fwd_ret", "SIGNED forward return (direction)"),
                     ("fwd_abs", "ABS forward return (breakout size)")):
        print(f"\n  ---- {lab} " + "-" * 50)
        print(f"     {'feature':16}{'slice':5}{'n':>7}{'spearman':>10}{'Q5-Q1 (pp)':>13}{'Q5':>9}{'Q1':>9}")
        for feat in FEATURES:
            for sl_name, sl in slices:
                r = _eval(sl, feat, tgt)
                if r is None:
                    print(f"     {feat:16}{sl_name:5}{'--':>7}")
                    continue
                n, sp, spread, q5, q1 = r
                print(f"     {feat:16}{sl_name:5}{n:>7}{sp:>+10.3f}{spread:>+13.1f}{q5:>+9.1f}{q1:>+9.1f}")
            print()

    if split is not None:
        print("  ---- survives IS AND OOS (same spearman sign, |Q5-Q1|>1.5pp both, on SIGNED return) ----")
        hits = []
        for feat in FEATURES:
            ri = _eval(P[P["date"] < split], feat, "fwd_ret")
            ro = _eval(P[P["date"] >= split], feat, "fwd_ret")
            if ri and ro and np.sign(ri[1]) == np.sign(ro[1]) and abs(ri[2]) > 1.5 and abs(ro[2]) > 1.5:
                hits.append((feat, ri[1], ro[1], ri[2], ro[2]))
        if not hits:
            print("     (none)")
        for f, spi, spo, si, so in sorted(hits, key=lambda x: -(abs(x[3]) + abs(x[4]))):
            print(f"     {f:16} spearman {spi:+.3f}/{spo:+.3f}   Q5-Q1 {si:+.1f}pp / {so:+.1f}pp")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=COHORT)
    ap.add_argument("--fwd", type=int, default=15, help="forward horizon, trading days (default 15)")
    ap.add_argument("--split", help="YYYY-MM-DD IS/OOS split")
    ap.add_argument("--build", action="store_true", help="(re)build the per-ticker OI snapshot cache and exit")
    args = ap.parse_args()
    args.tickers = [t.upper() for t in args.tickers]
    if args.build:
        for tk in args.tickers:
            build_oi_snapshot(tk, force=True)
    else:
        run(args)
