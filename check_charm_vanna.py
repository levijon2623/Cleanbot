# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_charm_vanna.py
====================

Second-order greeks -- CHARM (dDelta/dTime) and VANNA (dDelta/dVol) -- and whether
dealer charm/vanna exposure predicts underlying drift, with a focus on the
afternoon / into-the-close window.

Mechanism (the textbook narrative we are testing, not assuming):
  * CHARM: as time passes, OTM option deltas bleed toward 0 and ITM toward +/-1.
    If dealers are net long puts / short calls (the usual index convention), the
    passage of time forces them to BUY the underlying in an up-tape and SELL in a
    down-tape -- a drift AMPLIFIER that accelerates Thu/Fri and into every close,
    peaking on monthly OpEx. Sign of the forced flow tracks sign(charm exposure).
  * VANNA: as IV falls (the typical intraday grind, esp. afternoons), the delta of
    dealers' short puts shrinks -> they buy back hedges -> "vanna tailwind" / melt
    up. Rising IV (selloff) -> the reverse, a down-accelerant. Sign tracks
    sign(vanna exposure) * sign(dIV).

Data already on disk (no backfill):
  lake/silver/spot-exposures-1m/date=*/{T}.parquet -- per-MINUTE charm/vanna/gamma
    "per_one_percent_move" in {oi, vol, dir} weightings, ~503 sessions
    (2024-08-20 .. 2026-08-21), RTH + a little pre-market. oi = resting-OI
    convention; dir = signed-aggressor inferred (reads 0 until midday).
  historical/{T}.parquet -- 1-min underlying OHLC (same window).
  historical/GEX{T}.parquet -- DAILY net_charm / net_vanna close snapshots
    (call_charm+put_charm etc.), ~1100 sessions back to 2022 but no price, so the
    --daily test is still capped to the 2024-08+ OHLC window.

  --build [DATES]     scan silver -> _cv_cache/{T}.parquet (1-min charm/vanna/gamma)
  --snapshot          _cv_cache + OHLC -> _cv_snapshot.parquet (per ticker-day:
                        greek snapshot at 12:00/13:00/14:00/14:30 ET + the price
                        milestones needed for every forward-return target)
  --test intraday    does the T-snapshot of charm / vanna (raw, z-scored, signed)
                        predict the T -> 15:55 drift, the 15:00 -> 15:55 power
                        hour, the close -> next-open overnight? pooled spearman
                        (date-clustered), quintile buckets, gamma-sign & OpEx
                        conditionals, IS/OOS @ --split.
  --test daily       daily net_charm / net_vanna (close snapshot) -> next-session
                        open->close and the overnight gap. IS/OOS @ --split.
  --test power       a concrete "power-hour" paper strategy: at T, take a
                        directional position in the underlying sized by the
                        charm/vanna signal, exit 15:55. Win rate, avg move, and a
                        spread-cost sensitivity. This is the "can we exploit it"
                        read -- underlying move only; option translation noted.

Usage:
  python check_charm_vanna.py --build
  python check_charm_vanna.py --snapshot
  python check_charm_vanna.py --test intraday --split 2025-08-21
  python check_charm_vanna.py --test daily
  python check_charm_vanna.py --test power --decide 840
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
SILVER = "lake/silver/spot-exposures-1m"
CACHE = "_cv_cache"
SNAP = "_cv_snapshot.parquet"

CORE = ["SPY", "QQQ", "IWM", "NVDA", "META", "MSFT", "AAPL", "AMZN", "TSLA", "GLD", "GOOGL", "AVGO"]

# decision minutes (ET minutes-of-day) at which we snapshot the greeks
DECIDE_MODS = [720, 780, 840, 870]          # 12:00, 13:00, 14:00, 14:30
PRICE_MODS = [720, 780, 840, 870, 900, 955]  # + 15:00, 15:55
CLOSE_MOD = 955                              # 15:55 "into the close" mark
OPEN_MOD = 571                               # first RTH minute we trust

SPOT_COLS = [
    "minute_et", "price",
    "gamma_per_one_percent_move_oi", "charm_per_one_percent_move_oi", "vanna_per_one_percent_move_oi",
    "gamma_per_one_percent_move_vol", "charm_per_one_percent_move_vol", "vanna_per_one_percent_move_vol",
    "gamma_per_one_percent_move_dir", "charm_per_one_percent_move_dir", "vanna_per_one_percent_move_dir",
]
SHORT = {
    "gamma_per_one_percent_move_oi": "g_oi", "charm_per_one_percent_move_oi": "c_oi",
    "vanna_per_one_percent_move_oi": "v_oi",
    "gamma_per_one_percent_move_vol": "g_vol", "charm_per_one_percent_move_vol": "c_vol",
    "vanna_per_one_percent_move_vol": "v_vol",
    "gamma_per_one_percent_move_dir": "g_dir", "charm_per_one_percent_move_dir": "c_dir",
    "vanna_per_one_percent_move_dir": "v_dir",
}


# --------------------------------------------------------------------------- #
#  build: silver spot-exposures -> per-ticker 1-min cache
# --------------------------------------------------------------------------- #
def _load_spot(tk: str, force: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{tk}.parquet")
    if os.path.exists(fp) and not force:
        return pd.read_parquet(fp)
    frames = []
    for d in sorted(glob.glob(f"{SILVER}/date=*/")):
        p = os.path.join(d, f"{tk}.parquet")
        if os.path.exists(p):
            cols = [c for c in SPOT_COLS]
            frames.append(pl.read_parquet(p, columns=cols))
    if not frames:
        return pd.DataFrame()
    out = pl.concat(frames, how="vertical_relaxed").to_pandas()
    out = out.rename(columns=SHORT)
    out["minute_et"] = pd.to_datetime(out["minute_et"]).dt.tz_localize(None)
    out["date"] = out["minute_et"].dt.date
    out["mod"] = out["minute_et"].dt.hour * 60 + out["minute_et"].dt.minute
    out = out.sort_values("minute_et").drop_duplicates(["date", "mod"], keep="last").reset_index(drop=True)
    out.to_parquet(fp, index=False)
    return out


def build(a):
    dates = a.build if a.build else []
    for tk in a.tickers:
        df = _load_spot(tk, force=True)
        if df.empty:
            print(f"  {tk}: no spot-exposures data")
            continue
        print(f"  {tk}: {len(df):>7} rows  {df['date'].min()} .. {df['date'].max()}")


def _load_1m(tk):
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p, columns=["start_time", "open", "high", "low", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["ts"] = et
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 960)]
    return d.sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  snapshot: per ticker-day greek snapshot @ decision mods + price milestones
# --------------------------------------------------------------------------- #
def _asof(day_df: pd.DataFrame, mod: int, col: str):
    """last value at or before `mod` on this day; NaN if none / stale > 20min."""
    sub = day_df[day_df["mod"] <= mod]
    if sub.empty:
        return np.nan
    row = sub.iloc[-1]
    if mod - row["mod"] > 20:
        return np.nan
    return row[col]


def snapshot(a):
    rows = []
    for tk in a.tickers:
        sp = _load_spot(tk)
        px = _load_1m(tk)
        if sp.empty or px is None:
            print(f"  {tk}: missing data, skipped")
            continue
        sp_by = {d: g.sort_values("mod") for d, g in sp.groupby("date")}
        px_by = {d: g.sort_values("mod") for d, g in px.groupby("date")}
        days = sorted(set(sp_by) & set(px_by))
        for i, d in enumerate(days):
            s, p = sp_by[d], px_by[d]
            nd = days[i + 1] if i + 1 < len(days) else None
            rec = {"ticker": tk, "date": d}
            # price milestones (underlying, from OHLC close of the minute bar)
            for m in PRICE_MODS:
                rec[f"px_{m}"] = _asof(p, m, "close")
            o = p[p["mod"] >= OPEN_MOD]
            rec["px_open"] = o.iloc[0]["open"] if not o.empty else np.nan
            rec["px_close"] = p.iloc[-1]["close"] if not p.empty else np.nan
            # next-day open
            rec["px_nopen"] = np.nan
            if nd is not None:
                on = px_by[nd][px_by[nd]["mod"] >= OPEN_MOD]
                if not on.empty:
                    rec["px_nopen"] = on.iloc[0]["open"]
            # greek snapshots at each decision mod
            for m in DECIDE_MODS:
                for col in ("g_oi", "c_oi", "v_oi", "g_dir", "c_dir", "v_dir"):
                    rec[f"{col}_{m}"] = _asof(s, m, col)
                rec[f"pxg_{m}"] = _asof(s, m, "price")  # UW's own spot at that mod
            rows.append(rec)
        print(f"  {tk}: {len(days)} ticker-days")
    df = pd.DataFrame(rows)
    df.to_parquet(SNAP, index=False)
    print(f"\n  wrote {SNAP}  ({len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min()} .. {df['date'].max()})")


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan, 0
    xr = pd.Series(x[ok]).rank().to_numpy()
    yr = pd.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1]), int(ok.sum())


def _cluster_t(x, y, groups):
    """spearman-style slope significance with errors clustered by `groups` (date).
    returns (beta_on_ranks, t_stat, n). Simple: OLS of rank(y) ~ rank(x) with
    cluster-robust SE."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    g = np.asarray(groups)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    if len(x) < 50:
        return np.nan, np.nan, len(x)
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    xr = (xr - xr.mean()) / xr.std()
    yr = (yr - yr.mean()) / yr.std()
    X = np.column_stack([np.ones_like(xr), xr])
    beta = np.linalg.lstsq(X, yr, rcond=None)[0]
    resid = yr - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for gv in np.unique(g):
        m = g == gv
        Xg = X[m]
        ug = (Xg * resid[m, None]).sum(axis=0)
        meat += np.outer(ug, ug)
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))[1]
    return float(beta[1]), float(beta[1] / se) if se > 0 else np.nan, len(x)


def _zcol(df, col, by="ticker", win=60):
    """lookahead-free rolling z of `col` within each ticker: (x_i - mean[i-win:i-1])
    / std[i-win:i-1]."""
    out = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(by).groups.items():
        s = df.loc[idx, col].astype(float)
        mu = s.shift(1).rolling(win, min_periods=20).mean()
        sd = s.shift(1).rolling(win, min_periods=20).std()
        out.loc[idx] = (s - mu) / sd
    return out


def _opex(d):
    """monthly OpEx = 3rd Friday."""
    d = pd.Timestamp(d)
    return d.weekday() == 4 and 15 <= d.day <= 21


def _load_snap():
    if not os.path.exists(SNAP):
        raise SystemExit("run --snapshot first")
    df = pd.read_parquet(SNAP)
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.weekday
    df["opex"] = df["date"].apply(_opex)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  test: intraday
# --------------------------------------------------------------------------- #
def test_intraday(a):
    split = pd.Timestamp(a.split)
    df = _load_snap()
    md = a.decide
    print("=" * 104)
    print(f"  INTRADAY charm/vanna -> forward drift   (decision snapshot @ mod {md} = "
          f"{md // 60:02d}:{md % 60:02d} ET)   split {split.date()}")
    print("=" * 104)

    # forward-return targets
    df["r_aft"] = df[f"px_{CLOSE_MOD}"] / df[f"px_{md}"] - 1                 # T -> 15:55
    df["r_ph"] = df[f"px_{CLOSE_MOD}"] / df["px_900"] - 1                    # 15:00 -> 15:55
    df["r_on"] = df["px_nopen"] / df["px_close"] - 1                         # close -> next open
    df["r_full"] = df["px_close"] / df[f"px_{md}"] - 1                       # T -> close
    # morning move (control / momentum feature)
    df["mom_am"] = df[f"px_{md}"] / df["px_open"] - 1
    # IV proxy change: we don't have IV, but vanna itself * sign of a vol move.
    # use realized: (afternoon range so far) is unavailable; skip -> vanna raw only.

    # signed features
    for base in ("c_oi", "v_oi", "g_oi"):
        col = f"{base}_{md}"
        df[f"{base}_z"] = _zcol(df, col)
    df["cv_sign_agree"] = np.sign(df[f"c_oi_{md}"]) * np.sign(df[f"v_oi_{md}"])
    df["g_sign"] = np.sign(df[f"g_oi_{md}"])

    feats = [
        (f"c_oi_{md}", "charm_oi raw"),
        ("c_oi_z", "charm_oi z60"),
        (f"v_oi_{md}", "vanna_oi raw"),
        ("v_oi_z", "vanna_oi z60"),
        (f"g_oi_{md}", "gamma_oi raw (ctrl)"),
        ("g_oi_z", "gamma_oi z60 (ctrl)"),
        ("mom_am", "AM move (ctrl)"),
    ]
    targets = [("r_aft", "T->15:55"), ("r_ph", "15:00->15:55"), ("r_on", "close->nopen"), ("r_full", "T->close")]

    for tlab, tname in targets:
        print(f"\n  --- target: {tname} ---")
        for is_lab, sub in (("IS ", df[df.date < split]), ("OOS", df[df.date >= split])):
            print(f"    [{is_lab}]  n={sub[tlab].notna().sum()}   "
                  f"(mean {sub[tlab].mean() * 100:+.3f}%  sd {sub[tlab].std() * 100:.3f}%)")
            for fcol, flab in feats:
                rho, n = _spear(sub[fcol], sub[tlab])
                beta, t, _ = _cluster_t(sub[fcol], sub[tlab], sub["date"])
                star = "  <--" if abs(t) >= 2 and np.isfinite(t) else ""
                print(f"      {flab:22} rho {rho:+.3f}  (clus-t {t:+.2f}, n {n}){star}")

    # quintile buckets on the two headline features, full sample, IS vs OOS
    for fcol, flab in ((f"c_oi_{md}", "charm_oi raw"), ("c_oi_z", "charm_oi z"),
                       (f"v_oi_{md}", "vanna_oi raw"), ("v_oi_z", "vanna_oi z")):
        print(f"\n  --- quintiles of {flab}  ->  mean(T->15:55) / P(up) ---")
        for is_lab, sub in (("IS ", df[df.date < split].copy()), ("OOS", df[df.date >= split].copy())):
            s = sub[sub[fcol].notna() & sub["r_aft"].notna()].copy()
            if len(s) < 60:
                print(f"    [{is_lab}] thin"); continue
            s["q"] = pd.qcut(s[fcol], 5, labels=False, duplicates="drop")
            line = f"    [{is_lab}] "
            for q in sorted(s["q"].dropna().unique()):
                b = s[s["q"] == q]
                line += f" Q{int(q) + 1}:{b['r_aft'].mean() * 100:+.3f}%/{(b['r_aft'] > 0).mean():.0%}(n{len(b)}) "
            print(line)

    # gamma-sign conditional + OpEx conditional on the sign of charm
    print(f"\n  --- sign(charm_oi @ {md}) -> mean(T->15:55), conditioned ---")
    for cond_lab, cond in (("ALL", df["date"] > pd.Timestamp("2000")),
                           ("gamma_oi>0 (long-gamma)", df["g_sign"] > 0),
                           ("gamma_oi<0 (short-gamma)", df["g_sign"] < 0),
                           ("OpEx day", df["opex"]),
                           ("Thu/Fri", df["dow"] >= 3)):
        for is_lab, sm in (("IS ", df.date < split), ("OOS", df.date >= split)):
            s = df[cond & sm]
            pos = s[np.sign(s[f"c_oi_{md}"]) > 0]["r_aft"]
            neg = s[np.sign(s[f"c_oi_{md}"]) < 0]["r_aft"]
            if len(pos) < 20 or len(neg) < 20:
                continue
            print(f"    {cond_lab:26} [{is_lab}]  charm+>{pos.mean() * 100:+.3f}% (n{len(pos)})   "
                  f"charm-<{neg.mean() * 100:+.3f}% (n{len(neg)})   spread {(pos.mean() - neg.mean()) * 100:+.3f}pp")


# --------------------------------------------------------------------------- #
#  test: daily
# --------------------------------------------------------------------------- #
def _daily_px(tk):
    px = _load_1m(tk)
    if px is None:
        return None
    g = px.groupby("date")
    out = pd.DataFrame({
        "open": g.first()["open"],
        "close": g.last()["close"],
    }).reset_index()
    out["date"] = pd.to_datetime(out["date"])
    return out


def test_daily(a):
    split = pd.Timestamp(a.split)
    print("=" * 104)
    print(f"  DAILY net_charm / net_vanna (close snapshot) -> next session   split {split.date()}")
    print("=" * 104)
    allrows = []
    for tk in a.tickers:
        gp = f"{HIST}/GEX{tk}.parquet"
        if not os.path.exists(gp):
            continue
        g = pl.read_parquet(gp).to_pandas()
        g["date"] = pd.to_datetime(g["date"])
        px = _daily_px(tk)
        if px is None:
            continue
        m = g.merge(px, on="date", how="inner").sort_values("date").reset_index(drop=True)
        if len(m) < 120:
            continue
        m["ticker"] = tk
        # forward targets: next session
        m["r_oc_next"] = m["close"].shift(-1) / m["open"].shift(-1) - 1     # next open->close
        m["r_on_next"] = m["open"].shift(-1) / m["close"] - 1               # tonight's overnight gap
        m["r_cc_next"] = m["close"].shift(-1) / m["close"] - 1              # close->close
        for c in ("net_charm", "net_vanna", "net_gex"):
            s = m[c].astype(float)
            mu = s.shift(1).rolling(60, min_periods=20).mean()
            sd = s.shift(1).rolling(60, min_periods=20).std()
            m[f"{c}_z"] = (s - mu) / sd
        allrows.append(m)
    D = pd.concat(allrows, ignore_index=True)
    feats = [("net_charm", "net_charm raw"), ("net_charm_z", "net_charm z60"),
             ("net_vanna", "net_vanna raw"), ("net_vanna_z", "net_vanna z60"),
             ("net_gex_z", "net_gex z60 (ctrl)")]
    for tcol, tname in (("r_oc_next", "next open->close"), ("r_on_next", "overnight gap"),
                        ("r_cc_next", "next close->close")):
        print(f"\n  --- target: {tname} ---")
        for is_lab, sub in (("IS ", D[D.date < split]), ("OOS", D[D.date >= split])):
            print(f"    [{is_lab}] n={sub[tcol].notna().sum()}")
            for fcol, flab in feats:
                rho, n = _spear(sub[fcol], sub[tcol])
                beta, t, _ = _cluster_t(sub[fcol], sub[tcol], sub["date"])
                star = "  <--" if abs(t) >= 2 and np.isfinite(t) else ""
                print(f"      {flab:20} rho {rho:+.3f}  (clus-t {t:+.2f}, n {n}){star}")

    # quintiles of net_charm_z -> next open->close
    for fcol in ("net_charm_z", "net_vanna_z"):
        print(f"\n  --- quintiles of {fcol} -> next open->close / P(up) ---")
        for is_lab, sub in (("IS ", D[D.date < split].copy()), ("OOS", D[D.date >= split].copy())):
            s = sub[sub[fcol].notna() & sub["r_oc_next"].notna()].copy()
            if len(s) < 60:
                continue
            s["q"] = pd.qcut(s[fcol], 5, labels=False, duplicates="drop")
            line = f"    [{is_lab}] "
            for q in sorted(s["q"].dropna().unique()):
                b = s[s["q"] == q]
                line += f" Q{int(q) + 1}:{b['r_oc_next'].mean() * 100:+.3f}%/{(b['r_oc_next'] > 0).mean():.0%} "
            print(line)


# --------------------------------------------------------------------------- #
#  test: power  (concrete paper strategy)
# --------------------------------------------------------------------------- #
def test_power(a):
    split = pd.Timestamp(a.split)
    df = _load_snap()
    md = a.decide
    df["r_aft"] = df[f"px_{CLOSE_MOD}"] / df[f"px_{md}"] - 1
    df["c_z"] = _zcol(df, f"c_oi_{md}")
    df["v_z"] = _zcol(df, f"v_oi_{md}")
    df["g_sign"] = np.sign(df[f"g_oi_{md}"])
    df = df[df["r_aft"].notna() & df["c_z"].notna() & df["v_z"].notna()].copy()

    print("=" * 104)
    print(f"  POWER-HOUR paper strategy: decide @ {md // 60:02d}:{md % 60:02d}, hold underlying to 15:55")
    print(f"  position sign = f(charm/vanna signal);  split {split.date()};  underlying move only")
    print("=" * 104)

    def run(name, sig_fn, sub):
        sig = sig_fn(sub)
        take = sig != 0
        pnl = np.sign(sig[take]) * sub.loc[take, "r_aft"]
        if len(pnl) < 20:
            print(f"    {name:34} n={len(pnl):>4}  (thin)")
            return
        for cost in (0.0, 0.0003, 0.0006):
            net = pnl - cost
            print(f"    {name:34} n={len(pnl):>4}  cost{cost * 1e4:>3.0f}bp  "
                  f"avg {net.mean() * 1e4:+6.1f}bp  win {(net > 0).mean():.0%}  "
                  f"sum {net.sum() * 100:+6.1f}%")

    strategies = [
        ("charm sign only", lambda s: np.sign(s[f"c_oi_{md}"])),
        ("vanna sign only", lambda s: np.sign(s[f"v_oi_{md}"])),
        ("charm & vanna agree", lambda s: np.where(np.sign(s[f"c_oi_{md}"]) == np.sign(s[f"v_oi_{md}"]),
                                                    np.sign(s[f"c_oi_{md}"]), 0)),
        ("charm_z |z|>1", lambda s: np.where(s["c_z"].abs() > 1, np.sign(s["c_z"]), 0)),
        ("vanna_z |z|>1", lambda s: np.where(s["v_z"].abs() > 1, np.sign(s["v_z"]), 0)),
        ("charm sign, long-gamma only", lambda s: np.where(s["g_sign"] > 0, np.sign(s[f"c_oi_{md}"]), 0)),
        ("charm sign, short-gamma only", lambda s: np.where(s["g_sign"] < 0, np.sign(s[f"c_oi_{md}"]), 0)),
    ]
    for name, fn in strategies:
        print(f"\n  {name}")
        for is_lab, sub in (("IS ", df[df.date < split]), ("OOS", df[df.date >= split])):
            print(f"   [{is_lab}]")
            run(name, fn, sub)

    # per-ticker breakdown of the best generic (charm&vanna agree), OOS
    print("\n  --- per-ticker: charm&vanna agree, OOS ---")
    sub = df[df.date >= split]
    for tk in sorted(sub["ticker"].unique()):
        s = sub[sub["ticker"] == tk]
        sig = np.where(np.sign(s[f"c_oi_{md}"]) == np.sign(s[f"v_oi_{md}"]), np.sign(s[f"c_oi_{md}"]), 0)
        take = sig != 0
        if take.sum() < 10:
            continue
        pnl = np.sign(sig[take]) * s.loc[take, "r_aft"]
        print(f"    {tk:6} n={take.sum():>4}  avg {pnl.mean() * 1e4:+6.1f}bp  win {(pnl > 0).mean():.0%}")


def test_edge(a):
    """Consolidates the only three places anything showed a pulse in --test
    intraday/daily/power, so the (weak) result stays reproducible:
      (1) OpEx-day afternoon fade (a calendar effect, needs no greek read)
      (2) vanna_oi_z decile -> afternoon return (D10 = highest dealer vanna
          exposure = worst afternoon; consistent sign IS/OOS but decays ~3x OOS)
      (3) OpEx-WEEK Mon->Fri drift vs the week after (seasonal; the charm/vanna
          z-score does NOT predict which cycles -- sign flips IS/OOS)."""
    split = pd.Timestamp(a.split)
    df = _load_snap()
    md = 840
    df["r_aft"] = df[f"px_{CLOSE_MOD}"] / df[f"px_{md}"] - 1
    df["v_z"] = _zcol(df, f"v_oi_{md}")

    print("=" * 96)
    print("  (1) afternoon 14:00->15:55 by calendar bucket")
    print("=" * 96)
    for lab, m in (("OpEx day", df.opex), ("Thu/Fri non-OpEx", (df.dow >= 3) & ~df.opex),
                   ("Mon-Wed", df.dow <= 2)):
        for hl, sm in (("IS ", df.date < split), ("OOS", df.date >= split)):
            r = df[m & sm]["r_aft"].dropna()
            print(f"    {lab:18} [{hl}] n={len(r):>4}  mean {r.mean() * 1e4:+6.1f}bp  P(up) {(r > 0).mean():.0%}")

    print("\n" + "=" * 96)
    print("  (2) vanna_oi_z (14:00) decile  ->  afternoon 14:00->15:55")
    print("=" * 96)
    d = df[df["v_z"].notna() & df["r_aft"].notna()].copy()
    for hl, sm in (("IS ", d.date < split), ("OOS", d.date >= split)):
        s = d[sm].copy()
        s["dec"] = pd.qcut(s["v_z"], 10, labels=False, duplicates="drop")
        g = s.groupby("dec")["r_aft"].mean() * 1e4
        print(f"    [{hl}]  " + "  ".join(f"D{int(k) + 1}:{v:+.1f}" for k, v in g.items()))

    print("\n" + "=" * 96)
    print("  (3) OpEx-week (Mon->Fri) vs week-after drift, per ticker pooled")
    print("=" * 96)
    rows = []
    for tk in a.tickers:
        px = _load_1m(tk)
        if px is None:
            continue
        g = px.groupby("date")
        dd = pd.DataFrame({"open": g.first()["open"], "close": g.last()["close"]}).reset_index()
        dd["date"] = pd.to_datetime(dd["date"]); dd = dd.sort_values("date").reset_index(drop=True)
        fri = dd.index[(dd["date"].dt.weekday == 4) & dd["date"].dt.day.between(15, 21)]
        for j in fri:
            if j - 5 < 0 or j + 5 >= len(dd):
                continue
            rows.append((tk, dd.iloc[j]["date"],
                         dd.iloc[j]["close"] / dd.iloc[j - 4]["open"] - 1,
                         dd.iloc[j + 5]["close"] / dd.iloc[j + 1]["open"] - 1))
    R = pd.DataFrame(rows, columns=["tk", "fri", "r_week", "r_next"])
    for hl, sm in (("IS ", R.fri < split), ("OOS", R.fri >= split)):
        s = R[sm]
        print(f"    [{hl}] cycles={s.fri.nunique():>2} n={len(s):>3}  "
              f"opex-week {s.r_week.mean() * 100:+.2f}% (P(up) {(s.r_week > 0).mean():.0%})   "
              f"week-after {s.r_next.mean() * 100:+.2f}% (P(up) {(s.r_next > 0).mean():.0%})")


TESTS = {"intraday": test_intraday, "daily": test_daily, "power": test_power, "edge": test_edge}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", nargs="*", metavar="DATE")
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--tickers", nargs="+", default=CORE)
    ap.add_argument("--test", choices=list(TESTS))
    ap.add_argument("--split", default="2025-08-21")
    ap.add_argument("--decide", type=int, default=840, help="decision minute-of-day (720/780/840/870)")
    a = ap.parse_args()
    a.tickers = [t.upper() for t in a.tickers]
    if a.build is not None:
        build(a)
    elif a.snapshot:
        snapshot(a)
    elif a.test:
        TESTS[a.test](a)
    else:
        ap.print_help()
