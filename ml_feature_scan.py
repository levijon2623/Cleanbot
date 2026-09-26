# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scikit-learn>=1.5", "python-dotenv"]
# ///
"""
ml_feature_scan.py
==================
OFFLINE RESEARCH TOOL -- never in the execution path.

Assembles one wide, lookahead-free feature vector per flow-trigger event (EMA(5)
cum-flow crossover, >= trailing-60d p50 flow) across the rule-universe tickers,
labels each with the option-bracket outcome (tr 1.0 / rr 1.0 / EOD, the same
`_bracket_pnl` used everywhere else), and asks a gradient-boosted forest which
market-state variables separate winning triggers from losing ones -- and whether
that separation holds OUT of sample.

Anything that (a) ranks high on OOS permutation importance, (b) has an IS/OOS
sign-consistent univariate effect, and (c) isn't already a rule gate, is a
candidate to hand-build into a RULES entry and validate the normal way
(check_screen_candidate.py).  The model itself is NOT a signal source.

Features are pulled from data already on disk:
  historical/NETPREM{T}, {T}, GEX{T}         flow, price, daily GEX regime
  _dgex_cache, _cv_cache                     1-min OI/vol/dir gamma + charm/vanna
  _atm_iv_cache                              ATM 0/1DTE IV by 15-min bucket
  _imb_cache                                 aggressor-imbalance constructions (4 tk)
  _tide_cache/market_tide                    market-wide net premium
  _rr_cache                                  25-delta risk-reversal skew (OOS only)
  + check_adx_dmi / check_regime_state / macro_calendar helpers (VIX, SPY state, DMI)

Usage:
  python ml_feature_scan.py --build            # -> _ml_cache/dataset.parquet
  python ml_feature_scan.py                    # analyze (build first if missing)
  python ml_feature_scan.py --direction PUT    # restrict the analysis
  python ml_feature_scan.py --live-only        # only (ticker,dir) combos with an enabled rule
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HIST = "historical"
CACHE = "_ml_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD = 15 * 60 + 55
FLOW_PCT_FLOOR = 50          # loosest gate any rule uses -- the "worth a look" bar
DEFAULT_TICKERS = ["AMZN", "AVGO", "GLD", "IWM", "META", "MSFT", "NVDA", "QQQ",
                   "SPY", "TSLA", "AAPL", "GOOGL"]


# ----------------------------------------------------------------------------
# feature assembly
# ----------------------------------------------------------------------------
def _asof_map(sorted_keys, sorted_vals, q):
    """value of sorted_vals at the last sorted_keys <= q (else nan)."""
    i = int(np.searchsorted(sorted_keys, q, side="right")) - 1
    return sorted_vals[i] if i >= 0 else np.nan


def _underlying_minute(tk):
    """{date: (mod_arr, spot_arr, sess_open, hi_cummax_arr, lo_cummin_arr)} RTH."""
    import polars as pl
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return {}
    d = pl.read_parquet(p, columns=["start_time", "open", "high", "low", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 960)].sort_values(["date", "mod"])
    out = {}
    for dt, g in d.groupby("date"):
        mod = g["mod"].to_numpy()
        out[dt] = (mod, g["close"].to_numpy(float), float(g["open"].to_numpy(float)[0]),
                   np.maximum.accumulate(g["high"].to_numpy(float)),
                   np.minimum.accumulate(g["low"].to_numpy(float)))
    return out


def _daily_close(tk):
    import polars as pl
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return pd.Series(dtype=float)
    d = pl.read_parquet(p, columns=["start_time", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 960)]
    s = d.sort_values(["date", "mod"]).groupby("date")["close"].last().astype(float)
    s.index = pd.to_datetime(list(s.index))
    return s


def _load_1m_cache(path, cols):
    import polars as pl
    if not os.path.exists(path):
        return None
    d = pl.read_parquet(path).to_pandas()
    if "minute_et" in d.columns:
        d["minute_et"] = pd.to_datetime(d["minute_et"])
        d["date"] = d["minute_et"].dt.date
        d["mod"] = d["minute_et"].dt.hour * 60 + d["minute_et"].dt.minute
    return d[[c for c in cols if c in d.columns] + (["date", "mod"] if "date" not in cols else [])]


def _norm_by_trailing(df, valcol, days=60):
    """normalize an intraday series column by the trailing-`days` median of that
    ticker's daily-open value (lookahead-free: prior days only)."""
    firsts = df.sort_values(["date", "mod"]).groupby("date")[valcol].first()
    scale = firsts.abs().rolling(days, min_periods=10).median().shift(1)
    m = df["date"].map(scale.to_dict())
    return df[valcol] / m.replace(0, np.nan)


_BARS_COLS = ["option_chain_id", "option_type", "strike", "expiry", "minute_et",
              "high", "low", "close", "bid_close", "ask_close", "underlying_close"]


def _load_bars_dte01(tk, D):
    """0/1-DTE ATM option bars for one ticker, streamed + filtered so peak memory
    stays small (the QQQ screen-cache parquet is ~300MB on disk)."""
    import polars as pl
    src = None
    sc = f"_screen_cache/{tk}_bars.parquet"
    if os.path.exists(sc):
        src = sc
    elif os.path.exists(D.BARS_CACHE):
        try:
            if (pl.scan_parquet(D.BARS_CACHE).filter(pl.col("underlying_symbol") == tk)
                    .select(pl.len()).collect().item()) > 0:
                src = D.BARS_CACHE
        except Exception:
            src = None
    if src is None:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        if tb is None or tb.empty:
            return None
        tb["_dte"] = (pd.to_datetime(tb["expiry"]).dt.normalize() - pd.to_datetime(tb["date"])).dt.days
        return tb[(tb["_dte"] >= 0) & (tb["_dte"] <= 1)][_BARS_COLS + ["date"]].reset_index(drop=True)
    lf = pl.scan_parquet(src)
    if src == D.BARS_CACHE:
        lf = lf.filter(pl.col("underlying_symbol") == tk)
    lf = lf.with_columns(
        ((pl.col("expiry").cast(pl.Date) - pl.col("minute_et").cast(pl.Date)).dt.total_days()).alias("_dte")
    ).filter((pl.col("_dte") >= 0) & (pl.col("_dte") <= 1)).select([c for c in _BARS_COLS])
    tb = lf.collect().to_pandas()
    tb["minute_et"] = D._naive(tb["minute_et"])
    tb["date"] = tb["minute_et"].dt.date
    tb["expiry"] = pd.to_datetime(tb["expiry"]).dt.date
    return tb


def _opex_map(dates):
    """{date: trading-days until the next monthly 3rd-Friday opex}."""
    ds = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in dates}))
    out = {}
    for d in ds:
        # 3rd Friday of this month
        first = d.replace(day=1)
        fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
        third = fri + pd.Timedelta(days=14)
        if d.date() > third.date():
            nm = (first + pd.offsets.MonthBegin(1))
            fri = nm + pd.Timedelta(days=(4 - nm.weekday()) % 7)
            third = fri + pd.Timedelta(days=14)
        out[d.date()] = int(np.busday_count(d.date(), third.date()))
    return out


def build(args):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from check_adx_dmi import _daily_adx, _intraday_adx, _intra_asof
    from check_regime_state import _spy_daily, _spy_gex_sign, _vix
    from check_flow_zscore import annotate_flow_z
    from macro_calendar import is_macro_am_day, is_fomc_day
    from config import RULES

    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(f"{CACHE}/parts", exist_ok=True)
    tickers = [t.upper() for t in (args.tickers or DEFAULT_TICKERS)]
    live = {(r["ticker"], r["direction"]) for r in RULES if r.get("enabled", True)}

    # ---- market-wide, once ----
    spy = _spy_daily().set_index("date")
    spy_gex = _spy_gex_sign()
    vix_s, dvix_s = _vix()                    # prior-day level, prior 5d change (already shifted)
    vix_med = vix_s.rolling(60, min_periods=20).median()
    spy_min = _underlying_minute("SPY")
    try:
        import polars as pl
        td = pl.read_parquet("_tide_cache/market_tide.parquet").to_pandas()
        td["ts"] = pd.to_datetime(td["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        td["date"] = td["ts"].dt.date
        td["mod"] = td["ts"].dt.hour * 60 + td["ts"].dt.minute
        td = td.sort_values(["date", "mod"])
        td["cum"] = td.groupby("date", group_keys=False)["ncp"].cumsum() - td.groupby("date", group_keys=False)["npp"].cumsum()
        tide_by_day = {d: (g["mod"].to_numpy(), g["cum"].to_numpy(float)) for d, g in td.groupby("date")}
        _tl = td.groupby("date")["cum"].last().abs()
        _tl.index = pd.to_datetime(list(_tl.index))
        tide_scale = {d.date(): v for d, v in _tl.rolling(60, min_periods=10).median().shift(1).items()}
    except Exception as e:
        print(f"  (market tide unavailable: {e})")
        tide_by_day, tide_scale = {}, {}

    for tk in tickers:
        part = f"{CACHE}/parts/{tk}.parquet"
        if os.path.exists(part) and not args.rebuild:
            print(f"  {tk}: part exists -- skip (use --rebuild to force)", flush=True)
            continue
        rows = []
        flow = _flow_for(D, [tk])
        if flow.empty:
            print(f"  {tk}: no NETPREM -- skipped", flush=True); continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        annotate_flow_z(trigs, 45)
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        dadx = _daily_adx(tk)
        iadx = _intraday_adx(tk, 15, close_only=True)
        umin = _underlying_minute(tk)
        dc = _daily_close(tk)
        rv20 = (dc.pct_change().rolling(20).std() * np.sqrt(252)).shift(1)
        rv20 = {d.date(): v for d, v in rv20.items()}
        prev_close = dc.shift(1)
        prev_close = {d.date(): v for d, v in prev_close.items()}

        dgex = _load_1m_cache(f"_dgex_cache/{tk}.parquet",
                              ["minute_et", "gamma_per_one_percent_move_oi", "gamma_per_one_percent_move_vol",
                               "gamma_per_one_percent_move_dir"])
        if dgex is not None:
            dgex = dgex.rename(columns={"gamma_per_one_percent_move_oi": "g_oi",
                                        "gamma_per_one_percent_move_vol": "g_vol",
                                        "gamma_per_one_percent_move_dir": "g_dir"})
            dgex["g_oi_n"] = _norm_by_trailing(dgex, "g_oi")
            dgex["g_dir_n"] = dgex["g_dir"] / dgex.groupby("date")["g_oi"].transform("first").abs().replace(0, np.nan)
            dgex_by_day = {d: (g.sort_values("mod")["mod"].to_numpy(),
                               g.sort_values("mod")[["g_oi_n", "g_dir_n", "g_dir"]].to_numpy(float))
                           for d, g in dgex.groupby("date")}
        else:
            dgex_by_day = {}

        cv = _load_1m_cache(f"_cv_cache/{tk}.parquet", ["minute_et", "g_oi", "c_oi", "v_oi"])
        if cv is not None:
            cv["charm_n"] = cv["c_oi"] / cv["g_oi"].abs().replace(0, np.nan)
            cv["vanna_n"] = cv["v_oi"] / cv["g_oi"].abs().replace(0, np.nan)
            cv_by_day = {d: (g.sort_values("mod")["mod"].to_numpy(),
                             g.sort_values("mod")[["charm_n", "vanna_n"]].to_numpy(float))
                         for d, g in cv.groupby("date")}
        else:
            cv_by_day = {}

        aiv_days = {}
        if os.path.exists(f"_atm_iv_cache/{tk}.parquet"):
            import polars as pl
            aiv = pl.read_parquet(f"_atm_iv_cache/{tk}.parquet").to_pandas()
            aiv["date"] = pd.to_datetime(aiv["date"]).dt.date
            for r in aiv.itertuples():
                aiv_days.setdefault(r.date, []).append((int(r.mod15), float(r.iv_close)))
            for dd in aiv_days:
                aiv_days[dd].sort()

        imb = None
        for f in (os.listdir("_imb_cache") if os.path.isdir("_imb_cache") else []):
            if f.startswith(f"{tk}_"):
                import polars as pl
                imb = pl.read_parquet(f"_imb_cache/{f}").to_pandas()
                break
        if imb is not None:
            imb["date"] = pd.to_datetime(imb["date"]).dt.date
            imb["ratio"] = imb["signed_ct"] / imb["gross_ct"].replace(0, np.nan)
            imb["cum_ratio"] = imb.groupby("date")["ratio"].cumsum() / (imb.groupby("date").cumcount() + 1)
            imb_by_day = {d: (g.sort_values("mod")["mod"].to_numpy(),
                              g.sort_values("mod")["cum_ratio"].to_numpy(float))
                          for d, g in imb.groupby("date")}
        else:
            imb_by_day = {}

        try:
            import polars as pl
            rr = pl.read_parquet(f"_rr_cache/{tk}_dte30.parquet").to_pandas()
            rr["date"] = pd.to_datetime(rr["date"]).dt.date
            rr = rr.sort_values("date")
            rr["rr_d5"] = rr["rr"].diff(5)
            rr_lut = {r.date: (r.rr, r.rr_d5) for r in rr.itertuples()}
            rr_dates = sorted(rr_lut)
        except Exception:
            rr_lut, rr_dates = {}, []

        tb = _load_bars_dte01(tk, D)
        if tb is None or tb.empty:
            print(f"  {tk}: no option bars -- skipped", flush=True); continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {dd: g for dd, g in tb.groupby("date")}
        del tb, flow
        gc.collect()

        opex = _opex_map([t["date"] for t in trigs])
        n_kept = 0
        seq_ct = {}
        for t in trigs:
            thr = t.get("thr")
            if not thr or t["abs_flow"] < thr.get(FLOW_PCT_FLOOR, 1e99):
                continue
            d, ts = t["date"], t["ts"]
            mod = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
            direction = t["dir"]
            seq_ct[d] = seq_ct.get(d, 0) + 1

            # flow percentile (continuous-ish: interpolate rank of abs_flow in thr dict)
            ps = sorted(thr.items())
            fp = np.interp(t["abs_flow"], [v for _, v in ps], [k for k, _ in ps],
                           left=ps[0][0] - 10, right=ps[-1][0] + 5)
            fz = ((t["abs_flow"] - t["mu"]) / t["sd"]) if t.get("mu") and t.get("sd") else np.nan

            # ticker price state as-of trigger
            gap = ir = rf = np.nan
            um = umin.get(d)
            pc = prev_close.get(d)
            if um is not None:
                marr, sarr, so, hicm, locm = um
                j = int(np.searchsorted(marr, mod, side="right")) - 1
                if j >= 0:
                    spot = sarr[j]
                    ir = spot / so - 1.0
                    rf = (hicm[j] - locm[j]) / so
                    if pc and np.isfinite(pc):
                        gap = so / pc - 1.0
            # signed so + = price is moving WITH the trade (CALL up / PUT down)
            ret_with = (ir if direction == "CALL" else -ir) if np.isfinite(ir) else np.nan

            # DMI (signed so + = OPPOSES the trade -> a fade setup)
            ia = _intra_asof(iadx.get(d, []), ts)
            da = dadx.get(d)
            def _opp(pdi, mdi):
                s = pdi - mdi
                return s if direction == "PUT" else -s
            dmi_i = _opp(ia[0], ia[1]) if ia else np.nan
            adx_i = ia[2] if ia else np.nan
            dmi_d = _opp(da[0], da[1]) if da else np.nan
            adx_d = da[2] if da else np.nan

            rv = rv20.get(d, np.nan)
            iv = np.nan
            if d in aiv_days:
                b = aiv_days[d]
                mk = mod - (mod % 15)
                iv = _asof_map(np.array([x[0] for x in b]), np.array([x[1] for x in b]), mk)

            gexn = gdirn = gdir_act = agree = np.nan
            if d in dgex_by_day:
                mm, vv = dgex_by_day[d]
                k = int(np.searchsorted(mm, mod, side="right")) - 1
                if k >= 0:
                    gexn, gdirn, graw = vv[k]
                    gdir_act = float(abs(graw) > 0)
                    agree = np.sign(gexn) * np.sign(gdirn) if abs(graw) > 0 else 0.0
            charm = vanna = np.nan
            if d in cv_by_day:
                mm, vv = cv_by_day[d]
                k = int(np.searchsorted(mm, mod, side="right")) - 1
                if k >= 0:
                    charm, vanna = vv[k]

            imb_r = np.nan
            if d in imb_by_day:
                mm, vv = imb_by_day[d]
                k = int(np.searchsorted(mm, mod, side="right")) - 1
                if k >= 0:
                    imb_r = vv[k]

            # SPY state
            _ts = pd.Timestamp(d)
            srow = spy.loc[_ts] if _ts in spy.index else None
            spy_ir = np.nan
            sm = spy_min.get(d)
            if sm is not None:
                marr, sarr, so, _, _ = sm
                j = int(np.searchsorted(marr, mod, side="right")) - 1
                if j >= 0:
                    spy_ir = sarr[j] / so - 1.0
            tide = np.nan
            if d in tide_by_day:
                mm, vv = tide_by_day[d]
                k = int(np.searchsorted(mm, mod, side="right")) - 1
                if k >= 0:
                    sc = tide_scale.get(d, np.nan)
                    tide = vv[k] / sc if sc and np.isfinite(sc) else np.nan

            vix = vix_s.get(pd.Timestamp(d).date(), np.nan) if len(vix_s) else np.nan
            vmed = vix_med.get(pd.Timestamp(d).date(), np.nan) if len(vix_med) else np.nan
            vd5 = dvix_s.get(pd.Timestamp(d).date(), np.nan) if len(dvix_s) else np.nan

            rr_v = rr_d5 = np.nan
            if rr_dates:
                i = int(np.searchsorted(rr_dates, d, side="right")) - 1
                if i >= 0:
                    rr_v, rr_d5 = rr_lut[rr_dates[i]]

            base = dict(
                ticker=tk, direction=1 if direction == "CALL" else 0,
                is_live_combo=int((tk, direction) in live),
                hour=pd.Timestamp(ts).hour, mod=mod, mins_since_open=mod - 570,
                flow_pct=float(fp), flow_z=float(fz) if np.isfinite(fz) else np.nan,
                flow_log=float(np.log10(max(t["abs_flow"], 1.0))),
                trig_seq=seq_ct[d],
                vol_regime={"LOWVOL": 0, "NORMVOL": 1, "HIVOL": 2}.get(vol.get(d), np.nan),
                trend_regime={"DOWNTREND": -1, "CHOP": 0, "UPTREND": 1}.get(trd.get(d), np.nan),
                gex_sign=1.0 if gex.get(d) == "POSITIVE" else (-1.0 if gex.get(d) == "NEGATIVE" else np.nan),
                amp=amp.get(d, np.nan),
                gap=gap, intraday_ret=ir, range_frac=rf, ret_with_trade=ret_with,
                dmi_intra_opp=dmi_i, adx_intra=adx_i, dmi_daily_opp=dmi_d, adx_daily=adx_d,
                rv20=rv, atm_iv=iv, iv_rv=(iv / rv) if (np.isfinite(iv) and rv and np.isfinite(rv)) else np.nan,
                gex_oi_n=gexn, gex_dir_n=gdirn, gex_dir_active=gdir_act, gex_oi_dir_agree=agree,
                charm_n=charm, vanna_n=vanna, imb_cum_ratio=imb_r,
                spy_er1d=srow["er_1d"] if srow is not None else np.nan,
                spy_er10d=srow["er_10d"] if srow is not None else np.nan,
                spy_rv20=srow["rv_20d"] if srow is not None else np.nan,
                spy_drv5=srow["drv_5d"] if srow is not None else np.nan,
                spy_gap=srow["gap"] if srow is not None else np.nan,
                spy_intra_ret=spy_ir,
                spy_gex_sign=1.0 if spy_gex.get(d) == "POSITIVE" else (-1.0 if spy_gex.get(d) == "NEGATIVE" else np.nan),
                market_tide=tide,
                vix=vix, vix_vs_med=(vix - vmed) if (np.isfinite(vix) and np.isfinite(vmed)) else np.nan, vix_d5=vd5,
                dow=pd.Timestamp(ts).weekday(),
                is_macro_am=int(is_macro_am_day(d)), is_fomc=int(is_fomc_day(d)),
                dte_to_opex=opex.get(d, np.nan),
                rr_skew=rr_v, rr_skew_d5=rr_d5,
            )
            for dd in (0, 1):
                paths = D._option_paths(t, direction, [dd], bbd, bbc)
                for pth in paths:
                    pnl = D._bracket_pnl(*pth, 1.0, 1.0, None, EOD)
                    rows.append({**base, "dte": dd, "entry_prem": float(pth[0]),
                                 "entry_prem_log": float(np.log10(max(pth[0], 0.01))),
                                 "date": d, "y_pnl": float(pnl), "y_win": int(pnl > 0)})
                    n_kept += 1
        pd.DataFrame(rows).to_parquet(part, index=False)
        print(f"  {tk}: {n_kept} labelled trigger-rows -> {part}", flush=True)
        del rows, bbc, bbd, trigs, dgex_by_day, cv_by_day, aiv_days, imb_by_day
        gc.collect()

    parts = [f"{CACHE}/parts/{t}.parquet" for t in tickers if os.path.exists(f"{CACHE}/parts/{t}.parquet")]
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(f"{CACHE}/dataset.parquet", index=False)
    print(f"\n  wrote {CACHE}/dataset.parquet  ({len(df)} rows, {df['date'].min()} .. {df['date'].max()})")
    print(f"  base win rate {df['y_win'].mean():.3f}   mean pnl {df['y_pnl'].mean()*100:+.1f}%")


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 60:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def analyze(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    df = pd.read_parquet(f"{CACHE}/dataset.parquet")
    if args.tickers:
        df = df[df["ticker"].isin([t.upper() for t in args.tickers])]
    if args.direction:
        df = df[df["direction"] == (1 if args.direction.upper() == "CALL" else 0)]
    if args.live_only:
        df = df[df["is_live_combo"] == 1]
    df = df.sort_values("date").reset_index(drop=True)

    y = df["y_win"].to_numpy()
    ycont = df["y_pnl"].to_numpy()
    drop = {"y_win", "y_pnl", "date", "ticker", "is_live_combo"}
    feats = [c for c in df.columns if c not in drop]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    # ticker as an integer code (HGB treats it numeric; fine for a splitter)
    X["ticker_code"] = df["ticker"].astype("category").cat.codes
    feats = feats + ["ticker_code"]

    tr = (df["date"] < SPLIT).to_numpy()
    te = ~tr
    # HistGBM needs >=2 distinct values per column IN THE TRAINING SET
    dead = [c for c in feats if X.loc[tr, c].nunique(dropna=True) < 2]
    if dead:
        print(f"  (dropping {len(dead)} features with <2 distinct IS values: {', '.join(dead)})")
        feats = [c for c in feats if c not in dead]
        X = X[feats]
    print("=" * 104)
    print(f"  ML FEATURE SCAN   {len(df)} rows  ({tr.sum()} IS / {te.sum()} OOS)   "
          f"{'dir=' + args.direction if args.direction else 'both dirs'}"
          f"{' LIVE-combos-only' if args.live_only else ''}")
    print(f"  base win rate: IS {y[tr].mean():.3f}  OOS {y[te].mean():.3f}    "
          f"mean pnl: IS {ycont[tr].mean()*100:+.1f}%  OOS {ycont[te].mean()*100:+.1f}%")
    print("=" * 104)

    mdl = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.03, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=60, early_stopping=True, validation_fraction=0.15, random_state=0)
    mdl.fit(X[tr], y[tr])

    p_te = mdl.predict_proba(X[te])[:, 1]
    p_tr = mdl.predict_proba(X[tr])[:, 1]
    auc_tr = roc_auc_score(y[tr], p_tr)
    auc_te = roc_auc_score(y[te], p_te)
    print(f"\n  model ROC-AUC:  IS {auc_tr:.3f}   OOS {auc_te:.3f}   "
          f"({'no OOS signal -- importances below are noise' if auc_te < 0.53 else 'has some OOS signal'})")

    # does the model's probability, used as a FILTER, beat the unconditional book OOS?
    print("\n  -- model prob as an OOS trade filter (top-K% by P(win)) --")
    order = np.argsort(-p_te)
    pnl_te = ycont[te]
    for frac in (1.0, 0.6, 0.4, 0.25):
        k = max(20, int(len(order) * frac))
        sel = pnl_te[order[:k]]
        print(f"    keep top {frac*100:4.0f}%  n={k:5}  mean pnl {sel.mean()*100:+6.1f}%  win {(sel>0).mean():.3f}")

    # permutation importance on OOS (the honest ranking)
    print("\n  -- OOS permutation importance (mean AUC drop over 12 shuffles) --")
    pi = permutation_importance(mdl, X[te], y[te], scoring="roc_auc",
                                n_repeats=12, random_state=0, n_jobs=2)
    imp = pd.DataFrame({"feature": feats, "imp": pi.importances_mean, "sd": pi.importances_std})
    imp = imp.sort_values("imp", ascending=False).reset_index(drop=True)

    # univariate IS/OOS spearman(feature, pnl) for context
    rho_is = {c: _spear(X.loc[tr, c], ycont[tr]) for c in feats}
    rho_oos = {c: _spear(X.loc[te, c], ycont[te]) for c in feats}
    imp["rho_IS"] = imp["feature"].map(rho_is)
    imp["rho_OOS"] = imp["feature"].map(rho_oos)
    imp["consistent"] = np.sign(imp["rho_IS"]) == np.sign(imp["rho_OOS"])

    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(imp.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # top features -> univariate quintile expectancy IS vs OOS
    print("\n  -- top-12 features: mean pnl by quintile  (IS | OOS) --")
    for f in imp["feature"].head(12):
        s = X[f]
        if s.notna().sum() < 200 or s.nunique() < 5:
            lo = s.dropna()
            if lo.nunique() <= 6:
                cells = []
                for v in sorted(lo.unique()):
                    mi = (X[f] == v) & tr
                    mo = (X[f] == v) & te
                    cells.append(f"{v:g}: {ycont[mi].mean()*100:+.0f}|{ycont[mo].mean()*100:+.0f}")
                print(f"    {f:18} " + "  ".join(cells))
            continue
        try:
            q = pd.qcut(s, 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        cells = []
        for k in range(int(np.nanmax(q)) + 1):
            mi = (q == k).to_numpy() & tr
            mo = (q == k).to_numpy() & te
            cells.append(f"Q{k+1} {ycont[mi].mean()*100:+.0f}|{ycont[mo].mean()*100:+.0f}")
        print(f"    {f:18} " + "  ".join(cells))

    print("\n  -- CANDIDATES: OOS-important, IS/OOS sign-consistent, |rho|>=0.03 --")
    cand = imp[(imp["imp"] > imp["imp"].iloc[:3].mean() * 0.25) & imp["consistent"]
               & (imp["rho_OOS"].abs() >= 0.03) & (imp["rho_IS"].abs() >= 0.02)]
    if cand.empty:
        print("    (none clear -- the trigger outcome may be close to irreducible on these features)")
    else:
        for _, r in cand.iterrows():
            direction = "higher = better" if r["rho_OOS"] > 0 else "lower = better"
            print(f"    {r['feature']:18}  rho IS {r['rho_IS']:+.3f} / OOS {r['rho_OOS']:+.3f}   {direction}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="force-rebuild existing per-ticker parts")
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--direction", choices=("CALL", "PUT"), default=None)
    ap.add_argument("--live-only", action="store_true")
    a = ap.parse_args()
    if a.build or not os.path.exists(f"{CACHE}/dataset.parquet"):
        build(a)
    if not a.build:
        analyze(a)
