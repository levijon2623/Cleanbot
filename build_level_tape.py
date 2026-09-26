# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
build_level_tape.py
===================
Per ticker-day STRUCTURAL LEVELS, cached once so the peak-location study (and
anything else) can be re-run without re-scanning the silver tape.

Built for the question "do trade PEAKS land on identifiable resistance?", so
every level here is CAUSAL -- knowable at or before the minute it is compared
against. That matters more than usual: a full-day volume profile would contain
the peak it is supposed to explain.

LEVELS PRODUCED
  call_wall / put_wall   strike carrying the largest +/- dealer gamma, profiled
                         from the silver option tape as of --asof (09:35) over
                         DTE 0-3. Frozen at the open ON PURPOSE: a wall measured
                         at the peak minute has absorbed the very move being
                         tested. `cw_late`/`pw_late` (15:00) are kept only as a
                         stability diagnostic, not for scoring.
  gamma_sign             sign of total profiled net gamma (+ suppression).
  wpoc                   5-day rolling volume POC, PRIOR sessions only.
  ppoc / pvah / pval     prior session's POC and value area.
  pdh / pdl              prior session's high / low.
  ib_hi / ib_lo          first 60m range (valid only from 10:30 on).
  vwap / vwap_sd         session VWAP and its volume-weighted sd, PER MINUTE
                         (cumulative, so causal by construction). Bands are
                         vwap +/- k*vwap_sd. Written as a separate per-minute
                         frame; the day frame carries the scalar levels.
  atr                    prior 14-session ATR, the natural distance unit -- a
                         level 30bp away means something different in IWM than
                         in SPY, and raw bp would smuggle in a vol proxy.

NOT PRODUCED, AND WHY
  zero-gamma LEVEL       The silver tape only carries contracts that TRADED, so
                         a cumulative-net-gamma zero crossing would be computed
                         over a biased subset of the chain (check_gamma_walls.py
                         :138 documents the same limit for walls-by-strike).
                         `gamma_sign` carries the REGIME, which the tape can
                         support; the flip LEVEL it cannot. Do not fake it.

Usage:
  python build_level_tape.py --tickers SPY QQQ IWM
  python build_level_tape.py --tickers SPY QQQ IWM NVDA META --from 2023-10-12
"""
from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from check_gamma_walls import profile_at, walls, _rth_min

LAKE_BARS = "lake/silver/option-contracts-1m"
HIST = "historical"
CACHE = "_level_cache"
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
BIN_PCT = 0.0005
WPOC_WINDOW = 5
IB_MINS = 60
VA_FRAC = 0.70


# ----------------------------------------------------------------- silver side
def silver_dates() -> list[date]:
    out = []
    for n in os.listdir(LAKE_BARS):
        if n.startswith("date="):
            try:
                out.append(date.fromisoformat(n[5:]))
            except ValueError:
                pass
    return sorted(out)


def load_multi(d: date, tickers: list[str], max_dte: int, min_dte: int):
    """One partition read for ALL tickers. Scanning per-ticker would multiply
    235 MB of I/O by the ticker count for nothing."""
    p = os.path.join(LAKE_BARS, f"date={d.isoformat()}", "bars.parquet")
    if not os.path.exists(p):
        return None
    lf = (pl.scan_parquet(p)
          .filter(pl.col("underlying_symbol").is_in(tickers))
          .select("underlying_symbol", "option_chain_id", "option_type", "strike",
                  "expiry", "minute_et", "gamma_close", "iv_close", "open_interest",
                  "underlying_close"))
    df = _rth_min(lf.collect())
    if df.is_empty():
        return None
    df = df.with_columns(
        (pl.col("expiry").cast(pl.Date) - pl.lit(d)).dt.total_days().alias("dte"),
        pl.col("strike").cast(pl.Float64),
    ).filter((pl.col("dte") >= min_dte) & (pl.col("dte") <= max_dte))
    return df if not df.is_empty() else None


# ------------------------------------------------------------- underlying side
def und_1m(tk: str) -> pd.DataFrame:
    d = pl.read_parquet(f"{HIST}/{tk}.parquet",
                        columns=["start_time", "open", "high", "low", "close", "volume"]).to_pandas()
    et = (pd.to_datetime(d["start_time"], utc=True)
          .dt.tz_convert("America/New_York").dt.tz_localize(None))
    d["date"] = et.dt.date
    d["mod"] = (et.dt.hour * 60 + et.dt.minute).astype(int)
    d = d[(d["mod"] >= RTH_LO) & (d["mod"] <= RTH_HI)].copy()
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["close"]).sort_values(["date", "mod"]).reset_index(drop=True)


def vwap_frame(px: pd.DataFrame) -> pd.DataFrame:
    """Per-minute session VWAP and volume-weighted sd. Cumulative within the day,
    so the value at minute m uses only minutes <= m."""
    g = px.groupby("date", sort=False)
    tp = (px["high"] + px["low"] + px["close"]) / 3.0
    v = px["volume"].fillna(0.0)
    pv, pv2 = tp * v, tp * tp * v
    cum_v = g["volume"].cumsum().replace(0, np.nan)
    cum_pv = pv.groupby(px["date"], sort=False).cumsum()
    cum_pv2 = pv2.groupby(px["date"], sort=False).cumsum()
    vw = cum_pv / cum_v
    var = (cum_pv2 / cum_v) - vw * vw
    return pd.DataFrame({"date": px["date"], "mod": px["mod"],
                         "vwap": vw.to_numpy(float),
                         "vwap_sd": np.sqrt(np.clip(var.to_numpy(float), 0, None))})


def day_profile(g: pd.DataFrame, binw: float):
    """POC / value area for ONE session (used only for the PRIOR day's profile)."""
    lo, hi = float(g["low"].min()), float(g["high"].max())
    if not np.isfinite(lo) or hi <= lo or binw <= 0:
        return None
    edges = np.arange(np.floor(lo / binw) * binw, hi + binw, binw)
    if len(edges) < 3:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    n = len(centers)
    vol = np.zeros(n)
    for l, h, vv in zip(g["low"].to_numpy(float), g["high"].to_numpy(float),
                        g["volume"].to_numpy(float)):
        a = max(int(np.searchsorted(edges, l, "right")) - 1, 0)
        z = min(int(np.searchsorted(edges, h, "right")) - 1, n - 1)
        if z >= a:
            vol[a:z + 1] += vv / (z - a + 1)
    if vol.sum() <= 0:
        return None
    i = int(np.argmax(vol))
    total, loi, hii, acc = vol.sum(), i, i, vol[i]
    while acc < VA_FRAC * total and (loi > 0 or hii < n - 1):
        up = vol[hii + 1] if hii < n - 1 else -1.0
        dn = vol[loi - 1] if loi > 0 else -1.0
        if hii < n - 1 and (up >= dn or loi == 0):
            hii += 1; acc += vol[hii]
        elif loi > 0:
            loi -= 1; acc += vol[loi]
        else:
            break
    return float(centers[i]), float(centers[hii]), float(centers[loi])


def wpoc_map(px: pd.DataFrame, binw: float) -> dict:
    """{date -> 5-session rolling POC}, built from PRIOR sessions only.

    The .shift(1) is not cosmetic: check_wpoc_gate.py:66 records that the
    reference implementation of this idea included the CURRENT day, which is not
    knowable intraday."""
    tp = (px["high"] + px["low"] + px["close"]) / 3.0
    b = np.floor(tp / binw).astype(np.int64)
    prof = (pd.DataFrame({"date": px["date"], "b": b, "v": px["volume"]})
            .groupby(["date", "b"], observed=True)["v"].sum().unstack(fill_value=0.0))
    roll = prof.rolling(WPOC_WINDOW, min_periods=WPOC_WINDOW).sum().shift(1).dropna(how="all")
    if roll.empty:
        return {}
    return {dt: (float(bn) + 0.5) * binw
            for dt, bn in roll.idxmax(axis=1).items() if pd.notnull(bn)}


def underlying_levels(tk: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = und_1m(tk)
    binw = round(float(px["close"].median()) * BIN_PCT, 4) or 0.01
    wp = wpoc_map(px, binw)
    vwf = vwap_frame(px)

    rows, prev = [], None
    for d, g in px.groupby("date", sort=True):
        ib = g[g["mod"] < RTH_LO + IB_MINS]
        cur = dict(date=d, open=float(g["close"].iloc[0]),
                   high=float(g["high"].max()), low=float(g["low"].min()),
                   close=float(g["close"].iloc[-1]),
                   ib_hi=float(ib["high"].max()) if len(ib) else np.nan,
                   ib_lo=float(ib["low"].min()) if len(ib) else np.nan,
                   wpoc=wp.get(d, np.nan))
        if prev is not None:
            cur.update(pdh=prev["high"], pdl=prev["low"],
                       ppoc=prev.get("poc", np.nan), pvah=prev.get("vah", np.nan),
                       pval=prev.get("val", np.nan))
        p = day_profile(g, binw)
        if p:
            cur["poc"], cur["vah"], cur["val"] = p
        rows.append(cur)
        prev = cur

    day = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # prior 14-session ATR: the distance unit. shift(1) keeps it causal.
    tr = (day["high"] - day["low"]).astype(float)
    day["atr"] = tr.rolling(14, min_periods=5).mean().shift(1)
    # `poc`/`vah`/`val` are SAME-DAY (they contain the future) -- they exist only
    # to feed the next day's p* columns. Drop them so nothing scores against them.
    day = day.drop(columns=[c for c in ("poc", "vah", "val") if c in day.columns])
    return day, vwf


# -------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--from", dest="d0", default="2024-08-20")
    ap.add_argument("--to", dest="d1", default="2026-08-21")
    ap.add_argument("--asof", default="09:35", help="minute the walls are frozen at")
    ap.add_argument("--late", default="15:00", help="second wall snapshot (diagnostic only)")
    ap.add_argument("--max-dte", type=int, default=3)
    ap.add_argument("--min-dte", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    d0, d1 = date.fromisoformat(a.d0), date.fromisoformat(a.d1)
    h, m = map(int, a.asof.split(":")); asof = h * 60 + m
    h, m = map(int, a.late.split(":")); late = h * 60 + m

    tag = f"{d0.isoformat()}_{d1.isoformat()}_dte{a.min_dte}-{a.max_dte}"
    wall_fp = os.path.join(CACHE, f"walls_{tag}.parquet")

    # ---- underlying levels (cheap, per ticker)
    for tk in a.tickers:
        fp = os.path.join(CACHE, f"und_{tk}.parquet")
        vp = os.path.join(CACHE, f"vwap_{tk}.parquet")
        if os.path.exists(fp) and os.path.exists(vp) and not a.force:
            print(f"  {tk} underlying levels cached")
            continue
        day, vwf = underlying_levels(tk)
        day.to_parquet(fp, index=False)
        vwf.to_parquet(vp, index=False)
        print(f"  {tk} underlying levels: {len(day)} sessions -> {fp}")

    # ---- walls (expensive, one partition read per day for all tickers)
    if os.path.exists(wall_fp) and not a.force:
        print(f"  walls cached -> {wall_fp}")
        return
    days = [d for d in silver_dates() if d0 <= d <= d1]
    print(f"  profiling walls over {len(days)} sessions x {len(a.tickers)} tickers "
          f"(one partition read each)", flush=True)
    rows = []
    for k, d in enumerate(days):
        big = load_multi(d, a.tickers, a.max_dte, a.min_dte)
        if big is None:
            continue
        for tk in a.tickers:
            day = big.filter(pl.col("underlying_symbol") == tk)
            if day.is_empty():
                continue
            rec = dict(ticker=tk, date=d)
            for lbl, mod in (("", asof), ("_late", late)):
                prof = profile_at(day, mod, bs_fill=True)
                if prof is None:
                    continue
                spot, by_strike, cov, tot = prof
                w = walls(spot, by_strike)
                if w is None:
                    continue
                cw, pw, sign = w
                rec[f"cw{lbl}"] = cw
                rec[f"pw{lbl}"] = pw
                rec[f"spot{lbl}"] = spot
                if not lbl:
                    rec["gamma_sign"] = sign
                    rec["gamma_cov"] = (cov / tot) if tot else np.nan
                    rec["net_gamma"] = float(sum(by_strike.values()))
            if "cw" in rec or "pw" in rec:
                rows.append(rec)
        if k % 20 == 0:
            print(f"    {d}  ({k+1}/{len(days)})  rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(wall_fp, index=False)
    print(f"\n  walls: {len(df)} ticker-days -> {wall_fp}")
    if not df.empty:
        for tk, g in df.groupby("ticker"):
            drift_c = ((g.get("cw_late") - g.get("cw")).abs() > 0).mean() * 100 \
                if "cw_late" in g else np.nan
            print(f"    {tk:5} {len(g):>4} days   gamma_cov median "
                  f"{g['gamma_cov'].median()*100:.0f}%   call wall moved by 15:00 on "
                  f"{drift_c:.0f}% of days")


if __name__ == "__main__":
    main()
