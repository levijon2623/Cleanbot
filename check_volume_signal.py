# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_volume_signal.py
======================
The one untested dimension: UNDERLYING VOLUME.

check_signal_quality showed the flow trigger has ~no standalone directional
edge on the stock (MFE/MAE 1.00-1.02 vs a 1.00 random null, hit 47-49%).
Every feature tested so far -- greeks, GEX, IV, DMI, tide, RR skew -- is
priced/derived. Underlying volume was never used: the repo only consumes it as
a DAILY LOW/NORM/HIVOL regime label, and ml_feature_scan's ~45 features
included none of it. The minute series (`volume`, `total_volume` in
historical/{T}.parquet, 04:00-19:59 ET, no nulls) has never entered a test.

Volume is a PARTICIPATION measure, not a price derivative, so unlike delta
(which we showed is contemporaneous with the move) it could plausibly lead.

Four causal constructions, all known at the trigger minute:

  relvol      trailing-5min share volume / the trailing-20-SESSION average for
              that same minute-of-day  (is the trigger firing into real
              participation or a quiet tape?)
  vwap_dist   signed sgn*(spot/session_VWAP - 1): + = price already on our side
              of VWAP. Also reported unsigned.
  volconf     "volume-confirmed flow": flow percentile x relvol, cross-tabbed --
              does big options flow WITH stock-side participation behave
              differently from big flow without it?
  cumvol_pace session volume so far / trailing-20-session average by that
              minute-of-day (is the whole day unusually busy?)

Scored exactly like check_signal_quality: direction-adjusted underlying drift,
hit rate and MFE/MAE at +60m, day-level, IS/OOS, against the same random-entry
null (which calibrates at MFE/MAE 1.00, hit ~50%).

Usage:  python check_volume_signal.py [--tickers ...]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
CLOSE_MOD = 15 * 60 + 55
OPEN_MOD = 9 * 60 + 30
LOOKBACK = 20          # sessions for the minute-of-day baselines


def _load(tk):
    """RTH minute frame + the causal volume baselines."""
    import polars as pl
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p, columns=["start_time", "close", "volume"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= OPEN_MOD) & (d["mod"] <= 960)].copy()
    d["volume"] = d["volume"].astype(float).fillna(0.0)
    d = d.sort_values(["date", "mod"])

    # session VWAP + cumulative volume (causal within the day)
    d["pv"] = d["close"] * d["volume"]
    g = d.groupby("date")
    d["cumv"] = g["volume"].cumsum()
    d["vwap"] = g["pv"].cumsum() / d["cumv"].replace(0, np.nan)
    # trailing-5min volume
    d["v5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=1).sum())

    # minute-of-day baselines from PRIOR sessions only
    piv = d.pivot_table(index="date", columns="mod", values="volume", aggfunc="last")
    base = piv.rolling(LOOKBACK, min_periods=5).mean().shift(1)
    pivc = d.pivot_table(index="date", columns="mod", values="cumv", aggfunc="last")
    basec = pivc.rolling(LOOKBACK, min_periods=5).mean().shift(1)
    # 5-min baseline = 5 x the per-minute baseline (same units as v5)
    bmap = {(dt, m): base.at[dt, m] for dt in base.index for m in base.columns}
    cmap = {(dt, m): basec.at[dt, m] for dt in basec.index for m in basec.columns}

    out = {}
    for dt, gg in d.groupby("date"):
        out[dt] = dict(mod=gg["mod"].to_numpy(),
                       close=gg["close"].to_numpy(float),
                       v5=gg["v5"].to_numpy(float),
                       cumv=gg["cumv"].to_numpy(float),
                       vwap=gg["vwap"].to_numpy(float))
    return out, bmap, cmap


def _idx(arr, m):
    i = int(np.searchsorted(arr["mod"], m, side="right")) - 1
    return i if i >= 0 else None


def _exc(arr, i, sgn, horizon):
    s0 = arr["close"][i]
    if not np.isfinite(s0) or s0 <= 0:
        return None
    j = int(np.searchsorted(arr["mod"], arr["mod"][i] + horizon, side="right"))
    w = arr["close"][i:j]
    if len(w) < 2:
        return None
    r = sgn * (w / s0 - 1.0)
    return float(r[-1]), float(np.max(r)), float(np.min(r))


def _report(lbl, sub):
    v = sub["r60"].dropna()
    if len(v) < 40:
        return f"      {lbl:26} n={len(v):>5}  (thin)"
    i = sub[sub.date < SPLIT]["r60"].dropna()
    o = sub[sub.date >= SPLIT]["r60"].dropna()
    mfe, mae = sub["mfe"].dropna(), sub["mae"].dropna()
    ratio = abs(mfe.mean() / mae.mean()) if len(mae) and abs(mae.mean()) > 1e-12 else float("nan")
    return (f"      {lbl:26} n={len(v):>5}  {v.mean()*1e4:>+5.1f}bp  hit {(v>0).mean()*100:>4.1f}%  "
            f"IS {i.mean()*1e4 if len(i) else float('nan'):>+5.1f}/{(i>0).mean()*100 if len(i) else float('nan'):>4.1f}%  "
            f"OOS {o.mean()*1e4 if len(o) else float('nan'):>+5.1f}/{(o>0).mean()*100 if len(o) else float('nan'):>4.1f}%  "
            f"MFE/MAE {ratio:>4.2f}")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES

    tickers = ([t.upper() for t in a.tickers] if a.tickers
               else sorted({r["ticker"] for r in RULES if r.get("enabled", True)}))
    rows, nullrows = [], []
    rng = np.random.default_rng(11)

    for tk in tickers:
        loaded = _load(tk)
        if loaded is None:
            continue
        days, bmap, cmap = loaded
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)

        for t in trigs:
            thr = t.get("thr")
            if not thr or t["abs_flow"] < thr.get(50, 1e99):
                continue
            d, ts = t["date"], t["ts"]
            arr = days.get(d)
            if arr is None:
                continue
            m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
            if m > CLOSE_MOD - 60:
                continue
            i = _idx(arr, m)
            if i is None:
                continue
            e = _exc(arr, i, 1.0 if t["dir"] == "CALL" else -1.0, 60)
            if e is None:
                continue
            b5 = bmap.get((d, arr["mod"][i]))
            bc = cmap.get((d, arr["mod"][i]))
            vw = arr["vwap"][i]
            s0 = arr["close"][i]
            sgn = 1.0 if t["dir"] == "CALL" else -1.0
            rows.append(dict(
                ticker=tk, date=d, mod=m, dir=t["dir"],
                r60=e[0], mfe=e[1], mae=e[2],
                relvol=(arr["v5"][i] / (5.0 * b5)) if (b5 and np.isfinite(b5) and b5 > 0) else np.nan,
                cumpace=(arr["cumv"][i] / bc) if (bc and np.isfinite(bc) and bc > 0) else np.nan,
                vwap_sgn=(sgn * (s0 / vw - 1.0)) if (np.isfinite(vw) and vw > 0) else np.nan,
                vwap_abs=(abs(s0 / vw - 1.0)) if (np.isfinite(vw) and vw > 0) else np.nan,
                fpct=(90 if t["abs_flow"] >= thr.get(90, 1e99) else
                      80 if t["abs_flow"] >= thr.get(80, 1e99) else
                      65 if t["abs_flow"] >= thr.get(65, 1e99) else 50),
            ))

        for d, arr in days.items():
            if len(arr["mod"]) < 130:
                continue
            for _ in range(8):
                m = int(rng.choice(arr["mod"][(arr["mod"] >= 575) & (arr["mod"] <= CLOSE_MOD - 70)]))
                i = _idx(arr, m)
                if i is None:
                    continue
                sgn = float(rng.choice([1.0, -1.0]))
                e = _exc(arr, i, sgn, 60)
                if not e:
                    continue
                vw, s0 = arr["vwap"][i], arr["close"][i]
                nullrows.append(dict(
                    ticker=tk, date=d, mod=m, r60=e[0], mfe=e[1], mae=e[2],
                    vwap_sgn=(sgn * (s0 / vw - 1.0)) if (np.isfinite(vw) and vw > 0) else np.nan))

    R = pd.DataFrame(rows)
    NL = pd.DataFrame(nullrows)
    if R.empty:
        print("no triggers"); return

    def dd(df, extra=()):
        num = ["r60", "mfe", "mae"] + [c for c in ("relvol", "cumpace", "vwap_sgn", "vwap_abs") if c in df]
        keys = ["ticker", "date"] + list(extra)
        g = df.groupby(keys)[num].mean().reset_index()
        g["date"] = pd.to_datetime(g["date"]).dt.date
        return g

    print("=" * 122)
    print(f"  UNDERLYING-VOLUME SIGNAL TESTS   {len(R)} triggers, {len(tickers)} tickers   "
          f"day-level, +60m   split {SPLIT}")
    print("=" * 122)
    print(_report("RANDOM null", dd(NL)))
    print(_report("all triggers (p50+)", dd(R)))

    G = dd(R)
    print("\n  -- 1. RELATIVE VOLUME (trailing 5m vs 20-session same-minute baseline) --")
    for lbl, sub in _quint(G, "relvol"):
        print(_report(lbl, sub))

    print("\n  -- 2. VWAP DISTANCE (signed: + = price already on the trade's side) --")
    for lbl, sub in _quint(G, "vwap_sgn"):
        print(_report(lbl, sub))
    print("     unsigned |distance from VWAP|:")
    for lbl, sub in _quint(G, "vwap_abs"):
        print(_report(lbl, sub))

    print("\n  -- 3. CUMULATIVE VOLUME PACE (session-to-date vs 20-session norm) --")
    for lbl, sub in _quint(G, "cumpace"):
        print(_report(lbl, sub))

    print("\n  -- 4. VOLUME-CONFIRMED FLOW (flow percentile x relvol tercile) --")
    GT = dd(R, extra=("fpct",))
    GT["rv_t"] = pd.qcut(GT["relvol"], 3, labels=["lo", "mid", "hi"], duplicates="drop")
    for fp in (50, 65, 80, 90):
        for rv in ("lo", "mid", "hi"):
            s = GT[(GT["fpct"] == fp) & (GT["rv_t"] == rv)]
            if len(s) >= 60:
                print(_report(f"flow p{fp} x relvol {rv}", s))

    # ---- THE CONTROL: is VWAP-position just intraday mean reversion that any
    # random entry would capture, or does the flow trigger add to it? ----
    print("\n  -- 2b. CONTROL: same VWAP conditioning on RANDOM entries/directions --")
    if "vwap_sgn" in NL.columns:
        NG = NL.groupby(["ticker", "date"])[["r60", "mfe", "mae", "vwap_sgn"]].mean().reset_index()
        NG["date"] = pd.to_datetime(NG["date"]).dt.date
        for lbl, sub in _quint(NG, "vwap_sgn"):
            print(_report("NULL " + lbl, sub))
        print("     -> if the null shows the SAME monotone pattern, VWAP position is generic")
        print("        intraday mean reversion, not something the flow trigger provides.")

    print("\n  -- 2c. TRADE-LEVEL (no day averaging) VWAP quintiles --")
    RT = R.dropna(subset=["vwap_sgn"]).copy()
    RT["date"] = pd.to_datetime(RT["date"]).dt.date
    if len(RT) > 500:
        q = pd.qcut(RT["vwap_sgn"], 5, labels=False, duplicates="drop")
        for k in sorted(set(q.dropna())):
            print(_report(f"trade-level vwap Q{int(k)+1}", RT[q == k]))

    print("\n  -- 5. best cell, per-ticker stability check --")
    hi = G[G["relvol"] >= G["relvol"].quantile(0.8)]
    print(_report("relvol top 20% (all tk)", hi))
    for tk, s in hi.groupby("ticker"):
        if len(s) >= 40:
            i = s[s.date < SPLIT]["r60"]; o = s[s.date >= SPLIT]["r60"]
            print(f"        {tk:6} n={len(s):>4} {s.r60.mean()*1e4:>+5.1f}bp  "
                  f"IS {i.mean()*1e4 if len(i) else float('nan'):>+5.1f}  "
                  f"OOS {o.mean()*1e4 if len(o) else float('nan'):>+5.1f}")


def _quint(G, col, n=5):
    s = G.dropna(subset=[col])
    if len(s) < 200 or s[col].nunique() < n:
        return []
    q = pd.qcut(s[col], n, labels=False, duplicates="drop")
    out = []
    for k in sorted(set(q.dropna())):
        b = s[q == k]
        out.append((f"{col} Q{int(k)+1} ({b[col].min():.2f}-{b[col].max():.2f})", b))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    a = ap.parse_args()
    run(a)
