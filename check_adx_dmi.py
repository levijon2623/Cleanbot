# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_adx_dmi.py
================
Wilder's Directional Movement System -- ADX (trend STRENGTH, direction-agnostic)
and +DI / -DI (directional movement) -- as a gate on the deployed rules.

Backward-looking by construction (ADX lags ~2x its period), and every
backward-looking indicator tested here so far (EMA stack, SMA trend, GEX regime,
COR1M, charm/vanna) has been either a wash or a one-rule curio. This checks it
anyway, the same way:

  DAILY (period 14, Wilder RMA):  +DI14, -DI14, ADX14, DI spread, ADX 5d slope
    -- computed on PRIOR completed sessions (lookahead-free), from the ticker's
       1-min OHLC aggregated to daily.
  INTRADAY (period 14 on 15-min bars): the DI/ADX state as of the last 15-min
       bar that CLOSED at/before the trigger.

For each enabled config.RULES entry (deployed spec), split option-bracket P&L by:
  * daily ADX bucket        (<20 chop / 20-30 / >30 strong-trend)
  * daily DI direction      (+DI>-DI bull / -DI>+DI bear)
  * daily DI AGREES w/ the rule's direction  (CALL wants +DI>-DI, PUT the reverse)
  * ADX rising vs falling
  * intraday ADX <20 vs >=20  and  intraday DI-agrees
report IS/OOS @ 2025-08-21 + n; flag any clean both-halves lift.

Usage:
  python check_adx_dmi.py
  python check_adx_dmi.py --tickers IWM META --intraday
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

from check_flow_zscore import annotate_flow_z, _z_matched, _eod_mod_for
from check_config_walkforward import _flow_for, _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
P = 14


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _adx_frame(df: pd.DataFrame, n: int = P) -> pd.DataFrame:
    """df: columns o,h,l,c indexed 0..N (chronological). returns +di,-di,adx,
    di_spread, adx_slope5 (all shifted so row i uses only data through i-1 is the
    CALLER's job -- here they're aligned to the bar that produced them)."""
    h, l, c = df["h"], df["l"], df["c"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, n)
    pdi = 100 * _rma(pd.Series(plus_dm, index=df.index), n) / atr
    mdi = 100 * _rma(pd.Series(minus_dm, index=df.index), n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = _rma(dx, n)
    out = pd.DataFrame({"pdi": pdi, "mdi": mdi, "adx": adx})
    out["di_spread"] = out["pdi"] - out["mdi"]
    out["adx_slope5"] = out["adx"].diff(5)
    return out


def _daily_adx(tk: str) -> dict:
    """{date: (pdi, mdi, adx, di_spread, adx_slope5)} using PRIOR-day values
    (lookahead-free -- a 9:30 entry only knows yesterday's close)."""
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return {}
    d = pl.read_parquet(p, columns=["start_time", "open", "high", "low", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 960)]
    g = d.groupby("date")
    daily = pd.DataFrame({"o": g.first()["open"], "h": g.max()["high"],
                          "l": g.min()["low"], "c": g.last()["close"]}).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)
    a = _adx_frame(daily)
    out = {}
    for i in range(1, len(daily)):        # row i's trade uses row i-1's completed ADX
        r = a.iloc[i - 1]
        if pd.notna(r["adx"]):
            out[daily.iloc[i]["date"]] = (r["pdi"], r["mdi"], r["adx"], r["di_spread"], r["adx_slope5"])
    return out


def _intraday_adx(tk: str, bar_min: int = 15, close_only: bool = False) -> dict:
    """{date: sorted list of (mod, pdi, mdi, adx)} on bar_min-minute RTH bars,
    each value being that bar's completed ADX (asof-queryable at the trigger).
    close_only=True builds each bar's H/L from the 1-min CLOSES in the bucket
    (not the true 1-min H/L) -- this is what bot_runner can reconstruct live from
    per-minute spot, so validate with it before deploying."""
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return {}
    d = pl.read_parquet(p, columns=["start_time", "open", "high", "low", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["ts"] = et
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 960)].sort_values("ts").reset_index(drop=True)
    if close_only:
        d["high"] = d["close"]
        d["low"] = d["close"]
    d["bucket"] = (d["mod"] // bar_min)
    d["day"] = d["ts"].dt.date
    g = d.groupby(["day", "bucket"])
    bars = pd.DataFrame({
        "o": g.first()["open"], "h": g.max()["high"], "l": g.min()["low"],
        "c": g.last()["close"], "mod": g.last()["mod"],
    }).reset_index().sort_values(["day", "bucket"]).reset_index(drop=True)
    a = _adx_frame(bars.rename(columns={}))
    bars = pd.concat([bars, a[["pdi", "mdi", "adx"]]], axis=1)
    out = {}
    for day, gg in bars.groupby("day"):
        rows = [(int(r.mod), r.pdi, r.mdi, r.adx) for r in gg.itertuples() if pd.notna(r.adx)]
        if rows:
            out[day] = rows
    return out


def _intra_asof(rows, ts):
    m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
    hit = None
    for mod, pdi, mdi, adx in rows:
        if mod <= m:
            hit = (pdi, mdi, adx)
        else:
            break
    return hit


def _stat(pnls):
    if len(pnls) < 12:
        return f"n={len(pnls):>4}  (thin)"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    slices = " ".join(f"S{j+1}{np.mean(b)*100:+.0f}" if len(b) >= 5 else f"S{j+1}··" for j, b in enumerate(sl))
    return (f"n={len(v):>4}  IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%(n{len(i):>3})  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%(n{len(o):>3})  win {np.mean(v>0):.2f}  [{slices}]")


def run(a):
    import directional_flow_backtester as D
    from amt_profile import amt_open_map, amt_ok
    from config import RULES

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]
    tickers = sorted({r["ticker"] for r in rules})
    flow = _flow_for(D, tickers)

    print("=" * 116)
    print("  ADX / DMI GATE on the deployed rules   (daily period-14 Wilder, prior-session; "
          + ("+ intraday 15m" if a.intraday else "daily only") + ")")
    print(f"  split {SPLIT};  slice tags Sn = mean% in 4-month window n")
    print("=" * 116)

    for tk in tickers:
        tk_rules = [r for r in rules if r["ticker"] == tk]
        dadx = _daily_adx(tk)
        iadx = _intraday_adx(tk) if a.intraday else {}
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        amt = amt_open_map(tk) if any(r.get("amt_open") for r in tk_rules) else {}
        ema_stacks = {int(r["ema_confirm"]): D.load_ema_stack(HIST, tk, int(r["ema_confirm"]))
                      for r in tk_rules if r.get("ema_confirm")}

        trigs = D.triggers_for(flow, tk)
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        if not trigs or tb is None or tb.empty:
            print(f"\n  {tk}: no data"); continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        pcache = {}

        def paths(t, direction, dtes):
            key = (id(t), tuple(dtes))
            if key not in pcache:
                pcache[key] = D._option_paths(t, direction, list(dtes), bbd, bbc)
            return pcache[key]

        for r in tk_rules:
            direction = r["direction"].upper()
            dtes = tuple(r.get("dte", [0, 1]))
            tr, rr = float(r["target_roe"]), float(r["rr"])
            tstop = r.get("time_stop_mins"); eod = _eod_mod_for(r)
            want_bull = direction == "CALL"
            zs = r.get("flow_zscore")
            if zs:
                annotate_flow_z(trigs, int(zs.get("window_days") or 60))
                matched = [(t, None) for t in _z_matched(D, r, trigs, gex, vol, trd, amp, reg_src, float(zs["k"]))]
            else:
                D.annotate_flow_pct(trigs, int(r.get("flow_window_days") or 60))
                matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
            ema_stack = ema_stacks.get(int(r["ema_confirm"])) if r.get("ema_confirm") else None
            want_amt = r.get("amt_open")

            rows = []
            for t, _thr in matched:
                d, ts = t["date"], t["ts"]
                if want_amt and not amt_ok(want_amt, amt.get(d)):
                    continue
                if ema_stack is not None:
                    st = D.ema_state_at(ema_stack, ts)
                    if st is not None and st != ("BULL" if want_bull else "BEAR"):
                        continue
                da = dadx.get(d)
                ia = _intra_asof(iadx.get(d, []), ts) if a.intraday else None
                for p in paths(t, direction, dtes):
                    rows.append((d, D._bracket_pnl(*p, tr, rr, tstop, eod), da, ia))
            base = [(d, p) for d, p, _da, _ia in rows]
            if len(base) < 25:
                print(f"\n  {r['name']} ({tk} {direction}): {len(base)} trades -- skip"); continue

            print(f"\n  {r['name']}  ({tk} {direction})")
            print(f"    {'BASELINE':26} {_stat(base)}")

            def sub(pred):
                return [(d, p) for d, p, da, ia in rows if da is not None and pred(da, ia)]

            for lbl, pred in [
                ("daily ADX < 20 (chop)", lambda da, ia: da[2] < 20),
                ("daily ADX 20-30", lambda da, ia: 20 <= da[2] < 30),
                ("daily ADX >= 30 (trend)", lambda da, ia: da[2] >= 30),
                ("daily +DI > -DI (bull)", lambda da, ia: da[3] > 0),
                ("daily -DI > +DI (bear)", lambda da, ia: da[3] < 0),
                ("daily DI agrees w/ dir", lambda da, ia: (da[3] > 0) == want_bull),
                ("daily DI opposes dir", lambda da, ia: (da[3] > 0) != want_bull),
                ("daily ADX rising (5d+)", lambda da, ia: pd.notna(da[4]) and da[4] > 0),
                ("daily ADX falling", lambda da, ia: pd.notna(da[4]) and da[4] < 0),
                ("ADX>=25 & DI agrees", lambda da, ia: da[2] >= 25 and ((da[3] > 0) == want_bull)),
                ("ADX<20 & DI agrees", lambda da, ia: da[2] < 20 and ((da[3] > 0) == want_bull)),
            ]:
                s = sub(pred)
                if len(s) >= 12:
                    print(f"    {lbl:26} {_stat(s)}")

            if a.intraday:
                for lbl, pred in [
                    ("intra ADX < 20", lambda da, ia: ia is not None and ia[2] < 20),
                    ("intra ADX >= 20", lambda da, ia: ia is not None and ia[2] >= 20),
                    ("intra DI agrees", lambda da, ia: ia is not None and ((ia[0] > ia[1]) == want_bull)),
                    ("intra DI opposes", lambda da, ia: ia is not None and ((ia[0] > ia[1]) != want_bull)),
                ]:
                    s = [(d, p) for d, p, da, ia in rows if pred(da, ia)]
                    if len(s) >= 12:
                        print(f"    {lbl:26} {_stat(s)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--intraday", action="store_true")
    a = ap.parse_args()
    run(a)
