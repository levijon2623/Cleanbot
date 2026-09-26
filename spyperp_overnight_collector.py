# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27.0", "pandas>=2.0.0", "numpy>=1.26.0", "python-dotenv>=1.0.0", "requests>=2.31.0"]
# ///
"""
spyperp_overnight_collector.py
==============================
PASSIVE data collector (tier A) for the "long xyz:SP500 perp overnight, VIX-gated"
idea (see check_spy_perp_overnight.py -- in-sample +19bp/night / Sharpe ~3.5 when
VIX < its 60d median, but only 116 calm 2026 sessions -> need forward data,
especially a real vol event).

NO positions, NO capital.  Run it once a day (or weekly -- it catches up every
completed overnight window since the last logged one).  Per (cash close D ->
cash open D+1) it records, from the Hyperliquid candle/funding API + UW:

  perp_entry / perp_exit / perp_ret     -- xyz:SP500 near 16:00 ET and 09:35 ET
  funding_paid                          -- sum of hourly funding over the hold
  mae                                   -- min(low)/entry-1 over the overnight bars
  spy_close / spy_open / spy_gap        -- what actually happened (tracking check)
  track_err                             -- perp_ret - spy_gap
  vix_prev / vix_med60 / vix_favorable  -- the regime that was in effect
  net_ret                               -- perp_ret - funding_paid
  sim_{5,10,20}x + liq flags            -- leverage-agnostic P&L + liquidation check

Appends to _spyperp_overnight_log.parquet (idempotent -- re-running never dupes).

Usage:
  python spyperp_overnight_collector.py            # collect any new completed nights
  python spyperp_overnight_collector.py --report   # running stats on the log so far
  python spyperp_overnight_collector.py --backfill 150   # seed from HL candle history
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import httpx
except ImportError:
    httpx = None
import requests
from dotenv import load_dotenv

ET = ZoneInfo("America/New_York")
HL_API = "https://api.hyperliquid.xyz/info"
COIN = "xyz:SP500"
LOG = "_spyperp_overnight_log.parquet"
ENTRY_ET = (15, 58)     # ~cash close
EXIT_ET = (9, 35)       # ~few min after cash open
MAINT_MM = 0.0125
TAKER = 0.00045       # HL taker each side (market order)
MAKER = 0.00015      # HL maker each side (resting limit near the touch)
LEVS = (5, 10, 20)


# --------------------------------------------------------------------------- #
def _hl(body, tries=4):
    for i in range(tries):
        try:
            r = httpx.post(HL_API, json=body, headers={"Content-Type": "application/json"}, timeout=45)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def _ms(d: dt.datetime) -> int:
    return int(d.timestamp() * 1000)


def _hl_candles(start: dt.datetime, end: dt.datetime, interval="15m") -> pd.DataFrame:
    rows, t = [], start
    while t < end:
        t2 = min(t + dt.timedelta(days=30), end)
        r = _hl({"type": "candleSnapshot", "req": {"coin": COIN, "interval": interval,
                                                   "startTime": _ms(t), "endTime": _ms(t2)}})
        if isinstance(r, list):
            rows.extend(r)
        t = t2
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("t").sort_values("t")
    for c in ("o", "c", "h", "l"):
        df[c] = df[c].astype(float)
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df[["dt", "o", "c", "h", "l"]]


def _hl_funding(start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    rows, t = [], start
    while t < end:
        r = _hl({"type": "fundingHistory", "coin": COIN, "startTime": _ms(t)})
        if not isinstance(r, list) or not r:
            break
        rows.extend(r)
        nt = dt.datetime.fromtimestamp(rows[-1]["time"] / 1000, dt.timezone.utc) + dt.timedelta(hours=1)
        if nt <= t or len(r) < 400:
            break
        t = nt
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["dt"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df[["dt", "fundingRate"]]


def _asof(df: pd.DataFrame, ts: pd.Timestamp, col: str, max_gap_min=90):
    sub = df[df["dt"] <= ts]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    if (ts - row["dt"]).total_seconds() > max_gap_min * 60:
        return None
    return float(row[col])


# --------------------------------------------------------------------------- #
def _uw():
    load_dotenv()
    return {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json"}


def _spy_sessions(lookback_days: int) -> pd.DataFrame:
    """recent SPY RTH daily open/close from UW 1-min bars (no lake dependency)."""
    h = _uw()
    out = []
    day = datetime_now_et().date()
    got = 0
    while got < lookback_days + 4 and len(out) < lookback_days + 4:
        if day.weekday() < 5:
            r = requests.get(f"https://api.unusualwhales.com/api/stock/SPY/ohlc/1m",
                             headers=h, params={"date": day.isoformat()}, timeout=20)
            rows = r.json().get("data", []) if r.status_code == 200 else []
            bars = []
            for x in rows:
                if x.get("market_time") not in (None, "r"):
                    continue
                st = x.get("start_time") or ""
                try:
                    d_et = dt.datetime.fromisoformat(st.replace("Z", "+00:00")).astimezone(ET)
                except ValueError:
                    continue
                mod = d_et.hour * 60 + d_et.minute
                if 570 <= mod <= 959:
                    bars.append((mod, float(x["open"]), float(x["close"])))
            if bars:
                bars.sort()
                out.append({"date": day, "open": bars[0][1], "close": bars[-1][2]})
                got += 1
        day -= dt.timedelta(days=1)
    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


def datetime_now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def _vix_series() -> pd.DataFrame:
    r = requests.get("https://api.unusualwhales.com/api/stock/VIX/volatility/realized",
                     headers=_uw(), params={"timeframe": "2Y"}, timeout=20)
    rows = r.json().get("data", []) if r.status_code == 200 else []
    s = pd.Series({pd.Timestamp(x["date"]).date(): float(x["price"]) for x in rows if x.get("price")}).sort_index()
    df = pd.DataFrame({"vix": s})
    df["vix_prev"] = df["vix"].shift(1)
    df["med60"] = df["vix"].shift(1).rolling(60, min_periods=20).median()
    return df


# --------------------------------------------------------------------------- #
def _row_for(d_close, d_open, cndl, fund, spy, vix):
    te = pd.Timestamp(dt.datetime.combine(d_close, dt.time(*ENTRY_ET), ET)).tz_convert("UTC")
    tx = pd.Timestamp(dt.datetime.combine(d_open, dt.time(*EXIT_ET), ET)).tz_convert("UTC")
    pe = _asof(cndl, te, "c")
    px = _asof(cndl, tx, "c")
    if pe is None or px is None or pe <= 0:
        return None
    night = cndl[(cndl["dt"] > te) & (cndl["dt"] <= tx)]
    mae = float(night["l"].min() / pe - 1) if len(night) else 0.0
    fpaid = float(fund[(fund["dt"] > te) & (fund["dt"] <= tx)]["fundingRate"].sum()) if len(fund) else 0.0
    sc = spy.loc[spy["date"] == d_close, "close"]
    so = spy.loc[spy["date"] == d_open, "open"]
    spy_close = float(sc.iloc[0]) if len(sc) else np.nan
    spy_open = float(so.iloc[0]) if len(so) else np.nan
    gap = (spy_open / spy_close - 1) if np.isfinite(spy_close) and np.isfinite(spy_open) else np.nan
    vp = vm = np.nan
    fav = None
    if d_open in vix.index:
        vr = vix.loc[d_open]
        vp = float(vr["vix_prev"]) if pd.notna(vr["vix_prev"]) else np.nan
        vm = float(vr["med60"]) if pd.notna(vr["med60"]) else np.nan
        fav = bool(vp < vm) if np.isfinite(vp) and np.isfinite(vm) else None
    perp_ret = px / pe - 1
    net = perp_ret - fpaid
    row = dict(date_close=str(d_close), date_open=str(d_open),
               perp_entry=round(pe, 2), perp_exit=round(px, 2), perp_ret=perp_ret,
               funding_paid=fpaid, mae=mae, spy_close=spy_close, spy_open=spy_open,
               spy_gap=gap, track_err=(perp_ret - gap) if np.isfinite(gap) else np.nan,
               vix_prev=vp, vix_med60=vm, vix_favorable=fav, net_ret=net,
               collected_at=datetime_now_et().isoformat(timespec="seconds"))
    for L in LEVS:
        liq = bool(mae * L <= -(1 - MAINT_MM))
        row[f"liq_{L}x"] = liq
        row[f"sim_{L}x"] = -1.0 if liq else (net - 2 * TAKER) * L          # market in/out
        row[f"sim_{L}x_mkr"] = -1.0 if liq else (net - 2 * MAKER) * L      # limit in/out
    return row


def collect(a):
    if httpx is None:
        print("pip install httpx"); return
    existing = pd.read_parquet(LOG) if os.path.exists(LOG) else pd.DataFrame()
    have = set(existing["date_close"]) if len(existing) else set()

    lookback = a.backfill if a.backfill else 12
    spy = _spy_sessions(lookback)
    spy["date"] = pd.to_datetime(spy["date"]).dt.date
    if len(spy) < 2:
        print("could not get SPY sessions"); return
    start = pd.Timestamp(dt.datetime.combine(spy["date"].iloc[0], dt.time(0), ET)).tz_convert("UTC")
    end = pd.Timestamp(datetime_now_et()).tz_convert("UTC")
    c15 = _hl_candles(start.to_pydatetime(), end.to_pydatetime(), "15m")
    # 15m history is short (~7wk); 1h reaches ~6mo. Use both -- 15m where available
    # (finer MAE), 1h for the older backfill tail.
    if a.backfill or c15.empty or (start.tz_convert("UTC") < c15["dt"].min() - pd.Timedelta(days=2)):
        c1h = _hl_candles(start.to_pydatetime(), end.to_pydatetime(), "1h")
        cndl = pd.concat([c1h[c1h["dt"] < (c15["dt"].min() if len(c15) else end)], c15]).sort_values("dt")
        cndl = cndl.drop_duplicates("dt")
    else:
        cndl = c15
    fund = _hl_funding(start.to_pydatetime(), end.to_pydatetime())
    vix = _vix_series()

    new = []
    sd = list(spy["date"])
    for i in range(len(sd) - 1):
        d_close, d_open = sd[i], sd[i + 1]
        # only log a COMPLETED night (exit time has passed)
        tx = dt.datetime.combine(d_open, dt.time(*EXIT_ET), ET)
        if datetime_now_et() < tx + dt.timedelta(minutes=20):
            continue
        if str(d_close) in have:
            continue
        r = _row_for(d_close, d_open, cndl, fund, spy, vix)
        if r:
            new.append(r)

    if not new:
        print("no new completed overnight windows.")
        return
    out = pd.concat([existing, pd.DataFrame(new)], ignore_index=True)
    out = out.drop_duplicates("date_close", keep="last").sort_values("date_close").reset_index(drop=True)
    out.to_parquet(LOG, index=False)
    for r in new:
        f = "fav" if r["vix_favorable"] else ("LOW-vix-NO" if r["vix_favorable"] is False else "vix?")
        print(f"  {r['date_close']}->{r['date_open']}  perp {r['perp_ret']*1e4:+6.1f}bp  "
              f"gap {r['spy_gap']*1e4:+6.1f}bp  te {r['track_err']*1e4:+5.1f}bp  "
              f"fund {r['funding_paid']*1e4:+.2f}bp  mae {r['mae']*1e4:+5.0f}bp  [{f}]")
    print(f"  +{len(new)} rows -> {LOG} (total {len(out)})")


def report(_a):
    if not os.path.exists(LOG):
        print("no log yet -- run the collector first"); return
    df = pd.read_parquet(LOG)
    print(f"  {len(df)} nights  {df['date_close'].min()} .. {df['date_open'].max()}")
    print(f"  perp tracks SPY: corr {df['perp_ret'].corr(df['spy_gap']):+.3f}   "
          f"track-err sd {df['track_err'].std()*1e4:.0f}bp   funding {df['funding_paid'].mean()*1e4:+.2f}bp/night")
    print(f"  MAE: mean {df['mae'].mean()*1e4:.0f}bp  p10 {df['mae'].quantile(.1)*1e4:.0f}bp  "
          f"worst {df['mae'].min()*1e4:.0f}bp  ({df['mae'].min()*100:.1f}%)")
    for lbl, sub in (("VIX-favourable (<med)", df[df.vix_favorable == True]),
                     ("VIX >= med", df[df.vix_favorable == False]),
                     ("ALL", df)):
        if len(sub) < 5:
            continue
        n = sub["net_ret"]
        line = f"\n  [{lbl}] n={len(sub)}  net-pre-fee {n.mean()*1e4:+.1f}bp/night  win {(n>0).mean():.0%}  "
        line += f"Sharpe {n.mean()/n.std()*np.sqrt(252):+.2f}" if n.std() > 0 else ""
        print(line)
        for L in LEVS:
            def eq_end(col):
                v = sub[col].values
                e = np.cumprod(1 + np.clip(v, -1, None))
                return v.mean() * 100, (1 - e / np.maximum.accumulate(e)).max() * 100, e[-1]
            mt = eq_end(f"sim_{L}x"); mk = eq_end(f"sim_{L}x_mkr")
            print(f"    {L:2}x  taker[{mt[0]:+5.2f}%/n  DD{mt[1]:3.0f}%  ${mt[2]:.2f}]   "
                  f"maker[{mk[0]:+5.2f}%/n  DD{mk[1]:3.0f}%  ${mk[2]:.2f}]   liq {int(sub[f'liq_{L}x'].sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--backfill", type=int, default=0, metavar="N", help="seed from ~N days of HL history")
    a = ap.parse_args()
    if a.report:
        report(a)
    else:
        collect(a)
