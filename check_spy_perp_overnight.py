# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27.0", "polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_spy_perp_overnight.py
===========================
check_spy_overnight found the SPY night effect is real (+5bp/night, +14.5bp when
VIX<median) but a long CALL can't capture it (theta + spread > the drift).  The
user's fix: trade it on the Hyperliquid xyz:SP500 PERP -- delta 1, no theta,
only funding as carry.

Tests:
  1) does the perp TRACK SPY overnight?  perp 09:35ET / perp 16:00ET-prev  vs
     SPY open/close_prev.  (xyz oracle freezes off-hours -- see
     hl_weekend_backtest.py -- so the perp floats on HL order flow overnight then
     snaps back when the US cash session re-opens.)
  2) overnight perp MAX-ADVERSE-EXCURSION -> what leverage survives the gap tail
  3) overnight FUNDING paid by a long (hourly rate x hours held)
  4) SIM: long perp 16:00ET, flat 09:35ET, ONLY when VIX < 60d median.
     grid leverage {3,5,10,20,50} with a liquidation check, funding drag, fees.
     equity curve, maxDD, #liquidations, per-night mean, ann Sharpe.

  --fetch          pull xyz:SP500 1h + 15m candles + funding -> _hl_cache/
  (default)        the analysis

Usage:
  python check_spy_perp_overnight.py --fetch
  python check_spy_perp_overnight.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time

import numpy as np
import pandas as pd

try:
    import httpx
except ImportError:
    httpx = None

API = "https://api.hyperliquid.xyz/info"
CACHE = "_hl_cache"
HIST = "historical"
COIN = "SP500"
MAINT_MARGIN = 0.0125          # HL cross maintenance-margin ~ half the max leverage tier; conservative
TAKER = 0.00045               # HL taker fee each side
ENTRY_UTC = 20 * 60           # 20:00 UTC ~ 16:00 ET cash close
EXIT_UTC = 13 * 60 + 35       # 13:35 UTC ~ 09:35 ET cash open


def _post(body, tries=4):
    for i in range(tries):
        try:
            r = httpx.post(API, headers={"Content-Type": "application/json"}, json=body, timeout=45)
            if r.status_code == 200:
                return r.json()
            time.sleep(1.5 * (i + 1))
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def _ms(d):
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    full = f"xyz:{COIN}"
    for interval, days, span in (("1h", 400, 120), ("15m", 90, 40)):
        rows, t = [], now - dt.timedelta(days=days)
        while t < now:
            t2 = min(t + dt.timedelta(days=span), now)
            r = _post({"type": "candleSnapshot",
                       "req": {"coin": full, "interval": interval, "startTime": _ms(t), "endTime": _ms(t2)}})
            if isinstance(r, list):
                rows.extend(r)
            t = t2
        if rows:
            df = pd.DataFrame(rows).drop_duplicates("t").sort_values("t")
            for c in ("o", "c", "h", "l", "v"):
                df[c] = df[c].astype(float)
            df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df.to_parquet(f"{CACHE}/{COIN}_{interval}.parquet", index=False)
            print(f"  {interval}: {len(df)} candles  {df['dt'].min()} .. {df['dt'].max()}")
    fund, t = [], now - dt.timedelta(days=400)
    while t < now:
        r = _post({"type": "fundingHistory", "coin": full, "startTime": _ms(t)})
        if not isinstance(r, list) or not r:
            break
        fund.extend(r)
        nt = dt.datetime.fromtimestamp(r[-1]["time"] / 1000, dt.timezone.utc) + dt.timedelta(hours=1)
        if nt <= t or len(r) < 400:
            t = nt; break
        t = nt
    if fund:
        fd = pd.DataFrame(fund).drop_duplicates("time").sort_values("time")
        fd["fundingRate"] = fd["fundingRate"].astype(float)
        fd["dt"] = pd.to_datetime(fd["time"], unit="ms", utc=True)
        fd.to_parquet(f"{CACHE}/{COIN}_funding.parquet", index=False)
        print(f"  funding: {len(fd)} pts  {fd['dt'].min()} .. {fd['dt'].max()}")


def _spy_daily():
    import polars as pl
    d = pl.read_parquet(f"{HIST}/SPY.parquet", columns=["start_time", "open", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 959)].sort_values(["date", "mod"])
    g = d.groupby("date")
    x = pd.DataFrame({"o": g.first()["open"], "c": g.last()["close"]}).reset_index()
    x["date"] = pd.to_datetime(x["date"])
    return x.sort_values("date").reset_index(drop=True)


def _vixfav():
    if httpx is None:
        return {}
    from dotenv import load_dotenv
    load_dotenv()
    h = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json"}
    try:
        r = httpx.get("https://api.unusualwhales.com/api/stock/VIX/volatility/realized",
                      headers=h, params={"timeframe": "2Y"}, timeout=20).json().get("data", [])
        s = pd.Series({pd.Timestamp(x["date"]): float(x["price"]) for x in r if x.get("price")}).sort_index()
        med = s.shift(1).rolling(60, min_periods=20).median()
        return {d.date(): (bool(s[d] >= med[d]) if pd.notna(med[d]) else None) for d in s.index}
    except Exception as e:
        print(f"  (VIX unavailable: {e})")
        return {}


def _asof(df, ts):
    sub = df[df["dt"] <= ts]
    return sub.iloc[-1] if len(sub) else None


def run(a):
    cf = f"{CACHE}/{COIN}_1h.parquet"
    if not os.path.exists(cf):
        print("run --fetch first"); return
    c1 = pd.read_parquet(cf)
    c15 = pd.read_parquet(f"{CACHE}/{COIN}_15m.parquet") if os.path.exists(f"{CACHE}/{COIN}_15m.parquet") else c1
    fnd = pd.read_parquet(f"{CACHE}/{COIN}_funding.parquet")
    spy = _spy_daily()
    vf = _vixfav()

    # per session-pair: entry (D 20:00 UTC), exit (D+1 13:35 UTC)
    days = pd.to_datetime(sorted(spy["date"].dt.date.unique()))
    R = []
    # use whichever candle set spans MORE calendar days (1h has the long history)
    span1 = (c1["dt"].max() - c1["dt"].min()).days
    span15 = (c15["dt"].max() - c15["dt"].min()).days
    cand = c1 if span1 >= span15 else c15
    print(f"  using {'1h' if cand is c1 else '15m'} candles ({(cand['dt'].max()-cand['dt'].min()).days}d span)")
    for i in range(len(days) - 1):
        d0, d1 = days[i].date(), days[i + 1].date()
        te = pd.Timestamp(f"{d0} 20:00", tz="UTC")
        tx = pd.Timestamp(f"{d1} 13:35", tz="UTC")
        pe, px = _asof(cand, te), _asof(cand, tx)
        if pe is None or px is None or (te - pe["dt"]).total_seconds() > 5400 or (tx - px["dt"]).total_seconds() > 5400:
            continue
        p_entry, p_exit = float(pe["c"]), float(px["c"])
        night = cand[(cand["dt"] > pe["dt"]) & (cand["dt"] <= px["dt"])]
        mae = (night["l"].min() / p_entry - 1) if len(night) else 0.0
        srow = spy[spy["date"].dt.date == d1]
        sprev = spy[spy["date"].dt.date == d0]
        if srow.empty or sprev.empty:
            continue
        spy_gap = float(srow["o"].iloc[0]) / float(sprev["c"].iloc[0]) - 1
        fw = fnd[(fnd["dt"] > pe["dt"]) & (fnd["dt"] <= px["dt"])]
        fund_paid = float(fw["fundingRate"].sum())      # long pays sum of hourly rates
        R.append(dict(d0=d0, d1=d1, perp_ret=p_exit / p_entry - 1, spy_gap=spy_gap,
                      mae=mae, fund=fund_paid, vixfav=vf.get(d1)))
    df = pd.DataFrame(R)
    if df.empty:
        print("no overlapping sessions -- HL history too short"); return

    print("=" * 100)
    print(f"  xyz:SP500 PERP overnight  ({len(df)} sessions  {df.d0.min()} .. {df.d1.max()})")
    print("=" * 100)
    print(f"  perp overnight ret : mean {df.perp_ret.mean()*1e4:+.1f}bp  sd {df.perp_ret.std()*1e4:.0f}bp")
    print(f"  actual SPY gap     : mean {df.spy_gap.mean()*1e4:+.1f}bp  sd {df.spy_gap.std()*1e4:.0f}bp")
    print(f"  corr(perp_ret, spy_gap) = {df.perp_ret.corr(df.spy_gap):+.3f}   "
          f"tracking err (perp-gap) sd {(df.perp_ret-df.spy_gap).std()*1e4:.0f}bp")
    print(f"  funding paid by long/night: mean {df.fund.mean()*1e4:+.2f}bp  "
          f"(sum over the hold; +ve = long pays)")
    print(f"  overnight MAE (worst tick vs entry): mean {df.mae.mean()*1e4:.0f}bp  "
          f"p10 {df.mae.quantile(.1)*1e4:.0f}bp  min {df.mae.min()*1e4:.0f}bp")
    print(f"  -> max safe leverage to survive the worst MAE: {1/abs(df.mae.min()+ -MAINT_MARGIN):.1f}x  "
          f"(p10: {1/abs(df.mae.quantile(.1)+ -MAINT_MARGIN):.1f}x)")

    for lbl, sub in (("VIX < median (favourable)", df[df.vixfav == False]),
                     ("VIX >= median", df[df.vixfav == True]), ("ALL", df)):
        if len(sub) < 10:
            continue
        net = sub["perp_ret"] - sub["fund"]
        print(f"\n  [{lbl}]  n={len(sub)}   perp_ret {sub.perp_ret.mean()*1e4:+.1f}bp   "
              f"net of funding {net.mean()*1e4:+.1f}bp   %win {(net>0).mean():.0%}")
        for L in (3, 5, 10, 20, 50):
            liq = sub["mae"] * L <= -(1 - MAINT_MARGIN)
            pn = np.where(liq, -1.0, (net.values - 2 * TAKER) * L)
            eq = np.cumprod(1 + np.clip(pn, -1, None))
            dd = 1 - eq / np.maximum.accumulate(eq)
            shp = pn.mean() / pn.std() * np.sqrt(252) if pn.std() > 0 else float("nan")
            print(f"    {L:2}x  night mean {pn.mean()*100:+6.2f}%  liq {int(liq.sum())}  "
                  f"maxDD {dd.max()*100:4.0f}%  end ${eq[-1]:.2f}  Sharpe {shp:+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        run(a)
