# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27.0", "polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
hl_weekend_backtest.py
======================

Phase 1: does the Hyperliquid `xyz:` synthetic-equity perp overshoot on weekends
(when the oracle is FROZEN at Friday's close and the book is pure order flow),
and does fading that weekend drift pay once the oracle un-freezes at Monday's US
open?

STEP 0 finding (probes, 2026-09-02): the xyz oracle DOES freeze on weekends --
the perp's Monday price = wherever weekend order flow left it, no snap to a live
feed. Single-stock perps drift 0.3-5% over a weekend on thin books; macro perps
(SP500/GOLD) barely move. Funding stays near the hourly floor for liquid names.

Per (perp, weekend):
  fri_ref     -- perp close at Fri 20:00 UTC (US cash close, last real anchor)
  perp_mon    -- perp close just before Mon 14:30 UTC (US cash open)
  wknd_drift  -- perp_mon / fri_ref - 1          (frozen-oracle drift)
  real_gap    -- underlying Mon open / Fri close - 1  (what actually happened)
  overshoot   -- wknd_drift - real_gap
  fade P&L    -- SHORT the drift at Sun 22:00 UTC, cover at Mon --exit-hh:mm UTC;
                 pnl = -sign(wknd_drift) * (exit/entry - 1) + funding_received - fees
  (also the WITH-drift variant: does weekend flow predict the real move?)

Underlying map: xyz:SP500->SPY, xyz:GOLD->GLD, xyz:SILVER->SLV, else same ticker
(needs historical/{T}.parquet from ohlc-build).

Usage:
  python hl_weekend_backtest.py --fetch            # pull + cache HL candles/funding
  python hl_weekend_backtest.py                    # run the analysis
  python hl_weekend_backtest.py --exit 17:00 --min-drift 0.5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd
import httpx

API = "https://api.hyperliquid.xyz/info"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "_hl_cache")
HIST = "historical"

STOCKS = ["NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL", "MU", "AVGO"]
MACRO = {"SP500": "SPY", "GOLD": "GLD", "SILVER": "SLV"}
FEE_RT = 0.0010          # round-trip taker fee+slippage assumption (0.10%)


def _post(body, tries=4):
    for i in range(tries):
        try:
            r = httpx.post(API, headers={"Content-Type": "application/json"}, json=body, timeout=45)
            if r.status_code == 200:
                return r.json()
            time.sleep(1.5 * (i + 1))
        except httpx.HTTPError:
            time.sleep(1.5 * (i + 1))
    return None


def _ms(d: dt.date | dt.datetime) -> int:
    if isinstance(d, dt.date) and not isinstance(d, dt.datetime):
        d = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def fetch_all(coins, back_days=240):
    os.makedirs(CACHE, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=back_days)
    for coin in coins:
        full = f"xyz:{coin}"
        # candles: 15m, walk in ~20-day chunks (server caps ~5000 rows)
        cndl = []
        t = start
        while t < now:
            t2 = min(t + dt.timedelta(days=40), now)
            r = _post({"type": "candleSnapshot", "req": {"coin": full, "interval": "15m",
                                                         "startTime": _ms(t), "endTime": _ms(t2)}})
            if isinstance(r, list):
                cndl.extend(r)
            t = t2
        if cndl:
            df = pd.DataFrame(cndl).drop_duplicates("t").sort_values("t")
            df = df.rename(columns={"t": "open_ms", "T": "close_ms", "o": "o", "c": "c",
                                    "h": "h", "l": "l", "v": "v"})
            for c in ("o", "c", "h", "l", "v"):
                df[c] = df[c].astype(float)
            df["dt"] = pd.to_datetime(df["open_ms"], unit="ms", utc=True)
            df.to_parquet(os.path.join(CACHE, f"{coin}_candles.parquet"), index=False)
        # funding: paginate
        fund = []
        t = start
        while t < now:
            r = _post({"type": "fundingHistory", "coin": full, "startTime": _ms(t)})
            if not isinstance(r, list) or not r:
                break
            fund.extend(r)
            last = r[-1]["time"]
            nt = dt.datetime.fromtimestamp(last / 1000, dt.timezone.utc) + dt.timedelta(hours=1)
            if nt <= t:
                break
            t = nt
            if len(r) < 500:
                break
        if fund:
            fd = pd.DataFrame(fund).drop_duplicates("time").sort_values("time")
            fd["fundingRate"] = fd["fundingRate"].astype(float)
            fd["premium"] = fd["premium"].astype(float)
            fd["dt"] = pd.to_datetime(fd["time"], unit="ms", utc=True)
            fd.to_parquet(os.path.join(CACHE, f"{coin}_funding.parquet"), index=False)
        print(f"  {coin}: {len(cndl)} candles, {len(fund)} funding pts")


def _real_daily(tk):
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    import polars as pl
    df = pl.read_parquet(p).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York")
    mo = et.dt.hour * 60 + et.dt.minute
    g = df.assign(d=et.dt.date, mo=mo)
    op = g[(g["mo"] >= 570) & (g["mo"] <= 576)].groupby("d")["open"].first()
    cl = g[(g["mo"] >= 954) & (g["mo"] <= 960)].groupby("d")["close"].last()
    return op, cl


def _px_at(cndf, when_utc):
    m = cndf[cndf["dt"] <= when_utc]
    return float(m.iloc[-1]["c"]) if len(m) else None


def run(args):
    hh, mm = map(int, args.exit.split(":"))
    coins = STOCKS + list(MACRO)
    rows = []
    for coin in coins:
        cp = os.path.join(CACHE, f"{coin}_candles.parquet")
        if not os.path.exists(cp):
            continue
        cnd = pd.read_parquet(cp)
        fp = os.path.join(CACHE, f"{coin}_funding.parquet")
        fund = pd.read_parquet(fp) if os.path.exists(fp) else None
        real_tk = MACRO.get(coin, coin)
        rd = _real_daily(real_tk)

        # every Saturday inside the candle window
        d0 = cnd["dt"].min().date()
        d1 = cnd["dt"].max().date()
        d = d0
        while d <= d1:
            if d.weekday() == 5:  # Saturday
                fri = d - dt.timedelta(days=1)
                mon = d + dt.timedelta(days=2)
                fri_ref_t = dt.datetime(fri.year, fri.month, fri.day, 20, 0, tzinfo=dt.timezone.utc)
                entry_t = dt.datetime(d.year, d.month, d.day, 22, 0, tzinfo=dt.timezone.utc) + dt.timedelta(days=1)  # Sun 22:00
                mon_open_t = dt.datetime(mon.year, mon.month, mon.day, 14, 30, tzinfo=dt.timezone.utc)
                exit_t = dt.datetime(mon.year, mon.month, mon.day, hh, mm, tzinfo=dt.timezone.utc)

                fri_ref = _px_at(cnd, fri_ref_t)
                entry_px = _px_at(cnd, entry_t)
                perp_mon = _px_at(cnd, mon_open_t)
                exit_px = _px_at(cnd, exit_t)
                if None in (fri_ref, entry_px, perp_mon, exit_px):
                    d += dt.timedelta(days=1); continue

                wknd_drift = perp_mon / fri_ref - 1
                entry_drift = entry_px / fri_ref - 1
                real_gap = real_mon_close = np.nan
                if rd is not None:
                    op, cl = rd
                    if fri in cl.index and mon in op.index:
                        real_gap = op[mon] / cl[fri] - 1
                    if fri in cl.index and mon in cl.index:
                        real_mon_close = cl[mon] / cl[fri] - 1

                # funding accrued entry -> exit (short receives +funding when premium>0)
                fnd = 0.0
                if fund is not None:
                    w = fund[(fund["dt"] >= entry_t) & (fund["dt"] < exit_t)]
                    fnd = float(w["fundingRate"].sum())

                side = -np.sign(entry_drift)          # FADE the weekend drift
                if side == 0:
                    d += dt.timedelta(days=1); continue
                raw = side * (exit_px / entry_px - 1)
                funding_pnl = side * fnd * -1          # short pays -funding i.e. receives when funding>0? HL: longs pay shorts when funding>0
                # HL convention: funding>0 => longs pay shorts. short pnl from funding = +fnd (per hr summed). long = -fnd.
                funding_pnl = (fnd if side < 0 else -fnd)
                fade_pnl = raw + funding_pnl - FEE_RT
                with_pnl = -raw - funding_pnl - FEE_RT   # opposite trade (go WITH the drift)

                rows.append(dict(coin=coin, macro=coin in MACRO, sat=d,
                                 wknd_drift=wknd_drift, entry_drift=entry_drift,
                                 real_gap=real_gap, real_mon_close=real_mon_close,
                                 overshoot=wknd_drift - real_gap,
                                 revert_open_to_exit=(exit_px / perp_mon - 1),
                                 funding=fnd, fade_pnl=fade_pnl, with_pnl=with_pnl))
            d += dt.timedelta(days=1)

    R = pd.DataFrame(rows)
    if R.empty:
        print("no data -- run --fetch first")
        return
    R = R[R["entry_drift"].abs() >= args.min_drift / 100.0]

    def _blk(name, sub):
        if sub.empty:
            print(f"  {name:24} (no trades)"); return
        fp = sub["fade_pnl"].to_numpy() * 100
        wp = sub["with_pnl"].to_numpy() * 100
        os_ = sub["overshoot"].dropna().to_numpy() * 100
        rv = sub["revert_open_to_exit"].to_numpy() * 100
        print(f"  {name:24} n={len(sub):>3}  "
              f"FADE exp {fp.mean():+.2f}% (med {np.median(fp):+.2f}, win {(fp>0).mean()*100:.0f}%)   "
              f"WITH exp {wp.mean():+.2f}%   "
              f"overshoot med {np.median(os_):+.2f}pp   Mon-open->exit revert {rv.mean():+.2f}%")

    print("=" * 108)
    print(f"  HL WEEKEND FADE   exit Mon {args.exit} UTC   fee {FEE_RT*100:.2f}% RT   "
          f"|entry drift| >= {args.min_drift}%   ({len(R)} weekend-trades)")
    print("=" * 108)
    _blk("ALL", R)
    _blk("single stocks", R[~R["macro"]])
    _blk("macro (SP500/GOLD/SLV)", R[R["macro"]])
    print()
    for coin in R["coin"].unique():
        _blk(coin, R[R["coin"] == coin])
    print()
    # does |weekend drift| predict how much reverts?
    s = R.dropna(subset=["overshoot"])
    if len(s) > 20:
        c1 = np.corrcoef(s["entry_drift"].abs(), s["revert_open_to_exit"] * -np.sign(s["entry_drift"]))[0, 1]
        c2 = np.corrcoef(s["entry_drift"], s["real_gap"])[0, 1]
        print(f"  corr(|entry drift|, reversion in fade direction) = {c1:+.2f}")
        print(f"  corr(entry drift, real Fri->Mon gap)            = {c2:+.2f}   "
              f"(high = weekend flow PREDICTS the real move)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="pull + cache HL candles/funding, then exit")
    ap.add_argument("--back-days", type=int, default=240)
    ap.add_argument("--exit", default="16:00", help="Monday UTC time to close the fade (default 16:00 = ~11:30 ET)")
    ap.add_argument("--min-drift", type=float, default=0.0, help="only trade weekends where |entry drift| >= this %%")
    a = ap.parse_args()
    if a.fetch:
        fetch_all(STOCKS + list(MACRO), a.back_days)
    else:
        run(a)
