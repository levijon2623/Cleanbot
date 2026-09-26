# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_spy_overnight.py
======================
The SPY "night effect": historically ~all of SPY's price appreciation has
accrued CLOSE->OPEN (overnight), with the OPEN->CLOSE (intraday) session roughly
flat.  Confirm it in our window, characterise / condition it, then test whether
holding a 5-8DTE CALL overnight can capture it net of theta + the spread.

  --build          one lake pass -> _spyon_cache/calls.parquet
                   (SPY calls, bar-date DTE 0-10, |strike/spot-1| <= 0.05,
                   close/low/bid/ask per minute)
  (default)        1) daily overnight vs intraday vs close-to-close, cumulative
                      curves, gap distribution, skew
                   2) overnight drift conditioned on: VIX regime, weekday,
                      prior-day intraday return (continuation vs reversal), month
                   3) THE TRADE -- buy a call at ~15:55, exit next 09:35 (or
                      hold N nights / to next close), grid over entry DTE x
                      moneyness x exit, REALISTIC ask-in / bid-out fill, vs a
                      bull call spread.  Split by VIX regime.

Usage:
  python check_spy_overnight.py --build
  python check_spy_overnight.py
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
SILVER = "lake/silver/option-contracts-1m"
CACHE = "_spyon_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
COMM = 0.015
CLOSE_MOD = 955      # 15:55 entry
OPEN_MOD = 572       # 09:32 exit


def build():
    os.makedirs(CACHE, exist_ok=True)
    parts = sorted(glob.glob(f"{SILVER}/date=*/bars.parquet"))
    fr = []
    for i, p in enumerate(parts, 1):
        lf = (pl.scan_parquet(p)
              .filter((pl.col("underlying_symbol") == "SPY") & (pl.col("option_type") == "call"))
              .with_columns(((pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days()).alias("dte"))
              .filter((pl.col("dte") >= 0) & (pl.col("dte") <= 10))
              .filter((pl.col("strike") - pl.col("underlying_close")).abs() / pl.col("underlying_close") <= 0.05)
              .select(["option_chain_id", "strike", "expiry", "minute_et", "dte",
                       "close", "low", "bid_close", "ask_close", "underlying_close"]))
        fr.append(lf.collect())
        if i % 100 == 0:
            print(f"  {i}/{len(parts)}")
    b = pl.concat(fr).to_pandas()
    b["minute_et"] = pd.to_datetime(b["minute_et"]).dt.tz_localize(None)
    b["date"] = b["minute_et"].dt.date
    b["mod"] = b["minute_et"].dt.hour * 60 + b["minute_et"].dt.minute
    b["expiry"] = pd.to_datetime(b["expiry"]).dt.date
    b.to_parquet(f"{CACHE}/calls.parquet", index=False)
    print(f"  {len(b):,} rows  {b['date'].min()}..{b['date'].max()}")


def _spy_daily():
    d = pl.read_parquet(f"{HIST}/SPY.parquet", columns=["start_time", "open", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 959)].sort_values(["date", "mod"])
    g = d.groupby("date")
    x = pd.DataFrame({"open": g.first()["open"], "close": g.last()["close"]}).reset_index()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values("date").reset_index(drop=True)
    x["pc"] = x["close"].shift(1)
    x["on"] = x["open"] / x["pc"] - 1
    x["intr"] = x["close"] / x["open"] - 1
    x["cc"] = x["close"] / x["pc"] - 1
    x["dow"] = x["date"].dt.weekday
    x["month"] = x["date"].dt.month
    x["prev_intr"] = x["intr"].shift(1)
    return x.dropna().reset_index(drop=True)


def _vix():
    try:
        import requests
        from dotenv import load_dotenv
        load_dotenv()
        h = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json"}
        r = requests.get("https://api.unusualwhales.com/api/stock/VIX/volatility/realized",
                         headers=h, params={"timeframe": "2Y"}, timeout=20).json().get("data", [])
        s = pd.Series({pd.Timestamp(x["date"]): float(x["price"]) for x in r if x.get("price")}).sort_index()
        med = s.shift(1).rolling(60, min_periods=20).median()
        return {d.date(): (s[d], (s[d] >= med[d]) if pd.notna(med[d]) else None) for d in s.index}
    except Exception as e:
        print(f"  (VIX unavailable: {e})")
        return {}


def _m(x):
    return f"{x.mean()*1e4:+6.1f}bp  med {x.median()*1e4:+6.1f}  %up {(x>0).mean():.0%}  n{len(x)}"


def _bracket_hold(entry_ask, cl, bp, exit_i):
    """simple hold-to-exit_i: enter ask, exit bid at exit_i. No TP/SL."""
    px = bp[exit_i] if exit_i < len(bp) else bp[-1]
    return (px - entry_ask) / entry_ask - COMM


def run(a):
    D = _spy_daily()
    vix = _vix()
    D["vix"] = D["date"].map(lambda d: vix.get(d.date(), (np.nan, None))[0])
    D["vixfav"] = D["date"].map(lambda d: vix.get(d.date(), (np.nan, None))[1])

    print("=" * 100)
    print(f"  SPY NIGHT EFFECT   {D['date'].min().date()} .. {D['date'].max().date()}  ({len(D)} sessions)")
    print("=" * 100)
    print(f"  overnight (close->open)  {_m(D.on)}")
    print(f"  intraday  (open->close)  {_m(D.intr)}")
    print(f"  close-to-close           {_m(D.cc)}")
    print(f"  $1 overnight-only ${np.prod(1+D.on):.3f}   intraday-only ${np.prod(1+D.intr):.3f}   "
          f"buy&hold ${np.prod(1+D.cc):.3f}")
    print(f"  overnight ann {(1+D.on.mean())**252-1:+.1%}   intraday ann {(1+D.intr.mean())**252-1:+.1%}")
    print(f"  skew  overnight {D.on.skew():+.2f}   intraday {D.intr.skew():+.2f}")
    up, dn = D.on[D.on > 0], D.on[D.on < 0]
    print(f"  gap UP {len(up)} ({len(up)/len(D):.0%}) avg +{up.mean()*1e4:.0f}bp   "
          f"gap DOWN {len(dn)} avg {dn.mean()*1e4:.0f}bp")

    print("\n  -- overnight drift conditioned --")
    for lbl, sub in (("VIX favourable (>=med)", D[D.vixfav == True]),
                     ("VIX low", D[D.vixfav == False])):
        if len(sub) >= 20:
            print(f"    {lbl:24} {_m(sub.on)}")
    print("    by weekday (Mon..Fri, the overnight INTO that day):")
    for wd in range(5):
        s = D[D.dow == wd]
        if len(s) >= 15:
            print(f"      {['Mon','Tue','Wed','Thu','Fri'][wd]}  {_m(s.on)}")
    print("    prior-day intraday: after DOWN day vs after UP day (overnight = reversal or continuation?):")
    print(f"      after prev intraday < 0   {_m(D[D.prev_intr < 0].on)}")
    print(f"      after prev intraday > 0   {_m(D[D.prev_intr > 0].on)}")

    # ---------------- THE TRADE ----------------
    if not os.path.exists(f"{CACHE}/calls.parquet"):
        print("\n  (run --build for the option-trade test)")
        return
    b = pd.read_parquet(f"{CACHE}/calls.parquet")
    b["date"] = pd.to_datetime(b["date"]).dt.date
    b["expiry"] = pd.to_datetime(b["expiry"]).dt.date
    bd = {d: g for d, g in b.groupby("date")}
    days = sorted(bd)
    di = {d: i for i, d in enumerate(days)}
    dmap = {pd.Timestamp(r["date"]).date(): r for _, r in D.iterrows()}

    print("\n" + "=" * 100)
    print("  THE TRADE: buy a SPY call ~15:55, exit ~09:35 next session (1 night).  realistic ask-in/bid-out")
    print("  grid: entry DTE x moneyness(strike/spot-1) x [1-night | to-next-close | hold-to-expiry]")
    print("=" * 100)

    def pick(day, mod, dte_lo, dte_hi, mny):
        at = day[(day["mod"] <= mod) & (day["mod"] >= mod - 5)]
        at = at[(at["dte"] >= dte_lo) & (at["dte"] <= dte_hi)]
        if at.empty:
            return None
        at = at.sort_values("mod").groupby("option_chain_id").last().reset_index()
        spot = float(at["underlying_close"].iloc[-1])
        tgt = spot * (1 + mny)
        return at.iloc[(at["strike"] - tgt).abs().argmin()]

    cbars = {c: g.sort_values(["date", "mod"]) for c, g in b.groupby("option_chain_id")}

    favmap = {pd.Timestamp(r["date"]).date(): r["vixfav"] for _, r in D.iterrows()}

    def s(recs):
        if len(recs) < 12:
            return f"n{len(recs)} thin"
        v = np.array([p for _, p in recs])
        o = [p for dd, p in recs if dd >= SPLIT]; ii = [p for dd, p in recs if dd < SPLIT]
        lo = [p for dd, p in recs if favmap.get(dd) == False]   # noqa: E712 (np.bool_)
        hi = [p for dd, p in recs if favmap.get(dd) == True]     # noqa: E712
        return (f"n{len(v):>3}  avg {v.mean()*100:>+6.1f}%  IS {np.mean(ii)*100:>+6.1f}  "
                f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}  win {(v>0).mean():.2f}  "
                f"| VIXlo {np.mean(lo)*100 if len(lo) >= 8 else float('nan'):>+6.1f}(n{len(lo)})  "
                f"VIXhi {np.mean(hi)*100 if len(hi) >= 8 else float('nan'):>+6.1f}(n{len(hi)})")

    for dte_lo, dte_hi in ((5, 6), (7, 8)):
        for mny in (0.0, -0.02, -0.035, -0.05):
            res = {"1n": [], "sprd": []}
            for i in range(len(days) - 1):
                d0, d1 = days[i], days[i + 1]
                row = pick(bd[d0], CLOSE_MOD, dte_lo, dte_hi, mny)
                sh = pick(bd[d0], CLOSE_MOD, dte_lo, dte_hi, mny + 0.02)   # short leg 2% higher strike
                if row is None:
                    continue
                g = cbars.get(row["option_chain_id"])
                if g is None:
                    continue
                e = g[(g["date"] == d0) & (g["mod"] <= CLOSE_MOD)]
                if e.empty:
                    continue
                e = e.iloc[-1]
                ea, eb = float(e["ask_close"]), float(e["bid_close"])
                emid = (ea + eb) / 2 if eb > 0 else float(e["close"])
                if not np.isfinite(emid) or emid < 0.50:
                    continue
                entry = ea if ea > 0 else emid
                x1 = g[(g["date"] == d1) & (g["mod"] >= OPEN_MOD - 4) & (g["mod"] <= OPEN_MOD + 8)].sort_values("mod")
                if x1.empty:
                    continue
                exb = float(x1.iloc[0]["bid_close"])
                if not np.isfinite(exb) or exb <= 0:
                    exb = float(x1.iloc[0]["close"])
                res["1n"].append((d0, (exb - entry) / entry - COMM))
                # spread: also sell the higher strike (enter at its bid, buy back at its ask)
                if sh is not None and sh["option_chain_id"] != row["option_chain_id"]:
                    gs = cbars.get(sh["option_chain_id"])
                    if gs is not None:
                        es = gs[(gs["date"] == d0) & (gs["mod"] <= CLOSE_MOD)]
                        xs = gs[(gs["date"] == d1) & (gs["mod"] >= OPEN_MOD - 4) & (gs["mod"] <= OPEN_MOD + 8)].sort_values("mod")
                        if not es.empty and not xs.empty:
                            s_in = float(es.iloc[-1]["bid_close"])          # we SELL -> receive bid
                            s_out = float(xs.iloc[0]["ask_close"])          # buy back -> pay ask
                            if np.isfinite(s_in) and np.isfinite(s_out) and s_in > 0:
                                net_debit = entry - s_in
                                if net_debit > 0.05:
                                    pnl = ((exb - s_out) - net_debit) / net_debit - COMM
                                    res["sprd"].append((d0, pnl))
            print(f"\n  DTE {dte_lo}-{dte_hi}  mny {mny:+.1%}  (1-night, exit next 09:35)")
            print(f"    long call    {s(res['1n'])}")
            print(f"    +2% call spread {s(res['sprd'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    else:
        run(a)
