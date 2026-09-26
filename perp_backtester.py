# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
perp_backtester.py
==================

Experiment 2: take the SAME flow-trigger / regime / percentile-flow entry logic
the live bot uses, but instead of buying a call/put, go long/short the
underlying's Hyperliquid perp (delta 1:1, no theta).

  entry   -- market (taker) at the underlying's price at the trigger minute
  TP / SL -- resting limit orders (maker) at +tp% / -sl% in UNDERLYING terms
  exit    -- also max-hold (rule time_stop_mins) and an EOD flatten
  fees    -- --fee-in (taker, 0.045%) + --fee-out (maker, 0.015%); --sl-slippage
             (5bps) on stop-side fills since a stop limit gets run through more

P&L is reported as **% of NOTIONAL** (leverage-agnostic). --leverage N also shows
% of margin (= notional% x N) and the per-trade fee drag in margin terms
(at 20x a 0.06% notional round-trip = 1.2% of margin).

Underlying 1-min OHLC from historical/{T}.parquet.  Perp availability on the xyz
dex: META/MSFT/NVDA/AMZN/AVGO have direct perps, SPY->xyz:SP500, GLD->xyz:GOLD;
IWM and QQQ have NO xyz perp (flagged) -- backtested on the ETF path anyway
(perp ~= underlying intraday, verified session 12).

Usage:
  python perp_backtester.py --split 2025-08-21
  python perp_backtester.py --split 2025-08-21 --leverage 20 --vs-option
  python perp_backtester.py --split 2025-08-21 --tickers NVDA GLD
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl

from config import RULES
import directional_flow_backtester as D

HIST = "historical"
NO_PERP = {"IWM", "QQQ"}          # no xyz perp -- would trade the ETF
PERP_MAP = {"SPY": "xyz:SP500", "GLD": "xyz:GOLD", "SLV": "xyz:SILVER"}
TP_GRID = [0.003, 0.005, 0.008, 0.012, 0.018]
SL_GRID = [0.003, 0.005, 0.008, 0.012]


def _load_underlying_1m(tk: str) -> pd.DataFrame:
    df = pl.read_parquet(f"{HIST}/{tk}.parquet").to_pandas()
    df.columns = [c.lower() for c in df.columns]
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    mod = et.dt.hour * 60 + et.dt.minute
    m = (mod >= 570) & (mod <= 960)
    out = pd.DataFrame({
        "minute_et": et[m].values,
        "high": df["high"][m].astype(float).values,
        "low": df["low"][m].astype(float).values,
        "close": df["close"][m].astype(float).values,
    })
    out["minute_et"] = pd.to_datetime(out["minute_et"])
    out["date"] = out["minute_et"].dt.date
    return out.sort_values("minute_et").reset_index(drop=True)


def _sim_perp(t, is_long, day, tp, sl, tstop, eod_mod, fee_in, fee_out, sl_slip):
    """One perp trade -> (date, pnl_frac_of_notional) or None."""
    at = day[day["minute_et"] <= t["ts"]]
    if at.empty:
        return None
    entry = float(at.iloc[-1]["close"])
    fwd = day[day["minute_et"] > t["ts"]]
    if len(fwd) < 3 or entry <= 0:
        return None
    hi, lo, cl = fwd["high"].values, fwd["low"].values, fwd["close"].values
    mod = fwd["minute_et"].dt.hour.values * 60 + fwd["minute_et"].dt.minute.values
    held = (fwd["minute_et"].values - np.datetime64(pd.Timestamp(t["ts"]))) / np.timedelta64(1, "m")
    n = len(cl)
    eod = mod >= eod_mod
    eod_idx = int(np.argmax(eod)) if eod.any() else n - 1
    ts_idx = int(np.argmax(held >= tstop)) if (tstop and (held >= tstop).any()) else n - 1
    hard = min(eod_idx, ts_idx)

    tp_lvl = entry * (1 + tp) if is_long else entry * (1 - tp)
    sl_lvl = entry * (1 - sl) if is_long else entry * (1 + sl)
    if is_long:
        tp_hits = np.nonzero(hi[:hard + 1] >= tp_lvl)[0]
        sl_hits = np.nonzero(lo[:hard + 1] <= sl_lvl)[0]
    else:
        tp_hits = np.nonzero(lo[:hard + 1] <= tp_lvl)[0]
        sl_hits = np.nonzero(hi[:hard + 1] >= sl_lvl)[0]
    tp_i = int(tp_hits[0]) if len(tp_hits) else 10 ** 9
    sl_i = int(sl_hits[0]) if len(sl_hits) else 10 ** 9

    if tp_i <= sl_i and tp_i <= hard:
        move = tp if is_long else tp            # favourable
        fee = fee_in + fee_out
        pnl = move - fee
    elif sl_i < tp_i and sl_i <= hard:
        move = -(sl + sl_slip)
        fee = fee_in + fee_out
        pnl = move - fee
    else:
        exit_px = cl[hard]
        raw = (exit_px - entry) / entry
        move = raw if is_long else -raw
        pnl = move - (fee_in + fee_out)
    return (t["date"], pnl)


def _exp(pnls, split, side):
    v = [p for d, p in pnls if (d < split) == (side == "IS")] if split else [p for _, p in pnls]
    if not v:
        return None
    return float(np.mean(v)) * 100


def run(args):
    flow = (D.build_flow_netprem(HIST) if args.flow_source == "netprem"
            else D.build_flow_1m(args.lake))
    flow["minute_et"] = D._naive(flow["minute_et"])
    flow["date"] = flow["minute_et"].dt.date
    split = pd.to_datetime(args.split).date() if args.split else None
    eod_mod = 15 * 60 + 55
    if args.eod_flatten:
        h, m = map(int, args.eod_flatten.split(":"))
        eod_mod = h * 60 + m

    rules = [r for r in RULES if r.get("enabled", True)]
    if args.tickers:
        keep = {t.upper() for t in args.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]

    lev = args.leverage
    print("=" * 104)
    print(f"  PERP BACKTEST   ({len(rules)} rules"
          + (f"   IS < {split} <= OOS" if split else "") + f")   fee {args.fee_in*100:.3f}%/{args.fee_out*100:.3f}%"
          f"  sl-slip {args.sl_slippage*1e4:.0f}bps  flatten {eod_mod//60}:{eod_mod%60:02d}")
    print(f"  P&L = % of NOTIONAL" + (f"   ·  at {lev}x: x{lev} for % of margin, fee drag {(args.fee_in+args.fee_out)*lev*100:.2f}%/trade" if lev else ""))
    print("=" * 104)

    by_ticker = defaultdict(list)
    for r in rules:
        by_ticker[r["ticker"].upper()].append(r)

    for tk, tk_rules in by_ticker.items():
        try:
            day_px = _load_underlying_1m(tk)
        except FileNotFoundError:
            print(f"  {tk}: no historical/{tk}.parquet"); continue
        px_by_date = {d: g for d, g in day_px.groupby("date")}
        gex = D.load_gex(HIST, tk)
        vol = D.load_volume_regime(HIST, tk)
        trd = D.load_trend_regime(HIST, tk, args.trend_fast, args.trend_slow)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
                   "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        trigs = D.triggers_for(flow, tk)
        if not trigs:
            tkf, _ = D._screen_build_one(args.lake, tk)
            trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
        if not trigs:
            print(f"  {tk}: no flow triggers"); continue
        D.annotate_flow_pct(trigs, args.flow_window)

        # option baseline (for --vs-option)
        bbc = bbd = None
        if args.vs_option:
            tb = D._ticker_bars(tk)
            if tb is None or tb.empty:
                _, tb = D._screen_build_one(args.lake, tk)
            if tb is not None and not tb.empty:
                bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
                bbd = {d: g for d, g in tb.groupby("date")}

        for r in tk_rules:
            matched = [(t, thr) for t, thr in
                       D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
                       if t["abs_flow"] >= thr]
            is_long = r["direction"].upper() == "CALL"
            tstop = r.get("time_stop_mins")
            perp = "ETF (no xyz perp)" if tk in NO_PERP else PERP_MAP.get(tk, f"xyz:{tk}")
            print(f"\n  {r.get('name', tk)}   [{perp}]   {len(matched)} triggers   dir {'LONG' if is_long else 'SHORT'}")
            print("     TP\\SL " + "".join(f"{s*100:>10.1f}%" for s in SL_GRID))
            best = None
            for tp in TP_GRID:
                cells = []
                for sl in SL_GRID:
                    pnls = []
                    for t, _thr in matched:
                        day = px_by_date.get(t["date"])
                        if day is None:
                            continue
                        x = _sim_perp(t, is_long, day, tp, sl, tstop, eod_mod,
                                      args.fee_in, args.fee_out, args.sl_slippage)
                        if x is not None:
                            pnls.append(x)
                    oe, ie = _exp(pnls, split, "OOS"), _exp(pnls, split, "IS")
                    cells.append((oe, ie))
                    if oe is not None and (best is None or oe > best[0]):
                        best = (oe, ie, tp, sl, len(pnls))
                row = "".join((f"{c[0]:>+5.2f}/{c[1]:>+5.2f}" if c[0] is not None and c[1] is not None
                               else f"{'--':>11}") for c in cells)
                print(f"    {tp*100:>4.1f}%  {row}")
            if best:
                bm = f"   ·   {best[0]*lev:+.1f}% margin @ {lev}x" if lev else ""
                print(f"    -> best OOS {best[0]:+.2f}% / IS {best[1]:+.2f}%  of notional  "
                      f"(TP {best[2]*100:.1f}% / SL {best[3]*100:.1f}%, n={best[4]}){bm}")

            if bbd is not None:
                op = []
                for t, thr in matched:
                    for _, _, _, dd, pnl in D.simulate_trigger(
                            t, r["direction"].upper(), r.get("dte", [0, 1]), tstop,
                            bbd, bbc, only=(thr, float(r["target_roe"]), float(r["rr"]))):
                        op.append((dd, pnl))
                oe, ie = _exp(op, split, "OOS"), _exp(op, split, "IS")
                if oe is not None:
                    print(f"    OPTION (config TP+{r['target_roe']*100:.0f}%/RR{r['rr']:.1f}):  "
                          f"IS {ie:+.1f}%  OOS {oe:+.1f}%  of PREMIUM  (n={len(op)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", help="YYYY-MM-DD IS/OOS split")
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--flow-source", choices=("silver", "netprem"), default="netprem")
    ap.add_argument("--flow-window", type=int, default=60)
    ap.add_argument("--trend-fast", type=int, default=20)
    ap.add_argument("--trend-slow", type=int, default=50)
    ap.add_argument("--lake", default="lake/silver/option-contracts-1m")
    ap.add_argument("--leverage", type=int, default=None, help="also show margin-pct at Nx (HL caps most at 20x)")
    ap.add_argument("--fee-in", type=float, default=0.00045, help="taker fee, entry (default 0.045%%)")
    ap.add_argument("--fee-out", type=float, default=0.00015, help="maker fee, TP/SL exit (default 0.015%%)")
    ap.add_argument("--sl-slippage", type=float, default=0.0005, help="extra loss on stop-side fills (default 5bps)")
    ap.add_argument("--eod-flatten", metavar="HH:MM", help="override 15:55 EOD flatten")
    ap.add_argument("--vs-option", action="store_true", help="also print the option-version OOS expectancy per rule")
    a = ap.parse_args()
    if a.tickers:
        a.tickers = [t.upper() for t in a.tickers]
    run(a)
