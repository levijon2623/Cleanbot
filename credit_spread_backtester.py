# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0"]
# ///
"""
credit_spread_backtester.py
===========================

0DTE defined-risk credit spread on a gamma wall, from the silver option tape.

Per trading day:
  - wall levels as of --wall-asof (10:30) from the dealer gamma-by-strike
    profile (check_gamma_walls.profile_at / walls)
  - watch --window (11:00-15:00) for the underlying to touch within
    --touch-tol of a wall
  - on a CALL-wall touch  -> SELL a call credit spread  (short = wall strike,
    long = short + --width)
  - on a PUT-wall touch   -> SELL a put credit spread   (short = wall strike,
    long = short - --width)
  - CONSERVATIVE fills: enter selling the short at BID / buying the long at ASK;
    exit buying the short at ASK / selling the long at BID.  (This is the
    assumption L3's iron condors died on -- if it survives here it's real.)
  - manage minute-by-minute to close:
      take profit  when buy-back cost <= (1 - --tp) * credit
      stop         when buy-back cost >= --stop * credit
      else settle at 15:55 intrinsic (0DTE -> expires today)
  - PnL = (credit - exit_cost) * 100 - 4 legs * --commission

Grids --width x --tp x --stop and prints each cell (n, win%, avg/total $,
avg credit, avg hold). --split does IS/OOS walk-forward.

Filters:
  --net-pos     only days with net-positive dealer gamma at entry (the signal
                check_gamma_walls found: +gamma -> walls hold; IWM ~64%)
  --trend / --vol   prior-day regime gates (reuse directional_flow_backtester)
  --side call|put   restrict to one wall

Usage:
  python credit_spread_backtester.py IWM 2024-08-20 2026-08-21 --range --net-pos
  python credit_spread_backtester.py IWM 2024-08-20 2026-08-21 --range --net-pos --split 2025-08-21
  python credit_spread_backtester.py SPY,QQQ 2024-08-20 2026-08-21 --range --net-pos --width 5
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import numpy as np
import polars as pl

from check_gamma_walls import (LAKE, load_day, profile_at, walls, _minute_px,
                               _silver_dates, RTH_CLOSE)

EOD_MOD = 15 * 60 + 55
MAX_STALE = 8          # don't use a forward-filled quote older than this many minutes
GRID_WIDTH = [1.0, 2.0, 3.0]
GRID_TP = [0.4, 0.5, 0.6]
GRID_STOP = [1.5, 2.0, 2.5]


def _contract_quotes(day: pl.DataFrame, strike: float, is_call: bool):
    """{mod: (bid, ask)} for the 0DTE contract at `strike`, forward-filled."""
    cp = "call" if is_call else "put"
    c = (day.filter((pl.col("dte") == 0) & (pl.col("option_type") == cp)
                    & (pl.col("strike") == strike))
         .select("_mod", "bid_close", "ask_close").sort("_mod"))
    if c.is_empty():
        return None
    rows = c.to_dicts()
    out, last = {}, None
    for r in rows:
        b, a = r["bid_close"], r["ask_close"]
        if b is not None and a is not None and a > 0:
            last = (r["_mod"], float(b), float(a))
        if last is not None:
            out[r["_mod"]] = last
    return out  # {mod: (obs_mod, bid, ask)}


def _q(qmap, mod):
    """most recent (bid, ask) at/<=mod, if fresh enough."""
    best = None
    for m, v in qmap.items():
        if m > mod:
            break
        best = v
    if best is None:
        return None
    obs, b, a = best
    return (b, a) if mod - obs <= MAX_STALE else None


def one_trade(day, d, side, wall, width, tp, stop, commission, u_series):
    """Simulate one credit spread. Returns dict or None (couldn't quote)."""
    is_call = side == "call"
    short_k = wall
    long_k = short_k + width if is_call else short_k - width
    sq = _contract_quotes(day, short_k, is_call)
    lq = _contract_quotes(day, long_k, is_call)
    if not sq or not lq:
        return None

    # entry = first minute in the window with a live touch already found by caller;
    # caller passes the touch minute via u_series[0]
    entry_mod, _ = u_series[0]
    se = _q(sq, entry_mod)
    le = _q(lq, entry_mod)
    if not se or not le:
        return None
    credit = se[0] - le[1]                       # short bid - long ask
    if credit < 0.05:
        return None
    max_loss = width - credit

    exit_reason, exit_cost, exit_mod = None, None, None
    for mod, upx in u_series[1:]:
        s = _q(sq, mod)
        l = _q(lq, mod)
        if not s or not l:
            continue
        cost = s[1] - l[0]                       # buy short at ask, sell long at bid
        cost = max(0.0, min(cost, width))
        if cost <= (1 - tp) * credit:
            exit_reason, exit_cost, exit_mod = "TP", cost, mod
            break
        if cost >= stop * credit:
            exit_reason, exit_cost, exit_mod = "STOP", min(cost, width), mod
            break
    if exit_reason is None:                      # settle at expiry
        u_close = u_series[-1][1]
        itm = (u_close - short_k) if is_call else (short_k - u_close)
        exit_cost = max(0.0, min(itm, width))
        exit_reason, exit_mod = "EOD", u_series[-1][0]

    gross = (credit - exit_cost) * 100.0
    net = gross - 4 * commission
    return {"date": d, "side": side, "short_k": short_k, "credit": round(credit, 2),
            "exit": exit_reason, "exit_cost": round(exit_cost, 2),
            "hold": exit_mod - entry_mod, "pnl": round(net, 2),
            "win": net > 0, "max_loss": round(max_loss * 100 - 4 * commission, 2)}


def collect_touches(args, tk, days):
    """[(date, day_df, side, wall)] -- one entry per day (first wall touch in window)."""
    wh, wm = map(int, args.wall_asof.split(":"))
    wall_mod = wh * 60 + wm
    (w0h, w0m), (w1h, w1m) = (map(int, s.split(":")) for s in args.window.split("-"))
    win_lo, win_hi = w0h * 60 + w0m, w1h * 60 + w1m
    tol = args.touch_tol

    trd = vol = {}
    if args.trend or args.vol:
        from directional_flow_backtester import load_trend_regime, load_volume_regime
        if args.trend:
            trd = load_trend_regime(args.hist, tk)
        if args.vol:
            vol = load_volume_regime(args.hist, tk)

    out = []
    for d in days:
        if args.trend and trd.get(d) != args.trend:
            continue
        if args.vol and vol.get(d) != args.vol:
            continue
        day = load_day(tk, d, 2, 0)
        if day is None:
            continue
        pr = profile_at(day, wall_mod, False)
        if pr is None:
            continue
        spot0, bs, _, _ = pr
        w = walls(spot0, bs)
        if w is None:
            continue
        cw, pw, net = w
        if args.net_pos and net != "+":
            continue
        px = _minute_px(day, wall_mod, None)
        mods, pxs = px["_mod"].to_list(), px["px"].to_list()
        touch = None
        for m, p in zip(mods, pxs):
            if not (win_lo <= m <= win_hi):
                continue
            if args.side != "put" and cw is not None and p >= cw - tol:
                touch = ("call", m, cw); break
            if args.side != "call" and pw is not None and p <= pw + tol:
                touch = ("put", m, pw); break
        if touch is None:
            continue
        side, tm, wall = touch
        # underlying path entry->close for settlement / management ticks
        u = [(m, p) for m, p in zip(mods, pxs) if tm <= m <= RTH_CLOSE]
        if len(u) < 3:
            continue
        out.append((d, day, side, wall, u))
    return out


def _report(rows, label, split):
    print(f"\n  {label}")
    if not rows:
        print("    (no trades)")
        return
    for grid in [(w, tp, st) for w in GRID_WIDTH for tp in GRID_TP for st in GRID_STOP]:
        cell = [r for (g, r) in rows if g == grid]
        if len(cell) < 15:
            continue
        w, tp, st = grid

        def stat(rs):
            if not rs:
                return "  --"
            a = np.array([r["pnl"] for r in rs])
            wr = np.mean([r["win"] for r in rs]) * 100
            cr = np.mean([r["credit"] for r in rs])
            hd = np.mean([r["hold"] for r in rs])
            return f"n={len(rs):>3} win {wr:>3.0f}%  avg ${a.mean():>+6.1f}  tot ${a.sum():>+7.0f}  cr ${cr:.2f}  {hd:>3.0f}m"

        if split:
            is_ = [r for r in cell if r["date"] < split]
            oos = [r for r in cell if r["date"] >= split]
            print(f"    w{w:.0f} tp{tp:.0%} stop{st:.1f}x   IS {stat(is_)}")
            print(f"    {'':21}   OOS {stat(oos)}")
        else:
            print(f"    w{w:.0f} tp{tp:.0%} stop{st:.1f}x   {stat(cell)}")


def run(args):
    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    if args.range and len(args.dates) == 2:
        lo, hi = sorted(date.fromisoformat(x) for x in args.dates)
        all_days = [d for d in _silver_dates() if lo <= d <= hi]
    else:
        all_days = [date.fromisoformat(x) for x in args.dates]
    split = date.fromisoformat(args.split) if args.split else None

    for tk in tickers:
        print("=" * 78)
        print(f"  {tk}   0DTE credit spread on a gamma wall"
              + ("  [net-+gamma only]" if args.net_pos else "")
              + (f"  [{args.trend}]" if args.trend else "")
              + (f"  [{args.vol}]" if args.vol else ""))
        print(f"  wall-asof {args.wall_asof}  window {args.window}  touch ${args.touch_tol}  "
              f"comm ${args.commission}/leg   conservative fills")
        print("=" * 78)
        touches = collect_touches(args, tk, all_days)
        print(f"  entry days: {len(touches)}")
        if not touches:
            continue
        rows, skipped = [], 0
        for d, day, side, wall, u in touches:
            for grid in [(w, tp, st) for w in GRID_WIDTH for tp in GRID_TP for st in GRID_STOP]:
                w, tp, st = grid
                t = one_trade(day, d, side, wall, w, tp, st, args.commission, u)
                if t is None:
                    skipped += 1
                    continue
                rows.append((grid, t))
        print(f"  grid-trades: {len(rows)}   skipped (no quotes): {skipped}")
        _report(rows, f"{tk}  --  width x TP x stop  (cells with >=15 trades)", split)
        # exit-reason mix at a representative cell
        rep = [r for (g, r) in rows if g == (2.0, 0.5, 2.0)]
        if rep:
            from collections import Counter
            c = Counter(r["exit"] for r in rep)
            print(f"\n  exit mix @ w2/tp50/stop2.0:  " + "  ".join(f"{k} {v}" for k, v in c.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker", help="one ticker or comma list")
    ap.add_argument("dates", nargs="+", help="YYYY-MM-DD ... (or START END with --range)")
    ap.add_argument("--range", action="store_true")
    ap.add_argument("--split", metavar="YYYY-MM-DD", help="walk-forward: entries before this are IS")
    ap.add_argument("--wall-asof", default="10:30")
    ap.add_argument("--window", default="11:00-15:00")
    ap.add_argument("--touch-tol", type=float, default=0.5)
    ap.add_argument("--net-pos", action="store_true", help="only net-positive-gamma days at entry")
    ap.add_argument("--side", choices=("call", "put", "both"), default="both")
    ap.add_argument("--width", type=float, help="single spread width (default: grid 1/2/3)")
    ap.add_argument("--tp", type=float, help="single take-profit frac (default: grid .4/.5/.6)")
    ap.add_argument("--stop", type=float, help="single stop multiple (default: grid 1.5/2/2.5)")
    ap.add_argument("--commission", type=float, default=0.65, help="$/contract/leg (default 0.65)")
    ap.add_argument("--trend", choices=("UPTREND", "CHOP", "DOWNTREND"))
    ap.add_argument("--vol", choices=("LOWVOL", "NORMVOL", "HIVOL"))
    ap.add_argument("--hist", default="historical")
    a = ap.parse_args()
    if a.width:
        GRID_WIDTH[:] = [a.width]
    if a.tp:
        GRID_TP[:] = [a.tp]
    if a.stop:
        GRID_STOP[:] = [a.stop]
    run(a)
