# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0"]
# ///
"""
check_wall_overnight.py
=======================

Does a breached gamma wall pull price back overnight? And would a 1DTE spread
(held through the NEXT day's close) beat the same-day 0DTE version?

Per consecutive day pair (D, D+1), for the walls set on D at --wall-asof:

  1. BREACH RECOVERY -- for each wall that D CLOSED beyond (by > --tol):
       how often is price back on the safe side at D+1 close?  intraday D+1?
       + overshoot size, split by side and net-gamma sign

  2. 0DTE vs 1DTE HOLD -- for each wall TOUCHED in --window on D:
       held_0DTE = D   close on the safe side of the wall (settles OTM today)
       held_1DTE = D+1 close on the safe side          (settles OTM tomorrow)
       + max adverse excursion (how far beyond the wall the 2-day path goes --
         tells you how wide the long wing must be / whether a stop triggers)

Reuses check_gamma_walls' wall derivation (dealer gamma-by-strike from silver).

Usage:
  python check_wall_overnight.py IWM 2024-08-20 2026-08-21 --range
  python check_wall_overnight.py SPY,QQQ,IWM 2024-08-20 2026-08-21 --range --net-pos
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np

from check_gamma_walls import load_day, profile_at, walls, _minute_px, _silver_dates


def run(args, tk, days):
    wh, wm = map(int, args.wall_asof.split(":"))
    wall_mod = wh * 60 + wm
    (w0h, w0m), (w1h, w1m) = (map(int, s.split(":")) for s in args.window.split("-"))
    win_lo, win_hi = w0h * 60 + w0m, w1h * 60 + w1m
    tol = args.tol

    cache: dict = {}

    def get(d):
        if d not in cache:
            cache[d] = load_day(tk, d, 2, 0)
        return cache[d]

    breaches, touches = [], []
    for i in range(len(days) - 1):
        D, D1 = days[i], days[i + 1]
        dD, dD1 = get(D), get(D1)
        if dD is None or dD1 is None:
            continue
        pr = profile_at(dD, wall_mod, False)
        if pr is None:
            continue
        spot0, bs, _, _ = pr
        w = walls(spot0, bs)
        if w is None:
            continue
        cw, pw, net = w
        if args.net_pos and net != "+":
            continue
        pxD = _minute_px(dD, wall_mod, None)
        pxD1 = _minute_px(dD1)
        if pxD.is_empty() or pxD1.is_empty():
            continue
        modsD, pxsD = pxD["_mod"].to_list(), pxD["px"].to_list()
        pxs1 = pxD1["px"].to_list()
        D_close = pxsD[-1]
        D1_close, D1_hi, D1_lo = pxs1[-1], max(pxs1), min(pxs1)

        for side, wall, is_call in (("CALL", cw, True), ("PUT", pw, False)):
            if wall is None:
                continue

            breached = (D_close > wall + tol) if is_call else (D_close < wall - tol)
            if breached:
                over = (D_close - wall) if is_call else (wall - D_close)
                safe_c = (D1_close <= wall) if is_call else (D1_close >= wall)
                safe_i = (D1_lo <= wall) if is_call else (D1_hi >= wall)
                breaches.append({"D": D, "side": side, "net": net,
                                 "over_pct": over / D_close * 100,
                                 "safe_close": safe_c, "safe_intra": safe_i})

            hit = None
            for m, p in zip(modsD, pxsD):
                if win_lo <= m <= win_hi and (
                        (p >= wall - tol) if is_call else (p <= wall + tol)):
                    hit = m
                    break
            if hit is None:
                continue
            held0 = (D_close <= wall) if is_call else (D_close >= wall)
            held1 = (D1_close <= wall) if is_call else (D1_close >= wall)
            path = [p for m, p in zip(modsD, pxsD) if m >= hit] + pxs1
            mae = (max(path) - wall) if is_call else (wall - min(path))
            touches.append({"D": D, "side": side, "net": net,
                            "held0": held0, "held1": held1,
                            "mae_pct": max(0.0, mae) / wall * 100})

    print("=" * 74)
    print(f"  {tk}   gamma-wall overnight behaviour"
          + ("   [net-+gamma days only]" if args.net_pos else ""))
    print(f"  wall-asof {args.wall_asof}   touch window {args.window}   tol ${tol}")
    print("=" * 74)

    def pct(rows, key):
        return f"{sum(r[key] for r in rows):>4}/{len(rows):<4} ({100*sum(r[key] for r in rows)/len(rows):>3.0f}%)"

    print(f"\n  1. BREACH RECOVERY  -- walls D CLOSED beyond, then D+1:")
    if breaches:
        print(f"     n breached: {len(breaches)}   median overshoot {np.median([b['over_pct'] for b in breaches]):.2f}%")
        print(f"     back on safe side by D+1 CLOSE:     {pct(breaches, 'safe_close')}")
        print(f"     touched safe side intraday on D+1:  {pct(breaches, 'safe_intra')}")
        for s in ("CALL", "PUT"):
            sub = [b for b in breaches if b["side"] == s]
            if sub:
                print(f"       {s:4} by D+1 close: {pct(sub, 'safe_close')}")
        for nn in ("+", "-"):
            sub = [b for b in breaches if b["net"] == nn]
            if sub:
                print(f"       net {nn} at D close: {pct(sub, 'safe_close')}")
    else:
        print("     (no breached-at-close walls)")

    print(f"\n  2. WALL TOUCH -- 0DTE (D close) vs 1DTE (D+1 close) settlement:")
    if touches:
        n = len(touches)
        print(f"     n touches: {n}")
        print(f"     held 0DTE (safe at D close):    {pct(touches, 'held0')}")
        print(f"     held 1DTE (safe at D+1 close):  {pct(touches, 'held1')}")
        recov = [t for t in touches if not t["held0"] and t["held1"]]
        gaveup = [t for t in touches if t["held0"] and not t["held1"]]
        print(f"     0DTE loss -> 1DTE win (overnight recovery): {len(recov)}")
        print(f"     0DTE win  -> 1DTE loss (overnight give-back): {len(gaveup)}")
        for nn in ("+", "-"):
            sub = [t for t in touches if t["net"] == nn]
            if sub:
                print(f"       net {nn}:  0DTE {pct(sub, 'held0')}   1DTE {pct(sub, 'held1')}")
        m = np.array([t["mae_pct"] for t in touches])
        print(f"     max adverse excursion beyond wall (2-day path): "
              f"median {np.median(m):.2f}%   p75 {np.percentile(m, 75):.2f}%   p90 {np.percentile(m, 90):.2f}%")
    else:
        print("     (no wall touches in the window)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker", help="one ticker or comma list")
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--range", action="store_true")
    ap.add_argument("--wall-asof", default="10:30")
    ap.add_argument("--window", default="11:00-15:00")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--net-pos", action="store_true")
    a = ap.parse_args()
    tickers = [t.strip().upper() for t in a.ticker.split(",") if t.strip()]
    if a.range and len(a.dates) == 2:
        lo, hi = sorted(date.fromisoformat(x) for x in a.dates)
        dd = [d for d in _silver_dates() if lo <= d <= hi]
    else:
        dd = sorted(date.fromisoformat(x) for x in a.dates)
    for t in tickers:
        run(a, t, dd)
        print()
