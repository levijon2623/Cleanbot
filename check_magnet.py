# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0"]
# ///
"""
check_magnet.py
===============

"Magnet theory" (a WeBull-forum favourite): a stock that opens above or below a
large POSITIVE-gamma node tends to touch that node at some point in the session
(dealers long gamma there -> they buy weakness / sell strength -> price drawn in).

Per day, from the silver option tape (reuses check_gamma_walls):
  - profile dealer gamma-by-strike as of --open-asof (09:35 default)
  - node = the strike carrying the most POSITIVE net gamma
  - open_dist = (node - spot_open) / spot_open
  - touched = did RTH price reach the node (within --tol) after the open?
  - CONTROL: the mirror level -- same distance, OTHER side of spot -- touched?
    The magnet claim needs node-touch-rate > mirror-touch-rate at matched distance.

Splits: distance bucket, net-gamma sign (+ = suppression regime), node "hugeness"
(its share of the day's total positive gamma).

Usage:
  python check_magnet.py SPY 2024-08-20 2026-08-21 --range
  python check_magnet.py SPY,QQQ,IWM 2024-08-20 2026-08-21 --range --max-dte 3
  python check_magnet.py SPY 2024-08-20 2026-08-21 --range --min-node-share 0.25
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np

from check_gamma_walls import load_day, profile_at, _minute_px, _silver_dates

DIST_BUCKETS = [(0.0, 0.003), (0.003, 0.007), (0.007, 0.015), (0.015, 0.05)]
DIST_LABELS = ["<0.3%", "0.3-0.7%", "0.7-1.5%", ">1.5%"]


def _bucket(x):
    for i, (lo, hi) in enumerate(DIST_BUCKETS):
        if lo <= x < hi:
            return i
    return None


def run(args, tk, days):
    oh, om = map(int, args.open_asof.split(":"))
    open_mod = oh * 60 + om
    tol_bps = args.tol / 1e4

    # PASS 1: the ticker's unconditional post-open up/down range distribution --
    # the "would ANY level at this distance be touched" baseline that a real
    # magnet (or wall) has to beat.
    up_ranges, dn_ranges = [], []
    day_cache = {}
    for d in days:
        day = load_day(tk, d, args.max_dte, 0)
        day_cache[d] = day
        if day is None:
            continue
        px = _minute_px(day, open_mod, None)
        if px.is_empty():
            continue
        pxs = px["px"].to_list()
        o = pxs[0]
        if o <= 0:
            continue
        up_ranges.append((max(pxs) - o) / o)
        dn_ranges.append((o - min(pxs)) / o)
    up_ranges = np.array(up_ranges)
    dn_ranges = np.array(dn_ranges)

    def _p_up(dist):   return float(np.mean(up_ranges >= dist)) if len(up_ranges) else np.nan
    def _p_dn(dist):   return float(np.mean(dn_ranges >= dist)) if len(dn_ranges) else np.nan

    rows = []
    for d in days:
        day = day_cache.get(d)
        if day is None:
            continue
        pr = profile_at(day, open_mod, args.bs_fill)
        if pr is None:
            continue
        spot0, by_strike, cov, tot = pr
        if not by_strike:
            continue
        pos = {k: v for k, v in by_strike.items() if v > 0}
        if not pos:
            continue
        node = max(pos, key=pos.get)
        node_share = pos[node] / sum(pos.values())
        net_sign = "+" if sum(by_strike.values()) >= 0 else "-"
        if abs(node / spot0 - 1) < 1e-4:            # node ~= spot, nothing to test
            continue
        if node_share < args.min_node_share:
            continue
        open_dist = (node - spot0) / spot0
        above = node > spot0
        mirror = spot0 * (1 - open_dist)            # same distance, other side

        px = _minute_px(day, open_mod, None)
        if px.is_empty():
            continue
        pxs = px["px"].to_list()
        mods = px["_mod"].to_list()
        hi, lo = max(pxs), min(pxs)
        tol = spot0 * tol_bps

        node_hit = (hi >= node - tol) if above else (lo <= node + tol)
        mir_hit = (lo <= mirror + tol) if above else (hi >= mirror - tol)
        # baseline: P(a random level at this |dist| in the node's direction is touched)
        base = _p_up(abs(open_dist)) if above else _p_dn(abs(open_dist))
        ttt = None
        if node_hit:
            for m, p in zip(mods, pxs):
                if (p >= node - tol) if above else (p <= node + tol):
                    ttt = (m - open_mod) / (16 * 60 - open_mod)
                    break
        rows.append(dict(d=d, dist=abs(open_dist), above=above, net=net_sign,
                         share=node_share, node_hit=node_hit, mir_hit=mir_hit,
                         base=base, ttt=ttt))

    if not rows:
        print(f"  {tk}: no usable days"); return

    n = len(rows)
    nh = sum(r["node_hit"] for r in rows)
    mh = sum(r["mir_hit"] for r in rows)
    bexp = np.nanmean([r["base"] for r in rows])
    print("=" * 84)
    print(f"  {tk}   MAGNET TEST   open-asof {args.open_asof}   DTE 0-{args.max_dte}"
          + (f"   node share >= {args.min_node_share}" if args.min_node_share else ""))
    print(f"  {n} days   median dist {np.median([r['dist'] for r in rows])*100:.2f}%   "
          f"node above open {sum(r['above'] for r in rows)}/{n}")
    print("=" * 84)
    print(f"\n  NODE touched:     {nh}/{n}  ({100*nh/n:.0f}%)")
    print(f"  BASELINE expect:  {100*bexp:.0f}%   <- P(any level at that |dist|/side is touched, unconditional)")
    print(f"  MAGNET edge (node - baseline): {100*nh/n - 100*bexp:+.0f}pp   "
          f"(+ = magnet/drawn in · - = wall/repelled)")
    print(f"  [mirror-level touched: {100*mh/n:.0f}%  (sanity: should ~= its own baseline)]")
    tt = [r["ttt"] for r in rows if r["ttt"] is not None]
    if tt:
        print(f"  median time-to-touch: {np.median(tt)*100:.0f}% into the post-open session")

    def _line(lab, sub):
        if not sub:
            return
        sn = np.mean([r["node_hit"] for r in sub]) * 100
        sb = np.nanmean([r["base"] for r in sub]) * 100
        print(f"     {lab:14}n={len(sub):>4}   node {sn:>3.0f}%   baseline {sb:>3.0f}%   "
              f"edge {sn - sb:>+3.0f}pp")

    print(f"\n  by distance-at-open:")
    for i, lab in enumerate(DIST_LABELS):
        _line(lab, [r for r in rows if _bucket(r["dist"]) == i])
    print(f"\n  by net-gamma sign (+ = suppression regime):")
    for s in ("+", "-"):
        _line(f"net {s}", [r for r in rows if r["net"] == s])
    if not args.min_node_share:
        print(f"\n  by node hugeness (its share of the day's +gamma):")
        for lo, hi in [(0.0, 0.2), (0.2, 0.35), (0.35, 1.01)]:
            _line(f"share {lo:.2f}-{hi:.2f}", [r for r in rows if lo <= r["share"] < hi])
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker")
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--range", action="store_true")
    ap.add_argument("--open-asof", default="9:35")
    ap.add_argument("--max-dte", type=int, default=5)
    ap.add_argument("--min-node-share", type=float, default=0.0,
                    help="only count days where the node holds >= this share of the day's +gamma")
    ap.add_argument("--tol", type=float, default=5.0, help="touch tolerance in bps of spot (default 5)")
    ap.add_argument("--bs-fill", action="store_true", help="Black-Scholes-fill gamma for untraded contracts")
    a = ap.parse_args()
    tickers = [t.strip().upper() for t in a.ticker.split(",") if t.strip()]
    if a.range and len(a.dates) == 2:
        lo, hi = sorted(date.fromisoformat(x) for x in a.dates)
        dd = [d for d in _silver_dates() if lo <= d <= hi]
    else:
        dd = sorted(date.fromisoformat(x) for x in a.dates)
    for t in tickers:
        run(a, t, dd)
