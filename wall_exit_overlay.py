# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
wall_exit_overlay.py
====================

Exit-management overlay for the live directional RULES.  Yesterday's finding:
a gamma wall, once price reaches it, holds for the rest of the day ~40-64% of
the time (better on net-positive-gamma days -- IWM +gamma 64%, SPY +gamma 44%).

So: when an OPEN position's take-profit sits BEYOND a gamma wall, and price runs
into that wall and STALLS there (doesn't break through within --stall-mins) on a
+gamma day, the TP probably won't fill -- the wall caps the move.  Rather than
round-trip the open profit waiting for a TP that won't come (or an EOD/stop),
close at the wall.

This backtests that overlay against the plain TP/SL/EOD exit on the exact trades
the 9 enabled config.RULES would have taken (netprem flow, same fill model as
directional_flow_backtester).  Per rule and pooled it reports:

  plain    - expectancy / win% with the rule's own TP/SL/EOD exit
  overlay  - same trades, but the wall-stall early exit applied
  fired    - how many trades the overlay actually changed, and on THAT subset:
             plain vs overlay  ->  did cutting at the wall save money (wall held)
             or cost money (wall broke after we left)?

Wall = check_gamma_walls' dealer gamma-by-strike, profiled at the entry minute
(no lookahead -- OI is stable intraday).  The barrier is the call wall above spot
for a CALL, the put wall below for a PUT.

Usage:
  python wall_exit_overlay.py --split 2025-08-21
  python wall_exit_overlay.py --split 2025-08-21 --stall-mins 20 --tol 0.4 --no-net-pos
  python wall_exit_overlay.py --split 2025-08-21 --tickers IWM SPY QQQ
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from config import RULES
import directional_flow_backtester as D
from check_gamma_walls import load_day as wall_load_day, profile_at, walls

ALL_HOURS = list(range(9, 15))


def _rule_triggers(tk, tk_rules, flow, args):
    """Mirror directional_flow_backtester.run_rules' per-trigger filter, yielding
    (rule, trigger, flow_threshold) for every trade the rule would take."""
    gex = D.load_gex(args.hist, tk)
    vol = D.load_volume_regime(args.hist, tk)
    trd = D.load_trend_regime(args.hist, tk)
    _days = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _days}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
               "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
    trigs = D.triggers_for(flow, tk)
    D.annotate_flow_pct(trigs, args.flow_window)

    for r in tk_rules:
        direction = r["direction"].upper()
        hours = set(r.get("hours", ALL_HOURS))
        reg = r.get("regime")
        amp_min = r.get("amp_min")
        fpct = r.get("min_flow_pct", r.get("flow_pct"))
        for t in trigs:
            if t["dir"] != direction or t["hour"] not in hours or t["hour"] >= 15:
                continue
            d = t["date"]
            if reg is not None:
                val = (gex if reg.endswith("_GEX") else reg_src[reg]).get(d)
                if reg == "NEGATIVE_GEX" and val != "NEGATIVE": continue
                if reg == "POSITIVE_GEX" and val != "POSITIVE": continue
                if not reg.endswith("_GEX") and val != reg: continue
            if amp_min is not None and amp.get(d, -1) < amp_min:
                continue
            thr_map = t.get("thr")
            if not thr_map or int(fpct) not in thr_map:
                continue
            yield r, t, thr_map[int(fpct)]


def _sim(t, direction, dte, tr, rr, bars_by_date, bars_by_cid, wall, net_sign, args):
    """One trade -> (date, plain_pnl, overlay_pnl, fired) or None.
    Fill model copied from directional_flow_backtester.simulate_trigger."""
    day = bars_by_date.get(t["date"])
    if day is None:
        return None
    at_spot = day[day["minute_et"] <= t["ts"]]
    if at_spot.empty:
        return None
    spot = float(at_spot.iloc[-1]["underlying_close"])
    cid = D.pick_contract(day, t["ts"], direction, dte, spot)
    if cid is None:
        return None
    ent = bars_by_cid[cid]
    ent_row = ent[(ent["minute_et"] <= t["ts"]) & (ent["minute_et"] >= t["ts"] - pd.Timedelta(minutes=3))]
    if ent_row.empty:
        return None
    er = ent_row.iloc[-1]
    b, a = float(er["bid_close"]), float(er["ask_close"])
    entry_mid = (b + a) / 2.0 if b > 0 else float(er["close"])
    if entry_mid < 0.50:
        return None
    path = D.forward_path(bars_by_cid, cid, t["ts"])
    if path is None:
        return None
    lo = path["low"].values.astype(float)
    cl = path["close"].values.astype(float)
    und = path["underlying_close"].values.astype(float)
    pm = path["minute_et"]
    n = len(cl)
    cummax_cl = np.maximum.accumulate(cl)
    cummin_lo = np.minimum.accumulate(lo)
    eod = ((pm.dt.hour == 15) & (pm.dt.minute >= 55)).values
    ts_idx = int(np.argmax(eod)) if eod.any() else n - 1
    comm = D.COMMISSION_PCT

    tp = entry_mid * (1 + tr)
    sr = tr / rr
    sl = entry_mid * (1 - sr)
    tp_idx = int(np.searchsorted(cummax_cl, tp)) if cummax_cl[-1] >= tp else n
    sl_idx = int(np.searchsorted(-cummin_lo, -sl)) if cummin_lo[-1] <= sl else n
    plain_idx = min(tp_idx, sl_idx, ts_idx)

    def px_at(idx, is_tp_hit):
        if idx >= n:
            return cl[-1]
        if is_tp_hit:
            return tp
        if idx == sl_idx and sl_idx <= ts_idx and sl_idx <= tp_idx:
            return min(sl, cl[idx])
        return cl[idx]

    plain_is_tp = tp_idx <= sl_idx and tp_idx == plain_idx and tp_idx < n
    plain_px = px_at(plain_idx, plain_is_tp)
    plain_pnl = (plain_px - entry_mid) / entry_mid - comm

    # --- wall-stall overlay ---------------------------------------------------
    wall_idx = None
    if wall is not None and (net_sign == "+" or not args.net_pos):
        is_call = direction == "CALL"
        near = (und >= wall - args.tol) if is_call else (und <= wall + args.tol)
        if near.any():
            touch = int(np.argmax(near))
            chk = min(touch + args.stall_mins, n - 1)
            broke = (und[touch:chk + 1] >= wall + args.tol).any() if is_call \
                else (und[touch:chk + 1] <= wall - args.tol).any()
            # only lock in if the option is actually up by --min-profit at the check
            in_profit = cl[:chk + 1].max() >= entry_mid * (1 + args.min_profit)
            if not broke and in_profit:
                wall_idx = chk

    if wall_idx is not None and wall_idx < plain_idx:
        overlay_pnl = (cl[wall_idx] - entry_mid) / entry_mid - comm
        fired = True
    else:
        overlay_pnl = plain_pnl
        fired = False
    return (t["date"], plain_pnl, overlay_pnl, fired)


def run(args):
    flow = D.build_flow_netprem(args.hist) if args.flow_source == "netprem" else D.build_flow_1m(args.lake)
    flow["minute_et"] = D._naive(flow["minute_et"])
    flow["date"] = flow["minute_et"].dt.date
    split = pd.to_datetime(args.split).date() if args.split else None

    rules = [r for r in RULES if r.get("enabled", True)]
    if args.tickers:
        keep = {t.upper() for t in args.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]
    by_tk = defaultdict(list)
    for r in rules:
        by_tk[r["ticker"].upper()].append(r)

    print("=" * 96)
    print(f"  WALL-STALL EXIT OVERLAY   ({len(rules)} rules"
          + (f"   IS < {split} <= OOS" if split else "   full period") + ")")
    print(f"  wall @ entry minute   stall {args.stall_mins}m   tol ${args.tol}"
          f"   {'+gamma days only' if args.net_pos else 'all gamma signs'}")
    print("=" * 96)

    pooled = defaultdict(list)   # slice -> [(plain, overlay, fired)]
    for tk, tk_rules in by_tk.items():
        wall_cache: dict = {}

        def get_wall(d, mod):
            key = (d, mod // 15)
            if key not in wall_cache:
                dd = wall_load_day(tk, d, args.wall_max_dte, 0)
                if dd is None:
                    wall_cache[key] = (None, None, None)
                else:
                    pr = profile_at(dd, mod, False)
                    if pr is None:
                        wall_cache[key] = (None, None, None)
                    else:
                        spot0, bs, _, _ = pr
                        w = walls(spot0, bs)
                        wall_cache[key] = w if w is not None else (None, None, None)
            return wall_cache[key]

        trigs = D.triggers_for(flow, tk)
        tbars = D._ticker_bars(tk) if trigs else None
        if tbars is None or tbars.empty:
            _, tbars = D._screen_build_one(args.lake, tk)
        if tbars is None or tbars.empty:
            print(f"  {tk}: no option bars -- skipped")
            continue
        bars_by_cid = {cid: g.sort_values("minute_et") for cid, g in tbars.groupby("option_chain_id")}
        bars_by_date = {d: g for d, g in tbars.groupby("date")}

        per_rule = defaultdict(list)
        for r, t, thr in _rule_triggers(tk, tk_rules, flow, args):
            if t["abs_flow"] < thr:
                continue
            mod = pd.Timestamp(t["ts"]).hour * 60 + pd.Timestamp(t["ts"]).minute
            cw, pw, net = get_wall(t["date"], max(mod, 630))
            wall = cw if r["direction"].upper() == "CALL" else pw
            for dte in r.get("dte", [0, 1]):
                res = _sim(t, r["direction"].upper(), dte,
                           float(r["target_roe"]), float(r["rr"]),
                           bars_by_date, bars_by_cid, wall, net, args)
                if res is None:
                    continue
                d, pl, ov, fired = res
                per_rule[r["name"]].append((d, pl, ov, fired))
                sl_name = "OOS" if (split and d >= split) else ("IS" if split else "ALL")
                pooled[sl_name].append((pl, ov, fired))

        for name, rows in per_rule.items():
            _report_rule(name, rows, split)

    print("\n" + "=" * 96)
    print("  POOLED (all rules)")
    print("=" * 96)
    for sl_name, rows in pooled.items():
        _report_pool(sl_name, rows)


def _fmt(rows):
    pl = np.array([x[0] for x in rows]); ov = np.array([x[1] for x in rows])
    return (f"n={len(rows):>4}  plain exp {pl.mean()*100:>+6.1f}% win {(pl>0.02).mean()*100:>4.1f}%"
            f"   overlay exp {ov.mean()*100:>+6.1f}% win {(ov>0.02).mean()*100:>4.1f}%")


def _report_rule(name, rows, split):
    print(f"\n  {name}")
    if split:
        for sl in ("IS", "OOS"):
            sub = [(p, o, f) for d, p, o, f in rows if (d >= split) == (sl == "OOS")]
            if sub:
                print(f"    {sl:4} {_fmt(sub)}")
    else:
        print(f"    ALL  {_fmt([(p, o, f) for _, p, o, f in rows])}")
    fired = [(p, o) for _, p, o, f in rows if f]
    if fired:
        fp = np.array([x[0] for x in fired]); fo = np.array([x[1] for x in fired])
        saved = int((fo > fp + 1e-9).sum()); cost = int((fo < fp - 1e-9).sum())
        print(f"    overlay fired on {len(fired)}/{len(rows)}  "
              f"| on those: plain {fp.mean()*100:>+6.1f}% -> overlay {fo.mean()*100:>+6.1f}%  "
              f"(wall held / saved: {saved}, broke after exit / cost: {cost})")


def _report_pool(sl_name, rows):
    print(f"\n  [{sl_name}]  {_fmt(rows)}")
    fired = [(p, o) for p, o, f in rows if f]
    if fired:
        fp = np.array([x[0] for x in fired]); fo = np.array([x[1] for x in fired])
        d = (fo - fp)
        print(f"    fired {len(fired)}/{len(rows)} ({100*len(fired)/len(rows):.0f}%)  "
              f"| fired-subset  plain {fp.mean()*100:+.1f}% -> overlay {fo.mean()*100:+.1f}%  "
              f"(net {d.mean()*100:+.2f}pp/trade over the whole book: {d.sum()/len(rows)*100:+.2f}pp)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", help="YYYY-MM-DD walk-forward split")
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--stall-mins", type=int, default=15, help="minutes at the wall w/o a break-through -> exit (default 15)")
    ap.add_argument("--tol", type=float, default=0.5, help="$ distance to count price as 'at the wall' (default 0.5)")
    ap.add_argument("--min-profit", type=float, default=0.0,
                    help="only fire the overlay if the option is up >= this fraction at the wall (e.g. 0.3). Default 0 = any")
    ap.add_argument("--net-pos", dest="net_pos", action="store_true", default=True,
                    help="only apply the overlay on net-positive-gamma days (default on)")
    ap.add_argument("--no-net-pos", dest="net_pos", action="store_false",
                    help="apply the overlay regardless of gamma sign")
    ap.add_argument("--wall-max-dte", type=int, default=2, help="max DTE for the wall profile (default 2)")
    ap.add_argument("--flow-source", choices=("silver", "netprem"), default="netprem")
    ap.add_argument("--flow-window", type=int, default=60)
    ap.add_argument("--hist", default="historical")
    ap.add_argument("--lake", default="lake/silver/option-contracts-1m")
    args = ap.parse_args()
    run(args)
