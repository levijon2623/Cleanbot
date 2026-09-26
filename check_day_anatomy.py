# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_day_anatomy.py
====================
Session 18 found that trade-weighted screen stats are inflated by DAY CLUSTERING
(WMT LOWVOL CALL: 71 "trades" over 8 days; IBIT LOWVOL CALL p50 trade-OOS +43%
-> day-OOS +0.5%).  This answers the follow-up: on a high-trade day, WHAT are
those trades?

For a given (ticker, direction, regime, pct) it dumps, per day, every simulated
trade with:
  trigger time, DTE, contract, strike, moneyness at entry, entry premium,
  hold minutes, exit reason (TP / SL / EOD), P&L
plus per-day summary: #triggers vs #trades (the 0/1-DTE doubling), #distinct
contracts, strike spread, whether trades OVERLAP in time (concurrent risk), and
how correlated the outcomes are (all-win / all-lose days = one bet).

Usage:
  python check_day_anatomy.py WMT CALL --regime LOWVOL --pct 65 --dow thu
  python check_day_anatomy.py IBIT CALL --regime LOWVOL --pct 50 --top 3
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD = 15 * 60 + 55
_DOW = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}


def _exit_detail(entry_mid, cl, lo, mod, held, tr=1.0, rr=1.0, eod_mod=EOD):
    """Mirror _bracket_pnl but also return (reason, hold_min, exit_px)."""
    n = len(cl)
    cummax_cl = np.maximum.accumulate(cl)
    cummin_lo = np.minimum.accumulate(lo)
    eod = mod >= eod_mod
    ts_idx = int(np.argmax(eod)) if eod.any() else n - 1
    tp = entry_mid * (1 + tr)
    sl = entry_mid * (1 - tr / rr)
    tp_idx = int(np.searchsorted(cummax_cl, tp)) if cummax_cl[-1] >= tp else n
    sl_idx = int(np.searchsorted(-cummin_lo, -sl)) if cummin_lo[-1] <= sl else n
    exit_idx = min(tp_idx, sl_idx, ts_idx)
    if exit_idx >= n:
        px, why, i = cl[-1], "END", n - 1
    elif tp_idx <= sl_idx and tp_idx == exit_idx:
        px, why, i = tp, "TP", tp_idx
    elif sl_idx == exit_idx:
        px, why, i = min(sl, cl[sl_idx]), "SL", sl_idx
    else:
        px, why, i = cl[exit_idx], "EOD", exit_idx
    return why, float(held[i]), float(px), float(mod[i])


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    TK, DIR = a.ticker.upper(), a.direction.upper()
    dow_keep = {_DOW[d.lower()[:3]] for d in a.dow} if a.dow else None

    flow = _flow_for(D, [TK])
    gex = D.load_gex(HIST, TK); vol = D.load_volume_regime(HIST, TK); trd = D.load_trend_regime(HIST, TK)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
    trigs = D.triggers_for(flow, TK)
    D.annotate_flow_pct(trigs, 60)
    tb = D._ticker_bars(TK)
    if tb is None or tb.empty:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", TK)
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}
    # contract meta for labelling
    meta = tb.groupby("option_chain_id").agg(strike=("strike", "first"),
                                             expiry=("expiry", "first"),
                                             otype=("option_type", "first"))

    rule = {"ticker": TK, "direction": DIR, "hours": [9, 10, 11, 12, 13, 14],
            "regime": a.regime, "min_flow_pct": a.pct}
    if a.regime is None:
        rule.pop("regime")
    matched = D._rule_matched_trigs(rule, trigs, gex, vol, trd, amp, reg_src)

    byday = collections.defaultdict(list)
    trig_ct = collections.Counter()
    dtes = a.dte if a.dte else [0, 1]
    for t, _thr in matched:
        if dow_keep is not None and t["date"].weekday() not in dow_keep:
            continue
        d, ts = t["date"], t["ts"]
        day = bbd.get(d)
        if day is None:
            continue
        at = day[day["minute_et"] <= ts]
        if at.empty:
            continue
        spot = float(at.iloc[-1]["underlying_close"])
        fired = False
        for dd in dtes:
            cid = D.pick_contract(day, ts, DIR, dd, spot)
            paths = D._option_paths(t, DIR, [dd], bbd, bbc)
            for p in paths:
                why, hold, expx, xmod = _exit_detail(*p)
                pnl = D._bracket_pnl(*p, 1.0, 1.0, None, EOD)
                m = meta.loc[cid] if cid in meta.index else None
                strike = float(m["strike"]) if m is not None else float("nan")
                byday[d].append(dict(
                    ts=pd.Timestamp(ts), mod=pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute,
                    dte=dd, cid=cid, strike=strike, spot=spot,
                    mny=(strike / spot - 1) if np.isfinite(strike) else np.nan,
                    entry=float(p[0]), exit_px=expx, why=why, hold=hold, xmod=xmod, pnl=pnl))
                fired = True
        if fired:
            trig_ct[d] += 1

    if not byday:
        print("no trades"); return
    days = sorted(byday, key=lambda d: -len(byday[d]))
    print("=" * 112)
    print(f"  {TK} {DIR} {a.regime or 'ALL'} p{a.pct}"
          f"{' dow=' + '+'.join(a.dow) if a.dow else ''}   "
          f"{sum(len(v) for v in byday.values())} trades / {len(byday)} days")
    print("=" * 112)

    # per-day summary table
    print(f"\n  {'date':11} {'trg':>3} {'trd':>3} {'cids':>4} {'strikes':>18} {'span_min':>8} "
          f"{'overlap':>7} {'win':>4}  {'day mean':>9}")
    for d in sorted(byday):
        v = byday[d]
        cids = {x["cid"] for x in v}
        ks = sorted({x["strike"] for x in v})
        kstr = f"{ks[0]:g}" if len(ks) == 1 else f"{ks[0]:g}-{ks[-1]:g}({len(ks)})"
        mods = [x["mod"] for x in v]
        span = max(mods) - min(mods)
        # concurrency: max # of simultaneously-open trades
        ev = sorted([(x["mod"], 1) for x in v] + [(x["xmod"], -1) for x in v])
        cur = mx = 0
        for _, s in ev:
            cur += s; mx = max(mx, cur)
        wr = np.mean([x["pnl"] > 0 for x in v])
        print(f"  {str(d):11} {trig_ct[d]:>3} {len(v):>3} {len(cids):>4} {kstr:>18} {span:>8} "
              f"{mx:>7} {wr:>4.2f}  {np.mean([x['pnl'] for x in v])*100:>+8.1f}%")

    # detail for the top-N busiest days
    for d in days[:a.top]:
        v = sorted(byday[d], key=lambda x: (x["mod"], x["dte"]))
        print(f"\n  ---- {d}  ({trig_ct[d]} triggers -> {len(v)} trades, "
              f"{'IS' if d < SPLIT else 'OOS'}) ----")
        print(f"    {'time':>5} {'dte':>3} {'strike':>8} {'mny':>7} {'entry':>6} "
              f"{'exit':>6} {'why':>4} {'hold':>5} {'pnl':>8}   contract")
        for x in v:
            hh, mm = divmod(x["mod"], 60)
            print(f"    {hh:02d}:{mm:02d} {x['dte']:>3} {x['strike']:>8g} {x['mny']*100:>+6.1f}% "
                  f"{x['entry']:>6.2f} {x['exit_px']:>6.2f} {x['why']:>4} {x['hold']:>5.0f} "
                  f"{x['pnl']*100:>+7.1f}%   {x['cid']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker")
    ap.add_argument("direction", choices=("CALL", "PUT"))
    ap.add_argument("--regime", default=None)
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--dow", nargs="+", default=None)
    ap.add_argument("--dte", nargs="+", type=int, default=None)
    ap.add_argument("--top", type=int, default=2, help="detail the N busiest days")
    a = ap.parse_args()
    run(a)
