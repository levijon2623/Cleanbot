# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_sequential_fills.py
=========================
THE BACKTEST TAKES TRADES THE LIVE BOT CANNOT.

bot_runner.py:1288  ->  `if ticker in self.active_snipes: continue`
The live bot holds AT MOST ONE position per ticker and ignores every further
trigger until that position closes.  The backtest (`_rule_matched_trigs` ->
`_option_paths` -> `_bracket_pnl`) evaluates EVERY matched trigger independently,
so a day with 23 triggers becomes 23 "trades" -- all on the same contract, all
open simultaneously, nearly all exiting at the same EOD price (see
check_day_anatomy.py: WMT 2024-12-26 = 23 trades / 1-2 strikes / overlap 23 /
one common exit).  Those are one bet counted 23 times.

This replays each rule the way the bot actually runs it:
  walk matched triggers in chronological order; if flat, ENTER and compute the
  real exit (TP / SL / EOD); mark the ticker busy until that exit minute; skip
  every trigger before then; allow re-entry after.

Reports AS-SCREENED vs SEQUENTIAL per rule: n, days, trades/day, IS/OOS, win,
maxLL -- i.e. how much of the deployed book's backtest was double-counting.

Usage:
  python check_sequential_fills.py                 # all enabled config.RULES
  python check_sequential_fills.py --tickers LULU TSM
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _eod_mod(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _exit_of(entry_mid, cl, lo, mod, held, tr, rr, eod_m):
    """_bracket_pnl, but also return the exit MINUTE so we can block re-entry."""
    n = len(cl)
    cummax_cl = np.maximum.accumulate(cl)
    cummin_lo = np.minimum.accumulate(lo)
    eod = mod >= eod_m
    ts_idx = int(np.argmax(eod)) if eod.any() else n - 1
    tp = entry_mid * (1 + tr)
    sl = entry_mid * (1 - tr / rr)
    tp_idx = int(np.searchsorted(cummax_cl, tp)) if cummax_cl[-1] >= tp else n
    sl_idx = int(np.searchsorted(-cummin_lo, -sl)) if cummin_lo[-1] <= sl else n
    ei = min(tp_idx, sl_idx, ts_idx)
    if ei >= n:
        px, i = cl[-1], n - 1
    elif tp_idx <= sl_idx and tp_idx == ei:
        px, i = tp, tp_idx
    elif sl_idx == ei:
        px, i = min(sl, cl[sl_idx]), sl_idx
    else:
        px, i = cl[ei], ei
    import directional_flow_backtester as D
    pnl = (px - entry_mid) / entry_mid - D.COMMISSION_PCT
    return float(pnl), int(mod[i])


def _stat(pnls):
    if not pnls:
        return "n=   0"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    c = m = 0
    for _, p in sorted(pnls):
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    npop = sum(1 for b in sl if len(b) >= 5)
    days = len({d for d, _ in pnls})
    return (f"n={len(v):>4} d={days:>3} t/d {len(v)/days:>4.1f}  "
            f"IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  "
            f"win {np.mean(v > 0):.2f}  maxLL {m:>2}  pop {npop}/6")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]

    print("=" * 118)
    print("  AS-SCREENED (every trigger)  vs  SEQUENTIAL (one position per ticker, "
          "bot_runner.py:1288)")
    print("=" * 118)

    tot_a, tot_s = [], []
    for r in rules:
        tk = r["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, r.get("flow_window_days", 60))
        try:
            tb = D._ticker_bars(tk)
        except Exception:
            tb = None
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        if tb is None or tb.empty:
            print(f"  {r['name']:24} no bars"); continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        amt = amt_open_map(tk) if r.get("amt_open") else {}

        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
        if r.get("amt_open"):
            matched = [(t, th) for t, th in matched if amt_ok(r["amt_open"], amt.get(t["date"]))]
        matched.sort(key=lambda x: pd.Timestamp(x[0]["ts"]))

        tr_, rr_, em = float(r["target_roe"]), float(r["rr"]), _eod_mod(r)
        dtes = r.get("dte", [0, 1])
        allp, seqp = [], []
        busy_day, busy_until = None, -1
        for t, _th in matched:
            d = t["date"]
            m = pd.Timestamp(t["ts"]).hour * 60 + pd.Timestamp(t["ts"]).minute
            paths = D._option_paths(t, r["direction"], dtes, bbd, bbc)
            if not paths:
                continue
            for p in paths:                       # as-screened: every dte path
                allp.append((d, D._bracket_pnl(*p, tr_, rr_, None, em)))
            # sequential: one position, first available dte (the bot's fallback order)
            if busy_day != d:
                busy_day, busy_until = d, -1
            if m < busy_until:
                continue
            pnl, xm = _exit_of(*paths[0], tr_, rr_, em)
            seqp.append((d, pnl))
            busy_until = xm

        print(f"\n  {r['name']}")
        print(f"    as-screened  {_stat(allp)}")
        print(f"    SEQUENTIAL   {_stat(seqp)}")
        tot_a += allp; tot_s += seqp

    print("\n" + "=" * 118)
    print(f"  BLENDED BOOK\n    as-screened  {_stat(tot_a)}\n    SEQUENTIAL   {_stat(tot_s)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    a = ap.parse_args()
    run(a)
