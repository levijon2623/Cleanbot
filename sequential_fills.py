# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
sequential_fills.py
===================
Shared one-position-per-ticker fill model -- the way bot_runner ACTUALLY trades.

`bot_runner.py:1288`:  `if ticker in self.active_snipes: continue`
The live bot holds at most ONE position per ticker and ignores every further
trigger until it closes. The historical backtest path (`_rule_matched_trigs` ->
`_option_paths` -> `_bracket_pnl`) instead scores EVERY matched trigger
independently, so a day with 23 EMA re-crossings became 23 "trades" -- in
reality one contract, all opened concurrently, all exiting at the same EOD
price (check_day_anatomy.py). Session 18 measured the damage: blended book
OOS +15.6% as-screened vs +6.0% sequential.

Use `walk(items, rr, tstop, eod_mod)` anywhere a checker builds per-trigger
P&L, so every validation runs on the fill model that will actually execute.

items: chronologically-ordered list of
    (date, entry_mod, path, target_roe)
where `path` is one `_option_paths` tuple (entry_mid, cl, lo, mod, held) and
`target_roe` is that trigger's (possibly vol_overlay-scaled) TP.

Returns [(date, pnl)] for the trades a single-position bot could actually take.
"""
from __future__ import annotations

import numpy as np


def bracket_with_exit(entry_mid, cl, lo, mod, held, tr, rr, tstop, eod_mod):
    """`directional_flow_backtester._bracket_pnl`, plus the exit MINUTE.

    Kept byte-for-byte equivalent to _bracket_pnl in its P&L so sequential and
    as-screened numbers stay comparable -- the only addition is `exit_mod`,
    which is what lets the caller block re-entry while the position is open."""
    import directional_flow_backtester as D
    n = len(cl)
    cummax_cl = np.maximum.accumulate(cl)
    cummin_lo = np.minimum.accumulate(lo)
    eod = mod >= eod_mod
    ts_idx = int(np.argmax(eod)) if eod.any() else n - 1
    if tstop:
        th = held >= tstop
        if th.any():
            ts_idx = min(ts_idx, int(np.argmax(th)))
    tp = entry_mid * (1 + tr)
    sl = entry_mid * (1 - tr / rr)
    tp_idx = int(np.searchsorted(cummax_cl, tp)) if cummax_cl[-1] >= tp else n
    sl_idx = int(np.searchsorted(-cummin_lo, -sl)) if cummin_lo[-1] <= sl else n
    exit_idx = min(tp_idx, sl_idx, ts_idx)
    if exit_idx >= n:
        exit_px, i = cl[-1], n - 1
    elif tp_idx <= sl_idx and tp_idx == exit_idx:
        exit_px, i = tp, tp_idx
    elif sl_idx == exit_idx:
        exit_px, i = min(sl, cl[sl_idx]), sl_idx
    else:
        exit_px, i = cl[exit_idx], exit_idx
    pnl = (exit_px - entry_mid) / entry_mid - D.COMMISSION_PCT
    return float(pnl), int(mod[i])


def walk(items, rr, tstop, eod_mod):
    """One position per ticker, chronological. Enter when flat; block every
    trigger until the real exit minute; allow re-entry after."""
    out = []
    cur_day, busy_until = None, -1
    for d, m, path, tr in sorted(items, key=lambda x: (x[0], x[1])):
        if d != cur_day:
            cur_day, busy_until = d, -1
        if m < busy_until:
            continue
        pnl, xm = bracket_with_exit(*path, tr, rr, tstop, eod_mod)
        out.append((d, pnl))
        busy_until = xm
    return out
