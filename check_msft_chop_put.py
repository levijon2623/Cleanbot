# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_msft_chop_put.py
======================
Dedicated validation of the ADX/DMI candidate from check_adx_dmi.py:

  MSFT CHOP PUT  +  "intraday DMI opposes the put" (i.e. +DI > -DI: price
  grinding UP on 15-min bars when the put-flow trigger fires -> fade it).
  check_adx_dmi first pass: IS +12.2->+37.7% / OOS +10.7->+24.5% / win .57->.83
  (n 106/237).

Checks whether that survives:
  * BAR-LENGTH robustness   (10 / 15 / 20 / 30-min ADX bars -- is it only 15?)
  * di_spread MAGNITUDE     (hard sign-flip vs "clearly trending up")
  * the DAILY-DMI version   (no live intraday calc needed -- weaker in pass 1)
  * both filters stacked
  * hours / the CHOP regime already capturing it
  * 6 rolling calendar slices + maxLL + n-retention  (the fragility check)
  * BOOTSTRAP NULL          (gated OOS vs p95 of a same-size random subsample)

Deployed spec: regime CHOP, min_flow_pct 65, dte [0,1], hours 9-14, tr 1.0, rr 1.0.

Usage:  python check_msft_chop_put.py [--boot 1000]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_adx_dmi import _daily_adx, _intraday_adx, _intra_asof
from check_config_walkforward import _flow_for, _slice_idx
from check_flow_zscore import _eod_mod_for

HIST = "historical"
TK = "MSFT"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD = 15 * 60 + 55


def _stat(pnls, base_n=None):
    if len(pnls) < 10:
        return f"n={len(pnls):>4}  (thin)"
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
    slices = " ".join(f"S{j+1}{np.mean(b) * 100:+.0f}" if len(b) >= 5 else f"S{j+1}··"
                      for j, b in enumerate(sl))
    ret = f" ret {100 * len(v) / base_n:.0f}%" if base_n else ""
    return (f"n={len(v):>4}{ret}  IS {np.mean(i) * 100 if i else float('nan'):>+6.1f}%(n{len(i):>3})  "
            f"OOS {np.mean(o) * 100 if o else float('nan'):>+6.1f}%(n{len(o):>3})  win {np.mean(v > 0):.2f}  "
            f"maxLL {m:>2}  pop {npop}/6  [{slices}]")


def _boot(rows, gated_oos_mean, n_gated, k=1000):
    oos_all = [p for d, p, *_ in rows if d >= SPLIT]
    if len(oos_all) < n_gated + 5 or n_gated < 10:
        return None
    rng = np.random.default_rng(0)
    draws = [np.mean(rng.choice(oos_all, size=n_gated, replace=False)) * 100 for _ in range(k)]
    p95 = float(np.percentile(draws, 95))
    rank = float((np.array(draws) < gated_oos_mean).mean() * 100)
    return np.mean(draws), p95, rank


def run(a):
    import directional_flow_backtester as D

    flow = _flow_for(D, [TK])
    gex = D.load_gex(HIST, TK); vol = D.load_volume_regime(HIST, TK); trd = D.load_trend_regime(HIST, TK)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}

    dadx = _daily_adx(TK)
    # close_only=True -> H/L of each intraday bar built from 1-min CLOSES, exactly
    # what bot_runner reconstructs live from per-minute spot (see _dmi_confirm_ok)
    iadx = {bl: _intraday_adx(TK, bl, close_only=True) for bl in (10, 15, 20, 30)}

    trigs = D.triggers_for(flow, TK)
    tb = D._ticker_bars(TK)
    if tb is None or tb.empty:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", TK)
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}

    rule = {"ticker": TK, "direction": "PUT", "hours": [9, 10, 11, 12, 13, 14],
            "dte": [0, 1], "regime": "CHOP", "min_flow_pct": 65, "target_roe": 1.0, "rr": 1.0}
    D.annotate_flow_pct(trigs, 60)
    matched = D._rule_matched_trigs(rule, trigs, gex, vol, trd, amp, reg_src)

    rows = []   # (date, pnl, hour, daily_tuple, {bar: intra_tuple})
    for t, _thr in matched:
        d, ts = t["date"], t["ts"]
        da = dadx.get(d)
        ia = {bl: _intra_asof(iadx[bl].get(d, []), ts) for bl in iadx}
        for p in D._option_paths(t, "PUT", [0, 1], bbd, bbc):
            rows.append((d, D._bracket_pnl(*p, 1.0, 1.0, None, EOD), t["hour"], da, ia))

    base = [(d, p) for d, p, *_ in rows]
    print("=" * 112)
    print(f"  MSFT CHOP PUT  +  ADX/DMI filters   ({len(base)} option-trades, deployed spec)   split {SPLIT}")
    print("=" * 112)
    print(f"  {'BASELINE':34} {_stat(base)}")
    bn = len(base)

    def sub(pred):
        return [(d, p) for d, p, hr, da, ia in rows if pred(hr, da, ia)]

    print("\n  -- intraday DMI OPPOSES the put (+DI > -DI, price grinding up) : bar-length robustness --")
    for bl in (10, 15, 20, 30):
        s = sub(lambda hr, da, ia, b=bl: ia[b] is not None and ia[b][0] > ia[b][1])
        print(f"  {('intra'+str(bl)+'m  +DI>-DI'):34} {_stat(s, bn)}")
    print("  -- intraday DMI AGREES (should be the WORSE half) --")
    for bl in (15,):
        s = sub(lambda hr, da, ia, b=bl: ia[b] is not None and ia[b][0] <= ia[b][1])
        print(f"  {('intra'+str(bl)+'m  -DI>=+DI'):34} {_stat(s, bn)}")

    print("\n  -- intraday di_spread MAGNITUDE (15m; +DI - -DI) --")
    for lo, hi, lbl in [(0, 5, "0..5 (barely up)"), (5, 15, "5..15"), (15, 999, ">15 (clearly up)"),
                        (-999, 0, "< 0 (down)")]:
        s = sub(lambda hr, da, ia, L=lo, H=hi: ia[15] is not None and L <= (ia[15][0] - ia[15][1]) < H)
        print(f"  {('spread '+lbl):34} {_stat(s, bn)}")

    print("\n  -- intraday ADX level (15m) --")
    for lo, hi, lbl in [(0, 20, "ADX < 20 (chop)"), (20, 30, "ADX 20-30"), (30, 999, "ADX >= 30")]:
        s = sub(lambda hr, da, ia, L=lo, H=hi: ia[15] is not None and L <= ia[15][2] < H)
        print(f"  {('intra15m  '+lbl):34} {_stat(s, bn)}")

    print("\n  -- DAILY DMI (no live intraday calc needed) --")
    s_op = sub(lambda hr, da, ia: da is not None and da[3] > 0)          # +DI>-DI daily (opposes put)
    s_ag = sub(lambda hr, da, ia: da is not None and da[3] < 0)
    print(f"  {'daily +DI>-DI (opposes put)':34} {_stat(s_op, bn)}")
    print(f"  {'daily -DI>+DI (agrees)':34} {_stat(s_ag, bn)}")
    s_daily_op_adx = sub(lambda hr, da, ia: da is not None and da[3] > 0 and da[2] < 25)
    print(f"  {'daily opposes & ADX<25':34} {_stat(s_daily_op_adx, bn)}")

    print("\n  -- STACKED: daily opposes AND intra15m opposes --")
    s_both = sub(lambda hr, da, ia: da is not None and da[3] > 0 and ia[15] is not None and ia[15][0] > ia[15][1])
    print(f"  {'daily & intra15m both oppose':34} {_stat(s_both, bn)}")

    # winner = intra15m opposes; refine
    win_pred = lambda hr, da, ia: ia[15] is not None and ia[15][0] > ia[15][1]
    W = sub(win_pred)
    print("\n  -- refine the intra15m-opposes set --")
    for lo in (9, 10, 11, 12):
        s = [(d, p) for d, p, hr, da, ia in rows if win_pred(hr, da, ia) and lo <= hr < 15]
        print(f"  {('hours '+str(lo)+'-14'):34} {_stat(s, bn)}")
    for g, lbl in (("UPTREND", "trend=UPTREND"), ("CHOP", "trend=CHOP"), ("DOWNTREND", "trend=DOWNTREND")):
        s = [(d, p) for d, p, hr, da, ia in rows if win_pred(hr, da, ia) and trd.get(d) == g]
        if len(s) >= 8:
            print(f"  {lbl:34} {_stat(s, bn)}")

    # bootstrap null on the winner (OOS)
    w_oos = [p for d, p in W if d >= SPLIT]
    if len(w_oos) >= 10:
        r = _boot(rows, np.mean(w_oos) * 100, len(w_oos), k=a.boot)
        if r:
            mean, p95, rank = r
            verdict = "** BEATS null" if np.mean(w_oos) * 100 > p95 else "(within noise)"
            print(f"\n  bootstrap null (intra15m opposes, OOS n={len(w_oos)}): "
                  f"gated {np.mean(w_oos) * 100:+.1f}%  vs random-subsample mean {mean:+.1f}% / p95 {p95:+.1f}%  "
                  f"({rank:.0f}th pct)  {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    run(a)
