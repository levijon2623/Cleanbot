# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_open_delay.py
===================
Does waiting N minutes after the 9:30 open before allowing an entry help or
hurt?  The first (and last) few minutes are the noisiest -- widest spreads,
thinnest flow history, gap digestion.

For each deployed-spec rule: drop every trade whose entry minute-of-day is
< 570 + delay (570 = 9:30 ET) for delay in {0 baseline, 1, 3, 5, 10}, and
compare P&L (IS/OOS @ 2025-08-21 + 6 rolling slices + win + maxLL).  Also the
standalone P&L of just the trades in each opening window -- are the early
entries net-negative (delay helps) or net-positive (delay costs edge)?

Usage:
  python check_open_delay.py
  python check_open_delay.py --tickers IWM QQQ SPY
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_flow_zscore import annotate_flow_z, _z_matched, _eod_mod_for
from check_config_walkforward import _flow_for, _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
OPEN_MOD = 570
DELAYS = (0, 1, 3, 5, 10)


def _agg(pnls, base_n=None):
    if len(pnls) < 8:
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
    slc = " ".join(f"S{j+1}{np.mean(b) * 100:+.0f}" if len(b) >= 5 else f"S{j+1}··" for j, b in enumerate(sl))
    ret = f" ({100 * len(v) / base_n:.1f}%)" if base_n else ""
    return (f"n={len(v):>4}{ret}  avg {v.mean() * 100:>+6.1f}%  "
            f"IS {np.mean(i) * 100 if i else float('nan'):>+6.1f}%  OOS {np.mean(o) * 100 if o else float('nan'):>+6.1f}%  "
            f"win {np.mean(v > 0):.2f}  maxLL {m:>2}  [{slc}]")


def _rule_trades(D, r, flow_all, gex, vol, trd, amp, reg_src, amt, amt_ok, ema_stacks, bbd, bbc, iadx, intra, is_am):
    direction = r["direction"].upper()
    want_bull = direction == "CALL"
    dtes = tuple(r.get("dte", [0, 1]))
    tr, rr = float(r["target_roe"]), float(r["rr"])
    tstop = r.get("time_stop_mins"); eod = _eod_mod_for(r)
    trigs = D.triggers_for(flow_all, r["ticker"])
    zs = r.get("flow_zscore")
    if zs:
        annotate_flow_z(trigs, int(zs.get("window_days") or 60))
        matched = [(t, None) for t in _z_matched(D, r, trigs, gex, vol, trd, amp, reg_src, float(zs["k"]))]
    else:
        D.annotate_flow_pct(trigs, int(r.get("flow_window_days") or 60))
        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
    ema_stack = ema_stacks.get(int(r["ema_confirm"])) if r.get("ema_confirm") else None
    want_amt = r.get("amt_open")
    dmi = r.get("dmi_confirm")
    dmi_days = iadx.get(int(dmi.get("tf", 15))) if dmi else None
    out = []
    for t, _thr in matched:
        d, ts = t["date"], t["ts"]
        if r.get("skip_macro_am") and is_am(d):
            continue
        if want_amt and not amt_ok(want_amt, amt.get(d)):
            continue
        if ema_stack is not None:
            st = D.ema_state_at(ema_stack, ts)
            if st is not None and st != ("BULL" if want_bull else "BEAR"):
                continue
        if dmi:
            ia = intra(dmi_days.get(d, []), ts) if dmi_days is not None else None
            if ia is not None:
                agree = (ia[0] > ia[1]) == want_bull
                if (agree if dmi.get("mode") == "agree" else (not agree)) is False:
                    continue
        emod = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
        for p in D._option_paths(t, direction, list(dtes), bbd, bbc):
            out.append((d, D._bracket_pnl(*p, tr, rr, tstop, eod), emod))
    return out


def run(a):
    import directional_flow_backtester as D
    from amt_profile import amt_open_map, amt_ok
    from config import RULES
    from check_adx_dmi import _intraday_adx, _intra_asof
    from macro_calendar import is_macro_am_day

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]
    tickers = sorted({r["ticker"] for r in rules})
    flow_all = _flow_for(D, tickers)

    ALL = []
    print("=" * 116)
    print("  OPEN-DELAY SWEEP   (drop entries before 9:30 + delay)   split 2025-08-21")
    print("=" * 116)

    for tk in tickers:
        tk_rules = [r for r in rules if r["ticker"] == tk]
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        amt = amt_open_map(tk) if any(r.get("amt_open") for r in tk_rules) else {}
        ema_stacks = {int(r["ema_confirm"]): D.load_ema_stack(HIST, tk, int(r["ema_confirm"]))
                      for r in tk_rules if r.get("ema_confirm")}
        dmi_tfs = {int(r["dmi_confirm"].get("tf", 15)) for r in tk_rules if r.get("dmi_confirm")}
        iadx = {tf: _intraday_adx(tk, tf, close_only=True) for tf in dmi_tfs}
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {dd: g for dd, g in tb.groupby("date")}

        for r in tk_rules:
            rows = _rule_trades(D, r, flow_all, gex, vol, trd, amp, reg_src, amt, amt_ok,
                                ema_stacks, bbd, bbc, iadx, _intra_asof, is_macro_am_day)
            if len(rows) < 20:
                print(f"\n  {r['name']}: {len(rows)} trades -- skip"); continue
            ALL.extend(rows)
            base = [(d, p) for d, p, m in rows]
            print(f"\n  {r['name']}  ({tk} {r['direction']})")
            print(f"    {'delay  0m (baseline)':22} {_agg(base)}")
            for dl in DELAYS[1:]:
                kept = [(d, p) for d, p, m in rows if m >= OPEN_MOD + dl]
                print(f"    {('delay '+str(dl)+'m'):22} {_agg(kept, len(base))}")
            # standalone P&L of the opening windows
            for lo, hi in ((0, 1), (1, 3), (3, 5), (5, 10)):
                w = [(d, p) for d, p, m in rows if OPEN_MOD + lo <= m < OPEN_MOD + hi]
                if len(w) >= 5:
                    v = np.array([p for _, p in w])
                    print(f"      [9:{30+lo:02d}-9:{30+hi:02d})  n={len(v):>3}  avg {v.mean() * 100:+6.1f}%  win {np.mean(v > 0):.2f}")

    print("\n" + "=" * 116)
    print("  BLENDED BOOK")
    print("=" * 116)
    base = [(d, p) for d, p, m in ALL]
    print(f"    {'delay  0m (baseline)':22} {_agg(base)}")
    for dl in DELAYS[1:]:
        kept = [(d, p) for d, p, m in ALL if m >= OPEN_MOD + dl]
        print(f"    {('delay '+str(dl)+'m'):22} {_agg(kept, len(base))}")
    print("\n  standalone P&L by opening window (blended):")
    for lo, hi in ((0, 1), (1, 3), (3, 5), (5, 10), (10, 30), (30, 60)):
        w = [p for d, p, m in ALL if OPEN_MOD + lo <= m < OPEN_MOD + hi]
        if w:
            v = np.array(w)
            iv = [p for d, p, m in ALL if OPEN_MOD + lo <= m < OPEN_MOD + hi and d < SPLIT]
            ov = [p for d, p, m in ALL if OPEN_MOD + lo <= m < OPEN_MOD + hi and d >= SPLIT]
            print(f"    9:{30+lo:02d}-{'10:'+str(hi-30).zfill(2) if hi>=30 else '9:'+str(30+hi).zfill(2)}  "
                  f"n={len(v):>4}  avg {v.mean() * 100:+6.1f}%  "
                  f"IS {np.mean(iv) * 100 if iv else float('nan'):+6.1f}%  OOS {np.mean(ov) * 100 if ov else float('nan'):+6.1f}%  "
                  f"win {np.mean(v > 0):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    a = ap.parse_args()
    run(a)
