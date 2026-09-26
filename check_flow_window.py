# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_flow_window.py
=====================

Which trailing-window length should the flow-PERCENTILE threshold roll over?

Two numbers currently govern this, neither ever tested against alternatives:
  - directional_flow_backtester.FLOW_PCT_WINDOW_DAYS = 60 calendar days -- the
    window annotate_flow_pct() used to calibrate every deployed RULES threshold
    (the walk-forward IS/OOS split that produced config.py's min_flow_pct values).
  - bot_runner.py's live self-calibration (_live_flow_threshold) rolls a
    DIFFERENT window -- a hardcoded 90 calendar days -- once its own crossover
    history spans >= FLOW_HISTORY_MIN_DAYS=20 distinct trading days. That "20"
    was a bugfix minimum-sample gate (session 12, chosen to survive one choppy
    session flooding the pool), not a tuned lookback; the 90d window length next
    to it has never been swept either. So there's a live/backtest MISMATCH (60d
    vs 90d) on top of an unvalidated absolute choice.

This grids window_days and reruns every ENABLED config.RULES entry's full option
P&L (same _option_paths / _bracket_pnl machinery as the hold-time sweep) through
each window, IS/OOS -- percentile thresholds only change WHICH triggers clear the
bar, so option paths are computed once per candidate trigger and reused across
every window value tested.

Usage:
  python check_flow_window.py --split 2025-08-21
  python check_flow_window.py --tickers SPY QQQ --split 2025-08-21
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

HIST = "historical"
WINDOW_GRID = (10, 15, 20, 30, 45, 60, 90, 120, 180, 250)
DEPLOYED_BACKTEST_WINDOW = 60   # FLOW_PCT_WINDOW_DAYS -- calibrated every rule's min_flow_pct
LIVE_BOT_WINDOW = 90            # bot_runner._live_flow_threshold's hardcoded trailing pool


def _eod_mod_for(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _load_rules(a):
    """config.RULES (enabled only) by default, or --rule-file <json> (candidate/
    disabled rules, e.g. disabled_candidate_rules.json)."""
    if a.rule_file:
        import json
        with open(a.rule_file) as f:
            rules = json.load(f)
        for r in rules:
            r.setdefault("min_flow_pct", r.get("flow_pct"))
            r.setdefault("enabled", True)
        return rules
    from config import RULES
    return [r for r in RULES if r.get("enabled", True)]


def run(a):
    import directional_flow_backtester as D
    from amt_profile import amt_open_map, amt_ok

    split = pd.Timestamp(a.split).date()
    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date
    rules = _load_rules(a)
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]

    by_ticker = {}
    for r in rules:
        by_ticker.setdefault(r["ticker"], []).append(r)

    print("=" * 108)
    print("  FLOW-PERCENTILE TRAILING-WINDOW SWEEP   (window chosen days -> IS/OOS $ expectancy per rule)")
    print(f"  * = deployed backtest window ({DEPLOYED_BACKTEST_WINDOW}d)   ^ = live bot's actual window ({LIVE_BOT_WINDOW}d)")
    print("=" * 108)

    grand = {w: {"is": [], "oos": []} for w in WINDOW_GRID}
    for tk, tk_rules in by_ticker.items():
        sp = None
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        amt = {}
        for r in tk_rules:
            if r.get("amt_open"):
                amt = amt_open_map(tk)
                break
        ema_stacks = {}
        for r in tk_rules:
            if r.get("ema_confirm") and int(r["ema_confirm"]) not in ema_stacks:
                ema_stacks[int(r["ema_confirm"])] = D.load_ema_stack(HIST, tk, int(r["ema_confirm"]))

        trigs = D.triggers_for(flow, tk)
        if not trigs:
            tkf, _ = D._screen_build_one("lake/silver/option-contracts-1m", tk)
            trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        if not trigs or tb is None or tb.empty:
            print(f"\n  {tk}: no data, skipping rules {[r['name'] for r in tk_rules]}"); continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}

        path_cache = {}
        def paths_for(t, direction, dtes):
            key = id(t)
            if key not in path_cache:
                path_cache[key] = D._option_paths(t, direction, dtes, bbd, bbc)
            return path_cache[key]

        for r in tk_rules:
            base_tr, rr = float(r["target_roe"]), float(r["rr"])
            tstop = r.get("time_stop_mins")
            eod_mod = _eod_mod_for(r)
            want_amt = r.get("amt_open")
            want_bull = r["direction"].upper() == "CALL"
            ema_stack = ema_stacks.get(int(r["ema_confirm"])) if r.get("ema_confirm") else None

            print(f"\n  {r['name']}  ({tk} {r['direction']}, target_roe={base_tr} rr={rr})")
            print(f"    {'window':>8}  {'IS n':>6}  {'IS exp%':>8}  {'OOS n':>6}  {'OOS exp%':>9}")
            for w in WINDOW_GRID:
                D.annotate_flow_pct(trigs, window_days=w)
                matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
                pnls = []
                for t, thr in matched:
                    d, ts = t["date"], t["ts"]
                    if want_amt and not amt_ok(want_amt, amt.get(d)):
                        continue
                    if ema_stack is not None:
                        st = D.ema_state_at(ema_stack, ts)
                        if st is not None and st != ("BULL" if want_bull else "BEAR"):
                            continue
                    for p in paths_for(t, r["direction"].upper(), r.get("dte", [0, 1])):
                        pnls.append((d, D._bracket_pnl(*p, base_tr, rr, tstop, eod_mod)))
                isp = [p for d, p in pnls if d < split]
                oos = [p for d, p in pnls if d >= split]
                tag = ("*" if w == DEPLOYED_BACKTEST_WINDOW else " ") + ("^" if w == LIVE_BOT_WINDOW else " ")
                ie = f"{np.mean(isp)*100:>+7.1f}%" if isp else "    --  "
                oe = f"{np.mean(oos)*100:>+8.1f}%" if oos else "     --  "
                print(f"    {tag}{w:>5}d  {len(isp):>6}  {ie}  {len(oos):>6}  {oe}")
                if isp:
                    grand[w]["is"] += isp
                if oos:
                    grand[w]["oos"] += oos

    print("\n" + "=" * 108)
    print("  BLENDED ACROSS ALL RULES (equal per-trade weight)")
    print("=" * 108)
    print(f"    {'window':>8}  {'IS n':>6}  {'IS exp%':>8}  {'OOS n':>6}  {'OOS exp%':>9}")
    for w in WINDOW_GRID:
        isp, oos = grand[w]["is"], grand[w]["oos"]
        tag = ("*" if w == DEPLOYED_BACKTEST_WINDOW else " ") + ("^" if w == LIVE_BOT_WINDOW else " ")
        ie = f"{np.mean(isp)*100:>+7.1f}%" if isp else "    --  "
        oe = f"{np.mean(oos)*100:>+8.1f}%" if oos else "     --  "
        print(f"    {tag}{w:>5}d  {len(isp):>6}  {ie}  {len(oos):>6}  {oe}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--split", default="2025-08-21")
    ap.add_argument("--rule-file", default=None, help="candidate/disabled rules JSON instead of config.RULES")
    a = ap.parse_args()
    run(a)
