# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_flow_zscore.py
=====================

The deployed flow-SIZE gate (min_flow_pct) is a trailing-window PERCENTILE
RANK of this ticker's own trigger abs_flow -- non-parametric, only 5 discrete
buckets (50/65/80/90/95), robust to shape but coarse. This tests the other
classic "unusual" construction: a trailing-window Z-SCORE gate --
    abs_flow >= mean(trailing W days) + k * std(trailing W days)
-- lookahead-free exactly like annotate_flow_pct (only days STRICTLY BEFORE
the trigger). Options flow is famously fat-tailed / right-skewed, so a
parametric mean+k*sigma gate can select a meaningfully different trigger set
than a percentile rank even at "the same" nominal selectivity -- this measures
whether that difference is better, worse, or a wash, against each rule's own
deployed percentile baseline.

The EMA(5) crossover that decides WHEN a trigger fires (triggers_for) is left
untouched -- this only replaces HOW BIG a crossover has to be to trade.

Usage:
  python check_flow_zscore.py --split 2025-08-21
  python check_flow_zscore.py --tickers GLD SPY --split 2025-08-21
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

HIST = "historical"
W_GRID = (30, 45, 60, 90)
K_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)


def annotate_flow_z(trigs, window_days):
    """Attach t['mu'], t['sd'] = trailing-window_days mean/std of this ticker's
    trigger abs_flow, using only days STRICTLY BEFORE the trigger -- mirrors
    directional_flow_backtester.annotate_flow_pct's day-boundary logic exactly,
    swapping percentile-of-history for mean/std-of-history."""
    if not trigs:
        return
    order = np.argsort([t["ts"] for t in trigs], kind="stable")
    day_ns = np.array([pd.Timestamp(trigs[i]["ts"]).normalize().value for i in order])
    flows = np.array([trigs[i]["abs_flow"] for i in order], dtype=float)
    win = int(window_days) * 86_400_000_000_000
    first = day_ns[0]
    for pos, i in enumerate(order):
        cur = day_ns[pos]
        if cur - first < win:
            trigs[i]["mu"] = trigs[i]["sd"] = None
            continue
        lo = int(np.searchsorted(day_ns, cur - win, side="left"))
        hi = int(np.searchsorted(day_ns, cur, side="left"))
        hist = flows[lo:hi]
        if len(hist) >= 30:
            trigs[i]["mu"] = float(np.mean(hist))
            trigs[i]["sd"] = float(np.std(hist))
        else:
            trigs[i]["mu"] = trigs[i]["sd"] = None


def _eod_mod_for(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _z_matched(D, r, trigs, gex, vol, trd, amp, reg_src, k):
    direction = r["direction"].upper()
    hours = set(r.get("hours", D.ALL_HOURS))
    reg = r.get("regime")
    amp_min = r.get("amp_min")
    out = []
    for t in trigs:
        if t["dir"] != direction or t["hour"] not in hours or t["hour"] >= 15:
            continue
        d = t["date"]
        if reg is not None:
            val = (gex if reg.endswith("_GEX") else reg_src[reg]).get(d)
            if reg == "NEGATIVE_GEX" and val != "NEGATIVE":
                continue
            if reg == "POSITIVE_GEX" and val != "POSITIVE":
                continue
            if not reg.endswith("_GEX") and val != reg:
                continue
        if amp_min is not None and amp.get(d, -1) < amp_min:
            continue
        mu, sd = t.get("mu"), t.get("sd")
        if mu is None or sd is None or sd <= 0:
            continue
        if t["abs_flow"] >= mu + k * sd:
            out.append(t)
    return out


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
    print("  FLOW Z-SCORE GATE  (abs_flow >= mean + k*std, trailing W days, lookahead-free)")
    print("  cell = OOS exp% (n)   rows = k   cols = W days   [baseline] = deployed percentile gate")
    print("=" * 108)

    for tk, tk_rules in by_ticker.items():
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

            def score(matched_trigs):
                pnls = []
                for t in matched_trigs:
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
                return isp, oos

            # baseline: the rule's own deployed percentile gate
            base_w = int(r.get("flow_window_days") or 60)
            D.annotate_flow_pct(trigs, base_w)
            base_matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
            base_isp, base_oos = score([t for t, _thr in base_matched])

            print(f"\n  {r['name']}  ({tk} {r['direction']}, min_flow_pct={r.get('min_flow_pct')}, "
                  f"pct-window={base_w}d)")
            bie = f"{np.mean(base_isp)*100:+.1f}%" if base_isp else "--"
            boe = f"{np.mean(base_oos)*100:+.1f}%" if base_oos else "--"
            print(f"    [baseline percentile]  IS n={len(base_isp)} {bie}   OOS n={len(base_oos)} {boe}")

            cells = {}   # (k, w) -> oos list
            for w in W_GRID:
                annotate_flow_z(trigs, w)
                for k in K_GRID:
                    matched = _z_matched(D, r, trigs, gex, vol, trd, amp, reg_src, k)
                    _isp, oos = score(matched)
                    cells[(k, w)] = oos

            print("         " + "".join(f"{('W='+str(w)+'d'):>16}" for w in W_GRID))
            for k in K_GRID:
                row = f"    k={k:<4}"
                for w in W_GRID:
                    oos = cells[(k, w)]
                    cell = f"{np.mean(oos)*100:>+6.1f}%(n={len(oos):<4})" if oos else "     --        "
                    row += f"{cell:>16}"
                print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--split", default="2025-08-21")
    ap.add_argument("--rule-file", default=None, help="candidate/disabled rules JSON instead of config.RULES")
    a = ap.parse_args()
    run(a)
