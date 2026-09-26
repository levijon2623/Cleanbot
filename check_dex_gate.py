# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_dex_gate.py
=================
Follow-up on check_dex.py --test rules: `dex_pct` (within-ticker trailing-252d
percentile of net_dex, prior-day) as a per-rule gate.  Adds rigor:

  * EXACT deployed spec per rule -- flow_zscore vs min_flow_pct, flow_window_days,
    eod_flatten, time_stop_mins, amt_open, ema_confirm
  * threshold sweep  (dex_pct < {.15 .2 .25 .3 .35}  and  > {.65 .8})
  * BOOTSTRAP NULL -- is the gated OOS expectancy above the 95th pct of what a
    random subsample of the same size gives?  (i.e. is dex_pct doing anything
    beyond just shrinking n?)
  * endogeneity peek -- vol-regime day-mix in the dex-low vs dex-high subset
  * blended: apply one cut to every rule

Usage:
  python check_dex_gate.py [--cut 0.25] [--boot 500]
  python check_dex_gate.py --rule-file disabled_candidate_rules.json   # revive-check disabled rules
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from check_dex import _dex, _feat
from check_flow_zscore import annotate_flow_z, _z_matched, _eod_mod_for

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
LO_CUTS = [0.15, 0.20, 0.25, 0.30, 0.35]
HI_CUTS = [0.65, 0.80]


def _exp(pnls, half=None):
    v = [p for d, p in pnls if half is None or (half == "IS" and d < SPLIT) or (half == "OOS" and d >= SPLIT)]
    return (np.mean(v) * 100) if v else float("nan"), len(v)


def _row(lbl, pnls):
    ei, ni = _exp(pnls, "IS"); eo, no = _exp(pnls, "OOS")
    w = np.mean([p > 0 for _, p in pnls]) if pnls else float("nan")
    return f"    {lbl:16} n={len(pnls):>4}  IS {ei:>+6.1f}%(n{ni:>3})  OOS {eo:>+6.1f}%(n{no:>3})  win {w:.2f}"


def _load_rules(a):
    """config.RULES (enabled only) by default, or --rule-file <json> (candidate/
    disabled rules -- flow_pct -> min_flow_pct, enabled defaults True). Pass
    'config-disabled' to pull the enabled=False entries of config.RULES."""
    if a.rule_file == "config-disabled":
        from config import RULES
        return [r for r in RULES if not r.get("enabled", True)]
    if a.rule_file:
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
    try:
        from amt_profile import amt_open_map, amt_ok
    except Exception:
        amt_open_map = amt_ok = None

    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date
    rules = _load_rules(a)
    vol_all = {}
    blended = {"base": [], "gated": []}

    print("=" * 104)
    print(f"  DEX-PCT GATE follow-up   (deployed spec, threshold sweep, bootstrap null; blended cut = {a.cut})")
    print("=" * 104)

    for r in rules:
        tk = r["ticker"]; direction = r["direction"].upper()
        dx = _dex(tk)
        if dx is None:
            print(f"\n  {r['name']}: no GEX{tk}.parquet"); continue
        f = _feat(dx, False)
        dpct = {d.date(): v for d, v in f["dex_pct"].items() if pd.notna(v)}

        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        vol_all[tk] = vol
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        ema_stack = D.load_ema_stack(HIST, tk, int(r["ema_confirm"])) if r.get("ema_confirm") else None
        amt = amt_open_map(tk) if (r.get("amt_open") and amt_open_map) else {}

        trigs = D.triggers_for(flow, tk)
        if not trigs:
            tkf, _ = D._screen_build_one("lake/silver/option-contracts-1m", tk)
            trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}

        zs = r.get("flow_zscore")
        if zs:
            annotate_flow_z(trigs, int(zs.get("window_days") or 60))
            matched = _z_matched(D, r, trigs, gex, vol, trd, amp, reg_src, float(zs["k"]))
            matched = [(t, None) for t in matched]
        else:
            D.annotate_flow_pct(trigs, int(r.get("flow_window_days") or 60))
            matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)

        tr, rr = float(r["target_roe"]), float(r["rr"])
        tstop = r.get("time_stop_mins"); eod = _eod_mod_for(r)
        rows = []
        for t, _thr in matched:
            d, ts = t["date"], t["ts"]
            if r.get("amt_open") and amt_ok and not amt_ok(r["amt_open"], amt.get(d)):
                continue
            if ema_stack is not None:
                st = D.ema_state_at(ema_stack, ts)
                if st is not None and st != ("BULL" if direction == "CALL" else "BEAR"):
                    continue
            pc = dpct.get(d)
            for p in D._option_paths(t, direction, r.get("dte", [0, 1]), bbd, bbc):
                rows.append((d, D._bracket_pnl(*p, tr, rr, tstop, eod), pc))
        base = [(d, p) for d, p, _ in rows]
        if len(base) < 20:
            print(f"\n  {r['name']}: only {len(base)} trades"); continue

        print(f"\n  {r['name']}  ({tk} {direction})")
        print(_row("DEPLOYED", base))
        best = None
        for c in LO_CUTS:
            g = [(d, p) for d, p, pc in rows if pc is not None and pc < c]
            if len(g) >= 15:
                print(_row(f"dex_pct < {c}", g))
                eo, no = _exp(g, "OOS"); ei, _ = _exp(g, "IS")
                if no >= 15 and pd.notna(ei) and pd.notna(eo) and (best is None or min(ei, eo) > best[0]):
                    best = (min(ei, eo), c, g, ei, eo, no)
        for c in HI_CUTS:
            g = [(d, p) for d, p, pc in rows if pc is not None and pc > c]
            if len(g) >= 15:
                print(_row(f"dex_pct > {c}", g))

        # bootstrap null on the best lo-cut bucket
        if best:
            _, c, g, ei, eo, nk = best
            bd = [(d, p, pc) for d, p, pc in rows]
            oos_all = [p for d, p, _ in bd if d >= SPLIT]
            draws = []
            rng = np.random.default_rng(0)
            for _ in range(a.boot):
                draws.append(np.mean(rng.choice(oos_all, size=min(nk, len(oos_all)), replace=False)) * 100)
            p95 = np.percentile(draws, 95)
            rank = (np.array(draws) < eo).mean() * 100
            print(f"    -> best lo-cut {c}: OOS {eo:+.1f}%  vs random-subsample null "
                  f"mean {np.mean(draws):+.1f}% / p95 {p95:+.1f}%  (gated at {rank:.0f}th pct)"
                  f"  {'** beats null' if eo > p95 else '(within noise)'}")
            # endogeneity: vol-regime mix, dex-low vs dex-high
            lo_days = {d for d, _, pc in rows if pc is not None and pc < c}
            hi_days = {d for d, _, pc in rows if pc is not None and pc >= c}
            def mix(days):
                cnt = {"LOWVOL": 0, "NORMVOL": 0, "HIVOL": 0}
                for d in days:
                    cnt[vol.get(d, "NORMVOL")] = cnt.get(vol.get(d, "NORMVOL"), 0) + 1
                tot = max(sum(cnt.values()), 1)
                return " ".join(f"{k[:3]} {100*v/tot:.0f}%" for k, v in cnt.items())
            print(f"       vol-regime mix  dex<{c}: {mix(lo_days)}   |   dex>={c}: {mix(hi_days)}")

            blended["base"] += base
            blended["gated"] += g + [(d, p) for d, p, pc in rows if not (pc is not None and pc < a.cut)]
        else:
            blended["base"] += base
            blended["gated"] += base

    print("\n" + "-" * 60)
    eb, _ = _exp(blended["base"], "OOS"); eg, _ = _exp(blended["gated"], "OOS")
    ib, _ = _exp(blended["base"], "IS"); ig, _ = _exp(blended["gated"], "IS")
    print(f"  BLENDED (per-rule best lo-cut applied; unhelped rules unchanged)")
    print(f"    base   IS {ib:+.1f}%  OOS {eb:+.1f}%")
    print(f"    gated  IS {ig:+.1f}%  OOS {eg:+.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut", type=float, default=0.25)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--rule-file", default=None,
                    help="candidate/disabled rules JSON (or 'config-disabled' for config.RULES enabled=False)")
    a = ap.parse_args()
    run(a)
