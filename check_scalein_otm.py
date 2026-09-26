# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_scalein_otm.py
====================
Session 18, step #2 refinement.

check_scalein showed same-strike scale-in DILUTES a good entry: on the gated
days SEQ made +26.7% but scaling in made +21.6%, because the 2nd fill buys the
SAME contract at a worse price. The fix under test: put the adds one strike
FURTHER OTM, which is cheaper -- so the same premium budget buys more contracts
and keeps more directional convexity.

Ladders tested (add #k relative to the first fill's strike K):
  same      K,   K,   K      (the current pile-on)
  ladder    K, K+1, K+2      (each add one step further OTM)
  fixed1    K, K+1, K+1      (all adds one step OTM)
  atm1      K, ATM+1 at the add minute, ... (rolling, re-centred on live spot)
(+1 = further OTM: higher strike for CALLs, lower for PUTs.)

Accounting is BASKET-level, which is the honest way once strikes differ:
  EQUAL-DOLLAR (headline) -- every add deploys the SAME premium as the first
    fill, so a cheaper strike buys proportionally MORE contracts. Total risk =
    n_adds x the per-add budget, and return is on that total. This is the
    apples-to-apples test of "does moving OTM fix the dilution".
  PER-CONTRACT (also shown) -- 1 contract per add.
TP / stack-stop are evaluated on the BASKET's mark vs its total cost.

Usage:
  python check_scalein_otm.py
  python check_scalein_otm.py --max-adds 2 --stop 0.5
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
LADDERS = ("same", "ladder", "fixed1", "atm1")


def _eod_mod(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _series(bars):
    """cid-day frame -> (mod, close, low, mid) numpy."""
    b = bars.sort_values("minute_et")
    mod = (b["minute_et"].dt.hour * 60 + b["minute_et"].dt.minute).to_numpy()
    cl = b["close"].to_numpy(float)
    lo = b["low"].to_numpy(float)
    bid = b["bid_close"].to_numpy(float)
    ask = b["ask_close"].to_numpy(float)
    mid = np.where(bid > 0, (bid + ask) / 2.0, cl)
    return mod, cl, lo, mid


def _pick_strike(chain, first_strike, step, direction):
    """`step` strikes further OTM than first_strike, within this expiry's ladder."""
    ks = chain
    if len(ks) == 0:
        return None
    i = int(np.argmin(np.abs(ks - first_strike)))
    j = i + step if direction == "CALL" else i - step
    if j < 0 or j >= len(ks):
        return None
    return float(ks[j])


def _basket(day_bars, fills, direction, expiry, first_cid, tr, stop, eod_m,
            ladder, max_adds, equal_dollar=True):
    """Simulate a multi-strike scale-in basket. Returns dict or None."""
    sub = day_bars[(day_bars["option_type"] == ("call" if direction == "CALL" else "put"))
                   & (day_bars["expiry"] == expiry)]
    if sub.empty:
        return None
    ks = np.array(sorted(sub["strike"].unique()), float)
    by_k = {float(k): g for k, g in sub.groupby("strike")}
    first_row = day_bars[day_bars["option_chain_id"] == first_cid]
    if first_row.empty:
        return None
    K0 = float(first_row["strike"].iloc[0])

    # choose each add's strike
    picks = []
    for k in range(min(max_adds, len(fills))):
        if k == 0 or ladder == "same":
            strike = K0
        elif ladder == "ladder":
            strike = _pick_strike(ks, K0, k, direction)
        elif ladder == "fixed1":
            strike = _pick_strike(ks, K0, 1, direction)
        else:                                    # atm1: re-centre on live spot
            at = day_bars[day_bars["minute_et"].dt.hour * 60
                          + day_bars["minute_et"].dt.minute <= fills[k]]
            if at.empty:
                strike = _pick_strike(ks, K0, 1, direction)
            else:
                spot = float(at.iloc[-1]["underlying_close"])
                strike = _pick_strike(ks, spot, 1, direction)
        if strike is None or strike not in by_k:
            strike = K0
        picks.append(strike)

    legs = []          # (fill_mod, strike, series)
    for m, strike in zip(sorted(fills)[:max_adds], picks):
        g = by_k.get(strike)
        if g is None:
            return None
        legs.append((m, strike, _series(g[g["minute_et"].notna()])))
    if not legs:
        return None

    # walk the union timeline from the first fill
    t0 = legs[0][0]
    grid = sorted({int(x) for _, _, s in legs for x in s[0] if x >= t0})
    if not grid:
        return None
    held = []          # (contracts, cost, series)
    total_cost = 0.0
    li = 0
    mae = 0.0
    budget = None
    for m in grid:
        while li < len(legs) and legs[li][0] <= m:
            fm, strike, s = legs[li]
            mod, cl, lo, mid = s
            i = int(np.searchsorted(mod, m, side="right")) - 1
            px = mid[i] if i >= 0 else np.nan
            if np.isfinite(px) and px > 0:
                if budget is None:
                    budget = px                  # first fill defines the per-add $
                ct = (budget / px) if equal_dollar else 1.0
                held.append((ct, s))
                total_cost += ct * px
            li += 1
        if not held:
            continue
        val = valo = 0.0
        for ct, s in held:
            mod, cl, lo, mid = s
            i = int(np.searchsorted(mod, m, side="right")) - 1
            if i < 0:
                continue
            val += ct * cl[i]
            valo += ct * lo[i]
        if total_cost <= 0:
            continue
        mae = min(mae, valo / total_cost - 1.0)
        if val >= total_cost * (1 + tr):
            return dict(ret=tr, why="TP", legs=len(held), cost=total_cost, mae=mae)
        if valo <= total_cost * (1 - stop):
            return dict(ret=-stop, why="STOP", legs=len(held), cost=total_cost, mae=mae)
        if m >= eod_m:
            return dict(ret=val / total_cost - 1.0, why="EOD", legs=len(held),
                        cost=total_cost, mae=mae)
    if not held or total_cost <= 0:
        return None
    val = 0.0
    for ct, s in held:
        val += ct * s[1][-1]
    return dict(ret=val / total_cost - 1.0, why="END", legs=len(held), cost=total_cost, mae=mae)


def _line(lbl, recs):
    if len(recs) < 5:
        return f"    {lbl:24} n={len(recs):>3}  (thin)"
    v = np.array([r for _, r, _ in recs])
    i = np.array([r for d, r, _ in recs if d < SPLIT])
    o = np.array([r for d, r, _ in recs if d >= SPLIT])
    ma = np.array([m for _, _, m in recs], float)
    sl = [[] for _ in range(6)]
    for d, r, _ in recs:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(r)
    npop = sum(1 for b in sl if len(b) >= 4)
    return (f"    {lbl:24} n={len(v):>3}  ret {v.mean()*100:>+6.1f}%  "
            f"IS {i.mean()*100 if len(i) else float('nan'):>+6.1f}%  "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+6.1f}%  "
            f"win {(v>0).mean():.2f}  MAE {ma.mean()*100:>+5.0f}%  pop {npop}/6")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok
    from sequential_fills import bracket_with_exit

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]

    out = collections.defaultdict(list)
    seq_all, gated_keys = [], set()
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
            continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        amt = amt_open_map(tk) if r.get("amt_open") else {}

        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
        if r.get("amt_open"):
            matched = [(t, th) for t, th in matched if amt_ok(r["amt_open"], amt.get(t["date"]))]
        matched.sort(key=lambda x: pd.Timestamp(x[0]["ts"]))
        tr_, rr_, em = float(r["target_roe"]), float(r["rr"]), _eod_mod(r)
        dtes = r.get("dte", [0, 1])

        byday = collections.defaultdict(list)
        for t, _th in matched:
            d, ts = t["date"], t["ts"]
            day = bbd.get(d)
            if day is None:
                continue
            at = day[day["minute_et"] <= ts]
            if at.empty:
                continue
            spot = float(at.iloc[-1]["underlying_close"])
            m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
            paths = D._option_paths(t, r["direction"], dtes, bbd, bbc)
            if not paths:
                continue
            cid = None
            for dd in dtes:
                cid = D.pick_contract(day, ts, r["direction"], dd, spot)
                if cid is not None:
                    break
            byday[d].append((m, cid, paths[0]))

        for d, fills in byday.items():
            if len(fills) < 2:
                continue
            fills.sort(key=lambda x: x[0])
            key = (r["name"], d)
            pnl, _ = bracket_with_exit(*fills[0][2], tr_, rr_, r.get("time_stop_mins"), em)
            seq_all.append((d, pnl, np.nan))
            if fills[0][0] <= 660 and vol.get(d) in ("LOWVOL", "NORMVOL"):
                gated_keys.add(key)
            day = bbd[d]
            first_cid = fills[0][1]
            if not first_cid:
                continue
            fr = day[day["option_chain_id"] == first_cid]
            if fr.empty:
                continue
            expiry = fr["expiry"].iloc[0]
            fm = [m for m, _, _ in fills]
            for lad in LADDERS:
                res = _basket(day, fm, r["direction"], expiry, first_cid,
                              tr_, a.stop, em, lad, a.max_adds, equal_dollar=True)
                if res:
                    out[("eq", lad)].append((d, res["ret"], res["mae"]))
                    if key in gated_keys:
                        out[("eq_gated", lad)].append((d, res["ret"], res["mae"]))
                res1 = _basket(day, fm, r["direction"], expiry, first_cid,
                               tr_, a.stop, em, lad, a.max_adds, equal_dollar=False)
                if res1:
                    out[("ct", lad)].append((d, res1["ret"], res1["mae"]))
            if key in gated_keys:
                out[("seq_gated", "same")].append((d, pnl, np.nan))

    print("=" * 112)
    print(f"  OTM SCALE-IN LADDERS   max {a.max_adds} adds, basket stop -{int(a.stop*100)}%   "
          f"multi-fill days only   split {SPLIT}")
    print("=" * 112)
    print(_line("SEQUENTIAL (1 position)", seq_all))
    print("\n  -- EQUAL-DOLLAR adds (cheaper strike => more contracts; total risk = n_adds x budget) --")
    for lad in LADDERS:
        print(_line(f"{lad}", out[("eq", lad)]))
    print("\n  -- 1 CONTRACT per add (cheaper strike => less capital deployed) --")
    for lad in LADDERS:
        print(_line(f"{lad}", out[("ct", lad)]))
    if out[("seq_gated", "same")]:
        print("\n  -- on the EARLY + LOW/NORMVOL gated days only --")
        print(_line("SEQ (1 position)", out[("seq_gated", "same")]))
        for lad in LADDERS:
            print(_line(f"eq-dollar {lad}", out[("eq_gated", lad)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--max-adds", type=int, default=3)
    ap.add_argument("--stop", type=float, default=0.5)
    a = ap.parse_args()
    run(a)
