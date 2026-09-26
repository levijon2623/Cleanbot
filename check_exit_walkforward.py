# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_exit_walkforward.py
=========================
Three questions left open by check_exit_sweep (which found `trail 50%` lifts the
book from OOS +7.2% / total +52.84 to OOS +13.6% / total +91.66):

  1. WALK-FORWARD THE POLICY CHOICE. The sweep is a 40-policy grid search on one
     dataset. Here the policy is CHOSEN on everything before a slice and scored
     on that slice only, chained across slices -- the honest test of "would we
     have picked this in time".
  2. REALISTIC FILLS. A trailing stop exits on the way DOWN, so it is more
     exposed to the bid/ask than a static bracket. Enter at ASK, exit at BID.
  3. CONTAMINATION. Every rule/gate/percentile in config was selected while the
     bracket was tr1.0/rr1.0 = NO STOP. Under a trailing exit, do the deployed
     GATES still earn their keep, or were some of them compensating for the
     absent stop?

Usage:
  python check_exit_walkforward.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx, SLICE_EDGES, STACK_FIELDS

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _eod_mod(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _policies():
    P = []
    for tp in (0.75, 1.0, 1.5):
        for st in (0.50, 0.65, None):
            P.append(dict(name=f"fixed tp{int(tp*100)}/sl{int(st*100) if st else 'none'}",
                          kind="fixed", tp=tp, stop=st))
    for tr in (0.30, 0.40, 0.50, 0.60):
        P.append(dict(name=f"trail {int(tr*100)}%", kind="trail", tp=None, trail=tr, stop=None))
    P.append(dict(name="combo tp150+trail50", kind="trail", tp=1.5, trail=0.50, stop=None))
    return P


def _sim(path, pol, eod_m, realistic=False, cushion=None):
    """DEPRECATED SHIM -- delegates to sim_core.simulate.

    Kept so existing callers keep working, but they now get the CORRECTED fill
    model. `cushion` defaults to True whenever the payload carries the ask array
    (the 8-tuple from sim_core.build_candidates); legacy 7-tuple payloads cannot
    model it and silently run without. Pass cushion=False only to reproduce a
    historical number, never to make a decision. See sim_core's header.

    Returns (pnl, exit_mod) -- the tag is dropped for signature compatibility.
    """
    import sim_core
    if cushion is None:
        cushion = len(path) == 8
    pnl, xm, _tag = sim_core.simulate(path, pol, eod_m, realistic, cushion)
    return pnl, xm


def _walk(cand, pol, eod_m, realistic=False, cushion=None):
    """DEPRECATED SHIM -- delegates to sim_core.walk (same guard, corrected fills)."""
    import sim_core
    if cushion is None:
        cushion = any(len(p) == 8 for _d, _m, p in cand[:1]) if cand else False
    return sim_core.walk(cand, pol, eod_m, realistic=realistic, cushion=cushion)


def _walk_legacy(cand, pol, eod_m, realistic=False):
    cur, busy, out = None, -1, []
    for d, m, path in cand:
        if d != cur:
            cur, busy = d, -1
        if m < busy:
            continue
        pnl, xm = _sim(path, pol, eod_m, realistic)
        out.append((d, pnl))
        busy = xm
    return out


def _agg(pnls):
    if not pnls:
        return dict(n=0, mean=np.nan, tot=0.0)
    v = np.array([p for _, p in pnls])
    return dict(n=len(v), mean=v.mean(), tot=v.sum())


def _line(lbl, pnls):
    if len(pnls) < 15:
        return f"    {lbl:28} n={len(pnls):>4}  (thin)"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    c = m = 0
    for _, p in sorted(pnls):
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    return (f"    {lbl:28} n={len(v):>4}  {v.mean()*100:>+6.1f}%  "
            f"IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  "
            f"win {(v>0).mean():.2f}  maxLL {m:>2}  tot {v.sum():>+7.2f}")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok

    POL = _policies()
    rules = [r for r in RULES if r.get("enabled", True)]
    cands = {}          # rule -> {"dep": cand, "plain": cand, "eod": em}

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
        plain_r = {k: v for k, v in r.items() if k not in STACK_FIELDS}
        em = _eod_mod(r)
        dtes = r.get("dte", [0, 1])

        def build(rule_spec, use_amt):
            mt = D._rule_matched_trigs(rule_spec, trigs, gex, vol, trd, amp, reg_src)
            if use_amt and rule_spec.get("amt_open"):
                mt = [(t, th) for t, th in mt if amt_ok(rule_spec["amt_open"], amt.get(t["date"]))]
            mt.sort(key=lambda x: pd.Timestamp(x[0]["ts"]))
            out = []
            for t, _th in mt:
                d, ts = t["date"], t["ts"]
                day = bbd.get(d)
                if day is None:
                    continue
                at = day[day["minute_et"] <= ts]
                if at.empty:
                    continue
                spot = float(at.iloc[-1]["underlying_close"])
                cid = None
                for dd in dtes:
                    cid = D.pick_contract(day, ts, r["direction"], dd, spot)
                    if cid is not None:
                        break
                if cid is None:
                    continue
                ent = bbc[cid]
                er = ent[(ent["minute_et"] <= ts) & (ent["minute_et"] >= ts - pd.Timedelta(minutes=3))]
                if er.empty:
                    continue
                er = er.iloc[-1]
                b, k = float(er["bid_close"]), float(er["ask_close"])
                mid = (b + k) / 2.0 if b > 0 else float(er["close"])
                if mid < 0.50:
                    continue
                ask = k if k > 0 else mid
                fwd = ent[ent["minute_et"] > ts].sort_values("minute_et")
                if len(fwd) < 3:
                    continue
                pm = fwd["minute_et"]
                out.append((d, pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute,
                            (mid, ask, fwd["close"].to_numpy(float), fwd["high"].to_numpy(float),
                             fwd["low"].to_numpy(float), fwd["bid_close"].to_numpy(float),
                             (pm.dt.hour.values * 60 + pm.dt.minute.values).astype(int))))
            return out

        cands[r["name"]] = dict(dep=build(r, True), plain=build(plain_r, False), eod=em,
                                tr=float(r["target_roe"]), rr=float(r["rr"]))

    dep_pol = dict(name="DEPLOYED", kind="fixed", tp=1.0, stop=None)   # tr1/rr1 == no stop

    def book(pol, key="dep", realistic=False):
        out = []
        for rn, c in cands.items():
            p = pol
            if pol["name"] == "DEPLOYED":
                p = dict(name="DEPLOYED", kind="fixed", tp=c["tr"],
                         stop=(c["tr"] / c["rr"] if c["tr"] / c["rr"] < 1.0 else None))
            out += _walk(c[key], p, c["eod"], realistic)
        return out

    print("=" * 116)
    print("  EXIT POLICY: WALK-FORWARD, REALISTIC FILLS, AND GATE CONTAMINATION")
    print("=" * 116)

    # ---------- 1. walk-forward the policy choice ----------
    print("\n### 1. WALK-FORWARD the policy choice (pick on prior slices, score on the next)")
    allbooks = {p["name"]: book(p) for p in POL}
    allbooks["DEPLOYED"] = book(dep_pol)
    chained, chosen = [], []
    for k in range(1, 6):
        lo_edge = SLICE_EDGES[k]
        best, bestv = None, -9e9
        for nm, pn in allbooks.items():
            if nm == "DEPLOYED":
                continue
            tr = [(d, p) for d, p in pn if d < lo_edge]
            if len(tr) < 40:
                continue
            s = np.mean([p for _, p in tr])
            if s > bestv:
                best, bestv = nm, s
        if best is None:
            continue
        seg = [(d, p) for d, p in allbooks[best] if _slice_idx(d) == k]
        dseg = [(d, p) for d, p in allbooks["DEPLOYED"] if _slice_idx(d) == k]
        t50 = [(d, p) for d, p in allbooks["trail 50%"] if _slice_idx(d) == k]
        chained += seg
        chosen.append(best)
        print(f"    S{k+1}: picked {best:22} -> {np.mean([p for _,p in seg])*100:>+6.1f}% (n{len(seg)})   "
              f"deployed {np.mean([p for _,p in dseg])*100 if dseg else float('nan'):>+6.1f}%   "
              f"always-trail50 {np.mean([p for _,p in t50])*100 if t50 else float('nan'):>+6.1f}%")
    print()
    print(_line("WF-chained (S2..S6)", chained))
    print(_line("DEPLOYED same window", [(d, p) for d, p in allbooks["DEPLOYED"] if _slice_idx(d) >= 1]))
    print(_line("always-trail50 same win", [(d, p) for d, p in allbooks["trail 50%"] if _slice_idx(d) >= 1]))
    print(f"    policies picked: {chosen}")

    # ---------- 2. realistic fills ----------
    print("\n### 2. REALISTIC FILLS (enter ASK, exit BID)")
    for nm in ("DEPLOYED", "trail 50%", "trail 40%", "fixed tp100/slnone", "combo tp150+trail50"):
        pol = dep_pol if nm == "DEPLOYED" else next(p for p in POL if p["name"] == nm)
        print(_line(f"{nm} [mid]", book(pol)))
        print(_line(f"{nm} [real]", book(pol, realistic=True)))

    # ---------- 3. contamination ----------
    print("\n### 3. CONTAMINATION: do the GATES still earn their keep under a trailing exit?")
    t50 = next(p for p in POL if p["name"] == "trail 50%")
    print(_line("DEPLOYED spec / dep exit", book(dep_pol, "dep")))
    print(_line("PLAIN spec  / dep exit", book(dep_pol, "plain")))
    print(_line("DEPLOYED spec / trail50", book(t50, "dep")))
    print(_line("PLAIN spec  / trail50", book(t50, "plain")))
    print("\n  -- per-rule: gate value under the OLD exit vs under trail50 --")
    print(f"    {'rule':22} {'gates(old)':>12} {'gates(t50)':>12}   verdict")
    for rn, c in cands.items():
        if not c["dep"] or not c["plain"] or len(c["dep"]) == len(c["plain"]):
            continue
        pdep = dict(name="DEPLOYED", kind="fixed", tp=c["tr"],
                    stop=(c["tr"] / c["rr"] if c["tr"] / c["rr"] < 1.0 else None))
        old = (np.mean([p for _, p in _walk(c["dep"], pdep, c["eod"])])
               - np.mean([p for _, p in _walk(c["plain"], pdep, c["eod"])])) * 100
        new = (np.mean([p for _, p in _walk(c["dep"], t50, c["eod"])])
               - np.mean([p for _, p in _walk(c["plain"], t50, c["eod"])])) * 100
        verdict = ("gate still helps" if new > 2 else
                   "gate now NEUTRAL" if new > -2 else "gate now HURTS")
        print(f"    {rn:22} {old:>+11.1f}pp {new:>+11.1f}pp   {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.parse_args()
    run(a)
