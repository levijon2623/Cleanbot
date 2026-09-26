# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_holdtime.py
=================
Is "EARLY entry" a REGIME signal or just a MECHANICAL time-budget artifact?

check_size_gate found entries <= 11:00 return +27.0% (IS +27.9 / OOS +26.0, 6/6
slices, day-level bootstrap 100th pct) vs +5.5% for later entries. But an 09:35
entry has ~6h to reach a +100% take-profit before the 15:55 flatten, while a
14:30 entry has ~1.5h. More time = more chance of touching TP, with no market
insight required. That alone could manufacture the whole effect.

The controls, in order of strictness:

  A. entry-hour curve      -- P&L, win, and EXIT MIX by hour. If early's edge is
                              mechanical, early trades should show a much higher
                              TP-hit rate and the P&L curve should track it.
  B. time-budget curve     -- P&L vs minutes available to EOD, ignoring entry
                              hour. Same thing seen from the other side.
  C. MATCHED TIME STOP     -- the decisive one. Give EVERY trade the SAME budget
                              T (time_stop_mins = T) and keep only trades with
                              >= T minutes before the flatten, so early and late
                              have IDENTICAL opportunity. If early still wins,
                              it is a regime effect. If the gap collapses, it
                              was the clock.
  D. resolved-only         -- restrict to trades that hit TP or SL on their own
                              (never reached the EOD flatten): a budget-free cut.

Usage:  python check_holdtime.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
EARLY = 660          # 11:00 ET


def _eod_mod(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _bracket_full(entry_mid, cl, lo, mod, held, tr, rr, tstop, eod_m):
    """P&L + exit minute + exit reason (TP / SL / TSTOP / EOD / END)."""
    n = len(cl)
    cummax_cl = np.maximum.accumulate(cl)
    cummin_lo = np.minimum.accumulate(lo)
    eod = mod >= eod_m
    ts_idx = int(np.argmax(eod)) if eod.any() else n - 1
    why_t = "EOD" if eod.any() else "END"
    if tstop:
        th = held >= tstop
        if th.any() and int(np.argmax(th)) < ts_idx:
            ts_idx = int(np.argmax(th)); why_t = "TSTOP"
    tp = entry_mid * (1 + tr)
    sl = entry_mid * (1 - tr / rr)
    tp_idx = int(np.searchsorted(cummax_cl, tp)) if cummax_cl[-1] >= tp else n
    sl_idx = int(np.searchsorted(-cummin_lo, -sl)) if cummin_lo[-1] <= sl else n
    ei = min(tp_idx, sl_idx, ts_idx)
    import directional_flow_backtester as D
    if ei >= n:
        px, i, why = cl[-1], n - 1, "END"
    elif tp_idx <= sl_idx and tp_idx == ei:
        px, i, why = tp, tp_idx, "TP"
    elif sl_idx == ei:
        px, i, why = min(sl, cl[sl_idx]), sl_idx, "SL"
    else:
        px, i, why = cl[ei], ei, why_t
    return (px - entry_mid) / entry_mid - D.COMMISSION_PCT, int(mod[i]), why


def _split(sub, col="pnl"):
    if len(sub) < 6:
        return f"n={len(sub):>3} (thin)"
    v = sub[col].to_numpy()
    i = sub[sub.date < SPLIT][col].to_numpy()
    o = sub[sub.date >= SPLIT][col].to_numpy()
    return (f"n={len(v):>4}  all {v.mean()*100:>+6.1f}%  "
            f"IS {i.mean()*100 if len(i) else float('nan'):>+6.1f}%  "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+6.1f}%  win {(v>0).mean():.2f}")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok

    TSTOPS = (30, 60, 90, 120)
    rows = []
    for r in [x for x in RULES if x.get("enabled", True)]:
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

        cur, busy = None, -1
        for t, _th in matched:
            d, ts = t["date"], t["ts"]
            m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
            if d != cur:
                cur, busy = d, -1
            if m < busy:
                continue
            paths = D._option_paths(t, r["direction"], dtes, bbd, bbc)
            if not paths:
                continue
            pnl, xm, why = _bracket_full(*paths[0], tr_, rr_, r.get("time_stop_mins"), em)
            busy = xm
            rec = dict(rule=r["name"], date=d, mod=m, pnl=pnl, xmod=xm, why=why,
                       hold=xm - m, budget=em - m, early=m <= EARLY)
            for T in TSTOPS:
                p2, _x2, w2 = _bracket_full(*paths[0], tr_, rr_, T, em)
                rec[f"p{T}"] = p2
                rec[f"w{T}"] = w2
            rows.append(rec)

    R = pd.DataFrame(rows)
    if R.empty:
        print("no trades"); return
    print("=" * 118)
    print(f"  HOLD-TIME / ENTRY-TIME CONTROL   {len(R)} sequential trades   split {SPLIT}")
    print("=" * 118)
    print(f"  baseline   early(<=11:00) {_split(R[R.early])}")
    print(f"             late (>11:00)  {_split(R[~R.early])}")

    print("\n  -- A. entry-hour curve (exit mix is the tell) --")
    print(f"    {'hour':>5} {'n':>4} {'mean':>8} {'win':>5} {'TP%':>5} {'SL%':>5} {'EOD%':>5} "
          f"{'hold':>5} {'budget':>6}")
    for h, g in R.groupby(R["mod"] // 60):
        if len(g) < 5:
            continue
        wc = g["why"].value_counts(normalize=True)
        print(f"    {int(h):>5} {len(g):>4} {g.pnl.mean()*100:>+7.1f}% {(g.pnl>0).mean():>5.2f} "
              f"{wc.get('TP',0)*100:>5.0f} {wc.get('SL',0)*100:>5.0f} {wc.get('EOD',0)*100:>5.0f} "
              f"{g.hold.mean():>5.0f} {g.budget.mean():>6.0f}")

    print("\n  -- B. P&L by TIME BUDGET available (minutes to flatten) --")
    q = pd.qcut(R["budget"], 4, labels=False, duplicates="drop")
    for k in sorted(set(q.dropna())):
        g = R[q == k]
        print(f"    budget Q{k+1} ({g.budget.min():.0f}-{g.budget.max():.0f}m)  {_split(g)}  "
              f"TP {(g.why=='TP').mean()*100:.0f}%")

    print("\n  -- C. MATCHED TIME STOP (identical opportunity for early and late) --")
    print(f"    {'T':>5} {'kept':>5} {'early':>28} {'late':>28} {'delta':>8}")
    for T in TSTOPS:
        el = R[(R["budget"] >= T) & R["early"]]
        la = R[(R["budget"] >= T) & ~R["early"]]
        if len(el) < 8 or len(la) < 8:
            continue
        de = el[f"p{T}"].mean() * 100
        dl = la[f"p{T}"].mean() * 100
        eo = el[el.date >= SPLIT][f"p{T}"]
        lo = la[la.date >= SPLIT][f"p{T}"]
        print(f"    {T:>5} {len(el)+len(la):>5}   n={len(el):>3} {de:>+6.1f}% "
              f"(OOS {eo.mean()*100 if len(eo) else float('nan'):>+6.1f}%)   "
              f"n={len(la):>3} {dl:>+6.1f}% "
              f"(OOS {lo.mean()*100 if len(lo) else float('nan'):>+6.1f}%)   {de-dl:>+7.1f}pp")

    print("\n  -- D. resolved-only (hit TP or SL on their own, never reached the flatten) --")
    res = R[R["why"].isin(("TP", "SL"))]
    print(f"    early  {_split(res[res.early])}")
    print(f"    late   {_split(res[~res.early])}")
    print(f"    (resolved share: early {(R[R.early]['why'].isin(('TP','SL'))).mean()*100:.0f}%  "
          f"late {(R[~R.early]['why'].isin(('TP','SL'))).mean()*100:.0f}%)")

    print("\n  -- E. same-budget slice: only trades with 120-200 min of budget --")
    band = R[(R["budget"] >= 120) & (R["budget"] <= 200)]
    if len(band) > 20:
        print(f"    early  {_split(band[band.early])}")
        print(f"    late   {_split(band[~band.early])}")
    if len(band[band.early]) == 0:
        print("    (early trades never have a late-trade budget -- entry hour and time"
              " budget are near-collinear by construction)")

    print("\n  -- F. LATE-ENTRY CUTOFF: what does dropping the tail do to the book? --")
    base = R["pnl"]
    bi, bo = R[R.date < SPLIT]["pnl"], R[R.date >= SPLIT]["pnl"]
    print(f"    {'cutoff':>8} {'kept':>5} {'drop':>5}   {'per-trade':>27}   {'book total (1-lot units)':>26}")
    for cut, lbl in ((15 * 60, "none (15:00)"), (14 * 60, "14:00"), (13 * 60, "13:00"),
                     (12 * 60, "12:00"), (11 * 60, "11:00")):
        k = R[R["mod"] < cut]
        if len(k) < 20:
            continue
        ki, ko = k[k.date < SPLIT]["pnl"], k[k.date >= SPLIT]["pnl"]
        print(f"    {lbl:>8} {len(k):>5} {len(R)-len(k):>5}   "
              f"all {k.pnl.mean()*100:>+6.1f}% IS {ki.mean()*100:>+6.1f}% OOS {ko.mean()*100:>+6.1f}%   "
              f"tot {k.pnl.sum():>+7.2f}  IS {ki.sum():>+6.2f}  OOS {ko.sum():>+6.2f}")
    print(f"    (book total = sum of per-trade returns in 1-lot units; dropping trades"
          f" lowers total unless they were net-negative)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.parse_args()
    run(a)
