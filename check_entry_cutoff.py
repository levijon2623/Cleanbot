# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_entry_cutoff.py
=====================
Deployment validation for the two survivors of the session-18 hour work.

  1. NO_ENTRY_AFTER_ET = 14:00
     check_holdtime: hour-14 entries are n=93, -5.2%, win 0.42, TP 14%, EOD 71%
     (82 min of budget against a +100% target). Dropping them raised BOTH
     per-trade (+9.5 -> +12.7%) and TOTAL book P&L (+49.7 -> +54.5 lot-units).
  2. META LOWVOL PUT: exclude hour 10
     check_hour10: hour 10 beats the rest in 6/7 rules -- META is the sole
     exception (-5.8pp). The user's pre-project hand-tuned config ALSO excluded
     hour 10 for META (legacy POSITIVE_GEX PUT hours [11,13]) -- two independent
     methods landing on the same exception.

Both are validated the same way, on SEQUENTIAL fills:
  * per-RULE impact (does any single rule carry / get hurt by it?)
  * 6 calendar slices (is the improvement stable, or one window?)
  * DAY-LEVEL bootstrap (trades cluster intraday; days are the honest unit)
  * total-P&L accounting, since dropping trades must not just raise the average

Usage:  python check_entry_cutoff.py [--boot 4000]
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


def _stat(lbl, sub):
    if len(sub) < 5:
        return f"    {lbl:30} n={len(sub):>4}  (thin)"
    v = sub["pnl"].to_numpy()
    i = sub[sub.date < SPLIT]["pnl"].to_numpy()
    o = sub[sub.date >= SPLIT]["pnl"].to_numpy()
    sl = [[] for _ in range(6)]
    for d, p in zip(sub["date"], sub["pnl"]):
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    npop = sum(1 for b in sl if len(b) >= 4)
    slc = " ".join(f"S{j+1}{np.mean(b)*100:+.0f}" if len(b) >= 4 else f"S{j+1}··"
                   for j, b in enumerate(sl))
    return (f"    {lbl:30} n={len(v):>4}  {v.mean()*100:>+6.1f}%  "
            f"IS {i.mean()*100 if len(i) else float('nan'):>+6.1f}%  "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+6.1f}%  "
            f"win {(v>0).mean():.2f}  tot {v.sum():>+6.2f}  pop {npop}/6  [{slc}]")


def _dayboot(sub, pool, boot, lbl):
    """Is `sub`'s day-level mean distinguishable from a random draw of days?"""
    sd = sub.groupby("date")["pnl"].mean()
    pd_ = pool.groupby("date")["pnl"].mean()
    if len(sd) < 8 or len(pd_) <= len(sd) + 3:
        return
    rng = np.random.default_rng(0)
    draws = np.array([rng.choice(pd_.to_numpy(), size=len(sd), replace=False).mean() * 100
                      for _ in range(boot)])
    gm = sd.mean() * 100
    p5, p95 = np.percentile(draws, [5, 95])
    pct = (draws < gm).mean() * 100
    tag = ("** worse than null" if gm < p5 else
           "** better than null" if gm > p95 else "(within noise)")
    print(f"    {lbl:30} day-level {gm:>+6.1f}% (d={len(sd):>3})  "
          f"random p5 {p5:+.1f} / p95 {p95:+.1f}  ({pct:>3.0f}th pct)  {tag}")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok
    from sequential_fills import bracket_with_exit

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
            pnl, xm = bracket_with_exit(*paths[0], tr_, rr_, r.get("time_stop_mins"), em)
            busy = xm
            rows.append(dict(rule=r["name"], ticker=tk, date=d, mod=m, hour=m // 60, pnl=pnl))

    R = pd.DataFrame(rows)
    print("=" * 124)
    print(f"  ENTRY-CUTOFF VALIDATION   {len(R)} sequential trades / {R.date.nunique()} days   split {SPLIT}")
    print("=" * 124)

    # ---------------- 1. NO_ENTRY_AFTER 14:00 ----------------
    print("\n### 1. NO_ENTRY_AFTER_ET = 14:00 " + "#" * 88)
    keep = R[R["hour"] < 14]
    drop = R[R["hour"] >= 14]
    print(_stat("BOOK as-is", R))
    print(_stat("BOOK cutoff 14:00", keep))
    print(_stat("  dropped (hour 14) trades", drop))
    print(f"\n    total P&L (1-lot units): as-is {R.pnl.sum():+.2f}  ->  cutoff {keep.pnl.sum():+.2f}  "
          f"(delta {keep.pnl.sum()-R.pnl.sum():+.2f});  "
          f"OOS {R[R.date>=SPLIT].pnl.sum():+.2f} -> {keep[keep.date>=SPLIT].pnl.sum():+.2f}")

    print("\n  -- per-rule impact of the 14:00 cutoff --")
    print(f"    {'rule':24} {'n>=14':>6} {'h14 mean':>9} {'rule as-is':>11} {'rule cut':>10} {'tot delta':>10}")
    for rn, g in R.groupby("rule"):
        h14 = g[g["hour"] >= 14]
        kp = g[g["hour"] < 14]
        if len(h14) == 0:
            continue
        print(f"    {rn:24} {len(h14):>6} {h14.pnl.mean()*100:>+8.1f}% "
              f"{g.pnl.mean()*100:>+10.1f}% {kp.pnl.mean()*100 if len(kp) else float('nan'):>+9.1f}% "
              f"{-h14.pnl.sum():>+9.2f}")
    print("    (tot delta = change in that rule's summed P&L from dropping its hour-14 trades)")

    print("\n  -- day-level bootstrap --")
    _dayboot(drop, R, a.boot, "hour-14 trades")
    _dayboot(drop[drop.date >= SPLIT], R[R.date >= SPLIT], a.boot, "hour-14 trades (OOS)")

    # ---------------- 2. META hour-10 exclusion ----------------
    print("\n### 2. META LOWVOL PUT: exclude hour 10 " + "#" * 81)
    M = R[R["rule"] == "META LOWVOL PUT"]
    if len(M) == 0:
        print("  (no META trades)")
    else:
        m10 = M[M["hour"] == 10]
        mrest = M[M["hour"] != 10]
        print(_stat("META as-is", M))
        print(_stat("META excl. hour 10", mrest))
        print(_stat("  META hour-10 trades", m10))
        print(f"\n    total P&L: as-is {M.pnl.sum():+.2f} -> excl-h10 {mrest.pnl.sum():+.2f}  "
              f"(delta {mrest.pnl.sum()-M.pnl.sum():+.2f})")
        print()
        _dayboot(m10, M, a.boot, "META hour-10")
        # is META's hour-10 weakness distinguishable from other rules' hour 10?
        o10 = R[(R["hour"] == 10) & (R["rule"] != "META LOWVOL PUT")]
        print(_stat("  other rules' hour 10", o10))

    # ---------------- 3. combined ----------------
    print("\n### 3. BOTH CHANGES " + "#" * 101)
    both = R[(R["hour"] < 14) & ~((R["rule"] == "META LOWVOL PUT") & (R["hour"] == 10))]
    print(_stat("BOOK as-is", R))
    print(_stat("BOOK cutoff only", keep))
    print(_stat("BOOK both changes", both))
    print(f"\n    total P&L: as-is {R.pnl.sum():+.2f}  cutoff {keep.pnl.sum():+.2f}  "
          f"both {both.pnl.sum():+.2f}")
    print(f"    OOS total: as-is {R[R.date>=SPLIT].pnl.sum():+.2f}  "
          f"cutoff {keep[keep.date>=SPLIT].pnl.sum():+.2f}  "
          f"both {both[both.date>=SPLIT].pnl.sum():+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    run(a)
