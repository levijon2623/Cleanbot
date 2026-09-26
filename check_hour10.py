# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_hour10.py
===============
Is the 10:00 hour a REAL structural edge, or the surviving tail of the
time-budget artifact?

check_holdtime showed "entry <= 11:00" is mostly a clock effect -- but it also
showed the curve is NOT monotone in budget: hour 9 has the MOST time (365 min
avg) yet returned +6.0%, while hour 10 (314 min) returned +32.7%. If the edge
were purely "more minutes to reach TP", hour 9 should beat hour 10. It doesn't.

Market-structure claim under test: the opening auction's imbalance clears and
institutional/VWAP algos engage around 10:00, so flow signals in that window
carry more information than the 09:30-10:00 noise or the midday drift.

Tests:
  1. HOUR CURVE, sequential fills, deployed rules -- n / IS / OOS / win per hour
  2. HOUR 9 vs HOUR 10 head-to-head -- the natural experiment: near-matched time
     budget, so a gap here is NOT mechanical
  3. MATCHED TIME STOP by hour -- give every trade the same budget T; does hour
     10 still lead?
  4. PER-RULE -- does hour 10 lead in many rules independently (structural) or
     one (composition)?
  5. INDEPENDENT CORROBORATION -- _ml_cache/dataset.parquet: 145k labelled
     triggers across 12 tickers (incl. non-deployed ones), day-level means per
     hour so intraday clustering cannot inflate any bucket
  6. DAY-LEVEL BOOTSTRAP -- hour-10 days vs random days

Usage:  python check_hour10.py [--boot 2000]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _eod_mod(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _row(lbl, v, i, o, extra=""):
    return (f"    {lbl:22} n={len(v):>4}  all {np.mean(v)*100:>+6.1f}%  "
            f"IS {np.mean(i)*100 if len(i) else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if len(o) else float('nan'):>+6.1f}%  "
            f"win {np.mean(np.array(v) > 0):.2f}{extra}")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok
    from check_holdtime import _bracket_full

    TSTOPS = (60, 90, 120)
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
            rec = dict(rule=r["name"], ticker=tk, date=d, mod=m, hour=m // 60,
                       pnl=pnl, budget=em - m, why=why)
            for T in TSTOPS:
                rec[f"p{T}"], _x, _w = _bracket_full(*paths[0], tr_, rr_, T, em)
            rows.append(rec)

    R = pd.DataFrame(rows)
    if R.empty:
        print("no trades"); return
    print("=" * 112)
    print(f"  HOUR-10 STRUCTURAL TEST   {len(R)} sequential trades, deployed rules   split {SPLIT}")
    print("=" * 112)

    print("\n  -- 1. hour curve --")
    for h, g in R.groupby("hour"):
        if len(g) < 5:
            continue
        print(_row(f"hour {int(h)}", g.pnl.to_numpy(),
                   g[g.date < SPLIT].pnl.to_numpy(), g[g.date >= SPLIT].pnl.to_numpy(),
                   f"  budget {g.budget.mean():>3.0f}m  TP {(g.why=='TP').mean()*100:>2.0f}%"))

    print("\n  -- 2. hour 9 vs hour 10 (near-matched budget => not mechanical) --")
    h9, h10 = R[R.hour == 9], R[R.hour == 10]
    if len(h9) >= 8 and len(h10) >= 8:
        print(_row("hour 9", h9.pnl.to_numpy(), h9[h9.date < SPLIT].pnl.to_numpy(),
                   h9[h9.date >= SPLIT].pnl.to_numpy(), f"  budget {h9.budget.mean():.0f}m"))
        print(_row("hour 10", h10.pnl.to_numpy(), h10[h10.date < SPLIT].pnl.to_numpy(),
                   h10[h10.date >= SPLIT].pnl.to_numpy(), f"  budget {h10.budget.mean():.0f}m"))
        print(f"    -> budget differs by only {h9.budget.mean()-h10.budget.mean():.0f} min, "
              f"P&L by {(h10.pnl.mean()-h9.pnl.mean())*100:+.1f}pp")

    print("\n  -- 3. MATCHED TIME STOP by hour (identical opportunity) --")
    for T in TSTOPS:
        line = f"    T={T:>3}m  "
        for h in (9, 10, 11, 12, 13, 14):
            g = R[(R.hour == h) & (R.budget >= T)]
            if len(g) < 8:
                continue
            line += f"h{h} {g[f'p{T}'].mean()*100:>+5.1f}%(n{len(g)}) "
        print(line)
    for T in TSTOPS:
        line = f"    T={T:>3}m OOS  "
        for h in (9, 10, 11, 12, 13, 14):
            g = R[(R.hour == h) & (R.budget >= T) & (R.date >= SPLIT)]
            if len(g) < 6:
                continue
            line += f"h{h} {g[f'p{T}'].mean()*100:>+5.1f}%(n{len(g)}) "
        print(line)

    print("\n  -- 4. per-rule: is hour 10 the best hour in many rules? --")
    w = t = 0
    for rn, g in R.groupby("rule"):
        g10 = g[g.hour == 10]
        gother = g[g.hour != 10]
        if len(g10) < 4 or len(gother) < 8:
            continue
        t += 1
        better = g10.pnl.mean() > gother.pnl.mean()
        w += better
        print(f"    {rn:22} h10 n={len(g10):>3} {g10.pnl.mean()*100:>+6.1f}%   "
              f"other n={len(gother):>3} {gother.pnl.mean()*100:>+6.1f}%   "
              f"delta {(g10.pnl.mean()-gother.pnl.mean())*100:>+6.1f}pp {'*' if better else ''}")
    print(f"    -> hour 10 beats the rest in {w}/{t} rules")

    print("\n  -- 5. INDEPENDENT: 12-ticker trigger population (_ml_cache), DAY-LEVEL means --")
    p = "_ml_cache/dataset.parquet"
    if os.path.exists(p):
        M = pd.read_parquet(p, columns=["ticker", "date", "hour", "y_pnl", "is_live_combo"])
        # de-cluster: one observation per (ticker, date, hour)
        g = M.groupby(["ticker", "date", "hour"])["y_pnl"].mean().reset_index()
        g["date"] = pd.to_datetime(g["date"]).dt.date
        for lbl, sub in (("all 12 tickers", g),
                         ("non-deployed combos", g.merge(
                             M.groupby(["ticker", "date", "hour"])["is_live_combo"].max().reset_index(),
                             on=["ticker", "date", "hour"]).query("is_live_combo == 0"))):
            print(f"    [{lbl}]")
            for h in sorted(sub["hour"].unique()):
                s = sub[sub["hour"] == h]
                if len(s) < 30:
                    continue
                si, so = s[s.date < SPLIT], s[s.date >= SPLIT]
                print(f"      hour {int(h)}  n={len(s):>5}  all {s.y_pnl.mean()*100:>+6.1f}%  "
                      f"IS {si.y_pnl.mean()*100 if len(si) else float('nan'):>+6.1f}%  "
                      f"OOS {so.y_pnl.mean()*100 if len(so) else float('nan'):>+6.1f}%")
        # per-ticker: in how many of the 12 is hour 10 above that ticker's mean?
        cnt = 0; tot = 0
        for tk, s in g.groupby("ticker"):
            s10 = s[s.hour == 10]; rest = s[s.hour != 10]
            if len(s10) < 20 or len(rest) < 40:
                continue
            tot += 1
            cnt += s10.y_pnl.mean() > rest.y_pnl.mean()
        print(f"    -> hour 10 beats that ticker's other hours in {cnt}/{tot} tickers")
    else:
        print("    (_ml_cache/dataset.parquet missing -- run ml_feature_scan.py --build)")

    print("\n  -- 6. day-level bootstrap (hour-10 days vs random days), OOS --")
    oos = R[R.date >= SPLIT]
    d10 = oos[oos.hour == 10].groupby("date")["pnl"].mean()
    dall = oos.groupby("date")["pnl"].mean()
    if len(d10) >= 8 and len(dall) > len(d10) + 3:
        rng = np.random.default_rng(0)
        draws = np.array([rng.choice(dall.to_numpy(), size=len(d10), replace=False).mean() * 100
                          for _ in range(a.boot)])
        gm = d10.mean() * 100
        p95 = np.percentile(draws, 95)
        print(f"    hour-10 OOS day-level {gm:+.1f}% (d={len(d10)})  vs random mean "
              f"{draws.mean():+.1f}% / p95 {p95:+.1f}%  ({(draws < gm).mean()*100:.0f}th pct)  "
              f"{'** BEATS null' if gm > p95 else '(within noise)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run(a)
