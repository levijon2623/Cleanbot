# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "requests", "python-dotenv"]
# ///
"""
check_size_gate.py
==================
Validation of the ONE survivor of the session-18 scale-in work.

Every scale-in variant lost to simply taking the single position (same-strike,
capped, env-gated, delta-gated, OTM-laddered). What survived was the observation
that the ENVIRONMENT gate selects good DAYS: on early + LOW/NORMVOL days the
single position ran ~+32% IS / ~+25% OOS vs ~+13% / ~+5% elsewhere. The right
use of that is a SIZE multiplier on a single position, not more positions.

This validates it the way every other lever in this book gets validated, on the
SEQUENTIAL fill model (sequential_fills.walk -- the one the bot actually runs):

  1. component decomposition -- early / LOWVOL / NORMVOL / -GEX and combos, each
     with IS/OOS + 6 calendar slices (is any single piece carrying it?)
  2. WITHIN-RULE -- gated vs ungated inside each rule (is it a real lever, or
     just composition: gated days landing in the already-good rules?)
  3. BOOTSTRAP NULL -- gated OOS vs p95 of same-size random OOS subsamples
  4. VIX-OVERLAY INTERACTION -- the book already has a size multiplier keyed on
     VIX vs its 60d median; do these overlap or are they independent?
  5. SIZING SIM -- multiplier grid: P&L per unit capital, and what it does to
     same-day concentration vs MAX_PORTFOLIO_RISK_PCT

Thresholds are NOT re-fit here; the gate (first entry <= 11:00, vol regime in
LOWVOL/NORMVOL) was selected on the IS half in check_scalein and is taken as
given. Components are reported to expose how much of it is fragile.

Usage:
  python check_size_gate.py [--boot 2000]
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
GATE_MOD = 660          # 11:00 ET
GATE_VOL = ("LOWVOL", "NORMVOL")


def _eod_mod(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _line(lbl, sub, base_n=None):
    if len(sub) < 8:
        return f"    {lbl:34} n={len(sub):>4}  (thin)"
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
    share = f" ({100*len(v)/base_n:>3.0f}%)" if base_n else ""
    return (f"    {lbl:34} n={len(v):>4}{share}  all {v.mean()*100:>+6.1f}%  "
            f"IS {i.mean()*100 if len(i) else float('nan'):>+6.1f}%  "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+6.1f}%  "
            f"win {(v>0).mean():.2f}  pop {npop}/6  [{slc}]")


def _vix_fav():
    """{date: bool} prior-day VIX >= its trailing-60-session median (the same
    'favourable' definition the deployed VIX size overlay uses)."""
    try:
        from check_regime_state import _vix
        s, _ = _vix()
        if len(s) == 0:
            return {}
        med = s.rolling(60, min_periods=20).median()
        return {d: (bool(s[d] >= med[d]) if pd.notna(med[d]) and pd.notna(s[d]) else None)
                for d in s.index}
    except Exception as e:
        print(f"  (VIX unavailable: {e})")
        return {}


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok
    from sequential_fills import bracket_with_exit

    rules = [r for r in RULES if r.get("enabled", True)]
    vixfav = _vix_fav()

    rows = []
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

        cur_day, busy = None, -1
        for t, _th in matched:
            d, ts = t["date"], t["ts"]
            m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
            if d != cur_day:
                cur_day, busy = d, -1
            if m < busy:
                continue
            paths = D._option_paths(t, r["direction"], dtes, bbd, bbc)
            if not paths:
                continue
            pnl, xm = bracket_with_exit(*paths[0], tr_, rr_, r.get("time_stop_mins"), em)
            busy = xm
            rows.append(dict(rule=r["name"], ticker=tk, date=d, mod=m, pnl=pnl,
                             entry=float(paths[0][0]),
                             vol=vol.get(d), gex=gex.get(d), trend=trd.get(d),
                             vixfav=vixfav.get(d),
                             vix_size=r.get("vix_size", True)))

    R = pd.DataFrame(rows)
    if R.empty:
        print("no trades"); return
    R["early"] = R["mod"] <= GATE_MOD
    R["lowvol"] = R["vol"].isin(GATE_VOL)
    R["gate"] = R["early"] & R["lowvol"]
    N = len(R)

    print("=" * 126)
    print(f"  SIZE-GATE VALIDATION   {N} SEQUENTIAL trades / {R.date.nunique()} days   split {SPLIT}")
    print("=" * 126)
    print(_line("BASELINE (all)", R))

    print("\n  -- 1. component decomposition --")
    comps = [
        ("early (<=11:00)", R["early"]),
        ("late (>11:00)", ~R["early"]),
        ("vol LOWVOL", R["vol"] == "LOWVOL"),
        ("vol NORMVOL", R["vol"] == "NORMVOL"),
        ("vol HIVOL", R["vol"] == "HIVOL"),
        ("gex NEGATIVE", R["gex"] == "NEGATIVE"),
        ("gex POSITIVE", R["gex"] == "POSITIVE"),
        ("GATE early+LOW/NORM", R["gate"]),
        ("  gate + gex NEG", R["gate"] & (R["gex"] == "NEGATIVE")),
        ("NOT gate", ~R["gate"]),
    ]
    for lbl, m in comps:
        print(_line(lbl, R[m], N))

    print("\n  -- 2. WITHIN-RULE (is it a lever, or just composition?) --")
    for col, nm in (("gate", "GATE early+LOW/NORM"), ("early", "EARLY alone (<=11:00)")):
        print(f"\n    [{nm}]")
        w = t = 0
        for rn, g in R.groupby("rule"):
            gi, gn = g[g[col]], g[~g[col]]
            if len(gi) < 6 or len(gn) < 6:
                continue
            t += 1
            w += gi.pnl.mean() > gn.pnl.mean()
            gio, gno = gi[gi.date >= SPLIT], gn[gn.date >= SPLIT]
            print(f"      {rn:22} on n={len(gi):>3} {gi.pnl.mean()*100:>+6.1f}% "
                  f"(OOS {gio.pnl.mean()*100 if len(gio) else float('nan'):>+6.1f}%)   "
                  f"off n={len(gn):>3} {gn.pnl.mean()*100:>+6.1f}% "
                  f"(OOS {gno.pnl.mean()*100 if len(gno) else float('nan'):>+6.1f}%)   "
                  f"delta {(gi.pnl.mean()-gn.pnl.mean())*100:>+6.1f}pp")
        print(f"      -> helps in {w}/{t} rules with enough sample")

    print("\n  -- 3. bootstrap null (OOS vs random OOS subsamples) --")
    oos = R[R.date >= SPLIT]
    pool = oos["pnl"].to_numpy()
    for col, nm in (("gate", "gate"), ("early", "early")):
        g_oos = oos[oos[col]]["pnl"].to_numpy()
        if len(g_oos) < 10 or len(pool) <= len(g_oos) + 5:
            continue
        rng = np.random.default_rng(0)
        draws = np.array([rng.choice(pool, size=len(g_oos), replace=False).mean() * 100
                          for _ in range(a.boot)])
        gm = g_oos.mean() * 100
        p95 = np.percentile(draws, 95)
        rank = (draws < gm).mean() * 100
        print(f"    {nm:6} OOS {gm:+.1f}% (n={len(g_oos)})  vs random mean {draws.mean():+.1f}% / "
              f"p95 {p95:+.1f}%  ({rank:.0f}th pct)  "
              f"{'** BEATS null' if gm > p95 else '(within noise)'}")
    # day-level (trades cluster within a day -> resample DAYS, not trades)
    for col, nm in (("gate", "gate"), ("early", "early")):
        gd = oos[oos[col]].groupby("date")["pnl"].mean()
        ad = oos.groupby("date")["pnl"].mean()
        if len(gd) < 8 or len(ad) <= len(gd) + 3:
            continue
        rng = np.random.default_rng(1)
        draws = np.array([rng.choice(ad.to_numpy(), size=len(gd), replace=False).mean() * 100
                          for _ in range(a.boot)])
        gm = gd.mean() * 100
        p95 = np.percentile(draws, 95)
        print(f"    {nm:6} OOS DAY-level {gm:+.1f}% (d={len(gd)})  vs random p95 {p95:+.1f}%  "
              f"({(draws < gm).mean()*100:.0f}th pct)  "
              f"{'** BEATS null' if gm > p95 else '(within noise)'}")

    print("\n  -- 4. VIX-overlay interaction (book already sizes on VIX>=60d median) --")
    sub = R[R["vixfav"].notna()]
    if len(sub) > 50:
        ov = pd.crosstab(sub["gate"], sub["vixfav"])
        print(f"    overlap: gate&vixfav {int(ov.loc[True, True]) if True in ov.index and True in ov.columns else 0}, "
              f"gate&!vixfav {int(ov.loc[True, False]) if True in ov.index and False in ov.columns else 0}, "
              f"!gate&vixfav {int(ov.loc[False, True]) if False in ov.index and True in ov.columns else 0}, "
              f"!gate&!vixfav {int(ov.loc[False, False]) if False in ov.index and False in ov.columns else 0}")
        for gv in (True, False):
            for vv in (True, False):
                c = sub[(sub["gate"] == gv) & (sub["vixfav"] == vv)]
                if len(c) >= 8:
                    print(_line(f"gate={gv} vixfav={vv}", c, N))

    print("\n  -- 5. sizing sim (multiplier on gated trades only) --")
    for k in (1.0, 1.25, 1.5, 2.0):
        for half, s in (("IS", R[R.date < SPLIT]), ("OOS", R[R.date >= SPLIT])):
            w = np.where(s["gate"], k, 1.0)
            pnl_w = (s["pnl"].to_numpy() * w).sum() / len(s)
            cap = w.mean()
            print(f"    x{k:<5} [{half:3}]  P&L/trade(1-lot units) {pnl_w*100:>+6.2f}  "
                  f"capital {cap:.3f}x  ->  per unit capital {pnl_w/cap*100:>+6.2f}")
    # concentration: how often do gated trades collide same-day?
    perday = R[R["gate"]].groupby("date").size()
    print(f"\n    same-day gated trade count: mean {perday.mean():.2f}  max {int(perday.max())}  "
          f"days with >=3 {int((perday >= 3).sum())}  -> at 2x a {int(perday.max())}-trade day "
          f"is {2*int(perday.max())}x base premium (check vs MAX_PORTFOLIO_RISK_PCT)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run(a)
