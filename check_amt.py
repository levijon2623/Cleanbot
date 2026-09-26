# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_amt.py
============

Auction Market Theory, tested with the same rigour that busted magnet theory:
every "level acts as support / magnet" claim is checked against the ticker's
UNCONDITIONAL intraday-range distribution (the drift-free baseline), not just
"did it touch."

Daily volume profile from historical/{T}.parquet 1-min bars (each minute's
volume spread across the price bins its [low, high] spans):
  POC  -- highest-volume price bin
  VAH / VAL -- expand from POC until 70% of the day's volume is inside
  IB_hi / IB_lo -- high / low of the first --ib-mins minutes
  + open / close / high / low

Hypotheses (--test):
  levels   -- does today's RTH price touch yesterday's POC / VAH / VAL more than
              a random level at the same distance/side would? (magnet control)
  open     -- open INSIDE vs ABOVE vs BELOW prior value area -> day range, close
              location, trend-day rate. The classic "open drives day type."
  eighty   -- the "80% rule": open outside prior VA, price re-enters VA and holds
              -> does it then traverse to the far side of VA?
  ib       -- price breaks the Initial Balance high/low in the first half ->
              does it extend / close in that direction? (trend-day tell)
  naked    -- a POC from >=2 sessions ago never revisited ("naked POC") --
              fill rate vs a distance-matched baseline.

Usage:
  python check_amt.py SPY,QQQ,IWM 2024-08-20 2026-08-21 --range --test open
  python check_amt.py SPY 2024-08-20 2026-08-21 --range --test levels
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from amt_profile import HIST, RTH_LO, RTH_HI, _load_1m, build_profiles, amt_open_map


def _minute_by_date(tk):
    px = _load_1m(tk)
    return {d: g for d, g in px.groupby("date")}, px


# ---- unconditional range baseline (drift-free control) --------------------
def _range_dist(px_by_date, from_mod):
    up, dn = [], []
    for d, g in px_by_date.items():
        gg = g[g["mod"] >= from_mod]
        if gg.empty:
            continue
        o = gg["c"].iloc[0]
        if o <= 0:
            continue
        up.append((gg["h"].max() - o) / o)
        dn.append((o - gg["l"].min()) / o)
    return np.array(up), np.array(dn)


def test_levels(args, tk, prof, px_by_date):
    up, dn = _range_dist(px_by_date, RTH_LO)
    P_up = lambda x: float(np.mean(up >= x))
    P_dn = lambda x: float(np.mean(dn >= x))
    prof = prof.set_index("date")
    dates = list(prof.index)
    res = {lv: {"hit": 0, "base": 0.0, "n": 0} for lv in ("poc", "vah", "val")}
    for i in range(1, len(dates)):
        prev, cur = prof.loc[dates[i - 1]], prof.loc[dates[i]]
        g = px_by_date.get(dates[i].date())
        if g is None or g.empty:
            continue
        o = g["c"].iloc[0]
        hi, lo = g["h"].max(), g["l"].min()
        for lv in ("poc", "vah", "val"):
            L = prev[lv]
            if not np.isfinite(L) or o <= 0:
                continue
            dist = abs(L - o) / o
            if dist < 3e-4:
                continue
            above = L > o
            hit = (hi >= L) if above else (lo <= L)
            base = P_up(dist) if above else P_dn(dist)
            res[lv]["hit"] += hit
            res[lv]["base"] += base
            res[lv]["n"] += 1
    print(f"\n  {tk}   PRIOR-DAY LEVEL TOUCH  (vs distance-matched unconditional range)")
    for lv in ("poc", "vah", "val"):
        r = res[lv]
        if r["n"] == 0:
            continue
        hr, br = r["hit"] / r["n"] * 100, r["base"] / r["n"] * 100
        print(f"     prior {lv.upper():4}  n={r['n']:>4}   touched {hr:>3.0f}%   baseline {br:>3.0f}%   "
              f"edge {hr - br:>+3.0f}pp")


def test_open(args, tk, prof, px_by_date):
    prof = prof.set_index("date")
    dates = list(prof.index)
    buckets = {"inside": [], "above": [], "below": []}
    for i in range(1, len(dates)):
        prev, cur = prof.loc[dates[i - 1]], prof.loc[dates[i]]
        o, c, hi, lo = cur["open"], cur["close"], cur["high"], cur["low"]
        if not np.isfinite(prev["vah"]) or o <= 0:
            continue
        loc = ("above" if o > prev["vah"] else "below" if o < prev["val"] else "inside")
        rng = (hi - lo) / o
        close_up = (c - o) / o
        # trend day proxy: |close-open| > 0.6 * range AND close in top/bottom 20% of range
        pos = (c - lo) / (hi - lo) if hi > lo else 0.5
        trend = abs(close_up) > 0.5 * rng and (pos > 0.8 or pos < 0.2)
        rev_to_va = (lo <= prev["vah"] and hi >= prev["val"])   # traded back into prior VA
        buckets[loc].append((rng, close_up, pos, trend, rev_to_va))
    print(f"\n  {tk}   OPEN vs PRIOR VALUE AREA")
    print(f"     {'open loc':9}{'n':>5}{'med range':>11}{'med close-open':>15}"
          f"{'trend-day%':>12}{'back-in-VA%':>13}")
    for loc in ("inside", "above", "below"):
        b = buckets[loc]
        if not b:
            continue
        a = np.array([x[:3] for x in b])
        td = np.mean([x[3] for x in b]) * 100
        rv = np.mean([x[4] for x in b]) * 100
        print(f"     {loc:9}{len(b):>5}{np.median(a[:,0])*100:>10.2f}%{np.median(a[:,1])*100:>+14.2f}%"
              f"{td:>11.0f}%{rv:>12.0f}%")


def test_ib(args, tk, prof, px_by_date):
    """Does breaking the IB by --ib-mins+30 predict the close direction BEYOND
    just 'price is near the top/bottom of the day so far'? Baseline = same
    close-side rate for days NOT (yet) broken but with price equally extended."""
    prof = prof.set_index("date")
    dates = list(prof.index)
    chk = RTH_LO + args.ib_mins + 30      # decision point (e.g. 11:00 for 60m IB)
    brk_up, brk_dn, unbrk_up, unbrk_dn = [], [], [], []
    for i in range(len(dates)):
        cur = prof.loc[dates[i]]
        g = px_by_date.get(dates[i].date())
        if g is None or not np.isfinite(cur["ib_hi"]):
            continue
        pre = g[g["mod"] <= chk]
        post = g[g["mod"] > chk]
        if pre.empty or post.empty:
            continue
        ib_hi, ib_lo, o, c = cur["ib_hi"], cur["ib_lo"], cur["open"], cur["close"]
        px_now = pre["c"].iloc[-1]
        rng_hi, rng_lo = pre["h"].max(), pre["l"].min()
        broke_up = pre[pre["mod"] > RTH_LO + args.ib_mins]["h"].max() > ib_hi
        broke_dn = pre[pre["mod"] > RTH_LO + args.ib_mins]["l"].min() < ib_lo
        # "extended up" = price is in the top 20% of the pre-chk range
        ext_up = (px_now - rng_lo) / (rng_hi - rng_lo) > 0.8 if rng_hi > rng_lo else False
        ext_dn = (px_now - rng_lo) / (rng_hi - rng_lo) < 0.2 if rng_hi > rng_lo else False
        if broke_up and not broke_dn:
            brk_up.append(c > px_now)
        elif broke_dn and not broke_up:
            brk_dn.append(c < px_now)
        elif ext_up and not broke_up:
            unbrk_up.append(c > px_now)
        elif ext_dn and not broke_dn:
            unbrk_dn.append(c < px_now)
    print(f"\n  {tk}   IB ({args.ib_mins}m) BREAK -> close continuation  (decision @ {chk//60}:{chk%60:02d})")
    if brk_up:
        print(f"     broke IB-hi     n={len(brk_up):>4}   closes higher still: {np.mean(brk_up)*100:>3.0f}%")
    if unbrk_up:
        print(f"     top of range,   n={len(unbrk_up):>4}   closes higher still: {np.mean(unbrk_up)*100:>3.0f}%  <- baseline (not broken)")
    if brk_dn:
        print(f"     broke IB-lo     n={len(brk_dn):>4}   closes lower still:  {np.mean(brk_dn)*100:>3.0f}%")
    if unbrk_dn:
        print(f"     bottom of range n={len(unbrk_dn):>4}   closes lower still:  {np.mean(unbrk_dn)*100:>3.0f}%  <- baseline (not broken)")


def test_eighty(args, tk, prof, px_by_date):
    prof = prof.set_index("date")
    dates = list(prof.index)
    traversed, held_out, n = 0, 0, 0
    for i in range(1, len(dates)):
        prev = prof.loc[dates[i - 1]]
        g = px_by_date.get(dates[i].date())
        if g is None or not np.isfinite(prev["vah"]):
            continue
        vah, val = prev["vah"], prev["val"]
        o = g["c"].iloc[0]
        if val <= o <= vah:
            continue                       # opened inside -> not an 80%-rule setup
        above = o > vah
        # did price trade back INTO the VA and spend >= 30 min there?
        inva = g[(g["c"] <= vah) & (g["c"] >= val)]
        if len(inva) < 30:
            held_out += 1; n += 1; continue
        first_in = inva["mod"].iloc[0]
        after = g[g["mod"] >= first_in]
        # traverse = reach the FAR edge of the VA
        far_hit = (after["l"].min() <= val) if above else (after["h"].max() >= vah)
        traversed += far_hit
        n += 1
    if n:
        print(f"\n  {tk}   80% RULE  (open outside prior VA)")
        print(f"     n={n}   re-entered VA & held >=30m then traversed to far edge: "
              f"{traversed}/{n - held_out} ({traversed/max(n-held_out,1)*100:.0f}%)   "
              f"| never re-entered VA: {held_out}/{n}")


def test_rules(args, tk, prof, px_by_date):
    """Do rules for this ticker perform differently split by open-vs-prior-VA?
    Rules come from config.RULES, or --rule-file <json> (to probe REJECTED /
    disabled candidates -- do any turn valid inside one AMT bucket?)."""
    import directional_flow_backtester as D
    if args.rule_file:
        with open(args.rule_file) as f:
            src = json.load(f)
        for r in src:
            r.setdefault("min_flow_pct", r.get("flow_pct"))
        tk_rules = [r for r in src if r["ticker"].upper() == tk]
    else:
        from config import RULES
        tk_rules = [r for r in RULES if r.get("enabled", True) and r["ticker"].upper() == tk]
    if not tk_rules:
        print(f"  {tk}: no rules"); return
    amt = amt_open_map(tk)
    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date
    gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk)
    trd = D.load_trend_regime(HIST, tk)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
               "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
    trigs = D.triggers_for(flow, tk)
    if not trigs:
        tkf, _ = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
    D.annotate_flow_pct(trigs, 60)
    tb = D._ticker_bars(tk)
    if tb is None or tb.empty:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}
    split = pd.Timestamp("2025-08-21").date()

    print(f"\n  {tk}   config rules split by OPEN vs PRIOR VA")
    for r in tk_rules:
        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
        by_loc = {"below_va": [], "above_va": [], "inside_va": [], "ALL": []}
        for t, thr in matched:
            loc = amt.get(t["date"])
            for _, _, _, dd, pnl in D.simulate_trigger(t, r["direction"].upper(), r.get("dte", [0, 1]),
                                                       r.get("time_stop_mins"), bbd, bbc,
                                                       only=(thr, float(r["target_roe"]), float(r["rr"]))):
                by_loc["ALL"].append((dd, pnl))
                if loc:
                    by_loc[loc].append((dd, pnl))
        print(f"    {r['name']}  ({r['direction']})")
        for loc in ("ALL", "below_va", "inside_va", "above_va"):
            v = by_loc[loc]
            if len(v) < 20:
                print(f"      {loc:10} n={len(v):>4}  (thin)"); continue
            oo = [p for d, p in v if d >= split]
            ii = [p for d, p in v if d < split]
            print(f"      {loc:10} n={len(v):>4}  IS {np.mean(ii)*100 if ii else float('nan'):>+6.1f}%  "
                  f"OOS {np.mean(oo)*100 if oo else float('nan'):>+6.1f}%")


TESTS = {"levels": test_levels, "open": test_open, "ib": test_ib, "eighty": test_eighty,
         "rules": test_rules}


def run(args):
    for tk in args.tickers:
        try:
            prof = build_profiles(tk, args.bin_pct, args.ib_mins, args.va_frac, force=args.rebuild)
        except FileNotFoundError:
            print(f"  {tk}: no historical/{tk}.parquet"); continue
        if args.dates_range:
            prof = prof[(prof["date"] >= args.lo) & (prof["date"] <= args.hi)]
        px_by_date, _ = _minute_by_date(tk)
        if args.test == "naked":
            print("  naked-POC test not yet implemented"); continue
        TESTS[args.test](args, tk, prof, px_by_date)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker")
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--range", dest="dates_range", action="store_true")
    ap.add_argument("--test", choices=list(TESTS) + ["naked"], default="open")
    ap.add_argument("--rule-file", help="--test rules: probe a JSON list of candidate rules instead of config.RULES")
    ap.add_argument("--bin-pct", type=float, default=0.0005, help="volume bin width as frac of price (default 0.05%%)")
    ap.add_argument("--ib-mins", type=int, default=60, help="Initial Balance window in minutes (default 60)")
    ap.add_argument("--va-frac", type=float, default=0.70, help="value-area volume fraction (default 0.70)")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    a.tickers = [t.strip().upper() for t in a.ticker.split(",") if t.strip()]
    if a.dates_range and len(a.dates) == 2:
        a.lo, a.hi = (pd.Timestamp(x) for x in sorted(a.dates))
    run(a)
