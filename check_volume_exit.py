# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_volume_exit.py
====================
EXIT ON THE VOLUME TELL, THEN RE-ENTER ON THE NEXT TRIGGER.

WHY THIS IS NOT THE SEVENTH FAILED EXIT FAMILY
    Six exit families have been tested here and one principle came out of all of
    them: anything that banks gains earlier helps a losing population and hurts a
    winning one. But every one of those six PERMANENTLY REDUCES the day's
    exposure -- trails, give-backs, tighter stops, scale-outs. The single
    intervention that ever helped (+5.2pp, the leg-in vertical) was the one that
    did NOT reduce the position.
    This one does not reduce it either. `sim_core.walk` sets `busy` to the EXIT
    minute, so closing early FREES THE SLOT and the next trigger of the day
    becomes takeable. Exit at a running high, re-enter if the flow fires again,
    ride the next leg. That is a re-entry cycle, not a haircut.

=====================  THE LOOK-AHEAD THIS FIXES  ========================
check_peak_bar ranked the peak bar's volume at 0.709 -- but it normalised by the
mean volume over [entry, EOD], i.e. INCLUDING THE FUTURE. That is fine for
describing where peaks sit and useless as a signal. Here the rank is CAUSAL:
at each new running extreme, the bar's volume is ranked only against the volume
of the PRIOR new-extreme bars of the same move. Nothing after the current minute
is touched. Expect the edge to shrink versus the descriptive number; that is the
look-ahead coming out, not a bug.

WHY ETF TAPE VOLUME AND NOT MBO
    The MBO pilot's value was showing the signal is not a one-vendor artifact:
    ETF 1m tape volume and the futures order book agree +0.76 trade-by-trade on
    which minute was heavy (check_volume_agree). That licenses building the
    detector on plain 1m ETF volume -- 638 days, every ticker, live-available,
    no $179/mo feed. MBO stays a validation instrument, not a dependency.

WHAT IS MEASURED
    Baseline  = the deployed policy, untouched.
    Test      = same policy + signal exit, with re-entry allowed by the guard.
    Reported per rule and pooled, IS/OOS split at 2025-08-21, on the `bot` fill
    model. A day-block bootstrap because power is set by DAYS.

Usage:
  python check_volume_exit.py
  python check_volume_exit.py --thresh 0.8 --min-prior 4
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

import sim_core

RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
SPLIT = pd.Timestamp("2025-08-21").date()


def und_bars(tk):
    d = pl.read_parquet(f"historical/{tk}.parquet",
                        columns=["start_time", "high", "low", "volume"]).to_pandas()
    et = (pd.to_datetime(d["start_time"], utc=True)
          .dt.tz_convert("America/New_York").dt.tz_localize(None))
    d["date"] = et.dt.date
    d["mod"] = (et.dt.hour * 60 + et.dt.minute).astype(int)
    d = d[(d["mod"] >= RTH_LO) & (d["mod"] <= RTH_HI)]
    for c in ("high", "low", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return {dt: (g["mod"].to_numpy(np.int32), g["high"].to_numpy(float),
                 g["low"].to_numpy(float), g["volume"].to_numpy(float))
            for dt, g in d.sort_values("mod").groupby("date")}


def build_sig(cand, bars, up, thresh, min_prior):
    """Per candidate, a bool array over the option path's minutes.

    Fires when the CURRENT minute sets a new running extreme of the underlying
    AND its volume outranks `thresh` of the PRIOR new-extreme bars of this move.
    Strictly causal: only bars at or before the current minute are consulted.
    """
    out = []
    for d, m, path in cand:
        mods = path[-1]
        arr = bars.get(d)
        if arr is None:
            out.append(None)
            continue
        bm, bh, bl, bv = arr
        sig = np.zeros(len(mods), bool)
        prior_v, cur = [], (-np.inf if up else np.inf)
        # index into the underlying bars, advanced in step with the option path
        for i, mm in enumerate(mods):
            j = int(np.searchsorted(bm, mm, side="left"))
            if j >= bm.size or bm[j] != mm:
                continue
            x = bh[j] if up else bl[j]
            if not ((up and x > cur) or (not up and x < cur)):
                continue
            cur = x
            v = bv[j]
            if len(prior_v) >= min_prior and np.isfinite(v):
                p = np.asarray(prior_v, float)
                p = p[np.isfinite(p)]
                if p.size >= min_prior:
                    rank = ((p < v).mean() + (p <= v).mean()) / 2.0
                    if rank >= thresh:
                        sig[i] = True
            if np.isfinite(v):
                prior_v.append(v)
        out.append(sig)
    return out


def agg(res):
    if not res:
        return dict(n=0, days=0, mean=np.nan, tot=np.nan)
    df = pd.DataFrame(res, columns=["date", "pnl"])
    return dict(n=len(df), days=df["date"].nunique(),
                mean=float(df["pnl"].mean() * 100),
                tot=float(df["pnl"].sum() * 100))


def split_res(res):
    a = [r for r in res if r[0] < SPLIT]
    b = [r for r in res if r[0] >= SPLIT]
    return agg(a), agg(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.75,
                    help="causal volume rank a new extreme must clear to fire")
    ap.add_argument("--min-prior", type=int, default=3,
                    help="new extremes required before the rule may fire at all")
    ap.add_argument("--fill", default="bot")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = sim_core.research_rules()
    print(f"  {len(rules)} research rules, fill={a.fill}, "
          f"thresh={a.thresh}, min_prior={a.min_prior}\n")

    rows = []
    tot_base, tot_test = [], []
    for rule in rules:
        tk = rule["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        cand = sim_core.build_candidates(D, rule, trigs=trigs)
        if not cand:
            continue
        eod_m = sim_core.eod_mod(rule)
        bars = und_bars(tk)
        up = rule["direction"] == "CALL"
        sigs = build_sig(cand, bars, up, a.thresh, a.min_prior)

        # policy_for() maps the RULE's trail_pct / target_roe / rr onto the
        # tp/stop/trail keys simulate() actually reads. Passing the raw rule
        # silently yields tp=stop=trail=None -- i.e. hold-to-EOD for BOTH arms,
        # with the deployed brackets never applied.
        pol = sim_core.policy_for(rule)
        base = sim_core.walk(cand, pol, eod_m, fill=a.fill)
        test = sim_core.walk(cand, pol, eod_m, fill=a.fill, sigs=sigs)
        tot_base += base
        tot_test += test
        bi, bo = split_res(base)
        ti, to = split_res(test)
        rows.append(dict(name=rule["name"],
                         b_n=bi["n"] + bo["n"], t_n=ti["n"] + to["n"],
                         b_is=bi["tot"], t_is=ti["tot"],
                         b_oos=bo["tot"], t_oos=to["tot"]))
        print(f"  {rule['name']:26} trades {bi['n']+bo['n']:>4} -> {ti['n']+to['n']:>4}   "
              f"IS {bi['tot']:>+8.1f} -> {ti['tot']:>+8.1f}   "
              f"OOS {bo['tot']:>+8.1f} -> {to['tot']:>+8.1f}", flush=True)

    if not rows:
        print("  nothing"); return
    R = pd.DataFrame(rows)
    bi, bo = split_res(tot_base)
    ti, to = split_res(tot_test)

    print(f"\n{'='*88}")
    print(f"  VOLUME-SIGNAL EXIT WITH RE-ENTRY  vs  THE DEPLOYED POLICY")
    print(f"{'='*88}")
    print(f"  {'':12} {'trades':>8} {'days':>6} {'total %':>10} {'mean %/trade':>14}")
    for lbl, s in (("BASE  IS", bi), ("TEST  IS", ti), ("BASE  OOS", bo), ("TEST  OOS", to)):
        print(f"  {lbl:12} {s['n']:>8} {s['days']:>6} {s['tot']:>+10.1f} {s['mean']:>+14.2f}")
    print(f"\n  IS  delta {ti['tot']-bi['tot']:+.1f}pp total   "
          f"({ti['n']-bi['n']:+d} trades)")
    print(f"  OOS delta {to['tot']-bo['tot']:+.1f}pp total   "
          f"({to['n']-bo['n']:+d} trades)")

    print(f"\n  PER-RULE OOS (the one that counts)")
    R["d_oos"] = R["t_oos"] - R["b_oos"]
    R = R.sort_values("d_oos", ascending=False)
    win = int((R["d_oos"] > 0).sum())
    for r in R.itertuples():
        print(f"    {r.name:26} {r.b_oos:>+8.1f} -> {r.t_oos:>+8.1f}   "
              f"{r.d_oos:>+7.1f}pp")
    print(f"    {win}/{len(R)} rules improved OOS")

    print(f"\n  HOW TO READ IT")
    print(f"  The trade COUNT rising is the mechanism working: an early exit frees")
    print(f"  the slot and the next trigger gets taken. If the count does not rise,")
    print(f"  the signal is firing too late in the day to allow a re-entry and this")
    print(f"  is just another haircut.")
    print(f"  Judge on OOS and on how many rules move together -- a single rule")
    print(f"  carrying the total is the failure mode this project keeps hitting.")


if __name__ == "__main__":
    main()
