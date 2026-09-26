# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_reentry.py
================
DOES A SECOND LEG RECOVER THE TAIL THE TRAIL CUTS OFF?

THE PROBLEM THIS ATTACKS
    check_exit_bracketology settled that the exit cannot be improved: 18
    policies, and the chained walk-forward selection LOST to plain trail50
    (+3,334 vs +4,374) with a different winner almost every slice. The deeper
    finding was mechanical -- everything that protects the dead zone also caps
    the tail, and the tail is the whole edge. arm50/give25 lifts the win rate to
    50.5% and collapses total P&L from +3,988 to +495.

    So the tail cannot be protected on the way DOWN. The remaining question is
    whether it can be RE-ACQUIRED. GLD on 2026-09-18 is the motivating case:
    entered 399.24, trailed out at 398.14 -- the session low -- and GLD closed
    the day at 403.15 with that same contract bidding $3.90, +225% from entry.

    And if a second leg works, it earns a second prize: a mechanical re-entry
    makes a TIGHTER initial stop affordable, because being shaken out stops
    being terminal. Model C tests that directly.

THE THREE GATES (all must hold)
    REVISIT      the underlying returns to the original entry spot before 15:00
    VELOCITY     at the re-cross minute, the 1m bar range is in the top quartile
                 of its trailing 30m window OR 1m volume > 1.5x its trailing 30m
                 mean -- filtering slow, low-volume drifts back to breakeven
    FLOW         cumulative net options flow never turned against the original
                 direction during the drawdown. Flow that held while price fell
                 says shakeout; flow that flipped says the thesis died.

🚨 A JUDGEMENT CALL IN THE FLOW GATE, STATED NOT BURIED
    "Aligned with the trade direction" cannot mean sign(cum_flow) matches the
    direction: check_flow_align established that CALL triggers routinely fire on
    NEGATIVE cumulative flow (the trigger is an EMA CROSSOVER, not a level), so
    that reading would reject nearly every CALL before testing anything. It is
    implemented as the flow not having moved AGAINST the trade relative to where
    it stood at entry -- for a CALL, cum_flow at the re-cross >= cum_flow at
    entry. `--flow-gate sign` selects the stricter literal reading for contrast.

🚨 CAUSALITY AND THE SEQUENTIAL SLOT
    Every gate reads data at or before the re-cross minute; the rolling quartile
    and mean use trailing windows with the current minute EXCLUDED (shift 1).
    A re-entry OCCUPIES the one-position-per-ticker slot, so it can block a
    later trigger the baseline would have taken -- that cost is real and is
    carried, not netted out. The second leg buys the SAME contract, which is
    approximately ATM again by construction since spot has returned to the entry
    level.

MODELS
    A  baseline          trail50, exactly as deployed
    B  re-entry          trail50 + a second leg when all three gates pass
    C  tight + re-entry  trail`--tight` on leg one, trail50 on leg two

PRE-COMMITTED CRITERIA -- fixed before the first run
    R1  B total ROE > A total ROE
    R2  B's advantage is in the TAIL: B's p95 and mean-winner both exceed A's
    R3  >= 6 of 9 rules improve under B
    R4  C beats A, i.e. the re-entry genuinely pays for a tighter stop
    R5  the gates are selective: fewer than half of stopped-out trades qualify
        (a gate that admits everything is not a gate)

Usage:
  python check_reentry.py --paper
  python check_reentry.py --paper --tight 0.30 --flow-gate sign
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
RTH0, RTH1, CUTOFF = 570, 955, 900          # 15:00 = 900


def bars(tk):
    df = (pl.scan_parquet(f"historical/{tk}.parquet")
          .select("date", "minute_et", "close", "high", "low", "volume")
          .collect().to_pandas())
    t = pd.to_datetime(df["minute_et"])
    df["mod"] = t.dt.hour * 60 + t.dt.minute
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["mod"] >= RTH0) & (df["mod"] <= RTH1)].sort_values(["date", "mod"])
    out = {}
    for d, g in df.groupby("date"):
        g = g.copy()
        rng = g["high"] - g["low"]
        # trailing windows, current minute EXCLUDED
        g["rng_q75"] = rng.rolling(30, min_periods=10).quantile(0.75).shift(1)
        g["vol_avg"] = g["volume"].rolling(30, min_periods=10).mean().shift(1)
        g["rng"] = rng
        out[d] = g.set_index("mod")[["close", "high", "low", "volume",
                                     "rng", "rng_q75", "vol_avg"]]
    return out


def flows(D, tk):
    from check_config_walkforward import _flow_for
    f = _flow_for(D, [tk])
    if f.empty:
        return {}
    g = f[f["underlying_symbol"] == tk].copy()
    ts = pd.to_datetime(g["minute_et"])
    g["date"] = ts.dt.date
    g["mod"] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)
    return {d: dict(zip(x["mod"], x["cum_flow"].astype(float)))
            for d, x in g.groupby("date")}


def atm_path(D, bbc, bbd, d, ts, direction, dtes, spot):
    """A FRESH contract picked at the re-cross, not the one leg one held.

    Mirrors build_candidates' own payload construction (sim_core.py:492-517) --
    same 3-minute quote-staleness window, same mid convention, same $0.50 floor,
    same 3-bar minimum -- so the only difference from leg one is WHICH contract.
    Reuses D.pick_contract rather than re-deriving a picker.

    Matters because leg one's contract has decayed and moved relative to spot
    during the drawdown, while a fresh pick is ATM again by construction. Which
    of those the bot should buy on a re-entry is a real choice, not a detail.
    """
    day = bbd.get(d)
    if day is None:
        return None
    for dd in dtes:
        cid = D.pick_contract(day, ts, direction, dd, spot)
        if cid is not None:
            break
    else:
        return None
    if cid is None or cid not in bbc:
        return None
    ent = bbc[cid]
    er = ent[(ent["minute_et"] <= ts)
             & (ent["minute_et"] >= ts - pd.Timedelta(minutes=3))]
    if er.empty:
        return None
    er = er.iloc[-1]
    b, k = float(er["bid_close"]), float(er["ask_close"])
    mid = (b + k) / 2.0 if b > 0 else float(er["close"])
    if mid < 0.50:
        return None
    fwd = ent[ent["minute_et"] > ts].sort_values("minute_et")
    if len(fwd) < 3:
        return None
    pm = fwd["minute_et"]
    return (mid, k if k > 0 else mid,
            fwd["close"].to_numpy(float), fwd["high"].to_numpy(float),
            fwd["low"].to_numpy(float), fwd["bid_close"].to_numpy(float),
            fwd["ask_close"].to_numpy(float),
            (pm.dt.hour.values * 60 + pm.dt.minute.values).astype(int))


def slice_path(path, mod):
    """Re-form the sim payload starting at `mod`. Entry = that minute's mid."""
    mid0, ask0, cl, hi, lo, bid, ask, mods = path
    i = int(np.searchsorted(mods, mod))
    if i >= len(mods) - 3:
        return None
    b, a = bid[i], ask[i]
    if not np.isfinite(a) or a <= 0:
        return None
    m = (b + a) / 2 if b > 0 else a
    if m < 0.50:                      # the live entry floor applies to leg two
        return None
    s = slice(i + 1, None)
    return (m, a, cl[s], hi[s], lo[s], bid[s], ask[s], mods[s])


def find_reentry(B, F, d, dirn, entry_spot, from_mod, entry_cum, gate):
    """First minute >= from_mod passing all three gates. -> (mod, why) or None."""
    g = B.get(d)
    if g is None:
        return None, "no bars"
    cum = F.get(d, {})
    want_up = (dirn == "CALL")
    for m in range(int(from_mod), CUTOFF + 1):
        if m not in g.index:
            continue
        r = g.loc[m]
        # GATE 1: the revisit
        crossed = (r["high"] >= entry_spot) if want_up else (r["low"] <= entry_spot)
        if not crossed:
            continue
        # GATE 2: velocity / volume
        fast = (np.isfinite(r["rng_q75"]) and r["rng"] >= r["rng_q75"])
        heavy = (np.isfinite(r["vol_avg"]) and r["vol_avg"] > 0
                 and r["volume"] > 1.5 * r["vol_avg"])
        if not (fast or heavy):
            continue
        # GATE 3: the flow never turned against the trade
        c = cum.get(m)
        if c is None or entry_cum is None:
            continue
        if gate == "sign":
            ok = (c > 0) if want_up else (c < 0)
        else:
            ok = (c >= entry_cum) if want_up else (c <= entry_cum)
        if not ok:
            continue
        return m, "ok"
    return None, "no qualifying revisit"


def stat(rows):
    if not rows:
        return dict(n=0, tot=0.0, med=np.nan, p95=np.nan, avgwin=np.nan,
                    loss50=np.nan, win=np.nan, days=0)
    v = np.array([r["pnl"] for r in rows], float)
    w = v[v > 0]
    return dict(n=len(v), tot=float(v.sum()), med=float(np.median(v)),
                p95=float(np.percentile(v, 95)),
                avgwin=float(w.mean()) if len(w) else 0.0,
                loss50=float((v <= -50).mean() * 100),
                win=float((v > 0).mean() * 100),
                days=len({r["date"] for r in rows}))


HDR = (f"  {'model':22} {'n':>6} {'days':>5} {'total':>10} {'medROE':>8} "
       f"{'avg win':>9} {'p95':>9} {'loss50':>8} {'win':>7}")


def line(s, label):
    if not s["n"]:
        return f"  {label:22} {'(none)':>6}"
    return (f"  {label:22} {s['n']:>6} {s['days']:>5} {s['tot']:>+10.0f} "
            f"{s['med']:>+8.1f} {s['avgwin']:>+9.1f} {s['p95']:>+9.1f} "
            f"{s['loss50']:>7.1f}% {s['win']:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--tight", type=float, default=0.30,
                    help="leg-one trail for model C")
    ap.add_argument("--flow-gate", choices=("relative", "sign"),
                    default="relative")
    ap.add_argument("--leg2", choices=("same", "atm"), default="same",
                    help="re-enter the SAME contract, or pick a fresh ATM one")
    a = ap.parse_args()

    import directional_flow_backtester as D
    POL = dict(A=dict(name="trail50", kind="trail", trail=0.50),
               C=dict(name=f"trail{int(a.tight*100)}", kind="trail",
                      trail=a.tight))
    LEG2 = dict(name="trail50", kind="trail", trail=0.50)

    out = {k: [] for k in ("A", "B1", "B2", "C1", "C2")}
    gate_stats = dict(stopped=0, revisit=0, qualified=0, entered=0)
    per_rule = {}

    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dirn = rule["ticker"], rule["direction"]
        meta = []
        cand = sim_core.build_candidates(D, rule, meta_out=meta)
        if not cand:
            continue
        B, F = bars(tk), flows(D, tk)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        bbc = bbd = None
        if a.leg2 == "atm":
            tb = D._ticker_bars(tk)
            bbc = {c: g.sort_values("minute_et")
                   for c, g in tb.groupby("option_chain_id")}
            bbd = {d: g for d, g in tb.groupby("date")}
        per_rule[rule["name"]] = {"A": [], "B": [], "C": []}

        for model in ("A", "B", "C"):
            # `busy` is a MINUTE-OF-DAY and MUST reset each session. Carrying it
            # across days silently drops every candidate earlier in the clock
            # than the previous day's exit -- it cut the baseline from 349
            # trades to 30 before this was caught.
            busy, busy_day = -1, None
            pol1 = POL["C"] if model == "C" else POL["A"]
            for (d, m, path), mt in zip(cand, meta):
                if d != busy_day:
                    busy, busy_day = -1, d
                if m < busy:
                    continue
                pnl, xm, tag = sim_core.simulate(path, pol1, eod,
                                                 fill=a.fill, cush_cap=cap)
                r1 = dict(ticker=tk, rule=rule["name"], date=d, pnl=pnl * 100,
                          leg=1, tag=tag)
                busy = int(xm)
                if model == "A":
                    out["A"].append(r1)
                    per_rule[rule["name"]]["A"].append(r1)
                else:
                    out[f"{model}1"].append(r1)
                    per_rule[rule["name"]][model].append(r1)
                if model == "A" or pnl >= 0 or tag not in ("stop", "trail"):
                    continue
                if model == "B":
                    gate_stats["stopped"] += 1
                spot0 = mt.get("spot")
                cum0 = F.get(d, {}).get(int(m))
                if spot0 is None:
                    continue
                rm, _why = find_reentry(B, F, d, dirn, float(spot0),
                                        int(xm), cum0, a.flow_gate)
                if rm is None:
                    continue
                if model == "B":
                    gate_stats["qualified"] += 1
                if a.leg2 == "atm":
                    spot_rm = B[d].loc[rm, "close"] if rm in B[d].index else None
                    p2 = (atm_path(D, bbc, bbd, d,
                                   pd.Timestamp(d) + pd.Timedelta(minutes=int(rm)),
                                   dirn, rule.get("dte", [0, 1]), float(spot_rm))
                          if spot_rm is not None else None)
                else:
                    p2 = slice_path(path, rm)
                if p2 is None:
                    continue
                pnl2, xm2, tag2 = sim_core.simulate(p2, LEG2, eod,
                                                    fill=a.fill, cush_cap=cap)
                r2 = dict(ticker=tk, rule=rule["name"], date=d,
                          pnl=pnl2 * 100, leg=2, tag=tag2)
                out[f"{model}2"].append(r2)
                per_rule[rule["name"]][model].append(r2)
                busy = int(xm2)
                if model == "B":
                    gate_stats["entered"] += 1
        print(f"    {rule['name']} done", flush=True)

    A = stat(out["A"])
    Bt = stat(out["B1"] + out["B2"])
    Ct = stat(out["C1"] + out["C2"])
    print(f"\n{'='*100}")
    print(f"  1. THE THREE MODELS  (flow gate: {a.flow_gate}, "
          f"model C leg-one trail {a.tight:.0%})")
    print(f"{'='*100}")
    print(HDR)
    print(line(A, "A  trail50 baseline"))
    print(line(Bt, "B  trail50 + re-entry"))
    print(line(Ct, f"C  trail{int(a.tight*100)} + re-entry"))
    print(f"\n  {'':22} {'B - A':>17} {Bt['tot']-A['tot']:>+10.0f}")
    print(f"  {'':22} {'C - A':>17} {Ct['tot']-A['tot']:>+10.0f}")

    print(f"\n{'='*100}")
    print(f"  2. THE SECOND LEG ALONE -- is it a good trade on its own?")
    print(f"{'='*100}")
    print(HDR)
    print(line(stat(out["B1"]), "B leg 1 (stopped out)"))
    print(line(stat(out["B2"]), "B leg 2 (re-entries)"))
    print(line(stat(out["C2"]), "C leg 2 (re-entries)"))

    print(f"\n{'='*100}")
    print(f"  3. R5 -- ARE THE GATES SELECTIVE?")
    print(f"{'='*100}")
    s = gate_stats
    q = (s["qualified"] / s["stopped"] * 100) if s["stopped"] else 0
    print(f"  losing stop/trail exits      {s['stopped']:>6}")
    print(f"  passed all three gates       {s['qualified']:>6}  ({q:.1f}%)")
    print(f"  actually re-entered          {s['entered']:>6}  "
          f"(rest blocked by the $0.50 floor or <3 bars left)")
    print(f"\n  R5 gates selective (<50%)    {'PASS' if q < 50 else 'FAIL'}")

    print(f"\n{'='*100}")
    print(f"  4. R3 -- PER RULE")
    print(f"{'='*100}")
    print(f"  {'rule':24} {'A total':>10} {'B total':>10} {'delta':>9} "
          f"{'C total':>10} {'delta':>9}")
    nb = nc = 0
    for nm, v in per_rule.items():
        ta = sum(r["pnl"] for r in v["A"])
        tb = sum(r["pnl"] for r in v["B"])
        tc = sum(r["pnl"] for r in v["C"])
        nb += int(tb > ta); nc += int(tc > ta)
        print(f"  {nm:24} {ta:>+10.0f} {tb:>+10.0f} {tb-ta:>+9.0f} "
              f"{tc:>+10.0f} {tc-ta:>+9.0f}")
    n = len(per_rule)
    print(f"\n  R3 B improves {nb}/{n} rules  {'PASS' if nb >= 6 else 'FAIL'}")
    print(f"     C improves {nc}/{n} rules")

    print(f"\n{'='*100}")
    print(f"  SCORECARD")
    print(f"{'='*100}")
    print(f"  R1 B total > A total          {Bt['tot']:>+9.0f} vs {A['tot']:>+9.0f}  "
          f"{'PASS' if Bt['tot'] > A['tot'] else 'FAIL'}")
    t2 = (Bt['p95'] > A['p95']) and (Bt['avgwin'] > A['avgwin'])
    print(f"  R2 B's edge is in the tail    p95 {Bt['p95']:+.1f} vs {A['p95']:+.1f}, "
          f"avgwin {Bt['avgwin']:+.1f} vs {A['avgwin']:+.1f}  "
          f"{'PASS' if t2 else 'FAIL'}")
    print(f"  R4 C total > A total          {Ct['tot']:>+9.0f} vs {A['tot']:>+9.0f}  "
          f"{'PASS' if Ct['tot'] > A['tot'] else 'FAIL'}")
    print(f"  R3/R5 -- see blocks above")
    print(f"\n  A re-entry OCCUPIES the sequential slot, so B and C can miss a")
    print(f"  later trigger A took. That cost is inside these totals.")


if __name__ == "__main__":
    main()
