# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_ema_trigger.py
====================
REPLACE THE FLOW SPIKE WITH THE EMA TURN ITSELF -- is the whale needed at all?

THE QUESTION
    check_ema_phase found the book fires 80% of its triggers in Phase 0 (no turn
    yet) or Phase 3 (trend already obvious), the two worst phases, and only 76
    candidates in Phase 2 (the 5/21 breakout) which was best on every metric.
    If Phase 2 is where the money is, stop waiting for a whale to happen to fire
    there and trigger on the TRANSITION INTO PHASE 2 directly.

THE FOUR ARMS -- everything except the TRIGGER is held constant
    A  FLOW spike that lands in Phase 2      (the 76, recomputed)
    B  EMA cross into Phase 2, no flow gate
    C  EMA cross + cumulative flow ALIGNED   (sign agrees, magnitude clears)
    D  EMA cross + cumulative flow OPPOSED   (sign disagrees)
    C vs D is what tests whether flow alignment carries anything. If they match,
    the cumulative-flow filter is decoration and B is the whole story.

THE TRIGGER, exactly as specified
    CALL: e5 crosses ABOVE e21 on this bar, while e9 is still BELOW e21
    PUT : e5 crosses BELOW e21 on this bar, while e9 is still ABOVE e21
    i.e. the cross itself, not merely being in the state.

🚨 THE LOOK-AHEAD THIS AVOIDS
    "A threshold based on the day's total" cannot be used: the day's total flow
    is not knowable at 11:00. The filter instead uses CUMULATIVE flow AS OF the
    trigger minute, thresholded by `annotate_flow_pct`, which derives its
    percentile from PRIOR DAYS ONLY and needs >=30 prior triggers. That is the
    same gate the deployed rules already pass, so arm C is not a new knob --
    it is the existing flow gate pointed at a different moment.

EVERY ARM USES THE SAME EXIT (trail 50%) so the comparison is about the TRIGGER
    alone. Note this re-prices arm A: the 76 in check_ema_phase used each rule's
    own policy, so its numbers here will differ and these are the comparable ones.

Usage:
  python check_ema_trigger.py --paper
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

import sim_core
from check_ema_phase import ema_map, mirror

SPLIT = pd.Timestamp("2025-08-21").date()
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
POL = {"name": "trail50", "kind": "trail", "trail": 0.50}


def ema_raw(tk, warm=21):
    """{date: (mods, e5, e9, e21)} -- 1m closes, causal by construction."""
    d = pl.read_parquet(f"historical/{tk}.parquet",
                        columns=["start_time", "close"]).to_pandas()
    et = (pd.to_datetime(d["start_time"], utc=True)
          .dt.tz_convert("America/New_York").dt.tz_localize(None))
    d["date"] = et.dt.date
    d["mod"] = (et.dt.hour * 60 + et.dt.minute).astype(int)
    d = d[(d["mod"] >= RTH_LO) & (d["mod"] <= RTH_HI)]
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    out = {}
    for dt, g in d.dropna(subset=["close"]).sort_values("mod").groupby("date"):
        c = g["close"].astype(float)
        # .copy(): to_numpy() can hand back a read-only view of the Series buffer
        e = [c.ewm(span=s, adjust=False).mean().to_numpy().copy() for s in (5, 9, 21)]
        for x in e:
            x[:warm] = np.nan             # warmup: all three equal the first close
        out[dt] = (g["mod"].to_numpy(np.int32), *e)
    return out


def flow_lookup(D, tk):
    """{(date, mod): cum_flow} plus {date: thr-dict} from the REAL triggers, so
    the percentile gate in force on that date can be reused verbatim."""
    from check_config_walkforward import _flow_for
    f = _flow_for(D, [tk])
    if f.empty:
        return {}, {}
    g = f[f["underlying_symbol"] == tk].copy()
    g["date"] = pd.to_datetime(g["minute_et"]).dt.date
    g["mod"] = (pd.to_datetime(g["minute_et"]).dt.hour * 60
                + pd.to_datetime(g["minute_et"]).dt.minute)
    cum = {(r.date, int(r.mod)): float(r.cum_flow) for r in g.itertuples()}
    trigs = D.triggers_for(f, tk)
    D.annotate_flow_pct(trigs, 60)
    thr = {}
    for t in trigs:
        if t.get("thr") and t["date"] not in thr:
            thr[t["date"]] = t["thr"]
    return cum, thr


def ema_triggers(tk, D, direction):
    """Synthetic trigger dicts at every cross INTO Phase 2, with the cumulative
    flow and the date's percentile ladder attached."""
    em = ema_raw(tk)
    cum, thr = flow_lookup(D, tk)
    up = direction == "CALL"
    out = []
    for d, (mods, e5, e9, e21) in em.items():
        for i in range(1, len(mods)):
            if not (np.isfinite(e5[i]) and np.isfinite(e21[i]) and np.isfinite(e5[i - 1])):
                continue
            if up:
                fire = (e5[i] > e21[i]) and (e9[i] < e21[i]) and (e5[i - 1] <= e21[i - 1])
            else:
                fire = (e5[i] < e21[i]) and (e9[i] > e21[i]) and (e5[i - 1] >= e21[i - 1])
            if not fire:
                continue
            m = int(mods[i])
            cf = cum.get((d, m))
            if cf is None:
                continue
            out.append({"date": d, "ts": pd.Timestamp(d) + pd.Timedelta(minutes=m),
                        "hour": m // 60, "dir": direction,
                        "abs_flow": abs(cf), "cum": cf,
                        "thr": thr.get(d)})
    return out


def stats(res, label):
    if not res:
        return f"  {label:34} (none)"
    v = np.array([p for _, p in res], float) * 100
    d = {x for x, _ in res}
    i = [p for x, p in res if x < SPLIT]
    o = [p for x, p in res if x >= SPLIT]
    return (f"  {label:34} {len(v):>6} {len(d):>6} "
            f"{(v <= -50).mean()*100:>7.1f}% {(v > 0).mean()*100:>6.1f}% "
            f"{np.median(v):>+9.1f} "
            f"{(np.sum(i)*100 if i else np.nan):>+9.0f} "
            f"{(np.sum(o)*100 if o else np.nan):>+9.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--pct", type=int, default=65)
    a = ap.parse_args()

    import directional_flow_backtester as D

    arms = {k: [] for k in ("A flow spike in Phase 2", "B EMA cross, no flow gate",
                            "C EMA cross + flow ALIGNED", "D EMA cross + flow OPPOSED")}
    counts = []
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dirn = rule["ticker"], rule["direction"]
        cap = sim_core.CUSHION_CAP.get(tk)
        eod = sim_core.eod_mod(rule)
        up = dirn == "CALL"

        # ---- A: the real flow triggers, kept only where the EMA phase is 2
        from check_config_walkforward import _flow_for
        f = _flow_for(D, [tk])
        if f.empty:
            continue
        rt = D.triggers_for(f, tk)
        D.annotate_flow_pct(rt, rule.get("flow_window_days", 60))
        em_ph = ema_map(tk)
        keep = []
        for t in rt:
            e = em_ph.get(t["date"])
            if e is None:
                continue
            bm, ph = e
            m = pd.Timestamp(t["ts"]).hour * 60 + pd.Timestamp(t["ts"]).minute
            j = int(np.searchsorted(bm, m))
            if j < bm.size and bm[j] == m and mirror(ph[j], up) == 2:
                keep.append(t)
        cA = sim_core.build_candidates(D, rule, trigs=keep)
        arms["A flow spike in Phase 2"] += sim_core.walk(
            cA, POL, eod, fill=a.fill, cush_cap=cap)

        # ---- B/C/D: the EMA cross
        et = ema_triggers(tk, D, dirn)
        want = 1 if up else -1
        openg = [{**t, "thr": {p: 0.0 for p in range(1, 100)}} for t in et]
        align = [t for t in et if np.sign(t["cum"]) == want and t.get("thr")]
        oppos = [{**t, "thr": {p: 0.0 for p in range(1, 100)}}
                 for t in et if np.sign(t["cum"]) == -want]
        for lbl, tg in (("B EMA cross, no flow gate", openg),
                        ("C EMA cross + flow ALIGNED", align),
                        ("D EMA cross + flow OPPOSED", oppos)):
            c = sim_core.build_candidates(D, rule, trigs=tg)
            arms[lbl] += sim_core.walk(c, POL, eod, fill=a.fill, cush_cap=cap)
        counts.append((rule["name"], len(keep), len(et), len(align), len(oppos)))
        print(f"    {rule['name']:24} flow-P2 {len(keep):>4}   "
              f"EMA crosses {len(et):>4} (aligned {len(align):>4} / "
              f"opposed {len(oppos):>4})", flush=True)

    print(f"\n{'='*104}")
    print(f"  EMA TURN AS THE TRIGGER vs THE WHALE SPIKE   (all arms: trail 50%, "
          f"fill={a.fill})")
    print(f"{'='*104}")
    print(f"  {'arm':34} {'n':>6} {'days':>6} {'loss50':>8} {'win':>7} "
          f"{'med ROE':>9} {'IS tot':>10} {'OOS tot':>10}")
    for k in ("A flow spike in Phase 2", "B EMA cross, no flow gate",
              "C EMA cross + flow ALIGNED", "D EMA cross + flow OPPOSED"):
        print(stats(arms[k], k))

    print(f"\n  HOW TO READ IT")
    print(f"  C vs D is the test of whether cumulative flow carries anything. If")
    print(f"  they are alike, the flow filter is decoration and B is the result.")
    print(f"  B vs A is the test of whether the whale is needed at all.")
    print(f"  Judge on DAYS and on IS/OOS agreement, not on the totals: a trigger")
    print(f"  that fires on a handful of days cannot be gated on, however good.")


if __name__ == "__main__":
    main()
