# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_ema_phase.py
==================
EMA(5/9/21) REVERSAL PHASE AT THE TRIGGER MINUTE -- which phase is least toxic?

THE FOUR PHASES (for a CALL; mirrored for a PUT)
    0 KNIFE        5 < 9 < 21   price falling, no turn yet
    1 EARLY TURN   9 < 5 < 21   fast has crossed, macro still down
    2 BREAKOUT     9 < 21 < 5   fast through the slow trend, 9 not yet
    3 LATE CONFIRM 21 < 9 < 5   the full stack, obvious to everyone

🚨 THESE FOUR DO NOT TILE THE STATE SPACE. Three EMAs have SIX orderings; the
   definitions above cover four. The two missing are 5 < 21 < 9 (fast lowest,
   mid highest) and 21 < 5 < 9 (all above the slow line but fast below mid -- a
   pullback inside an uptrend). They are binned as OTHER and reported, because
   silently dropping them would make the four phases look exhaustive when a
   material share of triggers sits outside them.

WHY THE POPULATION IS CANDIDATES, NOT TRIGGERS
    `loss50` is realised P&L <= -50% (check_skip_hunt's definition), so it needs
    a simulated exit and only exists for trades a rule would take. That is also
    the right unit for a GATEKEEPER: the gate would accept or reject a
    candidate. Each candidate is simulated independently -- the sequential guard
    is deliberately not applied, or the measurement would confound "is this
    phase good" with "was the slot free".

CAUSALITY AND WARMUP
    An EMA uses only past closes, so the phase at the trigger minute is knowable
    live. But ewm(adjust=False) emits a value from bar 0 where all three EMAs
    equal the first close, making the ordering meaningless; the first 21 bars
    are blanked. (Same class of bug as the MACD warmup and the flow-percentile
    warmup, which silently distorted the front of every backtest here.)

CONTROLS
    PLACEBO: phase labels shuffled WITHIN each day. Same marginal distribution,
    same day composition, no link to price. Any metric that separates on the
    real labels must be flat on these.
    COUNT THE CELLS: 5 bins x 4 metrics is 20 comparisons. A single starred bin
    is a thread, not a finding (METHODOLOGY 7).

Usage:
  python check_ema_phase.py
  python check_ema_phase.py --paper --mae-mins 15
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
PHASE = {0: "0 KNIFE (5<9<21)", 1: "1 EARLY TURN (9<5<21)",
         2: "2 BREAKOUT (9<21<5)", 3: "3 LATE CONFIRM (21<9<5)",
         4: "- other (2 orderings)"}


def ema_map(tk, warm=21):
    """{date: (mods, phase_call)} -- phase index per minute, for a CALL."""
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
        e5 = c.ewm(span=5, adjust=False).mean().to_numpy()
        e9 = c.ewm(span=9, adjust=False).mean().to_numpy()
        e21 = c.ewm(span=21, adjust=False).mean().to_numpy()
        # ALL SIX orderings are labelled distinctly. Collapsing the two the CALL
        # definitions do not use into one "other" bucket loses exactly the two a
        # PUT needs: reflecting a call phase reverses every inequality, and
        # CALL P1 (e9<e5<e21) reflects to e21<e5<e9 -- an ordering the call
        # scheme never names. Five of the nine rules are PUT rules, so binning
        # those as "other" mislabelled most of the sample.
        ph = np.full(len(c), -2, np.int8)
        ph[(e5 < e9) & (e9 < e21)] = 0      # A
        ph[(e5 < e21) & (e21 < e9)] = 1     # B
        ph[(e9 < e5) & (e5 < e21)] = 2      # C
        ph[(e9 < e21) & (e21 < e5)] = 3     # D
        ph[(e21 < e5) & (e5 < e9)] = 4      # E
        ph[(e21 < e9) & (e9 < e5)] = 5      # F
        ph[:warm] = -1                      # warmup: ordering is meaningless
        out[dt] = (g["mod"].to_numpy(np.int32), ph)
    return out


#: ordering code -> phase, per direction. A PUT's phase is the reflection of the
#: CALL's (every inequality reversed), which maps onto DIFFERENT orderings:
#:      CALL  P0=A  P1=C  P2=D  P3=F   other: B, E
#:      PUT   P0=F  P1=E  P2=B  P3=A   other: C, D
_CALL = {0: 0, 2: 1, 3: 2, 5: 3, 1: 4, 4: 4}
_PUT = {5: 0, 4: 1, 1: 2, 0: 3, 2: 4, 3: 4}


def mirror(code, up):
    if code < 0:
        return int(code)
    return (_CALL if up else _PUT)[int(code)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--mae-mins", type=int, default=15)
    ap.add_argument("--seed", type=int, default=47)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rows = []
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk = rule["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        cand = sim_core.build_candidates(D, rule, trigs=trigs)
        if not cand:
            continue
        pol = sim_core.policy_for(rule)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        up = rule["direction"] == "CALL"
        em = ema_map(tk)
        for d, m, path in cand:
            f = em.get(d)
            if f is None:
                continue
            bm, ph = f
            j = int(np.searchsorted(bm, m))
            if j >= bm.size or bm[j] != m or ph[j] < 0:
                continue
            e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
            if e_mid <= 0:
                continue
            pnl, xm, tag = sim_core.simulate(path, pol, eod, fill=a.fill,
                                             cush_cap=cap)
            entry = min(round(e_mid + 0.01, 2), round(e_ask, 2))
            k = int(np.searchsorted(mods, m + a.mae_mins, side="right"))
            b = np.asarray(bid[:max(k, 1)], float)
            b = b[np.isfinite(b) & (b > 0)]
            mae = ((b.min() / entry - 1.0) * 100) if b.size and entry > 0 else np.nan
            rows.append(dict(rule=rule["name"], date=d, dir=rule["direction"],
                             phase=mirror(ph[j], up), pnl=pnl * 100,
                             loss50=float(pnl <= -0.50), win=float(pnl > 0),
                             mae=mae))
        print(f"    {rule['name']} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  nothing"); return
    R["oos"] = R["date"] >= SPLIT
    R.to_parquet("_ema_phase.parquet", index=False)
    # PLACEBO: shuffle phase within each day -- same marginals, same days
    R["plc"] = R.groupby("date")["phase"].transform(
        lambda s: rng.permutation(s.to_numpy()))

    print(f"\n{'='*100}")
    print(f"  EMA 5/9/21 REVERSAL PHASE AT THE TRIGGER MINUTE")
    print(f"  n={len(R):,} candidates, {R['date'].nunique()} days, "
          f"MAE over the first {a.mae_mins}m on the option BID")
    print(f"{'='*100}")

    def block(col, title):
        print(f"\n  {title}")
        print(f"  {'phase':26} {'n':>6} {'days':>6} {'loss50':>8} "
              f"{'win':>7} {'med ROE':>9} {'med MAE':>9}")
        for p in (0, 1, 2, 3, 4):
            g = R[R[col] == p]
            if g.empty:
                continue
            print(f"  {PHASE[p]:26} {len(g):>6} {g['date'].nunique():>6} "
                  f"{g['loss50'].mean()*100:>7.1f}% {g['win'].mean()*100:>6.1f}% "
                  f"{g['pnl'].median():>+9.1f} {g['mae'].median():>+9.1f}")

    block("phase", "ALL (IS + OOS)")
    print(f"\n  * PLACEBO -- phase shuffled within day; these must be flat")
    print(f"  {'phase':26} {'n':>6} {'days':>6} {'loss50':>8} "
          f"{'win':>7} {'med ROE':>9} {'med MAE':>9}")
    for p in (0, 1, 2, 3, 4):
        g = R[R["plc"] == p]
        if g.empty:
            continue
        print(f"  {PHASE[p]:26} {len(g):>6} {g['date'].nunique():>6} "
              f"{g['loss50'].mean()*100:>7.1f}% {g['win'].mean()*100:>6.1f}% "
              f"{g['pnl'].median():>+9.1f} {g['mae'].median():>+9.1f}")

    print(f"\n  IS / OOS SPLIT on loss50 (the pre-committed target)")
    print(f"  {'phase':26} {'IS n':>6} {'IS loss50':>10} {'OOS n':>7} "
          f"{'OOS loss50':>11} {'agree?':>8}")
    for p in (0, 1, 2, 3, 4):
        i = R[(R["phase"] == p) & (~R["oos"])]
        o = R[(R["phase"] == p) & (R["oos"])]
        if len(i) < 10 or len(o) < 10:
            continue
        base_i = R[~R["oos"]]["loss50"].mean()
        base_o = R[R["oos"]]["loss50"].mean()
        ag = "yes" if np.sign(i["loss50"].mean() - base_i) == \
                      np.sign(o["loss50"].mean() - base_o) else "NO"
        print(f"  {PHASE[p]:26} {len(i):>6} {i['loss50'].mean()*100:>9.1f}% "
              f"{len(o):>7} {o['loss50'].mean()*100:>10.1f}% {ag:>8}")
    print(f"  (agree = the phase sits on the SAME side of the book's base rate")
    print(f"   in both halves; a phase that flips is noise, not a gate)")

    print(f"\n  HOW TO READ IT")
    print(f"  Base rates: loss50 {R['loss50'].mean()*100:.1f}%, "
          f"win {R['win'].mean()*100:.1f}%, median ROE {R['pnl'].median():+.1f}%")
    print(f"  20 cells are on the table (5 bins x 4 metrics), so a lone")
    print(f"  interesting bin is expected by chance. The hypothesis is directional")
    print(f"  -- phases 1 and 2 BEST, 0 and 3 worst -- so read the ORDERING, and")
    print(f"  only believe it if IS and OOS agree.")


if __name__ == "__main__":
    main()
