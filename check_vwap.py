# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_vwap.py
=============
DOES THE TRIGGER CARE WHERE PRICE SITS RELATIVE TO INTRADAY VWAP?

WHY THIS AND NOT THE WHOLE INTUITION
    The live IWM read was a confluence: prior-day close, morning lows, the
    overnight range, peers rallying, VWAP rejected then reclaimed, and a flow
    spike. Six conditions on 503 sessions would fire a handful of times with six
    thresholds to choose -- a guaranteed beautiful backtest and nothing else.
    So this tests ONE piece, the way `amt_open`, `dmi_confirm` and
    `skip_macro_am` were each tested one at a time.

    VWAP is the piece with NO prior: it is not mentioned anywhere in config.py,
    and check_volume_signal looked at underlying volume rather than VWAP
    position. The support/level half of the intuition is already encoded --
    `amt_open` is exactly "where did we open relative to prior-day value", and
    it is deployed on SPY (below_va), QQQ (exclude above_va) and AVGO
    (inside_va), while the general "levels act as support" claim was busted by
    check_amt.

PRE-COMMITTED HYPOTHESIS -- stated before the run, so the argmax cannot become
the finding:
    H: a trigger firing while price is BELOW VWAP (for a CALL; above, for a PUT)
       outperforms one firing on the other side -- i.e. the trigger is better at
       buying a dip against VWAP than chasing an extension away from it.
    That is the directional read of the observation: price fell THROUGH VWAP,
    and the entry was the reclaim.
    The other variants below are EXPLORATORY and flagged as such.

🚨 A MAGNITUDE FLOOR, BECAUSE SIGN TESTS ARE NOT CRITERIA
    check_flow_spike pre-committed five sign tests and passed four of them on
    +0.34bp -- one cent on a $285 name. Here every criterion is in ROE per
    trade with an explicit floor, and the placebo is a RANDOM SPLIT of the same
    trade count, so "fewer trades looks better" cannot pass for a finding.

VWAP IS COMPUTED, NOT ASSUMED
    cum(typical x volume) / cum(volume), typical = (h+l+c)/3, anchored at the
    09:30 open from historical/{T}.parquet. Identical convention to live_state's
    _vwap_from_bars, so the chart you traded from and this study agree.

PRE-COMMITTED CRITERIA
    V1  the H variant beats baseline on ROE PER TRADE by >= 5.0
    V2  it also beats baseline on TOTAL ROE (so it is not just trading less)
    V3  IS and OOS agree in sign
    V4  it beats a random split of the same size (100 draws, 95th pct)
    V5  >= 6 of 9 rules improve

RESULT -- 2026-09-21. NULL. Four of five criteria fail, and the built-in
diagnostic fires: the hypothesis and ITS MIRROR both sit on top of baseline.

    variant                n     total   per trade    IS/tr   OOS/tr
    baseline (all)       392    +3,785       +9.7      +4.5    +14.8
    AGAINST vwap (H)     244    +2,535      +10.4     +12.6     +8.3
    WITH vwap            244    +2,026       +8.3      -5.4    +25.8
    near vwap (±15bp)    147      +323       +2.2      -1.9     +7.4
    reclaim <=15m        147      +184       +1.2      -1.1     +4.6

    H beats baseline by +0.7 ROE/trade and sits at the 63rd PERCENTILE of a
    random split of the same 244 trades (p50 +9.4, p95 +14.3). That is the
    middle of the noise band. Its mirror is +8.3 -- also baseline. When a split
    and its complement are both indistinguishable from the whole, the variable
    carries nothing; the pre-written note at the foot of this file says exactly
    that, and it is what happened.

    🚨 THE VARIANT CLOSEST TO THE ORIGINATING OBSERVATION IS THE WORST.
    The live read was "price fell THROUGH vwap and the entry was the reclaim".
    `reclaim <=15m` scores +1.2/trade against baseline's +9.7 -- the weakest
    of all five. `near vwap` is second worst at +2.2. Whatever made that trade
    work, proximity to VWAP and the reclaim were not it.

    Per rule 4 of 9 improve. IWM -- the ticker the observation came from -- is
    one of them (+252), but SPY -579, QQQ -483 and META -649 are larger, and
    1-of-9 on the originating ticker is what chance produces.

    CONSEQUENCE: VWAP stays off config.py. The support/level half of the same
    intuition remains encoded as `amt_open`, which DID survive testing.

Usage:
  python check_vwap.py --paper
  python check_vwap.py --paper --near 10
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import check_reentry as RE          # verified RTH bar loader (has volume)
import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
SEED = 20260921
FLOOR = 5.0                          # ROE per trade, the V1 bar


def vwap_series(g):
    """{mod: session vwap}, anchored at the first RTH bar. Same convention as
    live_state._vwap_from_bars, so the chart and this study cannot disagree."""
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    v = g["volume"].astype(float)
    cv = v.cumsum()
    cpv = (tp * v).cumsum()
    out = (cpv / cv.replace(0, np.nan))
    return out


def stat(rows):
    if not rows:
        return dict(n=0, tot=0.0, per=np.nan, win=np.nan)
    a = np.array([r["pnl"] for r in rows], float)
    return dict(n=len(a), tot=float(a.sum()), per=float(a.mean()),
                win=float((a > 0).mean() * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--near", type=float, default=15.0,
                    help="bp band around VWAP for the 'near' variant")
    ap.add_argument("--reclaim", type=int, default=15,
                    help="minutes: crossed VWAP toward the trade this recently")
    a = ap.parse_args()

    import directional_flow_backtester as D
    rows, per_rule = [], {}

    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dirn = rule["ticker"], rule["direction"]
        meta = []
        cand = sim_core.build_candidates(D, rule, meta_out=meta)
        if not cand:
            continue
        B = RE.bars(tk)
        VW = {d: vwap_series(g) for d, g in B.items()}
        pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        want_above = (dirn == "PUT")     # a PUT "against VWAP" means price ABOVE

        feats = []
        for (d, m, _p), mt in zip(cand, meta):
            g, vw = B.get(d), VW.get(d)
            f = dict(vbp=np.nan, recl=np.nan)
            if g is not None and vw is not None and m in g.index:
                v = vw.get(m, np.nan)
                if np.isfinite(v) and v > 0:
                    px = float(g.loc[m, "close"])
                    # signed AGAINST the trade: positive = price is on the side
                    # the trade is fading (below VWAP for a CALL)
                    raw = (px - float(v)) / float(v) * 1e4
                    f["vbp"] = raw if want_above else -raw
                    # did price cross VWAP toward the trade recently?
                    back = [q for q in range(max(m - a.reclaim, 570), m)
                            if q in g.index and q in vw.index]
                    if back:
                        side = [(float(g.loc[q, "close"]) - float(vw[q])) > 0
                                for q in back if np.isfinite(vw[q])]
                        now_ab = (px - float(v)) > 0
                        f["recl"] = bool(side) and any(s != now_ab for s in side)
            feats.append(f)

        variants = {
            "baseline (all)": lambda f: True,
            "AGAINST vwap (H)": lambda f: np.isfinite(f["vbp"]) and f["vbp"] > 0,
            "WITH vwap": lambda f: np.isfinite(f["vbp"]) and f["vbp"] < 0,
            f"near vwap (±{a.near:g}bp)":
                lambda f: np.isfinite(f["vbp"]) and abs(f["vbp"]) <= a.near,
            f"reclaim ≤{a.reclaim}m": lambda f: f["recl"] is True,
        }
        per_rule[rule["name"]] = {}
        for name, fn in variants.items():
            keep = [c for c, f in zip(cand, feats) if fn(f)]
            r = sim_core.walk(keep, pol, eod, fill=a.fill, cush_cap=cap)
            recs = [dict(rule=rule["name"], date=d, pnl=p * 100) for d, p in r]
            per_rule[rule["name"]][name] = recs
            for x in recs:
                rows.append(dict(variant=name, **x))
        print(f"    {rule['name']} done  "
              f"(vwap known on {np.isfinite([f['vbp'] for f in feats]).mean()*100:.0f}%"
              f" of candidates)", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  nothing"); return
    R["half"] = np.where(R["date"] <= SPLIT, "IS", "OOS")
    R.to_parquet("_vwap.parquet", index=False)
    names = list(dict.fromkeys(R["variant"]))

    print(f"\n{'='*104}")
    print(f"  1. THE BOOK UNDER EACH VWAP VARIANT  (sequentially walked)")
    print(f"{'='*104}")
    print(f"  {'variant':24} {'n':>6} {'total':>10} {'per trade':>11} "
          f"{'win':>7} {'IS/tr':>9} {'OOS/tr':>9}")
    base = None
    for nm in names:
        g = R[R["variant"] == nm]
        i_, o_ = g[g["half"] == "IS"]["pnl"], g[g["half"] == "OOS"]["pnl"]
        s = dict(n=len(g), tot=g["pnl"].sum(), per=g["pnl"].mean(),
                 win=(g["pnl"] > 0).mean() * 100)
        if base is None:
            base = s
        print(f"  {nm:24} {s['n']:>6} {s['tot']:>+10.0f} {s['per']:>+11.1f} "
              f"{s['win']:>6.1f}% {i_.mean():>+9.1f} {o_.mean():>+9.1f}")

    H = "AGAINST vwap (H)"
    h = R[R["variant"] == H]
    hb = R[R["variant"] == "WITH vwap"]

    print(f"\n{'='*104}")
    print(f"  2. V4 -- IS IT BETTER THAN A RANDOM SPLIT OF THE SAME SIZE?")
    print(f"     Taking fewer trades changes the mean by chance alone. The")
    print(f"     placebo draws {len(h)} trades at random from the baseline, 100x.")
    print(f"{'='*104}")
    b = R[R["variant"] == names[0]]["pnl"].to_numpy()
    rng = np.random.default_rng(SEED)
    draws = np.array([rng.choice(b, size=min(len(h), len(b)),
                                 replace=False).mean() for _ in range(100)])
    print(f"  H variant per-trade      {h['pnl'].mean():>+8.1f}")
    print(f"  random split p50         {np.percentile(draws, 50):>+8.1f}")
    print(f"  random split p95         {np.percentile(draws, 95):>+8.1f}")
    print(f"  H percentile vs placebo  {(draws < h['pnl'].mean()).mean()*100:>7.0f}%")

    print(f"\n{'='*104}")
    print(f"  3. V5 -- PER RULE  (H variant vs baseline, total ROE)")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'baseline':>12} {'AGAINST vwap':>14} {'delta':>9}")
    nimp = 0
    for nm, v in per_rule.items():
        tb = sum(x["pnl"] for x in v[names[0]])
        th = sum(x["pnl"] for x in v[H])
        nimp += int(th > tb)
        print(f"  {nm:24} {tb:>+12.0f} {th:>+14.0f} {th-tb:>+9.0f}")

    print(f"\n{'='*104}")
    print(f"  SCORECARD  (pre-committed, with a magnitude floor)")
    print(f"{'='*104}")
    m = lambda ok: "PASS" if ok else "FAIL"
    dper = h["pnl"].mean() - base["per"]
    print(f"  V1  per-trade beats baseline by >= {FLOOR:.0f}   "
          f"{dper:>+7.1f}   {m(dper >= FLOOR)}")
    print(f"  V2  total also improves           "
          f"{h['pnl'].sum() - base['tot']:>+7.0f}   "
          f"{m(h['pnl'].sum() > base['tot'])}")
    isv = h[h['half'] == 'IS']['pnl'].mean()
    osv = h[h['half'] == 'OOS']['pnl'].mean()
    print(f"  V3  IS/OOS agree in sign     {isv:>+6.1f}/{osv:>+6.1f}   "
          f"{m((isv > 0) == (osv > 0))}")
    print(f"  V4  beats the random split p95     "
          f"{h['pnl'].mean():>+7.1f}   {m(h['pnl'].mean() > np.percentile(draws, 95))}")
    print(f"  V5  rules improved                 {nimp:>4}/{len(per_rule)}   "
          f"{m(nimp >= 6)}")
    print(f"\n  'WITH vwap' is the mirror: {hb['pnl'].mean():+.1f}/trade on "
          f"{len(hb)} trades. If H and its mirror are BOTH near baseline, VWAP")
    print(f"  position carries nothing and the split is just sample noise.")


if __name__ == "__main__":
    main()
