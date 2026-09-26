# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_mbo_gating.py
===================
RETRACTION AND REPLACEMENT for the A5 control in `check_mbo_flow_interaction`
and `check_wpoc_gate`.

WHAT WENT WRONG
---------------
A5 compared the aligned-minus-opposed gap `g` against a "no-interaction
baseline" `base = 2*E[s*r]` estimated ON THE SAME ROWS, and required
`g > 1.5*base`. Write p = P(aligned), M_al = E[s*r|aligned],
M_op = E[s*r|opposed]. Then, exactly:

    g    = M_al + M_op
    base = 2*(p*M_al + (1-p)*M_op)
    g - base = (1 - 2p) * (M_al - M_op)

Our direction mix is ~50/50 by construction, so p ~= 0.5 in every cell and the
factor (1-2p) annihilates the statistic NO MATTER HOW LARGE the real effect is.
Worked counterexample: a trigger that earns +10bp when aligned and 0bp when
opposed -- a total, decisive interaction -- gives g=10, base=10, g-base=0.

And because base ~= g mechanically, the GATE itself is unpassable:
    base > 0 branch:  need g > 1.5*g   ->  impossible for g > 0
    base < 0 branch:  need g > 0.5|g|  ->  impossible for g < 0
So "0 of 24 cells passed A5" was a property of the estimator, not of the market.
It carries no information and the conclusion drawn from it is withdrawn.

WHAT REPLACES IT
----------------
Two statistics, because A5 was conflating two different questions.

  Q1 DEPLOYABLE GAIN -- "does gating the trigger on feature agreement beat not
     gating?" This is what you would actually trade.
         m     = E[d*r]                 the trigger's edge, ungated
         M_al  = E[d*r | aligned]       gate: take only agreeing triggers
         M_op  = E[d*r | opposed]       gate: take only disagreeing triggers
         gain  = M_al - m = (1-p)*g     (and the flip side, M_op - m = -p*g)
     A feature with only a MARGINAL effect still produces a real gain here, and
     that is fine -- it is still money. A5's fear of "a marginal in a costume"
     was a mechanism question, not a deployability question. Separated below.

  Q2 TRUE INTERACTION -- since d and s are both +/-1, the four (d,s) cells are
     saturated by {1, d, s, d*s}, so the interaction is identified exactly:
         gamma = 1/4 * [ E(r|+,+) - E(r|+,-) - E(r|-,+) + E(r|-,-) ]
     This is the quantity A5 was reaching for. It is zero when the feature is a
     pure marginal predictor and the trigger adds nothing to it.

INFERENCE
---------
Day-block bootstrap: resample DAYS with replacement, 2000x. Days move whole, so
this survives both the day-clustering of triggers (58/day on RTY) and the
overlap of forward windows inside a day. CI is the 2.5/97.5 percentile.

BARS (unchanged, pre-committed elsewhere)
    HL perp taker round trip   9.0 bp
    CME futures round trip     1.2 bp

Usage:  python check_mbo_gating.py
        python check_mbo_gating.py --boot 4000 --gated
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_mbo_flow_interaction import FEATS, HORIZONS, PROXY, SPLIT, load_panel

PERP_BAR = 9.0
FUT_BAR = 1.2


def cells(d, s, r):
    """Saturated 2x2 means plus the derived statistics. All in RETURN units."""
    out = {}
    for dv in (1.0, -1.0):
        for sv in (1.0, -1.0):
            m = (d == dv) & (s == sv)
            out[(dv, sv)] = (float(np.mean(r[m])), int(m.sum())) if m.sum() else (np.nan, 0)
    pp, pm = out[(1.0, 1.0)][0], out[(1.0, -1.0)][0]
    mp, mm = out[(-1.0, 1.0)][0], out[(-1.0, -1.0)][0]
    gamma = 0.25 * (pp - pm - mp + mm)
    al = (s == d)
    m_all = float(np.mean(d * r))
    M_al = float(np.mean(d[al] * r[al])) if al.sum() else np.nan
    M_op = float(np.mean(d[~al] * r[~al])) if (~al).sum() else np.nan
    return dict(gamma=gamma, m_all=m_all, M_al=M_al, M_op=M_op,
                gain_al=M_al - m_all, gain_op=M_op - m_all,
                p=float(al.mean()), cells=out)


def prep(sub, feat, h):
    x = sub[feat].to_numpy(float)
    d = sub["dir"].to_numpy(float)
    r = sub[f"fwd{h}"].to_numpy(float)
    day = sub["date"].to_numpy()
    m = np.isfinite(x) & np.isfinite(r) & (x != 0)
    return np.sign(x[m]), d[m], r[m], day[m]


def boot_ci(s, d, r, day, key, boot, rng):
    """Day-block bootstrap CI for one statistic of `cells`."""
    idx_by_day = {}
    for i, dd in enumerate(day):
        idx_by_day.setdefault(dd, []).append(i)
    days = np.array(list(idx_by_day.keys()), dtype=object)
    blocks = {k: np.array(v) for k, v in idx_by_day.items()}
    vals = []
    for _ in range(boot):
        pick = rng.choice(len(days), size=len(days), replace=True)
        ii = np.concatenate([blocks[days[j]] for j in pick])
        try:
            v = cells(d[ii], s[ii], r[ii])[key]
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < boot // 4:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run(a):
    rng = np.random.default_rng(31)
    for tk, sym in PROXY.items():
        P = load_panel(tk, sym, a.gated)
        if P is None or P.empty:
            print(f"\n  {tk}/{sym}: no panel")
            continue
        print("\n" + "=" * 122)
        print(f"  {tk} / {sym}   {len(P):,} triggers, {P['date'].nunique()} days"
              f"   {'(rule-gated)' if a.gated else '(ungated p50)'}")
        print("  gain_al = gating on AGREEMENT vs not gating.  gamma = the true"
              " 2x2 interaction A5 meant to measure.")
        print("=" * 122)
        print(f"  {'feature':11} {'h':>4} {'p(al)':>6} {'ungated':>8} {'gate_al':>8} "
              f"{'gain_al':>8} {'[95% CI]':>18} {'gamma':>8} {'[95% CI]':>18} "
              f"{'IS':>7} {'OOS':>7}")
        for f in FEATS:
            for h in HORIZONS:
                s, d, r, day = prep(P, f, h)
                if len(s) < 60 or (s == d).sum() < 20 or (s != d).sum() < 20:
                    continue
                c = cells(d, s, r)
                lo1, hi1 = boot_ci(s, d, r, day, "gain_al", a.boot, rng)
                lo2, hi2 = boot_ci(s, d, r, day, "gamma", a.boot, rng)

                def half(mask):
                    if mask.sum() < 40:
                        return np.nan
                    try:
                        return cells(d[mask], s[mask], r[mask])["gain_al"]
                    except Exception:
                        return np.nan
                dd = pd.Series(day)
                gi = half((dd < SPLIT).to_numpy())
                go = half((dd >= SPLIT).to_numpy())
                sig1 = "*" if np.isfinite(lo1) and lo1 * hi1 > 0 else " "
                sig2 = "*" if np.isfinite(lo2) and lo2 * hi2 > 0 else " "
                print(f"  {f:11} {h:>3}m {c['p']:>6.3f} {c['m_all']*1e4:>+8.2f} "
                      f"{c['M_al']*1e4:>+8.2f} {c['gain_al']*1e4:>+8.2f}{sig1}"
                      f"[{lo1*1e4:>+7.2f},{hi1*1e4:>+7.2f}] "
                      f"{c['gamma']*1e4:>+8.2f}{sig2}"
                      f"[{lo2*1e4:>+7.2f},{hi2*1e4:>+7.2f}] "
                      f"{gi*1e4 if np.isfinite(gi) else np.nan:>+7.2f} "
                      f"{go*1e4 if np.isfinite(go) else np.nan:>+7.2f}")
            print()

    print("=" * 122)
    print("  * = day-block bootstrap 95% CI excludes zero.")
    print(f"  A gain must clear {PERP_BAR:.1f}bp (perp) or {FUT_BAR:.1f}bp (futures)")
    print("  AND hold sign IS/OOS before it is worth a $179/mo live MBO feed.")
    print("  gamma near zero with a large gain_al = the feature is a pure MARGINAL")
    print("  predictor: it would pay, but the trigger contributes nothing to it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--gated", action="store_true")
    run(ap.parse_args())
