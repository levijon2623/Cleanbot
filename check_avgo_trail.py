# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_avgo_trail.py
===================
IS AVGO'S 50% TRAIL TOO TIGHT?  -- the dead zone, measured on AVGO's own peaks.

THE ARITHMETIC THAT DECIDES IT
    A trail of T exits at peak x (1 - T), so the trade only clears its entry if
        peak / entry  >  1 / (1 - T)
    T=0.30 -> needs a +43% peak      T=0.50 -> needs +100%
    T=0.40 -> needs +67%             T=0.65 -> needs +186%
    So a WIDER trail is harder to profit from, not easier -- it hands back more
    of whatever it caught. "Too tight" and "too loose" both have a precise
    meaning here and they run in opposite directions from the intuition.
    The question is therefore empirical: where does AVGO's peak distribution
    actually sit relative to those thresholds?

WHY AVGO SPECIFICALLY NEEDS ITS OWN ANSWER
    AVGO is PAPER_ONLY and is slated for removal on an A PRIORI screen -- a 6.7%
    median entry spread against 1.1-4.0% for the deployed book. That screen uses
    no performance data, so nothing here can rescue the rule: a trail cannot pay
    back 6.7% of round-trip friction. What this CAN do is say whether the trail
    width is a genuine second problem or a red herring, which matters if the
    same width is being carried by rules that do pass the spread screen.

EVERY ARM IS SCORED ON THE `bot` FILL MODEL, which is where the spread bites --
a mid-fill sweep would flatter every wide-trail arm by exactly the friction the
screen is worried about.

Usage:
  python check_avgo_trail.py
  python check_avgo_trail.py --rule "SMH LOWVOL PUT"
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
TRAILS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


def boot(v, days, n, rng):
    u = pd.unique(days)
    idx = {k: np.where(days == k)[0] for k in u}
    out = np.empty(n)
    for i in range(n):
        s = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        out[i] = v[s].sum()
    return tuple(np.percentile(out, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="AVGO HIVOL PUT")
    ap.add_argument("--fill", default="bot")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=37)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = {r["name"]: r for r in sim_core.research_rules(include_paper=True)}
    if a.rule not in rules:
        print(f"  no rule {a.rule!r}; have: {sorted(rules)}"); return
    rule = rules[a.rule]
    tk = rule["ticker"]
    flow = _flow_for(D, [tk])
    if flow.empty:
        print(f"  no flow for {tk}"); return
    trigs = D.triggers_for(flow, tk)
    D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
    cand = sim_core.build_candidates(D, rule, trigs=trigs)
    if not cand:
        print("  no candidates"); return
    eod_m = sim_core.eod_mod(rule)
    print(f"  {a.rule}: {len(cand)} candidates, deployed trail_pct="
          f"{rule.get('trail_pct')}, fill={a.fill}\n")

    # ---- 1. the peak distribution, which is what the dead zone acts on
    peaks = []
    for d, m, path in cand:
        e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
        if e_mid <= 0:
            continue
        n = int(np.searchsorted(mods, eod_m, side="right"))
        if n < 5:
            continue
        b = np.asarray(bid[:n], float)
        if b.max() <= 0:
            continue
        peaks.append((d, b.max() / e_mid - 1.0))
    P = pd.DataFrame(peaks, columns=["date", "peak"])
    print(f"  PEAK ROE DISTRIBUTION  (n={len(P)}, {P['date'].nunique()} days)")
    for q in (10, 25, 50, 75, 90, 95):
        print(f"    p{q:<3} {P['peak'].quantile(q/100)*100:>+8.1f}%")
    print(f"    mean {P['peak'].mean()*100:+.1f}%   "
          f"share peaking >0: {(P['peak']>0).mean()*100:.0f}%")

    print(f"\n  DEAD ZONE -- share of trades that could EVER profit at each trail")
    print(f"  {'trail':>7} {'needs peak':>12} {'% clearing it':>15}")
    for t in TRAILS:
        need = 1.0 / (1.0 - t) - 1.0
        print(f"  {t:>7.0%} {need*100:>+11.1f}% {(P['peak']>need).mean()*100:>14.1f}%")

    # ---- 2. the sweep
    print(f"\n  TRAIL SWEEP")
    print(f"  {'trail':>7} {'IS trd':>7} {'IS tot%':>10} {'OOS trd':>8} {'OOS days':>9} "
          f"{'OOS tot%':>10} {'OOS 95% CI':>24}")
    rows = []
    for t in TRAILS:
        pol = dict(name=f"trail{int(t*100)}", kind="trail", trail=t)
        r = sim_core.walk(cand, pol, eod_m, fill=a.fill)
        i = [(d, p) for d, p in r if d < SPLIT]
        o = [(d, p) for d, p in r if d >= SPLIT]
        it = sum(p for _, p in i) * 100
        ot = sum(p for _, p in o) * 100
        if o:
            v = np.array([p for _, p in o]) * 100
            dd = np.array([d for d, _ in o])
            lo_, hi_ = boot(v, dd, a.boot, rng)
        else:
            lo_ = hi_ = np.nan
        rows.append((t, len(i), it, len(o), len({d for d, _ in o}), ot, lo_, hi_))
        print(f"  {t:>7.0%} {len(i):>7} {it:>+10.1f} {len(o):>8} "
              f"{len({d for d,_ in o}):>9} {ot:>+10.1f} "
              f"[{lo_:>+9.1f},{hi_:>+9.1f}]")

    # ---- 3. the static bracket the rule would use with trail_pct = 0
    tr, rr = float(rule["target_roe"]), float(rule["rr"])
    stat = dict(name="static", kind="fixed", tp=tr, stop=tr / rr)
    r = sim_core.walk(cand, stat, eod_m, fill=a.fill)
    i = [(d, p) for d, p in r if d < SPLIT]
    o = [(d, p) for d, p in r if d >= SPLIT]
    print(f"  {'static':>7} {len(i):>7} {sum(p for _,p in i)*100:>+10.1f} "
          f"{len(o):>8} {len({d for d,_ in o}):>9} {sum(p for _,p in o)*100:>+10.1f}"
          f"   (tp {tr:.0%} / stop {tr/rr:.0%})")

    best = max(rows, key=lambda x: x[5])
    dep = rule.get("trail_pct")
    cur = next((x for x in rows if abs(x[0] - float(dep or 0)) < 1e-9), None)
    print(f"\n  deployed trail {dep}: OOS {cur[5]:+.1f}%" if cur else "")
    print(f"  best swept trail {best[0]:.0%}: OOS {best[5]:+.1f}% "
          f"[{best[6]:+.1f}, {best[7]:+.1f}]")
    if cur:
        print(f"  difference {best[5]-cur[5]:+.1f}pp -- and note every CI above is")
        print(f"  wide enough that the ORDERING matters more than any single cell.")

    print(f"\n  HOW TO READ IT")
    print(f"  If most peaks sit BELOW the threshold a trail needs, the trail is not")
    print(f"  'too tight' or 'too loose' -- it is unreachable, and the trade is decided")
    print(f"  by the EOD flatten rather than by the exit rule at all. Check the dead")
    print(f"  zone table before reading anything into the sweep.")
    print(f"  For AVGO specifically: a 6.7% median entry spread is charged on every")
    print(f"  round trip regardless of trail width, so a favourable cell here is not")
    print(f"  a reprieve from the a priori spread screen.")


if __name__ == "__main__":
    main()
