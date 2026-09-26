# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_rule_selection_wf.py
==========================
"What would a robust-only book actually have earned?" -- answered honestly.

THE PROBLEM WITH THE HINDSIGHT NUMBER
-------------------------------------
Scoring the 5 rules that tested well gives OOS +30.5% (n=184). But those 5 were
chosen AFTER seeing their performance, so that number is selection on the
outcome and is optimistically biased by an unknown amount. The same trap as
every grid search in this project, applied one level up -- at the book rather
than the parameter.

FOUR BOOKS, IN ASCENDING ORDER OF HOW MUCH THEY CHEAT
------------------------------------------------------
  all9        every enabled rule. No selection, no bias, includes known-bad rules.
  spread      A PRIORI LIQUIDITY SCREEN ONLY -- keep rules whose MEDIAN ENTRY
              SPREAD is under --max-spread. Uses NO performance data at all, so
              it carries ZERO selection bias and is implementable on day one.
              (AVGO 6.7% and SMH 9.2% fail it; the robust five are 1.1-4.0%.)
  walkfwd     at each slice, keep only rules that were PROFITABLE ON PRIOR
              SLICES ONLY (>= --min-n trades), then score forward. Chained.
              This is the honest version of "pick the good rules".
  hindsight   the 5 that tested well. Upper bound, NOT a forecast.

If `spread` captures most of `hindsight`, the deployable answer needs no
performance selection at all -- which would be the strongest possible result,
because a rule you can state in advance cannot be overfit.

Selection uses the rule's own deployed exit (`sim_core.policy_for`) and the
`bot` fill model throughout, so every book here is on the same basis.

Also prints, per request, EVERY TRADE for the thinly-sampled index rules so the
days can be inspected by hand for a contextual explanation -- SPY CHOP CALL
(13 trades, 2/6 slices) and QQQ HIVOL CALL (26 trades, 2/6) carry the two
largest OOS numbers in the hindsight book and deserve eyeballing.

Usage:
  python check_rule_selection_wf.py
  python check_rule_selection_wf.py --max-spread 0.05 --min-n 5
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx

SPLIT = pd.Timestamp("2025-08-21").date()
HINDSIGHT = ["SPY CHOP CALL", "QQQ HIVOL CALL", "IWM HIVOL CALL",
             "NVDA LOWVOL PUT", "META LOWVOL PUT"]
INSPECT = ["SPY CHOP CALL", "QQQ HIVOL CALL"]


def _stat(rows):
    if not rows:
        return None
    v = np.array([p for _d, p in rows], float)
    i = np.array([p for d, p in rows if d < SPLIT], float)
    o = np.array([p for d, p in rows if d >= SPLIT], float)
    sl = [[] for _ in range(6)]
    for d, p in rows:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    pop = [np.mean(b) for b in sl if len(b) >= 3]
    return dict(n=len(v), all=v.mean(), win=(v > 0).mean(),
                is_=i.mean() if len(i) else np.nan,
                oos=o.mean() if len(o) else np.nan,
                npop=len(pop), nposs=sum(1 for x in pop if x > 0),
                slices=[np.mean(b) if len(b) >= 3 else None for b in sl])


def _line(lbl, rows, w=12):
    s = _stat(rows)
    if s is None:
        return f"  {lbl:{w}}  (empty)"
    sl = " ".join(f"S{j+1}{x*100:+.0f}" if x is not None else f"S{j+1}··"
                  for j, x in enumerate(s["slices"]))
    return (f"  {lbl:{w}} n={s['n']:>4}  all {s['all']*100:>+6.1f}%  "
            f"IS {s['is_']*100:>+6.1f}%  OOS {s['oos']*100:>+6.1f}%  "
            f"win {s['win']:.2f}  sl {s['nposs']}/{s['npop']}  [{sl}]")


def run(a):
    import directional_flow_backtester as D
    import sim_core
    from config import RULES, TRAIL_PCT

    rules = [r for r in RULES if r.get("enabled", True)]
    book = {}
    spreads = {}
    print("  building...")
    for r in rules:
        cand = sim_core.build_candidates(D, r)
        if not cand:
            continue
        rows = sim_core.walk(cand, sim_core.policy_for(r, TRAIL_PCT),
                             sim_core.eod_mod(r), fill="bot")
        # entry spread as a fraction of entry premium, at the real trigger minutes
        sp = [2.0 * (float(p[1]) - float(p[0])) / float(p[0])
              for _d, _m, p in cand if float(p[0]) > 0 and float(p[1]) >= float(p[0])]
        book[r["name"]] = rows
        spreads[r["name"]] = float(np.median(sp)) if sp else np.nan
        print(f"    {r['name']:24} {len(rows):>4} trades   med spread "
              f"{spreads[r['name']]*100:>5.1f}%")

    # ---------------- the four books ----------------
    print("\n" + "=" * 118)
    print("  FOUR BOOKS, ASCENDING ORDER OF HOW MUCH THEY CHEAT")
    print("=" * 118)

    all9 = [x for rows in book.values() for x in rows]
    print(_line("all9", all9))

    keep_sp = [n for n, s in spreads.items() if np.isfinite(s) and s <= a.max_spread]
    spr = [x for n in keep_sp for x in book[n]]
    print(_line(f"spread<={a.max_spread*100:.0f}%", spr, w=12))
    print(f"    kept (a priori, NO performance data): {sorted(keep_sp)}")
    dropped = sorted(set(book) - set(keep_sp))
    print(f"    dropped: {dropped}")

    # ---- walk-forward selection ----
    sel_rows, dep_rows, log = [], [], []
    for k in range(1, 6):
        picks = []
        for n, rows in book.items():
            prior = [p for d, p in rows if (_slice_idx(d) is not None and _slice_idx(d) < k)]
            if len(prior) >= a.min_n and float(np.mean(prior)) > 0:
                picks.append(n)
        cur = [(d, p) for n in picks for d, p in book[n] if _slice_idx(d) == k]
        dep = [(d, p) for n in book for d, p in book[n] if _slice_idx(d) == k]
        if not dep:
            continue
        log.append((k, picks, len(cur), np.mean([p for _d, p in cur]) if cur else np.nan,
                    np.mean([p for _d, p in dep])))
        sel_rows += cur
        dep_rows += dep
    print(_line("walkfwd", sel_rows))

    hind = [x for n in HINDSIGHT if n in book for x in book[n]]
    print(_line("hindsight", hind))
    print("    ^ selection ON THE OUTCOME. Upper bound, not a forecast.")

    print("\n" + "-" * 118)
    print("  WALK-FORWARD DETAIL — rules chosen on PRIOR slices only")
    print("-" * 118)
    for k, picks, n, m, dm in log:
        print(f"  S{k+1}  n={n:>4}  selected {m*100:>+7.1f}%   all9 {dm*100:>+7.1f}%   "
              f"kept {len(picks)}: {', '.join(sorted(p.split()[0] for p in picks))}")
    if sel_rows and dep_rows:
        ms, md = np.mean([p for _d, p in sel_rows]), np.mean([p for _d, p in dep_rows])
        print(f"\n  chained: walk-forward {ms*100:>+6.1f}% (n={len(sel_rows)})   "
              f"vs all9 {md*100:>+6.1f}% (n={len(dep_rows)})   "
              f"-> {'SELECTION ADDS VALUE' if ms > md else 'selection does NOT help'}")

    # ---------------- per-trade listing for the thin rules ----------------
    gex = {}; vol = {}; trd = {}
    try:
        from macro_calendar import is_macro_am_day
    except Exception:
        is_macro_am_day = lambda d: None

    print("\n" + "=" * 118)
    print("  EVERY TRADE — the thinly-sampled index rules, for manual inspection")
    print("  (these two carry the largest OOS numbers in the hindsight book)")
    print("=" * 118)
    for r in rules:
        if r["name"] not in INSPECT:
            continue
        tk = r["ticker"]
        if tk not in gex:
            gex[tk] = D.load_gex("historical", tk)
            vol[tk] = D.load_volume_regime("historical", tk)
            trd[tk] = D.load_trend_regime("historical", tk)
        cand = sim_core.build_candidates(D, r)
        pol = sim_core.policy_for(r, TRAIL_PCT)
        em = sim_core.eod_mod(r)
        print(f"\n  {r['name']}   (regime={r.get('regime')}, p{r.get('min_flow_pct')}, "
              f"amt={r.get('amt_open')}, exit={pol['name']})")
        print(f"  {'date':12} {'time':>6} {'entry$':>7} {'exit':>8} {'pnl%':>8} "
              f"{'sl':>3}  {'trend':<10} {'vol':<8} {'gex':<9} macro")
        cur, busy = None, -1
        for d, m, path in cand:
            if d != cur:
                cur, busy = d, -1
            if m < busy:
                continue
            pnl, xm, tag = sim_core.simulate(path, pol, em, fill="bot")
            busy = xm
            k = _slice_idx(d)
            mac = is_macro_am_day(d)
            print(f"  {str(d):12} {m//60:>2}:{m%60:02d} {float(path[0]):>7.2f} "
                  f"{tag:>8} {pnl*100:>+7.1f}% {('S'+str(k+1)) if k is not None else '  -':>3}  "
                  f"{str(trd[tk].get(d)):<10} {str(vol[tk].get(d)):<8} "
                  f"{str(gex[tk].get(d)):<9} {'YES' if mac else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-spread", type=float, default=0.05,
                    help="a-priori liquidity screen: max median entry spread (default 5%%)")
    ap.add_argument("--min-n", type=int, default=5,
                    help="min prior trades before a rule may be selected")
    run(ap.parse_args())
