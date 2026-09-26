# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_fill_sensitivity.py
=========================
WHICH CONCLUSIONS SURVIVE THE FILL ASSUMPTION, AND WHICH ARE JUST THE ASSUMPTION?

The problem, stated plainly: for a contract whose bid/ask spread is wider than
the edge we claim to measure, choosing ANY price between bid and ask is
arbitrary, and that arbitrary choice sets the SIGN of the answer. Reporting one
point estimate for such a rule is not a measurement, it is a restatement of the
assumption.

For a tight, deep chain the question is nearly moot -- NVDA / SPY 0-1DTE quote a
penny or two wide and a mid+1 limit fills, so every model agrees. For a thin one
it is everything: SMH ATM 0DTE trades in the HUNDREDS of contracts with a wide
quote, and there the whole exercise turns on what we assume.

So this does not try to pick the right fill model. It scores every rule under
ALL FOUR and asks whether the verdict is stable:

    mid     mid in,  mid out                    frictionless upper bound
    bot     mid+1tick in (capped at ask),       MIRRORS bot_runner exactly
            bid - cushion out
    askbid  ask in, bid out
    worst   ask in, bid - 1.5x spread out       symmetric-pessimistic lower bound

`bot` is the honest central estimate -- bot_runner posts
`min(mid + 0.01, ask)` on entry (verified against live paper fills: entry_mid
1.29 -> entry_price 1.30) and `bid - spread*(0.5|1.5)` on exit. Every sim in
this repo charged the full ask on entry until 2026-09-10, inventing about a
half-spread of cost per trade.

CLASSIFICATION
--------------
  ROBUST            OOS keeps its sign across all four models -> we know something
  ASSUMPTION-BOUND  the sign flips inside the band -> we know nothing yet; the
                    number is our own assumption reflected back

Alongside each rule: the ticker's realised ENTRY SPREAD as a % of entry premium
(median and p90, measured at the actual trigger minutes, not a daily average)
and the fill-model spread in pp. When spread% > |OOS edge|, the rule is
mechanically assumption-bound and the label is redundant -- but it is printed so
the relationship is visible.

A NOTE ON WHAT THIS CANNOT DO
-----------------------------
`bot` assumes the mid+1 limit FILLS. On a thin chain it may not, so the trade is
missed or chased. That is fill PROBABILITY, not fill PRICE, and no amount of
bar data settles it -- it needs live evidence. The spread/volume columns are the
warning sign; treat `bot` as an UPPER bound on entry quality for anything whose
quote is wide.

Usage:  python check_fill_sensitivity.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
MODELS = ["mid", "bot", "askbid", "worst"]


def _oos(rows):
    o = [p for d, p in rows if d >= SPLIT]
    return float(np.mean(o)) if o else float("nan")


def _spread_profile(cand):
    """Realised entry spread as a fraction of entry premium, at the actual
    trigger minutes. payload = (mid, ask, ...) so spread = 2*(ask - mid)."""
    out = []
    for _d, _m, p in cand:
        mid, ask = float(p[0]), float(p[1])
        if mid > 0 and ask >= mid:
            out.append(2.0 * (ask - mid) / mid)
    return np.array(out) if out else np.array([np.nan])


def run(a):
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    rules = RULES if a.all else [r for r in RULES if r.get("enabled", True)]
    if a.rules:
        rules = [r for r in RULES if r["name"] in a.rules]

    print("=" * 124)
    print("  FILL-MODEL SENSITIVITY -- does the verdict survive the assumption?")
    for k in MODELS:
        print(f"    {k:8} {sim_core.FILL_MODELS[k]}")
    print("=" * 124)
    hdr = (f"  {'rule':24} {'n':>4} {'spr%med':>8} {'spr%p90':>8} " +
           " ".join(f"{m:>9}" for m in MODELS) + f" {'band':>7}  verdict")
    print(hdr)

    book = {m: [] for m in MODELS}
    rows = []
    for r in rules:
        cand = sim_core.build_candidates(D, r)
        if not cand:
            continue
        em = sim_core.eod_mod(r)
        pol = sim_core.policy_for(r, TRAIL_PCT)
        sp = _spread_profile(cand)
        res = {}
        for m in MODELS:
            w = sim_core.walk(cand, pol, em, fill=m)
            book[m] += w
            res[m] = _oos(w)
        vals = [res[m] for m in MODELS if np.isfinite(res[m])]
        band = (max(vals) - min(vals)) if vals else np.nan
        # The sign test uses only the PLAUSIBLE models. `mid` is a frictionless
        # reference nobody believes -- letting it vote makes a rule that every
        # realistic model agrees on look "assumption-bound" purely because the
        # fantasy bound sits on the other side of zero. (AMZN afternoon PUT is
        # exactly this: -7.1 / -1.6 / -8.1 realistic, +4.5 frictionless.)
        real = [res[m] for m in ("bot", "askbid", "worst") if np.isfinite(res[m])]
        signs = {np.sign(v) for v in real if v != 0}
        robust = len(signs) <= 1
        n = len(sim_core.walk(cand, pol, em, fill="bot"))
        if not robust:
            verdict = "ASSUMPTION-BOUND"
        elif real and np.mean(real) < 0:
            verdict = "ROBUST (negative)"
        else:
            verdict = "ROBUST"
        rows.append((r["name"], n, np.nanmedian(sp), np.nanpercentile(sp, 90), res, band, verdict))

    for nm, n, spm, spp, res, band, verdict in sorted(rows, key=lambda x: -x[4]["bot"]):
        cells = " ".join(f"{res[m]*100:>+8.1f}%" if np.isfinite(res[m]) else "     n/a "
                         for m in MODELS)
        print(f"  {nm:24} {n:>4} {spm*100:>7.1f}% {spp*100:>7.1f}% {cells} "
              f"{band*100:>6.1f}pp  {verdict}")

    print("\n" + "-" * 124)
    b = {m: _oos(book[m]) for m in MODELS}
    cells = " ".join(f"{b[m]*100:>+8.1f}%" for m in MODELS)
    bb = max(b.values()) - min(b.values())
    print(f"  {'BLENDED BOOK':24} {len(book['bot']):>4} {'':>8} {'':>8} {cells} {bb*100:>6.1f}pp")
    print(f"\n  The book's OOS edge under the honest central model (`bot`) is "
          f"{b['bot']*100:+.1f}%,\n  inside a {bb*100:.1f}pp band spanned by "
          f"defensible fill assumptions. Any per-rule claim\n  smaller than its own band is "
          f"a statement about the assumption, not about the rule.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--all", action="store_true",
                    help="include DISABLED rules. Worth doing: a rule killed under the old "
                         "ask-charging fill model may have been killed by the MODEL rather than "
                         "by its own performance -- but only where the spread is wide enough for "
                         "that to matter. On a tight-spread, deep-volume name a failure is REAL.")
    run(ap.parse_args())
