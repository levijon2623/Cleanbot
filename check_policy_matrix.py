# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_policy_matrix.py
======================
Does the flow trigger's edge SURVIVE the policy it is traded under, or is the
edge a property of the policy?

THE CONTRADICTION THIS EXISTS TO RESOLVE
    check_step1_redo   naked long, 50% trail, every trigger, within-day null
                       -> 87 of 150 cells BEAT their null, z up to +9.8
    check_structures   naked long, hold to EOD, one trade/ticker-day
                       -> gap -2.7pp, NOT significant

    Same trigger, opposite answers. Three things differed at once (exit,
    position policy, null construction), so neither result isolates anything.
    This crosses them deliberately.

THE MATRIX
    exit policy        eod      hold to the 15:55 flatten, no TP, no stop
                       trail50  the deployed trailing exit
                       static   the deployed bracket (tp=target_roe,
                                stop=target_roe/rr)
    position policy    seq      one position per ticker at a time (the live
                                guard, bot_runner.py:1288)
                       cap3     up to 3 CONCURRENT positions (the "pile on up
                                to 3" variant)
                       all      every trigger taken, unlimited concurrency

    3 x 3 = 9 cells, each with its OWN null so the comparison is like-for-like.

POSITION POLICY IS NOT AN EXIT POLICY, AND THE DIFFERENCE MATTERS
    `cap3` and `all` change WHICH TRIGGERS BECOME TRADES -- they alter the
    sample, not the trade. METHODOLOGY 3 is about exactly this: the book's OOS
    sign has already been shown to depend on the sampling rule. Crossing the two
    axes is the only way to tell an exit effect from a sample effect.

THE NULL
    Within-day permutation of outcomes across that day's entry minutes, as in
    check_step1_redo. Entry count, time-of-day profile and the policy itself are
    all preserved; only the signal's timing is destroyed.

    🚨 THIS NULL IS DEGENERATE FOR `all`, AND THE GAP THERE IS STRUCTURALLY ZERO.
    When every trigger is taken, permuting outcomes across entry minutes
    reassigns the SAME MULTISET of outcomes, so mean(real) == mean(null)
    identically -- no market fact can move it. The smoke test duly printed
    +0.0pp, which is arithmetic, not evidence. Same family as the A5 collapse in
    METHODOLOGY 6c: a control that cannot move is not a control.

    The null only bites when the position policy SELECTS a subset (`seq`,
    `cap3`), because then which trades survive depends on the ordering that the
    permutation destroys. `all` is therefore reported for its RETURN but its GAP
    is suppressed, and answering "does the signal work when you take
    everything?" needs a random-entry-minute null instead (the
    check_structures construction), which re-prices different contracts.

CORRECTNESS
    `walk_concurrent` generalises sim_core.walk to N concurrent positions. At
    max_open=1 it MUST reproduce sim_core.walk exactly; the run asserts this per
    cell and aborts on divergence (METHODOLOGY 1 -- hand-rolled simulators have
    drifted twice here).

Usage:
  python check_policy_matrix.py --tickers SPY QQQ IWM --nulls 200
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import SLICE_EDGES
import sim_core

PRE_LO = pd.Timestamp("2023-10-12").date()
DEPLOY_LO = SLICE_EDGES[0]
SPLIT = pd.Timestamp("2025-08-21").date()

EXITS = {
    "eod":     dict(name="eod", kind="fixed", tp=99.0, stop=None),
    "trail50": dict(name="trail50", kind="trail", trail=0.50),
    "static":  dict(name="static", kind="fixed", tp=1.00, stop=1.00),
}
POSITIONS = {"seq": 1, "cap3": 3, "all": 10 ** 6}


def presim(cand, pol, em):
    """[(date, mod, pnl, exit_mod)] -- each candidate simulated once."""
    out = []
    for d, m, path in cand:
        pnl, xm, _ = sim_core.simulate(path, pol, em, fill="bot")
        out.append((d, m, pnl, xm))
    return out


def walk_concurrent(items, max_open):
    """Admit a trade if fewer than `max_open` positions are currently open.

    `items` must be sorted by (date, mod). At max_open=1 this is exactly
    sim_core.walk's guard; the caller asserts that equivalence.
    """
    cur, open_until, out = None, [], []
    for d, m, pnl, xm in items:
        if d != cur:
            cur, open_until = d, []
        open_until = [x for x in open_until if x > m]
        if len(open_until) >= max_open:
            continue
        out.append((d, pnl))
        open_until.append(xm)
    return out


def null_draw(sim, max_open, rng, bucket="day"):
    """Permutation of outcomes across entry minutes, then re-apply the policy.

    🚨 `bucket` EXISTS BECAUSE THE DAY-WIDE PERMUTATION IS CONFOUNDED BY TIME
    BUDGET. An outcome generated at 14:30 carries a 14:30 time budget; moving it
    onto a 10:00 entry breaks the minute<->budget correspondence that real
    trades keep. With a `seq` guard -- which takes the FIRST trigger -- the real
    book gets the correctly-matched early outcome while the null gets a random,
    typically later and shorter-budget one. That alone manufactures a positive
    gap, and check_holdtime already showed "early" is mostly a time-budget
    artifact.

      day   permute across the whole session   (confounded, kept for comparison)
      hour  permute only WITHIN the same hour  (budgets stay comparable)

    If the gap collapses under `hour`, the apparent edge was budget, not signal
    -- and that verdict travels to check_step1_redo, which used the day-wide
    construction throughout.
    """
    by_key: dict = {}
    for d, m, pnl, xm in sim:
        k = (d, m // 60) if bucket == "hour" else (d,)
        by_key.setdefault(k, []).append((m, pnl, xm))
    out = []
    for k, items in by_key.items():
        d = k[0]
        mods = sorted(m for m, _, _ in items)
        outc = [(p, x) for _, p, x in items]
        rng.shuffle(outc)
        for m, (p, x) in zip(mods, outc):
            out.append((d, m, p, x))
    out.sort(key=lambda t: (t[0], t[1]))
    return walk_concurrent(out, max_open)


def score(rows):
    if not rows:
        return None
    d = np.array([x[0] for x in rows])
    p = np.array([x[1] for x in rows], float)
    w = {"n": len(p), "all": p.mean(), "days": len(set(d))}
    for lbl, lo, hi in (("pre", PRE_LO, DEPLOY_LO), ("is", DEPLOY_LO, SPLIT),
                        ("oos", SPLIT, SLICE_EDGES[-1])):
        m = (d >= lo) & (d < hi)
        w[lbl] = p[m].mean() if m.sum() else np.nan
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--null-bucket", default="day", choices=["day", "hour"],
                    help="'hour' controls for the time-budget confound")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    rng = np.random.default_rng(a.seed)

    # candidates once per (ticker, direction); exits change the SIM, positions
    # are bookkeeping on top, so presim runs 3x per pair, not 9x.
    pooled: dict = {k: {p: [] for p in POSITIONS} for k in EXITS}
    pooled_null: dict = {k: {p: [] for p in POSITIONS} for k in EXITS}

    for tk in a.tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        for direction in a.dirs:
            rule = {"name": f"{tk} {direction}", "ticker": tk,
                    "direction": direction, "dte": [0, 1],
                    "min_flow_pct": a.pct, "target_roe": 1.0, "rr": 1.0}
            cand = sim_core.build_candidates(D, rule, trigs=trigs, since=None)
            if not cand:
                continue
            em = sim_core.eod_mod(rule)
            for ek, pol in EXITS.items():
                sim = presim(cand, pol, em)
                ref = sim_core.walk(cand, pol, em, fill="bot")
                mine = walk_concurrent(sim, 1)
                if len(mine) != len(ref) or not np.allclose(
                        [p for _, p in mine], [p for _, p in ref]):
                    raise SystemExit(
                        f"ABORT: walk_concurrent(max_open=1) diverged from "
                        f"sim_core.walk on {tk} {direction} exit={ek} "
                        f"({len(mine)} vs {len(ref)}).")
                for pk, mx in POSITIONS.items():
                    pooled[ek][pk] += walk_concurrent(sim, mx)
                    for _ in range(a.nulls // 10):        # nulls are pooled
                        pooled_null[ek][pk] += null_draw(sim, mx, rng, a.null_bucket)
        print(f"  {tk} done", flush=True)

    print(f"\n{'='*116}")
    print(f"  POLICY MATRIX -- naked long, p{a.pct}, {' '.join(a.tickers)}")
    print(f"{'='*116}")
    print(f"  {'exit':9} {'position':9} {'n':>6} {'days':>5} {'REAL':>8} "
          f"{'NULL':>8} {'GAP':>8}   {'PRE':>8} {'IS':>8} {'OOS':>8}")
    rows = []
    for ek in EXITS:
        for pk in POSITIONS:
            s = score(pooled[ek][pk])
            nl = score(pooled_null[ek][pk])
            if not s or not nl:
                continue
            gap = s["all"] - nl["all"]
            degen = (pk == "all")
            if degen and abs(gap) > 1e-9:
                raise SystemExit(
                    f"ABORT: `all` gap is {gap:.2e}, expected exactly 0. The "
                    f"permutation null is supposed to be degenerate there; a "
                    f"non-zero value means the policy is dropping trades and "
                    f"the whole matrix needs re-checking.")
            if not degen:
                rows.append((ek, pk, s, nl, gap))
            gtxt = "  (degen)" if degen else f"{gap*100:>+7.1f}%"
            print(f"  {ek:9} {pk:9} {s['n']:>6} {s['days']:>5} "
                  f"{s['all']*100:>+7.1f}% {nl['all']*100:>+7.1f}% "
                  f"{gtxt}   {s['pre']*100:>+7.1f}% "
                  f"{s['is']*100:>+7.1f}% {s['oos']*100:>+7.1f}%")
        print()

    print(f"{'='*116}\n  WHAT MOVES THE GAP?\n{'='*116}")
    df = pd.DataFrame([(e, p, g) for e, p, _, _, g in rows],
                      columns=["exit", "position", "gap"])
    print("  mean gap by EXIT policy:")
    for k, v in df.groupby("exit")["gap"].mean().items():
        print(f"    {k:9} {v*100:+.1f}pp")
    print("  mean gap by POSITION policy:")
    for k, v in df.groupby("position")["gap"].mean().items():
        print(f"    {k:9} {v*100:+.1f}pp")
    print("\n  A gap that varies by EXIT means the signal's information decays")
    print("  over the hold -- it is about WHEN TO LEAVE, not only when to enter.")
    print("  A gap that varies by POSITION means the sample rule is doing the")
    print("  work, which is METHODOLOGY 3 and not an edge at all.")
    print(f"\n  {len(rows)} cells. No holdout remains -- this is diagnosis of an")
    print("  existing contradiction, not a search for a policy to deploy.")


if __name__ == "__main__":
    main()

