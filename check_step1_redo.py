# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_step1_redo.py
===================
RE-RUN "STEP 1" FROM SCRATCH on the enlarged calendar: every watchlist ticker x
CALL/PUT x flow-percentile threshold, with the deployed exit and NO conditioners,
gates, hour windows, regimes or AMT filters. Then ask whether the rules
originally crafted were in fact the best cells available -- and whether ANY cell
is distinguishable from chance.

WHY THIS IS NOT A RULE SEARCH, AND MUST NOT BE USED AS ONE
    There is no clean holdout left. The pre-sample window was spent on
    PRESAMPLE_PLAN.md on 2026-09-12, and the OOS window has been examined across
    ~20 sessions of work. So a 150-cell grid CANNOT certify a replacement rule:
    at 95% roughly 7.5 cells clear significance by construction, and there is no
    untouched data on which to check the winner. Anything promoted from this
    grid would be pure selection on the outcome.

    What the grid CAN answer, honestly:
      1. Where do the 9 deployed rules RANK among all 150 cells? (the user's
         actual question -- were they the best available?)
      2. Is the best cell distinguishable from what a 150-cell grid produces
         when the trigger carries no information at all?
    (2) is the decisive one. The book just failed its pre-sample holdout
    (-17.2%/trade); if the unconditioned grid is also indistinguishable from
    noise, the original selection was noise-mining and the holdout failure is
    explained rather than mysterious.

THE NULL -- this is the part that makes the grid readable
    For each cell, the trigger times are replaced by RANDOM times drawn from
    that ticker-day's own observed entry-time distribution, keeping the SAME
    number of entries per day. So the null asks precisely:

        does WHEN the flow trigger fires carry information,
        or is this just buying ATM options at comparable times of day?

    That is the right null for a timing signal. It preserves ticker, direction,
    day, entry count, time-of-day profile, the exit policy, the $0.50 floor and
    the sequential guard -- everything except the signal itself. Long options
    carry a heavy theta drift, so a random-entry null is a FAIR and demanding
    bar, not a straw man.

SCORING BASIS -- identical to the deployed book
    sim_core sequential fills (one position per ticker), fill="bot", the
    deployed trailing exit (config.TRAIL_PCT), $0.50 entry floor. dte [0,1].
    Reported per window: PRE (2023-10-12..2024-08-19), IS, OOS -- so decay is
    visible rather than averaged away.

Usage:
  python check_step1_redo.py --nulls 200
  python check_step1_redo.py --tickers SPY QQQ IWM --nulls 50
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
GRID_PCT = [50, 65, 80, 90, 95]


def deployed_cells():
    """{(ticker, direction): rule name} for every rule in config, enabled or not."""
    from config import RULES
    out = {}
    for r in RULES:
        out.setdefault((r["ticker"], r["direction"]), []).append(
            r["name"] + ("" if r.get("enabled", True) else " [disabled]"))
    return out


def cell_rule(tk, direction, pct):
    """A BARE rule: ticker, direction, dte, flow threshold. Nothing else.

    No `regime`, no `hours`, no `amt_open`, no `ema_confirm`, no `dmi_confirm`,
    no `vol_overlay` -- this is the unconditioned trigger the user asked for.
    `trail_pct` is left unset so sim_core.policy_for falls through to the
    deployed config.TRAIL_PCT.
    """
    return {"name": f"{tk} {direction} p{pct}", "ticker": tk,
            "direction": direction, "dte": [0, 1], "min_flow_pct": pct,
            "target_roe": 1.00, "rr": 1.0, "enabled": True}


def score(rows):
    """-> dict of per-window means. `rows` = [(date, pnl)]."""
    if not rows:
        return {}
    d = np.array([x[0] for x in rows])
    p = np.array([x[1] for x in rows], float)
    w = {}
    for lbl, lo, hi in (("pre", PRE_LO, DEPLOY_LO), ("is", DEPLOY_LO, SPLIT),
                        ("oos", SPLIT, SLICE_EDGES[-1])):
        m = (d >= lo) & (d < hi)
        w[lbl] = p[m].mean() if m.sum() else np.nan
        w["n_" + lbl] = int(m.sum())
    w["all"] = p.mean()
    w["n"] = len(p)
    w["win"] = float((p > 0).mean())
    w["days"] = len(set(d))
    return w


def presim(cand, pol, em):
    """[(date, mod, pnl, exit_mod)] -- simulate every candidate EXACTLY ONCE.

    `sim_core.simulate` is a pure function of the payload (the path carries its
    own minute array), so a candidate's pnl and exit minute do not depend on
    which entry minute it is attached to. That makes the null draws pure
    bookkeeping instead of thousands of re-simulations -- the difference
    between a multi-hour run and a few minutes.
    """
    out = []
    for d, m, path in cand:
        pnl, xm, _ = sim_core.simulate(path, pol, em, fill="bot")
        out.append((d, m, pnl, xm))
    return out


def seq_apply(items):
    """The one-position-per-ticker guard over pre-simulated items.

    A faithful restatement of sim_core.walk's guard for pre-simulated rows.
    Hand-rolled simulators have drifted twice in this repo (METHODOLOGY 1), so
    `main` ASSERTS this reproduces sim_core.walk exactly on the observed pass
    before any null is drawn; if it ever diverges the run aborts.
    """
    cur, busy, out = None, -1, []
    for d, m, pnl, xm in items:
        if d != cur:
            cur, busy = d, -1
        if m < busy:
            continue
        out.append((d, pnl))
        busy = xm
    return out


def null_draw(sim, rng):
    """One null: same entries per day, outcomes detached from their minutes.

    Within each day the (pnl, exit_minute) outcomes are permuted across that
    day's own entry minutes. Entry count, time-of-day profile, ticker, direction
    and the exit policy are all preserved exactly; only the signal's timing
    information is destroyed. The sequential guard is then re-applied, because
    which fills survive the one-position rule is itself part of what the timing
    buys.
    """
    by_day: dict = {}
    for d, m, pnl, xm in sim:
        by_day.setdefault(d, []).append((m, pnl, xm))
    out = []
    for d, items in by_day.items():
        mods = sorted(m for m, _, _ in items)
        outc = [(p, x) for _, p, x in items]
        rng.shuffle(outc)
        for m, (p, x) in zip(mods, outc):
            out.append((d, m, p, x))
    out.sort(key=lambda t: (t[0], t[1]))
    return seq_apply(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--pcts", nargs="*", type=int, default=GRID_PCT)
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="_step1_grid.parquet")
    ap.add_argument("--append", action="store_true",
                    help="merge into --out instead of replacing (one ticker per "
                         "process keeps peak RAM to a single ticker's bars)")
    ap.add_argument("--report", action="store_true",
                    help="skip scoring, just analyse an existing --out")
    a = ap.parse_args()

    if a.report:
        report(pd.read_parquet(a.out), deployed_cells())
        return

    import directional_flow_backtester as D
    from config import WATCHLIST, TRAIL_PCT
    tickers = a.tickers or sorted(WATCHLIST.keys())
    dep = deployed_cells()
    rng = np.random.default_rng(a.seed)

    print(f"STEP-1 REDO -- unconditioned grid, deployed exit, full calendar")
    print(f"  {len(tickers)} tickers x 2 directions x {len(a.pcts)} thresholds = "
          f"{len(tickers)*2*len(a.pcts)} cells")
    print(f"  null: {a.nulls} within-day entry-time permutations per cell")
    print(f"  windows: PRE {PRE_LO}..{DEPLOY_LO}  IS ..{SPLIT}  OOS ..{SLICE_EDGES[-1]}\n")

    from check_config_walkforward import _flow_for

    rows = []
    for tk in tickers:                            # ticker OUTER: one bars read each
        # Build the flow series and trigger list ONCE per ticker and hand it to
        # every cell. build_candidates would otherwise re-read NETPREM{tk} and
        # re-run triggers_for + annotate_flow_pct for all 10 cells -- the same
        # work ten times over, and on the post-backfill lake that is the whole
        # runtime. Triggers are direction-agnostic; `_rule_matched_trigs` filters
        # by direction and threshold downstream, which is what varies per cell.
        flow = _flow_for(D, [tk])
        if flow.empty:
            print(f"  {tk}: no flow series, skipped")
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        for direction in ("CALL", "PUT"):
            for pct in a.pcts:
                r = cell_rule(tk, direction, pct)
                cand = sim_core.build_candidates(D, r, trigs=trigs, since=None)
                if not cand:
                    continue
                pol = sim_core.policy_for(r, TRAIL_PCT)
                em = sim_core.eod_mod(r)
                ref = sim_core.walk(cand, pol, em, fill="bot")
                s = score(ref)
                if not s or s["n"] < 20:
                    continue
                sim = presim(cand, pol, em)
                mine = seq_apply(sim)
                if len(mine) != len(ref) or not np.allclose(
                        [p for _, p in mine], [p for _, p in ref]):
                    raise SystemExit(
                        f"ABORT: seq_apply diverged from sim_core.walk on "
                        f"{tk} {direction} p{pct} ({len(mine)} vs {len(ref)} trades). "
                        f"The null would not be measuring the same book.")
                nulls = np.array([np.mean([p for _, p in null_draw(sim, rng)])
                                  for _ in range(a.nulls)])
                s.update(ticker=tk, dir=direction, pct=pct,
                         null_mean=float(np.mean(nulls)),
                         null_p95=float(np.percentile(nulls, 95)),
                         z=float((s["all"] - np.mean(nulls)) /
                                 (np.std(nulls) if np.std(nulls) > 0 else np.nan)),
                         beats=bool(s["all"] > np.percentile(nulls, 95)),
                         deployed="; ".join(dep.get((tk, direction), [])))
                rows.append(s)
        print(f"  {tk} done ({sum(1 for x in rows if x['ticker']==tk)} cells)", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("  no cells with >=20 trades")
        return
    import os
    if a.append and os.path.exists(a.out):
        old = pd.read_parquet(a.out)
        old = old[~old["ticker"].isin(df["ticker"].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(a.out, index=False)
    print(f"  wrote {len(df)} cells -> {a.out}")
    if a.append:
        return                                   # analysis happens in --report
    report(df, dep)


def report(df, dep):
    # `is` is a Python keyword, so itertuples() silently renames that column to
    # a positional `_N`. Rename it explicitly rather than index by position.
    df = df.rename(columns={"is": "is_"})
    n = len(df)
    print(f"\n{'='*118}\n  ALL CELLS, ranked by ALL-PERIOD mean  (n>=20 trades)\n{'='*118}")
    print(f"  {'cell':18} {'n':>4} {'all':>8} {'PRE':>8} {'IS':>8} {'OOS':>8} "
          f"{'win':>5} {'null':>8} {'z':>6}  deployed rule")
    for r in df.sort_values("all", ascending=False).head(25).itertuples():
        star = " *" if r.beats else "  "
        print(f"  {r.ticker+' '+r.dir+' p'+str(r.pct):18} {r.n:>4} "
              f"{r.all*100:>+7.1f}% {r.pre*100:>+7.1f}% {r.is_*100:>+7.1f}% "
              f"{r.oos*100:>+7.1f}% {r.win:>5.2f} {r.null_mean*100:>+7.1f}% "
              f"{r.z:>6.2f}{star} {r.deployed[:34]}")

    print(f"\n{'='*118}\n  IS THE GRID DISTINGUISHABLE FROM CHANCE?\n{'='*118}")
    b = int(df["beats"].sum())
    print(f"  cells beating their OWN null's p95 : {b} of {n}")
    print(f"  expected by construction at p95     : {0.05*n:.1f}")
    print(f"  -> {'NO -- the grid is consistent with noise' if b <= 0.05*n*1.5 else 'more than chance'}"
          f"  ({b} vs {0.05*n:.1f})")
    print(f"  mean z across all cells: {df['z'].mean():+.2f}   "
          f"(0 = indistinguishable from random entries)")
    print(f"  cells with all-period mean > 0: {int((df['all']>0).sum())} of {n}"
          f"   ({100*(df['all']>0).mean():.0f}%)")

    print(f"\n{'='*118}\n  WHERE DO THE DEPLOYED RULES RANK?\n{'='*118}")
    df["rank"] = df["all"].rank(ascending=False).astype(int)
    dd = df[df["deployed"] != ""]
    if dd.empty:
        print("  no deployed ticker/direction pair produced a scorable cell")
    else:
        print(f"  {'cell':18} {'rank':>6} {'all':>8} {'OOS':>8} {'z':>6}  rule")
        for r in dd.sort_values("rank").itertuples():
            print(f"  {r.ticker+' '+r.dir+' p'+str(r.pct):18} {r.rank:>3}/{n:<3} "
                  f"{r.all*100:>+7.1f}% {r.oos*100:>+7.1f}% {r.z:>6.2f}  {r.deployed[:44]}")
    print(f"\n  A deployed rule ranking mid-pack is NOT evidence it was badly chosen:")
    print(f"  these cells are UNCONDITIONED, and the deployed rules add regime/hour")
    print(f"  gates on top. The comparison is 'was this ticker/direction a")
    print(f"  reasonable base to build on', not 'is the rule optimal'.")
    print(f"\n  {n} cells tested. Grid written to _step1_grid.parquet.")


if __name__ == "__main__":
    main()
