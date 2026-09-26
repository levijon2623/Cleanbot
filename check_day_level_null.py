# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_day_level_null.py
=======================
THREE CONTROLS AT THREE LEVELS, because "does the trigger work?" is not one
question and the answer depends on which level you hold fixed.

THE PROBLEM WITH THE EARLIER CONTROLS (user's catch, and it is correct)
    check_peak_profit re-timed each trigger to a random minute WITHIN THE SAME
    DAY; check_structures drew its random entry minute for the SAME DAY too.
    Both therefore hold the DAY fixed. If the trigger's information is mostly
    about WHICH DAYS ARE WORTH TRADING -- entirely plausible for a cumulative
    net-premium EMA crossover -- then the control inherits exactly that
    information and the comparison is blind to it by construction.

    Those results establish only that the trigger's WITHIN-DAY MINUTE selection
    carries nothing. They do not test day selection at all.

THE THREE ARMS
    real      the actual trigger: this day, this minute
    within    same ticker, SAME DAY, random minute in the same hour
              -> isolates MINUTE selection, blind to day selection
    offday    same ticker, SAME MINUTE-OF-DAY, a DIFFERENT date on which this
              ticker had NO trigger within +/-`--window` minutes of it
              -> isolates the TRIGGER MOMENT while allowing the control day to
                 be otherwise active. This is the user's proposal.
    quiet     same ticker, SAME MINUTE-OF-DAY, a date with NO trigger AT ALL
              -> tests DAY selection in its strongest form, but note the
                 control days are themselves a selected population (quiet days
                 are low-flow days), so a gap here conflates "the trigger
                 fired" with "this was an active day"

    Control dates are drawn from within +/-`--near-sessions` of the real date
    so calendar regime drift does not masquerade as an effect.

    Time-of-day is held fixed in `offday`/`quiet`, which is what keeps the
    TIME-BUDGET confound out (see check_policy_matrix: an outcome's time budget
    must match its entry minute, or the null manufactures a gap).

REPORTED AT TWO LEVELS
    MFE -- exit-independent, so it sees the trigger without the exit policy
    P&L under the deployed trail50 -- what it would actually have earned

Usage:
  python check_day_level_null.py --tickers SPY QQQ IWM --pct 65
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd

from check_config_walkforward import SLICE_EDGES
import sim_core

PRE_LO = pd.Timestamp("2023-10-12").date()
DEPLOY_LO = SLICE_EDGES[0]
SPLIT = pd.Timestamp("2025-08-21").date()
GRID = [50, 65, 80, 90, 95]


def _mk(date, ts, direction):
    """A synthetic UNCONDITIONAL trigger: thr=0 so it clears any threshold."""
    return {"date": date, "ts": ts, "hour": pd.Timestamp(ts).hour,
            "dir": direction, "abs_flow": 1.0, "thr": {p: 0.0 for p in GRID}}


def build_controls(trigs, pct, window, near, rng):
    """-> (within, offday, quiet) synthetic trigger lists matched to `trigs`."""
    passing = []
    for t in trigs:
        thr = t.get("thr")
        if thr and pct in thr and abs(float(t["abs_flow"])) >= thr[pct]:
            passing.append(t)
    if not passing:
        return [], [], []

    # Minute index of the PASSING triggers per date -- i.e. the minutes the bot
    # would actually have entered. Indexing on ALL raw crossovers instead makes
    # the control unconstructible: the bare EMA crossover fires ~23x per
    # ticker-day, so no date has a 30-minute window without one and `quiet` came
    # back empty. The control must mean "the bot would NOT have entered here",
    # which is about passing triggers, not raw crossings.
    by_date: dict = {d: [] for d in {t["date"] for t in trigs}}
    for t in passing:
        ts = pd.Timestamp(t["ts"])
        by_date[t["date"]].append(ts.hour * 60 + ts.minute)
    all_dates = sorted(by_date)
    quiet_dates = [d for d in all_dates if not by_date[d]]
    pos = {d: i for i, d in enumerate(all_dates)}

    within, offday, quiet = [], [], []
    for t in passing:
        ts = pd.Timestamp(t["ts"])
        m = ts.hour * 60 + ts.minute
        d = t["date"]

        # --- within: same day, random minute in the same hour
        within.append(_mk(d, ts.replace(minute=int(rng.integers(0, 60))), t["dir"]))

        # --- offday: same minute-of-day, a nearby date with no trigger near m
        i = pos.get(d)
        if i is not None:
            lo, hi = max(0, i - near), min(len(all_dates), i + near + 1)
            cands = [x for x in all_dates[lo:hi]
                     if x != d and not any(abs(mm - m) <= window for mm in by_date[x])]
            if cands:
                d2 = cands[int(rng.integers(0, len(cands)))]
                offday.append(_mk(d2, pd.Timestamp(d2).replace(
                    hour=ts.hour, minute=ts.minute), t["dir"]))

        # --- quiet: same minute-of-day, a date with no triggers at all
        if quiet_dates:
            d3 = quiet_dates[int(rng.integers(0, len(quiet_dates)))]
            quiet.append(_mk(d3, pd.Timestamp(d3).replace(
                hour=ts.hour, minute=ts.minute), t["dir"]))
    return within, offday, quiet


def measure(D, rule, trigs, em, pol):
    """-> (mfe_list, pnl_list) for one arm."""
    cand = sim_core.build_candidates(D, rule, trigs=trigs, since=None)
    if not cand:
        return [], []
    mfe, pnl = [], []
    for d, m, path in cand:
        e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
        if e_mid <= 0:
            continue
        k = int(np.searchsorted(mods, em, side="right"))
        if k < 2:
            continue
        b = np.asarray(bid[:k], float)
        mfe.append((d, b.max() / e_mid - 1.0))
    for d, p in sim_core.walk(cand, pol, em, fill="bot"):
        pnl.append((d, p))
    return mfe, pnl


def boot_indep(a_rows, b_rows, n=4000, seed=83):
    """Day-block bootstrap of mean(a) - mean(b), resampling each arm by its OWN
    sessions.

    The arms live on DIFFERENT dates by construction -- a control entry sits on
    a control day -- so there is no shared calendar to resample jointly and no
    pairing to preserve. Resampling each independently is the honest version;
    pairing them would invent a correspondence that does not exist. Day blocks
    still matter within each arm, because several entries share a session.
    """
    if not a_rows or not b_rows:
        return np.nan, np.nan, np.nan
    A: dict = {}
    for d, v in a_rows:
        A.setdefault(d, []).append(v)
    B: dict = {}
    for d, v in b_rows:
        B.setdefault(d, []).append(v)
    da, db = list(A), list(B)
    obs = (np.mean([v for r in A.values() for v in r])
           - np.mean([v for r in B.values() for v in r]))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        xa = np.concatenate([A[da[k]] for k in rng.choice(len(da), len(da), True)])
        xb = np.concatenate([B[db[k]] for k in rng.choice(len(db), len(db), True)])
        out.append(xa.mean() - xb.mean())
    return obs, *np.percentile(out, [2.5, 97.5])


def summar(rows, label, unit="%"):
    if not rows:
        print(f"    {label:28} (none)")
        return None
    v = np.array([r[1] for r in rows], float)
    d = np.array([r[0] for r in rows])
    w = []
    for lbl, lo, hi in (("PRE", PRE_LO, DEPLOY_LO), ("IS", DEPLOY_LO, SPLIT),
                        ("OOS", SPLIT, SLICE_EDGES[-1])):
        m = (d >= lo) & (d < hi)
        w.append(v[m].mean() * 100 if m.sum() else np.nan)
    print(f"    {label:28} n={len(v):>6}  mean {v.mean()*100:>+7.1f}{unit}   "
          f"PRE {w[0]:>+7.1f}  IS {w[1]:>+7.1f}  OOS {w[2]:>+7.1f}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--window", type=int, default=30,
                    help="offday: no trigger within +/- this many minutes")
    ap.add_argument("--near-sessions", type=int, default=20,
                    help="draw control dates within +/- this many sessions")
    ap.add_argument("--seed", type=int, default=67)
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import TRAIL_PCT
    rng = np.random.default_rng(a.seed)
    pol = dict(name="trail50", kind="trail", trail=0.50)

    ARMS = ("real", "within", "offday", "quiet")
    mfe = {k: [] for k in ARMS}
    pnl = {k: [] for k in ARMS}

    for tk in a.tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        within, offday, quiet = build_controls(
            trigs, a.pct, a.window, a.near_sessions, rng)
        print(f"  {tk}: controls built -- within {len(within)}, "
              f"offday {len(offday)}, quiet {len(quiet)}", flush=True)
        for direction in a.dirs:
            rule = {"name": f"{tk} {direction}", "ticker": tk,
                    "direction": direction, "dte": [0, 1],
                    "min_flow_pct": a.pct, "target_roe": 1.0, "rr": 1.0}
            em = sim_core.eod_mod(rule)
            for arm, src in (("real", trigs), ("within", within),
                             ("offday", offday), ("quiet", quiet)):
                if not src:
                    continue
                f, p = measure(D, rule, src, em, pol)
                mfe[arm] += f
                pnl[arm] += p
        print(f"  {tk} done", flush=True)

    print(f"\n{'='*112}")
    print(f"  THREE-LEVEL CONTROL  (p{a.pct}, {' '.join(a.tickers)})")
    print(f"{'='*112}")
    print(f"  PEAK PROFIT (MFE @bid) -- exit-independent")
    ref = {}
    for arm in ARMS:
        ref[arm] = summar(mfe[arm], arm)
    print(f"\n  P&L under the deployed trail50")
    refp = {}
    for arm in ARMS:
        refp[arm] = summar(pnl[arm], arm)

    # persist so the bootstrap can be re-run without rebuilding candidates
    recs = []
    for arm in ARMS:
        for src, kind in ((mfe[arm], "mfe"), (pnl[arm], "pnl")):
            for d, v in src:
                recs.append(dict(arm=arm, kind=kind, date=d, value=v))
    pd.DataFrame(recs).to_parquet("_day_level_null.parquet", index=False)

    print(f"\n{'='*112}\n  GAPS vs REAL, with day-block bootstrap 95% CI\n{'='*112}")
    print("  Arms sit on DIFFERENT dates (a control entry is on a control day),")
    print("  so the two arms are resampled INDEPENDENTLY by their own sessions")
    print("  rather than paired -- pairing across different days would be fiction.")
    for lbl, ref_, raw in (("peak profit", ref, mfe),
                           ("trail50 return", refp, pnl)):
        if ref_["real"] is None:
            continue
        print(f"\n  {lbl}:")
        for arm in ("within", "offday", "quiet"):
            if ref_.get(arm) is None:
                continue
            obs, c1, c2 = boot_indep(raw["real"], raw[arm])
            sig = np.isfinite(c1) and (c1 > 0 or c2 < 0)
            print(f"    real - {arm:8} {obs*100:>+7.2f}pp   "
                  f"95% CI [{c1*100:>+6.2f}, {c2*100:>+6.2f}]{'  *' if sig else ''}")
    print(f"\n  `within` blind to day selection (day held fixed).")
    print(f"  `offday` is the clean test of the TRIGGER MOMENT.")
    print(f"  `quiet`  tests DAY selection but its control days are themselves")
    print(f"           a selected (low-flow) population -- read it as an upper")
    print(f"           bound that conflates 'trigger fired' with 'active day'.")
    print(f"\n  No holdout remains; this is diagnosis, not a deployment case.")


if __name__ == "__main__":
    main()
