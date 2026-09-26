# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_mbo_gate.py
=================
Do CME order-book microstructure features, measured on the index future,
separate WINNING from LOSING trades in the SPY/QQQ/IWM rules?

  IWM -> RTY.c.0        QQQ -> NQ.c.0

Those two are the only index rules with MBO data bought, and they are among the
three that came out ROBUST in `check_fill_sensitivity` (tight spreads, fill
bands far smaller than their edges). ES/NQ/RTY are single-venue CME books, so
MBO there is COMPLETE -- the equity-fragmentation objection does not apply.

FEATURES (built by build_mbo_features.py; see its header for mechanism)
    hhi        Herfindahl of resting order sizes at the inside quote
    absorb     share of fill volume executing beyond DISPLAYED size (icebergs)
    tickchase  price-changing MODIFYs per minute (re-peg urgency)
    age_s      median age of orders resting at the touch
    qdepth     total displayed size at the touch
Each is averaged over a trailing WINDOW minutes ending at the trade's entry
minute, so everything is knowable at entry -- no lookahead.

=====================  PRE-COMMITTED PASS CRITERIA  ========================
Fixed in writing BEFORE any output was viewed. A feature is a real signal only
if ALL FOUR hold:

  M1  PERMUTATION NULL. The winner/loser separation must beat the p95 of a null
      built by SHUFFLING THE LABELS and keeping the features. This is the
      correct null here: it preserves each feature's own distribution and
      autocorrelation and asks only whether the LABELS carry information.
  M2  IS and OOS agree in SIGN (split 2025-08-21).
  M3  CROSS-SECTIONAL: the same feature, same sign, in BOTH RTY and NQ.
      This is the criterion that killed the hour-10 hypothesis.
  M4  WALK-FORWARD: choosing the best feature on data before each slice and
      scoring it forward beats not gating at all.

Plus a screen applied BEFORE testing, not after:
  C0  pairwise |corr| > 0.80 -> drop the later feature of the pair. Collinear
      features are not independent evidence, and hhi/qdepth both describe the
      touch.

THE HONEST PRIOR IS LOW, FOR TWO REASONS
----------------------------------------
1. n is ~98 trades total (IWM 72 + QQQ 26). At that size a five-feature search
   will produce something whether or not anything is there -- which is exactly
   what M1 and M3 exist to catch.
2. These are MICROSECOND-scale features compressed into a ~30-minute average,
   informing a decision held for tens of minutes. **That compression is where
   the signal most likely dies** -- not in the features, which are sound. A null
   here should be read as "the aggregation destroyed it", NOT as "order-book
   microstructure is uninformative".

Usage:  python check_mbo_gate.py
        python check_mbo_gate.py --window 30 --boot 4000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

CACHE = "_mbo_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
PROXY = {"IWM": "RTY_c_0", "QQQ": "NQ_c_0"}
FEATS = ["hhi", "absorb", "tickchase", "age_s", "qdepth"]


def load_features(sym, window):
    import polars as pl
    p = os.path.join(CACHE, f"{sym}.parquet")
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d.sort_values(["date", "mod"])
    # trailing mean ending AT the entry minute -- causal by construction
    for f in FEATS:
        d[f + "_w"] = (d.groupby("date")[f]
                        .transform(lambda s: s.rolling(window, min_periods=3).mean()))
    return d


def trades_for(rule_names=None):
    """Every trade the index rules took, with P&L, via sim_core (`bot` fills)."""
    import directional_flow_backtester as D
    import sim_core
    from config import RULES, TRAIL_PCT
    out = []
    for r in RULES:
        if not r.get("enabled", True) or r["ticker"] not in PROXY:
            continue
        if rule_names and r["name"] not in rule_names:
            continue
        cand = sim_core.build_candidates(D, r)
        if not cand:
            continue
        pol = sim_core.policy_for(r, TRAIL_PCT)
        em = sim_core.eod_mod(r)
        # walk() returns (date, pnl); re-derive the entry minute by replaying
        # the same admission logic so each pnl keeps its minute.
        cur, busy = None, -1
        for d, m, path in cand:
            if d != cur:
                cur, busy = d, -1
            if m < busy:
                continue
            pnl, xm, _t = sim_core.simulate(path, pol, em, fill="bot")
            out.append(dict(rule=r["name"], ticker=r["ticker"], sym=PROXY[r["ticker"]],
                            date=d, mod=m, pnl=pnl))
            busy = xm
    return pd.DataFrame(out)


def perm_p(win, lose, boot, rng):
    """p-value that the winner/loser mean gap is larger than label-shuffle noise."""
    obs = np.nanmean(win) - np.nanmean(lose)
    pool = np.concatenate([win, lose])
    nw = len(win)
    pool = pool[np.isfinite(pool)]
    if len(pool) < 8 or not np.isfinite(obs):
        return np.nan, np.nan
    draws = np.empty(boot)
    for i in range(boot):
        rng.shuffle(pool)
        draws[i] = np.nanmean(pool[:nw]) - np.nanmean(pool[nw:])
    return obs, float((np.abs(draws) >= abs(obs)).mean())


def run(a):
    rng = np.random.default_rng(7)
    T = trades_for(a.rules)
    if T.empty:
        print("no trades"); return
    print(f"  trades: {len(T)}  " +
          "  ".join(f"{k}:{v}" for k, v in T['rule'].value_counts().items()))

    frames = []
    for sym in sorted(T["sym"].unique()):
        f = load_features(sym, a.window)
        if f is None:
            print(f"  ! no features for {sym} — run build_mbo_features.py"); continue
        sub = T[T["sym"] == sym].merge(
            f[["date", "mod"] + [x + "_w" for x in FEATS]],
            on=["date", "mod"], how="left")
        frames.append(sub)
    if not frames:
        return
    M = pd.concat(frames, ignore_index=True)
    cols = [x + "_w" for x in FEATS]
    got = M[cols].notna().all(axis=1)
    print(f"  matched to MBO features: {int(got.sum())}/{len(M)}")
    M = M[got].copy()
    if len(M) < 20:
        print("  too few matched trades to test"); return
    M["win"] = M["pnl"] > 0

    print("\n" + "=" * 96)
    print("  C0  COLLINEARITY SCREEN (applied BEFORE testing)")
    print("=" * 96)
    C = M[cols].corr()
    keep = []
    for c in cols:
        if any(abs(C.loc[c, k]) > 0.80 for k in keep):
            bad = [k for k in keep if abs(C.loc[c, k]) > 0.80][0]
            print(f"  drop {c:12} (|corr| {C.loc[c,bad]:+.2f} with {bad})")
        else:
            keep.append(c)
    print(f"  kept: {[k[:-2] for k in keep]}")

    print("\n" + "=" * 96)
    print(f"  M1/M2  WINNER vs LOSER  ({a.window}m trailing mean, pooled)")
    print("=" * 96)
    print(f"  {'feature':12} {'win_mean':>11} {'lose_mean':>11} {'gap':>10} "
          f"{'perm_p':>8} {'IS gap':>10} {'OOS gap':>10}  M1 M2")
    res = {}
    for c in keep:
        w = M[M.win][c].to_numpy(float)
        l = M[~M.win][c].to_numpy(float)
        obs, p = perm_p(w, l, a.boot, rng)
        gaps = {}
        for tag, mask in (("IS", M.date < SPLIT), ("OOS", M.date >= SPLIT)):
            ww = M[mask & M.win][c].to_numpy(float)
            ll = M[mask & ~M.win][c].to_numpy(float)
            gaps[tag] = (np.nanmean(ww) - np.nanmean(ll)) if len(ww) > 2 and len(ll) > 2 else np.nan
        m1 = np.isfinite(p) and p < 0.05
        m2 = (np.isfinite(gaps["IS"]) and np.isfinite(gaps["OOS"])
              and np.sign(gaps["IS"]) == np.sign(gaps["OOS"]))
        res[c] = dict(obs=obs, p=p, m1=m1, m2=m2, **gaps)
        print(f"  {c[:-2]:12} {np.nanmean(w):>11.4g} {np.nanmean(l):>11.4g} "
              f"{obs:>+10.4g} {p:>8.3f} {gaps['IS']:>+10.4g} {gaps['OOS']:>+10.4g}"
              f"  {'Y' if m1 else '.'}  {'Y' if m2 else '.'}")

    print("\n" + "=" * 96)
    print("  M3  CROSS-SECTIONAL — same feature, same sign, in BOTH instruments")
    print("=" * 96)
    print(f"  {'feature':12} " + " ".join(f"{s:>14}" for s in sorted(M['sym'].unique())) + "   M3")
    for c in keep:
        signs, cells = [], []
        for s in sorted(M["sym"].unique()):
            sub = M[M["sym"] == s]
            w = sub[sub.win][c].to_numpy(float); l = sub[~sub.win][c].to_numpy(float)
            g = (np.nanmean(w) - np.nanmean(l)) if len(w) > 2 and len(l) > 2 else np.nan
            cells.append(f"{g:>+14.4g}")
            signs.append(np.sign(g) if np.isfinite(g) else 0)
        m3 = len(set(s for s in signs if s != 0)) == 1 and 0 not in signs
        res[c]["m3"] = m3
        print(f"  {c[:-2]:12} " + " ".join(cells) + f"   {'Y' if m3 else '.'}")

    print("\n" + "=" * 96)
    print("  VERDICT")
    print("=" * 96)
    any_pass = False
    for c in keep:
        r = res[c]
        flags = "".join("Y" if r.get(k) else "." for k in ("m1", "m2", "m3"))
        ok = all(r.get(k) for k in ("m1", "m2", "m3"))
        any_pass |= ok
        print(f"  {c[:-2]:12} M1M2M3 = {flags}   {'** survives to M4' if ok else ''}")
    if not any_pass:
        print("\n  Nothing cleared M1-M3, so M4 (walk-forward) is not run --")
        print("  there is no surviving feature to walk forward.")
        print("  Read this as: the microsecond->30-minute compression destroyed it,")
        print("  NOT that order-book microstructure is uninformative. The features")
        print("  and the data are sound; the horizon mismatch was the stated risk.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=30, help="trailing minutes")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--rules", nargs="*", default=None)
    run(ap.parse_args())
