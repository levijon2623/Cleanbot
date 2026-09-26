# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_hedge_participation.py
============================
REFINEMENT of check_flow_accumulation, from the user:

  "Dealer hedging, even in Negative GEX, is a directional force that can only
   affect price when it's stronger than any opposing market participation."

Correct, and it is the standard microstructure statement: price impact is
roughly ORDER FLOW IMBALANCE / MARKET DEPTH. check_flow_accumulation measured
only the numerator. A $50M hedging requirement is noise in a name trading $50M
a minute and a shove in one trading $2M a minute.

WHY THIS IS NOT check_volume_signal AGAIN
-----------------------------------------
check_volume_signal tested underlying volume as a standalone LEVEL (relvol,
VWAP distance, volume-confirmed flow, cumulative pace) -- all null. This tests
volume as a DENOMINATOR. A ratio is a different object from either of its parts,
and the null on the parts says nothing about the ratio.

CAUSALITY -- the two tests must not be conflated
------------------------------------------------
The opposing participation happens AFTER the trigger, so it is NOT knowable at
entry. Therefore:

  DIAGNOSTIC (uses forward volume -- LOOKAHEAD, never deployable): does
      participation in the window we actually hold moderate the outcome? This
      tests whether the MECHANISM is true. A lookahead test is legitimate for
      confirming physics; it is not a trading rule.

  CAUSAL (trailing volume only -- deployable): volume is strongly
      autocorrelated, so participation to date proxies participation to come.
      This is the version that could become a gate.

Both are printed and clearly labelled. Do not quote the diagnostic as tradeable.

THE PREDICTION BEING TESTED
---------------------------
Under NEGATIVE GEX with ALIGNED flow (dealers short gamma, hedging WITH the
move), the trade should work BEST when hedging pressure is LARGE RELATIVE TO
participation -- a thin tape lets the hedge move price -- and worst when heavy
participation absorbs it. Under POSITIVE GEX the interaction should be weak or
inverted, because dealers are damping rather than amplifying.

PRESSURE RATIO
--------------
    pressure = |cumulative net premium at the trigger| / underlying dollar volume

Premium dollars and traded notional are not the same unit, so the raw ratio is
not interpretable across tickers -- it is RANKED INTO QUARTILES WITHIN EACH
TICKER before pooling, which removes per-name scale entirely.

LIMITATION worth naming: net PREMIUM is a crude proxy for the share-equivalent
hedge. The true quantity is sum((ask_vol - bid_vol) * delta * 100), which needs
the per-contract greeks that the `_screen_cache` / `opt_bars_atm` caches DROP.
check_aggressor_imbalance.py already implements that "delta" construction; if
the premium version shows anything here, escalate to it before believing it.

Usage:  python check_hedge_participation.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
from check_exit_walkforward import _eod_mod
from check_flow_accumulation import _build, _seq

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
OPEN_MOD, CLOSE_MOD = 9 * 60 + 30, 16 * 60


def _dollar_volume(tk):
    """{date: (mods, cum_dollar_vol_to_date, per_minute_dollar_vol)} RTH only."""
    import polars as pl
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return {}
    d = pl.read_parquet(p, columns=["start_time", "close", "volume"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= OPEN_MOD) & (d["mod"] <= CLOSE_MOD)].copy()
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
    d["dv"] = d["close"] * d["volume"]
    d = d.dropna(subset=["dv"]).sort_values(["date", "mod"])
    out = {}
    for dt, g in d.groupby("date"):
        mods = g["mod"].to_numpy(int)
        dv = g["dv"].to_numpy(float)
        out[dt] = (mods, np.cumsum(dv), dv)
    return out


def _qcut_within(df, col, key="ticker", q=4):
    """Rank into quartiles WITHIN each ticker, then pool -- removes per-name scale."""
    out = pd.Series(np.nan, index=df.index)
    for tk, g in df.groupby(key):
        v = g[col].dropna()
        if len(v) < q * 4:
            continue
        try:
            out.loc[v.index] = pd.qcut(v, q, labels=False, duplicates="drop")
        except ValueError:
            continue
    return out


def _st(lbl, df):
    if len(df) < 12:
        return f"    {lbl:36} n={len(df):>4}  (thin)"
    v = df["pnl"].to_numpy(float)
    i = df[df.date < SPLIT]["pnl"].to_numpy(float)
    o = df[df.date >= SPLIT]["pnl"].to_numpy(float)
    sl = [[] for _ in range(6)]
    for d, p in zip(df["date"], v):
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    pop = [np.mean(b) for b in sl if len(b) >= 3]
    return (f"    {lbl:36} n={len(v):>4} d={df['date'].nunique():>4} "
            f"IS {i.mean()*100 if len(i) else float('nan'):>+7.1f}% "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+7.1f}% "
            f"win {(v>0).mean():>4.2f} sl {sum(1 for x in pop if x>0)}/{len(pop)}")


def run(a):
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    rules = [r for r in RULES if r.get("enabled", True)]
    # per-rule deployed exit: META/NVDA carry trail_pct 0, so a book-wide
    # trail would score trades those rules never take
    import sim_core
    dv_cache, rows = {}, []
    for r in rules:
        c = _build(D, r)
        if not c:
            continue
        tk = r["ticker"]
        if tk not in dv_cache:
            dv_cache[tk] = _dollar_volume(tk)
        dv = dv_cache[tk]
        for t in _seq(c, sim_core.policy_for(r, TRAIL_PCT), _eod_mod(r)):
            e = dv.get(t["date"])
            if e is None:
                continue
            mods, cumdv, perdv = e
            j = int(np.searchsorted(mods, t["mod"], side="right")) - 1
            if j < 1:
                continue
            to_date = float(cumdv[j])
            k = int(np.searchsorted(mods, t["mod"] + 60, side="right"))
            fwd = float(cumdv[min(k, len(cumdv) - 1)] - cumdv[j])
            if to_date <= 0 or fwd <= 0:
                continue
            t["dv_todate"] = to_date
            t["dv_fwd"] = fwd
            t["press_causal"] = abs(t["cum"]) / to_date          # CAUSAL
            t["press_fwd"] = abs(t["cum"]) / fwd                 # LOOKAHEAD
            rows.append(t)
        print(f"  {r['name']:26} {len(c):>5} triggers")

    R = pd.DataFrame(rows)
    if R.empty:
        print("nothing"); return
    R["aligned"] = R["cum_dir"] > 0
    R["qc"] = _qcut_within(R, "press_causal")
    R["qf"] = _qcut_within(R, "press_fwd")

    print("\n" + "=" * 116)
    print("  0. BASELINE")
    print("=" * 116)
    print(_st("ALL trades", R))

    print("\n" + "=" * 116)
    print("  1. CAUSAL -- pressure / participation TO DATE (deployable)")
    print("     Q1 = hedging small vs the tape ... Q4 = hedging LARGE vs the tape")
    print("=" * 116)
    for q in sorted(R["qc"].dropna().unique()):
        print(_st(f"Q{int(q)+1} press_causal", R[R["qc"] == q]))

    print("\n" + "=" * 116)
    print("  2. DIAGNOSTIC -- pressure / participation in the FORWARD 60m")
    print("     *** LOOKAHEAD. Tests whether the MECHANISM is real. NOT a trading rule. ***")
    print("=" * 116)
    for q in sorted(R["qf"].dropna().unique()):
        print(_st(f"Q{int(q)+1} press_fwd", R[R["qf"] == q]))

    print("\n" + "=" * 116)
    print("  3. THE PREDICTION -- within NEGATIVE GEX + ALIGNED, does thin participation help?")
    print("     theory: dealers short gamma hedge WITH the move; a thin tape lets it land.")
    print("=" * 116)
    for g in ("NEGATIVE", "POSITIVE"):
        for al, tag in ((True, "aligned"), (False, "opposed")):
            sub = R[(R["gex"] == g) & (R["aligned"] == al)]
            if len(sub) < 24:
                continue
            print(f"    -- {g} GEX / {tag}  (n={len(sub)}) --")
            lo = sub[sub["qc"].isin([0, 1])]
            hi = sub[sub["qc"].isin([2, 3])]
            print(_st("       causal: thin pressure (Q1-2)", lo))
            print(_st("       causal: HEAVY pressure (Q3-4)", hi))
            lof = sub[sub["qf"].isin([0, 1])]
            hif = sub[sub["qf"].isin([2, 3])]
            print(_st("       fwd*:   absorbed (Q1-2)", lof))
            print(_st("       fwd*:   HEAVY vs tape (Q3-4)", hif))

    print("\n" + "=" * 116)
    print("  4. Does the ratio explain the U-SHAPE that falsified plain accumulation?")
    print("     (cum_dir quartiles, split by causal pressure)")
    print("=" * 116)
    R["qd"] = pd.qcut(R["cum_dir"], 4, labels=False, duplicates="drop")
    for q in sorted(R["qd"].dropna().unique()):
        sub = R[R["qd"] == q]
        lo = sub[sub["qc"].isin([0, 1])]; hi = sub[sub["qc"].isin([2, 3])]
        print(f"    cum_dir Q{int(q)+1}:")
        print(_st("       thin pressure", lo))
        print(_st("       HEAVY pressure", hi))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run(ap.parse_args())
