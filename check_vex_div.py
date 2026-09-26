# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_vex_div.py
================
VANNA EXPOSURE x REALISED IV CHANGE -- the one form of the vanna hypothesis
`check_charm_vanna.py` could not test.

WHY THIS IS NOT A REPEAT
    check_charm_vanna (session 16) tested vanna as a directional predictor and
    found nothing.  But it tested the vanna LEVEL and its z-score, never the
    product the mechanism actually calls for.  Its own line 289 says so:
        "IV proxy change: we don't have IV, but vanna itself * sign of a vol
         move. use realised: ... unavailable; skip -> vanna raw only."
    The silver lake does carry `iv_close`, and `build_iv_surface.py` now has it
    cached, so the product is finally computable.

THE HYPOTHESIS (user's)
    "Vanna measures how a change in IV affects the delta of the market maker's
     book.  If VEX is highly positive, an IV spike forces dealers to buy the
     underlying, creating a synthetic trend."
    => signal = sign(VEX) x dIV  should predict the FORWARD underlying return.
    Positive VEX + IV rising  -> dealers buy  -> price up.
    Positive VEX + IV falling -> dealers sell -> price down.

WHAT WOULD FALSIFY IT
    A flat or sign-inconsistent quintile spread on the forward return, or an
    effect that does not survive the 1.2bp CME / 9.0bp perp cost bar.

METHOD -- the traps this repo has already fallen into, avoided by construction
  * NON-OVERLAPPING windows.  The forward horizon h is sampled every h minutes,
    never every minute.  Overlapping windows inflated a prior result from
    +5.15bp (p=0.771) to +51.89bp purely through autocorrelation; see
    METHODOLOGY.md 2a.
  * dIV IS MEASURED BACKWARD (t-30min -> t) and the target FORWARD (t -> t+h),
    so nothing contemporaneous leaks into the predictor.
  * dte>=1 IV ONLY.  The lake's 0DTE `iv_close` is corrupt over most of this
    sample (see check_ivr_termstructure.valid_0dte); the nearest SOUND expiry
    is used instead.
  * DAY-BLOCK bootstrap, resampling whole sessions -- same-day buckets are
    heavily autocorrelated and a bucket-level resample would fake precision.
  * CONTROLS reported alongside: dIV alone and sign(VEX) alone.  If the product
    does no better than its parts, there is no interaction.

Usage:
  python check_vex_div.py
  python check_vex_div.py --horizons 15 30 60 --lookback 30
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

CV = "_cv_cache"
IVS = "_ivs_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
#: tickers present in BOTH caches
TK = ["AVGO", "GLD", "IWM", "META", "MSFT", "NVDA", "QQQ", "SPY"]


def panel(tk: str, lookback: int) -> pd.DataFrame | None:
    """15-minute panel for one ticker: vanna exposure, front IV, price."""
    fp_c, fp_i = os.path.join(CV, f"{tk}.parquet"), os.path.join(IVS, f"{tk}.parquet")
    if not (os.path.exists(fp_c) and os.path.exists(fp_i)):
        return None

    cv = pl.read_parquet(fp_c).to_pandas()
    cv = cv[(cv["mod"] >= 570) & (cv["mod"] <= 960)].copy()
    cv["mod15"] = (cv["mod"] // 15) * 15
    cv = (cv.sort_values(["date", "mod"])
            .groupby(["date", "mod15"]).agg(v_oi=("v_oi", "last"),
                                            price=("price", "last")).reset_index())

    iv = pl.read_parquet(fp_i).to_pandas()
    iv = iv[(iv["dte"] >= 1) & (iv["mod15"] >= 570) & (iv["mod15"] <= 960)]
    front = iv.groupby(["date", "mod15"])["dte"].transform("min")
    iv = (iv[iv["dte"] == front].groupby(["date", "mod15"])["iv"].mean()
          .reset_index().rename(columns={"iv": "ivf"}))

    for f in (cv, iv):
        f["date"] = pd.to_datetime(f["date"]).dt.date
    d = cv.merge(iv, on=["date", "mod15"], how="inner").sort_values(["date", "mod15"])
    if d.empty:
        return None

    # scale-free vanna magnitude: today's value against the ticker's own recent
    # norm, using PRIOR sessions only.
    day = d.groupby("date")["v_oi"].median()
    norm = day.abs().shift(1).rolling(60, min_periods=20).median()
    d["vex_n"] = d["v_oi"] / d["date"].map(norm)

    k = max(1, lookback // 15)
    g = d.groupby("date")
    d["dIV"] = (d["ivf"] - g["ivf"].shift(k)) * 100.0        # IV POINTS, backward
    d["tk"] = tk
    return d


def build(lookback: int) -> pd.DataFrame:
    out = []
    for t in TK:
        p = panel(t, lookback)
        if p is not None:
            out.append(p)
    return pd.concat(out, ignore_index=True)


def _boot(df: pd.DataFrame, col: str, n=3000, seed=5):
    """Day-block bootstrap of the top-minus-bottom quintile forward return."""
    rng = np.random.default_rng(seed)
    days = df["date"].unique()
    dmap = {d: g for d, g in df.groupby("date")}
    obs = _spread(df, col)
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days), replace=True)
        s = _spread(pd.concat([dmap[days[k]] for k in pick], ignore_index=True), col)
        if np.isfinite(s):
            out.append(s)
    if not out:
        return obs, np.nan, np.nan
    return obs, *np.percentile(out, [2.5, 97.5])


def _spread(df: pd.DataFrame, col: str) -> float:
    v = df[col].to_numpy()
    r = df["fwd"].to_numpy()
    if len(v) < 50:
        return np.nan
    lo, hi = np.percentile(v, [20, 80])
    a, b = r[v <= lo], r[v >= hi]
    if not len(a) or not len(b):
        return np.nan
    return float(b.mean() - a.mean())


def run(a):
    d = build(a.lookback)
    print(f"panel: {len(d):,} ticker-15min rows, {d['date'].nunique()} sessions, "
          f"{d['tk'].nunique()} tickers")

    for h in a.horizons:
        k = max(1, h // 15)
        g = d.groupby(["tk", "date"])
        d["fwd"] = (g["price"].shift(-k) / d["price"] - 1) * 1e4        # BASIS POINTS
        # NON-OVERLAPPING: keep one observation every k buckets within a session
        keep = d.groupby(["tk", "date"]).cumcount() % k == 0
        s = d[keep].dropna(subset=["fwd", "dIV", "vex_n"]).copy()
        s["signal"] = np.sign(s["vex_n"]) * s["dIV"]
        s["magxdiv"] = s["vex_n"] * s["dIV"]

        pos = (s["vex_n"] > 0).mean()
        print(f"\n{'='*92}\nHORIZON {h}min   (dIV measured backward over "
              f"{a.lookback}min)   n={len(s):,} NON-OVERLAPPING obs")
        print("=" * 92)
        print(f"  share VEX>0: {pos*100:.0f}%   "
              f"|dIV| median {s['dIV'].abs().median():.2f} pts   "
              f"fwd mean {s['fwd'].mean():+.2f}bp  sd {s['fwd'].std():.1f}bp")
        if pos > 0.99 or pos < 0.01:
            print("  !! VEX SIGN IS A CONSTANT for these names. 'sign(VEX) x dIV' is then")
            print("     IDENTICALLY dIV (up to one global flip), so the hypothesis as stated")
            print("     -- 'if VEX is highly positive, an IV spike forces dealers to buy' --")
            print("     has no contrast to exploit: the condition is ALWAYS met. The only")
            print("     testable remnant is the MAGNITUDE form, VEX_n x dIV, reported below.")
            print("     (check_charm_vanna found the identical degeneracy in CHARM's sign.)")

        for col, lbl in (("signal", "sign(VEX) x dIV   <- H"),
                         ("magxdiv", "VEX_n x dIV  (magnitude)"),
                         ("dIV", "dIV alone      (ctrl)"),
                         ("vex_n", "VEX alone      (ctrl)")):
            row = []
            for tag, sub in (("ALL", s), ("IS", s[s["date"] < SPLIT]),
                             ("OOS", s[s["date"] >= SPLIT])):
                row.append(f"{tag} {_spread(sub, col):>+8.2f}bp")
            obs, c1, c2 = _boot(s, col)
            star = "  *" if np.isfinite(c1) and (c1 > 0 or c2 < 0) else ""
            print(f"    {lbl:24} " + "  ".join(row)
                  + f"   95% CI [{c1:>+7.2f}, {c2:>+7.2f}]bp{star}")

        # quintile monotonicity on the product -- an interaction should be ordered
        q = pd.qcut(s["signal"], 5, labels=False, duplicates="drop")
        cells = [f"Q{i+1} {s['fwd'][q==i].mean():+.2f}" for i in sorted(set(q.dropna()))]
        print(f"    quintiles of sign(VEX)xdIV: " + "  ".join(cells) + "  (bp)")
    print(f"\n  COST BARS: CME futures round trip ~1.2bp, HL perp taker ~9.0bp.")
    print(f"  TESTS: {len(a.horizons)} horizons x 4 statistics = "
          f"{len(a.horizons)*4} (expect ~{len(a.horizons)*4*0.05:.1f} false positives at 95%).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=[15, 30, 60])
    ap.add_argument("--lookback", type=int, default=30)
    run(ap.parse_args())
