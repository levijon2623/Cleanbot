# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_peak_anatomy.py
=====================
WHAT WAS GOING ON WHEN THE PREMIUM PEAKED? -- a description, not a verdict.

THE SHIFT FROM check_peak_levels
    That script asks "do peaks cluster at level X more than chance?" and needs a
    null clean enough to carry a p-value. Three builds of it produced fully
    significant tables that its own placebos exposed as artifact (geometry,
    measurement asymmetry, trigger selection). The peak, the levels and the
    trigger are entangled enough that careless comparison always yields stars.

    This script does the prior job instead: LOCATE the peak in time, then look at
    the underlying in that same window and describe what was mechanically there.
    Locating the peak of PREMIUM does locate the underlying's turn -- they sit a
    median 2.9bp apart in price (pre-flight Q1) -- so the premium peak is a valid
    pointer at an underlying event.

THE ONE BASELINE THAT MATTERS HERE: LEVEL DENSITY
    "Peaks land near a level" is vacuous if the levels tile the day so densely
    that EVERY price is near one. So for each day this computes the expected
    distance from a UNIFORM-RANDOM price in the day's range to its nearest
    overhead level, and reports the peak's actual distance against it.
        ratio ~ 1.0  -> peaks are no closer than an arbitrary price. Vacuous.
        ratio << 1.0 -> peaks really do sit nearer levels than the tiling alone
                        explains, and the question is worth a proper null.
    This is geometry-free: it does not resample windows, so none of the three
    biases that broke check_peak_levels can enter it.

WHAT IT REPORTS
  A WHEN      time-of-day of peaks, and minutes from entry to peak
  B OVERHEAD  which level sits nearest above a CALL peak (below a PUT trough),
              how far in ATR, and how often the peak OVERSHOT it
  C DENSITY   the vacuity check above
  D CHARACTER the peak bar itself: volume vs the day's mean, range vs ATR,
              RSI(14) on 1m and 5m -- is the turn marked on the tape at all?
  E SPLITS    by peak size and by dealer-gamma regime

Usage:
  python check_peak_anatomy.py --tickers SPY QQQ IWM
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

from check_peak_levels import load_levels, peak_rows, LEVELS, STATIC, VWSPEC

RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
UP_LEVELS = [c for c, s, _ in LEVELS if s in ("up", "both") and not c.startswith("plc")]
DN_LEVELS = [c for c, s, _ in LEVELS if s in ("dn", "both") and not c.startswith("plc")]
LBL = {c: l for c, _, l in LEVELS}

#: VWAP and its bands are computed FROM the path, so a large up-move lifts both
#: the extreme and the +2sd band together -- "the peak stopped at the band" is
#: then partly self-referential. Everything else is fixed before the move (or,
#: for the initial balance, by 10:30) and is exogenous to it. Section C is
#: therefore reported BOTH ways; the exogenous number is the honest one.
ENDO = set(VWSPEC)
UP_EXO = [c for c in UP_LEVELS if c not in ENDO]
DN_EXO = [c for c in DN_LEVELS if c not in ENDO]


def und_map_v(tk):
    """1m bars WITH volume -- the anatomy needs the tape, not just the path."""
    d = pl.read_parquet(f"historical/{tk}.parquet",
                        columns=["start_time", "high", "low", "close", "volume"]).to_pandas()
    et = (pd.to_datetime(d["start_time"], utc=True)
          .dt.tz_convert("America/New_York").dt.tz_localize(None))
    d["date"] = et.dt.date
    d["mod"] = (et.dt.hour * 60 + et.dt.minute).astype(int)
    d = d[(d["mod"] >= RTH_LO) & (d["mod"] <= RTH_HI)]
    for c in ("high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    out = {}
    for dt, g in d.sort_values("mod").groupby("date"):
        cl = g["close"].to_numpy(float)
        out[dt] = dict(mod=g["mod"].to_numpy(np.int32), hi=g["high"].to_numpy(float),
                       lo=g["low"].to_numpy(float), cl=cl,
                       vol=g["volume"].to_numpy(float),
                       rsi1=_rsi(cl, 14), rsi5=_rsi(cl[::5], 14))
    return out


def _rsi(c, n=14):
    """Wilder RSI. Returns an array aligned to `c` (NaN for the warmup)."""
    out = np.full(c.size, np.nan)
    if c.size < n + 1:
        return out
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, d.size):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        out[i + 1] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def levels_on(tk, d, mod, DAY, VWF):
    lv = DAY.get(tk, {}).get(d)
    if lv is None:
        return None
    out = {c: lv[c] for c in STATIC if np.isfinite(lv.get(c, np.nan))}
    if mod < RTH_LO + 60:
        out.pop("ib_hi", None); out.pop("ib_lo", None)
    vm = VWF.get(tk, {}).get(d)
    if vm is not None:
        mods, vw, sd = vm
        j = int(np.searchsorted(mods, mod, side="right")) - 1
        if j >= 0 and np.isfinite(vw[j]):
            for c, k in VWSPEC.items():
                out[c] = vw[j] + k * sd[j]
    return out


def nearest_beyond(px, levels, cols, up):
    """Nearest level AT OR BEYOND the extreme -- the one that could have capped
    it. Returns (name, distance_in_price) or (None, nan)."""
    best, bd = None, np.inf
    for c in cols:
        v = levels.get(c)
        if v is None or not np.isfinite(v):
            continue
        gap = (v - px) if up else (px - v)
        if gap >= 0 and gap < bd:
            best, bd = c, gap
    return best, (bd if best else np.nan)


def density_baseline(levels, cols, lo, hi, up, atr, tol=0.10, n=400):
    """How near to a level an ARBITRARY price in the day's range would be -- the
    share of the story that is just the levels tiling the range.

    Returns (median distance, P(within `tol` ATR)). Both are needed: comparing
    the peak's own gap against a per-day MEDIAN works for medians, but a
    threshold share has to be compared against a PROBABILITY, not against
    'was the day's median below the threshold'. Mixing those two inflated the
    within-0.10-ATR contrast to 28.5% vs 3.5% when the medians were only 1.3x
    apart.
    """
    vals = [levels[c] for c in cols if np.isfinite(levels.get(c, np.nan))]
    if not vals or hi <= lo or not (np.isfinite(atr) and atr > 0):
        return np.nan, np.nan
    xs = np.linspace(lo, hi, n)
    v = np.asarray(sorted(vals), float)
    if up:
        idx = np.searchsorted(v, xs, side="left")
        d = np.where(idx < v.size, v[np.clip(idx, 0, v.size - 1)] - xs, np.nan)
    else:
        idx = np.searchsorted(v, xs, side="right") - 1
        d = np.where(idx >= 0, xs - v[np.clip(idx, 0, v.size - 1)], np.nan)
    d = d[np.isfinite(d)]
    if not d.size:
        return np.nan, np.nan
    return float(np.median(d)), float((d / atr <= tol).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="+", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    a = ap.parse_args()

    print("  loading peaks...", flush=True)
    df = peak_rows(a.tickers, a.dirs, a.pct)
    if df.empty:
        print("  no trades"); return
    DAY, VWF = load_levels(a.tickers)
    und = {tk: und_map_v(tk) for tk in a.tickers}

    rows = []
    for r in df.itertuples():
        b = und.get(r.ticker, {}).get(r.date)
        lv0 = DAY.get(r.ticker, {}).get(r.date)
        if b is None or lv0 is None:
            continue
        atr = lv0["atr"]
        if not (np.isfinite(atr) and atr > 0):
            continue
        up = r.dir == "CALL"
        mods = b["mod"]
        s = int(np.searchsorted(mods, r.entry, side="left"))
        e = int(np.searchsorted(mods, r.eod, side="right"))
        if e - s < 10:
            continue
        seg = b["hi"][s:e] if up else b["lo"][s:e]
        j = int(np.argmax(seg) if up else np.argmin(seg))
        k = s + j
        px, pm = float(seg[j]), int(mods[k])
        lv = levels_on(r.ticker, r.date, pm, DAY, VWF)
        if not lv:
            continue
        cols = UP_LEVELS if up else DN_LEVELS
        xcols = UP_EXO if up else DN_EXO
        name, gap = nearest_beyond(px, lv, cols, up)
        base, base_p = density_baseline(lv, cols, lv0["low"], lv0["high"], up, atr)
        xname, xgap = nearest_beyond(px, lv, xcols, up)
        xbase, xbase_p = density_baseline(lv, xcols, lv0["low"], lv0["high"], up, atr)
        dayvol = np.nanmean(b["vol"]) or np.nan
        vd = b["vol"][np.isfinite(b["vol"])]
        rows.append(dict(
            ticker=r.ticker, dir=r.dir, date=r.date, peak_roe=r.peak_roe,
            entry=r.entry, eod=r.eod, peak_mod=pm, ttp=pm - r.entry,
            near=name, gap_atr=(gap / atr) if np.isfinite(gap) else np.nan,
            base_atr=(base / atr) if np.isfinite(base) else np.nan, base_p=base_p,
            xnear=xname, xgap_atr=(xgap / atr) if np.isfinite(xgap) else np.nan,
            xbase_atr=(xbase / atr) if np.isfinite(xbase) else np.nan, xbase_p=xbase_p,
            volx=float(b["vol"][k] / dayvol) if dayvol and np.isfinite(dayvol) else np.nan,
            vol_pct=float((vd <= b["vol"][k]).mean()) if vd.size else np.nan,
            rngx=float((b["hi"][k] - b["lo"][k]) / atr),
            rsi1=float(b["rsi1"][k]) if k < b["rsi1"].size else np.nan,
            rsi5=float(b["rsi5"][k // 5]) if (k // 5) < b["rsi5"].size else np.nan,
            # how far past the nearest level BELOW/ABOVE it pushed (overshoot)
            over=_overshoot(px, lv, cols, up, atr)))
    A = pd.DataFrame(rows)
    if A.empty:
        print("  nothing"); return
    A.to_parquet("_peak_anatomy.parquet", index=False)

    print(f"\n{'='*94}")
    print(f"  PEAK ANATOMY   n={len(A):,} trades, {A['date'].nunique()} days, "
          f"{' '.join(a.tickers)}")
    print(f"{'='*94}")

    print(f"\n  A. WHEN DOES THE PREMIUM PEAK?")
    for lo, hi, lbl in ((570, 600, "09:30-10:00"), (600, 660, "10:00-11:00"),
                        (660, 720, "11:00-12:00"), (720, 780, "12:00-13:00"),
                        (780, 840, "13:00-14:00"), (840, 900, "14:00-15:00"),
                        (900, 961, "15:00-16:00")):
        g = A[(A["peak_mod"] >= lo) & (A["peak_mod"] < hi)]
        bar = "#" * int(round(len(g) / max(len(A), 1) * 60))
        print(f"    {lbl}  {len(g):>6}  {len(g)/len(A)*100:>5.1f}%  {bar}")
    print(f"    minutes from entry to peak: median {A['ttp'].median():.0f}   "
          f"p25 {A['ttp'].quantile(.25):.0f}   p75 {A['ttp'].quantile(.75):.0f}")
    # The clock histogram above is NOT clean evidence of a late-day effect: the
    # window is [entry, EOD], and the argmax of a random path concentrates at the
    # ENDS of its interval (the arcsine law). Position WITHIN the window shows how
    # much of the 15:00-16:00 mass is just that boundary.
    span = (A["eod"] - A["entry"]).replace(0, np.nan)
    pos = ((A["peak_mod"] - A["entry"]) / span).dropna()
    print(f"    position of the peak WITHIN its own window (0=entry, 1=EOD):")
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        sh = ((pos >= lo) & (pos < hi)).mean() * 100
        print(f"      {lo:.1f}-{hi:.1f}  {sh:>5.1f}%  {'#' * int(round(sh / 1.5))}")
    print(f"    -> U-shaped mass at both ends is the arcsine law, not a clock effect.")

    print(f"\n  B. WHICH LEVEL SITS NEAREST BEYOND THE PEAK?")
    vc = A["near"].value_counts()
    print(f"    {'level':24} {'times nearest':>14} {'share':>7} {'med gap ATR':>12}")
    for c, n in vc.items():
        g = A[A["near"] == c]
        print(f"    {LBL.get(c, c):24} {n:>14,} {n/len(A)*100:>6.1f}% "
              f"{g['gap_atr'].median():>12.2f}")
    miss = A["near"].isna().sum()
    print(f"    {'(none beyond the peak)':24} {miss:>14,} {miss/len(A)*100:>6.1f}%")

    print(f"\n  C. IS PROXIMITY MEANINGFUL, OR DO THE LEVELS JUST TILE THE DAY?")
    for lbl, gc, bc, pc in (("ALL levels (incl. VWAP bands)", "gap_atr", "base_atr", "base_p"),
                            ("EXOGENOUS only (no VWAP family)", "xgap_atr", "xbase_atr", "xbase_p")):
        q = A.dropna(subset=[gc, bc, pc])
        if q.empty:
            continue
        ratio = (q[gc] / q[bc]).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"    -- {lbl}  (n={len(q):,})")
        print(f"       peak gap {q[gc].median():.3f} ATR   random price "
              f"{q[bc].median():.3f} ATR   ratio med {ratio.median():.2f}")
        # peak share vs the PROBABILITY a random price is that near (not vs
        # whether the day's median was that near -- see density_baseline)
        print(f"       within 0.10 ATR of a level: peak {(q[gc]<=0.10).mean()*100:.1f}%"
              f"   random {q[pc].mean()*100:.1f}%")
    print(f"    -> ratio near 1.00 means the levels tile the range and proximity")
    print(f"       carries no information; well below 1.00 means it might. The")
    print(f"       EXOGENOUS row is the one to trust: VWAP bands are built from the")
    print(f"       same path whose extreme they are being compared against.")

    print(f"\n  D. IS THE TURN MARKED ON THE TAPE AT THE PEAK BAR?")
    print(f"     (split by direction -- at a CALL peak RSI should be HIGH and at a")
    print(f"      PUT trough LOW, so pooling them averages the signal away)")
    print(f"    {'dir':6} {'n':>7} {'volx med':>9} {'vol pctile':>11} {'rngx':>7} "
          f"{'RSI1':>7} {'>70':>6} {'<30':>6} {'RSI5':>7}")
    for dr, g in A.groupby("dir"):
        r1, r5 = g["rsi1"].dropna(), g["rsi5"].dropna()
        print(f"    {dr:6} {len(g):>7,} {g['volx'].median():>9.2f} "
              f"{g['vol_pct'].median()*100:>10.0f}% {g['rngx'].median():>7.3f} "
              f"{r1.median():>7.1f} {(r1>70).mean()*100:>5.0f}% "
              f"{(r1<30).mean()*100:>5.0f}% {r5.median():>7.1f}")

    print(f"\n  E. BY PEAK SIZE   (exo = exogenous-levels-only ratio, the honest one)")
    print(f"    {'bucket':14} {'n':>7} {'gap':>7} {'ratio':>7} {'exo gap':>8} "
          f"{'exo ratio':>10} {'volx':>7} {'volpct':>7} {'ttp':>6}")
    for lbl, lo, hi in (("<+25%", -9, .25), ("+25..100%", .25, 1.0),
                        ("+100..200%", 1.0, 2.0), (">+200%", 2.0, 99)):
        g = A[(A["peak_roe"] >= lo) & (A["peak_roe"] < hi)].dropna(subset=["gap_atr"])
        if len(g) < 30:
            continue
        rr = (g["gap_atr"] / g["base_atr"]).replace([np.inf, -np.inf], np.nan).median()
        gx = g.dropna(subset=["xgap_atr", "xbase_atr"])
        xr = (gx["xgap_atr"] / gx["xbase_atr"]).replace([np.inf, -np.inf], np.nan).median()
        print(f"    {lbl:14} {len(g):>7,} {g['gap_atr'].median():>7.3f} {rr:>7.2f} "
              f"{gx['xgap_atr'].median():>8.3f} {xr:>10.2f} "
              f"{g['volx'].median():>7.2f} {g['vol_pct'].median()*100:>6.0f}% "
              f"{g['ttp'].median():>6.0f}")

    print(f"\n  HOW TO READ IT")
    print(f"  C is the gate. If peaks are no nearer a level than a random price is,")
    print(f"  there is nothing to build on and no null is worth constructing. D says")
    print(f"  whether the turn is visible on the tape at all -- an unmarked turn cannot")
    print(f"  be detected live, whatever the level structure says.")


def _overshoot(px, lv, cols, up, atr):
    """How far the extreme pushed PAST the nearest level it cleared, in ATR."""
    best = np.nan
    for c in cols:
        v = lv.get(c)
        if v is None or not np.isfinite(v):
            continue
        gap = (px - v) if up else (v - px)
        if gap >= 0 and (np.isnan(best) or gap < best):
            best = gap
    return float(best / atr) if np.isfinite(best) else np.nan


if __name__ == "__main__":
    main()
