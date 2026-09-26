# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_wall_break.py
===================
DOES NON-FADING VOLUME ACROSS REPEATED WALL TESTS PREDICT THE BREAK?

THE HYPOTHESIS (live IWM read, 2026-09-22)
    Price tests a call wall, it holds, volume does NOT fade across repeated
    tests, and then it breaks and squeezes. Mechanism: below the wall dealers
    are long gamma and suppress; above it the hedge flips and they buy strength.

WHAT check_wall_break_count ALREADY ESTABLISHED
    701 events with >=2 tests, 347 breaks, ~118 per ticker-year -- power is fine.
    Break rate rises 33% (1 test) -> ~50% (2+) and PLATEAUS.
    Negative gamma breaks 57.7% vs positive 43.7%, +14pp in the direction the
    mechanism predicts. That was the falsifier and it did not fire.

🚨 WHAT THAT STUDY GOT WRONG, FIXED HERE
    Its extension comparison was CIRCULAR: "broke" is defined as >=25bp
    sustained, so the held group is capped below 25bp by construction and the
    65bp-vs-17bp gap was definitional. Forward excursion is therefore measured
    over a FIXED WINDOW after the last test, independent of whether a break was
    declared.

🚨 THE PLACEBO, WHICH IS THE WHOLE ARGUMENT
    Walls-as-magnets is already busted (check_magnet, check_gamma_walls), POC
    went 0/48 cells, VWAP distance was null twice. So "price does something at a
    level" is not news. The claim only survives if the REAL wall beats a level
    at the SAME DISTANCE with no gamma behind it. The sham level borrows another
    ticker's relative wall distance on the same session -- identical geometry,
    no IWM gamma there. If the sham behaves the same, this is about levels, not
    about dealers, and METHODOLOGY 7's "stop assuming dealer hedging" stands.

🚨 AND A MAGNITUDE FLOOR, BECAUSE SIGN TESTS ARE NOT CRITERIA
    check_flow_spike passed four of five pre-committed criteria on +0.34bp.
    Every criterion below is in percentage points or basis points with a floor.

PRE-COMMITTED CRITERIA
    B1  non-fading volume raises the break rate by >= 10pp vs fading
    B2  and raises forward MFE by >= 15bp on the NON-CIRCULAR measure
    B3  the real wall beats the distance-matched sham on B1
    B4  IS and OOS agree in sign
    B5  the effect is larger in NEGATIVE gamma (mechanism consistency)

Usage:
  python check_wall_break.py
  python check_wall_break.py --fwd 60 --min-ext 25
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

RTH0, RTH1 = 570, 955
SPLIT = pd.Timestamp("2025-08-21").date()
RVOL_DAYS = 20


def load_bars(tk):
    df = (pl.scan_parquet(f"historical/{tk}.parquet")
          .select("date", "minute_et", "high", "low", "close", "volume")
          .collect().to_pandas())
    t = pd.to_datetime(df["minute_et"])
    df["mod"] = t.dt.hour * 60 + t.dt.minute
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["mod"] >= RTH0) & (df["mod"] <= RTH1)]
    return {d: g.sort_values("mod").set_index("mod") for d, g in df.groupby("date")}


def rvol_baselines(B):
    """{date: {mod: median volume over the prior RVOL_DAYS sessions}}.

    TIME-OF-DAY and TRAILING. A flat average would be wrong -- the intraday
    smile runs ~12x from open to midday -- and a full-sample average would let
    the session being scored into its own baseline.
    """
    days = sorted(B)
    out = {}
    for i, d in enumerate(days):
        prior = days[max(0, i - RVOL_DAYS):i]
        if len(prior) < 10:
            continue
        acc = {}
        for p in prior:
            g = B[p]
            for m, v in zip(g.index, g["volume"].to_numpy()):
                acc.setdefault(int(m), []).append(float(v))
        out[d] = {m: float(np.median(v)) for m, v in acc.items() if v}
    return out


def tests_of(g, lvl, side, tol_bp, gap):
    band = lvl * tol_bp / 1e4
    if side == "call":
        hit = g[(g["high"] >= lvl - band) & (g["close"] < lvl)]
    else:
        hit = g[(g["low"] <= lvl + band) & (g["close"] > lvl)]
    if hit.empty:
        return []
    out, last = [], -999
    for m in sorted(int(x) for x in hit.index):
        if m - last > gap:
            out.append(m)
        last = m
    return out


def broke(g, lvl, side, after, ext_bp, hold):
    ext = lvl * ext_bp / 1e4
    mods = np.array([int(x) for x in g.index])
    cl = g["close"].to_numpy()
    beyond = cl > lvl + ext if side == "call" else cl < lvl - ext
    run = 0
    for i in range(len(mods)):
        if mods[i] <= after:
            run = 0
            continue
        run = run + 1 if beyond[i] else 0
        if run >= hold:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--min-ext", type=float, default=25.0)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--fwd", type=int, default=60,
                    help="minutes after the LAST test for the non-circular MFE")
    a = ap.parse_args()

    fs = sorted(glob.glob("_level_cache/walls_*dte0-3.parquet"),
                key=os.path.getsize, reverse=True)
    W = pl.scan_parquet(fs[0]).collect().to_pandas()
    W["date"] = pd.to_datetime(W["date"]).dt.date
    tks = sorted(W["ticker"].unique())
    B = {tk: load_bars(tk) for tk in tks}
    RV = {tk: rvol_baselines(B[tk]) for tk in tks}
    print(f"  walls {fs[0]}\n  tickers {tks}\n")

    # relative wall distances per (date, ticker), for the distance-matched sham
    dist = {}
    for _i, r in W.iterrows():
        for side, lvl in (("call", r["cw"]), ("put", r["pw"])):
            if np.isfinite(lvl) and np.isfinite(r["spot"]) and r["spot"] > 0:
                dist[(r["date"], r["ticker"], side)] = (lvl - r["spot"]) / r["spot"]

    rows = []
    for _i, r in W.iterrows():
        tk, d = r["ticker"], r["date"]
        g = B[tk].get(d)
        base = RV[tk].get(d)
        if g is None or base is None or not np.isfinite(r.get("spot", np.nan)):
            continue
        for side in ("call", "put"):
            real = r["cw"] if side == "call" else r["pw"]
            if not np.isfinite(real) or real <= 0:
                continue
            # SHAM: another ticker's relative distance, same session & side
            sham = None
            for o in tks:
                if o == tk:
                    continue
                dd = dist.get((d, o, side))
                if dd is not None:
                    sham = r["spot"] * (1 + dd)
                    break
            for kind, lvl in (("wall", real), ("sham", sham)):
                if lvl is None or not np.isfinite(lvl) or lvl <= 0:
                    continue
                ts = tests_of(g, lvl, side, a.tol, a.gap)
                if len(ts) < 2:
                    continue
                # RVOL at each test minute
                rv = []
                for m in ts:
                    ref = base.get(int(m))
                    if ref and int(m) in g.index:
                        rv.append(float(g.loc[int(m), "volume"]) / ref)
                if len(rv) < 2:
                    continue
                # "did volume fade across the tests?" -- last vs first
                fade = rv[-1] / rv[0] if rv[0] > 0 else np.nan
                last = ts[-1]
                fwd = g[(g.index > last) & (g.index <= last + a.fwd)]
                if fwd.empty:
                    continue
                mfe = ((fwd["high"].max() - lvl) / lvl * 1e4 if side == "call"
                       else (lvl - fwd["low"].min()) / lvl * 1e4)
                rows.append(dict(
                    ticker=tk, date=d, side=side, kind=kind, n_tests=len(ts),
                    gamma_sign=r["gamma_sign"], rv_first=rv[0], rv_last=rv[-1],
                    rv_mean=float(np.mean(rv)), fade=fade,
                    broke=broke(g, lvl, side, last, a.min_ext, a.hold),
                    mfe_bp=mfe, half=("IS" if d <= SPLIT else "OOS")))
        # progress is per-ticker below

    R = pd.DataFrame(rows)
    if R.empty:
        print("  no events"); return
    R["nonfade"] = R["fade"] >= 1.0
    R.to_parquet("_wall_break.parquet", index=False)

    def block(title, sel):
        print(f"\n{'='*96}")
        print(f"  {title}")
        print(f"{'='*96}")
        print(f"  {'group':26} {'n':>6} {'break rate':>12} {'MFE p50':>9} "
              f"{'MFE p75':>9}")
        for lab, gg in sel:
            if not len(gg):
                continue
            print(f"  {lab:26} {len(gg):>6,} {gg['broke'].mean()*100:>11.1f}% "
                  f"{gg['mfe_bp'].median():>9.0f} "
                  f"{gg['mfe_bp'].quantile(.75):>9.0f}")

    w = R[R["kind"] == "wall"]
    s = R[R["kind"] == "sham"]
    block(f"1. B1/B2 -- VOLUME ACROSS THE TESTS (MFE over +{a.fwd}m, "
          f"NOT conditional on the break)",
          [("wall, volume HELD", w[w["nonfade"]]),
           ("wall, volume FADED", w[~w["nonfade"]]),
           ("wall, all", w)])

    block("2. B3 -- THE SAME CUT AT A DISTANCE-MATCHED SHAM LEVEL",
          [("sham, volume HELD", s[s["nonfade"]]),
           ("sham, volume FADED", s[~s["nonfade"]]),
           ("sham, all", s)])

    block("3. B5 -- BY GAMMA SIGN (wall only, volume held)",
          [(f"gamma {k}", gg) for k, gg in
           w[w["nonfade"]].groupby("gamma_sign")])

    block("4. B4 -- IS / OOS (wall only)",
          [(f"{h} volume {'HELD' if nf else 'FADED'}",
            w[(w["half"] == h) & (w["nonfade"] == nf)])
           for h in ("IS", "OOS") for nf in (True, False)])

    wh, wf = w[w["nonfade"]], w[~w["nonfade"]]
    sh, sf = s[s["nonfade"]], s[~s["nonfade"]]
    d_break = (wh["broke"].mean() - wf["broke"].mean()) * 100
    d_mfe = wh["mfe_bp"].median() - wf["mfe_bp"].median()
    d_sham = ((sh["broke"].mean() - sf["broke"].mean()) * 100
              if len(sh) and len(sf) else np.nan)
    print(f"\n{'='*96}")
    print(f"  SCORECARD  (pre-committed, with floors)")
    print(f"{'='*96}")
    m = lambda ok: "PASS" if ok else "FAIL"
    print(f"  B1  break rate +>=10pp when volume holds   {d_break:>+7.1f}pp  "
          f"{m(d_break >= 10)}")
    print(f"  B2  forward MFE +>=15bp                    {d_mfe:>+7.0f}bp  "
          f"{m(d_mfe >= 15)}")
    print(f"  B3  wall effect > sham effect              "
          f"{d_break:>+7.1f} vs {d_sham:>+6.1f}  {m(d_break > d_sham)}")
    for h in ("IS", "OOS"):
        hh, ff = w[(w['half'] == h) & w['nonfade']], w[(w['half'] == h) & ~w['nonfade']]
        if len(hh) and len(ff):
            print(f"  B4  {h:<3} break-rate delta                    "
                  f"{(hh['broke'].mean()-ff['broke'].mean())*100:>+7.1f}pp")
    print(f"  B5  see block 3 -- negative gamma should lead")


if __name__ == "__main__":
    main()
