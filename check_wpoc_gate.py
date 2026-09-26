# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_wpoc_gate.py
==================
🚨 A5 HERE IS WITHDRAWN (inherited from `check_mbo_flow_interaction`) -- the
   `gap - base2m` statistic is degenerate at a balanced direction mix and its
   gate is unpassable. See METHODOLOGY.md 6c. `check_wpoc_gating.py` re-runs the
   question on a statistic that can move; A1/A2/A4/A6, `build_panel` and
   `thin()` below are sound and are imported by it.
   THE VERDICT IS UNCHANGED: the corrected test returns |gain| < 0.6bp with a
   5m CI of [-0.20, +0.05]bp on n=81,727 non-overlapping triggers. wPOC is a
   genuinely POWERED null, not merely an asserted one.

Does the 5-day rolling WEEKLY POINT OF CONTROL condition our flow trigger?

WHY THIS AND NOT THE REST OF THE AMT PROPOSAL
----------------------------------------------
`check_amt.py` and `check_amt_variants.py` already tested prior-day POC/value
area (0 of 48 cells passed), `check_volume_signal.py` tested VWAP distance
(null), and `check_magnet.py` / `check_gamma_walls.py` tested GEX walls as
magnets (null). The ONE genuinely untested piece is the LONGER-MEMORY level: a
5-day rolling POC rather than yesterday's value area.

So this tests that single component before anything is built on top of it. A
multi-condition "confluence gate" has four tunable knobs (tolerance band, window
length, which GEX wall, three sizing regimes) against ~300 sequential trades --
the exact ratio that produced this project's false positives.

TWO FEATURES, because a structural LEVEL is not a directional METRIC
--------------------------------------------------------------------
  F1 REGIME   sign(spot - wPOC). Aligned = flow direction matches the structural
              regime (CALLs above wPOC = "supported"; CALLs below = "blocked").
              Largely DAY-CONSTANT -- the honest unit is the DAY, and effective
              n is days, not triggers.
  F2 PROXIMITY |spot - wPOC|, NORMALISED PER TICKER (raw bp scales with realised
              vol and with time since the node formed, so raw distance is a
              volatility proxy). Varies within-day, so trade-level is correct.
              **Direction PRE-COMMITTED: NEAR is better** -- that is what the
              trapped-liquidity mechanism predicts. Leaving it two-sided would
              be two bites at one apple.

=========================  PRE-COMMITTED CRITERIA  =========================
  A1  "supported" beats "blocked" on direction-adjusted drift, 5/15/30/60m
  A2  IS and OOS agree in SIGN
  A3  replicates across >= 6 of the 9 book tickers, same sign. wPOC needs NO
      purchased data -- 513 days x 9 tickers -- so this is the REAL
      cross-sectional test, not the softened one the MBO sample forced.
  A3-alt  temporal thirds of the full window, effect present in each
  A4  beats p95 of a WITHIN-DAY BLOCK PERMUTATION (clustering is severe: wPOC
      is constant within a day by construction)
  A5  exceeds `base2m` = 2 x E[sign(F1) x r], the gap a WORTHLESS trigger would
      produce from F1's marginal effect alone.  **F1 ONLY** -- F2 is unsigned,
      so the derivation does not apply and A4 carries the weight there.
  A6  exceeds the SAME effect computed from the EXISTING trend regime.
      sign(spot - wPOC) is heavily autocorrelated and close to a trend proxy;
      `load_trend_regime` is already a deployed gate. If wPOC cannot beat the
      thing we already have, it is a rename, not a discovery.

NON-OVERLAPPING WINDOWS: triggers closer together than the horizon share most of
their forward window. Overlap turned a 5bp effect into 52bp in the MBO work
earlier today, so triggers are THINNED to >= h minutes apart within each day.

🚨 CAUSALITY FIX vs THE PROPOSAL'S CODE: the reference implementation used
`daily_profile.rolling(5).sum()`, which INCLUDES THE CURRENT DAY. Today's volume
profile is not knowable at 10:00. This shifts by one day so the wPOC in force on
day D is built from D-5..D-1 only.

Usage:  python check_wpoc_gate.py
        python check_wpoc_gate.py --tickers IWM QQQ --boot 4000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
HORIZONS = [5, 15, 30, 60]
BIN_PCT = 0.0005
WINDOW = 5


def _bars(tk):
    import polars as pl
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p, columns=["start_time", "high", "low", "close", "volume"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= RTH_LO) & (d["mod"] <= RTH_HI)].copy()
    for c in ("high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["close"]).sort_values(["date", "mod"]).reset_index(drop=True)


def wpoc_map(d):
    """{date: wPOC price}, built from the PRIOR `WINDOW` sessions only.

    Binning-matrix method: bin the typical price, pivot to a (date x bin) volume
    matrix, rolling-sum the matrix, take the argmax bin per row. Efficient, and
    it never materialises a tick-level structure.
    """
    binw = round(float(d["close"].median()) * BIN_PCT, 4) or 0.01
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    b = np.floor(tp / binw).astype(np.int64)
    prof = (pd.DataFrame({"date": d["date"], "b": b, "v": d["volume"]})
            .groupby(["date", "b"], observed=True)["v"].sum().unstack(fill_value=0.0))
    # SHIFT(1): the level in force today is built from prior sessions ONLY
    roll = prof.rolling(WINDOW, min_periods=WINDOW).sum().shift(1)
    # the first WINDOW sessions are all-NaN after the causal shift, and idxmax
    # raises on an all-NaN row -- drop them rather than let it throw
    roll = roll.dropna(how="all")
    if roll.empty:
        return {}
    idx = roll.idxmax(axis=1)
    out = {}
    for dt, bn in idx.items():
        if pd.notnull(bn):
            out[dt] = (float(bn) + 0.5) * binw
    return out


def build_panel(D, tk):
    from check_config_walkforward import _flow_for
    from config import RULES
    d = _bars(tk)
    if d is None or d.empty:
        return None
    wp = wpoc_map(d)
    if not wp:
        return None

    px = {}
    for dt, g in d.groupby("date"):
        px[dt] = (g["mod"].to_numpy(int), g["close"].to_numpy(float))

    rule = next((r for r in RULES if r["ticker"] == tk and r.get("enabled", True)), None)
    flow = _flow_for(D, [tk])
    if flow.empty:
        return None
    trigs = D.triggers_for(flow, tk)
    D.annotate_flow_pct(trigs, (rule or {}).get("flow_window_days", 60))
    trd = D.load_trend_regime(HIST, tk)

    rows = []
    for t in trigs:
        thr = t.get("thr")
        if not thr or t["abs_flow"] < thr.get(50, 1e99):
            continue
        dt = t["date"]
        if dt not in px or dt not in wp:
            continue
        ts = pd.Timestamp(t["ts"])
        m = ts.hour * 60 + ts.minute
        mods, cl = px[dt]
        i = int(np.searchsorted(mods, m, side="right")) - 1
        if i < 0:
            continue
        spot = float(cl[i])
        w = wp[dt]
        if not np.isfinite(spot) or spot <= 0 or not np.isfinite(w) or w <= 0:
            continue
        r = dict(ticker=tk, date=dt, mod=int(mods[i]),
                 dir=1.0 if t["dir"] == "CALL" else -1.0,
                 spot=spot, wpoc=w,
                 f1=1.0 if spot > w else -1.0,
                 dist=abs(spot / w - 1.0),
                 trend=trd.get(dt))
        for h in HORIZONS:
            j = int(np.searchsorted(mods, mods[i] + h, side="right")) - 1
            r[f"fwd{h}"] = (float(cl[j]) / spot - 1.0) if (j > i) else np.nan
        rows.append(r)
    if not rows:
        return None
    P = pd.DataFrame(rows)
    # normalise distance WITHIN ticker -- raw bp is a volatility proxy
    P["dist_r"] = P["dist"].rank(pct=True)
    return P


def thin(P, h):
    """Keep triggers >= h minutes apart within a day -> non-overlapping windows."""
    keep = []
    for (_tk, _dt), g in P.groupby(["ticker", "date"], sort=False):
        last = -10 ** 9
        for i, m in zip(g.index, g["mod"]):
            if m - last >= h:
                keep.append(i)
                last = m
    return P.loc[keep]


def gap_base(sub, col, sgn_col="f1"):
    """(aligned-minus-opposed gap, no-interaction baseline 2m)."""
    s = sub[sgn_col].to_numpy(float)
    dd = sub["dir"].to_numpy(float)
    r = sub[col].to_numpy(float)
    m = np.isfinite(r) & np.isfinite(s) & (s != 0)
    if m.sum() < 60:
        return np.nan, np.nan, 0, 0
    s, dd, r = s[m], dd[m], r[m]
    al = (s == dd)
    if al.sum() < 20 or (~al).sum() < 20:
        return np.nan, np.nan, int(al.sum()), int((~al).sum())
    g = float(np.mean(dd[al] * r[al]) - np.mean(dd[~al] * r[~al]))
    return g, 2.0 * float(np.mean(s * r)), int(al.sum()), int((~al).sum())


def block_p(sub, col, boot, rng, sgn_col="f1"):
    obs, _, _, _ = gap_base(sub, col, sgn_col)
    if not np.isfinite(obs):
        return np.nan
    s = sub[sgn_col].to_numpy(float); dd = sub["dir"].to_numpy(float)
    r = sub[col].to_numpy(float); key = (sub["ticker"] + "|" + sub["date"].astype(str)).to_numpy()
    m = np.isfinite(r) & np.isfinite(s) & (s != 0)
    s, dd, r, key = s[m], dd[m], r[m], key[m]
    blocks = {}
    for i, k in enumerate(key):
        blocks.setdefault(k, []).append(i)
    blocks = [np.array(v) for v in blocks.values()]
    rr = r.copy()
    hits = 0
    for _ in range(boot):
        for b in blocks:
            rr[b] = r[rng.permutation(b)]
        al = (s == dd)
        if al.sum() < 5 or (~al).sum() < 5:
            continue
        g = np.mean(dd[al] * rr[al]) - np.mean(dd[~al] * rr[~al])
        if abs(g) >= abs(obs):
            hits += 1
    return hits / boot


def run(a):
    import directional_flow_backtester as D
    from config import RULES
    rng = np.random.default_rng(31)
    tickers = a.tickers or sorted({r["ticker"] for r in RULES if r.get("enabled", True)})

    parts = []
    for tk in tickers:
        P = build_panel(D, tk)
        if P is None or P.empty:
            print(f"  ! {tk}: no panel"); continue
        parts.append(P)
        print(f"  {tk:5} {len(P):>6} triggers  {P['date'].nunique():>4} days  "
              f"wPOC dist median {P['dist'].median()*1e4:>6.0f}bp  "
              f"above-wPOC {(P['f1']>0).mean()*100:>4.0f}%")
    if not parts:
        return
    A = pd.concat(parts, ignore_index=True)

    print("\n" + "=" * 112)
    print("  F1  REGIME: sign(spot - wPOC) vs flow direction")
    print("  NON-OVERLAPPING triggers only (>= h minutes apart within a day)")
    print("=" * 112)
    print(f"  {'h':>4} {'n':>6} {'days':>5} {'gap(bp)':>9} {'base2m':>9} {'trend(A6)':>10} "
          f"{'IS':>8} {'OOS':>8} {'blockp':>7}  A1 A2 A4 A5 A6")
    live = []
    for h in HORIZONS:
        col = f"fwd{h}"
        S = thin(A, h)
        S = S[np.isfinite(S[col])]
        if len(S) < 200:
            continue
        g, base, nal, nop = gap_base(S, col)
        if not np.isfinite(g):
            continue
        gi, _, _, _ = gap_base(S[S.date < SPLIT], col)
        go, _, _, _ = gap_base(S[S.date >= SPLIT], col)
        # A6: same construction on the EXISTING trend regime
        T = S.copy()
        T["tsign"] = T["trend"].map({"UPTREND": 1.0, "DOWNTREND": -1.0}).fillna(0.0)
        gt, _, _, _ = gap_base(T[T.tsign != 0], col, sgn_col="tsign")
        pv = block_p(S, col, a.boot, rng)
        a1 = g > 0
        a2 = np.isfinite(gi) and np.isfinite(go) and np.sign(gi) == np.sign(go)
        a4 = np.isfinite(pv) and pv < 0.05
        a5 = g > base * 1.25 if base > 0 else g > 0
        a6 = np.isfinite(gt) and g > abs(gt) * 1.25
        if a1 and a2:
            live.append(h)
        print(f"  {h:>3}m {len(S):>6} {S['date'].nunique():>5} {g*1e4:>+9.2f} "
              f"{base*1e4:>+9.2f} {gt*1e4 if np.isfinite(gt) else np.nan:>+10.2f} "
              f"{gi*1e4:>+8.2f} {go*1e4:>+8.2f} {pv:>7.3f}  "
              f"{'Y' if a1 else '.'}  {'Y' if a2 else '.'}  {'Y' if a4 else '.'}  "
              f"{'Y' if a5 else '.'}  {'Y' if a6 else '.'}")

    print("\n" + "=" * 112)
    print("  A3  CROSS-SECTIONAL — per ticker (needs >= 6 of 9 with the same sign)")
    print("=" * 112)
    for h in (live or HORIZONS[:2]):
        col = f"fwd{h}"
        S = thin(A, h)
        cells, signs = [], []
        for tk in tickers:
            sub = S[(S.ticker == tk) & np.isfinite(S[col])]
            g, _, _, _ = gap_base(sub, col)
            cells.append(f"{tk}:{g*1e4:>+7.1f}" if np.isfinite(g) else f"{tk}:   n/a")
            if np.isfinite(g):
                signs.append(np.sign(g))
        pos = sum(1 for s in signs if s > 0)
        print(f"  {h:>3}m  " + "  ".join(cells))
        print(f"        positive in {pos}/{len(signs)}   "
              f"A3 {'Y' if pos >= 6 else '.'}")

    print("\n" + "=" * 112)
    print("  F2  PROXIMITY — direction PRE-COMMITTED: NEAR beats FAR")
    print("  distance ranked within ticker (raw bp is a volatility proxy)")
    print("=" * 112)
    print(f"  {'h':>4} {'near(bp)':>10} {'far(bp)':>10} {'near-far':>10} "
          f"{'IS':>9} {'OOS':>9}")
    for h in HORIZONS:
        col = f"fwd{h}"
        S = thin(A, h)
        S = S[np.isfinite(S[col])]
        if len(S) < 200:
            continue
        adj = (S["dir"] * S[col]).to_numpy(float)
        near = adj[S["dist_r"] <= 0.25]
        far = adj[S["dist_r"] >= 0.75]
        if len(near) < 40 or len(far) < 40:
            continue
        isk = (S.date < SPLIT).to_numpy()
        ni, fi = adj[isk & (S["dist_r"] <= 0.25)], adj[isk & (S["dist_r"] >= 0.75)]
        no_, fo = adj[~isk & (S["dist_r"] <= 0.25)], adj[~isk & (S["dist_r"] >= 0.75)]
        print(f"  {h:>3}m {np.mean(near)*1e4:>+10.2f} {np.mean(far)*1e4:>+10.2f} "
              f"{(np.mean(near)-np.mean(far))*1e4:>+10.2f} "
              f"{(np.mean(ni)-np.mean(fi))*1e4 if len(ni)>10 and len(fi)>10 else np.nan:>+9.2f} "
              f"{(np.mean(no_)-np.mean(fo))*1e4 if len(no_)>10 and len(fo)>10 else np.nan:>+9.2f}")

    print("\n  NOTE: F2 must also clear the distance-matched range baseline in")
    print("  check_amt._range_dist before any proximity claim stands -- 'price")
    print("  reaches a level more than you'd think' is true of ANY level until")
    print("  you control for how far price normally travels.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    run(ap.parse_args())
