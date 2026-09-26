# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_peak_levels.py
====================
DO TRADE PEAKS LAND ON STRUCTURE? -- exhaustion / resistance at the MFE.

THE QUESTION
    check_peak_decay showed the peak is essentially unreachable by a causal
    trail. But if peaks CLUSTER at a level knowable in advance -- a gamma wall,
    a VWAP band, the weekly POC -- an exit would not need to DETECT a turn. It
    could rest an order at the level. That is a different mechanism from every
    exit tested here so far, all of which react to price rather than anticipate.

WHY THIS IS NOT THE FOUR NULLS ALREADY ON THE BOARD
    check_amt / check_amt_variants (prior-day POC, 0/48 cells), check_volume_signal
    (VWAP distance), check_magnet + check_gamma_walls (walls as magnets) and
    check_wpoc_gate (5-day POC, |gain| < 0.6bp on n=81,727) tested these levels
    as ENTRY conditioners: does the level predict forward drift? This asks where
    a move STOPS -- a different quantity. A level can be useless for "will price
    move" and still mark "where it runs out". Adjacent, not a re-run; but those
    nulls are the prior, and a weak positive here reads against them.

============  THREE CONFOUNDS, EACH FOUND BY A FAILING PLACEBO  =============
This test was built three times. Versions 1 and 2 produced clean, fully
significant tables that were entirely artifact. What follows is what the
placebos caught, in order:

(1) GEOMETRY. A CALL's peak value is essentially the underlying's running MAX,
    and the call wall sits ABOVE spot by construction, so the peak is
    mechanically closer to the wall than a random MINUTE is. -> the null must
    be a matched EXTREME, not a matched minute.

(2) MEASUREMENT ASYMMETRY. v1 scored the real trade at the price of its option
    peak MINUTE but scored the null as a window MAXIMUM. The option peak is the
    window max only 36% of the time, so a maximum was being compared with a
    non-maximum. -> score the real window's extreme, exactly as the null does.

(3) SELECTION. The real window always ENDS at the EOD flatten, so it starts as
    LATE as its duration allows, and price diffuses away from open-anchored
    levels through the session; worse, the flow trigger selects days with LARGER
    excursions, and a bigger move lands farther from any fixed price. Both shift
    every level and every placebo together -- which is precisely what was seen:
    v2 returned 0.21-0.34 across the board, placebos included.
    -> the null must match TIME OF DAY, OPPORTUNITY (how far the level sat ahead
       of entry) and DAY SIZE (realised range / ATR), leaving only "did the move
       terminate AT the level" free to vary.

THE TEST
    For each trade and each level L on the side that could stop it:
        d0 = (L - entry_px) * sgn / ATR     how far ahead the level sat
        s  = (extreme - entry_px) * sgn / ATR   how far the move actually ran
        |s - d0| = |extreme - L| / ATR      how close the move stopped to it
    Controls are the SAME ticker and SAME clock window on OTHER days, keeping
    only those whose own d0 is within --tol-d0 and whose own day size is within
    --tol-rng, and skipping days whose own trigger fired within --excl minutes.
    pct = share of controls stopping FARTHER from their level than the real
    trade did. 0.5 is chance.

    Day size and d0 are matched using the control day's REALISED range, which is
    look-ahead. That is legitimate for a descriptive question ("given a day like
    this and a level this far ahead, does the move stop there?") and is NOT a
    tradeable filter. It is applied identically to placebos, which is what makes
    the placebo reading meaningful.

PLACEBOS ARE THE FIRST THING TO READ
    plc_u / plc_up / plc_dn carry no structural meaning and MUST return 0.50.
    They are the only reason the three biases above were caught rather than
    published. If they miss 0.50, nothing else in the table is interpretable.
    (METHODOLOGY 7: pre-flight every control.)

Usage:
  python check_peak_levels.py --tickers SPY QQQ IWM
  python check_peak_levels.py --tickers SPY QQQ IWM --tol-d0 0.20 --no-match-rng
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

CACHE = "_level_cache"
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
NMIN = RTH_HI - RTH_LO + 1

LEVELS = [
    ("cw",   "up",   "call wall"),
    ("pw",   "dn",   "put wall"),
    ("wpoc", "both", "wPOC (5d, prior)"),
    ("vwap", "both", "session VWAP"),
    ("vwu1", "up",   "VWAP +1sd"),
    ("vwu2", "up",   "VWAP +2sd"),
    ("vwd1", "dn",   "VWAP -1sd"),
    ("vwd2", "dn",   "VWAP -2sd"),
    ("pdh",  "up",   "prior-day high"),
    ("pdl",  "dn",   "prior-day low"),
    ("pvah", "up",   "prior value-area hi"),
    ("pval", "dn",   "prior value-area lo"),
    ("ib_hi", "up",  "initial-balance hi"),
    ("ib_lo", "dn",  "initial-balance lo"),
    ("plc_u",  "both", "PLACEBO uniform-in-range"),
    ("plc_up", "up",   "PLACEBO open+0.8 ATR"),
    ("plc_dn", "dn",   "PLACEBO open-0.8 ATR"),
]
STATIC = ("cw", "pw", "wpoc", "pdh", "pdl", "pvah", "pval", "ib_hi", "ib_lo",
          "plc_u", "plc_up", "plc_dn")
VWSPEC = {"vwap": 0.0, "vwu1": 1.0, "vwu2": 2.0, "vwd1": -1.0, "vwd2": -2.0}


# --------------------------------------------------------------- trade peaks
def peak_rows(tickers, dirs, pct, verbose=True):
    """Cached: rebuilding these re-reads the 152M-row option bar cache once per
    ticker (~15 min), and every study of the peak wants the same table."""
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"peaks_{'-'.join(sorted(tickers))}_"
                             f"{'-'.join(sorted(dirs))}_p{pct}.parquet")
    if os.path.exists(fp):
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if verbose:
            print(f"    peaks cached: {len(df):,} trades -> {fp}")
        return df

    import sim_core
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    out = []
    for tk in tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        for direction in dirs:
            rule = {"name": f"{tk} {direction}", "ticker": tk, "direction": direction,
                    "dte": [0, 1], "min_flow_pct": pct, "target_roe": 1.0, "rr": 1.0}
            cand = sim_core.build_candidates(D, rule, trigs=trigs, since=None)
            eod_m = sim_core.eod_mod(rule)
            for d, m, path in cand:
                e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
                if e_mid <= 0:
                    continue
                n = int(np.searchsorted(mods, eod_m, side="right"))
                if n < 5:
                    continue
                b = np.asarray(bid[:n], float)
                if b.max() <= 0:
                    continue
                i = int(np.argmax(b))
                out.append(dict(ticker=tk, dir=direction, date=d, entry=int(m),
                                eod=int(mods[n - 1]),
                                peak_roe=float(b[i] / e_mid - 1.0)))
        if verbose:
            print(f"    {tk} peaks done", flush=True)
    df = pd.DataFrame(out)
    if not df.empty:
        df.to_parquet(fp, index=False)
    return df


# ------------------------------------------------------------------ the cube
class Cube:
    """Everything about one ticker on a (date x minute) grid, so a trade's
    controls across ~600 days are a vectorised gather instead of a loop.

    The window is always [entry, EOD] -- `eod` is fixed by the rule -- so a
    SUFFIX extreme from each minute answers every trade at once.
    """

    def __init__(self, tk, und, DAY, VWF):
        dates = sorted(d for d in und if d in DAY)
        self.tk, self.dates = tk, np.array(dates, dtype=object)
        self.didx = {d: i for i, d in enumerate(dates)}
        nD = len(dates)
        nan = np.full((nD, NMIN), np.nan)
        self.P = nan.copy()                      # price at minute m
        self.EU, self.EUm = nan.copy(), np.zeros((nD, NMIN), np.int32)
        self.ED, self.EDm = nan.copy(), np.zeros((nD, NMIN), np.int32)
        self.VW, self.VS = nan.copy(), nan.copy()
        for i, d in enumerate(dates):
            mods, hi, lo, cl = und[d]
            k = np.clip(mods - RTH_LO, 0, NMIN - 1)
            row = np.full(NMIN, np.nan)
            row[k] = cl
            self.P[i] = _ffill(row)
            # suffix max / min and the minute each occurs at
            h = np.full(NMIN, -np.inf); h[k] = hi
            l = np.full(NMIN, np.inf);  l[k] = lo
            self.EU[i], self.EUm[i] = _suffix(h, True)
            self.ED[i], self.EDm[i] = _suffix(l, False)
            vm = VWF.get(d)
            if vm is not None:
                mm, vw, sd = vm
                kk = np.clip(mm - RTH_LO, 0, NMIN - 1)
                rv = np.full(NMIN, np.nan); rv[kk] = vw
                rs = np.full(NMIN, np.nan); rs[kk] = sd
                self.VW[i], self.VS[i] = _ffill(rv), _ffill(rs)
        self.stat = {c: np.array([DAY[d].get(c, np.nan) for d in dates], float)
                     for c in STATIC}
        self.open = np.array([DAY[d]["open"] for d in dates], float)
        self.atr = np.array([DAY[d]["atr"] for d in dates], float)
        self.rng = np.array([(DAY[d]["high"] - DAY[d]["low"]) for d in dates], float)
        self.rr = self.rng / np.where(self.atr > 0, self.atr, np.nan)
        self.trig = np.zeros((nD, NMIN), bool)

    def mark_triggers(self, df, excl):
        for r in df.itertuples():
            i = self.didx.get(r.date)
            if i is None:
                continue
            a = max(0, r.entry - RTH_LO - excl)
            b = min(NMIN, r.entry - RTH_LO + excl + 1)
            self.trig[i, a:b] = True

    def vw_at(self, rows, mins, k):
        """VWAP band k at (row, minute) pairs."""
        c = np.clip(mins - RTH_LO, 0, NMIN - 1)
        return self.VW[rows, c] + k * self.VS[rows, c]


def _ffill(a):
    idx = np.where(np.isfinite(a), np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    return np.where(np.isfinite(out), out, np.nan)


def _suffix(v, want_max):
    """Suffix extreme and the index it occurs at, for every start position."""
    n = v.size
    best = np.empty(n)
    at = np.empty(n, np.int32)
    cur, ci = (-np.inf, n - 1) if want_max else (np.inf, n - 1)
    for i in range(n - 1, -1, -1):
        x = v[i]
        if (want_max and x >= cur) or (not want_max and x <= cur):
            cur, ci = x, i
        best[i], at[i] = cur, ci
    return np.where(np.isfinite(best), best, np.nan), at.astype(np.int32) + RTH_LO


# ------------------------------------------------------------------- levels
def load_levels(tickers):
    wf = [f for f in os.listdir(CACHE) if f.startswith("walls_")]
    if not wf:
        raise SystemExit("  no wall tape -- run build_level_tape.py first")
    best = walls = pick = None
    for f in wf:                      # WIDEST tape, not last alphabetically
        w = pd.read_parquet(os.path.join(CACHE, f))
        if best is None or len(w) > best:
            best, walls, pick = len(w), w, f
    print(f"  wall tape: {pick} ({best} ticker-days)")
    walls["date"] = pd.to_datetime(walls["date"]).dt.date
    wmap = {(r.ticker, r.date): (float(getattr(r, "cw", np.nan)),
                                 float(getattr(r, "pw", np.nan)))
            for r in walls.itertuples()}

    DAY, VWF = {}, {}
    for tk in tickers:
        fp = os.path.join(CACHE, f"und_{tk}.parquet")
        vp = os.path.join(CACHE, f"vwap_{tk}.parquet")
        if not (os.path.exists(fp) and os.path.exists(vp)):
            continue
        dd = pd.read_parquet(fp)
        dd["date"] = pd.to_datetime(dd["date"]).dt.date
        day = {}
        for r in dd.itertuples():
            atr = float(r.atr) if pd.notnull(r.atr) else np.nan
            if not (np.isfinite(atr) and atr > 0):
                continue
            o, hh, ll = float(r.open), float(r.high), float(r.low)
            cw, pw = wmap.get((tk, r.date), (np.nan, np.nan))
            f = lambda x: float(x) if pd.notnull(x) else np.nan
            lv = dict(atr=atr, open=o, high=hh, low=ll, cw=cw, pw=pw,
                      wpoc=f(r.wpoc), pdh=f(getattr(r, "pdh", None)),
                      pdl=f(getattr(r, "pdl", None)), pvah=f(getattr(r, "pvah", None)),
                      pval=f(getattr(r, "pval", None)),
                      ib_hi=f(r.ib_hi), ib_lo=f(r.ib_lo),
                      plc_up=o + 0.8 * atr, plc_dn=o - 0.8 * atr)
            seed = ((r.date.toordinal() * 31 + sum(ord(c) for c in tk)) * 2654435761) % (2 ** 32)
            lv["plc_u"] = (ll + np.random.default_rng(seed).random() * (hh - ll)) \
                if hh > ll else np.nan
            day[r.date] = lv
        DAY[tk] = day
        vv = pd.read_parquet(vp)
        vv["date"] = pd.to_datetime(vv["date"]).dt.date
        VWF[tk] = {d: (g["mod"].to_numpy(np.int32), g["vwap"].to_numpy(float),
                       g["vwap_sd"].to_numpy(float))
                   for d, g in vv.groupby("date")}
    return DAY, VWF


def und_map(tk):
    d = pl.read_parquet(f"historical/{tk}.parquet",
                        columns=["start_time", "high", "low", "close"]).to_pandas()
    et = (pd.to_datetime(d["start_time"], utc=True)
          .dt.tz_convert("America/New_York").dt.tz_localize(None))
    d["date"] = et.dt.date
    d["mod"] = (et.dt.hour * 60 + et.dt.minute).astype(int)
    d = d[(d["mod"] >= RTH_LO) & (d["mod"] <= RTH_HI)]
    for c in ("high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return {dt: (g["mod"].to_numpy(np.int32), g["high"].to_numpy(float),
                 g["low"].to_numpy(float), g["close"].to_numpy(float))
            for dt, g in d.sort_values("mod").groupby("date")}


# ---------------------------------------------------------------- the test
def score(df, cubes, tol_d0, tol_rng, match_rng, dmax, minctl, match_dir="sign"):
    recs = []
    for tk, grp in df.groupby("ticker"):
        C = cubes.get(tk)
        if C is None:
            continue
        allrows = np.arange(len(C.dates))
        for r in grp.itertuples():
            i = C.didx.get(r.date)
            if i is None:
                continue
            up = r.dir == "CALL"
            sgn = 1.0 if up else -1.0
            m = int(r.entry)
            if not (RTH_LO <= m <= RTH_HI):
                continue
            c = m - RTH_LO
            p0, atr = C.P[i, c], C.atr[i]
            ext = C.EU[i, c] if up else C.ED[i, c]
            extm = C.EUm[i, c] if up else C.EDm[i, c]
            if not (np.isfinite(p0) and np.isfinite(ext) and atr > 0):
                continue
            s = (ext - p0) * sgn / atr

            # ---- control pool: other days, same clock, no trigger nearby
            ok = (allrows != i) & (~C.trig[:, c])
            if match_rng and np.isfinite(C.rr[i]):
                ok &= (C.rr >= C.rr[i] / (1 + tol_rng)) & (C.rr <= C.rr[i] * (1 + tol_rng))
            rows = allrows[ok]
            if rows.size < minctl:
                continue
            p2, atr2 = C.P[rows, c], C.atr[rows]
            ext2 = C.EU[rows, c] if up else C.ED[rows, c]
            extm2 = C.EUm[rows, c] if up else C.EDm[rows, c]
            live = np.isfinite(p2) & np.isfinite(ext2) & (atr2 > 0)
            s2 = (ext2 - p2) * sgn / atr2
            # ---- DIRECTION MATCHING.
            # Without it the CALL control pool contains days that FELL, whose
            # up-extreme barely clears entry and therefore sits ~d0 away from any
            # overhead level. Controls look far, the real trade looks close, and
            # pct inflates -- a false-positive bias on exactly the directional
            # levels in question.
            # "sign" requires the control window to have moved our way AT ALL.
            # "pre" instead matches the PRE-ENTRY displacement from the open,
            # which is causal and closer to what the trigger actually keys on.
            # Neither matches MAGNITUDE: matching both d0 and s would pin
            # |s - d0| on both sides and the statistic could not move (the A5
            # degeneracy, which has already cost this project four findings).
            if match_dir == "sign":
                live &= (s2 > 0)
            elif match_dir == "pre":
                pre = (p0 - C.open[i]) * sgn / atr
                pre2 = (p2 - C.open[rows]) * sgn / atr2
                live &= np.isfinite(pre2) & (np.abs(pre2 - pre) <= tol_d0 * 2)
            if live.sum() < minctl:
                continue

            base = dict(ticker=tk, dir=r.dir, date=r.date, peak_roe=r.peak_roe)
            for col, side, _ in LEVELS:
                if (side == "up" and not up) or (side == "dn" and up):
                    continue
                if col in VWSPEC:
                    L = C.vw_at(np.array([i]), np.array([extm]), VWSPEC[col])[0]
                    L2 = C.vw_at(rows, extm2, VWSPEC[col])
                else:
                    L, L2 = C.stat[col][i], C.stat[col][rows]
                if not np.isfinite(L):
                    continue
                d0 = (L - p0) * sgn / atr
                if not (0.0 < d0 <= dmax):
                    continue          # behind the move, or out of reach entirely
                d02 = (L2 - p2) * sgn / atr2
                keep = live & np.isfinite(d02) & (np.abs(d02 - d0) <= tol_d0) & (d02 > 0)
                if col in ("ib_hi", "ib_lo"):
                    keep &= (extm2 >= RTH_LO + 60)
                    if extm < RTH_LO + 60:
                        continue
                if keep.sum() < minctl:
                    continue
                dr = abs(s - d0)
                dn = np.abs(s2[keep] - d02[keep])
                recs.append({**base, "level": col, "d0": float(d0), "d_real": float(dr),
                             "pct": float((dn > dr).mean()),
                             "d_null_med": float(np.median(dn)), "nctl": int(keep.sum())})
        print(f"    {tk} scored ({len(recs):,} rows)", flush=True)
    return pd.DataFrame(recs)


def boot_mean(vals, days, n, rng):
    uniq = pd.unique(days)
    idx = {d: np.where(days == d)[0] for d in uniq}
    out = np.empty(n)
    for i in range(n):
        sel = np.concatenate([idx[d] for d in rng.choice(uniq, size=len(uniq), replace=True)])
        out[i] = vals[sel].mean()
    return np.percentile(out, [2.5, 97.5])


def report(sc, boot, rng, title):
    if sc.empty:
        print(f"\n  {title}: no rows")
        return
    print(f"\n  {title}   (n={len(sc):,} trade-levels, {sc['date'].nunique()} days)")
    print(f"  {'level':26} {'n':>6} {'days':>5} {'ctl':>5} {'mean pct':>9} "
          f"{'95% CI':>17} {'med |d|':>8} {'null':>7}")
    for col, _, lbl in LEVELS:
        g = sc[sc["level"] == col]
        if len(g) < 30:
            continue
        v = g["pct"].to_numpy(float)
        ci = boot_mean(v, g["date"].to_numpy(), boot, rng)
        flag = "  <--" if (ci[0] > 0.5 or ci[1] < 0.5) else ""
        mark = "*" if lbl.startswith("PLACEBO") else " "
        print(f" {mark}{lbl:25} {len(g):>6} {g['date'].nunique():>5} "
              f"{g['nctl'].median():>5.0f} {v.mean():>9.3f} "
              f"[{ci[0]:>6.3f},{ci[1]:>6.3f}] {g['d_real'].median():>8.2f} "
              f"{g['d_null_med'].median():>7.2f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="+", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--tol-d0", type=float, default=0.15,
                    help="control's level must sit within this many ATR of the real d0")
    ap.add_argument("--tol-rng", type=float, default=0.30,
                    help="control day's range/ATR within this fraction of the real day's")
    ap.add_argument("--no-match-rng", dest="match_rng", action="store_false")
    ap.add_argument("--dmax", type=float, default=2.0, help="max level distance, ATR")
    ap.add_argument("--match-dir", choices=["none", "sign", "pre"], default="sign",
                    help="hold the control's DIRECTION comparable: 'sign' = the "
                         "control window moved our way at all; 'pre' = match the "
                         "pre-entry displacement from the open (causal)")
    ap.add_argument("--minctl", type=int, default=25)
    ap.add_argument("--excl", type=int, default=30)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    print("  loading peaks...", flush=True)
    df = peak_rows(a.tickers, a.dirs, a.pct)
    if df.empty:
        print("  no trades"); return
    DAY, VWF = load_levels(a.tickers)
    print("  building cubes...", flush=True)
    cubes = {}
    for tk in a.tickers:
        if tk not in DAY:
            continue
        cubes[tk] = Cube(tk, und_map(tk), DAY[tk], VWF.get(tk, {}))
        cubes[tk].mark_triggers(df[df["ticker"] == tk], a.excl)

    print(f"  scoring {len(df):,} trades (tol_d0={a.tol_d0} ATR, "
          f"match_rng={a.match_rng}, match_dir={a.match_dir})...", flush=True)
    sc = score(df, cubes, a.tol_d0, a.tol_rng, a.match_rng, a.dmax, a.minctl,
               a.match_dir)
    if sc.empty:
        print("  nothing scored"); return
    sc.to_parquet("_peak_levels.parquet", index=False)

    print(f"\n{'='*104}")
    print(f"  DO PEAKS LAND ON STRUCTURE?   control = same ticker, same clock, other day,")
    print(f"  matched on level-distance-ahead and day size.  pct > 0.5 = real peak stops")
    print(f"  CLOSER to the level than the control does.")
    print(f"{'='*104}")
    report(sc, a.boot, rng, "ALL TRADES")
    report(sc[sc["peak_roe"] >= 1.0], a.boot, rng, "LARGE PEAKS ONLY (peak ROE >= +100%)")
    near = sc[sc["d0"] <= 0.75]
    report(near, a.boot, rng, "LEVEL CLOSE AHEAD (d0 <= 0.75 ATR)")

    print(f"\n  HOW TO READ IT")
    print(f"  READ THE PLACEBOS (*) FIRST -- they carry no structural meaning and must")
    print(f"  sit at 0.50. Three earlier versions of this test produced fully significant")
    print(f"  tables that the placebos exposed as artifact. If they miss 0.50 again,")
    print(f"  nothing else here is interpretable.")
    print(f"  Then: 'mean pct' = share of matched controls stopping FARTHER from their")
    print(f"  level than the real trade did. 'ctl' is the median controls per trade.")
    print(f"  'med |d|' vs 'null' is the raw gap in ATR -- a significant pct with an")
    print(f"  identical median is a tail effect, not something to rest an order on.")


if __name__ == "__main__":
    main()
