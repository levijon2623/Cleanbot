# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
build_tpo.py
============
DEVELOPING TPO (Time Price Opportunity) profile, evaluated at an arbitrary minute.

WHAT IT BUILDS
    30-minute brackets (A = 09:30-10:00 ... M = 15:30-16:00) over ATR-normalised
    price bins. A bracket "prints" every bin its high-low range touched, so the
    TPO count of a bin is the NUMBER OF BRACKETS that traded there. Row sums give
    the POC (modal bin), and the Value Area is grown outward from the POC until
    it holds `va_frac` of all TPOs. Bins with a count of exactly 1 are SINGLE
    PRINTS -- the vacuums price moved through without spending time.

DEVELOPING, NOT FINAL -- this is the whole point
    At minute m only brackets that have STARTED are known, and the one in
    progress contributes only the range it has covered SO FAR. `profile_at(m)`
    therefore reconstructs what a trader could actually have seen at m. Using the
    finished day's profile would leak the afternoon into a 10:00 decision, which
    is the same look-ahead that made a full-session volume profile useless for
    explaining peaks earlier in this project.

WHY THIS IS NOT THE AMT WORK ALREADY RULED OUT
    check_amt / check_amt_variants tested the PRIOR day's volume POC and value
    area (0 of 48 cells passed) and check_wpoc_gate tested a 5-day rolling volume
    POC (a powered null, |gain| < 0.6bp on n=81,727). Both are VOLUME profiles of
    a COMPLETED period. This is a TIME profile of the CURRENT, INCOMPLETE session,
    and single prints have no volume-profile analogue at all. Adjacent, not a
    re-run -- but those nulls are the prior.

🚨 THE DEGENERACY THIS IS BUILT TO EXPOSE
    Early in the session almost every bin is a single print, because only one or
    two brackets exist. "Fires into a single print" would then mostly mean "fires
    early", and time-of-day is already known to matter here. `--preflight`
    measures the single-print share BY BRACKET before any hypothesis is tested.
    If the share does not fall sharply through the day, the classifier cannot
    separate the two ideas and the test is not worth running.

Usage:
  python build_tpo.py --preflight --tickers SPY QQQ IWM
  python build_tpo.py --preflight --bin-atr 0.05
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
BRACKET = 30
CACHE = "_level_cache"


def und_1m(tk):
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
            for dt, g in d.dropna(subset=["close"]).sort_values("mod").groupby("date")}


def atr_map(tk):
    """Prior-14-session ATR, already shifted causal in build_level_tape."""
    p = os.path.join(CACHE, f"und_{tk}.parquet")
    if not os.path.exists(p):
        return {}
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return {r.date: float(r.atr) for r in d.itertuples()
            if pd.notnull(r.atr) and r.atr > 0}


def bracket_of(mod):
    return int((mod - RTH_LO) // BRACKET)


class DayTPO:
    """Developing TPO for one session.

    A bracket's TPOs are CONTIGUOUS -- it prints every bin between its low and
    its high -- so each bracket reduces to an interval [lo_bin, hi_bin] and the
    count for a bin is how many intervals cover it. That makes `profile_at`
    O(brackets) instead of O(bins x brackets), and it is exact rather than an
    approximation.
    """

    def __init__(self, mods, hi, lo, atr, bin_atr, anchor):
        self.mods, self.hi, self.lo = mods, hi, lo
        self.w = max(1e-9, bin_atr * atr)
        self.anchor = anchor                    # price -> bin origin
        self.nb = 1 + bracket_of(RTH_HI)

    def _bin(self, px):
        return int(np.floor((px - self.anchor) / self.w))

    def intervals_at(self, m):
        """[(bracket, lo_bin, hi_bin)] using only bars at or before minute m."""
        out = []
        sel = self.mods <= m
        if not sel.any():
            return out
        b = ((self.mods[sel] - RTH_LO) // BRACKET).astype(int)
        h, l = self.hi[sel], self.lo[sel]
        for k in range(b.max() + 1):
            msk = b == k
            if not msk.any():
                continue
            out.append((k, self._bin(l[msk].min()), self._bin(h[msk].max())))
        return out

    def profile_at(self, m, va_frac=0.70):
        """-> dict with counts, poc, vah, val, single-print bins, n_brackets."""
        iv = self.intervals_at(m)
        if not iv:
            return None
        lo_b = min(x[1] for x in iv)
        hi_b = max(x[2] for x in iv)
        n = hi_b - lo_b + 1
        cnt = np.zeros(n, int)
        for _, a, z in iv:
            cnt[a - lo_b:z - lo_b + 1] += 1
        tot = cnt.sum()
        if tot <= 0:
            return None
        poc_i = int(np.argmax(cnt))
        loi = hii = poc_i
        acc = cnt[poc_i]
        while acc < va_frac * tot and (loi > 0 or hii < n - 1):
            up = cnt[hii + 1] if hii < n - 1 else -1
            dn = cnt[loi - 1] if loi > 0 else -1
            if hii < n - 1 and (up >= dn or loi == 0):
                hii += 1; acc += cnt[hii]
            elif loi > 0:
                loi -= 1; acc += cnt[loi]
            else:
                break
        return dict(lo_bin=lo_b, counts=cnt, n_brackets=len(iv),
                    poc=lo_b + poc_i, vah=lo_b + hii, val=lo_b + loi,
                    singles={lo_b + i for i in range(n) if cnt[i] == 1},
                    width=self.w, anchor=self.anchor)

    def classify(self, m, px, va_frac=0.70):
        """Where does price `px` sit in the DEVELOPING profile as of minute m?"""
        p = self.profile_at(m, va_frac)
        if p is None:
            return None
        b = self._bin(px)
        return dict(bin=b, single=(b in p["singles"]),
                    in_va=(p["val"] <= b <= p["vah"]),
                    above_va=(b > p["vah"]), below_va=(b < p["val"]),
                    dist_poc_atr=(b - p["poc"]) * p["width"] / max(self.w / max(1e-9, 1), 1e-9),
                    n_brackets=p["n_brackets"],
                    n_singles=len(p["singles"]), n_bins=len(p["counts"]))


def day_tpo(tk, d, bars, atrs, bin_atr):
    arr = bars.get(d)
    atr = atrs.get(d)
    if arr is None or not atr:
        return None
    mods, hi, lo, cl = arr
    return DayTPO(mods, hi, lo, atr, bin_atr, anchor=float(cl[0]))


def preflight(tickers, bin_atr, va_frac, sample):
    print(f"  PRE-FLIGHT -- can the single-print classifier actually separate?\n"
          f"  bins = {bin_atr} ATR, value area = {va_frac:.0%}\n")
    rows = []
    for tk in tickers:
        bars, atrs = und_1m(tk), atr_map(tk)
        days = sorted(set(bars) & set(atrs))[-sample:]
        for d in days:
            T = day_tpo(tk, d, bars, atrs, bin_atr)
            if T is None:
                continue
            mods = bars[d][0]
            for m in range(RTH_LO + BRACKET, RTH_HI + 1, 10):
                p = T.profile_at(m, va_frac)
                if p is None:
                    continue
                rows.append(dict(tk=tk, date=d, mod=m, brk=p["n_brackets"],
                                 nbins=len(p["counts"]),
                                 nsing=len(p["singles"]),
                                 share=len(p["singles"]) / len(p["counts"]),
                                 va_w=p["vah"] - p["val"] + 1))
    R = pd.DataFrame(rows)
    if R.empty:
        print("  no data"); return

    print(f"  SINGLE-PRINT SHARE OF BINS, BY BRACKET  (n={len(R):,} snapshots)")
    print(f"  {'bracket':>8} {'clock':>13} {'bins':>6} {'singles':>8} "
          f"{'share':>7} {'VA width':>9}")
    for b, g in R.groupby("brk"):
        t0 = RTH_LO + (b - 1) * BRACKET
        print(f"  {b:>8} {t0//60:02d}:{t0%60:02d}-{(t0+BRACKET)//60:02d}:"
              f"{(t0+BRACKET)%60:02d}   {g['nbins'].median():>6.0f} "
              f"{g['nsing'].median():>8.0f} {g['share'].median():>6.0%} "
              f"{g['va_w'].median():>9.0f}")
    early = R[R["brk"] <= 2]["share"].median()
    late = R[R["brk"] >= 10]["share"].median()
    print(f"\n  early (<=2 brackets) {early:.0%}  ->  late (>=10 brackets) {late:.0%}")
    if early > 0.9 and late > 0.5:
        print(f"  ⚠️  the share stays high all session: 'single print' would be")
        print(f"     close to unconditional and cannot separate from time of day.")
    elif early > 0.9:
        print(f"  NOTE: early snapshots are ~all single print BY CONSTRUCTION")
        print(f"     (one bracket = every bin printed once). Any test MUST either")
        print(f"     control for time of day or require a minimum bracket count.")
    print(f"\n  -> a usable test needs enough brackets for the profile to have")
    print(f"     shape. Suggest requiring n_brackets >= 4 (i.e. from ~11:30).")


def preflight_triggers(tickers, bin_atr, va_frac, pct, min_brackets):
    """WHERE DO REAL TRIGGERS LAND? -- the pre-flight that actually gates the test.

    A bin being a single print says nothing about whether a TRIGGER ever fires
    there. Price spends its time near the POC by construction, so the value area
    could absorb almost every trigger and leave the single-print group empty.
    The hypothesis needs BOTH groups populated on enough DAYS to compare.
    """
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rows = []
    for tk in tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        bars, atrs = und_1m(tk), atr_map(tk)
        cache = {}
        for t in trigs:
            # annotate_flow_pct attaches thr = {percentile: dollar_threshold},
            # or None during the warmup window (no same-day look-ahead, and it
            # needs >=30 prior triggers). Filtering on a non-existent
            # "flow_pct" key silently matched nothing.
            thr = t.get("thr")
            if not thr:
                continue
            cut = thr.get(pct) or thr.get(min(thr, key=lambda P: abs(P - pct)))
            if cut is None or t["abs_flow"] < cut:
                continue
            d = t["date"]
            ts = pd.Timestamp(t["ts"])
            m = ts.hour * 60 + ts.minute
            if d not in cache:
                cache[d] = day_tpo(tk, d, bars, atrs, bin_atr)
            T = cache[d]
            if T is None:
                continue
            arr = bars.get(d)
            j = int(np.searchsorted(arr[0], m))
            if j >= len(arr[0]) or arr[0][j] != m:
                continue
            c = T.classify(m, float(arr[3][j]), va_frac)
            if c is None or c["n_brackets"] < min_brackets:
                continue
            rows.append(dict(tk=tk, date=d, mod=m, dir=t["dir"], **c))
        print(f"    {tk} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  no classified triggers"); return
    print(f"\n  TRIGGER LOCATION IN THE DEVELOPING TPO "
          f"(p{pct}+, >= {min_brackets} brackets)")
    print(f"  n = {len(R):,} triggers on {R['date'].nunique()} days\n")
    print(f"  {'group':22} {'n':>8} {'share':>7} {'days':>6}")
    groups = [("single print", R["single"]),
              ("inside value area", R["in_va"] & ~R["single"]),
              ("above VA, not single", R["above_va"] & ~R["single"]),
              ("below VA, not single", R["below_va"] & ~R["single"])]
    for lbl, msk in groups:
        g = R[msk]
        print(f"  {lbl:22} {len(g):>8} {len(g)/len(R)*100:>6.1f}% "
              f"{g['date'].nunique():>6}")
    ns, nv = R["single"].sum(), (R["in_va"] & ~R["single"]).sum()
    ds = R[R["single"]]["date"].nunique()
    print(f"\n  the two groups the hypothesis compares: "
          f"single {ns:,} on {ds} days vs value-area {nv:,}")
    if ds < 30:
        print(f"  ⚠️  {ds} days of single-print triggers. Power is set by DAYS")
        print(f"     (METHODOLOGY 7) -- below ~30 this cannot resolve anything.")
    else:
        print(f"  both groups populated across enough days to be worth testing.")
    print(f"\n  SINGLE-PRINT SHARE BY HOUR -- the confound to carry into the test")
    for h, g in R.groupby(R["mod"] // 60):
        print(f"    {int(h):02d}:00  n={len(g):>6}  single {g['single'].mean()*100:>5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--bin-atr", type=float, default=0.10)
    ap.add_argument("--va-frac", type=float, default=0.70)
    ap.add_argument("--sample", type=int, default=60, help="most recent N days")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--preflight-triggers", action="store_true")
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--min-brackets", type=int, default=4)
    a = ap.parse_args()
    if a.preflight:
        preflight(a.tickers, a.bin_atr, a.va_frac, a.sample)
    elif a.preflight_triggers:
        preflight_triggers(a.tickers, a.bin_atr, a.va_frac, a.pct, a.min_brackets)
    else:
        print("  nothing to do -- pass --preflight (the test itself is "
              "check_tpo_singles.py, gated on this passing)")


if __name__ == "__main__":
    main()
