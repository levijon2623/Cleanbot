# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_tpo_singles.py
====================
DO FLOW TRIGGERS FIRING INTO SINGLE-PRINT VACUUMS SUFFER LESS ADVERSE SELECTION
THAN TRIGGERS FIRING INSIDE THE DEVELOPING VALUE AREA?

THE HYPOTHESIS
    A single print is a price bin the session traded through ONCE -- no time
    was spent there, so there is no accepted value and little resting interest.
    A trigger firing into that vacuum should meet less opposition than one
    firing inside the value area, where the session has already agreed on price
    and two-sided flow is dense.

TWO OUTCOMES, ON DELIBERATELY DIFFERENT POPULATIONS
    MAE     max adverse excursion of the UNDERLYING over the next H minutes,
            direction-adjusted, in ATR. Exit-free and option-free, so it runs on
            EVERY trigger -- 49,708 on 654 days. This is the powered test, and
            it isolates the trigger from the exit, which is the point.
    loss50  the rule's own simulated P&L ending <= -50% (check_skip_hunt's
            definition). Option-level and EXIT-DEPENDENT, so it only exists for
            candidates the rules would actually take -- a few thousand, on far
            fewer days. It answers "would a TPO gate help the book", which is a
            different and much weaker question than "is the trigger better".
    They are reported separately and never pooled.

=========================  THE CONTROLS  =========================
PLACEBO: a fake label assigned at the SAME per-day rate as the real one. It has
    no structural meaning, so it must show no difference. Without it a
    day-clustered bootstrap on 419 days will happily produce a confident number
    from nothing -- which is how three tests died earlier this month.
TIME OF DAY: the pre-flight found single-print share drifts 9.2% (11:00) ->
    5.5% (15:00), so an unstratified comparison partly measures the clock. The
    hour-matched estimate averages WITHIN-hour differences; read that one.
BRACKET FLOOR: profiles with < --min-brackets are excluded. Early in the session
    every bin is a single print by construction (one bracket = one print each),
    so without a floor "single print" would largely mean "before 11:00".

Usage:
  python check_tpo_singles.py
  python check_tpo_singles.py --horizon 30 --min-brackets 4
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sim_core
from build_tpo import und_1m, atr_map, day_tpo, RTH_LO

SPLIT = pd.Timestamp("2025-08-21").date()


def boot_diff(a, b, days_a, days_b, n, rng):
    """Day-clustered bootstrap of mean(a) - mean(b), resampling DAYS jointly."""
    days = np.array(sorted(set(days_a) | set(days_b)))
    ia = {d: np.where(days_a == d)[0] for d in days}
    ib = {d: np.where(days_b == d)[0] for d in days}
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        sa = np.concatenate([ia[d] for d in pick if len(ia[d])]) if any(len(ia[d]) for d in pick) else None
        sb = np.concatenate([ib[d] for d in pick if len(ib[d])]) if any(len(ib[d]) for d in pick) else None
        out[i] = (a[sa].mean() if sa is not None else np.nan) - \
                 (b[sb].mean() if sb is not None else np.nan)
    out = out[np.isfinite(out)]
    return tuple(np.percentile(out, [2.5, 97.5])) if out.size else (np.nan, np.nan)


def build_mae(tickers, bin_atr, va_frac, pct, min_brackets, horizon, rng):
    """One row per trigger: TPO location + forward MAE/MFE on the underlying."""
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
            thr = t.get("thr")
            if not thr:
                continue
            cut = thr.get(pct) or thr.get(min(thr, key=lambda P: abs(P - pct)))
            if cut is None or t["abs_flow"] < cut:
                continue
            d = t["date"]
            arr, atr = bars.get(d), atrs.get(d)
            if arr is None or not atr:
                continue
            ts = pd.Timestamp(t["ts"])
            m = ts.hour * 60 + ts.minute
            if d not in cache:
                cache[d] = day_tpo(tk, d, bars, atrs, bin_atr)
            T = cache[d]
            if T is None:
                continue
            mods, hi, lo, cl = arr
            j = int(np.searchsorted(mods, m))
            if j >= len(mods) or mods[j] != m:
                continue
            c = T.classify(m, float(cl[j]), va_frac)
            if c is None or c["n_brackets"] < min_brackets:
                continue
            k = int(np.searchsorted(mods, m + horizon, side="right"))
            if k - j < 5:
                continue
            up = t["dir"] == "CALL"
            p0 = float(cl[j])
            # adverse = worst move AGAINST the trade, as a POSITIVE ATR figure
            adv = (p0 - lo[j:k].min()) if up else (hi[j:k].max() - p0)
            fav = (hi[j:k].max() - p0) if up else (p0 - lo[j:k].min())
            rows.append(dict(tk=tk, date=d, mod=m, hour=m // 60, dir=t["dir"],
                             single=bool(c["single"]), in_va=bool(c["in_va"]),
                             mae=max(0.0, adv) / atr, mfe=max(0.0, fav) / atr))
        print(f"    {tk} done", flush=True)
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    # PLACEBO: same per-day rate, no meaning. Must come back flat.
    R["plc"] = False
    for d, g in R.groupby("date"):
        k = int(g["single"].sum())
        if k:
            R.loc[rng.choice(g.index.to_numpy(), size=min(k, len(g)),
                             replace=False), "plc"] = True
    return R


def paired_by_day(R, col, value="mae"):
    """Per-DAY difference: mean(labelled) - mean(value-area, unlabelled).

    Pooled means do NOT cancel the day effect. The placebo takes the same COUNT
    per day as the real label, so both concentrate on the same days -- and days
    dense in single prints are volatile days with high MAE throughout. Pooling
    then credits that day composition to the label. Differencing WITHIN each day
    removes it exactly: same session, same regime, same volatility.
    """
    out = []
    for d, g in R.groupby("date"):
        a = g[g[col]]
        b = g[(~g[col]) & g["in_va"]]
        if len(a) < 1 or len(b) < 3:
            continue
        out.append((d, a[value].mean() - b[value].mean(), len(a)))
    return pd.DataFrame(out, columns=["date", "diff", "n"])


def report_paired(R, col, label, nboot, rng, value="mae"):
    P = paired_by_day(R, col, value)
    if len(P) < 30:
        print(f"  {label}: only {len(P)} usable days")
        return
    v = P["diff"].to_numpy(float)
    idx = np.arange(len(v))
    bs = np.array([v[rng.choice(idx, size=len(idx), replace=True)].mean()
                   for _ in range(nboot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "  <--" if (lo > 0 or hi < 0) else ""
    print(f"  {label:28} {len(P):>6} {v.mean():>+10.4f} "
          f"[{lo:>+8.4f},{hi:>+8.4f}] {np.median(v):>+9.4f}{sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--bin-atr", type=float, default=0.10)
    ap.add_argument("--va-frac", type=float, default=0.70)
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--min-brackets", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    print(f"  building trigger panel (H={a.horizon}m, >= {a.min_brackets} brackets)")
    R = build_mae(a.tickers, a.bin_atr, a.va_frac, a.pct, a.min_brackets,
                  a.horizon, rng)
    if R.empty:
        print("  nothing"); return
    R.to_parquet("_tpo_singles.parquet", index=False)

    print(f"\n{'='*104}")
    print(f"  A. ADVERSE EXCURSION (ATR) IN THE {a.horizon}m AFTER THE TRIGGER")
    print(f"  n={len(R):,} triggers, {R['date'].nunique()} days. LOWER MAE = less "
          f"adverse selection.")
    print(f"{'='*104}")
    print(f"  POOLED (day effect NOT removed -- shown only to expose it)")
    for lbl, col in (("single print vs VA", "single"), ("* PLACEBO", "plc")):
        s = R[R[col]]
        v = R[(~R[col]) & R["in_va"]]
        print(f"    {lbl:24} n={len(s):>6,} MAE {s['mae'].mean():.3f}   "
              f"vs n={len(v):>6,} MAE {v['mae'].mean():.3f}   "
              f"diff {s['mae'].mean()-v['mae'].mean():>+7.3f}")

    print(f"\n  WITHIN-DAY PAIRED (the readable estimate)")
    print(f"  {'comparison':28} {'days':>6} {'mean diff':>10} "
          f"{'95% CI':>20} {'median':>9}")
    report_paired(R, "single", "single print vs VA", a.boot, rng)
    report_paired(R, "plc", "* PLACEBO (same rate/day)", a.boot, rng)

    print(f"\n  HOUR-MATCHED (the pre-flight found single-print share drifts by hour)")
    parts, wts = [], []
    for h, g in R.groupby("hour"):
        s, v = g[g["single"]], g[(~g["single"]) & g["in_va"]]
        if len(s) < 25 or len(v) < 25:
            continue
        d = s["mae"].mean() - v["mae"].mean()
        parts.append(d); wts.append(len(s))
        print(f"    {int(h):02d}:00  single n={len(s):>5} MAE {s['mae'].mean():.3f}   "
              f"VA n={len(v):>6} MAE {v['mae'].mean():.3f}   diff {d:>+7.3f}")
    if parts:
        w = np.array(wts, float)
        print(f"    weighted within-hour diff: "
              f"{np.average(parts, weights=w):+.3f} ATR")

    print(f"\n  FOR CONTEXT -- favourable excursion (is it just more movement?)")
    for lbl, msk in (("single print", R["single"]),
                     ("value area", (~R["single"]) & R["in_va"])):
        g = R[msk]
        print(f"    {lbl:14} MFE {g['mfe'].mean():.3f}   MAE {g['mae'].mean():.3f}   "
              f"MFE/MAE {g['mfe'].mean()/max(g['mae'].mean(),1e-9):.2f}")

    print(f"\n  HOW TO READ IT")
    print(f"  The PLACEBO row must be flat. If it is not, the day-clustered")
    print(f"  bootstrap is finding structure in a random label and the real row")
    print(f"  means nothing.")
    print(f"  A lower MAE for single prints WITH a similar MFE is the hypothesis")
    print(f"  holding. A lower MAE with a lower MFE is just quieter conditions --")
    print(f"  less of everything, not less adverse selection.")
    print(f"\n  loss50 (exit-dependent, far weaker) -> run --loss50 once this")
    print(f"  passes; there is no point pricing an exit on a dead trigger.")


if __name__ == "__main__":
    main()
