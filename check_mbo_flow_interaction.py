# /// script
# requires-python = ">=3.11"
# dependencies = ["databento", "polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scipy"]
# ///
"""
check_mbo_flow_interaction.py
=============================
🚨 THE A5 VERDICT FROM THIS SCRIPT IS WITHDRAWN -- USE `check_mbo_gating.py`.

A5 compares the gap against `base2m` estimated on the SAME rows. Algebraically,
with p = P(aligned):  gap - base2m = (1 - 2p)(M_al - M_op). Our direction mix is
~50/50 BY CONSTRUCTION, so p ~= 0.5, so the statistic vanishes regardless of the
true effect; the `gap > 1.5*base` gate is then unpassable in both branches. The
"0 of 24 cells passed A5" result is a property of the arithmetic, not of the
market, and carries no information. A1-A4, `load_panel` and `block_perm_p` below
are sound and are imported by the replacement. See METHODOLOGY.md section 6c.

Does the CME order book CONDITION our flow trigger?

THE REFRAME THIS TESTS
----------------------
`check_mbo_directional.py` found MBO carries a real but small directional edge
on the future -- `imb` +0.36 to +4.00bp, `age_diff` +1-2bp, IS/OOS consistent.
Far too small to clear an option's costs on its own. But standalone magnitude
was never the right test for a CONDITIONER: our flow trigger itself has
essentially no standalone edge (`check_signal_quality`: MFE/MAE 1.00-1.02, i.e.
indistinguishable from random entries), and the book still makes money.

So: at a flow trigger, does the state of the order book tell us whether THIS
trigger is worth taking? Same structure as the GEX x flow-sign work, which
produced the most theory-grounded result of the session.

  IWM -> RTY.c.0 (43 MBO days, 2,496 p50+ triggers)
  QQQ -> NQ.c.0  (12 MBO days,   859 p50+ triggers)
3,355 triggers -- 35x the 94 trades that defeated the gate test. Direction mix
is balanced (~50/50 CALL/PUT) on both, so there is no directional skew to
control for.

===================  PRE-COMMITTED CRITERIA (fixed before any output)  =========
  A1  ALIGNED beats OPPOSED on direction-adjusted forward drift
  A2  IS and OOS agree in SIGN (split 2025-08-21)
  A3  CROSS-SECTIONAL, SOFTENED FOR NQ's 12 DAYS -- a significance test on 12
      days is underpowered to the point of only producing false negatives, so
      it is replaced by:
        A3a  sign agreement with RTY
        A3b  magnitude within 1/3x .. 3x of RTY (sign agreement alone is a coin
             flip; a 60x magnitude gap is noise that landed on the right side)
        A3c  feature distributions KS-comparable between instruments -- this IS
             well powered (NQ has 12 DAYS but 4,692 MINUTE observations; the
             power problem is day-clustered inference, not observation count)
      A soft-A3 pass yields "CONSISTENT WITH replication", never "replicated".
  A3-alt  RTY's 43 days split into contiguous THIRDS (~830 triggers each) --
      the effect must appear in all three. Temporal rather than cross-sectional,
      but actually powered, which the NQ check is not.
  A4  beats p95 of a WITHIN-DAY BLOCK PERMUTATION. At 58 triggers/day the
      observations are heavily day-clustered; a naive shuffle would treat 2,496
      as independent and wildly overstate significance. Shuffling labels WITHIN
      each day preserves day structure and tests only the within-day association.
  A5  *** THE CONTROL THAT MATTERS MOST *** -- see below.

A5: WHY A MARGINAL PREDICTOR FAKES AN INTERACTION
-------------------------------------------------
Let s = sign(feature), r = forward return, d = trigger direction (+1/-1), and
m = E[s*r] the feature's MARGINAL effect. Suppose the trigger is pure noise,
independent of everything. Then:

    aligned (d == s):   d*r = s*r    ->  E = +m
    opposed (d == -s):  d*r = -s*r   ->  E = -m
    observed gap = m - (-m) = 2m

**A worthless trigger combined with any marginal predictor produces a gap of
exactly 2x the marginal effect.** So the no-interaction baseline is 2m, NOT
zero and NOT m. A5 requires the observed gap to exceed 2m by a clear margin;
otherwise we are re-measuring the 1-4bp marginal in a costume.

Secondary (NOT a gate): the GEX interaction, since flow-sign alignment REVERSED
by GEX regime and this is the natural place to ask whether book alignment does.

A POSITIVE RESULT COSTS $179/MONTH -- MBO features need a LIVE feed to trade on.
This is a purchase decision, not a pure research question.

Usage:  python check_mbo_flow_interaction.py
        python check_mbo_flow_interaction.py --boot 4000 --gated
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

CACHE = "_mbo_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
PROXY = {"IWM": "RTY_c_0", "QQQ": "NQ_c_0"}
FEATS = ["imb", "absorb_sgn", "tick_sgn", "hhi_diff", "age_diff"]
HORIZONS = [5, 15, 30, 60]


# ------------------------------------------------------------------ data
def load_panel(ticker, sym, gated=False):
    """One row per flow trigger that lands on a day we hold MBO for."""
    import polars as pl
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES

    p = os.path.join(CACHE, f"{sym}.parquet")
    if not os.path.exists(p):
        return None
    f = pl.read_parquet(p).to_pandas()
    f["date"] = pd.to_datetime(f["date"]).dt.date
    f = f.sort_values(["date", "mod"]).reset_index(drop=True)
    for h in HORIZONS:
        f[f"fwd{h}"] = f.groupby("date")["mid"].shift(-h) / f["mid"] - 1.0

    rule = next((r for r in RULES if r["ticker"] == ticker and r.get("enabled", True)), None)
    flow = _flow_for(D, [ticker])
    if flow.empty:
        return None
    trigs = D.triggers_for(flow, ticker)
    D.annotate_flow_pct(trigs, (rule or {}).get("flow_window_days", 60))
    days = set(f["date"])
    pct = (rule or {}).get("min_flow_pct", 50) if gated else 50
    hrs = set((rule or {}).get("hours", range(24))) if gated else set(range(24))

    rows = []
    for t in trigs:
        thr = t.get("thr")
        if not thr or t["abs_flow"] < thr.get(pct, 1e99):
            continue
        d = t["date"]
        if d not in days:
            continue
        ts = pd.Timestamp(t["ts"])
        if ts.hour not in hrs:
            continue
        rows.append(dict(date=d, mod=ts.hour * 60 + ts.minute,
                         dir=1.0 if t["dir"] == "CALL" else -1.0,
                         ticker=ticker, sym=sym))
    if not rows:
        return None
    T = pd.DataFrame(rows).drop_duplicates(subset=["date", "mod"])
    keep = ["date", "mod", "mid"] + FEATS + [f"fwd{h}" for h in HORIZONS]
    M = T.merge(f[keep], on=["date", "mod"], how="inner")

    gex = D.load_gex("historical", ticker)
    M["gex"] = M["date"].map(gex)
    return M


# ------------------------------------------------------------------ stats
def gap_and_baseline(sub, feat, h):
    """(observed aligned-vs-opposed gap, no-interaction baseline 2m, n_al, n_op).

    baseline = 2 * E[sign(feature) * fwd] on the SAME rows -- the gap a
    WORTHLESS trigger would produce given the feature's marginal effect.
    """
    x = sub[feat].to_numpy(float)
    d = sub["dir"].to_numpy(float)
    r = sub[f"fwd{h}"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(r) & (x != 0)
    if m.sum() < 60:
        return np.nan, np.nan, 0, 0
    x, d, r = x[m], d[m], r[m]
    s = np.sign(x)
    al = (s == d)
    if al.sum() < 20 or (~al).sum() < 20:
        return np.nan, np.nan, int(al.sum()), int((~al).sum())
    obs = float(np.mean(d[al] * r[al]) - np.mean(d[~al] * r[~al]))
    base = 2.0 * float(np.mean(s * r))
    return obs, base, int(al.sum()), int((~al).sum())


def block_perm_p(sub, feat, h, boot, rng):
    """p-value under WITHIN-DAY label shuffling (preserves day clustering)."""
    x = sub[feat].to_numpy(float)
    d = sub["dir"].to_numpy(float)
    r = sub[f"fwd{h}"].to_numpy(float)
    day = sub["date"].to_numpy()
    m = np.isfinite(x) & np.isfinite(r) & (x != 0)
    x, d, r, day = x[m], d[m], r[m], day[m]
    if len(x) < 60:
        return np.nan
    s = np.sign(x)
    obs = np.mean(d[s == d] * r[s == d]) - np.mean(d[s != d] * r[s != d]) \
        if (s == d).sum() > 5 and (s != d).sum() > 5 else np.nan
    if not np.isfinite(obs):
        return np.nan
    idx_by_day = {}
    for i, dd in enumerate(day):
        idx_by_day.setdefault(dd, []).append(i)
    blocks = [np.array(v) for v in idx_by_day.values()]
    hits = 0
    rr = r.copy()
    for _ in range(boot):
        for b in blocks:
            rr[b] = r[rng.permutation(b)]
        al = (s == d)
        if al.sum() < 5 or (~al).sum() < 5:
            continue
        g = np.mean(d[al] * rr[al]) - np.mean(d[~al] * rr[~al])
        if abs(g) >= abs(obs):
            hits += 1
    return hits / boot


def run(a):
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(11)

    panels = {}
    for tk, sym in PROXY.items():
        P = load_panel(tk, sym, a.gated)
        if P is None or P.empty:
            print(f"  ! no panel for {tk}"); continue
        panels[sym] = P
        print(f"  {tk} -> {sym}: {len(P):,} triggers on {P['date'].nunique()} days "
              f"(IS {int((P.date < SPLIT).sum())} / OOS {int((P.date >= SPLIT).sum())})")
    if "RTY_c_0" not in panels:
        print("  need RTY to proceed"); return
    R = panels["RTY_c_0"]

    print("\n" + "=" * 112)
    print("  A1/A2/A4/A5  —  RTY (the powered instrument).  All figures in bp.")
    print("  gap = aligned minus opposed, direction-adjusted.")
    print("  base2m = the gap a WORTHLESS trigger would produce from the feature's")
    print("           marginal effect alone. The gap must clearly EXCEED this.")
    print("=" * 112)
    print(f"  {'feature':11} {'h':>4} {'gap':>8} {'base2m':>8} {'gap-base':>9} "
          f"{'IS':>8} {'OOS':>8} {'blockp':>7} {'n_al/n_op':>12}  A1 A2 A4 A5")
    results = {}
    for f in FEATS:
        for h in HORIZONS:
            obs, base, nal, nop = gap_and_baseline(R, f, h)
            if not np.isfinite(obs):
                continue
            gi, _, _, _ = gap_and_baseline(R[R.date < SPLIT], f, h)
            go, _, _, _ = gap_and_baseline(R[R.date >= SPLIT], f, h)
            pv = block_perm_p(R, f, h, a.boot, rng)
            a1 = obs > 0
            a2 = np.isfinite(gi) and np.isfinite(go) and np.sign(gi) == np.sign(go)
            a4 = np.isfinite(pv) and pv < 0.05
            a5 = obs > base * 1.5 if base > 0 else obs > abs(base) * 0.5
            results[(f, h)] = dict(obs=obs, base=base, is_=gi, oos=go, p=pv,
                                   a1=a1, a2=a2, a4=a4, a5=a5)
            print(f"  {f:11} {h:>3}m {obs*1e4:>+8.2f} {base*1e4:>+8.2f} "
                  f"{(obs-base)*1e4:>+9.2f} {gi*1e4:>+8.2f} {go*1e4:>+8.2f} "
                  f"{pv:>7.3f} {nal:>5}/{nop:<6} "
                  f" {'Y' if a1 else '.'}  {'Y' if a2 else '.'}  "
                  f"{'Y' if a4 else '.'}  {'Y' if a5 else '.'}")
        print()

    # ---- A3-alt: temporal thirds of RTY ----
    print("=" * 112)
    print("  A3-alt  TEMPORAL REPLICATION — RTY's days split into contiguous thirds")
    print("  (powered, unlike the NQ check: ~830 triggers per third)")
    print("=" * 112)
    days = sorted(R["date"].unique())
    cut = [days[:len(days)//3], days[len(days)//3:2*len(days)//3], days[2*len(days)//3:]]
    live = [k for k, v in results.items() if v["a1"] and v["a2"]]
    if not live:
        print("  (no feature/horizon passed A1+A2; nothing to replicate)")
    for (f, h) in live:
        cells, signs = [], []
        for ds in cut:
            g, _, _, _ = gap_and_baseline(R[R["date"].isin(ds)], f, h)
            cells.append(f"{g*1e4:>+9.2f}" if np.isfinite(g) else "      n/a")
            signs.append(np.sign(g) if np.isfinite(g) else 0)
        ok = len(set(s for s in signs if s != 0)) == 1 and 0 not in signs
        results[(f, h)]["a3alt"] = ok
        print(f"  {f:11} {h:>3}m  " + "  ".join(cells) + f"   {'Y' if ok else '.'}")

    # ---- A3: NQ, softened ----
    if "NQ_c_0" in panels:
        N = panels["NQ_c_0"]
        print("\n" + "=" * 112)
        print("  A3  CROSS-SECTIONAL (SOFT) — NQ. Sign agreement + magnitude band 1/3x..3x.")
        print("  A pass here means CONSISTENT WITH replication, never 'replicated'.")
        print("=" * 112)
        print(f"  {'feature':11} {'h':>4} {'RTY gap':>10} {'NQ gap':>10} {'ratio':>8}"
              f"   A3a A3b")
        for (f, h) in live:
            gn, _, _, _ = gap_and_baseline(N, f, h)
            gr = results[(f, h)]["obs"]
            if not np.isfinite(gn):
                print(f"  {f:11} {h:>3}m {gr*1e4:>+10.2f} {'n/a':>10}")
                continue
            a3a = np.sign(gn) == np.sign(gr)
            ratio = (gn / gr) if gr != 0 else np.nan
            a3b = np.isfinite(ratio) and (1/3) <= abs(ratio) <= 3
            results[(f, h)]["a3a"], results[(f, h)]["a3b"] = a3a, a3b
            print(f"  {f:11} {h:>3}m {gr*1e4:>+10.2f} {gn*1e4:>+10.2f} "
                  f"{ratio:>8.2f}    {'Y' if a3a else '.'}   {'Y' if a3b else '.'}")

        print("\n  A3c  feature distributions KS-comparable (well powered: minute-level)")
        import polars as pl
        fr = pl.read_parquet(os.path.join(CACHE, "RTY_c_0.parquet")).to_pandas()
        fn = pl.read_parquet(os.path.join(CACHE, "NQ_c_0.parquet")).to_pandas()
        for f in FEATS:
            x, y = fr[f].dropna().to_numpy(), fn[f].dropna().to_numpy()
            if len(x) < 100 or len(y) < 100:
                continue
            ks = ks_2samp(x, y)
            print(f"     {f:11} KS={ks.statistic:>5.3f}  p={ks.pvalue:>8.2g}   "
                  f"RTY median {np.median(x):>+8.4f}  NQ median {np.median(y):>+8.4f}")

    # ---- verdict ----
    print("\n" + "=" * 112)
    print("  VERDICT")
    print("=" * 112)
    passed = []
    for (f, h), v in sorted(results.items()):
        flags = "".join("Y" if v.get(k) else "." for k in
                        ("a1", "a2", "a4", "a5", "a3alt", "a3a", "a3b"))
        if all(v.get(k) for k in ("a1", "a2", "a4", "a5", "a3alt", "a3a", "a3b")):
            passed.append((f, h))
            print(f"  {f:11} {h:>3}m  A1A2A4A5|A3alt|A3ab = {flags}   ** PASS")
    if not passed:
        print("  Nothing cleared every criterion.")
        strong = [(k, v) for k, v in results.items()
                  if v["a1"] and v["a2"] and v.get("a5")]
        if strong:
            print("\n  Closest (A1+A2+A5, i.e. right sign, stable halves, and exceeding")
            print("  the marginal-effect baseline) — but failing replication or the")
            print("  block permutation:")
            for (f, h), v in strong:
                print(f"    {f} {h}m: gap {v['obs']*1e4:+.2f}bp vs base {v['base']*1e4:+.2f}bp, "
                      f"blockp {v['p']:.3f}")
    else:
        print("\n  A positive result here is a $179/month decision (live MBO feed).")
        print("  Re-read A5 before acting: the gap must EXCEED 2x the marginal effect,")
        print("  or it is the same 1-4bp edge wearing a costume.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--gated", action="store_true",
                    help="apply the rule's own min_flow_pct + hours (smaller, more realistic)")
    run(ap.parse_args())
