# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_trough_anatomy.py
=======================
DOES THE SHAKEOUT BOTTOM LEAVE A FOOTPRINT?

THE QUESTION
    check_reentry established that re-entering a stopped-out trade on a FRESH
    ATM contract is a good trade -- 61 re-entries, +824, average winner +142.9
    against the book's +97.6 -- and that it still loses because it consumes the
    one-position slot a fresh trigger would have used. The next move is to scale
    IN rather than replace, which makes the timing of the reclaim the thing
    worth understanding.

    So: at the exact minute the UNDERLYING makes its maximum adverse excursion
    after entry, is that bar distinguishable? Capitulation volume, range
    expansion, an RSI extreme, or cumulative flow refusing to confirm the new
    low?

🚨 TWO OUTCOME-CONDITIONING TRAPS, AND THE CONTROLS FOR EACH
    This selects a bar by looking at the future TWICE, and the peak work went
    through four rebuilds learning what that does.

    1  THE TROUGH IS AN ARGMIN. Any bar picked as "the lowest" is guaranteed to
       look extreme on anything correlated with price. The control that worked
       for peaks (check_peak_bar) is used again: compare the trough against the
       OTHER BARS IN THE SAME DRAWDOWN THAT ALSO MADE A NEW ADVERSE EXTREME.
       Every comparison bar was, at its moment, "the worst so far" -- only one
       turned out to be the last. That removes the argmin advantage and asks the
       only useful question: at the time, was the real bottom different?
       Reported as P(trough ranks above a prior new-extreme bar); 0.5 is null.
       Ties take MID-RANKS -- scoring them as zero dragged the peak study's
       discrete features to a spurious 0.28-0.43.

    2  "TRADES THAT EVENTUALLY RECOVERED" IS THE OUTCOME. Measuring only
       recovered drawdowns tells you what a recovery looks like in hindsight,
       which no live bot can use. So every statistic is also computed on the
       drawdowns that did NOT recover, and the CONTRAST between them is the
       decision-relevant number. If a capitulation trough looks identical
       whether or not price comes back, the footprint cannot time a scale-in.

DEFINITIONS (all causal at the trough minute except the labels themselves)
    window     entry minute .. +120m, capped at 15:00 (the re-entry cutoff)
    trough     adverse extreme of the UNDERLYING in that window -- lowest low
               for a CALL, highest high for a PUT
    recovered  the underlying returns to the entry spot after the trough and
               before 15:00
    rolling stats use a trailing 30m window with the current minute EXCLUDED

Usage:
  python check_trough_anatomy.py --paper
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

import sim_core

RTH0, RTH1, CUTOFF = 570, 955, 900


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def bars(tk):
    df = (pl.scan_parquet(f"historical/{tk}.parquet")
          .select("date", "minute_et", "close", "high", "low", "volume")
          .collect().to_pandas())
    t = pd.to_datetime(df["minute_et"])
    df["mod"] = t.dt.hour * 60 + t.dt.minute
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["mod"] >= RTH0) & (df["mod"] <= RTH1)].sort_values(["date", "mod"])
    out = {}
    for d, g in df.groupby("date"):
        g = g.copy()
        g["rng"] = g["high"] - g["low"]
        g["rsi"] = rsi(g["close"])
        g["vol_avg"] = g["volume"].rolling(30, min_periods=10).mean().shift(1)
        g["rng_avg"] = g["rng"].rolling(30, min_periods=10).mean().shift(1)
        out[d] = g.set_index("mod")
    return out


def flows(D, tk):
    from check_config_walkforward import _flow_for
    f = _flow_for(D, [tk])
    if f.empty:
        return {}
    g = f[f["underlying_symbol"] == tk].copy()
    ts = pd.to_datetime(g["minute_et"])
    g["date"] = ts.dt.date
    g["mod"] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)
    return {d: dict(zip(x["mod"], x["cum_flow"].astype(float)))
            for d, x in g.groupby("date")}


def midrank_auc(target, pool):
    """P(target > a random pool member), ties at 0.5. 0.5 = no separation."""
    pool = np.asarray([p for p in pool if np.isfinite(p)], float)
    if not np.isfinite(target) or len(pool) == 0:
        return np.nan
    return float((pool < target).sum() + 0.5 * (pool == target).sum()) / len(pool)


def feats(g, m, cum, prev_extreme_mod):
    """The bar's footprint at minute m. None of it looks forward."""
    if m not in g.index:
        return None
    r = g.loc[m]
    va, ra = r["vol_avg"], r["rng_avg"]
    out = dict(
        vol_ratio=(r["volume"] / va) if (np.isfinite(va) and va > 0) else np.nan,
        rng_ratio=(r["rng"] / ra) if (np.isfinite(ra) and ra > 0) else np.nan,
        rsi=r["rsi"])
    # FLOW DIVERGENCE: price makes a new adverse extreme -- does cumulative flow
    # make a WORSE one too, or refuse to confirm?
    c0, c1 = cum.get(prev_extreme_mod), cum.get(m)
    out["flow_div"] = (c1 - c0) if (c0 is not None and c1 is not None) else np.nan
    return out


FEATS = ["vol_ratio", "rng_ratio", "rsi", "flow_div"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--window", type=int, default=120)
    a = ap.parse_args()

    import directional_flow_backtester as D
    rows = []
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dirn = rule["ticker"], rule["direction"]
        meta = []
        cand = sim_core.build_candidates(D, rule, meta_out=meta)
        if not cand:
            continue
        B, F = bars(tk), flows(D, tk)
        want_dn = (dirn == "CALL")          # a CALL's adverse move is DOWN
        for (d, m0, _p), mt in zip(cand, meta):
            g, cum = B.get(d), F.get(d, {})
            if g is None or mt.get("spot") is None:
                continue
            spot0 = float(mt["spot"])
            w = [x for x in range(int(m0), min(int(m0) + a.window, CUTOFF) + 1)
                 if x in g.index]
            if len(w) < 20:
                continue
            px = g.loc[w, "low" if want_dn else "high"]
            # running adverse extreme -> every "new worst so far" bar
            run = px.cummin() if want_dn else px.cummax()
            isnew = px.eq(run) & (run != run.shift())
            newbars = [int(x) for x in px.index[isnew]]
            if len(newbars) < 3:
                continue
            trough = int(px.idxmin() if want_dn else px.idxmax())
            depth = abs(float(px.loc[trough]) - spot0) / spot0
            after = [x for x in w if x > trough]
            rec = any((g.loc[x, "high"] >= spot0) if want_dn
                      else (g.loc[x, "low"] <= spot0) for x in after)
            # features at the trough, and at every EARLIER new-extreme bar
            prev = {b: newbars[max(i - 1, 0)] for i, b in enumerate(newbars)}
            ft = feats(g, trough, cum, prev[trough])
            if ft is None:
                continue
            pool = [feats(g, b, cum, prev[b]) for b in newbars if b != trough]
            pool = [p for p in pool if p is not None]
            if not pool:
                continue
            rec_row = dict(rule=rule["name"], ticker=tk, dir=dirn, date=d,
                           recovered=rec, depth=depth, n_new=len(newbars))
            for f in FEATS:
                rec_row[f] = ft[f]
                rec_row[f"auc_{f}"] = midrank_auc(ft[f], [p[f] for p in pool])
            rows.append(rec_row)
        print(f"    {rule['name']} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  nothing"); return
    R.to_parquet("_trough_anatomy.parquet", index=False)
    rec, non = R[R["recovered"]], R[~R["recovered"]]
    print(f"\n  {len(R):,} drawdowns  |  recovered {len(rec):,} "
          f"({len(rec)/len(R)*100:.0f}%)  not {len(non):,}")
    print(f"  median depth  recovered {rec['depth'].median()*100:.2f}%   "
          f"not {non['depth'].median()*100:.2f}%")
    print(f"  median new-extreme bars per drawdown {R['n_new'].median():.0f}")

    print(f"\n{'='*94}")
    print(f"  1. IS THE TROUGH BAR DIFFERENT FROM EARLIER 'NEW LOW' BARS?")
    print(f"     AUC vs bars that were also the worst-so-far. 0.50 = null.")
    print(f"{'='*94}")
    print(f"  {'feature':12} {'ALL':>10} {'recovered':>12} {'not rec':>10} "
          f"{'contrast':>10}")
    for f in FEATS:
        c = f"auc_{f}"
        av, rv, nv = R[c].mean(), rec[c].mean(), non[c].mean()
        print(f"  {f:12} {av:>10.3f} {rv:>12.3f} {nv:>10.3f} {rv-nv:>+10.3f}")
    print(f"\n  The CONTRAST column is the decision-relevant one: a footprint")
    print(f"  that shows up equally in drawdowns that never recovered cannot")
    print(f"  time a scale-in, however extreme it looks.")

    print(f"\n{'='*94}")
    print(f"  2. RAW VALUES AT THE TROUGH (medians)")
    print(f"{'='*94}")
    print(f"  {'feature':12} {'recovered':>12} {'not rec':>12} {'delta':>10}")
    for f in FEATS:
        rv, nv = rec[f].median(), non[f].median()
        print(f"  {f:12} {rv:>12.2f} {nv:>12.2f} {rv-nv:>+10.2f}")

    print(f"\n{'='*94}")
    print(f"  3. FLOW DIVERGENCE, THE HYPOTHESIS SPELLED OUT")
    print(f"{'='*94}")
    print(f"  'price makes a new adverse extreme but flow does not confirm'")
    for lab, s in (("recovered", rec), ("not recovered", non)):
        v = s["flow_div"].dropna()
        if not len(v):
            continue
        # a CALL wants flow to hold UP into the low; a PUT wants it to hold DOWN
        held = np.where(s.loc[v.index, "dir"] == "CALL", v > 0, v < 0)
        print(f"    {lab:14} flow held against the new extreme in "
              f"{held.mean()*100:>5.1f}% of {len(v):,} drawdowns")

    print(f"\n{'='*94}")
    print(f"  4. BY RULE -- is any separation book-wide or one ticker?")
    print(f"{'='*94}")
    print(f"  {'rule':24} {'n':>6} {'rec%':>6} " +
          "".join(f"{f[:9]:>10}" for f in FEATS))
    for nm, g2 in R.groupby("rule"):
        gr, gn = g2[g2["recovered"]], g2[~g2["recovered"]]
        cells = []
        for f in FEATS:
            c = f"auc_{f}"
            cells.append(gr[c].mean() - gn[c].mean()
                         if len(gr) and len(gn) else np.nan)
        print(f"  {nm:24} {len(g2):>6} {g2['recovered'].mean()*100:>5.0f}% " +
              "".join(f"{x:>+10.3f}" if np.isfinite(x) else f"{'--':>10}"
                      for x in cells))


if __name__ == "__main__":
    main()
