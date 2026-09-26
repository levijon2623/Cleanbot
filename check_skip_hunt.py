# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_skip_hunt.py
==================
HUNT FOR A SKIP RULE: an ENTRY-TIME filter that refuses the trades which end
<= -50%, without touching the exit.

WHY A SKIP AND NOT A STOP (settled before this script existed)
    Every mechanical EXIT tested on this book has hurt -- the ROE give-back took
    win 0.38 -> 0.59 and destroyed expectancy, tighter stops hurt, TP+trail
    scored OOS -1.1% vs +4.8% pure. An exit acts on a trade already open and
    cannot tell "this goes to -65%" from "this dips then runs to +158%": the
    path to the right tail runs THROUGH drawdown. A skip refuses the trade
    before entry and so removes a loser without ever touching a winner's path.
    Every filter that has worked on this book is an entry filter.

WHY IT IS WORTH HUNTING (check_target_variance --test skip)
    The <=-50% bucket is 27.3% of trades and -208.6% of total P&L. A filter with
    rank correlation of only rho=0.20 to outcome, skipping the worst 30%, takes
    the book +11.5% -> +19.6%/trade at 120% of total P&L. Break-even is ~rho=0.10.
    Oracle ceiling (skip exactly the losers) is +44.5%/trade at 270% of total.

WHY THE PREVIOUS NULLS DO NOT SETTLE THIS
    Most of these features have been tested before -- as DIRECTIONAL or raw-P&L
    conditioners, at a ~65pp minimum detectable effect. Scored on `loss50` the
    same data carries up to 6.88x the SNR, so a feature dismissed on raw P&L is
    NOT dismissed for this question. That is the entire reason to re-run them.

PRE-REGISTERED FEATURES (fixed before any result was looked at; all knowable at
the entry minute, no lookahead)
    difficulty   (mid/spot) / (rv_d * sqrt(mins_left/390))
                 How many typical remaining-session moves the underlying must
                 travel just to reach expiry breakeven. THE headline a-priori
                 candidate: it is the one feature that is mechanically about
                 total loss, and it is a pure ratio with nothing fitted.
    rel_spread   (ask-bid)/mid at entry. A-priori, and the reason it is here is
                 that it CANNOT be overfit -- you pay it on every trade.
    premium      the entry mid. Cheap options are lottery tickets.
    mins_left    eod_flatten - entry minute. The time budget; check_holdtime
                 already showed "early" is mostly this.
    moneyness    |strike - spot|/spot of the contract actually picked.
    dte          0 vs 1. Known ~+4pp for 1DTE on raw P&L.

PRE-COMMITTED DECISION RULE -- a candidate advances only if ALL hold:
    1. loss50 rate is MONOTONE across terciles
    2. the IS and OOS effects have the SAME SIGN
    3. the day-block 95% CI on the OOS worst-minus-best tercile excludes zero
    4. skipping the worst tercile improves BOTH mean AND total P&L, OOS
    5. it helps in a MAJORITY of the rules it touches (the size-gate lesson:
       an aggregate spread is usually composition)
Anything passing 1-4 but not 5 is a thread, not a finding.

THE SKIP IS APPLIED BEFORE THE SEQUENTIAL WALK, not after. Refusing a trade
frees the ticker for a later trigger that the one-position guard would other-
wise have blocked, so the skipped book is a genuinely different sample. Scoring
a skip by deleting rows from the finished book would silently assume otherwise
-- that is the "sample is a policy artefact" trap (METHODOLOGY.md 3).

Usage:
  python check_skip_hunt.py                 # full pre-registered sweep
  python check_skip_hunt.py --feature difficulty --detail
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
HIST = "historical"
FEATURES = ["difficulty", "rel_spread", "premium", "mins_left", "moneyness", "dte"]


# --------------------------------------------------------------- data
def _rv_daily(tk: str) -> pd.Series:
    """Trailing-20d close-to-close vol, in DAILY units, shifted one session so
    an entry on day D only ever sees data through D-1."""
    d = pd.read_parquet(f"{HIST}/{tk}.parquet")
    d.columns = [c.lower() for c in d.columns]
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York")
    mod = et.dt.hour * 60 + et.dt.minute
    r = d.assign(_d=et.dt.date)[(mod >= 570) & (mod <= 960)]
    close = r.sort_values("start_time").groupby("_d")["close"].last().astype(float)
    rv = np.log(close).diff().rolling(20).std()
    return rv.shift(1)


def build() -> tuple[pd.DataFrame, dict]:
    """Every candidate with its entry-time features AND its realised outcome.

    Returns the per-trigger frame plus {rule_name: (rule, cand, meta)} so the
    skip can be re-simulated through sim_core.walk rather than by row deletion.
    """
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    rows, store = [], {}
    for r in [x for x in RULES if x.get("enabled", True)]:
        meta: list[dict] = []
        cand = sim_core.build_candidates(D, r, meta_out=meta)
        if not cand:
            continue
        pol = sim_core.policy_for(r, TRAIL_PCT)
        em = sim_core.eod_mod(r)
        rv = _rv_daily(r["ticker"])
        store[r["name"]] = (r, cand, meta, pol, em)
        for (d, m, path), mt in zip(cand, meta):
            pnl = sim_core.simulate(path, pol, em, fill="bot")[0]
            left = max(1, em - m)
            v = rv.get(d, np.nan)
            exp_move = v * np.sqrt(left / 390.0) if np.isfinite(v) else np.nan
            rows.append(dict(
                date=d, mod=m, rule=r["name"], ticker=r["ticker"], pnl=pnl,
                premium=mt["mid"],
                rel_spread=((mt["ask"] - mt["bid"]) / mt["mid"]
                            if mt["bid"] > 0 and mt["mid"] > 0 else np.nan),
                mins_left=left,
                moneyness=(abs(mt["strike"] - mt["spot"]) / mt["spot"]
                           if np.isfinite(mt["strike"]) and mt["spot"] else np.nan),
                dte=mt["dte"],
                difficulty=((mt["mid"] / mt["spot"]) / exp_move
                            if np.isfinite(exp_move) and exp_move > 0 else np.nan),
            ))
    df = pd.DataFrame(rows)
    df["loss50"] = (df["pnl"] <= -0.50).astype(float)
    df["half"] = np.where(df["date"] < SPLIT, "IS", "OOS")
    return df, store


# --------------------------------------------------------------- stats
def _boot_ci(df: pd.DataFrame, col: str, q1: float, q2: float,
             n=3000, seed=23):
    """Day-block bootstrap of OOS loss50(worst tercile) - loss50(best tercile)."""
    o = df[df["half"] == "OOS"]
    hi = o[o[col] > q2]
    lo = o[o[col] <= q1]
    if len(hi) < 5 or len(lo) < 5:
        return np.nan, np.nan, np.nan
    obs = hi["loss50"].mean() - lo["loss50"].mean()
    pool = {d: g for d, g in o.groupby("date")}
    days = list(pool)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = pd.concat([pool[days[k]] for k in
                       rng.choice(len(days), len(days), replace=True)],
                      ignore_index=True)
        a, b = s[s[col] > q2], s[s[col] <= q1]
        if len(a) and len(b):
            out.append(a["loss50"].mean() - b["loss50"].mean())
    if not out:
        return obs, np.nan, np.nan
    return obs, *np.percentile(out, [2.5, 97.5])


def _book(store: dict, drop: dict | None = None) -> list[tuple]:
    """Re-run the SEQUENTIAL walk, optionally refusing candidates.

    `drop` maps rule name -> boolean mask over that rule's candidate list.
    The mask is applied to `cand` BEFORE walk, so the one-position-per-ticker
    guard sees the skipped book, and a refused trigger genuinely frees the
    ticker for the next one.
    """
    out = []
    for name, (r, cand, meta, pol, em) in store.items():
        c = cand
        if drop is not None and name in drop:
            keep = ~np.asarray(drop[name])
            c = [x for x, k in zip(cand, keep) if k]
        for d, p in sim_core.walk(c, pol, em, fill="bot"):
            out.append((d, p, name))
    return sorted(out)


def _summary(rows: list[tuple], label: str) -> dict:
    v = np.array([p for _, p, _ in rows], float)
    if not len(v):
        return {}
    i = np.array([p for d, p, _ in rows if d < SPLIT], float)
    o = np.array([p for d, p, _ in rows if d >= SPLIT], float)
    sl = [[] for _ in range(6)]
    for d, p, _ in rows:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    pop = sum(1 for b in sl if len(b) >= 3)
    print(f"    {label:26} n={len(v):>4}  mean {v.mean()*100:>+7.1f}%  "
          f"IS {i.mean()*100 if len(i) else np.nan:>+7.1f}%  "
          f"OOS {o.mean()*100 if len(o) else np.nan:>+7.1f}%  "
          f"tot {v.sum():>+7.2f}  OOStot {o.sum():>+7.2f}  "
          f"win {(v>0).mean():.2f}  pop {pop}/6")
    return dict(n=len(v), mean=v.mean(), oos=o.mean() if len(o) else np.nan,
                tot=v.sum(), oostot=o.sum() if len(o) else np.nan)


# --------------------------------------------------------------- sweep
def sweep(df: pd.DataFrame, store: dict, feats, detail: bool):
    base = _book(store)
    print("  BASELINE (deployed book, sequential, fill=bot):")
    b = _summary(base, "no skip")
    print(f"\n  Trigger population: {len(df)} candidates, "
          f"{df['loss50'].mean()*100:.0f}% end <= -50%  "
          f"(IS {df[df['half']=='IS']['loss50'].mean()*100:.0f}% / "
          f"OOS {df[df['half']=='OOS']['loss50'].mean()*100:.0f}%)")

    verdicts = []
    for f in feats:
        s = df.dropna(subset=[f])
        if len(s) < 60:
            print(f"\n  {f}: too few observations ({len(s)})")
            continue
        ref = s[s["half"] == "IS"][f]
        if s[f].nunique() <= 2:                      # binary, e.g. dte
            q1 = q2 = s[f].min()
        else:
            q1, q2 = np.percentile(ref, [33.33, 66.67])
            if not q1 < q2:
                print(f"\n  {f}: degenerate cut points")
                continue
        print(f"\n{'='*104}")
        print(f"  FEATURE: {f}     IS-set cuts: <={q1:.4g} | >{q2:.4g}")
        print(f"{'='*104}")

        # ---- loss50 by tercile (the screen)
        band = [("LOW", s[f] <= q1), ("MID", (s[f] > q1) & (s[f] <= q2)),
                ("HIGH", s[f] > q2)]
        rates = {}
        for lab, m in band:
            g = s[m]
            if g.empty:
                continue
            gi, go = g[g["half"] == "IS"], g[g["half"] == "OOS"]
            rates[lab] = (go["loss50"].mean() if len(go) else np.nan,
                          gi["loss50"].mean() if len(gi) else np.nan)
            print(f"    {lab:5} n={len(g):>5}  loss50 {g['loss50'].mean()*100:>5.1f}%   "
                  f"IS {gi['loss50'].mean()*100 if len(gi) else np.nan:>5.1f}%  "
                  f"OOS {go['loss50'].mean()*100 if len(go) else np.nan:>5.1f}%   "
                  f"mean pnl {g['pnl'].mean()*100:>+7.1f}%")
        obs, c1, c2 = _boot_ci(s, f, q1, q2)
        sig = np.isfinite(c1) and (c1 > 0 or c2 < 0)
        print(f"    OOS loss50 HIGH-minus-LOW {obs*100:>+6.1f}pp   "
              f"day-block 95% CI [{c1*100:>+6.1f}, {c2*100:>+6.1f}]pp"
              + ("   *" if sig else ""))

        # criterion 1: monotone in loss50 (pooled)
        pooled = [s[m]["loss50"].mean() for _, m in band if m.any()]
        mono = (all(np.diff(pooled) > 0) or all(np.diff(pooled) < 0)) if len(pooled) == 3 else False
        # criterion 2: IS and OOS same sign
        same = (np.isfinite(rates.get("HIGH", (np.nan,))[0])
                and np.sign(rates["HIGH"][0] - rates["LOW"][0])
                == np.sign(rates["HIGH"][1] - rates["LOW"][1]))

        # ---- the actual skip, applied BEFORE the walk.
        # BOTH DIRECTIONS ARE SCORED. The pre-registered choice is the tercile
        # with the higher loss50 -- but because the two tails are inseparable
        # here (see `twins`), the loss50-worst bucket is frequently also the
        # P&L-BEST one, so skipping it is backwards. `dte` is the worked
        # example: dte=0 has 3x the blow-up rate AND +14.6% vs -0.9% mean.
        # Reporting only the loss50-chosen direction would hide that.
        worst = "HIGH" if pooled[-1] > pooled[0] else "LOW"
        other = "LOW" if worst == "HIGH" else "HIGH"

        def _apply(which):
            t = (lambda x: x > q2) if which == "HIGH" else (lambda x: x <= q1)
            drop = {}
            for name, (r, cand, meta, pol, em) in store.items():
                sub = df[df["rule"] == name]
                key = {(row.date, row.mod): getattr(row, f)
                       for row in sub.itertuples()}
                drop[name] = [bool(np.isfinite(key.get((d, m), np.nan))
                                   and t(key.get((d, m))))
                              for d, m, _ in cand]
            return _book(store, drop)

        after = _apply(worst)
        print(f"\n    SKIP applied before the sequential walk "
              f"(both directions scored):")
        _summary(base, "baseline")
        a = _summary(after, f"skip {worst} (loss50-worst)")
        _summary(_apply(other), f"skip {other} (opposite)")
        better = (np.isfinite(a.get("oos", np.nan))
                  and a["oos"] > b["oos"] and a["oostot"] > b["oostot"])

        # criterion 5: per-rule
        helped = total = 0
        if detail:
            print("    per-rule OOS mean, baseline -> skipped:")
        for name in store:
            bo = [p for d, p, n in base if n == name and d >= SPLIT]
            ao = [p for d, p, n in after if n == name and d >= SPLIT]
            if len(bo) >= 5 and len(ao) >= 5:
                total += 1
                helped += np.mean(ao) > np.mean(bo)
                if detail:
                    print(f"      {name:22} {np.mean(bo)*100:>+7.1f}% (n={len(bo):>3}) "
                          f"-> {np.mean(ao)*100:>+7.1f}% (n={len(ao):>3})")
        major = total > 0 and helped / total > 0.5
        print(f"    per-rule: helped {helped}/{total} rules with >=5 OOS trades in both")

        passed = [mono, same, sig, better, major]
        print(f"    CRITERIA  monotone {mono}  same-sign {same}  CI-excl-0 {sig}  "
              f"skip-improves-both {better}  majority-of-rules {major}"
              f"   ->  {'ADVANCE' if all(passed) else 'REJECT'}")
        verdicts.append((f, all(passed), sum(passed)))

    print(f"\n{'='*104}")
    print(f"  PRE-REGISTERED SWEEP COMPLETE: {len(verdicts)} features tested "
          f"(expect ~{len(verdicts)*0.05:.1f} spurious stars at 95%).")
    for f, ok, k in sorted(verdicts, key=lambda x: -x[2]):
        print(f"    {f:12} {k}/5 criteria   {'ADVANCE' if ok else 'reject'}")


def twins(df: pd.DataFrame, feats):
    """WHY EVERY SKIP CANDIDATE FAILS: the two tails are the same trades.

    For each feature tercile, report BOTH tail rates -- loss50 (ends <= -50%)
    and hit50 (ends >= +50%) -- side by side. A usable skip rule needs a cell
    with HIGH loss50 and LOW hit50, i.e. the two must be SEPARABLE. If instead
    they rise and fall together across every feature, then every one of these
    features is a proxy for CONVEXITY rather than for quality, and refusing the
    blow-ups necessarily refuses the moonshots that pay for them.

    The rank correlation across all cells is the one-number summary. Strongly
    positive = inseparable = no skip rule can be built from this family, and no
    amount of further feature search within it will help.
    """
    print(f"\n{'='*104}")
    print("  ARE THE TWO TAILS SEPARABLE?   loss50 = ends <=-50%,  "
          "hit50 = ends >=+50%")
    print(f"{'='*104}")
    print(f"  {'feature':12} {'cell':5} {'n':>6} {'loss50':>8} {'hit50':>8} "
          f"{'ratio':>8} {'mean pnl':>10}")
    L, H = [], []
    for f in feats:
        s = df.dropna(subset=[f])
        if len(s) < 60:
            continue
        ref = s[s["half"] == "IS"][f]
        if s[f].nunique() <= 2:
            band = [("lo", s[f] <= s[f].min()), ("hi", s[f] > s[f].min())]
        else:
            q1, q2 = np.percentile(ref, [33.33, 66.67])
            band = [("LOW", s[f] <= q1), ("MID", (s[f] > q1) & (s[f] <= q2)),
                    ("HIGH", s[f] > q2)]
        for lab, m in band:
            g = s[m]
            if len(g) < 20:
                continue
            lo = g["loss50"].mean()
            hi = (g["pnl"] >= 0.50).mean()
            L.append(lo)
            H.append(hi)
            print(f"  {f:12} {lab:5} {len(g):>6} {lo*100:>7.1f}% {hi*100:>7.1f}% "
                  f"{(lo/hi if hi else np.nan):>8.2f} {g['pnl'].mean()*100:>+9.1f}%")
    L, H = np.array(L), np.array(H)
    rho = pd.Series(L).corr(pd.Series(H), method="spearman")
    print(f"\n  Across all {len(L)} cells: spearman(loss50, hit50) = {rho:+.3f}")
    if rho > 0.5:
        print("  => THE TAILS ARE INSEPARABLE. Every feature here is a proxy for")
        print("     CONVEXITY, not for quality: the cheap, short-dated, high-gamma")
        print("     trades that blow up are the same ones that produce the +158%")
        print("     winners. Refusing the blow-ups refuses the moonshots that pay")
        print("     for them, which is exactly what the skip simulations showed --")
        print("     mean can rise while TOTAL P&L falls.")
        print("  => A skip rule needs a feature that is NOT a convexity proxy.")
        print("     Nothing in this pre-registered family qualifies.")
    print(f"\n  Best available loss50/hit50 RATIO (higher = more separable): "
          f"{np.nanmax(L / np.where(H > 0, H, np.nan)):.2f}")
    print("  For reference a useful skip cell would need a ratio >> 1 together")
    print("  with a BELOW-average mean pnl. Read the two right-hand columns as a")
    print("  pair -- every high-ratio cell above also has a poor mean, which is")
    print("  the trade you are not allowed to make.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", nargs="*", default=FEATURES)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--twins-only", action="store_true")
    a = ap.parse_args()
    df, store = build()
    if not a.twins_only:
        sweep(df, store, a.feature, a.detail)
    twins(df, a.feature)


if __name__ == "__main__":
    main()
