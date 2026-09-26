# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_arrival.py
================
HOW FAST DOES THE EDGE ARRIVE? MINUTE-BY-MINUTE FROM THE TRIGGER.

THE OBSERVATION
    Watching IWM in the flow viewer, price appears to respond to the trigger
    almost immediately -- unlike the other CALL rules. If that is real it
    matters a lot: it would mean the edge is front-loaded, which bears directly
    on hold time, on the dead zone (a 50% trail needs peak ROE > +100% before it
    protects anything), and on whether a fast exit is even worth testing.

🚨 THE CONTROL THAT DECIDES IT: IWM MOVES ALL DAY
    "Price moved 12bp in five minutes after the trigger" is not a finding until
    you know what IWM does in ANY five minutes. So every real measurement is
    paired with a PLACEBO drawn from the SAME SESSION and the SAME HOUR, at
    least 30 minutes clear of the trigger, 20 draws per trade. The hour match
    matters because volatility has a strong intraday shape and the rules do not
    trade all hours equally -- an unmatched placebo would flatter any rule whose
    triggers cluster in the busy part of the session.

🚨 AND THE ONE THAT MAKES IT COMPARABLE: DOLLARS ARE NOT COMPARABLE
    IWM trades near $200, SPY near $600. A $0.15 move is not the same event in
    both. Everything is quoted in BASIS POINTS of the entry spot, and the
    headline statistic is the RESPONSE RATIO -- the median signed move divided
    by the median ABSOLUTE placebo move at the same horizon. That is unitless,
    so "IWM responds faster than the other CALL rules" becomes a claim you can
    actually read off one column.

POPULATION
    The trades the sequential walk actually TOOK, not every gated candidate.
    Two reasons: it is the book, and candidates inside a session can fire
    minutes apart, so their 5-minute forward windows would overlap and
    double-count the same move (METHODOLOGY 2a). The guard makes the taken
    trades non-overlapping by construction.

OPTION ROE
    "Net if you exited at +k" is produced by handing sim_core.simulate a `sig`
    array, NOT by re-deriving exit prices here. That routes the exit through the
    same _out() as every other exit -- entry at min(mid+0.01, ask), the measured
    FILL_COST on a non-adverse tag, COMMISSION_PCT -- so these numbers are
    directly comparable with every other study in the repo. A signal exit is
    ranked AFTER the protective levels, so a trade the trail would have closed
    first reports the trail's exit and is counted as pre-empted.

Usage:
  python check_arrival.py --paper
  python check_arrival.py --paper --tickers IWM
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import check_reentry as RE          # verified RTH bar loader
import sim_core

KS = [1, 2, 3, 4, 5, 10, 20, 30]
N_PLACEBO = 20
GAP = 30                            # placebo must clear the trigger by 30m
SEED = 20260920


def signed_bp(px, spot0, direction):
    """Move in the direction the trade wants, in basis points of entry spot."""
    d = (px - spot0) if direction == "CALL" else (spot0 - px)
    return d / spot0 * 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--min-gap", type=int, default=0,
                    help="drop trades with another GATE-PASSING trigger within "
                         "N minutes after entry -- see section 0")
    a = ap.parse_args()

    import directional_flow_backtester as D
    rules = sim_core.research_rules(include_paper=a.paper)
    if a.tickers:
        rules = [r for r in rules if r["ticker"] in set(a.tickers)]

    rows = []
    for rule in rules:
        tk, dirn = rule["ticker"], rule["direction"]
        meta = []
        cand = sim_core.build_candidates(D, rule, meta_out=meta)
        if not cand:
            continue
        B = RE.bars(tk)
        pol = sim_core.policy_for(rule)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        picks = []
        taken = sim_core.walk(cand, pol, eod, fill=a.fill, cush_cap=cap,
                              with_tags=True, picks_out=picks)
        rng = np.random.default_rng(SEED)
        # Every GATE-PASSING trigger, taken or not. A second trigger landing
        # inside the measurement horizon contaminates it: the move being
        # attributed to trigger #1 is partly trigger #2's, and the two windows
        # overlap (METHODOLOGY 2a). The sequential guard suppresses the TRADE,
        # not the SIGNAL, so this has to be measured separately.
        cand_by_day = {}
        for (cd, cm, _cp) in cand:
            cand_by_day.setdefault(cd, []).append(int(cm))
        for cd in cand_by_day:
            cand_by_day[cd].sort()

        for (d, pnl, tag), (ci, xm) in zip(taken, picks):
            _d, m0, path = cand[ci]
            spot0 = meta[ci].get("spot")
            g = B.get(d)
            if spot0 is None or g is None:
                continue
            spot0 = float(spot0)
            mods = path[7]
            # candidate placebo minutes: same session, same hour, clear of
            # the trigger by GAP minutes on both sides
            pool = [int(x) for x in g.index
                    if x // 60 == m0 // 60 and abs(x - m0) >= GAP]
            picks_p = (rng.choice(pool, size=min(N_PLACEBO, len(pool)),
                                  replace=False) if pool else [])

            nxt = [x for x in cand_by_day.get(d, []) if x > m0]
            gap = (nxt[0] - m0) if nxt else 10_000
            rec = dict(rule=rule["name"], ticker=tk, dir=dirn, date=d,
                       final=pnl * 100, final_tag=tag, gap=gap)
            for k in KS:
                # --- underlying, real ---
                mk = m0 + k
                rec[f"u{k}"] = (signed_bp(float(g.loc[mk, "close"]), spot0, dirn)
                                if mk in g.index else np.nan)
                # --- underlying, placebo (same session, same hour) ---
                vals = []
                for p0 in picks_p:
                    pk = int(p0) + k
                    if p0 in g.index and pk in g.index:
                        vals.append(signed_bp(float(g.loc[pk, "close"]),
                                              float(g.loc[p0, "close"]), dirn))
                rec[f"p{k}"] = np.median(vals) if vals else np.nan
                rec[f"pa{k}"] = np.median(np.abs(vals)) if vals else np.nan
                # --- option, net ROE if exited at +k (through simulate) ---
                if k - 1 < len(mods):
                    sig = np.zeros(len(mods), bool)
                    sig[k - 1] = True
                    p_, _x, tg = sim_core.simulate(path, pol, eod, fill=a.fill,
                                                   cush_cap=cap, sig=sig)
                    rec[f"o{k}"] = p_ * 100
                    rec[f"t{k}"] = tg
                else:
                    rec[f"o{k}"], rec[f"t{k}"] = np.nan, None
            rows.append(rec)
        print(f"    {rule['name']} done  ({len(taken)} trades)", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  nothing"); return
    R.to_parquet("_arrival.parquet", index=False)

    print(f"\n{'='*104}")
    print(f"  0. TRIGGER CLUSTERING -- how often does a SECOND gate-passing "
          f"trigger land in the window?")
    print(f"     If it does, the move credited to trigger #1 is partly #2's and "
          f"the windows overlap.")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'n':>5} {'med gap':>9} {'<=5m':>8} {'<=10m':>8} "
          f"{'<=30m':>8} {'none':>8}")
    for nm, g in R.groupby("rule", sort=False):
        gp = g["gap"]
        print(f"  {nm:24} {len(g):>5} "
              f"{(gp[gp < 10_000].median() if (gp < 10_000).any() else np.nan):>9.0f} "
              f"{(gp <= 5).mean()*100:>7.0f}% {(gp <= 10).mean()*100:>7.0f}% "
              f"{(gp <= 30).mean()*100:>7.0f}% {(gp >= 10_000).mean()*100:>7.0f}%")
    if a.min_gap:
        before = len(R)
        R = R[R["gap"] > a.min_gap]
        print(f"\n  --min-gap {a.min_gap}: kept {len(R)} of {before} trades "
              f"({len(R)/max(before,1)*100:.0f}%). Everything below is the "
              f"CLEAN subset.")
    else:
        print(f"\n  No filter applied. Re-run with --min-gap 10 to see the "
              f"uncontaminated subset.")
    if R.empty:
        print("  nothing left after the gap filter"); return

    def block(title, note, col, fmt="{:>+9.2f}"):
        print(f"\n{'='*104}")
        print(f"  {title}")
        if note:
            print(f"  {note}")
        print(f"{'='*104}")
        print(f"  {'rule':24} {'n':>5} " + "".join(f"{'+'+str(k)+'m':>9}"
                                                   for k in KS))
        for nm, g in R.groupby("rule", sort=False):
            cells = "".join(fmt.format(g[col(k)].median())
                            if g[col(k)].notna().any() else f"{'--':>9}"
                            for k in KS)
            print(f"  {nm:24} {len(g):>5} {cells}")

    block("1. UNDERLYING RESPONSE -- median signed move from the trigger, "
          "in bp of spot",
          "signed toward the trade: up for a CALL, down for a PUT",
          lambda k: f"u{k}")

    block("2. THE PLACEBO -- same session, same hour, >=30m clear of the "
          "trigger, 20 draws",
          "this is what the ticker does in ANY window of that length. "
          "It should sit near zero.",
          lambda k: f"p{k}")

    print(f"\n{'='*104}")
    print(f"  3. RESPONSE RATIO = median signed move / median ABSOLUTE placebo "
          f"move")
    print(f"     Unitless, so a $200 ticker and a $600 one are finally "
          f"comparable.")
    print(f"     1.0 means the post-trigger move is one typical move of that "
          f"horizon; 0.0 is null.")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'n':>5} " + "".join(f"{'+'+str(k)+'m':>9}" for k in KS))
    for nm, g in R.groupby("rule", sort=False):
        cells = ""
        for k in KS:
            num, den = g[f"u{k}"].median(), g[f"pa{k}"].median()
            cells += (f"{num/den:>+9.2f}" if np.isfinite(num) and
                      np.isfinite(den) and den > 0 else f"{'--':>9}")
        print(f"  {nm:24} {len(g):>5} {cells}")

    block("4. OPTION ROE IF YOU EXITED AT +k MINUTES (net, through simulate)",
          "median ROE%. The `final` column is what the deployed exit actually "
          "booked.",
          lambda k: f"o{k}", "{:>+9.1f}")
    print(f"  {'-'*100}")
    print(f"  {'rule':24} {'n':>5} {'final booked ROE (median)':>40}")
    for nm, g in R.groupby("rule", sort=False):
        print(f"  {nm:24} {len(g):>5} {g['final'].median():>+40.1f}")

    print(f"\n{'='*104}")
    print(f"  5. WAS THE TIMED EXIT PRE-EMPTED? -- % of trades the trail/stop "
          f"closed before +k")
    print(f"     A signal exit is ranked AFTER the protective levels, so these "
          f"are trades where")
    print(f"     the question 'what if you held to +k' does not arise.")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'n':>5} " + "".join(f"{'+'+str(k)+'m':>9}" for k in KS))
    for nm, g in R.groupby("rule", sort=False):
        cells = ""
        for k in KS:
            c = g[f"t{k}"]
            pre = (c.notna() & (c != "sig")).mean() * 100
            cells += f"{pre:>8.0f}%"
        print(f"  {nm:24} {len(g):>5} {cells}")

    print(f"\n{'='*104}")
    print(f"  6. THE CLAIM: IS IWM DIFFERENT FROM THE OTHER CALL RULES?")
    print(f"{'='*104}")
    calls = R[R["dir"] == "CALL"]
    if calls.empty:
        print("  no CALL rules in this run"); return
    iwm = calls[calls["ticker"] == "IWM"]
    oth = calls[calls["ticker"] != "IWM"]
    print(f"  {'group':24} {'n':>5} " + "".join(f"{'+'+str(k)+'m':>9}" for k in KS))
    for lab, g in (("IWM", iwm), ("other CALL rules", oth),
                   ("all PUT rules", R[R["dir"] == "PUT"])):
        if g.empty:
            continue
        cells = ""
        for k in KS:
            num, den = g[f"u{k}"].median(), g[f"pa{k}"].median()
            cells += (f"{num/den:>+9.2f}" if np.isfinite(num) and
                      np.isfinite(den) and den > 0 else f"{'--':>9}")
        print(f"  {lab:24} {len(g):>5} {cells}")
    print(f"\n  (response ratio again -- median signed move / median absolute "
          f"placebo move)")


if __name__ == "__main__":
    main()
