# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_multiplier.py
===================
SCALE IN ON A VALIDATED RECLAIM -- A SECOND CONCURRENT SLOT, NOT A REPLACEMENT.

THE ROAD HERE
    check_exit_bracketology: the exit cannot be improved. 18 policies, and
    walk-forward selection LOST to plain trail50 (+3,334 vs +4,374); everything
    that protected the dead zone also capped the tail, which is the whole edge.

    check_reentry: re-entering a stopped-out trade on a FRESH ATM contract is a
    genuinely good trade -- 61 re-entries, +824 total, average winner +142.9
    against the book's +97.6 -- and it STILL loses, because it consumes the one
    position slot. Each re-entry displaced a fresh trigger worth ~+24 while
    earning ~+13.5. A good trade and a worse allocation.

    So the question is no longer whether the reclaim is tradeable. It is whether
    it deserves EXTRA capital rather than REDIRECTED capital. Leg one is left
    alone on its own trail; a validated reclaim opens a SECOND slot.

WHAT check_trough_anatomy SAYS ABOUT WHY THIS COULD STILL FAIL
    The capitulation footprint is real but carries no information about
    recovery: volume AUC 0.606 at the trough, yet the recovered-minus-not
    contrast is -0.021 -- marginally STRONGER in drawdowns that never came back.
    Flow divergence is dead flat at 0.500. And 81% of drawdowns reclaim the
    entry within 120m anyway, so "it came back" is the default, not a signal.
    Any edge here cannot come from PREDICTING the reclaim. It has to come from
    the scale-in being good exposure on a day already proven to move.

🚨 TOTAL P&L IS NOT A FAIR COMPARISON AND IS NOT THE HEADLINE
    The sizer targets a fixed PREMIUM per position, so two concurrent legs
    commit twice the capital. Summing ROE across them flatters the Multiplier by
    exactly the amount of extra money at risk -- the same way any leveraged book
    beats an unleveraged one on gross P&L. Three denominators are reported:
        max drawdown       the risk the spec explicitly asked about
        position-minutes   capital actually deployed, concurrency included
        per-trade ROE      unaffected by how many slots are open
    and the pass/fail criterion is return per unit of drawdown, not return.

DESIGN
    slot 1   ordinary triggers, sequential, exactly as deployed
    slot 2   scale-ins ONLY. A fresh trigger can never take it, so the test
             measures the scale-in rather than general two-at-a-time trading
    trigger  leg one is >= --dd underwater, and the underlying then reclaims the
             entry spot WHILE LEG ONE IS STILL OPEN, passing the gates validated
             in check_reentry: (volume > 1.5x trailing 30m OR range in the
             trailing top quartile) AND flow not having turned against the trade
    leg two  a FRESH ATM contract at the reclaim, independent 50% trail

    --dd is measured on the OPTION by default. The spec said ">15% on the
    underlying", but check_trough_anatomy showed that move does not exist in
    these names: underlying drawdown p50 0.40%, p90 1.42%, max 9.40%, and 0 of
    1,722 drawdowns exceed 15%. --dd-basis underlying is kept for completeness.

PRE-COMMITTED CRITERIA -- fixed before the first run
    M1  Multiplier total ROE > baseline
    M2  return per unit of max drawdown improves   <-- the one that decides it
    M3  >= 6 of 9 rules improve
    M4  the scale-in leg is profitable standalone
    M5  scale-ins are selective: they fire on < 50% of eligible drawdowns

⚠️ PREDATES THE 2026-09-20 SPOT FIX (sim_core.py, the `bbd` sort). Baseline and
Multiplier share the same candidate build, so both arms carry the same stale
strike and the comparison holds -- but leg two is a FRESH ATM pick via
RE.atm_path, which takes its spot from the underlying's own bars and was always
correct. That asymmetry means leg two was, if anything, better-specified than
leg one here. Re-baseline before citing the absolute figures.

RESULT -- 2026-09-19, 151 sessions, 349 baseline trades. REJECTED.
    variant                      scale-ins   leg2 P&L   ret/DD   M2
    baseline                             --         --     4.60   --
    open    (the literal spec)           96        +39     4.31   FAIL
    stopped (fixed) --dd 0.10            53       +193     4.37   FAIL
    stopped (fixed) --dd 0.15            53       +193     4.37   FAIL
    stopped (fixed) --dd 0.25            52       +230     4.41   FAIL
    stopped (fixed) --dd 0.40            50       +120     4.29   FAIL
    any                                 121     +1,100     5.77   pass*

    THE SPEC AS WRITTEN IS A CLEAN NULL, not an underpowered one. 96 scale-ins
    returned +39 ROE in total for 1.28x the capital, and max drawdown got WORSE
    (-17.3% -> -18.7%). The scale-in leg's average winner was +74.1 against the
    book's +97.6 -- it is not a weaker version of the book's trade, it is a
    different and worse one. Max drawdown lands at -19.1% in every fixed
    `stopped` variant regardless of --dd, which is inert there because a leg
    that trail-stops out has always been >15% underwater first.

    * `any` passes all five criteria and should still be rejected: per-trade
    expectancy +9.1 vs the book's +11.4, and 1.33x the capital for 1.28x the
    return. See METHODOLOGY 7, "ret/DD can be bought with diversification".

    An earlier `stopped` run scored +1,158 / ret/DD 6.06 / 4-of-5 PASS. That was
    a bug -- the reclaim search started at the drawdown minute and rejected
    pre-exit hits, which quietly also filtered out every day that reclaimed
    while leg one was open. Written up in METHODOLOGY 7 as its own trap.

WHAT WOULD CHANGE THE ANSWER
    Nothing available now. The bootstrap on the (contaminated) best case already
    gave leg2 - leg1 expectancy +22.6 ROE/trade, 95% CI [-22.3, +85.5] on 27
    sessions. Power is set by DAYS. Halving that CI needs ~4x the sessions.

Usage:
  python check_multiplier.py --paper
  python check_multiplier.py --paper --when stopped --dd 0.25
  python check_multiplier.py --paper --dd 0.02 --dd-basis underlying
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import check_reentry as RE          # bars / flows / find_reentry / atm_path
import sim_core

TRAIL = dict(name="trail50", kind="trail", trail=0.50)
UNIT = 0.02                         # premium committed per position, of notional


def drawdown_minute(path, dd, basis, B, d, dirn, spot0):
    """First minute leg one is >= `dd` underwater. -> minute-of-day or None.

    On the OPTION basis this reads the trade CLOSE, not the bid. The bid goes to
    zero on missing quotes -- that is exactly the artifact that produced the GLD
    penny exits -- and a zero bid would register as an instant -100% drawdown on
    every one of them. The close is also what the bot's own trail watches, so
    the gate and the exit agree on what "down" means.
    """
    mid, _ask0, cl, _hi, _lo, _bid, _ask, mods = path
    if basis == "option":
        hit = np.where(np.isfinite(cl) & (cl / mid - 1.0 <= -dd))[0]
        return int(mods[hit[0]]) if len(hit) else None
    g = B.get(d)
    if g is None:
        return None
    want_dn = (dirn == "CALL")       # a CALL's adverse move is DOWN
    for m in mods:
        if m not in g.index:
            continue
        p = float(g.loc[m, "low" if want_dn else "high"])
        moved = (spot0 - p) if want_dn else (p - spot0)
        if moved / spot0 >= dd:
            return int(m)
    return None


def curve(rows):
    """Daily P&L in units of notional -> (final return, max drawdown)."""
    if not rows:
        return 0.0, 0.0
    df = pd.DataFrame(rows)
    daily = df.groupby("date")["pnl"].sum().sort_index() / 100.0 * UNIT
    eq = daily.cumsum()
    return float(eq.iloc[-1]), float((eq - eq.cummax()).min())


def stat(rows):
    if not rows:
        return dict(n=0, tot=0.0, med=np.nan, avgwin=np.nan, p95=np.nan,
                    win=np.nan, days=0, mins=0)
    v = np.array([r["pnl"] for r in rows], float)
    w = v[v > 0]
    return dict(n=len(v), tot=float(v.sum()), med=float(np.median(v)),
                avgwin=float(w.mean()) if len(w) else 0.0,
                p95=float(np.percentile(v, 95)),
                win=float((v > 0).mean() * 100),
                days=len({r["date"] for r in rows}),
                mins=int(sum(r["hold"] for r in rows)))


HDR = (f"  {'model':26} {'n':>5} {'days':>5} {'total':>9} {'medROE':>8} "
       f"{'avg win':>9} {'p95':>9} {'win':>7}")


def line(s, lab):
    if not s["n"]:
        return f"  {lab:26} {'(none)':>5}"
    return (f"  {lab:26} {s['n']:>5} {s['days']:>5} {s['tot']:>+9.0f} "
            f"{s['med']:>+8.1f} {s['avgwin']:>+9.1f} {s['p95']:>+9.1f} "
            f"{s['win']:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--dd", type=float, default=0.15)
    ap.add_argument("--dd-basis", choices=("option", "underlying"),
                    default="option")
    ap.add_argument("--flow-gate", choices=("relative", "sign"),
                    default="relative")
    ap.add_argument("--when", choices=("open", "stopped", "any"),
                    default="open",
                    help="open: reclaim while leg one still holds (a true "
                         "scale-in). stopped: reclaim after leg one exited -- "
                         "check_reentry's own population, but taken in slot 2 "
                         "so it displaces nothing. any: both.")
    a = ap.parse_args()

    import directional_flow_backtester as D
    base, leg1, leg2 = [], [], []
    per_rule = {}
    n_elig = n_fired = n_late = n_busy = n_nogate = n_nopath = 0

    for rule in sim_core.research_rules(include_paper=a.paper):
        tk, dirn = rule["ticker"], rule["direction"]
        meta = []
        cand = sim_core.build_candidates(D, rule, meta_out=meta)
        if not cand:
            continue
        B, F = RE.bars(tk), RE.flows(D, tk)
        tb = D._ticker_bars(tk)
        bbc = {c: g.sort_values("minute_et")
               for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        per_rule[rule["name"]] = {"base": [], "mult": []}

        # `busy` is a MINUTE-OF-DAY and MUST reset each session -- carrying it
        # across days silently drops every candidate earlier in the clock than
        # the previous day's exit (it cut check_reentry's baseline 349 -> 30).
        # `slot2` is the same thing for the scale-in slot.
        busy, slot2, busy_day = -1, -1, None
        for (d, m, path), mt in zip(cand, meta):
            if d != busy_day:
                busy, slot2, busy_day = -1, -1, d
            if m < busy:
                continue
            pnl, xm, tag = sim_core.simulate(path, TRAIL, eod, fill=a.fill,
                                             cush_cap=cap)
            r1 = dict(ticker=tk, rule=rule["name"], date=d, pnl=pnl * 100,
                      leg=1, tag=tag, hold=max(int(xm) - int(m), 0))
            busy = int(xm)
            base.append(r1)
            leg1.append(r1)
            per_rule[rule["name"]]["base"].append(r1)
            per_rule[rule["name"]]["mult"].append(r1)

            spot0 = mt.get("spot")
            if spot0 is None:
                continue
            ddm = drawdown_minute(path, a.dd, a.dd_basis, B, d, dirn,
                                  float(spot0))
            if ddm is None:
                continue
            n_elig += 1
            cum0 = F.get(d, {}).get(int(m))
            # --when stopped searches from leg one's EXIT, not from the drawdown
            # minute. Searching from the drawdown and rejecting anything before
            # the exit is not the same thing: find_reentry returns the FIRST
            # qualifying reclaim, so an early one that leg one rode through
            # would consume the search and hide the post-exit reclaim entirely.
            # That silently suppressed scale-ins in proportion to how low --dd
            # was set, which is the opposite of what --dd is supposed to do.
            start = int(xm) if a.when == "stopped" else int(ddm)
            rm, _why = RE.find_reentry(B, F, d, dirn, float(spot0), start, cum0,
                                       a.flow_gate)
            if rm is None:
                n_nogate += 1
                continue
            if (rm > xm) != (a.when == "stopped") and a.when != "any":
                # --when open    keeps only reclaims arriving while leg one is
                #                still alive -- a true scale-in.
                # --when stopped keeps only reclaims arriving after leg one was
                #                stopped out. That is check_reentry's own
                #                population; the difference here is that it goes
                #                into slot 2, so it displaces no fresh trigger.
                n_late += 1
                continue
            if rm < slot2:
                n_busy += 1
                continue
            spot_rm = B[d].loc[rm, "close"] if rm in B[d].index else None
            if spot_rm is None:
                n_nopath += 1
                continue
            p2 = RE.atm_path(D, bbc, bbd, d,
                             pd.Timestamp(d) + pd.Timedelta(minutes=int(rm)),
                             dirn, rule.get("dte", [0, 1]), float(spot_rm))
            if p2 is None:
                n_nopath += 1
                continue
            pnl2, xm2, tag2 = sim_core.simulate(p2, TRAIL, eod, fill=a.fill,
                                                cush_cap=cap)
            r2 = dict(ticker=tk, rule=rule["name"], date=d, pnl=pnl2 * 100,
                      leg=2, tag=tag2, hold=max(int(xm2) - int(rm), 0))
            leg2.append(r2)
            per_rule[rule["name"]]["mult"].append(r2)
            slot2 = int(xm2)
            n_fired += 1
        print(f"    {rule['name']} done", flush=True)

    if not base:
        print("  no candidates"); return
    pd.DataFrame(leg1 + leg2).to_parquet("_multiplier.parquet", index=False)

    A, M, L2 = stat(base), stat(leg1 + leg2), stat(leg2)
    ra, da = curve(base)
    rm_, dm = curve(leg1 + leg2)
    ca = ra / abs(da) if da else np.nan
    cm = rm_ / abs(dm) if dm else np.nan

    print(f"\n{'='*104}")
    print(f"  1. BASELINE vs MULTIPLIER   (scale-in at >= {a.dd:.0%} on the "
          f"{a.dd_basis}, reclaim '{a.when}', flow gate '{a.flow_gate}')")
    print(f"{'='*104}")
    print(HDR)
    print(line(A, "baseline (1 slot)"))
    print(line(M, "multiplier (2 slots)"))
    print(line(L2, "  the scale-in leg alone"))

    print(f"\n{'='*104}")
    print(f"  2. THE THREE DENOMINATORS -- is the extra return just extra money?")
    print(f"{'='*104}")
    print(f"  {'model':26} {'return':>9} {'max DD':>9} {'ret/DD':>8} "
          f"{'pos-min':>10} {'ROE/1k min':>12}")
    for lab, s, r, dd_ in (("baseline", A, ra, da), ("multiplier", M, rm_, dm)):
        c = r / abs(dd_) if dd_ else np.nan
        pm = (s["tot"] / s["mins"] * 1000) if s["mins"] else np.nan
        print(f"  {lab:26} {r*100:>+8.1f}% {dd_*100:>+8.1f}% {c:>8.2f} "
              f"{s['mins']:>10,} {pm:>+12.1f}")
    if A["mins"] and A["tot"]:
        print(f"\n  The Multiplier deploys {M['mins']/A['mins']:.2f}x the "
              f"position-minutes for {M['tot']/A['tot']:.2f}x the return.")
        print(f"  A return multiple below the capital multiple means the "
              f"scale-in earns LESS per dollar")
        print(f"  than the book it was bolted onto, whatever the gross total "
              f"says.")

    print(f"\n{'='*104}")
    print(f"  3. M5 -- HOW SELECTIVE IS THE SCALE-IN?")
    print(f"{'='*104}")
    print(f"  leg-one trades reaching {a.dd:.0%} underwater        {n_elig:>6}")
    print(f"    no qualifying reclaim before 15:00              {n_nogate:>6}")
    print(f"    reclaim on the wrong side of leg one's exit     {n_late:>6}")
    print(f"    scale-in slot already occupied                  {n_busy:>6}")
    print(f"    no fresh ATM contract available                 {n_nopath:>6}")
    print(f"  scale-ins taken                                   {n_fired:>6}"
          f"   ({n_fired/max(n_elig,1)*100:.1f}% of eligible)")

    print(f"\n{'='*104}")
    print(f"  4. M3 -- PER RULE")
    print(f"{'='*104}")
    print(f"  {'rule':26} {'baseline':>10} {'multiplier':>12} {'delta':>9} "
          f"{'scale-ins':>10}")
    nimp = 0
    for nm, v in per_rule.items():
        tb_ = sum(r["pnl"] for r in v["base"])
        tm_ = sum(r["pnl"] for r in v["mult"])
        k = sum(1 for r in v["mult"] if r["leg"] == 2)
        nimp += int(tm_ > tb_)
        print(f"  {nm:26} {tb_:>+10.0f} {tm_:>+12.0f} {tm_-tb_:>+9.0f} "
              f"{k:>10}")

    print(f"\n{'='*104}")
    print(f"  5. ROBUSTNESS -- does the verdict survive losing the best "
          f"scale-ins?")
    print(f"{'='*104}")
    print(f"  A scale-in leg of {len(leg2)} trades on a book whose edge lives "
          f"in the right tail can be one")
    print(f"  session. Recompute the whole comparison with the best k removed. "
          f"If M2 flips, the")
    print(f"  Multiplier is not a strategy, it is a story about a day.")
    big = sorted(range(len(leg2)), key=lambda i: -leg2[i]["pnl"])
    print(f"\n  {'variant':26} {'leg2':>9} {'total':>9} {'return':>9} "
          f"{'max DD':>9} {'ret/DD':>8} {'vs base':>9}")
    print(f"  {'baseline':26} {'--':>9} {A['tot']:>+9.0f} {ra*100:>+8.1f}% "
          f"{da*100:>+8.1f}% {ca:>8.2f} {'--':>9}")
    for k in (0, 1, 3, 5):
        keep = [leg2[i] for i in range(len(leg2)) if i not in set(big[:k])]
        rk, dk = curve(leg1 + keep)
        ck = rk / abs(dk) if dk else np.nan
        t2 = sum(r["pnl"] for r in keep)
        lab = "multiplier, as run" if k == 0 else f"  minus best {k}"
        if k and k <= len(leg2):
            lab += f" ({leg2[big[k-1]]['rule'].split()[0]}"
            lab += f" {leg2[big[k-1]]['pnl']:+.0f})"
        print(f"  {lab:26} {t2:>+9.0f} {A['tot']+t2:>+9.0f} {rk*100:>+8.1f}% "
              f"{dk*100:>+8.1f}% {ck:>8.2f} "
              f"{'PASS' if ck > ca else 'FAIL':>9}")

    n = len(per_rule)
    mark = lambda ok: "PASS" if ok else "FAIL"
    print(f"\n{'='*104}")
    print(f"  SCORECARD  (pre-committed before the first run)")
    print(f"{'='*104}")
    print(f"  M1  total ROE improves          {M['tot']:>+9.0f} vs "
          f"{A['tot']:>+9.0f}   {mark(M['tot'] > A['tot'])}")
    print(f"  M2  return per unit of max DD   {cm:>9.2f} vs {ca:>9.2f}   "
          f"{mark(cm > ca)}   <-- decides it")
    print(f"  M3  rules improved                    {nimp:>3}/{n}"
          f"{'':>12}   {mark(nimp >= 6)}")
    print(f"  M4  scale-in leg standalone     {L2['tot']:>+9.0f}"
          f"{'':>13}   {mark(L2['tot'] > 0)}")
    print(f"  M5  fires on <50% of eligible      {n_fired/max(n_elig,1)*100:>6.1f}%"
          f"{'':>13}   {mark(n_fired / max(n_elig, 1) < 0.5)}")


if __name__ == "__main__":
    main()
