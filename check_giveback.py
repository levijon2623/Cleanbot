# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_giveback.py
=================
THE HOLE THIS TESTS
-------------------
`bot_runner`'s monitor loop is an if/elif chain where `elif trail_stop is not
None:` SWALLOWS the take-profit branch -- when trailing is armed, `bid >=
tp_limit` is unreachable (deliberate: "TP+trail was OOS -1.1% vs +4.8% pure").
So the ONLY exit is `peak_bid * (1 - trail_pct)`.

With trail_pct 0.50 that stop only rises above the entry when
    peak * 0.5 > entry   <=>   peak > 2 * entry   <=>   peak ROE > +100%

    peak ROE     trail stop lands at
        0%            -50%
      +50%            -25%
      +92%             -4%     <- AVGO 2026-09-09, the live example
     +100%              0%
     +200%            +50%

**Every trade whose peak ROE falls between 0% and +100% is mathematically
guaranteed to exit at a loss.** The trail cannot protect a gain until the
position has more than doubled, and the TP that used to catch +100% is disabled.
AVGO on 2026-09-09 peaked +92% (7 cents under its $1.86 TP) and closed -46.2%.

`check_exit_sweep` / `check_exit_walkforward` swept fixed brackets, four price
trails and ONE tp+trail combo. A ROE-ARMED GIVE-BACK was never in the grid.

WHAT IS TESTED
--------------
The give-back is layered ON TOP of the deployed 50% price trail, never replacing
it. The loss side is therefore IDENTICAL to production and the only thing that
changes is that winners can be cut earlier:

    once peak ROE >= `arm`,  exit when ROE <= peak_ROE * (1 - `give`)
    exit level = max(price_trail_level, giveback_level)   -- whichever binds first

    give = 0.25  -> keep 75% of the best gain
    give = 0.50  -> keep half
    give = 1.00  -> breakeven-arm (give it all back, stop at entry)

This is a STRICT TIGHTENING, so the experiment isolates the user's own worry:
does cutting winners early cost more than the round-trips it saves?

RE-ENTRY IS MODELLED, because it is half the argument. `_walk2` holds one
position per ticker and frees the ticker at the exit minute, so an earlier exit
genuinely does become eligible to re-fire on the next trigger -- exactly what
AVGO did three times. Trade counts are reported so the extra re-entries are
visible rather than assumed.

THE LIVE EXIT CUSHION IS ALSO MODELLED (--cushion), and it matters here.
`bot_runner._fire_exit` prices the closing limit at `bid - spread*1.5` on a
losing exit and `bid - spread*0.5` on a profitable one, while the existing sims
exit at plain `bid`. Measured on the three live AVGO trades that gap was 11-14pp
of entry premium. It is not neutral between policies: the trail exits at a loss
(1.5x cushion) far more often than a give-back does (0.5x), so leaving it out
FLATTERS the deployed trail. Both views are printed.

PRE-COMMITTED PASS CRITERIA (fixed before any output was viewed). A give-back
policy replaces `trail50` only if ALL of:
  G1  book OOS > trail50's OOS, BOTH with and without the cushion modelled
  G2  IS > 0 and OOS > 0
  G3  >= 5 of 6 calendar slices populated and positive
  G4  it WINS THE WALK-FORWARD POLICY CHOICE -- chosen only on data before a
      slice and scored on that slice, chained. (A grid sweep alone is a
      14-policy search; G4 is the "would we have picked it in time" test.)

Usage:
  python check_giveback.py
  python check_giveback.py --rules "AVGO HIVOL PUT"
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
from check_exit_walkforward import _eod_mod

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


# ------------------------------------------------------------------ policies
def _policies(trail=0.50):
    P = [dict(name="static tp100/none", kind="fixed", tp=1.0, stop=None),
         dict(name="static tp100/sl50", kind="fixed", tp=1.0, stop=0.50),
         dict(name=f"trail{int(trail*100)} (DEPLOYED)", kind="trail", trail=trail)]
    # ARM LEVELS. The original grid stopped at 0.50 and the whole family failed
    # -- but a give-back armed at +20% fires on MOST trades and therefore caps
    # everything, which is the mechanism check_giveback found destroys
    # expectancy (the book's P&L is a right tail: top 1% of trades = 64% of
    # total). The HIGH arms are a different rule: check_peak_profit measured
    # only 25.5% of trades ever peaking above +100%, so an arm at 2.00-3.00
    # touches a small minority -- precisely the monsters the trail gives back
    # most of. Live example 2026-09-14: IWM peaked +364% and the 50% trail
    # booked +98%; arm=2.00/give=0.25 would have exited near +273%.
    # 1.00 and 1.50 are included so the dose-response is visible rather than
    # inferred from the endpoints.
    for arm in (0.20, 0.30, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00):
        for give in (0.25, 0.33, 0.50, 1.00):
            tag = "BE" if give >= 1.0 else f"g{int(give*100)}"
            P.append(dict(name=f"trail{int(trail*100)}+arm{int(arm*100)}/{tag}",
                          kind="trail", trail=trail, arm=arm, give=give))
    return P


def _sim(path, pol, eod_m, realistic=True, cushion=False):
    """path = (e_mid, e_ask, cl, hi, lo, bid, ask, mod) -> (pnl, exit_mod, tag).

    Levels are set off the entry MID (a live bracket is priced off the mid);
    the FILL is the ask on the way in and the bid on the way out, plus the
    live marketable-limit cushion when asked for.
    """
    import directional_flow_backtester as D
    e_mid, e_ask, cl, hi, lo, bid, ask, mod = path
    entry = e_ask if realistic else e_mid
    ref = e_mid
    n = len(cl)
    tp_lvl = ref * (1 + pol["tp"]) if pol.get("tp") else None
    hard = ref * (1 - pol["stop"]) if pol.get("stop") else None
    trail = pol.get("trail")
    arm, give = pol.get("arm"), pol.get("give")
    peak = ref

    def _out(i, lvl_hint, tag):
        px = cl[i] if not realistic else (bid[i] if bid[i] > 0 else min(lvl_hint, cl[i]))
        if realistic and cushion and bid[i] > 0 and ask[i] > bid[i]:
            sp = ask[i] - bid[i]
            px = max(0.01, px - sp * (0.5 if px > entry else 1.5))
        return (px - entry) / entry - D.COMMISSION_PCT, int(mod[i]), tag

    for i in range(n):
        if mod[i] >= eod_m:
            return _out(i, cl[i], "eod")
        lvl, tag = hard, "stop"
        if trail:
            t = peak * (1 - trail)
            if lvl is None or t > lvl:
                lvl, tag = t, "trail"
            if arm is not None:
                peak_roe = peak / ref - 1.0
                if peak_roe >= arm:
                    g = ref * (1.0 + peak_roe * (1.0 - give))
                    if g > lvl:
                        lvl, tag = g, "give"
        if lvl is not None and lo[i] <= lvl:
            return _out(i, lvl, tag)
        if tp_lvl is not None and cl[i] >= tp_lvl:
            return _out(i, tp_lvl, "tp")
        peak = max(peak, cl[i])
    return _out(n - 1, cl[-1], "eod")


def _walk(cand, pol, eod_m, realistic=True, cushion=False):
    """One position per ticker; the ticker frees up at the exit minute, so an
    earlier exit really does become eligible to re-fire."""
    cur, busy, out = None, -1, []
    for d, m, path in cand:
        if d != cur:
            cur, busy = d, -1
        if m < busy:
            continue
        pnl, xm, tag = _sim(path, pol, eod_m, realistic, cushion)
        out.append((d, pnl, tag))
        busy = xm
    return out


# ------------------------------------------------------------------ stats
def _st(rows):
    if not rows:
        return None
    v = np.array([p for _, p, _ in rows], float)
    i = np.array([p for d, p, _ in rows if d < SPLIT], float)
    o = np.array([p for d, p, _ in rows if d >= SPLIT], float)
    sl = [[] for _ in range(6)]
    for d, p, _ in rows:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    pop = [np.mean(b) for b in sl if len(b) >= 3]
    w = v[v > 0]; l = v[v <= 0]
    tags = {}
    for _, _, t in rows:
        tags[t] = tags.get(t, 0) + 1
    return dict(n=len(v), all=v.mean(), is_=i.mean() if len(i) else np.nan,
                oos=o.mean() if len(o) else np.nan, win=(v > 0).mean(),
                mw=w.mean() if len(w) else np.nan, ml=l.mean() if len(l) else np.nan,
                npop=len(pop), nposs=sum(1 for x in pop if x > 0), tot=v.sum(), tags=tags)


def _row(lbl, s, base_oos=None):
    if s is None or s["n"] < 12:
        return f"  {lbl:26} (thin)"
    d = f" {(s['oos']-base_oos)*100:>+6.1f}pp" if base_oos is not None else " " * 9
    tg = " ".join(f"{k}:{v}" for k, v in sorted(s["tags"].items(), key=lambda x: -x[1]))
    return (f"  {lbl:26} n={s['n']:>4} IS {s['is_']*100:>+6.1f}% OOS {s['oos']*100:>+7.1f}%{d} "
            f"win {s['win']:>4.2f} W {s['mw']*100:>+6.1f}% L {s['ml']*100:>+6.1f}% "
            f"sl {s['nposs']}/{s['npop']}  [{tg}]")


# ------------------------------------------------------------------ build
def _build(D, r, TRAIL_PCT):
    from check_config_walkforward import _flow_for
    tk = r["ticker"]
    flow = _flow_for(D, [tk])
    if flow.empty:
        return None
    gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
               "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
    trigs = D.triggers_for(flow, tk)
    D.annotate_flow_pct(trigs, r.get("flow_window_days", 60))
    try:
        tb = D._ticker_bars(tk)
    except Exception:
        tb = None
    if tb is None or tb.empty:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
    if tb is None or tb.empty:
        return None
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}
    from amt_profile import amt_open_map, amt_ok
    amt = amt_open_map(tk) if r.get("amt_open") else {}
    matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
    if r.get("amt_open"):
        matched = [(t, th) for t, th in matched if amt_ok(r["amt_open"], amt.get(t["date"]))]
    matched.sort(key=lambda x: pd.Timestamp(x[0]["ts"]))

    cand = []
    for t, _th in matched:
        d, ts = t["date"], t["ts"]
        day = bbd.get(d)
        if day is None:
            continue
        at = day[day["minute_et"] <= ts]
        if at.empty:
            continue
        spot = float(at.iloc[-1]["underlying_close"])
        cid = None
        for dd in r.get("dte", [0, 1]):
            cid = D.pick_contract(day, ts, r["direction"], dd, spot)
            if cid is not None:
                break
        if cid is None:
            continue
        ent = bbc[cid]
        er = ent[(ent["minute_et"] <= ts) & (ent["minute_et"] >= ts - pd.Timedelta(minutes=3))]
        if er.empty:
            continue
        er = er.iloc[-1]
        b, k = float(er["bid_close"]), float(er["ask_close"])
        mid = (b + k) / 2.0 if b > 0 else float(er["close"])
        if mid < 0.50:
            continue
        fwd = ent[ent["minute_et"] > ts].sort_values("minute_et")
        if len(fwd) < 3:
            continue
        pm = fwd["minute_et"]
        cand.append((d, pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute,
                     (mid, k if k > 0 else mid, fwd["close"].to_numpy(float),
                      fwd["high"].to_numpy(float), fwd["low"].to_numpy(float),
                      fwd["bid_close"].to_numpy(float), fwd["ask_close"].to_numpy(float),
                      (pm.dt.hour.values * 60 + pm.dt.minute.values).astype(int))))
    return cand


def run(a):
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.rules:
        rules = [r for r in rules if r["name"] in a.rules]

    store = {}
    for r in rules:
        c = _build(D, r, TRAIL_PCT)
        if c:
            store[r["name"]] = (r, c, _eod_mod(r))
            print(f"  {r['name']:26} {len(c):>5} triggers")
    if not store:
        print("nothing"); return

    POLS = _policies(TRAIL_PCT or 0.50)
    import sim_core

    def _pol_for(pol, r):
        """The DEPLOYED baseline must be each rule's REAL exit -- META/NVDA carry
        trail_pct 0, so a book-wide trail50 is not what they run. The give-back
        variants stay uniform: they are the treatment being swept."""
        return sim_core.policy_for(r, TRAIL_PCT) if "DEPLOYED" in pol["name"] else pol

    for cush in (False, True):
        head = "WITH live exit cushion (bid - 1.5x spread on losses, 0.5x on gains)" if cush \
               else "WITHOUT cushion (exit at plain bid -- what every prior sim did)"
        print("\n" + "=" * 132)
        print(f"  BOOK-WIDE EXIT POLICY SWEEP -- {head}")
        print("  W/L = mean winner / mean loser.  [tags] = which exit fired.")
        print("=" * 132)
        base = None
        res = {}
        for pol in POLS:
            rows = []
            for nm, (r, c, em) in store.items():
                rows += _walk(c, _pol_for(pol, r), em, realistic=True, cushion=cush)
            s = _st(rows)
            res[pol["name"]] = s
            if "DEPLOYED" in pol["name"]:
                base = s["oos"] if s else None
        for pol in POLS:
            print(_row(pol["name"], res[pol["name"]], base))
        if cush:
            globals()["_RES_CUSH"] = res
        else:
            globals()["_RES_PLAIN"] = res

    # ---------------- walk-forward the policy choice (G4) -----------------
    print("\n" + "=" * 132)
    print("  WALK-FORWARD POLICY CHOICE (G4) -- policy picked ONLY on data before each")
    print("  slice, then scored on that slice. Cushion modelled. This is the honest test.")
    print("=" * 132)
    per = {}
    for pol in POLS:
        rows = []
        for nm, (r, c, em) in store.items():
            rows += _walk(c, _pol_for(pol, r), em, realistic=True, cushion=True)
        per[pol["name"]] = rows
    chosen, wf_pnl, wf_dep = [], [], []
    for k in range(1, 6):
        picks = {}
        for pn, rows in per.items():
            prior = [p for d, p, _ in rows if _slice_idx(d) is not None and _slice_idx(d) < k]
            if len(prior) >= 20:
                picks[pn] = float(np.mean(prior))
        if not picks:
            continue
        best = max(picks, key=picks.get)
        cur = [p for d, p, _ in per[best] if _slice_idx(d) == k]
        dep = [p for d, p, _ in per[[p2["name"] for p2 in POLS if "DEPLOYED" in p2["name"]][0]]
               if _slice_idx(d) == k]
        if not cur:
            continue
        chosen.append((k, best, len(cur), float(np.mean(cur)),
                       float(np.mean(dep)) if dep else float("nan")))
        wf_pnl += cur
        wf_dep += dep
    print(f"  {'slice':6} {'policy chosen on prior data':30} {'n':>4} {'scored':>9} {'deployed':>9}")
    for k, b, n, m, dm in chosen:
        print(f"  S{k+1:<5} {b:30} {n:>4} {m*100:>+8.1f}% {dm*100:>+8.1f}%")
    if wf_pnl:
        print(f"\n  walk-forward chained:  chosen {np.mean(wf_pnl)*100:>+7.1f}% (n={len(wf_pnl)})"
              f"   vs deployed trail {np.mean(wf_dep)*100:>+7.1f}% (n={len(wf_dep)})")

    # ---------------- verdict against the pre-committed criteria ----------
    print("\n" + "=" * 132)
    print("  VERDICT vs PRE-COMMITTED CRITERIA")
    print("  G1 beats trail50 OOS in BOTH cushion views | G2 IS>0 & OOS>0 | G3 >=5/6 slices +ve | G4 wins walk-forward")
    print("=" * 132)
    RP, RC = globals()["_RES_PLAIN"], globals()["_RES_CUSH"]
    dep_name = [p["name"] for p in POLS if "DEPLOYED" in p["name"]][0]
    wf_best = chosen[-1][1] if chosen else None
    wf_win = bool(wf_pnl) and np.mean(wf_pnl) > np.mean(wf_dep)
    any_pass = False
    for pol in POLS:
        pn = pol["name"]
        if pn == dep_name or "static" in pn:
            continue
        sp, sc = RP.get(pn), RC.get(pn)
        if not sp or not sc:
            continue
        g1 = sp["oos"] > RP[dep_name]["oos"] and sc["oos"] > RC[dep_name]["oos"]
        g2 = sc["is_"] > 0 and sc["oos"] > 0
        g3 = sc["npop"] >= 5 and sc["nposs"] >= 5
        g4 = wf_win and wf_best == pn
        flags = "".join("Y" if x else "." for x in (g1, g2, g3, g4))
        if all((g1, g2, g3, g4)):
            any_pass = True
        print(f"  {pn:26} {flags}   {'** PASS' if all((g1,g2,g3,g4)) else ''}")
    if not any_pass:
        print("\n  No give-back policy cleared all four. Deployed trail stands.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="*", default=None)
    run(ap.parse_args())
