# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_candidate_retest.py
=========================
Re-test previously-REJECTED candidates under a different exit.

Every candidate in this book was screened under `target_roe 1.0 / rr 1.0`, i.e.
a take-profit at +100% and a stop at entry*(1-1.0) = ZERO. That bracket is wrong
in TWO directions at once:
    * it CAPS winners at +100% (a rule whose winners run to +250% scores +100%)
    * it never STOPS losers (they ride to near-zero by the 15:55 flatten)
A trailing exit inverts both. So the rules a trail helps most are exactly the
low-win-rate / high-maxLL profile that got candidates rejected -- e.g. TSLA CHOP
PUT was disabled on "win 0.41, maxLL 46".

`check_exit_walkforward` section 3 only showed the GATES survive an exit change
on the rules that were KEPT. It said nothing about rules that were THROWN OUT.

PRE-COMMITTED PASS CRITERIA (fixed before looking at any output, and deliberately
the bar that LULU HIVOL/UP PUT would have failed):
    C1  day-level IS  > 0
    C2  day-level OOS > 0
    C3  >= 5 of 6 calendar slices populated, and >= 5 of them positive
    C4  OOS day-level beats the p95 of a day-level bootstrap null drawn from
        that ticker+direction's own p50 trigger population
All four required. Trade-weighted stats and mid-fill stats do NOT count -- every
number here is DAY-LEVEL on REALISTIC fills (enter ask, exit bid), sequential.

Usage:  python check_candidate_retest.py [--boot 4000]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
from check_exit_walkforward import _walk, _eod_mod

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()

# "likely to move" = rejected on a big-loss / low-win profile a trail could fix
# "worth including" = marginal or held candidates
CANDIDATES = [
    # --- likely to move ---
    dict(tag="likely", name="TSLA CHOP PUT", ticker="TSLA", direction="PUT",
         hours=[9, 10, 11, 12, 13, 14], dte=[0, 1], regime="CHOP",
         amt_open="above_va", min_flow_pct=80, target_roe=0.80, rr=1.0),
    dict(tag="likely", name="SPY POS/PUT", ticker="SPY", direction="PUT",
         hours=[9, 10, 11, 12, 13, 14], dte=[0], regime="POSITIVE_GEX",
         amt_open="inside_va", min_flow_pct=80, target_roe=1.00, rr=1.0),
    dict(tag="likely", name="ORCL DOWNTREND PUT", ticker="ORCL", direction="PUT",
         hours=[9, 10, 11, 12, 13, 14], dte=[0, 1], regime="DOWNTREND",
         min_flow_pct=65, target_roe=1.00, rr=1.0),
    dict(tag="likely", name="IBIT UPTREND PUT", ticker="IBIT", direction="PUT",
         hours=[9, 10, 11, 12, 13, 14], dte=[0, 1], regime="UPTREND",
         min_flow_pct=50, target_roe=1.00, rr=1.0),
    # --- worth including ---
    dict(tag="incl", name="TSM CHOP PUT", ticker="TSM", direction="PUT",
         hours=[9, 10, 11, 12, 13, 14], dte=[0, 1], regime="CHOP",
         min_flow_pct=50, target_roe=1.00, rr=1.0),
    dict(tag="incl", name="NFLX CHOP PUT", ticker="NFLX", direction="PUT",
         hours=[9, 10, 11, 12, 13, 14], dte=[0, 1], regime="CHOP",
         min_flow_pct=50, target_roe=1.00, rr=1.0),
    dict(tag="incl", name="HOOD LOWVOL PUT", ticker="HOOD", direction="PUT",
         hours=[9, 10, 11, 12, 13, 14], dte=[0, 1], regime="LOWVOL",
         min_flow_pct=80, target_roe=1.00, rr=1.0),
    dict(tag="incl", name="META midday PUT amp>=1", ticker="META", direction="PUT",
         hours=[11, 12, 13, 14], dte=[0, 1], amp_min=1,
         min_flow_pct=65, target_roe=0.80, rr=1.0),
    dict(tag="incl", name="TSLL NORMVOL CALL", ticker="TSLL", direction="CALL",
         hours=[9, 10, 11, 12, 13, 14], dte=[0, 1], regime="NORMVOL",
         min_flow_pct=65, target_roe=0.40, rr=1.0),
    # --- NEW screen (2026-09-09), not a re-test: CVX came out of check_etf_screen
    # day-level positive in BOTH halves across pct 50/65/80. Friday-only expiry, so
    # the dte=[0,1] fallback restricts it to Thu(1DTE)/Fri(0DTE) on its own -- no dow
    # field needed, a Monday simply has no 0/1DTE chain to pick. Held to the SAME
    # four pre-committed criteria as the re-tests above.
    dict(tag="new", name="CVX NORMVOL PUT", ticker="CVX", direction="PUT",
         hours=[9, 10, 11, 12, 13, 14], dte=[0, 1], regime="NORMVOL",
         min_flow_pct=50, target_roe=1.00, rr=1.0),
]


def _cands_for(D, spec, *_ignored):
    """DEPRECATED SHIM -- delegates to sim_core.build_candidates.

    The old inline body emitted a 7-tuple payload with no ask array, so the
    live exit cushion could not be modelled and every verdict below was ~10pp
    optimistic. Positional extras (bbc/bbd/trigs/gex/...) are accepted and
    ignored so existing call sites keep working; sim_core loads what it needs.
    """
    import sim_core
    return sim_core.build_candidates(D, spec)


def _dayeval(pnls):
    """day-level stats: one observation per date."""
    if not pnls:
        return None
    df = pd.DataFrame(pnls, columns=["date", "pnl"])
    dm = df.groupby("date")["pnl"].mean()
    i = dm[[d < SPLIT for d in dm.index]]
    o = dm[[d >= SPLIT for d in dm.index]]
    sl = [[] for _ in range(6)]
    for d, p in dm.items():
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    pop = [np.mean(b) for b in sl if len(b) >= 3]
    return dict(nt=len(df), nd=len(dm), all=dm.mean(),
                is_=i.mean() if len(i) else np.nan, nis=len(i),
                oos=o.mean() if len(o) else np.nan, noos=len(o),
                win=(dm > 0).mean(), npop=len(pop), nposs=sum(1 for x in pop if x > 0),
                dm=dm, oos_days=o)


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import TRAIL_PCT
    from amt_profile import amt_open_map, amt_ok

    rows = []
    for spec in CANDIDATES:
        tk = spec["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            print(f"  {spec['name']}: no NETPREM"); continue
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
                   "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        try:
            tb = D._ticker_bars(tk)
        except Exception:
            tb = None
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        if tb is None or tb.empty:
            print(f"  {spec['name']}: no bars"); continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        amt = amt_open_map(tk) if spec.get("amt_open") else {}
        em = _eod_mod(spec)

        cand = _cands_for(D, spec, bbc, bbd, trigs, gex, vol, trd, amp, reg_src, amt, amt_ok)
        # null pool: same ticker+direction, p50, no other gate
        pool_spec = dict(ticker=tk, direction=spec["direction"],
                         hours=[9, 10, 11, 12, 13, 14], dte=spec.get("dte", [0, 1]),
                         min_flow_pct=50, target_roe=spec["target_roe"], rr=spec["rr"])
        pool = _cands_for(D, pool_spec, bbc, bbd, trigs, gex, vol, trd, amp, reg_src, {}, amt_ok)

        tr_, rr_ = float(spec["target_roe"]), float(spec["rr"])
        pols = {
            "static": dict(name="s", kind="fixed", tp=tr_,
                           stop=(tr_ / rr_ if tr_ / rr_ < 1.0 else None)),
            f"trail{int(TRAIL_PCT*100)}": dict(name="t", kind="trail", tp=None,
                                               trail=TRAIL_PCT, stop=None),
        }
        res = {}
        for pn, pol in pols.items():
            ev = _dayeval(_walk(cand, pol, em, realistic=True))
            if ev is None:
                continue
            pev = _dayeval(_walk(pool, pol, em, realistic=True))
            boot_p95 = np.nan
            if pev is not None and len(ev["oos_days"]) >= 6 and len(pev["oos_days"]) > len(ev["oos_days"]) + 3:
                rng = np.random.default_rng(0)
                draws = np.array([rng.choice(pev["oos_days"].to_numpy(),
                                             size=len(ev["oos_days"]), replace=False).mean()
                                  for _ in range(a.boot)])
                boot_p95 = float(np.percentile(draws, 95))
            ev["p95"] = boot_p95
            res[pn] = ev
        rows.append((spec, res))

    print("=" * 126)
    print("  REJECTED-CANDIDATE RE-TEST UNDER A DIFFERENT EXIT")
    print("  day-level | realistic fills (ask in / bid out) | sequential")
    print("  PASS = C1 day-IS>0  C2 day-OOS>0  C3 >=5/6 slices populated & positive  C4 day-OOS > bootstrap p95")
    print("=" * 126)
    for tag, label in (("likely", "LIKELY TO MOVE"), ("incl", "WORTH INCLUDING"),
                       ("new", "NEW SCREEN (not a re-test) -- same four criteria")):
        print(f"\n### {label}")
        print(f"  {'candidate':24} {'exit':9} {'trades':>7} {'days':>5} {'IS':>8} {'OOS':>8} "
              f"{'dwin':>5} {'slices':>7} {'p95':>7}  C1234  verdict")
        for spec, res in rows:
            if spec["tag"] != tag:
                continue
            best = None
            for pn, ev in res.items():
                c1 = ev["is_"] > 0
                c2 = ev["oos"] > 0
                c3 = ev["npop"] >= 5 and ev["nposs"] >= 5
                c4 = np.isfinite(ev["p95"]) and ev["oos"] > ev["p95"]
                flags = "".join("Y" if c else "." for c in (c1, c2, c3, c4))
                ok = all((c1, c2, c3, c4))
                if ok and (best is None):
                    best = pn
                print(f"  {spec['name'] if pn=='static' else '':24} {pn:9} "
                      f"{ev['nt']:>7} {ev['nd']:>5} "
                      f"{ev['is_']*100:>+7.1f}% {ev['oos']*100:>+7.1f}% {ev['win']:>5.2f} "
                      f"{ev['nposs']:>3}/{ev['npop']:<3} "
                      f"{ev['p95']*100 if np.isfinite(ev['p95']) else float('nan'):>+6.1f}%  "
                      f"{flags}   {'** PASS' if ok else ''}")
            if best is None:
                print(f"  {'':24} {'':9} {'':>7} {'':>5} {'':>8} {'':>8} {'':>5} {'':>7} {'':>7}         -> stays rejected")
            else:
                print(f"  {'':24} {'':9} {'':>7} {'':>5} {'':>8} {'':>8} {'':>5} {'':>7} {'':>7}         -> PASSES on {best}")

    npass = sum(1 for _s, r in rows
                for pn, ev in r.items()
                if ev["is_"] > 0 and ev["oos"] > 0 and ev["npop"] >= 5 and ev["nposs"] >= 5
                and np.isfinite(ev["p95"]) and ev["oos"] > ev["p95"])
    print(f"\n  {npass} candidate-exit combination(s) cleared all four criteria "
          f"out of {sum(len(r) for _s, r in rows)} tested.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    run(a)
