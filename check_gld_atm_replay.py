# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_gld_atm_replay.py
=======================
REPLAY THE 2026-09-18 GLD SESSION AT THE ATM STRIKE INSTEAD OF OTM+1.

WHY
    `GLD amp1 CALL` carries strike_offset: 1, so the live bot buys one strike
    OTM. bot_runner honours it (line 1785); sim_core.build_candidates does not
    -- D.pick_contract has no offset parameter at all -- so every backtest of
    GLD has been scoring a contract the bot does not buy. The offset itself was
    adopted from check_strike_selection (2026-09-07), which predates all three
    fill corrections, and under the corrected model check_strike_offset puts
    GLD's ATM arm 7.8pp of median ROE AHEAD of OTM+1.

    This replays the one session we have complete silver bars for, at both
    strikes, to see what the offset actually cost on the day it misfired.

WHAT IS HELD FIXED
    Same trigger times from the ledger, same day, same sequential guard (one
    position at a time), same trail50 exit, same botcap fill with GLD's measured
    CUSHION_CAP of 0.40. The ONLY thing that varies is which contract is bought:
      OTM+1  the contract the bot actually bought, matched out of the ledger
      ATM    nearest strike to spot at the entry minute
    So a difference here is the strike and nothing else.

🚨 ONE SESSION IS AN ANECDOTE
    Eight triggers on one day cannot settle whether the offset should go -- that
    needs check_strike_offset across the book, which already points the same way.
    This answers the narrower question actually asked: on THIS session, what did
    the offset cost? Read it as a worked example, not as evidence.

Usage:
  python check_gld_atm_replay.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import build_counterfactual as B
import sim_core

DATE, TK = "2026-09-18", "GLD"
CAP = sim_core.CUSHION_CAP.get(TK)


def atm_contract(df, spot, ts, dte_want=0):
    """Nearest strike to spot among contracts quoting at/just before ts."""
    d = pd.Timestamp(ts).date()
    x = df.copy()
    x["exp"] = pd.to_datetime(x["expiry"]).dt.date
    x = x[(x["option_type"] == "call")
          & (x["exp"].map(lambda e: (e - d).days) == dte_want)]
    if x.empty:
        return None
    x = x.loc[(x["strike"].astype(float) - spot).abs().idxmin()]
    return str(x["option_chain_id"]), float(x["strike"])


def walk(P, cid, mod0):
    """Forward path from mod0 -> the sim_core payload shape."""
    if cid not in P:
        return None
    mods, bid, ask = P[cid]
    i = np.searchsorted(mods, mod0)
    if i >= len(mods) - 3:
        return None
    b0, a0 = bid[i], ask[i]
    if not np.isfinite(a0) or a0 <= 0:
        return None
    mid = (b0 + a0) / 2 if b0 > 0 else a0
    f = slice(i + 1, None)
    cl = np.where(np.isfinite(bid[f]) & (bid[f] > 0), bid[f], ask[f])
    return (mid, a0, cl, cl, cl, bid[f], ask[f], mods[f]), mid


def sequential(ents, P, df, pick, pol, eod):
    """One position at a time -- the guard the live bot enforces."""
    out, busy = [], -1
    for e in ents:
        if e["mod"] < busy:
            out.append(dict(mod=e["mod"], skipped=True))
            continue
        got = pick(e)
        if got is None:
            out.append(dict(mod=e["mod"], skipped=False, nocontract=True))
            continue
        cid, strike = got
        w = walk(P, cid, e["mod"])
        if w is None:
            out.append(dict(mod=e["mod"], skipped=False, nocontract=True))
            continue
        path, entry = w
        if entry < 0.50:                       # the live $0.50 floor
            out.append(dict(mod=e["mod"], skipped=False, floored=True,
                            entry=entry, strike=strike))
            continue
        pnl, xm, tag = sim_core.simulate(path, pol, eod, fill="botcap",
                                         cush_cap=CAP)
        busy = int(xm)
        out.append(dict(mod=e["mod"], skipped=False, strike=strike,
                        entry=entry, pnl=pnl * 100, exit_mod=int(xm), tag=tag))
    return out


def show(rows, label):
    took = [r for r in rows if "pnl" in r]
    print(f"\n  {label}")
    print(f"  {'entry':>7} {'strike':>7} {'px':>6} {'exit':>6} {'tag':>6} {'ROE':>9}")
    for r in rows:
        t = f"{r['mod']//60:02d}:{r['mod']%60:02d}"
        if r.get("skipped"):
            print(f"  {t:>7} {'--':>7} {'--':>6} {'--':>6} {'--':>6} "
                  f"{'(position already open)':>9}")
        elif r.get("floored"):
            print(f"  {t:>7} {r['strike']:>7.0f} {r['entry']:>6.2f} {'--':>6} "
                  f"{'--':>6} {'(below $0.50 floor)':>9}")
        elif r.get("nocontract"):
            print(f"  {t:>7} {'--':>7} {'--':>6} {'--':>6} {'--':>6} "
                  f"{'(no quoted contract)':>9}")
        else:
            x = f"{r['exit_mod']//60:02d}:{r['exit_mod']%60:02d}"
            print(f"  {t:>7} {r['strike']:>7.0f} {r['entry']:>6.2f} {x:>6} "
                  f"{r['tag']:>6} {r['pnl']:>+9.1f}%")
    if took:
        v = [r["pnl"] for r in took]
        print(f"  -> {len(took)} round-trip(s), total {sum(v):+.1f}%, "
              f"median {np.median(v):+.1f}%")
    return took


def main():
    ents = B.ledger_entries(DATE, TK)
    P, df = B.paths(DATE, TK)
    print(f"  {DATE} {TK}: {len(ents)} ledger entries, "
          f"{len(P)} contracts with silver bars")

    # spot at each entry, straight off the ledger row the bot wrote
    raw = {r["mod"]: r for r in ents}
    spots = {}
    for line in open(B.LOG, encoding="utf-8-sig"):
        s = line.strip()
        if not s:
            continue
        try:
            r = __import__("json").loads(s[:s.rfind("}") + 1])
        except Exception:
            continue
        if (r.get("ticker") == TK and str(r.get("timestamp", ""))[:10] == DATE
                and str(r.get("action", "")).startswith("ENTRY")):
            ts = pd.Timestamp(r["timestamp"])
            spots[ts.hour * 60 + ts.minute] = float(r.get("spot_at_entry") or 0)

    rule = next(r for r in sim_core.research_rules(include_paper=True)
                if r["ticker"] == TK)
    pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
    print(f"  rule '{rule['name']}'  policy {pol}  cushion cap {CAP}\n")

    otm = sequential(ents, P, df, lambda e: (
        (lambda c: (c, float(e["contract"][-8:]) / 1000) if c else None)(
            B.match_contract(e["contract"], df))), pol, eod)
    atm = sequential(ents, P, df,
                     lambda e: atm_contract(df, spots.get(e["mod"], 0), e["ts"]),
                     pol, eod)

    a = show(otm, "OTM+1  (strike_offset 1 -- what the bot actually bought)")
    b = show(atm, "ATM    (nearest strike to spot -- offset removed)")

    print(f"\n{'='*72}")
    ta = sum(r["pnl"] for r in a) if a else 0.0
    tb = sum(r["pnl"] for r in b) if b else 0.0
    print(f"  OTM+1 {ta:>+8.1f}%   ATM {tb:>+8.1f}%   "
          f"difference {tb-ta:>+8.1f}pp")
    print(f"  One session, {len(a)} vs {len(b)} round-trip(s) -- a worked")
    print(f"  example of the offset's cost on this day, not evidence about the")
    print(f"  offset in general. check_strike_offset across the book is that.")


if __name__ == "__main__":
    main()
