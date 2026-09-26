# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_legin_vertical.py
=======================
LEG INTO A VERTICAL FROM A WINNING LONG: buy the ATM outright, and once it is up
+X%, sell the next OTM strike against it and hold to expiry.

WHY THIS IS NOT THE DEBIT VERTICAL ALREADY RULED OUT
    check_spread_feasibility killed entry-time debit verticals on friction:
    12.1% of capital round-tripped, because the short leg shrinks the net debit
    to ~$0.42 while you cross FOUR spreads (two legs, in and out). A leg-in on a
    0DTE crosses TWO:

        debit vertical, round-tripped  buy long + sell short, close both   = 4
        leg-in, held to expiry         buy long, sell short later, expire  = 2
        naked long, round-tripped      buy + sell                          = 2

    A 0DTE that finishes cleanly ITM or OTM never needs closing, so the leg-in
    carries roughly NAKED-LONG friction, not vertical friction. And the short
    leg is sold AFTER the underlying has moved toward it, so it fetches more
    premium than it would have at entry.

WHAT IT DOES TO THE DISTRIBUTION
    The credit is banked the moment it is sold, whatever happens next. Upside is
    capped at the strike width; downside is reduced by the credit. So it trades
    the right tail -- which check_giveback showed IS this book's entire P&L --
    for a certain, immediate reduction in cost basis. Whether that is a good
    trade is exactly what the arm labelled `naked` is here to answer.

SETTLEMENT, stated because it is where this could be wrong
    Held to expiry, a 0DTE vertical settles at intrinsic: both legs OTM -> the
    long expires worthless and the credit is kept; both ITM -> the full strike
    width; long ITM / short OTM -> the long's intrinsic. Assignment on a short
    ITM leg is economically covered by the long but operationally real on
    American-style ETF options, and is NOT modelled here. Treat the result as
    the economics, not the operations.

Usage:
  python check_legin_vertical.py --tickers SPY QQQ IWM --legin 25 50 100
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
import polars as pl

from build_structure_tape import day_chain, trigger_map, EOD, FLOOR
from check_config_walkforward import SLICE_EDGES

PRE_LO = pd.Timestamp("2023-10-12").date()
DEPLOY_LO = SLICE_EDGES[0]
SPLIT = pd.Timestamp("2025-08-21").date()


def series_for(sub, cid):
    """(mods, bid, ask) for one contract, sorted."""
    g = sub[sub["option_chain_id"] == cid].sort_values("m")
    return (g["m"].to_numpy(int), g["bid_close"].to_numpy(float),
            g["ask_close"].to_numpy(float))


def legs_at(sub, direction, m):
    """(atm_row, otm_row, spot) chosen at minute m from the 0DTE chain."""
    at = sub[sub["m"] <= m]
    if at.empty:
        return None
    spot = float(at.iloc[-1]["underlying_close"])
    if not np.isfinite(spot) or spot <= 0:
        return None
    typ = "call" if direction == "CALL" else "put"
    side = at[(at["option_type"] == typ) & (at["dte"] == 0)]
    if side.empty:
        return None
    last = side.sort_values("m").groupby("option_chain_id", as_index=False).last()
    last["mid"] = (last["bid_close"] + last["ask_close"]) / 2
    last = last.assign(dist=(last["strike"] - spot).abs()).dropna(subset=["dist"])
    if last.empty:
        return None
    atm = last.loc[last["dist"].idxmin()]
    if float(atm["mid"]) < FLOOR:
        return None
    o = (last[last["strike"] > atm["strike"]].sort_values("strike")
         if typ == "call" else
         last[last["strike"] < atm["strike"]].sort_values("strike", ascending=False))
    if o.empty:
        return None
    return atm, o.iloc[0], spot


def settle(sub, strike, typ):
    """Intrinsic value of a 0DTE leg at the last observed underlying print."""
    last = sub.sort_values("m").iloc[-1]
    s = float(last["underlying_close"])
    return max(0.0, s - strike) if typ == "call" else max(0.0, strike - s)


def run_one(sub, direction, m, legin_pct, dynamic=False):
    """-> (naked_pnl, legin_pnl, legged_in) for one trigger minute.

    `dynamic` changes WHICH strike is sold, and it is the whole point of the
    variant. With dynamic=False the short strike is fixed at ENTRY, so legging
    in at +100% sells a strike the underlying has already run toward -- a large
    credit, but capping into a width that is mostly consumed. With
    dynamic=True the strike is re-chosen AT THE LEG-IN MINUTE, so the cap sits
    above the current price and the runner still has room. That is "rolling the
    short up with the move" rather than selling a strike the move has passed.
    """
    L = legs_at(sub, direction, m)
    if L is None:
        return None
    atm, otm, spot = L
    typ = "call" if direction == "CALL" else "put"
    am, ab, aa = series_for(sub, atm["option_chain_id"])
    if len(am) < 3:
        return None
    i0 = int(np.searchsorted(am, m))
    if i0 >= len(am):
        return None
    entry = min(round((ab[i0] + aa[i0]) / 2 + 0.01, 2), round(aa[i0], 2))
    if entry <= 0:
        return None

    # --- naked: hold to EOD/expiry, settle at intrinsic
    naked = (settle(sub, float(atm["strike"]), typ) - entry) / entry

    # --- leg-in: sell an OTM once the long's BID is up legin_pct
    trigger_px = entry * (1 + legin_pct)
    credit, legged, short_leg = 0.0, False, otm
    for i in range(i0, len(am)):
        if am[i] >= EOD:
            break
        if ab[i] < trigger_px:
            continue
        if dynamic:
            Lnow = legs_at(sub, direction, int(am[i]))
            if Lnow is None:
                break
            # keep the ORIGINAL long; only the short is re-chosen. It must
            # still be beyond the long's strike or the structure inverts.
            cand_short = Lnow[1]
            k = float(cand_short["strike"])
            if (typ == "call" and k <= float(atm["strike"])) or \
               (typ == "put" and k >= float(atm["strike"])):
                break
            short_leg = cand_short
        om, ob, oa = series_for(sub, short_leg["option_chain_id"])
        j = int(np.searchsorted(om, am[i]))
        if j < len(om) and ob[j] > 0:
            credit = ob[j]              # sold at the BID -- we cross the spread
            legged = True
        break
    if not legged:
        return naked, naked, False
    width = abs(float(short_leg["strike"]) - float(atm["strike"]))
    long_v = settle(sub, float(atm["strike"]), typ)
    short_v = settle(sub, float(short_leg["strike"]), typ)
    legin = (min(long_v - short_v, width) + credit - entry) / entry
    return naked, legin, True


def summarise(rows, label):
    if not rows:
        print(f"    {label:22} (none)")
        return
    d = np.array([r[0] for r in rows])
    v = np.array([r[1] for r in rows], float)
    w = []
    for lo, hi in ((PRE_LO, DEPLOY_LO), (DEPLOY_LO, SPLIT), (SPLIT, SLICE_EDGES[-1])):
        msk = (d >= lo) & (d < hi)
        w.append(v[msk].mean() * 100 if msk.sum() else np.nan)
    print(f"    {label:22} n={len(v):>5}  mean {v.mean()*100:>+7.1f}%  "
          f"win {(v>0).mean():.2f}  PRE {w[0]:>+7.1f}  IS {w[1]:>+7.1f}  "
          f"OOS {w[2]:>+7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--legin", nargs="*", type=int, default=[25, 50, 100])
    ap.add_argument("--dates", nargs="*", default=None)
    a = ap.parse_args()

    import directional_flow_backtester as D
    tmap, _ = trigger_map(D, a.tickers, a.pct)
    dates = ([dt.date.fromisoformat(x) for x in a.dates] if a.dates
             else sorted({d for _, d in tmap}))
    print(f"  {len(dates)} dates with a p{a.pct} trigger")

    naked_rows = []
    legin_rows = {(p, dy): [] for p in a.legin for dy in (False, True)}
    legged_n = {(p, dy): 0 for p in a.legin for dy in (False, True)}
    done = 0
    for d in dates:
        ch = day_chain(d, a.tickers)
        if ch is None:
            continue
        for tk in a.tickers:
            sub = ch[ch["underlying_symbol"] == tk]
            if sub.empty:
                continue
            mins = tmap.get((tk, d))
            if not mins:
                continue
            m = min(mins)                      # one position per ticker-day
            for direction in a.dirs:
                first = True
                for p in a.legin:
                    for dy in (False, True):
                        r = run_one(sub, direction, m, p / 100.0, dy)
                        if r is None:
                            continue
                        nk, lg, did = r
                        if first:
                            naked_rows.append((d, nk))
                            first = False
                        legin_rows[(p, dy)].append((d, lg))
                        legged_n[(p, dy)] += int(did)
        done += 1
        if done % 50 == 0:
            print(f"    {done}/{len(dates)} dates", flush=True)

    print(f"\n{'='*104}")
    print(f"  LEG INTO A VERTICAL vs HOLD THE NAKED LONG "
          f"(p{a.pct}, {' '.join(a.tickers)}, held to expiry)")
    print(f"{'='*104}")
    summarise(naked_rows, "naked long (baseline)")
    for p in a.legin:
        for dy in (False, True):
            key = (p, dy)
            n = len(legin_rows[key])
            rate = 100 * legged_n[key] / n if n else 0
            tag = 'roll-up' if dy else 'fixed  '
            summarise(legin_rows[key], f"+{p}% {tag}")
            print(f"    {'':22} legged in on {rate:.0f}% of trades")
    print(f"\n  A leg-in that never fires is just the naked long, so the gap")
    print(f"  shrinks toward zero as the threshold rises. Read the leg-in RATE")
    print(f"  alongside the mean: a +100% threshold that fires on 25% of trades")
    print(f"  is only changing a quarter of the book.")
    print(f"\n  No holdout remains -- diagnosis, not a deployment case.")


if __name__ == "__main__":
    main()

