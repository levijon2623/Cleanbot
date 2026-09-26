# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
replay_day.py
=============
Replay ONE live session from the lake tape under alternative trail widths.

WHY THIS IS NOT check_avgo_trail
    That script sweeps trails over the whole backtest and found AVGO has only 10
    tradeable days (5 OOS), with every CI spanning 1,000+ points -- the trail
    question is unanswerable there. This answers a narrower and legitimate
    question instead: on THIS session, which the user watched, what would a
    different trail have done? It is a case study, not evidence about the rule.

WHY THE LEDGER SUPPLIES THE TRIGGERS
    The historical flow files end 2026-08-21, so build_candidates cannot produce
    triggers for a recent session. The live bot's own ENTRY records are the
    trigger times, and they are real: each carries the net_flow that fired it.

WHAT IT CORRECTS FOR
    The 2026-09-15 AVGO records are contaminated by the zero-bid bug: each
    phantom exit freed the slot and the bot re-entered minutes later, so 16
    ledger entries represent a handful of real positions repeatedly killed.
    This replay honours the SEQUENTIAL GUARD -- once in, no re-entry until the
    position actually closes on the tape -- so it reconstructs the session the
    bot would have had with config.BAD_TICK_* in place.

FILL CONVENTIONS, matched to bot_runner rather than to the simulator
    entry  the ledger's own logged entry price (what the bot actually paid)
    exit   the tape's bid, minus _fire_exit's cushion (0.5x spread taking
           profit, 1.5x otherwise). The calibration work (sim_core.CUSHION_CAP)
           argues that cushion is too harsh for AVGO; `--cap` applies the
           measured 0.60 instead, and both are printed so the choice is visible.

Usage:
  python replay_day.py --date 2026-09-15 --ticker AVGO --trails 0.50 0.75
  python replay_day.py --date 2026-09-15 --ticker AVGO --trails 0.5 0.75 --cap 0.60
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd
import polars as pl

LOG = "bot_executions_log.jsonl"
SILVER = "lake/silver/option-contracts-1m"
OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
EOD_FLATTEN = 15 * 60 + 55


def ledger_entries(date, ticker):
    out = []
    for line in open(LOG, encoding="utf-8-sig"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        c = s.rfind("}")
        try:
            r = json.loads(s[:c + 1])
        except Exception:
            continue
        if not str(r.get("action", "")).startswith("ENTRY"):
            continue
        if r.get("ticker") != ticker or str(r.get("timestamp", ""))[:10] != date:
            continue
        ts = pd.Timestamp(r["timestamp"])
        out.append(dict(mod=ts.hour * 60 + ts.minute, ts=ts,
                        contract=r.get("contract"),
                        entry=float(r.get("entry_price") or 0),
                        trail=r.get("trail_pct"), flow=r.get("net_flow")))
    return sorted(out, key=lambda x: x["mod"])


def paths(date, ticker):
    p = os.path.join(SILVER, f"date={date}", "bars.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"  no silver partition for {date}")
    df = (pl.scan_parquet(p)
          .filter(pl.col("underlying_symbol") == ticker)
          .select("option_chain_id", "minute_et", "bid_close", "ask_close",
                  "strike", "expiry", "option_type")
          .collect().to_pandas())
    df["mod"] = (pd.to_datetime(df["minute_et"]).dt.hour * 60
                 + pd.to_datetime(df["minute_et"]).dt.minute)
    out = {}
    for cid, g in df.sort_values("mod").groupby("option_chain_id"):
        out[str(cid)] = (g["mod"].to_numpy(int),
                         pd.to_numeric(g["bid_close"], errors="coerce").to_numpy(float),
                         pd.to_numeric(g["ask_close"], errors="coerce").to_numpy(float))
    return out, df


def match_contract(occ, df):
    m = OCC.match(occ or "")
    if not m:
        return None
    _, ymd, cp, strike = m.groups()
    exp = dt.date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    k = int(strike) / 1000.0
    typ = "call" if cp == "C" else "put"
    sub = df[(df["strike"].astype(float) == k)
             & (pd.to_datetime(df["expiry"]).dt.date == exp)
             & (df["option_type"] == typ)]
    return str(sub["option_chain_id"].iloc[0]) if len(sub) else None


def run_trail(series, entry_mod, entry_px, trail, cap):
    """-> (exit_mod, exit_px, peak, tag). Mirrors bot_runner's loop."""
    mods, bid, ask = series
    j0 = int(np.searchsorted(mods, entry_mod))
    peak = entry_px
    for j in range(j0, len(mods)):
        b, k = bid[j], ask[j]
        if not np.isfinite(b) or b <= 0:
            continue                      # the bad-tick guard: ignore, do not exit
        if b > peak:
            peak = b
        if mods[j] >= EOD_FLATTEN:
            return int(mods[j]), _fill(b, k, entry_px, cap), peak, "eod"
        lvl = peak * (1 - trail)
        if b <= lvl:
            return int(mods[j]), _fill(b, k, entry_px, cap), peak, "trail"
    j = len(mods) - 1
    return int(mods[j]), _fill(bid[j], ask[j], entry_px, cap), peak, "eod"


def _fill(b, k, entry, cap):
    sp = max(0.0, (k - b)) if np.isfinite(k) else 0.0
    m = 0.5 if b > entry else 1.5
    if cap is not None:
        m = min(m, cap)
    return max(0.01, round(b - sp * m, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-09-15")
    ap.add_argument("--ticker", default="AVGO")
    ap.add_argument("--trails", nargs="+", type=float, default=[0.50, 0.75])
    ap.add_argument("--cap", type=float, default=None,
                    help="exit cushion cap in spreads (sim_core.CUSHION_CAP)")
    a = ap.parse_args()

    ents = ledger_entries(a.date, a.ticker)
    if not ents:
        print(f"  no {a.ticker} entries on {a.date}"); return
    P, df = paths(a.date, a.ticker)
    print(f"  {len(ents)} ledger triggers, {len(P)} {a.ticker} contracts on tape")
    print(f"  cushion cap: {a.cap if a.cap is not None else 'none (bot 0.5/1.5)'}\n")

    for trail in a.trails:
        print(f"{'='*88}")
        print(f"  TRAIL {trail:.0%}   ({a.ticker} {a.date}, sequential guard honoured)")
        print(f"{'='*88}")
        print(f"  {'in':>6} {'out':>6} {'contract':22} {'entry':>7} {'exit':>7} "
              f"{'peak':>7} {'pnl%':>8} {'tag':>6}")
        busy, total, n = -1, 0.0, 0
        for e in ents:
            if e["mod"] < busy:
                continue                  # position still open -- guard blocks
            cid = match_contract(e["contract"], df)
            if cid is None or cid not in P:
                print(f"  {e['mod']//60:02d}:{e['mod']%60:02d}  -- no tape for "
                      f"{e['contract']}")
                continue
            xm, xp, pk, tag = run_trail(P[cid], e["mod"], e["entry"], trail, a.cap)
            pnl = (xp / e["entry"] - 1.0) * 100
            total += pnl
            n += 1
            busy = xm
            print(f"  {e['mod']//60:02d}:{e['mod']%60:02d} "
                  f"{xm//60:02d}:{xm%60:02d} {e['contract']:22} "
                  f"{e['entry']:>7.2f} {xp:>7.2f} {pk:>7.2f} {pnl:>+8.1f} {tag:>6}")
        print(f"\n  {n} trades, total {total:+.1f}%, mean {total/max(n,1):+.1f}%\n")

    print(f"  NOTE: one session, and the entries are the ones a BUGGED run")
    print(f"  produced -- the guard here suppresses the phantom re-entries but")
    print(f"  cannot invent triggers that the working bot might have taken later.")
    print(f"  Read it as what this day looked like, not as evidence on the trail.")


if __name__ == "__main__":
    main()
