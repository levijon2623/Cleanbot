# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_entry_fill.py
===================
WOULD THE BOT'S ENTRY LIMIT ACTUALLY HAVE FILLED?

THE HOLE THIS PLUGS
    Every backtest here assumes the entry fills at
        min(mid + 0.01, ask)                      (bot_runner.py, sim_core)
    That is a RESTING buy below the ask on any market wider than 2 cents. The
    simulator fills it unconditionally. Whether the market ever came down to it
    is a question nobody has asked, and it is the one remaining unmeasured leg
    of the fill model -- the exit side was calibrated from NBBO depth on
    2026-09-16 (sim_core.CUSHION_CAP), the entry side never was.

WHY THIS ESCAPES THE SURVIVORSHIP TRAP THAT LIMITS THE EXIT WORK
    The exit calibration can only see orders that FILLED -- a resting seller who
    never got hit leaves no print, so the measured cushion is biased toward zero.
    This test has no such problem, because it never asks "did our order fill".
    It asks "did the market's own offer come down to our price", which is fully
    observable in the NBBO path whether or not anyone traded there. The bot is
    in DRY_RUN, so there is no order to be survivor-biased about.

TWO PIECES OF EVIDENCE, reported separately because they differ in strength
    ask_touch    the NBBO ask fell to <= our limit. If we are the best bid at
                 that price the book is crossed and a trade occurs. STRONGEST.
    print_at     some trade printed at <= our limit. Slightly weaker: it shows a
                 seller accepted that price, but we might have been behind
                 others in the queue at the same price.
    Where depth exists (2026-09-01+), the resting bid size at our limit is shown
    too -- that is the queue we would have joined the back of.

🚨 THE WINDOW IS THE WHOLE ANSWER
    bot_runner._await_fill uses timeout=10.0 seconds. A resting bid below the
    ask has a very different fill probability in 10 seconds than "sometime that
    session", so a single number would be meaningless. Every window is reported.
    Read the 10s column as what the LIVE bot would get and the EOD column as the
    upper bound the simulator is implicitly assuming.

Usage:
  python check_entry_fill.py                    # the paper ledger
  python check_entry_fill.py --windows 10 30 60 300
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import polars as pl

LOG = "bot_executions_log.jsonl"
BRONZE = "lake/bronze/full-tape"
OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def parse_occ(sym):
    m = OCC.match(str(sym or ""))
    if not m:
        return None
    tk, ymd, cp, strike = m.groups()
    return dict(ticker=tk,
                expiry=dt.date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6])),
                option_type="call" if cp == "C" else "put",
                strike=int(strike) / 1000.0)


def entries():
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
        p = parse_occ(r.get("contract"))
        if not p:
            continue
        ts = pd.Timestamp(r["timestamp"])
        out.append(dict(**p, ts=ts, date=ts.date(),
                        limit=float(r.get("entry_price") or 0),
                        mid=float(r.get("entry_mid") or 0),
                        rule=r.get("regime"), artifact=bool(r.get("sequence_artifact"))))
    return pd.DataFrame(out)


def tape_for(date, spec):
    p = os.path.join(BRONZE, f"{date.isoformat()}.parquet")
    if not os.path.exists(p):
        return None
    df = (pl.scan_parquet(p)
          .filter((pl.col("underlying_symbol") == spec["ticker"])
                  & (pl.col("option_type") == spec["option_type"])
                  & (pl.col("strike").cast(pl.Float64) == spec["strike"])
                  & (pl.col("expiry").cast(pl.Date) == spec["expiry"]))
          .select("executed_at", "price", "nbbo_bid", "nbbo_ask",
                  "nbbo_bid_size")
          .collect())
    if df.is_empty():
        return None
    return df.with_columns(
        pl.col("price").cast(pl.Float64), pl.col("nbbo_ask").cast(pl.Float64),
        pl.col("nbbo_bid").cast(pl.Float64),
    ).sort("executed_at").to_pandas()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="*", type=int, default=[10, 30, 60, 300])
    a = ap.parse_args()

    E = entries()
    if E.empty:
        print("  no parseable entries"); return
    have = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(BRONZE, "*.parquet"))}
    E["in_lake"] = E["date"].astype(str).isin(have)
    print(f"  {len(E)} ledger entries, {E['in_lake'].sum()} on dates the lake covers")
    print(f"  dates: {sorted(E['date'].astype(str).unique())}\n")

    rows = []
    for (d, tk, ot, st, ex), g in E[E["in_lake"]].groupby(
            ["date", "ticker", "option_type", "strike", "expiry"]):
        T = tape_for(d, dict(ticker=tk, option_type=ot, strike=st, expiry=ex))
        if T is None or T.empty:
            print(f"    {tk} {ex} {st} {ot} on {d}: no tape rows")
            continue
        tt = pd.to_datetime(T["executed_at"], utc=True)
        for r in g.itertuples():
            t0 = pd.Timestamp(r.ts).tz_convert("UTC")
            lim = r.limit
            rec = dict(rule=r.rule, ts=r.ts, contract=f"{tk}{ex:%y%m%d}"
                                                    f"{'C' if ot=='call' else 'P'}"
                                                    f"{int(st*1000):08d}",
                       limit=lim, mid=r.mid, artifact=r.artifact)
            for w in a.windows:
                m = (tt >= t0) & (tt <= t0 + pd.Timedelta(seconds=w))
                sub = T[m.to_numpy()]
                # NO PRINTS IN THE WINDOW IS A NON-FILL, NOT A MISSING VALUE.
                # A resting buy fills only if someone SELLS to it; if nothing
                # traded at any price, nobody did. Recording NaN here was a real
                # bug: bool(np.nan) is True, so empty windows rendered as "Y" in
                # the per-entry table, and the headline used .dropna() so they
                # were dropped from the denominator entirely -- inflating the
                # fill rate by scoring only the moments that were liquid enough
                # to print. SMH 2026-09-10 11:13 showed "Y" at 10s with zero
                # prints and a true first touch 6,912 seconds later.
                rec[f"ask{w}"] = bool((sub["nbbo_ask"] <= lim + 1e-9).any()) \
                    if len(sub) else False
                rec[f"prt{w}"] = bool((sub["price"] <= lim + 1e-9).any()) \
                    if len(sub) else False
            # time to the first ask touch, all session
            after = T[(tt >= t0).to_numpy()]
            hit = after[after["nbbo_ask"] <= lim + 1e-9]
            rec["t_ask_s"] = ((pd.to_datetime(hit["executed_at"].iloc[0], utc=True)
                               - t0).total_seconds() if len(hit) else np.nan)
            rec["eod_ask"] = bool(len(hit) > 0)
            # queue we would join: resting bid size at/below our limit
            near = after[(after["nbbo_bid"] >= lim - 1e-9)]
            rec["q_at_limit"] = (pd.to_numeric(near["nbbo_bid_size"],
                                               errors="coerce").median()
                                 if len(near) else np.nan)
            rows.append(rec)

    if not rows:
        print("  nothing scored"); return
    R = pd.DataFrame(rows)
    R.to_parquet("_entry_fill.parquet", index=False)

    print(f"\n{'='*100}")
    print(f"  WOULD THE ENTRY LIMIT HAVE FILLED?   n={len(R)} paper entries")
    print(f"  ask_touch = the offer came down to our limit (strong)")
    print(f"  print_at  = a trade printed at/below our limit (weaker: queue)")
    print(f"{'='*100}")
    print(f"  {'window':>10} {'ask_touch':>12} {'print_at':>11}")
    for w in a.windows:
        ac = R[f"ask{w}"].dropna()
        pc = R[f"prt{w}"].dropna()
        print(f"  {str(w)+'s':>10} {ac.mean()*100:>11.0f}% {pc.mean()*100:>10.0f}%")
    print(f"  {'EOD':>10} {R['eod_ask'].mean()*100:>11.0f}% {'':>11}")
    tt = R["t_ask_s"].dropna()
    if len(tt):
        print(f"\n  time to first ask touch: median {tt.median():.0f}s   "
              f"p25 {tt.quantile(.25):.0f}s   p75 {tt.quantile(.75):.0f}s")
    q = R["q_at_limit"].dropna()
    if len(q):
        print(f"  resting bid size at our limit: median {q.median():.0f} contracts "
              f"-- the queue ahead of us")

    print(f"\n  PER ENTRY")
    print(f"  {'time':19} {'contract':22} {'limit':>6} {'mid':>6} "
          f"{'10s':>5} {'60s':>5} {'EOD':>5} {'t_fill':>9}")
    for r in R.sort_values("ts").itertuples():
        f10 = "Y" if r.ask10 is True else "n"
        f60 = "Y" if getattr(r, "ask60", None) is True else "n"
        fe = "Y" if r.eod_ask else "n"
        t = f"{r.t_ask_s:.0f}s" if np.isfinite(r.t_ask_s) else "never"
        print(f"  {str(r.ts)[:19]} {r.contract:22} {r.limit:>6.2f} {r.mid:>6.2f} "
              f"{f10:>5} {f60:>5} {fe:>5} {t:>9}")

    print(f"\n  HOW TO READ IT")
    print(f"  The 10s column is what the LIVE bot gets (bot_runner._await_fill")
    print(f"  timeout=10.0). The EOD column is what the SIMULATOR assumes, since it")
    print(f"  fills every candidate unconditionally. The gap between them is the")
    print(f"  size of the entry-side optimism in every backtest in this project.")
    print(f"  A low 10s number does NOT mean the edge is fake -- it means the bot")
    print(f"  takes fewer trades than the sim thinks, and WHICH ones it misses")
    print(f"  (the ones that ran away) is a separate and worse problem.")


if __name__ == "__main__":
    main()
