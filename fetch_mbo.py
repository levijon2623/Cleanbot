# /// script
# requires-python = ">=3.11"
# dependencies = ["databento", "python-dotenv", "polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
fetch_mbo.py
============
Buy CME MBO history for EXACTLY the days our index rules traded -- nothing more.

WHY TARGETED, NOT THE WHOLE WINDOW
----------------------------------
The full 2024-08-20..2026-08-21 window for ES+NQ+RTY is $1,675 and ~1 TB. But
our rules do not trade most days: SPY fires on 5 days, QQQ on 14, IWM on 44 (at
most 63 distinct dates). Pulling only those, with each rule matched to its CME
proxy, costs roughly $60 -- inside the $125 of free credits on a new account.

  SPY -> ES.c.0     QQQ -> NQ.c.0     IWM -> RTY.c.0

Those three are deliberate: SPY / QQQ / IWM are the ONLY rules that came out
ROBUST in `check_fill_sensitivity` (1-2% spreads, 7-15pp fill bands against
37-75% edges). ES/NQ/RTY are single-venue CME books, so MBO there is COMPLETE --
the equity-fragmentation objection does not apply to the index complex.

🚨 WHOLE UTC DAYS, NEVER RTH SLICES
-----------------------------------
Databento emits a synthetic snapshot of the full order book at UTC midnight.
Start a request mid-session and every order resting from before that point was
never seen as NEW -- so queue position, book state, and iceberg refill are
unreconstructable for precisely the large resting orders we care about. RTH
slicing saves ~28% and CORRUPTS THE FEATURES SILENTLY, which is the worst
possible failure. This script always requests midnight-to-midnight.

COST IS QUOTED BEFORE ANYTHING IS BOUGHT
----------------------------------------
Metadata calls are free. The script prints the exact cost of the exact date list
and REFUSES TO DOWNLOAD without `--confirm`. Downloads are resumable: a date
whose file already exists is skipped, so an interrupted run costs nothing extra.

WHAT WE'LL DO WITH IT (so the date choice makes sense)
------------------------------------------------------
First test is winners-vs-losers WITHIN traded days -- we already have per-trade
P&L, so no control days are needed for a first cut (`--controls 0`, the default).
That keeps the first purchase minimal. Control days (signal-quiet days) can be
added later with `--controls N` if the within-day comparison shows anything.

Usage:
  python fetch_mbo.py                      # price only, buys nothing
  python fetch_mbo.py --confirm            # actually download
  python fetch_mbo.py --rules "IWM HIVOL CALL" --confirm
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DATASET = "GLBX.MDP3"
OUTDIR = os.path.join("lake", "mbo")

#: rule ticker -> CME front-month continuous proxy
PROXY = {"SPY": "ES.c.0", "QQQ": "NQ.c.0", "IWM": "RTY.c.0"}


def _key():
    k = os.getenv("DB_KEY") or os.getenv("DATABENTO_API_KEY")
    if not k:
        sys.exit("set DB_KEY (or DATABENTO_API_KEY) in .env")
    return k


def traded_dates(rule_filter=None):
    """{proxy_symbol: sorted[date]} for the days each index rule actually traded.

    Uses sim_core so the dates match the deployed book exactly -- same gates,
    same sequential guard, same `bot` fill model as check_book_now.
    """
    import directional_flow_backtester as D
    import sim_core
    from config import RULES, TRAIL_PCT

    out = collections.defaultdict(set)
    per_rule = {}
    for r in RULES:
        if not r.get("enabled", True):
            continue
        tk = r["ticker"]
        if tk not in PROXY:
            continue
        if rule_filter and r["name"] not in rule_filter:
            continue
        cand = sim_core.build_candidates(D, r)
        if not cand:
            continue
        rows = sim_core.walk(cand, sim_core.policy_for(r, TRAIL_PCT),
                             sim_core.eod_mod(r), fill="bot")
        ds = sorted({d for d, _ in rows})
        per_rule[r["name"]] = (tk, PROXY[tk], len(rows), ds)
        out[PROXY[tk]].update(ds)
        print(f"  {r['name']:22} {tk}->{PROXY[tk]:9} {len(rows):>4} trades on {len(ds):>3} days")
    return {k: sorted(v) for k, v in out.items()}, per_rule


def slice_spread(dates):
    from check_config_walkforward import _slice_idx
    c = collections.Counter()
    for d in dates:
        k = _slice_idx(d)
        if k is not None:
            c[k + 1] += 1
    return dict(sorted(c.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true",
                    help="actually download. WITHOUT THIS NOTHING IS PURCHASED.")
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--controls", type=int, default=0,
                    help="extra non-traded days per symbol (default 0 -- the first "
                         "test is winners-vs-losers within traded days, which needs none)")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--budget", type=float, default=125.0,
                    help="refuse to download above this cost (default = the free credits)")
    a = ap.parse_args()

    import databento as db
    import pandas as pd

    print("=" * 84)
    print("  DAYS OUR INDEX RULES ACTUALLY TRADED")
    print("=" * 84)
    by_sym, per_rule = traded_dates(a.rules)
    if not by_sym:
        sys.exit("  no dates found")

    if a.controls:
        # non-traded weekdays drawn evenly across the same span
        for sym, ds in list(by_sym.items()):
            lo, hi = min(ds), max(ds)
            allw = [d.date() for d in pd.bdate_range(lo, hi)]
            spare = [d for d in allw if d not in set(ds)]
            step = max(1, len(spare) // max(a.controls, 1))
            by_sym[sym] = sorted(set(ds) | set(spare[::step][:a.controls]))
        print(f"\n  + {a.controls} control day(s) per symbol")

    print("\n" + "=" * 84)
    print("  COST OF EXACTLY THESE DAYS  (metadata is free; nothing bought yet)")
    print("  whole UTC days -- see the midnight-snapshot warning in the header")
    print("=" * 84)
    c = db.Historical(_key())
    total, plan = 0.0, []
    for sym, ds in sorted(by_sym.items()):
        sym_cost, sym_gb = 0.0, 0.0
        for d in ds:
            s = f"{d}T00:00"
            e = f"{pd.Timestamp(d) + pd.Timedelta(days=1):%Y-%m-%d}T00:00"
            try:
                cost = float(c.metadata.get_cost(dataset=DATASET, symbols=[sym],
                                                 schema="mbo", start=s, end=e,
                                                 stype_in="continuous"))
                size = float(c.metadata.get_billable_size(dataset=DATASET, symbols=[sym],
                                                          schema="mbo", start=s, end=e,
                                                          stype_in="continuous"))
            except Exception as ex:
                print(f"    ! {sym} {d}: {type(ex).__name__}: {str(ex)[:60]}")
                continue
            sym_cost += cost
            sym_gb += size / 1e9
            plan.append((sym, d, s, e, cost))
        total += sym_cost
        print(f"  {sym:9} {len(ds):>3} days   ${sym_cost:>7.2f}   {sym_gb:>7.1f} GB   "
              f"slices {slice_spread(ds)}")

    print(f"\n  {'TOTAL':9} {len(plan):>3} day-pulls  ${total:>7.2f}")
    if total > a.budget:
        print(f"  ⚠️  over the ${a.budget:.2f} budget — narrow with --rules, or raise --budget")

    if not a.confirm:
        print("\n  DRY RUN — nothing purchased. Re-run with --confirm to download.")
        return
    if total > a.budget:
        sys.exit("  refusing to download over budget.")

    print("\n" + "=" * 84)
    print("  DOWNLOADING (resumable — existing files are skipped)")
    print("=" * 84)
    done = spent = 0
    for sym, d, s, e, cost in plan:
        p = os.path.join(a.outdir, sym.replace(".", "_"), f"date={d}")
        f = os.path.join(p, "mbo.dbn.zst")
        if os.path.exists(f) and os.path.getsize(f) > 0:
            print(f"  ·  {sym} {d}  already have it")
            continue
        os.makedirs(p, exist_ok=True)
        try:
            data = c.timeseries.get_range(dataset=DATASET, symbols=[sym], schema="mbo",
                                          start=s, end=e, stype_in="continuous")
            data.to_file(f)
            done += 1
            spent += cost
            print(f"  ✓  {sym} {d}  ${cost:.2f}  {os.path.getsize(f)/1e6:>8.1f} MB"
                  f"   (spent ${spent:.2f})")
        except Exception as ex:
            print(f"  ✗  {sym} {d}  {type(ex).__name__}: {str(ex)[:70]}")
    print(f"\n  downloaded {done} day-pulls, ~${spent:.2f}")


if __name__ == "__main__":
    main()
