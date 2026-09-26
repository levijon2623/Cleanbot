# /// script
# requires-python = ">=3.11"
# dependencies = ["databento", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
inspect_mbo.py
==============
First look at the downloaded CME MBO. Verifies the data is what we think it is
BEFORE any feature work is built on top of it.

MEMORY DISCIPLINE: a single day is ~400 MB compressed and expands to several GB
as a DataFrame. This ITERATES the DBN store record-by-record and keeps only
counters, so peak memory stays flat regardless of file size. Never call
`.to_df()` on a whole MBO day on this box.

WHAT IT CHECKS, and why each matters
------------------------------------
  * record count + wall-clock span   -- did we get a whole UTC day?
  * the UTC-midnight SNAPSHOT        -- action 'R' (clear) followed by a burst
                                        of 'A' (add) at the session start is the
                                        full-book snapshot. WITHOUT IT the book
                                        cannot be reconstructed, and every
                                        queue/iceberg feature would be garbage
                                        computed on a partial book.
  * action distribution              -- A(dd) / C(ancel) / M(odify) / T(rade) /
                                        F(ill) / R(clear). The cancel-to-trade
                                        ratio is itself a candidate feature.
  * order_id populated               -- the whole point. No order ids means no
                                        lifecycle tracking and no icebergs.
  * distinct instrument_id           -- continuous `.c.0` symbols roll, so a
                                        day may contain more than one contract.

Usage:
  python inspect_mbo.py                       # one sample day per symbol
  python inspect_mbo.py --all                 # every day (slow)
  python inspect_mbo.py --limit 2000000
"""
from __future__ import annotations

import argparse
import collections
import glob
import os

ROOT = os.path.join("lake", "mbo")
# DBN action codes, per Databento's MBO schema
ACTIONS = {"A": "add", "C": "cancel", "M": "modify", "R": "clear",
           "T": "trade", "F": "fill", "N": "none"}


def inspect(path, limit=None):
    import databento as db
    store = db.DBNStore.from_file(path)

    n = 0
    acts = collections.Counter()
    sides = collections.Counter()
    instruments = collections.Counter()
    first_ts = last_ts = None
    with_oid = 0
    first_actions = []          # the opening sequence, to spot the snapshot
    max_oid = 0
    px_min, px_max = None, None

    for rec in store:
        a = getattr(rec, "action", None)
        if a is None:                      # metadata / non-MBO record
            continue
        a = chr(a) if isinstance(a, int) else str(a)
        n += 1
        acts[a] += 1
        s = getattr(rec, "side", None)
        sides[chr(s) if isinstance(s, int) else str(s)] += 1
        instruments[getattr(rec, "instrument_id", None)] += 1
        ts = getattr(rec, "ts_event", None)
        if ts:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        oid = getattr(rec, "order_id", 0) or 0
        if oid:
            with_oid += 1
            max_oid = max(max_oid, oid)
        p = getattr(rec, "price", None)
        if p is not None and p > 0:
            px_min = p if px_min is None else min(px_min, p)
            px_max = p if px_max is None else max(px_max, p)
        if len(first_actions) < 12:
            first_actions.append(a)
        if limit and n >= limit:
            break

    import pandas as pd
    f = pd.Timestamp(first_ts, unit="ns", tz="UTC") if first_ts else None
    l = pd.Timestamp(last_ts, unit="ns", tz="UTC") if last_ts else None
    print(f"\n  {os.path.relpath(path)}")
    print(f"    records            {n:,}{'  (truncated)' if limit and n >= limit else ''}")
    print(f"    span (UTC)         {f}  ->  {l}")
    print(f"    opening actions    {' '.join(first_actions)}")
    snap = "R" in first_actions[:3] or acts.get("R", 0) > 0
    print(f"    midnight snapshot  {'✅ present (clear+adds)' if snap else '⚠️  NO CLEAR RECORD'}")
    print(f"    actions            " + "  ".join(
        f"{k}({ACTIONS.get(k,'?')}):{v:,}" for k, v in acts.most_common()))
    print(f"    sides              {dict(sides)}")
    print(f"    order_id populated {with_oid:,} / {n:,} = {with_oid/max(n,1)*100:.1f}%"
          f"   max_id={max_oid}")
    print(f"    instruments        {len(instruments)} distinct "
          f"{[i for i,_ in instruments.most_common(3)]}")
    if px_min is not None:
        print(f"    price range (raw)  {px_min:,} .. {px_max:,}  "
              f"(DBN fixed-point, 1e-9)")
    ct = acts.get("C", 0) / max(acts.get("T", 0) + acts.get("F", 0), 1)
    print(f"    cancel:trade ratio {ct:,.1f}   <- candidate feature")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="every day, not one sample each")
    ap.add_argument("--limit", type=int, default=1_500_000,
                    help="max records per file (0 = whole file). Keeps memory and "
                         "time bounded; the action mix is stable well before this.")
    a = ap.parse_args()

    for sym in sorted(os.listdir(ROOT)):
        files = sorted(glob.glob(os.path.join(ROOT, sym, "date=*", "*.dbn.zst")))
        if not files:
            continue
        print("\n" + "=" * 78)
        print(f"  {sym}   {len(files)} day files")
        print("=" * 78)
        for p in (files if a.all else [files[0], files[len(files) // 2], files[-1]]):
            try:
                inspect(p, a.limit or None)
            except Exception as e:
                print(f"\n  {os.path.relpath(p)}\n    ! {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
