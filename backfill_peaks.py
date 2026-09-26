# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "pandas>=2.0.0", "python-dotenv", "httpx"]
# ///
"""
backfill_peaks.py
=================
Reconstructs the PEAK (max favourable excursion) for log entries that closed
before bot_runner started recording it, and amends the ledger in place.

WHY IT WAS MISSING
    `peak_bid` was trail STATE and was only updated when `trail_pct > 0`, so the
    static-bracket rules (NVDA, META) tracked no peak at all and printed a bare
    "STOP LOSS" / "TAKE PROFIT". Fixed 2026-09-14 by tracking `max_bid_seen`
    separately; entries closed before that deploy have no peak and this fills
    them.

WHY THE FULL TAPE AND NOT THE INTRADAY ENDPOINT
    `/option-contract/{occ}/intraday` gives 1-minute TRADE OHLC. Its
    `premium_bid_side`/`volume_bid_side` fields are AGGRESSOR-side volume -- how
    much traded at the bid -- NOT the NBBO bid quote. A peak taken from the
    trade `high` would therefore overstate what the bot could actually have sold
    into, since prints sit between the bid and the ask. The full tape carries
    `nbbo_bid` on every print, which is the same quantity `max_bid_seen` tracks
    live, so it is the only basis that makes the backfilled number comparable to
    the ones the bot records itself.

INTEGRITY
    * the ledger is COPIED to bot_executions_log.jsonl.bak-<date> first
    * reconstructed values go in their own fields (`peak_bid_reconstructed`,
      `peak_roe_reconstructed`, `peak_source`) and the `reason` string is NOT
      rewritten -- a backfilled number must never be indistinguishable from one
      the bot logged live
    * only records that already LACK a peak are touched; anything with a peak
      in its reason is left alone
    * bronze is deleted afterwards, per the established download->silver->delete
      practice

Usage:
  python backfill_peaks.py --date 2026-09-14 --dry-run
  python backfill_peaks.py --date 2026-09-14 --apply
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil

import pandas as pd
import polars as pl

LOG = "bot_executions_log.jsonl"


def load_log():
    """[(raw_line, parsed_or_None, trailing_note)] preserving file order."""
    out = []
    with open(LOG, encoding="utf-8-sig") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
                out.append((s, None, ""))
                continue
            cut = s.rfind("}")
            try:
                rec = json.loads(s[:cut + 1])
            except Exception:
                out.append((s, None, ""))
                continue
            out.append((s, rec, s[cut + 1:]))
    return out


def round_trips(rows, day):
    """[(exit_index, ticker, occ, entry_ts, exit_ts, entry_px)] for `day`."""
    open_by = {}
    trips = []
    for i, (_, rec, _) in enumerate(rows):
        if rec is None:
            continue
        ts = pd.Timestamp(rec.get("timestamp"))
        if ts.date() != day:
            continue
        act = str(rec.get("action", ""))
        tk = rec.get("ticker")
        if act.startswith("ENTRY"):
            open_by[tk] = (ts, rec.get("contract"), rec.get("price"))
        elif act == "EXIT" and tk in open_by:
            ent_ts, occ, ent_px = open_by.pop(tk)
            trips.append((i, tk, occ or rec.get("contract"), ent_ts, ts, ent_px))
    return trips


def bid_paths(day, occs):
    """{occ: DataFrame(executed_at, nbbo_bid)} from the full tape.

    Reads the extracted CSV directly rather than going through
    `uw_options_data_lake.build_one`. That path validates the header against a
    fixed 40-column schema and UW has since widened the tape to 49 -- it now
    also ships exchange_id, nbbo_{bid,ask}_{exchange_id,size,time,exchange}.
    Every original column is still present, so the addition is harmless here,
    but the strict-equality check rejects it. (The BUILDER still needs fixing
    separately or it will fail on every future session.)
    """
    from uw_options_data_lake import (Client, bronze_path, work_dir,
                                      DEFAULT_LAKE)
    from dotenv import load_dotenv
    load_dotenv(encoding="utf-8-sig")

    csv = work_dir(DEFAULT_LAKE) / f"{day.isoformat()}-option_trades.csv"
    bp = bronze_path(DEFAULT_LAKE, day)
    if bp.exists():
        src = pl.scan_parquet(bp)
    elif csv.exists():
        print(f"  reading extracted tape {csv.name} "
              f"({csv.stat().st_size/1e9:.2f} GB)", flush=True)
        src = pl.scan_csv(csv, infer_schema_length=50_000,
                          schema_overrides={"nbbo_bid": pl.Float64,
                                            "price": pl.Float64,
                                            "size": pl.Float64})
    else:
        raise SystemExit(
            f"no tape for {day}. Download it first:\n"
            f"    python uw_options_data_lake.py build {day} --confirm\n"
            f"(that will currently fail schema validation -- see the note in "
            f"bid_paths)")

    df = (src.filter(pl.col("option_chain_id").is_in(list(occs))
                     & (pl.col("canceled") == "f")
                     & (pl.col("nbbo_bid") > 0))
          .select("option_chain_id", "executed_at", "nbbo_bid", "price", "size")
          .collect().to_pandas())
    out = {}
    for occ, g in df.groupby("option_chain_id"):
        g = g.copy()
        g["executed_at"] = pd.to_datetime(g["executed_at"], utc=True)
        out[occ] = g.sort_values("executed_at")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.date)

    rows = load_log()
    trips = round_trips(rows, day)
    need = [t for t in trips if "peak" not in str(rows[t[0]][1].get("reason", ""))
            and rows[t[0]][1].get("peak_bid_reconstructed") is None]
    print(f"  {len(trips)} round trips on {day}; {len(need)} lack a peak")
    for _, tk, occ, e, x, px in need:
        print(f"    {tk} {occ}  {e.strftime('%H:%M:%S')} -> {x.strftime('%H:%M:%S')}"
              f"  entry ${px}")
    if not need:
        return

    paths = bid_paths(day, {t[2] for t in need})
    updates = {}
    for idx, tk, occ, ent, exi, ent_px in need:
        g = paths.get(occ)
        if g is None or g.empty:
            print(f"    {occ}: no tape rows"); continue
        w = g[(g["executed_at"] >= ent) & (g["executed_at"] <= exi)]
        if w.empty:
            print(f"    {occ}: no prints in window"); continue
        pk = float(w["nbbo_bid"].max())
        at = w.loc[w["nbbo_bid"].idxmax(), "executed_at"]
        roe = (pk / float(ent_px) - 1.0) * 100 if ent_px else None
        updates[idx] = dict(peak_bid_reconstructed=round(pk, 4),
                            peak_roe_reconstructed=(round(roe, 2) if roe is not None else None),
                            peak_at_reconstructed=at.tz_convert("America/New_York").isoformat(),
                            peak_source="full-tape nbbo_bid (backfilled "
                                        f"{dt.date.today()}; NOT logged live)")
        print(f"    {tk} {occ}: peak bid ${pk:.2f} at "
              f"{at.tz_convert('America/New_York').strftime('%H:%M:%S')}  "
              f"ROE {roe:+.1f}%  (exit was {rows[idx][1].get('pnl_pct')}%)")

    if not a.apply:
        print("\n  DRY RUN -- nothing written. Re-run with --apply.")
        return

    bak = f"{LOG}.bak-{dt.date.today()}"
    shutil.copy2(LOG, bak)
    print(f"\n  backed up -> {bak}")
    with open(LOG, "w", encoding="utf-8", newline="\n") as f:
        for i, (raw, rec, note) in enumerate(rows):
            if rec is None or i not in updates:
                f.write(raw + "\n")
                continue
            rec.update(updates[i])
            f.write(json.dumps(rec) + (note if note else "") + "\n")
    print(f"  amended {len(updates)} record(s) in {LOG}")
    print("  `reason` left untouched -- reconstructed values are in their own "
          "fields so they stay distinguishable from live-logged peaks.")


if __name__ == "__main__":
    main()
