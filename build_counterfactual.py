# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
build_counterfactual.py
=======================
Emit `bot_executions_counterfactual.jsonl` -- what the ledger WOULD have held if
the zero-bid exit bug had never fired. Explicitly part fiction, and labelled so.

WHY THIS IS A SEPARATE FILE AND NOT AN EDIT
    The execution log is the only OBSERVED record in this project. Everything
    else -- 716 days of backtest -- is simulation. Its unique value is that it
    can FALSIFY the simulator, which it did twice: it exposed the phantom-quote
    bug, and the one real AVGO trade of 2026-09-15 revealed the exit cushion was
    mispricing winners by 9.0pp. Correcting the log with the replay would make
    the record an output of the model it exists to check. That is circular, and
    the circularity is invisible once the file is written.
    So: the log stays observational and append-only. This file is derived, and
    every row says which it is.

EVERY ROW CARRIES PROVENANCE
    source       "observed"      copied verbatim from the log
                 "reconstructed" produced by replaying the tape
    fiction      false / true    the blunt version of the same thing
    basis        what the reconstruction assumed, per row

WHAT THE RECONSTRUCTION ASSUMES, and each of these could be wrong
  * config.BAD_TICK_* would have suppressed the phantom exits (it is new code
    that has not yet run in anger).
  * The sequential guard blocks re-entry until the position actually closes.
  * Silver minute bars (bid_close/ask_close) are the bid path the bot saw.
  * _fire_exit's cushion prices the exit. `--cap` applies the calibrated
    sim_core.CUSHION_CAP instead, which the 2026-09-16 depth work argues for.

🚨 THE ONE THING IT CANNOT DO
    It cannot invent triggers. The bugged run was almost never flat, so the
    working bot might have taken LATER triggers that never appear in the log.
    A reconstructed session is therefore a LOWER BOUND on activity, not the
    true counterfactual. Do not read a quiet reconstructed day as evidence the
    rule was quiet.

Usage:
  python build_counterfactual.py
  python build_counterfactual.py --cap        # use sim_core.CUSHION_CAP
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

import sim_core
from replay_day import ledger_entries, paths, match_contract, run_trail

LOG = "bot_executions_log.jsonl"
OUT = "bot_executions_counterfactual.jsonl"


def would_fill(date, contract, ts, limit, window_s):
    """Would a resting BUY at `limit` have filled within `window_s`?

    The order fills only if someone SELLS to it, i.e. the NBBO ask comes down to
    our price. No prints in the window means nobody sold at any price, which is
    a NON-fill -- not missing data. (check_entry_fill originally recorded that
    case as NaN, which rendered as a fill and was dropped from the denominator;
    correcting it moved the 10s fill rate from 57% to 27%.)
    -> (filled: bool, seconds_to_touch: float | None)
    """
    import re
    import polars as pl
    m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", str(contract or ""))
    p = f"lake/bronze/full-tape/{date}.parquet"
    if not m or not os.path.exists(p):
        return None, None                  # unknowable, not a fill claim
    tk, ymd, cp, strike = m.groups()
    exp = dt.date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    df = (pl.scan_parquet(p)
          .filter((pl.col("underlying_symbol") == tk)
                  & (pl.col("option_type") == ("call" if cp == "C" else "put"))
                  & (pl.col("strike").cast(pl.Float64) == int(strike) / 1000.0)
                  & (pl.col("expiry").cast(pl.Date) == exp))
          .select("executed_at", "nbbo_ask").collect().to_pandas())
    if df.empty:
        return None, None
    et = pd.to_datetime(df["executed_at"], utc=True)
    ask = pd.to_numeric(df["nbbo_ask"], errors="coerce")
    t0 = pd.Timestamp(ts).tz_convert("UTC")
    win = (et >= t0) & (et <= t0 + pd.Timedelta(seconds=window_s))
    filled = bool((ask[win] <= limit + 1e-9).any()) if win.any() else False
    after = (et >= t0) & (ask <= limit + 1e-9)
    secs = float((et[after].iloc[0] - t0).total_seconds()) if after.any() else None
    return filled, secs


def load_raw():
    out = []
    for line in open(LOG, encoding="utf-8-sig"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        c = s.rfind("}")
        try:
            out.append(json.loads(s[:c + 1]))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    # The calibrated cap is now the DEFAULT, because as of 2026-09-18 it is what
    # the live bot actually does -- _fire_exit was moved off the legacy
    # profitability-keyed 0.5/1.5 onto the measured per-ticker CUSHION_CAP. A
    # reconstruction priced by a rule the bot no longer runs would be fiction
    # about the wrong bot. `--cap` is kept as an accepted no-op so existing
    # commands and notes keep working; `--legacy-cushion` restores the old
    # pricing for comparing against reconstructions built before the change.
    ap.add_argument("--cap", action="store_true",
                    help="(deprecated no-op) the calibrated cap is now default")
    ap.add_argument("--legacy-cushion", action="store_true",
                    help="price exits with the RETIRED 0.5/1.5 rule instead of "
                         "config.CUSHION_CAP -- only for reproducing older runs")
    ap.add_argument("--window", type=int, default=10,
                    help="seconds a resting entry may wait (bot_runner._await_fill "
                         "uses timeout=10.0)")
    a = ap.parse_args()

    recs = load_raw()
    # sessions needing reconstruction = those carrying artifact entries
    bad = sorted({(r.get("ticker"), str(r.get("timestamp"))[:10])
                  for r in recs if r.get("sequence_artifact")})
    print(f"  {len(recs)} ledger records; {len(bad)} session(s) to reconstruct:")
    for tk, d in bad:
        print(f"    {tk} {d}")

    out, n_obs, n_rec = [], 0, 0
    for r in recs:
        key = (r.get("ticker"), str(r.get("timestamp"))[:10])
        if key in bad:
            continue                       # replaced wholesale below
        q = dict(r)
        q["source"] = "observed"
        q["fiction"] = False
        out.append(q)
        n_obs += 1

    # 🚨 A SESSION THAT CANNOT BE REPLAYED MUST NOT VANISH (2026-09-18).
    # Flagged sessions are held out of the observed copy above on the promise
    # that the replay below replaces them. When the replay cannot run -- UW
    # publishes the full tape a day late, so today's session has no silver
    # partition until tomorrow -- that promise silently breaks and the day
    # disappears from the ledger entirely. It did: a rebuild dropped all 16 GLD
    # rows for 2026-09-18 and still printed "wrote ...", with the reason one
    # `!` line further up the scroll.
    # A file with a hole in it is worse than one with an uncorrected day, because
    # the hole is invisible downstream. So failures fall back to the OBSERVED
    # rows, tagged `replay_failed` so nothing mistakes them for corrected.
    failed = {}
    for tk, d in bad:
        ents = ledger_entries(d, tk)
        if not ents:
            failed[(tk, d)] = "no ledger entries for the session"
            continue
        try:
            P, df = paths(d, tk)
        except SystemExit as e:
            print(f"    ! {tk} {d}: {e}")
            failed[(tk, d)] = str(e)
            continue
        cap = None if a.legacy_cushion else sim_core.CUSHION_CAP.get(tk)
        busy = -1
        for e in ents:
            if e["mod"] < busy:
                continue
            cid = match_contract(e["contract"], df)
            if cid is None or cid not in P:
                continue
            # ENTRY FILL GATE. The bug this file corrects is an EXIT bug, but a
            # counterfactual of live behaviour must also respect that a resting
            # buy below the ask often never fills: at the live 10s timeout only
            # 27% of paper entries would have. Reconstructing a trade whose
            # entry could not have happened would be fiction of the wrong kind.
            ok, secs = would_fill(d, e["contract"], e["ts"], e["entry"], a.window)
            if ok is False:
                out.append(dict(
                    ticker=tk, contract=e["contract"], regime=e.get("rule"),
                    action="ENTRY_UNFILLED", timestamp=pd.Timestamp(e["ts"]).isoformat(),
                    price=e["entry"], source="reconstructed", fiction=True,
                    basis=(f"resting buy at {e['entry']:.2f} -- the NBBO ask did "
                           f"not reach it within {a.window}s"
                           + (f"; first touch {secs:.0f}s later" if secs else
                              "; never touched that session")),
                    caveat="no trade reconstructed: the live bot would not have filled"))
                continue
            trail = float(e["trail"] or 0.50)
            xm, xp, pk, tag = run_trail(P[cid], e["mod"], e["entry"], trail, cap)
            busy = xm
            base = dict(ticker=tk, contract=e["contract"], regime=e.get("rule"),
                        source="reconstructed", fiction=True,
                        basis=("replayed from lake/silver minute bars with the "
                               "bad-tick guard and the sequential guard applied; "
                               f"trail={trail:.0%}, cushion="
                               f"{'CUSHION_CAP ' + str(cap) if cap else 'bot 0.5/1.5'}"),
                        caveat=("triggers come from the BUGGED run, which was "
                                "rarely flat -- later triggers the working bot "
                                "might have taken cannot be recovered"))
            ts = pd.Timestamp(e["ts"])
            out.append({**base, "action": "ENTRY_SHORT",
                        "timestamp": ts.isoformat(),
                        "price": e["entry"], "entry_price": e["entry"],
                        "net_flow": e.get("flow"), "dry_run": True})
            xts = ts.normalize() + pd.Timedelta(minutes=int(xm))
            out.append({**base, "action": "EXIT",
                        "timestamp": xts.isoformat(),
                        "price": xp, "exit_price": xp, "entry_price": e["entry"],
                        "pnl_pct": round((xp / e["entry"] - 1) * 100, 2),
                        "pnl_dollars": round((xp - e["entry"]) * 100, 2),
                        "peak_bid": pk, "reason": f"{tag.upper()} (replay)",
                        "entry_time": ts.isoformat(), "dry_run": True})
            n_rec += 2

    # Sessions the replay could not produce: put the OBSERVED rows back rather
    # than leave a hole, tagged so no consumer reads them as corrected.
    n_fail = 0
    for r in recs:
        key = (r.get("ticker"), str(r.get("timestamp"))[:10])
        if key not in failed:
            continue
        q = dict(r)
        q["source"] = "observed"
        q["fiction"] = False
        q["replay_failed"] = failed[key]
        q["basis"] = "UNCORRECTED -- session flagged for replay, replay unavailable"
        out.append(q)
        n_fail += 1

    out.sort(key=lambda r: str(r.get("timestamp")))
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  wrote {OUT}")
    print(f"    {n_obs} observed rows copied, {n_rec} reconstructed rows")
    if failed:
        print(f"\n  ⚠️  {len(failed)} SESSION(S) COULD NOT BE REPLAYED "
              f"({n_fail} rows kept UNCORRECTED):")
        for (tk, d), why in sorted(failed.items()):
            print(f"      {tk} {d}: {why}")
        print(f"      These rows carry `replay_failed` and are still the raw,")
        print(f"      distorted record. Re-run once the data exists -- UW")
        print(f"      publishes the full tape a day late, so a same-day session")
        print(f"      is never replayable until the following morning.")
    rec = [r for r in out if r.get("fiction")]
    if rec:
        ex = [r for r in rec if r.get("action") == "EXIT"]
        print(f"    reconstructed round-trips: {len(ex)}, "
              f"total {sum(float(r['pnl_pct']) for r in ex):+.1f}%")
        for r in ex:
            print(f"      {r['timestamp'][:19]} {r['ticker']:5} "
                  f"{r['entry_price']:>6.2f} -> {r['exit_price']:>6.2f}  "
                  f"{r['pnl_pct']:>+7.1f}%  {r['reason']}")
    print(f"\n  The log itself is untouched. Consumers that want the observed")
    print(f"  record read {LOG}; anything reading {OUT} is reading part fiction")
    print(f"  and every row says so.")


if __name__ == "__main__":
    main()
