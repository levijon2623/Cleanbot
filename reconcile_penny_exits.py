# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv", "numpy", "pandas"]
# ///
"""
reconcile_penny_exits.py
========================
Repair the ledger records corrupted by the zero-bid exit bug (config.BAD_TICK_*).

WHAT WENT WRONG
    A glitched quote printed bid <= 0, fell through `bid <= trail_stop`, and
    _fire_exit booked max(0.01, bid - cushion) = $0.01. 18 records: 16 AVGO on
    2026-09-15 and 2 SMH on 2026-09-10, every one an ATM 0/1DTE contract that
    could not possibly have been worth a penny. Logged total: -1793.1%.

WHAT THIS CAN AND CANNOT REPAIR
    CAN:    the exit PRICE and P&L of each record, from the contract's real NBBO
            bid at that timestamp (UW /option-contract/{id}/flow carries
            nbbo_bid per trade -- the same quantity the trail evaluates).
    CANNOT: make the day's trade SEQUENCE real. Each phantom exit freed the
            sequential slot, so the bot re-entered minutes later. Without the
            bug the earlier position would still have been open and the next
            ENTRY would never have fired. The 16 AVGO records are therefore not
            16 independent trades -- they are a handful of positions repeatedly
            killed and re-opened.
    So this script does NOT rewrite history into a clean counterfactual. It
    marks every affected record, corrects the price that was demonstrably wrong,
    and records WHY, leaving the sequence visible for what it was. Silently
    producing a tidy ledger would hide a bug's footprint inside the data used to
    judge the rules.

FIELDS ADDED (originals are never overwritten)
    exit_price_corrupt      the $0.01 that was booked
    pnl_pct_corrupt         the -99.x% that was booked
    exit_price              corrected to the real NBBO bid, minus the same
                            cushion _fire_exit applies
    pnl_pct / pnl_dollars   recomputed from the corrected price
    reconcile_source        endpoint + timestamp of the quote used
    reconcile_note          why the record was touched
    sequence_artifact       true when this record only exists because an earlier
                            phantom exit freed the slot

Usage:
  python reconcile_penny_exits.py            # dry run, prints the diff
  python reconcile_penny_exits.py --apply    # writes, after backing up
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys

import httpx
from dotenv import load_dotenv

from config import CUSHION_CAP, EXIT_FLOOR_FRAC, FILL_COST

LOG = "bot_executions_log.jsonl"
API = "https://api.unusualwhales.com/api"
PENNY = 0.02


def client():
    load_dotenv(".env", encoding="utf-8-sig")
    key = os.getenv("UW_API_KEY")
    if not key:
        sys.exit("  UW_API_KEY not in env")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json",
            "User-Agent": "webullrg-reconcile/1.0", "UW-CLIENT-API-ID": "100003"}


def fetch_quotes(h, contract, date):
    """-> {minute_utc_iso: (bid, ask)} from the contract's own trade tape.

    /flow carries nbbo_bid / nbbo_ask stamped at each execution, which is the
    same quote the bot would have been reading. Aggregated to the minute, last
    value wins (the bot acts on the latest quote it saw).
    """
    out, page, seen = {}, 0, 0
    while True:
        r = httpx.get(f"{API}/option-contract/{contract}/flow", headers=h,
                      params={"date": date, "limit": 500, "page": page}, timeout=60)
        if r.status_code != 200:
            print(f"    ! {contract} {date}: HTTP {r.status_code}")
            break
        d = r.json().get("data", [])
        if not d:
            break
        for t in d:
            ts = t.get("executed_at") or t.get("nbbo_bid_time")
            b, a = t.get("nbbo_bid"), t.get("nbbo_ask")
            if not ts or b is None:
                continue
            key = str(ts)[:16]                     # YYYY-MM-DDTHH:MM
            try:
                out[key] = (float(b), float(a) if a is not None else float(b))
            except (TypeError, ValueError):
                continue
        seen += len(d)
        page += 1
        if len(d) < 500 or page > 40:
            break
    print(f"    {contract} {date}: {seen} trades -> {len(out)} quoted minutes")
    return out


def fetch_intraday(h, contract, date):
    """FALLBACK for contracts /flow will not serve (it returns nothing for an
    EXPIRED contract, which both SMH 0DTE puts are). /intraday still carries
    minute OHLC of TRADES -- no NBBO, so the bid has to be approximated by the
    minute's LOW. That is deliberately conservative (a trade at the low of the
    minute is the closest thing to the bid side available) and no further
    cushion is applied on top. Marked as lower confidence in the record.
    """
    r = httpx.get(f"{API}/option-contract/{contract}/intraday", headers=h,
                  params={"date": date}, timeout=60)
    if r.status_code != 200:
        print(f"    ! {contract} {date}: intraday HTTP {r.status_code}")
        return {}
    out = {}
    for b in r.json().get("data", []):
        ts = b.get("start_time")
        lo, cl = b.get("low"), b.get("close")
        if not ts or lo is None:
            continue
        try:
            out[str(ts)[:16]] = (float(lo), float(cl if cl is not None else lo))
        except (TypeError, ValueError):
            continue
    print(f"    {contract} {date}: intraday fallback -> {len(out)} minutes "
          f"(TRADE prices, not NBBO)")
    return out


def quote_at(q, iso_ts, back=30):
    """Latest quote at or before `iso_ts`, searching back up to `back` minutes."""
    t = dt.datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    t = t.astimezone(dt.timezone.utc)
    for k in range(back + 1):
        key = (t - dt.timedelta(minutes=k)).strftime("%Y-%m-%dT%H:%M")
        if key in q:
            return q[key], k
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    raw = open(LOG, encoding="utf-8-sig").read().splitlines()
    recs = []
    for i, line in enumerate(raw):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        c = s.rfind("}")
        try:
            recs.append((i, json.loads(s[:c + 1]), s[c + 1:]))
        except Exception:
            pass

    bad = [(i, r) for i, r, _ in recs
           if r.get("action") == "EXIT" and r.get("exit_price") is not None
           and float(r["exit_price"]) <= PENNY]
    if not bad:
        print("  nothing to repair"); return
    print(f"  {len(bad)} corrupted exits\n")

    # entries are sequence artifacts when they follow a phantom exit of the
    # same ticker on the same day
    bad_ts = {(r.get("ticker"), str(r.get("timestamp"))[:10]): [] for _, r in bad}
    for _, r in bad:
        bad_ts[(r.get("ticker"), str(r.get("timestamp"))[:10])].append(str(r["timestamp"]))
    for k in bad_ts:
        bad_ts[k].sort()

    h = client()
    need = sorted({(r.get("contract"), str(r.get("timestamp"))[:10]) for _, r in bad})
    print("  fetching real quotes:")
    quotes = {}
    for c, d in need:
        q = fetch_quotes(h, c, d)
        if q:
            quotes[(c, d)] = (q, "nbbo")
        else:
            quotes[(c, d)] = (fetch_intraday(h, c, d), "trade")

    fixes = {}
    print(f"\n  {'time':19} {'tkr':5} {'entry':>7} {'logged':>7} {'REAL bid':>9} "
          f"{'corrected':>10} {'old pnl':>9} {'new pnl':>9} {'lag':>4}")
    tot_old = tot_new = 0.0
    unresolved = []
    for i, r in bad:
        con, date = r.get("contract"), str(r.get("timestamp"))[:10]
        q, kind = quotes.get((con, date), ({}, "nbbo"))
        got, lag = quote_at(q, str(r["timestamp"]))
        ep = float(r.get("entry_price") or 0)
        old = float(r.get("pnl_pct") or 0)
        tot_old += old
        if got is None or ep <= 0:
            unresolved.append((i, r))
            print(f"  {str(r['timestamp'])[:19]} {str(r.get('ticker')):5} {ep:>7.2f} "
                  f"{float(r['exit_price']):>7.2f} {'--':>9} {'UNRESOLVED':>10}")
            continue
        if kind == "nbbo":
            bid, ask = got
            spread = max(0.0, ask - bid)
            # _fire_exit's own arithmetic, mirrored so the repair is priced the
            # way a real fill would have been, not at an optimistic mid.
            # UPDATED 2026-09-18: that arithmetic changed. The legacy
            # profitability-keyed 0.5/1.5 was retired for the measured,
            # tag-keyed per-ticker cap now shared in config -- the very rule
            # whose absence overcharged these exits. Importing the constants
            # keeps ONE cushion model across the bot, the backtester and this
            # repair; mirroring a rule the bot no longer runs would re-create
            # the drift (METHODOLOGY 1).
            adverse = str(r.get("reason", "")).upper()
            adverse = adverse.startswith("TRAIL") or adverse.startswith("STOP LOSS")
            mult = (CUSHION_CAP.get(r.get("ticker"), 1.5) if adverse else FILL_COST)
            floor = max(0.01, round(bid * EXIT_FLOOR_FRAC, 2))
            px = max(floor, round(bid - spread * mult, 2))
        else:
            # trade-price fallback: the minute's LOW stands in for the bid and
            # already sits on the conservative side, so no cushion on top.
            bid, ask = got[0], got[1]
            px = max(0.01, round(bid, 2))
        new = (px / ep - 1.0) * 100
        tot_new += new
        fixes[i] = (px, new, bid, ask, lag, kind)
        print(f"  {str(r['timestamp'])[:19]} {str(r.get('ticker')):5} {ep:>7.2f} "
              f"{float(r['exit_price']):>7.2f} {bid:>9.2f} {px:>10.2f} "
              f"{old:>+9.1f} {new:>+9.1f} {lag:>4}m {kind}")

    print(f"\n  logged total     {tot_old:>+10.1f}%")
    print(f"  corrected total  {tot_new:>+10.1f}%   ({len(fixes)} repaired, "
          f"{len(unresolved)} unresolved)")
    print(f"  distortion removed {tot_new - tot_old:>+.1f}pp")

    if not a.apply:
        print(f"\n  DRY RUN -- rerun with --apply to write "
              f"(a .bak copy is made first)")
        return

    bak = f"{LOG}.bak-{dt.date.today().isoformat()}-penny"
    shutil.copy2(LOG, bak)
    print(f"\n  backup -> {bak}")

    out = list(raw)
    for i, r, tail in recs:
        changed = False
        if i in fixes:
            px, new, bid, ask, lag, kind = fixes[i]
            r["exit_price_corrupt"] = r["exit_price"]
            r["pnl_pct_corrupt"] = r.get("pnl_pct")
            r["exit_price"] = px
            r["price"] = px
            r["pnl_pct"] = round(new, 2)
            n = int(r.get("contracts") or 1)
            ep = float(r.get("entry_price") or 0)
            r["pnl_dollars"] = round((px - ep) * 100 * n, 2)
            r["pnl_dollars_per_contract"] = round((px - ep) * 100, 2)
            r["reconcile_confidence"] = "nbbo" if kind == "nbbo" else "trade-proxy"
            r["reconcile_source"] = (
                (f"UW /option-contract/{r.get('contract')}/flow "
                 f"nbbo_bid={bid} nbbo_ask={ask} lag={lag}m") if kind == "nbbo" else
                (f"UW /option-contract/{r.get('contract')}/intraday "
                 f"minute_low={bid} close={ask} lag={lag}m -- TRADE price, not "
                 f"NBBO; the contract had expired and /flow serves nothing"))
            r["reconcile_note"] = ("zero-bid exit bug: trail fired on a phantom "
                                   "quote and booked $0.01; repriced at the real "
                                   "NBBO bid with _fire_exit's cushion")
            changed = True
        elif any(i == j for j, _ in unresolved):
            r["reconcile_note"] = ("zero-bid exit bug: no real quote found within "
                                   "30m; exit price left as logged and NOT "
                                   "trustworthy")
            changed = True
        # mark entries that only exist because a phantom exit freed the slot
        if str(r.get("action", "")).startswith("ENTRY"):
            key = (r.get("ticker"), str(r.get("timestamp"))[:10])
            prior = [t for t in bad_ts.get(key, []) if t < str(r.get("timestamp"))]
            if prior:
                r["sequence_artifact"] = True
                r["reconcile_note"] = (f"follows {len(prior)} phantom exit(s) the "
                                       f"same session; this entry would not have "
                                       f"fired with the bad-tick guard in place")
                changed = True
        if changed:
            out[i] = json.dumps(r, ensure_ascii=False) + (tail or "")

    with open(LOG, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"  wrote {LOG}: {len(fixes)} exits repriced, "
          f"{sum(1 for _, r, _ in recs if r.get('sequence_artifact'))} entries "
          f"flagged as sequence artifacts")


if __name__ == "__main__":
    main()
