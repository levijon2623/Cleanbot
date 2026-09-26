# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
analyze_manual.py
=================
The discretionary trade record: what actually happened, joined to what you
said beforehand.

    python analyze_manual.py
    python analyze_manual.py --since 2026-09-23
    python analyze_manual.py --full        # per-trade rows

READS  manual_executions.jsonl  (open/close rows, linked by trade_id)
       _notes.jsonl             (pre-trade reads: dir + TAKE/PASS + why)

🚨 IT REFUSES TO DRAW A CONCLUSION ON A HANDFUL OF TRADES, ON PURPOSE.
    check_multiplier spent a full build on something that came down to ONE
    session. check_flow_spike passed four of five pre-committed criteria on
    +0.34bp. The cheap question -- "is there enough here to measure?" -- is
    the one that was not asked first in either case, so it is asked first
    here and the summary stays descriptive until it is answered. Counting
    your own trades is not a study; it is a diary, and a diary that prints a
    win rate starts to feel like evidence.

🚨 AND IT IS A SEPARATE POPULATION FROM bot_executions_log.jsonl.
    Never merge them. That ledger is what research_rules and the sim read;
    these trades were selected by a human on criteria that were mostly not
    written down, and pooling them would contaminate every backtest result
    with a selection effect nothing can model afterwards.

WHAT THE NOTE JOIN IS FOR
    A note says what you thought BEFORE the outcome. PASS entries matter as
    much as TAKE ones -- they are the only record of the setups you declined,
    and without them "my reads work" is unfalsifiable by construction. The
    join is by ticker and time proximity, and unmatched rows on BOTH sides
    are reported, because a trade with no note and a note with no trade are
    both things worth seeing.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

LEDGER = "manual_executions.jsonl"
NOTES = "_notes.jsonl"
NOTE_WINDOW_S = 30 * 60          # a note counts as "about" a trade within 30m

# Below this many CLOSED, non-adopted trades the script describes and does not
# conclude. Not a magic number -- it is simply far enough above 1 that a single
# lucky session cannot carry it, and the point is the refusal, not the value.
MIN_FOR_STATS = 20


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--full", action="store_true", help="per-trade rows")
    a = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rows = load(LEDGER)
    notes = load(NOTES)
    if not rows:
        print(f"\n  no {LEDGER} yet — it is written on the first manual fill.\n")
        return

    import datetime
    since = None
    if a.since:
        since = datetime.date.fromisoformat(a.since)

    def keep(r):
        if not since:
            return True
        return datetime.datetime.fromtimestamp(r["ts"]).date() >= since

    rows = [r for r in rows if keep(r)]
    opens = {r["trade_id"]: r for r in rows if r.get("kind") == "open"}
    closes = {r["trade_id"]: r for r in rows if r.get("kind") == "close"}

    trades, orphans = [], []
    for tid, o in opens.items():
        c = closes.get(tid)
        (trades if c else orphans).append((o, c))

    print(f"\n  {'=' * 66}")
    print(f"  DISCRETIONARY RECORD   {LEDGER}")
    print(f"  {'=' * 66}")
    print(f"  rows {len(rows)}   opens {len(opens)}   closes {len(closes)}")
    print(f"  closed round trips {len(trades)}   still open / never closed "
          f"{len(orphans)}")

    adopted = [t for t in trades if t[0].get("adopted")]
    own = [t for t in trades if not t[0].get("adopted")]
    if adopted:
        print(f"  of those, {len(adopted)} were ADOPTED positions — excluded "
              f"below (their entry context is the state at adoption, not the "
              f"state you decided in)")

    if orphans:
        print(f"\n  ⚠ OPENS WITH NO CLOSE — these are not losses, they are "
              f"MISSING DATA:")
        for o, _ in orphans:
            print(f"    {o.get('ticker')} {o.get('qty')}x {o.get('strike')}"
                  f"{o.get('right')} opened "
                  f"{datetime.datetime.fromtimestamp(o['ts']):%m-%d %H:%M}"
                  f"  ({'adopted' if o.get('adopted') else 'viewer'})")

    # The mirror orphan, and the one you actually hit: a position that existed
    # BEFORE the ledger did, closed after it. There is an exit with no entry.
    # Reporting only open-side orphans would make these vanish entirely.
    stray = [c for tid, c in closes.items() if tid not in opens]
    if stray:
        print(f"\n  ⚠ CLOSES WITH NO OPEN — the position predates this ledger, "
              f"so there is no entry context and no P&L to trust:")
        for c in stray:
            print(f"    {c.get('ticker')} {c.get('qty')}x {c.get('strike')}"
                  f"{c.get('right')} closed "
                  f"{datetime.datetime.fromtimestamp(c['ts']):%m-%d %H:%M}"
                  f" @ {c.get('exit')}  "
                  f"(reason: {c.get('exit_reason') or '?'})")

    if a.full and own:
        print(f"\n  {'when':<12}{'tkr':<5}{'contract':<12}{'qty':>4}"
              f"{'entry':>7}{'exit':>7}{'roe%':>8}{'pnl$':>9}{'held':>7}"
              f"{'note':>6}")
        for o, c in sorted(own, key=lambda t: t[0]["ts"]):
            n = match_note(o, notes)
            held = c.get("held_s")
            print(f"  {datetime.datetime.fromtimestamp(o['ts']):%m-%d %H:%M} "
                  f"{o.get('ticker',''):<5}"
                  f"{str(o.get('strike','')) + str(o.get('right','')):<12}"
                  f"{o.get('qty',0):>4}{o.get('entry') or 0:>7.2f}"
                  f"{c.get('exit') or 0:>7.2f}"
                  f"{c.get('roe_pct') or 0:>+8.1f}{c.get('pnl') or 0:>+9.2f}"
                  f"{(str(held // 60) + 'm') if held else '?':>7}"
                  f"{(n.get('action', '')[:4] if n else '—'):>6}")

    # ---- the note join -------------------------------------------------
    takes = [n for n in notes if n.get("action") == "TAKE"]
    passes = [n for n in notes if n.get("action") == "PASS"]
    matched = sum(1 for o, _ in own if match_note(o, notes))
    print(f"\n  {'-' * 66}")
    print(f"  PRE-TRADE NOTES   {NOTES}")
    print(f"  {len(notes)} note(s): {len(takes)} TAKE, {len(passes)} PASS")
    print(f"  trades with a matching note: {matched}/{len(own)}")
    if passes:
        print(f"  ⓘ the {len(passes)} PASS note(s) are the only record of "
              f"setups you declined — they are what make 'my reads work' "
              f"falsifiable at all")
    unmatched_takes = [n for n in takes if not any(
        abs(n.get("ts", 0) - o["ts"]) <= NOTE_WINDOW_S
        and n.get("ticker") == o.get("ticker") for o, _ in own)]
    if unmatched_takes:
        print(f"  ⚠ {len(unmatched_takes)} TAKE note(s) with no trade within "
              f"{NOTE_WINDOW_S//60}m — intended but not executed, or executed "
              f"outside the viewer")

    # ---- summary, gated on power ---------------------------------------
    print(f"\n  {'-' * 66}")
    if len(own) < MIN_FOR_STATS:
        pnl = sum(c.get("pnl") or 0 for _, c in own)
        print(f"  {len(own)} closed discretionary trade(s). Net ${pnl:+,.2f}.")
        print(f"\n  NO STATISTICS. {MIN_FOR_STATS} is the floor, and this is "
              f"not close.")
        print(f"  A win rate over {len(own)} trades is a number, not a "
              f"measurement: at this n it is dominated by which ones you")
        print(f"  happened to take, and the setups you passed on are not in "
              f"it at all. Keep trading and keep noting; the")
        print(f"  record is accumulating correctly, which is the whole point "
              f"of this file existing.")
        print(f"  {'=' * 66}\n")
        return

    roes = [c["roe_pct"] for _, c in own if c.get("roe_pct") is not None]
    pnls = [c["pnl"] for _, c in own if c.get("pnl") is not None]
    wins = [r for r in roes if r > 0]
    print(f"  {len(own)} closed trades")
    print(f"    win rate     {len(wins)/len(roes)*100:.0f}%  "
          f"({len(wins)}/{len(roes)})")
    print(f"    mean ROE     {statistics.mean(roes):+.1f}%")
    print(f"    median ROE   {statistics.median(roes):+.1f}%")
    print(f"    net P&L      ${sum(pnls):+,.2f}")
    print(f"    best / worst {max(roes):+.0f}% / {min(roes):+.0f}%")
    holds = [c["held_s"] for _, c in own if c.get("held_s")]
    if holds:
        print(f"    median hold  {statistics.median(holds)/60:.0f} min")

    by = defaultdict(list)
    for o, c in own:
        if c.get("roe_pct") is not None:
            by[o.get("ticker")].append(c["roe_pct"])
    print(f"\n    by ticker:")
    for tk, rs in sorted(by.items()):
        print(f"      {tk:<5} n={len(rs):<4} mean {statistics.mean(rs):+.1f}%")

    print(f"\n  ⓘ Still descriptive. Nothing here is a placebo-controlled "
          f"test and it cannot become one: the setups you")
    print(f"    declined are not in the ledger, so there is no comparison "
          f"population. Use the PASS notes for that.")
    print(f"  {'=' * 66}\n")


def match_note(o, notes):
    """The note written nearest this trade's open, same ticker, within the
    window. Proximity, not certainty -- it is a hint for review, not a key."""
    best, bd = None, NOTE_WINDOW_S + 1
    for n in notes:
        if n.get("ticker") != o.get("ticker"):
            continue
        d = abs(int(n.get("ts", 0)) - int(o.get("ts", 0)))
        if d < bd:
            best, bd = n, d
    return best if bd <= NOTE_WINDOW_S else None


if __name__ == "__main__":
    import datetime  # noqa: F401  (used in main)
    main()
