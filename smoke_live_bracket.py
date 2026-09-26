# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv", "requests"]
# ///
"""
smoke_live_bracket.py
=====================
STAGE 2.5: can we CREATE a broker-side TP/SL bracket, and tear it back down?

    python smoke_live_bracket.py                 # dry — prints the payload
    python smoke_live_bracket.py --go            # actually create it
    python smoke_live_bracket.py --cancel-only   # just tear down what exists

🚨 WHY THIS NEEDS A REAL POSITION, UNLIKE STAGE 1
    Stage 1 could be made safe with an unfillable price because it tested a
    single resting order. A TP/SL only becomes meaningful against something
    you hold, so this runs against whatever manual position is open.

🚨 AND WHY "UNREACHABLE LEVELS" IS ONLY HALF TRUE ON A 0DTE
    The take-profit is genuinely out of reach -- TP_MULT x the entry is a
    move the contract will not make while this script runs. The STOP is not.
    A decaying 0DTE is heading toward $0.01, so a stop there is a
    destination, not a floor. The exposure is the seconds the bracket is
    live: if the stop fired you would sell at ~the floor and give up the
    remaining premium, which the script prints as a dollar figure before it
    sends anything. That is the honest risk, and it is why the bracket is
    torn down immediately rather than left up to admire.

🚨 WHAT IT CANNOT PROVE
    That the OCO linkage works. Confirming that one leg filling cancels the
    other would require letting a leg trigger, which means giving up the
    position to find out. So this proves creation, visibility and teardown;
    the cancel-on-fill behaviour is Webull's own and stays an assumption.
    manual_orders.flatten cancels both legs before any exit precisely so we
    never depend on it.

WHAT A PASS LOOKS LIKE
    1. a manual hold exists, filled, with a known entry
    2. the payload is well formed (dry run shows it)
    3. place returns accepted + a combo_order_id
    4. BOTH legs appear in get_open_orders, sharing that combo id
    5. resting_orders() matches them to the contract
    6. cancel_resting() removes both, and the book is clear afterwards

    Step 4 failing means the write shape is wrong -- likely combo_type or the
    request class -- and the thing to do is print the raw response, not retry.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# deliberately absurd: a 0DTE that reaches +900% or drops to a penny while
# this script runs is not a scenario the levels need to survive
TP_MULT = 10.0
SL_ABS = 0.01
# Seconds between create -> verify -> tear down. Short on purpose: the whole
# exposure is the time the stop is live on a contract that decays toward it.
HOLD_S = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually create it")
    ap.add_argument("--cancel-only", action="store_true",
                    help="tear down any resting SELLs on the held contract")
    ap.add_argument("--ticker", help="which hold (default: the only one)")
    a = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    logging.disable(logging.INFO)

    import manual_orders as MO
    from webull_gamma_client import WebullGammaClient
    from config import PAPER_TRADING

    print("\n" + "=" * 70)
    print("  STAGE 2.5 — broker-side bracket: create, verify, tear down")
    print("=" * 70)

    holds = MO._read(MO.STATE) or {}
    if not holds:
        sys.exit("\n  🚫 no manual hold in live/manual_positions.json.\n"
                 "     Open one through the viewer first (Stage 2), then run\n"
                 "     this against it. Nothing to protect otherwise.")
    tk = a.ticker or (list(holds) if len(holds) == 1 else [None])[0]
    if tk not in holds:
        sys.exit(f"\n  🚫 pick one with --ticker: {sorted(holds)}")
    pos = holds[tk]
    print(f"\n[1/6] the hold")
    print(f"      {tk} {pos['qty']}x {pos['option_id']}  entry {pos.get('entry')}")
    if pos.get("pending"):
        sys.exit("  🚫 entry has not filled — nothing to bracket.")
    entry = float(pos.get("entry") or 0)
    if entry <= 0:
        sys.exit("  🚫 no known entry price.")

    wb = WebullGammaClient(os.getenv("WEBULL_APP_KEY"),
                           os.getenv("WEBULL_APP_SECRET"),
                           os.getenv("WEBULL_ACCOUNT_ID"),
                           paper_trading=PAPER_TRADING)

    class Eng:
        pass
    eng = Eng()
    eng.webull = wb
    eng.manual_holds = holds
    eng.active_snipes = {}

    # ---- tear-down-only path ------------------------------------------
    if a.cancel_only:
        print(f"\n[--] resting SELLs on {pos['option_id']}:")
        for r in MO.resting_orders(eng, pos["option_id"]):
            print(f"      {r['kind']:<12} x{r['qty']}  {r['coid']}")
        killed = MO.cancel_resting(eng, pos["option_id"], why="manual teardown")
        time.sleep(HOLD_S)
        left = MO.resting_orders(eng, pos["option_id"])
        print(f"      cancelled {len(killed)}; {len(left)} still resting")
        return

    tp = round(entry * TP_MULT, 2)
    sl = SL_ABS
    qty = int(pos["qty"])
    bid = None
    try:
        b, _a, ok = wb.get_live_option_quote_ex(pos["option_id"])
        bid = b if ok else None
    except Exception:
        pass
    print(f"\n[2/6] levels")
    print(f"      TP {tp:.2f}  = {TP_MULT:.0f}x the {entry:.2f} entry — "
          f"out of reach")
    print(f"      SL {sl:.2f}  — NOT out of reach on a decaying 0DTE")
    if bid:
        risk = max(0.0, (bid - sl)) * qty * 100
        print(f"\n      live bid {bid:.2f} x {qty} = "
              f"${bid * qty * 100:,.2f} of remaining premium")
        print(f"      if the stop fired while the bracket is up (~{HOLD_S*2}s), "
              f"you give up ~${risk:,.2f}")
        print(f"      it would need a {(1 - sl / bid) * 100:.0f}% drop in that "
              f"window")
    else:
        print("      ⚠ no live quote — cannot price the exposure. Stop here "
              "and check the subscription.")
        sys.exit(1)

    print(f"\n[3/6] payload (dry)")
    d = MO.attach_bracket(eng, tk, tp=tp, sl=sl, dry=True)
    if not d.get("ok"):
        sys.exit(f"  🚫 refused before sending: {d.get('err')}")
    print("   " + json.dumps(d["payload"], indent=2)[:1100].replace(
        "\n", "\n   "))

    if not a.go:
        print("\n  DRY — nothing sent. Re-run with --go to create it.")
        print("=" * 70)
        return

    print(f"\n[4/6] creating…")
    res = MO.attach_bracket(eng, tk, tp=tp, sl=sl)
    if not res.get("ok"):
        print(f"  🚫 {res.get('err')}")
        sys.exit("  Stage 2.5 FAILS here. Print the raw response and look at "
                 "combo_type / the request class before retrying.")
    combo = res.get("combo_order_id")
    print(f"      combo_order_id {combo}")

    time.sleep(HOLD_S)
    print(f"\n[5/6] does the broker show both legs?")
    rest = MO.resting_orders(eng, pos["option_id"])
    for r in rest:
        print(f"      {r['kind']:<12} x{r['qty']}  coid {r['coid']}")
    raw_combo = set()
    for o in (wb.get_open_orders() or []):
        cid = ((o.get("raw") or {}).get("combo_order_id"))
        if cid:
            raw_combo.add(cid)
    print(f"      combo ids on the book: {sorted(raw_combo)}")

    print(f"\n[6/6] tearing it down…")
    killed = MO.cancel_resting(eng, pos["option_id"], why="smoke test")
    time.sleep(HOLD_S)
    left = MO.resting_orders(eng, pos["option_id"])

    checks = [
        ("hold exists and is filled", entry > 0),
        ("place accepted", bool(res.get("ok"))),
        ("combo_order_id returned", bool(combo)),
        ("both legs visible on the book", len(rest) == 2),
        ("legs matched to the contract", len(rest) >= 1),
        ("cancel removed both", len(killed) == 2),
        ("book clear afterwards", not left),
    ]
    print("\n" + "=" * 70)
    for label, good in checks:
        print(f"  {'PASS' if good else 'FAIL'}  {label}")
    print("=" * 70)
    if all(g for _, g in checks):
        print("  Brackets can be created and torn down from the API.")
        print("  NOT proven: that one leg filling cancels the other. flatten()")
        print("  pulls both before any exit, so nothing depends on it.")
    else:
        print("  🚫 Do not wire this into the viewer yet.")
        if left:
            print(f"  ⚠️ {len(left)} order(s) STILL RESTING — cancel them in "
                  f"the app, or re-run with --cancel-only.")
    print()

    # leave the hold's record consistent with the book we just cleared
    pos.pop("bracket", None)
    MO._write_atomic(MO.STATE, holds)


if __name__ == "__main__":
    main()
