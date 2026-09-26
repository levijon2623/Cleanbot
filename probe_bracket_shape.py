# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv", "requests"]
# ///
"""
probe_bracket_shape.py
======================
WHICH combo framing does Webull accept for an options bracket?

    python probe_bracket_shape.py          # dry — show all three payloads
    python probe_bracket_shape.py --go     # submit each until one is accepted

🚨 WHY PROBE INSTEAD OF REASONING
    The pair was sent exactly as it READS BACK from get_open_orders and came
    back HTTP 417 OPENAPI_PROMPT_INSUFFICIENT_BUYING_POWER. That error is
    diagnostic: closing what you own needs no buying power, so the broker was
    treating two SELLs as independent -- 2x the position going out against 1x
    held, the excess naked. The read shape is not the write shape, and the
    SDK's own docs do not say which it wants. A rejected order costs nothing,
    so asking the API is cheaper and more reliable than guessing.

WHAT IT DOES
    Tries the framings in order, stops at the FIRST acceptance, then
    immediately tears down whatever it created:
      PAIR  STOP_PROFIT + STOP_LOSS   (as observed on the read; rejected once)
      OCO   both legs "OCO"           (one-cancel-others is the semantics we
                                       actually want)
      SLP   both "STOP_LOSS_PROFIT"   (ComboType 2, literally "Stop loss
                                       Profit")

🚨 EXPOSURE. The take-profit is far out of reach. The stop cannot be, on a
    decaying 0DTE -- it sits at the $0.01 floor, which is where the contract
    is heading. The window is the couple of seconds between acceptance and
    teardown, and the script prints the dollar figure before sending.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TP_MULT = 10.0
SL_ABS = 0.01
# SINGLE first: the _AT_CLOSE rejection says the broker saw MORE closing
# quantity than the position holds, and one order cannot do that.
SHAPES = ["SINGLE", "OCO", "SLP", "PAIR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--ticker")
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
    print("  BRACKET SHAPE PROBE — which combo framing does the API accept?")
    print("=" * 70)

    wb = WebullGammaClient(os.getenv("WEBULL_APP_KEY"),
                           os.getenv("WEBULL_APP_SECRET"),
                           os.getenv("WEBULL_ACCOUNT_ID"),
                           paper_trading=PAPER_TRADING)

    # 🚨 THE BROKER IS THE SOURCE OF TRUTH, not manual_positions.json --
    # which was stale by a whole contract when this was written.
    positions = [p for p in (wb.get_open_option_positions() or [])
                 if p.get("occ") and p.get("quantity")]
    if a.ticker:
        positions = [p for p in positions if p.get("underlying") == a.ticker]
    if not positions:
        sys.exit("\n  🚫 no open option position. Buy one first; a bracket "
                 "needs something to protect.")
    p = positions[0]
    occ, qty = p["occ"], int(abs(p["quantity"]))
    entry = float(p.get("cost_price") or 0)
    print(f"\n  position  {p['underlying']} {qty}x {occ}  cost {entry:.2f}")

    bid, ask, ok = wb.get_live_option_quote_ex(occ)
    if not ok or bid <= 0:
        sys.exit(f"  🚫 no live quote for {occ} — cannot price the exposure.")
    tp = round(max(entry, bid) * TP_MULT, 2)
    sl = SL_ABS
    print(f"  quote     {bid:.2f} / {ask:.2f}")
    print(f"  TP {tp:.2f} (out of reach)   SL {sl:.2f} (a 0DTE floor, NOT "
          f"out of reach)")
    print(f"  exposure if the stop fires before teardown: "
          f"~${max(0.0, bid - sl) * qty * 100:,.2f}")

    class Eng:
        pass
    eng = Eng()
    eng.webull = wb

    if not a.go:
        for shape in SHAPES:
            r = wb.place_option_bracket(p["underlying"], occ, qty, tp, sl,
                                        dry=True, combo=shape)
            tags = [o["combo_type"] for o in r["payload"]["new_orders"]]
            print(f"\n  [{shape}] combo_type tags: {tags}")
            print(f"        types: "
                  f"{[o['order_type'] for o in r['payload']['new_orders']]}")
        print("\n  DRY — nothing sent. Re-run with --go.")
        print("=" * 70)
        return

    winner = None
    for shape in SHAPES:
        print(f"\n  --- trying {shape} ---")
        res = wb.place_option_bracket(p["underlying"], occ, qty, tp, sl,
                                      combo=shape)
        if res.get("accepted"):
            print(f"  ✅ {shape} ACCEPTED   combo={res.get('combo_order_id')}")
            winner = shape
            break
        err = res.get("error") or json.dumps(res.get("raw"), default=str)[:200]
        print(f"  🚫 {shape} rejected: {err}")
        time.sleep(1.0)

    print("\n  --- tearing down whatever exists ---")
    time.sleep(2.0)
    rest = MO.resting_orders(eng, occ)
    for r in rest:
        print(f"      {r['kind']:<16} x{r['qty']}  {r['coid']}")
    killed = MO.cancel_resting(eng, occ, why="shape probe")
    time.sleep(2.0)
    left = MO.resting_orders(eng, occ)

    print("\n" + "=" * 70)
    if winner:
        print(f"  ACCEPTED FRAMING: {winner}")
        print(f"  legs seen on the book: {len(rest)}  (2 = a real pair)")
        print(f"  cancelled {len(killed)}, {len(left)} still resting")
        print(f"\n  Set combo=\"{winner}\" as the default in "
              f"place_option_bracket.")
    else:
        print("  NO FRAMING ACCEPTED.")
        print("  All three were refused. Most likely this account cannot hold")
        print("  a resting option bracket at all -- which on Level 2 with a")
        print("  small balance is entirely plausible, since the broker prices")
        print("  the unnetted leg as naked. Drop broker-side brackets and rely")
        print("  on the 15:55 sweep, which needs no buying power.")
    if left:
        print(f"\n  ⚠️ {len(left)} ORDER(S) STILL RESTING — cancel in the app.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
