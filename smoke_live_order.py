# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv", "requests"]
# ///
"""
smoke_live_order.py
===================
STAGE 1 of the live-execution rollout: prove the BROKER PLUMBING works.

    python smoke_live_order.py --ticker IWM
    python smoke_live_order.py --ticker IWM --go        # actually send

🚨 WHAT THIS DOES AND WHY IT IS SAFE
    Places ONE real BUY LIMIT for ONE contract at a price that CANNOT FILL --
    $0.01 on a contract quoted around a dollar. A buy limit far BELOW the
    market rests; it does not cross. Then it reads the order back, confirms
    the status, cancels it, and confirms the cancel.

    Maximum loss if every assumption here is wrong and it somehow fills: one
    contract at $0.01 = $1.00 plus fees.

🚨 WHAT IT DELIBERATELY DOES NOT TEST
    manual_orders' state machine -- the guards, the pending/hold promotion,
    the exit escalation. Those are covered against a fake broker by
    test_stage0 / test_exits / test_order_loop, which can simulate partial
    fills and cancel races that you cannot produce on demand against a live
    API. This covers the other half: that place / status / cancel actually
    behave the way that state machine assumes. Fakes cannot prove that, and
    this cannot prove what fakes prove.

WHAT A PASS LOOKS LIKE
    1. quote     a real two-sided market (else there is nothing to rest under)
    2. place     returns a client_order_id
    3. status    reads back SUBMITTED, filled_qty 0
    4. cancel    accepted
    5. status    reads back CANCELLED
    6. positions unchanged, and no new position appeared

    Any step failing means DO NOT proceed to Stage 2. The state machine is
    built on exactly these six behaviours.

DRY BY DEFAULT. Without --go it prints the order it would send and stops.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="IWM", choices=["IWM", "SPY", "QQQ"])
    ap.add_argument("--limit", type=float, default=0.01,
                    help="the unfillable price (default 0.01)")
    ap.add_argument("--wait", type=float, default=3.0,
                    help="seconds to let the order rest before cancelling")
    ap.add_argument("--go", action="store_true",
                    help="actually send. Without this, nothing is placed.")
    a = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    logging.disable(logging.INFO)

    from webull_gamma_client import WebullGammaClient
    from config import PAPER_TRADING
    import manual_orders

    print("\n" + "=" * 68)
    print("  STAGE 1 SMOKE TEST — broker plumbing, unfillable order")
    print("=" * 68)

    wb = WebullGammaClient(os.getenv("WEBULL_APP_KEY"),
                           os.getenv("WEBULL_APP_SECRET"),
                           os.getenv("WEBULL_ACCOUNT_ID"),
                           paper_trading=PAPER_TRADING)

    # ---- spot, WITHOUT the tick stream --------------------------------
    # 🚨 get_spot_price lives on the ENGINE and reads MQTT memory
    # (bot_runner.py:1215). This process has no stream, so it must come from
    # REST or not at all. Getting it WRONG is not an option either: spot picks
    # the strike, and a nonsense strike is a nonsense order. Fail loudly.
    print(f"\n[1/6] spot for {a.ticker} (UW 1m bars — no MQTT here)…")
    spot = 0.0
    try:
        from unusual_whales_client import UnusualWhalesClient
        uw = UnusualWhalesClient(os.getenv("UW_API_KEY"))
        bars = uw.get_intraday_bars(a.ticker, lookback_days=1) or []
        if bars:
            spot = float(bars[-1]["close"])
            print(f"      {spot}  (as of {bars[-1].get('minute_et')})")
    except Exception as e:
        print(f"      UW failed: {type(e).__name__}: {e}")
    if spot <= 0:
        sys.exit("  🚫 could not determine spot. Without it the strike choice "
                 "is a guess, and this script will not guess at a strike it "
                 "is about to send an order for.")

    # ---- pick a liquid 0DTE call near the money -----------------------
    # Passing spot also lets scan_option_chain's ±5% pre-filter do its job;
    # without it the call pulls snapshots for every contract on the board
    # (5,258 for IWM, at 0.3s per batch of 20 — about 80 seconds of nothing).
    print(f"\n[1b]  chain…")
    chain = wb.scan_option_chain(a.ticker, spot_price=spot)
    import datetime
    today = datetime.date.today()
    cands = []
    for c in (chain or {}).get("data", []):
        try:
            if c.get("expireDate") != today.isoformat():
                continue
            if c.get("callPut") != "Call":
                continue
            cands.append((abs(float(c["strikePrice"]) - spot), c))
        except (TypeError, ValueError, KeyError):
            continue
    if not cands:
        exps = sorted({c.get("expireDate") for c in (chain or {}).get("data", [])
                       if c.get("expireDate")})[:6]
        sys.exit(f"  🚫 no {a.ticker} calls expiring {today.isoformat()}.\n"
                 f"     expiries actually on the board: {exps}\n"
                 f"     (no 0DTE today? try --ticker SPY, which lists daily.)")
    cands.sort(key=lambda x: x[0])
    best = cands[0][1]
    occ = best.get("symbol")
    print(f"      {len(cands)} 0DTE calls; nearest ATM -> {occ} "
          f"(strike {best.get('strikePrice')}, vol {best.get('volume')}, "
          f"OI {best.get('openInterest')})")

    # ---- a real two-sided market, or there is nothing to rest under ----
    # Walk outward from ATM until one prices well clear of our limit. The
    # nearest strike is not automatically the best choice: a 0DTE can be
    # pennies late in the session, and a bid near our own limit is the one
    # case where this order could actually fill.
    print(f"\n[2/6] quote…")
    bid = ask = 0.0
    for _d, c in cands[:8]:
        o = c.get("symbol")
        b, k, ok = wb.get_live_option_quote_ex(o)
        flag = ("OK" if ok and b > 0 and k > 0 and b >= a.limit * 4
                else "too cheap/thin" if ok else "no quote")
        print(f"      {o}  bid {b} / ask {k}   {flag}")
        if flag == "OK":
            occ, bid, ask, best = o, b, k, c
            break
    if bid <= 0:
        sys.exit("  🚫 no 0DTE call with a two-sided quote comfortably above "
                 f"${a.limit:.2f}. Market closed, or everything near the money "
                 "is pennies. Nothing is safe to rest under — stop here.")
    print(f"      using {occ}: limit {a.limit:.2f} is {bid/a.limit:.0f}x below "
          f"the bid {bid:.2f} — it rests, it does not cross.")

    # ---- what we are about to do --------------------------------------
    print(f"\n[3/6] the order:")
    print(f"      BUY_TO_OPEN  1x {occ}  LIMIT {a.limit:.2f}   "
          f"(max risk ${a.limit*100:.2f})")
    if not a.go:
        print("\n  DRY — nothing sent. Re-run with --go to place it.")
        print("=" * 68)
        return

    before = {p["occ"]: p["quantity"] for p in wb.get_open_option_positions()}

    print(f"\n[4/6] placing…")
    res = wb.place_option_order(ticker=a.ticker, option_id=occ, action="BUY",
                                quantity=1, is_closing=False,
                                order_type="LIMIT", limit_price=a.limit)
    coid = (res or {}).get("client_order_id")
    print(f"      accepted={bool((res or {}).get('accepted'))}  coid={coid}")
    if not coid:
        print(f"      raw: {res}")
        sys.exit("  🚫 FAIL at step 4 — no client_order_id. manual_orders "
                 "treats this as 'broker rejected' and takes no position, "
                 "which is correct, but Stage 2 cannot proceed.")

    time.sleep(a.wait)
    print(f"\n[5/6] status after {a.wait:.0f}s…")
    st = wb.get_order_status(coid)
    print(f"      status={st.get('status')}  filled_qty={st.get('filled_qty')}"
          f"  fill_price={st.get('fill_price')}")
    if st.get("status") not in ("SUBMITTED", "PENDING", "QUEUED"):
        print(f"      ⚠️ expected SUBMITTED. raw: {st.get('raw')}")
    if float(st.get("filled_qty") or 0) > 0:
        print("  🚨 IT FILLED. Cancel is pointless; CLOSE THE POSITION NOW.")

    print(f"\n[6/6] cancelling…")
    cancelled = wb.cancel_option_order(coid)
    time.sleep(2.0)
    st2 = wb.get_order_status(coid)
    print(f"      cancel returned {cancelled}  ->  status={st2.get('status')}")

    after = {p["occ"]: p["quantity"] for p in wb.get_open_option_positions()}
    new = {k: v for k, v in after.items() if before.get(k) != v}

    print("\n" + "=" * 68)
    checks = [
        ("quote two-sided", ok and bid > 0 and ask > 0),
        ("place returned a coid", bool(coid)),
        ("status readable", st.get("status") is not None),
        ("did NOT fill", float(st.get("filled_qty") or 0) == 0),
        ("cancel accepted", bool(cancelled)),
        ("status now CANCELLED", st2.get("status") == "CANCELLED"),
        ("positions unchanged", not new),
    ]
    for label, good in checks:
        print(f"  {'PASS' if good else 'FAIL'}  {label}")
    if new:
        print(f"  🚨 position changed: {new}")
    print("=" * 68)
    if all(g for _, g in checks):
        print("  All six behaviours the state machine assumes are real.")
        print("  Stage 2 (1 contract, round trip through the viewer) is clear.")
    else:
        print("  🚫 DO NOT PROCEED TO STAGE 2. Fix the FAILs above first.")
    print()


if __name__ == "__main__":
    main()
