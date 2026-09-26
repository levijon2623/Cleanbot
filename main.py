"""Read-only Hyperliquid account inspector.

🚨 WHAT THIS FILE USED TO BE, AND WHY IT IS NOT THAT ANY MORE.
Until 2026-09-26 this file constructed a live HyperExposureClient and called
`client.set_take_profit()` -- a real reduce-only order against a funded account
-- inside `if __name__ == "__main__"`. Before an earlier fix it sat at module
scope, so merely IMPORTING it loaded the agent secret, opened an exchange
connection and attempted the order, which is what a routine import sweep did on
2026-09-11. The only thing that stopped a real order was that the call is
missing its four required arguments and raised TypeError first.

That is not a safety margin, it is a coincidence. A file named `main.py` in the
repo root is the first thing anyone runs to see whether the project works, so it
is the worst possible place to keep an execution experiment.

It now does exactly one thing: print what the account holds. `hl_gate` would
refuse an order from here anyway, but relying on the gate to save a file that is
TRYING to trade is the wrong way round.

    python main.py            # balances + open perp positions, nothing else
    python hl_gate.py         # show whether perp execution is armed, and why not

The directional options bot is `bot_runner.py`. This file is not its entrypoint
and nothing imports it.
"""
import os

from dotenv import load_dotenv

import hl_gate
from hyper_exposure_client import HyperExposureClient

if __name__ == "__main__":
    load_dotenv()

    wallet = os.getenv("MAIN_WALLET")
    if not wallet:
        raise SystemExit("MAIN_WALLET is not set in .env — nothing to inspect.")

    hl_gate.banner(force=True)

    # No agent secret is passed, so the client comes up READ-ONLY: self.exchange
    # stays None and hl_gate.check() refuses every action on `has_exchange`
    # before any flag is even consulted. Inspecting an account should not
    # require the ability to trade it.
    print("Connecting (read-only — no agent secret loaded)…")
    client = HyperExposureClient(main_wallet_address=wallet, agent_secret=None)

    client.get_balances()

    print("\n--- OPEN PERP POSITIONS ---")
    positions = client.get_open_positions()
    for p in positions:
        print(f"  {p['coin']:>12}  size {p['size']:+.4f}  "
              f"entry ${p['entry_price']:,.2f}  {p['leverage']}x")
    if not positions:
        print("  (none)")
