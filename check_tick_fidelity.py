# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
check_tick_fidelity.py
======================
DOES THE TICK STREAM CARRY THE WHOLE TAPE, OR A SAMPLE?

WHY THIS COMES BEFORE ANY CVD CODE
    Cumulative Volume Delta needs the AGGRESSOR SIDE of every trade. The only
    Webull topic carrying `side` is TICK. But the Data Streaming API docs state:

        "The server pushes messages at a maximum rate of 3 times per second
         per connection."

    IWM alone prints hundreds of trades a minute. If that cap counts MESSAGES
    and each batches many ticks, CVD is buildable. If it counts TICKS, CVD would
    be assembled from a fraction of a percent of the tape -- which is not a
    quiet CVD, it is a random walk with a plausible shape. There is no way to
    tell by reading the docs, and no historical tick data to check against, so
    it gets measured.

THE MEASUREMENT
    Subscribe to TICK and SNAPSHOT for ONE ticker. SNAPSHOT reports cumulative
    session volume, so over the same window:

        sum(tick volume received)  /  (snapshot_last - snapshot_first)

    ~1.0  -> the full tape is arriving; CVD is real
    <<1   -> sampled; the ratio IS the sampling rate

🚨 A DISTINCT session_id IS MANDATORY
    "A new connection with the same session_id will disconnect the previous
    one." Reusing the bot's id would silently kill the live bot's market data
    while it holds positions. The id below is unique per run. Each App Key also
    allows 5 concurrent connections; the bot uses one, this uses one more.

Usage:
  python check_tick_fidelity.py --ticker IWM --seconds 90
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from webull.data.data_streaming_client import DataStreamingClient
from webull.data.common.category import Category
from webull.data.common.subscribe_type import SubscribeType

from webull_gamma_client import _stdlib_ssl_context, quiet_webull_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="IWM")
    ap.add_argument("--seconds", type=int, default=90)
    # Webull's docs list TICK for stocks/futures/crypto only, but the SDK's
    # Category enum has US_OPTION and the bot already streams option QUOTES
    # that way. One OCC symbol here answers "do option ticks stream at all?" in
    # the same 90 seconds. Ticks for it are counted separately from the equity.
    ap.add_argument("--option", default=None,
                    help="also subscribe TICK on this OCC symbol, e.g. IWM260928C00285000")
    a = ap.parse_args()
    quiet_webull_logging()

    tick_n = 0
    tick_vol = 0
    sides = collections.Counter()
    snap_n = 0
    snap_first = snap_last = None
    samples = []
    opt = {"n": 0, "vol": 0, "samples": []}

    sc = DataStreamingClient(
        app_key=os.getenv("WEBULL_APP_KEY"),
        app_secret=os.getenv("WEBULL_APP_SECRET"),
        region_id="us",
        # unique per run -- see the docstring; reusing the bot's id kills the bot
        session_id=f"tickfid_{int(time.time()*1000)}",
        http_host="api.webull.com", mqtt_host="data-api.webull.com")
    quiet_webull_logging()

    ctx = _stdlib_ssl_context()
    if ctx is not None:
        sc._ssl_context = None
        sc.tls_set_context(ctx)

    def on_connect(client, api_client, session_id):
        print(f"  connected; subscribing TICK+SNAPSHOT for {a.ticker}")
        client.subscribe(symbols=[a.ticker],
                         category=Category.US_STOCK.name,
                         sub_types=[SubscribeType.TICK.name,
                                    SubscribeType.SNAPSHOT.name])
        if a.option:
            try:
                client.subscribe(symbols=[a.option],
                                 category=Category.US_OPTION.name,
                                 sub_types=[SubscribeType.TICK.name])
                print(f"  subscribed TICK for option {a.option}")
            except Exception as e:
                print(f"  option TICK subscribe REFUSED: {e}")

    def on_message(client, topic, payload):
        nonlocal tick_n, tick_vol, snap_n, snap_first, snap_last
        s = str(payload)
        tp = str(topic or "").lower()
        if "tick" in tp and a.option and f"symbol:{a.option}" in s:
            opt["n"] += 1
            opt["vol"] += sum(int(v) for v in re.findall(r'(?<![a-z_])volume:\s*"?(\d+)', s))
            if len(opt["samples"]) < 2:
                opt["samples"].append(s[:300])
        elif "tick" in tp:
            tick_n += 1
            if len(samples) < 3:
                samples.append(s[:400])
            # a message may batch several ticks -- count every volume field
            vs = re.findall(r'(?<![a-z_])volume:\s*"?(\d+)', s)
            for v in vs:
                tick_vol += int(v)
            for sd in re.findall(r'side:\s*"?([A-Za-z]+)', s):
                sides[sd] += 1
        elif "snapshot" in tp:
            snap_n += 1
            m = re.search(r'(?<![a-z_])volume:\s*"?(\d+)', s)
            if m:
                v = int(m.group(1))
                if snap_first is None:
                    snap_first = v
                snap_last = v

    # The SDK's own callback names -- NOT paho's. Assigning sc.on_connect /
    # sc.on_message leaves on_quotes_subscribe unset and the client raises
    # "SDK.InvalidParameter on_quotes_subscribe func must be set" on CONNACK.
    # Same two names webull_gamma_client.py:396-397 uses.
    sc.on_connect_success = on_connect
    sc.on_quotes_message = on_message

    import threading
    threading.Thread(target=sc.connect_and_loop_forever, daemon=True).start()
    quiet_webull_logging()        # the loop thread attaches its own handler

    print(f"  collecting {a.seconds}s ...")
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        time.sleep(5)
        el = int(time.time() - t0)
        print(f"    {el:>3}s  ticks {tick_n:>6}  tick_vol {tick_vol:>12,}  "
              f"snapshots {snap_n:>4}")

    print(f"\n{'='*78}")
    print(f"  RESULT for {a.ticker} over {a.seconds}s")
    print(f"{'='*78}")
    print(f"  tick messages          {tick_n:>12,}  "
          f"({tick_n/max(a.seconds,1):.1f}/sec)")
    print(f"  volume in those ticks  {tick_vol:>12,}")
    print(f"  snapshot messages      {snap_n:>12,}")
    if snap_first is not None and snap_last is not None:
        delta = snap_last - snap_first
        print(f"  snapshot cum volume    {snap_first:,} -> {snap_last:,}"
              f"   delta {delta:,}")
        if delta > 0:
            ratio = tick_vol / delta
            print(f"\n  FIDELITY  tick_vol / snapshot_delta = {ratio:.3f}")
            if ratio >= 0.9:
                print(f"  -> FULL TAPE. CVD is buildable from this stream.")
            elif ratio >= 0.2:
                print(f"  -> PARTIAL ({ratio*100:.0f}%). CVD would be biased; the")
                print(f"     missing trades are not missing at random (the cap")
                print(f"     drops messages when the tape is busiest).")
            else:
                print(f"  -> HEAVILY SAMPLED ({ratio*100:.1f}%). CVD is not")
                print(f"     buildable from this stream.")
        else:
            print(f"\n  snapshot volume did not advance -- market likely closed.")
    else:
        print(f"  no snapshot volume seen -- market likely closed, or no data.")

    if sides:
        print(f"\n  side values seen: {dict(sides)}")
    else:
        print(f"\n  no `side` field parsed -- the regex may need the raw shape:")
    for i, s in enumerate(samples):
        print(f"    sample {i}: {s}")

    if a.option:
        print(f"\n  OPTION TICKS for {a.option}: {opt['n']:,} messages, "
              f"{opt['vol']:,} contracts")
        if opt["n"]:
            print("  -> option ticks DO stream. Fidelity for options would need its")
            print("     own check (no option SNAPSHOT volume is subscribed here).")
            for s in opt["samples"]:
                print(f"    sample: {s}")
        else:
            print("  -> none received. Either option TICK is not served, or this")
            print("     contract did not trade in the window -- pick a busy ATM")
            print("     0DTE contract so silence means the former.")
    try:
        sc.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
