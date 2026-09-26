# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets", "polars>=1.0.0", "python-dotenv"]
# ///
"""
dxlink_streamer.py
==================
dxLink (dxFeed) WebSocket client for ORDER-LEVEL market data on CME.

WHY THIS EXISTS ALONGSIDE rithmic_streamer.py
---------------------------------------------
Two candidate paths to CME order-by-order data. dxFeed is the shorter one: the
CME "Market Depth (MBO)" entitlement is ALREADY ACTIVE ($39/mo, paid), whereas
Rithmic needs conformance (currently blocked on rp_code 13) plus ~$114/mo. The
dxLink protocol is also far easier -- JSON over WebSocket against a published
AsyncAPI spec, versus Rithmic's protobuf template dispatch.

*** THE QUESTION THIS CLIENT IS BUILT TO ANSWER ***
dxFeed sells TWO different order-level products and only one is what we need:

  Full Order Depth (FOD)     individual orders keyed by `index` only.
                             NO orderId. NO action types. You reconstruct the
                             book from index + eventFlags. Iceberg detection and
                             per-order lifecycle become INFERENCE.

  Enhanced Order Book (EOB,  `orderId`, `action` (NEW/REPLACE/MODIFY/DELETE/
  a.k.a. FOB)                PARTIAL/EXECUTE/TRADE/BUST), `actionTime`,
                             `auxOrderId` (LINKS TO THE AGGRESSOR -- CME is
                             explicitly named as a venue that supplies this),
                             `tradeId`, `tradePrice`, `executedSize`.
                             Lifecycle and aggressor become OBSERVED.

"MBO" in a product name is marketing. `orderId` and `action` being POPULATED is
a fact. `--mode fields` subscribes, samples live events, and reports the
non-null rate of every Order field -- answering FOD-vs-EOB empirically instead
of waiting on a support ticket.

Note EOB may need explicit enablement (the Java API uses `-Ddxscheme.fob=true`);
whether the dxLink path needs an equivalent, and whether the entitlement covers
it at all, is exactly what this measures.

PROTOCOL (dxLink, from the published AsyncAPI spec)
---------------------------------------------------
  ->  SETUP            channel 0, version, keepaliveTimeout
  <-  SETUP            server echo
  <-  AUTH_STATE       UNAUTHORIZED  (if a token is required)
  ->  AUTH             channel 0, token
  <-  AUTH_STATE       AUTHORIZED
  ->  CHANNEL_REQUEST  odd channel (1,3,5...), service FEED, {contract: AUTO}
  <-  CHANNEL_OPENED
  ->  FEED_SETUP       acceptDataFormat FULL|COMPACT, acceptEventFields
  <-  FEED_CONFIG      the fields the server WILL actually send (authoritative)
  ->  FEED_SUBSCRIPTION add: [{symbol, type}]
  <-  FEED_DATA        events
  ->  KEEPALIVE        periodically on channel 0

COMPACT format packs data as [eventTypeName, [v1..vN, v1..vN, ...]] -- a flat
value list repeated per event, decoded against the field order that FEED_CONFIG
returns (NOT the order we asked for; the server may trim or reorder).

SYMBOLOGY CAVEAT: dxFeed futures symbols are not bare root codes -- they look
like `/ESZ26:XCME`. The exact string for ES/NQ/RTY front month must be confirmed
(the `--symbols` default is a best guess and is expected to need correcting).
If a subscription returns nothing, suspect the symbol before the entitlement:
`--mode probe` dumps raw frames so a bad symbol is visible as silence rather
than an error.

CREDENTIALS: env only -- DXFEED_TOKEN, or DXFEED_USER / DXFEED_PWD. Never
hardcode. The demo endpoint needs no auth.

Usage:
  python dxlink_streamer.py --mode probe                     # demo, dump frames
  python dxlink_streamer.py --mode fields --symbols /ESZ26:XCME
  python dxlink_streamer.py --mode record --symbols /ESZ26:XCME --url wss://...
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import json
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import websockets

# VERIFIED WORKING 2026-09-10. The dxLink GitHub README's "demo.dxfeed.com/
# dxlink-ws/" returns HTTP 400 -- the KB path is the correct one. The demo
# replies AUTH_STATE=AUTHORIZED immediately, so no token is needed there.
DEMO_URL = "wss://demo.dxfeed.com/market-data/dxlink-ws"
VERSION = "0.1-py/1.0.0"

# Every Order/AnalyticOrder field we care about. The server replies in
# FEED_CONFIG with the subset it will actually send -- that reply is the
# authoritative FOD-vs-EOB answer, before a single event arrives.
ORDER_FIELDS = [
    "eventType", "eventSymbol", "eventFlags", "index", "time", "timeNanoPart",
    "sequence", "price", "size", "executedSize", "count", "side", "scope",
    "source", "exchangeCode", "marketMaker",
    # --- EOB / FOB only, below ---
    "action", "actionTime", "orderId", "auxOrderId", "tradeId", "tradePrice",
]
EOB_ONLY = {"action", "actionTime", "orderId", "auxOrderId", "tradeId", "tradePrice"}

#: `acceptEventFields` is keyed BY EVENT TYPE and the server returns the
#: INTERSECTION with what that event actually supports -- asking for Order
#: fields on a Quote yields a near-empty list, which looks like a failure but
#: isn't. So each event type gets its own request list.
FIELDS_BY_EVENT = {
    "Order": ORDER_FIELDS,
    "AnalyticOrder": ORDER_FIELDS + ["icebergPeakSize", "icebergHiddenSize",
                                     "icebergExecutedSize", "icebergType"],
    "SpreadOrder": ORDER_FIELDS + ["spreadSymbol"],
    "Quote": ["eventType", "eventSymbol", "bidPrice", "askPrice", "bidSize",
              "askSize", "bidTime", "askTime", "sequence"],
    "TimeAndSale": ["eventType", "eventSymbol", "time", "sequence", "price",
                    "size", "bidPrice", "askPrice", "aggressorSide",
                    "exchangeCode", "tradeThroughExempt"],
}


class DxLink:
    def __init__(self, url, token=None, symbols=(), event_type="Order",
                 data_format="FULL", outdir="lake/dxlink"):
        self.url, self.token = url, token
        self.symbols = list(symbols)
        self.event_type = event_type
        self.data_format = data_format
        self.outdir = outdir
        self.ch = 1                       # service channels are ODD
        self.ws = None
        self.authorized = False
        self.cfg_fields = None            # what FEED_CONFIG says we'll receive
        self.rows = []
        self.n_events = 0
        self.nonnull = collections.Counter()
        self.seen_types = collections.Counter()
        self.actions = collections.Counter()

    async def _send(self, msg):
        await self.ws.send(json.dumps(msg))

    async def _keepalive(self):
        while True:
            await asyncio.sleep(20)
            try:
                await self._send({"type": "KEEPALIVE", "channel": 0})
            except Exception:
                return

    # ---------------------------------------------------------- decoding
    def _decode(self, data):
        """FEED_DATA -> list of dicts, handling both FULL and COMPACT."""
        out = []
        if not data:
            return out
        if isinstance(data[0], dict):                      # FULL
            return list(data)
        # COMPACT: [typeName, [v1..vN, v1..vN, ...], typeName, [...], ...]
        i = 0
        while i + 1 < len(data):
            etype, values = data[i], data[i + 1]
            fields = (self.cfg_fields or {}).get(etype)
            if fields and values:
                w = len(fields)
                for k in range(0, len(values) - w + 1, w):
                    out.append(dict(zip(fields, values[k:k + w])))
            i += 2
        return out

    def _observe(self, ev):
        self.n_events += 1
        self.seen_types[ev.get("eventType", "?")] += 1
        for k, v in ev.items():
            if v not in (None, "", 0) or k in ("size", "price"):
                self.nonnull[k] += 1
        a = ev.get("action")
        if a is not None:
            self.actions[a] += 1

    # ---------------------------------------------------------- recording
    def flush(self):
        if not self.rows:
            return
        import polars as pl
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        p = os.path.join(self.outdir, f"date={day}")
        os.makedirs(p, exist_ok=True)
        f = os.path.join(p, f"order_{int(time.time())}.parquet")
        pl.DataFrame(self.rows, strict=False).write_parquet(f)
        print(f"  💾 {len(self.rows):,} events -> {f}", flush=True)
        self.rows.clear()

    # ---------------------------------------------------------- report
    def report(self):
        print("\n" + "=" * 78)
        print(f"  events seen: {self.n_events:,}   by type: {dict(self.seen_types)}")
        if self.cfg_fields:
            print(f"\n  FEED_CONFIG says the server will send, for {self.event_type}:")
            print(f"    {self.cfg_fields.get(self.event_type)}")
        if not self.n_events:
            print("\n  No events. Suspect the SYMBOL first (see the symbology caveat in")
            print("  the header) -- a wrong symbol is silent, not an error.")
            return
        print(f"\n  {'field':<18} {'non-null':>9}  {'rate':>6}")
        for f in FIELDS_BY_EVENT.get(self.event_type, ORDER_FIELDS):
            c = self.nonnull.get(f, 0)
            tag = "  <-- EOB only" if f in EOB_ONLY else ""
            print(f"  {f:<18} {c:>9,} {c/self.n_events*100:>5.1f}%{tag}")
        if self.actions:
            print(f"\n  action values seen: {dict(self.actions)}")
        if self.event_type not in ("Order", "AnalyticOrder", "SpreadOrder"):
            return
        got = sum(self.nonnull.get(f, 0) for f in EOB_ONLY)
        print("\n  " + "-" * 74)
        if got == 0:
            print("  VERDICT: no EOB fields populated -> this looks like FULL ORDER DEPTH")
            print("  (index-keyed, no orderId/action). Order lifecycle and icebergs would")
            print("  have to be INFERRED. Ask whether the Enhanced Order Book / FOB is")
            print("  available on this entitlement, and how to enable it over dxLink.")
        else:
            print("  VERDICT: EOB fields ARE populated -> ENHANCED ORDER BOOK. orderId +")
            print("  action give observed lifecycle; auxOrderId gives aggressor linkage.")
            print("  This is the feed the MBO work needs.")

    # ---------------------------------------------------------- run
    async def run(self, mode, seconds, dump):
        async with websockets.connect(self.url, max_size=None) as ws:
            self.ws = ws
            await self._send({"type": "SETUP", "channel": 0, "version": VERSION,
                              "keepaliveTimeout": 60, "acceptKeepaliveTimeout": 60})
            asyncio.create_task(self._keepalive())
            t0, opened, last = time.time(), False, time.time()

            while True:
                if seconds and time.time() - t0 > seconds:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=15)
                except asyncio.TimeoutError:
                    if not opened:
                        print("  ! no response — check the endpoint URL")
                        break
                    continue
                m = json.loads(raw)
                t = m.get("type")
                if dump and t != "FEED_DATA":
                    print(f"  <- {json.dumps(m)[:400]}")

                if t == "AUTH_STATE":
                    st = m.get("state")
                    print(f"  🔑 AUTH_STATE {st}")
                    if st == "UNAUTHORIZED":
                        if not self.token:
                            print("  ! server wants a token; set DXFEED_TOKEN "
                                  "(demo usually needs none)")
                            break
                        await self._send({"type": "AUTH", "channel": 0,
                                          "token": self.token})
                        continue
                    self.authorized = True
                    await self._open_channel()
                    opened = True

                elif t == "SETUP":
                    # Some endpoints authorise implicitly and never send
                    # AUTH_STATE. Do NOT race the AUTH_STATE handler here --
                    # opening the channel twice earns BAD_ACTION "Channel with
                    # id 1 already exists". _open_channel is idempotent.
                    pass

                elif t == "CHANNEL_OPENED":
                    print(f"  📬 channel {m.get('channel')} opened ({m.get('service')})")
                    await self._send({
                        "type": "FEED_SETUP", "channel": self.ch,
                        "acceptDataFormat": self.data_format,
                        "acceptEventFields": {
                            self.event_type: FIELDS_BY_EVENT.get(self.event_type,
                                                                 ORDER_FIELDS)},
                    })

                elif t == "FEED_CONFIG":
                    # FEED_CONFIG arrives TWICE: once right after FEED_SETUP
                    # (empty eventFields) and again once a subscription exists
                    # (the real, authoritative list). Only the populated one
                    # tells us anything, and we must subscribe only once.
                    ef = m.get("eventFields") or {}
                    got = ef.get(self.event_type, [])
                    if got:
                        self.cfg_fields = ef
                        print(f"  ⚙️  FEED_CONFIG format={m.get('dataFormat')} "
                              f"fields={got}")
                        if self.event_type in ("Order", "AnalyticOrder", "SpreadOrder"):
                            missing = sorted(f for f in EOB_ONLY if f not in got)
                            if missing:
                                print(f"     ⚠️  EOB fields NOT offered: {missing}")
                                print("        -> looks like Full Order Depth, not "
                                      "Enhanced Order Book")
                            else:
                                print("     ✅ ALL EOB fields offered -> Enhanced Order Book")
                    if not getattr(self, "_subscribed", False):
                        self._subscribed = True
                        await self._send({
                            "type": "FEED_SUBSCRIPTION", "channel": self.ch,
                            "add": [{"symbol": s, "type": self.event_type}
                                    for s in self.symbols],
                        })
                        print(f"  📡 subscribed {self.event_type} {self.symbols}")

                elif t == "FEED_DATA":
                    for ev in self._decode(m.get("data") or []):
                        self._observe(ev)
                        if mode == "record":
                            self.rows.append(ev)
                    if mode == "record" and len(self.rows) >= 100_000:
                        self.flush()
                    if time.time() - last >= 5:
                        last = time.time()
                        print(f"  📊 events={self.n_events:,}", flush=True)

                elif t == "ERROR":
                    print(f"  ❌ ERROR {m.get('error')}: {m.get('message')}")

        if mode == "record":
            self.flush()
        self.report()

    async def _open_channel(self):
        """Idempotent -- a second CHANNEL_REQUEST on the same id is a BAD_ACTION
        error, and both the SETUP and AUTH_STATE paths can reach here."""
        if getattr(self, "_ch_requested", False):
            return
        self._ch_requested = True
        await self._send({"type": "CHANNEL_REQUEST", "channel": self.ch,
                          "service": "FEED", "parameters": {"contract": "AUTO"}})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["probe", "fields", "record"], default="fields")
    ap.add_argument("--url", default=DEMO_URL,
                    help=f"default {DEMO_URL} (demo, delayed, no auth). Production "
                         "endpoint comes in the dxFeed welcome letter.")
    ap.add_argument("--symbols", nargs="*", default=["/ESZ26:XCME"],
                    help="dxFeed symbology, e.g. /ESZ26:XCME — CONFIRM THIS, a wrong "
                         "symbol is silent")
    ap.add_argument("--event", default="Order",
                    choices=["Order", "AnalyticOrder", "SpreadOrder", "Quote", "TimeAndSale"])
    ap.add_argument("--format", default="FULL", choices=["FULL", "COMPACT"])
    ap.add_argument("--seconds", type=int, default=60, help="0 = until interrupted")
    ap.add_argument("--outdir", default="lake/dxlink")
    a = ap.parse_args()

    token = os.getenv("DXFEED_TOKEN")
    if not token and os.getenv("DXFEED_USER") and os.getenv("DXFEED_PWD"):
        print("  ℹ️  DXFEED_USER/PWD are set but dxLink uses TOKEN auth — ask dxFeed/"
              "DeepCharts how to exchange those for a dxLink token.")
    c = DxLink(a.url, token, a.symbols, a.event, a.format, a.outdir)
    try:
        asyncio.run(c.run(a.mode, a.seconds, dump=(a.mode == "probe")))
    except KeyboardInterrupt:
        if a.mode == "record":
            c.flush()
        c.report()


if __name__ == "__main__":
    main()
