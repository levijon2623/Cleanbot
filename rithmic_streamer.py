# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets", "protobuf", "polars>=1.0.0", "python-dotenv"]
# ///
"""
rithmic_streamer.py
===================
Rithmic R|Protocol client for MARKET-BY-ORDER (Depth By Order) data on CME.

WHY MBO AND NOT THE OPTIONS TAPE
--------------------------------
Options have no MBO at any price -- OPRA is quotes and trades, and the options
books are price-aggregated by construction. Equities have MBO but are fragmented
across 16+ venues, so a single venue's order book is a partial view and order ids
don't survive across venues. **CME is one book**, so ES/NQ/RTY MBO is COMPLETE --
and for the index complex the futures ARE the price-discovery venue.

That matters because SPY / QQQ / IWM are the three rules that came out ROBUST in
`check_fill_sensitivity` (tight ~1-2% spreads, 7-15pp fill bands against 37-75%
edges). ES / NQ / RTY proxy exactly those three. This is the only place in the
project where MBO is both obtainable and pointed at something that works.

It also fits the mechanism we actually established: the edge is INFORMATIONAL,
not mechanical (premium weighting beat delta-weighted hedging demand; the
hedging/participation ratio failed even WITH lookahead). MBO measures order
INTENT and COMMITMENT -- iceberg refill, cancel-to-trade behaviour, queue
persistence -- which is an informational signal, not a hedging one.

WHAT MBO GIVES THAT MBP/FOOTPRINT CANNOT
----------------------------------------
`DepthByOrder` carries, per individual order:
    exchange_order_id      the order's identity across its whole life
    update_type            NEW / CHANGE / DELETE
    transaction_type       BUY / SELL
    depth_price, depth_size
    depth_order_priority   QUEUE POSITION -- the MBO-only field
    sequence_number, nanosecond timestamps
Aggressor-side footprint (what we already have on the options tape) is derived
from trades + BBO. It cannot see a resting order refill, an order pulled without
ever trading, or who is in front of whom in the queue.

CONFORMANCE (blocking -- do this before any of the above matters)
-----------------------------------------------------------------
Per Rithmic (2026-08-24), to be allowed onto any system with real market data:
  1. app_name MUST be exactly  "lele:HyperExposure"   (the "lele:" prefix is
     assigned by Rithmic; the old code sent "HyperExposureClient", which is both
     unprefixed AND a different name from the one submitted).
  2. Log in to the **ORDER PLANT** of Rithmic Test  (infra_type=ORDER_PLANT).
     The old code set no infra_type at all.
  3. Leave the app logged in, then email rapi@rithmic.com to say so.
Run:  python rithmic_streamer.py --mode conformance

COSTS (non-professional certification, from the same thread)
-----------------------------------------------------------
  paper-trading platform, self-written app     $99.99 / month
  CME market depth, per CME Group exchange     $14.00 / month
ES, NQ and RTY are all on **CME**, so ONE $14 depth entitlement covers all three.
Total ~$114/month.

  *** OPEN QUESTION TO ASK RITHMIC BEFORE SUBSCRIBING ***
  The published fee is for "market depth market data", which conventionally means
  aggregated DOM (MBP). Confirm IN WRITING that it entitles
  `RequestDepthByOrderUpdates` (template 117 / DepthByOrder), because MBP vs MBO
  is the entire point of this exercise. Ask also whether CME MBO is redistributed
  to non-professional accounts for ES/NQ/RTY specifically.

CREDENTIALS: read from the environment only -- RITHMIC_USER / RITHMIC_PWD.
Never hardcode them here; `.env` is the place, and the user maintains it.

Usage:
  python rithmic_streamer.py --mode conformance          # order plant, stay logged in
  python rithmic_streamer.py --mode probe                # ticker plant, show template ids
  python rithmic_streamer.py --mode record --symbols ESZ6 NQZ6 RTYZ6
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import os
import ssl
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import websockets

# 🚨 rithmic_protos/ IS NOT IN THE PUBLISHED REPOSITORY.
# It is Rithmic's own protobuf schema, which they distribute to their API
# customers under their agreement -- not ours to redistribute (see NOTICE). A
# bare ImportError here would read as "this project is broken", which is the
# wrong diagnosis and the one a reader would reach first, so say what is
# actually missing and how to get it.
try:
    import rithmic_protos.request_login_pb2 as req_login
    import rithmic_protos.response_login_pb2 as resp_login
    import rithmic_protos.request_heartbeat_pb2 as req_heartbeat
    import rithmic_protos.request_depth_by_order_updates_pb2 as req_mbo
    import rithmic_protos.response_depth_by_order_updates_pb2 as resp_mbo
    import rithmic_protos.depth_by_order_pb2 as dbo_pb
    import rithmic_protos.message_type_pb2 as msg_type
    import rithmic_protos.request_front_month_contract_pb2 as req_front
    import rithmic_protos.response_front_month_contract_pb2 as resp_front
except ImportError as _e:
    raise ImportError(
        "rithmic_protos/ is missing. It holds Rithmic's protobuf schema, which "
        "is theirs to distribute, not ours, so it is excluded from this "
        "repository (see NOTICE).\n\n"
        "  To use this module, get the R|Protocol API .proto files from Rithmic "
        "with your API credentials and generate the bindings yourself:\n"
        "    pip install grpcio-tools\n"
        "    python -m grpc_tools.protoc -I<their proto dir> "
        "--python_out=rithmic_protos <their proto dir>/*.proto\n"
        "    # then add an empty rithmic_protos/__init__.py\n\n"
        "  Nothing in the directional options bot needs this. If you are here "
        "for bot_runner.py, ignore this module entirely."
    ) from _e

URI = "wss://rituz00100.rithmic.com:443"
# 🚨 YOUR OWN, NOT THIS ONE. Rithmic assigns the app-name string per customer and
# the login is rejected if it does not match theirs exactly, so a hardcoded value
# is both useless to you and someone else's identifier to publish. Set
# RITHMIC_APP_NAME in .env to whatever Rithmic assigned you.
APP_NAME = os.getenv("RITHMIC_APP_NAME", "").strip() or "CHANGE_ME:SetRithmicAppName"
APP_VERSION = "1.0.0.0"
TEMPLATE_VERSION = "3.9"

T_LOGIN_REQ, T_LOGIN_RESP = 10, 11
T_HEARTBEAT_REQ, T_HEARTBEAT_RESP = 18, 19
T_FRONT_REQ, T_FRONT_RESP = 113, 114
T_MBO_REQ, T_MBO_RESP = 117, 118

INFRA = {"ticker": req_login.RequestLogin.SysInfraType.TICKER_PLANT,
         "order": req_login.RequestLogin.SysInfraType.ORDER_PLANT,
         "history": req_login.RequestLogin.SysInfraType.HISTORY_PLANT}


def _template_id(raw: bytes) -> int:
    """Every R|Protocol message carries template_id (field 154467). `MessageType`
    exists precisely to read it before dispatching -- the old code string-scraped
    `str(raw_bytes)` looking for 'update_type:', which only ever appears in
    protobuf TEXT format (i.e. after a successful parse). It therefore never
    matched, the book stayed empty, and get_live_imbalance() returned a neutral
    1.000 forever, silently."""
    m = msg_type.MessageType()
    try:
        m.ParseFromString(raw)
        return int(m.template_id)
    except Exception:
        return -1


class RithmicMBO:
    def __init__(self, user, password, system_name="Rithmic Test",
                 plant="ticker", symbols=(), exchange="CME", outdir="lake/mbo"):
        self.user, self.password = user, password
        self.system_name, self.plant = system_name, plant
        self.symbols = list(symbols)
        self.exchange = exchange
        self.outdir = outdir
        self.ws = None
        self.connected = False
        self._book = {}                       # order_id -> dict
        self._rows = []                       # buffered updates for recording
        self._seen_templates = collections.Counter()
        self._n_updates = 0

    # ------------------------------------------------------------ book
    def _apply(self, d):
        """Apply one DepthByOrder update to the live book."""
        oid = d.exchange_order_id
        if not oid:
            return
        ut = d.update_type
        if ut == dbo_pb.DepthByOrder.UpdateType.NEW:
            self._book[oid] = dict(side=d.transaction_type, price=d.depth_price,
                                   size=d.depth_size, t0=time.time(),
                                   peak=d.depth_size, iceberg=False,
                                   prio=d.depth_order_priority)
        elif ut == dbo_pb.DepthByOrder.UpdateType.CHANGE:
            o = self._book.get(oid)
            if o is None:
                return
            # size INCREASING on an existing order id = hidden quantity refilled.
            # A genuinely new order would arrive as NEW with a new id, so this is
            # the iceberg tell and it is only visible with order-level data.
            if d.depth_size > o["peak"]:
                o["iceberg"] = True
                o["peak"] = d.depth_size
            o["size"], o["price"] = d.depth_size, d.depth_price
            o["prio"] = d.depth_order_priority
        elif ut == dbo_pb.DepthByOrder.UpdateType.DELETE:
            self._book.pop(oid, None)

    def imbalance(self):
        """Committed-capital-weighted bid/ask ratio.

        Weighting is a PRIOR, not a measurement: flash orders (<1s) are
        discounted as likely non-committal, resting orders (>5s) are favoured,
        detected icebergs are weighted heavily. None of this is validated yet --
        it needs recorded history and the same walk-forward treatment as
        everything else in METHODOLOGY.md. Do not trade it on faith.
        """
        if not self._book:
            return 1.0
        bid = ask = 1e-3
        now = time.time()
        for o in list(self._book.values()):
            age = now - o["t0"]
            w = 0.2 if age < 1.0 else (1.5 if age > 5.0 else 1.0)
            if o["iceberg"]:
                w = 5.0
            v = o["size"] * w
            if o["side"] == dbo_pb.DepthByOrder.TransactionType.BUY:
                bid += v
            else:
                ask += v
        return round(bid / ask, 3)

    # ------------------------------------------------------------ record
    def _buffer(self, d):
        self._rows.append(dict(
            recv_ns=time.time_ns(), symbol=d.symbol, exchange=d.exchange,
            seq=d.sequence_number, update_type=int(d.update_type),
            side=int(d.transaction_type), price=d.depth_price, size=d.depth_size,
            priority=d.depth_order_priority, order_id=d.exchange_order_id,
            ssboe=d.ssboe, usecs=d.usecs,
            src_ssboe=d.source_ssboe, src_nsecs=d.source_nsecs))

    def flush(self):
        if not self._rows:
            return
        import polars as pl
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        p = os.path.join(self.outdir, f"date={day}")
        os.makedirs(p, exist_ok=True)
        f = os.path.join(p, f"mbo_{int(time.time())}.parquet")
        pl.DataFrame(self._rows).write_parquet(f)
        print(f"  💾 flushed {len(self._rows):,} updates -> {f}", flush=True)
        self._rows.clear()

    # ------------------------------------------------------------ protocol
    async def _login(self, ws):
        r = req_login.RequestLogin()
        r.template_id = T_LOGIN_REQ
        r.template_version = TEMPLATE_VERSION
        r.user = self.user
        r.password = self.password
        r.app_name = APP_NAME
        r.app_version = APP_VERSION
        r.system_name = self.system_name
        r.infra_type = INFRA[self.plant]
        await ws.send(r.SerializeToString())
        raw = await ws.recv()
        resp = resp_login.ResponseLogin()
        resp.ParseFromString(raw)
        ok = resp.template_id == T_LOGIN_RESP and not (resp.rp_code and resp.rp_code[0] != "0")
        print(f"  {'🟢' if ok else '❌'} login  system={self.system_name}  plant={self.plant}  "
              f"app_name={APP_NAME!r}  rp_code={list(resp.rp_code)}")
        return ok

    async def _heartbeat(self, ws):
        while self.connected:
            try:
                h = req_heartbeat.RequestHeartbeat()
                h.template_id = T_HEARTBEAT_REQ
                await ws.send(h.SerializeToString())
                await asyncio.sleep(3)
            except Exception:
                break

    async def _subscribe(self, ws):
        for s in self.symbols:
            r = req_mbo.RequestDepthByOrderUpdates()
            r.template_id = T_MBO_REQ
            r.symbol = s
            r.exchange = self.exchange
            r.request = req_mbo.RequestDepthByOrderUpdates.Request.SUBSCRIBE
            await ws.send(r.SerializeToString())
            print(f"  📡 subscribe DepthByOrder {s}@{self.exchange}")

    async def _run(self, mode, seconds):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        async with websockets.connect(URI, ssl=ctx) as ws:
            self.ws = ws
            if not await self._login(ws):
                return
            self.connected = True
            asyncio.create_task(self._heartbeat(ws))

            if mode == "conformance":
                print("\n  ✅ Logged in to the ORDER PLANT with the conformance app_name.")
                print("     LEAVE THIS RUNNING and email rapi@rithmic.com to say the app")
                print("     is logged in, per their 2026-08-24 instructions.\n")
            else:
                await self._subscribe(ws)

            t0, last = time.time(), time.time()
            while self.connected:
                if seconds and time.time() - t0 > seconds:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    continue
                tid = _template_id(raw)
                self._seen_templates[tid] += 1
                if tid in (T_HEARTBEAT_RESP, T_LOGIN_RESP):
                    continue
                if tid == T_MBO_RESP:
                    r = resp_mbo.ResponseDepthByOrderUpdates()
                    r.ParseFromString(raw)
                    print(f"  ↩️  subscribe ack rp_code={list(r.rp_code)}")
                    continue
                # any other template: try DepthByOrder. The streamed depth
                # template id is NOT in the request/response pair and is not
                # guessed here -- if it parses and carries a symbol, it's ours.
                d = dbo_pb.DepthByOrder()
                try:
                    d.ParseFromString(raw)
                except Exception:
                    continue
                if not d.symbol:
                    continue
                self._apply(d)
                self._n_updates += 1
                if mode == "record":
                    self._buffer(d)
                    if len(self._rows) >= 200_000:
                        self.flush()
                if time.time() - last >= 5:
                    last = time.time()
                    print(f"  📊 updates={self._n_updates:,} book={len(self._book):,} "
                          f"imbalance={self.imbalance():.3f} "
                          f"icebergs={sum(1 for o in self._book.values() if o['iceberg'])}",
                          flush=True)
            self.connected = False
        if mode == "record":
            self.flush()
        print("\n  template_ids seen:", dict(self._seen_templates))
        print("  (map the streamed DepthByOrder id from the R|Protocol reference guide "
              "in the dev-kit zip and pin it here once confirmed)")


async def diagnose(user, pwd, system):
    """Which SYSTEMS exist, and which PLANTS does this credential actually reach?

    `rp_code 13 / permission denied` on a login means the credential is VALID but
    not ENTITLED for that plant -- a bad password returns a different code. So
    probing each plant separates the two causes:
      * ONLY the order plant denied -> the Test credential is data-plant-only.
        Conformance needs order-plant permission; ask Rithmic to add it (they
        said "if you are an FCM or IB, use a trader login id for this step",
        which implies the order plant expects a TRADING login, not a data one).
      * EVERY plant denied -> account-level. Per Rithmic's 2026-08-20 mail you
        MUST first log in to Rithmic Test with R|Trader / R|Trader Pro and
        digitally sign any agreements, or R|API+ login fails even with valid
        credentials.
    """
    import rithmic_protos.request_rithmic_system_info_pb2 as rsi
    import rithmic_protos.response_rithmic_system_info_pb2 as rsr
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print("\n  -- step 1: which systems does the server advertise? --")
    try:
        async with websockets.connect(URI, ssl=ctx) as ws:
            r = rsi.RequestRithmicSystemInfo()
            r.template_id = 16
            await ws.send(r.SerializeToString())
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            resp = rsr.ResponseRithmicSystemInfo()
            resp.ParseFromString(raw)
            names = list(resp.system_name)
            print(f"     systems: {names}")
            print(f"     rp_code: {list(resp.rp_code)}")
            if system not in names:
                print(f"     ⚠️  {system!r} is NOT in that list -- the system_name string must match EXACTLY")
    except Exception as e:
        print(f"     ! {type(e).__name__}: {e}")

    print(f"\n  -- step 2: which plants does this credential reach on {system!r}? --")
    for plant in ("ticker", "history", "order"):
        try:
            async with websockets.connect(URI, ssl=ctx) as ws:
                r = req_login.RequestLogin()
                r.template_id = T_LOGIN_REQ
                r.template_version = TEMPLATE_VERSION
                r.user, r.password = user, pwd
                r.app_name, r.app_version = APP_NAME, APP_VERSION
                r.system_name = system
                r.infra_type = INFRA[plant]
                await ws.send(r.SerializeToString())
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                resp = resp_login.ResponseLogin()
                resp.ParseFromString(raw)
                code = list(resp.rp_code)
                ok = code in ([], ["0"])
                print(f"     {'🟢' if ok else '❌'} {plant:<8} rp_code={code}")
        except Exception as e:
            print(f"     ❌ {plant:<8} {type(e).__name__}: {e}")
    print("\n     If ONLY `order` is denied, this is an entitlement on the Test account,")
    print("     not a market-data subscription -- reply to rapi@rithmic.com saying the")
    print("     conformance login to the order plant returns rp_code 13 and ask them to")
    print("     enable it (or to confirm conformance can be done on the ticker plant).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["conformance", "probe", "record", "diagnose"],
                    default="probe")
    ap.add_argument("--system", default="Rithmic Test")
    ap.add_argument("--symbols", nargs="*", default=["ESZ6"],
                    help="front-month CME symbols. ES/NQ/RTY roll to the Z (Dec) "
                         "contract the Thursday before the 3rd Friday of Sep -- the old "
                         "hardcoded ESU6 goes stale at every quarterly roll.")
    ap.add_argument("--exchange", default="CME")
    ap.add_argument("--seconds", type=int, default=0, help="0 = run until interrupted")
    ap.add_argument("--outdir", default="lake/mbo")
    a = ap.parse_args()

    user = os.getenv("RITHMIC_USER")
    pwd = os.getenv("RITHMIC_PWD")
    if not user or not pwd:
        sys.exit("set RITHMIC_USER and RITHMIC_PWD in .env (never hardcode them here)")

    if a.mode == "diagnose":
        asyncio.run(diagnose(user, pwd, a.system))
        return

    plant = "order" if a.mode == "conformance" else "ticker"
    c = RithmicMBO(user, pwd, a.system, plant, a.symbols, a.exchange, a.outdir)
    try:
        asyncio.run(c._run(a.mode, a.seconds))
    except KeyboardInterrupt:
        c.connected = False
        if a.mode == "record":
            c.flush()
        print("\n  stopped.")


if __name__ == "__main__":
    main()
