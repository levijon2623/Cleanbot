import os
import time
import uuid
import re
import threading
import requests
from datetime import datetime
from dotenv import load_dotenv

# Webull Official SDK Imports
from webull.core.client import ApiClient
from webull.data.data_client import DataClient
from webull.trade.trade_client import TradeClient

# Webull MQTT Streaming Imports
from webull.data.data_streaming_client import DataStreamingClient
from webull.data.common.category import Category
from webull.data.common.subscribe_type import SubscribeType

from config import OPTIONS_TRADING_LEVEL


def quiet_webull_logging():
    """Silence the SDK's own file logging. Safe to call repeatedly.

    🚨 WHY. The SDK attaches a TimedRotatingFileHandler to its `webull.*`
    loggers, and several clients (data, trade, streaming) share ONE file,
    webull_data_streaming_sdk.log. On Windows a file cannot be renamed while
    another handle holds it open, so every hourly rollover raises

        PermissionError: [WinError 32] ... used by another process

    and logging prints a full "--- Logging error ---" traceback for EVERY
    record. On a 0.1s loop that buries the bot's own output. (Lines also
    appeared twice, which is the same cause: a handler added per client.)

    `bot_runner.py`'s `logging.getLogger().setLevel(CRITICAL)` does not help --
    these loggers carry their own handlers and levels, so they never consult the
    root. Handlers have to be removed from the `webull.*` tree directly, and
    propagate switched off so nothing re-emits upward.

    Nothing functional is lost: authentication, connection and subscription
    state are already printed by this module and by bot_runner.
    """
    import logging as _lg
    _lg.raiseExceptions = False          # never print a logging traceback again
    names = [n for n in _lg.root.manager.loggerDict
             if n == "webull" or n.startswith("webull.")]
    for n in names + ["webull"]:
        lg = _lg.getLogger(n)
        for h in list(lg.handlers):
            try:
                h.close()
            except Exception:
                pass
            lg.removeHandler(h)
        lg.propagate = False
        lg.setLevel(_lg.CRITICAL)


def _stdlib_ssl_context():
    """A REAL ssl.SSLContext for paho, not truststore's. None if not needed.

    🚨 WHY THIS EXISTS. `pip_system_certs.pth` in site-packages injects pip's
    vendored `truststore` into the `ssl` module at INTERPRETER STARTUP, so every
    `ssl.SSLContext(...)` in the process -- including the one paho builds inside
    `QuotesClient.__init__`'s bare `tls_set()` -- becomes the Windows-native
    verifier. Against data-api.webull.com:1883 that path raises

        SSLCertVerificationError: ('Peer sent no certificates to verify',)

    and it fails IDENTICALLY with `cert_reqs=CERT_NONE` and with an explicit
    certifi bundle, because truststore ignores both. Measured 2026-09-22: the
    network is fine (raw TLS to that host succeeds, TLSv1.3, cn=*.webull.com)
    and the Linux box is unaffected -- truststore's WINDOWS backend is the
    broken one, which is why the cloud deploy never saw this.

    The fix is surgical on purpose: build the context from the stdlib class,
    hand it to paho via `tls_set_context()`, and leave truststore injected so
    pip and `requests` keep the behaviour they were installed for. Returns None
    when `ssl` is unpatched, so this is a no-op everywhere else.
    """
    import ssl
    if ssl.SSLContext.__module__ == "ssl":
        return None                       # not patched; nothing to work around
    try:
        from pip._vendor import truststore
        truststore.extract_from_ssl()
        try:
            return ssl.create_default_context()
        finally:
            truststore.inject_into_ssl()   # restore for everyone else
    except Exception:
        return None


def _f(v, default=0.0):
    """float() that never raises."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

class WebullLiveTickBuilder:
    def __init__(self):
        # Stores the closed 2-minute candles for the EMA/MACD math
        self.historical_bars = {} 
        self.live_bars = {}
        
        # Shared state for Order Book Imbalance calculations without REST APIs
        self.rolling_bids = {}
        self.rolling_asks = {}
        self.live_imbalances = {}
        self.smoothing_period = 5 
        
        # { ticker: { minute_epoch: cumulative session volume } } from SNAPSHOT
        self.live_volume = {}

        # --- NEW: ZERO-LATENCY OPTIONS QUOTE MEMORY ---
        # Format: { "SPY260821C00500000": {"bid": 1.50, "ask": 1.52, "timestamp": 1234567.89} }
        self.live_option_quotes = {}

    def process_equity_tick(self, ticker, price, timestamp_seconds=None):
        """Builds 2-minute candles dynamically from sub-second equity ticks."""
        if timestamp_seconds is None:
            timestamp_seconds = int(time.time())
        # wall-clock time of the last tick per ticker: lets a consumer tell a
        # FROZEN price from a quiet one (manual_orders underlying TP/SL)
        self.__dict__.setdefault("last_tick_at", {})[ticker] = time.time()

        minute_bucket = (timestamp_seconds // 120) * 120
        
        if ticker not in self.live_bars:
            self.live_bars[ticker] = {
                'timestamp': minute_bucket, 'open': price, 'high': price, 
                'low': price, 'close': price
            }
            if ticker not in self.historical_bars:
                self.historical_bars[ticker] = []
            return

        current_bar = self.live_bars[ticker]
        
        if minute_bucket > current_bar['timestamp']:
            self.historical_bars[ticker].append({'close': current_bar['close']})
            if len(self.historical_bars[ticker]) > 40:
                self.historical_bars[ticker].pop(0)
                
            self.live_bars[ticker] = {
                'timestamp': minute_bucket, 'open': price, 'high': price, 
                'low': price, 'close': price
            }
        else:
            current_bar['high'] = max(current_bar['high'], price)
            current_bar['low'] = min(current_bar['low'], price)
            current_bar['close'] = price

    def process_equity_volume(self, ticker, cum_volume, timestamp_seconds=None):
        """Cumulative SESSION volume from a SNAPSHOT, bucketed by minute.

        SNAPSHOT reports volume CUMULATIVELY for the regular session, so a
        per-minute bar is the difference between consecutive minute buckets --
        which is why the latest value per minute is kept, matching how
        FlowMomentumTracker buckets cumulative flow.

        `volume` is the regular-session figure; the proto carries `ext_volume`
        and `ovn_volume` separately and those are deliberately NOT summed in.
        """
        if timestamp_seconds is None:
            timestamp_seconds = int(time.time())
        m = timestamp_seconds // 60
        b = self.live_volume.setdefault(ticker, {})
        # Cumulative session volume only ever rises. A LOWER value means a new
        # session (or a bad print), so start over rather than emit a negative
        # bar. Compare against the LATEST value seen, not this minute's -- a
        # reset almost always arrives on a new minute, which a same-minute
        # comparison sails straight past.
        if b:
            last = b[max(b)]
            if cum_volume < last:
                b.clear()
        b[m] = cum_volume
        if len(b) > 3000:
            for k in sorted(b)[:-3000]:
                del b[k]

    def minute_volume(self, ticker):
        """{minute_epoch: volume traded IN that minute}, closed minutes only."""
        b = self.live_volume.get(ticker) or {}
        ms = sorted(b)
        now_m = int(time.time() // 60)
        out = {}
        for i in range(1, len(ms)):
            m = ms[i]
            if m >= now_m:
                continue                     # still forming
            if ms[i - 1] != m - 1:
                continue                     # gap: the difference is not one bar
            d = b[m] - b[ms[i - 1]]
            if d >= 0:
                out[m] = d
        return out

    def process_equity_imbalance(self, ticker, bid_size, ask_size):
        """Tracks streaming NBBO size to calculate Live Order Book Imbalance locally."""
        if ticker not in self.rolling_bids:
            self.rolling_bids[ticker] = []
            self.rolling_asks[ticker] = []
            
        self.rolling_bids[ticker].append(bid_size)
        self.rolling_asks[ticker].append(ask_size)
        
        if len(self.rolling_bids[ticker]) > self.smoothing_period:
            self.rolling_bids[ticker].pop(0)
            self.rolling_asks[ticker].pop(0)
            
        avg_bid = sum(self.rolling_bids[ticker]) / len(self.rolling_bids[ticker])
        avg_ask = sum(self.rolling_asks[ticker]) / len(self.rolling_asks[ticker])
        
        if avg_ask > 0:
            self.live_imbalances[ticker] = round(avg_bid / avg_ask, 2)
        else:
            self.live_imbalances[ticker] = 1.0

    def process_option_quote(self, occ_symbol, bid_price, ask_price):
        """Updates the local memory with the absolute latest Option Bid/Ask."""
        self.live_option_quotes[occ_symbol] = {
            "bid": float(bid_price),
            "ask": float(ask_price),
            "timestamp": time.time()
        }

    def get_closes_for_math(self, ticker):
        if ticker not in self.historical_bars or ticker not in self.live_bars:
            return []
        closed_prices = [bar['close'] for bar in self.historical_bars[ticker]]
        live_price = self.live_bars[ticker]['close']
        return closed_prices + [live_price]

class WebullGammaClient:
    def __init__(self, app_key: str, app_secret: str, account_id: str, paper_trading: bool = False):
        print(f"⚙️ Initializing Webull Gamma Client (Paper Trading: {paper_trading})...")
        
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_id = account_id
        
        # FIX: Updated to region_id for the current Webull SDK
        self.api_client = ApiClient(app_key, app_secret, region_id="us")
        if paper_trading:
            self.api_client.add_endpoint("us", "api.sandbox.webull.com")
        else:
            self.api_client.add_endpoint("us", "api.webull.com")
            
        self.data_client = DataClient(self.api_client)
        self.trade_client = TradeClient(self.api_client)
        self.tick_builder = WebullLiveTickBuilder()
        
        # MQTT Streamer Instance
        self._snap_logged = False     # print the first SNAPSHOT parse once
        self.stream_client = None
        self.active_option_subs = set()
        
        # The clients above just attached their rotating file handlers; strip
        # them before the first hourly rollover fails. Called again after the
        # streaming client is built, since that adds its own.
        quiet_webull_logging()
        print("✅ Webull Data & Trade Clients successfully authenticated.")

    def seed_historical_candles(self, tickers: list):
        print("🌱 Seeding historical equity candles via Yahoo Finance...")
        for ticker in tickers:
            try:
                url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=2m&range=1d"
                headers = {'User-Agent': 'Mozilla/5.0'}
                data = requests.get(url, headers=headers, timeout=5).json()
                closes = data['chart']['result'][0]['indicators']['quote'][0]['close']
                
                if ticker not in self.tick_builder.historical_bars:
                    self.tick_builder.historical_bars[ticker] = []
                    
                for i in range(max(0, len(closes)-40), len(closes)):
                    if closes[i] is not None:
                        self.tick_builder.historical_bars[ticker].append({'close': float(closes[i])})
                print(f"  ├─ {ticker}: Successfully seeded {len(self.tick_builder.historical_bars[ticker])} historical 2m equity candles.")
            except Exception as e:
                print(f"  └─ 🚨 Failed to seed {ticker} history: {e}")

    def start_tick_stream(self, equity_tickers: list):
        """Initializes the Master MQTT Connection and subscribes to Equities."""
        if not equity_tickers: return
        
        session_id = f"tick_stream_{int(time.time())}"
        self.stream_client = DataStreamingClient(
            app_key=self.app_key, app_secret=self.app_secret,
            region_id="us", session_id=session_id,
            http_host="api.webull.com", mqtt_host="data-api.webull.com"
        )

        # QuotesClient.__init__ already called tls_set(); override it with a
        # stdlib context where truststore has been injected into ssl. No-op
        # otherwise. See _stdlib_ssl_context.
        quiet_webull_logging()     # DataStreamingClient added its own handler

        _ctx = _stdlib_ssl_context()
        if _ctx is not None:
            try:
                # paho refuses to reconfigure -- tls_set_context() raises
                # "SSL/TLS has already been configured" because
                # QuotesClient.__init__ called tls_set() first. Clearing the
                # slot lets the public path run and set _tls_insecure with it.
                self.stream_client._ssl_context = None
                self.stream_client.tls_set_context(_ctx)
                print("  🔒 MQTT TLS: stdlib context "
                      "(working around truststore in ssl)")
            except Exception as e:
                print(f"  ⚠️ MQTT TLS override failed ({e}); "
                      f"falling back to the SDK default")

        def on_connect(client, api_client, session_id):
            print(f"✅ Webull MQTT Connected! Subscribing to live equity ticks for {', '.join(equity_tickers)}...")
            # QUOTE is the order book ONLY -- no volume field (Webull Data
            # Streaming API docs, Topic-to-Payload Mapping). SNAPSHOT carries
            # cumulative session `volume` alongside price/open/high/low, which
            # is where the per-minute volume bars come from. Subscribing to both
            # costs one subscription, not one connection.
            client.subscribe(symbols=equity_tickers,
                             category=Category.US_STOCK.name,
                             sub_types=[SubscribeType.QUOTE.name,
                                        SubscribeType.SNAPSHOT.name])

        def on_message(client, topic, quotes):
            try:
                quote_str = str(quotes)
                tp = str(topic or "").lower()

                # ---------------------------------------------------------
                # SNAPSHOT -- routed by TOPIC, never by regex sniffing.
                # The payload contains ext_price / ovn_price / pre_close as
                # well as price, and the equity path below searches for a bare
                # `price:` ANYWHERE in the string. Letting a snapshot fall
                # through to it would feed ext_price into process_equity_tick
                # -> live_bars -> get_spot_price -> STRIKE SELECTION. Route
                # first, parse narrowly, return.
                # ---------------------------------------------------------
                if "snapshot" in tp:
                    sym = re.search(r'symbol:\s*"?([A-Z\.]+)"?', quote_str)
                    if not sym:
                        return
                    symbol = sym.group(1).strip()
                    # (?<![a-z_]) keeps ext_volume / ovn_volume out
                    vm = re.search(r'(?<![a-z_])volume:\s*"?(\d+)', quote_str)
                    if vm:
                        self.tick_builder.process_equity_volume(
                            symbol, int(vm.group(1)))
                    if not self._snap_logged:
                        self._snap_logged = True
                        print(f"  📊 first SNAPSHOT parsed: {symbol} "
                              f"vol={vm.group(1) if vm else 'NO MATCH'}")
                        if not vm:
                            print(f"     raw: {quote_str[:300]}")
                    return
                symbol_match = re.search(r'symbol:([^,]+)', quote_str)
                price_match = re.search(r'regularMarketPrice:([\d\.]+)', quote_str) or re.search(r'price:([\d\.]+)', quote_str)
                
                if symbol_match:
                    symbol = symbol_match.group(1)
                    
                    # --- ROUTE OPTIONS DATA ---
                    if len(symbol) > 6 and any(char.isdigit() for char in symbol): 
                        # Options symbol detected! Extract the live Bid/Ask.
                        ask_match = re.search(r'asks:\s*\[.*?price:([\d\.]+)', quote_str)
                        bid_match = re.search(r'bids:\s*\[.*?price:([\d\.]+)', quote_str)
                        
                        if ask_match and bid_match:
                            self.tick_builder.process_option_quote(
                                occ_symbol=symbol, 
                                bid_price=float(bid_match.group(1)), 
                                ask_price=float(ask_match.group(1))
                            )
                    
                    # --- ROUTE EQUITY DATA ---
                    else:
                        if price_match:
                            price = float(price_match.group(1))
                            self.tick_builder.process_equity_tick(symbol, price)
                            
                        # Equity order book imbalance
                        ask_size_match = re.search(r'asks:\s*\[.*?size:(\d+)\]', quote_str)
                        bid_size_match = re.search(r'bids:\s*\[.*?size:(\d+)\]', quote_str)
                        
                        if ask_size_match and bid_size_match:
                            self.tick_builder.process_equity_imbalance(
                                symbol, int(bid_size_match.group(1)), int(ask_size_match.group(1))
                            )
            except Exception:
                pass
                
        self.stream_client.on_connect_success = on_connect
        self.stream_client.on_quotes_message = on_message
        
        threading.Thread(target=self.stream_client.connect_and_loop_forever, daemon=True).start()

    # =====================================================================
    # DYNAMIC MQTT OPTION SUBSCRIPTIONS
    # =====================================================================
    def subscribe_to_option(self, occ_symbol: str):
        """Tells the active MQTT client to start streaming this specific option."""
        if not self.stream_client or occ_symbol in self.active_option_subs:
            return
            
        print(f"📡 [MQTT] Subscribing to live quotes for {occ_symbol}...")
        try:
            self.stream_client.subscribe(
                symbols=[occ_symbol], 
                category=Category.US_OPTION.name, 
                sub_types=[SubscribeType.QUOTE.name]
            )
            self.active_option_subs.add(occ_symbol)
        except Exception as e:
            print(f"  🚨 Failed to subscribe to {occ_symbol}: {e}")

    def unsubscribe_from_option(self, occ_symbol: str):
        """Frees up bandwidth when a trade is closed."""
        if not self.stream_client or occ_symbol not in self.active_option_subs:
            return
            
        print(f"🔌 [MQTT] Unsubscribing from {occ_symbol}...")
        try:
            self.stream_client.unsubscribe(
                symbols=[occ_symbol], 
                category=Category.US_OPTION.name, 
                sub_types=[SubscribeType.QUOTE.name]
            )
            self.active_option_subs.remove(occ_symbol)
            # Clear it from local memory
            if occ_symbol in self.tick_builder.live_option_quotes:
                del self.tick_builder.live_option_quotes[occ_symbol]
        except Exception as e:
            print(f"  🚨 Failed to unsubscribe from {occ_symbol}: {e}")

    def get_live_option_quote(self, occ_symbol: str):
        """Back-compat wrapper: (bid, ask) only.

        WARNING: a return of (0.0, 0.0) is AMBIGUOUS -- it means either "the
        market has no bid" or "we could not get a quote at all". Callers that
        manage risk MUST use get_live_option_quote_ex() instead and branch on
        `ok`; conflating the two is what let two SMH 0DTE positions ride from
        +144% to $0.01 unmonitored on 2026-09-10.
        """
        bid, ask, _ok = self.get_live_option_quote_ex(occ_symbol)
        return bid, ask

    @staticmethod
    def _snap_px(node, flat_key, list_key):
        """A price out of an option snapshot, whichever shape it arrives in.

        Returns None when the field is ABSENT, which the caller must not
        confuse with a genuine 0.00 bid. Flat string first (what the API
        actually sends), depth list second, in case another endpoint or a
        future version uses it.
        """
        v = node.get(flat_key)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        lst = node.get(list_key)
        if isinstance(lst, list) and lst and isinstance(lst[0], dict):
            try:
                return float(lst[0].get("price"))
            except (TypeError, ValueError):
                pass
        return None

    def get_live_option_quote_ex(self, occ_symbol: str):
        """(bid, ask, ok). `ok` is False when NO quote could be obtained.

        The distinction matters enormously: `ok=False` means we are BLIND on
        that contract and must escalate, whereas `ok=True` with bid 0.0 is a
        real market state (a worthless option) and is itself an exit signal.
        The old single-return form collapsed both into (0.0, 0.0), and the
        monitor loop's `if bid == 0.0: continue` then skipped the position's
        entire exit evaluation -- so an API stall (e.g. throttling from two bot
        instances running at once) silently suspended risk management.
        """
        # 1. WebSocket memory (instant), if the tick is fresh
        if occ_symbol in self.tick_builder.live_option_quotes:
            quote = self.tick_builder.live_option_quotes[occ_symbol]
            if (time.time() - quote["timestamp"]) < 10.0:
                return quote["bid"], quote["ask"], True

        # 2. REST fallback while the socket spins up / after a stale tick
        try:
            resp = self.data_client.option_market_data.get_option_snapshot(category="US_OPTION", symbols=occ_symbol)
            snapshots = resp.json() if hasattr(resp, 'json') else resp
            if isinstance(snapshots, dict) and "data" in snapshots: snapshots = snapshots["data"]
            if isinstance(snapshots, list) and len(snapshots) > 0:
                # 🚨 THE SNAPSHOT CARRIES FLAT `bid`/`ask` STRINGS.
                # This read askList/bidList, which the response does not have,
                # so `.get(..., [{"price": 0}])` took the DEFAULT every single
                # time and returned (0.0, 0.0, True). Per this function's own
                # docstring, ok=True with bid 0.0 is "a real market state (a
                # worthless option) and is itself an exit signal" -- so every
                # REST-fallback quote declared the contract worthless, and any
                # exit priced off it goes out at the 1c floor. Verified against
                # a live IWM 284C on 2026-09-23: the raw response had
                # bid "0.15" / ask "0.16" while this returned 0.0/0.0.
                s0 = snapshots[0] or {}
                bid = self._snap_px(s0, "bid", "bidList")
                ask = self._snap_px(s0, "ask", "askList")
                if bid is None and ask is None:
                    # ABSENT is not ZERO. A real 0.00 bid is a worthless
                    # option; a missing field means we are blind, and the two
                    # must not both come back as ok=True.
                    self._quote_fail(occ_symbol,
                                     f"no bid/ask in snapshot: "
                                     f"{sorted(s0)[:12]}")
                    return 0.00, 0.00, False
                return (bid or 0.0), (ask or 0.0), True
            self._quote_fail(occ_symbol, f"unexpected snapshot shape: {type(snapshots).__name__}")
        except Exception as e:
            # never silently swallow: a quiet quote outage IS the failure mode
            self._quote_fail(occ_symbol, f"{type(e).__name__}: {e}")

        return 0.00, 0.00, False

    def _quote_fail(self, occ_symbol: str, why: str):
        """Rate-limited logging of quote outages (once per contract per 30s)."""
        if not hasattr(self, "_qf_last"):
            self._qf_last, self._qf_count = {}, {}
        now = time.time()
        self._qf_count[occ_symbol] = self._qf_count.get(occ_symbol, 0) + 1
        if now - self._qf_last.get(occ_symbol, 0.0) >= 30.0:
            self._qf_last[occ_symbol] = now
            print(f"  ⚠️ [QUOTE] {occ_symbol}: no quote "
                  f"({self._qf_count[occ_symbol]} failures) — {why}")

    def calculate_live_imbalance(self, ticker: str) -> float:
        return self.tick_builder.live_imbalances.get(ticker, 1.0)

    def calculate_overnight_imbalance(self, ticker: str) -> float:
        try:
            endpoint_path = "/openapi/market-data/stocks/depths/list"
            params = {"symbol": ticker.upper(), "category": "US_STOCK", "depth": "10", "overnight_required": "true"}
            response = self.api_client.request("GET", endpoint_path, params=params, version="v2")
            data = response.json() if hasattr(response, 'json') else response
            if isinstance(data, dict) and 'data' in data: data = data['data']
            
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            total_bid_vol = sum([float(l.get('volume', l.get('size', 0))) for l in bids])
            total_ask_vol = sum([float(l.get('volume', l.get('size', 0))) for l in asks])
            return round(total_bid_vol / total_ask_vol, 2) if total_ask_vol > 0 else 1.0
        except Exception:
            return 1.0

    def get_account_health(self):
        try:
            account_api = self.trade_client.account_v2 if hasattr(self.trade_client, 'account_v2') else self.trade_client.account
            
            response = None
            for method_name in ['get_account_balance', 'get_balance', 'get_account']:
                if hasattr(account_api, method_name):
                    method = getattr(account_api, method_name)
                    try:
                        response = method(self.account_id)
                        break
                    except TypeError:
                        try:
                            response = method(self.account_id, currency='USD')
                            break
                        except Exception:
                            pass
                    except Exception:
                        pass
                        
            if not response:
                print("🚨 All account balance methods failed in SDK.")
                return None
                
            data = response.json() if hasattr(response, 'json') else response
            if isinstance(data, dict) and 'data' in data: data = data['data']
            
            return data
        except Exception as e:
            print(f"🚨 Failed to fetch account health: {e}")
            return None

    # Contract DIRECTORY per underlying, cached: {ticker: (epoch, [symbols])}.
    # Listings change a few times a day at most (new strikes on a big move),
    # so 15 minutes is plenty -- and it turns a basket shift during a fast
    # move from ~10 paged requests into none.
    _DIR_TTL = 900.0

    def _option_directory(self, ticker: str):
        cache = self.__dict__.setdefault("_dir_cache", {})
        hit = cache.get(ticker)
        if hit and time.time() - hit[0] < self._DIR_TTL:
            return hit[1]
        syms, last_id = [], None
        while True:
            kwargs = {"category": "US_OPTION", "underlying_symbols": ticker, "page_size": 1000}
            if last_id:
                kwargs["last_instrument_id"] = last_id
            resp = self.data_client.instrument.get_option_contracts(**kwargs)
            data = resp.json() if hasattr(resp, "json") else resp
            if not data or not isinstance(data, list):
                break
            syms.extend(c.get("symbol") for c in data if c.get("symbol"))
            if len(data) < 1000:
                break
            last_id = data[-1].get("instrument_id")
        if syms:
            cache[ticker] = (time.time(), syms)
        return syms

    def basket_chain(self, ticker: str, spot_price: float, max_dte: int = 7, band: float = 0.03):
        """The chain the options BASKET needs, fast: contract symbols only.

        🚨 WHY NOT scan_option_chain. That pulls a snapshot for every contract
        within 5% of spot and 14 days -- ~1,500 on SPY, 20 per request with a
        0.3s pause, so a basket shift took tens of seconds, during exactly the
        fast moves that trigger one. The basket only needs strike, expiry and
        call/put, all of which are IN the OCC symbol; the one snapshot field
        it stored (IV) is read nowhere. Same return shape, no snapshots, and
        the directory is cached (_option_directory). 2026-09-28.
        """
        try:
            syms = self._option_directory(ticker)
        except Exception as e:
            print(f"🚨 Option directory fetch failed for {ticker}: {e}")
            return None
        today = datetime.now().date()
        out = []
        for sym in syms:
            if sym[0].isdigit():                  # adjusted contracts (e.g. 1NVDA)
                continue
            try:
                strike = float(sym[-8:]) / 1000.0
                exp = datetime.strptime(f"20{sym[-15:-9]}", "%Y%m%d").date()
            except (ValueError, IndexError):
                continue
            if not (0 <= (exp - today).days <= max_dte):
                continue
            if abs(strike - spot_price) / spot_price > band:
                continue
            out.append({"symbol": sym, "strikePrice": str(strike),
                        "expireDate": exp.isoformat(),
                        "callPut": "Call" if sym[-9] == "C" else "Put"})
        return {"data": out}

    def scan_option_chain(self, ticker: str, spot_price: float = None):
        """Fetches the entire option chain directory and batches requests."""
        print(f"🔍 Fetching full options directory for {ticker}...")
        try:
            all_contracts = []
            last_id = None
            
            while True:
                kwargs = {"category": "US_OPTION", "underlying_symbols": ticker, "page_size": 1000}
                if last_id: kwargs["last_instrument_id"] = last_id
                    
                resp = self.data_client.instrument.get_option_contracts(**kwargs)
                data = resp.json() if hasattr(resp, "json") else resp
                
                if not data or not isinstance(data, list) or len(data) == 0: 
                    break
                
                all_contracts.extend(data)
                if len(data) < 1000: break
                last_id = data[-1].get("instrument_id")
                
            if not all_contracts:
                return {"data": []}

            # Filter out non-standard adjusted options (e.g., 1NVDA) that crash the Webull API
            symbols = [c.get("symbol") for c in all_contracts if c.get("symbol") and not c.get("symbol")[0].isdigit()]
            contract_details = {c.get("symbol"): c for c in all_contracts if c.get("symbol")}
            
            # --- NEW: THE PRE-FILTER OPTIMIZATION ---
            if spot_price and spot_price > 0:
                filtered_symbols = []
                current_time = datetime.now()
                
                for sym in symbols:
                    try:
                        # OCC Format: AAPL260828C00150000 
                        strike = float(sym[-8:]) / 1000.0
                        
                        date_str = sym[-15:-9]
                        exp_date = datetime.strptime(f"20{date_str}", "%Y%m%d")
                        
                        # CRITICAL FIX: Use .date() so today's 0DTEs evaluate to exactly 0 instead of -1
                        dte = (exp_date.date() - current_time.date()).days
                        
                        # Only fetch snapshots for strikes within 5% of spot price 
                        if abs(strike - spot_price) / spot_price <= 0.05 and 0 <= dte <= 14:
                            filtered_symbols.append(sym)
                    except Exception:
                        pass
                symbols = filtered_symbols
            # ----------------------------------------
            
            merged_chain_data = []
            print(f"  ├─ Discovered {len(contract_details)} total contracts. Pulling snapshots for {len(symbols)} relevant strikes...")
            
            for i in range(0, len(symbols), 20):
                batch = symbols[i:i + 20]
                try:
                    snap_resp = self.data_client.option_market_data.get_option_snapshot(category="US_OPTION", symbols=",".join(batch))
                    snapshots = snap_resp.json() if hasattr(snap_resp, "json") else snap_resp
                    if isinstance(snapshots, dict) and "data" in snapshots: snapshots = snapshots["data"]
                    if isinstance(snapshots, dict): snapshots = [snapshots]

                    if not snapshots: continue

                    for snap in snapshots:
                        sym = snap.get("symbol")
                        if not sym: continue
                        
                        # --- BULLETPROOF OCC PARSING ---
                        # OCC Format: AAPL260828C00150000
                        # sym[-15:-9] = '260828' (YYMMDD)
                        # sym[-9] = 'C' or 'P'
                        # sym[-8:] = '00150000' -> 150.00
                        try:
                            date_str = sym[-15:-9]
                            expire_val = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}"
                            strike_val = str(float(sym[-8:]) / 1000.0)
                            cp = "Call" if sym[-9] == "C" else "Put"
                            
                            oi_val = snap.get("openInterest") or snap.get("open_interest") or 0
                            iv_val = snap.get("impliedVolatility") or snap.get("implied_volatility") or 0
                            
                            merged_chain_data.append({
                                "symbol": sym,
                                "strikePrice": strike_val,
                                "expireDate": expire_val,
                                "callPut": cp,
                                "openInterest": int(float(oi_val)),
                                "volume": int(float(snap.get("volume") or 0)),
                                "impliedVolatility": float(iv_val)
                            })
                        except Exception:
                            pass
                except Exception as e:
                    print(f"  ⚠️ Webull Snapshot Rate Limit Warning: {e}")
                    time.sleep(1) # Back off if they yell at us
                
                # FIX: Relax rate limit from 0.1 to 0.3 to respect Webull OpenAPI limits
                time.sleep(0.3) 
                
            return {"data": merged_chain_data}
            
        except Exception as e:
            print(f"🚨 Option chain fetch failed: {e}")
            return None

    def place_option_bracket(self, ticker: str, option_id: str, quantity: int,
                             tp_limit: float, sl_stop: float,
                             tif: str = "GTC", dry: bool = False,
                             combo: str = "PAIR"):
        """Attach an OCO take-profit / stop-loss pair to an existing LONG
        option position.

        🚨 THE SDK HAS ALWAYS SUPPORTED THIS; WE NEVER WIRED IT.
        `order_v3.place_order(account_id, new_orders, client_combo_order_id)`
        takes a LIST and a combo id, `ComboType` carries STOP_LOSS /
        STOP_PROFIT / OCO / OTOCO, and `PlaceOptionRequest` -- the OPTIONS
        request class, not the equity one -- has set_client_combo_order_id.
        place_option_order hardcodes combo_type "NORMAL" and sends a
        single-element list, which is why it only ever made plain orders.

        The payload mirrors, field for field, what Webull's own app produced
        on this account on 2026-09-23: two SELL orders for the FULL position
        sharing one combo_order_id, one tagged STOP_PROFIT carrying a LIMIT
        price and one tagged STOP_LOSS carrying a stop price, both GTC.

        🚨 "PAIR" IS THE ONLY FRAMING THE ENDPOINT ACCEPTS. Probed against the
        live API 2026-09-24 (probe_bracket_shape.py):
          PAIR    STOP_PROFIT + STOP_LOSS  -> ACCEPTED, both legs on the book
          OCO     both legs "OCO"          -> OPENAPI_PARAM_ERR invalid combo_type
          SLP     both STOP_LOSS_PROFIT    -> OPENAPI_PARAM_ERR invalid combo_type
          SINGLE  one order, both prices   -> OPENAPI_PARAM_ERR invalid combo_type
        So most of `ComboType` is not usable on this endpoint despite being in
        the enum, and the shape a bracket READS BACK in is the shape it is
        WRITTEN in. The alternatives stay only so the probe can be re-run.

        🚨 A REJECTION HERE IS USUALLY ABOUT THE POSITION, NOT THE PAYLOAD.
        The first attempt failed with
        OPENAPI_PROMPT_INSUFFICIENT_BUYING_POWER_AT_CLOSE, which sent me
        hunting for a different framing -- the framing was right all along.
        That attempt was 2x an IWM 280 PUT; the one that succeeded was 1x an
        IWM 282 CALL. Same code, same two-legs-per-position ratio. The broker
        margins the UNNETTED leg, and a short 280 put wants ~$28,000 of
        collateral against a $38 account while a short call does not price
        the same way. If you see _AT_CLOSE, look at what is being bracketed
        and at the buying power before touching this function.

        `dry=True` returns the exact payload without sending it.
        """
        tp_leg = self._parse_occ_to_leg(option_id, "SELL", quantity)
        sl_leg = self._parse_occ_to_leg(option_id, "SELL", quantity)
        if not tp_leg or not sl_leg:
            return {"accepted": False, "error": f"bad OCC {option_id}"}

        # NOT `combo` -- that is the caller's framing selector, and assigning
        # the uuid to it shadowed the parameter so `mode` read a hex string.
        combo_id = uuid.uuid4().hex

        def _order(combo_type, order_type, **px):
            o = {
                "client_order_id": uuid.uuid4().hex,
                "combo_type": combo_type,
                "order_type": order_type,
                "quantity": str(quantity),
                "option_strategy": "SINGLE",
                "side": "SELL",
                "time_in_force": tif,
                "entrust_type": "QTY",
                "position_intent": "SELL_TO_CLOSE",
                "legs": [self._parse_occ_to_leg(option_id, "SELL", quantity)],
            }
            o.update({k: str(round(v, 2)) for k, v in px.items()})
            return o

        mode = str(combo).upper()
        if mode == "SINGLE":
            # ONE order carrying BOTH prices. ComboType 2 is literally named
            # "Stop loss Profit", which reads like one instruction rather than
            # two, and a single order cannot over-commit the position -- which
            # is exactly what OPENAPI_PROMPT_INSUFFICIENT_BUYING_POWER_AT_CLOSE
            # was complaining about.
            one = _order("STOP_LOSS_PROFIT", "STOP_LOSS_LIMIT",
                         limit_price=float(tp_limit),
                         stop_price=float(sl_stop))
            orders = [one]
        else:
            tag = {"PAIR": ("STOP_PROFIT", "STOP_LOSS"),
                   "OCO": ("OCO", "OCO"),
                   "SLP": ("STOP_LOSS_PROFIT", "STOP_LOSS_PROFIT")}.get(mode)
            if not tag:
                return {"accepted": False, "error": f"unknown combo {combo!r}"}
            orders = [_order(tag[0], "LIMIT", limit_price=float(tp_limit)),
                      _order(tag[1], "STOP_LOSS", stop_price=float(sl_stop))]
        tp, sl = orders[0], orders[-1]
        payload = {"account_id": self.account_id, "new_orders": orders,
                   "client_combo_order_id": combo_id}
        if dry:
            return {"accepted": False, "dry": True, "payload": payload,
                    "combo_order_id": combo_id}

        print(f"⚡ [WEBULL] BRACKET[{mode}] {quantity}x {option_id}  "
              f"TP {tp_limit:.2f} / SL {sl_stop:.2f}  ({tif})")
        try:
            resp = self.trade_client.order_v3.place_order(
                self.account_id, new_orders=orders,
                client_combo_order_id=combo_id)
            data = resp.json() if hasattr(resp, "json") else resp
        except Exception as e:
            print(f"  🚨 bracket place failed: {e}")
            return {"accepted": False, "error": str(e),
                    "combo_order_id": combo_id, "payload": payload}
        _oid, accepted = self._parse_place_response(data)
        coids = [o["client_order_id"] for o in orders]
        print(f"  └─ accepted={accepted} combo={combo_id} coids={coids}")
        return {"accepted": accepted, "combo_order_id": combo_id,
                "client_order_ids": coids, "raw": data}

    def _parse_occ_to_leg(self, option_id: str, action: str, quantity: int):
        match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', option_id)
        if match:
            underlying = match.group(1)
            date_str = match.group(2)
            cp = "CALL" if match.group(3) == "C" else "PUT"
            strike = str(float(match.group(4)) / 1000.0)
            formatted_date = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}"
            
            return {
                "side": action.upper(),
                "quantity": str(quantity),
                "symbol": underlying,
                "strike_price": strike,
                "option_expire_date": formatted_date,
                "instrument_type": "OPTION",
                "option_type": cp,
                "market": "US"
            }
        return None

    def _route_v2_order(self, payload: dict):
        intercepted_request = [None]
        original_get_response = getattr(self.api_client, 'get_response', None)
        
        if original_get_response:
            def mock_get_response(req):
                intercepted_request[0] = req
                raise Exception("Intercepted")
                
            self.api_client.get_response = mock_get_response
            try:
                self.trade_client.account.get_account_profile(self.account_id)
            except Exception:
                pass
            finally:
                self.api_client.get_response = original_get_response
                
            req = intercepted_request[0]
            if req:
                req._action_name = "/trading/orders/place"
                req._method = "POST"
                req._version = "v2"
                req._body_params = payload
                req._params = {} 
                
                response = self.api_client.get_response(req)
                result = response.json() if hasattr(response, 'json') else response
                return result
        return None

    def place_option_order(self, ticker: str, option_id: str, action: str, quantity: int = 1, is_closing: bool = False, order_type: str = "LIMIT", stop_price: float = None, limit_price: float = None):
        """Place a single-leg option order via the SDK's canonical order_v3 path.

        Returns {"accepted": bool, "client_order_id": str, "order_id": str|None, "raw": ...}.
        The caller confirms the actual fill with get_order_status(client_order_id).
        """
        # The caller decides the leg side via `action`; `is_closing` only picks the
        # position_intent. Supports all four: BUY_TO_OPEN (long entry),
        # SELL_TO_CLOSE (long exit), SELL_TO_OPEN (L3 premium selling),
        # BUY_TO_CLOSE (covering a short).
        order_action = str(action).upper()
        if order_action not in ("BUY", "SELL"):
            order_action = "SELL" if is_closing else "BUY"
        position_intent = f"{order_action}_TO_{'CLOSE' if is_closing else 'OPEN'}"
        order_type = order_type.upper()

        print(f"⚡ [WEBULL] {order_type} {order_action} {quantity}x {option_id} ({position_intent})...")

        if limit_price is None and order_type in ("LIMIT", "STOP_LOSS_LIMIT"):
            bid, ask = self.get_live_option_quote(option_id)
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                limit_price = min(round(mid + 0.02, 2), ask) if order_action == "BUY" else max(round(mid - 0.02, 2), bid)
            elif order_action == "SELL" and bid > 0:
                limit_price = bid
            elif order_action == "BUY" and ask > 0:
                limit_price = ask
            else:
                limit_price = 0.01
            limit_price = max(0.01, limit_price)

        leg = self._parse_occ_to_leg(option_id, order_action, quantity)
        if not leg:
            print(f"🚨 Invalid OCC format: {option_id}")
            return {"accepted": False, "client_order_id": None, "order_id": None, "raw": None}

        coid = uuid.uuid4().hex
        order = {
            "client_order_id": coid,
            "combo_type": "NORMAL",
            "order_type": order_type,
            "quantity": str(quantity),
            "option_strategy": "SINGLE",
            "side": order_action,
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            "position_intent": position_intent,
            "legs": [leg],
        }
        if order_type in ("LIMIT", "STOP_LOSS_LIMIT"):
            order["limit_price"] = str(round(limit_price, 2))
        if order_type in ("STOP_LOSS", "STOP_LOSS_LIMIT"):
            order["stop_price"] = str(round(max(0.01, stop_price or 0.01), 2))

        data = None
        err = None
        try:
            resp = self.trade_client.order_v3.place_order(self.account_id, new_orders=[order])
            data = resp.json() if hasattr(resp, "json") else resp
        except Exception as e:
            print(f"  ⚠️ order_v3.place_order failed ({e}); falling back to legacy route...")
            err = str(e)
            try:
                data = self._route_v2_order({"account_id": self.account_id, "new_orders": [order]})
                err = None
            except Exception as e2:
                print(f"  🚨 legacy route also failed: {e2}")
                err = str(e2)

        order_id, accepted = self._parse_place_response(data)
        print(f"  └─ accepted={accepted} coid={coid} order_id={order_id}")
        # 🚨 A coid IS RETURNED EVEN WHEN THE ORDER WAS REJECTED -- it is minted
        # locally above. Callers must check `accepted`; `error` carries the
        # broker's reason. Treating "has a coid" as "sent" left a manual IWM
        # close tracking an order that never existed, 2026-09-28 13:20.
        return {"accepted": accepted, "client_order_id": coid, "order_id": order_id,
                "raw": data, "error": err}

    @staticmethod
    def _parse_place_response(data):
        """(order_id, accepted) from a place-order response of unknown exact shape."""
        if not data:
            return None, False
        body = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(body, list) and body:
            body = body[0]
        if not isinstance(body, dict):
            return None, False
        oid = body.get("order_id") or body.get("orderId")
        has_err = bool(body.get("error_code") or body.get("errorCode")
                       or (body.get("code") not in (None, 0, "0", 200, "200", "OK")))
        if oid or body.get("client_order_id"):
            return oid, True
        return None, (not has_err)

    # =====================================================================
    # ORDER / POSITION RECONCILIATION (order_v3 + account_v2)
    # =====================================================================
    @staticmethod
    def _unwrap(d, *keys):
        """Pull a list payload out of the several envelope shapes Webull returns."""
        if isinstance(d, dict):
            for k in ("data",) + keys:
                v = d.get(k)
                if isinstance(v, list):
                    return v
                if isinstance(v, dict):
                    d = v
            return [d]
        return d if isinstance(d, list) else []

    def get_order_status(self, client_order_id: str):
        """{"status", "filled_qty", "fill_price", "raw"}.

        status is SUBMITTED / PARTIAL_FILLED / FILLED / CANCELLED / FAILED / UNKNOWN,
        or None when the status check itself failed (network/parse).
        """
        try:
            resp = self.trade_client.order_v3.get_order_detail(self.account_id, client_order_id)
            d = resp.json() if hasattr(resp, "json") else resp
        except Exception as e:
            print(f"  ⚠️ get_order_detail({client_order_id}) failed: {e}")
            return {"status": None, "filled_qty": 0.0, "fill_price": 0.0, "raw": None}

        node = d
        if isinstance(d, dict) and "data" in d and isinstance(d["data"], (dict, list)):
            node = d["data"]
        if isinstance(node, list) and node:
            node = node[0]
        if isinstance(node, dict) and isinstance(node.get("orders"), list) and node["orders"]:
            node = node["orders"][0]
        if not isinstance(node, dict):
            return {"status": "UNKNOWN", "filled_qty": 0.0, "fill_price": 0.0, "raw": d}

        status = str(node.get("status") or node.get("order_status") or "UNKNOWN").upper().replace(" ", "_")
        filled = _f(node.get("filled_quantity") or node.get("filledQuantity"))
        price = _f(node.get("filled_price") or node.get("avg_filled_price")
                   or node.get("avgFilledPrice") or node.get("avg_fill_price"))
        legs = node.get("legs") if isinstance(node.get("legs"), list) else []
        if filled == 0 and legs:
            filled = _f(legs[0].get("filled_quantity"))
        if price == 0 and legs:
            price = _f(legs[0].get("filled_price") or legs[0].get("avg_filled_price"))
        return {"status": status, "filled_qty": filled, "fill_price": price, "raw": d}

    def cancel_option_order(self, client_order_id: str) -> bool:
        try:
            resp = self.trade_client.order_v3.cancel_order(self.account_id, client_order_id)
            d = resp.json() if hasattr(resp, "json") else resp
            print(f"  ↩️  cancel {client_order_id}: {d}")
            return True
        except Exception as e:
            print(f"  ⚠️ cancel {client_order_id} failed: {e}")
            return False

    def get_open_orders(self):
        """List pending orders -> [{client_order_id, order_id, symbol, instrument_type, status}]."""
        out = []
        try:
            resp = self.trade_client.order_v3.get_order_open(self.account_id, page_size=50)
            d = resp.json() if hasattr(resp, "json") else resp
        except Exception as e:
            print(f"  ⚠️ get_order_open failed: {e}")
            return out
        for o in self._unwrap(d, "orders", "items"):
            if not isinstance(o, dict):
                continue
            node = o
            if isinstance(o.get("orders"), list) and o["orders"]:
                node = o["orders"][0]
            legs = node.get("legs") if isinstance(node.get("legs"), list) else []
            out.append({
                "client_order_id": o.get("client_order_id") or node.get("client_order_id"),
                "order_id": node.get("order_id"),
                "symbol": node.get("symbol") or (legs[0].get("symbol") if legs else None),
                "instrument_type": str(node.get("instrument_type") or "").upper(),
                "status": str(node.get("status") or "").upper(),
                "raw": o,
            })
        return out

    @staticmethod
    def _occ_from_leg(underlying, leg):
        """Rebuild an OCC symbol from Webull's per-leg option fields.

        {option_type: "CALL", option_expire_date: "2026-09-23",
         option_exercise_price: "284"} + "IWM" -> "IWM260923C00284000"

        🚨 THE STRIKE FIELD IS NAMED DIFFERENTLY IN DIFFERENT PAYLOADS.
        Position legs call it `option_exercise_price`; ORDER legs call it
        `strike_price` (observed 2026-09-23 on a live bracket). Both are
        accepted so this works on either, which is what lets resting orders be
        matched to the contract they would sell.

        Returns None rather than guessing if any piece is missing -- a wrong
        OCC would be handed straight to place_option_order.
        """
        try:
            und = "".join(c for c in str(underlying).upper() if c.isalpha())
            exp = str(leg.get("option_expire_date") or "")
            typ = str(leg.get("option_type") or "").upper()
            raw_k = leg.get("option_exercise_price")
            if raw_k in (None, ""):
                raw_k = leg.get("strike_price")
            k = float(raw_k)
            if not und or typ not in ("CALL", "PUT"):
                return None
            y, m, d = exp.split("-")
            if len(y) != 4 or len(m) != 2 or len(d) != 2 or k <= 0:
                return None
            return f"{und}{y[2:]}{m}{d}{typ[0]}{int(round(k * 1000)):08d}"
        except (AttributeError, TypeError, ValueError):
            return None

    def get_open_option_positions(self):
        """List option holdings -> [{occ, underlying, quantity, cost_price, instrument_id}]."""
        out = []
        try:
            resp = self.trade_client.account_v2.get_account_position(self.account_id)
            d = resp.json() if hasattr(resp, "json") else resp
        except Exception as e:
            print(f"  ⚠️ get_account_position failed: {e}")
            return out
        for p in self._unwrap(d, "positions", "holdings", "items", "result"):
            if not isinstance(p, dict):
                continue
            itype = str(p.get("instrument_type") or p.get("assetType") or "").upper()
            legs = p.get("legs") if isinstance(p.get("legs"), list) else []
            is_opt = ("OPTION" in itype) or any(
                str(l.get("instrument_type", "OPTION")).upper() == "OPTION" for l in legs)
            if not is_opt:
                continue
            qty = _f(p.get("quantity") or p.get("qty"))
            if qty == 0 and legs:
                qty = _f(legs[0].get("quantity"))
            if qty == 0:
                continue
            occ = p.get("symbol") or (legs[0].get("symbol") if legs else None)
            underlying = None
            if occ:
                m = re.match(r'^([A-Za-z]+)\d{6}[CP]\d{8}$', str(occ))
                underlying = m.group(1).upper() if m else str(occ).upper()
            # 🚨 WEBULL DOES NOT RETURN AN OCC SYMBOL HERE.
            # Both `symbol` and `legs[0].symbol` are the UNDERLYING ("IWM"),
            # so `occ` came back as "IWM" for a real IWM 284C position and
            # every consumer was handed a string that cannot identify a
            # contract. _reconcile_startup then asked for a quote on "IWM",
            # got none, and printed "no quote -- CANCEL MANUALLY" instead of
            # doing its job; nothing could adopt or price a held position
            # either. The contract IS in the leg, just spread across fields:
            #   option_type CALL/PUT · option_expire_date · option_exercise_price
            # Rebuild the OCC from those. Observed 2026-09-23 against a live
            # IWM 2x 284C 0DTE holding.
            if legs and (not occ or not re.match(
                    r'^[A-Za-z]+\d{6}[CP]\d{8}$', str(occ))):
                built = self._occ_from_leg(str(underlying or occ or ""), legs[0])
                if built:
                    occ, underlying = built, (underlying or "").upper() or None
            out.append({
                "occ": occ,
                "underlying": underlying,
                "quantity": qty,
                "cost_price": _f(p.get("cost_price") or p.get("cost_basis") or p.get("avg_cost")),
                "instrument_id": p.get("instrument_id") or p.get("ticker_id"),
                "raw": p,
            })
        return out