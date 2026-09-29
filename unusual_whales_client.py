import os
import json
import time
import threading
import websocket
import requests
from datetime import datetime, timedelta, timezone
from collections import deque
from dotenv import load_dotenv


_NY_TZ = None


def _now_et():
    global _NY_TZ
    if _NY_TZ is None:
        from zoneinfo import ZoneInfo
        _NY_TZ = ZoneInfo("America/New_York")
    return datetime.now(_NY_TZ)


def today_et() -> str:
    return _now_et().date().isoformat()


# =====================================================================
# 🚨 MARKET-HOURS GATE FOR EVERY UW POLLER -- THE BASIC PLAN'S 40,000/DAY
# =====================================================================
# Measured 2026-09-27 on a Sunday: ~84 requests/min with the market CLOSED.
# Nothing polled against a clock -- the main loop's net-prem-ticks every 15s
# per ticker and live_state's VWAP, GEX and sweep threads all ran 24/7. That is
# ~121k/day against a 40k limit, i.e. the quota was gone by early evening and
# the NEXT session would open with it spent. Every poller now asks this first.
# The bot itself stays up (EOD flatten, reconcile, Telegram, viewer): only the
# UW calls pause.
UW_OPEN_MOD = 9 * 60 + 25       # 09:25 ET: seed + warm-up before the bell
UW_CLOSE_MOD = 16 * 60 + 5      # 16:05 ET: the 0DTE tape prints to ~16:15, but
                                # nothing after 16:00 feeds a decision
_HOLIDAYS = {}


def _is_holiday(d) -> bool:
    """NYSE full-day closures, from market_calendar. If that import
    fails the gate degrades to weekdays-only -- loudly, once -- which costs one
    wasted session of requests a few times a year, not a missed trading day."""
    if d.year not in _HOLIDAYS:
        try:
            from market_calendar import market_holidays
            _HOLIDAYS[d.year] = set(market_holidays(d.year))
        except Exception as e:
            print(f"⚠️ [UW] holiday calendar unavailable ({e}); gating on weekdays only")
            _HOLIDAYS[d.year] = set()
    return d in _HOLIDAYS[d.year]


def uw_market_open(now=None) -> bool:
    """True while UW polling is worth a request: a trading weekday between
    09:25 and 16:05 ET."""
    now = now or _now_et()
    if now.weekday() >= 5 or _is_holiday(now.date()):
        return False
    mod = now.hour * 60 + now.minute
    return UW_OPEN_MOD <= mod < UW_CLOSE_MOD


def _tick_is_today(tick: dict, today: str) -> bool:
    """🚨 net-prem-ticks WITHOUT a date returns the most recent COMPLETED
    session -- so before today's first print (at the midnight rollover, or the
    old 09:28 cron start) it returns YESTERDAY'S full day. The seed summed it
    and every live day started from yesterday's total: measured 2026-09-27 on
    54 ticker-days, the first live crossover each morning equalled |previous
    day's total| (median ratio 1.001). Asking for date=<today> before the open
    is no fix -- UW answers 422. So filter on each tick's own `date`."""
    d = tick.get("date")
    if d:
        return str(d)[:10] == today
    tt = tick.get("tape_time") or ""
    if not tt:
        return False
    try:
        from zoneinfo import ZoneInfo
        ts = datetime.fromisoformat(str(tt).replace("Z", "+00:00"))
        return ts.astimezone(ZoneInfo("America/New_York")).date().isoformat() == today
    except ValueError:
        return False


def _net_gex_from_row(row: dict):
    """Pull a signed net-GEX number out of a UW row, tolerating the several
    field-naming schemes UW uses across endpoints/versions."""
    if not isinstance(row, dict):
        return None
    try:
        if row.get("net_gex") not in (None, ""):
            return float(row["net_gex"])
        for a, b in (("call_gex", "put_gex"), ("call_gamma", "put_gamma")):
            if a in row or b in row:
                return float(row.get(a, 0) or 0) + float(row.get(b, 0) or 0)
        if row.get("gamma_per_one_percent_move_oi") not in (None, ""):
            return float(row["gamma_per_one_percent_move_oi"])
    except (TypeError, ValueError):
        return None
    return None

class UnusualWhalesClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.unusualwhales.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        
        # ==========================================
        # ZERO-LATENCY MEMORY STATE (The Multiplexer)
        # ==========================================
        self.live_flow = {}            # { ticker: deque([net_premium, ...]) }
        self.live_gex = {}             # { ticker: { gex, dex, charm, vanna } }
        self.dark_pool_walls = {}      # { ticker: { price_level: total_volume } }
        self.iv_term_structure = {}    # { ticker: { days: volatility } }
        self.whale_sweeps = {}         # { ticker: deque([ {premium, strike, expiry, tags} ]) }
        
        self.ws = None
        self.is_connected = False

        # ==========================================
        # WEBSOCKET vs REST -- WHICH SOURCE FEEDS THE FLOW TRIGGER
        # ==========================================
        # 🚨 THE WEBSOCKET IS OPTIONAL, AND ITS ABSENCE IS NOT AN ERROR.
        # UW's Basic plan has no websocket. The trigger's input already had a
        # REST fallback (net-prem-ticks), so the bot runs fine without it --
        # arguably BETTER, since every backtest behind the deployed rules was
        # built from REST net-prem-ticks history. What was wrong is that a
        # refused socket retried every 2s all session, forever.
        #   UW_WEBSOCKET=false   never open it
        #   a 401/403 handshake  stop retrying after one clear message
        self.ws_enabled = os.getenv("UW_WEBSOCKET", "true").strip().lower() not in (
            "false", "0", "no", "off")
        self.ws_refused = False
        self._ws_last_msg = 0.0          # wall time of the last message of ANY kind

        # 🚨 A DEAD SOCKET MUST NOT FREEZE THE FLOW. get_live_net_premium used to
        # fall back to REST only when the socket buffer was EMPTY -- but after a
        # drop the buffer keeps its last 20 ticks forever, so the bot kept
        # re-reading ticks it had already counted and every tick printed while
        # the socket was down was lost for the rest of the day, silently. Now:
        # a socket quiet for WS_STALE_S hands that ticker to REST for the rest
        # of the session, and bumps flow_epoch so bot_runner rebuilds the day's
        # total from REST instead of carrying the gap.
        self.WS_STALE_S = 30.0
        self._flow_source = {}           # ticker -> "ws" | "rest"
        self._rest_sticky = {}           # ticker -> True once handed to REST
        self.flow_epoch = {}             # ticker -> int, bumped on every hand-over
        self.seed_minutes = {}           # ticker -> {tape_time: net_premium} from the seed

        # --- Intraday GEX recorder (Stage 1: gather data to decide later whether
        #     the strategy needs intraday regime awareness). Throttled writes. ---
        self.gex_log_path = os.getenv("GEX_LOG_PATH", "gex_history_log.jsonl")
        self._gex_log_state = {}  # ticker -> (last_write_epoch, last_regime)

    def start_multiplexer(self, tickers: list):
        """
        Connects to the Advanced Plan WebSocket and subscribes to all 5
        institutional data channels simultaneously.

        Returns without opening anything when UW_WEBSOCKET=false. The flow
        trigger then runs on REST net-prem-ticks, which is the same source the
        backtests were built on.
        """
        if not self.ws_enabled:
            print("🌊 [UW] WebSocket DISABLED (UW_WEBSOCKET=false) -- flow runs on "
                  "REST net-prem-ticks, polled every 15s per ticker.")
            return
        def safe_float(val):
            if val is None or val == "": return 0.0
            try: return float(val)
            except (ValueError, TypeError): return 0.0

        def on_message(ws, message):
            self._ws_last_msg = time.time()
            try:
                payload = json.loads(message)
                
                # UW Payloads are arrays: ["channel_name", {data_object}]
                if isinstance(payload, list) and len(payload) >= 2:
                    channel = str(payload[0])
                    data = payload[1]
                    ticker = data.get("ticker") or data.get("symbol")
                    
                    # ---------------------------------------------------------
                    # 1. NET FLOW (Zero-Latency Momentum)
                    # ---------------------------------------------------------
                    if channel.startswith("net_flow:"):
                        net_prem = safe_float(data.get("net_call_prem")) - safe_float(data.get("net_put_prem"))
                        if ticker not in self.live_flow:
                            self.live_flow[ticker] = deque(maxlen=20) # Keep last 20 ticks
                        self.live_flow[ticker].append({"net_premium": net_prem, "time": data.get("time")})
                        
                    # ---------------------------------------------------------
                    # 2. GEX & 2nd ORDER GREEKS (The Map & Timer)
                    # ---------------------------------------------------------
                    elif channel.startswith("gex:"):
                        raw_gamma = data.get("gamma_per_one_percent_move_oi")
                        # Distinguish "field absent" (schema drift -> UNKNOWN) from
                        # "field present and non-positive" (legitimate NEGATIVE gex).
                        if raw_gamma is None:
                            regime = "UNKNOWN"
                        else:
                            regime = "POSITIVE" if safe_float(raw_gamma) > 0 else "NEGATIVE"
                        self.live_gex[ticker] = {
                            "net_gex": safe_float(raw_gamma),
                            "net_dex": safe_float(data.get("delta_per_one_percent_move_oi")),
                            "charm": safe_float(data.get("charm_per_one_percent_move_oi")),
                            "vanna": safe_float(data.get("vanna_per_one_percent_move_oi")),
                            "regime": regime,
                            "last_updated": data.get("timestamp"),
                            "_local_ts": time.time()
                        }
                        self._record_gex(ticker, data, regime)


                    # ---------------------------------------------------------
                    # 3. OFF-LIT TRADES (Dark Pool Armor)
                    # ---------------------------------------------------------
                    elif channel == "off_lit_trades":
                        # We only care about MASSIVE prints to form support/resistance walls
                        vol = safe_float(data.get("volume"))
                        price = safe_float(data.get("price"))
                        notional_value = vol * price
                        
                        if notional_value >= 1_000_000: # $1M+ Dark Pool Print
                            if ticker not in self.dark_pool_walls:
                                self.dark_pool_walls[ticker] = {}
                            
                            # Aggregate volume at this specific price level
                            rounded_price = round(price, 2)
                            current_vol = self.dark_pool_walls[ticker].get(rounded_price, 0)
                            self.dark_pool_walls[ticker][rounded_price] = current_vol + vol

                    # ---------------------------------------------------------
                    # 4. IV TERM STRUCTURE (Volatility Arbitrage)
                    # ---------------------------------------------------------
                    elif channel.startswith("iv_term_structure:"):
                        days = data.get("days")
                        volatility = safe_float(data.get("volatility"))
                        
                        if ticker not in self.iv_term_structure:
                            self.iv_term_structure[ticker] = {}
                        self.iv_term_structure[ticker][days] = volatility

                    # ---------------------------------------------------------
                    # 5. LIVE OPTIONS TAPE (The Whale Sweep Detector)
                    # ---------------------------------------------------------
                    elif channel.startswith("option_trades:"):
                        tags = data.get("tags", [])
                        premium = safe_float(data.get("premium"))
                        
                        # Filter for Urgency: >$50k Premium, Hitting the Ask, and marked as a Sweep
                        if premium >= 50000 and "ask_side" in tags and "sweep" in tags:
                            if ticker not in self.whale_sweeps:
                                self.whale_sweeps[ticker] = deque(maxlen=10)
                                
                            self.whale_sweeps[ticker].append({
                                "strike": data.get("strike"),
                                "type": data.get("option_type"),
                                "expiry": data.get("expiry"),
                                "premium": premium,
                                "iv": data.get("implied_volatility"),
                                "timestamp": datetime.now().isoformat()
                            })
                            
            except Exception as e:
                pass # Silently drop malformed packets to maintain firehose speed

        def on_open(ws):
            print("🌊 [UW MULTIPLEXER] Connected to Advanced WebSocket!")
            self.is_connected = True
            
            # 1. Global Channels
            ws.send(json.dumps({"channel": "off_lit_trades", "msg_type": "join"}))
            
            # 2. Ticker-Specific Channels
            for ticker in tickers:
                channels = [
                    f"net_flow:{ticker}",
                    f"gex:{ticker}",
                    f"iv_term_structure:{ticker}",
                    f"option_trades:{ticker}"
                ]
                
                for ch in channels:
                    ws.send(json.dumps({"channel": ch, "msg_type": "join"}))
                    
            print(f"📡 [UW MULTIPLEXER] Now streaming 5 Dimensions of data for {len(tickers)} assets.")

        def on_error(ws, e):
            code = getattr(e, "status_code", None)
            text = str(e)
            if code in (401, 403) or " 401" in text or " 403" in text:
                # The plan does not include the socket (or the key is bad).
                # Retrying cannot fix either, so say it once and stop.
                self.ws_refused = True
            print(f"⚠️ [UW WS Error] {text[:160]}")

        def on_close(ws, c, m):
            self.is_connected = False
            print("🔴 [UW WS] Disconnected.")

        def run_ws():
            url = f"wss://api.unusualwhales.com/socket?token={self.api_key}"
            self.ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            # Reconnect with backoff 2s -> 60s. A connection that closes before
            # delivering a single message counts as a quick failure; three in a
            # row is a socket this plan will not serve, which some servers
            # signal by closing rather than by a 401/403.
            delay, quick_fails = 2.0, 0
            while not self.ws_refused:
                before = self._ws_last_msg
                self.ws.run_forever()
                if self.ws_refused:
                    break
                quick_fails = quick_fails + 1 if self._ws_last_msg == before else 0
                if quick_fails >= 3:
                    self.ws_refused = True
                    break
                time.sleep(delay)
                delay = min(delay * 2, 60.0) if quick_fails else 2.0
            print("🌊 [UW] WebSocket unavailable on this plan/key -- flow runs on "
                  "REST net-prem-ticks for the rest of the session. Set "
                  "UW_WEBSOCKET=false to skip the attempt entirely.")

        t = threading.Thread(target=run_ws, daemon=True)
        t.start()

    # ==========================================
    # DATA RETRIEVAL METHODS (Called by Bot Runner)
    # ==========================================
    
    def seed_daily_cumulative_flow(self, ticker: str) -> float:
        """
        REST Fallback: Fetches the entire day's options flow history and sums it up 
        to perfectly seed the bot's cumulative total if started mid-day.
        """
        url = f"{self.base_url}/api/stock/{ticker.upper()}/net-prem-ticks"
        # cleared FIRST, so a failed fetch leaves no minutes behind rather than
        # yesterday's -- which the ledger would then treat as already counted
        self.seed_minutes[ticker] = {}
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    return 0.0

                # net-prem-ticks returns PER-MINUTE INCREMENTS (verified 2026-08-28
                # against live SPY data: one-minute values oscillate around zero and
                # spike/revert, so the day's cumulative net premium == sum of ticks).
                # 🚨 RECORD WHICH MINUTES THE SEED COUNTED. It used to return only
                # the total, so on REST the first poll -- which returns the SAME
                # full day -- added every one of those minutes a second time.
                # Any mid-day restart (systemd restarts on a crash) doubled the
                # day's flow so far. bot_runner copies these into its
                # per-minute ledger so the first poll adds only what is new.
                total_cumulative = 0.0
                minutes = {}
                today = today_et()
                dropped = sum(1 for t in data if not _tick_is_today(t, today))
                data = [t for t in data if _tick_is_today(t, today)]
                if dropped:
                    print(f"  ├─ {ticker}: ignored {dropped} tick(s) from a previous "
                          f"session (UW returns the last completed day before the open)")
                for tick in data:
                    net_call = float(tick.get("net_call_premium", 0))
                    net_put = float(tick.get("net_put_premium", 0))
                    total_cumulative += (net_call - net_put)
                    if tick.get("tape_time"):
                        minutes[tick["tape_time"]] = net_call - net_put
                self.seed_minutes[ticker] = minutes

                # Soft sanity check: a single name's full-day net premium rarely
                # clears ~$500M. If it does, the feed schema may have changed to
                # running totals -- warn, but trust the (verified) summing model.
                if abs(total_cumulative) > 500_000_000:
                    print(f"  ⚠️ [SEED] {ticker}: summed day flow ${total_cumulative:,.0f} is unusually large. "
                          f"If bot fires immediately on every ticker, re-check net-prem-ticks schema "
                          f"(python diagnose_flow_schema.py {ticker} --raw).")

                return total_cumulative
        except Exception as e:
            pass
        return 0.0

    def ws_alive(self) -> bool:
        """Socket open and heard from within WS_STALE_S. Any channel counts:
        off_lit_trades and option_trades are busy enough that 30s of silence on
        the whole socket means it is gone, not that the market is quiet."""
        return (self.ws_enabled and not self.ws_refused and self.is_connected
                and time.time() - self._ws_last_msg < self.WS_STALE_S)

    def flow_source(self, ticker: str) -> str:
        """'ws' or 'rest' -- which source served the LAST get_live_net_premium
        call for this ticker. bot_runner counts the two differently."""
        return self._flow_source.get(ticker, "rest")

    def reset_session(self):
        """New trading day: a ticker handed to REST yesterday may use the socket
        again today. flow_epoch is NOT reset -- it only ever increases, so
        bot_runner can never mistake an old hand-over for a new one."""
        self._rest_sticky.clear()
        self._flow_source.clear()
        if hasattr(self, "_flow_rest_cache"):
            self._flow_rest_cache.clear()

    def get_live_net_premium(self, ticker: str):
        """Flow ticks for one ticker: the socket buffer while it is live, REST
        net-prem-ticks (the whole day so far) otherwise.

        Once a ticker the socket WAS serving goes quiet, it is handed to REST
        for the rest of the session and flow_epoch[ticker] is bumped. It does
        not flip back mid-session: every flip between sources is a chance to
        count a minute twice or not at all, and one clean hand-over plus a
        rebuild from REST is exact.
        """
        if self.ws_enabled and not self._rest_sticky.get(ticker):
            buf = self.live_flow.get(ticker)
            if self.ws_alive() and buf:
                self._flow_source[ticker] = "ws"
                return list(buf)
            if self._flow_source.get(ticker) == "ws":
                self._rest_sticky[ticker] = True
                self.flow_epoch[ticker] = self.flow_epoch.get(ticker, 0) + 1
                print(f"🔁 [UW] {ticker}: WebSocket quiet >{self.WS_STALE_S:.0f}s -- "
                      f"handing flow to REST for the rest of the session and "
                      f"rebuilding today's total from it.")
        self._flow_source[ticker] = "rest"

        # Market closed: no request. The main loop runs every 0.1s around the
        # clock, and this line is what stops it spending the day's quota on a
        # Saturday. [] means "nothing new" to the caller, which is the truth.
        if not uw_market_open():
            return []

        # REST Fallback -- rate limited so a dead WS + 0.1s main loop doesn't
        # fire ~70 requests/sec across the watchlist.
        if not hasattr(self, '_flow_rest_cache'):
            self._flow_rest_cache = {}
        now = time.time()
        cached = self._flow_rest_cache.get(ticker)
        if cached and (now - cached['time']) < 15:
            return cached['data']

        url = f"{self.base_url}/api/stock/{ticker.upper()}/net-prem-ticks"
        formatted_ticks = []
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                raw_data = response.json().get("data", [])

                # Standardize the REST response to perfectly match our WebSocket format.
                # Today's ticks ONLY -- see _tick_is_today.
                today = today_et()
                for tick in raw_data:
                    if not _tick_is_today(tick, today):
                        continue
                    net_call = float(tick.get("net_call_premium", 0))
                    net_put = float(tick.get("net_put_premium", 0))
                    formatted_ticks.append({
                        "net_premium": net_call - net_put,
                        "time": tick.get("tape_time", "")
                    })
        except Exception:
            pass

        self._flow_rest_cache[ticker] = {'time': now, 'data': formatted_ticks}
        return formatted_ticks

    def get_gex_regime(self, ticker: str):
        """Returns 0-latency GEX/DEX/Charm/Vanna, falls back to REST with strict caching."""
        # 1. 0-Latency WebSocket Memory -- but only while it's actually fresh.
        # Without this check, one REST write to self.live_gex at boot freezes
        # the regime for the whole session because this branch always wins.
        if ticker in self.live_gex:
            cached_ws = self.live_gex[ticker]
            if (time.time() - cached_ws.get("_local_ts", 0)) < 300:
                return cached_ws

        # Initialize strict REST cache if it doesn't exist
        if not hasattr(self, '_gex_rest_cache'):
            self._gex_rest_cache = {}
            
        # 2. Check if we already fetched REST recently (5-minute cooldown)
        current_time = time.time()
        if ticker in self._gex_rest_cache:
            cached_data = self._gex_rest_cache[ticker]
            if (current_time - cached_data['time']) < 300: # 300 seconds = 5 minutes
                return cached_data['data']
            
        # 3. REST Fallback (Only fires if WS is empty AND 5 mins have passed)
        url = f"{self.base_url}/api/stock/{ticker.upper()}/greek-exposure"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                history = response.json().get("data", [])
                # /greek-exposure is oldest-first live (2026-08); sort defensively
                # so a future order flip can't hand us a 1-year-old row.
                history = sorted(history, key=lambda r: r.get("date") or "")
                if history:
                    latest = history[-1]
                    net_gex = _net_gex_from_row(latest)
                    net_dex = float(latest.get("call_delta", 0) or 0) + float(latest.get("put_delta", 0) or 0)

                    if net_gex is None:
                        regime = "UNKNOWN"
                        net_gex = 0.0
                    else:
                        regime = "POSITIVE" if net_gex > 0 else ("NEGATIVE" if net_gex < 0 else "UNKNOWN")

                    regime_payload = {
                        "net_gex": net_gex,
                        "net_dex": net_dex,
                        "regime": regime,
                        "charm": 0.0,
                        "vanna": 0.0,
                        "_local_ts": current_time
                    }
                    
                    # Save to the cooldown cache
                    self._gex_rest_cache[ticker] = {'time': current_time, 'data': regime_payload}
                    
                    # Also seed the live memory so we never hit this again unless the WS dies
                    self.live_gex[ticker] = regime_payload
                    
                    return regime_payload
        except Exception: 
            pass
            
        # If everything fails, cache a None response for 1 minute to prevent spamming broken tickers
        self._gex_rest_cache[ticker] = {'time': current_time - 240, 'data': None}
        return None

    def _record_gex(self, ticker: str, data: dict, regime: str):
        """Append a throttled snapshot of the live gex: stream to a JSONL file.

        Purpose: accumulate real intraday GEX history so we can later measure how
        often the regime actually flips during RTH (and decide whether the
        strategy needs an intraday-GEX backtest, or a morning poll is enough).
        Writes at most once / 15s per ticker, plus immediately on a regime flip.
        """
        try:
            now = time.time()
            last_ts, last_regime = self._gex_log_state.get(ticker, (0.0, None))
            if regime == last_regime and (now - last_ts) < 15:
                return
            self._gex_log_state[ticker] = (now, regime)
            record = {
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "regime": regime,
                "net_gex": _net_gex_from_row(data),
                "gamma_1pct_oi": data.get("gamma_per_one_percent_move_oi"),
                "delta_1pct_oi": data.get("delta_per_one_percent_move_oi"),
                "uw_timestamp": data.get("timestamp") or data.get("time"),
                "price": (data.get("price") or data.get("spot")
                          or data.get("underlying_price") or data.get("close")),
            }
            with open(self.gex_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass  # never let logging break the WS handler

    def get_daily_gex_regime(self, ticker: str, prior_day: bool = False):
        """One-shot DAILY GEX from /greek-exposure REST -- for the bot's morning
        poll (fetched once at boot / on the daily rollover and held all session).

        prior_day=True returns the previous completed session's value, which is
        lookahead-free and matches a shifted backtest.
        """
        url = f"{self.base_url}/api/stock/{ticker.upper()}/greek-exposure"
        try:
            r = requests.get(url, headers=self.headers, timeout=8)
            if r.status_code != 200:
                print(f"  ⚠️ /greek-exposure {ticker}: HTTP {r.status_code}")
                return None
            hist = r.json().get("data", [])
            if not hist:
                return None
            # oldest-first live; sort defensively (the API docs example is newest-first)
            hist = sorted(hist, key=lambda x: x.get("date") or "")
            idx = -2 if (prior_day and len(hist) >= 2) else -1
            row = hist[idx]
            net_gex = _net_gex_from_row(row)
            if net_gex is None:
                return {"regime": "UNKNOWN", "net_gex": 0.0, "as_of": row.get("date")}
            return {
                "regime": "POSITIVE" if net_gex > 0 else ("NEGATIVE" if net_gex < 0 else "UNKNOWN"),
                "net_gex": net_gex,
                "as_of": row.get("date"),
            }
        except Exception as e:
            print(f"  ⚠️ daily GEX fetch failed for {ticker}: {e}")
            return None

    def get_dex_pct(self, ticker: str, window: int = 252):
        """Trailing-`window`-session percentile of net_dex (call_delta + put_delta)
        for the last COMPLETED session -- the `dex_pct_max` rule gate. Matches the
        backtest's shift(1) + rolling-252 percentile. Returns a float in [0,1] or
        None. Polled once at session start and held (like the daily GEX poll)."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        url = f"{self.base_url}/api/stock/{ticker.upper()}/greek-exposure"
        try:
            r = requests.get(url, headers=self.headers, params={"timeframe": "2Y"}, timeout=10)
            if r.status_code != 200:
                print(f"  ⚠️ /greek-exposure {ticker} (dex): HTTP {r.status_code}")
                return None
            hist = sorted(r.json().get("data", []), key=lambda x: x.get("date") or "")
            today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            series = []
            for row in hist:
                d = str(row.get("date") or "")[:10]
                if not d or d >= today:      # drop any partial 'today' row
                    continue
                try:
                    series.append(float(row["call_delta"]) + float(row["put_delta"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if len(series) < 40:
                print(f"  ⚠️ {ticker}: only {len(series)} net_dex days -- dex_pct unavailable")
                return None
            cur = series[-1]
            ref = series[-(window + 1):-1] if len(series) > window else series[:-1]
            return sum(1 for v in ref if v < cur) / len(ref) if ref else None
        except Exception as e:
            print(f"  ⚠️ dex_pct fetch failed for {ticker}: {e}")
            return None

    def get_spot_gex(self, ticker: str):
        """Live intraday spot GEX -- today's most recent minute from /spot-exposures.
        Returns {"oi": +1/-1, "dir": +1/-1} (sign only) or None if unavailable /
        dGEX not yet established. Used ONLY by the vol-regime target/stop overlay
        (config.RULES "vol_overlay" -- see check_dgex.py --test overlay); fetched
        fresh at each trigger, not session-cached like the daily GEX/trend polls --
        both signs are live-recomputed every minute and drift through the day."""
        url = f"{self.base_url}/api/stock/{ticker.upper()}/spot-exposures"
        try:
            r = requests.get(url, headers=self.headers, timeout=8)
            if r.status_code != 200:
                return None
            rows = r.json().get("data", [])
            if not rows:
                return None
            row = rows[-1]
            oi = float(row.get("gamma_per_one_percent_move_oi") or 0.0)
            dr = float(row.get("gamma_per_one_percent_move_dir") or 0.0)
            if oi == 0.0 or dr == 0.0:
                return None
            return {"oi": 1 if oi > 0 else -1, "dir": 1 if dr > 0 else -1}
        except Exception as e:
            print(f"  ⚠️ spot-GEX fetch failed for {ticker}: {e}")
            return None

    def get_daily_bars(self, ticker: str, limit: int = 90):
        """Recent daily OHLC as [{date, close, volume}, ...] ASCENDING (newest
        last). For bot_runner's trend / volume regime poll. During RTH the last
        row can be an in-progress bar -- callers should treat the last COMPLETED
        row (or drop today) as the prior-day state."""
        url = f"{self.base_url}/api/stock/{ticker.upper()}/ohlc/1d"
        try:
            r = requests.get(url, headers=self.headers, timeout=8)
            if r.status_code != 200:
                print(f"  ⚠️ /ohlc/1d {ticker}: HTTP {r.status_code}")
                return []
            rows = r.json().get("data", [])
            # /ohlc/1d splits each date into pre / regular / post (market_time
            # pr/r/po). Keep the regular-session bar -- that's the RTH close/volume
            # the trend/volume backtest uses.
            by_date = {}
            for x in rows:
                mt = x.get("market_time")
                if mt not in (None, "r"):
                    continue
                try:
                    by_date[str(x.get("date") or x.get("start_time") or "")[:10]] = {
                        "date": str(x.get("date") or x.get("start_time") or "")[:10],
                        "close": float(x.get("close")),
                        "volume": float(x.get("volume") or x.get("total_volume") or 0.0),
                    }
                except (TypeError, ValueError):
                    continue
            out = [by_date[d] for d in sorted(by_date) if d]
            return out[-int(limit):]
        except Exception as e:
            print(f"  ⚠️ daily bars fetch failed for {ticker}: {e}")
            return []

    def get_vix_state(self, window: int = 60):
        """VIX regime for the sizing overlay. Returns
            {"vix_prev": float, "median": float, "favorable": bool}
        where vix_prev = last COMPLETED daily VIX and median = trailing-`window`-
        session median of the sessions BEFORE it (lookahead-free), favorable =
        vix_prev >= median. None if the feed is unavailable -> caller fails open
        to full size. Source: /stock/VIX/volatility/realized `price` (daily 30d
        IV proxy for VIX; /ohlc/1d 422s for VIX)."""
        url = f"{self.base_url}/api/stock/VIX/volatility/realized"
        try:
            r = requests.get(url, headers=self.headers, params={"timeframe": "1Y"}, timeout=8)
            if r.status_code != 200:
                print(f"  ⚠️ VIX realized: HTTP {r.status_code}")
                return None
            rows = sorted(
                ((str(x.get("date") or "")[:10], float(x["price"]))
                 for x in r.json().get("data", []) if x.get("price")),
                key=lambda t: t[0],
            )
            vals = [v for _, v in rows if v > 0]
            if len(vals) < 25:
                return None
            vix_prev = vals[-1]
            ref = vals[-(window + 1):-1] if len(vals) > window else vals[:-1]
            median = float(sorted(ref)[len(ref) // 2]) if ref else vix_prev
            return {"vix_prev": vix_prev, "median": median, "favorable": vix_prev >= median}
        except Exception as e:
            print(f"  ⚠️ VIX state fetch failed: {e}")
            return None

    def get_intraday_bars(self, ticker: str, lookback_days: int = 3, ohlcv: bool = False,
                          end_date=None):
        """Recent 1-minute RTH bars over the last `lookback_days` trading days,
        ascending. Default: [{"minute_et": "YYYY-MM-DDTHH:MM", "close": float}]
        (EMA-stack seed). ohlcv=True also adds "mod"/"o"/"h"/"l"/"v" for the
        Auction-Market-Theory volume profile.

        `end_date` (a date) is the newest day counted; default today. During
        RTH "today" already has bars, so lookback_days=1 means TODAY SO FAR --
        a caller that wants the previous session must say so (see
        bot_runner._seed_amt_profiles)."""
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        out, got, tries = [], 0, 0
        day = end_date or datetime.now(et).date()
        while got < lookback_days and tries < lookback_days + 5:
            tries += 1
            if day.weekday() >= 5:
                day -= timedelta(days=1)
                continue
            url = f"{self.base_url}/api/stock/{ticker.upper()}/ohlc/1m"
            try:
                r = requests.get(url, headers=self.headers, params={"date": day.isoformat()}, timeout=10)
                rows = r.json().get("data", []) if r.status_code == 200 else []
            except Exception:
                rows = []
            day_bars = []
            for x in rows:
                if x.get("market_time") not in (None, "r"):
                    continue
                st = x.get("start_time") or ""
                try:
                    dt_utc = datetime.fromisoformat(st.replace("Z", "+00:00"))
                    dt_et = dt_utc.astimezone(et)
                    mod = dt_et.hour * 60 + dt_et.minute
                    if mod < 570 or mod > 960:          # RTH 09:30-16:00 ET
                        continue
                    b = {"minute_et": dt_et.strftime("%Y-%m-%dT%H:%M"),
                         "close": float(x.get("close"))}
                    if ohlcv:
                        b.update(mod=mod, o=float(x.get("open")), h=float(x.get("high")),
                                 l=float(x.get("low")), c=b["close"],
                                 v=float(x.get("volume") or x.get("total_volume") or 0.0))
                    day_bars.append(b)
                except (TypeError, ValueError):
                    continue
            if day_bars:
                out = day_bars + out
                got += 1
            day -= timedelta(days=1)
        out.sort(key=lambda b: b["minute_et"])
        return out

    def get_dark_pool_support(self, ticker: str, current_price: float, threshold_vol: int = 500000):
        """
        Finds the closest massive dark pool wall below the current price to act as a Stop Loss.
        """
        if ticker not in self.dark_pool_walls:
            return None
            
        walls = self.dark_pool_walls[ticker]
        valid_supports = [price for price, vol in walls.items() if price < current_price and vol >= threshold_vol]
        
        if valid_supports:
            return max(valid_supports) # Return the highest support level below current price
        return None

    def check_for_backwardation(self, ticker: str) -> bool:
        """
        Checks if short-term IV (1-7 days) has spiked higher than 30-day IV.
        A positive signal means an explosive move/earnings is imminent.
        """
        if ticker not in self.iv_term_structure:
            return False
            
        term = self.iv_term_structure[ticker]
        short_term_iv = term.get(7, 0) or term.get(5, 0) or term.get(1, 0)
        thirty_day_iv = term.get(30, 0)
        
        if short_term_iv > 0 and thirty_day_iv > 0:
            return short_term_iv > thirty_day_iv
        return False
        
    def get_market_tide(self):
        url = f"{self.base_url}/api/market/market-tide"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_gex_levels(self, ticker: str):
        url = f"{self.base_url}/api/stock/{ticker.upper()}/gex-levels"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200: return {}
        return response.json().get("data", {})

    def get_volatility_character(self, ticker: str):
        url = f"{self.base_url}/api/stock/{ticker.upper()}/volatility/character"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200: return {}
        return response.json().get("data", {})

    def get_interpolated_iv(self, ticker: str):
        """Fetches the 30-day forward-looking Interpolated IV."""
        url = f"{self.base_url}/api/stock/{ticker.upper()}/interpolated-iv"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200: return {}
        return response.json().get("data", {})

    def get_realized_volatility(self, ticker: str):
        """Fetches historical 30-day Realized Volatility."""
        url = f"{self.base_url}/api/stock/{ticker.upper()}/volatility/realized"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200: return {}
        return response.json().get("data", {})