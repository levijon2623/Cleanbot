import os
import json
import threading
import websocket
import eth_account
import time
import requests
import pandas as pd
from collections import deque
from dotenv import load_dotenv
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

import hl_gate

class HyperExposureClient:
    """
    A unified algorithmic trading client for the Hyperliquid L1 DEX.
    Handles agent authorization, account monitoring, and dynamic order execution.

    🚨 EVERY EXECUTION METHOD BELOW PASSES THROUGH hl_gate.check().
    Read the module docstring of hl_gate.py before changing any of them. The
    short version: this venue is off by default, needs a legal attestation as
    well as a capability flag, and caps notional and leverage. Reads and
    risk-REDUCING orders are never gated -- see hl_gate for why that asymmetry
    is deliberate rather than an oversight.

    Do NOT add a new path to `self.exchange.order()` without a gate call. The
    audit on 2026-09-26 found eleven ungated ones, which is how this file got
    its gate.
    """

    def __init__(self, main_wallet_address, agent_secret=None, perp_dexs=None):
        self.main_wallet = main_wallet_address
        self.perp_dexs = perp_dexs or ["", "xyz"]

        # Thread safety lock for data mutation
        self.data_lock = threading.Lock()
        
        # Initialize Info API (Public Data - WS enabled for live price streams)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=False, perp_dexs=self.perp_dexs)
        self._positions_cache = []
        self._last_positions_fetch = 0
        
        # 🚨 SET UNCONDITIONALLY, NOT INSIDE THE `if agent_secret` BELOW.
        # `get_resting_orders` and `get_open_positions` both read
        # self.main_wallet_address, and it used to be assigned only when an
        # agent secret was supplied -- so a read-only client (the documented way
        # to inspect an account without trade permission) raised AttributeError
        # on a method that needs no signing at all.
        self.main_wallet_address = main_wallet_address

        # Initialize Exchange API (Authenticated Actions - Requires Agent)
        #
        # 🚨 THE AGENT IS STILL LOADED WHEN TRADING IS DISABLED, ON PURPOSE.
        # Refusing to build the Exchange unless armed would look safer and be
        # worse: flattening a live perp needs a signature, and hl_gate lets
        # risk-reducing orders through precisely so a disabled flag can never
        # trap you in a leveraged position. Signing capability is the floor;
        # the gate is what decides what gets signed.
        self.exchange = None
        if agent_secret:
            self.agent_account = eth_account.Account.from_key(agent_secret)
            self.exchange = Exchange(
                wallet=self.agent_account,
                base_url=constants.MAINNET_API_URL,
                account_address=self.main_wallet,
                perp_dexs=self.perp_dexs
            )
            hl_gate.banner()

    @staticmethod
    def generate_agent(master_key):
        """Authorize a new agent wallet on-chain, signing with the MASTER key.

        🚨 THIS IS THE MOST PRIVILEGED CALL IN THE FILE AND IT IS INTERACTIVE
        ONLY. It signs with the master wallet, not an agent, and it mints a new
        private key. It is gated like an entry because it is the step that
        GRANTS trade permission -- locking the orders but not the authorization
        would be a lock on the door of an unwalled room.
        """
        hl_gate.check("generate_agent", "-",
                      notional_usd=hl_gate.max_notional(), has_exchange=True)

        master_wallet = eth_account.Account.from_key(master_key)
        exchange = Exchange(wallet=master_wallet, base_url=constants.MAINNET_API_URL)

        print(f"Authorizing agent for master wallet: {master_wallet.address}")
        result, agent_private_key = exchange.approve_agent("py_api_bridge")

        if result.get("status") != "ok":
            print(f"Failed to authorize agent: {result}")
            return result

        print("\nSuccess! Agent authorized on-chain.")

        # 🚨 THE KEY IS NOT PRINTED TO A NON-TTY. It used to go to stdout
        # unconditionally. Under systemd stdout IS the journal, so running this
        # from a unit or through `ssh host python ...` wrote a live trading key
        # into journalctl in plaintext, readable by anything that can read the
        # journal and retained for as long as the journal is. Nothing else in
        # this repo logs a credential; this was the one place.
        if os.isatty(1):
            print(f"NEW AGENT PRIVATE KEY: {agent_private_key}")
            print("\nIMPORTANT: put this in .env as AGENT_SECRET. It is not "
                  "stored anywhere and cannot be re-read.")
        else:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "conf", "agent_secret.new")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # 0600 before the write, not after -- between an 0644 create and a
            # later chmod there is a window in which any local user can read it.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(agent_private_key + "\n")
            print(f"  stdout is not a terminal, so the key was NOT printed.")
            print(f"  Written 0600 to {path} — move it into .env as "
                  f"AGENT_SECRET and delete it. (conf/ is gitignored.)")
        return result

    def seed_historical_candles(self, hl_symbols: list, interval="3m"):
        """
        Fetches historical candle snapshots via REST API to instantly seed the deque.
        """
        print("🌱 Seeding historical candle history via REST API...")
        start_time_ms = int((time.time() - (2 * 3600)) * 1000)
        
        # Initialize the raw data buffer if it doesn't exist yet
        if not hasattr(self, 'market_data'):
            self.market_data = {}
            
        for symbol in hl_symbols:
            try:
                payload = {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": interval,
                        "startTime": start_time_ms
                    }
                }
                
                response = requests.post("https://api.hyperliquid.xyz/info", json=payload, timeout=10)
                
                if response.status_code == 200:
                    candles = response.json()
                    
                    with self.data_lock:
                        if symbol not in self.market_data:
                            # Use deque for O(1) performance and automatic memory pruning
                            self.market_data[symbol] = deque(maxlen=100)
                            
                        for c in candles:
                            self.market_data[symbol].append({
                                'time': pd.to_datetime(c['t'], unit='ms'),
                                'open': float(c['o']),
                                'high': float(c['h']),
                                'low': float(c['l']),
                                'close': float(c['c']),
                                'volume': float(c['v'])
                            })
                    print(f"  ├─ {symbol}: Successfully seeded {len(self.market_data[symbol])} historical candles.")
                else:
                    print(f"  └─ ⚠️ Failed to seed {symbol}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"  └─ 🚨 Exception seeding {symbol}: {e}")

    def start_market_stream(self, hl_symbols: list, interval="3m"):
        """
        Opens an asynchronous WebSocket connection to Hyperliquid.
        Appends directly to a C-optimized deque to save massive CPU overhead.
        """
        if not hasattr(self, 'market_data'):
            self.market_data = {} 
        
        def on_message(ws, message):
            msg = json.loads(message)
            
            # Route only the live candlestick updates
            if msg.get("channel") == "candle":
                data = msg["data"]
                coin = data["s"] 
                
                new_candle = {
                    'time': pd.to_datetime(data['t'], unit='ms'),
                    'open': float(data['o']),
                    'high': float(data['h']),
                    'low': float(data['l']),
                    'close': float(data['c']),
                    'volume': float(data['v'])
                }
                
                # Thread safety lock for data appending
                with self.data_lock:
                    if coin not in self.market_data:
                        self.market_data[coin] = deque(maxlen=100)
                    self.market_data[coin].append(new_candle)

        def on_open(ws):
            print("\n🟢 Hyperliquid Market Stream Connected. Subscribing to tickers...")
            for symbol in hl_symbols:
                sub_payload = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "candle",
                        "coin": symbol,
                        "interval": interval
                    }
                }
                ws.send(json.dumps(sub_payload))

        def on_error(ws, error):
            print(f"🚨 WebSocket Error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            print("🔴 WebSocket Disconnected from Hyperliquid server.")

        def run_ws():
            reconnect_delay = 1
            while True:
                self.ws = websocket.WebSocketApp(
                    "wss://api.hyperliquid.xyz/ws",
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close
                )
                
                self.ws.run_forever()
                
                print(f"⚠️ Connection lost. Attempting reconnect in {reconnect_delay} seconds...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

        self.ws_thread = threading.Thread(target=run_ws)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def get_safe_market_df(self, symbol: str) -> pd.DataFrame:
        """
        Thread-safe getter for the live market data. 
        Converts the lightweight deque into a Pandas DataFrame strictly on-demand.
        """
        with self.data_lock:
            if hasattr(self, 'market_data') and symbol in self.market_data and len(self.market_data[symbol]) > 0:
                df = pd.DataFrame(list(self.market_data[symbol]))
                df.set_index('time', inplace=True)
                return df
            return None 

    def get_bbo(self, coin: str):
        """Fetches Best Bid and Best Offer for Maker execution."""
        try:
            # 1. Strip the prefix. The l2Book endpoint rejects the "dex" parameter,
            # so we just pass the raw coin name (e.g., "SP500").
            raw_coin = coin.replace("xyz:", "")
            payload = {"type": "l2Book", "coin": raw_coin}
                
            snap = self.info.post("/info", payload)
            
            # 2. Safety catch if the API rate limits or rejects the coin name
            if not snap:
                return None
                
            levels = snap.get("levels", [[], []])
            
            best_bid = float(levels[0][0]["px"]) if len(levels[0]) > 0 else 0.0
            best_ask = float(levels[1][0]["px"]) if len(levels[1]) > 0 else 0.0
            
            return {"bid": best_bid, "ask": best_ask}
            
        except Exception as e:
            print(f"🚨 Failed to fetch BBO for {coin}: {e}")
            return None              

    def get_balances(self):
        print(f"Checking L1 balances for: {self.main_wallet}\n")
        perps_state = self.info.user_state(self.main_wallet)
        print("--- PERPS ACCOUNT (Margin Summary) ---")
        print(json.dumps(perps_state.get("marginSummary", "No margin summary found"), indent=2))
        
        spot_state = self.info.spot_user_state(self.main_wallet)
        print("\n--- SPOT ACCOUNT (Balances) ---")
        print(json.dumps(spot_state.get("balances", "No spot balances found"), indent=2))

    def calculate_atr(self, ohlcv_df: pd.DataFrame, period: int = 14) -> float:
        df = ohlcv_df.copy()
        df['prev_close'] = df['close'].shift(1)
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['prev_close'])
        df['tr2'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        atr = df['tr'].ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return round(float(atr.iloc[-1]), 2)

    def get_tactical_signal(self, ohlcv_df: pd.DataFrame) -> str:
        if len(ohlcv_df) < 30:
            return "WAIT"

        ohlcv_df['EMA_5'] = ohlcv_df['close'].ewm(span=5, adjust=False).mean()
        ohlcv_df['EMA_9'] = ohlcv_df['close'].ewm(span=9, adjust=False).mean()
        ohlcv_df['EMA_21'] = ohlcv_df['close'].ewm(span=21, adjust=False).mean()

        ohlcv_df['EMA_12'] = ohlcv_df['close'].ewm(span=12, adjust=False).mean()
        ohlcv_df['EMA_26'] = ohlcv_df['close'].ewm(span=26, adjust=False).mean()
        ohlcv_df['MACD_Line'] = ohlcv_df['EMA_12'] - ohlcv_df['EMA_26']
        ohlcv_df['MACD_Signal'] = ohlcv_df['MACD_Line'].ewm(span=9, adjust=False).mean()
        ohlcv_df['MACD_Hist'] = ohlcv_df['MACD_Line'] - ohlcv_df['MACD_Signal']

        high = ohlcv_df['high']
        low = ohlcv_df['low']
        close_prev = ohlcv_df['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        ohlcv_df['ATR'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        
        current = ohlcv_df.iloc[-1]
        previous = ohlcv_df.iloc[-2]
        
        macd_bullish = current['MACD_Hist'] > 0
        macd_bearish = current['MACD_Hist'] < 0

        bearish_trend = (current['EMA_5'] < current['EMA_9']) and (current['EMA_9'] < current['EMA_21'])
        prev_bearish_trend = (previous['EMA_5'] < previous['EMA_9']) and (previous['EMA_9'] < previous['EMA_21'])
        
        if bearish_trend and not prev_bearish_trend and macd_bearish:
            return "EXECUTE_SHORT"
            
        bullish_trend = (current['EMA_5'] > current['EMA_9']) and (current['EMA_9'] > current['EMA_21'])
        prev_bullish_trend = (previous['EMA_5'] > previous['EMA_9']) and (previous['EMA_9'] > previous['EMA_21'])
        
        if bullish_trend and not prev_bullish_trend and macd_bullish:
            return "EXECUTE_LONG"
            
        return "WAIT"    

    def get_open_positions(self):
        try:
            if time.time() - self._last_positions_fetch < 2.5:
                return self._positions_cache
                
            wallet = getattr(self, 'main_wallet_address', getattr(self, 'main_wallet', None))
            if not wallet:
                return []
                
            active_positions = []
            dexes_to_check = ["", "xyz"] 
            
            for dex_name in dexes_to_check:
                payload = {"type": "clearinghouseState", "user": wallet}
                if dex_name:
                    payload["dex"] = dex_name
                    
                user_state = self.info.post("/info", payload)
                positions = user_state.get("assetPositions", [])
                
                for pos in positions:
                    details = pos.get("position", {})
                    size = float(details.get("szi", 0))
                    
                    if size != 0:
                        raw_coin = details.get("coin")
                        
                        formatted_coin = raw_coin
                        if dex_name and not raw_coin.startswith(f"{dex_name}:"):
                            formatted_coin = f"{dex_name}:{raw_coin}"
                            
                        active_positions.append({
                            "coin": formatted_coin,
                            "size": size,
                            "entry_price": float(details.get("entryPx", 0)),
                            "leverage": details.get("leverage", {}).get("value", 1)
                        })

            self._positions_cache = active_positions
            self._last_positions_fetch = time.time()
            return active_positions
            
        except Exception as e:
            print(f"🚨 [GATEKEEPER CRITICAL] Failed to fetch live positions from Hyperliquid: {e}")
            return []

    def close_position(self, symbol: str):
        open_positions = self.get_open_positions()
        pos = next((p for p in open_positions if p.get('coin') == symbol), None)
        
        if pos:
            sz = float(pos['size'])
            is_currently_long = sz > 0
            
            # Use the new safe getter to pull the latest price
            market_df = self.get_safe_market_df(symbol)
            current_price = market_df.iloc[-1]['close'] if market_df is not None else 0
            
            if current_price == 0:
                print(f"🚨 [CLOSE ERROR] No market data available to close {symbol}.")
                return None
                
            print(f"  🧹 [FLATTEN] Routing order to close {abs(sz)} {symbol}...")
            
            return self.place_perp_order(
                symbol=symbol,
                is_buy=not is_currently_long,
                sz=abs(sz),
                limit_px=current_price,
                reduce_only=True
            )
        else:
            print(f"  ⚠️ [FLATTEN] No active position found to close for {symbol}.")
            return None
    
    def start_trailing_stop(self, symbol: str, is_buy: bool, size: float, trail_mult: float, initial_price: float, leverage: int):
        def trail_monitor():
            import time
            import pandas as pd
            
            # --- FIX: Calculate actual ATR using the live safe dataframe ---
            current_atr = 0
            market_df = self.get_safe_market_df(symbol)
            
            if market_df is not None and len(market_df) >= 15:
                current_atr = self.calculate_atr(market_df, period=14)
            
            if current_atr == 0 or pd.isna(current_atr):
                current_atr = initial_price * 0.005
                print(f"  ⚠️ [ALGO TRAIL] ATR calculation failed. Using 0.5% safety fallback.")
                
            atr_distance = current_atr * trail_mult
            current_stop_px = initial_price - atr_distance if is_buy else initial_price + atr_distance
            
            print(f"🛡️ [ALGO TRAIL] Live tracker activated for {symbol} ({'LONG' if is_buy else 'SHORT'}) | Initial Stop: ${current_stop_px:,.2f}")
            
            high_water_mark = initial_price
            profit_locked = False 
            
            while True:
                time.sleep(5) 
                
                # Fetch fresh data using the safe lock
                market_df = self.get_safe_market_df(symbol)
                if market_df is None or len(market_df) == 0:
                    continue
                    
                current_price = market_df.iloc[-1]['close']
                
                # --- FIX: Recalculate dynamic ATR during the loop natively ---
                loop_atr = 0
                if len(market_df) >= 15:
                    loop_atr = self.calculate_atr(market_df, period=14)
                
                if loop_atr == 0 or pd.isna(loop_atr):
                    loop_atr = current_price * 0.005
                
                open_positions = self.get_open_positions()
                if not any(pos.get('coin') == symbol for pos in open_positions):
                    print(f"🛑 [ALGO TRAIL] No active position found for {symbol}. Shutting down tracker.")
                    break

                price_delta_pct = ((current_price - initial_price) / initial_price) * 100
                if not is_buy:
                    price_delta_pct = -price_delta_pct
                    
                roe_pct = price_delta_pct * leverage
                
                if roe_pct >= 5.0 and not profit_locked:
                    profit_locked = True
                    print(f"  🔒 [PROFIT LOCK] {symbol} crossed +5% ROE. Drawdown protection armed.")
                
                if profit_locked and roe_pct < 5.0:
                    print(f"  🛑 [PROFIT STOP] {symbol} fell below +5% ROE! Closing position to protect capital.")
                    self.place_perp_order(
                        symbol=symbol,
                        is_buy=not is_buy,
                        sz=size,
                        limit_px=current_price,
                        reduce_only=True
                    )
                    break
                
                active_mult = 0.75 if roe_pct >= 20.0 else trail_mult
                current_atr_distance = loop_atr * active_mult

                if is_buy: # LONG
                    if current_price > high_water_mark:
                        high_water_mark = current_price
                        
                    new_stop = high_water_mark - current_atr_distance
                    
                    if new_stop > current_stop_px:
                        old_stop = current_stop_px
                        current_stop_px = new_stop
                        print(f"  📈 [LIVE TRAIL] {symbol} (ROE: {roe_pct:+.2f}%) ➔ Stop UP: ${old_stop:,.2f} ➔ ${new_stop:,.2f} (ATR Mult: {active_mult})")
               
                else: # SHORT
                    if current_price < high_water_mark:
                        high_water_mark = current_price

                    new_stop = high_water_mark + current_atr_distance

                    if new_stop < current_stop_px:
                        old_stop = current_stop_px
                        current_stop_px = new_stop
                        print(f"  📉 [LIVE TRAIL] {symbol} (ROE: {roe_pct:+.2f}%) ➔ Stop DOWN: ${old_stop:,.2f} ➔ ${new_stop:,.2f} (ATR Mult: {active_mult})")

                # 🚨 THIS `else` USED TO BIND TO `if is_stopped_out`, NOT TO
                # `if is_buy`. Found 2026-09-26 while gating this file.
                # The stop-out test sat BETWEEN the long branch and the short
                # branch, so Python paired `else: # SHORT` with the nearest open
                # `if` -- the stop-out one. The comment said SHORT; the language
                # said "not stopped out", which for a long is the ordinary case.
                #
                # So on every quiet tick a LONG ran the short branch, whose
                # first statement is
                #     if current_price < high_water_mark: high_water_mark = ...
                # dragging the high-water mark DOWN to the current price. A
                # trailing stop whose high-water mark follows price down is not
                # a trailing stop: the ratchet at `new_stop > current_stop_px`
                # then computes from a depressed mark and stops advancing, so
                # the position gives back exactly the run the trail existed to
                # protect. Shorts were unaffected, which is why it survived.
                #
                # Indentation IS the control flow in Python, and a comment
                # naming a branch is not a test that the branch is reached.
                is_stopped_out = (current_price <= current_stop_px if is_buy
                                  else current_price >= current_stop_px)

                if is_stopped_out:
                    print(f"  🛑 [LIVE STOP-OUT] {symbol} hit trailing stop at ${current_price:,.2f}! Closing position.")

                    self.place_perp_order(
                        symbol=symbol,
                        is_buy=not is_buy,
                        sz=size,
                        limit_px=current_price,
                        reduce_only=True
                    )
                    break

        tracker_thread = threading.Thread(target=trail_monitor)
        tracker_thread.daemon = True
        tracker_thread.start()

    def _mark_px(self, symbol: str) -> float:
        """Best available price for `symbol`, for applying the notional cap.

        🚨 A CAP NEEDS A PRICE, AND A MISSING PRICE MUST NOT READ AS "CHEAP".
        Returns 0.0 when it cannot price the asset, and hl_gate.check() refuses
        on a zero notional rather than letting the order through on the
        technicality that 0 is under every cap. The same mistake on the Webull
        side -- a quote read from fields that do not exist, returning (0,0) --
        made every option look worthless for a week.
        """
        try:
            bbo = self.get_bbo(symbol) or {}
            bid, ask = float(bbo.get("bid") or 0), float(bbo.get("ask") or 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            if ask > 0:
                return ask
            if bid > 0:
                return bid
        except Exception:
            pass
        try:
            df = self.get_safe_market_df(symbol)
            if df is not None and len(df):
                return float(df.iloc[-1]["close"])
        except Exception:
            pass
        return 0.0

    def set_leverage(self, symbol: str, leverage: int, is_cross: bool = True):
        # Raising leverage increases risk without placing an order, so it is
        # gated like an entry. notional is not knowable here, so the cap is
        # applied at order time instead -- pass the cap itself to satisfy the
        # "a zero notional cannot be capped" check.
        hl_gate.check("set_leverage", symbol,
                      notional_usd=hl_gate.max_notional(),
                      leverage=leverage,
                      has_exchange=bool(self.exchange))
        try:
            print(f"🔧 Pushing leverage update: {symbol} -> {leverage}x (Cross Margin: {is_cross})")
            result = self.exchange.update_leverage(
                leverage=leverage, 
                name=symbol, 
                is_cross=is_cross
            )
            return result
        except Exception as e:
            print(f"🚨 Failed to update leverage for {symbol}: {e}")
            return None

    def place_perp_order(self, symbol: str, is_buy: bool, sz: float, limit_px: float, reduce_only: bool = False, post_only: bool = False):
        # 🚨 THE GATE IS OUTSIDE THE try/except, DELIBERATELY.
        # Everything below is wrapped in `except Exception: return None`, so a
        # Refused raised inside it would be swallowed and reported as "failed to
        # route order payload" -- which reads like a network problem and tells
        # you nothing about the flag you forgot. A refusal must reach the caller.
        hl_gate.check("place_perp_order", symbol,
                      notional_usd=abs(float(sz or 0)) * abs(float(limit_px or 0)),
                      reduce_only=reduce_only,
                      has_exchange=bool(self.exchange))
        try:
            if "SP500" in symbol:
                sz_decimals = 3
                px_decimals = 1
                slippage_pct = 0.0015  
            else:
                sz_decimals = 2
                px_decimals = 2
                slippage_pct = 0.01
                
            if post_only:
                # MAKER MODE: Place exactly at BBO, no slippage, Add Liquidity Only (ALO)
                slippage_px = round(limit_px, px_decimals)
                tif = "Alo"
            else:
                # TAKER MODE: Cross the spread, Immediate Or Cancel (IOC)
                slippage_px = limit_px * (1 + slippage_pct) if is_buy else limit_px * (1 - slippage_pct)
                slippage_px = round(slippage_px, px_decimals) 
                tif = "Ioc"
                
            sz = round(sz, sz_decimals)
            
            direction_str = "LONG" if is_buy else "SHORT"
            mode_str = "MAKER (Post-Only)" if post_only else "TAKER (IOC)"
            print(f"📦 [EXCHANGE] Routing {mode_str} {direction_str} order for {sz} {symbol} at ${slippage_px:,.2f}...")
            
            order_result = self.exchange.order(
                name=symbol,
                is_buy=is_buy,
                sz=sz,
                limit_px=slippage_px,
                order_type={"limit": {"tif": tif}}, 
                reduce_only=reduce_only
            )
            
            if order_result and order_result.get("status") == "ok":
                try:
                    statuses = order_result["response"]["data"]["statuses"]
                    status_detail = statuses[0]
                    
                    if "error" in status_detail:
                        print(f"  ❌ [EXCHANGE REJECTED] {status_detail['error']}")
                    else:
                        pass # Silenced for clean logs, handled in the engine loop
                except Exception as e:
                    pass
            else:
                print(f"  ⚠️ [EXCHANGE WARNING] Order returned abnormal status: {order_result}")
                
            return order_result
            
        except Exception as e:
            print(f"🚨 [EXCHANGE CRITICAL] Failed to route order payload: {e}")
            return None   

    def market_order(self, coin, is_buy, sz, slippage=0.01):
        """Executes an immediate market taker order."""
        # No limit price is supplied here, so the notional the cap is applied to
        # has to be looked up. _mark_px returns 0.0 if it cannot, and the gate
        # refuses on 0 rather than treating an unpriceable order as a small one.
        hl_gate.check("market_order", coin,
                      notional_usd=abs(float(sz or 0)) * self._mark_px(coin),
                      has_exchange=bool(self.exchange))

        action = "Buy/Long" if is_buy else "Sell/Short"
        print(f"Executing Market {action} for {sz} {coin}...")
        
        result = self.exchange.market_open(name=coin, is_buy=is_buy, sz=sz, px=None, slippage=slippage)
        self._parse_order_result(result)
        return result

    def limit_order(self, coin, is_buy, sz, limit_price, reduce_only=False):
        """Places a resting maker limit order on the book."""
        hl_gate.check("limit_order", coin,
                      notional_usd=abs(float(sz or 0)) * abs(float(limit_price or 0)),
                      reduce_only=reduce_only,
                      has_exchange=bool(self.exchange))

        action = "Buy/Long" if is_buy else "Sell/Short"
        order_type = {"limit": {"tif": "Gtc"}}
        
        print(f"Placing Limit {action} for {sz} {coin} @ ${limit_price} (Reduce Only: {reduce_only})...")
        
        result = self.exchange.order(
            name=coin, 
            is_buy=is_buy, 
            sz=sz, 
            limit_px=limit_price, 
            order_type=order_type,
            reduce_only=reduce_only
        )
        self._parse_order_result(result)
        return result

    def cancel_order(self, coin, oid):
        """Cancels a specific resting order by its OID."""
        # Cancelling can only reduce exposure, so it is never gated -- only
        # logged. gamma_scalper_engine cancels its resting maker order on every
        # loop where drift falls back inside the threshold; if a flag could
        # block that, stale orders would accumulate on the book.
        hl_gate.check("cancel_order", coin, reduce_only=True,
                      has_exchange=bool(self.exchange))

        print(f"Attempting to cancel Order #{oid} on {coin}...")
        result = self.exchange.cancel(name=coin, oid=oid)
        
        if result.get("status") == "ok":
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and isinstance(statuses[0], str) and statuses[0] == "success":
                print(f"SUCCESS: Order #{oid} has been removed from the book.")
            elif statuses and isinstance(statuses[0], dict) and "error" in statuses[0]:
                print(f"FAILED: {statuses[0]['error']}")
            else:
                print(f"Result: {result['response']}")
        else:
            print(f"API Error: {result}")
        return result

    def _parse_order_result(self, result):
        """Internal helper to parse and print standard order responses."""
        if result.get("status") == "ok":
            for status in result.get("response", {}).get("data", {}).get("statuses", []):
                if "resting" in status:
                    resting = status["resting"]
                    print(f"SUCCESS: Order #{resting['oid']} resting on the book.")
                elif "filled" in status:
                    filled = status["filled"]
                    print(f"SUCCESS: Order #{filled['oid']} filled {filled['totalSz']} @ ${filled['avgPx']}")
                elif "error" in status:
                    print(f"FAILED: {status['error']}")
        else:
            print(f"API Error: {result}")

    def set_stop_loss(self, coin, is_buy, sz, stop_price):
        """Deploys a reduce-only Stop-Loss market trigger order."""
        # reduce_only=True, so the gate logs this and always allows it. A stop
        # you cannot place because a flag is off is the worst of both worlds.
        hl_gate.check("set_stop_loss", coin,
                      notional_usd=abs(float(sz or 0)) * abs(float(stop_price or 0)),
                      reduce_only=True, has_exchange=bool(self.exchange))

        action = "Buy/Cover" if is_buy else "Sell/Close"
        print(f"Setting Stop-Loss {action} for {sz} {coin} at ${stop_price}...")
        
        order_type = {
            "trigger": {
                "triggerPx": stop_price,
                "isMarket": True,
                "tpsl": "sl"
            }
        }
        
        result = self.exchange.order(
            name=coin, 
            is_buy=is_buy, 
            sz=sz, 
            limit_px=stop_price, 
            order_type=order_type, 
            reduce_only=True
        )
        self._parse_order_result(result)
        return result

    def set_take_profit(self, coin, is_buy, sz, tp_price):
        """Deploys a reduce-only Take-Profit market trigger order."""
        hl_gate.check("set_take_profit", coin,
                      notional_usd=abs(float(sz or 0)) * abs(float(tp_price or 0)),
                      reduce_only=True, has_exchange=bool(self.exchange))

        action = "Buy/Cover" if is_buy else "Sell/Close"
        print(f"Setting Take-Profit {action} for {sz} {coin} at ${tp_price}...")
        
        order_type = {
            "trigger": {
                "triggerPx": tp_price,
                "isMarket": True,
                "tpsl": "tp"
            }
        }
        
        result = self.exchange.order(
            name=coin, 
            is_buy=is_buy, 
            sz=sz, 
            limit_px=tp_price, 
            order_type=order_type, 
            reduce_only=True
        )
        self._parse_order_result(result)
        return result

    def get_resting_orders(self, coin: str = None):
        """Fetches open orders, optionally filtered by a specific ticker."""
        # 1. Fetch all raw open orders from the builder DEX
        open_orders = self.info.open_orders(self.main_wallet_address, dex="xyz")
        
        # 2. If a specific coin was requested, filter the list
        if coin:
            filtered_orders = [order for order in open_orders if order.get("coin") == coin]
            return filtered_orders
            
        # 3. Otherwise, return the entire order book
        return open_orders

    def cancel_all_ticker_orders(self, coin, dex="xyz"):
        """Sweeps the order book and cancels all open orders for a specific asset."""
        hl_gate.check("cancel_all_ticker_orders", coin, reduce_only=True,
                      has_exchange=bool(self.exchange))

        print(f"Sweeping open orders for {coin}...")
        open_orders = self.info.open_orders(self.main_wallet, dex=dex)
        
        for order in open_orders:
            if order.get("coin") == coin:
                oid = order["oid"]
                print(f"Canceling Order #{oid}...")
                result = self.exchange.cancel(name=coin, oid=oid)
                
                if result.get("status") == "ok":
                    print(f"SUCCESS: Order #{oid} canceled.")
                else:
                    print(f"FAILED to cancel Order #{oid}: {result}")

    def find_ticker(self, search_term):
        """Diagnostic tool to search the engine's internal dictionary for valid tickers."""
        available_tickers = list(self.info.name_to_coin.keys())
        matches = [t for t in available_tickers if search_term.upper() in t.upper()]
        print(f"Matching '{search_term}' Tickers found:", matches)
        return matches

    def stream_live_prices(self, coin: str):
        print(f"Opening WebSocket connection for {coin} live trades...")
        print("Press Ctrl+C to stop the stream.\n")

        def handle_trade(trade_data):
            try:
                # 1. Safely locate the list of trades regardless of SDK version wrapper
                if isinstance(trade_data, list):
                    trades = trade_data
                elif isinstance(trade_data, dict):
                    # If it's a dictionary, look for the 'data' key or wrap the dict itself
                    trades = trade_data.get("data", [trade_data])
                else:
                    return  # Skip anything unrecognized
                
                # 2. Iterate through and extract data safely
                for trade in trades:
                    if isinstance(trade, dict):
                        px = trade.get("px")
                        sz = trade.get("sz")
                        side = trade.get("side")
                        
                        if px and side:
                            direction = "BUY  🟢" if side == "B" else "SELL 🔴"
                            print(f"{direction} | {sz} {coin} @ ${float(px):,.2f}")
                            
            except Exception:
                # If a malformed tick slips through, quietly ignore it 
                # so it doesn't crash your entire listener thread
                pass

        # 3. Subscribe to the live feed
        self.info.subscribe({"type": "trades", "coin": coin}, handle_trade)
        
        # 4. Keep the main script alive while the background listener runs
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nLive feed disconnected successfully.")

        # 🚨 A SECOND subscribe() USED TO SIT HERE, after the loop. It ran on
        # Ctrl-C -- i.e. the teardown path re-subscribed to the feed it was
        # tearing down, leaving a second handler attached to a socket nobody
        # reads. Removed 2026-09-26.


# ==========================================
# READ-ONLY SELF-CHECK
#
# 🚨 NOTHING BELOW PLACES AN ORDER, AND NOTHING SHOULD BE ADDED THAT DOES.
# `main.py` is the cautionary tale: a live `set_take_profit()` at module scope
# that fired on import and was stopped only by a TypeError. If you want to test
# execution, do it in a throwaway file you delete, not in a module something
# else might one day import.
# ==========================================
if __name__ == "__main__":
    load_dotenv()

    WALLET = os.getenv("MAIN_WALLET")
    AGENT = os.getenv("AGENT_SECRET")

    if not WALLET:
        print("Please configure your .env file with MAIN_WALLET and AGENT_SECRET.")
        raise SystemExit(1)

    hl_gate.banner(force=True)

    client = HyperExposureClient(main_wallet_address=WALLET, agent_secret=AGENT)
    client.get_balances()
    print("\n" + "=" * 40 + "\n")
    open_pos = client.get_open_positions()
    for p in open_pos:
        print(f"  {p['coin']:>12}  size {p['size']:+.4f}  entry "
              f"${p['entry_price']:,.2f}  {p['leverage']}x")
    if not open_pos:
        print("  (no open perp positions)")