import os
import time
import json
import logging
import threading
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import asyncio
import websockets

from webull_gamma_client import WebullGammaClient
from hyper_exposure_client import HyperExposureClient
from macro_options_scanner import MacroScanner
from local_black_scholes_engine import LocalBlackScholesEngine
from portfolio_manager import CapitalAllocator
from config import WATCHLIST, DRY_RUN, PAPER_TRADING
import hl_gate

# --- SUPPRESS NOISY WEBULL SDK ERROR LOGS ---
logging.getLogger('webull').setLevel(logging.CRITICAL)

POLL_INTERVAL = 3
BASE_FEE_RATE = 0.00035  

logging.basicConfig(
    filename="gamma_hedge_executions.jsonl",
    level=logging.INFO,
    format="%(message)s"
)

class GammaScalpEngine:
    def __init__(self):
        print("=== INITIALIZING GAMMA SCALPER ===")
        load_dotenv()
        
        self.scanner = MacroScanner()
        self.webull = WebullGammaClient(
            app_key=os.getenv("WEBULL_APP_KEY"),
            app_secret=os.getenv("WEBULL_APP_SECRET"),
            account_id=os.getenv("WEBULL_ACCOUNT_ID"),
            paper_trading=PAPER_TRADING
        )
        self.hyper = HyperExposureClient(
            main_wallet_address=os.getenv("MAIN_WALLET"),
            agent_secret=os.getenv("AGENT_SECRET")
        )
        # (constructed once; this line was duplicated, building and discarding
        #  a first engine on every startup)
        self.bs_engine = LocalBlackScholesEngine(risk_free_rate=0.045)
        self.active_hedges = {}
        self.in_flight_adjustments = {} 
        self.active_closing_orders = set() # NEW: Tracks resting Webull orders 
        
        # --- NEW: THREAD-SAFE API CACHE ---
        self._wb_fetch_lock = threading.Lock()
        self._wb_cached_positions = []
        self._last_wb_fetch = 0
        
        # --- UI WEBSOCKET STATE ---
        self.ui_clients = set()
        self.ui_logs = []
        self.ui_state_cache = {}
        self.total_scalps = 0

    def log_to_ui(self, message: str):
        print(message)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.ui_logs.insert(0, f"[{timestamp}] {message}")
        if len(self.ui_logs) > 30:
            self.ui_logs.pop()

    async def ws_handler(self, websocket):
        self.ui_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.ui_clients.remove(websocket)

    async def _ws_main(self):
        async with websockets.serve(self.ws_handler, "localhost", 8765):
            await asyncio.Future()  # run forever

    def start_ui_server(self):
        """Runs the WebSocket server on port 8765 safely using asyncio.run()."""
        asyncio.run(self._ws_main())

    def broadcast_state_loop(self):
        while True:
            time.sleep(1.0)
            if not self.ui_clients:
                continue
                
            net_delta = sum(item.get("wbDelta", 0) for item in self.ui_state_cache.values())
            
            payload = {
                "portfolioHealth": {
                    "webullCash": 10000.0,
                    "hlBuffer": 50000.0,
                    "netDelta": round(net_delta, 2),
                    "gammaScalpsToday": self.total_scalps,
                    "totalROE": 0.0
                },
                "activeHedges": list(self.ui_state_cache.values()),
                "directionalTrades": [],
                "logs": self.ui_logs
            }
            
            # Broadcast state to connected UI clients
            for client in list(self.ui_clients):
                try:
                    # Since broadcast loop runs in a thread, we can schedule coroutine safely
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(client.send(json.dumps(payload)))
                    loop.close()
                except Exception:
                    pass

    def _fetch_webull_positions_safely(self):
        """
        A centralized, rate-limited getter. 
        Ensures 50 threads don't make 50 separate API calls per second.
        Caches the Webull response for 10 seconds.
        """
        with self._wb_fetch_lock:
            current_time = time.time()
            if current_time - self._last_wb_fetch < 10.0:
                return self._wb_cached_positions
            
            try:
                trade_client = self.webull.trade_client
                if hasattr(trade_client, 'account_v2') and hasattr(trade_client.account_v2, 'get_account_position'):
                    response = trade_client.account_v2.get_account_position(self.webull.account_id)
                elif hasattr(trade_client.account, 'get_account_position'):
                    response = trade_client.account.get_account_position(self.webull.account_id)
                elif hasattr(trade_client.account, 'get_positions'):
                    response = trade_client.account.get_positions(self.webull.account_id)
                elif hasattr(trade_client.account, 'get_account_positions'):
                    response = trade_client.account.get_account_positions(self.webull.account_id)
                else:
                    return self._wb_cached_positions

                positions = response.json() if hasattr(response, 'json') else response
                
                if isinstance(positions, dict):
                    if 'data' in positions: positions = positions['data']
                    elif 'holdings' in positions: positions = positions['holdings']
                    
                if isinstance(positions, list):
                    self._wb_cached_positions = positions
                    
                self._last_wb_fetch = current_time
            except Exception as e:
                # If we still hit a rate limit, gracefully ignore and return the cache
                pass
                
        return self._wb_cached_positions

    def sync_local_ledger(self, ticker: str):
        self.bs_engine.clear_positions(ticker)
        try:
            positions = self._fetch_webull_positions_safely()
            if not positions: return
                
            for pos in positions:
                is_option = pos.get('assetType') == 'OPTION' or pos.get('instrument_type') == 'OPTION'
                sym = pos.get('ticker', {}).get('symbol') or pos.get('symbol')
                
                # Fallback for V2 'legs' format if base symbol is empty
                if not sym and 'legs' in pos and len(pos['legs']) > 0:
                    sym = pos['legs'][0].get('symbol', '')
                
                if is_option and sym and sym.startswith(ticker):
                    # Handle V2 'legs' format vs V1 'ticker' format
                    qty = int(float(pos.get('position', pos.get('quantity', 0))))
                    legs = pos.get('legs', [{}])
                    main_leg = legs[0] if legs else pos.get('ticker', {})
                    
                    strike = float(main_leg.get('strikePrice', main_leg.get('option_exercise_price', 0)))
                    call_put_str = main_leg.get('callPut', main_leg.get('option_type', 'call')).lower()
                    is_call = 'call' in call_put_str
                    
                    # V2 drops live IV from the account endpoint, fallback to 30% baseline if missing
                    greeks = pos.get('greeks', {})
                    raw_iv = float(greeks.get('impliedVolatility', 0.30)) 
                    iv = raw_iv if 0 < raw_iv < 1.0 else (raw_iv / 100.0 if raw_iv >= 1.0 else 0.30)
                    
                    expiry_raw = main_leg.get('expireDate', main_leg.get('option_expire_date', '2099-01-01'))
                    expiry_str = expiry_raw.split('T')[0][:10]
                    expiry_ts = datetime.strptime(f"{expiry_str} 16:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
                    
                    # Reconstruct the OCC string for V2
                    date_str = expiry_str[2:].replace("-", "") 
                    cp = "C" if is_call else "P"
                    strike_str = f"{int(strike * 1000):08d}"
                    contract_sym = f"{ticker}{date_str}{cp}{strike_str}"
                    
                    self.bs_engine.add_position(
                        underlying_ticker=ticker, symbol=contract_sym, strike=strike,
                        is_call=is_call, quantity=qty, implied_volatility=iv, expiry_timestamp=expiry_ts
                    )

            # --- NEW: IN-FLIGHT ORDER TRACKING & FILL DETECTION ---
            current_portfolio_occs = {p["symbol"] for p in self.bs_engine.positions.get(ticker, [])}
            
            filled_orders = []
            for resting_occ in list(self.active_closing_orders):
                # If the resting order is for this ticker, but the asset is no longer in our portfolio...
                if resting_occ.startswith(ticker) and resting_occ not in current_portfolio_occs:
                    filled_orders.append(resting_occ)
                    self.active_closing_orders.remove(resting_occ)
                    
            for occ in filled_orders:
                self.log_to_ui(f"🎉 [FILL CONFIRMED] Limit order for {occ} executed! Webull Position flattened.")

        except Exception:
            pass


    def calculate_optimal_threshold(self, ticker_price: float, current_gamma: float):
        if current_gamma == 0: return 0.10 
        min_price_swing = 2 * ticker_price * BASE_FEE_RATE
        min_delta_threshold = current_gamma * min_price_swing
        return max(0.02, min(0.20, min_delta_threshold * 1.20))

    def run_dynamic_grid_search(self, hl_symbol: str, current_gamma: float, current_theta: float, current_price: float):
        if current_gamma <= 0: return self.calculate_optimal_threshold(current_price, current_gamma)
        raw_df = self.hyper.get_safe_market_df(hl_symbol)
        if raw_df is None or len(raw_df) < 50: return self.calculate_optimal_threshold(current_price, current_gamma)
             
        df = raw_df.copy()
        threshold_grid = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
        best_threshold = threshold_grid[0]
        max_net_profit = -float('inf')
        time_fraction = len(df) / 480.0
        theta_cost = abs(current_theta) * time_fraction 

        for threshold in threshold_grid:
            required_price_swing = threshold / current_gamma
            scalp_count = 0
            last_hedge_price = df.iloc[0]['close']
            for price in df['close']:
                if abs(price - last_hedge_price) >= required_price_swing:
                    scalp_count += 1
                    last_hedge_price = price
            gross_profit = scalp_count * (0.5 * current_gamma * (required_price_swing ** 2))
            total_fees = scalp_count * (threshold * current_price * BASE_FEE_RATE)
            net_profit = gross_profit - total_fees - theta_cost
            if net_profit > max_net_profit:
                max_net_profit = net_profit
                best_threshold = threshold
                
        return max(self.calculate_optimal_threshold(current_price, current_gamma), best_threshold)

    def execute_atomic_roll(self, ticker: str, pos: dict, target_delta: float):
        action = "SELL" if pos["qty"] > 0 else "BUY"
        qty = abs(pos["qty"])
        self.log_to_ui(f"    ├─ 1. Liquidating leg: {action} TO CLOSE {qty}x {pos['symbol']}")
        self.webull.place_option_order(ticker, pos["symbol"], action=action, quantity=qty, is_closing=True)

    def monitor_position_health(self, ticker: str, current_price: float, regime: str) -> bool:
        if ticker not in self.bs_engine.positions: return False
        current_time = time.time()
        rolled = False
        for pos in list(self.bs_engine.positions[ticker]):
            # If we already have a resting limit order to close this, leave it alone!
            if pos["symbol"] in self.active_closing_orders:
                continue
                
            years_to_expiry = (pos["expiry_ts"] - current_time) / (365.25 * 24 * 3600)
            if years_to_expiry <= 0: continue
            greeks = self.bs_engine._calculate_contract_greeks(
                spot_price=current_price, strike=pos["strike"],
                time_to_expiry_years=years_to_expiry, iv=pos["iv"], is_call=pos["is_call"]
            )
            delta = abs(greeks["delta"])
            is_long = pos["qty"] > 0
            if regime == "LONG_GAMMA" and is_long and (delta < 0.20 or delta > 0.80):
                self.log_to_ui(f"🚨 [AUTO-ROLL] {ticker} Delta {delta:.2f}. Rolling leg.")
                self.active_closing_orders.add(pos["symbol"])
                self.execute_atomic_roll(ticker, pos, target_delta=0.50)
                rolled = True
            elif regime == "SHORT_VOL" and not is_long and (delta < 0.05 or delta > 0.30):
                self.log_to_ui(f"🚨 [AUTO-ROLL] {ticker} Delta {delta:.2f}. Rolling leg.")
                self.active_closing_orders.add(pos["symbol"])
                self.execute_atomic_roll(ticker, pos, target_delta=0.16)
                rolled = True
        return rolled

    def get_live_hyperliquid_position(self, hl_symbol: str) -> float:
        for pos in self.hyper.get_open_positions():
            if pos.get('coin') == hl_symbol: return float(pos.get('size', 0))
        return 0.0

    def manage_hedge_loop(self, ticker: str, hl_symbol: str, leverage: int, regime: str):
        """
        The core engine loop. Runs infinitely in a background thread for each asset.
        """
        print(f"🛡️ [HEDGE ENGINE] Spinning up dynamic delta tracker for {ticker} -> {hl_symbol}")

        if not DRY_RUN:
            # 🚨 CHECK THE GATE HERE, AT THREAD START, NOT AT THE FIRST ORDER.
            # hl_gate raises Refused, and the loop below ends in
            # `except Exception: time.sleep(5)` -- so a refusal discovered inside
            # the loop would be swallowed and retried silently every 5 seconds
            # for the rest of the session. The hedge would read as "running" in
            # the UI while placing nothing. Fail here, loudly, once.
            try:
                self.hyper.set_leverage(hl_symbol, leverage, is_cross=True)
            except hl_gate.Refused as e:
                print(f"  🔒 [HEDGE ENGINE] {ticker} -> {hl_symbol} NOT STARTED: {e}")
                self.log_to_ui(f"🔒 {hl_symbol} hedge disabled: {e}")
                self.ui_state_cache[ticker] = {
                    "ticker": ticker, "type": regime.replace("_", " "),
                    "wbDelta": 0.0, "hlPos": 0.0, "drift": 0.0,
                    "status": "PERP EXECUTION LOCKED",
                }
                return


        last_grid_search_time = 0 
        active_threshold = 0.05  
        active_maker_oid = None 
            
        while True:
            try:
                # Polling Webull portfolio safely using the new cached gatekeeper
                self.sync_local_ledger(ticker)
                
                # --- STEP 1: Fetch Underlying Price ---
                market_df = self.hyper.get_safe_market_df(hl_symbol)
                if market_df is not None: current_price = market_df.iloc[-1]['close']
                else:
                    time.sleep(1)
                    continue

                is_spx_hedge = "SP500" in hl_symbol or "SPX" in hl_symbol
                webull_spot_price = (current_price / 10.0) if is_spx_hedge else current_price

                if self.monitor_position_health(ticker, webull_spot_price, regime):
                    time.sleep(POLL_INTERVAL)
                    continue
                
                wb_greeks = self.bs_engine.get_live_portfolio_greeks(ticker, current_spot_price=webull_spot_price)
                webull_net_delta = wb_greeks.get("net_delta", 0.0)
                current_gamma = wb_greeks.get("net_gamma", 0.0)
                current_theta = wb_greeks.get("net_theta", 0.0)
                
                notional_risk = webull_net_delta * webull_spot_price
                target_perp_position = -(notional_risk / current_price)
                
                current_time = time.time()
                raw_perp_position = self.get_live_hyperliquid_position(hl_symbol)
                in_flight = self.in_flight_adjustments.get(hl_symbol, {'size': 0.0, 'timestamp': 0})
                in_flight_size = in_flight['size'] if (current_time - in_flight['timestamp']) <= 5.0 else 0.0
                if in_flight_size == 0.0: self.in_flight_adjustments[hl_symbol] = {'size': 0.0, 'timestamp': 0}
                    
                current_perp_position = raw_perp_position + in_flight_size
                delta_drift = target_perp_position - current_perp_position

                if (current_time - last_grid_search_time) >= 3600:
                    active_threshold = self.run_dynamic_grid_search(hl_symbol, current_gamma, current_theta, current_price)
                    last_grid_search_time = current_time

                self.ui_state_cache[ticker] = {
                    "ticker": ticker, "type": regime.replace("_", " "), 
                    "wbDelta": round(webull_net_delta, 2), "hlPos": round(current_perp_position, 2),
                    "drift": round(delta_drift, 2), "status": "CHASING BBO" if abs(delta_drift) >= active_threshold else "HEDGED"
                }

                if abs(delta_drift) < active_threshold:
                    if active_maker_oid and not DRY_RUN:
                        self.hyper.cancel_order(hl_symbol, active_maker_oid)
                        active_maker_oid = None
                    time.sleep(POLL_INTERVAL)
                    continue

                if (abs(delta_drift) * current_price) < 12.50:
                    time.sleep(POLL_INTERVAL)
                    continue
                     
                if active_maker_oid and not DRY_RUN:
                    self.hyper.cancel_order(hl_symbol, active_maker_oid)
                    active_maker_oid = None
                     
                action = "BUY" if delta_drift > 0 else "SELL"
                is_buy = (action == "BUY")
                size_to_trade = abs(delta_drift)
                bbo = self.hyper.get_bbo(hl_symbol) or {"bid": current_price, "ask": current_price}
                limit_px = bbo['bid'] if is_buy else bbo['ask']
                
                self.log_to_ui(f"🚨 MAKER SCALP: {action} {size_to_trade:.4f} {hl_symbol} @ ${limit_px:.2f}")
                
                if DRY_RUN:
                    self.in_flight_adjustments[hl_symbol] = {'size': size_to_trade if is_buy else -size_to_trade, 'timestamp': time.time()}
                    self.log_to_ui("  🟢 [DRY RUN] Sandbox Maker order logged locally.")
                else:
                    order_result = self.hyper.place_perp_order(
                        symbol=hl_symbol, is_buy=is_buy, sz=size_to_trade, limit_px=limit_px, post_only=True
                    )
                    if order_result and order_result.get("status") == "ok":
                        try:
                            statuses = order_result["response"]["data"]["statuses"]
                            if "resting" in statuses[0]:
                                active_maker_oid = statuses[0]["resting"]["oid"]
                                self.log_to_ui(f"  ✅ Resting safely on book (OID: {active_maker_oid})")
                        except Exception:
                            pass
                time.sleep(POLL_INTERVAL)
            # 🚨 Refused IS CAUGHT BEFORE THE BARE except, AND IT STOPS THE
            # THREAD. It is a configuration verdict, not a transient fault:
            # retrying it every 5 seconds cannot change the answer, and the bare
            # handler below would hide it while the UI still said "HEDGED".
            except hl_gate.Refused as e:
                print(f"  🔒 [HEDGE ENGINE] {hl_symbol} stopping — {e}")
                self.log_to_ui(f"🔒 {hl_symbol} hedge stopped: {e}")
                self.ui_state_cache.setdefault(ticker, {})["status"] = \
                    "PERP EXECUTION LOCKED"
                return
            except Exception:
                time.sleep(5)

    def run_master(self):
        """
        Starts the Hyperliquid WebSocket and spins up a dedicated hedging thread
        for every ticker in the config.
        """
        approved_tickers = self.scanner.scan_watchlist()
        
        print("\n🏦 Fetching live account balances for Capital Allocation...")
        
        wb_health = self.webull.get_account_health()
        wb_cash = float(wb_health.get('total_cash_balance', wb_health.get('dayBuyingPower', wb_health.get('overnightBuyingPower', 0)))) if wb_health else 0.0

        try:
            payload = {"type": "clearinghouseState", "user": self.hyper.main_wallet, "dex": "xyz"}
            hl_state = self.hyper.info.post("/info", payload)
            
            margin_summary = hl_state.get("marginSummary", {})
            hl_account_value = float(margin_summary.get("accountValue", 0))
            hl_margin_used = float(margin_summary.get("totalMarginUsed", 0))
            hl_cash = hl_account_value - hl_margin_used
        except Exception as e:
            hl_cash = 0.0

        # TEMPORARY BYPASS: If balances fail, inject virtual cash so the test runs
        if wb_cash == 0.0:
            print("  ⚠️ Webull Balance returned 0. Injecting $10k virtual cash to bypass Allocator block.")
            wb_cash = 10000.0
            
        # --- MISSING BYPASS ADDED HERE ---
        if hl_cash == 0.0:
            print("  💡 [SANDBOX] Injecting $50k Hyperliquid virtual cash to bypass Allocator block.")
            hl_cash = 50000.0

        # --- FORCE TRACK MANUAL POSITIONS ---
        manual_overrides = []
        try:
            pos_data = self._fetch_webull_positions_safely()
            
            print(f"\n  [DEBUG] Raw Webull Positions Returned: {len(pos_data) if isinstance(pos_data, list) else pos_data}")
            
            if isinstance(pos_data, list):
                approved_symbols = {t["ticker"] for t in approved_tickers}
                for p in pos_data:
                    is_option = p.get('assetType') == 'OPTION' or p.get('instrument_type') == 'OPTION'
                    if is_option:
                        sym = p.get('ticker', {}).get('symbol') or p.get('symbol')
                        
                        # Handle V2 'legs' fallback for symbol base
                        if not sym and 'legs' in p and len(p['legs']) > 0:
                            sym = p['legs'][0].get('symbol', '')
                            
                        # Extract the base ticker (e.g. 'AAPL' from 'AAPL241018C00200000')
                        base_ticker = None
                        for t in WATCHLIST.keys():
                            if sym and sym.startswith(t):
                                base_ticker = t
                                break
                                
                        if base_ticker and base_ticker not in approved_symbols:
                            print(f"  🔍 [MANUAL OVERRIDE] Found active {base_ticker} position! Bypassing Allocator & Forcing Hedge Thread...")
                            manual_overrides.append({
                                "ticker": base_ticker,
                                "regime": "LONG_GAMMA", 
                                "spot_price": 0,
                                "leverage": WATCHLIST[base_ticker]["max_leverage"]
                            })
                            approved_symbols.add(base_ticker)
        except Exception as e:
            print(f"  ⚠️ [DEBUG] Failed to fetch manual positions: {e}")

        allocator = CapitalAllocator(available_webull_cash=wb_cash, available_hl_cash=hl_cash)
        allocated_trades = allocator.allocate_portfolio(approved_tickers)

        # --- FORCE TRACK MANUAL POSITIONS ---
        # Run this AFTER the allocator. If the allocator rejected a trade but we hold the position, rescue it!
        allocated_symbols = {t["ticker"] for t in allocated_trades}
        manual_overrides = []
        
        try:
            pos_data = self._fetch_webull_positions_safely()
            
            if isinstance(pos_data, list):
                for p in pos_data:
                    is_option = p.get('assetType') == 'OPTION' or p.get('instrument_type') == 'OPTION'
                    if is_option:
                        sym = p.get('ticker', {}).get('symbol') or p.get('symbol')
                        
                        # Handle V2 'legs' fallback for symbol base
                        if not sym and 'legs' in p and len(p['legs']) > 0:
                            sym = p['legs'][0].get('symbol', '')
                            
                        # Extract the base ticker
                        base_ticker = None
                        for t in WATCHLIST.keys():
                            if sym and sym.startswith(t):
                                base_ticker = t
                                break
                                
                        if base_ticker and base_ticker not in allocated_symbols:
                            # Prevent duplicates if holding multiple options for the same ticker
                            if base_ticker not in [m["ticker"] for m in manual_overrides]:
                                print(f"  🔍 [MANUAL OVERRIDE] Found active {base_ticker} position! Bypassing Allocator & Forcing Hedge Thread...")
                                manual_overrides.append({
                                    "ticker": base_ticker,
                                    "regime": "LONG_GAMMA", 
                                    "spot_price": 0,
                                    "leverage": WATCHLIST[base_ticker]["max_leverage"]
                                })
        except Exception as e:
            pass

        # Merge newly funded trades with existing active positions
        final_execution_list = allocated_trades + manual_overrides

        if not final_execution_list:
            print("🛑 ALLOCATION FATAL: Insufficient capital to fund any approved trades and no active positions found. Sleeping.")
            return

        hl_symbols = [WATCHLIST[item["ticker"]]["hl_symbol"] for item in final_execution_list]
        self.hyper.start_market_stream(hl_symbols=hl_symbols, interval="1m")
        
        print("\n⏳ Allowing 5 seconds for WebSockets to saturate with prices...")
        time.sleep(5)
        
        for item in final_execution_list:
            ticker = item["ticker"]
            regime = item.get("regime", "LONG_GAMMA")
            settings = WATCHLIST[ticker]
            hl_symbol = settings["hl_symbol"]
            leverage = settings["max_leverage"]
            
            self.sync_local_ledger(ticker)
            
            thread = threading.Thread(
                target=self.manage_hedge_loop,
                args=(ticker, hl_symbol, leverage, regime)
            )
            thread.daemon = True
            thread.start()
            self.active_hedges[ticker] = thread
            time.sleep(0.5)  
            
        # Start UI WebSocket server and state broadcaster
        threading.Thread(target=self.start_ui_server, daemon=True).start()
        threading.Thread(target=self.broadcast_state_loop, daemon=True).start()
        
        self.log_to_ui("🌐 WebSocket UI Server broadcasting on ws://localhost:8765")
        self.log_to_ui("🟢 ALL HEDGE THREADS ACTIVE.")
        
        try:
            while True: time.sleep(60)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")

if __name__ == "__main__":
    bot = GammaScalpEngine()
    bot.run_master()