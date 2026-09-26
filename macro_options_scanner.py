import os
import math
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from webull_gamma_client import WebullGammaClient
from unusual_whales_client import UnusualWhalesClient
from local_black_scholes_engine import LocalBlackScholesEngine
from strike_selector import StrikeSelector
from config import WATCHLIST, OPTIONS_TRADING_LEVEL, DRY_RUN, PAPER_TRADING

class MacroScanner:
    def __init__(self):
        load_dotenv()
        self.webull = WebullGammaClient(
            app_key=os.getenv("WEBULL_APP_KEY"),
            app_secret=os.getenv("WEBULL_APP_SECRET"),
            account_id=os.getenv("WEBULL_ACCOUNT_ID"),
            paper_trading=PAPER_TRADING
        )
        
        # Initialize Level 3 Advanced Modules
        self.uw = UnusualWhalesClient(os.getenv("UW_API_KEY"))
        self.bs_engine = LocalBlackScholesEngine(risk_free_rate=0.045)
        self.strike_selector = StrikeSelector(self.bs_engine)

    def get_spot_price(self, ticker: str) -> float:
        """Fast helper to grab the live Spot Price for the Black-Scholes engine."""
        try:
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            data = requests.get(url, headers=headers, timeout=5).json()
            return data['chart']['result'][0]['meta']['regularMarketPrice']
        except Exception:
            return 0.0

    def get_uw_volatility_metrics(self, ticker: str):
        """
        Pulls both IV and RV natively from Unusual Whales, completely 
        bypassing Yahoo Finance and Webull rate limits.
        """
        iv = 0.0
        rv = 0.0
        
        # A bulletproof helper to catch API nulls and empty strings
        def safe_float(val):
            if val is None or val == "": 
                return 0.0
            try: 
                return float(val)
            except (ValueError, TypeError): 
                return 0.0
        
        try:
            # 1. Fetch Realized Volatility (Usually returns a time-series array)
            rv_data = self.uw.get_realized_volatility(ticker)
            if rv_data and isinstance(rv_data, list):
                
                # Hunt backwards for the most recent settled RV
                for day in reversed(rv_data):
                    temp_rv = safe_float(day.get("realized_volatility")) or safe_float(day.get("volatility"))
                    if temp_rv > 0:
                        rv = temp_rv
                        break 
                        
                # Hunt backwards for the most recent settled IV
                for day in reversed(rv_data):
                    temp_iv = safe_float(day.get("implied_volatility"))
                    if temp_iv > 0:
                        iv = temp_iv
                        break 

            # 2. If IV wasn't included, hit the dedicated Interpolated IV endpoint
            if iv == 0.0:
                iv_data = self.uw.get_interpolated_iv(ticker)
                if iv_data:
                    latest_iv = iv_data[-1] if isinstance(iv_data, list) else iv_data
                    iv = safe_float(latest_iv.get("interpolated_iv")) or safe_float(latest_iv.get("iv"))
            
            # 3. Normalize to percentages (e.g., 0.25 -> 25.0%)
            if 0 < iv < 2.0: iv *= 100
            if 0 < rv < 2.0: rv *= 100
                
        except Exception as e:
            print(f"  ⚠️ [DEBUG] Failed to parse UW Volatility: {e}")
            
        return iv, rv

    def calculate_iv_rank(self, ticker: str, current_iv: float) -> float:
        try:
            iv_52w_high, iv_52w_low = 75.0, 15.0
            if current_iv == 0.0: return 0.0
            
            iv_52w_high = max(iv_52w_high, current_iv + 1)
            iv_52w_low = min(iv_52w_low, current_iv)
            return round(((current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)) * 100, 2)
        except Exception:
            return 0.0

    def scan_watchlist(self):
        print("\n" + "="*70)
        print(f"🔍 TIER 1: LEVEL-{OPTIONS_TRADING_LEVEL} MACRO VOLATILITY SCANNER INITIATED")
        print("="*70)
        
        approved_tickers = []
        
        for ticker in WATCHLIST.keys():
            print(f"\nEvaluating {ticker}...")
            
            spot = self.get_spot_price(ticker)
            
            # --- NEW UNUSUAL WHALES VOLATILITY ENGINE ---
            iv, rv = self.get_uw_volatility_metrics(ticker)
            time.sleep(0.5) 
            
            # --- FETCH LEVEL 3 MACRO DATA ---
            gex_levels = self.uw.get_gex_levels(ticker)
            vol_char = self.uw.get_volatility_character(ticker)
            hurst = vol_char.get('hurst_exponent', 0.5) if vol_char else 0.5
            
            if iv == 0.0 or rv == 0.0:
                print(f"  ⚠️ [DATA ERROR] Could not fetch live Volatility metrics from UW for {ticker}. Skipping.")
                continue
                
            ivr = self.calculate_iv_rank(ticker, iv)
            vrp = iv - rv
            
            print(f"  ├─ Spot Price:       ${spot:,.2f}")
            if gex_levels:
                print(f"  ├─ GEX Walls:        Call: ${gex_levels.get('call_wall', 0)} | Put: ${gex_levels.get('put_wall', 0)}")
            print(f"  ├─ Hurst Exponent:   {hurst:.3f} {'(Trending)' if hurst > 0.5 else '(Mean-Reverting)'}")
            print(f"  ├─ Garman-Klass RV:  {rv:.2f}%")
            print(f"  ├─ Implied Vol (IV): {iv:.2f}%")
            print(f"  ├─ VRP Spread:       {vrp:+.2f}%")
            print(f"  └─ IV Rank (IVR):    {ivr:.1f}")
            
            if vrp < 0 and ivr < 30.0:
                print(f"  ✅ REGIME A (LONG GAMMA): {ticker} options are mathematically cheap.")
                print(f"     Action: BUY Put/Call & Delta Hedge with Trade.xyz Perp.")
                approved_tickers.append({
                    "ticker": ticker, 
                    "regime": "LONG_GAMMA",
                    "spot_price": spot,
                    "vrp": vrp,
                    "hurst": hurst,
                    "leverage": WATCHLIST[ticker].get("max_leverage", 10),
                    "est_option_cost": spot * 0.02
                })
                
            elif vrp > 10.0 and ivr > 75.0:
                if OPTIONS_TRADING_LEVEL >= 3:
                    print(f"  🔥 REGIME D (SHORT VOL): {ticker} IV is massively inflated (VRP: {vrp:.2f}%).")
                    
                    if hurst > 0.5:
                        print(f"  ⚠️ WARNING: Hurst is > 0.5 (Trending). Short Strangles are dangerous here!")
                        
                    print(f"  🧠 Engaging Strike Selector to find Optimal 16-Delta Strangle...")
                    
                    chain_data = self.webull.scan_option_chain(ticker)
                    strangle = self.strike_selector.find_optimal_strangle(
                        ticker=ticker,
                        spot_price=spot,
                        chain_data=chain_data,
                        target_dte=30,
                        target_delta=0.16,
                        fallback_iv=iv / 100.0
                    )
                    
                    if strangle:
                        call_sym = strangle['call']['symbol']
                        call_strike = strangle['call']['strike']
                        put_sym = strangle['put']['symbol']
                        put_strike = strangle['put']['strike']

                        print(f"\n  🏆 [LEVEL 3 CANDIDATE] {ticker} 16Δ strikes: "
                              f"call {call_sym} (${call_strike}) / put {put_sym} (${put_strike})")
                        # NOTE: this scanner is APPROVAL-ONLY. It no longer places naked
                        # strangle orders. Defined-risk execution + lifecycle management
                        # is being rebuilt in the L3 engine (see PremiumCollectionEngine).
                        strangle_meta = {"call": call_sym, "call_strike": call_strike,
                                         "put": put_sym, "put_strike": put_strike,
                                         "expiry": strangle.get("expiry"), "dte": strangle.get("dte")}
                    else:
                        strangle_meta = None

                    approved_tickers.append({
                        "ticker": ticker,
                        "regime": "SHORT_VOL",
                        "spot_price": spot,
                        "vrp": vrp,
                        "ivr": ivr,
                        "hurst": hurst,
                        "leverage": WATCHLIST[ticker].get("max_leverage", 10),
                        "est_option_cost": spot * 0.05,
                        "strangle": strangle_meta,
                        "tradeable": strangle_meta is not None and hurst <= 0.5,
                    })
                else:
                    print(f"  ❌ REGIME D (SHORT VOL): {ticker} IV is massively inflated.")
                    print(f"     Level 2 Restriction: CANNOT Sell Premium. SKIPPING trade.")
                
            else:
                print(f"  ➖ REGIME C (NEUTRAL): {ticker} No mathematical edge. SKIPPING.")
                
        return approved_tickers

if __name__ == "__main__":
    scanner = MacroScanner()
    scanner.scan_watchlist()