import time
from datetime import datetime
from webull_gamma_client import WebullGammaClient

class PremiumCollectionEngine:
    """
    Manages the lifecycle of Level 3 Short Premium trades (Strangles, Straddles, Short Puts).
    Enforces mechanical Take Profits (50%), Stop Losses (200%), and Time Stops (21 DTE).
    """
    def __init__(self, webull_client: WebullGammaClient, dry_run: bool = True):
        self.webull = webull_client
        self.DRY_RUN = dry_run
        
        # We store the state to push to the React UI
        self.ui_payload = []

    def fetch_short_positions(self):
        """Sweeps the Webull portfolio specifically for negative-quantity options."""
        short_legs = []
        try:
            trade_client = self.webull.trade_client
            if hasattr(trade_client, 'account_v2') and hasattr(trade_client.account_v2, 'get_account_positions'):
                resp = trade_client.account_v2.get_account_positions(self.webull.account_id)
            elif hasattr(trade_client.account, 'get_positions'):
                resp = trade_client.account.get_positions(self.webull.account_id)
            elif hasattr(trade_client.account, 'get_account_positions'):
                resp = trade_client.account.get_account_positions(self.webull.account_id)
            else:
                return []

            pos_data = resp.json() if hasattr(resp, 'json') else resp
            if isinstance(pos_data, dict) and 'data' in pos_data:
                pos_data = pos_data['data']

            if not isinstance(pos_data, list): 
                return []

            for pos in pos_data:
                asset_type = pos.get('assetType', pos.get('instrumentType', ''))
                qty = float(pos.get('position', pos.get('quantity', 0)))
                
                if asset_type == "OPTION" and qty < 0:
                    short_legs.append(pos)
                    
            return short_legs
            
        except Exception as e:
            print(f"🚨 [L3 ENGINE] Failed to fetch positions: {e}")
            return []

    def group_into_strategies(self, short_legs: list):
        """
        Groups individual short legs into defined strategies.
        If a ticker has 1 short Call and 1 short Put for the same expiry, it's a Strangle/Straddle.
        """
        strategies = {}
        
        for leg in short_legs:
            ticker = leg.get('ticker', leg.get('symbol', 'UNKNOWN')).split()[0] # Grab underlying
            occ_symbol = leg.get('symbol', '')
            qty = abs(float(leg.get('position', leg.get('quantity', 0))))
            avg_price = float(leg.get('costPrice', leg.get('avgPrice', 0)))
            live_mark = float(leg.get('lastPrice', leg.get('marketValue', 0)) / qty / 100) if qty else 0
            
            # Extract DTE from OCC Symbol (e.g., SPY260821C00550000)
            try:
                date_str = occ_symbol[len(ticker):len(ticker)+6]
                expiry_date = datetime.strptime(date_str, "%y%m%d")
                dte = (expiry_date - datetime.now()).days
            except Exception:
                dte = 0

            is_call = "C" in occ_symbol[len(ticker)+6:]

            if ticker not in strategies:
                strategies[ticker] = {"legs": [], "total_premium_sold": 0.0, "total_live_mark": 0.0, "dte": dte}
                
            strategies[ticker]["legs"].append({
                "occ": occ_symbol,
                "is_call": is_call,
                "qty": qty,
                "avg_price": avg_price,
                "live_mark": live_mark
            })
            
            strategies[ticker]["total_premium_sold"] += avg_price
            strategies[ticker]["total_live_mark"] += live_mark
            
            # Sync to the shortest DTE if multiple legs exist
            strategies[ticker]["dte"] = min(strategies[ticker]["dte"], dte)

        # Label the strategies
        for ticker, data in strategies.items():
            legs = data["legs"]
            if len(legs) == 2:
                if legs[0]["is_call"] != legs[1]["is_call"]:
                    data["strategy_name"] = "Short Strangle"
                else:
                    data["strategy_name"] = "Ratio Spread"
            elif len(legs) == 1:
                data["strategy_name"] = "Naked Call" if legs[0]["is_call"] else "Cash-Secured Put"
            else:
                data["strategy_name"] = "Complex Multi-Leg"

        return strategies

    def enforce_lifecycle_rules(self, strategies: dict):
        """
        Applies the quantitative risk-management rules to the grouped strategies.
        """
        self.ui_payload = []
        
        for ticker, data in strategies.items():
            premium_sold = data["total_premium_sold"]
            live_mark = data["total_live_mark"]
            dte = data["dte"]
            
            if premium_sold == 0: continue
            
            # ROE on a Short Premium trade: (Credit Received - Current Cost to Close) / Credit Received
            profit_loss = premium_sold - live_mark
            roe_pct = (profit_loss / premium_sold) * 100
            
            # --- BUILD UI PAYLOAD ---
            self.ui_payload.append({
                "ticker": ticker,
                "strategy": data["strategy_name"],
                "premiumSold": premium_sold,
                "liveMark": live_mark,
                "roe": round(roe_pct, 2),
                "dte": dte
            })
            
            # --- RULE 1: THE 50% TAKE PROFIT ---
            if live_mark <= (premium_sold * 0.50):
                self._execute_flatten(ticker, data, f"✅ 50% Max Profit Achieved (+{roe_pct:.1f}% ROE)")
                
            # --- RULE 2: THE 200% STOP LOSS (Value tripled = -200% ROE) ---
            elif live_mark >= (premium_sold * 3.0):
                self._execute_flatten(ticker, data, f"🚨 200% STOP LOSS BREACHED (-{abs(roe_pct):.1f}% ROE). Capital Preservation Override.")
                
            # --- RULE 3: THE 21 DTE GAMMA TRAP ---
            elif dte <= 21:
                self._execute_flatten(ticker, data, f"⏱️ 21 DTE TIME STOP. Flattening to escape Gamma Risk.")

    def _execute_flatten(self, ticker: str, data: dict, reason: str):
        """Buys to close all legs of the short strategy."""
        print("\n" + "⚓" * 30)
        print(f"L3 PREMIUM ENGINE: LIQUIDATING {ticker} ({data['strategy_name']})")
        print(f"Reason: {reason}")
        print("⚓" * 30)
        
        if self.DRY_RUN:
            print("  🟢 [DRY RUN] Webull Buy-to-Close orders bypassed.")
            return

        for leg in data["legs"]:
            occ = leg["occ"]
            qty = leg["qty"]
            
            # We add a 10% aggressive markup to the limit to guarantee instant fill
            limit_price = max(0.01, round(leg["live_mark"] * 1.10, 2))
            
            print(f"  ⚡ Routing Limit BUY to CLOSE {qty}x {occ} @ ${limit_price}")
            self.webull.place_option_order(
                ticker=ticker,
                option_id=occ,
                action="BUY",
                quantity=int(qty),
                is_closing=True,
                order_type="LIMIT",
                limit_price=limit_price
            )

    def run_cycle(self):
        """Main method to be called periodically by the Bot Runner."""
        short_legs = self.fetch_short_positions()
        if short_legs:
            strategies = self.group_into_strategies(short_legs)
            self.enforce_lifecycle_rules(strategies)
        else:
            self.ui_payload = []

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    webull = WebullGammaClient(
        app_key=os.getenv("WEBULL_APP_KEY"),
        app_secret=os.getenv("WEBULL_APP_SECRET"),
        account_id=os.getenv("WEBULL_ACCOUNT_ID"),
        paper_trading=False
    )
    
    l3_engine = PremiumCollectionEngine(webull, dry_run=True)
    l3_engine.run_cycle()
    print("\nSimulated UI Payload Output:")
    print(l3_engine.ui_payload)