import logging

class CapitalAllocator:
    """
    Acts as the Chief Risk Officer (CRO). 
    Allocates capital safely across both Webull (Options Premium) and 
    Hyperliquid (Delta Hedge Buffers), distinguishing between Hedged and Unhedged plays.
    """
    def __init__(self, available_webull_cash: float, available_hl_cash: float, total_equity: float = None):
        self.webull_cash = available_webull_cash
        self.hl_cash = available_hl_cash
        # Equity basis for the "low cash reserve" gate. Defaults to starting
        # Webull cash rather than a hardcoded number so the gate scales with
        # the real account instead of silently assuming ~$2,000.
        self.total_equity = float(total_equity) if total_equity and total_equity > 0 else float(available_webull_cash)
        self.allocated_trades = []

    def release_trade(self, webull_cost: float, hl_cost: float = 0.0):
        """Return capital to the pool when a position is closed."""
        self.webull_cash += max(0.0, float(webull_cost or 0.0))
        self.hl_cash += max(0.0, float(hl_cost or 0.0))
        
    def calculate_hyperliquid_cost(self, spot_price: float, leverage: int, drawdown_buffer_pct: float = 0.10) -> float:
        """
        Calculates the safe capital allocation for a Gamma Scalp on a single options contract.
        Assuming 1 contract = 100 shares max delta exposure.
        """
        max_notional_exposure = spot_price * 100.0
        initial_margin = max_notional_exposure / float(leverage)
        drawdown_buffer = max_notional_exposure * drawdown_buffer_pct
        
        return initial_margin + drawdown_buffer

    def score_opportunities(self, opportunities: list) -> list:
        """
        Assigns an 'Alpha Score' to batch trades to rank them from best to worst.
        """
        scored_list = []
        
        for trade in opportunities:
            strategy_type = trade.get("strategy_type", "GAMMA_SCALP")
            regime = trade.get("regime")
            vrp = trade.get("vrp", 0)     
            hurst = trade.get("hurst", 0.5)
            
            if strategy_type == "DIRECTIONAL":
                # Directional Flow trades are tactical. Score is based on momentum strength.
                alpha_score = trade.get("momentum_score", 50.0)
                
            elif regime == "LONG_GAMMA":
                alpha_score = abs(vrp)
                
            elif regime == "SHORT_VOL":
                mean_reversion_weight = max(0.1, 1.0 - hurst)
                alpha_score = vrp * mean_reversion_weight
                
            else:
                alpha_score = 0
                
            trade["alpha_score"] = alpha_score
            scored_list.append(trade)
            
        return sorted(scored_list, key=lambda x: x["alpha_score"], reverse=True)

    def allocate_portfolio(self, scanned_opportunities: list) -> list:
        """
        [MORNING BATCH PROCESSOR]
        Iterates through a ranked list and funds trades until capital is exhausted.
        """
        print("\n" + "="*60)
        print("💼 PORTFOLIO ALLOCATOR: RANKING & FUNDING TRADES")
        print("="*60)
        print(f"💰 Available Webull Cash: ${self.webull_cash:,.2f}")
        print(f"💰 Available HL Buffer:   ${self.hl_cash:,.2f}\n")
        
        if not scanned_opportunities:
            print("  No opportunities passed the Macro Scanner.")
            return []

        ranked_trades = self.score_opportunities(scanned_opportunities)
        
        for trade in ranked_trades:
            ticker = trade["ticker"]
            strategy_type = trade.get("strategy_type", "GAMMA_SCALP")
            spot = trade.get("spot_price", 0)
            est_option_price = trade.get("est_option_cost", 0) * 100 
            
            # DIRECTIONAL EXCEPTION: No HL Buffer Required
            if strategy_type == "DIRECTIONAL":
                required_hl_cash = 0.0
            else:
                leverage = trade.get("leverage", 10)
                required_hl_cash = self.calculate_hyperliquid_cost(spot, leverage)
            
            print(f"🎯 Evaluating {ticker} ({strategy_type}) | Alpha Score: {trade['alpha_score']:.2f}")
            print(f"   ├─ Required Webull Premium: ${est_option_price:,.2f}")
            print(f"   └─ Required HL Buffer:      ${required_hl_cash:,.2f}")
            
            if self.webull_cash >= est_option_price and self.hl_cash >= required_hl_cash:
                print(f"   ✅ APPROVED: Fully funded. Adding to execution queue.\n")
                self.webull_cash -= est_option_price
                self.hl_cash -= required_hl_cash
                self.allocated_trades.append(trade)
            else:
                print(f"   ❌ REJECTED: Insufficient Capital. Skipping.\n")
                
        print(f"📊 Allocation Complete. Selected {len(self.allocated_trades)} trades.")
        print(f"💵 Remaining Webull Cash: ${self.webull_cash:,.2f}")
        print(f"💵 Remaining HL Buffer:   ${self.hl_cash:,.2f}")
        print("="*60 + "\n")
        
        return self.allocated_trades

    def allocate_live_trade(self, ticker: str, strategy_type: str, exact_option_cost: float, spot_price: float = 0.0, leverage: int = 10, alpha_score: float = 50.0) -> bool:
        """
        [REAL-TIME PROCESSOR]
        Used by the Directional Bot Runner to ask for permission on the fly 
        when an intraday momentum signal triggers.
        """
        premium_cost = exact_option_cost * 100.0 # Convert to 100 shares
        required_hl_cash = 0.0
        
        # --- DYNAMIC ALPHA THRESHOLDING (Capital Preservation) ---
        # Equity basis is injected at construction (real account size), not hardcoded.
        estimated_total_equity = self.total_equity if self.total_equity > 0 else max(self.webull_cash, 1.0)
        cash_ratio = self.webull_cash / estimated_total_equity
        
        # If we have less than 30% of our cash remaining, we ONLY take A+ setups (Score > 80)
        if cash_ratio < 0.30 and alpha_score < 80.0:
            print(f"  💼 [CRO CHECK] {ticker} ({strategy_type}) Score: {alpha_score:.1f}")
            print(f"     └─ ❌ DENIED: Low Cash Reserve ({cash_ratio*100:.1f}%). Reserving capital for A+ setups only.\n")
            return False

        if strategy_type == "GAMMA_SCALP":
            required_hl_cash = self.calculate_hyperliquid_cost(spot_price, leverage)
            
        print(f"  💼 [CRO CHECK] Requesting capital for {strategy_type} on {ticker} (Score: {alpha_score:.1f})...")
        print(f"     ├─ Webull Required: ${premium_cost:.2f} (Available: ${self.webull_cash:.2f})")
        
        if strategy_type == "GAMMA_SCALP":
            print(f"     ├─ HL Required:     ${required_hl_cash:.2f} (Available: ${self.hl_cash:.2f})")
            
        if self.webull_cash >= premium_cost and self.hl_cash >= required_hl_cash:
            self.webull_cash -= premium_cost
            self.hl_cash -= required_hl_cash
            print(f"     └─ ✅ APPROVED. Executing trade.\n")
            return True
        else:
            print(f"     └─ ❌ DENIED. Insufficient buying power.\n")
            return False

# --- LOCAL DIAGNOSTIC TEST ---
if __name__ == "__main__":
    # Simulate a user who has $1,000 on Webull and $1,000 on Hyperliquid
    manager = CapitalAllocator(available_webull_cash=1000.00, available_hl_cash=1000.00)
    
    # 1. Test Morning Batch Allocation
    mock_scanner_output = [
        {"ticker": "NVDA", "strategy_type": "GAMMA_SCALP", "regime": "LONG_GAMMA", "vrp": -12.5, "spot_price": 115.0, "est_option_cost": 4.50, "leverage": 20},
        {"ticker": "TSLA", "strategy_type": "DIRECTIONAL", "momentum_score": 99.0, "spot_price": 225.0, "est_option_cost": 2.50}
    ]
    manager.allocate_portfolio(mock_scanner_output)
    
    # 2. Test Real-Time Intraday Approval (e.g., A massive flow signal triggers on AAPL at 12:00 PM)
    print("⏰ [12:00 PM] Intraday Flow Divergence Triggered on AAPL!")
    
    # The bot checks the live limit price, sees it costs $1.80 ($180 total), and asks for permission:
    is_approved = manager.allocate_live_trade(
        ticker="AAPL", 
        strategy_type="DIRECTIONAL", 
        exact_option_cost=1.80 
    )