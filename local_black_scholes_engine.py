import math
import time
from datetime import datetime

def norm_cdf(x: float) -> float:
    """
    Cumulative Distribution Function (CDF) for the standard normal distribution.
    We use math.erf (Error Function) built into standard Python to avoid needing 
    to install large dependencies like SciPy just for a few statistical formulas.
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x: float) -> float:
    """
    Probability Density Function (PDF) for the standard normal distribution.
    Used for calculating Gamma.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

class LocalBlackScholesEngine:
    """
    An in-memory Greek calculation engine. 
    It maintains a ledger of your active option positions and updates their 
    combined Delta, Gamma, and Theta tick-by-tick based on a live price feed, 
    completely bypassing broker REST API limits.
    """
    def __init__(self, risk_free_rate: float = 0.045):
        # The risk-free interest rate (e.g., 0.045 = 4.5% Treasury yield)
        self.r = risk_free_rate
        
        # Format: {"SPY": [{"symbol": "SPY26...", "strike": 550, "is_call": True, "qty": 5, "iv": 0.18, "expiry_ts": 172...}, ...]}
        self.positions = {}

    def add_position(self, underlying_ticker: str, symbol: str, strike: float, is_call: bool, 
                     quantity: int, implied_volatility: float, expiry_timestamp: float):
        """
        Registers an active option contract from Webull into the local memory ledger.
        """
        if underlying_ticker not in self.positions:
            self.positions[underlying_ticker] = []
            
        self.positions[underlying_ticker].append({
            "symbol": symbol,
            "strike": float(strike),
            "is_call": is_call,
            "qty": int(quantity),
            "iv": float(implied_volatility),
            "expiry_ts": float(expiry_timestamp)
        })
        print(f"📥 Loaded into local memory: {underlying_ticker} {'Call' if is_call else 'Put'} @ ${strike} (Qty: {quantity})")

    def clear_positions(self, underlying_ticker: str = None):
        """Wipes the ledger. Used when you close out positions on Webull."""
        if underlying_ticker:
            self.positions[underlying_ticker] = []
        else:
            self.positions.clear()

    def _calculate_contract_greeks(self, spot_price: float, strike: float, 
                                   time_to_expiry_years: float, iv: float, is_call: bool):
        """
        The core Black-Scholes-Merton mathematical formula.
        Returns the theoretical Delta, Gamma, and Theta for a SINGLE contract.
        """
        # Failsafe: Avoid division by zero if option expires today
        T = max(time_to_expiry_years, 0.0001)
        
        S = spot_price
        K = strike
        sigma = max(iv, 0.01) # Avoid 0 IV
        
        # d1 and d2 are the core probability factors in Black-Scholes
        d1 = (math.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        # Gamma is identical for both Calls and Puts
        gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))

        if is_call:
            delta = norm_cdf(d1)
            # Theta is traditionally displayed as daily decay (divided by 365)
            theta = (- (S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) 
                     - self.r * K * math.exp(-self.r * T) * norm_cdf(d2)) / 365.0
        else:
            delta = norm_cdf(d1) - 1.0
            theta = (- (S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) 
                     + self.r * K * math.exp(-self.r * T) * norm_cdf(-d2)) / 365.0

        return {"delta": delta, "gamma": gamma, "theta": theta}

    def get_live_portfolio_greeks(self, underlying_ticker: str, current_spot_price: float) -> dict:
        """
        Iterates over all owned contracts for a given ticker, calculates their live Greeks 
        based on the millisecond's Spot Price, and aggregates them into Webull-equivalent "Share Delta".
        """
        if underlying_ticker not in self.positions or not self.positions[underlying_ticker]:
            return {"net_delta": 0.0, "net_gamma": 0.0, "net_theta": 0.0}

        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        
        current_time = time.time()

        for pos in self.positions[underlying_ticker]:
            # Convert seconds until expiry into Years (required by Black-Scholes)
            seconds_to_expiry = pos["expiry_ts"] - current_time
            years_to_expiry = seconds_to_expiry / (365.25 * 24 * 3600)
            
            # If the option has expired, ignore it
            if years_to_expiry <= 0:
                continue
                
            # Run the math
            greeks = self._calculate_contract_greeks(
                spot_price=current_spot_price,
                strike=pos["strike"],
                time_to_expiry_years=years_to_expiry,
                iv=pos["iv"],
                is_call=pos["is_call"]
            )
            
            # MULTIPLIER: 1 Option Contract = 100 Shares
            # If Delta is 0.50 and we own 2 contracts, our Net Delta is 100 shares.
            position_multiplier = pos["qty"] * 100
            
            net_delta += greeks["delta"] * position_multiplier
            net_gamma += greeks["gamma"] * position_multiplier
            net_theta += greeks["theta"] * position_multiplier

        return {
            "net_delta": round(net_delta, 4),
            "net_gamma": round(net_gamma, 4),
            "net_theta": round(net_theta, 4)
        }

if __name__ == "__main__":
    print("🧠 Starting Local Black-Scholes Engine Test...\n")
    
    engine = LocalBlackScholesEngine(risk_free_rate=0.045)
    ticker = "SPY"
    
    # 1. Simulate the ONE-TIME Webull fetch at startup
    # Let's say Webull tells us we own 2 SPY Calls and 1 SPY Put
    # Expiry is exactly 14 days from now
    fourteen_days_out = time.time() + (14 * 24 * 3600)
    
    engine.add_position(ticker, strike=550.0, is_call=True, quantity=2, implied_volatility=0.15, expiry_timestamp=fourteen_days_out)
    engine.add_position(ticker, strike=540.0, is_call=False, quantity=-1, implied_volatility=0.18, expiry_timestamp=fourteen_days_out)
    
    # 2. Simulate the high-frequency Hyperliquid WebSocket stream
    # Notice how we can query the Greeks infinitely without making a single REST API call!
    print("\n🌊 Simulating live WebSocket spot price stream...")
    
    mock_price_stream = [545.00, 545.50, 546.25, 548.00, 550.50]
    
    for spot in mock_price_stream:
        live_greeks = engine.get_live_portfolio_greeks(ticker, current_spot_price=spot)
        print(f"Live Spot: ${spot:.2f} | Net Delta: {live_greeks['net_delta']:>7.2f} shares | Gamma: {live_greeks['net_gamma']:>6.2f} | Theta: ${live_greeks['net_theta']:>6.2f}")
        time.sleep(0.5)