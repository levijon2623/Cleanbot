import time
from datetime import datetime
from local_black_scholes_engine import LocalBlackScholesEngine

class StrikeSelector:
    """
    A quantitative tool to mathematically hunt down the optimal option strikes 
    for Volatility Arbitrage (Level 3 Short Premium).
    """
    def __init__(self, bs_engine: LocalBlackScholesEngine):
        self.bs_engine = bs_engine

    def find_optimal_strangle(self, ticker: str, spot_price: float, chain_data: dict, target_dte: int = 30, target_delta: float = 0.16, fallback_iv: float = 0.50):
        """
        Finds the Call and Put closest to the target Delta (default: 16 Delta) 
        at the expiration date closest to the target DTE.
        """
        print(f"🎯 Hunting for {target_delta * 100} Delta Strangle strikes for {ticker} (~{target_dte} DTE)...")
        
        contracts = chain_data.get("data", [])
        if not contracts:
            print("  ❌ Option chain is empty.")
            return None

        # 1. Identify all unique expiration dates in the chain
        current_time = time.time()
        unique_expiries = set()
        
        for c in contracts:
            exp_str = c.get("expireDate", "0000-00-00")
            if exp_str == "0000-00-00": continue
            
            try:
                exp_ts = datetime.strptime(f"{exp_str} 16:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
                days_to_exp = (exp_ts - current_time) / 86400.0
                if days_to_exp > 0:
                    unique_expiries.add((exp_str, days_to_exp))
            except Exception:
                continue

        if not unique_expiries:
            return None

        # 2. Find the Expiration Date closest to our Target DTE
        # E.g., if we want 30 DTE, it finds the weekly/monthly closest to 30 days.
        best_expiry = min(unique_expiries, key=lambda x: abs(x[1] - target_dte))
        target_exp_str, actual_dte = best_expiry
        
        print(f"  ├─ Selected Expiration: {target_exp_str} ({actual_dte:.1f} Days Out)")

        # 3. Filter contracts to only this expiration
        target_contracts = [c for c in contracts if c.get("expireDate") == target_exp_str]
        
        best_call = None
        best_put = None
        min_call_diff = float('inf')
        min_put_diff = float('inf')
        
        years_to_expiry = actual_dte / 365.25

        # 4. Run Black-Scholes on every contract to find the closest Delta matches
        for c in target_contracts:
            is_call = c.get("callPut") == "Call"
            strike = float(c.get("strikePrice", 0))
            raw_iv = float(c.get("impliedVolatility", 0))
            
            # Normalize IV for the math engine
            iv = raw_iv if raw_iv < 1.0 else raw_iv / 100.0
            
            if iv <= 0:
                if fallback_iv and fallback_iv > 0:
                    iv = fallback_iv
                else:
                    continue # Skip broken/zero-bid contracts

            greeks = self.bs_engine._calculate_contract_greeks(
                spot_price=spot_price,
                strike=strike,
                time_to_expiry_years=years_to_expiry,
                iv=iv,
                is_call=is_call
            )
            
            delta = greeks["delta"]
            
            # We want the absolute distance from our target. 
            # (Puts have negative delta, so we check distance from -0.16)
            if is_call:
                diff = abs(delta - target_delta)
                if diff < min_call_diff:
                    min_call_diff = diff
                    best_call = {
                        "symbol": c.get("symbol"),
                        "strike": strike,
                        "delta": delta,
                        "iv": iv * 100
                    }
            else:
                diff = abs(delta - (-target_delta))
                if diff < min_put_diff:
                    min_put_diff = diff
                    best_put = {
                        "symbol": c.get("symbol"),
                        "strike": strike,
                        "delta": delta,
                        "iv": iv * 100
                    }

        if best_call and best_put:
            print(f"  ├─ Found Call: {best_call['symbol']} (Strike: ${best_call['strike']}, Delta: {best_call['delta']:.2f})")
            print(f"  └─ Found Put : {best_put['symbol']} (Strike: ${best_put['strike']}, Delta: {best_put['delta']:.2f})")
            return {"call": best_call, "put": best_put, "expiry": target_exp_str, "dte": actual_dte}
        else:
            print("  ❌ Failed to find valid options for a strangle.")
            return None