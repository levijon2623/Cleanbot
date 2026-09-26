import os
import time
from dotenv import load_dotenv
from unusual_whales_client import UnusualWhalesClient

# The tickers you want to monitor
TICKERS = ["NVDA", "AAPL", "MSFT", "META", "AMZN", "SPY", "GOOGL"]

def run_diagnostic():
    load_dotenv()
    api_key = os.getenv("UW_API_KEY")
    
    if not api_key:
        print("🚨 Error: UW_API_KEY not found in .env file.")
        return
        
    print("🌊 Igniting Unusual Whales Flow Diagnostic Tool...")
    uw = UnusualWhalesClient(api_key)
    uw.start_multiplexer(TICKERS)
    
    print("⏳ Allowing 5 seconds for WebSockets to populate data caches...")
    time.sleep(5)
    
    print("\n" + "="*50)
    print(f"📊 LIVE INSTITUTIONAL NET FLOW (Raw Values)")
    print("="*50)
    
    try:
        while True:
            print("\033[H\033[J", end="") # Clears the terminal screen for a static dashboard effect
            print("="*50)
            print(f"📊 LIVE INSTITUTIONAL NET FLOW | {time.strftime('%H:%M:%S')}")
            print("="*50)
            print(f"{'TICKER':<10} | {'LIVE NET PREMIUM':<15} | {'MOMENTUM / REGIME'}")
            print("-" * 50)
            
            for ticker in TICKERS:
                # 1. Fetch live flow
                flow_data = uw.get_live_net_premium(ticker)
                if not flow_data:
                    print(f"{ticker:<10} | {'Awaiting Ticks...':<15} | ")
                    continue
                    
                latest_flow = float(flow_data[-1].get("net_premium", 0))
                
                # 2. Fetch Regime
                regime_data = uw.get_gex_regime(ticker)
                regime = regime_data.get("regime", "UNKNOWN") if regime_data else "UNKNOWN"
                
                # 3. Formatting
                flow_str = f"${latest_flow:,.0f}"
                if latest_flow > 0:
                    flow_str = f"\033[92m{flow_str}\033[0m" # Green
                elif latest_flow < 0:
                    flow_str = f"\033[91m{flow_str}\033[0m" # Red
                    
                print(f"{ticker:<10} | {flow_str:<24} | {regime}")
                
            print("\n(Press Ctrl+C to exit)")
            time.sleep(1) # Refresh dashboard every 1 second
            
    except KeyboardInterrupt:
        print("\n🛑 Diagnostic Tool Terminated.")

if __name__ == "__main__":
    run_diagnostic()