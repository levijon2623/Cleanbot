import math

class DirectionalSniper:
    """
    Translates underlying stock targets (from the Data Lake Simulator) 
    into exact Option Limit Prices using Gamma Convexity math.
    """
    
    @staticmethod
    def calculate_option_brackets(entry_ask_price: float, delta: float, gamma: float, 
                                  target_move_up: float, stop_move_down: float, is_call: bool):
        """
        Projects the expected price of the option at your target and stop levels.
        Formula: ΔP_opt = (Delta * ΔS) + (0.5 * Gamma * (ΔS)^2)
        """
        # Ensure Gamma is positive for long options
        gamma = abs(gamma)
        
        if is_call:
            # WIN Scenario: Stock goes up
            profit_at_target = (delta * target_move_up) + (0.5 * gamma * (target_move_up ** 2))
            expected_tp_price = entry_ask_price + profit_at_target
            
            # LOSS Scenario: Stock goes down (Delta is positive, move is negative)
            loss_at_stop = (delta * -stop_move_down) + (0.5 * gamma * (-stop_move_down ** 2))
            expected_sl_price = entry_ask_price + loss_at_stop
            
        else: # PUT OPTION
            # Put deltas are negative. 
            # WIN Scenario: Stock goes down
            profit_at_target = (delta * -target_move_up) + (0.5 * gamma * (-target_move_up ** 2))
            expected_tp_price = entry_ask_price + profit_at_target
            
            # LOSS Scenario: Stock goes up
            loss_at_stop = (delta * stop_move_down) + (0.5 * gamma * (stop_move_down ** 2))
            expected_sl_price = entry_ask_price + loss_at_stop

        # Round to nearest valid options tick (usually $0.01 or $0.05)
        return {
            "take_profit_limit": round(expected_tp_price, 2),
            "stop_loss_limit": round(expected_sl_price, 2)
        }

    @staticmethod
    def get_marketable_limit_exit(current_bid: float, current_ask: float, is_taking_profit: bool):
        """
        Brokers reject MARKET orders on options.
        To exit instantly when the stock hits our target, we use a "Marketable Limit".
        We price it aggressively through the spread to guarantee an instant fill.
        """
        spread = current_ask - current_bid
        
        if is_taking_profit:
            # We are happy to give up a tiny bit of the spread to secure the win instantly
            aggressive_sell_price = current_bid - (spread * 0.5)
        else:
            # STOP LOSS: Panic out. Smash the bid aggressively.
            aggressive_sell_price = current_bid - (spread * 1.5)
            
        return max(0.01, round(aggressive_sell_price, 2))

# =====================================================================
# EXAMPLE: Translating your TSLA Simulator Results to Live Execution
# =====================================================================
if __name__ == "__main__":
    # Your TSLA Data Lake Results:
    TSLA_OPTIMAL_TARGET = 1.50
    TSLA_OPTIMAL_STOP = 0.75
    
    # Live Webull Data for an ATM Call:
    option_ask = 3.50
    option_delta = 0.52
    option_gamma = 0.045
    
    print("\n" + "="*50)
    print("🎯 TSLA DIRECTIONAL SNIPER BRACKETS")
    print("="*50)
    
    brackets = DirectionalSniper.calculate_option_brackets(
        entry_ask_price=option_ask,
        delta=option_delta,
        gamma=option_gamma,
        target_move_up=TSLA_OPTIMAL_TARGET,
        stop_move_down=TSLA_OPTIMAL_STOP,
        is_call=True
    )
    
    print(f"Entry Price (Ask):  ${option_ask:.2f}")
    print(f"Take Profit Level:  ${brackets['take_profit_limit']:.2f}")
    print(f"Stop Loss Level:    ${brackets['stop_loss_limit']:.2f}")
    
    # ---------------------------------------------------------
    # Fast Forward 10 minutes: TSLA just hit +$1.50! 
    # ---------------------------------------------------------
    print("\n🚀 TARGET HIT! Flattening position...")
    live_bid = 4.35
    live_ask = 4.45
    
    # Calculate the aggressive limit to guarantee instant fill
    exit_limit = DirectionalSniper.get_marketable_limit_exit(live_bid, live_ask, is_taking_profit=True)
    
    print(f"Live Option NBBO:   ${live_bid:.2f} / ${live_ask:.2f}")
    print(f"Routing Limit Order to Sell @ ${exit_limit:.2f} to guarantee instant fill.")