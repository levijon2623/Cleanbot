import math
import os

# =====================================================================
# GLOBAL SAFETY TOGGLES (single source of truth for every engine)
# =====================================================================
# DRY_RUN defaults to True. Nothing places a live order unless the
# environment explicitly sets DRY_RUN=false (or 0 / no).
DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() not in ("false", "0", "no", "off")

# PAPER_TRADING routes the Webull client to the sandbox endpoint.
# Defaults to False (production market data). Set PAPER_TRADING=true for sandbox.
PAPER_TRADING = os.getenv("PAPER_TRADING", "false").strip().lower() in ("true", "1", "yes", "on")

# Discretionary orders fired from flow_viewer. DELIBERATELY INDEPENDENT of
# DRY_RUN: that flag means "the BOT sends nothing", and if manual orders rode on
# it the startup banner would read DRY_RUN = True while real orders left the
# machine -- which is the line you check to confirm you are safe. Default off;
# set MANUAL_TRADING_ARMED=true in the environment to enable. The armed state is
# echoed into live/state.json so the chart and the engine cannot disagree.
# Caps, the price collar and the 15:55 EOD flatten live in manual_orders.py.
MANUAL_TRADING_ARMED = os.getenv("MANUAL_TRADING_ARMED", "false").strip().lower() in ("true", "1", "yes", "on")

# Track the contracts listed in manual_orders.EXTRA_DIRECTIONS -- puts on
# SPY/QQQ/IWM, which no enabled rule asks for -- so they can be traded by hand
# from the viewer. Defaults ON because an empty put side in the strike strip is
# a silent dead end; set MANUAL_CHAIN_EXTRAS=false to drop the ~25 extra MQTT
# subscriptions. Tracking a contract cannot make the bot trade it: direction
# comes from config.RULES via _match_rule, not from what is in the basket.
MANUAL_CHAIN_EXTRAS = os.getenv("MANUAL_CHAIN_EXTRAS", "true").strip().lower() in ("true", "1", "yes", "on")

# =====================================================================
# ACCOUNT & TRADING PERMISSIONS
# =====================================================================
OPTIONS_TRADING_LEVEL = 3
MAX_PORTFOLIO_RISK_PCT = 0.02 # Max 2% risk of total account equity per trade

# --- position sizing: FLAT PREMIUM PARITY ---------------------------------
# Every signal commits the same premium dollars so the paper-trade ledger is a
# clean linear image of the backtest (which equal-weights per-trade ROE):
#   qty = max(1, round(equity * SIZING_TARGET_PREMIUM_PCT / (entry * 100)))
# then trimmed so qty * entry * 100 * stop_roe <= MAX_PORTFOLIO_RISK_PCT * equity
# (but never below 1 -- a single contract is always allowed even if its
# worst-case loss exceeds the 2% budget, so pricey underlyings like TSLA stay in
# the sample), and capped at SIZING_MAX_CONTRACTS for fill realism.
# Sizing basis is the STATIC equity (PAPER_EQUITY in DRY_RUN) -- no compounding,
# so trade size stays constant across the test.
SIZING_TARGET_PREMIUM_PCT = float(os.getenv("SIZING_TARGET_PREMIUM_PCT", "0.02") or 0.02)
SIZING_MAX_CONTRACTS = int(os.getenv("SIZING_MAX_CONTRACTS", "25") or 25)

# --- opening-bell noise guard --------------------------------------------
# No new entries before this ET time. The flow-percentile gate already keeps
# the bot out of the first ~10 min in 99.6% of sessions (check_open_delay: ~2
# early trades in 500 sessions) -- but the live FlowMomentumTracker CAN fire at
# 09:32 on a genuine early flow burst (it did twice in the first live week, both
# losers). This makes the live behaviour match the backtest's effective one and
# declines a trade window the 2yr backtest can't vouch for. delay 5m costs the
# backtest exactly 0 trades. HH:MM ET.
_neb = os.getenv("NO_ENTRY_BEFORE_ET", "09:35").split(":")
NO_ENTRY_BEFORE_MOD = int(_neb[0]) * 60 + int(_neb[1])

# --- VIX-regime sizing overlay ------------------------------------------
# check_regime_state / check_vix_overlay: the book's edge is conditional on
# prior-day VIX. When VIX_prev < its own trailing-VIX_MEDIAN_WINDOW-session
# median, scale every rule's target premium by VIX_SIZE_MULT (adaptive so it
# fires ~half the time in any vol era, not just Q1). Walk-forward OOS: per-
# capital return +16.7% -> ~+19.6%, max drawdown ~-15%, ~78% capital deployed.
# Multiplier not a hard gate -- low-VIX days still make ~+7%, worth trading
# small. Rules whose edge does NOT need equity vol opt out with vix_size:False
# (GLD = gold's own vol regime; AVGO = single name; MSFT CHOP PUT = a chop rule,
# it likes low vol -- walk-forward showed the overlay HURT it).
# Polled once per session (bot_runner._poll_session_vix), fail-open to 1.0x.
# --- Trailing exit ------------------------------------------------------
# check_exit_sweep / check_exit_walkforward (2026-09-08): the deployed bracket
# was target_roe 1.0 / rr 1.0, i.e. stop_roe = 1.0, i.e. the stop sat at
# entry*(1-1.0) = ZERO -- 9 of 11 rules had NO STOP and rode to +100% or the
# 15:55 flatten. (That is also why 1:1 always won the original hand-tuned
# win-rate sort: with no stop, win rate is mechanically maximised and the loss
# tail is unbounded.)
#
# Replacing it with "give back TRAIL_PCT from the running peak, no take-profit":
#   mid fills   OOS +7.2% -> +13.6%, total P&L +52.84 -> +91.66 (6/6 slices)
#   REAL fills  OOS +1.3% -> +4.8%,  total P&L +23.45 -> +33.08
# and it survives a walk-forward of the policy CHOICE (pick on prior slices,
# score on the next: +15.7% chained vs +10.4% deployed, winning 4 of 5 slices).
# The gates were NOT contaminated by the missing stop -- they are worth MORE
# under trailing (+3.2pp vs +1.6pp).
#
# Tighter trails are worse and die on realistic fills (trail 40% -> OOS -0.6%),
# because 1-min option premia are too noisy for a close trail. ATR trailing on
# the option's own true range is worst of all.
#
# Cost: win rate 0.54 -> 0.41 and maxLL 9 -> 17. It wins less often, bigger.
# Per-rule override with "trail_pct"; set 0 to keep the old static bracket.
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.50") or 0.50)

# --- quote-outage guard --------------------------------------------------
# How long an OPEN position may go with no obtainable quote before the bot
# stops waiting and forces an exit on the last known bid.
#
# WHY THIS EXISTS (2026-09-10): the monitor loop used to do
#   bid, ask = get_live_option_quote(...);  if bid == 0.0: continue
# and the client returned (0.0, 0.0) for BOTH "no bid" and "the API call
# failed". So any quote outage silently suspended risk management on every open
# position -- no trail check, no stop, no log. Two SMH 0DTE positions peaked at
# +74% and +144% and both filled at $0.01, because their trail levels ($1.68 and
# $1.03) were never evaluated while the bot was blind. Two bot instances were
# running that morning, so throttling is the likely trigger.
#
# 90s is deliberately short: a 0DTE can go from +144% to worthless inside a few
# minutes, and an exit on a stale book beats riding an unmanaged position to
# expiry. Raise it only if legitimate quote gaps prove longer than this.
QUOTE_BLIND_PANIC_S = float(os.getenv("QUOTE_BLIND_PANIC_S", "90") or 90)

# ---------------------------------------------------------------------
# BAD-TICK GUARD -- a DIFFERENT failure from the blind case above.
# Blind = no quote arrives at all, and QUOTE_BLIND_PANIC_S handles it.
# This is the opposite: a quote DOES arrive and it is garbage. On
# 2026-09-15 four AVGO 1DTE puts exited as "TRAIL STOP ... @ $0.01" --
# one of them 42 seconds after entry, on a $340 put with spot at $339.19
# that had quoted $3.42 moments earlier. An ATM put is not worth a penny.
# A zero bid falls through to `bid <= trail_stop`, fires, and _fire_exit
# floors the limit at max(0.01, ...), booking ~-99.7%.
#
# The guard is CONFIRMATION, not suppression: a genuinely worthless
# option prints a zero bid persistently and still exits, just N ticks
# later; a glitch prints it once and is ignored. Nine ledger records
# across AVGO and SMH -- the two widest-spread rules in the book (6.7%
# and 9.2% median entry spread) -- were corrupted this way.
#
# COLLAPSE also catches the non-zero version: a bid that drops below
# 25% of the last good one in a single tick. MIN_PREV keeps the guard
# off cheap options, where 0.20 -> 0.01 is an ordinary move rather than
# a glitch.
# 🚨 CONFIRM ON ELAPSED TIME, NOT ON TICK COUNT (2026-09-18).
# The original guard counted CONSECUTIVE TICKS, and the monitor loop sleeps
# 0.1s -- so "3 confirmations" demanded that a phantom quote survive THREE
# TENTHS OF A SECOND before being believed. That is not a filter. Worse, the
# quote can be served from cache, in which case three ticks may be the SAME
# bad quote read three times.
# On 2026-09-18 six GLD 0DTE calls exited at $0.01 on trail stops. UW's NBBO
# for those exact timestamps reads 0.67, 0.67, 1.04, 0.96, 0.80, 0.77 -- the
# real bids never collapsed at all, and most were ABOVE their own trail level,
# so the trail should never have fired. The guard engaged correctly and then
# confirmed the glitch in 0.3s.
# Tick counts are a proxy for "has this persisted"; the loop rate is what makes
# the proxy meaningless. Time is the quantity actually meant, so require it
# directly. A genuinely worthless option is still worthless N seconds later, so
# waiting costs nothing real -- whereas believing a phantom books -99%.
BAD_TICK_CONFIRMS = int(os.getenv("BAD_TICK_CONFIRMS", "3") or 3)
BAD_TICK_CONFIRM_S = float(os.getenv("BAD_TICK_CONFIRM_S", "15") or 15)
BAD_TICK_COLLAPSE = float(os.getenv("BAD_TICK_COLLAPSE", "0.25") or 0.25)
BAD_TICK_MIN_PREV = float(os.getenv("BAD_TICK_MIN_PREV", "0.20") or 0.20)

# ---------------------------------------------------------------------
# EXIT CUSHION -- and the penny floor that was doing catastrophic work
# silently.
#
# WHY THIS EXISTS (2026-09-18): eight GLD 0DTE calls, SEVEN of them booked
# at exactly $0.01 for ~-99%, one 51 seconds after entry. It reads exactly
# like the AVGO bad-tick episode, and the bad-tick guard above is NOT what
# failed -- the quotes were very likely fine. _fire_exit computed
#
#     cushion = spread * 1.5;  px = max(0.01, bid - cushion)
#
# and that goes NEGATIVE whenever spread >= (bid - 0.01)/1.5 -- a bid of
# $0.30 with a $0.19 spread, routine for a 0DTE drifting OTM. `max(0.01, …)`
# then turned "I computed a nonsense price" into "dump at a penny" without a
# word in the log. A guard against garbage quotes cannot help when the quote
# is sound and the ARITHMETIC is what breaks.
#
# The 1.5 is also simply the wrong number, and this project already measured
# the right one. sim_core retired the legacy profitability-keyed rule
# (0.5 profitable / 1.5 not) for a TAG-keyed per-ticker cap fitted to NBBO
# ticks: CUSHION_CAP, with FILL_COST=0.03 for exits whose trigger is not
# directionally adverse. GLD's measured cap is 0.40 spreads. Live was
# charging 1.5 -- 3.75x -- on one of the exact wide-spread names sim_core
# flags as being overcharged 2.4-3.8x. The correction was never ported out
# of research. bot_runner now imports those same constants, so there is ONE
# cushion model in the project and it cannot drift (METHODOLOGY 1).
#
# EXIT_FLOOR_FRAC is the backstop that makes the class of bug impossible
# rather than merely unlikely: an exit limit may never be priced below this
# fraction of the prevailing bid, whatever the cushion model returns. If it
# binds, the spread is pathological and the log says so out loud.
EXIT_FLOOR_FRAC = float(os.getenv("EXIT_FLOOR_FRAC", "0.5") or 0.5)

#: Slippage of a marketable SELL, in spreads, per ticker. Fitted in
#: measure_cushion_cap.py on NBBO ticks and previously defined inside sim_core,
#: where the LIVE bot could not reach it -- bot_runner imports no research
#: module (it would drag pandas/polars into the trading loop), so live went on
#: charging the retired 1.5 for months after research had re-measured it. The
#: constants live HERE, in the dependency-free layer both sides already import,
#: so there is exactly one definition. sim_core imports them from config.
#: A MISSING TICKER IS NOT A NEUTRAL DEFAULT -- both callers fall back to the
#: uncapped 1.5 and say so. Run measure_cushion_cap.py before adding one.
CUSHION_CAP = {
    "SPY": 1.50, "QQQ": 1.50, "IWM": 1.27, "NVDA": 1.22, "META": 1.26,
    "AVGO": 0.60, "MSFT": 0.62, "SMH": 0.46, "GLD": 0.40,
    # disabled-rule tickers, measured 2026-09-16 on the same 10 sessions
    "TSLA": 1.50, "AMZN": 0.90, "LULU": 0.63,
}

#: Exits whose TRIGGER is adverse by construction -- they fire because the bid
#: is falling, so it keeps falling while the order travels. Only these pay the
#: cap. `tp` fires on a RISING bid; `eod`/`time`/`sig` have no directional
#: trigger at all.
ADVERSE_TAGS = ("stop", "trail", "give")

#: Measured cost of a marketable sell, in spreads: 18.9M prints (dte<=1,
#: mid>=$0.30, 2026-09-01..09-15) show 86.4% filling at exactly the bid, 11.5%
#: better and 2.1% worse; mean 0.028 after clipping price improvement to zero
#: (we should not bank on being filled better than the bid).
FILL_COST = 0.03

# ---------------------------------------------------------------------
# ENTRY CHASE -- replaces "wait 10s, cancel, abandon the trade".
#
# The entry rests at min(mid + $0.01, ask). On a penny-wide market that
# clamps to the ask and fills instantly, but on anything wider it sits
# BELOW the offer and only fills if a seller comes down to it. Measured
# on the paper ledger against the real NBBO path: only 27% of entries
# would have filled inside the old 10-second window; 93% would have
# eventually. The bot was abandoning roughly three trades in four.
#
# Worse, WHICH ones it caught was adverse. A resting buy fills when the
# option gets CHEAPER -- i.e. when the underlying has moved against the
# position. Waiting passively selects for entries that are already
# losing, which is what the fill-time-vs-P&L split showed.
#
# So instead of waiting once and giving up, step the limit UP toward the
# offer every ENTRY_CHASE_S seconds. Direction matters: for a BUY, the
# ask is above and the bid below, so improving the price means moving
# toward the ASK. Stepping toward the bid would move away from a fill.
#
# CEILING: never pay more than the ask that was quoted WHEN THE SIGNAL
# FIRED (plus MAX_SLIP, default 0). If the offer runs away, the trade is
# abandoned rather than chased -- the thesis was priced off that quote,
# and chasing a moving market is how a sniper turns into a buyer of tops.
# Seconds to wait for a SHUTDOWN flatten to confirm before giving up and
# shouting. The order is left WORKING rather than cancelled: it is a closing
# order, and cancelling it would guarantee the position stays open.
SHUTDOWN_EXIT_S = float(os.getenv("SHUTDOWN_EXIT_S", "20") or 20)

ENTRY_CHASE_S = float(os.getenv("ENTRY_CHASE_S", "10") or 10)
ENTRY_CHASE_ROUNDS = int(os.getenv("ENTRY_CHASE_ROUNDS", "6") or 6)
ENTRY_CHASE_TICK = float(os.getenv("ENTRY_CHASE_TICK", "0.01") or 0.01)
ENTRY_CHASE_MAX_SLIP = float(os.getenv("ENTRY_CHASE_MAX_SLIP", "0") or 0.0)

VIX_SIZE_MULT = float(os.getenv("VIX_SIZE_MULT", "0.5") or 0.5)
VIX_MEDIAN_WINDOW = int(os.getenv("VIX_MEDIAN_WINDOW", "60") or 60)

# =====================================================================
# MASTER WATCHLIST & ASSET-SPECIFIC DNA (THE FINAL SAUCE)
# =====================================================================
WATCHLIST = {
    
    # ---------------------------------------------------------
    # 1. NVDA: The Momentum Rocket
    # ---------------------------------------------------------
    "NVDA": {
        "hl_symbol": "xyz:NVDA",
        "max_leverage": 20,
        # LEGACY GEX-TREE (inert -- USE_RULES=true routes entries through RULES, never
        # this branch). NVDA +GEX CALL RETESTED & KILLED 2026-09-07: POSITIVE_GEX/CALL
        # -6% to -9%/trade every percentile, both halves (NVDA is +GEX ~97% of days so
        # the gate was never a filter); NEGATIVE_GEX/CALL -40% to -50%, win ~0.03.
        # NVDA's live edge is "NVDA LOWVOL PUT" in RULES.
        "POSITIVE_GEX": {
            "CALL": {
                "enabled": False,
                "dte": [1],
                "hours": [10, 11, 12],
                "min_flow": 3_500_000,
                "min_flow_pct": 50,  # --flow-mode pct: trailing-60d percentile gate (see check_flow_drift.py)
                "target_roe": 1.00,  # +100%
                "stop_roe": 0.50     # -50.0%
            },
            "PUT": {
                "enabled": False
            }
        },
        "NEGATIVE_GEX": {
            "CALL": {
                "enabled": False
            },
            "PUT": {
                "enabled": True,   # inert (legacy tree); not retested -- left as-was
                "dte": [0],
                "hours": [9, 10, 11, 12, 13, 14, 15], # All Hours
                "min_flow": 3_500_000,
                "min_flow_pct": 50,  # --flow-mode pct
                "target_roe": 0.60,  # +60%
                "stop_roe": 0.40     # -40.0%
            }
        }
    },
    
    # ---------------------------------------------------------
    # 2. AAPL: The Opening Bell Magnet
    # ---------------------------------------------------------
    "AAPL": {
        "hl_symbol": "xyz:AAPL",
        "max_leverage": 20,
        # LEGACY GEX-TREE (inert -- USE_RULES=true never reads this branch). AAPL +GEX
        # CALL RETESTED & KILLED 2026-09-07: POSITIVE_GEX/CALL -2% to -6% all
        # percentiles/halves. The "-GEX call anomaly" (NEGATIVE_GEX/CALL) was an
        # in-sample artifact: IS +18% to +33% / OOS -34% to -36% -- fully reverses
        # out-of-sample. AAPL has NO RULES entry -> AAPL does not trade at all.
        "POSITIVE_GEX": {
            "CALL": {
                "enabled": False,
                "dte": [1],
                "hours": [9, 10, 11, 12],
                "min_flow": 3_500_000,
                "min_flow_pct": 50,  # --flow-mode pct
                "target_roe": 0.80,  # +80%
                "stop_roe": 0.40     # -40.0%
            },
            "PUT": {
                "enabled": False
            }
        },
        "NEGATIVE_GEX": {
            "CALL": {
                "enabled": False, # "rare -GEX Call Anomaly" -- IS-only artifact, OOS -35% (killed 2026-09-07)
                "dte": [0],
                "hours": [13, 14],
                "min_flow": 3_500_000,
                "min_flow_pct": 50,  # --flow-mode pct
                "target_roe": 0.80,  # +80%
                "stop_roe": 0.27     # -27.0%
            },
            "PUT": {
                "enabled": False
            }
        }
    },
    
    # ---------------------------------------------------------
    # 3. MSFT: The Low-Beta Battleship
    # ---------------------------------------------------------
    "MSFT": {
        "hl_symbol": "xyz:MSFT",
        "max_leverage": 20,
        "POSITIVE_GEX": {
            "CALL": {
                "enabled": True,
                "dte": [0, 1], # DTE Fallback: Searches for 0DTE first, defaults to 1DTE
                "hours": [9, 10, 11],
                "min_flow": 5_000_000,
                "min_flow_pct": 65,  # --flow-mode pct
                "target_roe": 1.00,  # +100%
                "stop_roe": 0.50,    # -50.0%
                "time_stop_mins": 90 # Strict 90-Minute 0DTE hold rule
            },
            "PUT": {
                "enabled": False
            }
        },
        "NEGATIVE_GEX": {
            "CALL": {
                "enabled": False
            },
            "PUT": {
                "enabled": True,
                "dte": [0],
                "hours": [9, 10, 11, 12, 13, 14, 15], # All Hours
                "min_flow": 750_000,
                "min_flow_pct": 50,  # --flow-mode pct ($0.75M was ~19th pctile / near-no-filter)
                "target_roe": 1.00,  # +100%
                "stop_roe": 0.50     # -50.0%
            }
        }
    },

    # ---------------------------------------------------------
    # 4. META: The Mean-Reverting Short
    # ---------------------------------------------------------
    "META": {
        "hl_symbol": "xyz:META",
        "max_leverage": 20,
        "POSITIVE_GEX": {
            "CALL": {
                "enabled": False # Permanently banned from Long-Delta in +GEX
            },
            "PUT": {
                "enabled": True,
                "dte": [3, 7],
                "hours": [11, 13],
                "min_flow": 10_000_000,
                "min_flow_pct": 80,  # --flow-mode pct
                "target_roe": 0.40,  # +40%
                "stop_roe": 0.25     # -25.0%
            }
        },
        "NEGATIVE_GEX": {
            "CALL": {
                "enabled": False
            },
            "PUT": {
                "enabled": False
            }
        }
    },
    
    # ---------------------------------------------------------
    # 5. AMZN: The Weekly Heavyweight
    # ---------------------------------------------------------
    "AMZN": {
        "hl_symbol": "xyz:AMZN",
        "max_leverage": 20,
        "POSITIVE_GEX": {
            "CALL": {
                "enabled": False
            },
            "PUT": {
                "enabled": True,
                "dte": [0],
                "hours": [11, 13],
                "min_flow": 3_500_000,
                "min_flow_pct": 65,  # --flow-mode pct
                "target_roe": 0.60,  # +60%
                "stop_roe": 0.40     # -40.0%
            }
        },
        "NEGATIVE_GEX": {
            "CALL": {
                "enabled": False
            },
            "PUT": {
                "enabled": False
            }
        }
    },

    # ---------------------------------------------------------
    # 6. SPY: The High-Frequency Micro Scalp
    # ---------------------------------------------------------
    "SPY": {
        "hl_symbol": "xyz:SP500",
        "max_leverage": 50,
        "POSITIVE_GEX": {
            "CALL": {
                "enabled": True,
                "dte": [0],
                "hours": [9, 10],
                "min_flow": 7_500_000,
                "min_flow_pct": 50,  # --flow-mode pct ($7.5M was ~28th pctile)
                "target_roe": 0.20,  # +20%
                "stop_roe": 0.10     # -10.0%
            },
            "PUT": {
                "enabled": True,
                "dte": [0],
                "hours": [12, 14],
                "min_flow": 30_000_000,
                "min_flow_pct": 80,  # --flow-mode pct
                "target_roe": 1.00,  # +100%
                "stop_roe": 0.66     # -66.0%
            }
        },
        "NEGATIVE_GEX": {
            "CALL": {
                "enabled": False
            },
            "PUT": {
                "enabled": True,
                "dte": [0],
                "hours": [12, 13, 14],
                "min_flow": 20_000_000,
                "min_flow_pct": 65,  # --flow-mode pct
                "target_roe": 1.00,  # +100%
                "stop_roe": 0.66     # -66.0%
            }
        }
    },
    
    # ---------------------------------------------------------
    # 7. GOOGL: The Chop Monster (Included for Completeness)
    # ---------------------------------------------------------
    "GOOGL": {
        "hl_symbol": "xyz:GOOGL",
        "max_leverage": 20,
        "POSITIVE_GEX": {
            "CALL": {
                "enabled": False # Banned: Route to L3 Short Premium Engine
            },
            "PUT": {
                "enabled": False
            }
        },
        "NEGATIVE_GEX": {
            "CALL": {
                "enabled": False
            },
            "PUT": {
                "enabled": True,
                "dte": [0],
                "hours": [13, 14],
                "min_flow": 1_000_000,
                "min_flow_pct": 50,  # --flow-mode pct
                "target_roe": 0.80,  # +80%
                "stop_roe": 0.40     # -40.0%
            }
        }
    },

    # ---------------------------------------------------------
    # 8. TSLA: re-evaluating (earlier write-off used flawed methods).
    #    All cells OFF for live; backtester grids them via defaults.
    # ---------------------------------------------------------
    "TSLA": {
        "hl_symbol": "xyz:TSLA",   # perp clone unverified; gamma-scalp shelved so unused
        "max_leverage": 20,
        "POSITIVE_GEX": {
            "CALL": {"enabled": False},
            "PUT":  {"enabled": False}
        },
        "NEGATIVE_GEX": {
            "CALL": {"enabled": False},
            "PUT":  {"enabled": False}
        }
    },

    # ---------------------------------------------------------
    # 9. MU: new candidate (regime-mixed: 59% POS / 41% NEG over 2022-26).
    #    All cells OFF for live; backtester grids them via defaults.
    # ---------------------------------------------------------
    "MU": {
        "hl_symbol": "xyz:MU",     # perp clone unverified; gamma-scalp shelved so unused
        "max_leverage": 20,
        "POSITIVE_GEX": {
            "CALL": {"enabled": False},
            "PUT":  {"enabled": False}
        },
        "NEGATIVE_GEX": {
            "CALL": {"enabled": False},
            "PUT":  {"enabled": False}
        }
    },

    # ---------------------------------------------------------
    # 10-12. QQQ / IWM / AVGO: RULES-only candidates. Live edge lives in
    #        RULES (HIVOL CALL for the index ETFs, HIVOL PUT for AVGO),
    #        not the GEX tree -- all cells OFF here.
    # ---------------------------------------------------------
    "QQQ": {
        "hl_symbol": "xyz:NDX100",
        "max_leverage": 50,
        "POSITIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}},
        "NEGATIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}}
    },

    "IWM": {
        "hl_symbol": "xyz:RUSSELL2000",
        "max_leverage": 50,
        "POSITIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}},
        "NEGATIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}}
    },

    "AVGO": {
        "hl_symbol": "xyz:AVGO",
        "max_leverage": 20,
        "POSITIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}},
        "NEGATIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}}
    },

    "GLD": {
        "hl_symbol": "xyz:GOLD",
        "max_leverage": 20,
        "POSITIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}},
        "NEGATIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}}
    },

    # 13. SMH (VanEck Semis ETF): RULES-only. Screened + netprem-validated
    #     2026-09-06 -> "SMH LOWVOL PUT". GEX tree OFF.
    "SMH": {
        "hl_symbol": "xyz:SMH",     # perp clone unverified; gamma-scalp shelved so unused
        "max_leverage": 20,
        "POSITIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}},
        "NEGATIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}}
    },

    # 14. LULU (Lululemon): RULES-only, Friday-weekly-only expiry. Screened +
    #     validated 2026-09-07 -> "LULU HIVOL/UP PUT" (dte [0] => 0DTE Fridays
    #     only). GEX tree OFF.
    "LULU": {
        "hl_symbol": "xyz:LULU",    # perp clone unverified; gamma-scalp shelved so unused
        "max_leverage": 20,
        "POSITIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}},
        "NEGATIVE_GEX": {"CALL": {"enabled": False}, "PUT": {"enabled": False}}
    }
}

# =====================================================================
# RULES  -- regime-conditioned entry recipes (replaces the GEX tree above
# for live trading; WATCHLIST stays for reference / the old backtest path)
# =====================================================================
# USE_RULES routes bot_runner's scanner through RULES instead of the
# WATCHLIST[ticker][POSITIVE_GEX|NEGATIVE_GEX] tree. Default True.
USE_RULES = os.getenv("USE_RULES", "true").strip().lower() not in ("false", "0", "no", "off")

# Path to the static percentile->$ flow-threshold file (built locally from the
# lake by `directional_flow_backtester.py --emit-thresholds`, then deployed to
# the cloud box which has no lake access). Refresh roughly monthly.
FLOW_THRESHOLDS_PATH = os.getenv("FLOW_THRESHOLDS_PATH", "flow_pct_thresholds.json")

# Derived from directional_flow_backtester.py --rule-file walk-forward
# (IS Aug'24-Aug'25 / OOS Aug'25-Aug'26, --flow-mode pct). Each rule that
# survived BOTH halves with a specific amplification gate.
#
# schema (also the format of candidate_rules.json):
#   name, ticker, direction ("CALL"/"PUT"), hours [ints], dte [ints],
#   min_flow_pct (one of 50/65/80/90/95) OR flow_abs ($),
#   target_roe, rr (R:R; stop_roe = target_roe / rr). NOTE rr 1.0 => stop_roe
#       1.0 => the static stop sits at ZERO (no stop). Left as-is because with
#       trail_pct set the trailing exit supersedes both TP and static stop; these
#       two fields still drive the SIZING risk-cap and the spread guard,
#   trail_pct (optional float, default config.TRAIL_PCT = 0.50 -- set 0 to keep
#       the old static TP/SL bracket): exit when the bid gives back this fraction
#       from the position's running peak bid. No take-profit is used while
#       trailing (the sweep found pure trailing beats TP+trail: combo
#       tp150+trail50 was OOS -1.1% on realistic fills vs +4.8% for pure trail50).
#       Peak starts at the entry price, so the effective initial stop is
#       -trail_pct. See the TRAIL_PCT block above for the validation,
#   regime (optional gate: NEGATIVE_GEX/POSITIVE_GEX/UPTREND/DOWNTREND/CHOP/
#           LOWVOL/NORMVOL/HIVOL -- or a LIST of them for an OR-match, e.g.
#           ["HIVOL", "UPTREND"] = fires when EITHER holds (LULU HIVOL/UP PUT)),
#   amp_min (optional: require >= N of {NEG-GEX, LOWVOL, CHOP}),
#   ema_confirm (optional: <tf> minutes -- require the underlying's price EMA
#       stack on tf-min bars to agree with direction, BULL for CALL / BEAR for
#       PUT; helps GLD only, hurts every other rule -- see session-12 sweep),
#   ema_spans (optional: fast->slow EMA spans, default [8, 21, 34]),
#   strike_offset (optional int, DEFAULT 0 = ATM -- N strikes OTM from ATM at
#       entry (CALL -> higher strike, PUT -> lower). check_strike_selection.py:
#       OTM is a per-rule call, not a blanket "OTM on volatile days" -- only
#       GLD amp1 CALL (offset 1) survived a realistic ask-in/bid-out fill test
#       (+7pp OOS both halves). OTM cuts win rate and roughly doubles $0.50-floor
#       rejects. Keep 0 for everything else),
#   vix_size (optional bool, DEFAULT True -- set False to exempt this rule from
#       the VIX-regime sizing overlay (see VIX_SIZE_MULT above). Off for rules
#       whose edge doesn't need equity vol: GLD (gold), AVGO (single name),
#       MSFT CHOP PUT (chop rule -- the overlay HURT it in the walk-forward)),
#   skip_macro_am (optional bool -- suppress this rule entirely on scheduled
#       8:30am-ET macro-release days: CPI / NFP (first Friday) / PCE, from
#       macro_calendar.is_macro_am_day. check_event_days.py: FOMC's 2pm IV crush
#       is real but the bot navigates it fine; the 8:30 releases are the drag,
#       and they specifically turn IWM/QQQ HIVOL CALL into a -65% / win-0.0 rule
#       in the 11:00-14:00 window -- every other rule is fine or better. Refresh
#       macro_calendar.CPI/PCE ~annually),
#   dmi_confirm (optional: {"tf": <min>, "mode": "oppose"|"agree", "period": 14} --
#       require the underlying's OWN Wilder DMI on tf-min bars to point AGAINST
#       ("oppose", a fade) or WITH ("agree", momentum-confirm) the trade
#       direction (+DI>-DI = up). bot_runner reconstructs the tf-min bars live
#       from per-minute spot (H/L from 1-min closes) -- validate close_only.
#       check_adx_dmi.py / check_msft_chop_put.py: the flow trigger is worth most
#       when it CONTRADICTS recent price action -- MSFT CHOP PUT + oppose: IS
#       +12->+40 / OOS +11->+23, win .57->.83, maxLL 26->7, slice coverage kept.
#       Only MSFT validated; every other rule re-introduced the n-halving
#       fragility the session-16 config-walkforward had just removed),
#   eod_flatten (optional: "HH:MM" ET to flatten instead of the 15:55 default --
#       --hold-sweep found time_stop_mins ALWAYS hurts, and only META wants an
#       earlier flatten (15:00, +6pp OOS); everything else keeps 15:55),
#   time_stop_mins (optional -- leave unset; the sweep shows it only ever hurts),
#   dex_pct_max (optional float in [0,1] -- gate: only take the trigger when the
#       ticker's net_dex (call_delta + put_delta from /greek-exposure, PRIOR
#       completed session) sits BELOW this trailing-252-session percentile of its
#       own history. Low dex_pct = dealer directional positioning unusually light
#       -> less stabilising hedge flow -> moves follow through (check_dex_gate.py:
#       QQQ/IWM HIVOL CALL + META LOWVOL PUT each beat a same-size random-subsample
#       bootstrap null, clean of vol-regime endogeneity; helped NO other rule).
#       Polled once at session start (bot_runner._poll_session_dex),
#   flow_zscore (optional -- {"k": float, "window_days": int}. REPLACES min_flow_pct
#       as the live SIZE gate once warmed (min_flow_pct/the static JSON stay as the
#       pre-warm-up fallback): trailing-window_days mean+k*std of this ticker's own
#       DAILY-MEDIAN crossover flow (day-median dedup, same anti-clustering fix as
#       the percentile gate -- see FLOW_HISTORY_MIN_DAYS below) instead of a
#       percentile rank. check_flow_zscore.py gridded k=0.5-3.0 x window=30-90d per
#       rule against the percentile baseline: a parametric mean+k*sigma gate can
#       select a meaningfully different (fat-tailed) trigger set. Only IWM/SPY CHOP
#       CALL showed a clean, non-thin, MONOTONIC-in-k improvement (the signature of
#       a real effect, not a cherry-picked cell) -- most other rules were a wash or
#       actively worse (AMZN afternoon PUT went negative at k>=1.0, not given this),
#   flow_window_days (optional -- overrides the live bot's default 90-calendar-day
#       trailing pool for its OWN self-calibrated min_flow_pct threshold once past
#       FLOW_HISTORY_MIN_DAYS. check_flow_window.py gridded 10-250d per rule: the
#       blended OOS optimum is ~45d, both 60d (the backtest's calibration window,
#       FLOW_PCT_WINDOW_DAYS) and the live default 90d already sit close to that
#       peak so this is usually left unset -- but GLD's own curve peaks sharply at
#       45d (OOS +36.1% vs 60d's +26.6%) and falls off a cliff past 120d),
#   amt_open (optional Auction-Market-Theory gate on where the session OPENED vs
#       the PRIOR day's volume value area: "below_va"|"inside_va"|"above_va" to
#       require exactly, or {"exclude": [...]} / {"require": [...]}. check_amt.py
#       --test rules found this splits several rules hard),
#   vol_overlay (optional target/stop-WIDTH overlay, entry conditions untouched --
#       {bucket: multiplier} on target_roe (stop_roe scales with it, R:R fixed).
#       bucket = the LIVE oi/dir spot-GEX combo at trigger time (spot-exposures
#       gamma_per_one_percent_move_oi vs _dir, sign only): "agree+"/"agree-" both
#       signs agree, "oi+/dir-" = textbook-pinned (lowest realized vol, 9/9
#       tickers IS+OOS), "oi-/dir+" = textbook-unpinned+flow-reinforcing (highest
#       realized vol, 9/9 tickers). Missing bucket -> 1.0 (unchanged). check_dgex.py
#       --test overlay found this only pays off on the two largest-sample rules
#       (IWM/QQQ HIVOL CALL) -- every other rule was flat or an IS-only overfit
#       artifact (worst: SPY POS/PUT went OOS +13.6%->+2.0%, NOT given this field),
#   enabled
#
# regime/flow_pct are computed live from PRIOR-day daily OHLC + a trailing
# window of the ticker's own trigger flow -- bot_runner must poll these at
# session start (see the GEX morning-poll pattern).
RULES = [
    {"name": "META LOWVOL PUT", "ticker": "META", "direction": "PUT",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0, 1], "regime": "LOWVOL", "eod_flatten": "15:00",
     "trail_pct": 0,
     "min_flow_pct": 65, "target_roe": 0.80, "rr": 1.0, "enabled": True},   # trail_pct 0 2026-09-08 (check_trail_regression, realistic fills BOTH sides): static bracket OOS +15.0% vs trail50 +10.0%. MECHANISM, not just the number -- META's winners top out near the +80% target, and a 50% trail gives back half of a peak instead of banking the TP. Keeps the book's most robust rule on the exit it was built for.
    # 15:00 flatten +6pp. amt_open + dex_pct_max REMOVED 2026-09-06: check_config_walkforward showed PLAIN is the single most robust rule in the book (n=608, IS +19 / OOS +22, ALL 6 calendar slices +); the gates cut it to 3 slices / n=139 and made it episodic (dex_pct 0.86 live = rule dark). dex_pct's this-morning bootstrap null was weaker than the rolling walk-forward

    {"name": "MSFT CHOP PUT", "ticker": "MSFT", "direction": "PUT",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0, 1], "regime": "CHOP",
     "dmi_confirm": {"tf": 15, "mode": "oppose"}, "vix_size": False,
     "min_flow_pct": 65, "target_roe": 1.00, "rr": 1.0, "enabled": True},   # vix_size off: walk-forward showed the VIX overlay HURT this chop rule (+27.8->+23.8). dmi_confirm 2026-09-06 (check_msft_chop_put, close_only bars): fade the put-flow trigger only when MSFT's own 15m DMI points UP (+DI>-DI). IS +12.2->+39.9 / OOS +10.7->+22.6, win 0.57->0.83, maxLL 26->7, n 237->101, slice coverage unchanged (4/6), beats bootstrap null. The DI-agrees half is a -2% loser

    {"name": "NVDA LOWVOL PUT", "ticker": "NVDA", "direction": "PUT",
     "hours": [10, 11, 12, 13, 14], "dte": [0, 1], "regime": "LOWVOL", "trail_pct": 0,
     "min_flow_pct": 80, "target_roe": 1.00, "rr": 1.5, "enabled": True},   # trail_pct 0 2026-09-08 (check_trail_regression, realistic fills BOTH sides): static bracket OOS +23.9% vs trail50 +8.4% -- the trail costs this rule 15pp. Same mechanism as META: winners top near the +100% target, so a 50%-from-peak trail exits around +50% where the static TP banks +100%. NOTE this rule has a REAL stop already (rr 1.5 -> stop_roe 0.667), unlike the rr-1.0 rules.
    # amt_open: above_va REMOVED 2026-09-07: retest during the NVDA/AAPL legacy-GEX-tree review showed PLAIN LOWVOL PUT p80 is already robust -- n=164, IS +29.8% / OOS +22.4%, win 0.72, maxLL 15, ALL 6 calendar slices + (check_screen_candidate.py, netprem flow). The gate was cutting n to ~88 without improving the (already clean) rule, same over-gating META LOWVOL PUT shed in the walk-forward. rr 1.5 kept from the original tune (validation was no-stop rr 1.0 and still clean; a tighter stop only helps). DMI-oppose overlay also beats null here (OOS +51%) but would halve n / drop slice coverage -- not added


    {"name": "TSLA CHOP PUT", "ticker": "TSLA", "direction": "PUT",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0, 1], "regime": "CHOP", "amt_open": "above_va",
     "min_flow_pct": 80, "target_roe": 0.80, "rr": 1.0, "enabled": False},  # DISABLED 2026-09-06: check_config_walkforward -- PLAIN core IS -2.9 / OOS -3.0, win 0.41, maxLL 46; the +22.9/+39.6 deployed number is carried 100% by the amt_open gate on n=92 across only 3 slices. Negative unconditional edge = fragile

    {"name": "SPY CHOP CALL", "ticker": "SPY", "direction": "CALL",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0], "regime": "CHOP", "amt_open": "below_va",
     "flow_zscore": {"k": 2.0, "window_days": 60},
     "min_flow_pct": 90, "target_roe": 1.00, "rr": 2.0, "enabled": True},   # amt: below_va bucket +58.8/+29.0 (n=86) vs +11.7/+14.0 all; inside_va OOS -48.8. flow_zscore: OOS +29.0%->+43.3% (n=52)

    {"name": "SPY POS/PUT", "ticker": "SPY", "direction": "PUT",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0], "regime": "POSITIVE_GEX", "amt_open": "inside_va",
     "min_flow_pct": 80, "target_roe": 1.00, "rr": 1.0, "enabled": False},  # DISABLED 2026-09-06: check_config_walkforward -- PLAIN core IS -11.0 / OOS -10.5, win 0.37, EVERY slice red; the +21.0/+13.6 deployed number is 100% the amt_open=inside_va gate carrying a -10% rule. If AMT decays you hold a loser

    {"name": "QQQ HIVOL CALL", "ticker": "QQQ", "direction": "CALL",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0, 1], "regime": "HIVOL", "amt_open": {"exclude": ["above_va"]},
     "vol_overlay": {"agree-": 1.5}, "skip_macro_am": True,
     "min_flow_pct": 95, "target_roe": 1.00, "rr": 1.0, "enabled": True},   # amt: above_va was -38/-43 (the whole IS drag, n=126); excluding it rescues the fragile rule. vol_overlay: OOS +79.6%->+90.5%. dex_pct_max REMOVED 2026-09-06. skip_macro_am 2026-09-06 (check_event_days): on CPI/NFP/PCE days this rule is -86% / win 0.07 in the 11:00-14:00 window (the 8:30 move is done; the late trigger buys exhausted momentum). NOTE core PLAIN still weak (IS -7.2) -- amt gate load-bearing; watch

    {"name": "IWM HIVOL CALL", "ticker": "IWM", "direction": "CALL",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0, 1], "regime": "HIVOL", "skip_macro_am": True,
     "min_flow_pct": 80, "target_roe": 1.00, "rr": 1.0, "enabled": True},   # skip_macro_am 2026-09-06 (check_event_days): CPI/NFP/PCE days = -66% / win 0.0 in the 11:00-14:00 window (n 35), vs +22% on normal days. STRIPPED TO PLAIN 2026-09-06: check_config_walkforward -- PLAIN (HIVOL + p80) is a robust all-weather rule (n=962, IS +13.8 / OOS +25.1, 5/6 slices +). The stack (vol_overlay + flow_zscore k2.5 + dex_pct<0.2) collapsed it to n=91 across only S2+S5 -- the "+145% OOS" was one 4-month window. vol_overlay had independent (modest) support and can be re-added deliberately; flow_zscore + dex_pct_max removed for good

    {"name": "AVGO HIVOL PUT", "ticker": "AVGO", "direction": "PUT",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0, 1], "regime": "HIVOL", "amt_open": "inside_va", "vix_size": False,
     "min_flow_pct": 80, "target_roe": 1.00, "rr": 1.0, "enabled": True},   # vix_size off: single name, VIX-agnostic in the walk-forward. STABILIZED 2026-09-06 after 0/4 paper: amt inside_va + p80 -> IS +27.4% / OOS +21.2% (n=153) vs deployed +4.8/+4.3. above_va open (gap/trend-up) was the whole drag; raw trend gate + TP/SL-tighten both failed

    {"name": "SMH LOWVOL PUT", "ticker": "SMH", "direction": "PUT",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0, 1], "regime": "LOWVOL",
     "min_flow_pct": 65, "target_roe": 1.00, "rr": 1.0, "enabled": True},   # NEW 2026-09-06: screened + netprem-validated IS +13.5% (n55) / OOS +16.3% (n144) win 0.61. No gate helps (AMT/trend/hours all IS/OOS-inconsistent). NOTE semi-sector short overlap w/ AVGO+NVDA PUT -- watch same-day sizing. IS/OOS n imbalanced 55/144 (LOWVOL regime clustered recent)

    {"name": "GLD amp1 CALL", "ticker": "GLD", "direction": "CALL",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0, 1], "amp_min": 1, "ema_confirm": 3,
     "flow_window_days": 45, "vix_size": False,
     "min_flow_pct": 90, "target_roe": 1.00, "rr": 1.0, "enabled": True},   # strike_offset REMOVED 2026-09-19 (was 1). Its source, check_strike_selection (2026-09-07), predates every fill correction: CUSHION_CAP, FILL_COST and the 27% entry-fill measurement all landed 09-16, and it scored TP/SL brackets rather than the deployed trail50. Re-run under the corrected model (check_strike_offset) puts GLD's ATM arm 7.8pp of median ROE AHEAD of OTM+1 (+12.7 vs +4.9). It was also never honoured consistently: bot_runner picks from a 5-strike MQTT basket anchored at the open, so on 2026-09-18 the offset applied at 09:49 (spot 399.36 -> strike 400) but not at 12:49 (spot 401.57 -> 402, the nearest) because strike 403 was never subscribed. Live was neither ATM nor OTM+1 but drifted between them as spot left the anchor. vix_size off. netprem +29.5/+26.7 w/ 3m EMA-stack confirm. flow_window_days 45

    {"name": "AMZN afternoon PUT", "ticker": "AMZN", "direction": "PUT",
     "hours": [12, 13, 14], "dte": [0, 1],
     "min_flow_pct": 65, "target_roe": 0.80, "rr": 1.0, "enabled": False},  # DISABLED 2026-09-08. The old "+2.6% (n=1175)" was AS-SCREENED (every trigger scored independently) on MID fills. Under the fill model the bot actually runs -- SEQUENTIAL (one position per ticker) + realistic ask-in/bid-out -- it is negative on BOTH exit policies: static bracket n=214 OOS -3.2%, trail50 n=230 OOS -1.6%, and negative in both halves either way. This is the LARGEST and cleanest sample in the book (~40% of all book trades), so the negative read is the most trustworthy one we have. Not an exit problem; the rule has no edge

    {"name": "LULU HIVOL/UP PUT", "ticker": "LULU", "direction": "PUT",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0], "regime": ["HIVOL", "UPTREND"],
     "dmi_confirm": {"tf": 15, "mode": "oppose"},
     "min_flow_pct": 65, "target_roe": 1.00, "rr": 1.0, "enabled": False},   # NEW 2026-09-07: Friday-weekly-only name -> dte [0] = 0DTE Fridays only (Thu 1DTE leg failed IS). Screened (check_etf_screen --dow) + validated (check_screen_candidate.py). BASELINE HIVOL∪UPTREND / Fri / p65: IS +24.9% / OOS +16.6%, win 0.63, 5/6 slices + (S6 -3), maxLL 34. dmi_confirm oppose (fade the put-flow trigger only when LULU's own 15m Wilder DMI points UP, +DI>-DI): IS +30.0 / OOS +53.0, win 0.71, maxLL 34->8, n 413->102, both sub-regimes hold under it (UP IS+38.9/OOS+42.9 n82; HIVOL IS+37.0/OOS+47.2 n78), beats bootstrap null (+53.0 vs p95 +29.5). regime is a LIST = OR-match. vix_size left ON (untested; HIVOL leg already implies elevated vol). Realistic ask-in/bid-out fill ~4pp haircut -> still ~+49%

    # --- watch, not live: held OOS but on a thin / unstable sample ---
    {"name": "META midday PUT (amp>=1)", "ticker": "META", "direction": "PUT",
     "hours": [11, 12, 13, 14], "dte": [0, 1], "amp_min": 1,
     "min_flow_pct": 65, "target_roe": 0.80, "rr": 1.0, "enabled": False},  # OOS +5.9%, overlaps LOWVOL rule

    {"name": "MSFT quiet PUT (amp>=2)", "ticker": "MSFT", "direction": "PUT",
     "hours": [9, 10, 11, 12, 13, 14], "dte": [0, 1], "amp_min": 2,
     "min_flow_pct": 50, "target_roe": 1.00, "rr": 1.0, "enabled": False},  # OOS +11.5% but n=28
]