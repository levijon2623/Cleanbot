import os
import re
import time
import json
import logging
import statistics
import threading
from collections import deque, defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from config import (WATCHLIST, DRY_RUN, PAPER_TRADING, MAX_PORTFOLIO_RISK_PCT,
                    SIZING_TARGET_PREMIUM_PCT, SIZING_MAX_CONTRACTS, NO_ENTRY_BEFORE_MOD,
                    VIX_SIZE_MULT, VIX_MEDIAN_WINDOW, TRAIL_PCT, QUOTE_BLIND_PANIC_S,
                    BAD_TICK_CONFIRMS, BAD_TICK_CONFIRM_S, BAD_TICK_COLLAPSE,
                    BAD_TICK_MIN_PREV,
                    ENTRY_CHASE_S, ENTRY_CHASE_ROUNDS, ENTRY_CHASE_TICK,
                    ENTRY_CHASE_MAX_SLIP, SHUTDOWN_EXIT_S,
                    EXIT_FLOOR_FRAC, CUSHION_CAP, FILL_COST,
                    RULES, USE_RULES, FLOW_THRESHOLDS_PATH)
import live_state                 # read-only chart emitter; cannot raise
import manual_orders              # discretionary orders from flow_viewer
from macro_calendar import is_macro_am_day
from amt_profile import profile_from_bars, classify_open, amt_ok
import market_calendar as MC
from telegram_notifier import (TelegramNotifier, load_trades, round_trips,
                               summarize, summarize_both, _et_date_str)
from webull_gamma_client import WebullGammaClient
from unusual_whales_client import UnusualWhalesClient
from local_black_scholes_engine import LocalBlackScholesEngine
from strike_selector import StrikeSelector
from portfolio_manager import CapitalAllocator

# =====================================================================
# MARKET CLOCK
# =====================================================================
# Every hour-of-day / EOD rule in config.py is expressed in US market
# (Eastern) time. Pin it explicitly so the bot behaves identically no
# matter what the host server's local clock is set to.
MARKET_TZ = ZoneInfo("America/New_York")

# _live_flow_threshold: the bot's own crossover history replaces the static
# (backtest-calibrated) JSON only once it spans this many DISTINCT trading days.
# The scanner logs ~70 crossovers/day/ticker, so a raw "30 samples" gate was
# satisfied on day one -- the bot ran on a 3-day percentile from the start.
# Percentile is taken over ONE value per day (that day's MEDIAN crossover flow),
# so a single choppy session cannot flood the sample with near-duplicate values.
#
# 🚨 CORRECTED 2026-09-20. This block used to claim the daily-median percentile
# "reproduces the backtest's per-crossover percentile (verified within ~5% on
# IWM/AMZN/NVDA/MU/SPY/META/GLD)". check_gate_divergence re-measured it and that
# is NOT TRUE for 4 of 9 deployed rules -- QQQ -14.1%, GLD -10.5%, SPY -8.3%,
# IWM -5.7% median gap, with the 90th percentile of |gap| reaching 26-30% on
# QQQ/GLD. Two of the four (SPY, IWM) were named in the original comment as
# verified. The aggregation is still DELIBERATE and defensible -- on the core
# five it scored BETTER than per-crossover (+5,114 vs +4,053) -- but it is a
# different statistic, not an equivalent one, and research and live will not
# agree on the threshold. Do not treat the two as interchangeable.
FLOW_HISTORY_MIN_DAYS = 20

def market_now() -> datetime:
    return datetime.now(MARKET_TZ)


def tracked_tickers() -> list:
    """The tickers the live loop polls, streams and seeds.

    RULES mode: ENABLED-rule tickers only. Trimmed from the whole WATCHLIST on
    2026-09-27 when UW went to the Basic plan (40,000 requests/day, no socket):
    REST flow every 15s is ~1,600 requests per ticker per session and VWAP bars
    another ~200, so the six names no enabled rule trades (AAPL AMZN GOOGL TSLA
    MU LULU) cost ~10,800 a day to compute a flow line nothing reads.

    What this drops for those names: live cum_flow, their crossovers in
    flow_trigger_log.jsonl, and their rows in the viewer. Re-enabling a rule
    re-adds its ticker on the next restart, and its gate history is rebuilt
    from historical/NETPREM{T}.parquet by rebuild_flow_trigger_log.py rather
    than waited for (refresh with `uw_options_data_lake netprem-build` first).
    Legacy (USE_RULES False) mode keeps the whole WATCHLIST, as before.
    """
    if not USE_RULES:
        return list(WATCHLIST.keys())
    out = []
    for r in RULES:
        if r.get("enabled", True) and r["ticker"] not in out:
            out.append(r["ticker"])
    return out


def _ema_stack_state(closes, spans=(8, 21, 34)):
    """closes = tf-minute bar closes oldest->newest. 'BULL' if
    ema(spans[0]) > ema(spans[1]) > ... (fast above slow), 'BEAR' if the mirror,
    'MIXED' otherwise. None if too few bars for the slowest EMA to settle.
    Matches directional_flow_backtester.load_ema_stack (adjust=False EMA)."""
    if len(closes) < max(spans) + 3:
        return None

    def _ema(vals, span):
        k = 2.0 / (span + 1.0)
        e = vals[0]
        for v in vals[1:]:
            e = v * k + e * (1.0 - k)
        return e

    es = [_ema(closes, s) for s in spans]
    if all(es[i] > es[i + 1] for i in range(len(es) - 1)):
        return "BULL"
    if all(es[i] < es[i + 1] for i in range(len(es) - 1)):
        return "BEAR"
    return "MIXED"


def _dmi_state(bars, period=14):
    """bars = list of (high, low, close) tf-minute bars oldest->newest.
    Returns +1 if +DI > -DI (price grinding up), -1 if -DI > +DI, 0 if flat,
    None if too few bars. Wilder RMA smoothing -- matches
    check_adx_dmi._adx_frame (ewm alpha=1/period, adjust=False)."""
    if len(bars) < period * 2 + 2:
        return None
    highs = [b[0] for b in bars]
    lows = [b[1] for b in bars]
    closes = [b[2] for b in bars]
    k = 1.0 / period
    atr = pdm_s = mdm_s = None
    for i in range(1, len(bars)):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        pdm = up if (up > dn and up > 0) else 0.0
        mdm = dn if (dn > up and dn > 0) else 0.0
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        if atr is None:
            atr, pdm_s, mdm_s = tr, pdm, mdm
        else:
            atr = atr + k * (tr - atr)
            pdm_s = pdm_s + k * (pdm - pdm_s)
            mdm_s = mdm_s + k * (mdm - mdm_s)
    if not atr:
        return 0
    pdi = 100.0 * pdm_s / atr
    mdi = 100.0 * mdm_s / atr
    if abs(pdi - mdi) < 1e-9:
        return 0
    return 1 if pdi > mdi else -1

logging.getLogger().setLevel(logging.CRITICAL)

class FlowMomentumTracker:
    """
    Minute-Bucketed Momentum Tracker.

    Mirrors the backtest (true_options_simulator.find_flow_entries):
      flow_ema_5 = cumulative_net_flow.ewm(span=5, adjust=False).mean()   # ~5-minute EMA
      bullish  = (cum.shift(1) <= ema.shift(1)) & (cum > ema)
      bearish  = (cum.shift(1) >= ema.shift(1)) & (cum < ema)
    evaluated once per CLOSED 1-minute bar (not per second).
    """
    EMA_SPAN = 5
    MIN_BARS = 2  # backtest applies ewm from the 2nd bar onward

    def __init__(self):
        # { ticker: { minute_epoch: latest cumulative flow seen in that minute } }
        self.minute_flow = {}
        self.last_eval_minute = {}

    def reset(self, ticker=None):
        if ticker is None:
            self.minute_flow.clear()
            self.last_eval_minute.clear()
        else:
            self.minute_flow.pop(ticker, None)
            self.last_eval_minute.pop(ticker, None)

    def closed_series(self, ticker, now_min=None):
        """(closed_minutes, cumulative_flow, ema) for CLOSED bars only.

        Extracted 2026-09-20 so the live chart renders the SAME numbers the
        signal is computed from. The emitter used to be able to re-derive this
        from UW, which is a second implementation of the trigger -- the exact
        shape of the two divergences found on 2026-09-19/20 (strike_offset
        honoured live but not in research; a stale entry spot in
        build_candidates). One implementation, two readers.

        Returns empty lists rather than raising when the series is too short.
        """
        now_min = int(time.time() // 60) if now_min is None else now_min
        buckets = self.minute_flow.get(ticker) or {}
        closed_mins = sorted(m for m in buckets if m < now_min)
        if len(closed_mins) < self.MIN_BARS:
            return closed_mins, [], []
        series = [buckets[m] for m in closed_mins]
        alpha = 2 / (self.EMA_SPAN + 1)
        ema = [series[0]]
        for val in series[1:]:
            ema.append(val * alpha + ema[-1] * (1 - alpha))
        return closed_mins, series, ema

    def update_and_check(self, ticker, cumulative_flow_value):
        now_min = int(time.time() // 60)
        buckets = self.minute_flow.setdefault(ticker, {})
        buckets[now_min] = cumulative_flow_value  # keep the latest value of the in-progress minute

        # bound memory (~2 sessions of minutes)
        if len(buckets) > 3000:
            for k in sorted(buckets)[:-3000]:
                del buckets[k]

        # only closed minutes feed the EMA (the current minute is still forming)
        closed_mins, series, ema = self.closed_series(ticker, now_min)
        if len(closed_mins) < self.MIN_BARS:
            return "WAIT"

        # evaluate each closed bar at most once
        last_min = closed_mins[-1]
        if self.last_eval_minute.get(ticker) == last_min:
            return "WAIT"
        self.last_eval_minute[ticker] = last_min

        curr_flow, prev_flow = series[-1], series[-2]
        curr_ema, prev_ema = ema[-1], ema[-2]

        if prev_flow <= prev_ema and curr_flow > curr_ema:
            return "LONG"
        if prev_flow >= prev_ema and curr_flow < curr_ema:
            return "SHORT"
        return "WAIT"

class FlowExecutionEngine:
    def __init__(self):
        load_dotenv()
        print("=== INITIALIZING ZERO-LATENCY FLOW EXECUTION ENGINE ===")
        
        self.webull = WebullGammaClient(
            app_key=os.getenv("WEBULL_APP_KEY"),
            app_secret=os.getenv("WEBULL_APP_SECRET"),
            account_id=os.getenv("WEBULL_ACCOUNT_ID"),
            paper_trading=PAPER_TRADING
        )
        self.uw = UnusualWhalesClient(os.getenv("UW_API_KEY"))
        # 🚨 MANUAL-ONLY WHEN THERE IS NO UW KEY (2026-10-02). Everything the
        # BOT trades on -- the flow feed, the regimes, VIX, the rule scanner --
        # is Unusual Whales. Without a key those calls would 401 into empty
        # data all day and the scanner would sit on a flat zero that looks like
        # "no triggers". So they are skipped, said so at startup, and the engine
        # keeps what needs only Webull: quotes, baskets, the manual desk, the
        # viewer (VIEWER_DATA, see webull_viewer_data) and Telegram.
        self.uw_enabled = bool((os.getenv("UW_API_KEY") or "").strip())
        self.bs_engine = LocalBlackScholesEngine(risk_free_rate=0.045)
        self.strike_selector = StrikeSelector(self.bs_engine)
        self.flow_tracker = FlowMomentumTracker()

        self.active_snipes = {}

        # NEW: Strike Drift Management State
        self.options_basket = {}
        self.active_ticker_occs = {}
        self.anchor_prices = {}
        self.is_rebalancing = {}

        # NEW: Cumulative Flow Tracking State
        # processed_ticks is a set for O(1) membership + a parallel deque that
        # holds insertion order, so trimming drops the OLDEST ids (a bare
        # set()[-250:] keeps an arbitrary 250 and lets old ticks be re-counted).
        # Keyed by tracked_tickers(), not WATCHLIST: live_state's VWAP thread
        # and the viewer both take their ticker list from these keys.
        self.cumulative_flow = {ticker: 0.0 for ticker in tracked_tickers()}
        self.processed_ticks = {ticker: set() for ticker in tracked_tickers()}
        self.processed_tick_order = {ticker: deque() for ticker in tracked_tickers()}
        # REST ledger: {ticker: {tape_time: net_premium}}. See _ingest_flow.
        self.rest_minute_vals = {ticker: {} for ticker in tracked_tickers()}
        self._flow_epoch_seen = {}

        # Capital allocator (CRO gate). Armed from live account data in run_brain().
        self.allocator = None
        self.account_equity = 0.0

        self.session_date = None  # ET trading day the cumulative flow belongs to

        # Daily GEX regime, polled once per session and HELD (matches the daily
        # backtest). { ticker: {"regime": ..., "net_gex": ..., "as_of": ...} }
        self.session_gex = {}

        # RULES mode: prior-day trend / volume state per rule-ticker, polled once
        # per session and held. { ticker: {"trend":..., "volume":..., "gex":..., "amp":int} }
        self.session_regime = {}
        # static percentile->$ flow thresholds (Lake units), deployed from the
        # local box; the bot's own rolling log below supersedes them at 30+ samples.
        self.flow_thresholds = self._load_flow_thresholds()
        # { ticker: deque([(date_iso, abs_cum_flow_at_crossover), ...]) } -- every
        # EMA crossover, for computing the ticker's own live flow percentiles.
        self.flow_trigger_log_path = os.getenv("FLOW_TRIGGER_LOG", "flow_trigger_log.jsonl")
        self.flow_trigger_hist = {}
        self._load_trigger_history()
        self._thr_warned = set()

        # rolling 1-min underlying closes (ET minute key -> px) for rules whose
        # config carries "ema_confirm"; seeded from UW at session start.
        self._ema_confirm_tickers = {
            r["ticker"] for r in RULES if r.get("enabled", True) and r.get("ema_confirm")
        } if USE_RULES else set()
        self.px_hist = defaultdict(lambda: deque(maxlen=1560))   # ~4 RTH days
        self._ema_warned = set()

        # rolling 1-min (minute_key, high, low, close) bars for rules carrying
        # "dmi_confirm" -- bucketed into tf-min bars for a live Wilder ADX/DMI
        # read. Seeded from UW at session start (real 1-min H/L); per-minute live
        # updates append (spot, spot, spot) so the current bar's H/L come from
        # 1-min closes, exactly as check_msft_chop_put validates (close_only).
        self._dmi_confirm_tickers = {
            r["ticker"] for r in RULES if r.get("enabled", True) and r.get("dmi_confirm")
        } if USE_RULES else set()
        self.bar_hist = defaultdict(lambda: deque(maxlen=1560))
        self._dmi_warned = set()

        # rules carrying skip_macro_am: suppressed on CPI/NFP/PCE mornings
        # (check_event_days -- 8:30am releases turn IWM/QQQ HIVOL CALL into a
        # -65% / win-0.0 rule in the 11:00-14:00 window). Warn once per day.
        self._macro_am_warned = set()

        # Auction-Market-Theory: rules gated on where today OPENED vs the prior
        # day's volume value area. { ticker: prior-day (vah, val) } seeded at
        # session start; { ticker: 'below_va'|'inside_va'|'above_va' } fixed on
        # the first RTH tick.
        self._amt_open_rules = {
            r["ticker"] for r in RULES if r.get("enabled", True) and r.get("amt_open")
        } if USE_RULES else set()
        self._prev_va = {}
        self._amt_open = {}

        # net_dex percentile gate (dex_pct_max): polled once per session and held,
        # like the daily GEX regime. { ticker: float in [0,1] }
        self._dex_pct_tickers = {
            r["ticker"] for r in RULES if r.get("enabled", True) and r.get("dex_pct_max") is not None
        } if USE_RULES else set()
        self.session_dex_pct = {}

        # VIX-regime sizing overlay: prior-day VIX vs its own trailing median,
        # polled once per session. session_vix_mult scales target premium for
        # rules without vix_size:False. Fails open to 1.0x.
        self._vix_size_on = USE_RULES and any(
            r.get("enabled", True) and r.get("vix_size", True) for r in RULES
        )
        self.session_vix = None
        self.session_vix_mult = 1.0

        self.log_file = "bot_executions_log.jsonl"
        # The AMENDED ledger. Same events, but sessions a bug corrupted get
        # replaced by a tape replay (build_counterfactual.py) and every row
        # carries source/fiction so the two can never be confused. The raw log
        # stays observational: it is the only thing here that can falsify the
        # simulator, and it has twice done exactly that.
        self.cf_file = os.getenv("COUNTERFACTUAL_LOG",
                                 "bot_executions_counterfactual.jsonl")

        # Telegram: push every ENTRY/EXIT + answer /today /all /pnl /open /status.
        # No-op stand-in if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID aren't set.
        self.tg = TelegramNotifier.from_env()

    # =====================================================================
    # RULES MODE -- regime poll, rule matching, live flow percentiles
    # =====================================================================
    def _load_flow_thresholds(self):
        try:
            with open(FLOW_THRESHOLDS_PATH) as f:
                return json.load(f)
        except Exception:
            return {}

    # ~70 crossovers/day/ticker -> ~120 trading days of headroom for the 90d window
    _FLOW_HIST_MAXLEN = 8500

    def _load_trigger_history(self):
        """Rehydrate the crossover history, DROPPING non-trading days.

        Belt and braces with the guard in _log_flow_trigger: rows written before
        that guard existed are still on disk, and `_daily_medians` would give a
        one-print Saturday the same weight as a full session. Filtering on load
        means the file does not have to be perfect for the gate to be.
        """
        import datetime as _dt
        skipped = 0
        try:
            with open(self.flow_trigger_log_path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        if _dt.date.fromisoformat(e["date"]).weekday() >= 5:
                            skipped += 1
                            continue
                        self.flow_trigger_hist.setdefault(
                            e["ticker"], deque(maxlen=self._FLOW_HIST_MAXLEN)).append(
                            (e["date"], float(e["abs_flow"])))
                    except Exception:
                        continue
        except FileNotFoundError:
            return
        if skipped:
            print(f"  ⚠️ flow history: skipped {skipped} non-trading-day row(s)")

    def _log_flow_trigger(self, ticker, abs_flow):
        """Record one EMA crossover for the ticker's own rolling percentile.

        Two guards, because this history is NOT inert: once a ticker spans
        FLOW_HISTORY_MIN_DAYS (20) distinct days its percentile SUPERSEDES the
        static threshold, and `_daily_medians` weights every day EQUALLY. A
        junk day therefore counts as much as a real session, and also pulls the
        ticker over the 20-day line sooner.

        1. NON-TRADING DAY. run_brain's loop is `while True` with no calendar
           gate, so the bot happily evaluates flow on a Saturday. On 2026-09-05
           and 09-06 it logged AVGO at 261,471 against a weekday median near
           22M -- two weekend "days" in a 14-day history.
        2. FROZEN FEED. Those two rows carried the IDENTICAL value, which is the
           real signature: a stale quote replayed, not a market. An exact repeat
           of the last logged value for a ticker is not a new crossover. This
           catches holidays and mid-session feed outages too, which a calendar
           check alone would miss.
        """
        now = market_now()
        if now.weekday() >= 5:
            return
        v = float(abs_flow)
        if getattr(self, "_last_trigger_flow", {}).get(ticker) == v:
            return                      # frozen feed replaying the same number
        if not hasattr(self, "_last_trigger_flow"):
            self._last_trigger_flow = {}
        self._last_trigger_flow[ticker] = v
        d = now.date().isoformat()
        self.flow_trigger_hist.setdefault(
            ticker, deque(maxlen=self._FLOW_HIST_MAXLEN)).append((d, v))
        try:
            with open(self.flow_trigger_log_path, "a") as f:
                f.write(json.dumps({"ticker": ticker, "date": d, "abs_flow": round(v, 2)}) + "\n")
        except Exception:
            pass

    @staticmethod
    def _percentile(vals, p):
        s = sorted(vals)
        if not s:
            return None
        return s[min(len(s) - 1, int(len(s) * p / 100.0))]

    @staticmethod
    def _daily_medians(hist, window_days):
        """One value/day = that day's MEDIAN crossover flow, over the trailing
        window_days -- the anti-clustering dedup (session-12 fix: a single choppy
        session logs ~70 near-duplicate crossovers/day, which would otherwise
        flood/skew both a percentile AND a mean+std estimate)."""
        cutoff = (market_now().date() - timedelta(days=window_days)).isoformat()
        by_day = defaultdict(list)
        for d, v in hist:
            if d >= cutoff:
                by_day[d].append(v)
        return [statistics.median(vs) for vs in by_day.values() if vs]

    def _live_flow_threshold(self, ticker, rule):
        """$ threshold for this rule. Once the bot's own crossover history spans
        >= FLOW_HISTORY_MIN_DAYS distinct trading days in the trailing window
        (rule["flow_window_days"], default 90d), its percentile (one value/day =
        that day's MEDIAN crossover flow) wins -- self-consistent units, robust
        to a single choppy session. Before that, the static JSON (backtest-
        calibrated; warn once that it's in Lake units). check_flow_window.py
        gridded 10-250d: the blended optimum is ~45d and both defaults (60d
        backtest / 90d live) already sit close to it, so this is usually left at
        the 90d default -- GLD is the one rule with a sharp enough per-ticker
        peak (45d) to override.

        rule["flow_zscore"] = {"k", "window_days"} takes priority once warmed --
        mean + k*std of the same daily-median pool, instead of a percentile rank
        (check_flow_zscore.py). Falls through to the percentile path (and its
        static-JSON fallback) until FLOW_HISTORY_MIN_DAYS is satisfied."""
        if rule.get("flow_abs") is not None:
            return float(rule["flow_abs"])
        hist = self.flow_trigger_hist.get(ticker)
        zs = rule.get("flow_zscore")
        if zs and hist:
            zdaily = self._daily_medians(hist, int(zs.get("window_days") or 90))
            if len(zdaily) >= FLOW_HISTORY_MIN_DAYS:
                sd = statistics.stdev(zdaily) if len(zdaily) > 1 else 0.0
                if sd > 0:
                    return statistics.mean(zdaily) + float(zs["k"]) * sd
        p = int(rule.get("min_flow_pct") or rule.get("flow_pct"))
        if hist:
            window_days = int(rule.get("flow_window_days") or 90)
            daily = self._daily_medians(hist, window_days)
            if len(daily) >= FLOW_HISTORY_MIN_DAYS:
                return self._percentile(daily, p)
        jt = self.flow_thresholds.get(ticker)
        if jt and f"p{p}" in jt:
            if ticker not in self._thr_warned:
                wd = int(rule.get("flow_window_days") or 90)
                have = len({d for d, _ in hist if d >= (market_now().date() - timedelta(days=wd)).isoformat()}) if hist else 0
                print(f"  ℹ️ {ticker}: flow threshold from {FLOW_THRESHOLDS_PATH} (asof {jt.get('asof')}, "
                      f"Lake units — self-calibration takes over at {FLOW_HISTORY_MIN_DAYS} distinct days, have {have})")
                self._thr_warned.add(ticker)
            return float(jt[f"p{p}"])
        return None

    def _seed_price_history(self, tickers):
        """Fill self.px_hist (ema_confirm) and self.bar_hist (dmi_confirm) for any
        rule-ticker from UW 1-min OHLC so the EMA stack / DMI is warm at 09:30
        (both run continuously across days)."""
        want_ema = [t for t in tickers if t in self._ema_confirm_tickers]
        want_dmi = [t for t in tickers if t in self._dmi_confirm_tickers]
        if not want_ema and not want_dmi:
            return
        print("\n📈 Seeding intraday price history (EMA-confirm / DMI-confirm rules)...")
        for tk in set(want_ema) | set(want_dmi):
            need_hl = tk in self._dmi_confirm_tickers
            try:
                bars = self.uw.get_intraday_bars(tk, lookback_days=3, ohlcv=need_hl)
            except Exception as e:
                bars = []
                print(f"  ├─ {tk}: intraday seed failed ({e})")
            if tk in self._ema_confirm_tickers:
                self.px_hist[tk].clear()
                for b in bars:
                    self.px_hist[tk].append((b["minute_et"], float(b["close"])))
            if tk in self._dmi_confirm_tickers:
                self.bar_hist[tk].clear()
                for b in bars:
                    c = float(b["close"])
                    self.bar_hist[tk].append((b["minute_et"], float(b.get("h", c)),
                                              float(b.get("l", c)), c))
            print(f"  ├─ {tk}: {len(bars)} 1-min bars seeded"
                  f"{' (+H/L for DMI)' if need_hl else ''}")

    def _seed_amt_profiles(self, tickers):
        """Build the PRIOR session's value area for each amt_open rule-ticker."""
        want = [t for t in tickers if t in self._amt_open_rules]
        if not want:
            return
        print("\n📐 Seeding prior-day value areas (Auction Market Theory rules)...")
        # 🚨 THE PRIOR SESSION, NAMED EXPLICITLY. lookback_days=1 counts from
        # TODAY, and during RTH today already has bars -- so every mid-session
        # restart built the "prior" VA from today's first minutes (2026-09-29
        # 09:54: SPY "prior VA" 765.13-765.89, the first 25 minutes of the
        # day). The backtest (amt_profile.amt_open_map) uses the previous
        # session's full RTH profile.
        # previous_trading_day is ON OR BEFORE its argument -- step back first
        prior = MC.previous_trading_day(market_now().date() - timedelta(days=1))
        for tk in want:
            try:
                bars = self.uw.get_intraday_bars(tk, lookback_days=1, ohlcv=True,
                                                 end_date=prior)
            except Exception as e:
                bars = []
                print(f"  ├─ {tk}: profile seed failed ({e})")
            if not bars:
                self._prev_va.pop(tk, None)
                continue
            # bin width = 0.05% of price
            mid = sorted(b["c"] for b in bars)[len(bars) // 2]
            prof = profile_from_bars(bars, round(mid * 0.0005, 2) or 0.01)
            if prof:
                self._prev_va[tk] = (prof["vah"], prof["val"])
                print(f"  ├─ {tk}: prior VA  {prof['val']:.2f} - {prof['vah']:.2f}  (POC {prof['poc']:.2f})")
            else:
                self._prev_va.pop(tk, None)
        self._amt_open = {}          # re-classify on the new session's open

    def _amt_open_state(self, ticker):
        """'below_va' / 'inside_va' / 'above_va' for today, from the 09:30 RTH
        bar's OPEN -- the backtest's definition -- and cached. None until then.

        🚨 IT USED TO BE "SPOT, WHENEVER FIRST ASKED". The heartbeat asks every
        minute, so a bot running overnight classified at 00:01 on the last
        print (2026-09-29 00:01: SPY "open 766.64"), and a bot restarted
        mid-session classified on the price two minutes after the restart
        (11:46: "open 763.69"). Neither is the open. It now waits for today's
        09:30 bar and reads its open from UW; until that exists it answers
        None and caches nothing, and _match_rule refuses an amt_open rule on
        None -- an unknown open is not a pass."""
        if ticker in self._amt_open:
            return self._amt_open[ticker]
        va = self._prev_va.get(ticker)
        if not va:
            return None
        now = market_now()
        if now.hour * 60 + now.minute < 9 * 60 + 31:     # the 09:30 bar has not closed
            return None
        if not MC.is_trading_day(now.date()):
            return None                   # no 09:30 bar is coming; do not ask UW
        try:
            bars = self.uw.get_intraday_bars(ticker, lookback_days=1, ohlcv=True,
                                             end_date=now.date())
        except Exception:
            bars = []
        first = next((b for b in bars if b["minute_et"][:10] == now.date().isoformat()
                      and b.get("mod") == 9 * 60 + 30), None)
        if not first:
            return None
        px = float(first["o"])
        loc = classify_open(px, va[0], va[1])
        if loc:
            self._amt_open[ticker] = loc
            print(f"  📐 {ticker}: opened {loc.replace('_', ' ')} "
                  f"(09:30 open {px:.2f} vs prior VA {va[1]:.2f}-{va[0]:.2f})")
        return loc

    def _record_price_minute(self, tickers, now):
        """Append the current spot as a 1-min bar for ema_confirm (close) and
        dmi_confirm (h=l=c=spot) tickers. Call once per wall-clock minute."""
        mkey = now.strftime("%Y-%m-%dT%H:%M")
        for tk in self._ema_confirm_tickers:
            if tk not in tickers:
                continue
            px = self.get_spot_price(tk)
            if px > 0 and (not self.px_hist[tk] or self.px_hist[tk][-1][0] != mkey):
                self.px_hist[tk].append((mkey, px))
        for tk in self._dmi_confirm_tickers:
            if tk not in tickers:
                continue
            px = self.get_spot_price(tk)
            if px > 0 and (not self.bar_hist[tk] or self.bar_hist[tk][-1][0] != mkey):
                self.bar_hist[tk].append((mkey, px, px, px))

    def _ema_confirm_ok(self, ticker, direction, rule):
        """Does the underlying's price EMA stack (rule['ema_confirm']-min bars,
        rule['ema_spans'] or 8/21/34) agree with the trade direction?
        Fails OPEN (returns True) when history isn't warm yet -- logged once."""
        tf = int(rule["ema_confirm"])
        spans = tuple(rule.get("ema_spans", (8, 21, 34)))
        hist = self.px_hist.get(ticker)
        if not hist:
            return True
        buckets = {}
        for mk, px in hist:
            try:
                mod = int(mk[11:13]) * 60 + int(mk[14:16])
            except (ValueError, IndexError):
                continue
            buckets[(mk[:10], mod // tf)] = px       # last close in the tf-bucket
        closes = [buckets[k] for k in sorted(buckets)]
        st = _ema_stack_state(closes, spans)
        if st is None:
            if ticker not in self._ema_warned:
                print(f"  ℹ️ {ticker}: EMA-{tf}m stack not warm yet ({len(closes)} bars) — ema_confirm open")
                self._ema_warned.add(ticker)
            return True
        return st == ("BULL" if direction == "CALL" else "BEAR")

    def _dmi_confirm_ok(self, ticker, direction, rule):
        """Wilder DMI on the underlying's rule['dmi_confirm']['tf']-min bars
        (period 14). mode 'agree' -> require +DI/-DI to point WITH the trade;
        'oppose' -> require it to point AGAINST (fade). Fails OPEN (True) until
        the DMI is warm -- logged once. Matches check_msft_chop_put (close_only)."""
        cfg = rule["dmi_confirm"]
        tf = int(cfg.get("tf", 15))
        period = int(cfg.get("period", 14))
        mode = cfg.get("mode", "oppose")
        hist = self.bar_hist.get(ticker)
        if not hist:
            return True
        buckets = {}
        last_key = last_mod = None
        for mk, h, l, c in hist:
            try:
                mod = int(mk[11:13]) * 60 + int(mk[14:16])
            except (ValueError, IndexError):
                continue
            key = (mk[:10], mod // tf)
            if key in buckets:
                ph, pl, _ = buckets[key]
                buckets[key] = (max(ph, h), min(pl, l), c)
            else:
                buckets[key] = (h, l, c)
            last_key, last_mod = key, mod
        # drop the in-progress final bucket so we read only COMPLETED tf-min bars
        # (matches check_msft_chop_put's _intra_asof "last bar that closed <= ts")
        if last_key is not None and last_mod is not None and (last_mod % tf) != tf - 1:
            buckets.pop(last_key, None)
        bars = [buckets[k] for k in sorted(buckets)]
        sig = _dmi_state(bars, period)
        if sig is None:
            if ticker not in self._dmi_warned:
                print(f"  ℹ️ {ticker}: DMI-{tf}m not warm yet ({len(bars)} bars) — dmi_confirm open")
                self._dmi_warned.add(ticker)
            return True
        if sig == 0:
            return True
        di_bull = sig > 0
        trade_bull = direction == "CALL"
        agree = di_bull == trade_bull
        return agree if mode == "agree" else (not agree)

    @staticmethod
    def _classify_regimes(bars, today_iso, fast=20, slow=50, slope_lag=10,
                          vol_win=20, vol_hi=1.25, vol_lo=0.80):
        """{'trend', 'volume'} as of the last COMPLETED session (drops today's
        in-progress bar). Mirrors directional_flow_backtester.load_trend_regime /
        load_volume_regime with the prior-day shift baked in."""
        b = [x for x in bars if x["date"] and x["date"] != today_iso]
        if len(b) < slow + slope_lag + 1:
            return {"trend": None, "volume": None}
        closes = [x["close"] for x in b]
        vols = [x["volume"] for x in b]
        end = len(closes)

        def sma(xs, n, e):
            seg = xs[e - n:e]
            return sum(seg) / n if len(seg) == n else None

        sf, ss, ss_prev = sma(closes, fast, end), sma(closes, slow, end), sma(closes, slow, end - slope_lag)
        trend = None
        if None not in (sf, ss, ss_prev):
            slope = ss - ss_prev
            trend = ("UPTREND" if (sf > ss and slope > 0) else
                     "DOWNTREND" if (sf < ss and slope < 0) else "CHOP")
        volume = None
        vseg = sorted(vols[end - vol_win:end])
        if len(vseg) == vol_win:
            med = (vseg[vol_win // 2 - 1] + vseg[vol_win // 2]) / 2.0 if vol_win % 2 == 0 else vseg[vol_win // 2]
            if med > 0:
                ratio = vols[end - 1] / med
                volume = "HIVOL" if ratio >= vol_hi else ("LOWVOL" if ratio <= vol_lo else "NORMVOL")
        return {"trend": trend, "volume": volume}

    def _poll_session_regimes(self, tickers):
        """Prior-day trend / volume state per rule-ticker, held for the session."""
        if not USE_RULES:
            return
        want = {r["ticker"] for r in RULES if r.get("enabled", True)}
        today = market_now().date().isoformat()
        print("\n📈 Polling daily trend / volume regime (held for the session)...")
        for t in tickers:
            if t not in want:
                continue
            bars = self.uw.get_daily_bars(t, limit=90)
            st = self._classify_regimes(bars, today)
            gx = (self.session_gex.get(t) or {}).get("regime")
            st["gex"] = gx if gx in ("POSITIVE", "NEGATIVE") else None
            st["amp"] = (int(st["gex"] == "NEGATIVE") + int(st["volume"] == "LOWVOL")
                         + int(st["trend"] == "CHOP")) if st["gex"] is not None else None
            self.session_regime[t] = st
            print(f"  ├─ {t}: trend {st['trend']}  vol {st['volume']}  gex {st['gex']}  amp {st['amp']}")

    @staticmethod
    def _combo_bucket(so, dr):
        """oi/dir spot-GEX combo bucket -- mirrors check_dgex.py's _combo_bucket.
        Both args are already sign-only (+1/-1); the vol_overlay rule field keys
        off this same 4-way label."""
        if so > 0 and dr > 0:
            return "agree+"
        if so < 0 and dr < 0:
            return "agree-"
        if so > 0 and dr < 0:
            return "oi+/dir-"
        return "oi-/dir+"

    def _match_rule(self, ticker, direction, state):
        """First enabled RULES entry matching (ticker, direction) whose regime /
        amp gate the current session state satisfies. Returns a copy with
        stop_roe = target_roe / rr, or None."""
        for r in RULES:
            if not r.get("enabled", True):
                continue
            if r["ticker"] != ticker or r["direction"] != direction:
                continue
            reg = r.get("regime")
            if reg is not None:
                regs = reg if isinstance(reg, (list, tuple)) else [reg]
                matched_reg = False
                for rg in regs:
                    if rg == "NEGATIVE_GEX":
                        want, got = "NEGATIVE", state.get("gex")
                    elif rg == "POSITIVE_GEX":
                        want, got = "POSITIVE", state.get("gex")
                    elif rg in ("LOWVOL", "NORMVOL", "HIVOL"):
                        want, got = rg, state.get("volume")
                    else:
                        want, got = rg, state.get("trend")
                    if got == want:
                        matched_reg = True
                        break
                if not matched_reg:
                    continue
            if r.get("amp_min") is not None:
                if state.get("amp") is None or state["amp"] < r["amp_min"]:
                    continue
            if r.get("amt_open"):
                # an UNKNOWN open refuses: amt_ok passes None (the backtest's
                # "no profile that day"), but live None means "not read yet"
                loc = self._amt_open_state(ticker)
                if loc is None or not amt_ok(r["amt_open"], loc):
                    continue
            if r.get("dex_pct_max") is not None:
                dp = self.session_dex_pct.get(ticker)
                if dp is None or dp >= r["dex_pct_max"]:
                    continue          # dealer directional positioning not light enough
            out = dict(r)
            out["stop_roe"] = r["target_roe"] / r["rr"]
            vo = r.get("vol_overlay")           # {"agree+"|"agree-"|"oi+/dir-"|"oi-/dir+": mult}
            if vo:                               # widen/tighten target+stop by the live GEX vol-regime
                sg = self.uw.get_spot_gex(ticker)
                mult = vo.get(self._combo_bucket(sg["oi"], sg["dir"]), 1.0) if sg else 1.0
                if mult != 1.0:
                    out["target_roe"] = round(r["target_roe"] * mult, 4)
                    out["stop_roe"] = out["target_roe"] / r["rr"]
                    print(f"  🌊 {ticker} vol-regime overlay: x{mult} -> "
                          f"target {out['target_roe']*100:.0f}% / stop {out['stop_roe']*100:.0f}%")
            ef = r.get("eod_flatten")           # "HH:MM" -> minute-of-day
            if ef:
                try:
                    _h, _m = map(int, str(ef).split(":"))
                    out["eod_flatten_mod"] = _h * 60 + _m
                except ValueError:
                    pass
            return out
        return None

    def _poll_daily_gex(self, tickers):
        """Fetch each ticker's daily GEX regime once and hold it for the session.

        The backtest uses one GEX value per day; live matches that with a poll
        rather than refreshing intraday. (The WS gex: stream is still recorded to
        gex_history_log.jsonl so we can later measure intraday flip frequency.)
        """
        print("\n📊 Polling daily GEX regime (held for the session)...")
        for t in tickers:
            rd = self.uw.get_daily_gex_regime(t)
            self.session_gex[t] = rd
            if rd:
                print(f"  ├─ {t}: {rd.get('regime', 'UNKNOWN')}  "
                      f"(net_gex {rd.get('net_gex', 0):,.0f}, as of {rd.get('as_of', '?')})")
            else:
                print(f"  ├─ {t}: UNAVAILABLE — ticker will be skipped until next poll")

    def _poll_session_dex(self, tickers):
        """net_dex trailing-252d percentile for the dex_pct_max rule-tickers,
        polled once and held (like the GEX regime). Low = dealer directional
        positioning unusually light -> moves follow through."""
        want = self._dex_pct_tickers & set(tickers)
        if not want:
            return
        print("\n📐 Polling net_dex percentile (dex_pct_max gate)...")
        for t in want:
            dp = self.uw.get_dex_pct(t)
            self.session_dex_pct[t] = dp
            cap = next((r["dex_pct_max"] for r in RULES
                        if r.get("enabled", True) and r["ticker"] == t and r.get("dex_pct_max") is not None), None)
            if dp is None:
                print(f"  ├─ {t}: UNAVAILABLE — dex_pct_max rules skip until next poll")
            else:
                gate = "OPEN" if (cap is not None and dp < cap) else "shut"
                print(f"  ├─ {t}: dex_pct {dp:.2f}  (cap {cap}) -> gate {gate}")

    def _poll_session_vix(self):
        """Prior-day VIX vs its own trailing-VIX_MEDIAN_WINDOW median, once per
        session. Sets self.session_vix_mult (VIX_SIZE_MULT below the median,
        1.0 at/above). Fails open to 1.0x. Applies to rules without
        vix_size:False -- see check_vix_overlay.py."""
        if not self._vix_size_on:
            return
        st = self.uw.get_vix_state(VIX_MEDIAN_WINDOW)
        self.session_vix = st
        if not st:
            self.session_vix_mult = 1.0
            print("\n📊 VIX regime: UNAVAILABLE — sizing overlay off (full size).")
            return
        self.session_vix_mult = 1.0 if st["favorable"] else VIX_SIZE_MULT
        tag = "favourable (full size)" if st["favorable"] else f"low (size x{VIX_SIZE_MULT})"
        print(f"\n📊 VIX regime: prev {st['vix_prev']:.2f} vs {VIX_MEDIAN_WINDOW}d median "
              f"{st['median']:.2f} -> {tag}")

    def _roll_session(self, new_date, tickers):
        """New ET trading day: zero the per-day flow state, re-seed, re-poll GEX,
        matching the backtest's daily reset."""
        print(f"\n🗓️  [SESSION ROLLOVER] {self.session_date} -> {new_date}. Resetting daily state...")
        self.session_date = new_date
        self._macro_am_warned = set()
        self.flow_tracker.reset()
        for t in tickers:
            self.cumulative_flow[t] = 0.0
            self.processed_ticks[t] = set()
            self.processed_tick_order[t] = deque()
        if self.uw_enabled:             # manual-only: nothing UW to reset or poll
            # yesterday's REST hand-overs do not carry into today
            if hasattr(self.uw, "reset_session"):
                self.uw.reset_session()
            for t in tickers:
                self.rest_minute_vals[t] = {}
                try:
                    self._seed_flow(t)
                except Exception as e:
                    print(f"  ⚠️ re-seed {t} failed: {e}")
            self._uw_session_polls(tickers)

        # 🚨 REBUILD THE OPTIONS BASKET. Everything above resets per-day
        # STATE; the basket is per-day INVENTORY and was never rebuilt here.
        # Under the old cron schedule the process died at 16:15 and restarted
        # at 09:28, so a fresh basket came free with the new process and this
        # gap was invisible. Running continuously under systemd it is not:
        # observed 2026-09-25, a bot up since the previous afternoon still
        # held the PREVIOUS session's 0DTE contracts. They had expired, so
        # live_option_quotes had no bid/ask for any of them, the viewer's
        # strike strip was empty, and _find_zero_latency_contract could not
        # have found a tradeable contract all day. The bot was running and
        # unable to trade.
        #
        # Drop the old subscriptions first -- except anything actually held,
        # for the same reason _rebalance_basket protects them: losing the
        # quote on an open position blinds the exit.
        held = {s.get("option_id") for s in
                (getattr(self, "active_snipes", None) or {}).values()}
        held |= {p.get("option_id") for p in
                 (getattr(self, "manual_holds", None) or {}).values()}
        for t in tickers:
            for occ in (self.active_ticker_occs.get(t) or set()):
                if occ in held:
                    continue
                try:
                    self.webull.unsubscribe_from_option(occ)
                except Exception:
                    pass
            self.active_ticker_occs[t] = set()
            self.options_basket.pop(t, None)
            self.anchor_prices.pop(t, None)
        self._initialize_options_basket(tickers)

    # =====================================================================
    # ORDER-FILL CONFIRMATION & POSITION RECONCILIATION
    # =====================================================================
    def _await_fill(self, coid, timeout=10.0, poll=1.5):
        """Poll an order until terminal or timeout. Returns (status, fill_price, filled_qty).
        status None => the status check itself kept failing."""
        deadline = time.time() + timeout
        last = ("UNKNOWN", 0.0, 0.0)
        while time.time() < deadline:
            s = self.webull.get_order_status(coid)
            st = s.get("status")
            last = (st, s.get("fill_price", 0.0), s.get("filled_qty", 0.0))
            if st in ("FILLED", "CANCELLED", "FAILED"):
                return last
            if st == "PARTIAL_FILLED" and s.get("filled_qty", 0) > 0:
                return last
            time.sleep(poll)
        return last

    def _shutdown_flatten(self):
        """Close open positions on shutdown -- and in LIVE mode actually SEND the order.

        🚨 THE BUG THIS REPLACES. The old handler called _log_exit directly for
        every open position. In DRY_RUN that is right: there is no position, so
        marking the ledger flat is the whole job. In LIVE it wrote "flat" to the
        ledger while the broker still had the position on, with nothing left
        running to manage it -- the worst possible failure, because the record
        says the risk is gone.

        THE RULE HERE: never log an exit that did not happen. If the order is not
        accepted, or does not confirm, the position stays in `active_snipes`, the
        ledger keeps it OPEN, and the operator is told in plain terms. A loud
        unknown beats a quiet lie.

        The order is deliberately NOT cancelled on timeout. It is a closing
        order; cancelling it guarantees the position stays open, whereas leaving
        it working may still get us out after the process exits.
        """
        for tk, snipe in list(self.active_snipes.items()):
            if snipe.get("exiting"):
                print(f"  ⏳ {tk}: exit already working — leaving it in place.")
                continue
            bid, ask = self.webull.get_live_option_quote(snipe["option_id"])
            if bid <= 0:
                # bad-tick lesson: do NOT price off a zero quote
                bid = snipe.get("last_good_bid", 0.0) or snipe.get("last_bid", 0.0)
                ask = snipe.get("last_ask", bid)

            if DRY_RUN:
                mark = bid if bid > 0 else snipe.get("entry_price", 0.0)
                self._log_exit(tk, snipe, round(mark, 2), "SHUTDOWN")
                self.active_snipes.pop(tk, None)
                continue

            if bid <= 0:
                print(f"  🚨 {tk}: NO USABLE QUOTE — no exit sent and NOTHING logged. "
                      f"THE POSITION IS STILL OPEN ({snipe['option_id']}). Flatten manually.")
                continue
            spread = max(0.0, ask - bid) if ask > bid else 0.0
            # A shutdown flatten has no directional trigger, so it pays the
            # plain marketable cost, not the adverse cap -- and it is floored,
            # because `max(0.01, …)` here would dump the book on the way out.
            px = max(max(0.01, round(bid * EXIT_FLOOR_FRAC, 2)),
                     round(bid - spread * FILL_COST, 2))
            qty = int(snipe.get("size", 1) or 1)
            print(f"  ⚡ [SHUTDOWN] {tk} SELL {snipe['option_id']} x{qty} @ limit ${px:.2f}")
            res = self.webull.place_option_order(
                ticker=tk, option_id=snipe["option_id"], action="SELL",
                quantity=qty, is_closing=True, order_type="LIMIT", limit_price=px,
            )
            coid = res.get("client_order_id")
            if not res.get("accepted") or not coid:
                print(f"  🚨 {tk}: shutdown exit NOT ACCEPTED — nothing logged. "
                      f"THE POSITION IS STILL OPEN. Flatten manually.")
                continue
            status, fill_px, fqty = self._await_fill(coid, timeout=SHUTDOWN_EXIT_S)
            if status in ("FILLED", "PARTIAL_FILLED") and fqty > 0:
                self._log_exit(tk, snipe, float(fill_px or px), "SHUTDOWN")
                self.active_snipes.pop(tk, None)
                print(f"  ✅ {tk} flattened @ ${fill_px or px:.2f}")
            else:
                print(f"  🚨 {tk}: shutdown exit did NOT confirm (status {status}) — "
                      f"order {coid} LEFT WORKING and nothing logged. Treat the "
                      f"position as OPEN until you have checked the broker.")

    def _chase_entry(self, ticker, option_id, qty, px, ask0):
        """Walk a resting BUY up toward the offer instead of abandoning it.

        The old behaviour placed once at min(mid+0.01, ask), waited 10s, then
        cancelled and dropped the trade. Against the real NBBO path that filled
        only 27% of paper entries, and the ones it did catch were adversely
        selected -- a resting buy fills when the option CHEAPENS, i.e. when the
        underlying has already moved against the position.

        Each rung: place, wait ENTRY_CHASE_S, and if unfilled cancel and re-price
        one tick nearer the offer, re-reading the quote so we chase the CURRENT
        ask rather than a stale one.

        CEILING = ask0 * (1 + ENTRY_CHASE_MAX_SLIP), where ask0 is the offer at
        SIGNAL TIME. With the default slip of 0 we will pay up to that ask and
        not a cent more. If the market runs away the trade is abandoned, because
        the entry was priced off that quote and chasing a departing offer is how
        a sniper becomes a buyer of tops.

        -> (status, fill_price, filled_qty). status "NOT_FILLED" means every rung
        expired; None means the status API failed and the caller must decide.
        """
        ceiling = round(ask0 * (1.0 + ENTRY_CHASE_MAX_SLIP), 2)
        px = min(round(px, 2), ceiling)
        for rung in range(max(1, ENTRY_CHASE_ROUNDS)):
            res = self.webull.place_option_order(
                ticker=ticker, option_id=option_id, action="BUY",
                quantity=qty, order_type="LIMIT", limit_price=px,
            )
            coid = res.get("client_order_id")
            if not res.get("accepted") or not coid:
                return "REJECTED", 0.0, 0.0
            status, fill_px, fqty = self._await_fill(coid, timeout=ENTRY_CHASE_S)
            if status in ("FILLED", "PARTIAL_FILLED") and fqty > 0:
                if rung:
                    print(f"  ⤴️ {ticker} entry filled on rung {rung + 1} @ ${fill_px or px:.2f}")
                return status, fill_px, fqty
            if status is None:
                return None, 0.0, 0.0          # unconfirmed: caller flags it
            self.webull.cancel_option_order(coid)
            if px >= ceiling - 1e-9:
                print(f"  ⏱️ {ticker} entry unfilled at the ceiling ${ceiling:.2f} "
                      f"after {rung + 1} rung(s) — abandoning.")
                return "NOT_FILLED", 0.0, 0.0
            nxt = min(round(px + ENTRY_CHASE_TICK, 2), ceiling)
            bid_n, ask_n = self.webull.get_live_option_quote(option_id)
            if ask_n and ask_n > 0:
                nxt = min(nxt, round(ask_n, 2))   # never bid through the offer
            if nxt <= px:
                # the offer has come to us and we still did not fill -- another
                # rung at the same price buys nothing
                print(f"  ⏱️ {ticker} entry stalled at ${px:.2f} (ask ${ask_n:.2f}) "
                      f"— abandoning.")
                return "NOT_FILLED", 0.0, 0.0
            px = nxt
            if rung + 1 < max(1, ENTRY_CHASE_ROUNDS):
                print(f"  ⤴️ {ticker} entry chase rung {rung + 2}: ${px:.2f} "
                      f"(ceiling ${ceiling:.2f})")
        print(f"  ⏱️ {ticker} entry unfilled after {ENTRY_CHASE_ROUNDS} rungs "
              f"(~{ENTRY_CHASE_ROUNDS * ENTRY_CHASE_S:.0f}s) — abandoning.")
        return "NOT_FILLED", 0.0, 0.0

    @staticmethod
    def _exit_cushion_spreads(ticker, reason):
        """Spreads of slippage to charge, from sim_core's MEASURED model.

        Keyed on WHY we are selling, not on whether the trade is green: a
        trail/stop fires precisely because the bid is falling, so it keeps
        falling while the order travels. EOD/time/blind flattens have no
        directional trigger and pay the plain marketable cost. Matching on the
        reason PREFIX matters -- "EOD STOP" and "TIME STOP" both contain "STOP"
        but neither is adverse.
        """
        r = (reason or "").upper()
        adverse = r.startswith("TRAIL") or r.startswith("STOP LOSS")
        if not adverse:
            return FILL_COST
        cap = CUSHION_CAP.get(ticker)
        if cap is None:
            # A missing ticker is NOT a neutral default (sim_core:177).
            print(f"  ⚠️ {ticker} has no measured CUSHION_CAP — falling back to "
                  f"1.5 spreads. Run measure_cushion_cap.py.")
            return 1.5
        return cap

    def _fire_exit(self, ticker, snipe, bid, ask, is_taking_profit, reason):
        """Place the closing SELL and (live) mark the snipe as 'exiting' for follow-up."""
        spread = max(0.0, ask - bid)
        mult = self._exit_cushion_spreads(ticker, reason)
        raw = round(bid - spread * mult, 2)
        # FLOOR, not max(0.01, …): a cushion that eats most of the bid is the
        # arithmetic failing, not a price. Clamping loudly beats booking -99%.
        floor = max(0.01, round(bid * EXIT_FLOOR_FRAC, 2))
        px = max(floor, raw)
        if raw < floor:
            print(f"  ⚠️ [{reason}] {ticker} cushion {mult:.2f}x spread "
                  f"${spread:.2f} would price at ${raw:.2f} vs bid ${bid:.2f} — "
                  f"clamped to ${px:.2f}. Spread is pathological for this bid.")
        print(f"  ⚡ [{reason}] {ticker} SELL {snipe['option_id']} @ limit ${px:.2f}")

        if DRY_RUN:
            # A marketable SELL limit does not fill AT the limit -- it fills
            # against the resting bid (86.4% at exactly the bid; sim_core:195).
            # Logging `px` is what made the paper ledger fiction: it recorded
            # the price we were WILLING to accept as the price we GOT, so an
            # aggressive limit booked a catastrophic fill that never happened.
            fill = min(bid, max(px, round(bid - spread * mult, 2))) if bid > 0 else px
            fill = max(0.01, round(fill, 2))
            self._log_exit(ticker, snipe, fill, reason,
                           bid=bid, ask=ask, limit_px=px)
            if self.allocator:
                self.allocator.release_trade(snipe.get("capital_committed", 0.0))
            del self.active_snipes[ticker]
            return

        res = self.webull.place_option_order(
            ticker=ticker, option_id=snipe['option_id'], action="SELL",
            quantity=int(snipe.get("size", 1) or 1),
            is_closing=True, order_type="LIMIT", limit_price=px,
        )
        coid = res.get("client_order_id")
        if not coid:
            print(f"  🚨 {ticker} exit order not accepted — brackets will re-fire next loop.")
            return
        snipe.update(exiting=True, exit_coid=coid, exit_placed_at=time.time(),
                     exit_retries=0, exit_is_tp=is_taking_profit, exit_reason=reason)

    def _manage_exit_order(self, ticker, snipe):
        """Follow up an in-flight exit: confirm fill, or cancel+re-price if it stalls."""
        s = self.webull.get_order_status(snipe["exit_coid"])
        st = s.get("status")

        if st == "FILLED" or (st == "PARTIAL_FILLED" and s.get("filled_qty", 0) > 0):
            fill_px = float(s.get("fill_price") or 0.0)
            print(f"  ✅ {ticker} exit FILLED @ ${fill_px:.2f}")
            self._log_exit(ticker, snipe, fill_px, snipe.get("exit_reason", "EXIT"))
            if self.allocator:
                self.allocator.release_trade(snipe.get("capital_committed", 0.0))
            del self.active_snipes[ticker]
            return
        # 🚨 RE-PRICE = CANCEL -> CONFIRMED -> PLACE. This used to cancel and
        # place in the same breath. Until the broker confirms the cancel, the
        # contract is still committed to the old SELL and a new one is refused
        # ("in excess of current holding quantity") -- the manual close hit
        # exactly that on 2026-09-28 (manual_orders.EXIT_REPLACE_WAIT_S).
        # The replacement now goes out only once the old order reads CANCELLED.
        if snipe.get("replace_pending"):
            if st in ("CANCELLED", "FAILED"):
                if time.time() - snipe.get("replace_try_at", 0) < 1.0:
                    return
                snipe["replace_try_at"] = time.time()
                self._place_exit_retry(ticker, snipe)
            elif time.time() - snipe.get("cancel_at", 0) > 6:
                self.webull.cancel_option_order(snipe["exit_coid"])
                snipe["cancel_at"] = time.time()
            return

        if st in ("CANCELLED", "FAILED"):
            print(f"  ↩️ {ticker} exit order {st}; re-arming brackets.")
            snipe["exiting"] = False
            return

        if time.time() - snipe.get("exit_placed_at", 0) < 20:
            return  # give it time

        retries = snipe.get("exit_retries", 0)
        bid, ask = self.webull.get_live_option_quote(snipe['option_id'])
        if bid == 0.0 and retries < 3:
            return  # no quote to re-price against yet
        self.webull.cancel_option_order(snipe["exit_coid"])
        snipe.update(replace_pending=True, cancel_at=time.time(), replace_try_at=0)

    def _place_exit_retry(self, ticker, snipe):
        """Send the re-priced exit once the old one is confirmed cancelled.
        Rejections are retried (>= 1s apart, 5 tries), then the position goes
        back to the bracket loop, which will fire a fresh exit."""
        retries = snipe.get("exit_retries", 0)
        bid, ask = self.webull.get_live_option_quote(snipe['option_id'])
        spread = max(0.0, ask - bid)
        if retries >= 3:
            px = 0.01
            print(f"  🛑 {ticker} exit stuck ({retries}x) — routing marketable dump @ $0.01.")
        else:
            px = max(0.01, round(bid - spread * (1.5 + retries), 2))
            print(f"  🔁 {ticker} exit retry {retries + 1} @ ${px:.2f}")
        res = self.webull.place_option_order(
            ticker=ticker, option_id=snipe['option_id'], action="SELL",
            quantity=int(snipe.get("size", 1) or 1),
            is_closing=True, order_type="LIMIT", limit_price=px,
        )
        coid = res.get("client_order_id")
        if coid and res.get("accepted") is not False:
            snipe.pop("replace_pending", None)
            snipe.pop("replace_rejects", None)
            snipe.update(exit_coid=coid, exit_placed_at=time.time(), exit_retries=retries + 1)
            return
        # rejected: a coid is minted locally, so `accepted` is the test
        n = snipe.get("replace_rejects", 0) + 1
        snipe["replace_rejects"] = n
        print(f"  ⚠️ {ticker} re-priced exit REJECTED ({n}/5): {res.get('error') or 'rejected'}")
        if n >= 5:
            print(f"  🚨 {ticker} exit rejected 5x — back to the bracket loop; "
                  f"CHECK THE BROKER")
            snipe.pop("replace_pending", None)
            snipe.pop("replace_rejects", None)
            snipe["exiting"] = False

    def _reconcile_startup(self, tickers):
        """Get flat & clean before trading: cancel stale watchlist option orders and
        FLATTEN any option position the bot has no record of (restart mid-trade,
        manual position).

        🚨 THE THREE HALVES HAVE DIFFERENT DRY_RUN RULES.
        Adoption places nothing and is how a discretionary position becomes
        visible to the entry block, the 15:55 flatten and the viewer, so it
        runs ALWAYS -- a paper bot that cannot see your real position is worse
        than useless, it is misleading. Cancelling and flattening both send
        orders, so those stay live-only.
        """
        print("\n🧭 [STARTUP RECONCILE] Auditing Webull for unknown orders / positions...")
        wl = set(tickers)

        # ---- adopt first, unconditionally ----
        for p in self.webull.get_open_option_positions():
            und, occ = p.get("underlying"), p.get("occ")
            qty = p.get("quantity", 0)
            if und in manual_orders.ALLOWED and occ and qty:
                manual_orders.adopt(self, und, occ, qty, p.get("cost_price"))
                # 🚨 SUBSCRIBE UNCONDITIONALLY, NOT ONLY ON A FIRST ADOPT.
                # This used to sit inside `if adopt(...)`. On a RESTART the
                # hold is rehydrated from disk, so adopt() returns False as a
                # no-op and the subscribe was skipped -- leaving a held
                # contract with no MQTT quote. Observed 2026-09-23: an IWM
                # 284C drifted three strikes out of the ATM+/-2 basket, so it
                # was not in the basket either, and the 15:55 flatten priced
                # off a missing bid and fell to its 0.01 floor. A position we
                # are responsible for exiting must ALWAYS be priceable.
                self.webull.subscribe_to_option(occ)

        if DRY_RUN:
            print("  📋 DRY_RUN — adoption done; no orders cancelled or "
                  "flattened.")
            print("  ✅ Startup reconcile complete.\n")
            return

        # Contracts we just adopted keep their resting orders. A TP/SL bracket
        # set by hand in the app rests as two SELLs on the held contract, and
        # wiping it at startup would silently delete protection the user put
        # there on purpose. They are cancelled at EXIT time instead, by
        # manual_orders.cancel_resting, which is when they actually conflict.
        protect = {}                    # client_order_id -> the contract it guards
        for p in (getattr(self, "manual_holds", None) or {}).values():
            oid = p.get("option_id")
            if oid:
                for r in manual_orders.resting_orders(self, oid):
                    protect[r["coid"]] = f"{oid} ({r['kind']})"

        for o in self.webull.get_open_orders():
            if "OPTION" not in o.get("instrument_type", ""):
                continue
            sym = str(o.get("symbol") or "")
            m = re.match(r'^([A-Za-z]+)', sym)
            und = m.group(1).upper() if m else sym.upper()
            coid = o.get("client_order_id")
            if und not in wl or not coid:
                continue
            if coid in protect:
                print(f"  🛡️ Keeping your resting order on {protect[coid]} — it "
                      f"protects an adopted position, and is cancelled "
                      f"automatically when we exit.")
                continue
            print(f"  ↩️ Cancelling stale open order {coid} ({sym})")
            self.webull.cancel_option_order(coid)

        time.sleep(1.0)

        for p in self.webull.get_open_option_positions():
            und = p.get("underlying")
            occ, qty = p.get("occ"), p.get("quantity", 0)
            if und not in wl or not occ or qty == 0:
                continue
            # 🚨 ADOPT, DO NOT FLATTEN, ON THE MANUAL TICKERS.
            # Flattening is right for a stale BOT position left by a crash. It
            # is exactly wrong for a discretionary position opened on purpose
            # in the broker app -- starting the bot would sell it out from
            # under you, and the config you need for manual trading
            # (DRY_RUN=false) is the very one that arms this path. Adopting
            # gives it the entry block, the 15:55 flatten and the viewer's
            # exit buttons instead. Everything else still gets flattened.
            if und in manual_orders.ALLOWED:
                continue                       # already adopted above
            bid, ask = self.webull.get_live_option_quote(occ)
            if bid <= 0:
                print(f"  ⚠️ Unknown {und} position {qty}x {occ} but no quote — CANCEL MANUALLY.")
                continue
            spread = max(0.0, ask - bid)
            # same reasoning as the shutdown flatten: no directional trigger,
            # and floored so a wide book cannot turn this into a penny dump
            px = max(max(0.01, round(bid * EXIT_FLOOR_FRAC, 2)),
                     round(bid - spread * FILL_COST, 2))
            print(f"  🚑 Unknown {und} position {qty}x {occ} — FLATTENING @ ${px:.2f}")
            self.webull.place_option_order(
                ticker=und, option_id=occ, action="SELL", quantity=int(abs(qty)),
                is_closing=True, order_type="LIMIT", limit_price=px,
            )
        print("  ✅ Startup reconcile complete.\n")

    def _ingest_flow(self, ticker: str, flow_ticks) -> float:
        """Fold one batch of flow ticks into cumulative_flow[ticker]; return it.

        🚨 REST AND WEBSOCKET TICKS ARE COUNTED DIFFERENTLY, ON PURPOSE.
        REST net-prem-ticks returns the WHOLE DAY on every poll, one row per
        minute. It used to go through the same id-set as the socket, keyed on
        f"{time}_{value}", which failed three ways once REST became the only
        source (UW Basic, 2026-09-27):
          1. the startup seed summed the day but marked nothing counted, so the
             first poll added the whole day AGAIN -- any mid-day restart doubled
             the day's flow so far;
          2. a minute still in progress, if its value updates between polls,
             gets a new id each time and is added once per value;
          3. the id set forgets its oldest entries past 500, after which the
             next full-day poll re-adds the forgotten ticks.
        So REST is kept as a per-MINUTE ledger: a new minute adds its value, a
        changed minute adds only the change. Idempotent under full-day re-polls
        and exact whether or not in-progress minutes move.

        A source hand-over (the socket went quiet; see
        UnusualWhalesClient.get_live_net_premium) bumps the client's
        flow_epoch. On seeing a new epoch the day is rebuilt from zero out of
        the REST response, which covers the whole day -- so ticks printed while
        the socket was down are recovered instead of lost.
        """
        epoch = getattr(self.uw, "flow_epoch", {}).get(ticker, 0)
        if epoch != self._flow_epoch_seen.get(ticker, 0):
            self._flow_epoch_seen[ticker] = epoch
            self.cumulative_flow[ticker] = 0.0
            self.rest_minute_vals[ticker] = {}
            self.processed_ticks[ticker] = set()
            self.processed_tick_order[ticker] = deque()
            print(f"  🔁 [FLOW] {ticker}: source hand-over -- rebuilding today's "
                  f"cumulative flow from REST.")

        if self.uw.flow_source(ticker) == "rest":
            ledger = self.rest_minute_vals.setdefault(ticker, {})
            for tick in flow_ticks:
                key = tick.get("time")
                if not key:
                    continue
                v = float(tick.get("net_premium", 0) or 0)
                old = ledger.get(key)
                if old is None:
                    self.cumulative_flow[ticker] += v
                elif v != old:
                    self.cumulative_flow[ticker] += v - old
                ledger[key] = v
        else:
            for tick in flow_ticks:
                tick_id = f"{tick.get('time')}_{tick.get('net_premium')}"
                if tick_id not in self.processed_ticks[ticker]:
                    self.cumulative_flow[ticker] += float(tick.get("net_premium", 0))
                    self._mark_tick_processed(ticker, tick_id)
        return self.cumulative_flow[ticker]

    def _uw_session_polls(self, tickers):
        """The once-per-session UW reads, in their original order -- shared by
        startup and the daily rollover, and skipped together in manual-only."""
        self._poll_daily_gex(tickers)
        self._poll_session_regimes(tickers)
        self._poll_session_dex(tickers)
        self._poll_session_vix()
        self._seed_price_history(tickers)
        self._seed_amt_profiles(tickers)

    def _seed_flow(self, ticker: str) -> float:
        """Seed today's cumulative flow from REST AND record which minutes that
        covered, so the first REST poll does not count them again."""
        total = self.uw.seed_daily_cumulative_flow(ticker)
        self.cumulative_flow[ticker] = total
        self.rest_minute_vals[ticker] = dict(
            getattr(self.uw, "seed_minutes", {}).get(ticker, {}))
        return total

    def _mark_tick_processed(self, ticker: str, tick_id: str):
        """Record a tick id, evicting the oldest once we exceed the cap."""
        self.processed_ticks[ticker].add(tick_id)
        order = self.processed_tick_order[ticker]
        order.append(tick_id)
        while len(order) > 500:
            evicted = order.popleft()
            self.processed_ticks[ticker].discard(evicted)

    def _arm_capital_allocator(self):
        """Build the CRO gate.

        DRY_RUN: use a fixed paper bankroll (PAPER_EQUITY env, default $30k) for
        BOTH cash and equity so the capital gate is deterministic and realistic
        regardless of what's actually in the linked Webull account (which may be
        $0 / market-data-only on the paper box). Live: pull real Webull cash,
        falling back to ACCOUNT_EQUITY when the SDK can't return a balance.
        """
        if DRY_RUN:
            paper = float(os.getenv("PAPER_EQUITY", os.getenv("ACCOUNT_EQUITY", "30000")) or 30000)
            self.account_equity = paper
            self.allocator = CapitalAllocator(
                available_webull_cash=paper, available_hl_cash=0.0, total_equity=paper,
            )
            print(f"  💼 [PAPER] Capital Allocator armed on ${paper:,.2f} pretend bankroll "
                  f"(set PAPER_EQUITY to change).")
            return

        wb_cash = 0.0
        try:
            health = self.webull.get_account_health() or {}
            wb_cash = float(
                health.get('total_cash_balance',
                health.get('dayBuyingPower',
                health.get('overnightBuyingPower',
                health.get('buying_power', 0)))) or 0
            )
        except Exception as e:
            print(f"  ⚠️ Could not read Webull account health: {e}")

        env_equity = float(os.getenv("ACCOUNT_EQUITY", "0") or 0)
        if wb_cash <= 0.0:
            wb_cash = env_equity
            print(f"  ⚠️ Webull cash unavailable — falling back to ACCOUNT_EQUITY=${wb_cash:,.2f}")

        self.account_equity = env_equity if env_equity > 0 else wb_cash
        self.allocator = CapitalAllocator(
            available_webull_cash=wb_cash,
            available_hl_cash=0.0,
            total_equity=self.account_equity,
        )
        print(f"  💼 Capital Allocator armed | Webull cash: ${wb_cash:,.2f} | Equity basis: ${self.account_equity:,.2f}")

    def get_spot_price(self, ticker: str) -> float:
        """ZERO-LATENCY GETTER: Reads directly from Webull MQTT memory."""
        bars = self.webull.tick_builder.live_bars
        if ticker in bars:
            return float(bars[ticker]['close'])
        return 0.0

    def _determine_required_dtes(self, ticker: str) -> dict:
        """Which (direction, DTE) combos the options basket must prep for this
        ticker. RULES mode: union over the enabled config.RULES entries. Legacy:
        the WATCHLIST GEX tree."""
        required = {"CALL": set(), "PUT": set()}

        if USE_RULES:
            for r in RULES:
                if r.get("enabled", True) and r["ticker"] == ticker:
                    required[r["direction"]].update(r.get("dte", [0, 1]))
            self._add_manual_extras(ticker, required)
            return required

        settings = WATCHLIST.get(ticker, {})
        for regime in ["POSITIVE_GEX", "NEGATIVE_GEX"]:
            for direction in ["CALL", "PUT"]:
                rules = settings.get(regime, {}).get(direction)
                if rules and "dte" in rules:
                    for dte in rules["dte"]:
                        required[direction].add(dte)
        self._add_manual_extras(ticker, required)
        return required

    def _add_manual_extras(self, ticker: str, required: dict):
        """Fold in directions tracked only so they can be traded BY HAND.

        SPY/QQQ/IWM are CALL-only in config.RULES, so their puts were never
        subscribed and the viewer's strike strip had an empty put side. These
        contracts are inventory, not permission: the scanner picks its
        direction from flow momentum and then demands an enabled rule for
        (ticker, direction) via _match_rule, so a put with no put rule can
        never be entered. See manual_orders.EXTRA_DIRECTIONS.

        Never raises -- a failure here must not stop the basket being built.
        """
        try:
            import manual_orders
            for d, dtes in manual_orders.extra_dtes(ticker, required).items():
                required[d].update(dtes)
        except Exception as e:                  # noqa: BLE001 -- deliberate
            print(f"  ⚠️ manual chain extras for {ticker}: {type(e).__name__}: {e}")

    def _build_basket_from_chain(self, ticker: str, spot: float, chain_data: dict, required_specs: dict):
        """Helper function to isolate the ATM +/- 2 strikes for required DTEs."""
        new_basket = {"CALL": {}, "PUT": {}}
        new_occs = set()
        
        for direction in ["CALL", "PUT"]:
            for dte in required_specs[direction]:
                new_basket[direction][dte] = []
                target_exp_str = None
                valid_contracts = []
                
                today_et = market_now().date()
                for c in chain_data.get("data", []):
                    exp_str = c.get("expireDate", "0000-00-00")
                    if exp_str == "0000-00-00": continue

                    try:
                        # EXACT calendar-day DTE, matching the backtest's
                        # `(expiry_date - trade_date).days`. The old +/-1.5-day
                        # tolerance could not distinguish a 0DTE from a 1DTE.
                        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                        if (exp_date - today_et).days == dte:
                            target_exp_str = exp_str
                            valid_contracts.append(c)
                    except Exception: pass

                if not valid_contracts: continue
                
                # Filter by Call/Put and sort by strike
                cp_filter = "Call" if direction == "CALL" else "Put"
                contracts = sorted([c for c in valid_contracts if c.get("callPut") == cp_filter], key=lambda x: float(x.get("strikePrice")))
                
                if not contracts: continue
                
                # Find ATM and grab +/- 2 strikes
                closest_idx = min(range(len(contracts)), key=lambda i: abs(float(contracts[i].get("strikePrice")) - spot))
                start_idx = max(0, closest_idx - 2)
                end_idx = min(len(contracts), closest_idx + 3)
                
                for c in contracts[start_idx:end_idx]:
                    occ = c.get("symbol")
                    new_basket[direction][dte].append({
                        "occ": occ, 
                        "strike": float(c.get("strikePrice")), 
                        "iv": float(c.get("impliedVolatility", 0.5))
                    })
                    new_occs.add(occ)
                    
        return new_basket, new_occs

    def _initialize_options_basket(self, tickers: list):
        """Runs ONCE at boot. Scans option chains and builds initial ATM basket."""
        print("\n🧺 INITIALIZING MORNING OPTIONS BASKET (Zero-Latency Prep)...")
        for ticker in tickers:
            required_specs = self._determine_required_dtes(ticker)
            if not required_specs["CALL"] and not required_specs["PUT"]: continue
            
            spot = 0.0
            retries = 0
            while spot == 0.0 and retries < 5:
                spot = self.get_spot_price(ticker)
                time.sleep(1)
                retries += 1
                
            if spot == 0.0: continue

            # --- USING THE HIGH-SPEED PRE-FILTER ---
            # basket_chain: symbols only, cached directory -- see its docstring
            chain_data = self.webull.basket_chain(
                ticker, spot, max_dte=max([0, *required_specs["CALL"], *required_specs["PUT"]]))
            if not chain_data: continue

            new_basket, new_occs = self._build_basket_from_chain(ticker, spot, chain_data, required_specs)
            
            self.options_basket[ticker] = new_basket
            self.active_ticker_occs[ticker] = new_occs
            self.anchor_prices[ticker] = spot
            self.is_rebalancing[ticker] = False
                    
            print(f"  ├─ {ticker}: Subscribing to {len(new_occs)} Plausible Strikes via MQTT... (Anchor: ${spot:.2f})")
            for occ in new_occs:
                self.webull.subscribe_to_option(occ)

    def _rebalance_basket(self, ticker: str, spot: float):
        """Background thread: re-centre the basket on `spot` (see the strike
        drift rebalancer in run_brain for when)."""
        try:
            print(f"  🔄 [BASKET REBALANCE] {ticker} drifted to ${spot:.2f}. Shifting strikes...")
            required_specs = self._determine_required_dtes(ticker)
            
            # --- USING THE HIGH-SPEED PRE-FILTER ---
            # basket_chain: symbols only, cached directory -- see its docstring
            chain_data = self.webull.basket_chain(
                ticker, spot, max_dte=max([0, *required_specs["CALL"], *required_specs["PUT"]]))
            if not chain_data: return
                
            new_basket, new_occs = self._build_basket_from_chain(ticker, spot, chain_data, required_specs)
            old_occs = self.active_ticker_occs.get(ticker, set())
            
            # SMART DIFFING (No Thrashing)
            to_add = new_occs - old_occs
            to_drop = old_occs - new_occs
            
            for occ in to_add:
                self.webull.subscribe_to_option(occ)
                
            held = {s['option_id'] for s in self.active_snipes.values()}
            # 🚨 MANUAL HOLDS COUNT AS HELD. A drift past 1.5% shifts the basket
            # and drops the old strikes; without this, a discretionary position
            # loses its MQTT quote while still open. manual_orders.flatten then
            # finds no bid and falls back to a 1c limit, and the viewer's Active
            # Hold goes blind on the one position nothing else is managing.
            held |= {p.get("option_id")
                     for p in (getattr(self, "manual_holds", None) or {}).values()}
            for occ in to_drop:
                # Never unsubscribe if we have an active snippet holding this contract!
                if occ not in held:
                    self.webull.unsubscribe_from_option(occ)
                
            self.options_basket[ticker] = new_basket
            self.active_ticker_occs[ticker] = new_occs
            self.anchor_prices[ticker] = spot
            
            print(f"  ✅ [BASKET REBALANCE] {ticker} completed. Added {len(to_add)}, Dropped {len(to_drop)}. New Anchor: ${spot:.2f}")
        except Exception as e:
            print(f"  🚨 Rebalance Exception for {ticker}: {e}")
        finally:
            self.is_rebalancing[ticker] = False

    def _find_zero_latency_contract(self, ticker: str, direction_str: str, dte: int,
                                    spot_price: float, offset: int = 0):
        """Pick the contract `offset` strikes OTM from ATM for this DTE (offset 0 =
        ATM, the default -- matches the backtest's nearest-strike choice). CALL
        goes to higher strikes, PUT to lower. Clamped to the quoted range.
        `strike_offset` per-rule field: GLD amp1 CALL uses 1 (check_strike_selection:
        OTM+1 survives realistic ask/bid fills, +7pp OOS). Penny/spread filters
        stay on the caller, against the entry mid, exactly like the backtest."""
        basket = self.options_basket.get(ticker, {}).get(direction_str, {}).get(dte, [])
        if not basket: return None

        quoted = []
        for leg in basket:
            bid, ask = self.webull.get_live_option_quote(leg["occ"])
            if ask <= 0.0: continue  # no live quote yet
            quoted.append({"symbol": leg["occ"], "strike": leg["strike"], "ask": ask, "bid": bid})
        if not quoted:
            return None

        quoted.sort(key=lambda c: c["strike"])
        atm_i = min(range(len(quoted)), key=lambda i: abs(quoted[i]["strike"] - spot_price))
        j = atm_i + offset if direction_str == "CALL" else atm_i - offset
        return quoted[max(0, min(j, len(quoted) - 1))]

    def log_trade(self, ticker, action, option_id, price, size, reason, flow_val, regime, extra=None):
        log_entry = {
            "timestamp": market_now().isoformat(),
            "ticker": ticker,
            "contract": option_id,
            "action": action,
            "price": round(price, 2),
            "size": size,
            "reason": reason,
            "net_flow": flow_val,
            "regime": regime,
            "dry_run": DRY_RUN,
        }
        if extra:
            log_entry.update(extra)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        # ---- mirror into the counterfactual ledger -------------------------
        # At execution time there IS no counterfactual: nothing is known to be
        # distorted yet, so the mirrored row is the SAME event marked observed.
        # It only diverges later, when build_counterfactual.py replays a session
        # whose records a bug corrupted and REPLACES those rows with fiction.
        # Writing both here keeps the two ledgers aligned day to day instead of
        # the counterfactual going stale between rebuilds.
        # NOTE: build_counterfactual regenerates this file wholesale from the raw
        # log, so these appends are idempotent -- a rebuild reproduces them. Do
        # not run a rebuild mid-session; it would race with these writes.
        try:
            with open(self.cf_file, "a") as f:
                f.write(json.dumps({**log_entry, "source": "observed",
                                    "fiction": False}) + "\n")
        except Exception as e:
            print(f"  ⚠️ counterfactual mirror failed ({type(e).__name__}) — "
                  f"raw log is still authoritative.")
        print(f"💾 [DATA COLLECTION] Logged {action} for {ticker} ({option_id})")
        try:
            self.tg.trade(log_entry)
        except Exception:
            pass

    def _log_exit(self, ticker, snipe, exit_px, reason, bid=None, ask=None,
                  limit_px=None):
        """Record the round-trip: entry/exit price, realized P&L, hold time.
        Fires for every close (DRY_RUN simulated fill or live confirmed fill) so
        the paper-trade log is a complete, analysable trade ledger.

        THE QUOTE AT THE EXIT IS RECORDED (2026-09-18). Eight GLD penny exits
        could not be diagnosed from the ledger, because it stored the fill and
        nothing about the book that produced it -- "TRAIL STOP (peak $1.30,
        +21%)" filled at $0.01 when the trail level was $0.65, and the log could
        not say whether the bid had collapsed, the spread had blown out, or the
        cushion arithmetic had gone negative. Those need different fixes. Now
        the bid, ask, spread and the level that fired are all on the row."""
        entry = float(snipe.get("entry_price") or 0.0)
        try:
            hold_mins = (market_now() - snipe["entry_time"]).total_seconds() / 60.0
        except Exception:
            hold_mins = None
        size = int(snipe.get("size", 1) or 1)
        pnl_pct = ((exit_px - entry) / entry * 100.0) if entry > 0 else None
        self.log_trade(
            ticker, "EXIT", snipe.get("option_id"), exit_px, size,
            reason, None, snipe.get("regime"),
            extra={
                "entry_price": round(entry, 2),
                "exit_price": round(float(exit_px), 2),
                "contracts": size,
                "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
                "pnl_dollars": round((exit_px - entry) * 100.0 * size, 2) if entry > 0 else None,
                "pnl_dollars_per_contract": round((exit_px - entry) * 100.0, 2) if entry > 0 else None,
                "hold_mins": round(hold_mins, 1) if hold_mins is not None else None,
                "dte": snipe.get("dte"),
                "is_call": snipe.get("is_call"),
                "entry_time": snipe["entry_time"].isoformat() if snipe.get("entry_time") else None,
                "unconfirmed": snipe.get("unconfirmed", False),
                "exit_bid": round(bid, 2) if bid is not None else None,
                "exit_ask": round(ask, 2) if ask is not None else None,
                "exit_spread": (round(max(0.0, ask - bid), 2)
                                if (bid is not None and ask is not None) else None),
                "exit_limit": round(limit_px, 2) if limit_px is not None else None,
                "trail_stop_at_exit": snipe.get("last_trail_stop"),
                "peak_bid": snipe.get("max_bid_seen"),
            })

    def _arm_telegram_commands(self):
        """Read-only Telegram commands: /today /all /pnl /open /status."""
        def _cf_trips():
            """Round-trips from the AMENDED ledger, or [] if it is absent.
            Never let a missing/derived file break a read-only command."""
            try:
                if not os.path.exists(self.cf_file):
                    return []
                return round_trips(load_trades(self.cf_file))
            except Exception:
                return []

        def _today(_a):
            d = _et_date_str()
            def _on_day(ts):
                return [t for t in ts
                        if (t.get("exit_ts") or t.get("entry_ts") or "").startswith(d)]
            return summarize_both(_on_day(round_trips(load_trades(self.log_file))),
                                  _on_day(_cf_trips()), f"Today {d}")

        def _all(_a):
            return summarize_both(round_trips(load_trades(self.log_file)),
                                  _cf_trips(), "All trades")

        def _open(_a):
            snm = self.active_snipes
            if not snm:
                return "no open positions."
            out = [f"<b>{len(snm)} open</b>"]
            for tk, s in snm.items():
                d = "CALL" if s.get("is_call") else "PUT"
                out.append(f"\U0001F7E1 <b>{tk}</b> {d}  entry ${float(s.get('entry_price') or 0):.2f}  "
                           f"TP ${float(s.get('tp_limit') or 0):.2f} / SL ${float(s.get('sl_limit') or 0):.2f}  "
                           f"{s.get('dte', '?')}DTE  <i>{s.get('regime', '?')}</i>")
            return "\n".join(out)

        def _status(_a):
            mode = "PAPER (DRY_RUN)" if DRY_RUN else ("SANDBOX" if PAPER_TRADING else "LIVE")
            return (f"<b>directional bot</b> · {mode}\n"
                    f"session {self.session_date}  ·  {len(self.active_snipes)} open  ·  "
                    f"equity ${self.account_equity:,.0f}\n"
                    f"tracking {len([r for r in RULES if r.get('enabled', True)])} rules")

        def _flow(_a):
            """Where each ticker sits against its OWN live gate, right now.

            Reads FlowMomentumTracker.closed_series and _live_flow_threshold --
            the same two calls the chart renders from, so this can never
            disagree with what the bot is about to do. Runs on the Telegram
            polling thread, not the monitor loop.
            """
            out = []
            names = sorted(set(self.cumulative_flow) | set(self.active_snipes))
            for tk in names:
                rule = next((r for r in RULES if r.get("enabled", True)
                             and r["ticker"] == tk), None)
                cum = float(self.cumulative_flow.get(tk, 0.0) or 0.0)
                thr = None
                if rule is not None:
                    try:
                        thr = float(self._live_flow_threshold(tk, rule))
                    except Exception:
                        thr = None
                _m, series, ema = self.flow_tracker.closed_series(tk)
                # ▲ = cumulative flow is ABOVE its EMA(5) (a bullish crossover
                # is what put it there); ▼ = below. The arrow is the state the
                # trigger reads, not a prediction.
                side = ("▲" if series[-1] > ema[-1] else "▼") if ema else "·"
                snipe = self.active_snipes.get(tk)
                if snipe:
                    e = float(snipe.get("entry_price") or 0) or None
                    b = float(snipe.get("last_good_bid") or 0) or None
                    roe = f"{(b - e) / e * 100:+.0f}%" if (e and b) else "?"
                    tail = (f"  <b>HOLDING</b> "
                            f"{'CALL' if snipe.get('is_call') else 'PUT'} {roe}")
                else:
                    tail = ""
                if thr:
                    pc = abs(cum) / thr * 100
                    gate = (f"{pc:>3.0f}% of gate" if pc < 100
                            else f"<b>OVER GATE</b> ({pc:.0f}%)")
                    gate += f"  (±${thr/1e6:.1f}M)"
                else:
                    gate = "gate warming"
                out.append(f"{side} <b>{tk}</b> ${cum/1e6:+.1f}M · {gate}{tail}")
            if not out:
                return "no tickers tracked yet."
            head = (f"<b>flow vs gate</b> · {market_now():%H:%M} ET · "
                    f"{len(self.active_snipes)} open")
            note = ("\n<i>gate is this ticker's own trailing percentile; it "
                    "moves during the session.</i>")
            return head + "\n" + "\n".join(out) + note

        self.tg.start_commands({"today": _today, "all": _all, "pnl": _all,
                                "open": _open, "status": _status, "flow": _flow})

    def run_brain(self):
        tickers_to_track = tracked_tickers()
        print(f"🎯 Tracking {len(tickers_to_track)} ticker(s): {', '.join(tickers_to_track)}")

        if self.uw_enabled:
            print("🌊 Starting Unusual Whales flow feed (WebSocket if the plan allows, else REST)...")
            self.uw.start_multiplexer(tickers_to_track)
        else:
            print("🖐️  MANUAL-ONLY: no UW_API_KEY -- the bot's flow feed, regimes and rules "
                  "are OFF.\n    Webull quotes, the manual desk and the viewer run as normal.")

        print("🔌 Igniting Webull MQTT Engine...")
        self.webull.start_tick_stream(tickers_to_track)

        if self.uw_enabled:
            # --- SEED DAILY CUMULATIVE FLOW FOR MID-DAY STARTS ---
            print("\n🌱 Seeding Mid-Day Cumulative Flow (REST API Fallback)...")
            for ticker in tickers_to_track:
                seeded_total = self._seed_flow(ticker)
                print(f"  ├─ {ticker}: Seeded ${seeded_total:,.0f} in historical flow "
                      f"({len(self.rest_minute_vals.get(ticker, {}))} minutes).")
        self.session_date = market_now().date()
        if self.uw_enabled:
            self._uw_session_polls(tickers_to_track)
        self._arm_telegram_commands()

        time.sleep(5)

        self._initialize_options_basket(tickers_to_track)

        print("\n💼 Arming Capital Allocator...")
        self._arm_capital_allocator()

        # always: the adoption half runs in DRY_RUN too (see _reconcile_startup)
        self._reconcile_startup(tickers_to_track)

        print(f"\n🚀 ZERO-LATENCY ENGINE ACTIVE (DRY_RUN = {DRY_RUN}) | Clock: {market_now():%Y-%m-%d %H:%M:%S %Z}")
        # 🚨 SPELL THE TWO FLAGS OUT SEPARATELY. They were decoupled on
        # 2026-09-23: DRY_RUN governs the BOT's orders, MANUAL_TRADING_ARMED
        # governs orders you fire from the viewer, and "DRY_RUN = True" no
        # longer implies nothing can leave this machine. This banner is the
        # line you read to confirm you are safe, so it has to say both.
        from config import MANUAL_TRADING_ARMED as _ARMED
        print(f"   bot orders    : "
              f"{'OFF — manual-only, no UW key' if not self.uw_enabled else 'PAPER — nothing sent' if DRY_RUN else '🔴 LIVE'}")
        print(f"   manual orders : "
              f"{'🔴 ARMED — REAL CAPITAL from the viewer' if _ARMED else 'DISARMED — refused'}")
        if _ARMED and DRY_RUN:
            print("   (armed hand, paper bot — the intended rollout config)")
        print("   exits of REAL positions always send, on either flag.")
        print("=" * 65)

        while True:
            try:
                current_time = market_now()

                # --- DAILY ROLLOVER ---
                # The backtest resets cumulative flow every trading day (per-day
                # cumsum). Mirror that so a bot left running across days doesn't
                # drift from the backtest's semantics.
                if current_time.date() != self.session_date:
                    self._roll_session(current_time.date(), tickers_to_track)

                # --- ENGINE HEARTBEAT ---
                if current_time.second == 0 and current_time.minute != getattr(self, 'last_heartbeat_minute', -1):
                    self.last_heartbeat_minute = current_time.minute
                    self._record_price_minute(tickers_to_track, current_time)
                    for _atk in (self._amt_open_rules if self.uw_enabled else ()):
                        if _atk not in self._amt_open:
                            self._amt_open_state(_atk)
                    spy_spot = self.get_spot_price("SPY")
                    print(f"[{current_time.strftime('%H:%M:%S')}] 📡 Engine Nominal | SPY Spot: ${spy_spot:.2f} | Active Positions: {len(self.active_snipes)}")

                # =========================================================
                # 0. STRIKE DRIFT REBALANCER
                # =========================================================
                for ticker in tickers_to_track:
                    spot_price = self.get_spot_price(ticker)
                    if spot_price == 0.0: continue

                    # 🚨 NO BASKET AT ALL -> BUILD ONE NOW. The basket is built
                    # at startup and at the midnight rollover, both of which
                    # skip a ticker whose spot is 0. A process that has been
                    # running still holds the last close, so the rollover
                    # works; a process STARTED overnight has no tick yet and
                    # builds nothing. Observed 2026-09-28: a 00:08 restart left
                    # every basket empty, the rebalancer below measured drift
                    # from a missing anchor (0%, never fires), and the strike
                    # strip stayed blank into the session. Retried at most
                    # once a minute per ticker.
                    if ticker not in self.anchor_prices:
                        retry = getattr(self, "_basket_retry", None)
                        if retry is None:
                            retry = self._basket_retry = {}
                        if (not self.is_rebalancing.get(ticker, False)
                                and time.time() - retry.get(ticker, 0) >= 60):
                            retry[ticker] = time.time()
                            self.is_rebalancing[ticker] = True
                            print(f"  🧺 [BASKET] {ticker}: no basket yet -- building at ${spot_price:.2f}")
                            threading.Thread(target=self._rebalance_basket,
                                             args=(ticker, spot_price), daemon=True).start()
                        continue

                    # 🚨 SHIFT ON ONE STRIKE, NOT 1.5%. The basket is ATM +/- 2
                    # strikes -- about +/-0.3% on SPY -- but only re-centred
                    # after a 1.5% move (~$11.50 on SPY), so on a fast day
                    # every strike in the viewer's strip went stale long before
                    # a shift fired. Observed 2026-09-28: a manual SPY put had
                    # to be placed from the phone. Now: re-centre once spot is
                    # one strike step from the anchor (step = the basket's own
                    # strike spacing), at most once per 10s per ticker. Each
                    # shift is instant (basket_chain, cached directory) and
                    # only adds/drops the edge strikes; held contracts are
                    # never unsubscribed (_rebalance_basket).
                    anchor = self.anchor_prices.get(ticker, spot_price)
                    ks = sorted({c["strike"] for d in (self.options_basket.get(ticker) or {}).values()
                                 for lst in d.values() for c in lst})
                    gaps = [b - a for a, b in zip(ks, ks[1:]) if b > a]
                    step = sorted(gaps)[len(gaps) // 2] if gaps else anchor * 0.015
                    last = getattr(self, "_last_rebalance", None)
                    if last is None:
                        last = self._last_rebalance = {}
                    if (abs(spot_price - anchor) >= step
                            and time.time() - last.get(ticker, 0) >= 10
                            and not self.is_rebalancing.get(ticker, False)):
                        last[ticker] = time.time()
                        self.is_rebalancing[ticker] = True
                        threading.Thread(target=self._rebalance_basket, args=(ticker, spot_price), daemon=True).start()

                # =========================================================
                # 1. ACTIVE POSITION MANAGER (Client-Side Brackets)
                # =========================================================
                for ticker, snipe in list(self.active_snipes.items()):
                    # Phase A: an exit order is already working -- confirm / re-price it.
                    if snipe.get("exiting"):
                        self._manage_exit_order(ticker, snipe)
                        continue

                    # ---------------------------------------------------------
                    # QUOTE HEALTH. `get_live_option_quote` returns (0.0, 0.0)
                    # for BOTH "no bid" and "no quote at all", and this line
                    # used to be `if bid == 0.0: continue` -- which skipped the
                    # WHOLE exit evaluation on either. On 2026-09-10 two SMH
                    # 0DTE positions peaked at +74% and +144% and both filled at
                    # $0.01: the trail level ($1.68 / $1.03) was never acted on
                    # because the loop was blind and silent while the API stalled
                    # (two bot instances were running, almost certainly throttled).
                    # Being blind on an open position is an EMERGENCY, not a skip.
                    # ---------------------------------------------------------
                    bid, ask, quote_ok = self.webull.get_live_option_quote_ex(snipe['option_id'])
                    now_ts = time.time()

                    # A $0.00 BID IS NOT A PRICE -- IT IS A MISSING QUOTE
                    # WEARING A NUMBER (2026-09-18).
                    # The call SUCCEEDS and hands back bid=0.00, so quote_ok is
                    # True and it flows into the exit logic as though it were a
                    # real collapse. Seven GLD positions exited that way at
                    # $0.01 / ~-99% in one session.
                    # It is not a real collapse. Across 40,000 NBBO prints on
                    # both GLD contracts that day, nbbo_bid <= 0 occurs ZERO
                    # times -- the real bid never left 0.17-3.15, and at the
                    # exact minutes the feed said $0.00 the book was quoting
                    # 0.81, 1.04, 2.46, 2.74. Every zero was an artifact.
                    # config.py:85 already caught this conflation once ("the
                    # client returned (0.0, 0.0) for BOTH 'no bid' and 'the API
                    # call failed'"); the resolution is the same. Route it to
                    # the BLIND path, which waits QUOTE_BLIND_PANIC_S and then
                    # prices off the last GOOD quote, instead of the exit path,
                    # which books a penny 0.3 seconds later.
                    # Only when a good bid was seen earlier -- with no prior
                    # quote there is nothing to call this a deviation from.
                    if quote_ok and bid <= 0.0 and snipe.get('last_good_bid', 0.0) > 0.0:
                        quote_ok = False
                        snipe['zero_bid_quotes'] = snipe.get('zero_bid_quotes', 0) + 1

                    if quote_ok:
                        snipe['last_quote_ts'] = now_ts
                        snipe['quote_fails'] = 0
                    else:
                        snipe['quote_fails'] = snipe.get('quote_fails', 0) + 1
                        blind_s = now_ts - snipe.get('last_quote_ts', snipe.get('entry_ts', now_ts))
                        # the main loop spins every 0.1s, so rate-limit or this
                        # alarm becomes 10 lines a second and hides itself
                        if now_ts - snipe.get('last_blind_warn', 0.0) >= 15.0:
                            snipe['last_blind_warn'] = now_ts
                            _z = snipe.get('zero_bid_quotes', 0)
                            print(f"  🚨 [BLIND] {ticker} {snipe['option_id']}: no quote for "
                                  f"{blind_s:.0f}s ({snipe['quote_fails']} fails"
                                  f"{f', {_z} zero-bid' if _z else ''}). "
                                  f"Position is UNMANAGED.")
                        # After QUOTE_BLIND_PANIC_S with no quote we stop waiting and
                        # try to get out. A marketable exit on a stale book beats
                        # riding an unmonitored 0DTE to expiry.
                        if blind_s > QUOTE_BLIND_PANIC_S and not snipe.get("exiting"):
                            lb = snipe.get('last_bid', 0.0)
                            la = snipe.get('last_ask', lb)
                            if lb > 0:
                                print(f"  🆘 [BLIND EXIT] {ticker}: forcing exit on last "
                                      f"known quote bid=${lb:.2f}")
                                self._fire_exit(ticker, snipe, lb, la, False,
                                                f"BLIND {blind_s:.0f}s")
                        continue

                    # ---- BAD-TICK GUARD (config.BAD_TICK_*) -------------------
                    # A zero or collapsed bid is AMBIGUOUS: a genuinely worthless
                    # option prints it every tick, a glitched feed prints it once.
                    # Acting on the first one books a catastrophic exit on a
                    # position that was worth real money a second earlier -- four
                    # AVGO 1DTE puts on 2026-09-15 exited at $0.01 / -99.7%, one
                    # of them 42s after entry with spot inside the strike.
                    # So: require confirmation before believing it. A real
                    # collapse still exits, N ticks later, which costs nothing
                    # when the option really is dead. Note the quote is NOT
                    # recorded as last_bid until it is believed, or a poisoned
                    # tick would also become the BLIND EXIT's fallback price.
                    _prev_good = snipe.get('last_good_bid', 0.0)
                    _suspect = (bid <= 0.0) or (
                        _prev_good >= BAD_TICK_MIN_PREV
                        and bid < _prev_good * BAD_TICK_COLLAPSE)
                    if _suspect:
                        snipe['bad_ticks'] = snipe.get('bad_ticks', 0) + 1
                        if snipe['bad_ticks'] == 1:
                            snipe['bad_tick_since'] = time.time()
                            print(f"  ⚠️ [BAD TICK] {ticker} bid ${bid:.2f} vs last "
                                  f"good ${_prev_good:.2f} — ignoring until it "
                                  f"holds {BAD_TICK_CONFIRM_S:.0f}s")
                        held_s = time.time() - snipe.get('bad_tick_since', time.time())
                        # BOTH conditions: the loop runs at 10Hz, so a tick count
                        # alone confirms in a fraction of a second, and elapsed
                        # time alone would confirm on one tick after a quote gap.
                        if (held_s < BAD_TICK_CONFIRM_S
                                or snipe['bad_ticks'] < BAD_TICK_CONFIRMS):
                            continue
                        print(f"  ❗ [BAD TICK CONFIRMED] {ticker} bid ${bid:.2f} held "
                              f"{held_s:.0f}s over {snipe['bad_ticks']} ticks — "
                              f"treating as real")
                    else:
                        if snipe.get('bad_ticks'):
                            print(f"  ✅ [BAD TICK CLEARED] {ticker} bid recovered to "
                                  f"${bid:.2f} after {snipe['bad_ticks']} bad ticks — "
                                  f"a phantom exit was avoided")
                        snipe['bad_ticks'] = 0
                        snipe.pop('bad_tick_since', None)
                        snipe['last_good_bid'] = bid

                    snipe['last_bid'], snipe['last_ask'] = bid, ask

                    # A CONFIRMED zero bid is not a reason to skip -- it means the
                    # option is worthless and we should be getting out, not
                    # waiting. Let it fall through to the exit logic below.

                    tp_limit = snipe['tp_limit']
                    sl_limit = snipe['sl_limit']
                    exit_triggered = False
                    is_taking_profit = False
                    reason = ""

                    hold_time_mins = (current_time - snipe['entry_time']).total_seconds() / 60.0
                    cur_mod = current_time.hour * 60 + current_time.minute

                    # Trailing exit (config.TRAIL_PCT / rule trail_pct). Peak is
                    # tracked on the BID -- the same side the exit fills against,
                    # and the same quantity the TP/SL checks already use, so the
                    # trail can never be triggered by a level we couldn't sell at.
                    trail_pct = snipe.get('trail_pct') or 0.0
                    trail_stop = None

                    # MAX FAVOURABLE EXCURSION, tracked for EVERY rule.
                    # `peak_bid` below is trail STATE and is seeded at the fill,
                    # so it can never read below the entry and it was only ever
                    # updated when trail_pct > 0. That left the static-bracket
                    # rules (trail_pct: 0 -- NVDA, META) logging no peak at all,
                    # which is how NVDA came to print a bare "STOP LOSS". Peak is
                    # now the main diagnostic for whether an exit captured what
                    # was on offer (only ~25% of trades ever peak above +100%,
                    # the level a 50% trail needs to exit green), so it is
                    # tracked separately here: its own key, no seed, untouched by
                    # the trail logic, and therefore able to report a peak BELOW
                    # the entry when a trade never went green.
                    if bid > snipe.get('max_bid_seen', 0.0):
                        snipe['max_bid_seen'] = bid
                    _ep = snipe.get('entry_price') or 0.0
                    _mx = snipe.get('max_bid_seen', 0.0)
                    peak_note = (f" (peak ${_mx:.2f}, {(_mx / _ep - 1) * 100:+.0f}%)"
                                 if _ep > 0 else "")

                    if trail_pct > 0:
                        if bid > snipe.get('peak_bid', 0.0):
                            snipe['peak_bid'] = bid
                        trail_stop = round(snipe['peak_bid'] * (1 - trail_pct), 2)
                        snipe['last_trail_stop'] = trail_stop   # for the exit row

                    if cur_mod >= snipe.get('eod_flatten_mod', 15 * 60 + 55):
                        reason = "EOD STOP" + peak_note; exit_triggered = True
                    elif snipe.get('time_stop_mins') and hold_time_mins >= snipe['time_stop_mins']:
                        reason = f"TIME STOP {int(hold_time_mins)}m" + peak_note
                        exit_triggered = True
                    elif trail_stop is not None:
                        # pure trailing: no take-profit while trailing (validated --
                        # TP+trail was OOS -1.1% on realistic fills vs +4.8% pure)
                        if bid <= trail_stop:
                            reason = f"TRAIL STOP" + peak_note
                            exit_triggered = True
                            is_taking_profit = bid > snipe.get('entry_price', 0.0)
                    elif bid >= tp_limit:
                        reason = "TAKE PROFIT" + peak_note
                        exit_triggered = True; is_taking_profit = True
                    elif bid <= sl_limit:
                        reason = "STOP LOSS" + peak_note; exit_triggered = True

                    if exit_triggered:
                        self._fire_exit(ticker, snipe, bid, ask, is_taking_profit, reason)

                # =========================================================
                # 2. PURE FLOW SCANNER & DICTIONARY ROUTER
                # =========================================================
                # Backtest hard-blocks any new entry at/after 15:00 ET (0DTE theta
                # bleed). Global gate, ahead of the per-cell `hours` list.
                entries_open = current_time.hour < 15

                # LIVE CHART STATE. Throttled to 5s internally, atomic write,
                # and it cannot raise -- see live_state.py. Placed BEFORE the
                # scanner's `break`/`continue` chain so it still runs after
                # 15:00 and for tickers holding a position; both are exactly
                # when you want to be watching.
                live_state.tick(self)
                # Discretionary orders from the viewer, and the 15:55 flatten
                # that trumps "unmanaged". Placed BEFORE the scanner so a manual
                # hold registered this pass blocks an entry on the same pass.
                manual_orders.poll(
                    self, current_time.hour * 60 + current_time.minute)

                # manual-only (no UW key): the scanner is the bot's UW signal -- skip it
                for ticker in (tickers_to_track if self.uw_enabled else ()):
                    settings = WATCHLIST.get(ticker, {})

                    flow_ticks = self.uw.get_live_net_premium(ticker)
                    if not flow_ticks: continue

                    # --- CUMULATIVE AGGREGATION (see _ingest_flow) ---
                    latest_cumulative_flow = self._ingest_flow(ticker, flow_ticks)

                    flow_momentum = self.flow_tracker.update_and_check(ticker, latest_cumulative_flow)
                    if flow_momentum not in ("LONG", "SHORT"): continue

                    direction_str = "CALL" if flow_momentum == "LONG" else "PUT"
                    # log EVERY EMA crossover -> builds the ticker's own flow-percentile
                    # history (self-consistent units), regardless of whether it trades.
                    self._log_flow_trigger(ticker, abs(latest_cumulative_flow))

                    # ---------------------------------------------------------
                    # ENTRY GUARDS -- moved BELOW the flow aggregation and the
                    # percentile log on 2026-09-20. They used to sit at the top
                    # of this loop, which meant a ticker holding a position (or
                    # any ticker after 15:00) was skipped ENTIRELY: no ticks
                    # consumed, no cumulative_flow update, no tracker update and
                    # no crossover logged.
                    #
                    # That made `_live_flow_threshold` a percentile of a SUBSET
                    # while the backtest's annotate_flow_pct uses EVERY
                    # crossover -- the same quantity computed from different
                    # populations on the two sides, which is the family of bug
                    # that also produced the strike_offset and stale-spot
                    # divergences. check_gate_divergence measured it: the live
                    # gate came out up to 18.6% LOOSER (GLD; QQQ -17.1%, IWM
                    # -12.6%) and flipped the verdict on 2.2-5.4% of triggers,
                    # because crossovers during a hold carry HIGH cumulative
                    # flow and dropping them pulls the percentile down.
                    #
                    # Entry behaviour is UNCHANGED -- the same two conditions
                    # still refuse the same trades. Only the bookkeeping above
                    # now runs for every ticker every pass, which it must, for
                    # the live gate to mean what the backtest validated.
                    # `break` became `continue` for the same reason: after 15:00
                    # the backtest still counts crossovers toward the percentile
                    # (only _rule_matched_trigs drops hour >= 15 from MATCHING),
                    # so the bot must keep accumulating rather than abandoning
                    # the loop. Costs nothing -- get_live_net_premium reads the
                    # WebSocket buffer, not the network.
                    # ---------------------------------------------------------
                    if not entries_open: continue
                    if ticker in self.active_snipes: continue
                    # A MANUAL hold blocks the bot from the same ticker. Without
                    # this the guard above only knows about the bot's OWN
                    # positions, so a discretionary IWM long and a rule entry
                    # would stack and double the exposure.
                    if ticker in (getattr(self, "manual_holds", None) or {}):
                        continue

                    if USE_RULES:
                        # RULES mode: match a regime-conditioned recipe (config.RULES).
                        state = self.session_regime.get(ticker)
                        if not state: continue
                        trade_rules = self._match_rule(ticker, direction_str, state)
                        if not trade_rules: continue
                        regime_key = trade_rules["name"]
                        if current_time.hour not in trade_rules.get("hours", []): continue
                        if (current_time.hour * 60 + current_time.minute) < NO_ENTRY_BEFORE_MOD:
                            continue          # skip the opening-bell noise window
                        if trade_rules.get("skip_macro_am") and is_macro_am_day(current_time.date()):
                            if regime_key not in self._macro_am_warned:
                                print(f"  📅 {regime_key}: scheduled 8:30am macro release today — rule suppressed.")
                                self._macro_am_warned.add(regime_key)
                            continue
                        if trade_rules.get("ema_confirm") and not self._ema_confirm_ok(
                                ticker, direction_str, trade_rules):
                            continue
                        if trade_rules.get("dmi_confirm") and not self._dmi_confirm_ok(
                                ticker, direction_str, trade_rules):
                            continue
                        min_flow = self._live_flow_threshold(ticker, trade_rules)
                        if min_flow is None: continue
                    else:
                        # Legacy GEX tree. A missing/unparseable GEX field must NOT
                        # silently route to the NEGATIVE_GEX branch.
                        regime_data = self.session_gex.get(ticker)
                        if not regime_data: continue
                        macro_regime = regime_data.get("regime", "UNKNOWN")
                        if macro_regime not in ("POSITIVE", "NEGATIVE"): continue
                        regime_key = "POSITIVE_GEX" if macro_regime == "POSITIVE" else "NEGATIVE_GEX"
                        trade_rules = settings.get(regime_key, {}).get(direction_str)
                        if not trade_rules: continue
                        if current_time.hour not in trade_rules.get("hours", []): continue
                        min_flow = trade_rules.get("min_flow", 100_000_000)

                    # Backtest gate: abs(cumulative flow) >= threshold. Direction
                    # comes ONLY from the EMA crossover, not the sign of the flow
                    # (matches true_options_simulator.find_flow_entries).
                    if abs(latest_cumulative_flow) < min_flow:
                        continue


                    # =========================================================
                    # TRIGGER CONFIRMED: ZERO-LATENCY BASKET LOOKUP
                    # =========================================================
                    spot_price = self.get_spot_price(ticker)
                    if spot_price == 0.0: continue
                    
                    found_leg = None
                    executed_dte = None
                    executed_ask = 0.0
                    
                    strike_offset = int(trade_rules.get("strike_offset", 0)) if USE_RULES else 0
                    for target_dte in trade_rules.get("dte", []):
                        # Instant memory lookup!
                        contract = self._find_zero_latency_contract(
                            ticker, direction_str, target_dte, spot_price, offset=strike_offset)
                        
                        if contract:
                            found_leg = contract
                            executed_dte = target_dte
                            executed_ask = contract["ask"]
                            break 
                                
                    if not found_leg:
                        print(f"  ⚠️ Valid liquid contracts not found in memory for {ticker} ({direction_str}).")
                        continue

                    option_id = found_leg["symbol"]
                    target_roe = trade_rules["target_roe"]
                    stop_roe = trade_rules["stop_roe"]
                    # trailing exit supersedes the static TP/SL once set. target_roe /
                    # stop_roe still drive SIZING and the spread guard below.
                    trail_pct = float(trade_rules.get("trail_pct", TRAIL_PCT) or 0.0)
                    # worst-case loss actually risked: the trail binds well before a
                    # stop_roe of 1.0 (= no stop) ever would.
                    risk_roe = min(stop_roe, trail_pct) if trail_pct > 0 else stop_roe

                    # Entry is priced at MID + 1 tick (capped at the ask): close to
                    # the backtest's bar-close fill, with better fill odds than
                    # resting at mid. Brackets are measured from this entry price.
                    entry_bid = found_leg.get("bid", 0.0)
                    entry_mid = (entry_bid + executed_ask) / 2.0 if entry_bid > 0 else executed_ask
                    entry_limit = min(round(entry_mid + 0.01, 2), round(executed_ask, 2))
                    entry_limit = max(0.01, entry_limit)
                    spread_frac = (executed_ask - entry_bid) / entry_mid if entry_mid > 0 else 1.0

                    # Backtest penny filter: reject if the entry price is under $0.50.
                    if entry_mid < 0.50:
                        continue

                    # If the round-trip spread is as wide as the risk being taken, the
                    # trade is untradeable -- you'd pay the stop in slippage on entry.
                    # With a trail the binding risk is trail_pct, not stop_roe.
                    if spread_frac >= risk_roe:
                        print(f"  🚫 [SPREAD] {ticker} {direction_str}: spread {spread_frac*100:.0f}% "
                              f">= risk {risk_roe*100:.0f}%. Untradeable, skipping.")
                        continue

                    tp_limit = round(entry_limit * (1 + target_roe), 2)
                    sl_limit = round(entry_limit * (1 - stop_roe), 2)

                    # =========================================================
                    # POSITION SIZING (flat premium parity) + capital allocator
                    # =========================================================
                    per_contract_cost = entry_limit * 100.0        # 1 contract = 100 shares
                    vix_mult = self.session_vix_mult if trade_rules.get("vix_size", True) else 1.0
                    qty = 1
                    if self.account_equity > 0 and per_contract_cost > 0:
                        target_premium = self.account_equity * SIZING_TARGET_PREMIUM_PCT * vix_mult
                        qty = max(1, round(target_premium / per_contract_cost))
                        # risk-cap backstop: trim toward the 2% worst-case budget,
                        # but never below 1 (a single contract is always allowed).
                        # Deliberately uses stop_roe, NOT risk_roe: sizing off the
                        # tighter trail would LOOSEN this trim and quietly increase
                        # position size alongside the exit change. Size as if the
                        # premium can go to zero; let the trail be pure upside.
                        risk_budget = self.account_equity * MAX_PORTFOLIO_RISK_PCT
                        while qty > 1 and qty * per_contract_cost * stop_roe > risk_budget:
                            qty -= 1
                        qty = min(qty, SIZING_MAX_CONTRACTS)

                    premium_cost = per_contract_cost * qty
                    planned_risk = premium_cost * stop_roe
                    if self.account_equity > 0 and qty == 1 and planned_risk > self.account_equity * MAX_PORTFOLIO_RISK_PCT:
                        print(f"  ⚠️ [RISK] {ticker} {direction_str}: 1-contract worst case ${planned_risk:,.0f} "
                              f"exceeds {MAX_PORTFOLIO_RISK_PCT*100:.0f}% budget "
                              f"${self.account_equity * MAX_PORTFOLIO_RISK_PCT:,.0f} — taking min size anyway.")

                    if self.allocator and not self.allocator.allocate_live_trade(
                        ticker=ticker,
                        strategy_type="DIRECTIONAL",
                        exact_option_cost=entry_limit * qty,
                        spot_price=spot_price,
                    ):
                        continue

                    print("\n" + "🐋" * 30)
                    print(f"🎯 INSTITUTIONAL WHALE SWEEP: {direction_str} on {ticker}")
                    print(f"  ├─ Master Config : {regime_key} | DTE: {executed_dte}")
                    print(f"  ├─ Live Net Flow : ${latest_cumulative_flow:,.0f} (|Threshold|: ${min_flow:,.0f})")
                    print(f"  ├─ Contract      : {option_id} | Bid ${entry_bid:.2f} / Mid ${entry_mid:.2f} / Ask ${executed_ask:.2f} -> Limit ${entry_limit:.2f}")
                    print(f"  ├─ Size          : {qty}x  (~${premium_cost:,.0f} premium, "
                          f"{SIZING_TARGET_PREMIUM_PCT*100:.1f}% target"
                          f"{f' x{vix_mult} VIX-throttle' if vix_mult != 1.0 else ''}; "
                          f"worst case ${planned_risk:,.0f})")
                    if trail_pct > 0:
                        print(f"  └─ Exit          : TRAIL {trail_pct*100:.0f}% from peak "
                              f"(initial stop ${entry_limit * (1 - trail_pct):.2f})")
                    else:
                        print(f"  └─ Brackets      : TP ${tp_limit:.2f} (+{target_roe*100:.0f}%) / SL ${sl_limit:.2f} (-{stop_roe*100:.0f}%)")
                    print("🐋" * 30 + "\n")

                    action_str = "ENTRY_LONG" if direction_str == "CALL" else "ENTRY_SHORT"

                    # ---- place + confirm the entry fill ----
                    actual_entry = entry_limit
                    entry_coid = None
                    unconfirmed = False

                    filled_qty = qty
                    if not DRY_RUN:
                        # Chase the offer instead of placing once and giving up:
                        # the old single 10s attempt filled 27% of paper entries
                        # and selected adversely (a resting buy fills when the
                        # option cheapens). _chase_entry cancels and re-prices a
                        # tick nearer the ask each rung, capped at the offer that
                        # was quoted when the signal fired.
                        status, fill_px, fqty = self._chase_entry(
                            ticker, option_id, qty, entry_limit, executed_ask,
                        )
                        if status in ("FILLED", "PARTIAL_FILLED") and fqty > 0:
                            actual_entry = fill_px if fill_px > 0 else entry_limit
                            filled_qty = int(fqty)
                            print(f"  ✅ {ticker} entry FILLED @ ${actual_entry:.2f} (qty {fqty:g}/{qty})")
                        elif status is None:
                            unconfirmed = True   # user policy: manage it, flag it
                            print(f"  ⚠️ {ticker} entry fill UNCONFIRMED (status API failed) — managing as filled + flagged.")
                        else:
                            print(f"  ❌ {ticker} entry abandoned ({status}) — releasing capital.")
                            if self.allocator:
                                self.allocator.release_trade(premium_cost)
                            continue

                    # brackets measured from the ACTUAL fill price
                    tp_limit = round(actual_entry * (1 + target_roe), 2)
                    sl_limit = round(actual_entry * (1 - stop_roe), 2)

                    self.active_snipes[ticker] = {
                        "option_id": option_id,
                        "entry_price": actual_entry,
                        "entry_mid": entry_mid,
                        "tp_limit": tp_limit,
                        "sl_limit": sl_limit,
                        # trailing exit: peak seeded at the fill, so the effective
                        # initial stop is -trail_pct from entry
                        "trail_pct": trail_pct,
                        "peak_bid": actual_entry,
                        # quote-outage guard: seeded at entry so a position that
                        # goes blind IMMEDIATELY still has a clock to measure
                        # against (see QUOTE_BLIND_PANIC_S in config).
                        "entry_ts": time.time(),
                        "last_quote_ts": time.time(),
                        "last_bid": actual_entry,
                        "last_ask": executed_ask if executed_ask > 0 else actual_entry,
                        "quote_fails": 0,
                        "is_call": (direction_str == "CALL"),
                        "entry_time": current_time,
                        "dte": executed_dte,
                        "time_stop_mins": trade_rules.get("time_stop_mins"),
                        "eod_flatten_mod": trade_rules.get("eod_flatten_mod", 15 * 60 + 55),
                        "capital_committed": premium_cost,   # what the allocator reserved (qty target)
                        "entry_coid": entry_coid,
                        "unconfirmed": unconfirmed,
                        "exiting": False,
                        "regime": regime_key,          # rule name (RULES) or POSITIVE/NEGATIVE_GEX
                        "size": filled_qty,
                        "flow_at_entry": latest_cumulative_flow,
                        "min_flow_used": min_flow,
                    }

                    self.log_trade(
                        ticker, action_str, option_id, actual_entry, filled_qty, "Whale Flow Spike",
                        latest_cumulative_flow, regime_key,
                        extra={
                            "direction": direction_str,
                            "entry_price": round(actual_entry, 2),
                            "entry_mid": round(entry_mid, 2),
                            "spot_at_entry": round(spot_price, 2),
                            "tp_limit": tp_limit, "sl_limit": sl_limit,
                            "target_roe": target_roe, "stop_roe": round(stop_roe, 4),
                            "trail_pct": trail_pct or None,
                            "dte": executed_dte,
                            "contracts": filled_qty,
                            "premium_committed": round(actual_entry * 100.0 * filled_qty, 2),
                            "sizing_target_pct": SIZING_TARGET_PREMIUM_PCT,
                            "vix_mult": vix_mult,
                            "vix_prev": round(self.session_vix["vix_prev"], 2) if self.session_vix else None,
                            "min_flow_threshold": round(float(min_flow), 2),
                            "session_regime": (self.session_regime.get(ticker) if USE_RULES else None),
                            "unconfirmed": unconfirmed,
                        })

                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n🛑 Shutting down Master Engine gracefully...")
                self._shutdown_flatten()
                break
            except Exception as e:
                print(f"🚨 Master Loop Exception: {e}")
                time.sleep(5)

if __name__ == "__main__":
    # A cloud host stops a process with SIGTERM, not Ctrl-C, so the graceful
    # shutdown path would never have run there -- the loop would simply be
    # killed with positions open and nothing logged. Translating SIGTERM into
    # KeyboardInterrupt routes both into _shutdown_flatten.
    import signal as _signal

    def _on_sigterm(_sig, _frm):
        raise KeyboardInterrupt

    try:
        _signal.signal(_signal.SIGTERM, _on_sigterm)
    except (ValueError, AttributeError, OSError):
        pass          # not available on every platform / non-main thread

    # 🚨 BEFORE FlowExecutionEngine(), WHICH AUTHENTICATES TO WEBULL.
    # Constructing the engine opens a broker session and writes conf/token.txt
    # -- the very file two engines contend on. Refusing after that point would
    # have already done the damage the check exists to prevent.
    try:
        import peer_guard
        peer_guard.enforce()
    except SystemExit:
        raise
    except Exception as _e:                    # noqa: BLE001 -- deliberate
        print(f"  ⚠️  peer check unavailable ({type(_e).__name__}: {_e}) — "
              f"continuing.")

    bot = FlowExecutionEngine()
    bot.run_brain()