"""
directional_flow_backtester.py
==============================

Is each per-ticker recipe in config.py (min_flow, target_roe, stop_roe) actually
the best one in the current lake?

Replicates the flow bot's entry logic (as aligned to true_options_simulator):
  - net flow per minute from silver: (ask_vol - bid_vol) * vwap * 100  (call +, put -)
  - cumulative per day, ewm(span=5) crossover -> CALL on bullish, PUT on bearish
  - regime bucket: --regime-by gex (default) or --regime-by trend
      gex:   --gex daily     = historical/GEX<T>.parquet, PRIOR day's close sign
             --gex intraday  = per-minute spot GEX (lake/silver/spot-exposures-1m)
                               sign at the signal minute
             --gex compare   = score every trigger under BOTH and show whether
                               the intraday label separates POS/NEG more cleanly
      trend: UPTREND/DOWNTREND/CHOP from the underlying's daily SMA state
             (SMA20 vs SMA50 + slope, prior day) -- tests whether a cell's edge
             is really trend-conditioned rather than GEX-conditioned
  - hours + DTE taken from config (NOT gridded); global block at hour >= 15
    (compare mode ignores config gates: hours 9-14, dte [0,1], no time-stop)
  - ATM contract (nearest strike), entry at mid, penny filter mid >= 0.50
  - brackets on the contract's 1-min high/low: TP / SL / 15:55 EOD / time_stop_mins

Then for every ENABLED config cell it grids min_flow x target_roe x R:R and
prints the config point next to the grid's best (by $ expectancy per signal).

Each qualifying crossover is one trade (no overlap prevention) -- matches the
simulator's per-signal methodology, not a realistic equity curve.

Usage:
    python directional_flow_backtester.py --build-bars      # one-time, ~mins (rebuild after lake grows)
    python directional_flow_backtester.py                   # --gex compare
    python directional_flow_backtester.py --gex intraday --flow-mode pct --split 2025-08-21
    python directional_flow_backtester.py --regime-by amp --flow-mode pct --split 2025-08-21
    python directional_flow_backtester.py --rule-file candidate_rules.json --flow-mode pct --split 2025-08-21
"""

import os
import sys
import glob
import json
import argparse
import datetime as dt
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WATCHLIST

DEFAULT_LAKE = "lake/silver/option-contracts-1m"
DEFAULT_HIST = "historical"
FLOW_CACHE = "flow_1m.parquet"
FLOW_CACHE_NETPREM = "flow_1m_netprem.parquet"
BARS_CACHE = "opt_bars_atm.parquet"
WATCH = list(WATCHLIST.keys())

# set from --flow-source in __main__; _prep / emit_thresholds read it
FLOW_SOURCE = "silver"   # "silver" | "netprem"

# minute-of-day (ET) at/after which an open position is flattened. 955 = 15:55,
# the live bot's default. Overridable with --eod-flatten HH:MM for the hold sweep.
EOD_FLATTEN_MOD = 15 * 60 + 55

CONTRACT_MULT = 100


def _naive(s):
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        try:
            return s.dt.tz_convert(None)
        except Exception:
            return s
GRID_TARGET_ROE = [0.20, 0.40, 0.60, 0.80, 1.00]
GRID_RR = [1.0, 1.5, 2.0, 3.0]
GRID_MIN_FLOW = [1e6, 2e6, 3.5e6, 5e6, 7.5e6, 10e6, 15e6, 20e6, 30e6]
# --flow-mode pct: gate on a trailing-window PERCENTILE of this ticker's own
# trigger flow instead of a fixed $ amount. check_flow_drift.py shows the fixed
# $ size of "big flow" moves 2-23x over 2y, so a single min_flow is non-stationary.
GRID_MIN_FLOW_PCT = [50, 65, 80, 90, 95]
FLOW_PCT_WINDOW_DAYS = 60   # trailing calendar days for the percentile reference
MIN_SIGNALS = 50          # per-signal counts are overlapping; small n = clustered overfit
SUSPECT_WINRATE = 88.0    # win rates above this on modest n are almost always artifacts
ALL_HOURS = list(range(9, 15))

# set from --flow-mode in __main__; report helpers read these
FLOW_MODE = "abs"                    # "abs" | "pct"
ACTIVE_FLOW_GRID = GRID_MIN_FLOW

# regime buckets: set from --regime-by in __main__.
#   gex    - POSITIVE/NEGATIVE_GEX (daily prior-day or intraday spot)
#   trend  - daily SMA state of the underlying
#   volume - the underlying's daily volume vs its trailing median
#   hour   - the entry hour of the signal (ET)
GEX_REGIME_KEYS = ("POSITIVE_GEX", "NEGATIVE_GEX")
TREND_REGIME_KEYS = ("UPTREND", "CHOP", "DOWNTREND")
VOLUME_REGIME_KEYS = ("LOWVOL", "NORMVOL", "HIVOL")
HOUR_REGIME_KEYS = ("h9", "h10", "h11", "h12", "h13", "h14")
# amp = how many of {prior-day NEG-GEX, LOWVOL, CHOP} hold -- the "dealer hedging
# amplifies the move" latent state that NEG-GEX / LOWVOL / CHOP each measure.
AMP_REGIME_KEYS = ("amp0", "amp1", "amp2", "amp3")
REGIME_KEYS = GEX_REGIME_KEYS
REGIME_BY = "gex"                    # gex | trend | volume | hour | amp
TREND_FAST, TREND_SLOW, TREND_SLOPE_LAG = 20, 50, 10
VOL_WINDOW, VOL_HI, VOL_LO = 20, 1.25, 0.80   # daily vol / trailing median thresholds


def _fmt_flow(k):
    return f"p{int(round(k)):>2d}" if FLOW_MODE == "pct" else f"${k/1e6:>5.1f}M"


def _cfg_flow(cell):
    return cell.get("min_flow_pct") if FLOW_MODE == "pct" else cell.get("min_flow")


def _flow_match(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) < 0.5 if FLOW_MODE == "pct" else abs(a - b) / max(b, 1) < 0.2


def annotate_flow_pct(trigs, window_days=FLOW_PCT_WINDOW_DAYS):
    """Attach t['thr'] = {P: dollar_threshold} -- the Pth percentile of this
    ticker's trigger abs_flow over the trailing `window_days`, using only days
    STRICTLY BEFORE the trigger (no same-day lookahead). t['thr'] is None until
    a full window of history exists / >=30 prior triggers."""
    if not trigs:
        return
    order = np.argsort([t["ts"] for t in trigs], kind="stable")
    day_ns = np.array([pd.Timestamp(trigs[i]["ts"]).normalize().value for i in order])
    flows = np.array([trigs[i]["abs_flow"] for i in order], dtype=float)
    win = int(window_days) * 86_400_000_000_000
    first = day_ns[0]
    for pos, i in enumerate(order):
        cur = day_ns[pos]
        if cur - first < win:
            trigs[i]["thr"] = None
            continue
        lo = int(np.searchsorted(day_ns, cur - win, side="left"))
        hi = int(np.searchsorted(day_ns, cur, side="left"))
        hist = flows[lo:hi]
        trigs[i]["thr"] = ({P: float(np.percentile(hist, P)) for P in GRID_MIN_FLOW_PCT}
                           if len(hist) >= 30 else None)


def flow_pairs_for(t):
    """[(grid_key, dollar_threshold), ...] for this trigger under the active mode."""
    if FLOW_MODE == "pct":
        thr = t.get("thr")
        return [] if not thr else [(P, thr[P]) for P in GRID_MIN_FLOW_PCT]
    return [(mf, mf) for mf in GRID_MIN_FLOW]
COMMISSION_PCT = 0.015   # round-trip commission + slippage, as a fraction of entry premium


# --------------------------------------------------------------------------
def build_flow_1m(lake_dir, force=False):
    if os.path.exists(FLOW_CACHE) and not force:
        return pd.read_parquet(FLOW_CACHE)
    import polars as pl
    parts = sorted(glob.glob(os.path.join(lake_dir, "date=*", "bars.parquet")))
    frames = []
    for i, p in enumerate(parts, 1):
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol").is_in(WATCH))
              .with_columns(
                  pl.when(pl.col("option_type") == "call")
                  .then((pl.col("ask_volume") - pl.col("bid_volume")) * pl.col("vwap") * 100)
                  .otherwise(-((pl.col("ask_volume") - pl.col("bid_volume")) * pl.col("vwap") * 100))
                  .alias("nf"))
              .group_by(["underlying_symbol", "minute_et"])
              .agg(pl.col("nf").sum().alias("net_flow_1m")))
        frames.append(lf.collect().to_pandas())
        if i % 25 == 0:
            print(f"  flow {i}/{len(parts)}")
    flow = pd.concat(frames, ignore_index=True)
    flow["minute_et"] = _naive(flow["minute_et"])
    flow["date"] = flow["minute_et"].dt.date
    flow = flow.sort_values(["underlying_symbol", "minute_et"])
    flow["cum_flow"] = flow.groupby(["underlying_symbol", "date"])["net_flow_1m"].cumsum()
    flow.to_parquet(FLOW_CACHE, index=False)
    return flow


def build_flow_netprem(hist_dir, force=False):
    """Same output shape as build_flow_1m (underlying_symbol, minute_et, date,
    net_flow_1m, cum_flow) but sourced from historical/NETPREM{T}.parquet:
    net_flow_1m = net_call_premium - net_put_premium per minute (UW server-side).
    This is the EXACT quantity the live bot accumulates -- running --rule-file on
    it checks whether the silver 'Lake units' translated the thresholds faithfully."""
    if os.path.exists(FLOW_CACHE_NETPREM) and not force:
        return pd.read_parquet(FLOW_CACHE_NETPREM)
    import polars as pl
    parts = sorted(glob.glob(os.path.join(hist_dir, "NETPREM*.parquet")))
    if not parts:
        sys.exit(f"no NETPREM*.parquet in {hist_dir} -- run: uw_options_data_lake.py netprem-build")
    frames = []
    for p in parts:
        tk = os.path.basename(p)[len("NETPREM"):-len(".parquet")]
        lf = (pl.scan_parquet(p)
              .select(pl.lit(tk).alias("underlying_symbol"),
                      pl.col("minute_et"),
                      pl.col("net_premium").cast(pl.Float64).fill_null(0.0).alias("net_flow_1m")))
        frames.append(lf.collect().to_pandas())
    flow = pd.concat(frames, ignore_index=True)
    flow["minute_et"] = _naive(flow["minute_et"])
    flow["date"] = flow["minute_et"].dt.date
    flow = flow.sort_values(["underlying_symbol", "minute_et"])
    flow["cum_flow"] = flow.groupby(["underlying_symbol", "date"])["net_flow_1m"].cumsum()
    flow.to_parquet(FLOW_CACHE_NETPREM, index=False)
    return flow


def build_bars(lake_dir):
    import polars as pl
    parts = sorted(glob.glob(os.path.join(lake_dir, "date=*", "bars.parquet")))
    frames = []
    for i, p in enumerate(parts, 1):
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol").is_in(WATCH))
              .with_columns(((pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days()).alias("dte"))
              .filter((pl.col("dte") >= 0) & (pl.col("dte") <= 8))
              .filter((pl.col("strike") - pl.col("underlying_close")).abs() / pl.col("underlying_close") <= 0.04)
              .select(["underlying_symbol", "option_chain_id", "option_type", "strike", "expiry",
                       "minute_et", "high", "low", "close", "bid_close", "ask_close", "underlying_close"]))
        frames.append(lf.collect().to_pandas())
        if i % 20 == 0:
            print(f"  bars {i}/{len(parts)}")
    bars = pd.concat(frames, ignore_index=True)
    bars["minute_et"] = _naive(bars["minute_et"])
    bars["date"] = bars["minute_et"].dt.date
    bars["expiry"] = pd.to_datetime(bars["expiry"]).dt.date
    for c in ("strike", "high", "low", "close", "bid_close", "ask_close", "underlying_close"):
        bars[c] = bars[c].astype("float32")
    bars.to_parquet(BARS_CACHE, index=False)
    print(f"  bars cache: {len(bars):,} rows -> {BARS_CACHE}")
    return bars


SCREEN_CACHE_DIR = "_screen_cache"


def _screen_build_one(lake_dir, ticker):
    """Build a single ticker's flow + ATM bars straight from silver (no shared
    cache, no config edit). Cached under _screen_cache/ so re-runs are instant."""
    os.makedirs(SCREEN_CACHE_DIR, exist_ok=True)
    fpath = os.path.join(SCREEN_CACHE_DIR, f"{ticker}_flow.parquet")
    bpath = os.path.join(SCREEN_CACHE_DIR, f"{ticker}_bars.parquet")
    if os.path.exists(fpath) and os.path.exists(bpath):
        return pd.read_parquet(fpath), pd.read_parquet(bpath)
    import polars as pl
    parts = sorted(glob.glob(os.path.join(lake_dir, "date=*", "bars.parquet")))
    fframes, bframes = [], []
    for i, p in enumerate(parts, 1):
        base = pl.scan_parquet(p).filter(pl.col("underlying_symbol") == ticker)
        fframes.append(base.with_columns(
            pl.when(pl.col("option_type") == "call")
            .then((pl.col("ask_volume") - pl.col("bid_volume")) * pl.col("vwap") * 100)
            .otherwise(-((pl.col("ask_volume") - pl.col("bid_volume")) * pl.col("vwap") * 100)).alias("nf"))
            .group_by(["underlying_symbol", "minute_et"])
            .agg(pl.col("nf").sum().alias("net_flow_1m")).collect().to_pandas())
        bframes.append(base.with_columns(
            ((pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days()).alias("dte"))
            .filter((pl.col("dte") >= 0) & (pl.col("dte") <= 8))
            .filter((pl.col("strike") - pl.col("underlying_close")).abs() / pl.col("underlying_close") <= 0.04)
            .select(["underlying_symbol", "option_chain_id", "option_type", "strike", "expiry",
                     "minute_et", "high", "low", "close", "bid_close", "ask_close", "underlying_close"])
            .collect().to_pandas())
        if i % 100 == 0:
            print(f"    {ticker} {i}/{len(parts)}")
    flow = pd.concat(fframes, ignore_index=True)
    if flow.empty:
        return flow, pd.DataFrame()
    flow["minute_et"] = _naive(flow["minute_et"])
    flow["date"] = flow["minute_et"].dt.date
    flow = flow.sort_values(["underlying_symbol", "minute_et"])
    flow["cum_flow"] = flow.groupby(["underlying_symbol", "date"])["net_flow_1m"].cumsum()
    bars = pd.concat(bframes, ignore_index=True)
    bars["minute_et"] = _naive(bars["minute_et"])
    bars["date"] = bars["minute_et"].dt.date
    bars["expiry"] = pd.to_datetime(bars["expiry"]).dt.date
    flow.to_parquet(fpath, index=False)
    bars.to_parquet(bpath, index=False)
    return flow, bars


# --------------------------------------------------------------------------
def triggers_for(flow, ticker):
    g = flow[flow["underlying_symbol"] == ticker]
    out = []
    for d, gd in g.groupby("date"):
        gd = gd.sort_values("minute_et")
        cum = gd["cum_flow"].values
        if len(cum) < 6:
            continue
        ema = pd.Series(cum).ewm(span=5, adjust=False).mean().values
        mt = gd["minute_et"].values
        hrs = gd["minute_et"].dt.hour.values
        for i in range(1, len(cum)):
            bull = cum[i - 1] <= ema[i - 1] and cum[i] > ema[i]
            bear = cum[i - 1] >= ema[i - 1] and cum[i] < ema[i]
            if bull or bear:
                out.append({"date": d, "ts": mt[i], "hour": int(hrs[i]),
                            "dir": "CALL" if bull else "PUT", "abs_flow": abs(cum[i])})
    return out


def load_gex(hist_dir, ticker):
    """{trading_day: 'POSITIVE'|'NEGATIVE'} using the PRIOR day's close GEX -
    that's all a 9:30 ET entry can actually know. ~8-24% of days flip label vs
    same-day (check_gex_lookahead.py), so this is not cosmetic."""
    p = os.path.join(hist_dir, f"GEX{ticker}.parquet")
    if not os.path.exists(p):
        return {}
    g = pd.read_parquet(p)
    g.columns = [c.lower() for c in g.columns]
    if "date" not in g.columns:
        return {}
    g["d"] = pd.to_datetime(g["date"], utc=True).dt.tz_localize(None).dt.date
    col = next((c for c in ("net_gex", "total_net_gex") if c in g.columns), None)
    if not col and {"call_gex", "put_gex"} <= set(g.columns):
        g["net_gex"] = g["call_gex"] + g["put_gex"]; col = "net_gex"
    if not col and {"call_gamma", "put_gamma"} <= set(g.columns):
        g["net_gex"] = g["call_gamma"] + g["put_gamma"]; col = "net_gex"
    if not col:
        return {}
    g = g.sort_values("d")
    g["eff"] = g[col].shift(1)
    g = g.dropna(subset=["eff"])
    return {r["d"]: ("POSITIVE" if r["eff"] > 0 else "NEGATIVE") for _, r in g.iterrows()}


def load_trend_regime(hist_dir, ticker, fast=TREND_FAST, slow=TREND_SLOW, slope_lag=TREND_SLOPE_LAG):
    """{trading_day: 'UPTREND'|'DOWNTREND'|'CHOP'} from the underlying's daily
    RTH close in historical/{T}.parquet. UPTREND = SMA(fast) > SMA(slow) and
    SMA(slow) rising; DOWNTREND = the mirror; CHOP = neither. Classified on the
    PRIOR day's close (a 9:30 entry only knows yesterday), same as load_gex."""
    p = os.path.join(hist_dir, f"{ticker}.parquet")
    if not os.path.exists(p):
        return {}
    df = pd.read_parquet(p)
    df.columns = [c.lower() for c in df.columns]
    if "start_time" not in df.columns or "close" not in df.columns:
        return {}
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York")
    mod = et.dt.hour * 60 + et.dt.minute
    rth = df.assign(_d=et.dt.date)[(mod >= 570) & (mod <= 960)]
    daily = rth.sort_values("start_time").groupby("_d")["close"].last().astype(float)
    if len(daily) < slow + slope_lag + 5:
        return {}
    sf = daily.rolling(fast).mean()
    ss = daily.rolling(slow).mean()
    slope = ss - ss.shift(slope_lag)
    reg = pd.Series("CHOP", index=daily.index, dtype=object)
    reg[(sf > ss) & (slope > 0)] = "UPTREND"
    reg[(sf < ss) & (slope < 0)] = "DOWNTREND"
    reg[ss.isna() | slope.isna()] = None
    reg = reg.shift(1).dropna()
    return dict(reg)


def load_amt_open(hist_dir, ticker):
    """{trading_day: 'below_va'|'inside_va'|'above_va'} -- where the session
    OPENED vs the PRIOR day's volume value area (Auction Market Theory).
    Empty if no historical/{T}.parquet."""
    try:
        from amt_profile import amt_open_map
        return amt_open_map(ticker, hist=hist_dir)
    except Exception:
        return {}


def _amt_ok(spec, loc):
    from amt_profile import amt_ok
    return amt_ok(spec, loc)


def _daily_underlying(hist_dir, ticker, field):
    """Per-trading-day RTH `field` (close or volume) from historical/{T}.parquet."""
    p = os.path.join(hist_dir, f"{ticker}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df.columns = [c.lower() for c in df.columns]
    if "start_time" not in df.columns or field not in df.columns:
        return None
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York")
    mod = et.dt.hour * 60 + et.dt.minute
    rth = df.assign(_d=et.dt.date)[(mod >= 570) & (mod <= 960)].sort_values("start_time")
    if field == "volume":
        return rth.groupby("_d")[field].sum().astype(float)
    return rth.groupby("_d")[field].last().astype(float)


def load_volume_regime(hist_dir, ticker, window=VOL_WINDOW, hi=VOL_HI, lo=VOL_LO):
    """{trading_day: 'LOWVOL'|'HIVOL'|'NORMVOL'} from the underlying's daily RTH
    share volume vs its trailing-`window` median (prior day). Low-volume days =
    thin liquidity = dealer hedging moves price more, so the flow signal should
    get more amplification -- the same mechanism as negative gamma."""
    vol = _daily_underlying(hist_dir, ticker, "volume")
    if vol is None or len(vol) < window + 5:
        return {}
    ratio = vol / vol.rolling(window).median()
    reg = pd.Series("NORMVOL", index=vol.index, dtype=object)
    reg[ratio >= hi] = "HIVOL"
    reg[ratio <= lo] = "LOWVOL"
    reg[ratio.isna()] = None
    return dict(reg.shift(1).dropna())


def load_amp_score(hist_dir, ticker, trend_fast=TREND_FAST, trend_slow=TREND_SLOW):
    """{trading_day: 0-3} = count of {prior-day NEGATIVE_GEX, LOWVOL, CHOP}.
    Only days present in all three sources are kept (their warm-ups differ)."""
    g = load_gex(hist_dir, ticker)
    v = load_volume_regime(hist_dir, ticker)
    tr = load_trend_regime(hist_dir, ticker, trend_fast, trend_slow)
    if not (g and v and tr):
        return {}
    days = set(g) & set(v) & set(tr)
    return {d: int(g[d] == "NEGATIVE") + int(v[d] == "LOWVOL") + int(tr[d] == "CHOP")
            for d in days}


def load_spot_gex(spot_dir, ticker):
    """{date: (ts_int64_sorted, sign_int8)} from the 1-min spot-GEX silver
    (lake/silver/spot-exposures-1m/date=*/{T}.parquet). Field:
    gamma_per_one_percent_move_oi -- same one the live WS gex: stream uses.
    Lets a signal at minute ts be labelled with the regime a bot polling at
    that moment would actually see, instead of one sign for the whole day."""
    files = sorted(glob.glob(os.path.join(spot_dir, "date=*", f"{ticker}.parquet")))
    if not files:
        return {}
    import polars as pl
    df = (pl.scan_parquet(files)
          .select(pl.col("minute_et").dt.replace_time_zone(None).alias("m"),
                  pl.col("gamma_per_one_percent_move_oi").alias("g"))
          .filter(pl.col("g").is_not_null())
          .collect().to_pandas())
    if df.empty:
        return {}
    df["date"] = df["m"].dt.date
    df["sign"] = np.sign(df["g"].values).astype("int8")
    out = {}
    for d, gd in df.sort_values("m").groupby("date"):
        out[d] = (gd["m"].values.astype("datetime64[ns]").astype("int64"), gd["sign"].values)
    return out


def spot_regime_at(spot_by_date, d, ts):
    """POSITIVE / NEGATIVE / None for the last spot-GEX reading at or before ts."""
    entry = spot_by_date.get(d)
    if entry is None:
        return None
    arr, signs = entry
    key = np.datetime64(pd.Timestamp(ts).to_datetime64(), "ns").astype("int64")
    i = int(np.searchsorted(arr, key, side="right")) - 1
    if i < 0:
        return None
    s = signs[i]
    return "POSITIVE" if s > 0 else ("NEGATIVE" if s < 0 else None)


EMA_SPANS = (8, 21, 34)


def load_ema_stack(hist_dir, ticker, tf_min, spans=EMA_SPANS):
    """asof-queryable pd.Series {minute -> 'BULL'|'BEAR'|'MIXED'} for the
    underlying price EMA stack on tf_min-minute RTH bars from historical/{T}.parquet.
    BULL = ema(spans[0]) > ema(spans[1]) > ... (fast above slow); BEAR = mirror.
    EMAs run continuously across days (a gap opens with yesterday's stack), like a
    trader's chart. Query with .asof(ts) -> the last bar that CLOSED at/before ts."""
    p = os.path.join(hist_dir, f"{ticker}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df.columns = [c.lower() for c in df.columns]
    if "start_time" not in df.columns or "close" not in df.columns:
        return None
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    mod = et.dt.hour * 60 + et.dt.minute
    s = (pd.Series(df["close"].astype(float).values, index=et)[(mod.values >= 570) & (mod.values <= 960)]
         .sort_index())
    s = s[~s.index.duplicated(keep="last")]
    bars = s.resample(f"{int(tf_min)}min").last().dropna()
    if len(bars) < max(spans) + 5:
        return None
    e = pd.DataFrame({sp: bars.ewm(span=sp, adjust=False).mean() for sp in spans})
    up = pd.Series(True, index=e.index)
    dn = pd.Series(True, index=e.index)
    for a, b in zip(spans[:-1], spans[1:]):
        up &= e[a] > e[b]
        dn &= e[a] < e[b]
    return pd.Series(np.where(up, "BULL", np.where(dn, "BEAR", "MIXED")), index=e.index)


def ema_state_at(stack, ts):
    """stack.asof(ts) with a tz-naive Timestamp; None if before the first bar."""
    if stack is None:
        return None
    v = stack.asof(pd.Timestamp(ts))
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v


# --------------------------------------------------------------------------
def simulate_trigger(t, direction, dtes, tstop, bars_by_date, bars_by_cid, only=None):
    """Run one crossover through the contract's forward 1-min path.
    only=None       -> full (flow_key, target_roe, R:R) grid, [(flow_key, tr, rr, date, pnl), ...]
    only=(thr,tr,rr)-> just that one recipe, [(thr, tr, rr, date, pnl), ...] gated on
                       abs_flow >= thr ($ threshold, caller resolves any percentile)."""
    out = []
    flow_pairs = [(only[0], only[0])] if only else flow_pairs_for(t)
    if not flow_pairs:
        return out
    tr_vals = [only[1]] if only else GRID_TARGET_ROE
    rr_vals = [only[2]] if only else GRID_RR
    day = bars_by_date.get(t["date"])
    if day is None:
        return out
    at_spot = day[day["minute_et"] <= t["ts"]]
    if at_spot.empty:
        return out
    spot = float(at_spot.iloc[-1]["underlying_close"])
    for target_dte in dtes:
        cid = pick_contract(day, t["ts"], direction, target_dte, spot)
        if cid is None:
            continue
        ent = bars_by_cid[cid]
        ent_row = ent[(ent["minute_et"] <= t["ts"]) & (ent["minute_et"] >= t["ts"] - pd.Timedelta(minutes=3))]
        if ent_row.empty:
            continue
        er = ent_row.iloc[-1]
        b, a = float(er["bid_close"]), float(er["ask_close"])
        entry_mid = (b + a) / 2.0 if b > 0 else float(er["close"])
        if entry_mid < 0.50:
            continue
        path = forward_path(bars_by_cid, cid, t["ts"])
        if path is None:
            continue
        lo = path["low"].values.astype(float)
        cl = path["close"].values.astype(float)
        pm = path["minute_et"]
        cummax_cl = np.maximum.accumulate(cl)
        cummin_lo = np.minimum.accumulate(lo)
        eod = (pm.dt.hour.values * 60 + pm.dt.minute.values >= EOD_FLATTEN_MOD)
        ts_idx = int(np.argmax(eod)) if eod.any() else len(cl) - 1
        if tstop:
            held = (pm - pd.Timestamp(t["ts"])).dt.total_seconds().values / 60.0
            tstop_hit = held >= tstop
            if tstop_hit.any():
                ts_idx = min(ts_idx, int(np.argmax(tstop_hit)))
        comm = COMMISSION_PCT
        for tr in tr_vals:
            tp = entry_mid * (1 + tr)
            tp_idx = int(np.searchsorted(cummax_cl, tp)) if cummax_cl[-1] >= tp else len(cl)
            for rr in rr_vals:
                sr = tr / rr
                sl = entry_mid * (1 - sr)
                sl_idx = int(np.searchsorted(-cummin_lo, -sl)) if cummin_lo[-1] <= sl else len(cl)
                exit_idx = min(tp_idx, sl_idx, ts_idx)
                if exit_idx >= len(cl):
                    exit_px = cl[-1]
                elif tp_idx <= sl_idx and tp_idx == exit_idx:
                    exit_px = tp
                elif sl_idx == exit_idx:
                    exit_px = min(sl, cl[sl_idx])
                else:
                    exit_px = cl[exit_idx]
                pnl_pct = (exit_px - entry_mid) / entry_mid - comm
                for key, thr in flow_pairs:
                    if t["abs_flow"] >= thr:
                        out.append((key, tr, rr, t["date"], pnl_pct))
    return out


# --------------------------------------------------------------------------
def pick_contract(day_bars, ts, direction, target_dte, spot, offset=0):
    """nearest-strike contract of the right type/expiry at/just before ts.

    `offset` steps N strikes OUT of the money (call: higher, put: lower), which
    is what a rule's `strike_offset` field means. It exists because bot_runner
    honoured strike_offset from 2026-09 while THIS function had no way to
    express it -- so every backtest scored the nearest strike no matter what the
    rule declared or the bot bought. GLD amp1 CALL ran that way for weeks.
    Default 0 keeps every existing caller byte-identical.
    """
    at = day_bars[(day_bars["minute_et"] <= ts) & (day_bars["minute_et"] >= ts - pd.Timedelta(minutes=3))]
    at = at[at["option_type"] == ("call" if direction == "CALL" else "put")]
    if at.empty:
        return None
    entry_date = pd.Timestamp(ts).date()
    at = at.assign(dte=(at["expiry"] - entry_date).map(lambda x: x.days))
    at = at[at["dte"] == target_dte]
    if at.empty:
        return None
    # last quote per contract, nearest strike to spot
    at = at.sort_values("minute_et").groupby("option_chain_id").last().reset_index()
    if not offset:
        row = at.iloc[(at["strike"] - spot).abs().argmin()]
        return row["option_chain_id"]
    at = at.sort_values("strike").reset_index(drop=True)
    i = int((at["strike"] - spot).abs().argmin())
    j = i + offset if direction == "CALL" else i - offset
    if j < 0 or j >= len(at):
        return None            # the offset strike is not quoted -- take nothing
    return at.iloc[j]["option_chain_id"]


def forward_path(bars_by_cid, cid, ts, max_days=8):
    df = bars_by_cid.get(cid)
    if df is None:
        return None
    fwd = df[df["minute_et"] > ts].sort_values("minute_et")
    if len(fwd) < 3:
        return None
    return fwd


# --------------------------------------------------------------------------
def _prep(args):
    if getattr(args, "flow_source", FLOW_SOURCE) == "netprem":
        flow = build_flow_netprem(args.hist)
        src = "netprem ($ net_call_premium - net_put_premium, live units)"
    else:
        flow = build_flow_1m(args.lake)
        src = "silver ((ask_vol - bid_vol) * vwap * 100, Lake units)"
    if not os.path.exists(BARS_CACHE):
        sys.exit("run with --build-bars first")
    flow["minute_et"] = _naive(flow["minute_et"])
    flow["date"] = flow["minute_et"].dt.date

    fdays = sorted(flow["date"].unique())
    bspan = pd.read_parquet(BARS_CACHE, columns=["minute_et"])["minute_et"]
    blo, bhi = _naive(bspan).min().date(), _naive(bspan).max().date()
    print(f"  flow source: {src}")
    print(f"  flow cache : {fdays[0]} -> {fdays[-1]}  ({len(fdays)} days)")
    print(f"  bars cache : {blo} -> {bhi}")
    if fdays[0] != blo or fdays[-1] != bhi:
        print("  ⚠️  flow and bars caches cover different windows -- rebuild both: --build-bars")
    if args.split:
        sp = pd.to_datetime(args.split).date()
        n_is = sum(d < sp for d in fdays)
        print(f"  split {sp}: {n_is} IS days / {len(fdays) - n_is} OOS days")
        if n_is == 0 or n_is == len(fdays):
            sys.exit("  one side of the walk-forward split is empty -- widen the data or move --split")
    return flow


_TB_CACHE: dict = {}


def _ticker_bars(tk):
    """Just one ticker's ATM option bars, read straight from the parquet with a
    predicate pushdown -- keeps peak memory to ~1 ticker instead of all 9.

    ONE-SLOT MEMO. A parameter sweep calls this once per (ticker, direction,
    threshold) cell, and since the 2026-09-12 backfill the cache is 152M rows /
    1.56 GB -- re-reading it 150 times costs hours. Holding a single ticker
    keeps the memory profile this function was written for, so callers that
    loop ticker-in-the-outer-position pay one read each. Callers that alternate
    tickers get the old behaviour, not a leak.
    """
    hit = _TB_CACHE.get(tk)
    if hit is not None:
        return hit
    _TB_CACHE.clear()
    tb = pd.read_parquet(BARS_CACHE, filters=[("underlying_symbol", "==", tk)])
    tb["minute_et"] = _naive(tb["minute_et"])
    tb["date"] = tb["minute_et"].dt.date
    _TB_CACHE[tk] = tb
    return tb


def run(args):
    """Single-labelling run. --regime-by gex: buckets are POSITIVE/NEGATIVE_GEX
    (--gex daily = prior-day historical/GEX{T}.parquet, --gex intraday = spot GEX
    at signal). --regime-by trend: buckets are UPTREND/DOWNTREND/CHOP from the
    underlying's daily SMA state (prior day). Feeds the config report / walk-forward."""
    tickers = [t.upper() for t in args.tickers]
    flow = _prep(args)
    use_intraday = REGIME_BY == "gex" and args.gex == "intraday"

    # bucket_key -> the regime value a matching trigger must have
    if REGIME_BY == "gex":
        bucket_values = {"POSITIVE_GEX": "POSITIVE", "NEGATIVE_GEX": "NEGATIVE"}
    else:
        bucket_values = {k: k for k in REGIME_KEYS}

    for tk in tickers:
        gex = load_gex(args.hist, tk) if REGIME_BY == "gex" else {}
        by_date = {}
        if REGIME_BY == "trend":
            by_date = load_trend_regime(args.hist, tk, args.trend_fast, args.trend_slow)
        elif REGIME_BY == "volume":
            by_date = load_volume_regime(args.hist, tk)
        elif REGIME_BY == "amp":
            by_date = {d: f"amp{s}" for d, s in load_amp_score(args.hist, tk, args.trend_fast, args.trend_slow).items()}
        spot_by_date = load_spot_gex(args.spot, tk) if use_intraday else {}
        if use_intraday and not spot_by_date:
            print(f"  ⚠️ {tk}: no spot-GEX in {args.spot} -- falling back to daily label")
        if REGIME_BY in ("trend", "volume", "amp"):
            if not by_date:
                print(f"  ⚠️ {tk}: no daily OHLC in {args.hist}/{tk}.parquet -- skipping")
                continue
            c = Counter(by_date.values())
            print(f"  {tk}: {REGIME_BY} days  " + "  ".join(f"{k} {c.get(k, 0)}" for k in REGIME_KEYS))
        trigs = triggers_for(flow, tk)
        if FLOW_MODE == "pct":
            annotate_flow_pct(trigs, args.flow_window)
        tbars = _ticker_bars(tk)
        bars_by_cid = {cid: g.sort_values("minute_et") for cid, g in tbars.groupby("option_chain_id")}
        bars_by_date = {d: g for d, g in tbars.groupby("date")}

        def regime_of(t):
            if REGIME_BY == "hour":
                return f"h{t['hour']}"
            if REGIME_BY in ("trend", "volume", "amp"):
                return by_date.get(t["date"])
            if use_intraday and spot_by_date:
                r = spot_regime_at(spot_by_date, t["date"], t["ts"])
                if r is not None:
                    return r
            return gex.get(t["date"])

        # results[(regime, direction)][(mf, tr, rr)] = list of (date, pnl_pct)
        results = defaultdict(lambda: defaultdict(list))
        for regime_key in REGIME_KEYS:
            want = bucket_values[regime_key]
            for direction in ("CALL", "PUT"):
                cell = WATCHLIST[tk].get(regime_key, {}).get(direction, {})
                hours = cell.get("hours", ALL_HOURS)
                dtes = cell.get("dte", [0, 1])
                tstop = cell.get("time_stop_mins")
                for t in trigs:
                    if (t["dir"] != direction or t["hour"] not in hours or t["hour"] >= 15
                            or regime_of(t) != want):
                        continue
                    for mf, tr, rr, d, pnl in simulate_trigger(t, direction, dtes, tstop, bars_by_date, bars_by_cid):
                        results[(regime_key, direction)][(mf, tr, rr)].append((d, pnl))

        if args.split:
            report_walkforward(tk, results, pd.to_datetime(args.split).date())
        else:
            report_ticker(tk, results)


def run_compare(args):
    """Score every trigger under BOTH the daily (prior-day) label and the
    intraday spot-GEX label, using regime-agnostic gates (all hours, dte 0/1,
    no time-stop), and show whether the intraday label separates the regimes
    more cleanly. This isolates the regime question from config tuning."""
    tickers = [t.upper() for t in args.tickers]
    flow = _prep(args)

    for tk in tickers:
        gex = load_gex(args.hist, tk)
        spot_by_date = load_spot_gex(args.spot, tk)
        if not spot_by_date:
            print(f"\n  {tk}: no spot-GEX in {args.spot} -- skipping (run spot-gex-build)")
            continue
        trigs = triggers_for(flow, tk)
        if FLOW_MODE == "pct":
            annotate_flow_pct(trigs, args.flow_window)
        tbars = _ticker_bars(tk)
        bars_by_cid = {cid: g.sort_values("minute_et") for cid, g in tbars.groupby("option_chain_id")}
        bars_by_date = {d: g for d, g in tbars.groupby("date")}

        # R[labelling][(regime, direction)][(mf, tr, rr)] -> [(date, pnl)]
        R = {"daily": defaultdict(lambda: defaultdict(list)),
             "intraday": defaultdict(lambda: defaultdict(list))}
        reclass = {"CALL": [0, 0], "PUT": [0, 0]}
        for direction in ("CALL", "PUT"):
            for t in trigs:
                if t["dir"] != direction or t["hour"] not in ALL_HOURS:
                    continue
                rd = gex.get(t["date"])
                if rd is None:
                    continue
                ri = spot_regime_at(spot_by_date, t["date"], t["ts"]) or rd
                reclass[direction][1] += 1
                if ri != rd:
                    reclass[direction][0] += 1
                trades = simulate_trigger(t, direction, [0, 1], None, bars_by_date, bars_by_cid)
                for mf, tr, rr, d, pnl in trades:
                    R["daily"][("POSITIVE_GEX" if rd == "POSITIVE" else "NEGATIVE_GEX", direction)][(mf, tr, rr)].append((d, pnl))
                    R["intraday"][("POSITIVE_GEX" if ri == "POSITIVE" else "NEGATIVE_GEX", direction)][(mf, tr, rr)].append((d, pnl))
        report_compare(tk, R, reclass)


# --------------------------------------------------------------------------
_RULE_REGIMES = {"NEGATIVE_GEX", "POSITIVE_GEX", "UPTREND", "DOWNTREND", "CHOP",
                 "LOWVOL", "NORMVOL", "HIVOL"}


def run_rules(args, rules):
    """Walk-forward a list of FIXED candidate rules (no grid -> nothing to
    overfit). Each rule: ticker, direction, hours, dte, flow_pct|flow_abs,
    target_roe, rr, optional regime / amp_min / time_stop_mins / ema_confirm /
    eod_flatten ("HH:MM") / name."""
    global EOD_FLATTEN_MOD
    # accept config.RULES field names too (min_flow_pct <-> flow_pct)
    for r in rules:
        if "flow_pct" not in r and "min_flow_pct" in r:
            r["flow_pct"] = r["min_flow_pct"]
    flow = _prep(args)
    split = pd.to_datetime(args.split).date() if args.split else None
    # if the user narrowed --tickers (not the default WATCH), run only those rules
    if args.tickers is not WATCH:
        keep = {t.upper() for t in args.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]
        if not rules:
            sys.exit(f"no rules for {sorted(keep)}")
    by_ticker = defaultdict(list)
    for r in rules:
        by_ticker[r["ticker"].upper()].append(r)

    print("\n" + "=" * 100)
    print(f"  CANDIDATE RULES   ({len(rules)} rules"
          + (f"   IS < {split} <= OOS" if split else "   full period") + ")")
    if args.ema_filter:
        print(f"  EMA-stack filter: {args.ema_filter}m bars, spans {args.ema_spans}, "
              f"mode={args.ema_mode} ({'replaces' if args.ema_mode == 'replace' else 'adds to'} the regime gate)")
    _ec = [r.get("name", r["ticker"]) for r in rules if r.get("ema_confirm")]
    if _ec:
        print(f"  per-rule ema_confirm on: {', '.join(_ec)}")
    _ao = [(r.get("name", r["ticker"]), r["amt_open"]) for r in rules if r.get("amt_open")]
    for nm, sp in _ao:
        print(f"  amt_open  {nm}: {sp}")
    print("=" * 100)
    allrows = []
    for tk, tk_rules in by_ticker.items():
        gex = load_gex(args.hist, tk)
        vol = load_volume_regime(args.hist, tk)
        trd = load_trend_regime(args.hist, tk, args.trend_fast, args.trend_slow)
        _days = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _days}
        amt = load_amt_open(args.hist, tk) if any(r.get("amt_open") for r in tk_rules) else {}
        need_pct = any(r.get("flow_pct") is not None for r in tk_rules)
        trigs = triggers_for(flow, tk)
        tbars = _ticker_bars(tk) if trigs else None
        if tbars is None or tbars.empty:
            # trigs and/or ATM bars not in the shared WATCH caches -- build from silver.
            # (in --flow-source netprem the flow is complete for all tickers, so only
            #  the bars half of _screen_build_one is needed here.)
            print(f"  {tk}: not in shared cache, building from silver ...")
            tk_flow, tbars = _screen_build_one(args.lake, tk)
            if not trigs:
                trigs = triggers_for(tk_flow, tk) if tk_flow is not None and not tk_flow.empty else []
        if not trigs or tbars is None or tbars.empty:
            print(f"  {tk}: no data -- skipping rules")
            continue
        if need_pct:
            annotate_flow_pct(trigs, args.flow_window)
        bars_by_cid = {cid: g.sort_values("minute_et") for cid, g in tbars.groupby("option_chain_id")}
        bars_by_date = {d: g for d, g in tbars.groupby("date")}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
                   "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}

        ema_stack = (load_ema_stack(args.hist, tk, args.ema_filter, args.ema_spans)
                     if args.ema_filter else None)
        if args.ema_filter and ema_stack is None:
            print(f"  {tk}: no minute OHLC for EMA stack -- EMA filter skipped for this ticker")
        # per-rule EMA confirmation (config field "ema_confirm": <tf minutes>).
        # cache one stack per (tf, spans) tuple for this ticker.
        rule_ema = {}
        for r in tk_rules:
            tf = r.get("ema_confirm")
            if not tf:
                continue
            sp = tuple(r.get("ema_spans", EMA_SPANS))
            key = (int(tf), sp)
            if key not in rule_ema:
                rule_ema[key] = load_ema_stack(args.hist, tk, int(tf), sp)

        for r in tk_rules:
            name = r.get("name", f"{tk} {r['direction']}")
            direction = r["direction"].upper()
            hours = set(r.get("hours", ALL_HOURS))
            dtes = r.get("dte", [0, 1])
            tstop = r.get("time_stop_mins")
            tr, rr = float(r["target_roe"]), float(r["rr"])
            reg = r.get("regime")
            amp_min = r.get("amp_min")
            if reg is not None and reg not in _RULE_REGIMES:
                sys.exit(f"rule {name!r}: unknown regime {reg!r}; use one of {sorted(_RULE_REGIMES)}")
            if r.get("flow_pct") is not None and int(r["flow_pct"]) not in GRID_MIN_FLOW_PCT:
                sys.exit(f"rule {name!r}: flow_pct must be one of {GRID_MIN_FLOW_PCT}")
            if r.get("flow_pct") is None and r.get("flow_abs") is None:
                sys.exit(f"rule {name!r}: needs flow_pct or flow_abs")

            ema_replaces_regime = args.ema_filter and args.ema_mode == "replace"
            r_ema = rule_ema.get((int(r["ema_confirm"]), tuple(r.get("ema_spans", EMA_SPANS)))) \
                if r.get("ema_confirm") else None
            want_stack = "BULL" if direction == "CALL" else "BEAR"
            amt_spec = r.get("amt_open")
            _eod_saved = EOD_FLATTEN_MOD
            if r.get("eod_flatten") and not args.eod_flatten:
                _eh, _em = map(int, str(r["eod_flatten"]).split(":"))
                EOD_FLATTEN_MOD = _eh * 60 + _em
            pnls = []
            for t in trigs:
                if t["dir"] != direction or t["hour"] not in hours or t["hour"] >= 15:
                    continue
                d = t["date"]
                if amt_spec is not None and not _amt_ok(amt_spec, amt.get(d)):
                    continue
                if r_ema is not None and ema_state_at(r_ema, t["ts"]) != want_stack:
                    continue
                if reg is not None and not ema_replaces_regime:
                    src = gex if reg.endswith("_GEX") else reg_src[reg]
                    val = src.get(d)
                    if reg == "NEGATIVE_GEX" and val != "NEGATIVE": continue
                    if reg == "POSITIVE_GEX" and val != "POSITIVE": continue
                    if reg in _RULE_REGIMES and not reg.endswith("_GEX") and val != reg: continue
                if amp_min is not None and not ema_replaces_regime and amp.get(d, -1) < amp_min:
                    continue
                if ema_stack is not None:
                    st = ema_state_at(ema_stack, t["ts"])
                    if st != ("BULL" if direction == "CALL" else "BEAR"):
                        continue
                if r.get("flow_pct") is not None:
                    thr_map = t.get("thr")
                    if not thr_map or int(r["flow_pct"]) not in thr_map:
                        continue
                    flow_thr = thr_map[int(r["flow_pct"])]
                else:
                    flow_thr = float(r["flow_abs"])
                for _, _, _, dd, pnl in simulate_trigger(t, direction, dtes, tstop,
                                                         bars_by_date, bars_by_cid,
                                                         only=(flow_thr, tr, rr)):
                    pnls.append((dd, pnl))

            EOD_FLATTEN_MOD = _eod_saved
            allrows.append((name, tk, r, pnls))

    def stat(pnls):
        if not pnls:
            return "n=0"
        a = np.array([p for _, p in pnls])
        return f"n={len(a):>4}  exp {a.mean()*100:>+6.1f}%  win {(a > 0.02).mean()*100:>4.1f}%  tot {a.sum()*100:>+7.0f}%"

    for name, tk, r, pnls in allrows:
        gate = []
        if r.get("regime"): gate.append(r["regime"])
        if r.get("amp_min") is not None: gate.append(f"amp>={r['amp_min']}")
        fl = f"p{r['flow_pct']}" if r.get("flow_pct") is not None else f"${r['flow_abs']/1e6:.1f}M"
        print(f"\n  {name}")
        print(f"    {tk} {r['direction']}  h{sorted(r.get('hours', ALL_HOURS))}  dte{r.get('dte', [0, 1])}  "
              f"flow {fl}  TP+{r['target_roe']*100:.0f}% R:R{r['rr']:.1f}"
              + (f"  [{', '.join(gate)}]" if gate else ""))
        if split:
            is_p = [(d, p) for d, p in pnls if d < split]
            oos_p = [(d, p) for d, p in pnls if d >= split]
            print(f"    IS   {stat(is_p)}")
            print(f"    OOS  {stat(oos_p)}")
            if is_p and oos_p:
                ie = np.mean([p for _, p in is_p]) * 100
                oe = np.mean([p for _, p in oos_p]) * 100
                v = ("HOLDS" if oe > 1 and ie > 1 else
                     "decays but stays +" if oe > 0 and ie > 0 else
                     "FAILS OOS" if ie > 1 >= oe else "weak/negative")
                print(f"    => {v}  (IS {ie:+.1f}% -> OOS {oe:+.1f}%)")
        else:
            print(f"    {stat(pnls)}")


TSTOP_GRID = [None, 20, 30, 45, 60, 90, 120]
EOD_GRID = [(15, 0), (15, 15), (15, 30), (15, 45), (15, 55)]


def _rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src):
    """(trigger, flow_$thr) for every trigger `r` (a config.RULES dict) actually
    takes -- direction / hour / regime / amp gates AND abs_flow >= the resolved
    percentile threshold. Mirrors run_rules + simulate_trigger's own flow gate."""
    direction = r["direction"].upper()
    hours = set(r.get("hours", ALL_HOURS))
    reg = r.get("regime")
    amp_min = r.get("amp_min")
    fpct = r.get("min_flow_pct", r.get("flow_pct"))
    out = []
    for t in trigs:
        if t["dir"] != direction or t["hour"] not in hours or t["hour"] >= 15:
            continue
        d = t["date"]
        if reg is not None:
            regs = reg if isinstance(reg, (list, tuple)) else [reg]
            ok = False
            for rg in regs:
                val = (gex if rg.endswith("_GEX") else reg_src[rg]).get(d)
                if rg == "NEGATIVE_GEX":
                    ok = val == "NEGATIVE"
                elif rg == "POSITIVE_GEX":
                    ok = val == "POSITIVE"
                else:
                    ok = val == rg
                if ok:
                    break
            if not ok:
                continue
        if amp_min is not None and amp.get(d, -1) < amp_min:
            continue
        if fpct is not None:
            tm = t.get("thr")
            if not tm or int(fpct) not in tm:
                continue
            thr = tm[int(fpct)]
        elif r.get("flow_abs") is not None:
            thr = float(r["flow_abs"])
        else:
            continue
        if t["abs_flow"] >= thr:
            out.append((t, thr))
    return out


def run_hold_sweep(args, rules):
    """For each enabled config rule, grid EOD-flatten time x time_stop_mins and
    show the OOS (and IS) $ expectancy matrix -- does cutting dead positions
    earlier / flattening before the close-bell theta cliff net out positive?"""
    global EOD_FLATTEN_MOD
    rules = [r for r in rules if r.get("enabled", True)]
    if args.tickers is not WATCH:
        keep = {t.upper() for t in args.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]
    flow = _prep(args)
    split = pd.to_datetime(args.split).date() if args.split else None
    by_ticker = defaultdict(list)
    for r in rules:
        by_ticker[r["ticker"].upper()].append(r)

    print("\n" + "=" * 100)
    print(f"  HOLD-TIME / EOD-FLATTEN SWEEP   ({len(rules)} rules"
          + (f"   IS < {split} <= OOS" if split else "") + ")")
    print(f"  rows = flatten time (ET)   cols = time_stop_mins   cell = OOS exp% (IS exp%)")
    print("=" * 100)

    for tk, tk_rules in by_ticker.items():
        gex = load_gex(args.hist, tk)
        vol = load_volume_regime(args.hist, tk)
        trd = load_trend_regime(args.hist, tk, args.trend_fast, args.trend_slow)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
                   "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        trigs = triggers_for(flow, tk)
        tbars = _ticker_bars(tk) if trigs else None
        if tbars is None or tbars.empty:
            tkf, tbars = _screen_build_one(args.lake, tk)
            if not trigs:
                trigs = triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
        if not trigs or tbars is None or tbars.empty:
            print(f"  {tk}: no data"); continue
        annotate_flow_pct(trigs, args.flow_window)
        bars_by_cid = {c: g.sort_values("minute_et") for c, g in tbars.groupby("option_chain_id")}
        bars_by_date = {d: g for d, g in tbars.groupby("date")}

        for r in tk_rules:
            matched = _rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
            tr, rr = float(r["target_roe"]), float(r["rr"])
            dtes = r.get("dte", [0, 1])
            # precompute each trigger's option path ONCE
            paths = [(t["date"], p) for t, _thr in matched
                     for p in _option_paths(t, r["direction"].upper(), dtes, bars_by_date, bars_by_cid)]
            print(f"\n  {r.get('name', tk)}   ({len(matched)} matched triggers, {len(paths)} filled)")
            print("           " + "".join(f"{('none' if ts is None else str(ts)):>11}" for ts in TSTOP_GRID))
            best = None
            for eh, em in EOD_GRID:
                eod_mod = eh * 60 + em
                cells = []
                for ts in TSTOP_GRID:
                    pnls = [(d, _bracket_pnl(pm[0], pm[1], pm[2], pm[3], pm[4], tr, rr, ts, eod_mod))
                            for d, pm in paths]
                    oos = [p for d, p in pnls if not split or d >= split]
                    isp = [p for d, p in pnls if split and d < split]
                    oe = np.mean(oos) * 100 if oos else None
                    ie = np.mean(isp) * 100 if isp else None
                    cells.append((oe, ie))
                    if oe is not None and (best is None or oe > best[0]):
                        best = (oe, f"{eh}:{em:02d}", ts)
                row = "".join((f"{c[0]:>+6.1f}({c[1]:>+4.1f})" if c[0] is not None and c[1] is not None
                               else f"{c[0]:>+6.1f}     " if c[0] is not None else f"{'--':>11}")
                              for c in cells)
                print(f"    {eh}:{em:02d}  {row}")
            if best:
                print(f"    -> best OOS {best[0]:+.1f}% at flatten {best[1]} / tstop {best[2]}   "
                      f"(baseline 15:55 / none = last row, first col)")


def _option_paths(t, direction, dtes, bars_by_date, bars_by_cid):
    """The forward option price path(s) for one trigger, computed ONCE so a sweep
    can apply many (tstop, eod) exits without re-picking the contract. Returns
    [(entry_mid, cl_arr, lo_arr, mod_arr, held_min_arr), ...] (one per filled dte)."""
    day = bars_by_date.get(t["date"])
    if day is None:
        return []
    at = day[day["minute_et"] <= t["ts"]]
    if at.empty:
        return []
    spot = float(at.iloc[-1]["underlying_close"])
    paths = []
    for target_dte in dtes:
        cid = pick_contract(day, t["ts"], direction, target_dte, spot)
        if cid is None:
            continue
        ent = bars_by_cid[cid]
        er = ent[(ent["minute_et"] <= t["ts"]) & (ent["minute_et"] >= t["ts"] - pd.Timedelta(minutes=3))]
        if er.empty:
            continue
        er = er.iloc[-1]
        b, a = float(er["bid_close"]), float(er["ask_close"])
        entry_mid = (b + a) / 2.0 if b > 0 else float(er["close"])
        if entry_mid < 0.50:
            continue
        fwd = forward_path(bars_by_cid, cid, t["ts"])
        if fwd is None:
            continue
        pm = fwd["minute_et"]
        paths.append((entry_mid,
                      fwd["close"].values.astype(float),
                      fwd["low"].values.astype(float),
                      (pm.dt.hour.values * 60 + pm.dt.minute.values).astype(int),
                      ((pm - pd.Timestamp(t["ts"])).dt.total_seconds().values / 60.0)))
    return paths


def _bracket_pnl(entry_mid, cl, lo, mod, held, tr, rr, tstop, eod_mod):
    """Exit one precomputed option path under (TP tr, R:R rr, tstop, eod_mod)."""
    n = len(cl)
    cummax_cl = np.maximum.accumulate(cl)
    cummin_lo = np.minimum.accumulate(lo)
    eod = mod >= eod_mod
    ts_idx = int(np.argmax(eod)) if eod.any() else n - 1
    if tstop:
        th = held >= tstop
        if th.any():
            ts_idx = min(ts_idx, int(np.argmax(th)))
    tp = entry_mid * (1 + tr)
    sl = entry_mid * (1 - tr / rr)
    tp_idx = int(np.searchsorted(cummax_cl, tp)) if cummax_cl[-1] >= tp else n
    sl_idx = int(np.searchsorted(-cummin_lo, -sl)) if cummin_lo[-1] <= sl else n
    exit_idx = min(tp_idx, sl_idx, ts_idx)
    if exit_idx >= n:
        exit_px = cl[-1]
    elif tp_idx <= sl_idx and tp_idx == exit_idx:
        exit_px = tp
    elif sl_idx == exit_idx:
        exit_px = min(sl, cl[sl_idx])
    else:
        exit_px = cl[exit_idx]
    return (exit_px - entry_mid) / entry_mid - COMMISSION_PCT


def emit_thresholds(args, path):
    """Write flow_pct_thresholds.json = {ticker: {p50,p65,p80,p90,p95, asof, window_days}}
    -- the current trailing-window percentile->$ map, for the cloud bot which
    has no lake access. Run locally, deploy the JSON, refresh ~monthly."""
    flow = _prep(args)
    out = {}
    for tk in [t.upper() for t in args.tickers]:
        trigs = triggers_for(flow, tk)
        if not trigs:                       # not in the shared WATCH cache
            tk_flow, _ = _screen_build_one(args.lake, tk)
            trigs = triggers_for(tk_flow, tk) if tk_flow is not None and not tk_flow.empty else []
        if not trigs:
            print(f"  {tk}: no triggers -- skipped")
            continue
        annotate_flow_pct(trigs, args.flow_window)
        recent = [t for t in sorted(trigs, key=lambda x: x["ts"]) if t.get("thr")]
        if not recent:
            continue
        thr = recent[-1]["thr"]
        out[tk] = {f"p{p}": round(thr[p]) for p in GRID_MIN_FLOW_PCT}
        out[tk]["asof"] = str(pd.Timestamp(recent[-1]["ts"]).date())
        out[tk]["window_days"] = args.flow_window
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {len(out)} tickers -> {path}")


def run_screen(args):
    """Fast per-ticker screen: build flow+bars from silver on the fly (cached in
    _screen_cache/), run the trend/volume/amp cuts, print the single best cell
    per ticker that is POSITIVE BOTH HALVES (n>=MIN_SIGNALS/half, sane win rate),
    else 'nothing'. No config edit, no shared-cache rebuild."""
    if not args.split:
        sys.exit("--screen needs --split (the both-halves test)")
    split = pd.to_datetime(args.split).date()
    print("\n" + "=" * 96)
    print(f"  SCREEN   {len(args.tickers)} tickers   IS < {split} <= OOS   flow-mode {FLOW_MODE}")
    print(f"  best cell POSITIVE BOTH HALVES, n>={MIN_SIGNALS}/half, win<{SUSPECT_WINRATE:.0f}%  (axes: trend/volume/amp)")
    print("=" * 96)
    for tk in [t.upper() for t in args.tickers]:
        flow, bars = _screen_build_one(args.lake, tk)
        if flow is None or flow.empty or bars.empty:
            print(f"  {tk:6}  no silver data")
            continue
        regs = {"trend": load_trend_regime(args.hist, tk, args.trend_fast, args.trend_slow),
                "volume": load_volume_regime(args.hist, tk),
                "amp": {d: f"amp{s}" for d, s in load_amp_score(args.hist, tk, args.trend_fast, args.trend_slow).items()}}
        if not any(regs.values()):
            print(f"  {tk:6}  no historical/{tk}.parquet + GEX{tk}.parquet  (run ohlc-build + gex-daily-build)")
            continue
        trigs = triggers_for(flow, tk)
        if FLOW_MODE == "pct":
            annotate_flow_pct(trigs, args.flow_window)
        bars_by_cid = {cid: g.sort_values("minute_et") for cid, g in bars.groupby("option_chain_id")}
        bars_by_date = {d: g for d, g in bars.groupby("date")}
        res = defaultdict(lambda: defaultdict(list))
        for t in trigs:
            if t["hour"] >= 15:            # global no-entry-after-15:00 cutoff
                continue
            trades = simulate_trigger(t, t["dir"], [0, 1], None, bars_by_date, bars_by_cid)
            if not trades:
                continue
            for axis, rmap in regs.items():
                b = rmap.get(t["date"])
                if b is None:
                    continue
                for fk, tr, rr, d, pnl in trades:
                    res[(axis, b, t["dir"])][(fk, tr, rr)].append((d, pnl))
        best = None
        for (axis, b, direction), grid in res.items():
            for k, pnls in grid.items():
                n_is, e_is, w_is = _stats(pnls, split, "IS")
                n_oos, e_oos, w_oos = _stats(pnls, split, "OOS")
                if n_is < MIN_SIGNALS or n_oos < MIN_SIGNALS or e_is is None or e_oos is None:
                    continue
                if e_is <= 1 or e_oos <= 1:
                    continue
                if (w_is and w_is >= SUSPECT_WINRATE) or (w_oos and w_oos >= SUSPECT_WINRATE):
                    continue
                sc = min(e_is, e_oos)
                if best is None or sc > best[0]:
                    best = (sc, b, direction, k, n_is, n_oos, e_is, e_oos, w_oos)
        if best is None:
            print(f"  {tk:6}  nothing")
        else:
            _, b, direction, (fk, tr, rr), n_is, n_oos, e_is, e_oos, w_oos = best
            print(f"  {tk:6}  {b:9}/{direction:4}  flow {_fmt_flow(fk)} TP+{tr*100:>3.0f}% R:R{rr:.1f}  |  "
                  f"IS n={n_is:>4} +{e_is:>5.1f}%   OOS n={n_oos:>4} +{e_oos:>5.1f}% win {w_oos:>3.0f}%")


def report_ticker(tk, results):
    print("\n" + "=" * 96)
    print(f"  {tk}   (regime: {REGIME_BY})")
    print("=" * 96)
    for regime_key in REGIME_KEYS:
        for direction in ("CALL", "PUT"):
            cell = WATCHLIST[tk].get(regime_key, {}).get(direction, {})
            enabled = cell.get("enabled", False)
            grid = results.get((regime_key, direction), {})
            if not grid and not enabled:
                continue
            cfg_mf = _cfg_flow(cell); cfg_tr = cell.get("target_roe"); cfg_sr = cell.get("stop_roe")
            cfg_rr = round(cfg_tr / cfg_sr, 2) if (cfg_tr and cfg_sr) else None
            hdr = f"\n  --- {regime_key} / {direction}   (config: {'ENABLED' if enabled else 'disabled'}"
            if enabled and cfg_tr:
                fl = f"flow {_fmt_flow(cfg_mf)}, " if cfg_mf is not None else ""
                hdr += f", {fl}TP +{cfg_tr*100:.0f}% / SL -{cfg_sr*100:.0f}% (R:R {cfg_rr})"
            print(hdr + ")")

            rows = []
            for (mf, tr, rr), pnls in grid.items():
                if len(pnls) < MIN_SIGNALS:
                    continue
                arr = np.array([p for _, p in pnls])
                exp = arr.mean() * 100
                wr = (arr > 0.02).mean() * 100
                # a huge win rate on a modest overlapping sample is an artifact --
                # drop it from "best" consideration but still show config's own row
                suspect = wr >= SUSPECT_WINRATE and len(arr) < 250
                rows.append({"mf": mf, "tr": tr, "rr": rr, "n": len(arr),
                             "exp": exp, "wr": wr, "tot": arr.sum() * 100, "suspect": suspect})
            if not rows:
                print(f"      (no grid cells with >= {MIN_SIGNALS} signals -- this recipe barely triggers in the lake)")
                continue
            clean = [r for r in rows if not r["suspect"]] or rows
            clean.sort(key=lambda r: r["exp"], reverse=True)
            rows = clean

            def fmt(r, tag=""):
                mark = " ⚠" if r.get("suspect") else ""
                return (f"      {tag:9} flow {_fmt_flow(r['mf'])}  TP +{r['tr']*100:>3.0f}%  "
                        f"R:R {r['rr']:>3.1f} (SL -{r['tr']/r['rr']*100:>4.1f}%)  |  "
                        f"n={r['n']:>4}  win {r['wr']:>4.1f}%  exp {r['exp']:>+6.1f}%  tot {r['tot']:>+8.0f}%{mark}")

            print("      " + "-" * 88)
            for r in rows[:5]:
                print(fmt(r, "BEST" if r is rows[0] else ""))
            if enabled and cfg_tr:
                cfg_hit = None
                for r in rows:
                    if abs(r["tr"] - cfg_tr) < 1e-6 and abs(r["rr"] - cfg_rr) < 0.05 and _flow_match(r["mf"], cfg_mf):
                        cfg_hit = r; break
                print("      " + "-" * 88)
                if cfg_hit:
                    print(fmt(cfg_hit, "YOUR CFG"))
                    delta = rows[0]["exp"] - cfg_hit["exp"]
                    verdict = "OPTIMAL" if delta < 0.5 else f"beaten by {delta:.1f}pp expectancy"
                    print(f"      => config is {verdict}")
                else:
                    fl = f"flow {_fmt_flow(cfg_mf)} / " if cfg_mf is not None else ""
                    print(f"      YOUR CFG ({fl}TP {cfg_tr*100:.0f}% / R:R {cfg_rr}) "
                          f"not in grid or < {MIN_SIGNALS} signals")


def _best_cell(grid):
    """(key, n, exp%, tot%) of the grid cell with the highest $ expectancy per
    signal, among cells with >= MIN_SIGNALS and not a tiny-sample high-winrate
    artifact. None if nothing qualifies."""
    best = None
    for k, pnls in grid.items():
        if len(pnls) < MIN_SIGNALS:
            continue
        arr = np.array([p for _, p in pnls])
        wr = (arr > 0.02).mean() * 100
        if wr >= SUSPECT_WINRATE and len(arr) < 250:
            continue
        exp = arr.mean() * 100
        if best is None or exp > best[2]:
            best = (k, len(arr), exp, arr.sum() * 100)
    return best


def report_compare(tk, R, reclass):
    print("\n" + "=" * 92)
    print(f"  {tk}   DAILY (prior-day) vs INTRADAY (spot-GEX at signal) regime labelling")
    print("  regime-agnostic gates: hours 9-14, dte [0,1], no time-stop, full min_flow/TP/RR grid")
    print("=" * 92)
    for direction in ("CALL", "PUT"):
        differ, total = reclass[direction]
        if total == 0:
            continue
        print(f"\n  --- {direction}   ({total} signals | intraday label differs from daily on "
              f"{differ} = {100*differ/total:.0f}%)")
        print(f"      {'label':9}{'regime':14}{'n':>6}{'exp%':>9}{'tot%':>10}   best recipe (grid)")
        print("      " + "-" * 78)
        for labelling in ("daily", "intraday"):
            for regime_key in ("POSITIVE_GEX", "NEGATIVE_GEX"):
                grid = R[labelling].get((regime_key, direction), {})
                bc = _best_cell(grid)
                if bc is None:
                    print(f"      {labelling:9}{regime_key:14}{'--':>6}{'':>9}{'':>10}   (< {MIN_SIGNALS} signals)")
                    continue
                (mf, tr, rr), n, exp, tot = bc
                print(f"      {labelling:9}{regime_key:14}{n:>6}{exp:>+9.1f}{tot:>+10.0f}"
                      f"   flow {_fmt_flow(mf)} TP+{tr*100:.0f}% RR{rr:.1f}")
        # separation verdict: does intraday widen the POS-vs-NEG expectancy gap?
        def gap(lab):
            p = _best_cell(R[lab].get(("POSITIVE_GEX", direction), {}))
            n = _best_cell(R[lab].get(("NEGATIVE_GEX", direction), {}))
            return None if (p is None or n is None) else abs(p[2] - n[2])
        gd, gi = gap("daily"), gap("intraday")
        if gd is not None and gi is not None:
            better = "WIDER (intraday separates regimes better)" if gi > gd + 0.5 else \
                     ("about the same" if abs(gi - gd) <= 0.5 else "NARROWER (intraday helps less here)")
            print(f"      => POS-vs-NEG expectancy gap: daily {gd:.1f}pp -> intraday {gi:.1f}pp  [{better}]")


def _stats(pnls, split, side):
    """(n, exp%, win%) for the IS (< split) or OOS (>= split) slice."""
    v = [p for d, p in pnls if (d < split) == (side == "IS")]
    if not v:
        return (0, None, None)
    a = np.array(v)
    return (len(a), a.mean() * 100, (a > 0.02).mean() * 100)


def _find_cell(grid, mf, tr, rr):
    """nearest grid key to a target recipe (flow snapped to nearest grid value)."""
    if tr is None:
        return None
    default = 80 if FLOW_MODE == "pct" else 3.5e6
    mfk = min(ACTIVE_FLOW_GRID, key=lambda x: abs(x - (mf if mf is not None else default)))
    key = (mfk, round(tr, 2), round(rr, 2))
    if key in grid:
        return key
    # tolerate float keys
    for k in grid:
        if abs(k[0] - mfk) < 1 and abs(k[1] - tr) < 1e-6 and abs(k[2] - rr) < 0.05:
            return k
    return None


def report_walkforward(tk, results, split):
    print("\n" + "=" * 104)
    print(f"  {tk}   WALK-FORWARD   (regime: {REGIME_BY}   IS: before {split}   |   OOS: {split}+)")
    print("=" * 104)
    for regime_key in REGIME_KEYS:
        for direction in ("CALL", "PUT"):
            cell = WATCHLIST[tk].get(regime_key, {}).get(direction, {})
            enabled = cell.get("enabled", False)
            grid = results.get((regime_key, direction), {})
            if not grid:
                continue
            cfg_mf = _cfg_flow(cell); cfg_tr = cell.get("target_roe"); cfg_sr = cell.get("stop_roe")
            cfg_rr = round(cfg_tr / cfg_sr, 2) if (cfg_tr and cfg_sr) else None

            # rank grid cells by IN-SAMPLE expectancy (clean = drop tiny high-winrate)
            ranked = []
            for k, pnls in grid.items():
                n_is, e_is, w_is = _stats(pnls, split, "IS")
                if n_is is None or n_is < MIN_SIGNALS:
                    continue
                suspect = w_is is not None and w_is >= SUSPECT_WINRATE and n_is < 200
                if suspect:
                    continue
                ranked.append((e_is, k, n_is))
            if not ranked:
                print(f"\n  {regime_key}/{direction} ({'ENABLED' if enabled else 'disabled'}): "
                      f"too few in-sample signals to fit.")
                continue
            ranked.sort(reverse=True)
            is_best_key = ranked[0][1]

            def line(tag, key):
                if key is None or key not in grid:
                    return f"    {tag:20} (recipe not in grid)"
                mf, tr, rr = key
                n_is, e_is, w_is = _stats(grid[key], split, "IS")
                n_oos, e_oos, w_oos = _stats(grid[key], split, "OOS")
                s_is = f"IS n={n_is:>4} exp {e_is:>+6.1f}%" if e_is is not None else "IS  --"
                s_oos = f"OOS n={n_oos:>4} exp {e_oos:>+6.1f}% win {w_oos:>4.1f}%" if e_oos is not None else "OOS  --"
                return (f"    {tag:20} flow {_fmt_flow(mf)} TP+{tr*100:>3.0f}% R:R{rr:>3.1f}  |  {s_is}   {s_oos}")

            print(f"\n  --- {regime_key} / {direction}   ({'ENABLED' if enabled else 'disabled'})")
            print("      " + "-" * 96)
            print(line("IS-OPTIMAL", is_best_key))
            if enabled and cfg_tr:
                print(line("YOUR CONFIG", _find_cell(grid, cfg_mf, cfg_tr, cfg_rr)))
            print(line("HEURISTIC +100/2.0", _find_cell(grid, cfg_mf, 1.00, 2.0)))

            # verdict
            _, _, _ = ranked[0]
            io_key = is_best_key
            n_oos, e_is_best_oos, _ = _stats(grid[io_key], split, "OOS")
            cfg_key = _find_cell(grid, cfg_mf, cfg_tr, cfg_rr) if (enabled and cfg_tr) else None
            heur_key = _find_cell(grid, cfg_mf, 1.00, 2.0)
            cfg_oos = _stats(grid[cfg_key], split, "OOS")[1] if cfg_key in grid else None
            heur_oos = _stats(grid[heur_key], split, "OOS")[1] if heur_key in grid else None
            bits = []
            if e_is_best_oos is not None:
                bits.append(f"IS-optimal holds OOS: {e_is_best_oos:+.1f}%")
            if cfg_oos is not None:
                bits.append(f"config OOS: {cfg_oos:+.1f}%")
            if heur_oos is not None:
                bits.append(f"heuristic OOS: {heur_oos:+.1f}%")
            print("      => " + "   |   ".join(bits))


# --------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)   # stream progress when piped to a file
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=WATCH)
    ap.add_argument("--lake", default=DEFAULT_LAKE)
    ap.add_argument("--hist", default=DEFAULT_HIST)
    ap.add_argument("--spot", default="lake/silver/spot-exposures-1m",
                    help="1-min spot-GEX silver dir (for --gex intraday / compare)")
    ap.add_argument("--gex", choices=("daily", "intraday", "compare"), default=None,
                    help="regime label: daily (prior-day GEX) | intraday (spot GEX at signal) | "
                         "compare (both, side by side). Default: compare, or intraday when --split.")
    ap.add_argument("--flow-mode", choices=("abs", "pct"), default="abs",
                    help="abs: fixed $ min_flow grid. pct: trailing-percentile grid "
                         "(self-scaling; use this over a multi-year span -- see check_flow_drift.py)")
    ap.add_argument("--flow-window", type=int, default=FLOW_PCT_WINDOW_DAYS,
                    help=f"trailing calendar days for --flow-mode pct (default {FLOW_PCT_WINDOW_DAYS})")
    ap.add_argument("--regime-by", choices=("gex", "trend", "volume", "hour", "amp"), default="gex",
                    help="bucket signals by: gex sign (default) | trend (daily SMA state) | "
                         "volume (underlying daily vol vs trailing median) | hour (ET entry hour) | "
                         "amp (count of {NEG-GEX, LOWVOL, CHOP}, 0-3). Non-gex modes ignore --gex.")
    ap.add_argument("--trend-fast", type=int, default=TREND_FAST, help="fast SMA (daily bars) for trend / amp")
    ap.add_argument("--trend-slow", type=int, default=TREND_SLOW, help="slow SMA (daily bars) for trend / amp")
    ap.add_argument("--rule-file", help="walk-forward a list of fixed rules (no grid). Pass a JSON path, "
                                        "or 'config' to use the enabled entries of config.RULES. "
                                        "Overrides --regime-by / --gex.")
    ap.add_argument("--screen", action="store_true",
                    help="fast per-ticker regime screen from silver (no config edit / cache rebuild). Needs --split.")
    ap.add_argument("--emit-thresholds", metavar="PATH",
                    help="write the percentile->$ flow-threshold JSON for the cloud bot, then exit")
    ap.add_argument("--ema-filter", type=int, choices=(1, 2, 3, 5), default=None,
                    help="RULE-FILE only: require the underlying's price EMA stack (on this many-minute "
                         "bars) to agree with the trade direction (BULL for CALL, BEAR for PUT). "
                         "Hybrid mode -- momentum confirmation from PRICE instead of the flow shape.")
    ap.add_argument("--ema-spans", default="8,21,34",
                    help="comma EMA spans, fast->slow (default 8,21,34). Stack is BULL when "
                         "ema(8) > ema(21) > ema(34).")
    ap.add_argument("--ema-mode", choices=("add", "replace"), default="add",
                    help="add: EMA stack layered ON TOP of the rule's regime gate (default). "
                         "replace: EMA stack REPLACES the rule's regime/amp gate.")
    ap.add_argument("--flow-source", choices=("silver", "netprem"), default="silver",
                    help="silver: (ask_vol - bid_vol) * vwap flow from the option tape (default, 'Lake units'). "
                         "netprem: net_call_premium - net_put_premium from historical/NETPREM{T}.parquet -- "
                         "the EXACT units the live bot consumes. Use with --rule-file / --emit-thresholds "
                         "to re-validate the rules and thresholds in live units.")
    ap.add_argument("--hold-sweep", action="store_true",
                    help="for each enabled config rule, grid EOD-flatten time x time_stop_mins "
                         "and print the OOS/IS expectancy matrix. Needs --split.")
    ap.add_argument("--eod-flatten", metavar="HH:MM",
                    help="override the 15:55 EOD flatten cutoff (one-off; affects --rule-file too)")
    ap.add_argument("--build-bars", action="store_true")
    ap.add_argument("--split", help="walk-forward: entries before this date (YYYY-MM-DD) are IS, rest OOS")
    args = ap.parse_args()
    args.ema_spans = tuple(int(x) for x in str(args.ema_spans).split(",") if x.strip())
    if args.eod_flatten:
        _h, _m = map(int, args.eod_flatten.split(":"))
        EOD_FLATTEN_MOD = _h * 60 + _m
    FLOW_SOURCE = args.flow_source
    FLOW_MODE = args.flow_mode
    ACTIVE_FLOW_GRID = GRID_MIN_FLOW_PCT if FLOW_MODE == "pct" else GRID_MIN_FLOW
    REGIME_BY = args.regime_by
    REGIME_KEYS = {"gex": GEX_REGIME_KEYS, "trend": TREND_REGIME_KEYS, "volume": VOLUME_REGIME_KEYS,
                   "hour": HOUR_REGIME_KEYS, "amp": AMP_REGIME_KEYS}[REGIME_BY]
    if args.gex is None:
        args.gex = "intraday" if (args.split and REGIME_BY == "gex") else "compare"
    if REGIME_BY != "gex":
        args.gex = "daily"   # non-gex regimes don't use GEX; keep run() out of compare
    if args.build_bars:
        for c in (FLOW_CACHE, FLOW_CACHE_NETPREM, BARS_CACHE):
            if os.path.exists(c):
                os.remove(c)
        build_flow_1m(args.lake, force=True)
        if args.flow_source == "netprem" or glob.glob(os.path.join(args.hist, "NETPREM*.parquet")):
            build_flow_netprem(args.hist, force=True)
        build_bars(args.lake)
        print("caches built.")
    elif args.emit_thresholds:
        emit_thresholds(args, args.emit_thresholds)
    elif args.hold_sweep:
        from config import RULES as _CFG_RULES
        run_hold_sweep(args, _CFG_RULES)
    elif args.screen:
        run_screen(args)
    elif args.rule_file:
        if args.rule_file == "config":
            from config import RULES
            rules = [r for r in RULES if r.get("enabled", True)]
        else:
            with open(args.rule_file) as f:
                rules = json.load(f)
        run_rules(args, rules)
    elif args.gex == "compare":
        if args.split:
            sys.exit("--gex compare has no walk-forward split; use --gex daily or intraday with --split")
        run_compare(args)
    else:
        run(args)
