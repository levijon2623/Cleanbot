"""
live_state.py
=============
Emits bot_runner's OWN state to a small JSON so flow_viewer.html can render the
live trigger. Read-only with respect to the engine.

🚨 WHY THE BOT EMITS INSTEAD OF THE VIEWER POLLING UW
    The cumulative net-premium series, its EMA(5), the crossover state and the
    live percentile gate already exist in FlowMomentumTracker and
    _live_flow_threshold. A separate process rebuilding them from the UW API
    would be a SECOND IMPLEMENTATION of the trigger -- which is precisely the
    shape of both divergences found on 2026-09-19/20: `strike_offset` honoured
    by bot_runner but not by research, and a stale entry spot in
    build_candidates that moved the selected strike on up to 81% of candidates.
    A chart that disagrees with the bot is worse than no chart, because it will
    be believed. So the numbers come out of the live objects, via
    FlowMomentumTracker.closed_series, and nothing here recomputes a signal.

🚨 THIS RUNS INSIDE A LIVE TRADING LOOP
    Three rules, none of them negotiable:
      1. `tick()` NEVER raises. Every path is inside one try/except that
         swallows and disables on repeat failure. A charting nicety must not be
         able to kill a process holding real positions.
      2. The write is ATOMIC -- temp file plus os.replace -- so the viewer can
         never read a half-written document and the bot never blocks on a
         reader.
      3. It is THROTTLED to once per MIN_INTERVAL seconds. The monitor loop runs
         at 0.1s; writing a file 600 times a minute would be absurd.

WHAT THE VIEWER GETS, AND WHAT IT CANNOT
    Everything here is already in memory, so the emit costs no UW calls and no
    rate limit. Note one real consequence of bot_runner.py:1708
    (`if ticker in self.active_snipes: continue`): while a position is open, the
    scanner skips that ticker entirely, so its flow series STOPS ADVANCING. That
    is reported honestly as state "holding" with a frozen series rather than
    smoothed over -- if the gap matters, the chart is where it will show up.

Usage (bot_runner, once inside the main loop):
    import live_state
    live_state.tick(self)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
import threading
import time
from zoneinfo import ZoneInfo

# Every UW poller in this file pauses outside 09:25-16:05 ET on trading days.
# UW's Basic plan allows 40,000 requests/day; these three threads plus the main
# loop measured ~84/min around the clock on 2026-09-27. See uw_market_open.
from unusual_whales_client import uw_market_open

CLOSED_SLEEP = 60.0   # seconds between "is the market open yet?" checks

_NY = ZoneInfo("America/New_York")


def _et_stamp(epoch_sec):
    """Real epoch -> the epoch Lightweight Charts must receive to PRINT the ET
    wall clock.

    LWC renders UTC. export_flow_tape stamps naive ET as if it were UTC so the
    axis reads 09:30..16:00, and the live pane MUST use the same convention or
    the viewer shows two different clocks depending on which mode you are in --
    16:44 live against 12:44 historical for the same instant. Handles the
    EDT/EST switch by going through the zone rather than a fixed offset.

    NOTE the top-level `ts` field is deliberately NOT passed through this: the
    viewer computes staleness as now - ts and needs a real epoch for that.
    """
    return int(_dt.datetime.fromtimestamp(epoch_sec, _NY)
               .replace(tzinfo=_dt.timezone.utc).timestamp())

# Anchored to THIS FILE, not the process CWD. bot_runner may be launched by a
# service manager whose WorkingDirectory is not the repo (the cloud tree is
# /root/bot_engine/Cleanbot), and a relative path would then scatter state.json
# somewhere the sync is not looking -- silently, since the emitter swallows its
# own errors by design.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "live", "state.json")
MIN_INTERVAL = 5.0          # seconds between writes
MAX_FAILS = 5               # give up permanently after this many errors

_last = 0.0
_fails = 0
_off = False

# { ticker: { minute_epoch: last spot seen in that minute } }
# The bot keeps only the CURRENT bar in webull.tick_builder.live_bars, so a
# price SERIES has to be accumulated. Done here rather than in bot_runner
# because it is a charting concern and bot_runner should not grow state for it.
# Bucketing by minute and keeping the last sample mirrors
# FlowMomentumTracker.update_and_check exactly, so the price and flow panes
# share an x-axis by construction instead of by coincidence.
_PX = {}
_PX_CAP = 3000              # ~2 sessions, same bound the tracker uses


def _sample_price(eng):
    """Read the in-memory spot for every tracked ticker. No network."""
    try:
        bars = eng.webull.tick_builder.live_bars
    except Exception:
        return
    now_min = int(time.time() // 60)
    for tk, bar in list(bars.items()):
        try:
            px = float(bar["close"])
        except (TypeError, ValueError, KeyError):
            continue
        if px <= 0:
            continue
        b = _PX.setdefault(tk, {})
        b[now_min] = px
        if len(b) > _PX_CAP:
            for k in sorted(b)[:-_PX_CAP]:
                del b[k]


# ---------------------------------------------------------------- VWAP
# { ticker: [[et_stamp, vwap], ...] }  refreshed by a BACKGROUND thread.
_VW = {}
_VW_THREAD = None
# { ticker: [[et_stamp, volume, rvol_or_None], ...] } for today
_VOL = {}


def _mod_epoch(day_iso, mod):
    """(YYYY-MM-DD, minute-of-day) -> real epoch seconds in ET."""
    d = _dt.date.fromisoformat(day_iso)
    return int(_dt.datetime(d.year, d.month, d.day, mod // 60, mod % 60,
                            tzinfo=_NY).timestamp())
VWAP_EVERY = 120.0          # seconds between refreshes of a given ticker
VWAP_STAGGER = 2.0          # seconds between tickers, so 15 do not burst


def _vwap_from_bars(bars):
    """Session VWAP: cum(typical * volume) / cum(volume), typical = (h+l+c)/3.

    UW's ohlc/1m is already RTH-filtered, so the anchor is the 09:30 open --
    a true session VWAP, not one anchored to whenever this process started.
    Bars with no volume are skipped rather than counted as zero, which would
    drag the average toward the last priced bar.
    """
    out, pv, vv = [], 0.0, 0.0
    for b in bars:
        try:
            v = float(b.get("v") or 0.0)
            h, l_, c = float(b["h"]), float(b["l"]), float(b["c"])
            mod = int(b["mod"])
            day = str(b["minute_et"])[:10]
        except (KeyError, TypeError, ValueError):
            continue
        if v <= 0:
            continue
        pv += (h + l_ + c) / 3.0 * v
        vv += v
        if vv <= 0:
            continue
        d = _dt.date.fromisoformat(day)
        stamp = int(_dt.datetime.combine(d, _dt.time(mod // 60, mod % 60),
                                         tzinfo=_dt.timezone.utc).timestamp())
        out.append([stamp, round(pv / vv, 4)])
    return out


# { ticker: {minute_of_day: median volume over the trailing 20 sessions} }
# Built ONCE per ticker per process, then reused -- it only changes daily.
_VOLBASE = {}
RVOL_DAYS = 20


def _volume_baseline(bars, today_iso):
    """{mod: median volume} for that minute across PRIOR sessions.

    MEDIAN, not mean, though 'average' was the ask. Per-minute volume across 20
    sessions has a CV near 0.88 on IWM and is right-skewed -- one 5x day drags a
    mean up and makes every subsequent RVOL read low. Swap `np.median` for
    `np.mean` here if you want the literal average; nothing else changes.

    TIME-OF-DAY, not a flat rolling window, because the intraday smile is
    enormous: IWM's 09:30 minute runs ~12x its midday volume and 15:55 ~5x.
    Dividing by a flat average would make every open look like a 12x RVOL.
    """
    by_mod = {}
    for b in bars:
        d = str(b.get("minute_et", ""))[:10]
        if not d or d >= today_iso:            # prior sessions only
            continue
        try:
            by_mod.setdefault(int(b["mod"]), []).append(float(b["v"]))
        except (KeyError, TypeError, ValueError):
            continue
    import statistics
    return {m: statistics.median(v) for m, v in by_mod.items() if v}


def _vwap_loop(eng):
    """VWAP, today's per-minute volume and the RVOL baseline, from ONE fetch.

    First pass per ticker pulls RVOL_DAYS+1 sessions to build the time-of-day
    baseline; every pass after that pulls a single day. Daemon, and it swallows
    everything -- this thread must never be able to affect trading.
    """
    while True:
        if not uw_market_open():
            time.sleep(CLOSED_SLEEP)
            continue
        try:
            names = sorted(set(getattr(eng, "cumulative_flow", {}) or {}))
            for tk in names:
                try:
                    today = _dt.datetime.now(_NY).date().isoformat()
                    need = tk not in _VOLBASE
                    bars = eng.uw.get_intraday_bars(
                        tk, lookback_days=(RVOL_DAYS + 1 if need else 1),
                        ohlcv=True)
                    if need:
                        base = _volume_baseline(bars, today)
                        if base:
                            _VOLBASE[tk] = base
                    tod = [b for b in bars
                           if str(b.get("minute_et", ""))[:10] == today]
                    if tk in WEEKLY_TICKERS:
                        try:
                            _weekly_session(tk, tod)
                        except Exception as e:      # never reach the loop
                            print(f"  ⚠️ [WEEKLY OHLC] {tk}: {type(e).__name__}: {e}")
                    s = _vwap_from_bars(tod)
                    if s:
                        _VW[tk] = s
                    # BACKFILL the price line from the same bars. _PX otherwise
                    # only accumulates from process start, so every restart
                    # truncated the underlying to "since the bot came up" while
                    # VWAP and volume showed the whole session -- three series
                    # on one pane covering different spans. Existing samples win
                    # (the tick stream is sub-minute; these bars are closed 1m).
                    b_px = _PX.setdefault(tk, {})
                    for b in tod:
                        try:
                            em = _mod_epoch(today, int(b["mod"])) // 60
                            b_px.setdefault(em, float(b["c"]))
                        except (KeyError, TypeError, ValueError):
                            continue
                    base = _VOLBASE.get(tk) or {}
                    vol = []
                    for b in tod:
                        try:
                            m, v = int(b["mod"]), float(b["v"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        ref = base.get(m)
                        vol.append([_et_stamp(_mod_epoch(today, m)), round(v),
                                    round(v / ref, 2) if ref else None])
                    if vol:
                        _VOL[tk] = vol
                except Exception:
                    pass
                time.sleep(VWAP_STAGGER)
            time.sleep(max(VWAP_EVERY - VWAP_STAGGER * max(len(names), 1), 5))
        except Exception:
            time.sleep(30)


def _start_vwap(eng):
    global _VW_THREAD
    if _VW_THREAD is not None or not hasattr(eng, "uw"):
        return
    try:
        import threading
        _VW_THREAD = threading.Thread(target=_vwap_loop, args=(eng,),
                                      name="live-vwap", daemon=True)
        _VW_THREAD.start()
    except Exception:
        _VW_THREAD = None


# ---------------------------------------------------------------- GEX
# { ticker: {"heat": [[strike, gamma], ...], "walls": {...}, "ts": epoch} }
_GEX = {}
# { ticker: [ {e, k, cg, pg, d, t}, ... ] } every row expiring in the next 7
# days from the latest pass. Kept OUT of _GEX, which is serialised into
# state.json every 5s; the weekly page reads it once a morning.
_GEX_NEAR = {}
_GEX_THREAD = None
# 🚨 5 MINUTES, AND THE COST WAS NEVER THE REASON IT WAS 10.
# Checked against the UW dashboard 2026-09-24: the plan is UNLIMITED daily
# requests, 35k used that day against a 344k peak, and ZERO 429s in 24h. (The
# 429 we chased that afternoon was WEBULL's get_order_detail, not UW -- worth
# recording because it sent the whole diagnosis down the wrong path.) A pass
# is ~90 calls across 9 tickers, so halving the interval adds a few thousand
# a day into a budget that has no ceiling.
#
# 🚨 AND WE CANNOT USE THE CHEAP ENDPOINT. /spot-exposures/strike is one call
# instead of ~14, but it aggregates EVERY expiry (483 strikes, 50..1480, no
# `expiry` field) and silently ignores expiry= / date= / expiries= -- the same
# accepted-and-discarded behaviour as `expiry` on expiry-strike. A 0-1DTE heat
# map has to page expiry-strike and filter client-side. Do not "optimise" this
# by switching endpoints; it returns a different quantity wearing the same name.
GEX_EVERY = 300.0           # seconds
GEX_STAGGER = 3.0
# 🚨 8 WAS NOT ENOUGH, AND THE SHORTFALL WAS INVISIBLE.
# The old comment ("6 covers IWM's 32 expiries") was true for IWM and wrong
# for SPY: on 2026-09-24 SPY burned all 8 pages / 4,000 rows and the EARLIEST
# expiry it had seen was 2026-10-30 -- the near-dated rows are not returned
# first, so a 0DTE heat map was built from a window that contained no 0DTE.
# It produced heat=[] , which is indistinguishable from "no strikes near
# spot", so the SPY and QQQ charts simply had no GEX for an unknown length of
# time. Paging now runs until the data ends or STOP_AFTER_MISSES pages in a
# row contain nothing wanted, and a shortfall says so out loud.
GEX_PAGES = 40
STOP_AFTER_MISSES = 6       # consecutive wanted-expiry-free pages -> give up
GEX_BAND = 0.03             # keep strikes within +-3% of spot


def _uw_get(uw, path, **params):
    """One authenticated GET through the bot's own client credentials."""
    import requests
    r = requests.get(uw.base_url + path, headers=uw.headers, params=params,
                     timeout=20)
    if r.status_code != 200:
        return None
    return r.json().get("data")


def _fetch_gex(uw, tk, spot):
    """0-1DTE gamma by strike, plus the all-OI major levels.

    🚨 `expiry` IS ACCEPTED AND SILENTLY IGNORED by spot-exposures/expiry-strike
    -- passing it returns the identical rows. Probed 2026-09-22. So the 0-1DTE
    restriction has to be done HERE, client-side, after paging. Do not "simplify"
    this by trusting the parameter.
    """
    import datetime as _d
    today = _d.datetime.now(_NY).date()
    # Page for the next 7 calendar days, not just 0-1DTE: that covers the
    # next listed expiry across any weekend or holiday, and every remaining
    # expiry of the week for the weekly page (_weekly_update) -- which is
    # therefore built from THESE rows at no extra request cost.
    lo_e, hi_e = today.isoformat(), (today + _d.timedelta(days=7)).isoformat()
    near = []
    pages, hits, misses, exhausted = 0, 0, 0, True
    for page in range(GEX_PAGES):
        rows = _uw_get(uw, f"/api/stock/{tk}/spot-exposures/expiry-strike",
                       limit=500, page=page)
        pages += 1
        if not rows:
            break
        found_here = 0
        for x in rows:
            e = x.get("expiry") or ""
            if not (lo_e <= e <= hi_e):
                continue
            found_here += 1
            try:
                near.append(dict(e=e, k=float(x["strike"]),
                                 cg=float(x.get("call_gamma_oi") or 0),
                                 pg=float(x.get("put_gamma_oi") or 0),
                                 d=x.get("date"), t=x.get("time")))
            except (TypeError, ValueError, KeyError):
                continue
        hits += found_here
        misses = 0 if found_here else misses + 1
        if len(rows) < 500:
            break                       # genuinely the end of the data
        if hits and misses >= STOP_AFTER_MISSES:
            break                       # we have what we came for
    else:
        exhausted = False               # hit the page cap without breaking
    # 🚨 0-1DTE = today + the NEXT LISTED expiry, not today + 1 calendar day.
    # The calendar version asked for Saturday every Friday (and for the
    # holiday before a long weekend), so the heat silently fell to 0DTE-only
    # on exactly those days. Fixed 2026-09-27.
    nxt = min((r["e"] for r in near if r["e"] > lo_e), default=None)
    want = {lo_e} | ({nxt} if nxt else set())
    agg = {}
    for r in near:
        if r["e"] in want:
            agg[r["k"]] = agg.get(r["k"], 0.0) + r["cg"] + r["pg"]
    hits = sum(1 for r in near if r["e"] in want)
    _GEX_NEAR[tk] = near
    # 🚨 SAY SO when the window never contained the near-dated expiries. An
    # empty heat is not the same as no gamma near spot, and conflating them
    # hid a broken SPY/QQQ heat map for an unknown period.
    if not hits:
        print(f"  ⚠️ [GEX] {tk}: {pages} page(s), no rows at {sorted(want)} — "
              f"heat will be EMPTY"
              + ("" if exhausted else f" (hit the {GEX_PAGES}-page cap; the "
                                      f"chain is bigger than the window)"))
    heat = []
    if agg and spot:
        lo, hi = spot * (1 - GEX_BAND), spot * (1 + GEX_BAND)
        heat = sorted([[k, round(v)] for k, v in agg.items()
                       if lo <= k <= hi and abs(v) > 0])
    lv = _uw_get(uw, f"/api/stock/{tk}/gex-levels", source="oi") or {}
    walls = {}
    for key in ("call_wall", "put_wall", "gamma_flip", "gamma_magnet"):
        try:
            walls[key] = float(lv[key])
        except (KeyError, TypeError, ValueError):
            pass
    # 🚨 KEEP nearby_flips, DO NOT COLLAPSE IT TO ONE NUMBER.
    # /gex-levels returns several flip candidates (IWM 2026-09-17 gave
    # ['291.23','277.53','277.48','95.01','94.87']). Picking "the" flip now
    # bakes in a free parameter, which is how check_multiplier went wrong --
    # the whole set is logged and the choice is made at analysis time, once,
    # in the open.
    flips = []
    for v in (lv.get("nearby_flips") or []):
        try:
            flips.append(float(v))
        except (TypeError, ValueError):
            pass
    g = dict(heat=heat, walls=walls, src=lv.get("source"),
             ts=int(time.time()), nearby_flips=flips,
             uw_time=lv.get("time"), uw_date=lv.get("date"))
    _log_levels(tk, g, spot)
    return g


LEVELS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "gex_levels_log.jsonl")


def _log_levels(tk, g, spot):
    """Forward record of the GAMMA FLIP and the other levels, per poll.

    🚨 WHY A NEW FILE AND NOT gex_history_log.jsonl.
    That one is the WebSocket `gex:` stream -- exposure MAGNITUDES (net_gex,
    gamma_1pct_oi) at ~15s throttle. This is REST /gex-levels -- price LEVELS
    at ~10min. Different endpoint, different cadence, different units. They
    join on (ticker, time) when needed; merging them would make both harder
    to read and neither more useful.

    🚨 WHY FORWARD-ONLY, WHEN date= BACKFILLS.
    /gex-levels?date=D works, but every historical response is stamped ~16:14
    ET -- built from that session's FINAL open interest, so applying it to
    that session is lookahead. The lookahead-free substitute (use D-1) drifts
    ~$3.52 a day on IWM, about 1.2% of spot, which is wider than the effect
    any level study could resolve. The live poll has neither problem: it is
    intraday-current and knowable at the time. So the only honest history is
    the one that starts now. Probed 2026-09-24.

    Never raises.
    """
    try:
        w = g.get("walls") or {}
        rec = dict(logged_at=int(time.time()),
                   et=_dt.datetime.now(_NY).isoformat(timespec="seconds"),
                   ticker=tk, spot=_num(spot, 4),
                   gamma_flip=w.get("gamma_flip"),
                   gamma_magnet=w.get("gamma_magnet"),
                   call_wall=w.get("call_wall"), put_wall=w.get("put_wall"),
                   nearby_flips=g.get("nearby_flips") or [],
                   # distance is the quantity every study wants; storing it
                   # avoids every consumer recomputing it differently
                   flip_dist_bp=(
                       _num((spot - w["gamma_flip"]) / w["gamma_flip"] * 1e4, 1)
                       if spot and w.get("gamma_flip") else None),
                   heat_strikes=len(g.get("heat") or []),
                   src=g.get("src"), uw_time=g.get("uw_time"))
        with open(LEVELS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":"),
                               default=str) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- WEEKLY GEX
# The week's 0-4DTE gamma map for weekly_gex.html (operator design, 2026-09-27).
#
#   * ONE SNAPSHOT PER TICKER PER MORNING, from the first GEX pass at or after
#     09:31 ET whose rows carry TODAY's date -- i.e. built on this morning's
#     open interest and a real opening price. Open interest only changes
#     overnight, so a snapshot is a map of where the day's levels STARTED;
#     the heat on the main chart is the intraday one.
#   * Each snapshot covers every REMAINING expiry of the week (today..Friday,
#     holidays removed via market_calendar.trading_days). An expiry's
#     column therefore freezes at the snapshot of its own expiry morning, and
#     elapsed days stay drawn until the week ends.
#   * EVERY snapshot is kept -- Monday's view of Friday, Tuesday's view of
#     Friday... -- so the page can show how a level migrated. A new ISO week
#     starts a new file; the old one stays in live/weekly_gex/ as an archive.
#     Those archives are the only look-ahead-free record of weekly levels:
#     UW's historical GEX is built on that session's FINAL open interest
#     (see _log_levels), so it cannot be reconstructed later.
#   * Costs no extra requests: built from _GEX_NEAR, which _fetch_gex already
#     paged. The price line is sampled from _PX.
#   * SESSION OHLC per day (RTH open / high / low / close, drawn in each
#     elapsed day's column) comes from the 1m bars the VWAP thread already
#     fetches every 2 minutes (_weekly_session) -- true highs and lows, not
#     5-minute closes. A day is marked FINAL by a pass at or after 16:02 ET;
#     the VWAP thread polls until 16:05, so that normally happens the same
#     afternoon. BACKSTOP: a day left unfinal (bot down at the close) is
#     fetched once from ohlc/1m by the next trading day's pass -- and for the
#     week's LAST day, which has no next day in its own week, by the first
#     pass of the following week, written into the archived file. One
#     request per ticker-day, only when needed. Half days need no special
#     case: a fetch after the day ends returns the whole shortened session.
WEEKLY_TICKERS = ("SPY", "QQQ", "IWM")
WEEKLY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live", "weekly_gex")
WEEKLY_AT_MOD = 9 * 60 + 31
WEEKLY_FINAL_MOD = 16 * 60 + 2
WEEKLY_BAND = 0.05          # strikes within +-5% of the snapshot spot
_WK = {"doc": None}
# The GEX thread (snapshots) and the VWAP thread (session OHLC) both write the
# week file; one lock around every read-modify-write keeps them from
# overwriting each other's update.
_WK_LOCK = threading.Lock()
_WK_WAIT_SAID = set()       # (ticker, date) we already said "waiting" for
_WK_BACKFILL_TRIED = set()  # (ticker, date) ohlc/1m backfills attempted
_WK_PREV_DONE = set()       # (ticker, monday) previous-week finalisations run


def _week_days(d):
    from market_calendar import trading_days
    monday = d - _dt.timedelta(days=d.weekday())
    return monday, trading_days(monday, monday + _dt.timedelta(days=4))


def _wk_doc(monday, days):
    doc = _WK["doc"]
    if doc and doc.get("week") == monday.isoformat():
        return doc
    path = os.path.join(WEEKLY_DIR, f"{monday.isoformat()}.json")
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        doc = None
    if not doc or doc.get("week") != monday.isoformat():
        doc = {"week": monday.isoformat(), "tickers": {}}
    doc["days"] = [d.isoformat() for d in days]     # holidays can be added late
    _WK["doc"] = doc
    return doc


def _expiry_heat(rows, spot):
    """One expiry's rows -> heat [[strike, net, call, put]] within the band,
    plus levels DERIVED from it (gex-levels is all-expiry, so per-expiry walls
    have to be computed here and are labelled as such on the page)."""
    agg = {}
    for r in rows:
        a = agg.setdefault(r["k"], [0.0, 0.0])
        a[0] += r["cg"]
        a[1] += r["pg"]
    lo, hi = spot * (1 - WEEKLY_BAND), spot * (1 + WEEKLY_BAND)
    heat = sorted([k, round(c + p), round(c), round(p)]
                  for k, (c, p) in agg.items() if lo <= k <= hi and (c or p))
    if not heat:
        return None
    times = sorted(r["t"] for r in rows if r.get("t"))
    return dict(heat=heat,
                call_wall=max(heat, key=lambda h: h[2])[0],
                put_wall=min(heat, key=lambda h: h[3])[0],
                peak=max(heat, key=lambda h: abs(h[1]))[0],
                uw_time_max=times[-1] if times else None)


def _px_5m_today(tk, today):
    """Today's 5-minute closes from _PX (last sample in each bucket)."""
    out = {}
    for em, px in sorted((_PX.get(tk) or {}).items()):
        t = _dt.datetime.fromtimestamp(em * 60, _NY)
        if t.date() != today:
            continue
        mod = t.hour * 60 + t.minute
        if 570 <= mod < 960:
            out[mod - mod % 5] = round(float(px), 4)
    return [[m, v] for m, v in sorted(out.items())]


def _ohlc_of(bars):
    """[(mod, o, h, l, c), ...] ascending -> {o, h, l, c} of the RTH session."""
    return dict(o=round(bars[0][1], 4), h=round(max(b[2] for b in bars), 4),
                l=round(min(b[3] for b in bars), 4), c=round(bars[-1][4], 4))


def _px_5m_of(bars):
    out = {}
    for mod, _o, _h, _l, c in bars:
        out[mod - mod % 5] = round(c, 4)
    return [[m, v] for m, v in sorted(out.items())]


def _day_bars_1m(uw, tk, day):
    """One completed day's RTH 1m bars from ohlc/1m: [(mod, o, h, l, c)]."""
    rows = _uw_get(uw, f"/api/stock/{tk}/ohlc/1m", date=day.isoformat()) or []
    out = {}
    for x in rows:
        if x.get("market_time") not in (None, "r"):
            continue
        try:
            t = _dt.datetime.fromisoformat(
                str(x["start_time"]).replace("Z", "+00:00")).astimezone(_NY)
            mod = t.hour * 60 + t.minute
            if t.date() == day and 570 <= mod < 960:
                out[mod] = (mod, float(x["open"]), float(x["high"]),
                            float(x["low"]), float(x["close"]))
        except (KeyError, TypeError, ValueError):
            continue
    return [out[m] for m in sorted(out)]


def _wk_write(doc, latest=True):
    doc["updated"] = int(time.time())
    _atomic_write(os.path.join(WEEKLY_DIR, f"{doc['week']}.json"), doc)
    if latest:
        _atomic_write(os.path.join(WEEKLY_DIR, "latest.json"), doc)


def _finalise_day(uw, t, tk, d):
    """Backstop: fetch a completed day once; set its final OHLC and, if the
    stored price line is short, its 5-minute closes. True if anything changed."""
    k = d.isoformat()
    have_final = (t.get("ohlc") or {}).get(k, {}).get("final")
    short_px = len(t["px"].get(k) or []) < 70
    if (have_final and not short_px) or (tk, k) in _WK_BACKFILL_TRIED:
        return False
    _WK_BACKFILL_TRIED.add((tk, k))
    bars = _day_bars_1m(uw, tk, d)
    if not bars:
        return False
    t.setdefault("ohlc", {})[k] = dict(_ohlc_of(bars), final=True)
    px = _px_5m_of(bars)
    if len(px) > len(t["px"].get(k) or []):
        t["px"][k] = px
    return True


def _weekly_session(tk, tod):
    """Today's running RTH OHLC from the VWAP thread's 1m bars (no request).
    FINAL once written by a pass at or after WEEKLY_FINAL_MOD."""
    now = _dt.datetime.now(_NY)
    today = now.date()
    bars = []
    for b in tod:
        try:
            mod = int(b["mod"])
            if 570 <= mod < 960:
                bars.append((mod, float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not bars:
        return
    bars.sort()
    with _WK_LOCK:
        monday, days = _week_days(today)
        if today not in days:
            return
        doc = _wk_doc(monday, days)
        t = doc["tickers"].setdefault(tk, {"snapshots": {}, "px": {}})
        new = dict(_ohlc_of(bars), final=now.hour * 60 + now.minute >= WEEKLY_FINAL_MOD)
        oh = t.setdefault("ohlc", {})
        if oh.get(today.isoformat()) != new and not (oh.get(today.isoformat()) or {}).get("final"):
            oh[today.isoformat()] = new
            _wk_write(doc)


def _weekly_update(uw, tk, spot):
    with _WK_LOCK:
        _weekly_update_locked(uw, tk, spot)


def _weekly_update_locked(uw, tk, spot):
    now = _dt.datetime.now(_NY)
    today = now.date()
    monday, days = _week_days(today)
    if today not in days:
        return
    doc = _wk_doc(monday, days)
    t = doc["tickers"].setdefault(tk, {"snapshots": {}, "px": {}})
    iso = today.isoformat()
    changed = False

    # ---- the previous week's last day(s): finalise into its archive, once
    prev = monday - _dt.timedelta(days=7)
    if (tk, prev) not in _WK_PREV_DONE:
        _WK_PREV_DONE.add((tk, prev))
        path = os.path.join(WEEKLY_DIR, f"{prev.isoformat()}.json")
        try:
            with open(path, encoding="utf-8") as f:
                pdoc = json.load(f)
        except (OSError, ValueError):
            pdoc = None
        pt = (pdoc or {}).get("tickers", {}).get(tk)
        if pt:
            done = False
            for k in pdoc.get("days", []):
                done |= _finalise_day(uw, pt, tk, _dt.date.fromisoformat(k))
            if done:
                _wk_write(pdoc, latest=False)

    # ---- the morning snapshot
    if iso not in t["snapshots"] and now.hour * 60 + now.minute >= WEEKLY_AT_MOD and spot:
        rows = _GEX_NEAR.get(tk) or []
        data_dates = sorted({r.get("d") for r in rows if r.get("d")})
        if rows and data_dates and data_dates[-1] == iso:
            exp = {}
            for d in days:
                if d < today:
                    continue
                h = _expiry_heat([r for r in rows if r["e"] == d.isoformat()], spot)
                if h:
                    exp[d.isoformat()] = h
            if exp:
                t["snapshots"][iso] = dict(
                    taken_et=now.isoformat(timespec="seconds"), spot=round(spot, 4),
                    uw_date=data_dates[-1], exp=exp)
                changed = True
                print(f"  📅 [WEEKLY GEX] {tk}: snapshot {iso} -- "
                      f"{len(exp)} expir{'y' if len(exp) == 1 else 'ies'} "
                      f"({', '.join(sorted(exp))}) at spot {spot:.2f}")
        elif (tk, iso) not in _WK_WAIT_SAID:
            _WK_WAIT_SAID.add((tk, iso))
            print(f"  ⏳ [WEEKLY GEX] {tk}: UW rows are dated "
                  f"{data_dates[-1] if data_dates else 'nothing'}, not {iso} -- "
                  f"waiting for today's open interest before snapshotting.")

    # ---- earlier days this week: final OHLC + full price line (backstop)
    for d in days:
        if d >= today:
            break
        changed |= _finalise_day(uw, t, tk, d)
    s = _px_5m_today(tk, today)
    if s and s != t["px"].get(iso):
        t["px"][iso] = s
        changed = True

    if changed:
        _wk_write(doc)


def _gex_loop(eng):
    """Daemon. Seven UW calls per ticker per pass, so it runs slowly and
    staggered -- and like the VWAP thread it can never reach the trading loop."""
    while True:
        if not uw_market_open():
            time.sleep(CLOSED_SLEEP)
            continue
        try:
            names = sorted({r["ticker"] for r in _rules()})
            for tk in names:
                try:
                    spot = (_PX.get(tk) or {})
                    spot = spot[max(spot)] if spot else None
                    g = _fetch_gex(eng.uw, tk, spot)
                    if g["heat"] or g["walls"]:
                        _GEX[tk] = g
                    if tk in WEEKLY_TICKERS:
                        try:
                            _weekly_update(eng.uw, tk, spot)
                        except Exception as e:      # never reach the loop
                            print(f"  ⚠️ [WEEKLY GEX] {tk}: {type(e).__name__}: {e}")
                    try:
                        cs = _fetch_contract_stats(eng.uw, tk)
                        if cs:
                            _CSTAT.update(cs)
                    except Exception:
                        pass
                except Exception:
                    pass
                time.sleep(GEX_STAGGER)
            time.sleep(max(GEX_EVERY - GEX_STAGGER * max(len(names), 1), 30))
        except Exception:
            time.sleep(60)


def _rules():
    from config import RULES
    return [r for r in RULES if r.get("enabled", True)]


# { option_symbol: {"v": volume, "oi": open_interest, "sweep": sweep_volume} }
# from /option-contracts, refreshed on the GEX thread. Contextual for the strike
# strip, NOT execution-critical -- the order path prices off the MQTT quote,
# which is live to the second. These can be ~10 minutes stale and that is fine.
_CSTAT = {}


def _fetch_contract_stats(uw, tk):
    rows = _uw_get(uw, f"/api/stock/{tk}/option-contracts",
                   limit=500, exclude_zero_vol_chains="true") or []
    out = {}
    for x in rows:
        s = x.get("option_symbol")
        if not s:
            continue
        try:
            out[s] = dict(v=int(x.get("volume") or 0),
                          oi=int(x.get("open_interest") or 0),
                          sweep=int(x.get("sweep_volume") or 0))
        except (TypeError, ValueError):
            continue
    return out


def parse_occ(oid):
    """OCC symbol -> (ticker, expiry, right, strike). Free, no network."""
    tk = "".join(c for c in oid[:6] if c.isalpha())
    r = oid[len(tk):]
    try:
        return (tk, f"20{r[0:2]}-{r[2:4]}-{r[4:6]}", r[6], int(r[7:]) / 1000.0)
    except (IndexError, ValueError):
        return (tk, None, None, None)


def _start_gex(eng):
    global _GEX_THREAD
    if _GEX_THREAD is not None or not hasattr(eng, "uw"):
        return
    try:
        import threading
        _GEX_THREAD = threading.Thread(target=_gex_loop, args=(eng,),
                                       name="live-gex", daemon=True)
        _GEX_THREAD.start()
    except Exception:
        _GEX_THREAD = None


def _merged_volume(eng, tk):
    """Per-minute volume: Webull SNAPSHOT where available, UW elsewhere.

    The UW poll refreshes every VWAP_EVERY seconds, so its last two minutes are
    always stale. Webull's SNAPSHOT stream carries cumulative session volume in
    real time, so the most recent closed minutes come from there and the earlier
    part of the session -- which the stream never saw if the bot started late --
    comes from UW. The RVOL BASELINE still has to be UW: 20 prior sessions are
    history, and the stream only knows today.
    """
    base = _VOLBASE.get(tk) or {}
    out = {}
    for t, v, r in (_VOL.get(tk) or []):        # UW, whole session
        out[t] = [t, v, r]
    try:
        live = eng.webull.tick_builder.minute_volume(tk)
    except Exception:
        live = {}
    for m, v in live.items():
        stamp = _et_stamp(int(m) * 60)
        mod = _dt.datetime.fromtimestamp(int(m) * 60, _NY)
        ref = base.get(mod.hour * 60 + mod.minute)
        out[stamp] = [stamp, int(v), round(v / ref, 2) if ref else None]
    return [out[k] for k in sorted(out)]


# --------------------------------------------------------- SWEEP FLOW
# { ticker: {minute_epoch: cumulative signed sweep premium} }
_SW = {}
_SW_SEEN = {}          # { ticker: set(alert keys) } -- the feed repeats rows
_SW_DAY = {}
_SW_THREAD = None
SWEEP_EVERY = 45.0
SWEEP_STAGGER = 1.5
SWEEP_VOI = 1.0        # volume / open interest floor -- the opening proxy


def _sweep_sign(row):
    """Signed aggressor premium for ONE alert, or None if it is filtered out.

    THE FILTERS, and why each one is here (METHODOLOGY 7 measured the cost of
    not having them on 131M contracts of dte<=1 volume):
      has_multileg  8.4% of volume is multi-leg and the plain calc counts a
                    vertical's long leg as conviction buying and its short leg
                    as conviction selling -- 683% of the net signed quantity.
      has_floor     negotiated/arranged prints are not urgent.
      has_sweep     an ISO takes whatever is available across exchanges; it is
                    the one execution style that cannot be patient.
      vol/OI >= 1   'bought at the ask' agrees with 'position opened' only
                    49.4% of the time, so opening has to be inferred. NOT via
                    `all_opening_trades`: measured 2026-09-22 it is False on
                    200 of 200 alerts -- a strict flag that is essentially
                    never true, and gating on it zeroes the series. Volume
                    exceeding open interest CANNOT be pure closing (you cannot
                    close more contracts than exist), which is the standard
                    heuristic and leaves 54% of sweeps.
    `has_multileg` / `has_floor` are already False on every alert in this feed
    (`has_singleleg` is True on all 200), so those two are no-ops here -- kept
    as guards in case another ticker's feed differs.
    Sign: calls positive, puts negative; magnitude is ask-side minus bid-side
    premium, so selling calls is bearish and selling puts is bullish.
    """
    try:
        if not row.get("has_sweep"):
            return None
        if row.get("has_multileg") or row.get("has_floor"):
            return None
        if float(row.get("volume_oi_ratio") or 0) < SWEEP_VOI:
            return None
        ask = float(row.get("total_ask_side_prem") or 0)
        bid = float(row.get("total_bid_side_prem") or 0)
    except (TypeError, ValueError):
        return None
    sgn = 1.0 if str(row.get("type", "")).lower() == "call" else -1.0
    return sgn * (ask - bid)


def _sweep_loop(eng):
    """Cumulative SWEEP-ONLY flow, beside the all-prints line.

    /flow-alerts is an ALERT feed, not the full tape -- only prints meeting UW's
    criteria appear. So this series is sparse and stepped where cum_flow is
    continuous, and the two are comparable in SHAPE and turning points, never in
    level. It is plotted on its own scale for that reason.
    """
    while True:
        if not uw_market_open():
            time.sleep(CLOSED_SLEEP)
            continue
        try:
            names = sorted({r["ticker"] for r in _rules()})
            today = _dt.datetime.now(_NY).date().isoformat()
            for tk in names:
                try:
                    if _SW_DAY.get(tk) != today:       # new session, start over
                        _SW_DAY[tk] = today
                        _SW[tk] = {}
                        _SW_SEEN[tk] = set()
                    rows = _uw_get(eng.uw, f"/api/stock/{tk}/flow-alerts",
                                   limit=200) or []
                    seen = _SW_SEEN.setdefault(tk, set())
                    buck = _SW.setdefault(tk, {})
                    for r in rows:
                        key = (r.get("created_at"), r.get("option_chain"),
                               r.get("total_premium"))
                        if key in seen:
                            continue
                        seen.add(key)
                        v = _sweep_sign(r)
                        if v is None or v == 0:
                            continue
                        ts = str(r.get("created_at") or "")
                        if not ts:
                            continue
                        try:
                            t = _dt.datetime.fromisoformat(
                                ts.replace("Z", "+00:00")).astimezone(_NY)
                        except ValueError:
                            continue
                        if t.date().isoformat() != today:
                            continue
                        m = int(t.timestamp() // 60)
                        buck[m] = buck.get(m, 0.0) + v
                except Exception:
                    pass
                time.sleep(SWEEP_STAGGER)
            time.sleep(max(SWEEP_EVERY - SWEEP_STAGGER * max(len(names), 1), 10))
        except Exception:
            time.sleep(45)


def _sweep_series(tk):
    """Cumulative, closed minutes only, on the shared ET grid."""
    b = _SW.get(tk) or {}
    if not b:
        return []
    now_m = int(time.time() // 60)
    out, run = [], 0.0
    for m in sorted(b):
        run += b[m]
        if m < now_m:
            out.append([_et_stamp(int(m) * 60), round(run)])
    return out


def _start_sweep(eng):
    global _SW_THREAD
    if _SW_THREAD is not None or not hasattr(eng, "uw"):
        return
    try:
        import threading
        _SW_THREAD = threading.Thread(target=_sweep_loop, args=(eng,),
                                      name="live-sweep", daemon=True)
        _SW_THREAD.start()
    except Exception:
        _SW_THREAD = None


def _closed_px(tk):
    """Closed minutes only -- the in-progress minute is still forming, exactly
    as the flow tracker treats it."""
    b = _PX.get(tk) or {}
    now_min = int(time.time() // 60)
    return [[_et_stamp(int(m) * 60), round(b[m], 4)]
            for m in sorted(b) if m < now_min]


def _forming_px(tk):
    """The IN-PROGRESS minute, as one point. `None` if there is none yet.

    🚨 WHY THIS IS SEPARATE FROM _closed_px RATHER THAN FOLDED INTO IT.
    Dropping the forming minute is what made the price line trail the market
    by up to a minute (median ~30s) while the `spot` readout beside it was
    current -- the chart and the number disagreed, which is exactly the kind
    of divergence this codebase keeps getting bitten by.

    But the flow series and its EMA MUST stay closed-only: the trigger is
    defined on closed-minute cumulative flow and the backtest computes it
    that way. So the fix is not to relax _closed_px -- it is to ship the
    forming point separately, let the PRICE line reach the present, and leave
    the flow pane on the grid the trigger actually uses. A line series has no
    OHLC to be wrong about; a forming point is just "the price now".

    🚨 STAMPED AT NOW, NOT AT THE MINUTE BOUNDARY.
    Closed points are stamped at their minute's START, which is correct for
    them but means the drawn right edge sits 60-120s behind the market
    (measured 96s on 2026-09-24 at 15:00:36, last point 14:59). The forming
    value is whatever _sample_price last read -- at most MIN_INTERVAL old --
    so stamping it at the current second puts the line where the price
    actually is. It lands off the minute grid on purpose; the flow pane keeps
    its own grid because the trigger lives there.
    """
    b = _PX.get(tk) or {}
    now = time.time()
    now_min = int(now // 60)
    if now_min not in b:
        return None
    return [_et_stamp(int(now)), round(b[now_min], 4)]


def _atomic_write(path, payload):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, path)          # atomic on Windows and POSIX
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _num(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, nd)


def decision_context(eng, tk):
    """The scalar market state for ONE ticker, right now.

    🚨 THE FIELD NAMES MIRROR flow_viewer's noteCtx() ON PURPOSE. A note says
    what you thought before the outcome; an execution row says what happened.
    They are only joinable into a dataset if they describe the state in the
    same words, so the two schemas are kept identical deliberately -- change
    one and change the other.

    Cheap: reads the caches the emitter already maintains (_PX, _VW, _GEX,
    the flow tracker), no network. Never raises -- a context that cannot be
    built must not stop a fill being recorded.
    """
    out = dict(ticker=tk, ts=int(time.time()), session=None, rule=None,
               spot=None, vwap=None, vwap_bp=None, cum=None, gate=None,
               gate_pct=None, ema=None, rvol=None,
               regime=dict(vol=None, trend=None, gex=None, amp=None),
               gex_call_wall=None, gex_put_wall=None, gex_flip=None,
               state=None, n_open=None)
    try:
        out["session"] = str(getattr(eng, "session_date", "") or "") or None
        out["n_open"] = len(getattr(eng, "active_snipes", {}) or {})

        rule = None
        try:
            rule = _rule_for(eng, tk)
        except Exception:
            rule = None
        if rule:
            out["rule"] = rule.get("name")

        px = (_PX.get(tk) or {})
        spot = px[max(px)] if px else None
        out["spot"] = _num(spot, 4)

        vw = _VW.get(tk) or []
        vwap = vw[-1][1] if vw else None
        out["vwap"] = _num(vwap, 4)
        if spot and vwap:
            out["vwap_bp"] = _num((spot - vwap) / vwap * 1e4, 1)

        cum = (getattr(eng, "cumulative_flow", {}) or {}).get(tk)
        out["cum"] = _num(cum, 0)
        if rule is not None:
            try:
                out["gate"] = _num(eng._live_flow_threshold(tk, rule), 0)
            except Exception:
                pass
        if out["cum"] is not None and out["gate"]:
            out["gate_pct"] = _num(abs(out["cum"]) / out["gate"] * 100, 1)

        tracker = getattr(eng, "flow_tracker", None)
        if tracker is not None:
            try:
                _m, _s, ema = tracker.closed_series(tk)
                out["ema"] = _num(ema[-1], 0) if len(ema) else None
            except Exception:
                pass

        v = _merged_volume(eng, tk)
        if v:
            out["rvol"] = _num(v[-1][2], 2)

        rg = (getattr(eng, "session_regime", {}) or {}).get(tk) or {}
        out["regime"] = dict(vol=rg.get("volume"), trend=rg.get("trend"),
                             gex=rg.get("gex"), amp=rg.get("amp"))

        g = _GEX.get(tk) or {}
        w = g.get("walls") or {}
        out["gex_call_wall"] = _num(w.get("call_wall"), 2)
        out["gex_put_wall"] = _num(w.get("put_wall"), 2)
        out["gex_flip"] = _num(w.get("gamma_flip"), 2)

        out["state"] = ("holding" if (getattr(eng, "active_snipes", {}) or {})
                        .get(tk) else "scanning")
    except Exception as e:                      # noqa: BLE001 -- deliberate
        out["context_error"] = f"{type(e).__name__}: {e}"
    return out


def _rule_for(eng, tk):
    """The enabled rule for this ticker, if any -- for the `rule` context
    field only. Mirrors how snapshot() resolves it."""
    for r in _rules():
        if r.get("ticker") == tk:
            return r
    return None


def snapshot(eng):
    """Build the payload from the engine's live objects. Pure reads."""
    from config import RULES
    _sample_price(eng)
    _start_vwap(eng)          # no-op after the first call
    _start_gex(eng)           # ditto
    _start_sweep(eng)
    tk_state = {}
    tracker = getattr(eng, "flow_tracker", None)
    rules_by_tk = {}
    for r in RULES:
        if r.get("enabled", True):
            rules_by_tk.setdefault(r["ticker"], r)

    tickers = set(getattr(eng, "cumulative_flow", {}) or {})
    tickers |= set(getattr(eng, "active_snipes", {}) or {})
    for tk in sorted(tickers):
        rule = rules_by_tk.get(tk)
        mins, series, ema = ([], [], [])
        if tracker is not None:
            mins, series, ema = tracker.closed_series(tk)
        snipe = (getattr(eng, "active_snipes", {}) or {}).get(tk)

        thr = None
        if rule is not None:
            try:
                thr = _num(eng._live_flow_threshold(tk, rule), 0)
            except Exception:
                thr = None

        pos = None
        if snipe:
            entry = _num(snipe.get("entry_price"))
            bid = _num(snipe.get("last_good_bid"))
            pos = dict(
                dir="CALL" if snipe.get("is_call") else "PUT",
                entry=entry, bid=bid,
                tp=_num(snipe.get("tp_limit")), sl=_num(snipe.get("sl_limit")),
                trail=_num(snipe.get("trail_stop")),
                peak=_num(snipe.get("peak_bid")),
                strike=_num(snipe.get("strike"), 2), dte=snipe.get("dte"),
                rule=snipe.get("regime"),
                roe=(_num((bid - entry) / entry * 100, 1)
                     if (entry and bid and entry > 0) else None),
            )

        # session_regime is {trend, volume, gex, amp}; the historical tape uses
        # {trend, vol, gex, amp}. Normalise to the tape's shape so the viewer
        # renders live and historical with ONE code path -- otherwise the live
        # header silently prints [object Object] while the historical one is
        # fine, which is the kind of divergence that survives review.
        _rg = (getattr(eng, "session_regime", {}) or {}).get(tk) or {}
        tk_state[tk] = dict(
            rule=(rule or {}).get("name"),
            direction=(rule or {}).get("direction"),
            min_flow_pct=(rule or {}).get("min_flow_pct"),
            regime=(dict(gex=_rg.get("gex"), vol=_rg.get("volume"),
                         trend=_rg.get("trend"), amp=_rg.get("amp"))
                    if _rg else None),
            thr=thr,
            cum=_num((getattr(eng, "cumulative_flow", {}) or {}).get(tk), 0),
            # ET-stamped so the axis matches the historical tape -- see _et_stamp
            flow=[[_et_stamp(int(m) * 60), _num(v, 0)]
                  for m, v in zip(mins, series)],
            ema=[[_et_stamp(int(m) * 60), _num(v, 0)]
                 for m, v in zip(mins, ema)],
            # Underlying, same minute grid. Accumulates from when live_state
            # STARTED, not from the open -- a bot restart resets it, and there
            # is no backfill because the bot holds only the current bar.
            px=_closed_px(tk),
            # the forming minute, shipped separately so the price line can
            # reach NOW while flow/EMA stay on the closed grid the trigger
            # is defined on -- see _forming_px
            px_live=_forming_px(tk),
            # Session VWAP, anchored at the 09:30 open (not at process start).
            # Refreshed on a BACKGROUND thread -- see _vwap_loop.
            vwap=_VW.get(tk) or [],
            # [[et_stamp, volume, rvol], ...] -- rvol is this minute's volume
            # over the MEDIAN of that same minute across the prior 20 sessions.
            vol=_merged_volume(eng, tk),
            rvol_days=RVOL_DAYS if tk in _VOLBASE else None,
            # 0-1DTE gamma by strike + the all-OI call/put wall and gamma flip.
            # Refreshed on a BACKGROUND thread every ~10m -- see _gex_loop.
            gex=_GEX.get(tk),
            # Sweep-only, single-leg, opening cumulative premium. Sparse by
            # construction -- /flow-alerts is an alert feed, not the tape --
            # so the viewer gives it its own scale. See _sweep_loop.
            sweep=_sweep_series(tk),
            spot=_num((_PX.get(tk) or {}).get(
                max((_PX.get(tk) or {0: 0}), default=0)), 4),
            # see the module docstring: the scanner skips busy tickers, so a
            # holding ticker's series is FROZEN, not merely quiet.
            state=("holding" if snipe else "scanning"),
            position=pos,
        )

    # Contracts the bot is ALREADY subscribed to, with live quotes -- the exact
    # universe a discretionary order may pick from. No new subscriptions, and
    # nothing tradeable that we cannot price.
    #
    # Since 2026-09-23 this includes SPY/QQQ/IWM PUTS, which no enabled rule
    # asks for. `bot: False` marks them: the contract is tradeable BY HAND but
    # the scanner can never enter it, because direction comes from config.RULES
    # (_match_rule), not from what happens to be in the basket.
    chains = {}
    try:
        import manual_orders
        botdirs = {(r["ticker"], "C" if r["direction"] == "CALL" else "P")
                   for r in _rules()}
        today = _dt.datetime.now(_NY).date()
        lq = eng.webull.tick_builder.live_option_quotes or {}
        for oid, q in lq.items():
            tk, exp, right, strike = parse_occ(oid)
            if tk not in manual_orders.ALLOWED or strike is None:
                continue
            b, k = float(q.get("bid") or 0), float(q.get("ask") or 0)
            if b <= 0 or k <= 0:
                continue
            spot = (_PX.get(tk) or {})
            spot = spot[max(spot)] if spot else None
            # ITM / ATM / OTM for the strike strip's colour guard. ATM is the
            # strike nearest spot per (ticker, right, EXPIRY) -- per expiry
            # because QQQ and IWM carry 0DTE and 1DTE at once, and resolving it
            # per (ticker, right) alone marked one expiry's ATM and left the
            # other side of the strip with no blue row at all.
            itm = (None if spot is None else
                   (strike < spot if right == "C" else strike > spot))
            try:
                dte = (_dt.date.fromisoformat(exp) - today).days
            except (TypeError, ValueError):
                dte = None
            st = _CSTAT.get(oid) or {}
            mid = (b + k) / 2
            chains.setdefault(tk, []).append(dict(
                id=oid, strike=strike, right=right, expiry=exp, dte=dte,
                bid=round(b, 2), ask=round(k, 2), mid=round(mid, 2),
                limit=min(round(mid + 0.01, 2), round(k, 2)),
                spread_pct=round((k - b) / mid * 100, 1) if mid > 0 else None,
                vol=st.get("v"), oi=st.get("oi"), atm=False, itm=itm,
                bot=bool((tk, right) in botdirs)))
        for tk, rows in chains.items():
            sp = (_PX.get(tk) or {})
            sp = sp[max(sp)] if sp else None
            if sp is not None:
                for key in {(x["right"], x["expiry"]) for x in rows}:
                    side = [x for x in rows
                            if (x["right"], x["expiry"]) == key]
                    min(side, key=lambda x: abs(x["strike"] - sp))["atm"] = True
            # expiry before strike: the strip groups by expiry, and without it
            # 0DTE and 1DTE interleave into repeated strikes with no way to
            # tell which row is which.
            rows.sort(key=lambda x: (x["right"], x["expiry"] or "", x["strike"]))
    except Exception:
        chains = {}

    # 🚨 `armed` NOW MEANS CAPABILITY *AND* AN OPEN WINDOW.
    # The env flag alone used to be it. With the viewer reachable from a
    # phone, a left-open tab must not stay armed, so the session arm expires
    # and the chart shows the countdown -- the banner has to reflect what
    # _execute will actually do, not what the config file says.
    _cap, _armed, _left, _side = False, False, 0.0, None
    try:
        import manual_orders
        from config import MANUAL_TRADING_ARMED as _cap
        _armed = manual_orders.is_armed(eng)
        _left = manual_orders.arm_left(eng)
        _side = manual_orders.sideline_state()
    except Exception:
        pass

    # 🚨 THE MANUAL CAP IS NOT `equity`. `equity` below is the bot's bankroll,
    # which under DRY_RUN is PAPER_EQUITY ($30k) on purpose. Manual orders
    # spend real cash, so the desk sizes off the real balance and the viewer
    # has to show THAT number or the premium bar is measured against money
    # that does not exist.
    manual_cap, cap_basis, real_cash = None, None, None
    try:
        import manual_orders
        manual_cap, cap_basis = manual_orders.max_premium(eng)
        real_cash = manual_orders.account_cash(eng)
    except Exception:
        pass

    return dict(
        ts=int(time.time()),
        # armed state comes from the ENGINE, never from the viewer's own config
        armed=bool(_armed),
        arm_capable=bool(_cap),          # MANUAL_TRADING_ARMED in config
        arm_secs_left=int(_left),        # 0 = window closed
        arm_window_s=int(getattr(__import__("manual_orders"),
                                 "ARM_WINDOW_S", 900)),
        sidelined=bool(_side),            # done for the day (manual_orders)
        sideline_at=(_side or {}).get("at"),
        manual=dict(getattr(eng, "manual_holds", {}) or {}),
        manual_cap=_num(manual_cap, 2),
        manual_cap_basis=cap_basis,
        real_cash=_num(real_cash, 2),
        chains=chains,
        session=str(getattr(eng, "session_date", "") or ""),
        dry_run=bool(globals().get("_DRY", True)),
        equity=_num(getattr(eng, "account_equity", None), 2),
        n_open=len(getattr(eng, "active_snipes", {}) or {}),
        tickers=tk_state,
    )


def tick(eng, path=OUT, force=False):
    """Throttled, atomic, never-raises. Call once per main-loop pass."""
    global _last, _fails, _off
    if _off:
        return False
    now = time.time()
    if not force and (now - _last) < MIN_INTERVAL:
        return False
    try:
        try:
            from config import DRY_RUN
            globals()["_DRY"] = DRY_RUN
        except Exception:
            pass
        _atomic_write(path, snapshot(eng))
        _last, _fails = now, 0
        return True
    except BaseException as e:                 # noqa: BLE001 -- deliberate
        _last = now
        _fails += 1
        if _fails >= MAX_FAILS:
            _off = True
            try:
                print(f"  ⚠️  live_state disabled after {_fails} failures "
                      f"({type(e).__name__}: {e}). Trading is unaffected.")
            except Exception:
                pass
        return False
