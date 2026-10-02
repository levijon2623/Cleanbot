"""
webull_viewer_data.py
=====================
The viewer's market data from WEBULL alone -- the "budget" path, so the live
chart and the manual desk run without an Unusual Whales subscription.

    VIEWER_DATA=auto     (default) Unusual Whales when UW_API_KEY is set, else Webull
    VIEWER_DATA=uw       Unusual Whales for everything below
    VIEWER_DATA=webull   Webull for everything below, even with a UW key

🚨 A SETTING, RESOLVED ONCE, NEVER A SILENT FALLBACK.
    Mixing two vendors on one chart without saying so is the failure this repo
    keeps paying for: an empty heat that looked like "no gamma", a feed that
    looked live and was stale. The source is chosen at startup, held for the
    process, and printed on the chart. If Webull data is missing, the chart
    shows it missing -- it does not quietly borrow from UW, or the reverse.

WHAT EACH PIECE IS BUILT FROM, IN WEBULL MODE
    GEX heat (0-1DTE, weekly, +index)  get_option_snapshot greeks + OI, the SAME
        rows ({e, k, cg, pg, d, t}) live_state builds from UW, in the SAME units
        (OI x gamma x 100 x S^2 x 1%), so every heat/weekly/blend function
        downstream runs unchanged. Validated 2026-10-01 (check_webull_gex.py,
        PASS: median r 0.998-0.999 vs UW on SPY/QQQ/IWM, 77 samples each).
    +index ratio  index forward by put-call parity on its own 0DTE mids over the
        ETF's Webull spot -- Webull has no index quote (SPX/NDX/RUT are
        INVALID_SYMBOL in US_STOCK and US_INDEX, probed 2026-10-01).
    VWAP, volume, RVOL  get_history_bar M1 (RTH only, newest first, <= 1650 per
        call, end_time in epoch ms pages back -- probed 2026-10-02).
    strike-strip volume / OI  from the same option snapshots as the heat.
    NOT AVAILABLE  the all-expiry walls + gamma flip (UW /gex-levels: needs the
        whole chain), cumulative flow and sweeps (Webull option prints carry no
        exchange or condition codes -- see webull_gamma_client).

🚨 SPY + SPX WALLS ARE NOT A SINGLE STRIKE ON WEBULL DATA.
    check_webull_blend.py (2026-10-01) passed QQQ+NDX and IWM+RUT on every
    criterion, and SPY+SPX on shape (r 0.976), peak (84%) and ratio (0.046%) --
    but its walls agreed with UW within one strike only 50% of the time.
    Several SPX strikes carry near-equal gamma, so which one is "the" wall flips
    between sources. live_state therefore flags any wall whose runner-up is
    within WALL_TIE of it (index_gex.walls_with_ties), and the viewer says so.

REST only. Its own DataClient on the bot's token file (reused, never replaced);
never a second streaming connection.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import math
import os
import threading
import time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
BATCH, PAUSE = 20, 0.3        # option snapshots per call, seconds between calls
BAR_MAX = 1650                # get_history_bar count ceiling (ILLEGAL_PARAMETER above)
DIR_TTL = 900                 # option directory cache, seconds

_SOURCE = None
_LOCK = threading.Lock()
_DATA = None                  # this module's own DataClient
_DIRS = type("Dirs", (), {})()  # stub for WebullGammaClient._option_directory's cache
STATS = dict(calls=0, fails=0, last_fail=None)


def source():
    """'uw' or 'webull', resolved ONCE per process from VIEWER_DATA."""
    global _SOURCE
    if _SOURCE is None:
        want = (os.getenv("VIEWER_DATA") or "auto").strip().lower()
        has_uw = bool((os.getenv("UW_API_KEY") or "").strip())
        if want == "uw" and not has_uw:
            print("  ⚠️ VIEWER_DATA=uw but UW_API_KEY is not set -- using Webull data")
            want = "webull"
        _SOURCE = want if want in ("uw", "webull") else ("uw" if has_uw else "webull")
        print(f"  📊 Viewer market data: {_SOURCE.upper()}"
              + ("" if want != "auto" else " (VIEWER_DATA=auto)"))
    return _SOURCE


def _client():
    global _DATA
    with _LOCK:
        if _DATA is None:
            from webull.core.client import ApiClient
            from webull.data.data_client import DataClient
            api = ApiClient(os.environ["WEBULL_APP_KEY"], os.environ["WEBULL_APP_SECRET"],
                            region_id="us")
            api.add_endpoint("us", "api.webull.com")
            _DATA = DataClient(api)
            _DIRS.data_client, _DIRS._DIR_TTL = _DATA, DIR_TTL
        return _DATA


def _json(resp):
    return resp.json() if hasattr(resp, "json") else resp


def _rows(obj):
    if isinstance(obj, dict) and "data" in obj:
        obj = obj["data"]
    if isinstance(obj, dict):
        obj = [obj]
    return obj or []


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _call(fn, **kw):
    STATS["calls"] += 1
    try:
        return _rows(_json(fn(**kw)))
    except Exception as e:                       # noqa: BLE001 -- counted, shown
        STATS["fails"] += 1
        STATS["last_fail"] = f"{time.strftime('%H:%M:%S')} {type(e).__name__}"
        return None


# ------------------------------------------------------------------ bars
def _bar_dict(x):
    """Webull M1 bar -> the dict UnusualWhalesClient.get_intraday_bars(ohlcv=True)
    returns: minute_et 'YYYY-MM-DDTHH:MM' (ET), mod, o/h/l/c/close, v. None if
    outside 09:30-16:00 ET or unparseable."""
    try:
        t = dt.datetime.strptime(str(x["time"]), "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(NY)
        mod = t.hour * 60 + t.minute
        if mod < 570 or mod > 960:
            return None
        c = float(x["close"])
        return dict(minute_et=t.strftime("%Y-%m-%dT%H:%M"), close=c, mod=mod,
                    o=float(x["open"]), h=float(x["high"]), l=float(x["low"]), c=c,
                    v=float(x.get("volume") or 0.0))
    except (KeyError, TypeError, ValueError):
        return None


def bars_1m(tk, sessions=1):
    """Today's (sessions=1) or the last N sessions' RTH 1-minute bars, ascending,
    in UW's bar shape. Pages back with end_time until N distinct days are held."""
    data = _client()
    out, days, end = {}, set(), None
    for _ in range(12):                          # hard stop: ~18 sessions per 4 calls
        kw = dict(symbol=tk, category="US_STOCK", timespan="M1", count=str(BAR_MAX))
        if end is not None:
            kw["end_time"] = str(end)
        got = _call(data.market_data.get_history_bar, **kw)
        if not got:
            break
        oldest = None
        for x in got:
            b = _bar_dict(x)
            if b:
                out[b["minute_et"]] = b
                days.add(b["minute_et"][:10])
            try:
                ms = int(dt.datetime.strptime(str(x["time"]), "%Y-%m-%dT%H:%M:%S.%f%z")
                         .timestamp() * 1000)
                oldest = ms if oldest is None else min(oldest, ms)
            except (KeyError, TypeError, ValueError):
                pass
        if len(days) >= sessions or oldest is None or len(got) < BAR_MAX:
            break
        end = oldest - 60_000
    keep = sorted(days)[-sessions:]
    return [out[k] for k in sorted(out) if k[:10] in keep]


# ------------------------------------------------------------------ options
def _directory(tk):
    import webull_gamma_client as W
    _client()
    return W.WebullGammaClient._option_directory(_DIRS, tk)


def expiries_for(syms, today, days):
    """Listed expiries (ISO) in [today, today+days]."""
    lo, hi = today, today + dt.timedelta(days=days)
    out = set()
    for s in syms:
        try:
            e = dt.date(2000 + int(s[-15:-13]), int(s[-13:-11]), int(s[-11:-9]))
        except (ValueError, IndexError):
            continue
        if lo <= e <= hi:
            out.add(e.isoformat())
    return sorted(out)


def want_01(listed, today_iso):
    """0-1DTE = today + the NEXT LISTED expiry (live_state._want_01's rule)."""
    nxt = min((e for e in listed if e > today_iso), default=None)
    return {today_iso} | ({nxt} if nxt else set())


def snapshot_rows(tk, center, band, expiries, s_for_units=None):
    """Option snapshots for `tk` within center*(1 +- band) on `expiries`.

    -> (rows [{e, k, cg, pg, d, t}], quotes {sym: (bid, ask)}, stats {sym: {v, oi}},
        coverage, n). cg/pg in UW's units: OI x gamma x 100 x S^2 x 1%, calls +,
        puts -. S is `s_for_units` (the index FORWARD for an index) or center."""
    data = _client()
    syms = _directory(tk)
    exp6 = {e[2:4] + e[5:7] + e[8:10]: e for e in expiries}
    lo, hi = center * (1 - band), center * (1 + band)
    want = [s for s in syms if s[-15:-9] in exp6 and lo <= int(s[-8:]) / 1000 <= hi]
    got = {}
    for i in range(0, len(want), BATCH):
        for x in _call(data.option_market_data.get_option_snapshot,
                       symbols=",".join(want[i:i + BATCH]), category="US_OPTION") or []:
            if isinstance(x, dict) and x.get("symbol"):
                got[x["symbol"]] = x
        time.sleep(PAUSE)
    S = s_for_units or center
    today = dt.datetime.now(NY).date().isoformat()
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    agg, quotes, stats, usable = {}, {}, {}, 0
    for s in want:
        x = got.get(s) or {}
        bid, ask = _num(x.get("bid")), _num(x.get("ask"))
        quotes[s] = (bid, ask)
        oi, vol, gam = _num(x.get("open_interest")), _num(x.get("volume")), _num(x.get("gamma"))
        stats[s] = dict(v=int(vol) if vol is not None else None,
                        oi=int(oi) if oi is not None else None, sweep=None)
        if oi is None or gam is None:
            continue
        usable += 1
        e, k, call = exp6[s[-15:-9]], int(s[-8:]) / 1000, s[-9] == "C"
        g = gam * oi * 100 * S * S * 0.01
        a = agg.setdefault((e, k), [0.0, 0.0])
        a[0 if call else 1] += g if call else -g
    rows = [dict(e=e, k=k, cg=v[0], pg=v[1], d=today, t=stamp) for (e, k), v in agg.items()]
    return rows, quotes, stats, (usable / len(want)) if want else 0.0, len(want)


def spot(tk):
    x = _call(_client().market_data.get_snapshot, symbols=tk, category="US_STOCK")
    if not x:
        return None
    return _num(x[0].get("price")) or _num(x[0].get("close"))


def index_forward(quotes, near):
    """Put-call parity forward from {sym: (bid, ask)} of ONE expiry."""
    import index_gex as IX
    chain = {}
    for s, (b, a) in quotes.items():
        if not b or not a or a < b:
            continue
        c = chain.setdefault(int(s[-8:]) / 1000, [None, None])
        c[0 if s[-9] == "C" else 1] = (b + a) / 2
    return IX.parity_forward({k: tuple(v) for k, v in chain.items()}, near=near)
