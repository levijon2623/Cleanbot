"""
check_webull_gex.py
===================
Can the manual layer's GEX heat be computed from WEBULL data instead of
Unusual Whales? Pre-registered comparison against the heat the chart draws.

    python check_webull_gex.py --probe        # what Webull's REST responses carry
    python check_webull_gex.py --selftest     # machinery checks, no network
    python check_webull_gex.py --once         # one paired sample now
    python check_webull_gex.py --loop         # sample every 5 min, 09:35-15:55
    python check_webull_gex.py --report 2026-10-01

🚨 RUN IT ON THE BOX, FROM THE BOT'S FOLDER.
    It uses REST only -- no streaming connection, so it cannot trip the
    one-connection freeze (memory: cleanbot-webull-second-connection). The
    Webull SDK re-uses the access token saved in conf/token.txt and only mints
    a new one when there is none; run from the bot's folder, it therefore
    shares the bot's token instead of replacing it. Snapshot load: ~16 calls
    per ticker per sample (20 contracts a call, 0.3s apart), every 5 minutes.

WHY
    UW is the hurdle for anyone else running the viewer. The manual layer's
    price already comes from Webull; its VWAP/volume could come from Webull
    1m bars and its strike-strip volume/OI from Webull option snapshots. The
    GEX heat is the part that has to be CALCULATED, so it is the part that has
    to be shown to agree with what we draw now before UW can be dropped.

WHAT IS COMPARED (one "sample" = one ticker at one minute)
    UW     the chart's own 0-1DTE heat: live_state._fetch_gex, i.e.
           spot-exposures/expiry-strike, today + the next listed expiry,
           call_gamma_oi + put_gamma_oi per strike, strikes within +-3% of spot.
    LOCAL  the same expiries and band from Webull: every contract's
           open_interest and gamma from get_option_snapshot, and per strike
           OI x gamma x 100 x spot^2 x 1%, calls +, puts -.
    BS     (reported, not scored) the same, with gamma recomputed by
           Black-Scholes from Webull's imp_vol (solved from the mid when
           missing); time to expiry runs to 16:00 ET on the expiry date, in
           years of 365 x 24 x 60 minutes.
    Scale is NOT compared (UW's units are theirs); shape and levels are.

AMENDMENT (2026-09-30, after --probe, before any comparison was run)
    The first draft computed gamma locally from IV. The probe showed Webull's
    option snapshot carries its own greeks (gamma, delta, theta, vega) with
    imp_vol and open_interest, so the PRIMARY profile now uses Webull's gamma
    -- that is what a Webull-only viewer would draw -- and the Black-Scholes
    profile becomes the report-only cross-check. C5 counts OI + gamma.
    Criteria and thresholds are unchanged.

PRE-REGISTERED CRITERIA (written 2026-09-30, before any comparison was run)
    Per ticker, over one full session of samples (>= 20):
      C1  median Pearson r of per-strike net GEX over common strikes  >= 0.80
      C2  median |UW|-weighted sign agreement per strike              >= 0.85
      C3  call wall AND put wall within one strike step of UW's, in   >= 80% of samples
      C4  peak (largest |net|) within one strike step, in             >= 70% of samples
      C5  coverage: median share of in-band contracts with usable OI
          and gamma                                                   >= 0.95
    PASS for a ticker = C1..C5. Overall PASS = SPY, QQQ and IWM all pass.
    Walls here are the 0-1DTE heat's own (largest call / most negative put
    strike), on BOTH sides, so like is compared with like -- not UW's
    all-expiry /gex-levels walls.
    Reported, not scored: median local/UW scale ratio, UW's gamma_flip vs a
    local 0-1DTE flip, the UW rows' age, Webull fetch latency and failures.

MACHINERY CHECKS (--selftest, METHODOLOGY 1)
    M1  Black-Scholes gamma equals the numerical second derivative of the
        Black-Scholes price (so the gamma is the gamma).
    M2  IV solver round-trips: price at a known vol, solve, recover the vol.
    M3  Planted heat: a synthetic chain with a known wall recovers that wall,
        and an identical profile scores r = 1, sign agreement = 1.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import math
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

NY = None
OUT = os.path.join(ROOT, "_webull_gex_cache")
TICKERS = ("SPY", "QQQ", "IWM")
RATE = 0.045                  # same default as local_black_scholes_engine
BATCH, PAUSE = 20, 0.3        # snapshot batching, as webull_gamma_client does
MIN_T = 5.0 / (365 * 24 * 60)  # never price with less than 5 minutes left


# ------------------------------------------------------------------ math
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(S, K, T, sig, r, call):
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if call else (K - S))
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    if call:
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_gamma(S, K, T, sig, r):
    if T <= 0 or sig <= 0 or S <= 0:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    return _npdf(d1) / (S * v)


def solve_iv(px, S, K, T, r, call, lo=0.01, hi=5.0):
    """Bisection on price; None when the price is outside the no-arbitrage
    range (a stale or crossed quote), rather than a made-up volatility."""
    if px is None or px <= 0 or T <= 0:
        return None
    if not (bs_price(S, K, T, lo, r, call) <= px <= bs_price(S, K, T, hi, r, call)):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, r, call) < px:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ scoring
def compare(uw, loc):
    """uw, loc: {strike: (call, put)} with put <= 0. Returns the sample's
    metrics over their common strikes."""
    ks = sorted(set(uw) & set(loc))
    if len(ks) < 5:
        return None
    a = [sum(uw[k]) for k in ks]
    b = [sum(loc[k]) for k in ks]
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    r = (sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)) if va and vb else None
    w = sum(abs(x) for x in a)
    sign = (sum(abs(x) for x, y in zip(a, b) if (x > 0) == (y > 0)) / w) if w else None
    step = statistics.median([k2 - k1 for k1, k2 in zip(ks, ks[1:])]) if len(ks) > 1 else 1.0

    def walls(d):
        return (max(ks, key=lambda k: d[k][0]), min(ks, key=lambda k: d[k][1]),
                max(ks, key=lambda k: abs(sum(d[k]))))
    uc, up, upk = walls(uw)
    lc, lp, lpk = walls(loc)
    near = lambda x, y: abs(x - y) <= step + 1e-9
    ratio = [y / x for x, y in zip(a, b) if abs(x) > 1e-9 and x * y > 0]
    return dict(n=len(ks), r=r, sign=sign, step=step,
                walls_ok=near(uc, lc) and near(up, lp), peak_ok=near(upk, lpk),
                uw_walls=[uc, up, upk], loc_walls=[lc, lp, lpk],
                scale=statistics.median(ratio) if ratio else None)


# ------------------------------------------------------------------ sources
def _clients():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    from webull.core.client import ApiClient
    from webull.data.data_client import DataClient
    import webull_gamma_client as W
    api = ApiClient(os.environ["WEBULL_APP_KEY"], os.environ["WEBULL_APP_SECRET"],
                    region_id="us")
    api.add_endpoint("us", "api.webull.com")
    data = DataClient(api)                       # market data only; no TradeClient
    with contextlib.suppress(Exception):
        W.quiet_webull_logging()
    import unusual_whales_client as U
    uw = U.UnusualWhalesClient(os.environ["UW_API_KEY"])   # REST; no socket
    return data, uw, W


def _json(resp):
    return resp.json() if hasattr(resp, "json") else resp


def _rows(obj):
    if isinstance(obj, dict) and "data" in obj:
        obj = obj["data"]
    if isinstance(obj, dict):
        obj = [obj]
    return obj or []


def _first(d, *names):
    for n in names:
        if isinstance(d, dict) and d.get(n) not in (None, ""):
            return d[n]
    return None


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def probe():
    """Print what each Webull REST response actually carries."""
    data, uw, W = _clients()
    tk = "SPY"
    print("=== equity snapshot (get_snapshot) ===")
    snap = _rows(_json(data.market_data.get_snapshot(symbols=tk, category="US_STOCK")))
    print(json.dumps(snap[:1], indent=1, default=str)[:1500])
    spot = _num(_first(snap[0], "price", "close", "last_price")) if snap else None
    print(f"spot {spot}")

    print("\n=== equity 1m history (get_history_bar, count 1200) ===")
    t0 = time.time()
    bars = _rows(_json(data.market_data.get_history_bar(
        symbol=tk, category="US_STOCK", timespan="M1", count="1200")))
    print(f"{len(bars)} bars in {time.time() - t0:.1f}s; first/last:")
    print(json.dumps([bars[0], bars[-1]] if bars else [], default=str)[:1200])

    print("\n=== option directory -> nearest 0DTE ATM call ===")
    stub = type("S", (), {})()
    stub.data_client, stub._DIR_TTL = data, 900
    syms = W.WebullGammaClient._option_directory(stub, tk)
    today = dt.datetime.now(_ny()).strftime("%y%m%d")
    todays = [s for s in syms if s[-15:-9] == today and s[-9] == "C"]
    occ = min(todays, key=lambda s: abs(int(s[-8:]) / 1000 - (spot or 0))) if todays else None
    print(f"{len(syms)} contracts listed; {len(todays)} 0DTE calls; picked {occ}")
    if not occ:
        return
    print("\n=== option snapshot (get_option_snapshot) ===")
    osnap = _rows(_json(data.option_market_data.get_option_snapshot(
        symbols=occ, category="US_OPTION")))
    print(json.dumps(osnap, indent=1, default=str)[:2500])
    print("\n=== option REST ticks (get_option_tick, count 30) -- any exchange / condition fields? ===")
    ticks = _rows(_json(data.option_market_data.get_option_tick(
        symbol=occ, category="US_OPTION", count="30")))
    print(json.dumps(ticks[:3], indent=1, default=str)[:2000])
    keys = sorted({k for t in ticks if isinstance(t, dict) for k in t})
    print(f"tick fields: {keys}")
    print("\n=== option 1m history (get_option_history_bars) ===")
    ob = _rows(_json(data.option_market_data.get_option_history_bars(
        symbols=occ, category="US_OPTION", timespan="M1", count="5")))
    print(json.dumps(ob[:2], indent=1, default=str)[:1500])


def _ny():
    global NY
    if NY is None:
        from zoneinfo import ZoneInfo
        NY = ZoneInfo("America/New_York")
    return NY


# ------------------------------------------------------------------ one sample
def _years_to(expiry_iso, now):
    y, m, d = map(int, expiry_iso.split("-"))
    close = dt.datetime(y, m, d, 16, 0, tzinfo=_ny())
    return max(MIN_T, (close - now).total_seconds() / 60 / (365 * 24 * 60))


def uw_heat(uw, tk, spot):
    """The chart's own 0-1DTE heat, split into call/put, via live_state."""
    import live_state as LS
    LS._log_levels = lambda *a, **k: None     # 🚨 never write the bot's levels log
    with contextlib.redirect_stdout(io.StringIO()):
        g = LS._fetch_gex(uw, tk, spot)
    near = LS._GEX_NEAR.get(tk) or []
    today = dt.datetime.now(_ny()).date().isoformat()
    nxt = min((r["e"] for r in near if r["e"] > today), default=None)
    want = {today} | ({nxt} if nxt else set())
    lo, hi = spot * (1 - LS.GEX_BAND), spot * (1 + LS.GEX_BAND)
    agg = {}
    for r in near:
        if r["e"] in want and lo <= r["k"] <= hi:
            a = agg.setdefault(r["k"], [0.0, 0.0])
            a[0] += r["cg"]
            a[1] += r["pg"]
    times = sorted(r["t"] for r in near if r["e"] in want and r.get("t"))
    return ({k: tuple(v) for k, v in agg.items() if v[0] or v[1]}, sorted(want),
            (g.get("walls") or {}).get("gamma_flip"), times[-1] if times else None)


def webull_heat(data, W, tk, spot, expiries, now):
    """Webull snapshots for the same expiries and band -> (webull-gamma heat,
    Black-Scholes heat, coverage, contracts, seconds, failures, raw rows)."""
    import live_state as LS
    stub = type("S", (), {})()
    stub.data_client, stub._DIR_TTL = data, 900
    syms = W.WebullGammaClient._option_directory(stub, tk)
    lo, hi = spot * (1 - LS.GEX_BAND), spot * (1 + LS.GEX_BAND)
    exp6 = {e[2:4] + e[5:7] + e[8:10]: e for e in expiries}
    want = [s for s in syms if s[-15:-9] in exp6 and lo <= int(s[-8:]) / 1000 <= hi]
    t0, fails, got = time.time(), 0, {}
    for i in range(0, len(want), BATCH):
        try:
            for x in _rows(_json(data.option_market_data.get_option_snapshot(
                    symbols=",".join(want[i:i + BATCH]), category="US_OPTION"))):
                if isinstance(x, dict) and x.get("symbol"):
                    got[x["symbol"]] = x
        except Exception:                       # noqa: BLE001 -- counted
            fails += 1
            time.sleep(1.0)
        time.sleep(PAUSE)
    heat, bsheat, usable, raw = {}, {}, 0, []
    for s in want:
        x = got.get(s) or {}
        call = s[-9] == "C"
        k = int(s[-8:]) / 1000
        oi, gam, iv = _num(x.get("open_interest")), _num(x.get("gamma")), _num(x.get("imp_vol"))
        bid, ask = _num(x.get("bid")), _num(x.get("ask"))
        raw.append([s, oi, gam, iv, bid, ask])
        if oi is None or gam is None:
            continue
        usable += 1
        sgn = 1.0 if call else -1.0
        unit = oi * 100 * spot * spot * 0.01
        a = heat.setdefault(k, [0.0, 0.0])
        a[0 if call else 1] += sgn * gam * unit
        T = _years_to(exp6[s[-15:-9]], now)
        if not iv and bid and ask:
            iv = solve_iv((bid + ask) / 2, spot, k, T, RATE, call)
        if iv:
            b = bsheat.setdefault(k, [0.0, 0.0])
            b[0 if call else 1] += sgn * bs_gamma(spot, k, T, iv, RATE) * unit
    tidy = lambda h: {k: tuple(v) for k, v in h.items() if v[0] or v[1]}
    return (tidy(heat), tidy(bsheat), (usable / len(want)) if want else 0.0,
            len(want), round(time.time() - t0, 1), fails, raw)


def spot_of(data, tk):
    x = _rows(_json(data.market_data.get_snapshot(symbols=tk, category="US_STOCK")))
    return _num(_first(x[0], "price", "close")) if x else None


def sample(data, uw, W):
    os.makedirs(OUT, exist_ok=True)
    now = dt.datetime.now(_ny())
    path = os.path.join(OUT, f"{now.date().isoformat()}.jsonl")
    for tk in TICKERS:
        try:
            spot = spot_of(data, tk)
            if not spot:
                print(f"  {tk}: no Webull spot -- skipped")
                continue
            uwh, exps, flip, uw_t = uw_heat(uw, tk, spot)
            wh, bsh, cov, n, secs, fails, raw = webull_heat(data, W, tk, spot, exps, now)
        except Exception as e:                  # noqa: BLE001 -- a sample, not the run
            print(f"  {tk}: sample failed: {type(e).__name__}: {e}")
            continue
        rec = dict(at=now.isoformat(timespec="seconds"), ticker=tk, spot=spot,
                   expiries=exps, uw_time=uw_t, uw_flip=flip, coverage=round(cov, 4),
                   contracts=n, fetch_s=secs, failures=fails,
                   uw={str(k): v for k, v in uwh.items()},
                   webull={str(k): v for k, v in wh.items()},
                   bs={str(k): v for k, v in bsh.items()}, raw=raw)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        m = compare(uwh, wh)
        print(f"  {now:%H:%M} {tk} spot {spot:.2f}  {n} contracts in {secs}s, coverage "
              f"{cov:.0%}, fails {fails}  |  " + (
                  f"r {m['r']:.2f}  sign {m['sign']:.2f}  walls {'ok' if m['walls_ok'] else 'DIFF'}"
                  f"  peak {'ok' if m['peak_ok'] else 'DIFF'}  UW {m['uw_walls']} "
                  f"WB {m['loc_walls']}" if m else "too few common strikes"))


def report(day):
    path = os.path.join(OUT, f"{day}.jsonl")
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"{day}: {len(recs)} samples from {path}\n")
    verdict = True
    for tk in TICKERS:
        rs = [r for r in recs if r["ticker"] == tk]
        f = lambda d: {float(k): tuple(v) for k, v in d.items()}
        prim = [m for m in (compare(f(r["uw"]), f(r["webull"])) for r in rs) if m]
        bs = [m for m in (compare(f(r["uw"]), f(r["bs"])) for r in rs) if m]
        if not prim:
            print(f"=== {tk}: no comparable samples\n")
            verdict = False
            continue
        med = lambda xs: statistics.median(xs) if xs else float("nan")
        c1 = med([m["r"] for m in prim if m["r"] is not None])
        c2 = med([m["sign"] for m in prim if m["sign"] is not None])
        c3 = sum(m["walls_ok"] for m in prim) / len(prim)
        c4 = sum(m["peak_ok"] for m in prim) / len(prim)
        c5 = med([r["coverage"] for r in rs])
        ok = [len(prim) >= 20, c1 >= 0.80, c2 >= 0.85, c3 >= 0.80, c4 >= 0.70, c5 >= 0.95]
        verdict &= all(ok)
        tag = lambda b: "PASS" if b else "FAIL"
        print(f"=== {tk}   {len(prim)} samples ({tag(ok[0])} n>=20)")
        print(f"  C1 median r           {c1:6.3f}  (>= 0.80)  {tag(ok[1])}")
        print(f"  C2 median sign agree  {c2:6.3f}  (>= 0.85)  {tag(ok[2])}")
        print(f"  C3 walls within 1 step {c3:5.0%}   (>= 80%)  {tag(ok[3])}")
        print(f"  C4 peak within 1 step  {c4:5.0%}   (>= 70%)  {tag(ok[4])}")
        print(f"  C5 median coverage     {c5:5.0%}   (>= 95%)  {tag(ok[5])}")
        print(f"  report: scale local/UW {med([m['scale'] for m in prim if m['scale']]):.3g}"
              f" | BS cross-check r {med([m['r'] for m in bs if m['r'] is not None]):.3f}"
              f" sign {med([m['sign'] for m in bs if m['sign'] is not None]):.3f}"
              f" | fetch {med([r['fetch_s'] for r in rs]):.1f}s, failures "
              f"{sum(r['failures'] for r in rs)}\n")
    print(f"OVERALL: {'PASS' if verdict else 'FAIL'}")


def selftest():
    fails = []

    def ok(label, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        cond or fails.append(label)
    S, K, T, sig, r = 500.0, 505.0, 2 / 365, 0.22, RATE
    h = 0.01
    num = (bs_price(S + h, K, T, sig, r, True) - 2 * bs_price(S, K, T, sig, r, True)
           + bs_price(S - h, K, T, sig, r, True)) / (h * h)
    ok(f"M1 gamma {bs_gamma(S, K, T, sig, r):.6f} == numeric {num:.6f}",
       abs(bs_gamma(S, K, T, sig, r) - num) < 1e-4)
    for call in (True, False):
        px = bs_price(S, K, T, sig, r, call)
        ok(f"M2 IV round-trip ({'call' if call else 'put'}) {solve_iv(px, S, K, T, r, call):.5f}",
           abs(solve_iv(px, S, K, T, r, call) - sig) < 1e-5)
    ok("M2 impossible price -> None, not a made-up vol",
       solve_iv(S * 2, S, K, T, r, True) is None)
    ks = [495 + i for i in range(11)]
    prof = {k: (1000.0 * math.exp(-((k - 502) / 2) ** 2), -800.0 * math.exp(-((k - 497) / 2) ** 2))
            for k in ks}
    m = compare(prof, prof)
    ok(f"M3 identical profile: r {m['r']:.3f}, sign {m['sign']:.3f}",
       abs(m["r"] - 1) < 1e-9 and abs(m["sign"] - 1) < 1e-9)
    ok(f"M3 planted walls recovered {m['uw_walls'][:2]} == [502, 497]",
       m["uw_walls"][:2] == [502, 497])
    shifted = {k: prof.get(k - 3, (0.0, 0.0)) for k in ks}
    m2 = compare(prof, shifted)
    ok(f"M3 walls shifted 3 strikes -> flagged (walls_ok {m2['walls_ok']})", not m2["walls_ok"])
    print(f"\n  {'all machinery checks pass' if not fails else f'{len(fails)} FAILED'}")
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probe", action="store_true")
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", action="store_true")
    g.add_argument("--report", metavar="YYYY-MM-DD")
    a = ap.parse_args()
    if a.probe:
        return probe()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.report:
        return report(a.report)
    if not selftest():
        sys.exit("machinery checks failed -- not sampling")
    data, uw, W = _clients()
    if a.once:
        return sample(data, uw, W)
    import market_calendar as MC
    while True:
        now = dt.datetime.now(_ny())
        mod = now.hour * 60 + now.minute
        if not MC.is_trading_day(now.date()) or mod > 15 * 60 + 55:
            break
        if mod >= 9 * 60 + 35:
            sample(data, uw, W)
        time.sleep(300 - (time.time() % 300) + 5)   # on the 5-minute grid
    report(dt.datetime.now(_ny()).date().isoformat())


if __name__ == "__main__":
    main()
