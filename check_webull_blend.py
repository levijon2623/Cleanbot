"""
check_webull_blend.py
=====================
Can the viewer's "+index" GEX heat (SPY+SPX, QQQ+NDX, IWM+RUT) be built from
WEBULL data alone? Pre-registered comparison against the UW blend the chart
draws -- the "budget" (UW-free) version of the toggle.

    python check_webull_blend.py --selftest     # machinery checks, no network
    python check_webull_blend.py --once         # one paired sample now
    python check_webull_blend.py --loop         # every 10 min at :x2:30, 09:42-15:55
    python check_webull_blend.py --report 2026-10-01

🚨 RUN IT ON THE BOX, FROM THE BOT'S FOLDER -- same rules as check_webull_gex.py
    (REST only, shares conf/token.txt, never a second streaming connection).
    It runs ALONGSIDE check_webull_gex.py's 5-minute loop, so it samples on a
    10-minute grid offset by 2.5 minutes to keep the two REST bursts apart.
    Load: ~110 snapshot calls per sample (3 ETF chains + SPX/NDX/RUT chains,
    20 contracts a call, 0.3s apart) -- about one minute every ten.

WHAT IS COMPARED (one sample = one ETF/index pair at one minute)
    UW blend  live_state's own: the ETF's 0-1DTE expiry-strike rows plus the
              index's, mapped by UW's session-median index/ETF ratio and split
              onto the ETF's $1 grid (index_gex.py), strikes within +-3%.
    WB blend  the same from Webull only: the ETF chain as check_webull_gex
              builds it, plus the index chain's OI x gamma x 100 x F^2 x 1%
              (calls +, puts -), mapped by a WEBULL ratio -- the index forward F
              from put-call parity on its own 0DTE mids (index_gex.parity_forward;
              Webull has no index quote: probed 2026-10-01, SPX/NDX/RUT are
              INVALID_SYMBOL in US_STOCK and US_INDEX) over the ETF's Webull spot.
              Same index_gex functions, same expiries, same band.

PRE-REGISTERED CRITERIA (written 2026-10-01 ~04:00 ET, before any blended
comparison was run; thresholds copied from check_webull_gex.py C1-C5)
    Per pair, over one session of samples (>= 20):
      B1  median Pearson r of per-strike blended net GEX            >= 0.80
      B2  median |UW|-weighted sign agreement per strike             >= 0.85
      B3  blended call wall AND put wall within one strike step,  in >= 80% of samples
      B4  blended peak (largest |net|) within one strike step,    in >= 70% of samples
      B5  median coverage of in-band INDEX contracts with usable
          OI and gamma                                               >= 0.95
      B6  median |Webull ratio / UW ratio - 1|                       <= 0.0015
          (0.15%: ~$1.1 of SPY -- one strike step -- since the ratio IS the map)
    PASS for a pair = B1..B6. Overall PASS = all three pairs pass.
    Reported, not scored: index share of the band's gamma (WB vs UW), the
    index-only profile's r, the native labels of the top index strikes,
    fetch time and failures.

MACHINERY CHECKS (--selftest)
    check_webull_gex's M1-M3, plus
    M4  the grid split conserves total gamma and recovers a planted index wall
        at the ETF strike its ratio maps it to;
    M5  put-call parity recovers a planted forward from Black-Scholes prices.
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

import check_webull_gex as C      # noqa: E402 -- shares math, clients, scoring
import index_gex as IX            # noqa: E402

OUT = C.OUT
PAIRS = (("SPY", "SPX"), ("QQQ", "NDX"), ("IWM", "RUT"))
BAND_PAD = 0.005                  # fetch the index band a little wide; the seed
                                  # ratio is only good to a fraction of a percent


def uw_blend(uw, etf, spot):
    """The chart's own blend via live_state, split into (call, put) per strike.
    Returns (heat, etf_only_heat, ratio, expiries, index rows) or None."""
    import live_state as LS
    LS._log_levels = lambda *a, **k: None      # 🚨 never write the bot's levels log
    # 🚨 and never write the bot's live/index_ratio.json from a check
    LS.IDX_RATIO_FILE = os.path.join(OUT, "index_ratio-check.json")
    with contextlib.redirect_stdout(io.StringIO()):
        g = LS._fetch_gex(uw, etf, spot)
        LS._fetch_index_blend(uw, etf, spot, g)
    b = g.get("blend")
    if not b:
        return None
    want = set(b["expiries"])
    rows, irows = LS._GEX_NEAR.get(etf) or [], LS._GEX_IDX_NEAR.get(etf) or []
    lo, hi = spot * (1 - LS.GEX_BAND), spot * (1 + LS.GEX_BAND)
    blended, _ = IX.blend_rows([r for r in rows if r["e"] in want],
                               [r for r in irows if r["e"] in want], b["ratio"])
    tup = lambda h: {k: tuple(v) for k, v in h.items()}
    return (tup(IX.heat_of(blended, want, lo, hi)), tup(IX.heat_of(rows, want, lo, hi)),
            b["ratio"], sorted(want), [r for r in irows if r["e"] in want], b.get("share"))


def webull_index(data, W, etf, idx, spot, expiries):
    """Webull snapshots of the index chain for the expiries, near the money.
    Returns (rows [{e,k,cg,pg}], forward, ratio, coverage, n, secs, fails)."""
    import live_state as LS
    stub = type("S", (), {})()
    stub.data_client, stub._DIR_TTL = data, 900
    syms = W.WebullGammaClient._option_directory(stub, idx)
    seed = spot * IX.SEED_RATIO[etf]
    band = LS.GEX_BAND + BAND_PAD
    lo, hi = seed * (1 - band), seed * (1 + band)
    exp6 = {e[2:4] + e[5:7] + e[8:10]: e for e in expiries}
    want = [s for s in syms if s[-15:-9] in exp6 and lo <= int(s[-8:]) / 1000 <= hi]
    t0, fails, got = time.time(), 0, {}
    for i in range(0, len(want), C.BATCH):
        try:
            for x in C._rows(C._json(data.option_market_data.get_option_snapshot(
                    symbols=",".join(want[i:i + C.BATCH]), category="US_OPTION"))):
                if isinstance(x, dict) and x.get("symbol"):
                    got[x["symbol"]] = x
        except Exception:                       # noqa: BLE001 -- counted
            fails += 1
            time.sleep(1.0)
        time.sleep(C.PAUSE)
    # forward from parity on the FIRST expiry (0DTE): {strike: (call mid, put mid)}
    first = min(exp6)
    chain = {}
    for s in want:
        if s[-15:-9] != first:
            continue
        x = got.get(s) or {}
        bid, ask = C._num(x.get("bid")), C._num(x.get("ask"))
        if not bid or not ask or ask < bid:
            continue
        k = int(s[-8:]) / 1000
        c = chain.setdefault(k, [None, None])
        c[0 if s[-9] == "C" else 1] = (bid + ask) / 2
    fwd = IX.parity_forward({k: tuple(v) for k, v in chain.items()}, near=seed)
    rows, usable = [], 0
    for s in want:
        x = got.get(s) or {}
        oi, gam = C._num(x.get("open_interest")), C._num(x.get("gamma"))
        if oi is None or gam is None:
            continue
        usable += 1
        if not fwd:
            continue
        v = gam * oi * 100 * fwd * fwd * 0.01
        call = s[-9] == "C"
        rows.append(dict(e=exp6[s[-15:-9]], k=int(s[-8:]) / 1000,
                         cg=v if call else 0.0, pg=0.0 if call else -v))
    return (rows, fwd, (fwd / spot) if fwd else None,
            (usable / len(want)) if want else 0.0, len(want),
            round(time.time() - t0, 1), fails)


def sample(data, uw, W):
    os.makedirs(OUT, exist_ok=True)
    now = dt.datetime.now(C._ny())
    path = os.path.join(OUT, f"blend-{now.date().isoformat()}.jsonl")
    for etf, idx in PAIRS:
        try:
            spot = C.spot_of(data, etf)
            if not spot:
                print(f"  {etf}: no Webull spot -- skipped")
                continue
            u = uw_blend(uw, etf, spot)
            if not u:
                print(f"  {etf}: no UW blend (index rows or ratio missing) -- skipped")
                continue
            uwb, uwe, uw_ratio, exps, uw_irows, uw_share = u
            we, _bs, ecov, en, esecs, efails, _raw = C.webull_heat(data, W, etf, spot, exps, now)
            irows, fwd, wb_ratio, icov, inn, isecs, ifails = webull_index(
                data, W, etf, idx, spot, exps)
        except Exception as e:                  # noqa: BLE001 -- a sample, not the run
            print(f"  {etf}: sample failed: {type(e).__name__}: {e}")
            continue
        import live_state as LS
        lo, hi = spot * (1 - LS.GEX_BAND), spot * (1 + LS.GEX_BAND)
        wbb = {k: [v[0], v[1]] for k, v in we.items()}
        wbi = {}
        if wb_ratio:
            for k, (c, p, _n) in IX.split_to_grid(IX.map_rows(irows, wb_ratio)).items():
                if lo <= k <= hi:
                    a = wbb.setdefault(k, [0.0, 0.0])
                    a[0] += c
                    a[1] += p
                    wbi[k] = [c, p]
        wbb = {k: tuple(v) for k, v in wbb.items() if v[0] or v[1]}
        top = lambda rows, r: IX.index_levels(idx, rows, set(exps), r, lo, hi) if r else []
        rec = dict(at=now.isoformat(timespec="seconds"), etf=etf, idx=idx, spot=spot,
                   expiries=exps, fwd=fwd, wb_ratio=wb_ratio, uw_ratio=uw_ratio,
                   idx_coverage=round(icov, 4), idx_contracts=inn, etf_coverage=round(ecov, 4),
                   fetch_s=round(esecs + isecs, 1), failures=efails + ifails,
                   uw_share=uw_share, wb_share=IX.share({k: list(v) for k, v in we.items()}, wbi),
                   uw={str(k): v for k, v in uwb.items()},
                   webull={str(k): v for k, v in wbb.items()},
                   uw_levels=[l["label"] for l in top(uw_irows, uw_ratio)],
                   wb_levels=[l["label"] for l in top(irows, wb_ratio)])
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        m = C.compare(uwb, wbb) if wb_ratio else None
        rr = (wb_ratio / uw_ratio - 1) if wb_ratio and uw_ratio else None
        print(f"  {now:%H:%M} {etf}+{idx} spot {spot:.2f} F {fwd or float('nan'):.2f} "
              f"ratio WB {wb_ratio or float('nan'):.5f} UW {uw_ratio:.5f} "
              f"({'—' if rr is None else f'{rr * 100:+.3f}%'})  idx {inn} contracts, cov "
              f"{icov:.0%}, {esecs + isecs:.0f}s, fails {efails + ifails}  |  " + (
                  f"r {m['r']:.2f} sign {m['sign']:.2f} walls {'ok' if m['walls_ok'] else 'DIFF'}"
                  f" peak {'ok' if m['peak_ok'] else 'DIFF'} UW {m['uw_walls']} WB {m['loc_walls']}"
                  if m else "no comparison"))


def report(day):
    path = os.path.join(OUT, f"blend-{day}.jsonl")
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"{day}: {len(recs)} blended samples from {path}\n")
    verdict = True
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    tag = lambda b: "PASS" if b else "FAIL"
    for etf, idx in PAIRS:
        rs = [r for r in recs if r["etf"] == etf]
        f = lambda d: {float(k): tuple(v) for k, v in d.items()}
        ms = [m for m in (C.compare(f(r["uw"]), f(r["webull"])) for r in rs if r.get("wb_ratio")) if m]
        if not ms:
            print(f"=== {etf}+{idx}: no comparable samples\n")
            verdict = False
            continue
        b1 = med([m["r"] for m in ms if m["r"] is not None])
        b2 = med([m["sign"] for m in ms if m["sign"] is not None])
        b3 = sum(m["walls_ok"] for m in ms) / len(ms)
        b4 = sum(m["peak_ok"] for m in ms) / len(ms)
        b5 = med([r["idx_coverage"] for r in rs])
        b6 = med([abs(r["wb_ratio"] / r["uw_ratio"] - 1) for r in rs if r.get("wb_ratio") and r.get("uw_ratio")])
        ok = [len(ms) >= 20, b1 >= 0.80, b2 >= 0.85, b3 >= 0.80, b4 >= 0.70, b5 >= 0.95, b6 <= 0.0015]
        verdict &= all(ok)
        lab = sum(r["uw_levels"] == r["wb_levels"] for r in rs) / len(rs)
        print(f"=== {etf}+{idx}   {len(ms)} samples ({tag(ok[0])} n>=20)")
        print(f"  B1 median r            {b1:6.3f}  (>= 0.80)  {tag(ok[1])}")
        print(f"  B2 median sign agree   {b2:6.3f}  (>= 0.85)  {tag(ok[2])}")
        print(f"  B3 walls within 1 step {b3:6.0%}  (>= 80%)   {tag(ok[3])}")
        print(f"  B4 peak within 1 step  {b4:6.0%}  (>= 70%)   {tag(ok[4])}")
        print(f"  B5 index coverage      {b5:6.0%}  (>= 95%)   {tag(ok[5])}")
        print(f"  B6 ratio |WB/UW - 1|   {b6 * 100:5.3f}%  (<= 0.15%) {tag(ok[6])}")
        print(f"  report: index share WB {med([r['wb_share'] for r in rs if r.get('wb_share') is not None]):.2f}"
              f" vs UW {med([r['uw_share'] for r in rs if r.get('uw_share') is not None]):.2f}"
              f" | top-3 native levels identical in {lab:.0%} | fetch "
              f"{med([r['fetch_s'] for r in rs]):.0f}s, failures {sum(r['failures'] for r in rs)}\n")
    print(f"OVERALL: {'PASS' if verdict else 'FAIL'}")


def selftest():
    ok_all = C.selftest()
    fails = []

    def ok(label, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        cond or fails.append(label)
    rows = [dict(e="E", k=7650.0, cg=900.0, pg=-100.0), dict(e="E", k=7700.0, cg=50.0, pg=-20.0)]
    g = IX.split_to_grid(IX.map_rows(rows, 10.0372))
    ok("M4 split conserves gamma",
       abs(sum(v[0] + v[1] for v in g.values()) - (900 - 100 + 50 - 20)) < 1e-9)
    ok(f"M4 planted SPX 7650 wall lands at SPY 762 (7650/10.0372 = {7650 / 10.0372:.2f})",
       max(g, key=lambda k: g[k][0]) == 762.0)
    F, T = 7655.3, 3 / 365 / 24
    chain = {k: (C.bs_price(F, k, T, 0.19, 0, True), C.bs_price(F, k, T, 0.19, 0, False))
             for k in range(7600, 7720, 5)}
    pf = IX.parity_forward(chain, near=7650)
    ok(f"M5 parity forward {pf:.3f} == planted {F}", pf is not None and abs(pf - F) < 1e-6)
    print(f"\n  {'blend machinery checks pass' if not fails else f'{len(fails)} FAILED'}")
    return ok_all and not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", action="store_true")
    g.add_argument("--report", metavar="YYYY-MM-DD")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.report:
        return report(a.report)
    if not selftest():
        sys.exit("machinery checks failed -- not sampling")
    data, uw, W = C._clients()
    if a.once:
        return sample(data, uw, W)
    import market_calendar as MC
    while True:
        now = dt.datetime.now(C._ny())
        mod = now.hour * 60 + now.minute
        if not MC.is_trading_day(now.date()) or mod > 15 * 60 + 55:
            break
        if mod >= 9 * 60 + 42:
            sample(data, uw, W)
        # next :x2:30 -- 2.5 minutes off check_webull_gex's 5-minute grid
        t = time.time()
        time.sleep(600 - ((t - 150) % 600))
    report(dt.datetime.now(C._ny()).date().isoformat())


if __name__ == "__main__":
    main()
