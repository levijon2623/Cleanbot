# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_tv_proxy.py
=================
CAN A TRADINGVIEW-COMPUTABLE SERIES REPRODUCE THE FLOW TRIGGER?

WHY THIS DECIDES THE TRADINGVIEW QUESTION
    TradingView cannot ingest our data. Pine Seeds -- the official
    bring-your-own-data channel -- is EOD only (1D+ timeframes, 5 updates a day)
    and new repositories are suspended; Pine Script has no HTTP. So a real
    TradingView indicator can only be built from what Pine can compute itself:
    the underlying's OHLCV.

    The deployed trigger is an EMA(5) crossover of CUMULATIVE NET OPTION PREMIUM
    (ask-side minus bid-side dollars, signed by call/put). The question is
    whether the same STRUCTURE on a price/volume series picks the same minutes.
    If it does, a Pine indicator is worth writing. If it does not, anything
    published under the bot's name would be a different signal wearing it.

WHAT THE BOOK ALREADY PREDICTS
    METHODOLOGY 7: "the edge is INFORMATIONAL, not MECHANICAL -- premium
    weighting (dollar conviction) beats delta weighting (mechanical hedging
    need), and the hedging/participation ratio failed EVEN WITH LOOKAHEAD."
    check_flow_threshold sharpened it: flow MAGNITUDE is load-bearing, not just
    the crossover -- dropping the percentile gate to p20 costs -9,899 ROE and
    the marginal band runs -10.6/trade against the deployed band's +12.6. A
    volume proxy carries no dollar conviction, so the prediction is failure.
    Measured anyway, because a prediction is not a result.

🚨 THE TRAP: OVERLAP IS HIGH BY CHANCE
    SPY alone throws ~58,000 crossovers across the sample -- several per session.
    Two dense event series will land within a few minutes of each other
    constantly for no reason at all. So every overlap number is reported beside
    a PLACEBO: the same proxy triggers with their minutes SHUFFLED WITHIN THE
    DAY, preserving the per-day count. Recall above the placebo is signal;
    recall near it is arithmetic.

🚨 THE OTHER TRAP: TIMEZONE
    historical/{T}.parquet stores minute_et tz-AWARE America/New_York, while the
    flow frame, the trigger list and the option bars are all tz-NAIVE ET. Taking
    `.values` on the aware column yields UTC-based datetime64 and shifts every
    proxy trigger 4-5 hours into the future -- build_candidates would then match
    the wrong minute's contract and report a plausible, meaningless number.
    `_bars` localizes to None explicitly. Do not remove it.

THE PROXIES -- each is three lines of Pine
    svol   cum(sign(close - open) * volume)          candle-direction volume
    obv    cum(sign(close - close[1]) * volume)      classic On Balance Volume
    dvol   cum((close - open) * volume)              dollar-weighted delta
    rand   cum(random_sign * volume)                 THE PLACEBO

⚠️ PREDATES THE 2026-09-20 SPOT FIX (sim_core.py, the `bbd` sort), so the
absolute P&L figures will move on re-baseline. The COMPARISON is unaffected:
the real trigger and all four proxies ran through the same build_candidates and
inherited the same stale spot, so the arms shift together. Section 1 (timing
fidelity) never touches build_candidates at all and stands as measured.

RESULT -- 2026-09-20, 9 tickers, ~470,000 real crossovers. NO TRADINGVIEW-NATIVE
INDICATOR IS POSSIBLE. The proxies reproduce the TIMING and none of the EDGE.

    Timing, k=0 (exact minute, same direction), every ticker consistent:
        svol / obv   19-28% recall against 12-15% chance   (~1.8x)
        dvol         16-24% against 10-11%
        rand         15-18% against 13-15%                 (~1.1x, the null)
    So the agreement is REAL -- option flow and underlying volume answer to the
    same order flow. At +-5 minutes everything converges to ~85% against ~78%
    chance, which is density, not agreement; that is what the shuffle control
    is for.

    P&L through the identical gates, exits and sequential guard:
        real     +3,601 / 336 trades
        svol       -561 / 355
        obv      -2,206 / 407
        dvol     -2,143 / 385
        rand     -4,217 / 417        <- the placebo

    svol clearly beats the placebo, so it is carrying information -- and it is
    still a losing book. Landing on ~1 in 4 of the right minutes while inventing
    355 trades destroys the entire +3,601 edge. A crossover of cumulative VOLUME
    knows WHEN the tape got busy; it does not know how many DOLLARS were behind
    it or which way they leaned, and check_flow_threshold showed the magnitude
    is the load-bearing part (p20 costs -9,899).

    CONSEQUENCE: the flow series must be SHIPPED to a chart, not recomputed in
    one. Hence export_flow_tape.py + flow_viewer.html rather than a Pine script.

Usage:
  python check_tv_proxy.py --paper
  python check_tv_proxy.py --paper --tickers SPY QQQ
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd
import polars as pl

import sim_core

RTH0, RTH1 = 570, 955
SEED = 20260920


def _bars(tk):
    df = (pl.scan_parquet(f"historical/{tk}.parquet")
          .select("date", "minute_et", "open", "high", "low", "close", "volume")
          .collect().to_pandas())
    # tz-AWARE -> naive ET. See the timezone note in the module docstring.
    df["minute_et"] = pd.to_datetime(df["minute_et"]).dt.tz_localize(None)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    m = df["minute_et"].dt.hour * 60 + df["minute_et"].dt.minute
    df["mod"] = m
    return df[(m >= RTH0) & (m <= RTH1)].sort_values(["date", "mod"])


def step_series(g, kind, rng):
    o, c, v = g["open"].to_numpy(), g["close"].to_numpy(), g["volume"].to_numpy()
    if kind == "svol":
        return np.sign(c - o) * v
    if kind == "obv":
        return np.sign(np.diff(c, prepend=c[0])) * v
    if kind == "dvol":
        return (c - o) * v
    if kind == "rand":
        return rng.choice([-1.0, 1.0], size=len(v)) * v
    raise ValueError(kind)


def proxy_triggers(bars, kind, seed_tag):
    """Same EMA(5) crossover as directional_flow_backtester.triggers_for, on a
    series Pine can compute. Mirrors that function's structure deliberately.

    Seeded per (ticker, kind) rather than handed a live rng, so section 2
    regenerates byte-identical triggers to section 1 without holding ~2M
    trigger dicts (9 tickers x 4 proxies x ~58k) in memory at once.
    """
    # crc32, NOT hash(): str hashing is salted per process (PYTHONHASHSEED), so
    # hash() would give a different placebo on every run and section 2 would
    # silently disagree with section 1.
    rng = np.random.default_rng(zlib.crc32(seed_tag.encode()) ^ SEED)
    out = []
    for d, g in bars.groupby("date"):
        if len(g) < 6:
            continue
        cum = np.cumsum(step_series(g, kind, rng))
        ema = pd.Series(cum).ewm(span=5, adjust=False).mean().to_numpy()
        mt = g["minute_et"].to_numpy()
        hrs = g["minute_et"].dt.hour.to_numpy()
        for i in range(1, len(cum)):
            bull = cum[i - 1] <= ema[i - 1] and cum[i] > ema[i]
            bear = cum[i - 1] >= ema[i - 1] and cum[i] < ema[i]
            if bull or bear:
                out.append({"date": d, "ts": mt[i], "hour": int(hrs[i]),
                            "dir": "CALL" if bull else "PUT",
                            "abs_flow": abs(float(cum[i]))})
    return out


def by_day(trigs):
    out = {}
    for t in trigs:
        out.setdefault(t["date"], []).append(
            (pd.Timestamp(t["ts"]).hour * 60 + pd.Timestamp(t["ts"]).minute,
             t["dir"]))
    for d in out:
        out[d].sort()
    return out


def recall(real, prox, k, directional=True):
    """Fraction of REAL triggers with a proxy trigger within +-k minutes."""
    hit = tot = 0
    for d, rs in real.items():
        ps = prox.get(d, [])
        if not ps:
            tot += len(rs)
            continue
        pm = np.array([m for m, _ in ps])
        pd_ = [dd for _, dd in ps]
        for m, dd in rs:
            tot += 1
            idx = np.where(np.abs(pm - m) <= k)[0]
            if len(idx) and (not directional or any(pd_[i] == dd for i in idx)):
                hit += 1
    return hit / max(tot, 1) * 100, tot


def shuffled(prox, rng):
    """Same count per day, random minutes. The only honest baseline for a
    'were they close together' statistic on two dense event series."""
    out = {}
    for d, ps in prox.items():
        ms = rng.integers(RTH0, RTH1 + 1, size=len(ps))
        out[d] = sorted((int(m), dd) for m, (_om, dd) in zip(ms, ps))
    return out


KINDS = ["svol", "obv", "dvol", "rand"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--tickers", nargs="*")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = sim_core.research_rules(include_paper=a.paper)
    if a.tickers:
        rules = [r for r in rules if r["ticker"] in set(a.tickers)]
    tks = sorted({r["ticker"] for r in rules})

    print(f"\n{'='*104}")
    print(f"  1. TIMING FIDELITY -- does a proxy crossover land on the real "
          f"trigger minute?")
    print(f"     recall = % of REAL triggers with a same-direction proxy "
          f"trigger within +-k minutes")
    print(f"     'chance' = the same proxy triggers with minutes SHUFFLED "
          f"within the day")
    print(f"{'='*104}")
    print(f"  {'ticker':8} {'proxy':6} {'n real':>8} {'n proxy':>8} " +
          "".join(f"{'k='+str(k):>9}{'chance':>9}" for k in (0, 2, 5)))
    keep = {}
    for tk in tks:
        rng = np.random.default_rng(SEED)
        f = _flow_for(D, [tk])
        if f.empty:
            continue
        real = by_day(D.triggers_for(f, tk))
        bars = _bars(tk)
        keep[tk] = dict(bars=bars, real_n=sum(len(v) for v in real.values()))
        for kind in KINDS:
            pt = proxy_triggers(bars, kind, f"{tk}:{kind}")
            pdd = by_day(pt)
            sh = shuffled(pdd, rng)
            cells = ""
            for k in (0, 2, 5):
                r_, _n = recall(real, pdd, k)
                c_, _ = recall(real, sh, k)
                cells += f"{r_:>8.1f}%{c_:>8.1f}%"
            print(f"  {tk:8} {kind:6} {keep[tk]['real_n']:>8,} "
                  f"{len(pt):>8,} {cells}")

    print(f"\n  A proxy is only informative where recall EXCEEDS chance. Two")
    print(f"  dense event series land near each other constantly; that is not")
    print(f"  agreement, it is density.")

    print(f"\n{'='*104}")
    print(f"  2. THE VERDICT -- P&L through the SAME gates, same exits, same "
          f"sequential guard")
    print(f"     Only the trigger series changes. build_candidates(trigs=...) "
          f"exists for exactly this.")
    print(f"{'='*104}")
    print(f"  {'rule':24} {'real':>14} " +
          "".join(f"{k:>14}" for k in KINDS))
    tot = {k: 0.0 for k in ["real"] + KINDS}
    cnt = {k: 0 for k in ["real"] + KINDS}
    for rule in rules:
        tk = rule["ticker"]
        if tk not in keep:
            continue
        pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        cells = []
        for src in ["real"] + KINDS:
            if src == "real":
                trigs = None
            else:
                trigs = proxy_triggers(keep[tk]["bars"], src, f"{tk}:{src}")
                D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
            try:
                cand = sim_core.build_candidates(D, rule, trigs=trigs)
            except Exception as e:
                cells.append(f"{'err':>14}"); continue
            r = sim_core.walk(cand, pol, eod, fill=a.fill, cush_cap=cap)
            s = sum(p for _d, p in r) * 100
            tot[src] += s; cnt[src] += len(r)
            cells.append(f"{s:>+9.0f}/{len(r):<4d}")
        print(f"  {rule['name']:24} " + "".join(cells), flush=True)
    print(f"  {'-'*100}")
    print(f"  {'BOOK TOTAL':24} " +
          "".join(f"{tot[s]:>+9.0f}/{cnt[s]:<4d}" for s in ["real"] + KINDS))

    print(f"\n  `rand` is the placebo -- cumulative volume with random signs.")
    print(f"  Any proxy that does not clearly beat it is reproducing the")
    print(f"  crossover's ARITHMETIC, not the flow's information.")


if __name__ == "__main__":
    main()
