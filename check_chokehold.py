# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_chokehold.py
==================
TIGHTEN THE TRAIL WHEN THE DETECTOR FIRES -- instead of flattening.

THE DESIGN, AND WHY IT IS NOT THE SEVENTH FAILED EXIT
    check_volume_exit closed the position on a signal and re-entered on the next
    trigger: -2029pp OOS. Six other families that CUT the day's exposure failed
    too. The only intervention that ever helped (+5.2pp, leg-in vertical) was the
    one that did not reduce the position.
    This does not reduce it either. Full size keeps running; only the stop
    distance changes, 0.50 -> 0.25 or 0.15, latched once armed. If price keeps
    making new highs the tight trail follows it up; if it rolls over, the tight
    trail is already underneath.

THE CAUSAL FEATURES -- strictly backward-looking, no look-ahead
    vol_spike    1m volume above the 75th pct of the PRIOR 30 bars
    range_exp    1m bar range above the 75th pct of the PRIOR 30 bars
    rsi_fade     RSI(14) 3-bar change in the most-diverging quartile of the
                 prior 30 bars (price makes a new extreme, RSI does not confirm)
    Every window is shifted by one bar, so the current bar never enters its own
    percentile. check_peak_detector's 38.5% precision used the whole move
    INCLUDING future bars for its normaliser -- a ceiling, not a detector, and
    the gap between the two is the look-ahead draining out.

🚨 DIRECTION ON rsi_fade IS EASY TO INVERT AND WOULD SILENTLY FLIP THE TEST.
   Fading means momentum-in-the-trade's-direction is weak. For a CALL that is a
   LOW 3-bar RSI change; for a PUT the move is downward, so divergence is a HIGH
   (less negative) change. Handled per direction, not by one global sign.

=====================  THE CONTROLS THAT DECIDE IT  =====================
  ALWAYS-TIGHT is the control that matters most. If an unconditional 0.25 or
  0.15 trail does as well, the detector is irrelevant and we have merely
  rediscovered that a tighter trail suits this book. The detector has to beat
  the baseline AND the unconditional arm.
  PLACEBO fires at the same per-trade RATE at random minutes. If random
  tightening does as well, the timing carries nothing.
  Any arm that only beats the baseline has proved nothing.

Usage:
  python check_chokehold.py
  python check_chokehold.py --tight 0.25 0.15 0.10 --paper
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60


def _rsi(c, n=14):
    out = np.full(c.size, np.nan)
    if c.size < n + 1:
        return out
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, d.size):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        out[i + 1] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def _macd_5m(mods, close, fast=5, slow=13, nchg=3):
    """MACD(fast, slow, signal=1) on 5-MINUTE bars, mapped back to 1m minutes.

    🚨 SIGNAL PERIOD 1. An EMA of period 1 IS the series, so the signal line
    equals the MACD line and the HISTOGRAM IS IDENTICALLY ZERO. A histogram
    feature here would never fire and would read as a clean null rather than as
    a degenerate definition. So the usable quantities are the LINE and its
    change; the change is the direct analogue of rsi_d3 (momentum of momentum,
    i.e. divergence).

    CAUSALITY: a 5m bar spanning [m-4, m] is only complete AT m, so its value is
    forward-filled to minutes >= m and never applied earlier. Assigning it to
    the whole bucket would leak up to four minutes of the future into every
    decision.
    """
    n = len(mods)
    if n < slow * 5 + nchg * 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    # 5m buckets keyed by CLOSING minute
    idx = (np.arange(n) // 5)
    closes, close_min = [], []
    for b in range(idx.max() + 1):
        sel = np.where(idx == b)[0]
        if not sel.size:
            continue
        closes.append(float(close[sel[-1]]))
        close_min.append(int(mods[sel[-1]]))
    s = pd.Series(closes, dtype=float)
    line = (s.ewm(span=fast, adjust=False).mean()
            - s.ewm(span=slow, adjust=False).mean())
    # WARMUP. ewm(adjust=False) emits a value from bar 0, where both EMAs are
    # simply the first close and the line is ~0 by construction -- not a reading,
    # an artefact. Blanked for `slow` bars so the early session cannot feed the
    # percentile. (The flow-percentile warmup taught this the expensive way:
    # unguarded warmup silently drops or distorts the front of every backtest.)
    line.iloc[:slow] = np.nan
    chg = line.diff(nchg)
    cm = np.asarray(close_min)
    out_l, out_c = np.full(n, np.nan), np.full(n, np.nan)
    for i, m in enumerate(mods):
        j = int(np.searchsorted(cm, m, side="right")) - 1   # last CLOSED 5m bar
        if j >= 0:
            out_l[i] = line.iloc[j]
            out_c[i] = chg.iloc[j]
    return out_l, out_c


def feature_map(tk, win=30, q=0.75, mom="rsi", nchg=3):
    """{date: (mods, fired_call, fired_put)} -- all windows strictly backward."""
    d = pl.read_parquet(f"historical/{tk}.parquet",
                        columns=["start_time", "high", "low", "close", "volume"]).to_pandas()
    et = (pd.to_datetime(d["start_time"], utc=True)
          .dt.tz_convert("America/New_York").dt.tz_localize(None))
    d["date"] = et.dt.date
    d["mod"] = (et.dt.hour * 60 + et.dt.minute).astype(int)
    d = d[(d["mod"] >= RTH_LO) & (d["mod"] <= RTH_HI)]
    for c in ("high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    out = {}
    for dt, g in d.sort_values("mod").groupby("date"):
        g = g.reset_index(drop=True)
        v = g["volume"].astype(float)
        rng = (g["high"] - g["low"]).astype(float)
        if mom == "macd":
            # MACD(5,13,1) line on 5m; its change is the divergence read
            _, ch = _macd_5m(g["mod"].to_numpy(np.int32),
                             g["close"].to_numpy(float), 5, 13, nchg)
            d3 = pd.Series(ch)
        else:
            rsi = pd.Series(_rsi(g["close"].to_numpy(float), 14))
            d3 = rsi.diff(nchg)
        # .shift(1) is what makes these causal: the current bar is excluded from
        # the window it is being compared against.
        vq = v.rolling(win, min_periods=20).quantile(q).shift(1)
        rq = rng.rolling(win, min_periods=20).quantile(q).shift(1)
        lo_q = d3.rolling(win, min_periods=20).quantile(1 - q).shift(1)
        hi_q = d3.rolling(win, min_periods=20).quantile(q).shift(1)
        vs = (v > vq).to_numpy()
        re = (rng > rq).to_numpy()
        # CALL: divergence = RSI rising least  -> d3 in the bottom quartile
        # PUT : the move is DOWN, so divergence = d3 in the TOP quartile
        fade_c = (d3 < lo_q).to_numpy()
        fade_p = (d3 > hi_q).to_numpy()
        out[dt] = (g["mod"].to_numpy(np.int32),
                   vs & re & fade_c, vs & re & fade_p)
    return out


def build_tighten(cand, fmap, up):
    """Per candidate: bool array over the option path's minutes, plus the ARM
    INDEX (first True) so a matched placebo can reuse the same distribution."""
    outs, rates, arms = [], [], []
    for d, m, path in cand:
        mods = path[-1]
        f = fmap.get(d)
        if f is None:
            outs.append(None); arms.append(None); continue
        bm, fc, fp = f
        src = fc if up else fp
        arr = np.zeros(len(mods), bool)
        for i, mm in enumerate(mods):
            j = int(np.searchsorted(bm, mm))
            if j < bm.size and bm[j] == mm and src[j]:
                arr[i] = True
        rates.append(arr.mean())
        outs.append(arr)
        arms.append(int(np.argmax(arr)) if arr.any() else None)
    return outs, (float(np.mean(rates)) if rates else np.nan), arms


def build_placebo(cand, arms, rng):
    """ARM-TIME MATCHED placebo -- the fix for a control that was comparing the
    wrong thing.

    The first version fired at the same RATE but at uniformly random minutes.
    Because the trail LATCHES, all that matters is WHEN it first fires, and a
    uniform draw arms around the middle of the trade while the real detector
    arms at bar ~20-70. Arming later means less time tight, which is closer to
    the (best) baseline -- so the placebo was winning on arm-time distribution,
    not on anything about price. At q=0.90 it beat the detector at 25% and lost
    at 15%, which is what a confounded control looks like.

    This draws each trade's arm index from ANOTHER trade's real arm index, so
    the arm-time distribution matches exactly and only the link to THIS trade's
    price action is broken.
    """
    pool = [x for x in arms if x is not None]
    outs = []
    for (d, m, path), own in zip(cand, arms):
        mods = path[-1]
        arr = np.zeros(len(mods), bool)
        if own is not None and pool:
            k = int(rng.choice(pool))
            if k < len(mods):
                arr[k] = True            # latching: one arm point is enough
        outs.append(arr)
    return outs


def tot(res, oos):
    return sum(p for d, p in res if ((d >= SPLIT) if oos else (d < SPLIT))) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tight", nargs="+", type=float, default=[0.25, 0.15])
    ap.add_argument("--fill", default="botcap")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--q", type=float, default=0.75,
                    help="percentile each feature must clear (0.75 = top quartile)")
    ap.add_argument("--mom", choices=["rsi", "macd"], default="rsi",
                    help="momentum-divergence feature: RSI(14) 1m, or the "
                         "MACD(5,13,1) LINE on 5m (signal=1 => zero histogram)")
    ap.add_argument("--nchg", type=int, default=3,
                    help="bars over which the divergence change is measured")
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    rules = [r for r in sim_core.research_rules(include_paper=a.paper)
             if (r.get("trail_pct") or 0.50) > 0]
    print(f"  {len(rules)} trailing rules, fill={a.fill}, "
          f"tight levels {a.tight}\n")

    arms = {}
    fire_rates = []
    arm_stats = []
    for rule in rules:
        tk = rule["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        cand = sim_core.build_candidates(D, rule, trigs=trigs)
        if not cand:
            continue
        pol = sim_core.policy_for(rule)
        eod = sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        up = rule["direction"] == "CALL"
        fmap = feature_map(tk, q=a.q, mom=a.mom, nchg=a.nchg)
        tg, rate, arms_idx = build_tighten(cand, fmap, up)
        # TIME TO ARM. If the detector latches within a minute or two of entry it
        # is not marking "a moment" -- it IS the always-tight arm, and the two
        # results should converge. This diagnoses that rather than assuming it.
        ttl = []
        for arr, (d, m, path) in zip(tg, cand):
            if arr is None or not arr.any():
                ttl.append(np.nan); continue
            ttl.append(float(np.argmax(arr)))
        ttl = np.array([x for x in ttl if np.isfinite(x)], float)
        never = sum(1 for x in tg if x is None or not x.any())
        arm_stats.append((rule["name"], len(cand), rate,
                          np.median(ttl) if ttl.size else np.nan,
                          never / max(len(cand), 1)))
        pc = build_placebo(cand, arms_idx, rng)
        fire_rates.append(rate)

        def W(**kw):
            return sim_core.walk(cand, pol, eod, fill=a.fill, cush_cap=cap, **kw)

        arms.setdefault("baseline trail50", []).extend(W())
        for t in a.tight:
            arms.setdefault(f"ALWAYS {t:.0%}", []).extend(
                sim_core.walk(cand, dict(pol, trail=t), eod, fill=a.fill, cush_cap=cap))
            arms.setdefault(f"detector -> {t:.0%}", []).extend(
                W(tightens=tg, trail_tight=t))
            arms.setdefault(f"* placebo -> {t:.0%}", []).extend(
                W(tightens=pc, trail_tight=t))
        print(f"  {rule['name']:24} {len(cand):>5} cands   "
              f"detector fires on {rate*100:>5.2f}% of minutes", flush=True)

    if not arms:
        print("  nothing"); return
    print(f"\n  ARM TIMING -- is this a moment, or effectively always-on?")
    print(f"  {'rule':24} {'cands':>6} {'fire%':>7} {'med bars to arm':>16} "
          f"{'never arms':>11}")
    for nm, nc, rt, md, nv in arm_stats:
        print(f"  {nm:24} {nc:>6} {rt*100:>6.2f}% {md:>16.0f} {nv*100:>10.0f}%")

    print(f"\n  mean detector fire rate {np.mean(fire_rates)*100:.2f}% of minutes "
          f"(~{np.mean(fire_rates)*390:.1f} bars/session)")

    print(f"\n{'='*92}")
    print(f"  CHOKEHOLD vs THE TWO CONTROLS")
    print(f"{'='*92}")
    print(f"  {'arm':24} {'IS trd':>7} {'IS tot%':>10} {'OOS trd':>8} "
          f"{'OOS tot%':>10} {'vs base':>9}")
    base = tot(arms["baseline trail50"], True)
    order = ["baseline trail50"]
    for t in a.tight:
        order += [f"ALWAYS {t:.0%}", f"detector -> {t:.0%}", f"* placebo -> {t:.0%}"]
    for k in order:
        r = arms.get(k)
        if not r:
            continue
        i = [(d, p) for d, p in r if d < SPLIT]
        o = [(d, p) for d, p in r if d >= SPLIT]
        print(f"  {k:24} {len(i):>7} {tot(r,0):>+10.1f} {len(o):>8} "
              f"{tot(r,1):>+10.1f} {tot(r,1)-base:>+9.1f}")

    print(f"\n  HOW TO READ IT")
    print(f"  The detector must beat BOTH the baseline AND the ALWAYS arm at the")
    print(f"  same tightness. Beating only the baseline means a tighter trail")
    print(f"  suits the book and the detector contributed nothing.")
    print(f"  The PLACEBO fires at the same rate at random minutes: if it matches")
    print(f"  the detector, the TIMING carries nothing and only the rate matters.")
    print(f"  Trade counts should be near-identical across arms -- this changes")
    print(f"  the stop, not whether a trade is taken.")


if __name__ == "__main__":
    main()

