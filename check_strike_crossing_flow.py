"""
check_strike_crossing_flow.py
=============================
DOES 0DTE/1DTE FLOW AT THE STRIKE SPOT IS WALKING TOWARD SAY WHETHER IT GETS
THERE -- AND WHETHER IT KEEPS GOING?

THE HYPOTHESIS (the operator's, 2026-09-26)
    On SPY / QQQ / IWM, buy-side volume at the +1 OTM strike rises AT OR BEFORE
    spot moves to it (front-running); once spot arrives or passes, that strike
    is unloaded (sell-side volume); and the cycle repeats one strike further
    on. If real, the flow at the next strike should carry information about
    near-term price movement.

WHAT THE FEASIBILITY PASS ESTABLISHED (price only, no flow looked at)
    Approach -> arrival   spot within $0.25 of a strike crosses it within 15m:
                          IWM 45-55%, SPY 58-63%, QQQ 62-69%
    Crossing -> onward    a first crossing reaches the NEXT strike within 30m:
                          IWM 14-22%, SPY 29-32%, QQQ 31-39%;  ~60% FALL BACK
    So the sharp question is whether flow separates the ~30% of crossings that
    carry on from the ~60% that fail.
    Data: silver 2023-10-12..2026-09-18. IWM has had a same-day expiry every
    session only since 2024-04-19, so IWM starts there in both arms.

🚨 THE TRAP THIS DESIGN EXISTS TO AVOID: MONEYNESS MECHANICS
    0DTE volume concentrates at the money, so volume at a strike rises as spot
    approaches it whether or not anyone knows anything. The buy/sell balance
    also drifts with moneyness (cheap OTM calls get bought as lottery tickets,
    ITM ones get sold). "Bought before arrival, sold after" can therefore
    appear with ZERO predictive content. Every feature is a RESIDUAL against
    what is normal for that ticker, expiry arm, option type, distance from spot
    and time of day -- a baseline fitted on IN-SAMPLE days only.

🚨 SPOT COMES FROM historical/{T}.parquet, NOT FROM THE OPTION ROWS
    Measured 2026-09-26: spot rebuilt from option rows' underlying_close is
    aligned at lag 0 but half-stale -- it correlates 0.40-0.55 with the PREVIOUS
    minute's move, because each row is stamped at that contract's last print.
    Half a minute of lag on spot would shift every event late and make ordinary
    flow read as front-running. The independent 1m close has no such lag.

🚨 THE FEATURE IS MULTI-LEG FILTERED (check_opra_multileg.py, 2026-09-26)
    Multi-leg prints 42% ask / 54% bid -- not at mid -- and leans the measure
    toward "selling". Dropping contract-minutes >25% multi-leg tracks the true
    single-leg measure at corr 0.976 (unfiltered 0.904). Per that script's
    pre-stated rule the filtered measure is PRIMARY; unfiltered is control C8.

DEFINITIONS (fixed before any outcome was computed)
    side s        +1: calls, spot moving UP through strikes
                  -1: puts,  spot moving DOWN through strikes
                  Worked in q = s * spot, so both sides use one code path.
    aggression    (sum ask_vol - sum bid_vol) / (sum ask_vol + sum bid_vol)
                  over a 10-minute window ENDING at the event minute
                  (inclusive -- all of that volume printed before the event
                  minute's close, which is where outcomes start). Missing if
                  ask+bid < 10 contracts.
    residual      aggression - baseline[ticker, arm, type, dist(0.1$),
                  30-min bucket], baseline = IS mean over every window at
                  |dist| <= 3, cells with < 50 windows dropped. dist is signed
                  distance in the OTM direction: call K - spot, put spot - K.
    arms          PRIMARY  same-day expiry (dte_days == 0)
                  SECONDARY next listed expiry (dte_rank == 1), scored with the
                  same criteria and reported as its own verdict
    Test A        APPROACH: first minute per strike/day/side in 09:45-15:30
                  where spot enters [K-0.25, K) having been <= K-0.50 within
                  the prior 15m. Feature: residual at K. Outcome: crosses
                  K+0.10 within 15m.
    Test B        FIRST CROSSING of K (hysteresis $0.10) in 09:45-15:30.
                  Feature: residual at the NEXT strike K+s. Outcome: reaches
                  K+s+0.10 before falling back to K-0.10, within 30m.
    effect        per ticker x side cell: P(outcome | top tercile) -
                  P(outcome | bottom tercile), tercile cuts from IS events.
                  POOLED = equal-weight mean of the 6 cells, in pp.

PRE-COMMITTED CRITERIA -- each of Test A and Test B, per arm
    C1  IS pooled effect  >= +5pp
    C2  OOS pooled effect >= +5pp                 (split 2025-08-21)
    C3  >= 5 of 6 equal-count calendar slices positive
    C4  >= 5 of 6 ticker x side cells positive (full sample)
    C5  OOS pooled effect > p95 of 1000 feature permutations within
        (ticker, side, hour) among OOS events
    C6  MOMENTUM: mean effect within terciles of the 10m signed spot move
        >= +2.5pp (full sample). Buying calls because spot rose is chasing,
        not front-running.
    C7  SENTIMENT: mean effect within terciles of broad call-minus-put
        aggression across every strike within $3 >= +2.5pp (full sample).
        The claim is about THIS strike; general bullishness is what the
        existing flow trigger already measures.
    C8  MULTI-LEG: the unfiltered feature, same test, >= +2.5pp same sign.
    PASS = all eight. Anything else is reported as what it is.
    5pp: pooled over ~9k IS events a 5pp tercile gap is ~4 SE, and it is small
    enough not to miss a modest real edge. Controls use the full sample
    because stratifying OOS three ways leaves ~2.9pp of noise per stratum.

REPORTED, NOT SCORED
    - forward spot return at 15m (A) / 30m (B), top minus bottom, in bp
    - Test A with the feature at K+s instead of K
    - PICTURE D: residual aggression at K and K+s from -30 to +30 minutes
      around each Test B crossing, split by continued vs fell back. This is
      where a "load, then unload, then move up a strike" cycle would show.
      Descriptive only.

🚨 "SELL-SIDE" IS NOT "UNLOADING"
    Bid-side volume is sellers hitting the bid: longs closing (the hypothesis)
    OR new writers opening. Open interest is daily, so they cannot be told
    apart intraday. This measures seller AGGRESSION and says so.

🚨 MACHINERY CHECKS THAT RUN FIRST (METHODOLOGY 6c / 6d)
    J1  200 random Test A features recomputed straight from the raw option
        rows; must match exactly, or the join is wrong and nothing is printed.
    J2  spot sits in [K-0.25, K) at >= 99% of Test A events.
    M1  a feature equal to outcome + noise must produce a pooled effect
        >= +20pp; a pure-noise feature must land inside its permutation null.
        If the statistic cannot see a planted effect it cannot see a real one.

RESULT -- 2026-09-26. FAIL, all four verdicts (A/B x 0DTE/next), 0 of 32
criteria passed. Machinery clean: J1 200/200 exact, J2 100.00% on 15,978
events (calls and puts), M1 planted +68.2pp / noise +0.32pp.

    The front-running hypothesis is not merely null; the sign is WRONG
    everywhere. More buy-aggression at the strike being approached means
    spot is LESS likely to reach it, in every ticker x side cell, both arms,
    IS and OOS:
        Test A 0DTE   IS -4.95pp  OOS -7.14pp   cells 0/6 positive
                      OOS arrival by tercile  low 63.3% / mid 63.0% / high 56.1%
        Test B 0DTE   IS -3.26pp  OOS -4.21pp   cells 0/6 positive
                      OOS continuation        low 30.2% / mid 32.1% / high 26.0%
    It survives momentum (C6), sentiment (C7), unfiltered volume (C8), and
    exact distance within the zone (post-hoc: -5.99pp).

🚨 BUT MOST OF THAT CONTRARIAN EFFECT IS TIME OF DAY, AND C5 IS WHAT SAID SO.
    The within-hour shuffle null is NOT centred on zero: its mean is -3.93pp
    (A) and -2.46pp (B). Hours whose residual aggression runs high are hours
    whose arrival rates run low, and the IS-fitted baseline does not fully
    remove that out of sample. Net of it:
        Test A  -3.20pp beyond the null mean -- observed -7.14 sits just
                inside the null's p5 (-6.85); marginal at best
        Test B  -1.76pp -- inside the null (p5 -4.96); indistinguishable
                from noise
    Forward spot move, top minus bottom tercile: -0.46bp at 15m (A),
    -1.84bp at 30m (B). Economically nothing.
    The raw -7pp would have read as a strong contrarian signal. It is about
    40% signal and 60% clock, and only a null that preserved the clock could
    tell. Keep that shuffle design for anything that residualises on
    time-of-day with an IS-only baseline.

PICTURE D -- what the "cycle" actually looks like, 0DTE, 17.8k crossings
    Before the crossing (-30 to -5 min) residual aggression at K and K+1 is
    flat, within +/-0.013 of normal, for crossings that continue AND ones
    that fail. There is no visible build-up of buying ahead of the move.
    At the crossing minute buying jumps (+0.046 continued, +0.053 failed) --
    the same size either way, so it carries no information: it is flow
    REACTING to the move, not leading it.
    After the crossing, seller aggression at K appears in both groups but is
    much stronger when the move FAILS (-0.050 at +10m vs -0.010) -- selling
    that coincides with price falling back, i.e. reaction again.
    "Load K+1, unload K, repeat" does not appear at 10-minute resolution.

WHAT THIS DID NOT TEST (not a reason to doubt the result, a map of what's left)
    - buy VOLUME level rather than the ask-vs-bid RATIO. A strike where both
      sides get busier moves the ratio little. TESTED 2026-09-26 in
      check_strike_crossing_volume.py: FAIL, OOS flat, and buy and sell
      volume turn out to rise together ahead of crossings.
    - sub-minute timing, and sweep/block-only flow. Silver cannot see either;
      the bronze tape can, but only 12 sessions exist so far.

Usage:  python check_strike_crossing_flow.py
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
import time

import numpy as np
import polars as pl

NSF = "_nsf_cache"
TICKERS = ["SPY", "QQQ", "IWM"]
START = {"SPY": dt.date(2023, 10, 12), "QQQ": dt.date(2023, 10, 12),
         "IWM": dt.date(2024, 4, 19)}
SPLIT = dt.date(2025, 8, 21)
WIN, MIN_AB, MSHARE = 10, 10, 0.25
ZONE, FAR, BAND, LOOK = 0.25, 0.50, 0.10, 15
H_A, H_B = 15, 30
T0, T1 = dt.time(9, 45), dt.time(15, 30)
DMAX = 3.0
MIN_CELL = 50
TAUS = list(range(-30, 31, 5))
FLOOR, CTRL_FLOOR = 5.0, 2.5
N_PERM = 1000
rng = np.random.default_rng(20260926)


# ---------------------------------------------------------------- per-day grid
def day_minutes(day: dt.date) -> pl.DataFrame:
    """Every RTH minute 09:30..15:59 ET, whether or not the price file has it.

    🚨 THE WINDOW IS 10 MINUTES OF CLOCK, NOT 10 ROWS. A rolling sum over rows
    spans 11+ minutes wherever the 1m price file skips one, and silently mixes
    in older volume. So the grid carries every minute and the sum is by time.
    """
    s = dt.datetime(day.year, day.month, day.day, 9, 30)
    e = dt.datetime(day.year, day.month, day.day, 15, 59)
    return pl.DataFrame({"minute_et": pl.datetime_range(
        s, e, "1m", time_unit="us", time_zone="America/New_York", eager=True)})


def windows(opt: pl.DataFrame, spot: pl.DataFrame, day: dt.date) -> pl.DataFrame:
    """Trailing-WIN aggression for every (type, strike, minute) near spot.

    Both variants: x = multi-leg filtered (PRIMARY), u = unfiltered (C8).
    """
    o = opt.with_columns(
        (pl.col("multi_volume") / pl.col("volume").clip(1)).alias("ms"))
    o = o.with_columns(
        pl.when(pl.col("ms") <= MSHARE).then(pl.col("ask_volume")).otherwise(0).alias("ax"),
        pl.when(pl.col("ms") <= MSHARE).then(pl.col("bid_volume")).otherwise(0).alias("bx"))
    o = o.group_by("option_type", "strike", "minute_et").agg(
        pl.col("ax").sum(), pl.col("bx").sum(),
        pl.col("ask_volume").sum().alias("au"), pl.col("bid_volume").sum().alias("bu"))
    lo, hi = spot["spot"].min() - DMAX - 1, spot["spot"].max() + DMAX + 1
    ks = o.filter(pl.col("strike").is_between(lo, hi)).select(
        "option_type", "strike").unique()
    grid = (ks.join(day_minutes(day), how="cross")
            .join(spot.select("minute_et", "spot"), on="minute_et", how="left"))
    g = (grid.join(o, on=["option_type", "strike", "minute_et"], how="left")
         .with_columns(pl.col("ax", "bx", "au", "bu").fill_null(0))
         .sort("option_type", "strike", "minute_et"))
    # window (t - 10m, t], closed right -- exactly what J1 recomputes
    roll = {c: pl.col(c).rolling_sum_by("minute_et", f"{WIN}m").over("option_type", "strike")
            for c in ("ax", "bx", "au", "bu")}
    g = g.with_columns(**{f"r{c}": e for c, e in roll.items()})
    dist = (pl.when(pl.col("option_type") == "call")
            .then(pl.col("strike") - pl.col("spot"))
            .otherwise(pl.col("spot") - pl.col("strike")))
    g = g.with_columns(dist.alias("dist")).filter(pl.col("dist").abs() <= DMAX)

    def agg(a, b):
        return pl.when((pl.col(a) + pl.col(b)) >= MIN_AB).then(
            (pl.col(a) - pl.col(b)) / (pl.col(a) + pl.col(b)))
    return g.with_columns(
        agg("rax", "rbx").alias("fx"), agg("rau", "rbu").alias("fu"),
        (pl.col("dist") * 10).round().cast(pl.Int16).alias("db"),
        ((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
          + pl.col("minute_et").dt.minute().cast(pl.Int32) - 570) // 30)
        .cast(pl.Int16).alias("tb"))


def sentiment(w: pl.DataFrame) -> pl.DataFrame:
    """Broad call-minus-put filtered aggression across every strike within $3."""
    s = w.group_by("minute_et", "option_type").agg(
        pl.col("rax").sum(), pl.col("rbx").sum())
    s = s.with_columns(((pl.col("rax") - pl.col("rbx"))
                        / (pl.col("rax") + pl.col("rbx")).clip(1)).alias("a"))
    s = s.pivot(on="option_type", index="minute_et", values="a").fill_null(0.0)
    for c in ("call", "put"):
        if c not in s.columns:
            s = s.with_columns(pl.lit(0.0).alias(c))
    return s.select("minute_et", (pl.col("call") - pl.col("put")).alias("sent"))


# ---------------------------------------------------------------- events
def events(px: np.ndarray, ok: np.ndarray):
    """Yield (test, side, k_q, i, outcome, fwd_bp) in q-space, strike k_q."""
    n = len(px)
    for s in (1, -1):
        q = px * s
        for k in range(int(np.floor(q.min())), int(np.ceil(q.max())) + 1):
            # -------- Test A: approach -> arrival
            for i in range(LOOK, n - H_A):
                if (ok[i] and k - ZONE <= q[i] < k and q[i - 1] < k - ZONE
                        and q[i - LOOK:i].min() <= k - FAR):
                    fut = q[i + 1:i + 1 + H_A]
                    # 🚨 DIFFERENCE over |level|, NOT q[i+H]/q[i]: for puts q
                    # is negative and a ratio of two negatives drops the sign,
                    # which would score a falling market as a gain on the put
                    # side.
                    yield ("A", s, k, i, int(fut.max() >= k + BAND),
                           (q[i + H_A] - q[i]) / abs(q[i]) * 1e4)
                    break
            # -------- Test B: first crossing -> next strike
            st = None
            for i in range(n - H_B):
                if q[i] <= k - BAND:
                    st = -1
                elif q[i] >= k + BAND:
                    if st == -1 and ok[i]:
                        res = 0
                        for p in q[i + 1:i + 1 + H_B]:
                            if p >= k + 1 + BAND:
                                res = 1; break
                            if p <= k - BAND:
                                break
                        yield ("B", s, k, i, res, (q[i + H_B] - q[i]) / abs(q[i]) * 1e4)
                        break
                    st = 1


# ---------------------------------------------------------------- statistics
def cell_effect(feat, y, lo, hi):
    top, bot = y[feat >= hi], y[feat <= lo]
    if len(top) < 20 or len(bot) < 20:
        return np.nan
    return (top.mean() - bot.mean()) * 100


def pooled(ev: pl.DataFrame, cuts: dict, feat="res") -> tuple[float, dict]:
    per = {}
    for (t, s), g in ev.group_by("ticker", "side"):
        c = cuts.get((t, s))
        if c is None:
            continue
        per[(t, s)] = cell_effect(g[feat].to_numpy(), g["y"].to_numpy(), *c)
    vals = [v for v in per.values() if np.isfinite(v)]
    return (float(np.mean(vals)) if vals else np.nan), per


def tercile_cuts(ev: pl.DataFrame, feat="res"):
    cuts = {}
    for (t, s), g in ev.group_by("ticker", "side"):
        x = g[feat].drop_nulls().to_numpy()
        if len(x) >= 60:
            cuts[(t, s)] = (np.quantile(x, 1 / 3), np.quantile(x, 2 / 3))
    return cuts


def stratified(ev, cuts, ctrl, feat="res"):
    """Mean over the three control terciles of the pooled effect inside each."""
    parts = []
    for (t, s), g in ev.filter(pl.col(ctrl).is_not_null()).group_by("ticker", "side"):
        c = g[ctrl].to_numpy()
        lo, hi = np.nanquantile(c, [1 / 3, 2 / 3])
        parts.append(g.with_columns(
            pl.when(pl.col(ctrl) <= lo).then(0).when(pl.col(ctrl) <= hi).then(1)
            .otherwise(2).alias("_st")))
    e = pl.concat(parts)
    vals = [pooled(e.filter(pl.col("_st") == k), cuts, feat)[0] for k in range(3)]
    return float(np.nanmean(vals)), vals


def permute_null(ev, cuts, n=N_PERM):
    base = ev.with_columns(pl.col("minute").dt.hour().alias("_hr"))
    groups = [g for _, g in base.group_by("ticker", "side", "_hr")]
    out = np.empty(n)
    for r in range(n):
        shuf = []
        for g in groups:
            f = g["res"].to_numpy().copy()
            rng.shuffle(f)
            shuf.append(g.with_columns(pl.Series("res", f)))
        out[r] = pooled(pl.concat(shuf), cuts)[0]
    return out


def score(ev: pl.DataFrame, label: str, verbose=True):
    ev = ev.filter(pl.col("res").is_not_null())
    is_, oos = ev.filter(~pl.col("oos")), ev.filter(pl.col("oos"))
    cuts = tercile_cuts(is_)
    c1, _ = pooled(is_, cuts)
    c2, _ = pooled(oos, cuts)
    # calendar slices over the union of sessions
    days = np.array(sorted(ev["date"].unique().to_list()), dtype="datetime64[D]")
    edges = np.array([days[int(len(days) * k / 6)] for k in range(1, 6)])
    idx = np.searchsorted(edges, ev["date"].to_numpy().astype("datetime64[D]"), side="right")
    sl = ev.with_columns(pl.Series("_sl", idx))
    slices = [pooled(sl.filter(pl.col("_sl") == i), cuts)[0] for i in range(6)]
    _, cells = pooled(ev, cuts)
    null = permute_null(oos, cuts)
    p95 = float(np.quantile(null, 0.95))
    c6, c6v = stratified(ev, cuts, "mom")
    c7, c7v = stratified(ev, cuts, "sent_s")
    ev_u = ev.filter(pl.col("res_u").is_not_null())
    cuts_u = tercile_cuts(ev_u.filter(~pl.col("oos")), "res_u")
    c8, _ = pooled(ev_u, cuts_u, "res_u")

    crit = [
        ("C1 IS pooled >= +5pp", c1, c1 >= FLOOR),
        ("C2 OOS pooled >= +5pp", c2, c2 >= FLOOR),
        ("C3 >= 5/6 slices positive", sum(v > 0 for v in slices),
         sum(v > 0 for v in slices) >= 5),
        ("C4 >= 5/6 cells positive", sum(v > 0 for v in cells.values()),
         sum(v > 0 for v in cells.values()) >= 5),
        (f"C5 OOS > perm p95 ({p95:+.2f})", c2, c2 > p95),
        ("C6 within momentum >= +2.5", c6, c6 >= CTRL_FLOOR),
        ("C7 within sentiment >= +2.5", c7, c7 >= CTRL_FLOOR),
        ("C8 unfiltered >= +2.5 same sign", c8,
         c8 >= CTRL_FLOOR and np.sign(c8) == np.sign(c2)),
    ]
    passed = all(c[2] for c in crit)
    if verbose:
        print(f"\n  ---- {label}   n={ev.height:,} (IS {is_.height:,} / OOS {oos.height:,})")
        for name, v, ok in crit:
            vv = f"{v:+.2f}pp" if isinstance(v, float) else f"{v}"
            print(f"    {'PASS' if ok else 'fail'}  {name:<34} {vv}")
        print("    slices: " + "  ".join(f"{v:+.1f}" for v in slices))
        print("    cells:  " + "  ".join(f"{t}{'+' if s > 0 else '-'} {v:+.1f}"
                                        for (t, s), v in sorted(cells.items())))
        print(f"    C6 strata {[round(x, 1) for x in c6v]}   "
              f"C7 strata {[round(x, 1) for x in c7v]}")
        print(f"    VERDICT: {'PASS' if passed else 'FAIL'}")
    return passed, cuts, c2, p95


def fwd_bp(ev, cuts):
    out = []
    for (t, s), g in ev.filter(pl.col("res").is_not_null()).group_by("ticker", "side"):
        c = cuts.get((t, s))
        if c is None:
            continue
        f, r = g["res"].to_numpy(), g["fwd"].to_numpy()
        out.append(r[f >= c[1]].mean() - r[f <= c[0]].mean())
    return float(np.mean(out))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    hist = {}
    for t in TICKERS:
        h = (pl.read_parquet(f"historical/{t}.parquet").select("minute_et", "close")
             .with_columns(pl.col("minute_et").dt.convert_time_zone("America/New_York"))
             .filter(pl.col("minute_et").dt.time().is_between(dt.time(9, 30), dt.time(15, 59)))
             .drop_nulls().unique("minute_et").sort("minute_et")
             .with_columns(pl.col("minute_et").dt.date().alias("date")))
        hist[t] = {d: g for (d,), g in h.group_by(["date"])}

    base_parts, ev_rows, feat_rows, prof_rows = [], [], [], []
    j1_sample = []
    files = sorted(glob.glob(os.path.join(NSF, "opt", "*.parquet")))
    # smoke-test hook: NSF_STRIDE=40 runs every 40th session. Its numbers are
    # not a result; only the unstrided run is scored.
    files = files[::int(os.getenv("NSF_STRIDE", "1"))]
    for n_f, f in enumerate(files):
        day = dt.date.fromisoformat(os.path.basename(f)[:10])
        opt_day = pl.read_parquet(f)
        for t in TICKERS:
            if day < START[t] or day not in hist[t]:
                continue
            sp = hist[t][day].rename({"close": "spot"})
            if sp.height < 300:
                continue
            px = sp["spot"].to_numpy()
            mins = sp["minute_et"].to_list()
            tm = np.array([m.time() for m in mins])
            ok = (tm >= T0) & (tm <= T1)
            evs = list(events(px, ok))
            for arm, flt in (("0dte", pl.col("dte_days") == 0),
                             ("next", pl.col("dte_rank") == 1)):
                o = opt_day.filter((pl.col("ticker") == t) & flt)
                if o.is_empty():
                    continue
                w = windows(o, sp, day)
                if day < SPLIT:
                    base_parts.append(w.filter(pl.col("fx").is_not_null() | pl.col("fu").is_not_null())
                                      .group_by("option_type", "db", "tb").agg(
                        pl.col("fx").sum().alias("sx"), pl.col("fx").count().alias("nx"),
                        pl.col("fu").sum().alias("su"), pl.col("fu").count().alias("nu"))
                        .with_columns(pl.lit(t).alias("ticker"), pl.lit(arm).alias("arm")))
                sent = sentiment(w)
                look = w.select("option_type", "strike", "minute_et", "fx", "fu",
                                "db", "tb", "dist")
                rows = []
                for test, s, k, i, y, fwd in evs:
                    K = float(k * s)                       # back to spot-space
                    fk = K if test == "A" else K + s       # A: this strike, B: next
                    mom = s * (px[i] / px[max(0, i - WIN)] - 1) * 1e4
                    rows.append((test, s, K, fk, i, y, fwd, mom, float(K + s)))
                if not rows:
                    continue
                e = pl.DataFrame(rows, schema=["test", "side", "K", "fk", "i", "y",
                                               "fwd", "mom", "K_next"], orient="row")
                # 🚨 minute taken from the price table's OWN column by index, so
                # it carries exactly the dtype and time zone the grid joins on.
                # Rebuilding it from Python datetimes risks a silent UTC/ET or
                # ns/us mismatch -- and a left join that matches nothing still
                # returns every row, just with null features.
                e = e.with_columns(sp["minute_et"].gather(e["i"]).alias("minute"),
                                   pl.when(pl.col("side") > 0).then(pl.lit("call"))
                                   .otherwise(pl.lit("put")).alias("option_type"))
                ej = e.join(look.rename({"strike": "fk", "minute_et": "minute"}),
                            on=["option_type", "fk", "minute"], how="left")
                # Test A secondary: the feature at K+s
                ej = ej.join(look.select("option_type", "strike", "minute_et",
                                         pl.col("fx").alias("fx_next"),
                                         pl.col("db").alias("db_next"),
                                         pl.col("tb").alias("tb_next"))
                             .rename({"strike": "K_next", "minute_et": "minute"}),
                             on=["option_type", "K_next", "minute"], how="left")
                ej = ej.join(sent.rename({"minute_et": "minute"}), on="minute", how="left")
                ej = ej.with_columns(pl.lit(t).alias("ticker"), pl.lit(arm).alias("arm"),
                                     pl.lit(day).alias("date"),
                                     (pl.col("sent") * pl.col("side")).alias("sent_s"))
                ev_rows.append(ej)
                if arm == "0dte" and rng.random() < 0.15:
                    a = ej.filter((pl.col("test") == "A") & pl.col("fx").is_not_null())
                    if a.height:
                        r = a.sample(1, seed=int(rng.integers(1e9))).row(0, named=True)
                        j1_sample.append((t, day, r, o))
                # PICTURE D, primary arm only
                if arm == "0dte":
                    b = e.filter(pl.col("test") == "B").with_columns(
                        pl.lit(0).alias("_"))
                    for tau in TAUS:
                        for which in ("K", "K_next"):
                            pr = b.with_columns(
                                (pl.col("minute") + pl.duration(minutes=tau)).alias("m2"),
                                pl.col(which).alias("k2"))
                            pj = pr.join(look.rename({"strike": "k2", "minute_et": "m2"}),
                                         on=["option_type", "k2", "m2"], how="inner")
                            prof_rows.append(pj.select(
                                "side", "K", "minute", "y", "fx", "db", "tb", "option_type")
                                .with_columns(pl.lit(t).alias("ticker"),
                                              pl.lit(day).alias("date"),
                                              pl.lit(tau).alias("tau"),
                                              pl.lit(which).alias("which")))
        if (n_f + 1) % 100 == 0:
            print(f"  {n_f + 1}/{len(files)} sessions  {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------ baseline
    base = (pl.concat(base_parts).group_by("ticker", "arm", "option_type", "db", "tb")
            .agg(pl.col("sx").sum(), pl.col("nx").sum(), pl.col("su").sum(), pl.col("nu").sum()))
    base = base.with_columns(
        pl.when(pl.col("nx") >= MIN_CELL).then(pl.col("sx") / pl.col("nx")).alias("bx"),
        pl.when(pl.col("nu") >= MIN_CELL).then(pl.col("su") / pl.col("nu")).alias("bu"))
    bk = ["ticker", "arm", "option_type", "db", "tb"]
    ev = pl.concat(ev_rows, how="diagonal_relaxed")
    ev = (ev.join(base.select(bk + ["bx", "bu"]), on=bk, how="left")
          .join(base.select(bk + ["bx"]).rename({"db": "db_next", "tb": "tb_next",
                                                 "bx": "bx_next"}),
                on=["ticker", "arm", "option_type", "db_next", "tb_next"], how="left")
          .with_columns((pl.col("fx") - pl.col("bx")).alias("res"),
                        (pl.col("fu") - pl.col("bu")).alias("res_u"),
                        (pl.col("fx_next") - pl.col("bx_next")).alias("res_next"),
                        (pl.col("date") >= SPLIT).alias("oos")))
    ev.write_parquet(os.path.join(NSF, "crossing_events.parquet"))
    print(f"\nbuilt {ev.height:,} event-arm rows in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------ J1 / J2
    print("\n=== machinery checks ===")
    bad = 0
    for t, day, r, o in j1_sample[:200]:
        m1 = r["minute"]
        raw = o.filter((pl.col("option_type") == r["option_type"]) &
                       (pl.col("strike") == r["fk"]) &
                       (pl.col("minute_et") <= m1) &
                       (pl.col("minute_et") > m1 - dt.timedelta(minutes=WIN)))
        raw = raw.filter((pl.col("multi_volume") / pl.col("volume").clip(1)) <= MSHARE)
        a, b_ = raw["ask_volume"].sum(), raw["bid_volume"].sum()
        direct = (a - b_) / (a + b_) if (a + b_) >= MIN_AB else None
        if direct is None or abs(direct - r["fx"]) > 1e-9:
            bad += 1
    print(f"  J1 feature recomputed from raw rows: {len(j1_sample[:200]) - bad}/"
          f"{len(j1_sample[:200])} exact")
    if bad:
        sys.exit("  J1 FAILED -- the feature join is wrong; no result is printed.")
    # dist is OTM-signed (call K-spot, put spot-K), so an approach from the
    # right side puts it in (0, 0.25] on BOTH sides -- if puts land at
    # (-0.25, 0] the mirroring is broken.
    a0 = ev.filter((pl.col("test") == "A") & (pl.col("arm") == "0dte")
                   & pl.col("dist").is_not_null())
    d_ok = a0.select(((pl.col("dist") > 0) & (pl.col("dist") <= ZONE + 1e-9)).mean()).item()
    print(f"  J2 spot in [K-0.25, K) at Test A events: {d_ok:.2%}  "
          f"(n={a0.height:,}, calls {a0.filter(pl.col('side') > 0).height:,} / "
          f"puts {a0.filter(pl.col('side') < 0).height:,})")
    if d_ok < 0.99:
        sys.exit("  J2 FAILED -- events are not where they claim to be.")

    # M1 synthetic
    for arm in ("0dte",):
        e = ev.filter((pl.col("test") == "B") & (pl.col("arm") == arm) &
                      pl.col("res").is_not_null())
        noise = rng.normal(0, 0.5, e.height)
        planted = e.with_columns(pl.Series("res", e["y"].to_numpy() + noise))
        cuts = tercile_cuts(planted.filter(~pl.col("oos")))
        pe, _ = pooled(planted.filter(pl.col("oos")), cuts)
        pure = e.with_columns(pl.Series("res", rng.normal(0, 1, e.height)))
        cp = tercile_cuts(pure.filter(~pl.col("oos")))
        pn, _ = pooled(pure.filter(pl.col("oos")), cp)
        nul = permute_null(pure.filter(pl.col("oos")), cp, n=200)
        print(f"  M1 planted effect -> {pe:+.1f}pp (need >= +20)   pure noise -> {pn:+.2f}pp "
              f"(null p5..p95 {np.quantile(nul, .05):+.2f}..{np.quantile(nul, .95):+.2f})")
        if pe < 20:
            sys.exit("  M1 FAILED -- the statistic cannot see a planted effect.")

    # ------------------------------------------------------------ coverage
    print("\n=== coverage: events with a usable feature ===")
    print(ev.group_by("test", "arm").agg(
        pl.len().alias("events"), pl.col("fx").is_not_null().mean().round(3).alias("has_feature"),
        pl.col("res").is_not_null().mean().round(3).alias("has_residual")).sort("test", "arm"))

    # ------------------------------------------------------------ scoring
    for test, name in (("A", "TEST A  approach -> arrival  (feature at K)"),
                       ("B", "TEST B  first crossing -> next strike  (feature at K+1)")):
        print(f"\n\n==================== {name}")
        for arm in ("0dte", "next"):
            e = ev.filter((pl.col("test") == test) & (pl.col("arm") == arm))
            passed, cuts, c2, p95 = score(e, f"{arm.upper()} arm")
            print(f"    reported: fwd {H_A if test == 'A' else H_B}m spot, top-minus-bottom "
                  f"= {fwd_bp(e, cuts):+.2f}bp (pooled)")
            if test == "A":
                e2 = e.with_columns(pl.col("res_next").alias("res")).filter(
                    pl.col("res").is_not_null())
                c = tercile_cuts(e2.filter(~pl.col("oos")))
                print(f"    reported: feature at K+1 instead  IS "
                      f"{pooled(e2.filter(~pl.col('oos')), c)[0]:+.2f}pp  OOS "
                      f"{pooled(e2.filter(pl.col('oos')), c)[0]:+.2f}pp")

    # ------------------------------------------------------------ PICTURE D
    print("\n\n==================== PICTURE D  (descriptive, 0DTE, Test B crossings)")
    pr = pl.concat(prof_rows).join(base.filter(pl.col("arm") == "0dte")
                                   .select("ticker", "option_type", "db", "tb", "bx"),
                                   on=["ticker", "option_type", "db", "tb"], how="left")
    pr = pr.with_columns((pl.col("fx") - pl.col("bx")).alias("r")).drop_nulls("r")
    pr.write_parquet(os.path.join(NSF, "crossing_profile.parquet"))
    tab = (pr.group_by("which", "y", "tau").agg(pl.col("r").mean().alias("m"), pl.len().alias("n"))
           .sort("which", "y", "tau"))
    for which, lab in (("K", "strike just crossed (K)"), ("K_next", "next strike (K+1)")):
        print(f"\n  residual aggression at the {lab}   (+ = more buying than normal)")
        print("    tau(min)  " + "".join(f"{t:>7}" for t in TAUS))
        for y, yl in ((1, "continued"), (0, "fell back/stalled")):
            r = tab.filter((pl.col("which") == which) & (pl.col("y") == y)).sort("tau")
            print(f"    {yl:<18}" + "".join(f"{v:>+7.3f}" for v in r["m"].to_list()))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
