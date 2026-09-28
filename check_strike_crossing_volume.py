"""
check_strike_crossing_volume.py
===============================
THE SAME QUESTION AS check_strike_crossing_flow.py, ASKED OF BUY *VOLUME*
INSTEAD OF THE BUY-vs-SELL *RATIO*.

WHY THIS EXISTS
    check_strike_crossing_flow (2026-09-26) tested aggression, the ratio
    (ask - bid) / (ask + bid), and FAILED with the sign reversed -- most of the
    reversal turning out to be time-of-day. The operator's hypothesis is worded
    as "+1 OTM strike volume (buy-side) INCREASING at or before the move". A
    strike where both sides get busier barely moves the ratio, so that test
    left this reading untouched. Stated as a gap in its RESULT block before
    this file was written, and pre-registered here before any volume outcome
    was computed.

WHAT IS REUSED, UNCHANGED (imported, not copied -- METHODOLOGY 1)
    events(), windows(), day_minutes(), sentiment(), and every constant:
    the same Test A approach events, the same Test B first crossings, the same
    outcomes, the same 10m window, the same multi-leg filter (>25% dropped),
    the same IS/OOS split at 2025-08-21, spot from historical/{T}.parquet.

THE FEATURE
    buyvol   log(1 + ask_volume) at the strike over the 10m window ending at
             the event minute, multi-leg filtered. Unlike the ratio it is
             defined for every window, including zero-volume ones.
    residual buyvol - baseline[ticker, arm, type, dist(0.1$), 30m bucket],
             baseline = IS mean over EVERY window at |dist| <= 3 (zeros
             included), cells with < 50 windows dropped. Same moneyness
             correction as before, for the same reason: volume at a strike
             rises as spot walks toward it with no information at all.
    Test A feature at K, Test B feature at K+s, exactly as before.

🚨 TWO CHANGES FROM THE RATIO TEST, BOTH FROM WHAT IT TAUGHT US
    1. TERCILES ARE CUT WITHIN (ticker, side, HOUR), from IS events.
       The ratio test cut once per ticker x side. Its within-hour shuffle
       null came back centred at -3.93pp, not zero: hours whose residual ran
       high were hours whose arrival rate ran low, so hour composition leaked
       into the top-vs-bottom comparison and made up ~60% of the headline.
       Cutting within the hour gives every hour equal weight in both the top
       and bottom tercile, so that leak cannot occur by construction.
    2. A NEW CONFOUND NEEDS ITS OWN CONTROL: MARKET ACTIVITY.
       Raw buy volume is high whenever the whole chain is busy, and busy
       markets move more -- so buy volume could "predict" arrival without
       saying anything about THIS strike. C9 stratifies on total ask+bid
       volume across every strike within $3, both types, same window.

PRE-COMMITTED CRITERIA -- each of Test A and Test B, per arm
    C1  IS pooled effect  >= +5pp
    C2  OOS pooled effect >= +5pp                 (split 2025-08-21)
    C3  >= 5 of 6 equal-count calendar slices positive
    C4  >= 5 of 6 ticker x side cells positive (full sample)
    C5  OOS pooled effect > p95 of 1000 feature shuffles within
        (ticker, side, hour) among OOS events
    C6  within 10m signed-momentum terciles        >= +2.5pp (full sample)
    C7  within broad call-minus-put sentiment terciles >= +2.5pp
    C8  unfiltered buy volume, same test          >= +2.5pp, same sign
    C9  within market-activity terciles           >= +2.5pp
    PASS = all nine. Direction pre-registered POSITIVE: more buying at the
    strike -> more likely to arrive / continue. A negative result is reported
    as what it is, and is not a finding in its own right.

MACHINERY CHECKS
    J1  200 random Test A features recomputed from raw option rows; exact.
    J2  spot in [K-0.25, K) at >= 99% of Test A events.
    M1  planted feature (outcome + noise) >= +20pp; pure noise inside its null.
    M2  NEW: the OOS shuffle null's MEAN must lie within +/-1pp of zero. That
        is the direct test of change (1). If it does not, the hour leak is not
        fixed, C1/C2 are also reported NET of the null mean, and that is said
        plainly rather than left for the reader to spot.

REPORTED, NOT SCORED
    - forward spot return 15m (A) / 30m (B), top minus bottom, bp
    - PICTURE D: residual log BUY volume and log SELL volume at K and K+s,
      -30..+30 min around each 0DTE Test B crossing, continued vs fell back.
      The ratio version could not separate "both sides busier" from "only
      buyers busier"; this one shows each side on its own.

RESULT -- 2026-09-26. FAIL, all four verdicts, 0 of 36 criteria passed.
Machinery: J1 200/200 exact, J2 100.00%, M1 planted +66.8pp. M1's noise draw
came out +3.01pp, just outside its own 200-shuffle p95 (+2.09) -- for an iid
feature the shuffle null is exact by exchangeability, so that is a ~1-in-100
draw, not a defect, and a too-narrow null could only have made passing
EASIER, which nothing did.

    M2 CONFIRMED THE HOUR FIX: shuffle-null means -0.34..+0.96pp in all four
    runs, against -3.93 / -2.46 in the ratio test. Cutting terciles within the
    hour removed the leak. Keep that design.

    Buy volume carries no forward information out of sample:
        Test A 0DTE   IS -6.02pp  OOS -1.82pp  (null p95 +3.27)
                      OOS arrival low / mid / high buy volume 63.2 / 62.9 / 61.4%
        Test B 0DTE   IS -5.08pp  OOS -1.61pp  (null p95 +4.15)
                      OOS continuation                        32.0 / 29.6 / 30.4%
    IS leans negative, OOS is flat, cells split by ticker (IWM slightly
    positive, SPY consistently negative), slices mixed. Not a signal in either
    direction. C9 is the most negative control (-9.38pp): holding market
    activity fixed, MORE buying concentrated at the strike goes with LESS
    arrival -- but with OOS flat that is not worth chasing.

PICTURE D ANSWERED THE QUESTION THIS FILE WAS WRITTEN FOR
    Ahead of a crossing (-30 to -5 min) both the strike being crossed and the
    next one run ~20-35% busier than normal -- and BUY and SELL volume rise
    TOGETHER, within a few hundredths of each other at every point:
        K+1, -10 min, continued   buy +0.339  sell +0.367
        K+1, -10 min, fell back   buy +0.364  sell +0.365
    A strike getting busier gets busier on both sides. There is no buy-only
    build-up, and the pre-crossing volume is if anything slightly HIGHER ahead
    of moves that fail. After the crossing, moves that continue show far more
    volume at K and K+1 (+1.0 to +1.8 log, several times normal) than moves
    that fail -- but that is inside the outcome window, flow following price.

CONCLUSION OF THE TWO STUDIES TOGETHER
    Across 3 years, ~34k crossing events, 0DTE and next-expiry, at 10-minute
    resolution: neither the buy/sell balance nor the buy volume at the strike
    spot is approaching, or at the strike beyond it, says whether spot gets
    there or keeps going. Flow at these strikes follows price. What remains
    untested is sub-minute timing and sweep/block-only flow, which needs the
    bronze trade tape accumulated over months.

Usage:  python check_strike_crossing_volume.py
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
import time

import numpy as np
import polars as pl

import check_strike_crossing_flow as F
from check_strike_crossing_flow import (NSF, TICKERS, START, SPLIT, WIN, MSHARE,
                                        ZONE, H_A, H_B, T0, T1, MIN_CELL, TAUS,
                                        events, windows, sentiment)

FLOOR, CTRL_FLOOR = 5.0, 2.5
N_PERM = 1000
rng = np.random.default_rng(20260927)
BK = ["ticker", "arm", "option_type", "db", "tb"]


# ---------------------------------------------------------------- statistics
def hour_of(e: pl.DataFrame) -> pl.DataFrame:
    return e.with_columns(pl.col("minute").dt.hour().alias("hr"))


def cut_table(is_ev: pl.DataFrame, feat: str) -> pl.DataFrame:
    """Tercile cuts per (ticker, side, hour) from IS events."""
    return (is_ev.filter(pl.col(feat).is_not_null())
            .group_by("ticker", "side", "hr")
            .agg(pl.col(feat).quantile(1 / 3).alias("lo"),
                 pl.col(feat).quantile(2 / 3).alias("hi"), pl.len().alias("n"))
            .filter(pl.col("n") >= 30).drop("n"))


def label(ev: pl.DataFrame, cuts: pl.DataFrame, feat: str) -> pl.DataFrame:
    e = ev.join(cuts, on=["ticker", "side", "hr"], how="inner")
    return e.with_columns(
        pl.when(pl.col(feat) <= pl.col("lo")).then(0)
        .when(pl.col(feat) >= pl.col("hi")).then(2).otherwise(1).alias("terc"))


def pooled(lab: pl.DataFrame) -> tuple[float, dict]:
    per = {}
    for (t, s), g in lab.group_by("ticker", "side"):
        top, bot = g.filter(pl.col("terc") == 2)["y"], g.filter(pl.col("terc") == 0)["y"]
        if top.len() >= 20 and bot.len() >= 20:
            per[(t, s)] = (top.mean() - bot.mean()) * 100
    vals = list(per.values())
    return (float(np.mean(vals)) if vals else np.nan), per


def shuffle_null(ev, cuts, feat, n):
    groups = [g for _, g in ev.group_by("ticker", "side", "hr")]
    out = np.empty(n)
    for r in range(n):
        parts = []
        for g in groups:
            f = g[feat].to_numpy().copy()
            rng.shuffle(f)
            parts.append(g.with_columns(pl.Series(feat, f)))
        out[r] = pooled(label(pl.concat(parts), cuts, feat))[0]
    return out


def within(lab, ctrl):
    """Mean over the three per-cell terciles of `ctrl` of the pooled effect."""
    parts = []
    for _, g in lab.filter(pl.col(ctrl).is_not_null()).group_by("ticker", "side"):
        lo, hi = np.quantile(g[ctrl].to_numpy(), [1 / 3, 2 / 3])
        parts.append(g.with_columns(pl.when(pl.col(ctrl) <= lo).then(0)
                                    .when(pl.col(ctrl) <= hi).then(1)
                                    .otherwise(2).alias("_st")))
    if not parts:
        return np.nan, [np.nan] * 3
    e = pl.concat(parts)
    v = [pooled(e.filter(pl.col("_st") == k))[0] for k in range(3)]
    return float(np.nanmean(v)), v


def score(ev: pl.DataFrame, label_txt: str):
    ev = hour_of(ev.filter(pl.col("res").is_not_null()))
    is_, oos = ev.filter(~pl.col("oos")), ev.filter(pl.col("oos"))
    cuts = cut_table(is_, "res")
    lab = label(ev, cuts, "res")
    c1, _ = pooled(lab.filter(~pl.col("oos")))
    c2, _ = pooled(lab.filter(pl.col("oos")))
    days = np.array(sorted(ev["date"].unique().to_list()), dtype="datetime64[D]")
    edges = np.array([days[int(len(days) * k / 6)] for k in range(1, 6)])
    lab = lab.with_columns(pl.Series("_sl", np.searchsorted(
        edges, lab["date"].to_numpy().astype("datetime64[D]"), side="right")))
    slices = [pooled(lab.filter(pl.col("_sl") == i))[0] for i in range(6)]
    _, cells = pooled(lab)
    null = shuffle_null(oos, cuts, "res", N_PERM)
    p95, nmean = float(np.quantile(null, 0.95)), float(null.mean())
    c6, v6 = within(lab, "mom")
    c7, v7 = within(lab, "sent_s")
    c9, v9 = within(lab, "act")
    evu = ev.filter(pl.col("res_u").is_not_null())
    labu = label(evu, cut_table(evu.filter(~pl.col("oos")), "res_u"), "res_u")
    c8, _ = pooled(labu)

    m2 = abs(nmean) <= 1.0
    crit = [
        ("C1 IS pooled >= +5pp", c1, c1 >= FLOOR),
        ("C2 OOS pooled >= +5pp", c2, c2 >= FLOOR),
        ("C3 >= 5/6 slices positive", sum(v > 0 for v in slices), sum(v > 0 for v in slices) >= 5),
        ("C4 >= 5/6 cells positive", sum(v > 0 for v in cells.values()),
         sum(v > 0 for v in cells.values()) >= 5),
        (f"C5 OOS > shuffle p95 ({p95:+.2f})", c2, c2 > p95),
        ("C6 within momentum >= +2.5", c6, c6 >= CTRL_FLOOR),
        ("C7 within sentiment >= +2.5", c7, c7 >= CTRL_FLOOR),
        ("C8 unfiltered >= +2.5 same sign", c8, c8 >= CTRL_FLOOR and np.sign(c8) == np.sign(c2)),
        ("C9 within market activity >= +2.5", c9, c9 >= CTRL_FLOOR),
    ]
    print(f"\n  ---- {label_txt}   n={ev.height:,} (IS {is_.height:,} / OOS {oos.height:,})")
    print(f"    M2 shuffle-null mean {nmean:+.2f}pp -> "
          f"{'hour leak FIXED (within +/-1pp)' if m2 else 'hour leak NOT fixed -- see net figures'}")
    for name, v, ok in crit:
        vv = f"{v:+.2f}pp" if isinstance(v, float) else f"{v}"
        print(f"    {'PASS' if ok else 'fail'}  {name:<36} {vv}")
    if not m2:
        print(f"    net of null mean:  IS {c1 - nmean:+.2f}pp   OOS {c2 - nmean:+.2f}pp")
    print("    slices: " + "  ".join(f"{v:+.1f}" for v in slices))
    print("    cells:  " + "  ".join(f"{t}{'+' if s > 0 else '-'} {v:+.1f}"
                                    for (t, s), v in sorted(cells.items())))
    print(f"    strata  C6 {[round(x, 1) for x in v6]}  C7 {[round(x, 1) for x in v7]}  "
          f"C9 {[round(x, 1) for x in v9]}")
    passed = all(c[2] for c in crit)
    print(f"    VERDICT: {'PASS' if passed else 'FAIL'}")

    fwd = []
    for (t, s), g in lab.group_by("ticker", "side"):
        top, bot = g.filter(pl.col("terc") == 2)["fwd"], g.filter(pl.col("terc") == 0)["fwd"]
        fwd.append(top.mean() - bot.mean())
    print(f"    reported: fwd spot, top-minus-bottom tercile = {np.mean(fwd):+.2f}bp")
    rates = (lab.filter(pl.col("oos")).group_by("ticker", "side", "terc")
             .agg(pl.col("y").mean()).group_by("terc").agg(pl.col("y").mean()).sort("terc"))
    print("    reported: OOS outcome rate low / mid / high buy volume = "
          + " / ".join(f"{v:.1%}" for v in rates["y"].to_list()))
    return passed


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

    base_parts, ev_rows, prof_rows, j1 = [], [], [], []
    files = sorted(glob.glob(os.path.join(NSF, "opt", "*.parquet")))
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
            tm = np.array([m.time() for m in sp["minute_et"].to_list()])
            evs = list(events(px, (tm >= T0) & (tm <= T1)))
            for arm, flt in (("0dte", pl.col("dte_days") == 0),
                             ("next", pl.col("dte_rank") == 1)):
                o = opt_day.filter((pl.col("ticker") == t) & flt)
                if o.is_empty():
                    continue
                w = windows(o, sp, day).with_columns(
                    pl.col("rax").log1p().alias("vx"), pl.col("rau").log1p().alias("vu"),
                    pl.col("rbx").log1p().alias("sx"))
                if day < SPLIT:
                    base_parts.append(w.group_by("option_type", "db", "tb").agg(
                        pl.col("vx").sum().alias("svx"), pl.col("vu").sum().alias("svu"),
                        pl.col("sx").sum().alias("ssx"), pl.len().alias("nw"))
                        .with_columns(pl.lit(t).alias("ticker"), pl.lit(arm).alias("arm")))
                sent = sentiment(w)
                act = (w.group_by("minute_et").agg((pl.col("rax") + pl.col("rbx")).sum()
                                                   .log1p().alias("act")))
                look = w.select("option_type", "strike", "minute_et", "vx", "vu", "sx",
                                "db", "tb", "dist", "rax")
                rows = []
                for test, s, k, i, y, fwd in evs:
                    K = float(k * s)
                    rows.append((test, s, K, K if test == "A" else K + s, i, y, fwd,
                                 s * (px[i] / px[max(0, i - WIN)] - 1) * 1e4, K + s))
                if not rows:
                    continue
                e = pl.DataFrame(rows, schema=["test", "side", "K", "fk", "i", "y", "fwd",
                                               "mom", "K_next"], orient="row")
                e = e.with_columns(sp["minute_et"].gather(e["i"]).alias("minute"),
                                   pl.when(pl.col("side") > 0).then(pl.lit("call"))
                                   .otherwise(pl.lit("put")).alias("option_type"))
                ej = (e.join(look.rename({"strike": "fk", "minute_et": "minute"}),
                             on=["option_type", "fk", "minute"], how="left")
                      .join(sent.rename({"minute_et": "minute"}), on="minute", how="left")
                      .join(act.rename({"minute_et": "minute"}), on="minute", how="left")
                      .with_columns(pl.lit(t).alias("ticker"), pl.lit(arm).alias("arm"),
                                    pl.lit(day).alias("date"),
                                    (pl.col("sent") * pl.col("side")).alias("sent_s")))
                ev_rows.append(ej)
                if arm == "0dte" and rng.random() < 0.15:
                    a = ej.filter((pl.col("test") == "A") & pl.col("vx").is_not_null())
                    if a.height:
                        j1.append((a.sample(1, seed=int(rng.integers(1e9))).row(0, named=True), o))
                if arm == "0dte":
                    b = e.filter(pl.col("test") == "B")
                    for tau in TAUS:
                        for which in ("K", "K_next"):
                            pj = (b.with_columns((pl.col("minute") + pl.duration(minutes=tau)).alias("m2"),
                                                 pl.col(which).alias("k2"))
                                  .join(look.rename({"strike": "k2", "minute_et": "m2"}),
                                        on=["option_type", "k2", "m2"], how="inner"))
                            prof_rows.append(pj.select("option_type", "y", "vx", "sx", "db", "tb")
                                             .with_columns(pl.lit(t).alias("ticker"),
                                                           pl.lit(tau).alias("tau"),
                                                           pl.lit(which).alias("which")))
        if (n_f + 1) % 100 == 0:
            print(f"  {n_f + 1}/{len(files)} sessions  {time.time() - t0:.0f}s", flush=True)

    base = (pl.concat(base_parts).group_by(BK)
            .agg(pl.col("svx", "svu", "ssx", "nw").sum())
            .filter(pl.col("nw") >= MIN_CELL)
            .with_columns((pl.col("svx") / pl.col("nw")).alias("bvx"),
                          (pl.col("svu") / pl.col("nw")).alias("bvu"),
                          (pl.col("ssx") / pl.col("nw")).alias("bsx")))
    ev = (pl.concat(ev_rows, how="diagonal_relaxed")
          .join(base.select(BK + ["bvx", "bvu"]), on=BK, how="left")
          .with_columns((pl.col("vx") - pl.col("bvx")).alias("res"),
                        (pl.col("vu") - pl.col("bvu")).alias("res_u"),
                        (pl.col("date") >= SPLIT).alias("oos")))
    ev.write_parquet(os.path.join(NSF, "crossing_volume_events.parquet"))
    print(f"\nbuilt {ev.height:,} event-arm rows in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------ machinery
    print("\n=== machinery checks ===")
    bad = 0
    for r, o in j1[:200]:
        m1 = r["minute"]
        raw = o.filter((pl.col("option_type") == r["option_type"]) & (pl.col("strike") == r["fk"]) &
                       (pl.col("minute_et") <= m1) &
                       (pl.col("minute_et") > m1 - dt.timedelta(minutes=WIN)) &
                       ((pl.col("multi_volume") / pl.col("volume").clip(1)) <= MSHARE))
        if abs(np.log1p(raw["ask_volume"].sum()) - r["vx"]) > 1e-9:
            bad += 1
    print(f"  J1 buy volume recomputed from raw rows: {len(j1[:200]) - bad}/{len(j1[:200])} exact")
    if bad:
        sys.exit("  J1 FAILED -- the feature join is wrong; no result is printed.")
    a0 = ev.filter((pl.col("test") == "A") & (pl.col("arm") == "0dte") & pl.col("dist").is_not_null())
    d_ok = a0.select(((pl.col("dist") > 0) & (pl.col("dist") <= ZONE + 1e-9)).mean()).item()
    print(f"  J2 spot in [K-0.25, K) at Test A events: {d_ok:.2%}  (n={a0.height:,})")
    if d_ok < 0.99:
        sys.exit("  J2 FAILED")
    e = hour_of(ev.filter((pl.col("test") == "B") & (pl.col("arm") == "0dte") & pl.col("res").is_not_null()))
    planted = e.with_columns(pl.Series("res", e["y"].to_numpy() + rng.normal(0, 0.5, e.height)))
    pc = cut_table(planted.filter(~pl.col("oos")), "res")
    pe = pooled(label(planted.filter(pl.col("oos")), pc, "res"))[0]
    pure = e.with_columns(pl.Series("res", rng.normal(0, 1, e.height)))
    qc = cut_table(pure.filter(~pl.col("oos")), "res")
    pn = pooled(label(pure.filter(pl.col("oos")), qc, "res"))[0]
    nul = shuffle_null(pure.filter(pl.col("oos")), qc, "res", 200)
    print(f"  M1 planted -> {pe:+.1f}pp (need >= +20)   noise -> {pn:+.2f}pp "
          f"(null p5..p95 {np.quantile(nul, .05):+.2f}..{np.quantile(nul, .95):+.2f})")
    if pe < 20:
        sys.exit("  M1 FAILED")

    print("\n=== coverage ===")
    print(ev.group_by("test", "arm").agg(pl.len().alias("events"),
          pl.col("res").is_not_null().mean().round(3).alias("has_residual"),
          (pl.col("rax") == 0).mean().round(3).alias("zero_buy_volume")).sort("test", "arm"))

    for test, name in (("A", "TEST A  approach -> arrival  (buy volume at K)"),
                       ("B", "TEST B  first crossing -> next strike  (buy volume at K+1)")):
        print(f"\n\n==================== {name}")
        for arm in ("0dte", "next"):
            score(ev.filter((pl.col("test") == test) & (pl.col("arm") == arm)), f"{arm.upper()} arm")

    print("\n\n==================== PICTURE D  (0DTE Test B crossings; residual log volume)")
    pr = (pl.concat(prof_rows).join(base.filter(pl.col("arm") == "0dte")
                                    .select("ticker", "option_type", "db", "tb", "bvx", "bsx"),
                                    on=["ticker", "option_type", "db", "tb"], how="left")
          .with_columns((pl.col("vx") - pl.col("bvx")).alias("buy"),
                        (pl.col("sx") - pl.col("bsx")).alias("sell")))
    tab = pr.group_by("which", "y", "tau").agg(pl.col("buy").mean(), pl.col("sell").mean()).sort("tau")
    for which, lab in (("K", "strike just crossed (K)"), ("K_next", "next strike (K+1)")):
        print(f"\n  {lab}   (log-volume residual; +0.10 ~ 10% more than normal)")
        print("    tau(min)            " + "".join(f"{t:>7}" for t in TAUS))
        for y, yl in ((1, "continued"), (0, "fell back")):
            r = tab.filter((pl.col("which") == which) & (pl.col("y") == y))
            print(f"    {yl:<10} BUY volume " + "".join(f"{v:>+7.3f}" for v in r["buy"].to_list()))
            print(f"    {yl:<10} SELL volume" + "".join(f"{v:>+7.3f}" for v in r["sell"].to_list()))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
