"""
check_big_sweep_lead.py
=======================
DOES LARGE SWEEP MONEY NEAR THE MONEY LEAD SPOT -- OR MARK THE END OF A RUN?

WHERE THIS COMES FROM
    check_sweep_lead (2026-09-27) failed both ways, and its counts explained
    why the question was never really asked: the ISO flag marks how an order
    was ROUTED, and nearly every "sweep" was small (median trigger premium
    $132 IWM .. $1,530 SPY). It closed by saying a test of genuinely large
    money needs a size floor chosen before looking, as a NEW pre-registration.
    This is that test. Nothing here was tuned on outcomes: the only things
    looked at before writing it were counts and premium sizes
    (scratchpad census, 2026-09-27), never a price after a trade.

THE FLOOR -- SCALED BY EACH TICKER'S OWN ACTIVITY (operator, 2026-09-27)
    An absolute $100k is large for SPY and almost never happens on IWM (~75
    near-money parents in three years). So the floor is SPY's $100k expressed
    as a FRACTION of SPY's own trailing activity, and that fraction is applied
    to every ticker, SPY included:
        floor(T, day) = F x avg20(T, day)
        avg20 = mean near-dated premium over the 20 sessions BEFORE `day`
                (single-leg, not canceled, same-day + next listed expiry, all
                strikes, both sides, 09:30-16:00) -- no look-ahead
        F     = median over IS days of 100,000 / avg20(SPY) = 2.376e-4
    Medians that implies, IS / OOS: SPY $100k / $192k, QQQ $56k / $128k,
    IWM $5k / $9k. The floor RISES through the sample because near-dated
    premium roughly doubled; "large" means large for that ticker at that time.

THE TRIGGER
    Parent order = child prints on ONE contract, same side, chained < 1s apart
    (a $100k+ sweep is typically 7 prints across 4 exchanges). Children must
    each be: not canceled; single-leg (no OPRA multi-leg or tied-to-stock
    code); an intermarket sweep (report_flags intermarket_sweep or code isoi);
    tagged ask_side (bought). The parent qualifies if:
        premium >= floor(T, day)
        expiry = same day or the next listed expiry
        |strike / underlying_price - 1| <= 0.5%   (trade-stamped spot; T1 in
                                                   check_sweep_lead: == 1m close)
        printed 10:01-15:30 ET (10:01 = first minute with a full 30-minute
        run behind it; see MOMENTUM)
    Direction: call bought = +1, put bought = -1. ITM, ATM and OTM all count:
    big near-the-money money is mostly delta, and the question is direction.
    One trigger per (ticker, direction) per 15 minutes; its SIZE is the summed
    premium of qualifying parents in the trigger's own minute.

THE OUTCOME
    Reference = the close of the trigger's own minute (the first close after
    the trade -- what someone watching the tape could act on).
    r_H = direction x (close[i+H] - close[i]) / close[i], in bp.
    SCORED at H = 5, 10 and 15 minutes. REPORTED at 30 and 60.

MOMENTUM, AND WHY EVERY COMPARISON IS MATCHED ON IT
    m30 = direction x the signed move over the 30 minutes ending at the last
    close BEFORE the trade (i-1 back to i-31), in bp. Big money arrives after
    runs; without matching, H- would score "runs end", which needs no sweep.
    Terciles of m30 per ticker x direction, cut on IS triggers.

THE CONTROL -- MATCHED MOMENTS WITHOUT BIG MONEY
    Up to 5 minutes on the SAME day, ticker and direction, same clock hour,
    same m30 tercile, >= 15 minutes from the trigger, and with NO qualifying-
    band parent of at least HALF the floor, in EITHER direction, in the 15
    minutes up to and including it. Small sweeps are allowed -- they run almost
    continuously (check_sweep_lead) and the claim is about big money.
    Same outcome rule. effect = mean over matched sets of (trigger r - mean of
    THAT set's controls) -- paired within the set; see check_sweep_lead.effect
    for the +3.27pp bias pooling caused. Per ticker x direction cell (>= 30
    sets), POOLED = equal-weight mean of the cells.

TWO HYPOTHESES, PRE-REGISTERED TOGETHER, SCORED SEPARATELY
    H+  LEAD        all triggers; big money precedes a move its way
    H-  EXHAUSTION  LATE triggers only -- m30 in the TOP tercile, i.e. the
                    money arrived after a run in its own direction. It passes
                    if those runs continue LESS than same-size runs without it.
                    The operator's prior: "large money that shows up late is
                    often a sign that the run has stopped."
    "Both results are usable, just not interchangeable" -- so neither can pass
    by the other failing; each has its own complete set of criteria.

PRE-COMMITTED CRITERIA, AT EACH SCORED HORIZON H in {5, 10, 15}
                                      H+ LEAD               H- EXHAUSTION
    C1  IS pooled effect              >= +3bp               <= -3bp
    C2  OOS pooled effect             >= +3bp               <= -3bp
    C3  6 equal-count calendar slices >= 5 positive         >= 5 negative
    C4  6 ticker x direction cells    >= 5 positive         >= 5 negative
        (a cell with < 30 sets counts as NOT agreeing)
    C5  OOS vs matched-set permutation > p95                < p5
    C6  persistence: pooled 30m effect >= +1.5bp            <= -1.5bp
        (same number at every H -- a 5m move that is gone by 30m is impact,
        not information)
    C7  SIZE: triggers >= 2.5 x floor effect(big) >         effect(big) <
        vs the rest                   effect(rest)          effect(rest)
    PASS at H = all seven. IS/OOS split 2025-08-21. C5's null reassigns which
    member of each matched set is the "trigger", 1000 times, OOS only.
    3bp: an ATM same-day option has delta ~0.5, so 3bp of SPY is ~7% of a
    midday premium -- the smallest move worth trading, at any horizon.
    MULTIPLICITY: three nested, correlated horizons x two hypotheses. A
    hypothesis is reported as passing at the horizons where all seven hold;
    a pass at one horizon only is flagged as such, not rounded up.

REPORTED, NOT SCORED
    - pooled effect and raw trigger/control means at 5, 10, 15, 30, 60m
    - same-day vs next-expiry triggers, split
    - counts at every filter stage, floors used, sets per cell

MACHINERY CHECKS, BEFORE ANY RESULT
    M1  planted: +10bp added to every trigger's r must score +10 +/- 0.5 at
        every horizon; shuffled-label null mean must sit within +/-0.5bp.
    M2  floors recomputed from _nsf_cache/near_prem_daily.parquet must give
        F within 1% of the pinned constant.

DATA WINDOW
    trades-core 2023-10-26..2026-09-25 (rolling vendor floor); the first 20
    sessions only feed avg20. IWM from 2024-04-19 (daily expiries).
    historical/{T}.parquet refreshed through 2026-09-25 on 2026-09-27; days
    with < 300 RTH minutes (e.g. SPY 2026-09-02, a partial file) are skipped.

RESULT -- 2026-09-27. ALL SIX VERDICTS FAIL. H- AT 15m MISSES ON C1 ALONE.
Machinery: M2 F recomputed 2.3763e-4 (pinned 2.376e-4). M1 planted +10bp ->
+10.00 at every horizon; shuffled-label null means -0.016 / +0.014 / -0.026bp.
Counts (read before any outcome, --counts): 13,267 parents over the floor ->
7,146 triggers (2,790 outside 10:01-15:30, mostly the open; 2,073 deduped).
Matched: 4,065 sets for H+, 1,191 late sets for H-; every cell >= 110.
IWM gives the most triggers -- its big orders are fat-tailed relative to its
own average. SPY OOS is thin (~120 sets) because its floor doubled.

    H+ LEAD (all triggers) -- nothing at any horizon
                  5m      10m     15m          reported 30m -1.98  60m -1.12
    IS pooled   +0.22   +0.11   -0.39
    OOS pooled  +0.01   +0.02   -0.98
    Big money near the money does not precede a move its way. 5-10m is flat
    to within a quarter of a bp.

    H- EXHAUSTION (late triggers: m30 top tercile)
                  5m      10m     15m          reported 30m -4.59  60m -1.56
    IS pooled   +0.03   -0.62   -2.16  (floor -3: C1 FAILS)
    OOS pooled  -0.17   -1.13   -3.23  (C2 passes)
    at 15m: C3 0/6 slices positive, C4 5/6 cells negative (SPY+ +0.3 the
    exception), C5 OOS -3.23 < null p5 -1.93, C6 30m -4.59, C7 big (>= 2.5x
    floor) -5.14 vs rest -2.33 -- six of seven hold; IS is 0.84bp short.

    READING. Pre-registered, this is a FAIL and is recorded as one. What it
    does show, consistently, is the operator's prior in its SHAPE: after a run,
    big bought sweeps in the run's direction are followed by the run
    continuing LESS than the same-size run without them -- nothing at 5m,
    building through 10 and 15m to -4.6bp at 30m (a horizon only the
    persistence check scored), strongest for the biggest money, and present
    in every calendar slice. It is a stall, not a lead, and the size of it
    (~3-5bp; an ATM same-day option at delta 0.5 makes roughly 7-10% of a
    midday premium on that) sits right at the tradeable floor.
    Not a rescue: 30m as a primary horizon, or a lower floor, would be
    choosing after seeing. The honest next step is a forward test of H- at
    15m and 30m on sessions after 2026-09-25, written down before they exist.
    Same-day triggers carry it (15m -3.27, 30m -4.51); next-expiry is weaker
    at 15m (-0.45) and flips by 60m.

Usage:  python check_big_sweep_lead.py [--counts]
        --counts   build triggers and controls, print counts only, no outcomes
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys
import time

import numpy as np
import polars as pl

from check_sweep_lead import load_spot, MULTI_TIED, CORE, OUT, TICKERS, START, SPLIT

F_PINNED = 2.376e-4
BAND, SEP, K_CTRL, DEDUP, QUIET, MOM = 0.005, 15, 5, 15, 15, 30
HS_SCORED, HS_REPORTED = (5, 10, 15), (5, 10, 15, 30, 60)
T0, T1 = dt.time(10, 1), dt.time(15, 30)
FLOOR_BP, PERSIST_BP, BIG_X, N_PERM, MIN_SETS = 3.0, 1.5, 2.5, 1000, 30
rng = np.random.default_rng(20260928)


# ---------------------------------------------------------------- floors
def load_floors():
    d = pl.read_parquet(os.path.join(OUT, "near_prem_daily.parquet"))
    s = d.filter((pl.col("ticker") == "SPY") & (pl.col("date") < SPLIT)).drop_nulls("avg20")
    f = float((100_000 / s["avg20"]).median())
    print(f"=== M2 floor fraction: recomputed F = {f:.4e}, pinned {F_PINNED:.4e}")
    if abs(f / F_PINNED - 1) > 0.01:
        sys.exit("  M2 FAILED -- the denominator changed since pre-registration.")
    return {(t, dd): a * F_PINNED
            for t, dd, a in d.drop_nulls("avg20").select("ticker", "date", "avg20").iter_rows()}


# ---------------------------------------------------------------- parents
def parents_for_day(path: str, day: dt.date) -> pl.DataFrame:
    """Ask-side single-leg ISO parents on same-day / next-listed expiry."""
    lf = pl.scan_parquet(path)
    nx = (lf.filter(pl.col("expiry") > day).group_by("underlying_symbol")
          .agg(pl.col("expiry").min().alias("nx")))
    return (lf.filter(pl.col("underlying_symbol").is_in(TICKERS)
                      & (pl.col("canceled").fill_null("f") != "t")
                      & ~pl.col("upstream_condition_detail").str.to_lowercase().is_in(list(MULTI_TIED))
                      & (pl.col("report_flags").str.contains("intermarket_sweep")
                         | (pl.col("upstream_condition_detail").str.to_lowercase() == "isoi"))
                      & pl.col("tags").str.contains("ask_side"))
            .join(nx, on="underlying_symbol", how="left")
            .filter((pl.col("expiry") == day) | (pl.col("expiry") == pl.col("nx")))
            .with_columns(pl.col("executed_at").dt.convert_time_zone("America/New_York").alias("ts"),
                          (pl.col("size") * pl.col("price") * 100).alias("prem"))
            .sort("underlying_symbol", "option_type", "strike", "expiry", "ts")
            .with_columns(
                ((pl.col("ts").diff().dt.total_microseconds() > 1_000_000)
                 | (pl.col("strike") != pl.col("strike").shift())
                 | (pl.col("expiry") != pl.col("expiry").shift())
                 | (pl.col("option_type") != pl.col("option_type").shift())
                 | (pl.col("underlying_symbol") != pl.col("underlying_symbol").shift()))
                .fill_null(True).cum_sum().alias("pid"))
            .group_by("pid")
            .agg(pl.col("underlying_symbol").first().alias("ticker"),
                 pl.col("option_type").first(), pl.col("strike").first(),
                 pl.col("expiry").first(), pl.col("ts").first(),
                 pl.col("underlying_price").first().alias("u"),
                 pl.col("prem").sum())
            .filter(((pl.col("strike") / pl.col("u") - 1).abs() <= BAND))
            .with_columns(pl.col("ts").dt.truncate("1m").alias("m"),
                          (pl.col("expiry") == day).alias("same_day"),
                          pl.when(pl.col("option_type") == "call").then(1).otherwise(-1).alias("side"))
            .collect())


def build_day(t, day, sp, par, floor, counts, trig_rows, ctrl_rows):
    px = sp["close"].to_numpy()
    mins = sp["minute_et"].to_list()
    idx = {m: i for i, m in enumerate(mins)}
    tm = np.array([m.time() for m in mins])
    n = len(px)
    ok = (tm >= T0) & (tm <= T1)
    p = par.filter(pl.col("ticker") == t)
    # quiet exclusion: minutes with a parent >= half the floor, EITHER direction
    loud = np.zeros(n, dtype=bool)
    for m in p.filter(pl.col("prem") >= 0.5 * floor)["m"].to_list():
        if m in idx:
            loud[idx[m]] = True
    loud_cum = np.r_[0, np.cumsum(loud)]

    def quiet(i):              # no loud minute in [i-QUIET, i]
        return loud_cum[i + 1] - loud_cum[max(0, i - QUIET)] == 0

    def rets(q, i):
        return [((q[i + h] - q[i]) / abs(q[i]) * 1e4) if i + h < n else np.nan
                for h in HS_REPORTED]

    q_counts = p.filter(pl.col("prem") >= floor)
    counts["parents_over_floor"] += q_counts.height
    for side in (1, -1):
        q = px * side
        big = (q_counts.filter(pl.col("side") == side)
               .group_by("m").agg(pl.col("prem").sum(), pl.col("same_day").any())
               .sort("m"))
        last = -10 ** 9
        for m, prem, sd in big.iter_rows():
            i = idx.get(m)
            if i is None:
                counts["no_price_minute"] += 1
                continue
            if not ok[i] or i < MOM + 1 or i + max(HS_SCORED) >= n:
                counts["outside_window"] += 1
                continue
            if i - last < DEDUP:
                counts["dedup_dropped"] += 1
                continue
            last = i
            m30 = (q[i - 1] - q[i - 1 - MOM]) / abs(q[i - 1 - MOM]) * 1e4
            trig_rows.append((t, day, side, i, mins[i].hour, float(prem), float(prem) / floor,
                              bool(sd), m30, *rets(q, i)))
        for i in range(MOM + 1, n - max(HS_SCORED)):
            if not ok[i] or not quiet(i):
                continue
            m30 = (q[i - 1] - q[i - 1 - MOM]) / abs(q[i - 1 - MOM]) * 1e4
            ctrl_rows.append((t, day, side, i, mins[i].hour, m30, *rets(q, i)))


# ---------------------------------------------------------------- statistics
RCOLS = [f"r{h}" for h in HS_REPORTED]


def match(trig: pl.DataFrame, ctrl: pl.DataFrame) -> pl.DataFrame:
    """Long form: one row per set member, role 1 trigger / 0 control."""
    keys = ["ticker", "date", "side", "hr", "mt"]
    j = trig.select(keys + ["tid", "i"]).join(
        ctrl.select(keys + ["i"] + RCOLS).rename({"i": "ci"}), on=keys, how="inner")
    j = j.filter((pl.col("ci") - pl.col("i")).abs() >= SEP)
    j = j.with_columns(pl.int_range(pl.len()).shuffle(seed=int(rng.integers(1e9)))
                       .over("tid").alias("_r")).filter(pl.col("_r") < K_CTRL)
    ctl = j.select("tid", pl.lit(0, pl.Int8).alias("role"), *RCOLS)
    have = j["tid"].unique().implode()
    tr = trig.filter(pl.col("tid").is_in(have)).select("tid", pl.lit(1, pl.Int8).alias("role"), *RCOLS)
    meta = trig.select("tid", "ticker", "side", "date", "oos", "big", "late", "same_day", "_sl")
    return pl.concat([tr, ctl]).join(meta, on="tid").sort("tid", "role", descending=[False, True])


def effect(long: pl.DataFrame, col: str, role=None) -> tuple[float, dict]:
    """Paired within-set effect (bp), per cell with >= MIN_SETS sets, pooled
    as the equal-weight mean of cells. `role` overrides the role column (the
    permutation null)."""
    if long.is_empty():
        return np.nan, {}
    tid = long["tid"].to_numpy()
    r = long[col].to_numpy().astype(float)
    ro = long["role"].to_numpy() if role is None else role
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(tid)]))
    valid = ~np.isnan(r)
    rt = np.bincount(grp, weights=np.where(valid & (ro == 1), r, 0), minlength=len(starts))
    nt = np.bincount(grp, weights=(valid & (ro == 1)), minlength=len(starts))
    rc = np.bincount(grp, weights=np.where(valid & (ro == 0), r, 0), minlength=len(starts))
    nc = np.bincount(grp, weights=(valid & (ro == 0)), minlength=len(starts))
    good = (nt == 1) & (nc > 0)
    d = np.where(good, rt - rc / np.maximum(nc, 1), np.nan)
    cell = (long["ticker"].to_numpy()[starts], long["side"].to_numpy()[starts])
    per = {}
    for key in set(zip(*cell)):
        sel = (cell[0] == key[0]) & (cell[1] == key[1]) & good
        if sel.sum() >= MIN_SETS:
            per[key] = float(np.nanmean(d[sel]))
    v = list(per.values())
    return (float(np.mean(v)) if v else np.nan), per


def perm_null(long: pl.DataFrame, col: str, n=N_PERM) -> np.ndarray:
    tid = long["tid"].to_numpy()
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    sizes = np.diff(np.r_[starts, len(tid)])
    out = np.empty(n)
    for k in range(n):
        role = np.zeros(len(tid), dtype=np.int8)
        role[starts + (rng.random(len(starts)) * sizes).astype(int)] = 1
        out[k] = effect(long, col, role)[0]
    return out


def score(lm: pl.DataFrame, name: str):
    print(f"\n  ---- {name}")
    n_sets = lm.filter(pl.col("role") == 1)
    print(f"    matched sets {n_sets.height:,} (IS {n_sets.filter(~pl.col('oos')).height:,} / "
          f"OOS {n_sets.filter(pl.col('oos')).height:,}), controls "
          f"{lm.filter(pl.col('role') == 0).height:,}")
    print("    sets per cell: " + "  ".join(
        f"{t}{'+' if s > 0 else '-'} {g.height}" for (t, s), g in
        sorted(n_sets.group_by("ticker", "side"), key=lambda kv: kv[0])))
    c6, _ = effect(lm, "r30")
    verdicts = {}
    for h in HS_SCORED:
        col = f"r{h}"
        is_, oos = lm.filter(~pl.col("oos")), lm.filter(pl.col("oos"))
        c1, _ = effect(is_, col)
        c2, _ = effect(oos, col)
        slices = [effect(lm.filter(pl.col("_sl") == k), col)[0] for k in range(6)]
        _, cells = effect(lm, col)
        null = perm_null(oos, col)
        p5, p95 = np.quantile(null, .05), np.quantile(null, .95)
        top, _ = effect(lm.filter(pl.col("big")), col)
        rest, _ = effect(lm.filter(~pl.col("big")), col)
        npos = sum(v > 0 for v in slices); nneg = sum(v < 0 for v in slices)
        cpos = sum(v > 0 for v in cells.values()); cneg = sum(v < 0 for v in cells.values())
        crit = [
            ("C1 IS pooled", c1, c1 >= FLOOR_BP, c1 <= -FLOOR_BP),
            ("C2 OOS pooled", c2, c2 >= FLOOR_BP, c2 <= -FLOOR_BP),
            ("C3 calendar slices (pos/neg)", f"{npos}/{nneg}", npos >= 5, nneg >= 5),
            (f"C4 cells (pos/neg of {len(cells)} scored)", f"{cpos}/{cneg}", cpos >= 5, cneg >= 5),
            (f"C5 OOS vs null p5..p95 {p5:+.2f}..{p95:+.2f}", c2, c2 > p95, c2 < p5),
            ("C6 persistence, pooled 30m", c6, c6 >= PERSIST_BP, c6 <= -PERSIST_BP),
            (f"C7 size: big {top:+.2f} vs rest {rest:+.2f}", top - rest, top > rest, top < rest),
        ]
        print(f"\n    H = {h}m {'':30}{'value':>10}   H+ LEAD   H- EXHAUST")
        for nm, v, lp, ln in crit:
            vv = f"{v:+.2f}" if isinstance(v, float) else v
            print(f"    {nm:<44}{vv:>10}   {'PASS' if lp else 'fail':<8}  {'PASS' if ln else 'fail'}")
        print("    slices: " + "  ".join(f"{v:+.1f}" for v in slices))
        print("    cells:  " + "  ".join(f"{t}{'+' if s > 0 else '-'} {v:+.1f}"
                                        for (t, s), v in sorted(cells.items())))
        verdicts[h] = (all(c[2] for c in crit), all(c[3] for c in crit))
    print("\n    reported, all sets:  " + "  ".join(
        f"{h}m {effect(lm, f'r{h}')[0]:+.2f}" for h in HS_REPORTED))
    for sd, lab in ((True, "same-day"), (False, "next-expiry")):
        sub = lm.filter(pl.col("same_day") == sd)
        print(f"    reported, {lab:<11} " + "  ".join(
            f"{h}m {effect(sub, f'r{h}')[0]:+.2f}" for h in HS_REPORTED))
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    floors = load_floors()
    spot = load_spot()
    counts = {k: 0 for k in ("parents_over_floor", "no_price_minute", "outside_window",
                             "dedup_dropped")}
    trig_rows, ctrl_rows = [], []
    files = sorted(glob.glob(os.path.join(CORE, "date=*", "trades.parquet")))
    for n_f, f in enumerate(files):
        day = dt.date.fromisoformat(f.split("date=")[1][:10])
        todo = [t for t in TICKERS if day >= START[t] and (t, day) in floors
                and day in spot[t] and spot[t][day].height >= 300]
        if not todo:
            continue
        par = parents_for_day(f, day)
        for t in todo:
            build_day(t, day, spot[t][day], par, floors[(t, day)], counts, trig_rows, ctrl_rows)
        if (n_f + 1) % 100 == 0:
            print(f"  {n_f + 1}/{len(files)} days  {time.time() - t0:.0f}s", flush=True)

    trig = pl.DataFrame(trig_rows, orient="row", schema=[
        "ticker", "date", "side", "i", "hr", "prem", "xfloor", "same_day", "m30", *RCOLS])
    ctrl = pl.DataFrame(ctrl_rows, orient="row", schema=[
        "ticker", "date", "side", "i", "hr", "m30", *RCOLS])
    trig = trig.with_columns(pl.int_range(pl.len()).alias("tid"),
                             (pl.col("date") >= SPLIT).alias("oos"),
                             (pl.col("xfloor") >= BIG_X).alias("big"))
    cuts = (trig.filter(~pl.col("oos")).group_by("ticker", "side")
            .agg(pl.col("m30").quantile(1 / 3).alias("m1"), pl.col("m30").quantile(2 / 3).alias("m2")))

    def mt(df):
        return df.join(cuts, on=["ticker", "side"]).with_columns(
            pl.when(pl.col("m30") <= pl.col("m1")).then(0)
            .when(pl.col("m30") <= pl.col("m2")).then(1).otherwise(2).alias("mt")).drop("m1", "m2")
    trig, ctrl = mt(trig), mt(ctrl)
    trig = trig.with_columns((pl.col("mt") == 2).alias("late"))
    days = np.array(sorted(trig["date"].unique().to_list()), dtype="datetime64[D]")
    edges = np.array([days[int(len(days) * k / 6)] for k in range(1, 6)])
    trig = trig.with_columns(pl.Series("_sl", np.searchsorted(
        edges, trig["date"].to_numpy().astype("datetime64[D]"), side="right")))
    trig.write_parquet(os.path.join(OUT, "big_sweep_triggers.parquet"))

    print("\n=== counts ===")
    for k, v in counts.items():
        print(f"  {k:<22} {v:>9,}")
    print(f"  {'triggers':<22} {trig.height:>9,}   control minutes {ctrl.height:,}")
    print(trig.group_by("ticker", "side").agg(
        pl.len().alias("n"), pl.col("oos").sum().alias("oos"), pl.col("late").sum().alias("late"),
        pl.col("big").sum().alias("big"), pl.col("same_day").mean().round(2).alias("same_day"),
        pl.col("prem").median().round(0).alias("med_prem")).sort("ticker", "side"))

    lm_all = match(trig, ctrl)
    lm_late = match(trig.filter(pl.col("late")), ctrl)
    for nm, lm in (("all triggers (H+)", lm_all), ("late triggers (H-)", lm_late)):
        s = lm.filter(pl.col("role") == 1)
        print(f"  matched {nm}: {s.height:,} sets  " + "  ".join(
            f"{t}{'+' if sd > 0 else '-'} {g.height}"
            for (t, sd), g in sorted(s.group_by("ticker", "side"), key=lambda kv: kv[0])))
    if a.counts:
        print(f"\n--counts: stopping before any outcome.  {time.time() - t0:.0f}s")
        return

    # ---------------------------------------------------------------- M1
    print("\n=== M1 machinery ===")
    ok = True
    for h in HS_SCORED:
        col = f"r{h}"
        planted = lm_all.with_columns(pl.when(pl.col("role") == 1).then(pl.col(col) + 10.0)
                                      .otherwise(pl.col(col)).alias(col))
        pe = effect(planted, col)[0] - effect(lm_all, col)[0]
        nul = perm_null(lm_all.filter(pl.col("oos")), col, n=200)
        print(f"  {h:>2}m  planted +10bp -> {pe:+.2f} (need 10 +/- 0.5)   "
              f"shuffled-label null mean {nul.mean():+.3f}bp (need within +/-0.5)")
        ok &= abs(pe - 10) <= 0.5 and abs(nul.mean()) <= 0.5
    if not ok:
        sys.exit("  M1 FAILED -- the statistic or its null is broken; no verdict printed.")

    print("\n\n==================== BIG ASK-SIDE SWEEPS, +/-0.5% OF SPOT, SAME-DAY / NEXT EXPIRY")
    v_all = score(lm_all, "H+ LEAD -- all triggers")
    v_late = score(lm_late, "H- EXHAUSTION -- late triggers (m30 top tercile)")
    print("\n==================== VERDICTS")
    for h in HS_SCORED:
        print(f"  {h:>2}m   H+ LEAD: {'PASS' if v_all[h][0] else 'FAIL'}    "
              f"H- EXHAUSTION: {'PASS' if v_late[h][1] else 'FAIL'}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
