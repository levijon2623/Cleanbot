"""
check_sweep_lead.py
===================
DO ASK-SIDE SWEEPS AT THE NEXT OTM STRIKE LEAD SPOT THERE -- OR MARK THE END
OF THE RUN?

WHERE THIS COMES FROM
    check_strike_crossing_flow and check_strike_crossing_volume (2026-09-26)
    found that neither the buy/sell balance nor buy volume at 10-minute
    resolution says whether spot reaches a strike: flow follows price. The one
    version of the operator's front-running idea still standing is the fast,
    informed one -- intermarket sweeps, which the minute table cannot see.
    silver/trades-core (built 2026-09-26, every checkable day matching
    option-contracts-1m volume exactly) can.

TWO HYPOTHESES, PRE-REGISTERED TOGETHER, SCORED SEPARATELY
    H+  LEAD        a sweep at the next strike precedes spot reaching it
    H-  EXHAUSTION  large money that shows up late marks the end of the run
    The operator's prior (2026-09-27) leans to H-, as did this project's earlier
    large-cap work (big OTM-call UOA reached its strike ~5pp LESS than a
    vol-matched baseline). "Both results are usable, just not interchangeable":
    H+ says join the move, H- says exit or fade it. So each has its own full set
    of mirrored criteria and a verdict is only ever issued for the one whose
    every criterion holds. Neither can pass by being the other's failure.

THE TRIGGER
    A trade that is ALL of: not canceled; single-leg (no OPRA multi-leg or
    tied-to-stock code); an intermarket sweep (report_flags intermarket_sweep,
    or code isoi); tagged ask_side (bought); same-day expiry; OTM by at most
    $1.00 at the last completed minute's close (calls above spot = side +1,
    puts below spot = side -1); printed 09:45-15:30 ET.
    One trigger per (ticker, side, strike) per 15 minutes -- sweeps arrive as
    bursts of child prints across exchanges, and counting each would count one
    decision dozens of times. The trigger's size is the total ask-side sweep
    premium at that strike in its first minute.

THE OUTCOME
    Reference level = the close of the trigger's own minute, i.e. the first
    close AFTER the trade. y = 1 if any close in the next 15 minutes reaches
    strike + $0.10 in the trigger's direction. Triggers whose own minute
    already closed through the strike are dropped and counted: their "arrival"
    happened before any outcome clock could start.

THE CONTROL -- MATCHED QUIET MOMENTS
    "Spot reached the strike 60% of the time after a sweep" is meaningless on
    its own: a strike $0.25 away is reached ~60% of the time regardless. Each
    trigger is matched to up to 5 minutes on the SAME day, same ticker, same
    side, same clock hour, whose nearest OTM strike sits in the same $0.10
    distance bucket, with NO ask-side sweep at that strike in the prior 5
    minutes, and at least 15 minutes from the trigger (so the two outcome
    windows do not share a path). Same outcome rule. Triggers with no match
    are dropped and counted.
    effect (pp) = per ticker x side cell, mean(y | trigger) - mean(y | matched
    controls); POOLED = equal-weight mean of the 6 cells.
    MOMENTUM-MATCHED arm: controls additionally restricted to the trigger's
    tercile of 10-minute signed spot move. Late money arrives AFTER a run by
    definition, so for H- in particular the claim must be "a sweep marks the end
    of a run better than a run of the same size without one" -- or it is just
    "runs end", which needs no sweep.

PRE-COMMITTED CRITERIA   (H+ as written; H- is the exact mirror)
                                      H+ LEAD                H- EXHAUSTION
    C1  IS pooled effect              >= +5pp                <= -5pp
    C2  OOS pooled effect             >= +5pp                <= -5pp
    C3  6 equal-count calendar slices >= 5 positive          >= 5 negative
    C4  6 ticker x side cells         >= 5 positive          >= 5 negative
    C5  OOS vs matched-set permutation > p95                 < p5
    C6  momentum-matched effect       >= +2.5pp              <= -2.5pp
    C7  SIZE: top-quartile-premium    effect(top) >          effect(top) <
        triggers vs the rest          effect(rest)           effect(rest)
    PASS = all seven for that hypothesis. IS/OOS split 2025-08-21.
    C5's null swaps which member of each matched set is labelled "trigger"
    (they are exchangeable if the sweep carries nothing), 1000 times, OOS only.
    C7 exists because real information should scale with money committed;
    noise does not. Quartile cuts per ticker x side from IS triggers.

REPORTED, NOT SCORED
    - forward 15m spot move after triggers vs controls, in bp
    - SELL-SIDE MIRROR (the "then it gets dumped" half): bid-side sweeps at a
      strike spot has passed by at most $1 -- does spot fall back through it
      within 15m more than matched moments?
    - counts at every filter stage, so the reader can see what the sample is

MACHINERY CHECKS, BEFORE ANY RESULT
    T1  trade-stamped underlying_price vs the 1m close of its minute: median
        |gap| and the lag that fits best. If trades' stamps lead or lag the
        1m file, "before" and "after" are not what they claim.
    M1  planted: a trigger set whose y is forced to 1 must score >= +20pp;
        C5 with the true labels shuffled must centre within +/-1pp of zero.

DATA WINDOW
    trades-core starts 2023-10-26 (the vendor's history floor is rolling;
    see uw_options_data_lake.py). IWM starts 2024-04-19 (daily expiries).
    The 1m price files end 2026-08-21 (QQQ, IWM) and 2026-09-02 (SPY) -- the
    refresh on 2026-09-27 failed on a revoked API key -- so the sample ends
    there, and a day with no price file is skipped, not guessed.

RESULT -- 2026-09-27. BOTH FAIL. Neither lead nor sweep-specific exhaustion.
Machinery: T1 trade-stamped spot == 1m close (median gap 0.00bp, corr
0.99-1.00 at lag 0, ~0 at +/-1). M1 planted +76.5pp; shuffled-label null mean
+0.04pp after the paired-statistic fix (it was +3.27pp before -- see effect()).

    COUNTS FIRST, BECAUSE THEY CHANGE WHAT "SWEEP" MEANS
    24.1M ask-side, single-leg, same-day sweep prints on three tickers --
    ~34k a day. Median trigger premium $1,530 (SPY), $952-1,041 (QQQ), $132
    (IWM). The ISO flag marks how an order was ROUTED, not who sent it; nearly
    all of these are small. 745k of 877k candidates were dropped as repeats
    within 15 min -- the next strike is swept almost continuously.
    Consequence for matching: a "quiet" minute (no sweep at that strike in the
    prior 5 min) is rare, so only 9,427 of 117,610 triggers found a match. The
    matched sample leans toward quieter hours and distances. Stated, not fixed.

                                     value    H+ LEAD   H- EXHAUSTION
    C1 IS pooled                     -1.51     fail       fail  (floor -5)
    C2 OOS pooled                    -3.14     fail       fail  (floor -5)
    C3 slices pos/neg                  1/5     fail       PASS
    C4 cells pos/neg                   0/6     fail       PASS
    C5 OOS vs null p5..p95 -1.67..+1.69        fail       PASS
    C6 momentum-matched              -0.66     fail       fail
    C7 top quartile -1.48 vs rest -2.29        PASS       fail
    cells: IWM- -1.7 IWM+ -1.4 QQQ- -2.0 QQQ+ -0.0 SPY- -8.3 SPY+ -0.2
    fwd 15m: triggers -0.64bp vs control minutes +0.15bp

    READING. The sign is consistent -- no cell positive, 5 of 6 slices
    negative, OOS past the null's p5 -- so after a sweep at the next strike,
    spot reaches it slightly LESS often than at matched quiet moments. But:
      - it is 1.5-3pp, under the 5pp floor;
      - it collapses to -0.66 once the control is matched on the 10-minute
        move: most of it is "runs end", and the sweep is just what shows up
        after runs;
      - bigger sweeps do it LESS, not more (C7 backwards for H-), which is the
        opposite of what informed money would look like.
    Together with check_strike_crossing_flow / _volume: at the strike spot is
    walking toward, options flow -- balance, volume, or sweeps -- follows
    price. What these sweeps are is overwhelmingly small routed orders; a test
    of genuinely LARGE money would need a premium floor (e.g. >= $100k) chosen
    before looking, and is a new pre-registration, not a rescue of this one.

    SELL-SIDE MIRROR (unscored): 107k bid-side sweeps at a strike spot has
    passed by <= $1; spot fell back through it within 15m 25-37% of the time.
    The earlier printout set that beside "~0.60" from check_strike_crossing_flow
    -- that comparison is INVALID (that was a 30m window from a first crossing
    at +$0.10; this is 15m from anywhere up to $1 past). No matched control
    was built for the mirror, so it supports no conclusion either way.

Usage:  python check_sweep_lead.py
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
import time

import numpy as np
import polars as pl

CORE = os.path.join("lake", "silver", "trades-core")
OUT = "_nsf_cache"
TICKERS = ["SPY", "QQQ", "IWM"]
START = {"SPY": dt.date(2023, 10, 26), "QQQ": dt.date(2023, 10, 26), "IWM": dt.date(2024, 4, 19)}
SPLIT = dt.date(2025, 8, 21)
MULTI_TIED = {"mlet", "mlat", "mlct", "mlft", "mesl", "masl", "mfsl",
              "tlet", "tlat", "tlct", "tlft", "tesl", "tasl", "tfsl"}
MAX_OTM, BAND, H, DEDUP, QUIET, SEP, K_CTRL = 1.00, 0.10, 15, 15, 5, 15, 5
T0, T1 = dt.time(9, 45), dt.time(15, 30)
FLOOR, CTRL_FLOOR, N_PERM = 5.0, 2.5, 1000
rng = np.random.default_rng(20260927)


def load_spot():
    out = {}
    for t in TICKERS:
        h = (pl.read_parquet(f"historical/{t}.parquet").select("minute_et", "close")
             .with_columns(pl.col("minute_et").dt.convert_time_zone("America/New_York"))
             .filter(pl.col("minute_et").dt.time().is_between(dt.time(9, 30), dt.time(15, 59)))
             .drop_nulls().unique("minute_et").sort("minute_et")
             .with_columns(pl.col("minute_et").dt.date().alias("date")))
        out[t] = {d: g for (d,), g in h.group_by(["date"])}
    return out


def sweeps_for_day(path: str, day: dt.date, ticker: str, side_tag: str) -> pl.DataFrame:
    """Single-leg, same-day-expiry intermarket sweeps on one side of the book.

    🚨 FILTERED BY TICKER. The file holds all three; without this, SPY's
    sweeps are scored against QQQ's price path and the result is plausible
    nonsense rather than an error.
    """
    t = pl.read_parquet(path)
    return (t.filter((pl.col("underlying_symbol") == ticker)
                     & (pl.col("canceled").fill_null("f") != "t")
                     & (pl.col("expiry") == day)
                     & ~pl.col("upstream_condition_detail").str.to_lowercase().is_in(list(MULTI_TIED))
                     & (pl.col("report_flags").str.contains("intermarket_sweep")
                        | (pl.col("upstream_condition_detail").str.to_lowercase() == "isoi"))
                     & pl.col("tags").str.contains(side_tag))
            .with_columns(pl.col("executed_at").dt.convert_time_zone("America/New_York")
                          .alias("ts"),
                          (pl.col("size") * pl.col("price") * 100).alias("prem")))


def arrive(q: np.ndarray, i: int, k: float) -> int | None:
    """1 if a close in (i, i+H] reaches k+BAND in q-space. None if no room."""
    if i + H >= len(q):
        return None
    return int(q[i + 1:i + 1 + H].max() >= k + BAND)


def fallback(q: np.ndarray, i: int, k: float) -> int | None:
    """Sell-side mirror: 1 if a close in (i, i+H] falls back to k-BAND."""
    if i + H >= len(q):
        return None
    return int(q[i + 1:i + 1 + H].min() <= k - BAND)


def build_day(t, day, sp, core_path, counts, trig_rows, ctrl_rows, mir_rows):
    px = sp["close"].to_numpy()
    mins = sp["minute_et"]
    idx = {m: i for i, m in enumerate(mins.to_list())}
    tm = np.array([m.time() for m in mins.to_list()])
    ok = (tm >= T0) & (tm <= T1)
    buys = sweeps_for_day(core_path, day, t, "ask_side")
    sells = sweeps_for_day(core_path, day, t, "bid_side")
    counts["sweep_prints_buy"] += buys.height

    for side in (1, -1):
        q = px * side
        otype = "call" if side > 0 else "put"
        b = buys.filter(pl.col("option_type") == otype).with_columns(
            pl.col("ts").dt.truncate("1m").alias("m"))
        # minutes that had ANY ask-side sweep at a strike -> quiet-control exclusion
        swept = {}
        for m, k in b.select("m", "strike").iter_rows():
            if m in idx:
                swept.setdefault(float(k), []).append(idx[m])
        # ---------------- triggers
        last_trig = {}
        g = (b.group_by("m", "strike").agg(pl.col("prem").sum(), pl.len().alias("n"))
             .sort("m"))
        for m, k, prem, _n in g.iter_rows():
            i = idx.get(m)
            if i is None or i < 11 or not ok[i]:
                continue
            k = float(k)
            kq = k * side
            prev = q[i - 1]                               # last completed close
            d_prev = kq - prev
            if not (0 < d_prev <= MAX_OTM):
                continue
            counts["candidate"] += 1
            key = kq
            if key in last_trig and i - last_trig[key] < DEDUP:
                counts["dedup_dropped"] += 1
                continue
            last_trig[key] = i
            d_ref = kq - q[i]                              # at the trade minute's close
            if d_ref <= 0:
                counts["already_through"] += 1
                continue
            y = arrive(q, i, kq)
            if y is None:
                continue
            mom = (q[i] - q[i - 10]) / abs(q[i - 10]) * 1e4
            fwd = (q[i + H] - q[i]) / abs(q[i]) * 1e4
            trig_rows.append((t, day, side, i, kq, round(d_ref, 4), int(d_ref * 10),
                              mins[i].hour, float(prem), y, mom, fwd))
        # ---------------- control pool: every eligible minute, nearest OTM strike
        for i in range(11, len(q) - H - 1):
            if not ok[i]:
                continue
            kq = np.floor(q[i]) + 1.0 if q[i] % 1 != 0 else q[i] + 1.0
            d = kq - q[i]
            if not (0 < d <= MAX_OTM):
                continue
            ks = kq * side
            if any(i - QUIET <= j <= i for j in swept.get(float(ks), [])):
                continue
            ctrl_rows.append((t, day, side, i, kq, int(d * 10), mins[i].hour,
                              arrive(q, i, kq),
                              (q[i] - q[i - 10]) / abs(q[i - 10]) * 1e4,
                              (q[i + H] - q[i]) / abs(q[i]) * 1e4))
        # ---------------- sell-side mirror: bid sweeps at a strike just passed
        s = (sells.filter(pl.col("option_type") == otype)
             .with_columns(pl.col("ts").dt.truncate("1m").alias("m"))
             .group_by("m", "strike").agg(pl.col("prem").sum()).sort("m"))
        last_m = {}
        for m, k, prem in s.iter_rows():
            i = idx.get(m)
            if i is None or i < 11 or not ok[i]:
                continue
            kq = float(k) * side
            d_itm = q[i - 1] - kq
            if not (0 < d_itm <= MAX_OTM):
                continue
            if kq in last_m and i - last_m[kq] < DEDUP:
                continue
            last_m[kq] = i
            if q[i] - kq <= 0:
                continue
            yb = fallback(q, i, kq)
            if yb is not None:
                mir_rows.append((t, day, side, i, kq, int((q[i] - kq) * 10), mins[i].hour, yb))


# ---------------------------------------------------------------- statistics
def match(trig: pl.DataFrame, ctrl: pl.DataFrame, momentum=False) -> pl.DataFrame:
    """Attach up to K_CTRL matched controls to each trigger; returns long form
    with a set id, a role (1 trigger / 0 control) and y."""
    keys = ["ticker", "date", "side", "hr", "db"]
    c = ctrl
    if momentum:
        keys = keys + ["mt"]
    j = trig.select(keys + ["tid", "i"]).join(
        c.select(keys + ["i", "y"]).rename({"i": "ci"}), on=keys, how="inner")
    j = j.filter((pl.col("ci") - pl.col("i")).abs() >= SEP)
    j = j.with_columns(pl.int_range(pl.len()).shuffle(seed=int(rng.integers(1e9)))
                       .over("tid").alias("_r")).filter(pl.col("_r") < K_CTRL)
    # explicit dtypes: a planted `pl.lit(1)` y is Int32 and the real one Int64,
    # and vstack refuses to guess
    ctl = j.select("tid", pl.lit(0, pl.Int8).alias("role"), pl.col("y").cast(pl.Int64))
    have = j["tid"].unique().implode()
    tr = trig.filter(pl.col("tid").is_in(have)).select(
        "tid", pl.lit(1, pl.Int8).alias("role"), pl.col("y").cast(pl.Int64))
    meta = trig.select("tid", "ticker", "side", "date", "oos", "big")
    return pl.concat([tr, ctl]).join(meta, on="tid")


def effect(long: pl.DataFrame) -> tuple[float, dict]:
    """Mean over matched sets of (trigger y - mean of THAT set's controls).

    🚨 PAIRED WITHIN THE SET, NOT POOLED. The first version compared the mean
    over triggers (one per set) with the mean over ALL controls pooled. Sets
    carry 1..5 controls, and how many depends on how common that distance x
    hour combination is -- which also moves the arrival rate. Pooling weights
    controls toward the big sets, so the two sides were averaged over
    different populations. M1 caught it before any verdict was printed: with
    trigger labels shuffled at random the "effect" centred at +3.27pp, not 0,
    and that bias would have sat inside the real statistic too.
    Differencing within each set makes the relabelled expectation exactly zero
    for any set size: a random pick and the mean of the rest have the same
    expectation. Corrected 2026-09-27, before any result had been seen.
    """
    d = (long.group_by("tid", "ticker", "side")
         .agg((pl.col("y").filter(pl.col("role") == 1).mean()
               - pl.col("y").filter(pl.col("role") == 0).mean()).alias("d"))
         .drop_nulls("d"))
    per = {}
    for (t, s), g in d.group_by("ticker", "side"):
        if g.height >= 30:
            per[(t, s)] = float(g["d"].mean()) * 100
    v = list(per.values())
    return (float(np.mean(v)) if v else np.nan), per


def perm_null(long: pl.DataFrame, n=N_PERM) -> np.ndarray:
    """Swap which member of each matched set is the 'trigger'."""
    base = long.sort("tid")
    tid = base["tid"].to_numpy()
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    sizes = np.diff(np.r_[starts, len(tid)])
    out = np.empty(n)
    for r in range(n):
        role = np.zeros(len(tid), dtype=np.int8)
        role[starts + (rng.random(len(starts)) * sizes).astype(int)] = 1
        out[r] = effect(base.with_columns(pl.Series("role", role)))[0]
    return out


def score(trig, ctrl, name):
    lm = match(trig, ctrl)
    lmm = match(trig, ctrl, momentum=True)
    is_, oos = lm.filter(~pl.col("oos")), lm.filter(pl.col("oos"))
    c1, _ = effect(is_)
    c2, _ = effect(oos)
    days = np.array(sorted(trig["date"].unique().to_list()), dtype="datetime64[D]")
    edges = np.array([days[int(len(days) * k / 6)] for k in range(1, 6)])
    lm = lm.with_columns(pl.Series("_sl", np.searchsorted(
        edges, lm["date"].to_numpy().astype("datetime64[D]"), side="right")))
    slices = [effect(lm.filter(pl.col("_sl") == i))[0] for i in range(6)]
    _, cells = effect(lm)
    null = perm_null(oos)
    p5, p95, nmean = np.quantile(null, .05), np.quantile(null, .95), null.mean()
    c6, _ = effect(lmm)
    top, _ = effect(lm.filter(pl.col("big")))
    rest, _ = effect(lm.filter(~pl.col("big")))
    npos = sum(v > 0 for v in slices); nneg = sum(v < 0 for v in slices)
    cpos = sum(v > 0 for v in cells.values()); cneg = sum(v < 0 for v in cells.values())
    crit = [
        ("C1 IS pooled", c1, c1 >= FLOOR, c1 <= -FLOOR),
        ("C2 OOS pooled", c2, c2 >= FLOOR, c2 <= -FLOOR),
        ("C3 calendar slices (pos/neg)", f"{npos}/{nneg}", npos >= 5, nneg >= 5),
        ("C4 ticker x side cells (pos/neg)", f"{cpos}/{cneg}", cpos >= 5, cneg >= 5),
        (f"C5 OOS vs null p5..p95 {p5:+.2f}..{p95:+.2f}", c2, c2 > p95, c2 < p5),
        ("C6 momentum-matched", c6, c6 >= CTRL_FLOOR, c6 <= -CTRL_FLOOR),
        (f"C7 size: top {top:+.2f} vs rest {rest:+.2f}", top - rest, top > rest, top < rest),
    ]
    print(f"\n  ---- {name}")
    print(f"    triggers matched {lm.filter(pl.col('role') == 1).height:,}  "
          f"(IS {is_.filter(pl.col('role') == 1).height:,} / OOS "
          f"{oos.filter(pl.col('role') == 1).height:,}), controls "
          f"{lm.filter(pl.col('role') == 0).height:,}   null mean {nmean:+.2f}")
    print(f"    {'':40}{'value':>10}   H+ LEAD   H- EXHAUST")
    for nm, v, lp, ln in crit:
        vv = f"{v:+.2f}" if isinstance(v, float) else v
        print(f"    {nm:<40}{vv:>10}   {'PASS' if lp else 'fail':<8}  {'PASS' if ln else 'fail'}")
    print("    slices: " + "  ".join(f"{v:+.1f}" for v in slices))
    print("    cells:  " + "  ".join(f"{t}{'+' if s > 0 else '-'} {v:+.1f}"
                                    for (t, s), v in sorted(cells.items())))
    hp = all(c[2] for c in crit); hn = all(c[3] for c in crit)
    print(f"    VERDICT  H+ LEAD: {'PASS' if hp else 'FAIL'}    "
          f"H- EXHAUSTION: {'PASS' if hn else 'FAIL'}")
    return lm


def main():
    t0 = time.time()
    spot = load_spot()
    counts = {k: 0 for k in ("sweep_prints_buy", "candidate", "dedup_dropped",
                             "already_through")}
    trig_rows, ctrl_rows, mir_rows = [], [], []
    t1_rows = []
    files = sorted(glob.glob(os.path.join(CORE, "date=*", "trades.parquet")))
    for n_f, f in enumerate(files):
        day = dt.date.fromisoformat(f.split("date=")[1][:10])
        for t in TICKERS:
            if day < START[t] or day not in spot[t] or spot[t][day].height < 300:
                continue
            build_day(t, day, spot[t][day], f, counts, trig_rows, ctrl_rows, mir_rows)
        # T1 sample: trade-stamped underlying vs 1m close, lags -1/0/+1
        if n_f % 20 == 0:
            tr = (pl.read_parquet(f, columns=["underlying_symbol", "executed_at", "underlying_price"])
                  .with_columns(pl.col("executed_at").dt.convert_time_zone("America/New_York")
                                .dt.truncate("1m").alias("minute_et"))
                  .group_by("underlying_symbol", "minute_et")
                  .agg(pl.col("underlying_price").last().alias("u")))
            for t in TICKERS:
                if day in spot[t]:
                    j = tr.filter(pl.col("underlying_symbol") == t).join(
                        spot[t][day].select("minute_et", "close"), on="minute_et")
                    t1_rows.append(j.select("minute_et", "u", "close")
                                   .with_columns(pl.lit(t).alias("t")))
        if (n_f + 1) % 100 == 0:
            print(f"  {n_f + 1}/{len(files)} days  {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- T1
    print("\n=== T1 trade-stamped underlying_price vs 1m close ===")
    t1 = pl.concat(t1_rows).sort("t", "minute_et")
    for t in TICKERS:
        g = t1.filter(pl.col("t") == t)
        u, c = g["u"].to_numpy(), g["close"].to_numpy()
        gap = np.abs(u - c) / c * 1e4
        du, dc = np.diff(u), np.diff(c)
        cors = {lag: np.corrcoef(du[max(0, -lag):len(du) - max(0, lag)],
                                 dc[max(0, lag):len(dc) - max(0, -lag)])[0, 1]
                for lag in (-1, 0, 1)}
        print(f"  {t}: n={len(u):,}  median |gap| {np.median(gap):.2f}bp  "
              f"corr of changes lag-1 {cors[-1]:.3f}  lag0 {cors[0]:.3f}  lag+1 {cors[1]:.3f}")

    trig = pl.DataFrame(trig_rows, orient="row", schema=[
        "ticker", "date", "side", "i", "kq", "d_ref", "db", "hr", "prem", "y", "mom", "fwd"])
    ctrl = pl.DataFrame(ctrl_rows, orient="row", schema=[
        "ticker", "date", "side", "i", "kq", "db", "hr", "y", "mom", "fwd"]).drop_nulls("y")
    trig = trig.with_columns(pl.int_range(pl.len()).alias("tid"),
                             (pl.col("date") >= SPLIT).alias("oos"))
    # size quartile (IS cuts) and momentum tercile (per ticker x side, all triggers)
    q75 = (trig.filter(~pl.col("oos")).group_by("ticker", "side")
           .agg(pl.col("prem").quantile(0.75).alias("p75")))
    trig = trig.join(q75, on=["ticker", "side"]).with_columns(
        (pl.col("prem") >= pl.col("p75")).alias("big"))
    mc = trig.group_by("ticker", "side").agg(pl.col("mom").quantile(1 / 3).alias("m1"),
                                             pl.col("mom").quantile(2 / 3).alias("m2"))

    def mt(df):
        return df.join(mc, on=["ticker", "side"]).with_columns(
            pl.when(pl.col("mom") <= pl.col("m1")).then(0)
            .when(pl.col("mom") <= pl.col("m2")).then(1).otherwise(2).alias("mt")
        ).drop("m1", "m2")
    trig, ctrl = mt(trig), mt(ctrl)
    trig.write_parquet(os.path.join(OUT, "sweep_triggers.parquet"))

    print("\n=== counts ===")
    for k, v in counts.items():
        print(f"  {k:<20} {v:>10,}")
    print(f"  {'triggers scored':<20} {trig.height:>10,}")
    print(trig.group_by("ticker", "side").agg(pl.len().alias("n"),
          pl.col("oos").sum().alias("oos"), pl.col("y").mean().round(3).alias("arrive"),
          pl.col("prem").median().round(0).alias("med_prem")).sort("ticker", "side"))

    # ---------------------------------------------------------------- M1
    print("\n=== M1 machinery ===")
    planted = trig.with_columns(pl.lit(1).alias("y"))
    pe, _ = effect(match(planted, ctrl))
    lm0 = match(trig, ctrl)
    nul = perm_null(lm0.filter(pl.col("oos")), n=200)
    print(f"  planted (all triggers y=1) -> {pe:+.1f}pp (need >= +20)   "
          f"shuffled-label null mean {nul.mean():+.2f}pp (need within +/-1)")
    if pe < 20 or abs(nul.mean()) > 1.0:
        sys.exit("  M1 FAILED -- the statistic or its null is broken; no verdict printed.")

    print("\n\n==================== ASK-SIDE SWEEPS, <= $1 OTM, SAME-DAY EXPIRY")
    lm = score(trig, ctrl, "0DTE, all triggers")
    tr = lm.filter(pl.col("role") == 1)["tid"].unique()
    fwd_t = trig.filter(pl.col("tid").is_in(tr))["fwd"].mean()
    print(f"    reported: fwd 15m spot, triggers {fwd_t:+.2f}bp vs all control minutes "
          f"{ctrl['fwd'].mean():+.2f}bp")

    # ---------------------------------------------------------------- mirror
    print("\n==================== SELL-SIDE MIRROR (reported, not scored)")
    mir = pl.DataFrame(mir_rows, orient="row",
                       schema=["ticker", "date", "side", "i", "kq", "db", "hr", "y"])
    print(f"  bid-side sweeps at a strike passed by <= $1: {mir.height:,}")
    # matched: same day/ticker/side/hour/distance-past-strike minutes, fall-back rate
    print(mir.group_by("ticker", "side").agg(pl.len().alias("n"),
          pl.col("y").mean().round(3).alias("fell_back_15m")).sort("ticker", "side"))
    print("  (NO matched control was built for the mirror -- these rates are not\n"
          "   comparable to check_strike_crossing_flow's ~0.60, which used a 30m\n"
          "   window from a first crossing. Unscored, supports no conclusion.)")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
