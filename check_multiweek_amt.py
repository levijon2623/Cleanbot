"""
check_multiweek_amt.py
======================
DO MULTI-WEEK VOLUME-PROFILE LEVELS HOLD PRICE -- AND DOES A LONGER WINDOW
MEAN A STRONGER LEVEL?

WHERE THIS COMES FROM
    The AMT work tested the PRIOR DAY's POC / value area (check_amt,
    check_amt_variants: 0 of 48 cells) and a 5-day rolling POC as a filter on
    the flow trigger (check_wpoc_gate / _gating: powered null). Nothing longer
    than a week was tested, the weekly VALUE AREA was never tested, and weekly
    levels were never tested as LEVELS -- only as a trigger filter. The
    operator (2026-09-28): "perhaps a 2 or 3 week POC is more meaningful than
    the 1 week, or a 3 week value area." The flow trigger plays no part here.

THE LEVELS
    Composite volume profile over the N COMPLETE sessions BEFORE day D, for
    N = 5, 10, 15 (1, 2, 3 weeks) -- never including D. Same construction as
    amt_profile.profile_from_bars (1m regular-session bars, each minute's
    volume spread evenly over the bins its low..high spans, bin width 0.05% of
    the ticker's median price, value area expanded from the POC to 70%, ties
    upward), summed across days on the SAME fixed bin grid. Levels: POC, VAH,
    VAL. T1 below checks N = 1 reproduces profile_from_bars.
    Regular session = bars flagged market_time "r" OR unflagged (SPY's 2025-26
    bars mostly carry no flag), 09:30-15:59 ET. Early closes come from the
    CALENDAR (day after Thanksgiving, Jul 3, Dec 24 when trading) and are cut
    at 13:00 -- unflagged days would otherwise carry after-hours bars. A
    session is COMPLETE if it has >= 380 regular bars or is an early close
    with >= 200. A truncated vendor day (QQQ/IWM 2024-10-28, 311 bars ending
    14:40 -- refetched 2026-09-28, still short at source) is left out of
    composites and events, and counted. Early closes feed composites but give
    no events (no 60-minute outcome after 12:00).
    Bins are exact INTEGER CENTS (bin = 0.05% of the median price, rounded to
    the cent: SPY 30c, QQQ 26c, IWM 11c).

THE EVENT -- FIRST TOUCH
    On a full session D, the FIRST 1m bar whose low..high contains the level.
    Scored only if that bar is 09:46-14:59 ET (15 minutes of approach behind
    it, 60 minutes of outcome ahead). A level first touched earlier -- incl.
    at the open -- gives no event that day. Approach side from the previous
    bar's close: above the level = testing it as SUPPORT (s = +1), below =
    testing it as RESISTANCE (s = -1).
    reaction r_h = s x (close[touch + h] - level) / level, in bp: positive =
    price is back on the side it came from (the level HELD), negative = it went
    through. SCORED at h = 15, 30, 60 minutes.

THE CONTROL -- PLACEBO LEVELS TOUCHED THE SAME WAY
    Every real level also spawns placebo levels at level x (1 +/- d), d =
    0.2% .. 2.0% in 0.1% steps, dropping any within 2 bins of ANY real level
    that day. (Widened from six steps of 0.25% after --counts showed 422
    matched 3-week sets with one cell at 29 -- counts only, no outcome seen.) Placebo first touches are found by the identical rule.
    Each real event is matched to up to 5 placebo events from the SAME day and
    ticker, same approach side, same half of the session (touch before / from
    12:00), same tercile of approach speed (s x the 15-minute move INTO the
    touch; terciles per ticker, cut on IS real events).
    "Price held at the 3-week VAL" means nothing alone; "it held more than at a
    random price reached the same way on the same day" is the claim.
    effect = mean over matched sets of (real r - mean of THAT set's placebo r),
    per ticker x level-type cell (>= 30 sets), POOLED = equal-weight mean of
    the 9 cells (SPY/QQQ/IWM x POC/VAH/VAL).

TWO HYPOTHESES, PRE-REGISTERED TOGETHER, SCORED SEPARATELY, ON N = 15
    HOLD   multi-week levels hold price (reaction > placebo)
    BREAK  they are crossed with follow-through (reaction < placebo) -- the
           "acceptance outside value moves to the next reference" reading
    Mirror criteria, as in the flow studies. Neither passes by the other failing.

PRE-COMMITTED CRITERIA, AT EACH h in {15, 30, 60}, N = 15 (3 weeks)
                                          HOLD                  BREAK
    C1  IS pooled effect                  >= +3bp               <= -3bp
    C2  OOS pooled effect                 >= +3bp               <= -3bp
    C3  6 equal-count calendar slices     >= 5 positive         >= 5 negative
    C4  9 ticker x level cells            >= 7 positive         >= 7 negative
    C5  OOS, WEEK-block bootstrap         p5 > 0                p95 < 0
    C6  LONGER IS STRONGER (the operator's hypothesis, made falsifiable):
        all-sample pooled effect ordered  3w >= 2w >= 1w        3w <= 2w <= 1w
        AND week-block bootstrap of 3w-1w p5 > 0                p95 < 0
    PASS at h = all six. IS/OOS split 2025-08-21.
    WHY WEEK BLOCKS: a 3-week POC barely moves from day to day, so touches of
    it on neighbouring days are one level tested repeatedly, not independent
    events. Resampling whole ISO weeks keeps them together; there are only
    ~50 independent 3-week stretches per ticker in the sample.
    3bp: the same economic floor as the flow studies (~7% of an ATM same-day
    option's premium at delta 0.5).

REPORTED, NOT SCORED
    - every window (1w, 2w, 3w) x horizon, pooled and per cell
    - HELD RATE (r_30 > 0), real vs matched placebo
    - by level type and by approach side (support vs resistance)
    - counts at every stage

MACHINERY CHECKS, BEFORE ANY RESULT
    T1  N = 1 composite == an independent, line-by-line integer-cent port of
        amt_profile.profile_from_bars, on 40 random days per ticker: POC / VAH
        / VAL identical on all 40.
        AMENDED 2026-09-28, BEFORE ANY OUTCOME WAS COMPUTED (--counts had not
        printed an event). As first written T1 compared against
        profile_from_bars itself and failed on IWM (24/40 exact). Two causes,
        both float noise, neither a level-logic bug: (a) profile_from_bars
        builds bin edges with np.arange, whose accumulated rounding decides
        which bin a price EXACTLY on an edge falls in -- 17% of IWM bar highs
        and lows sit on an 11-cent edge; (b) this script's first histogram
        used a cumulative-sum shortcut whose noise broke exact volume ties
        differently, and the value-area expansion ties upward. Integer cents
        and direct sums remove both. profile_from_bars agreement is still
        printed, for information.
    M1  planted +10bp on real events -> +10 +/- 0.5 at every h; within-set
        relabel null mean within +/-0.5bp.
    M2  causality: day D's levels are unchanged when D's own bars are zeroed.

DATA
    historical/{SPY,QQQ,IWM}.parquet regular-session 1m bars 2023-10-12 ..
    2026-09-25 (refreshed 2026-09-27; SPY 2026-09-02 refetched whole
    2026-09-28 -- it had been a partial file). The first 15 complete sessions
    only feed composites. No API requests.

RESULT -- 2026-09-28. ALL SIX VERDICTS FAIL. 3-week levels behave like
random prices reached the same way; longer is not stronger.
Machinery: T1 40/40 identical on all three tickers (profile_from_bars itself,
information only: SPY 38, QQQ 36, IWM 28 of 40 -- the float edges). M2 true.
M1 planted +10.00 at every h; relabel null means -0.04 / +0.04 / -0.11bp.
Counts (before any outcome): 756 first touches of a 3-week level in the
09:46-14:59 window, 478 matched sets (cells 37-70); 1-week 780, 2-week 585.
Most levels are never touched on a given day (~72%) or first touched outside
the window -- a 3-week level is usually far from price.

    N = 3 weeks, pooled effect (bp)     15m      30m      60m
    IS                                 -1.45    -1.56    -3.41
    OOS                                -2.41    +1.59    +3.39
    OOS week-bootstrap p5..p95     -9.3..+4.2  -7.6..+10.9  -4.4..+13.1
    cells +/- (of 9)                    5/4      5/4      3/6
    calendar slices +/-                 1/5      4/2      2/4
    Nothing holds its sign from IS to OOS at 30m or 60m, cells split, and
    every bootstrap band straddles zero.

    LONGER IS STRONGER (C6), all-sample pooled:
                      1w       2w       3w
        15m         -2.33    -4.78    -1.25
        30m         -3.50    -3.69    +0.80
        60m         -3.77    -4.52    -0.18
    The ordering the hypothesis needs (3w >= 2w >= 1w) fails at every h: 2w
    is the most negative. 3w sits nearer zero than 1w (3w-1w bootstrap p5
    > 0 at 30m and 60m), but "nearer zero" is not "stronger" -- 3-week
    levels are indistinguishable from placebo, not better than 1-week ones.
    Held rate at 30m (price back on its approach side): 3w real 0.444 vs
    placebo 0.399; 1w 0.419 vs 0.428; 2w 0.402 vs 0.425.

    READING. With 478 sets and week-clustered levels, this test can rule
    out a LARGE effect -- nothing like the 8-10bp its OOS bootstrap band
    would need to exclude zero -- but not a 2-3bp one. So: no multi-week
    POC, VAH or VAL acts as a strong support/resistance level on first touch
    on SPY/QQQ/IWM, and extending the window from one to three weeks does
    not create one. The only consistent sign anywhere is unscored and weak:
    1- and 2-week levels lean slightly NEGATIVE at every horizon (-2 to -5bp,
    price goes through them a little more than through random prices) --
    reported, not a finding; it was not the pre-registered window.
    Together with check_amt (daily, 0/48) and check_wpoc_gate (5-day POC as
    a trigger filter, powered null), no volume-profile level at any horizon
    tested -- 1 day to 3 weeks -- has shown an edge in this project.

    C3 NOTE: as first run, C3 printed nan -- each slice was split into nine
    cells and no cell reached 30 sets. It was changed after that run to the
    plain mean over a slice's sets. That was after outcomes were seen, and it
    changes no verdict: C1, C2 and C5 fail at every horizon for both
    hypotheses.

Usage:  python check_multiweek_amt.py [--counts]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import numpy as np
import polars as pl

from amt_profile import profile_from_bars

TICKERS = ["SPY", "QQQ", "IWM"]
WINDOWS = (5, 10, 15)
PRIMARY = 15
HS = (15, 30, 60)
SPLIT = dt.date(2025, 8, 21)
BIN_PCT, VA_FRAC = 0.0005, 0.70
T_LO, T_HI = 16, 329                    # bar index 09:46 .. 14:59
PLACEBO = [s * k / 1000 for k in range(2, 21) for s in (1, -1)]   # +-0.2% .. +-2.0%, 0.1% steps
K_CTRL, MIN_SETS, FLOOR_BP, N_BOOT = 5, 30, 3.0, 1000
LEVELS = ("poc", "vah", "val")


def _early_closes(years):
    from market_calendar import is_trading_day
    out = set()
    for y in years:
        thu = [dt.date(y, 11, d) for d in range(1, 31) if dt.date(y, 11, d).weekday() == 3]
        for d in (thu[3] + dt.timedelta(days=1), dt.date(y, 7, 3), dt.date(y, 12, 24)):
            if d.weekday() < 5 and is_trading_day(d):
                out.add(d)
    return out


EARLY_CLOSES = _early_closes(range(2023, 2027))
rng = np.random.default_rng(20260928)


# ---------------------------------------------------------------- data
def load_days(tk):
    h = (pl.read_parquet(f"historical/{tk}.parquet",
                         columns=["minute_et", "open", "high", "low", "close", "volume", "market_time"])
         .filter((pl.col("market_time") == "r") | pl.col("market_time").is_null())
         .with_columns(pl.col("minute_et").dt.convert_time_zone("America/New_York"))
         .filter(pl.col("minute_et").dt.time().is_between(dt.time(9, 30), dt.time(15, 59)))
         .unique("minute_et").sort("minute_et")
         .with_columns(pl.col("minute_et").dt.date().alias("d"),
                       (pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod")))
    binw = round(float(h["close"].median()) * BIN_PCT, 2) or 0.01
    days, info = {}, {"full": 0, "early": 0, "dropped": []}
    for (d,), g in h.group_by(["d"], maintain_order=True):
        if d in EARLY_CLOSES:
            g = g.filter(pl.col("mod") < 780)
        n = g.height
        if d in EARLY_CLOSES and n >= 200:
            kind = "early"
        elif d not in EARLY_CLOSES and n >= 380:
            kind = "full"
        else:
            info["dropped"].append((d, n))
            continue
        info[kind] += 1
        b = dict(kind=kind, mod=g["mod"].to_numpy(), o=g["open"].to_numpy(),
                 h=g["high"].to_numpy(), l=g["low"].to_numpy(),
                 c=g["close"].to_numpy(), v=g["volume"].to_numpy().astype(float))
        if kind == "full":
            # EVENTS index bars by minute (touch + h minutes), so a full day is
            # laid on a 390-minute grid: a missing minute cannot touch anything
            # and carries the last close forward. The raw bars still feed the
            # volume histogram.
            i = b["mod"] - 570
            keep = (i >= 0) & (i < 390)
            gh, gl, gc = (np.full(390, np.nan) for _ in range(3))
            gh[i[keep]], gl[i[keep]], gc[i[keep]] = b["h"][keep], b["l"][keep], b["c"][keep]
            for k in range(1, 390):
                if np.isnan(gc[k]):
                    gc[k] = gc[k - 1]
            b.update(gh=gh, gl=gl, gc=gc)
        days[d] = b
    return binw, dict(sorted(days.items())), info


def _bin(x, binw):
    """Price -> integer bin, exactly: integer cents // integer bin cents."""
    return np.round(np.asarray(x) * 100).astype(np.int64) // int(round(binw * 100))


def day_hist(b, binw, base, nb):
    """One session's volume by bin on the ticker's global grid: each minute
    adds v / width to bins low..high, one slice at a time (no cumulative-sum
    shortcut -- its float noise breaks exact volume ties; see T1)."""
    a = _bin(b["l"], binw) - base
    z = np.maximum(_bin(b["h"], binw) - base, a)
    vol = np.zeros(nb)
    for ai, zi, v in zip(a, z, b["v"]):
        vol[ai:zi + 1] += v / (zi - ai + 1)
    return vol


def levels_from_hist(vol, binw, base):
    nz = np.flatnonzero(vol > 0)
    if len(nz) < 3:
        return None
    lo0, hi0 = nz[0], nz[-1]
    v = vol[lo0:hi0 + 1]
    n = len(v)
    poc = int(np.argmax(v))
    total, acc, lo_i, hi_i = v.sum(), v[poc], poc, poc
    while acc < VA_FRAC * total and (lo_i > 0 or hi_i < n - 1):
        can_up, can_dn = hi_i < n - 1, lo_i > 0
        up = v[hi_i + 1] if can_up else -1.0
        dn = v[lo_i - 1] if can_dn else -1.0
        if can_up and (up >= dn or not can_dn):
            hi_i += 1; acc += v[hi_i]
        elif can_dn:
            lo_i -= 1; acc += v[lo_i]
        else:
            break
    ctr = lambda i: (base + lo0 + i + 0.5) * binw
    return dict(poc=ctr(poc), vah=ctr(hi_i), val=ctr(lo_i))


def composites(tk, binw, days):
    lo = min(float(b["l"].min()) for b in days.values())
    hi = max(float(b["h"].max()) for b in days.values())
    base = int(_bin(lo, binw)) - 2
    nb = int(_bin(hi, binw)) - base + 3
    keys = list(days)
    H = np.vstack([day_hist(days[k], binw, base, nb) for k in keys])
    out = {}
    for j, d in enumerate(keys):
        for N in (1,) + WINDOWS:
            if j >= N:
                lv = levels_from_hist(H[j - N:j].sum(axis=0), binw, base)
                if lv:
                    out[(d, N)] = lv
    return out, H, keys, base


def profile_reference(b, binw):
    """T1 reference: amt_profile.profile_from_bars line by line, with its
    bins in integer cents instead of an np.arange of floats."""
    a, z = _bin(b["l"], binw), _bin(b["h"], binw)
    lo = int(a.min()); n = int(z.max()) - lo + 1
    vol = np.zeros(n)
    for ai, zi, v in zip(a - lo, z - lo, b["v"]):
        if zi < ai:
            continue
        vol[ai:zi + 1] += v / (zi - ai + 1)
    poc = int(np.argmax(vol)); total = vol.sum()
    lo_i = hi_i = poc; acc = vol[poc]
    while acc < VA_FRAC * total and (lo_i > 0 or hi_i < n - 1):
        can_up, can_dn = hi_i < n - 1, lo_i > 0
        up = vol[hi_i + 1] if can_up else -1.0
        dn = vol[lo_i - 1] if can_dn else -1.0
        if can_up and (up >= dn or not can_dn):
            hi_i += 1; acc += vol[hi_i]
        elif can_dn:
            lo_i -= 1; acc += vol[lo_i]
        else:
            break
    ctr = lambda i: (lo + i + 0.5) * binw
    return dict(poc=ctr(poc), vah=ctr(hi_i), val=ctr(lo_i))


# ---------------------------------------------------------------- events
def first_touch(b, L):
    with np.errstate(invalid="ignore"):
        hit = (b["gl"] <= L) & (b["gh"] >= L)
    if not hit.any():
        return None
    f = int(np.argmax(hit))
    if f < T_LO or f > T_HI:
        return None
    c = b["gc"]
    if np.isnan(c[f - 16]):
        return None
    s = 1 if c[f - 1] > L else -1
    r = [s * (c[f + h] - L) / L * 1e4 for h in HS]
    app = s * (c[f - 16] - c[f - 1]) / c[f - 16] * 1e4
    return f, s, r, app


def build_events(tk, binw, days, lv):
    real, plc = [], []
    counts = dict(level_days=0, touched=0, touched_early_or_late=0, never=0)
    for d, b in days.items():
        if b["kind"] != "full":
            continue
        reals = {(N, k): lv[(d, N)][k] for N in WINDOWS if (d, N) in lv for k in LEVELS}
        if len(reals) < 3 * len(WINDOWS):
            continue
        for (N, k), L in reals.items():
            counts["level_days"] += 1
            e = first_touch(b, L)
            with np.errstate(invalid="ignore"):
                hit = ((b["gl"] <= L) & (b["gh"] >= L)).any()
            if e is None:
                counts["touched_early_or_late" if hit else "never"] += 1
                continue
            counts["touched"] += 1
            f, s, r, app = e
            real.append((tk, d, N, k, f, s, *r, app))
        allreal = np.array(list(reals.values()))
        seen = set()
        for L0 in allreal:
            for dd in PLACEBO:
                P = round(L0 * (1 + dd) / binw) * binw
                if P in seen or np.min(np.abs(allreal - P)) <= 2 * binw:
                    continue
                seen.add(P)
                e = first_touch(b, P)
                if e is None:
                    continue
                f, s, r, app = e
                plc.append((tk, d, f, s, *r, app))
    return real, plc, counts


# ---------------------------------------------------------------- statistics
RC = [f"r{h}" for h in HS]


def match(real, plc):
    keys = ["ticker", "date", "s", "pm", "at"]
    j = real.select(keys + ["rid"]).join(plc.select(keys + RC), on=keys, how="inner")
    j = j.with_columns(pl.int_range(pl.len()).shuffle(seed=int(rng.integers(1e9)))
                       .over("rid").alias("_r")).filter(pl.col("_r") < K_CTRL)
    ctl = j.select("rid", pl.lit(0, pl.Int8).alias("role"), *RC)
    tr = real.filter(pl.col("rid").is_in(j["rid"].unique().implode())).select(
        "rid", pl.lit(1, pl.Int8).alias("role"), *RC)
    meta = real.select("rid", "ticker", "level", "N", "date", "oos", "week", "_sl", "s")
    return pl.concat([tr, ctl]).join(meta, on="rid").sort("rid", "role", descending=[False, True])


def set_diffs(long, col, role=None):
    """One row per matched set: rid, cell, week, oos, d = real - mean(placebos)."""
    rid = long["rid"].to_numpy()
    r = long[col].to_numpy().astype(float)
    ro = long["role"].to_numpy() if role is None else role
    st = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    grp = np.repeat(np.arange(len(st)), np.diff(np.r_[st, len(rid)]))
    rt = np.bincount(grp, np.where(ro == 1, r, 0), len(st))
    nt = np.bincount(grp, (ro == 1).astype(float), len(st))
    rc = np.bincount(grp, np.where(ro == 0, r, 0), len(st))
    nc = np.bincount(grp, (ro == 0).astype(float), len(st))
    ok = (nt == 1) & (nc > 0)
    cell = np.char.add(long["ticker"].to_numpy()[st].astype(str),
                       long["level"].to_numpy()[st].astype(str))
    return dict(d=(rt - rc / np.maximum(nc, 1))[ok], cell=cell[ok],
                week=long["week"].to_numpy()[st][ok], oos=long["oos"].to_numpy()[st][ok],
                sl=long["_sl"].to_numpy()[st][ok])


def pooled(sd, mask=None):
    d, c = sd["d"], sd["cell"]
    if mask is not None:
        d, c = d[mask], c[mask]
    per = {}
    for k in np.unique(c):
        x = d[c == k]
        if len(x) >= MIN_SETS:
            per[k] = float(x.mean())
    return (float(np.mean(list(per.values()))) if per else np.nan), per


def week_boot(sd_list, mask_fn=None, n=N_BOOT):
    """Resample ISO weeks with replacement, jointly across several set lists
    (so a 3w-1w difference keeps its pairing). Returns [n, len(sd_list)]."""
    weeks = np.unique(np.concatenate([s["week"] for s in sd_list]))
    widx = {w: i for i, w in enumerate(weeks)}
    mats = []
    for s in sd_list:
        m = np.ones(len(s["d"]), bool) if mask_fn is None else mask_fn(s)
        cells = np.unique(s["cell"][m])
        S = np.zeros((len(weeks), len(cells))); C = np.zeros_like(S)
        ci = {c: i for i, c in enumerate(cells)}
        for d, c, w in zip(s["d"][m], s["cell"][m], s["week"][m]):
            S[widx[w], ci[c]] += d; C[widx[w], ci[c]] += 1
        mats.append((S, C))
    out = np.empty((n, len(sd_list)))
    for b in range(n):
        cnt = np.bincount(rng.integers(0, len(weeks), len(weeks)), minlength=len(weeks))
        for j, (S, C) in enumerate(mats):
            ss, cc = cnt @ S, cnt @ C
            keep = cc >= MIN_SETS
            out[b, j] = np.mean(ss[keep] / cc[keep]) if keep.any() else np.nan
    return out


def relabel_null(long, col, n=200):
    rid = long["rid"].to_numpy()
    st = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    sz = np.diff(np.r_[st, len(rid)])
    out = np.empty(n)
    for k in range(n):
        role = np.zeros(len(rid), np.int8)
        role[st + (rng.random(len(st)) * sz).astype(int)] = 1
        out[k] = pooled(set_diffs(long, col, role))[0]
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    reals, plcs = [], []
    print("=== sessions and machinery (T1, M2) ===")
    for tk in TICKERS:
        binw, days, info = load_days(tk)
        lv, H, keys, base = composites(tk, binw, days)
        print(f"  {tk}: bin {binw}  sessions full {info['full']} early-close {info['early']}  "
              f"dropped {[(str(d), n) for d, n in info['dropped']]}")
        # T1: N=1 composite for day j+1 == profile_from_bars(day j)
        idx = rng.choice(np.arange(1, len(keys) - 1), 40, replace=False)
        exact = pfb = 0
        for j in idx:
            b = days[keys[j]]
            mine = lv.get((keys[j + 1], 1))
            ref = profile_reference(b, binw)
            exact += bool(mine) and all(abs(ref[k] - mine[k]) < 1e-9 for k in LEVELS)
            p0 = profile_from_bars([dict(mod=m, o=o, h=h, l=l, c=c, v=v) for m, o, h, l, c, v in
                                    zip(b["mod"], b["o"], b["h"], b["l"], b["c"], b["v"])], binw)
            pfb += bool(p0 and mine) and all(abs(p0[k] - mine[k]) < binw / 100 for k in LEVELS)
        print(f"      T1 N=1 vs integer-cent reference: {exact}/40 identical   "
              f"(vs float-edged profile_from_bars, information only: {pfb}/40)")
        if exact < 40:
            sys.exit("  T1 FAILED -- composites do not reproduce the daily profile.")
        # M2: zero day D's own histogram -> D's levels unchanged
        j = len(keys) // 2
        H2 = H.copy(); H2[j] = 0
        same = all(levels_from_hist(H2[j - N:j].sum(axis=0), binw, base) == lv[(keys[j], N)]
                   for N in WINDOWS)
        print(f"      M2 day {keys[j]} levels unchanged with its own bars zeroed: {same}")
        if not same:
            sys.exit("  M2 FAILED -- a level uses its own day.")
        r, p, cnt = build_events(tk, binw, days, lv)
        print(f"      {cnt}")
        reals += r; plcs += p

    real = pl.DataFrame(reals, orient="row", schema=[
        "ticker", "date", "N", "level", "f", "s", *RC, "app"])
    plc = pl.DataFrame(plcs, orient="row", schema=["ticker", "date", "f", "s", *RC, "app"])
    real = real.with_columns(pl.int_range(pl.len()).alias("rid"),
                             (pl.col("date") >= SPLIT).alias("oos"),
                             (pl.col("f") >= 150).alias("pm"),
                             (pl.col("date").dt.iso_year().cast(pl.Int32) * 100
                              + pl.col("date").dt.week().cast(pl.Int32)).alias("week"))
    plc = plc.with_columns((pl.col("f") >= 150).alias("pm"))
    cuts = (real.filter(~pl.col("oos")).group_by("ticker")
            .agg(pl.col("app").quantile(1 / 3).alias("a1"), pl.col("app").quantile(2 / 3).alias("a2")))

    def at(df):
        return df.join(cuts, on="ticker").with_columns(
            pl.when(pl.col("app") <= pl.col("a1")).then(0).when(pl.col("app") <= pl.col("a2"))
            .then(1).otherwise(2).alias("at")).drop("a1", "a2")
    real, plc = at(real), at(plc)
    days_sorted = np.array(sorted(real["date"].unique().to_list()), dtype="datetime64[D]")
    edges = np.array([days_sorted[int(len(days_sorted) * k / 6)] for k in range(1, 6)])
    real = real.with_columns(pl.Series("_sl", np.searchsorted(
        edges, real["date"].to_numpy().astype("datetime64[D]"), side="right")))

    print("\n=== counts ===")
    print(real.group_by("N", "ticker").agg(pl.len().alias("first_touches"),
          pl.col("oos").sum().alias("oos")).sort("N", "ticker"))
    print(f"  placebo first touches: {plc.height:,}")
    L = {N: match(real.filter(pl.col("N") == N), plc) for N in WINDOWS}
    for N in WINDOWS:
        s = L[N].filter(pl.col("role") == 1)
        print(f"  {N:>2}d matched sets {s.height:,}  " + "  ".join(
            f"{t}-{k} {g.height}" for (t, k), g in sorted(s.group_by("ticker", "level"), key=lambda kv: kv[0])))
    if a.counts:
        print(f"\n--counts: stopping before any outcome.  {time.time() - t0:.0f}s")
        return

    print("\n=== M1 machinery ===")
    ok = True
    for h in HS:
        col = f"r{h}"
        base_e = pooled(set_diffs(L[PRIMARY], col))[0]
        pl_ = L[PRIMARY].with_columns(pl.when(pl.col("role") == 1).then(pl.col(col) + 10.0)
                                      .otherwise(pl.col(col)).alias(col))
        pe = pooled(set_diffs(pl_, col))[0] - base_e
        nul = relabel_null(L[PRIMARY], col)
        print(f"  {h:>2}m planted +10 -> {pe:+.2f}   relabel null mean {nul.mean():+.3f}bp")
        ok &= abs(pe - 10) <= 0.5 and abs(nul.mean()) <= 0.5
    if not ok:
        sys.exit("  M1 FAILED -- no verdict printed.")

    print("\n\n==================== MULTI-WEEK COMPOSITE LEVELS, FIRST TOUCH vs PLACEBO")
    verdict = {}
    for h in HS:
        col = f"r{h}"
        sd = {N: set_diffs(L[N], col) for N in WINDOWS}
        e_all = {N: pooled(sd[N])[0] for N in WINDOWS}
        p = sd[PRIMARY]
        c1, _ = pooled(p, ~p["oos"])
        c2, _ = pooled(p, p["oos"])
        # C3 per slice is the plain mean over its sets: split nine ways, a
        # slice has no cell with >= 30 sets (fixed after the first run printed
        # nan -- it cannot change any verdict; C1/C2/C5 fail throughout)
        slices = [float(p["d"][p["sl"] == k].mean()) for k in range(6)]
        _, cells = pooled(p)
        boot_oos = week_boot([p], lambda s: s["oos"])[:, 0]
        bd = week_boot([sd[PRIMARY], sd[5]])
        diff = bd[:, 0] - bd[:, 1]
        o5, o95 = np.nanquantile(boot_oos, [.05, .95])
        d5, d95 = np.nanquantile(diff, [.05, .95])
        npos = sum(v > 0 for v in slices); nneg = sum(v < 0 for v in slices)
        cpos = sum(v > 0 for v in cells.values()); cneg = sum(v < 0 for v in cells.values())
        up = e_all[15] >= e_all[10] >= e_all[5]
        dn = e_all[15] <= e_all[10] <= e_all[5]
        crit = [
            ("C1 IS pooled", c1, c1 >= FLOOR_BP, c1 <= -FLOOR_BP),
            ("C2 OOS pooled", c2, c2 >= FLOOR_BP, c2 <= -FLOOR_BP),
            ("C3 calendar slices (pos/neg)", f"{npos}/{nneg}", npos >= 5, nneg >= 5),
            (f"C4 cells (pos/neg of {len(cells)})", f"{cpos}/{cneg}", cpos >= 7, cneg >= 7),
            (f"C5 OOS week-boot p5..p95 {o5:+.2f}..{o95:+.2f}", c2, o5 > 0, o95 < 0),
            (f"C6 1w {e_all[5]:+.2f} 2w {e_all[10]:+.2f} 3w {e_all[15]:+.2f}; "
             f"3w-1w p5..p95 {d5:+.2f}..{d95:+.2f}", e_all[15] - e_all[5],
             up and d5 > 0, dn and d95 < 0),
        ]
        print(f"\n  ---- h = {h}m   (N = 3 weeks; sets {len(p['d']):,})")
        print(f"    {'':62}{'value':>8}   HOLD   BREAK")
        for nm, v, lh, lb in crit:
            vv = f"{v:+.2f}" if isinstance(v, float) else v
            print(f"    {nm:<62}{vv:>8}   {'PASS' if lh else 'fail':<6} {'PASS' if lb else 'fail'}")
        print("    slices: " + "  ".join(f"{v:+.1f}" for v in slices))
        print("    cells:  " + "  ".join(f"{k} {v:+.1f}" for k, v in sorted(cells.items())))
        verdict[h] = (all(c[2] for c in crit), all(c[3] for c in crit))

    print("\n==================== REPORTED")
    for N in WINDOWS:
        for h in HS:
            sd = set_diffs(L[N], f"r{h}")
            e, per = pooled(sd)
            print(f"  {N:>2}d {h:>2}m pooled {e:+.2f}   " + "  ".join(
                f"{k} {v:+.1f}" for k, v in sorted(per.items())))
    for N in WINDOWS:
        lm = L[N]
        rid = lm.filter(pl.col("role") == 1)
        held_r = (rid["r30"] > 0).mean()
        held_p = (lm.filter(pl.col("role") == 0)["r30"] > 0).mean()
        by = []
        for k in LEVELS:
            for s, nm in ((1, "support"), (-1, "resist")):
                sub = lm.filter((pl.col("level") == k) & (pl.col("s") == s))
                by.append(f"{k}/{nm} {pooled(set_diffs(sub, 'r30'))[0]:+.1f}" if sub.height else "")
        print(f"  {N:>2}d held@30m real {held_r:.3f} vs placebo {held_p:.3f}   " + "  ".join(by))

    print("\n==================== VERDICTS (N = 3 weeks)")
    for h in HS:
        print(f"  {h:>2}m   HOLD: {'PASS' if verdict[h][0] else 'FAIL'}    "
              f"BREAK: {'PASS' if verdict[h][1] else 'FAIL'}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
