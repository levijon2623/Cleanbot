"""
check_index_flow.py
===================
Does SPX's options flow add anything to SPY's flow trigger? Two pre-registered
tests on the deployed window (2024-08-20 .. lake end), through sim_core.

    python check_index_flow.py --fetch        # SPX net-prem-ticks -> _index_flow_cache/
    python check_index_flow.py --selftest     # machinery + placebo pre-flight, no scoring
    python check_index_flow.py                # both tests, scored

WHY (2026-10-01)
    On 90 sessions to 2026-09-30, SPX was 86% of gross minute flow in SPY+SPX,
    yet its cum flow was uncorrelated with SPY's (minute r -0.09, day-end r
    -0.29) and only 11 of 200 gate-passing SPY CALL triggers survived in the
    blend (memory: cleanbot-index-flow-blend). So "blend the index in" is a
    DIFFERENT signal, and the question is whether it, or the divergence
    between the two, carries information about SPY option trades.

DATA
    SPY  historical/NETPREMSPY.parquet (UW net-prem-ticks; checked IDENTICAL to
         the live API on 87 days / 33,930 minutes, 2026-10-01).
    SPX  same endpoint, 2024-06-03 .. 2026-09-25, cached by --fetch. Only SPX
         minutes inside SPY's own session grid (09:30-16:14) are used -- SPX's
         global-hours prints before 09:30 and after 16:15 would otherwise move
         the open's cum flow with information SPY's trigger never had.
    Both arms start their flow on 2024-06-03 so the trailing 60-day percentile
    gate warms up on identical calendars. The pre-sample holdout is not scored.

POPULATION (both tests) -- a SCREEN, not the deployed rule
    SPY, direction CALL and PUT x min_flow_pct 50 / 80 / 90 = 6 cells; hours
    9-14, entry >= 09:35 (NO_ENTRY_BEFORE_MOD), dte [0, 1], ATM, no regime or
    AMT gate, the book-default exit (policy_for -> trail50). Sequential, one
    position per ticker, fill="bot" with the exit cushion. Scored per trade.
    The deployed SPY CHOP CALL rule is reported alongside (too thin to score).

TEST A -- does a BLENDED trigger beat SPY's own?
    arms   SPY    cum(SPY)                        -- the incumbent
           BLEND  cum(SPY + SPX)                  -- the candidate
           SPX    cum(SPX) trading SPY options    -- report only
           PLACEBO x20  cum(SPY + SPX of ANOTHER day, same minute-of-day):
                  keeps the added series' size, shape and intraday profile and
                  destroys only its same-day information. If BLEND wins because
                  adding a big noisy series changes WHICH minutes cross, the
                  placebo wins too.
    A1  pooled OOS per-trade mean, BLEND - SPY                  >= +5pp
    A2  pooled IS per-trade mean, BLEND - SPY                   >= 0
    A3  pooled OOS TOTAL, BLEND - SPY                           >= 0  (both objectives)
    A4  A1's delta > p95 of the 20 placebo deltas
    A5  OOS delta > 0 in >= 4 of the 6 cells
    A6  A1's sign holds under fill = mid, bot AND worst
    A7  first-trigger-per-day sample (cap=1): OOS delta > 0
    PASS = A1..A7.

TEST B -- does SPX's flow STATE condition SPY's trigger? (ETF vs index)
    feature  at each SPY trigger minute t: SPX cum vs its own EMA(5), on the
             same closed minute (no lookahead). ALIGNED = SPX on the side of the
             trigger's direction; OPPOSED = the other side.
    policies KEEP-ALIGNED (skip opposed) and KEEP-OPPOSED (skip aligned). BOTH
             are scored and printed; a skip keeps the book flat, so the next
             trigger is eligible at once (sim_core.walk on the filtered list).
    For ONE policy to pass, all of:
    B1  pooled OOS per-trade gain vs unfiltered SPY             >= +5pp
    B2  pooled IS gain                                          >= 0
    B3  pooled OOS TOTAL of the filtered book                   >= unfiltered total
        (a skip that lifts the mean by cutting winners is the twins trap)
    B4  B1's gain > p95 of 20 placebo gains (SPX state taken from another day)
    B5  OOS gain > 0 in >= 4 of 6 cells
    B6  B1's sign holds under fill = mid, bot AND worst
    B7  first-trigger sample: OOS gain > 0
    If both policies "pass" (impossible unless the machinery is broken), FAIL.

REPORTED, NOT SCORED
    day-block bootstrap 90% CI of each pooled OOS delta; IS/OOS n and days;
    win rate; slice signs; the deployed SPY CHOP CALL rule under each arm.

MACHINERY / PLACEBO PRE-FLIGHT (--selftest, METHODOLOGY 7 "can it move?")
    M1  BLEND with SPX zeroed reproduces the SPY arm EXACTLY (same candidates)
    M2  the placebo derangement maps no day to itself, and its added series
        has the same per-day |flow| totals as SPX (a permutation, not a resample)
    M3  a feature planted to equal "trade won" gives a large B gain (the
        statistic can move); a constant feature gives exactly 0 gain
    M4  (scored run) candidates are built ONCE per direction over the union of
        every arm's trigger minutes (a payload depends only on date, minute and
        direction), then each arm takes its own; the SPY arm assembled that way
        must equal a direct per-arm build_candidates exactly. Added 2026-10-01
        for speed (46 builds x ~5 min), before any scoring.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC   # noqa: E402

CACHE = os.path.join("_index_flow_cache", "NETPREMSPX.parquet")
FLOW_START = pd.Timestamp("2024-06-03").date()
CELLS = [(d, p) for d in ("CALL", "PUT") for p in (50, 80, 90)]
AFTER = 9 * 60 + 35
N_PLACEBO = 20
SEED = 20261001
FILLS = ("mid", "bot", "worst")
A = 2 / 6                      # EMA(5), adjust=False


# ------------------------------------------------------------------ data
def fetch():
    import requests
    from dotenv import dotenv_values
    from unusual_whales_client import UnusualWhalesClient
    import market_calendar as MC
    uw = UnusualWhalesClient(dotenv_values(".env")["UW_API_KEY"])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    have = pd.read_parquet(CACHE) if os.path.exists(CACHE) else None
    done = set(have["date"].astype(str)) if have is not None else set()
    rows, d = [], FLOW_START
    while d <= dt.date.today():
        if MC.is_trading_day(d) and d.isoformat() not in done and d < dt.date.today():
            r = requests.get(uw.base_url + "/api/stock/SPX/net-prem-ticks", headers=uw.headers,
                             params={"date": d.isoformat()}, timeout=30)
            data = (r.json().get("data") if r.status_code == 200 else None) or []
            for x in data:
                if x.get("date") == d.isoformat():      # UW can ignore params
                    rows.append(dict(tape_time=x["tape_time"], date=x["date"],
                                     net=float(x.get("net_call_premium") or 0)
                                     - float(x.get("net_put_premium") or 0)))
        d += dt.timedelta(days=1)
    if rows:
        df = pd.DataFrame(rows)
        df["minute_et"] = (pd.to_datetime(df["tape_time"], utc=True)
                           .dt.tz_convert("America/New_York").dt.floor("min"))
        df = df.groupby(["date", "minute_et"], as_index=False)["net"].sum()
        df = df.rename(columns={"net": "net_premium"})
        if have is not None:
            df = pd.concat([have, df], ignore_index=True)
        df.to_parquet(CACHE, index=False)
    print(f"fetched {len(rows)} SPX rows -> {CACHE}")


def load():
    """(spy, spx): DataFrames [date, mod, minute_et (naive), net] on SPY's grid."""
    import directional_flow_backtester as D
    spy = pd.read_parquet("historical/NETPREMSPY.parquet", columns=["minute_et", "net_premium"])
    spx = pd.read_parquet(CACHE)
    out = []
    for df in (spy, spx):
        df = df.copy()
        df["date"] = df["minute_et"].dt.date
        df["mod"] = (df["minute_et"].dt.hour.astype(int) * 60 + df["minute_et"].dt.minute.astype(int))
        df["minute_et"] = D._naive(df["minute_et"])
        df["net"] = pd.to_numeric(df["net_premium"], errors="coerce").fillna(0.0)
        out.append(df[df["date"] >= FLOW_START][["date", "mod", "minute_et", "net"]])
    spy, spx = out
    return spy, spx


def flow_frame(spy, add=None):
    """The lake-shaped flow frame triggers_for() expects; `add` is a
    {(date, mod): net} map added onto SPY's own grid (SPY minutes only)."""
    df = spy.copy()
    if add is not None:
        df["net"] = df["net"].to_numpy() + np.array(
            [add.get((d, m), 0.0) for d, m in zip(df["date"], df["mod"])])
    df = df.sort_values("minute_et")
    df["underlying_symbol"] = "SPY"
    df["net_flow_1m"] = df["net"]
    df["cum_flow"] = df.groupby("date")["net_flow_1m"].cumsum()
    return df[["underlying_symbol", "minute_et", "date", "net_flow_1m", "cum_flow"]]


def spx_map(spx, perm=None):
    """{(date, mod): net}; with `perm` = {target_day: source_day}, each target
    day carries the SOURCE day's SPX minutes (the placebo)."""
    by = {d: dict(zip(g["mod"], g["net"])) for d, g in spx.groupby("date")}
    out = {}
    for d in by:
        src = by[perm[d]] if perm is not None else by[d]
        for m, v in src.items():
            out[(d, m)] = v
    return out


def derangement(days, rng):
    """A random permutation of trading days with NO fixed point."""
    days = list(days)
    while True:
        p = list(rng.permutation(days))
        if all(a != b for a, b in zip(days, p)):
            return dict(zip(days, p))


# ------------------------------------------------------------------ candidates
def screen_rule(direction, pct):
    return dict(name=f"SPY {direction} p{pct}", ticker="SPY", direction=direction,
                hours=list(range(9, 15)), min_flow_pct=pct, dte=[0, 1],
                target_roe=1.0, rr=1.0)


def arm_candidates(D, flow):
    """{(direction, pct): [(date, mod, payload)]} for the 6 cells, from ONE
    build per direction at p50 and the p80/p90 subsets by the trigger's own
    trailing threshold (identical to building each cell; avoids 3x the bar work)."""
    trigs = arm_trigs(D, flow)
    by_min = _by_min(trigs)
    out = {}
    for direction in ("CALL", "PUT"):
        base = SC.build_candidates(D, screen_rule(direction, 50), trigs=trigs)
        for pct in (50, 80, 90):
            keep = []
            for c in base:
                t = by_min.get((c[0], direction, c[1]))
                if t and t.get("thr") and t["abs_flow"] >= t["thr"][pct]:
                    keep.append(c)
            out[(direction, pct)] = keep
    return out, trigs


def arm_trigs(D, flow):
    trigs = D.triggers_for(flow, "SPY")
    D.annotate_flow_pct(trigs, 60)
    return trigs


def _mod(ts):
    ts = pd.Timestamp(ts)
    return ts.hour * 60 + ts.minute


def _by_min(trigs):
    return {(t["date"], t["dir"], _mod(t["ts"])): t for t in trigs}


def _gated(trigs, direction, pct):
    """The (date, mod) minutes an arm's p<pct> cell would hand build_candidates:
    direction, hours 9-14, and abs_flow >= its own trailing threshold."""
    return [(t["date"], _mod(t["ts"])) for t in trigs
            if t["dir"] == direction and 9 <= t["hour"] <= 14
            and t.get("thr") and t["abs_flow"] >= t["thr"][pct]]


def union_payloads(D, trig_sets):
    """ONE build_candidates call per direction over the UNION of every arm's
    p50-gated minutes. A payload depends only on (date, minute, direction) --
    spot, contract pick, entry quote and forward path -- never on which arm's
    flow produced the crossover, so each arm can then take its own minutes
    from this table. Still sim_core's builder, just called once instead of 46
    times; --selftest M4 asserts it reproduces a direct per-arm build exactly."""
    table = {}
    for direction in ("CALL", "PUT"):
        need = {}
        for trigs in trig_sets:
            for t in trigs:
                if (t["dir"] == direction and 9 <= t["hour"] <= 14 and t.get("thr")
                        and t["abs_flow"] >= t["thr"][50]):
                    need.setdefault((t["date"], _mod(t["ts"])), t)
        # every needed minute passes the p50 gate by construction
        stub = [dict(date=t["date"], ts=t["ts"], hour=t["hour"], dir=direction,
                     abs_flow=1.0, thr={p: 0.0 for p in D.GRID_MIN_FLOW_PCT})
                for t in need.values()]
        print(f"    union build {direction}: {len(stub)} minutes", flush=True)
        for c in SC.build_candidates(D, screen_rule(direction, 50), trigs=stub):
            table[(direction, c[0], c[1])] = c
    return table


def assemble(table, trigs):
    out = {}
    for direction, pct in CELLS:
        ms = sorted(set(_gated(trigs, direction, pct)))
        out[(direction, pct)] = [table[(direction, d, m)] for d, m in ms
                                 if (direction, d, m) in table]
    return out


def score(cands, fill="bot", cap=None):
    pol = SC.policy_for(screen_rule("CALL", 50))
    return {k: SC.walk(v, pol, SC.DEFAULT_EOD, fill=fill, cap=cap, after=AFTER)
            for k, v in cands.items()}


def pooled(rows_by_cell):
    return [r for k in CELLS for r in rows_by_cell[k]]


def mean_of(rows, oos):
    v = [r[1] for r in rows if (r[0] >= SC.SPLIT) == oos]
    return float(np.mean(v)) if v else float("nan"), float(np.sum(v)) if v else 0.0, len(v)


def boot_delta(a, b, n=2000, seed=SEED):
    """90% day-block CI of mean(b) - mean(a) on OOS days (days resampled intact)."""
    rng = np.random.default_rng(seed)
    da, db = {}, {}
    for d, x, *_ in a:
        if d >= SC.SPLIT:
            da.setdefault(d, []).append(x)
    for d, x, *_ in b:
        if d >= SC.SPLIT:
            db.setdefault(d, []).append(x)
    days = sorted(set(da) | set(db))
    if not days:
        return float("nan"), float("nan")
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days))
        xa = [x for i in pick for x in da.get(days[i], [])]
        xb = [x for i in pick for x in db.get(days[i], [])]
        if xa and xb:
            out.append(np.mean(xb) - np.mean(xa))
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


# ------------------------------------------------------------------ test B feature
def spx_state(spx_by_day):
    """{(date, mod): +1 if SPX cum > its EMA(5) at that closed minute, -1 if
    below, 0 if equal} -- the same recursion the trigger uses."""
    st = {}
    for d, mins in spx_by_day.items():
        c, e = 0.0, None
        for m in range(570, 975):
            c += mins.get(m, 0.0)
            e = c if e is None else A * c + (1 - A) * e
            st[(d, m)] = (c > e) - (c < e)
    return st


def filt(cands, state, keep):
    """keep='aligned' | 'opposed' -> the candidate list without the others."""
    out = {}
    for (direction, pct), cs in cands.items():
        sg = 1 if direction == "CALL" else -1
        out[(direction, pct)] = [c for c in cs
                                 if (state.get((c[0], c[1]), 0) * sg > 0) == (keep == "aligned")]
    return out


# ------------------------------------------------------------------ run
def selftest(D, spy, spx, assembled=None):
    """--selftest: M1-M3 on direct builds. In a scored run, `assembled` is the
    SPY arm taken from the union table, and M4 checks it against a direct build."""
    fails = []

    def ok(label, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        cond or fails.append(label)
    c0, _ = arm_candidates(D, flow_frame(spy))
    if assembled is None:
        cz, _ = arm_candidates(D, flow_frame(spy, add={k: 0.0 for k in spx_map(spx)}))
        ok("M1 BLEND with SPX zeroed == SPY arm (identical candidates)",
           all([(a[0], a[1]) for a in c0[k]] == [(b[0], b[1]) for b in cz[k]] for k in CELLS))
    else:
        same = all([(a[0], a[1], a[2][0], len(a[2][2])) for a in c0[k]]
                   == [(b[0], b[1], b[2][0], len(b[2][2])) for b in assembled[k]] for k in CELLS)
        ok(f"M4 union-table SPY arm == direct build_candidates ({sum(len(c0[k]) for k in CELLS)} cands)", same)
    rng = np.random.default_rng(SEED)
    days = sorted(spx["date"].unique())
    perm = derangement(days, rng)
    ok("M2 derangement has no fixed point", all(perm[d] != d for d in days))
    tot = spx.groupby("date")["net"].apply(lambda s: float(np.abs(s).sum()))
    pm = spx_map(spx, perm)
    ptot = {}
    for (d, _m), v in pm.items():
        ptot[d] = ptot.get(d, 0.0) + abs(v)
    ok("M2 placebo per-day |flow| totals are a permutation of SPX's",
       sorted(round(v, 2) for v in ptot.values()) == sorted(round(v, 2) for v in tot.values))
    base = score(c0)
    pol = SC.policy_for(screen_rule("CALL", 50))
    won = {}
    for k in CELLS:
        for c in c0[k]:
            pnl, _x, _t = SC.simulate(c[2], pol, SC.DEFAULT_EOD, fill="bot")
            won[(c[0], c[1])] = 1 if pnl > 0 else -1
    # planted: state "agrees" with the trade exactly when the trade wins
    plant = {}
    for k in CELLS:
        sg = 1 if k[0] == "CALL" else -1
        for c in c0[k]:
            plant[(c[0], c[1])] = sg * won[(c[0], c[1])]
    g_plant = mean_of(pooled(score(filt(c0, plant, "aligned"))), True)[0] - mean_of(pooled(base), True)[0]
    ok(f"M3 planted oracle feature moves the statistic (gain {g_plant * 100:+.1f}pp)", g_plant > 0.10)
    # a state that is "aligned" for EVERY candidate keeps them all -> gain must be exactly 0
    every = {(c[0], c[1]): (1 if k[0] == "CALL" else -1) for k in CELLS for c in c0[k]}
    g_c = mean_of(pooled(score(filt(c0, every, "aligned"))), True)[0] - mean_of(pooled(base), True)[0]
    ok(f"M3 a keep-everything filter gives exactly 0 gain ({g_c:+.6f})", g_c == 0.0)
    print(f"\n  {'machinery checks pass' if not fails else f'{len(fails)} FAILED'}")
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        return fetch()
    import directional_flow_backtester as D
    spy, spx = load()
    print(f"SPY flow {spy['date'].min()}..{spy['date'].max()} ({spy['date'].nunique()} d), "
          f"SPX {spx['date'].min()}..{spx['date'].max()} ({spx['date'].nunique()} d)")
    if a.selftest:
        sys.exit(0 if selftest(D, spy, spx) else 1)
    rng = np.random.default_rng(SEED)
    days = sorted(spx["date"].unique())
    real = spx_map(spx)
    arms = {"SPY": flow_frame(spy), "BLEND": flow_frame(spy, add=real)}
    spx_only = spy[["date", "mod", "minute_et"]].copy()
    spx_only["net"] = [real.get((d, m), 0.0) for d, m in zip(spx_only["date"], spx_only["mod"])]
    arms["SPX"] = flow_frame(spx_only)
    perms = [derangement(days, rng) for _ in range(N_PLACEBO)]
    print("  triggers for 3 arms + 20 placebos ...", flush=True)
    trig = {k: arm_trigs(D, f) for k, f in arms.items()}
    ptrigs = [arm_trigs(D, flow_frame(spy, add=spx_map(spx, p))) for p in perms]
    table = union_payloads(D, list(trig.values()) + ptrigs)
    cands = {k: assemble(table, t) for k, t in trig.items()}
    pcands = [assemble(table, t) for t in ptrigs]
    if not selftest(D, spy, spx, assembled=cands["SPY"]):
        sys.exit("machinery checks failed -- not scoring")

    res = {k: score(v) for k, v in cands.items()}
    base_o = mean_of(pooled(res["SPY"]), True)
    print("\n=== TEST A -- SPY trigger vs blended (pooled 6 cells, fill=bot, sequential, trail50)")
    for k in ("SPY", "BLEND", "SPX"):
        s = SC.stat(pooled(res[k]))
        print(f"  {k:6} n={s['n']:5} d={s['nd']:4}  IS {s['is_']*100:+6.1f}%  OOS {s['oos']*100:+6.1f}% "
              f"(n {s['noos']})  win {s['win']:.2f}  OOS total {mean_of(pooled(res[k]), True)[1]*100:+8.0f}  "
              f"slices +{s['nposs']}/{s['npop']}")
    bi, bo = mean_of(pooled(res["BLEND"]), False), mean_of(pooled(res["BLEND"]), True)
    si = mean_of(pooled(res["SPY"]), False)
    dA = bo[0] - base_o[0]
    pl = [mean_of(pooled(score(pc)), True)[0] - base_o[0] for pc in pcands]
    cell = {k: mean_of(res["BLEND"][k], True)[0] - mean_of(res["SPY"][k], True)[0] for k in CELLS}
    fsign = {f: mean_of(pooled(score(cands["BLEND"], fill=f)), True)[0]
             - mean_of(pooled(score(cands["SPY"], fill=f)), True)[0] for f in FILLS}
    ft = mean_of(pooled(score(cands["BLEND"], cap=1)), True)[0] - mean_of(pooled(score(cands["SPY"], cap=1)), True)[0]
    lo, hi = boot_delta(pooled(res["SPY"]), pooled(res["BLEND"]))
    crit = [("A1 OOS mean delta >= +5pp", dA >= 0.05, f"{dA*100:+.1f}pp  (90% CI {lo*100:+.1f}..{hi*100:+.1f})"),
            ("A2 IS mean delta >= 0", bi[0] - si[0] >= 0, f"{(bi[0]-si[0])*100:+.1f}pp"),
            ("A3 OOS total delta >= 0", bo[1] - base_o[1] >= 0, f"{(bo[1]-base_o[1])*100:+.0f}"),
            ("A4 beats p95 of 20 placebos", dA > np.percentile(pl, 95),
             f"placebo p50 {np.median(pl)*100:+.1f} p95 {np.percentile(pl, 95)*100:+.1f}pp"),
            ("A5 OOS delta > 0 in >= 4/6 cells", sum(v > 0 for v in cell.values()) >= 4,
             " ".join(f"{k[0][0]}{k[1]} {v*100:+.0f}" for k, v in cell.items())),
            ("A6 sign holds mid/bot/worst", all((v > 0) == (dA > 0) for v in fsign.values()) and dA > 0,
             " ".join(f"{f} {v*100:+.1f}" for f, v in fsign.items())),
            ("A7 first-trigger OOS delta > 0", ft > 0, f"{ft*100:+.1f}pp")]
    for lbl, okk, note in crit:
        print(f"  {'PASS' if okk else 'FAIL'}  {lbl:34} {note}")
    print(f"  TEST A: {'PASS' if all(c[1] for c in crit) else 'FAIL'}")

    print("\n=== TEST B -- SPX flow state as a conditioner on SPY triggers")
    real_state = spx_state({d: dict(zip(g["mod"], g["net"])) for d, g in spx.groupby("date")})
    pstates = []
    for p in perms:
        by = {d: dict(zip(g["mod"], g["net"])) for d, g in spx.groupby("date")}
        pstates.append(spx_state({d: by[p[d]] for d in by}))
    verdicts = {}
    for keep in ("aligned", "opposed"):
        fr = score(filt(cands["SPY"], real_state, keep))
        fo, fi = mean_of(pooled(fr), True), mean_of(pooled(fr), False)
        g = fo[0] - base_o[0]
        pg = [mean_of(pooled(score(filt(cands["SPY"], ps, keep))), True)[0] - base_o[0] for ps in pstates]
        cg = {k: mean_of(fr[k], True)[0] - mean_of(res["SPY"][k], True)[0] for k in CELLS}
        fs = {f: mean_of(pooled(score(filt(cands["SPY"], real_state, keep), fill=f)), True)[0]
              - mean_of(pooled(score(cands["SPY"], fill=f)), True)[0] for f in FILLS}
        f1 = (mean_of(pooled(score(filt(cands["SPY"], real_state, keep), cap=1)), True)[0]
              - mean_of(pooled(score(cands["SPY"], cap=1)), True)[0])
        lo, hi = boot_delta(pooled(res["SPY"]), pooled(fr))
        s = SC.stat(pooled(fr))
        print(f"  KEEP-{keep.upper():8} n={s['n']:5}  IS {s['is_']*100:+6.1f}%  OOS {s['oos']*100:+6.1f}% "
              f"(n {s['noos']})  win {s['win']:.2f}  OOS total {fo[1]*100:+8.0f} vs {base_o[1]*100:+8.0f}")
        crit = [("B1 OOS gain >= +5pp", g >= 0.05, f"{g*100:+.1f}pp  (90% CI {lo*100:+.1f}..{hi*100:+.1f})"),
                ("B2 IS gain >= 0", fi[0] - si[0] >= 0, f"{(fi[0]-si[0])*100:+.1f}pp"),
                ("B3 OOS total >= unfiltered", fo[1] >= base_o[1], f"{(fo[1]-base_o[1])*100:+.0f}"),
                ("B4 beats p95 of 20 placebos", g > np.percentile(pg, 95),
                 f"placebo p50 {np.median(pg)*100:+.1f} p95 {np.percentile(pg, 95)*100:+.1f}pp"),
                ("B5 OOS gain > 0 in >= 4/6 cells", sum(v > 0 for v in cg.values()) >= 4,
                 " ".join(f"{k[0][0]}{k[1]} {v*100:+.0f}" for k, v in cg.items())),
                ("B6 sign holds mid/bot/worst", all((v > 0) == (g > 0) for v in fs.values()) and g > 0,
                 " ".join(f"{f} {v*100:+.1f}" for f, v in fs.items())),
                ("B7 first-trigger OOS gain > 0", f1 > 0, f"{f1*100:+.1f}pp")]
        for lbl, okk, note in crit:
            print(f"    {'PASS' if okk else 'FAIL'}  {lbl:34} {note}")
        verdicts[keep] = all(c[1] for c in crit)
    nb = sum(verdicts.values())
    print(f"  TEST B: {'PASS (' + [k for k, v in verdicts.items() if v][0] + ')' if nb == 1 else ('FAIL -- both passed: machinery suspect' if nb == 2 else 'FAIL')}")

    print("\n=== REPORT -- deployed SPY CHOP CALL rule under each arm (its own gates and exit)")
    from config import RULES
    rule = next(r for r in RULES if r["name"] == "SPY CHOP CALL")
    for k, f in arms.items():
        trigs = D.triggers_for(f, "SPY")
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        rows = SC.walk(SC.build_candidates(D, rule, trigs=trigs), SC.policy_for(rule),
                       SC.eod_mod(rule), fill="bot", after=AFTER)
        print(SC.line(k, rows))


if __name__ == "__main__":
    main()
