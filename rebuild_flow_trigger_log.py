"""
rebuild_flow_trigger_log.py
===========================
Rebuilds flow_trigger_log.jsonl -- the history the live flow gate calibrates
on -- from clean, today-only net-premium history.

🚨 WHY IT HAS TO BE REBUILT (2026-09-27)
    The live bot seeded each day's cumulative flow from net-prem-ticks WITHOUT
    a date. Before a session's first print UW answers that with the PREVIOUS
    completed session, so every live day started from yesterday's full total.
    Measured on 54 ticker-days: the first live crossover each morning equalled
    |previous day's total|, median ratio 1.001 (e.g. SPY 2026-09-17: first
    crossover $61.6M, previous day $61.1M, that day's real total $7.9M).
    Crossover TIMING is untouched by a constant offset; the logged LEVEL is not.
    From 20 distinct days the gate calibrates on this log instead of the static
    JSON -- and it reached exactly 20 on 2026-09-24. The seed is fixed
    (unusual_whales_client._tick_is_today); this repairs the history it wrote.

ONE IMPLEMENTATION, NOT A COPY (METHODOLOGY 1)
    The replay drives bot_runner's own FlowMomentumTracker, minute by minute on
    a patched clock, and writes rows through bot_runner's own _log_flow_trigger,
    so its weekend and frozen-feed guards apply unchanged. The source is
    historical/NETPREM{T}.parquet: net-prem-ticks fetched one date at a time,
    therefore today-only by construction.

    Minutes fed: 09:30..16:04 ET. The live bot now stops polling UW at 16:05
    (uw_market_open), so bar 16:04 is never evaluated live and neither is it
    here. Logged value: |cumulative flow| at the CROSSOVER bar -- the backtest's
    convention. Live logs the first reading after the bar closes, which under a
    15s REST poll is the same number to within one poll.

VALIDATION BEFORE ANYTHING IS WRITTEN
    V1  On days the live bot actually ran, crossovers per ticker-day must match
        the old log's count -- timing was never corrupted, only level. If the
        median ratio is outside 0.8..1.25 the replay is not reproducing the
        tracker and the script refuses to write.
    V2  The thresholds the gate would use on the next session, old log vs
        rebuilt vs the static JSON, printed side by side -- so the size of the
        change is seen before it is deployed.

Usage:  python rebuild_flow_trigger_log.py [--asof 2026-09-28] [--out FILE]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
from collections import deque
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import polars as pl

import bot_runner as B
import config
import uw_options_data_lake as L

NY = ZoneInfo("America/New_York")
FIRST_MOD, LAST_MOD = 9 * 60 + 30, 16 * 60 + 4        # minutes fed, inclusive


def replay(tickers, days, out_path):
    """Write the rebuilt log. Returns {(ticker, date): crossover_count}."""
    if os.path.exists(out_path):
        os.remove(out_path)
    stub = SimpleNamespace(flow_trigger_log_path=out_path, flow_trigger_hist={},
                           _FLOW_HIST_MAXLEN=B.FlowExecutionEngine._FLOW_HIST_MAXLEN)
    counts = {}
    frames = {t: pl.read_parquet(f"historical/NETPREM{t}.parquet",
                                 columns=["date", "minute_et", "net_premium"])
              for t in tickers}
    real_time, real_now = B.time.time, B.market_now
    try:
        for d in days:
            for t in tickers:
                g = (frames[t].filter(pl.col("date") == d)
                     .with_columns(pl.col("minute_et").dt.convert_time_zone("America/New_York"))
                     .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                                    + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
                     .filter(pl.col("mod").is_between(FIRST_MOD, LAST_MOD))
                     .sort("minute_et"))
                if g.is_empty():
                    continue
                cum = g["net_premium"].cum_sum().to_list()
                mins = g["minute_et"].to_list()
                tr = B.FlowMomentumTracker()              # session roll resets it
                n = 0
                for i, (m, c) in enumerate(zip(mins, cum)):
                    B.time.time = lambda m=m: m.timestamp() + 30.0   # mid-minute m
                    sig = tr.update_and_check(t, c)
                    if sig in ("LONG", "SHORT"):
                        bar = cum[i - 1]                   # the bar just evaluated
                        B.market_now = lambda m=m: m
                        B.FlowExecutionEngine._log_flow_trigger(stub, t, abs(bar))
                        n += 1
                counts[(t, d.isoformat())] = n
    finally:
        B.time.time, B.market_now = real_time, real_now
    return counts


def load_log(path):
    hist = {}
    for line in open(path, encoding="utf-8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if dt.date.fromisoformat(e["date"]).weekday() >= 5:
            continue
        hist.setdefault(e["ticker"], deque(maxlen=B.FlowExecutionEngine._FLOW_HIST_MAXLEN)).append(
            (e["date"], float(e["abs_flow"])))
    return hist


def thresholds(hist, asof):
    """What _live_flow_threshold returns for each enabled rule on `asof`."""
    static = {}
    for p in (B.FLOW_THRESHOLDS_PATH,):
        try:
            static = json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    stub = SimpleNamespace(flow_trigger_hist=hist, flow_thresholds=static, _thr_warned=set(),
                           _daily_medians=B.FlowExecutionEngine._daily_medians,
                           _percentile=B.FlowExecutionEngine._percentile)
    real_now = B.market_now
    B.market_now = lambda: dt.datetime(asof.year, asof.month, asof.day, 9, 30, tzinfo=NY)
    out = {}
    try:
        import contextlib, io
        for r in config.RULES:
            if r.get("enabled", True) is False:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                out[r["name"] if "name" in r else r["ticker"]] = (
                    r["ticker"], B.FlowExecutionEngine._live_flow_threshold(stub, r["ticker"], r))
    finally:
        B.market_now = real_now
    return out, static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="first session the log will serve (default: next trading day)")
    ap.add_argument("--out", default="flow_trigger_log.rebuilt.jsonl")
    ap.add_argument("--old", default="flow_trigger_log.jsonl")
    a = ap.parse_args()

    asof = (dt.date.fromisoformat(a.asof) if a.asof
            else L.next_trading_day(dt.datetime.now(NY).date()))
    windows = [r.get("flow_window_days") or 90 for r in config.RULES if r.get("enabled", True) is not False]
    windows += [(r.get("flow_zscore") or {}).get("window_days") or 0 for r in config.RULES]
    span = max(windows)
    days = [d for d in L.trading_days(asof - dt.timedelta(days=span), asof - dt.timedelta(days=1))]
    tickers = list(config.WATCHLIST)
    print(f"serving {asof}; longest rule window {span}d -> {len(days)} sessions "
          f"{days[0]} .. {days[-1]}, {len(tickers)} tickers")

    counts = replay(tickers, days, a.out)
    n_rows = sum(1 for _ in open(a.out, encoding="utf-8"))
    print(f"rebuilt: {n_rows:,} crossovers written to {a.out}")

    # ---------------------------------------------------------------- V1
    old_counts = {}
    for line in open(a.old, encoding="utf-8"):
        e = json.loads(line)
        old_counts[(e["ticker"], e["date"])] = old_counts.get((e["ticker"], e["date"]), 0) + 1
    both = [(k, old_counts[k], counts[k]) for k in old_counts if k in counts and counts[k]]
    ratios = sorted(o / n for _, o, n in both)
    med = statistics.median(ratios) if ratios else float("nan")
    print(f"\nV1 crossovers per ticker-day, old live log vs rebuilt, on {len(both)} "
          f"ticker-days the live bot ran:")
    print(f"   median ratio old/rebuilt {med:.3f}   p10 {ratios[len(ratios)//10]:.2f}   "
          f"p90 {ratios[len(ratios)*9//10]:.2f}")
    if not 0.8 <= med <= 1.25:
        os.remove(a.out)
        sys.exit("   V1 FAILED -- the replay does not reproduce the tracker's crossover "
                 "count. Nothing written.")
    print("   ok -- timing reproduced; only the level differs")

    # ---------------------------------------------------------------- V2
    old_t, static = thresholds(load_log(a.old), asof)
    new_t, _ = thresholds(load_log(a.out), asof)
    print(f"\nV2 gate threshold each enabled rule would use on {asof}:")
    print(f"   {'rule':<34}{'old (inflated) log':>20}{'rebuilt log':>16}{'change':>9}")
    for name, (t, v_old) in old_t.items():
        v_new = new_t[name][1]
        chg = f"{(v_new / v_old - 1) * 100:+.0f}%" if v_old and v_new else "--"
        fmt = lambda v: f"${v / 1e6:,.1f}M" if v else "--"
        print(f"   {name[:33]:<34}{fmt(v_old):>20}{fmt(v_new):>16}{chg:>9}")


if __name__ == "__main__":
    main()
