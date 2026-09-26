"""
iron_condor_backtester.py
=========================

Validates the L3 / short-premium idea (currently TastyTrade mechanics taken on
faith) as a DEFINED-RISK iron condor, against the Silver options lake.

For each trading day it checks a volatility regime (IV percentile + VRP + Hurst),
and when it fires, opens a 30-DTE 16-delta iron condor, then walks it forward on
end-of-day marks applying: 50% take-profit / 2x-credit stop / 21-DTE time stop.
Sweeps a grid of (DTE, short delta, wing width, TP, SL) and reports expectancy.

Key inputs (all already in the lake -- nothing to add):
  - silver option-contracts-1m : per-contract EOD close / bid_close / ask_close /
                                 delta_close / iv_close / underlying_close
  - historical/<TICKER>.parquet : 1-min underlying bars -> daily realized vol, Hurst

Caveats:
  - 30-DTE trades opened after ~mid-July can't complete in the 5-month lake, so
    the effective window is ~Mar-Jul. Regime D may be rare -> thin sample.
  - IV "percentile" is over a trailing ~60-day window (the lake is 5 months), not
    52 weeks. Computed identically to how live should compute it.
  - Entry uses the trigger day's EOD marks (you'd really enter next open) -- ~1
    day optimistic.

Usage:
    python iron_condor_backtester.py                         # full matrix, all tickers
    python iron_condor_backtester.py --tickers SPY NVDA
    python iron_condor_backtester.py --rebuild               # re-scan the lake (slow, ~mins)
    python iron_condor_backtester.py --audit 30 0.16 0.04 0.50 2.0   # dump one cell's trades to CSV
    python iron_condor_backtester.py --lake <dir> --hist <dir>
"""

import os
import sys
import glob
import argparse
import datetime as dt
from collections import defaultdict

import pandas as pd

try:
    import polars as pl
except ImportError:
    sys.exit("polars is required to read the lake.")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DEFAULT_LAKE = "lake/silver/option-contracts-1m"
DEFAULT_HIST = "historical"
SNAP_CACHE = "ic_eod_snapshots.parquet"

WATCHLIST = ["NVDA", "AAPL", "MSFT", "META", "AMZN", "SPY", "GOOGL", "TSLA"]

# --- strategy grid ---
DTE_GRID        = [30, 45]
SHORT_DELTA_GRID = [0.10, 0.16, 0.20, 0.30]
WING_PCT_GRID   = [0.02, 0.04]      # long strike this far (x spot) beyond the short
TP_GRID         = [0.25, 0.50]      # close when spread value has decayed this fraction of credit
SL_GRID         = [2.0, 3.0]        # close when spread value reaches this multiple of credit
TIME_STOP_DTE   = 21

# --- regime gate (entry filter) ---
IV_PCT_WINDOW = 60      # trailing trading days for the IV percentile
IV_PCT_MIN    = 60.0    # today's ATM IV must be >= this percentile of the window
VRP_MIN       = 2.0     # IV% - RV% must exceed this (vol points)
HURST_MAX     = 0.55    # block when the underlying is trending (Hurst > this)
RV_WINDOW     = 20      # trading days for realized vol
HURST_WINDOW  = 60

MIN_TRADES_REPORT = 5
WIN_HURDLE_PCT = 0.0    # pnl > 0 counts as a win (credit strategy: any profit)


# --------------------------------------------------------------------------
# 1. EOD option snapshots
# --------------------------------------------------------------------------
KEEP = ["underlying_symbol", "option_chain_id", "option_type", "strike", "expiry",
        "close", "bid_close", "ask_close", "delta_close", "iv_close",
        "underlying_close", "volume", "open_interest"]


def build_eod_snapshots(lake_dir, tickers, cache_path, rebuild=False):
    if os.path.exists(cache_path) and not rebuild:
        df = pd.read_parquet(cache_path)
        df = df[df["underlying_symbol"].isin(tickers)]
        print(f"  loaded {len(df):,} EOD snapshots from {cache_path}")
        return df

    parts = sorted(glob.glob(os.path.join(lake_dir, "date=*", "bars.parquet")))
    if not parts:
        # maybe lake_dir already points at the bars, or a flat set
        parts = sorted(glob.glob(os.path.join(lake_dir, "*.parquet")))
    if not parts:
        sys.exit(f"no parquet partitions under {lake_dir}")

    frames = []
    for i, p in enumerate(parts, 1):
        date_str = None
        for seg in p.split(os.sep):
            if seg.startswith("date="):
                date_str = seg.split("=", 1)[1]
        lf = (
            pl.scan_parquet(p)
            .filter(pl.col("underlying_symbol").is_in(tickers))
            .sort("minute_et")
            .group_by("option_chain_id", maintain_order=True)
            .agg([pl.col(c).last() for c in KEEP if c != "option_chain_id"])
        )
        d = lf.collect()
        if len(d) == 0:
            continue
        pdf = d.to_pandas()
        pdf["date"] = pd.to_datetime(date_str).date() if date_str else pdf.get("minute_et")
        frames.append(pdf)
        if i % 20 == 0:
            print(f"  ... {i}/{len(parts)} partitions")

    df = pd.concat(frames, ignore_index=True)
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df.to_parquet(cache_path, index=False)
    print(f"  built {len(df):,} EOD snapshots -> {cache_path}")
    return df


# --------------------------------------------------------------------------
# 2. underlying-derived regime signals
# --------------------------------------------------------------------------
def load_underlying_daily(hist_dir, ticker):
    p = os.path.join(hist_dir, f"{ticker}.parquet")
    if not os.path.exists(p):
        return None
    u = pd.read_parquet(p)
    u.columns = [c.lower() for c in u.columns]
    tcol = "start_time" if "start_time" in u.columns else "timestamp"
    u["ts"] = pd.to_datetime(u[tcol], utc=True)
    u = u.sort_values("ts")
    u["d"] = u["ts"].dt.tz_convert("America/New_York").dt.date
    daily = u.groupby("d")["close"].last()
    return daily


def realized_vol(daily_close: pd.Series, asof, window):
    import numpy as np
    s = daily_close[daily_close.index <= asof].tail(window + 1)
    if len(s) < max(5, window // 2):
        return None
    lr = np.log(s / s.shift(1)).dropna()
    if len(lr) < 5:
        return None
    return float(lr.std(ddof=1) * (252 ** 0.5) * 100.0)


def hurst_exponent(daily_close: pd.Series, asof, window):
    import numpy as np
    s = daily_close[daily_close.index <= asof].tail(window + 1)
    lr = np.log(s / s.shift(1)).dropna().values
    if len(lr) < 20:
        return None
    lags = range(2, min(20, len(lr) // 2))
    tau = []
    for lag in lags:
        diff = lr[lag:] - lr[:-lag]
        tau.append(np.sqrt(np.std(diff)))
    tau = np.array(tau)
    lags = np.array(list(lags))
    if np.any(tau <= 0):
        return None
    slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return float(slope * 2.0)


# --------------------------------------------------------------------------
# 3. condor construction & simulation
# --------------------------------------------------------------------------
def _mark(row):
    b, a = row["bid_close"], row["ask_close"]
    if b and a and b > 0 and a > 0:
        return (b + a) / 2.0
    return row["close"]


def pick_expiry(day_df, entry_date, target_dte):
    exps = sorted(day_df["expiry"].unique())
    cand = [(e, (e - entry_date).days) for e in exps if (e - entry_date).days >= 7]
    if not cand:
        return None
    return min(cand, key=lambda x: abs(x[1] - target_dte))[0]


def nearest_by_delta(rows, target_signed_delta):
    rows = rows[(rows["bid_close"] > 0) & rows["delta_close"].notna()]
    if rows.empty:
        return None
    idx = (rows["delta_close"] - target_signed_delta).abs().idxmin()
    return rows.loc[idx]


def nearest_by_strike(rows, target_strike):
    rows = rows[rows["bid_close"] > 0]
    if rows.empty:
        return None
    idx = (rows["strike"] - target_strike).abs().idxmin()
    return rows.loc[idx]


def build_condor(day_df, entry_date, spot, dte, short_delta, wing_pct):
    exp = pick_expiry(day_df, entry_date, dte)
    if exp is None:
        return None
    chain = day_df[day_df["expiry"] == exp]
    calls = chain[chain["option_type"] == "call"]
    puts = chain[chain["option_type"] == "put"]

    sc = nearest_by_delta(calls, short_delta)
    sp = nearest_by_delta(puts, -short_delta)
    if sc is None or sp is None:
        return None

    wing = wing_pct * spot
    lc = nearest_by_strike(calls[calls["strike"] > sc["strike"]], sc["strike"] + wing)
    lp = nearest_by_strike(puts[puts["strike"] < sp["strike"]], sp["strike"] - wing)
    if lc is None or lp is None:
        return None

    # entry credit: sell shorts at bid, buy wings at ask (conservative fills)
    credit = (sc["bid_close"] + sp["bid_close"]) - (lc["ask_close"] + lp["ask_close"])
    if credit <= 0.05:
        return None

    call_width = lc["strike"] - sc["strike"]
    put_width = sp["strike"] - lp["strike"]
    max_loss = max(call_width, put_width) - credit
    if max_loss <= 0:
        return None

    return {
        "expiry": exp,
        "legs": {"sc": sc["option_chain_id"], "sp": sp["option_chain_id"],
                 "lc": lc["option_chain_id"], "lp": lp["option_chain_id"]},
        "strikes": {"sc": sc["strike"], "sp": sp["strike"],
                    "lc": lc["strike"], "lp": lp["strike"]},
        "credit": float(credit),
        "max_loss": float(max_loss),
        "call_width": float(call_width),
        "put_width": float(put_width),
        "short_call_delta": float(sc["delta_close"]),
        "short_put_delta": float(sp["delta_close"]),
    }


def spread_value_on_date(snap_by_date, d, legs):
    """Cost to close the condor on date d (per share). None if we can't mark it."""
    day = snap_by_date.get(d)
    if day is None:
        return None
    marks = {}
    for k, cid in legs.items():
        row = day.get(cid)
        if row is None:
            return None
        marks[k] = _mark(row)
    return (marks["sc"] + marks["sp"]) - (marks["lc"] + marks["lp"])


def settle_at_expiry(under_px, strikes, credit):
    sc_i = max(0.0, under_px - strikes["sc"])
    lc_i = max(0.0, under_px - strikes["lc"])
    sp_i = max(0.0, strikes["sp"] - under_px)
    lp_i = max(0.0, strikes["lp"] - under_px)
    return (sc_i + sp_i) - (lc_i + lp_i)  # cost to close = intrinsic of the spread


def simulate(condor, entry_date, fwd_dates, snap_by_date, underlying_daily, tp, sl):
    credit = condor["credit"]
    legs = condor["legs"]
    tp_target = credit * (1.0 - tp)
    sl_target = credit * sl

    for d in fwd_dates:
        dte_left = (condor["expiry"] - d).days
        val = spread_value_on_date(snap_by_date, d, legs)

        if d >= condor["expiry"] or val is None and dte_left <= 0:
            under_px = float(underlying_daily.get(condor["expiry"], underlying_daily.iloc[-1]))
            val = settle_at_expiry(under_px, condor["strikes"], credit)
            return _close(credit, val, condor, entry_date, d, "EXPIRY")

        if val is None:
            continue  # untradeable that day, carry on

        if val <= tp_target:
            return _close(credit, val, condor, entry_date, d, "TP")
        if val >= sl_target:
            return _close(credit, min(val, credit + condor["max_loss"]), condor, entry_date, d, "SL")
        if dte_left <= TIME_STOP_DTE:
            return _close(credit, val, condor, entry_date, d, "TIME")

    # ran out of lake data before any exit
    return None


def _close(credit, exit_val, condor, entry_date, exit_date, reason):
    pnl = credit - exit_val
    pnl = max(-condor["max_loss"], min(credit, pnl))
    return {
        "entry_date": entry_date, "exit_date": exit_date, "reason": reason,
        "credit": credit, "exit_value": exit_val,
        "pnl_per_share": pnl, "pnl_usd": pnl * 100.0,
        "pnl_pct_of_credit": pnl / credit if credit else 0.0,
        "max_loss": condor["max_loss"], "hold_days": (exit_date - entry_date).days,
        "call_delta": condor["short_call_delta"], "put_delta": condor["short_put_delta"],
    }


# --------------------------------------------------------------------------
# 4. driver
# --------------------------------------------------------------------------
def atm_iv_for_day(day_df, spot, target_dte, entry_date):
    exp = pick_expiry(day_df, entry_date, target_dte)
    if exp is None:
        return None
    chain = day_df[(day_df["expiry"] == exp) & (day_df["iv_close"] > 0)]
    if chain.empty:
        return None
    ivs = []
    for ot in ("call", "put"):
        side = chain[chain["option_type"] == ot]
        if side.empty:
            continue
        row = side.loc[(side["strike"] - spot).abs().idxmin()]
        ivs.append(row["iv_close"])
    return float(sum(ivs) / len(ivs)) if ivs else None


def run(args):
    tickers = [t.upper() for t in args.tickers]
    snap = build_eod_snapshots(args.lake, tickers, args.cache, rebuild=args.rebuild)

    # index: {ticker: {date: {chain_id: row_dict}}} and per-ticker sorted date list
    by_tk = {}
    for tk, g in snap.groupby("underlying_symbol"):
        bd = {}
        for d, gd in g.groupby("date"):
            bd[d] = {r["option_chain_id"]: r for _, r in gd.iterrows()}
        by_tk[tk] = {"by_date": bd, "dates": sorted(bd.keys())}

    grid = [(dte, sd, wp, tp, sl)
            for dte in DTE_GRID for sd in SHORT_DELTA_GRID for wp in WING_PCT_GRID
            for tp in TP_GRID for sl in SL_GRID]
    results = {cell: [] for cell in grid}
    all_trades = []  # for --audit

    audit_cell = None
    if args.audit:
        a = args.audit
        audit_cell = (int(a[0]), round(a[1], 2), round(a[2], 2), round(a[3], 2), round(a[4], 1))

    for tk in tickers:
        if tk not in by_tk:
            print(f"  {tk}: no lake data"); continue
        daily = load_underlying_daily(args.hist, tk)
        if daily is None:
            print(f"  {tk}: no historical/{tk}.parquet"); continue

        dates = by_tk[tk]["dates"]
        by_date = by_tk[tk]["by_date"]

        # rolling ATM-IV series (30d anchor) for the percentile
        iv_hist = {}
        for d in dates:
            day_df = pd.DataFrame(list(by_date[d].values()))
            if day_df.empty:
                continue
            spot = float(day_df["underlying_close"].median())
            iv = atm_iv_for_day(day_df, spot, 30, d)
            if iv:
                iv_hist[d] = iv * 100.0

        # Each grid cell is an independent strategy variant with its own
        # non-overlapping trade calendar (one open condor at a time per cell).
        cell_open_until = {cell: None for cell in grid}
        for i, d in enumerate(dates):
            if d not in iv_hist:
                continue
            # regime signals
            hist_ivs = [iv_hist[x] for x in dates[:i + 1] if x in iv_hist][-IV_PCT_WINDOW:]
            if len(hist_ivs) < 15:
                continue
            cur_iv = iv_hist[d]
            iv_pct = 100.0 * sum(1 for v in hist_ivs if v <= cur_iv) / len(hist_ivs)
            rv = realized_vol(daily, d, RV_WINDOW)
            hurst = hurst_exponent(daily, d, HURST_WINDOW)
            if rv is None:
                continue
            vrp = cur_iv - rv
            if not (iv_pct >= IV_PCT_MIN and vrp >= VRP_MIN):
                continue
            if hurst is not None and hurst > HURST_MAX:
                continue

            day_df = pd.DataFrame(list(by_date[d].values()))
            spot = float(day_df["underlying_close"].median())
            fwd = dates[i + 1:]
            if len(fwd) < 5:
                continue

            for cell in grid:
                if cell_open_until[cell] and d <= cell_open_until[cell]:
                    continue
                dte, sd, wp, tp, sl = cell
                condor = build_condor(day_df, d, spot, dte, sd, wp)
                if condor is None:
                    continue
                res = simulate(condor, d, fwd, by_date, daily, tp, sl)
                if res is None:
                    continue
                res.update(ticker=tk, iv_pct=iv_pct, vrp=vrp, hurst=hurst, cell=cell)
                results[cell].append(res)
                cell_open_until[cell] = res["exit_date"]
                if audit_cell and cell == audit_cell:
                    all_trades.append(res)

    report(results, tickers)
    if audit_cell:
        dump_audit(all_trades, audit_cell)


def report(results, tickers):
    print("\n" + "=" * 100)
    print(f"  IRON CONDOR MATRIX  |  {', '.join(tickers)}  |  gate: IVpct>={IV_PCT_MIN} VRP>={VRP_MIN} Hurst<={HURST_MAX}")
    print("=" * 100)
    print(f"{'DTE':>4} {'Δshort':>7} {'wing%':>6} {'TP':>5} {'SL':>4} | "
          f"{'n':>4} {'win%':>6} {'avgW$':>8} {'avgL$':>9} {'exp$':>8} {'tot$':>10} "
          f"{'cr$':>7} {'maxL$':>8} {'days':>5} | TP/SL/TIME/EXP")
    print("-" * 100)
    rows = []
    for cell, trades in results.items():
        if len(trades) < MIN_TRADES_REPORT:
            continue
        dte, sd, wp, tp, sl = cell
        wins = [t for t in trades if t["pnl_usd"] > WIN_HURDLE_PCT]
        losses = [t for t in trades if t["pnl_usd"] <= WIN_HURDLE_PCT]
        n = len(trades)
        wr = 100.0 * len(wins) / n
        avg_w = sum(t["pnl_usd"] for t in wins) / len(wins) if wins else 0.0
        avg_l = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0.0
        tot = sum(t["pnl_usd"] for t in trades)
        exp = tot / n
        cr = sum(t["credit"] for t in trades) / n * 100
        ml = sum(t["max_loss"] for t in trades) / n * 100
        days = sum(t["hold_days"] for t in trades) / n
        rc = defaultdict(int)
        for t in trades:
            rc[t["reason"]] += 1
        rows.append((exp, cell, n, wr, avg_w, avg_l, exp, tot, cr, ml, days, rc))

    for exp, cell, n, wr, avg_w, avg_l, e, tot, cr, ml, days, rc in sorted(rows, key=lambda x: x[0], reverse=True):
        dte, sd, wp, tp, sl = cell
        print(f"{dte:>4} {sd:>7.2f} {wp*100:>5.0f}% {tp:>5.2f} {sl:>4.1f} | "
              f"{n:>4} {wr:>5.1f}% {avg_w:>8.0f} {avg_l:>9.0f} {e:>8.0f} {tot:>10.0f} "
              f"{cr:>7.0f} {ml:>8.0f} {days:>5.0f} | "
              f"{rc['TP']}/{rc['SL']}/{rc['TIME']}/{rc['EXPIRY']}")

    if not rows:
        print("  No cells with enough trades. The regime gate may be too tight, or the")
        print("  lake window too short. Try lowering IV_PCT_MIN / VRP_MIN.")
    print()


def dump_audit(trades, cell):
    if not trades:
        print(f"\n  --audit {cell}: no trades in that cell.")
        return
    df = pd.DataFrame(trades)
    fn = f"ic_audit_{cell[0]}dte_{cell[1]}d_{int(cell[2]*100)}w_{cell[3]}tp_{cell[4]}sl.csv"
    df.to_csv(fn, index=False)
    print(f"\n  audit -> {fn}  ({len(df)} trades)")
    print(f"  win rate {100*(df.pnl_usd>0).mean():.1f}%  expectancy ${df.pnl_usd.mean():.0f}  "
          f"total ${df.pnl_usd.sum():.0f}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=WATCHLIST)
    ap.add_argument("--lake", default=DEFAULT_LAKE)
    ap.add_argument("--hist", default=DEFAULT_HIST)
    ap.add_argument("--cache", default=SNAP_CACHE)
    ap.add_argument("--rebuild", action="store_true", help="re-scan the lake for EOD snapshots")
    ap.add_argument("--audit", nargs=5, type=float, metavar=("DTE", "DELTA", "WINGPCT", "TP", "SL"),
                    help="dump one cell's trades to CSV, e.g. --audit 30 0.16 0.04 0.5 2.0")
    args = ap.parse_args()
    run(args)
