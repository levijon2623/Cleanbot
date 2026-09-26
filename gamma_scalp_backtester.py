"""
gamma_scalp_backtester.py
=========================

Is there a gamma-scalping edge in the watchlist names once you pay Hyperliquid
perp fees + funding + the option spread + theta?

For each entry day it buys 1 ATM call, then delta-hedges it minute-to-minute
(5-min bars) by shorting the underlying on a 20x perp, rehedging whenever the
delta drift exceeds a band. P&L = option P&L + hedge P&L - perp fees - funding.
Sweeps the rehedge band (this is the whole game) plus fee / funding / regime.

Uses:
  - ic_eod_snapshots.parquet  (built by iron_condor_backtester.py -- run that first,
    or this will build it)  -> contract pick, entry/exit marks, entry IV
  - historical/<TICKER>.parquet -> 1-min underlying, resampled to 5-min

20x LEVERAGE is modelled two ways:
  - it does NOT change the dollar edge (hedge P&L, fees, funding are all on the
    full hedge notional, which is set by the option delta, not by leverage)
  - it DOES shrink the capital deployed: hedge margin = notional / 20, plus a 10%
    drawdown buffer.  ROE is reported on (option premium + hedge margin + buffer).
  - a levered hedge on a separate venue can be LIQUIDATED by a move the options
    are winning on (options aren't margin on Hyperliquid).  Liquidations counted.

Usage:
    python gamma_scalp_backtester.py
    python gamma_scalp_backtester.py --tickers SPY NVDA
    python gamma_scalp_backtester.py --lake <silver dir> --hist <historical dir>
"""

import os
import sys
import argparse
import datetime as dt
from math import log, sqrt, erf, exp, pi

from collections import defaultdict

import numpy as np
import pandas as pd

SNAP_CACHE = "ic_eod_snapshots.parquet"
DEFAULT_LAKE = "lake/silver/option-contracts-1m"
DEFAULT_HIST = "historical"
WATCHLIST = ["NVDA", "AAPL", "MSFT", "META", "AMZN", "SPY", "GOOGL", "TSLA"]

R = 0.045                    # risk-free
ENTRY_DELTA = 0.50           # ATM call
DTE_GRID = [30, 45]
HOLD_TO_DTE = 21             # exit here (the "gamma trap" the engine also uses)
REHEDGE_BAND_GRID = [5, 10, 20, 40]     # rehedge when |delta drift| >= N share-equivs (per contract)
PERP_FEE_GRID = [("maker", 0.0), ("taker", 0.00035)]
FUNDING_MO_GRID = [("nofund", 0.0), ("fund_0.7%/mo", 0.007)]   # monthly funding paid on hedge notional
LEVERAGE = 20
DD_BUFFER = 0.10
REGIMES = ["all", "vrp_neg"]     # vrp_neg = only enter when ATM IV < trailing realized vol
RV_WINDOW = 20
BAR_MIN = 5                  # rehedge/mark cadence


# --------------------------------------------------------------------------
def norm_cdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))
def norm_pdf(x): return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def bs_delta_gamma(S, K, T, iv, is_call):
    T = max(T, 1e-6); iv = max(iv, 0.01)
    d1 = (log(S / K) + (R + 0.5 * iv * iv) * T) / (iv * sqrt(T))
    delta = norm_cdf(d1) if is_call else norm_cdf(d1) - 1.0
    gamma = norm_pdf(d1) / (S * iv * sqrt(T))
    return delta, gamma


# vectorised call delta over a spot array, constant iv (uses a fast erf approx)
def _erf_approx(x):
    # Abramowitz & Stegun 7.1.26
    s = np.sign(x); x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def bs_delta_vec(S, K, T, iv):
    S = np.asarray(S, float)
    T = np.maximum(np.asarray(T, float), 1e-6); iv = max(iv, 0.01)
    d1 = (np.log(S / K) + (R + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
    return 0.5 * (1.0 + _erf_approx(d1 / sqrt(2.0)))


# --------------------------------------------------------------------------
def load_underlying_5m(hist_dir, ticker):
    p = os.path.join(hist_dir, f"{ticker}.parquet")
    if not os.path.exists(p):
        return None
    u = pd.read_parquet(p)
    u.columns = [c.lower() for c in u.columns]
    tcol = "start_time" if "start_time" in u.columns else "timestamp"
    u["t"] = pd.to_datetime(u[tcol], utc=True)
    u = u.set_index("t").sort_index()
    # keep the extended session that's in the data (~4am-8pm ET); resample to 5m
    px = u["close"].resample(f"{BAR_MIN}min").last().dropna()
    return px


def daily_close(px5):
    return px5.groupby(px5.index.tz_convert("America/New_York").date).last()


def realized_vol(dclose, asof, window):
    s = dclose[dclose.index <= asof].tail(window + 1)
    if len(s) < 6:
        return None
    lr = np.log(s / s.shift(1)).dropna()
    return float(lr.std(ddof=1) * sqrt(252) * 100.0)


# --------------------------------------------------------------------------
def rehedge_pass(spot, target_perp, band, fee_rate, funding_mo):
    """spot, target_perp: aligned arrays over the hold.
    Returns hedge_pnl, fees, funding, rehedges, margin, liquidated."""
    n = len(spot)
    perp = np.empty(n)
    perp[0] = target_perp[0]
    fees = abs(perp[0]) * spot[0] * fee_rate
    rehedges = 1
    for i in range(1, n):
        if abs(target_perp[i] - perp[i - 1]) >= band:
            fees += abs(target_perp[i] - perp[i - 1]) * spot[i] * fee_rate
            perp[i] = target_perp[i]
            rehedges += 1
        else:
            perp[i] = perp[i - 1]

    dS = np.diff(spot)
    step_pnl = perp[:-1] * dS
    hedge_pnl = float(step_pnl.sum())

    bars_per_month = 30 * 24 * 60 / BAR_MIN
    funding = float(np.sum(np.abs(perp) * spot)) * (funding_mo / bars_per_month)

    max_notional = float(np.max(np.abs(perp) * spot))
    margin = max_notional / LEVERAGE * (1.0 + DD_BUFFER)
    worst = float(np.min(np.cumsum(step_pnl))) if n > 1 else 0.0
    return {"hedge_pnl": hedge_pnl, "fees": fees, "funding": funding,
            "rehedges": rehedges, "margin": margin, "liquidated": worst < -margin}


# --------------------------------------------------------------------------
def run(args):
    tickers = [t.upper() for t in args.tickers]
    if not os.path.exists(args.cache):
        sys.exit(f"{args.cache} not found -- run iron_condor_backtester.py first (it builds it).")
    snap = pd.read_parquet(args.cache)
    snap = snap[snap["underlying_symbol"].isin(tickers)].copy()
    snap["expiry"] = pd.to_datetime(snap["expiry"]).dt.date

    results = defaultdict(list)   # (dte, band, fee_lbl, fund_lbl, regime) -> [trade dicts]

    for tk in tickers:
        px5 = load_underlying_5m(args.hist, tk)
        if px5 is None:
            print(f"  {tk}: no historical parquet"); continue
        dclose = daily_close(px5)
        g = snap[snap["underlying_symbol"] == tk]
        by_date = {d: gd for d, gd in g.groupby("date")}
        dates = sorted(by_date)
        by_cid = {}
        for d, gd in by_date.items():
            for _, r in gd.iterrows():
                by_cid.setdefault(r["option_chain_id"], {})[d] = r

        for entry_date in dates:
            day = by_date[entry_date]
            calls = day[(day["option_type"] == "call") & (day["bid_close"] > 0) & day["delta_close"].notna()]
            if calls.empty:
                continue
            spot0 = float(day["underlying_close"].median())

            # regime signal
            rv = realized_vol(dclose, entry_date, RV_WINDOW)

            for dte in DTE_GRID:
                # pick expiry ~dte out, then the ATM call
                exps = sorted({e for e in calls["expiry"] if (e - entry_date).days >= HOLD_TO_DTE + 3})
                if not exps:
                    continue
                exp = min(exps, key=lambda e: abs((e - entry_date).days - dte))
                cc = calls[calls["expiry"] == exp]
                if cc.empty:
                    continue
                row = cc.loc[(cc["delta_close"] - ENTRY_DELTA).abs().idxmin()]
                K = float(row["strike"]); iv = float(row["iv_close"])
                if iv <= 0:
                    continue
                entry_ask = float(row["ask_close"]); cid = row["option_chain_id"]

                atm_iv_pct = iv * 100.0
                vrp = (atm_iv_pct - rv) if rv is not None else None

                # exit date = expiry - HOLD_TO_DTE, clipped to lake data
                exit_date = exp - dt.timedelta(days=HOLD_TO_DTE)
                fwd_dates = [d for d in dates if entry_date < d <= exit_date]
                if len(fwd_dates) < 3:
                    continue
                real_exit = fwd_dates[-1]
                exrow = by_cid.get(cid, {}).get(real_exit)
                if exrow is None:
                    # find the last date we have a mark for this contract
                    have = sorted(d for d in by_cid.get(cid, {}) if entry_date < d <= exit_date)
                    if not have:
                        continue
                    real_exit = have[-1]
                    exrow = by_cid[cid][real_exit]
                exit_bid = float(exrow["bid_close"])
                option_pnl = (exit_bid - entry_ask) * 100.0

                # 5-min spot path over the hold
                start_ts = pd.Timestamp(entry_date, tz="America/New_York").tz_convert("UTC")
                end_ts = pd.Timestamp(real_exit, tz="America/New_York").tz_convert("UTC") + pd.Timedelta(days=1)
                path = px5[(px5.index >= start_ts) & (px5.index < end_ts)]
                if len(path) < 20:
                    continue
                spot = path.values.astype(float)
                exp_ts = pd.Timestamp(exp, tz="America/New_York").tz_convert("UTC") + pd.Timedelta(hours=20)
                tgrid = np.maximum((exp_ts - path.index).total_seconds().values / (365.25 * 24 * 3600), 1e-6)

                # delta path (shares) is fixed for this position; only band/fee/funding vary
                target_perp = -(bs_delta_vec(spot, K, tgrid, iv) * 100.0)

                active_regimes = ["all"] + (["vrp_neg"] if (vrp is not None and vrp < 0) else [])
                for band in REHEDGE_BAND_GRID:
                    for (fee_lbl, fee_v) in PERP_FEE_GRID:
                        for (fund_lbl, fund_v) in FUNDING_MO_GRID:
                            hh = rehedge_pass(spot, target_perp, band, fee_v, fund_v)
                            total = option_pnl + hh["hedge_pnl"] - hh["fees"] - hh["funding"]
                            capital = entry_ask * 100.0 + hh["margin"]
                            rec = {
                                "ticker": tk, "entry": entry_date, "exit": real_exit,
                                "option_pnl": option_pnl, "hedge_pnl": hh["hedge_pnl"],
                                "fees": hh["fees"], "funding": hh["funding"],
                                "total": total, "roe": total / capital if capital else 0,
                                "rehedges": hh["rehedges"], "liq": hh["liquidated"],
                                "hold_days": (real_exit - entry_date).days,
                                "vrp": vrp, "atm_iv": atm_iv_pct,
                            }
                            for rg in active_regimes:
                                results[(dte, band, fee_lbl, fund_lbl, rg)].append(rec)
    report(results, tickers)


def report(results, tickers):
    print("\n" + "=" * 108)
    print(f"  GAMMA SCALP MATRIX  |  {', '.join(tickers)}  |  1 ATM call, 5-min rehedge, 20x perp hedge, exit @ {HOLD_TO_DTE} DTE")
    print("=" * 108)
    print(f"{'DTE':>4} {'band':>5} {'fee':>6} {'funding':>12} {'regime':>8} | "
          f"{'n':>4} {'win%':>6} {'tot$':>8} {'opt$':>8} {'hedge$':>9} {'fee$':>7} {'fund$':>8} "
          f"{'ROE%':>7} {'rehdg':>6} {'liq':>4} {'days':>5}")
    print("-" * 108)
    rows = []
    for key, trades in results.items():
        if len(trades) < 5:
            continue
        dte, band, fee, fund, rg = key
        n = len(trades)
        tot = np.mean([t["total"] for t in trades])
        rows.append((tot, key, n, trades))
    for tot, key, n, trades in sorted(rows, key=lambda x: x[0], reverse=True):
        dte, band, fee, fund, rg = key
        win = 100 * np.mean([t["total"] > 0 for t in trades])
        opt = np.mean([t["option_pnl"] for t in trades])
        hdg = np.mean([t["hedge_pnl"] for t in trades])
        fe = np.mean([t["fees"] for t in trades])
        fu = np.mean([t["funding"] for t in trades])
        roe = 100 * np.mean([t["roe"] for t in trades])
        rh = np.mean([t["rehedges"] for t in trades])
        liq = sum(t["liq"] for t in trades)
        dd = np.mean([t["hold_days"] for t in trades])
        print(f"{dte:>4} {band:>5} {fee:>6} {fund:>12} {rg:>8} | "
              f"{n:>4} {win:>5.0f}% {tot:>8.0f} {opt:>8.0f} {hdg:>9.0f} {fe:>7.0f} {fu:>8.0f} "
              f"{roe:>6.1f}% {rh:>6.0f} {liq:>4} {dd:>5.0f}")
    if not rows:
        print("  no cells with >= 5 trades")
    print()
    print("  tot$ = option_pnl + hedge_pnl - fees - funding, per 1-lot.  A working gamma scalp")
    print("  needs tot$ > 0 consistently (it is NOT enough for option_pnl alone to be positive --")
    print("  that's just being long vol).  Watch the liq column: a levered hedge liquidated by an")
    print("  adverse move leaves you naked directional.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=WATCHLIST)
    ap.add_argument("--lake", default=DEFAULT_LAKE)
    ap.add_argument("--hist", default=DEFAULT_HIST)
    ap.add_argument("--cache", default=SNAP_CACHE)
    args = ap.parse_args()
    run(args)
