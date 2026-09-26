# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0"]
# ///
"""
check_gamma_walls.py
====================

Derive an intraday dealer-gamma-by-strike profile for a ticker from the silver
option tape and assess whether call/put "walls" and a gamma-flip level are
identifiable and stable enough to build a 0DTE credit-spread strategy on. If the
walls are noisy / the OI coverage is thin, the rest of the idea is moot.

Per trading day, over the DTE window (default 0-7):
  - each contract's daily OI  = last non-null open_interest that day
  - each contract's gamma     = last gamma_close observed at/before each snapshot
                                (forward-filled from real trades; --bs-fill
                                recomputes from iv_close for uncovered contracts)
  - every --step minutes:  strike_gamma = SUM  sign * gamma * OI * spot^2
      sign = +1 call, -1 put   (standard: dealers long call gamma, short put gamma)
  - call wall  = strike above spot with the largest +gamma
    put wall   = strike below spot with the largest -gamma
    gamma flip = strike level where cumulative net gamma crosses zero

Reports, per day and pooled:
  - open vs close wall / flip drift (stability)
  - does the [put wall, call wall] band from the OPEN contain the day's range?
  - "respect": did an intraday extreme reach a wall (+-touch_tol) then close back inside?
  - OI coverage: share of DTE-window OI that has a same-day gamma observation
  - --xcheck: derived net gamma vs lake spot-exposures-1m gamma_per_one_percent_move_oi
  - --dump PATH: full (snapshot x strike) grid to CSV for one date

Usage:
  python check_gamma_walls.py SPY 2026-08-18 2026-08-21
  python check_gamma_walls.py QQQ 2026-08-21 --step 15 --max-dte 3
  python check_gamma_walls.py SPY 2026-08-21 --dump gw_SPY_0821.csv --xcheck
  python check_gamma_walls.py QQQ,IWM 2026-06-16 2026-08-21 --range --max-dte 2
  python check_gamma_walls.py SPY 2026-06-16 2026-08-21 --range --max-dte 2 --vol-filter LOWVOL
  python check_gamma_walls.py QQQ,IWM 2026-06-16 2026-08-21 --range --max-dte 2 --reject-test
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from datetime import date, datetime

import numpy as np
import polars as pl

LAKE = "lake"
BARS = "silver/option-contracts-1m"
SPOT_GEX = "silver/spot-exposures-1m"
RTH_OPEN = 9 * 60 + 30
RTH_CLOSE = 16 * 60
R_FREE = 0.045


def _rth_min(df: pl.DataFrame, col: str = "minute_et") -> pl.DataFrame:
    return df.with_columns(
        (pl.col(col).dt.hour().cast(pl.Int32) * 60 + pl.col(col).dt.minute().cast(pl.Int32)).alias("_mod")
    ).filter((pl.col("_mod") >= RTH_OPEN) & (pl.col("_mod") <= RTH_CLOSE))


def _bs_gamma(spot, strike, t_years, iv):
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (R_FREE + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (spot * iv * math.sqrt(t_years))


def load_day(ticker: str, d: date, max_dte: int, min_dte: int) -> pl.DataFrame | None:
    p = os.path.join(LAKE, BARS, f"date={d.isoformat()}", "bars.parquet")
    if not os.path.exists(p):
        return None
    lf = (pl.scan_parquet(p)
          .filter(pl.col("underlying_symbol") == ticker)
          .select("option_chain_id", "option_type", "strike", "expiry", "minute_et",
                  "gamma_close", "iv_close", "open_interest", "underlying_close",
                  "bid_close", "ask_close"))
    df = _rth_min(lf.collect())
    if df.is_empty():
        return None
    df = df.with_columns(
        (pl.col("expiry").cast(pl.Date) - pl.lit(d)).dt.total_days().alias("dte"),
        pl.col("strike").cast(pl.Float64),
    ).filter((pl.col("dte") >= min_dte) & (pl.col("dte") <= max_dte))
    return df if not df.is_empty() else None


def snapshots(step: int) -> list[int]:
    return list(range(RTH_OPEN + step, RTH_CLOSE + 1, step))


def profile_at(day: pl.DataFrame, mod: int, bs_fill: bool):
    """dealer gamma by strike as of minute-of-day `mod`. Returns
    (spot, {strike: net_gamma}, covered_oi, total_oi)."""
    upto = day.filter(pl.col("_mod") <= mod)
    if upto.is_empty():
        return None
    spot = float(upto.sort("_mod").select("underlying_close").tail(50).median().item() or 0.0)
    if spot <= 0:
        return None
    # per contract: OI (last seen all day), last gamma <= mod, last iv, strike, type, dte
    meta = (day.sort("_mod")
            .group_by("option_chain_id")
            .agg(pl.col("open_interest").drop_nulls().last().alias("oi"),
                 pl.col("strike").first(), pl.col("option_type").first(),
                 pl.col("dte").first(),
                 pl.col("iv_close").drop_nulls().last().alias("iv")))
    g = (upto.sort("_mod").group_by("option_chain_id")
         .agg(pl.col("gamma_close").drop_nulls().last().alias("gamma")))
    c = meta.join(g, on="option_chain_id", how="left").filter(pl.col("oi") > 0)
    total_oi = float(c.select(pl.col("oi").sum()).item() or 0.0)
    covered_oi = float(c.filter(pl.col("gamma").is_not_null()).select(pl.col("oi").sum()).item() or 0.0)

    rows = c.to_dicts()
    by_strike: dict[float, float] = {}
    for r in rows:
        gam = r["gamma"]
        if gam is None:
            if not bs_fill or r["iv"] is None:
                continue
            t_years = max(r["dte"], 0.5) / 365.0
            gam = _bs_gamma(spot, r["strike"], t_years, float(r["iv"]))
        sign = 1.0 if r["option_type"] == "call" else -1.0
        by_strike[r["strike"]] = by_strike.get(r["strike"], 0.0) + sign * gam * r["oi"] * spot * spot
    return spot, by_strike, covered_oi, total_oi


def walls(spot: float, by_strike: dict[float, float]):
    """(call_wall, put_wall, net_gamma_sign). call_wall = strike >= spot with the
    largest +gamma; put_wall = strike < spot with the largest -gamma. net sign of
    the whole profile: + = suppression regime, - = amplification. (No gamma-flip
    LEVEL -- that needs the full all-expiry chain the tape doesn't fully carry;
    use spot-exposures-1m for the regime read.)"""
    if not by_strike:
        return None
    above = [(k, v) for k, v in by_strike.items() if k >= spot]
    below = [(k, v) for k, v in by_strike.items() if k < spot]
    call_wall = max(above, key=lambda x: x[1])[0] if above and max(x[1] for x in above) > 0 else None
    put_wall = min(below, key=lambda x: x[1])[0] if below and min(x[1] for x in below) < 0 else None
    net_sign = "+" if sum(by_strike.values()) >= 0 else "-"
    return call_wall, put_wall, net_sign


# --------------------------------------------------------------------------
# --reject-test: your actual play -- let the morning range establish, watch
# for price to TEST a wall between --window, check RSI at the touch, then see
# whether the wall holds (rejection) or breaks for the rest of the day.
# --------------------------------------------------------------------------
def _minute_px(day: pl.DataFrame, lo: int | None = None, hi: int | None = None) -> pl.DataFrame:
    d = day
    if lo is not None:
        d = d.filter(pl.col("_mod") >= lo)
    if hi is not None:
        d = d.filter(pl.col("_mod") <= hi)
    return (d.group_by("_mod").agg(pl.col("underlying_close").median().alias("px"))
            .sort("_mod"))


def rsi_series(day: pl.DataFrame, period: int = 14, bar_min: int = 5):
    """Wilder RSI(period) on bar_min-bucketed underlying closes.
    Returns [(bucket_mod, close, rsi_or_nan), ...] sorted by bucket."""
    px = _minute_px(day).with_columns((pl.col("_mod") // bar_min * bar_min).alias("b"))
    bars = px.group_by("b").agg(pl.col("px").last()).sort("b")
    buckets, closes = bars["b"].to_list(), bars["px"].to_list()
    n = len(closes)
    if n <= period:
        return list(zip(buckets, closes, [float("nan")] * n))
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rsi = [float("nan")] * (period)
    rsi.append(100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss))
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi.append(100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss))
    return list(zip(buckets, closes, rsi))


def rsi_at(series, mod: int) -> float:
    val = float("nan")
    for b, _, r in series:
        if b > mod:
            break
        val = r
    return val


def run_reject(args, tk: str, days: list[date]):
    wh, wm = map(int, args.wall_asof.split(":"))
    wall_mod = wh * 60 + wm
    (w0h, w0m), (w1h, w1m) = (map(int, s.split(":")) for s in args.window.split("-"))
    win_lo, win_hi = w0h * 60 + w0m, w1h * 60 + w1m
    tol = args.touch_tol

    try:
        from directional_flow_backtester import load_trend_regime
        trd = load_trend_regime(args.hist, tk)
    except Exception:
        trd = {}

    print(f"{tk}  wall-asof {args.wall_asof}  entry-window {args.window}  touch-tol ${tol}  "
          f"reset {args.reset_frac:.2%}  dte {args.min_dte}-{args.max_dte}\n")

    ent = []   # one row per touch/attempt: {date, side, n, tm, rsi, net, trend, held_to_close}
    for d in days:
        day = load_day(tk, d, args.max_dte, args.min_dte)
        if day is None:
            continue
        pr = profile_at(day, wall_mod, args.bs_fill)
        if pr is None:
            continue
        spot0, bs, _, _ = pr
        w = walls(spot0, bs)
        if w is None:
            continue
        cw, pw, net = w
        rsi_s = rsi_series(day)
        px_all = _minute_px(day, wall_mod, None)   # walk price from when the levels are set
        mods, pxs = px_all["_mod"].to_list(), px_all["px"].to_list()
        reset = max(1.5 * tol, spot0 * args.reset_frac)
        tr = trd.get(d, "?")

        for side, wall, is_call in (("CALL", cw, True), ("PUT", pw, False)):
            if wall is None:
                continue
            state, attempt, day_rows, breached = "away", 0, [], False
            for m, px in zip(mods, pxs):
                if (is_call and px > wall + tol) or (not is_call and px < wall - tol):
                    breached = True
                    break
                near = (px >= wall - tol) if is_call else (px <= wall + tol)
                far = (px <= wall - reset) if is_call else (px >= wall + reset)
                if state == "away" and near:
                    attempt += 1
                    if win_lo <= m <= win_hi:
                        day_rows.append({"date": d, "side": side, "n": attempt, "tm": m,
                                         "rsi": rsi_at(rsi_s, m), "net": net, "trend": tr})
                    state = "at"
                elif state == "at" and far:
                    state = "away"
            for r in day_rows:
                r["held"] = not breached
                ent.append(r)

    if not ent:
        print("  no wall touches in the entry window across these days.")
        return

    def rate(rows):
        return f"{sum(r['held'] for r in rows):>4}/{len(rows):<4} ({100*sum(r['held'] for r in rows)/len(rows):>3.0f}%)"

    print("=" * 66)
    print("  P(wall holds to close) by which test of the wall you enter on")
    print("=" * 66)
    print(f"  all entries:                 {rate(ent)}")
    for n, lbl in ((1, "1st test (first approach)"),
                   (2, "2nd test (held 1st)"),
                   (3, "3rd+ test (held 2+)")):
        sub = [r for r in ent if (r["n"] >= 3 if n == 3 else r["n"] == n)]
        if sub:
            print(f"  {lbl:28} {rate(sub)}")

    print("\n  --- attempt x RSI ---")
    for side, chk, lbl in (("CALL", lambda r: r >= args.rsi_hi, f"CALL RSI>={args.rsi_hi:.0f}"),
                           ("PUT", lambda r: r <= args.rsi_lo, f"PUT RSI<={args.rsi_lo:.0f}")):
        s = [r for r in ent if r["side"] == side and r["rsi"] == r["rsi"]]
        for n in (1, 2, 3):
            sub = [r for r in s if (r["n"] >= 3 if n == 3 else r["n"] == n)]
            if not sub:
                continue
            ext = [r for r in sub if chk(r["rsi"])]
            oth = [r for r in sub if not chk(r["rsi"])]
            tag = "3rd+" if n == 3 else f"{n}{'st' if n == 1 else 'nd'}"
            line = f"  {side} {tag:5} {lbl}: {rate(ext) if ext else '   -      '}   other RSI: {rate(oth) if oth else '   -'}"
            print(line)

    print("\n  --- trend / net-gamma (all entries) ---")
    for k in ("UPTREND", "CHOP", "DOWNTREND"):
        sub = [r for r in ent if r["trend"] == k]
        if sub:
            print(f"  {k:10} {rate(sub)}")
    for k in ("+", "-"):
        sub = [r for r in ent if r["net"] == k]
        if sub:
            print(f"  net {k}      {rate(sub)}")


def _silver_dates() -> list[date]:
    out = []
    for p in glob.glob(os.path.join(LAKE, BARS, "date=*")):
        try:
            out.append(date.fromisoformat(os.path.basename(p).split("=", 1)[1]))
        except ValueError:
            pass
    return sorted(out)


def run(args):
    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    if args.range and len(args.dates) == 2:
        lo, hi = sorted(date.fromisoformat(x) for x in args.dates)
        all_days = [d for d in _silver_dates() if lo <= d <= hi]
    else:
        all_days = [date.fromisoformat(x) for x in args.dates]
    for tk in tickers:
        days = all_days
        if args.vol_filter:
            from directional_flow_backtester import load_volume_regime
            vol = load_volume_regime(args.hist, tk)
            if not vol:
                print(f"{tk}: no historical/{tk}.parquet for --vol-filter -- skipping\n")
                continue
            days = [d for d in all_days if vol.get(d) == args.vol_filter]
            print(f"{tk}: --vol-filter {args.vol_filter} kept {len(days)}/{len(all_days)} days")
        if args.trend_filter:
            from directional_flow_backtester import load_trend_regime
            trd = load_trend_regime(args.hist, tk)
            if not trd:
                print(f"{tk}: no historical/{tk}.parquet for --trend-filter -- skipping\n")
                continue
            days = [d for d in days if trd.get(d) == args.trend_filter]
            print(f"{tk}: --trend-filter {args.trend_filter} kept {len(days)}/{len(all_days)} days")
        if not days:
            print(f"{tk}: no days left after filtering\n")
            continue
        (run_reject if args.reject_test else run_one)(args, tk, days)
        print()


def run_one(args, tk: str, days: list[date]):
    snaps = snapshots(args.step)
    print(f"{tk}  dte {args.min_dte}-{args.max_dte}  snapshots every {args.step}m"
          + ("  [bs-fill]" if args.bs_fill else "") + "\n")

    hdr = f"  {'date':11}{'open_spot':>10}{'call_wall':>10}{'put_wall':>10}{'net':>4}" \
          f"{'cw_drift':>9}{'pw_drift':>9}{'cover%':>8}  band  touch"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    pool = {"band_ok": 0, "band_n": 0,
            "reach": 0, "reach_held": 0, "reach_n": 0,
            "cw_drift": [], "pw_drift": [], "cover": [],
            "cw_dist": [], "pw_dist": [],
            "band_pos": [0, 0], "band_neg": [0, 0]}   # [ok, n] split by open net-gamma sign
    dump_rows = []

    entry_mod = None
    if args.entry:
        eh, em = map(int, args.entry.split(":"))
        entry_mod = eh * 60 + em

    for d in days:
        day = load_day(tk, d, args.max_dte, args.min_dte)
        if day is None:
            print(f"  {d.isoformat():11}  (no silver bars)")
            continue
        # range/close measured from --entry onward (default whole RTH)
        fwd = day.filter(pl.col("_mod") >= entry_mod) if entry_mod else day
        lo_px = float(fwd.select(pl.col("underlying_close").min()).item())
        hi_px = float(fwd.select(pl.col("underlying_close").max()).item())
        close_px = float(day.sort("_mod").select("underlying_close").tail(30).median().item())

        use_snaps = [m for m in snaps if entry_mod is None or m >= entry_mod] or snaps
        series = []
        for m in use_snaps:
            pr = profile_at(day, m, args.bs_fill)
            if pr is None:
                continue
            spot, bs, cov_oi, tot_oi = pr
            w = walls(spot, bs)
            if w is None:
                continue
            cw, pw, net = w
            series.append((m, spot, cw, pw, net, cov_oi, tot_oi))
            if args.dump and d == days[0]:
                for k, gm in sorted(bs.items()):
                    dump_rows.append({"minute": m, "spot": round(spot, 2),
                                      "strike": k, "net_gamma": gm})

        if not series:
            print(f"  {d.isoformat():11}  (no profile)")
            continue
        m0, s0, cw0, pw0, net0, cov0, tot0 = series[0]
        _, _, cwN, pwN, _, _, _ = series[-1]
        cover = 100 * cov0 / tot0 if tot0 else 0.0
        cw_dr = (cwN - cw0) if (cw0 and cwN) else None
        pw_dr = (pwN - pw0) if (pw0 and pwN) else None

        band_ok = ""
        if cw0 and pw0:
            pool["band_n"] += 1
            inside = (hi_px <= cw0) and (lo_px >= pw0)
            if inside:
                pool["band_ok"] += 1
            band_ok = "yes" if inside else "no"
            bucket = pool["band_pos"] if net0 == "+" else pool["band_neg"]
            bucket[1] += 1
            bucket[0] += int(inside)
            pool["cw_dist"].append((cw0 - s0) / s0 * 100)
            pool["pw_dist"].append((s0 - pw0) / s0 * 100)

        touch = ""
        tol = args.touch_tol
        for wall, is_call in ((cw0, True), (pw0, False)):
            if wall is None:
                continue
            pool["reach_n"] += 1
            reached = (hi_px >= wall - tol) if is_call else (lo_px <= wall + tol)
            held = (close_px <= wall) if is_call else (close_px >= wall)
            if reached:
                pool["reach"] += 1
                if held:
                    pool["reach_held"] += 1
                    touch += ("C" if is_call else "P")   # reached & closed back inside
                else:
                    touch += ("c" if is_call else "p")   # reached & broke through

        if cw_dr is not None:
            pool["cw_drift"].append(abs(cw_dr))
        if pw_dr is not None:
            pool["pw_drift"].append(abs(pw_dr))
        pool["cover"].append(cover)

        print(f"  {d.isoformat():11}{s0:>10.2f}"
              f"{('-' if cw0 is None else f'{cw0:.1f}'):>10}"
              f"{('-' if pw0 is None else f'{pw0:.1f}'):>10}"
              f"{net0:>4}"
              f"{('-' if cw_dr is None else f'{cw_dr:+.1f}'):>9}"
              f"{('-' if pw_dr is None else f'{pw_dr:+.1f}'):>9}"
              f"{cover:>7.0f}%  {band_ok:5} {touch}")

    print("\n" + "=" * 74)
    print("  POOLED")
    print("=" * 74)
    if pool["cover"]:
        print(f"  OI coverage (open snapshot):  median {np.median(pool['cover']):.0f}%   "
              f"min {min(pool['cover']):.0f}%   "
              f"-> {'USABLE' if np.median(pool['cover']) >= 60 else 'THIN -- walls unreliable, try --bs-fill or a wider --max-dte'}")
    if pool["cw_dist"]:
        print(f"  wall distance from spot:  call +{np.median(pool['cw_dist']):.2f}%   "
              f"put -{np.median(pool['pw_dist']):.2f}%  (median)")
    if pool["cw_drift"]:
        print(f"  wall drift open->close:  call {np.median(pool['cw_drift']):.2f} pts   "
              f"put {np.median(pool['pw_drift']):.2f} pts  (median |move|; small = stable)")
    if pool["band_n"]:
        print(f"  open [put wall, call wall] band CONTAINED the day's H/L range: "
              f"{pool['band_ok']}/{pool['band_n']}  ({100*pool['band_ok']/pool['band_n']:.0f}%)"
              f"   <- the number that matters for selling BEYOND the band")
        for lbl, b in (("net +gamma (suppression) days", pool["band_pos"]),
                       ("net -gamma (amplification) days", pool["band_neg"])):
            if b[1]:
                print(f"      {lbl:34} {b[0]}/{b[1]}  ({100*b[0]/b[1]:.0f}%)")
    if pool["reach_n"]:
        print(f"  a wall was REACHED (within {args.touch_tol}$): "
              f"{pool['reach']}/{pool['reach_n']} wall-days ({100*pool['reach']/pool['reach_n']:.0f}%)")
        if pool["reach"]:
            print(f"    ...and of those, price closed back INSIDE the wall (respected): "
                  f"{pool['reach_held']}/{pool['reach']}  ({100*pool['reach_held']/pool['reach']:.0f}%)")
    print("\n  legend: band = open band contains day H/L.  touch: C/P = wall reached & held,")
    print("          c/p = wall reached & broke through, blank = never reached.  net: profile gamma sign.")

    if args.xcheck:
        _xcheck(tk, days[0], args)
    if args.dump and dump_rows:
        import csv
        with open(args.dump, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["minute", "spot", "strike", "net_gamma"])
            w.writeheader()
            w.writerows(dump_rows)
        print(f"\n  dumped {len(dump_rows)} rows ({days[0]}) -> {args.dump}")


def _xcheck(tk: str, d: date, args):
    p = os.path.join(LAKE, SPOT_GEX, f"date={d.isoformat()}", f"{tk}.parquet")
    if not os.path.exists(p):
        print(f"\n  --xcheck: no {p}")
        return
    uw = _rth_min(pl.read_parquet(p))
    day = load_day(tk, d, args.max_dte, args.min_dte)
    if day is None:
        return
    print(f"\n  XCHECK vs spot-exposures-1m gamma_per_one_percent_move_oi ({d})")
    print(f"  {'minute':>7}{'derived_net_gamma':>20}{'uw_gamma_oi':>16}{'ratio':>9}")
    for m in snapshots(max(args.step, 60)):
        pr = profile_at(day, m, args.bs_fill)
        if pr is None:
            continue
        _, bs, _, _ = pr
        derived = sum(bs.values())
        row = uw.filter(pl.col("_mod") <= m).sort("_mod").tail(1)
        if row.is_empty():
            continue
        uwg = float(row.select("gamma_per_one_percent_move_oi").item())
        agree = "OK " if (derived < 0) == (uwg < 0) else "DIFF"
        hh, mm = divmod(m, 60)
        print(f"  {hh:02d}:{mm:02d}  {derived:>18.3e}{uwg:>16.3e}   sign {agree}")
    print("  Sign agreement is the useful check -- the derived LEVEL is 0DTE-window-only and\n"
          "  noisier than UW's all-expiry figure. Use spot-exposures-1m for the regime read;\n"
          "  this script is for WALLS (strike selection).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker", help="one ticker, or a comma list e.g. QQQ,IWM")
    ap.add_argument("dates", nargs="+", help="one or more YYYY-MM-DD (or START END with --range)")
    ap.add_argument("--range", action="store_true", help="treat the two dates as an inclusive range over silver")
    ap.add_argument("--step", type=int, default=30, help="snapshot interval, minutes (default 30)")
    ap.add_argument("--max-dte", type=int, default=7)
    ap.add_argument("--min-dte", type=int, default=0)
    ap.add_argument("--entry", metavar="HH:MM", help="measure walls + rest-of-day range from this ET time (default: open)")
    ap.add_argument("--touch-tol", type=float, default=0.5, help="price-to-wall 'reached' tolerance, $ (default 0.5)")
    ap.add_argument("--bs-fill", action="store_true", help="recompute gamma from iv_close for contracts with no same-day trade")
    ap.add_argument("--xcheck", action="store_true", help="compare derived net gamma to lake spot-exposures-1m")
    ap.add_argument("--dump", metavar="PATH", help="write the first date's (snapshot x strike) grid to CSV")
    ap.add_argument("--reject-test", action="store_true",
                    help="your actual play: wall levels as of --wall-asof, watch --window for a "
                         "touch, check RSI at the touch, report whether the wall then holds")
    ap.add_argument("--wall-asof", metavar="HH:MM", default="10:30", help="snapshot time for the wall levels (default 10:30)")
    ap.add_argument("--window", metavar="HH:MM-HH:MM", default="11:00-15:00",
                    help="entry window -- a wall touch in here is an entry, bucketed by which test (default 11:00-15:00)")
    ap.add_argument("--reset-frac", type=float, default=0.0015,
                    help="price must retreat this fraction of spot from the wall before the next touch counts (default 0.0015)")
    ap.add_argument("--rsi-hi", type=float, default=70.0, help="RSI overbought threshold for CALL-wall touches (default 70)")
    ap.add_argument("--rsi-lo", type=float, default=30.0, help="RSI oversold threshold for PUT-wall touches (default 30)")
    ap.add_argument("--vol-filter", choices=("LOWVOL", "NORMVOL", "HIVOL"),
                    help="keep only days where the underlying's prior-day volume regime matches "
                         "(same LOWVOL/NORMVOL/HIVOL definition as directional_flow_backtester --regime-by volume). "
                         "Needs historical/{T}.parquet (ohlc-build).")
    ap.add_argument("--trend-filter", choices=("UPTREND", "CHOP", "DOWNTREND"),
                    help="keep only days where the underlying's prior-day SMA trend state matches "
                         "(same definition as directional_flow_backtester --regime-by trend).")
    ap.add_argument("--hist", default="historical", help="dir for --vol-filter/--trend-filter's historical/{T}.parquet lookup")
    run(ap.parse_args())
