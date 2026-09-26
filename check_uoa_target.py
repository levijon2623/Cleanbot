# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_uoa_target.py
====================
Does a big UOA print predict WHERE price goes?  Not the dealer-hedge reaction --
the strike itself as a directional target.

For each UOA flag (contract with day_vol > open_interest, a premium floor, and a
genuinely OTM strike), measure whether the underlying REACHES that strike by the
contract's expiration -- and compare against a distance/DTE-matched volatility
baseline (the same drift-free control that busted magnet theory: a strike 4% away
with 12 DTE gets touched a lot regardless; the question is whether UOA-flagged
strikes beat that base rate).

  touch      underlying [low,high] over (flag_date, expiry] reaches K
  itm_exp    close on expiry is beyond K (the bet actually pays, not just a wick)
  baseline   P(this underlying moves >= |mny| in the flagged direction within
             `dte` sessions), unconditional, from its own full daily history
  lift       touch - baseline   (paired, per flag)

Prototype cohort = the 5 tickers with a prebuilt _oi_cache (NVDA AVGO MU META
TSLA), DTE 7-60 / |mny| <= 25% (the cache's filter).  --tickers to override.

Usage:
  python check_uoa_target.py
  python check_uoa_target.py --tickers NVDA META --min-prem 500000 --min-vol 250
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
OI_CACHE = "_oi_cache"
COHORT = ["NVDA", "AVGO", "MU", "META", "TSLA"]

SILVER = "lake/silver/option-contracts-1m"
SC_SNAPSHOT = "_uoa_sc_snapshot.parquet"
LC_SNAPSHOT = "_uoa_lc_snapshot.parquet"
SC_WIN_LO, SC_WIN_HI = "2026-04-01", "2026-08-21"
SC_DTE_LO, SC_DTE_HI, SC_MNY = 7, 60, 0.25
OOS_SPLIT = "2025-08-21"


def _build_snapshot(tickers, cache, lo=None, hi=None):
    """One lazy pass over the silver partitions (optionally windowed to [lo,hi])
    -> EOD per-contract rows (OI, day_vol, ask/bid vol, premium), DTE 7-60 /
    |mny| <= 25%.  Same shape as _oi_cache/{tk}.parquet + a `tk` col."""
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    want = set(tickers)
    parts = sorted(glob.glob(f"{SILVER}/date=*/bars.parquet"))
    if lo or hi:
        parts = [p for p in parts if (lo or "0") <= p.split("date=")[1][:10] <= (hi or "9")]
    frames, pxframes = [], []
    for i, p in enumerate(parts, 1):
        d = p.split("date=")[1][:10]
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol").is_in(want))
              .select("underlying_symbol", "option_chain_id", "option_type", "strike", "expiry",
                      "minute_et", "open_interest", "volume", "ask_volume", "bid_volume",
                      "premium", "underlying_close")
              # CAST BEFORE MULTIPLYING -- polars `dt.hour()` is Int8, so
              # `hour * 60` OVERFLOWS silently (09:30 -> 28, not 570) and `_m`
              # wraps into -128..127. `_m` orders the `first`/`last` aggregates
              # below, so the wrap scrambled the daily open/close and the EOD
              # open-interest snapshot. Found and fixed 2026-09-12; any cached
              # `_uoa_cache` parquet built before that date is stale.
              .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                             + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("_m")))
        df = lf.collect()
        if df.is_empty():
            continue
        # underlying daily o/h/l/c from underlying_close (before the DTE/mny filter)
        px = (df.sort("_m").group_by("underlying_symbol").agg(
                  pl.col("underlying_close").first().alias("o"),
                  pl.col("underlying_close").max().alias("h"),
                  pl.col("underlying_close").min().alias("l"),
                  pl.col("underlying_close").last().alias("c"))
              .with_columns(pl.lit(d).alias("date")))
        pxframes.append(px.to_pandas())
        eod = (df.sort("_m").group_by(["underlying_symbol", "option_chain_id"]).agg(
                    pl.col("option_type").first(),
                    pl.col("strike").first().cast(pl.Float64),
                    pl.col("expiry").first(),
                    pl.col("open_interest").last().cast(pl.Float64),
                    pl.col("underlying_close").last().cast(pl.Float64),
                    pl.col("volume").sum().cast(pl.Float64).alias("day_vol"),
                    pl.col("ask_volume").sum().cast(pl.Float64).alias("ask_vol"),
                    pl.col("bid_volume").sum().cast(pl.Float64).alias("bid_vol"),
                    pl.col("premium").sum().cast(pl.Float64).alias("day_prem"))
               .with_columns(
                    ((pl.col("expiry").cast(pl.Date) - pl.lit(pd.Timestamp(d).date())).dt.total_days()).alias("dte"),
                    ((pl.col("strike") - pl.col("underlying_close")) / pl.col("underlying_close")).alias("mny"),
                    pl.lit(d).alias("date"))
               .filter((pl.col("dte") >= SC_DTE_LO) & (pl.col("dte") <= SC_DTE_HI)
                       & (pl.col("mny").abs() <= SC_MNY)))
        if not eod.is_empty():
            frames.append(eod.to_pandas())
        if i % 20 == 0:
            print(f"  snapshot {i}/{len(parts)}")
    out = pd.concat(frames, ignore_index=True).rename(columns={"underlying_symbol": "tk"})
    out["date"] = pd.to_datetime(out["date"])
    out["expiry"] = pd.to_datetime(out["expiry"])
    out.to_parquet(cache, index=False)
    px = pd.concat(pxframes, ignore_index=True).rename(columns={"underlying_symbol": "tk"})
    px["date"] = pd.to_datetime(px["date"])
    px.to_parquet(cache.replace(".parquet", ".px.parquet"), index=False)
    print(f"  cached {len(out):,} contract-days + {len(px):,} px-days -> {cache}")
    return out


def _daily_ohlc_px(px_df):
    """{tk -> DataFrame(date-indexed, h/l/c)} for every ticker in the snapshot.

    PREFERS historical/{tk}.parquet and only falls back to the snapshot's own
    px table. The cached table's `c` is NOT TRUSTWORTHY on any file built before
    2026-09-12: `_build_snapshot` ordered its first/last aggregates by `_m`,
    which overflowed Int8 and wrapped into -128..127, so `c` came from an
    arbitrary minute rather than the session's last. Measured against the true
    RTH closes that was a median 0.18-0.34% price error per day and, worse, the
    daily close-to-close RETURN disagreed in SIGN with truth on 14-23% of days
    -- enough to corrupt any forward-return target built on it.

    `h`/`l` are max/min and were never affected, and the contract-level
    open_interest snapshot is also fine (OI is constant within a session --
    verified 15,761/15,761 contracts), so `_uoa_*_snapshot.parquet` itself and
    everything downstream of the OI columns (e.g. check_breakout_filters) stand.
    """
    out, fell_back = {}, []
    for tk, g in px_df.groupby("tk"):
        d = _daily_ohlc(tk)
        if d is not None and not d.empty:
            out[tk] = d
        else:
            fell_back.append(tk)
            out[tk] = g.set_index("date").sort_index()[["h", "l", "c"]].astype(float)
    if fell_back:
        print(f"  !! no historical/ OHLC for {fell_back}; using the cached px table, "
              f"whose close is unreliable if the cache predates 2026-09-12")
    return out


def _spy_regime():
    """{date -> 'UP'|'DOWN'|'FLAT'} from SPY daily close: 20d SMA slope + price side."""
    d = _daily_ohlc("SPY")
    if d is None:
        return {}
    c = d["c"]
    sma = c.rolling(20).mean()
    slope = sma - sma.shift(10)
    reg = pd.Series("FLAT", index=c.index)
    reg[(c > sma) & (slope > 0)] = "UP"
    reg[(c < sma) & (slope < 0)] = "DOWN"
    return {k.date(): v for k, v in reg.items()}


def _daily_ohlc(tk):
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).to_pandas()
    d.columns = [c.lower() for c in d.columns]
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York")
    mo = et.dt.hour * 60 + et.dt.minute
    m = (mo >= 570) & (mo <= 960)
    g = pd.DataFrame({"date": et[m].dt.date.values,
                      "h": d["high"][m].astype(float).values,
                      "l": d["low"][m].astype(float).values,
                      "c": d["close"][m].astype(float).values})
    day = g.groupby("date").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    day.index = pd.to_datetime(day.index)
    return day.sort_index()


def _baseline_fn(day):
    """memoized P(move >= m in `direction` within h sessions), unconditional."""
    c = day["c"].values.astype(float)
    hi = day["h"].values.astype(float)
    lo = day["l"].values.astype(float)
    n = len(c)

    @functools.lru_cache(maxsize=None)
    def p(direction, h, m_bp):
        m = m_bp / 1e4
        h = int(h)
        if h < 1 or n - h < 30:
            return np.nan
        hits = tot = 0
        for t in range(0, n - h):
            if direction == "CALL":
                mx = hi[t + 1:t + 1 + h].max()
                hits += (mx / c[t] - 1) >= m
            else:
                mn = lo[t + 1:t + 1 + h].min()
                hits += (mn / c[t] - 1) <= -m
            tot += 1
        return hits / tot if tot else np.nan

    return p


def _flags_from(s, a):
    """UOA flags out of an EOD per-contract snapshot DataFrame."""
    s = s.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["expiry"] = pd.to_datetime(s["expiry"])
    is_call = s["option_type"].str.lower().eq("call")
    otm = (is_call & (s["mny"] > 0.005)) | (~is_call & (s["mny"] < -0.005))
    flag = (otm
            & (s["day_vol"] > s["open_interest"])
            & (s["day_prem"] >= a.min_prem)
            & (s["day_vol"] >= a.min_vol))
    f = s[flag].copy()
    f["standout"] = f["day_vol"] / f["open_interest"].clip(lower=1)
    f["aggr"] = f["ask_vol"] / (f["ask_vol"] + f["bid_vol"]).clip(lower=1)
    f["side"] = np.where(f["aggr"] >= 0.55, "ASK/buy",
                np.where(f["aggr"] <= 0.45, "BID/sell", "mid"))
    f["dir"] = np.where(is_call[flag], "CALL", "PUT")
    f["absmny"] = f["mny"].abs()
    return f


def run(a):
    print("=" * 100)
    print("  UOA FLAG -> does price REACH the flagged strike by expiry?  (vs distance/DTE vol baseline)")
    print(f"  flag = day_vol > OI  &  day_prem >= ${a.min_prem:,.0f}  &  day_vol >= {a.min_vol}  &  OTM")
    print("=" * 100)

    by_tk = px_by_tk = None
    if a.sc or a.lc:
        if a.sc:
            allsc = pd.read_parquet("_smallcap_triage_all.parquet")
            a.tickers = allsc[allsc["density"] >= a.min_density]["ticker"].tolist()
            print(f"  SMALL-CAP mode: {len(a.tickers)} tickers, 90d window ({SC_WIN_LO}..{SC_WIN_HI})")
            snap = _build_snapshot(a.tickers, SC_SNAPSHOT, SC_WIN_LO, SC_WIN_HI)
            pxpath = SC_SNAPSHOT.replace(".parquet", ".px.parquet")
        else:
            with open("largecap_universe.json") as fh:
                a.tickers = json.load(fh)["tickers"]
            print(f"  LARGE-CAP mode: {len(a.tickers)} tickers, FULL history (multi-regime, IS/OOS @ {OOS_SPLIT})")
            snap = _build_snapshot(a.tickers, LC_SNAPSHOT)
            pxpath = LC_SNAPSHOT.replace(".parquet", ".px.parquet")
        by_tk = {tk: g for tk, g in snap.groupby("tk")}
        px_by_tk = _daily_ohlc_px(pd.read_parquet(pxpath))

    reg = _spy_regime()
    split = pd.Timestamp(OOS_SPLIT)
    allf = []
    per_tk = []
    for k, tk in enumerate(a.tickers, 1):
        day = px_by_tk.get(tk) if px_by_tk is not None else _daily_ohlc(tk)
        if day is not None and len(day) < 40:
            day = None
        if by_tk is not None:
            s = by_tk.get(tk)
            f = _flags_from(s, a) if s is not None and len(s) else None
        else:
            fp = os.path.join(OI_CACHE, f"{tk}.parquet")
            f = _flags_from(pd.read_parquet(fp), a) if os.path.exists(fp) else None
        if day is None or f is None or f.empty:
            if not (a.sc or a.lc):
                print(f"  {tk}: no data / no flags")
            continue
        bp = _baseline_fn(day)
        didx = day.index
        last = didx[-1]
        rows = []
        for _, r in f.iterrows():
            D, E, K, dr = r["date"], r["expiry"], r["strike"], r["dir"]
            if E > last:
                continue
            path = day[(didx > D) & (didx <= E)]
            if path.empty:
                continue
            if dr == "CALL":
                touched = path["h"].max() >= K
                itm = path["c"].iloc[-1] >= K
            else:
                touched = path["l"].min() <= K
                itm = path["c"].iloc[-1] <= K
            ttt = np.nan
            if touched:
                hit = path[(path["h"] >= K)] if dr == "CALL" else path[(path["l"] <= K)]
                ttt = (hit.index[0] - D).days
            base = bp(dr, int(r["dte"]), round(r["absmny"] * 1e4))
            Dd = D.date() if hasattr(D, "date") else D
            rows.append({"tk": tk, "dir": dr, "side": r["side"], "dte": int(r["dte"]),
                         "absmny": r["absmny"], "standout": r["standout"], "aggr": r["aggr"],
                         "prem": r["day_prem"], "touched": int(touched), "itm_exp": int(itm),
                         "baseline": base, "ttt": ttt,
                         "regime": reg.get(Dd, "?"),
                         "half": "IS" if pd.Timestamp(D) < split else "OOS"})
        if not rows:
            continue
        fd = pd.DataFrame(rows).dropna(subset=["baseline"])
        if fd.empty:
            print(f"  {tk}: no usable flags after baseline join")
            continue
        fd["lift"] = fd["touched"] - fd["baseline"]
        allf.append(fd)
        src = by_tk[tk] if by_tk is not None else pd.read_parquet(os.path.join(OI_CACHE, f"{tk}.parquet"))
        avg_dvol = src.groupby("date")["day_vol"].sum().mean()
        per_tk.append((tk, len(fd), fd["touched"].mean(), fd["baseline"].mean(),
                       fd["lift"].mean(), fd["itm_exp"].mean(), avg_dvol))
        if (a.sc or a.lc) and k % 100 == 0:
            print(f"  ...{k}/{len(a.tickers)}")

    if not allf:
        print("  nothing"); return
    A = pd.concat(allf, ignore_index=True)
    # honest significance: cluster by ticker (flags within a name are heavily
    # autocorrelated -- overlapping windows, clustered days)
    tkm = pd.Series([x[4] for x in per_tk])
    if len(tkm) >= 8:
        print(f"\n  CLUSTERED (mean of {len(tkm)} per-ticker mean lifts): {tkm.mean():+.4f}  "
              f"t={tkm.mean() / (tkm.std() / np.sqrt(len(tkm))):+.1f}   "
              f"(pos: {(tkm > 0).sum()}/{len(tkm)})")

    def line(lbl, d):
        if len(d) < 10:
            print(f"    {lbl:26} n={len(d):>5}  (thin)"); return
        se = d["lift"].std() / np.sqrt(len(d))
        # clustered t across the tickers present in this slice
        cm = d.groupby("tk")["lift"].mean()
        ct = (cm.mean() / (cm.std() / np.sqrt(len(cm)))) if len(cm) >= 8 and cm.std() > 0 else None
        cts = f"{ct:+.1f}" if ct is not None else " n/a"
        print(f"    {lbl:28} n={len(d):>5}  touch {d['touched'].mean():.3f}  base {d['baseline'].mean():.3f}  "
              f"lift {d['lift'].mean():+.3f} (t={d['lift'].mean()/se:+.1f}, clus t={cts})  "
              f"itm {d['itm_exp'].mean():.3f}")

    def breakdowns(A):
        print(f"\n  by STANDOUT (day_vol / OI)")
        for lo, hi in [(1, 2), (2, 5), (5, 10), (10, 1e9)]:
            line(f"{lo}-{hi if hi < 1e8 else '+'}x", A[(A["standout"] >= lo) & (A["standout"] < hi)])
        print(f"\n  by DTE")
        for lo, hi in [(7, 15), (15, 30), (30, 45), (45, 61)]:
            line(f"{lo}-{hi}d", A[(A["dte"] >= lo) & (A["dte"] < hi)])
        print(f"\n  by |MONEYNESS|")
        for lo, hi in [(0.005, 0.03), (0.03, 0.07), (0.07, 0.13), (0.13, 0.26)]:
            line(f"{lo*100:.0f}-{hi*100:.0f}%", A[(A["absmny"] >= lo) & (A["absmny"] < hi)])
        print(f"\n  by AGGRESSOR (ask-side share)")
        for lo, hi, nm in [(0.0, 0.4, "sold (<40% ask)"), (0.4, 0.6, "mixed"), (0.6, 1.01, "bought (>60% ask)")]:
            line(nm, A[(A["aggr"] >= lo) & (A["aggr"] < hi)])

    # ---- the two candidate signals, split by market regime + IS/OOS ----
    print(f"\n{'='*72}\n  TARGET SIGNALS by SPY REGIME (at flag date) and IS/OOS half\n{'='*72}")
    cw = A[(A["dir"] == "CALL") & (A["side"] == "BID/sell")]      # call-writer bull-fade
    pb = A[(A["dir"] == "PUT") & (A["side"] == "ASK/buy")]        # aggressive put buyer
    for nm, sig in [("CALL BID/sell  (writer bull-fade: want lift < 0)", cw),
                    ("PUT  ASK/buy   (put buyer: want lift > 0)", pb)]:
        print(f"\n  {nm}")
        line("  ALL", sig)
        for rr in ("UP", "FLAT", "DOWN"):
            line(f"  SPY {rr}", sig[sig["regime"] == rr])
        for hh in ("IS", "OOS"):
            line(f"  {hh}", sig[sig["half"] == hh])

    pt = sorted(per_tk, key=lambda x: -x[6])
    wide = a.sc or a.lc
    print(f"\n  PER TICKER  (avg_dvol = mean daily option volume -- activity proxy)"
          + ("  [top/bottom 12 by activity]" if wide else ""))
    print(f"  {'tk':6}{'n':>5}{'touch':>8}{'base':>8}{'lift':>8}{'itm_exp':>9}{'avg_dvol':>12}")
    show = (pt[:12] + [("...", 0, 0, 0, 0, 0, 0)] + pt[-12:]) if (wide and len(pt) > 26) else pt
    for tk, n, tou, bas, lif, itm, dv in show:
        if tk == "...":
            print("  ...")
            continue
        print(f"  {tk:6}{n:>5}{tou:>8.3f}{bas:>8.3f}{lif:>+8.3f}{itm:>9.3f}{dv:>12,.0f}")
    if wide and len(pt) >= 20:
        # user's hypothesis: does lift decline with activity?
        act = pd.DataFrame(pt, columns=["tk", "n", "tou", "bas", "lift", "itm", "dv"])
        q = pd.qcut(act["dv"].rank(method="first"), 4, labels=["Q1 low act", "Q2", "Q3", "Q4 high act"])
        print(f"\n  lift by ACTIVITY quartile (user's 'more activity -> weaker UOA signal' test):")
        for name, g in act.groupby(q, observed=True):
            print(f"    {name:14} n_tk={len(g):>4}  mean per-ticker lift {g['lift'].mean():+.4f}  "
                  f"flag-weighted {np.average(g['lift'], weights=g['n']):+.4f}")

    print(f"\n  OVERALL");                     line("all flags", A)

    print(f"\n  by DIRECTION x SIDE  (ASK/buy = lifting the offer; BID/sell = hitting the bid)")
    print(f"  lift < 0 => price reaches the flagged strike LESS than a distance/DTE baseline")
    for dr in ("CALL", "PUT"):
        for sd in ("ASK/buy", "BID/sell", "mid"):
            line(f"{dr:4} {sd}", A[(A["dir"] == dr) & (A["side"] == sd)])

    print(f"\n  DIRECTION x SIDE x |MONEYNESS|")
    for dr in ("CALL", "PUT"):
        for sd in ("ASK/buy", "BID/sell"):
            for lo, hi in [(0.005, 0.03), (0.03, 0.07), (0.07, 0.13), (0.13, 0.26)]:
                line(f"{dr:4} {sd:9} {lo*100:.0f}-{hi*100:.0f}%",
                     A[(A["dir"] == dr) & (A["side"] == sd)
                       & (A["absmny"] >= lo) & (A["absmny"] < hi)])

    for dr in ("CALL", "PUT"):
        sub = A[A["dir"] == dr]
        print(f"\n{'='*60}\n  === {dr} flags only  (n={len(sub)}) ===\n{'='*60}")
        line(f"all {dr}", sub)
        breakdowns(sub)

    print()
    t = A.dropna(subset=["ttt"])
    if len(t):
        for dr in ("CALL", "PUT"):
            td = t[t["dir"] == dr]
            if len(td):
                print(f"  time-to-touch {dr}: median {td['ttt'].median():.0f}d  "
                      f"p25 {td['ttt'].quantile(.25):.0f}d  p75 {td['ttt'].quantile(.75):.0f}d")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=COHORT)
    ap.add_argument("--min-prem", type=float, default=250_000)
    ap.add_argument("--min-vol", type=int, default=100)
    ap.add_argument("--sc", action="store_true",
                    help="small-cap 90d mode: density-passers from _smallcap_triage_all.parquet")
    ap.add_argument("--lc", action="store_true",
                    help="large-cap FULL-history mode: largecap_universe.json (multi-regime)")
    ap.add_argument("--min-density", type=float, default=30.0, help="--sc only")
    a = ap.parse_args()
    if not hasattr(a, "lc"):
        a.lc = False
    a.tickers = [t.upper() for t in a.tickers]
    run(a)
