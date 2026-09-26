# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_gex_hunt.py
=================
VIABILITY TEST for a "hunt negative-GEX across the mid-mega universe and trade
its volatility" mode -- BEFORE building any dynamic scanner.

The flow trigger failed on the unfiltered small/mid-cap set (session 15: option
P&L -18.8%).  Question: does restricting to NEGATIVE dealer-GEX days (trending,
unpinned) on GENUINELY LIQUID names, with a CROSS-SECTIONAL flow normalisation
(no per-ticker percentile history for a name the scanner just found), rescue an
edge?

  --build          one lake pass -> _gexhunt_cache/{flow,bars}.parquet for the
                   top-N largecap_universe.json names, DTE 0-1, +/-1.5% strikes.
  (default)        EMA(5) cum-flow triggers; gate FIRST by flow z-score k>=2 /
                   trailing-pctile >=80 / raw pooled-median; only then walk the
                   ATM bracket with a REALISTIC ask-in/bid-out fill.  Split
                   pooled P&L by prior-day net_gex SIGN, and within NEGATIVE by
                   |net_gex| depth + vol regime.  IS/OOS @ 2025-08-21 + slices.
                   Bar: does -GEX-day P&L beat ZERO both halves after the haircut?

Usage:
  python check_gex_hunt.py --build --n 45
  python check_gex_hunt.py --n 45
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import polars as pl

from check_config_walkforward import _slice_idx

HIST = "historical"
SILVER = "lake/silver/option-contracts-1m"
CACHE = "_gexhunt_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
COMM = 0.015
EOD = 15 * 60 + 55


def _universe(n):
    u = json.load(open("largecap_universe.json"))["tickers"]
    out = []
    for t in u:
        if t not in out and os.path.exists(f"{HIST}/GEX{t}.parquet"):
            out.append(t)
        if len(out) >= n:
            break
    return out


def build(tickers):
    os.makedirs(CACHE, exist_ok=True)
    tset = list(tickers)
    parts = sorted(glob.glob(f"{SILVER}/date=*/bars.parquet"))
    fF, bF = [], []
    for i, p in enumerate(parts, 1):
        base = pl.scan_parquet(p).filter(pl.col("underlying_symbol").is_in(tset))
        fF.append(base.with_columns(
            pl.when(pl.col("option_type") == "call")
            .then((pl.col("ask_volume") - pl.col("bid_volume")) * pl.col("vwap") * 100)
            .otherwise(-((pl.col("ask_volume") - pl.col("bid_volume")) * pl.col("vwap") * 100)).alias("nf"))
            .group_by(["underlying_symbol", "minute_et"]).agg(pl.col("nf").sum().alias("net_flow_1m")).collect())
        bF.append(base.with_columns(
            ((pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days()).alias("dte"))
            .filter((pl.col("dte") >= 0) & (pl.col("dte") <= 1))
            .filter((pl.col("strike") - pl.col("underlying_close")).abs() / pl.col("underlying_close") <= 0.015)
            .select(["underlying_symbol", "option_chain_id", "option_type", "strike", "minute_et",
                     "close", "low", "bid_close", "ask_close", "underlying_close"]).collect())
        if i % 100 == 0:
            print(f"  {i}/{len(parts)}")
    flow = pl.concat(fF).to_pandas()
    flow["minute_et"] = pd.to_datetime(flow["minute_et"]).dt.tz_localize(None)
    flow["date"] = flow["minute_et"].dt.date
    flow = flow.sort_values(["underlying_symbol", "minute_et"])
    flow["cum_flow"] = flow.groupby(["underlying_symbol", "date"])["net_flow_1m"].cumsum()
    flow.to_parquet(f"{CACHE}/flow.parquet", index=False)
    bars = pl.concat(bF).to_pandas()
    bars["minute_et"] = pd.to_datetime(bars["minute_et"]).dt.tz_localize(None)
    bars["date"] = bars["minute_et"].dt.date
    bars["mod"] = bars["minute_et"].dt.hour * 60 + bars["minute_et"].dt.minute
    bars.to_parquet(f"{CACHE}/bars.parquet", index=False)
    print(f"  flow {len(flow):,} / {flow.underlying_symbol.nunique()} tk   bars {len(bars):,}")


def _gex(tk):
    g = pl.read_parquet(f"{HIST}/GEX{tk}.parquet").to_pandas()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    col = "net_gex_prior" if "net_gex_prior" in g.columns else "net_gex"
    return {d: v for d, v in zip(g["date"], g[col])}


def _triggers(gd):
    out = []
    for d, g in gd.groupby("date"):
        g = g.sort_values("minute_et")
        cum = g["cum_flow"].to_numpy(float)
        if len(cum) < 6:
            continue
        ema = pd.Series(cum).ewm(span=5, adjust=False).mean().to_numpy()
        mod = (g["minute_et"].dt.hour * 60 + g["minute_et"].dt.minute).to_numpy()
        for k in range(1, len(cum)):
            bull = cum[k - 1] <= ema[k - 1] and cum[k] > ema[k]
            bear = cum[k - 1] >= ema[k - 1] and cum[k] < ema[k]
            if (bull or bear) and 570 <= mod[k] <= 899:
                out.append({"date": d, "mod": int(mod[k]), "dir": "CALL" if bull else "PUT",
                            "abs_flow": abs(cum[k])})
    return out


def _annotate(trigs, win=60):
    if not trigs:
        return
    order = sorted(range(len(trigs)), key=lambda i: (trigs[i]["date"], trigs[i]["mod"]))
    dayns = np.array([pd.Timestamp(trigs[i]["date"]).value for i in order])
    fl = np.array([trigs[i]["abs_flow"] for i in order], float)
    w = win * 86_400_000_000_000
    for pos, i in enumerate(order):
        cur = dayns[pos]
        lo = int(np.searchsorted(dayns, cur - w)); hi = int(np.searchsorted(dayns, cur))
        h = fl[lo:hi]
        if len(h) >= 30 and h.std() > 0 and cur - dayns[0] >= w:
            trigs[i]["z"] = (trigs[i]["abs_flow"] - h.mean()) / h.std()
            trigs[i]["pct"] = float((h < trigs[i]["abs_flow"]).mean() * 100)
        else:
            trigs[i]["z"] = trigs[i]["pct"] = None


def _bracket(entry_ask, cl, lo, bp, mod, tr=1.0, rr=1.0):
    n = len(cl)
    cummax = np.maximum.accumulate(cl); cummin = np.minimum.accumulate(lo)
    eodx = mod >= EOD
    tsx = int(np.argmax(eodx)) if eodx.any() else n - 1
    tp = entry_ask * (1 + tr); sl = entry_ask * (1 - tr / rr)
    ti = int(np.searchsorted(cummax, tp)) if cummax[-1] >= tp else n
    si = int(np.searchsorted(-cummin, -sl)) if cummin[-1] <= sl else n
    ei = min(ti, si, tsx)
    if ei >= n:
        px = bp[-1]
    elif ti <= si and ti == ei:
        px = min(tp, bp[ei])
    elif si == ei:
        px = min(sl, bp[ei])
    else:
        px = bp[ei]
    return (px - entry_ask) / entry_ask - COMM


def _stat(pnls):
    if len(pnls) < 12:
        return f"n={len(pnls):>5}  (thin)"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]; o = [p for d, p in pnls if d >= SPLIT]
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    slc = " ".join(f"S{j+1}{np.mean(b)*100:+.0f}" if len(b) >= 5 else f"S{j+1}··" for j, b in enumerate(sl))
    return (f"n={len(v):>5}  avg {v.mean()*100:>+6.1f}%  IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  win {np.mean(v>0):.2f}  [{slc}]")


def run(a):
    flow = pd.read_parquet(f"{CACHE}/flow.parquet")
    bars = pd.read_parquet(f"{CACHE}/bars.parquet")
    flow["date"] = pd.to_datetime(flow["date"]).dt.date
    bars["date"] = pd.to_datetime(bars["date"]).dt.date
    tickers = sorted(set(flow["underlying_symbol"]) & set(bars["underlying_symbol"]))
    try:
        from directional_flow_backtester import load_volume_regime
    except Exception:
        def load_volume_regime(*_):
            return {}

    B = {g: {"NEG": [], "POS": []} for g in ("pct80", "z2.0", "raw")}
    negdeep = {"NEG-deep": [], "NEG-mild": []}
    negvol = {"LOWVOL": [], "NORMVOL": [], "HIVOL": []}
    gate_n = {g: 0 for g in B}

    for tk in tickers:
        gsign = _gex(tk); vreg = load_volume_regime(HIST, tk)
        gv = sorted(abs(v) for v in gsign.values() if v < 0)
        deep_cut = gv[int(len(gv) * 0.6)] if gv else 0.0
        trigs = _triggers(flow[flow.underlying_symbol == tk])
        _annotate(trigs)
        af = sorted(t["abs_flow"] for t in trigs)
        raw_thr = af[len(af) // 2] if af else 0.0

        tb = bars[bars.underlying_symbol == tk].sort_values("minute_et")
        # per contract: mod / close / low / bid / ask / strike / type
        CT, CL, LO, BD, AK, ST, OT = {}, {}, {}, {}, {}, {}, {}
        for cid, g in tb.groupby("option_chain_id"):
            CT[cid] = g["mod"].to_numpy()
            CL[cid] = g["close"].to_numpy(float); LO[cid] = g["low"].to_numpy(float)
            BD[cid] = g["bid_close"].to_numpy(float); AK[cid] = g["ask_close"].to_numpy(float)
            ST[cid] = float(g["strike"].iloc[0]); OT[cid] = g["option_type"].iloc[0]
        # per day: contracts + underlying-close series
        day_c, day_u = {}, {}
        for d, g in tb.groupby("date"):
            day_c[d] = [(cid, ST[cid], OT[cid]) for cid in g["option_chain_id"].unique()]
            day_u[d] = (g["mod"].to_numpy(), g["underlying_close"].to_numpy(float))

        for t in trigs:
            d = t["date"]
            sgnv = gsign.get(d)
            if sgnv is None or d not in day_c:
                continue
            gates = []
            if t.get("pct") is not None and t["pct"] >= 80:
                gates.append("pct80")
            if t.get("z") is not None and t["z"] >= 2.0:
                gates.append("z2.0")
            if t["abs_flow"] >= raw_thr:
                gates.append("raw")
            if not gates:
                continue
            um, uc = day_u[d]
            j = np.searchsorted(um, t["mod"], side="right") - 1
            if j < 0:
                continue
            spot = uc[j]
            want = "call" if t["dir"] == "CALL" else "put"
            cands = [(abs(s - spot), cid) for cid, s, ot in day_c[d] if ot == want]
            if not cands:
                continue
            cid = min(cands)[1]
            m = CT[cid]
            ei = np.searchsorted(m, t["mod"], side="right") - 1
            if ei < 0 or m[ei] < t["mod"] - 3:
                continue
            a_ = AK[cid][ei]; b_ = BD[cid][ei]
            mid = (a_ + b_) / 2 if b_ > 0 else CL[cid][ei]
            if not np.isfinite(mid) or mid < 0.50:
                continue
            fi = np.searchsorted(m, t["mod"], side="right")
            if len(m) - fi < 3:
                continue
            cl = CL[cid][fi:]; lo = LO[cid][fi:]; bp = BD[cid][fi:]; md = m[fi:]
            bp = np.where(np.isfinite(bp) & (bp > 0), bp, cl)
            if not (np.isfinite(cl).all() and np.isfinite(lo).all()):
                continue
            p = _bracket(a_ if a_ > 0 else mid, cl, lo, bp, md)
            sgn = "NEG" if sgnv < 0 else "POS"
            for gg in gates:
                B[gg][sgn].append((d, p)); gate_n[gg] += 1
            if sgn == "NEG" and "z2.0" in gates:
                negdeep["NEG-deep" if abs(sgnv) >= deep_cut else "NEG-mild"].append((d, p))
                if vreg.get(d) in negvol:
                    negvol[vreg[d]].append((d, p))

    print("=" * 104)
    print(f"  -GEX HUNT viability   ({len(tickers)} liquid names)   REALISTIC ask/bid fill   split {SPLIT}")
    print("=" * 104)
    for gg in ("pct80", "z2.0", "raw"):
        print(f"\n  flow gate = {gg}   (total gated trades {gate_n[gg]})")
        print(f"    prior-day NEG-GEX  {_stat(B[gg]['NEG'])}")
        print(f"    prior-day POS-GEX  {_stat(B[gg]['POS'])}")
    print("\n  within NEG-GEX + z>=2 gate:")
    for k, v in {**negdeep, **negvol}.items():
        print(f"    {k:12}  {_stat(v)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=45)
    a = ap.parse_args()
    if a.build:
        u = _universe(a.n)
        print(f"  building {len(u)}: {' '.join(u)}")
        build(u)
    else:
        run(a)
