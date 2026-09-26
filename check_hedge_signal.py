# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_hedge_signal.py
=====================
HEAD-TO-HEAD: does DELTA-WEIGHTED DEALER HEDGING DEMAND beat NET PREMIUM as the
bot's trigger series? This is not a gate test -- it is a candidate replacement
for `net_flow_1m` itself.

The deployed trigger is an EMA(5) crossover of cumulative intraday
`net_premium = net_call_premium - net_put_premium` (UW). That is aggressor-
signed, so its DIRECTION is right, but it is weighted by PREMIUM DOLLARS while
hedging scales with DELTA x CONTRACTS. Verified: premium and contract-count
aggression agree on sign only 73-78% of the time.

`build_hedge_flow.py` produced the mechanically correct series, per minute:
    hedge_sh = SUM (ask_vol - bid_vol) * delta * 100     (+ = dealers must BUY)

SERIES COMPARED (identical trigger construction, identical rule gates,
identical exits, identical sequential guard -- ONLY the underlying series changes):

  prem      cumulative UW net_premium                        <- DEPLOYED
  hedge     cumulative hedge_sh, whole chain
  hedge01   cumulative hedge_sh restricted to DTE 0-1        (gamma-heaviest)
  hratio    cumulative hedge_sh / cumulative gross_sh        (bounded [-1,1],
            drift-resistant -- check_flow_drift showed raw dollar magnitudes
            move 2-23x over two years, which a bounded ratio is immune to)

`min_flow_pct` is a PERCENTILE of the series against its own trailing history
(`annotate_flow_pct`), so it self-calibrates across unit systems and the
comparison stays apples-to-apples despite shares vs dollars.

Scored sequential, realistic fills, live exit cushion modelled, IS/OOS + slices.
A replacement has to beat the deployed series on BOTH halves and hold up in the
walk-forward, not just post a bigger OOS number.

Usage:  python check_hedge_signal.py
        python check_hedge_signal.py --rules "NVDA LOWVOL PUT"
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
from check_exit_walkforward import _eod_mod
from check_giveback import _sim

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _hedge_frame(tk):
    p = f"{HIST}/HEDGE{tk}.parquet"
    if not os.path.exists(p):
        return None
    import polars as pl
    d = pl.read_parquet(p).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d.sort_values(["date", "mod"])


def _series_for(tk, kind, D):
    """{date: (mods, cumulative series)} for the requested construction."""
    if kind == "prem":
        from check_config_walkforward import _flow_for
        f = _flow_for(D, [tk])
        if f.empty:
            return None
        f = f.sort_values("minute_et").copy()
        f["mod"] = f["minute_et"].dt.hour * 60 + f["minute_et"].dt.minute
        out = {}
        for d, g in f.groupby("date"):
            out[d] = (g["mod"].to_numpy(int), g["cum_flow"].to_numpy(float))
        return out
    h = _hedge_frame(tk)
    if h is None:
        return None
    out = {}
    for d, g in h.groupby("date"):
        mods = g["mod"].to_numpy(int)
        if kind == "hedge":
            s = np.cumsum(g["hedge_sh"].to_numpy(float))
        elif kind == "hedge01":
            s = np.cumsum(g["hedge_sh_d01"].to_numpy(float))
        elif kind in ("hedge_sl", "hedge_sld"):
            # SINGLE-LEG variants: a vertical's legs are each aggressor-tagged,
            # but the package's NET delta is far smaller than the legs imply, so
            # multi-leg volume inflates hedge_sh. Multi-leg share of volume runs
            # 12.2% (SPY) to 32.8% (GLD). This isolates whether the ~2.9pp
            # weighting deficit is really about DELTA-vs-PREMIUM weighting or
            # just about spread contamination -- which is the one thing
            # SpotGamma's proprietary HIRO classifier plausibly handles better.
            #   hedge_sl   netbuy scaled by the single-leg share
            #   hedge_sld  contract-minutes >50% multi-leg dropped entirely
            col = "hedge_sh_sl" if kind == "hedge_sl" else "hedge_sh_sld"
            if col not in g.columns:
                raise SystemExit(
                    f"{col} missing from HEDGE parquet -- re-run "
                    f"build_hedge_flow.py, the single-leg columns were added "
                    f"2026-09-14.")
            s = np.cumsum(g[col].to_numpy(float))
        elif kind == "lakeprem":
            # CONTROL: premium reconstructed from the SAME lake rows as `hedge`,
            # i.e. sum (ask_vol - bid_vol) * vwap * 100. If this matches UW `prem`,
            # the lake extraction is sound and delta-weighting is genuinely worse.
            # If it also underperforms, the problem is the extraction, not the
            # weighting -- and the whole comparison is invalid.
            s = np.cumsum(g["net_prem_lake"].to_numpy(float))
        elif kind == "hratio":
            num = np.cumsum(g["hedge_sh"].to_numpy(float))
            den = np.cumsum(np.abs(g["gross_sh"].to_numpy(float)))
            s = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        else:
            raise ValueError(kind)
        out[d] = (mods, s)
    return out


def _trigs(series, base_date_ts):
    """Same construction as directional_flow_backtester.triggers_for: EMA(5)
    crossover of the cumulative series. Returns trigger dicts."""
    out = []
    for d, (mods, cum) in series.items():
        if len(cum) < 6:
            continue
        ema = pd.Series(cum).ewm(span=5, adjust=False).mean().values
        for i in range(1, len(cum)):
            bull = cum[i - 1] <= ema[i - 1] and cum[i] > ema[i]
            bear = cum[i - 1] >= ema[i - 1] and cum[i] < ema[i]
            if bull or bear:
                m = int(mods[i])
                out.append({"date": d, "ts": base_date_ts(d, m), "hour": m // 60,
                            "dir": "CALL" if bull else "PUT",
                            "abs_flow": abs(float(cum[i])), "mod": m})
    return out


def _build(D, r, kind):
    from amt_profile import amt_open_map, amt_ok
    tk = r["ticker"]
    ser = _series_for(tk, kind, D)
    if not ser:
        return []
    gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
               "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}

    def _ts(d, m):
        return pd.Timestamp(d) + pd.Timedelta(minutes=int(m))

    trigs = _trigs(ser, _ts)
    if not trigs:
        return []
    D.annotate_flow_pct(trigs, r.get("flow_window_days", 60))
    try:
        tb = D._ticker_bars(tk)
    except Exception:
        tb = None
    if tb is None or tb.empty:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
    if tb is None or tb.empty:
        return []
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}
    amt = amt_open_map(tk) if r.get("amt_open") else {}
    matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
    if r.get("amt_open"):
        matched = [(t, th) for t, th in matched if amt_ok(r["amt_open"], amt.get(t["date"]))]
    matched.sort(key=lambda x: pd.Timestamp(x[0]["ts"]))

    cand = []
    for t, _th in matched:
        d, ts = t["date"], t["ts"]
        day = bbd.get(d)
        if day is None:
            continue
        at = day[day["minute_et"] <= ts]
        if at.empty:
            continue
        spot = float(at.iloc[-1]["underlying_close"])
        cid = None
        for dd in r.get("dte", [0, 1]):
            cid = D.pick_contract(day, ts, r["direction"], dd, spot)
            if cid is not None:
                break
        if cid is None:
            continue
        ent = bbc[cid]
        er = ent[(ent["minute_et"] <= ts) & (ent["minute_et"] >= ts - pd.Timedelta(minutes=3))]
        if er.empty:
            continue
        er = er.iloc[-1]
        b, k = float(er["bid_close"]), float(er["ask_close"])
        mid = (b + k) / 2.0 if b > 0 else float(er["close"])
        if mid < 0.50:
            continue
        fwd = ent[ent["minute_et"] > ts].sort_values("minute_et")
        if len(fwd) < 3:
            continue
        pm = fwd["minute_et"]
        cand.append((d, t["mod"],
                     (mid, k if k > 0 else mid, fwd["close"].to_numpy(float),
                      fwd["high"].to_numpy(float), fwd["low"].to_numpy(float),
                      fwd["bid_close"].to_numpy(float), fwd["ask_close"].to_numpy(float),
                      (pm.dt.hour.values * 60 + pm.dt.minute.values).astype(int))))
    return cand


def _walk(cand, pol, em):
    cur, busy, out = None, -1, []
    for d, m, path in cand:
        if d != cur:
            cur, busy = d, -1
        if m < busy:
            continue
        pnl, xm, _t = _sim(path, pol, em, True, True)
        out.append((d, pnl))
        busy = xm
    return out


def _st(lbl, rows, base=None):
    if len(rows) < 12:
        return f"    {lbl:22} n={len(rows):>4}  (thin)"
    v = np.array([p for _, p in rows], float)
    i = np.array([p for d, p in rows if d < SPLIT], float)
    o = np.array([p for d, p in rows if d >= SPLIT], float)
    sl = [[] for _ in range(6)]
    for d, p in rows:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    pop = [np.mean(b) for b in sl if len(b) >= 3]
    dl = f" {(o.mean()-base)*100:>+6.1f}pp" if (base is not None and len(o)) else " " * 9
    return (f"    {lbl:22} n={len(v):>4} d={len({d for d,_ in rows}):>4} "
            f"IS {i.mean()*100 if len(i) else float('nan'):>+7.1f}% "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+7.1f}%{dl} "
            f"win {(v>0).mean():>4.2f} sl {sum(1 for x in pop if x>0)}/{len(pop)}")


def run(a):
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.rules:
        rules = [r for r in rules if r["name"] in a.rules]
    # per-rule deployed exit: META/NVDA carry trail_pct 0, so a book-wide
    # trail would score trades those rules never take
    import sim_core
    KINDS = ["prem", "lakeprem", "hedge", "hedge_sl", "hedge_sld",
             "hedge01", "hratio"]
    book = {k: [] for k in KINDS}

    print("=" * 116)
    print("  DELTA-WEIGHTED HEDGING DEMAND vs NET PREMIUM  --  as the TRIGGER series")
    print("  identical gates, exits and sequential guard; only the series differs")
    print("=" * 116)
    for r in rules:
        print(f"\n  {r['name']}   ({r['ticker']} {r['direction']}, p{r.get('min_flow_pct')})")
        base = None
        for k in KINDS:
            c = _build(D, r, k)
            if not c:
                print(f"    {k:22} (no candidates)")
                continue
            rows = _walk(c, sim_core.policy_for(r, TRAIL_PCT), _eod_mod(r))
            book[k] += rows
            if k == "prem":
                base = np.mean([p for d, p in rows if d >= SPLIT]) if rows else None
            print(_st(k + (" (DEPLOYED)" if k == "prem" else ""), rows, base if k != "prem" else None))

    print("\n" + "=" * 116)
    print("  BOOK TOTAL")
    print("=" * 116)
    b = np.mean([p for d, p in book["prem"] if d >= SPLIT]) if book["prem"] else None
    for k in KINDS:
        print(_st(k + (" (DEPLOYED)" if k == "prem" else ""), book[k], b if k != "prem" else None))

    print("\n" + "=" * 116)
    print("  WALK-FORWARD THE SERIES CHOICE -- picked only on data before each slice")
    print("=" * 116)
    sel, dep = [], []
    for kk in range(1, 6):
        picks = {}
        for k in KINDS:
            prior = [p for d, p in book[k] if _slice_idx(d) is not None and _slice_idx(d) < kk]
            if len(prior) >= 20:
                picks[k] = float(np.mean(prior))
        if not picks:
            continue
        best = max(picks, key=picks.get)
        cur = [p for d, p in book[best] if _slice_idx(d) == kk]
        dp = [p for d, p in book["prem"] if _slice_idx(d) == kk]
        if not cur:
            continue
        print(f"  S{kk+1}  picked {best:8} scored {np.mean(cur)*100:>+7.1f}%   "
              f"deployed {np.mean(dp)*100 if dp else float('nan'):>+7.1f}%")
        sel += cur; dep += dp
    if sel:
        print(f"\n  chained: selected {np.mean(sel)*100:>+7.1f}% (n={len(sel)})   "
              f"vs deployed {np.mean(dep)*100:>+7.1f}% (n={len(dep)})   "
              f"-> {'SELECTION WINS' if np.mean(sel) > np.mean(dep) else 'DEPLOYED WINS'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="*", default=None)
    run(ap.parse_args())
