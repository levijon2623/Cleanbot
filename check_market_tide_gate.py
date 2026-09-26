# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_market_tide_gate.py
=========================
ml_feature_scan.py flagged ONE new, IS/OOS-consistent candidate: for the INDEX
CALL rules, the outcome improves when MARKET TIDE (market-wide cumulative
net-call-premium, ncp - npp, from _tide_cache) is high at the trigger minute --
top-tide-quintile index-CALL triggers were ~breakeven OOS while the rest bled
-15%.  rho IS +0.060 / OOS +0.063.

This validates it with the check_screen_candidate battery -- for each index CALL
rule (PLAIN: regime + min_flow_pct + hours, i.e. before its deployed extra gates)
and for the pooled index-CALL set:

  * BASELINE                IS/OOS/6-slice/maxLL/win
  * tide QUINTILE           mean pnl per quintile of the normalised tide level
  * tide >= 0 / < 0         the simplest gate
  * tide top-40% / top-60%  the ML "high tide" gate
  * threshold sweep on the normalised level
  * BOOTSTRAP NULL          gated OOS vs p95 of a same-size random OOS subsample

Tide level is normalised exactly as in ml_feature_scan: cum(ncp-npp) to the
trigger minute / trailing-60-session median of that day's end-of-day |cum|
(lookahead-free -- prior sessions only).

Usage:
  python check_market_tide_gate.py
  python check_market_tide_gate.py --rules QQQ:HIVOL:95 SPY:CHOP:90 IWM:HIVOL:80
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
from check_screen_candidate import _stat, _boot

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD = 15 * 60 + 55
TIDE_CACHE = "_tide_cache/market_tide.parquet"
DEFAULT_RULES = ["QQQ:HIVOL:95", "SPY:CHOP:90", "IWM:HIVOL:80"]


def _tide_level():
    """{date: (mod_arr, norm_cum_arr)} -- cum(ncp-npp) to each minute, divided by
    the trailing-60-session median of that day's end |cum| (prior sessions only)."""
    import polars as pl
    df = pl.read_parquet(TIDE_CACHE).to_pandas()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = df["ts"].dt.date
    df["mod"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[(df["mod"] >= 570) & (df["mod"] <= 960)].sort_values(["date", "mod"])
    df["cum"] = (df.groupby("date", group_keys=False)["ncp"].cumsum()
                 - df.groupby("date", group_keys=False)["npp"].cumsum())
    eod = df.groupby("date")["cum"].last().abs()
    eod.index = pd.to_datetime(list(eod.index))
    scale = {d.date(): v for d, v in eod.rolling(60, min_periods=15).median().shift(1).items()}
    out = {}
    for d, g in df.groupby("date"):
        sc = scale.get(d, np.nan)
        if sc and np.isfinite(sc) and sc > 0:
            out[d] = (g["mod"].to_numpy(), (g["cum"].to_numpy(float) / sc))
    return out


def _tide_at(tl, date, ts):
    arr = tl.get(date)
    if not arr:
        return np.nan
    m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
    mm, vv = arr
    i = int(np.searchsorted(mm, m, side="right")) - 1
    return vv[i] if i >= 0 else np.nan


def _bootline(rows, gated, tag):
    r = _boot(rows, np.mean([p for d, p in gated if d >= SPLIT]) * 100,
              len([p for d, p in gated if d >= SPLIT]), k=2000)
    if not r:
        return
    mean, p95, rank = r
    g = np.mean([p for d, p in gated if d >= SPLIT]) * 100
    verdict = "** BEATS null" if g > p95 else "(within noise)"
    print(f"    boot [{tag}] OOS n={len([p for d,p in gated if d>=SPLIT])}: gated {g:+.1f}%  "
          f"vs random {mean:+.1f}% / p95 {p95:+.1f}%  ({rank:.0f}th pct)  {verdict}")


def run(a):
    import directional_flow_backtester as D

    tl = _tide_level()
    tdates = sorted(tl)
    print(f"  market tide: {tdates[0]} .. {tdates[-1]}  ({len(tdates)} sessions)")

    flow_cache = {}
    pooled = []          # (date, pnl, tide) across all index CALL rules

    specs = [s.split(":") for s in (a.rules or DEFAULT_RULES)]
    for tk, reg, pct in specs:
        pct = int(pct)
        if tk not in flow_cache:
            from check_config_walkforward import _flow_for
            flow_cache[tk] = _flow_for(D, [tk])
        flow = flow_cache[tk]
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {dd: g for dd, g in tb.groupby("date")}

        rule = {"ticker": tk, "direction": "CALL", "hours": [9, 10, 11, 12, 13, 14],
                "regime": reg, "min_flow_pct": pct}
        matched = D._rule_matched_trigs(rule, trigs, gex, vol, trd, amp, reg_src)
        rows = []
        for t, _thr in matched:
            tv = _tide_at(tl, t["date"], t["ts"])
            for p in D._option_paths(t, "CALL", [0, 1], bbd, bbc):
                pnl = D._bracket_pnl(*p, 1.0, 1.0, None, EOD)
                rows.append((t["date"], pnl, tv))
                pooled.append((t["date"], pnl, tv))
        _report(f"{tk} {reg} CALL p{pct}  (plain)", rows)

    _report("POOLED index CALL (QQQ+SPY+IWM)", pooled)


def _report(title, rows):
    rr = [(d, p) for d, p, tv in rows if np.isfinite(tv)]
    if len(rr) < 30:
        print(f"\n  {title}: n={len(rr)} -- too thin\n"); return
    tv = np.array([tv for d, p, tv in rows if np.isfinite(tv)])
    base = [(d, p) for d, p, t in rows if np.isfinite(t)]
    bn = len(base)
    print("\n" + "=" * 108)
    print(f"  {title}   ({bn} option-trades w/ tide)   split {SPLIT}")
    print("=" * 108)
    print(f"  {'BASELINE':30} {_stat(base)}")

    # quintiles
    q = pd.qcut(pd.Series(tv), 5, labels=False, duplicates="drop")
    cells = []
    for k in range(int(np.nanmax(q)) + 1):
        sub = [(base[i][0], base[i][1]) for i in range(bn) if q.iloc[i] == k]
        vi = [p for d, p in sub if d < SPLIT]; vo = [p for d, p in sub if d >= SPLIT]
        cells.append(f"Q{k+1}[{np.mean([x for _,x in sub])*100:+.0f}: IS{np.mean(vi)*100 if vi else float('nan'):+.0f}/OOS{np.mean(vo)*100 if vo else float('nan'):+.0f} n{len(sub)}]")
    print(f"  tide quintile (low->high):  " + "  ".join(cells))

    for lbl, mask in (("tide >= 0", tv >= 0), ("tide < 0", tv < 0),
                      ("tide top 60%", tv >= np.nanquantile(tv, 0.40)),
                      ("tide top 40%", tv >= np.nanquantile(tv, 0.60)),
                      ("tide top 20%", tv >= np.nanquantile(tv, 0.80))):
        sub = [base[i] for i in range(bn) if mask[i]]
        print(f"  {lbl:30} {_stat(sub, bn)}")

    # threshold sweep
    print("  threshold sweep (keep tide >= x):")
    for x in (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
        sub = [base[i] for i in range(bn) if tv[i] >= x]
        if len(sub) >= 15:
            print(f"    >= {x:+.2f}   {_stat(sub, bn)}")

    top40 = [base[i] for i in range(bn) if tv[i] >= np.nanquantile(tv, 0.60)]
    if len([p for d, p in top40 if d >= SPLIT]) >= 10:
        _bootline([(d, p) for d, p, t in rows if np.isfinite(t)], top40, "tide top 40%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="+", default=None, help="TK:REGIME:PCT specs (default QQQ/SPY/IWM index CALLs)")
    a = ap.parse_args()
    run(a)
