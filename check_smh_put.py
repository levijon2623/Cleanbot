# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_smh_put.py
================
Validate the screen hit "SMH LOWVOL PUT" (silver units, IS +13.5% / OOS +16.8%,
win 63%) on NETPREM / live units -- the same path the deployed rules took.
Then sweep min_flow_pct + the AMT open-location gate (which stabilised ~6 other
rules) + hours.

Needs historical/NETPREMSMH.parquet (netprem-build SMH first), GEXSMH.parquet,
SMH.parquet, and _screen_cache/SMH_bars.parquet (from --screen).  IS/OOS @ 2025-08-21.

Usage:  python check_smh_put.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD = 15 * 60 + 55


def _stat(pnls):
    if len(pnls) < 8:
        return f"n={len(pnls):>3}  (thin)"
    v = [p for _, p in pnls]
    i = [p for d, p in pnls if d < SPLIT]; o = [p for d, p in pnls if d >= SPLIT]
    c = m = 0
    for _, p in pnls:
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    return (f"n={len(pnls):>3}  exp {np.mean(v)*100:>+6.1f}%  win {np.mean([x>0 for x in v]):.2f}  "
            f"IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%(n{len(i)})  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%(n{len(o)})  maxLL {m}")


def _smh_flow(D):
    df = pl.read_parquet(f"{HIST}/NETPREMSMH.parquet").to_pandas()
    df["minute_et"] = D._naive(df["minute_et"])
    df["date"] = df["minute_et"].dt.date
    df = df.sort_values("minute_et")
    df["net_flow_1m"] = pd.to_numeric(df["net_premium"], errors="coerce").fillna(0.0)
    df["underlying_symbol"] = "SMH"
    df["cum_flow"] = df.groupby("date")["net_flow_1m"].cumsum()
    return df[["underlying_symbol", "minute_et", "date", "net_flow_1m", "cum_flow"]]


def run():
    import directional_flow_backtester as D
    tk = "SMH"
    flow = _smh_flow(D)
    print(f"  SMH netprem: {flow['date'].min()} .. {flow['date'].max()}  ({flow['date'].nunique()} days)")

    gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
    try:
        from amt_profile import amt_open_map
        amt = amt_open_map(tk)
    except Exception as e:
        print(f"  amt_open_map failed: {e}"); amt = {}
    print(f"  regime days: LOWVOL {sum(v=='LOWVOL' for v in vol.values())}, "
          f"NORMVOL {sum(v=='NORMVOL' for v in vol.values())}, HIVOL {sum(v=='HIVOL' for v in vol.values())}")

    trigs = D.triggers_for(flow, tk)
    tkf, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
    if tb is None or tb.empty:
        print("  no SMH option bars"); return
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}

    for pct in (50, 65, 80, 90):
        rule = {"ticker": tk, "direction": "PUT", "hours": [9, 10, 11, 12, 13, 14],
                "dte": [0, 1], "regime": "LOWVOL", "min_flow_pct": pct}
        D.annotate_flow_pct(trigs, 60)
        matched = D._rule_matched_trigs(rule, trigs, gex, vol, trd, amp, reg_src)
        rows = []
        for t, thr in matched:
            d, ts = t["date"], t["ts"]
            for p in D._option_paths(t, "PUT", [0, 1], bbd, bbc):
                rows.append({"d": d, "hr": t["hour"], "trend": trd.get(d, "?"), "amt": amt.get(d),
                             "pnl": D._bracket_pnl(*p, 1.0, 1.0, None, EOD)})
        A = pd.DataFrame(rows)
        if A.empty:
            print(f"\n  p{pct}: no trades"); continue
        allp = [(x.d, x.pnl) for x in A.itertuples()]
        print(f"\n{'='*94}\n  min_flow_pct = {pct}   ({len(A)} option-trades)\n{'='*94}")
        print(f"    BASELINE (LOWVOL PUT)         {_stat(allp)}")
        print(f"    -- AMT open-location --")
        for loc in ("below_va", "inside_va", "above_va"):
            print(f"    amt={loc:10}             {_stat([(x.d, x.pnl) for x in A[A.amt == loc].itertuples()])}")
        print(f"    exclude above_va             {_stat([(x.d, x.pnl) for x in A[A.amt != 'above_va'].itertuples()])}")
        print(f"    -- trend at trigger --")
        for g in ("UPTREND", "CHOP", "DOWNTREND"):
            print(f"    trend={g:10}            {_stat([(x.d, x.pnl) for x in A[A.trend == g].itertuples()])}")
        print(f"    -- hours --")
        for lo in (9, 10, 12):
            print(f"    hours {lo}-14                 "
                  f"{_stat([(x.d, x.pnl) for x in A[(A.hr >= lo) & (A.hr < 15)].itertuples()])}")


if __name__ == "__main__":
    run()
