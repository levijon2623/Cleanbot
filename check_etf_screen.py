# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_etf_screen.py
===================
Generalised netprem / live-units validation for a --screen hit, the same path
SMH LOWVOL PUT took (see check_smh_put.py, which this replaces for SOXL/XLE/XLF).

Builds the ticker's netprem flow from historical/NETPREM{T}.parquet, runs the
EMA(5) cum-flow triggers, and for the given direction sweeps min_flow_pct x the
volume/trend regime x the AMT open-location gate x hours.  IS/OOS @ 2025-08-21.

Needs: historical/NETPREM{T}.parquet, GEX{T}.parquet, {T}.parquet, and
_screen_cache/{T}_bars.parquet (from directional_flow_backtester --screen).

Usage:
  python check_etf_screen.py XLE PUT
  python check_etf_screen.py SOXL CALL --regime NORMVOL
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD = 15 * 60 + 55


def _stat(pnls):
    """Trade-weighted stats PLUS the day-level (equal-weight-per-day) cut.

    A rule that fires 20x on 2 good days shows n=71 / win 0.76 trade-weighted but
    is really ~2 bets (see WMT LOWVOL CALL, session 18: trade +18.5/+31.3 ->
    day-level -16.3/+20.2 on 8 days, day-win 0.38). Intraday triggers on one
    ticker/day are heavily correlated -- always read `d=` and the D[..] block."""
    if len(pnls) < 8:
        return f"n={len(pnls):>3}  d={len({d for d, _ in pnls}):>3}  (thin)"
    v = [p for _, p in pnls]
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    c = m = 0
    for _, p in pnls:
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    day = {}
    for d, p in pnls:
        day.setdefault(d, []).append(p)
    dm = {d: float(np.mean(x)) for d, x in day.items()}
    di = [x for d, x in dm.items() if d < SPLIT]
    do = [x for d, x in dm.items() if d >= SPLIT]
    dayblk = (f"D[d={len(dm):>3} IS {np.mean(di) * 100 if di else float('nan'):>+6.1f}%(d{len(di)}) "
              f"OOS {np.mean(do) * 100 if do else float('nan'):>+6.1f}%(d{len(do)}) "
              f"win {np.mean([x > 0 for x in dm.values()]):.2f}]")
    return (f"n={len(pnls):>3}  d={len(dm):>3}  exp {np.mean(v) * 100:>+6.1f}%  "
            f"win {np.mean([x > 0 for x in v]):.2f}  "
            f"IS {np.mean(i) * 100 if i else float('nan'):>+6.1f}%(n{len(i)})  "
            f"OOS {np.mean(o) * 100 if o else float('nan'):>+6.1f}%(n{len(o)})  maxLL {m}  {dayblk}")


def _netprem_flow(D, tk):
    df = pl.read_parquet(f"{HIST}/NETPREM{tk}.parquet").to_pandas()
    df["minute_et"] = D._naive(df["minute_et"])
    df["date"] = df["minute_et"].dt.date
    df = df.sort_values("minute_et")
    df["net_flow_1m"] = pd.to_numeric(df["net_premium"], errors="coerce").fillna(0.0)
    df["underlying_symbol"] = tk
    df["cum_flow"] = df.groupby("date")["net_flow_1m"].cumsum()
    return df[["underlying_symbol", "minute_et", "date", "net_flow_1m", "cum_flow"]]


_DOW = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}


def run(tk, direction, want_regime, dow=None):
    import directional_flow_backtester as D
    dow_keep = {_DOW[d.lower()[:3]] for d in dow} if dow else None
    flow = _netprem_flow(D, tk)
    print(f"  {tk} netprem: {flow['date'].min()} .. {flow['date'].max()}  ({flow['date'].nunique()} days)")

    gex = D.load_gex(HIST, tk)
    vol = D.load_volume_regime(HIST, tk)
    trd = D.load_trend_regime(HIST, tk)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
               "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
    try:
        from amt_profile import amt_open_map
        amt = amt_open_map(tk)
    except Exception as e:
        print(f"  amt_open_map failed: {e}")
        amt = {}
    print(f"  volume days: LOWVOL {sum(v == 'LOWVOL' for v in vol.values())}, "
          f"NORMVOL {sum(v == 'NORMVOL' for v in vol.values())}, "
          f"HIVOL {sum(v == 'HIVOL' for v in vol.values())}   "
          f"| trend: UP {sum(v == 'UPTREND' for v in trd.values())}, "
          f"CHOP {sum(v == 'CHOP' for v in trd.values())}, "
          f"DOWN {sum(v == 'DOWNTREND' for v in trd.values())}")

    trigs = D.triggers_for(flow, tk)
    tkf, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
    if tb is None or tb.empty:
        print("  no option bars"); return
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}
    D.annotate_flow_pct(trigs, 60)

    regimes = ([want_regime] if want_regime else
               [None, "LOWVOL", "NORMVOL", "HIVOL", "UPTREND", "CHOP", "DOWNTREND",
                "POSITIVE_GEX", "NEGATIVE_GEX"])
    for reg in regimes:
        for pct in (50, 65, 80, 90):
            rule = {"ticker": tk, "direction": direction, "hours": [9, 10, 11, 12, 13, 14],
                    "dte": [0, 1], "min_flow_pct": pct}
            if reg:
                rule["regime"] = reg
            matched = D._rule_matched_trigs(rule, trigs, gex, vol, trd, amp, reg_src)
            rows = []
            for t, thr in matched:
                wd = t["date"].weekday()
                if dow_keep is not None and wd not in dow_keep:
                    continue
                # weekly-only expiry: Thu can only do 1DTE, Fri only 0DTE
                dtes = [0, 1] if dow_keep is None else ([1] if wd == 3 else [0] if wd == 4 else [0, 1])
                for p in D._option_paths(t, direction, dtes, bbd, bbc):
                    rows.append({"d": t["date"], "hr": t["hour"], "dow": wd,
                                 "trend": trd.get(t["date"], "?"),
                                 "gex": gex.get(t["date"], "?"),
                                 "amt": amt.get(t["date"]),
                                 "pnl": D._bracket_pnl(*p, 1.0, 1.0, None, EOD)})
            A = pd.DataFrame(rows)
            tag = f"{reg or 'ALL-regime'} / {direction}  min_flow_pct={pct}"
            if A.empty:
                print(f"\n  {tag}: no trades"); continue
            allp = [(x.d, x.pnl) for x in A.itertuples()]
            print(f"\n{'=' * 96}\n  {tag}   ({len(A)} option-trades)\n{'=' * 96}")
            print(f"    BASELINE                     {_stat(allp)}")
            for wd, nm in ((0, "Mon"), (1, "Tue"), (2, "Wed"), (3, "Thu"), (4, "Fri")):
                sub = [(x.d, x.pnl) for x in A[A.dow == wd].itertuples()]
                if sub:
                    print(f"    dow={nm}                      {_stat(sub)}")
            for loc in ("below_va", "inside_va", "above_va"):
                sub = [(x.d, x.pnl) for x in A[A.amt == loc].itertuples()]
                if len(sub) >= 8:
                    print(f"    amt={loc:10}             {_stat(sub)}")
            excl = [(x.d, x.pnl) for x in A[A.amt != 'above_va'].itertuples()]
            print(f"    exclude above_va             {_stat(excl)}")
            for g in ("UPTREND", "CHOP", "DOWNTREND"):
                sub = [(x.d, x.pnl) for x in A[A.trend == g].itertuples()]
                if len(sub) >= 8:
                    print(f"    trend={g:10}            {_stat(sub)}")
            for g in ("POSITIVE", "NEGATIVE"):
                sub = [(x.d, x.pnl) for x in A[A.gex == g].itertuples()]
                if len(sub) >= 8:
                    print(f"    gex={g:10}              {_stat(sub)}")
            for lo in (10, 12):
                print(f"    hours {lo}-14                 "
                      f"{_stat([(x.d, x.pnl) for x in A[(A.hr >= lo) & (A.hr < 15)].itertuples()])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("direction", choices=("CALL", "PUT"))
    ap.add_argument("--regime", default=None, help="pin one regime; default sweeps all")
    ap.add_argument("--dow", nargs="+", default=None,
                    help="restrict entries to these weekdays (e.g. --dow thu fri); also "
                         "forces Thu->1DTE, Fri->0DTE for weekly-only-expiry names")
    a = ap.parse_args()
    run(a.ticker.upper(), a.direction, a.regime, a.dow)
