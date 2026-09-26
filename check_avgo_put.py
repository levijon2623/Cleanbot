# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_avgo_put.py
=================
AVGO HIVOL PUT is the weakest deployed rule (backtest OOS ~+4-7%/trade, ~54%
win) and went 0/4 in paper -- the flow said PUT while AVGO trended UP for 2
days.  Can a trend / EMA / flow-strength gate stabilize it?  (User also asked
about conditional bracket tightening when trend is bullish -- tested too, but
this project has repeatedly found TP/SL fiddling doesn't help.)

Baseline = the deployed rule (regime HIVOL, min_flow_pct 65, hours 9-14, dte
0/1, tr 1.0 rr 1.0).  Reuses directional_flow_backtester.  IS/OOS @ 2025-08-21.

Usage:
  python check_avgo_put.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD = 15 * 60 + 55


def _consec_loss(pnls):
    m = c = 0
    for _, p in pnls:
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    return m


def _stat(pnls):
    if len(pnls) < 8:
        return f"n={len(pnls):>3}  (thin)"
    v = [p for _, p in pnls]
    i = [p for d, p in pnls if d < SPLIT]; o = [p for d, p in pnls if d >= SPLIT]
    return (f"n={len(pnls):>3}  exp {np.mean(v)*100:>+6.1f}%  win {np.mean([x>0 for x in v]):.2f}  "
            f"IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  "
            f"maxLL {_consec_loss(pnls)}")


def run():
    import directional_flow_backtester as D
    from config import RULES

    r = [x for x in RULES if x["name"] == "AVGO HIVOL PUT"][0]
    tk = "AVGO"
    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date

    gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
    ema3 = D.load_ema_stack(HIST, tk, 3)
    ema5 = D.load_ema_stack(HIST, tk, 5)
    try:
        from amt_profile import amt_open_map
        amt = amt_open_map(tk)
    except Exception:
        amt = {}

    trigs = D.triggers_for(flow, tk)
    if not trigs:
        tkf, _ = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
    tb = D._ticker_bars(tk)
    if tb is None or tb.empty:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}

    for pct in (65, 80, 90):
        rr = dict(r); rr["min_flow_pct"] = pct
        D.annotate_flow_pct(trigs, 60)
        matched = D._rule_matched_trigs(rr, trigs, gex, vol, trd, amp, reg_src)
        # tag each matched trigger
        rows = []
        for t, thr in matched:
            d, ts = t["date"], t["ts"]
            tr_reg = trd.get(d, "?")
            e3 = D.ema_state_at(ema3, ts); e5 = D.ema_state_at(ema5, ts)
            paths = D._option_paths(t, "PUT", r.get("dte", [0, 1]), bbd, bbc)
            for p in paths:
                base = D._bracket_pnl(*p, 1.0, 1.0, None, EOD)
                tight = D._bracket_pnl(*p, 0.5, 1.0, None, EOD)      # user's "tighter when bullish"
                rows.append({"d": d, "hr": t["hour"], "trend": tr_reg, "e3": e3, "e5": e5,
                             "amt": amt.get(d), "base": base, "tight": tight})
        A = pd.DataFrame(rows)
        if A.empty:
            print(f"\n  p{pct}: no trades"); continue
        B = [(x.d, x.base) for x in A.itertuples()]
        print(f"\n{'='*92}\n  min_flow_pct = {pct}   ({len(A)} option-trades)\n{'='*92}")
        print(f"    BASELINE (deployed)          {_stat(B)}")
        print(f"\n    -- trend regime split (PUT wants price falling) --")
        for g in ("UPTREND", "CHOP", "DOWNTREND"):
            print(f"    trend={g:10}             {_stat([(x.d, x.base) for x in A[A.trend == g].itertuples()])}")
        print(f"    drop UPTREND                  {_stat([(x.d, x.base) for x in A[A.trend != 'UPTREND'].itertuples()])}")
        print(f"\n    -- EMA stack at trigger (BEAR = aligned with PUT) --")
        for col, nm in (("e3", "3m"), ("e5", "5m")):
            for st in ("BEAR", "MIXED", "BULL"):
                print(f"    ema{nm}={st:6}               {_stat([(x.d, x.base) for x in A[A[col] == st].itertuples()])}")
            print(f"    ema{nm} != BULL               {_stat([(x.d, x.base) for x in A[A[col] != 'BULL'].itertuples()])}")
        print(f"\n    -- hours --")
        for lo in (9, 12, 13):
            hi = 15
            print(f"    hours {lo}-{hi-1:>2}                  "
                  f"{_stat([(x.d, x.base) for x in A[(A.hr >= lo) & (A.hr < hi)].itertuples()])}")
        print(f"\n    -- amt_open --")
        for loc in ("below_va", "inside_va", "above_va"):
            print(f"    amt={loc:10}               {_stat([(x.d, x.base) for x in A[A.amt == loc].itertuples()])}")
        print(f"\n    -- user's idea: conditional bracket (tr 0.5 when trend UPTREND, else 1.0) --")
        cond = [(x.d, x.tight if x.trend == "UPTREND" else x.base) for x in A.itertuples()]
        print(f"    conditional tighten          {_stat(cond)}")
        allt = [(x.d, x.tight) for x in A.itertuples()]
        print(f"    tr 0.5 always                {_stat(allt)}")

        # promising combo: drop UPTREND + ema5 != BULL
        combo = A[(A.trend != "UPTREND") & (A.e5 != "BULL")]
        print(f"\n    COMBO: drop UPTREND & ema5!=BULL   {_stat([(x.d, x.base) for x in combo.itertuples()])}")
        combo2 = A[(A.trend == "DOWNTREND")]
        print(f"    COMBO: DOWNTREND only              {_stat([(x.d, x.base) for x in combo2.itertuples()])}")


if __name__ == "__main__":
    run()
