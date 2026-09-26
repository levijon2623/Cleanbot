# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_breakout_filters.py
==========================
Two "breakouts are less likely" filters, tested on the deployed CALL rules
(SPY CHOP CALL, QQQ HIVOL CALL, IWM HIVOL CALL, GLD amp1 CALL) -- alone and
together, in two modes:

  A) SUPPRESS  -- drop the trigger entirely when the filter is bearish
  B) TIGHTEN   -- keep the trade but scale target_roe (stop scales with it,
                  R:R fixed) down by --tighten when the filter is bearish

Filters:
  cor1m  -- Cboe 1-Month Implied Correlation < --cor1m-low (default 8) at the
            trigger date  (macro: dispersion-crowded / fragile mega-cap regime)
  wcall  -- a BID-SIDE (written) OTM-call UOA flag on this ticker in the trailing
            --wcall-lookback sessions  (ticker-level: overhead supply capping
            the move) -- needs _uoa_etf_snapshot.parquet
            (scratchpad build_etf_oi.py: _build_snapshot(["SPY","QQQ","IWM","GLD"]))

Reuses directional_flow_backtester's trigger + option-path machinery.  Full
2yr window, IS/OOS @ 2025-08-21.

Usage:
  python check_breakout_filters.py
  python check_breakout_filters.py --tighten 0.5 --wcall-lookback 3
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HIST = "historical"
CSV = "CBOE 1-Month Implied Correlation Historical Data.csv"
ETF_SNAP = "_uoa_etf_snapshot.parquet"
SPLIT = pd.Timestamp("2025-08-21").date()
CALL_RULE_NAMES = {"SPY CHOP CALL", "QQQ HIVOL CALL", "IWM HIVOL CALL", "GLD amp1 CALL"}


def _cor1m():
    d = pd.read_csv(CSV)
    d["date"] = pd.to_datetime(d["Date"]).dt.date
    d["cor1m"] = pd.to_numeric(d["Price"], errors="coerce")
    return dict(zip(d["date"], d["cor1m"]))


def _wcall_flags(a):
    """{ticker: set(dates)} where BID-SIDE (written) OTM-call premium was in the
    top --wcall-pct of this ticker's own trailing-60d distribution -- a RELATIVE
    flag (index ETFs have constant background overwriting, so an absolute 'any
    flag' test is always-on). Flag persists `a.wcall_lookback` sessions."""
    if not os.path.exists(ETF_SNAP):
        print(f"  WARN {ETF_SNAP} missing -- wcall filter disabled")
        return {}
    s = pd.read_parquet(ETF_SNAP)
    s["date"] = pd.to_datetime(s["date"]).dt.date
    is_call = s["option_type"].str.lower().eq("call")
    aggr = s["ask_vol"] / (s["ask_vol"] + s["bid_vol"]).clip(lower=1)
    otm_written = s[is_call & (s["mny"] > 0.005) & (s["mny"] <= 0.10) & (aggr <= 0.45)].copy()
    daily = otm_written.groupby(["tk", "date"])["day_prem"].sum().reset_index()
    out = {}
    for tk, g in daily.groupby("tk"):
        g = g.sort_values("date").reset_index(drop=True)
        roll_thr = g["day_prem"].rolling(60, min_periods=20).quantile(a.wcall_pct / 100.0).shift(1)
        hot = g["date"][g["day_prem"] >= roll_thr].tolist()
        alld = g["date"].tolist()
        pos = {d: i for i, d in enumerate(alld)}
        live = set()
        for fd in hot:
            i = pos[fd]
            for j in range(i, min(i + a.wcall_lookback + 1, len(alld))):
                live.add(alld[j])
        out[tk] = live
    return out


def run(a):
    import directional_flow_backtester as D
    from config import RULES

    cor = _cor1m()
    wc = _wcall_flags(a)
    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date
    rules = [r for r in RULES if r.get("enabled", True) and r["name"] in CALL_RULE_NAMES]

    recs = []   # per option-trade: rule, date, half, pnl, cor1m_bear, wcall_bear
    for r in rules:
        tk = r["ticker"]
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        ema_stack = D.load_ema_stack(HIST, tk, int(r["ema_confirm"])) if r.get("ema_confirm") else None
        try:
            from amt_profile import amt_open_map, amt_ok
            amt = amt_open_map(tk) if r.get("amt_open") else {}
        except Exception:
            amt, amt_ok = {}, None

        trigs = D.triggers_for(flow, tk)
        if not trigs:
            tkf, _ = D._screen_build_one("lake/silver/option-contracts-1m", tk)
            trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
        D.annotate_flow_pct(trigs, int(r.get("flow_window_days") or 60))
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}

        base_tr, rr = float(r["target_roe"]), float(r["rr"])
        tstop = r.get("time_stop_mins")
        eod_mod = 15 * 60 + 55
        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
        for t, thr in matched:
            d, ts = t["date"], t["ts"]
            if r.get("amt_open") and amt_ok and not amt_ok(r["amt_open"], amt.get(d)):
                continue
            if ema_stack is not None:
                st = D.ema_state_at(ema_stack, ts)
                if st is not None and st != "BULL":
                    continue
            c1 = cor.get(d)
            cbear = (c1 is not None and c1 < a.cor1m_low)
            wbear = d in wc.get(tk, set())
            for p in D._option_paths(t, "CALL", r.get("dte", [0, 1]), bbd, bbc):
                recs.append({"rule": r["name"], "date": d,
                             "half": "IS" if d < SPLIT else "OOS",
                             "pnl_base": D._bracket_pnl(*p, base_tr, rr, tstop, eod_mod),
                             "pnl_tight": D._bracket_pnl(*p, base_tr * a.tighten, rr, tstop, eod_mod),
                             "cor1m_bear": cbear, "wcall_bear": wbear})
    A = pd.DataFrame(recs)
    if A.empty:
        print("  no trades"); return
    A["either"] = A["cor1m_bear"] | A["wcall_bear"]
    A["both"] = A["cor1m_bear"] & A["wcall_bear"]

    print("=" * 96)
    print(f"  BREAKOUT FILTERS on the 4 CALL rules   ({len(A)} option-trades, "
          f"cor1m<{a.cor1m_low}, wcall lookback {a.wcall_lookback}d, tighten x{a.tighten})")
    print("=" * 96)

    def stat(d, col="pnl_base"):
        if len(d) < 8:
            return f"n={len(d):>4} (thin)"
        return (f"n={len(d):>4}  exp {d[col].mean()*100:>+6.1f}%  "
                f"win {(d[col] > 0).mean():.2f}  IS {d[d.half=='IS'][col].mean()*100 if (d.half=='IS').any() else float('nan'):>+6.1f}%  "
                f"OOS {d[d.half=='OOS'][col].mean()*100 if (d.half=='OOS').any() else float('nan'):>+6.1f}%")

    print(f"\n  --- coverage ---")
    for f in ("cor1m_bear", "wcall_bear", "either", "both"):
        print(f"    {f:12} fires on {A[f].mean()*100:4.1f}% of trades ({int(A[f].sum())})")

    print(f"\n  --- BASELINE (no filter) ---")
    print(f"    ALL                 {stat(A)}")

    for f, nm in [("cor1m_bear", "COR1M < %g" % a.cor1m_low), ("wcall_bear", "written-call overhead"),
                  ("either", "EITHER"), ("both", "BOTH")]:
        print(f"\n  === filter: {nm} ===")
        print(f"    filter FIRES (bearish){'':1} {stat(A[A[f]])}")
        print(f"    filter QUIET          {stat(A[~A[f]])}")
        # A) suppress: keep only the quiet trades
        kept = A[~A[f]]
        drop = A[A[f]]
        blended_supp = kept["pnl_base"].mean() * 100
        print(f"    A) SUPPRESS -> keep {len(kept)}/{len(A)}: blended exp {blended_supp:+.1f}%  "
              f"(vs {A['pnl_base'].mean()*100:+.1f}% base; dropped {len(drop)} @ {drop['pnl_base'].mean()*100 if len(drop) else float('nan'):+.1f}%)")
        # B) tighten: flagged trades use pnl_tight, rest use pnl_base
        b = np.where(A[f], A["pnl_tight"], A["pnl_base"])
        bi = np.where(A[A.half == "IS"][f], A[A.half == "IS"]["pnl_tight"], A[A.half == "IS"]["pnl_base"])
        bo = np.where(A[A.half == "OOS"][f], A[A.half == "OOS"]["pnl_tight"], A[A.half == "OOS"]["pnl_base"])
        print(f"    B) TIGHTEN  -> blended exp {b.mean()*100:+.1f}%  IS {bi.mean()*100:+.1f}%  OOS {bo.mean()*100:+.1f}%  "
              f"(vs {A['pnl_base'].mean()*100:+.1f}% base)")

    print(f"\n  --- per-rule x COR1M<{a.cor1m_low} (base P&L) ---")
    for rn in sorted(A["rule"].unique()):
        sub = A[A["rule"] == rn]
        print(f"    {rn:16}  bear: {stat(sub[sub.cor1m_bear])}")
        print(f"    {'':16}  quiet:{stat(sub[~sub.cor1m_bear])}")

    A.to_parquet("_breakout_filters.parquet")
    print(f"\n  wrote _breakout_filters.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cor1m-low", type=float, default=8.0)
    ap.add_argument("--wcall-lookback", type=int, default=3)
    ap.add_argument("--wcall-pct", type=float, default=85.0, help="trailing-60d pct for the written-call flag")
    ap.add_argument("--tighten", type=float, default=0.5)
    a = ap.parse_args()
    run(a)
