# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_vix_overlay.py
====================
Walk-forward of a VIX-regime overlay on the deployed book.

check_regime_state found the book's edge is conditional on prior-day VIX:
+24%/trade / win 0.66 when VIX >= ~18, +7% / win 0.53 below, IS/OOS-consistent,
holds within 5 of 7 rules (GLD/AVGO VIX-agnostic).  This checks whether a
throttle on that is robust:

  threshold forms   fixed {16,17,18,19,20}  and  adaptive: VIX_prev >=
                    trailing-median(VIX, W) for W in {40,60,90,120} sessions
  action            hard gate (skip)  |  size x {0.0, 0.25, 0.5} below threshold
  scope             all rules  vs  exclude GLD + AVGO (VIX-agnostic)

Reported per config, vs baseline:  OOS mean, win, 6 calendar-slice means + SD,
equity-curve MAX DRAWDOWN (size-weighted cum P&L), n retention.  Plus a
bootstrap null on the hard-gate OOS and a per-rule breakdown of the pick.

All lookahead-free: VIX_prev = prior-session close, trailing median over prior
sessions only.

Usage:  python check_vix_overlay.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from check_config_walkforward import _flow_for, _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
VIX_AGNOSTIC = {"GLD amp1 CALL", "AVGO HIVOL PUT"}


def _vix_frame():
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    h = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json"}
    r = requests.get("https://api.unusualwhales.com/api/stock/VIX/volatility/realized",
                     headers=h, params={"timeframe": "2Y"}, timeout=20)
    rows = r.json().get("data", [])
    s = pd.Series({pd.Timestamp(x["date"]): float(x["price"]) for x in rows if x.get("price")}).sort_index()
    df = pd.DataFrame({"vix": s})
    df.index = pd.DatetimeIndex(df.index).date   # keep as datetime.date to match trade dates
    df["vix_prev"] = df["vix"].shift(1)
    for w in (40, 60, 90, 120):
        df[f"med{w}"] = df["vix"].shift(1).rolling(w, min_periods=20).median()
    return df


def _maxdd(seq):
    """max drawdown of a cumulative sum of per-trade returns (chronological)."""
    if not seq:
        return 0.0
    c = np.cumsum(seq)
    peak = np.maximum.accumulate(c)
    return float(np.max(peak - c))


def _line(rows, base_n=None):
    """rows = list of (date, size, ret).  size-weighted."""
    if len(rows) < 10:
        return "thin"
    rows = sorted(rows)
    w = np.array([s for _, s, _ in rows])
    r = np.array([x for _, _, x in rows])
    contrib = w * r
    nz = w > 0
    avg = contrib.sum() / max(w.sum(), 1e-9)          # capital-weighted mean return
    win = np.mean(r[nz] > 0) if nz.any() else np.nan
    is_c = [c for (d, _, _), c in zip(rows, contrib) if d < SPLIT]
    oos_c = [c for (d, _, _), c in zip(rows, contrib) if d >= SPLIT]
    is_w = [s for (d, s, _) in rows if d < SPLIT]
    oos_w = [s for (d, s, _) in rows if d >= SPLIT]
    is_avg = np.sum(is_c) / max(np.sum(is_w), 1e-9) * 100
    oos_avg = np.sum(oos_c) / max(np.sum(oos_w), 1e-9) * 100
    sl = [[] for _ in range(6)]
    for (d, s, x) in rows:
        k = _slice_idx(d)
        if k is not None and s > 0:
            sl[k].append(x)
    smeans = [np.mean(b) * 100 for b in sl if len(b) >= 5]
    sd = np.std(smeans) if smeans else float("nan")
    slc = " ".join(f"S{j+1}{np.mean(b)*100:+.0f}" if len(b) >= 5 else f"S{j+1}··" for j, b in enumerate(sl))
    dd = _maxdd([c for (d, _, _), c in zip(rows, contrib) if d >= SPLIT])
    ret = f" keep {100*w.sum()/base_n:.0f}%" if base_n else ""
    return (f"avg {avg*100:>+6.1f}%{ret}  IS {is_avg:>+6.1f}%  OOS {oos_avg:>+6.1f}%  win {win:.2f}  "
            f"sliceSD {sd:4.1f}  OOSmaxDD {dd*100:5.1f}  [{slc}]")


def run(a):
    import directional_flow_backtester as D
    from amt_profile import amt_open_map, amt_ok
    from config import RULES
    from check_adx_dmi import _intraday_adx, _intra_asof
    from macro_calendar import is_macro_am_day
    from check_open_delay import _rule_trades

    rules = [r for r in RULES if r.get("enabled", True)]
    tickers = sorted({r["ticker"] for r in rules})
    flow_all = _flow_for(D, tickers)
    vix = _vix_frame()

    TR = []
    for tk in tickers:
        tk_rules = [r for r in rules if r["ticker"] == tk]
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        amt = amt_open_map(tk) if any(r.get("amt_open") for r in tk_rules) else {}
        ema_stacks = {int(r["ema_confirm"]): D.load_ema_stack(HIST, tk, int(r["ema_confirm"]))
                      for r in tk_rules if r.get("ema_confirm")}
        dmi_tfs = {int(r["dmi_confirm"].get("tf", 15)) for r in tk_rules if r.get("dmi_confirm")}
        iadx = {tf: _intraday_adx(tk, tf, close_only=True) for tf in dmi_tfs}
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {dd: g for dd, g in tb.groupby("date")}
        for r in tk_rules:
            for d, p, m in _rule_trades(D, r, flow_all, gex, vol, trd, amp, reg_src, amt, amt_ok,
                                        ema_stacks, bbd, bbc, iadx, _intra_asof, is_macro_am_day):
                v = vix.loc[d] if d in vix.index else None
                vp = None if v is None or pd.isna(v["vix_prev"]) else float(v["vix_prev"])
                med = None if v is None else {w: (None if pd.isna(v[f"med{w}"]) else float(v[f"med{w}"]))
                                              for w in (40, 60, 90, 120)}
                TR.append((r["name"], d, p, vp, med))

    print("=" * 118)
    print(f"  VIX-OVERLAY WALK-FORWARD   ({len(TR)} trades, {sum(1 for t in TR if t[3] is None)} pre-VIX-coverage -> full size)")
    print("=" * 118)
    base = [(d, 1.0, p) for (nm, d, p, vp, med) in TR]
    print(f"\n  BASELINE   {_line(base)}")
    bn = sum(s for _, s, _ in base)

    def apply(thr_fn, mult, scope_all=True):
        out = []
        for (nm, d, p, vp, med) in TR:
            if not scope_all and nm in VIX_AGNOSTIC:
                out.append((d, 1.0, p)); continue
            if vp is None:
                out.append((d, 1.0, p)); continue
            thr = thr_fn(med)
            fav = vp >= thr if thr is not None else True
            out.append((d, 1.0 if fav else mult, p))
        return out

    print("\n  -- FIXED threshold, action = SIZE x mult below --")
    for lvl in (16, 17, 18, 19, 20):
        for mult in (0.5, 0.25, 0.0):
            print(f"    VIX>={lvl}  x{mult}   {_line(apply(lambda med, L=lvl: L, mult), bn)}")

    print("\n  -- ADAPTIVE threshold (VIX_prev >= trailing-median W), action = SIZE x mult --")
    for w in (40, 60, 90, 120):
        for mult in (0.5, 0.25, 0.0):
            print(f"    med{w}  x{mult}   {_line(apply(lambda med, W=w: med[W], mult), bn)}")

    print("\n  -- ADAPTIVE med60, x0.5, EXCLUDE GLD+AVGO from the overlay --")
    print(f"    med60 x0.5 ex-agnostic   {_line(apply(lambda med: med[60], 0.5, scope_all=False), bn)}")

    # bootstrap null on the hard-gate (mult=0) med60 OOS
    gated = apply(lambda med: med[60], 0.0)
    g_oos = [x for (d, s, x) in gated if d >= SPLIT and s > 0]
    all_oos = [p for (nm, d, p, vp, med) in TR if d >= SPLIT]
    if len(g_oos) >= 20 and len(all_oos) > len(g_oos) + 5:
        rng = np.random.default_rng(0)
        draws = [np.mean(rng.choice(all_oos, size=len(g_oos), replace=False)) * 100 for _ in range(a.boot)]
        p95 = np.percentile(draws, 95)
        gm = np.mean(g_oos) * 100
        print(f"\n  bootstrap null (med60 hard-gate, OOS n={len(g_oos)}): gated {gm:+.1f}%  "
              f"vs random-subsample mean {np.mean(draws):+.1f}% / p95 {p95:+.1f}%  "
              f"({'** beats null' if gm > p95 else 'within noise'})")

    # per-rule effect of med60 x0.5
    print("\n  -- per-rule: med60 x0.5 (capital-weighted avg return, baseline -> overlay) --")
    ov = apply(lambda med: med[60], 0.5)
    for nm in sorted({t[0] for t in TR}):
        idx = [i for i, t in enumerate(TR) if t[0] == nm]
        bl = np.array([TR[i][2] for i in idx])
        w = np.array([ov[i][1] for i in idx])
        rr = np.array([ov[i][2] for i in idx])
        print(f"    {nm:22}  base {bl.mean() * 100:>+6.1f}%   overlay {(w * rr).sum() / max(w.sum(), 1e-9) * 100:>+6.1f}%  "
              f"(size kept {100 * w.mean():.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    run(a)
