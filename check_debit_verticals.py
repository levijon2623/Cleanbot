# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_debit_verticals.py
=========================
For every ENABLED config.RULES entry, at the same flow triggers, compare the
deployed BARE long-option trade against a 0/1DTE DEBIT VERTICAL:

  CALL rule -> bull call spread : long ATM call  / short a call `width` OTM
  PUT  rule -> bear put spread  : long ATM put   / short a put  `width` OTM

The spread costs less (short leg credit), caps max profit at the strike width,
and bleeds less theta -- the question is whether that beats the lost 0DTE gamma
convexity of the bare option for these ~52-55%-directional, target-based setups.

Grid: width (short strike, % OTM) x exit (spread target_roe, or hold-to-EOD).
Reuses directional_flow_backtester triggers + bars.  IS/OOS @ 2025-08-21.

Usage:
  python check_debit_verticals.py
  python check_debit_verticals.py --widths 0.5 1 1.5 2 --targets 0.4 0.6 1.0
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD_MOD = 15 * 60 + 55
COMMISSION = 0.015           # per spread leg-pair round trip, fraction of net debit -- approx


def _leg_series(bbc, cid, ts):
    df = bbc.get(cid)
    if df is None:
        return None
    fwd = df[df["minute_et"] > ts].sort_values("minute_et")
    return fwd if len(fwd) >= 3 else None


def _slice(day, ts, otype, dte):
    """last quote per contract at/just before ts for the right type/expiry;
    returns df with strike + option_chain_id, sorted by strike."""
    at = day[(day["minute_et"] <= ts) & (day["minute_et"] >= ts - pd.Timedelta(minutes=3))]
    at = at[at["option_type"] == otype]
    if at.empty:
        return None
    ed = pd.Timestamp(ts).date()
    at = at.assign(_dte=(at["expiry"] - ed).map(lambda x: x.days))
    at = at[at["_dte"] == dte]
    if at.empty:
        return None
    return at.sort_values("minute_et").groupby("option_chain_id").last().reset_index().sort_values("strike")


def _ladder_legs(bbd, bbc, t, direction, dtes, long_otm, wide):
    """long = `long_otm`-th strike OTM, short = `wide` strikes further OTM.
    Returns (net_debit, strike_width, val_arr, mod_arr) or None."""
    day = bbd.get(t["date"])
    if day is None:
        return None
    at0 = day[day["minute_et"] <= t["ts"]]
    if at0.empty:
        return None
    spot = float(at0.iloc[-1]["underlying_close"])
    otype = "call" if direction == "CALL" else "put"
    for dte in dtes:
        sl = _slice(day, t["ts"], otype, dte)
        if sl is None or len(sl) < long_otm + wide + 1:
            continue
        ks = sl["strike"].values.astype(float)
        if direction == "CALL":
            otm = np.where(ks > spot)[0]
            if len(otm) < long_otm + wide:
                continue
            li = otm[long_otm - 1]; si = otm[long_otm - 1 + wide]
        else:
            otm = np.where(ks < spot)[0][::-1]        # descending toward spot
            if len(otm) < long_otm + wide:
                continue
            li = otm[long_otm - 1]; si = otm[long_otm - 1 + wide]
        lrow, srow = sl.iloc[li], sl.iloc[si]
        w = abs(float(srow["strike"]) - float(lrow["strike"]))
        if w <= 0:
            continue
        ls = _leg_series(bbc, lrow["option_chain_id"], t["ts"])
        ss = _leg_series(bbc, srow["option_chain_id"], t["ts"])
        if ls is None or ss is None:
            continue
        le, se = ls.iloc[0], ss.iloc[0]
        l_ent = float(le["ask_close"]) if le["ask_close"] > 0 else float(le["close"])
        s_ent = float(se["bid_close"]) if se["bid_close"] > 0 else float(se["close"])
        net_debit = l_ent - s_ent
        if net_debit < 0.05 or net_debit >= w:
            continue
        m = pd.merge(ls[["minute_et", "close"]].rename(columns={"close": "l"}),
                     ss[["minute_et", "close"]].rename(columns={"close": "s"}),
                     on="minute_et", how="inner")
        if len(m) < 3:
            continue
        val = (m["l"].astype(float) - m["s"].astype(float)).clip(lower=0, upper=w).values
        mod = (m["minute_et"].dt.hour * 60 + m["minute_et"].dt.minute).values
        return net_debit, w, val, mod
    return None


def _spread_exit(net_debit, val, mod, tr, hold_eod):
    cummax = np.maximum.accumulate(val)
    cummin = np.minimum.accumulate(val)
    eod_i = int(np.argmax(mod >= EOD_MOD)) if (mod >= EOD_MOD).any() else len(val) - 1
    if hold_eod:
        exit_val = val[eod_i]
    else:
        tp = net_debit * (1 + tr)
        sl = net_debit * (1 - tr)
        tp_i = int(np.searchsorted(cummax, tp)) if cummax[-1] >= tp else len(val)
        sl_i = int(np.searchsorted(-cummin, -sl)) if cummin[-1] <= sl else len(val)
        xi = min(tp_i, sl_i, eod_i)
        exit_val = (tp if tp_i == xi and tp_i <= sl_i else
                    sl if sl_i == xi else val[min(xi, len(val) - 1)])
    return (exit_val - net_debit) / net_debit - COMMISSION


def run(a):
    import directional_flow_backtester as D
    from config import RULES
    try:
        from amt_profile import amt_open_map, amt_ok
    except Exception:
        amt_open_map = amt_ok = None

    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date
    rules = [r for r in RULES if r.get("enabled", True)]

    print("=" * 104)
    print("  0/1DTE OTM-LADDER DEBIT VERTICAL vs BARE OPTION  (long Nth-OTM strike / short M strikes further)")
    print("  per deployed rule, IS/OOS @ 2025-08-21, fees in")
    print("=" * 104)

    grand = {}
    for r in rules:
        tk = r["ticker"]; direction = r["direction"].upper()
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        ema_stack = D.load_ema_stack(HIST, tk, int(r["ema_confirm"])) if r.get("ema_confirm") else None
        amt = amt_open_map(tk) if (r.get("amt_open") and amt_open_map) else {}

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
        tstop = r.get("time_stop_mins"); dtes = r.get("dte", [0, 1])
        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
        mt = []
        for t, thr in matched:
            d, ts = t["date"], t["ts"]
            if r.get("amt_open") and amt_ok and not amt_ok(r["amt_open"], amt.get(d)):
                continue
            if ema_stack is not None:
                st = D.ema_state_at(ema_stack, ts)
                if st is not None and st != ("BULL" if direction == "CALL" else "BEAR"):
                    continue
            mt.append(t)

        bare = []
        for t in mt:
            for p in D._option_paths(t, direction, dtes, bbd, bbc):
                bare.append((t["date"], D._bracket_pnl(*p, base_tr, rr, tstop, EOD_MOD)))
        b_is = [x for d, x in bare if d < SPLIT]; b_oos = [x for d, x in bare if d >= SPLIT]

        print(f"\n  {r['name']}  ({tk} {direction}, bare target_roe={base_tr} rr={rr})")
        print(f"    BARE                       n={len(bare):>4}  IS {np.mean(b_is)*100 if b_is else float('nan'):>+6.1f}%  "
              f"OOS {np.mean(b_oos)*100 if b_oos else float('nan'):>+6.1f}%  win {np.mean([x>0 for _,x in bare]):.2f}")

        exits = [("EOD", None, True)] + [(f"tp{t}", t, False) for t in a.targets]
        best = None
        for long_otm in a.long_otm:
            for wide in a.wide:
                paths = [(t["date"], _ladder_legs(bbd, bbc, t, direction, dtes, long_otm, wide)) for t in mt]
                paths = [(d, p) for d, p in paths if p is not None]
                if len(paths) < max(15, len(bare) * 0.25):
                    continue
                nd0 = np.mean([nd for _, (nd, w, val, mod) in paths])
                w0 = np.mean([w for _, (nd, w, val, mod) in paths])
                for nm, tr, heod in exits:
                    res = [(d, _spread_exit(nd, val, mod, tr or 0.0, heod)) for d, (nd, w, val, mod) in paths]
                    ri = [x for d, x in res if d < SPLIT]; ro = [x for d, x in res if d >= SPLIT]
                    oe = np.mean(ro) * 100 if ro else float("nan")
                    ie = np.mean(ri) * 100 if ri else float("nan")
                    tag = f"L{long_otm}otm/W{wide} {nm}"
                    print(f"    {tag:20}  n={len(res):>4}  debit~{nd0:.2f}/w{w0:.2f}  "
                          f"IS {ie:>+6.1f}%  OOS {oe:>+6.1f}%  win {np.mean([x>0 for _,x in res]):.2f}")
                    if ro and (best is None or oe > best[1]):
                        best = (tag, oe, ie, len(res))
        if best:
            b_oos_m = np.mean(b_oos) * 100 if b_oos else float("nan")
            print(f"    -> best spread OOS {best[1]:+.1f}% ({best[0]}, n={best[3]})  vs bare OOS {b_oos_m:+.1f}%")
            grand[r["name"]] = (b_oos_m, best[1])

    if grand:
        print("\n" + "-" * 60)
        print(f"  {'rule':22}{'bare OOS':>10}{'best spread OOS':>18}")
        tb_, ts_ = [], []
        for nm, (b, s) in grand.items():
            print(f"  {nm:22}{b:>+9.1f}%{s:>+17.1f}%")
            tb_.append(b); ts_.append(s)
        print(f"  {'MEAN':22}{np.mean(tb_):>+9.1f}%{np.mean(ts_):>+17.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long-otm", nargs="+", type=int, default=[1, 2],
                    help="long leg = Nth strike OTM (1 = nearest OTM)")
    ap.add_argument("--wide", nargs="+", type=int, default=[1, 2],
                    help="short leg = this many strikes further OTM than the long")
    ap.add_argument("--targets", nargs="+", type=float, default=[1.0, 2.0])
    a = ap.parse_args()
    run(a)
