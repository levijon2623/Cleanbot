# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_counterflow.py
====================
The bot enters on an EMA(5) crossover of cumulative net-premium flow. Question:
on the trades that LOSE, does the flow itself turn back against the position
(counter-flow) -- early enough, big enough, often enough -- to use as an exit?

For every deployed-spec trade:
  * baseline P&L (_bracket_pnl) -> winner / loser
  * post-entry favourable flow  fav[m] = (cum_flow[m] - cum_flow[entry]) signed
    for the trade (CALL: +, PUT: -).  fav < 0 == counter-flow.
  * first OPPOSITE EMA(5) crossover after entry (the entry signal, reversed)
  * counter-flow normalised by the entry trigger's own |abs_flow|
  * lead/lag: minute of first counter-flow  vs  minute the option first goes
    -15% underwater  (does the flow WARN, or just confirm?)

Then simulate three flow-exit overlays vs the baseline bracket, IS/OOS @ 2025-08-21
+ 6 slices, with the winners-vs-losers decomposition (does it save losers more
than it clips winners?):
  FX-x  first opposite EMA crossover >= --grace min after entry
  FX-k  counter-flow <= -k * |abs_flow|   (grid k)
  FX-kd FX-k but only while the option mark is currently down

Usage:
  python check_counterflow.py
  python check_counterflow.py --tickers META IWM --grace 5
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_flow_zscore import annotate_flow_z, _z_matched, _eod_mod_for
from check_config_walkforward import _flow_for, _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
COMM = 0.015
K_GRID = (0.25, 0.5, 0.75, 1.0, 1.5)


def _ema5(x):
    return pd.Series(x, dtype=float).ewm(span=5, adjust=False).mean().to_numpy()


def _first_opp_cross(cum, ema, start_i, want_bear):
    """first index > start_i where cum crosses ema in the direction opposite to
    the entry (want_bear True after a CALL entry, False after a PUT entry)."""
    for i in range(max(start_i + 1, 1), len(cum)):
        up = cum[i - 1] <= ema[i - 1] and cum[i] > ema[i]
        dn = cum[i - 1] >= ema[i - 1] and cum[i] < ema[i]
        if (want_bear and dn) or ((not want_bear) and up):
            return i
    return None


def _bracket(entry_mid, cl, lo, mod, held, tr, rr, tstop, eod, fx_mod=None):
    """_bracket_pnl + an optional extra exit at the first bar with mod >= fx_mod
    (marked at that bar's close). Returns (pnl, exit_reason)."""
    n = len(cl)
    cummax = np.maximum.accumulate(cl)
    cummin = np.minimum.accumulate(lo)
    eod_hit = mod >= eod
    ts_idx = int(np.argmax(eod_hit)) if eod_hit.any() else n - 1
    if tstop:
        th = held >= tstop
        if th.any():
            ts_idx = min(ts_idx, int(np.argmax(th)))
    tp = entry_mid * (1 + tr)
    sl = entry_mid * (1 - tr / rr)
    tp_idx = int(np.searchsorted(cummax, tp)) if cummax[-1] >= tp else n
    sl_idx = int(np.searchsorted(-cummin, -sl)) if cummin[-1] <= sl else n
    fx_idx = n
    if fx_mod is not None:
        w = np.where(mod >= fx_mod)[0]
        if len(w):
            fx_idx = int(w[0])
    exit_idx = min(tp_idx, sl_idx, ts_idx, fx_idx)
    if exit_idx >= n:
        px, why = cl[-1], "eod"
    elif tp_idx == exit_idx and tp_idx <= sl_idx:
        px, why = tp, "tp"
    elif sl_idx == exit_idx and sl_idx < fx_idx:
        px, why = min(sl, cl[sl_idx]), "sl"
    elif fx_idx == exit_idx:
        px, why = cl[fx_idx], "flow"
    else:
        px, why = cl[exit_idx], "tstop"
    return (px - entry_mid) / entry_mid - COMM, why


def _agg(pnls):
    if not pnls:
        return "n=0"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    slc = " ".join(f"S{j+1}{np.mean(b)*100:+.0f}" if len(b) >= 5 else f"S{j+1}··" for j, b in enumerate(sl))
    return (f"n={len(v):>4}  avg {v.mean()*100:>+6.1f}%  IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  win {np.mean(v>0):.2f}  [{slc}]")


def run(a):
    import directional_flow_backtester as D
    from amt_profile import amt_open_map, amt_ok
    from config import RULES
    from check_adx_dmi import _intraday_adx, _intra_asof
    from macro_calendar import is_macro_am_day

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]
    tickers = sorted({r["ticker"] for r in rules})
    flow_all = _flow_for(D, tickers)

    ROWS = []   # per trade: dict
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

        f = flow_all[flow_all["underlying_symbol"] == tk]
        flow_by_day = {}
        for d, g in f.groupby("date"):
            g = g.sort_values("minute_et")
            m = (pd.to_datetime(g["minute_et"]).dt.hour * 60 + pd.to_datetime(g["minute_et"]).dt.minute).to_numpy()
            cum = g["cum_flow"].to_numpy(float)
            flow_by_day[d] = (m, cum, _ema5(cum))

        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {dd: g for dd, g in tb.groupby("date")}

        for r in tk_rules:
            direction = r["direction"].upper()
            want_bull = direction == "CALL"
            dtes = tuple(r.get("dte", [0, 1]))
            tr, rr = float(r["target_roe"]), float(r["rr"])
            tstop = r.get("time_stop_mins"); eod = _eod_mod_for(r)
            trigs = D.triggers_for(flow_all[flow_all.underlying_symbol == tk], tk) \
                if False else D.triggers_for(flow_all, tk)
            zs = r.get("flow_zscore")
            if zs:
                annotate_flow_z(trigs, int(zs.get("window_days") or 60))
                matched = [(t, None) for t in _z_matched(D, r, trigs, gex, vol, trd, amp, reg_src, float(zs["k"]))]
            else:
                D.annotate_flow_pct(trigs, int(r.get("flow_window_days") or 60))
                matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
            ema_stack = ema_stacks.get(int(r["ema_confirm"])) if r.get("ema_confirm") else None
            want_amt = r.get("amt_open")
            dmi = r.get("dmi_confirm")
            dmi_days = iadx.get(int(dmi.get("tf", 15))) if dmi else None

            for t, _thr in matched:
                d, ts = t["date"], t["ts"]
                if r.get("skip_macro_am") and is_macro_am_day(d):
                    continue
                if want_amt and not amt_ok(want_amt, amt.get(d)):
                    continue
                if ema_stack is not None:
                    st = D.ema_state_at(ema_stack, ts)
                    if st is not None and st != ("BULL" if want_bull else "BEAR"):
                        continue
                if dmi:
                    ia = _intra_asof(dmi_days.get(d, []), ts) if dmi_days is not None else None
                    if ia is not None:
                        agree = (ia[0] > ia[1]) == want_bull
                        if (agree if dmi.get("mode") == "agree" else (not agree)) is False:
                            continue
                fd = flow_by_day.get(d)
                if fd is None:
                    continue
                fm, fcum, fema = fd
                emod = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
                ei = int(np.searchsorted(fm, emod, side="right")) - 1
                if ei < 3 or ei >= len(fcum) - 3:
                    continue
                abs_flow = max(abs(t["abs_flow"]), 1.0)
                # post-entry favourable flow, per flow-minute
                fav = (fcum - fcum[ei]) * (1.0 if want_bull else -1.0)
                oc_i = _first_opp_cross(fcum, fema, ei, want_bull)
                oc_mod = int(fm[oc_i]) if oc_i is not None else None

                for p in D._option_paths(t, direction, list(dtes), bbd, bbc):
                    base_pnl, base_why = _bracket(*p, tr, rr, tstop, eod)
                    entry_mid, cl, lo, pmod, held = p
                    # counter-flow features over the hold, sampled at each option bar
                    cf_norm = {}          # horizon(min) -> min(fav)/abs_flow within it
                    for H in (15, 30, 45, 60):
                        sel = (fm > emod) & (fm <= emod + H)
                        cf_norm[H] = float(np.min(fav[sel]) / abs_flow) if sel.any() else 0.0
                    # first minute counter-flow <= -0.5*abs_flow
                    below = np.where((fm > emod) & (fav <= -0.5 * abs_flow))[0]
                    t_cf = int(fm[below[0]]) - emod if len(below) else None
                    # first minute option <= -15%
                    uw = np.where(cl <= entry_mid * 0.85)[0]
                    t_uw = int(pmod[uw[0]]) - emod if len(uw) else None
                    ROWS.append(dict(
                        rule=r["name"], tk=tk, d=d, dir=direction, base=base_pnl, why=base_why,
                        win=base_pnl > 0, abs_flow=abs_flow, oc_mod=oc_mod, emod=emod,
                        cf15=cf_norm[15], cf30=cf_norm[30], cf45=cf_norm[45], cf60=cf_norm[60],
                        t_cf=t_cf, t_uw=t_uw,
                        path=p, tr=tr, rr=rr, tstop=tstop, eod=eod, fav=fav, fm=fm, ei_mod=emod))

    df = pd.DataFrame(ROWS)
    print("=" * 112)
    print(f"  COUNTER-FLOW on losing trades   ({len(df)} trades, {df['rule'].nunique()} rules)   grace={a.grace}m")
    print("=" * 112)

    # ---- 1. winners vs losers: counter-flow magnitude & timing ----
    W, L = df[df.win], df[~df.win]
    print(f"\n  winners n={len(W)}  losers n={len(L)}")
    for col, lab in (("cf15", "counterflow /|flow| in 15m"), ("cf30", "  ... 30m"),
                     ("cf60", "  ... 60m")):
        print(f"    {lab:32}  winners {W[col].mean():+.2f} (med {W[col].median():+.2f})   "
              f"losers {L[col].mean():+.2f} (med {L[col].median():+.2f})")
    lc = L.dropna(subset=["t_cf"]); lu = lc.dropna(subset=["t_uw"])
    print(f"    losers with counterflow<=-0.5|flow|: {len(lc)}/{len(L)} ({100*len(lc)/max(len(L),1):.0f}%)  "
          f"median t_cf {lc['t_cf'].median():.0f}m")
    lead = lu[lu["t_cf"] < lu["t_uw"]]
    print(f"    of losers with BOTH counterflow & -15% underwater ({len(lu)}): "
          f"counterflow LED the drawdown in {len(lead)} ({100*len(lead)/max(len(lu),1):.0f}%), "
          f"median lead {(lu['t_uw']-lu['t_cf']).median():.0f}m")
    # opposite EMA crossover timing
    for nm, sub in (("winners", W), ("losers", L)):
        has = sub[sub["oc_mod"].notna()]
        dtm = (has["oc_mod"] - has["emod"])
        print(f"    {nm}: opposite EMA-cross after entry in {len(has)}/{len(sub)} "
              f"({100*len(has)/max(len(sub),1):.0f}%), median {dtm.median():.0f}m after entry")

    # ---- 2. worst-quartile trades per rule ----
    print("\n  -- worst-quartile baseline trades per rule: did counter-flow fire before -50%? --")
    for rn, g in df.groupby("rule"):
        q = g[g["base"] <= g["base"].quantile(0.25)]
        if len(q) < 8:
            continue
        with_cf = q.dropna(subset=["t_cf"])
        print(f"    {rn:22} worst-Q n={len(q):>3}  avg {q['base'].mean()*100:+.0f}%  "
              f"counterflow fired {len(with_cf)}/{len(q)} ({100*len(with_cf)/len(q):.0f}%)  "
              f"median t_cf {with_cf['t_cf'].median() if len(with_cf) else float('nan'):.0f}m")

    # ---- 3. flow-exit overlays ----
    print("\n" + "=" * 112)
    print("  FLOW-EXIT OVERLAYS vs baseline bracket")
    print("=" * 112)

    def sim(fx_mod_fn, label):
        base, ov = [], []
        dW = dL = 0.0
        nW = nL = 0
        for r in ROWS:
            b = r["base"]
            fxm = fx_mod_fn(r)
            o, _why = _bracket(*r["path"], r["tr"], r["rr"], r["tstop"], r["eod"], fx_mod=fxm)
            base.append((r["d"], b)); ov.append((r["d"], o))
            if b > 0:
                dW += (o - b); nW += 1
            else:
                dL += (o - b); nL += 1
        print(f"\n  {label}")
        print(f"    baseline  {_agg(base)}")
        print(f"    overlay   {_agg(ov)}")
        print(f"    delta on baseline-winners: {100*dW/max(nW,1):+.2f}%/trade (n{nW})   "
              f"baseline-losers: {100*dL/max(nL,1):+.2f}%/trade (n{nL})   "
              f"net {100*(dW+dL)/len(ROWS):+.2f}%/trade")

    sim(lambda r: (r["oc_mod"] if (r["oc_mod"] is not None and r["oc_mod"] >= r["emod"] + a.grace) else None),
        f"FX-x  first opposite EMA crossover >= entry+{a.grace}m")
    for k in K_GRID:
        def fk(r, k=k):
            below = np.where((r["fm"] > r["ei_mod"]) & (r["fav"] <= -k * r["abs_flow"]))[0]
            return int(r["fm"][below[0]]) if len(below) else None
        sim(fk, f"FX-k  counter-flow <= -{k} x |abs_flow|")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--grace", type=int, default=5, help="min after entry before an opposite crossover can exit")
    a = ap.parse_args()
    run(a)
