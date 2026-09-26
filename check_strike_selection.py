# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_strike_selection.py
=========================
The bot always buys the ATM strike (nearest to spot).  Test whether picking a
strike or two OTM pays off -- overall, and conditioned on the vol regime
(prior-day VIX vs its trailing median; the ticker's LOWVOL/NORMVOL/HIVOL label;
20d realised vol).  Hypothesis: on volatile days the bigger move + the OTM
strike's extra convexity beats ATM; on quiet days ATM's higher delta wins.

For every deployed-spec trigger, walk the option-bracket P&L (same TP tr / SL
tr/rr, eod, tstop -- % ROE so it's strike-agnostic in the bracket, but the
strikes need different-sized underlying moves to hit it) for:
  ATM            nearest strike to spot
  OTM+1 / OTM+2  N strikes further OTM (call: higher, put: lower)
  OTM 0.5% / 1%  nearest strike at least that far OTM (moneyness, not strike count)
keeping the live $0.50 entry-premium floor (report the reject rate per config).

Then:
  * blended + per-rule P&L, IS/OOS @ 2025-08-21 + 6 slices + win + avg entry $
  * split each config by VIX-favourable / VIX-low, and by vol regime
  * an ADAPTIVE picker (OTM+1 when VIX favourable or HIVOL, ATM otherwise) vs
    always-ATM

Usage:
  python check_strike_selection.py
  python check_strike_selection.py --tickers IWM QQQ SPY META
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from check_flow_zscore import annotate_flow_z, _z_matched, _eod_mod_for
from check_config_walkforward import _flow_for, _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
COMM = 0.015


def _pick_cid(day_bars, ts, direction, target_dte, spot, offset=0, mny=None):
    at = day_bars[(day_bars["minute_et"] <= ts) & (day_bars["minute_et"] >= ts - pd.Timedelta(minutes=3))]
    at = at[at["option_type"] == ("call" if direction == "CALL" else "put")]
    if at.empty:
        return None, None
    ed = pd.Timestamp(ts).date()
    at = at.assign(dte=(at["expiry"] - ed).map(lambda x: x.days))
    at = at[at["dte"] == target_dte]
    if at.empty:
        return None, None
    at = at.sort_values("minute_et").groupby("option_chain_id").last().reset_index()
    ks = sorted(at["strike"].unique())
    if not ks:
        return None, None
    atm_i = int(np.argmin([abs(k - spot) for k in ks]))
    call = direction == "CALL"
    if mny is not None:
        cands = [k for k in ks if (k >= spot * (1 + mny))] if call else [k for k in ks if (k <= spot * (1 - mny))]
        if not cands:
            return None, None
        strike = cands[0] if call else cands[-1]
    else:
        j = min(atm_i + offset, len(ks) - 1) if call else max(atm_i - offset, 0)
        strike = ks[j]
    row = at[at["strike"] == strike].iloc[0]
    return row["option_chain_id"], (strike / spot - 1.0) * (1 if call else -1)


def _one_path(t, direction, dte, bbd, bbc, offset=0, mny=None, floor=0.50):
    """P&L-ready path for ONE (trigger, dte, strike-config). Returns
    (entry_bid, entry_ask, entry_mid, cl, lo, bidp, mod, held, mny)
    | ('REJECT', em, mny) | None."""
    day = bbd.get(t["date"])
    if day is None:
        return None
    at = day[day["minute_et"] <= t["ts"]]
    if at.empty:
        return None
    spot = float(at.iloc[-1]["underlying_close"])
    cid, m = _pick_cid(day, t["ts"], direction, dte, spot, offset, mny)
    if cid is None:
        return None
    ent = bbc.get(cid)
    if ent is None:
        return None
    er = ent[(ent["minute_et"] <= t["ts"]) & (ent["minute_et"] >= t["ts"] - pd.Timedelta(minutes=3))]
    if er.empty:
        return None
    er = er.iloc[-1]
    b, aa = float(er["bid_close"]), float(er["ask_close"])
    entry_mid = (b + aa) / 2.0 if b > 0 else float(er["close"])
    if not np.isfinite(entry_mid) or entry_mid <= 0:
        return None
    if entry_mid < floor:
        return ("REJECT", entry_mid, m)
    entry_ask = aa if aa > 0 else entry_mid
    entry_bid = b if b > 0 else entry_mid
    fwd = ent[ent["minute_et"] > t["ts"]].sort_values("minute_et")
    if len(fwd) < 3:
        return None
    cl = fwd["close"].to_numpy(float)
    lo = fwd["low"].to_numpy(float)
    bidp = fwd["bid_close"].to_numpy(float)
    bidp = np.where(np.isfinite(bidp) & (bidp > 0), bidp, cl)   # fall back to close
    if not (np.isfinite(cl).all() and np.isfinite(lo).all()):
        return None
    pm = fwd["minute_et"]
    return (entry_bid, entry_ask, entry_mid, cl, lo, bidp,
            (pm.dt.hour.values * 60 + pm.dt.minute.values).astype(int),
            ((pm - pd.Timestamp(t["ts"])).dt.total_seconds().values / 60.0), m)


def _bracket(pth, tr, rr, tstop, eod, fill="mid"):
    """fill='mid'  -> optimistic: enter at mid, TP fills at the target limit.
    fill='real'   -> enter at the ASK, TP/SL/EOD all fill at the BID that minute
                     (the cheap-OTM spread bites here)."""
    e_bid, e_ask, e_mid, cl, lo, bidp, mod, held, _m = pth
    n = len(cl)
    entry = e_ask if fill == "real" else e_mid
    exitser = bidp if fill == "real" else cl
    cummax = np.maximum.accumulate(cl); cummin = np.minimum.accumulate(lo)
    eod_hit = mod >= eod
    ts_idx = int(np.argmax(eod_hit)) if eod_hit.any() else n - 1
    if tstop:
        th = held >= tstop
        if th.any():
            ts_idx = min(ts_idx, int(np.argmax(th)))
    tp = entry * (1 + tr); sl = entry * (1 - tr / rr)
    tp_i = int(np.searchsorted(cummax, tp)) if cummax[-1] >= tp else n
    sl_i = int(np.searchsorted(-cummin, -sl)) if cummin[-1] <= sl else n
    ei = min(tp_i, sl_i, ts_idx)
    if ei >= n:
        px = exitser[-1]
    elif tp_i <= sl_i and tp_i == ei:
        px = tp if fill == "mid" else min(tp, exitser[ei])
    elif sl_i == ei:
        px = min(sl, exitser[ei])
    else:
        px = exitser[ei]
    return (px - entry) / entry - COMM


def _agg(pnls, base_n=None):
    if len(pnls) < 10:
        return f"n={len(pnls):>4} thin"
    v = np.array([p for _, p, _ in pnls])
    i = [p for d, p, _ in pnls if d < SPLIT]
    o = [p for d, p, _ in pnls if d >= SPLIT]
    sl = [[] for _ in range(6)]
    for d, p, _ in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    sd = np.std([np.mean(b) for b in sl if len(b) >= 5]) * 100
    slc = " ".join(f"S{j+1}{np.mean(b)*100:+.0f}" if len(b) >= 5 else f"S{j+1}··" for j, b in enumerate(sl))
    em = np.mean([e for _, _, e in pnls])
    ret = f" ({100*len(v)/base_n:.0f}%)" if base_n else ""
    return (f"n={len(v):>4}{ret}  avg {v.mean()*100:>+6.1f}%  IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  win {np.mean(v>0):.2f}  sliceSD {sd:4.1f}  mny {em*100:+.2f}%")


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

    # VIX favourable map (prior-day vs trailing-60 median)
    vixfav = {}
    try:
        import requests
        from dotenv import load_dotenv
        load_dotenv()
        h = {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}", "Accept": "application/json"}
        rr = requests.get("https://api.unusualwhales.com/api/stock/VIX/volatility/realized",
                          headers=h, params={"timeframe": "2Y"}, timeout=20).json().get("data", [])
        s = pd.Series({pd.Timestamp(x["date"]).date(): float(x["price"]) for x in rr if x.get("price")}).sort_index()
        med = s.shift(1).rolling(60, min_periods=20).median()
        vp = s.shift(1)
        vixfav = {d: (bool(vp[d] >= med[d]) if pd.notna(med[d]) else None) for d in s.index}
    except Exception as e:
        print(f"  (VIX unavailable: {e})")

    CFG = [("ATM", dict(offset=0)), ("OTM+1", dict(offset=1)), ("OTM+2", dict(offset=2))]
    ROWS = []   # long-form: one dict per (trigger, dte, cfg) with a VALID path (NO floor)

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
            direction = r["direction"].upper(); want_bull = direction == "CALL"
            dtes = tuple(r.get("dte", [0, 1]))
            tr, rr = float(r["target_roe"]), float(r["rr"])
            tstop = r.get("time_stop_mins"); eod = _eod_mod_for(r)
            trigs = D.triggers_for(flow_all, tk)
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
                vf, vr = vixfav.get(d), vol.get(d)
                for dte in dtes:
                    for cname, kw in CFG:
                        pth = _one_path(t, direction, dte, bbd, bbc, floor=0.0, **kw)
                        if pth is None or (isinstance(pth[0], str)):
                            continue
                        ROWS.append({
                            "rule": r["name"], "date": d, "vf": vf, "vr": vr, "cfg": cname,
                            "mny": pth[8], "entry_mid": pth[2],
                            "pnl_mid": _bracket(pth, tr, rr, tstop, eod, "mid"),
                            "pnl_real": _bracket(pth, tr, rr, tstop, eod, "real"),
                        })

    df = pd.DataFrame(ROWS)
    names = [c for c, _ in CFG]
    FLOORS = [0.0, 0.15, 0.25, 0.35, 0.50]

    def agg(s, col):
        if len(s) < 8:
            return f"n{len(s):>4} thin"
        v = s[col].to_numpy()
        o = s[s.date >= SPLIT][col]; i = s[s.date < SPLIT][col]
        return (f"n{len(s):>4}  avg {v.mean()*100:>+6.1f}  IS {i.mean()*100 if len(i) else float('nan'):>+6.1f}  "
                f"OOS {o.mean()*100 if len(o) else float('nan'):>+6.1f}  win {(v>0).mean():.2f}  "
                f"mny {s['mny'].mean()*100:+.2f}%")

    print("=" * 118)
    print(f"  STRIKE x ENTRY-FLOOR x FILL   ({len(df)} paths)   split {SPLIT}")
    print("  fill 'mid' = enter mid, TP at limit (optimistic).  'real' = enter ASK, exit BID (cheap-OTM spread bites)")
    print("=" * 118)

    for rn, g in df.groupby("rule"):
        if len(g[g.cfg == "ATM"]) < 20:
            continue
        print(f"\n  {rn}")
        for cfg in names:
            gc = g[g.cfg == cfg]
            if len(gc) < 15:
                continue
            for fl in FLOORS:
                s = gc[gc.entry_mid >= fl]
                keep = 100 * len(s) / max(len(g[g.cfg == "ATM"]), 1)
                print(f"    {cfg:6} floor {fl:.2f} ({keep:3.0f}% of ATM-n)  "
                      f"mid[{agg(s, 'pnl_mid')}]   real[{agg(s, 'pnl_real')}]")

    print("\n" + "=" * 118)
    print("  BLENDED BOOK: always-ATM vs GLD-only-OTM+1, at floor 0.00 and 0.50, both fills")
    print("=" * 118)
    for fl in (0.0, 0.50):
        for fill in ("pnl_mid", "pnl_real"):
            atm = df[(df.cfg == "ATM") & (df.entry_mid >= fl)]
            # GLD switched to OTM+1, rest ATM
            gld_otm = df[(df.rule == "GLD amp1 CALL") & (df.cfg == "OTM+1") & (df.entry_mid >= fl)]
            rest = df[(df.rule != "GLD amp1 CALL") & (df.cfg == "ATM") & (df.entry_mid >= fl)]
            mix = pd.concat([gld_otm, rest])
            print(f"  floor {fl:.2f}  {fill[4:]:4}   always-ATM {agg(atm, fill)}   |   GLD->OTM+1 {agg(mix, fill)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    a = ap.parse_args()
    run(a)
