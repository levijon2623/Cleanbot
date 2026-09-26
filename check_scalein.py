# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_scalein.py
================
The disciplined version of the pile-on (session 18, step #2).

check_pileon established:
  * uncapped scale-in returns +12.2% (IS +9.8 / OOS +13.7) on multi-fill days,
    but with mean stack 9.4 contracts and mean aggregate MAE -41% (worst -99%)
    -- because at target_roe 1.0 / rr 1.0 the stop sits at avg*(1-1.0) = 0, i.e.
    THERE IS NO STOP on the stack at all.
  * delta/gamma-based add gates fail (IS/OOS sign flip); the only IS/OOS-stable
    discriminators are ENVIRONMENTAL and known before the first add:
    first-fill-early, LOWVOL, negative GEX.

So this builds and validates:  cap the adds, gate on environment, and put a
REAL stop on the aggregate position (on average cost, which is what a scale-in
trader actually risks).

  baseline SEQ   one position, rule's own TP/SL          <- what the bot does now
  capped         <=N contracts, stack stop at -S on avg cost
  capped + gate  same, only on days passing the env gate

Reported per config: return ON DEPLOYED CAPITAL (comparable to SEQ's per-trade
%), P&L in 1-LOT UNITS (ret x contracts -- the actual portfolio effect), stack
size, MAE, IS/OOS and the 6 calendar slices.

Gate thresholds are chosen on the IS half ONLY, then read out OOS.

Usage:
  python check_scalein.py
  python check_scalein.py --tickers META NVDA IWM
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
MAX_ADDS = (1, 2, 3, 4)
STACK_STOP = (0.35, 0.50, 0.65)


def _eod_mod(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _scalein(bars, fills, tr, rr, eod_m, max_adds, stack_stop):
    """<=max_adds contracts on one contract; TP on running avg cost; a REAL
    stack stop at -stack_stop of avg cost; EOD flatten. Returns return on
    deployed capital + the risk it took to get there."""
    b = bars.sort_values("minute_et")
    mod = (b["minute_et"].dt.hour * 60 + b["minute_et"].dt.minute).to_numpy()
    cl = b["close"].to_numpy(float)
    lo = b["low"].to_numpy(float)
    bid = b["bid_close"].to_numpy(float)
    ask = b["ask_close"].to_numpy(float)
    mid = np.where(bid > 0, (bid + ask) / 2.0, cl)

    fills = sorted(fills)
    n_ct, cost, mae, fi = 0, 0.0, 0.0, 0
    for i in range(len(mod)):
        m = mod[i]
        while fi < len(fills) and fills[fi] <= m:
            if n_ct < max_adds and np.isfinite(mid[i]) and mid[i] > 0:
                n_ct += 1
                cost += mid[i]
            fi += 1
        if n_ct == 0:
            continue
        avg = cost / n_ct
        mae = min(mae, (lo[i] - avg) / avg)
        if cl[i] >= avg * (1 + tr):
            return dict(ret=tr, n=n_ct, mae=mae, why="TP")
        if lo[i] <= avg * (1 - stack_stop):
            return dict(ret=-stack_stop, n=n_ct, mae=mae, why="STOP")
        if m >= eod_m:
            return dict(ret=(cl[i] - avg) / avg, n=n_ct, mae=mae, why="EOD")
    if n_ct == 0:
        return None
    avg = cost / n_ct
    return dict(ret=(cl[-1] - avg) / avg, n=n_ct, mae=mae, why="END")


def _line(lbl, recs, base_n=None):
    """recs: [(date, ret_on_capital, n_contracts, mae)]"""
    if len(recs) < 5:
        return f"    {lbl:26} n={len(recs):>3}  (thin)"
    v = np.array([r for _, r, _, _ in recs])
    n = np.array([c for _, _, c, _ in recs], float)
    ma = np.array([m for _, _, _, m in recs], float)
    i = np.array([r for d, r, _, _ in recs if d < SPLIT])
    o = np.array([r for d, r, _, _ in recs if d >= SPLIT])
    lots = v * n                      # P&L in 1-lot units
    li = np.array([r * c for d, r, c, _ in recs if d < SPLIT])
    lo_ = np.array([r * c for d, r, c, _ in recs if d >= SPLIT])
    sl = [[] for _ in range(6)]
    for d, r, c, _ in recs:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(r)
    npop = sum(1 for b in sl if len(b) >= 4)
    ret = f" ({100*len(v)/base_n:>3.0f}% of days)" if base_n else ""
    return (f"    {lbl:26} n={len(v):>3}{ret}  cap-ret {v.mean()*100:>+6.1f}% "
            f"(IS {i.mean()*100 if len(i) else float('nan'):>+6.1f} / "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+6.1f})  "
            f"lots {lots.mean():>+5.2f} (IS {li.mean() if len(li) else float('nan'):>+5.2f} / "
            f"OOS {lo_.mean() if len(lo_) else float('nan'):>+5.2f})  "
            f"ct {n.mean():>4.1f}  MAE {ma.mean()*100:>+5.0f}%  win {(v>0).mean():.2f}  pop {npop}/6")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok
    from sequential_fills import bracket_with_exit

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]

    days = []      # one record per (rule, day)
    for r in rules:
        tk = r["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, r.get("flow_window_days", 60))
        try:
            tb = D._ticker_bars(tk)
        except Exception:
            tb = None
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        if tb is None or tb.empty:
            continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        amt = amt_open_map(tk) if r.get("amt_open") else {}

        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
        if r.get("amt_open"):
            matched = [(t, th) for t, th in matched if amt_ok(r["amt_open"], amt.get(t["date"]))]
        matched.sort(key=lambda x: pd.Timestamp(x[0]["ts"]))
        tr_, rr_, em = float(r["target_roe"]), float(r["rr"]), _eod_mod(r)
        dtes = r.get("dte", [0, 1])

        byday = collections.defaultdict(list)
        for t, _th in matched:
            d, ts = t["date"], t["ts"]
            day = bbd.get(d)
            if day is None:
                continue
            at = day[day["minute_et"] <= ts]
            if at.empty:
                continue
            spot = float(at.iloc[-1]["underlying_close"])
            m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
            paths = D._option_paths(t, r["direction"], dtes, bbd, bbc)
            if not paths:
                continue
            cid = None
            for dd in dtes:
                cid = D.pick_contract(day, ts, r["direction"], dd, spot)
                if cid is not None:
                    break
            byday[d].append((m, cid, paths[0]))

        for d, fills in byday.items():
            fills.sort(key=lambda x: x[0])
            # SEQ baseline: first fill, rule's own bracket
            pnl, _xm = bracket_with_exit(*fills[0][2], tr_, rr_, r.get("time_stop_mins"), em)
            cids = collections.Counter(c for _, c, _ in fills if c)
            if not cids:
                continue
            cid = cids.most_common(1)[0][0]
            cf = [m for m, c, _ in fills if c == cid]
            bars = bbc.get(cid)
            if bars is None:
                continue
            bars = bars[bars["date"] == d]
            if bars.empty:
                continue
            rec = dict(rule=r["name"], date=d, seq=pnl, nfill=len(fills),
                       first_mod=fills[0][0], vol=vol.get(d), gex=gex.get(d),
                       trend=trd.get(d), bars=bars, cf=cf, tr=tr_, rr=rr_, em=em)
            for N in MAX_ADDS:
                for S in STACK_STOP:
                    res = _scalein(bars, cf, tr_, rr_, em, N, S)
                    if res:
                        rec[f"r_{N}_{S}"] = res["ret"]
                        rec[f"n_{N}_{S}"] = res["n"]
                        rec[f"m_{N}_{S}"] = res["mae"]
            days.append(rec)

    R = pd.DataFrame(days)
    if R.empty:
        print("no data"); return
    print("=" * 122)
    print(f"  CAPPED SCALE-IN   {len(R)} rule-days  ({(R.nfill >= 2).sum()} multi-fill)   split {SPLIT}")
    print("=" * 122)

    seq = [(x["date"], x["seq"], 1.0, np.nan) for _, x in R.iterrows()]
    print(_line("SEQUENTIAL (bot today)", seq))

    print("\n  -- cap x stack-stop grid (all days) --")
    for N in MAX_ADDS:
        for S in STACK_STOP:
            col = f"r_{N}_{S}"
            if col not in R:
                continue
            sub = R.dropna(subset=[col])
            recs = [(x["date"], x[col], x[f"n_{N}_{S}"], x[f"m_{N}_{S}"]) for _, x in sub.iterrows()]
            print(_line(f"max {N} ct, stop -{int(S*100)}%", recs))

    # ---- environment gate, thresholds picked on IS ONLY ----
    print("\n" + "=" * 122)
    print("  ENVIRONMENT GATE (chosen on IS half only, then read out OOS)")
    print("=" * 122)
    IS = R[R.date < SPLIT]
    best = None
    for cut in (660, 690, 720, 750):                     # first fill before 11:00 / 11:30 / 12:00 / 12:30
        for envs in (("LOWVOL",), ("LOWVOL", "NORMVOL"), None):
            for negx in (True, False):
                m = IS["first_mod"] <= cut
                if envs:
                    m &= IS["vol"].isin(envs)
                if negx:
                    m &= IS["gex"] == "NEGATIVE"
                if m.sum() < 12:
                    continue
                for N in (2, 3):
                    for S in (0.35, 0.50):
                        col = f"r_{N}_{S}"
                        v = IS.loc[m, col].dropna()
                        if len(v) < 12:
                            continue
                        score = v.mean()
                        if best is None or score > best[0]:
                            best = (score, cut, envs, negx, N, S)
    if best is None:
        print("  no gate had enough IS support"); return
    _, cut, envs, negx, N, S = best
    gname = (f"first fill <= {cut//60:02d}:{cut%60:02d}"
             + (f", vol in {'/'.join(envs)}" if envs else "")
             + (", GEX NEGATIVE" if negx else ""))
    print(f"  IS-selected gate: {gname}   |   max {N} contracts, stack stop -{int(S*100)}%")

    mask = R["first_mod"] <= cut
    if envs:
        mask &= R["vol"].isin(envs)
    if negx:
        mask &= R["gex"] == "NEGATIVE"
    G = R[mask].dropna(subset=[f"r_{N}_{S}"])
    ng = R[~mask]
    base_n = len(R)
    print()
    print(_line("SEQ on gated days", [(x["date"], x["seq"], 1.0, np.nan) for _, x in G.iterrows()], base_n))
    print(_line(f"SCALE-IN on gated days", [(x["date"], x[f"r_{N}_{S}"], x[f"n_{N}_{S}"], x[f"m_{N}_{S}"])
                                            for _, x in G.iterrows()], base_n))
    print(_line("SEQ on NON-gated days", [(x["date"], x["seq"], 1.0, np.nan) for _, x in ng.iterrows()], base_n))
    print("\n  -> BOOK if we scale in only on gated days and stay 1-lot elsewhere:")
    blend = ([(x["date"], x[f"r_{N}_{S}"], x[f"n_{N}_{S}"], x[f"m_{N}_{S}"]) for _, x in G.iterrows()]
             + [(x["date"], x["seq"], 1.0, np.nan) for _, x in ng.iterrows()])
    print(_line("BLENDED", blend))
    print(_line("BLENDED all-SEQ (ref)", seq))

    # ---- the sizing reading -------------------------------------------------
    # bot_runner already sizes each position to a PREMIUM BUDGET (flat-premium
    # parity, SIZING_TARGET_PREMIUM_PCT), so "1 lot" is already N contracts.
    # That makes two things explicit:
    #   EQUAL-BUDGET scale-in = split the SAME budget across the adds -> total
    #     risk unchanged, so the only thing that matters is cap-ret (above).
    #   FULL-SIZE-per-add = k x the premium budget on one ticker-day, which
    #     breaks MAX_PORTFOLIO_RISK_PCT -- not a real option.
    # So the honest question is not "scale in?" but "is the gate a SIZE signal?"
    print("\n" + "=" * 122)
    print("  THE GATE AS A SIZE MULTIPLIER (bot already sizes to a premium budget)")
    print("=" * 122)
    gi = G[G.date < SPLIT]; go = G[G.date >= SPLIT]
    ni = ng[ng.date < SPLIT]; no = ng[ng.date >= SPLIT]
    print(f"    gated days      n={len(G):>3} (IS {len(gi)} / OOS {len(go)})   "
          f"SEQ cap-ret IS {gi['seq'].mean()*100:+.1f}% / OOS {go['seq'].mean()*100:+.1f}%")
    print(f"    non-gated days  n={len(ng):>3} (IS {len(ni)} / OOS {len(no)})   "
          f"SEQ cap-ret IS {ni['seq'].mean()*100:+.1f}% / OOS {no['seq'].mean()*100:+.1f}%")
    for k in (1.0, 1.5, 2.0):
        for half, gg, nn in (("IS", gi, ni), ("OOS", go, no)):
            if len(gg) == 0 or len(nn) == 0:
                continue
            lots = (len(gg) * k * gg["seq"].mean() + len(nn) * nn["seq"].mean()) / (len(gg) + len(nn))
            cap = (len(gg) * k + len(nn)) / (len(gg) + len(nn))
            print(f"    size x{k:<4} on gated  [{half:3}]  lots/day {lots:>+6.3f}  "
                  f"capital {cap:.2f}x  -> P&L per unit capital {lots/cap:>+6.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    a = ap.parse_args()
    run(a)
