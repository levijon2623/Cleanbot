# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_pileon.py
===============
What the backtest's "23 trades in a day" REALLY is, and whether it is a strategy.

check_day_anatomy showed the multi-trigger days are one contract, opened many
times, all exiting together. The bot can't do that (bot_runner.py:1288). But
re-entering the same decaying contract at lower/similar prices IS a real
strategy -- AVERAGING DOWN / scaling in -- so this measures it honestly:

  A) BACKTEST      mean of the individual per-trigger returns   <- what the
                   screen reports. Overstates by Jensen: mean(X/e_i) >= X/mean(e).
  B) SEQUENTIAL    first trigger only, one position (what the bot does today)
  C) AVERAGE-DOWN  the real thing: each trigger ADDS a contract to one position,
                   avg cost re-computed on every add, TP/SL measured against the
                   RUNNING AVG COST, whole stack exits together (TP / SL / EOD).
                   Return is on total capital deployed, not per-contract.

and the risk C actually carries: max contracts held, peak capital vs a 1-lot,
and the worst aggregate mark-to-market drawdown on avg cost.

Then it hunts for a metric KNOWN AT THE SECOND FILL that separates the days
where averaging down worked from the days it didn't -- i.e. is there a rule in
here, or is it just "days that went up".

Usage:
  python check_pileon.py                          # all enabled rules
  python check_pileon.py --tickers WMT --regime LOWVOL --direction CALL --pct 65 --dow thu
  python check_pileon.py --detail 3                # show the N busiest days trade-by-trade
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
_DOW = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}


def _eod_mod(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _avg_down(bars, fills, tr, rr, eod_m):
    """Scale-in simulation on ONE contract.

    bars  : that contract's day frame (minute_et, close, low, bid_close, ask_close)
    fills : sorted list of entry minutes (each adds 1 contract)
    TP/SL are checked against the RUNNING average cost after each add, exactly as
    a scale-in trader would. Returns dict or None."""
    b = bars.sort_values("minute_et")
    mod = (b["minute_et"].dt.hour * 60 + b["minute_et"].dt.minute).to_numpy()
    cl = b["close"].to_numpy(float)
    lo = b["low"].to_numpy(float)
    bid = b["bid_close"].to_numpy(float)
    ask = b["ask_close"].to_numpy(float)
    mid = np.where(bid > 0, (bid + ask) / 2.0, cl)

    fills = sorted(fills)
    n_ct, cost = 0, 0.0
    adds, mae = [], 0.0
    fi = 0
    for i in range(len(mod)):
        m = mod[i]
        while fi < len(fills) and fills[fi] <= m:      # add a contract
            px = mid[i]
            if np.isfinite(px) and px > 0:
                n_ct += 1
                cost += px
                adds.append(px)
            fi += 1
        if n_ct == 0:
            continue
        avg = cost / n_ct
        # mark-to-market drawdown on the whole stack
        mae = min(mae, (lo[i] - avg) / avg)
        if cl[i] >= avg * (1 + tr):
            return dict(ret=tr, why="TP", n=n_ct, avg=avg, adds=adds, mae=mae, xmod=int(m))
        if lo[i] <= avg * (1 - tr / rr):
            return dict(ret=-tr / rr, why="SL", n=n_ct, avg=avg, adds=adds, mae=mae, xmod=int(m))
        if m >= eod_m:
            return dict(ret=(cl[i] - avg) / avg, why="EOD", n=n_ct, avg=avg, adds=adds,
                        mae=mae, xmod=int(m))
    if n_ct == 0:
        return None
    avg = cost / n_ct
    return dict(ret=(cl[-1] - avg) / avg, why="END", n=n_ct, avg=avg, adds=adds,
                mae=mae, xmod=int(mod[-1]))


def _avg_down_delta(bars, fills, tr, rr, eod_m, garr, sgn, stop_on_fall=False):
    """Scale-in, but each ADD after the first is only taken if the position has
    grown MORE FAVOURABLE since the last add -- signed delta (sgn*delta, so + =
    moving ITM for both calls and puts) must be higher than at the previous add.

    This is the CAUSAL version of the delta idea: at add k you only need
    delta_1..delta_k, all of which are known. `stop_on_fall` additionally flattens
    the whole stack the first time delta drops back below the last add's delta
    (a trailing exit on directional state rather than on price)."""
    b = bars.sort_values("minute_et")
    mod = (b["minute_et"].dt.hour * 60 + b["minute_et"].dt.minute).to_numpy()
    cl = b["close"].to_numpy(float)
    lo = b["low"].to_numpy(float)
    bid = b["bid_close"].to_numpy(float)
    ask = b["ask_close"].to_numpy(float)
    mid = np.where(bid > 0, (bid + ask) / 2.0, cl)

    fills = sorted(fills)
    n_ct, cost, mae = 0, 0.0, 0.0
    adds, last_d = [], None
    fi = 0
    for i in range(len(mod)):
        m = mod[i]
        dnow = _asof(garr, m) if garr is not None else None
        dnow = (sgn * dnow[0]) if (dnow is not None and np.isfinite(dnow[0])) else None
        while fi < len(fills) and fills[fi] <= m:
            px = mid[i]
            take = np.isfinite(px) and px > 0
            if take and n_ct > 0:
                # only scale in while the trade is getting better
                take = (dnow is not None and last_d is not None and dnow > last_d)
            if take:
                n_ct += 1
                cost += px
                adds.append(px)
                if dnow is not None:
                    last_d = dnow
            fi += 1
        if n_ct == 0:
            continue
        if last_d is None and dnow is not None:
            last_d = dnow
        avg = cost / n_ct
        mae = min(mae, (lo[i] - avg) / avg)
        if cl[i] >= avg * (1 + tr):
            return dict(ret=tr, why="TP", n=n_ct, adds=adds, mae=mae)
        if lo[i] <= avg * (1 - tr / rr):
            return dict(ret=-tr / rr, why="SL", n=n_ct, adds=adds, mae=mae)
        if stop_on_fall and n_ct > 1 and dnow is not None and last_d is not None and dnow < last_d:
            return dict(ret=(cl[i] - avg) / avg, why="DFALL", n=n_ct, adds=adds, mae=mae)
        if m >= eod_m:
            return dict(ret=(cl[i] - avg) / avg, why="EOD", n=n_ct, adds=adds, mae=mae)
    if n_ct == 0:
        return None
    avg = cost / n_ct
    return dict(ret=(cl[-1] - avg) / avg, why="END", n=n_ct, adds=adds, mae=mae)


def _greeks_at(needs):
    """{(date, cid): {mod: (delta, gamma, theta)}} pulled from the silver lake.

    The cached bar files (_screen_cache / opt_bars_atm) drop the greeks, but
    silver carries delta_close / gamma_close / theta_close per contract-minute.
    Only the partitions for the days we actually need are scanned."""
    import glob
    import os
    import polars as pl
    out = {}
    for d in sorted({dd for dd, _ in needs}):
        p = f"lake/silver/option-contracts-1m/date={d}/bars.parquet"
        if not os.path.exists(p):
            continue
        want = [c for dd, c in needs if dd == d]
        try:
            df = (pl.scan_parquet(p)
                  .filter(pl.col("option_chain_id").is_in(want))
                  .select(["option_chain_id", "minute_et", "delta_close",
                           "gamma_close", "theta_close"])
                  .collect().to_pandas())
        except Exception:
            continue
        if df.empty:
            continue
        et = pd.to_datetime(df["minute_et"])
        if getattr(et.dt, "tz", None) is not None:
            et = et.dt.tz_convert("America/New_York").dt.tz_localize(None)
        df["mod"] = et.dt.hour * 60 + et.dt.minute
        for c, g in df.groupby("option_chain_id"):
            g = g.sort_values("mod")
            out[(d, c)] = (g["mod"].to_numpy(),
                           g[["delta_close", "gamma_close", "theta_close"]].to_numpy(float))
    return out


def _asof(arr, m):
    mm, vv = arr
    i = int(np.searchsorted(mm, m, side="right")) - 1
    return vv[i] if i >= 0 else None


def _agg(label, vals):
    if not vals:
        return f"    {label:16} (none)"
    v = np.array([p for _, p in vals])
    i = [p for d, p in vals if d < SPLIT]
    o = [p for d, p in vals if d >= SPLIT]
    return (f"    {label:16} n={len(v):>4}  all {v.mean()*100:>+6.1f}%  "
            f"IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  win {np.mean(v>0):.2f}")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok
    from sequential_fills import bracket_with_exit

    if a.tickers and a.direction:
        rules = [dict(name=f"{a.tickers[0]} {a.regime or 'ALL'} {a.direction}",
                      ticker=a.tickers[0].upper(), direction=a.direction.upper(),
                      hours=[9, 10, 11, 12, 13, 14], dte=a.dte or [0, 1],
                      regime=a.regime, min_flow_pct=a.pct, target_roe=1.0, rr=1.0)]
        rules[0] = {k: v for k, v in rules[0].items() if v is not None}
    else:
        rules = [r for r in RULES if r.get("enabled", True)]
        if a.tickers:
            keep = {t.upper() for t in a.tickers}
            rules = [r for r in rules if r["ticker"].upper() in keep]
    dow_keep = {_DOW[d.lower()[:3]] for d in a.dow} if a.dow else None

    BT, SEQ, AVG = [], [], []
    rows = []            # per (rule, day) record for the discriminator hunt
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
        if dow_keep is not None:
            matched = [(t, th) for t, th in matched if t["date"].weekday() in dow_keep]
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
            byday[d].append((m, cid, paths[0], t))

        for d, fills in byday.items():
            fills.sort(key=lambda x: x[0])
            # A) backtest: every trigger scored independently
            bt = [D._bracket_pnl(*p, tr_, rr_, r.get("time_stop_mins"), em) for _, _, p, _ in fills]
            BT += [(d, x) for x in bt]
            # B) sequential: first fill, real exit, re-entry only after
            seq, busy = [], -1
            for m, cid, p, t in fills:
                if m < busy:
                    continue
                pnl, xm = bracket_with_exit(*p, tr_, rr_, r.get("time_stop_mins"), em)
                seq.append(pnl); busy = xm
            SEQ += [(d, x) for x in seq]
            # C) average-down on the dominant contract of the day
            cids = collections.Counter(c for _, c, _, _ in fills if c)
            if not cids:
                continue
            cid = cids.most_common(1)[0][0]
            cfills = [m for m, c, _, _ in fills if c == cid]
            bars = bbc.get(cid)
            if bars is None:
                continue
            bars = bars[bars["date"] == d]
            if bars.empty:
                continue
            res = _avg_down(bars, cfills, tr_, rr_, em)
            if res is None:
                continue
            AVG.append((d, res["ret"]))
            rows.append(dict(rule=r["name"], date=d, ticker=tk, direction=r["direction"],
                             cid=cid, cfills=cfills,
                             nfill=len(fills), ncid=len(cids),
                             stack=res["n"], ret_bt=float(np.mean(bt)), ret_seq=float(np.mean(seq)) if seq else np.nan,
                             ret_avg=res["ret"], why=res["why"], mae=res["mae"],
                             first_mod=fills[0][0], last_mod=fills[-1][0],
                             adds=res["adds"], trend=trd.get(d), vol=vol.get(d), gex=gex.get(d)))

    print("=" * 106)
    print(f"  PILE-ON ANATOMY   {len(rows)} rule-days")
    print("=" * 106)
    print(_agg("A backtest", BT))
    print(_agg("B sequential", SEQ))
    print(_agg("C average-down", AVG))

    R = pd.DataFrame(rows)
    if R.empty:
        return
    multi = R[R["nfill"] >= 2]
    print(f"\n  multi-fill days: {len(multi)} / {len(R)}   "
          f"(single-fill days are identical in all three by construction)")
    if multi.empty:
        return

    # do the adds actually AVERAGE DOWN?
    slopes, first_last = [], []
    for _, x in multi.iterrows():
        ad = np.array(x["adds"], float)
        if len(ad) >= 2:
            slopes.append(np.polyfit(np.arange(len(ad)), ad / ad[0], 1)[0])
            first_last.append(ad[-1] / ad[0] - 1)
    print(f"  add-price drift: mean slope {np.mean(slopes)*100:+.2f}%/add   "
          f"last-vs-first {np.mean(first_last)*100:+.1f}%   "
          f"({np.mean([x < 0 for x in first_last])*100:.0f}% of days genuinely averaged DOWN)")

    print(f"\n  {'':22} {'backtest':>9} {'sequential':>11} {'avg-down':>9}")
    for lbl, sub in (("multi-fill days", multi),
                     ("  IS", multi[multi.date < SPLIT]), ("  OOS", multi[multi.date >= SPLIT])):
        if len(sub) < 3:
            continue
        print(f"  {lbl:22} {sub.ret_bt.mean()*100:>+8.1f}% {sub.ret_seq.mean()*100:>+10.1f}% "
              f"{sub.ret_avg.mean()*100:>+8.1f}%")
    print(f"\n  avg-down risk: mean stack {multi["stack"].mean():.1f} contracts "
          f"(max {int(multi['stack'].max())})  ->  peak capital {multi['stack'].mean():.1f}x a 1-lot")
    print(f"  aggregate MAE on avg cost: mean {multi['mae'].mean()*100:+.0f}%  "
          f"p10 {multi['mae'].quantile(.10)*100:+.0f}%  worst {multi['mae'].min()*100:+.0f}%")
    print(f"  exit mix: " + "  ".join(f"{k} {v}" for k, v in multi["why"].value_counts().items()))

    # ---- discriminator hunt: what is knowable at the 2nd fill? ----
    print("\n" + "=" * 106)
    print("  IS THERE A RULE?  avg-down outcome split by state KNOWN AT THE 2nd FILL")
    print("=" * 106)
    multi = multi.copy()
    multi["win"] = multi["ret_avg"] > 0
    multi["add2_vs_1"] = [ (np.array(x, float)[1] / np.array(x, float)[0] - 1) if len(x) >= 2 else np.nan
                           for x in multi["adds"] ]
    multi["start_hr"] = multi["first_mod"] // 60
    for col, label in (("add2_vs_1", "2nd add vs 1st (cheaper = averaging down)"),
                       ("nfill", "# fills that day"),
                       ("start_mod_", "first fill time"),
                       ("trend", "trend regime"), ("vol", "vol regime"), ("gex", "GEX regime")):
        if col == "start_mod_":
            sub = multi.dropna(subset=["first_mod"])
            q = pd.qcut(sub["first_mod"], min(3, sub["first_mod"].nunique()), labels=False, duplicates="drop")
            cells = [f"Q{k+1} {sub[q==k].ret_avg.mean()*100:+.0f}%(n{(q==k).sum()})"
                     for k in sorted(set(q.dropna()))]
        elif not pd.api.types.is_numeric_dtype(multi[col]):
            cells = [f"{g} {x.ret_avg.mean()*100:+.0f}%(n{len(x)})"
                     for g, x in multi.groupby(col) if len(x) >= 3]
        else:
            sub = multi.dropna(subset=[col])
            if sub[col].nunique() < 3:
                continue
            q = pd.qcut(sub[col], min(3, sub[col].nunique()), labels=False, duplicates="drop")
            cells = [f"Q{k+1} {sub[q==k].ret_avg.mean()*100:+.0f}%(n{(q==k).sum()})"
                     for k in sorted(set(q.dropna()))]
        print(f"  {label:42} " + "  ".join(cells))

    # ---- greeks: is the position getting MORE FAVOURABLE as we add? ----
    if a.greeks:
        needs = {(x["date"], x["cid"]) for _, x in multi.iterrows() if x["cid"]}
        gk = _greeks_at(needs)
        print("\n" + "=" * 106)
        print(f"  DELTA / GAMMA AT EACH ADD   ({len(gk)} contract-days resolved of {len(needs)})")
        print("=" * 106)
        d_ad, d_lvl, g_ch, g_lvl, full = [], [], [], [], []
        for _, x in multi.iterrows():
            arr = gk.get((x["date"], x["cid"]))
            if arr is None:
                d_ad.append(np.nan); d_lvl.append(np.nan)
                g_ch.append(np.nan); g_lvl.append(np.nan); full.append(np.nan)
                continue
            sgn = 1.0 if x["direction"] == "CALL" else -1.0
            vals = [_asof(arr, m) for m in x["cfills"]]
            vals = [v for v in vals if v is not None and np.isfinite(v[0])]
            if len(vals) < 2:
                d_ad.append(np.nan); d_lvl.append(np.nan)
                g_ch.append(np.nan); g_lvl.append(np.nan); full.append(np.nan)
                continue
            # signed so + = moving IN THE MONEY = trade growing more favourable
            dl = [sgn * v[0] for v in vals]
            gm = [v[1] for v in vals]
            d_ad.append(dl[1] - dl[0])          # known at 2nd fill
            d_lvl.append(dl[1])
            g_ch.append(gm[1] - gm[0])
            g_lvl.append(gm[1])
            full.append(float(np.mean(np.diff(dl))))   # hindsight: whole path
        multi["d_delta2"] = d_ad
        multi["delta2"] = d_lvl
        multi["d_gamma2"] = g_ch
        multi["gamma2"] = g_lvl
        multi["d_delta_path"] = full

        for col, label in (("d_delta2", "delta CHANGE 1st->2nd add (+ = going ITM)"),
                           ("delta2", "signed delta LEVEL at 2nd add"),
                           ("d_gamma2", "gamma change 1st->2nd add"),
                           ("gamma2", "gamma level at 2nd add"),
                           ("d_delta_path", "[hindsight] mean delta drift over ALL adds")):
            sub = multi.dropna(subset=[col])
            if len(sub) < 15 or sub[col].nunique() < 4:
                print(f"  {label:44} (thin)"); continue
            q = pd.qcut(sub[col], 3, labels=False, duplicates="drop")
            cells = []
            for k in sorted(set(q.dropna())):
                b = sub[q == k]
                bi = b[b.date < SPLIT]; bo = b[b.date >= SPLIT]
                cells.append(f"Q{k+1} {b.ret_avg.mean()*100:+.0f}%"
                             f"(n{len(b)} IS{bi.ret_avg.mean()*100 if len(bi) else float('nan'):+.0f}"
                             f"/OOS{bo.ret_avg.mean()*100 if len(bo) else float('nan'):+.0f})")
            print(f"  {label:44} " + "  ".join(cells))

        # ---- CAUSAL delta-gated scale-in: only add while delta keeps rising ----
        print("\n" + "=" * 106)
        print("  CAUSAL: scale in ONLY while signed delta keeps rising (vs plain scale-in)")
        print("=" * 106)
        import directional_flow_backtester as _D
        rebuilt = {}
        for _, x in multi.iterrows():
            arr = gk.get((x["date"], x["cid"]))
            key = (x["rule"], x["date"])
            rebuilt[key] = dict(base=x["ret_avg"], mae=x["mae"], stack=x["stack"], arr=arr,
                                direction=x["direction"], cfills=x["cfills"], cid=x["cid"],
                                ticker=x["ticker"])
        gated, gated_stop = [], []
        cache_bars = {}
        for (rname, d), rec in rebuilt.items():
            if rec["arr"] is None:
                continue
            tkk = rec["ticker"]
            if tkk not in cache_bars:
                try:
                    tbb = _D._ticker_bars(tkk)
                except Exception:
                    tbb = None
                if tbb is None or tbb.empty:
                    _, tbb = _D._screen_build_one("lake/silver/option-contracts-1m", tkk)
                cache_bars[tkk] = {c: g.sort_values("minute_et") for c, g in tbb.groupby("option_chain_id")}
            bb = cache_bars[tkk].get(rec["cid"])
            if bb is None:
                continue
            bb = bb[bb["date"] == d]
            if bb.empty:
                continue
            sgn = 1.0 if rec["direction"] == "CALL" else -1.0
            for store, stop in ((gated, False), (gated_stop, True)):
                rr2 = _avg_down_delta(bb, rec["cfills"], 1.0, 1.0, 15 * 60 + 55,
                                      rec["arr"], sgn, stop_on_fall=stop)
                if rr2:
                    store.append((d, rr2["ret"], rr2["n"], rr2["mae"]))
        for lbl, store in (("plain scale-in", [(x["date"], x["ret_avg"], x["stack"], x["mae"])
                                               for _, x in multi.iterrows()]),
                           ("delta-gated adds", gated),
                           ("delta-gated + exit", gated_stop)):
            if len(store) < 20:
                continue
            v = np.array([r for _, r, _, _ in store])
            di = np.array([r for d, r, _, _ in store if d < SPLIT])
            do = np.array([r for d, r, _, _ in store if d >= SPLIT])
            st = np.array([s for _, _, s, _ in store], float)
            ma = np.array([m for _, _, _, m in store], float)
            print(f"    {lbl:20} n={len(v):>3}  all {v.mean()*100:>+6.1f}%  "
                  f"IS {di.mean()*100 if len(di) else float('nan'):>+6.1f}%  "
                  f"OOS {do.mean()*100 if len(do) else float('nan'):>+6.1f}%  "
                  f"win {(v>0).mean():.2f}  stack {st.mean():.1f}  MAE {ma.mean()*100:>+5.0f}%")

        # the decision that matters: gate the 2nd+ add on rising delta
        ok = multi.dropna(subset=["d_delta2"])
        if len(ok) >= 30:
            hi = ok[ok["d_delta2"] > 0]
            lo = ok[ok["d_delta2"] <= 0]
            print(f"\n  GATE: only keep scaling when delta rose between add 1 and 2")
            for lbl, s in (("delta RISING", hi), ("delta falling", lo)):
                si = s[s.date < SPLIT]; so = s[s.date >= SPLIT]
                print(f"    {lbl:15} n={len(s):>3}  avg-down {s.ret_avg.mean()*100:>+6.1f}%  "
                      f"IS {si.ret_avg.mean()*100 if len(si) else float('nan'):>+6.1f}%  "
                      f"OOS {so.ret_avg.mean()*100 if len(so) else float('nan'):>+6.1f}%  "
                      f"win {(s.ret_avg > 0).mean():.2f}  "
                      f"MAE {s['mae'].mean()*100:>+5.0f}%  stack {s['stack'].mean():.1f}")

    if a.detail:
        print("\n" + "=" * 106)
        for _, x in multi.nlargest(a.detail, "nfill").iterrows():
            ad = np.array(x["adds"], float)
            print(f"\n  {x['rule']}  {x['date']}  ({'IS' if x['date'] < SPLIT else 'OOS'})")
            print(f"    {len(ad)} adds: " + " ".join(f"{v:.2f}" for v in ad))
            print(f"    avg cost {ad.mean():.3f}   exit {x['why']}   "
                  f"backtest {x['ret_bt']*100:+.1f}%  sequential {x['ret_seq']*100:+.1f}%  "
                  f"AVG-DOWN {x['ret_avg']*100:+.1f}%   stack MAE {x['mae']*100:+.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--direction", choices=("CALL", "PUT"), default=None)
    ap.add_argument("--regime", default=None)
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--dow", nargs="+", default=None)
    ap.add_argument("--dte", nargs="+", type=int, default=None)
    ap.add_argument("--detail", type=int, default=0)
    ap.add_argument("--greeks", action="store_true",
                    help="pull delta/gamma per add from silver and test 'is the "
                         "position getting more favourable' as a scale-in gate")
    a = ap.parse_args()
    run(a)
