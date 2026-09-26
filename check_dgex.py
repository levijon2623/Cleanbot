# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_dgex.py
==============

Directionalized GEX (dGEX) vs cumulative/standard GEX.

The lake already carries THREE live-recomputed intraday gamma-exposure series
per ticker in lake/silver/spot-exposures-1m (from UW's /stock/{t}/spot-exposures,
already backfilled for the 1-min flow project):

  gamma_per_one_percent_move_oi   -- textbook GEX: today's resting OPEN INTEREST
                                      x gamma, assuming the standard convention
                                      (dealers short calls / long puts). Frozen
                                      OI, but the gamma itself is recomputed every
                                      minute against the live spot/IV -- NOT the
                                      same thing as the daily historical/GEX{T}.parquet
                                      prior-close snapshot the live bot's "regime"
                                      gate actually uses (that one only updates
                                      once a day, at market close).
  gamma_per_one_percent_move_vol  -- same convention, but weighted by TODAY's
                                      traded volume instead of resting OI (fresh
                                      flow's gamma contribution only).
  gamma_per_one_percent_move_dir  -- DIRECTIONAL: infers actual dealer positioning
                                      from today's SIGNED (aggressor/bid-ask) volume
                                      instead of assuming the textbook convention.
                                      Reads exactly 0.0 until enough of today's
                                      flow has traded to establish a directional
                                      read (usually mid/late morning).

  --build              scan silver spot-exposures-1m -> _dgex_cache/{T}.parquet
  --test agreement     how often / when does sign(dir) disagree with sign(oi)?
  --test regime        forward realized-range & momentum-continuation, split by
                        sign(oi) vs sign(dir) vs the 4-way agree/disagree combo
                        -- which explains pinning/trending better?
  --test rules         SPY POS/PUT is the one deployed rule gated directly on
                        regime="POSITIVE_GEX" (prior-day OI-based). Strip that
                        gate, re-match every trigger, then split P&L by: the
                        live gate as-is, live-gate+dir-agrees, live-gate+dir-
                        disagrees, and a pure dir-only gate (no prior-day OI,
                        no AMT) -- does dGEX improve or replace the live gate?

Usage:
  python check_dgex.py --build 2024-08-20 2026-08-21
  python check_dgex.py --test agreement
  python check_dgex.py --test regime --split 2025-08-21
  python check_dgex.py --test rules --split 2025-08-21
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

HIST = "historical"
SILVER = "lake/silver/spot-exposures-1m"
CACHE = "_dgex_cache"
RULE_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "META", "MSFT", "AMZN", "AVGO", "GLD"]
SPOT_COLS = ["minute_et", "price", "gamma_per_one_percent_move_oi",
             "gamma_per_one_percent_move_vol", "gamma_per_one_percent_move_dir"]


def _load_spot(tk: str, force: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{tk}.parquet")
    if os.path.exists(fp) and not force:
        return pd.read_parquet(fp)
    frames = []
    for d in sorted(glob.glob(f"{SILVER}/date=*/")):
        p = os.path.join(d, f"{tk}.parquet")
        if os.path.exists(p):
            frames.append(pl.read_parquet(p, columns=SPOT_COLS))
    if not frames:
        return pd.DataFrame()
    out = pl.concat(frames).to_pandas()
    out["minute_et"] = pd.to_datetime(out["minute_et"]).dt.tz_localize(None)
    out["date"] = out["minute_et"].dt.date
    out["mod"] = out["minute_et"].dt.hour * 60 + out["minute_et"].dt.minute
    out = out.sort_values("minute_et").reset_index(drop=True)
    out.to_parquet(fp, index=False)
    return out


def build(a):
    for tk in a.tickers:
        df = _load_spot(tk, force=True)
        print(f"  {tk}: {len(df)} rows  {df['date'].min() if len(df) else '--'} .. "
              f"{df['date'].max() if len(df) else '--'}")


def _load_1m(tk):
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).to_pandas()
    d.columns = [c.lower() for c in d.columns]
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    mo = et.dt.hour * 60 + et.dt.minute
    m = (mo >= 570) & (mo <= 960)
    g = pd.DataFrame({"ts": et[m].values, "mod": mo[m].values,
                       "c": d["close"][m].astype(float).values,
                       "h": d["high"][m].astype(float).values,
                       "l": d["low"][m].astype(float).values})
    g["ts"] = pd.to_datetime(g["ts"]); g["date"] = g["ts"].dt.date
    return g.sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------
def test_agreement(a):
    print("=" * 100)
    print("  sign(dir) vs sign(oi) -- establishment rate & disagreement rate by time-of-day")
    print("=" * 100)
    buckets = [(570, 630, "9:30-10:30"), (630, 720, "10:30-12:00"), (720, 810, "12:00-13:30"),
               (810, 900, "13:30-15:00"), (900, 960, "15:00-16:00")]
    for tk in a.tickers:
        sp = _load_spot(tk)
        if sp.empty:
            print(f"  {tk}: no spot-exposures data"); continue
        sp = sp[(sp["mod"] >= 570) & (sp["mod"] <= 960)]
        soi = np.sign(sp["gamma_per_one_percent_move_oi"])
        sdir = np.sign(sp["gamma_per_one_percent_move_dir"])
        est = sdir != 0
        dis = est & (soi != sdir) & (soi != 0)
        print(f"\n  {tk}  (n={len(sp)})")
        for lo, hi, lbl in buckets:
            m = (sp["mod"] >= lo) & (sp["mod"] < hi)
            n = int(m.sum())
            if n == 0:
                continue
            est_pct = 100 * est[m].mean()
            dis_pct = 100 * dis[m].sum() / max(1, est[m].sum())
            print(f"    {lbl:12} n={n:>6}  dir-established={est_pct:5.1f}%   "
                  f"disagree-when-established={dis_pct:5.1f}%")
        overall_est = est.mean()
        overall_dis = dis.sum() / max(1, est.sum())
        print(f"    {'ALL DAY':12} n={len(sp):>6}  dir-established={100*overall_est:5.1f}%   "
              f"disagree-when-established={100*overall_dis:5.1f}%")


# --------------------------------------------------------------------------
def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def test_regime(a):
    split = pd.Timestamp(a.split).date()
    print("=" * 100)
    print("  forward 30m realized behaviour, split by sign(oi) / sign(dir) / 4-way combo")
    print("  (fwd |ret| = pinning proxy: lower = dampened/pinned; higher = amplified)")
    print("  (mom = spearman(trailing 15m ret, forward 30m ret): + = trend-continuation, - = mean-reversion)")
    print("=" * 100)
    for tk in a.tickers:
        sp = _load_spot(tk)
        px = _load_1m(tk)
        if sp.empty or px is None:
            print(f"  {tk}: missing data"); continue
        pxd = {d: g.set_index("mod") for d, g in px.groupby("date")}
        sp = sp[(sp["mod"] >= 575) & (sp["mod"] <= 900) & (sp["mod"] % 5 == 0)]
        rows = []
        for _, r in sp.iterrows():
            d, m = r["date"], int(r["mod"])
            dpx = pxd.get(d)
            if dpx is None:
                continue
            idx = dpx.index
            p0i = idx[idx <= m]
            plagi = idx[idx <= m - 15]
            pfwdi = idx[idx <= m + 30]
            if len(p0i) == 0 or len(plagi) == 0 or len(pfwdi) == 0 or pfwdi[-1] - m < 18:
                continue
            p0 = dpx.loc[p0i[-1], "c"]
            plag = dpx.loc[plagi[-1], "c"]
            fwd_seg = dpx.loc[(idx > m) & (idx <= m + 30)]
            if fwd_seg.empty:
                continue
            fret = dpx.loc[pfwdi[-1], "c"] / p0 - 1
            frange = (fwd_seg["h"].max() - fwd_seg["l"].min()) / p0
            lag_ret = p0 / plag - 1
            rows.append((d, r["gamma_per_one_percent_move_oi"], r["gamma_per_one_percent_move_dir"],
                         lag_ret, fret, frange))
        if not rows:
            print(f"  {tk}: no rows"); continue
        R = pd.DataFrame(rows, columns=["date", "oi", "dir", "lag_ret", "fret", "frange"])
        R["s_oi"] = np.sign(R["oi"]); R["s_dir"] = np.sign(R["dir"])
        print(f"\n  {tk}  (n={len(R)})")
        for lbl, sub in (("IS", R[R.date < split]), ("OOS", R[R.date >= split])):
            base_absret = sub["fret"].abs().mean() * 100
            base_range = sub["frange"].mean() * 100
            base_mom = _spear(sub["lag_ret"], sub["fret"])
            print(f"    {lbl}  n={len(sub):>5}  baseline |fwdret|={base_absret:.3f}%  "
                  f"range={base_range:.3f}%  mom={base_mom:+.3f}")
            for sig in ("s_oi", "s_dir"):
                for sv, sn in ((1, "POS"), (-1, "NEG")):
                    b = sub[sub[sig] == sv]
                    if len(b) < 30:
                        continue
                    print(f"      {sig}={sn:4} n={len(b):>5}  |fwdret|={b['fret'].abs().mean()*100:6.3f}%  "
                          f"range={b['frange'].mean()*100:6.3f}%  mom={_spear(b['lag_ret'], b['fret']):+.3f}")
            for so, sd, cn in ((1, 1, "agree+"), (-1, -1, "agree-"), (1, -1, "oi+/dir-"), (-1, 1, "oi-/dir+")):
                b = sub[(sub.s_oi == so) & (sub.s_dir == sd)]
                if len(b) < 20:
                    continue
                print(f"      combo {cn:9} n={len(b):>5}  |fwdret|={b['fret'].abs().mean()*100:6.3f}%  "
                      f"range={b['frange'].mean()*100:6.3f}%  mom={_spear(b['lag_ret'], b['fret']):+.3f}")


# --------------------------------------------------------------------------
def _dir_sign_asof(sp_by_date, d, ts):
    g = sp_by_date.get(d)
    if g is None:
        return 0
    m = ts.hour * 60 + ts.minute
    idx = g.index[g.index <= m]
    if len(idx) == 0:
        return 0
    return int(np.sign(g.loc[idx[-1], "gamma_per_one_percent_move_dir"]))


def test_rules(a):
    import directional_flow_backtester as D
    from config import RULES
    from amt_profile import amt_open_map

    split = pd.Timestamp(a.split).date()
    targets = [r for r in RULES if r.get("enabled", True) and r.get("regime", "").endswith("_GEX")]
    if not targets:
        print("  no live rules gate directly on regime=*_GEX"); return

    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date

    for r in targets:
        tk = r["ticker"]
        sp = _load_spot(tk)
        if sp.empty:
            print(f"  {tk}: no spot-exposures data, skipping"); continue
        sp_by_date = {d: g.set_index("mod") for d, g in sp.groupby("date")}
        amt = amt_open_map(tk)
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}

        trigs = D.triggers_for(flow, tk)
        if not trigs:
            tkf, _ = D._screen_build_one("lake/silver/option-contracts-1m", tk)
            trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
        D.annotate_flow_pct(trigs, 60)
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}

        r_bare = {k: v for k, v in r.items() if k not in ("regime", "amt_open")}
        matched = D._rule_matched_trigs(r_bare, trigs, gex, vol, trd, amp, reg_src)
        want_oi = "POSITIVE" if r["regime"] == "POSITIVE_GEX" else "NEGATIVE"
        want_amt = r.get("amt_open")

        buckets = {"ALL": [], "live_gate(oi+amt)": [], "live+dir_agrees": [], "live+dir_disagrees": [],
                   "dir_only(no_oi/amt)": []}
        for t, thr in matched:
            d, ts = t["date"], t["ts"]
            oi_ok = gex.get(d) == want_oi
            amt_ok = (amt.get(d) == want_amt) if want_amt else True
            live = oi_ok and amt_ok
            dsign = _dir_sign_asof(sp_by_date, d, pd.Timestamp(ts))
            dir_ok = (dsign > 0) if want_oi == "POSITIVE" else (dsign < 0)
            for _, _, _, dd, pnl in D.simulate_trigger(t, r["direction"].upper(), r.get("dte", [0, 1]),
                                                        r.get("time_stop_mins"), bbd, bbc,
                                                        only=(thr, float(r["target_roe"]), float(r["rr"]))):
                buckets["ALL"].append((dd, pnl))
                if live:
                    buckets["live_gate(oi+amt)"].append((dd, pnl))
                    if dsign != 0:
                        buckets["live+dir_agrees" if dir_ok else "live+dir_disagrees"].append((dd, pnl))
                if dsign != 0 and dir_ok:
                    buckets["dir_only(no_oi/amt)"].append((dd, pnl))

        print(f"\n  {r['name']}  ({r['ticker']} {r['direction']}, gated on {r['regime']}"
              f"{' + amt_open=' + str(want_amt) if want_amt else ''})")
        for k, v in buckets.items():
            if len(v) < 15:
                print(f"    {k:20} n={len(v):>4}  (thin)"); continue
            ii = [p for d, p in v if d < split]; oo = [p for d, p in v if d >= split]
            print(f"    {k:20} n={len(v):>4}  IS {np.mean(ii)*100 if ii else float('nan'):>+6.1f}%  "
                  f"OOS {np.mean(oo)*100 if oo else float('nan'):>+6.1f}%")


def _load_rules(a):
    """config.RULES (enabled only) by default, or --rule-file <json> (candidate/
    disabled rules, e.g. disabled_candidate_rules.json -- flow_pct -> min_flow_pct,
    enabled defaults True)."""
    if a.rule_file:
        import json
        with open(a.rule_file) as f:
            rules = json.load(f)
        for r in rules:
            r.setdefault("min_flow_pct", r.get("flow_pct"))
            r.setdefault("enabled", True)
        return rules
    from config import RULES
    return [r for r in RULES if r.get("enabled", True)]


# --------------------------------------------------------------------------
TR_MULT_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _eod_mod_for(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _sign_asof(sp_by_date, d, ts, col):
    g = sp_by_date.get(d)
    if g is None:
        return 0
    m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
    idx = g.index[g.index <= m]
    if len(idx) == 0:
        return 0
    return int(np.sign(g.loc[idx[-1], col]))


def _combo_bucket(so, sd):
    if so == 0 or sd == 0:
        return None
    if so > 0 and sd > 0:
        return "agree+"
    if so < 0 and sd < 0:
        return "agree-"
    if so > 0 and sd < 0:
        return "oi+/dir-"
    return "oi-/dir+"


def test_overlay(a):
    """Vol-regime OVERLAY, not an entry gate: every deployed rule keeps its live
    entry conditions untouched. At each trigger, bucket the underlying's live
    oi/dir combo (check_dgex.py --test regime found this brackets forward realized
    vol on 9/9 tickers IS+OOS: oi+/dir- = most pinned, oi-/dir+ = most amplified).
    Pick a target-width MULTIPLIER per bucket -- selected ONLY on the IS half of
    THIS ticker's own trades -- that scales (target_roe, and therefore the stop
    too, since R:R is held fixed) wider in the amplified bucket / tighter in the
    pinned one. Apply the IS-selected multiplier out-of-sample and compare the
    blended OOS expectancy against the rule's fixed, undifferentiated target."""
    import directional_flow_backtester as D
    from amt_profile import amt_open_map, amt_ok

    split = pd.Timestamp(a.split).date()
    flow = D.build_flow_netprem(HIST)
    flow["minute_et"] = D._naive(flow["minute_et"]); flow["date"] = flow["minute_et"].dt.date
    rules = _load_rules(a)

    print("=" * 100)
    print("  VOL-REGIME TARGET/STOP OVERLAY  (bucket mult chosen on IS, applied OOS)"
          + ("   [rule-file: " + a.rule_file + "]" if a.rule_file else ""))
    print("=" * 100)

    g_base_is, g_base_oos, g_ovl_is, g_ovl_oos = [], [], [], []
    for r in rules:
        tk = r["ticker"]
        sp = _load_spot(tk)
        if sp.empty:
            print(f"\n  {r['name']}: no spot-exposures data, skipped"); continue
        sp_by_date = {d: g.set_index("mod") for d, g in sp.groupby("date")}
        amt = amt_open_map(tk) if r.get("amt_open") else {}
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        ema_stack = D.load_ema_stack(HIST, tk, int(r["ema_confirm"])) if r.get("ema_confirm") else None

        trigs = D.triggers_for(flow, tk)
        if not trigs:
            tkf, _ = D._screen_build_one("lake/silver/option-contracts-1m", tk)
            trigs = D.triggers_for(tkf, tk) if tkf is not None and not tkf.empty else []
        D.annotate_flow_pct(trigs, 60)
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}

        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
        base_tr, rr = float(r["target_roe"]), float(r["rr"])
        tstop = r.get("time_stop_mins")
        eod_mod = _eod_mod_for(r)
        want_amt = r.get("amt_open")
        want_bull = r["direction"].upper() == "CALL"

        rows = []   # (date, bucket, path)
        for t, thr in matched:
            d, ts = t["date"], t["ts"]
            if want_amt and not amt_ok(want_amt, amt.get(d)):
                continue
            if ema_stack is not None:
                st = D.ema_state_at(ema_stack, ts)
                if st is not None and st != ("BULL" if want_bull else "BEAR"):
                    continue
            paths = D._option_paths(t, r["direction"].upper(), r.get("dte", [0, 1]), bbd, bbc)
            if not paths:
                continue
            so = _sign_asof(sp_by_date, d, ts, "gamma_per_one_percent_move_oi")
            sd = _sign_asof(sp_by_date, d, ts, "gamma_per_one_percent_move_dir")
            bucket = _combo_bucket(so, sd)
            for p in paths:
                rows.append((d, bucket, p))

        if not rows:
            print(f"\n  {r['name']}: no filled trades, skipped"); continue

        base = [(d, D._bracket_pnl(*p, base_tr, rr, tstop, eod_mod)) for d, b, p in rows]
        base_is = [pnl for d, pnl in base if d < split]
        base_oos = [pnl for d, pnl in base if d >= split]

        mults = {}
        for bucket in ("agree+", "agree-", "oi+/dir-", "oi-/dir+"):
            b_is = [p for d, b, p in rows if b == bucket and d < split]
            if len(b_is) < 20:
                mults[bucket] = 1.0
                continue
            best_mult = 1.0
            best_exp = float(np.mean([D._bracket_pnl(*p, base_tr, rr, tstop, eod_mod) for p in b_is]))
            for mlt in TR_MULT_GRID:
                exp_is = float(np.mean([D._bracket_pnl(*p, base_tr * mlt, rr, tstop, eod_mod) for p in b_is]))
                if exp_is > best_exp:
                    best_exp, best_mult = exp_is, mlt
            mults[bucket] = best_mult

        ovl = [(d, D._bracket_pnl(*p, base_tr * mults.get(b, 1.0), rr, tstop, eod_mod)) for d, b, p in rows]
        ovl_is = [pnl for d, pnl in ovl if d < split]
        ovl_oos = [pnl for d, pnl in ovl if d >= split]

        n_by_bucket = {b: sum(1 for d, bb, p in rows if bb == b) for b in mults}
        print(f"\n  {r['name']}  (base target_roe={base_tr} rr={rr}, n={len(rows)})")
        print(f"    baseline   IS {np.mean(base_is)*100 if base_is else float('nan'):>+6.1f}%   "
              f"OOS {np.mean(base_oos)*100 if base_oos else float('nan'):>+6.1f}%")
        print(f"    overlay    IS {np.mean(ovl_is)*100 if ovl_is else float('nan'):>+6.1f}%   "
              f"OOS {np.mean(ovl_oos)*100 if ovl_oos else float('nan'):>+6.1f}%   "
              f"mult: " + ", ".join(f"{b}={mults[b]}(n={n_by_bucket[b]})" for b in mults))

        g_base_is += base_is; g_base_oos += base_oos
        g_ovl_is += ovl_is; g_ovl_oos += ovl_oos

    print("\n" + "-" * 100)
    print(f"  TOTAL ({len(g_base_oos)} OOS trades)   "
          f"baseline IS {np.mean(g_base_is)*100:+.2f}% OOS {np.mean(g_base_oos)*100:+.2f}%   "
          f"overlay IS {np.mean(g_ovl_is)*100:+.2f}% OOS {np.mean(g_ovl_oos)*100:+.2f}%")


TESTS = {"agreement": test_agreement, "regime": test_regime, "rules": test_rules, "overlay": test_overlay}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", nargs="*", metavar="DATE")
    ap.add_argument("--tickers", nargs="+", default=RULE_TICKERS)
    ap.add_argument("--test", choices=list(TESTS))
    ap.add_argument("--split", default="2025-08-21")
    ap.add_argument("--rule-file", default=None, help="(--test overlay only) candidate/disabled rules JSON")
    a = ap.parse_args()
    a.tickers = [t.upper() for t in a.tickers]
    if a.build is not None:
        build(a)
    elif a.test:
        TESTS[a.test](a)
    else:
        ap.print_help()
