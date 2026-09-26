# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_config_walkforward.py
===========================
Audit of the ASSEMBLED config -- not a re-validation of any single lever.

Every enabled config.RULES entry has been through many one-lever "does X help"
checks against the SAME 2025-08-21 split, and the multi-gate rules (IWM / QQQ /
META especially) stack filters that were each validated against a SIMPLER
baseline, never against each other. This runs each rule's EXACT deployed spec --
regime, amp_min, min_flow_pct OR flow_zscore, flow_window_days, amt_open,
ema_confirm, eod_flatten, time_stop_mins, dex_pct_max, vol_overlay -- in ONE
pass, two ways:

  PLAIN     = regime + amp_min + min_flow_pct (60d pct window) + eod/tstop
  DEPLOYED  = PLAIN + every entry filter + vol_overlay target scaling

and reports, per rule and blended across the book:
  * canonical IS/OOS @ 2025-08-21  (n, expectancy, win, maxLL)
  * expectancy in 6 sequential ~4-month calendar SLICES -- is the edge stable
    or was it front-loaded / one-window?

Reads netprem/live-units flow straight from historical/NETPREM{T}.parquet (the
exact quantity the bot accumulates), option bars from silver, dex from
historical/GEX{T}.parquet, spot-exposures for vol_overlay from _cv_cache/ or
lake/silver/spot-exposures-1m.

Usage:
  python check_config_walkforward.py
  python check_config_walkforward.py --tickers IWM QQQ META
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

from check_flow_zscore import annotate_flow_z, _z_matched, _eod_mod_for
from check_dex import _dex, _feat
from check_dgex import _combo_bucket
# check_adx_dmi imports _flow_for/_slice_idx from this module -> import lazily in run()

HIST = "historical"
SILVER_SPOT = "lake/silver/spot-exposures-1m"
SPLIT = pd.Timestamp("2025-08-21").date()

# 6 sequential calendar slices spanning the lake (2024-08-20 .. 2026-08-21)
SLICE_EDGES = [pd.Timestamp(s).date() for s in
               ("2024-08-20", "2024-12-20", "2025-04-21", "2025-08-21",
                "2025-12-21", "2026-04-22", "2026-08-23")]

# 3 further slices covering the PRE-SAMPLE holdout, 2023-10-12 (the API's history
# floor) .. 2024-08-19. Kept as a SEPARATE list on purpose.
#
# Prepending these to SLICE_EDGES would have renumbered S1..S6 -- the old S1
# would become S4 -- silently invalidating every slice figure already recorded in
# METHODOLOGY.md, the memory file and dozens of script outputs. It would also
# blur the partition boundary that PRESAMPLE_PLAN.md exists to enforce: the
# pre-sample is a THIRD partition, not more in-sample, and it must stay visibly
# separate in every report. So `_slice_idx` still returns None before
# 2024-08-20 -- now deliberately, and documented -- and pre-sample coverage is
# reported as P1..P3 via `_pre_slice_idx`.
PRESAMPLE_EDGES = [pd.Timestamp(s).date() for s in
                   ("2023-10-12", "2023-12-20", "2024-04-20", "2024-08-20")]

STACK_FIELDS = ("amt_open", "ema_confirm", "ema_spans", "flow_zscore",
                "flow_window_days", "dex_pct_max", "vol_overlay", "dmi_confirm",
                "skip_macro_am")


def _slice_idx(d):
    """0..5 for the deployed IS/OOS window, else None (pre-sample included)."""
    for i in range(len(SLICE_EDGES) - 1):
        if SLICE_EDGES[i] <= d < SLICE_EDGES[i + 1]:
            return i
    return None


def _pre_slice_idx(d):
    """0..2 for the pre-sample holdout window, else None."""
    for i in range(len(PRESAMPLE_EDGES) - 1):
        if PRESAMPLE_EDGES[i] <= d < PRESAMPLE_EDGES[i + 1]:
            return i
    return None


def slice_label(d):
    """'S1'..'S6' | 'P1'..'P3' | None -- the partition-safe label for a date."""
    k = _slice_idx(d)
    if k is not None:
        return f"S{k + 1}"
    k = _pre_slice_idx(d)
    return f"P{k + 1}" if k is not None else None


def _maxll(pnls):
    """max consecutive losing trades (chronological)."""
    c = m = 0
    for _, p in sorted(pnls):
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    return m


def _line(lbl, pnls):
    if not pnls:
        return f"    {lbl:10} (no trades)"
    v = np.array([p for _, p in pnls])
    isp = [p for d, p in pnls if d < SPLIT]
    oos = [p for d, p in pnls if d >= SPLIT]
    ie = f"{np.mean(isp) * 100:+6.1f}%" if isp else "   -- "
    oe = f"{np.mean(oos) * 100:+6.1f}%" if oos else "   -- "
    return (f"    {lbl:10} n={len(v):>4}  IS {ie}(n{len(isp):>3})  OOS {oe}(n{len(oos):>3})  "
            f"win {np.mean(v > 0):.2f}  maxLL {_maxll(pnls):>2}")


def _slices_line(lbl, pnls):
    buckets = [[] for _ in range(6)]
    for d, p in pnls:
        i = _slice_idx(d)
        if i is not None:
            buckets[i].append(p)
    cells = []
    for i, b in enumerate(buckets):
        if b:
            cells.append(f"S{i+1} {np.mean(b) * 100:+5.1f}%(n{len(b)})")
        else:
            cells.append(f"S{i+1}   --  ")
    return f"    {lbl:10} " + "  ".join(cells)


# --------------------------------------------------------------------------- #
def _flow_for(D, tickers):
    frames = []
    for tk in tickers:
        p = f"{HIST}/NETPREM{tk}.parquet"
        if not os.path.exists(p):
            print(f"  ! no NETPREM{tk}.parquet -- {tk} rules will be skipped")
            continue
        df = pl.read_parquet(p).to_pandas()
        df["minute_et"] = D._naive(df["minute_et"])
        df["date"] = df["minute_et"].dt.date
        df["underlying_symbol"] = tk
        df["net_flow_1m"] = pd.to_numeric(df["net_premium"], errors="coerce").fillna(0.0)
        df = df.sort_values("minute_et")
        df["cum_flow"] = df.groupby("date")["net_flow_1m"].cumsum()
        frames.append(df[["underlying_symbol", "minute_et", "date", "net_flow_1m", "cum_flow"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _spot_signs(tk):
    """{date: {mod: (sign_oi, sign_dir)}} for vol_overlay. Prefer the _cv_cache
    1-min file (built by check_charm_vanna), else scan silver spot-exposures."""
    fp = os.path.join("_cv_cache", f"{tk}.parquet")
    if os.path.exists(fp):
        df = pd.read_parquet(fp, columns=["date", "mod", "g_oi", "g_dir"])
        df = df.rename(columns={"g_oi": "oi", "g_dir": "dir"})
    else:
        parts = sorted(glob.glob(f"{SILVER_SPOT}/date=*/{tk}.parquet"))
        if not parts:
            return {}
        cols = ["minute_et", "gamma_per_one_percent_move_oi", "gamma_per_one_percent_move_dir"]
        df = pl.concat([pl.read_parquet(p, columns=cols) for p in parts], how="vertical_relaxed").to_pandas()
        et = pd.to_datetime(df["minute_et"]).dt.tz_localize(None)
        df = pd.DataFrame({"date": et.dt.date, "mod": et.dt.hour * 60 + et.dt.minute,
                           "oi": df["gamma_per_one_percent_move_oi"],
                           "dir": df["gamma_per_one_percent_move_dir"]})
    out = {}
    for d, g in df.sort_values("mod").groupby("date"):
        out[d] = list(zip(g["mod"].to_numpy(), np.sign(g["oi"].to_numpy()), np.sign(g["dir"].to_numpy())))
    return out


def _bucket_asof(spot_signs, d, ts):
    rows = spot_signs.get(d)
    if not rows:
        return None
    m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
    so = sd = 0
    for mod, s_oi, s_dir in rows:
        if mod <= m:
            so, sd = s_oi, s_dir
        else:
            break
    return _combo_bucket(so, sd)


# --------------------------------------------------------------------------- #
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

    flow = _flow_for(D, tickers)
    if flow.empty:
        print("no flow data"); return

    print("=" * 116)
    print("  ASSEMBLED-CONFIG WALK-FORWARD   (netprem flow; PLAIN = regime+amp+pct60  vs  DEPLOYED = full stack)")
    print(f"  canonical split {SPLIT}   |   slices: " +
          " ".join(f"S{i+1}={SLICE_EDGES[i]}" for i in range(6)))
    print("=" * 116)

    book = {"PLAIN": [], "DEPLOYED": []}

    for tk in tickers:
        tk_rules = [r for r in rules if r["ticker"] == tk]
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        amt = amt_open_map(tk) if any(r.get("amt_open") for r in tk_rules) else {}
        ema_stacks = {int(r["ema_confirm"]): D.load_ema_stack(HIST, tk, int(r["ema_confirm"]))
                      for r in tk_rules if r.get("ema_confirm")}
        need_overlay = any(r.get("vol_overlay") for r in tk_rules)
        spot_signs = _spot_signs(tk) if need_overlay else {}
        need_dex = any(r.get("dex_pct_max") is not None for r in tk_rules)
        dpct = {}
        if need_dex:
            dx = _dex(tk)
            if dx is not None:
                dpct = {d.date(): v for d, v in _feat(dx, False)["dex_pct"].items() if pd.notna(v)}
        dmi_tfs = {int(r["dmi_confirm"].get("tf", 15)) for r in tk_rules if r.get("dmi_confirm")}
        iadx = {tf: _intraday_adx(tk, tf, close_only=True) for tf in dmi_tfs}

        trigs = D.triggers_for(flow, tk)
        tb = D._ticker_bars(tk)
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        if not trigs or tb is None or tb.empty:
            print(f"\n  {tk}: no triggers/bars -- skipping {[r['name'] for r in tk_rules]}"); continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        path_cache = {}

        def paths_for(t, direction, dtes):
            key = (id(t), tuple(dtes))
            if key not in path_cache:
                path_cache[key] = D._option_paths(t, direction, list(dtes), bbd, bbc)
            return path_cache[key]

        for r in tk_rules:
            direction = r["direction"].upper()
            dtes = tuple(r.get("dte", [0, 1]))
            base_tr, rr = float(r["target_roe"]), float(r["rr"])
            tstop = r.get("time_stop_mins")
            eod = _eod_mod_for(r)
            want_bull = direction == "CALL"
            plain_r = {k: v for k, v in r.items() if k not in STACK_FIELDS}

            def eval_cfg(deployed: bool):
                rr_use = r if deployed else plain_r
                zs = rr_use.get("flow_zscore")
                if deployed and zs:
                    annotate_flow_z(trigs, int(zs.get("window_days") or 60))
                    matched = [(t, None) for t in _z_matched(D, rr_use, trigs, gex, vol, trd, amp, reg_src, float(zs["k"]))]
                else:
                    win = int(rr_use.get("flow_window_days") or 60) if deployed else 60
                    D.annotate_flow_pct(trigs, win)
                    matched = D._rule_matched_trigs(rr_use, trigs, gex, vol, trd, amp, reg_src)
                matched = sorted(matched, key=lambda x: pd.Timestamp(x[0]["ts"]))
                want_amt = rr_use.get("amt_open") if deployed else None
                ema_stack = ema_stacks.get(int(r["ema_confirm"])) if (deployed and r.get("ema_confirm")) else None
                dcap = r.get("dex_pct_max") if deployed else None
                ov = r.get("vol_overlay") if deployed else None
                dmi = r.get("dmi_confirm") if deployed else None
                dmi_days = iadx.get(int(dmi.get("tf", 15))) if dmi else None
                skip_am = bool(r.get("skip_macro_am")) if deployed else False
                pnls = []
                for t, _thr in matched:
                    d, ts = t["date"], t["ts"]
                    if skip_am and is_macro_am_day(d):
                        continue
                    if want_amt and not amt_ok(want_amt, amt.get(d)):
                        continue
                    if dmi:
                        ia = _intra_asof(dmi_days.get(d, []), ts)
                        if ia is None:
                            pass  # not warm -> fail open (matches bot_runner)
                        else:
                            di_bull = ia[0] > ia[1]
                            agree = di_bull == want_bull
                            ok = agree if dmi.get("mode") == "agree" else (not agree)
                            if not ok:
                                continue
                    if ema_stack is not None:
                        st = D.ema_state_at(ema_stack, ts)
                        if st is not None and st != ("BULL" if want_bull else "BEAR"):
                            continue
                    if dcap is not None:
                        dp = dpct.get(d)
                        if dp is None or dp >= dcap:
                            continue
                    mult = 1.0
                    if ov:
                        b = _bucket_asof(spot_signs, d, ts)
                        mult = float(ov.get(b, 1.0)) if b else 1.0
                    paths = paths_for(t, direction, dtes)
                    if a.sequential:
                        # one position per ticker (bot_runner.py:1288). The bot
                        # takes the first fillable dte in the rule's order, so
                        # paths[0]; the walk enforces the no-re-entry window.
                        if paths:
                            m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
                            pnls.append((d, m, paths[0], base_tr * mult))
                    else:
                        for p in paths:
                            pnls.append((d, D._bracket_pnl(*p, base_tr * mult, rr, tstop, eod)))
                if a.sequential:
                    from sequential_fills import walk
                    return walk(pnls, rr, tstop, eod)
                return pnls

            plain = eval_cfg(False)
            dep = eval_cfg(True)
            book["PLAIN"] += plain
            book["DEPLOYED"] += dep

            gates = [f for f in ("amt_open", "ema_confirm", "flow_zscore", "flow_window_days",
                                 "dex_pct_max", "vol_overlay", "dmi_confirm", "skip_macro_am")
                     if r.get(f) is not None]
            print(f"\n  {r['name']}  ({tk} {direction})   stack: {', '.join(gates) or 'none'}")
            print(_line("PLAIN", plain))
            print(_line("DEPLOYED", dep))
            print(_slices_line("  pl-slice", plain))
            print(_slices_line("  dep-slice", dep))

    print("\n" + "=" * 116)
    print("  BLENDED BOOK")
    print("=" * 116)
    print(_line("PLAIN", book["PLAIN"]))
    print(_line("DEPLOYED", book["DEPLOYED"]))
    print(_slices_line("  pl-slice", book["PLAIN"]))
    print(_slices_line("  dep-slice", book["DEPLOYED"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--sequential", action="store_true",
                    help="score with the LIVE one-position-per-ticker guard "
                         "(bot_runner.py:1288) instead of every trigger independently")
    a = ap.parse_args()
    run(a)
