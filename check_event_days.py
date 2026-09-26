# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_event_days.py
===================
Does the bot's long-premium flow-scalp P&L suffer on SCHEDULED MACRO-EVENT days
via intraday IV crush?

  * EARNINGS  -- moot: the bot never holds overnight, and a single name's
    earnings crush is pre-open. Not tested.
  * FOMC      -- the pure intraday case: 2:00pm ET decision -> front-month IV
    collapses. Positions held across 2pm eat the crush regardless of direction.
    Dates hardcoded (public, certain).
  * 8:30am data (CPI / NFP / PPI / PCE) -- crush is PRE-open; by 9:30 the
    front-month IV has largely reset. Bot's real exposure is a whippy/gappy
    open + elevated realized vol, not intraday premium decay. NFP = first
    Friday (computed); CPI/PCE approximated + caught by the empirical detector.
  * EMPIRICAL "elevated IV at the open" day -- data-driven catch-all: ATM
    near-dated iv_close in the first 20 min vs the ticker's trailing-20d median.
    Flags CPI/PCE/Fed-speaker/anticipation days a hardcoded list would miss.

Per deployed rule + blended: split option-bracket P&L by event bucket
(IS/OOS @ 2025-08-21 + 6 rolling slices + win + n). Plus a direct IV-path
readout (ATM 0/1DTE mean iv_close by 15-min bucket, FOMC vs clean).

  --build            scan silver -> _atm_iv_cache/{T}.parquet (ATM near-dated IV)
  (default)          the event-bucket P&L split
  --iv-path          just the FOMC-vs-clean ATM-IV intraday curves

Usage:
  python check_event_days.py --build
  python check_event_days.py
  python check_event_days.py --iv-path --tickers SPY QQQ
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

from check_flow_zscore import annotate_flow_z, _z_matched, _eod_mod_for
from check_config_walkforward import _flow_for, _slice_idx
from macro_calendar import CPI, PCE, FOMC, nfp_days_in as _nfp_days

HIST = "historical"
SILVER = "lake/silver/option-contracts-1m"
CACHE = "_atm_iv_cache"
SPLIT = pd.Timestamp("2025-08-21").date()


# --------------------------------------------------------------------------- #
def build(tickers):
    os.makedirs(CACHE, exist_ok=True)
    parts = sorted(glob.glob(f"{SILVER}/date=*/bars.parquet"))
    for tk in tickers:
        fp = os.path.join(CACHE, f"{tk}.parquet")
        if os.path.exists(fp):
            print(f"  {tk}: cached"); continue
        frames = []
        for i, p in enumerate(parts, 1):
            lf = (pl.scan_parquet(p)
                  .filter((pl.col("underlying_symbol") == tk)
                          & (pl.col("iv_close").is_not_null())
                          & (((pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days()).is_between(0, 1))
                          & ((pl.col("strike") - pl.col("underlying_close")).abs()
                             / pl.col("underlying_close") <= 0.01))
                  .select("minute_et", "iv_close"))
            df = lf.collect()
            if df.height:
                frames.append(df)
            if i % 100 == 0:
                print(f"    {tk} {i}/{len(parts)}")
        if not frames:
            print(f"  {tk}: no ATM IV data"); continue
        out = pl.concat(frames).to_pandas()
        et = pd.to_datetime(out["minute_et"]).dt.tz_localize(None)
        out["date"] = et.dt.date
        out["mod15"] = (et.dt.hour * 60 + et.dt.minute) // 15 * 15
        g = out.groupby(["date", "mod15"])["iv_close"].mean().reset_index()
        g.to_parquet(fp, index=False)
        print(f"  {tk}: {len(g)} (date,mod15) rows  {g['date'].min()}..{g['date'].max()}")


def _atm_iv(tk):
    fp = os.path.join(CACHE, f"{tk}.parquet")
    if not os.path.exists(fp):
        return None
    df = pd.read_parquet(fp)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _iv_open_spike_days(tk, thresh=1.15):
    """days where the ATM near-dated IV in the first 20 min runs >= thresh x the
    ticker's trailing-20d median open IV (lookahead-free)."""
    df = _atm_iv(tk)
    if df is None:
        return set()
    op = df[df["mod15"] <= 585].groupby("date")["iv_close"].mean().sort_index()
    med = op.shift(1).rolling(20, min_periods=8).median()
    return set(op.index[(op / med) >= thresh])


# --------------------------------------------------------------------------- #
def _stat(pnls, base=None):
    if len(pnls) < 10:
        return f"n={len(pnls):>4}  (thin)"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    slices = " ".join(f"S{j+1}{np.mean(b) * 100:+.0f}" if len(b) >= 5 else f"S{j+1}··"
                      for j, b in enumerate(sl))
    ret = f" ({100 * len(v) / base:.0f}%)" if base else ""
    return (f"n={len(v):>4}{ret}  avg {np.mean(v) * 100:>+6.1f}%  "
            f"IS {np.mean(i) * 100 if i else float('nan'):>+6.1f}%  OOS {np.mean(o) * 100 if o else float('nan'):>+6.1f}%  "
            f"win {np.mean(v > 0):.2f}  [{slices}]")


def _matched_rows(D, r, flow, gex, vol, trd, amp, reg_src, amt, amt_ok, ema_stacks, bbd, bbc, iadx, intra):
    direction = r["direction"].upper()
    dtes = tuple(r.get("dte", [0, 1]))
    tr, rr = float(r["target_roe"]), float(r["rr"])
    tstop = r.get("time_stop_mins"); eod = _eod_mod_for(r)
    want_bull = direction == "CALL"
    zs = r.get("flow_zscore")
    trigs = D.triggers_for(flow, r["ticker"])
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
    out = []
    for t, _thr in matched:
        d, ts = t["date"], t["ts"]
        if want_amt and not amt_ok(want_amt, amt.get(d)):
            continue
        if ema_stack is not None:
            st = D.ema_state_at(ema_stack, ts)
            if st is not None and st != ("BULL" if want_bull else "BEAR"):
                continue
        if dmi:
            ia = intra(dmi_days.get(d, []), ts) if dmi_days is not None else None
            if ia is not None:
                di_bull = ia[0] > ia[1]
                agree = di_bull == want_bull
                ok = agree if dmi.get("mode") == "agree" else (not agree)
                if not ok:
                    continue
        for p in D._option_paths(t, direction, list(dtes), bbd, bbc):
            out.append((d, D._bracket_pnl(*p, tr, rr, tstop, eod), int(pd.Timestamp(ts).hour)))
    return out


def run(a):
    import directional_flow_backtester as D
    from amt_profile import amt_open_map, amt_ok
    from config import RULES
    from check_adx_dmi import _intraday_adx, _intra_asof

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]
    tickers = sorted({r["ticker"] for r in rules})
    flow_all = _flow_for(D, tickers)
    alld = sorted(pd.to_datetime(flow_all["date"].unique()))
    nfp = _nfp_days([d.date() for d in alld])

    print("=" * 116)
    print("  MACRO-EVENT-DAY P&L SPLIT   (deployed spec; FOMC hardcoded, NFP computed, IV-spike empirical)")
    print(f"  split {SPLIT}")
    print("=" * 116)

    book = {}
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
            _, tb = D._screen_build_one(SILVER, tk)
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        ivspike = _iv_open_spike_days(tk)

        for r in tk_rules:
            rows = _matched_rows(D, r, flow_all, gex, vol, trd, amp, reg_src, amt, amt_ok,
                                 ema_stacks, bbd, bbc, iadx, _intra_asof)
            if len(rows) < 25:
                print(f"\n  {r['name']}: {len(rows)} trades -- skip"); continue
            book.setdefault("ALL", []).extend((d, p) for d, p, h in rows)

            def bucket(name, pred):
                s = [(d, p) for d, p, h in rows if pred(d.isoformat(), h)]
                book.setdefault(name, []).extend(s)
                return s

            base = [(d, p) for d, p, h in rows]
            fomc = bucket("FOMC", lambda ds, h: ds in FOMC)
            fomc_pm = bucket("FOMC held->2pm", lambda ds, h: ds in FOMC and h <= 13)
            data830 = bucket("8:30 data (CPI/NFP/PCE)", lambda ds, h: ds in CPI or ds in nfp or ds in PCE)
            ivsp = bucket("IV-open-spike", lambda ds, h: pd.Timestamp(ds).date() in ivspike)
            clean = bucket("clean day", lambda ds, h: ds not in FOMC and ds not in CPI and ds not in nfp
                           and ds not in PCE and pd.Timestamp(ds).date() not in ivspike)

            print(f"\n  {r['name']}  ({tk} {r['direction']})")
            print(f"    {'BASELINE':24} {_stat(base)}")
            for nm, s in (("FOMC", fomc), ("FOMC held->2pm", fomc_pm),
                          ("8:30 data", data830), ("IV-open-spike", ivsp), ("clean day", clean)):
                print(f"    {nm:24} {_stat(s, len(base))}")

    print("\n" + "=" * 116)
    print("  BLENDED BOOK")
    print("=" * 116)
    base = book.get("ALL", [])
    print(f"    {'BASELINE':24} {_stat(base)}")
    for nm in ("FOMC", "FOMC held->2pm", "8:30 data (CPI/NFP/PCE)", "IV-open-spike", "clean day"):
        print(f"    {nm:24} {_stat(book.get(nm, []), len(base))}")


def iv_path(a):
    tickers = a.tickers or ["SPY", "QQQ", "IWM"]
    for tk in tickers:
        df = _atm_iv(tk)
        if df is None:
            print(f"  {tk}: no cache (run --build)"); continue
        df["is_fomc"] = df["date"].apply(lambda d: d.isoformat() in FOMC)
        print(f"\n  {tk}  ATM 0/1DTE mean iv_close by 15-min bucket   (FOMC n={df[df.is_fomc]['date'].nunique()} days"
              f" / clean n={df[~df.is_fomc]['date'].nunique()})")
        piv = df.groupby(["mod15", "is_fomc"])["iv_close"].mean().unstack()
        piv = piv[(piv.index >= 570) & (piv.index <= 945)]   # RTH, drop the 15:45 expiry blip
        f0 = piv[True].loc[570] if True in piv.columns and 570 in piv.index else np.nan
        c0 = piv[False].loc[570] if False in piv.columns and 570 in piv.index else np.nan
        for mod15, row in piv.iterrows():
            hh, mm = divmod(int(mod15), 60)
            fo = row.get(True, np.nan); cl = row.get(False, np.nan)
            mark = "  <- 2pm decision" if mod15 == 840 else ""
            print(f"    {hh:02d}:{mm:02d}   FOMC {fo:6.3f} ({100*(fo/f0-1):+5.1f}% vs open)   "
                  f"clean {cl:6.3f} ({100*(cl/c0-1):+5.1f}%){mark}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--iv-path", action="store_true")
    ap.add_argument("--tickers", nargs="+", default=None)
    a = ap.parse_args()
    if a.build:
        from config import RULES
        tks = a.tickers or sorted({r["ticker"] for r in RULES if r.get("enabled", True)})
        build([t.upper() for t in tks])
    elif a.iv_path:
        iv_path(a)
    else:
        run(a)
