# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_screen_candidate.py
=========================
Generalised version of check_msft_chop_put.py -- take a screen hit from
check_etf_screen.py through the same validation battery before it goes into
config.RULES:

  * BASELINE                the candidate exactly as screened  (IS/OOS/6 slices/maxLL)
  * weekday split           Thu (1DTE) vs Fri (0DTE) -- weekly-only-expiry names
  * intraday DMI filter     +DI vs -DI on 10/15/20/30-min bars, "opposes the trade"
                            (for a PUT: price grinding UP when put-flow fires -> fade)
  * di_spread magnitude     (15m)  +  intraday ADX level bucket
  * daily DMI filter        (no live intraday calc needed)
  * stacked daily & intra
  * refine the winner       by hour / sub-regime
  * BOOTSTRAP NULL          gated OOS vs p95 of a same-size random OOS subsample

Regime is matched as an OR over the volume regime (LOWVOL/NORMVOL/HIVOL) and the
trend regime (UPTREND/DOWNTREND/CHOP): --regime CHOP           -> trend==CHOP
                                       --regime UPTREND HIVOL  -> trend==UPTREND OR vol==HIVOL

Needs (same as check_etf_screen): historical/NETPREM{T}.parquet, GEX{T}.parquet,
{T}.parquet, and the silver lake for the option bars.

Usage:
  python check_screen_candidate.py TSM  PUT --regime CHOP          --dow thu fri --pct 50
  python check_screen_candidate.py LULU PUT --regime UPTREND HIVOL  --dow fri      --pct 65
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_adx_dmi import _daily_adx, _intraday_adx, _intra_asof
from check_config_walkforward import _flow_for, _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
EOD = 15 * 60 + 55
_DOW = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}
_VOL = {"LOWVOL", "NORMVOL", "HIVOL"}
_TRD = {"UPTREND", "DOWNTREND", "CHOP"}


def _stat(pnls, base_n=None):
    if len(pnls) < 10:
        return f"n={len(pnls):>4}  (thin)"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    c = m = 0
    for _, p in sorted(pnls):
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    npop = sum(1 for b in sl if len(b) >= 5)
    slices = " ".join(f"S{j+1}{np.mean(b) * 100:+.0f}" if len(b) >= 5 else f"S{j+1}··"
                      for j, b in enumerate(sl))
    ret = f" ret {100 * len(v) / base_n:>3.0f}%" if base_n else ""
    # DAY-LEVEL (equal-weight per day) -- the honest unit. Intraday triggers on one
    # ticker/day are ~one bet; a rule firing 20x on 2 good days looks huge
    # trade-weighted and is worthless (WMT LOWVOL CALL, session 18).
    day = {}
    for d, p in pnls:
        day.setdefault(d, []).append(p)
    dm = {d: float(np.mean(x)) for d, x in day.items()}
    di = [x for d, x in dm.items() if d < SPLIT]
    do = [x for d, x in dm.items() if d >= SPLIT]
    dayblk = (f"D[d={len(dm):>3} IS {np.mean(di) * 100 if di else float('nan'):>+6.1f}%(d{len(di)}) "
              f"OOS {np.mean(do) * 100 if do else float('nan'):>+6.1f}%(d{len(do)}) "
              f"win {np.mean([x > 0 for x in dm.values()]):.2f} t/d {len(v) / len(dm):.1f}]")
    return (f"n={len(v):>4}{ret}  IS {np.mean(i) * 100 if i else float('nan'):>+6.1f}%(n{len(i):>3})  "
            f"OOS {np.mean(o) * 100 if o else float('nan'):>+6.1f}%(n{len(o):>3})  win {np.mean(v > 0):.2f}  "
            f"maxLL {m:>2}  pop {npop}/6  [{slices}]  {dayblk}")


def _boot(rows, gated_oos_mean, n_gated, k=1000):
    oos_all = [p for d, p, *_ in rows if d >= SPLIT]
    if len(oos_all) < n_gated + 5 or n_gated < 10:
        return None
    rng = np.random.default_rng(0)
    draws = [np.mean(rng.choice(oos_all, size=n_gated, replace=False)) * 100 for _ in range(k)]
    p95 = float(np.percentile(draws, 95))
    rank = float((np.array(draws) < gated_oos_mean).mean() * 100)
    return float(np.mean(draws)), p95, rank


def run(a):
    import directional_flow_backtester as D

    TK, DIR = a.ticker.upper(), a.direction.upper()
    regset = {r.upper() for r in a.regime} if a.regime else set()
    vol_want = regset & _VOL
    trd_want = regset & _TRD
    dow_keep = {_DOW[d.lower()[:3]] for d in a.dow} if a.dow else None

    flow = _flow_for(D, [TK])
    if flow.empty:
        print(f"  no NETPREM{TK}.parquet"); return
    gex = D.load_gex(HIST, TK); vol = D.load_volume_regime(HIST, TK); trd = D.load_trend_regime(HIST, TK)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}

    dadx = _daily_adx(TK)
    iadx = {bl: _intraday_adx(TK, bl, close_only=True) for bl in (10, 15, 20, 30)}

    trigs = D.triggers_for(flow, TK)
    print(f"  {TK} {DIR}  regime={sorted(regset) or 'ALL'}  dow={sorted(dow_keep) if dow_keep else 'ALL'}  "
          f"pct={a.pct}   ({len(trigs)} raw triggers)")
    tb = D._ticker_bars(TK)
    if tb is None or tb.empty:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", TK)
    if tb is None or tb.empty:
        print("  no option bars"); return
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}

    D.annotate_flow_pct(trigs, 60)
    rule = {"ticker": TK, "direction": DIR, "hours": [9, 10, 11, 12, 13, 14],
            "dte": [0, 1], "min_flow_pct": a.pct, "target_roe": 1.0, "rr": 1.0}
    matched = D._rule_matched_trigs(rule, trigs, gex, vol, trd, amp, reg_src)

    def _regime_ok(d):
        if not regset:
            return True
        return (trd.get(d) in trd_want) or (vol.get(d) in vol_want)

    rows = []   # (date, pnl, hour, weekday, daily_tuple, {bar: intra_tuple})
    kept_days = set()
    for t, _thr in matched:
        d, ts, wd = t["date"], t["ts"], t["date"].weekday()
        if dow_keep is not None and wd not in dow_keep:
            continue
        if not _regime_ok(d):
            continue
        # weekly-only expiry: Thu -> 1DTE only, Fri -> 0DTE only
        dtes = [0, 1] if dow_keep is None else ([1] if wd == 3 else [0] if wd == 4 else [0, 1])
        da = dadx.get(d)
        ia = {bl: _intra_asof(iadx[bl].get(d, []), ts) for bl in iadx}
        for p in D._option_paths(t, DIR, dtes, bbd, bbc):
            rows.append((d, D._bracket_pnl(*p, 1.0, 1.0, None, EOD), t["hour"], wd, da, ia))
            kept_days.add(d)

    if len(rows) < 10:
        print(f"  only {len(rows)} option-trades after filters -- nothing to validate"); return

    base = [(d, p) for d, p, *_ in rows]
    bn = len(base)
    print("=" * 118)
    print(f"  {TK} {DIR} {'/'.join(sorted(regset)) or 'ALL'}"
          f"{' / dow ' + '+'.join(sorted(a.dow)) if a.dow else ''} / p{a.pct}   "
          f"({bn} option-trades, {len(kept_days)} days)   split {SPLIT}")
    print("=" * 118)
    print(f"  {'BASELINE':36} {_stat(base)}")

    # weekday split
    for wd, nm in ((3, "Thu (1DTE)"), (4, "Fri (0DTE)")):
        s = [(d, p) for d, p, hr, w, da, ia in rows if w == wd]
        if s:
            print(f"  {('dow=' + nm):36} {_stat(s, bn)}")

    def sub(pred):
        return [(d, p) for d, p, hr, w, da, ia in rows if pred(hr, da, ia)]

    # "opposes the trade" = DI points against the option direction
    #   PUT  -> want price grinding UP   -> +DI > -DI  -> di_spread > 0
    #   CALL -> want price grinding DOWN -> -DI > +DI  -> di_spread < 0
    def _op_intra(ia, b):
        return ia[b] is not None and ((ia[b][0] > ia[b][1]) if DIR == "PUT" else (ia[b][0] < ia[b][1]))

    def _sp(ia):
        if ia[15] is None:
            return None
        s = ia[15][0] - ia[15][1]
        return s if DIR == "PUT" else -s        # signed so >0 == "opposes"

    print("\n  -- intraday DMI OPPOSES the trade : bar-length robustness --")
    for bl in (10, 15, 20, 30):
        print(f"  {('intra' + str(bl) + 'm opposes'):36} {_stat(sub(lambda hr, da, ia, b=bl: _op_intra(ia, b)), bn)}")
    print("  -- intraday DMI AGREES (should be the WORSE half) --")
    print(f"  {'intra15m agrees':36} "
          f"{_stat(sub(lambda hr, da, ia: ia[15] is not None and not _op_intra(ia, 15)), bn)}")

    print("\n  -- intraday di_spread MAGNITUDE (15m; signed so + = opposes) --")
    for lo, hi, lbl in [(0, 5, "0..5 (barely)"), (5, 15, "5..15"), (15, 999, ">15 (clearly)"),
                        (-999, 0, "< 0 (agrees)")]:
        print(f"  {('spread ' + lbl):36} "
              f"{_stat(sub(lambda hr, da, ia, L=lo, H=hi: (_sp(ia) is not None) and L <= _sp(ia) < H), bn)}")

    print("\n  -- intraday ADX level (15m) --")
    for lo, hi, lbl in [(0, 20, "ADX < 20 (chop)"), (20, 30, "ADX 20-30"), (30, 999, "ADX >= 30")]:
        print(f"  {('intra15m ' + lbl):36} "
              f"{_stat(sub(lambda hr, da, ia, L=lo, H=hi: ia[15] is not None and L <= ia[15][2] < H), bn)}")

    print("\n  -- DAILY DMI (no live intraday calc needed) --")
    def _op_daily(da):
        return da is not None and ((da[3] > 0) if DIR == "PUT" else (da[3] < 0))
    print(f"  {'daily opposes':36} {_stat(sub(lambda hr, da, ia: _op_daily(da)), bn)}")
    print(f"  {'daily agrees':36} {_stat(sub(lambda hr, da, ia: da is not None and not _op_daily(da)), bn)}")
    print(f"  {'daily opposes & ADX<25':36} "
          f"{_stat(sub(lambda hr, da, ia: _op_daily(da) and da[2] < 25), bn)}")

    print("\n  -- STACKED: daily opposes AND intra15m opposes --")
    print(f"  {'daily & intra15m both oppose':36} "
          f"{_stat(sub(lambda hr, da, ia: _op_daily(da) and _op_intra(ia, 15)), bn)}")

    # winner candidate = intra15m opposes
    W = sub(lambda hr, da, ia: _op_intra(ia, 15))
    print("\n  -- refine intra15m-opposes: by hour --")
    for lo in (9, 10, 11, 12):
        s = [(d, p) for d, p, hr, w, da, ia in rows if _op_intra(ia, 15) and lo <= hr < 15]
        print(f"  {('hours ' + str(lo) + '-14'):36} {_stat(s, bn)}")
    if len(regset) > 1 or not regset:
        print("  -- refine intra15m-opposes: by sub-regime --")
        for g in ("UPTREND", "CHOP", "DOWNTREND"):
            s = [(d, p) for d, p, hr, w, da, ia in rows if _op_intra(ia, 15) and trd.get(d) == g]
            if len(s) >= 8:
                print(f"  {('trend=' + g):36} {_stat(s, bn)}")
        for g in ("LOWVOL", "NORMVOL", "HIVOL"):
            s = [(d, p) for d, p, hr, w, da, ia in rows if _op_intra(ia, 15) and vol.get(d) == g]
            if len(s) >= 8:
                print(f"  {('vol=' + g):36} {_stat(s, bn)}")

    w_oos = [p for d, p in W if d >= SPLIT]
    if len(w_oos) >= 10:
        r = _boot(rows, np.mean(w_oos) * 100, len(w_oos), k=a.boot)
        if r:
            mean, p95, rank = r
            verdict = "** BEATS null" if np.mean(w_oos) * 100 > p95 else "(within noise)"
            print(f"\n  bootstrap null (intra15m opposes, OOS n={len(w_oos)}): "
                  f"gated {np.mean(w_oos) * 100:+.1f}%  vs random OOS-subsample mean {mean:+.1f}% / "
                  f"p95 {p95:+.1f}%  ({rank:.0f}th pct)  {verdict}")

    # also bootstrap the BASELINE OOS itself (is the un-gated candidate even real?)
    b_oos = [p for d, p in base if d >= SPLIT]
    print(f"\n  BASELINE OOS mean {np.mean(b_oos) * 100:+.1f}%  (n={len(b_oos)})   "
          f"IS mean {np.mean([p for d, p in base if d < SPLIT]) * 100:+.1f}%   "
          f"-- realistic ask-in/bid-out fill costs ~4pp/trade on top")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker")
    ap.add_argument("direction", choices=("CALL", "PUT"))
    ap.add_argument("--regime", nargs="+", default=None,
                    help="OR over vol/trend regimes, e.g. --regime CHOP  or  --regime UPTREND HIVOL")
    ap.add_argument("--dow", nargs="+", default=None, help="restrict entries, e.g. --dow thu fri")
    ap.add_argument("--pct", type=int, default=65, help="min_flow_pct (default 65)")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    run(a)
