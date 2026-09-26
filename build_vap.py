# /// script
# requires-python = ">=3.11"
# dependencies = ["databento", "polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
build_vap.py
============
VOLUME AT PRICE inside each minute, plus a proper ICEBERG test, from raw MBO.

WHY THIS EXISTS
    check_peak_bar showed the peak bar carries more volume per unit of range
    than the earlier new-high bars of the same move (0.65 rank vs 0.49 placebo)
    -- the absorption SHAPE. But 1m OHLCV cannot say WHERE in the bar that
    volume traded. The distinction that matters:
        volume stacked at the TOP of the candle while price fails to advance
            = someone is absorbing the buying
        volume spread evenly, or concentrated where price left
            = just a busy minute
    Only order-by-order data can separate those, and it exists here for the
    futures proxies (RTY 44 days, NQ 14 days) in lake/mbo/*.dbn.zst.

THE ICEBERG TEST, done properly this time
    `build_mbo_features.absorb` counts, per ORDER, the excess of a fill over
    that order's displayed size. That works (27.9% of RTY minutes are non-zero,
    max 0.52) and it is NOT the dud the module docstring still claims -- line 17
    describes a superseded unknown-order-id definition, line 209 records that
    the old one measured ~0. The docstring is stale, not the feature.
    This adds the complementary LEVEL-based test, which catches hidden size the
    per-order test cannot see (refills arriving as brand-new order ids):
        iceberg = executed volume AT A PRICE / max displayed size EVER SHOWN there
    A level that trades 400 lots while never showing more than 20 was hiding
    380. Ratio >> 1 is hidden liquidity regardless of how the refills surfaced.

COLUMNS EMITTED (per RTH minute)
    vap_pos      volume-weighted position of trades in the minute's range, 0..1
    top25/bot25  share of volume in the top / bottom quarter of that range
    buy_top      buy-aggressor share of the volume in the top quarter
    sell_bot     sell-aggressor share of the volume in the bottom quarter
    vol_hhi      concentration of volume across price levels
    n_levels     distinct traded prices
    ice_ratio    volume-weighted mean of exec/max-displayed across levels
    ice_max      the worst single level
    ice_extreme  the same ratio AT the minute's high (and low) specifically
    tvol, rng_ticks, ret_ticks

POSITIVE CONTROL (do not skip)
    DBN's `side` on a trade is the AGGRESSOR side, but conventions bite. So
    `signed_vol` (buy minus sell) is correlated against the minute's own price
    change and printed. If that correlation is not clearly POSITIVE the
    convention is inverted and every side-split column here is backwards.
    This is the same role `imb` plays in build_mbo_features.

MEMORY: one day at a time via to_ndarray(), walked in a tight loop, released
before the next. Never hold two days. (build_mbo_features.py:46)

Usage:
  python build_vap.py --symbols RTY_c_0 --max-days 2      # validate first
  python build_vap.py
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import time

import numpy as np

ROOT = os.path.join("lake", "mbo")
CACHE = "_vap_cache"
UNDEF = np.iinfo(np.int64).max
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
A_ADD, A_CXL, A_MOD, A_CLR = b"A", b"C", b"M", b"R"
A_TRD, A_FIL = b"T", b"F"
S_BID, S_ASK = b"B", b"A"


def _et_minute(ts_ns: int):
    import pandas as pd
    t = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("America/New_York")
    return t.hour * 60 + t.minute, t.date()


def _emit(trades, exec_by_lvl, maxdisp_by_lvl, hid_by_lvl, cur_min, cur_date):
    """Turn one minute's trade list + level history into a feature row."""
    if not trades:
        return None
    p = np.array([t[0] for t in trades], float)
    q = np.array([t[1] for t in trades], float)
    sd = np.array([t[2] for t in trades])
    tot = q.sum()
    if tot <= 0:
        return None
    lo, hi = p.min(), p.max()
    rng = hi - lo
    # position of each trade within the minute's own traded range
    pos = (p - lo) / rng if rng > 0 else np.full(p.size, 0.5)
    top = pos >= 0.75
    bot = pos <= 0.25
    isbuy = (sd == S_BID)          # validated by the positive control below

    # --- level-based iceberg: traded more than was ever displayed there
    ratios, weights = [], []
    for lv, ev in exec_by_lvl.items():
        md = maxdisp_by_lvl.get(lv, 0)
        if ev > 0 and md > 0:
            ratios.append(ev / md)
            weights.append(ev)
    if ratios:
        r = np.array(ratios, float); w = np.array(weights, float)
        ice_ratio = float((r * w).sum() / w.sum())
        ice_max = float(r.max())
    else:
        ice_ratio = ice_max = np.nan
    # the same ratio specifically AT the minute's extremes -- where absorption
    # at a turn would have to show up
    def _at(level):
        ev = exec_by_lvl.get(level, 0)
        md = maxdisp_by_lvl.get(level, 0)
        return (ev / md) if (ev > 0 and md > 0) else np.nan
    ice_hi, ice_lo = _at(int(hi)), _at(int(lo))

    # --- HIDDEN SIZE (the real iceberg measure), split by where in the range it
    # sat. `hid_top` is the one that answers "was the buying at the top of the
    # candle being absorbed by size that was never shown?"
    hid_tot = float(sum(hid_by_lvl.values()))
    exec_tot = float(sum(exec_by_lvl.values())) or np.nan
    lvl_pos = {lv: ((lv - lo) / rng if rng > 0 else 0.5) for lv in exec_by_lvl}
    h_top = sum(v for lv, v in hid_by_lvl.items() if lvl_pos.get(lv, 0.5) >= 0.75)
    e_top = sum(v for lv, v in exec_by_lvl.items() if lvl_pos.get(lv, 0.5) >= 0.75)
    h_bot = sum(v for lv, v in hid_by_lvl.items() if lvl_pos.get(lv, 0.5) <= 0.25)
    e_bot = sum(v for lv, v in exec_by_lvl.items() if lvl_pos.get(lv, 0.5) <= 0.25)

    # volume concentration across traded prices
    vols = {}
    for pp, qq, _ in trades:
        vols[pp] = vols.get(pp, 0.0) + qq
    vv = np.array(list(vols.values()), float)
    hhi = float(((vv / vv.sum()) ** 2).sum()) if vv.sum() > 0 else np.nan

    return dict(
        date=cur_date, mod=cur_min,
        vap_pos=float((q * pos).sum() / tot),
        top25=float(q[top].sum() / tot), bot25=float(q[bot].sum() / tot),
        buy_top=float(q[top & isbuy].sum() / q[top].sum()) if q[top].sum() > 0 else np.nan,
        sell_bot=float(q[bot & ~isbuy].sum() / q[bot].sum()) if q[bot].sum() > 0 else np.nan,
        vol_hhi=hhi, n_levels=len(vols),
        # turnover ratios: churn relative to instantaneous depth, NOT icebergs
        ice_ratio=ice_ratio, ice_max=ice_max, ice_hi=ice_hi, ice_lo=ice_lo,
        # hidden size: the actual iceberg measure
        hid_share=float(hid_tot / exec_tot) if np.isfinite(exec_tot) else np.nan,
        hid_top=float(h_top / e_top) if e_top > 0 else np.nan,
        hid_bot=float(h_bot / e_bot) if e_bot > 0 else np.nan,
        hid_vol=hid_tot,
        tvol=float(tot), rng_ticks=float(rng),
        signed_vol=float((q[isbuy].sum() - q[~isbuy].sum()) / tot),
        first_px=float(p[0]), last_px=float(p[-1]),
    )


def vap_for_day(path):
    import databento as db
    arr = db.DBNStore.from_file(path).to_ndarray()
    act = arr["action"]; sd = arr["side"]; px = arr["price"]
    sz = arr["size"]; oid = arr["order_id"]; trc = arr["ts_recv"]

    orders = {}                     # oid -> (side, price, size)
    level = {}                      # price -> displayed size (both sides pooled:
                                    # an iceberg refills on ONE side, and pooling
                                    # keeps the denominator honest when the touch
                                    # flips mid-minute)
    trades, exec_lvl, maxdisp, hid_lvl = [], {}, {}, {}
    rows = []
    cur_min = cur_date = None
    n_trd = n_fil = 0

    def _bump_disp(p):
        v = level.get(p, 0)
        if v > maxdisp.get(p, 0):
            maxdisp[p] = v

    for i in range(len(arr)):
        a = act[i]
        o = oid[i]
        if a == A_ADD:
            p, q = int(px[i]), int(sz[i])
            if p != UNDEF:
                orders[o] = (sd[i], p, q)
                level[p] = level.get(p, 0) + q
                _bump_disp(p)
        elif a == A_CXL:
            prev = orders.pop(o, None)
            if prev is not None:
                level[prev[1]] = level.get(prev[1], 0) - prev[2]
                if level[prev[1]] <= 0:
                    level.pop(prev[1], None)
        elif a == A_MOD:
            prev = orders.pop(o, None)
            p, q = int(px[i]), int(sz[i])
            if prev is not None:
                level[prev[1]] = level.get(prev[1], 0) - prev[2]
                if level[prev[1]] <= 0:
                    level.pop(prev[1], None)
            if p != UNDEF:
                orders[o] = (sd[i], p, q)
                level[p] = level.get(p, 0) + q
                _bump_disp(p)
        elif a == A_FIL:
            n_fil += 1
            prev = orders.get(o)
            f = int(sz[i])
            if prev is not None:
                d = int(prev[2])
                # HIDDEN SIZE, per order: this execution was LARGER than what
                # this order had displayed, so the excess was never on the book.
                # This -- not the level turnover ratio -- is the iceberg test.
                # A level can trade 10x its displayed depth with nothing hidden
                # at all, if ten honest orders replenish it in sequence.
                if f > d:
                    hid_lvl[prev[1]] = hid_lvl.get(prev[1], 0.0) + (f - d)
                rem = d - f
                level[prev[1]] = level.get(prev[1], 0) - min(f, d)
                if level.get(prev[1], 0) <= 0:
                    level.pop(prev[1], None)
                if rem > 0:
                    orders[o] = (prev[0], prev[1], rem)
                else:
                    orders.pop(o, None)
        elif a == A_TRD:
            # the aggregate print: price, size, and the AGGRESSOR side
            p, q = int(px[i]), int(sz[i])
            if p != UNDEF and q > 0:
                n_trd += 1
                trades.append((p, float(q), sd[i]))
                exec_lvl[p] = exec_lvl.get(p, 0.0) + q
                _bump_disp(p)       # capture what was showing when it traded
        elif a == A_CLR:
            orders.clear(); level.clear()

        m, d = _et_minute(int(trc[i])) if (i % 1024 == 0) else (cur_min, cur_date)
        if cur_min is None:
            cur_min, cur_date = m, d
        elif m != cur_min:
            if RTH_LO <= cur_min <= RTH_HI:
                r = _emit(trades, exec_lvl, maxdisp, hid_lvl, cur_min, cur_date)
                if r:
                    rows.append(r)
            cur_min, cur_date = m, d
            trades, exec_lvl, maxdisp, hid_lvl = [], {}, {}, {}

    del arr
    gc.collect()
    return rows, n_trd, n_fil


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--max-days", type=int, default=0)
    a = ap.parse_args()

    import polars as pl
    os.makedirs(CACHE, exist_ok=True)
    syms = a.symbols or sorted(os.listdir(ROOT))
    for sym in syms:
        files = sorted(glob.glob(os.path.join(ROOT, sym, "date=*", "*.dbn.zst")))
        if a.max_days:
            files = files[:a.max_days]
        if not files:
            continue
        print(f"\n=== {sym}: {len(files)} days ===", flush=True)
        allrows, t0, TRD, FIL = [], time.time(), 0, 0
        for j, f in enumerate(files, 1):
            try:
                r, nt, nf = vap_for_day(f)
            except Exception as e:
                print(f"  ! {os.path.basename(os.path.dirname(f))}: {type(e).__name__}: {e}")
                continue
            allrows += r; TRD += nt; FIL += nf
            el = time.time() - t0
            print(f"  {j:>3}/{len(files)} {os.path.basename(os.path.dirname(f)):18} "
                  f"{len(r):>4} rows  T={nt:,} F={nf:,}  {el:>6.0f}s "
                  f"eta {el/j*(len(files)-j):>5.0f}s", flush=True)
            gc.collect()
        if not allrows:
            print(f"  no rows -- are there any 'T' records? T={TRD} F={FIL}")
            continue
        df = pl.DataFrame(allrows)
        df = df.with_columns(((pl.col("last_px") - pl.col("first_px"))).alias("ret_ticks"))
        p = os.path.join(CACHE, f"{sym}.parquet")
        df.write_parquet(p)
        print(f"  -> {p}  {df.height:,} rows")

        # ---- POSITIVE CONTROL: is `side` really the aggressor?
        d = df.filter(pl.col("ret_ticks").is_not_null() & pl.col("signed_vol").is_not_null())
        if d.height > 100:
            c = float(np.corrcoef(d["signed_vol"].to_numpy(),
                                  d["ret_ticks"].to_numpy())[0, 1])
            verdict = ("OK - side is the aggressor" if c > 0.15 else
                       "INVERTED - flip isbuy in _emit" if c < -0.15 else
                       "WEAK - inspect before trusting side splits")
            print(f"  POSITIVE CONTROL corr(signed_vol, ret_ticks) = {c:+.3f}   {verdict}")
        print(df.select(["vap_pos", "top25", "ice_ratio", "ice_max", "tvol"]).describe())


if __name__ == "__main__":
    main()
