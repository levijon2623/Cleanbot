# /// script
# requires-python = ">=3.11"
# dependencies = ["databento", "polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
build_mbo_features.py
=====================
Reconstructs the CME order book from MBO and emits FIVE order-level features per
RTH minute, for every downloaded day. Output: `_mbo_cache/{SYM}.parquet`.

THE FIVE FEATURES, and why each needs MBO specifically
------------------------------------------------------
  hhi            Herfindahl index of individual resting order sizes AT THE INSIDE
                 QUOTE. L2 shows "1,000 contracts at the bid" and cannot tell one
                 institutional wall from a hundred fragmented retail orders. High
                 HHI = a single committed actor defending the level.
  absorb         share of FILL volume that executes BEYOND THE DISPLAYED SIZE OF
                 THE ORDER IT HITS. CME publishes an iceberg's displayed quantity
                 and refreshes it as the hidden remainder executes, so the order
                 id stays KNOWN; the tell is an execution larger than the size we
                 had showing for it, and the excess was never on the book.
                 (The ORIGINAL definition -- fills against order ids never seen
                 added -- measured ~0 and was replaced; see the note at the A_FIL
                 branch. This docstring described that superseded version until
                 2026-09-15, which is why `absorb` was twice written off as a dud
                 feature. It is not: 27.9% of RTY minutes are non-zero, nonzero
                 median 0.0085, max 0.52.)
                 COMPANION, not a substitute: build_vap.py adds the LEVEL-based
                 view (executed volume vs the most ever displayed at that price).
                 That one catches hidden size arriving as fresh order ids, but it
                 also fires on honest replenishment, so it is turnover -- not an
                 iceberg claim on its own.
  tickchase      rate of price-changing MODIFYs per minute. Not total cancel
                 volume -- specifically market makers re-pegging to hold
                 top-of-book without crossing. Measures urgency, not churn.
  age_s          median age of orders resting at the touch. Committed capital
                 versus flickering quotes.
  qdepth         total displayed size at the touch.

Dropped as redundant during design: cancel:trade (subsumed by tickchase),
large-order arrival rate (subsumed by hhi), iceberg-via-size-increase (subsumed
by absorb, and more robust since it does not assume refills surface as MODIFY).
DEFERRED, not rejected: "anchor order abandonment" -- the best mechanism of the
set, but it carries four tunable definitions (which order is the anchor, what
counts as a sweep, how wide the window) and with only 98 trades a feature with
four knobs will find something whether or not anything is there.

TWO DATA TRAPS THIS HANDLES
---------------------------
1. `ts_event` is HISTORICAL during the opening snapshot -- a resting order
   carries the time it was ORIGINALLY placed (a 2024-12-20 file opens with
   records stamped 2024-12-15). Using it as a clock would put the whole snapshot
   in the wrong minute. **Time is kept with `ts_recv` (monotonic); `ts_event` is
   used ONLY to compute order age**, which is exactly what it is good for.
2. `price == INT64_MAX` is DBN's UNDEF_PRICE sentinel (it coincides with the
   `N` action). Filtered, or it poisons every price statistic.

MEMORY: one day at a time via `to_ndarray()` (~400 MB, 7.2M records), walked in
a tight loop over the structured array and released before the next. Never hold
two days. Never `.to_df()` an MBO day on this box.

Usage:
  python build_mbo_features.py                 # all symbols, all days
  python build_mbo_features.py --symbols RTY_c_0 --max-days 3
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import time

import numpy as np

ROOT = os.path.join("lake", "mbo")
CACHE = "_mbo_cache"
UNDEF = np.iinfo(np.int64).max
NS = 1_000_000_000
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60          # ET minutes
A_ADD, A_CXL, A_MOD, A_CLR = b"A", b"C", b"M", b"R"
A_TRD, A_FIL = b"T", b"F"
S_BID, S_ASK = b"B", b"A"


def _et_minute(ts_ns: int) -> int:
    """UTC ns -> ET minute-of-day. CME RTH is 13:30-20:00 UTC in EDT,
    14:30-21:00 in EST; pandas handles the transition."""
    import pandas as pd
    t = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("America/New_York")
    return t.hour * 60 + t.minute, t.date()


def features_for_day(path, sample_every=1):
    """Walk one day; emit a feature row per RTH minute."""
    import databento as db
    import pandas as pd

    arr = db.DBNStore.from_file(path).to_ndarray()
    act = arr["action"]; sd = arr["side"]; px = arr["price"]
    sz = arr["size"]; oid = arr["order_id"]
    tev = arr["ts_event"]; trc = arr["ts_recv"]

    orders = {}                 # oid -> (side, price, size, ts_event_first)
    level = {}                  # (side, price) -> total displayed size
    rows = []

    # per-minute accumulators, reset on each minute boundary
    cur_min = None
    cur_date = None
    n_tickchase = 0
    fill_known = fill_hidden = 0
    n_mod_price = n_mod_size = 0
    # side-split accumulators -> the SIGNED variants. A fill on the BID side
    # means a resting buyer was hit (aggressive SELLING), so bid-side absorption
    # is bearish pressure; ask-side is bullish. Signs are applied at emit.
    hid_b = hid_a = 0
    tick_b = tick_a = 0

    def _touch_stats(now_ns):
        """HHI / age / depth at the inside quote. One pass over the book."""
        # `side` can also be b'N' (none) -- those must NOT be treated as asks,
        # or the best ask collapses to a meaningless price.
        best_b, best_a = -1, 1 << 62
        for s, p, _q, _t in orders.values():
            if s == S_BID:
                if p > best_b:
                    best_b = p
            elif s == S_ASK and p < best_a:
                best_a = p
        if best_b <= 0 or best_a >= (1 << 62):
            return None
        sizes_b, sizes_a, ages, ages_b, ages_a = [], [], [], [], []
        for s, p, q, t0 in orders.values():
            if s == S_BID and p == best_b:
                sizes_b.append(q); ages.append((now_ns - t0) / NS)
                ages_b.append((now_ns - t0) / NS)
            elif s == S_ASK and p == best_a:
                sizes_a.append(q); ages.append((now_ns - t0) / NS)
                ages_a.append((now_ns - t0) / NS)
        def hhi(v):
            tot = float(sum(v))
            return float(sum((x / tot) ** 2 for x in v)) if tot > 0 else np.nan
        qb, qa = float(sum(sizes_b)), float(sum(sizes_a))
        hb, ha = hhi(sizes_b), hhi(sizes_a)
        return dict(
            hhi=np.nanmean([hb, ha]),
            qdepth=qb + qa,
            age_s=float(np.median(ages)) if ages else np.nan,
            n_orders_touch=len(sizes_b) + len(sizes_a),
            spread_ticks=(best_a - best_b),
            # ---- price, needed for forward returns ----
            mid=(best_b + best_a) / 2.0 / 1e9,
            # ---- SIGNED features. The unsigned ones above are magnitudes and
            # cannot predict DIRECTION; these can. `imb` (queue imbalance) is
            # also the POSITIVE CONTROL: it is one of the most robustly
            # documented short-horizon microstructure signals there is, so if
            # the reconstruction cannot reproduce it, the book is wrong and
            # every other null here is uninterpretable.
            imb=((qb - qa) / (qb + qa)) if (qb + qa) > 0 else np.nan,
            hhi_diff=(hb - ha) if (np.isfinite(hb) and np.isfinite(ha)) else np.nan,
            age_diff=((float(np.median(ages_b)) if ages_b else np.nan) -
                      (float(np.median(ages_a)) if ages_a else np.nan)),
        )

    n = len(arr)
    for i in range(n):
        a = act[i]
        o = oid[i]
        # NOTE: `size` is uint32 and `price` int64 in DBN. Every arithmetic use
        # is cast to Python int -- numpy uint32 subtraction WRAPS on underflow
        # (a partial fill would produce ~4 billion instead of a negative), which
        # silently corrupts level totals.
        if a == A_ADD:
            p, q = int(px[i]), int(sz[i])
            if p != UNDEF:
                orders[o] = (sd[i], p, q, int(tev[i]))
                k = (sd[i], p)
                level[k] = level.get(k, 0) + q
        elif a == A_CXL:
            prev = orders.pop(o, None)
            if prev is not None:
                k = (prev[0], prev[1])
                level[k] = level.get(k, 0) - prev[2]
                if level[k] <= 0:
                    level.pop(k, None)
        elif a == A_MOD:
            prev = orders.pop(o, None)
            p, q = int(px[i]), int(sz[i])
            if prev is not None:
                k = (prev[0], prev[1])
                level[k] = level.get(k, 0) - prev[2]
                if level[k] <= 0:
                    level.pop(k, None)
                if prev[1] != p:
                    n_mod_price += 1        # a re-peg: TICK CHASING
                    n_tickchase += 1
                    if prev[0] == S_BID:
                        tick_b += 1
                    elif prev[0] == S_ASK:
                        tick_a += 1
                else:
                    n_mod_size += 1
            if p != UNDEF:
                # a re-priced order loses queue priority, so its age restarts;
                # a size-only change keeps the original resting time.
                t0 = int(tev[i]) if (prev is None or prev[1] != p) else prev[3]
                orders[o] = (sd[i], p, q, t0)
                k = (sd[i], p)
                level[k] = level.get(k, 0) + q
        elif a == A_FIL:
            prev = orders.get(o)
            f = int(sz[i])
            if prev is None:
                # an id we never saw added at all
                fill_hidden += f
            else:
                # HIDDEN LIQUIDITY, corrected definition. CME publishes an
                # iceberg's DISPLAYED quantity and refreshes it as the hidden
                # remainder executes, so the order id stays KNOWN -- an
                # unknown-id test finds almost nothing (measured: absorb ~0).
                # The real tell is an execution LARGER than the size we had
                # displayed for that order: the excess was never on the book.
                d = int(prev[2])
                if f > d:
                    fill_hidden += (f - d)
                    fill_known += d
                    if prev[0] == S_BID:
                        hid_b += (f - d)
                    elif prev[0] == S_ASK:
                        hid_a += (f - d)
                else:
                    fill_known += f
                rem = d - f
                k = (prev[0], prev[1])
                level[k] = level.get(k, 0) - min(f, d)
                if rem > 0:
                    orders[o] = (prev[0], prev[1], rem, prev[3])
                else:
                    orders.pop(o, None)
                if level.get(k, 0) <= 0:
                    level.pop(k, None)
        elif a == A_CLR:
            orders.clear(); level.clear()

        # ---- minute boundary, keyed on ts_recv (monotonic) ----
        # ts_recv is monotonic; ts_event is NOT (the snapshot carries each
        # order's original placement time). Sampled every 1024 records -- at
        # ~5k records/minute that cannot skip a minute, and the timestamp
        # conversion is far too slow to run per record.
        m, d = _et_minute(int(trc[i])) if (i % 1024 == 0) else (cur_min, cur_date)
        if cur_min is None:
            cur_min, cur_date = m, d
        elif m != cur_min:
            if RTH_LO <= cur_min <= RTH_HI and orders:
                st = _touch_stats(int(trc[i]))
                if st:
                    tf = fill_known + fill_hidden
                    tk_tot = tick_b + tick_a
                    hd_tot = hid_b + hid_a
                    st.update(
                        date=cur_date, mod=cur_min,
                        absorb=(fill_hidden / tf) if tf > 0 else 0.0,
                        tickchase=float(n_tickchase),
                        fill_vol=tf, mod_price=n_mod_price, mod_size=n_mod_size,
                        # SIGNED: ask-side minus bid-side, so POSITIVE = bullish
                        # (hidden size absorbed on the OFFER = buyers taking, and
                        # re-pegging concentrated on the offer = sellers chasing up)
                        absorb_sgn=((hid_a - hid_b) / hd_tot) if hd_tot > 0 else 0.0,
                        tick_sgn=((tick_a - tick_b) / tk_tot) if tk_tot > 0 else 0.0,
                    )
                    rows.append(st)
            cur_min, cur_date = m, d
            n_tickchase = fill_known = fill_hidden = n_mod_price = n_mod_size = 0
            hid_b = hid_a = tick_b = tick_a = 0

    del arr
    gc.collect()
    return rows


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
        print(f"\n=== {sym}: {len(files)} days ===")
        allrows, t0 = [], time.time()
        for j, f in enumerate(files, 1):
            try:
                r = features_for_day(f)
            except Exception as e:
                print(f"  ! {os.path.basename(os.path.dirname(f))}: {type(e).__name__}: {e}")
                continue
            allrows += r
            el = time.time() - t0
            print(f"  {j:>3}/{len(files)} {os.path.basename(os.path.dirname(f)):18} "
                  f"{len(r):>4} min-rows   {el:>6.0f}s  eta {el/j*(len(files)-j):>5.0f}s",
                  flush=True)
            gc.collect()
        if not allrows:
            continue
        df = pl.DataFrame(allrows)
        p = os.path.join(CACHE, f"{sym}.parquet")
        df.write_parquet(p)
        print(f"  -> {p}  {df.height:,} rows, {df.width} cols")
        print(df.select(["hhi", "absorb", "tickchase", "age_s", "qdepth"]).describe())


if __name__ == "__main__":
    main()
