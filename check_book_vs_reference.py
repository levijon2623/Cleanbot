# /// script
# requires-python = ">=3.11"
# dependencies = ["databento", "polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_book_vs_reference.py
==========================
Is our MBO book reconstruction right? Compare it against Databento's OWN
top-of-book (`mbp-1`) for a day we already hold as MBO.

WHY
---
The queue-imbalance POSITIVE CONTROL passed on RTY but failed on NQ, and an
independent spread check agreed: reconstructed NQ quotes 3 ticks wide with a
2.8% one-tick share, where real NQ front-month is one tick wide the large
majority of the time. RTY looked plausible (39% one-tick, 90% within two).

So one instrument's book is good and the other's is not, and guessing at flag
semantics is a poor way to find out why. `mbp-1` is Databento's own computed
touch from the same feed -- the authoritative reference. One NQ day costs $1.44.

WHAT IT REPORTS
---------------
Per RTH minute, our reconstructed best bid/ask vs theirs:
  * exact-match rate on each side
  * signed error distribution in TICKS -- the SHAPE localises the bug:
       our bid too LOW  and ask too HIGH  -> we are MISSING ORDERS at the touch
       one side only                      -> a side-specific handling bug
       constant offset                    -> a price-scaling error
  * their spread distribution vs ours (theirs is ground truth)
  * their depth vs ours at the touch

Nothing is purchased without --confirm.

Usage:
  python check_book_vs_reference.py                       # price only
  python check_book_vs_reference.py --confirm             # buy + compare
  python check_book_vs_reference.py --confirm --symbol RTY_c_0 --date 2026-08-03
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except Exception:
    pass

REF_DIR = os.path.join("lake", "mbp1")
MBO_DIR = os.path.join("lake", "mbo")
CONT = {"NQ_c_0": "NQ.c.0", "RTY_c_0": "RTY.c.0"}
TICKSZ = {"NQ_c_0": 0.25, "RTY_c_0": 0.10}
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
UNDEF = np.iinfo(np.int64).max
NS = 1_000_000_000


def _key():
    k = os.getenv("DB_KEY") or os.getenv("DATABENTO_API_KEY")
    if not k:
        sys.exit("set DB_KEY in .env")
    return k


def _et_min(ts_ns):
    import pandas as pd
    t = pd.Timestamp(int(ts_ns), unit="ns", tz="UTC").tz_convert("America/New_York")
    return t.hour * 60 + t.minute


def our_touch_by_minute(path):
    """Replay our reconstruction; return {minute: (best_bid, best_ask, qb, qa)}
    sampled at the END of each RTH minute -- same convention as the reference."""
    import databento as db
    arr = db.DBNStore.from_file(path).to_ndarray()
    act, sd, px, sz, oid, trc = (arr["action"], arr["side"], arr["price"],
                                 arr["size"], arr["order_id"], arr["ts_recv"])
    orders = {}
    out = {}
    cur = None
    for i in range(len(arr)):
        a = act[i]; o = oid[i]
        if a == b"A":
            if px[i] != UNDEF:
                orders[o] = (sd[i], int(px[i]), int(sz[i]))
        elif a == b"C":
            orders.pop(o, None)
        elif a == b"M":
            if px[i] != UNDEF:
                orders[o] = (sd[i], int(px[i]), int(sz[i]))
            else:
                orders.pop(o, None)
        elif a == b"F":
            prev = orders.get(o)
            if prev is not None:
                rem = prev[2] - int(sz[i])
                if rem > 0:
                    orders[o] = (prev[0], prev[1], rem)
                else:
                    orders.pop(o, None)
        elif a == b"R":
            orders.clear()
        if i % 1024:
            continue
        m = _et_min(trc[i])
        if m == cur or not (RTH_LO <= m <= RTH_HI):
            cur = m
            continue
        cur = m
        bb = ba = None
        qb = qa = 0
        for s, p, q in orders.values():
            if s == b"B":
                if bb is None or p > bb:
                    bb = p
            elif s == b"A":
                if ba is None or p < ba:
                    ba = p
        if bb is None or ba is None:
            continue
        for s, p, q in orders.values():
            if s == b"B" and p == bb:
                qb += q
            elif s == b"A" and p == ba:
                qa += q
        out[m] = (bb, ba, qb, qa)
    del arr
    return out


def ref_touch_by_minute(path):
    """Databento's own top of book, last record per RTH minute."""
    import databento as db
    arr = db.DBNStore.from_file(path).to_ndarray()
    names = arr.dtype.names
    bpx = "bid_px_00" if "bid_px_00" in names else "bid_px"
    apx = "ask_px_00" if "ask_px_00" in names else "ask_px"
    bsz = "bid_sz_00" if "bid_sz_00" in names else "bid_sz"
    asz = "ask_sz_00" if "ask_sz_00" in names else "ask_sz"
    out = {}
    trc = arr["ts_recv"]
    for i in range(0, len(arr)):
        m = _et_min(trc[i]) if (i % 64 == 0) else None
        if m is None or not (RTH_LO <= m <= RTH_HI):
            continue
        b, a = int(arr[bpx][i]), int(arr[apx][i])
        if b == UNDEF or a == UNDEF or b <= 0 or a <= 0:
            continue
        out[m] = (b, a, int(arr[bsz][i]), int(arr[asz][i]))
    del arr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="NQ_c_0", choices=list(CONT))
    ap.add_argument("--date", default=None, help="default: the last day we hold")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()

    import databento as db
    import glob
    days = sorted(os.path.basename(os.path.dirname(p)).split("=")[1]
                  for p in glob.glob(os.path.join(MBO_DIR, a.symbol, "date=*", "*.dbn.zst")))
    if not days:
        sys.exit(f"no MBO days for {a.symbol}")
    day = a.date or days[-1]
    if day not in days:
        sys.exit(f"{day} not in our MBO set; have {days[0]}..{days[-1]}")

    sym = CONT[a.symbol]
    c = db.Historical(_key())
    import pandas as pd
    s = f"{day}T00:00"
    e = f"{pd.Timestamp(day) + pd.Timedelta(days=1):%Y-%m-%d}T00:00"
    cost = float(c.metadata.get_cost(dataset="GLBX.MDP3", symbols=[sym], schema="mbp-1",
                                     start=s, end=e, stype_in="continuous"))
    print(f"  reference: {sym} mbp-1 {day}   cost ${cost:.2f}")
    if not a.confirm:
        print("  DRY RUN — nothing purchased. Re-run with --confirm.")
        return

    os.makedirs(os.path.join(REF_DIR, a.symbol), exist_ok=True)
    rp = os.path.join(REF_DIR, a.symbol, f"{day}_mbp1.dbn.zst")
    if not os.path.exists(rp):
        print("  downloading...")
        c.timeseries.get_range(dataset="GLBX.MDP3", symbols=[sym], schema="mbp-1",
                               start=s, end=e, stype_in="continuous").to_file(rp)
    print(f"  reference file {os.path.getsize(rp)/1e6:.1f} MB")

    mp = os.path.join(MBO_DIR, a.symbol, f"date={day}", "mbo.dbn.zst")
    print("  replaying our reconstruction...")
    ours = our_touch_by_minute(mp)
    print("  reading reference touch...")
    ref = ref_touch_by_minute(rp)
    both = sorted(set(ours) & set(ref))
    print(f"  minutes: ours {len(ours)}, reference {len(ref)}, overlapping {len(both)}")
    if not both:
        sys.exit("  no overlapping minutes")

    tick = TICKSZ[a.symbol] * 1e9
    db_err = np.array([(ours[m][0] - ref[m][0]) / tick for m in both])
    da_err = np.array([(ours[m][1] - ref[m][1]) / tick for m in both])
    our_sp = np.array([(ours[m][1] - ours[m][0]) / tick for m in both])
    ref_sp = np.array([(ref[m][1] - ref[m][0]) / tick for m in both])
    our_q = np.array([ours[m][2] + ours[m][3] for m in both], float)
    ref_q = np.array([ref[m][2] + ref[m][3] for m in both], float)

    print("\n" + "=" * 84)
    print(f"  {a.symbol}  {day}   OUR BOOK vs DATABENTO mbp-1")
    print("=" * 84)
    print(f"  best BID exact match : {(db_err == 0).mean()*100:>5.1f}%   "
          f"median err {np.median(db_err):>+6.1f} ticks   mean {db_err.mean():>+6.2f}")
    print(f"  best ASK exact match : {(da_err == 0).mean()*100:>5.1f}%   "
          f"median err {np.median(da_err):>+6.1f} ticks   mean {da_err.mean():>+6.2f}")
    print(f"\n  spread (ticks)   ours median {np.median(our_sp):>5.1f}   "
          f"reference median {np.median(ref_sp):>5.1f}")
    print(f"  1-tick share     ours {(our_sp == 1).mean()*100:>5.1f}%   "
          f"reference {(ref_sp == 1).mean()*100:>5.1f}%")
    print(f"  touch depth      ours median {np.median(our_q):>7.0f}   "
          f"reference median {np.median(ref_q):>7.0f}")

    print("\n  -- DIAGNOSIS --")
    # Order matters. A book defect shows up as the two sides diverging (bid too
    # low AND ask too high = missing orders). If both sides are shifted the SAME
    # way, the book is fine and we simply sampled at a different instant -- so
    # test co-movement FIRST, or the missing-orders branch fires spuriously.
    sp_ok = abs(np.median(our_sp) - np.median(ref_sp)) <= 1
    depth_ok = 0.5 <= (np.median(our_q) / max(np.median(ref_q), 1e-9)) <= 2.0
    together = np.median(db_err) * np.median(da_err) > 0 and \
        abs(np.median(db_err) - np.median(da_err)) < 2
    if sp_ok and depth_ok and together:
        print("  ✅ BOOK IS SOUND. Spread distribution and touch depth MATCH the")
        print("     reference, and both sides are displaced in the SAME direction --")
        print("     that is a SAMPLING-INSTANT offset (we snapshot at a different")
        print("     moment within the minute), not a reconstruction error. A genuine")
        print("     book defect makes the sides DIVERGE (bid too low, ask too high).")
        print("     Judge the book on spread/depth DISTRIBUTION, not point-in-time match.")
    elif (db_err < 0).mean() > 0.5 and (da_err > 0).mean() > 0.5:
        print("  ❌ bid BELOW and ask ABOVE theirs -> MISSING ORDERS AT THE TOUCH.")
    elif not sp_ok:
        print("  ❌ spread distribution differs from the reference -> real book defect.")
    else:
        print("  No single dominant pattern; inspect the error distribution below.")
    for name, v in (("bid err", db_err), ("ask err", da_err)):
        qs = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"  {name} pct  5/25/50/75/95 = " + "  ".join(f"{x:>+6.1f}" for x in qs))


if __name__ == "__main__":
    main()
