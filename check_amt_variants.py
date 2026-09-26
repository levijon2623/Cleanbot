# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_amt_variants.py
=====================
The deployed `amt_open` gate classifies TODAY'S OPEN against YESTERDAY'S value
area (`amt_profile.classify_open`, called once per day in `amt_open_map`) and
then never looks again. It is a DAY-CONSTANT label fixed at 09:30.

But auction market theory is about acceptance and rejection AS THE SESSION
DEVELOPS. The current gate uses the auction's starting condition as a proxy for
the whole day's character. Four variants of "where is price, relative to value":

  open_prev    today's OPEN vs prior VA            <- THE DEPLOYED GATE (baseline)
  entry_prev   ENTRY price vs prior VA             <- has the label gone stale?
  entry_today  ENTRY price vs TODAY's developing VA (causal: profile built from
               09:30 to the entry minute only, never beyond)
  entry_ib     ENTRY price vs today's INITIAL BALANCE (first 60m hi/lo).
               `profile_from_bars` already computes ib_hi/ib_lo and the gate has
               never used them. Only defined after 10:30.
  migration    the PAIR (open_prev -> entry_prev). "Opened inside value, now
               above VAH" is acceptance migrating up -- a different state from
               "opened above and stayed above" (gap-and-go). This is the variant
               that actually captures what AMT is about; the others are snapshots.

STATISTICS -- the trap this test must not fall into:
`open_prev` is DAY-CONSTANT, which is why day-level aggregation has always been
valid for it. Every variant here except `open_prev` is a WITHIN-DAY feature, and
day-level aggregation of a within-day feature MANUFACTURES CORRELATION (the same
trap as VWAP position, relvol and hour-of-day). So the primary unit here is
TRADE level. Day counts are printed so clustering stays visible, and the
per-rule head-to-head also reports day level for continuity with the rest of the
book -- but the decision is made on trade level.

All P&L is SEQUENTIAL (one position per ticker, `_walk`) on REALISTIC fills
(enter ask, exit bid), scored with each rule's own deployed exit policy.

PRE-COMMITTED PASS CRITERIA (fixed before any output was viewed). A variant
replaces the deployed gate on a rule only if ALL of:
  V1  trade-level IS > 0 AND trade-level OOS > 0
  V2  trade-level OOS beats the DEPLOYED amt_open gate on that rule
  V3  >= 5 of 6 calendar slices populated, and >= 5 of them positive
  V4  OOS beats the p95 of a bootstrap null drawn from that rule's own
      UNGATED trigger population, matched on n
  V5  the same variant+direction shows up in the POOLED book-wide scan, not
      only on the one rule  (the cross-sectional check that killed hour-10)

V5 exists because this is a wide search: 4 variants x 3 location values x 3
gated rules. Without a pooled consistency requirement the winner is whichever
cell got lucky.

Usage:
  python check_amt_variants.py
  python check_amt_variants.py --rules "AVGO HIVOL PUT" "QQQ HIVOL CALL"
  python check_amt_variants.py --boot 4000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from amt_profile import build_profiles, amt_ok, classify_open, DEFAULT_BIN_PCT, DEFAULT_VA_FRAC
from check_config_walkforward import _slice_idx
from check_exit_walkforward import _walk, _eod_mod

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
IB_MINS = 60
LOCS = ("below_va", "inside_va", "above_va")


# ---------------------------------------------------------------- profiles
def _dev_profile_at(bins_vol, lo_bin, hi_bin, binw, va_frac=DEFAULT_VA_FRAC):
    """POC/VAH/VAL from a CAUSAL running histogram (bins seen so far only)."""
    n = hi_bin - lo_bin + 1
    if n < 3:
        return None
    vol = np.zeros(n)
    for b, v in bins_vol.items():
        vol[b - lo_bin] += v
    total = vol.sum()
    if total <= 0:
        return None
    poc_i = int(np.argmax(vol))
    li = hi_i = poc_i
    acc = vol[poc_i]
    while acc < va_frac * total and (li > 0 or hi_i < n - 1):
        can_up, can_dn = hi_i < n - 1, li > 0
        up = vol[hi_i + 1] if can_up else -1.0
        dn = vol[li - 1] if can_dn else -1.0
        if can_up and (up >= dn or not can_dn):
            hi_i += 1; acc += vol[hi_i]
        elif can_dn:
            li -= 1; acc += vol[li]
        else:
            break
    c = lambda i: (lo_bin + i + 0.5) * binw
    return dict(poc=c(poc_i), vah=c(hi_i), val=c(li))


def _day_locations(tk, want):
    """{(date, mod) -> dict of location labels} for every requested (date, mod).

    Causal by construction: the developing profile at minute m only ever
    accumulates bars with mod <= m, and the bin grid is anchored at 0 (a global
    lattice), so nothing about the day's eventual range leaks backwards.
    """
    import polars as pl
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return {}
    df = pl.read_parquet(p, columns=["start_time", "open", "high", "low", "close", "volume"]).to_pandas()
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = et.dt.date
    df["mod"] = et.dt.hour * 60 + et.dt.minute
    df = df[(df["mod"] >= RTH_LO) & (df["mod"] <= RTH_HI)].copy()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values(["date", "mod"])
    binw = round(float(df["close"].median()) * DEFAULT_BIN_PCT, 2) or 0.01

    # prior-session VA from the shared cache (same source the live gate uses)
    prof = build_profiles(tk).sort_values("date").reset_index(drop=True)
    prev_va = {}
    for i in range(1, len(prof)):
        prev_va[prof.iloc[i]["date"].date()] = (float(prof.iloc[i - 1]["vah"]),
                                                float(prof.iloc[i - 1]["val"]))

    by_day = {}
    for d, g in df.groupby("date"):
        by_day[d] = g

    out = {}
    for d, mods in want.items():
        g = by_day.get(d)
        if g is None:
            continue
        pv = prev_va.get(d)
        day_open = float(g.iloc[0]["open"])
        open_loc = classify_open(day_open, pv[0], pv[1]) if pv else None

        mods = sorted(set(mods))
        gm = g["mod"].to_numpy(int)
        gl = g["low"].to_numpy(float); gh = g["high"].to_numpy(float)
        gc = g["close"].to_numpy(float); gv = g["volume"].to_numpy(float)

        bins_vol, lo_bin, hi_bin = {}, None, None
        ib_hi = ib_lo = np.nan
        j = 0
        for m in mods:
            # accumulate every bar up to and including minute m
            while j < len(gm) and gm[j] <= m:
                a = int(np.floor(gl[j] / binw)); z = int(np.floor(gh[j] / binw))
                if z < a:
                    a, z = z, a
                share = gv[j] / (z - a + 1)
                for b in range(a, z + 1):
                    bins_vol[b] = bins_vol.get(b, 0.0) + share
                lo_bin = a if lo_bin is None else min(lo_bin, a)
                hi_bin = z if hi_bin is None else max(hi_bin, z)
                if gm[j] < RTH_LO + IB_MINS:
                    ib_hi = gh[j] if not np.isfinite(ib_hi) else max(ib_hi, gh[j])
                    ib_lo = gl[j] if not np.isfinite(ib_lo) else min(ib_lo, gl[j])
                j += 1
            if lo_bin is None:
                continue
            # spot at (or just before) the entry minute
            k = int(np.searchsorted(gm, m, side="right")) - 1
            if k < 0:
                continue
            spot = float(gc[k])

            rec = dict(open_prev=open_loc, entry_prev=None, entry_today=None, entry_ib=None)
            if pv:
                rec["entry_prev"] = classify_open(spot, pv[0], pv[1])
            dev = _dev_profile_at(bins_vol, lo_bin, hi_bin, binw)
            if dev:
                rec["entry_today"] = classify_open(spot, dev["vah"], dev["val"])
            if m >= RTH_LO + IB_MINS and np.isfinite(ib_hi) and np.isfinite(ib_lo):
                rec["entry_ib"] = ("above_va" if spot > ib_hi
                                   else "below_va" if spot < ib_lo else "inside_va")
            out[(d, m)] = rec
    return out


# ---------------------------------------------------------------- stats
def _stat(lbl, pnls, base_oos=None):
    if len(pnls) < 12:
        return f"    {lbl:34} n={len(pnls):>5}   (thin)"
    v = np.array([p for _, p in pnls], float)
    ds = [d for d, _ in pnls]
    i = np.array([p for d, p in pnls if d < SPLIT], float)
    o = np.array([p for d, p in pnls if d >= SPLIT], float)
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    pop = [np.mean(b) for b in sl if len(b) >= 3]
    nposs = sum(1 for x in pop if x > 0)
    nd = len(set(ds))
    delta = ""
    if base_oos is not None and len(o):
        delta = f" {(o.mean() - base_oos) * 100:>+6.1f}pp"
    return (f"    {lbl:34} n={len(v):>5} d={nd:>4} "
            f"IS {i.mean()*100 if len(i) else float('nan'):>+7.1f}% "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+7.1f}%"
            f"{delta}  win {(v>0).mean():>4.2f}  sl {nposs}/{len(pop)}")


def _oos(pnls):
    o = [p for d, p in pnls if d >= SPLIT]
    return float(np.mean(o)) if o else float("nan")


# ---------------------------------------------------------------- build
def _rule_candidates(D, r, TRAIL_PCT):
    """Every trigger for a rule WITH THE AMT GATE REMOVED, so the variants below
    can each impose their own location rule on the same underlying population.
    Builder and exit come from sim_core (one simulator, cushion modelled)."""
    import sim_core
    spec = {k: v for k, v in r.items() if k != "amt_open"}
    cand = sim_core.build_candidates(D, spec)
    if not cand:
        return None, None
    return cand, sim_core.policy_for(r, TRAIL_PCT)


def run(a):
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    rules = [r for r in RULES if r.get("enabled", True)]
    if a.rules:
        rules = [r for r in rules if r["name"] in a.rules]

    store = {}
    for r in rules:
        cand, pol = _rule_candidates(D, r, TRAIL_PCT)
        if not cand:
            print(f"  {r['name']}: no candidates"); continue
        want = {}
        for d, m, _ in cand:
            want.setdefault(d, []).append(m)
        locs = _day_locations(r["ticker"], want)
        store[r["name"]] = dict(rule=r, cand=cand, pol=pol, locs=locs,
                                em=_eod_mod(r))
        print(f"  {r['name']:26} {len(cand):>5} ungated candidates, "
              f"{len(locs):>5} located")

    if not store:
        print("nothing to test"); return

    VARIANTS = ["open_prev", "entry_prev", "entry_today", "entry_ib"]

    # ---------------- 1. POOLED book-wide scan ----------------------------
    print("\n" + "=" * 118)
    print("  1. POOLED SCAN -- is entry location informative ACROSS the book?")
    print(f"     ({len(store)} rule(s): {', '.join(store)} -- amt gate REMOVED from all, trade level)")
    print("=" * 118)
    pooled_all = []
    for nm, s in store.items():
        pooled_all += _walk(s["cand"], s["pol"], s["em"], realistic=True)
    base = _oos(pooled_all)
    print(_stat("BASELINE (no amt gate at all)", pooled_all))
    for var in VARIANTS:
        print(f"    -- {var} --")
        for loc in LOCS:
            sub = []
            for nm, s in store.items():
                keep = [c for c in s["cand"] if (s["locs"].get((c[0], c[1])) or {}).get(var) == loc]
                if keep:
                    sub += _walk(keep, s["pol"], s["em"], realistic=True)
            print(_stat(f"      {loc}", sub, base))

    # ---------------- 2. MIGRATION matrix ---------------------------------
    print("\n" + "=" * 118)
    print("  2. MIGRATION -- open location -> entry location (vs PRIOR value area)")
    print("     the state AMT actually cares about: did value get accepted higher/lower?")
    print("=" * 118)
    for o in LOCS:
        for e in LOCS:
            sub = []
            for nm, s in store.items():
                keep = [c for c in s["cand"]
                        if (s["locs"].get((c[0], c[1])) or {}).get("open_prev") == o
                        and (s["locs"].get((c[0], c[1])) or {}).get("entry_prev") == e]
                if keep:
                    sub += _walk(keep, s["pol"], s["em"], realistic=True)
            tag = f"{o:9} -> {e:9}" + ("   (held)" if o == e else "   (MIGRATED)")
            print(_stat(tag, sub, base))

    # ---------------- 3. per-rule head-to-head ----------------------------
    print("\n" + "=" * 118)
    print("  3. PER-RULE -- deployed amt_open gate vs each variant")
    print("     V1 IS>0 & OOS>0 | V2 beats deployed OOS | V3 >=5/6 slices +ve | V4 > bootstrap p95")
    print("=" * 118)
    rng = np.random.default_rng(0)
    for nm, s in store.items():
        r = s["rule"]
        gate = r.get("amt_open")
        print(f"\n  {nm}   deployed amt_open = {gate}")
        ungated = _walk(s["cand"], s["pol"], s["em"], realistic=True)
        print(_stat("UNGATED (gate removed)", ungated))
        dep = None
        if gate:
            keep = [c for c in s["cand"]
                    if amt_ok(gate, (s["locs"].get((c[0], c[1])) or {}).get("open_prev"))]
            dep = _walk(keep, s["pol"], s["em"], realistic=True)
            print(_stat("DEPLOYED gate (open_prev)", dep, _oos(ungated)))
        dep_oos = _oos(dep) if dep else _oos(ungated)
        # bootstrap null from the rule's own ungated OOS pool
        u_oos = np.array([p for d, p in ungated if d >= SPLIT], float)
        for var in VARIANTS:
            for loc in LOCS:
                keep = [c for c in s["cand"] if (s["locs"].get((c[0], c[1])) or {}).get(var) == loc]
                if len(keep) < 12:
                    continue
                pn = _walk(keep, s["pol"], s["em"], realistic=True)
                if len(pn) < 12:
                    continue
                o = np.array([p for d, p in pn if d >= SPLIT], float)
                p95 = float("nan")
                if len(o) >= 6 and len(u_oos) > len(o) + 3:
                    draws = np.array([rng.choice(u_oos, size=len(o), replace=False).mean()
                                      for _ in range(a.boot)])
                    p95 = float(np.percentile(draws, 95))
                i = np.array([p for d, p in pn if d < SPLIT], float)
                sl = [[] for _ in range(6)]
                for d, p in pn:
                    k = _slice_idx(d)
                    if k is not None:
                        sl[k].append(p)
                pop = [np.mean(b) for b in sl if len(b) >= 3]
                v1 = len(i) > 0 and i.mean() > 0 and len(o) > 0 and o.mean() > 0
                v2 = len(o) > 0 and o.mean() > dep_oos
                v3 = len(pop) >= 5 and sum(1 for x in pop if x > 0) >= 5
                v4 = np.isfinite(p95) and len(o) > 0 and o.mean() > p95
                flags = "".join("Y" if c else "." for c in (v1, v2, v3, v4))
                mark = "  ** V1-V4 PASS" if all((v1, v2, v3, v4)) else ""
                print(_stat(f"  {var}={loc}", pn, dep_oos) +
                      f"  p95 {p95*100 if np.isfinite(p95) else float('nan'):>+6.1f}%  {flags}{mark}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=3000)
    run(ap.parse_args())
