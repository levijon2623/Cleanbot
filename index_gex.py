"""
index_gex.py
============
The ETF's gamma heat with its INDEX's gamma added in: SPY + SPX, QQQ + NDX,
IWM + RUT. Pure math, no network -- the UW path (live_state) and the Webull
path (check_webull_blend.py) both feed it rows and get the same answer.

WHY
    On 2026-09-30 (UW, 16:15 ET snapshot) SPX carried 83-94% of the near-dated
    S&P gamma, NDX 17-54% of the Nasdaq-100's, RUT 38-68% of the Russell's.
    An ETF-only heat map leaves most of the S&P book off the chart, and SPX's
    big round strikes (7600 / 7650 / 7700) land on SPY 757.2 / 762.2 / 767.1
    -- two or three dollars from SPY's own round-number walls.

HOW
    1. Index strike K maps to ETF price K / ratio, where ratio = index / ETF
       price at the same moment (SPX/SPY ~10.04, NDX/QQQ ~41.1, RUT/IWM ~10.07).
    2. The mapped strike is fractional (762.16), so its gamma is SPLIT linearly
       between the two neighbouring ETF grid strikes (0.84 at 762, 0.16 at 763)
       -- total gamma is conserved and no strike is invented.
    3. Units need no rescaling: UW's call_gamma_oi / put_gamma_oi, and the
       Webull-side OI x gamma x 100 x S^2 x 1%, are both dollars of delta per
       1% move, so ETF and index gamma add directly.
    4. Levels are labelled by the NATIVE index strike that put them there
       ("SPX 7650"), because that is the number the index book actually trades.

🚨 THE RATIO IS THE WHOLE MAPPING, AND UW'S INDEX PRICES ARE ROUNDED.
    UW's /spot-exposures price for SPX and RUT moves in steps of 5, NDX in 10.
    One minute's ratio is therefore off by up to 0.03% (SPX) .. 0.09% (RUT);
    the MEDIAN over every matched minute of the session averages that out
    (2026-09-30: p10-p90 spread under $0.40 of ETF price, less than a strike).
    Never map with a single rounded print. Webull has no such problem -- the
    index forward comes from put-call parity on the index's own 0DTE options.
"""
from __future__ import annotations

import math
import statistics

INDEX_OF = {"SPY": "SPX", "QQQ": "NDX", "IWM": "RUT"}
# Only to pick which index strikes to fetch before the real ratio is known;
# never used to MAP. Off by a fraction of a percent at most.
SEED_RATIO = {"SPY": 10.04, "QQQ": 41.1, "IWM": 10.07}
GRID = 1.0          # the ETFs' near-dated strike step


def ratio_from_series(etf, idx, min_points=10):
    """Median index/ETF price ratio over matched minutes.

    etf, idx: {minute_key: price}. Returns (ratio, n) or (None, n) when there
    are fewer than min_points matched minutes -- a guessed ratio would move
    every mapped strike, so no ratio is better than a bad one.
    """
    rs = [idx[k] / etf[k] for k in set(etf) & set(idx) if etf[k] and idx[k]]
    if len(rs) < min_points:
        return None, len(rs)
    return statistics.median(rs), len(rs)


def map_rows(rows, ratio):
    """Index rows {k, cg, pg, ...} -> same rows plus m = k / ratio (ETF price)."""
    return [dict(r, m=r["k"] / ratio) for r in rows]


def split_to_grid(mapped, grid=GRID):
    """Mapped index rows -> {grid_strike: [call, put, {native: |gamma| share}]}.

    Linear split between the two neighbouring grid strikes; the third slot
    remembers which native strikes fed each grid strike so a level can be
    labelled by the index strike behind it.
    """
    out = {}
    for r in mapped:
        x = r["m"] / grid
        lo = math.floor(x)
        w = x - lo
        for kk, ww in ((lo, 1.0 - w), (lo + 1, w)):
            if ww <= 1e-12:
                continue
            k = round(kk * grid, 6)
            a = out.setdefault(k, [0.0, 0.0, {}])
            a[0] += r["cg"] * ww
            a[1] += r["pg"] * ww
            nat = r["k"]
            a[2][nat] = a[2].get(nat, 0.0) + (abs(r["cg"]) + abs(r["pg"])) * ww
    return out


def blend_rows(etf_rows, idx_rows, ratio, grid=GRID):
    """ETF rows + index rows split onto the ETF grid, as ETF-shaped rows
    ({e, k, cg, pg}) -- so anything that already builds a heat from ETF rows
    (live_state._expiry_heat, the 0-1DTE aggregation) works on the blend
    unchanged. Also returns the {(expiry, grid_k): {native: share}} provenance.
    """
    out = [dict(e=r["e"], k=r["k"], cg=r["cg"], pg=r["pg"]) for r in etf_rows]
    prov = {}
    by_exp = {}
    for r in idx_rows:
        by_exp.setdefault(r["e"], []).append(r)
    for e, rs in by_exp.items():
        for k, (c, p, nat) in split_to_grid(map_rows(rs, ratio), grid).items():
            out.append(dict(e=e, k=k, cg=c, pg=p))
            prov[(e, k)] = nat
    return out, prov


def heat_of(rows, expiries, lo, hi):
    """{strike: [call, put]} over the given expiries, strikes in [lo, hi]."""
    agg = {}
    for r in rows:
        if r["e"] in expiries and lo <= r["k"] <= hi:
            a = agg.setdefault(r["k"], [0.0, 0.0])
            a[0] += r["cg"]
            a[1] += r["pg"]
    return {k: v for k, v in agg.items() if v[0] or v[1]}


def native_label(idx, k, expiries, prov, etf_heat_at_k, idx_heat_at_k):
    """'SPX 7650' when the index dominates gamma at ETF strike k, else None."""
    if abs(idx_heat_at_k) <= abs(etf_heat_at_k):
        return None
    shares = {}
    for e in expiries:
        for nat, s in (prov.get((e, k)) or {}).items():
            shares[nat] = shares.get(nat, 0.0) + s
    if not shares:
        return None
    nat = max(shares, key=shares.get)
    return f"{idx} {nat:g}"


def index_levels(idx, idx_rows, expiries, ratio, lo, hi, n=3):
    """The index book's own biggest strikes in the band, NATIVE and mapped:
    [{native, at, net, kind}] -- kind 'call' / 'put' by the larger side.
    These are what the chart labels; they are not split, so 'at' is the
    exact ETF-price equivalent (762.16), not a grid strike."""
    agg = {}
    for r in idx_rows:
        if r["e"] not in expiries:
            continue
        m = r["k"] / ratio
        if not (lo <= m <= hi):
            continue
        a = agg.setdefault(r["k"], [0.0, 0.0])
        a[0] += r["cg"]
        a[1] += r["pg"]
    top = sorted(agg, key=lambda k: -abs(sum(agg[k])))[:n]
    return [dict(native=k, at=round(k / ratio, 2), net=round(sum(agg[k])),
                 kind="call" if abs(agg[k][0]) >= abs(agg[k][1]) else "put",
                 label=f"{idx} {k:g}")
            for k in sorted(top)]


def share(etf_heat, idx_grid_heat):
    """Index share of gross |net gamma| in the band (0..1), or None."""
    ge = sum(abs(v[0] + v[1]) for v in etf_heat.values())
    gi = sum(abs(v[0] + v[1]) for v in idx_grid_heat.values())
    return (gi / (ge + gi)) if (ge + gi) else None


# ------------------------------------------------------------- walls
#: A wall is a TIE when a strike more than one step away carries at least this
#: share of the wall's gamma. Set from the 2026-10-01 SPY+SPX disagreement:
#: when Webull and UW picked different put walls, Webull's pick held a median
#: 89% (min 73%) of UW's pick's gamma IN UW'S OWN DATA -- i.e. the two sources
#: were choosing between near-equals. 0.85 catches that case without flagging
#: every wall: a clear wall (runner-up well under 85%) stays a single strike.
WALL_TIE = 0.85


def walls_with_ties(heat, tie=WALL_TIE):
    """{strike: (call, put)} -> {call_wall, put_wall, peak} plus, for each, a
    `*_alt` runner-up strike when it is within `tie` of the winner and more
    than one strike step away (an adjacent strike is the same wall, not a
    rival). None when the heat is empty."""
    ks = sorted(heat)
    if not ks:
        return None
    steps = sorted(b - a for a, b in zip(ks, ks[1:]) if b > a)
    step = steps[len(steps) // 2] if steps else 1.0
    out = {}
    for key, val in (("call_wall", lambda k: heat[k][0]),
                     ("put_wall", lambda k: -heat[k][1]),
                     ("peak", lambda k: abs(heat[k][0] + heat[k][1]))):
        best = max(ks, key=val)
        out[key] = best
        rivals = [k for k in ks if abs(k - best) > step * 1.01 and val(k) > 0]
        if rivals and val(best) > 0:
            r = max(rivals, key=val)
            if val(r) >= tie * val(best):
                out[key + "_alt"] = r
    return out


# ------------------------------------------------------------- Webull side
def parity_forward(chain, near=None, n=5):
    """Index forward from put-call parity on ONE expiry's mids.

    chain: {strike: (call_mid, put_mid)}. F = K + C - P (0-1DTE: discounting
    is a few cents on SPX, below the strike grid by two orders). Uses the n
    strikes where |C - P| is smallest -- the ones nearest the money, where
    both legs are liquid -- and returns their median, or None.
    """
    pts = [(abs(c - p), k + c - p) for k, (c, p) in chain.items()
           if c is not None and p is not None and c > 0 and p > 0]
    if near is not None:
        pts = [x for x in pts if abs(x[1] - near) / near < 0.02]
    if len(pts) < 3:
        return None
    pts.sort()
    return statistics.median(f for _, f in pts[:n])
