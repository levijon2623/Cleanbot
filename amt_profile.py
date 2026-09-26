# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
amt_profile.py
==============

Daily volume-profile primitives shared by check_amt.py and
directional_flow_backtester.py (kept separate to avoid a circular import).

  profile_from_bars(bars, binw)         -> {poc, vah, val, ib_hi, ib_lo, open, close, high, low}
  build_profiles(ticker, ...)           -> per-day DataFrame, cached to _amt_cache/
  amt_open_map(ticker, ...)             -> {trading_day -> 'below_va'|'inside_va'|'above_va'}
                                           from the PRIOR session's value area
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

HIST = "historical"
CACHE = "_amt_cache"
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60
DEFAULT_BIN_PCT = 0.0005
DEFAULT_IB_MINS = 60
DEFAULT_VA_FRAC = 0.70


def _load_1m(tk: str) -> pd.DataFrame:
    import polars as pl
    df = pl.read_parquet(f"{HIST}/{tk}.parquet").to_pandas()
    df.columns = [c.lower() for c in df.columns]
    et = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    mod = et.dt.hour * 60 + et.dt.minute
    m = (mod >= RTH_LO) & (mod <= RTH_HI)
    out = pd.DataFrame({"dt": et[m].values, "mod": mod[m].values,
                        "o": df["open"][m].astype(float).values,
                        "h": df["high"][m].astype(float).values,
                        "l": df["low"][m].astype(float).values,
                        "c": df["close"][m].astype(float).values,
                        "v": df["volume"][m].astype(float).values})
    out["dt"] = pd.to_datetime(out["dt"])
    out["date"] = out["dt"].dt.date
    return out.sort_values("dt").reset_index(drop=True)


def profile_from_bars(bars, binw: float, ib_mins: int = DEFAULT_IB_MINS, va_frac: float = DEFAULT_VA_FRAC):
    """bars: iterable of dicts/rows with mod, o, h, l, c, v (mod = minute-of-day ET).
    Returns the profile dict or None."""
    b = [(float(x["mod"]), float(x["o"]), float(x["h"]), float(x["l"]), float(x["c"]), float(x["v"]))
         for x in bars]
    if len(b) < 10:
        return None
    lo = min(x[3] for x in b)
    hi = max(x[2] for x in b)
    if not np.isfinite(lo) or hi <= lo or binw <= 0:
        return None
    edges = np.arange(np.floor(lo / binw) * binw, hi + binw, binw)
    if len(edges) < 3:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    n = len(centers)
    vol = np.zeros(n)
    for mod, o, h, l, c, v in b:
        a = max(int(np.searchsorted(edges, l, "right")) - 1, 0)
        z = min(int(np.searchsorted(edges, h, "right")) - 1, n - 1)
        if z < a:
            continue
        vol[a:z + 1] += v / (z - a + 1)
    if vol.sum() <= 0:
        return None
    poc_i = int(np.argmax(vol))
    total = vol.sum()
    lo_i = hi_i = poc_i
    acc = vol[poc_i]
    while acc < va_frac * total and (lo_i > 0 or hi_i < n - 1):
        can_up, can_dn = hi_i < n - 1, lo_i > 0
        up = vol[hi_i + 1] if can_up else -1.0
        dn = vol[lo_i - 1] if can_dn else -1.0
        if can_up and (up >= dn or not can_dn):
            hi_i += 1; acc += vol[hi_i]
        elif can_dn:
            lo_i -= 1; acc += vol[lo_i]
        else:
            break
    ib = [x for x in b if x[0] < RTH_LO + ib_mins]
    return dict(poc=float(centers[poc_i]), vah=float(centers[hi_i]), val=float(centers[lo_i]),
                ib_hi=max(x[2] for x in ib) if ib else np.nan,
                ib_lo=min(x[3] for x in ib) if ib else np.nan,
                open=b[0][1], close=b[-1][4], high=hi, low=lo)


def build_profiles(tk: str, binw_pct: float = DEFAULT_BIN_PCT, ib_mins: int = DEFAULT_IB_MINS,
                   va_frac: float = DEFAULT_VA_FRAC, force: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{tk}_ib{ib_mins}.parquet")
    if os.path.exists(fp) and not force:
        return pd.read_parquet(fp)
    px = _load_1m(tk)
    binw = round(px["c"].median() * binw_pct, 2) or 0.01
    rows = []
    for d, g in px.groupby("date"):
        p = profile_from_bars(g.to_dict("records"), binw, ib_mins, va_frac)
        if p:
            p["date"] = pd.Timestamp(d)
            rows.append(p)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out.to_parquet(fp, index=False)
    return out


def amt_ok(spec, loc) -> bool:
    """Does open-location `loc` satisfy a rule's `amt_open` spec?
    spec: 'below_va'|'inside_va'|'above_va' (require exactly) |
          {'require': [...]} | {'exclude': [...]}.
    loc None (unknown) -> True (fail-open)."""
    if not spec or loc is None:
        return True
    if isinstance(spec, str):
        return loc == spec
    req, exc = spec.get("require"), spec.get("exclude")
    if req is not None:
        req = [req] if isinstance(req, str) else req
        if loc not in req:
            return False
    if exc is not None:
        exc = [exc] if isinstance(exc, str) else exc
        if loc in exc:
            return False
    return True


def classify_open(open_px: float, prev_vah: float, prev_val: float) -> str | None:
    if not (np.isfinite(prev_vah) and np.isfinite(prev_val)) or open_px <= 0:
        return None
    if open_px > prev_vah:
        return "above_va"
    if open_px < prev_val:
        return "below_va"
    return "inside_va"


def amt_open_map(tk: str, hist: str = HIST, **kw) -> dict:
    """{trading_day(date) -> 'below_va'|'inside_va'|'above_va'} from the PRIOR
    session's value area. Empty dict if no historical/{tk}.parquet."""
    if not os.path.exists(f"{hist}/{tk}.parquet"):
        return {}
    p = build_profiles(tk, **kw).sort_values("date").reset_index(drop=True)
    out = {}
    for i in range(1, len(p)):
        prev, cur = p.iloc[i - 1], p.iloc[i]
        loc = classify_open(cur["open"], prev["vah"], prev["val"])
        if loc:
            out[cur["date"].date()] = loc
    return out
