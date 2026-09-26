# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_signal_quality.py
=======================
How good is the ENTRY SIGNAL, with the option bracket taken out of the picture?

Everything measured so far bundles the trigger's directional call together with
theta, the +100% take-profit, the EOD flatten and the time budget. For an
ADVISORY signal -- one a human reads and exits on their own judgement -- only
the first part matters. So score the trigger on the UNDERLYING:

  hit      P(direction-adjusted move > 0) at +15 / +30 / +60 / +120m and to close
  drift    mean direction-adjusted move, in bp
  MFE/MAE  mean max-favourable vs max-adverse excursion, and their ratio
           ("edge ratio": >1 means the signal gives you room to work with, which
           is exactly what a discretionary exit needs). Random entries sit at ~1.

Gate variants, from the deployed spec outward -- the question being whether a
LOOSER signal set is still good enough to be worth a human's attention:

  deployed   the rule's own regime + min_flow_pct
  regime     regime only, p50 flow
  pXX        no regime, flow percentile only (50 / 65 / 80 / 90)

against a RANDOM-ENTRY null on the same ticker-days (random minute, random
direction), which calibrates what "no signal" looks like.

All aggregates are day-level (one obs per ticker-date-gate) so intraday
clustering cannot inflate anything.

Usage:
  python check_signal_quality.py
  python check_signal_quality.py --tickers SPY QQQ NVDA
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
CLOSE_MOD = 15 * 60 + 55
HORIZONS = (15, 30, 60, 120)


def _spot_by_day(tk):
    import polars as pl
    p = f"{HIST}/{tk}.parquet"
    if not os.path.exists(p):
        return {}
    d = pl.read_parquet(p, columns=["start_time", "close"]).to_pandas()
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["date"] = et.dt.date
    d["mod"] = et.dt.hour * 60 + et.dt.minute
    d = d[(d["mod"] >= 570) & (d["mod"] <= 960)].sort_values(["date", "mod"])
    return {dt: (g["mod"].to_numpy(), g["close"].to_numpy(float)) for dt, g in d.groupby("date")}


def _excursion(arr, m, sgn, horizon):
    """direction-adjusted (ret_at_h, MFE, MAE) over [m, m+horizon]."""
    mm, vv = arr
    i = int(np.searchsorted(mm, m, side="right")) - 1
    if i < 0:
        return None
    s0 = vv[i]
    if not np.isfinite(s0) or s0 <= 0:
        return None
    j = int(np.searchsorted(mm, m + horizon, side="right"))
    win = vv[i:j]
    if len(win) < 2:
        return None
    r = sgn * (win / s0 - 1.0)
    return float(r[-1]), float(np.max(r)), float(np.min(r))


def _agg(lbl, df, base=None):
    if len(df) < 30:
        return f"    {lbl:22} n={len(df):>5}  (thin)"
    out = f"    {lbl:22} n={len(df):>5}  "
    for H in HORIZONS:
        c = f"r{H}"
        v = df[c].dropna()
        if len(v) < 20:
            out += f"{H}m: --      "
            continue
        out += f"{H}m {v.mean()*1e4:>+5.1f}bp/{(v > 0).mean()*100:>4.1f}%  "
    mfe, mae = df["mfe60"].dropna(), df["mae60"].dropna()
    if len(mfe) > 20 and abs(mae.mean()) > 1e-9:
        out += f"| MFE {mfe.mean()*1e4:>5.1f} MAE {mae.mean()*1e4:>6.1f} ratio {abs(mfe.mean()/mae.mean()):>4.2f}"
    return out


def _isoos(lbl, df, col="r60"):
    v = df[col].dropna()
    if len(v) < 30:
        return f"      {lbl:20} (thin)"
    i = df[df.date < SPLIT][col].dropna()
    o = df[df.date >= SPLIT][col].dropna()
    return (f"      {lbl:20} n={len(v):>5}  all {v.mean()*1e4:>+5.1f}bp hit {(v>0).mean()*100:>4.1f}%  "
            f"IS {i.mean()*1e4 if len(i) else float('nan'):>+5.1f}bp/{(i>0).mean()*100 if len(i) else float('nan'):>4.1f}%  "
            f"OOS {o.mean()*1e4 if len(o) else float('nan'):>+5.1f}bp/{(o>0).mean()*100 if len(o) else float('nan'):>4.1f}%")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES

    rules = [r for r in RULES if r.get("enabled", True)]
    tickers = sorted({r["ticker"] for r in rules}) if not a.tickers else [t.upper() for t in a.tickers]
    rule_by_tk = {}
    for r in rules:
        rule_by_tk.setdefault(r["ticker"], []).append(r)

    rows, nullrows = [], []
    rng = np.random.default_rng(7)
    for tk in tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        spot = _spot_by_day(tk)
        if not spot:
            continue

        # which (regime, direction) this ticker's deployed rules use.
        # `regime` may be a LIST (OR-match, e.g. LULU ["HIVOL","UPTREND"]) -> tuple
        def _regt(rg):
            if rg is None:
                return ()
            return tuple(rg) if isinstance(rg, (list, tuple)) else (rg,)

        deployed = [(_regt(r.get("regime")), r["direction"], int(r.get("min_flow_pct") or 50))
                    for r in rule_by_tk.get(tk, [])]

        def _reg_ok(rgt, d):
            """day d satisfies ANY regime in rgt (empty tuple = no gate)."""
            if not rgt:
                return True
            for rg in rgt:
                if rg.endswith("_GEX"):
                    if gex.get(d) == rg.replace("_GEX", ""):
                        return True
                elif reg_src.get(rg, {}).get(d) == rg:
                    return True
            return False

        for t in trigs:
            thr = t.get("thr")
            if not thr:
                continue
            d, ts = t["date"], t["ts"]
            arr = spot.get(d)
            if arr is None:
                continue
            m = pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute
            if m > CLOSE_MOD - 15:
                continue
            sgn = 1.0 if t["dir"] == "CALL" else -1.0
            rec = dict(ticker=tk, date=d, mod=m, dir=t["dir"], hour=m // 60)
            ok = False
            for H in HORIZONS:
                e = _excursion(arr, m, sgn, H)
                if e:
                    rec[f"r{H}"] = e[0]
                    if H == 60:
                        rec["mfe60"], rec["mae60"] = e[1], e[2]
                    ok = True
                else:
                    rec[f"r{H}"] = np.nan
            ec = _excursion(arr, m, sgn, CLOSE_MOD - m)
            rec["rclose"] = ec[0] if ec else np.nan
            if not ok:
                continue
            # gate membership
            for P in (50, 65, 80, 90):
                rec[f"p{P}"] = bool(t["abs_flow"] >= thr.get(P, 1e99))
            regs = {rg for rgt, _dd, _p in deployed for rg in rgt}
            rec["in_regime"] = _reg_ok(tuple(regs), d) if regs else False
            rec["is_deployed"] = any(
                t["dir"] == dd and t["abs_flow"] >= thr.get(p, 1e99) and _reg_ok(rgt, d)
                for rgt, dd, p in deployed)
            rows.append(rec)

        # ---- random-entry null on the same ticker-days ----
        for d, arr in spot.items():
            mm = arr[0]
            if len(mm) < 130:
                continue
            for _ in range(3):
                m = int(rng.choice(mm[(mm >= 575) & (mm <= CLOSE_MOD - 130)]))
                sgn = float(rng.choice([1.0, -1.0]))
                rec = dict(ticker=tk, date=d, mod=m)
                for H in HORIZONS:
                    e = _excursion(arr, m, sgn, H)
                    rec[f"r{H}"] = e[0] if e else np.nan
                    if H == 60 and e:
                        rec["mfe60"], rec["mae60"] = e[1], e[2]
                nullrows.append(rec)

    R = pd.DataFrame(rows)
    NL = pd.DataFrame(nullrows)
    if R.empty:
        print("no triggers"); return

    def dedupe(df, keys):
        num = [c for c in df.columns if c.startswith(("r", "mfe", "mae"))]
        g = df.groupby(keys)[num].mean().reset_index()
        g["date"] = pd.to_datetime(g["date"]).dt.date
        return g

    print("=" * 126)
    print(f"  ENTRY-SIGNAL QUALITY on the UNDERLYING   {len(R)} triggers, {len(tickers)} tickers   "
          f"day-level   split {SPLIT}")
    print("=" * 126)
    print("  format:  <horizon> <mean drift bp>/<hit rate %>   |   MFE/MAE over +60m")

    print("\n  -- null: random entry, random direction, same ticker-days --")
    print(_agg("RANDOM", dedupe(NL, ["ticker", "date"])))

    print("\n  -- gate variants --")
    variants = [
        ("p50 (all triggers)", R[R["p50"]]),
        ("p65", R[R["p65"]]),
        ("p80", R[R["p80"]]),
        ("p90", R[R["p90"]]),
        ("regime + p50", R[R["in_regime"] & R["p50"]]),
        ("DEPLOYED spec", R[R["is_deployed"]]),
    ]
    for lbl, sub in variants:
        if sub.empty:
            continue
        print(_agg(lbl, dedupe(sub, ["ticker", "date"])))

    print("\n  -- IS/OOS stability at +60m --")
    for lbl, sub in variants:
        if sub.empty:
            continue
        print(_isoos(lbl, dedupe(sub, ["ticker", "date"])))
    print(_isoos("RANDOM null", dedupe(NL, ["ticker", "date"])))

    print("\n  -- by DIRECTION (deployed spec) --")
    for dd in ("CALL", "PUT"):
        sub = R[R["is_deployed"] & (R["dir"] == dd)]
        if not sub.empty:
            print(_isoos(dd, dedupe(sub, ["ticker", "date"])))

    print("\n  -- by TICKER (p80, +60m) : where is the signal actually directional? --")
    for tk, sub in R[R["p80"]].groupby("ticker"):
        g = dedupe(sub, ["ticker", "date"])
        v = g["r60"].dropna()
        if len(v) < 40:
            continue
        i = g[g.date < SPLIT]["r60"].dropna()
        o = g[g.date >= SPLIT]["r60"].dropna()
        flag = "  <-- both halves +" if (len(i) and len(o) and i.mean() > 0 and o.mean() > 0) else ""
        print(f"      {tk:6} n={len(v):>4}  {v.mean()*1e4:>+5.1f}bp hit {(v>0).mean()*100:>4.1f}%  "
              f"IS {i.mean()*1e4 if len(i) else float('nan'):>+5.1f}  "
              f"OOS {o.mean()*1e4 if len(o) else float('nan'):>+5.1f}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    a = ap.parse_args()
    run(a)
