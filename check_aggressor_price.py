# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
check_aggressor_price.py
========================
PRICE EACH SIDE AT ITS OWN SIDE OF THE BOOK, NOT AT A BLENDED VWAP.

THE FLAW THIS ADDRESSES
    build_flow_1m computes, per contract-minute:

        call:  +(ask_volume - bid_volume) * vwap * 100
        put:   the negative

    The VOLUME is already aggressor-filtered -- mid_volume and no_side_volume
    are excluded, which is the "mid-market exclusion" done right. But the PRICE
    is `vwap`, a volume-weighted average across EVERY print in that minute,
    including the mid prints whose volume was deliberately thrown away. So an
    aggressive lift is valued at a price partly set by passive mid executions.

    This reprices it:

        nf_agg = +(ask_volume * ask_close - bid_volume * bid_close) * 100

    Each side valued at its own side of the NBBO. On a wide 0DTE contract the
    half-spread is a material fraction of premium, so the two can disagree by
    more than rounding.

🚨 WHAT THIS DOES *NOT* FIX, STATED PLAINLY
    METHODOLOGY 7 measured three false assumptions behind cum_flow. This
    addresses only the pricing of the aggressor volume. It does NOT fix:
      - multi-leg legs counted as directional (8.4% of volume, 683% of the net
        signed quantity) -- needs condition codes from the BRONZE tape
      - opening vs closing (sign agreement with dOI is 49.4%, a coin flip)
    So a null here would mean "the pricing detail does not matter", NOT "the
    smarter flow idea is dead". The sweep-filtered version is a separate test
    and needs a re-backfill.

🚨 ask_close/bid_close ARE END-OF-MINUTE NBBO, not trade prices
    So this is also an approximation -- a better one, not a correct one. It
    cannot be otherwise without the trade-level tape.

PRE-COMMITTED CRITERIA -- with a magnitude floor
    A1  the repriced trigger beats the deployed one on TOTAL ROE
    A2  and on ROE PER TRADE by >= 5.0
    A3  IS and OOS agree in sign
    A4  >= 6 of 9 rules improve

RESULT -- 2026-09-22. NULL. The pricing detail does not matter.

    The two cumulative series correlate at 0.9990, median |difference| 4.0% of
    the level -- so the answer was visible before scoring ran.

        variant               n     total   per trade   IS/tr   OOS/tr
        deployed (vwap)     380    +4,726       +12.4    +7.1    +17.8
        aggressor-priced    384    +4,640       +12.1    +8.0    +16.1

    A1 total FAIL (-86) · A2 per-trade FAIL (-0.4) · A3 IS +0.9 / OOS -1.7,
    signs disagree FAIL · A4 5/9 rules FAIL.

    CAVEAT ON THE LEVELS: both arms were scored from a freshly built cache
    (4.27M rows, one lake pass), not the canonical FLOW_CACHE, so the deployed
    figure reads +4,726/380 here against +3,785/392 elsewhere. The A/B is
    internally valid -- same cache, same rules, one variable -- but do not
    compare these absolute numbers with other studies.

    🚨 WHAT THIS DOES NOT SETTLE. Only the PRICING of aggressor volume was
    changed. The two larger distortions measured in METHODOLOGY 7 are untouched:
    multi-leg legs counted as directional (8.4% of volume, 683% of the net
    signed quantity) and opening-vs-closing (sign agreement with dOI 49.4%, a
    coin flip). Both need condition codes from the BRONZE tape, which the 30s
    backfill discarded. A sweep-filtered flow remains untested and would need a
    re-backfill of the 151 days.

Usage:
  python check_aggressor_price.py --build      # one lake pass, cached
  python check_aggressor_price.py              # score it
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

import sim_core

CACHE = "_aggprice_cache/flow_agg.parquet"
SPLIT = pd.Timestamp("2025-08-21").date()


def _naive_et(df):
    """minute_et -> tz-NAIVE ET, matching what _flow_for produces.

    🚨 THE SILVER TAPE STORES minute_et TZ-AWARE America/New_York, while the
    flow frame, the trigger list and the option bars are all tz-NAIVE ET.
    Leaving it aware makes build_candidates' `day["minute_et"] <= ts` raise
    "Invalid comparison between dtype=datetime64[us] and Timestamp" -- and if it
    had silently coerced instead, every trigger would have been shifted 4-5
    hours and matched the wrong contract. Third time this trap has appeared
    (check_tv_proxy, export_flow_tape, here).
    """
    mt = pd.to_datetime(df["minute_et"])
    if getattr(mt.dtype, "tz", None) is not None:
        mt = mt.dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["minute_et"] = mt.astype("datetime64[us]")
    return df


def build(lake_dir):
    """One pass over the silver tape. Mirrors build_flow_1m's shape exactly so
    the ONLY difference is how the aggressor volume is priced."""
    from directional_flow_backtester import WATCH
    parts = sorted(glob.glob(os.path.join(lake_dir, "date=*", "bars.parquet")))
    print(f"  {len(parts)} partitions")
    frames = []
    for i, p in enumerate(parts, 1):
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol").is_in(WATCH))
              .with_columns(
                  # each side at ITS OWN side of the book
                  (pl.when(pl.col("option_type") == "call").then(1.0)
                   .otherwise(-1.0)
                   * (pl.col("ask_volume") * pl.col("ask_close")
                      - pl.col("bid_volume") * pl.col("bid_close")) * 100
                   ).alias("nf_agg"),
                  (pl.when(pl.col("option_type") == "call").then(1.0)
                   .otherwise(-1.0)
                   * (pl.col("ask_volume") - pl.col("bid_volume"))
                   * pl.col("vwap") * 100).alias("nf_vwap"))
              .group_by(["underlying_symbol", "minute_et"])
              .agg([pl.col("nf_agg").sum(), pl.col("nf_vwap").sum()]))
        frames.append(lf.collect().to_pandas())
        if i % 50 == 0:
            print(f"    {i}/{len(parts)}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = _naive_et(df)
    df = df.sort_values(["underlying_symbol", "minute_et"])
    df["date"] = pd.to_datetime(df["minute_et"]).dt.date
    for c in ("nf_agg", "nf_vwap"):
        df[f"cum_{c[3:]}"] = (df.groupby(["underlying_symbol", "date"])[c]
                              .cumsum())
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"  wrote {CACHE}  {len(df):,} rows")
    return df


def triggers(df, tk, col):
    """Same EMA(5) crossover as directional_flow_backtester.triggers_for."""
    g = df[df["underlying_symbol"] == tk]
    out = []
    for d, gd in g.groupby("date"):
        gd = gd.sort_values("minute_et")
        cum = gd[col].to_numpy(float)
        if len(cum) < 6:
            continue
        ema = pd.Series(cum).ewm(span=5, adjust=False).mean().to_numpy()
        mt = gd["minute_et"].to_numpy()
        hrs = pd.to_datetime(gd["minute_et"]).dt.hour.to_numpy()
        for i in range(1, len(cum)):
            bull = cum[i - 1] <= ema[i - 1] and cum[i] > ema[i]
            bear = cum[i - 1] >= ema[i - 1] and cum[i] < ema[i]
            if bull or bear:
                out.append({"date": d, "ts": mt[i], "hour": int(hrs[i]),
                            "dir": "CALL" if bull else "PUT",
                            "abs_flow": abs(float(cum[i]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--lake", default="lake/silver/option-contracts-1m")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fill", default="botcap")
    a = ap.parse_args()

    if a.build or not os.path.exists(CACHE):
        df = build(a.lake)
    else:
        df = _naive_et(pd.read_parquet(CACHE))   # cache may predate the fix
    print(f"  cache {len(df):,} rows, "
          f"{df['underlying_symbol'].nunique()} tickers\n")

    d = df.dropna(subset=["cum_agg", "cum_vwap"])
    if len(d):
        c = np.corrcoef(d["cum_agg"], d["cum_vwap"])[0, 1]
        rel = ((d["cum_agg"] - d["cum_vwap"]).abs()
               / d["cum_vwap"].abs().clip(lower=1)).median()
        print(f"  correlation of the two cumulative series: {c:.4f}")
        print(f"  median |difference| as a share of the level: {rel*100:.1f}%")
        print(f"  (if these are ~identical, the pricing detail cannot matter)\n")

    import directional_flow_backtester as D
    rows, per_rule = [], {}
    for rule in sim_core.research_rules(include_paper=a.paper):
        tk = rule["ticker"]
        pol, eod = sim_core.policy_for(rule), sim_core.eod_mod(rule)
        cap = sim_core.CUSHION_CAP.get(tk)
        per_rule[rule["name"]] = {}
        for lab, col in (("deployed (vwap)", "cum_vwap"),
                         ("aggressor-priced", "cum_agg")):
            tg = triggers(df, tk, col)
            if not tg:
                continue
            D.annotate_flow_pct(tg, rule.get("flow_window_days", 60))
            cand = sim_core.build_candidates(D, rule, trigs=tg)
            r = sim_core.walk(cand, pol, eod, fill=a.fill, cush_cap=cap)
            recs = [dict(rule=rule["name"], date=dd, pnl=p * 100)
                    for dd, p in r]
            per_rule[rule["name"]][lab] = recs
            rows += [dict(variant=lab, **x) for x in recs]
        print(f"    {rule['name']} done", flush=True)

    R = pd.DataFrame(rows)
    if R.empty:
        print("  nothing"); return
    R["half"] = np.where(R["date"] <= SPLIT, "IS", "OOS")
    R.to_parquet("_aggprice.parquet", index=False)

    print(f"\n{'='*92}")
    print(f"  THE BOOK UNDER EACH PRICING")
    print(f"{'='*92}")
    print(f"  {'variant':22} {'n':>6} {'total':>10} {'per trade':>11} "
          f"{'IS/tr':>9} {'OOS/tr':>9}")
    S = {}
    for nm in ("deployed (vwap)", "aggressor-priced"):
        q = R[R["variant"] == nm]
        if not len(q):
            continue
        S[nm] = dict(n=len(q), tot=q["pnl"].sum(), per=q["pnl"].mean(),
                     is_=q[q["half"] == "IS"]["pnl"].mean(),
                     oos=q[q["half"] == "OOS"]["pnl"].mean())
        print(f"  {nm:22} {S[nm]['n']:>6} {S[nm]['tot']:>+10.0f} "
              f"{S[nm]['per']:>+11.1f} {S[nm]['is_']:>+9.1f} "
              f"{S[nm]['oos']:>+9.1f}")

    print(f"\n  {'rule':24} {'deployed':>12} {'aggressor':>12} {'delta':>9}")
    nimp = 0
    for nm, v in per_rule.items():
        a_ = sum(x["pnl"] for x in v.get("deployed (vwap)", []))
        b_ = sum(x["pnl"] for x in v.get("aggressor-priced", []))
        nimp += int(b_ > a_)
        print(f"  {nm:24} {a_:>+12.0f} {b_:>+12.0f} {b_-a_:>+9.0f}")

    FLOOR = 5.0
    m = lambda ok: "PASS" if ok else "FAIL"
    if len(S) == 2:
        dv, ag = S["deployed (vwap)"], S["aggressor-priced"]
        print(f"\n{'='*92}\n  SCORECARD (floor {FLOOR:.0f} ROE/trade)\n{'='*92}")
        print(f"  A1  total improves        {ag['tot']:>+9.0f} vs "
              f"{dv['tot']:>+9.0f}   {m(ag['tot'] > dv['tot'])}")
        print(f"  A2  per trade +>={FLOOR:.0f}        "
              f"{ag['per']-dv['per']:>+9.1f}{'':>13}{m(ag['per']-dv['per'] >= FLOOR)}")
        print(f"  A3  IS/OOS agree in sign  IS {ag['is_']-dv['is_']:>+6.1f} / "
              f"OOS {ag['oos']-dv['oos']:>+6.1f}   "
              f"{m((ag['is_']-dv['is_'] > 0) == (ag['oos']-dv['oos'] > 0))}")
        print(f"  A4  rules improved        {nimp:>9}/{len(per_rule)}"
              f"{'':>13}{m(nimp >= 6)}")


if __name__ == "__main__":
    main()
