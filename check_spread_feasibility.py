# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_spread_feasibility.py
===========================
PRICE THE STRUCTURES BEFORE BACKTESTING THEM.

The motivation for spreads is that `check_step1_redo` found the flow trigger's
timing IS real (87 of 150 unconditioned cells beat a timing-destroying null,
mean z +2.66) but only 2 of 150 make money -- theta and spread eat it. So the
question is whether a structure paying less decay keeps more of the signal.

BUT A SPREAD ADDS A SECOND BID/ASK TO CROSS, twice (entry and exit). If the
extra friction exceeds the decay saved, the structure loses before any
simulation is run, and no backtest is needed to know it. That is cheap to
measure straight off the chain, so it is measured first -- the same discipline
as check_databento_cost pricing the feed before buying it.

WHAT IS MEASURED, at real trigger minutes
    For each structure, the ROUND-TRIP FRICTIONAL COST as a percentage of the
    capital actually put at risk:

      naked long   cross the ATM contract's spread twice
      debit vert   long ATM + short next OTM strike, same expiry
      diagonal     long the nearest expiry >= --far-dte at ATM,
                   short the near-dated OTM
      credit vert  short near-ATM + long further OTM, same expiry
                   (risk = width - credit, NOT the credit)

    Friction is quoted against the right denominator for each structure --
    net debit for debit structures, max-loss for the credit vertical -- because
    a credit spread's "cost" is not its premium.

    Also reported: how often each structure is even QUOTABLE at the trigger
    minute (both legs with a live two-sided market), since an unquotable
    structure is not a strategy. The diagonal is the one at risk here: the bot
    trades 0/1DTE and a >=7DTE leg is a different liquidity regime.

NOTHING HERE IS A BACKTEST. No P&L, no exits, no signal evaluation. It answers
only: what would it cost to trade these, and can they be traded at all?

Usage:
  python check_spread_feasibility.py --tickers SPY QQQ IWM --sample 250
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
import polars as pl

from uw_options_data_lake import silver_partition_path, DEFAULT_LAKE

SPLIT = pd.Timestamp("2025-08-21").date()


def trigger_minutes(D, tk, pct, rng, sample):
    """Real trigger (date, minute) pairs for this ticker, sampled."""
    from check_config_walkforward import _flow_for
    flow = _flow_for(D, [tk])
    if flow.empty:
        return []
    trigs = D.triggers_for(flow, tk)
    D.annotate_flow_pct(trigs, 60)
    keep = []
    for t in trigs:
        thr = t.get("thr")
        if not thr or pct not in thr:
            continue
        if abs(float(t["abs_flow"])) < thr[pct]:
            continue
        ts = pd.Timestamp(t["ts"])
        keep.append((t["date"], ts.hour * 60 + ts.minute))
    if not keep:
        return []
    idx = rng.choice(len(keep), min(sample, len(keep)), replace=False)
    return [keep[i] for i in sorted(idx)]


def chain_at(tk, d, mod):
    """The ticker's quoted chain at (or just before) `mod` on date `d`."""
    p = silver_partition_path(DEFAULT_LAKE, d)
    if not p.exists():
        return None
    df = (pl.scan_parquet(p)
          .filter((pl.col("underlying_symbol") == tk)
                  & (pl.col("bid_close") > 0) & (pl.col("ask_close") > 0)
                  & (pl.col("ask_close") >= pl.col("bid_close")))
          .with_columns(
              m=(pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                 + pl.col("minute_et").dt.minute().cast(pl.Int32)),
              dte=(pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days())
          .filter(pl.col("m").is_between(mod - 3, mod))
          .select("option_chain_id", "option_type", "strike", "expiry", "dte",
                  "m", "bid_close", "ask_close", "underlying_close")
          .collect().to_pandas())
    if df.empty:
        return None
    # last quote per contract within the 3-minute window
    df = df.sort_values("m").groupby("option_chain_id", as_index=False).last()
    return df


def legs(ch, direction, far_dte):
    """Pick the contracts each structure needs. -> dict or None per structure."""
    if ch is None or ch.empty:
        return {}
    spot = float(ch["underlying_close"].iloc[0])
    typ = "call" if direction == "CALL" else "put"
    side = ch[ch["option_type"] == typ].copy()
    if side.empty:
        return {}
    side["mid"] = (side["bid_close"] + side["ask_close"]) / 2
    side["sprd"] = side["ask_close"] - side["bid_close"]
    near = side[side["dte"].between(0, 1)]
    far = side[side["dte"] >= far_dte]
    if near.empty:
        return {}
    # ATM = nearest strike to spot in the near expiry
    near = near.assign(dist=(near["strike"] - spot).abs())
    exp = near.loc[near["dist"].idxmin(), "expiry"]
    near = near[near["expiry"] == exp].sort_values("strike")
    atm = near.loc[near["dist"].idxmin()]
    if atm["mid"] < 0.50:                     # the bot's own entry floor
        return {}
    # next strike OUT of the money in the trade's direction
    otm_side = near[near["strike"] > atm["strike"]] if typ == "call" \
        else near[near["strike"] < atm["strike"]]
    otm = (otm_side.iloc[0] if typ == "call" else otm_side.iloc[-1]) \
        if len(otm_side) else None
    out = {"spot": spot, "atm": atm, "otm": otm}
    if not far.empty:
        far = far.assign(dist=(far["strike"] - spot).abs())
        fexp = far.loc[far["dte"].idxmin(), "expiry"]
        far = far[far["expiry"] == fexp]
        out["far"] = far.loc[far["dist"].idxmin()]
    return out


def costs(L):
    """Round-trip friction as a % of capital at risk, per structure."""
    if not L or "atm" not in L:
        return {}
    a = L["atm"]
    o = L.get("otm")
    f = L.get("far")
    r = {}
    # naked long: cross one spread, twice
    r["naked"] = (2 * a["sprd"] / a["mid"] * 100, a["mid"], 1)
    # debit vertical: net debit, cross BOTH spreads twice
    if o is not None:
        deb = a["mid"] - o["mid"]
        if deb > 0.02:
            r["debit_vert"] = (2 * (a["sprd"] + o["sprd"]) / deb * 100, deb, 2)
    # diagonal: long far + short near OTM
    if f is not None and o is not None:
        deb = f["mid"] - o["mid"]
        if deb > 0.02:
            r["diagonal"] = (2 * (f["sprd"] + o["sprd"]) / deb * 100, deb, 2)
    # credit vertical: short ATM, long next OTM. Risk = width - credit.
    if o is not None:
        cred = a["mid"] - o["mid"]
        width = abs(float(a["strike"]) - float(o["strike"]))
        risk = width - cred
        if cred > 0.02 and risk > 0.02:
            r["credit_vert"] = (2 * (a["sprd"] + o["sprd"]) / risk * 100, risk, 2)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--far-dte", type=int, default=7)
    ap.add_argument("--sample", type=int, default=250)
    a = ap.parse_args()

    import directional_flow_backtester as D
    rng = np.random.default_rng(5)
    rows = []
    for tk in a.tickers:
        for direction in a.dirs:
            tms = trigger_minutes(D, tk, a.pct, rng, a.sample)
            print(f"  {tk} {direction}: sampling {len(tms)} trigger minutes", flush=True)
            for d, mod in tms:
                ch = chain_at(tk, d, mod)
                c = costs(legs(ch, direction, a.far_dte))
                for k, (pctcost, cap, nlegs) in c.items():
                    rows.append(dict(ticker=tk, dir=direction, date=d, mod=mod,
                                     structure=k, friction=pctcost, capital=cap))
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no quotable structures found")
        return
    df.to_parquet("_spread_feasibility.parquet", index=False)

    print(f"\n{'='*104}\n  ROUND-TRIP FRICTION, % of capital at risk"
          f"   (trigger minutes, p{a.pct}, far>= {a.far_dte}DTE)\n{'='*104}")
    tot = df.groupby(["ticker", "dir"]).size().rename("attempts")
    print(f"  {'structure':13} {'n':>6} {'quotable':>9} {'median':>9} "
          f"{'p25':>8} {'p75':>8} {'med capital':>12}")
    base = df[df["structure"] == "naked"]
    nbase = len(base)
    for s, g in df.groupby("structure"):
        q = np.percentile(g["friction"], [25, 50, 75])
        print(f"  {s:13} {len(g):>6} {100*len(g)/max(nbase,1):>8.0f}% "
              f"{q[1]:>8.1f}% {q[0]:>7.1f}% {q[2]:>7.1f}% "
              f"${g['capital'].median():>11.2f}")
    print(f"\n  'quotable' is relative to the naked long (n={nbase}): a structure")
    print(f"  that cannot be priced at the trigger minute is not a strategy.")

    print(f"\n  BY TICKER (median friction %):")
    piv = df.pivot_table(index="structure", columns="ticker", values="friction",
                         aggfunc="median")
    print(piv.round(1).to_string())

    nm = base["friction"].median()
    print(f"\n  THE COMPARISON THAT MATTERS")
    print(f"  naked long round trip costs {nm:.1f}% of premium. For a structure to")
    print(f"  be worth it, the decay it SAVES must exceed the extra friction it")
    print(f"  ADDS. Grid baseline: the unconditioned trigger loses ~4-20%/trade,")
    print(f"  so a structure adding more than ~10pp of friction cannot rescue it")
    print(f"  no matter how much theta it avoids.")
    for s, g in df.groupby("structure"):
        if s == "naked":
            continue
        d = g["friction"].median() - nm
        verdict = ("PLAUSIBLE -- cheaper than naked" if d < 0 else
                   "MARGINAL -- costs a little more" if d < 10 else
                   "IMPLAUSIBLE -- friction alone exceeds the edge")
        print(f"    {s:13} {d:+6.1f}pp vs naked   {verdict}")


if __name__ == "__main__":
    main()
