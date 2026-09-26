# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
build_hedge_flow.py
===================
Builds the DELTA-WEIGHTED DEALER HEDGING DEMAND series from the silver lake,
per underlying, per minute -- the mechanistically correct version of the flow
signal the bot currently trades on.

THE QUANTITY
------------
    hedge_shares = SUM over the whole chain of  (ask_vol - bid_vol) * delta * 100

  * `ask_vol - bid_vol` is CUSTOMER NET BUYING (aggressor side). If customers
    lift the ask, the market maker is the seller and gets SHORTER that contract.
  * the dealer's delta change is the negative of the customer's, so to flatten,
    the dealer must trade +(ask_vol - bid_vol) * delta * 100 shares.
  * put delta is already negative, so the sign falls out with no special-casing:
        customers BUY CALLS  -> +delta -> dealer BUYS  stock  -> bullish
        customers BUY PUTS   -> -delta -> dealer SELLS stock  -> bearish
  * POSITIVE hedge_shares = dealers must BUY the underlying.

WHY THIS BEATS THE DEPLOYED SIGNAL
----------------------------------
The bot triggers on UW `net_premium = net_call_premium - net_put_premium`. That
IS aggressor-signed (verified: corr +0.45..+0.50 and 73-78% sign agreement with
ask-minus-bid volume), so its DIRECTION is right. But it is weighted by PREMIUM
DOLLARS, and hedging scales with DELTA x CONTRACTS. One deep-ITM print at the
bid outweighs a hundred cheap OTM lifts at the ask -- which is exactly why the
premium and volume signals disagree on sign a quarter of the time. Premium is
dominated by expensive contracts; hedging demand by near-the-money ones.

WHAT ELSE IS CAPTURED (and why)
-------------------------------
  multi_frac    share of volume that was part of a MULTI-LEG order. A vertical's
                two legs each get aggressor-tagged, but the package's net delta
                is far smaller than the legs imply and the tag is often
                meaningless for a package. High multi_frac = inflated hedge_sh.
  unclass_frac  (mid_volume + no_side_volume) / volume -- real inventory changes
                with no attributable side, silently discarded by any aggressor
                measure. This is the honest noise floor.
  gross_sh      SUM (ask_vol + bid_vol) * |delta| * 100 -- total hedging ACTIVITY
                regardless of direction. hedge_sh / gross_sh is a bounded
                [-1, 1] "how one-sided is the hedging" ratio, drift-resistant in
                the way check_flow_drift showed raw dollar magnitudes are not.
  call/put and DTE splits, because near-dated dominates gamma.

CAVEATS THAT NO AMOUNT OF DATA FIXES
------------------------------------
  * not every counterparty is a dealer -- customer-to-customer crosses need no
    hedge at all, and are indistinguishable here.
  * dealers net inventory across strikes and hedge in bulk, so per-trade delta
    is an UPPER BOUND on hedging, not the realised quantity.
  * open-vs-close does NOT matter here and is not needed: if the customer buys,
    the MM sells and gets shorter regardless, so the immediate hedge is the same.
    Open/close governs whether GAMMA accumulates, which is the GEX stock and is
    already handled from `open_interest` at daily resolution.

OUTPUT
------
    historical/HEDGE{TICKER}.parquet   one row per RTH minute:
      minute_et, date, hedge_sh, hedge_sh_c, hedge_sh_p,
      hedge_sh_d01, hedge_sh_d27, hedge_sh_d8p,
      gross_sh, net_prem_lake, volume, multi_frac, unclass_frac,
      missing_delta_frac, n_contracts

MEMORY: one pass over the date partitions with ALL tickers filtered at once
(501 file opens total, not 501 x n_tickers), projecting 13 of 33 columns and
aggregating to per-minute before anything accumulates. Peak stays flat -- the
ml_feature_scan OOM on this 16GB box came from holding per-contract rows.

Usage:
  python build_hedge_flow.py
  python build_hedge_flow.py --tickers NVDA META --force
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time

import numpy as np
import polars as pl

LAKE = os.path.join("lake", "silver", "option-contracts-1m")
OUT = "historical"
RTH_LO, RTH_HI = 9 * 60 + 30, 16 * 60

BOOK = ["META", "MSFT", "NVDA", "SPY", "QQQ", "IWM", "AVGO", "SMH", "GLD"]

COLS = ["underlying_symbol", "option_type", "expiry", "minute_et",
        "ask_volume", "bid_volume", "mid_volume", "no_side_volume",
        "multi_volume", "volume", "delta_close", "vwap"]


def _one_date(path, tickers):
    """Aggregate one lake partition to per-(ticker, minute) hedging demand."""
    d = path.split("date=")[1].split(os.sep)[0]
    day = pl.Series([d]).str.to_date().item()
    lf = (pl.scan_parquet(path)
            .filter(pl.col("underlying_symbol").is_in(tickers))
            .select(COLS))
    df = lf.collect()
    if df.is_empty():
        return None

    df = df.with_columns([
        pl.col("minute_et").dt.hour().cast(pl.Int32).alias("_h"),
        pl.col("minute_et").dt.minute().cast(pl.Int32).alias("_m"),
        (pl.col("expiry").cast(pl.Date) - pl.lit(day)).dt.total_days().alias("dte"),
    ])
    df = df.with_columns((pl.col("_h") * 60 + pl.col("_m")).alias("mod"))
    df = df.filter((pl.col("mod") >= RTH_LO) & (pl.col("mod") <= RTH_HI))
    if df.is_empty():
        return None

    for c in ("ask_volume", "bid_volume", "mid_volume", "no_side_volume",
              "multi_volume", "volume", "delta_close", "vwap"):
        df = df.with_columns(pl.col(c).cast(pl.Float64))

    # customer net buying, and the share-equivalent hedge it forces on the dealer
    df = df.with_columns([
        (pl.col("ask_volume").fill_null(0.0) - pl.col("bid_volume").fill_null(0.0)).alias("netbuy"),
        (pl.col("ask_volume").fill_null(0.0) + pl.col("bid_volume").fill_null(0.0)).alias("grossbuy"),
        pl.col("delta_close").is_null().alias("nodelta"),
        # per-contract-minute multi-leg share. `multi_volume` TAGS volume that is
        # already counted in ask/bid/mid/no_side -- it is not a separate bucket --
        # so it cannot simply be subtracted from netbuy.
        (pl.col("multi_volume").fill_null(0.0)
         / pl.col("volume").fill_null(0.0).clip(lower_bound=1.0)
         ).clip(0.0, 1.0).alias("mfrac_c"),
    ])
    df = df.with_columns([
        (pl.col("netbuy") * pl.col("delta_close").fill_null(0.0) * 100.0).alias("hsh"),
        # --- TWO SINGLE-LEG VARIANTS, because stripping multi-leg requires an
        # assumption and the two available ones fail differently.
        # A vertical's legs each get aggressor-tagged, but the package's NET
        # delta is far smaller than the legs imply, so multi-leg volume inflates
        # hedge_sh. Multi-leg share runs 12.2% (SPY) to 32.8% (GLD).
        #   hsh_sl   scale netbuy by the single-leg share. ASSUMES multi-leg
        #            volume splits across aggressor sides in the same proportion
        #            as total volume -- wrong per-row, unbiased in aggregate.
        #   hsh_sld  drop any contract-minute more than half multi-leg. Assumes
        #            nothing about the split, but discards data and tilts the
        #            sample toward contracts that happen to be single-leg heavy.
        # They disagree exactly where the assumption matters, which is the point.
        (pl.col("netbuy") * (1.0 - pl.col("mfrac_c"))
         * pl.col("delta_close").fill_null(0.0) * 100.0).alias("hsh_sl"),
        pl.when(pl.col("mfrac_c") <= 0.5)
          .then(pl.col("netbuy") * pl.col("delta_close").fill_null(0.0) * 100.0)
          .otherwise(0.0).alias("hsh_sld"),
        (pl.col("grossbuy") * pl.col("delta_close").fill_null(0.0).abs() * 100.0).alias("gsh"),
        (pl.col("netbuy") * pl.col("vwap").fill_null(0.0) * 100.0).alias("nprem"),
        (pl.col("volume").fill_null(0.0)).alias("vol"),
        (pl.col("multi_volume").fill_null(0.0)).alias("mvol"),
        (pl.col("mid_volume").fill_null(0.0) + pl.col("no_side_volume").fill_null(0.0)).alias("uvol"),
    ])
    is_c = pl.col("option_type").cast(pl.Utf8).str.to_lowercase().str.starts_with("c")

    agg = df.group_by(["underlying_symbol", "mod"]).agg([
        pl.col("hsh").sum().alias("hedge_sh"),
        pl.col("hsh_sl").sum().alias("hedge_sh_sl"),
        pl.col("hsh_sld").sum().alias("hedge_sh_sld"),
        pl.when(is_c).then(pl.col("hsh")).otherwise(0.0).sum().alias("hedge_sh_c"),
        pl.when(~is_c).then(pl.col("hsh")).otherwise(0.0).sum().alias("hedge_sh_p"),
        pl.when(pl.col("dte") <= 1).then(pl.col("hsh")).otherwise(0.0).sum().alias("hedge_sh_d01"),
        pl.when((pl.col("dte") >= 2) & (pl.col("dte") <= 7)).then(pl.col("hsh")).otherwise(0.0).sum().alias("hedge_sh_d27"),
        pl.when(pl.col("dte") >= 8).then(pl.col("hsh")).otherwise(0.0).sum().alias("hedge_sh_d8p"),
        pl.col("gsh").sum().alias("gross_sh"),
        pl.col("nprem").sum().alias("net_prem_lake"),
        pl.col("vol").sum().alias("volume"),
        pl.col("mvol").sum().alias("multi_vol"),
        pl.col("uvol").sum().alias("unclass_vol"),
        pl.col("nodelta").mean().alias("missing_delta_frac"),
        pl.len().alias("n_contracts"),
    ])
    return agg.with_columns(pl.lit(day).alias("date"))


def run(a):
    tickers = [t.upper() for t in (a.tickers or BOOK)]
    files = sorted(glob.glob(os.path.join(LAKE, "date=*", "*.parquet")))
    if not files:
        sys.exit(f"no lake partitions under {LAKE}")
    todo = [t for t in tickers
            if a.force or not os.path.exists(os.path.join(OUT, f"HEDGE{t}.parquet"))]
    if not todo:
        print("  all present; --force to rebuild"); return
    print(f"  building hedge-flow for {len(todo)} ticker(s): {', '.join(todo)}")
    print(f"  {len(files)} lake partitions, {files[0].split('date=')[1][:10]} .. "
          f"{files[-1].split('date=')[1][:10]}")

    parts, t0 = [], time.time()
    for i, f in enumerate(files, 1):
        try:
            g = _one_date(f, todo)
        except Exception as e:
            print(f"    ! {f.split('date=')[1][:10]}: {e}", flush=True)
            continue
        if g is not None:
            parts.append(g)
        if i % 50 == 0:
            el = time.time() - t0
            print(f"    {i}/{len(files)}  {el:6.0f}s  eta {el/i*(len(files)-i):6.0f}s  "
                  f"rows so far {sum(p.height for p in parts):,}", flush=True)
            gc.collect()

    if not parts:
        sys.exit("no data collected")
    allg = pl.concat(parts, how="vertical_relaxed")
    del parts
    gc.collect()

    os.makedirs(OUT, exist_ok=True)
    for tk in todo:
        sub = allg.filter(pl.col("underlying_symbol") == tk).drop("underlying_symbol")
        if sub.is_empty():
            print(f"  {tk}: NO ROWS -- not in the lake?"); continue
        sub = sub.with_columns([
            (pl.col("multi_vol") / pl.col("volume").clip(lower_bound=1.0)).alias("multi_frac"),
            (pl.col("unclass_vol") / pl.col("volume").clip(lower_bound=1.0)).alias("unclass_frac"),
        ]).sort(["date", "mod"])
        p = os.path.join(OUT, f"HEDGE{tk}.parquet")
        sub.write_parquet(p)
        nd = sub["date"].n_unique()
        hs = sub["hedge_sh"].to_numpy()
        print(f"  {tk:5} {sub.height:>7,} min-rows  {nd:>4} days  "
              f"hedge_sh mean {np.nanmean(hs):>+12,.0f} sh  "
              f"multi {sub['multi_frac'].mean()*100:>4.1f}%  "
              f"unclass {sub['unclass_frac'].mean()*100:>4.1f}%  "
              f"nodelta {sub['missing_delta_frac'].mean()*100:>4.1f}%  -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    run(ap.parse_args())
