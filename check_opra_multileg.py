"""
check_opra_multileg.py
======================
DOES MULTI-LEG VOLUME CONTAMINATE THE ASK/BID AGGRESSION MEASURE AT NEAR
STRIKES -- MEASURED WITH OPRA CONDITION CODES, NOT ASSUMED?

WHY THIS RUNS BEFORE check_strike_crossing_flow.py
    That study measures buy-vs-sell aggression per strike,
        agg = (ask_volume - bid_volume) / (ask_volume + bid_volume),
    from the SILVER lake, which is three years deep but only knows HOW MUCH of a
    contract-minute was multi-leg (`multi_volume`), not WHICH SIDE it printed
    on. A spread's legs are not directional bets on their own, so if they print
    at the ask or bid they pollute the measure; if they print at mid they are
    excluded from it already and the problem is small.

    The BRONZE full tape has 12 sessions (2026-09-01..18) of individual trades,
    each with its OPRA `upstream_condition_detail` and a side tag. Twelve days
    cannot carry the crossing study, but they can settle what silver's control
    has to do. This looks at COMPOSITION ONLY -- no price outcome is touched --
    so it does not contaminate that study's pre-registration.

OPRA CODES, AS SEEN IN THIS FEED
    single leg   auto slan slai slcn slci slft isoi
    multi leg    mlet mlat mlct mlft mesl masl mfsl      <- "multi"
    tied-to-stock tlet tlat tlct tlft tesl tasl tfsl     <- "tied"
Tied-to-stock trades (buy-writes, delta-neutral combos) are option legs of a
package with stock; like multi-leg they are not standalone directional flow.
They are reported separately because silver's `multi_volume` may or may not
include them, and that is one of the things being checked.

🚨 THE PER-TRADE ask_vol / bid_vol COLUMNS ARE RUNNING TOTALS, NOT PER TRADE.
Measured 2026-09-26: they sum to `size` on 0.5% of rows. Side comes from the
`tags` field (ask_side / bid_side / mid_side / no_side). Anyone reaching for
those columns to classify individual trades gets the contract's cumulative day
volume instead -- plausible numbers, meaningless as a side.

WHAT IS MEASURED
    V1  silver multi_volume  vs  bronze multi-code volume, per contract-minute
    V2  silver ask/bid       vs  bronze side-tag volume, per contract-minute
    S   side mix (ask / bid / mid / none) of single-leg vs multi-leg volume
    F   the aggression measure over 10-minute windows, three ways:
          all      every trade (what silver gives, un-cleaned)
          clean    single-leg trades only (ground truth, bronze only)
          excl25   silver, dropping contract-minutes >25% multi (the planned
                   control arm)
        and how closely `all` and `excl25` track `clean`.

PRE-STATED READING (written before running)
    corr(all, clean) >= 0.95  -> contamination is second-order for this measure;
                                 the multi-leg control stays a robustness arm.
    corr(all, clean) <  0.90  -> contamination is first-order; the multi-leg
                                 treatment must become part of the PRIMARY
                                 feature, not a sensitivity check.
    in between                -> report, keep as robustness arm, flag it.
    And whichever silver-only arm tracks `clean` better is the one the study
    uses, since clean itself is only available for 12 days.

RESULT -- 2026-09-26, 11 paired sessions (09-17 has no silver), 165M contracts.
    V1/V2  Silver IS the OPRA tape, aggregated. Total, ask and bid volume match
           bronze on 100.0% of contract-minutes; `multi_volume` matches the
           seven multi-leg codes on 100.0% (ratio 1.0006) and EXCLUDES the
           tied-to-stock tl* codes. So silver's multi tag is exact, not an
           estimate, and needs no API work to use.
    S      Multi-leg is 8.2% of near-strike ask+bid volume, and it is NOT
           parked at mid: 42% ask / 54% bid, vs single-leg 48 / 48. It leans
           to the SELL side, so an unfiltered measure reads slightly more
           "selling" than the directional trades really show.
    F      corr(all, clean) = 0.904 pooled -- the "in between" band. 1DTE is
           worse (SPY 0.866, QQQ 0.872) than 0DTE (0.906-0.967).
           corr(excl25, clean) = 0.976 pooled, 0.969-0.993 in every cell, sign
           agreement 95.5%, at a cost of ~2% of windows.
    -> Per the reading stated above: excl25 tracks clean better, so it becomes
       the PRIMARY feature in check_strike_crossing_flow.py, and unfiltered
       becomes the robustness arm -- the reverse of the original plan.

Usage:  python check_opra_multileg.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import polars as pl

BRONZE = os.path.join("lake", "bronze", "full-tape")
NSF = "_nsf_cache"
TICKERS = ["SPY", "QQQ", "IWM"]
MULTI = {"mlet", "mlat", "mlct", "mlft", "mesl", "masl", "mfsl"}
TIED = {"tlet", "tlat", "tlct", "tlft", "tesl", "tasl", "tfsl"}
WIN = 10            # minutes, matches the crossing study's feature window
MIN_AB = 10         # ask+bid contracts in a window before the ratio means anything


def side_of(tags: pl.Expr) -> pl.Expr:
    return (pl.when(tags.str.contains("ask_side")).then(pl.lit("ask"))
            .when(tags.str.contains("bid_side")).then(pl.lit("bid"))
            .when(tags.str.contains("mid_side")).then(pl.lit("mid"))
            .otherwise(pl.lit("none")))


def load_bronze(path: str) -> pl.DataFrame:
    day = os.path.basename(path)[:10]
    keys = pl.read_parquet(os.path.join(NSF, "opt", f"{day}.parquet"),
                           columns=["ticker", "option_type", "strike", "expiry",
                                    "dte_rank"]).unique()
    t = (pl.scan_parquet(path)
         .filter(pl.col("underlying_symbol").is_in(TICKERS))
         .select("underlying_symbol", "executed_at", "size", "option_type",
                 "strike", "expiry", "upstream_condition_detail", "tags",
                 "canceled")
         .collect())
    t = (t.rename({"underlying_symbol": "ticker"})
         .with_columns(pl.col("expiry").str.to_date(),
                       pl.col("executed_at").dt.truncate("1m")
                       .dt.convert_time_zone("America/New_York").alias("minute_et"),
                       side_of(pl.col("tags")).alias("side"),
                       pl.col("upstream_condition_detail").str.to_lowercase().alias("code"))
         .with_columns(pl.when(pl.col("code").is_in(list(MULTI))).then(pl.lit("multi"))
                       .when(pl.col("code").is_in(list(TIED))).then(pl.lit("tied"))
                       .otherwise(pl.lit("single")).alias("leg")))
    return t.join(keys, on=["ticker", "option_type", "strike", "expiry"])


def main():
    days = sorted(glob.glob(os.path.join(BRONZE, "*.parquet")))
    print(f"bronze sessions: {len(days)}  ({os.path.basename(days[0])[:10]} .. "
          f"{os.path.basename(days[-1])[:10]})")
    # Silver has no 2026-09-16 or 2026-09-17, so those two bronze days have
    # nothing to be compared against. Dropped, and said so.
    paired = [p for p in days if os.path.exists(
        os.path.join(NSF, "opt", os.path.basename(p)[:10] + ".parquet"))]
    for p in sorted(set(days) - set(paired)):
        print(f"  {os.path.basename(p)[:10]}  skipped: no silver session to compare")
    days = paired

    frames = []
    for p in days:
        b = load_bronze(p)
        canc = b["canceled"].cast(pl.Utf8).fill_null("").str.to_lowercase()
        n_canc = canc.is_in(["true", "t", "1", "yes"]).sum()
        b = b.filter(~canc.is_in(["true", "t", "1", "yes"]))
        frames.append(b.with_columns(pl.lit(os.path.basename(p)[:10]).alias("date")))
        print(f"  {os.path.basename(p)[:10]}  {b.height:>9,} near-strike trades  "
              f"({n_canc} canceled dropped)")
    b = pl.concat(frames)

    # ------------------------------------------------------------------ S
    print("\n=== S. side mix by leg type (share of contracts) ===")
    s = (b.group_by("leg", "side").agg(pl.col("size").sum())
         .with_columns((pl.col("size") / pl.col("size").sum().over("leg")).alias("share"))
         .pivot(on="side", index="leg", values="share").fill_null(0.0))
    tot = b.group_by("leg").agg(pl.col("size").sum().alias("contracts"))
    s = s.join(tot, on="leg").with_columns(
        (pl.col("contracts") / pl.col("contracts").sum()).alias("of_all_volume"))
    with pl.Config(float_precision=3, tbl_cols=10):
        print(s.sort("leg"))
    ab = (b.filter(pl.col("side").is_in(["ask", "bid"]))
          .group_by("leg").agg(pl.col("size").sum().alias("ab")))
    ab = ab.with_columns((pl.col("ab") / pl.col("ab").sum()).alias("share_of_ask_plus_bid"))
    print("\n  share of the aggression measure's DENOMINATOR (ask+bid volume) by leg type:")
    for r in ab.sort("leg").iter_rows(named=True):
        print(f"    {r['leg']:<7} {r['share_of_ask_plus_bid']:.1%}")

    # -------------------------------------------------------- per contract-minute
    key = ["date", "ticker", "option_type", "strike", "expiry", "minute_et"]
    cm = b.group_by(key + ["dte_rank"]).agg(
        pl.col("size").filter(pl.col("side") == "ask").sum().alias("b_ask"),
        pl.col("size").filter(pl.col("side") == "bid").sum().alias("b_bid"),
        pl.col("size").filter((pl.col("side") == "ask") & (pl.col("leg") == "single")).sum().alias("c_ask"),
        pl.col("size").filter((pl.col("side") == "bid") & (pl.col("leg") == "single")).sum().alias("c_bid"),
        pl.col("size").filter(pl.col("leg") == "multi").sum().alias("b_multi"),
        pl.col("size").filter(pl.col("leg") == "tied").sum().alias("b_tied"),
        pl.col("size").sum().alias("b_vol"))

    sil = pl.concat([
        pl.read_parquet(os.path.join(NSF, "opt", f"{d}.parquet"))
        .with_columns(pl.lit(d).alias("date"))
        for d in sorted(b["date"].unique())])
    sil = sil.select(key + ["volume", "ask_volume", "bid_volume", "multi_volume"])
    j = sil.join(cm, on=key, how="full", coalesce=True).fill_null(0)

    # ------------------------------------------------------------------ V1 / V2
    print("\n=== V1/V2. silver vs bronze, per contract-minute ===")
    for a, c, name in (("volume", "b_vol", "total volume"),
                       ("ask_volume", "b_ask", "ask volume"),
                       ("bid_volume", "b_bid", "bid volume"),
                       ("multi_volume", "b_multi", "multi vs OPRA multi codes"),
                       ("multi_volume", None, "multi vs OPRA multi+tied")):
        x = j[a].to_numpy().astype(float)
        y = (j[c].to_numpy().astype(float) if c else
             (j["b_multi"] + j["b_tied"]).to_numpy().astype(float))
        exact = np.mean(x == y)
        within = np.mean(np.abs(x - y) <= np.maximum(1, 0.05 * np.maximum(x, y)))
        print(f"  {name:<28} totals silver {x.sum():>12,.0f}  bronze {y.sum():>12,.0f}  "
              f"ratio {x.sum() / max(y.sum(), 1):.4f}   rows exact {exact:.1%}  "
              f"within 5% {within:.1%}")

    # ------------------------------------------------------------------ F
    print(f"\n=== F. the aggression measure over {WIN}-minute windows ===")
    j = j.with_columns((pl.col("multi_volume") / pl.col("volume").clip(1)).alias("mshare"))
    j = j.with_columns(
        pl.when(pl.col("mshare") <= 0.25).then(pl.col("ask_volume")).otherwise(0).alias("x_ask"),
        pl.when(pl.col("mshare") <= 0.25).then(pl.col("bid_volume")).otherwise(0).alias("x_bid"),
        pl.col("minute_et").dt.truncate(f"{WIN}m").alias("win"))
    w = j.group_by(["date", "ticker", "option_type", "strike", "expiry", "dte_rank", "win"]).agg(
        pl.col("ask_volume").sum().alias("a_all"), pl.col("bid_volume").sum().alias("b_all"),
        pl.col("c_ask").sum().alias("a_cl"), pl.col("c_bid").sum().alias("b_cl"),
        pl.col("x_ask").sum().alias("a_x"), pl.col("x_bid").sum().alias("b_x"))

    def agg(a, b_):
        return pl.when((pl.col(a) + pl.col(b_)) >= MIN_AB).then(
            (pl.col(a) - pl.col(b_)) / (pl.col(a) + pl.col(b_)))

    w = w.with_columns(agg("a_all", "b_all").alias("all"),
                       agg("a_cl", "b_cl").alias("clean"),
                       agg("a_x", "b_x").alias("excl25"))
    rows = []
    for grp, g in [(("ALL",), w)] + sorted(w.group_by("ticker", "dte_rank"),
                                           key=lambda x: str(x[0])):
        out = {"cell": " ".join(str(v) for v in grp)}
        for arm in ("all", "excl25"):
            gg = g.select(arm, "clean").drop_nulls()
            if gg.height < 30:
                continue
            x, y = gg[arm].to_numpy(), gg["clean"].to_numpy()
            out[f"corr_{arm}"] = np.corrcoef(x, y)[0, 1]
            out[f"sign_agree_{arm}"] = np.mean(np.sign(x) == np.sign(y))
            out[f"mad_{arm}"] = np.median(np.abs(x - y))
            out[f"n_{arm}"] = gg.height
        rows.append(out)
    with pl.Config(float_precision=3, tbl_cols=12, tbl_width_chars=160):
        print(pl.DataFrame(rows))
    print("\n  corr = Pearson vs clean; sign_agree = same sign as clean;\n"
          "  mad = median |arm - clean| on a measure that runs -1..+1.")


if __name__ == "__main__":
    main()
