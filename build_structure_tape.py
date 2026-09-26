# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
build_structure_tape.py
=======================
Prices four option STRUCTURES on the same flow triggers, so the instrument can
be compared against the naked long the bot actually trades.

  naked        long ATM, near expiry (0/1DTE)          -- the baseline
  debit_vert   long ATM + short next OTM, same expiry
  diagonal     long nearest expiry >= --far-dte at ATM
               + short near-dated OTM                   -- never tested before
  credit_vert  short ATM + long next OTM, same expiry   -- inverted thesis

WHY EOD EXIT FOR EVERYTHING
    One exit rule for all four, so the comparison is purely the INSTRUMENT. A
    trail tuned for naked longs would flatter or penalise the others for reasons
    that have nothing to do with the structure. Exit policy is a separate
    question and only worth asking if a structure shows something.

    Consequence: with a hold-to-EOD rule the one-position-per-ticker guard
    admits ONE trade per ticker-day, so `n` here is days, not fills. That is
    also why the null below is not the within-day permutation used by
    check_step1_redo -- with one trade per day there is nothing to permute.

THE NULL -- entry at a random plausible minute instead of at the signal
    For every ticker-day that produced a trigger, the SAME structure is also
    priced at `--nulls` random minutes drawn from the pooled distribution of
    real trigger minutes. Same day, same structure, same exit, same legs logic
    -- only the timing is random. Each structure is therefore measured against
    ITS OWN control, which is the whole point: a structure could look better
    than naked simply by being a lower-variance instrument, and only its own
    null can separate that from the signal actually working in it.

FILLS ARE BOUNDED, NOT ASSUMED (METHODOLOGY 1a)
    Every structure is priced under BOTH:
      mid    every leg at mid, entry and exit      (optimistic bound)
      cross  pay the ask on every buy, hit the bid on every sell, both ends
             (pessimistic bound)
    A conclusion has to survive both, or it is a statement about the fill model.

RETURNS ARE ON CAPITAL AT RISK, so the four are comparable:
    debit structures  (exit_value - entry_cost) / entry_cost
    credit vertical   (entry_credit - exit_cost) / (width - credit)

Usage:
  python build_structure_tape.py --tickers SPY QQQ IWM --nulls 5
  python build_structure_tape.py --dates 2024-01-02 2024-06-28   # smoke test
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
import polars as pl

from uw_options_data_lake import trading_days, silver_partition_path, DEFAULT_LAKE

EOD = 955                      # 15:55 ET flatten, same as the bot
FLOOR = 0.50                   # bot_runner.py:1421 entry floor, applied to the long


def trigger_map(D, tickers, pct):
    """{(ticker, date): [minute, ...]} for real triggers clearing `pct`."""
    from check_config_walkforward import _flow_for
    out: dict = {}
    pool: list = []
    for tk in tickers:
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        for t in trigs:
            thr = t.get("thr")
            if not thr or pct not in thr:
                continue
            if abs(float(t["abs_flow"])) < thr[pct]:
                continue
            ts = pd.Timestamp(t["ts"])
            m = ts.hour * 60 + ts.minute
            if not (570 <= m <= EOD - 15):
                continue
            out.setdefault((tk, t["date"]), []).append(m)
            pool.append(m)
    return out, np.array(sorted(pool))


def day_chain(d, tickers):
    """All quoted contracts for `tickers` on date `d`, one partition read."""
    p = silver_partition_path(DEFAULT_LAKE, d)
    if not p.exists():
        return None
    df = (pl.scan_parquet(p)
          .filter(pl.col("underlying_symbol").is_in(tickers)
                  & (pl.col("bid_close") > 0) & (pl.col("ask_close") > 0)
                  & (pl.col("ask_close") >= pl.col("bid_close")))
          .with_columns(
              m=(pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                 + pl.col("minute_et").dt.minute().cast(pl.Int32)),
              dte=(pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days())
          .filter(pl.col("m").is_between(570, EOD))
          .select("underlying_symbol", "option_chain_id", "option_type", "strike",
                  "expiry", "dte", "m", "bid_close", "ask_close", "underlying_close")
          .collect().to_pandas())
    return df if not df.empty else None


def _quote(sub, cid, m):
    """Last two-sided quote for `cid` at or just before minute `m`."""
    g = sub[(sub["option_chain_id"] == cid) & (sub["m"] <= m)]
    if g.empty:
        return None
    r = g.iloc[-1]
    return float(r["bid_close"]), float(r["ask_close"])


def pick_legs(sub, direction, m, far_dte):
    """Contracts each structure needs, chosen at minute `m`."""
    at = sub[sub["m"] <= m]
    if at.empty:
        return None
    spot = float(at.iloc[-1]["underlying_close"])
    # `underlying_close` is occasionally NaN in the lake. Unguarded, every
    # strike distance becomes NaN and idxmin raises "Encountered all NA values"
    # -- which killed the first full build 80 dates in.
    if not np.isfinite(spot) or spot <= 0:
        return None
    typ = "call" if direction == "CALL" else "put"
    side = at[at["option_type"] == typ]
    if side.empty:
        return None
    # last quote per contract up to m
    last = side.sort_values("m").groupby("option_chain_id", as_index=False).last()
    last["mid"] = (last["bid_close"] + last["ask_close"]) / 2
    near = last[last["dte"].between(0, 1)]
    if near.empty:
        return None
    near = near.assign(dist=(near["strike"] - spot).abs()).dropna(subset=["dist"])
    if near.empty:
        return None
    exp = near.loc[near["dist"].idxmin(), "expiry"]
    near = near[near["expiry"] == exp].sort_values("strike")
    if near.empty:
        return None
    atm = near.loc[near["dist"].idxmin()]
    if float(atm["mid"]) < FLOOR:
        return None
    o = near[near["strike"] > atm["strike"]] if typ == "call" \
        else near[near["strike"] < atm["strike"]]
    otm = (o.iloc[0] if typ == "call" else o.iloc[-1]) if len(o) else None
    far = last[last["dte"] >= far_dte]
    fr = None
    if not far.empty:
        far = far.assign(dist=(far["strike"] - spot).abs()).dropna(subset=["dist"])
        if not far.empty:
            fexp = far.loc[far["dte"].idxmin(), "expiry"]
            far = far[far["expiry"] == fexp]
            if not far.empty:
                fr = far.loc[far["dist"].idxmin()]
    return {"spot": spot, "atm": atm, "otm": otm, "far": fr}


def _px(q, side, fill):
    """Price one leg. side=+1 buy, -1 sell. fill 'mid' or 'cross'."""
    b, a = q
    if fill == "mid":
        return (b + a) / 2
    return a if side > 0 else b


#: leg sides when OPENING each structure. +1 = buy, -1 = sell.
#: Closing reverses every side, which is why `value` takes `opening` -- under
#: the `cross` fill you pay the ask going in AND hit the bid coming out, on
#: every leg. Getting this wrong would quietly hand the spreads a free
#: half-spread at one end and make them look better than they are.
SIDES = {
    "naked":       {"atm": +1},
    "debit_vert":  {"atm": +1, "otm": -1},
    "diagonal":    {"far": +1, "otm": -1},
    "credit_vert": {"atm": -1, "otm": +1},
}


def value(sub, L, m, fill, structure, opening):
    """Net cash price of the structure at minute `m`.

    Positive = the package costs money (a debit). Negative = it pays you.
    `opening` flips every leg's side when closing the position.
    """
    sign = 1 if opening else -1
    tot = 0.0
    for key, side in SIDES[structure].items():
        leg = L.get(key)
        if leg is None:
            return None
        q = _quote(sub, leg["option_chain_id"], m)
        if q is None:
            return None
        s = side * sign
        tot += s * _px(q, s, fill)
    return tot


def price_one(sub, direction, m, far_dte, fill, structure):
    """-> (pnl_fraction, capital_at_risk) held from `m` to EOD, or None."""
    L = pick_legs(sub, direction, m, far_dte)
    if L is None:
        return None
    ent = value(sub, L, m, fill, structure, opening=True)
    if ent is None:
        return None

    if structure == "credit_vert":
        credit = -ent                            # opening a credit pays you
        a, o = L["atm"], L["otm"]
        width = abs(float(a["strike"]) - float(o["strike"]))
        risk = width - credit
        if credit <= 0.02 or risk <= 0.02:
            return None
        close = value(sub, L, EOD, fill, structure, opening=False)
        if close is None:
            return None
        # With opening=False the legs reverse, so `close` is the POSITIVE cost
        # of buying the spread back. P&L is credit MINUS that cost.
        # (This was `credit + close` until the mirror check in verify_signs
        # caught it: it made the credit vertical print ~+$1.00/trade of free
        # money and broke the identity that debit and credit P&L must sum to
        # zero under a mid fill.)
        return (credit - close) / risk, risk

    cost = ent
    if cost <= 0.02:
        return None
    close = value(sub, L, EOD, fill, structure, opening=False)
    if close is None:
        return None
    proceeds = -close                            # unwinding a debit pays you
    return (proceeds - cost) / cost, cost


STRUCTURES = ("naked", "debit_vert", "diagonal", "credit_vert")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--far-dte", type=int, default=7)
    ap.add_argument("--nulls", type=int, default=5)
    ap.add_argument("--dates", nargs="*", default=None, help="smoke-test subset")
    ap.add_argument("--out", default="_structure_tape.parquet")
    ap.add_argument("--seed", type=int, default=29)
    a = ap.parse_args()

    import directional_flow_backtester as D
    tmap, pool = trigger_map(D, a.tickers, a.pct)
    if not len(pool):
        raise SystemExit("no triggers")
    print(f"  {len(tmap)} ticker-days with a p{a.pct} trigger; "
          f"trigger-minute pool n={len(pool)} "
          f"(p10 {np.percentile(pool,10):.0f} med {np.median(pool):.0f} "
          f"p90 {np.percentile(pool,90):.0f})")

    dates = ([dt.date.fromisoformat(x) for x in a.dates] if a.dates
             else sorted({d for _, d in tmap}))
    rng = np.random.default_rng(a.seed)
    rows, done = [], 0
    for d in dates:
        ch = day_chain(d, a.tickers)
        if ch is None:
            continue
        for tk in a.tickers:
            sub_tk = ch[ch["underlying_symbol"] == tk]
            if sub_tk.empty:
                continue
            mins = tmap.get((tk, d))
            if not mins:
                continue
            # ONE position per ticker-day: the first trigger (hold to EOD)
            entries = [("real", min(mins))]
            entries += [("null", int(rng.choice(pool))) for _ in range(a.nulls)]
            for direction in a.dirs:
                for kind, m in entries:
                    for fill in ("mid", "cross"):
                        for st in STRUCTURES:
                            r = price_one(sub_tk, direction, m, a.far_dte, fill, st)
                            if r is None:
                                continue
                            rows.append(dict(date=d, ticker=tk, dir=direction,
                                             structure=st, kind=kind, fill=fill,
                                             mod=m, pnl=r[0], capital=r[1]))
        done += 1
        if done % 25 == 0:
            print(f"    {done}/{len(dates)} dates, {len(rows):,} rows", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("  nothing priced")
        return
    df.to_parquet(a.out, index=False)
    print(f"\n  wrote {len(df):,} rows -> {a.out}  "
          f"({df['date'].nunique()} dates)")
    print(f"\n  sanity -- median capital at risk by structure (fill=cross):")
    q = df[df["fill"] == "cross"].groupby("structure")["capital"].median()
    for s in STRUCTURES:
        if s in q:
            print(f"    {s:12} ${q[s]:.2f}")
    print(f"\n  coverage: real entries per structure")
    c = df[(df["kind"] == "real") & (df["fill"] == "cross")].groupby("structure").size()
    for s in STRUCTURES:
        print(f"    {s:12} {int(c.get(s, 0))}")


if __name__ == "__main__":
    main()
