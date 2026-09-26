# /// script
# requires-python = ">=3.11"
# dependencies = ["databento", "python-dotenv"]
# ///
"""
check_databento_cost.py
=======================
PRICE THE DATA BEFORE BUYING ANY OF IT.

Databento's metadata API quotes the cost of a query *before* you run it
(`metadata.get_cost`), and metadata calls are free. So the binding question --
"what would CME MBO for ES/NQ/RTY over our backtest window actually cost?" --
is answerable right now, for nothing.

WHY THIS MATTERS MORE THAN IT SOUNDS
------------------------------------
The whole reason Databento reopened the MBO thread is that it sells HISTORY.
That inverts the plan we'd been stuck with:

    old (Rithmic/dxFeed):  pay monthly -> record 3-6 months -> then test
    new (Databento):       buy history -> test NOW -> pay for live ONLY if
                           the research earns it

A live subscription bought before the research is exactly the "deploy on faith"
this project has spent months learning not to do. **Historical is pay-as-you-go
with no minimum, and new accounts get $125 of free credits** -- so the test can
likely be run for little or nothing. Get the cost first; decide after.

CME MBO IS HIGH VOLUME. Every order add/modify/cancel on the whole book, not
just trades. Cost and disk are the likely constraints, not access -- which is
why this prices several scopes, from one day to the full window.

Usage:  python check_databento_cost.py
        python check_databento_cost.py --start 2024-08-20 --end 2026-08-21
"""
from __future__ import annotations

import argparse
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DATASET = "GLBX.MDP3"          # CME Globex MDP 3.0 -- the only feed for Globex
# ES/NQ/RTY proxy SPY/QQQ/IWM, which are the three rules that came out ROBUST
# in check_fill_sensitivity. `.c.0` is the front-month continuous contract.
CONT = ["ES.c.0", "NQ.c.0", "RTY.c.0"]
PARENT = ["ES.FUT", "NQ.FUT", "RTY.FUT"]


def _key():
    k = os.getenv("DB_KEY") or os.getenv("DATABENTO_API_KEY")
    if not k:
        raise SystemExit("set DB_KEY (or DATABENTO_API_KEY) in .env")
    return k


def _fmt(c):
    return f"${c:,.2f}" if isinstance(c, (int, float)) else str(c)


def main():
    import databento as db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2024-08-20", help="backtest window start")
    ap.add_argument("--end", default="2026-08-21", help="lake coverage end")
    a = ap.parse_args()

    c = db.Historical(_key())
    print(f"  key loaded (len={len(_key())}, prefix={_key()[:5]}…)\n")

    print("=" * 78)
    print("  DATASET RANGE")
    print("=" * 78)
    try:
        r = c.metadata.get_dataset_range(dataset=DATASET)
        print(f"  {DATASET}: {r}")
    except Exception as e:
        print(f"  ! {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print("  COST BY SCOPE  (metadata calls are FREE; nothing is purchased here)")
    print("=" * 78)

    probes = [
        ("1 day  ES only          mbo", ["ES.c.0"], "mbo", "2026-08-20", "2026-08-21", "continuous"),
        ("1 day  ES+NQ+RTY        mbo", CONT, "mbo", "2026-08-20", "2026-08-21", "continuous"),
        ("1 week ES+NQ+RTY        mbo", CONT, "mbo", "2026-08-17", "2026-08-21", "continuous"),
        ("1 mo   ES+NQ+RTY        mbo", CONT, "mbo", "2026-07-21", "2026-08-21", "continuous"),
        ("FULL   ES+NQ+RTY        mbo", CONT, "mbo", a.start, a.end, "continuous"),
        # cheaper shapes, for comparison -- mbp-10 is aggregated (NOT order-level)
        ("FULL   ES+NQ+RTY     mbp-10", CONT, "mbp-10", a.start, a.end, "continuous"),
        ("FULL   ES+NQ+RTY     trades", CONT, "trades", a.start, a.end, "continuous"),
        ("FULL   ES+NQ+RTY       tbbo", CONT, "tbbo", a.start, a.end, "continuous"),
    ]

    for label, syms, schema, s, e, stype in probes:
        try:
            cost = c.metadata.get_cost(dataset=DATASET, symbols=syms, schema=schema,
                                       start=s, end=e, stype_in=stype)
            size = None
            try:
                size = c.metadata.get_billable_size(dataset=DATASET, symbols=syms,
                                                    schema=schema, start=s, end=e,
                                                    stype_in=stype)
            except Exception:
                pass
            gb = f"  {size/1e9:>8.2f} GB" if isinstance(size, (int, float)) else ""
            print(f"  {label:32} {_fmt(cost):>12}{gb}")
        except Exception as ex:
            print(f"  {label:32} ! {type(ex).__name__}: {str(ex)[:70]}")

    # ---- can we cut cost by taking only RTH, and only the days we trade? ----
    print("\n" + "=" * 78)
    print("  COST LEVERS  (billing is by DATA VOLUME, so both of these are real)")
    print("=" * 78)
    print("  CME trades ~23h/day but our rules only fire 09:30-16:00 ET, so most of")
    print("  a full-session pull is data we would never look at.")
    for label, s, e in [
        ("full session (1d, ES+NQ+RTY)", "2026-08-20T00:00", "2026-08-21T00:00"),
        ("RTH only 13:30-20:00 UTC   ", "2026-08-20T13:30", "2026-08-20T20:00"),
        ("cash open 13:30-15:00 UTC  ", "2026-08-20T13:30", "2026-08-20T15:00"),
    ]:
        try:
            cost = c.metadata.get_cost(dataset=DATASET, symbols=CONT, schema="mbo",
                                       start=s, end=e, stype_in="continuous")
            size = c.metadata.get_billable_size(dataset=DATASET, symbols=CONT,
                                                schema="mbo", start=s, end=e,
                                                stype_in="continuous")
            print(f"  {label:30} {_fmt(cost):>10}  {size/1e9:>7.2f} GB")
        except Exception as ex:
            print(f"  {label:30} ! {type(ex).__name__}: {str(ex)[:60]}")

    print("\n  Per-symbol RTH day rate (pull only the symbol whose rule traded):")
    for sym in CONT:
        try:
            cost = c.metadata.get_cost(dataset=DATASET, symbols=[sym], schema="mbo",
                                       start="2026-08-20T13:30", end="2026-08-20T20:00",
                                       stype_in="continuous")
            print(f"    {sym:10} {_fmt(cost):>8} / RTH day")
        except Exception as ex:
            print(f"    {sym:10} ! {str(ex)[:60]}")

    print("\n" + "-" * 78)
    print("  READ THIS BEFORE SPENDING")
    print("  * `mbo` is the ONLY order-level schema. `mbp-10` is aggregated depth")
    print("    (price levels, not orders) -- it CANNOT answer the iceberg/queue/")
    print("    order-lifecycle questions and is not a substitute.")
    print("  * $125 of free credits may cover a meaningful slice. Start with the")
    print("    smallest scope that can carry the walk-forward, not the full window.")
    print("  * A live subscription is NOT needed to run the research. Buy history,")
    print("    test under the same battery, and only then decide about live.")


if __name__ == "__main__":
    main()
