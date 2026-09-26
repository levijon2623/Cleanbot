# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_fill_calibration.py
=========================
IS THE `bot` FILL MODEL RIGHT?  -- the first test of it against real NBBO depth.

WHY THIS MATTERS MORE THAN ANY SINGLE RULE
    sim_core.FILL_MODELS["bot"] prices an exit at
        bid - spread x (0.5 if winning else 1.5)
    Those cushions were chosen as plausible pessimism and have never been
    checked against data, because until the 2026-09-01 tape widening there was
    no depth to check them with. Every backtest in this project rests on them.
    The AVGO verdict in particular -- a signal whose 96%-peak-positive edge is
    entirely consumed by friction -- is a statement about this model as much as
    about AVGO.

WHAT THE NEW FIELDS ALLOW
    /option-contract/{id}/flow now returns nbbo_bid_size and nbbo_ask_size
    alongside nbbo_bid / nbbo_ask and the executed price. So for every print:
      * how wide was the market, as a share of mid
      * how much size was resting, against the 1-2 contracts we actually send
      * WHERE IN THE SPREAD the trade actually executed
    The third is the calibration. If small sells routinely print AT the bid,
    then `bid - 1.5 x spread` is charging a cushion the tape says is not paid.

🚨 COVERAGE LIMIT, and it is a hard one
    The depth fields exist only from 2026-09-01 (confirmed by range-request:
    2026-08-21 = 40 columns, 2026-09-11 = 47, 2026-09-14 = 49; history is NOT
    rewritten). So this calibrates the MODEL on recent sessions. It cannot
    re-price the 2024-2026 backtest directly -- but a corrected cushion can then
    be applied to it, which is the point.

WHAT IS NOT CLAIMED
    The prints are other participants' fills, not ours. Trade direction is
    inferred from where the price sits relative to the quote, which misclassifies
    trades that arrive between updates. So this bounds the cushion; it does not
    prove what OUR order would have received. Read the SELL-side rows as
    "what the tape shows sellers getting", not as a promise.

Usage:
  python check_fill_calibration.py
  python check_fill_calibration.py --since 2026-09-01 --max-contracts 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

LOG = "bot_executions_log.jsonl"
API = "https://api.unusualwhales.com/api"


def head():
    load_dotenv(".env", encoding="utf-8-sig")
    k = os.getenv("UW_API_KEY")
    if not k:
        sys.exit("  UW_API_KEY not in env")
    return {"Authorization": f"Bearer {k}", "Accept": "application/json",
            "User-Agent": "cleanbot-fillcal/1.0", "UW-CLIENT-API-ID": "100003"}


def traded_contracts(since):
    """The instruments the bot ACTUALLY touched -- calibrating on anything else
    would measure a different liquidity profile than the one we pay."""
    out = {}
    for line in open(LOG, encoding="utf-8-sig"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        c = s.rfind("}")
        try:
            r = json.loads(s[:c + 1])
        except Exception:
            continue
        d = str(r.get("timestamp", ""))[:10]
        con = r.get("contract")
        if con and d >= since:
            out.setdefault((con, d), r.get("ticker"))
    return out


def fetch(h, con, date, cap=40):
    rows, page = [], 0
    while page < cap:
        r = httpx.get(f"{API}/option-contract/{con}/flow", headers=h,
                      params={"date": date, "limit": 500, "page": page}, timeout=60)
        if r.status_code != 200:
            break
        d = r.json().get("data", [])
        if not d:
            break
        rows += d
        page += 1
        if len(d) < 500:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-01")
    ap.add_argument("--max-contracts", type=int, default=10)
    ap.add_argument("--min-mid", type=float, default=0.30,
                    help="ignore prints below this mid -- a decayed option quoted "
                         "penny-wide at $0.035 is a 28%% spread and nothing like "
                         "the $0.60-3.50 range the bot actually enters at")
    a = ap.parse_args()

    h = head()
    want = traded_contracts(a.since)
    if not want:
        print(f"  no ledger contracts since {a.since}"); return
    keys = sorted(want)[:a.max_contracts]
    print(f"  calibrating on {len(keys)} traded contracts since {a.since}\n")

    recs = []
    for con, date in keys:
        rows = fetch(h, con, date)
        n_new = sum(1 for x in rows if x.get("nbbo_bid_size") is not None)
        print(f"    {con} {date}: {len(rows):>6} prints, {n_new:>6} with depth")
        for x in rows:
            try:
                b, k = float(x["nbbo_bid"]), float(x["nbbo_ask"])
                p, sz = float(x["price"]), int(x["size"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (k > b > 0):
                continue
            bs, ks = x.get("nbbo_bid_size"), x.get("nbbo_ask_size")
            mid = (b + k) / 2.0
            recs.append(dict(ticker=want[(con, date)], contract=con, date=date,
                             price=p, size=sz, bid=b, ask=k, mid=mid,
                             spread=k - b, spread_pct=(k - b) / mid * 100,
                             pos=(p - b) / (k - b),
                             bid_size=(int(bs) if bs is not None else np.nan),
                             ask_size=(int(ks) if ks is not None else np.nan)))
    if not recs:
        print("  nothing usable"); return
    T_all = pd.DataFrame(recs)
    T_all.to_parquet("_fill_calibration.parquet", index=False)
    T = T_all[T_all["mid"] >= a.min_mid].copy()
    print(f"\n  {len(T_all):,} prints with a two-sided quote; "
          f"{len(T):,} at mid >= ${a.min_mid:.2f} (the tradeable range)")

    print(f"\n  SPREAD BY OPTION PRICE -- why the filter is not optional")
    print(f"  {'mid bucket':16} {'prints':>8} {'med spread $':>13} {'med spread %':>13}")
    for lo, hi, lbl in ((0, .10, "< $0.10"), (.10, .30, "$0.10-0.30"),
                        (.30, 1.0, "$0.30-1.00"), (1.0, 3.0, "$1.00-3.00"),
                        (3.0, 1e9, "> $3.00")):
        g = T_all[(T_all["mid"] >= lo) & (T_all["mid"] < hi)]
        if g.empty:
            continue
        print(f"  {lbl:16} {len(g):>8} {g['spread'].median():>13.2f} "
              f"{g['spread_pct'].median():>12.1f}%")
    print(f"  -> a percentage spread on a near-worthless option is not a cost we pay;")
    print(f"     the bot never enters there. Everything below uses mid >= "
          f"${a.min_mid:.2f}.")

    print(f"\n{'='*92}")
    print(f"  1. HOW WIDE IS THE MARKET, REALLY?  (spread as % of mid)")
    print(f"{'='*92}")
    print(f"  {'ticker':7} {'prints':>8} {'p25':>7} {'median':>8} {'p75':>7} "
          f"{'p90':>7} {'median $':>9}")
    for tk, g in T.groupby("ticker"):
        print(f"  {str(tk):7} {len(g):>8} {g['spread_pct'].quantile(.25):>7.2f} "
              f"{g['spread_pct'].median():>8.2f} {g['spread_pct'].quantile(.75):>7.2f} "
              f"{g['spread_pct'].quantile(.90):>7.2f} {g['spread'].median():>9.2f}")

    print(f"\n{'='*92}")
    print(f"  2. IS THERE DEPTH FOR OUR ORDER?  (we send 1-2 contracts)")
    print(f"{'='*92}")
    D = T.dropna(subset=["bid_size"])
    if D.empty:
        print("  no depth fields on these dates")
    else:
        print(f"  {'ticker':7} {'prints':>8} {'bid sz p10':>11} {'median':>8} "
              f"{'ask sz med':>11} {'>=1 lot':>9} {'>=10 lots':>10}")
        for tk, g in D.groupby("ticker"):
            print(f"  {str(tk):7} {len(g):>8} {g['bid_size'].quantile(.10):>11.0f} "
                  f"{g['bid_size'].median():>8.0f} {g['ask_size'].median():>11.0f} "
                  f"{(g['bid_size']>=1).mean()*100:>8.0f}% "
                  f"{(g['bid_size']>=10).mean()*100:>9.0f}%")
        print(f"  -> depth is only a constraint if the '>=1 lot' column is not ~100%.")

    print(f"\n{'='*92}")
    print(f"  3. WHERE IN THE SPREAD DO TRADES ACTUALLY PRINT?")
    print(f"     pos = (price - bid) / (ask - bid).  0.0 = at the bid, 1.0 = at the ask")
    print(f"{'='*92}")
    print(f"  {'ticker':7} {'prints':>8} {'at/below bid':>13} {'lower half':>11} "
          f"{'upper half':>11} {'at/above ask':>13} {'median pos':>11}")
    for tk, g in T.groupby("ticker"):
        print(f"  {str(tk):7} {len(g):>8} {(g['pos']<=0.001).mean()*100:>12.0f}% "
              f"{((g['pos']>0.001)&(g['pos']<0.5)).mean()*100:>10.0f}% "
              f"{((g['pos']>=0.5)&(g['pos']<0.999)).mean()*100:>10.0f}% "
              f"{(g['pos']>=0.999).mean()*100:>12.0f}% {g['pos'].median():>11.2f}")

    print(f"\n{'='*92}")
    print(f"  4. THE CALIBRATION -- what cushion does the tape actually support?")
    print(f"{'='*92}")
    print(f"  Small SELLS only (size <= 5, printing in the lower half of the spread),")
    print(f"  measured as (bid - price) / spread: 0 = filled at the bid,")
    print(f"  +1.0 = a full spread WORSE than the bid.")
    sells = T[(T["size"] <= 5) & (T["pos"] < 0.5)].copy()
    sells["cushion"] = (sells["bid"] - sells["price"]) / sells["spread"]
    print(f"  {'ticker':7} {'n':>7} {'median':>8} {'p75':>7} {'p90':>7} "
          f"{'share <=0':>10}")
    for tk, g in sells.groupby("ticker"):
        if len(g) < 30:
            continue
        print(f"  {str(tk):7} {len(g):>7} {g['cushion'].median():>8.2f} "
              f"{g['cushion'].quantile(.75):>7.2f} {g['cushion'].quantile(.90):>7.2f} "
              f"{(g['cushion']<=0).mean()*100:>9.0f}%")
    if len(sells) >= 30:
        print(f"  {'ALL':7} {len(sells):>7} {sells['cushion'].median():>8.2f} "
              f"{sells['cushion'].quantile(.75):>7.2f} "
              f"{sells['cushion'].quantile(.90):>7.2f} "
              f"{(sells['cushion']<=0).mean()*100:>9.0f}%")
    print(f"\n  sim_core FILL_MODELS currently charges:")
    print(f"    winning exit  0.5 x spread")
    print(f"    losing exit   1.5 x spread     <-- compare against the medians above")
    print(f"  A median well BELOW 0.5 means the model is too pessimistic and every")
    print(f"  backtest here is understating returns; ABOVE 1.5 means the opposite.")
    print(f"  Caveat: these are other participants' prints and the side is INFERRED")
    print(f"  from quote position, so treat this as a bound, not as our fill.")


if __name__ == "__main__":
    main()
