# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
export_trade_tape.py
====================
Exports every simulated trade -- with its OPTION price path, the underlying's
path for the same session, and the entry/exit marks -- to one JSON file for the
trade viewer.

WHY THE OPTION PATH AND NOT JUST THE UNDERLYING
    The bot's P&L is the option's, and the option is where the behaviour that
    matters is visible: the trail ratchet, the dead zone below +100% peak ROE,
    the gap between the mid the bracket is priced off and the bid it exits at.
    A chart of the underlying alone cannot show why a trade that "looked right"
    still lost 50%.

WHAT IS IN EACH TRADE
    entry/exit minute and price, realised pnl, exit tag (trail/eod/tp/stop),
    the per-minute close/bid/ask of the contract from entry to exit, the peak
    and the trailing-stop line implied by the rule's policy, the underlying's
    1-minute closes for the whole session, and the day's regime tags.

The trades are the ones the SEQUENTIAL walk actually took (bot_runner.py:1288,
one position per ticker), recovered via sim_core.walk(picks_out=...) -- not
every candidate trigger, most of which the live bot could never have filled.

Default rule set is `sim_core.research_rules()` -- the 5-rule working set, with
MSFT/SMH/AVGO/GLD excluded (they remain enabled and paper-trading in config).

Usage:
  python export_trade_tape.py                     # -> _trade_tape.json
  python export_trade_tape.py --include-paper -o all.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import sim_core

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _underlying(tk: str) -> dict:
    """{date -> (mods, closes)} RTH 1-minute closes."""
    d = pd.read_parquet(f"{HIST}/{tk}.parquet")
    d.columns = [c.lower() for c in d.columns]
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York")
    mod = et.dt.hour * 60 + et.dt.minute
    m = (mod >= 570) & (mod <= 960)
    f = pd.DataFrame({"date": et[m].dt.date.values, "mod": mod[m].values,
                      "close": d["close"][m].astype(float).values})
    out = {}
    for dt, g in f.sort_values("mod").groupby("date"):
        out[dt] = (g["mod"].tolist(), [round(x, 4) for x in g["close"]])
    return out


def _trail_line(mids: np.ndarray, trail: float) -> list:
    """The trailing stop the policy implies, minute by minute, from the running
    peak. Drawn so the DEAD ZONE is visible: a 50% trail only rises above the
    entry once peak ROE exceeds +100%, so every trade peaking between 0% and
    +100% exits at a loss by construction (sim_core.policy_for)."""
    peak = np.maximum.accumulate(mids)
    return [round(float(x), 4) for x in peak * (1.0 - trail)]


def build(include_paper: bool, max_path: int):
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    rules = sim_core.research_rules(include_paper)
    gexd, vold, trdd, und = {}, {}, {}, {}
    trades = []

    for r in rules:
        tk = r["ticker"]
        meta: list[dict] = []
        cand = sim_core.build_candidates(D, r, meta_out=meta)
        if not cand:
            continue
        pol = sim_core.policy_for(r, TRAIL_PCT)
        em = sim_core.eod_mod(r)
        trail = float(r.get("trail_pct", TRAIL_PCT) or 0.0)
        picks: list[tuple] = []
        rows = sim_core.walk(cand, pol, em, fill="bot", with_tags=True,
                             picks_out=picks)
        if tk not in und:
            und[tk] = _underlying(tk)
            gexd[tk] = D.load_gex(HIST, tk)
            vold[tk] = D.load_volume_regime(HIST, tk)
            trdd[tk] = D.load_trend_regime(HIST, tk)

        for (d, pnl, tag), (ci, xm) in zip(rows, picks):
            _, mod, path = cand[ci]
            mt = meta[ci]
            e_mid, e_ask, cl, hi, lo, bid, ask, pmods = path
            # clip the stored path at the exit minute -- everything after is
            # not part of this trade and would misread as an opportunity missed
            n = int(np.searchsorted(pmods, xm, side="right")) or len(pmods)
            n = min(n, max_path)
            mids = (np.asarray(bid[:n], float) + np.asarray(ask[:n], float)) / 2.0
            mids = np.where(mids > 0, mids, np.asarray(cl[:n], float))
            u = und[tk].get(d, ([], []))
            trades.append(dict(
                date=str(d), ticker=tk, rule=r["name"], dir=r["direction"],
                dte=mt["dte"], strike=round(mt["strike"], 2),
                spot=round(mt["spot"], 2),
                entry_mod=int(mod), exit_mod=int(xm),
                entry=round(float(e_mid), 4),
                entry_bid=round(float(mt["bid"]), 4),
                entry_ask=round(float(mt["ask"]), 4),
                pnl=round(float(pnl), 6), tag=tag,
                half="IS" if d < SPLIT else "OOS",
                trail_pct=trail,
                mods=[int(x) for x in pmods[:n]],
                mid=[round(float(x), 4) for x in mids],
                bid=[round(float(x), 4) for x in bid[:n]],
                ask=[round(float(x), 4) for x in ask[:n]],
                trail_line=_trail_line(mids, trail) if trail > 0 else None,
                u_mods=u[0], u_px=u[1],
                gex=gexd[tk].get(d), vol=vold[tk].get(d), trend=trdd[tk].get(d),
            ))

    trades.sort(key=lambda t: (t["date"], t["entry_mod"]))
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="_trade_tape.json")
    ap.add_argument("--include-paper", action="store_true")
    ap.add_argument("--max-path", type=int, default=420)
    a = ap.parse_args()
    tr = build(a.include_paper, a.max_path)
    meta = dict(n=len(tr), generated=str(pd.Timestamp.now().date()),
                split=str(SPLIT),
                rules=sorted({t["rule"] for t in tr}),
                include_paper=a.include_paper)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, trades=tr), f, separators=(",", ":"))
    import os
    print(f"  wrote {a.out}  {len(tr)} trades, "
          f"{os.path.getsize(a.out)/1e6:.2f} MB")
    print(f"  rules: {', '.join(meta['rules'])}")
    print(f"  dates: {tr[0]['date']} .. {tr[-1]['date']}  "
          f"({len({t['date'] for t in tr})} sessions)")
    w = [t for t in tr if t["pnl"] > 0]
    print(f"  mean {np.mean([t['pnl'] for t in tr])*100:+.1f}%  "
          f"win {len(w)/len(tr):.2f}  tags: "
          + ", ".join(f"{k}={sum(1 for t in tr if t['tag']==k)}"
                      for k in sorted({t['tag'] for t in tr})))


if __name__ == "__main__":
    main()
