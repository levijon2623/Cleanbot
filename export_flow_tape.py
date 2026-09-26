# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0", "polars>=1.0.0"]
# ///
"""
export_flow_tape.py
===================
Emits one JSON per session for the local flow viewer: the underlying's candles,
the CUMULATIVE NET PREMIUM series the trigger is actually computed on, its
EMA(5), every crossover with its gate verdict, and the trades the sequential
walk took.

WHY LOCAL AND NOT TRADINGVIEW
    TradingView cannot ingest this. Pine Seeds is EOD-only and closed to new
    repositories; Pine Script has no HTTP. check_tv_proxy measured the fallback
    -- rebuilding the trigger from OHLCV that Pine CAN compute -- and the
    random-sign placebo beat the real proxies. So the series below has to be
    shipped to the chart, not recomputed in it.

WHAT THE VIEWER GETS THAT A PRICE CHART CANNOT SHOW
    `thr` is the magnitude gate: the trailing-60d percentile of this ticker's
    own trigger |cum flow|, computed from days STRICTLY BEFORE this one.
    check_flow_threshold established that it is load-bearing -- dropping it to
    p20 costs -9,899 ROE and the marginal band runs -10.6/trade against the
    deployed band's +12.6 -- so the viewer draws it as a band and every
    crossover is labelled with which gates it cleared. Seeing the refused
    crossovers is the point; a chart of only the taken trades cannot show why
    the others were right to refuse.

🚨 TIMEZONE
    historical/{T}.parquet stores minute_et tz-AWARE America/New_York; the flow
    frame, the trigger list and the option bars are tz-NAIVE ET. Everything here
    is normalised to naive ET first, then stamped with `_epoch`, which localises
    to UTC so that Lightweight Charts -- which renders UTC -- prints the ET wall
    clock. Do not "fix" that by using .timestamp() on a naive value: that reads
    the SERVER's timezone and silently shifts every session.

🚨 THE GATE MIRROR IS SELF-CHECKING
    `gate_flags` recomputes the per-gate verdicts for display, which is a second
    copy of logic that already lives in _rule_matched_trigs -- exactly the drift
    METHODOLOGY 1 warns about. So it is ASSERTED against that function's
    authoritative output on every run: if all-flags-true ever disagrees with
    "this trigger was matched", the export fails loudly instead of drawing a
    plausible lie. Keep the assertion.

Usage:
  python export_flow_tape.py                       # -> tape/
  python export_flow_tape.py --include-paper --tickers SPY QQQ
  python export_flow_tape.py --only-traded         # sessions with a fill only
  python -m http.server 8765                       # then open flow_viewer.html
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import polars as pl

import sim_core

HIST = "historical"
RTH0, RTH1 = 570, 960


def _epoch(d, mod):
    """Naive ET -> the epoch second Lightweight Charts should render.

    LWC draws UTC. Stamping naive ET AS IF it were UTC makes the axis print the
    ET wall clock, which is what every other number in this repo is quoted in.
    """
    ts = pd.Timestamp(d) + pd.Timedelta(minutes=int(mod))
    return int(ts.tz_localize("UTC").timestamp())


def bars_for(tk):
    df = (pl.scan_parquet(f"{HIST}/{tk}.parquet")
          .select("date", "minute_et", "open", "high", "low", "close", "volume")
          .collect().to_pandas())
    df["minute_et"] = pd.to_datetime(df["minute_et"]).dt.tz_localize(None)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    m = df["minute_et"].dt.hour * 60 + df["minute_et"].dt.minute
    df["mod"] = m
    df = df[(m >= RTH0) & (m <= RTH1)].sort_values(["date", "mod"])
    return {d: g for d, g in df.groupby("date")}


def flow_for(D, tk):
    """{date -> DataFrame(mod, cum, ema)} -- the trigger's actual input."""
    from check_config_walkforward import _flow_for
    f = _flow_for(D, [tk])
    if f.empty:
        return {}
    g = f[f["underlying_symbol"] == tk].copy()
    ts = pd.to_datetime(g["minute_et"])
    g["date"] = ts.dt.date
    g["mod"] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)
    out = {}
    for d, x in g.groupby("date"):
        x = x.sort_values("mod").copy()
        # span=5, adjust=False -- identical to triggers_for, or the drawn
        # crossovers will not be the ones the bot fired on. The EMA is computed
        # on the FULL session before clipping to RTH, because that is what
        # triggers_for sees; clipping first would re-seed it and move crossings.
        x["ema"] = x["cum_flow"].ewm(span=5, adjust=False).mean()
        x = x[(x["mod"] >= RTH0) & (x["mod"] <= RTH1)]
        out[d] = x[["mod", "cum_flow", "ema"]]
    return out


def gate_flags(rule, t, gex, vol, trd, amp, reg_src):
    """Per-gate verdict, for DISPLAY. Asserted against _rule_matched_trigs."""
    d = t["date"]
    f = {}
    f["dir"] = t["dir"] == rule["direction"].upper()
    hours = set(rule.get("hours", range(24)))
    f["hour"] = t["hour"] in hours and t["hour"] < 15
    reg = rule.get("regime")
    if reg is None:
        f["regime"] = True
    else:
        regs = reg if isinstance(reg, (list, tuple)) else [reg]
        ok = False
        for rg in regs:
            val = (gex if rg.endswith("_GEX") else reg_src[rg]).get(d)
            ok = (val == "NEGATIVE" if rg == "NEGATIVE_GEX" else
                  val == "POSITIVE" if rg == "POSITIVE_GEX" else val == rg)
            if ok:
                break
        f["regime"] = ok
    am = rule.get("amp_min")
    f["amp"] = True if am is None else amp.get(d, -1) >= am
    fp = rule.get("min_flow_pct", rule.get("flow_pct"))
    tm = t.get("thr")
    if fp is not None:
        f["flow"] = bool(tm) and int(fp) in tm and t["abs_flow"] >= tm[int(fp)]
    elif rule.get("flow_abs") is not None:
        f["flow"] = t["abs_flow"] >= float(rule["flow_abs"])
    else:
        f["flow"] = False
    # numpy bools reach here via abs_flow comparisons and json cannot encode
    # them. Cast at the source rather than with a default= hook, so the
    # assertion below compares plain bools too.
    return {k: bool(v) for k, v in f.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out-dir", default="tape")
    ap.add_argument("--include-paper", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--only-traded", action="store_true")
    ap.add_argument("--max-path", type=int, default=420)
    a = ap.parse_args()

    import directional_flow_backtester as D
    import export_trade_tape as ETT

    rules = sim_core.research_rules(a.include_paper)
    if a.tickers:
        rules = [r for r in rules if r["ticker"] in set(a.tickers)]
    if not rules:
        print("  no rules match"); return

    print("  building trades (sequential walk, same as the book)...", flush=True)
    trades = ETT.build(a.include_paper, a.max_path)
    if a.tickers:
        trades = [t for t in trades if t["ticker"] in set(a.tickers)]
    tmap = {}
    for t in trades:
        tmap.setdefault((t["ticker"], t["date"]), []).append(t)

    os.makedirs(a.out_dir, exist_ok=True)
    sessions, n_assert = [], 0
    for rule in rules:
        tk = rule["ticker"]
        print(f"  {rule['name']}...", flush=True)
        B, F = bars_for(tk), flow_for(D, tk)
        gex = D.load_gex(HIST, tk)
        vol = D.load_volume_regime(HIST, tk)
        trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL")
               + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
                   "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}

        from check_config_walkforward import _flow_for
        raw = _flow_for(D, [tk])
        trigs = D.triggers_for(raw, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))
        # AUTHORITATIVE pass set -- the same call build_candidates makes.
        matched = {(t["date"], pd.Timestamp(t["ts"]).hour * 60
                    + pd.Timestamp(t["ts"]).minute)
                   for t, _th in D._rule_matched_trigs(rule, trigs, gex, vol,
                                                       trd, amp, reg_src)}
        byday = {}
        for t in trigs:
            byday.setdefault(t["date"], []).append(t)

        for d in sorted(set(B) & set(F)):
            if d < sim_core.DEPLOYED_START:
                continue
            tr = tmap.get((tk, str(d)), [])
            if a.only_traded and not tr:
                continue
            g, fl = B[d], F[d]
            dts = byday.get(d, [])
            thr = next((t["thr"] for t in dts if t.get("thr")), None)

            # Minutes the sequential walk actually ENTERED on. A gate-passing
            # crossover that is not one of these was refused by the
            # one-position guard, not by a gate -- 6,400 pass the gates book-
            # wide and 392 become fills, so that middle state is 94% of
            # qualifying signals and the chart drew it identically to a fill.
            entered = {int(x["entry_mod"]) for x in tr}

            tj = []
            for t in dts:
                mod = (pd.Timestamp(t["ts"]).hour * 60
                       + pd.Timestamp(t["ts"]).minute)
                f = gate_flags(rule, t, gex, vol, trd, amp, reg_src)
                allp = all(f.values())
                # the self-check described in the module docstring
                assert allp == ((d, mod) in matched), (
                    f"gate mirror drifted: {tk} {d} {mod} flags={f} "
                    f"matched={(d, mod) in matched}")
                n_assert += 1
                tj.append(dict(t=_epoch(d, mod), dir=t["dir"],
                               flow=round(float(t["abs_flow"])),
                               pass_=allp, gates=f,
                               # "gated" | "guard" | "taken"
                               outcome=("taken" if (allp and mod in entered)
                                        else "guard" if allp else "gated")))

            payload = dict(
                ticker=tk, date=str(d), rule=rule["name"],
                direction=rule["direction"],
                regime=dict(gex=gex.get(d), vol=vol.get(d), trend=trd.get(d),
                            amp=amp.get(d)),
                gate=dict(min_flow_pct=rule.get("min_flow_pct"),
                          hours=rule.get("hours"), regime=rule.get("regime"),
                          amp_min=rule.get("amp_min")),
                thr={str(k): round(float(v)) for k, v in (thr or {}).items()},
                bars=[dict(t=_epoch(d, m), o=round(o, 4), h=round(h, 4),
                           l=round(lo, 4), c=round(c, 4), v=float(v))
                      for m, o, h, lo, c, v in zip(
                          g["mod"], g["open"], g["high"], g["low"],
                          g["close"], g["volume"])],
                flow=[dict(t=_epoch(d, m), cum=round(float(c)),
                           ema=round(float(e)))
                      for m, c, e in zip(fl["mod"], fl["cum_flow"], fl["ema"])],
                triggers=tj,
                trades=[dict(
                    rule=x["rule"], dir=x["dir"], tag=x["tag"],
                    pnl=x["pnl"], strike=x["strike"], dte=x["dte"],
                    entry_t=_epoch(d, x["entry_mod"]),
                    exit_t=_epoch(d, x["exit_mod"]),
                    entry=x["entry"], trail_pct=x["trail_pct"],
                    path=dict(
                        t=[_epoch(d, m) for m in x["mods"]],
                        mid=x["mid"], bid=x["bid"], ask=x["ask"],
                        trail=x["trail_line"])) for x in tr],
            )
            fn = f"{tk}_{d}.json"
            with open(os.path.join(a.out_dir, fn), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            sessions.append(dict(
                file=fn, ticker=tk, date=str(d), rule=rule["name"],
                n_trig=len(tj), n_pass=sum(1 for x in tj if x["pass_"]),
                n_trade=len(tr),
                pnl=round(sum(x["pnl"] for x in tr) * 100, 1),
                vol=vol.get(d), trend=trd.get(d), gex=gex.get(d)))

    sessions.sort(key=lambda s: (s["date"], s["ticker"]))
    idx = dict(generated=str(pd.Timestamp.now()),
               split=str(ETT.SPLIT), n=len(sessions),
               rules=sorted({s["rule"] for s in sessions}),
               sessions=sessions)
    with open(os.path.join(a.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f, separators=(",", ":"))

    mb = sum(os.path.getsize(os.path.join(a.out_dir, f))
             for f in os.listdir(a.out_dir)) / 1e6
    print(f"\n  wrote {len(sessions)} sessions to {a.out_dir}/  ({mb:.1f} MB)")
    print(f"  gate mirror verified on {n_assert:,} triggers -- no drift")
    print(f"  triggers {sum(s['n_trig'] for s in sessions):,}  "
          f"passing gates {sum(s['n_pass'] for s in sessions):,}  "
          f"filled {sum(s['n_trade'] for s in sessions):,}")
    print(f"\n  serve it:  python -m http.server 8765")
    print(f"  then open: http://localhost:8765/flow_viewer.html")


if __name__ == "__main__":
    main()
