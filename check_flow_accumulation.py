# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_flow_accumulation.py
==========================
MECHANISM TEST for the "skip the first trigger" effect found 2026-09-09.

THE HYPOTHESIS (user's): flow triggers accumulate over the session into real
dealer-hedging pressure that the underlying eventually succumbs to. The first
trigger of the day fires on a thin book -- no pressure has built yet -- so it is
noise. Later triggers fire on accumulated pressure, so they work.

WHY THIS IS A DIFFERENT PREDICTION, not a restatement:
  * it is NOT wall-clock time  -- already falsified (temporal cutoffs flat:
    after 10:00 +7.3%, 10:30 +6.1%, 11:00 +2.3% vs deployed +7.1%)
  * it is NOT ordinal position per se -- ordinal is only a PROXY for accumulation
  * it IS cumulative flow magnitude AND SIGN at the trigger minute

THE GAP IN THE CURRENT SYSTEM THIS EXPOSES
------------------------------------------
`directional_flow_backtester.triggers_for` sets

    dir       = "CALL" if cum crossed ABOVE its EMA(5) else "PUT"
    abs_flow  = abs(cum[i])                      <-- SIGN DISCARDED

and `min_flow_pct` gates on the percentile of that ABSOLUTE value against a
trailing multi-day window. So the direction comes from the CROSSOVER while the
magnitude gate is sign-blind. A PUT can therefore fire while cumulative flow is
strongly POSITIVE -- net call premium merely decelerating. Under the hedging
hypothesis those are opposite states:

    PUT + cum very NEGATIVE  -> accumulated put buying, dealers short puts,
                                must SELL underlying to hedge -> real pressure
    PUT + cum POSITIVE       -> no accumulated pressure, a deceleration blip

Early in the session cum is small and its sign is unstable, so trade #1 is
disproportionately the second kind. That would explain the ordinal effect AND
why the temporal control failed.

TESTS (all sequential, realistic fills, live exit cushion modelled, IS/OOS):
  1. aligned vs opposed        sign(cum) agrees with the trade direction?
  2. cum_dir deciles           signed-in-trade-direction magnitude -- monotone?
  3. ordinal WITHIN alignment  does "skip #1" survive controlling for alignment?
  4. alignment WITHIN trade #1 does alignment RESCUE the first trigger?   <-- KEY
  5. by GEX regime             negative GEX (dealers hedge WITH the move) should
                               amplify the effect; positive GEX should damp it

Test 4 is the discriminator. If alignment rescues trade #1, the mechanism is
accumulation and the right gate is a SIGN/MAGNITUDE rule, not an ordinal hack.
If trade #1 is bad even when aligned, accumulation is not the story.

Usage:  python check_flow_accumulation.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
from check_exit_walkforward import _eod_mod
from check_giveback import _sim

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _trigs_signed(flow, ticker):
    """triggers_for, but KEEPING the sign of cumulative flow (and the EMA gap)."""
    g = flow[flow["underlying_symbol"] == ticker]
    out = []
    for d, gd in g.groupby("date"):
        gd = gd.sort_values("minute_et")
        cum = gd["cum_flow"].values
        if len(cum) < 6:
            continue
        ema = pd.Series(cum).ewm(span=5, adjust=False).mean().values
        mt = gd["minute_et"].values
        for i in range(1, len(cum)):
            bull = cum[i - 1] <= ema[i - 1] and cum[i] > ema[i]
            bear = cum[i - 1] >= ema[i - 1] and cum[i] < ema[i]
            if bull or bear:
                out.append({"date": d, "ts": mt[i], "hour": int(pd.Timestamp(mt[i]).hour),
                            "dir": "CALL" if bull else "PUT", "abs_flow": abs(cum[i]),
                            "cum": float(cum[i]), "ema_gap": float(cum[i] - ema[i])})
    return out


def _build(D, r):
    """Sequential trades for a rule, each tagged with its accumulation state."""
    from check_config_walkforward import _flow_for
    from amt_profile import amt_open_map, amt_ok
    tk = r["ticker"]
    flow = _flow_for(D, [tk])
    if flow.empty:
        return []
    gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
               "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
    trigs = _trigs_signed(flow, tk)
    D.annotate_flow_pct(trigs, r.get("flow_window_days", 60))
    try:
        tb = D._ticker_bars(tk)
    except Exception:
        tb = None
    if tb is None or tb.empty:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
    if tb is None or tb.empty:
        return []
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    bbd = {d: g for d, g in tb.groupby("date")}
    amt = amt_open_map(tk) if r.get("amt_open") else {}
    matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
    if r.get("amt_open"):
        matched = [(t, th) for t, th in matched if amt_ok(r["amt_open"], amt.get(t["date"]))]
    matched.sort(key=lambda x: pd.Timestamp(x[0]["ts"]))

    # how many triggers (actionable or not) have already fired today
    ntrig = {}
    for t in sorted(trigs, key=lambda x: pd.Timestamp(x["ts"])):
        ntrig[(t["date"], pd.Timestamp(t["ts"]))] = ntrig.get(t["date"], 0)
        ntrig[t["date"]] = ntrig.get(t["date"], 0) + 1

    cand = []
    for t, _th in matched:
        d, ts = t["date"], t["ts"]
        day = bbd.get(d)
        if day is None:
            continue
        at = day[day["minute_et"] <= ts]
        if at.empty:
            continue
        spot = float(at.iloc[-1]["underlying_close"])
        cid = None
        for dd in r.get("dte", [0, 1]):
            cid = D.pick_contract(day, ts, r["direction"], dd, spot)
            if cid is not None:
                break
        if cid is None:
            continue
        ent = bbc[cid]
        er = ent[(ent["minute_et"] <= ts) & (ent["minute_et"] >= ts - pd.Timedelta(minutes=3))]
        if er.empty:
            continue
        er = er.iloc[-1]
        b, k = float(er["bid_close"]), float(er["ask_close"])
        mid = (b + k) / 2.0 if b > 0 else float(er["close"])
        if mid < 0.50:
            continue
        fwd = ent[ent["minute_et"] > ts].sort_values("minute_et")
        if len(fwd) < 3:
            continue
        pm = fwd["minute_et"]
        sgn = 1.0 if r["direction"] == "CALL" else -1.0
        cand.append(dict(
            date=d, mod=pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute,
            path=(mid, k if k > 0 else mid, fwd["close"].to_numpy(float),
                  fwd["high"].to_numpy(float), fwd["low"].to_numpy(float),
                  fwd["bid_close"].to_numpy(float), fwd["ask_close"].to_numpy(float),
                  (pm.dt.hour.values * 60 + pm.dt.minute.values).astype(int)),
            cum=t["cum"], cum_dir=t["cum"] * sgn, abs_flow=t["abs_flow"],
            gex=gex.get(d), ticker=tk, rule=r["name"]))
    return cand


def _seq(cand, pol, em):
    """Deployed guard; returns rows with pnl + ordinal position that day."""
    cur, busy, k, out = None, -1, 0, []
    for c in cand:
        if c["date"] != cur:
            cur, busy, k = c["date"], -1, 0
        if c["mod"] < busy:
            continue
        pnl, xm, _t = _sim(c["path"], pol, em, True, True)
        r = dict(c); r.pop("path")
        r["pnl"] = pnl; r["ord"] = k + 1
        out.append(r)
        busy = xm; k += 1
    return out


def _st(lbl, df):
    if len(df) < 12:
        return f"    {lbl:32} n={len(df):>5}  (thin)"
    v = df["pnl"].to_numpy(float)
    i = df[df.date < SPLIT]["pnl"].to_numpy(float)
    o = df[df.date >= SPLIT]["pnl"].to_numpy(float)
    sl = [[] for _ in range(6)]
    for d, p in zip(df["date"], v):
        kk = _slice_idx(d)
        if kk is not None:
            sl[kk].append(p)
    pop = [np.mean(b) for b in sl if len(b) >= 3]
    return (f"    {lbl:32} n={len(v):>5} d={df['date'].nunique():>4} "
            f"IS {i.mean()*100 if len(i) else float('nan'):>+7.1f}% "
            f"OOS {o.mean()*100 if len(o) else float('nan'):>+7.1f}% "
            f"win {(v>0).mean():>4.2f} sl {sum(1 for x in pop if x>0)}/{len(pop)}")


def run(a):
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    rules = [r for r in RULES if r.get("enabled", True)]
    # per-rule deployed exit: META/NVDA carry trail_pct 0, so a book-wide
    # trail would score trades those rules never take
    import sim_core
    rows = []
    for r in rules:
        c = _build(D, r)
        if not c:
            continue
        rows += _seq(c, sim_core.policy_for(r, TRAIL_PCT), _eod_mod(r))
        print(f"  {r['name']:26} {len(c):>5} triggers")
    R = pd.DataFrame(rows)
    if R.empty:
        print("nothing"); return
    R["aligned"] = R["cum_dir"] > 0

    print("\n" + "=" * 112)
    print("  1. ALIGNMENT -- does sign(cumulative flow) agree with the trade direction?")
    print("     (the system's `dir` comes from the EMA CROSSOVER; `abs_flow` discards this sign)")
    print("=" * 112)
    print(_st("ALL trades (baseline)", R))
    print(_st("ALIGNED  (pressure agrees)", R[R.aligned]))
    print(_st("OPPOSED  (decel. blip)", R[~R.aligned]))
    print(f"\n    aligned share: {R['aligned'].mean()*100:.0f}%   "
          f"by ordinal: " + "  ".join(
              f"#{k}:{R[R['ord']==k]['aligned'].mean()*100:.0f}%" for k in (1, 2, 3)
              if (R['ord'] == k).sum() > 5))

    print("\n" + "=" * 112)
    print("  2. MAGNITUDE -- cum flow signed INTO the trade direction, by quartile")
    print("=" * 112)
    R["q"] = pd.qcut(R["cum_dir"], 4, labels=False, duplicates="drop")
    for q in sorted(R["q"].dropna().unique()):
        sub = R[R["q"] == q]
        print(_st(f"Q{int(q)+1}  cum_dir {sub['cum_dir'].min()/1e6:>+7.1f}M..{sub['cum_dir'].max()/1e6:>+7.1f}M", sub))

    print("\n" + "=" * 112)
    print("  3. ORDINAL *WITHIN* ALIGNMENT -- does 'skip #1' survive controlling for it?")
    print("=" * 112)
    for al, tag in ((True, "ALIGNED"), (False, "OPPOSED")):
        print(f"    -- {tag} --")
        for k in (1, 2, 3):
            print(_st(f"      trade #{k}", R[(R.aligned == al) & (R["ord"] == k)]))
        print(_st(f"      trade #2+", R[(R.aligned == al) & (R["ord"] >= 2)]))

    print("\n" + "=" * 112)
    print("  4. KEY TEST -- does ALIGNMENT RESCUE THE FIRST TRIGGER?")
    print("     if yes: the mechanism is accumulation, and the gate should be sign/magnitude,")
    print("     not an ordinal hack. if no: accumulation is not the story.")
    print("=" * 112)
    f = R[R["ord"] == 1]
    print(_st("trade #1  ALL", f))
    print(_st("trade #1  ALIGNED", f[f.aligned]))
    print(_st("trade #1  OPPOSED", f[~f.aligned]))
    print(_st("trade #2+ ALL", R[R["ord"] >= 2]))
    print(_st("trade #2+ ALIGNED", R[(R["ord"] >= 2) & R.aligned]))

    print("\n" + "=" * 112)
    print("  5. BY GEX REGIME -- negative GEX = dealers hedge WITH the move (amplifying),")
    print("     so the accumulation effect should be STRONGER there if the mechanism is real")
    print("=" * 112)
    for g in ("NEGATIVE", "POSITIVE"):
        sub = R[R["gex"] == g]
        if len(sub) < 12:
            continue
        print(f"    -- {g} GEX --")
        print(_st("      aligned", sub[sub.aligned]))
        print(_st("      opposed", sub[~sub.aligned]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run(ap.parse_args())
