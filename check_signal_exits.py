# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_signal_exits.py
=====================
MOMENTUM-BASED EXITS vs THE DEPLOYED TRAIL.

THE DEFECT BEING TARGETED
    A trail of T only rises above the entry once peak ROE exceeds 1/(1-T)-1, so
    the deployed 50% trail cannot exit GREEN unless the trade peaked over +100%
    -- which check_peak_profit measured at only 25.5% of trades. Live on
    2026-09-14 IWM peaked +84% and +55% on two trades and booked -14% and -25%:
    both mathematically incapable of a gain.

    A momentum exit has NO DEAD ZONE. It fires on the turn regardless of how far
    up the position is, so it can book +50% on a trade that peaked at +84%.

THE COUNTERVAILING RISK, which is why this needs measuring and not arguing
    check_giveback showed that capping winners DESTROYS expectancy here: the
    book's P&L is a right tail (top 1% of trades = 64% of total). A momentum
    exit truncates that tail too. The question is purely whether it saves more
    in dead-zone losses than it gives up in cut-short winners.

THE RULES (user's, stated so they can fail; direction-adjusted for PUTs)
    rsi3       RSI(14) on 1-min underlying closes falling on 3 consecutive bars
    emacross   EMA(5) below EMA(9) on the 1-min underlying close
    either     whichever of the two fires first
    trail+sig  the deployed trail kept as a BACKSTOP, signal exits earlier
    (references: trail50 as deployed, and hold-to-EOD)

    Indicators are computed on the UNDERLYING, not the option: the signal is
    about direction, and 1-min option bars are far too noisy (wide quotes, thin
    prints) to carry an RSI worth reading.

HONEST FRAMING
    These were proposed after watching one session's tape, and NO HOLDOUT
    REMAINS (the pre-sample was spent 2026-09-12). So a win here is
    hypothesis-generating for FORWARD PAPER, not a deployment case. What makes
    it worth running at all is that RSI(14) and 5/9 EMA are conventional
    a-priori indicators rather than fitted parameters, and they target a
    mechanically identified defect rather than a pattern mined from returns.

CORRECTNESS
    The custom exit engine is verified against sim_core on the two policies
    sim_core can express (EOD and trail50) before any signal rule is scored;
    the run aborts on divergence (METHODOLOGY 1).

Usage:
  python check_signal_exits.py --tickers IWM --dirs CALL
  python check_signal_exits.py --tickers SPY QQQ IWM --dirs CALL PUT
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import SLICE_EDGES
import sim_core

PRE_LO = pd.Timestamp("2023-10-12").date()
DEPLOY_LO = SLICE_EDGES[0]
SPLIT = pd.Timestamp("2025-08-21").date()
HIST = "historical"


def indicators(tk):
    """{date: (mods, rsi_falling3, ema5_below9)} on 1-minute underlying closes."""
    d = pd.read_parquet(f"{HIST}/{tk}.parquet")
    d.columns = [c.lower() for c in d.columns]
    et = pd.to_datetime(d["start_time"], utc=True).dt.tz_convert("America/New_York")
    mod = et.dt.hour * 60 + et.dt.minute
    m = (mod >= 570) & (mod <= 960)
    f = pd.DataFrame({"date": et[m].dt.date.values, "mod": mod[m].values,
                      "close": d["close"][m].astype(float).values}).sort_values(
        ["date", "mod"])
    out = {}
    for dt_, g in f.groupby("date"):
        c = g["close"].to_numpy()
        if len(c) < 20:
            continue
        # Wilder RSI(14)
        delta = np.diff(c, prepend=c[0])
        up = np.where(delta > 0, delta, 0.0)
        dn = np.where(delta < 0, -delta, 0.0)
        au = pd.Series(up).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        ad = pd.Series(dn).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        rs = np.divide(au, ad, out=np.full_like(au, np.inf), where=ad > 0)
        rsi = 100 - 100 / (1 + rs)
        dr = np.diff(rsi, prepend=rsi[0])
        fall3 = np.zeros(len(c), bool)
        fall3[3:] = (dr[1:-2] < 0) & (dr[2:-1] < 0) & (dr[3:] < 0)
        rise3 = np.zeros(len(c), bool)
        rise3[3:] = (dr[1:-2] > 0) & (dr[2:-1] > 0) & (dr[3:] > 0)
        e5 = pd.Series(c).ewm(span=5, adjust=False).mean().to_numpy()
        e9 = pd.Series(c).ewm(span=9, adjust=False).mean().to_numpy()
        out[dt_] = (g["mod"].to_numpy(), fall3, rise3, e5 < e9, e5 > e9)
    return out


def run_exit(path, eod_m, mode, ind_day, direction, trail=0.50):
    """-> pnl fraction for one candidate under `mode`.

    THIS MIRRORS sim_core.simulate's CONVENTIONS EXACTLY, and the differences
    are not cosmetic -- the first draft got all five wrong and the verification
    caught a 4.57 vs 1.60 divergence on a single trade:
      * entry is the `bot` fill  min(mid+0.01, ask), NOT the mid
      * levels are struck off `ref` = the entry MID, not off the fill
      * the running peak tracks the bar CLOSE, not the bid
      * a level exit triggers intrabar on `lo[i]`, and cannot fill BETTER than
        the level that fired it
      * COMMISSION_PCT is netted, and EOD fires on the FIRST bar at/after eod_m
    Only the SIGNAL rules are new; everything else must match or the comparison
    is against a different simulator rather than a different exit.
    """
    import directional_flow_backtester as D
    e_mid, e_ask, cl, hi, lo, bid, ask, mods = path
    if e_mid <= 0:
        return None
    n = len(cl)
    if n < 2:
        return None
    entry = min(round(e_mid + 0.01, 2), round(e_ask, 2))
    ref = e_mid

    sig = np.zeros(n, bool)
    if mode in ("rsi3", "emacross", "either", "trail+sig") and ind_day is not None:
        imods, fall3, rise3, below, above = ind_day
        idx = np.clip(np.searchsorted(imods, np.asarray(mods, int)),
                      0, len(imods) - 1)
        s_rsi, s_ema = ((fall3[idx], below[idx]) if direction == "CALL"
                        else (rise3[idx], above[idx]))
        sig = (s_rsi if mode == "rsi3" else
               s_ema if mode == "emacross" else (s_rsi | s_ema))

    def _out(i, lvl_hint, tag):
        px = bid[i] if bid[i] > 0 else min(lvl_hint, cl[i])
        if tag in ("stop", "trail"):
            px = min(px, lvl_hint)
        if ask is not None and bid[i] > 0 and ask[i] > bid[i]:
            sp = ask[i] - bid[i]
            px = max(0.01, px - sp * (0.5 if px > entry else 1.5))
        # the exit MINUTE is returned too, so the caller can apply the
        # one-position-per-ticker guard. Without it only the as-screened
        # population is scorable -- and exits have already been shown to FLIP
        # SIGN between populations (momentum exits: +2.7pp bare, -12pp gated),
        # so the sequential book is the only one a live conclusion can rest on.
        return (px - entry) / entry - D.COMMISSION_PCT, int(mods[i])

    use_trail = mode in ("trail50", "trail+sig") and trail > 0
    use_sig = mode in ("rsi3", "emacross", "either", "trail+sig")
    peak = ref
    for i in range(n):
        if mods[i] >= eod_m:
            return _out(i, cl[i], "eod")
        if use_trail:
            lvl = peak * (1 - trail)
            if lo[i] <= lvl:
                return _out(i, lvl, "trail")
        if use_sig and sig[i]:
            return _out(i, cl[i], "sig")
        peak = max(peak, cl[i])
    return _out(n - 1, cl[-1], "eod")


def summarise(rows, label):
    if not rows:
        print(f"    {label:14} (none)")
        return
    d = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows], float)
    w = []
    for lo, hi in ((PRE_LO, DEPLOY_LO), (DEPLOY_LO, SPLIT), (SPLIT, SLICE_EDGES[-1])):
        m = (d >= lo) & (d < hi)
        w.append(p[m].mean() * 100 if m.sum() else np.nan)
    print(f"    {label:14} n={len(p):>5}  all {p.mean()*100:>+7.1f}%  "
          f"win {(p>0).mean():.2f}  PRE {w[0]:>+7.1f}  IS {w[1]:>+7.1f}  "
          f"OOS {w[2]:>+7.1f}  tot {p.sum():>+8.1f}")


MODES = ("eod", "trail50", "rsi3", "emacross", "either", "trail+sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["IWM"])
    ap.add_argument("--dirs", nargs="*", default=["CALL", "PUT"])
    ap.add_argument("--pct", type=int, default=65)
    ap.add_argument("--rule", nargs="*", default=None,
                    help="score DEPLOYED config rules by name (gates and all) "
                         "instead of bare ticker x direction cells")
    a = ap.parse_args()

    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for

    res = {m: [] for m in MODES}
    seq = {m: [] for m in MODES}

    if a.rule:
        from config import RULES
        for name in a.rule:
            dep = next((r for r in RULES if r["name"] == name), None)
            if dep is None:
                raise SystemExit(f"no rule named {name!r}")
            tk, direction = dep["ticker"], dep["direction"]
            ind = indicators(tk)
            flow = _flow_for(D, [tk])
            if flow.empty:
                continue
            trigs = D.triggers_for(flow, tk)
            D.annotate_flow_pct(trigs, dep.get("flow_window_days", 60))
            cand = sim_core.build_candidates(D, dep, trigs=trigs, since=None)
            if not cand:
                continue
            em = sim_core.eod_mod(dep)
            # the rule's OWN deployed exit, for reference
            res.setdefault("DEPLOYED", [])
            seq.setdefault("DEPLOYED", [])
            pol = sim_core.policy_for(dep, __import__("config").TRAIL_PCT)
            for d_, p_ in sim_core.walk(cand, pol, em, fill="bot"):
                seq["DEPLOYED"].append((d_, p_))
            for d_, m_, path in cand:
                res["DEPLOYED"].append(
                    (d_, sim_core.simulate(path, pol, em, fill="bot")[0]))
            # SEQUENTIAL: one position per ticker at a time (bot_runner.py:1288).
            # Applied per MODE, because which fills survive the guard depends on
            # when the exit fires -- a faster exit frees the ticker sooner and
            # admits trades the deployed trail would have blocked. That
            # interaction is part of what the exit does, so it must be re-run
            # rather than held fixed.
            for mode in MODES:
                busy, cur = -1, None
                for d_, m_, path in cand:
                    if d_ != cur:
                        cur, busy = d_, -1
                    r = run_exit(path, em, mode, ind.get(d_), direction)
                    if r is None:
                        continue
                    res[mode].append((d_, r[0]))
                    if m_ < busy:
                        continue
                    seq[mode].append((d_, r[0]))
                    busy = r[1]
            print(f"  {name}: {len(cand)} candidates", flush=True)

        for label, store in (("SEQUENTIAL (the live guard)", seq),
                             ("AS-SCREENED (every trigger)", res)):
            print(f"\n{'='*112}\n  SIGNAL EXITS ON DEPLOYED RULES -- {label}: "
                  f"{', '.join(a.rule)}\n{'='*112}")
            for m in ["DEPLOYED"] + list(MODES):
                summarise(store.get(m, []), m)
            if not store.get("DEPLOYED"):
                continue
            base = np.array([v for _, v in store["DEPLOYED"]], float)
            print(f"  vs the rule's OWN deployed exit:")
            for m in MODES:
                if not store.get(m):
                    continue
                v = np.array([x for _, x in store[m]], float)
                print(f"    {m:14} {(v.mean()-base.mean())*100:>+7.1f}pp per trade")
        print(f"\n  READ THE SEQUENTIAL BLOCK for any live conclusion. Exits have")
        print(f"  already flipped sign between these two populations once today.")
        print(f"  NOT A DEPLOYMENT CASE: no holdout remains.")
        return

    for tk in a.tickers:
        ind = indicators(tk)
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, 60)
        for direction in a.dirs:
            rule = {"name": f"{tk} {direction}", "ticker": tk,
                    "direction": direction, "dte": [0, 1],
                    "min_flow_pct": a.pct, "target_roe": 1.0, "rr": 1.0}
            cand = sim_core.build_candidates(D, rule, trigs=trigs, since=None)
            if not cand:
                continue
            em = sim_core.eod_mod(rule)

            # verify the engine against sim_core where sim_core can express it
            for chk, pol in (("eod", dict(name="eod", kind="fixed", tp=99.0, stop=None)),
                             ("trail50", dict(name="t", kind="trail", trail=0.50))):
                mine, ref = [], []
                for d_, m_, path in cand[:400]:
                    v = run_exit(path, em, chk, ind.get(d_), direction)
                    if v is None:
                        continue
                    mine.append(v[0])
                    ref.append(sim_core.simulate(path, pol, em, fill="bot")[0])
                if mine and not np.allclose(mine, ref, atol=1e-9):
                    bad = int(np.argmax(np.abs(np.array(mine) - np.array(ref))))
                    raise SystemExit(
                        f"ABORT: exit engine diverged from sim_core on {tk} "
                        f"{direction} mode={chk}: {mine[bad]:.6f} vs {ref[bad]:.6f}")

            for d_, m_, path in cand:
                day = ind.get(d_)
                for mode in MODES:
                    r = run_exit(path, em, mode, day, direction)
                    if r is not None:
                        res[mode].append((d_, r[0]))
        print(f"  {tk} done", flush=True)

    print(f"\n{'='*112}")
    print(f"  SIGNAL EXITS vs THE DEPLOYED TRAIL  "
          f"(p{a.pct}, {' '.join(a.tickers)}, {' '.join(a.dirs)})")
    print(f"{'='*112}")
    for m in MODES:
        summarise(res[m], m)
    base = np.array([v for _, v in res["trail50"]], float)
    print(f"\n  vs trail50 (the deployed exit):")
    for m in MODES:
        if m == "trail50" or not res[m]:
            continue
        v = np.array([x for _, x in res[m]], float)
        print(f"    {m:14} {(v.mean()-base.mean())*100:>+7.1f}pp per trade")
    print(f"\n  NOT A DEPLOYMENT CASE: no holdout remains, and these rules were")
    print(f"  proposed after watching a session. Read a win as a candidate for")
    print(f"  FORWARD PAPER testing. Check PRE/IS/OOS agree before believing any")
    print(f"  of it -- an exit that only helps in one window is fitted to it.")


if __name__ == "__main__":
    main()

