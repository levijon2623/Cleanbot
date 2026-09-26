# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_exit_sweep.py
===================
The one major parameter never swept: THE EXIT.

In this codebase `stop_roe = target_roe / rr`, so the deployed `target_roe 1.0
/ rr 1.0` puts the stop at entry*(1 - 1.0) = ZERO. Nine of eleven live rules
have NO STOP -- they ride to +100% or to the EOD flatten. That also explains
why 1:1 always won a win-rate sort during the original hand-tuning: with no
stop the only way to lose is to decay toward worthless, so win rate is
mechanically maximised while the loss TAIL is unbounded.

Given check_signal_quality showed the trigger has ~no directional edge on the
underlying, the option-side P&L is coming from the bracket -- which makes the
exit the highest-leverage untested parameter in the book.

Swept, all on the SEQUENTIAL fill model (one position per ticker; a tighter
stop frees the ticker earlier, so re-entry timing is re-walked per policy):

  fixed      TP x hard stop grid (stop expressed as an absolute % of premium)
  trail      give back X% from the running peak, with an initial disaster stop
  atr        trail by k x ATR of the OPTION's own 1-min true range
  combo      TP + trailing (bank the spike, else trail)

Reported per policy across the blended book: n, expectancy, IS/OOS, win, maxLL,
6 calendar slices, and TOTAL P&L (a tighter stop that raises expectancy but
cuts total return is not an improvement).

Usage:
  python check_exit_sweep.py
  python check_exit_sweep.py --per-rule
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _eod_mod(r):
    ef = r.get("eod_flatten")
    if ef:
        h, m = ef.split(":")
        return int(h) * 60 + int(m)
    return 15 * 60 + 55


def _atr(hi, lo, cl, n=14):
    prev = np.concatenate(([cl[0]], cl[:-1]))
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev), np.abs(lo - prev)))
    out = np.empty_like(tr)
    c = 0.0
    for i, x in enumerate(tr):
        c = x if i == 0 else c + (x - c) / n
        out[i] = c
    return out


def _sim(path, policy, eod_m):
    """path = (entry, cl, hi, lo, mod). Returns (pnl, exit_mod)."""
    import directional_flow_backtester as D
    entry, cl, hi, lo, mod = path
    n = len(cl)
    kind = policy["kind"]
    tp = entry * (1 + policy["tp"]) if policy.get("tp") else None
    hard = entry * (1 - policy["stop"]) if policy.get("stop") else None
    peak = entry
    atr = _atr(hi, lo, cl, policy.get("atr_n", 14)) if kind == "atr" else None
    for i in range(n):
        if mod[i] >= eod_m:
            px = cl[i]
            return (px - entry) / entry - D.COMMISSION_PCT, int(mod[i])
        # stop first (conservative: assume the low is hit before the close)
        lvl = hard
        if kind == "trail":
            t = peak * (1 - policy["trail"])
            lvl = t if lvl is None else max(lvl, t)
        elif kind == "atr":
            t = peak - policy["k"] * atr[i]
            lvl = t if lvl is None else max(lvl, t)
        if lvl is not None and lo[i] <= lvl:
            px = min(lvl, cl[i]) if i else lvl
            return (px - entry) / entry - D.COMMISSION_PCT, int(mod[i])
        if tp is not None and cl[i] >= tp:
            return policy["tp"] - D.COMMISSION_PCT, int(mod[i])
        peak = max(peak, cl[i])
    px = cl[-1]
    return (px - entry) / entry - D.COMMISSION_PCT, int(mod[-1])


def _policies():
    P = []
    for tp in (0.5, 0.75, 1.0, 1.5):
        for st in (0.30, 0.40, 0.50, 0.65, None):
            P.append(dict(name=f"fixed tp{int(tp*100)}/sl{int(st*100) if st else 'none'}",
                          kind="fixed", tp=tp, stop=st))
    for tr in (0.20, 0.30, 0.40, 0.50):
        for st in (0.50, None):
            P.append(dict(name=f"trail {int(tr*100)}%/init{int(st*100) if st else 'none'}",
                          kind="trail", tp=None, trail=tr, stop=st))
    for tr in (0.30, 0.40):
        P.append(dict(name=f"combo tp100+trail{int(tr*100)}",
                      kind="trail", tp=1.0, trail=tr, stop=0.50))
    for k in (1.5, 2.0, 3.0):
        for st in (0.50, None):
            P.append(dict(name=f"atr {k}x/init{int(st*100) if st else 'none'}",
                          kind="atr", tp=None, k=k, stop=st))
    P.append(dict(name="atr 2.0x + tp100", kind="atr", tp=1.0, k=2.0, stop=0.50))
    return P


def _line(lbl, pnls):
    if len(pnls) < 20:
        return f"    {lbl:26} n={len(pnls):>4}  (thin)"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    c = m = 0
    for _, p in sorted(pnls):
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    npop = sum(1 for b in sl if len(b) >= 4)
    return (f"    {lbl:26} n={len(v):>4}  {v.mean()*100:>+6.1f}%  "
            f"IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  "
            f"win {(v>0).mean():.2f}  maxLL {m:>2}  tot {v.sum():>+7.2f}  pop {npop}/6")


def run(a):
    import directional_flow_backtester as D
    from check_config_walkforward import _flow_for
    from config import RULES
    from amt_profile import amt_open_map, amt_ok

    POL = _policies()
    rules = [r for r in RULES if r.get("enabled", True)]
    if a.tickers:
        keep = {t.upper() for t in a.tickers}
        rules = [r for r in rules if r["ticker"].upper() in keep]

    book = {p["name"]: [] for p in POL}
    book["DEPLOYED (as-is)"] = []
    per_rule = {}

    for r in rules:
        tk = r["ticker"]
        flow = _flow_for(D, [tk])
        if flow.empty:
            continue
        gex = D.load_gex(HIST, tk); vol = D.load_volume_regime(HIST, tk); trd = D.load_trend_regime(HIST, tk)
        _d = set(gex) & set(vol) & set(trd)
        amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
        reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, r.get("flow_window_days", 60))
        try:
            tb = D._ticker_bars(tk)
        except Exception:
            tb = None
        if tb is None or tb.empty:
            _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
        if tb is None or tb.empty:
            continue
        bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
        bbd = {d: g for d, g in tb.groupby("date")}
        amt = amt_open_map(tk) if r.get("amt_open") else {}
        matched = D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)
        if r.get("amt_open"):
            matched = [(t, th) for t, th in matched if amt_ok(r["amt_open"], amt.get(t["date"]))]
        matched.sort(key=lambda x: pd.Timestamp(x[0]["ts"]))
        em = _eod_mod(r)
        dtes = r.get("dte", [0, 1])
        tr_dep, rr_dep = float(r["target_roe"]), float(r["rr"])

        # build the full path (incl. HIGH, which _option_paths drops) once per trigger
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
            for dd in dtes:
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
            entry = (b + k) / 2.0 if b > 0 else float(er["close"])
            if entry < 0.50:
                continue
            fwd = ent[ent["minute_et"] > ts].sort_values("minute_et")
            if len(fwd) < 3:
                continue
            pm = fwd["minute_et"]
            path = (entry, fwd["close"].to_numpy(float), fwd["high"].to_numpy(float),
                    fwd["low"].to_numpy(float),
                    (pm.dt.hour.values * 60 + pm.dt.minute.values).astype(int))
            cand.append((d, pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute, path))

        for pol in POL + [dict(name="DEPLOYED (as-is)", kind="fixed", tp=tr_dep,
                               stop=(tr_dep / rr_dep if tr_dep / rr_dep < 1.0 else None))]:
            cur, busy, out = None, -1, []
            for d, m, path in cand:
                if d != cur:
                    cur, busy = d, -1
                if m < busy:
                    continue
                pnl, xm = _sim(path, pol, em)
                out.append((d, pnl))
                busy = xm
            book[pol["name"]] += out
            per_rule.setdefault(r["name"], {})[pol["name"]] = out

    print("=" * 122)
    print("  EXIT-POLICY SWEEP   sequential fills   split " + str(SPLIT))
    print("  NOTE: deployed tr=1.0/rr=1.0 => stop at entry*(1-1.0) = 0 = NO STOP")
    print("=" * 122)
    print(_line("DEPLOYED (as-is)", book["DEPLOYED (as-is)"]))
    base_oos = np.mean([p for d, p in book["DEPLOYED (as-is)"] if d >= SPLIT])

    for grp, pref in (("FIXED tp x stop", "fixed"), ("TRAILING", "trail"),
                      ("COMBO", "combo"), ("ATR TRAIL", "atr")):
        print(f"\n  -- {grp} --")
        rows = [(p["name"], book[p["name"]]) for p in POL if p["name"].startswith(pref)]
        rows = [(n, v) for n, v in rows if len(v) >= 20]
        for n, v in sorted(rows, key=lambda x: -np.mean([p for d, p in x[1] if d >= SPLIT])):
            print(_line(n, v))

    print("\n  -- TOP 8 by OOS expectancy --")
    allp = [(n, v) for n, v in book.items() if len(v) >= 20]
    for n, v in sorted(allp, key=lambda x: -np.mean([p for d, p in x[1] if d >= SPLIT]))[:8]:
        print(_line(n, v))
    print("\n  -- TOP 8 by TOTAL P&L --")
    for n, v in sorted(allp, key=lambda x: -np.sum([p for _, p in x[1]]))[:8]:
        print(_line(n, v))

    if a.per_rule:
        print("\n" + "=" * 122)
        print("  PER-RULE: deployed vs that rule's best OOS policy")
        for rn, d in per_rule.items():
            dep = d.get("DEPLOYED (as-is)", [])
            if len(dep) < 10:
                continue
            dep_oos = np.mean([p for dd, p in dep if dd >= SPLIT])
            best = max(((n, v) for n, v in d.items() if len(v) >= 10),
                       key=lambda x: np.mean([p for dd, p in x[1] if dd >= SPLIT]))
            bo = np.mean([p for dd, p in best[1] if dd >= SPLIT])
            print(f"    {rn:22} deployed OOS {dep_oos*100:>+6.1f}%  ->  "
                  f"{best[0]:26} OOS {bo*100:>+6.1f}%  (+{(bo-dep_oos)*100:.1f}pp)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--per-rule", action="store_true")
    a = ap.parse_args()
    run(a)
