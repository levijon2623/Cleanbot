"""
check_flip_direction.py
=======================
"Switch put with call and we'll be rich." Take every deployed rule's OWN
trigger minutes -- after all of its gates -- and buy the OPPOSITE option at
those minutes. Same expiry rule, same strike offset, same exit, same fills.
Pre-registered 2026-10-01, the day SMH LOWVOL PUT bought puts into a rally
four times running (paper).

    python check_flip_direction.py --selftest
    python check_flip_direction.py

ARMS (per rule, through sim_core exactly as check_core_book scores the book:
build_candidates -> walk(policy_for(rule, TRAIL_PCT), eod_mod(rule), fill="bot"))
    ORIG    the deployed rule as is
    FLIP    the same minutes, the other right (CALL <-> PUT)
    RANDOM  x20  each minute independently takes ORIG's or FLIP's contract
            with p = 0.5 -- the "direction carries no information" placebo.
            If the trigger's direction is informative, ORIG > RANDOM > FLIP;
            if it is ANTI-informative, FLIP > RANDOM > ORIG.
    The comparison runs on COMMON minutes (both rights pass the $0.50 entry
    floor), so the only thing that differs is the instrument. The unrestricted
    ORIG book is printed beside it as a sanity check against check_core_book.

PRE-REGISTERED CRITERIA -- BOOK = every enabled rule, pooled (incl. paper-only)
    F1  OOS per-trade mean, FLIP - ORIG                         >= +5pp
    F2  IS per-trade mean, FLIP - ORIG                          >= 0
    F3  OOS total, FLIP - ORIG                                  >= 0
    F4  FLIP OOS mean > p95 of the 20 RANDOM arms
    F5  FLIP itself: >= 5 of 6 calendar slices populated AND positive (C3)
    F6  F1's sign holds under fill = mid, bot AND worst
    F7  FLIP beats ORIG OOS in >= 2/3 of the rules (cross-sectional, C5)
    PASS = F1..F7.

PER RULE (incl. SMH LOWVOL PUT) -- EXPLORATORY
    F1, F2, F3, F4, F6 per rule with a day-block 90% CI, and >= 30 OOS trades.
    Nine-plus rules is nine-plus comparisons, so a lone per-rule "pass" is a
    thread for a fresh-data test, never a config change. SMH's fill band is
    61.5pp (METHODOLOGY 1a): its verdict at any single fill is an assumption.

REPORTED, NOT SCORED
    ORIG vs the RANDOM band (does the deployed direction beat a coin?), each
    arm's IS/OOS/win/total, coverage (share of ORIG minutes with a FLIP
    contract), minutes where the flipped pick fell to a later dte.

MACHINERY (--selftest, METHODOLOGY 7 "can it move?")
    M1  the permissive stub path, fed ORIG's own minutes and direction,
        reproduces ORIG's candidates exactly -- so FLIP differs ONLY by the right
    M2  every FLIP contract is the opposite right of the ORIG contract
    M3  RANDOM with p = 1 equals ORIG and with p = 0 equals FLIP, trade for trade
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC   # noqa: E402

N_RANDOM = 20
SEED = 20261001
FILLS = ("mid", "bot", "worst")
OPP = {"CALL": "PUT", "PUT": "CALL"}


def _right(cid):
    """'C' or 'P' from an OCC-style contract id (char 9 from the end)."""
    s = str(cid)
    return s[-9] if len(s) >= 9 and s[-9] in "CP" else None


def build_at(D, rule, minutes, direction):
    """Candidates for `direction` at exactly these (date, mod) minutes, through
    build_candidates with every trigger-selection gate made a no-op -- the
    minutes were ALREADY selected by the real rule. Keeps dte list, strike
    offset and the $0.50 floor, which belong to the contract pick."""
    stub_rule = dict(name=f"{rule['name']} @{direction}", ticker=rule["ticker"],
                     direction=direction, hours=list(range(9, 15)),
                     dte=rule.get("dte", [0, 1]), flow_abs=0.0,
                     strike_offset=rule.get("strike_offset", 0),
                     target_roe=rule["target_roe"], rr=rule["rr"])
    trigs = []
    for d, m in minutes:
        ts = pd.Timestamp(dt.datetime.combine(d, dt.time(m // 60, m % 60)))
        trigs.append(dict(date=d, ts=ts.to_datetime64(), hour=m // 60,
                          dir=direction, abs_flow=1.0))
    meta = []
    c = SC.build_candidates(D, stub_rule, trigs=trigs, meta_out=meta)
    return c, meta


def rule_arms(D, rule):
    meta_o = []
    orig = SC.build_candidates(D, rule, meta_out=meta_o)
    mins = [(c[0], c[1]) for c in orig]
    flip, meta_f = build_at(D, rule, mins, OPP[rule["direction"]])
    fk = {(c[0], c[1]): (c, m) for c, m in zip(flip, meta_f)}
    common_o, common_f, mo, mf = [], [], [], []
    for c, m in zip(orig, meta_o):
        k = (c[0], c[1])
        if k in fk:
            common_o.append(c)
            mo.append(m)
            common_f.append(fk[k][0])
            mf.append(fk[k][1])
    return dict(orig_all=orig, orig=common_o, flip=common_f, meta_o=mo, meta_f=mf,
                coverage=(len(common_o) / len(orig)) if orig else float("nan"),
                later_dte=sum(1 for a, b in zip(mo, mf) if a["dte"] != b["dte"]))


def random_arm(orig, flip, rng, p=0.5):
    return [o if rng.random() < p else f for o, f in zip(orig, flip)]


def walk(rule, cands, fill="bot"):
    from config import TRAIL_PCT
    return SC.walk(cands, SC.policy_for(rule, TRAIL_PCT), SC.eod_mod(rule), fill=fill)


def m(rows, oos):
    v = [r[1] for r in rows if (r[0] >= SC.SPLIT) == oos]
    return (float(np.mean(v)) if v else float("nan")), float(np.sum(v)), len(v)


def boot(a, b, n=2000, seed=SEED):
    """90% day-block CI of OOS mean(b) - mean(a); days resampled intact."""
    rng = np.random.default_rng(seed)
    da, db = {}, {}
    for d, x, *_ in a:
        if d >= SC.SPLIT:
            da.setdefault(d, []).append(x)
    for d, x, *_ in b:
        if d >= SC.SPLIT:
            db.setdefault(d, []).append(x)
    days = sorted(set(da) | set(db))
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days))
        xa = [x for i in pick for x in da.get(days[i], [])]
        xb = [x for i in pick for x in db.get(days[i], [])]
        if xa and xb:
            out.append(np.mean(xb) - np.mean(xa))
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95))) if out else (np.nan, np.nan)


def rules():
    from config import RULES
    rs = [r for r in RULES if r.get("enabled", True)]
    return sorted(rs, key=lambda r: r["ticker"])      # one-slot bar cache: ticker outermost


def selftest(D, rule, arms):
    fails = []

    def ok(label, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        cond or fails.append(label)
    mins = [(c[0], c[1]) for c in arms["orig_all"]]
    same, _ = build_at(D, rule, mins, rule["direction"])
    key = lambda cs: [(c[0], c[1], c[2][0], len(c[2][2])) for c in cs]
    ok(f"M1 {rule['name']}: stub path at ORIG's minutes == ORIG ({len(mins)} cands)",
       key(same) == key(arms["orig_all"]))
    ro = {_right(x["cid"]) for x in arms["meta_o"]}
    pairs = all(_right(a["cid"]) and _right(b["cid"]) and _right(a["cid"]) != _right(b["cid"])
                for a, b in zip(arms["meta_o"], arms["meta_f"]))
    ok(f"M2 {rule['name']}: every FLIP contract is the other right (ORIG rights {ro})", pairs)
    rng = np.random.default_rng(SEED)
    r1 = walk(rule, random_arm(arms["orig"], arms["flip"], rng, p=1.0))
    r0 = walk(rule, random_arm(arms["orig"], arms["flip"], rng, p=0.0))
    ok(f"M3 {rule['name']}: RANDOM p=1 == ORIG and p=0 == FLIP",
       r1 == walk(rule, arms["orig"]) and r0 == walk(rule, arms["flip"]))
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    import directional_flow_backtester as D
    rs = rules()
    arms, ok_all = {}, True
    for r in rs:
        arms[r["name"]] = rule_arms(D, r)
        print(f"  built {r['name']:18} ORIG {len(arms[r['name']]['orig_all']):5}  common "
              f"{len(arms[r['name']]['orig']):5}  coverage {arms[r['name']]['coverage']:.0%}  "
              f"later-dte {arms[r['name']]['later_dte']}", flush=True)
        ok_all &= selftest(D, r, arms[r["name"]])
    if not ok_all:
        sys.exit("machinery checks failed -- not scoring")
    if a.selftest:
        return print("\n  machinery checks pass")

    rng = np.random.default_rng(SEED)
    res = {}
    for r in rs:
        A = arms[r["name"]]
        res[r["name"]] = dict(
            orig_all=walk(r, A["orig_all"]), orig=walk(r, A["orig"]), flip=walk(r, A["flip"]),
            rand=[walk(r, random_arm(A["orig"], A["flip"], rng)) for _ in range(N_RANDOM)],
            fills={f: (walk(r, A["orig"], f), walk(r, A["flip"], f)) for f in FILLS})
    pool = lambda key: sorted(x for v in res.values() for x in v[key])

    print("\n=== PER RULE (exploratory) -- common minutes, fill=bot")
    print(f"  {'rule':18} {'n':>4} {'ORIG IS':>8} {'ORIG OOS':>9} {'FLIP IS':>8} {'FLIP OOS':>9} "
          f"{'d OOS':>7} {'90% CI':>14} {'RAND p5..p95':>15}  verdict")
    beats = 0
    for r in rs:
        R = res[r["name"]]
        oi, oo, fi, fo = m(R["orig"], False), m(R["orig"], True), m(R["flip"], False), m(R["flip"], True)
        rd = [m(x, True)[0] for x in R["rand"]]
        lo, hi = boot(R["orig"], R["flip"])
        d = fo[0] - oo[0]
        beats += d > 0
        fs = [m(R["fills"][f][1], True)[0] - m(R["fills"][f][0], True)[0] for f in FILLS]
        crit = [d >= 0.05, fi[0] - oi[0] >= 0, fo[1] - oo[1] >= 0,
                fo[0] > np.nanpercentile(rd, 95), all(x > 0 for x in fs), fo[2] >= 30]
        print(f"  {r['name']:18} {oo[2]:4} {oi[0]*100:+7.1f}% {oo[0]*100:+8.1f}% {fi[0]*100:+7.1f}% "
              f"{fo[0]*100:+8.1f}% {d*100:+6.1f} [{lo*100:+5.1f},{hi*100:+5.1f}] "
              f"[{np.nanpercentile(rd, 5)*100:+5.1f},{np.nanpercentile(rd, 95)*100:+5.1f}]  "
              f"{'pass*' if all(crit) else 'fail'}  ({''.join('1' if c else '0' for c in crit)})")
    print("  (* exploratory: one of many rules; criteria F1 F2 F3 F4 F6 n>=30 in that order)")

    O, F = pool("orig"), pool("flip")
    rand = [sorted(x for v in res.values() for x in v["rand"][i]) for i in range(N_RANDOM)]
    oi, oo, fi, fo = m(O, False), m(O, True), m(F, False), m(F, True)
    rd = [m(x, True)[0] for x in rand]
    sF = SC.stat(F)
    fs = {f: m(sorted(x for v in res.values() for x in v["fills"][f][1]), True)[0]
          - m(sorted(x for v in res.values() for x in v["fills"][f][0]), True)[0] for f in FILLS}
    lo, hi = boot(O, F)
    d = fo[0] - oo[0]
    print("\n=== BOOK -- every enabled rule pooled, common minutes")
    for lbl, rows in (("ORIG (all minutes)", pool("orig_all")), ("ORIG", O), ("FLIP", F)):
        s = SC.stat(rows)
        print(f"  {lbl:19} n={s['n']:5} d={s['nd']:4}  IS {s['is_']*100:+6.1f}%  OOS {s['oos']*100:+6.1f}% "
              f"(n {s['noos']})  win {s['win']:.2f}  OOS total {m(rows, True)[1]*100:+8.0f}  "
              f"slices +{s['nposs']}/{s['npop']}")
    print(f"  RANDOM x{N_RANDOM}         OOS p5 {np.percentile(rd, 5)*100:+.1f}%  p50 {np.median(rd)*100:+.1f}%  "
          f"p95 {np.percentile(rd, 95)*100:+.1f}%   (ORIG {oo[0]*100:+.1f}% -- does direction beat a coin?)")
    crit = [("F1 OOS mean FLIP - ORIG >= +5pp", d >= 0.05, f"{d*100:+.1f}pp (90% CI {lo*100:+.1f}..{hi*100:+.1f})"),
            ("F2 IS mean FLIP - ORIG >= 0", fi[0] - oi[0] >= 0, f"{(fi[0]-oi[0])*100:+.1f}pp"),
            ("F3 OOS total FLIP - ORIG >= 0", fo[1] - oo[1] >= 0, f"{(fo[1]-oo[1])*100:+.0f}"),
            ("F4 FLIP OOS > p95 of RANDOM", fo[0] > np.percentile(rd, 95), f"{fo[0]*100:+.1f}% vs {np.percentile(rd, 95)*100:+.1f}%"),
            ("F5 FLIP >= 5/6 slices positive", sF["npop"] >= 5 and sF["nposs"] >= 5, f"+{sF['nposs']}/{sF['npop']}"),
            ("F6 sign holds mid/bot/worst", d > 0 and all(v > 0 for v in fs.values()),
             " ".join(f"{k} {v*100:+.1f}" for k, v in fs.items())),
            ("F7 FLIP beats ORIG in >= 2/3 rules", beats >= -(-2 * len(rs) // 3), f"{beats}/{len(rs)}")]
    for lbl, okk, note in crit:
        print(f"  {'PASS' if okk else 'FAIL'}  {lbl:36} {note}")
    print(f"  FLIP TEST: {'PASS' if all(c[1] for c in crit) else 'FAIL'}")


if __name__ == "__main__":
    main()
