# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_target_variance.py
========================
CAN A LOWER-VARIANCE TARGET BUY THE POWER THE BOOK DOES NOT HAVE?

THE PROBLEM THIS EXISTS TO SOLVE
    check_ivr_termstructure --test power established that power on this book is
    set by DAYS, not trades: the same conditioner on 159 OOS trades and on 1,147
    (7.2x more) gave a WORSE minimum detectable effect, 64.6pp -> 71.2pp, because
    both sit on the same 77 OOS days. Any day-level conditioner therefore has an
    MDE of ~65pp per trade. The two ways out are more calendar (wait) or a target
    with less variance than a +/-120pp option return. This tests the second.

THE CANDIDATE TARGETS
    raw            the option return itself                        (status quo)
    hit{k}         1[return >= +k%]                                bounded [0,1]
    cap{k}         min(return, +k%)   -- keeps the full downside, caps the tail
    wins{k}        clip(return, -k%, +k%)  -- symmetric winsorise
    for k in 5, 10, 25, 50.

THE TRAP THIS MUST NOT FALL INTO
    Variance reduction is worthless if it throws away the signal with the noise,
    and on THIS book that is the live risk, not a hypothetical one: the ROE
    give-back took win rate 0.38 -> 0.59 and DESTROYED expectancy, because the
    edge is a right tail (~38% win rate x ~+90% mean winner). A target that caps
    winners at +5% is measuring the half of the distribution that carries no
    edge. So variance is never reported alone -- every target is scored on

        SNR = |effect| / day-block SE

    against a KNOWN, REAL contrast, not on its SD. A target only wins if it
    raises SNR. Lower SD with lower SNR is a worse target, not a better one.

WHAT IS REPORTED
  --test dist     the distribution, bucketed at +/-5/10/25/50%, and how much of
                  total P&L lives in each bucket (the tail-dependence check)
  --test mde      per target: mean, SD, day-block SE of a half-vs-half OOS
                  split, and MDE -- all in that target's own units
  --test snr      the honest comparison: SNR of each target against real
                  contrasts already established on this book, plus an
                  INJECTED-EFFECT simulation that asks the like-for-like
                  question ("if a day-level edge of size X existed, which target
                  would see it first?")

Usage:
  python check_target_variance.py --test dist
  python check_target_variance.py --test mde
  python check_target_variance.py --test snr
  python check_target_variance.py --test all
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
import sim_core

SPLIT = pd.Timestamp("2025-08-21").date()
KS = (5, 10, 25, 50)
EDGES = [-np.inf, -0.50, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50, np.inf]
LABELS = ["<= -50%", "-50..-25", "-25..-10", "-10..-5", "-5..0",
          "0..+5", "+5..+10", "+10..+25", "+25..+50", "> +50%"]


def book(fill="bot", pop="seq") -> pd.DataFrame:
    """The deployed book through sim_core: sequential fills, per-rule exits."""
    import directional_flow_backtester as D
    from config import RULES, TRAIL_PCT

    out = []
    for r in [x for x in RULES if x.get("enabled", True)]:
        cand = sim_core.build_candidates(D, r)
        if not cand:
            continue
        pol = sim_core.policy_for(r, TRAIL_PCT)
        em = sim_core.eod_mod(r)
        if pop == "seq":
            rows = sim_core.walk(cand, pol, em, fill=fill)
        else:
            rows = [(d, sim_core.simulate(p, pol, em, fill=fill)[0]) for d, m, p in cand]
        out += [(d, p, r["name"], r["ticker"]) for d, p in rows]
    df = pd.DataFrame(sorted(out), columns=["date", "pnl", "rule", "ticker"])
    df["half"] = np.where(df["date"] < SPLIT, "IS", "OOS")
    return df


def targets(pnl: np.ndarray) -> dict[str, np.ndarray]:
    """Every candidate outcome variable, all derived from the same trade.

    Two families, and the distinction is the whole point:

      TRUNCATING (hit / cap / wins) throw the right tail away. On this book
      that tail IS the edge -- the >+50% bucket alone is 379% of total P&L --
      so they are structurally blind to any effect that works by making the
      tail fatter.

      COMPRESSING (log / sqrt) shrink the tail's VARIANCE while keeping its
      ORDER and its response to amplification. Multiplying a winner by 1.5
      shifts log1p by a constant instead of being clipped away, so a
      tail-amplifying effect stays visible. This is the family that has a
      chance of beating `raw` on the worst case rather than on one guess.
    """
    t = {"raw": pnl.copy()}
    for k in KS:
        f = k / 100.0
        t[f"hit{k}"] = (pnl >= f).astype(float)
        t[f"cap{k}"] = np.minimum(pnl, f)
        t[f"wins{k}"] = np.clip(pnl, -f, f)
        # LEFT-TAIL hit rate. Signed so that "better" is always "higher",
        # like every other target here. hit{k} asks whether the trade reached
        # the right tail; loss{k} asks whether it fell into the left one, which
        # is the quantity a loss-AVOIDANCE conditioner is actually about. The
        # first sweep omitted these entirely and that was the wrong question.
        t[f"loss{k}"] = -(pnl <= -f).astype(float)
    # -100% is attainable on a long option, and log1p(-1) is -inf, so floor just
    # above it. The floor is not a free parameter: total loss is total loss.
    t["log"] = np.log1p(np.maximum(pnl, -0.99))
    t["sqrt"] = np.sign(pnl) * np.sqrt(np.abs(pnl))
    return t


# ------------------------------------------------------------------ dist
def test_dist(df: pd.DataFrame):
    p = df["pnl"].to_numpy()
    b = pd.cut(p, EDGES, labels=LABELS, right=True)
    tot = p.sum()

    print(f"  n={len(p)}  mean {p.mean()*100:+.1f}%  median {np.median(p)*100:+.1f}%  "
          f"SD {p.std()*100:.1f}pp  win {(p>0).mean():.2f}")
    print(f"  total P&L (sum of returns) = {tot:+.2f}  "
          f"[skew {pd.Series(p).skew():+.2f}, kurtosis {pd.Series(p).kurt():+.1f}]\n")
    print(f"  {'bucket':10} {'n':>5} {'share':>7} {'mean':>9} "
          f"{'sum P&L':>9} {'% of tot':>9}   {'IS n':>5} {'OOS n':>6}")
    for lab in LABELS:
        m = np.asarray(b == lab)
        if not m.sum():
            print(f"  {lab:10} {0:>5}")
            continue
        s = p[m].sum()
        print(f"  {lab:10} {m.sum():>5} {m.mean()*100:>6.1f}% "
              f"{p[m].mean()*100:>+8.1f}% {s:>+9.2f} {s/tot*100:>8.1f}%   "
              f"{(m & (df['half']=='IS').to_numpy()).sum():>5} "
              f"{(m & (df['half']=='OOS').to_numpy()).sum():>6}")

    print("\n  TAIL DEPENDENCE -- what happens to total P&L if the top tail is capped:")
    for k in KS:
        c = np.minimum(p, k / 100.0)
        print(f"    cap at +{k:>2}%:  total {c.sum():>+7.2f} "
              f"({c.sum()/tot*100:>6.1f}% of uncapped)   mean {c.mean()*100:>+6.1f}%")
    top = np.sort(p)[::-1]
    for q in (1, 5, 10, 20):
        n = max(1, int(len(p) * q / 100))
        print(f"    top {q:>2}% of trades (n={n:>3}) carry "
              f"{top[:n].sum()/tot*100:>6.1f}% of total P&L")


# ------------------------------------------------------------------ mde
def _day_se(df: pd.DataFrame, y: np.ndarray, n=3000, seed=11) -> float:
    """Day-block bootstrap SE of a half-vs-half difference in `y`.

    Resamples whole SESSIONS, then splits the resampled days into two arms.
    This is the quantity that actually governs detectability on this book --
    see METHODOLOGY.md 7, 'power is set by days, not trades'.
    """
    d = df["date"].to_numpy()
    days = np.unique(d)
    idx = {k: np.where(d == k)[0] for k in days}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days), replace=True)
        a, b = [], []
        for j, k in enumerate(pick):
            (a if j % 2 == 0 else b).append(idx[days[k]])
        if a and b:
            out.append(y[np.concatenate(a)].mean() - y[np.concatenate(b)].mean())
    return float(np.std(out))


def test_mde(df: pd.DataFrame):
    oos = df[df["half"] == "OOS"].reset_index(drop=True)
    ts = targets(oos["pnl"].to_numpy())
    print(f"  OOS trades {len(oos)}, OOS days {oos['date'].nunique()}\n")
    print(f"  {'target':9} {'mean':>9} {'SD':>9} {'day SE':>9} {'MDE':>9}   "
          f"{'MDE / SD':>9}  {'MDE as % of mean':>17}")
    for name, y in ts.items():
        se = _day_se(oos, y)
        mde = 2.8 * se
        sc = 100 if name == "raw" or name.startswith(("cap", "wins")) else 100
        print(f"  {name:9} {y.mean()*sc:>+8.2f} {y.std()*sc:>8.2f} "
              f"{se*sc:>8.2f} {mde*sc:>8.2f}   {mde/y.std():>9.2f}  "
              f"{(mde/abs(y.mean()) if y.mean() else np.nan):>16.1f}x")
    print("\n  Units: raw/cap/wins in PERCENTAGE POINTS of return; hit{k} in "
          "PERCENTAGE POINTS of hit rate.")
    print("  'MDE / SD' is the scale-free one -- it is the detectable effect in "
          "standard deviations,\n  and it is what must FALL for a target to be "
          "worth switching to.")


# ------------------------------------------------------------------ snr
def _inject(p, tm, q, mode, win, rng):
    """Apply a synthetic day-level edge of strength `q` to the treated trades.

    THE SHAPE OF THE EFFECT IS NOT KNOWN IN ADVANCE, so it is a parameter, not
    an assumption. A target that only wins under one shape is not a safe
    default -- it has been tuned to a guess about the market.

      freq   q of treated trades re-drawn from the book's WINNER pool.
             "The regime produces winners more often."  Favours targets that
             resolve near zero, because it converts losers into winners.
      size   treated trades that are ALREADY winners are multiplied by 1+2q.
             "The regime makes the right tail fatter."  Favours targets that
             keep the tail -- raw, cap50, wins50.
      shift  every treated trade is shifted by +q (in return units).
             "The regime adds a constant."  The neutral shape.
      rescue q of treated LOSERS have their loss halved.
             "The regime mostly avoids the disasters."  This is the shape a
             risk/exit-side conditioner would have, and no target that discards
             the left tail can see it.
      disaster  q of treated trades re-drawn from the book's LOSER pool.
             "This state is disaster-prone."  The mirror of `freq`, and the
             shape that matters most in practice: a SKIP rule does not need to
             predict winners, only to flag the trades that blow up. Detecting
             this is what `loss{k}` exists for.
    """
    y = p.copy()
    if q <= 0:
        return y
    if mode == "freq":
        m = tm & (rng.random(len(p)) < q)
        if m.any():
            y[m] = rng.choice(win, m.sum(), replace=True)
    elif mode == "size":
        m = tm & (p > 0)
        y[m] = p[m] * (1 + 2 * q)
    elif mode == "shift":
        y[tm] = p[tm] + q
    elif mode == "rescue":
        m = tm & (p < 0) & (rng.random(len(p)) < q)
        y[m] = p[m] * 0.5
    elif mode == "disaster":
        lose = p[p <= -0.50]
        m = tm & (rng.random(len(p)) < q)
        if m.any() and len(lose):
            y[m] = rng.choice(lose, m.sum(), replace=True)
    return y


def test_snr(df: pd.DataFrame, reps: int, qs, modes):
    """The like-for-like question: if a real day-level edge existed, which
    target would see it first?

    Half the OOS days are randomly designated 'treated', a synthetic edge of
    strength q is injected into them (see `_inject` for the four shapes), and
    every target then measures the SAME injected reality. Reporting
    SNR = mean(effect) / sd(effect) across repetitions gives the z-score that
    target would be expected to produce. Higher is better; q=0 must give ~0 and
    is the null control.
    """
    oos = df[df["half"] == "OOS"].reset_index(drop=True)
    p = oos["pnl"].to_numpy()
    d = oos["date"].to_numpy()
    days = np.unique(d)
    win = p[p > 0]
    names = list(targets(p))
    print(f"  OOS trades {len(p)}, OOS days {len(days)}, "
          f"winner pool {len(win)} (mean {win.mean()*100:+.1f}%)")
    print(f"  {reps} reps per cell; SNR = mean(effect)/sd(effect) = expected z.")

    overall = {n: [] for n in names}
    for mode in modes:
        rng = np.random.default_rng(3)
        print(f"\n  --- injection shape: {mode.upper()} ---")
        print("  " + f"{'target':9}" + "".join(f"{'q=' + str(q):>10}" for q in qs))
        res = {n: [] for n in names}
        for q in qs:
            acc = {n: [] for n in names}
            for _ in range(reps):
                treat = set(rng.choice(days, len(days) // 2, replace=False))
                tm = np.isin(d, list(treat))
                ts = targets(_inject(p, tm, q, mode, win, rng))
                for n in names:
                    acc[n].append(ts[n][tm].mean() - ts[n][~tm].mean())
            for n in names:
                a = np.array(acc[n])
                res[n].append(a.mean() / a.std() if a.std() > 0 else np.nan)
        for n in names:
            print(f"  {n:9}" + "".join(f"{v:>10.2f}" for v in res[n]))
            # normalise each shape by `raw` so shapes are comparable
            r = [v for v, q in zip(res[n], qs) if q > 0]
            b = [v for v, q in zip(res["raw"], qs) if q > 0]
            overall[n].append(np.nanmean(np.abs(r)) / np.nanmean(np.abs(b))
                              if np.nanmean(np.abs(b)) else np.nan)

    print(f"\n  {'='*78}")
    print("  SNR RELATIVE TO `raw`, PER INJECTION SHAPE  (>1 beats the status quo)")
    print(f"  {'='*78}")
    print("  " + f"{'target':9}" + "".join(f"{m:>10}" for m in modes)
          + f"{'WORST':>10}")
    order = sorted(names, key=lambda n: -np.nanmin(overall[n]))
    for n in order:
        w = np.nanmin(overall[n])
        flag = "  <- status quo" if n == "raw" else ("  <- robust" if w > 1.15 else "")
        print("  " + f"{n:9}" + "".join(f"{v:>10.2f}" for v in overall[n])
              + f"{w:>10.2f}" + flag)
    print("\n  Rank on the WORST column, not the average: a target is only worth")
    print("  switching to if it beats `raw` whatever shape the real effect turns")
    print("  out to have. Winning on one shape means it was tuned to a guess.")

    # The per-shape columns are the actionable part: if you are willing to NAME
    # the effect you are hunting, the target follows from the table.
    print(f"\n  {'='*78}")
    print("  PRESCRIPTION -- if you can NAME the effect shape, pick its column")
    print(f"  {'='*78}")
    blurb = {
        "freq": "conditioner makes WINNERS MORE FREQUENT (entry-side regime)",
        "size": "conditioner makes the RIGHT TAIL FATTER (amplification)",
        "shift": "conditioner adds a CONSTANT to every trade",
        "rescue": "conditioner AVOIDS THE DISASTERS (exit / risk-side)",
        "disaster": "state is DISASTER-PRONE (what a SKIP rule must detect)",
    }
    for i, m in enumerate(modes):
        cand = {n: overall[n][i] for n in names if n != "raw"}
        top = max(cand, key=lambda k: cand[k])
        verdict = (f"use `{top}`  ({cand[top]:.2f}x raw)" if cand[top] > 1.15
                   else f"use `raw` -- nothing beats it ({top} only {cand[top]:.2f}x)")
        print(f"    {m:7} {blurb.get(m, ''):58} {verdict}")
    print("\n  The asymmetry is the finding. There is no universally better target,")
    print("  because on this book the right tail is BOTH the edge and the variance.")
    print("  But LOSS-AVOIDANCE is far cheaper to detect than an entry edge, and")
    print("  `loss50` -- the left-tail hit rate, 1[return <= -50%] -- is the metric")
    print("  for it. NOTE WHAT IT IS AND IS NOT: a MEASUREMENT transform used to")
    print("  score a candidate filter, never a trading rule. The bot's exit is")
    print("  untouched and it still collects the +158% tail. Nothing here caps a")
    print("  winner or turns the book into a scalper -- see test_skip's docstring")
    print("  for why the fix must be a SKIP (pre-entry) and never a stop.")


# ------------------------------------------------------------------ skip
def test_skip(df: pd.DataFrame, reps: int, fracs, rhos):
    """HOW GOOD WOULD A SKIP RULE HAVE TO BE TO BE WORTH ANYTHING?

    WHY A SKIP RULE AND NOT A STOP.  Every MECHANICAL EXIT tested on this book
    has hurt: the ROE give-back took win 0.38 -> 0.59 and destroyed expectancy,
    tighter stops hurt, TP+trail scored OOS -1.1% vs +4.8% pure. The reason is
    structural, not a tuning failure -- an exit acts on a trade already open and
    cannot tell "this goes to -65%" from "this dips then runs to +158%". The
    path to the right tail runs through drawdown, so anything that cuts the
    drawdown cuts the tail with it.

    A SKIP rule is a different object entirely. It refuses the trade BEFORE
    entry, so it removes the loser without ever touching a winner's path. That
    is why every filter that HAS worked here is an entry filter (regime gates,
    the VIX overlay, the 09:35 guard, the hour windows) and every exit
    modification has not. This is the honest answer to "how do I capture
    loss-avoidance without a mechanical exit that hurts": you do not touch the
    exit at all.

    THE MODEL.  A filter is given a noisy ranking ability `rho` -- the
    correlation between its score and the trade's true (rank-normalised)
    outcome. rho=1.0 is an oracle, rho=0 is a coin flip. It skips the worst
    `frac` of trades by that score. Reported: the book's resulting mean, and
    how much of the uncapped total P&L survives.

    READ THE rho=0 ROW FIRST. Skipping at random must leave the mean unchanged;
    any apparent gain there is the simulation lying to you.
    """
    oos = df[df["half"] == "OOS"].reset_index(drop=True)
    p = oos["pnl"].to_numpy()
    n = len(p)
    z = pd.Series(p).rank().to_numpy() / (n + 1)
    z = (z - z.mean()) / z.std()
    rng = np.random.default_rng(17)
    base, basetot = p.mean(), p.sum()
    print(f"  OOS n={n}, mean {base*100:+.1f}%, total {basetot:+.2f}, "
          f"disasters (<=-50%) {(p <= -0.5).mean()*100:.0f}% of trades\n")
    print(f"  {'rho':>5} |" + "".join(f"  skip {int(f*100):>2}%" for f in fracs)
          + "     <- book mean after the skip")
    print("  " + "-" * (7 + 10 * len(fracs)))
    for rho in rhos:
        row = []
        for f in fracs:
            acc = []
            for _ in range(reps):
                s = rho * z + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.normal(size=n)
                keep = s > np.quantile(s, f)
                acc.append(p[keep].mean() if keep.sum() else np.nan)
            row.append(np.nanmean(acc))
        print(f"  {rho:>5.2f} |" + "".join(f"{v*100:>+9.1f}%" for v in row))
    print("\n  Same grid as TOTAL P&L retained (the book also gets smaller, and a")
    print("  higher mean on far fewer trades is not obviously better):")
    print(f"  {'rho':>5} |" + "".join(f"  skip {int(f*100):>2}%" for f in fracs))
    print("  " + "-" * (7 + 10 * len(fracs)))
    for rho in rhos:
        row = []
        for f in fracs:
            acc = []
            for _ in range(reps):
                s = rho * z + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.normal(size=n)
                keep = s > np.quantile(s, f)
                acc.append(p[keep].sum())
            row.append(np.mean(acc) / basetot)
        print(f"  {rho:>5.2f} |" + "".join(f"{v*100:>+9.0f}%" for v in row))
    print(f"\n  ORACLE BOUND -- skip exactly the trades that lost >=50%:")
    k = p > -0.50
    print(f"    keeps {k.sum()}/{n} trades, mean {p[k].mean()*100:+.1f}% "
          f"(from {base*100:+.1f}%), total {p[k].sum():+.2f} "
          f"({p[k].sum()/basetot*100:.0f}% of uncapped)")
    print("    That is the ceiling. A real filter with rho~0.2-0.3 captures a")
    print("    small fraction of it -- read the grid above for how small.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="all",
                    choices=["dist", "mde", "snr", "skip", "all"])
    ap.add_argument("--fill", default="bot", choices=list(sim_core.FILL_MODELS))
    ap.add_argument("--pop", default="seq", choices=["seq", "screened"])
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--qs", nargs="*", type=float, default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--modes", nargs="*",
                    default=["freq", "size", "shift", "rescue", "disaster"])
    ap.add_argument("--fracs", nargs="*", type=float,
                    default=[0.10, 0.20, 0.30, 0.40])
    ap.add_argument("--rhos", nargs="*", type=float,
                    default=[0.0, 0.10, 0.20, 0.30, 0.50, 0.80])
    a = ap.parse_args()

    df = book(a.fill, a.pop)
    print(f"Book: fill={a.fill}, pop={a.pop}, n={len(df)} trades, "
          f"{df['date'].nunique()} sessions\n")
    if a.test in ("dist", "all"):
        print("=" * 100)
        print("DISTRIBUTION, bucketed at +/-5 / 10 / 25 / 50%")
        print("=" * 100)
        test_dist(df)
    if a.test in ("mde", "all"):
        print("\n" + "=" * 100)
        print("VARIANCE AND MINIMUM DETECTABLE EFFECT, per candidate target (OOS)")
        print("=" * 100)
        test_mde(df)
    if a.test in ("snr", "all"):
        print("\n" + "=" * 100)
        print("SIGNAL-TO-NOISE under an INJECTED day-level edge")
        print("=" * 100)
        test_snr(df, a.reps, a.qs, a.modes)
    if a.test in ("skip", "all"):
        print("\n" + "=" * 100)
        print("WHAT A SKIP RULE IS WORTH, as a function of how well it ranks")
        print("=" * 100)
        test_skip(df, a.reps, a.fracs, a.rhos)


if __name__ == "__main__":
    main()
