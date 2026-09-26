# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_ivr_termstructure.py
==========================
Two OPTIONS-MECHANICS conditioners on the deployed book, neither previously
tested:  IMPLIED-VOL RANK  and  TERM-STRUCTURE SLOPE.

WHY THIS IS NOT A REPEAT OF THE DIRECTIONAL WORK
    ~24 separate tests have now confirmed the flow trigger has no standalone
    directional edge, which makes every DIRECTIONAL conditioner provably futile
    (see METHODOLOGY.md 6c).  These two are NOT directional: they condition on
    the PRICE OF THE OPTION the bot buys, not on where the underlying goes.  A
    long-premium book can be edge-positive purely because it buys convexity
    cheap, and that channel has never been examined here.

THE HYPOTHESES (user's, stated so they can fail)
  H1 IVR   "Trigger at IVR 10 -> vega expansion is violent and profitable.
            Trigger at IVR 90 -> already priced, win rate collapses to IV crush."
            => book P&L should fall MONOTONICALLY in IVR.
  H2 TERM  "Are whales sweeping the front week while the back month is cheap, or
            is the whole surface lifting?"  Front-rich (BACKWARDATION) = the move
            is already paid for; front-cheap (CONTANGO) = convexity on sale.
            => book P&L should fall as ts_slope rises.

TWO PRIORS THAT ARGUE AGAINST H1, RECORDED BEFORE THE RESULT
  (a) SIGN CONFLICT.  check_regime_state / check_vix_overlay already found, and
      the book already DEPLOYS, the opposite sign at the market level: VIX_prev
      >= ~18 -> +24%/trade, below -> +7%.  High vol is where this book earns.
      H1 wants low vol.  If IVR reproduces the VIX sign, H1 is falsified AND the
      result is mostly a restatement of a lever already in production.
  (b) VEGA IS THE WRONG GREEK AT 0DTE.  A "violent vega expansion" needs vega.
      An ATM 0DTE option hours from expiry has almost none -- its P&L is gamma
      and theta.  `--test vega` measures this directly off the lake rather than
      asserting it: realised vega P&L per trade vs total P&L per trade.
      If vega is a rounding error, H1's stated MECHANISM is dead even if the
      IVR bucket spread happens to be non-zero (which would then need another
      explanation, e.g. IVR proxying realised-vol regime).

DATA
  _ivs_cache/{T}.parquet from `build_iv_surface.py` -- ATM IV per 15-min bucket
  per expiry, 501 sessions.  Derived per ticker-day, all from the PRIOR session's
  last bucket so any entry minute is lookahead-free (same convention as the
  deployed VIX overlay's vix_prev):
      iv_f     ATM IV of the front expiry (the 0/1DTE the bot actually buys)
      iv30     constant-maturity 30d ATM IV, interpolated in TOTAL VARIANCE,
               NaN unless the live expiry ladder brackets 30d (no extrapolation)
      ts       iv_f / iv30 - 1     >0 backwardation (front rich), <0 contango
      ivr_W    (iv30 - min)/(max - min) over the prior W sessions  [tastytrade]
      ivp_W    fraction of the prior W sessions with iv30 below today   [percentile]

METHOD
  Book = config.RULES as enabled, scored through sim_core (sequential fills,
  per-rule deployed exit via policy_for, fill="bot").  Cut into terciles by each
  feature.  Reported: n / all / IS / OOS / win / 6-slice coverage, a PER-RULE
  breakdown (the size-gate lesson: an aggregate spread is usually composition),
  and a DAY-BLOCK bootstrap of the low-minus-high OOS difference.

  Tests are COUNTED and printed.  With n~300 trades the detectable effect is
  large; `--test power` prints the minimum detectable spread so a null here is
  read as "underpowered" or "flat" honestly rather than as proof.

Usage:
  python build_iv_surface.py
  python check_ivr_termstructure.py --test ivr
  python check_ivr_termstructure.py --test ts
  python check_ivr_termstructure.py --test vega
  python check_ivr_termstructure.py --test power
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

from check_config_walkforward import _slice_idx
import sim_core

CACHE = "_ivs_cache"
SPLIT = pd.Timestamp("2025-08-21").date()
RTH_LAST = 960          # 16:00 bucket start; prior-session snapshot is <= this


# --------------------------------------------------------------- surface
def _cm_iv(dte: np.ndarray, iv: np.ndarray, target: float) -> float:
    """Constant-maturity ATM IV, interpolated in TOTAL VARIANCE (iv^2 * t).

    Returns NaN unless the live ladder BRACKETS `target`.  Flat-extrapolating a
    30d vol off a 5d and a 91d quote would manufacture the very number the test
    is about, so it is refused instead.
    """
    if len(dte) < 2:
        return np.nan
    t = np.maximum(dte.astype(float), 0.5)
    o = np.argsort(t)
    t, v = t[o], (iv[o] ** 2) * t[o]
    if target < t[0] or target > t[-1]:
        return np.nan
    return float(np.sqrt(np.interp(target, t, v) / target))


def surface(tk: str, cm: float, windows) -> pd.DataFrame | None:
    """Per-ticker daily IV features, indexed by the date they may be USED on
    (i.e. every column is already shifted one session forward)."""
    fp = os.path.join(CACHE, f"{tk}.parquet")
    if not os.path.exists(fp):
        return None
    df = pl.read_parquet(fp).to_pandas()
    # last RTH bucket of each session = the "close" snapshot
    df = df[df["mod15"] <= RTH_LAST]
    if df.empty:
        return None
    last = df.groupby("date")["mod15"].max()
    df = df.merge(last.rename("_lm"), on="date")
    df = df[df["mod15"] == df["_lm"]]

    rows = []
    for d, g in df.groupby("date"):
        dte, iv = g["dte"].to_numpy(), g["iv"].to_numpy()
        # FRONT LEG EXCLUDES dte==0.  At the prior session's CLOSE the dte==0
        # contract is minutes from expiry and its "IV" is a degenerate number
        # (it printed iv_front/iv30-1 down to -0.91, i.e. a 2-vol front against
        # a 20-vol 30d).  The bot's next-session 0DTE is TODAY's dte==1, so that
        # is the correct front leg and it is what the term structure must use.
        liv = dte >= 1
        if not liv.any():
            continue
        front = dte[liv].min()
        rows.append(dict(date=d,
                         iv_f=float(iv[dte == front].mean()),
                         front_dte=int(front),
                         iv30=_cm_iv(dte, iv, cm),
                         nexp=len(dte)))
    s = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    s["ts"] = s["iv_f"] / s["iv30"] - 1.0

    for w in windows:
        r = s["iv30"].rolling(w, min_periods=max(20, w // 3))
        lo, hi = r.min(), r.max()
        s[f"ivr_{w}"] = np.where(hi > lo, (s["iv30"] - lo) / (hi - lo), np.nan) * 100
        s[f"ivp_{w}"] = (s["iv30"]
                         .rolling(w, min_periods=max(20, w // 3))
                         .apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)) * 100

    # EVERY feature shifted one session: a gate on date D may only read D-1.
    feat = [c for c in s.columns if c != "date"]
    s[feat] = s[feat].shift(1)
    s["date"] = pd.to_datetime(s["date"]).dt.date
    return s.set_index("date")


# --------------------------------------------------------------- book
def book(fill="bot", pop="seq"):
    """[(date, pnl, rule, ticker)] scored through sim_core with each rule's
    DEPLOYED exit.

    pop="seq"       the deployed book: one position per ticker (bot_runner.py:1288).
                    n~297.  This is what the bot earns, but `--test power` shows
                    it can only resolve a ~65pp per-trade tercile spread.
    pop="screened"  EVERY candidate trigger scored independently.  NOT tradeable
                    -- the live bot cannot take overlapping positions -- and it
                    over-weights busy days, so it must never be read as a P&L.
                    It is used here only for the MECHANISM question ("does the
                    option's entry IV change its outcome?"), where the extra n is
                    what buys the power the deployed book does not have.
    """
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
            rows = [(d, sim_core.simulate(path, pol, em, fill=fill)[0])
                    for d, m, path in cand]
        for d, p in rows:
            out.append((d, p, r["name"], r["ticker"]))
    return sorted(out)


def attach(tr, cm, windows):
    """Join each trade to its ticker's PRIOR-session IV features."""
    surf = {}
    df = pd.DataFrame(tr, columns=["date", "pnl", "rule", "ticker"])
    for tk in df["ticker"].unique():
        s = surface(tk, cm, windows)
        if s is not None:
            surf[tk] = s
    keep, miss = [], {}
    for row in df.itertuples():
        s = surf.get(row.ticker)
        if s is None or row.date not in s.index:
            miss[row.ticker] = miss.get(row.ticker, 0) + 1
            continue
        keep.append({**row._asdict(), **s.loc[row.date].to_dict()})
    return pd.DataFrame(keep), miss


# --------------------------------------------------------------- reporting
def _stat(v: pd.DataFrame, lbl: str, width=26) -> str:
    if v.empty:
        return f"    {lbl:{width}} (none)"
    p, d = v["pnl"].to_numpy(), v["date"].to_numpy()
    i, o = p[d < SPLIT], p[d >= SPLIT]
    sl = [[] for _ in range(6)]
    for dd, pp in zip(d, p):
        k = _slice_idx(dd)
        if k is not None:
            sl[k].append(pp)
    pop = sum(1 for b in sl if len(b) >= 3)
    return (f"    {lbl:{width}} n={len(p):>4}  {p.mean()*100:>+7.1f}%  "
            f"IS {(i.mean()*100 if len(i) else np.nan):>+7.1f}%  "
            f"OOS {(o.mean()*100 if len(o) else np.nan):>+7.1f}%  "
            f"win {(p>0).mean():.2f}  pop {pop}/6")


def _terciles(df: pd.DataFrame, col: str):
    """IS-only cut points, applied to both halves -- picking them on the full
    sample would leak OOS information into the bucket boundaries."""
    ref = df[df["date"] < SPLIT][col].dropna()
    if len(ref) < 30:
        ref = df[col].dropna()
    if len(ref) < 15:
        return None
    q1, q2 = np.percentile(ref, [33.33, 66.67])
    if not (q1 < q2):
        return None
    return q1, q2


def _boot_diff(lo: pd.DataFrame, hi: pd.DataFrame, n=4000, seed=7):
    """Day-block bootstrap of mean(lo) - mean(hi) on the OOS half.

    Resamples whole DAYS, not trades: same-day trades across tickers share the
    market regime, and a trade-level bootstrap would treat them as independent
    and shrink the interval.
    """
    a = lo[lo["date"] >= SPLIT]
    b = hi[hi["date"] >= SPLIT]
    if a.empty or b.empty:
        return (np.nan, np.nan, np.nan)
    pool = {}
    for tag, sub in (("lo", a), ("hi", b)):
        for d, p in zip(sub["date"], sub["pnl"]):
            pool.setdefault(d, ([], []))[0 if tag == "lo" else 1].append(p)
    days = list(pool)
    rng = np.random.default_rng(seed)
    obs = a["pnl"].mean() - b["pnl"].mean()
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days), replace=True)
        L, H = [], []
        for k in pick:
            l, h = pool[days[k]]
            L += l
            H += h
        if L and H:
            out.append(np.mean(L) - np.mean(H))
    if not out:
        return (obs, np.nan, np.nan)
    return (obs, *np.percentile(out, [2.5, 97.5]))


def _cut_report(df, col, label, ntests):
    q = _terciles(df, col)
    if q is None:
        print(f"  {label}: insufficient spread to bucket")
        return ntests
    q1, q2 = q
    lo = df[df[col] <= q1]
    mid = df[(df[col] > q1) & (df[col] <= q2)]
    hi = df[df[col] > q2]
    print(f"\n  {label}   IS-set cuts: <={q1:.3g}  |  >{q2:.3g}"
          f"   (n missing feature: {df[col].isna().sum()})")
    print(_stat(lo, f"T1 LOW  ({col})"))
    print(_stat(mid, "T2 MID"))
    print(_stat(hi, "T3 HIGH"))
    obs, c1, c2 = _boot_diff(lo, hi)
    arrow = "H1 predicts LOW > HIGH"
    print(f"    OOS LOW-minus-HIGH {obs*100:>+7.1f}pp   day-block 95% CI "
          f"[{c1*100:>+7.1f}, {c2*100:>+7.1f}]pp   <- {arrow}"
          + ("   *" if np.isfinite(c1) and (c1 > 0 or c2 < 0) else ""))

    # per-rule: is the spread a lever, or just composition?
    print(f"    per-rule OOS (LOW / HIGH), rules with >=5 OOS trades in both:")
    any_rule = False
    for rn, g in df.groupby("rule"):
        gl = g[(g[col] <= q1) & (g["date"] >= SPLIT)]
        gh = g[(g[col] > q2) & (g["date"] >= SPLIT)]
        if len(gl) >= 5 and len(gh) >= 5:
            any_rule = True
            print(f"      {rn:22} LOW {gl['pnl'].mean()*100:>+7.1f}% (n={len(gl):>3})   "
                  f"HIGH {gh['pnl'].mean()*100:>+7.1f}% (n={len(gh):>3})   "
                  f"diff {(gl['pnl'].mean()-gh['pnl'].mean())*100:>+7.1f}pp")
    if not any_rule:
        print("      (no rule has >=5 OOS trades in both buckets -- "
              "any aggregate spread here is UNVERIFIABLE as a lever)")
    return ntests + 1


# --------------------------------------------------------------- vega
def test_vega(stride: int, band: float):
    """Can a vega move plausibly BE the book's P&L?  Measured, not asserted.

    For every ATM front-expiry (0/1DTE) contract in the lake at the bot's entry
    hours, `vega_close / close` is the option's percentage P&L per ONE VOL POINT
    of IV change.  Pair that with the realised distribution of intraday ATM-IV
    moves and you get the largest percentage move vega can produce over a hold,
    which is then compared with the book's actual ~+8%/trade.

    H1 needs this number to be BIG.  If it is small, "violent vega expansion"
    cannot be what the book is harvesting, whatever the IVR buckets say.
    """
    import glob
    parts = sorted(glob.glob(f"lake/silver/option-contracts-1m/date=*/bars.parquet"))
    parts = parts[::stride]
    print(f"  scanning {len(parts)} sessions (every {stride}th) for ATM front-expiry greeks")
    frames = []
    for i, p in enumerate(parts, 1):
        df = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol").is_in(BOOK_TK)
                      & pl.col("iv_close").is_not_null()
                      # THE BOT'S OWN $0.50 ENTRY FLOOR (bot_runner.py:1421,
                      # sim_core.build_candidates min_mid). Without it the scan
                      # admits contracts priced at a dime that the bot would
                      # never buy, and since vega/price explodes as price -> 0
                      # those rows dominated the median: the same scan reported
                      # 48% of premium per IV point at a $0.10 floor and ~4% at
                      # the real one. Measure the contract the bot can actually
                      # trade, not the cheapest thing on the board.
                      & (pl.col("close") >= 0.50)
                      & (pl.col("vega_close") > 0)
                      & (pl.col("underlying_close") > 0)
                      & (((pl.col("strike") - pl.col("underlying_close")).abs()
                          / pl.col("underlying_close")) <= band))
              .with_columns(dte=(pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days(),
                            # cast first: dt.hour() is Int8 and overflows at *60
                            mod=(pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                                 + pl.col("minute_et").dt.minute().cast(pl.Int32)),
                            date=pl.col("minute_et").dt.date())
              .filter(pl.col("dte").is_between(0, 1) & pl.col("mod").is_between(570, 900))
              # THE CONTRACT THE BOT ACTUALLY BUYS: the single NEAREST strike,
              # per ticker/minute/expiry/side. A +/-band instead of the nearest
              # strike mixes ITM contracts (price is mostly intrinsic, so
              # vega/price ~ 0) with near-worthless OTM ones (vega/price > 150%)
              # and the median of that mixture describes no tradeable contract.
              .with_columns(mny=(pl.col("strike") - pl.col("underlying_close")).abs())
              .filter(pl.col("mny") == pl.col("mny").min().over(
                  "underlying_symbol", "date", "mod", "dte", "option_type"))
              .select("underlying_symbol", "date", "mod", "dte", "close",
                      "iv_close", "vega_close", "gamma_close", "theta_close",
                      "underlying_close")
              .collect())
        if df.height:
            frames.append(df)
        if i % 25 == 0:
            print(f"    {i}/{len(parts)}", flush=True)
    d = pl.concat(frames).to_pandas()

    # UNITS ARE NOT ASSUMED -- they are fitted.  An earlier version of this test
    # divided vega_close by 100 on the guess that the feed quotes vega per 1.00
    # of vol, and concluded vega was a rounding error.  That was wrong by 100x
    # and inverted the verdict.  The convention is recovered from the feed's own
    # data below and printed, so the scale is never silently assumed again.
    scale = _fit_vega_units()

    # DROP ticker-days whose 0DTE greeks are the lake's broken ones -- see
    # valid_0dte(). Without this the 101-session medians were 41% per IV point
    # and -0.6%/day theta, while a single CLEAN session gave 2-6% and -194%/day.
    ok = valid_0dte()
    n0 = len(d)
    keep = [(t, dd) in ok or k != 0
            for t, dd, k in zip(d["underlying_symbol"], d["date"], d["dte"])]
    d = d[pd.Series(keep, index=d.index)]
    print(f"\n  0DTE validity gate: dropped {n0 - len(d):,} of {n0:,} contract-minutes "
          f"on ticker-days where the lake's 0DTE IV/greeks are corrupt")

    d["pct_per_volpt"] = d["vega_close"] * scale / d["close"] * 100
    d["theta_pct_per_day"] = d["theta_close"] / d["close"] * 100

    print(f"\n  ATM 0/1DTE contracts sampled: {len(d):,}  "
          f"({d['date'].nunique()} sessions, {d['underlying_symbol'].nunique()} tickers)")
    print(f"  median raw vega_close={d['vega_close'].median():.4f}  "
          f"median option px={d['close'].median():.2f}  "
          f"median theta_close={d['theta_close'].median():.4f}")
    print(f"  FITTED vega scale = {scale:.4g} price-units per IV POINT per unit of "
          f"vega_close\n  (1.0 => the feed quotes vega PER VOL POINT; "
          f"0.01 => per 1.00 vol)")

    print("\n  OPTION % P&L PER 1 IV POINT  (vega / price), by DTE:")
    for dte, g in d.groupby("dte"):
        q = np.percentile(g["pct_per_volpt"], [10, 50, 90])
        print(f"    dte={dte}  n={len(g):>8,}  p10 {q[0]:>7.2f}%  "
              f"median {q[1]:>7.2f}%  p90 {q[2]:>7.2f}%   per IV point")
    print("\n  by hour-of-day (dte=0):")
    z = d[d["dte"] == 0]
    for h, g in z.assign(h=z["mod"] // 60).groupby("h"):
        print(f"    {h:02d}:00  n={len(g):>8,}  vega {g['pct_per_volpt'].median():>7.2f}% "
              f"per IV pt   theta {g['theta_pct_per_day'].median():>9.1f}%/day")

    # realised intraday ATM IV moves, from the surface cache
    print("\n  REALISED ATM front IV moves over a 2-hour hold (from _ivs_cache):")
    mv = []
    for tk in BOOK_TK:
        s = _intraday_front(tk)
        if s is not None:
            mv.append(s)
    if mv:
        m = pd.concat(mv)
        q = np.percentile(m.abs(), [50, 90, 99])
        print(f"    n={len(m):,}   |dIV| median {q[0]:.2f}  p90 {q[1]:.2f}  "
              f"p99 {q[2]:.2f}  IV points")
        med_sens = d[d["dte"] == 0]["pct_per_volpt"].median()
        for lbl, iv_mv in (("median", q[0]), ("p90", q[1]), ("p99 (tail)", q[2])):
            print(f"      => {lbl:11} IV move x median 0DTE vega-sensitivity = "
                  f"{iv_mv*med_sens:>8.1f}% of premium")
    print("\n  Compare: the deployed book earns ~+8.4%/trade (check_book_now, fill=bot).")
    print("\n  READ THIS CAREFULLY. A big vega number here does NOT confirm H1.")
    print("  It establishes only that the CHANNEL is open -- that IV moves are")
    print("  large enough to matter to a 0DTE position. H1 additionally claims")
    print("  the channel is PREDICTABLE FROM IVR, and the tercile tests above")
    print("  are what test that. They found no usable signal.")
    print("  Note also that at 0DTE, 'IV' and the underlying move are not")
    print("  independent: a fitted 0DTE vol rises mechanically when spot runs.")
    print("  Part of what looks like vega P&L here is gamma P&L in vol clothing.")


def valid_0dte(min_ratio: float = 0.5):
    """{(ticker, date)} where the lake's 0DTE greeks are USABLE.

    The silver lake's `iv_close` is BROKEN FOR dte==0 over a long stretch of the
    sample: median 0DTE ATM IV reads 0.009-0.013 from ~2024-10 to ~2025-12 (a
    1% vol on a same-day ATM option), against a 0.10-0.17 one-to-three-day IV
    and a comparable RV20 on the same dates.  It reads sanely (0.21-0.34) from
    ~2026-01 on, and on 2024-08-20.  `vega_close` and `theta_close` are wrong in
    lockstep on exactly those dates (vega ~0.85 and theta ~-0.01 on a 0DTE ATM,
    i.e. a contract expiring today that decays 1%/day), which is the signature
    of a bad time-to-expiry in the vendor's pricer.

    The rest of the surface is FINE -- dte 1-3, 4-9, 25-35 and 50-70 all track
    realised vol across the whole window -- so `iv30`, the term structure (whose
    front leg already excludes dte==0) and everything the IVR tests use are
    unaffected.  Only 0DTE greeks need this gate.

    Gate: keep a ticker-day only if median iv(dte=0) >= `min_ratio` x median
    iv(dte in 1..3).  Physically motivated and needs no hardcoded date range.
    """
    ok = set()
    for tk in BOOK_TK:
        fp = os.path.join(CACHE, f"{tk}.parquet")
        if not os.path.exists(fp):
            continue
        df = pl.read_parquet(fp).to_pandas()
        z = df[df["dte"] == 0].groupby("date")["iv"].median()
        r = df[df["dte"].between(1, 3)].groupby("date")["iv"].median()
        j = pd.concat([z.rename("i0"), r.rename("i13")], axis=1, sort=True).dropna()
        for d in j.index[j["i0"] >= min_ratio * j["i13"]]:
            ok.add((tk, d))
    return ok


def _fit_vega_units(path: str | None = None) -> float:
    """Recover the feed's vega convention from the feed itself.

    Fits the greek identity on consecutive 1-minute bars of the SAME contract:

        dPrice - delta*dS  ~  vega * dIV[points]

    and returns the median PER-CONTRACT ratio (fitted slope / vega_close).  A
    result near 1.0 means vega_close is quoted per VOL POINT; near 0.01 means
    per 1.00 of vol.

    Per-contract, NOT pooled: a pooled fit divides one slope by a median vega
    taken across 0DTE and 1DTE contracts on different underlyings whose vegas
    differ by an order of magnitude, which is meaningless.  Pooling here
    returned 0.19 while the per-contract median returned 0.80.

    The estimate is biased DOWN: dIV is measured with error (errors-in-variables
    attenuates the slope toward zero) and `close` is a 1-minute print, not a
    clean mid.  So a median of ~0.8 is read as 1.0, not as 0.8.  The return is
    therefore snapped to the nearer of {1.0, 0.01} and both are printed.
    """
    import glob
    p = path or sorted(glob.glob(f"lake/silver/option-contracts-1m/date=*/bars.parquet"))[-1]
    d = (pl.scan_parquet(p)
         .filter(pl.col("underlying_symbol").is_in(BOOK_TK)
                 & pl.col("iv_close").is_not_null() & (pl.col("close") > 0.20)
                 & (pl.col("vega_close") > 0)
                 & (((pl.col("strike") - pl.col("underlying_close")).abs()
                     / pl.col("underlying_close")) <= 0.02))
         .with_columns(dte=(pl.col("expiry") - pl.col("minute_et").dt.date()).dt.total_days(),
                       mod=(pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                            + pl.col("minute_et").dt.minute().cast(pl.Int32)))
         .filter(pl.col("dte").is_between(0, 1) & pl.col("mod").is_between(575, 900))
         .sort("option_chain_id", "mod")
         .with_columns(dP=pl.col("close").diff().over("option_chain_id"),
                       dS=pl.col("underlying_close").diff().over("option_chain_id"),
                       dIV=pl.col("iv_close").diff().over("option_chain_id") * 100.0,
                       dm=pl.col("mod").diff().over("option_chain_id"))
         .filter(pl.col("dm") == 1)
         .select("option_chain_id", "dP", "dS", "dIV", "delta_close", "vega_close")
         .drop_nulls()
         .collect().to_pandas())
    r = []
    for _, g in d.groupby("option_chain_id"):
        m = g["dIV"].abs() > 0.05
        if m.sum() < 20:
            continue
        resid = g["dP"][m] - g["delta_close"][m] * g["dS"][m]
        v = g["vega_close"][m].median()
        if v > 0:
            r.append(np.polyfit(g["dIV"][m], resid, 1)[0] / v)
    if not r:
        raise RuntimeError("could not fit vega units")
    med = float(np.median(r))
    snapped = 1.0 if abs(np.log10(med / 1.0)) < abs(np.log10(med / 0.01)) else 0.01
    print(f"  [vega units] per-contract slope/vega: n={len(r)}  median {med:.3f}"
          f"  -> snapped to {snapped} (per {'VOL POINT' if snapped == 1 else '1.00 VOL'})")
    return snapped


def _intraday_front(tk: str, hold_buckets: int = 8):
    """Signed front-expiry ATM IV change over `hold_buckets` x 15min, in IV POINTS."""
    fp = os.path.join(CACHE, f"{tk}.parquet")
    if not os.path.exists(fp):
        return None
    df = pl.read_parquet(fp).to_pandas()
    # dte>=1 for the same reason valid_0dte() exists: the lake's 0DTE IV is
    # corrupt over most of this sample, so a "front IV move" measured on it is
    # measuring the vendor's bug. This is the NEAREST SOUND expiry instead.
    df = df[(df["mod15"] >= 570) & (df["mod15"] <= 955) & (df["dte"] >= 1)]
    if df.empty:
        return None
    front = df.groupby(["date", "mod15"])["dte"].transform("min")
    f = df[df["dte"] == front].groupby(["date", "mod15"])["iv"].mean().reset_index()
    f = f.sort_values(["date", "mod15"])
    g = f.groupby("date")["iv"]
    return ((g.shift(-hold_buckets) - f["iv"]) * 100).dropna()


BOOK_TK = ["AVGO", "GLD", "IWM", "META", "MSFT", "NVDA", "QQQ", "SMH", "SPY"]


# --------------------------------------------------------------- power
def test_power(df):
    """What spread could this sample even DETECT?  A null is only informative
    above this line; below it the honest verdict is 'underpowered', not 'flat'."""
    oos = df[df["date"] >= SPLIT]
    print(f"  OOS trades: {len(oos)}   OOS days: {oos['date'].nunique()}   "
          f"per-trade SD: {oos['pnl'].std()*100:.1f}pp")
    # empirical day-block SE of a tercile-vs-tercile difference, using a random
    # 1/3-vs-1/3 split so the SE reflects THIS sample's clustering, not an ideal one.
    rng = np.random.default_rng(11)
    days = oos["date"].unique()
    diffs = []
    dmap = {d: g["pnl"].to_numpy() for d, g in oos.groupby("date")}
    for _ in range(3000):
        pick = rng.choice(len(days), len(days), replace=True)
        a, b = [], []
        for j, k in enumerate(pick):
            (a if j % 2 == 0 else b).extend(dmap[days[k]])
        if a and b:
            diffs.append(np.mean(a) - np.mean(b))
    se = np.std(diffs)
    print(f"  day-block SE of a half-vs-half OOS difference: {se*100:.1f}pp")
    print(f"  MINIMUM DETECTABLE SPREAD (80% power, 5% two-sided): "
          f"~{2.8*se*100:.1f}pp per trade")
    print("  => a LOW-minus-HIGH tercile spread smaller than this is not evidence "
          "of a flat effect;\n     it is evidence that this sample cannot see one.")


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="ivr",
                    choices=["ivr", "ts", "vega", "power", "all"])
    ap.add_argument("--cm", type=float, default=30.0, help="constant maturity, days")
    ap.add_argument("--windows", nargs="*", type=int, default=[60, 120, 252])
    ap.add_argument("--fill", default="bot", choices=list(sim_core.FILL_MODELS))
    ap.add_argument("--stride", type=int, default=5, help="--test vega date stride")
    ap.add_argument("--band", type=float, default=0.015)
    ap.add_argument("--pop", default="seq", choices=["seq", "screened"],
                    help="seq = the deployed book (tradeable, n~297); "
                         "screened = every trigger (NOT tradeable, mechanism only)")
    a = ap.parse_args()

    if a.test == "vega":
        test_vega(a.stride, a.band)
        return

    print(f"Scoring (fill={a.fill}, pop={a.pop}, per-rule deployed exits)...")
    if a.pop == "screened":
        print("  !! pop=screened counts overlapping triggers the live bot cannot take."
              "\n     Read the BUCKET CONTRASTS only -- the levels are not a P&L.")
    tr = book(a.fill, a.pop)
    df, miss = attach(tr, a.cm, a.windows)
    print(f"  {len(tr)} trades -> {len(df)} with IV features"
          + (f"   (dropped: {miss})" if miss else ""))
    print(_stat(df, "BOOK (all trades)"))
    ncov = df["iv30"].notna().mean()
    print(f"  iv30 available on {ncov*100:.0f}% of trades "
          f"(NaN = the ladder did not bracket {a.cm:.0f}d that session)")

    n = 0
    if a.test in ("ivr", "all"):
        print("\n" + "=" * 96)
        print("H1  IMPLIED VOL RANK   -- H1 predicts LOW IVR beats HIGH IVR")
        print("=" * 96)
        for w in a.windows:
            n = _cut_report(df.dropna(subset=[f"ivr_{w}"]), f"ivr_{w}",
                            f"IV RANK, {w}-session min-max", n)
        for w in a.windows:
            n = _cut_report(df.dropna(subset=[f"ivp_{w}"]), f"ivp_{w}",
                            f"IV PERCENTILE, {w}-session", n)
    if a.test in ("ts", "all"):
        print("\n" + "=" * 96)
        print(f"H2  TERM STRUCTURE  ts = iv_front/iv{a.cm:.0f} - 1   "
              "(>0 backwardation, <0 contango)")
        print("H2 predicts LOW ts (contango, front cheap) beats HIGH ts")
        print("=" * 96)
        n = _cut_report(df.dropna(subset=["ts"]), "ts", "TERM-STRUCTURE SLOPE", n)
        print(f"\n  ts distribution: p10 {df['ts'].quantile(.10):+.3f}  "
              f"median {df['ts'].median():+.3f}  p90 {df['ts'].quantile(.90):+.3f}   "
              f"(share backwardated: {(df['ts']>0).mean()*100:.0f}%)")
    if a.test in ("power", "all"):
        print("\n" + "=" * 96)
        print("POWER")
        print("=" * 96)
        test_power(df)
    if n:
        print(f"\n  TESTS RUN IN THIS PASS: {n} tercile cuts "
              f"(at 95%, expect ~{n*0.05:.1f} false positives by chance).")


if __name__ == "__main__":
    main()
