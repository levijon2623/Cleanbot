# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
sim_core.py
===========
THE single simulation core. Every backtest in this repo should route through
here so that fixing a fill model fixes it everywhere at once.

WHY THIS MODULE EXISTS
----------------------
As of 2026-09-10 there were FIVE near-identical candidate builders and TWO
divergent exit simulators in the tree. The newer one modelled the live exit
cushion; the older one did not, and was ~10pp optimistic per trade. The stale
one was still behind `check_book_now` (what the book earns) and
`check_trail_regression` (the per-rule trail-vs-static calls that set
`trail_pct` in config). That is exactly how the original sequential-fill bug
survived for months: two code paths, one quietly wrong, no single place to fix.

THE THREE THINGS THIS GETS RIGHT, ALL LEARNED THE HARD WAY
----------------------------------------------------------
1. SEQUENTIAL FILLS. `bot_runner.py:1288` = `if ticker in self.active_snipes:
   continue` -- the live bot holds AT MOST ONE POSITION PER TICKER. Scoring
   every matched trigger independently inflates trade count ~10x and OOS
   expectancy ~2.5x. `walk()` enforces the real guard.

2. REALISTIC FILLS. Enter at the ASK, exit at the BID. Costs ~10pp book-wide
   versus mid-to-mid, and more for trailing exits, which leave on the way down.

3. THE LIVE EXIT CUSHION (`cushion=True`, the DEFAULT). `bot_runner._fire_exit`
   prices the closing limit at `bid - spread*1.5` on a losing exit and
   `bid - spread*0.5` on a profitable one -- Webull is limit-only for options,
   so the bot must post marketable. Verified against live paper fills on
   2026-09-09: AVGO trail level $0.65 -> exit $0.50 (cushion $0.15 = 1.5 x $0.10
   spread), and $0.975 -> $0.74. Cost 11.5pp and 14.1pp of entry premium.
   Book-wide it takes the deployed trail from OOS +17.4% to +7.1%.

   It is NOT neutral between policies: losing exits pay 3x the cushion, so it
   penalises low-win-rate policies (the trail, win 0.38) harder than
   high-win-rate ones. Any policy comparison run WITHOUT it is biased toward
   whichever policy loses more often.

   NOTE the true fill sits BETWEEN two bounds: DRY_RUN books at the cushioned
   limit (zero price improvement, pessimistic) while a plain-bid sim is
   optimistic. Live is in between. `cushion=False` is kept only to reproduce
   historical numbers, never to make a decision.

WHAT `walk()` ALSO CONTROLS -- and why it belongs here
------------------------------------------------------
The position policy DETERMINES THE SAMPLE, not just the P&L. The guard decides
which triggers become trades; the EXIT decides when the ticker frees up, which
decides which SUBSEQUENT triggers become trades. Measured on identical rules and
triggers: the deployed trail yields 310 trades, a faster exit yields 603. The
book's OOS SIGN changes with the sampling rule (+8.9% as-screened, +7.1%
sequential, -6.1% fast-exit, -12.0% first-trigger-only). So `cap`, `skip` and
`after` live here alongside the exit, because they are the same decision.

    cap    max trades per ticker-day   (None = deployed, unlimited)
    skip   ignore the first N ACTIONABLE triggers per ticker-day. Triggers
           arriving while a position is open are already blocked by the guard
           and do NOT consume the skip budget. NOTE: skipping keeps us FLAT, so
           the next trigger is eligible IMMEDIATELY -- skip=1 is NOT "start at
           trade #2", it builds a different (generally earlier) sequence.
    after  no entry before this minute-of-day

PAYLOAD FORMAT
--------------
    (entry_mid, entry_ask, close[], high[], low[], bid[], ask[], mod[])
An 8-tuple. The legacy 7-tuple (no `ask[]`) is still accepted for backward
compatibility, but cushion modelling is impossible without it and is silently
disabled -- `build_candidates()` always emits the 8-tuple.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()
DEFAULT_EOD = 15 * 60 + 55


# ----------------------------------------------------------------- rules
def eod_mod(rule) -> int:
    """Minute-of-day this rule flattens. `eod_flatten: "HH:MM"` or 15:55."""
    ef = (rule or {}).get("eod_flatten")
    if ef:
        h, m = str(ef).split(":")
        return int(h) * 60 + int(m)
    return DEFAULT_EOD


def policy_for(rule, trail_pct_default=0.50) -> dict:
    """The rule's DEPLOYED exit, exactly as bot_runner would run it.

    `trail_pct: 0` keeps the static bracket. Otherwise the trail supersedes the
    take-profit entirely -- bot_runner's monitor loop is an if/elif chain where
    `elif trail_stop is not None:` SWALLOWS the tp branch, so `bid >= tp_limit`
    is unreachable while trailing. That is deliberate (TP+trail scored OOS -1.1%
    vs +4.8% pure) but it has a consequence worth remembering: a 50% trail only
    rises above the entry once peak ROE exceeds +100%, so any trade peaking
    between 0% and +100% exits at a loss BY CONSTRUCTION.
    """
    tp_ = rule.get("trail_pct", trail_pct_default)
    tp_ = float(tp_ if tp_ is not None else trail_pct_default)
    if tp_ > 0:
        return dict(name=f"trail{int(tp_*100)}", kind="trail", trail=tp_)
    tr, rr = float(rule["target_roe"]), float(rule["rr"])
    stop = tr / rr
    return dict(name="static", kind="fixed", tp=tr, stop=(stop if stop < 1.0 else None))


# ----------------------------------------------------------------- sim
#: Fill models, worst-to-best. `bot` MIRRORS bot_runner.
#:
#: bot_runner does NOT pay the ask. It posts
#:     entry_limit = min(round((bid + ask)/2 + 0.01, 2), round(ask, 2))
#: i.e. MID + ONE TICK capped at the ask (bot_runner.py:1416), and exits at
#:     max(0.01, round(bid - spread * (0.5 if taking profit else 1.5), 2))
#: So the live model is ASYMMETRIC: optimistic entry, pessimistic exit. Charging
#: the full ask on entry -- what every sim in this repo did before 2026-09-10 --
#: invents roughly a half-spread of cost per trade, which on a wide 0DTE contract
#: can exceed the edge being measured.
#:
#: THE FILL ASSUMPTION IS NOT A DETAIL. For any ticker whose spread exceeds its
#: measured edge, the assumption DETERMINES THE SIGN of the answer, and picking
#: a price between bid and ask is arbitrary. Use `check_fill_sensitivity.py` to
#: see which conclusions survive the WHOLE BAND rather than trusting one point.
#:
#: CAVEAT on `bot`: DRY_RUN assumes the mid+1 limit fills. On a tight, deep chain
#: (NVDA/SPY 0-1DTE) that matches live experience. On a thin, wide one (SMH ATM
#: 0DTE trades in the hundreds, not thousands) the limit may not fill at all, so
#: the trade is missed or chased. `bot` is therefore an UPPER bound on entry
#: quality for illiquid names; fill-PROBABILITY risk is reported separately as
#: spread/volume, never baked into the P&L.
FILL_MODELS = {
    "mid":    "mid in, mid out (frictionless -- fantasy upper bound)",
    "bot":    "mid+1tick in (capped at ask), bid-cushion out -- MIRRORS bot_runner",
    "askbid": "ask in, bid out (no cushion)",
    "worst":  "ask in, bid-1.5xspread out (symmetric-pessimistic lower bound)",
    "botcap": "like `bot`, but the exit cushion is CAPPED at the bid's own "
              "measured intra-minute movement (see CUSHION_CAP)",
}

#: Per-ticker cap on the exit cushion, in spreads. Measured 2026-09-16 from the
#: 2026-09-01..09-15 bronze tape (18.9M prints, dte<=1, mid>=$0.30), which is the
#: first data carrying NBBO depth.
#:
#: WHAT THE MEASUREMENT ACTUALLY SHOWED, because the naive reading is wrong:
#:   * A fill, GIVEN a quote, is nearly free: 86% of small sells print at exactly
#:     the bid, 11.5% better, 2.1% worse; clipped mean cost 0.028 spreads. So the
#:     1.5x cushion is NOT modelling fill slippage.
#:   * What it is really proxying is TRIGGER TIMING. `simulate` evaluates the
#:     trail on a minute-CLOSE bid; the live bot evaluates continuously and exits
#:     on the first intra-minute touch, which in a falling market is worse than
#:     the close. That is a genuine, systematic, one-way cost.
#:   * Within-minute bid dispersion is large (mean 3.90 spreads overall) but it is
#:     SYMMETRIC NOISE and must NOT be added to the cost -- only the adverse half
#:     is real, and its size is not directly observable.
#:
#: The cap below is the physically motivated bound on that timing cost: the bid
#: cannot cost you more than it actually moved, so the charge is limited to half
#: the mean within-minute range.
#:
#: ⚠️ `botcap` IS NOT UNIFORMLY MORE OPTIMISTIC THAN `bot`. An earlier version of
#: this note claimed it was, and that stopped being true once the multiplier was
#: keyed on the exit TAG rather than on profitability. A PROFITABLE trail/stop
#: now pays the cap (e.g. 1.22) where the legacy rule charged 0.5, because the
#: trigger is adverse whether or not the trade is green. So the re-pricing cuts
#: both ways and a rule's direction cannot be assumed from the cap alone.
#: SPY and QQQ are unchanged (their bid moves further than 1.5 spreads anyway);
#: the relief lands on the wide-spread names, where 1.5 spreads was charging
#: 2.4-3.8x the bid's observed movement.
#: A MISSING TICKER IS NOT A NEUTRAL DEFAULT -- `botcap` falls back to the
#: uncapped 1.5 and the rule is silently NOT re-priced. That produced a spurious
#: "+0.0, unchanged" for AMZN/LULU/TSLA on the first disabled-rule sweep, which
#: read as "the cap changes nothing here" when it meant "never measured".
#: Run measure_cushion_cap.py before adding any ticker.
#: MOVED TO config.py (2026-09-18) and re-exported here, so every existing
#: `sim_core.CUSHION_CAP` caller keeps working unchanged. They had to move
#: because bot_runner imports no research module and was therefore still
#: charging the RETIRED 1.5 on every live exit -- on GLD, whose measured cap is
#: 0.40, that priced exits below the floor and booked seven ~-99% fills in one
#: session. A constant the live bot cannot import is a constant that will drift
#: away from the thing it is supposed to govern (METHODOLOGY 1).
from config import ADVERSE_TAGS, CUSHION_CAP, FILL_COST  # noqa: F401  (re-export)


def simulate(path, pol, eod_m, realistic=True, cushion=True, fill=None, sig=None,
             cush_cap=None, tighten=None, trail_tight=None):
    """One trade. -> (pnl_fraction, exit_minute_of_day, exit_tag).

    Levels are set off the entry MID (a live bracket is priced off the mid).
    `fill` selects a FILL_MODELS entry and overrides realistic/cushion.
    Legacy mapping when `fill` is None:
        realistic=False               -> "mid"
        realistic=True, cushion=False -> "askbid"
        realistic=True, cushion=True  -> "bot"     <-- the HONEST CENTRAL model

    NOTE on that last line: it used to map to "worst" (ask in, bid-1.5x out).
    That over-corrected. The pre-2026-09-10 sims were too OPTIMISTIC (no exit
    cushion); mapping the legacy flags to "worst" made every un-migrated caller
    too PESSIMISTIC instead, because bot_runner never pays the ask -- it posts
    min(mid+0.01, ask). "bot" is what the bot actually does, so it is the right
    default. "worst" remains available but must now be asked for explicitly, as
    a bound rather than an estimate.
    """
    import directional_flow_backtester as D
    if len(path) == 8:
        e_mid, e_ask, cl, hi, lo, bid, ask, mod = path
    else:                                  # legacy 7-tuple: no cushion possible
        e_mid, e_ask, cl, hi, lo, bid, mod = path
        ask, cushion = None, False

    if fill is None:
        fill = "mid" if not realistic else ("bot" if cushion else "askbid")
    if fill not in FILL_MODELS:
        raise ValueError(f"unknown fill model {fill!r}; expected {list(FILL_MODELS)}")
    if fill in ("bot", "botcap") and ask is None:
        fill = "askbid"                    # legacy payload cannot price a cushion
    realistic = fill != "mid"
    cushion = fill in ("bot", "botcap", "worst")
    if fill != "botcap":
        cush_cap = None                    # the cap applies to `botcap` alone

    if fill == "mid":
        entry = e_mid
    elif fill in ("bot", "botcap"):
        entry = min(round(e_mid + 0.01, 2), round(e_ask, 2))   # bot_runner.py:1416
    else:
        entry = e_ask
    ref = e_mid
    n = len(cl)
    tp_lvl = ref * (1 + pol["tp"]) if pol.get("tp") else None
    hard = ref * (1 - pol["stop"]) if pol.get("stop") else None
    trail = pol.get("trail")
    arm, give = pol.get("arm"), pol.get("give")
    if tp_lvl is not None and trail and not pol.get("allow_tp_with_trail"):
        import warnings
        warnings.warn(
            f"policy {pol.get('name', pol)!r} sets BOTH tp and trail. "
            f"bot_runner cannot run this -- its if/elif chain makes the "
            f"take-profit unreachable while trailing, so these results describe "
            f"a policy the live bot would not execute. Pass "
            f"allow_tp_with_trail=True to silence this if the divergence is "
            f"intended.", RuntimeWarning, stacklevel=2)
    peak = ref

    def _out(i, lvl_hint, tag):
        px = cl[i] if not realistic else (bid[i] if bid[i] > 0 else min(lvl_hint, cl[i]))
        # a level-triggered exit cannot fill BETTER than the level that fired it
        if realistic and tag in ("stop", "trail", "give"):
            px = min(px, lvl_hint)
        if cushion and ask is not None and bid[i] > 0 and ask[i] > bid[i]:
            sp = ask[i] - bid[i]
            if cush_cap is None:
                m = 0.5 if px > entry else 1.5          # legacy `bot`, untouched
            else:
                # `botcap`. Two corrections to the legacy rule:
                #
                # 1. IT KEYS ON THE TAG, NOT ON PROFITABILITY. The legacy rule
                #    picks 0.5 vs 1.5 on `px > entry`, but adverse slippage is
                #    not about whether the trade made money -- it is about WHY
                #    we are selling. A trail or stop fires precisely because the
                #    bid is falling, so the bid keeps falling while the order
                #    travels. An EOD flatten has no directional trigger at all,
                #    and a take-profit fires on a RISING bid. Charging an
                #    unprofitable EOD exit 1.5 spreads models nothing.
                # 2. NON-ADVERSE EXITS PAY ONLY THE MEASURED FILL COST. The
                #    order is marketable (limit below the bid), so it crosses and
                #    fills AT the bid; 18.9M prints put that cost at 0.028
                #    spreads. Worked example, AVGO 2026-09-15: booked 4.22 at the
                #    limit vs a 4.50 bid with 226 contracts resting -- the sim
                #    understated a real winner by 9.0pp.
                m = cush_cap if tag in ADVERSE_TAGS else FILL_COST
            px = max(0.01, px - sp * m)
        return (px - entry) / entry - D.COMMISSION_PCT, int(mod[i]), tag

    armed = False
    for i in range(n):
        if mod[i] >= eod_m:
            return _out(i, cl[i], "eod")
        # THE CHOKEHOLD. `tighten` is a per-minute bool; once it fires the trail
        # switches to `trail_tight` and STAYS there (it latches -- a detector
        # that flickers on and off would otherwise widen the stop again right
        # after flagging danger).
        # This is NOT a position reduction: full size keeps running, only the
        # stop distance changes. That matters, because the one exit intervention
        # that ever helped here (+5.2pp, leg-in vertical) was the only one that
        # did not cut the position, and six families that did cut it all failed.
        if tighten is not None and not armed and i < len(tighten) and tighten[i]:
            armed = True
        trail = (trail_tight if (armed and trail_tight is not None)
                 else pol.get("trail"))
        lvl, tag = hard, "stop"
        if trail:
            t = peak * (1 - trail)
            if lvl is None or t > lvl:
                lvl, tag = t, "trail"
            if arm is not None:
                proe = peak / ref - 1.0
                if proe >= arm:
                    g = ref * (1.0 + proe * (1.0 - give))
                    if g > lvl:
                        lvl, tag = g, "give"
        if lvl is not None and lo[i] <= lvl:
            return _out(i, lvl, tag)
        # 🚨 TP + TRAIL IS NOT A POLICY THE LIVE BOT CAN RUN.
        # bot_runner's monitor loop is an if/elif chain where
        # `elif trail_stop is not None:` SWALLOWS the take-profit branch, so
        # `bid >= tp_limit` is unreachable whenever a trail is armed -- that is
        # deliberate (TP+trail was OOS -1.1% vs +4.8% pure). Here the two are
        # ranked instead of exclusive, so a policy setting BOTH is scored with a
        # take-profit the bot would never fire. No deployed rule does this --
        # policy_for returns a trail OR a bracket, never both -- but a
        # hand-built exploratory policy can, so it is flagged rather than
        # silently mis-scored. See check_exit_bracketology's tp*+trail50 rows.
        if tp_lvl is not None and cl[i] >= tp_lvl:
            return _out(i, tp_lvl, "tp")
        # DISCRETIONARY SIGNAL EXIT, ranked AFTER the protective levels so it can
        # never pre-empt a stop or trail that would have fired on the same bar.
        # `sig` is a per-minute bool array aligned with `mod`; it exits at the
        # close, through the same _out() as everything else. Defaulting to None
        # keeps every existing caller byte-identical -- check_signal_exits once
        # diverged from this function on five fill conventions at a cost of
        # 4.57 vs 1.60 in reported return, so signal exits live HERE now.
        if sig is not None and i < len(sig) and sig[i]:
            return _out(i, cl[i], "sig")
        peak = max(peak, cl[i])
    return _out(n - 1, cl[-1], "eod")


def walk(cand, pol, eod_m, realistic=True, cushion=True, fill=None,
         cap=None, skip=0, after=None, with_tags=False, picks_out=None,
         sigs=None, cush_cap=None, tightens=None, trail_tight=None):
    """Sequential one-position-per-ticker replay. -> [(date, pnl)] or
    [(date, pnl, tag)] when `with_tags`. `cand` must be sorted by (date, mod).

    `sigs`: optional list aligned with `cand`, each entry a per-minute bool
    array (or None) handed to `simulate` as a discretionary signal exit. Because
    the sequential guard sets `busy` to the EXIT minute, a signal that closes a
    position early genuinely FREES THE SLOT -- the next trigger that day can be
    taken. So a signal exit is a re-entry policy, not a position reduction, which
    is a different animal from the trail/give-back/scale-out family.

    `picks_out`: pass a LIST to receive, for each emitted trade, a
    `(cand_index, exit_minute)` pair -- which candidate was actually TAKEN and
    when it closed. The sequential guard means most candidates are refused, so
    there is otherwise no way to map a result row back to its entry minute or
    its option price path. Used by the trade charting export.
    """
    cur, busy, ntd, nsk, out = None, -1, 0, 0, []
    for ci, (d, m, path) in enumerate(cand):
        if d != cur:
            cur, busy, ntd, nsk = d, -1, 0, 0
        if m < busy:
            continue                       # position open -- the live guard
        if after is not None and m < after:
            continue
        if nsk < skip:
            nsk += 1
            continue                       # stay FLAT; next trigger eligible now
        if cap is not None and ntd >= cap:
            continue
        pnl, xm, tag = simulate(path, pol, eod_m, realistic, cushion, fill,
                                sig=(sigs[ci] if sigs is not None else None),
                                cush_cap=cush_cap,
                                tighten=(tightens[ci] if tightens is not None else None),
                                trail_tight=trail_tight)
        out.append((d, pnl, tag) if with_tags else (d, pnl))
        if picks_out is not None:
            picks_out.append((ci, xm))
        busy = xm
        ntd += 1
    return out


# ----------------------------------------------------------------- build
#: Rules that remain ENABLED and PAPER-TRADING in config.py, but are EXCLUDED
#: FROM RESEARCH by default from 2026-09-12 (user's decision).  This list changes
#: nothing about what the bot does -- `bot_runner` never reads it.  It exists so
#: that new studies default to the working set instead of silently re-deriving
#: conclusions from four rules already slated for removal.
#:
#:   AVGO / GLD / SMH  fail the A PRIORI median-entry-spread screen (6.7%, 6.2%,
#:                     9.2% vs 1.1-4.0% for the rest).  That screen uses no
#:                     performance data at all, so these three carry ZERO
#:                     selection bias -- the strongest kind of removal.
#:   MSFT CHOP PUT     PASSES the spread screen (3.7%).  It is excluded on its
#:                     P&L, which is selection on the outcome -- but it is
#:                     negative in BOTH halves (IS -13.6% / OOS -13.9%, n=36,
#:                     win 0.36), and a both-halves loser is far harder to
#:                     attribute to selection than a best-of-N pick.
#:
#: See check_core_book.py.  Pass include_paper=True to score the full book.
PAPER_ONLY = ("MSFT CHOP PUT", "SMH LOWVOL PUT", "AVGO HIVOL PUT", "GLD amp1 CALL")


def research_rules(include_paper: bool = False):
    """The rule set a NEW STUDY should default to: enabled, minus `PAPER_ONLY`.

    Always prefer this over `[r for r in RULES if r.get("enabled")]` in new
    analysis code, so the working set is defined in one place. `include_paper`
    restores the full enabled book for regression checks against older results.
    """
    from config import RULES
    rs = [r for r in RULES if r.get("enabled", True)]
    if include_paper:
        return rs
    return [r for r in rs if r["name"] not in PAPER_ONLY]


#: First session of the DEPLOYED research window. `build_candidates` refuses to
#: emit candidates before this date unless explicitly asked.
#:
#: WHY THIS EXISTS. On 2026-09-12 the lake was backfilled to 2023-10-12 as a
#: PRE-SAMPLE HOLDOUT (PRESAMPLE_PLAN.md). Within minutes `check_book_now`
#: silently reported n=336 instead of 297 and an in-sample mean of -1.1%
#: instead of +4.9% -- because it has no date floor, so the holdout landed in
#: its "IS" bucket and contaminated it. Every one of the ~88 scripts here has
#: the same shape, so the floor belongs in the ONE builder they all route
#: through, not in each of them.
#:
#: Pass `since=None` to include the pre-sample window. `check_presample.py` is
#: the only caller that should, and only after PRESAMPLE_PLAN.md is signed.
DEPLOYED_START = _dt.date(2024, 8, 20)


def build_candidates(D, rule, trigs=None, min_mid=0.50, meta_out=None,
                     since="deployed"):
    """The ONE candidate builder. -> [(date, minute_of_day, payload)] sorted.

    Applies the rule's regime/hour/dte/flow-percentile gates and the `amt_open`
    gate, picks the contract the bot would pick, and enforces the $0.50 entry
    floor that `bot_runner.py:1421` enforces live. Pass `trigs` to score an
    ALTERNATIVE trigger series against the same gates (see check_hedge_signal).

    `meta_out`: pass a LIST to receive one dict of ENTRY-TIME metadata per
    candidate, in the same order as the returned list -- spot, dte, strike, the
    entry bid/ask/mid and the contract id. It exists so a study can build
    entry-time features (spread, premium, moneyness, time budget) WITHOUT
    re-deriving the trigger match and contract pick in a second, drifting copy
    of this function -- the failure METHODOLOGY.md 1 is about. The returned
    tuple shape is deliberately unchanged, so `walk`/`simulate` and every
    existing caller are unaffected.
    """
    from check_config_walkforward import _flow_for
    from amt_profile import amt_open_map, amt_ok

    tk = rule["ticker"]
    if trigs is None:
        flow = _flow_for(D, [tk])
        if flow.empty:
            return []
        trigs = D.triggers_for(flow, tk)
        D.annotate_flow_pct(trigs, rule.get("flow_window_days", 60))

    gex = D.load_gex(HIST, tk)
    vol = D.load_volume_regime(HIST, tk)
    trd = D.load_trend_regime(HIST, tk)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP")
           for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol,
               "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}

    try:
        tb = D._ticker_bars(tk)
    except Exception:
        tb = None
    if tb is None or tb.empty:
        _, tb = D._screen_build_one("lake/silver/option-contracts-1m", tk)
    if tb is None or tb.empty:
        return []
    bbc = {c: g.sort_values("minute_et") for c, g in tb.groupby("option_chain_id")}
    # 🚨 THE SORT IS LOAD-BEARING (added 2026-09-20). `_ticker_bars` is NOT
    # time-ordered -- it is (contract x minute) rows straight off a parquet
    # predicate pushdown -- so without this, `at.iloc[-1]["underlying_close"]`
    # below returned the last row in FILE ORDER at/before the trigger, not the
    # latest MINUTE. The resulting spot was stale by a median +35bp (META),
    # +34bp (NVDA), +108bp (AVGO) and -5 to -7bp (SPY/QQQ), and because the
    # staleness points backwards it biased UP after a decline (PUT triggers) and
    # DOWN after a rally (CALL triggers) -- a clean direction-split artefact.
    # It changed the SELECTED STRIKE on 8.8% (SMH) to 81.1% (QQQ) of candidates,
    # typically by one strike, systematically toward ITM. Sorting brings spot to
    # within +-0.6bp of the underlying's own 1m close, so `underlying_close` was
    # never the problem. See check_arrival.py, which is what surfaced it.
    bbd = {d: g.sort_values("minute_et") for d, g in tb.groupby("date")}

    matched = D._rule_matched_trigs(rule, trigs, gex, vol, trd, amp, reg_src)
    if rule.get("amt_open"):
        amt = amt_open_map(tk)
        matched = [(t, th) for t, th in matched if amt_ok(rule["amt_open"], amt.get(t["date"]))]
    matched.sort(key=lambda x: pd.Timestamp(x[0]["ts"]))

    floor = DEPLOYED_START if since == "deployed" else since
    out = []
    for t, _th in matched:
        d, ts = t["date"], t["ts"]
        if floor is not None and d < floor:
            continue                       # pre-sample holdout -- see DEPLOYED_START
        day = bbd.get(d)
        if day is None:
            continue
        at = day[day["minute_et"] <= ts]
        if at.empty:
            continue
        spot = float(at.iloc[-1]["underlying_close"])
        cid, cdte = None, None
        # STRIKE OFFSET: honour what the rule declares, because bot_runner does
        # (bot_runner.py:1785). Until 2026-09-19 this call had no offset
        # argument at all, so research scored the nearest strike while the live
        # bot bought `strike_offset` strikes OTM -- a silent divergence for any
        # rule carrying the field. Default 0 = ATM = unchanged for every rule
        # that does not set it.
        _off = int(rule.get("strike_offset", 0) or 0)
        for dd in rule.get("dte", [0, 1]):
            cid = D.pick_contract(day, ts, rule["direction"], dd, spot,
                                  offset=_off)
            if cid is not None:
                cdte = dd
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
        if mid < min_mid:
            continue                       # bot_runner.py:1421 entry floor
        fwd = ent[ent["minute_et"] > ts].sort_values("minute_et")
        if len(fwd) < 3:
            continue
        pm = fwd["minute_et"]
        out.append((d, pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute,
                    (mid, k if k > 0 else mid,
                     fwd["close"].to_numpy(float), fwd["high"].to_numpy(float),
                     fwd["low"].to_numpy(float), fwd["bid_close"].to_numpy(float),
                     fwd["ask_close"].to_numpy(float),
                     (pm.dt.hour.values * 60 + pm.dt.minute.values).astype(int))))
        if meta_out is not None:
            meta_out.append(dict(date=d, cid=cid, dte=cdte, spot=spot,
                                 strike=float(er.get("strike", float("nan"))),
                                 bid=b, ask=k, mid=mid,
                                 mod=out[-1][1]))
    return out


# ----------------------------------------------------------------- stats
def stat(rows):
    """Standard IS/OOS/slice summary. `rows` = [(date, pnl)] or [(date, pnl, tag)]."""
    from check_config_walkforward import _slice_idx
    if not rows:
        return None
    v = np.array([r[1] for r in rows], float)
    ds = [r[0] for r in rows]
    i = np.array([r[1] for r in rows if r[0] < SPLIT], float)
    o = np.array([r[1] for r in rows if r[0] >= SPLIT], float)
    sl = [[] for _ in range(6)]
    for r in rows:
        k = _slice_idx(r[0])
        if k is not None:
            sl[k].append(r[1])
    pop = [np.mean(b) for b in sl if len(b) >= 3]
    w, l = v[v > 0], v[v <= 0]
    tags = {}
    for r in rows:
        if len(r) > 2:
            tags[r[2]] = tags.get(r[2], 0) + 1
    return dict(n=len(v), nd=len(set(ds)), all=v.mean(),
                is_=i.mean() if len(i) else np.nan, nis=len(i),
                oos=o.mean() if len(o) else np.nan, noos=len(o),
                win=(v > 0).mean(), mw=w.mean() if len(w) else np.nan,
                ml=l.mean() if len(l) else np.nan,
                npop=len(pop), nposs=sum(1 for x in pop if x > 0),
                tot=v.sum(), tags=tags)


def line(lbl, rows, base_oos=None, width=26):
    s = stat(rows)
    if s is None or s["n"] < 12:
        return f"  {lbl:{width}} n={0 if s is None else s['n']:>5}  (thin)"
    d = f" {(s['oos']-base_oos)*100:>+6.1f}pp" if base_oos is not None else ""
    return (f"  {lbl:{width}} n={s['n']:>5} d={s['nd']:>4} "
            f"IS {s['is_']*100:>+7.1f}% OOS {s['oos']*100:>+7.1f}%{d} "
            f"win {s['win']:>4.2f} sl {s['nposs']}/{s['npop']}")
