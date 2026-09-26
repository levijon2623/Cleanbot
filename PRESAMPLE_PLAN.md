# Pre-sample holdout: pre-registration

**Window:** 2023-10-12 → 2024-08-19, 214 trading sessions.
**Status:** written 2026-09-12, BEFORE any pre-sample data was scored.
**Binding rule:** nothing in this window is looked at until this document is
approved. Every test run on it is listed here. Anything not listed is a new
hypothesis and must wait for new forward data, not this window.

---

## 1. Why this document exists

2023-10-12 is the API's history floor — both `net-prem-ticks` and
`option-trades/full-tape` 403 before it, measured by binary search against the
live API. **There is no more history behind this.** Every rule, parameter, exit,
regime gate and conclusion in this project was developed on 2024-08-20 onward,
so these 214 sessions are the only genuinely unseen data that will ever exist
for this book. Once looked at, they are spent.

That is the entire argument for pre-registering: a holdout you search freely is
not a holdout.

## 2. What this buys — stated honestly, before the fact

The book does not trade every session. Measured on the existing lake:

| book | OOS trading days | OOS sessions | fill rate |
|---|---|---|---|
| all9 | 77 | 252 | 30.6% |
| core5 | 46 | 252 | 18.3% |

So 214 pre-sample sessions yield roughly **60 fill-days (all9)** or **39
(core5)** — not 214.

**And ~48 of the 214 sessions are lost to flow-percentile warmup.**
`annotate_flow_pct` sets `thr = None` until a *full* `window_days` of prior
trigger history exists, and `flow_pairs_for` then returns `[]` — the trigger is
**dropped entirely, not merely mis-scored**. The pre-sample begins at the API's
history floor, so there is nothing behind 2023-10-12 to warm it with. **Slice P1
(2023-10-12 → 2023-12-20, 48 sessions) will therefore be empty by construction**,
and the usable window is ~166 sessions, ~46 fill-days (all9) / ~30 (core5).

Holdout days go 77 → ~123, and the minimum detectable effect scales by
√(77/123) = 0.79:

> **MDE ≈ 65pp → ≈ 51pp per trade.** A ~20% resolution gain, not a halving.

P1 being empty is expected and is **not** evidence of anything. Do not read it
as a regime result.

**This is decisive.** 49pp is still far too coarse to power a conditioner
search — the IVR, term-structure and skip-rule hunts would all remain
underpowered. So the backfill is *not* worth spending on feature discovery.

It is worth spending on **one question it can actually answer**: does the book
itself survive on data nobody has touched? That is a single high-level test with
a large expected effect (the book claims +19.1%/trade), and it is exactly the
kind of question 39–60 fresh days can settle.

## 3. The tests — complete list, five of them

Scored through `sim_core` exactly as deployed: sequential fills, per-rule
`policy_for` exits, `fill="bot"`, `$0.50` entry floor. No parameter is refit, no
threshold is re-chosen, no rule is re-selected. Inference is a day-block
bootstrap resampling whole sessions, 95% percentile CI.

| # | Hypothesis | Statistic | Passes if |
|---|---|---|---|
| **T1a** | **The core5 book is profitable on unseen data.** The primary test. | mean return/trade, pre-sample | CI lower bound **> 0** |
| **T1b** | **Index-only (SPY/QQQ/IWM) is profitable on unseen data.** The contamination-free version of T1a — see below. | mean return/trade, pre-sample | CI lower bound **> 0** |
| T2 | The a-priori spread screen still helps. | spread6 − all9, pre-sample | point estimate **> 0** (CI reported, not required — this was never significant in-sample either) |
| T3 | The four excluded rules are still bad. | MSFT / SMH / AVGO / GLD mean, each | each **< 0** |
| T4 | The VIX-regime conditioning holds. | high-VIX − low-VIX mean | **> 0**, same sign as deployed |
| T5 | The sampling-policy invariance holds. | sequential vs as-screened OOS sign | **same sign** |

Six tests. At 95%, **expect ~0.3 false positives** — so one marginal star is
noise and will be reported as such. T1a and T1b are not independent (T1b is a
subset of T1a) and count as one finding, not two.

### Readiness finding: it is IWM, not META/NVDA (recorded 2026-09-12, pre-scoring)

The full-sample P3 check overturned the n=25 preview that prompted the split.
Measured on all 214 sessions, share offering a same-day expiry, pre-sample vs
today:

| ticker | pre-sample | today | |
|---|---|---|---|
| SPY | 100.0% | 100.0% | comparable |
| QQQ | 100.0% | 100.0% | comparable |
| **IWM** | **77.1%** | 100.0% | **−22.9pp** |
| META | 21.0% | 30.2% | −9.1pp, comparable |
| NVDA | 21.0% | 30.2% | −9.1pp, comparable |

META/NVDA are fine. **IWM is the problem, and it sits inside T1b.** By slice,
IWM has a 0DTE on 60.4% of P1, **63.9% of P2**, and **100.0% of P3** — it gained
full daily expiries at ~2024-04-18, essentially the P2/P3 boundary. Of the 49
sessions with no IWM 0DTE, 19 are in P1 (empty anyway) and **30 fall in P2 and
do contaminate T1b**.

Consequence, fixed before any outcome was seen: **T1b is reported twice** — over
the whole usable window (P2+P3) with IWM's per-slice expiry mix disclosed, and
restricted to **P3 only (83 sessions), where all three index rules have exactly
today's contract structure.** Where the two disagree, the P3-restricted read
governs the structural question. SPY and QQQ alone are comparable throughout and
are reported separately as a third cut.

### Why T1 is split

Readiness check P3 found that in the pre-sample window **META and NVDA offer a
same-day expiry on only ~20% of sessions — exactly 1-in-5, i.e. Friday-only
weeklies — against 30.2% today.** SPY/QQQ/IWM are 100% in both windows. If that
holds on the full sample, then META LOWVOL PUT and NVDA LOWVOL PUT would be
filling **1DTE contracts pre-sample where they fill 0DTE today**: a structurally
different trade, not merely an unluckier one. Two of core5's five rules are
affected.

So T1a (whole book) can fail for a reason that has nothing to do with the edge
decaying, and **T1b is the clean read** — the three index rules pick an
identical contract in both windows. Both are reported; where they disagree, T1b
governs the conclusion about the edge and T1a governs the conclusion about the
book as currently configured. The realised 0DTE/1DTE mix per rule is reported
alongside, so the reader can see the shift rather than take it on trust.

## 4. What will NOT be done

- No feature search, ML scan, or grid of any kind.
- No new conditioners (IVR, term structure, skip rules, greeks — none).
- No parameter tuning, threshold re-fitting, or rule re-selection.
- No re-running a test with a different cut after seeing its result.
- No promoting a "thread" found here into a finding without fresh forward data.

If a result suggests a new hypothesis, it is written down and tested on **future
forward data**, never on this window again.

## 5. Interpretation rules, fixed in advance

**Regime confound — the big one.** 2023-10 → 2024-08 was a low-VIX bull run. The
deployed VIX overlay says this book earns **+24%/trade when VIX ≥ ~18 and only
+7% below**. So a *weaker* pre-sample result is the predicted outcome, not a
refutation. Before interpreting T1, report the pre-sample VIX distribution
against the IS and OOS windows, and compare the book like-for-like within VIX
buckets. A soft T1 in a low-VIX window is consistent with the book; a soft T1
with a VIX mix comparable to OOS is not.

**Partition hygiene.** The pre-sample is a **third partition**. It is not merged
into IS, and the 2025-08-21 IS/OOS split does not move. After this test it is
burned for hypothesis testing and may be used only as additional in-sample
context, always labelled.

**The in-sample baseline moved before this test was run, and the comparison
must use the NEW numbers.** Backfilling netprem warmed the flow-percentile
window for the start of the deployed period, which had silently produced **zero
trades** for its first ~2 months in every backtest this project has ever run.
Correcting it adds 39 trades to the in-sample half and every book goes
IS-negative:

| book | n (was) | all (was) | IS (was) | OOS |
|---|---|---|---|---|
| all9 | 336 (297) | +4.9% (+8.4%) | **−1.1%** (+4.9%) | +11.5% (unchanged) |
| spread6 | 253 (220) | +8.1% (+13.7%) | **−2.2%** (+5.8%) | +21.8% (unchanged) |
| core5 | 207 (184) | +11.4% (+19.1%) | **−2.5%** (+8.8%) | +30.5% (unchanged) |

OOS is bit-identical — its trailing window was always fully warmed. So the book
now reads **IS −2.5% / OOS +30.5%**, an inversion of the usual selection
pattern, and it exists because the rules were chosen on an in-sample window that
silently excluded its own worst stretch. Any pre-sample result is compared
against **IS −2.5%**, not the retired +8.8%.

**A negative T1 is a real finding, not a data problem.** If core5 is negative
pre-sample with a CI excluding zero, and the VIX mix does not explain it, the
honest conclusion is that the book's edge does not generalise backward — and
that outranks every positive result in this repo. It gets written up as such.

## 6. Prerequisites before any scoring

1. **Extend `SLICE_EDGES`.** It currently starts 2024-08-20 and `_slice_idx`
   returns `None` before that, so pre-sample trades are **silently dropped**
   from every slice-coverage statistic. Add edges back to 2023-10-12
   (≈ 2024-04-20, 2023-12-20, 2023-10-12) — three more slices.
2. **Verify coverage.** Every one of the 214 sessions must have netprem, OHLC,
   GEX and a silver partition, or the missing days are listed and excluded
   explicitly. No silent gaps.
3. **Verify 0DTE availability.** The rules are 0/1DTE. SPY/QQQ/IWM had daily
   expiries through this window; META/NVDA rely on weeklies landing on the trade
   day (they run 57.6% / 62.8% 0DTE in the current lake). Confirm the pre-sample
   rate is comparable — a large drop would mean the book is structurally
   different there, not just unluckier.
4. **Re-check the greek/IV corruption window.** The lake's `iv_close` is broken
   for `dte==0` from ~2024-10 to ~2025-12. Establish whether the pre-sample
   window is clean before any IV-dependent number is quoted from it.

## 6a. RESULTS — recorded 2026-09-12, run once

```
                          n    mean     win   days   slices         95% CI
  T1a core5              55  -17.2%    0.27    30   P1=3 P2=20 P3=32   [-30.1,  -0.1]  FAIL
  T1b index-only         45  -17.0%    0.27    24   P1=3 P2=16 P3=26   [-31.9,  +2.8]  FAIL
  T1b-P3 (clean struct)  26  -22.4%    0.23    13   P3=26              [-41.1,  +6.4]  FAIL
  T1b-SQ SPY+QQQ         14  -39.7%    0.07     8   P2=8 P3=6          [-48.3, -19.4]  FAIL
  T2  spread6-all9           +2.7pp                                    [-10.8, +15.9]  PASS
  T3  all four excluded rules negative pre-sample                                      PASS
  T4  VIX high-low           +3.5pp                                    [-27.8, +34.8]  PASS
  T5  seq -17.2% vs screened -8.1%, same sign                                          PASS
```

**T1 FAILS. The book loses money on data it has never seen**, and the CI on the
primary test excludes zero. The three windows now read:

> **pre-sample −17.2%  ·  in-sample −2.4%  ·  out-of-sample +30.5%**

Negative in two of three. The +30.5% is the outlier, not the rule.

**The VIX confound does NOT explain it.** That was the pre-registered escape
hatch and it does not hold. The pre-sample is indeed much calmer (VIX median
13.7, 12.7% of days ≥18, vs 17.6/46.2% in-sample), and the deployed overlay
predicts a *weaker* +7% rather than +24% — but it predicts **weaker, not
negative**, and −17.2% is nowhere near +7%. Decisively, **T4 cuts VIX inside the
pre-sample and both buckets are deeply negative**: high-VIX days −15.9%,
low-VIX −19.3%. Conditioning on VIX does not rescue the window.

**The cleanest cut is the worst.** T1b-SQ (SPY+QQQ, the only two rules whose
contract structure is identical in both windows, so no expiry-structure excuse
is available) returns **−39.7% with a win rate of 0.07 — one winner in
fourteen**, CI [−48.3, −19.4]. The structural confounds that were supposed to
protect T1a are not what is driving the failure; removing them makes it worse.

**T3 is the one real positive, and it is worth having.** All four excluded rules
are negative on unseen data too — AVGO −24.2%, SMH −27.8%, MSFT −24.0%,
GLD −11.8%. The removal was correct, and now for a reason that owes nothing to
selection on in-sample P&L.

**T2, T4 and T5 pass on their pre-registered criteria but carry almost no
information** — every interval is enormous ([−10.8,+15.9], [−27.8,+34.8]). They
should not be cited as support for anything.

### Caveats, stated with the result rather than after it
- n=55 (core5) over 30 trading days is thin, and the T1a CI upper bound is −0.1,
  i.e. only just excluding zero. T1b spans zero.
- P1=3 trades, as predicted by the warmup argument in section 2. Not a regime result.
- **NVDA LOWVOL PUT contributes zero pre-sample trades**, so "core5" is really
  four rules here. GLD/IWM/META/MSFT each show a >15pp shift in realised 0DTE
  share between the windows.
- What is *not* a caveat: the direction is consistent across all four cuts, and
  it sharpens as the structural confounds are removed.

## 7. Sign-off

**Approved by: user (Levi) — 2026-09-12**, after the T1a/T1b split in §3 was
added in response to the P3 expiry-structure finding.

The tests in §3 are run exactly once, by `check_presample.py`, and the results
are recorded whatever they say.

**Execution gate:** scoring does not begin until `check_presample_ready.py`
reports 214/214 silver partitions. Scoring a partially-backfilled window would
spend the holdout on an arbitrary subset of it, which is the same mistake as
searching it.
