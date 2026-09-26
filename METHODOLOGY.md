# Cleanbot Research Methodology

How to test an idea against this book without fooling yourself. Every rule here
was learned by getting it wrong first; the cost of each mistake is noted so the
rule doesn't get quietly dropped later.

Last updated 2026-09-10.

---

## 0. The one-paragraph version

Score every idea **sequentially** (one position per ticker), on **realistic
fills** (ask in, bid out) **with the live exit cushion**, at the **right unit**
(day-level for day-constant features, trade-level for within-day ones), split
**IS/OOS at 2025-08-21** across **6 calendar slices**, against **criteria
committed in writing before looking at output**, and — if the idea was selected
from a search — **walk-forward the selection itself**. Anything that skips a
step has produced a false positive here before.

---

## 1. Simulation: use `sim_core`, never hand-roll

`sim_core.py` is the only simulator. `build_candidates` / `simulate` / `walk` /
`policy_for` / `eod_mod` / `stat`. If you write your own, it will drift, and the
drift will be silent.

**This has happened twice.**

- The original backtest scored every matched trigger independently. The live bot
  holds **at most one position per ticker** (`bot_runner.py:1288`). Trade count
  was inflated ~10x and OOS expectancy ~2.5x.
- A second simulator appeared that modelled the live exit cushion; the older one
  did not and was ~10pp optimistic per trade. The stale one stayed behind
  `check_book_now` (what the book earns) and `check_trail_regression` (which
  sets `trail_pct` in config) for a full day of analysis.

**Three things the core gets right:**

| | why |
|---|---|
| sequential fills | the live one-position-per-ticker guard |
| realistic fills | enter ASK, exit BID — ~10pp book-wide vs mid |
| **exit cushion** | `bot_runner._fire_exit` posts `bid − 1.5×spread` on a loss, `bid − 0.5×spread` on a gain (Webull is limit-only) |

The cushion is **not neutral between policies**: losing exits pay 3× the
cushion, so it penalises low-win-rate policies harder. *Any policy comparison
run without it is biased toward whichever policy loses more often.* Verified
against live paper fills 2026-09-09 (11.5pp and 14.1pp of entry premium).

The true fill sits **between two bounds**: DRY_RUN books at the cushioned limit
(zero price improvement, pessimistic), a plain-bid sim is optimistic. Quote the
range, not a point.

### 1a. 🚨 The fill assumption can BE the result

**bot_runner does not pay the ask.** It posts `min(mid + 0.01, ask)` on entry
(`bot_runner.py:1416`; verified live — `entry_mid 1.29 → entry_price 1.30`) and
`bid − spread×(0.5|1.5)` on exit. The live model is **asymmetric**: optimistic
entry, pessimistic exit. Every sim here charged the full ask until 2026-09-10,
inventing about a half-spread per trade — worth +2.7pp book-wide and far more on
wide-quote names. Use `fill="bot"`; it is the honest central estimate.

**For a contract whose spread exceeds the edge, any price between bid and ask is
arbitrary and that arbitrary choice sets the SIGN of the answer.** A single point
estimate for such a rule is not a measurement, it is the assumption restated.

Run `check_fill_sensitivity.py` and classify:

| | |
|---|---|
| **ROBUST** | OOS keeps its sign across `mid` / `bot` / `askbid` / `worst` |
| **ASSUMPTION-BOUND** | the sign flips inside the band — we know nothing yet |

Measured 2026-09-10 (entry spread as % of premium, at the actual trigger
minutes):

```
                spr%med  spr%p90     mid      bot   askbid    worst    band
SPY CHOP CALL      1.1%     2.1%  +81.5%   +74.7%   +76.8%   +74.6%   6.9pp  ROBUST
QQQ HIVOL CALL     1.2%     2.4%  +51.9%   +44.6%   +47.3%   +44.2%   7.8pp  ROBUST
NVDA LOWVOL PUT    1.5%     3.2%  +27.3%   +21.1%   +23.9%   +18.9%   8.4pp  ROBUST
META LOWVOL PUT    4.0%     9.0%  +23.1%   +10.5%   +15.0%    +7.9%  15.3pp  ROBUST
MSFT CHOP PUT      3.7%     6.1%   -5.0%   -13.9%   -10.3%   -15.1%  10.1pp  ROBUST (neg)
GLD amp1 CALL      6.2%    14.1%   +2.1%   -17.4%    -7.9%   -20.6%  22.8pp  ASSUMPTION-BOUND
AVGO HIVOL PUT     6.7%    18.6%  +25.2%    -9.7%    +3.6%   -15.4%  40.7pp  ASSUMPTION-BOUND
SMH LOWVOL PUT     9.2%    28.4%  +49.9%    -4.2%   +18.9%   -11.6%  61.5pp  ASSUMPTION-BOUND
BLENDED BOOK                     +30.4%   +11.5%   +19.6%    +8.8%  21.6pp
```

**The two regimes are stark.** Tight, deep chains (SPY/QQQ/NVDA ~1% spread) give
7–8pp bands against 20–75% edges — the fill model is nearly moot, which matches
live experience that a mid+1 limit always fills there. Wide, thin chains flip
sign inside their own band: **SMH spans 61.5pp and is unmeasurable with this
data.**

**Rule: any per-rule claim smaller than its own band is a statement about the
assumption, not about the rule.** The blended book's +11.5% sits inside a 21.6pp
band — treat it accordingly.

**What this cannot settle:** `bot` assumes the mid+1 limit *fills*. On a thin
chain it may not, so the trade is missed or chased. That is fill PROBABILITY,
not fill PRICE, and no bar data resolves it — it needs live evidence. Treat
`bot` as an **upper bound on entry quality** wherever the quote is wide.

**Always use `sim_core.policy_for(rule)`** for "the deployed exit". Hardcoding a
trail ignores per-rule `trail_pct: 0` (META, NVDA) and silently changes the
sample. This produced mislabelled "DEPLOYED" baselines on 2026-09-10.

---

## 2. The unit of observation

Getting this wrong manufactures correlation out of nothing.

- **Day-constant features** (regime, VIX, GEX, `amt_open`) → **day-level**
  (equal-weight per day). Trades on one ticker-day are ~one bet.
- **Within-day features** (VWAP position, relvol, hour, entry location,
  cumulative flow) → **trade-level**. Day-level aggregation of a within-day
  feature *manufactures* correlation.
- **Rule of thumb:** if trades/days > ~5, re-cut equal-weight-per-day and report
  both. Screeners print `d=<days>` and a `D[...]` block for exactly this.

### 2a. 🚨 Overlapping forward windows

**If the forward horizon exceeds the sampling interval, the observations are not
independent and nearly every downstream number is wrong.** A 120m forward return
sampled every minute overlaps ~99% with the next one. 16,812 minute rows carry
roughly *43 days* of information.

This is not a small correction. `tick_sgn` at 120m on RTY posted **+51.89bp**,
cleared every pre-committed criterion, and beat its time-of-day control. On
non-overlapping windows it was **+5.15bp, p=0.771**. The overlap was the result.

Two acceptable treatments, in order of preference:

1. **Thin to non-overlapping windows.** RTH/horizon disjoint windows per day —
   at 120m that is 3/day. Run **every phase offset** as its own internally
   non-overlapping sample and require the sign to hold across phases; never pool
   phases, which reintroduces exactly the overlap you removed.
   (`check_mbo_nonoverlap.py`, and `thin()` in `check_wpoc_gate.py`.)
2. **Resample whole days**, so overlapping rows move together
   (`boot_ci` in `check_mbo_gating.py`).

**The trap inside the fix:** a day-block bootstrap that resamples days and then
*shuffles outcomes within* each day is NOT a valid null here — the shuffle
destroys the very autocorrelation the overlap creates, producing a null that is
easier to beat than reality and therefore an understated p. That is the bug that
made `tick_sgn` look real. Permute the **feature**, or resample days intact.

---

## 3. 🚨 The sample is a policy artefact

The single most important thing on this page.

The guard decides which triggers become trades. **The exit decides when the
ticker frees up, which decides which *subsequent* triggers become trades.** Two
exit policies don't score the same trades differently — they produce *different
trade populations*.

Measured on identical rules and triggers (2026-09-11, each rule scored on its
OWN deployed exit — the earlier printing of this table applied a book-wide
trail50, which is not what META/NVDA run):

```
as-screened     n=2139  IS +3.8%  OOS +10.5%   slices 4/6
seq (deployed)  n= 297  IS +3.0%  OOS  +9.3%   slices 3/6   <- deployed
seq (fast exit) n= 603  IS -3.1%  OOS  -6.1%   slices 1/6
first-trigger   n= 187  IS -1.2%  OOS  -5.6%   slices 2/6
```

**The book's OOS sign depends on the sampling rule.** The exit choice alone
moves the sample +103%.

**Consequence:** slice coverage — the criterion that has rejected almost every
candidate — is partly a property of *the guard*, not of *the rule*. Never report
"failed on slice coverage" as if it were a fact about the idea.

**The partial fix — two questions, two samples:**

| question | sample | policy-dependent? |
|---|---|---|
| does this gate select better triggers? | first trigger per ticker-day | **no — invariant** |
| what will the bot earn? | sequential + realistic + cushion | yes, correctly |

**Caveat that limits it:** first-trigger skews to the earliest trigger of the
day, and later-session beats the open. So it is **hour-confounded — valid for
relative gate comparisons, never for absolute expectancy.**

**Not part of this confound:** sizing / contracts-per-position. Everything is
scored as per-trade ROE, which is size-invariant. Sizing changes dollar P&L and
portfolio risk, not n, not trade identity, not any verdict.

---

## 4. Pre-committed criteria

Write the pass criteria into the script docstring **before** running it. This is
the single highest-yield habit in the project — the give-back family and the AMT
variants both had winners on raw OOS and both failed criteria fixed in advance.

The standard four:

```
C1  day-level (or trade-level, per §2) IS  > 0
C2  ... OOS > 0
C3  >= 5 of 6 calendar slices populated AND positive
C4  OOS beats the p95 of a bootstrap null drawn from that ticker+direction's
    own p50 trigger population, matched on n
```

Add **C5 cross-sectional consistency** for any wide search: the effect must show
up pooled across rules, not only in the one cell that got lucky. This is what
killed the hour-10 hypothesis.

Set the bar at a level a *known-bad* rule would fail. The candidate re-test bar
was deliberately set where LULU would have failed.

---

## 5. Walk-forward the *selection*, not just the strategy

A grid sweep is a search. Scoring the winner on the same data is circular.

**Pick the policy using only data before each slice, score it on that slice,
chain.** This is the most decisive arbiter in the project:

- killed the ROE give-back family (chained +4.5% vs deployed +8.9%; the "winner"
  changed every slice — the signature of noise)
- killed the delta-weighted hedging series (+5.1% vs +8.9%)
- **passed** skip-first-trigger (+11.5% vs +8.9%)

---

## 6. Build the control that could kill your own result

Every finding needs the test that would falsify it. Real examples:

| finding | control | outcome |
|---|---|---|
| "skip the first trigger" | **temporal** cutoffs (no entry before 10:00/10:30/11:00) | flat → effect is **ordinal, not hour-bias**. Survived. |
| delta-weighted flow loses 7.3pp | `lakeprem` — premium rebuilt from the *same lake rows* | already loses 4.4pp → **most of the gap was the data source, not the weighting** |
| trade #1 is terrible on multi-trigger days | — | **confounded**: a day only *has* a 2nd trigger if trade #1 exited early, i.e. failed. Largely tautological. |
| any entry-signal claim | random-entry null | calibrates at MFE/MAE 1.00, hit ~50% |

If you can't think of the control, you don't understand the claim yet.

---

## 6a. Explanations for nulls need the same standard as findings

There is an elaborate apparatus above for validating positive results —
pre-committed criteria, walk-forward, bootstrap, cross-sectional replication.
None of it was ever applied to *explanations of negative results*. But "the
compression killed it" is a causal claim about the data exactly as much as "this
rule has edge" is.

Two errors on 2026-09-10, both mine, both in the MBO thread:

**"The compression killed it."** The gate test came back 0/4 and I asserted that
microsecond-to-30-minute aggregation had destroyed a real signal. I had raised
that risk *before* running, as a caveat — and when the null matched my
pre-registered worry, it felt confirmed.

> **A pre-registered caveat is a hypothesis, not a finding.** Naming a risk in
> advance does not privilege it as the explanation. A null is consistent with
> many causes.

When finally tested, it was not merely unproven but *contradicted*: the features
predict at every horizon 1m–60m, and `imb` **increases** with horizon
(0.36 → 4.00bp). Compression would produce the opposite shape. The real answer
was duller — the features were always small.

**"The book is broken."** The NQ positive control came back flat, the
reconstructed spread was 3 ticks against a remembered "NQ is 1 tick wide >90% of
the time", and I declared the reconstruction broken. The prior was stale: it
dates from NQ at 12–15k, and at 29,500 one tick is 0.085bp so multi-tick spreads
are normal. Databento's own reference: 3.6% one-tick, median 3 ticks — matching
our book exactly.

Three compounding mistakes worth naming separately:
- I went looking for corroboration *after* deciding, and found it (5.7%
  unmatched cancels — real, but weak evidence when sought that way).
- I under-weighted contradicting evidence already in hand: zero crossed quotes,
  stable book size, and **RTY looking fine at 39% one-tick.**
- Two instruments disagreed with my expectation in opposite directions and I
  concluded the *instrument* was wrong. **When measurements disagree with an
  expectation, the expectation is a candidate for being the odd one out.**

### The rule

**When a result has several explanations, naming the discriminating test is
useful; picking a favourite is not.** "0/4 — here are three candidate causes and
here is the test that separates them" loses nothing and claims nothing unearned.

### And the part that actually costs something

Both errors propagated into **durable artifacts** — "the compression destroyed
it" was baked into `check_mbo_gate.py`'s printed verdict, and "🚨 THE BOOK
RECONSTRUCTION IS BROKEN" into the memory file. Those are later read as
established fact, by a future session or by a human months on.

> **A wrong claim in a script's output text or the memory file has a far longer
> half-life than a wrong claim in conversation. The standard for artifacts
> should be HIGHER than for speech, not lower.**

Both were caught the same way everything else here is caught: by building the
thing that could prove it wrong (§6). The positive control caught the second;
a $1.44 reference purchase settled it. Neither was caught by reasoning harder.

## 6b. This is a two-person job, and the division of labour is specific

The assistant is fast, tireless, and will happily build any test you name. It is
also confidently wrong on a predictable schedule, and — this is the part that
matters — **confidently wrong looks exactly like confidently right.** There is no
tell in the prose. The output is equally fluent either way.

So the human's job here is *not* checking arithmetic or code; that is the thing
the machine is actually good at. The job is upstream of that:

| the assistant is reliable at | the human has to supply |
|---|---|
| building the test once specified | whether it is the right question |
| finding bugs when pointed at code | skepticism about framing |
| recalling a method and applying it | **priors, and knowing when they are stale** |
| executing the battery consistently | refusing the first answer |

**Evidence from this project, not theory.** The following did not come from the
assistant being careful; they came from being pushed:

- *"What do high-trade days actually look like?"* → the pile-on → the
  sequential-fill discovery, which cut the book's OOS edge by ~2/3.
- Pasting the live paper log → reconciling it exposed the **exit cushion**, worth
  ~10pp/trade, wrong in every sim for months.
- *"For wide-spread tickers, any price between bid and ask is arbitrary"* → the
  fill-sensitivity band → three rules revealed as unmeasurable and a **real bug
  found in the entry model** (bot_runner never pays the ask).
- *"Does MBO condition the flow trigger?"* → the A5 derivation → caught two
  significant-looking false positives.
- *"Soften A3 for the 12-day sample"* → a criterion that could otherwise only
  produce false negatives.
- Order-size HHI, hidden-liquidity absorption, tick-chasing velocity — all three
  of the MBO features that showed anything were the human's, not the machine's.

And the two occasions the assistant asserted an explanation it had not tested
(§6a) were caught by controls, not by reasoning — controls that exist because
this document demands them.

**The operating rule:** treat assistant output as a well-read colleague's first
draft — fast, broad, usually right on mechanics, and occasionally confidently
wrong about which question is being answered. Accepting the first answer is the
failure mode. Nothing in this file survives contact with a human who stops
pushing.

## 6c. 🚨 A baseline can cancel your own statistic

Section 6 says build the control that could kill your result. This section is
the failure mode of doing that badly, and it cost us a headline conclusion.

The `base2m` control in `check_mbo_flow_interaction` (and, inherited,
`check_wpoc_gate`) subtracted a "no-interaction baseline" from the observed
aligned-minus-opposed gap. With `p = P(aligned)`, `M_al = E[s·r|aligned]`,
`M_op = E[s·r|opposed]`:

```
gap        = M_al + M_op
base2m     = 2(p·M_al + (1-p)·M_op)
gap - base = (1 - 2p)(M_al - M_op)
```

Our direction mix is ~50/50 **by construction**, so `p ≈ 0.5`, so `(1-2p) ≈ 0`,
so the statistic is ~0 *no matter how large the real effect is*. A trigger
earning +10bp aligned and 0bp opposed — a total interaction — gives
`gap=10, base=10, difference=0`.

Worse, the gate `gap > 1.5·base` is **unpassable in both branches** once
`base ≈ gap`: for `gap>0` it asks `gap > 1.5·gap`; for `gap<0` it asks
`gap > 0.5|gap|`. "0 of 24 cells passed A5" was a property of the arithmetic,
not of the market, and carried exactly zero information.

**The rule:** a control estimated from the same rows as the statistic can be
*algebraically entangled* with it. Before trusting any `observed − baseline`,
**do the algebra symbolically and check the difference does not collapse under
the conditions your data actually satisfies** — then confirm it numerically by
feeding it a synthetic case where the effect is known to be huge. If the
statistic does not move on that case, it cannot detect anything.

Replacements live in `check_mbo_gating.py` / `check_wpoc_gating.py`, and they
split the question A5 was conflating:

- **deployable gain** `E[d·r|aligned] − E[d·r]` — what gating would actually add
- **true interaction** `¼[E(r|+,+) − E(r|+,-) − E(r|-,+) + E(r|-,-)]` — the
  saturated 2×2 coefficient, which is what A5 was reaching for

---

## 6d. A pre-committed check needs a SECOND check that can tell its failure modes apart

§4 says write the pass criteria before running. Do that and you will eventually
watch one fail *ambiguously* — and a criterion you cannot act on has not earned
its place.

`check_flow_accel` needed the 30s tape buckets joined to trigger candidates on
`(date, ticker, minute)`. A silent misjoin is the worst outcome available here:
every downstream table still populates, and the answer is confident and
meaningless. So the criterion was written in advance:

> trigger minutes must sit far right in their own day's premium distribution;
> a median near the 50th percentile means the join is wrong

It came back at **53.6%** — 9% above the 90th, SPY at the 34.5th. Dead on the
stated failure line.

**But that criterion could not distinguish its two failure modes.** A median near
50 is produced *equally well* by a broken join and by a correct join measuring a
premise that was never true. Those demand opposite responses — debug the
pipeline, or rewrite the question — and the check that fired could not say which.

What resolved it was a second measurement taken for an unrelated purpose. The
per-minute fidelity filter compares the tape's own value against UW's published
value **at each trigger minute**, and its median ratio was `0.000` on 2,380
candidates. A join off by a minute or a ticker cannot produce a zero there. So
the join was right and the *premise* was wrong: the "whale spike" trigger is an
EMA(5) crossover of **cumulative** net premium, which can cross a lagging EMA on
a completely ordinary minute. There is frequently no whale in the trigger candle
at all. (What the candle does have is direction — 68.7% of its aggressive
premium runs one way, against ~50% for a two-sided minute. The trigger selects
persistence, not bursts.)

**The rule:** for any check whose failure would send you debugging, ask in
advance *what else* could produce that same failure — and arrange a second,
mechanically independent quantity that separates the cases. Cheapest form: a
check that compares your reconstruction against the vendor's own published
number at the exact rows you are joining on. It validates the join and the
values in one pass, and unlike a distributional sanity check it has no premise
of its own to be wrong about.

**The corollary that cost the most here:** a name can smuggle in a premise.
"Whale spike trigger" was used for weeks, including by me, and the phrase
asserts a burst that the trigger definition never required. Before measuring a
feature *of* a thing, confirm the thing has the property its name claims.

---

## 7. Known traps

- **Win rate is not the objective.** The book's edge is a right tail: ~38% win
  rate × ~+90% mean winner. The ROE give-back took win rate 0.38 → 0.59 and
  *destroyed* expectancy by capping winners at +21%. The 1:1 RR bracket "won"
  every historical sort for the same reason.
- **A parameter uniform across every rule with no explanatory comment is a RED
  FLAG** (inherited default), not evidence of validation. `rr: 1.0` was
  everywhere; it put the stop at `entry*(1-1.0)` = **zero**. Nine of eleven
  rules had no stop.
- **The trail has a dead zone.** `elif trail_stop is not None:` swallows the TP
  branch, so a 50% trail only rises above entry once peak ROE exceeds +100%.
  **Any trade peaking between 0% and +100% exits at a loss by construction.**
- **IS/OOS sign flips are noise, not puzzles.** Stop slicing when cells fall
  below ~50 observations. At n=310 a four-way cut is reading tea leaves.
- **A "how strong" split at the SAMPLE median can manufacture the gradient that
  sells the story — and it inverts when measured causally.** "Fading the bull"
  looked convex in tide strength: splitting the fade bin at the pooled median of
  cum-flow gave win 63.9% / medROE +14.8 above it against 41.3% / −10.4 below.
  That median is computed from trades the strategy never saw. Re-measured as a
  **trailing 2h quantile with the current minute excluded**, the ordering
  REVERSED — weak tide +7.4 medROE, strong tide +0.8. The gradient *was* the
  look-ahead. Any "more of X gives more of Y" claim has to be cut on a threshold
  available at the time of the trade, and the causal version is the only one
  that counts.
- 🚨 **A "search then reject" is NOT the same as "search from later" — it
  silently adds a selection criterion you never wrote down.** `check_multiplier`
  looked for the reclaim with `find_reentry(..., from_mod=drawdown_minute)` and
  then dropped any hit landing before leg one's exit. `find_reentry` returns the
  **first** qualifying minute, so a reclaim that leg one simply rode through
  consumed the search and the post-exit reclaim was never seen. The surviving
  sample was therefore not "reclaims after the stop" but "reclaims after the
  stop, **on days that did not also reclaim while the position was open**" — an
  extra filter nobody specified. It was worth **+1,158 on 34 trades, ret/DD 6.06
  vs 4.60, 4 of 5 pre-committed criteria PASS**. Searching from the exit instead,
  which is what the hypothesis actually said, gave **+193 on 53 trades and M2
  FAIL at every `--dd` from 0.10 to 0.40**. The 19 trades the bug was hiding were
  the bad ones. Whenever a filter is expressed as *reject after the fact*, check
  whether the thing being filtered is a **first match** — if it is, the rejection
  is also deciding what got searched.
- **A rationale invented for a sample is not a test of that sample.** The bug
  above describes a real hypothesis ("a drawdown that does not reclaim while you
  hold it, but does after you are stopped, is the better re-entry"). Re-running
  it deliberately would score the *same 34 trades* whose CI already spanned zero.
  Post-hoc rationale does not convert an artefact into evidence; only new days
  do.
- **`ret/DD` can be bought with diversification rather than earned.** The
  `--when any` variant improved return per unit of max drawdown 4.60 → 5.77 and
  passed all five criteria, while its per-trade expectancy was **+9.1 vs the
  book's +11.4** and it deployed **1.33× the capital for 1.28× the return**. Two
  concurrent legs are partly uncorrelated, so drawdown grows slower than return
  even when every added trade is below average. Any concurrency test must carry
  a **capital denominator** (position-minutes) next to the drawdown one, or it
  will reward leverage and call it edge.
- 🚨 **A signal can land on the right minutes and still be worth nothing.**
  `check_tv_proxy` rebuilt the trigger from OHLCV a Pine script can compute.
  Timing agreement was REAL and consistent across all nine tickers: exact-minute,
  same-direction recall of **19-28% against a 12-15% shuffle baseline (~1.8x)**,
  with the random-sign placebo sitting at ~1.1x. It reproduced roughly one
  trigger minute in four. Run through the same gates, exits and sequential
  guard, it returned **-561 against the real trigger's +3,601**. Timing is not
  information: a cumulative-VOLUME crossover knows when the tape got busy, not
  how many dollars were behind it or which way they leaned — and
  `check_flow_threshold` showed the magnitude is the load-bearing part. Never
  accept a correlation/recall statistic as the verdict on a signal; the verdict
  is P&L through the real pipeline, with a placebo in the same table.
- **Two dense event series overlap constantly for no reason.** The same study at
  ±5 minutes showed ~85% recall — against ~78% for randomly shuffled minutes.
  Any "these fire at the same time" claim needs a count-matched, within-day
  shuffle beside it, or the tolerance window is doing all the work.
- 🚨🚨 **`.iloc[-1]` AFTER A BOOLEAN FILTER IS THE LAST ROW IN FILE ORDER, NOT
  THE LATEST TIMESTAMP — AND `groupby` DOES NOT SORT.** `build_candidates` took
  the entry spot as `day[day["minute_et"] <= ts].iloc[-1]["underlying_close"]`,
  where `day` came from `{d: g for d, g in tb.groupby("date")}`. `_ticker_bars`
  returns (contract × minute) rows straight off a parquet predicate pushdown and
  is **not** time-ordered, so that spot was an arbitrary earlier price. Note the
  sibling line one above it — `bbc` — *was* sorted, which is what made the bug
  invisible for so long: the file looked like it had already thought about
  ordering. Median staleness ran +35bp (META), +34bp (NVDA), +108bp (AVGO),
  −5 to −7bp (SPY/QQQ), and it **changed the selected strike on 8.8%–81.1% of
  candidates**, typically by one, systematically toward ITM. Trade counts moved
  up to +42% (IWM 74 → 105) once fixed, because a different strike means a
  different path, exit and sequential-guard state. Whenever a reference price is
  pulled out of a frame, assert it against an independent source — here the
  underlying's own 1m close agreed to ±0.6bp once sorted.
- 🚨 **A profile that is FLAT across every horizon is a constant offset, not a
  response.** The bug above surfaced as NVDA showing "+174bp at +1m" and still
  +165bp at +30m. A real reaction decays, grows, or reverses; it does not hold a
  constant value for half an hour. Any event-study row that reads the same at
  +1m and +30m is measuring its own baseline, and the fastest check is the k=0
  cell — which should be ~0 by construction and was not.
- **A second event inside the measurement window is not rare — measure it.**
  Gate-passing triggers land a median of 8–12 minutes apart, so ~20% of 5-minute
  forward windows and ~50% of 10-minute ones contain another trigger. The
  sequential guard suppresses the second TRADE but not the second SIGNAL, so
  event studies need their own spacing filter (`check_arrival --min-gap`), not
  the guard.
- 🚨🚨 **A SELECTION RULE MUST BE RUN AGAINST BOTH DEFENSIBLE OBJECTIVES BEFORE
  IT MEANS ANYTHING.** `check_flow_walkforward` picked each rule's flow
  percentile on prior slices only, a textbook-clean walk-forward. Selecting on
  **total** P&L it beat the deployed book +6,361 vs +4,444 and passed all four
  pre-committed criteria. Selecting on **per-trade** — equally defensible, and
  the capital-neutral one this repo otherwise insists on — it scored **+3,482,
  i.e. −962 against deployed**, and failed. Same data, same protocol, same
  slices; the objective was an implementation choice made while writing the
  script. The per-rule picks diverged too (NVDA `p20/p20/p35/p35/p50` became
  `p20/p50/p80/p80/p80`). Whichever had been the default would have been
  reported with equal confidence. **A result that changes sign with the
  objective is a coin flip, not an edge** — so write both into the script from
  the start and report them together, the way `--fill` sensitivity is already
  mandatory for fill assumptions (§1a).
- **Adaptive beats static only until you check.** In the same study `FIXED p90`
  scored **identically under both objectives** (+4,879) because it performs no
  selection at all, and beat the incumbent on 39% fewer trades either way. When
  a static choice is invariant and an adaptive one is not, the adaptation is
  fitting the objective, not the market.
- **An aggregate gradient does not license a global parameter change.**
  `check_flow_threshold` showed book-total rising monotonically with the flow
  percentile in both halves. Per rule the optimum ran **both directions** —
  IWM and QQQ wanted tighter, SPY and NVDA wanted looser. "Raise everything to
  the aggregate argmax" was never what the aggregate said.
- 🚨🚨 **A SIGN TEST IS NOT A CRITERION. EVERY PRE-COMMITTED THRESHOLD NEEDS A
  MAGNITUDE FLOOR.** `check_flow_spike` pre-committed five criteria — "is the
  forward return positive", "does it beat the same-direction group", "does it
  beat the placebo", "do both halves agree" — and a **+0.34 basis point** result
  passed four of them. That is one cent on a $285 underlying, against a ~3%
  option round trip: roughly 100× too small to trade, and 0.04bp clear of its
  own placebo. The criteria were satisfiable by noise as written. Every
  threshold in a scorecard must be expressed in units that could pay for
  themselves — here a 5bp floor, added as S1b *alongside* the original verdicts
  rather than replacing them, so the goalposts stay visible. Pairs with the
  objective-sensitivity trap above: one lets the objective pick the answer, this
  one lets zero pass for an answer.
- **Stratify on the defining variable before concluding — and then count the
  cells.** The same study pooled 6× spikes with 58× ones. Splitting by magnitude
  showed the effect *reversing* in the 50×+ bucket, which is exactly where the
  originating observation sat. But four buckets × two groups × four horizons is
  **32 cells**, so a lone standout is arithmetic. Stratification is how you
  avoid burying an effect; it is not how you find one.
- 🚨🚨 **`cum_flow` MEASURES ACTIVITY IMBALANCE, NOT CONVICTION — AND IT STILL
  WORKS.** `build_flow_1m` computes `call: +(ask_vol − bid_vol)·vwap·100`,
  `put:` the negative. That embeds three claims, and all three are measurably
  false (measured 2026-09-22, 131M contracts of dte≤1 volume):
    - **Aggressor side = customer direction.** 47.5% prints at the ask, 48.7%
      at the bid, so the *net* signed quantity is only **1.2% of volume**. Mid
      and no-side (3.8%) are discarded.
    - **Call-buy = bullish.** **8.4% of volume is MULTI-LEG** and is counted as
      directional — a vertical's long leg reads as conviction buying and its
      short leg as conviction selling. That is **683% of the net signed
      quantity**: the signal is a small residual sitting on a much larger flow
      of non-directional activity.
    - **Every print opens a position.** Against *next-day* OI (the intraday
      field is an end-of-day figure repeated on every minute bar, so an
      intraday delta is 0 by construction — do not measure it that way), 1DTE
      contracts run **39% opening / 56% churn / 5% closing**, and the sign of
      `(ask−bid)` agrees with the sign of ΔOI **49.4% of the time. A coin
      flip.** "Bought at the ask" carries no information about open vs close.
  **And the level gate is still load-bearing** — dropping it to p20 costs
  −9,899 ROE, monotone in both halves, hour control exactly zero twice. Both
  facts hold at once, so the right reading is that `cum_flow` proxies *unusual,
  imbalanced options activity*, not directional conviction. That reframing
  predicts what the repo already found: mechanical tests keep failing (GEX
  conditioning, gamma walls, the hedging ratio — all would work if the dealer
  reading were literally right), and **passive** crossings beat aggressive ones
  +20.3 vs +9.7/trade, because a quiet drift to a level may reflect accumulated
  position where a violent move reflects churn and spread legs. Treat it as an
  interpretation that fits six results, not as a tested hypothesis.
- **The condition codes needed to do better are in the BRONZE full tape, and the
  30s backfill threw them away.** `upstream_condition_detail` carries `isoi`
  (intermarket sweep), `mlet`/`mlat`/`mesl`/`tlet` (multi-leg), and `tags`
  carries `{ask_side,bullish}` style labels per print. `_flow30_cache` kept only
  ca/cb/pa/pb aggregates, so any sweep-only or single-leg-only flow series needs
  a RE-BACKFILL of the 151 days, not a re-read of the cache. Live, the same
  information arrives on `/flow-alerts` as `has_sweep` / `has_multileg` /
  `has_floor` / `all_opening_trades`.
- **Premium ≠ volume ≠ delta.** They agree on sign only ~75% of the time.
- **The edge is INFORMATIONAL, not MECHANICAL.** Premium weighting (dollar
  conviction) beats delta weighting (mechanical hedging need). The hedging /
  participation ratio failed *even with lookahead*. GEX conditioning was
  IS/OOS-inconsistent. Stop building tests that assume dealer hedging is the
  mechanism.
- **A null result needs power before it means anything.** "No effect" and "no
  ability to see an effect" print identically. Always report the CI, not just
  the point estimate: wPOC's null is real because the 5m CI is
  **[-0.20, +0.05]bp on n=81,727 non-overlapping triggers**; the MBO 12-day NQ
  panel proves nothing at all despite looking similar.
- **Count your tests before you read the stars.** 40 cells × 2 statistics at 95%
  yields ~4 false positives by construction. `check_mbo_gating` returned 6.
  A lone starred cell in a grid is a thread, never a finding.
- 🚨🚨 **PRE-FLIGHT EVERY CONTROL: CAN IT ACTUALLY MOVE? A control that holds
  fixed the thing you are testing cannot detect it, and returns a confident
  null.** This has now happened FOUR times here, each time producing a clean,
  significant-looking, completely uninformative answer:
    1. **A5** (`check_mbo_flow_interaction`) — `gap − base2m = (1−2p)(M_al−M_op)`
       vanished at p≈0.5, which the data satisfied by construction. The gate was
       additionally unpassable in both branches.
    2. **`all` position policy** (`check_policy_matrix`) — taking every trigger
       means a within-day permutation reassigns the *same multiset* of outcomes,
       so `mean(real) ≡ mean(null)` and the gap is identically 0.0.
    3. **Within-day nulls** (`check_peak_profit`, `check_structures`) — resample
       the entry minute while holding the DAY fixed, then ask whether the trigger
       picks good days. Blind by construction; it produced "no edge" twice and
       the day-level test later showed positive point estimates.
    4. **Time-budget permutation** (`check_step1_redo`) — the inverse failure: a
       control that moves for the WRONG reason. Permuting outcomes across entry
       minutes breaks the minute↔budget correspondence and manufactures a gap
       out of nothing (~60–90% of that script's headline).

  **The three checks, before trusting any gap:**
    * **Symbolic** — write the statistic out and confirm it does not collapse
      under the conditions your data actually satisfies (METHODOLOGY 6c).
    * **Degeneracy** — verify the null *can* return non-zero. If the control
      shares a conserved quantity with the real arm (same multiset, same day,
      same subject), it cannot. Assert it in code: `check_policy_matrix` aborts
      if the `all` gap is not exactly 0.0, because a non-zero value would mean
      the policy is silently dropping trades.
    * **Confound direction** — name what the control holds fixed AND what it
      varies, then ask whether each biases the gap up or down. Two opposing
      biases (time budget up, first-trigger ordinal down) mean the result is not
      bounded in either direction, which is not the same as "roughly right".
  A null result is only as strong as the control's ability to have said
  otherwise. State what the control varies, not just that you used one.
- 🚨🚨 **PUT A PLACEBO *SUBJECT* IN EVERY TABLE, NOT JUST A CONTROL ARM.** The
  check above guards the control. It does not guard the MEASUREMENT. A placebo
  subject — a fake version of the thing being tested, scored through the exact
  same pipeline and required to return the chance value — catches bias in the
  pipeline itself, which a control arm cannot see.
  `check_peak_levels` ("do trade peaks stop at gamma walls / VWAP bands /
  wPOC?") was built **four** times. Three fake levels rode along in every run:
  a uniform-random price in the day's range, and open ± 0.8 ATR. Each build
  returned a clean, heavily starred table; the placebos failed in a DIFFERENT
  PLACE each time, and every failure was invisible from the real rows alone:
    1. **Geometry** — a CALL's peak is the underlying's running MAX and the call
       wall sits above spot, so the peak is mechanically nearer the wall than a
       random MINUTE. Fix: the null must be a matched EXTREME.
    2. **Measurement asymmetry** — the real trade was scored at the price of its
       option-peak MINUTE while the null was scored as a window MAXIMUM. The
       option peak IS the window max only 36% of the time, so a maximum was
       being compared against a non-maximum. Every level, and every placebo,
       read 0.21–0.37. Fix: score both sides the same way.
    3. **Selection** — the real window always ENDS at the EOD flatten, so it
       starts as late as its duration allows, and the flow trigger picks days
       with LARGER excursions. Both push the real extreme farther from any fixed
       price. Fix: match the control on time of day, on how far the level sat
       ahead, and on day size (range/ATR).
    4. **Subset re-introduction** — after all that the pooled table finally
       calibrated (placebos 0.50–0.52), but the `peak ROE ≥ +100%` cut had
       placebos at **0.34 / 0.76 / 0.76**, because conditioning on a big peak
       re-selects big moves that the controls were never matched on. Call wall
       read 0.727 and put wall 0.749 — sitting directly on top of a 0.758
       placebo.
  **The lesson that generalises:** the large-peak subset was the one worth
  looking at and the one that looked most convincing, and it was pure artifact.
  A placebo that is calibrated in the POOLED table is *not* calibrated in every
  SUBSET of it — re-read the placebos inside each cut, or do not report the cut.
  Corollary: distinguish ENDOGENOUS levels from exogenous ones. VWAP and its
  bands are computed from the same path whose extreme they are being compared
  against, so "the peak stopped at +2sd" is partly self-referential; walls,
  wPOC, prior-day and initial balance are fixed before the move and are not.
- 🚨 **POWER IS SET BY DAYS, NOT BY TRADES — so more trades buy nothing.**
  `check_ivr_termstructure --test power` scored the same conditioner on the
  deployed book (159 OOS trades) and on every screened trigger (1,147 OOS
  trades, 7.2× more). The minimum detectable tercile spread got **worse**:
  64.6pp → 71.2pp. Both samples sit on the **same 77 OOS days**, and the
  day-block bootstrap correctly refuses to count same-day trades as independent.
  Consequences, and they are structural:
    * **Any day-level conditioner on this book has an MDE of roughly 65pp per
      trade.** IVR, term structure, VIX, GEX, AMT — all of them. Effects
      smaller than that are invisible *no matter how the trades are counted*.
    * Inflating n by relaxing the sequential-fill guard is not a power
      strategy, it is a way to make a CI look narrow while it isn't.
    * The only real fixes are more calendar (wait) or a per-trade target with
      less variance than a ±120pp option return.
  Report the MDE next to every null on this book. "Flat" and "invisible" are
  different claims and this sample usually only supports the second.
- 🚨 **A LOWER-VARIANCE TARGET CANNOT RESCUE THIS BOOK — THE TAIL IS BOTH THE
  EDGE AND THE VARIANCE.** `check_target_variance.py` swept 14 candidate
  outcome variables (hit/cap/winsorise at ±5/10/25/50, plus log and signed-sqrt
  compressions) against four injected effect shapes. **None beats the raw
  return on the worst case**; best alternatives are `sqrt` 0.69× and `log`
  0.56×. The distribution says why: median trade **−24.6%**, mean +8.4%, skew
  +4.19, kurtosis +24.4; the **>+50% bucket is 379% of total P&L**, the top 1%
  of trades (n=2) is 64% of it, and capping at **any** of +5/10/25/50% turns the
  book **negative** (cap50 → −39.86 from +25.04). Truncating targets are
  therefore structurally blind to any effect that works by fattening the tail.
  **But the per-shape result is actionable and asymmetric**: if you can NAME the
  effect you are hunting, the target follows —
    * winners more FREQUENT (entry-side regime) → `wins5`, **2.00×** raw
    * AVOIDS THE DISASTERS (risk-side) → **`loss50`, 6.88×** raw
    * state is DISASTER-PRONE (what a skip rule must detect) → `loss50`, **2.49×**
    * right tail FATTER (amplification) → nothing beats `raw`
  `loss{k}` = `−1[return ≤ −k%]`, the LEFT-tail hit rate. The first sweep tested
  only right-tail hits and caps and so asked the wrong question entirely; adding
  the left tail moved the best loss-avoidance detector from `cap5` (4.45×) to
  `loss50` (**6.88×**). Since the ≤−50% bucket is 27.3% of trades and **−208.6%
  of total P&L**, this is both the biggest prize and the cheapest thing to see.
  **Spend the 77 OOS days there, measured on `loss50`.** Pre-commit the target
  with the hypothesis — choosing it afterwards is an 18-way comparison.
- 🚨 **THE TWO TAILS ARE THE SAME TRADES — so no convexity-proxy feature can
  ever be a skip rule.** `check_skip_hunt.py` pre-registered six entry-time
  features and five criteria. The `loss50` screen worked exactly as advertised:
  four of six predict the ≤−50% outcome strongly and IS/OOS-consistently
  (`difficulty` −27.9pp [−40.2,−16.2], `premium` −21.9pp, `mins_left` +15.7pp,
  `dte` −18.5pp) where raw P&L at ~65pp MDE would have seen nothing. **All six
  still reject**, for one reason: `spearman(loss50, hit50) = +0.664` across 17
  tercile cells. Cheap / short-dated / low-difficulty trades blow up *and*
  moonshot, so skipping them raises the mean while **total P&L falls**
  (`difficulty` LOW skip: OOS mean +11.5→+13.7% but total +25.04→+14.57).
  Before hunting further, check `--twins-only`: if a feature's `loss50` and
  `hit50` move together it is a convexity proxy and cannot work, whatever its
  significance. A usable feature must separate the tails, and none of the
  obvious ones do.
- **Score BOTH skip directions.** A sweep that picks the direction by `loss50`
  — refuse whatever blows up most — is backwards whenever the tails are
  inseparable, which here is always. Skipping the loss50-*best* tercile of
  `premium` beat skipping the worst on every axis. This was a live defect in
  `check_skip_hunt` until 2026-09-12.
- **"Only some tickers have daily expiries" contaminates any DTE feature.**
  Only SPY, QQQ and IWM have daily expirations (100% dte0); the rest run
  29–82%. Since `build_candidates` takes the first LISTED expiry in
  `rule["dte"]`, a dte=1 fill means "no 0DTE existed that day", so any dte cut
  is ticker identity in disguise. Within-rule, dte=1 is *better* in 4 of the 6
  rules that see both.
- 🚨 **LOSS AVOIDANCE MUST BE A SKIP, NEVER A STOP — and the distinction is
  structural, not a tuning question.** Every mechanical EXIT tested here has
  hurt: the ROE give-back took win 0.38 → 0.59 and destroyed expectancy, tighter
  stops hurt, TP+trail scored OOS −1.1% vs +4.8% pure. An exit acts on a trade
  already open and **cannot distinguish "this goes to −65%" from "this dips then
  runs to +158%"** — the path to the right tail runs through drawdown, so cutting
  the drawdown cuts the tail. A SKIP rule refuses the trade *before entry* and so
  removes a loser without ever touching a winner's path. Note that every filter
  that has ever worked on this book is an entry filter (regime gates, VIX
  overlay, the 09:35 guard, hour windows) and every exit modification has not.
  `check_target_variance.py --test skip` prices the prize: with a filter whose
  rank correlation to outcome is only **ρ=0.20**, skipping the worst 30% takes
  the book from **+11.5% → +19.6%/trade at 120% of total P&L**; ρ=0.30 gives
  +23.5% and 145%. Break-even is about **ρ=0.10** — below that a skip rule
  destroys more P&L than it saves. The oracle ceiling (skip exactly the ≤−50%
  trades) is +44.5%/trade and 270% of total. Validate any such rule against the
  **ρ=0 row**, which must come back flat (it does: +11.4/+12.1/+11.6/+11.3%).
- **Fit your units; never assume them.** The silver lake's `vega_close` is
  quoted **per vol POINT**. Assuming the other common convention (per 1.00 of
  vol) put a 100× error in `check_ivr_termstructure --test vega` and inverted
  its verdict from "vega is a rounding error" to "vega dominates". Recover the
  scale from the data — fit `dPrice − delta·dS ~ vega·dIV` on consecutive
  minutes of the **same contract** and read the slope — and fit it
  **per-contract, not pooled**: pooling divides one slope by a median vega taken
  across contracts whose vegas differ by an order of magnitude, and it returned
  0.19 where the per-contract median returned 0.80 (≈1.0 after attenuation).
  Cross-check against a greek you can sanity-bound: ATM 0DTE theta **must** be
  ~−200%/day, and it reading −0.6%/day is what exposed the error.
- 🚨 **A data lake can change scale mid-history. Sanity-check per partition.**
  `lake/silver/option-contracts-1m` carries a **corrupt `iv_close` for `dte==0`
  only**, from ~2024-10 to ~2025-12: median 0DTE ATM IV reads **0.009–0.013**
  (a 1% vol on a same-day option) against a 0.10–0.17 one-to-three-day IV and a
  comparable RV20. `vega_close`/`theta_close` are wrong in lockstep (vega ~0.85,
  theta ~−0.01 on a contract expiring that day) — the signature of a bad
  time-to-expiry in the vendor's pricer. Dates from ~2026-01 read sanely.
  The rest of the surface (dte 1–3, 4–9, 25–35, 50–70) tracks realised vol
  across the **whole** window and is fine. Gate 0DTE greeks with
  `check_ivr_termstructure.valid_0dte()`; anything using `iv30` or a `dte>=1`
  front leg is unaffected. **A global min/max filter does not catch this** —
  0.013 passes `0.01 < iv < 5.0` cleanly. Compare each partition against an
  independent quantity (RV20) instead.
- **Polars `dt.hour()` is Int8, so `hour * 60` silently overflows.** 09:30
  becomes 28, not 570, and the whole column wraps into −128..127 with no error.
  Always `.cast(pl.Int32)` before multiplying. This shipped undetected in
  `check_uoa_target.py`, where `_m` ordered the `first`/`last` aggregates: the
  cached daily open/close came from arbitrary minutes, a median 0.18–0.34%
  price error whose daily **return disagreed in sign with truth on 14–23% of
  days**. `h`/`l` (max/min) and the open-interest snapshot (OI is constant
  within a session — 15,761/15,761 contracts) were unaffected. Pandas
  `.dt.hour` is int64 and has never had this problem, which is why the pattern
  looked safe.
- **Data quality is per-ticker.** Multi-leg share of volume runs 12% (SPY) to
  **33% (GLD)**; unclassified another 6–14%. GLD is ~46% contaminated and
  `GLD amp1 CALL` trades on that flow.

---

## 8. Walk-forward geometry

- IS/OOS split: **2025-08-21**
- `SLICE_EDGES`: 2024-08-20, 2024-12-20, 2025-04-21, 2025-08-21, 2025-12-21,
  2026-04-22, 2026-08-23 (6 slices)
- A slice counts as populated at >= 3 observations
- Lake coverage: 2024-08-20 → 2026-08-21; netprem/OHLC → 2026-09-04

---

## 9. Before screening a new ticker

Check **premium depth first** — it costs nothing and rejects most candidates.
`bot_runner.py:1421` hard-rejects `entry_mid < 0.50`.

Measure the ATM mid on **DTE 0/1 sessions only** (longer-dated samples inflate
it badly). Calibration: META $4.36 / 100% clear the floor; NVDA $1.39 / 96%;
GLD $1.27 / 82%. Versus NOK $0.20 / **10%**, SOFI $0.23 / **5%**.

**A sub-$20 underlying cannot support a 0DTE ATM entry.** SOFI has 100k chain
volume and is still untradeable — liquidity is not the constraint, price level
is.
