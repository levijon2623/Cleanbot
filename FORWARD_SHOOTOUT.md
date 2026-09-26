# Forward shootout: pre-registration

**Status:** REGISTERED BUT DORMANT — written 2026-09-21, **not run**.
**Window:** forward paper sessions from 2026-09-22 onward.
**Binding rule:** the specifications in §2 are frozen. If this is ever scored,
it is scored exactly as written here, once, and the result is recorded whatever
it says. Any change to a threshold makes it a new document.

---

## 1. Why this exists

Two hypotheses survived 2026-09-20/21 without being settled:

- **Parameter edge.** `check_flow_threshold` found book total rising
  monotonically with the flow percentile in both halves, and `FIXED p90` scored
  +4,879 on 121 trades against the deployed mix's +4,444 on 197 — more money on
  39% fewer trades, and **invariant to the selection objective**, which the
  adaptive version was not (`check_flow_walkforward` reversed sign between
  `total` and `per_trade`).
- **Structural edge.** `check_crossing_quality` pre-committed that *aggressive*
  crossings (Δcum ≥ 5× the trailing median) outperform, and that failed 0 of 6.
  Its pre-specified mirror — *passive* crossings, where cum barely moved and the
  EMA converged onto it — scored +20.3/trade against baseline +9.7 on the full
  book, beat a random split of equal size at the 100th percentile, and degraded
  **more slowly** than baseline when its best sessions were removed.

Neither can be tested on the pre-sample holdout. `PRESAMPLE_PLAN.md` was
pre-registered 2026-09-12, run once, and §4 forbids re-running it with a
different cut or promoting a thread found afterwards. **That window is spent,
and it returned core5 −17.2%/trade on unseen data with a CI excluding zero.**
So the only admissible data is forward data.

## 2. The specifications — frozen

Core five (`sim_core.research_rules(include_paper=False)`), scored through
`sim_core` exactly as deployed: sequential fills, per-rule `policy_for` exits,
`fill="botcap"`, $0.50 entry floor. No parameter is refit.

| model | flow gate | crossing filter |
|---|---|---|
| **Baseline** | deployed `min_flow_pct` per rule | none |
| **A — parameter** | fixed p90, all rules | none |
| **B — structural** | deployed `min_flow_pct` | passive only (`Δcum / trailing-30m median |Δcum| < 1.0`) |
| **C — intersection** | fixed p90 | passive only |
| **Mirror** | deployed `min_flow_pct` | **aggressive only** (`≥ 5.0×`) |
| **Placebo** | — | random subset matched to each model's trade count, 100 draws |

`Δcum` is the one-minute change in cumulative net premium at the crossover
minute; the median is trailing 30 minutes with the **current minute excluded**.

**Reported for every model:** total ROE, n, ROE per trade, win rate, IS/OOS
split, and the placebo's p50/p95 at that n.

**The Mirror and the Placebo are not decoration.** If passive and aggressive
both sit on baseline, crossing quality carries nothing. If a model fails to beat
a random subset of its own size, "it trades less" is the whole result. Both
checks correctly called `check_vwap` a null.

## 3. What this buys — stated honestly, before the fact

Measured on the deployed window (core five, 503 sessions):

| model | trades | per session | mean ROE | SD |
|---|---|---|---|---|
| Baseline | 234 | 0.47 | +17.3 | 109.0 |
| Aggressive ≥5× | 130 | 0.26 | +17.8 | 111.0 |
| Passive <1× | 150 | 0.30 | **+30.9** | 114.3 |

Observed effect, passive − baseline: **+13.6 ROE/trade**, against a pooled
**SD of 111.7**. At 80% power and 95% two-sided that needs

> **≈ 1,056 trades per arm → ≈ 3,540 sessions → ≈ 14 years.**

Minimum detectable effect on the passive arm, by window:

| sessions | ≈ trades | MDE (ROE/trade) |
|---|---|---|
| 63 (a quarter) | 19 | **+102** |
| 126 (half year) | 38 | +72 |
| 252 (one year) | 75 | **+51** |
| 504 (two years) | 150 | +36 |

**A full year of forward data resolves +51/trade. The effect is +13.6.** Two
years resolves +36, still 2.6× the effect. This is the same arithmetic that made
`PRESAMPLE_PLAN` decline to spend the holdout on feature discovery, for the same
reason: the per-trade dispersion of this book (SD ≈ 111 against a mean of ≈ 17)
swamps everything smaller than a regime-sized effect.

**Model C, the size diagnostic.** Passive keeps 64% of baseline trades; a p90
level gate keeps roughly the top decile of the level distribution; the two are
close to independent. So C lands near **6.4% of crossings ≈ 7.5 trades per year**
on the core five. C cannot be evaluated and could not be deployed if it were —
one bad quarter is the entire sample.

## 4. Verdict, recorded before any forward data is seen

**The shootout as specified is underpowered by roughly an order of magnitude and
is therefore NOT SCHEDULED.** Registering it dormant is deliberate: it fixes the
specifications now, so that if the conditions in §5 are ever met the test can be
run without the thresholds having drifted in the meantime, and it prevents the
alternative failure — scoring a quarter of forward data, seeing +40/trade inside
an MDE of ±102, and calling it confirmation.

**Do not read this window early.** Section 3 says what an early read is worth.

## 5. What would make it live

Any one of these changes the arithmetic enough to reconsider:

1. **Trade rate rises materially** — more tickers in the book, or a looser
   structure that fills more often. Power scales with n, not with calendar time.
2. **Per-trade dispersion falls** — a materially different exit would do it; SD
   111 is dominated by the 50% trail's tail.
3. **A lower-variance statistic is pre-registered instead** — win rate is
   binomial and better behaved (baseline 37.0% vs passive 43.8%), though a 7pp
   proportion difference still needs several hundred per arm.
4. **The prior question is settled.** Given pre-sample −17.2%, in-sample −2.4%
   and out-of-sample +30.5%, "which filter is better" is subordinate to "does
   this book work at all". A filter comparison inside a book that does not
   generalise is a comparison of two negative numbers.

## 6. What will NOT be done

- No scoring of a partial window and reporting it as a result.
- No adjusting the 1.0× passive threshold or the p90 level after seeing data.
- No re-testing on `PRESAMPLE_PLAN`'s window, which is spent.
- No promoting the passive-crossing thread to a deployed gate on the strength of
  the in-sample result alone.

## 7. Sign-off

Unsigned. Registered dormant 2026-09-21 pending §5.
