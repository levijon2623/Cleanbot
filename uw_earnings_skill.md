---
name: uw-earnings-vol-scan
description: >
  Scan upcoming earnings for IV-crush opportunities and walk newer options traders
  through executing long call calendar spreads to capture the earnings volatility risk
  premium using the Unusual Whales API. Use whenever the user wants to scan the earnings
  calendar, analyze a single ticker's pre-earnings setup, interpret
  Recommended / Consider / Avoid verdicts, or turn a recommendation into an actual call
  calendar spread trade. Especially useful for vibe coders and newer options traders
  who need plain-language guidance on the strategy, setup, output, sizing, and risk.
---

# uw-earnings-vol-scan: Unusual Whales Earnings Volatility Scanner

This skill turns any AI assistant into a working pre-earnings scanner for **long call calendar spreads** built around the IV-crush risk premium.
The assistant writes one self-contained Python script (included in full below), verifies it with the script's own selftest, runs it against the Unusual Whales API, and walks the user from "what can I still trade before the close tonight?" all the way to "here are the strikes, expirations, and size for your broker."
No repo clone, no virtualenv, no setup ritual: the user pastes a URL into their AI tool, and the assistant handles the script end to end.
The one thing the user supplies is their UW API key, and the recommended way is a local `.env` file (never committed) that the script reads itself; pasting the key into chat still works as a fallback.

> **Disclaimer: read before using.**
> This skill is provided for **educational and informational purposes only**.
> It is **not investment advice**, not a recommendation to buy or sell any security, and not personalized to your financial situation.
> Trading options involves substantial risk, including the loss of your entire investment, and is not suitable for everyone.
> Past performance, including the backtest figures cited below, does not guarantee future results.
> You are solely responsible for your trading decisions; consult a licensed financial professional before acting on anything this skill produces.
> Unusual Whales accepts no liability for losses incurred from its use.

## When to use this skill

Trigger when the user:

- Wants to know what is tradeable before tonight's close, or wants a heads-up on the next N days.
- Asks about a specific ticker's earnings setup ("does NVDA look good this week?").
- Asks what the scanner output means (verdicts, IV/RV, slope, expected move).
- Asks "what do I do now?" after seeing a Recommended ticker.
- Has questions about the underlying calendar-spread / IV-crush strategy.

## Strategy in 60 seconds

Leading up to an earnings announcement, the nearest option expiry after the announcement features structurally inflated IV.
Why? -> Market participants hate uncertainty and are willing to overpay for options to to eliminate that uncertainty; option sellers assume this risk and are paid a premium (on average) for providing this service to the market.

How do we capture this effect? -> With a **long call calendar spread**.
Using roughly ATM strikes, **sell** the near-dated call (front leg), **buy** a farther-dated call (back leg).
You pay a net debit, which is your max theoretical loss before commissions and fees.
This max loss is a feature compared to other structures that capture this effect like short straddles and short strangles, which have theoretically unbounded losses.

The long call calendar spread profits when the underlying price stays near the strike _and_ the front leg IV collapses drastically compared to the back leg IV.

This scanner narrows the earnings universe to names where three conditions hold simultaneously:

1. **Liquidity**: 30-day average share volume ≥ 1,500,000 (proxy for tight option spreads).
2. **IV/RV richness**: IV30 / RV30 ≥ 1.25 (implied vol sits ≥ 25% above trailing realized vol).
3. **Term-structure backwardation**: IV slope from the nearest expiration out to 45 DTE ≤ −0.00406 per day (front IV materially above back IV).

The **slope filter is the gating filter**.
Without backwardation, the calendar mechanics don't work and the verdict is always Avoid regardless of the other two.

### Why IV and RV are compared with no lag (do not "fix" this)

The IV/RV filter compares **today's** IV30 against RV30 measured over the **trailing** 30 trading days.
Those two windows point in opposite directions: IV30 looks forward from today, RV30 looks backward from today.
The mismatch is deliberate, 1.25 is calibrated against exactly this construction, and it must not be "corrected".

Any careful reader eventually notices the misalignment and reaches for the obvious fix: lag the IV so it is quoted 30 days ago, then compare it against the RV that actually realized over the following 30 days.
That construction is genuinely correct for a _different_ question.
Here it is wrong, for three reasons in descending order of force:

1. **The lag would remove the earnings event from the IV window, destroying the signal.**
   The scan runs on the trade-by date, roughly one day before the announcement.
   Today's IV30 spans today to today+30 and therefore _contains_ the earnings event, which is the entire reason it is elevated.
   IV lagged back 30 days spans today−30 to today, which contains no earnings event one day in the future.
   The lagged ratio would measure baseline structural volatility risk premium instead of earnings volatility richness, and would essentially never trip 1.25.
   The filter would stop finding anything, and it would look like the market had simply gone quiet.
2. **The correctly aligned RV is unobservable at trade time.**
   The RV that truly matches today's IV30 covers today through today+30, which has not happened yet.
   At the moment the order must be placed, the only two implementable choices are today's IV against trailing RV, or month-stale IV that cannot be traded against.
   There is no third option before the close.
3. **The threshold is calibrated to this construction.**
   1.25 has no independent theoretical justification.
   It exists because it produced the 10.80% pass rate and +7.28% mean return in the backtest, measured this way.
   Change the alignment and the published figures no longer describe what the scanner does.

The distinction that governs this: **measuring realized volatility risk premium** (take IV at time t, wait, measure RV over [t, t+30]) genuinely requires the alignment, and that is what a backtest does after the fact.
**Pricing vol at trade time** ("is the premium I am about to collect rich relative to how this stock actually moves?") is inherently contemporaneous.
This scanner does the second.
If you are wiring these numbers into a tool that offers a lag-aligned or "forecast-aligned" mode, that mode is correct for measuring VRP and wrong for this filter; pin the contemporaneous mode explicitly and comment why.

**To any future agent or contributor:** this is the single most likely thing for a capable reader to "helpfully" repair.
It has been raised, investigated, and rejected on the evidence.
Do not add a lag without new evidence, and if you do, re-derive the threshold in the same change.

## Realistic expectations

Always set expectations honestly before presenting results; this strategy has positive expectancy but high variance, and a user who places three trades a quarter is sampling noise, not edge.

**Source:** [Volatility Vibes original strategy and backtest](https://www.youtube.com/watch?v=oW6MHjzxHpU), covering US equity earnings events **2007 through 2024**.
The figures below are from that backtest, applied to the filtered earnings universe this scanner produces.

| Metric                | Decimal   | Rounded                                                                                                                                                                                                     |
| --------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Universe pass rate    | 0.1080    | **10.80%** of all earnings events                                                                                                                                                                           |
| Count of trades       | 7,313     | -                                                                                                                                                                                                           |
| Mean per-trade return | +0.072829 | **+7.28%** of debit paid                                                                                                                                                                                    |
| Std dev               | 0.286135  | **28.6%**                                                                                                                                                                                                   |
| Min                   | −1.071140 | **−107.1%** tail event: a near-total loss on a call calendar (rare; requires a very large gap that pushes both legs near zero before you can exit), with the extra ~7% from commissions, fees, and slippage |
| P25                   | −0.062751 | **−6.28%**                                                                                                                                                                                                  |
| P50 (median)          | +0.111473 | **+11.15%**                                                                                                                                                                                                 |
| P75                   | +0.264228 | **+26.42%**                                                                                                                                                                                                 |
| Max                   | +0.788395 | **+78.84%**                                                                                                                                                                                                 |

Talking points to surface for the user:

- **Roughly a quarter to a half of trades lose money** even after passing all three filters.
  This is a portfolio strategy, not stock-picking.
- Std dev of ~29% means single-trade outcomes swing wide.
  **Sizing matters more than selection.**
- Edge only shows up over dozens of trades.
  Don't judge the system on a sample of 3.

### Internalize the variance with a Monte Carlo simulation

The user got here by pasting a URL into an AI assistant, which means they _have_ an AI assistant.
Use that.
Before the user puts on a single trade, tell them to ask you (or any AI tool) to run a 50,000–100,000-path Monte Carlo of this strategy using the backtest's per-trade distribution and report:

- Fraction of 1-month windows that end down money.
- Fraction of 3-month windows that end down money.
- Worst 1-month and worst 3-month drawdowns observed across the simulation.
- 5th-percentile, median, and 95th-percentile cumulative returns over 1 and 3 months.
- 5th-percentile, median, and 95th-percentile cumulative returns over 5 years

Copy-pasteable prompt:

> "Run a 50,000-path Monte Carlo of a trading strategy with per-trade mean return +7% and per-trade std dev 29%.
> The original strategy takes roughly 430 trades per year, so assume I ease into this strategy at half effort and execute 18 trades per month (adjust if I tell you otherwise).
> Report: (1) the fraction of 1-month and 3-month windows that end down money, (2) the 5th-percentile / median / 95th-percentile cumulative return over 1 and 3 months, (3) the worst 1-month and worst 3-month drawdown in the simulation, and (4) the 5th-percentile / median / 95th percentile cumulative return over 5 years."

The point is _internalization_, not analysis.
A user who has seen with their own eyes that a positive-EV strategy still produces losing months at meaningful frequency is much less likely to abandon the system after a normal drawdown.

> **You are getting paid to take risk that other investors do not want to take.
> That bargain is the source of the edge, and it means you WILL take losses.
> Embrace that, or don't run this strategy.**

## How to run the scan

**Precondition: this skill needs BOTH code execution and outbound network.**
Running the scan requires a real code-execution tool (Python 3.9 or newer) **and** the ability to reach `api.unusualwhales.com`.
Environments split cleanly on the second requirement:

- **Both available, so the scan runs right here:** Claude Code, the ChatGPT desktop app (now the Codex app), Cursor, and most local or IDE-based coding agents.
- **Code runs but the network is blocked, so the scan cannot run here:** the web analysis sandboxes - Claude Desktop, Claude.ai's Analysis tool, and ChatGPT's Code Interpreter on chatgpt.com.
  They execute Python fine, but a proxy blocks outbound requests to the API host.
  Verified 2026-07-16: it surfaces as a network/egress denial, not an auth error, so do not read it as a bad key.

If you cannot run the scan here - no code-execution tool at all, or a sandbox that blocks the network - **do not hand-estimate anything and do not blame the user's API key.**
Tell the user plainly why, then help them run it on their own machine: save the fenced script below to `uw_earnings_vol_scan.py`, run `python uw_earnings_vol_scan.py --selftest`, then `python uw_earnings_vol_scan.py --scan` with `UW_API_KEY` set (or a `.env` file present), in a local terminal.
The code and the results are identical; only the network path changes.

**Do not hand-estimate values, fabricate API responses, or eyeball the math**: the numerical outputs of this skill drive real trades and have no value if approximated.

**The script below is the specification.**
Everything the scan computes (the three filters, Yang-Zhang, the term structure, the trade-by calendar, every API call, every failure mode) lives in it.
Your job is to run it and interpret the results for the user, not to reimplement it.

1. **Write the script.**
   Copy the entire fenced Python block in **The scanner script** below to a file named `uw_earnings_vol_scan.py`, **verbatim**.
   Do not edit it, do not "improve" it, do not summarize it, do not translate it to another language, and do not skip the selftest section at the bottom.
   Do not change the constants; they are calibrated (see **Reference: filters and verdicts**).
2. **Run the selftest before anything else:** `python uw_earnings_vol_scan.py --selftest`.
   It makes **zero API calls**, needs no API key, and takes about a second.
   It checks Yang-Zhang against a known fixture, the term-structure interpolation, the NYSE holiday table, Eastern-time DST handling, the trade-by calendar, tonight's buckets, all eight rows of the verdict matrix, the threshold directions, the `.env` parser, the request headers, and every fatal-error path.
   It currently reports **124 checks**; treat the exact count as informational and the pass/fail as binding.
   **If it does not report every check passing, stop.**
   Re-copy the block and run it again.
   Never scan with a file that fails its own selftest: that is the whole reason the selftest exists, since a silently mistyped constant produces plausible numbers rather than an error.
3. **Get the API key.**
   Supply it in this order of preference, most secure first:

   1. Use `UW_API_KEY` from the environment if it is already present.
   2. Otherwise prefer a `.env` file in the working directory containing `UW_API_KEY=your-key`; the script reads it automatically.
      When you create it, also ensure `.gitignore` contains a `.env` line (create `.gitignore` if it is absent), and warn the user that this file holds their key and must never be committed.
      This keeps the key out of the chat transcript and shell history.
   3. Only as a last resort, pass `--api-key` (which puts the key in plaintext in the command, the chat, and shell history).

   Never echo the key back in plaintext.
   Note the distinction: the list above is the _recommendation_ order by safety, but the script's _resolution precedence_ is the conventional `--api-key` > `UW_API_KEY` env > `.env`, so an explicit flag always wins.

4. **Run the scan.**
   - **Tonight's action list, which is the default and is what you should almost always run:** `python uw_earnings_vol_scan.py --scan`.
     It returns only the names reporting after today's close or before the next session's open, so **every row it returns can still be traded before today's close** and `trade_by` is today for all of them.
     It costs 2 requests per name plus 2 per calendar day covered: measured on 2026-07-15, that was 17 names in **36 requests and about a second**.
   - **A forward window, for research or a heads-up only:** `python uw_earnings_vol_scan.py --scan --days N`.
     **This is not the default and its rows are not a to-do list.**
     The options market reprices continuously, so a name that passes today can fail by its own `trade_by` date, and a window that starts today also surfaces names whose deadline has already passed.
     It costs 2 requests per day plus 2 per name, and names accumulate fast: a 7-day window scored 113 names on 2026-07-15, which is a couple hundred requests and about a minute.
     Do not run it idly, and do not present its output as tonight's trades.
   - One name: `python uw_earnings_vol_scan.py --ticker NVDA`.
     Here `--days` means something different: it only sets how far ahead to look for that name's report date, and it does default to 7.
   - Add `--rich` for a colored terminal table, but only when the user wants something human-readable or screenshot-able.
     The default output is machine-readable and is what you should parse.
5. **Present the results** per **Presenting results** below.
6. **If the user picks a Recommended ticker**, walk them through **Executing a Recommended trade**.

If Python is genuinely unavailable but another runtime is, port the script faithfully rather than improvising from the prose in this file: the code is unambiguous and this file no longer restates the math.
Port the selftest too, and run it.

### If the script reports an error

It fails loudly and specifically on purpose.
Read what it prints and relay it; do not guess.

Every fatal error prints as two sibling keys: `error:` says what happened, and `help:` says what to do about it.
**Relay the script's own `help:` text rather than substituting remedies from this file.**
Several of these remedies are computed at runtime (the Cloudflare one depends on which HTTP client the environment actually has), so the script knows things this prose cannot.

- **`error: no API key`** -> the user needs to supply their UW key.
  Relay the script's own `help:`, which lists the three ways in order of preference (environment `UW_API_KEY`, a local gitignored `.env`, then `--api-key` as a plaintext last resort); prefer setting up a `.env` over pasting the key into chat.
- **`401`** -> bad or missing key.
  Tell the user to check `UW_API_KEY`.
  Do not retry.
- **`403` mentioning Cloudflare or Error 1010** -> **this is not an auth problem.**
  The request was blocked upstream before it ever reached UW's auth layer.
  **Relay the script's `help:` line verbatim: the remedy depends on which HTTP client is in play and the script already worked that out.**
  On the stdlib fallback it will suggest installing `requests`; on `requests` or `httpx` it will point at the network (a VPN, proxy, or datacenter IP), because those clients already send an ordinary signature and installing them again is not a fix.
  **Do not tell the user their key or subscription is bad**, and do not remove the `UW-CLIENT-API-ID` header; it is not the cause.
  This misdiagnosis is the single most common failure with this API.
- **`404`** -> the route does not exist, which means the script was edited or mis-transcribed.
  It is **not** an "empty data" signal.
  Re-copy the script rather than inventing endpoint variants.
- **A network / connection failure, or your environment reports the request was blocked** (`host_not_allowed`, a proxy denial, a refused connection, a DNS error, or a traceback that never reached UW) -> **this is the sandbox, not the key.**
  Some web analysis sandboxes run Python but block outbound network to the API host, and the block can even arrive looking like a generic `403`.
  Do not retry and do not tell the user their key or subscription is bad; help them run the script on their own machine instead (see the precondition at the top of **How to run the scan**).
- **Daily quota exhausted** -> not recoverable in-session.
  The quota resets at **20:00 US/Eastern**, year-round.
  That is the end of the post-market session, not a UTC boundary; the two coincide only under EDT, so do not "correct" this to midnight UTC.
  Say plainly that the reset lands after the close, so a quota exhausted during the session is gone for today's trading; don't fake progress.
  They can wait or upgrade their UW tier.
- **A year outside the market-holiday table** -> the script carries a hand-verified NYSE closure table that currently ends after **2028**, and past its horizon it stops rather than guessing.
  This is deliberate: no formula predicts an ad-hoc closure like a presidential funeral, and a guessed holiday would move a `trade_by` onto a closed day and silently drop the name from tonight's list.
  The fix is to extend the table from the published NYSE calendar, not to work around the error.
- **Skipped tickers** -> normal.
  The script fails closed on any malformed or insufficient data, skips that one name, keeps the batch going, and reports the count and the specific reason for each.
  Surface those reasons rather than hiding the gap between the earnings-calendar count and the table size.

## Reference: filters and verdicts

**The script owns all of this.**
It is reproduced here so you can explain a verdict to a user without making them read code, and so a reviewer can see the numbers.
It is **not** a spec to implement against.
If this section and the script ever disagree, **the script is correct and this section is the bug.**

**Thresholds (do not change).**
They come from the backtest and are calibrated as a set, not individually:

- `AVG_VOLUME_THRESHOLD   = 1_500_000`
- `IV_RV_RATIO_THRESHOLD  = 1.25`
- `TS_SLOPE_THRESHOLD     = -0.00406`
- `RV_WINDOW_TRADING_DAYS = 30`

**What the two "30"s mean.**
They are different units, deliberately:

- `iv30` is ATM implied vol interpolated to **30 calendar days** (about 21 trading days), read off the term-structure curve.
- `rv30` is Yang-Zhang realized vol over the trailing **30 trading days** (about 44 calendar days), annualized by `sqrt(251)`.

**Why `sqrt(251)` and not the textbook `sqrt(252)`.**
Juneteenth became a full NYSE closure in 2022, so the trading year genuinely got a day shorter: counting weekdays minus the NYSE closure table gives exactly 251 for 2026, 2027 and 2028.
This is a deliberate departure, not a typo, and it is measured rather than argued: `rv30` scales by `sqrt(251/252)`, so `IV/RV` rises **0.199%** and the filter loosens very slightly, and on the 2026-07-15 live scan **0 names changed verdict**.
The threshold stays at 1.25 rather than chase a 0.2% effect with false precision.
Be aware that the backtest figures above were produced at 252, over a 2007-2024 period that mostly predates Juneteenth; 252 was right for the backtest and 251 is right for today.

So `IV30 / RV30` compares two annualized rates measured over horizons that genuinely do not match.
This is known, not an oversight.
Both quantities are annualized _rates_ rather than accumulated variances, so the window length does not appear in the units, and 1.25 was calibrated against this exact pairing.
Tested across 113 earnings names on 2026-07-15: moving the RV window to 21 trading days flips **5.3%** of verdicts and **loosens** the filter (26.5% -> 30.1% pass on this filter alone).
It is a legitimate experiment, but the window and the threshold have to move together (roughly 1.27 for a 21-day window), and that new pair has zero P&L validation behind it.
The reasoning is preserved in the constants block in the script.

**Per-filter pass condition:**

- `volume_pass = avg_volume >= 1_500_000`
- `iv_rv_pass  = iv_rv_ratio >= 1.25`
- `slope_pass  = slope <= -0.00406`

**Verdict matrix:**

| Slope | Volume | IV/RV | Verdict         |
| ----- | ------ | ----- | --------------- |
| PASS  | PASS   | PASS  | **Recommended** |
| PASS  | PASS   | FAIL  | Consider        |
| PASS  | FAIL   | PASS  | Consider        |
| PASS  | FAIL   | FAIL  | Avoid           |
| FAIL  | any    | any   | Avoid           |

The slope filter is the gating filter.
If slope fails, the verdict is always Avoid regardless of the other two.

> **"Recommended" is a label, not a recommendation.**
> It means the ticker passed all three statistical filters, nothing more.
> It is **not** advice to enter the trade.
> The user still has to verify the option chain has tradeable strikes and tight spreads, confirm no news/dividend/M&A complications, size the position responsibly, and judge it against their own risk tolerance and account constraints.
> Surface this framing to the user whenever you present results; do not let "Recommended" be read as an endorsement.

## Trade-by dates

The script computes a `trade_by` date for every result: the last session in which the position can still be opened before the print.
PM reports trade by that day's close; AM reports trade by the previous session's close, rolling backwards over weekends and US market holidays.
15 minutes before the close is the textbook entry, but see **Common pitfalls**: getting the trade on matters more than precision timing.

**Missed deadlines.**
If the trade-by date for a Recommended ticker has already passed (or is today and the US market is already closed), **do not enter**.
The IV crush is already pricing in, or the earnings event has already printed; the strategy's edge is gone for that name.
Tell the user explicitly that the deadline was missed rather than silently presenting a stale recommendation.

**Where lapsed deadlines actually show up: only under `--days N`.**
The default tonight-only scan cannot produce one, because every name it returns has a `trade_by` of today by construction.
That is why its output states `trade_by` once in a header line instead of repeating a column that never varies.
A `--days N` window, by contrast, starts today, so a name that reported this morning legitimately carries a `trade_by` of yesterday.
Every row there has its own `trade_by` column, and any value earlier than the `window` start date has lapsed.
The live 2-day scan on 2026-07-15 returned ELV and TRX as Recommended with a `trade_by` of 2026-07-14, both already gone.
**That is the trap in window mode: a lapsed name still reads as "Recommended".**

## Presenting results

The script's default output is machine-readable, already sorted (trade-by ascending, then verdict, then ticker), with counts, pass/fail flags, and skip reasons already computed.
**Do not recompute, re-sort, or re-derive any of it.**
Render it for the user in whatever format suits the environment:

- **Web UI / markdown-capable chat (Claude.ai, ChatGPT):** a markdown table.
- **Terminal / CLI:** re-run with `--rich` for a colored table, or render the default output as a plain ASCII table.
- **Fallback:** plain text, one ticker per line.

**The two scan modes emit different shapes, and the difference is load-bearing rather than cosmetic.**
Read the header to see which one you have; do not assume.

- **Tonight (`--scan`)** opens with `session:`, then a single `trade_by:` line that applies to every row, then `covers:` naming the two report slots it swept.
  Its rows carry **no** `trade_by` column, because the value is identical for every name.
- **Window (`--scan --days N`)** opens with `window:` instead, and every row carries its **own** `trade_by` column, which is what makes lapsed deadlines visible.

Both then share `as_of`, `bars_through`, `rv_window_trading_days`, `counts`, `skipped`, a `results[N]{...}` table, and a `help[N]` block.

**Batch-scan columns:** ticker, earnings (`YYYY-MM-DD AM`/`PM`), trade-by (**window mode only**; in tonight mode it is the header line, so do not invent a column for it), volume (`4.2M` + PASS/FAIL), IV/RV (2 decimals + PASS/FAIL), slope (5 decimals + PASS/FAIL), expected move (informational only, `-` when missing), verdict.
Every one of these is a field in the script's output; the pass/fail flags are emitted, so read them rather than comparing against thresholds yourself.

Whatever the format, carry these through:

- The **counts** line, e.g. `3 Recommended | 5 Consider | 12 Avoid   (20 scanned)`.
- The **skipped count and its per-ticker reasons**, e.g. `7 tickers skipped`.
  Never hide the gap between the earnings-calendar count and the table size, and never say just "insufficient data" when the script has told you exactly which check failed and why.
- `bars_through` whenever the user asks how current the numbers are.
  It is the last completed session the volatility is measured through; today's partial bar is deliberately excluded.
- The **"Recommended" is a label** framing above.

**Single-ticker analysis** (`--ticker XYZ`) returns the same fields as a labeled list rather than a table row, plus the raw `iv30` and `rv30` to 4 decimals so the user can see the inputs behind the ratio.

`--rich` exists because subscribers will want to screenshot the table.
A screenshot strips every word of surrounding context, including the disclaimer at the top of this file, so the rich view carries its own one-line disclaimer footer.
Do not remove it.

## Executing a Recommended trade

**Account requirements (verify first).**
A long call calendar is a multi-leg debit spread.
At most US brokers this requires:

- **Options approval Level 2 or higher** (sometimes called "Spreads" approval).
  Cash accounts and Level 1 approvals typically cannot trade spreads; the order ticket will reject.
  If the user gets a cryptic rejection, this is almost always why.
- **A margin account** (the spread itself uses no margin since it's a debit, but most brokers gate spread trading behind a margin account anyway).
- The broker must support **multi-leg / combo orders** as a single ticket.
  If theirs doesn't, do not leg in manually around earnings; find a different broker for this strategy.

Once those are confirmed, walk the user through these decisions:

1. **Trade-by deadline.**
   Enter before the close of the trade-by date (see the trade-by section).
   **Getting the trade on matters more than entering precisely 15 minutes before close.**
   Don't pass on a Recommended ticker because the user can't be at their terminal at 3:45pm ET.
   Precision in _structure_ (right strikes, right expirations, ATM, combo order) is far more important than precision in _timing_.
2. **Strike.** reasonably ATM; a liquid strike near the current stock price.
   **Tie-breaker:** if two strikes are exactly equidistant (e.g., stock at $100.00 with $97.50 and $102.50 strikes), pick the **higher** strike.
   A slightly OTM call calendar has slightly cleaner short-leg theta and a cleaner ITM-assignment outcome if the stock pins.
3. **Expirations.**
   Both legs must extend _past_ earnings.
   - **Short leg (sell):** the _first_ listed expiration that occurs after the earnings event (usually a weekly when one exists).
   - **Long leg (buy):** **pick the liquid contract.**
     Aim for the expiration nearest ~45 DTE, but if a weekly landing near 30–45 DTE has thin open interest or wide bid/ask, **prefer the next monthly OPEX out**: the slightly longer-dated leg with tight markets beats fighting for fills on a thin weekly.
     Liquidity is the rule; ~45 DTE is the target.
     (Cross-reference: see "weekly expirations" in Common pitfalls.)
4. **Order type.**
   Enter as a **single combo (spread) order**, limit at or below the mid-price.
   Never leg in separately around earnings.
5. **Position sizing.**
   Max loss = `debit paid × contracts × 100`.
   Treat that as the at-risk amount.
   - The Volatility Vibes reference video recommends **10% Kelly sizing, ≈ 6% of bankroll per trade**.
   - **Practitioner refinement: start much smaller (closer to 1–2% of bankroll per trade**) while you build experience with the strategy's variance and confidence in your execution mechanics.
     Scale up gradually as you accumulate trades and see how the variance actually plays out in your account.
   - **Cap total concurrent exposure, not just per-trade size.**
     A single tonight-only scan routinely produces multiple Recommended names, and **they all enter the same afternoon**: the 2026-07-15 scan returned 7 Recommended in one session.
     This is sharper than it looks, because tonight mode compresses the clustering into a few hours rather than spreading it across a week.
     Sizing each one independently at 2% can quietly stack to 15–20% of bankroll at risk simultaneously, with all positions exposed to the same macro window.
     As a starting rule, **cap the sum of open-trade max losses at ~5–10% of bankroll**, and lower the per-trade size when the calendar fills up.
     Concurrent positions are correlated by macro events (rate decisions, geopolitics, sector shocks) even when the underlyings differ.
6. **Exit.**
   Close the entire spread the morning after earnings, ideally within the first 5-30 minutes but NOT immediately at open since contract spreads will be brutally wide.
   The edge lives in the IV-crush window.
   **Holding past that window means you are no longer expressing the "earnings volatility is overpriced" phenomenon underpinning this research.**
   Close.
   - **Exit patience on large moves:** when the underlying jumps significantly on the print, exiting the calendar takes time.
     **Use limit orders and be patient.**
     A market exit at the open after a big move will eat far more slippage than the delta you'd recover by waiting 5–30 minutes for spreads to tighten.
7. **Slippage sanity check.**
   Before sending the order, look at the bid/ask on both legs.
   If either spread is > 10% of its mid, friction can eat the edge; consider passing even on a Recommended name.
8. **Paper-trade the first few.**
   Before risking real capital, **paper-trade 3–5 entries end-to-end** at your real broker (entry combo, exit combo, ticket cancellation, partial fills).
   Paper trading, rather than just visualizing the trade, will catch errors (wrong expiration, wrong leg side, market instead of limit order, sell-to-open instead of buy-to-open) before you make a real money mistake.

## Common pitfalls

- **Cheap tickers with poor strike availability and spreads.**
  Even a Recommended verdict requires scrutiny on low-priced names.
  - **Avoid tickers priced under $5.**
    Strike spacing usually rules out a true ATM combo and the available strikes carry wide bid/ask.
  - **Be suspicious of tickers under $10.**
    Strike availability is still challenging; verify a clean ATM combo exists at the listed strikes before committing.
- **No weekly expirations.**
  If a ticker has no weeklies, the first expiration after earnings can be far out and contract liquidity tends to be poor.
  **Treat the absence of weeklies as a red flag**; verify both legs have meaningful open interest before trading.
- **Strike-level illiquidity.**
  Even on high-volume stocks, a specific ATM strike may have low OI.
  If OI < ~500 contracts at the ATM strike, expect bad fills.
- **Earnings date drift.**
  Companies sometimes move their report date.
  Re-check before entering; the scanner uses what UW publishes when the scan runs.
- **Special dividends / splits / M&A news.**
  These distort calendar spreads.
  Skim the news before trading.
- **Mismatched expiration cycle.**
  If the only post-earnings expirations are far out or there's no clean back leg near ~45 DTE, the structure may not match what was scored.
  Move on or downgrade.
- **Short-leg early assignment risk.**
  After a large post-earnings move, the short call can land deep ITM.
  If an **ex-dividend date falls between earnings and the front-leg expiry**, early assignment is possible, which means you might wake up short 100 shares per contract and owe the dividend.
  Before entering, check the ex-div calendar for the back-leg window.
  If a dividend is sandwiched in, either skip the trade, close the short leg before the ex-div date, or be prepared to manage the assignment (close the resulting stock position immediately at the open).
- **Correlation hiding inside "diversification".**
  A single session's Recommended list often clusters by sector (one earnings season, similar reporting cadence, similar macro exposure): on 2026-07-15, tonight's 7 Recommended names included three financials reporting together.
  Five different tickers is not five independent bets; they share a macro environment (rate decisions, geopolitics, sector rotations) that can crush all five calendars at once.
  Don't mistake a Recommended count for diversification; cross-check the sector mix and consider trimming concentration before sizing.
- **Sizing creep after a winner.**
  Don't double up after a +25% trade: a −60% trade is one event away.
  Variance is the dominant feature, not skill.

## When to push back on the user

- **"I'll pick one Recommended name and go big."** -> Restate the variance and sizing math.
  This is a portfolio strategy.
- **"I'll hold past the day after earnings."** -> Explain that the edge is the IV crush, and once that window has passed you're left with a peaked-payoff position that needs the stock to _stay near the strike_, not the directional setup most retail traders expect when they "hold for a rally."
  Close on schedule.
- **"Can I apply this to put calendars?"** -> Technically yes, if the bid-ask spreads are materially tighter in the ATM put calendar then the ATM put calendar is a less costly way to capture the earnings volatility premium.
- **"Can I apply this to iron condors / strangles?"** -> No.
  The scan is calibrated only for **long calendar spreads**; other structures need different filters.
- **"Can I override the slope filter?"** -> No.
  It's the gating filter for a reason; without backwardation the calendar mechanics don't work.
- **"Just tweak the threshold / window a bit."** -> No.
  The four constants are calibrated as a set against one backtest; moving one silently invalidates the published figures.
  See **Reference: filters and verdicts**.
- **"The IV/RV comparison looks misaligned, let me fix it."** -> No, and this one is seductive enough that it has its own section: **Why IV and RV are compared with no lag**.
- **API errors (`401`, `403`, quota).** -> The script diagnoses these itself and prints what happened; relay that, and see **If the script reports an error**.
  The one that matters: a Cloudflare / Error 1010 `403` is **not** an auth problem, so never tell the user their key or subscription is bad on the strength of a 403 alone.

## Installing this skill (for end users)

This skill is one file.
Drop it in wherever your AI assistant looks for instructions; no other setup is required.

- **Claude Code (CLI).**
  Ask Claude to run one of these from any project, then reload the session:
  - **macOS / Linux (bash, zsh):**
    ```
    curl --create-dirs -o ~/.claude/skills/uw-earnings-vol-scan/SKILL.md <skill-url>
    ```
  - **Windows (PowerShell):** `~` does not expand inside the quoted `-o` path the way bash does, so use `$env:USERPROFILE` explicitly and invoke real curl (`curl.exe`, since `curl` is aliased to `Invoke-WebRequest` in PowerShell):
    ```
    curl.exe --create-dirs -o "$env:USERPROFILE\.claude\skills\uw-earnings-vol-scan\SKILL.md" <skill-url>
    ```
  - **Fallback:** paste this file's contents into a new file at the path above by hand.
- **Claude.ai web (Projects).**
  Create a project, upload this SKILL.md to **Project Knowledge**, and invoke from any chat in the project.
  Note: the Analysis sandbox on claude.ai blocks outbound network to the API host, so Claude there cannot run the scan itself; it will read this skill and walk you through running the script on your own machine.
  To have the assistant run the scan directly, use Claude Code or the ChatGPT desktop (Codex) app instead.
- **ChatGPT (Custom GPT).**
  Create a Custom GPT and **upload this SKILL.md as a Knowledge file** (Configure -> Knowledge -> Upload files).
  Do **not** paste the full SKILL.md into the Instructions box as Custom GPT instructions cap at ~8,000 characters and will silently truncate this file.
  In the Instructions box, write a one-line pointer like: _"For any earnings-scan or calendar-spread question, follow the procedure in the attached `SKILL.md` Knowledge file exactly."_
  A Custom GPT runs on chatgpt.com, where Code Interpreter is network-blocked, so it **cannot** reach the API itself (verified 2026-07-16); it will guide the user to run the script locally.
  The ChatGPT path that runs the scan directly is the **ChatGPT desktop app (now Codex)**, not a chatgpt.com Custom GPT.
- **Cursor.**
  Save this file at `.cursor/rules/uw-earnings-vol-scan.mdc` in your workspace since that's the current rules path.
  (Cursor's older `.cursorrules` single-file format still works but is deprecated; new projects should prefer the `.cursor/rules/*.mdc` directory.)

Across all environments, the user supplies their **Unusual Whales API key**, and the assistant should prefer the most secure path that fits the environment: an existing `UW_API_KEY` env var, else a local `.env` file (gitignored, never committed) holding `UW_API_KEY=your-key` that the script reads itself, else the key pasted into chat and passed via `--api-key` as a plaintext last resort.
The assistant writes and runs the scanner script itself, so the user never clones a repo or makes a virtualenv.
The one hard requirement for running the scan **in place** is that the assistant's environment can **execute Python and reach `api.unusualwhales.com`**.
A chat with no code execution, or a web analysis sandbox that blocks outbound network (Claude Desktop, Claude.ai Analysis, ChatGPT Code Interpreter), cannot scan in place.
Even there the skill is still usable: the assistant walks the user through running the script on their own machine (see **How to run the scan**).

## The scanner script

Copy this **entire block, verbatim** to `uw_earnings_vol_scan.py` and run `python uw_earnings_vol_scan.py --selftest` before anything else. Do not edit it, do not change the constants, and do not skip the selftest. See **How to run the scan** above.

This script is the specification for every number this skill produces. If you find yourself about to compute a volatility, a slope, a ratio, or a trade-by date by hand, stop: run the script instead.

```python
#!/usr/bin/env python3
"""
uw_earnings_vol_scan.py - Unusual Whales earnings scanner for long call calendar spreads.

Scans upcoming earnings and scores each optionable name against three filters
(liquidity, IV/RV richness, term-structure backwardation), producing a
Recommended / Consider / Avoid verdict per the Volatility Vibes strategy.

USAGE
    python uw_earnings_vol_scan.py --selftest         # zero API calls; run this FIRST
    python uw_earnings_vol_scan.py --scan [--days N]  # scan the earnings window (7)
    python uw_earnings_vol_scan.py --ticker XYZ       # full metrics for one name

    Add --rich for a human-readable ANSI table. Default output is machine-readable.
    The API key comes from UW_API_KEY, a local .env file, or --api-key.

WHY THIS FILE EXISTS
    The numbers here drive real trades. Specified as prose, they were interpreted
    differently by every agent that read them. Specified as code, they are
    identical on every run and checkable by --selftest before a single API call.

    This script is deliberately dependency-free (stdlib + an HTTP client that is
    almost certainly already installed) and deliberately one file, so an agent can
    write it to disk verbatim and run it in any environment that has Python.

    ALWAYS RUN --selftest BEFORE A SCAN. It proves the file transcribed correctly:
    Yang-Zhang against a known fixture, the term-structure interpolation, the
    trade-by calendar, and all 8 rows of the verdict matrix. It takes about a
    second and makes "did this copy correctly" checkable instead of assumed.

DISCLAIMER
    Educational and informational purposes only. Not investment advice. A
    "Recommended" verdict means a ticker passed three statistical filters,
    nothing more. Trading options risks the loss of your entire investment.
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone

# ============================================================================
# CONSTANTS
# ============================================================================
# These four numbers ARE the strategy. They come from the Volatility Vibes
# backtest (US equity earnings 2007-2024): 10.80% universe pass rate, 7,313
# trades, +7.28% mean per-trade return, 28.6% std dev. Changing any of them
# invalidates those figures.

AVG_VOLUME_THRESHOLD = 1_500_000    # 30-day average share volume (liquidity proxy)
IV_RV_RATIO_THRESHOLD = 1.25        # iv30 / rv30; calibrated AGAINST RV_WINDOW=30 below
TS_SLOPE_THRESHOLD = -0.00406       # per-day IV slope, nearest DTE out to 45 DTE

# RV window. Held at 30 trading days: (30, 1.25) is the only pair with
# backtest evidence behind it (7,313 trades, 2007-2024).
#
# EXPERIMENT OPPORTUNITY: 30 trading days is ~44 calendar days, while IV30
# is 30 calendar days (~21 trading days), so the durations are different.
# Tested 2026-07-15 across 113 earnings names: dropping to 21 flips 5.3% of
# verdicts and LOOSENS the filter (26.5% -> 30.1% pass on this filter alone).
# Worth exploring. But if you change this and you want to stay consistent
# with the backtest data, you MUST also move IV_RV_RATIO_THRESHOLD
# (~1.27 for window 21) in the same commit, or else you will be operating off
# the backtest with no P&L validation for the new 21/21 durations.
RV_WINDOW_TRADING_DAYS = 30

# NOT the same 30 as RV_WINDOW_TRADING_DAYS. This one is CALENDAR days, because
# that is what an option's DTE is measured in. The names are verbose on purpose:
# "IV30 vs RV30" hides a 30-calendar vs 30-trading-day mismatch that is deliberate
# and documented in the Skill. Do not collapse these into one constant.
IV_HORIZON_CALENDAR_DAYS = 30

SLOPE_HORIZON_CALENDAR_DAYS = 45    # far end of the slope measurement
# Annualization factor for realized vol. 251, not the conventional 252, because
# Juneteenth became a full NYSE closure in 2022.
#
# KNOW WHAT THIS COSTS. The backtest behind IV_RV_RATIO_THRESHOLD ran 2007-2024
# at sqrt(252), and most of that period really did have 252 days. Moving to 251
# scales rv30 by sqrt(251/252) = 0.998, so iv_rv rises 0.199% and the filter
# LOOSENS very slightly. Measured on the 2026-07-15 live scan: 0 of 13 names
# changed verdict, and the shift is ~7x smaller than the 30-to-21-RV-window
# change described in "EXPERIMENT OPPORTUNITY".
#
# The IV_RV_RATIO_THRESHOLD is left at 1.25 even though this is technically
# a departure from the backtest conditions.
TRADING_PERIODS_PER_YEAR = 251

API_BASE = "https://api.unusualwhales.com/api"
UW_CLIENT_API_ID = "100002"  # fixed client id for UW usage attribution; do not change
MAX_INFLIGHT_TICKERS = 4     # keep a 7-day scan to a couple of minutes
RATE_LIMIT_MAX_RETRIES = 3

# Sent on every request, by every backend. Cloudflare blocks the default urllib
# User-Agent (see HTTP section).
USER_AGENT = "uw-earnings-vol-scan/1.0 (+https://unusualwhales.com)"


class SkipTicker(Exception):
    """This ticker cannot be scored. Record the reason, keep the batch going."""


class FatalError(Exception):
    """The whole run cannot continue (auth, quota, bad endpoint).

    Carries the remedy in `help` rather than buried in the message body, so
    main() can emit an `error:` / `help:` pair. An agent reads `help:` to
    decide what to do next; leave it None only when nothing actionable exists.
    """

    def __init__(self, message, help_text=None):
        super().__init__(message)
        self.help = help_text


# ============================================================================
# HTTP
# ============================================================================
# Prefer a battle-tested client, but urllib is a working fallback, not a cursed
# one, PROVIDED it sends a User-Agent.
#
# Cloudflare fronts this API and bans urllib's default User-Agent with a 403
# (Error 1010, "access denied ... based on your browser's signature").
# That 403 lands before the request ever reaches UW's auth layer, which is
# why it gets misdiagnosed as a bad API key.
#
# ANY User-Agent but the default passes so spoofing a browser buys nothing.

def _pick_http_backend():
    try:
        import requests  # noqa: F401
        return "requests"
    except ImportError:
        pass
    try:
        import httpx  # noqa: F401
        return "httpx"
    except ImportError:
        pass
    # stderr, not stdout: this is a diagnostic for a human, and stdout carries
    # the machine-readable result an agent parses.
    sys.stderr.write(
        "NOTE: neither `requests` nor `httpx` is installed; using the urllib\n"
        "fallback. It works (a User-Agent is set to clear Cloudflare), but it has\n"
        "no connection pooling and leans on an upstream bot rule we do not\n"
        "control. Prefer `pip install requests` where that is possible.\n\n"
    )
    return "urllib"


HTTP_BACKEND = None  # resolved lazily so --selftest needs no HTTP client at all


def _raw_get(url, params, headers, timeout=30):
    """Return (status_code, body_text, response_headers_lowercased)."""
    global HTTP_BACKEND
    if HTTP_BACKEND is None:
        HTTP_BACKEND = _pick_http_backend()

    if HTTP_BACKEND == "requests":
        import requests
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        return r.status_code, r.text, {k.lower(): v for k, v in r.headers.items()}

    if HTTP_BACKEND == "httpx":
        import httpx
        r = httpx.get(url, params=params, headers=headers, timeout=timeout)
        return r.status_code, r.text, {k.lower(): v for k, v in r.headers.items()}

    from urllib.error import HTTPError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    full = url + ("?" + urlencode(params) if params else "")
    # `headers` MUST carry a User-Agent or Cloudflare 403s this branch. Client
    # always supplies one; any new caller of _raw_get has to as well.
    try:
        with urlopen(Request(full, headers=headers), timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8"), \
                {k.lower(): v for k, v in resp.headers.items()}
    except HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), \
            {k.lower(): v for k, v in (e.headers or {}).items()}


class Client:
    """UW API client with the failure handling the strategy needs.

    Serializes every request once a per-minute 429 is seen, which is cheaper than
    thrashing against the limit for the rest of a batch.
    """

    def __init__(self, api_key):
        # Every backend sends exactly these. User-Agent is not decoration: drop
        # it and the urllib fallback 403s on every call (see the HTTP section).
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "UW-CLIENT-API-ID": UW_CLIENT_API_ID,
        }
        self._lock = threading.Lock()
        self._serialize = False

    def get(self, path, params=None):
        """GET {API_BASE}{path} -> the parsed `data` list. Raises SkipTicker/FatalError."""
        if self._serialize:
            with self._lock:
                return self._get_once(path, params)
        return self._get_once(path, params)

    def _get_once(self, path, params):
        url = API_BASE + path
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            status, body, hdrs = _raw_get(url, params or {}, self.headers)

            if status == 200:
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    raise SkipTicker(f"{path} returned unparseable JSON")
                if not isinstance(payload, dict) or "data" not in payload:
                    raise SkipTicker(f"{path} response has no 'data' key")
                data = payload["data"]
                if not isinstance(data, list):
                    raise SkipTicker(f"{path} 'data' is not a list")
                return data

            if status == 401:
                raise FatalError(
                    f"401 from UW: bad or missing API key.\n  endpoint: {path}",
                    "set a working key in UW_API_KEY or pass --api-key. "
                    "Get one at https://unusualwhales.com/pricing?product=api.",
                )

            if status == 403:
                low = body.lower()
                # A 403 comes from one of three places, and the body plus headers
                # say which. The remedies are mutually exclusive, so getting this
                # wrong points the user at the wrong problem:
                #   host / network / allowlist language -> the environment's proxy
                #   browser-signature / bot / 1010      -> Cloudflare
                #   key / plan / subscription           -> UW auth
                # Check egress FIRST. A web AI sandbox (Claude Desktop, claude.ai
                # Analysis, chatgpt.com) mints a synthetic 403 from its egress
                # allowlist before the request ever leaves the box. It carries
                # neither a real UW body nor Cloudflare's headers, so both branches
                # below would misread it -- the generic one as an expired key,
                # which is a brand-new subscriber's worst first-run message.
                deny = hdrs.get("x-deny-reason", "")  # _raw_get lowercases keys
                if deny == "host_not_allowed" or any(
                        s in low for s in ("not in allowlist", "egress", "host_not_allowed")):
                    raise FatalError(
                        "403 from the sandbox's egress proxy, NOT from UW. An egress\n"
                        "  allowlist blocked the request before it left the environment,\n"
                        "  so it never reached UW's auth layer: your API key and\n"
                        "  subscription are NOT the problem.\n"
                        f"  x-deny-reason: {deny or '(none)'}\n  body: {body[:200]}",
                        "this environment blocks outbound network to "
                        "api.unusualwhales.com. Allow that host in your egress/network "
                        "settings, or run the script on your own machine, where the "
                        "API is reachable.",
                    )
                cf = any(s in low for s in
                         ("cloudflare", "1010", "access denied", "bot", "browser integrity"))
                if cf:
                    # Do NOT re-add "your key is wrong" guidance here. A CF 403
                    # is decided before UW ever sees the token.
                    if HTTP_BACKEND == "urllib":
                        fix = ("this path already sends a User-Agent, which is what "
                               "normally clears this block, so Cloudflare's rule has "
                               "most likely tightened. Try `pip install requests` "
                               "and re-run.")
                    else:
                        # Already on a normal client with a normal UA, so the
                        # signature is not the problem; the network probably is.
                        fix = (f"{HTTP_BACKEND} already sends an ordinary client "
                               "signature, so the block is more likely your network "
                               "than your code. A VPN, proxy, or datacenter IP can "
                               "trip Cloudflare. Try a different network.")
                    raise FatalError(
                        "403 from Cloudflare, NOT an auth error. The request was\n"
                        "  blocked upstream before it reached UW's auth layer, so your\n"
                        "  API key and subscription are NOT the problem.\n"
                        f"  http client tried: {HTTP_BACKEND}\n"
                        f"  body: {body[:200]}",
                        fix,
                    )
                raise FatalError(
                    "403 from UW: the key is valid but this request was refused.\n"
                    f"  endpoint: {path}\n  body: {body[:200]}",
                    "your key may have expired, check your plan at "
                    "https://unusualwhales.com/settings.",
                )

            if status == 404:
                raise FatalError(
                    f"404: the route {path} does not exist.\n"
                    "  This is a wrong endpoint path, not an 'empty data' signal.",
                    "do not invent path variants to work around this. Verify the "
                    "route against https://api.unusualwhales.com/docs#/",
                )

            if status == 429:
                daily = hdrs.get("x-uw-daily-req-count")
                cap = hdrs.get("x-uw-token-req-limit")
                if daily is not None and cap is not None:
                    try:
                        if int(daily) >= int(cap):
                            raise FatalError(
                                f"daily UW request quota exhausted ({daily} of {cap} "
                                "used).\n  This does not recover in-session: the quota "
                                "resets at 20:00 US/Eastern.",
                                "wait for the 20:00 US/Eastern reset, or upgrade your "
                                "UW tier at https://unusualwhales.com/pricing?product=api. "
                                "Note the reset lands after the close, so a quota "
                                "exhausted during the session is gone for today's trading.",
                            )
                    except ValueError:
                        pass
                if attempt == RATE_LIMIT_MAX_RETRIES:
                    raise SkipTicker(f"rate limited on {path} after {attempt} retries")
                self._serialize = True  # per-minute cap hit; stop hammering it
                try:
                    wait_ms = min(int(hdrs.get("x-uw-req-per-minute-reset", 1000)), 60_000)
                except ValueError:
                    wait_ms = 1000
                time.sleep(max(wait_ms, 100) / 1000.0)
                continue

            if 500 <= status < 600:
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                raise SkipTicker(f"{path} returned {status} twice")

            raise SkipTicker(f"{path} returned unexpected status {status}")

        raise SkipTicker(f"{path} exhausted retries")


# ============================================================================
# US MARKET CALENDAR
# ============================================================================
# Self-contained on purpose: no pandas_market_calendars, no tzdata. The trade-by
# date is the single most consequential output of this scanner (miss it and the
# edge is gone), so it must not depend on a package that may not be installed.

# NYSE full-day closures. Source: https://www.nyse.com/markets/hours-calendars
#
# A TABLE, NOT A COMPUTATION, on purpose. These dates are published years ahead
# and never move, so an algorithm buys only unbounded years, at the price of code
# no human can check. This table is checkable against the NYSE page in about two
# minutes, which is the entire point.
#
# A table also allows for ad-hoc closures; the NYSE shuts for national days of
# mourning (2025-01-09, President Carter) and for emergencies (2021-10-29/30,
# Hurricane Sandy).
#
# Each row is MM-DD in date order and normally reads: New Year, MLK, Presidents,
# Good Friday, Memorial, Juneteenth, July 4, Labor, Thanksgiving, Christmas.
# Two things break that shape and are correct, not typos:
#   - a year with New Year's Day on a SATURDAY has no January entry, because the
#     NYSE does not close the preceding Friday (see 2028);
#   - ad-hoc closures appear inline, in date order (see 2025-01-09).
#
# PROVENANCE: 2026, 2027 and 2028 were transcribed and verified against the NYSE
# page on 2026-07-15. The table stops at 2028 because that is as far as the NYSE
# publishes. Later years were deliberately NOT extrapolated, because a guessed
# holiday is indistinguishable from a real one at the call site and would defeat
# the loud failure below.
#
# MAINTENANCE: check the NYSE page once a year and append the newly published
# year. Running past the last year below is a hard, loud failure BY DESIGN, and
# must never soften into a guess: a missing holiday does not produce an
# obviously-wrong date, it quietly moves a deadline onto a closed day, drops the
# name out of tonight's list, and you learn about it after the print.
#
# This table therefore hard-stops the scanner in early 2029. That is intended:
# stopping is correct, and guessing is not.
_MARKET_HOLIDAYS = {
    2025: "01-01 01-09 01-20 02-17 04-18 05-26 06-19 07-04 09-01 11-27 12-25",
    2026: "01-01 01-19 02-16 04-03 05-25 06-19 07-03 09-07 11-26 12-25",
    2027: "01-01 01-18 02-15 03-26 05-31 06-18 07-05 09-06 11-25 12-24",
    2028: "01-17 02-21 04-14 05-29 06-19 07-04 09-04 11-23 12-25",
}

MARKET_HOLIDAYS = {
    y: frozenset(date(y, int(md[:2]), int(md[3:])) for md in mds.split())
    for y, mds in _MARKET_HOLIDAYS.items()
}


def _nth_weekday(year, month, weekday, n):
    """n-th `weekday` (Mon=0) of month. n=-1 means the last one.

    Only survives for the DST boundaries in _eastern_now; the holiday calendar
    is now a table.
    """
    if n == -1:
        d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        return d - timedelta(days=(d.weekday() - weekday) % 7)
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def market_holidays(year):
    """NYSE full-day closures for `year`. Raises rather than guessing."""
    try:
        return MARKET_HOLIDAYS[year]
    except KeyError:
        lo, hi = min(MARKET_HOLIDAYS), max(MARKET_HOLIDAYS)
        raise FatalError(
            f"the NYSE holiday table does not cover {year}; it runs {lo}-{hi}.\n"
            "  Refusing to guess. Every trade-by date this scanner prints depends\n"
            "  on knowing which days the market is closed, and a missing holiday\n"
            "  fails silently: the deadline lands on a closed day, the name drops\n"
            "  out of tonight's list, and the print happens without you.",
            f"add {year} to _MARKET_HOLIDAYS from "
            "https://www.nyse.com/markets/hours-calendars, then re-run. Do not "
            "widen the range without transcribing the real dates.",
        )


def is_trading_day(d):
    return d.weekday() < 5 and d not in market_holidays(d.year)


def previous_trading_day(d):
    """The latest trading day <= d."""
    for _ in range(15):  # longest real closure stretch is a few days
        if is_trading_day(d):
            return d
        d -= timedelta(days=1)
    raise ValueError(f"no trading day found on or before {d}")


def next_trading_day(d):
    """The earliest trading day > d.

    This is what makes an AM report "tonight's problem": on the Thursday before
    Good Friday, the next session is Monday, so Monday's pre-open reporters must
    be entered before today's close or not at all.
    """
    for _ in range(15):
        d += timedelta(days=1)
        if is_trading_day(d):
            return d
    raise ValueError(f"no trading day found after {d}")


def trade_by_date(report_date, report_time):
    """The last session in which the position can still be opened before the print.

    PM report on X -> enter before X's close.
    AM report on X -> enter before the PREVIOUS session's close, rolling back over
    weekends and holidays.

    `report_time` is UW's value: "premarket" means AM, anything else (the
    afterhours endpoint returns "postmarket") means PM.
    """
    if report_time == "premarket":
        return previous_trading_day(report_date - timedelta(days=1))
    return previous_trading_day(report_date)


def _eastern_now(utc_now):
    """US/Eastern wall clock, without tzdata (absent on many Windows installs).

    US DST: 2nd Sunday of March 07:00 UTC to 1st Sunday of November 06:00 UTC.
    """
    y = utc_now.year
    start = datetime.combine(_nth_weekday(y, 3, 6, 2), datetime.min.time()) + timedelta(hours=7)
    end = datetime.combine(_nth_weekday(y, 11, 6, 1), datetime.min.time()) + timedelta(hours=6)
    return utc_now + timedelta(hours=-4 if start <= utc_now < end else -5)


def today_eastern():
    return _eastern_now(datetime.now(timezone.utc).replace(tzinfo=None)).date()


# ============================================================================
# MATH
# ============================================================================

def yang_zhang(bars, window=RV_WINDOW_TRADING_DAYS):
    """Annualized Yang-Zhang realized volatility over the last `window` returns.

    `bars` is a chronological list of dicts with open/high/low/close floats, and
    needs window+1 of them: the extra bar supplies close_{i-1} for the first return.

    Returns None when the inputs cannot produce a usable number, which the caller
    must treat as "skip this ticker" rather than as a zero.
    """
    if len(bars) < window + 1:
        return None
    w = bars[-(window + 1):]

    close_sq = open_sq = rs_sum = 0.0
    for i in range(1, window + 1):
        o, h, l, c = w[i]["open"], w[i]["high"], w[i]["low"], w[i]["close"]
        prev_c = w[i - 1]["close"]
        if min(o, h, l, c, prev_c) <= 0:
            return None  # a non-positive price makes the logs undefined
        log_ho = math.log(h / o)
        log_lo = math.log(l / o)
        log_co = math.log(c / o)
        log_oc = math.log(o / prev_c)
        log_cc = math.log(c / prev_c)
        open_sq += log_oc ** 2
        close_sq += log_cc ** 2
        rs_sum += log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    n = window
    close_vol = close_sq / (n - 1)
    open_vol = open_sq / (n - 1)
    rs_vol = rs_sum / (n - 1)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    variance = open_vol + k * close_vol + (1 - k) * rs_vol
    if variance <= 0:
        return None  # numerical pathology, not a real zero-vol stock
    return math.sqrt(variance * TRADING_PERIODS_PER_YEAR)


def _interp_flat(curve, x):
    """Linear interpolation with FLAT extrapolation outside the endpoints.

    `curve` is [(dte, iv), ...] sorted ascending by dte. Clamps end slopes
    instead of extending them.
    """
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


def iv30_from_term_structure(curve):
    """ATM IV at 30 calendar days, interpolated off the term-structure curve.

    Deliberately NOT the /interpolated-iv endpoint. The ratio's numerator and both
    ends of the slope must come from ONE curve, or nothing forces them to agree.
    This also saves an API call.
    """
    return _interp_flat(curve, IV_HORIZON_CALENDAR_DAYS)


def term_structure_slope(curve):
    """Per-day ATM IV slope from the nearest expiration out to 45 DTE.

    Raises SkipTicker rather than flat-extrapolating when the curve never reaches
    45 DTE to ensure a real slope since that measure is the gating filter.
    """
    nearest_dte = curve[0][0]
    if nearest_dte >= SLOPE_HORIZON_CALENDAR_DAYS:
        raise SkipTicker(
            f"nearest expiration is {nearest_dte} DTE, at or past the "
            f"{SLOPE_HORIZON_CALENDAR_DAYS}-day slope horizon, so the slope is undefined"
        )
    if curve[-1][0] < SLOPE_HORIZON_CALENDAR_DAYS:
        raise SkipTicker(
            f"no expiration reaches {SLOPE_HORIZON_CALENDAR_DAYS} DTE "
            f"(furthest is {curve[-1][0]}), so the slope cannot be measured"
        )
    iv_far = _interp_flat(curve, SLOPE_HORIZON_CALENDAR_DAYS)
    iv_near = curve[0][1]
    return (iv_far - iv_near) / (SLOPE_HORIZON_CALENDAR_DAYS - nearest_dte), nearest_dte


def verdict(slope_pass, volume_pass, iv_rv_pass):
    """Recommended / Consider / Avoid.

    Slope is the gating filter: without backwardation the calendar mechanics do
    not work, so a failed slope is Avoid no matter what else passes.
    """
    if not slope_pass:
        return "Avoid"
    if volume_pass and iv_rv_pass:
        return "Recommended"
    if volume_pass or iv_rv_pass:
        return "Consider"
    return "Avoid"


# ============================================================================
# FETCH
# ============================================================================

def _to_float(row, field, where):
    """UW encodes OHLC and IV as strings but volume as a number. Coerce everything.

    Untyped string/number arithmetic produces silently wrong volatility instead of
    an error, so every numeric field goes through here.
    """
    try:
        v = row[field]
    except (KeyError, TypeError):
        raise SkipTicker(f"{where} entry missing '{field}'")
    if v is None:
        raise SkipTicker(f"{where} entry has null '{field}'")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise SkipTicker(f"{where} entry has non-numeric '{field}': {v!r}")


def tonight_buckets(today):
    """The (date, endpoint) pairs whose names ALL have trade_by == today.

    Exactly two, and this is the whole scanner's default job:

      1. names reporting AFTER tonight's close, and
      2. names reporting BEFORE the next session's open.

    Both are entered before today's close, which is why they are one list rather
    than two. (2) is why the holiday table exists: on the Thursday before Good
    Friday the next session is Monday, so Monday's pre-open reporters are
    tonight's problem, and a calendar that only skipped weekends would put them
    on Friday, hide them from this list, and lose the trade.

    Deliberately NOT a forward window. A name reporting on Thursday can pass all
    three filters on Tuesday and fail them by Thursday because the options
    market is continuously repricing outcomes. A multi-day list invites acting
    on a number that has already moved; this list is only ever about the next few hours.
    """
    return ((today, "afterhours"), (next_trading_day(today), "premarket"))


def window_buckets(today, days):
    """Every (date, endpoint) pair in a `days`-long forward window.

    Research and heads-up use only. See tonight_buckets on why the numbers in a
    multi-day scan are not tradeable as-is.
    """
    return tuple((today + timedelta(days=i), when)
                 for i in range(days)
                 for when in ("premarket", "afterhours"))


def fetch_earnings(client, buckets):
    """{symbol: {...}} for optionable names in the given (date, endpoint) buckets.

    Deduped by symbol across every bucket: a name can appear on more than one
    calendar date, and it must only be scored (and traded) once.
    """
    universe = {}
    for day, when in buckets:
        endpoint = "/earnings/premarket" if when == "premarket" else "/earnings/afterhours"
        page = 0
        while True:
            rows = client.get(endpoint, {"date": day.isoformat(),
                                         "limit": 100, "page": page})
            for r in rows:
                sym = (r.get("symbol") or "").strip().upper()
                if not sym or not r.get("has_options"):
                    continue
                if sym in universe:
                    continue
                em = r.get("expected_move_perc")
                try:
                    em = float(em) if em is not None else None
                except (TypeError, ValueError):
                    em = None
                universe[sym] = {
                    "ticker": sym,
                    "report_date": day,
                    # The afterhours endpoint returns "postmarket", not
                    # "afterhours". Anything that is not "premarket" is a PM print.
                    "report_time": "premarket" if (r.get("report_time") == "premarket"
                                                   or when == "premarket") else "postmarket",
                    "expected_move_perc": em,
                }
            if len(rows) < 100:
                break
            page += 1
    return universe


def fetch_ohlc(client, ticker, today=None):
    """(bars, bars_through) - completed regular-session daily bars, oldest first.

    PARTIAL BAR: any bar dated today (Eastern) is dropped. The textbook entry is
    15 minutes before the close on the trade-by date, so at the moment this
    scanner is meant to run, today's bar is still forming: its "close" is just the
    last print and its high/low have not finished widening.

    Dropping today's bar unconditionally, rather than only while the market is
    open, costs one session of freshness after the close but makes the result
    independent of what time of day the scan runs. `bars_through` is returned so
    the caller can surface which session the numbers actually end on.
    """
    today = today or today_eastern()
    rows = client.get(f"/stock/{ticker}/ohlc/1d", {"timeframe": "3M"})
    bars = []
    for r in rows:
        if r.get("market_time") != "r":  # regular session only; drop pre/post
            continue
        try:
            d = date.fromisoformat(r["date"][:10])
        except (KeyError, TypeError, ValueError):
            raise SkipTicker("ohlc entry has a missing or malformed 'date'")
        if d >= today:
            continue
        bars.append({
            "date": d,
            "open": _to_float(r, "open", "ohlc"),
            "high": _to_float(r, "high", "ohlc"),
            "low": _to_float(r, "low", "ohlc"),
            "close": _to_float(r, "close", "ohlc"),
            "volume": _to_float(r, "volume", "ohlc"),
        })
    bars.sort(key=lambda b: b["date"])
    need = RV_WINDOW_TRADING_DAYS + 1
    if len(bars) < need:
        raise SkipTicker(f"only {len(bars)} completed daily bars available, need {need}")
    return bars, bars[-1]["date"]


def fetch_term_structure(client, ticker):
    """[(dte, iv), ...] sorted ascending, expired entries dropped."""
    rows = client.get(f"/stock/{ticker}/volatility/term-structure")
    curve = []
    for r in rows:
        dte = _to_float(r, "dte", "term-structure")
        if dte <= 0:
            continue
        curve.append((int(dte), _to_float(r, "volatility", "term-structure")))
    curve.sort(key=lambda p: p[0])
    # Same DTE twice would make the interpolation ambiguous; keep the first.
    curve = [p for i, p in enumerate(curve) if i == 0 or p[0] != curve[i - 1][0]]
    if len(curve) < 2:
        raise SkipTicker(
            f"only {len(curve)} usable term-structure point(s) after dropping "
            "expired entries; a slope needs at least 2"
        )
    return curve


# ============================================================================
# SCORING
# ============================================================================

def score_ticker(client, meta, today=None):
    """Score one name. Raises SkipTicker with a human-readable reason."""
    ticker = meta["ticker"]
    bars, bars_through = fetch_ohlc(client, ticker, today=today)
    curve = fetch_term_structure(client, ticker)

    rv30 = yang_zhang(bars)
    if rv30 is None or rv30 <= 0:
        raise SkipTicker("Yang-Zhang returned no usable realized volatility")

    iv30 = iv30_from_term_structure(curve)
    if iv30 <= 0:
        raise SkipTicker(f"interpolated iv30 is {iv30}, which cannot form a ratio")

    slope, nearest_dte = term_structure_slope(curve)
    iv_rv_ratio = iv30 / rv30
    avg_volume = sum(b["volume"] for b in bars[-RV_WINDOW_TRADING_DAYS:]) / RV_WINDOW_TRADING_DAYS

    volume_pass = avg_volume >= AVG_VOLUME_THRESHOLD
    iv_rv_pass = iv_rv_ratio >= IV_RV_RATIO_THRESHOLD
    slope_pass = slope <= TS_SLOPE_THRESHOLD

    # report_date is None only for a --ticker lookup on a name with no earnings in
    # the window. Emit no trade-by rather than inventing one: a fabricated deadline
    # is worse than an absent one.
    report_date = meta.get("report_date")
    return {
        "ticker": ticker,
        "report_date": report_date,
        "report_time": meta.get("report_time"),
        "trade_by": trade_by_date(report_date, meta["report_time"]) if report_date else None,
        "verdict": verdict(slope_pass, volume_pass, iv_rv_pass),
        "avg_volume": avg_volume,
        "volume_pass": volume_pass,
        "iv_rv_ratio": iv_rv_ratio,
        "iv_rv_pass": iv_rv_pass,
        "slope": slope,
        "slope_pass": slope_pass,
        "iv30": iv30,
        "rv30": rv30,
        "nearest_dte": nearest_dte,
        "expected_move_perc": meta.get("expected_move_perc"),
        "bars_through": bars_through,
    }


VERDICT_ORDER = {"Recommended": 0, "Consider": 1, "Avoid": 2}


def scan(client, days=None, today=None):
    """Score the actionable set. Never raises SkipTicker.

    days=None (the default) means tonight only: every name returned has a
    trade_by of today. Pass days=N for a forward research window instead.
    """
    from concurrent.futures import ThreadPoolExecutor

    today = today or today_eastern()
    buckets = tonight_buckets(today) if days is None else window_buckets(today, days)
    universe = fetch_earnings(client, buckets)
    results, skipped = [], []

    def work(meta):
        try:
            return ("ok", score_ticker(client, meta, today=today))
        except SkipTicker as e:
            return ("skip", {"ticker": meta["ticker"], "reason": str(e)})

    if universe:
        with ThreadPoolExecutor(MAX_INFLIGHT_TICKERS) as pool:
            for kind, payload in pool.map(work, sorted(universe.values(),
                                                       key=lambda m: m["ticker"])):
                (results if kind == "ok" else skipped).append(payload)

    results.sort(key=lambda r: (r["trade_by"], VERDICT_ORDER[r["verdict"]], r["ticker"]))
    skipped.sort(key=lambda r: r["ticker"])
    return results, skipped, len(universe)


# ============================================================================
# OUTPUT
# ============================================================================

def _fmt_earnings(r):
    if not r.get("report_date"):
        return "-"
    return f"{r['report_date'].isoformat()} {'AM' if r['report_time'] == 'premarket' else 'PM'}"


def _fmt_trade_by(r):
    return r["trade_by"].isoformat() if r.get("trade_by") else "-"


def _fmt_em(r):
    em = r.get("expected_move_perc")
    return "-" if em is None else f"{em * 100:.1f}%"


def _pf(ok):
    return "PASS" if ok else "FAIL"


def _row(*vals):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(vals)
    return buf.getvalue()


def _bars_through_line(results):
    stamps = sorted({r["bars_through"] for r in results})
    if not stamps:
        return None
    if len(stamps) == 1:
        return stamps[0].isoformat()
    return f"mixed ({stamps[0].isoformat()}..{stamps[-1].isoformat()})"


def print_scan(results, skipped, universe_n, days, today, out=sys.stdout):
    counts = {v: sum(1 for r in results if r["verdict"] == v) for v in VERDICT_ORDER}
    w = out.write
    if days is None:
        # Tonight mode: trade_by is today for every row by construction, so it is
        # stated once here instead of repeated down a column that never varies.
        w(f"session: {today.isoformat()}\n")
        w(f"trade_by: {today.isoformat()} (every name below; enter before today's close)\n")
        w(f"covers: reports after tonight's close, plus pre-open reports on "
          f"{next_trading_day(today).isoformat()}\n")
    else:
        end = today + timedelta(days=days - 1)
        w(f"window: {today.isoformat()} to {end.isoformat()} ({days} days)\n")
    w(f"as_of: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
    bt = _bars_through_line(results)
    if bt:
        w(f"bars_through: {bt}\n")
    w(f"rv_window_trading_days: {RV_WINDOW_TRADING_DAYS}\n")
    w(f"counts: {counts['Recommended']} recommended, {counts['Consider']} consider, "
      f"{counts['Avoid']} avoid ({len(results)} scored)\n")
    w(f"skipped: {len(skipped)} of {universe_n}\n")
    # The pass flags are emitted rather than left for the reader to derive: an
    # agent re-deriving them from the thresholds is the same "prose interpreted N
    # ways" hazard this script exists to remove, and Consider does not say WHICH
    # of the two non-gating filters passed.
    tb = "" if days is None else "trade_by,"
    w(f"results[{len(results)}]"
      "{ticker,earnings," + tb + "verdict,avg_volume,volume_pass,iv_rv,iv_rv_pass,"
      "slope,slope_pass,expected_move}:\n")
    for r in results:
        cells = [r["ticker"], _fmt_earnings(r)]
        if days is not None:
            cells.append(_fmt_trade_by(r))
        cells += [r["verdict"], f"{r['avg_volume']:.0f}", _pf(r["volume_pass"]),
                  f"{r['iv_rv_ratio']:.2f}", _pf(r["iv_rv_pass"]),
                  f"{r['slope']:.5f}", _pf(r["slope_pass"]), _fmt_em(r)]
        w("  " + _row(*cells) + "\n")
    if skipped:
        w(f"skipped[{len(skipped)}]" "{ticker,reason}:\n")
        for s in skipped:
            w("  " + _row(s["ticker"], s["reason"]) + "\n")
    w("help[2]:\n" if days is None else "help[4]:\n")
    w("  A Recommended verdict means three filters passed, nothing more. It is not advice.\n")
    if days is not None:
        # Only reachable in window mode; in tonight mode every trade_by is today.
        w(f"  A trade_by before {today.isoformat()} has already passed; that name is no longer tradeable.\n")
        w("  These numbers are live and move: re-run on each name's trade_by date before acting.\n")
    w("  Run --ticker TICKER for the full pass/fail detail on one name.\n")


def print_ticker(r, out=sys.stdout):
    pf = _pf
    w = out.write
    w(f"ticker: {r['ticker']}\n")
    w(f"earnings: {_fmt_earnings(r)}\n")
    w(f"trade_by: {_fmt_trade_by(r)}\n")
    if not r.get("report_date"):
        w("note: not on the earnings calendar in the scanned window, so there is no\n"
          "  trade-by deadline. These filters only carry their backtested meaning\n"
          "  immediately before an earnings print.\n")
    w(f"verdict: {r['verdict']}\n")
    w(f"avg_volume: {r['avg_volume']:.0f} {pf(r['volume_pass'])}\n")
    w(f"iv_rv: {r['iv_rv_ratio']:.2f} {pf(r['iv_rv_pass'])}\n")
    w(f"slope: {r['slope']:.5f} {pf(r['slope_pass'])}\n")
    w(f"iv30: {r['iv30']:.4f}\n")
    w(f"rv30: {r['rv30']:.4f}\n")
    w(f"rv_window_trading_days: {RV_WINDOW_TRADING_DAYS}\n")
    w(f"nearest_dte: {r['nearest_dte']}\n")
    w(f"expected_move: {_fmt_em(r)}\n")
    w(f"as_of: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
    w(f"bars_through: {r['bars_through'].isoformat()}\n")


# Lineage colors
_COLORS = {"Recommended": "\x1b[38;2;0;102;0m", "Consider": "\x1b[38;2;255;153;0m",
           "Avoid": "\x1b[38;2;128;0;0m"}
_DIM, _BOLD, _RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"


def _c(text, color):
    if os.environ.get("NO_COLOR"):
        return text
    return f"{color}{text}{_RESET}"


def print_rich(results, skipped, universe_n, days, today, out=sys.stdout):
    """Human-facing table. Opt-in only.

    Never auto-detected from TTY: agents frequently run commands in a pty, and
    would then receive ANSI escapes and try to parse them as data.
    """
    w = out.write
    tb_col = days is not None  # in tonight mode every trade_by is today; see print_scan
    if tb_col:
        end = today + timedelta(days=days - 1)
        title = f"Earnings volatility scan  {today.isoformat()} to {end.isoformat()}"
    else:
        title = f"Trade before tonight's close  {today.isoformat()}"
    hdr = (f"{'TICKER':<8}{'EARNINGS':<16}" + (f"{'TRADE BY':<12}" if tb_col else "")
           + f"{'VOLUME':>12}  {'IV/RV':>7}  {'SLOPE':>9}  {'EXP MOVE':>8}  VERDICT")
    w("\n" + _c(title, _BOLD) + "\n")
    w(_c(hdr, _DIM) + "\n")
    w(_c("-" * len(hdr), _DIM) + "\n")
    for r in results:
        vol = f"{r['avg_volume'] / 1e6:.1f}M"
        w(f"{r['ticker']:<8}{_fmt_earnings(r):<16}"
          + (f"{_fmt_trade_by(r):<12}" if tb_col else "")
          + f"{vol:>12}  {r['iv_rv_ratio']:>7.2f}  {r['slope']:>9.5f}  {_fmt_em(r):>8}  "
          f"{_c(r['verdict'], _COLORS[r['verdict']])}\n")
    counts = {v: sum(1 for r in results if r["verdict"] == v) for v in VERDICT_ORDER}
    w(f"\n{counts['Recommended']} Recommended | {counts['Consider']} Consider | "
      f"{counts['Avoid']} Avoid   ({len(results)} scanned)\n")
    if skipped:
        w(f"{len(skipped)} of {universe_n} tickers skipped (insufficient data "
          f"or malformed response)\n")
    bt = _bars_through_line(results)
    if bt:
        w(_c(f"bars through {bt} | rv window {RV_WINDOW_TRADING_DAYS} trading days", _DIM) + "\n")
    # A screenshot of this table travels without a single word of surrounding
    # context, including the Skill's disclaimer and the "Recommended is a label"
    # framing. This footer is the only thing that goes with the image.
    w("\n" + _c("\"Recommended\" is a label, not financial advice. Educational use only; "
                "options trading can lose your entire investment.", _DIM) + "\n")


# ============================================================================
# SELFTEST
# ============================================================================
# Zero API calls. This is what makes an embedded script safe to hand-copy: it
# turns "did the agent transcribe a thousand lines faithfully" from unknowable
# into checkable, for about a second of CPU.

# 31 deterministic OHLC bars selected to assert agreement with the backtest's
# reference data.
_YZ_FIXTURE = [
    (100.1862, 100.8624, 99.6885, 99.7951), (99.8149, 100.4163, 99.4251, 99.7943),
    (99.5029, 100.3244, 99.0811, 99.2525), (99.0117, 100.08, 98.0337, 99.2959),
    (99.6541, 100.1912, 98.96, 99.5828), (99.2841, 100.403, 98.3116, 99.6889),
    (99.4844, 100.1892, 98.6406, 99.3759), (98.9857, 99.7603, 97.9865, 98.0269),
    (98.1415, 98.6876, 97.1393, 97.6424), (97.5222, 98.1522, 96.2088, 96.9032),
    (96.6763, 98.4024, 96.2992, 97.588), (97.8279, 98.4224, 96.5621, 97.0952),
    (97.3281, 98.5038, 96.518, 98.1145), (98.2539, 98.6224, 97.1965, 97.3031),
    (97.3735, 97.5194, 96.3401, 97.1207), (96.6221, 96.8124, 94.9607, 95.8215),
    (95.878, 96.2388, 95.761, 95.8568), (95.8032, 96.1644, 95.3191, 95.4362),
    (95.9463, 96.9256, 95.4038, 96.5065), (95.9518, 96.9897, 95.617, 96.8351),
    (97.1241, 98.3634, 96.9463, 97.5517), (97.7841, 98.7533, 96.8788, 96.9084),
    (96.5134, 98.1275, 96.4756, 97.468), (97.1655, 97.59, 96.2822, 97.4361),
    (97.3646, 98.5589, 97.2055, 98.0702), (98.4007, 99.1846, 97.1384, 97.424),
    (96.9913, 97.4276, 96.2098, 96.4445), (96.7238, 97.4115, 96.521, 97.1377),
    (97.3385, 98.3919, 97.1297, 97.5331), (97.1042, 98.1768, 96.6852, 97.2167),
    (97.0746, 97.3518, 95.523, 96.221),
]
# Produced by the original strategy's own independent pandas implementation on
# these exact bars, at ITS sqrt(252) annualization. This is a CROSS-IMPLEMENTATION
# check, not the code asserting its own output: it is the only evidence that this
# file's Yang-Zhang variance (the k constant, the rolling windows, and the log
# returns) agrees with the implementation the 1.25 threshold was calibrated on.
#
# DO NOT update this number to match a code change. If it disagrees, the code is
# wrong. The one legitimate difference is the annualization factor, which is a
# separate, documented decision (see TRADING_PERIODS_PER_YEAR), so the check below
# rescales the reference by sqrt(periods/252) rather than loosening its tolerance.
# That keeps the variance held to the reference EXACTLY while leaving the
# annualization free to be argued about on its own terms.
_YZ_REFERENCE_ANNUALIZATION = 252
_YZ_EXPECTED_AT_252 = 0.1754215728753446  # full precision


def _raises_fatal(fn):
    """True if `fn()` raises FatalError. Keeps try/except out of the check list."""
    try:
        fn()
        return False
    except FatalError:
        return True


def selftest():
    checks = []

    def ok(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # -- Yang-Zhang against the reference implementation ----------------------
    bars = [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in _YZ_FIXTURE]
    rv = yang_zhang(bars)
    # Bit-exact, not 4dp: the two implementations differ ONLY by the annualization
    # constant, so once that is rescaled out there is nothing left to round away.
    # A 4dp check would silently absorb a real variance bug of up to 5e-5.
    _want = _YZ_EXPECTED_AT_252 * math.sqrt(
        TRADING_PERIODS_PER_YEAR / _YZ_REFERENCE_ANNUALIZATION)
    ok("yang_zhang variance matches the reference implementation exactly "
       f"(annualization rescaled {_YZ_REFERENCE_ANNUALIZATION} -> {TRADING_PERIODS_PER_YEAR})",
       rv is not None and abs(rv - _want) < 1e-12, f"got {rv!r}, want {_want!r}")
    ok("the annualization factor is the ONLY departure from the reference",
       rv is not None and abs(rv * math.sqrt(_YZ_REFERENCE_ANNUALIZATION
                                             / TRADING_PERIODS_PER_YEAR)
                              - _YZ_EXPECTED_AT_252) < 1e-12)
    ok("yang_zhang needs window+1 bars", yang_zhang(bars[:30]) is None)
    ok("yang_zhang rejects non-positive prices",
       yang_zhang([{"open": 1.0, "high": 1.0, "low": 1.0, "close": 0.0}] * 31) is None)

    # -- Term structure -------------------------------------------------------
    # Hand-checkable: iv(30)=0.45 sits exactly on a knot; iv(45) interpolates
    # halfway between the 30 and 60 knots -> 0.45 + 0.5*(0.40-0.45) = 0.425;
    # slope = (0.425-0.60)/(45-7) = -0.00460526...
    curve = [(7, 0.60), (30, 0.45), (60, 0.40)]
    ok("iv30 lands on the 30-day knot", abs(iv30_from_term_structure(curve) - 0.45) < 1e-12)
    ok("interpolation at 45 DTE is linear between knots",
       abs(_interp_flat(curve, 45) - 0.425) < 1e-12)
    slope, nearest = term_structure_slope(curve)
    ok("slope matches the hand calculation", abs(slope - (-0.175 / 38)) < 1e-12,
       f"got {slope!r}")
    ok("nearest_dte is the front expiration", nearest == 7)
    ok("flat extrapolation below the first knot",
       abs(iv30_from_term_structure([(40, 0.5), (60, 0.4)]) - 0.5) < 1e-12)
    ok("flat extrapolation above the last knot",
       abs(_interp_flat([(7, 0.6), (20, 0.5)], 45) - 0.5) < 1e-12)

    # Item 6: a curve that never reaches 45 DTE must raise, not flat-extrapolate.
    try:
        term_structure_slope([(7, 0.6), (30, 0.5)])
        ok("curve short of 45 DTE raises", False, "no exception raised")
    except SkipTicker:
        ok("curve short of 45 DTE raises", True)
    try:
        term_structure_slope([(50, 0.6), (80, 0.5)])
        ok("nearest expiration past 45 DTE raises", False, "no exception raised")
    except SkipTicker:
        ok("nearest expiration past 45 DTE raises", True)

    # -- Market calendar: the holiday TABLE ------------------------------------
    # These test what can actually go wrong with a hand-maintained table (a
    # typo, an omitted line, a year that silently ran out) rather than an
    # algorithm's arithmetic. That is the trade this table was made for.
    _years = sorted(MARKET_HOLIDAYS)
    ok("holiday table years are contiguous (no year silently skipped)",
       _years == list(range(_years[0], _years[-1] + 1)), f"got {_years}")
    for _y in _years:
        _hs = MARKET_HOLIDAYS[_y]
        # A weekend date in the table is always a transcription error: the NYSE
        # observes every closure on a weekday.
        _wknd = sorted(d for d in _hs if d.weekday() >= 5)
        ok(f"no {_y} holiday falls on a weekend", not _wknd, f"got {_wknd}")
        ok(f"every {_y} holiday is dated within {_y}", all(d.year == _y for d in _hs))
        # 9 is legitimate (a Saturday New Year's drops the January entry); more
        # than 11 means a duplicate, fewer than 9 means a dropped line.
        ok(f"{_y} has a plausible closure count ({len(_hs)})", 9 <= len(_hs) <= 11,
           f"got {len(_hs)}")

    h26 = market_holidays(2026)
    for iso in ("2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
                "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"):
        ok(f"{iso} is a 2026 market holiday", date.fromisoformat(iso) in h26)
    ok("July 4 2026 falls on a Saturday and is observed Friday July 3",
       date(2026, 7, 3) in h26 and date(2026, 7, 4).weekday() == 5)
    # NYSE does NOT close the preceding Friday when New Year's Day is a Saturday,
    # so 2028 correctly carries no January entry at all.
    ok("Jan 1 2028 is a Saturday and Dec 31 2027 stays a trading day",
       date(2028, 1, 1).weekday() == 5 and is_trading_day(date(2027, 12, 31)))
    # Note 2028 still lists MLK Day in January; what a Saturday New Year's drops
    # is the Jan 1/Jan 2 entry specifically, not the whole month.
    ok("the 2028 table carries no New Year's entry (Saturday New Year's)",
       date(2028, 1, 1) not in market_holidays(2028)
       and date(2028, 1, 2) not in market_holidays(2028))
    ok("an ad-hoc closure no formula could predict is carried (Carter, 2025-01-09)",
       date(2025, 1, 9) in market_holidays(2025))

    # The loud failure. This is the whole safety property of a hardcoded table:
    # past its horizon it must STOP, never guess. A silent guess moves a deadline
    # onto a closed day and the name vanishes from tonight's list.
    try:
        market_holidays(max(_years) + 1)
        ok("a year past the holiday table raises", False, "no exception raised")
    except FatalError as _e:
        ok("a year past the holiday table raises FatalError", True)
        ok("that failure names the NYSE source and the year to add",
           "nyse.com" in (_e.help or "") and str(max(_years) + 1) in (_e.help or ""))
    ok("a year before the holiday table also raises",
       _raises_fatal(lambda: market_holidays(min(_years) - 1)))

    # -- Eastern time / DST (the only surviving use of _nth_weekday) -----------
    ok("2026 DST starts the 2nd Sunday of March", _nth_weekday(2026, 3, 6, 2) == date(2026, 3, 8))
    ok("2026 DST ends the 1st Sunday of November", _nth_weekday(2026, 11, 6, 1) == date(2026, 11, 1))
    for _iso, _want in (("2026-01-15T12:00", -5), ("2026-03-08T06:59", -5),
                        ("2026-03-08T07:00", -4), ("2026-07-15T12:00", -4),
                        ("2026-11-01T05:59", -4), ("2026-11-01T06:00", -5)):
        _u = datetime.fromisoformat(_iso)
        _off = int((_eastern_now(_u) - _u).total_seconds() // 3600)
        ok(f"{_iso}Z is UTC{_want} in New York", _off == _want, f"got UTC{_off}")

    # -- Trade-by dates -------------------------------------------------------
    D = date.fromisoformat
    cases = [
        # (report_date, report_time, expected_trade_by, why)
        ("2026-07-16", "postmarket", "2026-07-16", "PM Thursday: enter before that close"),
        ("2026-07-16", "premarket", "2026-07-15", "AM Thursday: enter the prior session"),
        ("2026-07-20", "premarket", "2026-07-17", "AM Monday: back over the weekend to Friday"),
        # The required case: an AM report the Monday after a Friday holiday.
        ("2026-04-06", "premarket", "2026-04-02", "AM Monday after Good Friday -> Thursday"),
        ("2026-06-22", "premarket", "2026-06-18", "AM Monday after Juneteenth Friday"),
        ("2026-07-06", "premarket", "2026-07-02", "AM Monday after July 4 observed Friday"),
        ("2027-01-04", "premarket", "2026-12-31", "AM Monday after New Year's Friday"),
        ("2028-01-03", "premarket", "2027-12-31", "AM Monday, Dec 31 open (Sat New Year's)"),
        ("2026-11-27", "postmarket", "2026-11-27", "PM the Friday after Thanksgiving"),
        ("2026-11-27", "premarket", "2026-11-25", "AM Friday after Thanksgiving Thursday"),
    ]
    for rd, rt, want, why in cases:
        got = trade_by_date(D(rd), rt)
        ok(f"trade_by {rd} {rt} -> {want} ({why})", got == D(want), f"got {got}")

    ok("a PM report on a holiday rolls back to the prior session",
       trade_by_date(D("2026-07-03"), "postmarket") == D("2026-07-02"))
    ok("previous_trading_day is inclusive on a normal session",
       previous_trading_day(D("2026-07-15")) == D("2026-07-15"))

    # -- Tonight-only buckets --------------------------------------------------
    # The defining property, and the reason this mode can drop the trade_by
    # column: EVERY name these two buckets can return has trade_by == today.
    # Checked against trade_by_date itself, so the two cannot drift apart.
    for _iso, _why in (("2026-07-15", "an ordinary midweek session"),
                       ("2026-07-17", "a Friday: next session is Monday"),
                       ("2026-04-02", "the Thursday before Good Friday"),
                       ("2026-07-02", "the Thursday before July 4 observed"),
                       ("2026-11-25", "the day before Thanksgiving"),
                       ("2026-12-24", "the day before Christmas Friday")):
        _t = D(_iso)
        (_pm_day, _pm_when), (_am_day, _am_when) = tonight_buckets(_t)
        ok(f"tonight buckets on {_iso} ({_why}): both resolve to trade_by == {_iso}",
           trade_by_date(_pm_day, "postmarket") == _t
           and trade_by_date(_am_day, "premarket") == _t,
           f"pm {_pm_day} -> {trade_by_date(_pm_day, 'postmarket')}, "
           f"am {_am_day} -> {trade_by_date(_am_day, 'premarket')}")
        ok(f"tonight buckets on {_iso} look at tonight's close and the next open",
           (_pm_when, _am_when) == ("afterhours", "premarket")
           and _pm_day == _t and _am_day == next_trading_day(_t))

    ok("the Thursday before Good Friday reaches MONDAY's pre-open reporters",
       tonight_buckets(D("2026-04-02"))[1][0] == D("2026-04-06"))
    ok("tonight is exactly 2 API buckets, not a window", len(tonight_buckets(D("2026-07-15"))) == 2)
    ok("a 7-day window is 14 buckets", len(window_buckets(D("2026-07-15"), 7)) == 14)

    # A name with no earnings in the window must show no deadline, never a made-up one.
    off_calendar = {"ticker": "XYZ", "report_date": None, "report_time": None,
                    "trade_by": None, "expected_move_perc": None}
    ok("a name off the earnings calendar formats no earnings date",
       _fmt_earnings(off_calendar) == "-")
    ok("a name off the earnings calendar formats no trade-by date",
       _fmt_trade_by(off_calendar) == "-")

    # -- Verdict matrix, all 8 rows ------------------------------------------
    matrix = [
        # slope, volume, iv_rv, expected
        (True, True, True, "Recommended"),
        (True, True, False, "Consider"),
        (True, False, True, "Consider"),
        (True, False, False, "Avoid"),
        (False, True, True, "Avoid"),
        (False, True, False, "Avoid"),
        (False, False, True, "Avoid"),
        (False, False, False, "Avoid"),
    ]
    for s, v, i, want in matrix:
        got = verdict(s, v, i)
        ok(f"verdict(slope={s:<5} volume={v:<5} iv_rv={i:<5}) -> {want}", got == want,
           f"got {got}")

    # -- Threshold directions -------------------------------------------------
    # A sign error here would silently invert the gating filter. Backwardation is
    # NEGATIVE slope, so steeper (more negative) must pass and flat must fail.
    ok("steep backwardation passes the slope filter",
       verdict(-0.005 <= TS_SLOPE_THRESHOLD, True, True) == "Recommended")
    ok("a flat term structure fails the slope filter",
       verdict(0.0 <= TS_SLOPE_THRESHOLD, True, True) == "Avoid")
    ok("contango fails the slope filter",
       verdict(0.002 <= TS_SLOPE_THRESHOLD, True, True) == "Avoid")
    ok("thresholds match the backtest",
       (AVG_VOLUME_THRESHOLD, IV_RV_RATIO_THRESHOLD, TS_SLOPE_THRESHOLD)
       == (1_500_000, 1.25, -0.00406))
    ok("RV window is the backtested 30 trading days", RV_WINDOW_TRADING_DAYS == 30)

    # -- .env parser ----------------------------------------------------------
    # Pure string parsing, no filesystem: the file loader is a thin wrapper over
    # this. These cover the shapes a real .env throws at it, and the one that
    # would silently corrupt a key (a `#` inside the value must survive).
    _pv = _parse_dotenv_value
    ok("plain KEY=VALUE is read", _pv("UW_API_KEY=abc123", "UW_API_KEY") == "abc123")
    ok("a double-quoted value is unquoted",
       _pv('UW_API_KEY="abc123"', "UW_API_KEY") == "abc123")
    ok("a single-quoted value is unquoted",
       _pv("UW_API_KEY='abc123'", "UW_API_KEY") == "abc123")
    ok("an `export ` prefix is stripped",
       _pv("export UW_API_KEY=abc123", "UW_API_KEY") == "abc123")
    ok("surrounding whitespace around key and value is ignored",
       _pv("  UW_API_KEY = abc123  ", "UW_API_KEY") == "abc123")
    ok("comment and blank lines are skipped, a later key is still found",
       _pv("# a comment\n\nOTHER=1\nUW_API_KEY=abc123\n", "UW_API_KEY") == "abc123")
    ok("a .env without the requested key returns None",
       _pv("OTHER_KEY=value\n", "UW_API_KEY") is None)
    ok("an empty value returns None (falls through to the loud error)",
       _pv("UW_API_KEY=", "UW_API_KEY") is None)
    ok("a `#` inside the value is preserved, not treated as a comment",
       _pv("UW_API_KEY=ab#c123", "UW_API_KEY") == "ab#c123")

    # -- Request headers ------------------------------------------------------
    # Cloudflare 403s urllib's default User-Agent, so losing this header breaks
    # every request on the fallback path with an error that reads like a bad API
    # key. Cheap to assert, offline, and nothing else would catch its removal.
    _h = Client("test-key-not-used").headers
    ok("requests carry a User-Agent", "User-Agent" in _h)
    ok("the User-Agent is not urllib's Cloudflare-blocked default",
       not _h.get("User-Agent", "").lower().startswith("python-urllib"))
    ok("the API key reaches the Authorization header",
       _h["Authorization"] == "Bearer test-key-not-used")
    ok("UW client attribution is sent", _h["UW-CLIENT-API-ID"] == UW_CLIENT_API_ID)

    # -- Every fatal error carries a separate, actionable help field -----------
    # AXI: the remedy is its own `help:` key, never prose inside the body, so an
    # agent can act on it without parsing English out of a paragraph. Drives all
    # five raise sites through a stub transport, so still zero API calls.
    global _raw_get, HTTP_BACKEND
    _saved_raw, _saved_backend = _raw_get, HTTP_BACKEND

    def _stub(status, body="", hdrs=None):
        def f(url, params, headers, timeout=30):
            return status, body, hdrs or {}
        return f

    _quota = {"x-uw-daily-req-count": "5000", "x-uw-token-req-limit": "5000"}
    _cases = [
        ("401 bad key", 401, "", {}, "requests"),
        ("403 Cloudflare via urllib", 403, "cloudflare 1010 access denied", {}, "urllib"),
        ("403 Cloudflare via requests", 403, "cloudflare 1010 access denied", {}, "requests"),
        ("403 sandbox egress", 403,
         "Host not in allowlist: api.unusualwhales.com. Add this host to your "
         "network egress settings to allow access.",
         {"x-deny-reason": "host_not_allowed"}, "requests"),
        ("403 inactive key", 403, "forbidden", {}, "requests"),
        ("404 bad route", 404, "", {}, "requests"),
        ("429 daily quota", 429, "", _quota, "requests"),
    ]
    _helps = {}
    for _label, _st, _body, _hdrs, _backend in _cases:
        _raw_get, HTTP_BACKEND = _stub(_st, _body, _hdrs), _backend
        try:
            Client("k").get("/test")
            _e = None
        except FatalError as _caught:
            _e = _caught
        ok(f"{_label} raises FatalError with an actionable help field",
           _e is not None and bool(_e.help),
           "raised but help is empty" if _e else "did not raise FatalError")
        ok(f"{_label} keeps the remedy out of the message body",
           _e is not None and "fix:" not in str(_e).lower())
        _helps[_label] = _e.help if _e is not None else None
    _raw_get, HTTP_BACKEND = _saved_raw, _saved_backend

    # A CF 403 on requests must NOT say "pip install requests"; we are already on it.
    ok("the Cloudflare remedy adapts to the http backend in use",
       _helps["403 Cloudflare via urllib"] != _helps["403 Cloudflare via requests"])
    ok("the Cloudflare remedy never tells you to install the client you are using",
       "pip install requests" not in (_helps["403 Cloudflare via requests"] or ""))

    # A sandbox egress 403 must be diagnosed as the network, never as the key.
    # The bug this guards: its body has none of the Cloudflare tokens, so before
    # the egress branch existed it fell through to the generic "your key may have
    # expired" -- exactly wrong for a new subscriber in a blocked sandbox.
    _egress = (_helps["403 sandbox egress"] or "").lower()
    ok("the egress 403 remedy points at the network or the user's own machine",
       "egress" in _egress or "own machine" in _egress)
    ok("the egress 403 never blames the user's key",
       "key may have expired" not in _egress)
    ok("the egress 403 is not misread as an inactive key (the branch actually fired)",
       _helps["403 sandbox egress"] != _helps["403 inactive key"])

    # -- Report ---------------------------------------------------------------
    failed = [c for c in checks if not c[1]]
    for name, passed, detail in checks:
        if not passed:
            print(f"FAIL  {name}" + (f"  [{detail}]" if detail else ""))
    print(f"\nselftest: {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("\nDO NOT RUN A SCAN. This file did not transcribe correctly, or a\n"
              "constant was changed. Re-copy it from the Skill and re-run --selftest.")
        return 1
    print("All checks passed with zero API calls. Safe to scan.")
    return 0


# ============================================================================
# CLI
# ============================================================================

def _parse_dotenv_value(text, name):
    """Return the value of `name` from .env-format `text`, or None.

    Intentionally minimal: KEY=VALUE lines, an optional `export ` prefix, `#`
    comments, blank lines, and matched surrounding single/double quotes. It does
    NOT strip inline trailing comments, because an API key may legitimately
    contain `#` and corrupting the key silently is worse than a stray character.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, val = line.partition("=")
        if not sep or key.strip() != name:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        return val or None
    return None


def _load_dotenv_key(name, path=".env"):
    """Read `name` from a `.env` in the working directory. None if absent/unreadable.

    Deliberately does not raise: a missing or malformed `.env` should fall
    through to the existing loud "no API key" error, which already tells the
    user how to supply one. Never prints the value.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    return _parse_dotenv_value(text, name)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="uw_earnings_vol_scan.py",
        description="Scan upcoming earnings for long call calendar spread setups.",
        epilog="Educational use only. Not investment advice.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true",
                      help="verify this file transcribed correctly (0 API calls). Run first.")
    mode.add_argument("--scan", action="store_true",
                      help="tonight's action list: names reporting after today's close "
                           "or before the next session's open. Every result is "
                           "tradeable before today's close.")
    mode.add_argument("--ticker", metavar="XYZ", help="full metrics for one ticker")
    p.add_argument("--days", type=int, default=None, metavar="N",
                   help="widen --scan into an N-day forward window instead of tonight "
                        "only. Research/heads-up use: the options market reprices "
                        "continuously, so a name that passes today can fail by its own "
                        "trade_by date. With --ticker, how far ahead to look for that "
                        "name's report date (default 7).")
    p.add_argument("--rich", action="store_true",
                   help="human-readable ANSI table instead of machine-readable output")
    p.add_argument("--api-key", default=None,
                   help="UW API key. Prefer UW_API_KEY in the environment or a "
                        "UW_API_KEY entry in a local .env file; this flag is a "
                        "plaintext last resort.")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.days is not None and args.days < 1:
        p.error("--days must be at least 1")

    # Resolution precedence: explicit flag > ambient env var > .env in the cwd.
    # (The RECOMMENDATION order in the Skill is the reverse by safety: env, then
    # .env, then --api-key as a last resort, because --api-key is plaintext.)
    api_key = (args.api_key
               or os.environ.get("UW_API_KEY")
               or _load_dotenv_key("UW_API_KEY"))
    if not api_key:
        print("error: no API key\n"
              "help: provide your UW API key one of three ways, most secure first:\n"
              "  (1) set UW_API_KEY in your environment;\n"
              "  (2) create a `.env` file in this directory containing "
              "`UW_API_KEY=your-key`\n"
              "      (add `.env` to .gitignore and never commit it);\n"
              "  (3) pass --api-key (least preferred: it puts the key in plaintext).\n"
              "  Find your key at https://unusualwhales.com/settings.")
        return 1

    client = Client(api_key)
    today = today_eastern()

    try:
        if args.ticker:
            ticker = args.ticker.strip().upper()
            # The trade-by date needs the report date, and the only documented
            # source for it is the earnings calendar, so this walks a window.
            # A name that is not on it gets no trade-by rather than a made-up one.
            # --ticker keeps a 7-day default because "is ABC coming up at all?" is
            # a different question from --scan's "what do I trade tonight?".
            lookahead = args.days if args.days is not None else 7
            meta = fetch_earnings(client, window_buckets(today, lookahead)).get(
                ticker, {"ticker": ticker, "report_date": None, "report_time": None,
                         "expected_move_perc": None})
            try:
                r = score_ticker(client, meta, today=today)
            except SkipTicker as e:
                print(f"ticker: {ticker}\nverdict: skipped\nreason: {e}")
                return 0  # a name we cannot score is a valid answer, not a failure
            if args.rich:
                # `lookahead`, not None: --ticker is a lookup, not tonight's list,
                # so its trade_by is real information and must stay on screen.
                print_rich([r], [], 1, lookahead, today)
            else:
                print_ticker(r)
            return 0

        results, skipped, universe_n = scan(client, args.days, today=today)
        if args.rich:
            print_rich(results, skipped, universe_n, args.days, today)
        else:
            print_scan(results, skipped, universe_n, args.days, today)
        return 0

    except FatalError as e:
        # stdout, not stderr: an agent only reads stdout, so an error it cannot
        # see is an error it cannot act on. `help:` is a sibling key, never
        # folded into the body, so the remedy stays machine-readable.
        print(f"error: {e}")
        if e.help:
            print(f"help: {e.help}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
```
