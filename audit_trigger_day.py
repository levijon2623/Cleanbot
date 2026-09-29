"""
audit_trigger_day.py
====================
Which rule triggers SHOULD the bot have taken on a given day, and did it?

    python audit_trigger_day.py 2026-09-28                 # every enabled rule
    python audit_trigger_day.py 2026-09-24 2026-09-25 --rule "NVDA LOWVOL PUT"
    python audit_trigger_day.py 2026-09-28 --all           # list every crossover
    python audit_trigger_day.py 2026-09-28 --journal dump.txt   # off the box

🚨 WHY THIS EXISTS (2026-09-29)
    No paper trade for three sessions after the move to the cloud, and the bot
    could not say why: after a crossover, run_brain refuses an entry through a
    chain of SILENT `continue`s -- regime, hours, the 09:35 window, the macro
    morning, EMA/DMI confirmation, the flow gate. Only a missing contract, a
    wide spread and the sizing warning ever print. So "were triggers missed?"
    could not be read from the log; it had to be rebuilt from minute data.
    That rebuild found the answer in minutes: zero triggers cleared the
    correct gate on any of the five rule-days, and 10 NVDA triggers on 09-25
    that cleared the (then inflated) gate had nothing to buy, because an
    overnight restart had left the options basket empty.

ONE IMPLEMENTATION, NOT A COPY (METHODOLOGY 1)
    Every gate is the BOT'S OWN METHOD, called on an engine object built
    without starting anything and fed data as of the audited day:
      crossovers   FlowMomentumTracker on a patched clock (as
                   rebuild_flow_trigger_log.py does)
      regime       _poll_session_regimes (trend / volume / GEX / amp)
      rule match   _match_rule, once without and once with its amt_open gate,
                   so a refusal names which of the two refused
      AMT open     _seed_amt_profiles + _amt_open_state (prior session's VA vs
                   the 09:30 open -- fixed live 2026-09-29; before that the bot
                   classified on whatever spot it had at 00:01 or after a
                   restart, so live AMT verdicts before then can differ)
      confirms     _ema_confirm_ok, _dmi_confirm_ok
      threshold    _live_flow_threshold on the trigger log, days BEFORE the one
                   audited (what the bot had loaded that morning)
    The gate ORDER is run_brain's: 15:00 cutoff, regime/rule, hours, 09:35,
    macro morning, EMA, DMI, threshold, |cum flow| >= threshold.

WHAT IT ASSUMES
    The session state a bot started BEFORE THE OPEN would have held: GEX and
    daily bars as of the previous session, 1-minute seeds ending the evening
    before. A bot restarted mid-session re-polls and can hold a different GEX
    row (today's). During the session the EMA/DMI history grows by one bar
    per minute with high = low = close, exactly as _record_price_minute
    appends live.
    Flow: the day's net-prem-ticks, today-only (UW, by date; or
    historical/NETPREM{T}.parquet when it already holds the day).
    Before 2026-09-28 the LIVE bot ran on inflated cumulative flow (see
    rebuild_flow_trigger_log.py); verdicts here are the CORRECT gate.

WHAT IT CANNOT SEE
    Anything after the gate needs the option book at that second: the
    contract lookup, the $0.50 floor, the spread guard, the allocator. The
    first and third print in the journal and are listed for comparison; the
    floor and the allocator refuse silently.
    One position per ticker: a PASS after an earlier PASS is marked as such,
    because whether the bot was still holding depends on how the first exited.

UPTIME, FROM THE JOURNAL
    On the box, `journalctl -u cleanbot` gives when the engine was active and
    when each ticker's basket was built. A crossover is marked "bot down" if
    it landed in a restart gap (or in the two closed bars the tracker needs
    after one), and "no basket" if the contracts had not been subscribed yet.
    Off the box, pass a dump with --journal, or the column is left out.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, deque

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
with contextlib.redirect_stdout(io.StringIO()):
    import bot_runner as B                                  # noqa: E402
import config                                               # noqa: E402
import market_calendar as MC                                # noqa: E402
import unusual_whales_client as U                           # noqa: E402

NY = B.MARKET_TZ
ENGINE = B.FlowExecutionEngine
FIRST_MOD, LAST_MOD = 9 * 60 + 30, 16 * 60 + 4     # minutes fed, as the rebuild


def at(day, mod, sec=0):
    return dt.datetime(day.year, day.month, day.day, mod // 60, mod % 60, sec, tzinfo=NY)


# ------------------------------------------------------------------ clock
class _Clock(dt.datetime):
    """unusual_whales_client does `from datetime import datetime` and asks
    datetime.now() for "today" -- this answers with the audited moment."""
    fixed = None

    @classmethod
    def now(cls, tz=None):
        return cls.fixed.astimezone(tz) if tz else cls.fixed


@contextlib.contextmanager
def uw_clock(moment):
    real = U.datetime
    _Clock.fixed = moment
    U.datetime = _Clock
    try:
        yield
    finally:
        U.datetime = real


class DatedUW:
    """The UW client, answering as of the evening before `day` -- i.e. what a
    bot started before that day's open would have fetched."""

    def __init__(self, uw, day):
        self.uw, self.day = uw, day
        # previous_trading_day is ON OR BEFORE its argument -- step back first,
        # or "the evening before" is the audited day's own evening (lookahead)
        prev = MC.previous_trading_day(day - dt.timedelta(days=1))
        self.eve = dt.datetime(prev.year, prev.month, prev.day, 20, 0, tzinfo=NY)

    def get_daily_bars(self, ticker, limit=90):
        rows = [b for b in self.uw.get_daily_bars(ticker, limit=1000)
                if b["date"] and b["date"] <= self.day.isoformat()]
        return rows[-int(limit):]

    def get_intraday_bars(self, ticker, lookback_days=3, ohlcv=False, end_date=None):
        if end_date is not None:                  # the caller named the day itself
            return self.uw.get_intraday_bars(ticker, lookback_days=lookback_days,
                                             ohlcv=ohlcv, end_date=end_date)
        with uw_clock(self.eve):
            return self.uw.get_intraday_bars(ticker, lookback_days=lookback_days,
                                             ohlcv=ohlcv)

    def get_daily_gex_regime(self, ticker, prior_day=False):
        """Last /greek-exposure row BEFORE the audited day."""
        import requests
        r = requests.get(f"{self.uw.base_url}/api/stock/{ticker.upper()}/greek-exposure",
                         headers=self.uw.headers, timeout=15)
        rows = sorted((x for x in r.json().get("data", [])
                       if str(x.get("date") or "")[:10] < self.day.isoformat()),
                      key=lambda x: x.get("date") or "")
        if not rows:
            return None
        g = U._net_gex_from_row(rows[-1])
        reg = "UNKNOWN" if not g else ("POSITIVE" if g > 0 else "NEGATIVE")
        return {"regime": reg, "net_gex": g or 0.0, "as_of": rows[-1].get("date")}

    def get_spot_gex(self, ticker):
        return None           # vol_overlay only resizes the target; not a gate


# ------------------------------------------------------------------ data
def day_flow(uw, tk, day):
    """[(minute datetime ET, net premium)] for 09:30..16:04, today-only."""
    p = os.path.join("historical", f"NETPREM{tk}.parquet")
    if os.path.exists(p):
        import pandas as pd
        df = pd.read_parquet(p, columns=["date", "minute_et", "net_premium"])
        g = df[df["date"].astype(str) == day.isoformat()]
        if len(g):
            t = pd.to_datetime(g["minute_et"]).dt.tz_convert(NY)
            rows = sorted(zip(t, g["net_premium"].astype(float)))
            return [(m.to_pydatetime(), v) for m, v in rows
                    if FIRST_MOD <= m.hour * 60 + m.minute <= LAST_MOD], "parquet"
    import requests
    r = requests.get(f"{uw.base_url}/api/stock/{tk}/net-prem-ticks", headers=uw.headers,
                     params={"date": day.isoformat()}, timeout=60)
    out = []
    for x in r.json().get("data", []):
        m = dt.datetime.fromisoformat(x["tape_time"].replace("Z", "+00:00")).astimezone(NY)
        if m.date() != day:
            # 🚨 UW answers 200 with UNFILTERED data when it ignores a param
            # (memory: cleanbot-uw-ignored-params) -- check the rows, not the status
            raise SystemExit(f"UW returned {m.date()} rows for {tk} {day} -- date ignored")
        if FIRST_MOD <= m.hour * 60 + m.minute <= LAST_MOD:
            out.append((m, float(x["net_call_premium"]) - float(x["net_put_premium"])))
    return sorted(out), "UW"


def crossovers(series):
    """[(seen, direction, |cum|, cum)] -- seen = the minute the bot evaluates the
    closed crossover bar, which is when its gates run."""
    tr, out, cum, run = B.FlowMomentumTracker(), [], [], 0.0
    for _, v in series:
        run += v
        cum.append(run)
    real = B.time.time
    try:
        for i, (m, _) in enumerate(series):
            B.time.time = lambda m=m: m.timestamp() + 30.0
            sig = tr.update_and_check("X", cum[i])
            if sig in ("LONG", "SHORT"):
                out.append((m, "CALL" if sig == "LONG" else "PUT", abs(cum[i - 1]), cum[i - 1]))
    finally:
        B.time.time = real
    return out, (cum[-1] if cum else 0.0)


def trigger_hist(path, day):
    hist = {}
    for line in open(path, encoding="utf-8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e["date"] >= day.isoformat() or dt.date.fromisoformat(e["date"]).weekday() >= 5:
            continue
        hist.setdefault(e["ticker"], deque(maxlen=ENGINE._FLOW_HIST_MAXLEN)).append(
            (e["date"], float(e["abs_flow"])))
    return hist


# ------------------------------------------------------------------ journal
_TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)")


def journal_lines(day, path):
    if path:
        return open(path, encoding="utf-8", errors="replace").read().splitlines()
    try:
        since = (day - dt.timedelta(days=4)).isoformat()
        r = subprocess.run(["journalctl", "-u", "cleanbot", "-o", "short-iso", "--no-pager",
                            "--since", f"{since} 00:00", "--until", f"{day} 23:59:59"],
                           capture_output=True, text=True, timeout=120)
        return r.stdout.splitlines() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


_PID = re.compile(r"cleanbot\[(\d+)\]")


def parse_journal(lines, day):
    """Engine-active intervals, the basket times of the process behind each,
    and the bot's own entry-side messages for the day. Needs `-o short-iso`.

    Keyed by PROCESS ID: the basket is built during startup, BEFORE the
    "ENGINE ACTIVE" line, so a time window starting at that line would miss
    it -- only the pid says which process a basket belongs to."""
    ivals, cur, baskets, said = [], None, defaultdict(list), defaultdict(list)
    for ln in lines:
        m = _TS.match(ln)
        if not m:
            continue
        t = dt.datetime.fromisoformat(m.group(1)).replace(tzinfo=NY)
        mp = _PID.search(ln)
        pid = mp.group(1) if mp else None
        if "ZERO-LATENCY ENGINE ACTIVE" in ln:
            if cur:
                ivals.append((cur[0], t, cur[1]))
            cur = (t, pid)
        elif "Stopping cleanbot" in ln or "Stopped cleanbot" in ln:
            if cur:
                ivals.append((cur[0], t, cur[1]))
                cur = None
        mb = re.search(r"├─ ([A-Z]+): Subscribing to \d+ Plausible", ln)
        if mb and pid:
            baskets[(pid, mb.group(1))].append(t)
        if t.date() == day:
            for pat in (r"WHALE SWEEP: (?:CALL|PUT) on ([A-Z]+)",
                        r"Valid liquid contracts not found in memory for ([A-Z]+)",
                        r"\[SPREAD\] ([A-Z]+)", r"\[RISK\] ([A-Z]+)",
                        r"([A-Z]+) entry abandoned"):
                mm = re.search(pat, ln)
                if mm:
                    said[mm.group(1)].append(f"{t:%H:%M} {ln.split(': ', 1)[-1].strip()[:90]}")
    if cur:
        ivals.append((cur[0], at(day, 23 * 60 + 59), cur[1]))
    return ivals, baskets, said


def uptime(seen, tk, ivals, baskets):
    for a, b, pid in ivals:
        # the tracker needs two CLOSED bars after a (re)start to evaluate one
        if a + dt.timedelta(minutes=2) <= seen < b:
            if not any(t <= seen for t in baskets.get((pid, tk), [])):
                return "bot up, NO BASKET"
            return "bot up"
        if a <= seen < b:
            return "bot warming up"
    return "BOT DOWN"


# ------------------------------------------------------------------ audit
def build_engine(uw, day, log_path):
    """A FlowExecutionEngine with nothing started: only the attributes its gate
    methods read, loaded as of the morning of `day`."""
    e = ENGINE.__new__(ENGINE)
    e.uw = DatedUW(uw, day)
    e.session_gex, e.session_regime, e.session_dex_pct = {}, {}, {}
    rules = [r for r in B.RULES if r.get("enabled", True)]
    e._ema_confirm_tickers = {r["ticker"] for r in rules if r.get("ema_confirm")}
    e._dmi_confirm_tickers = {r["ticker"] for r in rules if r.get("dmi_confirm")}
    e._amt_open_rules = {r["ticker"] for r in rules if r.get("amt_open")}
    e._dex_pct_tickers = {r["ticker"] for r in rules if r.get("dex_pct_max") is not None}
    e.px_hist = defaultdict(lambda: deque(maxlen=1560))
    e.bar_hist = defaultdict(lambda: deque(maxlen=1560))
    e._ema_warned, e._dmi_warned, e._macro_am_warned, e._thr_warned = set(), set(), set(), set()
    e._prev_va, e._amt_open = {}, {}
    e.flow_thresholds = e._load_flow_thresholds()
    e.flow_trigger_hist = trigger_hist(log_path, day)
    tickers = sorted({r["ticker"] for r in rules})
    for tk in tickers:
        e.session_gex[tk] = e.uw.get_daily_gex_regime(tk)
    e._poll_session_regimes(tickers)
    e._seed_price_history(tickers)
    e._seed_amt_profiles(tickers)
    return e


def gate(e, rule, day, seen, cum_abs):
    """The first run_brain gate that refuses, or None. Mirrors run_brain's order."""
    tk, d = rule["ticker"], rule["direction"]
    if seen.hour >= 15:
        return "after 15:00"
    state = e.session_regime.get(tk)
    if not state:
        return "no session regime"
    bare = {k: v for k, v in rule.items() if k not in ("amt_open", "dex_pct_max")}
    real = B.RULES
    try:
        B.RULES = [bare]
        if not e._match_rule(tk, d, state):
            want = rule.get("regime") or f"amp >= {rule.get('amp_min')}"
            return (f"regime (needs {want}; trend {state.get('trend')} vol "
                    f"{state.get('volume')} gex {state.get('gex')} amp {state.get('amp')})")
        B.RULES = [rule]
        tr = e._match_rule(tk, d, state)
        if not tr:
            return f"amt_open (opened {e._amt_open_state(tk)}, rule {rule.get('amt_open')})"
    finally:
        B.RULES = real
    if seen.hour not in tr.get("hours", []):
        return f"hour {seen.hour} not in {tr.get('hours')}"
    if seen.hour * 60 + seen.minute < B.NO_ENTRY_BEFORE_MOD:
        return "before 09:35"
    if tr.get("skip_macro_am") and B.is_macro_am_day(day):
        return "macro morning"
    if tr.get("ema_confirm") and not e._ema_confirm_ok(tk, d, tr):
        return "ema_confirm"
    if tr.get("dmi_confirm") and not e._dmi_confirm_ok(tk, d, tr):
        return "dmi_confirm"
    thr = e._live_flow_threshold(tk, tr)
    if thr is None:
        return "no threshold"
    if cum_abs < thr:
        return f"flow {cum_abs / 1e6:.1f}M < {thr / 1e6:.1f}M"
    return None


def audit(day, rules, uw, log_path, jpath, show_all):
    print(f"\n{'#' * 78}\n#  {day}  ({day:%A})\n{'#' * 78}")
    real_now = B.market_now
    B.market_now = lambda: at(day, 9 * 60 + 30)
    try:
        e = build_engine(uw, day, log_path)
    finally:
        B.market_now = real_now
    jl = journal_lines(day, jpath)
    ivals, baskets, said = parse_journal(jl, day) if jl else ([], {}, {})
    if jl is None:
        print("  (no journal -- uptime and the bot's own messages are not checked)")
    else:
        up = [f"{max(a, at(day, 0)):%H:%M}-{min(b, at(day, 23 * 60 + 59)):%H:%M}"
              for a, b, _ in ivals if b.date() >= day and a.date() <= day]
        print(f"  engine active: {', '.join(up) or 'NEVER'}")

    for rule in rules:
        tk = rule["ticker"]
        series, src = day_flow(uw, tk, day)
        if not series:
            print(f"\n=== {rule['name']}: no flow data for {day}")
            continue
        xs, total = crossovers(series)
        # The day's minutes join the EMA/DMI history as _record_price_minute
        # appends them live: at second 0 of minute k, keyed k, priced at the
        # spot then -- i.e. bar k-1's close -- with high = low = close.
        closes = sorted((dt.datetime.fromisoformat(b["minute_et"]).replace(tzinfo=NY)
                         + dt.timedelta(minutes=1), b["close"])
                        for b in uw.get_intraday_bars(tk, 1, end_date=day)
                        if b["minute_et"][:10] == day.isoformat())
        refused, passes, first_pass = defaultdict(int), [], None
        rows, near = [], []
        j = 0
        try:
            for seen, d, cabs, c in xs:
                while j < len(closes) and closes[j][0] <= seen:
                    k, px = closes[j]
                    mk = k.strftime("%Y-%m-%dT%H:%M")
                    if tk in e._ema_confirm_tickers:
                        e.px_hist[tk].append((mk, px))
                    if tk in e._dmi_confirm_tickers:
                        e.bar_hist[tk].append((mk, px, px, px))
                    j += 1
                if d != rule["direction"]:
                    continue
                B.market_now = lambda seen=seen: seen
                with contextlib.redirect_stdout(io.StringIO()):
                    why = gate(e, rule, day, seen, cabs)
                st = uptime(seen, tk, ivals, baskets) if jl else ""
                if why is None:
                    tag = "PASS" if first_pass is None else "PASS (after an earlier pass)"
                    first_pass = first_pass or seen
                    passes.append(seen)
                    rows.append(f"  {seen:%H:%M}  cum {c / 1e6:+8.1f}M  {tag:<28} {st}")
                else:
                    key = ("flow below threshold" if why.startswith("flow")
                           else why.split(" (")[0].split(" not in")[0])
                    refused[key] += 1
                    line = f"  {seen:%H:%M}  cum {c / 1e6:+8.1f}M  refused: {why:<40} {st}"
                    if show_all:
                        rows.append(line)
                    elif why.startswith("flow"):
                        near.append((cabs, seen, line))
            if tk in e._amt_open_rules and tk not in e._amt_open:
                B.market_now = lambda: at(day, 9 * 60 + 35)   # for the report line
                with contextlib.redirect_stdout(io.StringIO()):
                    e._amt_open_state(tk)
        finally:
            B.market_now = real_now
        n_dir = sum(1 for x in xs if x[1] == rule["direction"])
        print(f"\n=== {rule['name']}   flow {src}, day total {total / 1e6:+.1f}M, "
              f"{len(xs)} crossovers ({n_dir} {rule['direction']})")
        st = e.session_regime.get(tk) or {}
        print(f"    regime trend {st.get('trend')}  vol {st.get('volume')}  gex {st.get('gex')}"
              f"  amp {st.get('amp')}" + (f"  opened {e._amt_open.get(tk)}"
                                         if tk in e._amt_open_rules else ""))
        print(f"    refused: " + (", ".join(f"{k} x{v}" for k, v in
                                           sorted(refused.items(), key=lambda kv: -kv[1])) or "-"))
        if not show_all and near:
            # the closest calls: the five flow checks nearest their threshold
            keep = sorted(near, key=lambda x: -x[0])[:5]
            rows += [ln for _, _, ln in keep]
            rows.sort(key=lambda ln: ln[2:7])
        for r in rows:
            print(r)
        print(f"    => {len(passes)} PASS" + (f", first {first_pass:%H:%M}" if passes else ""))
        for s in said.get(tk, []):
            print(f"    bot said: {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("days", nargs="+", help="YYYY-MM-DD")
    ap.add_argument("--rule", action="append", help="rule name (repeatable); default all enabled")
    ap.add_argument("--log", default=os.getenv("FLOW_TRIGGER_LOG", "flow_trigger_log.jsonl"),
                    help="crossover history the threshold is computed from")
    ap.add_argument("--journal", help="journalctl -u cleanbot -o short-iso dump (off the box)")
    ap.add_argument("--all", action="store_true", help="list every crossover, not just flow checks")
    a = ap.parse_args()

    rules = [r for r in B.RULES if r.get("enabled", True)]
    if a.rule:
        rules = [r for r in rules if r["name"] in a.rule]
        if not rules:
            sys.exit(f"no enabled rule named {a.rule}")
    uw = U.UnusualWhalesClient(os.environ["UW_API_KEY"])     # REST only; no socket
    for d in a.days:
        day = dt.date.fromisoformat(d)
        if not MC.is_trading_day(day):
            print(f"\n{day}: not a trading day")
            continue
        audit(day, rules, uw, a.log, a.journal, a.all)
    print("\nNot replayed (needs the option book at that second): contract lookup, "
          "$0.50 floor, spread guard, allocator.\nThe journal shows the lookup and "
          "the spread guard as 'bot said'; the floor and the allocator refuse silently.")


if __name__ == "__main__":
    main()
