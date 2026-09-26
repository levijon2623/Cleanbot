"""US macro-release calendar -- the scheduled events that move index-ETF vol.

Shared by check_event_days.py (backtest) and bot_runner.py (live) so the two
never drift.  REFRESH ~ANNUALLY:
  * CPI  -- BLS, 8:30am ET, ~2nd week (dates vary; late-2025 gov shutdown moved a
    couple). https://www.bls.gov/schedule/news_release/cpi.htm
  * PCE  -- BEA Personal Income & Outlays, 8:30am ET, ~month-end.
  * NFP  -- BLS Employment Situation, first Friday, 8:30am ET (computed, not
    listed -- occasionally slips to the 2nd Friday; rare).
  * FOMC -- rate decision, 2:00pm ET (published years ahead).

check_event_days.py finding (session 16, 2026-09-06): FOMC's 2pm IV crush is
real but the bot navigates it fine (net positive, tiny sample). The 8:30am
releases are the drag -- and specifically they turn IWM/QQQ HIVOL CALL into a
-65% / win-0.0 rule in the 11:00-14:00 window (the tradeable move already
happened at 8:30; the late-morning flow trigger is buying exhausted momentum).
Every other rule is fine or better on those days. Hence `skip_macro_am` on
just those two rules.
"""
from __future__ import annotations

import datetime as _dt

# --- CPI report release dates (8:30am ET) --------------------------------------
CPI = frozenset({
    "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
    "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11", "2025-12-18",
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12",
    "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14",
    "2026-11-10", "2026-12-10",
})

# --- PCE (Personal Income & Outlays) release dates (8:30am ET) -----------------
PCE = frozenset({
    "2024-08-30", "2024-09-27", "2024-10-31", "2024-11-27", "2024-12-20",
    "2025-01-31", "2025-02-28", "2025-03-28", "2025-04-30", "2025-05-30",
    "2025-06-27", "2025-07-31", "2025-08-29", "2025-09-26", "2025-12-19",
    "2026-01-30", "2026-02-27", "2026-03-27", "2026-04-30", "2026-05-29",
    "2026-06-26", "2026-07-31",
})

# --- FOMC rate-decision days (announcement 2:00pm ET) -------------------------
FOMC = frozenset({
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
})


def _iso(d) -> str:
    return d.isoformat() if isinstance(d, (_dt.date, _dt.datetime)) else str(d)[:10]


def is_nfp_day(d) -> bool:
    """first Friday of the month (BLS Employment Situation, 8:30am ET)."""
    d = d if isinstance(d, _dt.date) else _dt.date.fromisoformat(_iso(d))
    return d.weekday() == 4 and d.day <= 7


def is_macro_am_day(d) -> bool:
    """True on CPI / PCE / NFP mornings -- scheduled 8:30am ET releases that gap
    the index ETFs and make the 11:00-14:00 HIVOL flow triggers a fade."""
    iso = _iso(d)
    return iso in CPI or iso in PCE or is_nfp_day(d)


def is_fomc_day(d) -> bool:
    return _iso(d) in FOMC


def nfp_days_in(dates) -> set:
    """first Friday of every month spanned by `dates` (an iterable of date)."""
    out = set()
    for ym in sorted({(d.year, d.month) for d in dates}):
        first = _dt.date(ym[0], ym[1], 1)
        out.add((first + _dt.timedelta(days=(4 - first.weekday()) % 7)).isoformat())
    return out
