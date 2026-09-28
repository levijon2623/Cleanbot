"""
market_calendar.py
==================
NYSE full-day holidays and trading-day helpers, with no dependencies.

Split out of uw_options_data_lake on 2026-09-28. The live bot needs the
calendar (unusual_whales_client's market-hours gate, live_state's weekly GEX
page) but runs on the cloud box without the research lake -- which imports
polars and httpx and is not deployed there. Importing the calendar from the
lake silently failed on the box: the UW gate fell back to "weekdays only" and
every weekly-GEX snapshot raised ModuleNotFoundError. uw_options_data_lake
re-exports these names, so research scripts are unchanged.

Early closes (13:00) are NOT in this table -- only full-day closures.
"""
from __future__ import annotations

from datetime import date, timedelta

# Full-day NYSE closures, from https://www.nyse.com/markets/hours-calendars.
_MARKET_HOLIDAYS = {
    2023: "01-02 01-16 02-20 04-07 05-29 06-19 07-04 09-04 11-23 12-25",
    2024: "01-01 01-15 02-19 03-29 05-27 06-19 07-04 09-02 11-28 12-25",
    2025: "01-01 01-09 01-20 02-17 04-18 05-26 06-19 07-04 09-01 11-27 12-25",
    2026: "01-01 01-19 02-16 04-03 05-25 06-19 07-03 09-07 11-26 12-25",
    2027: "01-01 01-18 02-15 03-26 05-31 06-18 07-05 09-06 11-25 12-24",
    2028: "01-17 02-21 04-14 05-29 06-19 07-04 09-04 11-23 12-25",
}

MARKET_HOLIDAYS: dict[int, frozenset[date]] = {
    y: frozenset(date(y, int(md[:2]), int(md[3:])) for md in mds.split())
    for y, mds in _MARKET_HOLIDAYS.items()
}


def market_holidays(year: int) -> frozenset[date]:
    try:
        return MARKET_HOLIDAYS[year]
    except KeyError:
        lo, hi = min(MARKET_HOLIDAYS), max(MARKET_HOLIDAYS)
        raise ValueError(
            f"the NYSE holiday table does not cover {year}; it runs {lo}-{hi}. "
            f"Add {year} to market_calendar._MARKET_HOLIDAYS from "
            f"https://www.nyse.com/markets/hours-calendars.") from None


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in market_holidays(d.year)


def previous_trading_day(d: date) -> date:
    for _ in range(15):
        if is_trading_day(d): return d
        d -= timedelta(days=1)
    raise ValueError(f"no trading day found on or before {d}")


def next_trading_day(d: date) -> date:
    for _ in range(15):
        d += timedelta(days=1)
        if is_trading_day(d): return d
    raise ValueError(f"no trading day found after {d}")


def trading_days(start: date, end: date) -> list[date]:
    if end < start: return []
    days: list[date] = []
    d = start
    while d <= end:
        if is_trading_day(d): days.append(d)
        d += timedelta(days=1)
    return days
