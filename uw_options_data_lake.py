# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars>=1.0.0",
#     "httpx>=0.27.0",
#     "tzdata>=2024.1",
#     "python-dotenv",
# ]
# ///

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import polars as pl
from dotenv import load_dotenv

# =====================================================================
# CONSTANTS
# =====================================================================
API_BASE = "https://api.unusualwhales.com/api"
FULL_TAPE_PATH = "/option-trades/full-tape"
UW_CLIENT_API_ID = "100003" 
USER_AGENT = "uw-options-data-lake/1.0 (+https://unusualwhales.com)"
HTTP_TIMEOUT = 60.0
STREAM_CHUNK = 1 << 20
DOWNLOAD_MAX_RETRIES = 3
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
INFER_SCHEMA_ROWS = 200_000

TIMESTAMP_COLUMNS: tuple[str, ...] = ("executed_at", "created_at")
BRONZE_TIMESTAMP_DTYPE = pl.Datetime(time_unit="us", time_zone="UTC")

EXPECTED_COLUMNS: tuple[str, ...] = (
    "id", "underlying_symbol", "executed_at", "nbbo_bid", "nbbo_ask", "size",
    "price", "option_chain_id", "alert_score", "created_at", "report_flags",
    "tags", "expiry", "option_type", "open_interest", "strike", "premium",
    "aggregated_trade_id", "volume", "underlying_price", "ewma_nbbo_ask",
    "ewma_nbbo_bid", "implied_volatility", "delta", "theta", "gamma", "vega",
    "rho", "theo", "upstream_condition_detail", "market_center_locate",
    "canceled", "trade_id", "exchange", "ask_vol", "bid_vol", "no_side_vol",
    "mid_vol", "multi_vol", "stock_multi_vol",
)

ZIP_BYTES_PER_DAY = 2_600_000_000
CSV_BYTES_PER_DAY = 7_500_000_000
PARQUET_BYTES_PER_DAY_ESTIMATE = 1_800_000_000
SILVER_BYTES_PER_DAY_ESTIMATE = 350_000_000

DEFAULT_LAKE = Path("lake")
BRONZE_SUBDIR = "bronze"
BRONZE_DATASET = "full-tape"
WORK_SUBDIR = "_work"
SILVER_SUBDIR = "silver"
SILVER_BARS_DATASET = "option-contracts-1m"
SILVER_BARS_FILE = "bars.parquet"
SCREENS_SUBDIR = "screens"

SILVER_COLUMNS: tuple[str, ...] = (
    "option_chain_id", "underlying_symbol", "security_type", "option_type",
    "strike", "expiry", "minute_utc", "minute_et", "open", "high", "low",
    "close", "volume", "trade_count", "premium", "vwap", "ask_volume",
    "bid_volume", "mid_volume", "no_side_volume", "multi_volume", "bid_close",
    "ask_close", "underlying_open", "underlying_close", "iv_close", "delta_close",
    "gamma_close", "theta_close", "vega_close", "rho_close", "theo_close",
    "open_interest",
)

SILVER_SOURCE_COLUMNS: tuple[str, ...] = (
    "option_chain_id", "underlying_symbol", "executed_at", "price", "size",
    "premium", "canceled", "volume", "nbbo_bid", "nbbo_ask", "underlying_price",
    "implied_volatility", "delta", "gamma", "theta", "vega", "rho", "theo",
    "open_interest", "option_type", "strike", "expiry", "tags",
    "ask_vol", "bid_vol", "mid_vol", "no_side_vol", "multi_vol",
)

SILVER_SIDE_COUNTERS: tuple[str, ...] = (
    "ask_vol", "bid_vol", "mid_vol", "no_side_vol", "multi_vol",
)

SILVER_SIDE_VOLUMES: tuple[str, ...] = (
    "ask_volume", "bid_volume", "mid_volume", "no_side_volume", "multi_volume",
)

SILVER_MINUTE_UTC_DTYPE = pl.Datetime(time_unit="us", time_zone="UTC")
SILVER_SECURITY_TYPES: tuple[str, ...] = ("equity", "etf", "index")
CANCELED_DOMAIN: tuple[str, ...] = ("t", "f")

EASTERN_TZ = "America/New_York"
EASTERN = ZoneInfo(EASTERN_TZ)
SILVER_MINUTE_ET_DTYPE = pl.Datetime(time_unit="us", time_zone=EASTERN_TZ)
QUOTA_RESET_HOUR_ET = 20
DEV_HISTORIC_EMAIL = "dev@unusualwhales.com"

# --- GEX / greek-exposure backfill -----------------------------------
# historical/GEX{T}.parquet is what directional_flow_backtester.py (--hist) and
# the HyperExposureClient simulators read; the 1-min spot series lands in the
# silver tree next to the option bars.
DEFAULT_HISTORICAL = Path("historical")
SPOT_GEX_DATASET = "spot-exposures-1m"

# config.WATCHLIST keys (the flow-bot universe). Override with --tickers.
GEX_TICKERS: tuple[str, ...] = ("SPY", "NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL")

REST_RETRY_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})
REST_SLEEP_SECONDS = 0.25
REST_BACKOFF_SECONDS = 1.5

# (call_field, put_field, net_output) for /greek-exposure daily rows.
GREEK_EXPOSURE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("call_gamma", "put_gamma", "net_gex"),
    ("call_delta", "put_delta", "net_dex"),
    ("call_charm", "put_charm", "net_charm"),
    ("call_vanna", "put_vanna", "net_vanna"),
)
GREEK_EXPOSURE_NUMERIC: tuple[str, ...] = tuple(
    c for a, b, _ in GREEK_EXPOSURE_PAIRS for c in (a, b)
)

SPOT_GEX_NUMERIC: tuple[str, ...] = (
    "gamma_per_one_percent_move_oi", "gamma_per_one_percent_move_vol",
    "gamma_per_one_percent_move_dir", "charm_per_one_percent_move_oi",
    "charm_per_one_percent_move_vol", "charm_per_one_percent_move_dir",
    "vanna_per_one_percent_move_oi", "vanna_per_one_percent_move_vol",
    "vanna_per_one_percent_move_dir", "price",
)

# /stock/{t}/ohlc/{candle} -> historical/{T}.parquet (the backtesters' underlying
# OHLC input). One file per ticker spanning a growing date range; incremental.
OHLC_NUMERIC: tuple[str, ...] = ("open", "high", "low", "close", "volume", "total_volume")
DEFAULT_OHLC_CANDLE = "1m"

# /stock/{t}/net-prem-ticks -> historical/NETPREM{T}.parquet. Per-minute options
# aggressor flow aggregated server-side by UW: call/put volume split into
# ask-side vs bid-side (the raw bid/ask imbalance), net premium ($, the exact
# quantity the LIVE bot consumes as net_call_premium - net_put_premium), and
# net_delta (the share-equivalent dealer hedge). One incremental file per ticker,
# same shape as the OHLC backfill.
#
# 🚨 HISTORY FLOOR: A ROLLING WINDOW OF ~35 MONTHS. IT MOVES A DAY A DAY.
#   2026-09-12  floor measured 2023-10-12 (binary search)
#   2026-09-26  floor measured 2023-10-26 -- the trades-core backfill got
#               HTTP 403 on every day 2023-10-12..10-25 and 200 from 10-26
# Fourteen days later, fourteen days lost: ~1,066 calendar days back from today.
# /option-trades/full-tape and net-prem-ticks shared the floor when first
# measured; assume both roll.
#
# This note used to say the opposite -- "a FIXED floor, not a rolling window,
# nothing already downloaded is expiring" -- on the strength of one measurement,
# which cannot distinguish a fixed floor from a slow-moving one. Two
# measurements a fortnight apart can. The consequence is concrete: data ALREADY
# on disk is safe, but any backfill not yet run loses its oldest day every day
# it waits. Plan backfills from the oldest end.
#
# Also permanently unavailable, vendor-side: 2025-04-04 (zip fails CRC) and
# 2025-09-29 (zip has a bad header). Both re-downloaded 2026-09-26 with the same
# error, and both are absent from option-contracts-1m for the same reason.
NETPREM_NUMERIC: tuple[str, ...] = ("net_call_premium", "net_put_premium", "net_delta")
NETPREM_PREFIX = "NETPREM"

# =====================================================================
# US MARKET CALENDAR
# =====================================================================
# The table and helpers live in market_calendar (no dependencies) so the live
# bot can use them on the box, where this module is not deployed. Re-exported
# here unchanged; market_holidays keeps this module's FatalError message.
import market_calendar as _cal
from market_calendar import (MARKET_HOLIDAYS, is_trading_day, previous_trading_day,  # noqa: F401
                             next_trading_day, trading_days)


def market_holidays(year: int) -> frozenset[date]:
    try:
        return _cal.market_holidays(year)
    except ValueError:
        lo, hi = min(MARKET_HOLIDAYS), max(MARKET_HOLIDAYS)
        raise FatalError(
            f"the NYSE holiday table does not cover {year}; it runs {lo}-{hi}.",
            f"add {year} to market_calendar._MARKET_HOLIDAYS from https://www.nyse.com/markets/hours-calendars, then re-run."
        )


def eastern_now() -> datetime:
    return datetime.now(EASTERN)

def most_recent_available_date(now_et: datetime | None = None) -> date:
    now_et = now_et or eastern_now()
    d = now_et.date()
    if not (is_trading_day(d) and now_et.hour >= QUOTA_RESET_HOUR_ET):
        d -= timedelta(days=1)
    return previous_trading_day(d)

# =====================================================================
# ERROR HANDLING
# =====================================================================
class FatalError(Exception):
    def __init__(self, message: str, help_text: str | None = None) -> None:
        super().__init__(message)
        self.help = help_text

class NoDataForDate(Exception):
    pass

def out(line: str = "") -> None:
    sys.stdout.write(line + "\n")

def progress(line: str = "") -> None:
    sys.stderr.write(line + "\n")

def emit_error(err: FatalError) -> None:
    out(f"error: {err}")
    if err.help: out(f"help: {err.help}")

def human_bytes(n: float) -> str:
    size = float(n)
    if size < 1000: return f"{int(size)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        size /= 1000.0
        if size < 1000 or unit == "TB":
            return f"{size:.2f} {unit}"
    return f"{size:.2f} TB"

# =====================================================================
# LAKE LAYOUT
# =====================================================================
def bronze_dir(lake: Path) -> Path: return lake / BRONZE_SUBDIR / BRONZE_DATASET
def work_dir(lake: Path) -> Path: return lake / WORK_SUBDIR
def bronze_path(lake: Path, d: date) -> Path: return bronze_dir(lake) / f"{d.isoformat()}.parquet"

def present_dates(lake: Path) -> dict[date, int]:
    bd = bronze_dir(lake)
    if not bd.is_dir(): return {}
    found: dict[date, int] = {}
    for p in bd.glob("*.parquet"):
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.parquet", p.name)
        if m: found[date.fromisoformat(m.group(1))] = p.stat().st_size
    return found

def silver_dir(lake: Path) -> Path: return lake / SILVER_SUBDIR
def silver_bars_dir(lake: Path) -> Path: return silver_dir(lake) / SILVER_BARS_DATASET
def silver_partition_dir(lake: Path, d: date) -> Path: return silver_bars_dir(lake) / f"date={d.isoformat()}"
def silver_partition_path(lake: Path, d: date) -> Path: return silver_partition_dir(lake, d) / SILVER_BARS_FILE

def present_silver_dates(lake: Path) -> dict[date, int]:
    bd = silver_bars_dir(lake)
    if not bd.is_dir(): return {}
    found: dict[date, int] = {}
    for p in bd.glob("date=*"):
        if not p.is_dir(): continue
        m = re.fullmatch(r"date=(\d{4}-\d{2}-\d{2})", p.name)
        if m:
            d = date.fromisoformat(m.group(1))
            bars = p / SILVER_BARS_FILE
            if bars.is_file(): found[d] = bars.stat().st_size
    return found

# =====================================================================
# CONVERTER: Bronze Parquet
# =====================================================================
@dataclass
class SourceMetrics:
    rows: int
    empty_tokens: dict[str, int]

@dataclass
class BronzeMetrics:
    rows: int
    columns: tuple[str, ...]
    null_counts: dict[str, int]
    dtypes: dict[str, pl.DataType]

@dataclass
class ConvertResult:
    date: date | None
    parquet_path: Path
    csv_path: Path
    rows: int
    parquet_bytes: int

def open_tape_zip(zip_path: Path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise FatalError(f"{zip_path.name} is not a valid zip file.", "re-download the date.")

def extract_csv(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open_tape_zip(zip_path) as zf:
        names = zf.namelist()
        entry = names[0]
        dest = dest_dir / Path(entry).name
        with zf.open(entry) as src, open(dest, "wb") as dst:
            while True:
                chunk = src.read(STREAM_CHUNK)
                if not chunk: break
                dst.write(chunk)
        return dest

def _empty_or_null(col: str) -> pl.Expr:
    return (pl.col(col).is_null() | (pl.col(col).cast(pl.String).str.strip_chars() == "")).sum()

def source_metrics(csv_path: Path) -> SourceMetrics:
    lf = pl.scan_csv(csv_path, infer_schema_length=INFER_SCHEMA_ROWS)
    agg = lf.select([pl.len().alias("_rows_")] + [_empty_or_null(c).alias(c) for c in TIMESTAMP_COLUMNS])
    row = agg.collect().row(0, named=True)
    return SourceMetrics(rows=int(row["_rows_"]), empty_tokens={c: int(row[c]) for c in TIMESTAMP_COLUMNS})

def write_bronze_parquet(csv_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_csv(csv_path, infer_schema_length=INFER_SCHEMA_ROWS)
    lf = lf.with_columns(
        pl.col(c).str.to_datetime(time_unit="us", time_zone="UTC", strict=False) for c in TIMESTAMP_COLUMNS
    )
    lf.sink_parquet(out_path, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)

def bronze_metrics(parquet_path: Path) -> BronzeMetrics:
    lf = pl.scan_parquet(parquet_path)
    schema = lf.collect_schema()
    agg = lf.select([pl.len().alias("_rows_")] + [pl.col(c).null_count().alias(c) for c in TIMESTAMP_COLUMNS])
    row = agg.collect().row(0, named=True)
    return BronzeMetrics(
        rows=int(row["_rows_"]), columns=tuple(schema.names()),
        null_counts={c: int(row[c]) for c in TIMESTAMP_COLUMNS}, dtypes={c: schema[c] for c in TIMESTAMP_COLUMNS}
    )

def validate_bronze(src: SourceMetrics, bronze: BronzeMetrics) -> list[str]:
    problems: list[str] = []
    # SUBSET, not equality. UW WIDENS this tape over time and the check used to
    # demand an exact match, so an ADDITIVE change broke ingestion outright:
    #   ..2026-08-28  40 cols
    #     2026-09-01  47  + exchange_id, nbbo_{bid,ask}_{exchange_id,size,time}
    #     2026-09-14  49  + nbbo_bid_exchange, nbbo_ask_exchange
    # The rollout is STAGED and still moving (two fields landed 2026-09-14), so
    # pinning to 49 would simply break on the next wave. Old dates still serve
    # 40 -- the change is NOT retroactive -- so the lake stays internally
    # consistent and mixed-width partitions are expected from 2026-09-01 on.
    # A MISSING expected column is still fatal: that is the corruption this
    # check exists to catch. Extras are reported and kept.
    missing = [c for c in EXPECTED_COLUMNS if c not in bronze.columns]
    if missing:
        problems.append(f"missing expected column(s): {', '.join(missing)}")
    extra = [c for c in bronze.columns if c not in EXPECTED_COLUMNS]
    if extra:
        progress(f"  note: {len(extra)} new tape column(s) kept: {', '.join(extra)}")
    if bronze.rows != src.rows: problems.append(f"row count {bronze.rows} != source {src.rows}")
    for c in TIMESTAMP_COLUMNS:
        new_nulls = bronze.null_counts[c] - src.empty_tokens[c]
        if new_nulls > 0: problems.append(f"{c} gained {new_nulls} null(s) beyond source.")
    return problems

def convert_zip(zip_path: Path, lake: Path, d: date | None = None) -> ConvertResult:
    out_path = bronze_path(lake, d) if d else lake / (zip_path.stem + ".parquet")
    progress(f"extracting {zip_path.name} ...")
    csv_path = extract_csv(zip_path, work_dir(lake))
    progress(f"scanning source rows in {csv_path.name} ...")
    src = source_metrics(csv_path)
    progress(f"writing bronze parquet ({src.rows:,} rows) ...")
    write_bronze_parquet(csv_path, out_path)
    bronze = bronze_metrics(out_path)
    problems = validate_bronze(src, bronze)
    if problems:
        out_path.unlink(missing_ok=True)
        raise FatalError("bronze validation failed: " + "; ".join(problems))
    return ConvertResult(date=d, parquet_path=out_path, csv_path=csv_path, rows=bronze.rows, parquet_bytes=out_path.stat().st_size)

# =====================================================================
# HTTP CLIENT
# =====================================================================
class Client:
    def __init__(self, api_key: str) -> None:
        self.headers = {"Authorization": f"Bearer {api_key}", "Accept": "*/*", "User-Agent": USER_AGENT, "UW-CLIENT-API-ID": UW_CLIENT_API_ID}

    def _full_tape_url(self, d: date) -> str:
        return f"{API_BASE}{FULL_TAPE_PATH}/{d.isoformat()}"

    def download(self, d: date, dest: Path) -> Path:
        url = self._full_tape_url(d)
        with httpx.stream("GET", url, headers=self.headers, timeout=HTTP_TIMEOUT, follow_redirects=True) as r:
            if r.status_code == 404: raise NoDataForDate()
            if r.status_code != 200: raise FatalError(f"HTTP {r.status_code}", "Failed to download.")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=STREAM_CHUNK): f.write(chunk)
        return dest

    def probe_boundary(self) -> tuple[date | None, int | None]:
        # Minimal probe fallback
        return None, None

    def get_json(self, path: str, params: dict[str, str] | None = None) -> dict:
        """GET a JSON endpoint under API_BASE with retry/backoff on 429/5xx.
        404 -> NoDataForDate. Sleeps REST_SLEEP_SECONDS after a success so bulk
        loops stay under the rate limit."""
        url = f"{API_BASE}{path}"
        last: str | None = None
        for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
            try:
                r = httpx.get(url, headers=self.headers, params=params,
                              timeout=HTTP_TIMEOUT, follow_redirects=True)
            except httpx.HTTPError as e:
                last = str(e)
                time.sleep(REST_BACKOFF_SECONDS * attempt)
                continue
            if r.status_code in (404, 422):
                # 404 = no data for that date; 422 = ticker not valid for this
                # endpoint (e.g. an index on /ohlc/1m). Either way: skip, don't abort.
                raise NoDataForDate()
            if r.status_code in REST_RETRY_STATUS:
                last = f"HTTP {r.status_code}"
                time.sleep(REST_BACKOFF_SECONDS * attempt)
                continue
            if r.status_code != 200:
                raise FatalError(f"HTTP {r.status_code} for {url}", (r.text or "")[:300])
            time.sleep(REST_SLEEP_SECONDS)
            return r.json()
        raise FatalError(f"{url} failed after {DOWNLOAD_MAX_RETRIES} attempts", last)

def build_one(client: Client, d: date, lake: Path) -> ConvertResult | None:
    dest = bronze_path(lake, d)
    if dest.exists(): return None
    zip_path = work_dir(lake) / f"full_tape_{d.strftime('%Y%m%d')}.zip"
    try:
        client.download(d, zip_path)
    except NoDataForDate:
        progress(f" {d.isoformat()}: no tape (skipped)")
        return None
    result = convert_zip(zip_path, lake, d=d)
    zip_path.unlink(missing_ok=True)
    result.csv_path.unlink(missing_ok=True)
    return result

# =====================================================================
# TRADES-CORE -- the trade-level tape, filtered and kept; the raw day is not
# =====================================================================
# lake/silver/trades-core/date=YYYY-MM-DD/trades.parquet
#
# WHY A SECOND SILVER LAYER. option-contracts-1m aggregates a day to contract-
# minutes, which throws away the two things a front-running test needs: WHEN
# inside the minute a trade printed, and WHAT KIND of trade it was (sweep,
# block, auction, multi-leg). This keeps individual trades, but only for the
# tickers asked for and only the columns that carry those things -- ~30 MB/day
# against ~1 GB for the full bronze parquet and ~2.6 GB for the download.
#
# 🚨 THE FULL BRONZE DAY IS NEVER WRITTEN. The existing `build` path writes the
# 49-column parquet and keeps it; 714 such days would be ~670 GB, which does not
# fit on this machine. Here the CSV is filtered on the way to parquet and the
# zip and CSV are deleted in a `finally`, so a crash mid-day cannot strand
# 10 GB of temp files. The price: re-deriving a column not kept here means
# re-downloading. The columns below were chosen to be the complete set a
# trade-level flow study reads; widen TRADES_CORE_COLUMNS BEFORE a backfill,
# not after.
#
# 🚨 VALIDATED AGAINST option-contracts-1m, EVERY DAY. check_opra_multileg
# proved (2026-09-26) that silver's per-contract-minute volume equals the tape
# exactly. So each day's kept, non-canceled volume per ticker must match
# silver's to within 0.1%, or the day is NOT written. That is an independent
# source for the same number, which is the check that catches a bad filter, a
# truncated download or a parse change -- a self-consistency count would not.
TRADES_CORE_DATASET = "trades-core"
TRADES_CORE_FILE = "trades.parquet"
TRADES_CORE_TICKERS: tuple[str, ...] = ("SPY", "QQQ", "IWM")
# Only columns present in the ORIGINAL 40-column tape, so every day in the
# 2023-10-12.. history has all of them (see validate_bronze on the widening).
TRADES_CORE_COLUMNS: tuple[str, ...] = (
    "underlying_symbol", "executed_at", "option_type", "strike", "expiry",
    "size", "price", "nbbo_bid", "nbbo_ask", "underlying_price",
    "upstream_condition_detail", "report_flags", "tags", "exchange",
    "canceled",
)
TRADES_CORE_TOLERANCE = 0.001
TRADES_CORE_RETRIES = 3


def trades_core_path(lake: Path, d: date) -> Path:
    return silver_dir(lake) / TRADES_CORE_DATASET / f"date={d.isoformat()}" / TRADES_CORE_FILE


def _trades_core_frame(lf: pl.LazyFrame, tickers: Sequence[str]) -> pl.DataFrame:
    """Filter and type one day's tape. Explicit casts: CSV inference and the
    bronze parquet disagree on some types (expiry is String in bronze), and a
    lake whose partitions disagree on a dtype fails at scan time, months later."""
    ts = pl.col("executed_at")
    if lf.collect_schema().get("executed_at") == pl.String:
        ts = ts.str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
    return (lf.filter(pl.col("underlying_symbol").is_in(list(tickers)))
            .select(TRADES_CORE_COLUMNS)
            .with_columns(
                ts.alias("executed_at"),
                pl.col("expiry").cast(pl.String).str.slice(0, 10).str.to_date(),
                pl.col("strike").cast(pl.Float64),
                pl.col("size").cast(pl.Int64),
                pl.col("price", "nbbo_bid", "nbbo_ask", "underlying_price").cast(pl.Float64),
                pl.col("upstream_condition_detail", "report_flags", "tags", "exchange",
                       "canceled").cast(pl.String))
            .sort("executed_at")
            .collect())


def _validate_trades_core(df: pl.DataFrame, lake: Path, d: date,
                          tickers: Sequence[str]) -> tuple[list[str], str]:
    problems: list[str] = []
    nulls = df["executed_at"].null_count()
    if nulls:
        problems.append(f"{nulls} unparseable executed_at")
    sil = silver_partition_path(lake, d)
    if not sil.exists():
        return problems, "no option-contracts-1m partition to check against"
    ref = (pl.scan_parquet(sil).filter(pl.col("underlying_symbol").is_in(list(tickers)))
           .group_by("underlying_symbol").agg(pl.col("volume").sum()).collect())
    # fill_null first: `null != "t"` is null, and filter() drops nulls -- a
    # blank flag would silently remove a real trade from the volume count.
    got = (df.filter(pl.col("canceled").fill_null("f") != "t").group_by("underlying_symbol")
           .agg(pl.col("size").sum()))
    j = ref.join(got, on="underlying_symbol", how="full", coalesce=True).fill_null(0)
    worst = 0.0
    for r in j.iter_rows(named=True):
        denom = max(r["volume"], 1)
        err = abs(r["size"] - r["volume"]) / denom
        worst = max(worst, err)
        if err > TRADES_CORE_TOLERANCE:
            problems.append(f"{r['underlying_symbol']} volume {r['size']:,} vs silver "
                            f"{r['volume']:,} ({err:.3%})")
    return problems, f"matches silver volume (worst {worst:.4%})"


def build_trades_core_one(client: "Client", d: date, lake: Path,
                          tickers: Sequence[str] = TRADES_CORE_TICKERS) -> str:
    """One day -> trades-core. Returns a one-line status. Raises FatalError."""
    out_path = trades_core_path(lake, d)
    if out_path.exists():
        return "exists"
    t0 = time.time()
    day_work = work_dir(lake) / f"core_{d.isoformat()}"
    zip_path = day_work / f"full_tape_{d.strftime('%Y%m%d')}.zip"
    zip_bytes = 0
    try:
        local = bronze_path(lake, d)
        if local.exists():
            # already have the full bronze day: derive, don't re-download
            df = _trades_core_frame(pl.scan_parquet(local), tickers)
            src = "local bronze"
        else:
            for attempt in range(1, TRADES_CORE_RETRIES + 1):
                try:
                    client.download(d, zip_path)
                    break
                except NoDataForDate:
                    return "no tape (holiday or not yet published)"
                except (httpx.HTTPError, FatalError) as e:
                    if attempt == TRADES_CORE_RETRIES:
                        raise FatalError(f"download failed after {attempt} attempts: {e}")
                    time.sleep(10 * attempt)
            zip_bytes = zip_path.stat().st_size
            csv_path = extract_csv(zip_path, day_work)
            zip_path.unlink(missing_ok=True)       # free 2.6 GB before parsing
            df = _trades_core_frame(pl.scan_csv(csv_path, infer_schema_length=INFER_SCHEMA_ROWS), tickers)
            src = f"download {human_bytes(zip_bytes)}"
        problems, note = _validate_trades_core(df, lake, d, tickers)
        if problems:
            raise FatalError("trades-core validation failed: " + "; ".join(problems))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        df.write_parquet(tmp, compression=PARQUET_COMPRESSION,
                         compression_level=PARQUET_COMPRESSION_LEVEL)
        tmp.replace(out_path)                        # atomic: no half-written day
        return (f"{df.height:,} trades, {human_bytes(out_path.stat().st_size)}, "
                f"{src}, {note}, {time.time() - t0:.0f}s")
    finally:
        if day_work.exists():
            for p in day_work.iterdir():
                p.unlink(missing_ok=True)
            day_work.rmdir()


# =====================================================================
# SILVER BUILDER
# =====================================================================
def source_noncanceled_volume(bronze_parquet: Path) -> int:
    v = pl.scan_parquet(bronze_parquet).filter(pl.col("canceled") == "f").select(pl.col("size").sum()).collect().item()
    return int(v or 0)

def _price_pass(df: pl.DataFrame) -> pl.DataFrame:
    live = (df.filter(pl.col("canceled") == "f")
            .with_columns(pl.col("executed_at").dt.truncate("1m").alias("minute_utc"))
            .sort(["option_chain_id", "executed_at", "volume"]))
    return live.group_by(["option_chain_id", "minute_utc"], maintain_order=True).agg(
        pl.col("price").first().alias("open"), pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"), pl.col("price").last().alias("close"),
        pl.col("size").sum().alias("volume"), pl.len().cast(pl.Int64).alias("trade_count"),
        pl.col("premium").sum().alias("premium"), pl.col("nbbo_bid").last().alias("bid_close"),
        pl.col("nbbo_ask").last().alias("ask_close"), pl.col("underlying_price").first().alias("underlying_open"),
        pl.col("underlying_price").last().alias("underlying_close"), pl.col("implied_volatility").last().alias("iv_close"),
        pl.col("delta").last().alias("delta_close"), pl.col("gamma").last().alias("gamma_close"),
        pl.col("theta").last().alias("theta_close"), pl.col("vega").last().alias("vega_close"),
        pl.col("rho").last().alias("rho_close"), pl.col("theo").last().alias("theo_close"),
        pl.col("open_interest").last().alias("open_interest"), pl.col("underlying_symbol").first().alias("underlying_symbol"),
        pl.col("option_type").first().alias("option_type"), pl.col("strike").first().alias("strike"),
        pl.col("expiry").first().alias("expiry"), pl.col("tags").first().alias("tags"),
    )

def _side_pass(df: pl.DataFrame) -> pl.DataFrame:
    per_minute = (df.with_columns(pl.col("executed_at").dt.truncate("1m").alias("minute_utc"))
                  .group_by(["option_chain_id", "minute_utc"])
                  .agg([pl.col(c).max().alias(c) for c in SILVER_SIDE_COUNTERS])
                  .sort(["option_chain_id", "minute_utc"]))
    deltas = per_minute.with_columns([
        (pl.col(counter) - pl.col(counter).shift(1).over("option_chain_id")).fill_null(pl.col(counter)).clip(lower_bound=0).alias(out_name)
        for counter, out_name in zip(SILVER_SIDE_COUNTERS, SILVER_SIDE_VOLUMES)
    ])
    return deltas.select(["option_chain_id", "minute_utc", *SILVER_SIDE_VOLUMES])

def build_bars(bronze_parquet: Path) -> pl.DataFrame:
    df = pl.read_parquet(bronze_parquet, columns=list(SILVER_SOURCE_COLUMNS))
    bars = _price_pass(df).join(_side_pass(df), on=["option_chain_id", "minute_utc"], how="left")
    bars = bars.with_columns(
        (pl.col("premium") / pl.col("volume") / 100).alias("vwap"),
        pl.col("minute_utc").dt.convert_time_zone(EASTERN_TZ).alias("minute_et"),
        pl.when(pl.col("tags").str.contains("etf", literal=True)).then(pl.lit("etf"))
        .when(pl.col("tags").str.contains("index", literal=True)).then(pl.lit("index"))
        .otherwise(pl.lit("equity")).alias("security_type"),
        pl.col("expiry").str.to_date().alias("expiry"),
    )
    return bars.select(SILVER_COLUMNS).sort(["option_chain_id", "minute_utc"])

@dataclass
class SilverBuildResult:
    date: date
    partition_path: Path
    bars: int
    parquet_bytes: int

def build_silver_one(lake: Path, d: date) -> SilverBuildResult | None:
    dest = silver_partition_path(lake, d)
    if dest.exists(): return None
    bronze = bronze_path(lake, d)
    if not bronze.exists(): return None
    bars = build_bars(bronze)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (SILVER_BARS_FILE + ".tmp")
    bars.write_parquet(tmp, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)
    tmp.replace(dest)
    
    return SilverBuildResult(
        date=d, 
        partition_path=dest, 
        bars=bars.height, 
        parquet_bytes=dest.stat().st_size
    )

# =====================================================================
# GEX / GREEK-EXPOSURE BUILDERS
# =====================================================================
def historical_gex_path(out_dir: Path, ticker: str) -> Path:
    return out_dir / f"GEX{ticker}.parquet"

def spot_gex_partition_path(lake: Path, d: date, ticker: str) -> Path:
    return silver_dir(lake) / SPOT_GEX_DATASET / f"date={d.isoformat()}" / f"{ticker}.parquet"

@dataclass
class GexDailyResult:
    ticker: str
    path: Path
    rows: int
    start: date | None
    end: date | None

def build_greek_exposure_one(client: Client, ticker: str, timeframe: str, out_dir: Path) -> GexDailyResult:
    """One /greek-exposure call backfills the full timeframe of daily greek
    exposure. Writes historical/GEX{T}.parquet with explicit net_* columns plus
    net_gex_prior (yesterday's close, for a lookahead-free regime label)."""
    payload = client.get_json(f"/stock/{ticker}/greek-exposure", {"timeframe": timeframe})
    rows = payload.get("data") or []
    if not rows:
        raise NoDataForDate()
    # UW sends every value as a JSON string but a field can be null in old rows
    # and numeric-looking later; scan all rows so polars keeps it as Utf8.
    df = pl.DataFrame(rows, infer_schema_length=None)
    missing = [c for c in ("call_gamma", "put_gamma", "date") if c not in df.columns]
    if missing:
        raise FatalError(
            f"/greek-exposure {ticker}: response missing {missing}; got {df.columns}.",
            "UW schema changed - update GREEK_EXPOSURE_PAIRS.",
        )
    df = df.with_columns(
        pl.col(c).cast(pl.Float64, strict=False) for c in GREEK_EXPOSURE_NUMERIC if c in df.columns
    )
    df = (df.with_columns(pl.col("date").str.to_date(strict=False))
            .filter(pl.col("date").is_not_null())
            .sort("date"))
    df = df.with_columns(
        (pl.col(a) + pl.col(b)).alias(net)
        for a, b, net in GREEK_EXPOSURE_PAIRS
        if a in df.columns and b in df.columns
    )
    df = df.with_columns(pl.col("net_gex").shift(1).alias("net_gex_prior"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = historical_gex_path(out_dir, ticker)
    tmp = dest.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)
    tmp.replace(dest)
    return GexDailyResult(ticker, dest, df.height, df["date"].min(), df["date"].max())

def build_spot_gex_one(client: Client, ticker: str, d: date, lake: Path) -> int | None:
    """One /spot-exposures call = one ticker's per-minute spot GEX for date d.
    Returns row count, or None when the partition already exists / no data."""
    dest = spot_gex_partition_path(lake, d, ticker)
    if dest.exists():
        return None
    try:
        payload = client.get_json(f"/stock/{ticker}/spot-exposures", {"date": d.isoformat()})
    except NoDataForDate:
        return None
    rows = payload.get("data") or []
    if not rows:
        return None
    df = pl.DataFrame(rows, infer_schema_length=None)
    if "time" not in df.columns:
        raise FatalError(f"/spot-exposures {ticker} {d}: no 'time' column; got {df.columns}.")
    df = df.with_columns(
        pl.col(c).cast(pl.Float64, strict=False) for c in SPOT_GEX_NUMERIC if c in df.columns
    )
    tscols = [c for c in ("time", "start_time") if c in df.columns]
    df = df.with_columns(
        pl.col(c).str.to_datetime(time_unit="us", time_zone="UTC", strict=False) for c in tscols
    ).filter(pl.col("time").is_not_null()).sort("time")
    # start_time is UW's exact per-minute bucket (:00 seconds); time is the
    # calc instant within it. Bucket off start_time when present.
    minute_src = pl.col("start_time") if "start_time" in df.columns else pl.col("time").dt.truncate("1m")
    df = df.with_columns(
        pl.lit(ticker).alias("ticker"),
        minute_src.alias("minute_utc"),
    )
    df = df.with_columns(
        pl.col("minute_utc").dt.convert_time_zone(EASTERN_TZ).alias("minute_et"),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)
    tmp.replace(dest)
    return df.height

def historical_ohlc_path(out_dir: Path, ticker: str) -> Path:
    return out_dir / f"{ticker}.parquet"

def _ohlc_normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Reconcile a fresh UW OHLC frame with a pre-existing / purchased file:
    start_time -> datetime[us, UTC] (string or datetime in), and the older
    `candle_volume` column name -> `volume`."""
    if "candle_volume" in df.columns and "volume" not in df.columns:
        df = df.rename({"candle_volume": "volume"})
    for tcol in ("start_time", "end_time"):
        if tcol not in df.columns:
            continue
        dt = df.schema[tcol]
        if dt == pl.String:
            df = df.with_columns(
                pl.col(tcol).str.to_datetime(time_unit="us", time_zone="UTC", strict=False))
        elif isinstance(dt, pl.Datetime):
            col = pl.col(tcol)
            col = col.dt.replace_time_zone("UTC") if dt.time_zone is None else col.dt.convert_time_zone("UTC")
            df = df.with_columns(col.dt.cast_time_unit("us"))
        elif tcol == "start_time":
            raise FatalError(f"historical OHLC 'start_time' dtype {dt}; expected String or Datetime.")
    return df

def _existing_ohlc_dates(dest: Path) -> set[date]:
    if not dest.exists():
        return set()
    df = _ohlc_normalize(pl.read_parquet(dest))
    if "start_time" not in df.columns:
        return set()
    vals = df.select(pl.col("start_time").dt.date()).to_series().drop_nulls().to_list()
    return set(vals)

def _fetch_ohlc_day(client: Client, ticker: str, candle: str, d: date) -> pl.DataFrame | None:
    try:
        payload = client.get_json(f"/stock/{ticker}/ohlc/{candle}", {"date": d.isoformat()})
    except NoDataForDate:
        return None
    rows = payload.get("data") or []
    if not rows:
        return None
    df = pl.DataFrame(rows, infer_schema_length=None)
    if "start_time" not in df.columns:
        raise FatalError(f"/stock/{ticker}/ohlc/{candle} {d}: no 'start_time'; got {df.columns}.")
    df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False) for c in OHLC_NUMERIC if c in df.columns)
    return _ohlc_normalize(df)

@dataclass
class OhlcBuildResult:
    ticker: str
    path: Path
    rows: int
    fetched_days: int
    start: date | None
    end: date | None

def build_ohlc_one(client: Client, ticker: str, candle: str, days: list[date], out_dir: Path) -> OhlcBuildResult | None:
    """Top up historical/{T}.parquet with any of `days` not already present.
    start_time is normalised to datetime[us, UTC] (backtesters parse it either
    way); minute_et and date are (re)derived. Merges onto a pre-existing file."""
    dest = historical_ohlc_path(out_dir, ticker)
    have = _existing_ohlc_dates(dest)
    todo = [d for d in days if d not in have]
    if not todo:
        return None
    frames = [df for d in todo if (df := _fetch_ohlc_day(client, ticker, candle, d)) is not None]
    if not frames:
        return None
    combined = pl.concat(frames, how="diagonal_relaxed")
    if dest.exists():
        old = _ohlc_normalize(pl.read_parquet(dest)).drop("minute_et", "date", strict=False)
        combined = pl.concat([old, combined], how="diagonal_relaxed")
    combined = (combined.unique(subset=["start_time"], keep="last")
                        .sort("start_time")
                        .with_columns(
                            pl.col("start_time").dt.date().alias("date"),
                            pl.col("start_time").dt.convert_time_zone(EASTERN_TZ).alias("minute_et"),
                        ))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    combined.write_parquet(tmp, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)
    tmp.replace(dest)
    return OhlcBuildResult(ticker, dest, combined.height, len(frames), combined["date"].min(), combined["date"].max())

def historical_netprem_path(out_dir: Path, ticker: str) -> Path:
    return out_dir / f"{NETPREM_PREFIX}{ticker}.parquet"

def _netprem_normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Cast the string $ / delta fields to Float64 and (re)derive tape_time ->
    datetime[us, UTC]. Idempotent, so it also reconciles an on-disk file."""
    df = df.with_columns(
        pl.col(c).cast(pl.Float64, strict=False) for c in NETPREM_NUMERIC if c in df.columns
    )
    if "tape_time" in df.columns and df.schema["tape_time"] == pl.String:
        df = df.with_columns(
            pl.col("tape_time").str.to_datetime(time_unit="us", time_zone="UTC", strict=False))
    elif "tape_time" in df.columns and isinstance(df.schema["tape_time"], pl.Datetime):
        tz = df.schema["tape_time"].time_zone
        col = pl.col("tape_time")
        col = col.dt.replace_time_zone("UTC") if tz is None else col.dt.convert_time_zone("UTC")
        df = df.with_columns(col.dt.cast_time_unit("us"))
    return df

def _existing_netprem_dates(dest: Path) -> set[date]:
    if not dest.exists():
        return set()
    df = _netprem_normalize(pl.read_parquet(dest))
    if "tape_time" not in df.columns:
        return set()
    return set(df.select(pl.col("tape_time").dt.date()).to_series().drop_nulls().to_list())

def _fetch_netprem_day(client: Client, ticker: str, d: date) -> pl.DataFrame | None:
    """One /net-prem-ticks call. None for an empty day (weekend/holiday) or a
    403 (date is before UW's FIXED 2023-10-12 history floor) -- both mean
    'skip'. Not a rolling window: see the NETPREM_* block above."""
    try:
        payload = client.get_json(f"/stock/{ticker}/net-prem-ticks", {"date": d.isoformat()})
    except NoDataForDate:
        return None
    except FatalError as e:
        if str(e).startswith("HTTP 403"):
            return None
        raise
    rows = payload.get("data") or []
    if not rows:
        return None
    df = pl.DataFrame(rows, infer_schema_length=None)
    if "tape_time" not in df.columns:
        raise FatalError(f"/stock/{ticker}/net-prem-ticks {d}: no 'tape_time'; got {df.columns}.")
    return _netprem_normalize(df).with_columns(pl.lit(ticker).alias("ticker"))

@dataclass
class NetPremBuildResult:
    ticker: str
    path: Path
    rows: int
    fetched_days: int
    start: date | None
    end: date | None

def build_netprem_one(client: Client, ticker: str, days: list[date], out_dir: Path) -> NetPremBuildResult | None:
    """Top up historical/NETPREM{T}.parquet with any of `days` not already
    present. Derives minute_utc/minute_et/date plus net_premium (call - put $,
    the live bot's signal) and net_volume (call - put contracts). Incremental,
    dedupes on tape_time. Mirrors build_ohlc_one."""
    dest = historical_netprem_path(out_dir, ticker)
    have = _existing_netprem_dates(dest)
    todo = [d for d in days if d not in have]
    if not todo:
        return None
    frames = [df for d in todo if (df := _fetch_netprem_day(client, ticker, d)) is not None]
    if not frames:
        return None
    combined = pl.concat(frames, how="diagonal_relaxed")
    if dest.exists():
        old = _netprem_normalize(pl.read_parquet(dest)).drop(
            "minute_utc", "minute_et", "date", "net_premium", "net_volume", strict=False)
        combined = pl.concat([old, combined], how="diagonal_relaxed")
    combined = (combined.unique(subset=["tape_time"], keep="last")
                        .sort("tape_time")
                        .with_columns(
                            pl.col("tape_time").alias("minute_utc"),
                            pl.col("tape_time").dt.date().alias("date"),
                            pl.col("tape_time").dt.convert_time_zone(EASTERN_TZ).alias("minute_et"),
                        ))
    have_cols = set(combined.columns)
    if {"net_call_premium", "net_put_premium"} <= have_cols:
        combined = combined.with_columns(
            (pl.col("net_call_premium") - pl.col("net_put_premium")).alias("net_premium"))
    if {"net_call_volume", "net_put_volume"} <= have_cols:
        combined = combined.with_columns(
            (pl.col("net_call_volume") - pl.col("net_put_volume")).alias("net_volume"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    combined.write_parquet(tmp, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)
    tmp.replace(dest)
    return NetPremBuildResult(ticker, dest, combined.height, len(frames), combined["date"].min(), combined["date"].max())

# =====================================================================
# CLI RUNNER
# =====================================================================
def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv() # Force the sandbox to read the .env file
    
    parser = argparse.ArgumentParser(prog="uw_options_data_lake.py")
    parser.add_argument("--api-key", default=None, help="UW API key")
    sub = parser.add_subparsers(dest="command")
    
    b = sub.add_parser("build", help="download+convert+validate a date or range")
    b.add_argument("start", type=date.fromisoformat)
    b.add_argument("end", type=date.fromisoformat, nargs="?", default=None)
    b.add_argument("--confirm", action="store_true")

    tc = sub.add_parser("trades-core-build",
                        help="trade-level tape for a few tickers -> silver/trades-core (raw day deleted)")
    tc.add_argument("start", type=date.fromisoformat)
    tc.add_argument("end", type=date.fromisoformat, nargs="?", default=None)
    tc.add_argument("--tickers", nargs="+", default=list(TRADES_CORE_TICKERS))
    tc.add_argument("--workers", type=int, default=3,
                    help="days downloaded in parallel (each needs ~11 GB of temp space)")
    tc.add_argument("--confirm", action="store_true")

    sb = sub.add_parser("silver-build", help="build 1-min per-contract bars")
    sb.add_argument("start", type=date.fromisoformat)
    sb.add_argument("end", type=date.fromisoformat, nargs="?", default=None)
    sb.add_argument("--confirm", action="store_true")

    gd = sub.add_parser("gex-daily-build", help="backfill daily net-GEX regime series -> historical/GEX{T}.parquet")
    gd.add_argument("--tickers", nargs="+", default=list(GEX_TICKERS))
    gd.add_argument("--tickers-file", type=Path, default=None,
                    help="JSON {\"tickers\": [...]} (e.g. smallcap_universe.json) -- overrides --tickers")
    gd.add_argument("--timeframe", default="10Y",
                    help="UW timeframe: YTD, 6M, 1Y, 2Y, ... (default 10Y = all UW has; "
                         "currently caps at 2022-03-30 regardless)")
    gd.add_argument("--out-dir", type=Path, default=DEFAULT_HISTORICAL)

    sg = sub.add_parser("spot-gex-build", help="backfill 1-min spot GEX -> lake/silver/spot-exposures-1m/")
    sg.add_argument("start", type=date.fromisoformat)
    sg.add_argument("end", type=date.fromisoformat, nargs="?", default=None)
    sg.add_argument("--tickers", nargs="+", default=list(GEX_TICKERS))
    sg.add_argument("--tickers-file", type=Path, default=None,
                    help="JSON {\"tickers\": [...]} (e.g. smallcap_universe.json) -- overrides --tickers")
    sg.add_argument("--confirm", action="store_true")

    oh = sub.add_parser("ohlc-build", help="backfill underlying OHLC -> historical/{T}.parquet (incremental)")
    oh.add_argument("start", type=date.fromisoformat)
    oh.add_argument("end", type=date.fromisoformat, nargs="?", default=None)
    oh.add_argument("--tickers", nargs="+", default=list(GEX_TICKERS))
    oh.add_argument("--tickers-file", type=Path, default=None,
                    help="JSON {\"tickers\": [...]} (e.g. smallcap_universe.json) -- overrides --tickers")
    oh.add_argument("--candle", default=DEFAULT_OHLC_CANDLE, help="1m, 5m, 1h, 1d (default 1m)")
    oh.add_argument("--out-dir", type=Path, default=DEFAULT_HISTORICAL)
    oh.add_argument("--confirm", action="store_true")

    np_ = sub.add_parser("netprem-build",
                         help="backfill 1-min net-premium / bid-ask flow -> historical/NETPREM{T}.parquet (incremental)")
    np_.add_argument("start", type=date.fromisoformat)
    np_.add_argument("end", type=date.fromisoformat, nargs="?", default=None)
    np_.add_argument("--tickers", nargs="+", default=list(GEX_TICKERS))
    np_.add_argument("--tickers-file", type=Path, default=None,
                    help="JSON {\"tickers\": [...]} (e.g. smallcap_universe.json) -- overrides --tickers")
    np_.add_argument("--out-dir", type=Path, default=DEFAULT_HISTORICAL)
    np_.add_argument("--confirm", action="store_true")

    args = parser.parse_args(argv)
    if getattr(args, "tickers_file", None):
        with open(args.tickers_file) as f:
            payload = json.load(f)
        args.tickers = payload["tickers"] if isinstance(payload, dict) else payload
        print(f"  --tickers-file {args.tickers_file}: {len(args.tickers)} tickers")
    api_key = args.api_key or os.getenv("UW_API_KEY")

    if args.command == "build":
        if not api_key: raise FatalError("API Key required.")
        client = Client(api_key)
        for d in trading_days(args.start, args.end or args.start):
            progress(f"Building bronze for {d.isoformat()}...")
            build_one(client, d, DEFAULT_LAKE)
        print("Bronze build complete!")

    elif args.command == "trades-core-build":
        if not api_key: raise FatalError("API Key required.")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import shutil
        client = Client(api_key)
        tickers = [t.upper() for t in args.tickers]
        days = [d for d in trading_days(args.start, args.end or args.start)
                if not trades_core_path(DEFAULT_LAKE, d).exists()]
        need = len([d for d in days if not bronze_path(DEFAULT_LAKE, d).exists()])
        progress(f"trades-core: {len(days)} day(s) to build, {need} need a download, "
                 f"{args.workers} in parallel, tickers {' '.join(tickers)}")
        if need > 200 and not args.confirm:
            raise FatalError(f"{need} downloads planned; re-run with --confirm.")
        # 🚨 REFUSE TO START WITHOUT ROOM FOR EVERY WORKER. A disk that fills
        # mid-CSV-extract leaves a truncated file that parses into a short day;
        # the silver volume check would catch it, but only after the damage.
        free = shutil.disk_usage(DEFAULT_LAKE).free
        want = args.workers * (ZIP_BYTES_PER_DAY + CSV_BYTES_PER_DAY) * 1.2
        if free < want:
            raise FatalError(f"{human_bytes(free)} free, {human_bytes(want)} needed for "
                             f"{args.workers} workers; lower --workers.")
        ok = failed = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(build_trades_core_one, client, d, DEFAULT_LAKE, tickers): d
                    for d in days}
            for f in as_completed(futs):
                d = futs[f]
                try:
                    out(f"  {d.isoformat()}  {f.result()}")
                    ok += 1
                except Exception as e:                    # one bad day must not stop 700
                    out(f"  {d.isoformat()}  FAILED: {e}")
                    failed += 1
                done = ok + failed
                if done % 10 == 0 or done == len(days):
                    el = time.time() - t0
                    progress(f"  -- {done}/{len(days)} days, {el / 60:.0f} min elapsed, "
                             f"~{el / done * (len(days) - done) / 60:.0f} min left")
        print(f"trades-core complete: {ok} ok, {failed} failed"
              + ("  (re-run the same range to retry; finished days are skipped)" if failed else ""))

    elif args.command == "silver-build":
        for d in trading_days(args.start, args.end or args.start):
            progress(f"Building silver bars for {d.isoformat()}...")
            build_silver_one(DEFAULT_LAKE, d)
        print("Silver build complete!")

    elif args.command == "gex-daily-build":
        if not api_key: raise FatalError("API Key required.")
        client = Client(api_key)
        for tk in (t.upper() for t in args.tickers):
            progress(f"Fetching /greek-exposure {tk} (timeframe={args.timeframe}) ...")
            try:
                res = build_greek_exposure_one(client, tk, args.timeframe, args.out_dir)
            except (NoDataForDate, FatalError) as e:
                progress(f"  {tk}: skipped ({e})")
                continue
            out(f"  {tk}: {res.rows} days  {res.start} -> {res.end}  ({res.path})")
        print("Daily GEX build complete!")

    elif args.command == "spot-gex-build":
        if not api_key: raise FatalError("API Key required.")
        client = Client(api_key)
        days = trading_days(args.start, args.end or args.start)
        tickers = [t.upper() for t in args.tickers]
        total = len(days) * len(tickers)
        progress(f"spot-GEX: {len(days)} trading days x {len(tickers)} tickers = {total} requests")
        if total > 200 and not args.confirm:
            raise FatalError(f"{total} requests planned; re-run with --confirm.")
        done = skipped = 0
        bad = set()
        for d in days:
            for tk in tickers:
                if tk in bad:
                    skipped += 1
                    continue
                try:
                    n = build_spot_gex_one(client, tk, d, DEFAULT_LAKE)
                except (NoDataForDate, FatalError) as e:
                    progress(f"  {tk}: dropping from run ({e})")
                    bad.add(tk); skipped += 1
                    continue
                if n is None: skipped += 1
                else: done += 1
            progress(f"  {d.isoformat()}: wrote={done} skipped={skipped}")
        print(f"Spot GEX build complete! wrote {done}, skipped {skipped}"
              + (f"  (dropped: {', '.join(sorted(bad))})" if bad else ""))

    elif args.command == "ohlc-build":
        if not api_key: raise FatalError("API Key required.")
        client = Client(api_key)
        days = trading_days(args.start, args.end or args.start)
        tickers = [t.upper() for t in args.tickers]
        total = len(days) * len(tickers)
        progress(f"OHLC ({args.candle}): up to {len(days)} days x {len(tickers)} tickers = {total} requests "
                 "(already-present dates are skipped per ticker)")
        if total > 200 and not args.confirm:
            raise FatalError(f"{total} requests planned; re-run with --confirm.")
        skipped = []
        for tk in tickers:
            progress(f"  {tk}: checking gaps ...")
            try:
                res = build_ohlc_one(client, tk, args.candle, days, args.out_dir)
            except (NoDataForDate, FatalError) as e:
                out(f"  {tk}: SKIPPED — not served by /ohlc/{args.candle} ({e})")
                skipped.append(tk)
                continue
            if res is None:
                out(f"  {tk}: up to date / no data returned")
            else:
                out(f"  {tk}: +{res.fetched_days} days -> {res.rows} rows  {res.start} -> {res.end}  ({res.path})")
        print(f"OHLC build complete!" + (f"  (skipped: {', '.join(skipped)})" if skipped else ""))

    elif args.command == "netprem-build":
        if not api_key: raise FatalError("API Key required.")
        client = Client(api_key)
        days = trading_days(args.start, args.end or args.start)
        tickers = [t.upper() for t in args.tickers]
        total = len(days) * len(tickers)
        progress(f"net-prem-ticks: up to {len(days)} days x {len(tickers)} tickers = {total} requests "
                 "(already-present dates skipped per ticker; 403 = older than UW's ~2yr window)")
        if total > 200 and not args.confirm:
            raise FatalError(f"{total} requests planned; re-run with --confirm.")
        skipped = []
        for tk in tickers:
            progress(f"  {tk}: checking gaps ...")
            try:
                res = build_netprem_one(client, tk, days, args.out_dir)
            except (NoDataForDate, FatalError) as e:
                out(f"  {tk}: SKIPPED ({e})")
                skipped.append(tk)
                continue
            if res is None:
                out(f"  {tk}: up to date / no data in range")
            else:
                out(f"  {tk}: +{res.fetched_days} days -> {res.rows} rows  {res.start} -> {res.end}  ({res.path})")
        print("net-prem-ticks build complete!" + (f"  (skipped: {', '.join(skipped)})" if skipped else ""))

    return 0

if __name__ == "__main__":
    sys.exit(main())