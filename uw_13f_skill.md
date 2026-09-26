# Institutional 13F Data Skill

You are an expert analyst for institutional 13F filing data via the Unusual Whales API. Your primary data source is **live API calls**, but there are opportunities to persist data locally. The local SQLite database is an accelerant for CIK look-ups, cross-institutional analytics, and pre-fetched data analysis, allowing you to handle analytics queries, data pipeline tasks, API exploration, and DB maintenance.

## Authentication

Load the token from a `.env` file, an environment variable, or directly from the user:

```python
headers = {
    "Accept": "application/json, text/plain",
    "Authorization": f"Bearer YOUR_TOKEN",
}
```

## API Endpoint Reference

Base URL: `https://api.unusualwhales.com`

All responses return `{"data": [...]}`. Max `limit` per page is 500.

| Endpoint              | Method & URL                              | Path Param                   | Query Params                                                                                                                                                                                                        |
| --------------------- | ----------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **List Institutions** | `GET /api/institutions`                   | —                            | `name`, `tags[]`, `min_total_value`, `max_total_value`, `min_share_value`, `max_share_value`, `order`, `order_direction`, `limit`, `page`                                                                           |
| **Latest Filings**    | `GET /api/institutions/latest_filings`    | —                            | `name`, `date`, `order`, `order_direction`, `limit`, `page`                                                                                                                                                         |
| **Activity (v2)**     | `GET /api/institution/{cik}/activity/v2`  | `cik` (10-digit zero-padded) | `start_date`, `end_date` (YYYY-MM-DD, filter on `report_date`), `ticker_symbol` (comma-separated, prefix `-` to exclude), `order` (`units` \| `units_change`), `order_direction` (`asc` \| `desc`), `limit`, `page` |
| **Holdings**          | `GET /api/institution/{cik}/holdings`     | `cik`                        | `date`, `security_types`, `limit`, `page`, `order`, `order_direction`                                                                                                                                               |
| **Sectors**           | `GET /api/institution/{cik}/sectors`      | `cik`                        | `date`, `limit`                                                                                                                                                                                                     |
| **Ownership**         | `GET /api/institution/{ticker}/ownership` | `ticker` (e.g. `"AAPL"`)     | `date`, `tags`, `order`, `order_direction`, `limit`, `page`                                                                                                                                                         |

> **CRITICAL**: Activity, Holdings, and Sectors allow the institution's name (e.g. `"Berkshire Hathaway Inc"`) or its **CIK** (e.g. `"0001067983"`) to be used for the `{name}` path param. Strongly prefer the 10-digit zero-padded CIK string over the institution name, which is prone to typos and difficult character handling.

> **Parameter policy**: Only send query params that are explicitly needed. Do not send params with default values — the API may return errors when it receives unexpected defaults.

## Data Quality Patterns

The API has several data quirks. Apply these transformations when processing response JSON:

### 1. Dirty Integer Casting

The API returns numeric values as string floats (e.g. `"123456.0"`). Convert via float first:

```python
DIRTY_INT_COLS = [
    "share_value", "fund_value", "total_value", "call_value", "put_value",
    "warrant_value", "pfd_value", "debt_value", "share_holdings", "call_holdings",
    "put_holdings", "warrant_holdings", "fund_holdings", "pfd_holdings", "debt_holdings",
]

def clean_int(val):
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

for record in records:
    for col in DIRTY_INT_COLS:
        record[col] = clean_int(record.get(col))
```

### 2. Date Handling

Some date strings from the API may be empty or malformed. Guard before parsing:

```python
from datetime import date

def safe_date(val):
    if not val:
        return None
    try:
        return date.fromisoformat(val)
    except ValueError:
        return None
```

## Local SQLite Reference

The Unusual Whales API tracks **~9,000 institutional 13F filers**. A one-time bulk load into a local SQLite database is highly recommended because:

- **CIK lookup is the most common operation** — you have an institution name and need its 10-digit CIK to call Activity, Holdings, or Sectors. A local `WHERE name LIKE ?` query is instant vs. paginating through the full API.
- **Browsing 9,000 records via paginated API calls is impractical** — at 500 per page, that's 18+ requests just to scan the full list.
- **Cross-institution analytics are only possible locally** — aggregations, filtering by hedge fund status, ranking by AUM, and tag-based filtering all require the full dataset in one place.
- **The data changes slowly** — institutions file 13F quarterly, so a local DB stays current with infrequent refreshes.
- **Tagged institutions surface curated watchlists** — Unusual Whales tags ~1,500 institutions with strategy and style labels. Querying by tags locally lets you instantly filter to the institutions that matter most (see [Filter by tags](#filter-by-tags-local-db) for the full list of available tags).

**Path**: `institutions_13f.db` (or wherever the user chooses)

```python
import sqlite3
conn = sqlite3.connect("institutions_13f.db")
conn.row_factory = sqlite3.Row  # access columns by name
cursor = conn.execute("SELECT * FROM institutions")
rows = cursor.fetchall()
conn.close()
```

### `institutions` table (CIK is primary key)

```sql
CREATE TABLE IF NOT EXISTS institutions (
    cik TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT,
    date TEXT,
    description TEXT,
    tags TEXT,            -- JSON array, e.g. '["activist","tiger_club"]'
    filing_date TEXT,
    website TEXT,
    founder_img_url TEXT,
    logo_url TEXT,
    is_hedge_fund INTEGER,  -- 0 or 1
    name_changes TEXT,      -- JSON array
    people TEXT,            -- JSON array
    share_value INTEGER,
    fund_value INTEGER,
    total_value INTEGER,
    call_value INTEGER,
    put_value INTEGER,
    warrant_value INTEGER,
    pfd_value INTEGER,
    debt_value INTEGER,
    share_holdings INTEGER,
    call_holdings INTEGER,
    put_holdings INTEGER,
    warrant_holdings INTEGER,
    fund_holdings INTEGER,
    pfd_holdings INTEGER,
    debt_holdings INTEGER,
    buy_value REAL,
    sell_value REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

Recommended indexes: `filing_date`, `total_value`, `name`

### `institution_activity` table (composite key: cik + ticker + security_type + put_call + report_date)

```sql
CREATE TABLE IF NOT EXISTS institution_activity (
    cik                TEXT NOT NULL,
    ticker             TEXT NOT NULL,
    report_date        TEXT NOT NULL,
    filing_date        TEXT,
    security_type      TEXT,
    put_call           TEXT,
    units              INTEGER,
    units_change       INTEGER,
    shares_outstanding INTEGER,
    close              REAL,
    avg_price          REAL,
    buy_price          REAL,
    sell_price         REAL,
    price_on_filing    REAL,
    price_on_report    REAL,
    updated_at         TEXT DEFAULT CURRENT_TIMESTAMP
);
```

Recommended indexes: `cik`, `report_date`, `ticker`, `(cik, report_date)`

### When to use local DB vs live API

- **Local DB**: CIK lookups by name, cross-institution analytics, pre-fetched activity analysis, counting/aggregation over known data
- **Live API**: Fresh data, specific date queries, endpoints not stored locally (holdings, sectors, ownership, latest filings)

## Pagination Pattern

Use this loop for any paginated endpoint (all except Sectors which has only `limit`):

```python
import time, requests

records = []
page = 0
while True:
    params = {"limit": 500, "page": page}
    # Add other params only if explicitly needed
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        break
    records.extend(data)
    page += 1
    time.sleep(0.5)  # Rate limiting
```

## Common Analytical Patterns

### Resolve institution name to CIK (local DB)

```python
conn = sqlite3.connect("institutions_13f.db")
rows = conn.execute(
    "SELECT cik, name, total_value FROM institutions WHERE name LIKE ? ORDER BY total_value DESC",
    ["%morgan stanley%"]
).fetchall()
conn.close()
```

> Note: SQLite `LIKE` is case-insensitive for ASCII by default.

### Top N institutions by AUM (local DB)

```python
conn = sqlite3.connect("institutions_13f.db")
rows = conn.execute(
    "SELECT name, cik, total_value FROM institutions WHERE total_value IS NOT NULL ORDER BY total_value DESC LIMIT 10"
).fetchall()
conn.close()
```

### Classify institutional positioning in a ticker (live API)

The Ownership endpoint returns every institution with a footprint in a given ticker. Key response fields:

- `units` — current share count (0 = position closed)
- `units_changed` — change from prior quarter
- `first_buy` — the quarter-end date the position was first opened

Use `tags[]` to narrow the universe to curated firms (activists, specialists, Tiger cubs, etc.) instead of the full ~9,000 institutions. The `hedge_fund` tag is excluded here because it matches ~1,300 institutions and would flood results — include it when you want broad coverage rather than focused signal:

```python
tags = [
    "known", "activist", "value_investor", "public_companies",
    "biotech", "tiger_club", "technology", "small_cap",
    "credit", "13d_activist", "energy", "event",
    "real_estate", "esg",
]
resp = requests.get(
    f"{base}/api/institution/DELL/ownership",
    headers=headers,
    params={"tags[]": tags},
)
records = resp.json().get("data", [])
```

Then classify each institution's positioning into four buckets:

```python
quarter_end = "2024-09-30"  # the report date for the quarter you're analyzing

closed   = [r for r in records if r["units"] == 0]
new      = [r for r in records if r["units"] > 0 and r["first_buy"] == quarter_end]
added    = [r for r in records if r["units_changed"] > 0 and r["first_buy"] != quarter_end]
held     = [r for r in records if r["units_changed"] == 0 and r["first_buy"] != quarter_end]
```

| Bucket   | Meaning                                             |
| -------- | --------------------------------------------------- |
| `closed` | Exited — sold entire position this quarter          |
| `new`    | Initiated — first-ever buy was this quarter         |
| `added`  | Increased — added shares to a pre-existing position |
| `held`   | Static — no change to an existing position          |

This pattern surfaces conviction signals: who is building, who is exiting, and who just initiated. An empty `held` list (as seen with DELL in Q3 2024) indicates a polarizing name — every tagged institution that owned it either added or sold.

### Filter by tags (local DB)

Unusual Whales curates ~1,500 tagged institutions. Tags are stored as JSON arrays, so use `LIKE` for filtering:

```python
conn = sqlite3.connect("institutions_13f.db")

# All activists, ranked by AUM
activists = conn.execute(
    "SELECT name, cik, tags, total_value FROM institutions WHERE tags LIKE '%activist%' ORDER BY total_value DESC"
).fetchall()

# Institutions with ANY tag (the curated watchlist)
tagged = conn.execute(
    "SELECT name, cik, tags, total_value FROM institutions WHERE tags IS NOT NULL AND tags != '[]' ORDER BY total_value DESC"
).fetchall()

conn.close()
```

Available tags: `hedge_fund`, `known`, `activist`, `value_investor`, `public_companies`, `biotech`, `tiger_club` (note: typo in data, refers to Tiger cubs), `technology`, `small_cap`, `credit`, `13d_activist`, `energy`, `event`, `real_estate`, `esg`.

### Filter by tags (live API)

The API also supports tag filtering directly, useful when you don't have a local DB:

```python
resp = requests.get(f"{base}/api/institutions", headers=headers, params={"tags[]": "tiger_club", "limit": 50})
```

## Tactical Holdings Analysis (local DB + live API)

This end-to-end example shows how to go from ambiguous user references like "Druckenmiller, Tepper, and Alkeon" to actionable portfolio-level insight using the local SQLite database and the Holdings endpoint.

### Step 1: Resolve ambiguous references to CIK

Users rarely provide a CIK or exact fund name. They reference institutions by **person name** ("Druckenmiller"), **fund name** ("Alkeon"), or a mix of both. The challenge is that person names and fund names live in different columns:

| User says       | Actual fund name              | Matched via                  |
| --------------- | ----------------------------- | ---------------------------- |
| "Druckenmiller" | DUQUESNE FAMILY OFFICE LLC    | `people` column              |
| "Tepper"        | APPALOOSA LP                  | `people` column              |
| "Alkeon"        | ALKEON CAPITAL MANAGEMENT LLC | `name` / `short_name` column |

A search against `name` alone would miss Druckenmiller and Tepper entirely. A search against `people` alone would miss Alkeon. You need a combined approach.

#### Combined name + people search

Search all three columns (`name`, `short_name`, `people`) and deduplicate by CIK:

```python
import sqlite3

def resolve_institutions(db_path, queries):
    """Resolve a list of ambiguous institution references to CIK identifiers.

    Searches name, short_name, and people columns to handle both
    fund-name references ('Alkeon') and person-name references ('Druckenmiller').

    Returns a dict mapping each query to its match(es).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    results = {}

    for q in queries:
        pattern = f"%{q}%"
        rows = conn.execute(
            """
            SELECT DISTINCT cik, name, short_name, people, total_value
            FROM institutions
            WHERE name LIKE ?
               OR short_name LIKE ?
               OR people LIKE ?
            ORDER BY total_value DESC
            """,
            [pattern, pattern, pattern],
        ).fetchall()
        results[q] = [dict(r) for r in rows]

    conn.close()
    return results
```

Usage:

```python
matches = resolve_institutions("institutions_13f.db", ["Druckenmiller", "Tepper", "Alkeon"])
```

#### Present results for user confirmation

Always present resolved CIKs for the user to confirm before making API calls. Multiple matches are possible (e.g., searching "Morgan" could match Morgan Stanley, JP Morgan, and others):

```python
for query, hits in matches.items():
    if len(hits) == 0:
        print(f"  '{query}': no match found — try a broader search term")
    elif len(hits) == 1:
        h = hits[0]
        print(f"  '{query}' -> {h['name']} (CIK: {h['cik']})")
    else:
        print(f"  '{query}': multiple matches — please clarify:")
        for h in hits:
            print(f"    - {h['name']} (CIK: {h['cik']}, AUM: ${h['total_value']:,})")
```

For the example query `["Druckenmiller", "Tepper", "Alkeon"]`, this produces:

```
  'Druckenmiller' -> DUQUESNE FAMILY OFFICE LLC (CIK: 0001536411)
  'Tepper' -> APPALOOSA LP (CIK: 0001656456)
  'Alkeon' -> ALKEON CAPITAL MANAGEMENT LLC (CIK: 0001230239)
```

Each resolves to exactly one match, so no disambiguation is needed — proceed with these CIKs.

> **Why confirm?** CIK resolution is the foundation for every subsequent API call. A wrong CIK means you silently analyze the wrong fund. The confirmation step costs nothing and prevents compounding errors downstream.

### Step 2: Fetch holdings for each institution

Use the Holdings endpoint with the confirmed CIKs. Order by `value` descending to get the largest positions first:

```python
import time, requests

def fetch_holdings(cik, date, headers):
    """Fetch full holdings for an institution on a given report date."""
    url = f"https://api.unusualwhales.com/api/institution/{cik}/holdings"
    params = {"date": date, "order": "value", "order_direction": "desc", "limit": 500}
    records, page = [], 0
    while True:
        params["page"] = page
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        records.extend(data)
        page += 1
        time.sleep(0.5)
    return records
```

Usage for the three resolved CIKs:

```python
date = "2025-12-31"
ciks = {
    "Duquesne (Druckenmiller)": "0001536411",
    "Appaloosa (Tepper)":       "0001656456",
    "Alkeon":                   "0001230239",
}

portfolios = {}
for label, cik in ciks.items():
    portfolios[label] = fetch_holdings(cik, date, headers)
    print(f"  {label}: {len(portfolios[label])} positions")
```

#### Key response fields

| Field                        | Type   | What it tells you                                                                 |
| ---------------------------- | ------ | --------------------------------------------------------------------------------- |
| `units` / `units_change`     | int    | Current size and QoQ delta. `units == 0` means position closed                    |
| `value`                      | int    | Current market value of the position                                              |
| `perc_of_share_value`        | float  | Position weight in the portfolio (concentration signal)                           |
| `historical_units`           | int[8] | **8-quarter position trajectory** — index 0 is current quarter, index 7 is oldest |
| `avg_price` / `close`        | str    | Cost basis vs. current price                                                      |
| `change_perc`                | str    | QoQ % change in units (`"-1.0"` = fully closed, `null` = new position)            |
| `first_buy`                  | date   | Quarter the position was first opened                                             |
| `security_type` / `put_call` | str    | `"Share"`, `"Fund"`, or `"Option"` with `"call"`/`"put"`                          |
| `sector`                     | str    | Sector classification (`null` for ETFs/Funds)                                     |

### Step 3: Classify portfolio changes

Bucket every holding into one of five categories to surface the actionable narrative:

```python
def classify_holdings(holdings, report_date):
    """Classify holdings into new, closed, added, trimmed, and held buckets."""
    new, closed, added, trimmed, held = [], [], [], [], []

    for h in holdings:
        if h["units"] == 0:
            closed.append(h)
        elif h["first_buy"] == report_date:
            new.append(h)
        elif h["units_change"] > 0:
            added.append(h)
        elif h["units_change"] < 0:
            trimmed.append(h)
        else:
            held.append(h)

    return {"new": new, "closed": closed, "added": added, "trimmed": trimmed, "held": held}
```

| Bucket    | Meaning                                             | Sort by                                          |
| --------- | --------------------------------------------------- | ------------------------------------------------ |
| `new`     | Initiated this quarter (`first_buy == report_date`) | `value` desc — largest new bets first            |
| `closed`  | Fully exited (`units == 0`)                         | `abs(units_change)` desc — biggest exits first   |
| `added`   | Increased existing position                         | `change_perc` desc — highest conviction adds     |
| `trimmed` | Reduced but still holding                           | `change_perc` asc — deepest trims first          |
| `held`    | Unchanged from prior quarter                        | `perc_of_share_value` desc — largest static bets |

> **Tip — spot coordinated themes**: When multiple closed positions share a sector or `first_buy` date, that signals a deliberate thematic exit, not individual name decisions. For example, Tepper's Q4 2025 filing shows 6 regional banks (KEY, CFG, CMA, WAL, ZION, TFC) all initiated in Q3 2025 and all closed in Q4 — a clear tactical trade that ran its course.

### Step 4: Extract conviction signals from `historical_units`

The `historical_units` array is the single most insight-dense field in the response. It contains 8 quarters of position sizing in chronological order (index 0 = current, index 7 = oldest), letting you classify an investor's _behavior pattern_ for each position:

```python
def classify_trajectory(hist):
    """Classify the 8-quarter trajectory of a position.

    Returns one of: 'building', 'harvesting', 'new_conviction',
    'volatile', 'steady', or 'closing'.
    """
    active = [h for h in hist if h > 0]
    if not active:
        return "closing"

    quarters_held = len(active)
    current, previous = hist[0], hist[1] if len(hist) > 1 else 0

    if current == 0:
        return "closing"
    if quarters_held <= 2 and current > previous:
        return "new_conviction"  # Recently initiated and still building
    if quarters_held <= 2:
        return "new_conviction"

    # Check if monotonically increasing or decreasing over active quarters
    diffs = [hist[i] - hist[i + 1] for i in range(len(active) - 1)]
    increases = sum(1 for d in diffs if d > 0)
    decreases = sum(1 for d in diffs if d < 0)

    if increases >= len(diffs) * 0.7:
        return "building"     # Consistently adding over multiple quarters
    if decreases >= len(diffs) * 0.7:
        return "harvesting"   # Consistently trimming over multiple quarters
    if max(hist) > 2 * min(active):
        return "volatile"     # Large swings — trading around a core position
    return "steady"
```

#### Trajectory patterns to highlight for the user

| Pattern            | Example (Tepper Q4 2025)                      | Signal                                                                    |
| ------------------ | --------------------------------------------- | ------------------------------------------------------------------------- |
| **building**       | TSM: `[1130K, 1060K, 1025K, 270K, 250K, ...]` | Multi-quarter accumulation — growing conviction                           |
| **harvesting**     | BABA: `[5138K, 6450K, 7067K, 9230K, ...]`     | Systematically reducing — taking profits or losing conviction             |
| **new_conviction** | EWY: `[1875K, 0, 0, 0, ...]`                  | Brand new large bet — highest-signal for fresh thesis                     |
| **volatile**       | MU: `[1500K, 500K, 825K, 400K, 1200K, ...]`   | Trading around the name — size swings indicate tactical, not buy-and-hold |

#### Combine everything into a portfolio summary

```python
for label, holdings in portfolios.items():
    buckets = classify_holdings(holdings, date)
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    for bucket_name in ["new", "closed", "added", "trimmed"]:
        items = buckets[bucket_name]
        if not items:
            continue
        print(f"\n  {bucket_name.upper()} ({len(items)} positions):")
        for h in sorted(items, key=lambda x: abs(x["units_change"] or 0), reverse=True)[:5]:
            traj = classify_trajectory(h["historical_units"])
            chg = f"{float(h['change_perc']):+.0%}" if h["change_perc"] else "new"
            print(f"    {h['ticker']:6s} {chg:>8s}  ${h['value']:>13,}  [{traj}]  {h['full_name']}")
```

This produces a concise, scannable summary per institution — bucket, size, conviction trajectory — that tells you exactly where each investor is putting capital to work and where they're pulling it back.

## Gotchas

1. **CIK format**: Always 10-digit zero-padded string (`"0001067983"`), never an integer
2. **Activity response lacks CIK**: The activity endpoint does NOT include the CIK in its response — you must inject it from the request parameter
3. **`put_call` field**: `null` for shares/funds, `"put"` or `"call"` for options positions
4. **Ownership uses ticker**: The ownership endpoint takes a `{ticker}` path param (e.g. `"AAPL"`), not a CIK — it is the only endpoint that works this way
5. **Parameter policy**: Only send query params when explicitly provided — do not send defaults, as the API may return errors
6. **Verified CIKs for testing**: `0001067983` (Berkshire Hathaway), `0001536411` (Duquesne / Druckenmiller), `0001656456` (Appaloosa / Tepper), `0001230239` (Alkeon Capital). These are concentrated portfolios that return quickly — avoid large bank filers (Morgan Stanley, BofA) for casual testing as they have thousands of holdings requiring many paginated requests
7. **SQLite array columns**: `tags`, `name_changes`, and `people` are stored as JSON strings — use `json.loads()` to parse them back to lists
