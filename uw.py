# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv", "pandas"]
# ///
"""
uw.py
=====
ASK UNUSUAL WHALES A QUESTION FROM THE TERMINAL, WITHOUT WRITING A SCRIPT.

WHY THIS EXISTS
    Four throwaway scripts were written in a single session (2026-09-18) purely
    to wrap the same API: probe_netprem_gran, tune_aggressor, validate_tape_netprem,
    gld_zero_probe. Each re-implemented auth, pagination and the ET timestamp
    conversion. That is not a scripting habit, it is a missing tool.

    Worse, it is the drift METHODOLOGY 1 is about, and it is already at scale
    here: `unusual_whales_client.py` exists and bot_runner uses it, but ~20
    research scripts hand-roll
        {"Authorization": f"Bearer {os.getenv('UW_API_KEY')}"}
    instead. Twenty copies of a header is twenty places for a base URL, a
    header, or a retry policy to go quietly out of date -- exactly how live came
    to be running a cushion rule the research had retired months earlier.

    So `get()` below is importable. A new study should call it rather than
    build copy 21:
        from uw import get
        rows = get("stock/SPY/net-prem-ticks", date="2026-09-18")

WHY IT IS NOT AN MCP SERVER
    Considered and rejected for this workflow. MCP's advantages are typed tool
    discovery and reach into other clients (Desktop); with Remote Control on
    this session and an agent that reads the repo anyway, neither applies. A CLI
    is the same capability with no new process model. Revisit if the tools are
    ever wanted somewhere a terminal is not.

🚨 THE DESIGN RULE: RETURN ANSWERS, NOT DATA
    A single contract-flow query is 20,000 prints. Printing that is useless --
    it buries the answer and, when an agent is reading, costs the context the
    analysis needs. So every command prints a SUMMARY sized for a human or an
    agent, and `--out` writes the full rows to parquet for anything bulk.
    Nothing here ever streams a raw payload to the terminal.

Usage:
  python uw.py quotes GLD260918C00402000 2026-09-18 --at 13:19,13:54
  python uw.py flow SPY 2026-09-18
  python uw.py get stock/SPY/net-prem-ticks date=2026-09-18 --rows 5
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

API = "https://api.unusualwhales.com/api"
_HDRS = None


def _headers():
    global _HDRS
    if _HDRS is None:
        from dotenv import load_dotenv
        load_dotenv(".env", encoding="utf-8-sig")
        k = os.getenv("UW_API_KEY")
        if not k:
            sys.exit("  UW_API_KEY not in env (.env, or export it)")
        _HDRS = {"Authorization": f"Bearer {k}", "Accept": "application/json",
                 "User-Agent": "cleanbot-uw/1.0", "UW-CLIENT-API-ID": "100003"}
    return _HDRS


class UWError(RuntimeError):
    """Raised, never sys.exit -- this module is imported as well as run.

    `get()` originally exited the process on a non-200. That is right for a CLI
    and wrong for a library: the first script to import it died mid-probe on a
    422 it was deliberately provoking. Only main() turns these into exits.
    """


def get(path: str, pages: int = 1, limit: int | None = None, **params):
    """Rows from any UW endpoint. THE one place auth and paging live.

    🚨 PAGING IS VERIFIED BY DEDUPE, NOT ASSUMED (2026-09-18).
        `page=` is ACCEPTED AND SILENTLY IGNORED by at least
        /option-contract/{id}/flow -- pages 0, 10 and 39 return the identical
        500 rows. A loop that trusts it fetches the same rows N times and
        reports N*500 as a row count. Two analyses in one session did exactly
        that and produced a "40,000 prints" figure that was 500 prints repeated.
        So: dedupe on `id` and STOP as soon as a page adds nothing new. That is
        correct whether the endpoint pages properly, ignores the parameter, or
        invents a third behaviour later.

    `capped` on the result reports whether the server truncated: a response
    exactly at the limit means you are looking at a WINDOW, not the full set,
    and for this endpoint that window is the most recent prints.
    """
    import httpx
    if limit is not None:
        params["limit"] = limit
    out, seen = [], set()
    for p in range(max(1, pages)):
        q = dict(params)
        if p:
            q["page"] = p
        try:
            # follow_redirects: httpx defaults to OFF, and UW answers some
            # endpoints (notably option-trades/full-tape) with a 302 to a signed
            # download URL. Without this, a live dataset reports HTTP 302 and
            # reads as "unavailable" -- it did, for four dates, one of which was
            # already sitting on disk.
            r = httpx.get(f"{API}/{path.lstrip('/')}", headers=_headers(),
                          params=q, timeout=60, follow_redirects=True)
        except Exception as e:
            raise UWError(f"request failed: {type(e).__name__}: {e}") from e
        if r.status_code != 200:
            if p == 0:
                raise UWError(f"HTTP {r.status_code} on {path}\n  {r.text[:300]}")
            break
        d = r.json()
        rows = d.get("data", d) if isinstance(d, dict) else d
        if not rows:
            break
        fresh = []
        for row in rows:
            k = row.get("id") if isinstance(row, dict) else None
            k = k if k is not None else repr(row)
            if k not in seen:
                seen.add(k)
                fresh.append(row)
        out += fresh
        if not fresh:                 # the page repeated -- paging is a no-op
            break
        if limit is not None and len(rows) < limit:
            break                     # short page = genuinely the end
    get.capped = bool(limit) and len(out) >= limit
    return out


def _et(df, col="executed_at"):
    """UW stamps UTC; every question here is asked in market time."""
    t = pd.to_datetime(df[col], utc=True, errors="coerce")
    return t.dt.tz_convert("America/New_York")


def _num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _save(df, out):
    if not out:
        return
    (df.to_parquet(out, index=False) if out.endswith(".parquet")
     else df.to_csv(out, index=False))
    print(f"\n  {len(df):,} rows -> {out}")


# ----------------------------------------------------------------- commands
def cmd_quotes(a):
    """NBBO around named minutes for one contract -- the exit forensic.

    Built for the question that cost an afternoon on 2026-09-18: the bot booked
    seven GLD exits at $0.01 on a $0.00 bid, and answering "was the book really
    empty?" meant writing a script. The answer was 40,000 prints with ZERO
    instances of nbbo_bid <= 0. That is one command now.
    """
    rows = get(f"option-contract/{a.contract}/flow", limit=500, date=a.date)
    if not rows:
        sys.exit("  no prints returned (check the contract id and date)")
    df = pd.DataFrame(rows)
    df = _num(df, ["nbbo_bid", "nbbo_ask", "price", "size", "premium"])
    df["et"] = _et(df)
    df["hm"] = df["et"].dt.strftime("%H:%M")
    df = df.dropna(subset=["nbbo_bid"]).sort_values("et")

    z = df["nbbo_bid"] <= 0.0
    print(f"  {a.contract}  {a.date}")
    print(f"  {len(df):,} prints with an NBBO   "
          f"{df['hm'].iloc[0]} - {df['hm'].iloc[-1]} ET")
    if get.capped:
        print(f"  ⚠️  TRUNCATED: the API caps this endpoint at 500 and ignores")
        print(f"      `page`, so this is the most RECENT 500 prints, not the")
        print(f"      session. Minutes outside the span above are unanswered")
        print(f"      here -- absence below is not evidence.")
    print(f"  bid range      {df['nbbo_bid'].min():.2f} - {df['nbbo_bid'].max():.2f}")
    print(f"  nbbo_bid <= 0  {z.sum()} ({z.mean()*100:.2f}%)"
          + (f"   first at {df.loc[z,'hm'].iloc[0]}" if z.any() else "   <- never"))
    if "nbbo_ask" in df:
        sp = (df["nbbo_ask"] - df["nbbo_bid"]).median()
        print(f"  median spread  {sp:.2f}")

    if a.at:
        want = {s.strip() for s in a.at.split(",") if s.strip()}
        sub = df[df["hm"].isin(want)]
        print(f"\n  {'minute':>7} {'bid lo':>8} {'bid hi':>8} {'ask':>8} {'n':>6}")
        for hm in sorted(want):
            g = sub[sub["hm"] == hm]
            if g.empty:
                print(f"  {hm:>7} {'no prints in this minute':>32}")
                continue
            print(f"  {hm:>7} {g['nbbo_bid'].min():>8.2f} {g['nbbo_bid'].max():>8.2f} "
                  f"{g['nbbo_ask'].median():>8.2f} {len(g):>6}")
    _save(df, a.out)


def cmd_flow(a):
    """Net premium per minute for a ticker-day, summarised.

    UW serves this at ONE MINUTE only -- every granularity spelling returns the
    same 405 rows. Sub-minute work has to come from the tape (backfill_flow30).
    """
    rows = get(f"stock/{a.ticker}/net-prem-ticks", date=a.date)
    if not rows:
        sys.exit("  no rows (non-trading day, or the ticker has no flow)")
    df = pd.DataFrame(rows)
    df = _num(df, ["net_call_premium", "net_put_premium"])
    df["et"] = _et(df, "tape_time")
    df["hm"] = df["et"].dt.strftime("%H:%M")
    df["net"] = df["net_call_premium"] - df["net_put_premium"]
    rth = df[(df["et"].dt.hour * 60 + df["et"].dt.minute).between(570, 960)]

    print(f"  {a.ticker}  {a.date}   {len(df)} rows, "
          f"{df['hm'].iloc[0]} - {df['hm'].iloc[-1]} ET")
    print(f"  cumulative net premium  ${rth['net'].sum()/1e6:+,.1f}M  (RTH)")
    print(f"  per-minute  median ${rth['net'].median()/1e3:+,.0f}k   "
          f"p90 |net| ${rth['net'].abs().quantile(0.9)/1e3:,.0f}k")
    big = rth.reindex(rth["net"].abs().sort_values(ascending=False).index).head(a.top)
    print(f"\n  {a.top} largest minutes")
    print(f"  {'minute':>7} {'net $M':>10}")
    for _, r in big.iterrows():
        print(f"  {r['hm']:>7} {r['net']/1e6:>+10.2f}")
    _save(df, a.out)


def cmd_get(a):
    """Any endpoint, row-capped. The escape hatch for exploring.

    Capped on purpose: the point of this tool is to see the SHAPE of a response
    -- field names, row count, spacing -- not to dump it. Use --out for the rows.
    """
    params = {}
    for kv in a.params:
        if "=" not in kv:
            sys.exit(f"  params must be k=v, got {kv!r}")
        k, v = kv.split("=", 1)
        params[k] = v
    rows = get(a.path, pages=a.pages, **params)
    print(f"  {a.path}  ->  {len(rows):,} rows")
    if not rows:
        return
    if isinstance(rows[0], dict):
        print(f"  fields: {sorted(rows[0].keys())}\n")
    df = pd.DataFrame(rows)
    for c in ("tape_time", "executed_at", "start_time"):
        if c in df.columns:
            t = _et(df, c)
            gap = t.sort_values().diff().dt.total_seconds().dropna()
            if not gap.empty:
                print(f"  {c}: modal gap {gap.mode().iloc[0]:.0f}s  "
                      f"min {gap.min():.0f}s  span {t.min():%H:%M} - {t.max():%H:%M} ET\n")
            break
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(df.head(a.rows).to_string(index=False))
    _save(df, a.out)


def main():
    ap = argparse.ArgumentParser(
        description="Query Unusual Whales. Prints summaries; --out writes rows.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quotes", help="NBBO for one contract, optionally at named minutes")
    q.add_argument("contract", help="OCC id, e.g. GLD260918C00402000")
    q.add_argument("date")
    q.add_argument("--at", help="comma-separated ET minutes, e.g. 13:19,13:54")
    # no --pages: this endpoint caps at 500 and ignores `page`. Offering the
    # flag would imply a reach the API does not have.
    q.add_argument("--out")
    q.set_defaults(fn=cmd_quotes)

    f = sub.add_parser("flow", help="net premium per minute for a ticker-day")
    f.add_argument("ticker")
    f.add_argument("date")
    f.add_argument("--top", type=int, default=8)
    f.add_argument("--out")
    f.set_defaults(fn=cmd_flow)

    g = sub.add_parser("get", help="any endpoint, row-capped")
    g.add_argument("path", help="e.g. stock/SPY/net-prem-ticks")
    g.add_argument("params", nargs="*", help="k=v pairs")
    g.add_argument("--rows", type=int, default=5)
    g.add_argument("--pages", type=int, default=1)
    g.add_argument("--out")
    g.set_defaults(fn=cmd_get)

    a = ap.parse_args()
    try:
        a.fn(a)
    except UWError as e:
        sys.exit(f"  {e}")


if __name__ == "__main__":
    main()
