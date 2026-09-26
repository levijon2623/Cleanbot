# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "httpx", "python-dotenv"]
# ///
"""
backfill_flow30.py
==================
30-SECOND NET-PREMIUM SERIES for the book's candidate days, from the trade tape.

WHY THIS EXISTS
    UW's /net-prem-ticks serves ONE MINUTE only -- every granularity parameter
    returns the identical 405 rows at 60s spacing. So any sub-minute question
    (does the whale accelerate or die into the close of the entry candle?) has
    to be rebuilt from the trade tape.

WHY IT IS TRUSTWORTHY NOW, AND WAS NOT BEFORE
    A first attempt inferred the aggressor from price-vs-quote and scored a
    median pearson of 0.673 against UW's own 1m series -- a DIFFERENT quantity,
    which would have answered a question about our classifier rather than about
    the whale. The tape already carries UW's classification in `tags`
    (ask_side / bid_side / mid_side). Using the published label instead of
    re-deriving it scores 0.959, with several days landing exactly on 1.000.
    mid_side prints (~7% of volume) are treated as ZERO -- tested against
    splitting them and leaning them ask-ward, both of which scored worse.
    (METHODOLOGY 1: a second, drifting copy of logic that already exists.)

STORAGE, AND WHY THE BUCKETS ARE DECOMPOSED
    A day's bronze parquet is ~1 GB and the extraction working set ~8 GB, so the
    tape is DELETED after each day is extracted. Only the derived 30s series is
    kept -- ~90 KB per session, ~13 MB for the whole book.

    Because the tape does not survive, each bucket stores SUFFICIENT STATISTICS
    rather than a finished number: calls-lifted, calls-hit, puts-lifted,
    puts-hit. A signed net and a two-sided gross would have looked adequate and
    silently foreclosed every directional question -- the first thing asked of
    this data -- at the cost of a full re-download.

RESUMABLE, AND ORDERED BY VALUE
    Days already extracted are skipped, so this can be killed and restarted.
    Days are processed in DESCENDING candidate count, so a partial run is still
    a usable sample rather than an arbitrary calendar slice.

RECONCILIATION IS STORED, NOT ENFORCED -- AND UW'S OWN SERIES IS KEPT
    The 30s buckets are re-summed to 1m and correlated against UW's series for
    that ticker-day, but that score is INFORMATION, not a gate, because the
    day is the wrong unit to judge. On QQQ 2026-09-15 the day scores 0.488;
    drop the ten worst minutes and the remaining 381 correlate at 1.000. The
    error is not diffuse -- it is a handful of minutes holding same-second,
    same-size call+put synthetics and multi-strike packages (one 09:41 combo
    booked $54.5M against UW's $2.3M), which UW excludes from a DIRECTIONAL net
    premium and we do not.

    Those combos cannot be identified structurally: aggregated_trade_id is
    unpopulated, and a same-second-multi-contract rule flags 100% of prints in
    names this liquid. A size cut would work but is disqualified on its face --
    this study is ABOUT large prints, so a magnitude filter would delete the
    subject.

    So UW's 1m series is saved next to the 30s buckets (uw1m.parquet). A
    consumer can then check fidelity AT THE MINUTES IT ACTUALLY MEASURES rather
    than discarding a session over minutes it never looks at. That filter
    selects on fidelity, not on outcome: which minutes are triggers was decided
    by UW's series in the backtester, so a phantom tape spike elsewhere cannot
    manufacture one.

Usage:
  python backfill_flow30.py --plan            # build the day list first
  python backfill_flow30.py --run             # process, resumable
  python backfill_flow30.py --run --limit 5   # a few days, to sanity-check
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

CACHE = "_flow30_cache"
PLAN = os.path.join(CACHE, "plan.json")
BRONZE = "lake/bronze/full-tape"
WORK = "lake/_work"
TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "META", "AVGO", "SMH", "GLD", "MSFT"]
API = "https://api.unusualwhales.com/api"


def _headers():
    from dotenv import load_dotenv
    load_dotenv(".env", encoding="utf-8-sig")
    k = os.getenv("UW_API_KEY")
    if not k:
        sys.exit("  UW_API_KEY not in env")
    return {"Authorization": f"Bearer {k}", "Accept": "application/json",
            "User-Agent": "cleanbot-flow30/1.0", "UW-CLIENT-API-ID": "100003"}


# ---------------------------------------------------------------- planning
def build_plan():
    import sim_core
    import directional_flow_backtester as D
    cnt = {}
    for rule in sim_core.research_rules(include_paper=True):
        try:
            cand = sim_core.build_candidates(D, rule)
        except Exception as e:
            print(f"    {rule['name']}: {type(e).__name__}")
            continue
        for d, m, _ in cand:
            cnt[d.isoformat()] = cnt.get(d.isoformat(), 0) + 1
        print(f"    {rule['name']} done", flush=True)
    plan = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    os.makedirs(CACHE, exist_ok=True)
    json.dump(plan, open(PLAN, "w"), indent=1)
    print(f"\n  {len(plan)} candidate days -> {PLAN}")
    print(f"  top 10: {plan[:10]}")
    print(f"  total candidates {sum(cnt.values()):,}")


# ---------------------------------------------------------------- extraction
def uw_1m(h, tk, d):
    import httpx
    try:
        r = httpx.get(f"{API}/stock/{tk}/net-prem-ticks", headers=h,
                      params={"date": d}, timeout=60)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    rows = r.json().get("data", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    t = pd.to_datetime(df["tape_time"], utc=True, errors="coerce").dt.tz_convert(
        "America/New_York")
    df["mod"] = t.dt.hour * 60 + t.dt.minute
    for c in ("net_call_premium", "net_put_premium"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df["net_call_premium"] - df["net_put_premium"]).groupby(df["mod"]).sum()


def extract_day(d, h):
    """-> (DataFrame of 30s buckets, {ticker: reconciliation pearson})"""
    p = os.path.join(BRONZE, f"{d}.parquet")
    if not os.path.exists(p):
        return None, {}
    df = (pl.scan_parquet(p)
          .filter(pl.col("underlying_symbol").is_in(TICKERS))
          .select("underlying_symbol", "executed_at", "price", "size",
                  "premium", "option_type", "tags")
          .collect().to_pandas())
    if df.empty:
        return None, {}
    t = pd.to_datetime(df["executed_at"], utc=True).dt.tz_convert("America/New_York")
    df["mod"] = t.dt.hour * 60 + t.dt.minute
    df["half"] = (t.dt.second >= 30).astype(int)          # H1=0, H2=1
    tg = df["tags"].astype(str)
    # UW's OWN label -- do not re-derive it from price vs quote
    sgn = np.where(tg.str.contains("ask_side", na=False), 1.0,
                   np.where(tg.str.contains("bid_side", na=False), -1.0, 0.0))
    prem = pd.to_numeric(df["premium"], errors="coerce").fillna(
        pd.to_numeric(df["price"], errors="coerce")
        * pd.to_numeric(df["size"], errors="coerce") * 100).to_numpy()
    # Store the FOUR COMPONENTS, not a derived total. The tape is deleted after
    # extraction, so anything not decomposable now needs a re-download later --
    # and `net` and `gross` alone cannot answer a directional question (gross
    # pools both sides; net cancels them). From ca/cb/pa/pb everything is
    # recoverable: net = (ca-cb)-(pa-pb), gross = ca+cb+pa+pb, bullish-directional
    # premium = ca+pb, bearish = cb+pa.
    is_call = (df["option_type"] == "call").to_numpy()
    ask, bid = (sgn > 0), (sgn < 0)
    df["ca"] = prem * (is_call & ask)         # calls lifted
    df["cb"] = prem * (is_call & bid)         # calls hit
    df["pa"] = prem * (~is_call & ask)        # puts lifted
    df["pb"] = prem * (~is_call & bid)        # puts hit
    df["midp"] = prem * (sgn == 0)            # unsigned, kept for completeness
    g = (df.groupby(["underlying_symbol", "mod", "half"])
           .agg(ca=("ca", "sum"), cb=("cb", "sum"), pa=("pa", "sum"),
                pb=("pb", "sum"), midp=("midp", "sum"), n=("ca", "size"))
           .reset_index())
    g["net"] = (g["ca"] - g["cb"]) - (g["pa"] - g["pb"])
    g["gross"] = g["ca"] + g["cb"] + g["pa"] + g["pb"]
    g["date"] = d

    recon, uw = {}, []
    for tk in TICKERS:
        sub = g[g["underlying_symbol"] == tk]
        if sub.empty:
            continue
        mine = sub.groupby("mod")["net"].sum()
        theirs = uw_1m(h, tk, d)
        if theirs is None:
            continue
        uw.append(pd.DataFrame({"underlying_symbol": tk,
                                "mod": theirs.index.astype(int),
                                "uw_net": theirs.to_numpy(), "date": d}))
        j = pd.concat([theirs.rename("u"), mine.rename("m")], axis=1).dropna()
        j = j[(j.index >= 570) & (j.index <= 960)]
        recon[tk] = float(j["u"].corr(j["m"])) if len(j) >= 60 else float("nan")
    return g, recon, (pd.concat(uw, ignore_index=True) if uw else None)


def fetch_uw_only(d, h):
    """UW's 1m series for one date, WITHOUT touching the tape.

    Days extracted before uw1m.parquet existed need only this -- re-downloading
    a gigabyte of tape to fetch 405 rows per ticker would be absurd.
    """
    uw = []
    for tk in TICKERS:
        s = uw_1m(h, tk, d)
        if s is not None:
            uw.append(pd.DataFrame({"underlying_symbol": tk,
                                    "mod": s.index.astype(int),
                                    "uw_net": s.to_numpy(), "date": d}))
    return pd.concat(uw, ignore_index=True) if uw else None


def purge(d, keep=False):
    """Delete the day's tape. `keep` protects sessions that were ALREADY on disk.

    Ten bronze sessions predate this script and are the sample every other study
    validates against -- deleting them to save space we are not short of would
    be an expensive way to reclaim 10 GB. Only tape THIS run downloaded is
    removed; the working directory is always cleared, since it is scratch.
    """
    if not keep:
        try:
            os.remove(os.path.join(BRONZE, f"{d}.parquet"))
        except OSError:
            pass
    if os.path.isdir(WORK):
        for f in os.listdir(WORK):
            try:
                fp = os.path.join(WORK, f)
                shutil.rmtree(fp) if os.path.isdir(fp) else os.remove(fp)
            except OSError:
                pass


def run(limit, min_recon):
    if not os.path.exists(PLAN):
        sys.exit("  no plan -- run with --plan first")
    plan = json.load(open(PLAN))
    h = _headers()
    os.makedirs(CACHE, exist_ok=True)
    mpath = os.path.join(CACHE, "manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    def paths(d):
        o = os.path.join(CACHE, f"date={d}")
        return os.path.join(o, "flow30.parquet"), os.path.join(o, "uw1m.parquet")

    # A day is DONE only if both artifacts exist; days extracted before
    # uw1m.parquet existed are repaired from the API alone, no tape download.
    repair = [d for d, _ in plan
              if man.get(d, {}).get("status") == "ok"
              and os.path.exists(paths(d)[0]) and not os.path.exists(paths(d)[1])]
    todo = [(d, n) for d, n in plan
            if d not in man or (man[d].get("status") == "ok"
                                and not os.path.exists(paths(d)[0]))]
    if limit:
        todo = todo[:limit]
    if repair:
        print(f"  repairing {len(repair)} extracted days (uw1m only, "
              f"no download)", flush=True)
        for j, d in enumerate(repair, 1):
            u = fetch_uw_only(d, h)
            if u is not None:
                pl.from_pandas(u).write_parquet(paths(d)[1])
            if j % 10 == 0 or j == len(repair):
                print(f"      {j}/{len(repair)}", flush=True)
    print(f"\n  {len(man)} days in manifest, {len(todo)} to process "
          f"(of {len(plan)} planned)\n", flush=True)

    t0 = time.time()
    for i, (d, n) in enumerate(todo, 1):
        el = time.time() - t0
        eta = (el / max(i - 1, 1)) * (len(todo) - i + 1) if i > 1 else 0
        print(f"  [{i}/{len(todo)}] {d}  ({n} candidates)  "
              f"elapsed {el/60:.0f}m  eta {eta/60:.0f}m", flush=True)
        keep = os.path.exists(os.path.join(BRONZE, f"{d}.parquet"))
        try:
            if keep:
                print("      tape already on disk -- not downloading, "
                      "not deleting", flush=True)
            else:
                r = subprocess.run([sys.executable, "uw_options_data_lake.py",
                                    "build", d, "--confirm"],
                                   capture_output=True, text=True, timeout=3600)
                if r.returncode != 0:
                    man[d] = {"status": "build_failed",
                              "err": (r.stderr or r.stdout or "")[-300:]}
                    print(f"      build failed: {man[d]['err'][:120]}", flush=True)
                    purge(d, keep)
                    json.dump(man, open(mpath, "w"), indent=1)
                    continue
            g, recon, uw = extract_day(d, h)
            if g is None or g.empty:
                man[d] = {"status": "no_rows"}
            else:
                out = os.path.join(CACHE, f"date={d}")
                os.makedirs(out, exist_ok=True)
                pl.from_pandas(g).write_parquet(os.path.join(out, "flow30.parquet"))
                if uw is not None:
                    pl.from_pandas(uw).write_parquet(os.path.join(out, "uw1m.parquet"))
                ok = {k: v for k, v in recon.items()
                      if np.isfinite(v) and v >= min_recon}
                man[d] = {"status": "ok", "rows": int(len(g)),
                          "candidates": n, "recon": recon,
                          "tickers_ok": sorted(ok)}
                med = np.nanmedian(list(recon.values())) if recon else float("nan")
                print(f"      {len(g):,} buckets, median recon {med:.3f}, "
                      f"{len(ok)}/{len(recon)} tickers >= {min_recon}", flush=True)
        except Exception as e:
            man[d] = {"status": "error", "err": f"{type(e).__name__}: {e}"[:300]}
            print(f"      ERROR {man[d]['err'][:120]}", flush=True)
        finally:
            purge(d, keep)
            json.dump(man, open(mpath, "w"), indent=1)

    ok = [d for d, v in man.items() if v.get("status") == "ok"]
    print(f"\n  done: {len(ok)} sessions extracted, manifest -> {mpath}")


# ---------------------------------------------------------------- coverage
def coverage(min_recon):
    """How many CANDIDATES -- not days -- sit on tape that reconciles?

    "151 days extracted" is the wrong acceptance number. Reconciliation is a
    property of the TICKER-DAY, and a candidate belongs to one ticker, so a
    session can land with eight good tickers and still lose its densest rule.
    2026-03-09 is the case in point: the single highest-candidate day in the
    book (85) reconciles on only 3 of 9 tickers, with MSFT at 0.051 -- not a
    scale error but an unrelated series.

    Writes candidate_index.json ((date, ticker) -> count), which the H1/H2 test
    needs anyway to know which candidates it is allowed to use.
    """
    import sim_core
    import directional_flow_backtester as D
    idx = {}
    for rule in sim_core.research_rules(include_paper=True):
        tk = rule["ticker"]
        try:
            cand = sim_core.build_candidates(D, rule)
        except Exception:
            continue
        for dt, _m, _p in cand:
            idx[f"{dt.isoformat()}|{tk}"] = idx.get(f"{dt.isoformat()}|{tk}", 0) + 1
    json.dump(idx, open(os.path.join(CACHE, "candidate_index.json"), "w"), indent=1)

    mpath = os.path.join(CACHE, "manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    tot = extracted = usable = 0
    per_tk = {}
    for key, n in idx.items():
        d, tk = key.split("|")
        tot += n
        e = per_tk.setdefault(tk, [0, 0, 0])
        e[0] += n
        m = man.get(d)
        if not m or m.get("status") != "ok":
            continue
        extracted += n
        e[1] += n
        r = m.get("recon", {}).get(tk)
        if r is not None and np.isfinite(r) and r >= min_recon:
            usable += n
            e[2] += n
    print(f"  candidates          {tot:,}")
    print(f"  on extracted days   {extracted:,} ({extracted/tot*100:.0f}%)")
    print(f"  ALSO reconciling    {usable:,} ({usable/tot*100:.0f}% of book, "
          f"{usable/max(extracted,1)*100:.0f}% of what has landed)\n")
    print(f"  {'ticker':8} {'cand':>7} {'extracted':>10} {'usable':>8} {'keep':>7}")
    for tk, (a, b, c) in sorted(per_tk.items(), key=lambda kv: -kv[1][0]):
        print(f"  {tk:8} {a:>7,} {b:>10,} {c:>8,} "
              f"{(c/b*100 if b else 0):>6.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-recon", type=float, default=0.90)
    a = ap.parse_args()
    if a.plan:
        build_plan()
    elif a.run:
        run(a.limit, a.min_recon)
    elif a.coverage:
        coverage(a.min_recon)
    else:
        print("  pass --plan, --run or --coverage")


if __name__ == "__main__":
    main()
