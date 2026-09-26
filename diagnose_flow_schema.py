"""
diagnose_flow_schema.py
=======================

One-shot diagnostic for the open question in bot_runner's flow logic:

    Does Unusual Whales' net-prem-ticks feed return a RUNNING TOTAL
    (cumulative day flow as of each tick) or a PER-TICK INCREMENT
    (the net premium that printed in that interval)?

It matters because bot_runner does `cumulative_flow[ticker] += tick.net_premium`
for every unseen tick, and seed_daily_cumulative_flow() sums the whole day.
  - If the feed is INCREMENTS  -> summing is correct.
  - If the feed is RUNNING TOTALS -> summing double-counts the entire day and
    the bot is seeded above every min_flow threshold before it trades.

Usage:
    python diagnose_flow_schema.py                      # REST test, SPY + NVDA
    python diagnose_flow_schema.py SPY NVDA AAPL        # REST test, custom tickers
    python diagnose_flow_schema.py SPY --date 2026-08-27
    python diagnose_flow_schema.py SPY --raw            # also dump a raw record
    python diagnose_flow_schema.py SPY --save flow.json # save raw response
    python diagnose_flow_schema.py SPY --ws --ws-seconds 90   # also sample the live WS

Reads UW_API_KEY from .env (same as the bot).
"""

import os
import sys
import json
import time
import argparse
import statistics
from dotenv import load_dotenv
import requests

BASE_URL = "https://api.unusualwhales.com"

# Thresholds the bot actually compares cumulative flow against (from config.py)
TYPICAL_MIN_FLOW = 3_500_000


# ----------------------------------------------------------------------------
# Field detection
# ----------------------------------------------------------------------------
def detect_fields(record: dict):
    """Find the call-premium / put-premium (or single net-premium) keys."""
    keys = list(record.keys())
    lower = {k.lower(): k for k in keys}

    def find(*needles):
        for lk, k in lower.items():
            if all(n in lk for n in needles):
                return k
        return None

    call_key = find("call", "prem")
    put_key = find("put", "prem")
    net_key = None
    if not (call_key and put_key):
        # look for a single already-netted premium field
        for lk, k in lower.items():
            if "prem" in lk and "call" not in lk and "put" not in lk:
                net_key = k
                break
    time_key = (find("tape", "time") or find("time") or find("date")
                or lower.get("timestamp"))
    return call_key, put_key, net_key, time_key


def net_series(data, call_key, put_key, net_key):
    out = []
    for rec in data:
        try:
            if net_key:
                out.append(float(rec.get(net_key, 0) or 0))
            else:
                nc = float(rec.get(call_key, 0) or 0)
                np_ = float(rec.get(put_key, 0) or 0)
                out.append(nc - np_)
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def call_put_series(data, call_key, put_key):
    cs, ps = [], []
    for rec in data:
        try:
            cs.append(float(rec.get(call_key, 0) or 0) if call_key else 0.0)
            ps.append(float(rec.get(put_key, 0) or 0) if put_key else 0.0)
        except (TypeError, ValueError):
            cs.append(0.0)
            ps.append(0.0)
    return cs, ps


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def sign_change_frac(seq):
    if len(seq) < 3:
        return 0.0
    changes = 0
    comparisons = 0
    prev = None
    for v in seq:
        s = (v > 0) - (v < 0)
        if s == 0:
            continue
        if prev is not None:
            comparisons += 1
            if s != prev:
                changes += 1
        prev = s
    return changes / comparisons if comparisons else 0.0


def nondecreasing_frac(seq):
    if len(seq) < 2:
        return 1.0
    up = sum(1 for a, b in zip(seq, seq[1:]) if b >= a - 1e-9)
    return up / (len(seq) - 1)


def fmt(n):
    return f"{n:,.0f}"


def analyze(label, data, call_key, put_key, net_key):
    print("=" * 78)
    print(f"  {label}   ({len(data)} ticks)")
    print("=" * 78)

    if not data:
        print("  No data returned.\n")
        return None

    s = net_series(data, call_key, put_key, net_key)
    cs, ps = call_put_series(data, call_key, put_key)
    diffs = [b - a for a, b in zip(s, s[1:])]

    total = sum(s)
    last = s[-1]
    first = s[0]
    max_abs = max(abs(v) for v in s)
    mean_abs = statistics.fmean(abs(v) for v in s)

    s_scf = sign_change_frac(s)
    d_scf = sign_change_frac(diffs)
    call_nondecr = nondecreasing_frac(cs) if call_key else None
    call_all_nonneg = all(v >= -1e-6 for v in cs) if call_key else None
    diff_sum = sum(diffs)

    ratio_total_maxabs = abs(total) / max(max_abs, 1.0)

    print(f"  net (call-put) series:")
    print(f"    first tick .............. {fmt(first)}")
    print(f"    last  tick .............. {fmt(last)}")
    print(f"    max |tick| ............. {fmt(max_abs)}")
    print(f"    mean |tick| ........... {fmt(mean_abs)}")
    print(f"    SUM of all ticks ...... {fmt(total)}     <- what the bot currently seeds")
    print(f"    |SUM| / max|tick| ..... {ratio_total_maxabs:,.1f}x")
    print(f"    sign-change frac (s) .. {s_scf:.2f}   (near 0 = smooth curve, ~0.5 = oscillates)")
    print(f"    sign-change frac (Δs).. {d_scf:.2f}   (per-tick change direction)")
    print(f"    Σ(Δs) vs (last-first).. {fmt(diff_sum)} vs {fmt(last - first)}")
    if call_key:
        print(f"  raw {call_key!r}:")
        print(f"    first / last .......... {fmt(cs[0])} / {fmt(cs[-1])}")
        print(f"    non-decreasing frac .. {call_nondecr:.2f}   (near 1.0 = accumulating = running total)")
        print(f"    always >= 0 .......... {call_all_nonneg}")

    # ------------------------------------------------------------------
    # Weighted verdict
    # ------------------------------------------------------------------
    running_signals = []
    increment_signals = []

    if ratio_total_maxabs > 5:
        running_signals.append(f"|SUM| is {ratio_total_maxabs:.0f}x the largest single tick "
                               f"(a sum of increments ≈ a few big ticks, not 50x+)")
    else:
        increment_signals.append(f"|SUM| is only {ratio_total_maxabs:.1f}x the largest single tick "
                                 f"(consistent with summing independent increments)")

    if s_scf < 0.15:
        running_signals.append(f"net series barely changes sign ({s_scf:.2f}) -> it's a curve, not a delta stream")
    elif s_scf > 0.30:
        increment_signals.append(f"net series changes sign often ({s_scf:.2f}) -> looks like per-interval deltas")

    if call_key and call_nondecr is not None:
        if call_nondecr > 0.90 and call_all_nonneg:
            running_signals.append(f"{call_key!r} is monotonically non-decreasing ({call_nondecr:.2f}) and never negative "
                                   f"-> classic accumulator")
        elif call_nondecr < 0.75:
            increment_signals.append(f"{call_key!r} frequently decreases ({call_nondecr:.2f}) -> per-interval values")

    if abs(total) > 3_000_000_000:
        running_signals.append(f"|SUM| = ${abs(total):,.0f} exceeds a plausible single-name day "
                               f"(a whole S&P name rarely nets > ~$2-3B/day)")

    if abs(last) > 50_000_000 and abs(last) > 3 * mean_abs:
        running_signals.append(f"last tick alone (${abs(last):,.0f}) is already threshold-sized "
                               f"and >> mean tick -> last tick looks like a running total")

    print()
    print("  RUNNING-TOTAL signals:")
    for x in running_signals or ["  (none)"]:
        print(f"    • {x}")
    print("  INCREMENT signals:")
    for x in increment_signals or ["  (none)"]:
        print(f"    • {x}")

    verdict = "UNCLEAR"
    if len(running_signals) >= 2 and len(running_signals) > len(increment_signals):
        verdict = "RUNNING_TOTAL"
    elif len(increment_signals) >= 2 and len(increment_signals) > len(running_signals):
        verdict = "INCREMENT"

    print()
    print(f"  >>> VERDICT for {label}: {verdict}")

    if verdict == "RUNNING_TOTAL":
        print(f"      seed_daily_cumulative_flow SHOULD return the LAST tick ({fmt(last)}),")
        print(f"      NOT the sum ({fmt(total)}). The live loop's `cumulative_flow += net_premium`")
        print(f"      is also wrong for this feed -- it should track the latest value, not add.")
    elif verdict == "INCREMENT":
        print(f"      Current summing logic is CORRECT. Full-day net ≈ {fmt(total)}")
        print(f"      (vs typical min_flow {fmt(TYPICAL_MIN_FLOW)}).")
    else:
        print(f"      Inspect the raw records below by hand. Compare a couple of consecutive")
        print(f"      ticks: if tick N+1 ≈ tick N + small change, it's a running total.")

    return {"label": label, "verdict": verdict, "sum": total, "last": last,
            "max_abs": max_abs, "n": len(data)}


# ----------------------------------------------------------------------------
# REST
# ----------------------------------------------------------------------------
def fetch_rest(ticker, headers, date=None):
    url = f"{BASE_URL}/api/stock/{ticker.upper()}/net-prem-ticks"
    params = {}
    if date:
        params["date"] = date
    r = requests.get(url, headers=headers, params=params, timeout=15)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for {ticker}: {r.text[:400]}")
        return None
    body = r.json()
    if isinstance(body, dict):
        return body.get("data", body.get("chains", [])), body
    return body, {"data": body}


# ----------------------------------------------------------------------------
# WebSocket (optional) -- tests the live path the bot actually uses
# ----------------------------------------------------------------------------
def sample_ws(tickers, api_key, seconds):
    try:
        import websocket  # websocket-client, already a bot dependency
    except ImportError:
        print("  websocket-client not installed; skipping --ws test.")
        return {}

    print("=" * 78)
    print(f"  LIVE WEBSOCKET SAMPLE  ({seconds}s, channels net_flow:<ticker>)")
    print("=" * 78)
    print("  NOTE: WS uses field names net_call_prem / net_put_prem")
    print("        REST uses net_call_premium / net_put_premium\n")

    collected = {t: [] for t in tickers}
    url = f"wss://api.unusualwhales.com/socket?token={api_key}"

    def on_open(ws):
        for t in tickers:
            ws.send(json.dumps({"channel": f"net_flow:{t}", "msg_type": "join"}))
        print(f"  joined net_flow for {', '.join(tickers)} ... collecting\n")

    def on_message(ws, msg):
        try:
            payload = json.loads(msg)
            if isinstance(payload, list) and len(payload) >= 2:
                ch, data = str(payload[0]), payload[1]
                if ch.startswith("net_flow:"):
                    tk = data.get("ticker") or data.get("symbol") or ch.split(":", 1)[1]
                    if tk in collected:
                        collected[tk].append(data)
        except Exception:
            pass

    ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message)
    import threading
    th = threading.Thread(target=ws.run_forever, daemon=True)
    th.start()
    time.sleep(seconds)
    ws.close()
    time.sleep(1)

    results = {}
    for t, msgs in collected.items():
        if len(msgs) < 3:
            print(f"  {t}: only {len(msgs)} messages -- not enough to judge "
                  f"(market closed? quiet name? try longer --ws-seconds)")
            continue
        if msgs and msgs[0].get("net_call_premium") is not None and msgs[0].get("net_call_prem") is None:
            ck, pk = "net_call_premium", "net_put_premium"
        else:
            ck, pk = "net_call_prem", "net_put_prem"
        print(f"  {t}: {len(msgs)} messages, fields: {sorted(msgs[0].keys())}")
        r = analyze(f"WS net_flow:{t}", msgs, ck, pk, None)
        if r:
            results[t] = r
    return results


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", default=["SPY", "NVDA"], help="tickers to test (default: SPY NVDA)")
    ap.add_argument("--date", help="YYYY-MM-DD (defaults to today's session on UW's side)")
    ap.add_argument("--raw", action="store_true", help="print a couple of raw records verbatim")
    ap.add_argument("--save", metavar="FILE", help="save the raw JSON response(s) to this file")
    ap.add_argument("--file", metavar="FILE", help="analyze a previously --save'd JSON instead of hitting the API")
    ap.add_argument("--ws", action="store_true", help="also sample the live WebSocket net_flow feed")
    ap.add_argument("--ws-seconds", type=int, default=60, help="how long to sample the WS (default 60)")
    args = ap.parse_args()

    api_key = None
    if not args.file:
        load_dotenv()
        api_key = os.getenv("UW_API_KEY")
        if not api_key:
            print("UW_API_KEY not found in .env")
            sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"} if api_key else {}
    tickers = [t.upper() for t in args.tickers]

    saved = {}
    summaries = []

    # --- offline mode: analyze a saved file ---
    if args.file:
        with open(args.file) as f:
            blob = json.load(f)
        # accept either {"SPY": {"data": [...]}} (from --save) or a bare {"data":[...]} or a bare list
        if isinstance(blob, list):
            blob = {"FILE": {"data": blob}}
        elif "data" in blob and all(not isinstance(v, dict) for v in blob.values()):
            blob = {"FILE": blob}
        for tk, body in blob.items():
            data = body.get("data", []) if isinstance(body, dict) else body
            if not data:
                print(f"{tk}: no data array"); continue
            ck, pk, nk, time_key = detect_fields(data[0])
            print(f"\n{tk}: fields -> call={ck!r} put={pk!r} net={nk!r} time={time_key!r}")
            if args.raw:
                for rec in data[:2] + data[-2:]:
                    print("   ", json.dumps(rec, default=str))
            r = analyze(f"FILE {tk}", data, ck, pk, nk)
            if r:
                summaries.append(r)
        _print_overall(summaries)
        return

    for tk in tickers:
        res = fetch_rest(tk, headers, args.date)
        if not res:
            continue
        data, body = res
        saved[tk] = body

        if not data:
            print(f"\n{tk}: response had no 'data' array. Top-level keys: {list(body.keys())}\n")
            continue

        call_key, put_key, net_key, time_key = detect_fields(data[0])
        print(f"\n{tk}: detected fields -> call={call_key!r} put={put_key!r} "
              f"net={net_key!r} time={time_key!r}")
        print(f"     all keys in a record: {sorted(data[0].keys())}")

        if args.raw:
            print("\n  --- first 2 records ---")
            for rec in data[:2]:
                print("   ", json.dumps(rec, indent=2, default=str))
            print("  --- last 2 records ---")
            for rec in data[-2:]:
                print("   ", json.dumps(rec, indent=2, default=str))
            if time_key:
                print(f"\n  time span: {data[0].get(time_key)}  ->  {data[-1].get(time_key)}")

        s = analyze(f"REST net-prem-ticks {tk}", data, call_key, put_key, net_key)
        if s:
            summaries.append(s)
        print()

    if args.ws:
        ws_res = sample_ws(tickers, api_key, args.ws_seconds)
        for r in ws_res.values():
            summaries.append(r)

    if args.save and saved:
        with open(args.save, "w") as f:
            json.dump(saved, f, indent=2, default=str)
        print(f"Raw response(s) written to {args.save}")

    _print_overall(summaries)


def _print_overall(summaries):
    print("\n" + "#" * 78)
    print("  OVERALL")
    print("#" * 78)
    verdicts = {s["verdict"] for s in summaries}
    for s in summaries:
        print(f"  {s['label']:<32} {s['verdict']:<14} "
              f"sum={fmt(s['sum'])}  last={fmt(s['last'])}")
    print()
    if verdicts == {"INCREMENT"}:
        print("  All feeds look like INCREMENTS. bot_runner's summing is correct.")
        print("  You can relax / remove the $2B sanity guard in seed_daily_cumulative_flow.")
    elif verdicts == {"RUNNING_TOTAL"}:
        print("  All feeds look like RUNNING TOTALS. This is the bug.")
        print("  Fix: seed_daily_cumulative_flow -> return last tick's net value.")
        print("       get_live_net_premium consumers -> track latest value, don't += each tick.")
        print("       (FlowMomentumTracker already takes a level, so feed it s[-1] directly.)")
    else:
        print(f"  Mixed / unclear verdicts: {verdicts}")
        print("  Re-run with --raw and eyeball consecutive ticks before changing the math.")
    print()


if __name__ == "__main__":
    main()
