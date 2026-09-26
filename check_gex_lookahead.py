"""
check_gex_lookahead.py
======================

The backtest maps one GEX row per calendar day onto every minute of that day and
uses its sign (net_gex > 0 -> POSITIVE_GEX branch). If that row was computed with
information only known at/after that day's close, the backtest has lookahead.

This script answers "is that a real problem for my results?" three ways:

  1. SCHEMA   - is the file date-only (a standard daily series) or does it carry
                an intraday timestamp?
  2. PERSISTENCE - how often does sign(net_gex) change from one day to the next?
                bot_runner only uses the sign. If the sign is the same on day D
                and day D-1 ~95%+ of the time, then using D vs D-1 barely changes
                the regime label, so same-day-vs-prior-day lookahead can't be
                materially inflating results through the regime split.
  3. LIVE DRIFT (optional, --live) - pull UW /greek-exposure now; tells you the
                field shape and whether "today" already has a row. Run it again
                later in the session and after the close to see if today's
                net_gex value moves intraday (= the series is updated live and
                whatever moment the historical parquet was pulled is baked in).

DEFINITIVE follow-up (do this regardless): in true_options_simulator.py, shift
gex_history so each trading day uses the PREVIOUS day's GEX, re-run, and diff the
matrix. Small change => lookahead immaterial. Big change => use the shifted
(prior-day) series as canonical. See --emit-shift-snippet.

Usage:
    python check_gex_lookahead.py SPY
    python check_gex_lookahead.py SPY --parquet historical/GEXSPY.parquet
    python check_gex_lookahead.py SPY --file ~/Downloads/gamma_exposure_example_data.csv
    python check_gex_lookahead.py SPY --live
    python check_gex_lookahead.py --emit-shift-snippet
"""

import os
import sys
import argparse
import pandas as pd


SHIFT_SNIPPET = '''
# --- true_options_simulator.py / true_options_trade_auditor.py ---
# Replace the same-day GEX map with the PRIOR trading day's value (lookahead-free).
# In load_local_gex(), instead of:
#     return dict(zip(df_gex['date'], df_gex[gex_col]))
# do:
    df_gex = df_gex.sort_values('date').reset_index(drop=True)
    df_gex['gex_effective'] = df_gex[gex_col].shift(1)      # yesterday's close GEX
    df_gex = df_gex.dropna(subset=['gex_effective'])
    return dict(zip(df_gex['date'], df_gex['gex_effective']))
# Re-run the simulator and diff the matrix against the un-shifted run.
'''


def load_gex(ticker, parquet, file):
    path = file or parquet or f"historical/GEX{ticker}.parquet"
    if not os.path.exists(path):
        print(f"  file not found: {path}")
        return None, path
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    return df, path


def pick_gex_col(df):
    for col in ("net_gex", "total_net_gex"):
        if col in df.columns:
            return col
    if "call_gex" in df.columns and "put_gex" in df.columns:
        df["net_gex"] = df["call_gex"] + df["put_gex"]
        return "net_gex"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker", nargs="?", default="SPY")
    ap.add_argument("--parquet", help="explicit path to GEX parquet")
    ap.add_argument("--file", help="explicit path to a GEX file (csv or parquet)")
    ap.add_argument("--live", action="store_true", help="also pull UW /greek-exposure now")
    ap.add_argument("--ws", action="store_true", help="watch the UW gex: WS channel to see if the bot's GEX moves intraday")
    ap.add_argument("--ws-seconds", type=int, default=120, help="how long to watch the gex: channel (default 120)")
    ap.add_argument("--emit-shift-snippet", action="store_true")
    args = ap.parse_args()

    if args.emit_shift_snippet:
        print(SHIFT_SNIPPET)
        return

    df, path = load_gex(args.ticker, args.parquet, args.file)
    if df is None:
        sys.exit(1)

    print("=" * 74)
    print(f"  1. SCHEMA  ({path})")
    print("=" * 74)
    print(f"  rows: {len(df)}")
    print(f"  columns: {list(df.columns)}")
    print(f"  dtypes:\n{df.dtypes.to_string()}")
    time_cols = [c for c in df.columns if any(k in c for k in ("time", "timestamp", "hour", "minute", "datetime"))]
    date_cols = [c for c in df.columns if "date" in c and c not in time_cols]
    print(f"\n  date-like columns: {date_cols or '(none)'}")
    print(f"  intraday time columns: {time_cols or '(NONE -> pure daily series, one row per trading day)'}")

    dcol = date_cols[0] if date_cols else df.columns[0]
    parsed = pd.to_datetime(df[dcol], utc=True, errors="coerce")
    if parsed.notna().any():
        has_intraday = (parsed.dt.hour.fillna(0).ne(0) | parsed.dt.minute.fillna(0).ne(0)).any()
        print(f"  '{dcol}' carries a non-midnight time component: {has_intraday}")
        print(f"  date range: {parsed.min()}  ->  {parsed.max()}")
    print("\n  head:\n" + df.head(3).to_string())
    print("\n  tail:\n" + df.tail(3).to_string())

    gcol = pick_gex_col(df)
    if not gcol:
        print("\n  could not identify a net GEX column; stopping.")
        sys.exit(1)

    # order by date ascending
    df["_d"] = pd.to_datetime(df[dcol], utc=True, errors="coerce")
    df = df.sort_values("_d").dropna(subset=["_d"]).reset_index(drop=True)
    s = df[gcol].astype(float)

    print("\n" + "=" * 74)
    print(f"  2. REGIME PERSISTENCE  (column: {gcol!r}, n={len(s)})")
    print("=" * 74)
    sign = s.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    same = (sign == sign.shift(1)).iloc[1:]
    flip_days = (~same).sum()
    pct_same = 100.0 * same.mean() if len(same) else float("nan")
    pos = (sign > 0).mean() * 100
    print(f"  POSITIVE_GEX days: {pos:.0f}%    NEGATIVE_GEX days: {100 - pos:.0f}%")
    print(f"  sign(net_gex) unchanged day-over-day: {pct_same:.1f}%  ({flip_days} flips in {len(same)} transitions)")

    # how much would using yesterday's GEX change the regime label?
    prev_sign = sign.shift(1)
    disagree = (sign != prev_sign).iloc[1:]
    print(f"\n  If the backtest used PRIOR-day GEX instead of same-day:")
    print(f"    regime label would differ on {disagree.sum()} / {len(disagree)} days "
          f"({100.0*disagree.mean():.1f}%).")
    if disagree.mean() < 0.05:
        print("    => < 5%. Any same-day lookahead in this file is immaterial to the")
        print("       regime split. Safe to proceed; still worth the shift-and-rerun once.")
    else:
        print("    => materially different. Re-run the simulator with prior-day GEX")
        print("       (python check_gex_lookahead.py --emit-shift-snippet) and treat")
        print("       that as canonical.")

    # magnitude drift near zero = sign is fragile
    near_zero = (s.abs() < 0.1 * s.abs().median()).sum()
    print(f"\n  days with |net_gex| < 10% of median (sign is fragile there): {near_zero}")

    if args.live:
        print("\n" + "=" * 74)
        print("  3. LIVE UW /greek-exposure")
        print("=" * 74)
        try:
            import requests
            from dotenv import load_dotenv
            load_dotenv()
            key = os.getenv("UW_API_KEY")
            r = requests.get(
                f"https://api.unusualwhales.com/api/stock/{args.ticker.upper()}/greek-exposure",
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                timeout=15,
            )
            print(f"  HTTP {r.status_code}")
            data = r.json().get("data", []) if r.status_code == 200 else []
            if data:
                print(f"  {len(data)} rows. newest 3:")
                for row in data[-3:]:
                    print("   ", row)
                print("\n  -> Compare the newest row's date to today. If it's already")
                print("     present mid-session, re-run this in an hour and after the")
                print("     close: if today's net_gex value moves, the series is")
                print("     updated intraday and the parquet baked in one snapshot.")
        except Exception as e:
            print(f"  live pull failed: {e}")

    if args.ws:
        watch_ws_gex(args.ticker, args.ws_seconds)


def watch_ws_gex(ticker, seconds):
    """Subscribe to the UW gex: channel and see whether the regime value the
    LIVE bot uses (gamma_per_one_percent_move_oi) actually moves intraday.

    - value changes materially over the window  -> it's an at-spot recompute,
      the bot IS regime-aware intraday (and the daily backtest can't model that).
    - value is static / one push per day         -> the bot's 'intraday' GEX is
      really a daily figure, same as the backtest.
    """
    print("\n" + "=" * 74)
    print(f"  4. LIVE WS gex:{ticker}  ({seconds}s)  -- field: gamma_per_one_percent_move_oi")
    print("=" * 74)
    try:
        import json as _json
        import time as _time
        import threading as _threading
        import websocket
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("UW_API_KEY")
    except Exception as e:
        print(f"  cannot start WS: {e}")
        return

    seen = []

    def on_open(ws):
        ws.send(_json.dumps({"channel": f"gex:{ticker.upper()}", "msg_type": "join"}))
        print(f"  joined gex:{ticker.upper()} ... watching\n")

    def on_message(ws, msg):
        try:
            p = _json.loads(msg)
            if isinstance(p, list) and len(p) >= 2 and str(p[0]).startswith("gex:"):
                d = p[1]
                g = d.get("gamma_per_one_percent_move_oi")
                seen.append((d.get("timestamp") or d.get("time"), g))
                print(f"  {d.get('timestamp') or d.get('time')}  gamma_per_1%_oi = {g}")
        except Exception:
            pass

    ws = websocket.WebSocketApp(f"wss://api.unusualwhales.com/socket?token={key}",
                               on_open=on_open, on_message=on_message)
    th = _threading.Thread(target=ws.run_forever, daemon=True)
    th.start()
    _time.sleep(seconds)
    ws.close()
    _time.sleep(1)

    vals = [float(v) for _, v in seen if v not in (None, "")]
    if len(vals) < 2:
        print(f"\n  only {len(vals)} readings -- inconclusive (market closed? quiet ticker? "
              f"try a longer --ws-seconds during RTH).")
        return
    lo, hi = min(vals), max(vals)
    signs = {1 if v > 0 else (-1 if v < 0 else 0) for v in vals}
    spread = (hi - lo) / (abs(sum(vals) / len(vals)) or 1)
    print(f"\n  {len(vals)} readings | min {lo:.4g}  max {hi:.4g}  (range = {spread*100:.1f}% of mean)")
    print(f"  distinct signs seen: {signs}")
    if len(signs) > 1:
        print("  -> regime FLIPPED during the window. Live is genuinely intraday-regime-aware;")
        print("     the daily backtest cannot represent this.")
    elif spread > 0.05:
        print("  -> value moves intraday (at-spot recompute) but sign held. Live reacts to")
        print("     spot; the daily backtest is a coarser approximation.")
    else:
        print("  -> value barely moved. The bot's 'live' GEX is effectively a daily figure,")
        print("     same granularity as the backtest.")


if __name__ == "__main__":
    main()
