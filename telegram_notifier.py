"""
telegram_notifier.py
====================
Push the directional bot's trades to a Telegram chat + answer a few read-only
commands (/today /all /pnl /open /status /help).

Zero new deps (uses `requests`, already required). Fail-safe: a Telegram outage
never touches the trading path -- sends go through a background queue/worker and
every network call is wrapped.

Wire-up (bot_runner.py):
    from telegram_notifier import TelegramNotifier, load_trades, summarize
    self.tg = TelegramNotifier.from_env()          # None-safe if env not set
    ...
    self.tg.send(...)                              # in log_trade()
    self.tg.start_commands({                       # after the engine is up
        "today":  lambda a: ...,
        "open":   lambda a: ...,
        ...
    })

Env:
    TELEGRAM_BOT_TOKEN   from @BotFather
    TELEGRAM_CHAT_ID     your numeric chat id (message the bot, then hit
                         https://api.telegram.org/bot<token>/getUpdates)
"""
from __future__ import annotations

import html
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone

import requests

API = "https://api.telegram.org/bot{token}/{method}"
_TIMEOUT = 12


# --------------------------------------------------------------------------
# trade-log helpers (module-level so bot_runner handlers can reuse them)
# --------------------------------------------------------------------------
def load_trades(path: str = "bot_executions_log.jsonl") -> list[dict]:
    """Parse the execution log, tolerating hand-edits but never SILENTLY.

    A record dropped without a word makes reporting quietly wrong: a real NVDA
    exit went missing from every P&L summary for a day because someone appended
    a `# Note: ...` to the end of its line. We still never raise -- the bot must
    not die on a bad log line -- but a dropped line now prints.
    """
    out = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    out.append(json.loads(line))
                    continue
                except json.JSONDecodeError:
                    pass
                # recover a trailing hand-note: {...}  # free text
                cut = line.rfind("}")
                if cut > 0:
                    try:
                        rec = json.loads(line[:cut + 1])
                        rec.setdefault("note", line[cut + 1:].lstrip("# \t"))
                        out.append(rec)
                        continue
                    except json.JSONDecodeError:
                        pass
                print(f"  ! {path}:{i} unparseable, record DROPPED: {line[:80]}")
    except FileNotFoundError:
        pass
    return out


def round_trips(entries: list[dict], include_artifacts: bool = False) -> list[dict]:
    """Pair ENTRY -> next EXIT on the same contract. Unpaired ENTRYs are 'open'.

    `sequence_artifact` entries are EXCLUDED by default. Those are trades that
    only exist because an earlier bug closed a position and freed the sequential
    slot -- on 2026-09-15 the zero-bid exit bug turned ONE real AVGO position
    into sixteen ledger round-trips, so a naive sum reported the session as
    19 trades / -311.5% when it was one trade at +35.7%.
    The rows stay in the log: they are real events and the only record that the
    bug happened. They are simply not COUNTED as independent trades. Pass
    include_artifacts=True to see the raw sequence.
    """
    open_by_contract: dict[str, dict] = {}
    trips = []
    for e in entries:
        c = e.get("contract")
        act = e.get("action")
        if act in ("ENTRY_LONG", "ENTRY_SHORT"):
            if e.get("sequence_artifact") and not include_artifacts:
                continue          # its EXIT then finds no open entry and drops too
            open_by_contract[c] = e
        elif act == "EXIT" and c in open_by_contract:
            en = open_by_contract.pop(c)
            trips.append({
                "ticker": e.get("ticker"), "contract": c,
                "regime": e.get("regime") or en.get("regime"),
                "dir": en.get("direction") or ("CALL" if e.get("is_call") else "PUT"),
                "entry_px": en.get("entry_price", en.get("price")),
                "exit_px": e.get("exit_price", e.get("price")),
                "contracts": e.get("contracts") or en.get("contracts") or en.get("size") or 1,
                "pnl_pct": e.get("pnl_pct"), "pnl_dollars": e.get("pnl_dollars"),
                "reason": e.get("reason"), "hold_mins": e.get("hold_mins"),
                "entry_ts": en.get("timestamp"), "exit_ts": e.get("timestamp"),
                "dry_run": e.get("dry_run", en.get("dry_run")),
            })
    for c, en in open_by_contract.items():
        trips.append({"ticker": en.get("ticker"), "contract": c, "regime": en.get("regime"),
                      "dir": en.get("direction"), "entry_px": en.get("entry_price", en.get("price")),
                      "exit_px": None, "pnl_pct": None, "pnl_dollars": None, "reason": "OPEN",
                      "hold_mins": None, "entry_ts": en.get("timestamp"), "exit_ts": None,
                      "dry_run": en.get("dry_run")})
    return trips


def _fmt_trip(t: dict) -> str:
    d = t["dir"] or "?"
    if t["exit_px"] is None:
        return (f"\U0001F7E1 <b>{t['ticker']}</b> {d}  entry ${_n(t['entry_px'])}  "
                f"<i>{t['regime']}</i>  (open)")
    pp, pd_ = t.get("pnl_pct"), t.get("pnl_dollars")
    ic = "✅" if (pp or 0) > 0 else "❌"
    held = f"{t['hold_mins']:.0f}m" if t.get("hold_mins") is not None else "?"
    nc = t.get("contracts") or 1
    qty = f"{nc}x " if nc and nc != 1 else ""
    return (f"{ic} <b>{t['ticker']}</b> {d}  {qty}${_n(t['entry_px'])}→${_n(t['exit_px'])}  "
            f"<b>{_pct(pp)}</b> (${_n(pd_, 0)})  {t['reason']} · {held} · <i>{t['regime']}</i>")


def _n(x, dp=2):
    try:
        return f"{float(x):.{dp}f}"
    except (TypeError, ValueError):
        return "?"


def _pct(x):
    try:
        return f"{float(x):+.1f}%"
    except (TypeError, ValueError):
        return "?"


def summarize(trips: list[dict], title: str = "All trades") -> str:
    closed = [t for t in trips if t["exit_px"] is not None]
    open_ = [t for t in trips if t["exit_px"] is None]
    if not closed and not open_:
        return f"<b>{title}</b>\nno trades logged yet."
    wins = [t for t in closed if (t.get("pnl_pct") or 0) > 0]
    tot_pct = sum(t.get("pnl_pct") or 0 for t in closed)
    tot_usd = sum(t.get("pnl_dollars") or 0 for t in closed)
    lines = [f"<b>{title}</b>  ({len(closed)} closed, {len(open_)} open)"]
    if closed:
        lines.append(f"W/L <b>{len(wins)}/{len(closed) - len(wins)}</b>  "
                     f"({100 * len(wins) / len(closed):.0f}%)   "
                     f"Σ <b>{tot_pct:+.0f}%</b>  (${tot_usd:+,.0f})")
        by_rule: dict[str, list] = {}
        for t in closed:
            by_rule.setdefault(t["regime"] or "?", []).append(t)
        for rn, ts in sorted(by_rule.items(), key=lambda kv: sum(x.get("pnl_dollars") or 0 for x in kv[1])):
            w = sum(1 for x in ts if (x.get("pnl_pct") or 0) > 0)
            lines.append(f"  • {rn}: {w}/{len(ts)}  "
                         f"${sum(x.get('pnl_dollars') or 0 for x in ts):+,.0f}")
        best = max(closed, key=lambda x: x.get("pnl_pct") or -1e9)
        worst = min(closed, key=lambda x: x.get("pnl_pct") or 1e9)
        lines.append(f"best {best['ticker']} {_pct(best.get('pnl_pct'))} · "
                     f"worst {worst['ticker']} {_pct(worst.get('pnl_pct'))}")
    for t in open_:
        lines.append(_fmt_trip(t))
    return "\n".join(lines)


COUNTERFACTUAL_LOG = "bot_executions_counterfactual.jsonl"


def summarize_both(trips: list[dict], cf_trips: list[dict],
                   title: str = "All trades") -> str:
    """Observed summary, with the amended ledger appended when it disagrees.

    The two ledgers answer different questions and must not be blended:
      observed       what the bot DID, including trades a bug caused
      counterfactual what it WOULD have done, replayed from the tape
    So this prints the observed summary in full and adds a short delta block
    only when the amended ledger differs -- if nothing was ever corrupted the
    two are identical and the extra block is noise.

    Divergence is worth surfacing precisely because it is the bug's footprint.
    On 2026-09-15 the zero-bid exit bug turned one real AVGO position into
    sixteen ledger round-trips: observed -311.5%, counterfactual +35.7%.
    """
    base = summarize(trips, title)
    if not cf_trips:
        return base
    cc = [t for t in cf_trips if t["exit_px"] is not None]
    oc = [t for t in trips if t["exit_px"] is not None]
    o_pct = sum(t.get("pnl_pct") or 0 for t in oc)
    c_pct = sum(t.get("pnl_pct") or 0 for t in cc)
    if len(cc) == len(oc) and abs(c_pct - o_pct) < 0.05:
        return base                       # nothing amended; do not clutter
    cw = sum(1 for t in cc if (t.get("pnl_pct") or 0) > 0)
    lines = [base, "",
             f"<i>amended ledger (part fiction)</i>: {len(cc)} closed, "
             f"W/L <b>{cw}/{len(cc) - cw}</b>  Σ <b>{c_pct:+.0f}%</b>",
             f"<i>vs observed</i> {o_pct:+.0f}% over {len(oc)} "
             f"→ <b>{c_pct - o_pct:+.0f}pp</b> across "
             f"<b>{len(cc) - len(oc):+d}</b> trades"]
    return "\n".join(lines)


def _et_date_str():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = str(chat_id)
        self._q: queue.Queue[str] = queue.Queue(maxsize=200)
        self._handlers: dict = {}
        self._offset = 0
        self._stop = threading.Event()
        threading.Thread(target=self._sender, name="tg-send", daemon=True).start()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_env(cls):
        tok, cid = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        if not tok or not cid:
            print("  ℹ️ Telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — notifications OFF")
            return _NullNotifier()
        n = cls(tok, cid)
        n.send("\U0001F916 <b>Directional bot online.</b>  /help for commands.")
        print("  ✅ Telegram notifier armed")
        return n

    # -- outbound ------------------------------------------------------
    def send(self, text: str):
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass

    def trade(self, entry: dict):
        """Format one log_trade() dict as a push."""
        act, tk = entry.get("action"), entry.get("ticker")
        rg = entry.get("regime", "?")
        if act in ("ENTRY_LONG", "ENTRY_SHORT"):
            d = entry.get("direction") or ("CALL" if act == "ENTRY_LONG" else "PUT")
            nf = entry.get("net_flow")
            nfs = f"  flow ${nf/1e6:+.1f}M" if isinstance(nf, (int, float)) else ""
            tag = " <i>[paper]</i>" if entry.get("dry_run") else ""
            nc = entry.get("contracts") or entry.get("size") or 1
            qty = f"{nc}x " if nc and nc != 1 else ""
            prem = entry.get("premium_committed")
            prems = f"  (~${prem:,.0f})" if isinstance(prem, (int, float)) else ""
            self.send(f"\U0001F7E2 <b>ENTRY {tk} {d}</b>{tag}\n"
                      f"{qty}${_n(entry.get('entry_price', entry.get('price')))}{prems}  "
                      f"TP ${_n(entry.get('tp_limit'))} / SL ${_n(entry.get('sl_limit'))}  "
                      f"{entry.get('dte', '?')}DTE\n<i>{rg}</i>{nfs}")
        elif act == "EXIT":
            pp = entry.get("pnl_pct")
            ic = "✅" if (pp or 0) > 0 else "❌"
            self.send(f"{ic} <b>EXIT {tk}</b>  ${_n(entry.get('entry_price'))}→"
                      f"${_n(entry.get('exit_price'))}\n"
                      f"<b>{_pct(pp)}</b>  (${_n(entry.get('pnl_dollars'), 0)})  "
                      f"{entry.get('reason', '?')} · {_n(entry.get('hold_mins'), 0)}m\n<i>{rg}</i>")

    def _sender(self):
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                requests.post(API.format(token=self.token, method="sendMessage"),
                              json={"chat_id": self.chat_id, "text": text,
                                    "parse_mode": "HTML", "disable_web_page_preview": True},
                              timeout=_TIMEOUT)
            except Exception:
                pass

    # -- inbound commands --------------------------------------------
    def start_commands(self, handlers: dict):
        """handlers: {"today": fn(args)->str, ...}. Adds /help automatically."""
        self._handlers = dict(handlers)
        self._drain()      # skip commands queued while the bot was down
        threading.Thread(target=self._poll, name="tg-poll", daemon=True).start()

    def _drain(self):
        try:
            r = requests.get(API.format(token=self.token, method="getUpdates"),
                             params={"offset": -1, "timeout": 0}, timeout=_TIMEOUT)
            for upd in (r.json().get("result", []) if r.status_code == 200 else []):
                self._offset = max(self._offset, upd.get("update_id", 0))
        except Exception:
            pass

    def _help(self):
        cmds = sorted(self._handlers)
        return "<b>commands</b>\n" + "\n".join(f"/{c}" for c in cmds) + "\n/help"

    def _poll(self):
        while not self._stop.is_set():
            try:
                r = requests.get(API.format(token=self.token, method="getUpdates"),
                                 params={"offset": self._offset + 1, "timeout": 30,
                                         "allowed_updates": '["message"]'}, timeout=40)
                data = r.json() if r.status_code == 200 else {}
            except Exception:
                time.sleep(3)
                continue
            for upd in data.get("result", []):
                self._offset = max(self._offset, upd.get("update_id", 0))
                msg = upd.get("message") or {}
                if str(msg.get("chat", {}).get("id")) != self.chat_id:
                    continue                                   # ignore everyone else
                txt = (msg.get("text") or "").strip()
                if not txt.startswith("/"):
                    continue
                parts = txt[1:].split()
                cmd = parts[0].split("@")[0].lower()
                args = parts[1:]
                if cmd == "help":
                    self.send(self._help()); continue
                fn = self._handlers.get(cmd)
                if not fn:
                    self.send(f"unknown: /{cmd}\n{self._help()}"); continue
                try:
                    out = fn(args)
                except Exception as e:
                    out = f"⚠️ /{cmd} failed: {e}"
                if out:
                    self.send(out)

    def stop(self):
        self._stop.set()


class _NullNotifier:
    """Stand-in when Telegram env isn't configured -- every call is a no-op."""
    def send(self, *_a, **_k): pass
    def trade(self, *_a, **_k): pass
    def start_commands(self, *_a, **_k): pass
    def stop(self, *_a, **_k): pass
