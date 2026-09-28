# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
serve_viewer.py
===============
One command to bring the flow viewer up: serves this directory on localhost and
starts live_sync alongside it, then opens the browser.

    python serve_viewer.py

    --no-live     historical tape only, do not start live_sync
    --port 8765   default
    --no-open     do not launch a browser

🚨 BINDS TO 127.0.0.1, DELIBERATELY
    `python -m http.server` defaults to 0.0.0.0, which would put the tape and
    live/state.json -- open positions, strikes, entry prices, account equity --
    in front of anything on the LAN. This binds loopback only. If you ever need
    it from another machine, tunnel it (`ssh -L 8765:127.0.0.1:8765 ...`) rather
    than changing the bind.

live_sync is a CHILD process and is terminated on exit, so Ctrl-C leaves nothing
running. If CLEANBOT_SSH_HOST is unset the server still starts and the viewer
works in historical mode; it just says so rather than failing.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser


NOTES = "_notes.jsonl"
MAX_NOTE = 64 * 1024


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass                                   # the access log is pure noise here

    def end_headers(self):
        # live/state.json is rewritten every 5s; a cached copy would freeze the
        # chart in a way that looks exactly like a stalled bot.
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def _secret_ok(self):
        """Shared secret + Origin. An order endpoint is not a display
        endpoint: any local process, and some cross-origin form posts, can
        reach the bind address. The secret lives in _viewer.json (gitignored)
        and the page reads it from /order-config.

        🚨 THE ORIGIN CHECK IS CSRF PROTECTION, NOT AUTHENTICATION.
        It stops a page you visit from POSTing here on your behalf. It proves
        nothing about who is calling. The secret is the authentication, and
        over anything but loopback the transport has to be trusted -- which
        is the entire reason this is bound to a Tailscale address rather than
        exposed publicly. A bearer token in a plaintext header is fine inside
        an encrypted mesh and is NOT fine on the open internet.
        """
        want = _order_secret()
        if not want:
            return False
        if self.headers.get("X-Order-Key") != want:
            return False
        origin = self.headers.get("Origin")
        if origin and not _origin_ok(origin):
            return False
        return True

    def do_GET(self):
        if self.path.split("?")[0] == "/events":
            return self._json(_recent_events())
        if self.path.split("?")[0] == "/order-config":
            # served only to a loopback GET; it is the same machine either way
            return self._json({"key": _order_secret() or "",
                               "max_contracts": 20, "max_premium": 5000})
        return super().do_GET()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.rstrip("/") == "/order":
            return self._do_order()
        return self._do_note()

    def _do_order(self):
        """Write an order REQUEST. This process NEVER places an order.

        bot_runner picks the file up, re-validates everything (armed state,
        caps, spread collar, quote age, duplicate holds) and executes. The
        viewer holds no credentials and its UI limits are a convenience, not
        the guard -- manual_orders.py is the guard.
        """
        if not self._secret_ok():
            return self._json({"ok": False, "err": "bad key or origin"}, 403)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 8192:
                return self._json({"ok": False, "err": "bad size"}, 413)
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            act = str(req.get("action") or "buy").lower()
            if act not in ("buy", "close", "panic", "arm", "disarm", "levels"):
                return self._json({"ok": False, "err": "action"}, 400)
            # underlying TP/SL prices: numbers or null, nothing else
            for k in ("tp", "sl", "ul_tp", "ul_sl"):
                if req.get(k) is not None and not isinstance(req[k], (int, float)):
                    return self._json({"ok": False, "err": f"{k} must be a number"}, 400)
            # 911 and arm/disarm are account-wide: no ticker to validate
            if act not in ("panic", "arm", "disarm") and str(
                    req.get("ticker", "")).upper() not in (
                    "IWM", "SPY", "QQQ"):
                return self._json({"ok": False, "err": "ticker"}, 400)
            if act == "buy":
                if not str(req.get("option_id") or ""):
                    return self._json({"ok": False, "err": "option_id"}, 400)
                q = int(req.get("qty") or 0)
                if q < 1 or q > 20:
                    return self._json({"ok": False, "err": "qty 1..20"}, 400)
            req["requested_at"] = int(time.time())
            d = os.path.join("live", "orders")
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f"{int(time.time()*1e6)}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(req, f, separators=(",", ":"))
        except Exception as e:
            return self._json({"ok": False, "err": str(e)}, 400)
        return self._json({"ok": True, "queued": os.path.basename(p)})

    def _do_note(self):
        """Append a discretionary note to _notes.jsonl.

        🚨 THE POINT IS THE PRE-SPECIFICATION. A trade rationale recalled after
        the outcome is unfalsifiable -- the same setup that failed never gets
        enumerated. Written HERE, before the outcome is known, with the full
        state snapshot attached, it becomes a testable record. That is why PASS
        entries matter as much as TAKE ones and the UI insists on the choice.

        Loopback-only by construction (the server binds 127.0.0.1), so no auth;
        the size cap is for accidents, not attackers.
        """
        if self.path.rstrip("/") != "/note":
            self.send_error(404); return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_NOTE:
                self.send_error(413); return
            rec = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(rec, dict):
                raise ValueError("not an object")
            rec["logged_at"] = int(time.time())
            with open(NOTES, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "err": str(e)}).encode())
            return
        body = json.dumps({"ok": True, "n": _count_notes()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_BIND = {"host": "127.0.0.1", "port": 8765}


def _origin_ok(origin):
    """Is this Origin one we serve on?

    Loopback always. Otherwise the address we are actually bound to, plus
    anything explicitly listed as `allowed_origins` in _viewer.json -- needed
    because a Tailscale box is reachable by IP *and* by MagicDNS name, and a
    browser sends whichever one you typed.
    """
    import urllib.parse
    try:
        u = urllib.parse.urlparse(origin)
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        return True
    if host == str(_BIND["host"]).lower():
        return True
    for a in (_conf().get("allowed_origins") or []):
        try:
            if (urllib.parse.urlparse(a).hostname or a).lower() == host:
                return True
        except ValueError:
            continue
    return False


def _conf():
    # utf-8-sig: PowerShell writes this file with a BOM (see live_sync.conf)
    try:
        with open("_viewer.json", encoding="utf-8-sig") as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def _order_secret():
    # utf-8-sig: PowerShell writes _viewer.json with a BOM, json.load raises on
    # it, and the except below turns that into "no secret configured" -- i.e.
    # every order 403s with a config file that reads correctly. See live_sync.conf.
    try:
        with open("_viewer.json", encoding="utf-8-sig") as f:
            return (json.load(f).get("order_key") or "").strip() or None
    except (OSError, ValueError):
        return None


def _recent_events(limit=60):
    """The telemetry feed: queued requests and bot_runner's verdicts.

    Requests appear as REQ the moment they are written; the matching
    .result.json appears when the bot has ruled on it. So a request that sits
    on REQ with no verdict means the BOT is not running -- which is exactly
    what you want to see rather than a silent nothing.
    """
    d = os.path.join("live", "orders")
    out = []
    try:
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            try:
                with open(p, encoding="utf-8") as f:
                    j = json.load(f)
            except (OSError, ValueError):
                continue
            if fn.endswith(".result.json"):
                out.append(dict(kind="EXEC" if j.get("ok") else "GUARD",
                                at=j.get("at"), msg=j.get("msg", ""),
                                ok=bool(j.get("ok"))))
            elif str(j.get("action") or "buy").lower() == "panic":
                out.append(dict(kind="REQ", at=j.get("requested_at"),
                                msg="911 FLATTEN ALL", ok=None))
            elif str(j.get("action") or "buy").lower() == "close":
                out.append(dict(kind="REQ", at=j.get("requested_at"),
                                msg=f"CLOSE {j.get('ticker')} "
                                    f"({j.get('mode') or 'marketable'})",
                                ok=None))
            else:
                out.append(dict(kind="REQ", at=j.get("requested_at"),
                                msg=f"{j.get('ticker')} {j.get('option_id')} "
                                    f"qty {j.get('qty')}", ok=None))
    except OSError:
        pass
    out.sort(key=lambda x: x.get("at") or 0)
    return out[-limit:]


def _count_notes():
    try:
        with open(NOTES, encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--local", action="store_true",
                    help="bot_runner runs on THIS machine -- read "
                         "live/state.json directly, start no sync")
    ap.add_argument("--bind", default=None,
                    help="address to listen on (default 127.0.0.1, or "
                         "`bind` in _viewer.json). Use your Tailscale IP to "
                         "reach it from your own devices.")
    ap.add_argument("--i-know-this-is-public", action="store_true",
                    help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.local:
        try:
            import json as _j
            a.local = bool(_j.load(open("_viewer.json", encoding="utf-8-sig"))
                           .get("local"))
        except Exception:
            pass

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    if not os.path.exists("flow_viewer.html"):
        sys.exit("  flow_viewer.html not found -- run this from the repo root.")

    have_tape = os.path.exists(os.path.join("tape", "index.json"))
    print(f"  root      {root}")
    # Say this OUT LOUD at startup. A missing/unreadable order_key makes every
    # BUY return 403 from a server that is otherwise working perfectly, and the
    # browser shows only "REFUSED" -- nothing that points at _viewer.json.
    print(f"  orders    {'armable — order_key loaded' if _order_secret() else
                         'DISABLED — no order_key in _viewer.json (all /order '
                         'requests will 403)'}")
    print(f"  tape      {'ready' if have_tape else 'MISSING -- run '
                         'export_flow_tape.py'}")

    child = None
    if a.local:
        # bot_runner is running on THIS machine, so live_state writes
        # live/state.json directly. Starting live_sync here would pull the
        # cloud's copy over the top of it -- silently, and the file would look
        # perfectly healthy while showing a bot that is not the one running.
        p = os.path.join("live", "state.json")
        age = (int(time.time() - os.path.getmtime(p))
               if os.path.exists(p) else None)
        print(f"  live      LOCAL bot — no sync. live/state.json "
              + (f"{age}s old" if age is not None else "not written yet"))
        if age is None or age > 120:
            print(f"            (start bot_runner.py; the file appears within 5s)")
    elif not a.no_live:
        # Same resolution live_sync uses: _viewer.json, overridden by env.
        try:
            import live_sync
            cfg = live_sync.conf()
        except Exception:
            cfg = {"host": os.getenv("CLEANBOT_SSH_HOST"), "remote": "?"}
        if not cfg.get("host"):
            print("  live      no SSH host configured -- historical only.")
            # `set NAME=value` is cmd.exe syntax. In PowerShell `set` is an
            # alias for Set-Variable, so it fails SILENTLY -- no error, no env
            # var, and the child python sees nothing. Print the right form for
            # the shell actually in use.
            print('            create _viewer.json in the repo root:')
            print('              {"host": "user@your-box",')
            print('               "key": "C:/Users/you/.ssh/cleanbot_ed25519"}')
            print("            then restart this script for the live pane.")
        else:
            remote = cfg["remote"]
            # Echo the REMOTE PATH, not just the pid. A sync pointed at the
            # wrong path looks identical to a healthy one from out here -- the
            # process is up and the connection is open, it just never delivers.
            print(f"  live      {cfg['host']}:{remote}")
            log = open("_live_sync.log", "w", encoding="utf-8")
            child = subprocess.Popen([sys.executable, "-u", "live_sync.py"],
                                     stdout=log, stderr=subprocess.STDOUT)
            print(f"            live_sync pid {child.pid} "
                  f"-> live/state.json   (log: _live_sync.log)")

    # 🚨 THE BIND ADDRESS IS A SECURITY DECISION, SO IT IS EXPLICIT.
    # Default stays loopback. A Tailscale address is fine: the mesh is
    # encrypted and device-authenticated, so a bearer token in a header is an
    # acceptable second factor there. 0.0.0.0 is NOT fine -- that puts an
    # endpoint which PLACES REAL TRADES on every interface the box has,
    # including its public one, protected by a 32-char string in plaintext
    # HTTP. That is refused rather than warned about.
    bind = a.bind or _conf().get("bind") or "127.0.0.1"
    loopback = bind in ("127.0.0.1", "localhost", "::1")
    if bind in ("0.0.0.0", "::") and not a.i_know_this_is_public:
        sys.exit(
            f"\n  🚫 refusing to bind {bind}.\n"
            f"     /order places real trades and this server has no TLS, so\n"
            f"     every interface on the box -- including the public one --\n"
            f"     would be one guessed header away from your broker.\n"
            f"     Bind your Tailscale address instead:\n"
            f"       python serve_viewer.py --bind 100.x.y.z\n")
    _BIND.update(host=bind, port=a.port)

    url = f"http://{bind}:{a.port}/flow_viewer.html"
    srv = http.server.ThreadingHTTPServer(
        (bind, a.port), functools.partial(Handler, directory=root))
    print(f"\n  {url}")
    if loopback:
        print("  (loopback only — Ctrl-C to stop everything)\n")
    else:
        allowed = _conf().get("allowed_origins") or []
        print(f"  ⚠  bound to {bind} — reachable by anything that can route "
              f"to it.")
        print(f"     Orders need the X-Order-Key header AND a session arm "
              f"that expires.")
        print(f"     extra allowed origins: {allowed or '(none — add '
                 f'`allowed_origins` to _viewer.json if you use a DNS name)'}")
        print("     Ctrl-C to stop everything\n")
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping…")
    finally:
        srv.server_close()
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
            print("  live_sync stopped.")


if __name__ == "__main__":
    main()
