# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
live_sync.py
============
Pulls the cloud bot's `live/state.json` down to this desktop so flow_viewer.html
can show the live trigger next to the local historical tape.

WHY A PULL AND NOT A SERVER
    The data is split: lake/ (22k parquet), historical/ (3k) and the 260MB
    viewer tape are all on THIS machine, while bot_runner runs on the cloud box.
    Only live/state.json is remote, and it is 13KB rewritten every 5s. So the
    chart stays here with the tape and the small file travels -- rather than
    shipping 260MB up, or exposing a port on a host that is holding real
    positions.

🚨 NOTHING IS EXPOSED AND NO CREDENTIAL IS HANDLED HERE
    Transport is your own `ssh`. Authentication is your existing key or agent --
    this file never reads, stores or passes a password, and the remote host
    never opens a listening port for this. If you would rather not run a sync at
    all, the alternative is an SSH tunnel with the viewer on the box
    (`ssh -L 8765:127.0.0.1:8765 HOST`), but the box has no tape, so you would
    lose historical mode.

🚨 ONE CONNECTION, NOT 720 HANDSHAKES AN HOUR
    Windows OpenSSH has no ControlMaster, so a `scp` every 5 seconds means a
    fresh TCP+auth handshake every 5 seconds -- ~500ms of work each, and a very
    noisy auth log. Default mode instead opens ONE ssh session that cats the
    file on a loop, separating records with 0x1E (ASCII Record Separator, which
    cannot occur in compact JSON). The remote loop is BOUNDED (--cycles) so a
    session stranded by a dropped connection dies by itself within the hour
    instead of lingering on a trading host.

    `--mode scp` is the fallback if the remote shell is not POSIX.

Env (set these yourself; nothing here writes to .env):
    CLEANBOT_SSH_HOST      required -- "user@host" or an ssh_config alias
    CLEANBOT_SSH_PORT      optional -- defaults to ssh's own default
    CLEANBOT_SSH_KEY       optional -- path to a private key (-i)
    CLEANBOT_REMOTE_STATE  optional -- default ~/Cleanbot/live/state.json

Usage (PowerShell -- `set X=y` is cmd.exe syntax and does nothing here):
    $env:CLEANBOT_SSH_HOST = "user@1.2.3.4"
    python live_sync.py --once      # verify auth
    python live_sync.py             # then leave it running

Or just `python serve_viewer.py`, which starts this and the web server together.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

RS = b"\x1e"
LOCAL = os.path.join("live", "state.json")
REMOTE_DEFAULT = "/root/bot_engine/Cleanbot/live/state.json"
CONF = "_viewer.json"          # gitignored by the `_*.json` rule


def conf():
    """Settings from _viewer.json, overridden by environment variables.

    Exists because env vars are the wrong mechanism here: a VS Code terminal
    inherits the environment VS CODE had when it launched, not the registry, so
    `[Environment]::SetEnvironmentVariable(..., "User")` does not reach a new
    terminal until the editor restarts. That produced a sync quietly pointed at
    the wrong remote path. A file in the repo has none of that ambiguity.

    Holds no secret: a host address and the PATH to a key, never key material.
    """
    c = {}
    try:
        # 🚨 utf-8-sig, NOT utf-8. PowerShell's `>`, `Out-File` and
        # `ConvertTo-Json | Set-Content` all write a UTF-8 BOM here by default,
        # and `json.load` on a BOM raises -- which this except swallows. The
        # result is a config file that looks perfectly correct in an editor
        # while every setting in it is silently ignored. `utf-8-sig` reads both.
        with open(CONF, encoding="utf-8-sig") as f:
            c = json.load(f)
    except (OSError, ValueError):
        pass
    return {
        "host": os.getenv("CLEANBOT_SSH_HOST") or c.get("host"),
        "key": os.getenv("CLEANBOT_SSH_KEY") or c.get("key"),
        "port": os.getenv("CLEANBOT_SSH_PORT") or c.get("port"),
        "remote": (os.getenv("CLEANBOT_REMOTE_STATE") or c.get("remote")
                   or REMOTE_DEFAULT),
    }


def ssh_base():
    c = conf()
    host = c["host"]
    if not host:
        # `set NAME=value` is cmd.exe syntax and fails SILENTLY in PowerShell,
        # where `set` aliases Set-Variable -- no error, no env var. Spell out
        # the form that actually works on this platform.
        sys.exit(
            f'  No SSH host configured.\n'
            f'  Easiest: create {CONF} in the repo root --\n'
            f'    {{"host": "user@your-box",\n'
            f'     "key":  "C:/Users/you/.ssh/cleanbot_ed25519",\n'
            f'     "remote": "{REMOTE_DEFAULT}"}}\n'
            f'  Or set CLEANBOT_SSH_HOST in the environment. NOTE that a VS Code\n'
            f'  terminal inherits the environment VS CODE started with, so a\n'
            f'  persisted User variable will not appear until the editor is\n'
            f'  restarted -- which is exactly why {CONF} exists.')
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15",
           "-o", "ServerAliveCountMax=3"]
    if c["port"]:
        cmd += ["-p", str(c["port"])]
    if c["key"]:
        # IdentitiesOnly stops ssh offering every other identity first, which
        # on a server with MaxAuthTries can exhaust the limit before the key we
        # actually want is tried.
        cmd += ["-i", os.path.expanduser(c["key"]), "-o", "IdentitiesOnly=yes"]
    return cmd, host


def write_atomic(path, raw):
    """Validate, then replace. The viewer must never read a partial document."""
    obj = json.loads(raw)                    # raises on a truncated record
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return obj


def describe(obj):
    n = len(obj.get("tickers") or {})
    age = max(0, int(time.time()) - int(obj.get("ts") or 0))
    hold = [t for t, s in (obj.get("tickers") or {}).items()
            if s.get("state") == "holding"]
    return (f"{obj.get('session', '?')}  {n} tickers  "
            f"{obj.get('n_open', 0)} open"
            + (f" [{', '.join(hold)}]" if hold else "")
            + f"  {'PAPER' if obj.get('dry_run') else 'LIVE MONEY'}"
            + f"  bot ts {age}s ago" + ("  ⚠ STALE" if age > 90 else ""))


def stream(args, remote):
    """One ssh session; remote cats the file on a bounded loop."""
    base, host = ssh_base()
    # `sleep` inside the loop, and a bounded iteration count so a session left
    # behind by a dead connection expires on its own.
    script = (f"i=0; while [ $i -lt {args.cycles} ]; do "
              f"cat {remote} 2>/dev/null; printf '\\036'; "
              f"sleep {args.interval}; i=$((i+1)); done")
    proc = subprocess.Popen(base + [host, script],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buf, n, last, empty = b"", 0, None, 0
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while RS in buf:
                rec, buf = buf.split(RS, 1)
                rec = rec.strip()
                if not rec:
                    # An EMPTY record means `cat` found nothing -- the remote
                    # path is wrong, or the bot has not written yet. This used
                    # to `continue` silently, so a sync pointed at a
                    # non-existent file looked healthy forever: process up,
                    # connection open, nothing ever delivered. Say it.
                    empty += 1
                    if empty in (1, 12, 120) or empty % 720 == 0:
                        print(f"  empty record x{empty}: {remote} is missing or "
                              f"empty on {host}.")
                        if empty == 1:
                            print(f"    check CLEANBOT_REMOTE_STATE -- it is "
                                  f"currently '{remote}'")
                    continue
                empty = 0
                try:
                    obj = write_atomic(args.out, rec)
                except (ValueError, OSError) as e:
                    print(f"  skipped a record: {type(e).__name__}: {e}")
                    continue
                n += 1
                cur = obj.get("ts")
                if cur != last:
                    print(f"  [{n:>5}] {describe(obj)}")
                    last = cur
    finally:
        proc.terminate()
        err = (proc.stderr.read() or b"").decode(errors="replace").strip()
    return n, err


def once_ssh(args, remote):
    """A single fetch over the SAME transport `stream` uses.

    --once used to call once_scp regardless of --mode, so the connectivity test
    exercised a path the tool does not normally take. That matters here: scp on
    OpenSSH 9.x runs over the SFTP subsystem, so a server without it fails scp
    with a bare "Connection closed" while plain ssh works perfectly.
    """
    base, host = ssh_base()
    r = subprocess.run(base + [host, f"cat {remote}"],
                       capture_output=True, timeout=30)
    if r.returncode != 0:
        return None, ((r.stderr or b"").decode(errors="replace").strip()
                      or f"ssh exited {r.returncode}")
    raw = (r.stdout or b"").strip()
    if not raw:
        return None, f"{remote} is empty or does not exist on the remote"
    try:
        return write_atomic(args.out, raw), None
    except ValueError as e:
        return None, f"remote file is not valid JSON ({e})"


def once_scp(args, remote):
    base, host = ssh_base()
    c = conf()
    scp = ["scp", "-q", "-o", "BatchMode=yes"]
    if c["port"]:
        scp += ["-P", str(c["port"])]
    if c["key"]:
        scp += ["-i", os.path.expanduser(c["key"]), "-o", "IdentitiesOnly=yes"]
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        r = subprocess.run(scp + [f"{host}:{remote}", tmp],
                           capture_output=True)
        if r.returncode != 0:
            return None, (r.stderr or b"").decode(errors="replace").strip()
        with open(tmp, "rb") as f:
            return write_atomic(args.out, f.read()), None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=LOCAL)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--cycles", type=int, default=720,
                    help="remote loop iterations before the session retires")
    ap.add_argument("--mode", choices=("stream", "scp"), default="stream")
    ap.add_argument("--once", action="store_true",
                    help="fetch a single snapshot and exit -- use to test auth")
    a = ap.parse_args()
    remote = conf()["remote"]
    _base, host = ssh_base()
    print(f"  {host}:{remote}\n  -> {a.out}   mode={a.mode} "
          f"every {a.interval}s")

    if a.once or a.mode == "scp":
        fetch = once_scp if a.mode == "scp" else once_ssh
        while True:
            obj, err = fetch(a, remote)
            if err:
                print(f"  {a.mode} fetch failed: {err}")
            elif obj:
                print(f"  {describe(obj)}")
            if a.once:
                return 0 if obj else 1
            time.sleep(a.interval)

    back = a.interval
    while True:
        t0 = time.time()
        try:
            n, err = stream(a, remote)
        except KeyboardInterrupt:
            print("\n  stopped."); return 0
        except Exception as e:                       # noqa: BLE001
            n, err = 0, f"{type(e).__name__}: {e}"
        ran = time.time() - t0
        if err:
            print(f"  ssh ended after {ran:.0f}s ({n} records): {err}")
        else:
            print(f"  ssh session ended after {ran:.0f}s ({n} records)")
        # a session that ran a while was healthy -- reconnect promptly. One that
        # died instantly is a real failure, so back off instead of hammering.
        back = a.interval if ran > 60 else min(back * 2, 300)
        print(f"  reconnecting in {back}s…")
        try:
            time.sleep(back)
        except KeyboardInterrupt:
            print("\n  stopped."); return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n  stopped.")
