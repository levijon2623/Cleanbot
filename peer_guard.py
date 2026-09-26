"""
peer_guard.py
=============
Refuse to start a second engine against the same brokerage account.

🚨 WHY THIS EXISTS, WITH A DATE ON IT.
On 2026-09-24 the cloud bot ran 09:28-13:42 under its cron schedule while a
local bot ran all afternoon. Two engines on one Webull account: both
authenticate against the same token endpoint, both run the startup reconcile,
both would adopt the same positions, and both write `conf/token.txt`. It very
likely contributed to the Webull 429 that took an hour to diagnose -- and the
only thing that kept it from being worse was that both were in DRY_RUN.

The habit that protected against this was "kill the cloud one in the morning".
Under systemd that habit actively breaks: `kill <pid>` looks like a failure,
`Restart=on-failure` brings it back in 30s, and you carry on believing it is
dead. An interlock is cheap; remembering is not.

DIRECTION IS ONE-WAY, DELIBERATELY.
The laptop can reach the cloud box; the box cannot reach the laptop (no
inbound route, no stable address). So the LOCAL engine asks the REMOTE one
whether it is running, and stands down if it is. The remote is primary
because it is the one that survives a closed lid -- which is what the 15:55
flatten depends on.

🚨 AND IT MUST NEVER FIRE ON THE BOX ITSELF.
A bot that SSHes to its own host, finds its own service active and exits is a
self-deadlock that would take the cloud engine down permanently. Two
independent guards: the check only runs when `_viewer.json` names a `host`
(the box's own config deliberately has none), and it is skipped when this
machine's hostname matches the configured host.

FAILURE IS NOT REFUSAL. If SSH cannot answer -- network down, key missing,
box asleep -- we do not know, and refusing to start on "do not know" would
mean a broken laptop network stops you trading. It warns loudly and proceeds.
Only a confirmed ACTIVE remote blocks startup.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess

CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "_viewer.json")
REMOTE_UNIT = "cleanbot"
SSH_TIMEOUT = 8            # seconds; a hung check must not delay the open
OVERRIDE_ENV = "CLEANBOT_ALLOW_DUPLICATE"


def _conf():
    # utf-8-sig: PowerShell writes this file with a BOM (see live_sync.conf)
    try:
        with open(CONF, encoding="utf-8-sig") as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def _is_self(host):
    """Are we the machine we are about to interrogate?"""
    if not host:
        return True
    target = host.split("@")[-1].split(":")[0].strip().lower()
    me = socket.gethostname().lower()
    names = {me, me.split(".")[0]}
    try:
        names.add(socket.getfqdn().lower())
    except OSError:
        pass
    if target in names or target in ("127.0.0.1", "localhost", "::1"):
        return True
    try:
        _n, _a, ips = socket.gethostbyname_ex(me)
        if target in set(ips):
            return True
    except (OSError, socket.gaierror):
        pass

    # 🚨 BIND-TEST, BECAUSE NAME RESOLUTION MISSES THE PUBLIC IP.
    # On the cloud box `gethostbyname_ex(hostname)` returns a private address,
    # so naming the host by its PUBLIC ip (root@<public-ip> -- exactly how
    # the laptop addresses it, see _viewer.json, which is gitignored) slipped
    # past every check above and the box would have interrogated itself. It
    # resolved to "unknown" and proceeded,
    # so it was not a deadlock, but only by accident. You can bind() only to
    # an address that belongs to a local interface, which is the one test
    # that cannot be fooled by DNS.
    try:
        ip = socket.gethostbyname(target)
    except (OSError, socket.gaierror):
        return False
    for fam in (socket.AF_INET,):
        try:
            with socket.socket(fam, socket.SOCK_STREAM) as s:
                s.bind((ip, 0))
                return True                    # the address is ours
        except OSError:
            pass
    return False


def remote_state():
    """(state, detail). state is 'active' | 'inactive' | 'unknown' | 'skip'."""
    if os.getenv(OVERRIDE_ENV, "").strip().lower() in ("1", "true", "yes"):
        return "skip", f"{OVERRIDE_ENV} is set — check bypassed"

    c = _conf()
    host = c.get("host")
    if not host:
        # the cloud box's own _viewer.json has no `host`, which is what stops
        # this from ever running there
        return "skip", "no `host` in _viewer.json — nothing to check"
    if _is_self(host):
        return "skip", f"{host} is this machine — not checking myself"

    key = c.get("key")
    args = ["ssh", "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={SSH_TIMEOUT}",
            "-o", "StrictHostKeyChecking=accept-new"]
    if key:
        args += ["-i", key]
    if c.get("port"):
        args += ["-p", str(c["port"])]
    # 🚨 LABELLED, NOT POSITIONAL. `systemctl is-active` on a unit that does
    # not exist prints NOTHING, so splitting on whitespace slid the pgrep
    # count into the unit slot and reported "unit=0". The verdict happened to
    # stay right; the message did not, and a diagnostic that lies about what
    # it saw is worse than no diagnostic.
    # 🚨 THE BRACKET TRICK IS LOAD-BEARING, NOT STYLE.
    # `pgrep -f bot_runner.py` matches OUR OWN ssh command line, because that
    # line contains the string "bot_runner.py". Measured against the live box
    # 2026-09-24: plain -> 3 processes, anchored `python.*bot_runner` -> 3,
    # `[b]ot_runner[.]py` -> 1, with systemd reporting a single MainPID. The
    # pattern `[b]...` matches the literal text "bot_runner.py" but NOT our
    # own line, which contains the brackets. Without it the guard reports a
    # phantom process on every run and blocks local startup forever -- a
    # deadlock caused by the very tool meant to prevent one.
    args += [host, f"echo unit=$(systemctl is-active {REMOTE_UNIT} "
                   f"2>/dev/null || echo none); "
                   f"echo procs=$(pgrep -c -f '[b]ot_runner[.]py' "
                   f"2>/dev/null || echo 0)"]
    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           timeout=SSH_TIMEOUT + 7)
    except (subprocess.TimeoutExpired, OSError) as e:
        return "unknown", f"{type(e).__name__} contacting {host}"

    fields = {}
    for tok in (r.stdout or "").split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            fields[k] = v
    unit = fields.get("unit", "")
    try:
        procs = int(fields.get("procs", 0))
    except ValueError:
        procs = 0
    if not fields:
        return "unknown", (f"{host}: no parseable reply "
                           f"(rc={r.returncode} "
                           f"{(r.stderr or '').strip()[:100]})")

    if unit == "active" or procs > 0:
        return "active", (f"{host}: unit={unit or '?'}, "
                          f"{procs} bot_runner process(es)")
    if unit in ("inactive", "failed", "unknown", "none", "activating"):
        return "inactive", f"{host}: unit={unit}, no bot_runner processes"
    return "unknown", (f"{host}: ssh rc={r.returncode} "
                       f"{(r.stderr or '').strip()[:120]}")


def enforce(exit_on_conflict=True):
    """Print the verdict; return True if it is safe to continue."""
    state, detail = remote_state()
    if state == "skip":
        return True
    if state == "inactive":
        print(f"  🔒 peer check: remote engine is DOWN ({detail}) — clear to "
              f"start.")
        return True
    if state == "unknown":
        # not knowing is not the same as knowing it is running
        print(f"  ⚠️  peer check INCONCLUSIVE: {detail}")
        print(f"      Proceeding — a broken network here must not stop you "
              f"trading. If the cloud engine IS up, you now have two.")
        return True

    print("\n" + "=" * 68)
    print("  🚫 REFUSING TO START — ANOTHER ENGINE IS ALREADY RUNNING")
    print(f"     {detail}")
    print()
    print("     Two engines on one Webull account means duplicate startup")
    print("     reconciles, both adopting the same positions, and a shared")
    print("     token file. Stop the remote one first:")
    print(f"       ssh {_conf().get('host')} systemctl stop {REMOTE_UNIT}")
    print()
    print("     `kill <pid>` is NOT enough — systemd restarts it in 30s.")
    print(f"     To run both anyway: set {OVERRIDE_ENV}=true")
    print("=" * 68 + "\n")
    if exit_on_conflict:
        raise SystemExit(2)
    return False


if __name__ == "__main__":
    s, d = remote_state()
    print(f"  state={s}  {d}")
