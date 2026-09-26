# Cloud deployment — bot + viewer over Tailscale

Written for: whoever operates this box, which is currently you.

## What this gets you

The bot runs whether your laptop is open or not — which matters because the
**15:55 EOD flatten only runs while `bot_runner` does**. Sleep, reboot and a
closed lid all defeat it locally. That flatten is the protection against
forgetting a 0DTE, so it is the strongest argument for the move.

The viewer runs on the same box and is reachable from your own devices, and
from nothing else.

## Why Tailscale and not a public URL

`/order` places real trades. It authenticates with a 32-character bearer
token in a **plaintext HTTP header** — entirely appropriate inside an
encrypted, device-authenticated mesh, and the wrong primitive for the open
internet. Tailscale gives you phone access without ever creating a
discoverable endpoint whose failure mode is someone trading your account.

`serve_viewer` **refuses** to bind `0.0.0.0` for this reason. That is not a
warning you can click through.

## Steps

**1. Tailscale on the box and on your devices**

```sh
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4          # -> 100.x.y.z, put this in viewer.service
```

Install the app on your phone/laptop and sign into the same tailnet.

**2. `_viewer.json` on the box**

```json
{
  "local": true,
  "bind": "100.x.y.z",
  "order_key": "<32 random chars — NOT the one from your laptop>",
  "allowed_origins": ["http://your-box.tailnet-name.ts.net:8765"]
}
```

`local: true` makes the viewer read `live/state.json` off disk and never
start `live_sync` — on the box that writes that file, a sync would pull a
copy over the top of it.

`allowed_origins` exists because Tailscale reaches a host by IP *and* by
MagicDNS name, and the browser sends whichever you typed. Omit it if you
always use the IP.

Generate a **new** `order_key` for the box. Reusing your laptop's means one
leak compromises both.

**3. Services**

```sh
sudo cp deploy/cleanbot.service /etc/systemd/system/
sudo cp deploy/viewer.service   /etc/systemd/system/cleanbot-viewer.service
# edit the --bind address in cleanbot-viewer.service first
sudo systemctl daemon-reload
sudo systemctl enable --now cleanbot cleanbot-viewer
journalctl -u cleanbot -f
```

**4. Confirm the banner before trusting anything**

```
bot orders    : PAPER — nothing sent
manual orders : 🔴 ARMED — REAL CAPITAL from the viewer
exits of REAL positions always send, on either flag.
```

Then open `http://100.x.y.z:8765/flow_viewer.html` from a device on the
tailnet. The staging banner should read **WINDOW CLOSED — ARM TO TRADE**.

## The arming model, which changed for this

Two factors:

| | what it is | where it lives | expires |
|---|---|---|---|
| `MANUAL_TRADING_ARMED` | the **capability** | systemd unit | no |
| session arm | the **act** | the ARM button | **15 min** |

It used to be one flag, which had to be set by hand per launch precisely
because it was the only gate. A permanently-capable process whose window is
shut is safer than one armed for eight hours because you typed an env var at
09:00 — so the capability now lives in the unit file and the window is what
you actually operate.

A restart clears the window. Nothing persists an arm to disk, deliberately.

**Exits are gated on neither.** A position you own must always be closeable,
so `flatten`, the 15:55 sweep and 911 all work with the window shut and with
`DRY_RUN=true`.

## What did not change

- `DRY_RUN` still governs the **bot's** own orders and is separate from your
  hand. Leave it `true` until the bot's live rollout is its own decision.
- The viewer still never places an order. It writes a request file;
  `bot_runner` re-validates every cap, collar and guard and executes.
- `.env` stays on the box and out of git. The unit reads it via
  `EnvironmentFile`; do not put secrets in the unit itself, since unit files
  are world-readable by default.

## Rollback

```sh
sudo systemctl disable --now cleanbot-viewer cleanbot
```

Then run locally as before. Nothing in the repo is cloud-specific except
these files and `_viewer.json`.
