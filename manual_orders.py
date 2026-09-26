"""
manual_orders.py
================
Discretionary orders fired from flow_viewer, executed by bot_runner.

🚨 THE VIEWER NEVER PLACES AN ORDER. It writes a REQUEST file; bot_runner picks
it up, validates it and executes. That is deliberate:
  - only one process ever holds Webull credentials or an order connection
  - the marketable-limit convention, the allocator and the ledger stay in ONE
    place instead of being reimplemented in a web handler
  - the resulting position lives in bot_runner's memory, so the EOD flatten and
    the entry block work on it automatically rather than by remembering to

🚨 ARMING IS SEPARATE FROM DRY_RUN, ON PURPOSE
    `DRY_RUN` means "the BOT sends nothing". If manual orders rode on it, the
    startup banner would read `DRY_RUN = True` while real orders were leaving
    the machine -- and that banner is exactly what you check to confirm you are
    safe. `MANUAL_TRADING_ARMED` is independent, defaults False, and is echoed
    into live/state.json so the chart and the engine can never disagree.

POSITION HANDLING -- chosen by the user 2026-09-22
    Manual fills are NOT adopted into the bot's bracket logic: no trail, no
    stop. Two things are enforced anyway:
      1. AN ENTRY BLOCK. `active_snipes` is what the scanner's guard keys on,
         so without this the bot would happily open its own IWM position on top
         of a manual one and double the exposure.
      2. AN EOD FLATTEN, which TRUMPS "unmanaged". A 0DTE with no stop that is
         simply forgotten expires worthless; the user has done it before. This
         is the one exit that protects against absence rather than price.

VALIDATION, all of it before anything is sent
    armed · known contract with a live quote · qty and premium caps · a price
    collar against the live mid, so a stale quote cannot fill something absurd.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live", "orders")
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live",
                     "manual_positions.json")

ALLOWED = {"IWM", "SPY", "QQQ"}

# 🚨 CONTRACTS TRACKED FOR THE HAND, NOT FOR THE BOT.
# The deployed rules for these three are CALL-only (SPY CHOP CALL, QQQ HIVOL
# CALL, IWM HIVOL CALL), so `_determine_required_dtes` only ever asked for
# calls, only calls were subscribed on MQTT, and the strike strip could show
# nothing else. Discretionary trading needs both sides.
#
# THIS CANNOT MAKE THE BOT TRADE A PUT. The basket is INVENTORY; config.RULES
# is AUTHORITY. bot_runner derives its direction from flow momentum and then
# requires `_match_rule(ticker, direction, state)` to return an enabled rule
# for that exact (ticker, direction) pair -- there is no SPY/QQQ/IWM PUT rule,
# so the scanner hits `if not trade_rules: continue` and the extra contracts
# sit in memory unused. They cost MQTT subscriptions (~25 more) and nothing
# else. Adding a PUT rule to config.RULES is what would change behaviour, and
# that is a research decision, not this one.
EXTRA_DIRECTIONS = {"IWM": ("PUT",), "SPY": ("PUT",), "QQQ": ("PUT",)}

MAX_CONTRACTS = 20
# 🚨 A CEILING, NOT THE CAP. The binding limit is a fraction of REAL account
# cash -- see max_premium(). A fixed $5,000 was 2.5x the actual account on
# 2026-09-24, so the desk would happily stage an order the broker had to
# reject, and the viewer's "over the cap" warning could never fire.
MAX_PREMIUM = 5_000.0          # dollars per order, absolute ceiling
# Self-discipline on top of buying power: the most of it one order may spend.
#
# 🚨 THIS WAS 1.0 -- I.E. NO LIMIT AT ALL -- UNTIL 2026-09-26.
# That was defensible while the account held test money and buying power was
# itself the binding constraint: the worst case was losing a few hundred dollars
# you had already written off. It stops being defensible the moment the account
# is funded, because "no extra limit" means one mistyped quantity on one 0DTE
# can commit the entire balance to a contract that may be worth nothing by
# 16:00. A cap whose value is 100% is not a cap, it is a comment.
#
# 0.20 is the figure to start from: five independent mistakes to lose the
# account rather than one, and still ample for the 1-3 contract orders this desk
# actually sends. Raise it deliberately, per-environment, not by editing this
# line -- and note that this is a fraction of REAL option buying power, so it
# tightens automatically as the account shrinks.
MANUAL_MAX_PREMIUM_PCT = float(
    (os.getenv("MANUAL_MAX_PREMIUM_PCT") or "0.20").strip() or 0.20)

_CASH = {"v": None, "at": 0.0, "dt": None}
CASH_TTL_S = 300.0


def account_cash(eng, force=False):
    """REAL option buying power, never the paper bankroll.

    🚨 DRY_RUN's PAPER_EQUITY ($30k by default) is a research device -- it
    keeps the bot's capital gate deterministic regardless of what is in the
    linked account (bot_runner:1189). Manual orders spend ACTUAL money
    whatever DRY_RUN says, so sizing them off that number would let a $68
    account stage $5,000 orders. Same separation as ARMED vs DRY_RUN.

    🚨 THE FIELD NAMES ARE FROM THE ACTUAL PAYLOAD, NOT FROM GUESSES.
    The first version tried cash_balance / buying_power / net_liquidation and
    matched NONE of them, so it silently fell back to a constant that was
    larger than the whole account. Observed 2026-09-24:
      total_net_liquidation_value, total_cash_balance, total_market_value,
      day_trades_left, account_currency_assets[0].{option_buying_power,
      cash_balance, day_buying_power, net_liquidation_value}
    `option_buying_power` is the one that governs buying an option, so it is
    preferred over cash.

    Cached for CASH_TTL_S. Returns None only if it has NEVER been read.
    """
    if not force and _CASH["v"] is not None and \
            time.time() - _CASH["at"] < CASH_TTL_S:
        return _CASH["v"]
    try:
        h = eng.webull.get_account_health() or {}
        assets = h.get("account_currency_assets") or []
        node = assets[0] if isinstance(assets, list) and assets else {}
        for src, keys in ((node, ("option_buying_power", "day_buying_power",
                                  "cash_balance", "net_liquidation_value")),
                          (h, ("total_cash_balance",
                               "total_net_liquidation_value"))):
            for k in keys:
                v = src.get(k)
                if v not in (None, ""):
                    _CASH.update(v=float(v), at=time.time(),
                                 dt=h.get("day_trades_left"))
                    return _CASH["v"]
        print(f"  ⚠️ account_cash: no known balance field in {sorted(h)}")
    except Exception as e:                      # noqa: BLE001 -- deliberate
        print(f"  ⚠️ account_cash: {type(e).__name__}: {e}")
    # keep the LAST GOOD value rather than inventing one -- a transient API
    # blip must not raise the cap
    _CASH["at"] = time.time()
    return _CASH["v"]


def max_premium(eng):
    """(cap_dollars, basis) for one discretionary order.

    🚨 UNREADABLE MEANS ZERO, NOT A DEFAULT. A cap we cannot compute is not a
    cap. Returning a constant here is how a $68 account got a $250 allowance
    on the first attempt at this function; refusing is the only safe answer,
    and the message says exactly why.
    """
    bp = account_cash(eng)
    if bp is None:
        return 0.0, "buying power unreadable — orders refused until it reads"
    if bp <= 0:
        return 0.0, "no option buying power"
    cap = min(MAX_PREMIUM, bp * MANUAL_MAX_PREMIUM_PCT)
    if MANUAL_MAX_PREMIUM_PCT < 1.0:
        return cap, (f"{MANUAL_MAX_PREMIUM_PCT:.0%} of ${bp:,.2f} "
                     f"option buying power")
    return cap, (f"${bp:,.2f} option buying power"
                 if cap < MAX_PREMIUM else f"${MAX_PREMIUM:,.0f} ceiling")
# 🚨 THE COLLAR GATES ON SPREAD WIDTH, NOT ON LIMIT-vs-MID.
# The first version compared the limit to the mid -- but the limit IS derived
# from the mid (min(mid+0.01, ask)), so it can never be more than ~1c away and
# the check could never fire. A bid 0.10 / ask 3.00 quote (a 187% spread,
# plainly stale) passed it and filled at 1.56. Spread width is the thing that
# actually says "do not trust this quote". IWM/SPY/QQQ 0DTE normally run
# 1-4% of mid, so 15% is loose enough for a genuinely fast tape and still
# rejects a blown-out book.
MAX_SPREAD_PCT = 0.15
MAX_QUOTE_AGE = 20.0           # seconds; live_option_quotes carries a timestamp
EOD_FLATTEN_MOD = 15 * 60 + 55  # 15:55 ET

# ---- exits ---------------------------------------------------------------
EXIT_MARKETABLE = "marketable"
EXIT_911 = "911"
EXIT_CHASE_S = 20.0            # unfilled this long -> cancel and escalate
# An unfilled ENTRY is not chased. An exit must happen at almost any price; an
# entry is optional, and a marketable limit that has not filled in 20s means
# the book moved away -- the setup you pressed the button on is gone. Chasing
# it is how you buy the top of the move you were trying to catch.
ENTRY_CHASE_S = 20.0
# 🚨 DO NOT HAMMER get_order_detail. poll() runs every loop pass, and checking
# an in-flight order on every one of them earned a 429 TOO_MANY_REQUESTS from
# Webull on 2026-09-23 -- which then read as "still working" (see below) and
# kept the order in flight, which kept it being polled. The rate limit and the
# blind-status bug fed each other.
STATUS_POLL_S = 3.0
# Consecutive unreadable status checks before we stop guessing and go look at
# the actual position.
BLIND_BEFORE_VERIFY = 5

# ---- session arming ------------------------------------------------------
# 🚨 TWO FACTORS, BECAUSE REMOTE CHANGES THE THREAT.
# MANUAL_TRADING_ARMED (env, read at import, dies with the process) says this
# machine MAY trade by hand. That was sufficient when arming meant walking to
# your own keyboard. Once the viewer is reachable from a phone over Tailscale,
# a forgotten tab or a picked-up unlocked laptop is armed indefinitely -- so
# the env flag now only grants the CAPABILITY, and an explicit session arm
# grants the WINDOW. The window expires on its own.
#
# EXITS ARE NOT GATED ON EITHER. Arming governs opening; a position you
# already own must always be closeable. Same rule as before.
ARM_WINDOW_S = 900.0           # 15 minutes per arm
ARM_MAX_S = 3600.0             # refuse to arm for longer than this


def arm(eng, seconds=ARM_WINDOW_S):
    """Open the trading window. Returns (ok, message)."""
    from config import MANUAL_TRADING_ARMED
    if not MANUAL_TRADING_ARMED:
        return False, ("MANUAL_TRADING_ARMED is False — the capability is off "
                       "on this machine, so there is nothing to arm")
    s = max(60.0, min(float(seconds or ARM_WINDOW_S), ARM_MAX_S))
    eng.manual_armed_until = time.time() + s
    _event(True, f"ARMED for {s/60:.0f} min — expires "
                 f"{_dt.datetime.fromtimestamp(eng.manual_armed_until):%H:%M:%S}")
    return True, f"armed {s/60:.0f} min"


def disarm(eng, why="manual"):
    eng.manual_armed_until = 0.0
    _event(True, f"DISARMED ({why})")
    return True, "disarmed"


def arm_left(eng):
    """Seconds of arming left. 0 when closed. Never negative."""
    return max(0.0, float(getattr(eng, "manual_armed_until", 0.0) or 0.0)
               - time.time())


def is_armed(eng):
    """Capability AND an open window."""
    try:
        from config import MANUAL_TRADING_ARMED
    except Exception:
        return False
    return bool(MANUAL_TRADING_ARMED) and arm_left(eng) > 0


def extra_dtes(ticker, required):
    """DTEs to track for `ticker` in the manual-only directions.

    Mirrors whatever DTEs the ticker ALREADY tracks for its rules, so the
    manual side of the strip shows the same expiries the bot's own side does
    (SPY 0DTE only; QQQ/IWM 0 and 1). Falls back to 0DTE if a ticker somehow
    has no rules at all. Returns {} when the feature is switched off.
    """
    from config import MANUAL_CHAIN_EXTRAS
    if not MANUAL_CHAIN_EXTRAS:
        return {}
    dirs = EXTRA_DIRECTIONS.get(ticker)
    if not dirs:
        return {}
    have = set().union(*required.values()) if required else set()
    return {d: (have or {0}) for d in dirs}


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_atomic(path, obj):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def _result(req_path, ok, msg, extra=None):
    out = dict(ok=ok, msg=msg, at=int(time.time()))
    if extra:
        out.update(extra)
    try:
        _write_atomic(req_path.replace(".json", ".result.json"), out)
        os.unlink(req_path)
    except OSError:
        pass
    print(f"  {'✅' if ok else '🚫'} manual order: {msg}")
    return out


def load_positions(eng):
    """Rehydrate manual holds after a restart, so the block and the EOD flatten
    survive the bot being bounced mid-session."""
    if getattr(eng, "manual_holds", None) is None:
        eng.manual_holds = _read(STATE) or {}
    return eng.manual_holds


def save_positions(eng):
    try:
        _write_atomic(STATE, getattr(eng, "manual_holds", {}) or {})
    except OSError:
        pass


def adopt(eng, tk, occ, qty, cost):
    """Take an EXISTING broker position under management as a manual hold.

    🚨 THIS IS WHAT THE STARTUP RECONCILE SHOULD DO FOR SPY/QQQ/IWM.
    Its old behaviour -- FLATTEN anything the bot has no record of -- is right
    for a stale bot position left by a crash, and exactly wrong for a
    discretionary position opened deliberately in the broker app: starting the
    bot would sell it out from under you. Adopting instead gives the position
    everything a viewer-placed one gets:
      · the entry block, so the bot will not stack its own trade on top
      · the 15:55 EOD flatten
      · live P&L and the exit buttons in the viewer

    `cost` is Webull's per-share cost_price (0.18 for a $18 contract), the
    same units as a quote, so it drops straight into `entry`.

    Does NOT adopt over an existing hold -- a hold already here came from a
    known fill and carries state (exit_coid) this would destroy.
    """
    holds = load_positions(eng)
    cur = holds.get(tk)
    if cur:
        # 🚨 A HOLD ON THE SAME TICKER IS NOT THE SAME POSITION.
        # This used to `return False` on any existing hold, so swapping one
        # contract for another in the broker app (sell the put, buy the call)
        # left us tracking a contract you no longer owned while the one you
        # DID own had no entry block and no 15:55 flatten. Observed
        # 2026-09-24: recorded 2x IWM 280P, actually held 1x IWM 282C.
        if cur.get("option_id") == occ:
            if int(cur.get("qty") or 0) != int(abs(qty)):
                cur["qty"] = int(abs(qty))
                save_positions(eng)
                _event(True, f"{tk} size corrected to {cur['qty']}x from the "
                             f"broker")
            return False
        if cur.get("pending") or cur.get("exit_coid"):
            # mid-flight: something is in the air on the old contract and
            # stomping the record would orphan it
            _event(False, f"{tk} broker shows {occ} but our hold "
                          f"({cur.get('option_id')}) has an order in flight "
                          f"— NOT replacing it. Check the broker.")
            return False
        _event(True, f"{tk} hold was {cur.get('option_id')}, broker shows "
                     f"{occ} — the record was stale, replacing it")
        holds.pop(tk, None)
    holds[tk] = dict(ticker=tk, option_id=occ, qty=int(abs(qty)),
                     limit=None, entry=float(cost or 0) or None,
                     coid=None, opened=int(time.time()), dry=False,
                     trade_id=f"adopt-{occ}-{int(time.time())}",
                     strike=None, expiry=None, right=None,
                     adopted=True, note="adopted at startup")
    t, exp, right, strike = _parse_occ(occ)
    holds[tk].update(strike=strike, expiry=exp, right=right)
    save_positions(eng)
    # 🚨 `entry` here is the broker's COST BASIS, not a fill we observed, and
    # `opened` is when we adopted it, not when you bought it. The row is
    # flagged adopted=True so analysis can exclude these: the entry context
    # is the state at ADOPTION, which is not the state you decided in.
    _log_manual("open", holds[tk], _context(eng, tk), source="adopted")
    _event(True, f"ADOPTED existing {tk} {int(abs(qty))}x {occ}"
                 + (f" @ {float(cost):.2f}" if cost else "")
                 + " — entry block + 15:55 flatten now cover it")
    return True


def _parse_occ(oid):
    """OCC -> (ticker, expiry, right, strike). Mirrors live_state.parse_occ;
    duplicated so this module has no import-time dependency on the emitter."""
    try:
        tk = "".join(c for c in str(oid)[:6] if c.isalpha())
        r = str(oid)[len(tk):]
        return (tk, f"20{r[0:2]}-{r[2:4]}-{r[4:6]}", r[6], int(r[7:]) / 1000.0)
    except (IndexError, ValueError):
        return (None, None, None, None)


def resting_orders(eng, option_id):
    """Open broker orders that would SELL this exact contract.

    🚨 INCLUDING ONES THIS API CANNOT CREATE. A TP/SL bracket set by hand in
    the Webull app shows up here as TWO orders sharing a `combo_order_id`
    (STOP_LOSS + STOP_PROFIT, each for the FULL position). Observed
    2026-09-23 on a live IWM 284C.

    That matters because the position payload has no available/reserved
    quantity field -- nothing tells us whether those contracts are already
    spoken for. Sending our own SELL on top is either rejected for
    insufficient quantity or, worse, is not, and fills into a short.

    Matching is by OCC rebuilt from the order LEG, because every order's
    top-level `symbol` is the underlying ("IWM"), not the contract.
    """
    out = []
    try:
        for o in (eng.webull.get_open_orders() or []):
            if "OPTION" not in str(o.get("instrument_type", "")).upper():
                continue
            coid = o.get("client_order_id")
            if not coid:
                continue
            raw = o.get("raw") or {}
            for od in (raw.get("orders") or [raw]):
                for lg in (od.get("legs") or []):
                    if str(lg.get("side", "SELL")).upper() != "SELL":
                        continue
                    built = eng.webull._occ_from_leg(lg.get("symbol"), lg)
                    if built and built == option_id:
                        out.append(dict(coid=coid,
                                        kind=raw.get("combo_type")
                                             or od.get("order_type") or "?",
                                        qty=od.get("total_quantity")))
    except Exception as e:                      # noqa: BLE001 -- deliberate
        # 🚨 LOUD, AND IN THE FEED. Failing to LOOK for resting orders is not
        # the same as there being none: the exit still goes out (getting out
        # matters more than being tidy), but it may now be a second SELL on
        # top of a live bracket. That has to be visible in the viewer, not
        # only in a console nobody is reading during a flatten.
        _event(False, f"could not check resting orders on {option_id} "
                      f"({type(e).__name__}: {e}) — the exit will still be "
                      f"sent, but CHECK THE BROKER for duplicate SELLs")
    return out


def cancel_resting(eng, option_id, why="exit"):
    """Clear the book on a contract before we sell it. Returns what it killed."""
    killed = []
    for r in resting_orders(eng, option_id):
        try:
            if eng.webull.cancel_option_order(r["coid"]):
                killed.append(f"{r['kind']} x{r['qty']}")
        except Exception as e:                  # noqa: BLE001
            print(f"  ⚠️ cancel resting {r['coid']}: {type(e).__name__}: {e}")
    if killed:
        _event(True, f"cancelled {len(killed)} resting order(s) on "
                     f"{option_id} before the {why}: {', '.join(killed)}")
    return killed


def attach_bracket(eng, tk, tp=None, sl=None, tp_pct=None, sl_pct=None,
                   dry=False, combo="PAIR"):
    """Put a broker-side TP/SL on a manual hold.

    🚨 WHY THIS IS WORTH HAVING AT ALL, given we already flatten at 15:55:
    our sweep only runs while bot_runner does. Process dies, laptop sleeps,
    MQTT stalls, the machine reboots -- nothing flattens. A GTC bracket
    living at the broker survives every one of those. For the failure the
    user actually named (forgetting a 0DTE), it is strictly more robust than
    anything we can run locally. That is the argument; convenience is not.

    🚨 AND WHY IT IS MANUAL-ONLY. The bot's own exits are a measured model --
    the trail, CUSHION_CAP, EXIT_FLOOR_FRAC, the fill behaviour sim_core
    calibrated. A broker stop underneath one of those would fire first and
    silently change the thing that was measured. Never bracket active_snipes.

    Prices: absolute (`tp`/`sl`) or as a percentage of the entry fill
    (`tp_pct`/`sl_pct`, e.g. 100 and -50). Absolute wins if both are given.

    A stop on a 0DTE option triggers on the OPTION's price, which is erratic
    on a thin book -- this is a disaster backstop, not a risk tool, and a
    tight stop will be taken out by noise.
    """
    holds = load_positions(eng)
    pos = holds.get(tk)
    if not pos:
        return {"ok": False, "err": f"no manual hold on {tk}"}
    if pos.get("pending"):
        return {"ok": False, "err": f"{tk} entry has not filled yet — there "
                                    f"is no position to protect"}
    if pos.get("exit_coid"):
        return {"ok": False, "err": f"{tk} is already exiting"}
    if tk in (getattr(eng, "active_snipes", {}) or {}):
        return {"ok": False, "err": f"{tk} is a BOT position — its exits are "
                                    f"a measured model, never bracket it"}
    if pos.get("bracket"):
        return {"ok": False, "err": f"{tk} already has a bracket "
                                    f"({pos['bracket'].get('combo_order_id')})"}
    # a bracket on top of existing resting SELLs is the double-sell we spent
    # yesterday building cancel_resting to prevent -- refuse to create one
    existing = resting_orders(eng, pos["option_id"])
    if existing:
        return {"ok": False, "err": f"{tk} already has {len(existing)} resting "
                                    f"SELL(s) — cancel them first"}

    entry = float(pos.get("entry") or 0)
    if entry <= 0:
        return {"ok": False, "err": f"{tk} has no known entry price"}
    if tp is None and tp_pct is not None:
        tp = entry * (1 + float(tp_pct) / 100.0)
    if sl is None and sl_pct is not None:
        sl = entry * (1 + float(sl_pct) / 100.0)
    if tp is None or sl is None:
        return {"ok": False, "err": "need tp and sl (absolute or _pct)"}
    tp, sl = round(float(tp), 2), round(float(sl), 2)
    if sl < 0.01 or tp < 0.01:
        return {"ok": False, "err": f"prices must be >= 0.01 (tp {tp}, sl {sl})"}
    if not (sl < tp):
        return {"ok": False, "err": f"stop {sl} must be BELOW target {tp} — "
                                    f"this is a long position"}

    res = eng.webull.place_option_bracket(
        ticker=tk, option_id=pos["option_id"], quantity=int(pos["qty"]),
        tp_limit=tp, sl_stop=sl, dry=dry, combo=combo)
    if dry:
        return {"ok": True, "dry": True, "payload": res.get("payload")}
    if not res.get("accepted"):
        err = str(res.get("error") or res.get("raw"))
        # 🚨 NAME THE LIKELY CAUSE. This error cost an hour of hunting for a
        # payload bug that did not exist: the shape was right and the broker
        # was margining the unnetted leg. Two closing orders exist against one
        # position, and until one fills the other is a potential short --
        # cheap on a call, ~strike x 100 of collateral on a put.
        if "BUYING_POWER" in err.upper():
            err += ("  |  NOT a payload problem: the broker margins the "
                    "unnetted leg. Puts are far more expensive to collateralise "
                    "than calls, and a small account will be refused on a put "
                    "bracket it would accept on a call.")
        _event(False, f"{tk} bracket REJECTED (TP {tp:.2f} / SL {sl:.2f}): {err}")
        return {"ok": False, "err": err}

    pos["bracket"] = dict(combo_order_id=res.get("combo_order_id"),
                          client_order_ids=res.get("client_order_ids"),
                          tp=tp, sl=sl, at=int(time.time()))
    save_positions(eng)
    _event(True, f"{tk} bracket attached: TP {tp:.2f} / SL {sl:.2f} on "
                 f"{pos['qty']}x {pos['option_id']} (GTC, broker-side — "
                 f"survives this process dying)")
    return {"ok": True, "tp": tp, "sl": sl,
            "combo_order_id": res.get("combo_order_id")}


def _quote(eng, option_id):
    q = (eng.webull.tick_builder.live_option_quotes or {}).get(option_id)
    if not q:
        return None, None, None
    b, a = float(q.get("bid") or 0), float(q.get("ask") or 0)
    ts = q.get("timestamp")
    age = (time.time() - float(ts)) if ts else None
    return (b if b > 0 else None), (a if a > 0 else None), age


def _execute(eng, req, path, now_mod=None):
    from config import MANUAL_TRADING_ARMED, DRY_RUN
    tk = str(req.get("ticker", "")).upper()
    oid = str(req.get("option_id", ""))
    qty = int(req.get("qty") or 0)

    # 🚨 EXITS ARE NOT GATED ON ARMED. Arming is what lets a position be
    # OPENED; gating the exit on it too would mean disarming mid-session traps
    # you in a 0DTE with no stop. Nor are they gated on the EOD window, the
    # spread collar or the quote age -- every one of those is a reason to
    # REFUSE AN ENTRY and none of them is a reason to refuse an exit.
    act = str(req.get("action") or "buy").lower()
    if act in ("arm", "disarm"):
        ok, msg = (arm(eng, req.get("seconds")) if act == "arm"
                   else disarm(eng, "viewer"))
        return _result(path, ok, msg)
    if act == "panic":
        r = panic(eng, "911")
        return _result(path, not r["skipped"],
                       f"911: closed {len(r['closed'])}"
                       + (f", COULD NOT CLOSE {r['skipped']}"
                          if r["skipped"] else ""))
    if tk not in ALLOWED:
        return _result(path, False, f"{tk} not in {sorted(ALLOWED)}")
    if act == "close":
        mode = EXIT_911 if str(req.get("mode")) == EXIT_911 else EXIT_MARKETABLE
        if tk not in load_positions(eng):
            # also clears a hold whose position you closed in the broker app,
            # which would otherwise keep the ticker blocked and send 15:55
            # chasing something you no longer own
            return _result(path, False, f"no manual hold on {tk} to close")
        ok = flatten(eng, tk, "viewer close", mode)
        return _result(path, ok, f"{tk} close sent ({mode})" if ok
                       else f"{tk} close FAILED — position still open")
    if not MANUAL_TRADING_ARMED:
        return _result(path, False, "MANUAL_TRADING_ARMED is False — not sent")
    if arm_left(eng) <= 0:
        return _result(path, False,
                       "the arming window is CLOSED — arm from the viewer "
                       f"(it opens for {ARM_WINDOW_S/60:.0f} min and expires "
                       f"on its own)")
    # 🚨 NO BUYS ONCE THE FLATTEN WINDOW IS OPEN. poll() runs the EOD sweep
    # BEFORE it drains requests, so an order fired at 15:56 would open a
    # position the sweep has already passed over -- it would sit there until
    # the NEXT loop pass sold it, paying the spread both ways for nothing, and
    # at 15:59 it would sit there until tomorrow.
    if now_mod is not None and now_mod >= EOD_FLATTEN_MOD:
        return _result(path, False, f"past the {EOD_FLATTEN_MOD//60}:"
                                    f"{EOD_FLATTEN_MOD%60:02d} flatten — no new "
                                    f"positions")
    if qty < 1 or qty > MAX_CONTRACTS:
        return _result(path, False, f"qty {qty} outside 1..{MAX_CONTRACTS}")
    if tk in (getattr(eng, "active_snipes", {}) or {}):
        return _result(path, False, f"{tk} already held by the BOT")
    if tk in load_positions(eng):
        return _result(path, False, f"{tk} already held manually")

    bid, ask, age = _quote(eng, oid)
    if bid is None or ask is None:
        return _result(path, False, f"no live quote for {oid}")
    if age is not None and age > MAX_QUOTE_AGE:
        return _result(path, False, f"quote is {age:.0f}s stale (max "
                                    f"{MAX_QUOTE_AGE:.0f}s)")
    mid = (bid + ask) / 2.0
    if mid > 0 and (ask - bid) / mid > MAX_SPREAD_PCT:
        return _result(path, False,
                       f"spread {bid:.2f}/{ask:.2f} is "
                       f"{(ask-bid)/mid*100:.0f}% of mid (max "
                       f"{MAX_SPREAD_PCT:.0%}) — quote not trustworthy")
    # the same marketable limit bot_runner._fire_entry uses (bot_runner.py:1416)
    limit = min(round(mid + 0.01, 2), round(ask, 2))
    prem = limit * qty * 100
    cap, basis = max_premium(eng)
    if prem > cap:
        return _result(path, False,
                       f"premium ${prem:,.0f} over the ${cap:,.0f} cap "
                       f"({basis})")

    # 🚨 ARMED MEANS REAL. DRY_RUN GOVERNS THE BOT, NOT YOUR HAND.
    # This used to send only when DRY_RUN was False, which meant the single
    # config that let you trade by hand also turned the bot loose on all nine
    # tickers -- there was no way to run a live hand against a paper bot, which
    # is the only sane way to roll this out. The flags are now what the module
    # docstring always said they were: MANUAL_TRADING_ARMED decides whether
    # YOUR orders are real, DRY_RUN decides whether the BOT's are. There is
    # deliberately no "armed but pretend" state; the whole lesson of this
    # codebase is that a thing which looks live while being fake is the
    # dangerous thing. Disarmed shows the entire UI and refuses at the guard.
    res = eng.webull.place_option_order(
        ticker=tk, option_id=oid, action="BUY", quantity=qty,
        is_closing=False, order_type="LIMIT", limit_price=limit)
    coid = (res or {}).get("client_order_id")
    if not coid:
        return _result(path, False, f"broker rejected: {res}")

    # 🚨 PENDING, NOT HELD. The old code wrote `fill = limit` the instant the
    # order was SENT -- the same assume-don't-confirm bug that was on the exit
    # side, and worse here: an unfilled marketable limit created a hold for a
    # position that does not exist, which 15:55 then tried to SELL. It also
    # recorded the price we were WILLING to pay as the price we GOT, so the
    # panel P&L and every exit decision inherited the slippage as profit --
    # the "paper ledger fiction" _fire_exit's docstring warns about, rebuilt.
    # _check_entries promotes this to a real hold on a CONFIRMED fill, at the
    # ACTUAL fill price and the ACTUAL filled quantity.
    load_positions(eng)[tk] = dict(
        ticker=tk, option_id=oid, qty=qty, limit=limit, entry=None,
        coid=coid, entry_coid=coid, entry_at=time.time(), pending=True,
        trade_id=coid,          # the broker's own id links open to close
        opened=int(time.time()), dry=False,
        strike=req.get("strike"), expiry=req.get("expiry"),
        right=req.get("right"), note=str(req.get("note", ""))[:200])
    save_positions(eng)
    return _result(path, True,
                   f"{tk} BUY {qty}x {oid} @ limit {limit:.2f} "
                   f"(${prem:,.0f}) — SENT, awaiting fill",
                   dict(limit=limit, qty=qty, premium=prem, coid=coid))


KEEP_EVENTS = 400              # the viewer only ever reads the newest 60

# 🚨 A SEPARATE FILE FROM bot_executions_log.jsonl, AND IT MUST STAY SEPARATE.
# That ledger is the RESEARCH POPULATION -- research_rules and the sim read it,
# and every study's trade counts come out of it. Discretionary trades are
# selected by a human on unrecorded criteria; folding them in would silently
# contaminate the population that every backtest result is measured against.
# They are a dataset of their OWN, to be analysed on their own terms.
#
# Never pruned. The telemetry feed (live/orders) is a rolling window capped at
# KEEP_EVENTS; this is the permanent record. A manual round trip used to exist
# ONLY in that rolling window, so the P&L of a completed discretionary trade
# evaporated -- which is exactly the mechanism by which winners get remembered
# and the identical setups that failed never get enumerated.
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "manual_executions.jsonl")


def _log_manual(kind, pos, ctx=None, **extra):
    """Append one append-only row. `kind` is "open" or "close", linked by
    trade_id, so an entry survives even if the exit never happens (expiry, a
    crash, a hand-close in the app) and orphans stay VISIBLE rather than
    quietly absent. Never raises -- a ledger write must not break a fill."""
    try:
        # A hold persisted by an older build has no trade_id. Synthesise a
        # stable one rather than writing null, or the open/close join silently
        # collapses every such trade into a single "None" bucket.
        tid = pos.get("trade_id") or (
            f"legacy-{pos.get('option_id')}-{pos.get('opened') or 0}")
        rec = dict(kind=kind, ts=int(time.time()), trade_id=tid,
                   ticker=pos.get("ticker"), option_id=pos.get("option_id"),
                   strike=pos.get("strike"), right=pos.get("right"),
                   expiry=pos.get("expiry"), qty=pos.get("qty"),
                   entry=pos.get("entry"), entry_limit=pos.get("limit"),
                   adopted=bool(pos.get("adopted")),
                   dry=bool(pos.get("dry")), note=pos.get("note") or "")
        rec.update(extra)
        if ctx:
            rec["ctx"] = ctx
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":"),
                               default=str) + "\n")
    except Exception as e:                      # noqa: BLE001 -- deliberate
        print(f"  ⚠️ manual ledger write failed ({kind}): "
              f"{type(e).__name__}: {e}")


def _context(eng, tk):
    """Market state at this instant, for the ledger. Same schema as the
    viewer's note capture, so notes and fills join field-for-field."""
    try:
        import live_state
        return live_state.decision_context(eng, tk)
    except Exception:
        return None


def _event(ok, msg):
    """A telemetry event with no originating request -- EOD sweeps, 911s and
    fill confirmations. Written into live/orders as a bare .result.json so it
    lands in the viewer's Guard Telemetry feed alongside everything else; an
    exit that happened without you asking is exactly what the feed is for.

    Pruned, because these are no longer only user-initiated: fills, stalls and
    escalations all write one, the viewer re-lists the whole directory every
    5s, and nothing else ever deletes them.
    """
    try:
        _write_atomic(os.path.join(DIR, f"{int(time.time()*1e6)}.result.json"),
                      dict(ok=bool(ok), msg=str(msg), at=int(time.time())))
        fns = sorted(f for f in os.listdir(DIR) if f.endswith(".result.json"))
        for f in fns[:-KEEP_EVENTS]:            # filenames are microsecond ts
            try:
                os.unlink(os.path.join(DIR, f))
            except OSError:
                pass
    except OSError:
        pass
    print(f"  {'✅' if ok else '🚫'} {msg}")


def _exit_price(bid, ask, mode):
    """Where to put a closing SELL limit.

    🚨 THE LIMIT IS NOT THE PRICE YOU GET. A marketable SELL limit fills
    against the RESTING BID (86.4% at exactly the bid; sim_core:195), so the
    number here is not what you receive -- it is how far the bid may fall
    while the order travels and still find you. Both modes normally fill at
    the bid. 911 is not "sell for half"; it is "sell even if the bid halves
    on the way".

    marketable : the house convention for a non-directional exit, the same
                 arithmetic as the shutdown flatten (bot_runner.py:879) and
                 the startup reconcile -- bid less FILL_COST spreads, floored
                 at EXIT_FLOOR_FRAC of the bid.
    911        : the floor itself.
    """
    from config import EXIT_FLOOR_FRAC, FILL_COST
    b = float(bid or 0.0)
    floor = max(0.01, round(b * EXIT_FLOOR_FRAC, 2))
    if mode == EXIT_911 or b <= 0:
        return floor
    spread = max(0.0, float(ask or 0.0) - b)
    return max(floor, round(b - spread * FILL_COST, 2))


def flatten(eng, tk, reason="MANUAL EOD", mode=EXIT_MARKETABLE):
    """Place a closing SELL for a manual hold.

    🚨 THE HOLD IS NOT DROPPED UNTIL THE FILL IS CONFIRMED. The first version
    popped manual_holds[tk] the moment the order was SENT. A rejected or
    unfilled exit therefore left a live 0DTE position the bot had forgotten:
    the ticker unblocked for a bot entry on top of it, and the 15:55 sweep no
    longer knew it existed. The coid is stored on the position and
    `_check_exits` pops it on FILLED, chasing if it stalls.

    Returns True when an exit is in flight or already filled.
    """
    from config import DRY_RUN
    pos = load_positions(eng).get(tk)
    if not pos:
        return False
    if pos.get("exit_coid"):
        return True                     # already in flight; _check_exits owns it
    if pos.get("pending"):
        # Nothing is owned yet -- the right "close" is to cancel the ENTRY.
        # Selling here would open a short in a contract we do not hold.
        if not pos.get("cancel_sent") and pos.get("entry_coid"):
            try:
                eng.webull.cancel_option_order(pos["entry_coid"])
            except Exception as e:              # noqa: BLE001
                print(f"  ⚠️ cancel pending entry {tk}: {type(e).__name__}: {e}")
            pos["cancel_sent"] = True
            save_positions(eng)
            _event(True, f"{tk} entry still pending — CANCELLED instead of sold "
                         f"({reason})")
        return True
    # 🚨 THREE VALUES. `_quote` grew an `age` return when the spread collar was
    # rewritten (2026-09-22) and this call was left unpacking two, so flatten()
    # raised ValueError on EVERY call -- swallowed by poll()'s except, which
    # printed a warning and left the position open. The one exit that protects
    # against forgetting a 0DTE was silently dead. Caught by test_order_loop.
    bid, ask, _age = _quote(eng, pos["option_id"])
    px = _exit_price(bid, ask, mode)
    qty = int(pos["qty"])

    if pos.get("dry"):
        # a simulated fill gets a simulated exit: book at the bid, not at the
        # limit, the way _fire_exit does
        fill = max(0.01, round(min(bid or px, max(px, bid or px)), 2))
        _event(True, f"{tk} CLOSED {qty}x @ ~{fill:.2f} ({reason}) [DRY]")
        entry = float(pos.get("entry") or 0)
        _log_manual("close", pos, _context(eng, tk), exit=fill,
                    exit_mode=mode, exit_reason=reason,
                    roe_pct=(round((fill - entry) / entry * 100, 2)
                             if entry > 0 else None))
        load_positions(eng).pop(tk, None)
        save_positions(eng)
        return True

    # 🚨 A REAL POSITION ALWAYS GETS A REAL EXIT. No DRY_RUN check, no ARMED
    # check. `dry` is a property of the POSITION, not of a global flag: a hold
    # adopted from the broker, or one opened while armed, is real regardless of
    # what the bot is doing. Both flags govern OPENING -- gating the exit on
    # them means a paper or disarmed bot watches your 0DTE expire because it
    # was not "allowed" to close it, which is the exact failure the 15:55
    # sweep exists to prevent. Closing is only ever risk-reducing, and the
    # only way manual_holds contains a real position is that you own it.
    if DRY_RUN:
        print(f"  ⚠️ {tk}: DRY_RUN=True but this position is REAL — sending a "
              f"real closing order. DRY_RUN governs the BOT's trades.")

    # 🚨 DOES THE POSITION STILL EXIST? Ask before selling it.
    # We adopt at startup and then never re-check. On 2026-09-23 a broker-side
    # bracket closed an adopted IWM position mid-session; our hold went stale,
    # and at 15:55 the sweep sent a SELL for 2 contracts that were not there.
    # It was tagged SELL_TO_CLOSE so nothing bad came of it, but the order
    # should never have left. One extra call on a path taken a handful of
    # times a day is cheap insurance against selling what you do not own.
    # `None` means we could not tell -- proceed, because failing to check is
    # not a reason to block an exit.
    if _position_gone(eng, pos["option_id"]) is True:
        # 🚨 PULL THE BOOK ON THE WAY OUT. If a bracket leg filled and the OCO
        # linkage did not cancel its partner, that partner is still resting on
        # a contract we no longer own -- a live SELL with nothing behind it.
        cancel_resting(eng, pos["option_id"], why="position gone")
        _event(True, f"{tk} is no longer held at the broker (closed by hand "
                     f"or by your own bracket) — nothing to sell, releasing "
                     f"the hold instead of sending an order")
        load_positions(eng).pop(tk, None)
        save_positions(eng)
        return True

    # 🚨 CLEAR THE BOOK FIRST. A bracket set by hand in the broker app rests
    # as two SELLs for the FULL position; ours would be a third. Cancelling is
    # unambiguously right here -- flatten is only ever called because you have
    # decided to be out, and a TP that fires after we sell is a naked short.
    if cancel_resting(eng, pos["option_id"], why=reason):
        # the bracket is gone from the book, so the record of it must go too,
        # or a re-armed exit would refuse to re-attach one later
        pos.pop("bracket", None)
        save_positions(eng)

    try:
        res = eng.webull.place_option_order(
            ticker=tk, option_id=pos["option_id"], action="SELL",
            quantity=qty, is_closing=True, order_type="LIMIT", limit_price=px)
    except Exception as e:                      # noqa: BLE001 -- deliberate
        _event(False, f"{tk} exit FAILED to send: {type(e).__name__}: {e} "
                      f"— POSITION STILL OPEN")
        return False
    coid = (res or {}).get("client_order_id")
    if not coid:
        _event(False, f"{tk} exit NOT ACCEPTED by the broker — POSITION STILL "
                      f"OPEN ({res})")
        return False
    pos.update(exit_coid=coid, exit_at=time.time(), exit_px=px,
               exit_mode=mode, exit_reason=reason)
    save_positions(eng)
    _event(True, f"{tk} exit sent: SELL {qty}x @ limit {px:.2f} "
                 f"({reason}, {mode})")
    return True


def _poll_due(pos, key):
    """True if this order is due for a status check. Rate-limit protection.

    🚨 THE KEY IS PER-PHASE, NOT PER-POSITION. Entry and exit polls both live
    on the same dict; sharing one timestamp meant a just-promoted entry
    blocked the first exit status check for STATUS_POLL_S, and on a fast
    close the exit looked unconfirmed for no reason.
    """
    if time.time() - float(pos.get(key) or 0) < STATUS_POLL_S:
        return False
    pos[key] = time.time()
    return True


def _status(eng, coid):
    """(status, filled_qty, fill_price, readable).

    🚨 `readable=False` MEANS WE ARE BLIND, NOT THAT THE ORDER IS WORKING.
    get_order_status returns status=None when the CHECK failed -- a 429, a
    network blip, a parse error. Treating that as "not filled yet" is what
    kept a dead IWM exit in flight for minutes on 2026-09-23, warning every
    20s about an order for a position that no longer existed. The distinction
    is the same one get_live_option_quote_ex draws with its `ok` flag, and it
    matters for the same reason.
    """
    try:
        s = eng.webull.get_order_status(coid) or {}
    except Exception as e:                      # noqa: BLE001 -- deliberate
        print(f"  ⚠️ order status {coid}: {type(e).__name__}: {e}")
        return None, 0.0, 0.0, False
    st = s.get("status")
    return (st, float(s.get("filled_qty") or 0),
            float(s.get("fill_price") or 0), st is not None)


def _position_gone(eng, option_id):
    """Does the broker still show this contract? None if we cannot tell.

    The ground truth an order status cannot give us. A hold closed by hand in
    the app, or by a broker-side bracket, leaves our memory stale and every
    order we send against it meaningless.
    """
    try:
        pos = eng.webull.get_open_option_positions() or []
    except Exception:                           # noqa: BLE001
        return None
    for p in pos:
        if p.get("occ") == option_id and p.get("quantity"):
            return False
    return True


def _check_entries(eng):
    """Promote a PENDING manual entry to a real hold, on a confirmed fill only.

    Mirrors _check_exits. The hold's `qty` and `entry` come from the BROKER's
    filled quantity and fill price, never from what we asked for, so the
    position in memory is the position you own and the P&L is the P&L you got.

    🚨 THE CANCEL/FILL RACE IS HANDLED BY NOT POPPING ON CANCEL-SENT.
    A stalled entry sends a cancel and then WAITS for the status to confirm
    it. If the cancel lost the race and the order actually filled, the next
    pass sees FILLED and creates the hold correctly. Popping at cancel-time
    would have left a real position with nothing tracking it -- no entry
    block, no 15:55 flatten, invisible in the viewer.
    """
    for tk, pos in list((getattr(eng, "manual_holds", None) or {}).items()):
        if not pos.get("pending"):
            continue
        coid = pos.get("entry_coid")
        if not coid or not _poll_due(pos, "last_entry_poll"):
            continue
        st, fq, fpx, readable = _status(eng, coid)
        if not readable:
            # blind, not "still working" -- never advance the stall clock on a
            # check that did not happen
            pos["blind"] = int(pos.get("blind") or 0) + 1
            if pos["blind"] == BLIND_BEFORE_VERIFY:
                _event(False, f"{tk} entry status unreadable "
                              f"{pos['blind']}x (rate limit / network) — "
                              f"state UNKNOWN, CHECK THE BROKER")
            pos["entry_at"] = time.time()       # do not time out while blind
            save_positions(eng)
            continue
        pos.pop("blind", None)
        if st == "FILLED" or (st == "PARTIAL_FILLED" and fq > 0):
            want = int(pos.get("qty") or 0)
            got = int(fq) or want
            fill = fpx or float(pos.get("limit") or 0)
            if st == "PARTIAL_FILLED" and got < want:
                # own what filled, cancel the rest -- a hold whose qty does not
                # match the account is a flatten that leaves a stub behind
                try:
                    eng.webull.cancel_option_order(coid)
                except Exception:               # noqa: BLE001
                    pass
            pos.update(pending=False, qty=got, entry=round(fill, 2),
                       entry_coid=None, filled_at=int(time.time()))
            save_positions(eng)
            slip = (fill - float(pos.get("limit") or fill))
            # context captured AT THE FILL, not at the click and not at the
            # exit -- it is the state the position was actually opened into
            _log_manual("open", pos, _context(eng, tk), source="viewer",
                        partial=(got < want), requested_qty=want,
                        slip=round(slip, 4) if pos.get("limit") else None)
            _event(True, f"{tk} entry FILLED {got}x @ {fill:.2f}"
                         + (f" (limit {pos['limit']:.2f}, "
                            f"{slip:+.2f} slip)" if pos.get("limit") else "")
                         + (f" [PARTIAL of {want}]"
                            if got < want else ""))
            continue
        if st in ("CANCELLED", "FAILED"):
            load_positions(eng).pop(tk, None)
            save_positions(eng)
            _event(False, f"{tk} entry {st} — no position taken")
            continue
        if time.time() - float(pos.get("entry_at") or 0) > ENTRY_CHASE_S:
            if pos.get("cancel_sent"):
                continue                        # waiting on the status to settle
            try:
                eng.webull.cancel_option_order(coid)
            except Exception as e:              # noqa: BLE001
                print(f"  ⚠️ cancel entry {tk}: {type(e).__name__}: {e}")
            pos["cancel_sent"] = True
            save_positions(eng)
            _event(False, f"{tk} entry unfilled after {ENTRY_CHASE_S:.0f}s — "
                          f"cancel sent, not chasing")


def _check_exits(eng):
    """Follow up in-flight manual exits. Pops ONLY on a confirmed fill.

    Manual holds have no bracket loop behind them -- this is the entire
    follow-up machinery, so it has to do the three things _manage_exit_order
    does for the bot: confirm, re-arm on a cancel, and escalate on a stall.
    """
    from config import DRY_RUN
    for tk, pos in list((getattr(eng, "manual_holds", None) or {}).items()):
        coid = pos.get("exit_coid")
        if not coid or not _poll_due(pos, "last_exit_poll"):
            continue
        st, fq, fpx, readable = _status(eng, coid)
        if not readable:
            # 🚨 BLIND IS NOT "STILL WORKING". This is the 2026-09-23 failure:
            # a 429 returned status=None, the old code fell through to "keep
            # waiting", and it warned every 20s about an order for a position
            # that had already been closed by a broker-side bracket. After a
            # few blind checks, stop guessing and ask the ACCOUNT instead.
            pos["blind"] = int(pos.get("blind") or 0) + 1
            pos["exit_at"] = time.time()        # do not time out while blind
            if pos["blind"] >= BLIND_BEFORE_VERIFY:
                gone = _position_gone(eng, pos["option_id"])
                if gone is True:
                    _event(True, f"{tk} position no longer exists at the "
                                 f"broker (closed by hand or by your own "
                                 f"bracket) — releasing the stale hold")
                    load_positions(eng).pop(tk, None)
                    save_positions(eng)
                    continue
                if pos["blind"] == BLIND_BEFORE_VERIFY:
                    _event(False, f"{tk} exit status unreadable "
                                  f"{pos['blind']}x and the position still "
                                  f"shows — state UNKNOWN, CHECK THE BROKER")
            save_positions(eng)
            continue
        pos.pop("blind", None)
        if st == "FILLED" or (st == "PARTIAL_FILLED" and fq > 0):
            fill = fpx or float(pos.get("exit_px") or 0)
            entry = float(pos.get("entry") or 0)
            roe = ((fill - entry) / entry * 100) if entry > 0 else None
            _event(True, f"{tk} exit FILLED @ {fill:.2f}"
                         + (f" ({roe:+.0f}% on {entry:.2f})" if roe is not None
                            else "")
                         + (" [PARTIAL]" if st == "PARTIAL_FILLED" else ""))
            held = (int(time.time()) - int(pos.get("filled_at")
                                           or pos.get("opened") or 0))
            _log_manual("close", pos, _context(eng, tk),
                        exit=round(fill, 2), exit_limit=pos.get("exit_px"),
                        exit_mode=pos.get("exit_mode"),
                        exit_reason=pos.get("exit_reason"),
                        roe_pct=round(roe, 2) if roe is not None else None,
                        pnl=(round((fill - entry) * int(pos.get("qty") or 0)
                                   * 100, 2) if entry > 0 else None),
                        held_s=held if held > 0 else None,
                        partial=(st == "PARTIAL_FILLED"))
            load_positions(eng).pop(tk, None)
            save_positions(eng)
            continue
        if st in ("CANCELLED", "FAILED"):
            # re-arm: the next flatten() call (EOD sweep or a button) re-sends
            pos.pop("exit_coid", None)
            save_positions(eng)
            _event(False, f"{tk} exit {st} — hold RESTORED, still open")
            continue
        if time.time() - float(pos.get("exit_at") or 0) > EXIT_CHASE_S:
            if pos.get("exit_mode") == EXIT_911:
                # Before warning again: is the position even there? A 911 that
                # cannot fill is usually a book problem, but on 2026-09-23 it
                # was a hold that had been closed by the user's own bracket --
                # and the loop warned six times about an order that could
                # never fill because there was nothing to sell.
                if _position_gone(eng, pos["option_id"]) is True:
                    _event(True, f"{tk} 911 could not fill because the "
                                 f"position is already GONE at the broker — "
                                 f"releasing the stale hold")
                    load_positions(eng).pop(tk, None)
                    save_positions(eng)
                    continue
                # already at the floor; nothing left to give. Say so LOUDLY
                # rather than escalating into a market order on a 0DTE book.
                _event(False, f"{tk} 911 exit unfilled after "
                              f"{time.time()-float(pos['exit_at']):.0f}s at "
                              f"limit {float(pos.get('exit_px') or 0):.2f} — "
                              f"CLOSE IT IN THE BROKER APP")
                pos["exit_at"] = time.time()    # re-warn, do not spam
                save_positions(eng)
                continue
            if not DRY_RUN:
                eng.webull.cancel_option_order(coid)
            pos.pop("exit_coid", None)
            save_positions(eng)
            _event(False, f"{tk} exit stalled {EXIT_CHASE_S:.0f}s — cancelled, "
                          f"escalating to 911 pricing")
            flatten(eng, tk, f"{pos.get('exit_reason','EXIT')} ESCALATED",
                    EXIT_911)


def panic(eng, reason="911"):
    """Emergency flatten: EVERY manual hold and EVERY bot position.

    Manual holds go out at the 911 price. Bot positions go through
    bot_runner._fire_exit instead -- not because they deserve a softer price,
    but because that path keeps the ledger, the allocator release and
    active_snipes consistent, and hands the order to _manage_exit_order, which
    cancels and re-prices if it stalls. A manual hold has no such loop, which
    is precisely why it needs the deep limit and the bot's does not.

    A position already mid-exit is LEFT ALONE. Firing a second SELL on top of
    an in-flight one is how you end up short a contract you never owned.
    """
    done, skipped = [], []
    for tk in list((getattr(eng, "manual_holds", None) or {})):
        if flatten(eng, tk, reason, EXIT_911):
            done.append(f"manual {tk}")
        else:
            skipped.append(f"manual {tk} (send failed)")

    for tk, snipe in list((getattr(eng, "active_snipes", None) or {}).items()):
        if snipe.get("exiting"):
            skipped.append(f"bot {tk} (exit already in flight)")
            continue
        bid, ask, _age = _quote(eng, snipe.get("option_id"))
        if not bid or not ask:
            skipped.append(f"bot {tk} (no live quote — CHECK IT MANUALLY)")
            continue
        try:
            eng._fire_exit(tk, snipe, float(bid), float(ask), False,
                           f"{reason} FLATTEN")
            done.append(f"bot {tk}")
        except Exception as e:                  # noqa: BLE001 -- deliberate
            skipped.append(f"bot {tk} ({type(e).__name__}: {e})")

    _event(not skipped,
           f"🚨 911 FLATTEN — closed: {', '.join(done) or 'nothing open'}"
           + (f" | NOT CLOSED: {', '.join(skipped)}" if skipped else ""))
    return dict(closed=done, skipped=skipped)


def poll(eng, now_mod=None):
    """Called once per bot_runner loop pass. Never raises."""
    try:
        load_positions(eng)
        # ENTRIES, THEN EXITS, THEN THE EOD SWEEP. A pending entry that just
        # filled must become a hold BEFORE the sweep looks at what is held, or
        # a position filled at 15:54:59 is invisible to the 15:55 flatten and
        # sits over the weekend. Confirming exits before the sweep likewise
        # stops it re-firing on something already sold.
        _check_entries(eng)
        _check_exits(eng)
        # EOD FLATTEN -- it must run even if a request is malformed, and even
        # after entries have closed for the day. It goes out MARKETABLE and
        # `_check_exits` escalates to 911 pricing if it has not filled in 20s,
        # so the good price is tried first without risking the position.
        if now_mod is not None and now_mod >= EOD_FLATTEN_MOD:
            for tk in list(getattr(eng, "manual_holds", {}) or {}):
                flatten(eng, tk, "EOD 15:55", EXIT_MARKETABLE)
        if not os.path.isdir(DIR):
            return
        for fn in sorted(os.listdir(DIR)):
            if not fn.endswith(".json") or fn.endswith(".result.json"):
                continue
            p = os.path.join(DIR, fn)
            req = _read(p)
            if req is None:
                try:
                    os.unlink(p)
                except OSError:
                    pass
                continue
            _execute(eng, req, p, now_mod)
    except Exception as e:                      # noqa: BLE001 -- deliberate
        try:
            print(f"  ⚠️ manual_orders.poll: {type(e).__name__}: {e}")
        except Exception:
            pass

