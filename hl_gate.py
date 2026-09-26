"""
hl_gate.py
==========
The single choke point every Hyperliquid order passes through.

🚨 WHY THIS FILE EXISTS, WITH A DATE ON IT.
Audited 2026-09-26, ahead of publishing this toolkit. `hyper_exposure_client.py`
exposed eleven methods that reach `exchange.order()` and NOT ONE of them checked
a safety flag. The flags lived in the CALLERS: `gamma_scalper_engine` guards four
call sites with `if not DRY_RUN`, and misses `close_position` entirely.
`market_order`, `limit_order`, `set_stop_loss` and `set_take_profit` were
reachable with no flag at all from anywhere in the repo.

That is not a safety model, it is a habit. The same audit found `main.py` at
module scope calling `client.set_take_profit()` -- an unguarded import placed a
live reduce-only order on a funded account, and the ONLY reason it did not is
that the call is missing its four required arguments and raised TypeError first.
Fill those arguments in and importing a file trades your account.

So the gate moved to the exchange boundary. A gate you have to remember to call
is not a gate; this one cannot be bypassed by forgetting.

Compare the Webull side, which has had this for weeks: DRY_RUN, a separate
MANUAL_TRADING_ARMED capability, a session arm that expires, per-order premium
caps, a ticker allow-list, a peer interlock and a ledger. Hyperliquid had none
of it -- while being the more dangerous venue of the two, because perps are
leveraged and liquidatable and options are not.

---------------------------------------------------------------------------
TWO INDEPENDENT FACTORS, BECAUSE ONE IS A REFLEX.

    HL_TRADING_ENABLED=true            the capability
    HL_LEGAL_ACK="<the exact phrase>"  the attestation

Enabling a feature must not imply you attested to anything, so they are separate
and BOTH are required. The attestation is a PHRASE, not a boolean, for the same
reason `serve_viewer` spells out `--i-know-this-is-public`: `HL_LEGAL_ACK=true`
is something you set by reflex at 09:25 while fixing something else. A phrase
you have to copy is a phrase you had to read.

---------------------------------------------------------------------------
🚨 RISK-REDUCING ORDERS ARE NOT GATED. THIS IS DELIBERATE.

Anything that can only make your exposure smaller -- reduce-only orders, the
flatten, stop-losses, take-profits, cancels -- passes on nothing but a live
exchange object. It is still logged; it is never refused.

The reason is the lesson the options side already learned the hard way: `dry` is
a property of the POSITION, not a global flag. A real position always gets a
real exit. If the flag were a prerequisite for exiting, then turning the feature
off while holding a leveraged perp would trap you in it, and an unreachable
stop-loss on a 20x position is a liquidation, not a safety feature.

It self-selects correctly, too: `close_position` reads real positions off-chain.
If it finds one, the position is real regardless of what DRY_RUN says.

---------------------------------------------------------------------------
🚨 EVERY LIMIT FAILS CLOSED. AN UNSET CAP IS NOT AN INFINITE CAP.

HL_MAX_NOTIONAL_USD defaults to 0 and HL_ALLOWED_COINS defaults to empty, so a
box with HL_TRADING_ENABLED=true and nothing else set trades NOTHING. You have
to say what you want, in dollars, per coin.

This is the opposite of the mistake still sitting in `manual_orders.py`, where
MANUAL_MAX_PREMIUM_PCT = 1.0 means a single fat-fingered order may spend 100% of
buying power. That default was written when the account held test money.

HL_MAX_LEVERAGE defaults to 1 -- i.e. none. Leverage is the entire difference
between "a trade that went against me" and "a trade that closed itself at the
worst possible price while I was asleep."

---------------------------------------------------------------------------
NOT LEGAL ADVICE. See LEGAL_NOTICE below, and read it before setting the ack.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# THE LEGAL NOTICE
# ---------------------------------------------------------------------------
# The sections cited below were confirmed by the operator on 2026-09-26.
#
# What is written here is a plain-language summary of why this venue is locked.
# It is NOT a legal opinion, it was NOT drafted by a lawyer, and it is not a
# substitute for reading the statute or taking advice. It states no dollar
# thresholds on purpose: the ECP definition has several limbs, they are amended,
# and a number remembered wrongly is worse than a pointer to the section.
#
# 🚨 DO NOT EDIT THIS TO MAKE IT SHORTER OR SOFTER, and do not move it behind a
# flag. The whole point of the copied-phrase gate below is that this text is read
# once, in full, by anyone who turns the feature on. Every line of it is either
# a fact about the venue or a fact about the law; none of it is decoration.

CITATION = """\
    "Eligible contract participant" is defined at Commodity Exchange Act
    § 1a(18), codified at 7 U.S.C. § 1a(18). The retail commodity transaction
    provisions are at CEA § 2(c)(2), codified at 7 U.S.C. § 2(c)(2)."""

LEGAL_NOTICE = f"""
================================================================================
  HYPERLIQUID PERPETUAL FUTURES -- READ THIS BEFORE ENABLING
================================================================================

  WHAT THIS FEATURE DOES
    Places leveraged perpetual futures orders on Hyperliquid, an offshore
    decentralised exchange, signed by an agent key held in your .env. These are
    real orders against real funds on a venue with no customer-protection
    regime, no segregated accounts, no SIPC or FDIC coverage, and no recourse.
    Positions are liquidatable. You can lose the entire margin balance, and a
    gap through your stop can cost more than the notional you intended to risk.

  WHY IT IS LOCKED
    Hyperliquid is not registered with the CFTC as a designated contract market
    and its perpetual futures are not listed on any US exchange. US persons are
    generally restricted from trading off-exchange leveraged derivatives of this
    kind; the principal carve-out turns on qualifying as an "eligible contract
    participant", which is an asset-based test that ordinary retail traders do
    not meet. Hyperliquid's own terms of service separately restrict access by
    US persons and by persons in other sanctioned or excluded jurisdictions.

    Relevant authority:
{CITATION}

    ==> AS OF THIS RELEASE, IF YOU ARE A US PERSON AND YOU DO NOT MEET THAT
        THRESHOLD, TRADING HYPERLIQUID PERPETUALS IS NOT LAWFUL FOR YOU.
        Using a VPN, a non-US wallet, or an intermediary to obtain access does
        not change that, and may add offences of its own.

  WHAT YOU ARE ATTESTING
    By setting HL_LEGAL_ACK you state that you have read this notice, that you
    have determined your own eligibility, and that you accept full legal and
    financial responsibility for every order this software sends. The authors of
    this software are not your lawyers, are not your brokers, and are not a
    party to your trades. Nothing here is legal, tax or investment advice.

  IF YOU ARE UNSURE, YOU ARE NOT ELIGIBLE. LEAVE THIS FEATURE OFF.
    It is off by default, and the rest of this toolkit -- the flow engine, the
    options executor, the research scripts -- runs completely without it.
================================================================================
"""

# The literal string HL_LEGAL_ACK must equal. Long on purpose: it has to be
# copied, which means the notice above has to be looked at.
ACK_PHRASE = "I HAVE READ THE HYPERLIQUID LEGAL NOTICE AND I AM ELIGIBLE"

# ---------------------------------------------------------------------------
# CONFIGURATION -- all fail-closed
# ---------------------------------------------------------------------------
LEDGER = "hl_executions.jsonl"
KEEP_EVENTS = 400


def _flag(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("true", "1", "yes", "on")


def _num(name, default):
    """float(os.getenv(...)) with the empty string treated as absent.

    🚨 `float("")` raises, and `float(os.getenv(name) or default)` turns a
    deliberate HL_MAX_NOTIONAL_USD= (set, but empty) into the default. Both are
    wrong in the unsafe direction for a cap, so parse it explicitly and refuse
    to guess: an unparseable cap is 0, which refuses everything.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"  🚨 {name}={raw!r} is not a number — treating the cap as 0 "
              f"(refuse everything). Fix the value.")
        return 0.0


def enabled():
    return _flag("HL_TRADING_ENABLED")


def acknowledged():
    # Compared case-sensitively after stripping surrounding whitespace and
    # quotes. Shells vary in whether they keep the quotes; the operator should
    # not have to care which one they used.
    got = (os.getenv("HL_LEGAL_ACK") or "").strip().strip('"').strip("'")
    return got == ACK_PHRASE


def dry_run():
    """True if orders must not leave the machine.

    🚨 HL_DRY_RUN CANNOT BE *LESS* SAFE THAN THE GLOBAL DRY_RUN.
    If the global flag says the bot sends nothing, a Hyperliquid-specific
    override must not quietly re-arm a leveraged venue. HL_DRY_RUN=false is
    honoured only when config.DRY_RUN is already false. It can always be used
    to make things safer.
    """
    try:
        from config import DRY_RUN as GLOBAL_DRY
    except Exception:
        GLOBAL_DRY = True                      # no config -> assume the safe end
    local = os.getenv("HL_DRY_RUN")
    if local is None or not local.strip():
        return bool(GLOBAL_DRY)
    local_dry = local.strip().lower() not in ("false", "0", "no", "off")
    return bool(GLOBAL_DRY) or local_dry


def max_notional():
    return _num("HL_MAX_NOTIONAL_USD", 0.0)


def max_leverage():
    return max(1.0, _num("HL_MAX_LEVERAGE", 1.0))


def allowed_coins():
    """Set of permitted coin names, upper-cased. Empty set == allow nothing.

    Hyperliquid names builder-dex assets as `dex:COIN` (e.g. `xyz:SP500`), so
    the allow-list is matched against BOTH the full name and the part after the
    colon -- otherwise listing SP500 would silently fail to match xyz:SP500 and
    read as "the allow-list is broken" rather than "you spelled it differently".
    """
    raw = (os.getenv("HL_ALLOWED_COINS") or "").replace(";", ",")
    return {c.strip().upper() for c in raw.split(",") if c.strip()}


def coin_allowed(coin):
    allow = allowed_coins()
    if not allow:
        return False
    name = str(coin or "").strip().upper()
    return name in allow or name.split(":")[-1] in allow


# ---------------------------------------------------------------------------
# LEDGER
# ---------------------------------------------------------------------------
def _log(event, **fields):
    """Append one decision to hl_executions.jsonl. Never raises.

    Every verdict is logged, pass AND refuse. A refusal you cannot see later is
    indistinguishable from an order that was never attempted -- which is the
    shape of most of the bugs this project has found.
    """
    row = {"ts": time.time(),
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "event": event}
    row.update(fields)
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LEDGER)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except OSError:
        pass
    return row


# ---------------------------------------------------------------------------
# THE BANNER
# ---------------------------------------------------------------------------
_banner_shown = False


def banner(force=False):
    """Print the arm state once per process. Full notice whenever NOT armed."""
    global _banner_shown
    if _banner_shown and not force:
        return
    _banner_shown = True

    en, ack, dry = enabled(), acknowledged(), dry_run()
    if en and ack and not dry:
        print("\n" + "=" * 78)
        print("  🔴 HYPERLIQUID PERP EXECUTION IS LIVE")
        print(f"     coins      : {sorted(allowed_coins()) or 'NONE'}")
        print(f"     max order  : ${max_notional():,.2f} notional")
        print(f"     max leverage: {max_leverage():g}x")
        print("     Real leveraged orders will leave this machine.")
        print("=" * 78 + "\n")
        _log("armed", coins=sorted(allowed_coins()),
             max_notional=max_notional(), max_leverage=max_leverage())
        return

    reasons = []
    if not en:
        reasons.append("HL_TRADING_ENABLED is not set")
    if not ack:
        reasons.append("HL_LEGAL_ACK is absent or does not match")
    if dry:
        reasons.append("DRY_RUN is in force")
    print("\n  🔒 Hyperliquid perp execution is OFF — " + "; ".join(reasons))
    if not ack:
        print(LEGAL_NOTICE)
        print(f"  To enable, after reading the above:\n"
              f'    HL_LEGAL_ACK="{ACK_PHRASE}"\n'
              f"    HL_TRADING_ENABLED=true\n"
              f"    HL_ALLOWED_COINS=<coins>   HL_MAX_NOTIONAL_USD=<dollars>\n")


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------
class Refused(Exception):
    """Raised by check() when an order must not be sent.

    🚨 AN EXCEPTION, NOT A FALSY RETURN. Every execution method in
    hyper_exposure_client already returns None on failure and several callers do
    not inspect the result at all, so a refusal that looks like `None` would be
    indistinguishable from "the exchange was down" -- and in the one place it
    matters most, would read as "the order went out." A raise cannot be ignored
    by accident.
    """


def check(action, coin, notional_usd=0.0, leverage=None, reduce_only=False,
          has_exchange=True):
    """Decide whether one Hyperliquid action may proceed. Raise Refused if not.

    `notional_usd` is size * price, i.e. position value, NOT margin posted.
    Sizing a cap on margin would make the cap mean ten different things at ten
    different leverages.
    """
    ctx = {"action": action, "coin": coin,
           "notional": round(float(notional_usd or 0.0), 2),
           "reduce_only": bool(reduce_only)}

    if not has_exchange:
        _log("refused", reason="no agent secret — client is read-only", **ctx)
        raise Refused("Hyperliquid client has no agent secret; it is read-only. "
                      "Set AGENT_SECRET to execute.")

    # --- risk-reducing actions pass on nothing else. See the docstring. ---
    if reduce_only:
        _log("allowed", reason="risk-reducing — never gated", **ctx)
        return True

    banner()

    if not enabled():
        _log("refused", reason="HL_TRADING_ENABLED not set", **ctx)
        raise Refused(
            "Hyperliquid perp trading is disabled. It is off by default and "
            "must be turned on deliberately: see hl_gate.LEGAL_NOTICE, then set "
            "HL_TRADING_ENABLED=true and HL_LEGAL_ACK.")

    if not acknowledged():
        _log("refused", reason="HL_LEGAL_ACK missing or mismatched", **ctx)
        raise Refused(
            "HL_TRADING_ENABLED is set but the legal acknowledgement is not. "
            "Read hl_gate.LEGAL_NOTICE in full, then set:\n"
            f'    HL_LEGAL_ACK="{ACK_PHRASE}"')

    if dry_run():
        _log("blocked_dry", reason="DRY_RUN in force", **ctx)
        raise Refused(
            "DRY_RUN is in force, so no live order will be sent. Set DRY_RUN="
            "false (and HL_DRY_RUN=false) only when you mean it.")

    if not coin_allowed(coin):
        _log("refused", reason="coin not in HL_ALLOWED_COINS", **ctx,
             allow=sorted(allowed_coins()))
        raise Refused(
            f"{coin} is not in HL_ALLOWED_COINS "
            f"({sorted(allowed_coins()) or 'empty — nothing is permitted'}). "
            f"An empty allow-list refuses everything, deliberately.")

    cap = max_notional()
    nominal = abs(float(notional_usd or 0.0))
    if cap <= 0:
        _log("refused", reason="HL_MAX_NOTIONAL_USD is 0/unset", **ctx)
        raise Refused(
            "HL_MAX_NOTIONAL_USD is 0 or unset, which refuses every order. An "
            "unset cap is not an infinite cap — say what you want in dollars.")
    if nominal > cap:
        _log("refused", reason="over notional cap", cap=cap, **ctx)
        raise Refused(f"order notional ${nominal:,.2f} exceeds "
                      f"HL_MAX_NOTIONAL_USD ${cap:,.2f}")
    if nominal <= 0:
        # A zero-notional order means the price lookup returned 0 or the size
        # rounded away. Either way the cap could not be applied, so it did not
        # protect anything -- do not let it through on a technicality.
        _log("refused", reason="notional is 0 — cap could not be applied", **ctx)
        raise Refused(
            f"cannot price {coin} (notional came out 0), so the "
            f"HL_MAX_NOTIONAL_USD cap cannot be applied. Refusing.")

    if leverage is not None:
        lev_cap = max_leverage()
        if float(leverage) > lev_cap:
            _log("refused", reason="over leverage cap", cap=lev_cap,
                 leverage=leverage, **ctx)
            raise Refused(f"{leverage}x exceeds HL_MAX_LEVERAGE {lev_cap:g}x")

    _log("allowed", **ctx)
    return True


if __name__ == "__main__":
    print(f"\n  HL_TRADING_ENABLED : {enabled()}")
    print(f"  HL_LEGAL_ACK       : {'matches' if acknowledged() else 'ABSENT/MISMATCH'}")
    print(f"  effective dry_run  : {dry_run()}")
    print(f"  HL_ALLOWED_COINS   : {sorted(allowed_coins()) or 'empty (refuses all)'}")
    print(f"  HL_MAX_NOTIONAL_USD: ${max_notional():,.2f}")
    print(f"  HL_MAX_LEVERAGE    : {max_leverage():g}x")
    banner(force=True)
    try:
        check("selftest", "BTC", notional_usd=100.0)
        print("  verdict: a $100 BTC entry WOULD be sent.\n")
    except Refused as e:
        print(f"  verdict: refused — {e}\n")
