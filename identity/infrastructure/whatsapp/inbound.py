"""Reading what Meta sends us: the signature, and the messages inside the body.

Both halves run on the webhook path, and both are written to the same rule: **this must
never raise.** Meta replays a delivery it does not see acknowledged for up to seven days,
so one unhandled payload shape becomes a week of retries.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


def signature_is_valid(*, raw_body: bytes, header: str | None, app_secret: str) -> bool:
    """Did Meta sign these exact bytes?

    `raw_body` must be the bytes as received. Parsing the JSON and re-serialising it
    produces a different byte string — Meta escapes non-ASCII, so an Arabic parent's
    profile name is enough to break a signature computed over re-encoded JSON, and it
    breaks for *some* parents only, which is the worst way to find out.

    Returns `False` rather than raising on a missing or malformed header, so the caller has
    one branch: unsigned and wrongly-signed are the same answer.

    **An unset `app_secret` disables the check and returns True.** That is a deliberate
    development affordance, and it is the one place in this file where the safe default
    loses to a usable one: rejecting everything when nothing is configured meant a webhook
    that verified fine, delivered nothing, and logged a 403 nobody connected to a missing
    variable.

    What it costs is real and worth stating once. The signature is the only thing that
    distinguishes Meta from anyone who has found the URL, and this webhook's payload names
    the sender — so an unsigned deployment lets a caller assert "this parent just sent the
    code phrase" for any number they like, and sign in as that parent. Set the secret
    before anyone but you can reach the URL. The webhook router logs a warning on every
    unverified delivery so that the state cannot be forgotten quietly.
    """
    if not app_secret:
        return True
    if not header:
        return False
    prefix = "sha256="
    if not header.startswith(prefix):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[len(prefix) :].strip())


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """One text message a parent sent, and the number of ours they sent it to.

    `phone_number_id` is the field this type exists for. Meta stamps every delivery with
    `value.metadata.phone_number_id`, naming which of our WhatsApp numbers received the
    message — and with one number per school that *is* the school, decided before any
    database is opened. See `identity/domain/schools.py`.

    It deserves the same trust as `wa_id` and for the same reason: both come from Meta,
    inside a body whose signature was checked against the app secret, and neither is ever
    supplied by a browser. It is `""` on a delivery that carries no metadata, which a
    caller must treat as "unknown school" and refuse — never as "the default school".
    """

    wa_id: str
    text: str
    message_id: str
    phone_number_id: str = ""


def inbound_text_messages(raw_body: bytes) -> list[InboundMessage]:
    """Every text message in one webhook delivery.

    Defensive at every level, because this parses a payload from outside the estate and
    runs on a path that must never raise. A payload that does not look the way the
    documentation says is simply an empty list.

    Non-text messages are skipped rather than reported. Parents send stickers, voice notes
    and photographs to any number they can see, and none of those carry a nonce.

    One delivery can legitimately carry several messages — the shape is
    `entry[] -> changes[] -> value.messages[]` — which is why this returns a list rather
    than the first message it finds. `value.metadata` sits alongside `value.messages`, so
    the school is read from the same dict the messages come out of and applies to all of
    them: one delivery is addressed to exactly one of our numbers.
    """
    import json

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable body is simply no messages
        return []
    if not isinstance(payload, dict):
        return []

    found: list[InboundMessage] = []
    for entry in _as_list(payload.get("entry")):
        for change in _as_list(_get(entry, "changes")):
            value = _get(change, "value")
            if not isinstance(value, dict):
                continue
            metadata = _get(value, "metadata")
            phone_number_id = str(
                (metadata.get("phone_number_id") if isinstance(metadata, dict) else "") or ""
            )
            for message in _as_list(value.get("messages")):
                if not isinstance(message, dict) or message.get("type") != "text":
                    continue
                text = message.get("text")
                body = text.get("body", "") if isinstance(text, dict) else ""
                # `from` is the sender's wa_id. It comes from Meta, never from anything a
                # browser told us, which is the entire basis for trusting the number.
                sender = str(message.get("from") or "")
                if sender:
                    found.append(
                        InboundMessage(
                            wa_id=sender,
                            text=str(body or ""),
                            message_id=str(message.get("id") or ""),
                            phone_number_id=phone_number_id,
                        )
                    )
    return found


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


def _get(container: object, key: str) -> object:
    return container.get(key) if isinstance(container, dict) else None


__all__ = ["InboundMessage", "inbound_text_messages", "signature_is_valid"]
