"""Sending a WhatsApp message, and proving that an inbound one really came from Meta.

The seam is a `Protocol` and the implementation is a plain class that satisfies it
structurally, which is the pattern `records/lms.py` established for Moodle: the service
that decides *what* to say never imports an HTTP client, so the rule "a code goes only to
the number that asked for it" is testable with a list.

**Why this costs nothing to run.** A parent messages us first, which opens what Meta calls
a customer service window; inside it a business may send free-form messages, and Meta's
pricing page states that service conversations are free (since 1 November 2024) and that
under per-message pricing (since 1 July 2025) "All non-template messages are free". We
never send a template, so we are never billed. That is a policy rather than a contract,
so `WhatsAppUnavailable` is a normal outcome the caller must handle rather than a crash —
the day it becomes chargeable, or the day the window semantics change, this seam is the
only place that has to know.

**The window is the one hard deadline.** It lasts 24 hours from the parent's message and
our reply is sent seconds later, so it is never close — but a reply attempted after it
closes is *rejected*, not queued, and the only re-engagement path is a paid template. A
delayed send is therefore a failed send, which is why nothing here retries in the
background.

Nothing in this module reads the clock, a database, or the environment. Credentials arrive
as constructor arguments, and `app.py` is the only place that knows an environment exists.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import threading
from typing import Final, Protocol
from urllib.parse import quote

logger = logging.getLogger(__name__)

#: Meta's Graph API version. Pinned rather than floating: an unpinned version silently
#: changes payload shapes under a service that is not being deployed, and the failure
#: appears as a parse error in a webhook nobody was touching that week.
GRAPH_VERSION: Final[str] = "v26.0"

_GRAPH_HOST: Final[str] = "https://graph.facebook.com"


class WhatsAppUnavailable(RuntimeError):
    """The message could not be sent, for any reason at all.

    One exception for transport failure, timeout, a refused token, a closed customer
    service window and an unexpected status alike — the `LmsUnavailable` rule. A caller
    that could tell "Meta is down" from "our token is wrong" is a caller that can probe
    this service's configuration from the outside, and neither answer changes what it does
    next: the parent is told the code could not be sent, and nothing is bound.
    """


class WhatsAppGateway(Protocol):
    """Sending one text message to one WhatsApp user.

    Deliberately the whole interface. This service sends exactly one kind of message —
    a short verification code, inside a window the parent opened — so a seam offering
    templates, media or read receipts would be surface nobody implements correctly and
    nobody tests.
    """

    def send_text(self, to_wa_id: str, body: str) -> None:
        """Send `body` to `to_wa_id`. Raises `WhatsAppUnavailable` on any failure.

        `to_wa_id` is WhatsApp's own form of the number: E.164 digits with **no** leading
        `+`, exactly as it arrives in a webhook payload. Passing a `+`-prefixed number is
        the single most likely integration mistake, so implementations normalise rather
        than reject — a code that fails to reach a parent because of a punctuation
        difference is a support call nobody can diagnose from the outside.
        """


def wa_id_of(phone: str) -> str:
    """WhatsApp's form of a number: digits only, no `+`, no spaces.

    The inverse of what the rest of the estate stores. `sis/` keeps E.164 with the plus
    because that is the unambiguous written form; Meta's `wa_id` drops it. Converting in
    one named place keeps the difference from being rediscovered at each call site.
    """
    return "".join(character for character in str(phone) if character.isdigit())


def e164_or_raise(number: str, *, setting: str) -> str:
    """Accept a number only in explicit international form, and say so loudly otherwise.

    This exists because of a failure with no symptom. Meta requires the school's number in
    full international form with no leading zero, and `wa.me` does not validate: configure
    the Egyptian national spelling `01288339613` and every parent gets a link to
    `wa.me/01288339613`, which is a different number that does not exist. The link opens,
    WhatsApp shows an empty chat, no message ever reaches the webhook, and nothing anywhere
    logs an error. Refusing at startup turns a silent estate-wide outage into a deploy that
    does not come up.

    National-form input is refused rather than converted, because converting requires a
    default country this service has no business holding an opinion about — `sis/` owns
    that rule, and two places deciding it is how the two disagree.
    """
    cleaned = str(number).strip()
    digits = wa_id_of(cleaned)
    if not cleaned.startswith("+") or not digits or len(digits) < 8:
        raise RuntimeError(
            f"{setting} must be a phone number in international form beginning with '+', "
            f"for example '+201288339613'. Got {number!r}. WhatsApp reads a national "
            f"spelling such as '01288339613' as an entirely different number, and the "
            f"resulting links fail silently."
        )
    return "+" + digits


def click_to_chat_link(business_number: str, message: str) -> str:
    """The `wa.me` link that opens WhatsApp with `message` already typed.

    **It is never sent automatically.** WhatsApp deliberately requires the parent to tap
    send, and any wording that implies otherwise produces a queue of parents who clicked
    and then waited. The screen that shows this link has to say so.

    `business_number` must already be E.164 — see `e164_or_raise`, which the composition
    root applies once at startup so this cannot be reached with a national spelling.

    `quote` with `safe=""` rather than `urlencode`, because every reserved character has to
    be escaped: an unescaped `&` or `#` truncates the message at exactly the point where
    the nonce would have been, and the resulting message arrives looking almost right.
    """
    return f"https://wa.me/{wa_id_of(business_number)}?text={quote(message, safe='')}"


def signature_is_valid(*, raw_body: bytes, header: str | None, app_secret: str) -> bool:
    """Did Meta sign these exact bytes?

    `raw_body` must be the bytes as received. Parsing the JSON and re-serialising it
    produces a different byte string — Meta escapes non-ASCII, so an Arabic parent's
    profile name is enough to break a signature computed over re-encoded JSON, and it
    breaks for *some* parents only, which is the worst way to find out.

    Returns `False` rather than raising on a missing or malformed header, so the caller has
    one branch: unsigned and wrongly-signed are the same answer.
    """
    if not header or not app_secret:
        return False
    prefix = "sha256="
    if not header.startswith(prefix):
        return False
    expected = hmac.new(
        app_secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header[len(prefix) :].strip())


class CloudApiWhatsAppGateway:
    """`WhatsAppGateway` over Meta's Cloud API.

    One POST, no retries. A verification code is worth sending once: by the time a retry
    would land the parent has usually asked for another, and two codes in a thread is a
    support call. A failure raises and the browser is told to try again — which re-opens
    the window cleanly rather than racing it.

    The access token must be a **System User** token. The one offered in the App Dashboard
    is a user token that expires in under 24 hours, which makes a pilot appear to work all
    day and die overnight with a `190`.
    """

    #: Meta answers in well under a second; a code the parent is waiting for is worthless
    #: late, so the timeout is short and a slow send is treated as a failed one.
    DEFAULT_TIMEOUT_SECONDS: Final[float] = 8.0

    def __init__(
        self,
        *,
        phone_number_id: str,
        access_token: str,
        timeout_seconds: float | None = None,
        graph_version: str = GRAPH_VERSION,
    ) -> None:
        if not phone_number_id or not access_token:
            raise RuntimeError(
                "The WhatsApp gateway needs IDENTITY_WHATSAPP_PHONE_NUMBER_ID and "
                "IDENTITY_WHATSAPP_TOKEN."
            )
        self._phone_number_id = phone_number_id
        self._access_token = access_token
        self._timeout = timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS
        self._graph_version = graph_version
        self._client = None
        self._lock = threading.Lock()

    def _http(self):
        """One pooled client per process, built on first use.

        Built lazily so that importing this module — which the tests do — never opens a
        socket, and under a lock so two simultaneous first requests do not build two.
        """
        import httpx

        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=httpx.Timeout(self._timeout),
                    transport=httpx.HTTPTransport(retries=0),
                    follow_redirects=False,
                )
            return self._client

    def send_text(self, to_wa_id: str, body: str) -> None:
        import httpx

        url = (
            f"{_GRAPH_HOST}/{self._graph_version}/"
            f"{quote(self._phone_number_id, safe='')}/messages"
        )
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": wa_id_of(to_wa_id),
            "type": "text",
            # A verification code has no link in it, but `preview_url` defaults to letting
            # WhatsApp render one if a message ever does — and a link preview in a security
            # code is exactly the shape a phishing message takes.
            "text": {"preview_url": False, "body": body},
        }
        try:
            response = self._http().post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._access_token}"},
            )
        except httpx.HTTPError as error:
            raise WhatsAppUnavailable(f"WhatsApp could not be reached: {error}") from error

        if response.status_code >= 400:
            # The status and Meta's own error code, never the token and never the full
            # URL — the phone number id is in the path and the token is in a header, and
            # both end up in an aggregator once they reach a log line.
            logger.warning(
                "WhatsApp refused a send: status=%s code=%s",
                response.status_code,
                _error_code(response),
            )
            raise WhatsAppUnavailable(
                f"WhatsApp refused the message with status {response.status_code}."
            )


def _error_code(response: object) -> str:
    """Meta's machine-readable error code, or `""` when the body is not what we expect.

    Separate and defensive because this runs on the failure path: an error handler that
    itself raises turns a diagnosable refusal into a 500 with no explanation.
    """
    try:
        body = response.json()  # type: ignore[attr-defined]
        error = body.get("error") if isinstance(body, dict) else None
        return str(error.get("code", "")) if isinstance(error, dict) else ""
    except Exception:  # noqa: BLE001 - never fail while reporting a failure
        return ""


class RecordingWhatsAppGateway:
    """The in-repo fake: keeps every message instead of sending it.

    The default when no credentials are configured, so the whole flow is developable and
    testable with no Meta account at all — and so a forgotten configuration in production
    fails as "no code arrived" with the code sitting in a log, rather than as a stack trace
    on a parent's screen.

    `log_bodies` is off by default because the body *is* the verification code. Turning it
    on is how a developer reads the code during local work, and it is why the warning below
    says what it says.
    """

    def __init__(self, *, log_bodies: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self._log_bodies = log_bodies

    def send_text(self, to_wa_id: str, body: str) -> None:
        self.sent.append((wa_id_of(to_wa_id), body))
        if self._log_bodies:
            logger.warning(
                "WhatsApp is not configured; the verification code for %s is in this log "
                "line: %s. This must never be enabled where real parents can be verified.",
                wa_id_of(to_wa_id),
                body,
            )
        else:
            logger.warning(
                "WhatsApp is not configured; a verification code for %s was discarded "
                "rather than sent.",
                wa_id_of(to_wa_id),
            )


# The selected gateway, chosen once at startup by `app.py` and read per request. A
# module-level slot rather than a FastAPI dependency, matching `records/lms.py`: the choice
# is a property of the process, not of a request, and a dependency would let a test
# override it for one route and leave the webhook talking to Meta.
_gateway: WhatsAppGateway = RecordingWhatsAppGateway()


def set_gateway(gateway: WhatsAppGateway) -> None:
    global _gateway
    _gateway = gateway


def get_gateway() -> WhatsAppGateway:
    return _gateway


def inbound_text_messages(raw_body: bytes) -> list[tuple[str, str, str]]:
    """Every text message in one webhook delivery, as `(wa_id, text, message_id)`.

    Defensive at every level, because this parses a payload from outside the estate and
    runs on a path that must never raise: Meta replays a delivery it does not see
    acknowledged for up to seven days, so one unhandled shape becomes a week of retries.
    A payload that does not look the way the documentation says is simply an empty list.

    Non-text messages are skipped rather than reported. Parents send stickers, voice notes
    and photographs to any number they can see, and none of those carry a nonce.

    One delivery can legitimately carry several messages — the shape is
    `entry[] -> changes[] -> value.messages[]` — which is why this returns a list rather
    than the first message it finds.
    """
    import json

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable body is simply no messages
        return []
    if not isinstance(payload, dict):
        return []

    found: list[tuple[str, str, str]] = []
    for entry in _as_list(payload.get("entry")):
        for change in _as_list(_get(entry, "changes")):
            value = _get(change, "value")
            if not isinstance(value, dict):
                continue
            for message in _as_list(value.get("messages")):
                if not isinstance(message, dict) or message.get("type") != "text":
                    continue
                text = message.get("text")
                body = text.get("body", "") if isinstance(text, dict) else ""
                # `from` is the sender's wa_id. It comes from Meta, never from anything a
                # browser told us, which is the entire basis for trusting the number.
                sender = str(message.get("from") or "")
                if sender:
                    found.append((sender, str(body or ""), str(message.get("id") or "")))
    return found


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


def _get(container: object, key: str) -> object:
    return container.get(key) if isinstance(container, dict) else None


# ---------------------------------------------------------------------------
# Process-wide configuration
# ---------------------------------------------------------------------------
#
# Set once by `app.py` at startup and read per request, the same shape as the gateway slot
# above. These are credentials and a phone number rather than behaviour, so they are values
# rather than a Protocol — but they follow the same rule: nothing outside the composition
# root reads the environment.

_verify_token: str = ""
_app_secret: str = ""
_business_number: str = ""


def configure(*, verify_token: str, app_secret: str, business_number: str) -> None:
    """Install the webhook secrets and the school's own number.

    `business_number` must already have been through `e164_or_raise`; this does not
    re-check it, because the composition root is the place where a bad value should stop a
    deploy rather than a request.
    """
    global _verify_token, _app_secret, _business_number
    _verify_token = verify_token
    _app_secret = app_secret
    _business_number = business_number


def get_verify_token() -> str:
    """The string Meta echoes back during the subscription handshake."""
    return _verify_token


def get_app_secret() -> str:
    """The App Secret every inbound payload is signed with.

    Empty means unconfigured, and `signature_is_valid` returns `False` for every message in
    that state — so a webhook that was never given a secret rejects everything rather than
    accepting anything. Fail closed: an open webhook is an endpoint through which anyone
    can claim to be any parent's phone.
    """
    return _app_secret


def get_business_number() -> str:
    """The school's WhatsApp number in E.164, as the link and the UI both show it."""
    return _business_number
