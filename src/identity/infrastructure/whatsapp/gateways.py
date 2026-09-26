"""Sending a message: Meta's Cloud API, and the in-repo fake that stands in for it.

Both satisfy `application/ports/messaging.WhatsAppGateway`. Neither reads the environment
— credentials arrive as constructor arguments from the composition root, which is what
lets a two-school deployment build two of these with different tokens.
"""
from __future__ import annotations

import logging
import threading
from typing import Final
from urllib.parse import quote

from identity.domain.errors import WhatsAppUnavailable
from identity.domain.phone import wa_id_of

logger = logging.getLogger(__name__)

#: Pinned, never "latest" — Meta ships breaking changes between versions and an unpinned
#: client starts failing on a date nobody chose. Overridable because the version a school's
#: app was created against is a fact about that app: the console shows the one it generated
#: its examples for, and following it removes one variable from every "why did this stop
#: working" conversation.
DEFAULT_GRAPH_VERSION: Final[str] = "v22.0"

_GRAPH_HOST: Final[str] = "https://graph.facebook.com"


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
        graph_version: str = DEFAULT_GRAPH_VERSION,
    ) -> None:
        if not phone_number_id or not access_token:
            raise ValueError(
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
        """One pooled client per gateway, built on first use.

        Built lazily so that importing this module — which the tests do — never opens a
        socket, and under a lock so two simultaneous first requests do not build two.

        Pooled rather than per-call, and that is the latency decision in this file: a new
        `httpx.Client` per send means a fresh TCP handshake and a fresh TLS handshake to
        `graph.facebook.com` on every verification code, which is two round trips to Meta
        before the request that matters even starts.
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
            code = _error_code(response)
            logger.warning(
                "WhatsApp refused a send: status=%s code=%s%s",
                response.status_code,
                code,
                _diagnosis(code, response.status_code),
            )
            raise WhatsAppUnavailable(
                f"WhatsApp refused the message with status {response.status_code}."
            )

    def close(self) -> None:
        """Release the pooled client. Called from the app's shutdown hook."""
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


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

    def close(self) -> None:  # noqa: D102 - symmetry with the real gateway
        return None


#: Meta's numeric error codes, and what each one actually means for this deployment.
#: Written out because the raw code is a search away from an answer and a sentence is
#: not — and because the first of these is not a maybe: a token taken from the App
#: Dashboard expires in under 24 hours, so a pilot that worked all afternoon stops
#: overnight with nothing in the logs but `code=190`.
_DIAGNOSES: Final[dict[str, str]] = {
    "190": (
        " — the access token has expired or been revoked. A token copied from the App "
        "Dashboard lasts under 24 hours; generate a System User token with "
        "whatsapp_business_messaging and no expiry, and set IDENTITY_WHATSAPP_TOKEN to it."
    ),
    "131030": (
        " — this recipient is not on the app's allowed list. In development mode Cloud "
        "API only messages numbers added under 'Manage phone number list'."
    ),
    "131047": (
        " — more than 24 hours since the parent's last message, so the service window is "
        "closed and only a template may be sent. Should be impossible here: the parent "
        "messages first and the code goes back within seconds."
    ),
    "131056": " — too many messages to this number too quickly; WhatsApp is pacing us.",
    "100": (
        " — a bad parameter, usually the phone number id. Check "
        "IDENTITY_WHATSAPP_PHONE_NUMBER_ID is the ID beside the number, not the number."
    ),
    "133010": " — the sending number is not registered on Cloud API.",
    "80007": " — the account has hit its rate limit for this window.",
}


def _diagnosis(code: str, status: int) -> str:
    """A sentence a person can act on, appended to the log line.

    Falls back to nothing rather than to a guess: an invented explanation for an
    unfamiliar code is worse than the bare code, which is at least searchable.
    """
    known = _DIAGNOSES.get(str(code))
    if known:
        return known
    if status in (401, 403):
        return " — the token was rejected. Check IDENTITY_WHATSAPP_TOKEN."
    return ""


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


__all__ = [
    "DEFAULT_GRAPH_VERSION",
    "CloudApiWhatsAppGateway",
    "RecordingWhatsAppGateway",
]
