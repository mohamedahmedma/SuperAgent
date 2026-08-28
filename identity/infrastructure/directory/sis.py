"""`GuardianDirectory` over `sis/`'s HTTP API.

**POST for the resolve, and the method carries meaning.** The number travels in a body
rather than a path so it stays out of `sis/`'s access logs, out of any proxy in between,
and out of anything that records a URL. That is the same reasoning that gave guardians a
`public_id` in the first place, and a GET would undo it on the one request whose whole
purpose is to stop holding numbers.

**No cache.** A guardian link can be revoked by a registrar the minute a court order
arrives, and a cached "yes" would keep letting somebody in for as long as the entry lived.
This is asked once per verification, not once per page view, so the cost of asking every
time is a single request per parent per login.

**No retry.** The parent is waiting, `sis/` is on the same network, and a retry storm
against a service that is already struggling is how a slow dependency becomes an outage.

## Two timeouts, not one

`resolve` is the sign-in itself: the parent cannot proceed without it, so it gets the full
budget. `children_of` only decorates a token with a claim the chat backend fetches for
itself anyway, and it runs *inside* the latency of a parent's sign-in — so it gets a
tighter one. With a single budget, a slow SIS added the full timeout to every parent's
login in exchange for saving the backend one call it makes regardless.
"""
from __future__ import annotations

import logging
import threading
from typing import Final

from identity.domain.errors import GuardianDirectoryUnavailable
from identity.domain.guardians import ChildRef, GuardianRef

logger = logging.getLogger(__name__)

#: `sis/`'s code for "that reference names nothing here", which is an ordinary answer
#: rather than a failure — see `resolve`.
_UNKNOWN_REFERENCE: Final[str] = "unknown_reference"


class SisGuardianDirectory:
    """The real directory, over one pooled HTTP client."""

    DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0
    #: See the module docstring. Deliberately well under `DEFAULT_TIMEOUT_SECONDS`.
    DEFAULT_CHILDREN_TIMEOUT_SECONDS: Final[float] = 1.5

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float | None = None,
        children_timeout_seconds: float | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("The SIS guardian directory needs IDENTITY_SIS_BASE_URL.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS
        self._children_timeout = (
            children_timeout_seconds or self.DEFAULT_CHILDREN_TIMEOUT_SECONDS
        )
        self._client = None
        self._lock = threading.Lock()

    def _headers(self, school_code: str | None) -> dict[str, str]:
        """The headers every call to SIS carries.

        `X-School-Code` is what carries physical separation across the wire: SIS resolves
        it to one school's database and answers from that and nothing else. Sent only when
        there is a school to name, so a single-school SIS — which ignores the header
        entirely — keeps answering exactly as it did before.
        """
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        if school_code:
            headers["X-School-Code"] = school_code
        return headers

    def _http(self):
        """One pooled client per process, built on first use and under a lock.

        The pool is the latency decision here. Per-call clients would mean a TCP and a TLS
        handshake to SIS on every parent's sign-in, which is two round trips before the
        request that matters begins — paid by the parent, on the one call they are waiting
        on.
        """
        import httpx

        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout),
                    transport=httpx.HTTPTransport(retries=0),
                    # A redirect would re-send the number, and the API key, to whatever
                    # host the response names. Treated as a hard failure below.
                    follow_redirects=False,
                )
            return self._client

    def resolve(
        self, phone_e164: str, *, school_code: str | None = None
    ) -> GuardianRef | None:
        import httpx

        headers = self._headers(school_code)
        try:
            response = self._http().post(
                "/v1/guardians/resolve", json={"phone": phone_e164}, headers=headers
            )
        except httpx.HTTPError as error:
            raise GuardianDirectoryUnavailable(
                f"The school's system of record could not be reached: {error}"
            ) from error

        if response.is_redirect:
            raise GuardianDirectoryUnavailable(
                f"The guardian lookup was redirected to "
                f"{response.headers.get('location')!r}. IDENTITY_SIS_BASE_URL must name "
                f"the SIS service's own origin."
            )

        if response.status_code == 404 and _error_code(response) == _UNKNOWN_REFERENCE:
            # The number is well formed and belongs to nobody here. An ordinary answer.
            return None

        if response.status_code >= 400:
            logger.warning(
                "The guardian lookup was refused: status=%s code=%s",
                response.status_code,
                _error_code(response),
            )
            raise GuardianDirectoryUnavailable(
                f"The guardian lookup failed with status {response.status_code}."
            )

        try:
            body = response.json()
            public_id = str(body["public_id"])
        except Exception as error:  # noqa: BLE001 - any unreadable body is one outcome
            raise GuardianDirectoryUnavailable(
                "The guardian lookup returned a body this service could not read."
            ) from error

        if not public_id:
            # A blank handle would bind an account to nothing while looking like success —
            # the one shape of answer that must never be treated as a resolution.
            raise GuardianDirectoryUnavailable(
                "The guardian lookup returned an empty handle."
            )

        return GuardianRef(
            public_id=public_id,
            full_name_ar=str(body.get("full_name_ar") or ""),
            full_name_en=str(body.get("full_name_en") or ""),
            preferred_language=str(body.get("preferred_language") or "ar"),
        )

    def children_of(
        self, public_id: str, *, school_code: str | None = None
    ) -> list[ChildRef]:
        """Ask SIS by the opaque handle, never by the number that found her.

        The same discipline `GuardianRef` states: once the phone number has resolved to a
        handle it has done its job, and nothing downstream — token, account row, audit
        line, or this call — needs to know it again.

        Reads `year_level` and `gender` because they are what let a parent say "my son"
        or ask about "the fees" and be understood. Ignores everything else SIS returns.

        Runs under the shorter budget: this decorates a token, and a parent must not wait
        on it. See the module docstring.
        """
        import httpx

        headers = self._headers(school_code)
        try:
            response = self._http().get(
                f"/v1/guardians/by-id/{public_id}/students",
                headers=headers,
                timeout=httpx.Timeout(self._children_timeout),
            )
        except httpx.HTTPError as error:
            raise GuardianDirectoryUnavailable(
                f"The children lookup could not reach the school: {error}"
            ) from error

        if response.status_code == 404:
            # No such handle. Not an outage, and not this method's business to explain.
            return []
        if response.status_code >= 400:
            raise GuardianDirectoryUnavailable(
                f"The children lookup failed with status {response.status_code}."
            )

        try:
            rows = (response.json() or {}).get("students") or []
        except Exception as error:  # noqa: BLE001 - any unreadable body is one outcome
            raise GuardianDirectoryUnavailable(
                "The children lookup returned a body this service could not read."
            ) from error

        found: list[ChildRef] = []
        for row in rows:
            student_id = str(row.get("student_number") or "").strip()
            if not student_id:
                continue
            found.append(
                ChildRef(
                    student_id=student_id,
                    full_name_ar=str(row.get("full_name_ar") or "").strip(),
                    full_name_en=str(row.get("full_name_en") or "").strip(),
                    year_level=str(row.get("year_level") or "").strip(),
                    gender=str(row.get("gender") or "unspecified").strip() or "unspecified",
                )
            )
        return found

    def close(self) -> None:
        """Release the pooled client. Called from the app's shutdown hook."""
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


def _error_code(response: object) -> str:
    """`sis/`'s error code, or `""` when the body is not the expected envelope.

    Defensive because it runs on the failure path, where raising again would replace a
    diagnosable refusal with a stack trace.
    """
    try:
        body = response.json()  # type: ignore[attr-defined]
        detail = body.get("detail") if isinstance(body, dict) else None
        return str(detail.get("code", "")) if isinstance(detail, dict) else ""
    except Exception:  # noqa: BLE001 - never fail while reporting a failure
        return ""


__all__ = ["SisGuardianDirectory"]
