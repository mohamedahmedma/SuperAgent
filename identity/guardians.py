"""Asking the school's system of record whether a verified number belongs to a parent.

The seam between this service and `sis/`, in the shape `records/sis_adapter.py`
established: a `Protocol` for the question, a plain class that answers it over HTTP, and a
deterministic fake beside them that is the default when nothing is configured.

**Why identity does not hold guardian data itself.** Who a child's parents are is the
registrar's fact, entered from paperwork and amended by custody decisions. Copying it here
would create a second answer to "may this adult see this child", and the two would
disagree the first time a court order was applied to one of them. This service therefore
learns exactly one thing — a stable handle for the adult who owns a number — and forgets
the number afterwards.

**Proving a phone is not the same as being allowed to read anything.** WhatsApp can tell
us that somebody controls a number; only `sis/` knows whether the school has that number
on file, and only `sis/` decides which children it may be told about. The handle returned
here is an identity, never a permission — the token minted from it carries no authority
that `sis/` will not re-check at the moment a record is actually asked for.

Nothing here reads the clock, the environment, or a database.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Final, Protocol

logger = logging.getLogger(__name__)

#: `sis/`'s own machine-readable code for "that names nothing on file". Recognised
#: explicitly rather than by the bare 404, because a 404 is also what a misconfigured base
#: URL returns — and treating "wrong URL" as "not a parent" would tell every family in the
#: school that they are not registered.
_UNKNOWN_REFERENCE: Final[str] = "unknown_reference"


class GuardianDirectoryUnavailable(RuntimeError):
    """The question could not be answered, for any reason at all.

    Transport failure, timeout, refusal, an unreadable body and an unexpected status all
    collapse here — the `LmsUnavailable` rule. Deliberately distinct from "no such
    guardian", which is a `None` return: one means *try again later* and the other means
    *this number is not a parent here*, and a caller that confused them would either tell a
    real parent they are unknown or promise an unknown caller that the school is merely
    busy.
    """


@dataclass(frozen=True, slots=True)
class GuardianRef:
    """A parent, named by a handle rather than by the number that found her.

    Carries no phone number on purpose. Once this is in hand the number has done its job,
    and everything downstream — the token, the account row, the audit line — refers to her
    by `public_id`. A number is PII that changes; a handle is neither.
    """

    public_id: str
    full_name_ar: str = ""
    full_name_en: str = ""
    preferred_language: str = "ar"

    @property
    def display_name(self) -> str:
        """Her name in whichever script the school recorded, Arabic first."""
        return self.full_name_ar or self.full_name_en


class GuardianDirectory(Protocol):
    """Resolving a verified phone number to the parent the school has on file."""

    def resolve(self, phone_e164: str) -> GuardianRef | None:
        """The guardian reachable on this number, or `None` when it reaches nobody.

        `None` is an ordinary answer, not an error: most numbers in the world are not this
        school's parents, and the flow that calls this has to say so politely rather than
        fail. Raises `GuardianDirectoryUnavailable` when the question could not be put at
        all, which is a different situation and gets a different reply.
        """


class SisGuardianDirectory:
    """`GuardianDirectory` over `sis/`'s `POST /v1/guardians/resolve`.

    **POST, and the method carries meaning.** The number travels in a body rather than a
    path so it stays out of `sis/`'s access logs, out of any proxy in between, and out of
    anything that records a URL. That is the same reasoning that gave guardians a
    `public_id` in the first place, and a GET would undo it on the one request whose whole
    purpose is to stop holding numbers.

    No cache. A guardian link can be revoked by a registrar the minute a court order
    arrives, and a cached "yes" would keep letting somebody in for as long as the entry
    lived. This is asked once per verification, not once per page view, so the cost of
    asking every time is a single request per parent per login.

    No retry. The parent is waiting, `sis/` is on the same network, and a retry storm
    against a service that is already struggling is how a slow dependency becomes an
    outage.
    """

    DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float | None = None,
    ) -> None:
        if not base_url:
            raise RuntimeError(
                "The SIS guardian directory needs IDENTITY_SIS_BASE_URL."
            )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS
        self._client = None
        self._lock = threading.Lock()

    def _http(self):
        """One pooled client per process, built on first use and under a lock."""
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

    def resolve(self, phone_e164: str) -> GuardianRef | None:
        import httpx

        headers = {"X-API-Key": self._api_key} if self._api_key else {}
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


class FakeGuardianDirectory:
    """A directory held in a dict. The default when no SIS is configured.

    Being the default matters: it means the whole verification flow runs end to end on a
    laptop with no second service, and it means a production deployment that forgot to set
    `IDENTITY_SIS_BASE_URL` refuses every parent rather than authenticating them against
    nothing.

    `unavailable` exists so a test can assert what happens when the school's records are
    unreachable, which is the branch nobody writes by hand and everybody needs.
    """

    def __init__(
        self,
        guardians: dict[str, GuardianRef] | None = None,
        *,
        unavailable: bool = False,
    ) -> None:
        self.guardians = dict(guardians or {})
        self.unavailable = unavailable
        self.asked: list[str] = []

    def resolve(self, phone_e164: str) -> GuardianRef | None:
        self.asked.append(phone_e164)
        if self.unavailable:
            raise GuardianDirectoryUnavailable("The fake directory is switched off.")
        return self.guardians.get(phone_e164)


# Chosen once at startup by `app.py`, read per request. A module-level slot rather than a
# FastAPI dependency, matching `records/lms.py`: the choice belongs to the process, and a
# dependency override in one test would leave the webhook talking to the real SIS.
_directory: GuardianDirectory = FakeGuardianDirectory()


def set_directory(directory: GuardianDirectory) -> None:
    global _directory
    _directory = directory


def get_directory() -> GuardianDirectory:
    return _directory
