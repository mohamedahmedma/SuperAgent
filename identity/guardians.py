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


@dataclass(frozen=True, slots=True)
class ChildRef:
    """One child, as much of her as belongs in a token and no more.

    A name to greet her by, the year she is in, and whether she is a son or a daughter —
    exactly what is needed to understand a parent who writes "my son" rather than a name.

    Nothing about her RECORD is here: no marks, no attendance, no birth date, no contact
    details. This travels in a bearer token that lives in a browser and rides every
    request into every access log, so what it carries is the minimum that makes the
    feature work, and the reader is expected to fetch anything else it needs.
    """

    student_id: str
    full_name_ar: str = ""
    full_name_en: str = ""
    year_level: str = ""
    gender: str = "unspecified"

    @property
    def display_name(self) -> str:
        return self.full_name_ar or self.full_name_en or self.student_id

    def as_claim(self) -> dict:
        """The compact form that goes into the token. Short keys, because this is paid
        on every request a parent's browser makes."""
        return {
            "id": self.student_id,
            "ar": self.full_name_ar,
            "en": self.full_name_en,
            "yr": self.year_level,
            "g": self.gender,
        }


class GuardianDirectory(Protocol):
    """Resolving a verified phone number to the parent the school has on file."""

    def children_of(
        self, public_id: str, *, school_code: str | None = None
    ) -> list[ChildRef]:
        """Every child this guardian may be told about, by her opaque handle.

        Empty is an ordinary answer — a parent whose only link carries a custody
        restriction has no children *to be told about* — and is not distinguishable here
        from having none, deliberately.

        Raises `GuardianDirectoryUnavailable` when the question could not be put, so a
        caller can tell an outage from an empty family. A token minted during an outage
        simply carries no children; it must never carry an empty list as though that were
        the answer.

        `school_code` selects which school's database answers. Schools are separated
        physically, so a handle only means anything inside the school that issued it:
        asked of another school's database the same handle resolves to nobody, which is
        the isolation working rather than a failure. `None` is a single-school deployment.
        """

    def resolve(
        self, phone_e164: str, *, school_code: str | None = None
    ) -> GuardianRef | None:
        """The guardian reachable on this number, or `None` when it reaches nobody.

        `None` is an ordinary answer, not an error: most numbers in the world are not this
        school's parents, and the flow that calls this has to say so politely rather than
        fail. Raises `GuardianDirectoryUnavailable` when the question could not be put at
        all, which is a different situation and gets a different reply.

        `school_code` selects the database. A number that reaches a parent at one branch
        legitimately reaches nobody at another, and under physical separation that is the
        only answer this service can give: the row is not in the file it is connected to.
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
        """
        import httpx

        headers = self._headers(school_code)
        try:
            response = self._http().get(
                f"/v1/guardians/by-id/{public_id}/students", headers=headers
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
        children: dict[str, list["ChildRef"]] | None = None,
        unavailable: bool = False,
    ) -> None:
        self.guardians = dict(guardians or {})
        self.unavailable = unavailable
        self.asked: list[str] = []
        #: Which school each lookup was scoped to, so a test can assert that the school
        #: reached the directory and not merely that a lookup happened.
        self.asked_schools: list[str | None] = []
        #: `{public_id: [ChildRef, ...]}`. Empty by default, so a test that only cares
        #: about sign-in gets a token with no children rather than having to say so.
        self.children: dict[str, list[ChildRef]] = dict(children or {})

    def resolve(
        self, phone_e164: str, *, school_code: str | None = None
    ) -> GuardianRef | None:
        self.asked.append(phone_e164)
        self.asked_schools.append(school_code)
        if self.unavailable:
            raise GuardianDirectoryUnavailable("The fake directory is switched off.")
        return self.guardians.get(phone_e164)

    def children_of(
        self, public_id: str, *, school_code: str | None = None
    ) -> list[ChildRef]:
        if self.unavailable:
            raise GuardianDirectoryUnavailable("The fake directory is switched off.")
        return list(self.children.get(public_id, ()))


# Chosen once at startup by `app.py`, read per request. A module-level slot rather than a
# FastAPI dependency, matching `records/lms.py`: the choice belongs to the process, and a
# dependency override in one test would leave the webhook talking to the real SIS.
_directory: GuardianDirectory = FakeGuardianDirectory()


def set_directory(directory: GuardianDirectory) -> None:
    global _directory
    _directory = directory


def get_directory() -> GuardianDirectory:
    return _directory
