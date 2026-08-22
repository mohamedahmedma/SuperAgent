"""Which children a guardian may be told about — asked, not stored.

This service used to hold that answer in its own `guardians` / `students` /
`guardian_students` tables. It no longer does. The school's own SIS owns who a child's
parents are, because that is the registrar's fact: entered from paperwork, amended by
custody decisions, and audited there. Two copies of it would disagree the first time a
court order was applied to one of them, and the copy that was wrong would be the one
answering a parent.

So `records/` keeps its job and gives up its data. It still decides **whether** a request
may proceed — it verifies the token, insists the signed `guardian_id` matches the one in
the path, writes the audit row, and refuses everything it cannot justify. What it no
longer does is remember who anybody's parents are.

The seam is `records/lms.py`'s, which this service already uses for grades: a `Protocol`
for the question, a plain class per backend, a deterministic fake beside them, and a
module-level slot chosen once at startup. `LmsUnavailable` has a sibling here for the same
reason — a caller must not be able to tell "the directory is down" from "not your child",
because the first is a retry and the second is an answer.

**A restricted guardian is simply not told.** `sis/` filters to links carrying
`can_view_records`, so a parent barred by a court order comes back holding no children at
all rather than holding children this service then has to remember to hide.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Final, Protocol
from urllib.parse import quote

logger = logging.getLogger(__name__)

#: SIS's own machine-readable code for "that names nothing on file". Recognised explicitly
#: rather than by the bare 404, because a misconfigured base URL answers 404 too — and
#: reading that as "not your child" would tell every parent in the school the same thing.
_UNKNOWN_REFERENCE: Final[str] = "unknown_reference"


class GuardianDirectoryUnavailable(RuntimeError):
    """The question could not be put. Distinct from "no", which is an empty result."""


@dataclass(frozen=True, slots=True)
class PermittedStudent:
    """A child this guardian may be told about, as far as the system of record is concerned.

    Deliberately thin, and thinner than the ORM row it replaces. `grade_level` and
    `section` survive because the Moodle course-binding lookup keys on them; SIS does not
    report them on this route and leaves them empty, which is correct for the SIS path
    where grades are keyed on the student number alone.
    """

    student_id: str
    full_name_ar: str = ""
    full_name_en: str = ""
    grade_level: str = ""
    section: str = ""

    @property
    def external_id(self) -> str:
        """The name the previous ORM row used. Kept so call sites did not all have to move."""
        return self.student_id


class GuardianDirectory(Protocol):
    """The two authorization questions this service asks, and nothing else."""

    def children_of(self, guardian_id: str) -> list[PermittedStudent]:
        """Every child this guardian may be told about. Empty when there are none.

        Empty is an ordinary answer: a parent whose only link carries a custody restriction
        has no children *to be told about*, and that is different from not being a parent.
        Both come back empty here on purpose — the distinction is one this service is not
        entitled to reveal, since a caller who could tell them apart could detect a
        restriction from the outside.
        """

    def permits(self, guardian_id: str, student_id: str) -> PermittedStudent | None:
        """That one child, if this guardian may be told about her. `None` otherwise."""


class SisGuardianDirectory:
    """The school's SIS, asked by the opaque handle the parent's token carries.

    Uses SIS's by-handle route rather than its by-phone one, so a parent's phone number
    never reaches this service, its logs, or the chat backend beyond it.

    No cache. A registrar revoking access the minute a court order arrives must take effect
    on the next question, and a cached "yes" would keep answering for as long as the entry
    lived. This is one request per parent question, not per page view.
    """

    DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0

    def __init__(
        self, *, base_url: str, api_key: str = "", timeout_seconds: float | None = None
    ) -> None:
        if not base_url:
            raise RuntimeError("The SIS guardian directory needs SIS_BASE_URL.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS
        self._client = None
        self._lock = threading.Lock()

    def _http(self):
        import httpx

        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout),
                    transport=httpx.HTTPTransport(retries=0),
                    follow_redirects=False,
                )
            return self._client

    def children_of(self, guardian_id: str) -> list[PermittedStudent]:
        import httpx

        if not guardian_id:
            return []
        path = f"/v1/guardians/by-id/{quote(guardian_id, safe='')}/students"
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        try:
            response = self._http().get(path, headers=headers)
        except httpx.HTTPError as error:
            raise GuardianDirectoryUnavailable(
                f"The school's system of record could not be reached: {error}"
            ) from error

        if response.is_redirect:
            raise GuardianDirectoryUnavailable(
                f"The guardian lookup was redirected to "
                f"{response.headers.get('location')!r}. SIS_BASE_URL must name the SIS "
                f"service's own origin."
            )
        if response.status_code == 404 and _error_code(response) == _UNKNOWN_REFERENCE:
            # No such guardian. Indistinguishable, to the caller, from a guardian with no
            # visible children — see `children_of` on the Protocol.
            return []
        if response.status_code >= 400:
            logger.warning(
                "guardian lookup refused: status=%s code=%s",
                response.status_code,
                _error_code(response),
            )
            raise GuardianDirectoryUnavailable(
                f"The guardian lookup failed with status {response.status_code}."
            )

        try:
            rows = (response.json() or {}).get("students") or []
        except Exception as error:  # noqa: BLE001 - any unreadable body is one outcome
            raise GuardianDirectoryUnavailable(
                "The guardian lookup returned a body this service could not read."
            ) from error

        return [
            PermittedStudent(
                student_id=str(row.get("student_number") or ""),
                full_name_ar=str(row.get("full_name_ar") or ""),
                full_name_en=str(row.get("full_name_en") or ""),
            )
            for row in rows
            if row.get("student_number")
        ]

    def permits(self, guardian_id: str, student_id: str) -> PermittedStudent | None:
        """Answered from the list rather than by a second call.

        One request instead of two, and — more importantly — one source of truth for the
        turn: a caller cannot be told "yes, that child" by a lookup that disagrees with the
        list it was given a moment earlier.
        """
        wanted = str(student_id)
        return next(
            (child for child in self.children_of(guardian_id) if child.student_id == wanted),
            None,
        )


def _error_code(response: object) -> str:
    """SIS's error code, or `""`. Defensive: this runs on the failure path."""
    try:
        body = response.json()  # type: ignore[attr-defined]
        detail = body.get("detail") if isinstance(body, dict) else None
        return str(detail.get("code", "")) if isinstance(detail, dict) else ""
    except Exception:  # noqa: BLE001 - never fail while reporting a failure
        return ""


class FakeGuardianDirectory:
    """A directory in a dict, and the default when no SIS is configured.

    Being the default is the safe direction: an unconfigured deployment tells every parent
    they have no children on file, which is a support call. The alternative default — one
    that answered from stale local tables — would be a deployment quietly serving records
    from data nobody is maintaining.
    """

    def __init__(
        self,
        children: dict[str, list[PermittedStudent]] | None = None,
        *,
        unavailable: bool = False,
    ) -> None:
        self.children = dict(children or {})
        self.unavailable = unavailable

    def children_of(self, guardian_id: str) -> list[PermittedStudent]:
        if self.unavailable:
            raise GuardianDirectoryUnavailable("The fake directory is switched off.")
        return list(self.children.get(guardian_id, ()))

    def permits(self, guardian_id: str, student_id: str) -> PermittedStudent | None:
        return next(
            (c for c in self.children_of(guardian_id) if c.student_id == str(student_id)),
            None,
        )


_directory: GuardianDirectory = FakeGuardianDirectory()


def set_directory(directory: GuardianDirectory) -> None:
    global _directory
    _directory = directory


def get_directory() -> GuardianDirectory:
    return _directory
