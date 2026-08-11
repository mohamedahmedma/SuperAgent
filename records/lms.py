"""The swap seam.

Everything LMS-shaped is behind `LmsAdapter`. Routes never import a Moodle symbol,
never see a web-service function name, never handle a Moodle error type. That is what
makes "replace Moodle later" a real option rather than an intention: the blast radius
of the swap is this one file.

THE PROTOCOL CHANGED, AND WHY.
Its first shape was per-course lists of assignments, which the facade then aggregated
itself — because that was the only shape core Moodle's web services could produce.
Measuring against a live instance showed that shape was unusable: the exclusion flag is
not exposed at all, so an excused assignment is indistinguishable from a counted one,
and re-deriving a percentage from the numbers that ARE exposed gives 50% or 30% for a
child genuinely on 90%.

So `local_schoolapi` was written to answer per student per term, returning a figure
Moodle itself computed. This protocol now matches that: one call, already correct.
`records.grading` no longer aggregates anything for the Moodle path — the arithmetic
it used to do is arithmetic nobody should be doing outside the gradebook.

Failures are normalised to one exception on purpose. `LmsUnavailable` is what makes the
honest-failure path possible end to end: the route turns it into a 503 with
`code: "lms_unavailable"`, and the agent turns that into "I cannot reach the school
records right now" — never into a plausible-sounding grade.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LmsUnavailable(RuntimeError):
    """The system of record could not be reached or did not answer usefully.

    Deliberately does not distinguish "down", "timed out" and "returned nonsense".
    All three mean the same thing to the parent, and collapsing them stops a caller
    from probing the LMS's health through this service.
    """


# ---------------------------------------------------------------------------
# What an adapter returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectGrade:
    """One subject's result, as the system of record already computed it.

    Two percentages, because a school that grades attendance produces two legitimate
    answers to "how is she doing in maths" and they are not interchangeable:

    `percentage` is the official course total, attendance and all. `academic_percentage`
    is the assessments alone. A parent almost always means the second; the school's
    record is the first. Neither is a rounding of the other — measured on a real
    instance, 65% against 80% for the same child.

    `academic_percentage` is None when it could not be derived EXACTLY — the course uses
    a weighted or drop-lowest scheme that cannot be reconstructed from points.
    `academic_unavailable` says which. It is never an approximation, because a consumer
    that ignores a caveat flag puts the approximation in front of a parent.
    """

    course_ref: str
    subject_name: str
    percentage: float | None
    academic_percentage: float | None = None
    academic_unavailable: str = ""
    graded_count: int = 0
    excluded_count: int = 0
    pending_count: int = 0
    is_complete: bool = False
    # Moodle's own gradebook-category subtotals. Exact under every aggregation scheme,
    # so this is the reliable route to a partial subject grade when the derived
    # `academic_percentage` is unavailable.
    categories: tuple[dict, ...] = ()


@dataclass(frozen=True)
class SubjectAttendance:
    """One subject's attendance, as the system of record computed it.

    `percentage` is points-weighted, not a day count: the school's own status values
    decide what an excused absence or a late arrival costs. A student marked Excused
    throughout reads 50% on a default status set, where counting present days would say
    0% and accuse them of missing school they were excused from.

    None when no register has been taken — which is not zero. A child cannot be absent
    from a class nobody recorded.
    """

    course_ref: str
    subject_name: str
    percentage: float | None
    taken_sessions: int = 0
    by_status: tuple[dict, ...] = ()
    # Carried so a term-level figure can be aggregated CORRECTLY across subjects.
    # Averaging the per-subject percentages would weight a subject with two registers
    # the same as one with forty; summing points and maxima is what the LMS itself does
    # across activities.
    points: float = 0.0
    max_points: float = 0.0


class LmsAdapter(Protocol):
    """What the facade needs from a system of record. Nothing more.

    Both calls take the SCHOOL's student reference — the number on a letter home — not
    an internal LMS id. That keeps the contract free of Moodle and lets the facade key
    everything on the identifier a registrar can actually look up.

    The adapter is not an authorisation boundary and must never be asked to be one. The
    facade has already decided this guardian may see this student before either method
    is called.
    """

    def get_subject_grades(self, *, student_ref: str, term: str) -> list[SubjectGrade]:
        """Every subject's result for one student in one term."""
        ...

    def get_subject_attendance(self, *, student_ref: str, term: str) -> list[SubjectAttendance]:
        """Every subject's attendance for one student in one term."""
        ...


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeLms:
    """Deterministic fixtures, so the service and its tests need no Moodle.

    Also the reference for what a correct adapter returns — particularly a subject
    where `percentage` and `academic_percentage` differ, which is the case a real
    adapter is most likely to flatten into one number.
    """

    grades: dict[tuple[str, str], list[SubjectGrade]] = field(default_factory=dict)
    attendance: dict[tuple[str, str], list[SubjectAttendance]] = field(default_factory=dict)
    # Set to raise instead of answering, so the honest-failure path can be tested
    # without taking a real service down.
    unavailable: bool = False

    def get_subject_grades(self, *, student_ref: str, term: str) -> list[SubjectGrade]:
        if self.unavailable:
            raise LmsUnavailable("FakeLms configured as unavailable")
        return list(self.grades.get((student_ref, term), []))

    def get_subject_attendance(self, *, student_ref: str, term: str) -> list[SubjectAttendance]:
        if self.unavailable:
            raise LmsUnavailable("FakeLms configured as unavailable")
        return list(self.attendance.get((student_ref, term), []))


# ---------------------------------------------------------------------------
# Moodle
# ---------------------------------------------------------------------------


class MoodleAdapter:
    """Talks to `local_schoolapi` on a Moodle instance.

    Deliberately thin. Every figure it returns was computed by Moodle or by the plugin
    reading Moodle's own aggregates; this class transports and reshapes, and does no
    arithmetic whatsoever. The moment it starts calculating a percentage is the moment
    it can disagree with the gradebook a teacher is looking at.

    Two Moodle behaviours it has to know about, both of which look like bugs elsewhere:

    Moodle signals FAILURE WITH HTTP 200 and an `exception` key in the body. A client
    that trusts the status code treats every refusal as success and returns an empty
    result — which the assistant would report to a parent as "no grades recorded".

    Moodle REDIRECTS any request whose host is not `$CFG->wwwroot`, answering with an
    HTML page instead of JSON. Calling 127.0.0.1 when wwwroot says localhost produces
    something that parses as neither, and reads exactly like a broken endpoint.
    """

    #: How long a student's payload stays cached. Short: the plugin already caches
    #: server-side and invalidates on the grade event, so this exists only to stop one
    #: chat turn — list children, then grades, then attendance — making the same call
    #: three times. Long enough to help, short enough that a corrected mark is not stale
    #: for a parent already on the phone to the school.
    CACHE_TTL_SECONDS = 30

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout_seconds: float | None = None,
        cache_ttl_seconds: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        # A hung call must not hang a chat turn. The parent is waiting on a streamed
        # answer, so failing at 15s with "I can't reach the records" beats succeeding
        # at 90s.
        self.timeout_seconds = timeout_seconds or float(os.getenv("MOODLE_TIMEOUT_SECONDS") or 15)
        self.cache_ttl_seconds = (
            self.CACHE_TTL_SECONDS if cache_ttl_seconds is None else cache_ttl_seconds
        )

        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str], tuple[float, Any]] = {}

    # -- transport ----------------------------------------------------------

    def _call(self, function: str, **params) -> Any:
        """One web service call. Every failure becomes LmsUnavailable.

        Including a refusal: a token whose capability was revoked, a function removed
        from the external service, a student the plugin declined to resolve. From the
        parent's side these are indistinguishable from the server being down, and
        distinguishing them in the response would let a caller probe Moodle's
        configuration through this service.
        """
        import requests

        payload = {
            "wstoken": self.token,
            "wsfunction": function,
            "moodlewsrestformat": "json",
            **params,
        }

        try:
            response = requests.post(
                f"{self.base_url}/webservice/rest/server.php",
                data=payload,
                timeout=self.timeout_seconds,
                # Do NOT follow redirects. Moodle bounces a wrong host to its wwwroot
                # and answers with HTML; following it turns a configuration error into
                # a confusing parse failure instead of a clear one.
                allow_redirects=False,
            )
        except Exception as exc:
            raise LmsUnavailable(f"{function}: transport failure — {exc}") from exc

        if response.status_code in (301, 302, 303, 307, 308):
            raise LmsUnavailable(
                f"{function}: Moodle redirected to {response.headers.get('location')!r}. "
                "MOODLE_BASE_URL must match the site's configured wwwroot exactly."
            )

        if response.status_code != 200:
            raise LmsUnavailable(f"{function}: HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise LmsUnavailable(f"{function}: response was not JSON") from exc

        # The one that catches people out: HTTP 200, and an error in the body.
        if isinstance(body, dict) and "exception" in body:
            logger.warning(
                "Moodle refused %s: %s — %s",
                function, body.get("errorcode"), body.get("message"),
            )
            raise LmsUnavailable(f"{function}: {body.get('errorcode')}")

        return body

    def _cached(self, function: str, student_ref: str, term: str) -> Any:
        key = (function, student_ref, term)
        now = time.monotonic()

        with self._lock:
            hit = self._cache.get(key)
            if hit and (now - hit[0]) < self.cache_ttl_seconds:
                return hit[1]

        # Deliberately outside the lock: a slow Moodle would otherwise block every other
        # student's request behind this one. The cost is that two concurrent requests
        # for the same student may both call — which is a wasted call, not a wrong
        # answer, and far cheaper than serialising the whole service.
        payload = self._call(function, studentidnumber=student_ref, term=term)

        with self._lock:
            self._cache[key] = (now, payload)
            # The cache is per-process and per-request-burst, not a store. Trimming
            # keeps a long-lived worker from accumulating every student it ever served.
            if len(self._cache) > 512:
                cutoff = now - self.cache_ttl_seconds
                self._cache = {k: v for k, v in self._cache.items() if v[0] >= cutoff}

        return payload

    # -- the protocol -------------------------------------------------------

    def get_subject_grades(self, *, student_ref: str, term: str) -> list[SubjectGrade]:
        payload = self._cached("local_schoolapi_get_student_grades", student_ref, term)

        # `found: false` means no such active student. That is NOT an error and must not
        # become one: the facade deliberately makes "no such student", "not your child"
        # and "records restricted" indistinguishable, and raising here would leak the
        # difference back to a caller who could then enumerate the school roll.
        if not isinstance(payload, dict) or not payload.get("found"):
            return []

        return [self._to_subject_grade(row) for row in payload.get("subjects") or []]

    def get_subject_attendance(self, *, student_ref: str, term: str) -> list[SubjectAttendance]:
        payload = self._cached("local_schoolapi_get_student_attendance", student_ref, term)

        if not isinstance(payload, dict) or not payload.get("found"):
            return []

        return [self._to_subject_attendance(row) for row in payload.get("subjects") or []]

    # -- reshaping ----------------------------------------------------------

    @staticmethod
    def _as_float(value: Any) -> float | None:
        """None stays None. Zero stays zero.

        The distinction this preserves is the whole point: a null percentage means
        nothing has been graded, a zero means the child scored nothing. Coercing with
        `float(value or 0)` collapses them and tells a parent their child got nothing
        when in truth nothing has been marked.
        """
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _to_subject_grade(cls, row: dict) -> SubjectGrade:
        academic = row.get("academic") or {}
        return SubjectGrade(
            course_ref=str(row.get("idnumber") or row.get("courseid") or ""),
            subject_name=str(row.get("fullname") or row.get("shortname") or ""),
            percentage=cls._as_float(row.get("percentage")),
            academic_percentage=cls._as_float(academic.get("percentage")),
            academic_unavailable=str(academic.get("unavailable") or ""),
            graded_count=int(row.get("gradedcount") or 0),
            excluded_count=int(row.get("excludedcount") or 0),
            pending_count=int(row.get("pendingcount") or 0),
            is_complete=bool(row.get("iscomplete")),
            categories=tuple(row.get("categories") or ()),
        )

    @classmethod
    def _to_subject_attendance(cls, row: dict) -> SubjectAttendance:
        return SubjectAttendance(
            course_ref=str(row.get("idnumber") or row.get("courseid") or ""),
            subject_name=str(row.get("fullname") or row.get("shortname") or ""),
            percentage=cls._as_float(row.get("percentage")),
            taken_sessions=int(row.get("takensessions") or 0),
            by_status=tuple(row.get("bystatus") or ()),
            points=cls._as_float(row.get("points")) or 0.0,
            max_points=cls._as_float(row.get("maxpoints")) or 0.0,
        )


# The process-wide adapter. `app.py` sets it at startup; tests override it. A
# module-level slot rather than a FastAPI dependency because it is genuinely process
# configuration, not per-request state.
_adapter: LmsAdapter | None = None


def set_adapter(adapter: LmsAdapter) -> None:
    global _adapter
    _adapter = adapter


def get_adapter() -> LmsAdapter:
    if _adapter is None:
        raise LmsUnavailable("No LMS adapter configured.")
    return _adapter
