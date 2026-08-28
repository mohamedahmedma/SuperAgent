"""The second implementation of `LmsAdapter` — the school's own SIS on :8300.

Proof that the seam in [lms.py](lms.py) is real: the facade, its routes, its assembler
and the agent's tool layer are untouched by this file existing. `RECORDS_LMS=sis` and
the same contract answers from a different system of record.

`sis/` is a narrower system than Moodle, and the two places that narrowness shows are
the two places this adapter had a decision to make:

**One stated figure per subject, and no arithmetic anywhere.** SIS records the mark a
teacher wrote and nothing else — no assignment ledger, no weighting, no drop-lowest. So
`percentage` and `academic_percentage` carry the same number rather than two: SIS grades
have no attendance component mixed into them, which is the whole reason those two fields
differ on the Moodle path. Reporting the academic figure as "unavailable" when the school
has stated it plainly would put a caveat in front of a parent about a mark that is not
in doubt.

**A mark stated only as "17 out of 20" stays a caveat, never a division.** SIS may state
points without a percentage, and rescaling one into the other is exactly the arithmetic
that put 50% in front of a child on 90% during the Moodle work. `academic_unavailable`
says so instead; see `_to_subject_grade`.

Course bindings still decide what a parent sees, unchanged. Under this adapter a
binding's `lms_idnumber` must be the SIS **subject code** (`MATH`), because that is the
reference SIS reports and `records.assembler` matches on it. An unbound subject is
dropped, silently and deliberately, exactly as it is for Moodle.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from records.env import outbound_pool_size
from records.lms import LmsUnavailable, SubjectAttendance, SubjectGrade

if TYPE_CHECKING:  # pragma: no cover - typing only
    from records.calendar import SchoolCalendar

logger = logging.getLogger(__name__)

#: The SIS error code for "this code names nothing" — an unknown student, or an unknown
#: term. The only 404 this adapter is willing to read as "nothing on file".
_UNKNOWN_REFERENCE = "unknown_reference"


def _int_env(name: str, default: int) -> int:
    """The variable as a positive int, or the documented default on anything unusable.

    A typo'd `SIS_POOL_SIZE=forty` must not take the facade down; it should run with the
    documented default. The same rule `sis.config._int_env` follows, for the same reason.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s is not a number (%r); using %d.", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s must be positive (got %d); using %d.", name, value, default)
        return default
    return value


def _as_float(value: Any) -> float | None:
    """None stays None. Zero stays zero.

    Kept local to this adapter rather than shared, so no adapter reaches into another's
    internals. It exists because `float(value or 0)` is the one-character mistake that
    tells a parent their child scored nothing in a subject nobody has marked.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SisAdapter:
    """Reads one student's term marks from the Student Information Service.

    `GET /v1/guardians/by-id/{public_id}/students/{student_number}/grades?term=` with a
    `reader`-scoped key, and its attendance sibling. Nothing else.

    **The guardian handle travels with the request.** It used to not: this adapter called
    the registrar-scoped `/v1/students/{n}/...` routes, so the facade decided who a parent
    was allowed to see and then asked SIS a question that named no parent. SIS re-checks
    the link on these routes, from the registrar's own data, on this request — which means
    the decision is made twice, independently, and a fully compromised facade reaches one
    family rather than the school.

    The handle is the `guardian_id` claim off the parent's identity token, which is SIS's
    own `public_id`. It is never a phone number: this process, and the chat backend beyond
    it, have no reason to hold one.

    Failures normalise to `LmsUnavailable`, the same as the Moodle path, so the route
    degrades exactly as it already does — a 503 with `code: "lms_unavailable"`, which the
    agent turns into "I cannot reach the school records right now" and never into a
    plausible-sounding grade. No new exception type is introduced, because a route that
    has to learn a second failure vocabulary is a route that will handle one of them
    wrong.

    No response cache. The Moodle adapter needs one because a single chat turn makes
    three calls; this adapter serves one endpoint, and a cache here would buy nothing but
    a window in which a corrected mark is stale for a parent already on the phone.
    """

    #: Connections held open to the SIS.
    #:
    #: **This must not be smaller than the worker pool that calls it.** It was 10, on the
    #: reasoning that "a pool larger than the SIS's own worker count only queues" — which
    #: is true of the SIS end and misses what happens at this one. FastAPI serves sync
    #: endpoints from an anyio threadpool of 40 by default, so 40 requests can be in flight
    #: here while only 10 may hold a connection: the other 30 block inside `httpx` waiting
    #: for one, having already been counted as in-flight. The queue forms on this side of
    #: the wire, where it is invisible to the SIS and shows up only as latency.
    #:
    #: Measured against a stub answering in 20ms, with 40 calling threads:
    #:
    #:     max_connections=10   231 req/s   p50  86ms   p95 453ms
    #:     max_connections=40   715 req/s   p50  47ms   p95  84ms
    #:
    #: Three times the throughput and a fifth of the tail, for a number. Sized to the
    #: caller rather than guessed, and overridable for a SIS that genuinely needs
    #: protecting — see `SIS_POOL_SIZE`.
    POOL_SIZE = 40

    #: Beyond this many *idle* connections, close rather than keep. Keepalive is what makes
    #: the second call in a request skip a TCP and TLS handshake, so it is deliberately the
    #: full pool: a facade that makes two calls per parent question and then drops the
    #: connection pays the handshake again on the next one.
    KEEPALIVE_SIZE = 40

    #: A hung call must not hang a chat turn. Failing at 10s with "I can't reach the
    #: records" beats succeeding at 90s to a parent watching a streamed answer.
    DEFAULT_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float | None = None,
        pool_size: int | None = None,
        calendar: "SchoolCalendar | None" = None,
    ):
        """`calendar` answers when a term runs, and is required for attendance.

        Injected rather than constructed here, and rather than reached for through the
        module-level slot, because this adapter is one of two components that need term
        dates and they must not resolve them separately. A caller that only reads grades
        may leave it `None`; attendance then reports nothing, which is the honest answer
        for a deployment that never wired one.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("SIS_TIMEOUT_SECONDS") or self.DEFAULT_TIMEOUT_SECONDS
        )
        self.pool_size = pool_size or outbound_pool_size()
        # Keepalive matches the pool: a facade that makes two calls per parent
        # question and then drops the connection pays the handshake again next time.
        self.keepalive_size = self.pool_size

        self._lock = threading.Lock()
        self._session: Any = None
        self._calendar = calendar

    # -- transport ----------------------------------------------------------

    def _get_session(self) -> Any:
        """One pooled client for the process, built on first use.

        `httpx`, not `requests`: it is what `records/requirements.txt` already declares,
        and a second HTTP client in a five-package service is a dependency added for
        nothing.

        Retries are pinned to zero. httpx already defaults that way, but stating it is
        the point: a retry budget multiplies the timeout by the attempt count, and three
        attempts at 10s is a chat turn nobody waits for. A SIS that is down should be
        reported as down within one timeout.

        **Read before locking.** The client is built once and then read on every request
        from every worker thread; taking a mutex to re-read an attribute that has not
        changed since startup serialises the one path that most needs not to be. The
        unlocked read is safe because the attribute is only ever assigned a fully-built
        client — a thread either sees `None` and joins the slow path, or sees a client
        that is ready to use. The lock still guards construction, and the second check
        inside it is what stops two threads that both saw `None` building two pools.
        """
        session = self._session
        if session is not None:
            return session

        with self._lock:
            if self._session is None:
                import httpx

                self._session = httpx.Client(
                    base_url=self.base_url,
                    headers={"X-API-Key": self.api_key, "Accept": "application/json"},
                    timeout=httpx.Timeout(self.timeout_seconds),
                    transport=httpx.HTTPTransport(
                        retries=0,
                        limits=httpx.Limits(
                            max_connections=self.pool_size,
                            max_keepalive_connections=self.keepalive_size,
                        ),
                    ),
                    # Never follow a redirect: every request carries `X-API-Key`, so a
                    # 302 to another host is a credential handed to whoever controls it.
                    # A misconfigured base URL must fail, not leak the key.
                    follow_redirects=False,
                )

        return self._session

    def _get(self, path: str, params: dict[str, str]) -> dict | None:
        """One SIS call. Returns the body, or None for "nothing on file".

        Every other outcome — transport failure, a refused key, a body that is not the
        JSON object promised — becomes `LmsUnavailable`. Including refusals: a revoked
        key and a switched-off SIS are indistinguishable from the parent's side, and
        distinguishing them in the response lets a caller probe the SIS's configuration
        through this service.
        """
        session = self._get_session()

        try:
            # Relative to the client's base_url, so a SIS mounted under a path prefix
            # joins correctly instead of having its prefix truncated by concatenation.
            response = session.get(path.lstrip("/"), params=params)
        except Exception as exc:
            raise LmsUnavailable(f"{path}: transport failure — {exc}") from exc

        if response.status_code in (301, 302, 303, 307, 308):
            raise LmsUnavailable(
                f"{path}: SIS redirected to {response.headers.get('location')!r}. "
                "SIS_BASE_URL must name the service's own origin."
            )

        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError as exc:
                raise LmsUnavailable(f"{path}: response was not JSON") from exc
            if not isinstance(body, dict):
                raise LmsUnavailable(f"{path}: response was not a JSON object")
            return body

        code = self._error_code(response)

        # "No such student" is not an error here. `records/` deliberately makes "no such
        # student", "not your child" and "records restricted" indistinguishable, and an
        # exception on this branch hands back the one signal that design removes — the
        # difference an outsider would need to enumerate the school roll.
        #
        # Only SIS's own `unknown_reference` counts. A bare 404 from a wrong path or a
        # proxy answers the same status and means the opposite: a configuration mistake
        # reported as "no grades recorded" for every child in the school.
        if response.status_code == 404 and code == _UNKNOWN_REFERENCE:
            return None

        logger.warning(
            "SIS refused %s: HTTP %s (%s)", path, response.status_code, code or "no code"
        )
        raise LmsUnavailable(f"{path}: HTTP {response.status_code} ({code or 'no code'})")

    @staticmethod
    def _error_code(response: Any) -> str:
        """SIS's machine-readable failure code, or "" if the body did not carry one.

        A body that is not SIS's envelope is itself the finding: something between here
        and the SIS answered instead of it.
        """
        try:
            detail = (response.json() or {}).get("detail")
        except ValueError:
            return ""
        if isinstance(detail, dict):
            return str(detail.get("code") or "")
        return ""

    @staticmethod
    def _guardian_path(guardian_ref: str) -> str:
        """The prefix both reads hang off, guardian-scoped whenever there is a guardian.

        Falls back to the registrar-scoped prefix when `guardian_ref` is empty, which is
        what an internal caller with no parent in hand gets — a reconciliation job, or a
        test. It is not a way for a parent-facing route to opt out: those routes take the
        handle off a verified token and always have one, and `records.auth` has already
        refused the request if they do not.
        """
        if not guardian_ref:
            return "/v1"
        return f"/v1/guardians/by-id/{quote(guardian_ref, safe='')}"

    # -- the protocol -------------------------------------------------------

    def get_subject_grades(
        self, *, student_ref: str, term: str, guardian_ref: str = ""
    ) -> list[SubjectGrade]:
        # Both segments are quoted with no safe characters: a stray "/" in a school's
        # numbering — or in a handle — would otherwise rewrite the path and ask a
        # different question.
        path = f"{self._guardian_path(guardian_ref)}/students/{quote(student_ref, safe='')}/grades"
        payload = self._get(path, {"term": term})

        if payload is None:
            return []

        return [self._to_subject_grade(row) for row in payload.get("grades") or []]

    def get_subject_attendance(
        self, *, student_ref: str, term: str, guardian_ref: str = ""
    ) -> list[SubjectAttendance]:
        """The daily register for a term, as one term-level entry.

        **The shapes genuinely differ, and the mapping is a decision rather than a
        translation.** Moodle records attendance per *subject*: forty registers in maths,
        two in the elective, and the contract aggregates them by points so the elective
        cannot mask a term of absences. SIS records one mark per *day* for the whole
        school day — there is no per-subject register to aggregate.

        So this returns a single entry standing for the term. That is not a subject
        pretending to be one: it is the register SIS actually keeps, reported at the
        granularity it is kept. Every term-level figure the parent-facing contract
        publishes — present, absent, late, excused, sessions, rate — comes out exactly
        right, because summing one entry is summing the whole register.

        `points` and `max_points` carry days-in-the-room over days-recorded, which is the
        same ratio the contract's `attendance_rate` publishes. `recorded` is the only
        denominator SIS will state: it counts days a mark exists for, so an unmarked
        Tuesday is never silently counted as a holiday or as an absence.

        The term is resolved to a date window first, because SIS's register is addressed
        by dates and knows nothing about term codes.
        """
        window = self._term_window(term)
        if window is None:
            # No such term, so no window to ask about. Empty rather than an exception:
            # a term the school has not configured has no register, which is a real
            # answer and not an outage.
            return []
        from_date, to_date = window

        path = (
            f"{self._guardian_path(guardian_ref)}"
            f"/students/{quote(student_ref, safe='')}/attendance"
        )
        payload = self._get(path, {"from": from_date, "to": to_date})
        if payload is None:
            return []

        counts = payload.get("counts") or {}
        recorded = int(counts.get("recorded") or 0)
        if recorded <= 0:
            # Nobody has taken a register for this child this term. Empty, so the
            # contract reports `attendance_rate: null` rather than 0% — a child cannot be
            # absent from classes nobody recorded.
            return []

        # Everything except an unexcused absence. **Not** SIS's `in_the_room`, which is
        # present-plus-late and treats an excused day as missed.
        #
        # The two services genuinely disagree here, and this contract's answer is the one
        # a parent is shown. `AttendanceAssembler.PRESENT_LIKE` counts excused as present,
        # and the template that renders this says so out loud: "Excused absences are
        # authorised by the school and are counted as attended in that percentage."
        # Mapping SIS's narrower figure through would have published 80% where the
        # contract promises 90% — a child with a doctor's note shown to her parent as
        # having missed school.
        attended = max(recorded - int(counts.get("absent") or 0), 0)

        return [
            SubjectAttendance(
                course_ref=term,
                subject_name="",
                percentage=round((attended / recorded) * 100, 2),
                taken_sessions=recorded,
                by_status=(
                    {"description": "Present", "count": int(counts.get("present") or 0)},
                    {"description": "Absent", "count": int(counts.get("absent") or 0)},
                    {"description": "Late", "count": int(counts.get("late") or 0)},
                    {"description": "Excused", "count": int(counts.get("excused") or 0)},
                ),
                points=float(attended),
                max_points=float(recorded),
            )
        ]

    def _term_window(self, term_code: str) -> tuple[str, str] | None:
        """A term code as the pair of dates SIS's register is addressed by.

        Delegated to the injected calendar, which is the one component that knows when a
        term runs. Caching lives there too, so the dates this adapter uses and the dates
        the route reports can never drift apart.
        """
        if self._calendar is None:
            logger.warning(
                "No calendar was supplied to the SIS adapter; attendance cannot be "
                "resolved to a date window and is reported as unavailable."
            )
            return None

        term = self._calendar.term(term_code)
        if term is None or term.starts_on is None or term.ends_on is None:
            return None
        return (term.starts_on.date().isoformat(), term.ends_on.date().isoformat())

    # -- reshaping ----------------------------------------------------------

    @staticmethod
    def _to_subject_grade(row: dict) -> SubjectGrade:
        """One SIS grade line as a `SubjectGrade`. No arithmetic, by design.

        `is_graded` is read from the wire rather than derived from truthiness, because
        `if row.get("percentage"):` is false for a stated 0.0 — a mark a child earned,
        reported as never marked.

        Points without a percentage is the case worth reading twice. SIS lets a school
        state "17 out of 20" and nothing else, and turning that into 85% is this adapter
        inventing a figure the school never wrote. So the percentage stays absent and
        `academic_unavailable` carries the reason, which the contract renders as a caveat
        — a blank with no reason beside it reads as "not marked yet" and is wrong.

        The counts are per subject and mean what they say: SIS holds exactly one figure
        per subject per term, so a subject is one graded item or one pending one.
        """
        percentage = _as_float(row.get("percentage"))
        points = _as_float(row.get("points"))

        graded = (
            bool(row["is_graded"])
            if "is_graded" in row
            else (percentage is not None or points is not None)
        )

        return SubjectGrade(
            # The subject code, because that is what a course binding is keyed on. Falls
            # back to nothing rather than to a name: an unmatched binding drops the line,
            # which is the safe failure — a name here would match nothing anyway.
            course_ref=str(row.get("subject_code") or ""),
            # The protocol has one name slot and SIS is bilingual. English first because
            # the assembler overrides it from the binding whenever one carries a label;
            # this is the fallback that keeps a line from being nameless.
            subject_name_ar=str(row.get("subject_name_ar") or ""),
            subject_name=str(
                row.get("subject_name_en")
                or row.get("subject_name_ar")
                or row.get("subject_code")
                or ""
            ),
            percentage=percentage,
            # Same stated figure, not a copy of a different one: a SIS mark has no
            # attendance component to strip out. See the module docstring.
            academic_percentage=percentage,
            academic_unavailable=(
                "points_not_percentage" if graded and percentage is None else ""
            ),
            graded_count=1 if graded else 0,
            # SIS has no exclusion concept, so zero is a fact rather than a placeholder.
            excluded_count=0,
            pending_count=0 if graded else 1,
            is_complete=graded,
        )
