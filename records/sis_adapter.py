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
from typing import Any
from urllib.parse import quote

from records.lms import LmsUnavailable, SubjectAttendance, SubjectGrade

logger = logging.getLogger(__name__)

#: The SIS error code for "this code names nothing" — an unknown student, or an unknown
#: term. The only 404 this adapter is willing to read as "nothing on file".
_UNKNOWN_REFERENCE = "unknown_reference"


def _as_float(value: Any) -> float | None:
    """None stays None. Zero stays zero.

    The same rule `MoodleAdapter._as_float` enforces, restated rather than imported so
    neither adapter reaches into the other's internals. Both exist because
    `float(value or 0)` is the one-character mistake that tells a parent their child
    scored nothing in a subject nobody has marked.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SisAdapter:
    """Reads one student's term marks from the Student Information Service.

    `GET /v1/students/{student_number}/grades?term=` with a `reader`-scoped key, and
    nothing else. The adapter is not an authorisation boundary: the facade has already
    decided this guardian may see this student before it is called.

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

    #: Connections held open to the SIS. Sized for concurrent parents in one worker, not
    #: for throughput — a pool larger than the SIS's own worker count only queues.
    POOL_SIZE = 10

    #: A hung call must not hang a chat turn. Failing at 10s with "I can't reach the
    #: records" beats succeeding at 90s to a parent watching a streamed answer.
    DEFAULT_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float | None = None,
        pool_size: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("SIS_TIMEOUT_SECONDS") or self.DEFAULT_TIMEOUT_SECONDS
        )
        self.pool_size = pool_size or self.POOL_SIZE

        self._lock = threading.Lock()
        self._session: Any = None

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
        """
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
                            max_keepalive_connections=self.pool_size,
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

    # -- the protocol -------------------------------------------------------

    def get_subject_grades(self, *, student_ref: str, term: str) -> list[SubjectGrade]:
        # The student number is quoted with no safe characters: a stray "/" in a school's
        # numbering would otherwise rewrite the path and ask a different question.
        path = f"/v1/students/{quote(student_ref, safe='')}/grades"
        payload = self._get(path, {"term": term})

        if payload is None:
            return []

        return [self._to_subject_grade(row) for row in payload.get("grades") or []]

    def get_subject_attendance(self, *, student_ref: str, term: str) -> list[SubjectAttendance]:
        """Refuses, loudly, because the alternative is a comfortable lie.

        SIS records grades only — attendance stays in the systems that own it. Returning
        `[]` would be read by the assembler as "no register has anything against this
        child" and reported to a parent as a clean record, which is the same sentence a
        school would use for a child with perfect attendance. `LmsUnavailable` produces
        "I cannot reach the school records right now" instead: unhelpful, and true.
        """
        raise LmsUnavailable(
            "SIS does not serve attendance in this phase — it records grades only. "
            "Attendance is unavailable while RECORDS_LMS=sis."
        )

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
