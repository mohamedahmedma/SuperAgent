"""The four reads this facade serves, with no FastAPI in them.

Each is the same three steps in the same order, and the order is a security property
rather than a style:

    1. resolve the child through the guardian link      (may refuse)
    2. resolve the term through the school's calendar   (may refuse)
    3. ask the system of record                         (may be unavailable)

**Step 3 never runs before step 1 succeeds.** The marks call's arguments are all known
before the link check returns — `student_ref` is the path's own student id — so the two
could be issued together and the request would be one round trip faster. They are not,
deliberately: that would ask the system of record for a child's marks before establishing
that this parent may see them, and "the LMS is never asked about a student the link check
excluded" is a stated property of this service. The caller here is a language model
reading untrusted parent text, which is exactly why the ordering is not a formality.

Lifted out of `records/routes.py`, where the same three steps were written out four times
with the differences buried in the middle.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from records.application.access import AccessService
from records.application.assembly import AttendanceAssembler, GradeAssembler
from records.domain.errors import CalendarUnavailable, StudentNotFound, UnknownTerm
from records.domain.grading import GradingPolicy
from records.domain.marks import SubjectAttendance
from records.domain.people import PermittedStudent
from records.domain.terms import SchoolTerm
from records.ports.calendar import SchoolCalendar
from records.ports.lms import LmsAdapter


@dataclass(frozen=True, slots=True)
class GradesResult:
    student: PermittedStudent
    term: SchoolTerm
    courses: list
    primary_figure: str
    as_of: datetime


@dataclass(frozen=True, slots=True)
class CourseDetailResult:
    student: PermittedStudent
    term: SchoolTerm
    course: object
    as_of: datetime


@dataclass(frozen=True, slots=True)
class AttendanceResult:
    student: PermittedStudent
    term: SchoolTerm
    counts: dict[str, int]
    total_sessions: int
    attendance_rate: float | None
    recent_days: list
    as_of: datetime


class RecordsService:
    """Every parent-facing read, over the ports the deployment wired in.

    Built per request by `api/deps.py` and holding nothing but references — the pooled
    HTTP clients live on the adapters, which are process-wide — so constructing one is a
    few attribute assignments on a path that must stay cheap.
    """

    def __init__(
        self,
        *,
        access: AccessService,
        calendar: SchoolCalendar,
        lms: LmsAdapter,
        policy: GradingPolicy,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._access = access
        self._calendar = calendar
        self._lms = lms
        self._policy = policy
        self._clock = clock
        # One assembler each, not one per request. They are stateless — `assemble` is a
        # pure function of its argument and the policy — so rebuilding them per call was
        # allocation on the hot path for nothing.
        self._grades = GradeAssembler(policy)
        self._attendance = AttendanceAssembler()

    # -- terms --------------------------------------------------------------

    def current_term(self) -> SchoolTerm | None:
        """The term the school is in, read from its own calendar.

        Read from the calendar rather than a table here, like every other term in this
        service. Reading a local table would answer with whatever this service last
        happened to hold — which, now that terms live in SIS, is nothing.
        """
        return self._calendar.current_term()

    def resolve_term(self, term_code: str | None) -> SchoolTerm:
        """The named term, or the current one when the caller named none.

        Falling back to the current term is what lets a parent ask "how is she doing"
        without naming one. When today falls in no term — a holiday, a gap between years —
        the most recently started term wins, because "the term that just ended" is what a
        parent means in August.

        An unknown code is a refusal rather than a silent fallback to the current term: a
        parent asking about last term and being shown this one is a wrong answer that
        looks right.

        Note the two failures are different and stay different. "No such term" is the
        caller's to fix and is a 404; a calendar that cannot be reached is a 503, and
        `CalendarUnavailable` propagates untouched to say so. Only one of them is the
        parent's problem.
        """
        if term_code:
            found = self._calendar.term(term_code)
            if found is None:
                raise UnknownTerm(f"No term '{term_code}'.")
            return found

        current = self._calendar.current_term()
        if current is None:
            raise UnknownTerm("No terms are configured.")
        return current

    # -- the reads ----------------------------------------------------------

    def students(self, *, guardian_id: str, school_code: str | None = None):
        """Children this guardian may ask about. The agent's first call in a conversation."""
        return self._access.permitted_students(
            guardian_external_id=guardian_id, school_code=school_code
        )

    def grades(
        self, *, guardian_id: str, student_id: str, term_code: str | None,
        school_code: str | None = None,
    ) -> GradesResult:
        """Every subject's rollup for one student in one term."""
        student = self._resolve_student(guardian_id, student_id, school_code)
        term = self.resolve_term(term_code)

        # ONE call for the whole term, however many subjects. The per-course loop this
        # replaces was a round trip per subject, and it aggregated the results itself —
        # producing a figure that could disagree with the gradebook.
        subjects = self._lms.get_subject_grades(
            student_ref=student.external_id,
            term=term.code,
            # The parent this read is on behalf of, carried to the system of record so it
            # makes the same decision independently. Taken from the verified token, never
            # from anything the model or the caller supplied.
            guardian_ref=guardian_id,
        )

        # An empty list is the honest answer, and the agent renders it as "nothing
        # recorded for her this term" rather than inventing subjects.
        return GradesResult(
            student=student,
            term=term,
            courses=self._grades.assemble(subjects),
            primary_figure=self._policy.primary_figure,
            as_of=self._clock(),
        )

    def course_detail(
        self, *, guardian_id: str, student_id: str, course_id: str,
        term_code: str | None, school_code: str | None = None,
    ) -> CourseDetailResult:
        """One subject in detail — the figures behind "why is her maths grade 72"."""
        student = self._resolve_student(guardian_id, student_id, school_code)
        term = self.resolve_term(term_code)

        subjects = self._lms.get_subject_grades(
            student_ref=student.external_id, term=term.code, guardian_ref=guardian_id
        )

        # Filtered from what this child actually has rather than trusted from the path, so
        # a caller cannot read an arbitrary subject by guessing its id: the code has to
        # appear in the set the system of record returned for *this* child in *this*
        # term, and one that is not hers matches nothing.
        for course in self._grades.assemble(subjects):
            if course.course_id == course_id:
                return CourseDetailResult(
                    student=student, term=term, course=course, as_of=self._clock()
                )
        raise StudentNotFound("No such subject for this student this term.")

    def attendance(
        self, *, guardian_id: str, student_id: str, term_code: str | None,
        school_code: str | None = None,
    ) -> AttendanceResult:
        """Attendance totals for a term, with the recent days behind them."""
        student = self._resolve_student(guardian_id, student_id, school_code)
        term = self.resolve_term(term_code)

        # ONE call for the term. The shape this replaces walked every session in every
        # course, and the system of record handed back the whole class each time — so the
        # facade received attendance for children this guardian may not see. That is now
        # impossible rather than filtered.
        subjects: list[SubjectAttendance] = self._lms.get_subject_attendance(
            student_ref=student.external_id, term=term.code, guardian_ref=guardian_id
        )

        visible = self._attendance.visible(subjects)
        return AttendanceResult(
            student=student,
            term=term,
            counts=self._attendance.counts(visible),
            total_sessions=sum(s.taken_sessions for s in visible),
            # The system of record's own weighting, aggregated across subjects by points
            # rather than by averaging percentages — see AttendanceAssembler. A school
            # decides what a late arrival or an excused absence costs; this reports it.
            attendance_rate=self._attendance.term_percentage(visible),
            recent_days=self._attendance.recent_days(visible),
            as_of=self._clock(),
        )

    # -- internals ----------------------------------------------------------

    def _resolve_student(
        self, guardian_id: str, student_id: str, school_code: str | None
    ) -> PermittedStudent:
        """Step 1, always, and always before the system of record is asked."""
        return self._access.resolve(
            guardian_external_id=guardian_id,
            student_external_id=student_id,
            school_code=school_code,
        )


__all__ = [
    "AttendanceResult",
    "CourseDetailResult",
    "GradesResult",
    "RecordsService",
]
