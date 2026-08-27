"""HTTP surface. Two routers, two key scopes, one shared audit trail.

The URL shape is load-bearing. Every parent-facing read lives under
`/v1/guardians/{guardian_id}/...`, which makes the subject a required part of the
path rather than an optional parameter someone can forget to pass. A route that reads
a student record without a guardian in its URL cannot be written by accident here —
there is nowhere to put it.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

import logging

from records import audit, auth, lms
from records.guardian_directory import PermittedStudent
from records.calendar import (
    CalendarUnavailable,
    SchoolTerm,
    get_calendar,
)
from records.assembler import AttendanceAssembler, GradeAssembler
from records.grading import DEFAULT_POLICY
from records.schemas import (
    AttendanceSummaryOut,
    CourseGrade,
    CourseGradeDetailOut,
    ErrorOut,
    StudentGradesOut,
    StudentListOut,
    StudentRef,
    TermOut,
)

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/v1", tags=["records"])
admin_router = APIRouter(prefix="/v1/admin", tags=["admin"])

# Documented once and attached to every parent-facing route, so the generated OpenAPI
# tells an integrator what the failure modes are without reading this file.
_AGENT_RESPONSES = {
    401: {"model": ErrorOut, "description": "Missing or invalid API key, or missing/invalid identity token."},
    403: {"model": ErrorOut, "description": "Identity token does not authorise the guardian named in the path."},
    404: {"model": ErrorOut, "description": "No such student record for this guardian."},
    503: {"model": ErrorOut, "description": "System of record unreachable. Do not answer from memory."},
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _student_ref(student: PermittedStudent) -> StudentRef:
    return StudentRef(
        student_id=student.external_id,
        full_name_ar=student.full_name_ar,
        full_name_en=student.full_name_en,
        grade_level=student.grade_level,
        section=student.section,
        gender=getattr(student, "gender", "") or "unspecified",
    )


def _term_out(term: SchoolTerm) -> TermOut:
    """The contract's term, from the school's own calendar.

    No timezone repair here any more. `SchoolTerm` is built at the boundary and is never
    naive, which is what the ORM row this replaced could not promise — SQLite handed those
    back without a zone, and every caller had to remember to re-attach one before
    comparing.
    """
    return TermOut(
        term_id=term.code,
        name_ar=term.name_ar,
        name_en=term.name_en,
        academic_year=term.academic_year,
        starts_on=term.starts_on,
        ends_on=term.ends_on,
        is_closed=term.is_closed,
        is_current=term.is_current,
    )


def _resolve_term(term_code: str | None) -> SchoolTerm:
    """Named term, or the one we are in — asked of the school, not remembered here.

    Falling back to the current term is what lets a parent ask "how is she doing" without
    naming one. When today falls in no term — a holiday, a gap between years — the most
    recently started term wins, because "the term that just ended" is what a parent means
    in August.

    A calendar that cannot be reached is a 503, never a 404. "No such term" and "the
    school's records are down" are different things to tell a parent, and only one of them
    is their problem.
    """
    try:
        calendar = get_calendar()
        if term_code:
            term = calendar.term(term_code)
            if term is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "unknown_term", "message": f"No term '{term_code}'."},
                )
            return term

        current = calendar.current_term()
    except CalendarUnavailable as error:
        logger.error("School calendar unavailable: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "lms_unavailable",
                "message": "The school's records are temporarily unavailable.",
            },
        ) from error

    if current is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "unknown_term", "message": "No terms are configured."},
        )
    return current


def _lms_ref(student: PermittedStudent) -> str:
    """What the LMS is asked about.

    The school's own student number, never an internal LMS id. Keeping the contract on
    the identifier a registrar can look up is what lets the system of record be replaced
    without the facade's data changing.

    Named apart from `_student_ref` above, which builds the API's StudentRef object —
    two functions with one name is how the wrong one gets called.
    """
    return student.external_id


# ---------------------------------------------------------------------------
# Parent-facing reads. Agent key + guardian in the path, always.
# ---------------------------------------------------------------------------


@agent_router.get("/terms", response_model=list[TermOut])
def list_terms(
    caller: auth.ServiceCaller = Depends(auth.require_agent),
) -> list[TermOut]:
    """Terms the school has configured. Carries no student data, so needs no subject.

    Read from the school's own calendar rather than a table here, like every other term
    in this service. Reading the local table would answer with whatever this service last
    happened to hold — which, now that terms live in SIS, is nothing.
    """
    try:
        calendar = get_calendar()
        current = calendar.current_term()
    except CalendarUnavailable as error:
        logger.error("School calendar unavailable while listing terms: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "lms_unavailable",
                "message": "The school's records are temporarily unavailable.",
            },
        ) from error

    # `current_term` is the only listing the port promises, so the answer is that term
    # when there is one. A caller wanting the whole year asks SIS, which owns the
    # calendar; this route exists so an agent can name the term it is asking about.
    return [_term_out(current)] if current is not None else []


@agent_router.get(
    "/guardians/{guardian_id}/students",
    response_model=StudentListOut,
    responses=_AGENT_RESPONSES,
)
def list_students(
    guardian_id: str,
    subject: auth.ParentSubject = Depends(auth.require_parent_subject),
) -> StudentListOut:
    """Children this guardian may ask about.

    The agent's first call in any conversation. An unknown guardian and a guardian
    with no visible children both return an empty list — see `permitted_students`.
    """
    students = auth.permitted_students(
        guardian_external_id=subject.guardian_id, school_code=subject.school_code
    )
    return StudentListOut(
        guardian_id=subject.guardian_id, students=[_student_ref(s) for s in students]
    )


@agent_router.get(
    "/guardians/{guardian_id}/students/{student_id}/grades",
    response_model=StudentGradesOut,
    responses=_AGENT_RESPONSES,
)
def get_grades(
    guardian_id: str,
    student_id: str,
    request: Request,
    term: str | None = Query(default=None, description="Term code; defaults to the current term."),
    subject: auth.ParentSubject = Depends(auth.require_parent_subject),
) -> StudentGradesOut:
    """Every subject's rollup for one student in one term."""
    student = auth.resolve_permitted_student(
        guardian_external_id=subject.guardian_id,
        student_external_id=student_id,
        caller=subject.caller,
        endpoint=str(request.url.path),
        school_code=subject.school_code,
    )
    resolved_term = _resolve_term(term)
    adapter = _adapter_or_503()

    try:
        # ONE call for the whole term, however many subjects. The per-course loop this
        # replaces was a round trip per subject, and it aggregated the results itself —
        # producing a figure that could disagree with the gradebook.
        subjects = adapter.get_subject_grades(
            student_ref=_lms_ref(student),
            term=resolved_term.code,
            # The parent this read is on behalf of, carried to the system of record so it
            # makes the same decision independently. Taken from the verified token, never
            # from anything the model or the caller supplied.
            guardian_ref=subject.guardian_id,
        )
    except lms.LmsUnavailable:
        _raise_unavailable(request, subject.caller, subject.guardian_id, student_id)

    # An empty list is the honest answer, and the agent renders it as "nothing recorded
    # for her this term" rather than inventing subjects. A subject with no mark for this
    # child in this term has no row to return, so nothing has to be filtered out here.
    courses: list[CourseGrade] = GradeAssembler(DEFAULT_POLICY).assemble(subjects)

    return StudentGradesOut(
        primary_figure=DEFAULT_POLICY.primary_figure,
        student=_student_ref(student),
        term=_term_out(resolved_term),
        courses=courses,
        as_of=_now(),
    )


@agent_router.get(
    "/guardians/{guardian_id}/students/{student_id}/grades/{course_id}",
    response_model=CourseGradeDetailOut,
    responses=_AGENT_RESPONSES,
)
def get_course_detail(
    guardian_id: str,
    student_id: str,
    course_id: str,
    request: Request,
    term: str | None = Query(default=None),
    subject: auth.ParentSubject = Depends(auth.require_parent_subject),
) -> CourseGradeDetailOut:
    """One subject in detail — the figures behind "why is her maths grade 72".

    The per-ASSESSMENT breakdown is not served yet, and returns an empty list rather
    than a fabricated one. `local_schoolapi` summarises per subject; it has no
    per-assessment endpoint, and the core Moodle call that would provide one cannot
    express whether an assessment was excused — so a list built from it would show a
    counted zero where the child was excused, which is precisely the error this system
    exists to prevent.

    An empty list is honest but not harmless: the agent must say it does not have the
    breakdown, NOT that there are no assessments. The template that renders this
    distinguishes the two.

    The subject-level figures below are complete and correct, so "her maths is 72,
    with one assessment excused and one still to be marked" is answerable today; only
    the item-by-item list is missing.
    """
    student = auth.resolve_permitted_student(
        guardian_external_id=subject.guardian_id,
        student_external_id=student_id,
        caller=subject.caller,
        endpoint=str(request.url.path),
        school_code=subject.school_code,
    )
    resolved_term = _resolve_term(term)
    adapter = _adapter_or_503()

    try:
        subjects = adapter.get_subject_grades(
            student_ref=_lms_ref(student),
            term=resolved_term.code,
            guardian_ref=subject.guardian_id,
        )
    except lms.LmsUnavailable:
        _raise_unavailable(request, subject.caller, subject.guardian_id, student_id)

    # Filtered from what this child actually has rather than trusted from the path, so a
    # caller cannot read an arbitrary subject by guessing its id: the code has to appear
    # in the set the system of record returned for *this* child in *this* term, and one
    # that is not hers matches nothing.
    courses = [
        course
        for course in GradeAssembler(DEFAULT_POLICY).assemble(subjects)
        if course.course_id == course_id
    ]
    if not courses:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "No such subject for this student this term."},
        )

    return CourseGradeDetailOut(
        student=_student_ref(student),
        term=_term_out(resolved_term),
        course=courses[0],
        assignments=[],
        as_of=_now(),
    )


@agent_router.get(
    "/guardians/{guardian_id}/students/{student_id}/attendance",
    response_model=AttendanceSummaryOut,
    responses=_AGENT_RESPONSES,
)
def get_attendance(
    guardian_id: str,
    student_id: str,
    request: Request,
    term: str | None = Query(default=None),
    subject: auth.ParentSubject = Depends(auth.require_parent_subject),
) -> AttendanceSummaryOut:
    """Attendance totals for a term, with the recent days behind them."""
    student = auth.resolve_permitted_student(
        guardian_external_id=subject.guardian_id,
        student_external_id=student_id,
        caller=subject.caller,
        endpoint=str(request.url.path),
        school_code=subject.school_code,
    )
    resolved_term = _resolve_term(term)
    adapter = _adapter_or_503()

    subjects: list = []
    try:
        # ONE call for the term. The shape this replaces walked every session in every
        # course, and the system of record handed back the whole class each time — so the
        # facade received attendance for children this guardian may not see. That is now
        # impossible rather than filtered.
        subjects = adapter.get_subject_attendance(
            student_ref=_lms_ref(student),
            term=resolved_term.code,
            guardian_ref=subject.guardian_id,
        )
    except lms.LmsUnavailable:
        _raise_unavailable(request, subject.caller, subject.guardian_id, student_id)

    assembler = AttendanceAssembler()
    visible = assembler.visible(subjects)
    counts = assembler.counts(visible)

    return AttendanceSummaryOut(
        student=_student_ref(student),
        term=_term_out(resolved_term),
        present_count=counts["present"],
        absent_count=counts["absent"],
        late_count=counts["late"],
        excused_count=counts["excused"],
        total_sessions=sum(s.taken_sessions for s in visible),
        # The system of record's own weighting, aggregated across subjects by points
        # rather than by averaging percentages — see AttendanceAssembler. A school
        # decides what a late arrival or an excused absence costs; this reports it.
        attendance_rate=assembler.term_percentage(visible),
        recent_days=assembler.recent_days(visible),
        as_of=_now(),
    )


def _adapter_or_503():
    try:
        return lms.get_adapter()
    except lms.LmsUnavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "lms_unavailable", "message": "System of record is unreachable."},
        )


def _raise_unavailable(request, caller, guardian_id, student_id):
    """Report the failed read, then fail loudly.

    The report matters as much as the error: a spike of `lms_unavailable` against one
    student is how a sync problem gets noticed before a parent reports it. It is a
    structured log line rather than a row because this service holds no database — see
    `records.audit`, and `sis/` for the trail that records answers about children.
    """
    audit.refused(
        audit.LMS_UNAVAILABLE,
        endpoint=str(request.url.path),
        guardian_id=guardian_id,
        student_id=student_id,
        request_id=caller.request_id,
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "lms_unavailable", "message": "System of record is unreachable."},
    )


# ---------------------------------------------------------------------------
# Admin. Separate scope; cannot read a student record through these routes.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Guardian administration — moved, not removed
# ---------------------------------------------------------------------------
#
# These three routes used to write this service's own guardian tables. Those tables are no
# longer read: who a child's parents are is the registrar's fact and lives in SIS, and this
# service asks rather than remembers (see records/guardian_directory.py).
#
# They answer 410 instead of being deleted outright. A removed route is a 404, which a
# caller reads as "wrong URL" and retries; a 410 says the thing itself is gone and names
# where it went. What they must never do is what they would do if left alone — accept the
# write, return 201, and change nothing, so that a registrar believes a parent has been
# granted access to their child's records when nobody has.
#
# They take no credential. They hold no data and reveal nothing a reader of this repo does
# not already know; a caller who cannot authenticate is exactly the caller most in need of
# being told the route moved, and there is no longer an admin scope here to gate them with.

_MOVED_TO_SIS = {
    "code": "moved",
    "message": (
        "Guardians are managed in the school's SIS. Upload them there "
        "(POST /v1/imports/guardians/preview) and change records access with "
        "PATCH /v1/students/{student_number}/guardians/{phone}."
    ),
}


@admin_router.post("/guardians", status_code=status.HTTP_410_GONE)
def create_guardian_moved() -> dict:
    """Gone. Guardians are created by the SIS spreadsheet import."""
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_MOVED_TO_SIS)


@admin_router.post("/guardians/{guardian_id}/students", status_code=status.HTTP_410_GONE)
def link_student_moved(guardian_id: str) -> dict:
    """Gone. Links are created by the SIS import and amended on the SIS access route."""
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_MOVED_TO_SIS)


@admin_router.delete(
    "/guardians/{guardian_id}/students/{student_id}", status_code=status.HTTP_410_GONE
)
def unlink_student_moved(guardian_id: str, student_id: str) -> dict:
    """Gone. Unlinking is a SIS operation."""
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_MOVED_TO_SIS)
