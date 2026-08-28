"""The parent-facing reads.

The URL shape is load-bearing. Every one of these lives under
`/v1/guardians/{guardian_id}/...`, which makes the subject a required part of the path
rather than an optional parameter someone can forget to pass. A route that reads a student
record without a guardian in its URL cannot be written by accident here — there is nowhere
to put it.

Each handler is now an adapter: take the request, call one use case, map the result. The
three-step order that matters — link check, then term, then the system of record — lives
in `application/reads.py`, where it is stated once instead of four times, and no
`try/except` appears below because `api/errors.py` turns a domain error into a status in
one place.
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from records.api.deps import AgentCaller, ParentSubjectDep, RecordsServiceDep
from records.api.schemas.contract import (
    AttendanceSummaryOut,
    CourseGradeDetailOut,
    ErrorOut,
    StudentGradesOut,
    StudentListOut,
    StudentRef,
    TermOut,
)
from records.application import audit
from records.domain.people import PermittedStudent
from records.domain.terms import SchoolTerm

router = APIRouter(prefix="/v1", tags=["records"])

# Documented once and attached to every parent-facing route, so the generated OpenAPI
# tells an integrator what the failure modes are without reading this file.
AGENT_RESPONSES = {
    401: {"model": ErrorOut, "description": "Missing or invalid API key, or missing/invalid identity token."},
    403: {"model": ErrorOut, "description": "Identity token does not authorise the guardian named in the path."},
    404: {"model": ErrorOut, "description": "No such student record for this guardian."},
    503: {"model": ErrorOut, "description": "System of record unreachable. Do not answer from memory."},
}


def _student_ref(student: PermittedStudent) -> StudentRef:
    return StudentRef(
        student_id=student.external_id,
        full_name_ar=student.full_name_ar,
        full_name_en=student.full_name_en,
        grade_level=student.grade_level,
        section=student.section,
        gender=student.gender,
    )


def _term_out(term: SchoolTerm) -> TermOut:
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


@router.get("/terms", response_model=list[TermOut])
def list_terms(service: RecordsServiceDep, _: AgentCaller) -> list[TermOut]:
    """Terms the school has configured. Carries no student data, so needs no subject.

    `current_term` is the only listing the port promises, so the answer is that term when
    there is one. A caller wanting the whole year asks SIS, which owns the calendar; this
    route exists so an agent can name the term it is asking about.
    """
    current = service.current_term()
    return [_term_out(current)] if current is not None else []


@router.get(
    "/guardians/{guardian_id}/students",
    response_model=StudentListOut,
    responses=AGENT_RESPONSES,
)
def list_students(guardian_id: str, subject: ParentSubjectDep,
                  service: RecordsServiceDep) -> StudentListOut:
    """Children this guardian may ask about.

    The agent's first call in any conversation. An unknown guardian and a guardian with no
    visible children both return an empty list — see `AccessService.permitted_students`.
    """
    students = service.students(
        guardian_id=subject.guardian_id, school_code=subject.school_code
    )
    return StudentListOut(
        guardian_id=subject.guardian_id,
        students=[_student_ref(s) for s in students],
    )


@router.get(
    "/guardians/{guardian_id}/students/{student_id}/grades",
    response_model=StudentGradesOut,
    responses=AGENT_RESPONSES,
)
def get_grades(
    guardian_id: str,
    student_id: str,
    request: Request,
    subject: ParentSubjectDep,
    service: RecordsServiceDep,
    term: str | None = Query(default=None, description="Term code; defaults to the current term."),
) -> StudentGradesOut:
    """Every subject's rollup for one student in one term."""
    with audit.reporting_unavailable(request, subject, student_id):
        result = service.grades(
            guardian_id=subject.guardian_id,
            student_id=student_id,
            term_code=term,
            school_code=subject.school_code,
        )
    return StudentGradesOut(
        primary_figure=result.primary_figure,
        student=_student_ref(result.student),
        term=_term_out(result.term),
        courses=result.courses,
        as_of=result.as_of,
    )


@router.get(
    "/guardians/{guardian_id}/students/{student_id}/grades/{course_id}",
    response_model=CourseGradeDetailOut,
    responses=AGENT_RESPONSES,
)
def get_course_detail(
    guardian_id: str,
    student_id: str,
    course_id: str,
    request: Request,
    subject: ParentSubjectDep,
    service: RecordsServiceDep,
    term: str | None = Query(default=None),
) -> CourseGradeDetailOut:
    """One subject in detail — the figures behind "why is her maths grade 72".

    The per-ASSESSMENT breakdown is not served yet, and returns an empty list rather than
    a fabricated one. `local_schoolapi` summarises per subject; it has no per-assessment
    endpoint, and the core Moodle call that would provide one cannot express whether an
    assessment was excused — so a list built from it would show a counted zero where the
    child was excused, which is precisely the error this system exists to prevent.

    An empty list is honest but not harmless: the agent must say it does not have the
    breakdown, NOT that there are no assessments. The template that renders this
    distinguishes the two.

    The subject-level figures below are complete and correct, so "her maths is 72, with
    one assessment excused and one still to be marked" is answerable today; only the
    item-by-item list is missing.
    """
    with audit.reporting_unavailable(request, subject, student_id):
        result = service.course_detail(
            guardian_id=subject.guardian_id,
            student_id=student_id,
            course_id=course_id,
            term_code=term,
            school_code=subject.school_code,
        )
    return CourseGradeDetailOut(
        student=_student_ref(result.student),
        term=_term_out(result.term),
        course=result.course,
        assignments=[],
        as_of=result.as_of,
    )


@router.get(
    "/guardians/{guardian_id}/students/{student_id}/attendance",
    response_model=AttendanceSummaryOut,
    responses=AGENT_RESPONSES,
)
def get_attendance(
    guardian_id: str,
    student_id: str,
    request: Request,
    subject: ParentSubjectDep,
    service: RecordsServiceDep,
    term: str | None = Query(default=None),
) -> AttendanceSummaryOut:
    """Attendance totals for a term, with the recent days behind them."""
    with audit.reporting_unavailable(request, subject, student_id):
        result = service.attendance(
            guardian_id=subject.guardian_id,
            student_id=student_id,
            term_code=term,
            school_code=subject.school_code,
        )
    counts = result.counts
    return AttendanceSummaryOut(
        student=_student_ref(result.student),
        term=_term_out(result.term),
        present_count=counts["present"],
        absent_count=counts["absent"],
        late_count=counts["late"],
        excused_count=counts["excused"],
        total_sessions=result.total_sessions,
        attendance_rate=result.attendance_rate,
        recent_days=result.recent_days,
        as_of=result.as_of,
    )


__all__ = ["AGENT_RESPONSES", "router"]
