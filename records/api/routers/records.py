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
    ClassroomStatusOut,
    ClassTeacherOut,
    CourseGradeDetailOut,
    ErrorOut,
    StudentClassOut,
    StudentGradesOut,
    StudentListOut,
    StudentRef,
    StudentSubjectsOut,
    StudentTeachersOut,
    StudySubjectOut,
    TermOut,
    TimetableLessonOut,
    TimetableOut,
    TimetablePeriodOut,
    TimetableStatusOut,
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


@router.get(
    "/guardians/{guardian_id}/students/{student_id}/timetable",
    response_model=TimetableOut,
    responses=AGENT_RESPONSES,
)
def get_timetable(
    guardian_id: str,
    student_id: str,
    request: Request,
    subject: ParentSubjectDep,
    service: RecordsServiceDep,
    term: str | None = Query(default=None, description="Term code; defaults to the current term."),
) -> TimetableOut:
    """One child's weekly timetable — the class she sits in, and what it does.

    **The caller does not name a class, and cannot.** A timetable belongs to a room; a child
    reaches one through the placement she holds for the term, and that placement is the
    system of record's to resolve. So this route takes a student, exactly as the grades and
    attendance routes do, and the room is resolved behind it — which is what keeps a chat
    service from holding a class code that goes stale the next time a child moves.

    Three answers, and `status` says which. Two of them have no lessons and they mean
    different things: `no_class` is a fact about the child, `no_timetable` a fact about the
    school. A caller must not render them alike — see `TimetableStatusOut`.
    """
    with audit.reporting_unavailable(request, subject, student_id):
        result = service.timetable(
            guardian_id=subject.guardian_id,
            student_id=student_id,
            term_code=term,
            school_code=subject.school_code,
        )
    week = result.timetable
    return TimetableOut(
        student=_student_ref(result.student),
        term=_term_out(result.term),
        status=TimetableStatusOut(week.status.value),
        class_code=week.class_code,
        class_name_ar=week.class_name_ar,
        class_name_en=week.class_name_en,
        days=list(week.days),
        periods=[
            TimetablePeriodOut(
                period_number=period.period_number,
                name_ar=period.name_ar,
                name_en=period.name_en,
                starts_at=period.starts_at,
                ends_at=period.ends_at,
                is_teaching=period.is_teaching,
            )
            for period in week.periods
        ],
        lessons=[
            TimetableLessonOut(
                day_of_week=lesson.day_of_week,
                period_number=lesson.period_number,
                subject_code=lesson.subject_code,
                subject_name_ar=lesson.subject_name_ar,
                subject_name_en=lesson.subject_name_en,
            )
            for lesson in week.lessons
        ],
        teaching_slots=week.teaching_slots,
        as_of=result.as_of,
    )


# ---------------------------------------------------------------------------
# The room she sits in. Three routes, one read.
# ---------------------------------------------------------------------------
#
# Three URLs because a parent asks three different questions, and one read behind them
# because all three describe the SAME room — resolved from one time-bounded placement. Asked
# as three reads, a placement edited in between would answer with one room's subjects beside
# another room's staff. See records/ports/classroom.py.
#
# None of the three takes a class. A parent has no class code and this service holds none;
# the system of record resolves the room from the child, which is what keeps a cached code
# from eventually naming a room she has left.


@router.get(
    "/guardians/{guardian_id}/students/{student_id}/class",
    response_model=StudentClassOut,
    responses=AGENT_RESPONSES,
)
def get_class(
    guardian_id: str,
    student_id: str,
    request: Request,
    subject: ParentSubjectDep,
    service: RecordsServiceDep,
    term: str | None = Query(default=None, description="Term code; defaults to the current term."),
) -> StudentClassOut:
    """Which class a child is in.

    The answer a parent wants is `class_name_ar` / `class_name_en` — what the school calls
    the room, "Primary 1 Class 1" or "3/1". `class_code` is the internal key for the same
    room and is not what a family should be shown.

    The class is the one she was in **for this term**. `status: no_class` means no placement
    covered it — she had left, or had not yet joined — which is a real answer and not a
    missing record.
    """
    with audit.reporting_unavailable(request, subject, student_id):
        result = service.classroom(
            guardian_id=subject.guardian_id,
            student_id=student_id,
            term_code=term,
            school_code=subject.school_code,
        )
    room = result.classroom
    return StudentClassOut(
        student=_student_ref(result.student),
        term=_term_out(result.term),
        status=ClassroomStatusOut(room.status.value),
        class_code=room.class_code,
        class_name_ar=room.class_name_ar,
        class_name_en=room.class_name_en,
        year_level_code=room.year_level_code,
        year_level_name_ar=room.year_level_name_ar,
        year_level_name_en=room.year_level_name_en,
        as_of=result.as_of,
    )


@router.get(
    "/guardians/{guardian_id}/students/{student_id}/subjects",
    response_model=StudentSubjectsOut,
    responses=AGENT_RESPONSES,
)
def get_subjects(
    guardian_id: str,
    student_id: str,
    request: Request,
    subject: ParentSubjectDep,
    service: RecordsServiceDep,
    term: str | None = Query(default=None, description="Term code; defaults to the current term."),
) -> StudentSubjectsOut:
    """What a child studies — the subjects assigned to her rung, in the school's own order.

    Her rung's assignment board rather than the year's whole catalogue, and rather than the
    subjects she happens to have marks in: a subject nobody has marked yet is still one she
    studies, and a mark can exist against a subject the board has since dropped.

    An empty list with `status: ok` means the school has not curated the board for this year.
    That is not the child studying nothing.
    """
    with audit.reporting_unavailable(request, subject, student_id):
        result = service.classroom(
            guardian_id=subject.guardian_id,
            student_id=student_id,
            term_code=term,
            school_code=subject.school_code,
        )
    room = result.classroom
    return StudentSubjectsOut(
        student=_student_ref(result.student),
        term=_term_out(result.term),
        status=ClassroomStatusOut(room.status.value),
        class_code=room.class_code,
        class_name_ar=room.class_name_ar,
        class_name_en=room.class_name_en,
        subjects=[
            StudySubjectOut(
                code=item.code, name_ar=item.name_ar, name_en=item.name_en
            )
            for item in room.subjects
        ],
        as_of=result.as_of,
    )


@router.get(
    "/guardians/{guardian_id}/students/{student_id}/teachers",
    response_model=StudentTeachersOut,
    responses=AGENT_RESPONSES,
)
def get_teachers(
    guardian_id: str,
    student_id: str,
    request: Request,
    subject: ParentSubjectDep,
    service: RecordsServiceDep,
    term: str | None = Query(default=None, description="Term code; defaults to the current term."),
) -> StudentTeachersOut:
    """Who teaches a child, and what each of them teaches her.

    One entry per (teacher, subject), so a subject with two teachers has two entries and a
    teacher taking two subjects appears twice. A caller asking "who teaches her maths"
    filters this by subject and must render everyone it finds — taking the first would hide
    a co-teacher.

    **Who teaches the room now, not who taught it in the term asked about.** The school's
    assignment table carries no term, so this cannot be reported historically. Teachers who
    have left are excluded: someone who is gone is not the answer to "who teaches my
    daughter maths", and naming them sends a parent to ask for them.

    Nothing here carries a teacher's email, phone, staff number or username.
    """
    with audit.reporting_unavailable(request, subject, student_id):
        result = service.classroom(
            guardian_id=subject.guardian_id,
            student_id=student_id,
            term_code=term,
            school_code=subject.school_code,
        )
    room = result.classroom
    return StudentTeachersOut(
        student=_student_ref(result.student),
        term=_term_out(result.term),
        status=ClassroomStatusOut(room.status.value),
        class_code=room.class_code,
        class_name_ar=room.class_name_ar,
        class_name_en=room.class_name_en,
        teachers=[
            ClassTeacherOut(
                full_name_ar=teacher.full_name_ar,
                full_name_en=teacher.full_name_en,
                subject_code=teacher.subject_code,
                subject_name_ar=teacher.subject_name_ar,
                subject_name_en=teacher.subject_name_en,
            )
            for teacher in room.teachers
        ],
        as_of=result.as_of,
    )


__all__ = ["AGENT_RESPONSES", "router"]
