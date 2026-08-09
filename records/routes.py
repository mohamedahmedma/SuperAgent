"""HTTP surface. Two routers, two key scopes, one shared audit trail.

The URL shape is load-bearing. Every parent-facing read lives under
`/v1/guardians/{guardian_id}/...`, which makes the subject a required part of the
path rather than an optional parameter someone can forget to pass. A route that reads
a student record without a guardian in its URL cannot be written by accident here —
there is nowhere to put it.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from records import auth, lms
from records.db import get_db
from records.grading import GradingPolicy, compute_rollup
from records.models import (
    AccessAudit,
    ApiKey,
    CourseBinding,
    Guardian,
    GuardianStudent,
    ReportCard,
    Student,
    Term,
)
from records.schemas import (
    ApiKeyIn,
    ApiKeyOut,
    AttendanceSummaryOut,
    AuditEntryOut,
    CourseGrade,
    CourseGradeDetailOut,
    ErrorOut,
    GuardianIn,
    GuardianLinkIn,
    ReportCardOut,
    StudentGradesOut,
    StudentListOut,
    StudentRef,
    TermOut,
)

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


def _student_ref(student: Student) -> StudentRef:
    return StudentRef(
        student_id=student.external_id,
        full_name_ar=student.full_name_ar,
        full_name_en=student.full_name_en,
        grade_level=student.grade_level,
        section=student.section,
    )


def _term_out(term: Term) -> TermOut:
    now = _now()
    starts, ends = term.starts_on, term.ends_on
    # SQLite hands back naive datetimes even from a timezone-aware column, so the
    # comparison below would raise on a perfectly valid row. Normalise rather than
    # forbid SQLite, which is the default for local development.
    if starts.tzinfo is None:
        starts = starts.replace(tzinfo=timezone.utc)
    if ends.tzinfo is None:
        ends = ends.replace(tzinfo=timezone.utc)

    return TermOut(
        term_id=term.code,
        name_ar=term.name_ar,
        name_en=term.name_en,
        academic_year=term.academic_year,
        starts_on=starts,
        ends_on=ends,
        is_closed=term.is_closed,
        is_current=starts <= now <= ends,
    )


def _resolve_term(db: Session, term_code: str | None) -> Term:
    """Named term, or the one we are in.

    Falling back to the current term is what lets a parent ask "how is she doing"
    without naming a term. When no term contains today — a school holiday, a gap
    between years — the most recently started term wins, because "the term that just
    ended" is what a parent means in August.
    """
    if term_code:
        term = db.query(Term).filter(Term.code == term_code).first()
        if term is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "unknown_term", "message": f"No term '{term_code}'."},
            )
        return term

    now = _now().replace(tzinfo=None)
    current = (
        db.query(Term)
        .filter(Term.starts_on <= now, Term.ends_on >= now)
        .order_by(Term.starts_on.desc())
        .first()
    )
    if current is not None:
        return current

    fallback = db.query(Term).order_by(Term.starts_on.desc()).first()
    if fallback is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "unknown_term", "message": "No terms are configured."},
        )
    return fallback


def _bindings_for(db: Session, student: Student, term: Term) -> list[CourseBinding]:
    """Published courses for this student's grade level and section in this term.

    Unpublished bindings are invisible here by design — a registrar entering next
    term's courses, or a teacher's sandbox, cannot reach a parent.
    """
    return (
        db.query(CourseBinding)
        .filter(
            CourseBinding.term_id == term.id,
            CourseBinding.grade_level == student.grade_level,
            CourseBinding.section == student.section,
            CourseBinding.is_published.is_(True),
        )
        .all()
    )


def _course_grade(binding: CourseBinding, assignments, policy: GradingPolicy) -> CourseGrade:
    rollup = compute_rollup(assignments, policy)
    return CourseGrade(
        course_id=str(binding.lms_course_id),
        subject_code=binding.subject_code,
        subject_name_ar=binding.subject_name_ar,
        subject_name_en=binding.subject_name_en,
        computed_percentage=rollup.percentage,
        letter_grade=rollup.letter,
        passed=rollup.passed,
        graded_count=rollup.graded_count,
        excused_count=rollup.excused_count,
        missing_count=rollup.missing_count,
        pending_count=rollup.pending_count,
        is_complete=rollup.is_complete,
    )


# ---------------------------------------------------------------------------
# Parent-facing reads. Agent key + guardian in the path, always.
# ---------------------------------------------------------------------------


@agent_router.get("/terms", response_model=list[TermOut])
def list_terms(
    db: Session = Depends(get_db),
    caller: auth.Caller = Depends(auth.require_agent_key),
) -> list[TermOut]:
    """Terms the school has configured. Carries no student data, so needs no subject."""
    terms = db.query(Term).order_by(Term.starts_on.desc()).all()
    return [_term_out(t) for t in terms]


@agent_router.get(
    "/guardians/{guardian_id}/students",
    response_model=StudentListOut,
    responses=_AGENT_RESPONSES,
)
def list_students(
    guardian_id: str,
    db: Session = Depends(get_db),
    subject: auth.ParentSubject = Depends(auth.require_parent_subject),
) -> StudentListOut:
    """Children this guardian may ask about.

    The agent's first call in any conversation. An unknown guardian and a guardian
    with no visible children both return an empty list — see `permitted_students`.
    """
    students = auth.permitted_students(db, guardian_external_id=subject.guardian_id)
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
    db: Session = Depends(get_db),
    subject: auth.ParentSubject = Depends(auth.require_parent_subject),
) -> StudentGradesOut:
    """Every subject's rollup for one student in one term."""
    student = auth.resolve_permitted_student(
        db,
        guardian_external_id=subject.guardian_id,
        student_external_id=student_id,
        caller=subject.caller,
        endpoint=str(request.url.path),
    )
    resolved_term = _resolve_term(db, term)
    bindings = _bindings_for(db, student, resolved_term)

    # No LMS account yet, or no published courses: an empty `courses` list is the
    # honest answer to both, and the agent renders it as "nothing recorded for her
    # this term" rather than inventing subjects.
    courses: list[CourseGrade] = []
    if student.lms_user_id is not None and bindings:
        adapter = _adapter_or_503()
        policy = GradingPolicy()
        try:
            for binding in bindings:
                assignments = adapter.get_assignment_grades(
                    lms_user_id=student.lms_user_id, course_id=binding.lms_course_id
                )
                courses.append(_course_grade(binding, assignments, policy))
        except lms.LmsUnavailable:
            _raise_unavailable(db, subject.caller, request, subject.guardian_id, student_id)

    return StudentGradesOut(
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
    db: Session = Depends(get_db),
    subject: auth.ParentSubject = Depends(auth.require_parent_subject),
) -> CourseGradeDetailOut:
    """Assignment-level breakdown behind one subject — "why is her maths grade 72"."""
    student = auth.resolve_permitted_student(
        db,
        guardian_external_id=subject.guardian_id,
        student_external_id=student_id,
        caller=subject.caller,
        endpoint=str(request.url.path),
    )
    resolved_term = _resolve_term(db, term)

    # Re-derived from the permitted binding set rather than trusted from the path, so
    # a caller cannot read an arbitrary course by guessing its id.
    binding = next(
        (b for b in _bindings_for(db, student, resolved_term) if str(b.lms_course_id) == course_id),
        None,
    )
    if binding is None or student.lms_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "No such subject for this student this term."},
        )

    adapter = _adapter_or_503()
    try:
        assignments = adapter.get_assignment_grades(
            lms_user_id=student.lms_user_id, course_id=binding.lms_course_id
        )
    except lms.LmsUnavailable:
        _raise_unavailable(db, subject.caller, request, subject.guardian_id, student_id)

    return CourseGradeDetailOut(
        student=_student_ref(student),
        term=_term_out(resolved_term),
        course=_course_grade(binding, assignments, GradingPolicy()),
        assignments=assignments,
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
    db: Session = Depends(get_db),
    subject: auth.ParentSubject = Depends(auth.require_parent_subject),
) -> AttendanceSummaryOut:
    """Attendance totals for a term, with the recent days behind them."""
    student = auth.resolve_permitted_student(
        db,
        guardian_external_id=subject.guardian_id,
        student_external_id=student_id,
        caller=subject.caller,
        endpoint=str(request.url.path),
    )
    resolved_term = _resolve_term(db, term)
    bindings = _bindings_for(db, student, resolved_term)

    days = []
    if student.lms_user_id is not None and bindings:
        adapter = _adapter_or_503()
        try:
            days = adapter.get_attendance(
                lms_user_id=student.lms_user_id,
                course_ids=[b.lms_course_id for b in bindings],
                since=resolved_term.starts_on,
            )
        except lms.LmsUnavailable:
            _raise_unavailable(db, subject.caller, request, subject.guardian_id, student_id)

    counts = {"present": 0, "absent": 0, "late": 0, "excused": 0}
    for day in days:
        if day.status in counts:
            counts[day.status] += 1

    total = sum(counts.values())
    # Excused absence counts as attended for the rate. A child off school with a
    # doctor's note has not "missed" school in the sense a parent is asking about,
    # and reporting otherwise turns a legitimate absence into an accusation.
    attended = counts["present"] + counts["late"] + counts["excused"]

    return AttendanceSummaryOut(
        student=_student_ref(student),
        term=_term_out(resolved_term),
        present_count=counts["present"],
        absent_count=counts["absent"],
        late_count=counts["late"],
        excused_count=counts["excused"],
        total_sessions=total,
        attendance_rate=round((attended / total) * 100.0, 2) if total else None,
        recent_days=sorted(days, key=lambda d: d.date, reverse=True)[:30],
        as_of=_now(),
    )


@agent_router.get(
    "/guardians/{guardian_id}/students/{student_id}/report-cards/{term_id}",
    response_model=ReportCardOut,
    responses=_AGENT_RESPONSES,
)
def get_report_card(
    guardian_id: str,
    student_id: str,
    term_id: str,
    request: Request,
    db: Session = Depends(get_db),
    subject: auth.ParentSubject = Depends(auth.require_parent_subject),
) -> ReportCardOut:
    """The published snapshot for a term. Served frozen — never recomputed."""
    student = auth.resolve_permitted_student(
        db,
        guardian_external_id=subject.guardian_id,
        student_external_id=student_id,
        caller=subject.caller,
        endpoint=str(request.url.path),
    )
    term = _resolve_term(db, term_id)

    card = (
        db.query(ReportCard)
        .filter(
            ReportCard.student_id == student.id,
            ReportCard.term_id == term.id,
            ReportCard.superseded_at.is_(None),
        )
        .order_by(ReportCard.version.desc())
        .first()
    )
    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "No published report card for this term."},
        )

    return ReportCardOut(
        student=_student_ref(student),
        term=_term_out(term),
        version=card.version,
        published_at=card.published_at,
        is_superseded=card.superseded_at is not None,
        payload=card.payload or {},
    )


def _adapter_or_503():
    try:
        return lms.get_adapter()
    except lms.LmsUnavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "lms_unavailable", "message": "System of record is unreachable."},
        )


def _raise_unavailable(db, caller, request, guardian_id, student_id):
    """Audit the failed read, then fail loudly.

    The audit row matters as much as the error: a spike of `lms_unavailable` against
    one student is how a sync problem gets noticed before a parent reports it.
    """
    auth.write_audit(
        db,
        endpoint=str(request.url.path),
        allowed=False,
        reason="lms_unavailable",
        guardian_external_id=guardian_id,
        student_external_id=student_id,
        api_key_prefix=caller.prefix,
        request_id=caller.request_id,
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "lms_unavailable", "message": "System of record is unreachable."},
    )


# ---------------------------------------------------------------------------
# Admin. Separate scope; cannot read a student record through these routes.
# ---------------------------------------------------------------------------


@admin_router.post("/guardians", status_code=status.HTTP_201_CREATED)
def create_guardian(
    body: GuardianIn,
    db: Session = Depends(get_db),
    caller: auth.Caller = Depends(auth.require_admin_key),
) -> dict:
    if db.query(Guardian).filter(Guardian.external_id == body.external_id).first():
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": "Guardian exists."})

    guardian = Guardian(**body.model_dump())
    db.add(guardian)
    db.commit()
    return {"guardian_id": guardian.external_id}


@admin_router.post("/guardians/{guardian_id}/students", status_code=status.HTTP_201_CREATED)
def link_student(
    guardian_id: str,
    body: GuardianLinkIn,
    db: Session = Depends(get_db),
    caller: auth.Caller = Depends(auth.require_admin_key),
) -> dict:
    """Link a guardian to a student, and set whether they may see records.

    Creating the link and granting records access are one call but two decisions —
    `can_view_records` defaults to False, so an import that omits it grants nothing.
    """
    guardian = db.query(Guardian).filter(Guardian.external_id == guardian_id).first()
    student = db.query(Student).filter(Student.external_id == body.student_id).first()
    if guardian is None or student is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Unknown guardian or student."})

    link = (
        db.query(GuardianStudent)
        .filter(GuardianStudent.guardian_id == guardian.id, GuardianStudent.student_id == student.id)
        .first()
    )
    if link is None:
        link = GuardianStudent(guardian_id=guardian.id, student_id=student.id)
        db.add(link)

    link.relationship_type = body.relationship_type
    link.can_view_records = body.can_view_records
    link.restriction_note = body.restriction_note
    db.commit()
    return {"guardian_id": guardian_id, "student_id": body.student_id, "can_view_records": link.can_view_records}


@admin_router.delete("/guardians/{guardian_id}/students/{student_id}")
def unlink_student(
    guardian_id: str,
    student_id: str,
    db: Session = Depends(get_db),
    caller: auth.Caller = Depends(auth.require_admin_key),
) -> dict:
    guardian = db.query(Guardian).filter(Guardian.external_id == guardian_id).first()
    student = db.query(Student).filter(Student.external_id == student_id).first()
    if guardian is None or student is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Unknown guardian or student."})

    db.query(GuardianStudent).filter(
        GuardianStudent.guardian_id == guardian.id, GuardianStudent.student_id == student.id
    ).delete()
    db.commit()
    return {"deleted": True}


@admin_router.post("/api-keys", response_model=ApiKeyOut, status_code=status.HTTP_201_CREATED)
def create_api_key(
    body: ApiKeyIn,
    db: Session = Depends(get_db),
    caller: auth.Caller = Depends(auth.require_admin_key),
) -> ApiKeyOut:
    """Mint a key. The secret in this response is the only copy that will ever exist."""
    if body.scope not in ("agent", "admin"):
        raise HTTPException(status_code=400, detail={"code": "bad_request", "message": "scope must be agent or admin."})

    raw, prefix, key_hash = auth.generate_api_key()
    expires_at = None
    if body.expires_in_days:
        from datetime import timedelta

        expires_at = _now() + timedelta(days=body.expires_in_days)

    db.add(ApiKey(prefix=prefix, key_hash=key_hash, label=body.label, scope=body.scope, expires_at=expires_at))
    db.commit()
    return ApiKeyOut(prefix=prefix, label=body.label, scope=body.scope, api_key=raw, expires_at=expires_at)


@admin_router.get("/audit", response_model=list[AuditEntryOut])
def read_audit(
    guardian_id: str | None = Query(default=None),
    student_id: str | None = Query(default=None),
    limit: int = Query(default=100, le=1000),
    db: Session = Depends(get_db),
    caller: auth.Caller = Depends(auth.require_admin_key),
) -> list[AuditEntryOut]:
    """Read the access log. Read only — there is no write or delete path anywhere."""
    query = db.query(AccessAudit)
    if guardian_id:
        query = query.filter(AccessAudit.guardian_external_id == guardian_id)
    if student_id:
        query = query.filter(AccessAudit.student_external_id == student_id)

    rows = query.order_by(AccessAudit.created_at.desc()).limit(limit).all()
    return [
        AuditEntryOut(
            guardian_id=r.guardian_external_id,
            student_id=r.student_external_id,
            endpoint=r.endpoint,
            allowed=r.allowed,
            reason=r.reason,
            api_key_prefix=r.api_key_prefix,
            request_id=r.request_id,
            created_at=r.created_at,
        )
        for r in rows
    ]
