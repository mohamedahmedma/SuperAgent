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

import logging

from records import auth, lms
from records.guardian_directory import PermittedStudent
from records.calendar import (
    CalendarUnavailable,
    SchoolTerm,
    get_calendar,
)
from records.assembler import AttendanceAssembler, GradeAssembler, term_prefix
from records.db import get_db
from records.grading import DEFAULT_POLICY
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


def _student_ref(student: Student) -> StudentRef:
    return StudentRef(
        student_id=student.external_id,
        full_name_ar=student.full_name_ar,
        full_name_en=student.full_name_en,
        grade_level=student.grade_level,
        section=student.section,
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


def _resolve_term(db: Session, term_code: str | None) -> SchoolTerm:
    """Named term, or the one we are in — asked of the school, not remembered here.

    Falling back to the current term is what lets a parent ask "how is she doing" without
    naming one. When today falls in no term — a holiday, a gap between years — the most
    recently started term wins, because "the term that just ended" is what a parent means
    in August.

    `db` is still taken and no longer read. Kept so that every caller's signature is
    unchanged while the calendar moves; removing it is a separate, mechanical edit.

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


def _term_argument(adapter: object, term: SchoolTerm) -> str:
    """Name a term the way this backend expects to be asked.

    `term_prefix` produces `"2026-T1-"`, which is a **Moodle idnumber convention**: its
    courses are called `2026-T1-G7A-MATH`, and the trailing hyphen is what stops
    `"2026-T1"` also matching `"2026-T10"`. A backend that stores marks against the
    school's own term codes wants the bare `"2026-T1"` and matches nothing at all when
    handed the prefixed form — which fails as an empty report rather than an error, so the
    parent is told their child has no marks this term.

    Keyed off the same capability as subject naming because it is the same underlying
    fact: whether this backend speaks Moodle's conventions or the school's own.
    """
    if getattr(adapter, "reports_own_subjects", False):
        return term.code
    return term_prefix(term.code)


def _local_term_id(db: Session, term_code: str) -> int | None:
    """This service's own row id for a term, or `None` when it has none.

    Terms themselves now come from the school's calendar, but two tables here still key
    on a local id: `course_bindings`, which is Moodle-mapping data this service genuinely
    owns, and `report_cards`, which are frozen snapshots it published. Both are looked up
    by the code the calendar reports rather than by an id the caller carries, so the local
    row is an implementation detail of those two features and not part of the term.
    """
    row = db.query(Term).filter(Term.code == term_code).first()
    return row.id if row is not None else None


def _bindings_for(
    db: Session, student: PermittedStudent, term: SchoolTerm
) -> list[CourseBinding]:
    """Published courses for this student's grade level and section in this term.

    Unpublished bindings are invisible here by design — a registrar entering next
    term's courses, or a teacher's sandbox, cannot reach a parent.

    Only ever reached on a backend that does *not* name its own subjects; see
    `LmsAdapter.reports_own_subjects`. A school with no local term row has no bindings
    either, which is the same empty answer by a shorter route.
    """
    term_id = _local_term_id(db, term.code)
    if term_id is None:
        return []
    return (
        db.query(CourseBinding)
        .filter(
            CourseBinding.term_id == term_id,
            CourseBinding.grade_level == student.grade_level,
            CourseBinding.section == student.section,
            CourseBinding.is_published.is_(True),
        )
        .all()
    )


def _lms_ref(student: Student) -> str:
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
    db: Session = Depends(get_db),
    caller: auth.Caller = Depends(auth.require_agent_key),
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
    adapter = _adapter_or_503()

    # A backend that names its own subjects needs no binding table — see
    # `LmsAdapter.reports_own_subjects`. SIS stores marks against the school's own subject
    # codes, so requiring bindings there would drop every subject and tell a parent their
    # child has no marks until the whole curriculum had been re-entered into this service,
    # which is supposed to hold no data of its own.
    unbound = getattr(adapter, "reports_own_subjects", False)

    # No published courses: an empty `courses` list is the honest answer, and the agent
    # renders it as "nothing recorded for her this term" rather than inventing subjects.
    #
    # Asked for BEFORE the LMS call rather than filtered afterwards. If nothing is
    # published there is nothing a parent may see, so there is no reason to ask the
    # system of record about this child at all.
    bindings = [] if unbound else _bindings_for(db, student, resolved_term)

    courses: list[CourseGrade] = []
    if bindings or unbound:
        try:
            # ONE call for the whole term, however many subjects. The per-course loop
            # this replaces was a round trip per subject, and it aggregated the results
            # itself — producing a figure that could disagree with the gradebook.
            subjects = adapter.get_subject_grades(
                student_ref=_lms_ref(student),
                term=_term_argument(adapter, resolved_term),
            )
        except lms.LmsUnavailable:
            _raise_unavailable(db, subject.caller, request, subject.guardian_id, student_id)

        assembler = GradeAssembler(DEFAULT_POLICY)
        courses = (
            assembler.assemble_unbound(subjects)
            if unbound
            else assembler.assemble(subjects, bindings)
        )

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
    db: Session = Depends(get_db),
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
        db,
        guardian_external_id=subject.guardian_id,
        student_external_id=student_id,
        caller=subject.caller,
        endpoint=str(request.url.path),
    )
    resolved_term = _resolve_term(db, term)
    adapter = _adapter_or_503()
    unbound = getattr(adapter, "reports_own_subjects", False)

    # Re-derived from what this student may actually see rather than trusted from the
    # path, so a caller cannot read an arbitrary course by guessing its id. On the
    # unbound path the guard is the same in substance: the subject has to appear in the
    # set the system of record returns for *this* child in *this* term, so an id that is
    # not one of hers matches nothing below.
    binding = None
    if not unbound:
        binding = next(
            (
                b
                for b in _bindings_for(db, student, resolved_term)
                if str(b.lms_course_id) == course_id
            ),
            None,
        )
        if binding is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "not_found",
                    "message": "No such subject for this student this term.",
                },
            )

    try:
        subjects = adapter.get_subject_grades(
            student_ref=_lms_ref(student),
            term=_term_argument(adapter, resolved_term),
        )
    except lms.LmsUnavailable:
        _raise_unavailable(db, subject.caller, request, subject.guardian_id, student_id)

    assembler = GradeAssembler(DEFAULT_POLICY)
    if unbound:
        # `assemble_unbound` sets `course_id` to the school's own subject code, which is
        # what this route's path segment carries on this backend — so the filter here is
        # the same identity check the binding lookup performs on the Moodle path.
        courses = [
            course
            for course in assembler.assemble_unbound(subjects)
            if course.course_id == course_id
        ]
    else:
        courses = assembler.assemble(subjects, [binding])
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
    adapter = _adapter_or_503()
    # See the grades route: a backend that names its own subjects needs no binding table,
    # and gating the call behind one reports a term of registers as no attendance at all.
    unbound = getattr(adapter, "reports_own_subjects", False)
    bindings = [] if unbound else _bindings_for(db, student, resolved_term)

    subjects: list = []
    if bindings or unbound:
        try:
            # ONE call for the term. The shape this replaces walked every session in
            # every course, and the system of record handed back the whole class each
            # time — so the facade received attendance for children this guardian may
            # not see. That is now impossible rather than filtered.
            subjects = adapter.get_subject_attendance(
                student_ref=_lms_ref(student),
                term=_term_argument(adapter, resolved_term),
            )
        except lms.LmsUnavailable:
            _raise_unavailable(db, subject.caller, request, subject.guardian_id, student_id)

    assembler = AttendanceAssembler(bindings, unbound=unbound)
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
            ReportCard.term_id == _local_term_id(db, term.code),
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

_MOVED_TO_SIS = {
    "code": "moved",
    "message": (
        "Guardians are managed in the school's SIS. Upload them there "
        "(POST /v1/imports/guardians/preview) and change records access with "
        "PATCH /v1/students/{student_number}/guardians/{phone}."
    ),
}


@admin_router.post("/guardians", status_code=status.HTTP_410_GONE)
def create_guardian_moved(
    caller: auth.Caller = Depends(auth.require_admin_key),
) -> dict:
    """Gone. Guardians are created by the SIS spreadsheet import."""
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_MOVED_TO_SIS)


@admin_router.post("/guardians/{guardian_id}/students", status_code=status.HTTP_410_GONE)
def link_student_moved(
    guardian_id: str, caller: auth.Caller = Depends(auth.require_admin_key)
) -> dict:
    """Gone. Links are created by the SIS import and amended on the SIS access route."""
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_MOVED_TO_SIS)


@admin_router.delete(
    "/guardians/{guardian_id}/students/{student_id}", status_code=status.HTTP_410_GONE
)
def unlink_student_moved(
    guardian_id: str,
    student_id: str,
    caller: auth.Caller = Depends(auth.require_admin_key),
) -> dict:
    """Gone. Unlinking is a SIS operation."""
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_MOVED_TO_SIS)


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
