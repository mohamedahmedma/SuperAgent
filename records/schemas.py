"""The contract — the only thing the agent is allowed to depend on.

These shapes are the reason the facade exists. Behind them Moodle can be swapped for
an SIS, or for the school's own system, without the agent's tool layer changing a
line. So they are written in the school's vocabulary — subject, term, attendance —
never in the LMS's.

Two conventions run through everything here, both of them defences against a language
model reading a record wrong:

**Nothing is silently absent.** A missing grade carries a `status` saying *why* it is
missing. A model handed `null` will narrate a plausible reason; a model handed
`"excused"` will not.

**Every payload is stamped.** `as_of` tells the agent how fresh the figure is, so
"your child's grade is 84" can be said with the honest qualifier attached.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SubmissionStatus(str, Enum):
    """Why a grade is or is not a number.

    `EXCUSED` is the one that matters and the one most systems get wrong. An excused
    assignment leaves the denominator entirely — it is not a zero, and treating it as
    one silently lowers a real child's real grade. It is a distinct member here so the
    aggregation cannot collapse it by accident.
    """

    GRADED = "graded"
    EXCUSED = "excused"
    SUBMITTED_UNGRADED = "submitted_ungraded"
    MISSING = "missing"
    NOT_DUE = "not_due"


class StudentRef(BaseModel):
    """A student as a parent would recognise them. No LMS ids leak across the seam."""

    student_id: str = Field(description="The school's student number.")
    full_name_ar: str = ""
    full_name_en: str = ""
    grade_level: str = ""
    section: str = ""


class TermOut(BaseModel):
    term_id: str = Field(description="Stable term code, e.g. '2026-T1'.")
    name_ar: str = ""
    name_en: str = ""
    academic_year: str = ""
    starts_on: datetime
    ends_on: datetime
    is_closed: bool = Field(
        description="Closed terms are served from frozen report cards, not recomputed."
    )
    is_current: bool = False


class AssignmentGrade(BaseModel):
    """One assignment, as the parent sees it."""

    assignment_id: str
    title: str = ""
    due_date: datetime | None = None
    status: SubmissionStatus

    # None whenever status is not GRADED. Pairing them means the agent never has to
    # infer meaning from a bare null.
    score: float | None = None
    max_score: float | None = None
    percentage: float | None = None

    # Which weighted category this assignment counts toward, e.g. "homework".
    category: str = ""
    graded_at: datetime | None = None


class CourseGrade(BaseModel):
    """One subject's rollup for one term — the answer to "how is my child doing in maths".

    `computed_percentage` is calculated by this service, not by Moodle, because Moodle
    has no term concept to roll up to. `excused_count` is surfaced rather than hidden
    so the figure can be explained: a parent comparing two children's grades deserves
    to know one of them had three assignments excused.
    """

    course_id: str
    subject_code: str = ""
    subject_name_ar: str = ""
    subject_name_en: str = ""
    teacher_name: str = ""

    computed_percentage: float | None = None
    letter_grade: str = ""
    passed: bool | None = None

    graded_count: int = 0
    excused_count: int = 0
    missing_count: int = 0
    pending_count: int = 0

    # False when some assignments in the term are not yet graded — the difference
    # between "this is the final figure" and "this is the figure so far".
    is_complete: bool = False


class StudentGradesOut(BaseModel):
    """Every subject for one student in one term."""

    student: StudentRef
    term: TermOut
    courses: list[CourseGrade] = []
    as_of: datetime


class CourseGradeDetailOut(BaseModel):
    """The assignment-level breakdown behind one subject's figure."""

    student: StudentRef
    term: TermOut
    course: CourseGrade
    assignments: list[AssignmentGrade] = []
    as_of: datetime


class AttendanceDay(BaseModel):
    date: datetime
    status: str = Field(description="present | absent | late | excused")
    course_id: str = ""
    note: str = ""


class AttendanceSummaryOut(BaseModel):
    """Attendance for one student over a window.

    Counts first, days second. The agent answers "how many days has she missed" far
    more often than "list them", and a summary that forces it to count a list is a
    summary that will eventually be counted wrong.
    """

    student: StudentRef
    term: TermOut
    present_count: int = 0
    absent_count: int = 0
    late_count: int = 0
    excused_count: int = 0
    total_sessions: int = 0
    attendance_rate: float | None = None
    recent_days: list[AttendanceDay] = []
    as_of: datetime


class ReportCardOut(BaseModel):
    """A published snapshot. Never recomputed on read."""

    student: StudentRef
    term: TermOut
    version: int
    published_at: datetime
    is_superseded: bool = False
    payload: dict = Field(
        default_factory=dict,
        description="The frozen figures exactly as published.",
    )


class StudentListOut(BaseModel):
    """The students one guardian is permitted to ask about.

    An empty list is a legitimate, non-error answer — it is what a guardian with no
    visible children gets, and it is deliberately indistinguishable from a guardian
    whose children are all restricted. Distinguishing them would let a caller probe
    for the existence of a restriction.
    """

    guardian_id: str
    students: list[StudentRef] = []


class ErrorOut(BaseModel):
    """Machine-readable failure.

    `code` exists so the agent branches on a value rather than on prose. The one that
    matters most is `lms_unavailable`: it must produce "I cannot reach the school
    records right now", never an answer the model invented to be helpful.
    """

    code: str = Field(
        description=(
            "not_authorized | unknown_student | unknown_term | "
            "records_restricted | lms_unavailable | not_found"
        )
    )
    message: str = ""


# ---------------------------------------------------------------------------
# Admin-side write shapes. Separate scope, separate key, never reachable with an
# agent key — see records.auth.
# ---------------------------------------------------------------------------


class GuardianIn(BaseModel):
    external_id: str
    full_name_ar: str = ""
    full_name_en: str = ""
    phone: str = ""
    email: str = ""
    preferred_language: str = "ar"


class GuardianLinkIn(BaseModel):
    """Linking a guardian to a student.

    `can_view_records` defaults to False so that creating the link is not, by itself,
    a grant. Records access is an explicit act with a reason attached.
    """

    student_id: str
    relationship_type: str = "parent"
    can_view_records: bool = False
    restriction_note: str = ""


class ApiKeyIn(BaseModel):
    label: str
    scope: str = Field(default="agent", description="agent | admin")
    expires_in_days: int | None = None


class ApiKeyOut(BaseModel):
    """The only response that ever contains the secret, and only at creation."""

    prefix: str
    label: str
    scope: str
    api_key: str | None = Field(
        default=None,
        description="Full secret. Shown once, at creation, and never recoverable.",
    )
    expires_at: datetime | None = None


class AuditEntryOut(BaseModel):
    guardian_id: str = ""
    student_id: str = ""
    endpoint: str = ""
    allowed: bool = False
    reason: str = ""
    api_key_prefix: str = ""
    request_id: str = ""
    created_at: datetime
