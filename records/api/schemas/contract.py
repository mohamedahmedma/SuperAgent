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
    gender: str = Field(
        default="unspecified",
        description=(
            "The child's sex as the system of record states it: male, female, or "
            "unspecified. `unspecified` means the school has not recorded one — it is "
            "not a default, and a caller must not let it satisfy a filter for either sex."
        ),
    )


class TermOut(BaseModel):
    term_id: str = Field(description="Stable term code, e.g. '2026-T1'.")
    name_ar: str = ""
    name_en: str = ""
    academic_year: str = ""
    starts_on: datetime
    ends_on: datetime
    is_closed: bool = Field(
        description="Whether the school has closed this term to further marking."
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


class AcademicGrade(BaseModel):
    """The subject with non-assessment activities removed.

    Separate from the headline figure rather than replacing it, because a school that
    grades attendance produces two true answers to "how is she doing in maths" and they
    are not interchangeable. Measured on a real instance: 65% official, 80% academic,
    same child, same subject.

    `percentage` is None whenever it could not be derived EXACTLY — never an estimate.
    A consumer that ignores a caveat flag puts the estimate in front of a parent, so
    there is no estimate to ignore.
    """

    percentage: float | None = None
    letter_grade: str = ""
    passed: bool | None = None
    unavailable: str = Field(
        default="",
        description=(
            "Empty when `percentage` is present. Otherwise why not: "
            "'aggregation_not_summable' (the course uses a weighted or drop-lowest "
            "scheme that cannot be re-derived — read `categories` instead), "
            "'no_assessment_items', 'nothing_gradeable'."
        ),
    )


class GradeCategory(BaseModel):
    """A gradebook category subtotal, computed by the system of record.

    Exact under every aggregation scheme, because the LMS applied that scheme itself.
    This is the dependable route to a partial subject grade when `academic` could not
    be derived.
    """

    name: str = ""
    percentage: float | None = None


class CourseGrade(BaseModel):
    """One subject for one term — the answer to "how is my child doing in maths".

    Every figure here was computed by the system of record. This service classifies
    (letter, pass/fail) but does not calculate, because a second implementation of the
    gradebook's arithmetic is a second answer waiting to contradict the first.

    `excused_count` is surfaced rather than hidden so a figure can be explained: a
    parent comparing two children deserves to know one had three assessments excused.
    """

    course_id: str
    subject_code: str = ""
    subject_name_ar: str = ""
    subject_name_en: str = ""
    teacher_name: str = ""

    computed_percentage: float | None = Field(
        default=None,
        description=(
            "The school's OFFICIAL course total, exactly as the gradebook has it — "
            "including attendance and any other graded activity. Null when nothing is "
            "gradeable yet, which is not the same as zero."
        ),
    )
    letter_grade: str = ""
    passed: bool | None = None

    academic: AcademicGrade = Field(
        default_factory=AcademicGrade,
        description="The same subject counting assessments only.",
    )
    categories: list[GradeCategory] = Field(
        default_factory=list,
        description="The system of record's own category subtotals.",
    )

    graded_count: int = 0
    excused_count: int = 0
    missing_count: int = 0
    pending_count: int = 0

    # False when some assessments in the term are not yet graded — the difference
    # between "this is the final figure" and "this is the figure so far".
    is_complete: bool = False


class StudentGradesOut(BaseModel):
    """Every subject for one student in one term."""

    student: StudentRef
    term: TermOut
    courses: list[CourseGrade] = []
    primary_figure: str = Field(
        default="academic",
        description=(
            "Which figure this school wants led with: 'academic' (assessments only) or "
            "'official' (the gradebook total, including attendance). BOTH are always "
            "present on every course — this only says which to say first when one "
            "number is wanted. Set per deployment via RECORDS_PRIMARY_GRADE."
        ),
    )
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


class TimetableStatusOut(str, Enum):
    """Why a week is or is not a grid of lessons.

    On the payload rather than left to be inferred from an empty list, for the reason
    `SubmissionStatus` exists: a model handed emptiness narrates a plausible reason for it,
    and the plausible reason here — "she has no lessons" — is false in two of the three
    cases.
    """

    #: She has a class and it has lessons in it.
    OK = "ok"
    #: No placement covered this term. She had left, or had not yet joined; a child
    #: enrolled for next September has no class today. Nothing is missing.
    NO_CLASS = "no_class"
    #: She has a class and nobody has laid out its week for this term. A fact about the
    #: school's admin, and never to be reported as the child having no lessons.
    NO_TIMETABLE = "no_timetable"


class TimetablePeriodOut(BaseModel):
    """One slot in the school's day — a row of the grid, breaks included."""

    period_number: int = Field(description="Its position in the day. 1 is the first.")
    name_ar: str = ""
    name_en: str = ""
    starts_at: str = Field(
        default="",
        description=(
            "`HH:MM`, or empty when the school has not fixed this boundary — a school "
            "settles how many periods it runs long before it agrees when each one rings. "
            "Never a placeholder: an invented 08:00 cannot afterwards be told from an "
            "agreed one."
        ),
    )
    ends_at: str = ""
    is_teaching: bool = Field(
        default=True,
        description=(
            "`false` for break, assembly or prayer — a slot the day contains and no class "
            "timetables a lesson into. A client drawing the day must show it; a caller "
            "counting lessons must not."
        ),
    )


class TimetableLessonOut(BaseModel):
    """One lesson, in one slot of one day."""

    day_of_week: str = Field(description="Lowercase, e.g. 'sunday'.")
    period_number: int
    subject_code: str = Field(
        default="",
        description=(
            "Empty for a stated free period — a slot the class deliberately has off, which "
            "is different from a slot nobody has planned (no row at all)."
        ),
    )
    subject_name_ar: str = ""
    subject_name_en: str = ""


class TimetableOut(BaseModel):
    """One child's week: the class she sat in for the term, and what that class does.

    The grid and the lessons travel together because read apart they can disagree, and the
    symptom is a client drawing a seven-row day against an eight-period grid.

    `class_code` is the class she was in **for this term**, not her current one. A child who
    moved 3A -> 3B in March still reads 3A for Term 1, because that is the week she actually
    sat.
    """

    student: StudentRef
    term: TermOut
    status: TimetableStatusOut = Field(
        description="Which of the three answers this is. Branch on this, never on whether "
        "`lessons` is empty."
    )
    class_code: str = Field(
        default="",
        description="Empty exactly when `status` is `no_class`.",
    )
    class_name_ar: str = ""
    class_name_en: str = ""
    days: list[str] = Field(
        default_factory=list,
        description=(
            "The school's working days, **in the school's own order**. Do not sort this: "
            "the week begins on Saturday at some schools and Sunday at others, and only "
            "the school knows which."
        ),
    )
    periods: list[TimetablePeriodOut] = Field(default_factory=list)
    lessons: list[TimetableLessonOut] = Field(
        default_factory=list,
        description="Ordered by day — in the school's week order — then by period.",
    )
    teaching_slots: int = Field(
        default=0,
        description=(
            "How many slots this week could hold a lesson: teaching periods times open "
            "days. The denominator for 'how full is this timetable'."
        ),
    )
    as_of: datetime


class ClassroomStatusOut(str, Enum):
    """Whether there is a room to talk about at all.

    Two members, and the other two ways these answers can be empty are deliberately NOT
    here: a room with no subject board and a room with no staffing are properties of those
    lists, because a room can lack either, both or neither and one enum cannot say so.

    What must never be inferred is this one. An empty `subjects` beside `no_class` means
    "there is no room to ask about"; beside `ok` it means "nobody has curated the board".
    A consumer that read emptiness alone would tell a parent her child studies nothing.
    """

    OK = "ok"
    #: No placement covered this term. She had left, or had not yet joined; a child
    #: enrolled for next September has no class today. Nothing is missing.
    NO_CLASS = "no_class"


class StudentClassOut(BaseModel):
    """Which class a child is in — the answer to "what class is my son in".

    `class_name_ar` / `class_name_en` are what the school itself calls the room: "Primary 1
    Class 1", "3/1". **That is the answer to give a parent.** `class_code` is the school's
    internal key for the same room ("3A", "P1-01") and is carried for correlation, not for
    reading out to a family.

    The class is the one she was in **for this term**, not her current one: a child who
    moved rooms in March still reads her autumn room for Term 1.
    """

    student: StudentRef
    term: TermOut
    status: ClassroomStatusOut
    class_code: str = Field(
        default="", description="Empty exactly when `status` is `no_class`."
    )
    class_name_ar: str = ""
    class_name_en: str = ""
    year_level_code: str = Field(
        default="", description="The rung the room sits on, e.g. `AR-P4`."
    )
    year_level_name_ar: str = Field(
        default="",
        description="The rung's human name — 'Year 3'. Empty when the school's ladder is "
        "missing the rung its own class points at, which is broken data rather than a "
        "normal state; the class name is still returned.",
    )
    year_level_name_en: str = ""
    as_of: datetime


class StudySubjectOut(BaseModel):
    """One subject the child is taught."""

    code: str = Field(description="The school's own subject code, e.g. `MATH`.")
    name_ar: str = ""
    name_en: str = ""


class StudentSubjectsOut(BaseModel):
    """What a child studies — the subjects assigned to her rung, in the school's order.

    The rung's assignment board, not the year's whole catalogue: a school teaches Physics
    but only Secondary sits it. Retired subjects are excluded, because a dropped subject is
    not one she studies — though it still resolves for marks already stated against it.

    An empty list with `status: ok` means nobody has curated this rung's board for the year.
    That is a fact about the school's admin and must not be reported as the child studying
    nothing.
    """

    student: StudentRef
    term: TermOut
    status: ClassroomStatusOut
    class_code: str = ""
    class_name_ar: str = ""
    class_name_en: str = ""
    subjects: list[StudySubjectOut] = Field(
        default_factory=list,
        description="In the school's own display order. Do not re-sort: alphabetical "
        "ordering differs between the two scripts, so the same board would read in two "
        "different orders.",
    )
    as_of: datetime


class ClassTeacherOut(BaseModel):
    """One teacher in the child's class, and the subject they teach her.

    Carries a name and a subject and nothing else — no staff number, no email, no phone, no
    username. A parent needs to know who teaches their child, not how to reach a member of
    staff directly, and the school's own contact routes are the school's to publish. The
    shape is the privacy boundary rather than a filter a consumer has to apply.
    """

    full_name_ar: str = Field(
        default="",
        description="Empty when the school recorded only one spelling; the other is then "
        "populated.",
    )
    full_name_en: str = ""
    subject_code: str = ""
    subject_name_ar: str = ""
    subject_name_en: str = ""


class StudentTeachersOut(BaseModel):
    """Who teaches a child, and what each of them teaches her.

    **One entry per (teacher, subject).** A teacher taking two subjects appears twice, and
    a subject taught by two teachers has two entries — the system of record permits both, so
    a consumer must render a list and never assume one name per subject. Taking the first
    would hide a co-teacher from the parent who asked.

    **Present tense, and this is the one caveat worth reading.** `term` resolves which ROOM
    she sat in; the school's assignment table carries no term, so `teachers` is who teaches
    that room *now*. It is not who taught it last November, and must not be described as
    such.

    An empty list with `status: ok` means nobody has entered the room's staffing — about
    the school, not the child.
    """

    student: StudentRef
    term: TermOut
    status: ClassroomStatusOut
    class_code: str = ""
    class_name_ar: str = ""
    class_name_en: str = ""
    teachers: list[ClassTeacherOut] = Field(
        default_factory=list,
        description="Ordered by the school's subject display order, then subject, then "
        "teacher. Teachers who have left the school are excluded.",
    )
    as_of: datetime


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
#
# `GuardianIn` and `GuardianLinkIn` used to live here. They described writes this service
# no longer accepts: guardians and their custody flags are the registrar's, entered in
# `sis/`, and the routes that took these shapes answer 410 naming where they went. The
# schemas outlived the routes by exactly as long as nobody looked.
# ---------------------------------------------------------------------------

# `ApiKeyIn`, `ApiKeyOut` and `AuditEntryOut` used to be here. Their routes are gone with
# the tables behind them: this service mints no credentials — its own is one secret in the
# environment, see `records.auth` — and the access audit is kept by `sis/`, where the
# decision it records is actually made.
