"""The weekly plan over HTTP: a school's periods, and one lesson per class per slot.

Its own router rather than more routes on `structure`, because a timetable is asked about
differently from the ladder it hangs on. The structure routes answer "what does this school
consist of" and are read once a term by a registrar setting up; these answer "what is 3A
doing on Tuesday" and are read by whoever is standing in front of 3A.

Four shapes of route, and the middle one is the whole feature:

  `/schools/{code}/timetable-periods`   the school's day — how many periods, when they ring
  `/timetable`                          lessons: read a week, place lessons, clear slots
  `/timetable/week`                     one class's week drawn against the school's own grid
  `/guardians/by-id/.../timetable`      the same week, reached from a child by her parent

**The last one exists because a parent has no class code.** A timetable is a statement about
a room, and the only route into it a family has is the child placed in that room. So that
route takes a student number and resolves the class itself, in one transaction — the same
shape the guardian-scoped grades and attendance routes take, and for the same reason: the
caller is a chat service running a language model over text a stranger can write, so the
link is re-checked here rather than trusted from whatever it was handed.

**Writes are whole-batch and all-or-nothing.** `PUT /timetable` takes every lesson a
registrar has laid out and either applies all of them or refuses the lot. A partially
applied week is worse than an empty one because it looks finished, and the rules that can
refuse it — a day the school does not open, a subject that rung is not assigned — are
exactly the rules a person gets wrong while typing quickly.

**Two refusals are deliberately different statuses.** A slot clash is 409: the request was
well formed and the stored state forbids it. A Friday at a school that shuts on Friday is
422: that is not a thing the request could ever have meant. A registrar has to be able to
tell "you already put something there" from "that is not a day".

Attendance is not touched anywhere in this module. A timetable is a plan; the register is a
record of what happened, and connecting them is not this stage.
"""
from datetime import time
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from sis.api.deps import (
    Principal,
    RequestId,
    get_query_service,
    get_timetable_service,
    require_permission,
)
from sis.domain.rbac import Permission, RoleCode
from sis.api.routers import domain_errors, error_responses
from sis.application.services import (
    QueryService,
    StudentWeek,
    TimetableService,
    WeekPlan,
)
from sis.domain.timetable import (
    MAX_PERIODS_PER_DAY,
    TimetableEntry,
    TimetablePeriod,
    TimetableSlot,
)
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    SchoolCode,
    StudentNumber,
    TermCode,
    YearCode,
)

router = APIRouter(prefix="/v1", tags=["timetable"])

Reader = Annotated[Principal, Depends(require_permission(Permission.TIMETABLE_READ))]
Registrar = Annotated[Principal, Depends(require_permission(Permission.TIMETABLE_WRITE))]
Timetables = Annotated[TimetableService, Depends(get_timetable_service)]
Queries = Annotated[QueryService, Depends(get_query_service)]


# -- Shapes -----------------------------------------------------------------

#: Said once, so the request and the response cannot describe the same optionality
#: differently. The same sentence term dates carry, one level down.
_TIME_NOTE = (
    "Optional. `null` means the school has not fixed this boundary yet — a school settles "
    "how many periods it runs long before it agrees when each one rings. Never a "
    "placeholder: an invented 08:00 is indistinguishable afterwards from an agreed one."
)


class TimetablePeriodIn(BaseModel):
    """One slot in the school's day."""

    period_number: int = Field(
        ge=1,
        le=MAX_PERIODS_PER_DAY,
        description="Its position in the day. 1 is the first period.",
        examples=[1],
    )
    name_en: str = Field(default="", description="English label. Used for breaks.")
    name_ar: str = Field(default="", description="Arabic label. Used for breaks.")
    starts_at: time | None = Field(default=None, description=f"When it begins. {_TIME_NOTE}")
    ends_at: time | None = Field(default=None, description=f"When it ends. {_TIME_NOTE}")
    is_teaching: bool = Field(
        default=True,
        description="`false` for break, assembly or prayer — a slot the day contains and "
        "no class timetables a lesson into. It carries no other meaning.",
    )


class TimetablePeriodsIn(BaseModel):
    """The school's whole day, replaced in one call.

    Whole-grid rather than one period at a time, because "we run seven periods, not eight"
    is a single decision. Sending it as an upsert plus a guessed delete is how period 8
    survives at some schools and not others.
    """

    periods: list[TimetablePeriodIn] = Field(
        description="Every period the day has, in any order. Sending an empty list clears "
        "the day, which is refused while lessons are timetabled into it."
    )


class TimetablePeriodOut(BaseModel):
    school_code: str
    period_number: int
    name_en: str
    name_ar: str
    starts_at: time | None = Field(default=None, description=_TIME_NOTE)
    ends_at: time | None = Field(default=None, description=_TIME_NOTE)
    is_teaching: bool
    is_timed: bool = Field(
        description="Whether both ends are on file. The supported way to ask — one time "
        "alone is not a range, and a client testing `starts_at` would call a half-filled "
        "period timed."
    )

    @classmethod
    def of(cls, period: TimetablePeriod) -> "TimetablePeriodOut":
        return cls(
            school_code=str(period.school_code),
            period_number=period.period_number,
            name_en=period.name_en,
            name_ar=period.name_ar,
            starts_at=period.starts_at,
            ends_at=period.ends_at,
            is_teaching=period.is_teaching,
            is_timed=period.is_timed,
        )


class TimetableEntryIn(BaseModel):
    """One lesson, in one slot."""

    class_code: str = Field(examples=["3A"])
    term_code: str = Field(
        examples=["2026-T1"],
        description="A timetable is a statement about a stretch of the year, and the term "
        "is that stretch. Part of the slot's identity, so re-planning between terms does "
        "not overwrite what the last one did.",
    )
    day_of_week: str = Field(
        examples=["sunday"],
        description="Must be one of the school's own working days. A school that shuts on "
        "Friday refuses a Friday lesson rather than rendering it off the edge of the grid.",
    )
    period_number: int = Field(ge=1, le=MAX_PERIODS_PER_DAY, examples=[2])
    subject_code: str | None = Field(
        default=None,
        examples=["MATH"],
        description="`null` states a free period — a slot the class deliberately has off, "
        "which is different from a slot nobody has planned (no row at all).\n\n"
        "When set, the subject must be one assigned to this class's grade. That is stage "
        "5's rule arriving here, and it is also what keeps the Arabic and Languages "
        "sections apart: they are different rungs with different assignments.",
    )


class TimetableEntriesIn(BaseModel):
    """Every lesson to place, applied as one transaction."""

    academic_year_code: str = Field(examples=["2025-2026"])
    entries: list[TimetableEntryIn]


class TimetableSlotIn(BaseModel):
    """One slot to empty."""

    class_code: str = Field(examples=["3A"])
    term_code: str = Field(examples=["2026-T1"])
    day_of_week: str = Field(examples=["sunday"])
    period_number: int = Field(ge=1, le=MAX_PERIODS_PER_DAY, examples=[2])


class TimetableSlotsIn(BaseModel):
    academic_year_code: str = Field(examples=["2025-2026"])
    slots: list[TimetableSlotIn]


class TimetableChangesIn(BaseModel):
    """The additions, edits and removals made before the user pressed Save."""

    academic_year_code: str = Field(examples=["2025-2026"])
    entries: list[TimetableEntryIn] = Field(default_factory=list)
    clear_slots: list[TimetableSlotIn] = Field(default_factory=list)


class TimetableTermCopyIn(BaseModel):
    """Copy one class's saved timetable into another term of the same year."""

    academic_year_code: str = Field(examples=["2025-2026"])
    class_code: str = Field(examples=["3A"])
    source_term_code: str = Field(examples=["2025-2026-T1"])
    target_term_code: str = Field(examples=["2025-2026-T2"])

class GradeBreakIn(BaseModel):
    academic_year_code: str
    year_level_code: str
    period_number: int | None = Field(default=None, ge=1, le=MAX_PERIODS_PER_DAY)
    break_duration_minutes: int = Field(default=0, ge=0, le=180)


class TimetableEntryOut(BaseModel):
    academic_year_code: str
    class_code: str
    term_code: str
    day_of_week: str
    period_number: int
    subject_code: str | None = Field(
        default=None, description="`null` is a stated free period. See `TimetableEntryIn`."
    )
    teacher_staff_number: str | None = Field(
        default=None,
        description="Who is timetabled to take this lesson, when anything has said so — "
        "which over HTTP is never. `PUT /timetable` has no input for it and writes `null`, "
        "so on any school set up through this API the column is empty; a demo loader does "
        "populate it, which is why the field is read back rather than hardcoded here.\n\n"
        "**This is not the way to find out who teaches a class.** Staffing lives in the "
        "teacher assignments a principal and a year supervisor write — see "
        "`/guardians/by-id/.../classroom` and the teacher routes — and a client reading it "
        "from here would report every lesson as unstaffed.",
    )

    @classmethod
    def of(cls, entry: TimetableEntry) -> "TimetableEntryOut":
        return cls(
            academic_year_code=str(entry.academic_year_code),
            class_code=str(entry.slot.class_code),
            term_code=str(entry.slot.term_code),
            day_of_week=str(entry.slot.day_of_week),
            period_number=entry.slot.period_number,
            subject_code=None if entry.subject_code is None else str(entry.subject_code),
            teacher_staff_number=entry.teacher_staff_number,
        )


class WeekPlanOut(BaseModel):
    """One class's week: the grid, the school's own days, and what is in the slots."""

    academic_year_code: str
    class_code: str
    term_code: str
    days: list[str] = Field(
        description="The school's working days, **in the school's own order**. Do not sort "
        "this: the week begins on Saturday at some schools and Sunday at others, and only "
        "the school knows which."
    )
    periods: list[TimetablePeriodOut] = Field(
        description="The rows of the grid, in period order. Read together with the lessons "
        "so a screen cannot draw a seven-row grid against an eight-period day."
    )
    entries: list[TimetableEntryOut] = Field(
        description="Ordered by day — in the school's week order — then by period."
    )
    teaching_slots: int = Field(
        description="How many slots this week could hold a lesson: teaching periods times "
        "open days. The denominator for 'how full is this timetable'."
    )

    @classmethod
    def of(cls, plan: WeekPlan) -> "WeekPlanOut":
        return cls(
            academic_year_code=plan.academic_year_code,
            class_code=plan.class_code,
            term_code=plan.term_code,
            days=[str(day) for day in plan.days],
            periods=[TimetablePeriodOut.of(period) for period in plan.periods],
            entries=[TimetableEntryOut.of(entry) for entry in plan.entries],
            teaching_slots=plan.teaching_slots,
        )


class StudentLessonOut(BaseModel):
    """One lesson on a child's week, named rather than coded.

    Leaner than `TimetableEntryOut` on purpose: the class, term and year are stated once on
    the envelope, so repeating them on all thirty-five rows would be payload a parent-facing
    client pays for on every question and reads never.

    It carries the subject's **names** because this is the one timetable route whose reader
    is a person rather than a registrar's screen. `MATH` is the school's own key and means
    nothing to a family; the names are what a parent recognises.
    """

    day_of_week: str
    period_number: int
    subject_code: str | None = Field(
        default=None,
        description="`null` is a stated free period — the class deliberately has this slot "
        "off. A slot nobody has planned has no row here at all.",
    )
    subject_name_ar: str = Field(
        default="",
        description="Empty when the subject row could not be loaded, or when the slot is a "
        "stated free period. The lesson is still listed: a week that quietly drops a row is "
        "one nobody can tell is incomplete.",
    )
    subject_name_en: str = ""


class StudentWeekOut(BaseModel):
    """One child's week: the class she sat in for the term, and what that class does.

    Two `null` states, and they are not the same answer:

    * `class_code: null` — no placement covered this term. She had left, or had not yet
      joined. `lessons` is then empty because there is no room to ask about.
    * `class_code` set with `lessons: []` — she has a class and nobody has laid out its
      grid yet. That is a fact about the school, not about the child.

    A client that renders both as "no timetable" tells a parent something false in the
    second case, which is why the codes are on the payload rather than inferred from
    emptiness.
    """

    student_number: str
    term_code: str
    academic_year_code: str | None = Field(
        default=None, description="`null` when no placement covered the term."
    )
    class_code: str | None = Field(
        default=None,
        description="The class she sat in **for this term**, not her current one — a child "
        "who moved 3A->3B in March still reads 3A for Term 1, because that is the week she "
        "actually sat. `null` means no placement covered the term.",
    )
    class_name_ar: str | None = None
    class_name_en: str | None = None
    days: list[str] = Field(
        default_factory=list,
        description="The school's working days, **in the school's own order**. Do not sort "
        "this: the week begins on Saturday at some schools and Sunday at others.",
    )
    periods: list[TimetablePeriodOut] = Field(
        default_factory=list,
        description="The rows of the grid, in period order, breaks included. Read together "
        "with the lessons so a client cannot draw a seven-row grid against an eight-period "
        "day.",
    )
    lessons: list[StudentLessonOut] = Field(
        default_factory=list,
        description="Ordered by day — in the school's week order — then by period.",
    )
    teaching_slots: int = Field(
        default=0,
        description="How many slots this week could hold a lesson: teaching periods times "
        "open days. The denominator for 'how full is this timetable'.",
    )

    @classmethod
    def of(cls, week: StudentWeek) -> "StudentWeekOut":
        section = week.class_section
        plan = week.plan
        if plan is None or section is None:
            return cls(
                student_number=week.student_number, term_code=week.term_code
            )
        return cls(
            student_number=week.student_number,
            term_code=week.term_code,
            academic_year_code=plan.academic_year_code,
            class_code=str(section.code),
            class_name_ar=section.name_ar,
            class_name_en=section.name_en,
            days=[str(day) for day in plan.days],
            periods=[TimetablePeriodOut.of(period) for period in plan.periods],
            lessons=[
                StudentLessonOut(
                    day_of_week=str(entry.slot.day_of_week),
                    period_number=entry.slot.period_number,
                    subject_code=(
                        None if entry.subject_code is None else str(entry.subject_code)
                    ),
                    subject_name_ar=(
                        week.subjects[str(entry.subject_code)].name_ar
                        if str(entry.subject_code) in week.subjects
                        else ""
                    ),
                    subject_name_en=(
                        week.subjects[str(entry.subject_code)].name_en
                        if str(entry.subject_code) in week.subjects
                        else ""
                    ),
                )
                for entry in plan.entries
            ],
            teaching_slots=plan.teaching_slots,
        )


class ClearedOut(BaseModel):
    removed: int = Field(description="How many lessons were actually removed.")


# -- The school's day -------------------------------------------------------


@router.get(
    "/schools/{school_code}/timetable-periods",
    response_model=list[TimetablePeriodOut],
    summary="The school's day, in period order",
    description="How many periods the school runs and, where it has decided, when each one "
    "rings. Empty until a school lays one out — which is a real answer, not a 404: a school "
    "exists before its bell schedule does.\n\n"
    "Held per school rather than per class because the bell is: second period starts at the "
    "same moment in 3A and in 5B.",
    responses=error_responses(401, 403, 404, 422),
)
def list_timetable_periods(
    school_code: str, timetables: Timetables, caller: Reader
) -> list[TimetablePeriodOut]:
    with domain_errors():
        periods = timetables.list_periods(SchoolCode(school_code))
    return [TimetablePeriodOut.of(period) for period in periods]


@router.put(
    "/schools/{school_code}/timetable-periods",
    response_model=list[TimetablePeriodOut],
    summary="Replace the school's day",
    description="Sets the whole grid at once and answers with it as stored. Idempotent: "
    "sending the same day twice changes nothing.\n\n"
    "**Removing a period that lessons are timetabled into is refused** (409). Shortening "
    "the day would otherwise leave lessons in a period the grid no longer draws — rows "
    "nothing is broken enough to notice. Clear those lessons first. The check spans every "
    "year the school has run, not just the current one, because last year's stranded "
    "lessons are the ones nobody would think to look for.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def set_timetable_periods(
    school_code: str,
    body: TimetablePeriodsIn,
    timetables: Timetables,
    caller: Registrar,
) -> list[TimetablePeriodOut]:
    if caller.profile is not None and not (
        caller.profile.is_system_admin
        or caller.profile.has_role(RoleCode.SCHOOL_MANAGER.value)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "manager_required", "message": "Only the school manager can change the number of periods."},
        )
    caller.narrow(
        Permission.TIMETABLE_WRITE, lambda scopes: scopes.for_school(school_code)
    )
    with domain_errors():
        stored = timetables.set_periods(
            SchoolCode(school_code),
            [
                TimetablePeriod(
                    school_code=school_code,
                    period_number=period.period_number,
                    name_en=period.name_en,
                    name_ar=period.name_ar,
                    starts_at=period.starts_at,
                    ends_at=period.ends_at,
                    is_teaching=period.is_teaching,
                )
                for period in body.periods
            ],
        )
    return [TimetablePeriodOut.of(period) for period in stored]


# -- The lessons ------------------------------------------------------------

@router.put("/timetable/break", status_code=204)
def set_grade_break(body: GradeBreakIn, timetables: Timetables, caller: Registrar) -> None:
    caller.narrow(
        Permission.TIMETABLE_WRITE,
        lambda scopes: scopes.for_year_level(
            school_id=caller.school_id, year_level_code=body.year_level_code
        ),
    )
    with domain_errors():
        timetables.set_break_for_level(AcademicYearCode(body.academic_year_code), YearCode(body.year_level_code), body.period_number, body.break_duration_minutes)


@router.get(
    "/timetable/week",
    response_model=WeekPlanOut,
    summary="One class's week, drawn against the school's own grid",
    description="The screen this feature exists for, in one request: the school's working "
    "days, its period grid, and the lessons in between. One route rather than three because "
    "three can disagree — a period removed between the second and the third call leaves a "
    "screen drawing a lesson in a row that is no longer there.\n\n"
    "The Arabic and Languages sections need no parameter here. A class belongs to a rung and "
    "a rung to exactly one track, so asking for a class has already chosen a section.",
    responses=error_responses(401, 403, 404, 422),
)
def read_week(
    timetables: Timetables,
    caller: Reader,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    class_code: Annotated[str, Query(examples=["3A"])],
    term: Annotated[str, Query(examples=["2026-T1"])],
) -> WeekPlanOut:
    caller.narrow(
        Permission.TIMETABLE_READ,
        lambda scopes: scopes.for_class(
            academic_year_code=academic_year, class_code=class_code
        ),
    )
    with domain_errors():
        plan = timetables.week_for_class(
            AcademicYearCode(academic_year), ClassCode(class_code), TermCode(term)
        )
    return WeekPlanOut.of(plan)


@router.get(
    "/guardians/by-id/{public_id}/students/{student_number}/timetable",
    response_model=StudentWeekOut,
    summary="A child's week, read by one of her guardians",
    description="The same week as the route above, for a caller that holds a guardian "
    "handle and a student number rather than a registrar's authority and a class code.\n\n"
    "**The class is resolved here, not by the caller.** A parent has no class code and a "
    "parent-facing service has no business holding one — it changes mid-year, and a client "
    "that cached one would eventually ask for the week of a room the child has left. This "
    "route resolves her placement *for the term asked about* and reads that class's week in "
    "one transaction, so the two cannot disagree.\n\n"
    "The guardian-to-child link is re-checked on this request: a caller naming a child who "
    "is not hers, or whose access a court order has restricted, gets the same 404 as one "
    "naming a child who does not exist. This is the route a parent-facing chat service "
    "should use.\n\n"
    "A child with no placement covering the term answers 200 with `class_code: null` rather "
    "than 404 — she was enrolled for September, or has left, and neither is a missing "
    "record. See `StudentWeekOut` on why that is not the same as a class with no lessons.",
    responses=error_responses(401, 403, 404, 422),
)
def read_guardian_student_week(
    public_id: str,
    student_number: str,
    timetables: Timetables,
    queries: Queries,
    caller: Reader,
    request_id: RequestId,
    term: Annotated[
        str,
        Query(
            description="Term code. Required, for the reason the guardian grades route "
            "gives: a bare 'this term' would be answered from a clock, and a timetable is "
            "a statement about a named stretch of the year.",
            examples=["2026-T1"],
        ),
    ],
) -> StudentWeekOut:
    # No `caller.narrow`, and the asymmetry with every other route in this module is
    # deliberate — the same one the guardian-scoped grades route makes. A registrar's key
    # is bounded by the rooms they hold; this caller is a service acting for one family,
    # and what bounds it is the link re-checked below, from the registrar's own data, on
    # this request. Narrowing by scope would refuse it for holding no class at all.
    with domain_errors():
        queries.require_guardian_may_see(
            public_id,
            StudentNumber(student_number),
            actor=caller.prefix,
            request_id=request_id,
        )
        week = timetables.week_for_student(
            StudentNumber(student_number), TermCode(term)
        )
    return StudentWeekOut.of(week)


@router.get(
    "/timetable",
    response_model=list[TimetableEntryOut],
    summary="Every lesson in a year, optionally one term or one grade",
    description="The whole-school view — how a clash is spotted by eye, and the read a "
    "teacher-allocation screen starts from. Ordered by class, then by the school's own "
    "week, then by period.\n\n"
    "`year_level` cuts it to one grade, and a grade-scoped caller must pass it: without "
    "it the read is bounded only by the school, which is what a supervisor of one rung "
    "does not hold. Naming the grade is what lets their `timetable.read` grant match.",
    responses=error_responses(401, 403, 404, 422),
)
def list_timetable(
    timetables: Timetables,
    caller: Reader,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    term: Annotated[str | None, Query(examples=["2026-T1"])] = None,
    year_level: Annotated[
        str | None,
        Query(
            examples=["AR-P4"],
            description="Restrict to one grade. Required of a grade-scoped caller.",
        ),
    ] = None,
) -> list[TimetableEntryOut]:
    # Without this the route was gated on holding `timetable.read` *somewhere* and never
    # on where — so a supervisor of one rung received every lesson in the school. The
    # narrowing is the same shape as the subject and student listings: the more the query
    # names, the more grants can match it.
    if year_level is None:
        caller.narrow(
            Permission.TIMETABLE_READ, lambda scopes: scopes.for_year(academic_year)
        )
    else:
        caller.narrow(
            Permission.TIMETABLE_READ,
            lambda scopes: scopes.for_year_level(
                school_id=caller.school_id, year_level_code=year_level
            ),
        )
    with domain_errors():
        entries = timetables.entries_for_year(
            AcademicYearCode(academic_year),
            term_code=None if term is None else TermCode(term),
            year_level_code=None if year_level is None else YearCode(year_level),
        )
    return [TimetableEntryOut.of(entry) for entry in entries]


@router.put(
    "/timetable",
    response_model=list[TimetableEntryOut],
    summary="Place lessons in slots",
    description="Idempotent by slot: `(class, term, day, period)` is the identity, so "
    "re-sending Sunday period 2 for 3A replaces what is there rather than adding a second "
    "lesson at the same moment. That is what makes laying out a grid safe to click twice.\n\n"
    "**All or nothing.** Every rule is checked over the whole batch before anything is "
    "written, so a thirty-five-slot week with one bad cell is refused entire. A partly "
    "applied week looks finished and is not.\n\n"
    "Refusals, and why they differ:\n\n"
    "* **422** — the request could not have meant anything valid: a day the school does not "
    "open, a period it does not run, a term belonging to another year.\n"
    "* **409** — well formed, and the stored state forbids it: two lessons sent for one "
    "slot, a period that is a break, or a subject that this class's grade is not assigned "
    "to teach.\n"
    "* **404** — a code that names nothing: an unknown class, term, subject or year.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def place_lessons(
    body: TimetableEntriesIn, timetables: Timetables, caller: Registrar
) -> list[TimetableEntryOut]:
    # Every class the body touches, not just the first: a year supervisor holds this rung
    # and a posting that reaches one class off it is refused whole. The write is one
    # transaction, so a partial answer was never available anyway.
    caller.narrow_all(
        Permission.TIMETABLE_WRITE,
        lambda scopes: [
            scopes.for_class(
                academic_year_code=body.academic_year_code, class_code=code
            )
            for code in {entry.class_code for entry in body.entries}
        ],
    )
    with domain_errors():
        placed = timetables.place(
            AcademicYearCode(body.academic_year_code),
            [
                TimetableEntry(
                    slot=TimetableSlot(
                        class_code=entry.class_code,
                        term_code=entry.term_code,
                        day_of_week=entry.day_of_week,
                        period_number=entry.period_number,
                    ),
                    academic_year_code=body.academic_year_code,
                    subject_code=entry.subject_code,
                )
                for entry in body.entries
            ],
        )
    return [TimetableEntryOut.of(entry) for entry in placed]


@router.put(
    "/timetable/week",
    response_model=list[TimetableEntryOut],
    summary="Save timetable changes",
    description="Persists all additions, edits and cleared slots from one timetable editing "
    "session in a single transaction. Either every visible change reaches the database or "
    "none does. The caller must hold `timetable.write` for every class named in either "
    "part of the request.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def save_week_changes(
    body: TimetableChangesIn, timetables: Timetables, caller: Registrar
) -> list[TimetableEntryOut]:
    class_codes = {entry.class_code for entry in body.entries} | {
        slot.class_code for slot in body.clear_slots
    }
    caller.narrow_all(
        Permission.TIMETABLE_WRITE,
        lambda scopes: [
            scopes.for_class(
                academic_year_code=body.academic_year_code, class_code=code
            )
            for code in class_codes
        ],
    )
    with domain_errors():
        stored = timetables.save_changes(
            AcademicYearCode(body.academic_year_code),
            [
                TimetableEntry(
                    slot=TimetableSlot(
                        class_code=entry.class_code,
                        term_code=entry.term_code,
                        day_of_week=entry.day_of_week,
                        period_number=entry.period_number,
                    ),
                    academic_year_code=body.academic_year_code,
                    subject_code=entry.subject_code,
                )
                for entry in body.entries
            ],
            [
                TimetableSlot(
                    class_code=slot.class_code,
                    term_code=slot.term_code,
                    day_of_week=slot.day_of_week,
                    period_number=slot.period_number,
                )
                for slot in body.clear_slots
            ],
        )
    return [TimetableEntryOut.of(entry) for entry in stored]


@router.post(
    "/timetable/copy-term",
    response_model=list[TimetableEntryOut],
    summary="Copy a class timetable from one term to another",
    description="Copies a saved weekly timetable into an empty target term. The target is "
    "never overwritten: if it already has any entries the request is refused, so the two "
    "terms remain independently editable after the initial default is applied.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def copy_term_week(
    body: TimetableTermCopyIn, timetables: Timetables, caller: Registrar
) -> list[TimetableEntryOut]:
    caller.narrow_all(
        Permission.TIMETABLE_WRITE,
        lambda scopes: [
            scopes.for_class(
                academic_year_code=body.academic_year_code, class_code=body.class_code
            )
        ],
    )
    with domain_errors():
        copied = timetables.copy_term_week(
            AcademicYearCode(body.academic_year_code),
            ClassCode(body.class_code),
            TermCode(body.source_term_code),
            TermCode(body.target_term_code),
        )
    return [TimetableEntryOut.of(entry) for entry in copied]


@router.post(
    "/timetable/clear",
    response_model=ClearedOut,
    status_code=status.HTTP_200_OK,
    summary="Empty slots",
    description="Removes the lessons in these slots and answers how many there were.\n\n"
    "Not the same as placing a lesson with no subject. That states \"this class has this "
    "period free\"; this states \"nobody has planned this slot\". Both are real, and a "
    "registrar has to be able to say which one they mean.\n\n"
    "A POST rather than a DELETE because it carries a body of slots — a DELETE with a "
    "request body is permitted but is dropped by enough proxies to be a poor bet for a "
    "route that silently doing nothing is indistinguishable from succeeding on.",
    responses=error_responses(401, 403, 404, 422),
)
def clear_slots(
    body: TimetableSlotsIn, timetables: Timetables, caller: Registrar
) -> ClearedOut:
    caller.narrow_all(
        Permission.TIMETABLE_WRITE,
        lambda scopes: [
            scopes.for_class(
                academic_year_code=body.academic_year_code, class_code=code
            )
            for code in {slot.class_code for slot in body.slots}
        ],
    )
    with domain_errors():
        removed = timetables.clear(
            AcademicYearCode(body.academic_year_code),
            [
                TimetableSlot(
                    class_code=slot.class_code,
                    term_code=slot.term_code,
                    day_of_week=slot.day_of_week,
                    period_number=slot.period_number,
                )
                for slot in body.slots
            ],
        )
    return ClearedOut(removed=removed)
