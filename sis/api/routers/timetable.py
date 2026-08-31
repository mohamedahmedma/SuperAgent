"""The weekly plan over HTTP: a school's periods, and one lesson per class per slot.

Its own router rather than more routes on `structure`, because a timetable is asked about
differently from the ladder it hangs on. The structure routes answer "what does this school
consist of" and are read once a term by a registrar setting up; these answer "what is 3A
doing on Tuesday" and are read by whoever is standing in front of 3A.

Three shapes of route, and the middle one is the whole feature:

  `/schools/{code}/timetable-periods`   the school's day — how many periods, when they ring
  `/timetable`                          lessons: read a week, place lessons, clear slots
  `/timetable/week`                     one class's week drawn against the school's own grid

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

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from sis.api.deps import Principal, get_timetable_service, require_permission
from sis.domain.rbac import Permission
from sis.api.routers import domain_errors, error_responses
from sis.application.services import TimetableService, WeekPlan
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
    TermCode,
    YearCode,
)

router = APIRouter(prefix="/v1", tags=["timetable"])

Reader = Annotated[Principal, Depends(require_permission(Permission.TIMETABLE_READ))]
Registrar = Annotated[Principal, Depends(require_permission(Permission.TIMETABLE_WRITE))]
Timetables = Annotated[TimetableService, Depends(get_timetable_service)]


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
        description="Always `null` in this stage: teacher records exist but nobody manages "
        "them yet. The field and its conflict rule are already in place, so assigning a "
        "teacher later is a write rather than a migration.",
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
