"""The weekly plan: which class sits which lesson, on which day, in which period.

This is a *plan*, and everything about the module follows from that. A grade is a fact
about a child that must never be silently changed; a timetable is a statement about next
week that a school rewrites whenever a teacher leaves. So the rules here are structural —
no two lessons in one room at one time — and nothing here records what actually happened.
Attendance stays where it is, one mark per child per day; this module gives it nothing it
did not have, and is not a step toward per-lesson registers in this stage.

Four decisions shape it.

**The week comes from the school, never from a constant.** `School.working_days` has been
carrying the answer since it was added, saying "consumed by the future timetable" — this is
that consumer. A school that opens Saturday to Wednesday has a five-day grid starting on
Saturday, and a school that opens three days a week has three columns. `WorkingDay` is
reused rather than re-declared, so the two cannot drift.

**A period is a slot in the school's day, not a time range on a lesson.** The bell rings
once for the whole building, so the times live on `TimetablePeriod` — one row per period per
school — and a lesson names the period number. Storing a start time on every lesson would
mean a school moving second period edits every class's Tuesday instead of one row, and the
copies would disagree within a term.

**Period times are optional, for the same reason term dates are.** A school lays out "we
run seven periods" long before it has agreed when each one rings, and a placeholder time is
worse than none: nothing downstream can tell an invented 08:00 from an agreed one.

**A slot is identified by (class, term, day, period).** The term is part of it because a
timetable is a statement about a stretch of the year, and schools genuinely re-plan between
terms. That also makes "one class cannot be in two places at once" a uniqueness constraint
the database can hold, rather than a rule a service has to remember to check.

The academic track needs no mention anywhere in this module, and that is deliberate: a
class belongs to a rung, and a rung belongs to exactly one track. The Arabic and Languages
sections therefore have separate timetables by construction, and a `track_id` here could
only ever disagree with the rung's own.
"""
from dataclasses import dataclass
from datetime import time

from sis.domain.errors import ValidationError
from sis.domain.structure import WorkingDay
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    SubjectCode,
    TermCode,
)

__all__ = [
    "MAX_PERIODS_PER_DAY",
    "TimetableEntry",
    "TimetablePeriod",
    "TimetableSlot",
]

#: The most periods one school day may be divided into.
#:
#: A bound rather than a policy: no school runs a twenty-period day, and the value of
#: saying so is that a typo of `70` for `7` is refused at the boundary instead of rendering
#: a grid seventy rows tall. Raise it if a school genuinely needs more — nothing depends on
#: the number itself.
MAX_PERIODS_PER_DAY = 20


def _period_number(value: object, field: str = "period_number") -> int:
    """A period number is a whole number from 1 to `MAX_PERIODS_PER_DAY`.

    `bool` is checked before `int` because `True` is an `int` in Python and would sail
    through as period 1 — the same trap `display_order` and `term_count` guard against.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("period number must be a whole number", field=field)
    if not 1 <= value <= MAX_PERIODS_PER_DAY:
        raise ValidationError(
            f"period number must be between 1 and {MAX_PERIODS_PER_DAY}", field=field
        )
    return value


def _working_day(value: object, field: str = "day_of_week") -> WorkingDay:
    """Coerce to `WorkingDay`, naming the field so the message lands on the right box.

    Whether the day is one the *school* opens is not asked here. This type knows what a
    weekday is; only the school knows which of them it teaches on, and a domain object
    that reached for that answer would need the school passed into every constructor.
    The check lives in the service, next to the school it has to read.
    """
    if isinstance(value, WorkingDay):
        return value
    try:
        return WorkingDay(str(value))
    except ValueError:
        raise ValidationError(
            f"unknown working day {value!r}; expected one of "
            + ", ".join(day.value for day in WorkingDay),
            field=field,
        ) from None


@dataclass(frozen=True, slots=True)
class TimetablePeriod:
    """One slot in a school's day — "period 3", "break" — shared by every class in it.

    Held per school rather than per class because the bell is: second period starts at the
    same moment in 3A and in 5B, and a school that moves it should edit one row. The names
    are optional and exist for the slots that are not lessons, so a grid can show "Break"
    between periods 3 and 4 without a class having to schedule anything into it.

    `is_teaching=False` marks exactly those slots. Nothing may be scheduled into one, which
    is the only rule this flag carries — it is not a claim about supervision or duty.
    """

    school_code: str
    period_number: int
    name_en: str = ""
    name_ar: str = ""
    starts_at: time | None = None
    ends_at: time | None = None
    is_teaching: bool = True

    def __post_init__(self) -> None:
        code = str(self.school_code or "").strip().upper()
        if not code:
            raise ValidationError("school code is required", field="school_code")
        object.__setattr__(self, "school_code", code)
        object.__setattr__(
            self, "period_number", _period_number(self.period_number)
        )
        object.__setattr__(self, "name_en", str(self.name_en or "").strip())
        object.__setattr__(self, "name_ar", str(self.name_ar or "").strip())
        # A stated range must not be inverted; an absent one is checked no further. Same
        # rule as a term's dates, and for the same reason — see the module docstring.
        if (
            self.starts_at is not None
            and self.ends_at is not None
            and self.ends_at <= self.starts_at
        ):
            raise ValidationError(
                "a period must end after it starts", field="ends_at"
            )

    @property
    def is_timed(self) -> bool:
        """Whether the school has stated both ends of this period.

        One end alone is not a time range, for the reason a term with only a start date is
        not a window: every question asked of a period is asked of the whole slot.
        """
        return self.starts_at is not None and self.ends_at is not None


@dataclass(frozen=True, slots=True)
class TimetableSlot:
    """Where a lesson sits: one class, one term, one day, one period.

    Split out from `TimetableEntry` because it is the identity, and identity is what the
    conflict rules are about. Two entries with equal slots are the same lesson stated
    twice — never two lessons — and that is a sentence worth having a type for rather than
    a four-field comparison repeated at each call site.
    """

    class_code: ClassCode | str
    term_code: TermCode | str
    day_of_week: WorkingDay | str
    period_number: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_code", ClassCode(self.class_code))
        object.__setattr__(self, "term_code", TermCode(self.term_code))
        object.__setattr__(self, "day_of_week", _working_day(self.day_of_week))
        object.__setattr__(
            self, "period_number", _period_number(self.period_number)
        )

    @property
    def key(self) -> tuple[str, str, str, int]:
        """The slot as comparable strings, for keying a dict of what is already placed."""
        return (
            str(self.class_code),
            str(self.term_code),
            str(self.day_of_week),
            self.period_number,
        )


@dataclass(frozen=True, slots=True)
class TimetableEntry:
    """One lesson in the weekly plan.

    The five things a lesson connects to are all here, and only two of them are optional:

    * **class** and **term** — through `slot`, and both required. A lesson nobody attends
      at a time nobody named is not a lesson.
    * **academic year** — carried rather than derived, because a term names one and a class
      names one and this type is where they have to be the *same* one. The service sets it
      from the class; a caller does not supply it.
    * **subject** — optional, so a school can lay out its grid before deciding what goes in
      each slot, and so a slot can honestly hold nothing. When it *is* set, the service
      checks the subject is one that rung is assigned to teach (stage 5) — a Primary class
      with Physics on Tuesday is exactly the leak that assignment work closed.
    * **teacher** — optional, and left unset by everything in this stage. Teacher records
      exist (`teachers`, revision 0007) but nobody manages them yet, so the column is here
      to be filled in later rather than to be filled in now. The conflict rule that will
      matter then — one teacher, one room, one moment — is already enforced.
    """

    slot: TimetableSlot
    academic_year_code: AcademicYearCode | str
    subject_code: SubjectCode | str | None = None
    teacher_staff_number: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slot, TimetableSlot):
            raise ValidationError("slot is required", field="slot")
        if not isinstance(self.academic_year_code, AcademicYearCode):
            object.__setattr__(
                self, "academic_year_code", AcademicYearCode(self.academic_year_code)
            )
        if self.subject_code is not None and not isinstance(
            self.subject_code, SubjectCode
        ):
            object.__setattr__(self, "subject_code", SubjectCode(self.subject_code))
        if self.teacher_staff_number is not None:
            staff = str(self.teacher_staff_number).strip()
            object.__setattr__(self, "teacher_staff_number", staff or None)

    @property
    def is_free(self) -> bool:
        """Whether this slot holds no subject — a period the class has nothing timetabled in.

        A stated fact, not an absence of one. A row with no subject says "this slot is
        deliberately empty"; a slot with no row says "nobody has planned this far yet", and
        a screen that rendered the two identically would give a registrar no way to tell a
        finished timetable with a free period from an unfinished one.
        """
        return self.subject_code is None
