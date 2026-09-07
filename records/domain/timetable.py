"""One child's week, as a system of record states it.

Values, not rows, like [marks.py](marks.py): this service asks, reshapes, and forgets.
Two things about the shape are decisions rather than translations.

**The grid travels with the lessons.** A week is `periods` — the rows of the school's day,
breaks included — *and* `lessons`, which fill some of them. Read apart they can disagree,
and the symptom is a client drawing a seven-row day against an eight-period grid. They are
one value here for the same reason SIS serves them from one route.

**"She has no class" and "her class has no timetable" are different answers.** The first is
a fact about the child — she had left, or had not yet joined — and the second is a fact
about the school, which has not typed the week in yet. `TimetableStatus` names which,
because a consumer that inferred it from an empty list would tell a parent her daughter has
no lessons when the truth is that nobody has written them down. That is the same rule
`SubjectGrade.academic_unavailable` exists for: a model handed an empty list will narrate a
plausible reason for it, and a model handed a reason will not.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TimetableStatus(str, Enum):
    """Why a week is or is not a grid of lessons.

    Never inferred from emptiness by a consumer — see the module docstring.
    """

    #: She has a class and it has lessons in it.
    OK = "ok"
    #: No placement covered the term asked about. There is no room to ask about, so there
    #: is nothing missing: a child enrolled for next September has no class today.
    NO_CLASS = "no_class"
    #: She has a class and nobody has laid out its week for this term. A fact about the
    #: school's admin, not about the child's day.
    NO_TIMETABLE = "no_timetable"


@dataclass(frozen=True)
class TimetablePeriodSlot:
    """One slot in the school's day — a row of the grid, shared by every class in it.

    Held per school rather than per lesson because the bell is: second period starts at the
    same moment in 3A and in 5B.

    `is_teaching` is `False` for a break, assembly or prayer. Such a slot is part of the day
    and no class timetables a lesson into it, so a client rendering the day must draw it and
    a caller counting lessons must not.
    """

    period_number: int
    name_ar: str = ""
    name_en: str = ""
    #: `HH:MM`, or `""` when the school has not fixed this boundary. Never a placeholder:
    #: an invented `08:00` is indistinguishable afterwards from an agreed one, which is the
    #: same rule the absent grade follows. A school settles how many periods it runs long
    #: before it agrees when each one rings.
    starts_at: str = ""
    ends_at: str = ""
    is_teaching: bool = True

    @property
    def is_timed(self) -> bool:
        """Whether both ends are on file. One end alone is not a range."""
        return bool(self.starts_at and self.ends_at)


@dataclass(frozen=True)
class TimetableLesson:
    """One lesson, in one slot of one day.

    Carries the subject's **names** and not only its code, because the reader at the end of
    this contract is a parent. `MATH` is the school's own filing key and means nothing to a
    family.
    """

    day_of_week: str
    period_number: int
    subject_code: str = ""
    subject_name_ar: str = ""
    subject_name_en: str = ""

    @property
    def is_free(self) -> bool:
        """A slot the class deliberately has off — a stated fact, not an absence of one.

        A lesson with no subject says "this period is free"; a slot with no lesson at all
        says "nobody has planned this far yet". A reader that rendered the two identically
        would give a parent no way to tell a finished week with a free period from an
        unfinished one.
        """
        return not self.subject_code


@dataclass(frozen=True)
class StudentTimetable:
    """One child's week: the class she sits in for a term, and what that class does.

    `class_code` is empty exactly when `status` is `NO_CLASS`. Both are carried rather than
    one derived from the other, so a consumer can branch on the reason without having to
    know which emptiness means what.
    """

    status: TimetableStatus = TimetableStatus.NO_CLASS
    class_code: str = ""
    class_name_ar: str = ""
    class_name_en: str = ""
    #: The school's working days, **in the school's own order**. Never sorted by a consumer:
    #: the week begins on Saturday at some schools and Sunday at others, and only the school
    #: knows which.
    days: tuple[str, ...] = ()
    periods: tuple[TimetablePeriodSlot, ...] = ()
    #: Ordered by day — in the school's week order — then by period.
    lessons: tuple[TimetableLesson, ...] = ()
    #: How many slots this week could hold a lesson: teaching periods times open days. The
    #: denominator for "how full is this timetable".
    teaching_slots: int = 0

    @property
    def has_class(self) -> bool:
        """Whether a placement covered the term at all."""
        return self.status is not TimetableStatus.NO_CLASS

    def lessons_on(self, day: str) -> tuple[TimetableLesson, ...]:
        """That day's lessons, in period order.

        Here rather than in a template because a day with no lessons is a real answer a
        renderer has to be able to ask for, and matching the day is a fold on the string
        the school stated rather than a comparison a caller should be writing itself.
        """
        wanted = str(day).strip().lower()
        return tuple(
            lesson
            for lesson in self.lessons
            if lesson.day_of_week.strip().lower() == wanted
        )


__all__ = [
    "StudentTimetable",
    "TimetableLesson",
    "TimetablePeriodSlot",
    "TimetableStatus",
]
