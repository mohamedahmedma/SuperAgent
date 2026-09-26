"""The daily register: who was in the room, on which day.

One mark per child per day. That is the whole model, and the restraint is deliberate —
attendance per lesson would need a timetable (periods, which subject when, which teacher),
and this service holds none of those things. A daily register is what a form teacher takes
at the start of the day and what a parent is telephoned about, and it can be built honestly
out of what is already here.

Three rules shape this module, and each is the same rule the rest of the service already
follows one level up.

**An unrecorded day is not an attendance.** There is no `UNKNOWN` state and no row written
to mean "nobody took the register". A day with no row is a day nobody marked, and counting
it as present flatters the school while counting it as absent accuses a child. Every count
in this file therefore reports what it counted out of, so a rate is never computed against
a denominator that includes days nobody looked. This is invariant 1 — a blank is not a zero
— applied to a different column.

**A mark belongs to the class she was in that day.** `class_section_id` is stored on the
mark rather than resolved from the child at read time, for the same reason it is stored on
a grade: a child who moves from 3A to 3B in March was in 3A in October, and her October
register has to keep saying so.

**Nothing here reads the clock.** Every question that depends on a date takes it as an
argument, so a service's behaviour in a test is a function of its inputs rather than of the
day the suite runs.
"""
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import ClassVar

from sis.domain.errors import ValidationError
from sis.domain.value_objects import ClassCode, StudentNumber

__all__ = ["AttendanceMark", "AttendanceState", "AttendanceTally"]


class AttendanceState(StrEnum):
    """What was recorded about one child on one day.

    Four states, and the two beyond present/absent are the ones that earn their keep:

    `LATE` is a distinct state rather than a present-with-a-note because a school acts on
    it differently — a child late eleven times is a conversation with a parent, and folding
    those eleven days into "present" makes the pattern invisible. She was in the room, so
    she is not absent either.

    `EXCUSED` is an absence the school has accepted: illness with a note, a medical
    appointment, a bereavement. Kept apart from `ABSENT` because the number that matters to
    anybody asking about a child is unexplained absence, and a system that cannot separate
    the two reports a child recovering from surgery exactly as it reports truancy.

    Stored as text so a database dump is readable and so adding a state is not a
    renumbering of the ones already written. Deliberately absent: any member meaning
    "unknown" — see the module docstring.
    """

    PRESENT = "present"
    ABSENT = "absent"
    LATE = "late"
    EXCUSED = "excused"

    @property
    def was_in_the_room(self) -> bool:
        """Whether the child was physically there. Late still counts as there."""
        return self in (AttendanceState.PRESENT, AttendanceState.LATE)

    @property
    def is_absence(self) -> bool:
        """Whether she was away, explained or not."""
        return self in (AttendanceState.ABSENT, AttendanceState.EXCUSED)


@dataclass(frozen=True, slots=True)
class AttendanceMark:
    """One child, one day, one state, and the class she was in when it was taken.

    Frozen, like every other record in this domain: a mark handed to two services must not
    change shape under one of them. Correcting a register is writing a new mark for that
    day, which the repository upserts onto the same `(student, date)` — so a correction
    replaces the statement rather than accumulating a second contradictory one.
    """

    student_number: StudentNumber | str
    on_date: date
    state: AttendanceState | str
    class_section_id: int
    class_code: ClassCode | str
    note: str = ""

    #: A note is required for an excused absence and optional otherwise. "Excused by whom,
    #: for what" is the entire content of the state, and an excused absence with no reason
    #: on file is indistinguishable from a registrar clicking the wrong button.
    NOTE_REQUIRED_FOR: ClassVar[frozenset[str]] = frozenset({AttendanceState.EXCUSED})

    def __post_init__(self) -> None:
        if not isinstance(self.student_number, StudentNumber):
            object.__setattr__(self, "student_number", StudentNumber(self.student_number))
        if not isinstance(self.class_code, ClassCode):
            object.__setattr__(self, "class_code", ClassCode(self.class_code))

        if not isinstance(self.state, AttendanceState):
            try:
                object.__setattr__(
                    self, "state", AttendanceState(str(self.state).strip().lower())
                )
            except ValueError:
                raise ValidationError(
                    f"{self.state!r} is not an attendance state; expected one of "
                    + ", ".join(member.value for member in AttendanceState),
                    field="state",
                ) from None

        object.__setattr__(self, "note", str(self.note or "").strip())

        if self.state in self.NOTE_REQUIRED_FOR and not self.note:
            raise ValidationError(
                "an excused absence needs a reason on file; without one it cannot be "
                "told apart from an ordinary absence marked by mistake",
                field="note",
            )

        if not isinstance(self.class_section_id, int) or isinstance(
            self.class_section_id, bool
        ):
            raise ValidationError(
                "class_section_id must be the surrogate id of the class she was in",
                field="class_section_id",
            )

    @property
    def key(self) -> tuple[str, date]:
        """`(student, day)` — the pair that is unique. A second mark for a day replaces it."""
        return (str(self.student_number), self.on_date)


@dataclass(frozen=True, slots=True)
class AttendanceTally:
    """Counts over a set of marks, and the number of days those marks cover.

    Counts, and no rate. `present / recorded` is a division this type could perform in one
    line and deliberately does not, because the figure it produces is read as "her
    attendance is 94%" while meaning "94% of the days somebody remembered to take the
    register". A school that stopped marking on Fridays would show a rising attendance rate.
    Callers that genuinely want a percentage have `recorded` to divide by and have to say
    so in their own code, where a reader can see the denominator.

    `recorded` is not "school days" either. This service does not hold a calendar, so it
    cannot know whether an unmarked Tuesday was a holiday or an oversight, and it does not
    guess: `recorded` is the number of days a mark exists for, and nothing else.
    """

    present: int = 0
    absent: int = 0
    late: int = 0
    excused: int = 0

    @property
    def recorded(self) -> int:
        """Days a mark exists for. The only honest denominator this service has."""
        return self.present + self.absent + self.late + self.excused

    @property
    def in_the_room(self) -> int:
        """Days she was there, late included."""
        return self.present + self.late

    @property
    def away(self) -> int:
        """Days she was away, explained or not."""
        return self.absent + self.excused

    def with_mark(self, state: AttendanceState) -> "AttendanceTally":
        """This tally plus one mark. Returns a new tally; nothing here mutates."""
        counts = {
            "present": self.present,
            "absent": self.absent,
            "late": self.late,
            "excused": self.excused,
        }
        counts[state.value] += 1
        return AttendanceTally(**counts)


def tally(marks: "list[AttendanceMark] | tuple[AttendanceMark, ...]") -> AttendanceTally:
    """Count a set of marks by state.

    Every state appears in the result even at zero, which is the same choice
    `sis.domain.imports.tally` makes and for the same reason: a screen that renders only
    the non-zero counts silently stops mentioning absence in the week a class has none,
    and the reader cannot tell "no absences" from "we stopped counting".
    """
    counted = AttendanceTally()
    for mark in marks:
        state = mark.state
        if not isinstance(state, AttendanceState):  # pragma: no cover - post_init coerces
            state = AttendanceState(str(state))
        counted = counted.with_mark(state)
    return counted
