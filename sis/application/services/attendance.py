"""Taking the register, and reading it back.

Two use cases, and the first one is the whole reason this service exists rather than a bare
repository call:

`register_for_class` builds the day's register out of the *enrolments*, then folds in
whatever marks were recorded. That order matters. Built from the marks instead, the screen
would show four children on a day somebody marked four and stopped — and the thirty-six
missing children would look like a school with excellent attendance rather than a register
half taken. Every child placed in the class on that day appears, and the ones nobody marked
appear with no state at all.

`take_register` writes a whole day in one transaction, and refuses a child who was not in
the class that day. That refusal is the one rule here worth arguing about, and it is
deliberate: a register naming a child who had already transferred is either a stale screen
or the wrong class, and writing it would file her attendance under a room she had left —
the same failure invariant 2 exists to prevent for marks.

Ports only. No sqlalchemy, no fastapi, no clock: the day is always an argument, so a test
of "she is not on Monday's register" does not depend on the day the suite runs.
"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from sis.application.ports.unit_of_work import UnitOfWork
from sis.domain.attendance import AttendanceMark, AttendanceState, AttendanceTally, tally
from sis.domain.errors import UnknownReference, ValidationError
from sis.domain.people import Student
from sis.domain.value_objects import AcademicYearCode, ClassCode, StudentNumber

__all__ = [
    "AttendanceService",
    "RegisterEntry",
    "ClassRegister",
    "StudentAttendance",
]


@dataclass(frozen=True, slots=True)
class RegisterEntry:
    """One line of a day's register: the child, and what was recorded about her if anything.

    `mark` is `None` when nobody marked her, and that is a third state distinct from present
    and absent. A screen that renders `None` as either one is claiming a fact the school
    never stated — which is the same mistake as rendering a blank grade as a zero.
    """

    student_number: str
    student: Student | None
    mark: AttendanceMark | None = None

    @property
    def state(self) -> AttendanceState | None:
        return None if self.mark is None else self.mark.state

    @property
    def is_marked(self) -> bool:
        return self.mark is not None


@dataclass(frozen=True, slots=True)
class ClassRegister:
    """A class, a day, and every child placed in it on that day.

    `counts` covers only the children who were marked; `unmarked` is reported separately and
    never folded into it. That separation is the point: "28 present, 2 absent, 10 unmarked"
    is a true and actionable statement about a register that is not finished, and "28 present
    out of 40" is not.
    """

    academic_year_code: str
    class_code: str
    on_date: date
    entries: tuple[RegisterEntry, ...]
    counts: AttendanceTally

    @property
    def unmarked(self) -> int:
        return sum(1 for entry in self.entries if not entry.is_marked)

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def is_complete(self) -> bool:
        """Whether every child on the register has a mark. Not whether they were present."""
        return self.size > 0 and self.unmarked == 0


@dataclass(frozen=True, slots=True)
class StudentAttendance:
    """One child's marks over a range, with the counts and the range that produced them.

    The range is carried back deliberately. A count of absences means nothing without the
    window it was counted over, and a card that shows "3 absences" beside a filter the
    reader has forgotten is a number they will quote to a parent.
    """

    student_number: str
    from_date: date | None
    to_date: date | None
    marks: tuple[AttendanceMark, ...]
    counts: AttendanceTally


class AttendanceService:
    """The daily register: build one, take one, read one child's back."""

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    # -- Reads -------------------------------------------------------------

    def register_for_class(
        self,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        on_date: date,
    ) -> ClassRegister:
        """The register of one class on one day: every child placed, marked or not.

        An unknown class is a refusal rather than an empty register, for the reason every
        read in this service refuses: "no such class" and "a class with nobody in it" render
        identically, and only one of them is a typo the caller can fix.
        """
        with self._uow_factory() as uow:
            section_ids = uow.class_sections.ids_for(
                [(str(academic_year_code), str(class_code))]
            )
            section_id = section_ids.get((str(academic_year_code), str(class_code)))
            if section_id is None:
                raise UnknownReference(
                    f"no class {class_code} in academic year {academic_year_code}",
                    field="class_code",
                )

            # The enrolments come first and decide who is on the register. See the module
            # docstring: built from the marks instead, a half-taken register would read as a
            # small class with perfect attendance.
            placements = uow.enrolments.roster_on(
                academic_year_code, class_code, on_date
            )
            numbers = [StudentNumber(str(p.student_number)) for p in placements]
            students = uow.students.get_many(numbers) if numbers else {}
            marks = uow.attendance.marks_for_class(section_id, on_date)

            entries = tuple(
                RegisterEntry(
                    student_number=str(placement.student_number),
                    student=students.get(str(placement.student_number)),
                    mark=marks.get(str(placement.student_number)),
                )
                for placement in placements
            )

        return ClassRegister(
            academic_year_code=str(academic_year_code),
            class_code=str(class_code),
            on_date=on_date,
            entries=entries,
            counts=tally([entry.mark for entry in entries if entry.mark is not None]),
        )

    def for_guardian_student(
        self,
        public_id: str,
        student_number: StudentNumber,
        *,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> StudentAttendance:
        """The same record, for a caller who holds only a guardian handle.

        The sibling of `QueryService.guardian_student_term_grades`, and it exists for the
        same reason: the service that answers a parent must not be the service that decides
        which child it may answer about. The link is re-checked here, against the
        registrar's own data, on this request.

        Attendance needed this as much as marks did and did not have it. Until it did, the
        only guardian-scoped read in the service was grades, so anything asking about a
        child's absences had to be trusted to have filtered correctly first — and the
        caller doing the asking is a language model's tool.

        The check delegates to `QueryService` rather than repeating the link query. Two
        implementations of "may this parent see this child" is one more than can ever be
        right, and the one that drifted would be the one answering a parent.
        """
        from sis.application.services.queries import QueryService

        QueryService(self._uow_factory).require_guardian_may_see(public_id, student_number)
        return self.for_student(student_number, from_date=from_date, to_date=to_date)

    def for_student(
        self,
        student_number: StudentNumber,
        *,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> StudentAttendance:
        """One child's attendance over a range, with counts. Unknown child is a refusal.

        **Not authorisation-checked.** This is the registrar's view, reached by a registrar
        or reader key naming any child in the school. A parent-facing caller must use
        `for_guardian_student`, which re-checks the link first.
        """
        with self._uow_factory() as uow:
            if uow.students.get(student_number) is None:
                raise UnknownReference(
                    f"no student {student_number}", field="student_number"
                )
            marks = tuple(
                uow.attendance.marks_for_student(
                    student_number, from_date=from_date, to_date=to_date
                )
            )
        return StudentAttendance(
            student_number=str(student_number),
            from_date=from_date,
            to_date=to_date,
            marks=marks,
            counts=tally(list(marks)),
        )

    # -- Write -------------------------------------------------------------

    def take_register(
        self,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        on_date: date,
        states: Mapping[str, str],
        *,
        notes: Mapping[str, str] | None = None,
        actor: str = "",
    ) -> ClassRegister:
        """Record a day for a whole class, and return the register as it now stands.

        `states` maps student number to state, and a child left out of it is left alone
        rather than marked: taking the register for the twelve children present so far must
        not silently mark the other twenty-eight absent. Erasing a mark is a separate act
        this service does not offer, because "she was marked present by mistake" is
        corrected by marking her correctly, not by removing the statement.

        Every child named must be placed in this class on this day. A name that is not is
        refused with the number in the message — a stale screen or the wrong class, and
        writing it would file her attendance under a room she had left.
        """
        notes = notes or {}
        if not states:
            raise ValidationError(
                "no attendance was stated; the register has nothing to record",
                field="states",
            )

        with self._uow_factory() as uow:
            section_ids = uow.class_sections.ids_for(
                [(str(academic_year_code), str(class_code))]
            )
            section_id = section_ids.get((str(academic_year_code), str(class_code)))
            if section_id is None:
                raise UnknownReference(
                    f"no class {class_code} in academic year {academic_year_code}",
                    field="class_code",
                )

            placed = {
                str(p.student_number)
                for p in uow.enrolments.roster_on(
                    academic_year_code, class_code, on_date
                )
            }
            strangers = sorted(set(states) - placed)
            if strangers:
                raise ValidationError(
                    "not on this register on "
                    f"{on_date.isoformat()}: {', '.join(strangers)}. A child who has "
                    "transferred is marked in the class she is in now.",
                    field="states",
                )

            marks = [
                AttendanceMark(
                    student_number=number,
                    on_date=on_date,
                    state=state,
                    class_section_id=section_id,
                    class_code=class_code,
                    note=notes.get(number, ""),
                )
                for number, state in states.items()
            ]
            uow.attendance.upsert_many(marks, recorded_by=actor)
            uow.commit()

        # Read back rather than assembling the answer from what was just written: the
        # register the caller renders is then the register the database holds, including the
        # children this call did not name.
        return self.register_for_class(academic_year_code, class_code, on_date)
