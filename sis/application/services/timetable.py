"""Laying out a week, and refusing the ways it can be wrong.

The table can hold two of the five rules on its own — one class in one slot, one teacher in
one slot — because both are uniqueness and uniqueness is what a database is for. The other
three need to read something else, and that is what this module is:

**A lesson may only sit on a day the school opens.** `School.working_days` has carried the
answer since it was added, saying it was for "the future timetable". This is that consumer.
A school teaching Saturday to Wednesday has a five-column grid starting on Saturday, and a
Friday lesson at a school that shuts on Friday is refused rather than rendered off the edge
of every screen.

**A lesson may only sit in a period the school runs, and only in a teaching one.** The grid
is `timetable_periods`; scheduling into period 9 of a seven-period day, or into the break,
is a mistake with no sensible reading.

**A lesson may only name a subject that rung is assigned to teach.** This is stage 5's rule
arriving where it was always heading: Physics is assigned to Secondary, so Physics on a
Primary class's Tuesday is the same leak the assignment work closed, one table further on.
It is also where the academic tracks separate — the Arabic section's Secondary 1 and the
Languages section's are different rungs with different assignments, so neither can borrow
the other's subjects, and no code here mentions a track at all.

Two more properties worth stating because they are easy to lose:

**Writing a grid is one transaction.** A registrar lays out a week and either all of it
lands or none of it does. A half-written Tuesday is worse than an empty one, because it
looks finished.

**Nothing here touches attendance.** A timetable is a plan and the register is a record.
Per-lesson attendance would need exactly this table and is deliberately not built on it.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from sis.application.ports.unit_of_work import UnitOfWork
from sis.domain.errors import DomainRuleViolation, UnknownReference, ValidationError
from sis.domain.structure import School, WorkingDay
from sis.domain.timetable import TimetableEntry, TimetablePeriod, TimetableSlot
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    SchoolCode,
    TermCode,
    YearCode,
)

__all__ = ["TimetableConflict", "TimetableService", "WeekPlan"]


class TimetableConflict(DomainRuleViolation):
    """Two lessons cannot occupy one slot.

    Its own class rather than a bare `DomainRuleViolation` because the API answers it 409
    while the neighbouring refusals — an unknown class, a day the school does not open —
    are 404 and 422. A registrar reading "conflict" has to be able to tell "you have
    already put something there" apart from "that is not a thing you can put it on".
    """


@dataclass(frozen=True, slots=True)
class WeekPlan:
    """One class's week: the grid it is drawn on, and what is in it.

    The periods travel with the entries because every caller needs both and reading them
    apart is how a screen renders a seven-row grid against an eight-period day. `days` is
    the school's own week in the school's own order — a client must not sort it, because
    "the first day" is Saturday at some schools and Sunday at others.
    """

    academic_year_code: str
    class_code: str
    term_code: str
    #: The school's working days, in the order the school stated them.
    days: tuple[WorkingDay, ...]
    periods: tuple[TimetablePeriod, ...]
    entries: tuple[TimetableEntry, ...]

    @property
    def teaching_slots(self) -> int:
        """How many slots this week could hold a lesson: teaching periods x open days."""
        return len(self.days) * sum(1 for period in self.periods if period.is_teaching)


class TimetableService:
    """Reads and writes the weekly plan, enforcing what the schema cannot.

    Takes a unit-of-work factory like every other service here, so the rules below can be
    tested against dictionaries rather than against a database.
    """

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    # -- The school's day ---------------------------------------------------

    def list_periods(self, school_code: SchoolCode) -> Sequence[TimetablePeriod]:
        with self._uow_factory() as uow:
            self._require_school(uow, school_code)
            return uow.timetable.list_periods(school_code)

    def set_periods(
        self, school_code: SchoolCode, periods: Sequence[TimetablePeriod]
    ) -> Sequence[TimetablePeriod]:
        """Replace the school's day, refusing to strand a lesson.

        Whole-grid, because "we run seven periods, not eight" is one decision. The check
        that earns its place is the last one: shortening the day would otherwise leave
        lessons timetabled into a period that no longer exists — rows the grid cannot draw,
        which is the worst kind of orphan because nothing is broken enough to notice.
        """
        with self._uow_factory() as uow:
            self._require_school(uow, school_code)
            numbers = [period.period_number for period in periods]
            if len(numbers) != len(set(numbers)):
                raise ValidationError(
                    "each period number may appear once", field="periods"
                )
            wanted = set(numbers)
            in_use = {
                entry.slot.period_number
                for entry in self._entries_for_school(uow, school_code)
            }
            stranded = sorted(in_use - wanted)
            if stranded:
                raise DomainRuleViolation(
                    "period(s) "
                    + ", ".join(str(number) for number in stranded)
                    + " still have lessons timetabled in them; clear those lessons before "
                    "removing the period",
                    field="periods",
                )
            stored = uow.timetable.replace_periods(
                school_code, [replace(period, school_code=str(school_code)) for period in periods]
            )
            uow.commit()
            return stored

    # -- One class's week ---------------------------------------------------

    def week_for_class(
        self,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        term_code: TermCode,
    ) -> WeekPlan:
        """Everything needed to draw one class's timetable, in one transaction.

        The grid, the school's week and the lessons together, for the reason every other
        combined read in this service exists: fetched separately they can disagree, and a
        registrar would see a lesson in a period the grid no longer draws.
        """
        with self._uow_factory() as uow:
            year, school = self._require_year_and_school(uow, academic_year_code)
            self._require_class(uow, academic_year_code, class_code)
            self._require_term(uow, academic_year_code, term_code)
            periods = tuple(uow.timetable.list_periods(SchoolCode(str(school.code))))
            entries = uow.timetable.list_entries(
                academic_year_code, class_code=class_code, term_code=term_code
            )
        return WeekPlan(
            academic_year_code=str(academic_year_code),
            class_code=str(class_code),
            term_code=str(term_code),
            days=tuple(school.working_days),
            periods=periods,
            entries=self._in_week_order(entries, school),
        )

    def entries_for_year(
        self,
        academic_year_code: AcademicYearCode,
        *,
        term_code: TermCode | None = None,
        year_level_code: YearCode | None = None,
    ) -> Sequence[TimetableEntry]:
        """Every lesson in the year, for the whole-school view a clash is spotted in.

        `year_level_code` cuts it down to one grade, which is the only form a
        grade-scoped supervisor may read: the whole-school view is precisely the
        unrelated-grade access their role is bounded away from.
        """
        with self._uow_factory() as uow:
            _, school = self._require_year_and_school(uow, academic_year_code)
            if term_code is not None:
                self._require_term(uow, academic_year_code, term_code)
            entries = uow.timetable.list_entries(
                academic_year_code, term_code=term_code, year_level_code=year_level_code
            )
        return self._in_week_order(entries, school)

    def place(
        self,
        academic_year_code: AcademicYearCode,
        entries: Sequence[TimetableEntry],
    ) -> Sequence[TimetableEntry]:
        """Put these lessons in these slots, all of them or none.

        Every rule this module exists for runs here, before anything is written. They are
        checked over the whole batch rather than per entry so that a thirty-five-slot grid
        with one bad cell is refused as a whole — a partially applied week looks finished
        and is not.
        """
        if not entries:
            return ()
        with self._uow_factory() as uow:
            _, school = self._require_year_and_school(uow, academic_year_code)
            open_days = set(school.working_days)
            grid = {
                period.period_number: period
                for period in uow.timetable.list_periods(SchoolCode(str(school.code)))
            }
            if not grid:
                raise DomainRuleViolation(
                    f"school {school.code} has no timetable periods yet; set the school's "
                    "day before timetabling a lesson into it",
                    field="period_number",
                )

            seen: set[tuple[str, str, str, int]] = set()
            prepared: list[TimetableEntry] = []
            for entry in entries:
                slot = entry.slot
                if slot.key in seen:
                    raise TimetableConflict(
                        f"two lessons were sent for {slot.class_code} on "
                        f"{slot.day_of_week} period {slot.period_number}",
                        field="slot",
                    )
                seen.add(slot.key)

                # Requirement 2: the week is the school's, never a constant.
                if slot.day_of_week not in open_days:
                    raise ValidationError(
                        f"school {school.code} does not open on {slot.day_of_week}; its "
                        "week is " + ", ".join(day.value for day in school.working_days),
                        field="day_of_week",
                    )
                period = grid.get(slot.period_number)
                if period is None:
                    raise ValidationError(
                        f"school {school.code} has no period {slot.period_number}",
                        field="period_number",
                    )
                if not period.is_teaching:
                    raise DomainRuleViolation(
                        f"period {slot.period_number} is not a teaching period",
                        field="period_number",
                    )

                section = self._require_class(uow, academic_year_code, slot.class_code)
                self._require_term(uow, academic_year_code, slot.term_code)

                # Stage 5's rule, one table further on: a subject appears only where it is
                # assigned, and the rung's track is what makes the two sections separate.
                if entry.subject_code is not None:
                    assigned = uow.subjects.list_for_year(
                        academic_year_code,
                        include_inactive=False,
                        year_level_code=section.year_level_code,
                    )
                    if not any(
                        str(subject.code) == str(entry.subject_code)
                        for subject in assigned
                    ):
                        raise DomainRuleViolation(
                            f"{entry.subject_code} is not assigned to "
                            f"{section.year_level_code}, so {slot.class_code} does not "
                            "teach it",
                            field="subject_code",
                        )

                prepared.append(replace(entry, academic_year_code=academic_year_code))

            uow.timetable.upsert_entries(prepared)
            uow.commit()
        return tuple(prepared)

    def clear(
        self,
        academic_year_code: AcademicYearCode,
        slots: Sequence[TimetableSlot],
    ) -> int:
        """Empty these slots; how many lessons were removed.

        Not the same as placing a lesson with no subject. That states "this class has this
        period free"; this states "nobody has planned this slot", and a registrar has to be
        able to say either.
        """
        if not slots:
            return 0
        with self._uow_factory() as uow:
            self._require_year_and_school(uow, academic_year_code)
            removed = uow.timetable.delete_entries(slots)
            uow.commit()
        return removed

    # -- Shared checks ------------------------------------------------------

    @staticmethod
    def _require_school(uow: UnitOfWork, school_code: SchoolCode) -> School:
        school = uow.schools.get(school_code)
        if school is None:
            raise UnknownReference(f"no school {school_code}", field="school_code")
        return school

    @staticmethod
    def _require_year_and_school(
        uow: UnitOfWork, academic_year_code: AcademicYearCode
    ) -> tuple[object, School]:
        """The year and the school that runs it, which every rule below needs.

        Read together because the school is where the week and the period grid come from,
        and a year whose school is missing is a broken foreign key rather than a state a
        caller should be handed as an empty timetable.
        """
        year = uow.academic_years.get(academic_year_code)
        if year is None:
            raise UnknownReference(
                f"no academic year {academic_year_code}", field="academic_year_code"
            )
        school = uow.schools.get(SchoolCode(str(year.school_code)))
        if school is None:
            raise UnknownReference(
                f"no school {year.school_code}", field="school_code"
            )
        return year, school

    @staticmethod
    def _require_class(
        uow: UnitOfWork, academic_year_code: AcademicYearCode, class_code: ClassCode
    ):
        section = uow.class_sections.get(academic_year_code, class_code)
        if section is None:
            raise UnknownReference(
                f"no class {class_code} in {academic_year_code}", field="class_code"
            )
        return section

    @staticmethod
    def _require_term(
        uow: UnitOfWork, academic_year_code: AcademicYearCode, term_code: TermCode
    ):
        term = uow.terms.get(term_code)
        if term is None:
            raise UnknownReference(f"no term {term_code}", field="term_code")
        # The term and the class must belong to the same year, or a lesson would be planned
        # for a stretch of time the class does not exist in.
        if str(term.academic_year_code) != str(academic_year_code):
            raise ValidationError(
                f"term {term_code} belongs to {term.academic_year_code}, not "
                f"{academic_year_code}",
                field="term_code",
            )
        return term

    @staticmethod
    def _entries_for_school(
        uow: UnitOfWork, school_code: SchoolCode
    ) -> Sequence[TimetableEntry]:
        """Every lesson at a school, across its years. Only `set_periods` needs this.

        Across years rather than in the current one because the period grid is the
        school's and outlives any year: shortening the day would strand last year's
        lessons just as surely as this year's, and last year's are the ones nobody would
        think to look at.
        """
        entries: list[TimetableEntry] = []
        for year in uow.academic_years.list_all(school_code):
            entries.extend(uow.timetable.list_entries(year.code))
        return entries

    @staticmethod
    def _in_week_order(
        entries: Sequence[TimetableEntry], school: School
    ) -> tuple[TimetableEntry, ...]:
        """Sort by the school's own week, then period.

        The repository cannot do this: `day_of_week` sorts alphabetically in SQL, which
        puts Friday first and Wednesday second at every school on earth. Only the school
        knows its week begins on Saturday or Sunday, so the ordering is applied here where
        the school is in hand.
        """
        order = {day: index for index, day in enumerate(school.working_days)}
        return tuple(
            sorted(
                entries,
                key=lambda entry: (
                    str(entry.slot.class_code),
                    str(entry.slot.term_code),
                    # A day the school has since stopped opening sorts last rather than
                    # crashing the sort — the lesson is stale, not unrenderable.
                    order.get(entry.slot.day_of_week, len(order)),
                    entry.slot.period_number,
                ),
            )
        )
