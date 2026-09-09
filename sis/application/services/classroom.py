"""The room a child sits in: what it is called, what it studies, and who teaches it.

Three questions a parent asks separately and one answer, because all three hang off the
same fact — **which room she was placed in for the term** — and that fact is resolved
exactly once here.

That is the whole reason this module exists rather than three services. `resolve_section_for_term`
is the single implementation of invariant 2 (a placement is a window, not a column), and
asking it three times in three transactions is how a parent is told her daughter is in 3A,
studies 3B's subjects, and is taught by 3A's teachers — three separately-correct answers
that are collectively wrong. `TimetableService.week_for_student` states the same rule for
the same reason; this is that rule applied to the other three reads.

**What the term does and does not pin.** The term resolves the ROOM: a child who moved
3A->3B in March is in 3A for Term 1, forever, and that is what the timetable and the marks
both already promise. It does **not** pin the STAFFING. `teacher_class_sections` carries no
term and no validity window — both of its write paths delete and re-insert — so who teaches
a room is a present-tense fact about that room and nothing here can honestly report who
taught it last November. The API says so out loud rather than copying a historical guarantee
the schema cannot keep.

**Three empties, and they mean three different things.** None of them is an error:

  no section          she had left, or had not yet joined. A fact about the CHILD.
  section, no subjects  nobody has curated this rung's subject board for this year. About the SCHOOL.
  section, no teachers  nobody has entered this room's staffing. Also about the SCHOOL.

Flattened together they become "your daughter has no teachers", which is false in two of
the three cases and alarming in all of them. The read model keeps them apart and the API
keeps them apart; see `StudentClassroom`.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sis.application.ports.repositories import SectionTeacher
from sis.application.ports.unit_of_work import UnitOfWork
from sis.application.services.queries import resolve_section_for_term
from sis.domain.errors import UnknownReference
from sis.domain.structure import ClassSection, Subject, YearLevel
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    SchoolCode,
    StudentNumber,
    TermCode,
    YearCode,
)

__all__ = ["ClassroomService", "StudentClassroom"]


@dataclass(frozen=True, slots=True)
class StudentClassroom:
    """One child's room for one term, everything about it that a parent may be told.

    `class_section is None` is the only state that says something about the child — no
    placement covered the term. Both lists are then empty because there is no room to ask
    about, and a caller must not read that as "she has no subjects".

    With a section present, each list is independently meaningful when empty: see the
    module docstring. `year_level` rides along because a parent thinks in year groups
    ("Year 3") where a registrar thinks in rooms ("3A"), and it is `None` only when the
    school's ladder is missing the rung its own class points at — a broken reference, not
    a normal state, and reported as absent rather than raised because one unnamed rung must
    not cost a parent the rest of the answer.
    """

    student_number: str
    term_code: str
    class_section: ClassSection | None
    year_level: YearLevel | None = None
    #: The rung's assigned subject board, in the school's own `display_order`. Retired
    #: subjects are excluded: a dropped subject is not one she studies, though it must
    #: still resolve for the marks already stated against it.
    subjects: tuple[Subject, ...] = ()
    #: One entry per (teacher, subject) in the room, so a teacher who takes two subjects
    #: appears twice — which is what a caller renders.
    teachers: tuple[SectionTeacher, ...] = ()

    @property
    def has_class(self) -> bool:
        """Whether a placement covered the term at all."""
        return self.class_section is not None


class ClassroomService:
    """Reads a child's room, its subject board and its staff, in one transaction.

    Takes a unit-of-work factory like every other service here, so the rules above can be
    tested against dictionaries rather than a database.
    """

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def for_student(
        self, student_number: StudentNumber, term_code: TermCode
    ) -> StudentClassroom:
        """Her room for that term, with what it studies and who teaches it.

        Structurally the same read as `TimetableService.week_for_student`, and deliberately
        so: refuse an unknown student, an unknown term and a term whose year is missing,
        then resolve the placement, then gather. A caller that could not tell a typo'd term
        from "she has no class this term" would render the first as an empty classroom and
        send a parent looking for something that was never missing.

        No authorisation happens here. Whether this guardian may be told about this child is
        `QueryService.require_guardian_may_see`, which is where every parent-facing read in
        this service makes that decision, and it runs before this is called.
        """
        with self._uow_factory() as uow:
            if uow.students.get(student_number) is None:
                raise UnknownReference(
                    f"no student {student_number}", field="student_number"
                )
            term = uow.terms.get(term_code)
            if term is None:
                raise UnknownReference(f"no term {term_code}", field="term_code")
            # The term's own year. A term whose year is missing is a broken foreign key
            # rather than a state to answer around.
            year = uow.academic_years.get(
                AcademicYearCode(str(term.academic_year_code))
            )
            if year is None:
                raise UnknownReference(
                    f"no academic year {term.academic_year_code}",
                    field="academic_year_code",
                )

            section = resolve_section_for_term(
                uow.enrolments, student_number, term, year
            )
            if section is None:
                return StudentClassroom(
                    student_number=str(student_number),
                    term_code=str(term_code),
                    class_section=None,
                )

            year_code = AcademicYearCode(str(year.code))
            level_code = YearCode(str(section.year_level_code))
            # The rung, for its human name. Absent rather than fatal: a class pointing at a
            # rung the ladder does not have is broken data, and it must not cost a parent
            # the class name, the subjects and the teachers as well.
            level = uow.year_levels.get(
                level_code, SchoolCode(str(year.school_code))
            )
            # The assignment board, not the year's whole catalogue: a school teaches
            # Physics, but only Secondary sits it. Already in `display_order`.
            subjects = tuple(
                uow.subjects.list_for_year(
                    year_code, include_inactive=False, year_level_code=level_code
                )
            )
            teachers = tuple(
                uow.teachers.teaching_for_section(
                    academic_year_code=year_code,
                    year_level_code=level_code,
                    class_code=ClassCode(str(section.code)),
                )
            )

        return StudentClassroom(
            student_number=str(student_number),
            term_code=str(term_code),
            class_section=section,
            year_level=level,
            subjects=subjects,
            teachers=teachers,
        )
