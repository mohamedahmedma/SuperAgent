"""The mark sheet: one class, one subject, one term, and the figures a teacher states.

The sibling of `AttendanceService`, one column over, and shaped the same way on purpose —
the two screens a teacher uses every week should not behave differently for no reason:

`sheet` builds the list from the **enrolments** and folds the marks in, so a sheet with
four figures on it shows thirty children and twenty-six blanks rather than four children
who all did well. That is the same ordering decision the register makes, and it exists here
for the same reason: a partly-marked class must not read as a small one.

`record` writes a whole sheet in one transaction and refuses a child who was not in the
class for that term.

**Blank stays blank.** A cell nobody typed into is not sent, and a cell explicitly cleared
is sent as `null` and stored as SQL NULL. Neither is `0`. This is invariant 1 and the whole
of `SubjectGrade`'s docstring, and it survives here by never constructing a `Percentage`
from a missing value — the service refuses to guess what an empty box meant.

**The class is the one she sat in for that term**, resolved through
`resolve_sections_for_term` rather than from her current placement, so a child who moves in
March keeps her Term 1 marks under the room she earned them in. That function is the single
place that rule lives; this module calls it and does not re-derive it.

Nothing here decides *who* may write. The subject boundary — a teacher records their own
subject and no other — is `TeachingService`, checked by the route before this is reached.
Two modules because the question "is this a valid mark for this child" and the question "are
you the person who may state it" have different answers and different failure modes.
"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from sis.application.ports.unit_of_work import UnitOfWork
from sis.application.services.queries import resolve_sections_for_term
from sis.domain.errors import DomainRuleViolation, UnknownReference, ValidationError
from sis.domain.grades import SubjectGrade
from sis.domain.people import Student
from sis.domain.structure import AcademicYear, Subject, Term
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    Percentage,
    StudentNumber,
    SubjectCode,
    TermCode,
)

__all__ = ["MarkSheet", "MarkSheetLine", "MarkSheetService", "StatedMark"]


@dataclass(frozen=True, slots=True)
class StatedMark:
    """One figure a caller is stating, before it is a `SubjectGrade`.

    `percentage` of `None` with `clear=False` means "not stated in this request" and the
    row is left alone; `clear=True` means "erase what is on file", which is a different
    request and has to be spelled differently. Without the second, a screen that sends
    every visible row would wipe every mark it did not happen to have loaded.
    """

    student_number: str
    percentage: float | None = None
    points: float | None = None
    max_points: float | None = None
    clear: bool = False


@dataclass(frozen=True, slots=True)
class MarkSheetLine:
    """One child on the sheet, and her figure for this subject if she has one."""

    student_number: str
    student: Student | None
    grade: SubjectGrade | None = None

    @property
    def is_graded(self) -> bool:
        """Delegates rather than re-deriving: no call site re-implements null-vs-zero."""
        return self.grade is not None and self.grade.is_graded


@dataclass(frozen=True, slots=True)
class MarkSheet:
    """A class, a subject, a term, and every child who sat in it."""

    academic_year_code: str
    class_code: str
    subject_code: str
    term_code: str
    term_is_closed: bool
    subject: Subject | None
    lines: tuple[MarkSheetLine, ...] = ()

    @property
    def size(self) -> int:
        return len(self.lines)

    @property
    def graded(self) -> int:
        """How many carry a stated figure. The rest are awaiting one, not scoring zero."""
        return sum(1 for line in self.lines if line.is_graded)

    @property
    def ungraded(self) -> int:
        return self.size - self.graded


class MarkSheetService:
    """Read a class's sheet for one subject, and record the figures stated on it."""

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def sheet(
        self,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        subject_code: SubjectCode,
        term_code: TermCode,
    ) -> MarkSheet:
        """Every child in the class for that term, with her mark for this subject."""
        with self._uow_factory() as uow:
            context = self._context(uow, academic_year_code, class_code, subject_code, term_code)
            numbers, students = self._roll(uow, context)
            on_file = {
                str(grade.student_number): grade
                for grade in uow.grades.list_for_class(
                    context.section_id, term_code, subject_code=subject_code
                )
            }
            return MarkSheet(
                academic_year_code=str(academic_year_code),
                class_code=str(class_code),
                subject_code=str(subject_code),
                term_code=str(term_code),
                term_is_closed=context.term.is_closed,
                subject=context.subject,
                lines=tuple(
                    MarkSheetLine(
                        student_number=number,
                        student=students.get(number),
                        grade=on_file.get(number),
                    )
                    for number in numbers
                ),
            )

    def record(
        self,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        subject_code: SubjectCode,
        term_code: TermCode,
        marks: Sequence[StatedMark],
    ) -> MarkSheet:
        """Write the stated figures, and answer with the sheet as it now stands.

        Idempotent by `(child, subject, term)`, which is the key the table is unique on —
        so saving the same sheet twice corrects it rather than filing a second set of
        figures beside the first.

        A child not in this class for this term is refused with her number in the message,
        not skipped. Skipping would let a stale screen silently drop a mark a teacher
        believed they had entered, and a mark nobody can see is missing is the worst kind.
        """
        if not marks:
            raise ValidationError(
                "no marks were stated; the sheet has nothing to record", field="marks"
            )

        with self._uow_factory() as uow:
            context = self._context(uow, academic_year_code, class_code, subject_code, term_code)
            if context.term.is_closed:
                raise DomainRuleViolation(
                    f"term {term_code} is closed; reopen it before changing its marks",
                    field="term_code",
                )
            numbers, _ = self._roll(uow, context)
            on_roll = set(numbers)

            strangers = sorted({mark.student_number for mark in marks} - on_roll)
            if strangers:
                raise ValidationError(
                    f"not in {class_code} for {term_code}: {', '.join(strangers)}. A mark "
                    "is filed under the class the child sat in when she earned it.",
                    field="marks",
                )

            writable = [
                SubjectGrade(
                    student_number=mark.student_number,
                    subject_code=subject_code,
                    term_code=term_code,
                    class_section_id=context.section_id,
                    class_code=class_code,
                    # `clear` writes NULL; a stated figure writes a `Percentage`. There is
                    # no third path, and in particular no `or 0` — see the module docstring.
                    percentage=(
                        None
                        if mark.clear or mark.percentage is None
                        else Percentage(mark.percentage)
                    ),
                    points=None if mark.clear else mark.points,
                    max_points=None if mark.clear else mark.max_points,
                )
                for mark in marks
            ]
            uow.grades.upsert_many(writable)
            uow.commit()

        # Read back rather than assembling from what was written, so the sheet a teacher
        # sees after saving is the sheet the database holds.
        return self.sheet(academic_year_code, class_code, subject_code, term_code)

    # -- Internals ----------------------------------------------------------------

    def _context(
        self,
        uow: UnitOfWork,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        subject_code: SubjectCode,
        term_code: TermCode,
    ) -> "_Context":
        """Resolve, and refuse anything that does not exist, one message per thing.

        Every lookup here answers `None` for a code nobody has created, and each is turned
        into its own refusal naming its own field: "no such term" and "no such subject"
        are different mistakes and a caller fixes them differently.
        """
        section_id = uow.class_sections.ids_for(
            [(str(academic_year_code), str(class_code))]
        ).get((str(academic_year_code), str(class_code)))
        if section_id is None:
            raise UnknownReference(
                f"no class {class_code} in academic year {academic_year_code}",
                field="class_code",
            )
        term = uow.terms.get(term_code)
        if term is None:
            raise UnknownReference(f"no term {term_code}", field="term_code")
        year = uow.academic_years.get(academic_year_code)
        if year is None:
            raise UnknownReference(
                f"no academic year {academic_year_code}", field="academic_year_code"
            )
        subject = uow.subjects.get(subject_code, academic_year_code)
        if subject is None:
            raise UnknownReference(
                f"no subject {subject_code} in {academic_year_code}", field="subject_code"
            )
        return _Context(
            section_id=section_id,
            term=term,
            year=year,
            subject=subject,
            class_code=str(class_code),
        )

    @staticmethod
    def _roll(
        uow: UnitOfWork, context: "_Context"
    ) -> tuple[list[str], Mapping[str, Student]]:
        """The children whose class *for this term* is this one.

        Built by asking `resolve_sections_for_term` — the same function the report card
        uses — rather than by taking today's roster, so a mark entered on this sheet and
        the mark read back on a report agree about which room a child sat in. Today's
        roster would disagree with both the moment anybody transfers.

        The candidates are the class's roll on each end of the term's window, and the
        resolver then decides which of them this term actually belongs to. A child who
        left in November appears via the start-of-term end of that window, which is the
        point: she earned a term of marks and a sheet that omitted her could not record
        them.
        """
        starts_on, ends_on = context.term.resolution_window(context.year)
        year_code = AcademicYearCode(str(context.year.code))
        class_code = ClassCode(str(context.class_code))

        candidates: list[StudentNumber] = []
        seen: set[str] = set()
        for day in (ends_on, starts_on):
            for placement in uow.enrolments.roster_on(year_code, class_code, day):
                number = str(placement.student_number)
                if number not in seen:
                    seen.add(number)
                    candidates.append(StudentNumber(number))

        if not candidates:
            return [], {}

        resolved = resolve_sections_for_term(
            uow.enrolments, candidates, context.term, context.year
        )
        # Repositories return the domain ClassSection, whose stable identity is
        # (academic_year_code, code); it deliberately does not expose the database PK.
        # Comparing a non-existent `.id` made every real student disappear from the sheet.
        numbers = sorted(
            number
            for number in seen
            if resolved.get(number) is not None
            and str(resolved[number].academic_year_code) == str(context.year.code)
            and str(resolved[number].code) == context.class_code
        )
        students = uow.students.get_many([StudentNumber(n) for n in numbers])
        return numbers, students


@dataclass(frozen=True, slots=True)
class _Context:
    """What one request resolved to, so the two public methods share one lookup."""

    section_id: int
    term: Term
    year: AcademicYear
    subject: Subject | None
    class_code: str = ""
