"""`GradeImportService` against fakes: no database, no clock, no file on disk.

These tests exist to hold invariant 1 in place across a boundary where it is easiest to
lose. A blank mark is `None`, and `None` has to survive being written into an import
batch as a payload value, read back out of that batch by `commit`, rebuilt into a row and
finally stored — four hops, any one of which could substitute a zero and produce a stored
grade that is byte-identical to one a teacher awarded. `test_blank_grade_...` and
`test_earned_zero_...` are deliberately a pair: neither alone would catch a service that
collapsed the two cases, because a passing "blank stays blank" test is satisfied by a
service that also refuses to store a genuine zero.

Everything below is a fake with only the methods this service calls. Implementing the
whole Protocol would be a second repository to maintain, and the Protocols are structural
precisely so a test does not have to.
"""
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, date, datetime

import pytest

from sis.application.dto import (
    GradeCommitCommand,
    GradePreviewCommand,
    ParsedGradeRow,
    ParseResult,
    RowCode,
)
from sis.application.dto.common import RowOutcome as RowDiagnostic
from sis.application.ports.repositories import ClassSectionKey, GradeKey
from sis.application.services.grade_import import GradeImportService
from sis.domain.grades import SubjectGrade
from sis.domain.imports import ImportBatch, ImportRow, RowOutcome
from sis.domain.people import Student
from sis.domain.structure import ClassSection, Subject, Term
from sis.domain.value_objects import (
    ClassCode,
    Percentage,
    StudentNumber,
    SubjectCode,
    TermCode,
)

YEAR = "2025-2026"
TERM = "2026-T1"
CLASS = "3A"
SECTION_ID = 41
BATCH_ID = "batch-0001"
# Fixed and timezone-aware. `ImportBatch` rejects a naive datetime, and a preview whose
# TTL is measured against a real clock is a test that expires while it runs.
NOW = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- fakes


class FakeTerms:
    def __init__(self, *terms: Term) -> None:
        self._by_code = {str(t.code): t for t in terms}

    def get_many(self, codes: Collection[TermCode]) -> Mapping[str, Term]:
        return {str(c): self._by_code[str(c)] for c in codes if str(c) in self._by_code}


class FakeStudents:
    def __init__(self, *students: Student) -> None:
        self._by_number = {str(s.student_number): s for s in students}

    def get_many(self, numbers: Collection[StudentNumber]) -> Mapping[str, Student]:
        return {
            str(n): self._by_number[str(n)] for n in numbers if str(n) in self._by_number
        }


class FakeSubjects:
    def __init__(self, *subjects: Subject) -> None:
        self._by_code = {str(s.code): s for s in subjects}

    def get_many(self, codes: Collection[SubjectCode]) -> Mapping[str, Subject]:
        return {str(c): self._by_code[str(c)] for c in codes if str(c) in self._by_code}


class FakeEnrolments:
    """Placement lookup, ignoring the date on purpose.

    The date arithmetic of invariant 2 belongs to `resolve_sections_for_term` and is
    tested where it lives; answering every day with the same class here keeps these
    assertions about what happens to the *mark*.
    """

    def __init__(self, placements: Mapping[str, ClassSection]) -> None:
        self._placements = dict(placements)

    def class_sections_on(
        self, student_ids: Collection[StudentNumber], on_date: date
    ) -> Mapping[str, ClassSection]:
        return {
            str(s): self._placements[str(s)]
            for s in student_ids
            if str(s) in self._placements
        }


class FakeClassSections:
    def __init__(self, ids: Mapping[ClassSectionKey, int]) -> None:
        self._ids = dict(ids)

    def ids_for(
        self, keys: Collection[ClassSectionKey]
    ) -> Mapping[ClassSectionKey, int]:
        return {k: self._ids[k] for k in keys if k in self._ids}


class FakeGrades:
    """The stored marks. `stored` is what the assertions read — this fake is the table."""

    def __init__(self, *grades: SubjectGrade) -> None:
        self.stored: dict[GradeKey, SubjectGrade] = {g.identity: g for g in grades}

    def get_many(self, keys: Collection[GradeKey]) -> Mapping[GradeKey, SubjectGrade]:
        return {k: self.stored[k] for k in keys if k in self.stored}

    def upsert_many(self, grades: Sequence[SubjectGrade]) -> Mapping[GradeKey, bool]:
        created: dict[GradeKey, bool] = {}
        for grade in grades:
            created[grade.identity] = grade.identity not in self.stored
            self.stored[grade.identity] = grade
        return created


class FakeImports:
    def __init__(self) -> None:
        self.batches: dict[str, ImportBatch] = {}
        self.rows: dict[str, list[ImportRow]] = {}

    def add(self, batch: ImportBatch, rows: Sequence[ImportRow]) -> None:
        self.batches[batch.batch_id] = batch
        self.rows[batch.batch_id] = list(rows)

    def get(self, batch_id: str) -> ImportBatch | None:
        return self.batches.get(batch_id)

    def save(self, batch: ImportBatch) -> ImportBatch:
        self.batches[batch.batch_id] = batch
        return batch

    def list_rows(
        self,
        batch_id: str,
        *,
        outcomes: Collection[RowOutcome] | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> Sequence[ImportRow]:
        rows = self.rows.get(batch_id, [])
        if outcomes is not None:
            rows = [r for r in rows if r.outcome in outcomes]
        return rows[offset : None if limit is None else offset + limit]

    def replace_rows(self, batch_id: str, rows: Sequence[ImportRow]) -> None:
        self.rows[batch_id] = list(rows)


class FakeUnitOfWork:
    """One in-memory transaction. Returned by the factory *unchanged* between calls.

    Preview and commit are two `with` blocks, and in production what carries state
    between them is the database. Here it is this object, so the factory hands back the
    same instance rather than a fresh one — a new fake per call would lose the batch
    preview wrote and make every commit test fail as `ImportBatchNotFound`.
    """

    def __init__(
        self,
        *,
        terms: FakeTerms,
        students: FakeStudents,
        subjects: FakeSubjects,
        enrolments: FakeEnrolments,
        class_sections: FakeClassSections,
        grades: FakeGrades,
    ) -> None:
        self.terms = terms
        self.students = students
        self.subjects = subjects
        self.enrolments = enrolments
        self.class_sections = class_sections
        self.grades = grades
        self.imports = FakeImports()
        self.commits = 0

    def __enter__(self) -> "FakeUnitOfWork":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        return None


class FakeGradeParser:
    """Hands back fixed rows. The bytes are ignored: reading xlsx is not under test."""

    def __init__(
        self,
        rows: Sequence[ParsedGradeRow],
        diagnostics: Sequence[RowDiagnostic] = (),
    ) -> None:
        self._rows = list(rows)
        self._diagnostics = list(diagnostics)

    def parse(self, content: bytes, filename: str) -> ParseResult[ParsedGradeRow]:
        return ParseResult(
            rows=self._rows,
            diagnostics=self._diagnostics,
            total_lines=len(self._rows) + len(self._diagnostics),
        )


# --------------------------------------------------------------------------- world


def _term(code: str = TERM, *, is_closed: bool = False) -> Term:
    return Term(
        code=code,
        academic_year_code=YEAR,
        name_en="Term 1",
        name_ar="الفصل الأول",
        starts_on=date(2025, 9, 1),
        ends_on=date(2025, 12, 15),
        sequence=1,
        is_closed=is_closed,
    )


def _section() -> ClassSection:
    return ClassSection(
        code=CLASS,
        academic_year_code=YEAR,
        year_level_code="Y3",
        name_en="Class 3A",
        name_ar="الصف الثالث أ",
    )


@pytest.fixture
def uow() -> FakeUnitOfWork:
    """Two children, one subject, one term, both children placed in 3A."""
    section = _section()
    return FakeUnitOfWork(
        terms=FakeTerms(_term()),
        students=FakeStudents(
            Student("0071", "سلمى أحمد", "Salma Ahmed"),
            Student("0072", "ليان محمد", "Layan Mohamed"),
        ),
        subjects=FakeSubjects(Subject(code="MATH", name_en="Maths", name_ar="رياضيات")),
        enrolments=FakeEnrolments({"0071": section, "0072": section}),
        class_sections=FakeClassSections({(YEAR, CLASS): SECTION_ID}),
        grades=FakeGrades(),
    )


def _service(uow: FakeUnitOfWork, rows: Sequence[ParsedGradeRow]) -> GradeImportService:
    return GradeImportService(
        lambda: uow,
        FakeGradeParser(rows),
        clock=lambda: NOW,
        new_batch_id=lambda: BATCH_ID,
    )


def _row(
    line: int,
    *,
    student: str = "0071",
    subject: str = "MATH",
    term: str | None = None,
    percentage: Percentage | None = None,
    points: float | None = None,
    max_points: float | None = None,
) -> ParsedGradeRow:
    return ParsedGradeRow(
        line_number=line,
        student_number=StudentNumber(student),
        subject_code=SubjectCode(subject),
        term_code=None if term is None else TermCode(term),
        percentage=percentage,
        points=points,
        max_points=max_points,
    )


def _preview_command() -> GradePreviewCommand:
    return GradePreviewCommand(
        term_code=TermCode(TERM),
        filename="term1-maths.csv",
        content=b"student,mark\n",
        actor="registrar@school",
    )


def _codes(result: object) -> list[RowCode]:
    return [row.code for row in result.rows]  # type: ignore[attr-defined]


def _stored(uow: FakeUnitOfWork, student: str = "0071") -> SubjectGrade:
    return uow.grades.stored[(student, "MATH", TERM)]


# --------------------------------------------------------------------------- tests


def test_blank_grade_stays_none_through_preview_and_commit(uow: FakeUnitOfWork) -> None:
    """Invariant 1, the load-bearing one: a blank cell is `None` and never `0.0`.

    Asserted as both `is None` and `!= 0` on purpose. The first alone passes for a
    service that stores `Percentage(0)` only if a later reader happens to normalise it
    back; the second states the fact a parent would be told — this child did not score
    zero, this child has not been marked.
    """
    service = _service(uow, [_row(2, percentage=None)])

    preview = service.preview(_preview_command())
    payload = preview.rows[0].payload

    assert _codes(preview) == [RowCode.OK]
    assert payload["percentage"] is None
    assert payload["percentage"] != 0
    assert payload["is_graded"] is False

    service.commit(GradeCommitCommand(batch_id=preview.batch_id, actor="registrar"))
    grade = _stored(uow)

    assert grade.percentage is None
    assert grade.percentage != 0
    assert grade.percentage != 0.0
    assert grade.is_graded is False
    assert grade.is_awaiting_grade is True
    assert grade.is_zero is False
    # The only substitution path is the named one, and it is the caller's choice.
    assert grade.value_or(-1.0) == -1.0


def test_earned_zero_is_stored_and_reads_back_as_zero(uow: FakeUnitOfWork) -> None:
    """The sibling of the test above: a teacher's zero is a mark and must survive as one.

    Without this, "never store 0.0" is satisfiable by a service that drops every zero,
    and a child who genuinely scored nothing would be reported as unmarked.
    """
    service = _service(uow, [_row(2, percentage=Percentage(0))])

    preview = service.preview(_preview_command())
    assert preview.rows[0].payload["percentage"] == 0.0
    assert preview.rows[0].payload["is_graded"] is True

    service.commit(GradeCommitCommand(batch_id=preview.batch_id, actor="registrar"))
    grade = _stored(uow)

    assert grade.percentage == Percentage(0)
    assert grade.percentage is not None
    assert grade.percentage.value == 0.0
    assert grade.is_graded is True
    assert grade.is_zero is True
    assert grade.is_awaiting_grade is False


def test_blank_and_zero_in_one_file_stay_apart(uow: FakeUnitOfWork) -> None:
    """The two cases in one upload, which is how a real sheet carries them."""
    service = _service(
        uow,
        [_row(2, student="0071", percentage=None), _row(3, student="0072", percentage=Percentage(0))],
    )

    preview = service.preview(_preview_command())
    service.commit(GradeCommitCommand(batch_id=preview.batch_id, actor="registrar"))

    blank = _stored(uow, "0071")
    zero = _stored(uow, "0072")
    assert blank.percentage is None
    assert zero.percentage == Percentage(0)
    assert blank.percentage != zero.percentage
    assert (blank.is_zero, zero.is_zero) == (False, True)


def test_blank_cell_does_not_erase_a_mark_already_on_file(uow: FakeUnitOfWork) -> None:
    """The other half of invariant 1: a blank is "no statement", not "clear this".

    A school re-uploading one subject's sheet with the rest of the columns empty would
    otherwise wipe the term, and the wipe reads downstream as marks never entered.
    """
    uow.grades.stored[("0071", "MATH", TERM)] = SubjectGrade(
        student_number=StudentNumber("0071"),
        subject_code=SubjectCode("MATH"),
        term_code=TermCode(TERM),
        class_section_id=SECTION_ID,
        class_code=ClassCode(CLASS),
        percentage=Percentage(74),
    )
    service = _service(uow, [_row(2, percentage=None)])

    preview = service.preview(_preview_command())
    result = service.commit(
        GradeCommitCommand(batch_id=preview.batch_id, actor="registrar")
    )

    assert _codes(result) == [RowCode.OK]
    assert _stored(uow).percentage == Percentage(74)


def test_points_and_max_points_are_accepted_instead_of_a_percentage(
    uow: FakeUnitOfWork,
) -> None:
    """"17 out of 20" is a stated figure; the raw pair is kept beside the restatement."""
    service = _service(uow, [_row(2, points=17.0, max_points=20.0)])

    preview = service.preview(_preview_command())
    service.commit(GradeCommitCommand(batch_id=preview.batch_id, actor="registrar"))
    grade = _stored(uow)

    assert _codes(preview) == [RowCode.OK]
    assert grade.percentage == Percentage(85.0)
    assert (grade.points, grade.max_points) == (17.0, 20.0)
    assert grade.is_graded is True


def test_a_figure_out_of_range_is_rejected_with_its_own_code(uow: FakeUnitOfWork) -> None:
    """17 out of 15 is a conversation about the scale, not a mistyped cell."""
    service = _service(uow, [_row(2, points=17.0, max_points=15.0)])

    preview = service.preview(_preview_command())

    assert _codes(preview) == [RowCode.GRADE_OUT_OF_RANGE]
    assert preview.rows[0].field == "percentage"

    service.commit(GradeCommitCommand(batch_id=preview.batch_id, actor="registrar"))
    assert uow.grades.stored == {}


def test_unknown_student_subject_and_term_each_get_their_own_code(
    uow: FakeUnitOfWork,
) -> None:
    """Four rows, three distinct rejections, and the good one still written.

    The counts per kind are the whole diagnostic — "3 rows name a subject that does not
    exist" and "3 rows name a student who does not exist" are different afternoons of
    work. And invariant 4 in the same assertion: one bad row must never discard the good
    ones, so line 5 is stored while lines 2-4 are reported.
    """
    service = _service(
        uow,
        [
            _row(2, term="2026-T9"),
            _row(3, student="9999"),
            _row(4, subject="PHYS"),
            _row(5, percentage=Percentage(88)),
        ],
    )

    preview = service.preview(_preview_command())
    assert _codes(preview) == [
        RowCode.UNKNOWN_TERM,
        RowCode.UNKNOWN_STUDENT,
        RowCode.UNKNOWN_SUBJECT,
        RowCode.OK,
    ]
    assert [row.field for row in preview.rows[:3]] == [
        "term_code",
        "student_number",
        "subject_code",
    ]

    result = service.commit(
        GradeCommitCommand(batch_id=preview.batch_id, actor="registrar")
    )

    assert result.rejected_count == 3
    assert list(uow.grades.stored) == [("0071", "MATH", TERM)]
    assert _stored(uow).percentage == Percentage(88)
