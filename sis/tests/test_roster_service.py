"""Roster preview and commit, against fake repositories and a fake parser.

`RosterImportService` takes a unit-of-work factory, a parser, a clock and a TTL, so every
rule it enforces is reachable here with four dicts and a literal `b"roster"`. No engine,
no fixture, no temp xlsx: if this file ever needs one, a use case has started importing
something below the application layer.

Two assertions recur and both are deliberate. Preview is checked by what is *absent* from
the stores — "no student was created" is not a report to read, it is a dict that stayed
empty — and commit is checked against the stores rather than against its own result, since
a service reporting a write it never made is exactly the failure worth catching.
"""
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from itertools import count

import pytest

from sis.application.dto import (
    ImportCommitResult,
    ImportPreviewResult,
    ParsedRosterRow,
    ParseResult,
    RosterCommitCommand,
    RosterPreviewCommand,
    RowCode,
    RowOutcome,
)
from sis.application.services.roster_import import RosterImportService
from sis.domain.errors import ImportBatchAlreadyCommitted
from sis.domain.imports import ImportBatch
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import AcademicYear, ClassSection
from sis.domain.value_objects import AcademicYearCode, ClassCode, StudentNumber

YEAR = AcademicYearCode("2025-2026")
YEAR_STARTS = date(2025, 9, 1)
TERM_1_DAY = date(2025, 11, 5)
TRANSFER_DAY = date(2026, 3, 10)
NOW = datetime(2025, 10, 1, 9, 0, tzinfo=UTC)


# --- fakes -----------------------------------------------------------------


class _Parser:
    """Returns a canned parse. Never sees the bytes, which is the port's whole contract."""

    def __init__(self, result: ParseResult[ParsedRosterRow]) -> None:
        self._result = result

    def parse(self, content: bytes, filename: str) -> ParseResult[ParsedRosterRow]:
        return self._result


class _Years:
    def __init__(self, year: AcademicYear) -> None:
        self.rows = {str(year.code): year}

    def get(self, code: AcademicYearCode) -> AcademicYear | None:
        return self.rows.get(str(code))


class _Sections:
    def __init__(self, *sections: ClassSection) -> None:
        self.rows = {s.identity: s for s in sections}

    def get_many(
        self, keys: Collection[tuple[str, str]]
    ) -> Mapping[tuple[str, str], ClassSection]:
        return {key: self.rows[key] for key in keys if key in self.rows}


class _Students:
    def __init__(self, *students: Student) -> None:
        self.saved = {str(s.student_number): s for s in students}
        self.upsert_calls = 0

    def get_many(self, numbers: Collection[StudentNumber]) -> Mapping[str, Student]:
        return {str(n): self.saved[str(n)] for n in numbers if str(n) in self.saved}

    def upsert_many(self, students: Sequence[Student]) -> Mapping[str, bool]:
        self.upsert_calls += 1
        flags: dict[str, bool] = {}
        for student in students:
            number = str(student.student_number)
            flags[number] = number not in self.saved
            self.saved[number] = student
        return flags


class _Enrolments:
    """Placements as a flat list, because that is what invariant 2 says they are.

    A dict keyed by student number would be the very `students.class_code` column the
    domain refuses, and a fake shaped that way cannot fail the transfer test — it would
    make double-enrolment unrepresentable instead of detectable.
    """

    def __init__(self, *enrolments: ClassEnrolment) -> None:
        self.rows = list(enrolments)
        self.closed: list[tuple[str, date]] = []
        self.upsert_calls = 0

    def list_for_students(
        self, numbers: Collection[StudentNumber]
    ) -> Mapping[str, Sequence[ClassEnrolment]]:
        wanted = {str(n) for n in numbers}
        found: dict[str, list[ClassEnrolment]] = {}
        for row in self.rows:
            number = str(row.student_number)
            if number in wanted:
                found.setdefault(number, []).append(row)
        return found

    def close_open_enrolment(
        self, student_id: StudentNumber, *, ends_on: date
    ) -> ClassEnrolment | None:
        number = str(student_id)
        for index, row in enumerate(self.rows):
            if str(row.student_number) == number and row.is_open:
                self.rows[index] = replace(row, ends_on=ends_on)
                self.closed.append((number, ends_on))
                return self.rows[index]
        return None

    def upsert_many(
        self, enrolments: Sequence[ClassEnrolment]
    ) -> Mapping[tuple[str, str, str, date], bool]:
        self.upsert_calls += 1
        flags: dict[tuple[str, str, str, date], bool] = {}
        for enrolment in enrolments:
            key = _key(enrolment)
            match = next((i for i, r in enumerate(self.rows) if _key(r) == key), None)
            flags[key] = match is None
            if match is None:
                self.rows.append(enrolment)
            else:
                self.rows[match] = enrolment
        return flags


class _Imports:
    def __init__(self) -> None:
        self.batches: dict[str, ImportBatch] = {}
        self.rows: dict[str, list[object]] = {}

    def add(self, batch: ImportBatch, rows: Sequence[object]) -> None:
        self.batches[batch.batch_id] = batch
        self.rows[batch.batch_id] = list(rows)

    def get(self, batch_id: str) -> ImportBatch | None:
        return self.batches.get(batch_id)

    def save(self, batch: ImportBatch) -> ImportBatch:
        self.batches[batch.batch_id] = batch
        return batch

    def list_rows(self, batch_id: str, **_: object) -> Sequence[object]:
        return list(self.rows.get(batch_id, ()))

    def replace_rows(self, batch_id: str, rows: Sequence[object]) -> None:
        self.rows[batch_id] = list(rows)


class _Uow:
    def __init__(self, students: _Students, enrolments: _Enrolments) -> None:
        self.academic_years = _Years(
            AcademicYear(
                code=YEAR,
                school_code="MAIN",
                name_en="2025-2026",
                name_ar="٢٠٢٥-٢٠٢٦",
                starts_on=YEAR_STARTS,
                ends_on=date(2026, 6, 30),
            )
        )
        self.class_sections = _Sections(_section("3A"), _section("3B"))
        self.students = students
        self.enrolments = enrolments
        self.imports = _Imports()
        self.commits = 0

    def __enter__(self) -> "_Uow":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1


# --- builders --------------------------------------------------------------


def _key(enrolment: ClassEnrolment) -> tuple[str, str, str, date]:
    return (
        str(enrolment.student_number),
        str(enrolment.academic_year_code),
        str(enrolment.class_code),
        enrolment.starts_on,
    )


def _section(code: str) -> ClassSection:
    return ClassSection(
        code=ClassCode(code),
        academic_year_code=YEAR,
        year_level_code="Y3",
        name_en=f"Year 3 {code[-1]}",
        name_ar=f"الثالث {code[-1]}",
    )


def _row(
    line: int,
    number: str,
    *,
    name_en: str = "",
    class_code: str | None = "3A",
    starts_on: date | None = None,
) -> ParsedRosterRow:
    return ParsedRosterRow(
        line_number=line,
        student_number=StudentNumber(number),
        full_name_ar="",
        full_name_en=name_en or f"Child {number}",
        class_code=ClassCode(class_code) if class_code else None,
        starts_on=starts_on,
    )


def _school(
    *, students: Collection[Student] = (), enrolments: Collection[ClassEnrolment] = ()
) -> _Uow:
    return _Uow(_Students(*students), _Enrolments(*enrolments))


def _service(
    uow: _Uow,
    rows: Sequence[ParsedRosterRow],
    *,
    diagnostics: Sequence[RowOutcome] = (),
) -> RosterImportService:
    ids = count(1)
    return RosterImportService(
        lambda: uow,
        _Parser(
            ParseResult(rows=tuple(rows), diagnostics=tuple(diagnostics), total_lines=len(rows))
        ),
        preview_ttl=timedelta(hours=1),
        clock=lambda: NOW,
        batch_ids=lambda: f"batch-{next(ids)}",
    )


def _preview(service: RosterImportService, **kwargs: object) -> ImportPreviewResult:
    return service.preview(
        RosterPreviewCommand(
            academic_year_code=YEAR,
            filename="roster.csv",
            content=b"roster",
            actor="registrar",
            **kwargs,  # type: ignore[arg-type]
        )
    )


def _commit(service: RosterImportService, batch_id: str) -> ImportCommitResult:
    return service.commit(RosterCommitCommand(batch_id=batch_id, actor="registrar"))


# --- tests -----------------------------------------------------------------


def test_preview_writes_no_student_and_no_placement() -> None:
    """The half of decision 4 that cannot be proved by reading the result.

    A preview that reports "2 will be created" while having created them is
    indistinguishable from a correct one at the API boundary; only the stores tell the
    truth, so they are what is asserted here.
    """
    uow = _school()
    service = _service(uow, [_row(2, "0071"), _row(3, "0072")])

    result = _preview(service)

    assert result.ok_count == 2
    assert uow.students.saved == {}
    assert uow.students.upsert_calls == 0
    assert uow.enrolments.rows == []
    assert uow.enrolments.upsert_calls == 0
    assert uow.imports.batches[result.batch_id].writable_rows == 2


def test_commit_applies_the_previewed_rows() -> None:
    uow = _school()
    service = _service(uow, [_row(2, "0071"), _row(3, "0072")])
    preview = _preview(service)

    result = _commit(service, preview.batch_id)

    assert result.ok_count == 2
    assert sorted(uow.students.saved) == ["0071", "0072"]
    assert [str(e.class_code) for e in uow.enrolments.rows] == ["3A", "3A"]
    # The placement starts when the school year did, not when the file was uploaded:
    # importing in November must not record every child as having joined in November.
    assert {e.starts_on for e in uow.enrolments.rows} == {YEAR_STARTS}
    assert all(e.is_open for e in uow.enrolments.rows)


def test_a_number_repeated_inside_the_file_is_written_once() -> None:
    uow = _school()
    service = _service(uow, [_row(2, "0071"), _row(3, "0071"), _row(4, "0072")])

    preview = _preview(service)
    result = _commit(service, preview.batch_id)

    assert preview.count(RowCode.DUPLICATE_IN_FILE) == 1
    assert [row.line for row in preview.failures()] == [3]
    assert result.ok_count == 2
    assert sorted(uow.students.saved) == ["0071", "0072"]
    assert len(uow.enrolments.rows) == 2


def test_a_number_already_belonging_to_another_child_is_refused() -> None:
    """Refused rather than overwritten, and the cost of that is the point.

    A rejected line costs the registrar one edit. A silent overwrite reassigns the key
    every grade, enrolment and `records/` guardian link hangs from to a different human
    being, and nothing anywhere reports it.
    """
    uow = _school(students=[Student(student_number="0071", full_name_ar="", full_name_en="Sara Ali")])
    service = _service(uow, [_row(2, "0071", name_en="Noor Hassan"), _row(3, "0072")])

    preview = _preview(service)
    result = _commit(service, preview.batch_id)

    assert preview.count(RowCode.DUPLICATE_EXISTING) == 1
    assert result.ok_count == 1
    assert uow.students.saved["0071"].full_name_en == "Sara Ali"
    assert [str(e.student_number) for e in uow.enrolments.rows] == ["0072"]


def test_a_row_naming_an_unknown_class_writes_nothing() -> None:
    uow = _school()
    service = _service(uow, [_row(2, "0071", class_code="9Z")])

    preview = _preview(service)
    result = _commit(service, preview.batch_id)

    assert preview.count(RowCode.UNKNOWN_CLASS) == 1
    assert preview.failures()[0].field == "class_code"
    assert result.ok_count == 0
    assert uow.students.saved == {}
    assert uow.enrolments.rows == []


def test_a_transfer_closes_the_prior_placement_instead_of_doubling_it() -> None:
    """Invariant 2: 3A -> 3B in March, and Term 1 still resolves to 3A afterwards.

    The last assertion is the one that matters. Closing the old row rather than editing
    its class is what keeps "which class was she in during Term 1" answerable in June;
    a single mutable column can say 3A or 3B, and both readings are wrong once she has
    moved.
    """
    child = Student(student_number="0071", full_name_ar="", full_name_en="Child 0071")
    uow = _school(
        students=[child],
        enrolments=[
            ClassEnrolment(
                student_number="0071",
                academic_year_code=YEAR,
                class_code=ClassCode("3A"),
                starts_on=YEAR_STARTS,
            )
        ],
    )
    service = _service(uow, [_row(2, "0071", class_code="3B", starts_on=TRANSFER_DAY)])

    preview = _preview(service)
    assert uow.enrolments.closed == []  # still nothing written

    result = _commit(service, preview.batch_id)

    assert result.ok_count == 1
    assert uow.enrolments.closed == [("0071", date(2026, 3, 9))]
    assert len(uow.enrolments.rows) == 2
    assert [e for e in uow.enrolments.rows if e.is_open][0].class_code == ClassCode("3B")
    covering = [e for e in uow.enrolments.rows if e.covers(TERM_1_DAY)]
    assert [str(e.class_code) for e in covering] == ["3A"]


def test_committing_the_same_batch_twice_is_refused() -> None:
    uow = _school()
    service = _service(uow, [_row(2, "0071")])
    preview = _preview(service)
    _commit(service, preview.batch_id)

    with pytest.raises(ImportBatchAlreadyCommitted) as raised:
        _commit(service, preview.batch_id)

    assert raised.value.code == "batch_already_committed"
    assert len(uow.enrolments.rows) == 1
    assert uow.enrolments.upsert_calls == 1


def test_reimporting_the_same_file_writes_nothing_a_second_time() -> None:
    """The double-click's other shape: a fresh preview of a file already committed.

    Every row re-evaluates to `unchanged`, so the commit sends no upsert at all rather
    than stacking a second placement on top of the first or re-dating the one on file.
    """
    uow = _school()
    service = _service(uow, [_row(2, "0071"), _row(3, "0072")])
    _commit(service, _preview(service).batch_id)

    result = _commit(service, _preview(service).batch_id)

    assert result.ok_count == 2
    assert len(uow.enrolments.rows) == 2
    assert uow.enrolments.upsert_calls == 1
    assert uow.students.upsert_calls == 1


def test_one_bad_row_does_not_discard_the_good_ones() -> None:
    uow = _school()
    service = _service(
        uow,
        [_row(2, "0071"), _row(3, "0072", class_code="9Z"), _row(4, "0073")],
        diagnostics=[
            RowOutcome(line=5, code=RowCode.MISSING_STUDENT_NUMBER, message="no number in this row")
        ],
    )

    preview = _preview(service)
    result = _commit(service, preview.batch_id)

    assert [row.line for row in preview.rows] == [2, 3, 4, 5]
    assert result.ok_count == 2
    assert result.count(RowCode.UNKNOWN_CLASS) == 1
    # The parser's own diagnostic survives preview and commit unchanged; it is a row's
    # verdict, not a batch failure.
    assert result.count(RowCode.MISSING_STUDENT_NUMBER) == 1
    assert sorted(uow.students.saved) == ["0071", "0073"]
    assert len(uow.enrolments.rows) == 2
