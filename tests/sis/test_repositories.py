"""Repository tests against a real database, built by alembic.

These are deliberately not unit tests. Every service above this layer is tested with
fake repositories precisely so it needs no database, and that leaves exactly three
things unproven — the three this file asserts. A fake cannot insert a duplicate,
because a dict key collapses one; a fake cannot turn `None` into `0.0`, because there is
no column to coalesce it in; and a fake cannot get a date-window `WHERE` clause wrong,
because there is no SQL. Those failures live in the translation between domain objects
and rows, so they are only observable against a real engine.

The schema comes from `alembic upgrade head` and never from `create_all` (invariant 8).
Building it from ORM metadata instead would test a schema no deployment has ever run:
the migration is the artefact that reaches production, so it is the one the tests stand
on. A column a migration forgot would pass a metadata-built suite and fail the school.
"""
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from sis.config import reset_settings_cache
from sis.domain.grades import SubjectGrade
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import AcademicYear, ClassSection, School, Subject, Term, YearLevel
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    Percentage,
    StudentNumber,
    SubjectCode,
    TermCode,
)
from sis.infrastructure.db.session import get_engine, reset_engine
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "sis" / "alembic.ini"

YEAR = AcademicYearCode("2025-2026")
YEAR_STARTS = date(2025, 9, 1)
YEAR_ENDS = date(2026, 6, 30)

# Term 1 sits wholly before the March transfer below, which is what makes it the
# interesting question: it is the term whose answer a `students.class_code` column
# would silently rewrite.
TERM_1 = TermCode("2026-T1")
TERM_1_STARTS = date(2025, 9, 1)
TERM_1_ENDS = date(2025, 12, 15)

TRANSFER_ON = date(2026, 3, 15)
STUDENT = StudentNumber("S001")

# `ends_on` is a child's last day in a class, not the day after, so a placement opening
# on the 15th closes the previous one on the 14th.
_ONE_DAY = timedelta(days=1)


@pytest.fixture
def sis_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """An empty, migrated database wired into the process-wide engine."""
    url = f"sqlite:///{(tmp_path / 'sis.db').as_posix()}"
    monkeypatch.setenv("SIS_DATABASE_URL", url)
    # Both caches, in this order: clearing the settings alone changes nothing, because
    # the engine already holds a connection to whichever database ran before.
    reset_settings_cache()
    reset_engine()

    command.upgrade(Config(str(_ALEMBIC_INI)), "head")
    yield

    reset_engine()
    reset_settings_cache()


def _query(sql: str) -> list[Any]:
    """Read rows outside the ORM, so an assertion sees the column and not the mapper."""
    with get_engine().connect() as connection:
        return list(connection.execute(text(sql)))


def _count(table: str) -> int:
    return _query(f"SELECT COUNT(*) FROM {table}")[0][0]


def _seed_structure() -> None:
    """A year, a rung, two sections and one child — the cast every test below needs."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.schools.upsert_many([School(code="MAIN", name_en="Main School", name_ar="المدرسة")])
        uow.academic_years.upsert_many(
            [
                AcademicYear(
                    code=YEAR,
                    school_code="MAIN",
                    name_en="2025-2026",
                    name_ar="٢٠٢٥-٢٠٢٦",
                    starts_on=YEAR_STARTS,
                    ends_on=YEAR_ENDS,
                    is_current=True,
                )
            ]
        )
        uow.year_levels.upsert_many(
            [YearLevel(code="3", school_code="MAIN", name_en="Year 3", name_ar="السنة الثالثة", display_order=3)]
        )
        uow.class_sections.upsert_many(
            [
                ClassSection(
                    code=code,
                    academic_year_code=YEAR,
                    year_level_code="3",
                    name_en=f"Year 3 {code[-1]}",
                    name_ar=f"الثالث {code[-1]}",
                )
                for code in ("3A", "3B")
            ]
        )
        uow.students.upsert_many(
            [Student(student_number=STUDENT, full_name_ar="ليلى أحمد", full_name_en="Layla Ahmed")]
        )
        uow.commit()


def _seed_grade_references() -> int:
    """Add the term and subjects a mark points at; return the id of section 3A."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.terms.upsert_many(
            [
                Term(
                    code=TERM_1,
                    academic_year_code=YEAR,
                    name_en="Term 1",
                    name_ar="الفصل الأول",
                    starts_on=TERM_1_STARTS,
                    ends_on=TERM_1_ENDS,
                    sequence=1,
                )
            ]
        )
        uow.subjects.upsert_many(
            [
                Subject(
                    code="MATH",
                    academic_year_code=YEAR,
                    name_en="Mathematics",
                    name_ar="الرياضيات",
                ),
                Subject(
                    code="SCI",
                    academic_year_code=YEAR,
                    name_en="Science",
                    name_ar="العلوم",
                ),
            ]
        )
        section_id = uow.class_sections.ids_for([(str(YEAR), "3A")])[(str(YEAR), "3A")]
        uow.commit()
    return section_id


def _transfer_3a_to_3b() -> None:
    """September in 3A, March in 3B — built the way a transfer is actually performed.

    Written through `close_open_enrolment` plus a second insert rather than by hand,
    because the assertions below are only worth anything if the history they read was
    produced by the code path a registrar's transfer goes through. Two rows written
    directly would prove the query works on data no production write ever produces.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.enrolments.upsert_many(
            [
                ClassEnrolment(
                    student_number=STUDENT,
                    academic_year_code=YEAR,
                    class_code="3A",
                    starts_on=YEAR_STARTS,
                )
            ]
        )
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        # Her last day in 3A is the day before 3B begins; ending it on the 15th would put
        # her in two classes on one day, and a report cut on that date would say so.
        uow.enrolments.close_open_enrolment(STUDENT, ends_on=TRANSFER_ON - _ONE_DAY)
        uow.enrolments.upsert_many(
            [
                ClassEnrolment(
                    student_number=STUDENT,
                    academic_year_code=YEAR,
                    class_code="3B",
                    starts_on=TRANSFER_ON,
                )
            ]
        )
        uow.commit()


# ---------------------------------------------------------------------------
# Bulk upsert is idempotent: a re-run updates, it does not duplicate.
# ---------------------------------------------------------------------------


def test_student_upsert_does_not_duplicate_on_a_second_run(sis_database: None) -> None:
    roll = [
        Student(student_number="S001", full_name_ar="ليلى أحمد", full_name_en="Layla Ahmed"),
        Student(student_number="S002", full_name_ar="عمر خالد", full_name_en="Omar Khaled"),
    ]

    with SqlAlchemyUnitOfWork() as uow:
        first = uow.students.upsert_many(roll)
        uow.commit()
    with SqlAlchemyUnitOfWork() as uow:
        second = uow.students.upsert_many(roll)
        uow.commit()

    assert first == {"S001": True, "S002": True}
    # `False` is the reportable half of invariant 3: the caller can say "2 already
    # present" without re-reading the table.
    assert second == {"S001": False, "S002": False}
    assert _count("students") == 2


def test_class_section_upsert_reports_existing_and_writes_no_second_row(
    sis_database: None,
) -> None:
    _seed_structure()
    sections = [
        ClassSection(
            code="3A",
            academic_year_code=YEAR,
            year_level_code="3",
            name_en="Year 3 A",
            name_ar="الثالث أ",
        )
    ]

    with SqlAlchemyUnitOfWork() as uow:
        again = uow.class_sections.upsert_many(sections)
        uow.commit()

    assert again == {(str(YEAR), "3A"): False}
    assert _count("class_sections") == 2


def test_enrolment_upsert_keyed_on_start_date_does_not_stack_placements(
    sis_database: None,
) -> None:
    _seed_structure()
    placement = [
        ClassEnrolment(
            student_number=STUDENT,
            academic_year_code=YEAR,
            class_code="3A",
            starts_on=YEAR_STARTS,
        )
    ]
    key = (str(STUDENT), str(YEAR), "3A", YEAR_STARTS)

    with SqlAlchemyUnitOfWork() as uow:
        first = uow.enrolments.upsert_many(placement)
        uow.commit()
    with SqlAlchemyUnitOfWork() as uow:
        second = uow.enrolments.upsert_many(placement)
        uow.commit()

    assert first == {key: True}
    assert second == {key: False}
    # A second row here would read downstream as an overlapping placement — the child in
    # two classes at once — rather than as the duplicate it is.
    assert _count("class_enrolments") == 1


def test_grade_upsert_restates_a_mark_rather_than_adding_a_second(
    sis_database: None,
) -> None:
    _seed_structure()
    section_id = _seed_grade_references()

    def mark(value: float) -> SubjectGrade:
        return SubjectGrade(
            student_number=STUDENT,
            subject_code="MATH",
            term_code=TERM_1,
            class_section_id=section_id,
            class_code="3A",
            percentage=Percentage(value),
        )

    with SqlAlchemyUnitOfWork() as uow:
        first = uow.grades.upsert_many([mark(72.0)])
        uow.commit()
    with SqlAlchemyUnitOfWork() as uow:
        corrected = uow.grades.upsert_many([mark(81.0)])
        uow.commit()

    key = (str(STUDENT), "MATH", str(TERM_1))
    assert first == {key: True}
    assert corrected == {key: False}
    assert _count("subject_grades") == 1

    with SqlAlchemyUnitOfWork() as uow:
        stored = uow.grades.get(STUDENT, SubjectCode("MATH"), TERM_1)
    assert stored is not None
    assert stored.percentage == Percentage(81.0)


# ---------------------------------------------------------------------------
# Invariant 1: a blank mark is NULL in the column, and an earned zero is 0.
# ---------------------------------------------------------------------------


def test_a_null_percentage_round_trips_as_null_and_a_zero_stays_zero(
    sis_database: None,
) -> None:
    """The two halves of invariant 1, asserted against the column itself.

    Read back through the repository alone this test would still pass if the write path
    stored `0.0` and the read path happened to reverse it. So the raw column is checked
    too: `percentage` must be SQL NULL for the unmarked subject, because that is the
    value every other reader of this database — a report, a `records/` adapter, a
    DBA's query — will see, and the one that decides what a parent is told.
    """
    _seed_structure()
    section_id = _seed_grade_references()

    with SqlAlchemyUnitOfWork() as uow:
        uow.grades.upsert_many(
            [
                # Nobody has marked science yet.
                SubjectGrade(
                    student_number=STUDENT,
                    subject_code="SCI",
                    term_code=TERM_1,
                    class_section_id=section_id,
                    class_code="3A",
                    percentage=None,
                ),
                # A teacher looked at the maths paper and awarded nothing.
                SubjectGrade(
                    student_number=STUDENT,
                    subject_code="MATH",
                    term_code=TERM_1,
                    class_section_id=section_id,
                    class_code="3A",
                    percentage=Percentage(0.0),
                ),
            ]
        )
        uow.commit()

    stored = dict(
        _query(
            "SELECT s.code, g.percentage FROM subject_grades g "
            "JOIN subjects s ON s.id = g.subject_id"
        )
    )
    assert stored["SCI"] is None
    assert stored["MATH"] == 0.0

    with SqlAlchemyUnitOfWork() as uow:
        science = uow.grades.get(STUDENT, SubjectCode("SCI"), TERM_1)
        maths = uow.grades.get(STUDENT, SubjectCode("MATH"), TERM_1)

    assert science is not None and maths is not None
    assert science.percentage is None
    assert science.is_graded is False
    assert science.is_zero is False
    # The mirror-image failure: an earned zero is falsy, so a truthiness test on the read
    # path would hand it back as "not graded yet".
    assert maths.percentage == Percentage(0.0)
    assert maths.is_graded is True
    assert maths.is_zero is True


# ---------------------------------------------------------------------------
# Invariant 2: placement is a dated membership, and history stays true.
# ---------------------------------------------------------------------------


def test_class_section_on_returns_the_historical_class_after_a_transfer(
    sis_database: None,
) -> None:
    """A Term 1 date resolves to 3A after a March move to 3B, forever.

    This is the query invariant 2 exists for, and the failure it prevents is the quiet
    one. A `students.class_code` column asked for "3A in Term 1" after the transfer finds
    nothing and renders a finished, marked, published term as *no marks recorded* — no
    error anywhere — and the registrar re-enters grades that were never lost.
    """
    _seed_structure()
    _transfer_3a_to_3b()

    with SqlAlchemyUnitOfWork() as uow:
        in_term_1 = uow.enrolments.class_section_on(STUDENT, date(2025, 11, 15))
        last_day_in_3a = uow.enrolments.class_section_on(STUDENT, TRANSFER_ON - _ONE_DAY)
        first_day_in_3b = uow.enrolments.class_section_on(STUDENT, TRANSFER_ON)
        # The placement opened in March has no end date, so "today" is answered by it for
        # every day from the transfer onwards.
        today = uow.enrolments.class_section_on(STUDENT, date.today())
        before_she_arrived = uow.enrolments.class_section_on(STUDENT, date(2025, 8, 1))

    assert in_term_1 is not None and str(in_term_1.code) == "3A"
    assert last_day_in_3a is not None and str(last_day_in_3a.code) == "3A"
    assert first_day_in_3b is not None and str(first_day_in_3b.code) == "3B"
    assert today is not None and str(today.code) == "3B"
    # Not a class she might be in — a real answer, and not to be confused with "3A".
    assert before_she_arrived is None

    # Both placements survive the transfer; neither was edited to point at the other.
    with SqlAlchemyUnitOfWork() as uow:
        history = uow.enrolments.list_for_student(STUDENT)
    assert [(str(e.class_code), e.starts_on, e.ends_on) for e in history] == [
        ("3A", YEAR_STARTS, TRANSFER_ON - _ONE_DAY),
        ("3B", TRANSFER_ON, None),
    ]


def test_class_sections_on_answers_in_bulk_exactly_as_the_single_lookup_does(
    sis_database: None,
) -> None:
    """The bulk form is what a grade import uses; a disagreement here misfiles marks."""
    _seed_structure()
    _transfer_3a_to_3b()

    with SqlAlchemyUnitOfWork() as uow:
        term_1_day = date(2025, 11, 15)
        bulk = uow.enrolments.class_sections_on([STUDENT], term_1_day)
        single = uow.enrolments.class_section_on(STUDENT, term_1_day)
        after = uow.enrolments.class_sections_on([STUDENT], TRANSFER_ON)

    assert single is not None
    assert str(bulk[str(STUDENT)].code) == str(single.code) == "3A"
    assert str(after[str(STUDENT)].code) == "3B"


def test_a_child_with_no_placement_is_absent_rather_than_guessed(
    sis_database: None,
) -> None:
    _seed_structure()

    with SqlAlchemyUnitOfWork() as uow:
        uow.students.upsert_many(
            [Student(student_number="S002", full_name_ar="عمر خالد", full_name_en="Omar Khaled")]
        )
        uow.commit()
    with SqlAlchemyUnitOfWork() as uow:
        resolved = uow.enrolments.class_sections_on(
            [STUDENT, StudentNumber("S002")], date(2025, 11, 15)
        )

    assert resolved == {}
