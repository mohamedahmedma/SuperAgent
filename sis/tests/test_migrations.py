"""The migrations, run against a database that has rows in it.

Every other test in this suite starts from a freshly migrated, empty database. That is the
right default and it has one blind spot, which cost real time to find: a revision that
rebuilds a table behaves completely differently when another table's rows reference it.
Revision 0003 rebuilds `subjects`, and `subject_grades` points at it.

Two distinct failures hid in that gap, and both are what this file is here to catch:

  * SQLite refuses to DROP a table that a foreign key still references, so the rebuild
    died half-done on any database with a single mark in it — while passing on an empty
    one.
  * With the pragma that fixes the first issue issued carelessly, the migration ran inside
    a transaction nobody committed. SQLite's DDL is not transactional, so the *schema*
    changes stuck while the backfill and the `alembic_version` stamp were rolled back. The
    result was a half-migrated database and an exit code of zero, which is the worst
    combination available: the next revision to run would start from a state no revision
    had ever produced.

So these tests assert on data, not on schema. A schema assertion would have passed
throughout both bugs.
"""
from datetime import date

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig

from sis.domain.grades import SubjectGrade
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import AcademicYear, ClassSection, School, Subject, Term, YearLevel
from sis.domain.value_objects import Percentage, StudentNumber, TermCode
from sis.infrastructure.db.session import reset_engine
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

from .conftest import _ALEMBIC_INI

YEAR = "2025-2026"
TERM = "2026-T1"


def _alembic() -> AlembicConfig:
    return AlembicConfig(str(_ALEMBIC_INI))


def _seed_a_school_with_marks() -> None:
    """One year, one class, one child, one subject and two marks — one of them a zero.

    The zero is here because it is the value a careless migration is most likely to lose:
    anything that rebuilds a column with a `NOT NULL DEFAULT 0`, or that reads a percentage
    through a falsy check, turns a real zero and an unmarked subject into the same thing.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.schools.upsert_many([School(code="MAIN", name_en="Main School", name_ar="المدرسة")])
        uow.academic_years.upsert_many(
            [
                AcademicYear(
                    code=YEAR,
                    school_code="MAIN",
                    name_en="2025-2026",
                    name_ar="٢٠٢٥-٢٠٢٦",
                    starts_on=date(2025, 9, 1),
                    ends_on=date(2026, 6, 30),
                    is_current=True,
                )
            ]
        )
        uow.year_levels.upsert_many(
            [YearLevel(code="3", school_code="MAIN", name_en="Year 3", name_ar="السنة 3", display_order=3)]
        )
        uow.class_sections.upsert_many(
            [
                ClassSection(
                    code="3A",
                    academic_year_code=YEAR,
                    year_level_code="3",
                    name_en="Year 3 A",
                    name_ar="السنة 3 أ",
                )
            ]
        )
        uow.terms.upsert_many(
            [
                Term(
                    code=TERM,
                    academic_year_code=YEAR,
                    name_en="Term 1",
                    name_ar="الفصل الأول",
                    starts_on=date(2025, 9, 1),
                    ends_on=date(2025, 12, 20),
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
                    display_order=1,
                ),
                Subject(
                    code="SCI",
                    academic_year_code=YEAR,
                    name_en="Science",
                    name_ar="العلوم",
                    display_order=2,
                ),
            ]
        )
        uow.students.upsert_many(
            [
                Student(
                    student_number="10432",
                    full_name_ar="سارة محمد",
                    full_name_en="Sara Mohamed",
                )
            ]
        )
        uow.enrolments.upsert_many(
            [
                ClassEnrolment(
                    student_number="10432",
                    academic_year_code=YEAR,
                    class_code="3A",
                    starts_on=date(2025, 9, 1),
                )
            ]
        )
        section_id = uow.class_sections.ids_for([(YEAR, "3A")])[(YEAR, "3A")]
        uow.grades.upsert_many(
            [
                SubjectGrade(
                    student_number="10432",
                    subject_code="MATH",
                    term_code=TERM,
                    class_section_id=section_id,
                    class_code="3A",
                    percentage=Percentage(88.5),
                ),
                # A genuine zero, not a blank.
                SubjectGrade(
                    student_number="10432",
                    subject_code="SCI",
                    term_code=TERM,
                    class_section_id=section_id,
                    class_code="3A",
                    percentage=Percentage(0),
                ),
            ]
        )
        uow.commit()


def _marks_on_file() -> dict[str, float | None]:
    """The marks as the service reads them back, keyed by subject code."""
    with SqlAlchemyUnitOfWork() as uow:
        grades = uow.grades.list_for_student(
            StudentNumber("10432"), term_code=TermCode(TERM)
        )
        return {
            str(grade.subject_code): (
                None if grade.percentage is None else float(grade.percentage.value)
            )
            for grade in grades
        }


@pytest.fixture()
def school_with_marks(database: str) -> str:
    """A migrated database with a school in it. `database` gives the schema at head."""
    _seed_a_school_with_marks()
    return database


def test_the_head_revision_round_trips_a_database_that_has_rows_in_it(
    school_with_marks: str,
) -> None:
    """Downgrade to 0002 and back to head, with marks on file the whole way.

    This is the test that would have caught both bugs in the docstring. It fails if the
    rebuild cannot run against referenced rows, and it fails if the revision is not
    actually stamped — the second `upgrade` would then have nothing to do and the backfill
    would never have happened.
    """
    before = _marks_on_file()
    assert before == {"MATH": 88.5, "SCI": 0.0}

    reset_engine()
    command.downgrade(_alembic(), "0002")
    reset_engine()
    command.upgrade(_alembic(), "head")
    reset_engine()

    after = _marks_on_file()
    assert after == before, "a mark changed value across a downgrade and upgrade"
    # The zero specifically, spelled out: it must still be a zero and not a blank.
    assert after["SCI"] == 0.0 and after["SCI"] is not None


def test_the_subject_backfill_uses_the_year_the_marks_were_stated_in(
    school_with_marks: str,
) -> None:
    """After 0003 re-runs, every subject sits in the year of the term its marks belong to.

    The property that matters is stated as a join rather than as a column check: a mark
    must resolve to a subject whose year is the year of its own term. Get the backfill
    wrong and this is what breaks — the marks all still exist, and every report card prints
    a subject the school never taught that year.
    """
    reset_engine()
    command.downgrade(_alembic(), "0002")
    reset_engine()
    command.upgrade(_alembic(), "head")
    reset_engine()

    from sqlalchemy import text

    from sis.infrastructure.db.session import get_engine

    with get_engine().connect() as connection:
        mismatched = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM subject_grades g
                JOIN subjects s ON s.id = g.subject_id
                JOIN terms t ON t.id = g.term_id
                WHERE s.academic_year_id <> t.academic_year_id
                """
            )
        ).scalar_one()
        orphans = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM subject_grades g
                LEFT JOIN subjects s ON s.id = g.subject_id
                WHERE s.id IS NULL
                """
            )
        ).scalar_one()
        dangling = connection.exec_driver_sql("PRAGMA foreign_key_check").all()

    assert mismatched == 0, "a mark resolves to a subject from a different academic year"
    assert orphans == 0, "a mark lost its subject in the rebuild"
    assert not dangling, f"the rebuild left dangling references: {dangling}"


def test_a_downgrade_refuses_rather_than_deleting_a_subject_it_cannot_keep(
    school_with_marks: str,
) -> None:
    """Two years each teaching MATH cannot fit in one global catalogue, so 0002 refuses.

    The alternative — picking one row and dropping the other — would silently repoint a
    year's marks at a subject its school never taught. A migration that cannot preserve
    the data has to stop, and say which codes are in the way.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.schools.upsert_many([School(code="MAIN", name_en="Main School", name_ar="المدرسة")])
        uow.academic_years.upsert_many(
            [
                AcademicYear(
                    code="2026-2027",
                    school_code="MAIN",
                    name_en="2026-2027",
                    name_ar="٢٠٢٦-٢٠٢٧",
                    starts_on=date(2026, 9, 1),
                    ends_on=date(2027, 6, 30),
                    is_current=False,
                )
            ]
        )
        # The same code in a second year — which is exactly what 0003 made possible.
        uow.subjects.upsert_many(
            [
                Subject(
                    code="MATH",
                    academic_year_code="2026-2027",
                    name_en="Mathematics",
                    name_ar="الرياضيات",
                )
            ]
        )
        uow.commit()

    reset_engine()
    with pytest.raises(RuntimeError, match="more than one academic year"):
        command.downgrade(_alembic(), "0002")
    reset_engine()

    # And the refusal cost nothing: the marks are all still there.
    assert _marks_on_file() == {"MATH": 88.5, "SCI": 0.0}
