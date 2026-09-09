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


def test_0010_gives_every_existing_subject_the_grades_it_was_already_taught_on(
    school_with_marks: str,
) -> None:
    """Requirement 5, and the one place it can actually be broken.

    Before 0010 a subject was implicitly taught on every rung of its school — every screen
    that offered a subject offered all of them. After 0010, anything reading through
    `subject_year_levels` shows a subject only where a row says so, so an empty table would
    silently empty every existing school's marks pickers: the subjects would all still be
    on file, and none of them would appear anywhere.

    So the assertion is not "the table exists" but "the rows say what the school already
    meant": every (subject, rung) pair of every school, and no pair that crosses schools.

    Asserted after a real downgrade and upgrade rather than against the fixture's own
    database, because that is the only way to see the backfill *run* — the template these
    tests copy was migrated before any of these rows existed.
    """
    reset_engine()
    command.downgrade(_alembic(), "0002")
    reset_engine()
    command.upgrade(_alembic(), "head")
    reset_engine()

    from sqlalchemy import text

    from sis.infrastructure.db.session import get_engine

    with get_engine().connect() as connection:
        pairs = set(
            connection.execute(
                text(
                    """
                    SELECT s.code, l.code
                    FROM subject_year_levels a
                    JOIN subjects s ON s.id = a.subject_id
                    JOIN year_levels l ON l.id = a.year_level_id
                    """
                )
            ).all()
        )
        crossed = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM subject_year_levels a
                JOIN subjects s ON s.id = a.subject_id
                JOIN academic_years y ON y.id = s.academic_year_id
                JOIN year_levels l ON l.id = a.year_level_id
                WHERE l.school_id <> y.school_id
                """
            )
        ).scalar_one()

    # The fixture teaches MATH and SCI in one school with one rung, `3`. Both subjects were
    # available on that rung before this revision, and both still are.
    assert pairs == {("MATH", "3"), ("SCI", "3")}
    assert crossed == 0, "a subject was assigned to a rung of a school that does not teach it"

    # And the marks the backfill was protecting are untouched by it.
    assert _marks_on_file() == {"MATH": 88.5, "SCI": 0.0}


def test_0011_keeps_the_dates_a_school_already_stated(school_with_marks: str) -> None:
    """Widening NOT NULL to NULL must not be an excuse to touch a single row.

    The fixture's term has real dates. After a full downgrade and upgrade they are still
    the same dates — the revision changes what the column *permits*, and a school that has
    dated its terms notices nothing at all.
    """
    reset_engine()
    command.downgrade(_alembic(), "0002")
    reset_engine()
    command.upgrade(_alembic(), "head")
    reset_engine()

    with SqlAlchemyUnitOfWork() as uow:
        term = uow.terms.get(TermCode(TERM))
    assert term is not None
    assert term.starts_on == date(2025, 9, 1)
    assert term.ends_on == date(2025, 12, 20)
    assert term.is_dated is True


def test_0011_downgrade_refuses_rather_than_inventing_dates_for_an_undated_term(
    school_with_marks: str,
) -> None:
    """A term nobody has dated cannot be narrowed back to NOT NULL, so the downgrade stops.

    The alternative is the dangerous one and is why this is asserted rather than left to
    the code review: filling the gap from the year's window would look like a clean
    downgrade and would turn "not decided yet" into a boundary that decides which class a
    mark is filed under. A migration that cannot preserve the meaning has to refuse.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.terms.upsert_many(
            [
                Term(
                    code="2026-T2",
                    academic_year_code=YEAR,
                    name_en="Term 2",
                    name_ar="الفصل الثاني",
                    sequence=2,
                )
            ]
        )
        uow.commit()

    reset_engine()
    with pytest.raises(RuntimeError, match="cannot restore NOT NULL term dates"):
        command.downgrade(_alembic(), "0010")
    reset_engine()

    # The refusal cost nothing: the undated term and every mark are where they were.
    with SqlAlchemyUnitOfWork() as uow:
        undated = uow.terms.get(TermCode("2026-T2"))
    assert undated is not None and undated.is_dated is False
    assert _marks_on_file() == {"MATH": 88.5, "SCI": 0.0}
