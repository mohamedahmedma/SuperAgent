"""Guardian repositories against a real database, built by alembic.

Deliberately not unit tests. The service above this layer is tested with fakes precisely
so it needs no database, and that leaves exactly the things a fake cannot prove:

* **A unique index that actually refuses.** A fake keyed on a dict collapses a collision
  instead of raising, so "one number belongs to one adult" is only observable here — and
  it is the constraint a future parent login by verification code would rest on.
* **A join that resolves a person through *any* of her numbers.** The fake can be written
  to pass that by construction; only real SQL proves it.
* **A permission filter applied in the query rather than after the rows are loaded.**

The schema comes from `alembic upgrade head` and never from `create_all` (invariant 8):
the migration is the artefact that reaches production, so it is the one the tests stand on.

A note on why the earlier bug mattered. `upsert_many` once collapsed two rows for the same
mother last-wins and silently dropped the alternate number one of them carried. The fake
accumulated correctly and the SQL did not, so every service test passed — which is exactly
the divergence this file exists to catch.
"""
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from sis.config import reset_settings_cache
from sis.domain.guardians import Guardian, RelationshipType, StudentGuardian
from sis.domain.people import Student
from sis.domain.structure import AcademicYear, ClassSection, School, YearLevel
from sis.domain.value_objects import AcademicYearCode, Phone, StudentNumber
from sis.infrastructure.db.session import get_engine, reset_engine
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

_ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"

YEAR = AcademicYearCode("2025-2026")
LAYLA = StudentNumber("S001")
OMAR = StudentNumber("S002")

MOTHER = Phone("+201001234567")
MOTHER_ALT = Phone("+201119998888")
FATHER = Phone("+201002223333")


@pytest.fixture
def sis_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """An empty, migrated database wired into the process-wide engine."""
    url = f"sqlite:///{(tmp_path / 'sis.db').as_posix()}"
    monkeypatch.setenv("SIS_DATABASE_URL", url)
    # Both caches, in this order: clearing the settings alone changes nothing, because the
    # engine already holds a connection to whichever database ran before.
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


@pytest.fixture
def two_children(sis_database: None) -> None:
    """Two children on the roll. Guardians attach to students; they never create them."""
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
            [YearLevel(code="3", school_code="MAIN", name_en="Year 3", name_ar="السنة الثالثة", display_order=3)]
        )
        uow.class_sections.upsert_many(
            [
                ClassSection(
                    code="3A",
                    academic_year_code=YEAR,
                    year_level_code="3",
                    name_en="Year 3 A",
                    name_ar="الثالث أ",
                )
            ]
        )
        uow.students.upsert_many(
            [
                Student(student_number=LAYLA, full_name_ar="ليلى أحمد", full_name_en="Layla Ahmed"),
                Student(student_number=OMAR, full_name_ar="عمر خالد", full_name_en="Omar Khaled"),
            ]
        )
        uow.commit()


def test_a_guardian_upsert_does_not_duplicate_on_a_second_run(two_children: None) -> None:
    """Idempotence against the database, where a dict key cannot hide a collision."""
    guardian = Guardian(phones=(MOTHER,), full_name_en="Fatma Ali")

    with SqlAlchemyUnitOfWork() as uow:
        created = uow.guardians.upsert_many([guardian])
        uow.commit()
    assert created == {str(MOTHER): True}

    with SqlAlchemyUnitOfWork() as uow:
        again = uow.guardians.upsert_many([guardian])
        uow.commit()
    assert again == {str(MOTHER): False}

    assert _count("guardians") == 1
    assert _count("guardian_phones") == 1


def test_a_second_number_finds_the_same_guardian(two_children: None) -> None:
    """The lookup that makes `guardian_phones` worth having rather than a column."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many(
            [Guardian(phones=(MOTHER, MOTHER_ALT), full_name_en="Fatma Ali")]
        )
        uow.commit()

    assert _count("guardian_phones") == 2
    with SqlAlchemyUnitOfWork() as uow:
        by_primary = uow.guardians.get(MOTHER)
        by_alternate = uow.guardians.get(MOTHER_ALT)

    assert by_primary == by_alternate
    # Primary first, always: it is the identity every stored link names.
    assert [str(phone) for phone in by_alternate.phones] == [str(MOTHER), str(MOTHER_ALT)]


def test_two_rows_for_one_mother_keep_both_her_numbers(two_children: None) -> None:
    """The regression this file was written for.

    She is on two of her children's rows and states her second number on only one of
    them. A last-wins collapse keeps whichever row the file ended with and loses a number
    the school was given — silently, because every row still looks correct.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many(
            [
                Guardian(phones=(MOTHER, MOTHER_ALT), full_name_en="Fatma Ali"),
                Guardian(phones=(MOTHER,), full_name_en="Fatma Ali"),
            ]
        )
        uow.commit()

    assert _count("guardians") == 1
    assert _count("guardian_phones") == 2


def test_numbers_accumulate_across_uploads(two_children: None) -> None:
    """A later upload mentioning one number must not drop the other already on file."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many(
            [Guardian(phones=(MOTHER, MOTHER_ALT), full_name_en="Fatma Ali")]
        )
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many([Guardian(phones=(MOTHER,), full_name_en="Fatma Ali")])
        uow.commit()

    assert _count("guardian_phones") == 2


def test_the_database_refuses_one_number_for_two_adults(two_children: None) -> None:
    """The constraint a future parent login depends on, asserted against the schema.

    Checked here and not only in the importer that usually catches it first: without the
    index, a verification code sent to one family could unlock another's records.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many([Guardian(phones=(MOTHER,), full_name_en="Fatma Ali")])
        uow.commit()

    with pytest.raises(IntegrityError):
        with get_engine().begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO guardians (public_id, full_name_ar, full_name_en, "
                    "preferred_language, is_active, created_at, updated_at) VALUES "
                    "('other-public-id', '', 'Someone Else', 'ar', 1, "
                    "'2026-01-01 00:00:00', '2026-01-01 00:00:00')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO guardian_phones (guardian_id, phone, is_primary, created_at) "
                    "SELECT id, :phone, 1, '2026-01-01 00:00:00' FROM guardians "
                    "WHERE public_id = 'other-public-id'"
                ),
                {"phone": str(MOTHER)},
            )


def test_one_adult_links_to_several_children_and_back(two_children: None) -> None:
    """The many-to-many, in both directions, through a real join."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many(
            [
                Guardian(phones=(MOTHER, MOTHER_ALT), full_name_en="Fatma Ali"),
                Guardian(phones=(FATHER,), full_name_en="Hassan Mahmoud"),
            ]
        )
        uow.student_guardians.upsert_many(
            [
                StudentGuardian(
                    student_number=LAYLA,
                    guardian_phone=MOTHER,
                    relationship_type=RelationshipType.MOTHER,
                    can_view_records=True,
                ),
                StudentGuardian(
                    student_number=OMAR,
                    guardian_phone=MOTHER,
                    relationship_type=RelationshipType.MOTHER,
                    can_view_records=True,
                ),
                StudentGuardian(
                    student_number=LAYLA,
                    guardian_phone=FATHER,
                    relationship_type=RelationshipType.FATHER,
                    can_view_records=True,
                ),
            ]
        )
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        for_child = uow.student_guardians.list_for_student(LAYLA)
        # Asked by her *second* number, which is the parent-login shape.
        for_mother = uow.student_guardians.list_students_for_guardian(MOTHER_ALT)

    assert {str(link.guardian_phone) for link in for_child} == {str(MOTHER), str(FATHER)}
    assert {str(link.student_number) for link in for_mother} == {"S001", "S002"}


def test_a_link_upsert_restates_rather_than_adding_a_second(two_children: None) -> None:
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many([Guardian(phones=(MOTHER,), full_name_en="Fatma Ali")])
        uow.student_guardians.upsert_many(
            [
                StudentGuardian(
                    student_number=LAYLA,
                    guardian_phone=MOTHER,
                    relationship_type=RelationshipType.MOTHER,
                )
            ]
        )
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        again = uow.student_guardians.upsert_many(
            [
                StudentGuardian(
                    student_number=LAYLA,
                    guardian_phone=MOTHER,
                    relationship_type=RelationshipType.GUARDIAN,
                    can_view_records=True,
                )
            ]
        )
        uow.commit()

    assert again == {("S001", str(MOTHER)): False}
    assert _count("student_guardians") == 1

    with SqlAlchemyUnitOfWork() as uow:
        (link,) = uow.student_guardians.list_for_student(LAYLA)
    assert link.relationship_type is RelationshipType.GUARDIAN
    assert link.can_view_records is True


def test_a_guardian_with_two_numbers_is_listed_once_per_child(
    two_children: None,
) -> None:
    """The join is filtered to the primary number.

    Unfiltered it returns one row per number she holds, quietly duplicating a mother with
    two lines into two entries on her child's contact list — a bug that looks like a data
    problem rather than a query one.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many(
            [Guardian(phones=(MOTHER, MOTHER_ALT), full_name_en="Fatma Ali")]
        )
        uow.student_guardians.upsert_many(
            [StudentGuardian(student_number=LAYLA, guardian_phone=MOTHER)]
        )
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        links = uow.student_guardians.list_for_student(LAYLA)

    assert len(links) == 1


def test_a_restricted_link_is_filtered_in_sql(two_children: None) -> None:
    """A child a parent may not see is never loaded into the process answering them.

    Filtered in the query rather than in Python, so there is no object for a later bug to
    leak — and a registrar's view can still ask for everything explicitly.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many([Guardian(phones=(MOTHER,), full_name_en="Fatma Ali")])
        uow.student_guardians.upsert_many(
            [
                StudentGuardian(
                    student_number=LAYLA,
                    guardian_phone=MOTHER,
                    relationship_type=RelationshipType.MOTHER,
                    can_view_records=True,
                ),
                StudentGuardian(
                    student_number=OMAR,
                    guardian_phone=MOTHER,
                    relationship_type=RelationshipType.MOTHER,
                    can_view_records=False,
                    restriction_note="court order 2026/114",
                ),
            ]
        )
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        visible = uow.student_guardians.list_students_for_guardian(MOTHER)
        everything = uow.student_guardians.list_students_for_guardian(
            MOTHER, viewable_only=False
        )

    assert [str(link.student_number) for link in visible] == ["S001"]
    assert len(everything) == 2


def test_the_database_refuses_two_primary_contacts_for_one_child(
    two_children: None,
) -> None:
    """A partial unique index, so the office always has exactly one first call."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many(
            [
                Guardian(phones=(MOTHER,), full_name_en="Fatma Ali"),
                Guardian(phones=(FATHER,), full_name_en="Hassan Mahmoud"),
            ]
        )
        uow.commit()

    with pytest.raises(IntegrityError):
        with SqlAlchemyUnitOfWork() as uow:
            uow.student_guardians.upsert_many(
                [
                    StudentGuardian(
                        student_number=LAYLA,
                        guardian_phone=MOTHER,
                        is_primary_contact=True,
                    ),
                    StudentGuardian(
                        student_number=LAYLA,
                        guardian_phone=FATHER,
                        is_primary_contact=True,
                    ),
                ]
            )
            uow.commit()


def test_unlinking_removes_the_link_and_leaves_the_guardian(two_children: None) -> None:
    """A link made in error is erased; the adult stays on file for her other children."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many([Guardian(phones=(MOTHER,), full_name_en="Fatma Ali")])
        uow.student_guardians.upsert_many(
            [StudentGuardian(student_number=LAYLA, guardian_phone=MOTHER)]
        )
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        removed = uow.student_guardians.unlink(LAYLA, MOTHER)
        missing = uow.student_guardians.unlink(OMAR, MOTHER)
        uow.commit()

    assert (removed, missing) == (True, False)
    assert _count("student_guardians") == 0
    assert _count("guardians") == 1


def test_the_stored_primary_number_is_never_demoted(two_children: None) -> None:
    """Her identity is the number every stored link names; moving it detaches them all."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many([Guardian(phones=(MOTHER,), full_name_en="Fatma Ali")])
        uow.student_guardians.upsert_many(
            [StudentGuardian(student_number=LAYLA, guardian_phone=MOTHER)]
        )
        uow.commit()

    # A later sheet lists her numbers the other way round.
    with SqlAlchemyUnitOfWork() as uow:
        uow.guardians.upsert_many(
            [Guardian(phones=(MOTHER_ALT, MOTHER), full_name_en="Fatma Ali")]
        )
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        guardian = uow.guardians.get(MOTHER_ALT)
        (link,) = uow.student_guardians.list_for_student(LAYLA)

    assert str(guardian.primary_phone) == str(MOTHER)
    assert str(link.guardian_phone) == str(MOTHER)
