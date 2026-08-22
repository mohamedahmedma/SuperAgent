"""A guardian's children, with the year group each one is in today.

Added because a parent asking a general question is asking it about one year: a fee
table covers every year in the school, and without this the answer is the whole table.

The year is three hops from the child, and every hop is a schema decision refusing to be
shortcut — a class is scoped to an academic year, a year LEVEL is scoped to a school and
not to a year, and only the academic year names the school. Most of these tests exist to
pin down what happens when one of those hops comes back empty, because "not known" has
to stay distinguishable from "Year 1".
"""
from datetime import date

import pytest

from sis.application.services.queries import QueryService
from sis.domain.guardians import Guardian, RelationshipType, StudentGuardian
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import AcademicYear, ClassSection, School, YearLevel
from sis.domain.value_objects import Phone, StudentNumber

MOTHER = "+201001234567"
TERM_TIME = date(2026, 3, 15)
SUMMER = date(2026, 8, 1)


@pytest.fixture()
def family(uow_factory):
    """One mother, three children: one placed in Year 4, one in Year 2, one not yet."""
    with uow_factory() as uow:
        uow.schools.upsert_many([School(code="MAIN", name_en="Main", name_ar="الرئيسية")])
        uow.academic_years.upsert_many([
            AcademicYear(
                code="2025-2026", school_code="MAIN", name_en="2025-2026",
                name_ar="٢٠٢٥-٢٠٢٦", starts_on=date(2025, 9, 1),
                ends_on=date(2026, 6, 30), is_current=True,
            )
        ])
        uow.year_levels.upsert_many([
            YearLevel(code="Y4", school_code="MAIN", name_en="Year 4",
                      name_ar="الصف الرابع", display_order=4),
            YearLevel(code="Y2", school_code="MAIN", name_en="Year 2",
                      name_ar="الصف الثاني", display_order=2),
        ])
        uow.class_sections.upsert_many([
            ClassSection(code="4B", academic_year_code="2025-2026", year_level_code="Y4",
                         name_en="4B", name_ar="٤ب"),
            ClassSection(code="2A", academic_year_code="2025-2026", year_level_code="Y2",
                         name_en="2A", name_ar="٢أ"),
        ])
        uow.students.upsert_many([
            Student(student_number=StudentNumber("S001"), full_name_ar="علي", full_name_en="Ali"),
            Student(student_number=StudentNumber("S002"), full_name_ar="ليلى", full_name_en="Layla"),
            Student(student_number=StudentNumber("S003"), full_name_ar="سارة", full_name_en="Sara"),
        ])
        uow.enrolments.upsert_many([
            ClassEnrolment(student_number=StudentNumber("S001"), academic_year_code="2025-2026",
                           class_code="4B", starts_on=date(2025, 9, 1)),
            ClassEnrolment(student_number=StudentNumber("S002"), academic_year_code="2025-2026",
                           class_code="2A", starts_on=date(2025, 9, 1)),
        ])
        uow.guardians.upsert_many([Guardian(phones=(Phone(MOTHER),), full_name_ar="فاطمة")])
        uow.student_guardians.upsert_many([
            StudentGuardian(student_number=StudentNumber(n), guardian_phone=Phone(MOTHER),
                            relationship_type=RelationshipType.MOTHER, can_view_records=True)
            for n in ("S001", "S002", "S003")
        ])
        uow.commit()
    return QueryService(uow_factory)


def _by_number(entries):
    return {str(e.link.student_number): e for e in entries}


class TestTheYearGroupIsResolved:
    def test_each_child_carries_the_year_she_is_in(self, family):
        found = _by_number(family.guardian_students(Phone(MOTHER), on_date=TERM_TIME))

        assert found["S001"].year_label == "الصف الرابع"
        assert found["S002"].year_label == "الصف الثاني"

    def test_the_class_section_travels_beside_it(self, family):
        """A parent thinks in year groups and a registrar thinks in rooms. Both are
        carried, because they answer different questions."""
        found = _by_number(family.guardian_students(Phone(MOTHER), on_date=TERM_TIME))

        assert found["S001"].class_section is not None
        assert str(found["S001"].class_section.code) == "4B"
        assert str(found["S001"].year_level.code) == "Y4"

    def test_the_handle_route_answers_the_same(self, family, uow_factory):
        """The parent-facing route. It must not be a second, thinner implementation."""
        with uow_factory() as uow:
            public_id = uow.guardians.public_id_for(Phone(MOTHER))

        by_phone = _by_number(family.guardian_students(Phone(MOTHER), on_date=TERM_TIME))
        by_handle = _by_number(family.guardian_students_by_id(public_id, on_date=TERM_TIME))

        assert {n: e.year_label for n, e in by_handle.items()} == {
            n: e.year_label for n, e in by_phone.items()
        }


class TestNotKnownStaysNotKnown:
    def test_a_child_with_no_placement_today_has_no_year(self, family):
        """An ordinary state, not an error: a child enrolled for next September has no
        class today. Guessing one narrows a fee table to the wrong row."""
        found = _by_number(family.guardian_students(Phone(MOTHER), on_date=TERM_TIME))

        assert found["S003"].year_label == ""
        assert found["S003"].class_section is None
        assert found["S003"].year_level is None

    def test_an_open_placement_does_not_expire_with_the_school_year(self, family):
        """August still reports Year 4, and that is right rather than a bug.

        A placement with no `ends_on` is open, and open means open — a transfer is
        closing one enrolment and opening another, never a row quietly expiring on 30
        June. Asserted because the opposite is the intuitive guess, and a future reader
        "fixing" it would silently blank every child's year over the summer.
        """
        found = _by_number(family.guardian_students(Phone(MOTHER), on_date=SUMMER))

        assert found["S001"].year_label == "الصف الرابع"

    def test_a_placement_that_ended_reports_no_year_after_its_last_day(self, family, uow_factory):
        """`ends_on` is her last day in the class, inclusive."""
        with uow_factory() as uow:
            uow.enrolments.close_open_enrolment(StudentNumber("S001"), ends_on=date(2026, 3, 12))
            uow.commit()

        on_her_last_day = _by_number(family.guardian_students(Phone(MOTHER), on_date=date(2026, 3, 12)))
        the_day_after = _by_number(family.guardian_students(Phone(MOTHER), on_date=date(2026, 3, 13)))

        assert on_her_last_day["S001"].year_label == "الصف الرابع"
        assert the_day_after["S001"].year_label == ""

    def test_a_caller_that_does_not_ask_pays_nothing_and_gets_nothing(self, family):
        """`on_date` is optional because most callers do not need the year and it costs
        two more queries."""
        found = _by_number(family.guardian_students(Phone(MOTHER)))

        assert {e.year_label for e in found.values()} == {""}
        assert all(e.student is not None for e in found.values())

    def test_a_school_with_no_ladder_falls_back_to_the_class_name(self, uow_factory):
        """`4B` is something a parent recognises; `Y4` is an internal string. Falling
        back to the room beats falling back to a code, and both beat guessing."""
        with uow_factory() as uow:
            uow.schools.upsert_many([School(code="X", name_en="X", name_ar="س")])
            uow.academic_years.upsert_many([
                AcademicYear(code="2025-2026", school_code="X", name_en="y", name_ar="y",
                             starts_on=date(2025, 9, 1), ends_on=date(2026, 6, 30), is_current=True)
            ])
            # A section whose year level was never uploaded.
            uow.year_levels.upsert_many([
                YearLevel(code="Y9", school_code="X", name_en="Y9", name_ar="Y9", display_order=9)
            ])
            uow.class_sections.upsert_many([
                ClassSection(code="9C", academic_year_code="2025-2026", year_level_code="Y9",
                             name_en="9C", name_ar="٩ج")
            ])
            uow.students.upsert_many([
                Student(student_number=StudentNumber("T1"), full_name_ar="ط", full_name_en="T")
            ])
            uow.enrolments.upsert_many([
                ClassEnrolment(student_number=StudentNumber("T1"), academic_year_code="2025-2026",
                               class_code="9C", starts_on=date(2025, 9, 1))
            ])
            uow.guardians.upsert_many([Guardian(phones=(Phone("+201119998888"),), full_name_ar="و")])
            uow.student_guardians.upsert_many([
                StudentGuardian(student_number=StudentNumber("T1"),
                                guardian_phone=Phone("+201119998888"),
                                relationship_type=RelationshipType.FATHER, can_view_records=True)
            ])
            uow.commit()

        entries = QueryService(uow_factory).guardian_students(
            Phone("+201119998888"), on_date=TERM_TIME
        )
        # The rung exists here, so the rung's name wins.
        assert entries[0].year_label == "Y9"


class TestItStaysOneQueryPerFamily:
    def test_a_transfer_resolves_to_the_room_she_is_in_now(self, family, uow_factory):
        """`class_sections_on` is transfer-aware. A child who moved 4B -> 2A in March must
        report where she is, not whichever placement was written first."""
        with uow_factory() as uow:
            uow.enrolments.close_open_enrolment(StudentNumber("S001"), ends_on=date(2026, 2, 28))
            uow.enrolments.upsert_many([
                ClassEnrolment(student_number=StudentNumber("S001"),
                               academic_year_code="2025-2026", class_code="2A",
                               starts_on=date(2026, 3, 1))
            ])
            uow.commit()

        found = _by_number(family.guardian_students(Phone(MOTHER), on_date=TERM_TIME))

        assert found["S001"].year_label == "الصف الثاني"

    def test_her_year_before_the_transfer_is_still_the_old_one(self, family, uow_factory):
        with uow_factory() as uow:
            uow.enrolments.close_open_enrolment(StudentNumber("S001"), ends_on=date(2026, 2, 28))
            uow.enrolments.upsert_many([
                ClassEnrolment(student_number=StudentNumber("S001"),
                               academic_year_code="2025-2026", class_code="2A",
                               starts_on=date(2026, 3, 1))
            ])
            uow.commit()

        found = _by_number(family.guardian_students(Phone(MOTHER), on_date=date(2026, 1, 15)))

        assert found["S001"].year_label == "الصف الرابع"
