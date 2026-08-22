"""The child's sex: recorded, imported, and never guessed.

Added so a parent can write "my son" and be understood without being asked which child.
Most of these cases are about the value that is neither male nor female — because every
child already on file is `unspecified`, and a system that treats that as a default sex
will name the wrong child while the parent watches.
"""
import pytest

from sis.application.dto import ParsedRosterRow
from sis.domain.people import Gender, Student
from sis.domain.value_objects import StudentNumber
from sis.infrastructure.parsers.roster import _gender


class TestTheDomainValue:
    def test_a_child_with_nothing_recorded_is_unspecified_not_null(self):
        """One spelling of "the school has not said", not two."""
        child = Student(student_number="S1", full_name_ar="ليلى", full_name_en="")
        assert child.gender is Gender.UNSPECIFIED

    @pytest.mark.parametrize("stored,expected", [
        ("male", Gender.MALE),
        ("FEMALE", Gender.FEMALE),
        ("Male", Gender.MALE),
        (Gender.FEMALE, Gender.FEMALE),
    ])
    def test_a_stored_string_becomes_the_enum(self, stored, expected):
        """So a row read back from the database compares equal to one a service built."""
        child = Student(student_number="S1", full_name_ar="ع", full_name_en="", gender=stored)
        assert child.gender is expected

    @pytest.mark.parametrize("junk", ["", None, "لا اعرف", "other", "1", "  "])
    def test_an_unrecognised_value_degrades_rather_than_raising(self, junk):
        """Lossless — the school simply has not said. Refusing the row would make one bad
        cell cost a child her entire record."""
        child = Student(student_number="S1", full_name_ar="ع", full_name_en="", gender=junk)
        assert child.gender is Gender.UNSPECIFIED

    def test_it_is_a_string_so_it_stores_and_compares_without_a_special_case(self):
        assert Gender.MALE == "male"
        assert str(Gender.FEMALE) == "female"


class TestTheSpreadsheetCell:
    @pytest.mark.parametrize("cell", ["ذكر", "ذكور", "ولد", "m", "M", "male", "Boy", "طالب"])
    def test_the_words_a_school_types_for_a_boy(self, cell):
        assert _gender(cell) is Gender.MALE

    @pytest.mark.parametrize("cell", ["أنثى", "انثى", "اناث", "بنت", "f", "female", "Girl", "طالبة"])
    def test_the_words_a_school_types_for_a_girl(self, cell):
        """Both hamza spellings, because a registrar types whichever their keyboard gives."""
        assert _gender(cell) is Gender.FEMALE

    @pytest.mark.parametrize("cell", ["", "   ", None, "-", "n/a", "غير محدد", "لا ينطبق"])
    def test_anything_else_is_unspecified_and_never_a_guess(self, cell):
        assert _gender(cell) is Gender.UNSPECIFIED

    def test_a_cell_is_matched_whole_never_by_containment(self):
        """The relationship resolver needs containment because registrars write "big
        brother". Nobody writes "quite male", and the single letters here would match
        inside unrelated words."""
        assert _gender("female student") is Gender.UNSPECIFIED


class TestTheImport:
    def test_a_row_with_no_gender_column_parses_as_unspecified(self):
        row = ParsedRosterRow(
            line_number=2,
            student_number=StudentNumber("S1"),
            full_name_ar="ليلى",
            full_name_en="Layla",
        )
        assert row.gender is Gender.UNSPECIFIED

    def test_re_importing_a_sheet_without_the_column_does_not_erase_a_recorded_sex(self):
        """The rule that matters most in practice. A school uploads a roster with genders,
        then re-uploads a corrected name list that has no gender column — without this,
        every child silently reverts to unspecified and the feature stops working with no
        error anywhere."""
        from sis.application.services.roster_import import _Assertion, RosterImportService
        from datetime import date
        from sis.domain.value_objects import AcademicYearCode, ClassCode

        existing = Student(
            student_number="S1", full_name_ar="علي", full_name_en="Ali", gender=Gender.MALE
        )
        silent = _Assertion(
            line=2,
            number=StudentNumber("S1"),
            name_ar="علي",
            name_en="Ali",
            year_code=AcademicYearCode("2025-2026"),
            class_code=ClassCode("3A"),
            starts_on=date(2025, 9, 1),
            gender=Gender.UNSPECIFIED,
        )

        merged = RosterImportService._merged_student(None, silent, existing)

        assert merged.gender is Gender.MALE

    def test_a_stated_sex_corrects_a_blank_one(self):
        from sis.application.services.roster_import import _Assertion, RosterImportService
        from datetime import date
        from sis.domain.value_objects import AcademicYearCode, ClassCode

        existing = Student(student_number="S1", full_name_ar="علي", full_name_en="Ali")
        stated = _Assertion(
            line=2,
            number=StudentNumber("S1"),
            name_ar="علي",
            name_en="Ali",
            year_code=AcademicYearCode("2025-2026"),
            class_code=ClassCode("3A"),
            starts_on=date(2025, 9, 1),
            gender=Gender.MALE,
        )

        merged = RosterImportService._merged_student(None, stated, existing)

        assert merged.gender is Gender.MALE

    def test_the_payload_carries_it_across_the_preview_commit_boundary(self):
        """Commit rebuilds the assertion from this dict rather than from the file, so a
        field missing here previews correctly and is silently dropped on the way to
        being stored."""
        from sis.application.services.roster_import import _Assertion, _K_GENDER
        from datetime import date
        from sis.domain.value_objects import AcademicYearCode, ClassCode

        payload = _Assertion(
            line=2,
            number=StudentNumber("S1"),
            name_ar="علي",
            name_en="Ali",
            year_code=AcademicYearCode("2025-2026"),
            class_code=ClassCode("3A"),
            starts_on=date(2025, 9, 1),
            gender=Gender.FEMALE,
        ).payload

        assert payload[_K_GENDER] == "female"


class TestItSurvivesTheDatabase:
    def test_a_recorded_sex_round_trips(self, uow_factory):
        from sis.domain.people import Student as S

        with uow_factory() as uow:
            uow.students.upsert_many([
                S(student_number="G001", full_name_ar="علي", full_name_en="Ali", gender=Gender.MALE),
                S(student_number="G002", full_name_ar="ليلى", full_name_en="Layla", gender=Gender.FEMALE),
                S(student_number="G003", full_name_ar="سيد", full_name_en="Sayed"),
            ])
            uow.commit()

        with uow_factory() as uow:
            found = uow.students.get_many([
                StudentNumber("G001"), StudentNumber("G002"), StudentNumber("G003"),
            ])

        assert found["G001"].gender is Gender.MALE
        assert found["G002"].gender is Gender.FEMALE
        assert found["G003"].gender is Gender.UNSPECIFIED

    def test_correcting_a_sex_is_an_update_not_a_second_child(self, uow_factory):
        from sis.domain.people import Student as S

        with uow_factory() as uow:
            uow.students.upsert_many([S(student_number="G010", full_name_ar="ع", full_name_en="A")])
            uow.commit()
        with uow_factory() as uow:
            created = uow.students.upsert_many([
                S(student_number="G010", full_name_ar="ع", full_name_en="A", gender=Gender.MALE)
            ])
            uow.commit()

        assert created["G010"] is False
        with uow_factory() as uow:
            assert uow.students.get_many([StudentNumber("G010")])["G010"].gender is Gender.MALE

    def test_re_importing_an_unchanged_child_writes_nothing(self, uow_factory):
        """`gender` joins the one list that drives the insert, the update AND the
        has-anything-changed comparison. A column in two of the three saves on creation
        and is silently ignored on every correction afterwards."""
        from sis.domain.people import Student as S

        child = S(student_number="G020", full_name_ar="ع", full_name_en="A", gender=Gender.FEMALE)
        with uow_factory() as uow:
            uow.students.upsert_many([child])
            uow.commit()
        with uow_factory() as uow:
            before = uow.students.get_many([StudentNumber("G020")])["G020"]
            uow.students.upsert_many([child])
            uow.commit()
        with uow_factory() as uow:
            after = uow.students.get_many([StudentNumber("G020")])["G020"]

        assert before == after
