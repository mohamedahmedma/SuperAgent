"""The value objects, which are where a bad spreadsheet cell is supposed to die.

`Percentage` gets most of the attention because it is the type that carries invariant 1:
0 and 100 are both marks a child can earn, so both ends are inclusive, while the *absence*
of a mark is `None` and never this type at all. The rejections matter as much as the
acceptances — a NaN reaching storage is a grade that is neither present nor absent, and
every later comparison against it silently answers "no".

Framework-free, like the module under test: no fixtures, no database, no clock.
"""
import math

import pytest

from sis.domain.errors import (
    InvalidCode,
    InvalidPercentage,
    InvalidPhone,
    InvalidStudentNumber,
)
from sis.domain.value_objects import (
    ClassCode,
    Percentage,
    Phone,
    StudentNumber,
    SubjectCode,
    TermCode,
    YearCode,
)


# --------------------------------------------------------------------------- Percentage


@pytest.mark.parametrize("value", [0, 0.0, 1, 49.5, 99.99, 100, 100.0])
def test_percentage_accepts_the_whole_inclusive_range(value: float) -> None:
    """Both ends are inclusive: full marks are earned, and so is a zero."""
    assert Percentage(value).value == float(value)


@pytest.mark.parametrize(
    "value",
    [101, 100.0001, -1, -0.5, float("nan"), float("inf"), float("-inf")],
)
def test_percentage_rejects_out_of_range_and_non_finite_values(value: float) -> None:
    """NaN and infinity are listed beside 101 because they fail *differently*.

    Every comparison against NaN answers False, so a range test written as
    `if 0 <= n <= 100: ok` lets NaN straight through. This case is what pins the negated
    form in `Percentage._validated` in place.
    """
    with pytest.raises(InvalidPercentage):
        Percentage(value)


def test_percentage_rejects_values_that_are_not_numbers() -> None:
    for value in ("85", None, True):
        with pytest.raises(InvalidPercentage):
            Percentage(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("cell", [None, "", "  ", "-", "N/A", "none", "غ", "لم يرصد"])
def test_parse_optional_reads_an_absent_cell_as_none_and_not_as_zero(cell: object) -> None:
    """Invariant 1 at its source. `None` here is what a blank column means everywhere.

    Asserted as `is None` *and* as unequal to zero: a parser that returned `Percentage(0)`
    for a blank would satisfy a truthiness check and would tell a parent their child
    scored nothing in a subject nobody has marked.
    """
    parsed = Percentage.parse_optional(cell)
    assert parsed is None
    assert parsed != 0
    assert parsed != Percentage(0)


def test_parse_optional_reads_a_written_zero_as_an_earned_zero() -> None:
    """The other side of the pair: `"0"` is a figure a teacher stated."""
    assert Percentage.parse_optional("0") == Percentage(0)
    assert Percentage.parse_optional(0) == Percentage(0)


def test_parse_requires_a_figure_where_parse_optional_permits_a_blank() -> None:
    assert Percentage.parse("88%") == Percentage(88)
    with pytest.raises(InvalidPercentage):
        Percentage.parse("")


def test_percentage_is_ordered_so_comparisons_need_no_reach_into_value() -> None:
    assert Percentage(49) < Percentage(50) <= Percentage(50)


def test_percentage_stores_the_stated_figure_without_rounding() -> None:
    """Rounding here would change what the teacher wrote; presentation rounds, not this."""
    assert Percentage(66.6666).value == 66.6666


def test_percentage_of_nan_is_caught_before_it_can_be_compared() -> None:
    """Guards the ordering above: a NaN admitted here is unequal to itself forever."""
    with pytest.raises(InvalidPercentage):
        Percentage(float("nan"))
    assert not math.isnan(Percentage(0).value)


# --------------------------------------------------------------------------- codes


def test_codes_normalise_case_and_invisible_characters() -> None:
    """`"3a"` and `"3A‏"` are one class, not three. The registrar cannot see them."""
    assert ClassCode(" 3a ").value == "3A"
    assert SubjectCode("math‏").value == "MATH"
    assert TermCode("﻿2026-t1").value == "2026-T1"


def test_codes_reject_what_does_not_survive_a_url_or_a_csv_cell() -> None:
    for bad in ("", "   ", "3 A", "3/A", "رياضيات", "-3A"):
        with pytest.raises(InvalidCode):
            SubjectCode(bad)


def test_a_numeric_student_number_from_excel_keeps_its_digits() -> None:
    """Excel types a numeric column as a number; `12345.0` must not become `"12345.0"`."""
    assert StudentNumber(12345).value == "12345"
    assert StudentNumber(12345.0).value == "12345"
    with pytest.raises(InvalidStudentNumber):
        StudentNumber(123.45)


def test_a_student_number_keeps_leading_zeros() -> None:
    """`int("0071")` is 71 — a different child, or no child at all."""
    assert StudentNumber("0071").value == "0071"


def test_distinct_code_types_are_not_interchangeable() -> None:
    """Why these are separate types: a swapped argument pair must not compare equal."""
    assert SubjectCode("T1") != TermCode("T1")
    assert YearCode("Y3") != ClassCode("Y3")


# --------------------------------------------------------------------------- Phone

EG = "+20"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Already international, in the three ways a human writes that.
        ("+201001234567", "+201001234567"),
        ("00201001234567", "+201001234567"),
        ("201001234567", "+201001234567"),
        # National, with the separators registrars actually type.
        ("01001234567", "+201001234567"),
        ("0100 123 4567", "+201001234567"),
        ("0100-123-4567", "+201001234567"),
        ("(0100) 123.4567", "+201001234567"),
        # Arabic-Indic and extended Arabic-Indic digits.
        ("٠١٠٠١٢٣٤٥٦٧", "+201001234567"),
        ("۰۱۰۰۱۲۳۴۵۶۷", "+201001234567"),
        # A foreign parent, unaffected by the default country.
        ("+966501234567", "+966501234567"),
    ],
)
def test_every_spelling_of_one_number_collapses_to_one_value(
    raw: str, expected: str
) -> None:
    """The deduplication the guardian importer rests on.

    These are all the same woman. If two of them failed to collide she would become two
    guardians, each holding half her children, and nothing would report it.
    """
    assert Phone.parse(raw, default_country_code=EG).value == expected


@pytest.mark.parametrize("raw", [1001234567, 1001234567.0])
def test_a_phone_survives_excel_typing_it_as_a_number(raw: object) -> None:
    """The failure that would otherwise make half a school's parents unreachable.

    A phone column formatted as a number arrives with its leading zero already gone —
    and `str(1001234567.0)` is `'1001234567.0'`, which reaches nobody at all.
    """
    assert Phone.parse(raw, default_country_code=EG).value == "+201001234567"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_a_blank_phone_is_absence_not_an_error(raw: object) -> None:
    """`None` so a caller can tell "no second number" from "unreadable second number"."""
    assert Phone.parse_optional(raw, default_country_code=EG) is None


@pytest.mark.parametrize(
    "raw",
    [
        "abc",
        # Letters are refused rather than stripped: discarding them leaves a
        # plausible-looking number nobody answers, with no record anything was lost.
        "0100 call after 5",
        "123",  # too few digits to be a phone
        "+2010012345678901",  # past E.164's fifteen
        True,  # a bool is an int, and would otherwise become the number 1
    ],
)
def test_an_unusable_phone_is_refused(raw: object) -> None:
    with pytest.raises(InvalidPhone):
        Phone.parse(raw, default_country_code=EG)


def test_constructing_a_phone_directly_requires_e164() -> None:
    """Normalisation happens in exactly one place.

    An entity rebuilt from a stored row must not quietly accept a half-normalised number,
    because a national-format value in the database is one that matches no other spelling
    of itself.
    """
    assert Phone("+201001234567").value == "+201001234567"
    with pytest.raises(InvalidPhone):
        Phone("01001234567")


def test_constructing_a_phone_rejects_unfolded_arabic_digits() -> None:
    """`str.isdigit()` answers True for ٠١٢, so the check cannot be written with it.

    A number stored in Arabic-Indic digits compares unequal to the same number typed on a
    Latin keyboard — the exact duplicate this type exists to prevent.
    """
    with pytest.raises(InvalidPhone):
        Phone("+٢٠١٠٠١٢٣٤٥٦٧")


def test_a_local_number_beginning_with_the_country_code_is_not_truncated() -> None:
    """`20xxxxx` is ambiguous, and the length is what resolves it.

    Treating every bare number starting `20` as already-international would eat the first
    two digits of a perfectly good local landline.
    """
    assert Phone.parse("2012345", default_country_code=EG).value == "+202012345"
