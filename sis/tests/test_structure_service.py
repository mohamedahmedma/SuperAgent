"""Structure generation proved against dictionaries, with no database anywhere.

Why this file has no engine, no session and no migration: `StructureGenerationService`
depends on the `UnitOfWork` port and on nothing else, so its whole world here is three
dicts and a counter. The day this file needs a fixture is the day the layering broke, so
the strongest assertion in the module is the one made by what it does *not* import.

The fakes implement only the methods the service actually calls. A fake mirroring the
full Protocol would quietly absorb a use case that started reaching for a query it has no
business needing; here that reach is an `AttributeError` in a millisecond-long test.
"""
from collections.abc import Collection, Mapping, Sequence
from datetime import date

import pytest

from sis.application.dto import GenerateStructureCommand, GenerateStructureResult
from sis.application.services.structure import StructureGenerationService
from sis.domain.errors import DomainRuleViolation
from sis.domain.structure import AcademicYear, ClassSection, YearLevel
from sis.domain.value_objects import AcademicYearCode, YearCode

YEAR = AcademicYearCode("2025-2026")


# --- fakes -----------------------------------------------------------------


class _Years:
    """The one lookup generation makes before it will write anything."""

    def __init__(self, *years: AcademicYear) -> None:
        self.rows = {str(y.code): y for y in years}

    def get(self, code: AcademicYearCode) -> AcademicYear | None:
        return self.rows.get(str(code))


class _Levels:
    """Year levels, plus a log of every write attempt.

    `writes` is what makes "idempotent" testable as an absence: asserting that a re-run
    reports 45 existing items proves the *report* is right, while asserting that the
    repository was never called proves nothing was sent to the database at all. The
    second is the claim invariant 3 actually makes.
    """

    def __init__(self, *levels: YearLevel) -> None:
        self.rows: dict[str, YearLevel] = {str(l.code): l for l in levels}
        self.writes: list[tuple[str, ...]] = []

    def list_for_school(self, school_code: object) -> Sequence[YearLevel]:
        """One school's rungs, and only that school's.

        The filter is real rather than ignored, because the failure it guards against is
        the one the generator would otherwise have: reading across every school, another
        branch's `Y1` reports as "already present" and nothing is generated — leaving a new
        school with a year, no rungs, and a run that claimed success.
        """
        return sorted(
            (
                level
                for level in self.rows.values()
                if str(level.school_code) == str(school_code)
            ),
            key=lambda level: level.sort_key,
        )

    def upsert_many(self, levels: Sequence[YearLevel]) -> Mapping[str, bool]:
        self.writes.append(tuple(str(l.code) for l in levels))
        flags: dict[str, bool] = {}
        for level in levels:
            code = str(level.code)
            flags[code] = code not in self.rows
            self.rows[code] = level
        return flags


class _Sections:
    """Class sections keyed by `(academic year, code)` — never by code alone."""

    def __init__(self, *sections: ClassSection) -> None:
        self.rows: dict[tuple[str, str], ClassSection] = {s.identity: s for s in sections}
        self.writes: list[tuple[str, ...]] = []

    def list_for_year(
        self, academic_year_code: AcademicYearCode, *, year_level_code: YearCode | None = None
    ) -> Sequence[ClassSection]:
        return [s for key, s in self.rows.items() if key[0] == str(academic_year_code)]

    def upsert_many(
        self, sections: Sequence[ClassSection]
    ) -> Mapping[tuple[str, str], bool]:
        self.writes.append(tuple(str(s.code) for s in sections))
        flags: dict[tuple[str, str], bool] = {}
        for section in sections:
            flags[section.identity] = section.identity not in self.rows
            self.rows[section.identity] = section
        return flags


class _Uow:
    """A unit of work that is a `with` block and a counter. No transaction to speak of."""

    def __init__(self, levels: _Levels, sections: _Sections) -> None:
        self.academic_years = _Years(
            AcademicYear(
                code=YEAR,
                school_code="MAIN",
                name_en="2025-2026",
                name_ar="٢٠٢٥-٢٠٢٦",
                starts_on=date(2025, 9, 1),
                ends_on=date(2026, 6, 30),
            )
        )
        self.year_levels = levels
        self.class_sections = sections
        self.commits = 0

    def __enter__(self) -> "_Uow":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1


def _school(*, levels: Collection[YearLevel] = (), sections: Collection[ClassSection] = ()) -> _Uow:
    return _Uow(_Levels(*levels), _Sections(*sections))


def _generate(uow: _Uow, command: GenerateStructureCommand, **kwargs: object) -> GenerateStructureResult:
    # A factory, matching the real constructor: the service opens its own transaction.
    # Handing it an already-entered unit of work is what broke POST /v1/structure/generate.
    return StructureGenerationService(lambda: uow).generate(command, **kwargs)  # type: ignore[arg-type]


def _codes(result: GenerateStructureResult, kind: str, *, created: bool | None = None) -> tuple[str, ...]:
    return tuple(
        item.code
        for item in result.items
        if item.kind == kind and (created is None or item.created is created)
    )


# --- tests -----------------------------------------------------------------


def test_uniform_generation_creates_every_rung_and_every_section() -> None:
    uow = _school()

    result = _generate(
        uow, GenerateStructureCommand(academic_year_code=YEAR, year_count=5, classes_per_year=8)
    )

    assert _codes(result, "year_level") == ("Y1", "Y2", "Y3", "Y4", "Y5")
    assert set(_codes(result, "class_section")) == {
        f"Y{n}{suffix}" for n in range(1, 6) for suffix in "ABCDEFGH"
    }
    assert result.created_count == 45
    assert result.existing_count == 0
    assert uow.commits == 1


def test_per_year_counts_run_through_the_same_path_as_uniform() -> None:
    """Different widths per rung, one loop — the shape invariant 3 depends on.

    Two generation paths would mean the idempotency fix lands on one of them, and the
    school whose years are uneven is the one that silently doubles on a second click.
    """
    uow = _school()

    result = _generate(
        uow, GenerateStructureCommand(academic_year_code=YEAR, classes_by_year={"Y1": 3, "Y2": 5})
    )

    assert _codes(result, "class_section") == ("Y1A", "Y1B", "Y1C", "Y2A", "Y2B", "Y2C", "Y2D", "Y2E")
    assert result.created_count == 10
    assert ("2025-2026", "Y2E") in uow.class_sections.rows


def test_rerunning_the_same_request_creates_nothing_and_keeps_renamed_labels() -> None:
    """The second click. Nothing is written, and a human's rename survives it.

    The repository log is checked as well as the report: an upsert of all 45 planned rows
    would also report "45 already present", while quietly restoring "Year 3" over the
    registrar's "Third Primary". Idempotent here means *no write was sent*, not that the
    database absorbed one.
    """
    command = GenerateStructureCommand(
        academic_year_code=YEAR, year_count=5, classes_per_year=8
    )
    uow = _school()
    _generate(uow, command)
    uow.year_levels.rows["Y3"] = uow.year_levels.rows["Y3"].renamed(name_en="Third Primary")
    uow.year_levels.writes.clear()
    uow.class_sections.writes.clear()

    result = _generate(uow, command)

    assert result.created_count == 0
    assert result.existing_count == 45
    assert uow.year_levels.writes == []
    assert uow.class_sections.writes == []
    assert [item.name_en for item in result.items if item.code == "Y3"] == ["Third Primary"]


def test_a_sixth_year_is_added_without_touching_the_first_five() -> None:
    command = GenerateStructureCommand(
        academic_year_code=YEAR, year_count=5, classes_per_year=8
    )
    uow = _school()
    _generate(uow, command)
    uow.year_levels.writes.clear()
    uow.class_sections.writes.clear()

    result = _generate(
        uow, GenerateStructureCommand(academic_year_code=YEAR, year_count=6, classes_per_year=8)
    )

    assert _codes(result, "year_level", created=True) == ("Y6",)
    assert _codes(result, "class_section", created=True) == tuple(f"Y6{s}" for s in "ABCDEFGH")
    assert result.existing_count == 45
    assert uow.year_levels.writes == [("Y6",)]
    # `display_order` comes from the code, not from the loop: a request naming only `Y6`
    # is at position 1 of itself and would otherwise print above `Y1`.
    assert uow.year_levels.rows["Y6"].display_order == 6


def test_a_fourth_class_is_added_to_one_year_only() -> None:
    uow = _school()
    _generate(
        uow, GenerateStructureCommand(academic_year_code=YEAR, year_count=5, classes_per_year=3)
    )
    uow.class_sections.writes.clear()

    result = _generate(
        uow, GenerateStructureCommand(academic_year_code=YEAR, classes_by_year={"Y2": 4})
    )

    assert _codes(result, "class_section", created=True) == ("Y2D",)
    assert uow.class_sections.writes == [("Y2D",)]
    # Nothing was removed from the rungs the request did not name, and nothing was added.
    assert sum(1 for key in uow.class_sections.rows if key[1].startswith("Y1")) == 3
    assert sum(1 for key in uow.class_sections.rows if key[1].startswith("Y2")) == 4


def test_a_conflicting_code_convention_is_refused_before_any_write() -> None:
    """A school on `G1..G6` must not silently acquire a parallel `Y1..Y6` ladder.

    Refused rather than warned about, because the failure is invisible on success: every
    row is created, nothing collides, and the registrar's only evidence is a set of empty
    classes weeks later. Codes are identity, so the repair is deletion, not a rename.
    """
    uow = _school(
        levels=[
            YearLevel(code=YearCode(f"G{n}"), school_code="MAIN", name_en=f"Grade {n}", name_ar=f"صف {n}", display_order=n)
            for n in (1, 2, 3)
        ]
    )

    with pytest.raises(DomainRuleViolation) as raised:
        _generate(
            uow,
            GenerateStructureCommand(academic_year_code=YEAR, year_count=3, classes_per_year=2),
        )

    assert raised.value.code == "rule_violation"
    assert raised.value.field == "year_code_template"
    assert uow.year_levels.writes == []
    assert uow.class_sections.writes == []
    assert uow.commits == 0


def test_a_second_ladder_is_built_when_a_human_states_it_explicitly() -> None:
    uow = _school(
        levels=[YearLevel(code=YearCode("G1"), school_code="MAIN", name_en="Grade 1", name_ar="صف ١", display_order=1)]
    )

    result = _generate(
        uow,
        GenerateStructureCommand(academic_year_code=YEAR, year_count=2, classes_per_year=1),
        allow_new_convention=True,
    )

    assert _codes(result, "year_level", created=True) == ("Y1", "Y2")
    assert uow.commits == 1
