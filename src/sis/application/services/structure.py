"""Generating the year levels and class sections of one academic year (invariant 3).

One method, one loop, both request shapes. `GenerateStructureCommand.resolved_counts()`
collapses "5 years x 8 classes" and "year 1 has 3, year 2 has 5" into the same ordered
sequence of `(year code, class count)` pairs before this service sees either, so there is
no branch here that could receive the idempotency fix on one side and not the other.

Idempotence is achieved by *not writing what already exists* rather than by writing
everything and letting an upsert absorb the collision. The difference is invisible until
a registrar renames "Year 3" to "Third Primary" (invariant 6 says labels are renameable)
and then re-runs generation to add a sixth rung: an upsert of every planned row would
quietly restore "Year 3" from the template, so a safe-to-repeat operation would silently
undo a human's edit. Only genuinely missing rows are handed to `upsert_many`; everything
already on file is reported back with the labels it actually carries.

Nothing is ever deleted. Asking for 3 classes in a year that has 5 creates nothing and
removes nothing -- retiring a section is a separate, deliberate act, because deleting one
here would take its enrolments and grades with it on a call the registrar thinks of as
"generate the missing classes".
"""
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Final

from sis.application.dto.structure import (
    GeneratedItem,
    GenerateStructureCommand,
    GenerateStructureResult,
)
from sis.application.ports.unit_of_work import UnitOfWork
from sis.domain.errors import (
    DomainRuleViolation,
    DuplicateCode,
    UnknownReference,
    ValidationError,
)
from sis.domain.structure import ClassSection, YearLevel
from sis.domain.value_objects import AcademicYearCode, ClassCode, YearCode

# The numeric tail of a code: `Y1` -> 1, `KG10` -> 10, `FOUNDATION` -> no match.
_TRAILING_DIGITS: Final[re.Pattern[str]] = re.compile(r"(\d+)$")

# How many existing codes to quote back when refusing. Enough to recognise the
# convention, few enough that the message stays readable in a toast.
_SAMPLE_SIZE: Final[int] = 5


def _stem(code: str) -> str:
    """The code with its numeric tail removed -- `Y1`, `Y2`, `Y10` all stem to `Y`.

    This is the whole of what "code convention" means here, and it is deliberately
    crude. A stricter comparison (template regexes, say) would refuse the registrar who
    types `classes_by_year={"G6": 3}` against a school on `G1..G5` merely because the
    unused `year_code_template` still holds its `Y{n}` default; a looser one (set
    overlap) would refuse that same call for naming a rung that does not exist yet,
    which is precisely the "add a sixth year" case invariant 3 has to allow.
    """
    return _TRAILING_DIGITS.sub("", code)


def _ordinal(code: str, position: int) -> int:
    """Which rung of the ladder this is, for `YearLevel.display_order`.

    Read from the code rather than from the loop counter, because a call that names only
    `Y6` is at position 1 of its own request and is the sixth rung of the school. Getting
    this from the counter would print the new rung above `Y1`. Codes with no number fall
    back to the position; `YearLevel.sort_key` breaks the resulting ties by code, so a
    mixed ladder degrades to alphabetical order rather than to an arbitrary one.
    """
    match = _TRAILING_DIGITS.search(code)
    return int(match.group(1)) if match else position


class StructureGenerationService:
    """Creates a year's ladder of rungs and sections, idempotently and additively.

    Depends on `UnitOfWork` alone. Its unit tests supply fakes holding dicts, which is
    why "re-running adds only what is missing" is a test that runs in a millisecond
    rather than one that needs a migration.
    """

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        # A FACTORY, not an entered unit of work -- matching the other three services.
        # Taking an already-entered one meant `generate` re-entered it and every call to
        # POST /v1/structure/generate raised "This unit of work is already open", because
        # the API layer's dependency yields the transaction it has itself opened. Owning
        # the transaction here also keeps generation atomic: a ladder is written whole or
        # not at all, rather than leaving half a year's classes behind a failure.
        self._uow_factory = uow_factory

    def generate(
        self,
        command: GenerateStructureCommand,
        *,
        allow_new_convention: bool = False,
    ) -> GenerateStructureResult:
        """Create whatever the command names and does not already exist.

        `allow_new_convention` is the escape hatch for the refusal in
        `_refuse_parallel_ladder` -- a deliberate second ladder, stated by a human, never
        a default.
        """
        academic_year_code = command.academic_year_code
        # The DTO annotates this as a value object but cannot enforce it; an API layer
        # handing over the raw string from a pydantic model would otherwise reach the
        # repositories as a `str` and match nothing.
        if not isinstance(academic_year_code, AcademicYearCode):
            academic_year_code = AcademicYearCode(academic_year_code)

        requested = command.resolved_counts()

        with self._uow_factory() as uow:
            year = uow.academic_years.get(academic_year_code)
            if year is None:
                raise UnknownReference(
                    f"no academic year is on file under {academic_year_code}; create the "
                    "year before generating its classes",
                    field="academic_year_code",
                )

            # The school comes from the year rather than from the command, and that is what
            # keeps this whole service unchanged by multi-school: a year names exactly one
            # school, so a caller that could already generate a ladder can still do it with
            # the same arguments, and cannot generate into the wrong branch by omission.
            school_code = year.school_code

            # One school's ladder, and only that school's. Read across all schools this
            # would report another branch's `Y1` as "already present" and generate nothing
            # — leaving a new school with a year, no rungs, and a run that claimed success.
            existing_levels = {
                str(l.code): l for l in uow.year_levels.list_for_school(school_code)
            }
            self._refuse_parallel_ladder(
                requested, existing_levels, allowed=allow_new_convention
            )
            # Section codes are unique per academic year, and this query is already
            # scoped to one, so the code alone is a sufficient key here.
            existing_sections = {
                str(s.code): s
                for s in uow.class_sections.list_for_year(academic_year_code)
            }

            items: list[GeneratedItem] = []
            new_levels: list[YearLevel] = []
            new_sections: list[ClassSection] = []
            # Where in `items` each newly built row sits, so the repository's verdict can
            # be written back into it once the upsert has run.
            level_slots: list[tuple[int, str]] = []
            section_slots: list[tuple[int, tuple[str, str]]] = []
            # A code repeated *within one request* is ambiguous in a way a code already
            # on file is not: `{"y1": 3, "Y1": 5}` normalises to one rung asking for two
            # different class counts, and silently taking the last would create three
            # sections where the registrar can see five written down.
            seen_levels: set[str] = set()
            seen_sections: set[tuple[str, str]] = set()

            for position, (year_code, class_count) in enumerate(requested, start=1):
                level_code = str(year_code)
                if level_code in seen_levels:
                    raise DuplicateCode(
                        f"{level_code} is named twice in one request, with two different "
                        "class counts; state it once",
                        field="classes_by_year",
                    )
                seen_levels.add(level_code)

                n = _ordinal(level_code, position)
                level = existing_levels.get(level_code)
                if level is None:
                    level = YearLevel(
                        code=year_code,
                        school_code=school_code,
                        name_en=self._format(
                            command.year_name_en_template,
                            "year_name_en_template",
                            n=n,
                            code=level_code,
                        ),
                        name_ar=self._format(
                            command.year_name_ar_template,
                            "year_name_ar_template",
                            n=n,
                            code=level_code,
                        ),
                        display_order=n,
                    )
                    new_levels.append(level)
                    level_slots.append((len(items), level_code))
                items.append(
                    GeneratedItem(
                        kind="year_level",
                        code=level_code,
                        name_ar=level.name_ar,
                        name_en=level.name_en,
                        created=False,
                    )
                )

                for suffix in command.class_suffixes[:class_count]:
                    class_code = ClassCode(
                        self._format(
                            command.class_code_template,
                            "class_code_template",
                            year=level_code,
                            suffix=suffix,
                            n=n,
                        )
                    )
                    key = (str(academic_year_code), str(class_code))
                    if key in seen_sections:
                        raise DuplicateCode(
                            f"{class_code} would be generated twice in one request; "
                            "check class_suffixes and class_code_template",
                            field="class_code_template",
                        )
                    seen_sections.add(key)
                    section = existing_sections.get(str(class_code))
                    if section is None:
                        section = ClassSection(
                            code=class_code,
                            academic_year_code=academic_year_code,
                            year_level_code=year_code,
                            # The rung's *stored* name, so sections added to a renamed
                            # year level read "Third Primary C" rather than reviving the
                            # template's "Year 3 C" beside its own siblings.
                            name_en=self._format(
                                command.class_name_en_template,
                                "class_name_en_template",
                                year=level_code,
                                suffix=suffix,
                                year_name=level.name_for("en"),
                                n=n,
                            ),
                            name_ar=self._format(
                                command.class_name_ar_template,
                                "class_name_ar_template",
                                year=level_code,
                                suffix=suffix,
                                year_name=level.name_for("ar"),
                                n=n,
                            ),
                        )
                        new_sections.append(section)
                        section_slots.append((len(items), key))
                    items.append(
                        GeneratedItem(
                            kind="class_section",
                            code=str(section.code),
                            name_ar=section.name_ar,
                            name_en=section.name_en,
                            created=False,
                            # Read off the row, not off the request: a section moved to
                            # another rung by hand must be reported where it actually is.
                            year_level_code=str(section.year_level_code),
                        )
                    )

            # Rungs before sections, in one transaction: a section carries its year level
            # code, so the reverse order fails a foreign key on a school's first run.
            level_flags = uow.year_levels.upsert_many(new_levels) if new_levels else {}
            section_flags = (
                uow.class_sections.upsert_many(new_sections) if new_sections else {}
            )
            uow.commit()

        # The repository's flag, not our own read, decides `created`: between the load
        # above and the write, a concurrent request may have created the same rung, and
        # reporting it as newly created would tell one of the two registrars that their
        # click was the one that took effect. Absent keys default to True, since only
        # rows believed missing were sent.
        for index, code in level_slots:
            items[index] = replace(items[index], created=level_flags.get(code, True))
        for index, key in section_slots:
            items[index] = replace(items[index], created=section_flags.get(key, True))

        return GenerateStructureResult(
            academic_year_code=str(academic_year_code), items=tuple(items)
        )

    @staticmethod
    def _refuse_parallel_ladder(
        requested: Sequence[tuple[YearCode, int]],
        existing: Mapping[str, YearLevel],
        *,
        allowed: bool,
    ) -> None:
        """Refuse to build a second ladder beside the one the school already stands on.

        The failure this exists to stop is silent and expensive. A school running on
        `G1..G6` whose registrar generates with the default `Y{n}` template gets a full
        set of `Y1..Y6` rungs and their sections created *successfully*: nothing
        collides, nothing errors, and the result reads "48 created". But every child on
        file is enrolled against the `G` sections, so the new ladder is empty and the old
        one is invisible to anything filtering on the new codes. The registrar's evidence
        that something is wrong is a class list full of empty classes, weeks later.

        Codes are identity (invariant 6), so this cannot be repaired by renaming; it is
        repaired by deleting a parallel ladder that grades may by then hang from. Hence a
        refusal *before* the first write rather than a warning after it.
        """
        if allowed or not existing:
            return
        if {_stem(str(code)) for code, _ in requested} & {
            _stem(code) for code in existing
        }:
            return
        sample = ", ".join(sorted(existing)[:_SAMPLE_SIZE])
        raise DomainRuleViolation(
            f"this school's year levels are coded {sample}; the requested codes follow a "
            "different convention and would create a second ladder beside them, leaving "
            "every existing enrolment on the old codes. Re-run using the codes already "
            "on file, or state explicitly that a second ladder is intended.",
            field="year_code_template",
        )

    @staticmethod
    def _format(template: str, field: str, **values: object) -> str:
        """Interpolate a caller-supplied template, turning its typos into bad input.

        The command validates that the *required* placeholders are present but not that
        every placeholder is one this service supplies, so `"Year {year}"` would escape as
        a bare `KeyError` -- which, per `SisError`, is the shape reserved for "the code is
        broken". It is not; it is a form the registrar filled in wrongly.
        """
        try:
            return template.format(**values)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValidationError(
                f"{field} uses a placeholder this service does not supply: {exc}",
                field=field,
            ) from None
