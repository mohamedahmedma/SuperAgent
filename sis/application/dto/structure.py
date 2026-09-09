"""The command and result for generating a year's ladder of classes (decision 3).

Why one command carries two mutually exclusive shapes instead of two commands existing:
a registrar asks for "five years of eight classes" and for "year 1 has 3, year 2 has 5"
in the same breath, and both are the same operation with a different count per rung.
`resolved_counts()` collapses either shape into one ordered sequence of
`(year code, class count)` pairs, so there is exactly *one* generation loop downstream.
Two commands would mean two loops, and the second is the one that never receives the
idempotency fix -- generation stops being safe to re-run for half the school.

Everything here is a plain dataclass. The API layer owns pydantic; a DTO that inherited
from `BaseModel` would make these services untestable without a request object.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from string import ascii_uppercase
from typing import ClassVar, Final, Literal

from sis.domain.errors import ValidationError
from sis.domain.value_objects import AcademicYearCode, YearCode

# Suffixes for the sections inside one year level: 3A, 3B, 3C. Latin letters even in an
# Arabic school, because this is half of a *code* (decision 7) and codes have to survive
# a URL, a CSV cell and a Moodle idnumber unescaped; the Arabic name goes in `name_ar`.
_DEFAULT_SUFFIXES: Final[tuple[str, ...]] = tuple(ascii_uppercase)

# What a code template must interpolate. A template missing its placeholder generates
# the same code N times, which idempotency then collapses into a single row -- so the
# operator asks for 5 year levels, is told "1 created, 4 already present", and nothing
# looks like an error anywhere.
_YEAR_PLACEHOLDER: Final[str] = "{n}"
_CLASS_PLACEHOLDERS: Final[tuple[str, ...]] = ("{year}", "{suffix}")

StructureItemKind = Literal["year_level", "class_section"]


@dataclass(frozen=True, slots=True)
class GenerateStructureCommand:
    """Generate the year levels and class sections of one academic year.

    Give **either** `classes_per_year` with `year_count` (uniform) **or**
    `classes_by_year` (explicit, one entry per rung). Supplying both, or neither, is
    refused at construction: a command that silently preferred one field would create a
    structure nobody asked for, and the registrar would discover it as missing classes
    weeks later when a teacher cannot find a section.

    Only the *code* templates are checked for their placeholders. Name templates are not,
    because names are labels and duplicate labels are harmless and renameable
    (decision 7) -- duplicate codes are the failure described above.
    """

    academic_year_code: AcademicYearCode

    # Uniform form: `year_count` rungs, each with `classes_per_year` sections.
    year_count: int | None = None
    classes_per_year: int | None = None

    # Explicit form: year level code -> how many sections that rung gets. Insertion
    # order is preserved and is the creation order, so two runs of the same request
    # produce the same sequence and previews are comparable.
    classes_by_year: Mapping[str, int] | None = None

    year_code_template: str = "Y{n}"
    year_name_en_template: str = "Year {n}"
    year_name_ar_template: str = "السنة {n}"
    class_code_template: str = "{year}{suffix}"
    class_name_en_template: str = "{year_name} {suffix}"
    class_name_ar_template: str = "{year_name} {suffix}"
    class_suffixes: tuple[str, ...] = field(default=_DEFAULT_SUFFIXES)

    # Ceilings on a typo, not on a school. A mistyped "500" in a form that generates
    # rows without further confirmation writes thousands of classes that then have to be
    # deleted by hand, one enrolment check at a time.
    MAX_YEARS: ClassVar[int] = 30
    MAX_CLASSES_PER_YEAR: ClassVar[int] = 60

    def __post_init__(self) -> None:
        uniform = self.classes_per_year is not None
        explicit = self.classes_by_year is not None
        if uniform == explicit:
            raise ValidationError(
                "give either classes_per_year with year_count (uniform) or "
                "classes_by_year (one count per year level), not both and not neither",
                field="classes_per_year",
            )
        if uniform:
            if self.year_count is None:
                raise ValidationError(
                    "year_count is required alongside classes_per_year",
                    field="year_count",
                )
            self._check_years(self.year_count)
            self._check_classes(self.classes_per_year, field_name="classes_per_year")
        else:
            assert self.classes_by_year is not None  # narrowed by `explicit`
            if self.year_count is not None:
                raise ValidationError(
                    "year_count is implied by the keys of classes_by_year; remove it",
                    field="year_count",
                )
            if not self.classes_by_year:
                raise ValidationError(
                    "classes_by_year names no year levels", field="classes_by_year"
                )
            self._check_years(len(self.classes_by_year))
            for code, count in self.classes_by_year.items():
                self._check_classes(count, field_name=f"classes_by_year[{code}]")

        if _YEAR_PLACEHOLDER not in self.year_code_template:
            raise ValidationError(
                f"year_code_template must contain {_YEAR_PLACEHOLDER}",
                field="year_code_template",
            )
        missing = [p for p in _CLASS_PLACEHOLDERS if p not in self.class_code_template]
        if missing:
            raise ValidationError(
                f"class_code_template must contain {' and '.join(missing)}",
                field="class_code_template",
            )
        if not self.class_suffixes:
            raise ValidationError("class_suffixes is empty", field="class_suffixes")

    def _check_years(self, count: int) -> None:
        if count < 1 or count > self.MAX_YEARS:
            raise ValidationError(
                f"a year count must be between 1 and {self.MAX_YEARS} (got {count})",
                field="year_count",
            )

    def _check_classes(self, count: int | None, *, field_name: str) -> None:
        # Zero is allowed: creating the rungs now and their sections after enrolment
        # numbers are known is a real registrar workflow, and refusing it would push
        # that school into hand-creating every section.
        if count is None or count < 0 or count > len(self.class_suffixes):
            # Quote the suffixes themselves. The old message said only "between 0 and 1",
            # which is true and useless: the reader has asked for 4 classes and is told a
            # bound, with nothing naming the field that produced it. The commonest cause
            # is a suffix list typed as "ABCD" instead of "A,B,C,D", which parses as one
            # suffix -- and that is invisible unless the list is printed back.
            shown = ", ".join(self.class_suffixes[:8])
            if len(self.class_suffixes) > 8:
                shown += ", ..."
            raise ValidationError(
                f"{field_name} must be between 0 and {len(self.class_suffixes)}, got "
                f"{count}. Only {len(self.class_suffixes)} class suffix"
                f"{'' if len(self.class_suffixes) == 1 else 'es'} "
                f"{'is' if len(self.class_suffixes) == 1 else 'are'} available: [{shown}]. "
                "Give one suffix per class, comma separated (A,B,C,D), or leave the "
                "suffixes field blank to use A-Z.",
                field=field_name,
            )
        if count > self.MAX_CLASSES_PER_YEAR:
            raise ValidationError(
                f"{field_name} exceeds the limit of {self.MAX_CLASSES_PER_YEAR}",
                field=field_name,
            )

    def resolved_counts(self) -> tuple[tuple[YearCode, int], ...]:
        """The single normalised form both shapes reduce to, in creation order.

        Generated codes are built through `YearCode` rather than left as strings so that
        a template producing `"Y 1"` fails here, before anything is written -- otherwise
        the first four rungs are created and the fifth aborts the request, leaving a
        half-generated year that the next run has to reconcile.
        """
        if self.classes_by_year is not None:
            return tuple((YearCode(code), n) for code, n in self.classes_by_year.items())

        per_year, years = self.classes_per_year, self.year_count
        if per_year is None or years is None:  # unreachable; __post_init__ guarantees it
            raise ValidationError(
                "uniform generation needs both year_count and classes_per_year",
                field="classes_per_year",
            )
        return tuple(
            (YearCode(self.year_code_template.format(n=n)), per_year)
            for n in range(1, years + 1)
        )


@dataclass(frozen=True, slots=True)
class GeneratedItem:
    """One year level or class section the run touched, and whether it is new.

    `created=False` is a success, not a skip: decision 3 makes generation idempotent, so
    re-running a request that produced 40 classes must report 40 already-present items
    rather than 40 conflicts. The registrar reads this list to answer "did my second
    click double the school", and the answer has to be legible at a glance.
    """

    kind: StructureItemKind
    code: str
    name_ar: str
    name_en: str
    created: bool
    # Set on class sections, `None` on year levels: which rung the section belongs to.
    year_level_code: str | None = None


@dataclass(frozen=True, slots=True)
class GenerateStructureResult:
    academic_year_code: str
    items: tuple[GeneratedItem, ...]

    @property
    def created(self) -> tuple[GeneratedItem, ...]:
        return tuple(item for item in self.items if item.created)

    @property
    def already_existing(self) -> tuple[GeneratedItem, ...]:
        return tuple(item for item in self.items if not item.created)

    @property
    def created_count(self) -> int:
        return sum(1 for item in self.items if item.created)

    @property
    def existing_count(self) -> int:
        return sum(1 for item in self.items if not item.created)


# -- The year's term sections -------------------------------------------------------
#
# A school states how many terms it runs (`schools.term_count`, 1, 2 or 3). The year's
# term sections follow that number rather than being typed in a second time, which is the
# whole of stage 6's first requirement: choosing two terms produces two term sections.
#
# Naming is fixed here rather than asked for. A registrar setting a school up has already
# answered "how many terms"; asking them to name Term 1 "Term 1" three more times is a
# form that exists only to be filled in identically every time. The labels stay editable
# afterwards — they are labels, and `code` is the identity.

#: Bilingual labels by sequence. Index 0 is unused so the tuple reads 1-based, matching
#: `sequence` and the code suffix — an off-by-one here would name Term 2 "الفصل الأول".
TERM_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("", ""),
    ("Term 1", "الفصل الأول"),
    ("Term 2", "الفصل الثاني"),
    ("Term 3", "الفصل الثالث"),
)


def term_code_for(academic_year_code: str, sequence: int) -> str:
    """`2025-2026` + 2 -> `2025-2026-T2`.

    Derived rather than stored so the same year always produces the same term codes, which
    is what makes the sync idempotent: a second run finds the codes it would have created
    already present instead of creating a parallel set beside them. Fits `TermCode`'s 32
    characters by construction — an academic year code is at most 16.
    """
    return f"{academic_year_code}-T{sequence}"


@dataclass(frozen=True, slots=True)
class TermPlan:
    """What one run of the term sync did, and what it deliberately did not do.

    Returned rather than swallowed because the interesting case is `kept`. Dropping a
    school from three terms to two cannot silently delete a term that already holds marks
    — so the sync removes only the empty shells it would itself have created, keeps
    anything a child has been graded in, and says which. A caller that ignores this gets
    correct data; a caller that shows it can tell the registrar why the year still has
    three terms on screen.
    """

    academic_year_code: str
    #: What the school asked for: `schools.term_count`.
    term_count: int
    #: Codes this run inserted.
    created: tuple[str, ...] = ()
    #: Surplus terms removed because nothing referenced them.
    removed: tuple[str, ...] = ()
    #: Surplus terms left in place because marks are stated against them.
    kept: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether this run wrote anything. A settled year syncs to nothing at all."""
        return bool(self.created or self.removed)
