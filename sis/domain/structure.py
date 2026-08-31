"""The skeleton a school hangs its children on: years, rungs, classes, terms, subjects.

These are the things a registrar sets up in August and then mostly leaves alone. They
are frozen because they are *referenced* by enrolments and grades rather than owned by
them: a mutable `ClassSection` handed to three services is a class that changes shape
under two of them, and the symptom of that is a report card whose header disagrees with
its body.

Immutability is also how decision 7 — codes immutable, labels renameable — is expressed
rather than merely documented. There is no setter for a code, and `renamed()` returns a
new object carrying the *same* code, so the only rename a caller can express is one that
keeps every enrolment and grade attached. Renaming "3A" to "3 Alpha" cannot accidentally
become "create a new class and orphan its students", because nothing here can change a
code at all.

Constructors accept either the value object or the raw cell it came from, so a parser
can build a `Subject` straight from a spreadsheet row and get `InvalidCode` at the row
that caused it. No framework imports: these types are what the service unit tests are
written against, with no database in sight.
"""
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum
from typing import ClassVar, TypeVar

from sis.domain.errors import InvalidDateRange, ValidationError
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    SchoolCode,
    SubjectCode,
    TermCode,
    YearCode,
)

_C = TypeVar("_C")
_E = TypeVar("_E", bound="_Named")


def _coerce(field: str, kind: type[_C], entity: object) -> None:
    """Replace a raw cell with its value object, in place, during `__post_init__`."""
    raw = getattr(entity, field)
    if not isinstance(raw, kind):
        raw = kind(raw)  # type: ignore[call-arg]  # every code type takes one argument
        object.__setattr__(entity, field, raw)


def _coerce_names(entity: object) -> None:
    """Trim both labels and refuse a nameable thing with no name in either language.

    Blank in *one* language is allowed and common — a school types Arabic names for its
    classes and English ones for nothing else — but blank in both is rejected here
    rather than accepted and rendered later, because an empty label reaches the parent
    as a report card line with a mark and no subject beside it.
    """
    name_en = str(getattr(entity, "name_en", "") or "").strip()
    name_ar = str(getattr(entity, "name_ar", "") or "").strip()
    if not name_en and not name_ar:
        raise ValidationError(
            "a name is required in at least one of English or Arabic", field="name_en"
        )
    object.__setattr__(entity, "name_en", name_en)
    object.__setattr__(entity, "name_ar", name_ar)


def _check_range(starts_on: date, ends_on: date, field: str) -> None:
    """A single-day period is legal; an inverted one is not. See `InvalidDateRange`."""
    if ends_on < starts_on:
        raise InvalidDateRange(
            f"end date {ends_on.isoformat()} precedes start date {starts_on.isoformat()}",
            field=field,
        )


class _Named:
    """Bilingual-label behaviour, shared by everything in this module (invariant 7)."""

    __slots__ = ()

    name_en: str
    name_ar: str

    def name_for(self, language: str) -> str:
        """The label to show, falling back to the other language when one is blank.

        The fallback is the whole point: half of a school's structure is named in Arabic
        only, and returning `""` for its English label produces a printed class list of
        blank rows that reads as data loss rather than as a missing translation.
        """
        preferred = self.name_ar if language.lower().startswith("ar") else self.name_en
        return preferred or self.name_ar or self.name_en

    def renamed(self: _E, *, name_en: str | None = None, name_ar: str | None = None) -> _E:
        """A copy wearing new labels and the same code — decision 7 in one method."""
        return replace(
            self,
            name_en=self.name_en if name_en is None else name_en,
            name_ar=self.name_ar if name_ar is None else name_ar,
        )


class Stage(StrEnum):
    """The division of a school a rung belongs to: garden, primary, preparatory, secondary.

    A grouping, deliberately, and not a level of the hierarchy in its own right. Schools in
    scope talk about "the garden" and "the secondary school" as parts of one institution,
    and a registrar looking for Year 8 looks under preparatory — so this exists to make a
    list of fourteen rungs readable, and for nothing else. It carries no rules: no rung is
    barred from a class, a term or a subject because of its stage, and a school that does
    not use the distinction leaves everything in UNSPECIFIED and never sees it.

    Stored as text rather than an integer so a database dump is readable and so adding a
    stage later is not a renumbering of the ones already stored. UNSPECIFIED exists because
    it is what every rung was before this field did, and a migration must not have to guess.
    """

    UNSPECIFIED = "unspecified"
    GARDEN = "garden"
    PRIMARY = "primary"
    PREPARATORY = "preparatory"
    SECONDARY = "secondary"

    @property
    def order(self) -> int:
        """Youngest first, which is the order a school lists its own divisions in."""
        return _STAGE_ORDER[self]


class SchoolLanguage(StrEnum):
    """The teaching-language sections a school operates."""

    ARABIC = "arabic"
    LANGUAGES = "languages"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class AcademicTrack(_Named):
    """One academic structure inside a school: Arabic or Languages."""

    code: str
    school_code: SchoolCode | str
    language_type: SchoolLanguage | str
    name_en: str
    name_ar: str
    display_order: int
    is_active: bool = True

    def __post_init__(self) -> None:
        code = str(self.code or "").strip().upper()
        if not code:
            raise ValidationError("track code is required", field="track_code")
        object.__setattr__(self, "code", code)
        _coerce("school_code", SchoolCode, self)
        _coerce_names(self)
        if not isinstance(self.language_type, SchoolLanguage):
            try:
                object.__setattr__(self, "language_type", SchoolLanguage(str(self.language_type)))
            except ValueError:
                raise ValidationError(
                    "track language must be arabic or languages", field="language_type"
                ) from None
        if self.language_type is SchoolLanguage.BOTH:
            raise ValidationError("a track cannot contain both languages", field="language_type")


class WorkingDay(StrEnum):
    """Stable weekday identifiers saved now and consumed by the future timetable."""

    SATURDAY = "saturday"
    SUNDAY = "sunday"
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"


GRADE_LIMITS: dict[Stage, int] = {
    Stage.GARDEN: 3,
    Stage.PRIMARY: 6,
    Stage.PREPARATORY: 3,
    # The current Arabic and Languages blueprints each define three secondary grades.
    Stage.SECONDARY: 3,
}


_STAGE_ORDER = {
    Stage.GARDEN: 0,
    Stage.PRIMARY: 1,
    Stage.PREPARATORY: 2,
    Stage.SECONDARY: 3,
    # Last, not first: an ungrouped rung is one nobody has classified yet, and it belongs
    # at the bottom of the list rather than above the kindergarten.
    Stage.UNSPECIFIED: 4,
}


@dataclass(frozen=True, slots=True)
class School(_Named):
    """One school. The service held exactly one for its whole life and now holds several.

    What this type is really for is the boundary. Every year, rung, class, mark and
    placement below it belongs to one school, and the failure it exists to prevent is the
    quiet one: two branches each running a 3A, and a register showing a child at a school
    she has never attended. Nothing here crosses that boundary except a child herself — a
    Student is a person rather than a school's property, so a child moving between branches
    is a transfer and not a second record of the same girl.

    `is_active` closes a branch without deleting it, for the same reason a retired subject
    is deactivated rather than dropped: the registers and marks of the years it ran are
    still true.

    The rest is what the school states about how it runs, once, when it is created: which
    teaching-language sections it operates, how many grades it teaches in each stage — zero
    meaning it does not run that stage at all — how many terms its year is divided into, and
    which days of the week it opens. Those are answers only the school has, and holding them
    here is what lets the rest of the service stop assuming a four-stage, two-term, Sunday-to-
    Thursday school: a rung cannot be added on a stage the school does not teach, and the
    timetable that comes later reads its week from `working_days` rather than from a constant.
    """

    code: SchoolCode | str
    name_en: str
    name_ar: str
    is_active: bool = True
    language_type: SchoolLanguage | str = SchoolLanguage.BOTH
    kg_grade_count: int = 3
    primary_grade_count: int = 6
    preparatory_grade_count: int = 3
    secondary_grade_count: int = 3
    term_count: int = 2
    working_days: tuple[WorkingDay | str, ...] = (
        WorkingDay.SUNDAY,
        WorkingDay.MONDAY,
        WorkingDay.TUESDAY,
        WorkingDay.WEDNESDAY,
        WorkingDay.THURSDAY,
    )

    def __post_init__(self) -> None:
        _coerce("code", SchoolCode, self)
        _coerce_names(self)
        if not isinstance(self.language_type, SchoolLanguage):
            try:
                object.__setattr__(self, "language_type", SchoolLanguage(str(self.language_type)))
            except ValueError:
                raise ValidationError(
                    "language_type must be arabic, languages or both", field="language_type"
                ) from None
        counts = {stage: self.grade_count_for(stage) for stage in GRADE_LIMITS}
        for stage, count in counts.items():
            # `bool` is an `int` in Python and `True` would pass the range check as 1, which
            # would read back as a school teaching one year of primary.
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValidationError(
                    f"{stage.value} grade count must be a whole number",
                    field=f"{stage.value}_grade_count",
                )
            if not 0 <= count <= GRADE_LIMITS[stage]:
                raise ValidationError(
                    f"{stage.value} grade count must be between 0 and {GRADE_LIMITS[stage]}",
                    field=f"{stage.value}_grade_count",
                )
        if not any(counts.values()):
            raise ValidationError("at least one educational level is required", field="levels")
        if self.term_count not in (1, 2, 3):
            raise ValidationError("term_count must be 1, 2 or 3", field="term_count")
        # Deduplicated, and in the order the caller gave them: a school that lists Sunday
        # twice means Sunday once, and nothing here reorders a week the school stated.
        days: list[WorkingDay] = []
        for raw in self.working_days:
            try:
                day = raw if isinstance(raw, WorkingDay) else WorkingDay(str(raw))
            except ValueError:
                raise ValidationError(
                    f"unknown working day {raw!r}", field="working_days"
                ) from None
            if day not in days:
                days.append(day)
        if not days:
            raise ValidationError("at least one working day is required", field="working_days")
        object.__setattr__(self, "working_days", tuple(days))

    def grade_count_for(self, stage: Stage) -> int:
        """How many grades this school teaches in `stage`; 0 when it does not run it."""
        return {
            Stage.GARDEN: self.kg_grade_count,
            Stage.PRIMARY: self.primary_grade_count,
            Stage.PREPARATORY: self.preparatory_grade_count,
            Stage.SECONDARY: self.secondary_grade_count,
        }.get(stage, 0)

    def allows_stage(self, stage: Stage) -> bool:
        """Whether a rung may sit on `stage`.

        `UNSPECIFIED` is always allowed: it is the rung nobody has classified yet, not a
        claim about a stage, and refusing it would make an unsorted ladder unimportable.
        """
        return stage is Stage.UNSPECIFIED or self.grade_count_for(stage) > 0


# What a school states about how it runs, as opposed to its code and its two labels. Named
# once, here, because the write path has to tell the two apart: a school is stored through an
# upsert, and an upsert that mentions only the labels must not restate the configuration.
SCHOOL_CONFIGURATION: frozenset[str] = frozenset(
    {
        "language_type",
        "kg_grade_count",
        "primary_grade_count",
        "preparatory_grade_count",
        "secondary_grade_count",
        "term_count",
        "working_days",
    }
)


@dataclass(frozen=True, slots=True)
class AcademicYear(_Named):
    """One school year, e.g. `2025-2026`, and the dates it spans.

    `is_current` is a stated flag rather than a comparison against today, because the
    domain never reads the clock and because the changeover is an administrative act:
    a registrar builds next year's classes in July while this year is still the one
    report cards are issued against.
    """

    code: AcademicYearCode | str
    school_code: SchoolCode | str
    name_en: str
    name_ar: str
    starts_on: date
    ends_on: date
    is_current: bool = False

    def __post_init__(self) -> None:
        _coerce("code", AcademicYearCode, self)
        _coerce("school_code", SchoolCode, self)
        _coerce_names(self)
        _check_range(self.starts_on, self.ends_on, "ends_on")

    def contains(self, day: date) -> bool:
        """Both ends inclusive — the last day of the year is a day of the year."""
        return self.starts_on <= day <= self.ends_on


@dataclass(frozen=True, slots=True)
class YearLevel(_Named):
    """A rung of the ladder — "Year 3", "Grade 10" — and **not** scoped to a year.

    This is the decision most schemas get wrong. "Year 3" is the same rung in 2025-2026
    as in 2030-2031: different children stand on it, but the rung itself does not change
    identity. Scoping it to an academic year would create one `YearLevel` row per year
    per rung, so a twelve-rung school accumulates twelve new rows every August, and —
    far worse — "how do this year's Year 3 results compare with last year's" stops being
    a filter on one year level and becomes a join between two *different* Year 3s that
    happen to share a label. Every cross-year question then depends on a text match
    against a name a registrar is free to retype. Cohorts live in `ClassSection` and in
    enrolment, which are the things that genuinely differ year to year.

    `display_order` exists because sorting on `code` is wrong in exactly the place it
    matters: lexicographically `"Y10"` sorts before `"Y9"`, so a school with more than
    nine rungs prints its year list out of order, and the person reading it assumes the
    data is scrambled rather than the sort. Storing the order explicitly also lets a
    school whose ladder is `KG1, KG2, Y1…` place its kindergarten rungs first without
    inventing codes that happen to sort correctly.
    """

    code: YearCode | str
    school_code: SchoolCode | str
    name_en: str
    name_ar: str
    display_order: int
    stage: Stage = Stage.UNSPECIFIED
    track_code: str | None = None

    def __post_init__(self) -> None:
        _coerce("code", YearCode, self)
        _coerce("school_code", SchoolCode, self)
        _coerce_names(self)
        # The raw string a spreadsheet or a form sends is accepted, so a caller never has
        # to import the enum to state a stage. An unknown value is refused rather than
        # quietly falling back to UNSPECIFIED: "secondry" would otherwise create a rung
        # that is silently missing from the secondary group on every screen that groups.
        if not isinstance(self.stage, Stage):
            try:
                object.__setattr__(self, "stage", Stage(str(self.stage).strip().lower()))
            except ValueError:
                raise ValidationError(
                    f"{self.stage!r} is not a stage; expected one of "
                    + ", ".join(member.value for member in Stage),
                    field="stage",
                ) from None
        if self.track_code is not None:
            track = str(self.track_code).strip().upper()
            object.__setattr__(self, "track_code", track or None)
        # A bool is an int, and `display_order=True` would silently mean "position 1".
        if isinstance(self.display_order, bool) or not isinstance(self.display_order, int):
            raise ValidationError(
                "display order must be a whole number", field="display_order"
            )

    @property
    def sort_key(self) -> tuple[int, int, str]:
        """Stage, then stated order, then code so the sort is stable across processes.

        Stage leads because that is how the list is read: a registrar scanning fourteen
        rungs finds the garden block, then primary. Within a stage `display_order` still
        decides, which is the reason it was stored explicitly in the first place.
        """
        return (self.stage.order, self.display_order, str(self.code))


@dataclass(frozen=True, slots=True)
class ClassSection(_Named):
    """One room of children, e.g. `3A`, within one academic year.

    Scoped to an academic year and to nothing narrower. The `3A` of 2025-2026 and the
    `3A` of 2026-2027 are different groups of children who share a label, so the year is
    part of this thing's identity — without it, September's intake inherits last June's
    enrolments and grades.

    It is emphatically **not** scoped to a term. A class outlives a term: the same 3A
    runs from September to June across three terms, and giving each term its own class
    row would triple the structure, force a re-import of every roster each term, and
    make "which class is she in" a question with three answers. What genuinely varies
    within a year is *membership*, and that is why placement is a time-bounded enrolment
    (invariant 2) rather than a column here: a child moving 3A->3B in March is one
    membership ending and another beginning, leaving her Term 1 grades resolving to 3A
    forever. Were the class term-scoped, that move would instead be modelled by editing
    a class, and the Term 1 report would silently reprint under the new one.
    """

    code: ClassCode | str
    academic_year_code: AcademicYearCode | str
    year_level_code: YearCode | str
    name_en: str
    name_ar: str
    # `None` means "no stated limit", never 0 — same distinction as invariant 1. A
    # capacity of 0 is a class that admits nobody, which is a thing a registrar can
    # legitimately configure while closing a section.
    capacity: int | None = None

    def __post_init__(self) -> None:
        _coerce("code", ClassCode, self)
        _coerce("academic_year_code", AcademicYearCode, self)
        _coerce("year_level_code", YearCode, self)
        _coerce_names(self)
        if self.capacity is not None and (
            isinstance(self.capacity, bool) or not isinstance(self.capacity, int)
        ):
            raise ValidationError("capacity must be a whole number", field="capacity")
        if self.capacity is not None and self.capacity < 0:
            raise ValidationError("capacity may not be negative", field="capacity")

    @property
    def identity(self) -> tuple[str, str]:
        """The pair that is actually unique. Use this as a dict key, never `code` alone."""
        return (str(self.academic_year_code), str(self.code))


@dataclass(frozen=True, slots=True)
class Term(_Named):
    """A reporting period inside one academic year, e.g. `2026-T1`.

    Terms are what make a grade answerable: a mark belongs to a term, and a term belongs
    to a year. `is_closed` is stated rather than derived from `ends_on` because a school
    keeps entering late marks for a week after a term ends, and a clock-derived close
    would reject them; `TermClosed` is raised only once a human has said so.

    **The dates are optional, and that is a deliberate widening.** A school declares how
    many terms it runs when it is created, and the year's term sections are built from that
    number — in June, months before anyone has decided when the second term starts. Forcing
    a date at that moment would mean either blocking the setup until the calendar is
    settled, or inviting a registrar to type a placeholder; and a placeholder date is worse
    than no date, because nothing downstream can tell it from a real one. `None` means "the
    school has not said yet", which is a fact the service can carry honestly.

    What that costs, stated plainly because it is the reason the dates existed: a term with
    no window cannot answer "which class was this child in during it" on its own. Invariant
    2 still holds — the answer is resolved against the *year's* window instead, which is the
    same rule one level up. See `resolution_window` and its caller in `queries.py`.

    `sequence` is what orders terms, not the dates, so an undated term still sorts into its
    right place on every screen.
    """

    code: TermCode | str
    academic_year_code: AcademicYearCode | str
    name_en: str
    name_ar: str
    starts_on: date | None = None
    ends_on: date | None = None
    sequence: int = 1
    is_closed: bool = False

    # A stated range is still the input to invariant 2: a placement covers a term when the
    # membership overlaps it, which is why a stated range must never be inverted. An absent
    # one is checked no further — there is nothing to invert.
    def __post_init__(self) -> None:
        _coerce("code", TermCode, self)
        _coerce("academic_year_code", AcademicYearCode, self)
        _coerce_names(self)
        if self.starts_on is not None and self.ends_on is not None:
            _check_range(self.starts_on, self.ends_on, "ends_on")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValidationError("sequence must be a whole number", field="sequence")

    @property
    def is_dated(self) -> bool:
        """Whether the school has stated both ends of this term.

        One end alone is not a window. A term with a start and no end is a school that has
        said when teaching begins and not when reporting closes, and every question asked
        of a term is asked of the whole period — so a half-stated range answers nothing
        that a wholly absent one does not.
        """
        return self.starts_on is not None and self.ends_on is not None

    def resolution_window(self, fallback: "AcademicYear") -> tuple[date, date]:
        """The days this term is resolved against: its own, or the year's when it has none.

        The fallback is the honest one rather than a convenience. A term inside a year is,
        at minimum, inside that year — so a school that has not yet dated its terms is
        asking the same question about a wider period, not a different question. What it
        must never become is a guess at a narrower period: splitting the year into equal
        thirds would produce dates nobody stated and file a child's marks under the class
        she happened to be in on a boundary the service invented.
        """
        if self.is_dated:
            return self.starts_on, self.ends_on
        return fallback.starts_on, fallback.ends_on

    def contains(self, day: date) -> bool:
        """Both ends inclusive; the last day of term is a teaching day.

        An undated term contains no day at all. That is a refusal, not an oversight: the
        school has said nothing about when this term runs, and answering `True` for every
        day — or for none by accident — would put a decision in this method that belongs to
        whoever knows the year. Callers that need a window use `resolution_window`.
        """
        return self.is_dated and self.starts_on <= day <= self.ends_on

    def overlaps(self, starts_on: date, ends_on: date | None) -> bool:
        """Whether an open-ended period touches this term — `None` end means "still on".

        `False` for an undated term, for the reason `contains` returns `False`.
        """
        if not self.is_dated:
            return False
        return starts_on <= self.ends_on and (ends_on is None or ends_on >= self.starts_on)

    @property
    def sort_key(self) -> tuple[int, str]:
        """Chronological order without parsing the code, which schools format freely."""
        return (self.sequence, str(self.code))


@dataclass(frozen=True, slots=True)
class Subject(_Named):
    """A taught subject inside one academic year, e.g. `MATH` in `2025-2026`.

    Identity is the pair, not the code. A school sets its own catalogue each year, so
    `MATH` in two years is two subjects — and the cost of that, which the caller has to
    know about, is that a mark on one is not comparable to a mark on the other. Anything
    wanting a child's mathematics across three years has to match on the code string and
    accept that the school may have meant something different by it each time.

    `is_active` retires a subject instead of deleting it: grades already stated against
    a dropped subject must keep resolving, and a code that vanishes turns last year's
    report card into a row of marks with no heading.
    """

    code: SubjectCode | str
    academic_year_code: AcademicYearCode | str
    name_en: str
    name_ar: str
    display_order: int = 0
    is_active: bool = True

    DEFAULT_ORDER: ClassVar[int] = 0

    def __post_init__(self) -> None:
        _coerce("code", SubjectCode, self)
        _coerce("academic_year_code", AcademicYearCode, self)
        _coerce_names(self)
        if isinstance(self.display_order, bool) or not isinstance(self.display_order, int):
            raise ValidationError(
                "display order must be a whole number", field="display_order"
            )

    @property
    def sort_key(self) -> tuple[int, str]:
        """Order on a report card: stated order first, then code for a stable tie-break."""
        return (self.display_order, str(self.code))
