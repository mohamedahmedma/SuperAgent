"""The child, and the class she was in — two records on purpose, never one.

`Student` is an identity and a pair of names. `ClassEnrolment` is where she sat and
between which dates. Fusing them into one row with a `class_code` column is the single
most common shape for this data and the reason so many school systems cannot answer a
question about last term; decision 2 and the `ClassEnrolment` docstring below spell out
what that costs.

Framework-free, like the rest of `sis/domain/`: stdlib and the sibling value objects
only. No sqlalchemy, no pydantic, no `sis.config`. **Nothing here reads the clock** —
every date-sensitive question takes the date as an argument, so a service's behaviour
in a unit test is a function of its inputs rather than of the day the suite happens to
run. "Is she currently in 3A" is `enrolment.covers(today)` with a `today` the caller
supplies.
"""
from dataclasses import dataclass
from datetime import date

from sis.domain.errors import InvalidDateRange, ValidationError
from sis.domain.value_objects import AcademicYearCode, ClassCode, StudentNumber


def _coerce(entity: object, field: str, kind: type) -> None:
    """Replace a raw cell with its value object, in place, during `__post_init__`.

    The same contract `sis.domain.structure` gives its entities, and it exists for the
    same reason: annotating `student_number: StudentNumber` documents an intention that
    nothing enforces, so a caller passing the string `"0071"` builds a `Student` that
    compares unequal to every `StudentNumber("0071")` in the service and whose
    `.value` access raises `AttributeError` several layers away. Normalising here means a
    code is a code by the time the object exists, whichever agent constructed it.
    """
    raw = getattr(entity, field)
    if not isinstance(raw, kind):
        object.__setattr__(entity, field, kind(raw))


def _clean_name(raw: object, *, field: str) -> str:
    """Names are labels, so anything is allowed in them except a lie about emptiness."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValidationError(f"{field} must be text", field=field)
    return raw.strip()


@dataclass(frozen=True, slots=True)
class Student:
    """A child: her school number, her names, and whether she is still on the roll.

    Thin by design, and the omissions are the design. There is no `class_code` here
    (that is `ClassEnrolment`), no year group, no grades, and no guardian *column* — a
    child's guardians are `Guardian` and `StudentGuardian` in `sis.domain.guardians`, a
    separate many-to-many, for the same reason class placement is separate: one child
    ordinarily has more than one guardian and one guardian more than one child, so a
    column here would hold the father and silently lose the mother. Every one of those
    fields would be a fact that is true on one date and false on another, stored as
    though it were true forever.

    `is_active` is a flag rather than a deletion because a child who leaves in March
    still has a Term 1 report card, and rows that are deleted to "clean up the roster"
    take that report card with them. Inactive means "do not offer her in the roster
    pickers"; it never means her history stops resolving.
    """

    student_number: StudentNumber | str
    full_name_ar: str
    full_name_en: str
    is_active: bool = True

    #: Stated, never derived. Age is a function of a date and today, and storing the age
    #: instead would be a number that is wrong from the day after it is written.
    date_of_birth: date | None = None

    #: The child's own contact details, which are **not** her guardian's. A school holds
    #: both, they differ, and merging them is how a message meant for a parent goes to a
    #: nine-year-old's tablet. Guardian numbers live in `sis.domain.guardians`, where the
    #: permission to hear about her marks lives with them.
    contact_phone: str = ""
    contact_email: str = ""
    address: str = ""

    def __post_init__(self) -> None:
        _coerce(self, "student_number", StudentNumber)
        name_ar = _clean_name(self.full_name_ar, field="full_name_ar")
        name_en = _clean_name(self.full_name_en, field="full_name_en")
        # Both names are wanted (decision 7) but only one is *required*: a roster
        # exported from an Arabic-first SIS has no English column, and refusing the file
        # would push the registrar into typing transliterations by hand — which is how a
        # school ends up with two spellings of one child and no way to tell they match.
        if not name_ar and not name_en:
            raise ValidationError(
                "a student needs a name in Arabic or English", field="full_name_ar"
            )
        object.__setattr__(self, "full_name_ar", name_ar)
        object.__setattr__(self, "full_name_en", name_en)

        for field in ("contact_phone", "contact_email", "address"):
            object.__setattr__(self, field, str(getattr(self, field) or "").strip())

        # A birth date in the future is a typo — a year mistyped, or a form filled in with
        # the current year by default — and it is worth refusing at the door because every
        # age computed from it afterwards is negative and nothing downstream checks.
        if self.date_of_birth is not None and self.date_of_birth > date.today():
            raise ValidationError(
                f"date of birth {self.date_of_birth.isoformat()} is in the future",
                field="date_of_birth",
            )

    def age_on(self, day: date) -> int | None:
        """Her age in whole years on `day`, or `None` when no birth date is on file.

        Takes the day rather than reading the clock, like everything else in this module:
        "how old is she" has a different answer on the day before her birthday and the day
        after, and a domain that reads `date.today()` cannot be tested for either.

        `None` rather than 0 for a missing birth date, and the distinction is the same one
        this service makes about a blank mark: nought years old is a statement about a
        newborn, and "nobody has recorded her birthday" is not a statement about her age
        at all.
        """
        if self.date_of_birth is None:
            return None
        birthday = self.date_of_birth
        years = day.year - birthday.year
        # Subtract a year when the birthday has not come round yet. The tuple comparison
        # handles 29 February without a special case: in a non-leap year the birthday is
        # read as falling on the 29th, which has not occurred, so she turns a year older on
        # the 1st of March.
        if (day.month, day.day) < (birthday.month, birthday.day):
            years -= 1
        return years


@dataclass(frozen=True, slots=True)
class ClassEnrolment:
    """A placement of one student in one class section, bounded by dates.

    **This is decision 2, and it is the reason this type exists at all.** Class
    placement is a membership over time, not a column on the student. A child who moves
    from 3A to 3B in March was genuinely in 3A for Term 1: her Term 1 marks were earned
    there, her Term 1 report card names it, and a parent asking about Term 1 must be
    told 3A. A `students.class_code` column cannot say that. Overwriting it on transfer
    rewrites history — the Term 1 report re-prints with 3B on it, which is a class the
    child had never entered when those marks were given — and *not* overwriting it makes
    the current class wrong instead. Both readings are wrong; there is no third option
    while the fact lives in one column.

    The failure that follows is quieter than a wrong class name and worse. Ask a
    single-column system for "3A, Term 1" after the transfer and it finds no such child,
    so the term reads as *nothing recorded* — a completed, marked, published term
    rendered as missing data. Nobody sees an error. The registrar sees an empty grade
    sheet for a term she remembers filling in, and re-enters marks that were never lost.

    Storing the interval instead makes the question a lookup: resolve the class for a
    term by finding the enrolment covering that term's dates. Both rows survive, both
    are true, and neither has to be edited when the next transfer happens.

    `ends_on` of `None` means the placement is open — she is in this class now and no
    end has been decided. That is the same "unknown" as `None` in a grade (decision 1):
    it is not a date far in the future, and code must not invent one, because a sentinel
    like 9999-12-31 sorts and compares like a real answer and will eventually be printed
    to somebody as one.
    """

    student_number: StudentNumber | str
    # The class is identified by (year, code) rather than code alone: `ClassCode` is
    # unique *within* an academic year, so a bare "3A" names a different room of
    # children every September and cannot resolve a placement on its own.
    academic_year_code: AcademicYearCode | str
    class_code: ClassCode | str
    starts_on: date
    ends_on: date | None = None

    def __post_init__(self) -> None:
        _coerce(self, "student_number", StudentNumber)
        _coerce(self, "academic_year_code", AcademicYearCode)
        _coerce(self, "class_code", ClassCode)
        if not isinstance(self.starts_on, date):
            raise ValidationError("starts_on must be a date", field="starts_on")
        if self.ends_on is not None and not isinstance(self.ends_on, date):
            raise ValidationError("ends_on must be a date", field="ends_on")
        if self.ends_on is not None and self.ends_on < self.starts_on:
            raise InvalidDateRange(
                f"enrolment ends before it starts ({self.ends_on} < {self.starts_on})",
                field="ends_on",
            )

    @property
    def is_open(self) -> bool:
        """True while the placement has no end — her current class."""
        return self.ends_on is None

    def covers(self, on: date) -> bool:
        """Was this placement active on day `on`? Both ends inclusive.

        `ends_on` is the child's **last day** in the class, not the day after. That
        matches what a human states — "she left 3A on 12 March" — so the registrar's
        date is stored verbatim and no layer has to add or subtract a day to talk to
        another. An exclusive end would mean the UI writes a date nobody typed, and the
        resulting off-by-one is invisible until it lands on a term boundary: the child
        belongs to two classes on one day, or to none, on exactly the day a report is
        cut.

        Takes the date as an argument rather than calling `date.today()`, so services
        built on this are testable at any point in a school year without freezing time.
        """
        if on < self.starts_on:
            return False
        return self.ends_on is None or on <= self.ends_on

    def overlaps_period(self, start: date, end: date) -> bool:
        """Did this placement cover any day of `start`..`end` (inclusive)?

        This is how a term resolves to a class: pass the term's first and last day. A
        transfer mid-term makes two enrolments answer True, which is correct and is the
        caller's decision to resolve (usually "the one covering the term's end"), not a
        fact this predicate may quietly discard.
        """
        if end < start:
            raise InvalidDateRange(
                f"period ends before it starts ({end} < {start})", field="ends_on"
            )
        if end < self.starts_on:
            return False
        return self.ends_on is None or start <= self.ends_on

    def conflicts_with(self, other: "ClassEnrolment") -> bool:
        """Would these two placements put one child in two classes on the same day?

        The test behind `OverlappingEnrolment`. Two enrolments for *different* students
        never conflict. Note that this is a comparison of values, so a placement does
        conflict with a copy of itself: when re-checking a stored enrolment the caller
        must exclude that row by its own identity, or an untouched edit reports a clash
        with the record it is editing.
        """
        if other.student_number != self.student_number:
            return False
        if other.ends_on is None:
            return self.ends_on is None or other.starts_on <= self.ends_on
        return self.overlaps_period(other.starts_on, other.ends_on)
