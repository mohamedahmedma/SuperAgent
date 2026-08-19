"""Values that are validated at construction or not at all.

Why these are types rather than `str` and `float`: every one of them is a value that
arrives from a spreadsheet cell, and a spreadsheet cell is the least trustworthy input
in this service. A `str` parameter named `subject_code` accepts `"  math "`, `""`, and
the class *name* by mistake; a `SubjectCode` accepts none of those, and it does so at
the boundary rather than three layers later when a foreign key fails and the operator
is shown "integrity error".

The second reason is that they are not interchangeable with each other. `SubjectCode`
and `TermCode` are both short uppercase strings, so a call like
`get_grade(student, term_code, subject_code)` with the last two arguments swapped is
invisible to a type checker and to every test that happens to use the same value for
both. Distinct types make that swap a type error.

Normalisation happens once, here. Codes are uppercased and cleaned of the invisible
characters Excel and Arabic keyboards leave behind, because the alternative is that
`"3A"` and `"3a"` become two classes and a registrar spends an afternoon working out
why half a year group has no grades.

**The domain never reads the clock and never touches a database.** Nothing in this
module imports sqlalchemy, fastapi or pydantic, and it must stay that way: these types
are what the unit tests of `application/services/` are written against.
"""
import math
import re
from dataclasses import dataclass
from typing import ClassVar, Final

from sis.domain.errors import (
    InvalidCode,
    InvalidPercentage,
    InvalidPhone,
    InvalidStudentNumber,
    ValidationError,
)

# Characters that carry no identity but survive a copy-paste out of Excel or an
# Arabic-first word processor. The right-to-left mark in particular is invisible in
# every UI a human will look at, so a subject code that visibly reads "MATH" compares
# unequal to "MATH" and the import rejects a subject that is plainly on screen.
# Non-breaking space becomes an ordinary space rather than vanishing: deleting it would
# silently turn "3 A" into the *valid* code "3A", inventing an identity nobody typed.
# Written as codepoints on purpose: a literal containing these characters is a literal
# no reviewer can see, and one of them would eventually be deleted by an editor that
# strips "stray" bytes.
_INVISIBLE: Final[dict[int, str | None]] = {
    0x200B: None,  # zero width space
    0x200C: None,  # zero width non-joiner
    0x200D: None,  # zero width joiner
    0x200E: None,  # left-to-right mark
    0x200F: None,  # right-to-left mark
    0x202A: None,  # left-to-right embedding
    0x202B: None,  # right-to-left embedding
    0x202C: None,  # pop directional formatting
    0x2066: None,  # left-to-right isolate
    0x2067: None,  # right-to-left isolate
    0x2068: None,  # first strong isolate
    0x2069: None,  # pop directional isolate
    0xFEFF: None,  # byte order mark, left behind by a UTF-8-BOM CSV
    0x00A0: " ",  # non-breaking space -> ordinary space, see comment above
}

# Codes travel through URLs, CSV cells and Moodle idnumbers. This set is the
# intersection of what survives all three unescaped. No spaces, no slashes, no
# non-ASCII: Arabic belongs in `name_ar`, which is a label and may hold anything.
_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9][A-Z0-9._-]*$")

# Beyond this a float has lost integer precision, so coercing it back to a digit string
# would produce a *different* identifier than the one in the file.
_EXACT_FLOAT_LIMIT: Final[int] = 2**53


@dataclass(frozen=True, slots=True)
class _Code:
    """Shared machinery for the immutable codes of decision 7. Never used directly.

    Subclasses exist as distinct types purely so that the compiler distinguishes them;
    all of the behaviour lives here. Each subclass sets `MAX_LENGTH` — which is also
    what the infrastructure layer should size its columns from, so the database and the
    validator cannot drift apart and produce a code that passes validation and is then
    truncated on write.
    """

    value: str

    MAX_LENGTH: ClassVar[int] = 32
    LABEL: ClassVar[str] = "code"
    FIELD: ClassVar[str] = "code"
    ERROR: ClassVar[type[ValidationError]] = InvalidCode

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self._normalise(self.value))

    @classmethod
    def _normalise(cls, raw: object) -> str:
        """Coerce, clean and validate. The only path by which a code comes into being."""
        text = cls._as_text(raw)
        cleaned = text.translate(_INVISIBLE).strip().upper()
        if not cleaned:
            raise cls.ERROR(f"{cls.LABEL} is required", field=cls.FIELD)
        if len(cleaned) > cls.MAX_LENGTH:
            raise cls.ERROR(
                f"{cls.LABEL} may not exceed {cls.MAX_LENGTH} characters",
                field=cls.FIELD,
            )
        if not _CODE_PATTERN.match(cleaned):
            raise cls.ERROR(
                f"{cls.LABEL} must start with a letter or digit and may then contain "
                f"only letters, digits, '.', '-' and '_' (got {cleaned!r})",
                field=cls.FIELD,
            )
        return cleaned

    @classmethod
    def _as_text(cls, raw: object) -> str:
        """Accept the cell types a spreadsheet actually produces, and nothing else.

        A student number column of digits is typed as a *number* by Excel and by
        openpyxl, so rejecting non-strings here would fail every numeric roster ever
        exported. Integral floats are accepted for the same reason and converted through
        `int` so that `12345.0` does not become the string `"12345.0"`, which matches no
        student on file.

        `bool` is rejected explicitly because it is a subclass of `int`, and `True`
        would otherwise quietly become the code `"1"`.
        """
        if isinstance(raw, str):
            return raw
        if isinstance(raw, bool):
            raise cls.ERROR(f"{cls.LABEL} must be text, not a boolean", field=cls.FIELD)
        if isinstance(raw, int):
            return str(raw)
        if isinstance(raw, float):
            if math.isfinite(raw) and raw.is_integer() and abs(raw) < _EXACT_FLOAT_LIMIT:
                return str(int(raw))
            raise cls.ERROR(
                f"{cls.LABEL} is not a whole number; format the column as text",
                field=cls.FIELD,
            )
        raise cls.ERROR(f"{cls.LABEL} is required", field=cls.FIELD)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class AcademicYearCode(_Code):
    """Identifies one school year, e.g. `2025-2026`.

    Exists so the year is a row that everything else points at rather than a string
    repeated on each class and term. The failure that avoids is the one every school
    database eventually has: `2025-2026`, `2025-26` and `2025/2026` all present, each
    holding a third of the year's classes, and a "list this year's classes" query that
    quietly returns a third of them.
    """

    MAX_LENGTH: ClassVar[int] = 16
    LABEL: ClassVar[str] = "academic year code"
    FIELD: ClassVar[str] = "academic_year_code"


@dataclass(frozen=True, slots=True)
class YearCode(_Code):
    """Identifies a rung of the ladder — `Y3`, `G10` — not a year group in one year.

    Short by design. It is the prefix operators type into class codes and import
    headers all day, and a long one gets abbreviated inconsistently in the very files
    this service has to match against.
    """

    MAX_LENGTH: ClassVar[int] = 16
    LABEL: ClassVar[str] = "year level code"
    FIELD: ClassVar[str] = "year_level_code"


@dataclass(frozen=True, slots=True)
class ClassCode(_Code):
    """Identifies one class section within one academic year, e.g. `3A`.

    Unique per academic year, not globally: `3A` exists every year and is a different
    room of children each time. Making it globally unique would force codes like
    `2025-2026-3A` into every roster file a school produces, and schools produce files
    that say `3A`.
    """

    MAX_LENGTH: ClassVar[int] = 32
    LABEL: ClassVar[str] = "class code"
    FIELD: ClassVar[str] = "class_code"


@dataclass(frozen=True, slots=True)
class SubjectCode(_Code):
    """Identifies a subject, e.g. `MATH`, `ARB`.

    Global rather than per-year-level, for the same reason `YearLevel` is not scoped to
    a year: one `MATH` means a child's maths marks are comparable across the six years
    she is at the school, and several `MATH`s mean that comparison is a join nobody
    writes correctly.
    """

    MAX_LENGTH: ClassVar[int] = 32
    LABEL: ClassVar[str] = "subject code"
    FIELD: ClassVar[str] = "subject_code"


@dataclass(frozen=True, slots=True)
class TermCode(_Code):
    """Identifies a term, e.g. `2026-T1`.

    Deliberately carries the year in the conventional form used by `records/` term
    codes, because a bare `T1` is ambiguous the moment the school has two years of
    history and every grade query has to be qualified by an academic year the caller
    may not have.
    """

    MAX_LENGTH: ClassVar[int] = 32
    LABEL: ClassVar[str] = "term code"
    FIELD: ClassVar[str] = "term_code"


@dataclass(frozen=True, slots=True)
class StudentNumber(_Code):
    """The school's own identifier for a child — what a parent reads off a letter.

    This is the join key every importer matches on — and the one the `records/` adapter
    reads across the service boundary — which is why decision 7 makes it immutable:
    re-issuing a student number detaches every grade, every enrolment and every guardian
    link from the child they belong to, and none of those rows report an error when it
    happens.

    Kept as text, never as an integer, because leading zeros are significant in most
    schools' numbering and `int("0071")` is `71` — a different child, or no child.
    Arabic-Indic digits are rejected rather than transliterated: two roster rows that
    differ only by digit script must be corrected by a human, not silently merged into
    one student.
    """

    MAX_LENGTH: ClassVar[int] = 64
    LABEL: ClassVar[str] = "student number"
    FIELD: ClassVar[str] = "student_number"
    ERROR: ClassVar[type[ValidationError]] = InvalidStudentNumber


# Arabic-Indic and extended Arabic-Indic digits, folded to ASCII. `StudentNumber`
# deliberately *rejects* these; `Phone` deliberately converts them, and the difference is
# not an inconsistency. A student number is an opaque identifier, so two rows differing
# only by digit script are two spellings a human must reconcile — merging them invents an
# identity. A phone number is a quantity that has to be dialled, and ٠١٠٠١٢٣٤٥٦٧ is the
# same number as 01001234567 by arithmetic, not by assumption.
_ARABIC_DIGITS: Final[dict[int, str]] = {
    **{0x0660 + offset: str(offset) for offset in range(10)},
    **{0x06F0 + offset: str(offset) for offset in range(10)},
}

# What a human puts *between* the digits of a phone number. Removed rather than rejected,
# because "0100 123 4567" and "0100-123-4567" are one number written by two registrars.
_PHONE_SEPARATORS: Final[frozenset[str]] = frozenset(" -.()/–—")

# Latin digits only, for `str.strip()` membership tests. See `Phone._validated` for why
# `str.isdigit()` is not good enough.
_ASCII_DIGITS: Final[str] = "0123456789"


@dataclass(frozen=True, slots=True)
class Phone:
    """A guardian's phone number in E.164, e.g. `+201001234567`.

    This is the *identity* of a guardian, not merely a way to contact one. A school runs
    no parent-numbering system for an importer to key on, so the phone is what recognises
    the same mother across three of her children's rows and across next term's re-upload.
    Everything about the normalisation below exists to make that recognition reliable: if
    `+201001234567` and `0100 123 4567` do not collide, she becomes two people, each
    holding half her children.

    **Constructing one directly requires E.164 already.** `Phone("+201001234567")` is
    valid and `Phone("01001234567")` raises, so an entity rebuilt from a stored row can
    never quietly accept a half-normalised number. Turning what a registrar typed into
    E.164 happens in exactly one place — `parse()` — because that conversion needs to know
    a default country, and a value object that guessed one per call site would produce two
    different numbers from one spreadsheet.

    The country code is a *parameter* of `parse()` rather than something read from
    configuration, for the same reason `ClassEnrolment.covers()` takes the date instead of
    calling `date.today()`: the domain must stay a function of its inputs.
    """

    value: str

    MIN_DIGITS: ClassVar[int] = 8
    MAX_DIGITS: ClassVar[int] = 15  # E.164's own ceiling
    MAX_LENGTH: ClassVar[int] = 16  # '+' plus MAX_DIGITS; the column is sized from this
    LABEL: ClassVar[str] = "phone number"
    FIELD: ClassVar[str] = "phone"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self._validated(self.value))

    @classmethod
    def _validated(cls, raw: object) -> str:
        """Accept an already-normalised E.164 string, and nothing else."""
        if not isinstance(raw, str):
            raise InvalidPhone(f"{cls.LABEL} must be text", field=cls.FIELD)
        text = raw.strip()
        # `str.isdigit()` is deliberately not used: it answers True for ٠١٢, so it would
        # accept a number whose digits were never folded to ASCII — one that compares
        # unequal to the same number typed on a Latin keyboard, which is the exact
        # duplicate this type exists to prevent.
        body = text[1:]
        if not text.startswith("+") or not body or body.strip(_ASCII_DIGITS):
            raise InvalidPhone(
                f"{cls.LABEL} must already be in international form, e.g. +201001234567 "
                f"(got {raw!r}); use Phone.parse() to normalise what a registrar typed",
                field=cls.FIELD,
            )
        digits = text[1:]
        if not (cls.MIN_DIGITS <= len(digits) <= cls.MAX_DIGITS):
            raise InvalidPhone(
                f"{cls.LABEL} must have between {cls.MIN_DIGITS} and {cls.MAX_DIGITS} "
                f"digits (got {len(digits)})",
                field=cls.FIELD,
            )
        return "+" + digits

    @classmethod
    def parse(cls, raw: object, *, default_country_code: str) -> "Phone":
        """Build one from a spreadsheet cell. Raises `InvalidPhone` on a blank."""
        parsed = cls.parse_optional(raw, default_country_code=default_country_code)
        if parsed is None:
            raise InvalidPhone(f"a {cls.LABEL} is required here", field=cls.FIELD)
        return parsed

    @classmethod
    def parse_optional(
        cls, raw: object, *, default_country_code: str
    ) -> "Phone | None":
        """Normalise a cell to E.164, or `None` when the cell is genuinely empty.

        `None` for blank rather than an exception, so a caller can tell "this guardian has
        no second number" apart from "this guardian's number is unreadable" — the same
        distinction `Percentage.parse_optional` draws for an ungraded subject.
        """
        text = cls._as_text(raw)
        if text is None:
            return None

        cleaned = text.translate(_INVISIBLE).translate(_ARABIC_DIGITS).strip()
        if not cleaned:
            return None

        international = cleaned.startswith("+")
        body = cleaned[1:] if international else cleaned

        # Letters are refused rather than stripped. "0100 call after 5" reduces to a
        # plausible-looking number if non-digits are simply discarded, and the school then
        # holds a number nobody answers with no record that anything was thrown away.
        for character in body:
            if not character.isdigit() and character not in _PHONE_SEPARATORS:
                raise InvalidPhone(
                    f"{cls.LABEL} may hold only digits and separators "
                    f"(got {raw!r})",
                    field=cls.FIELD,
                )

        digits = "".join(character for character in body if character.isdigit())
        if not digits:
            return None

        return cls("+" + cls._to_international(digits, international, default_country_code))

    @classmethod
    def _to_international(
        cls, digits: str, has_plus: bool, default_country_code: str
    ) -> str:
        """Resolve national spellings against a default country. See the ambiguity note.

        Three of the four cases are unambiguous. A leading `+` and a leading `00` both
        state "this is already international" outright; a leading `0` states "this is
        national" just as plainly. Bare digits with no prefix at all are the ambiguous
        one, and they are common precisely because **Excel types a phone column as a
        number** — `01001234567` reaches this service as `1001234567.0`, the leading zero
        gone, which is the same trap `StudentNumber` documents. Guessing wrong there is
        not cosmetic: it is a whole school of parents who cannot be reached.
        """
        country = "".join(c for c in default_country_code if c.isdigit())
        if has_plus:
            return digits
        if digits.startswith("00"):
            # The international-access prefix. What follows already carries its own
            # country code, so nothing is prepended.
            return digits[2:]
        if digits.startswith("0"):
            return country + digits.lstrip("0")
        # Bare digits. Treat them as already carrying the country code only when doing so
        # leaves a subscriber number long enough to be one; otherwise a local number that
        # merely happens to begin with the country's own digits would be truncated.
        if (
            country
            and digits.startswith(country)
            and len(digits) - len(country) >= cls.MIN_DIGITS - 1
        ):
            return digits
        return country + digits

    @classmethod
    def _as_text(cls, raw: object) -> str | None:
        """Accept the cell types a spreadsheet actually produces. `None` means empty.

        The float branch is the one that matters. A phone column formatted as a number
        arrives as `1001234567.0`, and the naive `str()` of that is `"1001234567.0"` —
        which reaches nobody. `bool` is rejected explicitly because it is a subclass of
        `int` and `True` would otherwise become the number `1`.
        """
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw
        if isinstance(raw, bool):
            raise InvalidPhone(
                f"{cls.LABEL} must be text, not a boolean", field=cls.FIELD
            )
        if isinstance(raw, int):
            return str(raw)
        if isinstance(raw, float):
            if math.isfinite(raw) and raw.is_integer() and abs(raw) < _EXACT_FLOAT_LIMIT:
                return str(int(raw))
            raise InvalidPhone(
                f"{cls.LABEL} is not a whole number; format the column as text",
                field=cls.FIELD,
            )
        raise InvalidPhone(f"{cls.LABEL} must be text", field=cls.FIELD)

    def __str__(self) -> str:
        return self.value


# The only spellings of "no mark was given" this service will accept. Deliberately
# short and explicit: anything unrecognised is rejected as a bad row rather than read
# as "not graded", because guessing turns an unfamiliar mark into a blank on a report
# card, and a blank is what the school shows a parent whose child *does* have a grade.
_ABSENT_TOKENS: Final[frozenset[str]] = frozenset(
    {"", "-", "--", "—", "n/a", "na", "none", "null", "nil", "غ", "غير مرصود", "لم يرصد"}
)


@dataclass(frozen=True, order=True, slots=True)
class Percentage:
    """A stated mark out of 100, inclusive of both ends.

    Ordered, so "who is below 50" is a comparison rather than a reach into `.value`.

    100 is inclusive because full marks are a thing a child earns, and 0 is inclusive
    because a zero is a mark — the *absence* of a mark is `None`, never this type. That
    distinction is decision 6 made concrete: a `Percentage` in hand always means
    somebody stated a figure.

    The figure is stored exactly as given and never rounded here. Decision 6 says a
    grade is a stated number, so a service that rounds is a service that changes what
    the teacher wrote; presentation rounding belongs in the UI, where it is visible.
    """

    value: float

    MINIMUM: ClassVar[float] = 0.0
    MAXIMUM: ClassVar[float] = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self._validated(self.value))

    @classmethod
    def _validated(cls, raw: object) -> float:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise InvalidPercentage(
                f"percentage must be a number (got {type(raw).__name__})",
                field="percentage",
            )
        number = float(raw)
        # NaN and infinity are checked before the range test, because every comparison
        # against NaN answers False — `not (0 <= nan <= 100)` is the only form of this
        # test that catches it, and it is the form people get wrong. See
        # `InvalidPercentage` for where the NaN comes from in the first place.
        if not math.isfinite(number):
            raise InvalidPercentage(
                "percentage is not a finite number; an empty divisor in the "
                "spreadsheet produces this",
                field="percentage",
            )
        if number < cls.MINIMUM or number > cls.MAXIMUM:
            raise InvalidPercentage(
                f"percentage must be between {cls.MINIMUM:g} and {cls.MAXIMUM:g} "
                f"(got {number:g})",
                field="percentage",
            )
        return number

    @classmethod
    def parse(cls, raw: object) -> "Percentage":
        """Build one from a spreadsheet cell. Raises `InvalidPercentage` on absence.

        Use this when a mark is required. Use `parse_optional` for a column where a
        blank legitimately means "not graded yet".
        """
        parsed = cls.parse_optional(raw)
        if parsed is None:
            raise InvalidPercentage("a percentage is required here", field="percentage")
        return parsed

    @classmethod
    def parse_optional(cls, raw: object) -> "Percentage | None":
        """Build one from a cell, or `None` when the cell says no mark was given.

        `None` here is the "not graded yet" of `SubjectGrade.percentage`, and it is why
        this method exists separately from the constructor: the conversion from an empty
        cell to a stored value happens in exactly one place, so no parser can decide on
        its own that blank means zero.
        """
        if raw is None:
            return None
        if isinstance(raw, str):
            text = raw.translate(_INVISIBLE).strip()
            if text.casefold() in _ABSENT_TOKENS:
                return None
            # A trailing '%' is what a human types; the number is the same either way.
            text = text.removesuffix("%").strip()
            try:
                return cls(float(text))
            except ValueError:
                raise InvalidPercentage(
                    f"{raw!r} is not a percentage", field="percentage"
                ) from None
        return cls(raw)  # type: ignore[arg-type]  # validated in __post_init__

    def __str__(self) -> str:
        return f"{self.value:g}"
