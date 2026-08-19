"""Finding a file's columns by their names. Never by their positions.

Position is not evidence. "The student number is the first column" holds until a school
adds a serial column in front of it, and then every grade in the upload attaches to the
child one row's identity to the left — silently, because each row is still well formed.
This module maps headers to fields by matching text, and a file whose headers it cannot
match is refused with a diagnostic quoting the headers it actually found, so the fix is
a rename in the spreadsheet rather than a mystery.

Matching is deliberately generous about *spelling* and strict about *meaning*. Headers
are folded before comparison: NFKC (which collapses the Arabic presentation forms that
arrive from older Windows exports, where a header rendered "الصف" is stored as a run of
contextual ligature codepoints and compares unequal to the same word typed today),
invisible bidi marks removed, case ignored, spaces and punctuation ignored, Arabic-Indic
digits and letter variants unified. "Student No.", "student_no" and "STUDENT NO" are one
header; "اسم الطالب" and "أسم الطالبه" are one header.

What it will not do is guess at meaning. Two traps are wired in on purpose, both of them
columns that read the same and mean different things:

* English "Grade" is a *mark* in a grade sheet and a *year level* in a roster. It is an
  alias of the mark only; a year level must be spelled "Year" or "Grade Level".
* Arabic "الفصل" is a class section, while "الفصل الدراسي" is a term. Folding removes the
  space between the words but not the words, so the two stay distinguishable.

An unqualified "Name" is an alias of nothing. A single name column is no evidence of
which language it holds, and a wrong guess prints an Arabic name into the English field
of every report card the school issues.
"""
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Final

from sis.domain.errors import UnreadableImportFile

__all__ = [
    "ACADEMIC_YEAR_CODE",
    "CAN_VIEW_RECORDS",
    "CLASS_CODE",
    "ColumnMap",
    "ColumnSpec",
    "ENDS_ON",
    "FULL_NAME_AR",
    "FULL_NAME_EN",
    "GUARDIAN_ALT_PHONE",
    "GUARDIAN_NAME_AR",
    "GUARDIAN_NAME_EN",
    "GUARDIAN_PHONE",
    "IS_PRIMARY_CONTACT",
    "MAX_POINTS",
    "PERCENTAGE",
    "POINTS",
    "RELATIONSHIP_LABEL",
    "RELATIONSHIP_TYPE",
    "RESTRICTION_NOTE",
    "STARTS_ON",
    "STUDENT_NUMBER",
    "SUBJECT_CODE",
    "TERM_CODE",
    "YEAR_LEVEL_CODE",
    "map_columns",
    "normalise_digits",
    "normalise_header",
]

# Arabic-Indic (٠-٩) and extended Arabic-Indic (۰-۹) digits, plus the separators that
# travel with them. Real school spreadsheets are full of these: a teacher's numeric
# keypad on an Arabic layout produces them, and `float("٨٥")` raises, so a whole column
# of perfectly good marks reads as unparseable and the registrar is told her file is
# broken when it is not.
_DIGITS: Final[dict[int, str]] = {
    **{0x0660 + digit: str(digit) for digit in range(10)},
    **{0x06F0 + digit: str(digit) for digit in range(10)},
    0x066B: ".",  # Arabic decimal separator
    0x066C: "",  # Arabic thousands separator
    0x066A: "%",  # Arabic percent sign
}

# Diacritics and the elongation dash. None of them change which word a header is, and
# all of them survive a copy out of a formatted document.
_ARABIC_MARKS: Final[dict[int, None]] = {
    codepoint: None for codepoint in (*range(0x064B, 0x0653), 0x0640, 0x0670)
}

# Letters a typist chooses between without meaning anything by it. Unifying them is what
# makes "اسم الطالب" match "أسم الطالبه".
_ARABIC_LETTERS: Final[dict[int, str]] = {
    0x0622: "ا",  # آ  -> ا
    0x0623: "ا",  # أ  -> ا
    0x0625: "ا",  # إ  -> ا
    0x0671: "ا",  # ٱ  -> ا
    0x0629: "ه",  # ة  -> ه
    0x0649: "ي",  # ى  -> ي
    0x0624: "و",  # ؤ  -> و
    0x0626: "ي",  # ئ  -> ي
}


def normalise_digits(text: str) -> str:
    """Rewrite Arabic-Indic digits and separators as ASCII.

    Applied to numeric cells and to headers, and **never to a student number**. The
    domain rejects Arabic-Indic digits in `StudentNumber` on purpose: two roster rows
    whose numbers differ only by digit script are two spellings a human must reconcile,
    because transliterating them here would merge two children into one — or split one
    child's grades across two identities — with no record that a choice was made. A mark
    carries no identity, so normalising one is arithmetic rather than a decision.
    """
    return text.translate(_DIGITS)


def normalise_header(text: object) -> str:
    """Fold a header down to the key it is matched by. Empty when it names nothing."""
    if text is None:
        return ""
    folded = unicodedata.normalize("NFKC", str(text))
    # Category Cf is every invisible formatting codepoint at once — the RTL mark, the
    # zero-width joiners, the isolates, a stray BOM from a UTF-8 CSV. A header carrying
    # one is visibly identical to the alias it fails to match.
    folded = "".join(char for char in folded if unicodedata.category(char) != "Cf")
    folded = normalise_digits(folded.casefold())
    folded = folded.translate(_ARABIC_MARKS).translate(_ARABIC_LETTERS)
    # Everything that is not a letter or digit goes, which is what makes the match
    # insensitive to spaces, underscores, dots, colons and brackets in one rule. '%' is
    # kept because a column headed "%" is a real grade column and would otherwise fold
    # to nothing.
    return "".join(char for char in folded if char.isalnum() or char == "%")


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One field, and every header text that legitimately names it."""

    #: The name the parsed row will carry — matches the DTO attribute it feeds.
    field: str
    #: Written in natural spelling; folded once at construction. Quoted verbatim in the
    #: missing-column diagnostic, so they must stay readable.
    aliases: tuple[str, ...]
    required: bool = False
    #: Whether `ColumnMap.number` may rewrite this column's digits. See `normalise_digits`.
    numeric: bool = False
    keys: frozenset[str] = dataclass_field(default=frozenset(), init=False, repr=False)

    def __post_init__(self) -> None:
        keys = {normalise_header(alias) for alias in (self.field, *self.aliases)}
        object.__setattr__(self, "keys", frozenset(keys - {""}))


@dataclass(frozen=True, slots=True)
class ColumnMap:
    """Which header in this file feeds which field, plus what could not be resolved."""

    #: field -> the header text exactly as it appeared, which is also the row-dict key.
    mapping: Mapping[str, str]
    #: Every header the file had, in order. The diagnostic's evidence.
    headers: tuple[str, ...] = ()
    #: Required fields no header claimed.
    missing: tuple[str, ...] = ()
    #: Headers that matched nothing. Not an error — schools carry extra columns.
    unmapped: tuple[str, ...] = ()
    #: field -> the several headers that all claimed it, or header -> the several fields
    #: that all claimed it. Either way nothing is chosen; see `require`.
    ambiguous: Mapping[str, tuple[str, ...]] = dataclass_field(default_factory=dict)
    #: The specs this map was built from, kept so the diagnostic can offer the spellings
    #: that would have worked instead of only naming the field that is absent.
    specs: tuple[ColumnSpec, ...] = ()

    def has(self, field: str) -> bool:
        return field in self.mapping

    def header_for(self, field: str) -> str | None:
        return self.mapping.get(field)

    def value(self, cells: Mapping[str, object], field: str) -> object:
        """The raw cell, or `None` when this file has no such column.

        `None` for an absent column and `None` for a blank cell are deliberately the same
        answer: both mean nothing was stated, and decision 1 says nothing-stated is never
        zero. A caller that must distinguish them asks `has()`.
        """
        header = self.mapping.get(field)
        return None if header is None else cells.get(header)

    def text(self, cells: Mapping[str, object], field: str) -> str | None:
        """The cell as text, blank meaning `None`.

        Deliberately *not* folded the way a header is. A header is matched and thrown
        away; a name is stored and shown to a parent, so rewriting its letters here would
        mean the school's records spell a child's name differently from the school's own
        spreadsheet. Only surrounding whitespace goes.
        """
        value = self.value(cells, field)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def number(self, cells: Mapping[str, object], field: str) -> object:
        """The cell prepared for `Percentage.parse_optional` or an `int`/`float` caller.

        Numbers pass through as they are; text has its Arabic-Indic digits rewritten. An
        ASCII comma is left alone, because "85,5" is eighty-five point five to half of
        Europe and eight hundred and fifty-five to the other half — a row rejected for an
        unreadable mark is recoverable, a mark silently multiplied by ten is not.
        """
        value = self.value(cells, field)
        if value is None or not isinstance(value, str):
            return value
        text = normalise_digits(unicodedata.normalize("NFKC", value)).strip()
        return text or None

    def require(self) -> None:
        """Raise unless every required field resolved to exactly one column.

        Ambiguity is refused rather than resolved by column order. A file holding both
        "Student ID" and "Student Number" is a file where somebody merged two exports,
        and picking the leftmost silently attaches grades to whichever identity happened
        to be first — the failure this whole module exists to prevent. The registrar
        deletes a column and re-uploads, which takes a minute and is a decision she made.
        """
        problems = [
            f"the column for {spec.field!r} was not found "
            f"(accepted: {', '.join(repr(alias) for alias in spec.aliases)})"
            for spec in self.specs
            if spec.field in self.missing
        ]
        problems += [
            f"{field!r} is claimed by more than one column: "
            f"{', '.join(repr(name) for name in names)}"
            for field, names in self.ambiguous.items()
        ]
        if not problems:
            return
        found = ", ".join(repr(header) for header in self.headers) or "none at all"
        raise UnreadableImportFile(
            f"{'; '.join(problems)}. The columns in this file are: {found}.",
            field="headers",
        )


def map_columns(headers: Sequence[str], specs: Iterable[ColumnSpec]) -> ColumnMap:
    """Match this file's headers against `specs`. Reports; does not raise.

    Separated from `ColumnMap.require` so a caller may inspect what was found before
    deciding whether a missing optional column matters.
    """
    specs = tuple(specs)
    claims: dict[str, list[str]] = {}
    contested: dict[str, tuple[str, ...]] = {}
    unmapped: list[str] = []

    for header in headers:
        key = normalise_header(header)
        claimants = [spec.field for spec in specs if key in spec.keys]
        if not claimants:
            unmapped.append(header)
        elif len(claimants) > 1:
            # One header naming two fields is a spec bug, not a file problem, but it is
            # reported the same way rather than resolved silently.
            contested[header] = tuple(claimants)
        else:
            claims.setdefault(claimants[0], []).append(header)

    mapping = {field: found[0] for field, found in claims.items() if len(found) == 1}
    contested.update(
        {field: tuple(found) for field, found in claims.items() if len(found) > 1}
    )
    # A contested field is not also reported as missing: telling a registrar that
    # 'student_number' was not found, about a file with two columns of student numbers,
    # sends her to add a third.
    missing = tuple(
        spec.field
        for spec in specs
        if spec.required and spec.field not in mapping and spec.field not in contested
    )
    return ColumnMap(
        mapping=mapping,
        headers=tuple(headers),
        missing=missing,
        unmapped=tuple(unmapped),
        ambiguous=contested,
        specs=specs,
    )


# ---------------------------------------------------------------------------
# The shared columns. Importers compose these; `required` is theirs to set, since the
# same subject column is mandatory in a grade file and absent from a roster.
# ---------------------------------------------------------------------------

# A bare "الرقم", "No." or "ID" is not here on purpose: it is as often the serial number
# down the left margin, and reading a serial as a student number attaches an upload to
# whichever children happen to sit at those positions. A national ID is likewise a
# different identifier, not a synonym.
STUDENT_NUMBER: Final[ColumnSpec] = ColumnSpec(
    field="student_number",
    aliases=(
        "student number",
        "student no",
        "student id",
        "student code",
        "admission number",
        "رقم الطالب",
        "رقم الطالبة",
        "الرقم الأكاديمي",
        "رقم القيد",
        "رقم التسجيل",
    ),
    required=True,
)

FULL_NAME_AR: Final[ColumnSpec] = ColumnSpec(
    field="full_name_ar",
    aliases=(
        "name ar",
        "arabic name",
        "name (arabic)",
        "student name (arabic)",
        "اسم الطالب",
        "اسم الطالبة",
        "الاسم بالعربي",
        "الاسم العربي",
        "الاسم رباعي",
    ),
)

FULL_NAME_EN: Final[ColumnSpec] = ColumnSpec(
    field="full_name_en",
    aliases=(
        "name en",
        "english name",
        "name (english)",
        "student name (english)",
        "الاسم بالانجليزي",
        "الاسم الانجليزي",
        "الاسم باللغة الانجليزية",
    ),
)

CLASS_CODE: Final[ColumnSpec] = ColumnSpec(
    field="class_code",
    aliases=("class", "class code", "section", "class/section", "الفصل", "الشعبة"),
)

# "grade" is absent here and present on PERCENTAGE — see the module docstring.
YEAR_LEVEL_CODE: Final[ColumnSpec] = ColumnSpec(
    field="year_level_code",
    aliases=("year", "year level", "grade level", "الصف", "المرحلة", "الصف الدراسي"),
)

ACADEMIC_YEAR_CODE: Final[ColumnSpec] = ColumnSpec(
    field="academic_year_code",
    aliases=("academic year", "school year", "العام الدراسي", "السنة الدراسية"),
)

TERM_CODE: Final[ColumnSpec] = ColumnSpec(
    field="term_code",
    aliases=("term", "semester", "term code", "الفصل الدراسي", "الترم", "الفترة"),
)

SUBJECT_CODE: Final[ColumnSpec] = ColumnSpec(
    field="subject_code",
    aliases=("subject", "subject code", "course", "المادة", "المقرر", "اسم المادة"),
)

PERCENTAGE: Final[ColumnSpec] = ColumnSpec(
    field="percentage",
    aliases=(
        "percentage",
        "percent",
        "%",
        "grade",
        "mark",
        "marks",
        "score",
        "result",
        "الدرجة",
        "النسبة",
        "النسبة المئوية",
        "العلامة",
    ),
    numeric=True,
)

# Kept disjoint from PERCENTAGE's aliases: a header both could claim would be refused as
# ambiguous, which is correct behaviour and a terrible default experience.
POINTS: Final[ColumnSpec] = ColumnSpec(
    field="points",
    aliases=("points", "points earned", "raw score", "الدرجة المكتسبة", "الدرجة المستحقة"),
    numeric=True,
)

MAX_POINTS: Final[ColumnSpec] = ColumnSpec(
    field="max_points",
    aliases=("max points", "out of", "total points", "الدرجة العظمى", "النهاية العظمى"),
    numeric=True,
)

# --- Guardians -------------------------------------------------------------
# The guardian's own names are separate specs from FULL_NAME_AR / FULL_NAME_EN, not
# aliases of them. A real guardian sheet carries the *child's* name too, so a registrar
# can read the file she is uploading, and folding the two into one spec would make every
# such file ambiguous — or, worse, silently file the child's name as her mother's.

GUARDIAN_PHONE: Final[ColumnSpec] = ColumnSpec(
    field="phone",
    aliases=(
        "phone",
        "phone number",
        "mobile",
        "mobile number",
        "contact",
        "contact number",
        "guardian phone",
        "parent phone",
        "رقم الهاتف",
        "رقم الجوال",
        "رقم الموبايل",
        "الجوال",
        "الهاتف",
        "الموبايل",
        "هاتف ولي الأمر",
        "رقم ولي الأمر",
        "رقم التواصل",
    ),
    required=True,
)

# A second number for the *same* adult. The only way to reach the second phone a guardian
# holds, because the primary number is her identity: two rows carrying two unfamiliar
# numbers are two different people and no name-matching can safely say otherwise.
GUARDIAN_ALT_PHONE: Final[ColumnSpec] = ColumnSpec(
    field="alt_phone",
    aliases=(
        "alt phone",
        "alternate phone",
        "second phone",
        "other phone",
        "phone 2",
        "mobile 2",
        "whatsapp",
        "whatsapp number",
        "رقم آخر",
        "رقم اخر",
        "هاتف آخر",
        "رقم بديل",
        "الرقم الثاني",
        "رقم الواتساب",
    ),
)

GUARDIAN_NAME_AR: Final[ColumnSpec] = ColumnSpec(
    field="guardian_name_ar",
    aliases=(
        "guardian name ar",
        "guardian name (arabic)",
        "parent name (arabic)",
        "guardian arabic name",
        "اسم ولي الأمر",
        "اسم ولي الامر",
        "اسم ولي الأمر بالعربي",
        "اسم الوالد",
        "اسم الوصي",
    ),
)

GUARDIAN_NAME_EN: Final[ColumnSpec] = ColumnSpec(
    field="guardian_name_en",
    aliases=(
        "guardian name en",
        "guardian name (english)",
        "parent name (english)",
        "guardian english name",
        "guardian name",
        "parent name",
        "اسم ولي الأمر بالانجليزي",
        "اسم ولي الامر بالانجليزي",
    ),
)

RELATIONSHIP_TYPE: Final[ColumnSpec] = ColumnSpec(
    field="relationship_type",
    aliases=(
        "relationship",
        "relation",
        "relationship type",
        "guardian type",
        "kinship",
        "صلة القرابة",
        "القرابة",
        "صفة ولي الأمر",
        "نوع القرابة",
        "العلاقة",
    ),
)

# Free text kept beside the bucketed type. When this column is absent the parser falls
# back to the relationship cell's own words, so "big brother" survives either way.
RELATIONSHIP_LABEL: Final[ColumnSpec] = ColumnSpec(
    field="relationship_label",
    aliases=(
        "relationship label",
        "relationship detail",
        "relation detail",
        "relationship note",
        "تفاصيل القرابة",
        "وصف القرابة",
        "ملاحظة القرابة",
    ),
)

IS_PRIMARY_CONTACT: Final[ColumnSpec] = ColumnSpec(
    field="is_primary_contact",
    aliases=(
        "primary contact",
        "main contact",
        "is primary",
        "primary",
        "جهة الاتصال الرئيسية",
        "المسؤول الأول",
        "الاتصال الأساسي",
    ),
)

# Absent means granted, which is the opposite of the database's own default. See
# `ParsedGuardianRow.can_view_records`: the column protects a machine that never asked
# the question, and the sheet is a registrar who has.
CAN_VIEW_RECORDS: Final[ColumnSpec] = ColumnSpec(
    field="can_view_records",
    aliases=(
        "can view records",
        "may view records",
        "records access",
        "access",
        "can see grades",
        "يمكنه الاطلاع",
        "الاطلاع على السجلات",
        "صلاحية الاطلاع",
    ),
)

RESTRICTION_NOTE: Final[ColumnSpec] = ColumnSpec(
    field="restriction_note",
    aliases=(
        "restriction note",
        "restriction",
        "restriction reason",
        "access note",
        "سبب المنع",
        "ملاحظة التقييد",
        "سبب التقييد",
    ),
)

STARTS_ON: Final[ColumnSpec] = ColumnSpec(
    field="starts_on",
    aliases=("start date", "starts on", "from", "effective from", "تاريخ البداية", "من تاريخ"),
)

ENDS_ON: Final[ColumnSpec] = ColumnSpec(
    field="ends_on",
    aliases=("end date", "ends on", "to", "until", "تاريخ النهاية", "إلى تاريخ"),
)
