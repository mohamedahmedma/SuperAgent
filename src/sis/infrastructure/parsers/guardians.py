"""Reading a guardians upload into rows, and everything else into diagnostics.

The implementation behind `sis.application.ports.parsers.GuardianFileParser`, assembled
the same way `roster.py` is: `workbook.load_sheet` turns bytes into lines of named cells,
`columns.map_columns` decides which header feeds which field, and everything below decides
what one line means.

**A line that cannot be understood is a value, not an exception.** Every per-row failure
becomes one `RowOutcome` and the loop continues, because a registrar uploading 300 parents
has mistyped one number and must not lose the other 299. Only a whole-file problem raises.

One thing here is not pure in the way the roster parser is: turning `01001234567` into
`+201001234567` needs a default country, which the file does not state. It arrives in the
constructor rather than from the environment, so the same bytes still parse to the same
rows wherever this runs and a test can pin it.

`relationship_type` is the one field that **cannot fail a row**. Unrecognised text buckets
to `RelationshipType.OTHER` and survives verbatim in `relationship_label`, so closing the
vocabulary costs nothing a human typed — which is why degrading is safe here and would not
be for a grade, where a guess replaces a fact nobody can recover.

The private helpers at the bottom (`_text`, `_require_a_faithful_read`) are deliberate
copies of `roster.py`'s rather than shared imports: `grades.py` already keeps its own, so
three small copies is the convention here, and reaching across modules for another
parser's underscore-prefixed function would be the novelty.
"""
from collections.abc import Mapping
from typing import Final

from sis.application.dto import ParsedGuardianRow, ParseResult, RowCode, RowOutcome
from sis.domain.errors import UnreadableImportFile, ValidationError
from sis.domain.guardians import RelationshipType
from sis.domain.value_objects import Phone, StudentNumber
from sis.infrastructure.parsers.columns import (
    CAN_VIEW_RECORDS,
    GUARDIAN_ALT_PHONE,
    GUARDIAN_NAME_AR,
    GUARDIAN_NAME_EN,
    GUARDIAN_PHONE,
    IS_PRIMARY_CONTACT,
    RELATIONSHIP_LABEL,
    RELATIONSHIP_TYPE,
    RESTRICTION_NOTE,
    STUDENT_NUMBER,
    ColumnMap,
    ColumnSpec,
    map_columns,
    normalise_header,
)
from sis.infrastructure.parsers.workbook import MAX_ROWS, Sheet, load_sheet

__all__ = ["GUARDIAN_COLUMNS", "SpreadsheetGuardianParser"]

#: The columns a guardians file may carry. `STUDENT_NUMBER` and `GUARDIAN_PHONE` are the
#: required pair — who this adult belongs to, and how to reach her. The
#: one-name-column-at-least rule is enforced below, since `ColumnMap.require` can only
#: speak about single fields.
GUARDIAN_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    STUDENT_NUMBER,
    GUARDIAN_PHONE,
    GUARDIAN_ALT_PHONE,
    GUARDIAN_NAME_AR,
    GUARDIAN_NAME_EN,
    RELATIONSHIP_TYPE,
    RELATIONSHIP_LABEL,
    IS_PRIMARY_CONTACT,
    CAN_VIEW_RECORDS,
    RESTRICTION_NOTE,
)

# Mirrors `_NAME_LEN` in `sis.infrastructure.db.models`, for the reason `roster.py` gives:
# an over-long name must be a rejected row with a reason rather than a value that passes
# validation and is silently truncated on write.
_MAX_NAME_LENGTH: Final[int] = 160

# Bilingual synonyms for the closed vocabulary, keyed by `normalise_header` output so a
# registrar's spacing, casing, diacritics and Arabic letter variants all fold away. Reusing
# that function as a general text fold is deliberate: the words a school types about "the
# mother" deserve exactly as much generosity as the words it types in a column heading.
_RELATIONSHIPS: Final[dict[str, RelationshipType]] = {
    normalise_header(word): kind
    for words, kind in (
        (("mother", "mom", "mum", "أم", "الأم", "والدة", "الوالدة"), RelationshipType.MOTHER),
        (("father", "dad", "أب", "الأب", "والد", "الوالد"), RelationshipType.FATHER),
        (
            ("guardian", "parent", "legal guardian", "ولي الأمر", "وصي", "الوصي"),
            RelationshipType.GUARDIAN,
        ),
        (
            ("sibling", "brother", "sister", "أخ", "أخت", "الأخ", "الأخت", "شقيق", "شقيقة"),
            RelationshipType.SIBLING,
        ),
        (
            (
                "grandparent",
                "grandfather",
                "grandmother",
                "جد",
                "جدة",
                "الجد",
                "الجدة",
            ),
            RelationshipType.GRANDPARENT,
        ),
        (("other", "أخرى", "اخرى", "غير ذلك"), RelationshipType.OTHER),
    )
    for word in words
}

# Shortest synonym eligible for substring matching. Below this, Arabic's two-letter
# kinship words ("أم", "أخ") sit inside longer unrelated words -- "ولي الأمر" contains
# "الأم" -- and would claim cells that are not about them.
_MIN_CONTAINED: Final[int] = 4

# The same table as a longest-first sequence, so "grandmother" is tested before "mother"
# and a grandparent is never filed as a parent. Built once at import; see `_relationship`.
_CONTAINABLE: Final[tuple[tuple[str, RelationshipType], ...]] = tuple(
    sorted(
        ((word, kind) for word, kind in _RELATIONSHIPS.items() if len(word) >= _MIN_CONTAINED),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)

# What a registrar writes in a yes/no column. Anything unrecognised falls back to the
# field's own default rather than being read as "no": guessing `False` on
# `can_view_records` would silently bar a parent the school meant to admit, and the
# registrar would only find out when the parent phoned to say the app shows nothing.
_TRUE_WORDS: Final[frozenset[str]] = frozenset(
    normalise_header(word)
    for word in ("yes", "y", "true", "1", "x", "نعم", "صح", "مسموح", "متاح")
)
_FALSE_WORDS: Final[frozenset[str]] = frozenset(
    normalise_header(word)
    for word in ("no", "n", "false", "0", "لا", "خطأ", "ممنوع", "غير مسموح")
)


class SpreadsheetGuardianParser:
    """Turns a CSV or XLSX guardians sheet into `ParsedGuardianRow`s plus diagnostics.

    Stateless apart from the default country code, so one instance serves the life of the
    process.

    Duplicates within one file are deliberately not detected here, matching
    `SpreadsheetRosterParser`: the service has to check the pairing against stored links
    anyway, and splitting the check would report the same line twice.
    """

    def __init__(self, *, default_country_code: str) -> None:
        self._default_country_code = default_country_code

    def parse(self, content: bytes, filename: str) -> ParseResult[ParsedGuardianRow]:
        """Read `content`; see `GuardianFileParser.parse` for the errors this may raise."""
        sheet = load_sheet(content, filename)
        columns = map_columns(sheet.headers, GUARDIAN_COLUMNS)
        columns.require()
        _require_a_name_column(columns)
        _require_a_faithful_read(sheet, columns)

        rows: list[ParsedGuardianRow] = []
        diagnostics: list[RowOutcome] = []
        for row in sheet.rows:
            outcome = _parse_row(
                row.line, row.cells, columns, self._default_country_code
            )
            if isinstance(outcome, RowOutcome):
                diagnostics.append(outcome)
            else:
                rows.append(outcome)

        return ParseResult(
            rows=tuple(rows),
            diagnostics=tuple(diagnostics),
            total_lines=sheet.total_lines,
            headers=sheet.headers,
        )


def _parse_row(
    line: int,
    cells: Mapping[str, object],
    columns: ColumnMap,
    default_country_code: str,
) -> ParsedGuardianRow | RowOutcome:
    """One data line: the pairing it asserts, or the reason it asserts nothing usable."""
    raw_number = columns.value(cells, STUDENT_NUMBER.field)
    raw_phone = columns.value(cells, GUARDIAN_PHONE.field)
    payload: dict[str, object] = {
        STUDENT_NUMBER.field: _text(raw_number),
        GUARDIAN_PHONE.field: _text(raw_phone),
    }

    try:
        student_number = StudentNumber(raw_number)  # type: ignore[arg-type]  # takes cells
    except ValidationError as error:
        return RowOutcome.from_error(
            line, RowCode.MISSING_STUDENT_NUMBER, error, payload=payload
        )

    try:
        phone = Phone.parse(raw_phone, default_country_code=default_country_code)
    except ValidationError as error:
        # One code for blank, malformed and wrong-length alike. The registrar's fix is the
        # same in every case — look at the phone cell on that line — and the difference is
        # carried in `message`.
        return RowOutcome.from_error(line, RowCode.MISSING_PHONE, error, payload=payload)

    alt_phone: Phone | None = None
    raw_alt = columns.value(cells, GUARDIAN_ALT_PHONE.field)
    if raw_alt is not None:
        payload[GUARDIAN_ALT_PHONE.field] = _text(raw_alt)
        try:
            alt_phone = Phone.parse_optional(
                raw_alt, default_country_code=default_country_code
            )
        except ValidationError as error:
            # Rejected rather than dropped. A second number that cannot be read is a cell
            # the registrar typed and meant; silently discarding it would leave her
            # believing the school can reach a parent on a line it never stored.
            return RowOutcome.from_error(
                line, RowCode.MISSING_PHONE, error, payload=payload
            )

    name_ar = columns.text(cells, GUARDIAN_NAME_AR.field) or ""
    name_en = columns.text(cells, GUARDIAN_NAME_EN.field) or ""
    if not name_ar and not name_en:
        return RowOutcome(
            line=line,
            code=RowCode.INVALID_NAME,
            message="a guardian needs a name in Arabic or in English; both cells are empty",
            payload=payload,
            field=GUARDIAN_NAME_AR.field,
        )
    for field_name, name in (
        (GUARDIAN_NAME_AR.field, name_ar),
        (GUARDIAN_NAME_EN.field, name_en),
    ):
        if len(name) > _MAX_NAME_LENGTH:
            return RowOutcome(
                line=line,
                code=RowCode.INVALID_NAME,
                message=(
                    f"this name is longer than {_MAX_NAME_LENGTH} characters; "
                    "have two columns run together?"
                ),
                payload=payload | {"name": name},
                field=field_name,
            )

    relationship, fallback_label = _relationship(columns.text(cells, RELATIONSHIP_TYPE.field))
    label = columns.text(cells, RELATIONSHIP_LABEL.field) or fallback_label

    return ParsedGuardianRow(
        line_number=line,
        student_number=student_number,
        phone=phone,
        full_name_ar=name_ar,
        full_name_en=name_en,
        relationship_type=relationship,
        relationship_label=label,
        alt_phone=alt_phone,
        is_primary_contact=_flag(
            columns.text(cells, IS_PRIMARY_CONTACT.field), default=False
        ),
        can_view_records=_flag(
            columns.text(cells, CAN_VIEW_RECORDS.field), default=True
        ),
        restriction_note=columns.text(cells, RESTRICTION_NOTE.field) or "",
    )


def _relationship(raw: str | None) -> tuple[RelationshipType, str]:
    """Bucket free text into the closed vocabulary. Never rejects a row.

    Returns the type and the label to fall back on. A cell that matched a synonym exactly
    contributes no label — `RelationshipType.MOTHER` already says everything "mother"
    said — while anything else is handed back verbatim, so "big brother" is stored as
    `SIBLING` *and* keeps the word "big".

    The containment pass is what makes that second case work at all. Registrars write
    "big brother" and "الأخ الأكبر", not "sibling", and an exact-match-only table would
    bucket every one of them to `OTHER` — leaving the closed vocabulary technically
    correct and useless for the counting it exists to support.

    Two guards keep containment from over-reaching. Candidates are tried **longest
    first**, so "grandmother" claims the cell before "mother" can; and words shorter than
    `_MIN_CONTAINED` are exact-match only, because Arabic's two-letter kinship words
    ("أم", "أخ") appear inside longer unrelated ones and would otherwise match half the
    column.
    """
    if not raw or not raw.strip():
        return RelationshipType.OTHER, ""
    folded = normalise_header(raw)
    matched = _RELATIONSHIPS.get(folded)
    if matched is not None:
        return matched, ""
    for word, kind in _CONTAINABLE:
        if word in folded:
            return kind, raw.strip()
    return RelationshipType.OTHER, raw.strip()


def _flag(raw: str | None, *, default: bool) -> bool:
    """Read a yes/no cell, falling back to `default` on anything unrecognised.

    Falling back rather than rejecting is the right way round here because both flags have
    a safe-by-construction default and neither is worth failing a whole row over. An
    unrecognised word in `can_view_records` keeps the grant the registrar's upload implies
    rather than inventing a restriction she never wrote.
    """
    if raw is None:
        return default
    text = normalise_header(raw)
    if not text:
        return default
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return default


def _require_a_name_column(columns: ColumnMap) -> None:
    """Refuse a file that names nobody.

    Either guardian-name column alone is fine; neither is not. `ColumnSpec.required`
    cannot say "one of these two", so the rule lives here — in the same shape and with the
    same evidence `ColumnMap.require` produces, so a registrar sees one kind of message.
    """
    if columns.has(GUARDIAN_NAME_AR.field) or columns.has(GUARDIAN_NAME_EN.field):
        return
    found = ", ".join(repr(header) for header in columns.headers) or "none at all"
    accepted = ", ".join(
        repr(alias)
        for spec in (GUARDIAN_NAME_AR, GUARDIAN_NAME_EN)
        for alias in spec.aliases
    )
    raise UnreadableImportFile(
        f"no guardian name column was found; an Arabic or an English one is required "
        f"(accepted: {accepted}). The columns in this file are: {found}.",
        field="headers",
    )


def _require_a_faithful_read(sheet: Sheet, columns: ColumnMap) -> None:
    """Refuse a file this parser has demonstrably not read in full.

    Both cases would otherwise import cleanly and report success over an incomplete
    result — the registrar has no way to notice that a quarter of the parents never
    arrived, or that the second column called "Phone" was the one that mattered.
    """
    if sheet.truncated:
        raise UnreadableImportFile(
            f"this file holds more than {MAX_ROWS:,} rows and the read stopped there; "
            f"split it and upload the parts. Importing what was read would attach part of "
            f"a year group's parents and call it done.",
            field="file",
        )
    used = set(columns.mapping.values())
    shadowed = tuple(header for header in sheet.duplicate_headers if header in used)
    if shadowed:
        raise UnreadableImportFile(
            f"{', '.join(repr(header) for header in shadowed)} appears more than once; "
            f"only the first column of that name would be read and the rest ignored. "
            f"Delete or rename the duplicate and upload again.",
            field="headers",
        )


def _text(raw: object) -> str:
    """A cell as text for a diagnostic payload; an absent cell as `''`.

    Integral floats go through `int` because a phone column arrives from openpyxl as a
    number, and quoting `1001234567.0` back at a registrar who typed `01001234567` sends
    her looking for a decimal point that is not in her file.
    """
    if raw is None:
        return ""
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    return str(raw).strip()
