"""Reading a roster upload into rows, and everything else into diagnostics.

The implementation behind `sis.application.ports.parsers.RosterFileParser`, assembled
from the two halves of this package: `workbook.load_sheet` turns bytes into lines of
named cells, `columns.map_columns` decides which header feeds which field, and everything
below decides what one line means. It is a pure function of the bytes it is handed — no
session, no clock, no disk — which is what lets the import service be tested with a
literal `b"..."` and what makes a preview taken on Monday still mean the same thing on
Tuesday.

**A line that cannot be understood is a value, not an exception.** Every per-row failure
here becomes one `RowOutcome` and the loop continues, because a registrar uploading 300
children has mistyped one number and must not lose the other 299 (decision 4). Only a
whole-file problem raises: a format nobody reads, a file that will not open, headers that
name no student number column, or a read this parser knows was incomplete.

Placement is deliberately thin. A row carries at most the class code the file states; the
enrolment *window* that makes placement a time-bounded membership rather than a column on
the student (decision 2) is filled by the service from the academic year. `columns` offers
a `STARTS_ON` spec and this parser pointedly does not read it: a date guessed from an
ambiguous `03/04/2026` would be stored as fact, and `RowCode` — closed on purpose — has no
member with which to reject a bad one. A column this parser cannot reject is a column it
must not read.
"""
from collections.abc import Mapping
from typing import Final

from dataclasses import replace

from sis.application.dto import ParsedRosterRow, ParseResult, RowCode, RowOutcome
from sis.domain.errors import UnreadableImportFile, ValidationError
from sis.domain.people import Gender
from sis.domain.value_objects import ClassCode, StudentNumber
from sis.infrastructure.parsers.columns import (
    normalise_header,
    CLASS_CODE,
    FULL_NAME_AR,
    GENDER,
    FULL_NAME_EN,
    STUDENT_NUMBER,
    ColumnMap,
    ColumnSpec,
    map_columns,
)
from sis.infrastructure.parsers.workbook import MAX_ROWS, Sheet, load_sheet

__all__ = ["ROSTER_COLUMNS", "SpreadsheetFamilyRosterParser", "SpreadsheetRosterParser"]

#: The columns a roster file may carry. `STUDENT_NUMBER` is the only required one — a
#: school that keeps names in Arabic alone files a perfectly good roster — so the
#: one-name-column-at-least rule is enforced below rather than by `ColumnMap.require`,
#: which can only speak about single fields.
ROSTER_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    STUDENT_NUMBER,
    FULL_NAME_AR,
    FULL_NAME_EN,
    CLASS_CODE,
    GENDER,
)

# Mirrors `_NAME_LEN` in `sis.infrastructure.db.models`. Checked here so an over-long name
# is a rejected row with a reason, rather than a value that passes every validation and is
# then truncated on write — which loses the tail of a name silently and leaves two
# children looking like one.
_MAX_NAME_LENGTH: Final[int] = 160


class SpreadsheetRosterParser:
    """Turns a CSV or XLSX roster into `ParsedRosterRow`s plus per-line diagnostics.

    Stateless, so one instance serves the life of the process. Unlike the grade parser it
    needs no per-upload defaults: `ParsedRosterRow.class_code` is legitimately optional,
    and choosing between the class a row names and the class the request names is the
    service's decision, made where both are visible.

    Duplicate student numbers within one file are deliberately not detected here. The
    service must check them against stored students anyway, and splitting the check would
    report the second copy of a number twice — once as `DUPLICATE_IN_FILE` from a parser
    and again from the service — which is the sort of double count that makes a registrar
    stop trusting the report.
    """

    def parse(self, content: bytes, filename: str) -> ParseResult[ParsedRosterRow]:
        """Read `content`; see `RosterFileParser.parse` for the errors this may raise."""
        sheet = load_sheet(content, filename)
        columns = map_columns(sheet.headers, ROSTER_COLUMNS)
        columns.require()
        _require_a_name_column(columns)
        _require_a_faithful_read(sheet, columns)

        rows: list[ParsedRosterRow] = []
        diagnostics: list[RowOutcome] = []
        for row in sheet.rows:  # blank lines were skipped by the reader, not counted here
            outcome = _parse_row(row.line, row.cells, columns)
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


class SpreadsheetFamilyRosterParser:
    """Read the optional guardian columns from the same physical roster row.

    A file with no guardian columns keeps the legacy roster behaviour.  Once any
    guardian column is present, the guardian parser validates that part and a bad
    guardian rejects the whole family row instead of creating an orphan student.
    """

    def __init__(self, *, default_country_code: str) -> None:
        from sis.infrastructure.parsers.guardians import SpreadsheetGuardianParser

        self._roster = SpreadsheetRosterParser()
        self._guardians = SpreadsheetGuardianParser(
            default_country_code=default_country_code
        )

    def parse(self, content: bytes, filename: str) -> ParseResult[ParsedRosterRow]:
        from sis.infrastructure.parsers.columns import map_columns
        from sis.infrastructure.parsers.guardians import GUARDIAN_COLUMNS

        roster = self._roster.parse(content, filename)
        guardian_map = map_columns(roster.headers, GUARDIAN_COLUMNS)
        guardian_fields = {spec.field for spec in GUARDIAN_COLUMNS if spec.field != "student_number"}
        if not guardian_fields.intersection(guardian_map.mapping):
            return roster

        guardians = self._guardians.parse(content, filename)
        by_line = {row.line_number: row for row in guardians.rows}
        rejected_lines = {row.line for row in guardians.diagnostics}
        rows = tuple(
            replace(row, guardian=by_line[row.line_number])
            for row in roster.rows
            if row.line_number in by_line and row.line_number not in rejected_lines
        )
        diagnostics = tuple(
            sorted((*roster.diagnostics, *guardians.diagnostics), key=lambda row: row.line)
        )
        return ParseResult(
            rows=rows,
            diagnostics=diagnostics,
            total_lines=roster.total_lines,
            headers=roster.headers,
        )


def _parse_row(
    line: int, cells: Mapping[str, object], columns: ColumnMap
) -> ParsedRosterRow | RowOutcome:
    """One data line: the row it asserts, or the reason it asserts nothing usable."""
    raw_number = columns.value(cells, STUDENT_NUMBER.field)
    payload: dict[str, object] = {STUDENT_NUMBER.field: _text(raw_number)}

    try:
        student_number = StudentNumber(raw_number)  # type: ignore[arg-type]  # takes cells
    except ValidationError as error:
        # One code for a blank cell and for a malformed one: `MISSING_STUDENT_NUMBER` is
        # the nearest member of a closed vocabulary, and the registrar's fix — go and type
        # the number properly — is the same either way. The difference is in `message`.
        return RowOutcome.from_error(line, RowCode.MISSING_STUDENT_NUMBER, error, payload=payload)

    name_ar = columns.text(cells, FULL_NAME_AR.field) or ""
    name_en = columns.text(cells, FULL_NAME_EN.field) or ""
    if not name_ar and not name_en:
        return RowOutcome(
            line=line,
            code=RowCode.INVALID_NAME,
            message="a child needs a name in Arabic or in English; both cells are empty",
            payload=payload,
            field=FULL_NAME_AR.field,
        )
    for field_name, name in ((FULL_NAME_AR.field, name_ar), (FULL_NAME_EN.field, name_en)):
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

    class_code: ClassCode | None = None
    raw_class = columns.value(cells, CLASS_CODE.field)
    if raw_class is not None:
        payload[CLASS_CODE.field] = _text(raw_class)
        try:
            class_code = ClassCode(raw_class)  # type: ignore[arg-type]  # takes cells
        except ValidationError as error:
            # `UNKNOWN_CLASS` rather than a malformed-code member, which the closed
            # vocabulary does not have: a code that cannot even be constructed names no
            # class this school runs, and that is what the registrar has to act on.
            return RowOutcome.from_error(line, RowCode.UNKNOWN_CLASS, error, payload=payload)

    return ParsedRosterRow(
        line_number=line,
        student_number=student_number,
        full_name_ar=name_ar,
        full_name_en=name_en,
        class_code=class_code,
        gender=_gender(columns.text(cells, GENDER.field)),
    )


def _require_a_name_column(columns: ColumnMap) -> None:
    """Refuse a file that names nobody.

    Either name column alone is fine; neither is not. `ColumnSpec.required` cannot say
    "one of these two", so the rule lives here — in the same shape and with the same
    evidence `ColumnMap.require` produces, so a registrar sees one kind of message.
    """
    if columns.has(FULL_NAME_AR.field) or columns.has(FULL_NAME_EN.field):
        return
    found = ", ".join(repr(header) for header in columns.headers) or "none at all"
    accepted = ", ".join(
        repr(alias) for spec in (FULL_NAME_AR, FULL_NAME_EN) for alias in spec.aliases
    )
    raise UnreadableImportFile(
        f"no name column was found; an Arabic or an English one is required "
        f"(accepted: {accepted}). The columns in this file are: {found}.",
        field="headers",
    )


def _require_a_faithful_read(sheet: Sheet, columns: ColumnMap) -> None:
    """Refuse a file this parser has demonstrably not read in full.

    Both cases would otherwise import cleanly and report success over an incomplete
    result, which is the failure mode the whole preview flow exists to make impossible:
    the registrar has no way to notice that a quarter of a year group never arrived, or
    that the second column called "Class" was the one that mattered.
    """
    if sheet.truncated:
        raise UnreadableImportFile(
            f"this file holds more than {MAX_ROWS:,} rows and the read stopped there; "
            f"split it and upload the parts. Importing what was read would write part of "
            f"a year group and call it done.",
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


#: What a registrar actually types in a gender column, folded through the same function
#: that folds headers — so spacing, casing, diacritics and Arabic letter variants all
#: disappear before the lookup. The same generosity the relationship words get, and for
#: the same reason: a school's spelling of "female" is not its fault.
_GENDERS: Final[dict[str, Gender]] = {
    normalise_header(word): value
    for words, value in (
        (("m", "male", "boy", "ذكر", "ذكور", "ولد", "طالب"), Gender.MALE),
        (("f", "female", "girl", "أنثى", "انثى", "اناث", "بنت", "طالبة"), Gender.FEMALE),
    )
    for word in words
}


def _gender(raw: str | None) -> Gender:
    """Read one cell. Never rejects a row, and never guesses.

    Anything unrecognised — a blank, a dash, a word nobody anticipated — becomes
    `UNSPECIFIED`, which is a real value meaning "the school has not said" rather than a
    default sex. That degradation is lossless in a way guessing would not be: the chat
    service treats `UNSPECIFIED` as matching neither "my son" nor "my daughter", so an
    unreadable cell costs one clarifying question, where a wrong guess would show one
    child's records while the parent asked about another.

    Deliberately exact-match only, with no containment pass. The relationship resolver
    needs containment because registrars write "big brother"; nobody writes "quite male",
    and the Arabic single letters here are short enough that containment would match
    inside unrelated words.
    """
    if not raw or not raw.strip():
        return Gender.UNSPECIFIED
    return _GENDERS.get(normalise_header(raw), Gender.UNSPECIFIED)


def _text(raw: object) -> str:
    """A cell as text for a diagnostic payload; an absent cell as `''`.

    Integral floats go through `int` because a column of digits arrives from openpyxl as a
    number, and quoting `12345.0` back at a registrar who typed `12345` sends her looking
    for a decimal point that is not in her file.
    """
    if raw is None:
        return ""
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    return str(raw).strip()
