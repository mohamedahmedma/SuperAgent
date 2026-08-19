"""Reading a marks upload, where the empty cell is the thing that matters.

The implementation behind `sis.application.ports.parsers.GradeFileParser`, built on
`workbook.load_sheet` and `columns.map_columns` exactly as the roster parser is, and pure
over the bytes it is given for the same reasons. A bad mark on line 14 is one `RowOutcome`
and lines 1 to 13 still import (decision 4); only a whole-file problem raises.

**The rule this module exists to hold: a blank marks cell parses to `None`, never to
`0.0`** (decision 1). The conversion happens in exactly two named places — `Percentage.
parse_optional` for the percentage column and `_is_unmarked` for the raw points column —
and both answer "nobody has marked this yet". Zero is a mark a child can earn, so a parser
that defaulted an empty cell to zero would not be filling a gap, it would be inventing a
failing grade that no later layer can tell from one a teacher typed, and what a parent is
eventually shown is 0% in a subject nobody has marked. Anywhere below, `percentage or 0.0`
is that bug spelled out; `ParsedGradeRow.is_graded` is the way to ask the question.

An unmarked row is still a row. It is returned with `percentage=None` rather than dropped,
because "enrolled in this subject, not yet marked" is a fact worth carrying: a subject
missing from a report reads as one the child does not take.

Figures pass through unchanged (decision 5). Points are never divided into a percentage, a
percentage is never rescaled, nothing is summed and no lowest mark is dropped, so every
grade this service stores is one a teacher can find in the sheet they uploaded. A row may
state both a percentage and a points pair; if the two disagree, both are kept and neither
is corrected, because correcting one means choosing which figure the school meant.
"""
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final

from sis.application.dto import ParsedGradeRow, ParseResult, RowCode, RowOutcome
from sis.domain.errors import InvalidPercentage, UnreadableImportFile, ValidationError
from sis.domain.value_objects import Percentage, StudentNumber, SubjectCode, TermCode
from sis.infrastructure.parsers.columns import (
    MAX_POINTS,
    PERCENTAGE,
    POINTS,
    STUDENT_NUMBER,
    SUBJECT_CODE,
    TERM_CODE,
    ColumnMap,
    ColumnSpec,
    map_columns,
)
from sis.infrastructure.parsers.workbook import MAX_ROWS, Sheet, load_sheet

__all__ = ["GRADE_COLUMNS", "SpreadsheetGradeParser"]

# A subject is required of the *file* only when the upload does not name one; see the
# parser's docstring. `replace` keeps the alias list in `columns` as the single place
# header spellings live — a second copy of them here is a second thing to keep in step.
_REQUIRED_SUBJECT_CODE: Final[ColumnSpec] = replace(SUBJECT_CODE, required=True)

#: The columns a marks file may carry. Which of them must be present depends on the
#: upload, so the specs are assembled per parse rather than fixed here.
GRADE_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    STUDENT_NUMBER,
    SUBJECT_CODE,
    TERM_CODE,
    PERCENTAGE,
    POINTS,
    MAX_POINTS,
)

# Spellings of "no mark was given" for the raw *points* column. This mirrors the private
# `_ABSENT_TOKENS` of `sis.domain.value_objects`, which `Percentage.parse_optional` applies
# to the percentage column and which cannot be reused here because points are not a
# percentage and may exceed 100. Duplicated deliberately rather than approximated: were the
# two lists to drift, one column would read "غ" as unmarked while the other rejected the
# row, and the same file would half-import depending on which column the school filled in.
# Should the domain ever make its set public, delete this and import it.
_UNMARKED_TOKENS: Final[frozenset[str]] = frozenset(
    {"", "-", "--", "—", "n/a", "na", "none", "null", "nil", "غ", "غير مرصود", "لم يرصد"}
)


@dataclass(frozen=True, slots=True)
class SpreadsheetGradeParser:
    """Turns a CSV or XLSX marks sheet into `ParsedGradeRow`s plus per-line diagnostics.

    The two defaults exist because `ParsedGradeRow.subject_code` is not optional and a
    school uploading one sheet per subject names that subject in the request, not in a
    column repeated 600 times. Without them such a file would parse into 600 diagnostics
    reading "no subject" — a correct file rejected for lacking redundancy. A subject or
    term named in the row always wins over the default, so a whole-school export that does
    carry the columns is read the way it was written.

    Frozen: an instance is a configuration, not somewhere to accumulate state between
    uploads. A parser that remembered anything about the last file would make the same
    bytes parse differently on the second call, and a preview would stop predicting its
    own commit.
    """

    default_subject_code: SubjectCode | None = None
    default_term_code: TermCode | None = None

    def parse(self, content: bytes, filename: str) -> ParseResult[ParsedGradeRow]:
        """Read `content`; see `GradeFileParser.parse` for the errors this may raise."""
        sheet = load_sheet(content, filename)
        columns = map_columns(sheet.headers, self._specs())
        columns.require()
        _require_a_marks_column(columns)
        _require_a_faithful_read(sheet, columns)

        rows: list[ParsedGradeRow] = []
        diagnostics: list[RowOutcome] = []
        for row in sheet.rows:  # blank lines were skipped by the reader, not counted here
            outcome = self._parse_row(row.line, row.cells, columns)
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

    def _specs(self) -> tuple[ColumnSpec, ...]:
        """The columns this particular upload must and may have."""
        subject = SUBJECT_CODE if self.default_subject_code is not None else _REQUIRED_SUBJECT_CODE
        return (STUDENT_NUMBER, subject, TERM_CODE, PERCENTAGE, POINTS, MAX_POINTS)

    def _parse_row(
        self, line: int, cells: Mapping[str, object], columns: ColumnMap
    ) -> ParsedGradeRow | RowOutcome:
        """One data line: the mark it states, or the reason it states none usable."""
        raw_number = columns.value(cells, STUDENT_NUMBER.field)
        payload: dict[str, object] = {STUDENT_NUMBER.field: _text(raw_number)}
        try:
            student_number = StudentNumber(raw_number)  # type: ignore[arg-type]  # takes cells
        except ValidationError as error:
            return RowOutcome.from_error(
                line, RowCode.MISSING_STUDENT_NUMBER, error, payload=payload
            )

        subject_code = self.default_subject_code
        raw_subject = columns.value(cells, SUBJECT_CODE.field)
        if raw_subject is not None:
            try:
                subject_code = SubjectCode(raw_subject)  # type: ignore[arg-type]  # takes cells
            except ValidationError as error:
                return RowOutcome.from_error(
                    line, RowCode.UNKNOWN_SUBJECT, error, payload=payload
                )
        elif subject_code is None:
            # Only reachable when the file has a subject column and this row left it
            # empty: a missing column was refused at the header, above.
            return RowOutcome(
                line=line,
                code=RowCode.UNKNOWN_SUBJECT,
                message="this row names no subject and the upload names none either",
                payload=payload,
                field=SUBJECT_CODE.field,
            )
        payload[SUBJECT_CODE.field] = str(subject_code)

        term_code = self.default_term_code
        raw_term = columns.value(cells, TERM_CODE.field)
        if raw_term is not None:
            try:
                term_code = TermCode(raw_term)  # type: ignore[arg-type]  # takes cells
            except ValidationError as error:
                return RowOutcome.from_error(line, RowCode.UNKNOWN_TERM, error, payload=payload)

        marks = _parse_marks(line, cells, columns, payload)
        if isinstance(marks, RowOutcome):
            return marks
        percentage, points, max_points = marks

        return ParsedGradeRow(
            line_number=line,
            student_number=student_number,
            subject_code=subject_code,
            term_code=term_code,
            percentage=percentage,
            points=points,
            max_points=max_points,
        )


def _parse_marks(
    line: int, cells: Mapping[str, object], columns: ColumnMap, payload: Mapping[str, object]
) -> tuple[Percentage | None, float | None, float | None] | RowOutcome:
    """The figures this row states, with every absent one as `None`.

    Stating no figure at all is not a failure and does not return an outcome: the caller
    builds a row whose `is_graded` is False. That is how "not marked yet" is recorded, and
    it is the whole of decision 1 at the point where a cell becomes a value.
    """
    percentage: Percentage | None = None
    if columns.has(PERCENTAGE.field):
        raw = columns.number(cells, PERCENTAGE.field)
        try:
            # The only conversion permitted for this column. A blank cell, or one holding
            # "-" or "غ", returns None — not graded yet — and never 0.0 (decision 1).
            percentage = Percentage.parse_optional(raw)
        except InvalidPercentage as error:
            # A figure the domain refused is out of range; anything else is not a figure.
            # The split is worth making: 105 is a teacher marking out of 120, which is a
            # conversation about the scale, while "abs" is a cell to retype.
            code = (
                RowCode.GRADE_OUT_OF_RANGE
                if _as_number(raw) is not None
                else RowCode.INVALID_GRADE
            )
            return _bad_mark(line, code, error.message, payload, PERCENTAGE.field, raw)

    if not columns.has(POINTS.field):
        return percentage, None, None

    raw_points = columns.number(cells, POINTS.field)
    raw_max = columns.number(cells, MAX_POINTS.field)
    max_points = _as_number(raw_max)
    if max_points is None and not _is_unmarked(raw_max):
        return _bad_mark(
            line,
            RowCode.INVALID_GRADE,
            f"{_text(raw_max)!r} is not a maximum mark",
            payload,
            MAX_POINTS.field,
            raw_max,
        )
    if max_points is not None and max_points <= 0:
        return _bad_mark(
            line,
            RowCode.INVALID_GRADE,
            "the maximum mark must be greater than zero",
            payload,
            MAX_POINTS.field,
            raw_max,
        )

    if _is_unmarked(raw_points):
        # Blank points: nobody has marked this yet. The maximum is kept because it is the
        # scale the sheet states rather than a mark — "out of 20, unmarked" is the honest
        # record, and `is_graded` still reads False.
        return percentage, None, max_points

    points = _as_number(raw_points)
    if points is None:
        return _bad_mark(
            line,
            RowCode.INVALID_GRADE,
            f"{_text(raw_points)!r} is not a mark",
            payload,
            POINTS.field,
            raw_points,
        )
    if max_points is None:
        return _bad_mark(
            line,
            RowCode.INVALID_GRADE,
            "a mark in points needs the maximum it was marked out of",
            payload,
            MAX_POINTS.field,
            raw_max,
        )
    if points < 0:
        return _bad_mark(
            line,
            RowCode.GRADE_OUT_OF_RANGE,
            "a mark cannot be negative",
            payload,
            POINTS.field,
            raw_points,
        )
    if points > max_points:
        return _bad_mark(
            line,
            RowCode.GRADE_OUT_OF_RANGE,
            f"{points:g} is more than the stated maximum of {max_points:g}",
            payload,
            POINTS.field,
            raw_points,
        )
    return percentage, points, max_points


def _require_a_marks_column(columns: ColumnMap) -> None:
    """Refuse a file that states no marks, and points with nothing to measure them against.

    A bare `18` is not a mark: out of 20 it is good work and out of 100 it is a failure.
    Choosing a denominator would mean deriving a figure the school never stated (decision
    5), so the file is refused while the registrar can still add the column.
    """
    found = ", ".join(repr(header) for header in columns.headers) or "none at all"
    if not columns.has(PERCENTAGE.field) and not columns.has(POINTS.field):
        accepted = ", ".join(
            repr(alias) for spec in (PERCENTAGE, POINTS) for alias in spec.aliases
        )
        raise UnreadableImportFile(
            f"no marks column was found (accepted: {accepted}). "
            f"The columns in this file are: {found}.",
            field="headers",
        )
    if columns.has(POINTS.field) and not columns.has(MAX_POINTS.field):
        accepted = ", ".join(repr(alias) for alias in MAX_POINTS.aliases)
        raise UnreadableImportFile(
            f"a points column needs the maximum beside it (accepted: {accepted}), "
            f"because a mark out of 20 and a mark out of 100 are not the same mark. "
            f"The columns in this file are: {found}.",
            field="headers",
        )


def _require_a_faithful_read(sheet: Sheet, columns: ColumnMap) -> None:
    """Refuse a file this parser has demonstrably not read in full. Mirrors `roster.py`.

    Either case would otherwise import cleanly and report success over an incomplete
    result — a registrar has no way to notice that a quarter of a year group never
    arrived, or that the second column called "Grade" was the one holding the marks.
    """
    if sheet.truncated:
        raise UnreadableImportFile(
            f"this file holds more than {MAX_ROWS:,} rows and the read stopped there; "
            f"split it and upload the parts. Importing what was read would write part of "
            f"a year group's marks and call it done.",
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


def _bad_mark(
    line: int,
    code: RowCode,
    message: str,
    payload: Mapping[str, object],
    field: str,
    raw: object,
) -> RowOutcome:
    """A rejected marks cell, carrying the offending value so the UI can highlight it."""
    return RowOutcome(
        line=line,
        code=code,
        message=message,
        payload={**payload, "value": _text(raw)},
        field=field,
    )


def _is_unmarked(raw: object) -> bool:
    """True when a marks cell says no mark was given — blank, or one of the tokens.

    Note what this does *not* treat as unmarked: `0`. A zero is a mark a child earned, and
    reading it as "not graded" hides a failing grade exactly as surely as the reverse
    invents one.
    """
    return raw is None or (isinstance(raw, str) and raw.strip().casefold() in _UNMARKED_TOKENS)


def _as_number(raw: object) -> float | None:
    """The finite number a cell states, or `None` when it states no number.

    `None` here means "not a number", not "not marked" — callers ask `_is_unmarked` first.
    `bool` is excluded because it is a subclass of `int`, and `True` would otherwise arrive
    as a mark of 1.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        number = float(raw)
        return number if math.isfinite(number) else None
    if isinstance(raw, str):
        # A trailing '%' is what a human types into a marks column and the figure is the
        # same without it. Nothing else is rewritten here: `columns.number` has already
        # dealt with Arabic-Indic digits, and a comma is left to fail loudly rather than
        # be read as a decimal point in one country and a thousands separator in another.
        text = raw.strip().removesuffix("%").strip()
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _text(raw: object) -> str:
    """A cell as text for a diagnostic payload; an absent cell as `''`.

    Integral floats go through `int`: a column of digits arrives from openpyxl as a number,
    and quoting `12345.0` back at a registrar who typed `12345` sends her looking for a
    decimal point that is not in her file.
    """
    if raw is None:
        return ""
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    return str(raw).strip()
