"""Turning an uploaded file into lines of named cells. Two formats, and no guessing.

`.xlsx` is read with openpyxl and `.csv` with the standard library. **Every other
extension is refused**, and the refusal is a design decision rather than a gap waiting
for a library.

PDF and DOCX are deliberately unsupported. Extracting a table from either means
inferring where the columns are — from x-coordinates on a page, or from a run of tab
stops — and the inference is right most of the time. For roster data, *most of the time*
is the failure: a column boundary read one glyph wide attaches the wrong child to the
wrong class, and nothing downstream can tell, because a mis-parsed row is a perfectly
well-formed row. The registrar sees 300 accepted rows and no error. Grades then follow
the enrolment onto the wrong report card, and the first person to notice is a parent.
A file this service cannot read *by name* is refused so the school converts it once,
visibly, instead of this service quietly inventing a layout. `.xls`, the pre-2007 binary
format, is refused for the same reason its own error says so: it is not this format.

The module reads bytes and returns values. It opens no session, touches no disk and
reads no clock, so an importer built on it is unit-testable with a literal `b"..."`.

Only the first worksheet is read. A school workbook's later sheets are lookup tables,
last year's copy, or the teacher's working area, and importing them because they exist
is how last year's roster gets committed over this year's.

Caps exist because a 40 KB xlsx is a zip archive that can inflate to gigabytes, and
because `SIS_MAX_UPLOAD_BYTES` bounds the *upload*, not what it expands into. Row and
column caps truncate — the shortfall is visible in `total_lines` and `truncated`, which
is a number the registrar can question. The text-volume caps refuse the whole file,
because a cell holding a megabyte of text is not a roster and truncating it would store
a silently mangled child's name.
"""
import codecs
import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Final

import openpyxl

from sis.domain.errors import ImportFailure, UnreadableImportFile, UnsupportedFileType

__all__ = [
    "MAX_CELL_CHARS",
    "MAX_COLUMNS",
    "MAX_ROWS",
    "MAX_TOTAL_CHARS",
    "SUPPORTED_EXTENSIONS",
    "Sheet",
    "SheetRow",
    "load_sheet",
]

# A whole school's roster is a few thousand lines; a per-subject grade sheet is a few
# hundred. The cap is an order of magnitude above the real files so that hitting it
# means "this is not a roster", not "this school is large".
MAX_ROWS: Final[int] = 25_000

# Beyond this, columns are not read. A required column parked past column 64 shows up as
# a missing-header diagnostic quoting the headers that *were* found, which is a message
# a registrar can act on.
MAX_COLUMNS: Final[int] = 64

# No name, code, class or mark is this long. A cell that is refuses the file.
MAX_CELL_CHARS: Final[int] = 4_096

# The binding memory bound. Per-cell and per-row caps still allow their product to be
# enormous, so the total volume of text is what is actually counted.
MAX_TOTAL_CHARS: Final[int] = 4_000_000

SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = (".xlsx", ".csv")

# Delimiters offered to the sniffer. Semicolon is first-class: Excel writes it instead of
# a comma under Arabic and European locales, and a semicolon file read as comma-separated
# parses as one column named after the whole header line.
_CSV_DELIMITERS: Final[str] = ",;\t|"

_SNIFF_BYTES: Final[int] = 8_192


@dataclass(frozen=True, slots=True)
class SheetRow:
    """One data line: its number as the registrar's spreadsheet shows it, and its cells.

    `line` counts from 1 and includes the header row, matching Excel's own gutter and
    `RowOutcome.line`. A zero-based data index here is the classic support call where
    "row 12 is wrong" points at row 13 on her screen and she edits another child's marks.
    """

    line: int
    #: Keyed by the header text exactly as it appeared. Normalising the key here would
    #: leave `columns.py` unable to quote the file's real spelling back to a human.
    #: Every row carries every header, `None` where the line was short.
    cells: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Sheet:
    """What one uploaded file turned out to contain."""

    headers: tuple[str, ...] = ()
    rows: tuple[SheetRow, ...] = ()
    #: Data lines seen, excluding the header and excluding blank lines that were skipped.
    total_lines: int = 0
    #: True when a cap stopped the read. The caller must say so; a short list nobody
    #: questions is how half a year group silently fails to arrive.
    truncated: bool = False
    #: Repeated header text. The first occurrence wins, and the caller is told, because
    #: two columns called "Grade" mean one of them is being ignored.
    duplicate_headers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.rows


def load_sheet(content: bytes, filename: str) -> Sheet:
    """Read `content` — named `filename` — into lines of named cells.

    Raises `UnsupportedFileType` for any extension but `.xlsx` and `.csv`, and
    `UnreadableImportFile` if the file cannot be read or breaches a volume cap. Nothing
    else: a row this cannot make sense of is the importer's problem, not this module's.
    """
    extension = _extension(filename)
    if extension == ".xlsx":
        raw_rows, truncated = _read_xlsx(content)
    elif extension == ".csv":
        raw_rows, truncated = _read_csv(content)
    else:
        raise UnsupportedFileType(
            f"{filename!r} is a {extension or 'nameless'} file; this service reads "
            f"{' and '.join(SUPPORTED_EXTENSIONS)} only. Save the sheet as .xlsx or "
            f"export it as CSV — a PDF or Word table cannot be read without guessing "
            f"where its columns are, and a guessed column attaches a child to the "
            f"wrong class.",
            field="filename",
        )
    return _assemble(raw_rows, truncated)


def _extension(filename: str) -> str:
    """Lowercased suffix, tolerant of a browser sending a full Windows path."""
    return PurePosixPath(filename.replace("\\", "/").strip()).suffix.lower()


# ---------------------------------------------------------------------------
# Readers. Each returns (line number, cell values) pairs plus a truncation flag.
# ---------------------------------------------------------------------------


def _read_xlsx(content: bytes) -> tuple[list[tuple[int, list[object]]], bool]:
    """Stream the first worksheet, bounded by the row and column caps."""
    try:
        book = openpyxl.load_workbook(
            io.BytesIO(content), read_only=True, data_only=True
        )
    except Exception as exc:  # openpyxl raises a zoo of types for a bad archive
        raise UnreadableImportFile(
            "this .xlsx file could not be opened; it may be truncated, "
            "password-protected, or saved in another format under an .xlsx name",
            field="file",
        ) from exc

    try:
        if not book.worksheets:
            raise UnreadableImportFile("the workbook contains no sheets", field="file")
        sheet = book.worksheets[0]
        # The dimension recorded in the file is written by whatever produced it, and a
        # crafted one declares 16384 columns so that every row read is padded to that
        # width. Recomputing from the cells actually present is what keeps the read
        # proportional to the data rather than to a number an attacker chose.
        if hasattr(sheet, "reset_dimensions"):
            sheet.reset_dimensions()

        rows: list[tuple[int, list[object]]] = []
        truncated = False
        for index, values in enumerate(
            sheet.iter_rows(max_col=MAX_COLUMNS, values_only=True), start=1
        ):
            if len(rows) > MAX_ROWS:  # +1 for the header line
                truncated = True
                break
            rows.append((index, _trimmed(values)))
        return rows, truncated
    except ImportFailure:
        raise
    except Exception as exc:
        raise UnreadableImportFile(
            "this .xlsx file could not be read past its first rows", field="file"
        ) from exc
    finally:
        # read_only mode holds the archive open. Without this the handle survives until
        # the garbage collector runs, which under load is long enough to matter.
        book.close()


def _read_csv(content: bytes) -> tuple[list[tuple[int, list[object]]], bool]:
    """Decode, detect the delimiter, and read with the stdlib reader."""
    text = _decode(content)
    reader = csv.reader(
        io.StringIO(text, newline=""), delimiter=_sniff_delimiter(text[:_SNIFF_BYTES])
    )
    rows: list[tuple[int, list[object]]] = []
    truncated = False
    try:
        for values in reader:
            if len(rows) > MAX_ROWS:
                truncated = True
                break
            # `reader.line_num` rather than a counter: a quoted cell containing a
            # newline spans two physical lines, and a counter would report every row
            # after it against the wrong line of the file.
            rows.append((reader.line_num, _trimmed(values)))
    except csv.Error as exc:
        raise UnreadableImportFile(
            f"this CSV file is malformed at line {reader.line_num}", field="file"
        ) from exc
    return rows, truncated


def _trimmed(values: Sequence[object]) -> list[object]:
    """Cap the width and drop trailing empties left by a sheet's unused columns."""
    row = list(values[:MAX_COLUMNS])
    while row and (row[-1] is None or row[-1] == ""):
        row.pop()
    return row


def _decode(content: bytes) -> str:
    """Decode a CSV, refusing rather than mojibaking when no encoding fits.

    The ladder is UTF-16 by byte-order mark (Excel's "Unicode text" export), then UTF-8,
    then cp1256 — which is what Excel on an Arabic Windows writes and the only reason a
    legacy guess is made at all. cp1256 is last because it decodes *any* byte sequence,
    so putting it earlier would turn a UTF-8 Arabic name into confident nonsense. When
    the guess is wrong the damage is visible: the registrar sees mangled names in the
    preview and fixes the export, which is exactly what preview-then-commit is for.
    """
    if content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return content.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise UnreadableImportFile(
                "this file begins as UTF-16 but is not valid UTF-16", field="file"
            ) from exc
    for encoding in ("utf-8-sig", "cp1256"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnreadableImportFile(
        "the text encoding of this CSV could not be determined; re-export it as "
        "UTF-8 so Arabic names survive",
        field="file",
    )


def _sniff_delimiter(sample: str) -> str:
    """Sniff the separator, falling back to a comma when the sample is inconclusive.

    A single-column file gives the sniffer nothing to work with and it raises; that file
    parses identically under any delimiter, so the fallback is not a guess.
    """
    try:
        return csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS).delimiter
    except csv.Error:
        return ","


# ---------------------------------------------------------------------------
# Assembly. Format-independent, so both readers get identical row semantics.
# ---------------------------------------------------------------------------


def _assemble(raw_rows: list[tuple[int, list[object]]], truncated: bool) -> Sheet:
    """Take the header line, then key every later line by it."""
    budget = _TextBudget()
    header_index = _first_populated(raw_rows)
    if header_index is None:
        return Sheet(truncated=truncated)

    columns, duplicates = _header_columns(raw_rows[header_index][1], budget)
    headers = tuple(header for _, header in columns)

    rows: list[SheetRow] = []
    total_lines = 0
    for line, values in raw_rows[header_index + 1 :]:
        cells = {
            header: _clean(values[position] if position < len(values) else None, budget)
            for position, header in columns
        }
        # A blank line is skipped, not counted and not rejected: it is the empty row a
        # spreadsheet leaves under the data, and reporting it as a bad row would put a
        # failure in front of a registrar who has nothing to fix.
        if all(value is None for value in cells.values()):
            continue
        total_lines += 1
        rows.append(SheetRow(line=line, cells=cells))

    return Sheet(
        headers=headers,
        rows=tuple(rows),
        total_lines=total_lines,
        truncated=truncated,
        duplicate_headers=duplicates,
    )


def _first_populated(raw_rows: list[tuple[int, list[object]]]) -> int | None:
    """Index of the first line with any content. That line is the header, unconditionally.

    No attempt is made to find the row that "looks like" headers. Scoring candidate rows
    is the same guess this module refuses everywhere else, and it fails on the one file
    where it matters — a sheet whose title banner happens to hold plausible words. When a
    file does carry a banner above its headers, the banner becomes the header row and the
    missing-column diagnostic quotes it back, so the registrar sees precisely what was
    read instead of wondering why 300 rows vanished.
    """
    for index, (_, values) in enumerate(raw_rows):
        if any(value is not None and str(value).strip() for value in values):
            return index
    return None


def _header_columns(
    values: list[object], budget: "_TextBudget"
) -> tuple[tuple[tuple[int, str], ...], tuple[str, ...]]:
    """Pair each named column with its position; first spelling of a repeat wins."""
    columns: list[tuple[int, str]] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for position, value in enumerate(values):
        header = _clean(value, budget)
        # An unnamed column is dropped rather than given a positional name. Anything that
        # then read it would be reading a column by where it sits, which is the one thing
        # this package will not do.
        if header is None:
            continue
        text = str(header)
        if text in seen:
            duplicates.append(text)
            continue
        seen.add(text)
        columns.append((position, text))
    return tuple(columns), tuple(duplicates)


def _clean(value: object, budget: "_TextBudget") -> object:
    """Strip text, treat an emptied cell as absent, and charge it to the text budget.

    Blank becomes `None`, never `""` and never `0.0`: an empty mark cell means "not
    graded yet", and every layer above depends on that distinction (decision 1). Native
    types are otherwise passed through untouched, because openpyxl already gives numbers
    and dates as numbers and dates, and stringifying them here would force every caller
    to parse them back.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    if len(value) > MAX_CELL_CHARS:
        raise UnreadableImportFile(
            f"a cell in this file holds {len(value)} characters; no name, code or mark "
            f"is longer than {MAX_CELL_CHARS}, so this is not a roster",
            field="file",
        )
    budget.spend(len(value))
    text = value.strip()
    return text or None


class _TextBudget:
    """Counts characters read so a file cannot expand into memory unnoticed."""

    __slots__ = ("_spent",)

    def __init__(self) -> None:
        self._spent = 0

    def spend(self, characters: int) -> None:
        self._spent += characters
        if self._spent > MAX_TOTAL_CHARS:
            raise UnreadableImportFile(
                f"this file expands to more than {MAX_TOTAL_CHARS} characters of text; "
                f"split it, or check that it is the file you meant to upload",
                field="file",
            )
