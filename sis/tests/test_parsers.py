"""Parser tests written against files, never against the parsers' internals.

Every case here starts as bytes — an .xlsx assembled in memory with openpyxl, a .csv
encoded from text — because bytes are the only input the import boundary ever receives.
A test that handed `_parse_row` a ready-made dict of cells would keep passing while the
real path broke on the header row, and the header row is where school files actually go
wrong.

Two invariants carry the weight here, and each has a test that fails loudly the moment an
implementation becomes "helpful" about it:

* **A blank marks cell is `None`, and `0` is a mark.** The two must never converge, in
  either direction: a defaulted zero invents a failing grade for a child nobody has
  marked, and a zero read as "unmarked" hides a failing grade a teacher did give.
* **A column is found by its header text, never by its position.** The reordering test
  puts a class code in the first column on purpose. Position-based reading would still
  produce well-formed rows there — every value is a valid code — so nothing downstream
  could notice; only an assertion on which field got which value catches it.
"""
import io
from collections.abc import Callable, Sequence

import openpyxl
import pytest

from sis.application.dto import RowCode
from sis.domain.errors import UnreadableImportFile, UnsupportedFileType
from sis.domain.value_objects import Percentage, SubjectCode, TermCode
from sis.domain.guardians import RelationshipType
from sis.infrastructure.parsers import (
    ROSTER_COLUMNS,
    SpreadsheetGradeParser,
    SpreadsheetGuardianParser,
    SpreadsheetRosterParser,
    load_sheet,
    map_columns,
)

Rows = Sequence[Sequence[object]]

ROSTER = SpreadsheetRosterParser()

#: The subject and term a school states in the request when it uploads one sheet per
#: subject. Supplied here so a test file may carry marks and nothing else.
MARKS = SpreadsheetGradeParser(
    default_subject_code=SubjectCode("MATH"), default_term_code=TermCode("2026-T1")
)

#: Egypt, pinned rather than taken from the environment: these tests assert exact E.164
#: output, and a deployment default leaking in would make them pass or fail by locale.
GUARDIANS = SpreadsheetGuardianParser(default_country_code="+20")


def as_xlsx(rows: Rows) -> bytes:
    """A real .xlsx in memory.

    Written with openpyxl and read back through openpyxl on purpose: the round trip is
    what makes a cell this test writes as `85` arrive as the *number* 85 rather than the
    string "85", which is the difference the parser has to cope with in a school's file
    and would be papered over by a hand-built list of cells.
    """
    book = openpyxl.Workbook()
    sheet = book.active
    for row in rows:
        sheet.append(list(row))
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def as_csv(rows: Rows, *, encoding: str = "utf-8") -> bytes:
    """The literal bytes of a CSV. No quoting: no value below contains a delimiter."""
    lines = [",".join("" if cell is None else str(cell) for cell in row) for row in rows]
    return ("\r\n".join(lines) + "\r\n").encode(encoding)


FILE_FORMATS = [
    pytest.param(as_xlsx, "roster.xlsx", id="xlsx"),
    pytest.param(as_csv, "roster.csv", id="csv"),
]


# ---------------------------------------------------------------------------
# Headers: found by name, in either language, in any order and any spelling.
# ---------------------------------------------------------------------------


def test_english_headers_are_read_with_the_line_numbers_the_registrar_sees() -> None:
    content = as_xlsx(
        [
            ("Student Number", "Name (English)", "Name (Arabic)", "Class"),
            ("S001", "Layla Hassan", "ليلى حسن", "3A"),
            ("S002", "Omar Nabil", "عمر نبيل", "3B"),
        ]
    )

    result = ROSTER.parse(content, "roster.xlsx")

    assert result.diagnostics == ()
    assert result.total_lines == 2
    assert [row.line_number for row in result.rows] == [2, 3]
    assert [str(row.student_number) for row in result.rows] == ["S001", "S002"]
    assert [str(row.class_code) for row in result.rows] == ["3A", "3B"]
    assert result.rows[0].full_name_en == "Layla Hassan"
    assert result.rows[0].full_name_ar == "ليلى حسن"


def test_arabic_headers_name_the_same_fields_as_english_ones() -> None:
    content = as_csv(
        [
            ("رقم الطالب", "اسم الطالب", "الفصل"),
            ("S001", "ليلى حسن", "3A"),
        ]
    )

    result = ROSTER.parse(content, "كشف الطلاب.csv")

    assert result.diagnostics == ()
    row = result.rows[0]
    assert str(row.student_number) == "S001"
    assert row.full_name_ar == "ليلى حسن"
    assert row.full_name_en == ""
    assert str(row.class_code) == "3A"


def test_column_order_carries_no_meaning() -> None:
    """The trap: read by position, every field still validates — and every one is wrong.

    `3B` is a well-formed student number and `1001` a well-formed class code, so a
    position-based reader produces 300 accepted rows with the children attached to the
    wrong classes, and nothing further down the stack can tell.
    """
    content = as_xlsx(
        [
            ("Class", "Name (English)", "Student Number"),
            ("3B", "Omar Nabil", "1001"),
        ]
    )

    row = ROSTER.parse(content, "roster.xlsx").rows[0]

    assert str(row.student_number) == "1001"
    assert str(row.class_code) == "3B"
    assert row.full_name_en == "Omar Nabil"


def test_unknown_columns_are_carried_past_untouched() -> None:
    """Schools file rosters with the columns their own office needs. Extras are not errors.

    "الصف" is in here deliberately: it is the year level, a column this service maps in
    other files, and a roster must ignore it rather than mistake it for the class section.
    """
    content = as_csv(
        [
            ("Student Number", "Nationality", "Name (English)", "Notes", "الصف", "Class"),
            ("S001", "EG", "Layla Hassan", "transferred in", "Y3", "3A"),
        ]
    )

    result = ROSTER.parse(content, "roster.csv")

    assert result.diagnostics == ()
    assert str(result.rows[0].student_number) == "S001"
    assert str(result.rows[0].class_code) == "3A"
    columns = map_columns(result.headers, ROSTER_COLUMNS)
    assert columns.unmapped == ("Nationality", "Notes", "الصف")


def test_case_punctuation_and_whitespace_are_not_evidence() -> None:
    """Both halves of a line are messy here: the headers and the cells they name.

    A header is matched and discarded, so folding it is free. A cell is *stored*, so only
    surrounding whitespace goes — except in a code, which the domain normalises to one
    spelling so that "3a" and "3A" cannot become two classes.
    """
    content = as_xlsx(
        [
            ("  STUDENT   NO.  ", "\tenglish_name ", " Class "),
            ("  s001  ", "  Layla Hassan  ", " 3a "),
        ]
    )

    row = ROSTER.parse(content, "roster.xlsx").rows[0]

    assert str(row.student_number) == "S001"
    assert str(row.class_code) == "3A"
    assert row.full_name_en == "Layla Hassan"


# ---------------------------------------------------------------------------
# Arabic-Indic digits. Arithmetic in a marks column, identity in a student number.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("٨٥", 85.0),
        ("٩٩٫٥", 99.5),  # ٫ is the Arabic decimal separator, not a comma
        ("١٠٠", 100.0),
        ("٧٥٪", 75.0),  # ٪ is the Arabic percent sign
        ("۸۵", 85.0),  # extended Arabic-Indic, as a Persian keyboard layout produces
    ],
)
def test_arabic_indic_digits_in_a_marks_column_are_read_as_numbers(
    cell: str, expected: float
) -> None:
    """`float("٨٥")` raises, and a whole column of good marks would read as unparseable."""
    content = as_xlsx([("Student Number", "Percentage"), ("S001", cell)])

    row = MARKS.parse(content, "marks.xlsx").rows[0]

    assert row.percentage == Percentage(expected)


def test_arabic_indic_digits_in_a_student_number_reject_only_that_row() -> None:
    """The asymmetry is deliberate: a mark is arithmetic, a student number is identity.

    Transliterating "١٠٠١" into "1001" would merge two roster rows into one child — or
    split one child's grades across two identities — with no record that a choice was
    made. The row is refused so a human reconciles it, and the rows around it still land.
    """
    content = as_csv(
        [
            ("Student Number", "Name (English)"),
            ("S001", "Layla Hassan"),
            ("١٠٠١", "Omar Nabil"),
            ("S003", "Nour Adel"),
        ]
    )

    result = ROSTER.parse(content, "roster.csv")

    assert [str(row.student_number) for row in result.rows] == ["S001", "S003"]
    assert [(d.line, d.code) for d in result.diagnostics] == [
        (3, RowCode.MISSING_STUDENT_NUMBER)
    ]
    assert result.total_lines == 3


# ---------------------------------------------------------------------------
# Whole-file refusals. Each one has to say what the registrar should go and change.
# ---------------------------------------------------------------------------


def test_a_missing_required_header_quotes_the_headers_that_were_found() -> None:
    """"Not found" without evidence sends a registrar looking for a column she has.

    The message has to carry both halves — the spellings that would have worked and the
    spellings the file actually holds — because the fix is a rename, and she cannot make
    it without seeing that her column is called "Pupil".
    """
    content = as_csv([("Pupil", "Subject", "Mark"), ("S001", "MATH", "85")])

    with pytest.raises(UnreadableImportFile) as raised:
        SpreadsheetGradeParser().parse(content, "marks.csv")

    message = raised.value.message
    assert raised.value.field == "headers"
    assert "'student_number' was not found" in message
    assert "'student number'" in message  # an accepted spelling is offered
    for header in ("'Pupil'", "'Subject'", "'Mark'"):
        assert header in message


def test_a_roster_with_no_name_column_is_refused_with_the_same_evidence() -> None:
    """One of two columns is required, which `ColumnSpec.required` cannot express."""
    content = as_csv([("Student Number", "Class"), ("S001", "3A")])

    with pytest.raises(UnreadableImportFile) as raised:
        ROSTER.parse(content, "roster.csv")

    message = raised.value.message
    assert "no name column was found" in message
    assert "'Student Number', 'Class'" in message


@pytest.mark.parametrize("filename", ["roster.pdf", "roster.docx", "roster.xls", "roster"])
def test_unreadable_formats_are_refused_by_name_before_anything_is_parsed(
    filename: str,
) -> None:
    """Refused on the extension, not on the content — a .pdf holding a table is still a .pdf.

    Extracting a table from a PDF or a Word document means inferring where the columns
    are, and an inference that is right most of the time produces well-formed rows with
    the wrong child in them. Refusing by name makes the school convert the file once,
    visibly.
    """
    with pytest.raises(UnsupportedFileType) as raised:
        load_sheet(b"%PDF-1.7 anything at all", filename)

    assert raised.value.code == "unsupported_file_type"
    assert ".xlsx" in raised.value.message and ".csv" in raised.value.message


def test_the_parsers_refuse_the_same_extensions_their_reader_does() -> None:
    """The refusal must survive the parser's own entry point, not only `load_sheet`."""
    with pytest.raises(UnsupportedFileType):
        ROSTER.parse(b"anything", "roster.docx")
    with pytest.raises(UnsupportedFileType):
        MARKS.parse(b"anything", "marks.pdf")


# ---------------------------------------------------------------------------
# Blank lines and blank cells. Two different kinds of nothing.
# ---------------------------------------------------------------------------


def test_blank_lines_are_skipped_rather_than_rejected() -> None:
    """The empty row a spreadsheet leaves under its data is not something to report.

    A diagnostic for it puts a failure in front of a registrar with nothing to fix, and
    counting it inflates "300 rows read" past what the file holds.
    """
    content = as_xlsx(
        [
            ("Student Number", "Name (English)"),
            ("S001", "Layla Hassan"),
            (None, None),
            ("S002", "Omar Nabil"),
            (None, None),
        ]
    )

    result = ROSTER.parse(content, "roster.xlsx")

    assert result.diagnostics == ()
    assert result.total_lines == 2
    assert [str(row.student_number) for row in result.rows] == ["S001", "S002"]


def test_a_skipped_line_does_not_shift_the_lines_after_it() -> None:
    """Line numbers are the registrar's gutter, so a skipped row must not renumber.

    A counter over surviving rows would report "row 3 is wrong" about what her screen
    shows as row 5, and she edits another child's record.
    """
    content = as_csv(
        [
            ("Student Number", "Name (English)"),
            ("S001", "Layla Hassan"),
            (),
            ("", "", ""),
            ("S002", "Omar Nabil"),
        ]
    )

    result = ROSTER.parse(content, "roster.csv")

    assert [row.line_number for row in result.rows] == [2, 5]


@pytest.mark.parametrize(
    ("build", "filename"),
    [pytest.param(as_xlsx, "marks.xlsx", id="xlsx"), pytest.param(as_csv, "marks.csv", id="csv")],
)
def test_a_blank_mark_is_none_and_a_zero_is_a_mark(
    build: Callable[[Rows], bytes], filename: str
) -> None:
    """Decision 1 at the exact point a cell becomes a value, in both file formats.

    The pair matters more than either half. `0.0` for a blank invents a failing grade in
    a subject nobody has marked; `None` for a stated zero hides one a teacher gave. Both
    are invisible afterwards, because a stored figure looks the same either way.
    """
    content = build(
        [
            ("Student Number", "Percentage"),
            ("S001", None),
            ("S002", 0),
            ("S003", "-"),
            ("S004", "غ"),
            ("S005", 55),
        ]
    )

    result = MARKS.parse(content, filename)

    assert result.diagnostics == ()
    marks = {str(row.student_number): row for row in result.rows}
    assert marks["S001"].percentage is None
    assert marks["S001"].is_graded is False
    assert marks["S002"].percentage == Percentage(0.0)
    assert marks["S002"].is_graded is True
    assert [marks[number].percentage for number in ("S003", "S004")] == [None, None]
    assert marks["S005"].percentage == Percentage(55.0)
    # An unmarked row is still a row: "takes this subject, not yet marked" is a fact, and
    # dropping it would read as a subject the child does not take.
    assert len(result.rows) == 5


def test_an_unmarked_row_keeps_the_scale_its_points_column_states() -> None:
    """"Out of 20, unmarked" is the honest record; the maximum is a scale, not a mark."""
    content = as_xlsx(
        [
            ("Student Number", "Points", "Out Of"),
            ("S001", None, 20),
            ("S002", 0, 20),
        ]
    )

    result = MARKS.parse(content, "marks.xlsx")

    unmarked, zero = result.rows
    assert (unmarked.points, unmarked.max_points, unmarked.is_graded) == (None, 20.0, False)
    assert (zero.points, zero.max_points, zero.is_graded) == (0.0, 20.0, True)


# ---------------------------------------------------------------------------
# Guardians: one row is one pairing, and the phone is the identity.
# ---------------------------------------------------------------------------


def test_a_guardian_sheet_carrying_the_child_s_name_reads_the_guardian_s() -> None:
    """The collision the separate name specs exist to prevent.

    A real guardians sheet carries the *student's* name so the registrar can read the file
    she is uploading. Folding those headers into one spec would file the child's name as
    her mother's, and every row would still look perfectly well-formed.
    """
    content = as_xlsx(
        [
            ("Student Number", "Student Name (Arabic)", "Guardian Name (Arabic)", "Phone"),
            ("S001", "ليلى حسن", "فاطمة علي", "01001234567"),
        ]
    )

    (row,) = GUARDIANS.parse(content, "guardians.xlsx").rows

    assert row.full_name_ar == "فاطمة علي"
    assert str(row.phone) == "+201001234567"


def test_a_phone_typed_as_a_number_by_excel_is_recovered() -> None:
    """Through a real .xlsx, because the round trip is what produces the float."""
    content = as_xlsx(
        [
            ("Student Number", "Guardian Name (English)", "Phone"),
            ("S001", "Fatma Ali", 1001234567),
        ]
    )

    (row,) = GUARDIANS.parse(content, "guardians.xlsx").rows

    assert str(row.phone) == "+201001234567"


@pytest.mark.parametrize(
    ("stated", "expected", "label"),
    [
        ("mother", RelationshipType.MOTHER, ""),
        ("الأم", RelationshipType.MOTHER, ""),
        ("أب", RelationshipType.FATHER, ""),
        ("ولي الأمر", RelationshipType.GUARDIAN, ""),
        # Bucketed *and* kept: closing the vocabulary costs nothing a human typed.
        ("big brother", RelationshipType.SIBLING, "big brother"),
        ("الأخ الأكبر", RelationshipType.SIBLING, "الأخ الأكبر"),
        # "grandmother" contains "mother"; longest-first matching is what stops a
        # grandparent being filed as a parent.
        ("grandmother", RelationshipType.GRANDPARENT, ""),
        # Unrecognised is never a rejection — it degrades and keeps the words.
        ("cousin", RelationshipType.OTHER, "cousin"),
        ("", RelationshipType.OTHER, ""),
    ],
)
def test_a_relationship_is_bucketed_without_losing_what_was_typed(
    stated: str, expected: RelationshipType, label: str
) -> None:
    content = as_csv(
        [
            ("Student Number", "Guardian Name (English)", "Phone", "Relationship"),
            ("S001", "Someone", "01001234567", stated),
        ]
    )

    (row,) = GUARDIANS.parse(content, "guardians.csv").rows

    assert row.relationship_type is expected
    assert row.relationship_label == label


def test_a_relationship_never_rejects_a_row() -> None:
    """Unlike a grade, an unfamiliar word here loses nothing — the label keeps it."""
    content = as_csv(
        [
            ("Student Number", "Guardian Name (English)", "Phone", "Relationship"),
            ("S001", "Someone", "01001234567", "قريب من الدرجة الثانية"),
        ]
    )

    result = GUARDIANS.parse(content, "guardians.csv")

    assert result.diagnostics == ()
    assert len(result.rows) == 1


def test_records_access_is_granted_unless_the_sheet_withholds_it() -> None:
    """The asymmetry with the database default, asserted where it is decided.

    Absent means granted because a registrar reviewing a preview has stated who these
    people are; the column's own default protects code paths that never asked.
    """
    content = as_csv(
        [
            ("Student Number", "Guardian Name (English)", "Phone", "Can View Records"),
            ("S001", "Mother", "01001234567", ""),
            ("S001", "Brother", "01002223333", "no"),
            ("S001", "Father", "01005554444", "yes"),
            # Unrecognised falls back to the default rather than inventing a restriction
            # nobody wrote.
            ("S001", "Aunt", "01007776666", "maybe"),
        ]
    )

    rows = GUARDIANS.parse(content, "guardians.csv").rows

    assert [row.can_view_records for row in rows] == [True, False, True, True]


def test_a_second_number_is_read_and_a_repeat_of_the_first_is_collapsed() -> None:
    """The only way to reach the second phone one adult holds."""
    content = as_csv(
        [
            ("Student Number", "Guardian Name (English)", "Phone", "Alt Phone"),
            ("S001", "Fatma Ali", "01001234567", "01119998888"),
            ("S002", "Hassan", "01002223333", "01002223333"),
        ]
    )

    both, repeated = GUARDIANS.parse(content, "guardians.csv").rows

    assert [str(p) for p in both.phones] == ["+201001234567", "+201119998888"]
    # Stating the same number twice is a filled-in column, not a second way to be reached.
    assert [str(p) for p in repeated.phones] == ["+201002223333"]


def test_a_bad_phone_rejects_only_its_own_row() -> None:
    """Invariant 4 at the parser boundary: 299 good rows survive one mistyped cell."""
    content = as_csv(
        [
            ("Student Number", "Guardian Name (English)", "Phone"),
            ("S001", "Fatma Ali", "01001234567"),
            ("S002", "Unreachable", "call me later"),
            ("S003", "Omar Samir", "01002223333"),
        ]
    )

    result = GUARDIANS.parse(content, "guardians.csv")

    assert len(result.rows) == 2
    (bad,) = result.diagnostics
    assert (bad.line, bad.code) == (3, RowCode.MISSING_PHONE)


def test_a_guardians_file_with_no_phone_column_is_refused_outright() -> None:
    """A whole-file problem, not a per-row one: every row would fail for one reason."""
    content = as_csv(
        [
            ("Student Number", "Guardian Name (English)"),
            ("S001", "Fatma Ali"),
        ]
    )

    with pytest.raises(UnreadableImportFile) as refusal:
        GUARDIANS.parse(content, "guardians.csv")
    assert "phone" in str(refusal.value)


def test_a_guardians_file_with_no_guardian_name_column_is_refused() -> None:
    """Either script alone is fine; neither is not — and the message says which are accepted."""
    content = as_csv(
        [
            ("Student Number", "Phone"),
            ("S001", "01001234567"),
        ]
    )

    with pytest.raises(UnreadableImportFile) as refusal:
        GUARDIANS.parse(content, "guardians.csv")
    assert "guardian name" in str(refusal.value).lower()
