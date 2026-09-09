"""File readers: bytes in, named cells out.

The two halves are kept apart because they fail differently. `workbook` decides whether a
file can be read at all — a wrong extension or a crafted archive is a whole-file refusal.
`columns` decides what its headers mean — a wrong header is a diagnostic naming what was
found, so the registrar can rename a column instead of guessing.

Neither half validates anything about a school. Resolving a class code, checking a term is
open and refusing a duplicate enrolment all belong to the services, after parsing, where
the outcome is visible as a row the registrar can read.
"""
from sis.infrastructure.parsers.grades import GRADE_COLUMNS, SpreadsheetGradeParser
from sis.infrastructure.parsers.guardians import (
    GUARDIAN_COLUMNS,
    SpreadsheetGuardianParser,
)
from sis.infrastructure.parsers.roster import (
    ROSTER_COLUMNS,
    SpreadsheetFamilyRosterParser,
    SpreadsheetRosterParser,
)
from sis.infrastructure.parsers.columns import (
    ACADEMIC_YEAR_CODE,
    CAN_VIEW_RECORDS,
    CLASS_CODE,
    ENDS_ON,
    FULL_NAME_AR,
    FULL_NAME_EN,
    GUARDIAN_ALT_PHONE,
    GUARDIAN_NAME_AR,
    GUARDIAN_NAME_EN,
    GUARDIAN_PHONE,
    IS_PRIMARY_CONTACT,
    MAX_POINTS,
    PERCENTAGE,
    POINTS,
    RELATIONSHIP_LABEL,
    RELATIONSHIP_TYPE,
    RESTRICTION_NOTE,
    STARTS_ON,
    STUDENT_NUMBER,
    SUBJECT_CODE,
    TERM_CODE,
    YEAR_LEVEL_CODE,
    ColumnMap,
    ColumnSpec,
    map_columns,
    normalise_digits,
    normalise_header,
)
from sis.infrastructure.parsers.workbook import (
    MAX_CELL_CHARS,
    MAX_COLUMNS,
    MAX_ROWS,
    MAX_TOTAL_CHARS,
    SUPPORTED_EXTENSIONS,
    Sheet,
    SheetRow,
    load_sheet,
)

__all__ = [
    "ACADEMIC_YEAR_CODE",
    "CAN_VIEW_RECORDS",
    "CLASS_CODE",
    "GRADE_COLUMNS",
    "GUARDIAN_ALT_PHONE",
    "GUARDIAN_COLUMNS",
    "GUARDIAN_NAME_AR",
    "GUARDIAN_NAME_EN",
    "GUARDIAN_PHONE",
    "IS_PRIMARY_CONTACT",
    "RELATIONSHIP_LABEL",
    "RELATIONSHIP_TYPE",
    "RESTRICTION_NOTE",
    "ROSTER_COLUMNS",
    "SpreadsheetGradeParser",
    "SpreadsheetGuardianParser",
    "SpreadsheetRosterParser",
    "SpreadsheetFamilyRosterParser",
    "ColumnMap",
    "ColumnSpec",
    "ENDS_ON",
    "FULL_NAME_AR",
    "FULL_NAME_EN",
    "MAX_CELL_CHARS",
    "MAX_COLUMNS",
    "MAX_POINTS",
    "MAX_ROWS",
    "MAX_TOTAL_CHARS",
    "PERCENTAGE",
    "POINTS",
    "SUPPORTED_EXTENSIONS",
    "STARTS_ON",
    "STUDENT_NUMBER",
    "SUBJECT_CODE",
    "Sheet",
    "SheetRow",
    "TERM_CODE",
    "YEAR_LEVEL_CODE",
    "load_sheet",
    "map_columns",
    "normalise_digits",
    "normalise_header",
]
