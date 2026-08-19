"""What a parser hands back: the rows it understood, and the lines it could not.

Both halves travel together because a partially readable file is the normal case, not an
error case. A school roster arrives with a merged header cell, a child whose number was
typed as a date, and 298 perfectly good rows; a parser that raised on line 14 would cost
the registrar all 298. `rows` carries what survived, `diagnostics` carries one
`RowOutcome` per line that did not, and the import service reports both.

The generic parameter is the row shape rather than a common base class so that a service
receives `ParseResult[ParsedGradeRow]` and can reach `row.percentage` without a cast. The
two row types share no fields worth abstracting: a roster row asserts a placement, a grade
row asserts a figure.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field

from sis.application.dto.common import RowCode, RowOutcome


@dataclass(frozen=True, slots=True)
class ParseResult[RowT]:
    """Rows understood, plus a diagnostic for every line that was not.

    `total_lines` counts data lines the reader saw, excluding the header and excluding
    blank lines it skipped. It is not `len(rows) + len(diagnostics)` in general — it is
    recorded separately so "the file had 300 rows and we understood 298" can be stated to
    the registrar without inferring it, and so a silently truncated read (a row cap hit)
    is visible as a number that does not add up rather than as a short list nobody
    questions.
    """

    rows: Sequence[RowT] = field(default_factory=tuple)
    diagnostics: Sequence[RowOutcome] = field(default_factory=tuple)
    total_lines: int = 0
    #: Header names as they actually appeared, kept so a "missing column" message can
    #: quote what the file did contain. A registrar fixing a spreadsheet needs to see
    #: that their column is called "Student ID" and not "student number".
    headers: Sequence[str] = field(default_factory=tuple)

    @property
    def understood(self) -> int:
        return len(self.rows)

    @property
    def rejected(self) -> int:
        return len(self.diagnostics)

    @property
    def is_empty(self) -> bool:
        """True when nothing was understood.

        Distinct from a file that failed to open, which raises instead. An empty result
        means the file was read and had no usable rows — a template someone downloaded
        and uploaded without filling in, most often.
        """
        return not self.rows

    def with_diagnostic(self, line_number: int, code: RowCode, message: str) -> "ParseResult[RowT]":
        """Return a copy with one more rejected line. Rows are never dropped by this."""
        return ParseResult(
            rows=self.rows,
            # `line`, not `line_number`: `RowOutcome` names the Excel gutter number `line`.
            # The two spellings coexisted across parallel modules and this call was the
            # one that raised `TypeError` the first time a parser rejected a line.
            diagnostics=(*self.diagnostics, RowOutcome(line=line_number, code=code, message=message)),
            total_lines=self.total_lines,
            headers=self.headers,
        )
