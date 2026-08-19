"""Ports for turning an uploaded file into rows an import service can reason about.

A parser is the only code in this service that knows what a spreadsheet is, and reading
one is deliberately the only thing it does. **It touches no database, no disk and no
network.** It is handed a buffer that is already in memory and returns values: it opens
no session, resolves no code against stored data, reads no clock and writes no temp
file. Two things follow from that, and both are the reason the constraint is written
down rather than merely observed:

* An import service can be unit-tested by handing a fake parser a literal ``b"..."``.
  Without this rule the cheapest test of "one bad row must not discard the good ones"
  needs a real xlsx fixture, a temp directory and a database.
* Parsing is reproducible. A parser that looked up a class code while reading would make
  the same file parse differently on Tuesday, and a preview taken on Monday would then
  be committed against rows the registrar never saw -- the exact failure the two-step
  flow in `sis.domain.imports` exists to remove. Validation against stored structure is
  the service's job, after parsing, where its results are visible as row outcomes.

`filename` travels beside the content instead of a path because it is evidence, not a
location: it is how the parser chooses a CSV or XLSX reader and what `UnsupportedFileType`
quotes back to the registrar. Passing a path would make some caller responsible for
cleaning up a temp file whose only purpose was to be read into the memory it came from.

A malformed row is a *value*, never an exception. Each parser returns the rows it
understood together with one `ImportRow` diagnostic per line it could not, so line 14
being unreadable still leaves lines 1-13 and 15-200 importable. `UnreadableImportFile`
is the single error a parser may raise, and it means the opposite thing: the file could
not be opened at all, so there were never any rows to keep.
"""
from typing import Protocol

from sis.application.dto import (
    ParsedGradeRow,
    ParsedGuardianRow,
    ParsedRosterRow,
    ParseResult,
)


class RosterFileParser(Protocol):
    """Reads a roster upload: who a child is, and the class placement being asserted.

    A parsed roster row carries the placement's own dates or term alongside the class
    code, because placement is a time-bounded membership rather than a column on the
    student (decision 2). A parser that flattened the file to "student -> current class"
    would discard the only information that can answer "which class was she in during
    Term 1" after a child moves 3A->3B in March, and that answer cannot be reconstructed
    later from a single column.
    """

    def parse(self, content: bytes, filename: str) -> ParseResult[ParsedRosterRow]:
        """Parse `content`, whose original name was `filename`.

        Raises `UnsupportedFileType` if no reader claims the extension, and
        `UnreadableImportFile` if the file cannot be opened. Nothing else.
        """
        ...


class GradeFileParser(Protocol):
    """Reads a marks upload: a student, a subject, a term, and a stated figure.

    The rule that shapes every implementation: **a blank cell parses to `None`, never to
    `0.0`** (decision 1). Zero is a mark a child can earn, so a parser that defaults an
    empty cell to zero is not filling in a gap -- it is inventing a failing grade, and
    the invention is undetectable downstream because a stored `0.0` from an empty cell
    is byte-identical to one a teacher typed. Reporting "not graded yet" as 0% tells a
    parent something false about their child.

    Percentages arrive as the figure the file stated. No parser sums, weights, rescales
    or drops a lowest mark (decision 5); a grade this service returns is one a teacher
    can recognise in the sheet they uploaded.
    """

    def parse(self, content: bytes, filename: str) -> ParseResult[ParsedGradeRow]:
        """Parse `content`, whose original name was `filename`.

        Raises `UnsupportedFileType` if no reader claims the extension, and
        `UnreadableImportFile` if the file cannot be opened. Nothing else.
        """
        ...


class GuardianFileParser(Protocol):
    """Reads a guardians upload: a child, an adult responsible for her, and a phone.

    One row is one *pairing*, not one person, so a mother, a father and a big brother for
    one child are three rows sharing a student number. That shape is what lets a child
    have any number of guardians without the file growing a column per parent -- and a
    file with a fixed `mother_phone`/`father_phone` pair is exactly the schema that has
    nowhere to put the grandmother who actually collects her.

    A second number for the *same* adult is the one thing that does not work this way. It
    comes from an alternate-phone column on her own row, because the primary phone is her
    identity: two rows carrying two unfamiliar numbers are two different people by
    definition, and no amount of name-matching can safely say otherwise.

    Implementations normalise a phone to E.164 while parsing, which is the only step here
    that needs to know something the file does not state -- a default country, for the
    numbers a registrar types in national form. It is passed to the implementation's
    constructor rather than read from the environment, so the same bytes parse to the
    same rows wherever this runs.
    """

    def parse(self, content: bytes, filename: str) -> ParseResult[ParsedGuardianRow]:
        """Parse `content`, whose original name was `filename`.

        Raises `UnsupportedFileType` if no reader claims the extension, and
        `UnreadableImportFile` if the file cannot be opened. Nothing else.
        """
        ...
