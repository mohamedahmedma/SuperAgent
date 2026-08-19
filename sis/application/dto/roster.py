"""Roster import: who a child is, and the class placement being asserted for them.

The shape that matters here is that a row carries a *placement*, not a class column. A
child moving 3A->3B in March is two enrolments with adjacent windows, not one field
overwritten — which is the only reason "which class was she in during Term 1" is still
answerable in June (decision 2). A DTO that flattened this to `class_code` on the student
would discard that at the boundary, before any service could preserve it.
"""
from dataclasses import dataclass
from datetime import date

from sis.domain.value_objects import AcademicYearCode, ClassCode, StudentNumber


@dataclass(frozen=True, slots=True)
class ParsedRosterRow:
    """One roster line the parser understood.

    Codes are already value objects: constructing a `StudentNumber` is pure validation
    with no stored data behind it, so a malformed number is caught at parse time and
    becomes a diagnostic rather than travelling as a `str` that fails three layers later.
    Anything requiring the database — does this class exist, is this number already taken
    — is deliberately absent, because a parser that resolved codes against stored data
    would make the same file parse differently on different days.

    `class_code` is optional because a school may upload one file per class (the target
    comes from the request) or one file for the whole school (the target comes from the
    row). Both are real; the service resolves which applies.
    """

    line_number: int
    student_number: StudentNumber
    full_name_ar: str
    full_name_en: str
    class_code: ClassCode | None = None
    #: Placement start. Defaults at the service layer to the academic year's start rather
    #: than to "today", so importing a roster in November does not record every child as
    #: having joined in November.
    starts_on: date | None = None

    @property
    def has_name(self) -> bool:
        """A child must be nameable in at least one script to be worth a row."""
        return bool(self.full_name_ar.strip() or self.full_name_en.strip())


@dataclass(frozen=True, slots=True)
class RosterPreviewCommand:
    """Parse and validate a roster upload, writing an ImportBatch and nothing else.

    `content` is bytes already in memory. No path, because the API layer must never write
    a file of children's names to disk for a parser to read back.
    """

    academic_year_code: AcademicYearCode
    filename: str
    content: bytes
    actor: str
    #: Target class when the file is one-class-per-upload. None means each row must name
    #: its own class, and a row that names none is rejected rather than guessed at.
    class_code: ClassCode | None = None
    default_starts_on: date | None = None


@dataclass(frozen=True, slots=True)
class RosterCommitCommand:
    """Apply a previously previewed batch.

    Carries the batch id rather than the file: commit replays what the registrar actually
    saw. Re-reading the upload would let a different file be applied than the one that was
    reviewed, which is precisely the substitution the two-step flow exists to prevent.
    """

    # `str`, matching `ImportBatch.batch_id`, which is a uuid4 hex and never an integer.
    batch_id: str
    actor: str
