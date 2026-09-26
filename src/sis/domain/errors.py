"""Domain errors, each carrying the machine-readable code the API will return.

Why these are not `ValueError`: the API layer has to turn a failure into
`{"code": ..., "message": ...}`, and deriving that code from an exception's *type name*
or its message text is how a refactor silently changes a caller's contract. Every error
here states its own `code`, so the router's job is a one-line translation and the
vocabulary is greppable from one file.

The codes are a closed vocabulary in the same style as `records/routes.py`
(`not_found`, `not_authorized`, `conflict`). They appear verbatim in HTTP responses and
in `ImportRow.code`, so **renaming one is a breaking API change**.

Nothing here imports sqlalchemy, fastapi or pydantic, and nothing here knows an HTTP
status code exists. The mapping from error to status lives in the API layer, because
the same `DuplicateCode` is a 409 to a registrar and a rejected row inside an import --
one error, two renderings.
"""
from typing import ClassVar


class SisError(Exception):
    """Root of everything this service raises deliberately.

    A caller that catches `SisError` catches every *expected* failure and nothing else.
    That distinction matters at the import boundary: a domain error means "this row is
    bad, record it and continue", while a `KeyError` escaping from a parser means the
    code is broken and the whole batch must fail loudly rather than be recorded as
    thirty rejected rows.
    """

    code: ClassVar[str] = "sis_error"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        # Which input caused it, when there is a single one. The registrar UI uses this
        # to highlight a cell; without it the operator is told "invalid code" about a
        # spreadsheet with eleven code columns.
        self.field = field

    def as_dict(self) -> dict[str, str | None]:
        """The exact shape the API returns and an `ImportRow` stores."""
        return {"code": self.code, "message": self.message, "field": self.field}


# ---------------------------------------------------------------------------
# Validation. A value that cannot exist. Raised by value objects.
# ---------------------------------------------------------------------------


class ValidationError(SisError):
    """A supplied value is not a member of the type it claims to be."""

    code: ClassVar[str] = "invalid_value"


class InvalidCode(ValidationError):
    """A code is empty, too long, or holds characters that do not survive a round trip.

    Codes travel through URLs, CSV cells and Moodle idnumbers. A code containing a space
    or a slash is the kind of value that works in the database and breaks in exactly one
    of those three places, months later.
    """

    code: ClassVar[str] = "invalid_code"


class InvalidStudentNumber(ValidationError):
    """The school's own identifier for a child failed validation.

    Separate from `InvalidCode` because it is the single most common bad cell in a
    roster upload, and the registrar UI counts it separately to answer "is this file
    wrong, or is this one row wrong".
    """

    code: ClassVar[str] = "invalid_student_number"


class InvalidPhone(ValidationError):
    """A phone number that cannot be dialled: blank, or the wrong number of digits.

    Separate from `InvalidCode` because a phone number is the identity of a guardian —
    it is what recognises the same mother across three of her children's rows — so a bad
    one is not merely a bad cell, it is a family that cannot be reached and, later, a
    parent who cannot receive a login code. The registrar UI counts it on its own for the
    same reason it counts `InvalidStudentNumber` separately.
    """

    code: ClassVar[str] = "invalid_phone"


class InvalidPercentage(ValidationError):
    """A percentage outside 0..100, or NaN.

    NaN is called out because it is what a spreadsheet produces for `=A1/B1` when B1 is
    empty, and it compares false to everything including itself. A NaN that reaches the
    database is a grade that is neither present nor absent, and every later comparison
    against it silently answers "no".
    """

    code: ClassVar[str] = "invalid_percentage"


class InvalidDateRange(ValidationError):
    """A period whose end precedes its start.

    Caught in the domain because the database cannot express it portably, and because
    an inverted range makes a term contain no days. Every "which class was she in"
    query then returns nothing, which reads as missing data rather than as bad
    configuration.
    """

    code: ClassVar[str] = "invalid_date_range"


# ---------------------------------------------------------------------------
# Rule violations. Values are well formed, the state they imply is not.
# ---------------------------------------------------------------------------


class DomainRuleViolation(SisError):
    """A well-formed request that the school's rules forbid."""

    code: ClassVar[str] = "rule_violation"


class DuplicateCode(DomainRuleViolation):
    """Something already exists under this code.

    Not an error during structure generation. Decision 3 makes generation idempotent,
    so an existing class is *expected* there and reported as "already present". This is
    an error only when a caller explicitly asks to create one.
    """

    code: ClassVar[str] = "duplicate_code"


class UnknownReference(DomainRuleViolation):
    """A code that names nothing: an unknown subject, term, class or student.

    Deliberately one error rather than four. Callers branch on `field` to learn which
    reference failed; the code stays stable so the import row vocabulary does not grow
    a new member every time a foreign key is added.
    """

    code: ClassVar[str] = "unknown_reference"


class ImmutableFieldChange(DomainRuleViolation):
    """An attempt to change a code that is identity rather than a label.

    Decision 7. `YearLevel.code`, `ClassSection.code`, `Subject.code`, `Term.code` and
    `Student.student_number` are what grades and enrolments are attached by. Renaming a
    class must not detach its students, which is exactly what happens when a rename is
    implemented as "write a new code": the old rows keep pointing at a code nothing
    resolves any more. Labels (`name_ar`, `name_en`) are freely renameable.
    """

    code: ClassVar[str] = "immutable_field"


class OverlappingEnrolment(DomainRuleViolation):
    """A student would be in two classes on the same day.

    The rule decision 2 exists to protect. A child with two live placements makes
    "which class was she in during Term 1" ambiguous, and every report that answers it
    picks arbitrarily, so two screens in the same system disagree about one child.
    """

    code: ClassVar[str] = "overlapping_enrolment"


class NotEnrolled(DomainRuleViolation):
    """A grade was submitted for a student with no placement covering that term.

    Refused rather than stored, because `SubjectGrade` carries the class the mark was
    earned in. Storing it with a guessed class produces a grade sheet where a child
    appears in a class she was never in.
    """

    code: ClassVar[str] = "not_enrolled"


class TermClosed(DomainRuleViolation):
    """A write against a term the school has closed.

    Closing a term is how a school freezes what it published. Allowing a late write
    would let this year's correction silently change what last year's report said.
    """

    code: ClassVar[str] = "term_closed"


# ---------------------------------------------------------------------------
# Imports. The preview-then-commit substrate (decisions 4 and 5).
# ---------------------------------------------------------------------------


class ImportFailure(SisError):
    """Base for whole-file import failures.

    Named `ImportFailure` rather than `ImportError` because the latter is a builtin.
    Shadowing it inside a package makes a genuine module-loading failure unreadable in
    a traceback, and an `except ImportError` anywhere in this package would then
    quietly swallow a bad spreadsheet.

    Everything under this class kills the *batch*. A bad row does not; it becomes an
    `ImportRow` carrying a `RowCode` (decision 5).
    """

    code: ClassVar[str] = "import_failed"


class UnsupportedFileType(ImportFailure):
    """A file extension no parser claims.

    Its own code so the UI can say "save this as .xlsx or .csv" instead of "could not
    read file", which sends the registrar looking for corruption that is not there.
    `.xls`, the pre-2007 binary format, is deliberately unsupported.
    """

    code: ClassVar[str] = "unsupported_file_type"


class UnreadableImportFile(ImportFailure):
    """The file cannot be opened at all: truncated, encrypted, or not a spreadsheet.

    The *only* error a parser may raise. A malformed row is a per-row diagnostic
    (decision 5) because one bad student number must not discard the twenty-nine good
    rows above it. This error means there were never any rows to keep.
    """

    code: ClassVar[str] = "unreadable_file"


class UploadTooLarge(ImportFailure):
    """The upload exceeds `SIS_MAX_UPLOAD_BYTES`.

    Checked before parsing, not during. A parser that discovers the problem has already
    read the file into memory, which is the resource exhaustion the limit exists to
    prevent.
    """

    code: ClassVar[str] = "upload_too_large"


class ImportBatchNotFound(ImportFailure):
    """No such preview batch."""

    code: ClassVar[str] = "batch_not_found"


class ImportBatchExpired(ImportFailure):
    """The preview aged out before it was committed.

    The TTL is what stops a preview taken against last week's class list from being
    committed today, after the structure it validated against has changed. The
    registrar re-uploads and sees the current outcomes, which may differ. That is the
    point.
    """

    code: ClassVar[str] = "batch_expired"


class ImportBatchAlreadyCommitted(ImportFailure):
    """A second commit of the same batch.

    Almost always a double-clicked button or a retried request. Refusing is what makes
    commit idempotent from the caller's side: the first call's outcome stands, and the
    second is told so rather than re-applying thirty enrolments.
    """

    code: ClassVar[str] = "batch_already_committed"


class ImportContentMismatch(ImportFailure):
    """The committed batch is not the file the registrar previewed.

    Guards the gap between preview and commit (decision 4). Without this check, a client
    that re-uploads between the two steps commits rows a human never saw, which is
    precisely the "wrong child attached to the wrong class" failure the two-step flow
    exists to prevent.
    """

    code: ClassVar[str] = "content_mismatch"
