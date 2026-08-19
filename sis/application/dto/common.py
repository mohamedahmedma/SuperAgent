"""The shared vocabulary of import outcomes and paging.

Why `RowCode` is a closed enum rather than a free-text reason: a registrar who uploads
1200 rows does not read 1200 messages. She filters — "show me the 14 that failed", then
"show me the 9 that failed for the same reason" — and a screen that says *9 rows have an
unknown class* is what turns a rejected file into one fix. Free text cannot be counted,
because "unknown class 3A" and "class 3B not found" are one problem spelled two ways and
tally as two. It cannot be translated either: this school reads Arabic, and a reason
assembled inside a parser is a string no translation file has ever seen. The code is the
part that is aggregated, filtered and translated; `message` is the human sentence beside
it, and nothing may depend on its wording.

The vocabulary being *closed* is the other half. A parser that needs a reason not listed
here must add a member — a reviewed, one-line change that shows up in the UI's filter
list and in the translation table — rather than inventing a diagnostic that renders as
an untranslated blank in production. Members are wire values: they appear verbatim in
HTTP responses and in stored batches, so **renaming one is a breaking change**, exactly
as in `sis.domain.errors`.

Everything here is a dataclass. Pydantic lives in `api/` and only there; these types are
what the service unit tests construct by hand, and they must stay cheap to construct.
"""
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Final

from sis.domain.errors import SisError

__all__ = [
    "RowCode",
    "RowOutcome",
    "ImportPreviewResult",
    "ImportCommitResult",
    "PageRequest",
    "Page",
    "tally_by_code",
]


class RowCode(StrEnum):
    """Every outcome a single import row can have. Closed on purpose — see the module docstring.

    `StrEnum` so the member *is* its wire value: `RowCode.OK == "ok"` holds, JSON
    encoding needs no custom hook, and a stored batch read back from the database
    reconstructs by value. The cost is that a typo in a comparison against a bare string
    is silently false, which is why services should compare against members.
    """

    # The only non-failure. A row that will be written on commit, or was.
    OK = "ok"

    # --- Identity collisions ------------------------------------------------
    # Split in two because the registrar's fix differs: a duplicate inside the file is
    # edited in the spreadsheet, while a duplicate against the database usually means
    # the file was already uploaded and the right answer is to upload nothing.
    DUPLICATE_IN_FILE = "duplicate_in_file"
    DUPLICATE_EXISTING = "duplicate_existing"

    # --- References that name nothing --------------------------------------
    # Four members where `UnknownReference` is one domain error, because "3 rows name a
    # subject that does not exist" and "3 rows name a student who does not exist" are
    # different afternoons of work, and the count per kind is the whole diagnostic.
    UNKNOWN_CLASS = "unknown_class"
    UNKNOWN_SUBJECT = "unknown_subject"
    UNKNOWN_TERM = "unknown_term"
    UNKNOWN_STUDENT = "unknown_student"

    # --- Malformed cells ----------------------------------------------------
    MISSING_STUDENT_NUMBER = "missing_student_number"
    # One member for blank, malformed and wrong-length alike, matching how
    # MISSING_STUDENT_NUMBER already covers both an empty cell and an unparsable one: the
    # registrar's fix is the same in every case — look at the phone column of that row.
    MISSING_PHONE = "missing_phone"
    INVALID_NAME = "invalid_name"
    INVALID_GRADE = "invalid_grade"
    # Distinct from INVALID_GRADE: a mark of 105 is a teacher marking out of 120, which
    # is a conversation about the grading scale, not about a mistyped cell.
    GRADE_OUT_OF_RANGE = "grade_out_of_range"

    # --- Rules the row breaks ----------------------------------------------
    ALREADY_ENROLLED_ELSEWHERE = "already_enrolled_elsewhere"

    # --- Failures that belong to the batch, recorded per row ----------------
    # These three exist as row codes as well as errors so that a commit can report *why*
    # nothing was written in the same shape as everything else. A UI that has one
    # renderer for row outcomes should not need a second one for these.
    CHANGED_SINCE_PREVIEW = "changed_since_preview"
    BATCH_EXPIRED = "batch_expired"
    BATCH_ALREADY_COMMITTED = "batch_already_committed"


# Shared read-only defaults. A `dict` default_factory would give every outcome its own
# empty dict and invite a caller to mutate one after construction.
_EMPTY_PAYLOAD: Final[Mapping[str, object]] = MappingProxyType({})
_EMPTY_TOTALS: Final[Mapping[RowCode, int]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class RowOutcome:
    """What happened to one row of one uploaded file.

    `line` is the number the registrar sees in Excel's own gutter — 1-based, counting
    the header. Reporting a zero-based data index instead produces the classic support
    call where "row 12 is wrong" points at row 13 on her screen, and she edits the wrong
    child's marks.

    `payload` carries the row's identifying cells (student number, class code, the value
    that failed) so the UI can render a table without holding the original file. It is
    deliberately untyped: each importer puts in what its own columns are, and pinning a
    schema here would make this module know about grades, rosters and structure files.
    """

    line: int
    code: RowCode
    message: str
    payload: Mapping[str, object] = _EMPTY_PAYLOAD
    # Which column failed, when there is a single one — mirrors `SisError.field`, and is
    # what lets the UI highlight a cell rather than a whole row.
    field: str | None = None

    @property
    def is_ok(self) -> bool:
        return self.code is RowCode.OK

    @classmethod
    def accepted(cls, line: int, *, payload: Mapping[str, object] | None = None) -> "RowOutcome":
        """A row that passed. Message stays empty; there is nothing to tell a human."""
        return cls(line=line, code=RowCode.OK, message="", payload=payload or _EMPTY_PAYLOAD)

    @classmethod
    def from_error(
        cls,
        line: int,
        code: RowCode,
        error: SisError,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> "RowOutcome":
        """Turn a caught domain error into a recorded row rather than a dead batch.

        This is the concrete form of decision 4: one bad row must never discard the good
        ones. The caller chooses the `RowCode` because the domain's `UnknownReference`
        deliberately does not know whether the missing thing was a class or a subject —
        the importer does, from the column it was reading.
        """
        return cls(
            line=line,
            code=code,
            message=error.message,
            payload=payload or _EMPTY_PAYLOAD,
            field=error.field,
        )


def tally_by_code(outcomes: Iterable[RowOutcome]) -> Mapping[RowCode, int]:
    """Count outcomes per code, omitting codes that did not occur.

    Omitting the zeros is the point: the UI renders this mapping directly as the filter
    chips above the table, and a file with two bad rows should show two chips, not
    fifteen of which thirteen read `0`.
    """
    counts = Counter(outcome.code for outcome in outcomes)
    # Ordered by the enum, not by frequency, so the same file always renders its chips in
    # the same order and a registrar comparing two uploads is comparing like with like.
    return MappingProxyType({code: counts[code] for code in RowCode if counts[code]})


@dataclass(frozen=True, slots=True)
class _ImportResult:
    """Shared shape of preview and commit: the rows, and the counts drawn from them.

    `totals` is stored rather than computed on access because a committed batch is
    persisted and read back for audit long after the rows may have been trimmed, and a
    total that recomputes from a truncated row list quietly reports a smaller file than
    was actually processed.
    """

    batch_id: str
    rows: tuple[RowOutcome, ...]
    totals: Mapping[RowCode, int] = _EMPTY_TOTALS

    @property
    def total_rows(self) -> int:
        return sum(self.totals.values())

    @property
    def ok_count(self) -> int:
        return self.totals.get(RowCode.OK, 0)

    @property
    def rejected_count(self) -> int:
        return self.total_rows - self.ok_count

    @property
    def has_rejections(self) -> bool:
        return self.rejected_count > 0

    def count(self, code: RowCode) -> int:
        """Zero for a code that did not occur — see `tally_by_code` on the missing keys."""
        return self.totals.get(code, 0)

    def failures(self) -> tuple[RowOutcome, ...]:
        """Just the rows a human has to act on. The 14 out of 1200."""
        return tuple(row for row in self.rows if not row.is_ok)


@dataclass(frozen=True, slots=True)
class ImportPreviewResult(_ImportResult):
    """What the registrar is shown *before* anything is written.

    Carries `batch_id` because the commit step names this exact preview rather than
    re-uploading, and `expires_at` because a preview validated against last week's class
    list must not be committable today (see `ImportBatchExpired`). Both are surfaced on
    the DTO so the UI can display the deadline instead of discovering it as a 409 after
    the registrar has finished reviewing 1200 rows.
    """

    expires_at: datetime | None = None

    @classmethod
    def from_rows(
        cls,
        batch_id: str,
        rows: Iterable[RowOutcome],
        *,
        expires_at: datetime | None = None,
    ) -> "ImportPreviewResult":
        materialised = tuple(rows)
        return cls(
            batch_id=batch_id,
            rows=materialised,
            totals=tally_by_code(materialised),
            expires_at=expires_at,
        )


@dataclass(frozen=True, slots=True)
class ImportCommitResult(_ImportResult):
    """What was actually written, row by row.

    Not the same object as the preview and not assumed to match it. A row that previewed
    `ok` can commit as `changed_since_preview` — another registrar enrolled the child in
    the meantime — and reusing the preview's result would report that row as written when
    it was not. `ok_count` here is the number of rows that hit the database.
    """

    committed_at: datetime | None = None

    @classmethod
    def from_rows(
        cls,
        batch_id: str,
        rows: Iterable[RowOutcome],
        *,
        committed_at: datetime | None = None,
    ) -> "ImportCommitResult":
        materialised = tuple(rows)
        return cls(
            batch_id=batch_id,
            rows=materialised,
            totals=tally_by_code(materialised),
            committed_at=committed_at,
        )


@dataclass(frozen=True, slots=True)
class PageRequest:
    """A window over a listing, clamped at construction.

    Clamped rather than rejected because the caller that asks for 10000 students is a
    UI bug or a curious URL, and answering it with 200 rows is better service than a 422
    — but answering it in full is how one request pulls an entire school into memory and
    times out for everyone else. The clamp is silent by design; `limit` after
    construction is the truth, and callers that echo it back tell the client what they
    actually got.
    """

    DEFAULT_LIMIT: ClassVar[int] = 50
    MAX_LIMIT: ClassVar[int] = 200

    limit: int = DEFAULT_LIMIT
    offset: int = 0

    def __post_init__(self) -> None:
        limit = self.limit if self.limit > 0 else self.DEFAULT_LIMIT
        object.__setattr__(self, "limit", min(limit, self.MAX_LIMIT))
        object.__setattr__(self, "offset", max(self.offset, 0))

    def next_page(self) -> "PageRequest":
        return PageRequest(limit=self.limit, offset=self.offset + self.limit)


# No `slots=True` here: combining it with PEP 695 type parameters loses the generic on
# some 3.12 patch releases, because `slots=True` rebuilds the class. Not worth the risk
# for a wrapper that is constructed once per request.
@dataclass(frozen=True)
class Page[ItemT]:
    """One window of results plus the count of everything that matched.

    `total` is the size of the whole result set, not of `items`. Without it the UI can
    only offer "next" and never "page 7 of 40", and a registrar scanning for one child
    in a year group of 600 has no idea whether she is near the end.
    """

    items: tuple[ItemT, ...]
    total: int
    request: PageRequest = PageRequest()

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def has_more(self) -> bool:
        return self.request.offset + self.count < self.total

    @classmethod
    def of(
        cls, items: Iterable[ItemT], total: int, request: PageRequest | None = None
    ) -> "Page[ItemT]":
        return cls(
            items=tuple(items), total=total, request=request or PageRequest()
        )

    @classmethod
    def empty(cls, request: PageRequest | None = None) -> "Page[ItemT]":
        return cls(items=(), total=0, request=request or PageRequest())
