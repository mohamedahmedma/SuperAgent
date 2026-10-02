"""The preview-then-commit substrate: a batch, its rows, and the fate of each row.

Decision 4 in one sentence: **an import must be able to describe what it would do to
every row before it writes anything at all.** These types are how that description is
carried, which is why they are inert data with no repository, no session and no clock
inside them.

The failure this exists to prevent is not a crash. A roster import that mis-parses a
column succeeds — it attaches the wrong child to the wrong class, and it does so
silently, because "student 0071 was placed in 3B" looks exactly like a correct row to
every layer below the human reading it. There is no later error to catch: the grades
that follow land on the wrong report card and are perfectly well-formed. The only place
that mistake is detectable is on a screen, before the write, with the registrar looking
at it. So preview must produce a complete, per-row account of the outcome, and it must
produce it *without* the write having happened.

That has three consequences visible in this module:

* An `ImportRow` carries the row's `payload` as parsed data rather than a database
  object, so a preview can be built by a service that has read the file and touched
  nothing else.
* A rejected row is a *value*, not a raised exception. One bad student number must not
  discard the twenty-nine good rows above it, so a per-row failure is recorded here and
  the batch continues. Only `ImportFailure` and its subclasses kill a batch.
* `ImportBatch` pins a `content_hash`. The gap between preview and commit is a second
  HTTP request, and without the hash a client that re-uploads in between commits rows
  no human ever saw — reintroducing exactly the failure the two-step flow removes.

Nothing here imports sqlalchemy, fastapi or pydantic, and nothing reads the clock. `now`
is a parameter on every method that needs it (see `ImportBatch.is_expired`), because a
service that expires a batch by calling `datetime.now()` inside the domain can only be
tested by sleeping or by monkeypatching a module global.
"""
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field as dataclass_field, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from sis.domain.errors import (
    ImportBatchAlreadyCommitted,
    ImportBatchExpired,
    ImportContentMismatch,
    InvalidDateRange,
    SisError,
    ValidationError,
)

_EMPTY_PAYLOAD: Mapping[str, object] = MappingProxyType({})


class ImportKind(StrEnum):
    """What a file is claiming to be. Chosen by the caller, never sniffed.

    A `StrEnum` because these values are persisted, put in URLs and shown in JSON; an
    `IntEnum` would store `0` and `1` and make the database unreadable to the person
    debugging an import at 7am, while a bare `str` would let `"rosters"` reach the
    parser registry and fail as "unknown kind" three layers from the typo.

    Kind is *not* inferred from the file's columns. A grades sheet whose header row was
    edited can look like a roster, and guessing wrong means writing enrolments from a
    file of marks.
    """

    ROSTER = "roster"
    GRADES = "grades"
    GUARDIANS = "guardians"


class ImportStatus(StrEnum):
    """Where a batch is in the two-step flow. Only `PREVIEWED` may be committed."""

    PREVIEWED = "previewed"
    COMMITTED = "committed"
    EXPIRED = "expired"


class RowOutcome(StrEnum):
    """What happened, or would happen, to one row.

    The distinction between `CREATED`, `UPDATED` and `UNCHANGED` is what makes a preview
    worth reading: a registrar re-uploading a corrected file needs to see that 200 rows
    are unchanged and 3 are updated. Collapsing those into "ok" hides the three rows
    that are the entire reason for the upload.

    `SKIPPED` is deliberately separate from `REJECTED`. A skipped row is one the file
    asked to be ignored — a blank line, a totals row — and it is not a defect the
    registrar has to fix. Counting it as a rejection makes a clean file look broken.
    """

    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ImportRow:
    """One line of the uploaded file and its fate. A preview is a list of these.

    `code` is the machine-readable half of the outcome and is the vocabulary
    `sis.domain.errors` refers to as the row code. For an accepted row it is the
    `RowOutcome` value; for a rejected one it is the failing `SisError.code`, reused
    verbatim rather than remapped. That reuse is the point: `invalid_student_number`
    means the same thing whether it arrives as an HTTP error body or as one red line in
    a preview table, so the UI needs one lookup table and the vocabulary cannot drift
    into two versions of itself.

    `message` is prose for a human and may be translated or reworded at any time.
    Nothing may branch on it.
    """

    line_number: int
    payload: Mapping[str, object] = _EMPTY_PAYLOAD
    outcome: RowOutcome = RowOutcome.CREATED
    code: str = RowOutcome.CREATED.value
    message: str | None = None
    field: str | None = None

    def __post_init__(self) -> None:
        if self.line_number < 1:
            raise ValidationError(
                f"line number must be 1 or greater (got {self.line_number})",
                field="line_number",
            )
        # Copied, then frozen behind a read-only view. The parser hands over the same
        # dict it built its row from; without the copy a later normalisation pass
        # mutates the payload the registrar already saw in the preview, and the
        # committed row stops matching the screen it was approved on.
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def is_rejected(self) -> bool:
        return self.outcome is RowOutcome.REJECTED

    @property
    def is_written(self) -> bool:
        """True when committing this row changes stored data."""
        return self.outcome in (RowOutcome.CREATED, RowOutcome.UPDATED)

    @classmethod
    def accepted(
        cls,
        line_number: int,
        payload: Mapping[str, object],
        outcome: RowOutcome,
        *,
        message: str | None = None,
    ) -> "ImportRow":
        """Record a row that will be written, or that needs no write."""
        if outcome is RowOutcome.REJECTED:
            raise ValidationError(
                "a rejected row must be built with ImportRow.rejected",
                field="outcome",
            )
        return cls(
            line_number=line_number,
            payload=payload,
            outcome=outcome,
            code=outcome.value,
            message=message,
        )

    @classmethod
    def rejected(
        cls,
        line_number: int,
        payload: Mapping[str, object],
        error: SisError,
    ) -> "ImportRow":
        """Turn a caught domain error into a row outcome instead of a batch failure.

        Takes the error object rather than a code string so the row's `code`, `message`
        and `field` can only ever come from one place. A caller that passed a hand-typed
        code would eventually pass one no UI knows about, and the registrar would be
        shown a blank cell where the reason should be.
        """
        return cls(
            line_number=line_number,
            payload=payload,
            outcome=RowOutcome.REJECTED,
            code=error.code,
            message=error.message,
            field=error.field,
        )


def tally(rows: Iterable[ImportRow]) -> dict[str, int]:
    """Count rows by outcome, with every outcome present so zeros are explicit.

    A missing key and a zero read identically to a template that uses `.get(..., 0)`,
    and differently to one that does not; seeding all five removes the question.
    """
    counts = {outcome.value: 0 for outcome in RowOutcome}
    for row in rows:
        counts[row.outcome.value] += 1
    return counts


@dataclass(frozen=True, slots=True)
class ImportBatch:
    """A previewed file awaiting a decision: its identity, its verdict, its deadline.

    Holds counts rather than the rows themselves. A batch is summary — "37 created, 2
    rejected" — and is small enough to list; the rows live beside it and are fetched for
    one batch at a time. Embedding them would make every "recent imports" query drag
    thousands of parsed cells through memory.
    """

    batch_id: str
    kind: ImportKind
    content_hash: str
    actor: str
    created_at: datetime
    expires_at: datetime
    status: ImportStatus = ImportStatus.PREVIEWED
    counts: Mapping[str, int] = dataclass_field(default_factory=dict)
    committed_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("batch_id", "content_hash", "actor"):
            if not str(getattr(self, name)).strip():
                raise ValidationError(f"{name} is required", field=name)
        for name in ("created_at", "expires_at", "committed_at"):
            moment: datetime | None = getattr(self, name)
            # Naive datetimes compare fine and expire wrongly. A TTL measured against a
            # local-time `created_at` and a UTC `now` is off by the offset, which in this
            # region means every preview is born already expired -- or never expires.
            if moment is not None and moment.tzinfo is None:
                raise ValidationError(f"{name} must be timezone-aware", field=name)
        if self.expires_at <= self.created_at:
            raise InvalidDateRange(
                "a preview must expire after it was created", field="expires_at"
            )
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    @property
    def total_rows(self) -> int:
        return sum(self.counts.values())

    @property
    def rejected_rows(self) -> int:
        return self.counts.get(RowOutcome.REJECTED.value, 0)

    @property
    def writable_rows(self) -> int:
        """Rows a commit would actually write. Zero means the commit is a no-op."""
        return self.counts.get(RowOutcome.CREATED.value, 0) + self.counts.get(
            RowOutcome.UPDATED.value, 0
        )

    def is_expired(self, now: datetime) -> bool:
        """`now` is a parameter so a test can be a function call rather than a sleep."""
        return now >= self.expires_at

    def status_at(self, now: datetime) -> ImportStatus:
        """The status a reader should see, aging a stale preview out on read.

        Expiry is a fact about a timestamp, not an event someone remembered to record.
        Deriving it here means a batch whose sweeper never ran still reports `EXPIRED`
        instead of offering a commit button that will fail.
        """
        if self.status is ImportStatus.PREVIEWED and self.is_expired(now):
            return ImportStatus.EXPIRED
        return self.status

    def assert_committable(self, now: datetime, content_hash: str) -> None:
        """The complete gate between preview and write. Raises, or returns None.

        Lives on the batch rather than in the service so that the three ways a commit
        can be illegitimate cannot be checked in two places and drift apart. The hash
        comparison is last because the other two produce the clearer message when both
        apply.
        """
        current = self.status_at(now)
        if current is ImportStatus.COMMITTED:
            raise ImportBatchAlreadyCommitted(
                f"batch {self.batch_id} was already committed", field="batch_id"
            )
        if current is ImportStatus.EXPIRED:
            raise ImportBatchExpired(
                f"batch {self.batch_id} expired; upload the file again to see the "
                "current outcomes",
                field="batch_id",
            )
        if content_hash != self.content_hash:
            raise ImportContentMismatch(
                "this file is not the one that was previewed", field="content_hash"
            )

    def committed(self, at: datetime) -> "ImportBatch":
        """The same batch, marked done. Returns a new value; batches are never mutated."""
        return replace(self, status=ImportStatus.COMMITTED, committed_at=at)

    def expired(self) -> "ImportBatch":
        """Persist what `status_at` already derives, for a sweeper reclaiming rows."""
        return replace(self, status=ImportStatus.EXPIRED)
