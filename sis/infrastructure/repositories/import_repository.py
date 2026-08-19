"""`ImportBatchRepository` over SQLAlchemy: a preview, its rows, and their fate.

Decision 4 is a storage problem before it is a UI one. The batch header and every one of
its rows are written at *preview* time, so the outcomes the registrar approves are the
outcomes a commit replays — and so one rejected row is a stored value rather than an
exception that discards the twenty-nine good rows above it.

Two things in here are not obvious from the domain types:

**`ImportRow.batch_id` is not `ImportBatch.batch_id`.** The row's is the surrogate
integer FK; the batch's is the opaque public string handed to the client. They are
different types with the same name, and joining on the wrong one either matches nothing
or matches everything. Every query below goes through `_pk`, which converts the public
id to the FK exactly once.

**Timestamps come back naive from SQLite.** `DateTime(timezone=True)` has no offset to
store there, so a moment written as aware UTC is read back without a tzinfo — and
`ImportBatch.__post_init__` rejects naive datetimes, by design, because a TTL measured
against a local-time `created_at` expires every preview at birth or never. `_as_utc`
reattaches UTC on the way out, which is sound only because everything written here is
UTC to begin with.

Nothing here commits: an import that committed per row could not be abandoned.
"""
from collections.abc import Collection, Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import ScalarSelect, delete, func, insert, select
from sqlalchemy.orm import Session

from sis.domain.errors import ImportBatchNotFound
from sis.domain.imports import (
    ImportBatch,
    ImportKind,
    ImportRow,
    ImportStatus,
    RowOutcome,
)
from sis.infrastructure.db import models

# Statuses a sweeper may reclaim. `COMMITTED` is absent on purpose: a committed batch is
# the record of what was actually written to the school's data, and deleting it turns
# "who imported these 300 grades, from which file" into an unanswerable question.
_SWEEPABLE = (ImportStatus.PREVIEWED.value, ImportStatus.EXPIRED.value)


class SqlAlchemyImportBatchRepository:
    """Preview batches and their rows. One class, because a row has no life alone."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, batch: ImportBatch, rows: Sequence[ImportRow]) -> None:
        """Store the header and every row of a preview in one write."""
        record = models.ImportBatch(
            batch_id=batch.batch_id,
            kind=batch.kind.value,
            status=batch.status.value,
            content_hash=batch.content_hash,
            actor=batch.actor,
            counts=dict(batch.counts),
            created_at=batch.created_at,
            expires_at=batch.expires_at,
            committed_at=batch.committed_at,
        )
        self._session.add(record)
        # Flushed rather than deferred: the rows need the surrogate id this assigns, and
        # a duplicate `batch_id` must fail here rather than after hundreds of row inserts.
        self._session.flush()
        self._insert_rows(record.id, rows)
        self._session.flush()

    def get(self, batch_id: str) -> ImportBatch | None:
        record = self._find(batch_id)
        return None if record is None else _batch_to_domain(record)

    def save(self, batch: ImportBatch) -> ImportBatch:
        """Persist a status transition on an existing batch.

        Status, counts and `committed_at` only. `kind` and `content_hash` describe the
        file the preview was taken against, and a save that rewrote them would let a
        commit quietly re-point at a file no human ever reviewed.
        """
        record = self._require(batch.batch_id)
        record.status = batch.status.value
        record.counts = dict(batch.counts)
        record.committed_at = batch.committed_at
        self._session.flush()
        return batch

    def list_rows(
        self,
        batch_id: str,
        *,
        outcomes: Collection[RowOutcome] | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> Sequence[ImportRow]:
        """Rows in file order, filtered and paged."""
        stmt = (
            select(models.ImportRow)
            .where(models.ImportRow.batch_id == self._pk(batch_id))
            .order_by(models.ImportRow.line_number)
        )
        stmt = _filtered(stmt, outcomes)
        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return [_row_to_domain(row) for row in self._session.scalars(stmt)]

    def count_rows(
        self, batch_id: str, *, outcomes: Collection[RowOutcome] | None = None
    ) -> int:
        stmt = select(func.count()).select_from(models.ImportRow).where(
            models.ImportRow.batch_id == self._pk(batch_id)
        )
        return self._session.scalar(_filtered(stmt, outcomes)) or 0

    def replace_rows(self, batch_id: str, rows: Sequence[ImportRow]) -> None:
        """Swap a preview's predicted outcomes for the ones the commit produced.

        A delete-then-insert rather than a per-row update: the commit's outcomes are a
        complete restatement, and matching them up line by line would leave a row the
        second pass never mentioned sitting in the batch as a stale prediction.
        """
        record = self._require(batch_id)
        self._session.execute(
            delete(models.ImportRow).where(models.ImportRow.batch_id == record.id)
        )
        self._insert_rows(record.id, rows)
        self._session.flush()

    def delete_expired(self, now: datetime) -> int:
        """Drop previews past their TTL; returns how many batches went."""
        stale = list(
            self._session.scalars(
                select(models.ImportBatch.id).where(
                    models.ImportBatch.expires_at <= now,
                    models.ImportBatch.status.in_(_SWEEPABLE),
                )
            )
        )
        if not stale:
            return 0
        # Rows first, explicitly. `ondelete="CASCADE"` would cover it, but only while
        # SQLite's per-connection foreign-key pragma is on; a sweeper that leaves orphan
        # rows behind leaves them invisibly, and they are never read again to notice.
        self._session.execute(
            delete(models.ImportRow).where(models.ImportRow.batch_id.in_(stale))
        )
        self._session.execute(
            delete(models.ImportBatch).where(models.ImportBatch.id.in_(stale))
        )
        self._session.flush()
        return len(stale)

    # -- internals -----------------------------------------------------------

    def _insert_rows(self, batch_pk: int, rows: Sequence[ImportRow]) -> None:
        if not rows:
            return
        self._session.execute(
            insert(models.ImportRow),
            [
                {
                    "batch_id": batch_pk,
                    "line_number": row.line_number,
                    # `payload` is a read-only MappingProxyType and the JSON serialiser
                    # cannot encode one; copying is also what keeps the stored preview
                    # identical to the screen it was approved on.
                    "payload": dict(row.payload),
                    "outcome": row.outcome.value,
                    "code": row.code,
                    "message": row.message,
                    "field": row.field,
                }
                for row in rows
            ],
        )

    def _pk(self, batch_id: str) -> ScalarSelect[int]:
        """The surrogate id for a public batch id, as a subquery — see the module docstring."""
        return (
            select(models.ImportBatch.id)
            .where(models.ImportBatch.batch_id == batch_id)
            .scalar_subquery()
        )

    def _find(self, batch_id: str) -> models.ImportBatch | None:
        return self._session.scalars(
            select(models.ImportBatch).where(models.ImportBatch.batch_id == batch_id)
        ).first()

    def _require(self, batch_id: str) -> models.ImportBatch:
        record = self._find(batch_id)
        if record is None:
            raise ImportBatchNotFound(
                f"no import batch {batch_id}; previews are dropped once they expire",
                field="batch_id",
            )
        return record


def _filtered(stmt: Any, outcomes: Collection[RowOutcome] | None) -> Any:
    """Narrow to the outcomes asked for, treating an empty collection as *none of them*.

    Widening an empty filter to "everything" is the tempting reading and the wrong one: a
    preview screen asking for rejections would answer with the whole two-thousand-row
    file the moment its filter list came back empty.
    """
    if outcomes is None:
        return stmt
    return stmt.where(models.ImportRow.outcome.in_([outcome.value for outcome in outcomes]))


def _as_utc(moment: datetime) -> datetime:
    """Reattach UTC to a timestamp SQLite handed back without one. See the module docstring."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _batch_to_domain(record: models.ImportBatch) -> ImportBatch:
    return ImportBatch(
        batch_id=record.batch_id,
        kind=ImportKind(record.kind),
        content_hash=record.content_hash,
        actor=record.actor,
        created_at=_as_utc(record.created_at),
        expires_at=_as_utc(record.expires_at),
        status=ImportStatus(record.status),
        counts=dict(record.counts or {}),
        committed_at=None if record.committed_at is None else _as_utc(record.committed_at),
    )


def _row_to_domain(record: models.ImportRow) -> ImportRow:
    return ImportRow(
        line_number=record.line_number,
        payload=dict(record.payload or {}),
        outcome=RowOutcome(record.outcome),
        code=record.code,
        message=record.message,
        field=record.field,
    )


__all__ = ["SqlAlchemyImportBatchRepository"]
