"""`ParentChunkRepository` over SQLAlchemy.

Writes are Postgres upserts, `INSERT ... ON CONFLICT (chunk_id) DO UPDATE`, one statement
per batch. The store this replaced issued a SELECT and then an INSERT or UPDATE for every
chunk, so a document with a thousand parent chunks cost two thousand round trips, and two
uploads of the same document could race between the SELECT and the INSERT.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.application.ports.repositories import ParentChunkRecord
from backend.db.models import ParentChunk

#: The columns a record carries, selected directly rather than through ORM entities.
_RECORD_COLUMNS = (
    ParentChunk.chunk_id,
    ParentChunk.text,
    ParentChunk.filename,
    ParentChunk.file_type,
    ParentChunk.file_path,
    ParentChunk.page_number,
    ParentChunk.parent_chunk_id,
    ParentChunk.root_chunk_id,
    ParentChunk.chunk_level,
    ParentChunk.chunk_idx,
    ParentChunk.modality,
    ParentChunk.asset_ids,
)


class SqlAlchemyParentChunkRepository:
    #: Rows per INSERT: comfortably under Postgres's limit on bind parameters per statement.
    BATCH_SIZE = 500

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_many(self, chunks: Sequence[ParentChunkRecord]) -> None:
        # One row per id, the last occurrence winning: ON CONFLICT cannot update the same
        # row twice within a single statement.
        latest = {chunk.chunk_id: chunk for chunk in chunks}
        if not latest:
            return
        now = datetime.now(UTC)
        rows = [_values(chunk, now) for chunk in latest.values()]
        for start in range(0, len(rows), self.BATCH_SIZE):
            statement = insert(ParentChunk).values(rows[start:start + self.BATCH_SIZE])
            self._session.execute(
                statement.on_conflict_do_update(
                    index_elements=[ParentChunk.chunk_id],
                    set_={name: statement.excluded[name] for name in rows[0] if name != "chunk_id"},
                )
            )

    def get_many(self, chunk_ids: Sequence[str]) -> Sequence[ParentChunkRecord]:
        if not chunk_ids:
            return []
        rows = self._session.execute(select(*_RECORD_COLUMNS).where(ParentChunk.chunk_id.in_(list(chunk_ids))))
        return [_record(row) for row in rows]

    def delete_by_filename(self, filename: str) -> Sequence[str]:
        removed = self._session.scalars(
            delete(ParentChunk)
            .where(ParentChunk.filename == filename)
            .returning(ParentChunk.chunk_id)
            .execution_options(synchronize_session=False)
        )
        return list(removed)

    def sections(self, level: int) -> Sequence[ParentChunkRecord]:
        rows = self._session.execute(
            select(*_RECORD_COLUMNS)
            .where(ParentChunk.chunk_level == level)
            .order_by(ParentChunk.filename, ParentChunk.chunk_idx)
        )
        return [_record(row) for row in rows]


def _values(chunk: ParentChunkRecord, updated_at: datetime) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "filename": chunk.filename,
        "file_type": chunk.file_type,
        "file_path": chunk.file_path,
        "page_number": chunk.page_number,
        "parent_chunk_id": chunk.parent_chunk_id,
        "root_chunk_id": chunk.root_chunk_id,
        "chunk_level": chunk.chunk_level,
        "chunk_idx": chunk.chunk_idx,
        "modality": chunk.modality,
        "asset_ids": list(chunk.asset_ids),
        "updated_at": updated_at,
    }


def _record(row) -> ParentChunkRecord:
    return ParentChunkRecord(
        chunk_id=row.chunk_id,
        text=row.text,
        filename=row.filename,
        file_type=row.file_type or "",
        file_path=row.file_path or "",
        page_number=row.page_number or 0,
        parent_chunk_id=row.parent_chunk_id or "",
        root_chunk_id=row.root_chunk_id or "",
        chunk_level=row.chunk_level or 0,
        chunk_idx=row.chunk_idx or 0,
        modality=row.modality or "text",
        asset_ids=tuple(row.asset_ids or ()),
    )


__all__ = ["SqlAlchemyParentChunkRepository"]
