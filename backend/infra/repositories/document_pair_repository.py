"""`DocumentPairRepository` over SQLAlchemy.

Storage only. What a pair means — one row per filename, empty rows removed, which half
answers which language — is `DocumentPairService`'s (`backend/indexing/pair_store.py`).
Nothing here commits.

Reads select the four columns a record carries rather than ORM entities, and a save is a
single Postgres upsert. Routing reads this table on every question; building and
tracking an ORM object per pair only to copy four strings out of it cost more than the
query itself, and a save that looked the row up before writing it paid a round trip the
upsert does not need.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.application.ports.repositories import DocumentPairRecord
from backend.db.models import DocumentPair

_RECORD_COLUMNS = (
    DocumentPair.pair_id,
    DocumentPair.title,
    DocumentPair.filename_ar,
    DocumentPair.filename_en,
)


class SqlAlchemyDocumentPairRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_all(self) -> Sequence[DocumentPairRecord]:
        return self._records(
            select(*_RECORD_COLUMNS).order_by(DocumentPair.created_at.desc(), DocumentPair.pair_id)
        )

    def get(self, pair_id: str) -> DocumentPairRecord | None:
        if not pair_id:
            return None
        records = self._records(select(*_RECORD_COLUMNS).where(DocumentPair.pair_id == pair_id))
        return records[0] if records else None

    def holding(self, filename: str) -> Sequence[DocumentPairRecord]:
        return self._records(
            select(*_RECORD_COLUMNS)
            .where(or_(DocumentPair.filename_ar == filename, DocumentPair.filename_en == filename))
            .order_by(DocumentPair.created_at, DocumentPair.pair_id)
        )

    def paired(self) -> Sequence[DocumentPairRecord]:
        return self._records(
            select(*_RECORD_COLUMNS)
            .where(DocumentPair.filename_ar != "", DocumentPair.filename_en != "")
            .order_by(DocumentPair.created_at, DocumentPair.pair_id)
        )

    def save(self, pair: DocumentPairRecord) -> None:
        now = datetime.now(UTC)
        statement = insert(DocumentPair).values(
            pair_id=pair.pair_id,
            title=pair.title,
            filename_ar=pair.filename_ar,
            filename_en=pair.filename_en,
            created_at=now,
            updated_at=now,
        )
        # Executed immediately, so later reads in the same transaction see it.
        self._session.execute(
            statement.on_conflict_do_update(
                index_elements=[DocumentPair.pair_id],
                set_={
                    name: statement.excluded[name]
                    for name in ("title", "filename_ar", "filename_en", "updated_at")
                },
            )
        )

    def delete_empty(self) -> None:
        self._session.execute(
            delete(DocumentPair)
            .where(DocumentPair.filename_ar == "", DocumentPair.filename_en == "")
            .execution_options(synchronize_session=False)
        )

    def _records(self, statement) -> list[DocumentPairRecord]:
        return [
            DocumentPairRecord(
                pair_id=row.pair_id,
                title=row.title or "",
                filename_ar=row.filename_ar or "",
                filename_en=row.filename_en or "",
            )
            for row in self._session.execute(statement)
        ]


__all__ = ["SqlAlchemyDocumentPairRepository"]
