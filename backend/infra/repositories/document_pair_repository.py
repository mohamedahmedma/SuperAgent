"""`DocumentPairRepository` over SQLAlchemy.

Storage only. What a pair means — one row per filename, empty rows removed, which half
answers which language — is `DocumentPairService`'s (`backend/indexing/pair_store.py`).
Nothing here commits.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from backend.application.ports.repositories import DocumentPairRecord
from backend.db.models import DocumentPair


class SqlAlchemyDocumentPairRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_all(self) -> Sequence[DocumentPairRecord]:
        rows = self._session.scalars(
            select(DocumentPair).order_by(DocumentPair.created_at.desc(), DocumentPair.pair_id)
        )
        return [_record(row) for row in rows]

    def get(self, pair_id: str) -> DocumentPairRecord | None:
        row = self._session.get(DocumentPair, pair_id) if pair_id else None
        return None if row is None else _record(row)

    def holding(self, filename: str) -> Sequence[DocumentPairRecord]:
        rows = self._session.scalars(
            select(DocumentPair)
            .where(or_(DocumentPair.filename_ar == filename, DocumentPair.filename_en == filename))
            .order_by(DocumentPair.created_at, DocumentPair.pair_id)
        )
        return [_record(row) for row in rows]

    def paired(self) -> Sequence[DocumentPairRecord]:
        rows = self._session.scalars(
            select(DocumentPair)
            .where(DocumentPair.filename_ar != "", DocumentPair.filename_en != "")
            .order_by(DocumentPair.created_at, DocumentPair.pair_id)
        )
        return [_record(row) for row in rows]

    def save(self, pair: DocumentPairRecord) -> None:
        row = self._session.get(DocumentPair, pair.pair_id)
        if row is None:
            row = DocumentPair(pair_id=pair.pair_id)
            self._session.add(row)
        row.title = pair.title
        row.filename_ar = pair.filename_ar
        row.filename_en = pair.filename_en
        row.updated_at = datetime.now(UTC)
        # The sessions this runs on do not autoflush, and the service reads its own writes
        # back within the same transaction.
        self._session.flush()

    def delete_empty(self) -> None:
        self._session.execute(
            delete(DocumentPair).where(DocumentPair.filename_ar == "", DocumentPair.filename_en == "")
        )


def _record(row: DocumentPair) -> DocumentPairRecord:
    return DocumentPairRecord(
        pair_id=row.pair_id,
        title=row.title or "",
        filename_ar=row.filename_ar or "",
        filename_en=row.filename_en or "",
    )


__all__ = ["SqlAlchemyDocumentPairRepository"]
