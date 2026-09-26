"""`SectionSummaryRepository` and `CorpusDigestRepository` over SQLAlchemy.

`SectionRecord` is imported where a record is built, not at the top of the module:
`backend.indexing` runs its package `__init__` — the document loader, Milvus, the
embedder — on any import from it, and opening a unit of work must not cost that.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.application.ports.repositories import DigestRecord
from backend.db.models import CorpusDigest, SectionSummary

if TYPE_CHECKING:
    from backend.indexing.section_summary import SectionRecord


class SqlAlchemySectionSummaryRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def for_profile(self, profile: str) -> Sequence[SectionRecord]:
        rows = self._session.scalars(select(SectionSummary).where(SectionSummary.profile == profile))
        return [_section(row) for row in rows]

    def hashes(self, profile: str) -> dict[str, str]:
        rows = self._session.execute(
            select(SectionSummary.chunk_id, SectionSummary.content_sha256).where(
                SectionSummary.profile == profile
            )
        ).all()
        return {chunk_id: digest for chunk_id, digest in rows}

    def upsert_many(self, profile: str, records: Sequence[SectionRecord]) -> int:
        now = datetime.now(UTC)
        for record in records:
            row = self._session.get(SectionSummary, {"chunk_id": record.chunk_id, "profile": profile})
            if row is None:
                row = SectionSummary(chunk_id=record.chunk_id, profile=profile)
                self._session.add(row)
            row.content_sha256 = record.content_sha256
            row.filename = record.filename
            row.chunk_level = record.chunk_level
            row.summary = record.summary
            row.answers = list(record.answers)
            row.topics = list(record.topics)
            row.question_vectors = [list(vector) for vector in record.question_vectors]
            row.embedding_model = record.embedding_model
            row.model_used = record.model_used
            row.updated_at = now
        self._session.flush()
        return len(records)

    def delete_except(self, profile: str, live_chunk_ids: Collection[str]) -> int:
        statement = delete(SectionSummary).where(SectionSummary.profile == profile)
        live = set(live_chunk_ids)
        if live:
            statement = statement.where(SectionSummary.chunk_id.not_in(live))
        result = self._session.execute(statement.execution_options(synchronize_session=False))
        return int(result.rowcount or 0)


class SqlAlchemyCorpusDigestRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, profile: str) -> DigestRecord | None:
        row = self._session.get(CorpusDigest, profile)
        if row is None:
            return None
        return DigestRecord(
            paragraph=row.paragraph or "",
            sections_sha256=row.sections_sha256 or "",
            section_count=row.section_count or 0,
            floor=float(row.floor or 0.0),
            floor_sha256=row.floor_sha256 or "",
            question_count=row.question_count or 0,
            model_used=row.model_used or "",
        )

    def save(self, profile: str, digest: DigestRecord) -> None:
        row = self._session.get(CorpusDigest, profile)
        if row is None:
            row = CorpusDigest(profile=profile)
            self._session.add(row)
        row.paragraph = digest.paragraph
        row.sections_sha256 = digest.sections_sha256
        row.section_count = digest.section_count
        row.floor = digest.floor
        row.floor_sha256 = digest.floor_sha256
        row.question_count = digest.question_count
        row.model_used = digest.model_used
        row.updated_at = datetime.now(UTC)
        self._session.flush()


def _section(row: SectionSummary) -> SectionRecord:
    from backend.indexing.section_summary import SectionRecord

    return SectionRecord(
        chunk_id=row.chunk_id,
        content_sha256=row.content_sha256,
        filename=row.filename,
        chunk_level=row.chunk_level,
        summary=row.summary,
        answers=list(row.answers or []),
        topics=list(row.topics or []),
        question_vectors=[list(vector) for vector in (row.question_vectors or [])],
        embedding_model=row.embedding_model or "",
        model_used=row.model_used,
    )


__all__ = ["SqlAlchemyCorpusDigestRepository", "SqlAlchemySectionSummaryRepository"]
