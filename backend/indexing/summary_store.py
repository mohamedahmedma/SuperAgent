"""Persistence for section catalogue entries.

Thin on purpose. The interesting decisions — what a section record contains, when it
may be reused — live in section_summary.py; this only reads and writes them.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from sqlalchemy import select

from backend.db.models import SectionSummary
from backend.indexing.section_summary import SectionRecord
from backend.infra.database import SessionLocal

logger = logging.getLogger(__name__)


def _to_record(row: SectionSummary) -> SectionRecord:
    return SectionRecord(
        chunk_id=row.chunk_id,
        content_sha256=row.content_sha256,
        filename=row.filename,
        chunk_level=row.chunk_level,
        summary=row.summary,
        answers=list(row.answers or []),
        topics=list(row.topics or []),
        question_vectors=[list(v) for v in (row.question_vectors or [])],
        embedding_model=row.embedding_model or "",
        model_used=row.model_used,
    )


def load_records(profile: str, session_factory=SessionLocal) -> List[SectionRecord]:
    """Every catalogued section for a profile. Empty on any failure.

    Failing to an empty list rather than raising is deliberate: the caller is the scope
    gate's reference builder, and an empty reference makes the gate abstain. A database
    hiccup must degrade scope checking, never break the request.
    """
    try:
        with session_factory() as session:
            rows = session.execute(
                select(SectionSummary).where(SectionSummary.profile == profile)
            ).scalars().all()
            return [_to_record(row) for row in rows]
    except Exception:
        logger.warning("could not load section summaries for profile %s", profile, exc_info=True)
        return []


def existing_hashes(profile: str, session_factory=SessionLocal) -> Dict[str, str]:
    """chunk_id -> content hash, for deciding what needs re-summarising."""
    try:
        with session_factory() as session:
            rows = session.execute(
                select(SectionSummary.chunk_id, SectionSummary.content_sha256)
                .where(SectionSummary.profile == profile)
            ).all()
            return {chunk_id: digest for chunk_id, digest in rows}
    except Exception:
        logger.warning("could not read section summary hashes", exc_info=True)
        return {}


def save_records(
    profile: str,
    records: Sequence[SectionRecord],
    session_factory=SessionLocal,
) -> int:
    """Upsert catalogue entries. Returns how many were written."""
    if not records:
        return 0
    written = 0
    with session_factory() as session:
        for record in records:
            row = session.get(SectionSummary, {"chunk_id": record.chunk_id, "profile": profile})
            if row is None:
                row = SectionSummary(chunk_id=record.chunk_id, profile=profile)
                session.add(row)
            row.content_sha256 = record.content_sha256
            row.filename = record.filename
            row.chunk_level = record.chunk_level
            row.summary = record.summary
            row.answers = list(record.answers)
            row.topics = list(record.topics)
            row.question_vectors = [list(v) for v in record.question_vectors]
            row.embedding_model = record.embedding_model
            row.model_used = record.model_used
            row.updated_at = datetime.utcnow()
            written += 1
        session.commit()
    return written


def delete_missing(
    profile: str,
    live_chunk_ids: Sequence[str],
    session_factory=SessionLocal,
) -> int:
    """Drop entries for sections the corpus no longer has.

    Without this, a deleted section keeps voting on scope forever — the gate would go
    on admitting questions about a topic the corpus can no longer answer, and the
    evidence ladder would have to catch every one of them.
    """
    live = set(live_chunk_ids)
    removed = 0
    with session_factory() as session:
        rows = session.execute(
            select(SectionSummary).where(SectionSummary.profile == profile)
        ).scalars().all()
        for row in rows:
            if row.chunk_id not in live:
                session.delete(row)
                removed += 1
        if removed:
            session.commit()
    return removed
