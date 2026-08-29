"""Persistence for section catalogue entries.

Thin on purpose. The interesting decisions — what a section record contains, when it
may be reused — live in section_summary.py; this only reads and writes them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, List, Optional, Sequence

from sqlalchemy import select

from backend.db.models import CorpusDigest, SectionSummary
from backend.indexing.section_summary import SectionRecord
from backend.infra.database import SessionLocal

logger = logging.getLogger(__name__)


@dataclass
class DigestRecord:
    """The corpus-level description and the floor derived from the same corpus."""

    paragraph: str = ""
    sections_sha256: str = ""
    section_count: int = 0
    floor: float = 0.0
    floor_sha256: str = ""
    question_count: int = 0
    model_used: str = ""


def load_digest(profile: str, session_factory=SessionLocal) -> DigestRecord:
    """The stored digest, or an empty one.

    Empty on failure for the same reason `load_records` returns []: this feeds the
    scope gate, and a database hiccup must cost the gate its description, not the
    request. An empty paragraph makes the caller fall back to the topic list.
    """
    try:
        with session_factory() as session:
            row = session.get(CorpusDigest, profile)
            if row is None:
                return DigestRecord()
            return DigestRecord(
                paragraph=row.paragraph or "",
                sections_sha256=row.sections_sha256 or "",
                section_count=row.section_count or 0,
                floor=float(row.floor or 0.0),
                floor_sha256=row.floor_sha256 or "",
                question_count=row.question_count or 0,
                model_used=row.model_used or "",
            )
    except Exception:
        logger.warning("could not load corpus digest for profile %s", profile, exc_info=True)
        return DigestRecord()


def save_digest(profile: str, record: DigestRecord, session_factory=SessionLocal) -> bool:
    """Upsert the digest. False on failure — the caller logs, the build still counts."""
    try:
        with session_factory() as session:
            row = session.get(CorpusDigest, profile)
            if row is None:
                row = CorpusDigest(profile=profile)
                session.add(row)
            row.paragraph = record.paragraph
            row.sections_sha256 = record.sections_sha256
            row.section_count = record.section_count
            row.floor = record.floor
            row.floor_sha256 = record.floor_sha256
            row.question_count = record.question_count
            row.model_used = record.model_used
            # Naive UTC, matching the timezone-less column. `utcnow()` is deprecated.
            row.updated_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
        return True
    except Exception:
        logger.warning("could not save corpus digest for profile %s", profile, exc_info=True)
        return False


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
            # Naive UTC, matching the timezone-less column. `utcnow()` is deprecated.
            row.updated_at = datetime.now(UTC).replace(tzinfo=None)
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
