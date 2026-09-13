"""Persistence for the scope catalogue: section entries and the corpus digest.

Thin on purpose. The interesting decisions — what a section record contains, when it
may be reused — live in section_summary.py; this only reads and writes them.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from backend.application.ports.repositories import DigestRecord
from backend.application.ports.unit_of_work import UnitOfWorkFactory
from backend.indexing.section_summary import SectionRecord
from backend.infra.unit_of_work import SqlAlchemyUnitOfWork

logger = logging.getLogger(__name__)


class SectionCatalogueStore:
    """The catalogue a scope gate is built from, one profile at a time."""

    def __init__(self, unit_of_work: UnitOfWorkFactory = SqlAlchemyUnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def load_digest(self, profile: str) -> DigestRecord:
        """The stored digest, or an empty one.

        Empty on failure for the same reason `load_records` returns []: this feeds the
        scope gate, and a database hiccup must cost the gate its description, not the
        request. An empty paragraph makes the caller fall back to the topic list.
        """
        try:
            with self._unit_of_work() as uow:
                return uow.corpus_digests.get(profile) or DigestRecord()
        except Exception:
            logger.warning("could not load corpus digest for profile %s", profile, exc_info=True)
            return DigestRecord()

    def save_digest(self, profile: str, record: DigestRecord) -> bool:
        """Upsert the digest. False on failure — the caller logs, the build still counts."""
        try:
            with self._unit_of_work() as uow:
                uow.corpus_digests.save(profile, record)
                uow.commit()
            return True
        except Exception:
            logger.warning("could not save corpus digest for profile %s", profile, exc_info=True)
            return False

    def load_records(self, profile: str) -> List[SectionRecord]:
        """Every catalogued section for a profile. Empty on any failure.

        Failing to an empty list rather than raising is deliberate: the caller is the scope
        gate's reference builder, and an empty reference makes the gate abstain. A database
        hiccup must degrade scope checking, never break the request.
        """
        try:
            with self._unit_of_work() as uow:
                return list(uow.section_summaries.for_profile(profile))
        except Exception:
            logger.warning("could not load section summaries for profile %s", profile, exc_info=True)
            return []

    def existing_hashes(self, profile: str) -> Dict[str, str]:
        """chunk_id -> content hash, for deciding what needs re-summarising."""
        try:
            with self._unit_of_work() as uow:
                return dict(uow.section_summaries.hashes(profile))
        except Exception:
            logger.warning("could not read section summary hashes", exc_info=True)
            return {}

    def save_records(self, profile: str, records: Sequence[SectionRecord]) -> int:
        """Upsert catalogue entries. Returns how many were written."""
        if not records:
            return 0
        with self._unit_of_work() as uow:
            written = uow.section_summaries.upsert_many(profile, records)
            uow.commit()
        return written

    def delete_missing(self, profile: str, live_chunk_ids: Sequence[str]) -> int:
        """Drop entries for sections the corpus no longer has.

        Without this, a deleted section keeps voting on scope forever — the gate would go
        on admitting questions about a topic the corpus can no longer answer, and the
        evidence ladder would have to catch every one of them.
        """
        with self._unit_of_work() as uow:
            removed = uow.section_summaries.delete_except(profile, live_chunk_ids)
            if removed:
                uow.commit()
        return removed

