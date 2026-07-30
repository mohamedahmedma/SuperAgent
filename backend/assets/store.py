"""AssetStore: persistence for asset occurrences and the global extraction cache.

Mirrors the ParentChunkStore pattern (Postgres for durability, Redis for hot reads)
and adds the one thing image ingest cannot do without — a content-addressed extraction
cache, so the expensive half of ingest is paid per distinct image rather than per
occurrence.

The session factory and blob store are injected rather than imported, so the whole
repository is exercisable against in-memory SQLite in tests without touching Postgres.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from sqlalchemy import func

from backend.assets.dossier import (
    DOSSIER_VERSION,
    AssetDossier,
    AssetRole,
    AssetTier,
    ExtractionPayload,
    ExtractionStatus,
    migrate_payload,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Naive UTC, matching the timezone-less DateTime columns these rows write to.
    `datetime.utcnow()` would do the same but is deprecated."""
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class DeleteResult:
    assets_deleted: int = 0
    blobs_deleted: int = 0
    blobs_retained: int = 0


@dataclass
class BackfillReport:
    scanned: int = 0
    migrated: int = 0
    marked_stale: int = 0
    failed: int = 0
    by_version: Dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "migrated": self.migrated,
            "marked_stale": self.marked_stale,
            "failed": self.failed,
            "by_version": dict(self.by_version),
        }


class AssetStore:
    def __init__(
        self,
        session_factory: Optional[Callable] = None,
        blob_store=None,
        cache=None,
        cache_enabled: bool = True,
    ):
        self._session_factory = session_factory
        self._blob_store = blob_store
        self._cache = cache
        self._cache_enabled = cache_enabled

    # -- lazily resolved collaborators -----------------------------------------
    # Imported on first use so that constructing an AssetStore (which happens at
    # module import in api/resources.py) never opens a database connection or
    # instantiates a blob backend as a side effect.

    @property
    def session_factory(self) -> Callable:
        if self._session_factory is None:
            from backend.infra.database import SessionLocal

            self._session_factory = SessionLocal
        return self._session_factory

    @property
    def blob_store(self):
        if self._blob_store is None:
            from backend.assets.blobs import get_blob_store

            self._blob_store = get_blob_store()
        return self._blob_store

    @property
    def cache(self):
        if self._cache is None and self._cache_enabled:
            from backend.infra.cache import cache

            self._cache = cache
        return self._cache

    @staticmethod
    def _models():
        from backend.db.models import AssetExtraction, DocumentAsset

        return DocumentAsset, AssetExtraction

    @staticmethod
    def _cache_key(asset_id: str) -> str:
        return f"asset:{asset_id}"

    # -- occurrences ------------------------------------------------------------

    @staticmethod
    def _to_row_values(dossier: AssetDossier) -> dict:
        return {
            "sha256": dossier.sha256,
            "profile": dossier.profile,
            "dossier_version": dossier.dossier_version,
            "filename": dossier.source.filename,
            "page_number": int(dossier.source.page_number or 0),
            "role": dossier.role.value,
            "tier": dossier.tier.value,
            "status": dossier.status.value,
            "storage_uri": dossier.blob.uri,
            "content_type": dossier.blob.content_type,
            "byte_size": int(dossier.blob.byte_size or 0),
            "width": int(dossier.blob.width or 0),
            "height": int(dossier.blob.height or 0),
            "dossier": dossier.model_dump(mode="json"),
            "updated_at": _utcnow(),
        }

    @staticmethod
    def _from_row(row) -> AssetDossier:
        return AssetDossier.model_validate(row.dossier)

    def record_many(self, dossiers: Sequence[AssetDossier]) -> int:
        """Upsert asset occurrences. Idempotent on asset_id, so re-ingesting a document
        updates in place instead of duplicating rows."""
        items = [d for d in dossiers if d and d.asset_id]
        if not items:
            return 0

        DocumentAsset, _ = self._models()
        session = self.session_factory()
        written = 0
        try:
            existing = {
                row.asset_id: row
                for row in session.query(DocumentAsset)
                .filter(DocumentAsset.asset_id.in_([d.asset_id for d in items]))
                .all()
            }
            for dossier in items:
                dossier.touch()
                values = self._to_row_values(dossier)
                row = existing.get(dossier.asset_id)
                if row is not None:
                    for key, value in values.items():
                        setattr(row, key, value)
                else:
                    session.add(DocumentAsset(asset_id=dossier.asset_id, **values))
                written += 1
            session.commit()
        finally:
            session.close()

        if self.cache is not None:
            for dossier in items:
                self.cache.set_json(self._cache_key(dossier.asset_id), dossier.model_dump(mode="json"))
        return written

    def record(self, dossier: AssetDossier) -> AssetDossier:
        self.record_many([dossier])
        return dossier

    def get(self, asset_id: str) -> Optional[AssetDossier]:
        key = (asset_id or "").strip()
        if not key:
            return None

        if self.cache is not None:
            cached = self.cache.get_json(self._cache_key(key))
            if cached:
                try:
                    return AssetDossier.model_validate(cached)
                except Exception:
                    # A cached payload written by an older schema must never take down
                    # a read; fall through to the database, which is authoritative.
                    logger.warning("Discarding unreadable cached dossier for %s", key)

        DocumentAsset, _ = self._models()
        session = self.session_factory()
        try:
            row = session.query(DocumentAsset).filter(DocumentAsset.asset_id == key).first()
            if row is None:
                return None
            dossier = self._from_row(row)
        finally:
            session.close()

        if self.cache is not None:
            self.cache.set_json(self._cache_key(key), dossier.model_dump(mode="json"))
        return dossier

    def get_many(self, asset_ids: Iterable[str]) -> List[AssetDossier]:
        ids = [item.strip() for item in asset_ids if item and item.strip()]
        if not ids:
            return []
        DocumentAsset, _ = self._models()
        session = self.session_factory()
        try:
            rows = session.query(DocumentAsset).filter(DocumentAsset.asset_id.in_(ids)).all()
            found = {row.asset_id: self._from_row(row) for row in rows}
        finally:
            session.close()
        return [found[item] for item in ids if item in found]

    def list_by_filename(self, filename: str, indexable_only: bool = False) -> List[AssetDossier]:
        if not filename:
            return []
        DocumentAsset, _ = self._models()
        session = self.session_factory()
        try:
            query = session.query(DocumentAsset).filter(DocumentAsset.filename == filename)
            if indexable_only:
                query = query.filter(DocumentAsset.status == ExtractionStatus.EXTRACTED.value)
            rows = query.order_by(DocumentAsset.page_number, DocumentAsset.asset_id).all()
            dossiers = [self._from_row(row) for row in rows]
        finally:
            session.close()
        return [d for d in dossiers if d.is_indexable] if indexable_only else dossiers

    # -- extraction cache -------------------------------------------------------

    def find_extraction(
        self,
        sha256: str,
        profile: str,
        dossier_version: int = DOSSIER_VERSION,
    ) -> Optional[ExtractionPayload]:
        """The cache lookup that makes repeat images free. Called before any model."""
        if not sha256:
            return None
        _, AssetExtraction = self._models()
        session = self.session_factory()
        try:
            row = (
                session.query(AssetExtraction)
                .filter(
                    AssetExtraction.sha256 == sha256,
                    AssetExtraction.profile == profile,
                    AssetExtraction.dossier_version == dossier_version,
                )
                .first()
            )
            if row is None:
                return None
            try:
                return ExtractionPayload.model_validate(row.payload)
            except Exception:
                logger.exception("Corrupt extraction payload for sha256=%s — ignoring cache", sha256)
                return None
        finally:
            session.close()

    def save_extraction(
        self,
        sha256: str,
        profile: str,
        payload: ExtractionPayload,
        dossier_version: int = DOSSIER_VERSION,
    ) -> None:
        if not sha256:
            return
        _, AssetExtraction = self._models()
        session = self.session_factory()
        try:
            row = (
                session.query(AssetExtraction)
                .filter(
                    AssetExtraction.sha256 == sha256,
                    AssetExtraction.profile == profile,
                    AssetExtraction.dossier_version == dossier_version,
                )
                .first()
            )
            values = {
                "payload": payload.model_dump(mode="json"),
                "model_used": payload.provenance.model_used,
                "confidence": float(payload.provenance.confidence or 0.0),
                "needs_review": bool(payload.provenance.needs_review),
                "updated_at": _utcnow(),
            }
            if row is not None:
                for key, value in values.items():
                    setattr(row, key, value)
            else:
                session.add(
                    AssetExtraction(
                        sha256=sha256,
                        profile=profile,
                        dossier_version=dossier_version,
                        **values,
                    )
                )
            session.commit()
        finally:
            session.close()

    def attach_cached_extraction(self, dossier: AssetDossier) -> bool:
        """Populate a pending dossier from the cache. True when the caller can skip
        extraction entirely — the single most valuable branch in the ingest path."""
        if dossier.extraction is not None:
            return False
        cached = self.find_extraction(dossier.sha256, dossier.profile, dossier.dossier_version)
        if cached is None:
            return False
        dossier.extraction = cached
        dossier.status = ExtractionStatus.EXTRACTED
        return True

    # -- deletion ---------------------------------------------------------------

    def delete_by_filename(self, filename: str, gc_orphan_blobs: bool = True) -> DeleteResult:
        """Remove a document's asset occurrences.

        Blobs are content-addressed and shared, so one is deleted only once no other
        document references its digest. Extraction rows are deliberately KEPT: they are
        derived text keyed by an irreversible digest, they are the expensive artifact,
        and re-uploading the same document is the common case.
        """
        result = DeleteResult()
        if not filename:
            return result

        DocumentAsset, _ = self._models()
        session = self.session_factory()
        try:
            rows = session.query(DocumentAsset).filter(DocumentAsset.filename == filename).all()
            if not rows:
                return result

            asset_ids = [row.asset_id for row in rows]
            digests = {row.sha256: row.storage_uri for row in rows if row.sha256}

            session.query(DocumentAsset).filter(DocumentAsset.filename == filename).delete(
                synchronize_session=False
            )
            session.commit()
            result.assets_deleted = len(asset_ids)

            still_referenced = {
                row.sha256
                for row in session.query(DocumentAsset.sha256)
                .filter(DocumentAsset.sha256.in_(list(digests)))
                .distinct()
                .all()
            } if digests else set()
        finally:
            session.close()

        if self.cache is not None:
            for asset_id in asset_ids:
                self.cache.delete(self._cache_key(asset_id))

        # The attribute index is derived from these assets; leaving rows behind would
        # let a deleted product keep matching catalogue filters.
        try:
            from backend.assets.entity_store import get_entity_index

            get_entity_index().delete_assets(asset_ids)
        except Exception:
            logger.exception("Failed to clear the attribute index for %s", filename)

        for digest, uri in digests.items():
            if digest in still_referenced:
                result.blobs_retained += 1
                continue
            if not gc_orphan_blobs or not uri:
                result.blobs_retained += 1
                continue
            try:
                if self.blob_store.delete(uri):
                    result.blobs_deleted += 1
            except Exception:
                # A blob that will not delete must not fail the document deletion —
                # the index is already consistent; this is storage housekeeping.
                logger.exception("Failed to delete orphaned blob %s", uri)

        return result

    # -- maintenance ------------------------------------------------------------

    def iter_stale(
        self,
        target_version: int = DOSSIER_VERSION,
        batch_size: int = 200,
    ) -> Iterator[List[AssetDossier]]:
        """Yield batches of occurrences older than `target_version`.

        Keyset pagination on asset_id rather than OFFSET: the backfill mutates the rows
        it scans, and OFFSET over a shifting result set silently skips records.
        """
        DocumentAsset, _ = self._models()
        cursor = ""
        while True:
            session = self.session_factory()
            try:
                rows = (
                    session.query(DocumentAsset)
                    .filter(
                        DocumentAsset.dossier_version < target_version,
                        DocumentAsset.asset_id > cursor,
                    )
                    .order_by(DocumentAsset.asset_id)
                    .limit(batch_size)
                    .all()
                )
                if not rows:
                    return
                cursor = rows[-1].asset_id
                batch = [self._from_row(row) for row in rows]
            finally:
                session.close()
            yield batch

    def backfill(
        self,
        target_version: int = DOSSIER_VERSION,
        batch_size: int = 200,
        dry_run: bool = False,
    ) -> BackfillReport:
        """Walk stale occurrences up to `target_version`.

        Migrations that only reshape stored data are applied in place. Migrations that
        need fresh model output cannot be honoured offline, so those rows are marked
        STALE for the extraction pipeline to pick up — a schema change never invents
        data it does not have.
        """
        report = BackfillReport()

        for batch in self.iter_stale(target_version=target_version, batch_size=batch_size):
            updated: List[AssetDossier] = []
            for dossier in batch:
                report.scanned += 1
                report.by_version[dossier.dossier_version] = (
                    report.by_version.get(dossier.dossier_version, 0) + 1
                )
                try:
                    payload, needs_reextraction = migrate_payload(
                        dossier.model_dump(mode="json"), target_version
                    )
                    migrated = AssetDossier.model_validate(payload)
                except Exception:
                    logger.exception("Backfill failed for asset %s", dossier.asset_id)
                    report.failed += 1
                    continue

                if needs_reextraction:
                    migrated.status = ExtractionStatus.STALE
                    report.marked_stale += 1
                else:
                    report.migrated += 1
                updated.append(migrated)

            if updated and not dry_run:
                self.record_many(updated)

        return report

    def stats(self) -> dict:
        """Operational counters: how much of the corpus is extracted, stale, or failed."""
        DocumentAsset, AssetExtraction = self._models()
        session = self.session_factory()
        try:
            by_status: Dict[str, int] = {}
            for status, count in (
                session.query(DocumentAsset.status, func.count(DocumentAsset.asset_id))
                .group_by(DocumentAsset.status)
                .all()
            ):
                by_status[status] = int(count)

            total_assets = session.query(DocumentAsset).count()
            distinct_digests = session.query(DocumentAsset.sha256).distinct().count()
            extractions = session.query(AssetExtraction).count()
        finally:
            session.close()

        return {
            "assets": total_assets,
            "distinct_images": distinct_digests,
            "cached_extractions": extractions,
            # >1.0 means the cache is paying for itself: that many occurrences are
            # being served per image actually extracted.
            "dedup_ratio": round(total_assets / distinct_digests, 3) if distinct_digests else 0.0,
            "by_status": by_status,
            "dossier_version": DOSSIER_VERSION,
        }


_store: Optional[AssetStore] = None


def get_asset_store() -> AssetStore:
    global _store
    if _store is None:
        _store = AssetStore()
    return _store


def set_asset_store(store: Optional[AssetStore]) -> None:
    global _store
    _store = store


__all__ = [
    "AssetStore",
    "BackfillReport",
    "DeleteResult",
    "get_asset_store",
    "set_asset_store",
    "AssetRole",
    "AssetTier",
    "ExtractionStatus",
]
