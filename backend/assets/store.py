"""AssetStore: persistence for asset occurrences and the global extraction cache.

Mirrors ParentChunkStore (Postgres for durability, Redis for hot reads) and adds the one
thing image ingest cannot do without — a content-addressed extraction cache, so the
expensive half of ingest is paid per distinct image rather than per occurrence.

The database is reached through a unit of work, and the blob store and cache are injected
too, so tests run the whole store against a throwaway Postgres schema and a temporary
blob directory.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set

from backend.application.ports.unit_of_work import UnitOfWorkFactory
from backend.assets.dossier import (
    DOSSIER_VERSION,
    AssetDossier,
    AssetRole,
    AssetTier,
    ExtractionPayload,
    ExtractionStatus,
    migrate_payload,
)
from backend.infra.unit_of_work import SqlAlchemyUnitOfWork

logger = logging.getLogger(__name__)


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
        unit_of_work: UnitOfWorkFactory = SqlAlchemyUnitOfWork,
        blob_store=None,
        cache=None,
        cache_enabled: bool = True,
    ):
        self._unit_of_work = unit_of_work
        self._blob_store = blob_store
        self._cache = cache
        self._cache_enabled = cache_enabled

    # -- lazily resolved collaborators -----------------------------------------
    # Resolved on first use so that constructing an AssetStore never instantiates a blob
    # backend or a Redis client as a side effect.

    @property
    def blob_store(self):
        if self._blob_store is None:
            from backend.composition import default_services

            self._blob_store = default_services().blob_store
        return self._blob_store

    @property
    def cache(self):
        if self._cache is None and self._cache_enabled:
            from backend.composition import default_services

            self._cache = default_services().cache
        return self._cache

    @staticmethod
    def _cache_key(asset_id: str) -> str:
        return f"asset:{asset_id}"

    # -- occurrences ------------------------------------------------------------

    def record_many(self, dossiers: Sequence[AssetDossier]) -> int:
        """Upsert asset occurrences. Idempotent on asset_id, so re-ingesting a document
        updates in place instead of duplicating rows."""
        items = [d for d in dossiers if d and d.asset_id]
        if not items:
            return 0

        for dossier in items:
            dossier.touch()
        with self._unit_of_work() as uow:
            uow.document_assets.upsert_many(items)
            uow.commit()

        if self.cache is not None:
            for dossier in items:
                self.cache.set_json(self._cache_key(dossier.asset_id), dossier.model_dump(mode="json"))
        return len(items)

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

        with self._unit_of_work() as uow:
            dossier = uow.document_assets.get(key)
        if dossier is None:
            return None

        if self.cache is not None:
            self.cache.set_json(self._cache_key(key), dossier.model_dump(mode="json"))
        return dossier

    def get_many(self, asset_ids: Iterable[str]) -> List[AssetDossier]:
        ids = [item.strip() for item in asset_ids if item and item.strip()]
        if not ids:
            return []
        with self._unit_of_work() as uow:
            found = {dossier.asset_id: dossier for dossier in uow.document_assets.get_many(ids)}
        return [found[item] for item in ids if item in found]

    def displayable_hashes_by_filename(self, filenames) -> Dict[str, Set[str]]:
        """Content hashes of the images each of `filenames` can actually SHOW.

        Two scalar columns, no dossiers: this runs on the retrieval path (see
        `DocumentPairService.superseded_filenames`, which uses it to avoid excluding a
        translation that is the only side carrying a picture), so it must stay one
        indexed query and must not pay to rebuild an AssetDossier per row.

        "Displayable" is `storage_uri != ""`. A row whose bytes were never stored is a
        figure the presenter can only return as metadata — no picture — so counting it
        would keep a redundant document eligible in exchange for nothing.

        Keyed by sha256 rather than by count because the same image is usually embedded
        in both halves of a translated pair: comparing counts would call those halves
        different, comparing content correctly calls them the same.
        """
        names = [name for name in (filenames or []) if name]
        if not names:
            return {}
        with self._unit_of_work() as uow:
            rows = uow.document_assets.displayable_hashes(names)
        hashes: Dict[str, Set[str]] = {name: set() for name in names}
        for filename, sha256 in rows:
            if sha256:
                hashes[filename].add(sha256)
        return hashes

    def list_by_filename(self, filename: str, indexable_only: bool = False) -> List[AssetDossier]:
        if not filename:
            return []
        with self._unit_of_work() as uow:
            dossiers = list(uow.document_assets.list_by_filename(filename, extracted_only=indexable_only))
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
        return self.find_extractions([sha256], profile, dossier_version).get(sha256)

    def find_extractions(
        self,
        digests: Iterable[str],
        profile: str,
        dossier_version: int = DOSSIER_VERSION,
    ) -> Dict[str, ExtractionPayload]:
        """The same lookup for a whole document, in one query.

        `find_extraction` is called once per image, and a document's digests are all in
        hand before any of them is triaged — so the per-image version paid a round trip
        each to answer a question one `IN` could. On a 400-image catalogue that is 400
        queries before the first model call.

        Digests missing from the result simply have no cache entry; the caller reads this
        as a dict and falls through exactly as it did on a `None`. Corrupt payloads are
        dropped individually rather than failing the batch: one unreadable row must not
        send a whole document back through vision.
        """
        keys = {item for item in digests if item}
        if not keys:
            return {}
        with self._unit_of_work() as uow:
            stored = uow.asset_extractions.find_many(keys, profile, dossier_version)
        found: Dict[str, ExtractionPayload] = {}
        for extraction in stored:
            try:
                found[extraction.sha256] = ExtractionPayload.model_validate(extraction.payload)
            except Exception:
                logger.exception(
                    "Corrupt extraction payload for sha256=%s — ignoring cache", extraction.sha256
                )
        return found

    def save_extraction(
        self,
        sha256: str,
        profile: str,
        payload: ExtractionPayload,
        dossier_version: int = DOSSIER_VERSION,
    ) -> None:
        if not sha256:
            return
        with self._unit_of_work() as uow:
            uow.asset_extractions.save(
                sha256,
                profile,
                dossier_version,
                payload.model_dump(mode="json"),
                model_used=payload.provenance.model_used,
                confidence=float(payload.provenance.confidence or 0.0),
                needs_review=bool(payload.provenance.needs_review),
            )
            uow.commit()

    def mark_reviewed(self, asset_id: str) -> Optional[AssetDossier]:
        """Record that a human accepted this asset's extraction.

        Keyed by the extraction, which is keyed by DIGEST — so accepting one occurrence
        accepts the same bytes wherever else they appear. That is the intended meaning
        rather than a shortcut: the extraction being judged is the same extraction, and
        a logo reviewed on page 1 should not queue itself again on page 40.

        Returns the asset as it now reads, or None if there is no such asset.
        """
        dossier = self.get(asset_id)
        if dossier is None or not dossier.sha256:
            return None
        with self._unit_of_work() as uow:
            # BOTH stores. The extraction is the shared, content-addressed copy; each
            # occurrence keeps its own dossier JSON and that is what `get` hydrates
            # from, so clearing only the extraction leaves every read still flagged.
            uow.asset_extractions.clear_needs_review(
                dossier.sha256, dossier.profile, dossier.dossier_version
            )
            uow.document_assets.clear_needs_review(dossier.sha256)
            uow.commit()
        if dossier.extraction is not None:
            dossier.extraction.provenance.needs_review = False
        return dossier

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

        with self._unit_of_work() as uow:
            removed = uow.document_assets.delete_by_filename(filename)
            if not removed:
                return result
            uow.commit()
            digests = {ref.sha256: ref.storage_uri for ref in removed if ref.sha256}
            still_referenced = uow.document_assets.referenced_digests(digests) if digests else set()

        asset_ids = [ref.asset_id for ref in removed]
        result.assets_deleted = len(asset_ids)

        if self.cache is not None:
            for asset_id in asset_ids:
                self.cache.delete(self._cache_key(asset_id))

        # The attribute index is derived from these assets; leaving rows behind would
        # let a deleted product keep matching catalogue filters.
        try:
            from backend.composition import default_services

            default_services().entity_index.delete_assets(asset_ids)
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
        it scans, and OFFSET over a shifting result set silently skips records. Each
        batch is its own short transaction, so a long backfill never holds one open.
        """
        cursor = ""
        while True:
            with self._unit_of_work() as uow:
                batch = list(
                    uow.document_assets.older_than(target_version, after_asset_id=cursor, limit=batch_size)
                )
            if not batch:
                return
            cursor = batch[-1].asset_id
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
        with self._unit_of_work() as uow:
            by_status = uow.document_assets.status_counts()
            total_assets = uow.document_assets.occurrence_count()
            distinct_digests = uow.document_assets.distinct_image_count()
            extractions = uow.asset_extractions.count()

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


__all__ = [
    "AssetStore",
    "BackfillReport",
    "DeleteResult",
    "AssetRole",
    "AssetTier",
    "ExtractionStatus",
]
