"""Removing one document from every store that holds a piece of it.

A document is not one row anywhere. It is leaf vectors in Milvus, parent chunks in
Postgres with their Redis entries, and — when the profile extracts images — asset
occurrences and the blobs behind them. Deleting it means all of those, in an order that
leaves retrieval consistent at every point, which is why it is one class rather than
three callers each remembering a third of the job.

It was `api.resources.delete_document_transactionally`, a module function over objects
built at import. The behaviour is unchanged; what moved is where its collaborators come
from.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from backend.indexing.milvus_client import MilvusStore
    from backend.indexing.parent_chunk_store import ParentChunkStore

logger = logging.getLogger(__name__)


class DocumentRemover:
    """Deletes a document's vectors, its parent chunks, and optionally its assets.

    `asset_store` is a callable rather than the store itself: a profile with images
    disabled never needs one, and resolving it eagerly would open a blob backend for a
    deployment that has no assets at all.
    """

    def __init__(
        self,
        *,
        milvus: "MilvusStore",
        parent_chunks: "ParentChunkStore",
        asset_store: Callable[[], object],
        profile: Optional[Callable[[], object]] = None,
    ) -> None:
        self._milvus = milvus
        self._parent_chunks = parent_chunks
        self._asset_store = asset_store
        self._profile = profile

    def _current_profile(self):
        if self._profile is not None:
            return self._profile()
        from backend.profiles import get_profile

        return get_profile()

    def remove(
        self,
        filename: str,
        job_manager=None,
        job_id=None,
        include_assets: bool = True,
    ) -> int:
        """
        Consistently and transactionally delete all data associated with a document
        (Milvus 2.5+ maintains BM25 index statistics automatically on the server side).
        Steps:
        1. Initialize the Milvus collection.
        2. Delete the Milvus vector data.
        3. Delete the L1/L2 parent chunks in PostgreSQL and the corresponding Redis cache.

        `include_assets=False` keeps step 3's asset occurrences. It exists for a caller
        that replaces a document AFTER parsing it, because parsing is not side-effect-free:
        figure enrichment runs inside `load_document` and commits this filename's
        `document_assets` rows before the replace is reached, so a cleanup that ran
        afterwards would delete the rows the parse had just written. Skipping the delete is
        safe rather than merely convenient — `build_asset_id` is deterministic in
        (filename, page_number, image bytes), so re-ingesting a document regenerates the
        same ids and `record_many` upserts them in place. The one residue is a replacement
        that DROPS an image, whose row lingers addressed by a digest no chunk now
        references; `delete_by_filename` keeps the extraction cache regardless, so nothing
        expensive is at stake either way.

        That residue used to be worse than lingering. While the id was the image's ordinal,
        a replacement that added or removed a figure renumbered every later one, so a
        surplus row was not merely unreferenced — its id was handed to a DIFFERENT image,
        and an anchor stored in an older answer resolved to the wrong picture.
        """
        if job_manager and job_id:
            job_manager.update_step(job_id, "prepare", 50, "running", "Initializing Milvus collection")

        self._milvus.init_collection()
        delete_expr = f'filename == "{filename}"'

        if job_manager and job_id:
            job_manager.complete_step(job_id, "prepare", "Preparation complete")
            # Kept for compatibility with the existing frontend deletion steps
            job_manager.update_step(job_id, "bm25", 100, "completed", "BM25 full-text search statistics synced automatically (maintained server-side by Milvus)")

        # Delete Milvus vectors
        if job_manager and job_id:
            job_manager.update_step(job_id, "milvus", 20, "running", "Physically deleting vector chunks in Milvus")

        chunks_deleted = 0
        try:
            result = self._milvus.delete(delete_expr)
            chunks_deleted = result.get("delete_count", 0) if isinstance(result, dict) else 0
        except Exception as e:
            raise RuntimeError(f"Failed to delete Milvus vectors: {str(e)}") from e

        if job_manager and job_id:
            job_manager.complete_step(job_id, "milvus", f"Vector data cleanup complete, {chunks_deleted} records deleted")

        # Delete ParentChunk rows in Postgres and the Redis cache
        if job_manager and job_id:
            job_manager.update_step(job_id, "parent_store", 20, "running", "Cleaning up parent chunks in the PostgreSQL database and Redis")

        try:
            self._parent_chunks.delete_by_filename(filename)
        except Exception as e:
            raise RuntimeError(f"Failed to clean up PostgreSQL parent chunks and cache: {str(e)}") from e

        # Asset occurrences and their orphaned blobs. Reported inside the existing
        # parent_store step rather than as a new one, so the delete job's step contract
        # (DELETE_STEPS, rendered by the frontend) stays unchanged.
        asset_summary = ""
        profile = self._current_profile()
        if include_assets and profile.assets.enabled:
            try:
                deleted = self._asset_store().delete_by_filename(
                    filename, gc_orphan_blobs=profile.assets.gc_orphan_blobs
                )
                if deleted.assets_deleted:
                    asset_summary = (
                        f", {deleted.assets_deleted} assets removed "
                        f"({deleted.blobs_deleted} blobs freed, {deleted.blobs_retained} still shared)"
                    )
            except Exception as e:
                # Vectors and parent chunks are already gone, so retrieval is consistent.
                # Losing asset rows here is storage housekeeping, not a correctness
                # problem, and must not fail a delete the user already saw succeed.
                logger.exception("Asset cleanup failed for %s", filename)
                asset_summary = f", asset cleanup deferred ({e})"

        if job_manager and job_id:
            job_manager.complete_step(
                job_id, "parent_store", f"Parent chunks and Redis cache cleared{asset_summary}"
            )

        return chunks_deleted


__all__ = ["DocumentRemover"]
