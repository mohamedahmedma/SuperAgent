import logging
import os
from pathlib import Path

from backend.indexing import (
    DocumentLoader,
    MilvusWriter,
    ParentChunkStore,
    embedding_service,
)
from backend.indexing.milvus_client import get_milvus_store
from backend.profiles import get_profile

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = get_milvus_store()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)


def delete_document_transactionally(filename: str, job_manager=None, job_id=None) -> int:
    """
    Consistently and transactionally delete all data associated with a document
    (Milvus 2.5+ maintains BM25 index statistics automatically on the server side).
    Steps:
    1. Initialize the Milvus collection.
    2. Delete the Milvus vector data.
    3. Delete the L1/L2 parent chunks in PostgreSQL and the corresponding Redis cache.
    """
    if job_manager and job_id:
        job_manager.update_step(job_id, "prepare", 50, "running", "Initializing Milvus collection")

    milvus_manager.init_collection()
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
        result = milvus_manager.delete(delete_expr)
        chunks_deleted = result.get("delete_count", 0) if isinstance(result, dict) else 0
    except Exception as e:
        raise RuntimeError(f"Failed to delete Milvus vectors: {str(e)}") from e

    if job_manager and job_id:
        job_manager.complete_step(job_id, "milvus", f"Vector data cleanup complete, {chunks_deleted} records deleted")

    # Delete ParentChunk rows in Postgres and the Redis cache
    if job_manager and job_id:
        job_manager.update_step(job_id, "parent_store", 20, "running", "Cleaning up parent chunks in the PostgreSQL database and Redis")

    try:
        parent_chunk_store.delete_by_filename(filename)
    except Exception as e:
        raise RuntimeError(f"Failed to clean up PostgreSQL parent chunks and cache: {str(e)}") from e

    # Asset occurrences and their orphaned blobs. Reported inside the existing
    # parent_store step rather than as a new one, so the delete job's step contract
    # (DELETE_STEPS, rendered by the frontend) stays unchanged.
    asset_summary = ""
    profile = get_profile()
    if profile.assets.enabled:
        try:
            from backend.assets.store import get_asset_store

            deleted = get_asset_store().delete_by_filename(
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


def is_supported_document(filename: str) -> bool:
    """Accepted upload extensions come from the profile: an e-commerce catalogue and a
    document KB do not necessarily ingest the same file types."""
    file_lower = filename.lower()
    extensions = tuple(get_profile().ingest.supported_extensions)
    return bool(extensions) and file_lower.endswith(extensions)


async def save_upload_file(file, file_path: Path) -> None:
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def ensure_upload_dir() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
