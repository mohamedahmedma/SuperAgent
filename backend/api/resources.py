import os
from pathlib import Path

from backend.indexing import (
    DocumentLoader,
    MilvusWriter,
    ParentChunkStore,
    embedding_service,
)
from backend.indexing.milvus_client import get_milvus_store

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

    if job_manager and job_id:
        job_manager.complete_step(job_id, "parent_store", "Parent chunks and Redis cache cleared")

    return chunks_deleted


def is_supported_document(filename: str) -> bool:
    file_lower = filename.lower()
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith((".docx", ".doc"))
        or file_lower.endswith((".xlsx", ".xls"))
        or file_lower.endswith((".html", ".htm"))
    )


async def save_upload_file(file, file_path: Path) -> None:
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def ensure_upload_dir() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
