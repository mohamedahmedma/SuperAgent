"""Embeds documents and writes them to Milvus - supports dense + sparse vectors.

Index-time deduplication (the last line of defense after parser-level page-furniture
removal): exact duplicates are dropped by normalized-text hash BEFORE the embedding
call (a duplicate never costs an embedding), and — opt-in — near-duplicates are
dropped by embedding cosine similarity. Semantic dedup is disabled by default on
purpose: chunks differing only in numbers (fee rows, term dates) embed almost
identically, and silently losing them is worse than indexing some boilerplate. The
DataProcessing reference (duplicate_tools.py) shipped with it disabled for the same
reason. Enable with SEMANTIC_DEDUP_ENABLED=true, tune SEMANTIC_DEDUP_THRESHOLD.

Dedup scope is one write_documents call (one document upload) — cross-file dedup
would require querying the store and belongs to a future compaction job.
"""
import hashlib
import logging
import os

from backend.indexing.embedding import EmbeddingService, embedding_service as _default_embedding_service
from backend.indexing.milvus_client import MilvusStore, get_milvus_store

logger = logging.getLogger(__name__)


def _normalized_text_key(text: str) -> str:
    normalized = " ".join((text or "").lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _max_cosine_similarity(embedding, kept_embeddings) -> float:
    """Highest cosine similarity between `embedding` and any kept embedding.
    Computes true cosine (not raw dot) so non-normalized embedding services
    still compare correctly."""
    import numpy as np

    if not kept_embeddings:
        return 0.0
    matrix = np.asarray(kept_embeddings, dtype="float32")
    vector = np.asarray(embedding, dtype="float32")
    vector_norm = np.linalg.norm(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * vector_norm
    denom[denom == 0] = 1e-12
    return float(np.max(matrix @ vector / denom))


class MilvusWriter:
    """Service that embeds documents and writes them to Milvus - supports hybrid retrieval"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusStore = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or get_milvus_store()
        self.semantic_dedup_enabled = os.getenv("SEMANTIC_DEDUP_ENABLED", "false").lower() == "true"
        try:
            self.semantic_dedup_threshold = float(os.getenv("SEMANTIC_DEDUP_THRESHOLD", "0.97"))
        except ValueError:
            self.semantic_dedup_threshold = 0.97

    def write_documents(self, documents: list[dict], batch_size: int = 50, progress_callback=None):
        if not documents:
            return

        dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

        total = len(documents)
        self.milvus_manager.init_collection(dense_dim)

        seen_text_keys: set = set()
        kept_embeddings: list = []
        exact_skipped = 0
        semantic_skipped = 0

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]

            unique_batch = []
            for doc in batch:
                key = _normalized_text_key(doc["text"])
                if key in seen_text_keys:
                    exact_skipped += 1
                    continue
                seen_text_keys.add(key)
                unique_batch.append(doc)

            if unique_batch:
                texts = [doc["text"] for doc in unique_batch]
                dense_embeddings = self.embedding_service.get_embeddings(texts)

                insert_pairs = []
                for doc, dense_emb in zip(unique_batch, dense_embeddings):
                    if (
                        self.semantic_dedup_enabled
                        and _max_cosine_similarity(dense_emb, kept_embeddings) >= self.semantic_dedup_threshold
                    ):
                        semantic_skipped += 1
                        continue
                    if self.semantic_dedup_enabled:
                        kept_embeddings.append(dense_emb)
                    insert_pairs.append((doc, dense_emb))

                if insert_pairs:
                    insert_data = [
                        {
                            "dense_embedding": dense_emb,
                            "text": doc["text"],
                            "filename": doc["filename"],
                            "file_type": doc["file_type"],
                            "file_path": doc.get("file_path", ""),
                            "page_number": doc.get("page_number", 0),
                            "chunk_idx": doc.get("chunk_idx", 0),
                            "chunk_id": doc.get("chunk_id", ""),
                            "parent_chunk_id": doc.get("parent_chunk_id", ""),
                            "root_chunk_id": doc.get("root_chunk_id", ""),
                            "chunk_level": doc.get("chunk_level", 0),
                        }
                        for doc, dense_emb in insert_pairs
                    ]

                    self.milvus_manager.insert(insert_data)

            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)

        if exact_skipped or semantic_skipped:
            logger.info(
                "Index-time dedup: skipped %d exact and %d semantic duplicates, wrote %d of %d chunks",
                exact_skipped,
                semantic_skipped,
                total - exact_skipped - semantic_skipped,
                total,
            )
