"""Text embedding service - dense vectors only (Milvus 2.5+ natively supports Chinese tokenization and BM25 full-text search)"""
import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


def _create_dense_embedder() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


class EmbeddingService:
    """Text embedding service - local dense vector model"""

    def __init__(self, state_path=None):
        self._embedder = _create_dense_embedder()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embedder.embed_documents(texts)
        except Exception as e:
            raise Exception(f"Local dense embedding model call failed: {str(e)}") from e


# Process-wide singleton instance
embedding_service = EmbeddingService()

# A turn can need the query vector more than once — the domain gate classifies with it
# before retrieval searches with it. A bge-m3 forward pass on CPU is the most expensive
# non-network step in a turn, so the second caller must not pay for it again.
#
# Keyed on the already-normalized query text, which is what both callers hold. Small and
# bounded: this is a within-turn memo, not a semantic cache, and stale entries are
# harmless because the same text always embeds to the same vector for a fixed model.
_QUERY_VECTOR_CACHE_SIZE = 64


@lru_cache(maxsize=_QUERY_VECTOR_CACHE_SIZE)
def _embed_query_cached(text: str) -> tuple:
    return tuple(embedding_service.get_embeddings([text])[0])


def embed_query(text: str) -> list[float]:
    """The query's dense vector, computed at most once per distinct text.

    Returns a fresh list each call so a caller mutating it cannot corrupt the memo.
    """
    return list(_embed_query_cached(text))


def reset_query_vector_cache() -> None:
    """For tests and for re-indexing with a different embedding model."""
    _embed_query_cached.cache_clear()
