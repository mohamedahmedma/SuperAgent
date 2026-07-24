"""Text embedding service - dense vectors only (Milvus 2.5+ natively supports Chinese tokenization and BM25 full-text search)"""
import os
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
