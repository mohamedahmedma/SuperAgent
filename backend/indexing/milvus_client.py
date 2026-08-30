"""Milvus access layer: stateless Store + short-lived gRPC connections (avoids holding stale channels long-term)."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker, Function, FunctionType

QUERY_MAX_LIMIT = 16384
T = TypeVar("T")

# Analyzer driving BM25 tokenization of the `bm25_text` field (the sparse half of
# hybrid retrieval). "standard" uses Unicode word-boundary segmentation, which is
# correct for Arabic, English, and most scripts; "chinese" (jieba) is only right
# for a CJK corpus and tokenizes Arabic poorly, degrading keyword recall.
# NOTE: this is schema-level — changing it only affects NEWLY created
# collections; an existing collection must be rebuilt for the change to apply.
TEXT_ANALYZER_TYPE = os.getenv("MILVUS_TEXT_ANALYZER", "standard").strip() or "standard"


# Interrogative/filler words. The stock "_english_" stop list is built for prose
# search and keeps every one of these, but in a Q&A system they open almost every
# query while carrying no signal about which chunk answers it — on a small corpus
# their IDF stays high enough to pull in unrelated chunks. Stripped from both the
# index and the query, since one analyzer serves both.
_QUESTION_STOP_WORDS = [
    "what", "how", "can", "i", "my", "do", "does", "did", "which", "when",
    "where", "who", "why", "should", "would", "could", "me", "you", "your",
    "am", "get", "need", "want", "please", "there", "any", "about", "tell",
]


def build_analyzer_params() -> dict:
    """BM25 analyzer config.

    The bare "standard" tokenizer indexes every surface form of every word,
    stop words included: "what", "can", "the" become terms, and on a corpus this
    small their IDF is not low enough to stop them dragging in unrelated chunks.
    Adding lowercase + stop-word removal + stemming means "payments" and "pay"
    hit the same term and function words stop scoring at all.

    The English stemmer/stop list is a no-op on Arabic tokens (it only strips
    ASCII suffixes), so this stays correct for the mixed-script corpus.

    Arabic is folded and light-stemmed BEFORE it reaches Milvus, by
    `backend.text_matching.search_key` — applied to `bm25_text` on the way in and to the
    sparse query on the way out. It is done there rather than as an analyzer filter for
    two reasons: Milvus ships no Arabic filter chain to configure, and doing it in
    Python makes the index side and the query side one tested function instead of two
    server-side behaviours that have to be trusted to agree. What is left for the
    analyzer is the stop list, which is genuinely symmetric here because Milvus applies
    one list to indexed text and query alike.
    """
    if TEXT_ANALYZER_TYPE == "chinese":
        return {"type": "chinese"}
    # Imported here rather than at module scope: text_matching pulls in camel-tools and
    # snowballstemmer, and this module is imported by tooling (schema checks, admin
    # scripts) that never builds an analyzer.
    from backend.text_matching import arabic_stop_words_for_analyzer

    return {
        "tokenizer": "standard",
        "filter": [
            "lowercase",
            {
                "type": "stop",
                "stop_words": (
                    ["_english_"] + _QUESTION_STOP_WORDS + arabic_stop_words_for_analyzer()
                ),
            },
            {"type": "stemmer", "language": "english"},
        ],
    }


@dataclass(frozen=True)
class MilvusSettings:
    host: str
    port: str
    collection_name: str
    uri: str
    timeout: float

    @classmethod
    def from_env(cls) -> MilvusSettings:
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        collection = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        timeout = float(os.getenv("MILVUS_TIMEOUT", "30"))
        return cls(
            host=host,
            port=port,
            collection_name=collection,
            uri=f"http://{host}:{port}",
            timeout=timeout,
        )


@contextmanager
def milvus_client_session(settings: MilvusSettings | None = None) -> Iterator[MilvusClient]:
    """A single RPC session: opens a connection, closes it when done, and does not cache the gRPC channel."""
    cfg = settings or MilvusSettings.from_env()
    client = MilvusClient(uri=cfg.uri, timeout=cfg.timeout)
    try:
        yield client
    finally:
        client.close()


def _decode_asset_ids(raw) -> list[str]:
    """asset_ids is stored as a JSON array in a VARCHAR column. Malformed or legacy
    values decode to an empty list rather than breaking a search result."""
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(item) for item in decoded if item] if isinstance(decoded, list) else []


def _normalize_filter(filter_expr: str) -> str:
    return filter_expr.strip() if filter_expr.strip() else "id >= 0"


class MilvusStore:
    """Milvus collection read/write; holds no connection itself -- all IO goes through milvus_client_session."""

    def __init__(self, settings: MilvusSettings | None = None):
        self._settings = settings or MilvusSettings.from_env()

    @property
    def collection_name(self) -> str:
        return self._settings.collection_name

    def _run(self, operation: Callable[[MilvusClient], T]) -> T:
        with milvus_client_session(self._settings) as client:
            return operation(client)

    @contextmanager
    def session(self) -> Iterator[MilvusClient]:
        """Reuses a single connection within one business flow (e.g., an entire upload), closing it when done."""
        with milvus_client_session(self._settings) as client:
            yield client

    @staticmethod
    def ensure_collection(client: MilvusClient, collection_name: str, dense_dim: int) -> None:
        if client.has_collection(collection_name):
            return

        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        # `text` is what the Agent and citations read: it keeps the section-path
        # prefix for topical context. It is NOT the BM25 input — that prefix
        # repeats on nearly every chunk of a document, so indexing it made common
        # query words match the whole corpus at near-identical scores and swamped
        # the genuine dense hits during RRF fusion.
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        # `bm25_text` is the sparse half's input: same body, document-root
        # heading stripped. See DocumentLoader._apply_bm25_section_prefix.
        schema.add_field(
            "bm25_text",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params=build_analyzer_params(),
            enable_match=True,
        )
        schema.add_field("filename", DataType.VARCHAR, max_length=255)
        schema.add_field("file_type", DataType.VARCHAR, max_length=50)
        schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_level", DataType.INT64)
        # "text" | "figure" | "table" — lets a query restrict to (or exclude) chunks
        # that came from an image without inspecting their content.
        schema.add_field("modality", DataType.VARCHAR, max_length=20)
        # JSON array of asset_ids. A figure chunk points at the image that produced
        # it, so retrieval can show the picture beside the text that matched.
        schema.add_field("asset_ids", DataType.VARCHAR, max_length=2048)

        bm25_function = Function(
            name="text_bm25_emb",
            function_type=FunctionType.BM25,
            input_field_names=["bm25_text"],
            output_field_names=["sparse_embedding"],
        )
        schema.add_function(bm25_function)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_embedding",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 256},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"drop_ratio_build": 0.2},
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def init_collection(self, dense_dim: int | None = None) -> None:
        if dense_dim is None:
            dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

        def _init(client: MilvusClient) -> None:
            self.ensure_collection(client, self.collection_name, dense_dim)

        self._run(_init)

    def insert(self, data: list[dict]):
        return self._run(lambda client: client.insert(self.collection_name, data))

    def query(
        self,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
        limit: int = 10000,
        offset: int = 0,
    ):
        expr = _normalize_filter(filter_expr)
        fields = output_fields or ["filename", "file_type"]

        def _query(client: MilvusClient):
            return client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=fields,
                limit=min(limit, QUERY_MAX_LIMIT),
                offset=offset,
            )

        return self._run(_query)

    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        """Paginated fetch; completed within a single session to avoid opening a new connection per page."""
        fields = output_fields or ["filename", "file_type"]
        expr = _normalize_filter(filter_expr)

        def _query_all(client: MilvusClient) -> list:
            out: list = []
            offset = 0
            while True:
                batch = client.query(
                    collection_name=self.collection_name,
                    filter=expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                    offset=offset,
                )
                if not batch:
                    break
                out.extend(batch)
                if len(batch) < QUERY_MAX_LIMIT:
                    break
                offset += len(batch)
            return out

        return self._run(_query_all)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []
        quoted_ids = ", ".join(f'"{item}"' for item in ids)
        return self.query(
            filter_expr=f"chunk_id in [{quoted_ids}]",
            output_fields=[
                "text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_idx",
                "modality",
                "asset_ids",
            ],
            limit=len(ids),
        )

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = "",
    ) -> list[dict]:
        output_fields = [
            "text",
            "filename",
            "file_type",
            "page_number",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
            "modality",
            "asset_ids",
        ]
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        sparse_search = AnnSearchRequest(
            data=[query],
            anns_field="sparse_embedding",
            # drop_ratio_search=0 keeps low-weight sparse terms: rare/exact
            # keywords are precisely the ones a 0.2 ratio discards.
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        reranker = RRFRanker(k=rrf_k)

        def _search(client: MilvusClient):
            return client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_search, sparse_search],
                ranker=reranker,
                limit=top_k,
                output_fields=output_fields,
            )

        results = self._run(_search)
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("text", ""),
                    "filename": hit.get("filename", ""),
                    "file_type": hit.get("file_type", ""),
                    "page_number": hit.get("page_number", 0),
                    "chunk_id": hit.get("chunk_id", ""),
                    "parent_chunk_id": hit.get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("root_chunk_id", ""),
                    "chunk_level": hit.get("chunk_level", 0),
                    "chunk_idx": hit.get("chunk_idx", 0),
                    "modality": hit.get("modality", "text"),
                    "asset_ids": _decode_asset_ids(hit.get("asset_ids")),
                    "score": hit.get("distance", 0.0),
                })
        return formatted_results

    def dense_retrieve(
        self,
        dense_embedding: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict]:
        def _search(client: MilvusClient):
            return client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=top_k,
                output_fields=[
                    "text",
                    "filename",
                    "file_type",
                    "page_number",
                    "chunk_id",
                    "parent_chunk_id",
                    "root_chunk_id",
                    "chunk_level",
                    "chunk_idx",
                    "modality",
                    "asset_ids",
                ],
                filter=filter_expr,
            )

        results = self._run(_search)
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "file_type": hit.get("entity", {}).get("file_type", ""),
                    "page_number": hit.get("entity", {}).get("page_number", 0),
                    "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                    "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                    "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                    "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                    "modality": hit.get("entity", {}).get("modality", "text"),
                    "asset_ids": _decode_asset_ids(hit.get("entity", {}).get("asset_ids")),
                    "score": hit.get("distance", 0.0),
                })
        return formatted_results

    def delete(self, filter_expr: str):
        return self._run(
            lambda client: client.delete(collection_name=self.collection_name, filter=filter_expr)
        )

    def has_collection(self) -> bool:
        return self._run(lambda client: client.has_collection(self.collection_name))

    def drop_collection(self) -> None:
        def _drop(client: MilvusClient) -> None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)

        self._run(_drop)


_store: MilvusStore | None = None


def get_milvus_store() -> MilvusStore:
    global _store
    if _store is None:
        _store = MilvusStore()
    return _store
