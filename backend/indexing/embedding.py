"""Text embedding service - dense vectors only (Milvus 2.5+ natively supports Chinese tokenization and BM25 full-text search)

## Why there is a backend choice here

The model is bge-m3, and on CPU it costs ~2.0 GB resident and ~91 ms per query. Both
numbers are fine once. The problem is that they are paid *per process*: this module's
singleton means every uvicorn worker loads its own copy, so four workers cost 8 GB of
identical weights to serve one model. That is the wrong thing to scale, and shrinking
the model does not fix it — int8 dynamic quantisation only reaches ~1.0 GB, because
XLM-R's 250k-token embedding table is not a Linear layer and stays fp32.

What fixes it is having ONE copy. So the embedder is selected by `EMBEDDING_BACKEND`:

  local   (default)  the model in this process, exactly as before.
  openai             an OpenAI-compatible /v1/embeddings endpoint.

`openai` is deliberately not "use a commercial API". It is a protocol, and the same
setting covers both deployments that matter:

  * bge-m3 behind HuggingFace text-embeddings-inference or Infinity, on one container.
    Same model, same weights, same vectors — so no re-index, no accuracy question, no
    per-token bill. API workers drop to a few hundred MB and become cheap to add.
  * a hosted embedding provider, if one is ever measured to be good enough on this
    corpus's languages.

The second is a decision that needs evidence, not a config change: switching model
changes the vector space, which means a full re-index and a retrieval quality question
in every language the corpus serves. The first needs neither.

## And why loading is lazy

`EmbeddingService()` used to construct the model at import. Importing this module — or
anything that transitively imports it, which is most of the backend — therefore cost
~110 s and 2 GB even in a process that would never embed anything, including one
configured to use a remote backend. It loads on first use instead.
"""
import logging
import os
import threading
from functools import lru_cache
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


class _Embedder(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...


def _create_dense_embedder() -> "_Embedder":
    backend = (os.getenv("EMBEDDING_BACKEND") or "local").strip().lower()
    if backend in ("openai", "http", "remote"):
        return _RemoteEmbedder()
    if backend not in ("local", ""):
        logger.warning("unknown EMBEDDING_BACKEND=%r; using the local model", backend)

    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    logger.info("loading local embedding model %s on %s", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


class _RemoteEmbedder:
    """An OpenAI-compatible /v1/embeddings client.

    Batched, because one request per text turns a 200-chunk write into 200 round trips.
    Not retried here: the callers that must not lose a vector (the catalogue build, the
    Milvus writer) already check the returned count and refuse to store a short or
    misaligned result, which is the failure that actually matters.
    """

    def __init__(self):
        self.base_url = (os.getenv("EMBEDDING_BASE_URL") or "").strip().rstrip("/")
        if not self.base_url:
            raise ValueError(
                "EMBEDDING_BACKEND is remote but EMBEDDING_BASE_URL is not set "
                "(for example http://embedder:8080/v1)"
            )
        self.model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.api_key = (os.getenv("EMBEDDING_API_KEY") or "").strip()
        self.timeout = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS") or 30.0)
        self.batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE") or 64))
        self._session = None
        self._session_lock = threading.Lock()
        logger.info("embedding via %s (model %s)", self.base_url, self.model)

    def _get_session(self):
        """One pooled Session for the process, not a fresh connection per call.

        `requests.post` opens a new TCP connection — and, over HTTPS, a new TLS
        handshake — for every call it makes. On the request path that is a round trip
        or three added to every query, and under load it also exhausts ephemeral ports.
        A Session keeps connections alive and reuses them; the pool is sized to the
        server's worker threads so concurrent callers do not serialise on one socket.
        """
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    import requests
                    from requests.adapters import HTTPAdapter

                    pool = max(10, int(os.getenv("EMBEDDING_POOL_SIZE") or 32))
                    session = requests.Session()
                    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool)
                    session.mount("http://", adapter)
                    session.mount("https://", adapter)
                    self._session = session
        return self._session

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        session = self._get_session()

        vectors: List[List[float]] = []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = session.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": batch},
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            # Sorted by index rather than trusted in arrival order: the OpenAI schema
            # permits any order, and a silently permuted batch would attach every
            # vector to the wrong chunk — corruption that no later check would catch.
            items = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
            if len(items) != len(batch):
                raise ValueError(
                    f"embedding endpoint returned {len(items)} vectors for {len(batch)} inputs"
                )
            vectors.extend([float(value) for value in item["embedding"]] for item in items)
        return vectors


def _wrap(embedder: "_Embedder") -> "_Embedder":
    """Apply request coalescing, unless it is switched off.

    On by default because it has no cost when idle — see CoalescingEmbedder — and it is
    the difference between per-query cost falling under load and rising under load.
    """
    size = int(os.getenv("EMBEDDING_COALESCE_MAX_BATCH") or 16)
    if size <= 1:
        logger.info("embedding request coalescing disabled")
        return embedder
    return CoalescingEmbedder(embedder, max_batch=size)


class _Waiting:
    """One caller's slot in a coalesced batch."""

    __slots__ = ("text", "event", "vector", "error")

    def __init__(self, text: str):
        self.text = text
        self.event = threading.Event()
        self.vector: Optional[List[float]] = None
        self.error: Optional[BaseException] = None


class CoalescingEmbedder:
    """Merges concurrently-arriving single-text embeds into one batched call.

    Under load this is close to free throughput. Measured on bge-m3 at query length,
    per-query cost falls from ~85 ms alone to ~44 ms in a batch of 16 — the model pays
    a largely fixed overhead per forward pass, so a batch of sixteen costs far less than
    sixteen batches of one. Against a remote endpoint the same coalescing turns sixteen
    HTTP round trips into one.

    **No artificial delay.** The obvious design waits a few milliseconds hoping company
    arrives, which taxes every request to benefit the busy case. This instead exploits
    queueing that already exists: while one batch is in flight the embedder is busy, so
    callers arriving during it naturally pile up, and the leader sweeps whatever
    accumulated as soon as it finishes. Batches grow as load grows and collapse to one
    when idle, and a lone caller waits for nothing.

    One thread at a time holds leadership and drains the queue; everyone else blocks on
    their own event. Multi-text calls (ingest, catalogue builds) bypass all of this —
    they are already batches, and their sizes would distort a query-latency queue.
    """

    def __init__(self, inner: "_Embedder", max_batch: int = 16):
        self._inner = inner
        self._max_batch = max(1, max_batch)
        self._lock = threading.Lock()
        self._queue: List[_Waiting] = []
        self._leader = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if len(texts) != 1:
            return self._inner.embed_documents(texts)

        waiting = _Waiting(texts[0])
        with self._lock:
            self._queue.append(waiting)
            lead = not self._leader
            if lead:
                self._leader = True

        if lead:
            self._drain()
        else:
            waiting.event.wait()

        if waiting.error is not None:
            raise waiting.error
        return [waiting.vector]

    def _drain(self) -> None:
        """Run batches until the queue empties, then hand leadership back.

        The `finally` is load-bearing twice over: leadership must be released even if
        the embedder raises, or the process would never batch again; and every waiter
        must be woken even on failure, or its thread blocks forever on an event nobody
        will set.
        """
        try:
            while True:
                with self._lock:
                    batch = self._queue[: self._max_batch]
                    del self._queue[: self._max_batch]
                    if not batch:
                        self._leader = False
                        return
                self._run(batch)
        except BaseException:
            with self._lock:
                self._leader = False
                stranded = self._queue[:]
                self._queue.clear()
            for item in stranded:
                item.error = RuntimeError("embedding batch leader failed")
                item.event.set()
            raise

    def _run(self, batch: List[_Waiting]) -> None:
        try:
            vectors = self._inner.embed_documents([item.text for item in batch])
            if len(vectors) != len(batch):
                raise ValueError(
                    f"embedder returned {len(vectors)} vectors for {len(batch)} texts"
                )
            for item, vector in zip(batch, vectors):
                item.vector = vector
        except BaseException as exc:
            for item in batch:
                item.error = exc
        finally:
            for item in batch:
                item.event.set()


class EmbeddingService:
    """Text embedding service - dense vectors, local model or remote endpoint.

    The embedder is built on first use and then reused. Construction is guarded because
    two requests arriving together into a cold process would otherwise both load the
    model, and briefly hold two copies of it.
    """

    def __init__(self, state_path=None, embedder: Optional["_Embedder"] = None):
        self._embedder = embedder
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        """Whether the embedder is built. Read by the readiness probe, which must not
        trigger the build itself — a probe is not a reason to load 2 GB."""
        return self._embedder is not None

    def _get_embedder(self) -> "_Embedder":
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    self._embedder = _wrap(_create_dense_embedder())
        return self._embedder

    def warm_up(self) -> None:
        """Build the embedder now rather than on the first user's request.

        Worth calling at startup: on the local backend the first call pays the model
        load, and a request that arrives during it waits it out.
        """
        try:
            self._get_embedder()
        except Exception:
            logger.warning("embedder could not be warmed up", exc_info=True)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._get_embedder().embed_documents(texts)
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
