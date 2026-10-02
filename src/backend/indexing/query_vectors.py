"""A query's dense vector, computed once per distinct text for the whole deployment.

RAG_FIX_PLAN item 18. With a hosted embedder every query vector is a paid round trip,
~545 ms from production (item 39), and a school's parents ask the same few hundred
questions. The per-process memo in front of it saw each question once per worker
and kept 64; a second worker, a second replica or a restart paid again.

Three layers, cheapest first. This is a near cache in front of a shared one, read
cache-aside:

  1. **This process's memory**, the latest `memo_size` texts: a dictionary hit, and the
     reason a turn that embeds the same text twice (the domain gate, then retrieval)
     pays once.
  2. **Redis**, shared by every worker and replica, for `ttl_seconds`.
  3. **The embedder.**

**Single flight.** The same text asked at the same moment is computed once: the first
caller computes it, and the others wait for that result rather than each paying the
provider. That is the moment it matters, when many parents ask about one announcement.

**Never stale, so never invalidated.** A vector is a pure function of the model and the
text, and the key names the embedding backend, endpoint, model and width. A change of
model is therefore a new key space, not an invalidation problem. The shared copy is
float32, the precision Milvus stores and searches in. This process's memo keeps exactly
what the embedder returned.

**Fails open.** A Redis that cannot be reached is a miss, and the embedder answers.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import struct
import threading
from array import array
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)


def _encode(vector: Sequence[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def _decode(raw: str) -> array:
    packed = base64.b64decode(raw)
    return array("d", struct.unpack(f"<{len(packed) // 4}f", packed))


class QueryVectorCache:
    """Query text -> dense vector, through memory, Redis, then the embedder."""

    def __init__(
        self,
        embed: Callable[[str], Sequence[float]],
        *,
        redis: Optional[Callable[[], Any]] = None,
        key: Callable[[str], str] = lambda name: name,
        namespace: str = "",
        ttl_seconds: int = 0,
        memo_size: int = 256,
    ) -> None:
        self._embed = embed
        self._redis = redis if ttl_seconds > 0 else None
        self._key = key
        self._namespace = namespace
        self._ttl = int(ttl_seconds)
        self._memo_size = max(1, int(memo_size))
        # Held as arrays of doubles: exactly what the embedder returned, at 8 KB a vector
        # where a list of 1,024 Python floats is ~32 KB. Only the shared copy is float32.
        self._memo: "OrderedDict[str, array]" = OrderedDict()
        self._inflight: dict[str, Future] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, embed: Callable[[str], Sequence[float]], cache: Any) -> "QueryVectorCache":
        """`QUERY_VECTOR_CACHE_TTL_SECONDS` (7 days; 0 keeps only this process's memo)."""
        model = ":".join((
            os.getenv("EMBEDDING_BACKEND") or "local",
            (os.getenv("EMBEDDING_BASE_URL") or "").rstrip("/"),
            os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3",
            os.getenv("DENSE_EMBEDDING_DIM") or "1024",
        ))
        return cls(
            embed,
            redis=getattr(cache, "client", None),
            key=getattr(cache, "key", lambda name: name),
            namespace=hashlib.sha256(model.encode("utf-8")).hexdigest()[:16],
            ttl_seconds=int(os.getenv("QUERY_VECTOR_CACHE_TTL_SECONDS") or 7 * 24 * 3600),
        )

    def vector(self, text: str) -> list[float]:
        """The text's vector, as a fresh list a caller may change without harming the cache."""
        with self._lock:
            hit = self._memo.get(text)
            if hit is not None:
                self._memo.move_to_end(text)
                return hit.tolist()
            flight = self._inflight.get(text)
            leader = flight is None
            if leader:
                flight = self._inflight[text] = Future()
        if not leader:
            return flight.result().tolist()
        try:
            vector = self._shared(text)
            if vector is None:
                vector = array("d", self._embed(text))
                self._share(text, vector)
            with self._lock:
                self._memo[text] = vector
                if len(self._memo) > self._memo_size:
                    self._memo.popitem(last=False)
            flight.set_result(vector)
            return vector.tolist()
        except BaseException as exc:
            flight.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(text, None)

    def clear(self) -> None:
        """Forget this process's memo. The shared layer is keyed by model, so it stays."""
        with self._lock:
            self._memo.clear()

    # -- the shared layer -----------------------------------------------------------

    def _redis_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self._key(f"query_vector:{self._namespace}:{digest}")

    def _shared(self, text: str) -> Optional[array]:
        if self._redis is None:
            return None
        try:
            raw = self._redis().get(self._redis_key(text))
            return _decode(raw) if raw else None
        except Exception:
            logger.debug("query vector cache unavailable; embedding instead", exc_info=True)
            return None

    def _share(self, text: str, vector: array) -> None:
        if self._redis is None:
            return
        try:
            self._redis().set(self._redis_key(text), _encode(vector), ex=self._ttl)
        except Exception:
            logger.debug("query vector cache unavailable; not shared", exc_info=True)


__all__ = ["QueryVectorCache"]
