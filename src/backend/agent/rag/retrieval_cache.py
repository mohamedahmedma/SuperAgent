"""Retrieval results, shared by every worker, until the corpus changes.

RAG_FIX_PLAN item 18. Once the query vector is known (`backend/indexing/query_vectors.py`),
retrieval is still a hybrid search, a merge of parent chunks and the filters: measured at
22 ms p50, 43 ms p95 and 7.8 ms of CPU per call, for a result of ~9.5 KB. A Redis read
of that result costs ~2 ms. More once reranking is switched on.

**The key is everything that can change what retrieval returns:**

  * the question as it is searched (normalised), and how many results were asked for;
  * the filter expression. That is the leaf level, and the language-pairing exclusion,
    so pairing two documents changes the key without an invalidation;
  * a fingerprint of the retrieval settings and of the embedding model, so a change of
    either is a new key space;
  * the corpus version (below).

Not the user. The knowledge base is one corpus per profile (keys are profile-prefixed)
and retrieval applies no per-user scope. A per-user filter added to retrieval must join
the key.

**Invalidation is a version, not a delete.** `CorpusVersion` is a counter in Redis that
every write to the corpus bumps: the vector store's inserts and deletes, and the parent
chunks a merge reads (see `backend/composition.py`). An entry carries the version it
was computed under, and an entry from an older version is a miss. Both are read in ONE
round trip (MGET). The version is read BEFORE the result is computed and stamped on
it, so a write that races a retrieval can only leave an entry that is ignored, never one
that is served.

Only a complete result is kept: a failed embedding or search is not an answer to
remember. And it fails open, since a Redis that cannot be reached is a miss.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_VERSION_KEY = "corpus_version"


class CorpusVersion:
    """A counter every write to the searchable corpus moves forward."""

    def __init__(self, *, redis: Optional[Callable[[], Any]] = None,
                 key: Callable[[str], str] = lambda name: name) -> None:
        self._redis = redis
        self._key = key(_VERSION_KEY)

    @classmethod
    def for_cache(cls, cache: Any) -> "CorpusVersion":
        """Over the cache's Redis. A cache with none behind it (a test's stand-in) counts nothing."""
        return cls(redis=getattr(cache, "client", None), key=getattr(cache, "key", lambda name: name))

    @property
    def key(self) -> str:
        return self._key

    def bump(self) -> None:
        """The corpus changed: every cached retrieval is now stale.

        Never raises into the write that called it. A bump that could not reach Redis
        leaves entries to expire by their TTL, which is the bound on staleness.
        """
        if self._redis is None:
            return
        try:
            self._redis().incr(self._key)
        except Exception:
            logger.warning("could not mark the corpus changed; cached retrievals expire by TTL",
                           exc_info=True)


@dataclass(frozen=True)
class RetrievalTicket:
    """What a lookup learned, for storing the result it missed: where, and under which version."""

    key: str = ""
    version: str = ""

    @property
    def storable(self) -> bool:
        return bool(self.key)


class RetrievalCache:
    """Retrieval results by question, filter and settings, valid for one corpus version."""

    def __init__(
        self,
        *,
        redis: Optional[Callable[[], Any]] = None,
        key: Callable[[str], str] = lambda name: name,
        corpus: CorpusVersion,
        fingerprint: str,
        ttl_seconds: int = 0,
    ) -> None:
        self._redis = redis if ttl_seconds > 0 else None
        self._key = key
        self._corpus = corpus
        self._fingerprint = fingerprint
        self._ttl = int(ttl_seconds)

    @classmethod
    def from_environment(cls, cache: Any, corpus: CorpusVersion, settings: dict) -> "RetrievalCache":
        """`RETRIEVAL_CACHE_TTL_SECONDS` (an hour; 0 turns the cache off).

        `settings` is everything retrieval is configured with; its fingerprint is part of
        every key, so a deployment that changes one starts from an empty cache.
        """
        fingerprint = hashlib.sha256(
            json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return cls(
            redis=getattr(cache, "client", None),
            key=getattr(cache, "key", lambda name: name),
            corpus=corpus,
            fingerprint=fingerprint,
            ttl_seconds=int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS") or 3600),
        )

    def lookup(self, query: str, top_k: int, filter_expr: str) -> tuple[Optional[dict], RetrievalTicket]:
        """The cached result, or None and the ticket to store the computed one under."""
        if self._redis is None:
            return None, RetrievalTicket()
        digest = hashlib.sha256(
            json.dumps([query, int(top_k), filter_expr], ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        entry_key = self._key(f"retrieval:{self._fingerprint}:{digest}")
        try:
            version, raw = self._redis().mget([self._corpus.key, entry_key])
        except Exception:
            logger.debug("retrieval cache unavailable; searching instead", exc_info=True)
            return None, RetrievalTicket()
        version = str(version or "0")
        ticket = RetrievalTicket(entry_key, version)
        if not raw:
            return None, ticket
        try:
            entry = json.loads(raw)
        except ValueError:
            return None, ticket
        if entry.get("version") != version:
            return None, ticket
        return entry.get("result"), ticket

    def store(self, ticket: RetrievalTicket, result: dict) -> None:
        """Keep `result` under the ticket's version, if it is a result worth keeping."""
        if self._redis is None or not ticket.storable or not _complete(result):
            return
        try:
            payload = json.dumps({"version": ticket.version, "result": result}, ensure_ascii=False)
            self._redis().set(ticket.key, payload, ex=self._ttl)
        except Exception:
            logger.debug("retrieval cache unavailable; result not kept", exc_info=True)


def _complete(result: Any) -> bool:
    """A result worth remembering: the full hybrid search, with no error.

    Not the dense-only fallback. That runs when the hybrid search failed, and keeping it
    would serve the degraded result for an hour after the search recovered. An honest
    empty result IS kept: the corpus version retires it when a document arrives.
    """
    if not isinstance(result, dict):
        return False
    meta = result.get("meta") or {}
    return meta.get("retrieval_mode") == "hybrid" and not meta.get("retrieval_error")


__all__ = ["CorpusVersion", "RetrievalCache", "RetrievalTicket"]
