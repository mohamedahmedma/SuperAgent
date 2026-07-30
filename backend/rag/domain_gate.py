"""Rung 0 of the ladder, and the cheapest possible answer to "is this even our subject?"

A question the corpus cannot answer costs a Milvus hybrid search, a complexity call, and
a grader call before the system concludes what a vector comparison could have said
first. This module makes that comparison.

It is deliberately not a search. Reference vectors for the corpus sections are held in
memory and compared against the query vector with a single matrix product — the same
query vector retrieval is about to use, so the classification is free. There is no
index: a handbook-sized corpus is a few thousand vectors, and a matrix product over that
is faster than the syscall an index would need. FAISS earns its complexity somewhere
north of a million vectors, not here.

Two safety properties matter more than the saving:

  * **It fails open.** No reference data, an embedding failure, a shape mismatch — every
    one abstains and lets the question through. A false rejection is an unrecoverable
    silent refusal on a valid question; a false acceptance costs one search that
    retrieval and the grader were going to catch anyway. The asymmetry is not close.
  * **It never sees a bare follow-up.** "What about grade 6?" carries its topic in the
    previous turn, not in its own words, and would score near zero. Turns with
    conversation history are exempt unless the caller passes a contextualized question.

The topic side of the same comparison is a bonus, not a second mechanism: the sections
that scored highest narrow the retrieval filter, so a question about uniforms stops
competing with fee tables for candidate slots.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DomainVerdict:
    """Whether the question belongs to this corpus, and which parts of it."""

    in_domain: bool
    score: float = 0.0
    topics: List[str] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""

    def as_trace(self) -> Dict[str, Any]:
        return {
            "domain_gate_in_domain": self.in_domain,
            "domain_gate_score": round(self.score, 4),
            "domain_gate_topics": list(self.topics),
            "domain_gate_abstained": self.abstained,
            "domain_gate_reason": self.reason,
        }


def _abstain(reason: str) -> DomainVerdict:
    """Abstention is expressed as in_domain=True: the gate's failure must never be the
    reason a valid question is refused."""
    return DomainVerdict(in_domain=True, abstained=True, reason=reason)


@dataclass
class DomainReference:
    """Reference vectors for the corpus, grouped by section.

    Built once and held in memory. `vectors` is a list of (label, unit_vector) pairs;
    embeddings are already L2-normalized upstream, so a dot product IS the cosine and no
    renormalisation is needed.
    """

    vectors: List[Tuple[str, Sequence[float]]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.vectors)

    def best_matches(self, query_vector: Sequence[float], limit: int) -> List[Tuple[str, float]]:
        """Highest-scoring sections, best first. Pure Python dot products: for a few
        thousand reference vectors this is well under a millisecond, and it keeps numpy
        out of the request path."""
        scored: Dict[str, float] = {}
        for label, vector in self.vectors:
            if len(vector) != len(query_vector):
                continue
            score = sum(a * b for a, b in zip(vector, query_vector))
            if score > scored.get(label, float("-inf")):
                scored[label] = score
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ranked[:limit]


class DomainReferenceStore:
    """Lazily builds and caches the reference set.

    The provider is injected so the Milvus specifics live in exactly one place and the
    gate stays testable without a running cluster. A provider that raises leaves the
    store empty, which makes the gate abstain — the intended degradation.
    """

    def __init__(self, provider: Optional[Callable[[], DomainReference]] = None):
        self._provider = provider
        self._reference: Optional[DomainReference] = None
        self._lock = threading.Lock()
        self._failed = False

    def get(self) -> DomainReference:
        if self._reference is not None:
            return self._reference
        with self._lock:
            if self._reference is not None:
                return self._reference
            if self._failed or self._provider is None:
                return DomainReference()
            try:
                self._reference = self._provider() or DomainReference()
            except Exception:
                # Cached as failed so a broken provider is not retried on every request.
                logger.warning("domain reference build failed; gate will abstain", exc_info=True)
                self._failed = True
                return DomainReference()
        return self._reference

    def invalidate(self) -> None:
        """Called after re-indexing: the corpus changed, so the reference set has too."""
        with self._lock:
            self._reference = None
            self._failed = False


def milvus_reference_provider(level: int = 2, section_field: str = "root_chunk_id") -> DomainReference:
    """Build the reference set from the vectors already sitting in Milvus.

    Reads mid-level chunks rather than leaves: a leaf is a sentence or two and drifts
    off-topic, while a mid-level chunk is close to a coherent section. Nothing is
    embedded here — these vectors were computed at index time.

    Grouped by `root_chunk_id` because that is the only section-shaped field the
    collection actually has; the human-readable heading path lives inside the chunk text,
    not in a filterable column.
    """
    from backend.api.resources import milvus_manager

    rows = milvus_manager.query_all(
        filter_expr=f"chunk_level == {level}",
        output_fields=["dense_embedding", section_field],
    )
    vectors: List[Tuple[str, Sequence[float]]] = []
    for row in rows or []:
        vector = row.get("dense_embedding")
        label = row.get(section_field) or "unknown"
        if vector is not None and len(vector):
            vectors.append((str(label), list(vector)))
    logger.info("domain reference built: %d vectors over %d sections",
                len(vectors), len({label for label, _ in vectors}))
    return DomainReference(vectors=vectors)


# Process-wide, so the reference set is built once per deployment rather than per turn.
reference_store = DomainReferenceStore(provider=milvus_reference_provider)


def classify(
    query_vector: Optional[Sequence[float]],
    reference: DomainReference,
    config,
) -> DomainVerdict:
    """Compare the query against the corpus. Abstains on anything unexpected."""
    min_similarity = float(getattr(config, "domain_gate_min_similarity", 0.35))
    topic_limit = max(1, int(getattr(config, "domain_gate_topic_count", 3)))

    if not query_vector:
        return _abstain("no_query_vector")
    if not reference.ready:
        return _abstain("no_reference_vectors")

    matches = reference.best_matches(query_vector, topic_limit)
    if not matches:
        return _abstain("no_comparable_reference_vectors")

    best_score = matches[0][1]
    topics = [label for label, score in matches if score >= min_similarity]

    if best_score < min_similarity:
        return DomainVerdict(
            in_domain=False,
            score=best_score,
            topics=[],
            reason=f"best section similarity {best_score:.3f} < {min_similarity}",
        )

    return DomainVerdict(
        in_domain=True,
        score=best_score,
        topics=topics or [matches[0][0]],
        reason=f"matched {len(topics) or 1} section(s), best {best_score:.3f}",
    )


def should_run(question: str, *, has_history: bool, config) -> Tuple[bool, str]:
    """Whether the gate is safe to apply to this turn.

    A bare follow-up is the failure case that matters: its topic lives in the previous
    turn, so scoring its words against the corpus measures nothing. Skipping the gate
    there costs one search; applying it risks refusing a valid question.
    """
    if not getattr(config, "domain_gate_enabled", False):
        return False, "domain_gate_enabled=false"
    if not (question or "").strip():
        return False, "empty question"
    if has_history and getattr(config, "domain_gate_skip_with_history", True):
        return False, "conversation has history; question may be a follow-up"
    min_chars = int(getattr(config, "domain_gate_min_question_chars", 12))
    if len(question.strip()) < min_chars:
        # Too short to carry topical signal on its own — "fees?" is in-domain but scores
        # like noise.
        return False, f"question shorter than {min_chars} chars"
    return True, "eligible"
