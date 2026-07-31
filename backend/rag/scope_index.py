"""The scope index: one vector per question the corpus can answer.

Replaces scoring a query against chunk vectors, which never separated well. A chunk is
a fragment, so in a corpus of any size *something* is always vaguely near anything —
measured on a school handbook, an off-topic question about poetry scored 0.46 against a
best in-corpus score of 0.58. Twelve points of separation is not a decision boundary.

Comparing a question against the questions the corpus can answer is the most
homogeneous comparison available, and homogeneity is what lets one cut point behave the
same way tomorrow as today.

## The cut point is derived, never chosen

A hand-tuned threshold is the part of a design like this that rots: it is picked
against one corpus, and every re-index, every new document and every deployment moves
it without telling anyone.

So it is computed instead. Hold each catalogued question out, score it against all the
others, and you have the distribution of what a *genuinely in-scope* question scores
against this index. A low percentile of that distribution is the floor. It is a
statistic of the corpus, recomputed whenever the corpus changes, and it transfers to a
new deployment with no calibration at all.

## And it is only ever a floor on ASKING

Falling below it does not refuse anything. It escalates to a model that reads the
question, and only that model may end a turn. So a badly derived floor costs cheap
model calls, never a wrong refusal — which is the property that makes shipping this
safe before anyone has looked at a single score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Percentile of in-scope self-similarity used as the escalate-below floor. Low on
# purpose: the cost of escalating unnecessarily is one small model call, while the cost
# of admitting silently is a search that would have happened anyway.
DEFAULT_FLOOR_PERCENTILE = 10.0


@dataclass
class ScopeMatch:
    """One anticipated question and how closely the query resembled it."""

    question: str
    score: float
    chunk_id: str = ""
    topics: List[str] = field(default_factory=list)


@dataclass
class ScopeIndex:
    """Question vectors plus the floor derived from them."""

    questions: List[str] = field(default_factory=list)
    vectors: Any = None  # (n_questions, dim) float32 matrix
    chunk_ids: List[str] = field(default_factory=list)
    topics: List[List[str]] = field(default_factory=list)
    floor: float = 0.0
    catalogue: str = ""

    def __post_init__(self):
        """Accept a plain list of vectors and coerce it once.

        Matching needs a matrix, but a caller holding a list of vectors should not have
        to know that — and converting per query would put the cost back on the request
        path this class exists to keep cheap.
        """
        if self.vectors is not None and not isinstance(self.vectors, np.ndarray):
            self.vectors = np.asarray(self.vectors, dtype=np.float32) if len(self.vectors) else None

    @property
    def ready(self) -> bool:
        return self.vectors is not None and len(self.vectors) > 0

    def best_matches(self, query_vector: Sequence[float], limit: int = 3) -> List[ScopeMatch]:
        """Closest anticipated questions, best first.

        Max over questions, not a mean over sections: a user asks one question, not the
        average of the several a section covers.

        One matrix-vector product. The Python-loop version measured 16ms per query
        against 222 questions — small in isolation, but it is paid on every turn and
        grows linearly with the corpus, which is the wrong shape for something described
        as free.
        """
        if query_vector is None or not len(query_vector) or not self.ready:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape[0] != self.vectors.shape[1]:
            # A re-index under a different embedding model. Returning nothing makes the
            # rung abstain, rather than making every question look off-topic.
            return []

        scores = self.vectors @ query
        count = min(max(1, limit), scores.shape[0])
        # argpartition finds the top-k without sorting the rest.
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        return [
            ScopeMatch(
                question=self.questions[position],
                score=float(scores[position]),
                chunk_id=self.chunk_ids[position],
                topics=list(self.topics[position]),
            )
            for position in top
        ]

    def as_trace(self, matches: Sequence[ScopeMatch]) -> Dict[str, Any]:
        return {
            "scope_floor": round(self.floor, 4),
            "scope_index_size": len(self.vectors),
            "scope_top_score": round(matches[0].score, 4) if matches else None,
            "scope_top_question": matches[0].question if matches else None,
        }


def percentile(values: Sequence[float], point: float) -> float:
    """Linear-interpolated percentile. Small enough to keep numpy out of the import
    graph, and this runs at index time over a few hundred numbers."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (max(0.0, min(100.0, point)) / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def derive_floor(
    vectors: Sequence[Sequence[float]],
    chunk_ids: Sequence[str],
    point: float = DEFAULT_FLOOR_PERCENTILE,
) -> float:
    """The score below which a question stops looking in-scope for THIS corpus.

    Leave-one-out: every catalogued question is scored against every question from a
    DIFFERENT section, and the low percentile of those maxima becomes the floor.

    Different sections on purpose. Scoring a question against its own siblings measures
    how repetitive one section's questions are, which says nothing about scope — and it
    would put the floor near 1.0, escalating almost every real query. What matters is
    how well an in-scope question about topic A matches an index that mostly covers
    B, C and D, because that is the position every incoming query is in.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0

    # One n x n product instead of n^2 Python dot products — 3.4s became milliseconds
    # on a 222-question catalogue, and this runs at boot.
    similarity = matrix @ matrix.T
    sections = np.asarray(chunk_ids)
    same_section = sections[:, None] == sections[None, :]
    similarity[same_section] = -np.inf

    row_max = similarity.max(axis=1)
    maxima = [float(value) for value in row_max if np.isfinite(value)]

    if not maxima:
        # Every question belongs to the same section, so there is no cross-section
        # evidence to derive from. Zero means "escalate nothing on score alone", which
        # keeps the model in charge rather than inventing a boundary.
        return 0.0
    return percentile(maxima, point)


def build_index(
    records: Sequence[Any],
    embed,
    *,
    floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
    catalogue: str = "",
) -> ScopeIndex:
    """Assemble the index, embedding only what is not already stored.

    A record carrying `question_vectors` is used as-is. That is the difference between
    a process starting in milliseconds and one spending 22 seconds re-deriving vectors
    it already had — and the index builds lazily on first use, so that wait would have
    landed on the first user after every deploy.

    `embed(texts) -> list[vector]` is injected so this is testable without a model, and
    is called only for records missing their vectors.
    """
    questions: List[str] = []
    chunk_ids: List[str] = []
    topics: List[List[str]] = []
    stored: List[Optional[Sequence[float]]] = []
    for record in records:
        answers = getattr(record, "answers", None) or []
        cached = list(getattr(record, "question_vectors", None) or [])
        # Positional pairing, so a partial or stale vector list is discarded whole
        # rather than silently misaligning questions with other questions' vectors.
        usable_cache = len(cached) == len(answers)
        for position, question in enumerate(answers):
            questions.append(question)
            chunk_ids.append(getattr(record, "chunk_id", ""))
            topics.append(list(getattr(record, "topics", None) or []))
            stored.append(cached[position] if usable_cache else None)

    if not questions:
        logger.warning("scope index: no catalogued questions; the gate will abstain")
        return ScopeIndex(catalogue=catalogue)

    missing = [position for position, vector in enumerate(stored) if vector is None]
    if missing:
        logger.info("scope index: embedding %d question(s) not already stored", len(missing))
        fresh = list(embed([questions[position] for position in missing]))
        if len(fresh) != len(missing):
            logger.error(
                "scope index: embedder returned %d vectors for %d questions; abstaining",
                len(fresh), len(missing),
            )
            return ScopeIndex(catalogue=catalogue)
        for position, vector in zip(missing, fresh):
            stored[position] = vector

    vectors = stored

    matrix = np.asarray(vectors, dtype=np.float32)
    floor = derive_floor(matrix, chunk_ids, floor_percentile)
    logger.info(
        "scope index: %d questions over %d sections, derived floor %.4f (p%.0f)",
        len(questions), len(set(chunk_ids)), floor, floor_percentile,
    )
    return ScopeIndex(
        questions=questions,
        vectors=matrix,
        chunk_ids=chunk_ids,
        topics=topics,
        floor=floor,
        catalogue=catalogue,
    )
