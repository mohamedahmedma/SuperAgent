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
    vectors: List[Sequence[float]] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    topics: List[List[str]] = field(default_factory=list)
    floor: float = 0.0
    catalogue: str = ""

    @property
    def ready(self) -> bool:
        return bool(self.vectors)

    def best_matches(self, query_vector: Sequence[float], limit: int = 3) -> List[ScopeMatch]:
        """Closest anticipated questions, best first.

        Max over questions, not a mean over sections: a user asks one question, not the
        average of the several a section covers.
        """
        if not query_vector:
            return []
        dimension = len(query_vector)
        scored: List[Tuple[float, int]] = []
        for position, vector in enumerate(self.vectors):
            if len(vector) != dimension:
                continue
            scored.append((sum(a * b for a, b in zip(vector, query_vector)), position))
        scored.sort(reverse=True)
        return [
            ScopeMatch(
                question=self.questions[position],
                score=score,
                chunk_id=self.chunk_ids[position],
                topics=list(self.topics[position]),
            )
            for score, position in scored[:limit]
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
    if len(vectors) < 2:
        return 0.0

    maxima: List[float] = []
    for index, vector in enumerate(vectors):
        best = None
        for other, candidate in enumerate(vectors):
            if other == index or chunk_ids[other] == chunk_ids[index]:
                continue
            if len(candidate) != len(vector):
                continue
            score = sum(a * b for a, b in zip(vector, candidate))
            if best is None or score > best:
                best = score
        if best is not None:
            maxima.append(best)

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
    """Embed every catalogued question and derive the floor.

    `embed(texts) -> list[vector]` is injected so this is testable without a model and
    so the caller controls batching.
    """
    questions: List[str] = []
    chunk_ids: List[str] = []
    topics: List[List[str]] = []
    for record in records:
        for question in getattr(record, "answers", None) or []:
            questions.append(question)
            chunk_ids.append(getattr(record, "chunk_id", ""))
            topics.append(list(getattr(record, "topics", None) or []))

    if not questions:
        logger.warning("scope index: no catalogued questions; the gate will abstain")
        return ScopeIndex(catalogue=catalogue)

    vectors = list(embed(questions))
    if len(vectors) != len(questions):
        logger.error(
            "scope index: embedder returned %d vectors for %d questions; abstaining",
            len(vectors), len(questions),
        )
        return ScopeIndex(catalogue=catalogue)

    floor = derive_floor(vectors, chunk_ids, floor_percentile)
    logger.info(
        "scope index: %d questions over %d sections, derived floor %.4f (p%.0f)",
        len(questions), len(set(chunk_ids)), floor, floor_percentile,
    )
    return ScopeIndex(
        questions=questions,
        vectors=vectors,
        chunk_ids=chunk_ids,
        topics=topics,
        floor=floor,
        catalogue=catalogue,
    )
