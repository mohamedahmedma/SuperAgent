"""Local confidence assessment: deciding when the grader is not worth calling.

The evidence grader is one synchronous LLM call on every retrieval, and on a
decomposed question one per sub-agent. Much of the time it confirms what retrieval
already made obvious — the top chunk plainly contains what was asked for.

This module answers "is that obvious?" using only signals already in hand: how many
chunks came back, how much of the question's own vocabulary appears in them, and how
far the best hit stands above the rest. No model, no network, microseconds.

The bias is deliberately toward CALLING the grader. Skipping it wrongly means
answering from weak evidence, which is the failure this whole pipeline exists to
prevent; calling it unnecessarily only costs latency. So every signal must agree
before grading is skipped, and anything unusual falls through to the grader.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.text_normalization import normalize_query

logger = logging.getLogger(__name__)

# Words that open a question without saying anything about which passage answers it.
# Shared with the BM25 analyzer for exactly the same reason.
try:  # pragma: no cover - import shape only
    from backend.indexing.milvus_client import _QUESTION_STOP_WORDS as _STOP_WORDS
except Exception:  # pragma: no cover
    _STOP_WORDS = []

_STOP = set(_STOP_WORDS) | {
    "the", "a", "an", "of", "for", "to", "in", "on", "at", "and", "or", "is", "are",
    "was", "were", "be", "with", "that", "this", "it", "as", "by", "from",
    # Arabic function words: the corpus is Arabic-first and these carry no more
    # signal than "the" does.
    "من", "في", "على", "عن", "الى", "إلى", "هو", "هي", "ما", "هل", "مع", "أو", "او",
}

_TOKEN_RE = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


@dataclass
class ConfidenceVerdict:
    """Whether retrieval is clearly good enough to skip grading, and why."""

    confident: bool
    term_coverage: float = 0.0
    chunk_count: int = 0
    top_score: Optional[float] = None
    score_margin: Optional[float] = None
    reasons: List[str] = field(default_factory=list)

    def as_trace(self) -> Dict[str, Any]:
        return {
            "grading_confident": self.confident,
            "grading_term_coverage": round(self.term_coverage, 3),
            "grading_chunk_count": self.chunk_count,
            "grading_reason": "; ".join(self.reasons) or "n/a",
        }


def content_tokens(text: str) -> List[str]:
    """Meaningful tokens: normalised, stop-words removed, single characters dropped."""
    normalized = (normalize_query(text) or text or "").lower()
    return [
        token for token in _TOKEN_RE.findall(normalized)
        if len(token) > 1 and token not in _STOP
    ]


def term_coverage(question: str, docs: Sequence[dict], top_n: int = 3) -> float:
    """Fraction of the question's content words that appear in the top chunks.

    A prefix match rather than equality, so "partner" covers "partners" and Arabic
    inflections match their stem. That is deliberately generous: this signal is only
    ever used to SKIP work, so over-matching costs a little latency while
    under-matching costs nothing but a grader call that was going to happen anyway.
    """
    tokens = content_tokens(question)
    if not tokens:
        return 0.0

    haystack = " ".join(
        (normalize_query(doc.get("text", "")) or doc.get("text", "")).lower()
        for doc in list(docs)[:top_n]
    )
    if not haystack:
        return 0.0

    matched = 0
    for token in tokens:
        stem = token[:5] if len(token) > 5 else token
        if stem in haystack:
            matched += 1
    return matched / len(tokens)


def _scores(docs: Sequence[dict]) -> List[float]:
    values: List[float] = []
    for doc in docs:
        score = doc.get("rerank_score")
        if score is None:
            score = doc.get("score")
        if score is not None:
            try:
                values.append(float(score))
            except (TypeError, ValueError):
                continue
    return values


def has_rerank_scores(docs: Sequence[dict]) -> bool:
    """Whether the documents actually carry reranker scores.

    Checked on the documents rather than on `meta["rerank_applied"]`, which is set
    before the HTTP call and stays true when that call fails — so it means "a rerank
    was attempted", not "these scores are calibrated". Reading a raw RRF fusion score
    against a threshold meant for a 0-1 relevance score compares two different units
    and rejects everything.
    """
    return any(doc.get("rerank_score") is not None for doc in docs)


def assess(
    question: str,
    docs: Sequence[dict],
    meta: Optional[Dict[str, Any]],
    config,
) -> ConfidenceVerdict:
    """Is retrieval so clearly on-target that grading would only confirm it?

    Every condition must hold. Any doubt returns not-confident, which routes to the
    grader exactly as before.
    """
    docs = list(docs or [])
    meta = meta or {}
    scores = _scores(docs)
    top = scores[0] if scores else None
    margin = (top - (sum(scores[1:]) / len(scores[1:]))) if len(scores) > 1 and top is not None else None

    verdict = ConfidenceVerdict(
        confident=False,
        chunk_count=len(docs),
        top_score=top,
        score_margin=margin,
    )

    if not docs:
        # No documents is a decision the grader does not need to make either, but it
        # is emphatically not confidence — the caller already short-circuits it.
        verdict.reasons.append("no_documents")
        return verdict

    if len(docs) < config.skip_grading_min_chunks:
        verdict.reasons.append(f"only {len(docs)} chunk(s) < {config.skip_grading_min_chunks}")
        return verdict

    coverage = term_coverage(question, docs, config.skip_grading_top_chunks)
    verdict.term_coverage = coverage
    if coverage < config.skip_grading_term_coverage:
        verdict.reasons.append(
            f"term coverage {coverage:.2f} < {config.skip_grading_term_coverage}"
        )
        return verdict

    # A rerank score is a calibrated 0-1 relevance judgement, so when one is present
    # it is worth requiring. RRF fusion scores are not comparable across queries, so
    # no threshold is applied to them — coverage carries the decision instead.
    if has_rerank_scores(docs) and top is not None:
        if top < config.skip_grading_min_rerank_score:
            verdict.reasons.append(
                f"top rerank score {top:.3f} < {config.skip_grading_min_rerank_score}"
            )
            return verdict
        verdict.reasons.append(f"rerank score {top:.3f} ok")

    verdict.confident = True
    verdict.reasons.append(
        f"{len(docs)} chunks, term coverage {coverage:.2f} >= {config.skip_grading_term_coverage}"
    )
    return verdict


def should_grade(question: str, docs: Sequence[dict], meta: Optional[Dict[str, Any]], config):
    """(grade_needed, verdict) for the configured grading mode."""
    mode = getattr(config, "grading_mode", "always")

    if mode == "never":
        verdict = ConfidenceVerdict(confident=True, chunk_count=len(docs or []))
        verdict.reasons.append("grading_mode=never")
        return False, verdict

    if mode != "uncertain_only":
        verdict = ConfidenceVerdict(confident=False, chunk_count=len(docs or []))
        verdict.reasons.append("grading_mode=always")
        return True, verdict

    verdict = assess(question, docs, meta, config)
    return (not verdict.confident), verdict
