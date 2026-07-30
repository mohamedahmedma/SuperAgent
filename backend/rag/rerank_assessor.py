"""Rung 2: a cross-encoder scoring the question against each retrieved chunk.

This is the rung the first attempt at conditional grading was missing, and its absence
is why that attempt had to be rolled back. Lexical overlap reads zero whenever the
question and the corpus differ in language or vocabulary, so it could not tell "retrieval
missed" from "retrieval hit and worded it differently". A cross-encoder reads the pair
together and returns a calibrated relevance score, which is comparable across queries and
indifferent to which language each side is written in.

Two things follow from having per-chunk calibrated scores, and both were impossible before:

  * The LLM grader becomes the exception. When every chunk scores high or every chunk
    scores low, the answer is already known and the call is redundant.
  * Context trimming becomes safe. A chunk excluded by a calibrated scorer was positively
    judged, not merely unmatched by a word list.

On the pool size: reranking exactly the final `top_k` is close to pointless, because
permuting four chunks you were going to send anyway leaves the prompt byte-identical. The
value is in compressing a wider pool — recall more cheaply, then let the cross-encoder
choose. The pool is configurable for that reason, and the assessor scores whatever it is
handed.

A cross-encoder is one forward pass per pair with nothing generated. An LLM asked to rank
is a generation call, which costs output tokens and duplicates the grader that already
exists one rung up.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Sequence

from backend.rag.evidence import AssessmentContext, Certainty, ChunkAssessment, EvidenceReport

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()
_model_failed = False


def _load_model(model_name: str, device: str):
    """Load the cross-encoder once per process.

    Failure is cached: a missing dependency or an absent model must not be retried on
    every request, and the ladder has a rung above this one that still works.
    """
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    with _model_lock:
        if _model is not None or _model_failed:
            return _model
        try:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(model_name, device=device)
            logger.info("cross-encoder reranker loaded: %s on %s", model_name, device)
        except Exception:
            logger.warning(
                "cross-encoder %s unavailable; the ladder will fall through to the LLM grader",
                model_name, exc_info=True,
            )
            _model_failed = True
    return _model


def reset_model_cache() -> None:
    """For tests, and for switching models without a restart."""
    global _model, _model_failed
    with _model_lock:
        _model = None
        _model_failed = False


def score_pairs(question: str, texts: Sequence[str], config) -> Optional[List[float]]:
    """Calibrated relevance for each (question, chunk) pair, or None if unavailable."""
    if not texts:
        return []
    model = _load_model(
        getattr(config, "rerank_cross_encoder_model", "BAAI/bge-reranker-v2-m3"),
        getattr(config, "rerank_cross_encoder_device", "cpu"),
    )
    if model is None:
        return None

    char_limit = int(getattr(config, "rerank_doc_char_limit", 2000))
    pairs = [(question, (text or "")[:char_limit]) for text in texts]
    raw = model.predict(pairs)
    # Some checkpoints emit logits rather than probabilities; a sigmoid is idempotent on
    # values already in [0, 1] only approximately, so normalise explicitly instead.
    return [_to_unit(value) for value in raw]


def _to_unit(value) -> float:
    import math

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= number <= 1.0:
        return number
    return 1.0 / (1.0 + math.exp(-number))


class CrossEncoderAssessor:
    """MEDIUM certainty: a real semantic judgement, but not a reading of the question.

    It knows whether a passage is about what was asked. It cannot tell that a question is
    under-specified, or that two candidate readings exist — those need the rung above.
    So it concludes only at the two ends of its range and abstains in the middle, which
    is exactly where the LLM grader earns its cost.
    """

    name = "cross_encoder"
    certainty = Certainty.MEDIUM

    def assess(self, ctx: AssessmentContext) -> Optional[EvidenceReport]:
        if not ctx.docs:
            return None
        if not getattr(ctx.config, "rerank_cross_encoder_enabled", False):
            return None

        scores = score_pairs(ctx.question, [doc.get("text", "") for doc in ctx.docs], ctx.config)
        if scores is None:
            return None

        sufficient_at = float(getattr(ctx.config, "rerank_sufficient_score", 0.6))
        irrelevant_at = float(getattr(ctx.config, "rerank_irrelevant_score", 0.05))
        min_supporting = max(1, int(getattr(ctx.config, "rerank_min_supporting_chunks", 1)))

        chunks = [
            ChunkAssessment(index=i, score=score, signals={"cross_encoder": score})
            for i, score in enumerate(scores, 1)
        ]
        supporting = [chunk for chunk in chunks if chunk.score >= sufficient_at]
        best = max(scores) if scores else 0.0

        report = EvidenceReport(
            question=ctx.question,
            chunks=chunks,
            assessed_by=[self.name],
            confidence=best,
        )

        if best < irrelevant_at:
            # Nothing in the pool is about this question. That is a conclusion, not a
            # guess, and it saves the grader a call to say the same thing.
            report.certainty = Certainty.MEDIUM
            report.relevance = "none"
            report.sufficiency = "none"
            report.preferred_route = "no_knowledge"
            report.reasons.append(f"best cross-encoder score {best:.3f} < {irrelevant_at}")
            return report

        if len(supporting) >= min_supporting:
            report.certainty = Certainty.MEDIUM
            report.relevance = "strong"
            report.sufficiency = "sufficient"
            report.preferred_route = "answer"
            for chunk in chunks:
                # Recorded per chunk, which is what makes trimming a consequence of the
                # judgement rather than a separate guess about it.
                chunk.supported = chunk.score >= sufficient_at
            report.reasons.append(
                f"{len(supporting)} chunk(s) scored >= {sufficient_at}, best {best:.3f}"
            )
            return report

        # In between: relevant enough not to dismiss, not clearly sufficient. Report the
        # scores and let the grader decide — the report stays LOW so no policy acts on it.
        report.certainty = Certainty.LOW
        report.reasons.append(f"best cross-encoder score {best:.3f} inconclusive")
        return report
