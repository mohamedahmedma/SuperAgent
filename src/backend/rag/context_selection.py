"""Adaptive context sizing: sending only as many chunks as the answer needs.

Retrieval returns a fixed `top_k` because that is what a vector search is good at.
Answering does not need a fixed number — some questions are settled by one passage
and some genuinely need four. Sending all four every time pays for the worst case on
every turn, and the retrieved set is not paid for once: it stays in the message
history, so a follow-up tool call or a later turn re-sends it.

This module trims the retrieved set to its useful prefix using signals already in
hand — no model, no network. It runs AFTER grading, so the grader still judges on
complete evidence and only the answer prompt sees the trimmed set.

The failure this must not cause is a confident answer built on a passage that merely
*mentions* the question's terms without containing the facts. Term coverage measures
question vocabulary, not answer completeness, so coverage alone is never enough to
justify a cut. Three separate guards stand in front of it:

  1. Questions asking for breadth ("list all", "compare", "جميع") are never trimmed —
     a partial answer to those looks complete and reads as authoritative.
  2. A short chunk cannot end selection. Heading-only leaves match a question's words
     perfectly while containing nothing; roughly a fifth of a typical corpus is these.
  3. Trimming only happens in the regime where retrieval is already demonstrably on
     target, reusing the same assessment that decides whether to skip the grader.

Anything unusual keeps the full set. Trimming too little costs tokens; trimming too
much costs correctness, and only one of those is recoverable.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.rag.confidence import assess, content_tokens
from backend.text_normalization import normalize_query

logger = logging.getLogger(__name__)

# Questions whose answer is a set rather than a fact. Trimming these produces an
# answer that is wrong by omission while looking complete, which is worse than a long
# prompt. Matched on the normalized question, so Arabic forms sit alongside English.
_EXHAUSTIVE_MARKERS = (
    "list all", "list the", "all of the", "each of", "every ",
    "compare", "difference between", "versus", " vs ",
    "how many", "how much", "breakdown", "summar", "overview",
    "جميع", "كل ", "قارن", "الفرق بين", "اذكر", "عدد ",
)


@dataclass
class ContextSelection:
    """How many chunks were kept, and why the rest were dropped."""

    kept: int
    available: int
    coverage: float = 0.0
    trimmed: bool = False
    reasons: List[str] = field(default_factory=list)

    def as_trace(self) -> Dict[str, Any]:
        return {
            "context_chunks_kept": self.kept,
            "context_chunks_available": self.available,
            "context_trimmed": self.trimmed,
            "context_coverage": round(self.coverage, 3),
            "context_selection_reason": "; ".join(self.reasons) or "n/a",
        }


def _normalized(text: str) -> str:
    return (normalize_query(text) or text or "").lower()


def wants_exhaustive_answer(question: str) -> bool:
    """Whether the question asks for a complete set rather than a single fact."""
    normalized = _normalized(question)
    return any(marker in normalized for marker in _EXHAUSTIVE_MARKERS)


def _covered_by(tokens: Sequence[str], haystack: str) -> set:
    """Which of `tokens` appear in `haystack`, matching on a stem prefix."""
    return {token for token in tokens if (token[:5] if len(token) > 5 else token) in haystack}


def _is_substantive(doc: dict, min_chars: int) -> bool:
    """Whether a chunk carries enough text to plausibly hold an answer.

    A heading — "Term Dates", "الشراكات" — matches its question's vocabulary
    completely and answers nothing. Such a chunk may be kept as context, but it can
    never be the reason selection stops.
    """
    return len((doc.get("text") or "").strip()) >= min_chars


def select_context(
    question: str,
    docs: Sequence[dict],
    config,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[dict], ContextSelection]:
    """Return the useful prefix of `docs` for answering `question`, and why.

    Documents keep their retrieval order, so citation markers stay aligned with the
    list the caller goes on to format.
    """
    docs = list(docs or [])
    selection = ContextSelection(kept=len(docs), available=len(docs))

    mode = getattr(config, "context_selection_mode", "off")
    if mode != "adaptive":
        selection.reasons.append(f"context_selection_mode={mode}")
        return docs, selection

    floor = max(1, int(getattr(config, "context_min_chunks", 2)))
    if len(docs) <= floor:
        selection.reasons.append(f"only {len(docs)} chunk(s), at or below the floor of {floor}")
        return docs, selection

    if wants_exhaustive_answer(question):
        # Breadth questions need every passage that contributes an item. Cutting here
        # yields a list that is silently short, which reads as complete and is not.
        selection.reasons.append("question asks for a complete set")
        return docs, selection

    # Same bar as skipping the grader: trim only where retrieval is demonstrably on
    # target. When the evidence is merely adequate, the extra chunks are the margin
    # that keeps the answer honest.
    verdict = assess(question, docs, meta or {}, config)
    if not verdict.confident:
        selection.coverage = verdict.term_coverage
        selection.reasons.append("retrieval not confidently on target: " + "; ".join(verdict.reasons))
        return docs, selection

    tokens = content_tokens(question)
    if not tokens:
        selection.reasons.append("no content words to measure against")
        return docs, selection

    target = float(getattr(config, "context_target_coverage", 1.0))
    min_chars = int(getattr(config, "context_min_chunk_chars", 120))

    covered: set = set()
    kept: List[dict] = []
    for doc in docs:
        kept.append(doc)
        covered |= _covered_by(tokens, _normalized(doc.get("text", "")))
        coverage = len(covered) / len(tokens)

        if len(kept) < floor:
            continue
        # Only a chunk substantial enough to hold the answer may end selection. A
        # heading that completed the coverage count proves nothing.
        if not _is_substantive(doc, min_chars):
            continue
        if coverage >= target:
            break

    selection.kept = len(kept)
    selection.coverage = len(covered) / len(tokens)
    selection.trimmed = len(kept) < len(docs)
    if selection.trimmed:
        selection.reasons.append(
            f"{len(kept)} of {len(docs)} chunks cover {selection.coverage:.0%} of the question"
        )
    else:
        selection.reasons.append(f"all {len(docs)} chunks needed to reach {selection.coverage:.0%} coverage")
    return kept, selection
