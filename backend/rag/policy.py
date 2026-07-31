"""Decisions taken from an EvidenceReport. Pure functions, no signals of their own.

Every function here reads a report and profile config and returns a decision. None of
them computes a signal, calls a model, or touches graph state beyond the few scalars
passed in explicitly. That is the rule that keeps the pipeline coherent: when a new
decision is needed, it becomes a function here reading the existing report, rather than
a new component deriving its own view of the evidence.

Each policy declares the certainty it requires. Below that floor it degrades to its
conservative default — it does not go looking for a cheaper signal to justify acting.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from backend.rag.evidence import Certainty, EvidenceReport, parse_certainty

# Routing happens on every turn and cannot abstain, so it has no floor of its own —
# instead it refuses to claim sufficiency the report does not support.
#
# Trimming context is optional and destructive: sending fewer chunks than were judged
# is unrecoverable if wrong, so it requires a semantic assessment. LOW (lexical) is
# specifically not enough, which is the lesson from the first attempt.
CONTEXT_TRIM_MIN_CERTAINTY = Certainty.MEDIUM


def can_ask_human(report: EvidenceReport, *, hitl_rounds: int, config) -> Tuple[bool, str]:
    """Whether it is worth interrupting the user, and why not when it is not.

    Two rules, both learned from what the generic version actually produced.

    **A question must name what it needs.** "I found relevant content, but the evidence
    isn't enough to determine an answer. Please provide more detail" asks the user to
    guess what the system wants. It reads as a failure, it is unanswerable in any
    useful way, and the evidence that prompted it was good enough to attempt an answer
    from. So a HITL turn now requires named slots or named options — something the user
    can actually respond to. Without them, answering from partial evidence is better,
    because the answer prompt already says to be explicit about what it does not know.

    **Once per question.** After the user has answered, asking again spends their
    patience on a system that is not converging. Retrieval that is still ambiguous
    after a clarification will not be rescued by a second one.
    """
    limit = int(getattr(config, "max_hitl_rounds", 1))
    if hitl_rounds >= limit:
        return False, f"already asked {hitl_rounds} time(s), limit {limit}"

    if report.ambiguity == "missing_slot":
        if not report.missing_slots:
            return False, "missing_slot without naming which slot"
        return True, "missing a named condition"

    if report.ambiguity == "multiple_candidates":
        if not report.hitl_options:
            return False, "multiple_candidates without naming the candidates"
        return True, "several named candidates"

    return False, "nothing specific to ask for"


def decide_route(
    report: EvidenceReport,
    *,
    has_docs: bool,
    rewrite_count: int,
    is_sub_agent: bool,
    config,
    hitl_rounds: int = 0,
) -> Tuple[str, str]:
    """The next graph step, and why.

    Ported from the original grade-driven routing so behaviour is unchanged when an LLM
    assessor filled the report. Two deliberate departures: it refuses to invent
    sufficiency where nothing assessed the evidence, and it will not interrupt the user
    unless it can say what it needs — see `can_ask_human`.
    """
    max_rewrites = int(getattr(config, "max_rewrites", 1))
    required = parse_certainty(getattr(config, "evidence_required_certainty", "high"))
    ask_allowed, ask_reason = can_ask_human(report, hitl_rounds=hitl_rounds, config=config)

    if not has_docs or report.relevance == "none":
        return "no_knowledge", "no evidence retrieved"

    # Assessment could not reach the standard this profile requires. Retrieval worked,
    # so this is neither missing knowledge nor a bad question — it is a technical
    # failure, and saying "try again" is honest where answering or denying would not be.
    if not report.meets(required):
        return (
            "retrieval_error",
            f"assessment reached {report.certainty.name.lower()}, "
            f"profile requires {required.name.lower()}",
        )

    # Only a language model sets these, so a cheap rung can never route to a human.
    if ask_allowed and report.ambiguity == "missing_slot":
        return "clarify", f"missing: {', '.join(report.missing_slots)}"
    if ask_allowed and report.ambiguity == "multiple_candidates":
        return "scope_select", "several named candidates in the evidence"

    answer_is_supported = report.relevance == "strong" and report.sufficiency == "sufficient"
    if report.preferred_route == "answer" and answer_is_supported:
        return "answer", "evidence sufficient"

    # Sub-questions get no correction pass: partial evidence is left for synthesis to
    # merge across siblings, which is a better fix than rewriting one sub-question.
    if is_sub_agent:
        if report.sufficiency in ("partial", "sufficient"):
            return "answer", "sub-agent keeps usable evidence for synthesis"
        return "no_knowledge", "sub-agent found nothing usable"

    if report.preferred_route == "rewrite":
        if rewrite_count < max_rewrites:
            return "rewrite", "evidence insufficient, one rewrite remains"
        return _partial_or_nothing(report, ask_allowed, ask_reason, "rewrites exhausted")

    if report.sufficiency == "partial":
        if rewrite_count < max_rewrites:
            return "rewrite", "partial evidence, one rewrite remains"
        return _partial_or_nothing(report, ask_allowed, ask_reason, "rewrites exhausted")

    if answer_is_supported:
        return "answer", "evidence sufficient"

    return "no_knowledge", "evidence not sufficient to answer"


def _partial_or_nothing(report, ask_allowed: bool, ask_reason: str, context: str) -> Tuple[str, str]:
    """What to do with partial evidence once rewriting is spent.

    Answer from it, unless there is a specific question worth asking. Partial evidence
    is still evidence, and the answer prompt already requires the model to be explicit
    about what the sources do not cover — which is more useful to the user than a
    request to rephrase.
    """
    if ask_allowed:
        route = "scope_select" if report.ambiguity == "multiple_candidates" else "clarify"
        return route, f"{context}, {ask_reason}"
    if report.sufficiency == "partial":
        return "answer", f"{context}; answering from partial evidence ({ask_reason})"
    return "no_knowledge", context


def select_context_indices(
    report: EvidenceReport,
    docs: Sequence[dict],
    config,
) -> Tuple[Optional[List[int]], str]:
    """Which chunks (1-based) to send to the answer prompt, or None to send all.

    The safety property that the first attempt at this got wrong: never answer from a
    narrower set than the one judged sufficient. So the only chunks droppable are those
    an assessment positively excluded — either the grader did not cite them among its
    supporting chunks, or a calibrated scorer put them below the bar.

    An empty `supported_indices()` means "nobody judged chunks individually", which must
    read as unknown, never as "none of them".
    """
    if getattr(config, "context_selection_mode", "off") != "adaptive":
        return None, f"context_selection_mode={getattr(config, 'context_selection_mode', 'off')}"

    if not report.meets(CONTEXT_TRIM_MIN_CERTAINTY):
        return None, (
            f"certainty {report.certainty.name.lower()} below "
            f"{CONTEXT_TRIM_MIN_CERTAINTY.name.lower()} required to trim"
        )

    supported = report.supported_indices()
    if not supported:
        return None, "no per-chunk judgement available"

    floor = max(1, int(getattr(config, "context_min_chunks", 1)))
    if len(supported) < floor:
        # Pad from the top of the ranking rather than dropping below the floor: the
        # assessment named fewer chunks than the profile is willing to send.
        extra = [i for i in range(1, len(docs) + 1) if i not in supported]
        supported = sorted(supported + extra[: floor - len(supported)])

    if len(supported) >= len(docs):
        return None, "every chunk carried evidence"
    return supported, f"{len(supported)} of {len(docs)} chunks carried the evidence"
