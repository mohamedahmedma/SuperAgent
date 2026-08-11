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


def can_ask_human(
    report: EvidenceReport,
    *,
    hitl_rounds: int,
    config,
    scope_options: Sequence[str] = (),
    is_followup: bool = False,
) -> Tuple[bool, str]:
    """Whether it is worth interrupting the user, and why not when it is not.

    Four rules, all learned from what the generic version actually produced.

    **A question must name what it needs.** "I found relevant content, but the evidence
    isn't enough to determine an answer. Please provide more detail" asks the user to
    guess what the system wants. It reads as a failure, it is unanswerable in any
    useful way, and the evidence that prompted it was good enough to attempt an answer
    from. So a HITL turn requires named slots or named options — something the user can
    actually respond to. Without them, answering from partial evidence is better,
    because the answer prompt already says to be explicit about what it does not know.

    **The names may come from the corpus, not only from the grader.** `scope_options`
    are catalogued questions the corpus can answer, matched against this query at
    request time for the cost of one dot product. When the grader says the question has
    several readings but does not enumerate them — a verdict it reaches far more
    reliably than the list it is supposed to attach — those questions are the list. The
    ask that used to be vetoed for lack of specifics now happens, with better specifics
    than the grader would have written, and it happens identically every time because
    nothing sampled it.

    **Once per question.** After the user has answered, asking again spends their
    patience on a system that is not converging. Retrieval that is still ambiguous
    after a clarification will not be rescued by a second one.

    **A follow-up has already picked its subject.** When the turn inherited its subject
    from the conversation, the CATALOGUE directions stop being grounds to interrupt.
    They are computed by similarity against the whole question, and on a follow-up they
    answer "what else is near this?" — which is not the same as "what did the user
    fail to say". Asking someone who just narrowed the topic to re-pick it from a list
    reads as not having been listened to, and it is the exact failure this rule was
    written for: a fees question asked after a uniform question came back offering a
    choice between uniform directions.

    The GRADER's own options survive that rule, and deliberately so. Those come from a
    model that read the retrieved chunks and found several distinct answers sitting in
    them, which is real ambiguity in the evidence rather than a guess about the query.
    """
    limit = int(getattr(config, "max_hitl_rounds", 1))
    if hitl_rounds >= limit:
        return False, f"already asked {hitl_rounds} time(s), limit {limit}"

    if report.ambiguity == "missing_slot":
        if not report.missing_slots:
            return False, "missing_slot without naming which slot"
        return True, "missing a named condition"

    if report.ambiguity == "multiple_candidates":
        if report.hitl_options:
            return True, "several named candidates"
        if is_followup:
            return False, "a follow-up inherits its subject; catalogued directions cannot narrow it"
        if len(scope_options) > 1:
            return True, f"{len(scope_options)} catalogued directions to choose from"
        return False, "multiple_candidates without naming the candidates"

    return False, "nothing specific to ask for"


def offerable_directions(scope_options: Sequence[str], *, is_followup: bool = False) -> List[str]:
    """The corpus directions worth putting to the user, or nothing.

    One direction is not a choice — it is the answer, and asking about it would be a
    question with a single option. The detector has already kept only the catalogued
    questions that clear the corpus's derived floor, that are not paraphrases of one
    another, and that are not far behind the leader, so reaching two here means the
    query landed near two genuinely different things the corpus can answer without
    picking either.

    A follow-up offers nothing, for the reason `can_ask_human` gives: its subject came
    from the conversation, so a list of what else the corpus knows cannot narrow it.
    """
    if is_followup:
        return []
    options = [str(option).strip() for option in scope_options if str(option).strip()]
    return options if len(options) > 1 else []


def decide_route(
    report: EvidenceReport,
    *,
    has_docs: bool,
    rewrite_count: int,
    is_sub_agent: bool,
    config,
    hitl_rounds: int = 0,
    scope_options: Sequence[str] = (),
    is_followup: bool = False,
) -> Tuple[str, str]:
    """The next graph step, and why.

    **A denial requires that nothing on the subject was retrieved.** That is the one
    invariant here, and every branch below is written to preserve it. `no_knowledge`
    means the corpus has nothing about what was asked — it does not mean the snippets
    fell short of settling the question. Those are different facts and the user reads
    them as the same sentence: a system that says "no reliable relevant information"
    while holding three chunks about partnerships is simply wrong, and the user has no
    way to recover from it.

    The failure that motivated this: "what is partner" retrieved the partner section and
    the figure listing every partner, the grader judged that those snippets do not
    *define* the term, and the turn ended in a denial with the partner image attached to
    it. Evidence on the subject now routes to `answer` — after one rewrite if the budget
    allows — and the answer prompt is already required to say what the sources leave
    open. Only `relevance == "none"` still denies.

    Two older departures from the original grade-driven routing stand: it refuses to
    invent sufficiency where nothing assessed the evidence, and it will not interrupt
    the user unless it can say what it needs — see `can_ask_human`.
    """
    max_rewrites = int(getattr(config, "max_rewrites", 1))
    required = parse_certainty(getattr(config, "evidence_required_certainty", "high"))
    directions = offerable_directions(scope_options, is_followup=is_followup)
    ask_allowed, ask_reason = can_ask_human(
        report,
        hitl_rounds=hitl_rounds,
        config=config,
        scope_options=directions,
        is_followup=is_followup,
    )

    if not has_docs:
        return "no_knowledge", "no evidence retrieved"
    # The only judgement that may end a turn in a denial: an assessor read the snippets
    # and placed them on a different subject.
    if report.relevance == "none":
        return "no_knowledge", "retrieved evidence is about a different subject"

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

    if report.relevance == "strong" and report.sufficiency == "sufficient":
        return "answer", "evidence sufficient"

    # Everything past here retrieved something that no assessor placed on a different
    # subject. `on_subject` is that judgement, asked once: below sufficient, but about
    # what was asked. It is the difference between an answer that reports what the
    # sources cover and a denial that contradicts them.
    on_subject = report.relevance in ("weak", "strong")

    # Sub-questions get no correction pass: evidence that falls short is left for
    # synthesis to merge across siblings, which is a better fix than rewriting one
    # sub-question. A sibling may carry the half this one is missing.
    if is_sub_agent:
        if on_subject:
            return "answer", "sub-agent keeps on-subject evidence for synthesis"
        return "no_knowledge", "sub-agent retrieved nothing judged to be on the subject"

    if on_subject or report.preferred_route == "rewrite":
        # ASK BEFORE REWRITING. Both close the same gap — the question did not say which
        # of several things it meant — and only one of them can actually close it.
        #
        # The rewrite guesses: a FAST_MODEL call to plan it, a second retrieval, a second
        # grader call, and then an answer over the union of both passes. Measured on this
        # deployment, "what is partner" took that route and spent 38 seconds and 12.3K
        # tokens to produce a good answer to a question the user had not quite asked. The
        # same turn, when routed to the user instead, came back in one exchange with the
        # direction named — faster, cheaper, and certain rather than inferred.
        #
        # So when there are real directions to offer, offer them. `can_ask_human` still
        # holds the once-per-question limit, so the fallback below is what a user who has
        # already answered gets, and the rewrite remains for questions with nothing
        # specific to ask about.
        #
        # `directions` is empty on a follow-up, which is what keeps this branch off the
        # turn that motivated the rule: the subject was settled by the previous message,
        # so a list of neighbouring catalogue entries cannot narrow anything and asking
        # from it interrupts someone who has already answered the question being asked.
        if directions and hitl_rounds < int(getattr(config, "max_hitl_rounds", 1)):
            return "scope_select", f"{len(directions)} catalogued directions, asking before rewriting"
        # A rewrite is the cheap chance to close the gap before answering from less
        # than the question asked for. It is not a gate: when the budget is spent, the
        # evidence is answered from, never denied.
        if rewrite_count < max_rewrites:
            return "rewrite", "evidence below sufficient, one rewrite remains"
        return _answer_from_what_there_is(report, ask_allowed, ask_reason, "rewrites exhausted")

    # Docs exist and nothing concluded anything about them — not even that they are
    # off-subject. Only reachable below the HIGH requirement, where no rung reads
    # meaning; claiming an answer from it would be the fabricated-grade pattern.
    return "no_knowledge", "no assessment placed the retrieved evidence on the subject"


def _answer_from_what_there_is(report, ask_allowed: bool, ask_reason: str, context: str) -> Tuple[str, str]:
    """What to do with on-subject evidence once rewriting is spent.

    Answer from it, unless there is a specific question worth asking. Evidence that
    falls short of settling the question is still evidence, and the answer prompt
    already requires the model to be explicit about what the sources do not cover —
    which is more useful to the user than a request to rephrase, and far more useful
    than a denial issued while holding the material.
    """
    if ask_allowed:
        route = "scope_select" if report.ambiguity == "multiple_candidates" else "clarify"
        return route, f"{context}, {ask_reason}"
    if report.relevance in ("weak", "strong"):
        grade = (
            "partial evidence" if report.sufficiency == "partial"
            else f"{report.relevance} on-subject evidence"
        )
        return "answer", f"{context}; answering from {grade} ({ask_reason})"
    return "no_knowledge", context


def select_context_indices(
    report: EvidenceReport,
    docs: Sequence[dict],
    config,
    *,
    answer_ceiling: Optional[int] = None,
) -> Tuple[Optional[List[int]], str]:
    """Which chunks (1-based) to send to the answer prompt, or None to send all.

    The safety property that the first attempt at this got wrong: never answer from a
    narrower set than the one judged sufficient. So the only chunks droppable are those
    an assessment positively excluded — either the grader did not cite them among its
    supporting chunks, or a calibrated scorer put them below the bar.

    An empty `supported_indices()` means "nobody judged chunks individually", which must
    read as unknown, never as "none of them".

    `answer_ceiling` is the one bound that does not need a judgement: however the doc set
    grew, one answer never needs more chunks than a single retrieval would have returned.
    It only ever engages after a rewrite merged two passes, and it keeps the best-ranked
    prefix — the grader still judged the whole union, so nothing was hidden from the
    decision, only from the prompt.
    """
    if getattr(config, "context_selection_mode", "off") != "adaptive":
        return None, f"context_selection_mode={getattr(config, 'context_selection_mode', 'off')}"

    if not report.meets(CONTEXT_TRIM_MIN_CERTAINTY):
        return None, (
            f"certainty {report.certainty.name.lower()} below "
            f"{CONTEXT_TRIM_MIN_CERTAINTY.name.lower()} required to trim"
        )

    ceiling = int(answer_ceiling) if answer_ceiling else 0

    supported = report.supported_indices()
    if not supported:
        if ceiling and len(docs) > ceiling:
            return (
                list(range(1, ceiling + 1)),
                f"no per-chunk judgement; kept the top {ceiling} of {len(docs)} by rank",
            )
        return None, "no per-chunk judgement available"

    if ceiling and len(supported) > ceiling:
        supported = supported[:ceiling]
        return supported, f"{len(supported)} chunks carried the evidence, capped at {ceiling}"

    floor = max(1, int(getattr(config, "context_min_chunks", 1)))
    if len(supported) < floor:
        # Pad from the top of the ranking rather than dropping below the floor: the
        # assessment named fewer chunks than the profile is willing to send.
        extra = [i for i in range(1, len(docs) + 1) if i not in supported]
        supported = sorted(supported + extra[: floor - len(supported)])

    if len(supported) >= len(docs):
        return None, "every chunk carried evidence"
    return supported, f"{len(supported)} of {len(docs)} chunks carried the evidence"
