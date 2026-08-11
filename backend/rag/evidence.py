"""The evidence report: one object every retrieval decision reads.

Routing, how much context to send, which images to attach, whether to rewrite — these
look like separate features but they are all answers to one question: *what do we know
about the retrieved evidence, and what does that justify doing?*

Before this module, there was no object for that knowledge, so it lived in control
flow. Each consumer reconstructed it, and the conditional grader made things worse by
FABRICATING a grade when it was skipped — synthetic "strong"/"sufficient" values that
downstream code could not tell apart from a real judgement except by checking a
side-channel flag. Two components computed the same signal independently and agreed
only by luck.

The fix is a single currency with honest provenance:

  * Assessors form a **cost-ordered ladder**. Each declares the certainty it can
    establish. The ladder climbs only as far as the configured requirement, so cheap
    signals answer the easy cases and the LLM is reached only when they cannot.
  * The report has the **same shape on every path**, and carries `certainty` plus
    `assessed_by`. Nothing downstream asks "did the grader run?" — it asks how certain
    the report is. There is no fabricated grade because no type here implies an LLM
    produced it.
  * Policies are **pure functions over the report**. Each declares the minimum
    certainty it needs. Below that, a policy degrades to its conservative default; it
    never goes and computes a signal of its own. That single rule is what stops this
    class of drift from coming back.

Fields that only a language model can honestly fill — ambiguity, HITL prompts —
default to the inert value. A cheap assessor therefore cannot invent an ambiguity it
never assessed; the property is structural rather than special-cased.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


class Certainty(IntEnum):
    """How much the report's conclusion can be trusted.

    Ordered, so a policy states a floor and the ladder states a target. The ordering is
    about the *kind* of evidence behind a conclusion, not the confidence number inside
    it: a lexical signal is LOW however emphatic it sounds, because it measures word
    overlap and not meaning.
    """

    NONE = 0      # nothing has looked at the evidence
    LOW = 1       # structural or lexical proxies only — no semantic judgement
    MEDIUM = 2    # a calibrated semantic score (cross-encoder relevance)
    HIGH = 3      # a language model read the chunks and the question together


_CERTAINTY_BY_NAME = {level.name.lower(): level for level in Certainty}


def parse_certainty(value, default: Certainty = Certainty.HIGH) -> Certainty:
    if isinstance(value, Certainty):
        return value
    return _CERTAINTY_BY_NAME.get(str(value or "").strip().lower(), default)


@dataclass
class ChunkAssessment:
    """What is known about one retrieved chunk.

    `index` is 1-based to match the `[n]` citation markers the answer will use, so a
    consumer can map a citation back to an assessment without a second convention.
    """

    index: int
    score: Optional[float] = None
    supported: Optional[bool] = None  # None = nobody judged this chunk individually
    signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvidenceReport:
    """The single source of truth about one retrieval."""

    question: str = ""
    chunks: List[ChunkAssessment] = field(default_factory=list)
    certainty: Certainty = Certainty.NONE

    relevance: str = "unknown"      # none | weak | strong | unknown
    sufficiency: str = "unknown"    # none | partial | sufficient | unknown
    # Only a language model may move this off "none": deciding that a question is
    # under-specified requires reading it, not scoring it.
    ambiguity: str = "none"         # none | missing_slot | multiple_candidates
    confidence: float = 0.0

    # Whether the retrieved material actually VARIES by the conditions the turn carried
    # forward — "grades up to Year 6", "girls only". yes | no | unknown, and `unknown`
    # is the honest default because only a rung that read the chunks can say.
    #
    # This exists because a carried condition is a reading of the CONVERSATION, not a
    # fact about the corpus, and the two come apart constantly. Per-year fee tables vary
    # by year, so narrowing to Year 6 is correct and necessary. A single admissions
    # document list does not vary by anything, so narrowing to Year 6 matches nothing —
    # and treating that as a failed requirement is how a question with a perfectly good
    # answer in hand ended in a denial. Nothing but a reader can tell those apart.
    constraints_discriminate: str = "unknown"

    # Set only by an assessor that actually chose a route. Cheap assessors leave it
    # None and let the route policy decide.
    preferred_route: Optional[str] = None
    missing_slots: List[str] = field(default_factory=list)
    hitl_prompt: str = ""
    hitl_options: List[str] = field(default_factory=list)

    assessed_by: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks) and self.relevance != "none"

    def supported_indices(self) -> List[int]:
        """1-based indices of chunks judged to carry the evidence.

        Empty when nothing judged chunks individually — which a caller must read as
        "unknown", never as "none of them".
        """
        return [chunk.index for chunk in self.chunks if chunk.supported]

    def meets(self, floor: Certainty) -> bool:
        return self.certainty >= floor

    def as_trace(self) -> Dict[str, Any]:
        """Flattened for the trace. Existing field names are preserved so the frontend
        and any integrating client keep working unchanged."""
        return {
            "evidence_relevance": self.relevance,
            "evidence_answerability": self.sufficiency,
            "evidence_ambiguity": self.ambiguity,
            "evidence_confidence": self.confidence,
            "evidence_reason": "; ".join(self.reasons)[:400] or "n/a",
            "evidence_certainty": self.certainty.name.lower(),
            "evidence_assessed_by": list(self.assessed_by),
            "evidence_supported_chunks": self.supported_indices(),
            "evidence_constraints_discriminate": self.constraints_discriminate,
            "missing_slots": list(self.missing_slots),
        }


@dataclass
class AssessmentContext:
    """Everything an assessor may read. Passed whole so adding a signal source does not
    change the interface every assessor implements."""

    question: str
    docs: List[dict]
    retrieval_meta: Dict[str, Any] = field(default_factory=dict)
    query_embedding: Optional[Any] = None
    config: Any = None
    # Conditions carried in from earlier turns. Passed ALONGSIDE the question rather
    # than folded into it, which was the mistake that motivated this field: a grader
    # asked "do these snippets answer 'what documents are required (grades up to Year
    # 6)'" reads a general document list, sees no year mentioned, and can honestly say
    # it is about a different subject. Relevance is judged against the question; the
    # conditions are reported on separately.
    constraints: List[str] = field(default_factory=list)


class Assessor(Protocol):
    """One rung of the ladder.

    `certainty` is the level this assessor can establish when it reaches a conclusion —
    declared up front so the ladder can order rungs by cost and stop as soon as the
    requirement is met, without knowing what any rung does.
    """

    name: str
    certainty: Certainty

    def assess(self, ctx: AssessmentContext) -> Optional[EvidenceReport]:
        """Return a report, or None to abstain and let the next rung try."""
        ...


def _blank_chunks(docs: Sequence[dict]) -> List[ChunkAssessment]:
    return [ChunkAssessment(index=i) for i, _ in enumerate(docs, 1)]


class StructuralAssessor:
    """Rung 0: facts about the retrieval itself, before anything reads the text.

    Only conclusive in the negative — no documents means no evidence, and that needs no
    model. When documents exist it abstains rather than guessing, which is the whole
    point of a ladder.
    """

    name = "structural"
    certainty = Certainty.HIGH  # "there is nothing here" is not a judgement call

    def assess(self, ctx: AssessmentContext) -> Optional[EvidenceReport]:
        if ctx.docs:
            return None
        return EvidenceReport(
            question=ctx.question,
            chunks=[],
            certainty=Certainty.HIGH,
            relevance="none",
            sufficiency="none",
            confidence=1.0,
            preferred_route="no_knowledge",
            assessed_by=[self.name],
            reasons=["no_retrieved_documents"],
        )


class LexicalAssessor:
    """Rung 1: term overlap between the question and the top chunks.

    Deliberately weak, and LOW certainty however high the overlap. It measures shared
    vocabulary, so it reads 0 whenever the question and the corpus differ in language —
    a cross-lingual hit scores the same as a total miss. It also cannot distinguish a
    passage that answers the question from a heading that merely names it.

    So it is kept as a **cheap negative filter**: a low score is worth recording as a
    reason to keep climbing, while a high score never concludes anything on its own.
    """

    name = "lexical"
    certainty = Certainty.LOW

    def assess(self, ctx: AssessmentContext) -> Optional[EvidenceReport]:
        # Imported here so the module graph stays acyclic: confidence.py owns the
        # tokenisation primitives and knows nothing about reports.
        from backend.rag.confidence import term_coverage

        top_n = int(getattr(ctx.config, "skip_grading_top_chunks", 3))
        coverage = term_coverage(ctx.question, ctx.docs, top_n)

        chunks = _blank_chunks(ctx.docs)
        for chunk in chunks:
            chunk.signals["lexical_coverage"] = coverage

        return EvidenceReport(
            question=ctx.question,
            chunks=chunks,
            certainty=Certainty.LOW,
            # No claim about meaning, so relevance and sufficiency stay unknown. Only
            # the numbers are reported; interpreting them is the ladder's job.
            relevance="unknown",
            sufficiency="unknown",
            confidence=coverage,
            assessed_by=[self.name],
            reasons=[f"lexical coverage {coverage:.2f}"],
        )


class AssessmentLadder:
    """Runs assessors cheapest-first, stopping once the required certainty is reached.

    Escalation, not branching: a new signal is a new rung, and no consumer changes.
    """

    def __init__(self, assessors: Sequence[Assessor], required: Certainty = Certainty.HIGH):
        # Sorted by the certainty each rung can establish, which is also cost order —
        # a stronger conclusion always costs more here.
        self._assessors = sorted(assessors, key=lambda a: a.certainty)
        self._required = required

    def run(self, ctx: AssessmentContext) -> EvidenceReport:
        best = EvidenceReport(question=ctx.question, chunks=_blank_chunks(ctx.docs))

        for assessor in self._assessors:
            report = self._safe_assess(assessor, ctx)
            if report is None:
                continue
            # Stamped here, not left to each rung to remember. Provenance is the reason
            # this object can be trusted, so it cannot depend on a convention.
            if assessor.name not in report.assessed_by:
                report.assessed_by.append(assessor.name)
            best = _merge(best, report)
            if best.meets(self._required):
                break

        if not best.assessed_by:
            best.reasons.append("no assessor reached a conclusion")
        return best

    def _safe_assess(self, assessor: Assessor, ctx: AssessmentContext) -> Optional[EvidenceReport]:
        """An assessor failing must degrade the report's certainty, never the request.

        A rung is an optimisation or an opinion; losing one means climbing further, and
        the ladder already has a rung that cannot fail for lack of a network.
        """
        try:
            return assessor.assess(ctx)
        except Exception:
            logger.warning("evidence assessor %s failed, continuing", assessor.name, exc_info=True)
            return None


def _merge(base: EvidenceReport, incoming: EvidenceReport) -> EvidenceReport:
    """Fold a higher rung's conclusion onto what earlier rungs found.

    Per-chunk signals accumulate — a cheap score and a calibrated one are both worth
    keeping — while conclusions are taken from the more certain report. Reasons are kept
    in full: the trace should show the whole climb, not just the last step.
    """
    merged = EvidenceReport(
        question=incoming.question or base.question,
        certainty=max(base.certainty, incoming.certainty),
        assessed_by=base.assessed_by + [n for n in incoming.assessed_by if n not in base.assessed_by],
        reasons=base.reasons + incoming.reasons,
    )

    authoritative = incoming if incoming.certainty >= base.certainty else base
    other = base if authoritative is incoming else incoming

    merged.relevance = _prefer(authoritative.relevance, other.relevance, "unknown")
    merged.sufficiency = _prefer(authoritative.sufficiency, other.sufficiency, "unknown")
    merged.ambiguity = _prefer(authoritative.ambiguity, other.ambiguity, "none")
    merged.constraints_discriminate = _prefer(
        authoritative.constraints_discriminate, other.constraints_discriminate, "unknown"
    )
    merged.confidence = authoritative.confidence or other.confidence
    merged.preferred_route = authoritative.preferred_route or other.preferred_route
    merged.missing_slots = authoritative.missing_slots or other.missing_slots
    merged.hitl_prompt = authoritative.hitl_prompt or other.hitl_prompt
    merged.hitl_options = authoritative.hitl_options or other.hitl_options

    by_index: Dict[int, ChunkAssessment] = {}
    for chunk in list(base.chunks) + list(incoming.chunks):
        existing = by_index.get(chunk.index)
        if existing is None:
            by_index[chunk.index] = ChunkAssessment(
                index=chunk.index,
                score=chunk.score,
                supported=chunk.supported,
                signals=dict(chunk.signals),
            )
            continue
        existing.signals.update(chunk.signals)
        if chunk.score is not None:
            existing.score = chunk.score
        if chunk.supported is not None:
            existing.supported = chunk.supported
    merged.chunks = [by_index[key] for key in sorted(by_index)]
    return merged


def _prefer(primary: str, fallback: str, inert: str) -> str:
    """Take the authoritative value unless it declined to say anything."""
    if primary and primary not in ("unknown", inert):
        return primary
    if primary and primary != "unknown":
        return primary
    return fallback if fallback else inert


def build_ladder(config, extra: Optional[Sequence[Assessor]] = None) -> AssessmentLadder:
    """Assemble the ladder from profile config.

    Which rungs exist is domain data: an e-commerce profile wants attribute matching
    where a document corpus wants section similarity. Adding a rung is a registration,
    not a change to any consumer.
    """
    required = parse_certainty(getattr(config, "evidence_required_certainty", "high"))
    assessors: List[Assessor] = [StructuralAssessor()]
    if getattr(config, "evidence_lexical_enabled", True):
        assessors.append(LexicalAssessor())
    assessors.extend(extra or [])
    return AssessmentLadder(assessors, required=required)
