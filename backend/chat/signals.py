"""What we know about a turn before any expensive work happens.

The agent's first model call carries the system prompt and every tool schema, and it
is paid on every turn — including "thanks", and including questions this corpus was
never going to answer. Retrieval then costs a hybrid search and an evidence
assessment before reaching the same conclusion.

This module produces the signals that let those be avoided, using the same shape as
`backend/rag/evidence.py`: **detectors form a cost-ordered ladder, and the result
carries its own provenance.** Nothing downstream asks "did the classifier LLM run?" —
it asks how certain the signals are. `Certainty` is imported from there rather than
redefined, because a second vocabulary for the same idea is how the two drift apart.

The asymmetry that shapes the ladder: a *high* vector similarity is strong evidence a
question belongs to this corpus, but a *low* one is weak evidence that it does not.
A miss can mean the question is off-topic, or that it is phrased in another language,
or in vocabulary the corpus spells differently. So the cheap rung is trusted to admit
and never to reject — rejection has to be escalated to something that reads meaning.

Detectors report. They never decide: turning signals into actions is
`backend/chat/turn_policy.py`, which is pure.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence

from backend.chat.language import detect_language
from backend.rag.evidence import Certainty
from backend.text_normalization import normalize_query

logger = logging.getLogger(__name__)


class Scope(str, Enum):
    """Whether this corpus is the right place to look."""

    IN_DOMAIN = "in_domain"
    OUT_OF_DOMAIN = "out_of_domain"
    UNKNOWN = "unknown"


@dataclass
class RequestSignals:
    """Everything known about a turn before the agent runs."""

    question: str = ""
    language: str = "en"

    # The question with its references resolved against the conversation, when a
    # resolver ran. Empty means nothing resolved it and `question` is the whole truth —
    # which is why this is not defaulted to `question`: "nobody looked" and "a resolver
    # read the conversation and found nothing to change" are different facts, and only
    # the second is evidence.
    resolved_question: str = ""
    # Conditions set in earlier turns that still bind this answer ("grades up to Year
    # 6"). These narrow an answer, not just a search: retrieval can return the right
    # fee tables and the answer still list every year unless something says otherwise.
    carried_constraints: List[str] = field(default_factory=list)
    # standalone | followup | correction | new_topic. See backend/chat/resolution.py.
    followup_intent: str = "standalone"

    scope: Scope = Scope.UNKNOWN
    scope_certainty: Certainty = Certainty.NONE
    # Corpus sections the question resembles. A HINT for retrieval, never a filter —
    # see turn_policy for why narrowing must never be able to hide a document.
    candidate_sections: List[str] = field(default_factory=list)

    # A closed-set social utterance and nothing else: "thanks", "شكرا".
    is_social: bool = False
    # Profile fields the message appears to disclose. Drives post-turn capture and
    # whether the profile-management tool is worth binding — never an extraction.
    personal_data: List[str] = field(default_factory=list)

    # Rung 1's evidence, kept so rung 2 can be prompted with it rather than left to
    # guess from a topic list. Populated by CatalogueScopeDetector.
    scope_matches: List[Any] = field(default_factory=list)

    # Catalogued questions this query resembled, one per corpus section, best first —
    # the DIRECTIONS the corpus could take this question in. Two or more means the
    # question did not pick between them, and routing may put that choice to the user
    # instead of guessing. These are real questions written at index time, which is what
    # makes a scope_select answerable rather than "please provide more detail".
    scope_options: List[str] = field(default_factory=list)

    assessed_by: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def meets(self, floor: Certainty) -> bool:
        return self.scope_certainty >= floor

    @property
    def effective_question(self) -> str:
        """What the turn is actually about: the resolved form when there is one."""
        return self.resolved_question or self.question

    def as_trace(self) -> Dict[str, Any]:
        return {
            "request_scope": self.scope.value,
            "request_scope_certainty": self.scope_certainty.name.lower(),
            "request_language": self.language,
            "request_resolved_question": self.resolved_question or None,
            "request_carried_constraints": list(self.carried_constraints),
            "request_followup_intent": self.followup_intent,
            "request_is_social": self.is_social,
            "request_personal_data": list(self.personal_data),
            "request_candidate_sections": list(self.candidate_sections),
            "request_scope_options": list(self.scope_options),
            "request_top_match": (
                {"question": self.scope_matches[0].question,
                 "score": round(self.scope_matches[0].score, 4)}
                if self.scope_matches else None
            ),
            "request_assessed_by": list(self.assessed_by),
            "request_reason": "; ".join(self.reasons)[:400] or "n/a",
        }


@dataclass
class SignalContext:
    """What a detector may read. Passed whole so a new signal source does not change
    the interface every detector implements."""

    question: str
    history: Sequence[Any] = ()
    config: Any = None
    # Set by the turn planner before any detector runs — see backend/chat/resolution.py.
    # Detectors read `text_to_score` rather than either field directly, so that "which
    # words describe this turn" is answered in one place instead of once per rung.
    resolved_question: str = ""
    carried_constraints: Sequence[str] = ()
    followup_intent: str = "standalone"

    @property
    def has_history(self) -> bool:
        return bool(self.history)

    @property
    def text_to_score(self) -> str:
        """The words that describe this turn, for anything measuring it.

        The resolved question when a resolver produced one, because it names the
        subject in full. Otherwise the message prefixed by the previous user turn,
        which is the fallback this system used everywhere before resolution existed: a
        blunt instrument — it averages two subjects and the longer one wins — but still
        better than scoring a bare "and what about those?" on its own, which measures
        nothing and looks identical to being off-topic.
        """
        if self.resolved_question:
            return self.resolved_question
        if not self.has_history:
            return self.question
        previous = _last_user_text(self.history)
        return f"{previous}\n{self.question}" if previous else self.question


class Detector(Protocol):
    """One rung. `certainty` is what this detector can establish about scope."""

    name: str
    certainty: Certainty

    def detect(self, ctx: SignalContext, signals: RequestSignals) -> Optional[RequestSignals]:
        """Return updated signals, or None to abstain and let the next rung try."""
        ...


# ---------------------------------------------------------------------------
# Rung 0 — social lookup
# ---------------------------------------------------------------------------

_PUNCTUATION = re.compile(r"[!?.,;:،؛؟…\-–—\s]+")


def _social_key(text: str) -> str:
    """Normalise a message down to just its words, for exact matching."""
    normalized = (normalize_query(text) or text or "").strip().lower()
    return " ".join(part for part in _PUNCTUATION.split(normalized) if part)


class SocialDetector:
    """Recognises a message that is *entirely* a social pleasantry.

    A lookup, not a classifier, and deliberately so. "thanks" is social; "thanks, and
    what are the fees?" is a question wearing a greeting, and a keyword or prefix
    match cannot tell them apart. Getting that wrong means answering a real question
    with "You're welcome!", which is visible to the user in a way that the opposite
    mistake — one wasted search nobody sees — never is.

    So the whole message must match, exactly, after normalisation. Anything with an
    extra clause falls through.
    """

    name = "social"
    certainty = Certainty.HIGH  # an exact match on a closed set is not a judgement call

    def detect(self, ctx: SignalContext, signals: RequestSignals) -> Optional[RequestSignals]:
        phrases = {
            _social_key(phrase)
            for phrase in (getattr(ctx.config, "social_phrases", None) or [])
        }
        key = _social_key(ctx.question)
        if not key or key not in phrases:
            return None

        signals.is_social = True
        # Social turns are not "out of domain" — they are simply not questions. Calling
        # them out-of-domain would route them to a refusal, which is the wrong reply to
        # "thank you".
        signals.scope = Scope.UNKNOWN
        signals.scope_certainty = Certainty.HIGH
        signals.reasons.append(f"whole message matches the social phrase {key!r}")
        return signals


# ---------------------------------------------------------------------------
# Rung 1 — corpus similarity
# ---------------------------------------------------------------------------

class CorpusSimilarityDetector:
    """Compares the question against corpus section vectors.

    Free: the embedding is the one retrieval is about to use anyway, and the
    comparison is a dot product against vectors held in memory.

    **Admits confidently, rejects tentatively.** A strong match is real evidence the
    corpus covers this, and is treated as MEDIUM — enough to proceed without asking a
    model. A weak match is only LOW, because the honest explanations for it include
    "the question is in Arabic and the corpus is in English" and "the corpus calls
    this something else". Rejecting on that alone is how a working system starts
    silently refusing valid questions.

    A follow-up is scored by its RESOLVED wording, so "what about grade 6?" inherits its
    subject instead of scoring like noise — see `SignalContext.text_to_score`.
    """

    name = "corpus_similarity"
    certainty = Certainty.MEDIUM

    def detect(self, ctx: SignalContext, signals: RequestSignals) -> Optional[RequestSignals]:
        from backend.chat.language import detect_language as _  # noqa: F401  (module cohesion)
        from backend.indexing.embedding import embed_query
        from backend.rag.domain_gate import classify, reference_store

        text = ctx.text_to_score
        normalized = normalize_query(text) or text
        try:
            vector = embed_query(normalized)
        except Exception:
            logger.warning("corpus similarity: embedding failed, abstaining", exc_info=True)
            return None

        verdict = classify(vector, reference_store.get(), ctx.config)
        if verdict.abstained:
            signals.reasons.append(f"corpus similarity abstained ({verdict.reason})")
            return None

        signals.candidate_sections = list(verdict.topics)
        if verdict.in_domain:
            signals.scope = Scope.IN_DOMAIN
            signals.scope_certainty = Certainty.MEDIUM
            signals.reasons.append(f"corpus match {verdict.score:.3f}")
        else:
            # Recorded, but at LOW — the next rung decides whether this is real.
            signals.scope = Scope.OUT_OF_DOMAIN
            signals.scope_certainty = Certainty.LOW
            signals.reasons.append(f"no corpus match ({verdict.reason})")
        return signals


def _last_user_text(history: Sequence[Any]) -> str:
    for message in reversed(list(history)):
        role = getattr(message, "type", None) or (
            message.get("role") if isinstance(message, dict) else None
        )
        if role in ("human", "user"):
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                ]
                return " ".join(part for part in parts if part)
    return ""


# ---------------------------------------------------------------------------
# Rung 2 — the envelope call
# ---------------------------------------------------------------------------

class EnvelopeDetector:
    """One small model call that settles scope and reads personal data together.

    This rung is what makes the cheap one safe to ship. The vector gate never has to
    be perfectly calibrated, because being wrong costs this call rather than a silent
    refusal — and because it only ever escalates, the in-corpus majority never
    reaches here at all.

    Both signals come back in a single envelope. Asking separately would double the
    cost of the turns that can least afford it.

    `invoke` is injected so this is testable without a model and swappable per
    deployment; a detector that raises abstains, and abstention means the question
    proceeds.
    """

    name = "envelope"
    certainty = Certainty.HIGH

    def __init__(self, invoke=None):
        self._invoke = invoke

    def detect(self, ctx: SignalContext, signals: RequestSignals) -> Optional[RequestSignals]:
        invoke = self._invoke or _default_envelope_invoke
        result = invoke(ctx.question, ctx.history, ctx.config)
        if not isinstance(result, dict):
            return None

        signals.personal_data = [str(item) for item in (result.get("personal_data") or [])]

        verdict = str(result.get("scope") or "").strip().lower()
        if verdict == Scope.OUT_OF_DOMAIN.value:
            signals.scope = Scope.OUT_OF_DOMAIN
            signals.scope_certainty = Certainty.HIGH
            signals.reasons.append(result.get("reason") or "classifier: outside this corpus")
        elif verdict == Scope.IN_DOMAIN.value:
            # The override that rescues a false rejection from the rung below.
            signals.scope = Scope.IN_DOMAIN
            signals.scope_certainty = Certainty.HIGH
            signals.reasons.append(result.get("reason") or "classifier: belongs to this corpus")
        else:
            signals.reasons.append("classifier returned no usable scope verdict")
            return signals

        # Disclosing a profile field is itself evidence of engagement — someone
        # answering "he is 9" is continuing a conversation, not changing the subject.
        if signals.personal_data and signals.scope is Scope.OUT_OF_DOMAIN:
            signals.scope = Scope.IN_DOMAIN
            signals.reasons.append("overridden: message discloses profile information")
        return signals


def _default_envelope_invoke(question, history, config):  # pragma: no cover - needs a model
    raise NotImplementedError(
        "No envelope classifier is wired. Register one via SignalLadder(envelope=...) "
        "or leave request_envelope_enabled false."
    )


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

class SignalLadder:
    """Runs detectors cheapest-first, stopping once scope is settled well enough."""

    def __init__(self, detectors: Sequence[Detector], required: Certainty = Certainty.MEDIUM):
        self._detectors = list(detectors)
        self._required = required

    def run(self, ctx: SignalContext) -> RequestSignals:
        signals = RequestSignals(
            question=ctx.question,
            language=detect_language(ctx.question),
            resolved_question=ctx.resolved_question,
            carried_constraints=list(ctx.carried_constraints or []),
            followup_intent=ctx.followup_intent,
        )

        for detector in self._detectors:
            try:
                updated = detector.detect(ctx, signals)
            except Exception:
                # A detector is an optimisation. Losing one costs the savings it would
                # have produced, never the turn.
                logger.warning("signal detector %s failed, continuing", detector.name, exc_info=True)
                continue
            if updated is None:
                continue
            signals = updated
            if detector.name not in signals.assessed_by:
                signals.assessed_by.append(detector.name)
            if self._stop(signals):
                break

        if not signals.assessed_by:
            signals.reasons.append("no detector reached a conclusion")
        return signals

    def _stop(self, signals: RequestSignals) -> bool:
        """Stop on a settled scope — but never on a tentative rejection.

        An OUT_OF_DOMAIN verdict below HIGH is exactly the case the next rung exists
        to re-examine, so it must not end the climb no matter what the configured
        floor says.
        """
        if signals.is_social:
            return True
        if signals.scope is Scope.OUT_OF_DOMAIN and signals.scope_certainty < Certainty.HIGH:
            return False
        return signals.meets(self._required)


def build_ladder(config, envelope_invoke=None) -> SignalLadder:
    """Assemble from profile config. Which rungs exist is deployment data.

    Ordered cheapest-first. The scope rungs are imported lazily because they reach the
    catalogue store and the embedder, and this module is imported on every request.
    """
    detectors: List[Detector] = []
    if getattr(config, "social_phrases", None):
        detectors.append(SocialDetector())

    if getattr(config, "scope_index_enabled", False):
        from backend.rag.scope_detector import CatalogueScopeDetector, ScopeModelDetector

        detectors.append(CatalogueScopeDetector())
        if getattr(config, "request_envelope_enabled", False):
            detectors.append(ScopeModelDetector(invoke=envelope_invoke))
    else:
        # Superseded by the catalogue index, which separates far better — a chunk is a
        # fragment, so something is always vaguely near anything. Kept for deployments
        # that have not built a catalogue yet.
        if getattr(config, "domain_gate_enabled", False):
            detectors.append(CorpusSimilarityDetector())
        if getattr(config, "request_envelope_enabled", False):
            detectors.append(EnvelopeDetector(invoke=envelope_invoke))

    return SignalLadder(detectors, required=Certainty.MEDIUM)
