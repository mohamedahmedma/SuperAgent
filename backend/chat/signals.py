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
from backend.text_matching import name_key
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

    # Whether this message is about one of the caller's own children, as opposed to
    # the school in general. Reported by the classifier node, which reads the message
    # and the conversation; never derived from a word list, because the commonest form
    # in Arabic attaches the possessive to the end of the word ("جدوله" — his
    # timetable) where no marker can reach it.
    #
    # A HINT about the message, in the same class as `personal_data`. It says nothing
    # about whether this caller has children or may read them; that is decided later,
    # by code holding a verified identity this module deliberately cannot see.
    about_child: bool = False
    # How the child was referred to: son | daughter | child | plural | named |
    # context | none. Narrows which child is meant without asking, when the roster
    # makes it unambiguous.
    child_reference: str = "none"
    # A name the message actually contained, verbatim. Empty unless `child_reference`
    # is "named". Never resolved here — matching it to a real child is done against
    # the school's own roster, by something that has one.
    child_name: str = ""
    # What answering would have to READ: the child's own record, the school's material,
    # or both. See CHILD_QUESTION_KINDS.
    #
    # Defaults to `both`, and that default is the whole safety argument for the tool
    # narrowing it drives: a classifier that did not run, abstained, returned a value
    # outside the closed set, or hit its rate limit leaves this at `both`, which binds
    # every tool and is exactly the behaviour that existed before this field. Narrowing
    # is an optimisation, so it may only ever happen on a positive answer.
    child_question_kind: str = "both"

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
            "request_about_child": self.about_child,
            "request_child_reference": self.child_reference,
            "request_child_question_kind": self.child_question_kind,
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

#: How a message referred to a child. Closed, because every value has a rule attached
#: downstream — a free-text reference would be a string nobody could branch on.
CHILD_REFERENCES = ("none", "son", "daughter", "child", "plural", "named", "context")

#: What answering a child question actually needs to READ. Closed for the same reason,
#: and `both` is first because it is the value everything degrades to.
#:
#: `about_child` cannot stand in for this, and the prompt says why in as many words: "the
#: question is about a general school matter but asked FOR that child specifically ('what
#: are the fees for my son?' — the fee schedule is general, the year group is his)". That
#: is `about_child` true and a KNOWLEDGE question, and «مصاريف ابني» is one of the
#: commonest messages this deployment gets. A tool binding that read `about_child` as
#: "records" would answer it from the wrong place every time.
CHILD_QUESTION_KINDS = ("both", "records", "school_matter")


class EnvelopeDetector:
    """The classification node: one small model call, three decisions.

    A node, not an agent. It runs at a fixed point in the turn, reads one message,
    returns a structured verdict, and stops. Nothing here chooses tools, loops, or
    talks to the user — those are the properties that make an agent expensive and
    unpredictable, and a classifier needs none of them. It costs one call on
    FAST_MODEL and that is the whole of it.

    **Three decisions in one envelope.** Scope, disclosed profile fields, and — where
    the deployment has children to talk about — whether the message is about one of
    the caller's own. They share every scrap of context they need, so asking
    separately would triple the cost of the one call every turn already pays.

    This rung is also what makes the cheap ones safe to ship: the vector gate never
    has to be perfectly calibrated, because being wrong costs this call rather than a
    silent refusal.

    **It is never told who is asking.** `SignalContext` carries no identity, by
    design. This node reports what the MESSAGE says; whether the caller has children,
    or may read them, is settled afterwards by code that has verified a signature. A
    classifier that knew the caller could be argued into changing its answer about
    them.

    `invoke` is injected so this is testable without a model and swappable per
    deployment. A detector that raises abstains, and abstention means the question
    proceeds — losing this node costs the savings it would have produced, never the
    turn.
    """

    name = "envelope"
    certainty = Certainty.HIGH

    def __init__(self, invoke=None):
        self._invoke = invoke

    def detect(self, ctx: SignalContext, signals: RequestSignals) -> Optional[RequestSignals]:
        invoke = self._invoke or _default_envelope_invoke
        # `text_to_score`, not the raw message: a follow-up's own words do not name its
        # subject, and judging "and what about those?" on its own measures nothing and
        # looks identical to being off-topic. This is the same text the rung below
        # scored, so the two cannot disagree about what the turn is about.
        result = invoke(ctx.text_to_score, ctx.history, ctx.config)
        if not isinstance(result, dict):
            return None

        signals.personal_data = [str(item) for item in (result.get("personal_data") or [])]
        # Deliberately NOT text_to_score. With no resolver that falls back to gluing the
        # previous user turn onto this one — the module's own "blunt instrument" — and a
        # name from the turn before is exactly the carried-over guess the check exists to
        # catch. The resolved question is a different matter: a resolver naming the child
        # is the designed path, and its name is evidence.
        self._read_child(
            result, signals, classified_text=ctx.resolved_question or ctx.question
        )

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

        # A question about this parent's own child is by definition this assistant's
        # subject. Enforced here rather than asked for in the prompt, because a prompt
        # is a request and this has to hold every time: refusing "how is my son doing?"
        # as off-topic is the single worst answer this deployment can give.
        if signals.about_child and signals.scope is Scope.OUT_OF_DOMAIN:
            signals.scope = Scope.IN_DOMAIN
            signals.reasons.append("overridden: the message is about the caller's child")
        return signals

    @staticmethod
    def _read_child(result: dict, signals: RequestSignals, *, classified_text: str = "") -> None:
        """Take the child verdict, distrusting every field of it.

        A model returning `about_child` with a reference outside the closed set, or a
        name it invented, would otherwise select a child — and selecting the wrong
        child shows one family's marks while naming another. Everything unrecognised
        degrades to the shape that asks rather than the shape that guesses.
        """
        if not bool(result.get("about_child")):
            return
        signals.about_child = True

        reference = str(result.get("child_reference") or "").strip().lower()
        signals.child_reference = reference if reference in CHILD_REFERENCES else "context"
        if signals.child_reference == "none":
            # "About a child, referred to in no way at all" is not a state. Read it as
            # the conversation carrying the subject, which is what it almost always is.
            signals.child_reference = "context"

        name = " ".join(str(result.get("child_name") or "").split())
        # A name is only meaningful when the model said the message contained one.
        # Carried on any other reference kind it is a guess.
        if signals.child_reference != "named":
            name = ""
        elif not _names_the_child(classified_text, name):
            # Measured, not hypothetical: asked to classify "طيب وجدوله؟" after a turn
            # about علي, the model returns `named` with "علي" — resolving the reference
            # itself, which the prompt forbids precisely because it is not the thing
            # holding the school's list of children.
            #
            # It guesses right when one child has been discussed and wrong the moment
            # two have, and a wrong name here OVERRIDES a correct pin. Enforced in code
            # rather than asked for again in wording, because a prompt is a request and
            # this has to hold every time.
            #
            # Checked against the text the node actually classified, which is the
            # resolved question when a resolver produced one — so a name a resolver
            # legitimately supplied still counts.
            signals.reasons.append(
                "classifier named a child the message does not name — treated as context"
            )
            name = ""
            signals.child_reference = "context"

        signals.child_name = name
        if signals.child_reference == "named" and not name:
            signals.child_reference = "context"

        # Distrusted exactly like the reference above it, and degrading to the same
        # place: anything outside the closed set becomes `both`, which binds every tool.
        # A wrong value here does not select a child — it selects a TOOL — so the cost of
        # being wrong is an answer looked up in the wrong place, and the cost of
        # abstaining is one tool schema nobody used.
        kind = str(result.get("child_question_kind") or "").strip().lower()
        signals.child_question_kind = kind if kind in CHILD_QUESTION_KINDS else "both"

        signals.reasons.append(
            f"classifier: about the caller's child ({signals.child_reference}, "
            f"needs {signals.child_question_kind})"
        )


def _names_the_child(text: str, name: str) -> bool:
    """Whether `name` actually appears in the classified message.

    Folded on both sides with the same function the roster matcher uses, so a difference
    of alif form, teh marbuta, tatweel or an invisible character is not read as a
    different name. A plain containment test after that: the question is only ever "did
    these words appear", never "who is this".

    `name_key`, not `normalize_query`: the latter repairs PDF damage but preserves hamza
    and teh marbuta, so it would still read «أحمد» reported by the classifier and «احمد»
    typed by the parent as two different names — which is exactly the mismatch this
    check exists to survive.
    """
    if not name:
        return False
    haystack = name_key(text)
    needle = name_key(name)
    return bool(needle) and needle in haystack


def _default_envelope_invoke(question, history, config):  # pragma: no cover - needs a model
    """One structured call on FAST_MODEL. The classification node's only I/O.

    Shaped exactly like `backend/rag/scope_detector._default_scope_invoke`, its
    catalogue-backed twin, because the two answer the same question with different
    evidence and diverging on the plumbing would make them diverge on the verdict. The
    difference is what they can show the model: that one has real indexed questions and
    similarity scores, this one has a prose description of what the deployment covers.

    Weaker evidence is why the prompt leans harder on explicit rules and worked
    examples — see `chat/request_envelope.j2`.
    """
    import os

    from langchain.chat_models import init_chat_model
    from pydantic import BaseModel, Field
    from typing import List as _List, Literal as _Literal

    from backend.assets.vision import call_with_rate_limit_retry, invoke_structured
    from backend.llm import sampling
    from backend.profiles import get_profile
    from backend.prompts import render

    profile = get_profile()
    personal_fields = list(getattr(config, "personal_data_fields", None) or [])
    child_context = bool(getattr(config, "child_context_enabled", False))

    class RequestEnvelope(BaseModel):
        """Every field is declared and every field is required.

        Providers enforcing OpenAI-style strict structured output reject a schema whose
        properties are optional, and a model told to "omit the field when it does not
        apply" omits it rather than sending the empty value. Defaults here mean a
        missing field is read as its safe value instead of failing the call — the same
        arrangement `RewritePlan` documents in rag/rewrite.j2.
        """

        scope: _Literal["in_domain", "out_of_domain"] = Field(
            description="Whether the message is this assistant's subject"
        )
        reason: str = Field(default="", description="One short sentence")
        personal_data: _List[str] = Field(
            default_factory=list,
            description="Field names from the supplied list that the message discloses",
        )
        about_child: bool = Field(
            default=False,
            description=(
                "True when the message asks about a particular child's own situation "
                "rather than about the school in general"
            ),
        )
        child_reference: _Literal[
            "none", "son", "daughter", "child", "plural", "named", "context"
        ] = Field(default="none", description="How the child was referred to")
        child_name: str = Field(
            default="",
            description="A name the message actually contained; empty otherwise",
        )
        child_question_kind: _Literal["both", "records", "school_matter"] = Field(
            default="both",
            description=(
                "What answering needs to read: 'records' for the child's own marks, "
                "attendance or report; 'school_matter' for the school's own material "
                "asked about this child; 'both' when it needs each, or when unsure"
            ),
        )

    prompt = render(
        "chat/request_envelope.j2",
        question=question,
        persona=profile.identity.persona,
        coverage=getattr(config, "coverage", "") or "",
        history=_history_text(history, config),
        personal_fields=personal_fields,
        child_context=child_context,
    )
    model = init_chat_model(
        model=getattr(config, "scope_summary_model", "") or os.getenv("FAST_MODEL"),
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        **sampling("scope"),
    )

    # Same quota as everything else in the turn, so the same treatment. A 429 here makes
    # this node abstain, which leaves scope UNKNOWN — safe, since nothing may end a turn
    # on an unsettled scope, but it spends the search this node existed to avoid and it
    # loses the child signal for the turn.
    class _Retry:
        vision_retry_attempts = int(getattr(config, "model_retry_attempts", 3))
        vision_retry_base_seconds = float(getattr(config, "model_retry_base_seconds", 5.0))
        vision_retry_max_seconds = float(getattr(config, "model_retry_max_seconds", 60.0))

    result = call_with_rate_limit_retry(
        lambda: invoke_structured(model, RequestEnvelope, [{"role": "user", "content": prompt}]),
        config=_Retry(),
        description="request classifier",
    )
    return result if isinstance(result, dict) else result.model_dump()


def _history_text(history, config) -> str:
    """The recent conversation as plain dialogue, for the classifier to read.

    Both sides. A follow-up points at what the ASSISTANT said as often as at what the
    user did — "and what about her attendance?" refers to a child only the assistant
    named — and a history containing one side cannot resolve that.

    Reuses the renderer query resolution already uses, so the two nodes read the same
    conversation in the same shape. A second way of flattening history is a second set
    of truncation rules to keep in step.
    """
    from backend.chat.resolution import conversation_text

    limit = int(getattr(config, "query_resolution_history_messages", 6) or 6)
    try:
        return conversation_text(history or [], limit=limit)
    except Exception:  # pragma: no cover - a classifier must never fail on its context
        return ""


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
