"""What the user asked, as opposed to what they typed.

A follow-up carries its subject in the conversation rather than in its own words. Three
places in this system used to work around that independently, and each invented its own
idea of what the turn was about:

  * the scope detector concatenated the previous user turn with this one and embedded
    the pair
  * the agent wrote a retrieval query in its own words, with no instruction to resolve
    anything
  * the HITL resume path built `"{answer}: {previous query}"` with string formatting

None of those is resolution. Concatenation produces a vector averaging two subjects, and
the older turn — usually the longer and more content-bearing one — dominates it: asking
"and what is the fees for this years" after a question about uniforms scored nearest to
the catalogue's *uniform* questions, and the user was offered a choice between uniform
directions for a question about fees. Formatting is worse still, because it cannot
express replacement: "no I mean the fees" concatenated onto the reading it was
correcting retrieves both readings and answers from the union.

So resolution happens once, here, and its result is the single thing the rest of the
turn reads. That is the same rule `backend/chat/turn_policy.py` states for decisions:
one component establishes a fact, everything else reads it, and there is no second
derivation to drift from the first.

## It is gated, not paid for

`needs_resolution` settles most turns locally and for free. A first message has nothing
to inherit. A message long enough to carry its own subject, with no anaphora and no
leading conjunction, is already standalone. What reaches the model is the short,
referential minority — and on those turns the call usually pays for itself twice over,
because a resolved question matches the scope catalogue above its floor (so the scope
model never runs) and does not trigger the clarification round-trip that an
under-specified one does.

## Failure is abstention

Every failure path returns the message unchanged with `resolved=False`, which is exactly
the behaviour that existed before this module. A resolver that is misconfigured, rate
limited, or returning nonsense costs the improvement, never the turn.
"""
from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from backend.rag.query_translation import needs_translation, remember_translation
from backend.text_normalization import normalize_query
from typing import List as _List, Literal as _Literal
from pydantic import Field

from backend.structured_output import StructuredOutput

logger = logging.getLogger(__name__)

# The four readings of a message, given what came before it.
STANDALONE = "standalone"
FOLLOWUP = "followup"
CORRECTION = "correction"
NEW_TOPIC = "new_topic"

INTENTS = (STANDALONE, FOLLOWUP, CORRECTION, NEW_TOPIC)

_WHITESPACE = re.compile(r"\s+")


@dataclass
class ResolvedQuestion:
    """One turn's information need, with its inherited conditions named separately.

    `resolved` distinguishes "a model read the conversation and this is what it said"
    from "nothing ran, so this is the raw message". Callers need that distinction: the
    first is evidence about the turn, the second is the absence of evidence, and
    treating them alike is how a disabled resolver would start looking like a confident
    verdict of `standalone`.
    """

    question: str
    constraints: List[str] = field(default_factory=list)
    intent: str = STANDALONE
    resolved: bool = False
    reason: str = ""
    #: The question in the CORPUS's language, when this turn needed translating and the
    #: translation passed verification. Retrieval reads it; nothing else does. Empty
    #: whenever translation was not needed, not enabled, or not trustworthy.
    search_text: str = ""

    @property
    def is_followup(self) -> bool:
        """Whether the subject came from the conversation rather than this message.

        Read by routing: a turn whose subject was settled one message ago must not be
        handed back to the user as a choice between subjects.
        """
        return self.intent in (FOLLOWUP, CORRECTION)

    @property
    def supersedes_pending_question(self) -> bool:
        """Whether a pending clarification should be abandoned rather than resumed.

        The resume path assumes the reply either picks an offered option or fills a
        named slot. A correction does neither — it says the question was read wrongly —
        and a new topic abandons it outright. Both are answered by starting a fresh
        turn from `question`, not by folding the reply into the old query.
        """
        return self.intent in (CORRECTION, NEW_TOPIC)

    def as_trace(self) -> dict:
        return {
            "turn_resolved_question": self.question if self.resolved else None,
            "turn_carried_constraints": list(self.constraints),
            "turn_followup_intent": self.intent,
            "turn_search_text": self.search_text or None,
        }


def unresolved(question: str, reason: str) -> ResolvedQuestion:
    """The message as typed. Every abstention path returns this."""
    return ResolvedQuestion(question=question, intent=STANDALONE, resolved=False, reason=reason)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def needs_resolution(question: str, history: Sequence[Any], config) -> Tuple[bool, str]:
    """Whether this message can only be understood from the conversation.

    Free, and deliberately biased toward *not* calling the model: a false negative
    leaves today's behaviour, while a false positive spends a small call on a question
    that did not need one. The signals are the ones that actually distinguish a
    referential message — it points at something ("this", "these", "دي"), it opens as a
    continuation ("and what...", "وما"), or it is too short to carry a subject at all.
    """
    if not getattr(config, "query_resolution_enabled", False):
        return False, "query resolution disabled"
    if not history:
        return False, "first message in the session, nothing to inherit"

    text = _normalized(question)
    if not text:
        return False, "empty message"

    for opener in _phrases(getattr(config, "followup_openers", None)):
        if text.startswith(opener):
            return True, f"opens as a continuation ({opener!r})"

    for marker in _phrases(getattr(config, "followup_markers", None)):
        if _contains_marker(text, marker):
            return True, f"refers back ({marker!r})"

    ceiling = int(getattr(config, "query_resolution_max_chars", 0) or 0)
    if ceiling and len(text) <= ceiling:
        return True, f"short enough ({len(text)} chars) to be carrying its subject in the conversation"

    return False, "message carries its own subject"


def _contains_marker(text: str, marker: str) -> bool:
    """Substring match, but only on a whole word for markers that are one.

    "it" must not fire on "admission", and "ده" must not fire on "دهانات". A marker
    containing a space is already specific enough that a plain substring test is right
    — and for Arabic it is the only test available, because the language attaches
    conjunctions and articles directly to the word.
    """
    if " " in marker:
        return marker in text
    return re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", text) is not None


def _phrases(values) -> List[str]:
    return [phrase for phrase in (_normalized(item) for item in (values or [])) if phrase]


def _normalized(text: Any) -> str:
    raw = str(text or "")
    return _WHITESPACE.sub(" ", (normalize_query(raw) or raw)).strip().lower()


# ---------------------------------------------------------------------------
# Conversation rendering
# ---------------------------------------------------------------------------

def message_role_and_text(message: Any) -> Tuple[str, str]:
    """`("user"|"assistant"|"", text)` for a LangChain message or a plain dict."""
    role = getattr(message, "type", None) or (
        message.get("role") if isinstance(message, dict) else None
    )
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        text = " ".join(part for part in parts if part)
    elif isinstance(content, str):
        text = content
    else:
        text = ""

    if role in ("human", "user"):
        return "user", text
    if role in ("ai", "assistant"):
        return "assistant", text
    return "", text


def conversation_text(history: Sequence[Any], limit: int = 6, max_chars: int = 600) -> str:
    """The last `limit` messages as plain dialogue, newest last.

    Both sides, not just the user's. The assistant's replies are what a follow-up
    usually points at — "the fees you mentioned" refers to something only the assistant
    said — and a history containing only the user's turns cannot resolve that.

    Each message is clipped to `max_chars` because a resolver needs the *subject* of an
    earlier answer, not the answer. An untrimmed assistant turn is easily longer than
    everything else in this prompt combined.
    """
    lines: List[str] = []
    for message in list(history)[-max(1, limit):]:
        role, text = message_role_and_text(message)
        # Blocks out, before the clip. A rendered record is most of the message it is
        # attached to, so a 600-character window spent on lesson rows leaves nothing of
        # the sentence that says what the turn was about — and a follow-up then resolves
        # against the table instead of the subject. See `answer_blocks.strip_answer_blocks`.
        if role == "assistant":
            from backend.chat.answer_blocks import strip_answer_blocks

            text = strip_answer_blocks(text)
        clean = _WHITESPACE.sub(" ", text or "").strip()
        if not role or not clean:
            continue
        if len(clean) > max_chars:
            clean = clean[:max_chars].rstrip() + "…"
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {clean}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_question(
    question: str,
    history: Sequence[Any],
    config,
    *,
    invoke=None,
    hitl_prompt: str = "",
    hitl_options: Sequence[str] = (),
) -> ResolvedQuestion:
    """The standalone form of `question`, or the message itself when nothing ran.

    Never raises. `invoke` is injected so this is testable without a model and
    replaceable per deployment; the same signature is what the tests use.
    """
    # Checked before the gate, and separately from it: a pending clarification is reason
    # enough to resolve a message the gate would have passed over, but a deployment that
    # switched this off must get no model call on any path.
    if not getattr(config, "query_resolution_enabled", False):
        return unresolved(question, "query resolution disabled")

    wanted, reason = needs_resolution(question, history, config)
    # The gate is a UNION, and the prompt is assembled from whichever half fired. Two jobs
    # want a small model to read this message: working out what a follow-up refers to, and
    # writing it in the language the corpus is indexed in. They are independent — an
    # Arabic message that stands on its own needs only the second, an English follow-up
    # only the first — so gating one behind the other would either skip most translations
    # or pay for resolution on every turn. One call does whichever is needed, the template
    # includes only those instructions, and when neither is wanted there is no call.
    translating, translation_reason = needs_translation(question)
    if not wanted and not translating and not hitl_prompt:
        return unresolved(question, reason)

    rendered = conversation_text(
        history,
        limit=int(getattr(config, "query_resolution_history_messages", 6) or 6),
    )
    if not rendered and not hitl_prompt and not translating:
        return unresolved(question, "no usable conversation text to resolve against")

    call = invoke or _default_resolve_invoke
    try:
        result = _call_resolver(
            call, question, rendered, config, hitl_prompt, list(hitl_options or []),
            resolving=bool(wanted or hitl_prompt),
            translating=translating,
        )
    except Exception:
        logger.warning("query resolution failed; using the message as written", exc_info=True)
        return unresolved(question, "resolver error")

    if not isinstance(result, dict):
        return unresolved(question, "resolver returned no usable result")

    resolved_text = str(result.get("question") or "").strip()
    intent = str(result.get("intent") or "").strip().lower()
    if intent not in INTENTS:
        intent = FOLLOWUP if resolved_text and resolved_text != question else STANDALONE

    # A resolver that returns nothing has abstained, whatever else it said. Substituting
    # an empty question would search for nothing and deny the turn. A translation that
    # arrived on the same call still stands: it was verified separately and is already in
    # the memo retrieval reads.
    if not resolved_text:
        outcome = unresolved(question, "resolver returned an empty question")
        outcome.search_text = _accept_translation(question, question, result, translating)
        return outcome

    limit = max(0, int(getattr(config, "carried_constraint_limit", 4) or 0))
    constraints: List[str] = []
    for item in result.get("constraints") or []:
        text = _WHITESPACE.sub(" ", str(item or "")).strip()
        if text and text.lower() not in {existing.lower() for existing in constraints}:
            constraints.append(text)

    # A new topic inherits nothing by definition. Enforced here rather than asked for in
    # the prompt, because a constraint that survives a subject change is the failure
    # this whole mechanism exists to prevent, only pointed the other way.
    if intent == NEW_TOPIC:
        constraints = []

    return ResolvedQuestion(
        question=resolved_text,
        constraints=constraints[:limit],
        intent=intent,
        resolved=True,
        reason=reason or translation_reason or "resolved against a pending clarification",
        search_text=_accept_translation(question, resolved_text, result, translating),
    )


def _accepts_job_flags(call) -> bool:
    """Whether `call` can be told WHICH jobs this turn needs.

    `invoke` is a documented extension point — "replaceable per deployment" — and it grew
    two keyword arguments when one call started doing resolution and translation. An older
    implementation that does not accept them raises `TypeError`, which the caller's broad
    `except` would swallow as "resolver error": resolution would stop happening, silently,
    and look like a model that had simply gone quiet. Asking the signature is cheaper than
    finding that out from a drop in follow-up accuracy.
    """
    try:
        parameters = inspect.signature(call).parameters
    except (TypeError, ValueError):
        return True
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return True
    return {"resolving", "translating"} <= set(parameters)


def _call_resolver(call, question, rendered, config, hitl_prompt, hitl_options, **flags):
    """The resolver call, with the job flags when the callable can take them."""
    if _accepts_job_flags(call):
        return call(question, rendered, config, hitl_prompt, hitl_options, **flags)
    logger.debug(
        "resolver callable predates the job flags; calling it the old way (no translation)"
    )
    return call(question, rendered, config, hitl_prompt, hitl_options)


def _accept_translation(raw: str, resolved: str, result: dict, translating: bool) -> str:
    """The reply's `search_text`, if it survives the same checks a standalone one faces.

    Validated against the RESOLVED question, because that is what it is a translation of:
    "وللدولي؟" carries no year group, and the resolution that recovered "Year 3" is what
    the search has to keep. Checking it against the raw message would let a translation
    drop the one term that identifies the answer and call it clean.

    Each field stands or falls alone. A translation that fails here is simply not used —
    the resolution it arrived with is unaffected, and retrieval searches the question as
    written, which is what it did before any of this existed.

    Remembered under the raw message too when the two differ, so a retrieval that never
    saw the resolution still finds it: the graph is handed whatever query the agent wrote,
    and that is as often the user's words as the resolver's.
    """
    if not translating:
        return ""
    candidate = str(result.get("search_text") or "").strip()
    if not candidate:
        return ""
    if not remember_translation(resolved, candidate):
        logger.info("resolver's translation failed verification; searching as written")
        return ""
    if raw.strip() and raw.strip() != resolved:
        remember_translation(raw, candidate)
    return candidate


# Defined once, at import, rather than inside `_default_resolve_invoke` on every call: a class built
# per call is a pydantic model built per call, and its JSON schema regenerated with
# it — measured at 4.5% and 10.4% of the serving process's CPU (RAG_FIX_PLAN item 47).
class ResolvedQuery(StructuredOutput):
    question: str = Field(
        description="The user's latest message rewritten so it stands on its own, in their language"
    )
    constraints: _List[str] = Field(
        default_factory=list,
        description="Conditions carried over from earlier turns that still bind the answer",
    )
    intent: _Literal["standalone", "followup", "correction", "new_topic"] = Field(
        default="followup",
        description="How the latest message relates to the conversation before it",
    )
    # Always declared, never conditional. Providers enforcing OpenAI-style STRICT
    # structured output (Groq among them) require every declared property to be
    # present, and a model told to "leave the unused field empty" tends to omit it
    # instead — the trap `rewrite_query_once` documents. So the field is always in the
    # schema and the PROMPT decides whether to fill it; an empty string is the normal
    # answer on a turn that needs no translation, and the caller only reads it when it
    # asked for one.
    search_text: str = Field(
        default="",
        description=(
            "The question translated for SEARCHING only, when asked for; otherwise an "
            "empty string"
        ),
    )


def _default_resolve_invoke(  # pragma: no cover - needs a model
    question, history, config, hitl_prompt, hitl_options,
    *, resolving: bool = True, translating: bool = False,
):
    """One small structured call on FAST_MODEL, carrying whichever jobs this turn needs."""
    import os

    from langchain.chat_models import init_chat_model

    from backend.assets.vision import invoke_structured
    from backend.llm import sampling
    from backend.profiles import get_profile
    from backend.prompts import resolve as resolve_prompt
    from backend.composition import default_services

    profile = get_profile()


    prompt = resolve_prompt(
        getattr(config, "query_resolution_prompt", "") or "",
        "chat/resolve_question.j2",
        question=question,
        history=history,
        persona=profile.identity.persona,
        hitl_prompt=hitl_prompt or "",
        hitl_options=list(hitl_options or []),
        resolving=resolving,
        translating=translating,
        search_language=(profile.retrieval.query_translation_language or "en"),
    )
    model = init_chat_model(
        model=os.getenv("FAST_MODEL"),
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        **default_services().provider_http.model_kwargs(),
        **sampling("resolve"),
    )

    # A 429 is retried in the client, under the turn's rate-limit policy
    # (backend/llm_http.py). Retrying here as well multiplied the attempts (item 37).
    # One that still fails makes the resolver abstain: safe, but it spends the
    # clarification round trip this call exists to avoid.
    result = invoke_structured(model, ResolvedQuery, [{"role": "user", "content": prompt}])
    return result if isinstance(result, dict) else result.model_dump()


__all__ = [
    "CORRECTION",
    "FOLLOWUP",
    "INTENTS",
    "NEW_TOPIC",
    "STANDALONE",
    "ResolvedQuestion",
    "conversation_text",
    "message_role_and_text",
    "needs_resolution",
    "resolve_question",
    "unresolved",
]
