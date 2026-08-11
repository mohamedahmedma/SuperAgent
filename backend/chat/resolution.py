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

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from backend.text_normalization import normalize_query

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
    if not wanted and not hitl_prompt:
        return unresolved(question, reason)

    rendered = conversation_text(
        history,
        limit=int(getattr(config, "query_resolution_history_messages", 6) or 6),
    )
    if not rendered and not hitl_prompt:
        return unresolved(question, "no usable conversation text to resolve against")

    call = invoke or _default_resolve_invoke
    try:
        result = call(question, rendered, config, hitl_prompt, list(hitl_options or []))
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
    # an empty question would search for nothing and deny the turn.
    if not resolved_text:
        return unresolved(question, "resolver returned an empty question")

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
        reason=reason or "resolved against a pending clarification",
    )


def _default_resolve_invoke(question, history, config, hitl_prompt, hitl_options):  # pragma: no cover - needs a model
    """One small structured call on FAST_MODEL."""
    import os

    from langchain.chat_models import init_chat_model
    from pydantic import BaseModel, Field
    from typing import List as _List, Literal as _Literal

    from backend.assets.vision import call_with_rate_limit_retry, invoke_structured
    from backend.llm import sampling
    from backend.profiles import get_profile
    from backend.prompts import resolve as resolve_prompt

    profile = get_profile()

    class ResolvedQuery(BaseModel):
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

    prompt = resolve_prompt(
        getattr(config, "query_resolution_prompt", "") or "",
        "chat/resolve_question.j2",
        question=question,
        history=history,
        persona=profile.identity.persona,
        hitl_prompt=hitl_prompt or "",
        hitl_options=list(hitl_options or []),
    )
    model = init_chat_model(
        model=os.getenv("FAST_MODEL"),
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        **sampling("resolve"),
    )

    # Same quota as every other call in the turn, so the same treatment. A 429 here
    # would make the resolver abstain, which is safe but spends the clarification
    # round-trip this call exists to avoid.
    class _Retry:
        vision_retry_attempts = int(getattr(config, "model_retry_attempts", 2))
        vision_retry_base_seconds = float(getattr(config, "model_retry_base_seconds", 2.0))
        vision_retry_max_seconds = float(getattr(config, "model_retry_max_seconds", 6.0))

    result = call_with_rate_limit_retry(
        lambda: invoke_structured(model, ResolvedQuery, [{"role": "user", "content": prompt}]),
        config=_Retry(),
        description="query resolution",
    )
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
