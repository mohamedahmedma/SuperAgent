"""What to do with a turn, given what we know about it.

Pure functions over `RequestSignals` and profile config — no model calls, no I/O, no
graph state. The same rule as `backend/rag/policy.py`: when a new decision is needed
it becomes a function here reading the existing signals, rather than a component that
derives its own view and drifts.

## The retrieval question

There are three ways to act on "this question probably isn't in the corpus", and two
of them are the same mistake:

  * **Unbind the knowledge tool.** If the classifier is wrong, the agent cannot search
    and answers from nothing.
  * **Bind it but short-circuit it internally.** If the classifier is wrong, the agent
    searches and gets a canned refusal. Same wrong answer, having also paid for the
    tool schema and a round trip.

Both make a cheap, fallible signal into an unappealable verdict. So neither is used.

What this module does instead is scale the action to the confidence behind it:

  * **HIGH-certainty out-of-domain** — a model read the question and said so. End the
    turn before the agent runs: static reply, no tool schemas, no search.
  * **Anything less** — the knowledge tool stays bound AND fully functional. Weak
    signals become a retrieval *hint* (`candidate_sections`), which reorders what is
    searched first and can never remove a document from reach.

There is deliberately no state in which the agent can call the knowledge tool and be
silently denied an answer. The cheap rung's failures cost latency; only the expensive
rung's failures can cost a refusal, and that is the one that read the question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.chat.language import ARABIC, ENGLISH
from backend.chat.signals import RequestSignals, Scope
from backend.rag.evidence import Certainty

# Only a detector that read the question may end a turn. This floor is the whole
# safety argument of this module; lowering it hands refusals to a dot product.
SHORT_CIRCUIT_MIN_CERTAINTY = Certainty.HIGH


@dataclass
class TurnPlan:
    """The decision for one turn."""

    # When set, the turn is over: reply with this text and run no model.
    static_reply: Optional[str] = None
    # Tool names to bind for this turn. None means "bind everything the profile allows"
    # — the safe default, and what any uncertainty produces.
    exposed_tools: Optional[List[str]] = None
    # Sections to search first. A hint; never a filter.
    retrieval_sections: List[str] = field(default_factory=list)
    # Language to answer in, as a detector code. Carried on the plan rather than read
    # from the signals downstream because the agent is built from the plan, and the
    # prompt has to be able to say it.
    language: str = ENGLISH
    # Whether this turn is worth a post-turn profile extraction.
    capture_user_info: bool = False
    reasons: List[str] = field(default_factory=list)

    @property
    def short_circuit(self) -> bool:
        return self.static_reply is not None

    def as_trace(self) -> Dict[str, Any]:
        return {
            "turn_short_circuit": self.short_circuit,
            "turn_exposed_tools": list(self.exposed_tools) if self.exposed_tools is not None else None,
            "turn_retrieval_sections": list(self.retrieval_sections),
            "turn_language": self.language,
            "turn_capture_user_info": self.capture_user_info,
            "turn_reason": "; ".join(self.reasons) or "n/a",
        }


def localized(copy, language: str) -> str:
    """Pick the copy for a language, falling back rather than failing.

    A missing translation must degrade to the other language, never to an empty
    message — a blank reply is worse than one in the wrong language.
    """
    if copy is None:
        return ""
    if isinstance(copy, str):
        return copy
    preferred = getattr(copy, language, None) or getattr(copy, "get", lambda _k, _d=None: None)(language)
    if preferred:
        return preferred
    for fallback in (ENGLISH, ARABIC):
        alternative = getattr(copy, fallback, None) or getattr(copy, "get", lambda _k, _d=None: None)(fallback)
        if alternative:
            return alternative
    return ""


def _tools_for(config, *, drop: tuple = ()) -> List[str]:
    allowed = list(getattr(config, "tools", None) or [])
    return [name for name in allowed if name not in drop]


def resolve_turn(
    signals: RequestSignals,
    *,
    agent_config,
    copy_config,
    rag_config=None,
) -> TurnPlan:
    """Decide how this turn should run."""
    plan = TurnPlan()
    # Independent of scope: someone can disclose their phone number in the middle of
    # an off-topic message, and that is still worth keeping.
    plan.capture_user_info = bool(signals.personal_data)
    plan.language = signals.language

    if signals.is_social:
        return _plan_social(signals, plan, agent_config, copy_config)

    if signals.scope is Scope.OUT_OF_DOMAIN and signals.scope_certainty >= SHORT_CIRCUIT_MIN_CERTAINTY:
        return _plan_out_of_domain(signals, plan, copy_config)

    if signals.scope is Scope.OUT_OF_DOMAIN:
        # Believed, but not by anything that read the question. The tool stays bound
        # and working; only its ordering is nudged.
        plan.reasons.append(
            f"out-of-domain at {signals.scope_certainty.name.lower()} certainty — "
            "knowledge tool stays available"
        )

    plan.retrieval_sections = list(signals.candidate_sections)
    if plan.retrieval_sections:
        plan.reasons.append(f"search {len(plan.retrieval_sections)} section(s) first")
    if not plan.reasons:
        plan.reasons.append("no signal changed the default plan")
    return plan


def _plan_social(signals, plan: TurnPlan, agent_config, copy_config) -> TurnPlan:
    """A pleasantry. Cheap either way, but not a refusal.

    Default is still a model call — with every tool unbound, so it costs the lean
    prompt and nothing else. That keeps replies varied; an identical canned string for
    every greeting reads as a script, and greetings are the first thing a user sees.
    Deployments that would rather pay nothing can switch to static.
    """
    plan.exposed_tools = []
    if getattr(agent_config, "social_reply_mode", "model") == "static":
        plan.static_reply = localized(getattr(copy_config, "social", None), signals.language)
        if plan.static_reply:
            plan.reasons.append("social phrase, static reply")
            return plan
        plan.reasons.append("social phrase, static reply configured but no copy — using the model")
    else:
        plan.reasons.append("social phrase, model reply with no tools bound")
    return plan


def _plan_out_of_domain(signals, plan: TurnPlan, copy_config) -> TurnPlan:
    plan.static_reply = localized(getattr(copy_config, "out_of_domain", None), signals.language)
    plan.exposed_tools = []
    if not plan.static_reply:
        # Refusing with an empty string is worse than answering. Fall through to the
        # agent and let it explain itself.
        plan.static_reply = None
        plan.exposed_tools = None
        plan.reasons.append("out of domain but no copy configured — falling through to the agent")
        return plan
    plan.reasons.append("out of domain, confirmed by a classifier that read the question")
    return plan
