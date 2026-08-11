"""The turn planner: what a turn costs is decided here, before anything expensive.

Every turn currently pays the same price — the agent's system prompt plus every tool
schema, then whatever the agent decides to do. That price is right for a question the
corpus can answer and wrong for "thanks", and wrong again for a question about
football.

This module runs first and produces a `TurnPlan`. It is the "orchestrator node" in
graph terms, but it is deliberately not a node: the agent is LangChain's prebuilt
`create_agent`, and wrapping it in a hand-built StateGraph to gain one pre-step would
mean re-implementing tool execution, streaming and recursion limits to change
something that can be decided before the agent is constructed at all.

What it can do, in order of how much it saves:

  * **End the turn.** A confirmed out-of-domain question gets profile copy in its own
    language, and the agent is never built. No system prompt, no schemas, no search.
  * **Narrow the tools.** Fewer schemas on the wire, on a cost that is paid every turn
    and grows with every capability added.
  * **Hint retrieval.** Sections to search first — ordering, never filtering.

All of it degrades to today's behaviour. The ladder's rungs are off by default, every
detector failure abstains, and an empty plan means "build the agent exactly as before".
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from backend.chat.request_context import ChatRequestContext
from backend.chat.resolution import ResolvedQuestion, resolve_question, unresolved
from backend.chat.signals import RequestSignals, SignalContext, build_ladder
from backend.chat.turn_policy import TurnPlan, resolve_turn
from backend.profiles import get_profile

logger = logging.getLogger(__name__)


def plan_turn(
    question: str,
    history: Optional[Sequence[Any]] = None,
    ctx: Optional[ChatRequestContext] = None,
    *,
    envelope_invoke=None,
    resolve_invoke=None,
    resolution: Optional[ResolvedQuestion] = None,
) -> tuple:
    """Decide how this turn should run. Returns `(plan, signals)`.

    Both are returned because they answer different questions: the plan is what to do,
    the signals are why — and the trace wants both.

    Resolution runs FIRST, before any detector, because every rung below it measures
    words and a follow-up's own words do not name its subject. `resolution` may be
    passed in by a caller that already resolved the message — the HITL resume path
    does, since it has to know whether the reply is a correction before it can decide
    whether to resume at all, and resolving twice would pay for the same call twice.

    Never raises. A planner that fails must cost the savings it would have produced,
    never the turn, so anything unexpected returns the empty plan, which is exactly the
    behaviour that existed before this module.
    """
    profile = get_profile()
    try:
        config = _LadderConfig(profile.agent, profile.rag)
        messages = list(history or [])
        resolved = resolution or resolve_question(
            question, messages, config, invoke=resolve_invoke
        )
        signals = build_ladder(config, envelope_invoke=envelope_invoke).run(
            SignalContext(
                question=question,
                history=messages,
                config=config,
                resolved_question=resolved.question if resolved.resolved else "",
                carried_constraints=list(resolved.constraints),
                followup_intent=resolved.intent,
            )
        )
        signals.reasons.insert(0, f"resolution: {resolved.reason}")
        plan = resolve_turn(
            signals,
            agent_config=profile.agent,
            copy_config=profile.user_copy,
            rag_config=profile.rag,
        )
    except Exception:
        logger.warning("turn planning failed; running the turn unchanged", exc_info=True)
        return TurnPlan(reasons=["planner error — defaults applied"]), RequestSignals(question=question)

    _hand_to_graph(ctx, plan)
    _emit(ctx, signals, plan)
    return plan, signals


def resolve_turn_question(
    question: str,
    history: Optional[Sequence[Any]] = None,
    *,
    invoke=None,
    hitl_prompt: str = "",
    hitl_options: Sequence[str] = (),
) -> ResolvedQuestion:
    """Resolve a message against the conversation, using the active profile's settings.

    Exposed separately from `plan_turn` for the one caller that has to act on the
    result before planning anything: a resumed clarification needs to know whether the
    reply corrects the question or answers it, and those go down different paths.
    Never raises, for the same reason `plan_turn` does not.
    """
    try:
        profile = get_profile()
        return resolve_question(
            question,
            list(history or []),
            _LadderConfig(profile.agent, profile.rag),
            invoke=invoke,
            hitl_prompt=hitl_prompt,
            hitl_options=hitl_options,
        )
    except Exception:  # pragma: no cover - resolution must never break a turn
        logger.warning("query resolution failed; using the message as written", exc_info=True)
        return unresolved(question, "resolver error")


def _hand_to_graph(ctx: Optional[ChatRequestContext], plan: TurnPlan) -> None:
    """Put the plan's retrieval hints where the RAG graph reads them.

    Guarded for the same reason `_emit` is, and it is the same class of thing: both are
    hints the turn is better off without than dead for. This function's failure mode
    would otherwise be the one this module promises cannot happen — a planner fault
    costing the turn rather than the saving.
    """
    if ctx is None:
        return
    try:
        ctx.note_turn_plan(
            plan.retrieval_sections,
            plan.scope_options,
            carried_constraints=plan.carried_constraints,
            is_followup=plan.is_followup,
        )
    except Exception:  # pragma: no cover - a hint must never break a turn
        logger.debug("could not hand the turn plan to the graph", exc_info=True)


class _LadderConfig:
    """A read-only view over the two profile sections the detectors need.

    The social phrases and the envelope switch live under `agent`; the corpus-gate
    thresholds live under `rag`, because they belong to the same reference index the
    retriever uses. Detectors should not have to know which section a setting came
    from, so they get one object that answers for both.
    """

    __slots__ = ("_agent", "_rag")

    def __init__(self, agent, rag):
        self._agent = agent
        self._rag = rag

    def __getattr__(self, name):
        if hasattr(self._agent, name):
            return getattr(self._agent, name)
        return getattr(self._rag, name)


def _emit(ctx: Optional[ChatRequestContext], signals: RequestSignals, plan: TurnPlan) -> None:
    """Surface the decision as a progress step, and only when it changed something.

    A plan that changed nothing is the common case and saying so on every turn would
    bury the steps that matter.
    """
    if ctx is None:
        return
    try:
        if plan.short_circuit:
            ctx.emit_rag_step("🚪", "Answered without searching", "; ".join(plan.reasons)[:90])
        elif plan.exposed_tools is not None:
            ctx.emit_rag_step(
                "🎯", f"Narrowed to {len(plan.exposed_tools)} tool(s)", "; ".join(plan.reasons)[:90]
            )
        elif plan.retrieval_sections:
            ctx.emit_rag_step(
                "🧭", f"Prioritising {len(plan.retrieval_sections)} knowledge section(s)",
                "; ".join(signals.reasons)[:90],
            )
    except Exception:  # pragma: no cover - progress reporting must never break a turn
        logger.debug("could not emit turn-plan step", exc_info=True)


def plan_trace(plan: TurnPlan, signals: Optional[RequestSignals] = None) -> dict:
    """Trace fields for a planned turn, for the client and for LangSmith."""
    trace = dict(plan.as_trace())
    if signals is not None:
        trace.update(signals.as_trace())
    return trace
