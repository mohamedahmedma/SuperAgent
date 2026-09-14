"""What the model is shown for a turn.

The agent's context — the persistent note, recent history with rendered records stripped,
the planner's reading of this message, and the message itself — and the direct-answer
prompt used when a clarification is resumed without the agent.

Moved out of `service.py` with its behaviour unchanged.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.chat.answer_blocks import strip_answer_blocks
from backend.chat.resolution import conversation_text
from backend.profiles import get_profile
from backend.prompts import resolve as resolve_prompt


def _format_retrieved_chunks(docs: list[dict]) -> str:
    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(formatted)


def build_resume_answer_messages(
    pending_hitl: dict,
    user_answer: str,
    docs: list[dict],
    *,
    resolved_question: str = "",
    constraints: list[str] | None = None,
    history: list | None = None,
) -> list:
    """The direct-answer call taken when a clarification is resumed.

    This path bypasses the agent, so everything the agent would have had must be handed
    over explicitly — and three things were not. The conversation, without which the
    model cannot honour a condition set before the clarification. The resolved question,
    so it answers what was asked rather than reassembling it from three fragments. And
    the conditions themselves, because retrieval returning the right chunks does not
    stop an answer from covering every year group in them.
    """
    original_question = pending_hitl.get("original_question") or ""
    prompt = pending_hitl.get("prompt") or ""
    context = _format_retrieved_chunks(docs)
    conditions = [str(item) for item in (constraints or []) if str(item).strip()]
    system = SystemMessage(
        content=resolve_prompt(get_profile().agent.resume_answer_prompt, "agent/resume_answer.j2")
    )

    sections = []
    dialogue = conversation_text(history or [], limit=get_profile().agent.query_resolution_history_messages)
    if dialogue:
        sections.append(f"The conversation so far:\n{dialogue}")
    sections.append(f"Original question:\n{original_question}")
    sections.append(f"HITL follow-up question:\n{prompt}")
    sections.append(f"User's answer:\n{user_answer}")
    if resolved_question:
        sections.append(
            "Read in context, the question to answer is:\n"
            f"{resolved_question}"
        )
    if conditions:
        # Same three-way rule as tools/knowledge_result.j2, and for the same reason: a
        # condition the material does not vary by must not be able to suppress an answer
        # that is sitting in the chunks below.
        sections.append(
            "Conditions the user set earlier and has not withdrawn: "
            + "; ".join(conditions)
            + "\nWhere the material below distinguishes by them, answer for their case. "
            "Where it states one rule for everyone, give that rule in full and say it "
            "applies regardless of "
            + " or ".join(conditions)
            + ". A general rule IS the answer to a specific question — never refuse "
            "because the conditions are not named in the material."
        )
    sections.append(f"Retrieved chunks:\n{context}")
    # No trailing "answer from the chunks and cite them" instruction: the system
    # message above (agent.resume_answer_prompt) already says exactly that, and
    # repeating it here paid for the same rule twice per resume.
    return [system, HumanMessage(content="\n\n".join(sections))]


def _turn_context_message(turn_plan) -> SystemMessage | None:
    """What the planner worked out about this message, or nothing to say.

    Rendered next to the user's message rather than into the system prompt, which is
    ordered most-static-first for prompt caching — a per-turn line placed there would
    invalidate the cached prefix on every turn. A message that stands on its own and
    carries no inherited conditions renders empty and produces no message at all.
    """
    if turn_plan is None:
        return None
    resolved = (getattr(turn_plan, "resolved_question", "") or "").strip()
    constraints = [str(item) for item in (getattr(turn_plan, "carried_constraints", None) or [])]
    child_hint = (getattr(turn_plan, "child_hint", "") or "").strip()
    child_year = (getattr(turn_plan, "child_year", "") or "").strip()
    # `child_options` is deliberately absent. A turn that could not settle which child is
    # meant no longer reaches the agent at all — the planner ends it with the question
    # and the candidates as selectable options — so there is nothing to render and no
    # reason to pay for a render. See the note where that block used to be in
    # agent/turn_context.j2.
    #
    # This condition is the feature's single point of failure: a plan carrying a child
    # and nothing else renders nothing at all unless the child is named here too.
    if not resolved and not constraints and not child_hint:
        return None
    rendered = resolve_prompt(
        "",
        "agent/turn_context.j2",
        resolved_question=resolved,
        constraints=constraints,
        child_hint=child_hint,
        child_year=child_year,
    )
    return SystemMessage(content=rendered) if rendered else None


def build_context_messages(
    messages: list,
    persistent_note: str,
    user_text: str,
    turn_plan=None,
) -> list:
    short_term = messages[-get_profile().agent.context_window_messages:] if len(messages) > get_profile().agent.context_window_messages else messages
    context_messages: list = []
    if persistent_note:
        context_messages.append(
            SystemMessage(
                content=(
                    "[Persistent conversation note (your working memory)]\n"
                    f"{persistent_note}\n"
                    "Refer to the note above to keep the conversation coherent, and avoid re-answering questions that have already been resolved."
                )
            )
        )
    # Same reason `conversation_text` strips them: the agent is deciding what this turn
    # needs, and a previous turn's table is not evidence about this one. The reader keeps
    # the block; the model gets the sentence.
    context_messages.extend(
        AIMessage(content=strip_answer_blocks(message.content))
        if isinstance(message, AIMessage) and isinstance(message.content, str)
        else message
        for message in short_term
    )
    # After the history and before the message it describes, so the model reads the
    # conversation, then what that conversation makes this message mean, then the
    # message itself.
    turn_context = _turn_context_message(turn_plan)
    if turn_context is not None:
        context_messages.append(turn_context)
    context_messages.append(HumanMessage(content=user_text))
    return context_messages
