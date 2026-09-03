import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from backend.assets.delivery import ClientCapabilities
from backend.chat.assets_bridge import (
    asset_ids_for_answer,
    asset_ids_for_turn,
    attach_assets_to_trace,
    build_asset_references,
    effective_capabilities,
    load_asset_state,
    save_asset_state,
    trace_for_storage,
)
from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_context import load_child_state, save_child_state
from backend.chat.finalize import Finalizer, finalize_text, message_text
from backend.chat.orchestrator import plan_turn, resolve_turn_question
from backend.chat.request_context import ChatRequestContext
from backend.chat.resolution import ResolvedQuestion, conversation_text
from backend.chat.runtime import create_agent_for_request, fast_model, model
from backend.chat.storage import storage
from backend.profiles import get_profile
from backend.prompts import resolve as resolve_prompt
from backend.schemas.chat import PendingHitlState, normalize_rag_trace
from backend.tools import GROUNDED_TOOLS

logger = logging.getLogger(__name__)

_PROFILE = get_profile()
_COPY = _PROFILE.user_copy

CONTEXT_WINDOW_MESSAGES = _PROFILE.agent.context_window_messages
PENDING_HITL_KEY = "pending_hitl"
HITL_STATUSES = {"needs_clarification", "needs_scope_selection"}
HITL_ROUTES = {"clarify", "scope_select"}


def _is_hitl_trace(rag_trace: dict | None) -> bool:
    if not isinstance(rag_trace, dict):
        return False
    status = rag_trace.get("retrieval_status")
    route = rag_trace.get("route")
    return status in HITL_STATUSES or route in HITL_ROUTES


def _hitl_route_from_trace(rag_trace: dict) -> str:
    status = rag_trace.get("retrieval_status")
    route = rag_trace.get("route")
    if status == "needs_scope_selection" or route == "scope_select":
        return "scope_select"
    return "clarify"


def _hitl_prompt_from_trace(rag_trace: dict) -> str:
    prompt = (rag_trace.get("hitl_prompt") or "").strip()
    if prompt:
        return prompt
    route = _hitl_route_from_trace(rag_trace)
    if route == "scope_select":
        return _COPY.hitl_scope_agent
    return _COPY.hitl_clarify_agent


def _hitl_options_from_trace(rag_trace: dict) -> list[str]:
    options = rag_trace.get("hitl_options") or []
    if not isinstance(options, list):
        return []
    return [str(option).strip() for option in options if str(option).strip()]


def _format_hitl_message(prompt: str, options: list[str] | None = None) -> str:
    clean_prompt = prompt.strip()
    clean_options = [item for item in (options or []) if item]
    if not clean_options:
        return clean_prompt
    option_lines = "\n".join(f"- {item}" for item in clean_options)
    return f"{clean_prompt}\n\nAvailable options:\n{option_lines}"


def _existing_hitl_answers(pending_hitl: dict | None) -> list[str]:
    if not isinstance(pending_hitl, dict):
        return []
    answers = pending_hitl.get("answers") or []
    if not isinstance(answers, list):
        return []
    return [str(answer).strip() for answer in answers if str(answer).strip()]


def _build_pending_hitl(
    rag_trace: dict,
    original_question: str,
    previous_answers: list[str] | None = None,
    resume_state: dict | None = None,
) -> dict:
    prompt = _hitl_prompt_from_trace(rag_trace)
    options = _hitl_options_from_trace(rag_trace)
    route = _hitl_route_from_trace(rag_trace)
    return PendingHitlState(
        id=uuid4().hex,
        original_question=original_question,
        prompt=prompt,
        options=options,
        route=route,
        retrieval_status=(
            "needs_scope_selection" if route == "scope_select" else "needs_clarification"
        ),
        answers=previous_answers or [],
        resume_state=resume_state,
        created_at=datetime.now(timezone.utc).isoformat(),
    ).model_dump()


def _build_hitl_event(pending_hitl: dict) -> dict:
    return {
        "id": pending_hitl["id"],
        "prompt": pending_hitl["prompt"],
        "options": pending_hitl["options"],
        "route": pending_hitl["route"],
        "retrieval_status": pending_hitl["retrieval_status"],
        "original_question": pending_hitl["original_question"],
    }


def _build_hitl_resume_query(pending_hitl: dict, user_text: str) -> str:
    original_question = pending_hitl.get("original_question") or ""
    prompt = pending_hitl.get("prompt") or ""
    previous_answers = _existing_hitl_answers(pending_hitl)

    lines = [
        "This is a continuation request after HITL clarification in the previous RAG flow.",
        "Do not treat the user's follow-up as a standalone new question; return to the original question and continue answering it.",
        f"Original question: {original_question}",
    ]
    if prompt:
        lines.append(f"HITL question: {prompt}")
    if previous_answers:
        lines.append("The user has previously provided:")
        lines.extend(f"- {answer}" for answer in previous_answers)
    lines.extend([
        f"User's input this round: {user_text}",
        "Please form a complete query based on the above and continue with the original Agent/RAG flow.",
    ])
    return "\n".join(lines)


@dataclass
class _TurnEntry:
    """How this message relates to a clarification the assistant is still waiting on.

    Computed once, before anything expensive, because the answer changes which of two
    paths the turn takes — and both entry points need the same answer.
    """

    pending_hitl: dict | None = None
    invalid_pending_hitl: bool = False
    is_hitl_resume: bool = False
    resume_state: dict | None = None
    resolution: ResolvedQuestion | None = None
    superseded: bool = False
    effective_user_text: str = ""
    hitl_answers: list = field(default_factory=list)
    original_question: str = ""


def _enter_turn(user_text: str, messages: list, metadata: dict) -> _TurnEntry:
    """Decide whether this message answers the pending clarification or replaces it.

    The resume path assumes the reply either picks one of the offered options or fills
    a named slot. A reply that does neither — "no, I meant the fees", or a change of
    subject entirely — is not something to fold into the old query, because folding is
    concatenation and concatenation retrieves both readings. Those replies abandon the
    pending clarification and start a fresh turn from the resolved question instead.

    The resolution computed here is handed to `plan_turn`, so a turn that takes the
    fresh path pays for exactly one resolver call rather than two.
    """
    entry = _TurnEntry(effective_user_text=user_text, original_question=user_text)

    stored = metadata.get(PENDING_HITL_KEY)
    entry.pending_hitl = _current_pending_hitl(stored)
    entry.invalid_pending_hitl = stored is not None and entry.pending_hitl is None
    if not isinstance(entry.pending_hitl, dict):
        return entry

    entry.resolution = resolve_turn_question(
        user_text,
        messages,
        hitl_prompt=entry.pending_hitl.get("prompt") or "",
        hitl_options=entry.pending_hitl.get("options") or [],
    )
    entry.superseded = entry.resolution.supersedes_pending_question
    entry.is_hitl_resume = not entry.superseded
    entry.resume_state = _pending_resume_state(entry.pending_hitl)
    entry.hitl_answers = [*_existing_hitl_answers(entry.pending_hitl), user_text]

    if entry.superseded:
        # A fresh turn: the pending state is dropped rather than resumed, and the
        # question this turn is about is the one the user has just corrected it to.
        entry.original_question = entry.resolution.question or user_text
        return entry

    entry.effective_user_text = _build_hitl_resume_query(entry.pending_hitl, user_text)
    entry.original_question = entry.pending_hitl.get("original_question") or user_text
    return entry


def _current_pending_hitl(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    try:
        return PendingHitlState.model_validate(value).model_dump()
    except ValueError:
        return None


def _pending_resume_state(pending_hitl: dict | None) -> dict | None:
    if not isinstance(pending_hitl, dict):
        return None
    resume_state = pending_hitl.get("resume_state")
    return dict(resume_state) if isinstance(resume_state, dict) else None


def _extract_ai_content(msg) -> str:
    """The text of a model message, as a user may see it.

    Reading the content and cleaning it are one step on purpose. This function is the
    only way a direct `model.invoke`/`model.astream` result becomes a string in this
    module, so putting the transcript strip anywhere else would leave the paths that
    bypass the agent — the HITL resume answer, the persistent note — able to put a raw
    Harmony envelope in front of a user. See `backend/chat/finalize.py`.
    """
    return finalize_text(message_text(msg))


def _format_retrieved_chunks(docs: list[dict]) -> str:
    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(formatted)


def _build_resume_answer_messages(
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
        content=resolve_prompt(_PROFILE.agent.resume_answer_prompt, "agent/resume_answer.j2")
    )

    sections = []
    dialogue = conversation_text(history or [], limit=_PROFILE.agent.query_resolution_history_messages)
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


def _terminal_reply(status: str, language: str) -> str:
    """The profile's own wording for an outcome the model does not need to compose."""
    from backend.chat.turn_policy import localized

    copy = _COPY.retrieval_error if status == "retrieval_error" else _COPY.no_knowledge
    return localized(copy, language)


def _message_data_for_save(messages: list, rag_trace: dict | None) -> list:
    """Per-message extras for the save: the finished turn's trace, on its answer.

    The stored copy keeps its assets as ids rather than renditions — the renditions on
    the wire were built for the client that asked, and rebuilding them on load is a
    keyed lookup. Every save path goes through here so the two cannot drift.
    """
    return [None] * (len(messages) - 1) + [{"rag_trace": trace_for_storage(rag_trace)}]


async def _stream_static_reply(
    turn_plan,
    turn_signals,
    user_text: str,
    user_id: str,
    session_id: str,
    messages: list,
    metadata: dict,
    persistent_note: str,
    asset_state,
    is_first_message: bool,
):
    """Emit a planned reply that no model composed, and persist the turn.

    Streamed through the same event shapes as an agent reply — `session_title`,
    `content`, `trace`, `[DONE]` — because a client must not need to know which path
    produced its answer. The saving is that no agent was built: no system prompt, no
    tool schemas, no search.
    """
    if is_first_message:
        title = generate_session_title(user_text)
        yield f"data: {json.dumps({'type': 'session_title', 'title': title, 'session_id': session_id})}\n\n"

    reply = turn_plan.static_reply or ""
    yield f"data: {json.dumps({'type': 'content', 'content': reply})}\n\n"

    rag_trace = normalize_rag_trace({**turn_plan.as_trace(), **turn_signals.as_trace()})
    yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    save_meta = dict(metadata)
    save_asset_state(save_meta, asset_state)
    # A turn the corpus never saw contributes nothing worth summarising, so the
    # persistent note is deliberately left alone — updating it would spend a model
    # call on the one path whose whole point is not making one.
    save_meta[PENDING_HITL_KEY] = None
    if is_first_message:
        save_meta.setdefault("title", generate_session_title(user_text))

    messages.append(AIMessage(content=reply))
    extra_message_data = _message_data_for_save(messages, rag_trace)
    storage.save(user_id, session_id, messages, metadata=save_meta,
                 extra_message_data=extra_message_data)

    yield "data: [DONE]\n\n"


def _no_knowledge_response() -> str:
    return _COPY.no_knowledge


def _retrieval_error_response() -> str:
    return _COPY.retrieval_error


def _turn_is_asking_a_question(ctx) -> bool:
    """Whether retrieval has decided this turn ends in a question, not an answer.

    Read from the live trace rather than passed in, because the decision is made by the
    knowledge tool part-way through the turn — after the stream has already started.
    Named because two places have to consult it and a copy of the lookup in each is how
    they come apart: the second one did, and a clarification prompt was shown twice.
    """
    stored = ctx.peek_rag_trace()
    return _is_hitl_trace(
        normalize_rag_trace(stored.get("rag_trace") if stored else None)
    )


def _evidence_texts(rag_trace: dict | None) -> list:
    """The text of every chunk this turn retrieved, as the grounding check reads it."""
    chunks = (rag_trace or {}).get("retrieved_chunks") or []
    return [
        str(chunk.get("text") or "")
        for chunk in chunks
        if isinstance(chunk, dict) and chunk.get("text")
    ]


def _grounding_expected(turn_plan) -> bool:
    """Whether this turn is one whose figures have to come from somewhere.

    A turn that bound a grounded tool is answerable from evidence, so a figure in its
    answer is a claim about that evidence. A social reply or an out-of-domain refusal
    bound nothing to cite, never claimed to, and must not be checked as though it had —
    that is how a check earns the right to be enforced rather than merely observed.
    """
    exposed = getattr(turn_plan, "exposed_tools", None)
    bound = _PROFILE.agent.tools if exposed is None else exposed
    return bool(set(bound) & GROUNDED_TOOLS)


def _nothing_usable_reply(finalizer: Finalizer, turn_plan) -> str:
    """Copy for a turn whose model output contained no answer at all.

    Measured on the live provider: on one turn in three the model emitted a transcript —
    reasoning plus a fabricated tool call — and never opened a final channel. The
    finalizer correctly withholds all of it, and the turn then had nothing to say, so
    the parent got an empty bubble. Suppressing a non-answer is right; showing nothing
    in its place is not.

    `retrieval_error` is the honest copy for it. The knowledge base was fine — the
    model's reply was unusable — and what that copy tells a parent is exactly what this
    situation warrants: a brief technical problem, try again in a moment. Inventing a
    cheerier message would be claiming to know something about a turn that produced no
    information at all.

    Returns "" when the turn legitimately had nothing to say — a social reply that was
    short-circuited, or a plan that answered without the model — so this never
    manufactures an error out of a quiet success.
    """
    if turn_plan is not None and getattr(turn_plan, "short_circuit", False):
        return ""
    # Only when the finalizer actually withheld something. A model that returned an
    # empty string on its own is a different fault and not one this copy describes.
    trace = finalizer.as_trace()
    withheld = (
        trace.get("finalize_dropped_tool_call_messages")
        or trace.get("finalize_harmony_messages")
        or trace.get("finalize_dropped_chars")
    )
    if not withheld:
        return ""
    logger.warning(
        "the model produced no answer channel this turn; serving the retry copy"
    )
    return _COPY.retrieval_error


def _enforce_grounding(finalizer: Finalizer, rag_trace: dict | None, turn_plan) -> str:
    """Verify the assembled answer; return replacement copy when it must not stand.

    Returns "" when the answer is fine, when the check is off, or when this turn is not
    one the check applies to. The verdict is recorded on the finalizer either way, so
    `observe` mode produces the same trace as `enforce` and a deployment can measure the
    check against its own corpus before letting it act.
    """
    mode = _PROFILE.agent.answer_grounding_mode
    if mode == "off" or not (finalizer.answer or "").strip():
        return ""
    if not _grounding_expected(turn_plan):
        return ""
    report = finalizer.verify(
        _evidence_texts(rag_trace),
        floor=_PROFILE.agent.answer_grounding_number_floor,
    )
    if report.ok or mode != "enforce":
        return ""
    return _COPY.unverified_answer


def _resume_rag_from_hitl_sync(
    pending_hitl: dict,
    user_answer: str,
    ctx: ChatRequestContext,
    resolved: ResolvedQuestion | None = None,
) -> dict:
    from backend.rag.pipeline import resume_rag_from_hitl

    resume_state = _pending_resume_state(pending_hitl)
    if not resume_state:
        return {}
    return resume_rag_from_hitl(
        resume_state,
        user_answer,
        ctx,
        resolved=resolved,
        # The USER's question, not the query the agent wrote for the tool. The resume
        # state carries the latter, so anything the user said and the agent did not
        # repeat was already lost before this call.
        original_question=pending_hitl.get("original_question") or "",
    )


def _resume_constraints(rag_result: dict) -> list[str]:
    trace = rag_result.get("rag_trace") or {}
    return [str(item) for item in (trace.get("turn_carried_constraints") or [])]


def _answer_resumed_rag_sync(
    pending_hitl: dict,
    user_answer: str,
    rag_result: dict,
    history: list | None = None,
) -> str:
    docs = rag_result.get("docs") or []
    trace = rag_result.get("rag_trace") or {}
    status = rag_result.get("retrieval_status") or trace.get("retrieval_status")
    route = rag_result.get("route") or trace.get("route")
    if status == "retrieval_error" or route == "retrieval_error":
        return _retrieval_error_response()
    if status == "no_knowledge" or route == "no_knowledge" or not docs:
        return _no_knowledge_response()
    res = model.invoke(
        _build_resume_answer_messages(
            pending_hitl,
            user_answer,
            docs,
            resolved_question=rag_result.get("question") or "",
            constraints=_resume_constraints(rag_result),
            history=history,
        )
    )
    return _extract_ai_content(res)


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
    child_options = [str(item) for item in (getattr(turn_plan, "child_options", None) or [])]
    # This condition is the feature's single point of failure: a plan carrying a child
    # and nothing else renders nothing at all unless the child is named here too.
    if not resolved and not constraints and not child_hint and not child_options:
        return None
    rendered = resolve_prompt(
        "",
        "agent/turn_context.j2",
        resolved_question=resolved,
        constraints=constraints,
        child_hint=child_hint,
        child_year=child_year,
        child_options=child_options,
    )
    return SystemMessage(content=rendered) if rendered else None


def _build_context_messages(
    messages: list,
    persistent_note: str,
    user_text: str,
    turn_plan=None,
) -> list:
    short_term = messages[-CONTEXT_WINDOW_MESSAGES:] if len(messages) > CONTEXT_WINDOW_MESSAGES else messages
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
    context_messages.extend(short_term)
    # After the history and before the message it describes, so the model reads the
    # conversation, then what that conversation makes this message mean, then the
    # message itself.
    turn_context = _turn_context_message(turn_plan)
    if turn_context is not None:
        context_messages.append(turn_context)
    context_messages.append(HumanMessage(content=user_text))
    return context_messages


def _should_update_persistent_note(messages: list, current_note: str) -> bool:
    """Only pay for note maintenance once short-term context actually starts trimming."""
    return bool(current_note) or len(messages) > CONTEXT_WINDOW_MESSAGES


async def update_persistent_note(
    current_note: str,
    user_text: str,
    ai_response: str,
    history_messages: list | None = None,
) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _update_persistent_note_sync(
            current_note,
            user_text,
            ai_response,
            history_messages=history_messages,
        ),
    )


def generate_session_title(user_text: str) -> str:
    compact_title = " ".join(user_text.split()).strip(" \t\r\n。！？!?，,；;：:")
    return compact_title[:16] or _COPY.new_session_title


def _update_persistent_note_sync(
    current_note: str,
    user_text: str,
    ai_response: str,
    *,
    history_messages: list | None = None,
) -> str:
    try:
        history_text = ""
        if history_messages:
            history_lines = []
            for message in history_messages:
                role = "User" if isinstance(message, HumanMessage) else "AI"
                history_lines.append(f"{role}: {_extract_ai_content(message)}")
            history_text = (
                "\n\n▼ Prior conversation to summarize together when the note is first created:\n"
                + "\n".join(history_lines)
                + "\n\n"
            )
        instructions = resolve_prompt(
            _PROFILE.agent.persistent_note_prompt,
            "agent/persistent_note.j2",
            max_chars=_PROFILE.agent.persistent_note_max_chars,
        )
        prompt = (
            f"{instructions}\n\n"
            f"▼ Existing note:\n{current_note if current_note else 'None'}\n\n"
            f"{history_text}"
            f"▼ Latest turn:\nUser: {user_text}\nAI: {ai_response}\n\n"
            "Output the updated note directly (plain text, no explanations or Markdown code blocks):"
        )
        res = fast_model.invoke([HumanMessage(content=prompt)])
        # Enforced here, not merely asked for in the prompt. Models cannot count
        # characters, and this note is injected into the context of every later turn —
        # so a note that overruns its budget is not a one-off, it is a permanent
        # per-turn tax for the rest of the session.
        return (res.content or "").strip()[: _PROFILE.agent.persistent_note_max_chars]
    except Exception as e:
        print(f"Context Manager Error: {e}")
        return current_note


def _resolve_caller(
    caller: CallerIdentity | None, user_id: str
) -> tuple[CallerIdentity, str]:
    """Settle identity once, at the top of a turn, for both entry points.

    Returns `(caller, user_id)` guaranteed to agree. A provided `caller` wins, because
    it came from a verified token while the positional `user_id` is only a convenience
    default; deriving one from the other here means no code further down has to wonder
    which is authoritative.
    """
    if caller is not None:
        return caller, caller.user_id
    return CallerIdentity.for_user(user_id), user_id


def chat_with_agent(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    client_capabilities: ClientCapabilities | None = None,
    *,
    caller: CallerIdentity | None = None,
):
    """Serve one turn.

    `caller` carries the verified identity — including, for a signed-in parent, the
    token the records tool relays. Keyword-only and optional so every existing caller
    (tests, jobs, any integration) keeps working and simply serves a turn that is not
    a parent session.

    When given, it is authoritative: `user_id` is taken from it rather than from the
    positional argument, so the storage key and the identity can never disagree.
    """
    caller, user_id = _resolve_caller(caller, user_id)
    messages, metadata = storage.load_with_meta(user_id, session_id)
    asset_state = load_asset_state(metadata)
    # The pin travels by reference into the turn's context and back out into
    # `save_meta`, so a turn that resolves a child has already recorded it.
    child_state = load_child_state(metadata, guardian_id=caller.guardian_id if caller else "")
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0
    entry = _enter_turn(user_text, messages, metadata)
    pending_hitl = entry.pending_hitl
    invalid_pending_hitl = entry.invalid_pending_hitl
    is_hitl_resume = entry.is_hitl_resume
    resume_state = entry.resume_state
    effective_user_text = entry.effective_user_text
    hitl_answers = entry.hitl_answers
    original_question = entry.original_question
    history_for_answer = list(messages)

    ctx = ChatRequestContext.for_sync(
        user_id=user_id,
        session_id=session_id,
        asset_state=asset_state,
        caller=caller,
        child=child_state,
    )
    ctx.reset_knowledge_tool_budget()

    try:
        messages.append(HumanMessage(content=user_text))
        storage.save(user_id, session_id, messages)

        if is_hitl_resume and resume_state:
            rag_result = _resume_rag_from_hitl_sync(
                pending_hitl, user_text, ctx, entry.resolution
            )
            rag_trace = normalize_rag_trace(
                rag_result.get("rag_trace") if isinstance(rag_result, dict) else None
            )
            next_pending_hitl = None
            if _is_hitl_trace(rag_trace):
                next_pending_hitl = _build_pending_hitl(
                    rag_trace,
                    original_question or user_text,
                    previous_answers=hitl_answers,
                    resume_state=rag_result.get("hitl_resume_state"),
                )
                response_content = _format_hitl_message(
                    next_pending_hitl["prompt"],
                    next_pending_hitl["options"],
                )
            else:
                response_content = _answer_resumed_rag_sync(
                    pending_hitl, user_text, rag_result, history_for_answer
                )
        else:
            turn_plan, _turn_signals = plan_turn(
                effective_user_text, messages[:-1], ctx, resolution=entry.resolution
            )
            if turn_plan.short_circuit:
                # A confirmed out-of-domain question, or a social turn a profile
                # answers statically. The agent is never built, so this costs neither
                # the system prompt nor a single tool schema.
                response_content = turn_plan.static_reply
                rag_trace = normalize_rag_trace(turn_plan.as_trace())
                next_pending_hitl = None
            else:
                request_agent = create_agent_for_request(ctx, turn_plan.exposed_tools, turn_plan.language)
                context_messages = _build_context_messages(
                    messages[:-1], persistent_note, effective_user_text, turn_plan
                )
                result = request_agent.invoke(
                    {"messages": context_messages},
                    config={"recursion_limit": _PROFILE.agent.recursion_limit},
                )

                response_content = ""
                if isinstance(result, dict):
                    if "output" in result:
                        response_content = str(result["output"])
                    elif "messages" in result and result["messages"]:
                        response_content = message_text(result["messages"][-1])
                    else:
                        response_content = str(result)
                elif hasattr(result, "content"):
                    response_content = message_text(result)
                else:
                    response_content = str(result)
                # Same rules as the streamed path. The agent loop has ended, so the last
                # message answered rather than called a tool — but it may still be
                # wearing its transcript. See `backend/chat/finalize.py`.
                raw_response = response_content
                response_content = finalize_text(response_content)
                if raw_response.strip() and not response_content.strip():
                    # Everything the model said was transcript and none of it was an
                    # answer. The streamed path reaches this through the finalizer's
                    # counters; here the comparison IS the signal.
                    logger.warning(
                        "the model produced no answer channel this turn; serving the retry copy"
                    )
                    response_content = _COPY.retrieval_error

                stored_trace = ctx.take_rag_trace()
                rag_trace = normalize_rag_trace(stored_trace.get("rag_trace") if stored_trace else None)
                resume_state_from_trace = stored_trace.get("hitl_resume_state") if stored_trace else None
                next_pending_hitl = None
                if _is_hitl_trace(rag_trace):
                    next_pending_hitl = _build_pending_hitl(
                        rag_trace,
                        original_question or user_text,
                        previous_answers=hitl_answers,
                        resume_state=resume_state_from_trace,
                    )
                    response_content = _format_hitl_message(
                        next_pending_hitl["prompt"],
                        next_pending_hitl["options"],
                    )
                else:
                    # Same check the streamed path runs, and it has to be here too: this
                    # path serves the same answers to the same parents, and a rule that
                    # holds on one of two entry points is not a rule.
                    sync_finalizer = Finalizer()
                    sync_finalizer.replace_answer(response_content)
                    replacement = _enforce_grounding(sync_finalizer, rag_trace, turn_plan)
                    if replacement:
                        response_content = replacement
                    sync_finalizer.log_summary()
                    if rag_trace:
                        rag_trace.update(sync_finalizer.as_trace())

        # Assets this turn surfaced, rendered for whatever the caller can display.
        capabilities = effective_capabilities(client_capabilities, _PROFILE.assets.delivery)
        asset_references = build_asset_references(
            asset_ids_for_answer(response_content, ctx, rag_trace, _PROFILE.assets.delivery),
            capabilities,
            _PROFILE.assets.delivery,
        )
        rag_trace = attach_assets_to_trace(rag_trace, asset_references)

        save_meta = dict(metadata)
        save_asset_state(save_meta, asset_state)
        save_child_state(save_meta, child_state)
        if invalid_pending_hitl:
            save_meta[PENDING_HITL_KEY] = None
        if is_first_message:
            save_meta["title"] = generate_session_title(user_text)
        if next_pending_hitl:
            save_meta[PENDING_HITL_KEY] = next_pending_hitl
        else:
            # `superseded` clears it too: the user replaced the question rather than
            # answering it, so the clarification is spent whichever path ran.
            if is_hitl_resume or entry.superseded:
                save_meta[PENDING_HITL_KEY] = None
            if _should_update_persistent_note(messages, persistent_note):
                save_meta["persistent_note"] = _update_persistent_note_sync(
                    persistent_note,
                    effective_user_text,
                    response_content,
                    history_messages=messages[:-1] if not persistent_note else None,
                )

        messages.append(AIMessage(content=response_content))
        extra_message_data = _message_data_for_save(messages, rag_trace)
        storage.save(
            user_id,
            session_id,
            messages,
            metadata=save_meta,
            extra_message_data=extra_message_data,
        )

        return {
            "response": response_content,
            "rag_trace": rag_trace,
            "assets": [
                reference.model_dump(mode="json", exclude_none=True)
                for reference in asset_references
            ],
        }
    finally:
        ctx.close()


async def chat_with_agent_stream(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    client_capabilities: ClientCapabilities | None = None,
    *,
    caller: CallerIdentity | None = None,
):
    """Serve one turn, streaming. See `chat_with_agent` for `caller`.

    The two entry points take identity the same way on purpose: a parameter present on
    one and missing on the other is how a feature ends up working in the sync path and
    silently dead in the streaming one that users actually hit.
    """
    caller, user_id = _resolve_caller(caller, user_id)
    initial_step = {
        "type": "rag_step",
        "step": {
            "icon": "📨",
            "label": "Request received, preparing response",
            "detail": "",
            "elapsed_ms": 0,
            "stage_elapsed_ms": 0,
        },
    }
    yield f"data: {json.dumps(initial_step)}\n\n"

    messages, metadata = storage.load_with_meta(user_id, session_id)
    asset_state = load_asset_state(metadata)
    # The pin travels by reference into the turn's context and back out into
    # `save_meta`, so a turn that resolves a child has already recorded it.
    child_state = load_child_state(metadata, guardian_id=caller.guardian_id if caller else "")
    capabilities = effective_capabilities(client_capabilities, _PROFILE.assets.delivery)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0
    # On a worker thread: deciding whether this message answers the pending
    # clarification or replaces it may cost a small model call, and the event loop is
    # already streaming tokens to other requests.
    entry = await asyncio.to_thread(_enter_turn, user_text, list(messages), metadata)
    pending_hitl = entry.pending_hitl
    invalid_pending_hitl = entry.invalid_pending_hitl
    is_hitl_resume = entry.is_hitl_resume
    resume_state = entry.resume_state
    effective_user_text = entry.effective_user_text
    hitl_answers = entry.hitl_answers
    original_question = entry.original_question
    history_for_answer = list(messages)

    output_queue = asyncio.Queue()
    ctx = ChatRequestContext.for_stream(
        user_id=user_id,
        session_id=session_id,
        output_queue=output_queue,
        asset_state=asset_state,
        caller=caller,
        child=child_state,
    )
    ctx.reset_knowledge_tool_budget()

    try:
        messages.append(HumanMessage(content=user_text))
        storage.save(user_id, session_id, messages)

        if is_hitl_resume and resume_state:
            loop = asyncio.get_running_loop()
            resume_future = loop.run_in_executor(
                None,
                lambda: _resume_rag_from_hitl_sync(
                    pending_hitl, user_text, ctx, entry.resolution
                ),
            )

            while not resume_future.done():
                try:
                    event = await asyncio.wait_for(output_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                yield f"data: {json.dumps(event)}\n\n"

            while not output_queue.empty():
                event = output_queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"

            rag_result = await resume_future
            rag_trace = normalize_rag_trace(
                rag_result.get("rag_trace") if isinstance(rag_result, dict) else None
            )
            next_pending_hitl = None
            full_response = ""

            if _is_hitl_trace(rag_trace):
                next_pending_hitl = _build_pending_hitl(
                    rag_trace,
                    original_question or user_text,
                    previous_answers=hitl_answers,
                    resume_state=rag_result.get("hitl_resume_state"),
                )
                full_response = _format_hitl_message(
                    next_pending_hitl["prompt"],
                    next_pending_hitl["options"],
                )
            elif not (rag_result.get("docs") if isinstance(rag_result, dict) else None):
                full_response = _no_knowledge_response()
                yield f"data: {json.dumps({'type': 'content', 'content': full_response})}\n\n"
            else:
                answer_messages = _build_resume_answer_messages(
                    pending_hitl,
                    user_text,
                    rag_result.get("docs") or [],
                    resolved_question=rag_result.get("question") or "",
                    constraints=_resume_constraints(rag_result),
                    history=history_for_answer,
                )
                async for msg in model.astream(answer_messages):
                    content = _extract_ai_content(msg)
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

            # Assets get their own event, ahead of the trace, so a client can render
            # images without parsing the trace — which is diagnostic and may change.
            asset_references = build_asset_references(
                asset_ids_for_answer(full_response, ctx, rag_trace, _PROFILE.assets.delivery),
                capabilities,
                _PROFILE.assets.delivery,
            )
            if asset_references:
                payload = [
                    reference.model_dump(mode="json", exclude_none=True)
                    for reference in asset_references
                ]
                yield f"data: {json.dumps({'type': 'assets', 'assets': payload})}\n\n"
            rag_trace = attach_assets_to_trace(rag_trace, asset_references)

            if rag_trace:
                yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

            if next_pending_hitl:
                yield f"data: {json.dumps({'type': 'hitl_request', 'hitl': _build_hitl_event(next_pending_hitl)})}\n\n"

            yield "data: [DONE]\n\n"

            save_meta = dict(metadata)
            save_asset_state(save_meta, asset_state)
            save_child_state(save_meta, child_state)
            if invalid_pending_hitl:
                save_meta[PENDING_HITL_KEY] = None
            if next_pending_hitl:
                save_meta[PENDING_HITL_KEY] = next_pending_hitl
            else:
                save_meta[PENDING_HITL_KEY] = None
                if _should_update_persistent_note(messages, persistent_note):
                    try:
                        save_meta["persistent_note"] = await update_persistent_note(
                            persistent_note,
                            effective_user_text,
                            full_response,
                            history_messages=messages[:-1] if not persistent_note else None,
                        )
                    except Exception as e:
                        print(f"Update persistent note error: {e}")

            messages.append(AIMessage(content=full_response))
            extra_message_data = _message_data_for_save(messages, rag_trace)
            storage.save(
                user_id,
                session_id,
                messages,
                metadata=save_meta,
                extra_message_data=extra_message_data,
            )
            return

        # Plan the turn before building the agent. A confirmed out-of-domain question
        # ends here, having cost neither the system prompt nor a single tool schema.
        #
        # On a worker thread, because planning is the one blocking stretch left in this
        # generator and it is not a short one: rung 1 runs a bge-m3 forward pass (~91 ms
        # of saturated CPU) and rung 2, when it runs, is a round trip to the scope model.
        # Left on the event loop that time is stolen from every other request in the
        # process — including tokens already streaming to users who asked earlier.
        turn_plan, turn_signals = await asyncio.to_thread(
            lambda: plan_turn(
                effective_user_text, messages[:-1], ctx, resolution=entry.resolution
            )
        )
        if turn_plan.short_circuit:
            async for chunk in _stream_static_reply(
                turn_plan, turn_signals, user_text, user_id, session_id,
                messages, metadata, persistent_note, asset_state, is_first_message,
            ):
                yield chunk
            return

        request_agent = create_agent_for_request(ctx, turn_plan.exposed_tools, turn_plan.language)
        context_messages = _build_context_messages(
            messages[:-1], persistent_note, effective_user_text, turn_plan
        )

        session_title = None
        if is_first_message:
            session_title = generate_session_title(user_text)
            yield f"data: {json.dumps({'type': 'session_title', 'title': session_title, 'session_id': session_id})}\n\n"

        full_response = ""
        agent_error = None
        # Every chunk the model produces crosses this, and nothing reaches the browser
        # that it did not return. Held out here rather than inside the worker because
        # the grounding check below needs it once the stream has finished — see
        # `backend/chat/finalize.py`.
        finalizer = Finalizer()

        async def _agent_worker():
            nonlocal full_response, agent_error
            try:
                async for msg, _metadata in request_agent.astream(
                    {"messages": context_messages},
                    stream_mode="messages",
                    config={"recursion_limit": _PROFILE.agent.recursion_limit},
                ):
                    if isinstance(msg, ToolMessage):
                        finalizer.note_tool_result(msg)
                        continue
                    if not isinstance(msg, AIMessageChunk):
                        continue

                    # Not `continue`-d on a tool-call chunk any more: the model's prose
                    # arrives in DIFFERENT chunks of the same message, so skipping only
                    # the chunks carrying a tool-call delta forwarded all of it. The
                    # finalizer suppresses by message, which is the unit the rule is
                    # actually about.
                    content = finalizer.consider(msg)
                    if content and not _turn_is_asking_a_question(ctx):
                        full_response += content
                        await output_queue.put({"type": "content", "content": content})

                # The last message is only known to be over once the stream ends, so
                # whatever it was still holding is released here.
                #
                # Through the SAME gate as the chunks above, and that is not belt and
                # braces: the finalizer holds the opening of a message until it can rule
                # out a transcript header, so a reply SHORTER than that hold never takes
                # the streaming path at all and arrives entirely as this tail. A
                # clarification prompt is exactly that short, and it reaches the user as
                # a `hitl_request` event — putting it on the wire as content too showed
                # it twice.
                tail = finalizer.finish()
                if tail and not _turn_is_asking_a_question(ctx):
                    full_response += tail
                    await output_queue.put({"type": "content", "content": tail})
            except Exception as e:
                agent_error = str(e)
                await output_queue.put({"type": "error", "content": str(e)})
            finally:
                # The graph ends itself on a terminal tool result rather than spending a
                # model call to reword profile copy — see `_end_turn_on_terminal_retrieval`.
                # It leaves no assistant content behind, so the copy is put on the wire here.
                terminal_status = ctx.short_circuit_status()
                if terminal_status:
                    reply = _terminal_reply(terminal_status, turn_plan.language)
                    full_response = reply
                    await output_queue.put({"type": "content", "content": reply})
                await output_queue.put(None)

        agent_task = asyncio.create_task(_agent_worker())

        try:
            while True:
                event = await output_queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            raise
        finally:
            if not agent_task.done():
                agent_task.cancel()

        stored_trace = ctx.take_rag_trace()
        rag_trace = normalize_rag_trace(stored_trace.get("rag_trace") if stored_trace else None)
        resume_state_from_trace = stored_trace.get("hitl_resume_state") if stored_trace else None
        next_pending_hitl = None
        hitl_response_content = ""
        if _is_hitl_trace(rag_trace):
            next_pending_hitl = _build_pending_hitl(
                rag_trace,
                original_question or user_text,
                previous_answers=hitl_answers,
                resume_state=resume_state_from_trace,
            )
            hitl_response_content = _format_hitl_message(
                next_pending_hitl["prompt"],
                next_pending_hitl["options"],
            )
        else:
            # The answer is settled and the evidence is in hand, so this is the first
            # moment the two can be compared. A failure replaces what was streamed
            # rather than appending to it: the reader has already seen the figure, and
            # a correction underneath it would leave both on screen.
            replacement = _enforce_grounding(finalizer, rag_trace, turn_plan)
            if not replacement and not full_response.strip():
                replacement = _nothing_usable_reply(finalizer, turn_plan)
            if replacement:
                full_response = finalizer.replace_answer(replacement)
                yield f"data: {json.dumps({'type': 'content_replace', 'content': replacement})}\n\n"
            finalizer.log_summary()

        asset_references = build_asset_references(
            asset_ids_for_answer(full_response, ctx, rag_trace, _PROFILE.assets.delivery),
            capabilities,
            _PROFILE.assets.delivery,
        )
        if asset_references:
            payload = [
                reference.model_dump(mode="json", exclude_none=True)
                for reference in asset_references
            ]
            yield f"data: {json.dumps({'type': 'assets', 'assets': payload})}\n\n"
        rag_trace = attach_assets_to_trace(rag_trace, asset_references)
        if rag_trace:
            # What the finalize stage withheld, and what it made of the answer. Recorded
            # on every turn it ran, not only the failing ones: "nothing was dropped" and
            # "the stage never ran" are different facts, and a provider switch is
            # exactly when telling them apart matters.
            rag_trace.update(finalizer.as_trace())

        if rag_trace:
            yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

        if next_pending_hitl:
            yield f"data: {json.dumps({'type': 'hitl_request', 'hitl': _build_hitl_event(next_pending_hitl)})}\n\n"

        yield "data: [DONE]\n\n"

        save_meta = dict(metadata)
        save_asset_state(save_meta, asset_state)
        save_child_state(save_meta, child_state)
        if invalid_pending_hitl:
            save_meta[PENDING_HITL_KEY] = None
        if session_title:
            save_meta["title"] = session_title

        if next_pending_hitl:
            save_meta[PENDING_HITL_KEY] = next_pending_hitl
            full_response = hitl_response_content
        else:
            # `superseded` clears it too: the user replaced the question rather than
            # answering it, so the clarification is spent whichever path ran.
            if (is_hitl_resume or entry.superseded) and not agent_error:
                save_meta[PENDING_HITL_KEY] = None
            if _should_update_persistent_note(messages, persistent_note):
                try:
                    save_meta["persistent_note"] = await update_persistent_note(
                        persistent_note,
                        effective_user_text,
                        full_response,
                        history_messages=messages[:-1] if not persistent_note else None,
                    )
                except Exception as e:
                    print(f"Update persistent note error: {e}")

        messages.append(AIMessage(content=full_response))
        extra_message_data = _message_data_for_save(messages, rag_trace)
        storage.save(
            user_id,
            session_id,
            messages,
            metadata=save_meta,
            extra_message_data=extra_message_data,
        )
    finally:
        ctx.close()
