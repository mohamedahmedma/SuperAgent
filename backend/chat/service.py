import asyncio
import json
import logging

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from backend.assets.delivery import ClientCapabilities
from backend.chat.answer_blocks import (
    attach_answer_blocks,
    resolve_figure_markers,
    settle_answer_blocks,
)
from backend.chat.answer_checks import (
    enforce_forced_tool_ran,
    enforce_records_agreement,
    nothing_usable_reply,
    resumed_static_reply,
    terminal_reply,
)
from backend.chat.assets_bridge import (
    asset_ids_for_answer,
    attach_assets_to_trace,
    build_asset_references,
    effective_capabilities,
    trace_for_storage,
)
from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_context import load_child_state, save_child_state
from backend.chat.clarification import (
    PENDING_HITL_KEY,
    build_hitl_event,
    build_pending_hitl,
    child_choice_pending,
    enter_turn,
    format_hitl_message,
    is_hitl_trace,
    pending_resume_state,
    pin_the_child_the_parent_named,
)
from backend.chat.context_messages import (
    build_context_messages,
    build_resume_answer_messages,
)
from backend.chat.finalize import Finalizer, finalize_text, message_text
from backend.chat.orchestrator import plan_turn, resolve_turn_question
from backend.chat.request_context import ChatRequestContext
from backend.chat.resolution import ResolvedQuestion
from backend.chat.runtime import create_agent_for_request, fast_model, model
from backend.chat.storage import ConversationStorage
from backend.composition import Services, default_services
from backend.profiles import get_profile
from backend.prompts import resolve as resolve_prompt
from backend.schemas.chat import normalize_rag_trace

logger = logging.getLogger(__name__)

_PROFILE = get_profile()
_COPY = _PROFILE.user_copy

CONTEXT_WINDOW_MESSAGES = _PROFILE.agent.context_window_messages


def _extract_ai_content(msg) -> str:
    """The text of a model message, as a user may see it.

    Reading the content and cleaning it are one step on purpose. This function is the
    only way a direct `model.invoke`/`model.astream` result becomes a string in this
    module, so putting the transcript strip anywhere else would leave the paths that
    bypass the agent — the HITL resume answer, the persistent note — able to put a raw
    Harmony envelope in front of a user. See `backend/chat/finalize.py`.
    """
    return finalize_text(message_text(msg))


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
    is_first_message: bool,
    pending_hitl: dict | None = None,
    child_state=None,
    *,
    conversations: ConversationStorage,
):
    """Emit a planned reply that no model composed, and persist the turn.

    Streamed through the same event shapes as an agent reply — `session_title`,
    `content`, `trace`, `[DONE]` — because a client must not need to know which path
    produced its answer. The saving is that no agent was built: no system prompt, no
    tool schemas, no search.

    `pending_hitl` is set when the planned reply is a QUESTION rather than an answer —
    today, "which of your children?". It rides the same `hitl_request` event the
    retrieval clarifications use, so a client that already renders selectable options
    renders these without knowing a planner produced them, and it is persisted so the
    next message is read as the answer rather than as a new question.
    """
    if is_first_message:
        title = generate_session_title(user_text)
        yield f"data: {json.dumps({'type': 'session_title', 'title': title, 'session_id': session_id})}\n\n"

    reply = turn_plan.static_reply or ""
    if pending_hitl:
        reply = format_hitl_message(reply, pending_hitl["options"])
    yield f"data: {json.dumps({'type': 'content', 'content': reply})}\n\n"
    if pending_hitl:
        yield f"data: {json.dumps({'type': 'hitl_request', 'hitl': build_hitl_event(pending_hitl)})}\n\n"

    rag_trace = normalize_rag_trace({**turn_plan.as_trace(), **turn_signals.as_trace()})
    yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    save_meta = dict(metadata)
    # A turn the corpus never saw contributes nothing worth summarising, so the
    # persistent note is deliberately left alone — updating it would spend a model
    # call on the one path whose whole point is not making one.
    #
    # The pin is NOT in that category, and leaving it out was a bug. Both agent paths
    # save it, and a turn that ends here can still have settled a child: the parent
    # answers "which child?", `pin_the_child_the_parent_named` writes the pin, and the
    # re-planned turn then ends on static copy — a social reply, or the same question
    # again. Without this the choice they just made is thrown away.
    if child_state is not None:
        save_child_state(save_meta, child_state)
    save_meta[PENDING_HITL_KEY] = pending_hitl or None
    if is_first_message:
        save_meta.setdefault("title", generate_session_title(user_text))

    messages.append(AIMessage(content=reply))
    extra_message_data = _message_data_for_save(messages, rag_trace)
    conversations.save(user_id, session_id, messages, metadata=save_meta,
                 extra_message_data=extra_message_data)

    yield "data: [DONE]\n\n"


def _turn_is_asking_a_question(ctx) -> bool:
    """Whether retrieval has decided this turn ends in a question, not an answer.

    Read from the live trace rather than passed in, because the decision is made by the
    knowledge tool part-way through the turn — after the stream has already started.
    Named because two places have to consult it and a copy of the lookup in each is how
    they come apart: the second one did, and a clarification prompt was shown twice.
    """
    stored = ctx.peek_rag_trace()
    return is_hitl_trace(
        normalize_rag_trace(stored.get("rag_trace") if stored else None)
    )


def _tool_messages_in(result) -> list:
    """Every tool result in a finished agent run, for the path that has no stream.

    The streamed path sees each `ToolMessage` go past and hands it to the finalizer
    there. The synchronous path gets one dictionary at the end instead, so the same
    messages have to be read back out of it — same objects, same order, arrived
    differently. Tolerant of every shape `invoke` can return, because a grounding check
    that raised on an unexpected result would take the answer down with it.
    """
    if not isinstance(result, dict):
        return []
    return [m for m in (result.get("messages") or []) if isinstance(m, ToolMessage)]


def _resume_rag_from_hitl_sync(
    pending_hitl: dict,
    user_answer: str,
    ctx: ChatRequestContext,
    resolved: ResolvedQuestion | None = None,
) -> dict:
    from backend.rag.pipeline import resume_rag_from_hitl

    resume_state = pending_resume_state(pending_hitl)
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
    static_reply = resumed_static_reply(rag_result)
    if static_reply is not None:
        return static_reply
    res = model.invoke(
        build_resume_answer_messages(
            pending_hitl,
            user_answer,
            rag_result.get("docs") or [],
            resolved_question=rag_result.get("question") or "",
            constraints=_resume_constraints(rag_result),
            history=history,
        )
    )
    return _extract_ai_content(res)


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
    services: Services | None = None,
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
    conversations = (services or default_services()).conversations
    messages, metadata = conversations.load_with_meta(user_id, session_id)
    # The pin travels by reference into the turn's context and back out into
    # `save_meta`, so a turn that resolves a child has already recorded it.
    child_state = load_child_state(metadata, guardian_id=caller.guardian_id if caller else "")
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0
    entry = enter_turn(user_text, messages, metadata, resolve=resolve_turn_question)
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
        caller=caller,
        child=child_state,
    )
    ctx.reset_knowledge_tool_budget()
    # Settled before anything plans this turn, so the pin is in place by the time the
    # planner resolves a child. `child_state` is threaded by reference, so the choice is
    # already in what this turn will persist.
    if entry.child_choice:
        pin_the_child_the_parent_named(ctx, entry.child_choice)

    # The records this answer shows as tables, as data. Empty on every path that does not
    # reach an agent answer, which is most of the branches below.
    answer_blocks: list = []

    try:
        messages.append(HumanMessage(content=user_text))
        conversations.save(user_id, session_id, messages)

        if is_hitl_resume and resume_state:
            rag_result = _resume_rag_from_hitl_sync(
                pending_hitl, user_text, ctx, entry.resolution
            )
            rag_trace = normalize_rag_trace(
                rag_result.get("rag_trace") if isinstance(rag_result, dict) else None
            )
            next_pending_hitl = None
            if is_hitl_trace(rag_trace):
                next_pending_hitl = build_pending_hitl(
                    rag_trace,
                    original_question or user_text,
                    previous_answers=hitl_answers,
                    resume_state=rag_result.get("hitl_resume_state"),
                )
                response_content = format_hitl_message(
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
                # A confirmed out-of-domain question, a social turn a profile answers
                # statically, or a parent with two children who both match what they
                # said. The agent is never built, so this costs neither the system
                # prompt nor a single tool schema.
                next_pending_hitl = child_choice_pending(
                    turn_plan, original_question or user_text
                )
                response_content = (
                    format_hitl_message(turn_plan.static_reply, next_pending_hitl["options"])
                    if next_pending_hitl
                    else turn_plan.static_reply
                )
                rag_trace = normalize_rag_trace(turn_plan.as_trace())
            else:
                request_agent = create_agent_for_request(ctx, turn_plan.exposed_tools, turn_plan.language)
                context_messages = build_context_messages(
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
                # The graph ended itself on a terminal retrieval result, so its last message
                # is the TOOL's and not an answer. The streamed path serves the profile's own
                # reply here; the commit that taught it that never reached this path, which
                # then handed the tool result to the evidence cut and served its
                # could-not-verify copy. See `_end_turn_on_terminal_retrieval`.
                terminal_status = ctx.short_circuit_status()
                if terminal_status:
                    response_content = terminal_reply(terminal_status, turn_plan.language)
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
                if is_hitl_trace(rag_trace):
                    next_pending_hitl = build_pending_hitl(
                        rag_trace,
                        original_question or user_text,
                        previous_answers=hitl_answers,
                        resume_state=resume_state_from_trace,
                    )
                    response_content = format_hitl_message(
                        next_pending_hitl["prompt"],
                        next_pending_hitl["options"],
                    )
                else:
                    # Same check the streamed path runs, and it has to be here too: this
                    # path serves the same answers to the same parents, and a rule that
                    # holds on one of two entry points is not a rule.
                    sync_finalizer = Finalizer()
                    # Including what the tools returned. The streamed path collects these
                    # as they arrive; here the whole conversation is in hand at once, so
                    # they are replayed off it. Without this the check on THIS path had no
                    # record of a child's marks and passed every answer about them.
                    for tool_message in _tool_messages_in(result):
                        sync_finalizer.note_tool_result(tool_message)
                    sync_finalizer.replace_answer(response_content)
                    replacement = (
                        enforce_records_agreement(sync_finalizer, ctx, turn_plan)
                        or enforce_forced_tool_ran(sync_finalizer, ctx, turn_plan)
                    )
                    if replacement:
                        response_content = replacement
                    else:
                        # Same rule as the streamed path: the record goes under the
                        # sentence when the answer stands, and never under a refusal —
                        # and the figure markers resolve in the same order there.
                        response_content, answer_blocks = settle_answer_blocks(
                            resolve_figure_markers(response_content, ctx), ctx
                        )
                        rag_trace = attach_answer_blocks(rag_trace, answer_blocks)
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
        save_child_state(save_meta, child_state)
        if invalid_pending_hitl:
            save_meta[PENDING_HITL_KEY] = None
        if is_first_message:
            save_meta["title"] = generate_session_title(user_text)
        if next_pending_hitl:
            save_meta[PENDING_HITL_KEY] = next_pending_hitl
        else:
            # Answered, replaced, or settled by naming a child — every way a
            # clarification ends is decided in one place. See `TurnEntry`.
            if entry.spends_the_pending_question():
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
        conversations.save(
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
            "answer_blocks": answer_blocks,
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
    services: Services | None = None,
):
    """Serve one turn, streaming. See `chat_with_agent` for `caller`.

    The two entry points take identity the same way on purpose: a parameter present on
    one and missing on the other is how a feature ends up working in the sync path and
    silently dead in the streaming one that users actually hit.
    """
    caller, user_id = _resolve_caller(caller, user_id)
    conversations = (services or default_services()).conversations
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

    messages, metadata = conversations.load_with_meta(user_id, session_id)
    # The pin travels by reference into the turn's context and back out into
    # `save_meta`, so a turn that resolves a child has already recorded it.
    child_state = load_child_state(metadata, guardian_id=caller.guardian_id if caller else "")
    capabilities = effective_capabilities(client_capabilities, _PROFILE.assets.delivery)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0
    # On a worker thread: deciding whether this message answers the pending
    # clarification or replaces it may cost a small model call, and the event loop is
    # already streaming tokens to other requests.
    entry = await asyncio.to_thread(
        enter_turn, user_text, list(messages), metadata, resolve=resolve_turn_question
    )
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
        caller=caller,
        child=child_state,
    )
    ctx.reset_knowledge_tool_budget()
    # Settled before anything plans this turn, so the pin is in place by the time the
    # planner resolves a child. `child_state` is threaded by reference, so the choice is
    # already in what this turn will persist.
    if entry.child_choice:
        pin_the_child_the_parent_named(ctx, entry.child_choice)

    try:
        messages.append(HumanMessage(content=user_text))
        conversations.save(user_id, session_id, messages)

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

            if is_hitl_trace(rag_trace):
                next_pending_hitl = build_pending_hitl(
                    rag_trace,
                    original_question or user_text,
                    previous_answers=hitl_answers,
                    resume_state=rag_result.get("hitl_resume_state"),
                )
                full_response = format_hitl_message(
                    next_pending_hitl["prompt"],
                    next_pending_hitl["options"],
                )
            elif (static_reply := resumed_static_reply(rag_result)) is not None:
                # Nothing to answer from: an outage, or a search that found nothing. The
                # same rule the sync path applies — see `resumed_static_reply`.
                full_response = static_reply
                yield f"data: {json.dumps({'type': 'content', 'content': full_response})}\n\n"
            else:
                answer_messages = build_resume_answer_messages(
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
                yield f"data: {json.dumps({'type': 'hitl_request', 'hitl': build_hitl_event(next_pending_hitl)})}\n\n"

            yield "data: [DONE]\n\n"

            save_meta = dict(metadata)
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
            conversations.save(
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
                messages, metadata, persistent_note, is_first_message,
                child_choice_pending(turn_plan, entry.original_question or user_text),
                child_state,
                conversations=conversations,
            ):
                yield chunk
            return

        request_agent = create_agent_for_request(ctx, turn_plan.exposed_tools, turn_plan.language)
        context_messages = build_context_messages(
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
                    reply = terminal_reply(terminal_status, turn_plan.language)
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
        if is_hitl_trace(rag_trace):
            next_pending_hitl = build_pending_hitl(
                rag_trace,
                original_question or user_text,
                previous_answers=hitl_answers,
                resume_state=resume_state_from_trace,
            )
            hitl_response_content = format_hitl_message(
                next_pending_hitl["prompt"],
                next_pending_hitl["options"],
            )
        else:
            # The answer is settled and the evidence is in hand, so this is the first
            # moment the two can be compared. A failure replaces what was streamed
            # rather than appending to it: the reader has already seen the figure, and
            # a correction underneath it would leave both on screen.
            replacement = enforce_records_agreement(finalizer, ctx, turn_plan)
            if not replacement:
                replacement = enforce_forced_tool_ran(finalizer, ctx, turn_plan)
            if not replacement and not full_response.strip():
                replacement = nothing_usable_reply(finalizer, turn_plan)
            if replacement:
                full_response = finalizer.replace_answer(replacement)
                yield f"data: {json.dumps({'type': 'content_replace', 'content': replacement})}\n\n"
            else:
                # Nothing was withheld, so the record goes underneath the sentence — as
                # the TOOL rendered it, never as the model retyped it.
                #
                # After the replacement decisions, deliberately: a refusal must not be
                # followed by the very table it declined to stand behind.
                #
                # `content_replace` rather than a `content` append, because the client
                # ASSIGNS on replace — so what the bubble ends up holding is exactly what
                # gets stored, with no chance of the two diverging.
                #
                # Figure markers resolve first and on the same event: the raw `[FIGURE 1]`
                # has already streamed into the bubble, and this is what takes it back out
                # and puts the picture where it pointed.
                settled, answer_blocks = settle_answer_blocks(
                    resolve_figure_markers(full_response, ctx), ctx
                )
                if settled != full_response:
                    full_response = finalizer.replace_answer(settled)
                    # The data goes AHEAD of the text that carries its markers, so a client
                    # that draws the tables already holds them when their places arrive,
                    # rather than printing the markdown for one event and then swapping.
                    if answer_blocks:
                        yield f"data: {json.dumps({'type': 'answer_blocks', 'answer_blocks': answer_blocks})}\n\n"
                    yield f"data: {json.dumps({'type': 'content_replace', 'content': settled})}\n\n"
                rag_trace = attach_answer_blocks(rag_trace, answer_blocks)
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
            yield f"data: {json.dumps({'type': 'hitl_request', 'hitl': build_hitl_event(next_pending_hitl)})}\n\n"

        yield "data: [DONE]\n\n"

        save_meta = dict(metadata)
        save_child_state(save_meta, child_state)
        if invalid_pending_hitl:
            save_meta[PENDING_HITL_KEY] = None
        if session_title:
            save_meta["title"] = session_title

        if next_pending_hitl:
            save_meta[PENDING_HITL_KEY] = next_pending_hitl
            full_response = hitl_response_content
        else:
            # Answered, replaced, or settled by naming a child — every way a
            # clarification ends is decided in one place. See `TurnEntry`.
            if entry.spends_the_pending_question(agent_error=bool(agent_error)):
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
        conversations.save(
            user_id,
            session_id,
            messages,
            metadata=save_meta,
            extra_message_data=extra_message_data,
        )
    finally:
        ctx.close()
