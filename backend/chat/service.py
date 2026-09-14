"""The two chat entry points, and the collaborators a turn is assembled from.

`chat_with_agent` returns one dict; `chat_with_agent_stream` streams events to the web app.
Every decision a turn makes lives in `backend/chat/turn_pipeline.py`, so what is left here
is only delivery: how the agent is run, which calls leave the event loop, and when each
event goes on the wire relative to the save.

The collaborators below stay module attributes on purpose. They are gathered into a
`TurnCollaborators` at call time, so a test that patches one of them here — the planner,
the agent factory, a model — still reaches the turn it drives.
"""
import asyncio
import json
import logging

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from backend.assets.delivery import ClientCapabilities
from backend.chat.caller_identity import CallerIdentity
from backend.chat.clarification import build_hitl_event, pending_resume_state
from backend.chat.finalize import Finalizer, visible_text
from backend.chat.orchestrator import plan_turn, resolve_turn_question
from backend.chat.request_context import ChatRequestContext
from backend.chat.resolution import ResolvedQuestion
from backend.chat.runtime import create_agent_for_request, fast_model, model
from backend.chat.turn_pipeline import StreamedAnswer, TurnCollaborators, TurnPipeline, resolve_caller
from backend.composition import Services, default_services
from backend.profiles import get_profile
from backend.prompts import resolve as resolve_prompt

logger = logging.getLogger(__name__)

_PROFILE = get_profile()
_COPY = _PROFILE.user_copy

#: The first event of every streamed turn, sent before anything slow happens.
_REQUEST_RECEIVED = {
    "type": "rag_step",
    "step": {
        "icon": "📨",
        "label": "Request received, preparing response",
        "detail": "",
        "elapsed_ms": 0,
        "stage_elapsed_ms": 0,
    },
}
_DONE = "data: [DONE]\n\n"


def _event(payload) -> str:
    return f"data: {json.dumps(payload)}\n\n"


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
                history_lines.append(f"{role}: {visible_text(message)}")
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


def _pipeline(services: Services | None) -> TurnPipeline:
    """A turn pipeline over this module's collaborators, as they are at call time."""
    return TurnPipeline(TurnCollaborators(
        conversations=(services or default_services()).conversations,
        profile=_PROFILE,
        plan=plan_turn,
        resolve_question=resolve_turn_question,
        create_agent=create_agent_for_request,
        resume_retrieval=_resume_rag_from_hitl_sync,
        answer_model=model,
        session_title=generate_session_title,
        update_note=_update_persistent_note_sync,
        update_note_async=update_persistent_note,
        context_type=ChatRequestContext,
    ))


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
    caller, user_id = resolve_caller(caller, user_id)
    pipeline = _pipeline(services)
    turn = pipeline.open(user_text, user_id, session_id, caller)
    pipeline.enter(turn)
    ctx = pipeline.sync_context(turn)
    try:
        pipeline.record_question(turn)
        answered = None

        if pipeline.resumes_a_search(turn):
            prompt = pipeline.settle_resumed_search(turn, pipeline.run_resumed_search(turn))
            if prompt is not None:
                turn.answer = visible_text(pipeline.collaborators.answer_model.invoke(prompt))
        else:
            pipeline.plan(turn)
            if turn.plan.short_circuit:
                pipeline.settle_short_circuit(turn)
            else:
                agent, context_messages, config = pipeline.agent_call(turn)
                pipeline.name_the_session(turn)
                result = agent.invoke({"messages": context_messages}, config=config)
                answered = pipeline.read_invoked_answer(turn, result)
                pipeline.settle_agent_answer(turn, answered)

        pipeline.attach_assets(turn, client_capabilities)
        if answered is not None:
            pipeline.record_finalize_stage(turn, answered)

        save_meta = pipeline.save_metadata(turn)
        if pipeline.note_is_due(turn):
            args, kwargs = pipeline.note_request(turn)
            save_meta["persistent_note"] = pipeline.collaborators.update_note(*args, **kwargs)
        pipeline.commit(turn, save_meta)
        return pipeline.response(turn)
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

    One generator with every branch inline, deliberately. A branch delegated to a nested
    async generator would be left suspended when the client disconnects — Python does not
    finalize it at the outer `yield` — and the handler that cancels the agent would run
    later, at garbage collection, instead of now.
    """
    caller, user_id = resolve_caller(caller, user_id)
    pipeline = _pipeline(services)
    yield _event(_REQUEST_RECEIVED)

    turn = pipeline.open(user_text, user_id, session_id, caller)
    # On a worker thread: deciding whether this message answers the pending clarification
    # or replaces it may cost a small model call, and the event loop is already streaming
    # tokens to other requests.
    await asyncio.to_thread(pipeline.enter, turn)

    output_queue = asyncio.Queue()
    ctx = pipeline.stream_context(turn, output_queue)
    try:
        pipeline.record_question(turn)

        if pipeline.resumes_a_search(turn):
            loop = asyncio.get_running_loop()
            search = loop.run_in_executor(None, lambda: pipeline.run_resumed_search(turn))
            while not search.done():
                try:
                    event = await asyncio.wait_for(output_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                yield _event(event)
            while not output_queue.empty():
                yield _event(output_queue.get_nowait())

            prompt = pipeline.settle_resumed_search(turn, await search)
            if turn.next_pending is None:
                if prompt is None:
                    yield _event({"type": "content", "content": turn.answer})
                else:
                    async for msg in pipeline.collaborators.answer_model.astream(prompt):
                        content = visible_text(msg)
                        if content:
                            turn.answer += content
                            yield _event({"type": "content", "content": content})

            # Assets get their own event, ahead of the trace, so a client can render images
            # without parsing the trace — which is diagnostic and may change.
            pipeline.attach_assets(turn, client_capabilities)
            if turn.asset_references:
                yield _event({"type": "assets", "assets": turn.asset_payload()})
            if turn.rag_trace:
                yield _event({"type": "trace", "rag_trace": turn.rag_trace})
            if turn.next_pending:
                yield _event({"type": "hitl_request", "hitl": build_hitl_event(turn.next_pending)})
            yield _DONE

            save_meta = pipeline.save_metadata(turn)
            await _update_note_streamed(pipeline, turn, save_meta)
            pipeline.commit(turn, save_meta)
            return

        # Plan the turn before building the agent. A confirmed out-of-domain question ends
        # here, having cost neither the system prompt nor a single tool schema.
        #
        # On a worker thread, because planning is the one blocking stretch left in this
        # generator and it is not a short one: rung 1 runs a bge-m3 forward pass (~91 ms of
        # saturated CPU) and rung 2, when it runs, is a round trip to the scope model. Left
        # on the event loop that time is stolen from every other request in the process —
        # including tokens already streaming to users who asked earlier.
        await asyncio.to_thread(pipeline.plan, turn)

        if turn.plan.short_circuit:
            # Streamed through the same event shapes as an agent reply, because a client must
            # not need to know which path produced its answer. A planned QUESTION rides the
            # same `hitl_request` event the retrieval clarifications use.
            pipeline.settle_short_circuit(turn)
            if turn.title:
                yield _event({"type": "session_title", "title": turn.title, "session_id": session_id})
            yield _event({"type": "content", "content": turn.answer})
            if turn.next_pending:
                yield _event({"type": "hitl_request", "hitl": build_hitl_event(turn.next_pending)})
            yield _event({"type": "trace", "rag_trace": turn.rag_trace})
            pipeline.commit(turn, pipeline.save_metadata(turn))
            yield _DONE
            return

        agent, context_messages, config = pipeline.agent_call(turn)
        pipeline.name_the_session(turn)
        if turn.title:
            yield _event({"type": "session_title", "title": turn.title, "session_id": session_id})

        full_response = ""
        # Every chunk the model produces crosses this, and nothing reaches the browser that it
        # did not return. Held out here rather than inside the worker because settling the
        # answer needs it once the stream has finished — see `backend/chat/finalize.py`.
        finalizer = Finalizer()

        async def _agent_worker():
            nonlocal full_response
            try:
                async for msg, _metadata in agent.astream(
                    {"messages": context_messages},
                    stream_mode="messages",
                    config=config,
                ):
                    if isinstance(msg, ToolMessage):
                        finalizer.note_tool_result(msg)
                        continue
                    if not isinstance(msg, AIMessageChunk):
                        continue

                    # Not `continue`-d on a tool-call chunk: the model's prose arrives in
                    # DIFFERENT chunks of the same message, so skipping only the chunks
                    # carrying a tool-call delta forwarded all of it. The finalizer
                    # suppresses by message, which is the unit the rule is actually about.
                    content = finalizer.consider(msg)
                    if content and not pipeline.is_asking_a_question(turn):
                        full_response += content
                        await output_queue.put({"type": "content", "content": content})

                # The last message is only known to be over once the stream ends, so whatever
                # it was still holding is released here — through the SAME gate as the chunks
                # above. A reply shorter than the finalizer's hold arrives entirely as this
                # tail, and a clarification prompt is exactly that short: it reaches the user
                # as a `hitl_request` event, and putting it on the wire as content too showed
                # it twice.
                tail = finalizer.finish()
                if tail and not pipeline.is_asking_a_question(turn):
                    full_response += tail
                    await output_queue.put({"type": "content", "content": tail})
            except Exception as e:
                turn.agent_error = str(e)
                await output_queue.put({"type": "error", "content": str(e)})
            finally:
                # The graph ends itself on a terminal tool result rather than spending a model
                # call to reword profile copy, leaving no assistant content behind, so the copy
                # is put on the wire here.
                reply = pipeline.terminal_reply(turn)
                if reply is not None:
                    full_response = reply
                    await output_queue.put({"type": "content", "content": reply})
                await output_queue.put(None)

        agent_task = asyncio.create_task(_agent_worker())
        try:
            while True:
                event = await output_queue.get()
                if event is None:
                    break
                yield _event(event)
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

        answered = StreamedAnswer(text=full_response, finalizer=finalizer)
        settlement = pipeline.settle_agent_answer(turn, answered)
        if settlement.replacement:
            yield _event({"type": "content_replace", "content": settlement.replacement})
        elif settlement.settled is not None:
            # The data goes AHEAD of the text that carries its markers, so a client that draws
            # the tables already holds them when their places arrive. `content_replace` rather
            # than an append, because the client ASSIGNS on replace — what the bubble ends up
            # holding is exactly what gets stored.
            if settlement.blocks:
                yield _event({"type": "answer_blocks", "answer_blocks": settlement.blocks})
            yield _event({"type": "content_replace", "content": settlement.settled})

        pipeline.attach_assets(
            turn, client_capabilities, answer=full_response if settlement.asking else None
        )
        if turn.asset_references:
            yield _event({"type": "assets", "assets": turn.asset_payload()})
        pipeline.record_finalize_stage(turn, answered)
        if turn.rag_trace:
            yield _event({"type": "trace", "rag_trace": turn.rag_trace})
        if turn.next_pending:
            yield _event({"type": "hitl_request", "hitl": build_hitl_event(turn.next_pending)})
        yield _DONE

        save_meta = pipeline.save_metadata(turn)
        await _update_note_streamed(pipeline, turn, save_meta)
        pipeline.commit(turn, save_meta)
    finally:
        ctx.close()


async def _update_note_streamed(pipeline: TurnPipeline, turn, save_meta: dict) -> None:
    """Maintain the persistent note off the event loop, when the turn is due for it."""
    if not pipeline.note_is_due(turn):
        return
    args, kwargs = pipeline.note_request(turn)
    try:
        save_meta["persistent_note"] = await pipeline.collaborators.update_note_async(*args, **kwargs)
    except Exception as e:
        print(f"Update persistent note error: {e}")
