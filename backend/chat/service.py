import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from backend.chat.request_context import ChatRequestContext
from backend.chat.runtime import create_agent_for_request, fast_model, model
from backend.chat.storage import storage
from backend.schemas.chat import PendingHitlState, normalize_rag_trace

CONTEXT_WINDOW_MESSAGES = 6
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
        return "I found several knowledge base scopes that might be relevant. Please choose which one you'd like to continue querying."
    return "I found relevant knowledge, but I'm still missing one key piece of information. Please provide it so I can continue."


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
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, str):
                text += block
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        return text
    return str(content or "")


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
) -> list:
    original_question = pending_hitl.get("original_question") or ""
    prompt = pending_hitl.get("prompt") or ""
    context = _format_retrieved_chunks(docs)
    system = SystemMessage(
        content=(
            "You are a helpful knowledge-base assistant. "
            "Answer the user's original question using only the retrieved chunks. "
            "You MUST cite source chunks inline with [1], [2], etc. "
            "If the chunks are insufficient, say so honestly. "
            "Do not mention internal HITL or RAG implementation details."
        )
    )
    human = HumanMessage(
        content=(
            "Original question:\n"
            f"{original_question}\n\n"
            "HITL follow-up question:\n"
            f"{prompt}\n\n"
            "User's answer:\n"
            f"{user_answer}\n\n"
            "Retrieved chunks:\n"
            f"{context}\n\n"
            "Please answer the original question based on the retrieved chunks, and cite them using [1], [2], etc."
        )
    )
    return [system, human]


def _no_knowledge_response() -> str:
    return "The knowledge base does not contain reliable relevant information, so this question can't be answered from it right now."


def _retrieval_error_response() -> str:
    return "A temporary technical issue prevented searching the knowledge base. Please try again in a moment."


def _resume_rag_from_hitl_sync(pending_hitl: dict, user_answer: str, ctx: ChatRequestContext) -> dict:
    from backend.rag.pipeline import resume_rag_from_hitl

    resume_state = _pending_resume_state(pending_hitl)
    if not resume_state:
        return {}
    return resume_rag_from_hitl(resume_state, user_answer, ctx)


def _answer_resumed_rag_sync(pending_hitl: dict, user_answer: str, rag_result: dict) -> str:
    docs = rag_result.get("docs") or []
    trace = rag_result.get("rag_trace") or {}
    status = rag_result.get("retrieval_status") or trace.get("retrieval_status")
    route = rag_result.get("route") or trace.get("route")
    if status == "retrieval_error" or route == "retrieval_error":
        return _retrieval_error_response()
    if status == "no_knowledge" or route == "no_knowledge" or not docs:
        return _no_knowledge_response()
    res = model.invoke(_build_resume_answer_messages(pending_hitl, user_answer, docs))
    return _extract_ai_content(res)


def _build_context_messages(
    messages: list,
    persistent_note: str,
    user_text: str,
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
    return compact_title[:16] or "New session"


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
        prompt = (
            "You are a [Context Manager Agent], responsible for maintaining the \"persistent note\" across multi-turn conversations.\n"
            "The note is the model's long-term working memory under a limited context window, recording resolved questions and key facts.\n\n"
            "Update rules:\n"
            "1. Intelligently merge new information into the existing note; do not simply append it.\n"
            "2. Filter out noise and keep it under 500 characters, using concise bullet points.\n"
            "3. If information conflicts, keep the most reliable or most recent version.\n\n"
            f"▼ Existing note:\n{current_note if current_note else 'None'}\n\n"
            f"{history_text}"
            f"▼ Latest turn:\nUser: {user_text}\nAI: {ai_response}\n\n"
            "Output the updated note directly (plain text, no explanations or Markdown code blocks):"
        )
        res = fast_model.invoke([HumanMessage(content=prompt)])
        return (res.content or "").strip()
    except Exception as e:
        print(f"Context Manager Error: {e}")
        return current_note


def chat_with_agent(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
):
    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0
    stored_pending_hitl = metadata.get(PENDING_HITL_KEY)
    pending_hitl = _current_pending_hitl(stored_pending_hitl)
    invalid_pending_hitl = stored_pending_hitl is not None and pending_hitl is None
    is_hitl_resume = isinstance(pending_hitl, dict)
    resume_state = _pending_resume_state(pending_hitl)
    effective_user_text = (
        _build_hitl_resume_query(pending_hitl, user_text)
        if is_hitl_resume
        else user_text
    )
    hitl_answers = _existing_hitl_answers(pending_hitl)
    if is_hitl_resume:
        hitl_answers = [*hitl_answers, user_text]
    original_question = (
        pending_hitl.get("original_question")
        if is_hitl_resume
        else user_text
    )

    ctx = ChatRequestContext.for_sync(user_id=user_id, session_id=session_id)
    ctx.reset_knowledge_tool_budget()

    try:
        messages.append(HumanMessage(content=user_text))
        storage.save(user_id, session_id, messages)

        if is_hitl_resume and resume_state:
            rag_result = _resume_rag_from_hitl_sync(pending_hitl, user_text, ctx)
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
                response_content = _answer_resumed_rag_sync(pending_hitl, user_text, rag_result)
        else:
            request_agent = create_agent_for_request(ctx)
            context_messages = _build_context_messages(messages[:-1], persistent_note, effective_user_text)
            result = request_agent.invoke(
                {"messages": context_messages},
                config={"recursion_limit": 8},
            )

            response_content = ""
            if isinstance(result, dict):
                if "output" in result:
                    response_content = result["output"]
                elif "messages" in result and result["messages"]:
                    msg = result["messages"][-1]
                    response_content = getattr(msg, "content", str(msg))
                else:
                    response_content = str(result)
            elif hasattr(result, "content"):
                response_content = result.content
            else:
                response_content = str(result)

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

        save_meta = dict(metadata)
        if invalid_pending_hitl:
            save_meta[PENDING_HITL_KEY] = None
        if is_first_message:
            save_meta["title"] = generate_session_title(user_text)
        if next_pending_hitl:
            save_meta[PENDING_HITL_KEY] = next_pending_hitl
        else:
            if is_hitl_resume:
                save_meta[PENDING_HITL_KEY] = None
            if _should_update_persistent_note(messages, persistent_note):
                save_meta["persistent_note"] = _update_persistent_note_sync(
                    persistent_note,
                    effective_user_text,
                    response_content,
                    history_messages=messages[:-1] if not persistent_note else None,
                )

        messages.append(AIMessage(content=response_content))
        extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
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
        }
    finally:
        ctx.close()


async def chat_with_agent_stream(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
):
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
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0
    stored_pending_hitl = metadata.get(PENDING_HITL_KEY)
    pending_hitl = _current_pending_hitl(stored_pending_hitl)
    invalid_pending_hitl = stored_pending_hitl is not None and pending_hitl is None
    is_hitl_resume = isinstance(pending_hitl, dict)
    resume_state = _pending_resume_state(pending_hitl)
    effective_user_text = (
        _build_hitl_resume_query(pending_hitl, user_text)
        if is_hitl_resume
        else user_text
    )
    hitl_answers = _existing_hitl_answers(pending_hitl)
    if is_hitl_resume:
        hitl_answers = [*hitl_answers, user_text]
    original_question = (
        pending_hitl.get("original_question")
        if is_hitl_resume
        else user_text
    )

    output_queue = asyncio.Queue()
    ctx = ChatRequestContext.for_stream(
        user_id=user_id,
        session_id=session_id,
        output_queue=output_queue,
    )
    ctx.reset_knowledge_tool_budget()

    try:
        messages.append(HumanMessage(content=user_text))
        storage.save(user_id, session_id, messages)

        if is_hitl_resume and resume_state:
            loop = asyncio.get_running_loop()
            resume_future = loop.run_in_executor(
                None,
                lambda: _resume_rag_from_hitl_sync(pending_hitl, user_text, ctx),
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
                )
                async for msg in model.astream(answer_messages):
                    content = _extract_ai_content(msg)
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

            if rag_trace:
                yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

            if next_pending_hitl:
                yield f"data: {json.dumps({'type': 'hitl_request', 'hitl': _build_hitl_event(next_pending_hitl)})}\n\n"

            yield "data: [DONE]\n\n"

            save_meta = dict(metadata)
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
            extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
            storage.save(
                user_id,
                session_id,
                messages,
                metadata=save_meta,
                extra_message_data=extra_message_data,
            )
            return

        request_agent = create_agent_for_request(ctx)
        context_messages = _build_context_messages(messages[:-1], persistent_note, effective_user_text)

        session_title = None
        if is_first_message:
            session_title = generate_session_title(user_text)
            yield f"data: {json.dumps({'type': 'session_title', 'title': session_title, 'session_id': session_id})}\n\n"

        full_response = ""
        agent_error = None

        async def _agent_worker():
            nonlocal full_response, agent_error
            try:
                async for msg, _metadata in request_agent.astream(
                    {"messages": context_messages},
                    stream_mode="messages",
                    config={"recursion_limit": 8},
                ):
                    if not isinstance(msg, AIMessageChunk):
                        continue
                    if getattr(msg, "tool_call_chunks", None):
                        continue

                    content = ""
                    if isinstance(msg.content, str):
                        content = msg.content
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, str):
                                content += block
                            elif isinstance(block, dict) and block.get("type") == "text":
                                content += block.get("text", "")

                    if content:
                        stored_trace = ctx.peek_rag_trace()
                        rag_trace = normalize_rag_trace(
                            stored_trace.get("rag_trace") if stored_trace else None
                        )
                        if _is_hitl_trace(rag_trace):
                            continue
                        full_response += content
                        await output_queue.put({"type": "content", "content": content})
            except Exception as e:
                agent_error = str(e)
                await output_queue.put({"type": "error", "content": str(e)})
            finally:
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

        if rag_trace:
            yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

        if next_pending_hitl:
            yield f"data: {json.dumps({'type': 'hitl_request', 'hitl': _build_hitl_event(next_pending_hitl)})}\n\n"

        yield "data: [DONE]\n\n"

        save_meta = dict(metadata)
        if invalid_pending_hitl:
            save_meta[PENDING_HITL_KEY] = None
        if session_title:
            save_meta["title"] = session_title

        if next_pending_hitl:
            save_meta[PENDING_HITL_KEY] = next_pending_hitl
            full_response = hitl_response_content
        else:
            if is_hitl_resume and not agent_error:
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
        extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
        storage.save(
            user_id,
            session_id,
            messages,
            metadata=save_meta,
            extra_message_data=extra_message_data,
        )
    finally:
        ctx.close()
