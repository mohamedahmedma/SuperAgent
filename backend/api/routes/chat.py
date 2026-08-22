import json
import re

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from backend.chat import chat_with_agent, chat_with_agent_stream
from backend.chat.caller_identity import CallerIdentity
from backend.infra.auth import AuthenticatedUser, get_current_user
from backend.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

# The conversation this message belongs to.
THREAD_HEADER = "X-Thread-ID"
# What a client that names no thread gets. Every such caller of one user shares it, which
# is survivable only because storage keys on (user_id, session_id) — see `_thread_id`.
DEFAULT_THREAD = "default_session"
# Long enough for a UUID and then some; short enough that the storage column
# (String(120), backend/db/models.py:38) cannot be overrun by a header.
_MAX_THREAD_ID = 120
_THREAD_ID_SAFE = re.compile(r"[^A-Za-z0-9._:-]")


def _thread_id(header_value: str | None, body_value: str | None) -> str:
    """Which conversation this message belongs to.

    The header is the contract; the body field is the older spelling of the same thing
    and stays supported, so a client that has not been redeployed keeps working. The
    header wins when both are present and disagree — it is the one the caller set
    deliberately, and a body default silently overriding it is how a client ends up
    writing into `default_session` while believing otherwise.

    Sanitised rather than rejected. This value becomes part of a cache key
    (`chat_messages:{user_id}:{session_id}`, backend/chat/storage.py:24) and a database
    column, and a caller-supplied string reaching either unfiltered is worth closing off
    even though the key is already namespaced per user. A client sending something odd
    should get a working conversation, not a 422 it cannot act on.
    """
    raw = (header_value or body_value or "").strip()
    if not raw:
        return DEFAULT_THREAD
    cleaned = _THREAD_ID_SAFE.sub("-", raw)[:_MAX_THREAD_ID]
    return cleaned or DEFAULT_THREAD



# Deliberately `def`, not `async def`. `chat_with_agent` is synchronous from end to
# end — the embedder's forward pass, the scope model call, retrieval, and every LLM
# call in the turn. Declared `async`, all of that ran on the event loop, so one turn
# blocked every other request in the process for its entire duration: a second user
# asking a question waited out the first user's answer before being looked at.
#
# A plain `def` hands the whole thing to Starlette's worker threadpool instead, which
# is what that pool is for. Concurrent turns then overlap, and the loop stays free to
# accept connections and stream other responses.
@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    x_thread_id: str | None = Header(default=None, alias=THREAD_HEADER),
):
    try:
        session_id = _thread_id(x_thread_id, request.session_id)
        resp = chat_with_agent(
            request.message,
            current_user.username,
            session_id,
            client_capabilities=request.client_capabilities,
            # Identity is assembled HERE, at the HTTP boundary, from a token whose
            # signature has been verified — and nowhere else. Everything downstream
            # receives it and cannot change it.
            caller=CallerIdentity.from_principal(current_user),
        )
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "The upstream model service triggered rate limiting/quota limits (429). "
                        "Please check your account quota/model status.\n"
                        f"Original error: {message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    x_thread_id: str | None = Header(default=None, alias=THREAD_HEADER),
):
    # Built once, outside the generator. The generator body runs after the response has
    # started, by which point the dependency-injected principal is no longer something
    # to be reaching into — capturing the value up front keeps the streaming path
    # identical to the sync one rather than subtly different.
    caller = CallerIdentity.from_principal(current_user)

    # Resolved outside the generator, for the same reason `caller` is: the generator
    # body runs after the response has started and must not be reaching back into
    # request-scoped dependencies.
    session_id = _thread_id(x_thread_id, request.session_id)

    async def event_generator():
        try:
            async for chunk in chat_with_agent_stream(
                request.message,
                current_user.username,
                session_id,
                client_capabilities=request.client_capabilities,
                caller=caller,
            ):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
