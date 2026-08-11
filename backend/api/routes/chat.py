import json
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.chat import chat_with_agent, chat_with_agent_stream
from backend.chat.caller_identity import CallerIdentity
from backend.infra.auth import AuthenticatedUser, get_current_user
from backend.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


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
    request: ChatRequest, current_user: AuthenticatedUser = Depends(get_current_user)
):
    try:
        session_id = request.session_id or "default_session"
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
    request: ChatRequest, current_user: AuthenticatedUser = Depends(get_current_user)
):
    # Built once, outside the generator. The generator body runs after the response has
    # started, by which point the dependency-injected principal is no longer something
    # to be reaching into — capturing the value up front keeps the streaming path
    # identical to the sync one rather than subtly different.
    caller = CallerIdentity.from_principal(current_user)

    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
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
