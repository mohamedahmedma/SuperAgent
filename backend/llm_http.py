"""The HTTP clients every chat model talks to its provider through — shared, and reusable.

RAG_FIX_PLAN item 48. Measured under load, the backend opened a NEW connection to the
model provider for roughly two model calls in three: 492 distinct sockets in 40 seconds.
Against the local stub that is nearly free; against a real provider over HTTPS it is a
TCP + TLS handshake per call — measured to Groq at +208 ms over a reused connection —
and a turn streams twice before its first word (the tool decision, then the answer).

The cause is in the OpenAI SDK, not in how models are built: `AsyncStream.__stream__`
stops reading at the `[DONE]` event and then calls `response.aclose()` with the end of
the chunked body still unread. httpx cannot hand back to the pool a connection whose
response it has not finished reading, so it closes it. Measured directly: the same
streams read with plain httpx kept one socket; through the SDK they kept none.

`DrainOnCloseTransport` wraps the transport and, when a response is closed early, reads
what is left before closing — after `[DONE]` that is a few bytes already in the socket
buffer, so the connection goes back to the pool instead of being thrown away. The drain
is capped in bytes and in time: a stream abandoned in the middle of an answer (a parent
pressed stop) is closed as before rather than read to the end, and a cancellation is
never swallowed.

One `ProviderHttpClients` per process, owned by the composition root, handed to every
chat model: a single pool per provider rather than one per model object.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import anyio
import httpx2
import openai

logger = logging.getLogger(__name__)

#: What is read after an early close before giving up and discarding the connection.
DRAIN_MAX_BYTES = 64 * 1024
DRAIN_MAX_SECONDS = 0.25


class _DrainOnClose(httpx2.AsyncByteStream):
    """The response body, with a close that first reads what is left of it.

    A plain iterator over the inner stream rather than an async generator of its own: a
    second generator iterating the same httpcore generator is what turns an early close
    into "generator is already running" errors. The drain is bounded by an anyio cancel
    scope, the cancellation httpcore is written for, and in bytes.
    """

    def __init__(self, inner: httpx2.AsyncByteStream) -> None:
        self._inner = inner
        self._iterator = None

    def __aiter__(self):
        self._iterator = self._inner.__aiter__()
        return self

    async def __anext__(self) -> bytes:
        return await self._iterator.__anext__()

    async def aclose(self) -> None:
        try:
            if self._iterator is not None:
                with anyio.move_on_after(DRAIN_MAX_SECONDS):
                    read = 0
                    while read <= DRAIN_MAX_BYTES:
                        read += len(await self._iterator.__anext__())
        except (StopAsyncIteration, httpx2.HTTPError, OSError, RuntimeError):
            # Finished (the tail was read), or the stream could not be finished cleanly:
            # either way the close below decides whether the connection is reused.
            pass
        finally:
            self._iterator = None
            await self._inner.aclose()


class DrainOnCloseTransport(httpx2.AsyncBaseTransport):
    """An async transport whose responses finish reading before they close."""

    def __init__(self, inner: httpx2.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await self._inner.handle_async_request(request)
        response.stream = _DrainOnClose(response.stream)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def _limits() -> httpx2.Limits:
    """The SDK's own pool sizes, with idle connections kept longer than its 5 seconds.

    Between one parent's turns a connection sits idle for longer than five seconds, and
    each expiry is another handshake. Providers keep idle connections open for far longer
    than that; `LLM_KEEPALIVE_SECONDS` tunes it.
    """
    return httpx2.Limits(
        max_connections=int(os.getenv("LLM_MAX_CONNECTIONS") or 1000),
        max_keepalive_connections=int(os.getenv("LLM_MAX_KEEPALIVE_CONNECTIONS") or 100),
        keepalive_expiry=float(os.getenv("LLM_KEEPALIVE_SECONDS") or 30.0),
    )


class ProviderHttpClients:
    """One sync and one async HTTP client, shared by every chat model in the process."""

    def __init__(self) -> None:
        self.sync_client = openai.DefaultHttpxClient(limits=_limits())
        self.async_client = openai.DefaultAsyncHttpxClient(
            transport=DrainOnCloseTransport(httpx2.AsyncHTTPTransport(limits=_limits()))
        )

    def model_kwargs(self) -> dict[str, Any]:
        """What to pass to `init_chat_model` so a model uses these clients."""
        return {"http_client": self.sync_client, "http_async_client": self.async_client}

    def close(self) -> None:
        self.sync_client.close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.async_client.aclose())
        else:
            loop.create_task(self.async_client.aclose())
