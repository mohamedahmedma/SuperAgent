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

The same clients carry the live turn's rate-limit policy (item 37, the rules in
`backend/provider_quota.py`). `GatedTransport` holds or refuses a call the provider has
said it will reject, learns the quota from every response, and tells the SDK whether a
retry is worth making through the `x-should-retry` header the SDK obeys. So the SDK
stays the ONE place that retries, and it retries only what the policy allows.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import anyio
import httpx2
import openai

from backend.provider_quota import (
    RETRYABLE_STATUSES,
    ProviderGate,
    ProviderQuotas,
    QuotaExhausted,
)

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


def _quota_key(request: httpx2.Request) -> str:
    """Which quota a request counts against: the provider's host, and the model."""
    model = "*"
    try:
        body = json.loads(request.content or b"{}")
        if isinstance(body, dict) and body.get("model"):
            model = str(body["model"])
    except (httpx2.RequestNotRead, ValueError, TypeError):
        pass
    return f"{request.url.host}/{model}"


def _refusal(request: httpx2.Request, exc: QuotaExhausted) -> httpx2.Response:
    """The 429 the provider would have sent, made here instead of on the wire.

    Shaped like the provider's, so the SDK raises the same `RateLimitError` and every
    caller's existing rate-limit handling applies unchanged. `x-should-retry: false`,
    because the wait is already known to be longer than a turn may spend.
    """
    return httpx2.Response(
        429,
        headers={"retry-after": f"{exc.retry_after:.3f}", "x-should-retry": "false"},
        json={"error": {"message": str(exc), "type": "rate_limit", "code": "quota_cooldown"}},
        request=request,
    )


def _judge(gate: ProviderGate, response: httpx2.Response) -> None:
    gate.observe(response.status_code, response.headers)
    if response.status_code in RETRYABLE_STATUSES:
        response.headers["x-should-retry"] = "true" if gate.should_retry(response.headers) else "false"


class GatedTransport(httpx2.BaseTransport):
    """A sync transport that applies the provider's quota before and after each call.

    The hold is bounded by the gate's `max_wait`, so a pool thread waits a few seconds
    at most, and only when the provider has said the call would fail sooner.
    """

    def __init__(self, inner: httpx2.BaseTransport, quotas: ProviderQuotas) -> None:
        self._inner = inner
        self._quotas = quotas

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        gate = self._quotas.gate(_quota_key(request))
        try:
            wait = gate.wait_before_sending()
        except QuotaExhausted as exc:
            return _refusal(request, exc)
        if wait:
            time.sleep(wait)
        try:
            response = self._inner.handle_request(request)
        except httpx2.TransportError:
            gate.observe_failure()
            raise
        _judge(gate, response)
        return response

    def close(self) -> None:
        self._inner.close()


class AsyncGatedTransport(httpx2.AsyncBaseTransport):
    """The same, for the async client: a hold here costs no thread at all."""

    def __init__(self, inner: httpx2.AsyncBaseTransport, quotas: ProviderQuotas) -> None:
        self._inner = inner
        self._quotas = quotas

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        gate = self._quotas.gate(_quota_key(request))
        try:
            wait = gate.wait_before_sending()
        except QuotaExhausted as exc:
            return _refusal(request, exc)
        if wait:
            await anyio.sleep(wait)
        try:
            response = await self._inner.handle_async_request(request)
        except httpx2.TransportError:
            gate.observe_failure()
            raise
        _judge(gate, response)
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


def _timeout() -> httpx2.Timeout:
    """How long a model call may wait on the provider, stated rather than inherited.

    Unstated, the SDK's default applies: 600 s to read, so a provider that stalls holds
    a parent's turn, and a thread, for ten minutes. The read timeout bounds each wait
    for data, which on a stream is the gap between two chunks, not the whole answer.
    """
    return httpx2.Timeout(
        float(os.getenv("LLM_READ_TIMEOUT_SECONDS") or 60.0),
        connect=float(os.getenv("LLM_CONNECT_TIMEOUT_SECONDS") or 5.0),
    )


class ProviderHttpClients:
    """One sync and one async HTTP client, shared by every chat model in the process.

    Built for the live turn: the retry policy is the profile's `rag.model_retry_*`, the
    most a parent's turn should spend on a rate limit.
    """

    def __init__(self, *, quotas: ProviderQuotas | None = None) -> None:
        from backend.profiles import get_profile

        rag = get_profile().rag
        # `attempts` counts the first try, so it allows one fewer retry.
        self.max_retries = max(0, int(rag.model_retry_attempts) - 1)
        self.quotas = quotas or ProviderQuotas(
            max_wait=float(rag.model_retry_max_seconds),
            default_cooldown=float(rag.model_retry_base_seconds),
        )
        self.sync_client = openai.DefaultHttpxClient(
            transport=GatedTransport(httpx2.HTTPTransport(limits=_limits()), self.quotas)
        )
        self.async_client = openai.DefaultAsyncHttpxClient(
            transport=AsyncGatedTransport(
                DrainOnCloseTransport(httpx2.AsyncHTTPTransport(limits=_limits())), self.quotas
            )
        )

    def model_kwargs(self) -> dict[str, Any]:
        """What to pass to `init_chat_model` so a model uses these clients and this policy."""
        return {
            "http_client": self.sync_client,
            "http_async_client": self.async_client,
            "max_retries": self.max_retries,
            "timeout": _timeout(),
        }

    def close(self) -> None:
        self.sync_client.close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.async_client.aclose())
        else:
            loop.create_task(self.async_client.aclose())
