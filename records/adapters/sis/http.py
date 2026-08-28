"""One pooled HTTP client, built the same way for all three SIS adapters.

There were three of these — in the marks adapter, the guardian directory and the calendar
— and they had drifted. The marks adapter set explicit connection limits; the other two
inherited httpx's defaults and so kept only 20 connections alive, paying a fresh TCP
handshake per request above that concurrency. All three talk to **one** service from
**one** worker pool, so three different answers to "how wide is the pool" meant two of
them were wrong.

## The three rules that are not negotiable

**No retries.** A retry budget multiplies the timeout by the attempt count, and three
attempts at 10s is a chat turn nobody waits for. A SIS that is down should be reported as
down within one timeout.

**No redirects.** Every request carries `X-API-Key`, so a 302 to another host is the
school's credential handed to whoever controls it. A misconfigured base URL must fail
loudly, not leak the key.

**Read before locking.** The client is built once and then read on every request from
every worker thread. Taking a mutex to re-read an attribute that has not changed since
startup serialises the one path that most needs not to be — so the fast path reads it
unlocked, which is safe because the attribute is only ever assigned a fully-built client.
A thread either sees `None` and joins the slow path, or sees a client ready to use. The
lock still guards construction, and the second check inside it is what stops two threads
that both saw `None` from building two pools.
"""
from __future__ import annotations

import threading
from typing import Any

from records.config import outbound_pool_size


class PooledClient:
    """A lazily-built, process-wide `httpx.Client` for one SIS base URL.

    Lazy because importing an adapter must not open a socket — the test suites import
    every module before running anything — and because the pool size is read at build
    time, so a test that repoints configuration before the first request still takes
    effect.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._headers = dict(headers or {})
        self._client: Any = None
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return self._base_url

    def get(self) -> Any:
        """The client, built on first use. See the module docstring on the unlocked read."""
        client = self._client
        if client is not None:
            return client

        import httpx

        with self._lock:
            if self._client is None:
                pool = outbound_pool_size()
                self._client = httpx.Client(
                    base_url=self._base_url,
                    headers=self._headers or None,
                    timeout=httpx.Timeout(self._timeout),
                    transport=httpx.HTTPTransport(
                        retries=0,
                        limits=httpx.Limits(
                            max_connections=pool,
                            # Keepalive matches the pool rather than httpx's default of
                            # 20: a facade that makes two calls per parent question and
                            # then drops the connection pays the handshake again on the
                            # next one.
                            max_keepalive_connections=pool,
                        ),
                    ),
                    follow_redirects=False,
                )
            return self._client

    def close(self) -> None:
        """Release the pool. Called from the app's shutdown hook.

        Without it, `uvicorn --reload` leaks a connection pool per reload until the
        process runs out of sockets.
        """
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


#: Status codes that mean "somebody else will answer this". Never followed — see the
#: module docstring — and reported as a configuration failure rather than a redirect.
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def error_code(response: Any) -> str:
    """SIS's machine-readable failure code, or `""` when the body is not its envelope.

    Defensive because it runs on the failure path: an error handler that itself raises
    turns a diagnosable refusal into a 500 with no explanation. A body that is not SIS's
    envelope is itself the finding — something between here and the SIS answered instead
    of it.
    """
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - never fail while reporting a failure
        return ""
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("code") or "")
    return ""


__all__ = ["PooledClient", "REDIRECT_STATUSES", "error_code"]
