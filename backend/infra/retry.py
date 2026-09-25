"""Retrying a database write through the failures that pass by themselves.

RAG_FIX_PLAN item 21. A turn's save runs in the background (`backend/chat/background.py`)
so the parent is answered without waiting for it, and a save that failed was logged and
dropped: a Postgres restart, a dropped connection or a pool timeout under load cost the
parent the answer they had just read. These are the failures that succeed if tried again
a moment later. A write the database REFUSED (a constraint, a missing user) fails the
same way every time and is raised at once.

Exponential backoff with jitter, so writes that failed together do not retry together,
bounded to a few seconds in all. A caller must make the write safe to repeat: a
connection lost after the commit reached the database makes the retry run a write that
has already happened. The conversation save does this with per-message idempotency keys
(`MessageToStore.key`).
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

from sqlalchemy import exc as sa_exc

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Attempts in all, and the first wait: 0.2 s, then twice as long each time, with jitter.
#: Five attempts wait at most ~3 s between them, enough to ride out a failover.
DEFAULT_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 0.2


def is_transient(error: BaseException) -> bool:
    """Whether trying again could succeed.

    A lost or refused connection, a pool that had no connection to give, a serialization
    failure or deadlock (psycopg2 reports those as OperationalError): yes. An integrity
    error, a programming error, bad data: no.
    """
    if isinstance(error, sa_exc.DBAPIError) and error.connection_invalidated:
        return True
    return isinstance(error, (sa_exc.OperationalError, sa_exc.InterfaceError, sa_exc.TimeoutError))


def retry_transient(
    operation: Callable[[], T],
    *,
    describe: str = "database write",
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """`operation()`, tried again after a transient failure, up to `attempts` times."""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return operation()
        except Exception as error:
            if attempt >= attempts or not is_transient(error):
                raise
            delay = base_delay * (2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            logger.warning("%s failed (attempt %d of %d), retrying in %.2fs: %s",
                           describe, attempt, attempts, delay, error)
            sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


__all__ = ["is_transient", "retry_transient"]
