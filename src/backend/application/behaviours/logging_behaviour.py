"""Log which use case ran, and how it ended.

Named logging_behaviour.py rather than logging.py on purpose: a module called `logging` in a
package that also imports the standard library's `logging` is a trap that only fires once
somebody adds an implicit relative import, and the cost of avoiding it is one word.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class Logging:
    """One line when a use case starts, one when it ends or fails.

    The message TYPE is logged, never the message itself. A command carries a parent's
    question, a child's name and an access token, and none of that belongs in a log line -
    see the ChatRequestContext comments on why the backend holds a token it never decodes.
    """

    async def __call__(self, message: Any, nxt: Callable[[Any], Awaitable[Any]]) -> Any:
        name = type(message).__name__
        logger.debug("%s starting", name)
        try:
            result = await nxt(message)
        except Exception as exc:
            # exception(), not error(): the traceback is the useful half, and this is the
            # outermost place that still knows which use case it belonged to.
            logger.exception("%s failed: %s", name, type(exc).__name__)
            raise
        logger.debug("%s done", name)
        return result
