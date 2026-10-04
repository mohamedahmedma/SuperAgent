"""How long a use case took.

Kept separate from Logging so a deployment can have one without the other, and so the clock
can be substituted in a test without also silencing the log.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class Timing:
    """Measure a use case, and report it whether it succeeded or not.

    `time.perf_counter` rather than `time.time`: this measures an elapsed interval, and a
    wall clock that steps backwards over an NTP correction would report a negative duration.

    The clock is injectable because the alternative is a test that sleeps. Pass any
    zero-argument callable returning seconds as a float.
    """

    def __init__(self, *, threshold_seconds: float = 0.0, clock: Callable[[], float] | None = None):
        #: Below this, nothing is logged. Zero logs everything, which is the default because
        #: a threshold chosen before there are measurements is a guess.
        self.threshold_seconds = threshold_seconds
        self._clock = clock or time.perf_counter

    async def __call__(self, message: Any, nxt: Callable[[Any], Awaitable[Any]]) -> Any:
        name = type(message).__name__
        started = self._clock()
        try:
            return await nxt(message)
        finally:
            # finally, so a failed use case is still measured. A handler that takes nine
            # seconds and then raises is the one worth knowing the duration of.
            elapsed = self._clock() - started
            if elapsed >= self.threshold_seconds:
                logger.info("%s took %.1f ms", name, elapsed * 1000)
