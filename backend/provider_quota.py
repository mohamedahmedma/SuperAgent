"""What this process has learned about a provider's quota, and the rule for acting on it.

RAG_FIX_PLAN item 37. A provider's rate limit was handled by retrying, at two layers
that did not know about each other, on threads a turn cannot spare:

  * the OpenAI SDK retries a 429 twice and sleeps whatever `Retry-After` says, up to 60 s
    in the locked openai 2.x (two minutes in 3.x), on the calling thread;
  * three chat-path calls wrapped that in a retry of their own.

Measured against a provider answering 429 with `retry-after: 3`, one resolver call sent
6 requests and held its thread 15.1 s, under a profile that says "2 attempts, never wait
more than 6 s". Each rejected turn also went on sending: every turn fired, all were
rejected together, and all slept together.

This module holds the standard answers, one per failure:

  * **The quota is read from the provider, not guessed.** A 429's `Retry-After`, and
    `x-ratelimit-remaining-*: 0` with its `x-ratelimit-reset-*`, put that provider and
    model into a cooldown. A call made during it waits, when the wait fits the budget,
    or fails at once without being sent. Nothing is sent that the provider has already
    said it will reject. The headers are account-wide, so each replica learns the same
    state from its own responses and nothing needs sharing.
  * **A wait longer than the budget is not waited.** The profile's
    `model_retry_max_seconds` is the most a live turn waits on a quota. Past it, the
    call fails fast, and the turn's own fallbacks take over.
  * **Retries stop when most calls are failing.** gRPC's retry throttling (gRFC A6): a
    token count that failures drain and successes refill. Below half of it, nothing
    retries, so an outage is not met with double the traffic.

The transports that apply these rules to HTTP are in `backend/llm_http.py` (the chat
models) and `backend/indexing/embedding.py` (the embedding provider, a second quota).
"""
from __future__ import annotations

import email.utils
import logging
import random
import re
import threading
import time
from typing import Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

#: The statuses a client may retry: throttled, timed out, or failed on the server's side.
RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})

#: A duration as providers write it: "7.66s", "2m59.56s", "1h2m", "250ms".
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(value: Optional[str]) -> Optional[float]:
    """Seconds in `value`: a bare number of seconds, or a Go-style "2m59.56s"."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    parts = _DURATION_PART.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text.replace(" ", ""):
        return None
    return sum(float(number) * _UNIT_SECONDS[unit] for number, unit in parts)


def retry_after(headers: Mapping[str, str], *, now: Optional[float] = None) -> Optional[float]:
    """How long the provider asked to wait, in seconds, if it said.

    `retry-after-ms` first (OpenAI's, the most precise), then `retry-after` as seconds
    or as an HTTP date.
    """
    millis = headers.get("retry-after-ms")
    if millis is not None:
        try:
            return max(0.0, float(millis) / 1000.0)
        except ValueError:
            pass
    raw = headers.get("retry-after")
    if raw is None:
        return None
    seconds = parse_duration(raw)
    if seconds is not None:
        return seconds
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


def exhausted_for(headers: Mapping[str, str]) -> Optional[float]:
    """How long a quota the response reports as used up stays used up, if one is.

    `x-ratelimit-remaining-requests` and `-tokens` count what is left AFTER this
    response; at 0 the next call is rejected until the matching `-reset-` says.
    """
    longest = None
    for kind in ("requests", "tokens"):
        remaining = headers.get(f"x-ratelimit-remaining-{kind}")
        if remaining is None:
            continue
        try:
            if float(remaining) > 0:
                continue
        except ValueError:
            continue
        reset = parse_duration(headers.get(f"x-ratelimit-reset-{kind}"))
        if reset is not None and (longest is None or reset > longest):
            longest = reset
    return longest


class QuotaExhausted(Exception):
    """A call not sent, because the provider has said it would reject it."""

    def __init__(self, key: str, retry_after: float) -> None:
        super().__init__(
            f"rate limit: holding calls to {key} for {retry_after:.1f}s, as the provider asked"
        )
        self.key = key
        self.retry_after = retry_after


class RetryBudget:
    """gRPC's retry throttling (gRFC A6), for one provider and model.

    Starts full at `max_tokens`. A failed call takes one token and a successful one gives
    back `ratio`; retries are allowed only while more than half remain. So a provider
    failing most calls stops being retried within a few calls, which keeps the
    process from doubling its load on a provider that is already overloaded, and
    retrying comes back as calls succeed again (about one retry allowed per ten successes).
    """

    def __init__(self, max_tokens: float = 10.0, ratio: float = 0.1) -> None:
        self._max = float(max_tokens)
        self._ratio = float(ratio)
        self._tokens = self._max
        self._lock = threading.Lock()

    def record_failure(self) -> None:
        with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0)

    def record_success(self) -> None:
        with self._lock:
            self._tokens = min(self._max, self._tokens + self._ratio)

    def allows_retry(self) -> bool:
        with self._lock:
            return self._tokens > self._max / 2


class ProviderGate:
    """One provider account and model: its cooldown and its retry budget.

    `max_wait` is the longest a caller may be held. `default_cooldown` is used for a 429
    that names no delay.
    """

    def __init__(
        self,
        key: str,
        *,
        max_wait: float,
        default_cooldown: float = 1.0,
        budget: Optional[RetryBudget] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.key = key
        self.max_wait = max(0.0, float(max_wait))
        self.default_cooldown = max(0.0, float(default_cooldown))
        self.budget = budget or RetryBudget()
        self._clock = clock
        self._blocked_until = 0.0
        self._lock = threading.Lock()

    def remaining(self) -> float:
        """Seconds left in the current cooldown; 0 when there is none."""
        with self._lock:
            return max(0.0, self._blocked_until - self._clock())

    def wait_before_sending(self) -> float:
        """Seconds to hold this call before sending it; raises when longer than allowed.

        Waiters are spread over up to a quarter of the remaining cooldown, so the calls
        held by one cooldown do not all arrive together the instant it ends. That
        spreading is standard jitter.
        """
        with self._lock:
            remaining = self._blocked_until - self._clock()
        if remaining <= 0:
            return 0.0
        if remaining > self.max_wait:
            raise QuotaExhausted(self.key, remaining)
        return min(self.max_wait, remaining + random.uniform(0.0, remaining * 0.25))

    def observe(self, status: int, headers: Mapping[str, str]) -> None:
        """Learn from a response: its quota headers, and whether the call succeeded."""
        cooldown = exhausted_for(headers)
        if status == 429:
            stated = retry_after(headers)
            cooldown = max(cooldown or 0.0, stated if stated is not None else self.default_cooldown)
        if cooldown:
            with self._lock:
                self._blocked_until = max(self._blocked_until, self._clock() + cooldown)
        if status in RETRYABLE_STATUSES:
            self.budget.record_failure()
        elif status < 400:
            self.budget.record_success()

    def observe_failure(self) -> None:
        """A call that got no response at all: a connection error or a timeout."""
        self.budget.record_failure()

    def should_retry(self, headers: Mapping[str, str]) -> bool:
        """Whether a retryable failure is worth retrying.

        Not when the provider asked for longer than a turn may wait: the same call is
        rejected again, and the turn only finds out later. Not when the retry budget is
        spent.
        """
        stated = retry_after(headers)
        if stated is not None and stated > self.max_wait:
            return False
        return self.budget.allows_retry()


class ProviderQuotas:
    """The gates of one process, one per provider host and model.

    Keyed per model because that is how providers count: Groq's and OpenAI's limits are
    per model, so a grader hitting its tokens-per-minute limit says nothing about the
    answering model's.
    """

    def __init__(self, *, max_wait: float, default_cooldown: float = 1.0) -> None:
        self.max_wait = max_wait
        self.default_cooldown = default_cooldown
        self._gates: Dict[str, ProviderGate] = {}
        self._lock = threading.Lock()

    def gate(self, key: str) -> ProviderGate:
        gate = self._gates.get(key)
        if gate is None:
            with self._lock:
                gate = self._gates.get(key)
                if gate is None:
                    gate = ProviderGate(
                        key, max_wait=self.max_wait, default_cooldown=self.default_cooldown
                    )
                    self._gates[key] = gate
        return gate

    def refusing_for(self) -> float:
        """How long some model's calls will be refused: the longest cooldown past `max_wait`.

        0 when every model can be called now or within the budget. The turn limits ask
        this before starting a turn (`backend/chat/admission.py`): a turn started while
        one of its models refuses calls fails partway, having spent the calls before it.
        """
        with self._lock:
            gates = list(self._gates.values())
        longest = max((gate.remaining() for gate in gates), default=0.0)
        return longest if longest > self.max_wait else 0.0


__all__ = [
    "ProviderGate",
    "ProviderQuotas",
    "QuotaExhausted",
    "RETRYABLE_STATUSES",
    "RetryBudget",
    "exhausted_for",
    "parse_duration",
    "retry_after",
]
