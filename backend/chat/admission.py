"""Whether a turn may start: the checks made at the door, before any work is spent on it.

RAG_FIX_PLAN items 38 and 37. Nothing limited how many turns one user could start: a
script, a stuck retry loop or a double-sending client could open turns without bound,
each spending provider quota and a pool thread. And nothing stopped a turn from starting
while the model provider was refusing calls, so during a quota wall every turn started,
spent its first model calls, and failed partway. Measured under a wall of 60 requests a
minute per model: 39 of 60 turns failed mid-turn.

Three checks, cheapest first, each answered with 429 and `Retry-After`. This is the
layout Stripe describes for its own API: a request-rate limiter, a concurrent-request
limiter, and load shedding.

  1. **The provider is refusing calls** (item 37). Read from this process's quota gates
     (`backend/provider_quota.py`): while a model is in a cooldown longer than a turn may
     wait, a turn started now cannot finish, so none is started. No Redis involved.
  2. **Turn rate, per user** (item 38). GCRA, the generic cell rate algorithm: a token
     bucket kept as one timestamp. `CHAT_TURNS_PER_MINUTE` sustained,
     `CHAT_TURN_BURST` at once.
  3. **Concurrent turns, per user** (item 38). A lease per turn in a sorted set,
     released when the turn's stream ends, and expiring by itself
     (`CHAT_TURN_LEASE_SECONDS`) if the process holding it dies. `CHAT_CONCURRENT_TURNS`.

Checks 2 and 3 live in Redis, so every worker and replica shares one count. Each is one
Lua script, so it is atomic and costs one round trip. Each reads Redis's clock, so
replicas whose clocks drift still agree. Both FAIL OPEN: when Redis cannot be reached
the turn is admitted and the failure logged, because an outage of the limiter must not
become an outage of the product.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)

#: How soon to try again after being refused for having turns in flight. A turn's end
#: cannot be predicted; this is a typical turn.
_IN_PROGRESS_RETRY_SECONDS = 5.0

# The GCRA step. KEYS[1] holds the theoretical arrival time (TAT) of the next turn.
# ARGV: interval between turns at the sustained rate (s), burst size.
# Returns {1, "0"} when admitted, {0, seconds until one would be}.
_RATE_SCRIPT = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local interval = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local tat = tonumber(redis.call('GET', KEYS[1]) or now)
if tat < now then tat = now end
local next_tat = tat + interval
local allowed_at = next_tat - interval * burst
if allowed_at > now then
  return {0, tostring(allowed_at - now)}
end
redis.call('SET', KEYS[1], tostring(next_tat), 'PX', math.ceil((next_tat - now) * 1000))
return {1, '0'}
"""

# The concurrency lease. KEYS[1] is a sorted set of this user's leases scored by expiry.
# ARGV: limit, lease seconds, lease id. Expired leases are dropped before counting.
_LEASE_SCRIPT = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[1]) then
  return 0
end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[3])
redis.call('EXPIRE', KEYS[1], math.ceil(tonumber(ARGV[2])))
return 1
"""


@dataclass(frozen=True)
class Refusal:
    """A turn not started, why, and how long until trying again is worthwhile."""

    reason: str  # the copy key: provider_busy | too_many_turns | turn_in_progress
    retry_after: float

    @property
    def retry_after_seconds(self) -> int:
        return max(1, int(-(-self.retry_after // 1)))  # rounded up

    def message(self, copy: Any, language: str) -> str:
        from backend.chat.turn_policy import localized

        return localized(getattr(copy, self.reason), language).format(
            seconds=self.retry_after_seconds
        )


class TurnLease:
    """A turn's place in its user's concurrency limit, given back exactly once."""

    def __init__(self, release: Optional[Callable[[], None]] = None) -> None:
        self._release = release
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            release, self._release = self._release, None
        if release is None:
            return
        try:
            release()
        except Exception:  # expires by itself; a failed release only delays that
            logger.warning("could not release a turn lease; it will expire", exc_info=True)


class TurnAdmission:
    """The door every chat turn passes through."""

    def __init__(
        self,
        *,
        redis: Optional[Callable[[], Any]] = None,
        key: Callable[[str], str] = lambda name: name,
        quotas: Any = None,
        turns_per_minute: float = 0.0,
        burst: int = 1,
        concurrent: int = 0,
        lease_seconds: float = 300.0,
    ) -> None:
        self._redis = redis
        self._key = key
        self._quotas = quotas
        self.turns_per_minute = max(0.0, float(turns_per_minute))
        self.burst = max(1, int(burst))
        self.concurrent = max(0, int(concurrent))
        self.lease_seconds = max(1.0, float(lease_seconds))
        self._scripts: dict = {}
        self._last_logged = 0.0

    @classmethod
    def from_environment(cls, cache: Any, quotas: Any) -> "TurnAdmission":
        """The shipped limits, each overridable. 0 turns a per-user check off.

        A cache without a Redis client behind it (a test's stand-in) leaves only the
        provider check.
        """
        return cls(
            redis=getattr(cache, "client", None),
            key=getattr(cache, "key", lambda name: name),
            quotas=quotas,
            turns_per_minute=float(os.getenv("CHAT_TURNS_PER_MINUTE") or 12),
            burst=int(os.getenv("CHAT_TURN_BURST") or 6),
            concurrent=int(os.getenv("CHAT_CONCURRENT_TURNS") or 3),
            lease_seconds=float(os.getenv("CHAT_TURN_LEASE_SECONDS") or 300),
        )

    def admit(self, user: str) -> Union[TurnLease, Refusal]:
        """A lease to hold for the life of the turn, or the reason there is none."""
        refusing_for = self._quotas.refusing_for() if self._quotas is not None else 0.0
        if refusing_for > 0:
            return Refusal("provider_busy", refusing_for)
        if self._redis is None or (not self.turns_per_minute and not self.concurrent):
            return TurnLease()
        try:
            client = self._redis()
            if self.turns_per_minute:
                admitted, wait = self._script(client, "rate", _RATE_SCRIPT)(
                    keys=[self._key(f"turns:rate:{user}")],
                    args=[60.0 / self.turns_per_minute, self.burst],
                )
                if not int(admitted):
                    return Refusal("too_many_turns", float(wait))
            if not self.concurrent:
                return TurnLease()
            leases = self._key(f"turns:inflight:{user}")
            lease_id = uuid.uuid4().hex
            if not int(self._script(client, "lease", _LEASE_SCRIPT)(
                keys=[leases], args=[self.concurrent, self.lease_seconds, lease_id]
            )):
                return Refusal("turn_in_progress", _IN_PROGRESS_RETRY_SECONDS)
            return TurnLease(lambda: client.zrem(leases, lease_id))
        except Exception:
            self._log_open()
            return TurnLease()

    def _script(self, client: Any, name: str, source: str):
        script = self._scripts.get((id(client), name))
        if script is None:
            script = client.register_script(source)
            self._scripts[(id(client), name)] = script
        return script

    def _log_open(self) -> None:
        now = time.monotonic()
        if now - self._last_logged >= 60:
            self._last_logged = now
            logger.warning("turn limits unavailable (Redis); admitting turns unlimited",
                           exc_info=True)


__all__ = ["Refusal", "TurnAdmission", "TurnLease"]
