import json
import os
from typing import Any, Optional

import redis

from backend.profiles import get_profile


class RedisCache:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        # Profile-scoped so two profiles sharing a Redis instance cannot read each
        # other's cached parent chunks. REDIS_KEY_PREFIX still overrides (see
        # profiles/registry.py ENV_OVERRIDES).
        self.key_prefix = get_profile().identity.redis_key_prefix
        self.default_ttl = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        # How long a stalled Redis may hold a caller before the call becomes a miss
        # (RAG_FIX_PLAN item 44). Stated rather than inherited: redis-py 8's default is
        # 5 s per call, and a turn makes several cache calls before its first word, so
        # a stalled Redis cost a turn that many times 5 s. The cache is an optimisation,
        # and the database answers in well under a second.
        self.socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS") or 1.0)
        self.connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT_SECONDS") or 1.0)
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.connect_timeout,
                # A pooled connection idle this long is checked before use, so one the
                # server or a proxy dropped is replaced rather than failing a call.
                health_check_interval=30,
            )
        return self._client

    def client(self):
        """The shared client, for callers that need more than get/set (the turn limits)."""
        return self._get_client()

    def key(self, key: str) -> str:
        """`key` in this profile's namespace."""
        return self._key(key)

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def get_json(self, key: str) -> Optional[Any]:
        try:
            value = self._get_client().get(self._key(key))
            if not value:
                return None
            return json.loads(value)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False)
            self._get_client().setex(self._key(key), ttl or self.default_ttl, payload)
        except Exception:
            return

    def delete(self, key: str) -> None:
        try:
            self._get_client().delete(self._key(key))
        except Exception:
            return

    def delete_pattern(self, pattern: str) -> None:
        # SCAN, not KEYS: KEYS walks the whole keyspace in one command and blocks every
        # other client of the server until it finishes.
        try:
            client = self._get_client()
            batch = []
            for key in client.scan_iter(match=self._key(pattern), count=500):
                batch.append(key)
                if len(batch) >= 500:
                    client.delete(*batch)
                    batch = []
            if batch:
                client.delete(*batch)
        except Exception:
            return
