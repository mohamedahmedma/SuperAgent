"""RAG_FIX_PLAN items 38 and 37: the checks a turn passes before any work is spent on it.

The per-user limits are Lua scripts, so they run against a REAL Redis: `TEST_REDIS_URL`,
which CI provides as a service, or the local one on 6379. Each test owns a key prefix
and deletes it afterwards.
"""
import logging
import os
import time
import unittest
import uuid
from unittest.mock import patch

from backend.chat.admission import Refusal, TurnAdmission, TurnLease
from backend.provider_quota import ProviderQuotas

TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


def _redis():
    import redis

    try:
        client = redis.Redis.from_url(TEST_REDIS_URL, socket_timeout=2, socket_connect_timeout=2)
        client.ping()
        return client
    except Exception:
        return None


_CLIENT = _redis()
requires_redis = unittest.skipUnless(_CLIENT is not None, f"no Redis at {TEST_REDIS_URL}")


class _RedisTest(unittest.TestCase):
    def setUp(self):
        self.prefix = f"test-admission-{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        if _CLIENT is not None:
            for key in _CLIENT.scan_iter(f"{self.prefix}:*"):
                _CLIENT.delete(key)

    def door(self, **limits):
        limits.setdefault("turns_per_minute", 0)
        limits.setdefault("concurrent", 0)
        return TurnAdmission(redis=lambda: _CLIENT, key=lambda name: f"{self.prefix}:{name}", **limits)


@requires_redis
class TurnRateTests(_RedisTest):
    def test_a_burst_is_admitted_then_turns_come_at_the_sustained_rate(self):
        door = self.door(turns_per_minute=60, burst=3)  # one a second, three at once
        for _ in range(3):
            self.assertIsInstance(door.admit("parent"), TurnLease)
        refused = door.admit("parent")
        self.assertIsInstance(refused, Refusal)
        self.assertEqual("too_many_turns", refused.reason)
        self.assertGreater(refused.retry_after, 0.5)
        self.assertLessEqual(refused.retry_after, 1.0)
        time.sleep(refused.retry_after + 0.05)
        self.assertIsInstance(door.admit("parent"), TurnLease)

    def test_one_parents_limit_is_not_anothers(self):
        door = self.door(turns_per_minute=60, burst=1)
        door.admit("first")
        self.assertIsInstance(door.admit("first"), Refusal)
        self.assertIsInstance(door.admit("second"), TurnLease)

    def test_every_replica_shares_one_count(self):
        """Two doors on one Redis are two workers: the limit is the user's, not the process's."""
        first, second = self.door(turns_per_minute=60, burst=2), self.door(turns_per_minute=60, burst=2)
        first.admit("parent")
        second.admit("parent")
        self.assertIsInstance(first.admit("parent"), Refusal)


@requires_redis
class ConcurrentTurnTests(_RedisTest):
    def test_turns_in_flight_are_capped_and_a_finished_turn_frees_its_place(self):
        door = self.door(concurrent=2)
        first, second = door.admit("parent"), door.admit("parent")
        refused = door.admit("parent")
        self.assertEqual("turn_in_progress", refused.reason)
        first.release()
        first.release()  # released once, however often it is asked
        self.assertIsInstance(door.admit("parent"), TurnLease)
        second.release()

    def test_a_lease_left_by_a_dead_process_expires_by_itself(self):
        door = self.door(concurrent=1, lease_seconds=1)
        door.admit("parent")  # never released
        self.assertIsInstance(door.admit("parent"), Refusal)
        time.sleep(1.1)
        self.assertIsInstance(door.admit("parent"), TurnLease)


class FailOpenTests(unittest.TestCase):
    def test_a_limiter_that_cannot_reach_redis_admits_the_turn(self):
        import redis

        def unreachable():
            raise redis.ConnectionError("no route to Redis")

        door = TurnAdmission(redis=unreachable, turns_per_minute=12, concurrent=3)
        with self.assertLogs("backend.chat.admission", logging.WARNING):
            self.assertIsInstance(door.admit("parent"), TurnLease)


class ProviderBusyTests(unittest.TestCase):
    def test_no_turn_starts_while_the_provider_is_refusing_calls(self):
        quotas = ProviderQuotas(max_wait=6.0)
        quotas.gate("provider/stub-grade").observe(429, {"retry-after": "40"})

        def must_not_be_asked():
            raise AssertionError("the provider check needs no Redis round trip")

        refused = TurnAdmission(redis=must_not_be_asked, quotas=quotas, turns_per_minute=12).admit("p")
        self.assertEqual("provider_busy", refused.reason)
        self.assertGreater(refused.retry_after, 35)

    def test_a_cooldown_short_enough_to_wait_out_does_not_refuse_the_turn(self):
        quotas = ProviderQuotas(max_wait=6.0)
        quotas.gate("provider/stub-grade").observe(429, {"retry-after": "2"})
        self.assertIsInstance(TurnAdmission(quotas=quotas).admit("p"), TurnLease)


class _ChatRoute:
    """The real chat router on a bare app, with auth, services and the turn stubbed."""

    def __init__(self, door):
        from types import SimpleNamespace

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from backend.api.deps import get_services
        from backend.api.routes import chat as routes
        from backend.infra.auth import AuthenticatedUser, get_current_user

        self.started = 0

        async def turn(*args, **kwargs):
            self.started += 1
            yield 'data: {"type": "content", "content": "answer"}\n\n'
            yield "data: [DONE]\n\n"

        self._patch = patch.object(routes, "chat_with_agent_stream", turn)
        self._patch.start()
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[get_services] = lambda: SimpleNamespace(turn_admission=door)
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser("parent", "user")
        self.client = TestClient(app)

    def ask(self, message="What are the fees?"):
        return self.client.post("/chat/stream", json={"message": message, "session_id": "s"})

    def close(self):
        self._patch.stop()


class ChatRouteTests(_RedisTest):
    def test_a_refused_turn_is_a_429_with_retry_after_and_starts_nothing(self):
        quotas = ProviderQuotas(max_wait=6.0)
        quotas.gate("provider/stub-model").observe(429, {"retry-after": "40"})
        route = _ChatRoute(TurnAdmission(quotas=quotas))
        try:
            response = route.ask()
            self.assertEqual(429, response.status_code)
            self.assertGreaterEqual(int(response.headers["Retry-After"]), 39)
            self.assertIn("try again in", response.json()["detail"])
            self.assertIn("ثانية", route.ask("كم الرسوم الدراسية؟").json()["detail"])
            self.assertEqual(0, route.started)
        finally:
            route.close()

    @requires_redis
    def test_a_finished_stream_gives_its_place_back(self):
        route = _ChatRoute(self.door(concurrent=1))
        try:
            for _ in range(3):  # one at a time, so each needs the place the last one held
                response = route.ask()
                self.assertEqual(200, response.status_code, response.text)
                self.assertIn("[DONE]", response.text)
                deadline = time.time() + 2
                while _CLIENT.zcard(f"{self.prefix}:turns:inflight:parent") and time.time() < deadline:
                    time.sleep(0.02)
            self.assertEqual(3, route.started)
        finally:
            route.close()


class RefusalCopyTests(unittest.TestCase):
    def test_the_parent_is_told_in_their_language_how_long_to_wait(self):
        from backend.profiles import get_profile

        copy = get_profile().user_copy
        self.assertIn("5 seconds", Refusal("too_many_turns", 4.2).message(copy, "en"))
        self.assertIn("5 ثانية", Refusal("too_many_turns", 4.2).message(copy, "ar"))
        self.assertIn("41 seconds", Refusal("provider_busy", 40.01).message(copy, "en"))
        self.assertTrue(Refusal("turn_in_progress", 5).message(copy, "en"))


if __name__ == "__main__":
    unittest.main()
