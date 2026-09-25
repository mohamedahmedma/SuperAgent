"""RAG_FIX_PLAN item 37: a provider's rate limit costs a live turn at most the profile's budget.

Before: one resolver call against a provider answering 429 sent 6 requests and held its
thread 15.1 s at `retry-after: 3`, and 126 s at `retry-after: 30`, under a profile that
allows 2 attempts and 6 s. The end-to-end tests here drive the real OpenAI SDK, through
LangChain as production builds it, against a server that answers scripted statuses and
counts what it is sent.
"""
import asyncio
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch

import openai

from backend.llm_http import ProviderHttpClients
from backend.provider_quota import (
    ProviderGate,
    ProviderQuotas,
    QuotaExhausted,
    RetryBudget,
    exhausted_for,
    parse_duration,
    retry_after,
)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class ParsingTests(unittest.TestCase):
    def test_durations_as_providers_write_them(self):
        self.assertEqual(3.0, parse_duration("3"))
        self.assertAlmostEqual(7.66, parse_duration("7.66s"))
        self.assertAlmostEqual(179.56, parse_duration("2m59.56s"))  # Groq's reset format
        self.assertAlmostEqual(0.25, parse_duration("250ms"))
        self.assertEqual(3720.0, parse_duration("1h2m"))
        for nonsense in (None, "", "soon", "2m59x"):
            self.assertIsNone(parse_duration(nonsense))

    def test_retry_after_in_every_spelling(self):
        self.assertEqual(1.5, retry_after({"retry-after-ms": "1500", "retry-after": "9"}))
        self.assertEqual(9.0, retry_after({"retry-after": "9"}))
        self.assertAlmostEqual(
            30.0, retry_after({"retry-after": "Thu, 01 Jan 2026 00:00:30 GMT"}, now=1767225600.0))
        self.assertIsNone(retry_after({}))

    def test_a_used_up_quota_is_read_from_the_rate_limit_headers(self):
        self.assertEqual(120.0, exhausted_for({"x-ratelimit-remaining-requests": "0",
                                               "x-ratelimit-reset-requests": "2m"}))
        self.assertEqual(120.0, exhausted_for({"x-ratelimit-remaining-requests": "0",
                                               "x-ratelimit-reset-requests": "2m",
                                               "x-ratelimit-remaining-tokens": "0",
                                               "x-ratelimit-reset-tokens": "7.5s"}))
        self.assertIsNone(exhausted_for({"x-ratelimit-remaining-requests": "12",
                                         "x-ratelimit-reset-requests": "2m"}))


class RetryBudgetTests(unittest.TestCase):
    def test_retries_stop_once_half_the_tokens_are_spent_and_return_with_successes(self):
        budget = RetryBudget(max_tokens=10, ratio=0.1)
        for _ in range(4):
            budget.record_failure()
        self.assertTrue(budget.allows_retry())  # 6 > 5
        budget.record_failure()
        self.assertFalse(budget.allows_retry())  # 5, not above half
        for _ in range(11):
            budget.record_success()
        self.assertTrue(budget.allows_retry())


class ProviderGateTests(unittest.TestCase):
    def gate(self, **kwargs):
        self.clock = _Clock()
        kwargs.setdefault("max_wait", 6.0)
        return ProviderGate("groq/m", clock=self.clock, **kwargs)

    def test_a_short_stated_wait_holds_the_next_call_with_jitter_then_releases_it(self):
        gate = self.gate()
        gate.observe(429, {"retry-after": "3"})
        wait = gate.wait_before_sending()
        self.assertGreaterEqual(wait, 3.0)
        self.assertLessEqual(wait, 3.75)
        self.clock.now += 3.5
        self.assertEqual(0.0, gate.wait_before_sending())

    def test_a_wait_longer_than_the_budget_is_refused_not_waited(self):
        gate = self.gate()
        gate.observe(429, {"retry-after": "30"})
        with self.assertRaises(QuotaExhausted) as caught:
            gate.wait_before_sending()
        self.assertAlmostEqual(30.0, caught.exception.retry_after)

    def test_a_429_that_names_no_delay_cools_down_for_the_default(self):
        gate = self.gate(default_cooldown=2.0)
        gate.observe(429, {})
        self.assertGreaterEqual(gate.wait_before_sending(), 2.0)

    def test_a_success_reporting_the_quota_used_up_cools_down_before_any_429(self):
        gate = self.gate()
        gate.observe(200, {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "40s"})
        with self.assertRaises(QuotaExhausted):
            gate.wait_before_sending()

    def test_no_retry_past_the_budget_or_once_the_retry_budget_is_spent(self):
        gate = self.gate()
        self.assertFalse(gate.should_retry({"retry-after": "30"}))
        self.assertTrue(gate.should_retry({"retry-after": "2"}))
        for _ in range(5):
            gate.observe(503, {})
        self.assertFalse(gate.should_retry({}))

    def test_each_model_has_its_own_gate(self):
        quotas = ProviderQuotas(max_wait=6.0)
        self.assertIs(quotas.gate("h/a"), quotas.gate("h/a"))
        self.assertIsNot(quotas.gate("h/a"), quotas.gate("h/b"))


class _ScriptedProvider:
    """A keep-alive chat-completions server answering from a script, then 200s.

    `script` is a list of (status, headers), one per request, consumed in order; after it
    runs out every request gets a completion. `per_model` scripts one model only.
    `requests` counts what arrived, by model.
    """

    def __init__(self, script=(), per_model=None):
        self.script, self.per_model = list(script), per_model
        self.requests, self._lock = [], threading.Lock()
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(64)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}/v1"
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        buffer = b""
        try:
            while True:
                while b"\r\n\r\n" not in buffer:
                    data = conn.recv(65536)
                    if not data:
                        return
                    buffer += data
                head, _, body = buffer.partition(b"\r\n\r\n")
                length = next(int(l.split(b":", 1)[1]) for l in head.split(b"\r\n")
                              if l.lower().startswith(b"content-length:"))
                while len(body) < length:
                    body += conn.recv(65536)
                buffer = body[length:]
                model = json.loads(body[:length])["model"]
                with self._lock:
                    self.requests.append(model)
                    scripted = (self.per_model in (None, model)) and self.script
                    status, headers = self.script.pop(0) if scripted else (200, {})
                if status == 200:
                    payload = {"id": "c", "object": "chat.completion", "created": 0, "model": model,
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": "ok"}}]}
                else:
                    payload = {"error": {"message": "Rate limit reached", "type": "requests"}}
                raw = json.dumps(payload).encode()
                lines = [f"HTTP/1.1 {status} X", "Content-Type: application/json",
                         f"Content-Length: {len(raw)}"] + [f"{k}: {v}" for k, v in headers.items()]
                conn.sendall(("\r\n".join(lines) + "\r\n\r\n").encode() + raw)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self._sock.close()


class LiveTurnPolicyTests(unittest.TestCase):
    """Through `init_chat_model` with the shared clients, as every chat-path model is built."""

    def setUp(self):
        self.clients = ProviderHttpClients(quotas=ProviderQuotas(max_wait=1.0, default_cooldown=0.1))
        self.servers = []

    def tearDown(self):
        self.clients.close()
        for server in self.servers:
            server.close()

    def provider(self, *args, **kwargs):
        server = _ScriptedProvider(*args, **kwargs)
        self.servers.append(server)
        return server

    def model(self, server, name="m"):
        from langchain.chat_models import init_chat_model

        return init_chat_model(model=name, model_provider="openai", api_key="x",
                               base_url=server.url, **self.clients.model_kwargs())

    def test_a_wait_longer_than_the_budget_fails_fast_on_one_request(self):
        server = self.provider([(429, {"retry-after": "30"})])
        started = time.perf_counter()
        with self.assertRaises(openai.RateLimitError):
            self.model(server).invoke("q")
        self.assertLess(time.perf_counter() - started, 1.0)
        self.assertEqual(1, len(server.requests))

    def test_a_short_wait_is_waited_once_and_the_retry_succeeds(self):
        server = self.provider([(429, {"retry-after": "0.3"})])
        started = time.perf_counter()
        self.assertEqual("ok", self.model(server).invoke("q").content)
        self.assertGreaterEqual(time.perf_counter() - started, 0.3)
        self.assertEqual(2, len(server.requests))

    def test_a_cooldown_holds_back_calls_the_provider_would_reject(self):
        server = self.provider([(429, {"retry-after": "30"})])
        model = self.model(server)
        for _ in range(3):
            with self.assertRaises(openai.RateLimitError):
                model.invoke("q")
        self.assertEqual(1, len(server.requests), "calls during the cooldown reached the provider")

    def test_a_quota_reported_used_up_is_respected_before_any_429(self):
        server = self.provider([(200, {"x-ratelimit-remaining-requests": "0",
                                       "x-ratelimit-reset-requests": "30s"})])
        model = self.model(server)
        self.assertEqual("ok", model.invoke("q").content)
        with self.assertRaises(openai.RateLimitError):
            model.invoke("q")
        self.assertEqual(1, len(server.requests))

    def test_the_cooldown_is_per_model(self):
        server = self.provider([(429, {"retry-after": "30"})], per_model="grader")
        with self.assertRaises(openai.RateLimitError):
            self.model(server, "grader").invoke("q")
        self.assertEqual("ok", self.model(server, "answerer").invoke("q").content)

    def test_retries_stop_when_most_calls_are_failing(self):
        server = self.provider([(503, {})] * 40)
        model = self.model(server)
        for _ in range(8):
            with self.assertRaises(openai.InternalServerError):
                model.invoke("q")
        # Each of the first calls retried once; with the budget spent, the last did not.
        self.assertLess(len(server.requests), 16)
        before = len(server.requests)
        with self.assertRaises(openai.InternalServerError):
            model.invoke("q")
        self.assertEqual(before + 1, len(server.requests))

    def test_the_streamed_answer_obeys_the_same_policy_without_holding_a_thread(self):
        server = self.provider([(429, {"retry-after": "30"})])
        model = self.model(server)

        async def main():
            started = time.perf_counter()
            try:
                with self.assertRaises(openai.RateLimitError):
                    async for _ in model.astream("q"):
                        pass
                return time.perf_counter() - started
            finally:
                # Its connections belong to this loop, so it is closed before the loop is.
                await self.clients.async_client.aclose()

        self.assertLess(asyncio.run(main()), 1.0)
        self.assertEqual(1, len(server.requests))


class ClientPolicyTests(unittest.TestCase):
    def test_the_policy_is_the_profiles_live_turn_budget(self):
        from backend.profiles import get_profile

        rag = get_profile().rag
        clients = ProviderHttpClients()
        try:
            kwargs = clients.model_kwargs()
            self.assertEqual(rag.model_retry_attempts - 1, kwargs["max_retries"])
            self.assertEqual(rag.model_retry_max_seconds, clients.quotas.max_wait)
            self.assertEqual(60.0, kwargs["timeout"].read)  # not the SDK's 600 s
            self.assertEqual(5.0, kwargs["timeout"].connect)
        finally:
            clients.close()


class BusyCopyTests(unittest.TestCase):
    """A rate-limited answer is shown as the profile's copy, not as the exception."""

    def rate_limited(self, seconds):
        import httpx2

        response = httpx2.Response(429, headers={"retry-after": seconds},
                                   request=httpx2.Request("POST", "http://provider/v1/chat/completions"))
        return openai.RateLimitError("rate limit: holding calls to provider/stub-model", response=response,
                                     body=None)

    def test_the_parent_is_told_how_long_to_wait_in_their_language(self):
        from backend.chat.service import _agent_failure_text

        english = _agent_failure_text(self.rate_limited("6.2"), "en")
        self.assertIn("7 seconds", english)
        self.assertNotIn("stub-model", english)
        self.assertIn("7 ثانية", _agent_failure_text(self.rate_limited("6.2"), "ar"))

    def test_any_other_failure_is_reported_as_before(self):
        from backend.chat.service import _agent_failure_text

        self.assertEqual("boom", _agent_failure_text(RuntimeError("boom"), "en"))


class _Throttled:
    def __init__(self, status, headers):
        self.status_code, self.headers = status, headers

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return {"data": [{"index": 0, "embedding": [0.6, 0.8]}]}


class EmbeddingQuotaTests(unittest.TestCase):
    """The embedding provider is a second quota, with its own gate."""

    def remote(self):
        from backend.indexing.embedding import _RemoteEmbedder

        with patch.dict("os.environ", {"EMBEDDING_BACKEND": "openai", "EMBEDDING_BASE_URL": "http://e/v1",
                                       "EMBEDDING_MODEL": "m"}, clear=False):
            remote = _RemoteEmbedder()
        remote.gate.max_wait = 1.0
        return remote

    def test_a_throttled_embedding_is_retried_once_after_the_stated_wait(self):
        answers = [_Throttled(429, {"retry-after": "0.2"}), _Throttled(200, {})]
        with patch("requests.Session.post", side_effect=lambda *a, **k: answers.pop(0)) as post:
            self.assertEqual([[0.6, 0.8]], self.remote().embed_documents(["q"]))
        self.assertEqual(2, post.call_count)

    def test_an_embedding_quota_past_the_budget_fails_fast_and_is_then_not_sent(self):
        import requests

        remote = self.remote()
        with patch("requests.Session.post",
                   return_value=_Throttled(429, {"retry-after": "30"})) as post:
            with self.assertRaises(requests.HTTPError):
                remote.embed_documents(["q"])
            with self.assertRaises(QuotaExhausted):
                remote.embed_documents(["q"])
        self.assertEqual(1, post.call_count)


if __name__ == "__main__":
    unittest.main()
