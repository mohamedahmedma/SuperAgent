"""RAG_FIX_PLAN item 18: a query vector is computed once per text for the whole deployment."""
import threading
import time
import unittest
import uuid

from backend.indexing.query_vectors import QueryVectorCache
from tests.general.test_turn_admission import _CLIENT, requires_redis


class _Embedder:
    def __init__(self, delay=0.0, fail=False):
        self.calls, self.delay, self.fail = [], delay, fail
        self._lock = threading.Lock()

    def __call__(self, text):
        with self._lock:
            self.calls.append(text)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("provider down")
        return [0.25, -0.5, float(len(text))]


class MemoTests(unittest.TestCase):
    def test_a_text_is_embedded_once_and_each_caller_gets_its_own_copy(self):
        embed = _Embedder()
        cache = QueryVectorCache(embed)
        first = cache.vector("fees?")
        first.append(99.0)
        self.assertEqual([0.25, -0.5, 5.0], cache.vector("fees?"))
        self.assertEqual(["fees?"], embed.calls)

    def test_the_memo_is_bounded(self):
        embed = _Embedder()
        cache = QueryVectorCache(embed, memo_size=2)
        for text in ("a", "b", "c", "a"):
            cache.vector(text)
        self.assertEqual(["a", "b", "c", "a"], embed.calls)  # "a" was evicted by "c"

    def test_the_same_text_asked_at_once_is_computed_once(self):
        embed = _Embedder(delay=0.2)
        cache = QueryVectorCache(embed)
        results = []
        threads = [threading.Thread(target=lambda: results.append(cache.vector("uniform?")))
                   for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(1, len(embed.calls))
        self.assertEqual([[0.25, -0.5, 8.0]] * 8, results)

    def test_a_failure_reaches_every_waiter_and_is_not_remembered(self):
        embed = _Embedder(delay=0.1, fail=True)
        cache = QueryVectorCache(embed)
        errors = []

        def ask():
            try:
                cache.vector("fees?")
            except RuntimeError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=ask) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(4, len(errors))
        embed.fail = False
        self.assertEqual([0.25, -0.5, 5.0], cache.vector("fees?"))

    def test_with_no_ttl_the_shared_layer_is_never_asked(self):
        def must_not_be_asked():
            raise AssertionError("no Redis round trip with the shared layer off")

        cache = QueryVectorCache(_Embedder(), redis=must_not_be_asked, ttl_seconds=0)
        self.assertEqual([0.25, -0.5, 1.0], cache.vector("x"))

    def test_an_unreachable_redis_is_a_miss(self):
        import redis

        def unreachable():
            raise redis.ConnectionError("down")

        embed = _Embedder()
        cache = QueryVectorCache(embed, redis=unreachable, ttl_seconds=60)
        self.assertEqual([0.25, -0.5, 1.0], cache.vector("x"))
        self.assertEqual(["x"], embed.calls)


@requires_redis
class SharedLayerTests(unittest.TestCase):
    def setUp(self):
        self.prefix = f"test-qv-{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        for key in _CLIENT.scan_iter(f"{self.prefix}:*"):
            _CLIENT.delete(key)

    def worker(self, embed, namespace="bge-m3"):
        return QueryVectorCache(embed, redis=lambda: _CLIENT, key=lambda name: f"{self.prefix}:{name}",
                                namespace=namespace, ttl_seconds=60)

    def test_a_second_worker_reads_what_the_first_computed(self):
        first, second = _Embedder(), _Embedder()
        question = "كم الرسوم؟"
        self.worker(first).vector(question)
        self.assertEqual([0.25, -0.5, float(len(question))], self.worker(second).vector(question))
        self.assertEqual([], second.calls)

    def test_another_model_is_another_key_space(self):
        self.worker(_Embedder(), namespace="bge-m3").vector("fees?")
        other = _Embedder()
        self.worker(other, namespace="qwen3").vector("fees?")
        self.assertEqual(["fees?"], other.calls)

    def test_vectors_keep_float32_precision(self):
        values = [0.1234567, -0.7654321, 1e-4]
        QueryVectorCache(lambda text: values, redis=lambda: _CLIENT,
                         key=lambda name: f"{self.prefix}:{name}", ttl_seconds=60).vector("q")
        read = self.worker(_Embedder(), namespace="").vector("q")
        for expected, got in zip(values, read):
            self.assertAlmostEqual(expected, got, places=6)


if __name__ == "__main__":
    unittest.main()
