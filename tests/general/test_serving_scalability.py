"""Request coalescing, and the probes that make a fleet of replicas routable.

Coalescing is concurrency code, so most of these tests are about what happens when it
goes wrong rather than when it goes right. Its failure modes are the expensive kind:
a waiter whose event is never set blocks a server thread forever, and a leader that
dies without releasing leadership stops the process batching for the rest of its life.
Neither shows up as an error — both show up as the service getting slower and then
stopping.
"""
import threading
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.indexing.embedding as embedding_module
from backend.api.routes.health import router as health_router
from backend.indexing.embedding import CoalescingEmbedder, EmbeddingService


class RecordingEmbedder:
    """Records the shape of every batch it is handed."""

    def __init__(self, delay=0.0, fail=False):
        self.batches = []
        self.delay = delay
        self.fail = fail
        self._lock = threading.Lock()

    def embed_documents(self, texts):
        with self._lock:
            self.batches.append(list(texts))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("embedder exploded")
        return [[float(len(text))] for text in texts]


class CoalescingTests(unittest.TestCase):
    def test_a_lone_caller_waits_for_nobody(self):
        """The design rejects an artificial wait window precisely so this stays true:
        an idle service must not be slower than an unbatched one."""
        inner = RecordingEmbedder()
        embedder = CoalescingEmbedder(inner, max_batch=16)

        start = time.time()
        result = embedder.embed_documents(["only me"])
        elapsed = time.time() - start

        self.assertEqual([[7.0]], result)
        self.assertEqual([["only me"]], inner.batches)
        self.assertLess(elapsed, 0.25, "a solo call must not sit in a delay window")

    def test_concurrent_callers_are_merged_into_fewer_calls(self):
        """The point of the whole class. Callers arriving while a batch is in flight
        are swept by the leader's next pass rather than each paying a forward pass."""
        inner = RecordingEmbedder(delay=0.05)
        embedder = CoalescingEmbedder(inner, max_batch=16)

        results = {}
        barrier = threading.Barrier(12)

        def call(index):
            barrier.wait()
            results[index] = embedder.embed_documents([f"query-{index:02d}"])

        threads = [threading.Thread(target=call, args=(i,)) for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(12, len(results))
        self.assertLess(len(inner.batches), 12, "no coalescing happened at all")
        self.assertEqual(12, sum(len(batch) for batch in inner.batches))

    def test_every_caller_gets_the_vector_for_its_own_text(self):
        """The corruption this guards is silent: swap two vectors between callers and
        both still get a plausible answer to somebody else's question."""
        inner = RecordingEmbedder(delay=0.02)
        embedder = CoalescingEmbedder(inner, max_batch=8)

        mistakes = []
        barrier = threading.Barrier(16)

        def call(index):
            text = "x" * (index + 1)          # length is the identity here
            barrier.wait()
            vector = embedder.embed_documents([text])[0]
            if vector != [float(index + 1)]:
                mistakes.append((text, vector))

        threads = [threading.Thread(target=call, args=(i,)) for i in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual([], mistakes)

    def test_a_batch_never_exceeds_its_limit(self):
        inner = RecordingEmbedder(delay=0.05)
        embedder = CoalescingEmbedder(inner, max_batch=4)

        barrier = threading.Barrier(20)

        def call(index):
            barrier.wait()
            embedder.embed_documents([f"q{index}"])

        threads = [threading.Thread(target=call, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertTrue(inner.batches)
        self.assertLessEqual(max(len(batch) for batch in inner.batches), 4)

    def test_a_failing_embedder_raises_for_every_caller_in_the_batch(self):
        """Not just the leader. A waiter left un-woken holds a server thread until the
        process dies."""
        inner = RecordingEmbedder(delay=0.02, fail=True)
        embedder = CoalescingEmbedder(inner, max_batch=8)

        outcomes = []
        barrier = threading.Barrier(8)

        def call(index):
            barrier.wait()
            try:
                embedder.embed_documents([f"q{index}"])
                outcomes.append("returned")
            except Exception:
                outcomes.append("raised")

        threads = [threading.Thread(target=call, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(8, len(outcomes), "a caller never returned — it is still blocked")
        self.assertEqual({"raised"}, set(outcomes))

    def test_the_batcher_still_works_after_a_failure(self):
        """Leadership has to be released on the error path, or the first failure would
        leave the process unable to ever batch again."""
        inner = RecordingEmbedder(fail=True)
        embedder = CoalescingEmbedder(inner, max_batch=8)
        with self.assertRaises(Exception):
            embedder.embed_documents(["boom"])

        inner.fail = False
        self.assertEqual([[2.0]], embedder.embed_documents(["ok"]))

    def test_multi_text_calls_bypass_the_queue(self):
        """Ingest hands over hundreds of chunks at once. Those are already batches, and
        putting them through a query-latency queue would stall real queries behind them."""
        inner = RecordingEmbedder()
        embedder = CoalescingEmbedder(inner, max_batch=4)

        embedder.embed_documents(["a", "b", "c", "d", "e", "f"])
        self.assertEqual([["a", "b", "c", "d", "e", "f"]], inner.batches)

    def test_it_can_be_switched_off(self):
        with patch.dict("os.environ", {"EMBEDDING_COALESCE_MAX_BATCH": "1"}, clear=False):
            inner = RecordingEmbedder()
            self.assertIs(inner, embedding_module._wrap(inner))

    def test_it_is_on_by_default(self):
        with patch.dict("os.environ", {"EMBEDDING_COALESCE_MAX_BATCH": ""}, clear=False):
            self.assertIsInstance(
                embedding_module._wrap(RecordingEmbedder()), CoalescingEmbedder
            )


class ReadinessTests(unittest.TestCase):
    """Liveness and readiness answer different questions, and conflating them drops
    requests during a rolling deploy."""

    def client(self):
        app = FastAPI()
        app.include_router(health_router)
        return TestClient(app)

    def test_health_needs_no_dependency(self):
        """A shared-Postgres blip must not fail liveness on every replica at once and
        get the whole fleet restarted."""
        with patch("backend.infra.database.engine") as engine:
            engine.connect.side_effect = RuntimeError("database down")
            response = self.client().get("/health")
        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["status"])

    def test_ready_is_false_while_the_embedder_is_still_loading(self):
        """The ~110 s bge-m3 load is a window in which the process answers /health
        perfectly and can serve nobody."""
        service = EmbeddingService()
        self.assertFalse(service.is_ready)
        with patch.object(embedding_module, "embedding_service", service):
            response = self.client().get("/ready")
        self.assertEqual(503, response.status_code)
        self.assertEqual("not_ready", response.json()["status"])
        self.assertFalse(response.json()["checks"]["embedder"]["ok"])

    def test_a_readiness_probe_does_not_itself_load_the_model(self):
        """Otherwise every probe from every replica competes for the CPU the request
        path is contending for."""
        with patch.object(embedding_module, "_create_dense_embedder") as create:
            service = EmbeddingService()
            with patch.object(embedding_module, "embedding_service", service):
                self.client().get("/ready")
            create.assert_not_called()

    def test_ready_reports_which_dependency_failed(self):
        service = EmbeddingService(embedder=RecordingEmbedder())
        with patch.object(embedding_module, "embedding_service", service):
            with patch("backend.infra.database.engine") as engine:
                engine.connect.side_effect = RuntimeError("no route to host")
                response = self.client().get("/ready")

        body = response.json()
        self.assertEqual(503, response.status_code)
        self.assertTrue(body["checks"]["embedder"]["ok"])
        self.assertFalse(body["checks"]["database"]["ok"])
        self.assertIn("no route to host", body["checks"]["database"]["detail"])

    def test_probes_are_unauthenticated(self):
        """Load balancers and orchestrators hold no credentials."""
        from backend.api.router import router

        paths = {route.path for route in router.routes}
        self.assertIn("/health", paths)
        self.assertIn("/ready", paths)


if __name__ == "__main__":
    unittest.main()
