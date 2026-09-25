"""Where the embedder comes from, and when it is built.

Both questions are about memory rather than correctness of any single vector. bge-m3 is
~2.0 GB resident, and it was constructed at module import — so every process that
imported this backend paid it, once per process, whether or not it would ever embed
anything. Four uvicorn workers meant four copies of identical weights.

So: construction is lazy (a process that does not embed does not pay), and the backend
is selectable (a fleet of workers can share one copy behind an OpenAI-compatible
endpoint, or an API can serve it, without any caller knowing the difference).

The remote path's tests are mostly about NOT corrupting the corpus. A wrong vector is
worse than no vector: it is stored, it is searched, and nothing downstream can tell.
"""
import unittest
from unittest.mock import patch

import backend.indexing.embedding as embedding_module
from backend.indexing.embedding import (
    CoalescingEmbedder,
    EmbeddingService,
    _RemoteEmbedder,
    _create_dense_embedder,
)


class Sentinel:
    """Stands in for a real embedder so no test here loads 2 GB of weights."""

    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text))] for text in texts]


class LazyConstructionTests(unittest.TestCase):
    def test_constructing_the_service_does_not_construct_the_embedder(self):
        """The regression this guards is silent and expensive: it does not fail, it
        just costs ~2 GB and ~110 s in a process that had no use for either."""
        with patch.object(embedding_module, "_create_dense_embedder") as create:
            EmbeddingService()
            create.assert_not_called()

    def test_the_embedder_is_built_once_and_reused(self):
        sentinel = Sentinel()
        with patch.object(embedding_module, "_create_dense_embedder", return_value=sentinel) as create:
            service = EmbeddingService()
            service.get_embeddings(["a"])
            service.get_embeddings(["b"])
            service.get_embeddings(["c"])
        self.assertEqual(1, create.call_count)

    def test_warm_up_builds_it_before_any_request(self):
        sentinel = Sentinel()
        with patch.object(embedding_module, "_create_dense_embedder", return_value=sentinel) as create:
            service = EmbeddingService()
            service.warm_up()
            create.assert_called_once()

    def test_warm_up_does_not_take_the_process_down(self):
        """Boot-time warming is an optimisation. A provider that is briefly unreachable
        must not stop the app from starting."""
        with patch.object(embedding_module, "_create_dense_embedder", side_effect=RuntimeError("no")):
            EmbeddingService().warm_up()   # must not raise

    def test_an_empty_list_never_reaches_the_embedder(self):
        with patch.object(embedding_module, "_create_dense_embedder") as create:
            self.assertEqual([], EmbeddingService().get_embeddings([]))
            create.assert_not_called()


class BackendSelectionTests(unittest.TestCase):
    def test_the_default_is_the_local_model(self):
        with patch.dict("os.environ", {"EMBEDDING_BACKEND": ""}, clear=False):
            with patch("langchain_huggingface.HuggingFaceEmbeddings") as local:
                _create_dense_embedder()
                local.assert_called_once()

    def test_remote_is_selected_without_importing_the_local_model(self):
        env = {"EMBEDDING_BACKEND": "openai", "EMBEDDING_BASE_URL": "http://embedder:8080/v1"}
        with patch.dict("os.environ", env, clear=False):
            with patch("langchain_huggingface.HuggingFaceEmbeddings") as local:
                self.assertIsInstance(_create_dense_embedder(), _RemoteEmbedder)
                local.assert_not_called()

    def test_an_unknown_backend_falls_back_to_local_rather_than_failing(self):
        with patch.dict("os.environ", {"EMBEDDING_BACKEND": "typo"}, clear=False):
            with patch("langchain_huggingface.HuggingFaceEmbeddings") as local:
                _create_dense_embedder()
                local.assert_called_once()

    def test_remote_without_a_url_says_so_at_construction(self):
        env = {"EMBEDDING_BACKEND": "openai", "EMBEDDING_BASE_URL": ""}
        with patch.dict("os.environ", env, clear=False):
            with self.assertRaises(ValueError) as caught:
                _create_dense_embedder()
        self.assertIn("EMBEDDING_BASE_URL", str(caught.exception))


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": self._data}


class RemoteEmbedderTests(unittest.TestCase):
    def remote(self, **env):
        settings = {
            "EMBEDDING_BACKEND": "openai",
            "EMBEDDING_BASE_URL": "http://embedder:8080/v1",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
        }
        settings.update(env)
        with patch.dict("os.environ", settings, clear=False):
            return _RemoteEmbedder()

    def test_vectors_are_returned_in_the_order_the_texts_were_given(self):
        """The OpenAI schema does not promise ordered `data`, and an out-of-order batch
        would attach every vector to the wrong chunk — corruption with no symptom until
        someone notices retrieval has quietly stopped working."""
        remote = self.remote()
        # Unit vectors pointing different ways: told apart by DIRECTION, because returned
        # vectors are normalised and would no longer be told apart by length.
        shuffled = [
            {"index": 2, "embedding": [0.0, 0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
            {"index": 1, "embedding": [0.0, 1.0, 0.0]},
        ]
        with patch("requests.Session.post", return_value=FakeResponse(shuffled)):
            self.assertEqual([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                             remote.embed_documents(["a", "b", "c"]))

    def test_a_short_response_raises_rather_than_misaligning(self):
        remote = self.remote()
        with patch("requests.Session.post", return_value=FakeResponse([{"index": 0, "embedding": [1.0]}])):
            with self.assertRaises(ValueError):
                remote.embed_documents(["a", "b"])

    def test_texts_are_sent_in_batches(self):
        remote = self.remote(EMBEDDING_BATCH_SIZE="2")
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append(json["input"])
            return FakeResponse([{"index": i, "embedding": [float(i)]} for i in range(len(json["input"]))])

        with patch("requests.Session.post", side_effect=fake_post):
            vectors = remote.embed_documents(["a", "b", "c", "d", "e"])

        self.assertEqual([["a", "b"], ["c", "d"], ["e"]], posts)
        self.assertEqual(5, len(vectors))

    def test_the_api_key_is_sent_only_when_set(self):
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured.update(headers)
            return FakeResponse([{"index": 0, "embedding": [1.0]}])

        with patch("requests.Session.post", side_effect=fake_post):
            self.remote(EMBEDDING_API_KEY="").embed_documents(["a"])
        self.assertNotIn("Authorization", captured)

        with patch("requests.Session.post", side_effect=fake_post):
            self.remote(EMBEDDING_API_KEY="secret").embed_documents(["a"])
        self.assertEqual("Bearer secret", captured.get("Authorization"))

    def test_it_posts_to_the_embeddings_route_of_the_configured_base(self):
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["model"] = json["model"]
            return FakeResponse([{"index": 0, "embedding": [1.0]}])

        with patch("requests.Session.post", side_effect=fake_post):
            self.remote(EMBEDDING_BASE_URL="http://embedder:8080/v1/").embed_documents(["a"])

        self.assertEqual("http://embedder:8080/v1/embeddings", captured["url"])
        self.assertEqual("BAAI/bge-m3", captured["model"])


if __name__ == "__main__":
    unittest.main()


class RemoteForServingTests(unittest.TestCase):
    """RAG_FIX_PLAN item 39: what a hosted embedder needs that the local model does not."""

    def remote(self, **env):
        settings = {"EMBEDDING_BACKEND": "openai", "EMBEDDING_BASE_URL": "http://e/v1",
                    "EMBEDDING_MODEL": "BAAI/bge-m3"}
        settings.update(env)
        with patch.dict("os.environ", settings, clear=False):
            return _RemoteEmbedder()

    def test_a_provider_vector_comes_back_unit_length(self):
        """The dense lane searches by inner product, which is cosine only for unit
        vectors, and the stored ones are unit length."""
        with patch("requests.Session.post",
                   return_value=FakeResponse([{"index": 0, "embedding": [3.0, 4.0]}])):
            self.assertEqual([[0.6, 0.8]], self.remote().embed_documents(["a"]))

    def test_an_already_unit_vector_is_left_as_it_is(self):
        with patch("requests.Session.post",
                   return_value=FakeResponse([{"index": 0, "embedding": [0.6, 0.8]}])):
            self.assertEqual([[0.6, 0.8]], self.remote().embed_documents(["a"]))

    def test_a_remote_embedder_is_not_coalesced_by_default(self):
        """One batch in flight made every caller wait out another's round trip:
        p50 2,445 ms coalesced vs 631 ms not, at 32 concurrent callers."""
        with patch.dict("os.environ", {"EMBEDDING_COALESCE_MAX_BATCH": ""}, clear=False):
            remote = self.remote()
            self.assertIs(remote, embedding_module._wrap(remote))

    def test_coalescing_a_remote_embedder_is_still_possible_on_request(self):
        with patch.dict("os.environ", {"EMBEDDING_COALESCE_MAX_BATCH": "8"}, clear=False):
            self.assertIsInstance(embedding_module._wrap(self.remote()), CoalescingEmbedder)

    def test_prewarm_opens_the_requested_sockets_concurrently(self):
        import threading
        import time as _time

        active, peak, lock = [0], [0], threading.Lock()

        def slow_post(url, json=None, headers=None, timeout=None):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            _time.sleep(0.05)
            with lock:
                active[0] -= 1
            return FakeResponse([{"index": 0, "embedding": [1.0]}])

        with patch("requests.Session.post", side_effect=slow_post):
            opened = self.remote().prewarm(8)
        self.assertEqual(8, opened)
        self.assertGreater(peak[0], 1, "prewarm made its calls one at a time")

    def test_prewarm_never_opens_more_than_the_pool_holds(self):
        remote = self.remote(EMBEDDING_POOL_SIZE="12")
        with patch("requests.Session.post",
                   return_value=FakeResponse([{"index": 0, "embedding": [1.0]}])) as post:
            self.assertEqual(12, remote.prewarm(500))
        self.assertEqual(12, post.call_count)

    def test_a_failed_prewarm_call_is_counted_not_raised(self):
        with patch("requests.Session.post", side_effect=ConnectionError("provider down")):
            self.assertEqual(0, self.remote().prewarm(4))

    def test_warm_up_opens_the_pool_of_a_remote_embedder(self):
        remote = self.remote()
        with patch.object(remote, "prewarm", return_value=32) as prewarm:
            with patch.object(embedding_module, "_create_dense_embedder", return_value=remote):
                with patch.dict("os.environ", {"EMBEDDING_PREWARM_CONNECTIONS": "",
                                               "EMBEDDING_COALESCE_MAX_BATCH": ""}, clear=False):
                    EmbeddingService().warm_up()
        prewarm.assert_called_once_with(remote.pool_size)

    def test_warm_up_can_be_told_not_to_open_the_pool(self):
        remote = self.remote()
        with patch.object(remote, "prewarm", return_value=0) as prewarm:
            with patch.object(embedding_module, "_create_dense_embedder", return_value=remote):
                with patch.dict("os.environ", {"EMBEDDING_PREWARM_CONNECTIONS": "0"}, clear=False):
                    EmbeddingService().warm_up()
        prewarm.assert_called_once_with(0)

    def test_a_prewarm_that_raises_does_not_take_startup_down(self):
        remote = self.remote()
        with patch.object(remote, "prewarm", side_effect=RuntimeError("no network")):
            with patch.object(embedding_module, "_create_dense_embedder", return_value=remote):
                EmbeddingService().warm_up()  # must not raise


class _FlakyServer:
    """An embeddings endpoint on a real socket that mistreats its first N connections.

    `drop`: close the connection without a word, as a server reaping an idle keep-alive
    socket does. `stall`: read the request and never answer, the other face of the same
    race. Every later connection gets a proper 200.
    """

    def __init__(self, bad=1, mode="drop"):
        import socket
        import threading

        self.bad, self.mode, self.seen = bad, mode, 0
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}/v1"
        self._held = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        import json as _json

        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.seen += 1
            request = self._read_request(conn)
            if self.seen <= self.bad:
                if self.mode == "drop":
                    conn.close()
                else:
                    self._held.append(conn)  # never answered
                continue
            body = _json.dumps({"data": [{"index": 0, "embedding": [0.6, 0.8]}]}).encode()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            conn.close()
            del request

    @staticmethod
    def _read_request(conn):
        """The WHOLE request - headers, then Content-Length bytes of body. Answering after
        one recv() and closing with body still unread makes Windows reset the connection,
        which would fail a good request and make this server lie about what it did."""
        blank_line, crlf = b"\r\n\r\n", b"\r\n"
        data = b""
        while blank_line not in data:
            chunk = conn.recv(65536)
            if not chunk:
                return data
            data += chunk
        head, _, body = data.partition(blank_line)
        length = next((int(line.split(b":", 1)[1]) for line in head.split(crlf)
                       if line.lower().startswith(b"content-length:")), 0)
        while len(body) < length:
            chunk = conn.recv(65536)
            if not chunk:
                break
            body += chunk
        return head + blank_line + body

    def close(self):
        self._sock.close()
        for conn in self._held:
            conn.close()


class ConnectionFailureTests(unittest.TestCase):
    """RAG_FIX_PLAN item 39: one failed connection must not end a turn."""

    def remote(self, url, **env):
        settings = {"EMBEDDING_BACKEND": "openai", "EMBEDDING_BASE_URL": url, "EMBEDDING_MODEL": "m"}
        settings.update(env)
        with patch.dict("os.environ", settings, clear=False):
            return _RemoteEmbedder()

    def test_a_dropped_connection_is_retried_on_a_fresh_one(self):
        server = _FlakyServer(bad=1, mode="drop")
        try:
            self.assertEqual([[0.6, 0.8]], self.remote(server.url).embed_documents(["q"]))
            self.assertEqual(2, server.seen)
        finally:
            server.close()

    def test_a_request_that_is_never_answered_is_retried_after_the_query_timeout(self):
        import time as _time

        server = _FlakyServer(bad=1, mode="stall")
        try:
            remote = self.remote(server.url, EMBEDDING_QUERY_TIMEOUT_SECONDS="0.5")
            started = _time.perf_counter()
            self.assertEqual([[0.6, 0.8]], remote.embed_documents(["q"]))
            self.assertLess(_time.perf_counter() - started, 5.0, "waited out more than the query timeout")
        finally:
            server.close()

    def test_a_server_that_keeps_failing_still_surfaces_an_error(self):
        """Retries are bounded: a dead endpoint is an error, not a hang."""
        import requests

        server = _FlakyServer(bad=10, mode="drop")
        try:
            with self.assertRaises(requests.exceptions.ConnectionError):
                self.remote(server.url).embed_documents(["q"])
            self.assertEqual(3, server.seen)  # the first try and two retries
        finally:
            server.close()

    def test_a_batch_keeps_the_longer_ingest_timeout(self):
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["timeout"] = timeout
            return FakeResponse([{"index": i, "embedding": [1.0]} for i in range(len(json["input"]))])

        remote = self.remote("http://e/v1", EMBEDDING_TIMEOUT_SECONDS="30",
                             EMBEDDING_QUERY_TIMEOUT_SECONDS="10", EMBEDDING_CONNECT_TIMEOUT_SECONDS="5")
        with patch("requests.Session.post", side_effect=fake_post):
            remote.embed_documents(["one question"])
            self.assertEqual((5.0, 10.0), captured["timeout"])
            remote.embed_documents(["chunk a", "chunk b"])
            self.assertEqual((5.0, 30.0), captured["timeout"])
