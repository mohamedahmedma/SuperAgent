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
from backend.indexing.embedding import EmbeddingService, _RemoteEmbedder, _create_dense_embedder


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
        shuffled = [
            {"index": 2, "embedding": [3.0]},
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0]},
        ]
        with patch("requests.Session.post", return_value=FakeResponse(shuffled)):
            self.assertEqual([[1.0], [2.0], [3.0]], remote.embed_documents(["a", "b", "c"]))

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
