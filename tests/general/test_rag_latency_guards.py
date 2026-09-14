import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.composition import Services, set_default_services


REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeEmbeddingService:
    def __init__(self):
        self.calls = 0

    def get_embeddings(self, texts):
        self.calls += 1
        return [[0.1, 0.2]]


class FakeMilvusStore:
    def hybrid_retrieve(self, **kwargs):
        raise RuntimeError("hybrid unavailable")

    def dense_retrieve(self, **kwargs):
        return [{
            "text": "fallback result",
            "filename": "doc.md",
            "page_number": 1,
            "chunk_id": "chunk-1",
            "score": 0.9,
        }]


class FakeParentChunks:
    def get_documents_by_ids(self, chunk_ids):
        return []


def load_utils(env):
    """Re-execute `backend/rag/utils.py` under `env`, with fake retrieval dependencies.

    The module is re-executed rather than imported because it reads its rerank settings
    from the environment AT IMPORT, and these tests are about what those settings do.

    What it needs stubbed has shrunk. The module used to open a Milvus client and bind
    the embedder at import, so both had to be in `sys.modules` before it ran; it now
    resolves them from the process container per call, so they are simply named in a
    `Services`. Only `embed_query` is still a module-level import, and that is what the
    one remaining stub covers.
    """
    embedding_service = FakeEmbeddingService()
    milvus_store = FakeMilvusStore()

    fake_indexing = types.ModuleType("backend.indexing")
    fake_indexing.__path__ = []

    fake_embedding = types.ModuleType("backend.indexing.embedding")
    # Both the domain gate and retrieval ask for the query vector; the real module
    # memoizes so only one forward pass happens. The stub delegates so tests that
    # assert on what was embedded still see the call.
    fake_embedding.embed_query = lambda text: embedding_service.get_embeddings([text])[0]
    fake_embedding.reset_query_vector_cache = lambda: None

    set_default_services(Services(
        milvus=milvus_store,
        embedder=embedding_service,
        parent_chunks=FakeParentChunks(),
    ))

    module_name = f"rag_utils_under_test_{id(embedding_service)}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "backend" / "rag" / "utils.py",
    )
    module = importlib.util.module_from_spec(spec)

    with (
        patch.dict(os.environ, env, clear=False),
        patch.dict(
            sys.modules,
            {
                "backend.indexing": fake_indexing,
                "backend.indexing.embedding": fake_embedding,
            },
        ),
    ):
        spec.loader.exec_module(module)

    return module, embedding_service


class RagLatencyGuardTests(unittest.TestCase):
    def tearDown(self):
        # `load_utils` installs a container of fakes; give the process back its own.
        set_default_services(None)

    def test_placeholder_rerank_settings_are_treated_as_disabled(self):
        utils, _ = load_utils({
            "RERANK_MODEL": "your_rerank_model",
            "RERANK_BINDING_HOST": "https://your-rerank-host",
            "RERANK_API_KEY": "your_rerank_api_key",
            "AUTO_MERGE_ENABLED": "false",
        })

        with patch.object(utils.requests, "post") as post:
            docs, meta = utils._rerank_documents(
                "query",
                [{"text": "doc", "chunk_id": "chunk-1", "score": 0.9}],
                1,
            )

        self.assertFalse(utils.RERANK_ENABLED)
        self.assertFalse(meta["rerank_enabled"])
        self.assertEqual(1, len(docs))
        post.assert_not_called()

    def test_dense_fallback_reuses_the_query_embedding(self):
        utils, embedding_service = load_utils({
            "RERANK_MODEL": "",
            "RERANK_BINDING_HOST": "",
            "RERANK_API_KEY": "",
            "AUTO_MERGE_ENABLED": "false",
        })

        result = utils.retrieve_documents("query", top_k=1)

        self.assertEqual(1, embedding_service.calls)
        self.assertEqual("dense_fallback", result["meta"]["retrieval_mode"])
        self.assertEqual(1, len(result["docs"]))

    def test_rewrite_single_choice_uses_one_model_call(self):
        utils, _ = load_utils({"AUTO_MERGE_ENABLED": "false"})

        class Model:
            def __init__(self, payload):
                self.calls = 0
                self.payload = payload
                self.schema = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, messages):
                self.calls += 1
                return self.schema(**self.payload)

        cases = [
            ({
                "method": "step_back",
                "step_back_question": "What is a more abstract way to ask this?",
                "hyde_document": "",
            }, "step_back", "Step-back question"),
            ({
                "method": "hyde",
                "step_back_question": "",
                "hyde_document": "A possible answer-style document",
            }, "hyde", "Hypothetical answer document"),
        ]
        for payload, expected_method, expected_marker in cases:
            with self.subTest(method=expected_method):
                model = Model(payload)
                utils._get_rewrite_model = lambda: model

                result = utils.rewrite_query_once("specific question")

                self.assertEqual(1, model.calls)
                self.assertEqual(expected_method, result["rewrite_method"])
                self.assertIn(expected_marker, result["rewritten_query"])


if __name__ == "__main__":
    unittest.main()
