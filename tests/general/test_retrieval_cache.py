"""RAG_FIX_PLAN item 18: a retrieval is reused until the corpus it searched changes."""
import unittest
import uuid

from backend.rag.retrieval_cache import CorpusVersion, RetrievalCache
from tests.general.test_turn_admission import _CLIENT, requires_redis

RESULT = {
    "docs": [{"text": "Fees are reviewed each year.", "filename": "fees.pdf", "score": 0.031,
              "asset_ids": ["fees::img1"], "page_number": 2}],
    "meta": {"retrieval_mode": "hybrid", "retrieval_top_k": 5, "candidate_k": 20},
}


@requires_redis
class RetrievalCacheTests(unittest.TestCase):
    def setUp(self):
        self.prefix = f"test-rc-{uuid.uuid4().hex[:8]}"
        self.corpus = CorpusVersion(redis=lambda: _CLIENT, key=self.key)

    def tearDown(self):
        for key in _CLIENT.scan_iter(f"{self.prefix}:*"):
            _CLIENT.delete(key)

    def key(self, name):
        return f"{self.prefix}:{name}"

    def cache(self, fingerprint="settings-a"):
        return RetrievalCache(redis=lambda: _CLIENT, key=self.key, corpus=self.corpus,
                              fingerprint=fingerprint, ttl_seconds=60)

    def test_a_stored_result_comes_back_as_it_went_in(self):
        cache = self.cache()
        missed, ticket = cache.lookup("fees", 5, "chunk_level == 3")
        self.assertIsNone(missed)
        cache.store(ticket, RESULT)
        hit, _ = cache.lookup("fees", 5, "chunk_level == 3")
        self.assertEqual(RESULT, hit)

    def test_a_change_to_the_corpus_retires_every_entry(self):
        cache = self.cache()
        _, ticket = cache.lookup("fees", 5, "")
        cache.store(ticket, RESULT)
        self.corpus.bump()
        self.assertIsNone(cache.lookup("fees", 5, "")[0])

    def test_a_write_racing_a_retrieval_leaves_an_entry_that_is_never_served(self):
        """The version is read before the search: a document that lands in between must
        not be missing from a result served afterwards."""
        cache = self.cache()
        _, ticket = cache.lookup("fees", 5, "")
        self.corpus.bump()  # a document arrives while the search runs
        cache.store(ticket, RESULT)
        self.assertIsNone(cache.lookup("fees", 5, "")[0])

    def test_anything_that_changes_the_answer_changes_the_key(self):
        cache = self.cache()
        _, ticket = cache.lookup("fees", 5, "chunk_level == 3")
        cache.store(ticket, RESULT)
        for query, top_k, filter_expr, fingerprint in (
            ("fees?", 5, "chunk_level == 3", "settings-a"),
            ("fees", 8, "chunk_level == 3", "settings-a"),
            ("fees", 5, 'chunk_level == 3 and filename not in ["fees_en.pdf"]', "settings-a"),
            ("fees", 5, "chunk_level == 3", "settings-b"),
        ):
            with self.subTest(query=query, top_k=top_k, filter_expr=filter_expr, fingerprint=fingerprint):
                self.assertIsNone(self.cache(fingerprint).lookup(query, top_k, filter_expr)[0])

    def test_only_a_complete_search_is_kept(self):
        cache = self.cache()
        for meta in ({"retrieval_mode": "failed", "retrieval_error": "embedding_failed"},
                     {"retrieval_mode": "dense_fallback"}):
            with self.subTest(meta=meta):
                _, ticket = cache.lookup("fees", 5, "")
                cache.store(ticket, {"docs": [], "meta": meta})
                self.assertIsNone(cache.lookup("fees", 5, "")[0])

    def test_an_honest_empty_result_is_kept(self):
        cache = self.cache()
        empty = {"docs": [], "meta": {"retrieval_mode": "hybrid", "retrieval_empty": True}}
        _, ticket = cache.lookup("swimming pool?", 5, "")
        cache.store(ticket, empty)
        self.assertEqual(empty, cache.lookup("swimming pool?", 5, "")[0])


@requires_redis
class RetrieveDocumentsTests(unittest.TestCase):
    """`retrieve_documents` itself: a repeated question is not searched twice."""

    def setUp(self):
        from unittest.mock import patch

        import backend.rag.utils as utils

        self.prefix = f"test-rd-{uuid.uuid4().hex[:8]}"
        key = lambda name: f"{self.prefix}:{name}"  # noqa: E731
        self.corpus = CorpusVersion(redis=lambda: _CLIENT, key=key)
        cache = RetrievalCache(redis=lambda: _CLIENT, key=key, corpus=self.corpus,
                               fingerprint="f", ttl_seconds=60)
        self.searches = []

        def search(query, top_k, filter_expr):
            self.searches.append((query, filter_expr))
            return {"docs": [{"text": f"about {query}"}], "meta": {"retrieval_mode": "hybrid"}}

        self._patches = [
            patch.object(utils, "_retrieval_cache", lambda: cache),
            patch.object(utils, "_search", search),
            patch.object(utils, "language_filter_clause",
                         lambda language: f' and filename not in ["{language}"]' if language else ""),
        ]
        for patcher in self._patches:
            patcher.start()
        self.retrieve = utils.retrieve_documents

    def tearDown(self):
        for patcher in self._patches:
            patcher.stop()
        for key in _CLIENT.scan_iter(f"{self.prefix}:*"):
            _CLIENT.delete(key)

    def test_a_repeated_question_is_served_from_the_cache(self):
        first = self.retrieve("school fees", top_k=5)
        second = self.retrieve("school fees", top_k=5)
        self.assertEqual(1, len(self.searches))
        self.assertEqual(first["docs"], second["docs"])
        self.assertEqual("hit", second["meta"]["retrieval_cache"])
        self.assertNotIn("retrieval_cache", first["meta"])

    def test_a_change_to_the_corpus_means_searching_again(self):
        self.retrieve("school fees", top_k=5)
        self.corpus.bump()
        self.retrieve("school fees", top_k=5)
        self.assertEqual(2, len(self.searches))

    def test_the_language_routing_is_part_of_the_question(self):
        self.retrieve("school fees", top_k=5, language="en")
        self.retrieve("school fees", top_k=5, language="ar")
        self.assertEqual(2, len(self.searches))


class CorpusChangeTests(unittest.TestCase):
    """Every write a retrieval could see bumps the corpus version, and nothing else does."""

    def test_the_vector_store_reports_its_writes_and_only_its_writes(self):
        from contextlib import contextmanager
        from unittest.mock import MagicMock, patch

        import backend.indexing.milvus_client as milvus_client

        client = MagicMock()

        @contextmanager
        def session(settings=None):
            yield client

        changes = []
        store = milvus_client.MilvusStore(
            milvus_client.MilvusSettings("h", "1", "c", "http://h:1", 1.0), on_change=lambda: changes.append(1)
        )
        with patch.object(milvus_client, "milvus_client_session", session):
            store.query("id >= 0")
            store.has_collection()
            self.assertEqual([], changes)
            store.insert([{"text": "x"}])
            store.delete('filename == "fees.pdf"')
            store.drop_collection()
        self.assertEqual(3, len(changes))

    def test_the_parent_chunk_store_reports_committed_writes(self):
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        from backend.indexing.parent_chunk_store import ParentChunkStore
        from tests.general.test_asset_delivery import DictCache

        uow = MagicMock()
        uow.parent_chunks.delete_by_filename.return_value = ["c1"]

        @contextmanager
        def unit_of_work():
            yield uow

        changes = []
        store = ParentChunkStore(unit_of_work=unit_of_work, cache=DictCache(),
                                 on_change=lambda: changes.append(1))
        store.upsert_documents([{"chunk_id": "c1", "text": "Fees", "filename": "fees.pdf"}])
        store.delete_by_filename("fees.pdf")
        self.assertEqual(2, len(changes))

    def test_the_composition_root_connects_both_stores_to_the_corpus_version(self):
        from backend.composition import Services
        from tests.general.test_asset_delivery import DictCache

        services = Services(cache=DictCache())
        self.assertEqual(services.corpus_version.bump, services.milvus._on_change)
        self.assertEqual(services.corpus_version.bump, services.parent_chunks._on_change)


class FailOpenTests(unittest.TestCase):
    def test_an_unreachable_redis_is_a_miss_and_nothing_raises(self):
        import redis

        def unreachable():
            raise redis.ConnectionError("down")

        corpus = CorpusVersion(redis=unreachable)
        cache = RetrievalCache(redis=unreachable, corpus=corpus, fingerprint="f", ttl_seconds=60)
        result, ticket = cache.lookup("fees", 5, "")
        self.assertIsNone(result)
        cache.store(ticket, RESULT)
        corpus.bump()

    def test_with_no_ttl_redis_is_never_asked(self):
        def must_not_be_asked():
            raise AssertionError("no round trip with the cache off")

        cache = RetrievalCache(redis=must_not_be_asked, corpus=CorpusVersion(), fingerprint="f", ttl_seconds=0)
        result, ticket = cache.lookup("fees", 5, "")
        cache.store(ticket, RESULT)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
