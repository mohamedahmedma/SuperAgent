"""Integration: the real Redis and the real Milvus collection.

The retrieval tests run against whatever corpus is actually indexed, so they assert
properties that hold for any corpus rather than facts about one document — "a query
returns chunks, scored, in descending order, with the fields retrieval needs" rather
than "question X returns chunk 7". The latter would be a better test and a worse one:
it would break the moment anybody re-indexed.

Nothing here writes to the shared collection.
"""
import threading
import time
import unittest

from backend.indexing.embedding import embed_query, embedding_service
from backend.indexing.milvus_client import MilvusStore
from backend.infra.cache import cache
from tests.general.integration_support import (
    TEST_PREFIX,
    corpus_indexed,
    requires_corpus,
    requires_embedder,
    requires_milvus,
    requires_redis,
)

EN_QUERIES = [
    "when does the second term start",
    "how much are the school fees",
    "what documents are required for admission",
    "is there a bus service",
    "what is the uniform policy",
]
AR_QUERIES = [
    "متى يبدأ الفصل الدراسي الثاني",
    "كم تبلغ الرسوم الدراسية",
    "ما هي المستندات المطلوبة للتقديم",
    "هل توجد خدمة أتوبيس",
    "ما هو نظام الزي المدرسي",
]


@requires_redis
class RedisCacheTests(unittest.TestCase):
    def setUp(self):
        self.keys = []

    def tearDown(self):
        for key in self.keys:
            cache.delete(key)

    def key(self, name):
        full = f"{TEST_PREFIX}:{name}"
        self.keys.append(full)
        return full

    def test_a_value_round_trips(self):
        key = self.key("roundtrip")
        cache.set_json(key, {"chunk": "text", "n": 3}, ttl=30)
        self.assertEqual({"chunk": "text", "n": 3}, cache.get_json(key))

    def test_a_missing_key_reads_as_none(self):
        self.assertIsNone(cache.get_json(self.key("never-written")))

    def test_arabic_survives_the_round_trip(self):
        key = self.key("arabic")
        payload = {"text": "الرسوم الدراسية للصف الخامس", "ok": True}
        cache.set_json(key, payload, ttl=30)
        self.assertEqual(payload, cache.get_json(key))

    def test_a_nested_structure_survives(self):
        key = self.key("nested")
        payload = {"docs": [{"id": i, "meta": {"page": i * 2}} for i in range(20)]}
        cache.set_json(key, payload, ttl=30)
        self.assertEqual(payload, cache.get_json(key))

    def test_a_deleted_key_is_gone(self):
        key = self.key("deleted")
        cache.set_json(key, {"a": 1}, ttl=30)
        cache.delete(key)
        self.assertIsNone(cache.get_json(key))

    def test_a_short_ttl_expires(self):
        key = self.key("expiring")
        cache.set_json(key, {"a": 1}, ttl=1)
        self.assertIsNotNone(cache.get_json(key))
        time.sleep(1.6)
        self.assertIsNone(cache.get_json(key))

    def test_keys_are_namespaced_by_prefix(self):
        """Two profiles sharing one Redis must not read each other's parent chunks."""
        key = self.key("prefixed")
        cache.set_json(key, {"a": 1}, ttl=30)
        client = cache._get_client()
        self.assertTrue(client.exists(f"{cache.key_prefix}:{key}"))

    def test_concurrent_writers_do_not_corrupt_values(self):
        errors = []
        barrier = threading.Barrier(16)

        def write(index):
            key = f"{TEST_PREFIX}:conc-{index}"
            self.keys.append(key)
            try:
                barrier.wait(timeout=30)
                cache.set_json(key, {"index": index}, ttl=30)
                if cache.get_json(key) != {"index": index}:
                    errors.append(index)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual([], errors)

    def test_a_broken_payload_degrades_to_none(self):
        """The cache is an optimisation; a bad value must never raise into a turn."""
        key = self.key("corrupt")
        cache._get_client().setex(f"{cache.key_prefix}:{key}", 30, "not json{{{")
        self.assertIsNone(cache.get_json(key))


@requires_milvus
class MilvusCollectionTests(unittest.TestCase):
    def setUp(self):
        self.store = MilvusStore()

    def test_the_collection_exists(self):
        self.assertTrue(self.store.has_collection())

    @requires_corpus(1)
    def test_the_collection_holds_chunks(self):
        self.assertGreater(corpus_indexed(), 0)

    @requires_corpus(1)
    def test_stored_chunks_carry_the_fields_retrieval_reads(self):
        rows = self.store.query_all(output_fields=["chunk_id", "text", "filename", "file_type"])
        self.assertTrue(rows)
        row = rows[0]
        for field in ("chunk_id", "text", "filename"):
            self.assertIn(field, row)
        self.assertTrue(str(row["text"]).strip(), "an indexed chunk has empty text")

    @requires_corpus(1)
    def test_chunks_can_be_fetched_by_id(self):
        rows = self.store.query_all(output_fields=["chunk_id"])
        wanted = [row["chunk_id"] for row in rows[:3]]
        fetched = self.store.get_chunks_by_ids(wanted)
        self.assertEqual(set(wanted), {row["chunk_id"] for row in fetched})

    @requires_corpus(1)
    def test_an_unknown_id_returns_nothing_rather_than_raising(self):
        self.assertEqual([], self.store.get_chunks_by_ids(["no-such-chunk-id-at-all"]))

    def test_an_empty_id_list_returns_nothing(self):
        self.assertEqual([], self.store.get_chunks_by_ids([]))


@requires_milvus
@requires_embedder
class RetrievalTests(unittest.TestCase):
    """End-to-end retrieval: real embedder, real vector store, real BM25."""

    def setUp(self):
        self.store = MilvusStore()

    @requires_corpus(5)
    def test_dense_retrieval_returns_scored_chunks(self):
        for query in EN_QUERIES[:3]:
            with self.subTest(query=query):
                docs = self.store.dense_retrieve(embed_query(query), top_k=5)
                self.assertTrue(docs, f"no results for {query!r}")
                self.assertLessEqual(len(docs), 5)
                for doc in docs:
                    self.assertIn("text", doc)

    @requires_corpus(5)
    def test_dense_results_are_ordered_best_first(self):
        docs = self.store.dense_retrieve(embed_query(EN_QUERIES[0]), top_k=8)
        scores = [doc.get("score") for doc in docs if doc.get("score") is not None]
        if len(scores) > 1:
            self.assertEqual(sorted(scores, reverse=True), scores)

    @requires_corpus(5)
    def test_hybrid_retrieval_returns_results_for_both_languages(self):
        for query in EN_QUERIES[:3] + AR_QUERIES[:3]:
            with self.subTest(query=query):
                docs = self.store.hybrid_retrieve(embed_query(query), query, top_k=5)
                self.assertTrue(docs, f"hybrid search returned nothing for {query!r}")

    @requires_corpus(5)
    def test_hybrid_never_returns_more_than_top_k(self):
        for k in (1, 3, 8):
            with self.subTest(top_k=k):
                docs = self.store.hybrid_retrieve(embed_query(EN_QUERIES[0]), EN_QUERIES[0], top_k=k)
                self.assertLessEqual(len(docs), k)

    @requires_corpus(5)
    def test_results_are_distinct_chunks(self):
        docs = self.store.hybrid_retrieve(embed_query(EN_QUERIES[0]), EN_QUERIES[0], top_k=8)
        ids = [doc.get("chunk_id") for doc in docs if doc.get("chunk_id")]
        self.assertEqual(len(ids), len(set(ids)), "the same chunk came back twice")

    @requires_corpus(5)
    def test_the_same_query_retrieves_the_same_chunks_twice(self):
        """Retrieval has to be deterministic, or every downstream measurement is noise."""
        query = EN_QUERIES[1]
        first = [d.get("chunk_id") for d in self.store.hybrid_retrieve(embed_query(query), query, top_k=5)]
        second = [d.get("chunk_id") for d in self.store.hybrid_retrieve(embed_query(query), query, top_k=5)]
        self.assertEqual(first, second)

    @requires_corpus(5)
    def test_an_empty_query_does_not_raise(self):
        try:
            self.store.hybrid_retrieve(embed_query("placeholder"), "", top_k=3)
        except Exception as exc:
            self.fail(f"an empty query raised {type(exc).__name__}: {exc}")

    @requires_corpus(5)
    def test_a_nonsense_query_returns_without_error(self):
        docs = self.store.hybrid_retrieve(embed_query("zzzz qqqq xyzzy"), "zzzz qqqq xyzzy", top_k=5)
        self.assertIsInstance(docs, list)

    @requires_corpus(5)
    def test_a_very_long_query_is_handled(self):
        long_query = ("what are the school fees and admission requirements " * 40).strip()
        docs = self.store.hybrid_retrieve(embed_query(long_query), long_query, top_k=3)
        self.assertIsInstance(docs, list)

    @requires_corpus(5)
    def test_injection_shaped_text_is_treated_as_a_query_not_an_expression(self):
        """Milvus filters are expressions; a query must never reach one."""
        for hostile in ('" or chunk_id != "', "'; drop collection; --", "1 == 1"):
            with self.subTest(query=hostile):
                docs = self.store.hybrid_retrieve(embed_query(hostile), hostile, top_k=3)
                self.assertIsInstance(docs, list)
        self.assertTrue(self.store.has_collection(), "the collection did not survive")


@requires_embedder
class EmbedderTests(unittest.TestCase):
    """The real bge-m3, including under the concurrency the coalescer now handles."""

    def test_a_vector_has_the_configured_width(self):
        import os

        expected = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
        self.assertEqual(expected, len(embed_query("how much are the fees")))

    def test_vectors_are_normalised(self):
        for query in EN_QUERIES[:2] + AR_QUERIES[:2]:
            with self.subTest(query=query):
                vector = embed_query(query)
                norm = sum(value * value for value in vector) ** 0.5
                self.assertAlmostEqual(1.0, norm, places=3)

    def test_the_same_text_always_embeds_identically(self):
        self.assertEqual(embed_query("what is the uniform policy"),
                         embed_query("what is the uniform policy"))

    def test_different_texts_embed_differently(self):
        self.assertNotEqual(embed_query("school fees"), embed_query("bus routes"))

    def test_a_returned_vector_cannot_corrupt_the_cache(self):
        first = embed_query("mutation check")
        first[0] = 12345.0
        self.assertNotEqual(12345.0, embed_query("mutation check")[0])

    def test_related_text_scores_above_unrelated_text(self):
        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b))

        query = embed_query("how much are the school fees")
        related = embed_query("what is the tuition cost per year")
        unrelated = embed_query("the weather forecast for tomorrow")
        self.assertGreater(cosine(query, related), cosine(query, unrelated))

    def test_the_same_question_in_arabic_scores_above_an_unrelated_english_one(self):
        """Cross-lingual alignment is the property the whole bilingual gate rests on."""
        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b))

        english = embed_query("when does the second term start")
        arabic_same = embed_query("متى يبدأ الفصل الدراسي الثاني")
        english_other = embed_query("how do I bake a chocolate cake")
        self.assertGreater(cosine(english, arabic_same), cosine(english, english_other))

    def test_batch_and_single_embedding_agree(self):
        """Coalescing turns singles into batches, so the two paths must not diverge."""
        texts = EN_QUERIES[:3]
        batched = embedding_service.get_embeddings(texts)
        for text, vector in zip(texts, batched):
            single = embedding_service.get_embeddings([text])[0]
            for a, b in zip(vector, single):
                self.assertAlmostEqual(a, b, places=5)

    def test_concurrent_callers_each_get_their_own_vector(self):
        """The real model behind the real coalescer, under real threads.

        Asserted by nearest-neighbour rather than by equality. Coalescing puts a query
        into a batch, and a transformer's output is not bit-identical across batch
        shapes — padding to a different longest-sequence changes the last decimals.
        Measured here: a concurrent result matches its own sequential result at cosine
        0.9999999 while the closest *other* query sits near 0.90. So demanding equality
        would fail on float noise, and demanding "closer to itself than to anything
        else" is what actually catches a caller being handed someone else's vector.
        """
        texts = [f"{q} number {i}" for i, q in enumerate(EN_QUERIES + AR_QUERIES)]
        expected = {text: embedding_service.get_embeddings([text])[0] for text in texts}

        results = {}
        barrier = threading.Barrier(len(texts))

        def call(text):
            barrier.wait(timeout=60)
            results[text] = embedding_service.get_embeddings([text])[0]

        threads = [threading.Thread(target=call, args=(t,)) for t in texts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        self.assertEqual(len(texts), len(results), "a caller never returned")

        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b))

        for text in texts:
            own = cosine(results[text], expected[text])
            best_other = max(cosine(results[text], expected[o]) for o in texts if o != text)
            self.assertGreater(
                own, best_other,
                f"{text!r} was handed a vector closer to another query's",
            )
            self.assertGreater(own, 0.999, f"{text!r} drifted further than batching explains")

    def test_empty_input_returns_nothing(self):
        self.assertEqual([], embedding_service.get_embeddings([]))

    def test_whitespace_only_text_still_embeds(self):
        self.assertEqual(len(embed_query("a")), len(embed_query("   ")))

    def test_a_long_document_embeds_without_error(self):
        long_text = "The school covers admissions, fees and transport. " * 200
        vectors = embedding_service.get_embeddings([long_text])
        self.assertEqual(1, len(vectors))


if __name__ == "__main__":
    unittest.main()
