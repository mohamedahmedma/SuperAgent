"""The scope catalogue and the parent chunks it is built from, against real Postgres.

`test_integration_database.py` runs the catalogue against a live estate and skips when
there is none, which is every CI run. These cover the same stores in a throwaway schema,
so the repository behind them is exercised wherever the suite runs.
"""
import unittest

from backend.application.ports.repositories import DigestRecord
from backend.db.models import CorpusDigest, ParentChunk, SectionSummary
from backend.indexing.parent_chunk_store import ParentChunkStore
from backend.indexing.section_summary import SectionRecord
from backend.indexing.summary_store import SectionCatalogueStore
from tests.general.postgres_support import postgres_schema


def record(chunk_id, answers=("what is it?",), vectors=None, sha="h1", summary="A section."):
    return SectionRecord(
        chunk_id=chunk_id,
        content_sha256=sha,
        filename="doc.docx",
        chunk_level=1,
        summary=summary,
        answers=list(answers),
        topics=["fees"],
        question_vectors=list(vectors or []),
        embedding_model="BAAI/bge-m3",
        model_used="test",
    )


def unavailable():
    raise RuntimeError("database down")


class SectionCatalogueTests(unittest.TestCase):
    def setUp(self):
        schema = postgres_schema(self, SectionSummary, CorpusDigest)
        self.catalogue = SectionCatalogueStore(unit_of_work=schema.unit_of_work)

    def test_a_record_round_trips_with_its_vectors_exact(self):
        precise = [0.1234567890123456, -0.9876543210987654, 1e-8]
        self.catalogue.save_records("school", [record("s1", ("q one?", "q two?"), [precise, [0.5]])])

        [loaded] = self.catalogue.load_records("school")
        self.assertEqual(["q one?", "q two?"], loaded.answers)
        self.assertEqual([precise, [0.5]], loaded.question_vectors)

    def test_saving_a_section_again_updates_it(self):
        self.catalogue.save_records("school", [record("s1", ("first?",))])
        self.catalogue.save_records("school", [record("s1", ("second?",), sha="h2")])

        [loaded] = self.catalogue.load_records("school")
        self.assertEqual((["second?"], "h2"), (loaded.answers, loaded.content_sha256))

    def test_profiles_do_not_see_each_others_sections(self):
        self.catalogue.save_records("first", [record("shared", ("from first?",))])
        self.catalogue.save_records("second", [record("shared", ("from second?",))])

        self.assertEqual(["from first?"], self.catalogue.load_records("first")[0].answers)
        self.assertEqual(["from second?"], self.catalogue.load_records("second")[0].answers)

    def test_hashes_and_deleting_what_the_corpus_no_longer_has(self):
        self.catalogue.save_records("school", [record("s1", sha="a"), record("s2", sha="b"), record("s3", sha="c")])
        self.catalogue.save_records("other", [record("s2")])

        self.assertEqual({"s1": "a", "s2": "b", "s3": "c"}, self.catalogue.existing_hashes("school"))
        self.assertEqual(1, self.catalogue.delete_missing("school", ["s1", "s3"]))
        self.assertEqual({"s1", "s3"}, {r.chunk_id for r in self.catalogue.load_records("school")})
        self.assertEqual(1, len(self.catalogue.load_records("other")), "another profile lost a section")

    def test_an_empty_save_writes_nothing(self):
        self.assertEqual(0, self.catalogue.save_records("school", []))
        self.assertEqual([], self.catalogue.load_records("school"))

    def test_a_digest_round_trips_and_a_second_save_updates_it(self):
        self.assertTrue(self.catalogue.save_digest("school", DigestRecord(paragraph="first", floor=0.1)))
        self.assertTrue(self.catalogue.save_digest("school", DigestRecord(
            paragraph="second", sections_sha256="abc", section_count=3, floor=0.5770054,
        )))

        digest = self.catalogue.load_digest("school")
        self.assertEqual(("second", "abc", 3), (digest.paragraph, digest.sections_sha256, digest.section_count))
        self.assertAlmostEqual(0.5770054, digest.floor, places=6)

    def test_an_absent_digest_reads_as_empty(self):
        self.assertEqual(DigestRecord(), self.catalogue.load_digest("school"))

    def test_an_unreachable_database_degrades_reads_rather_than_raising(self):
        """These feed the scope gate, and a database hiccup must never fail the request."""
        broken = SectionCatalogueStore(unit_of_work=unavailable)
        self.assertEqual([], broken.load_records("school"))
        self.assertEqual({}, broken.existing_hashes("school"))
        self.assertEqual(DigestRecord(), broken.load_digest("school"))
        self.assertFalse(broken.save_digest("school", DigestRecord(paragraph="p")))


class FakeCache:
    def __init__(self):
        self.store = {}

    def get_json(self, key):
        return self.store.get(key)

    def set_json(self, key, value, ttl=None):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)


class ParentChunkStoreTests(unittest.TestCase):
    def setUp(self):
        self.cache = FakeCache()
        self.store = ParentChunkStore(
            unit_of_work=postgres_schema(self, ParentChunk).unit_of_work, cache=self.cache
        )

    @staticmethod
    def chunk(chunk_id, *, filename="fees.pdf", idx=0, level=1, text="Fees.", **extra):
        return {"chunk_id": chunk_id, "filename": filename, "chunk_idx": idx, "chunk_level": level,
                "text": text, **extra}

    def test_writing_a_chunk_again_updates_it(self):
        self.store.upsert_documents([self.chunk("c1", text="old")])
        self.store.upsert_documents([self.chunk("c1", text="new", modality="figure", asset_ids=["a1"])])
        self.cache.store.clear()

        [doc] = self.store.get_documents_by_ids(["c1"])
        self.assertEqual(("new", "figure", ["a1"]), (doc["text"], doc["modality"], doc["asset_ids"]))

    def test_a_repeated_id_in_one_write_keeps_the_last(self):
        self.store.upsert_documents([self.chunk("c1", text="first"), self.chunk("c1", text="last")])
        self.cache.store.clear()
        self.assertEqual("last", self.store.get_documents_by_ids(["c1"])[0]["text"])

    def test_reads_keep_the_requested_order_and_skip_unknown_ids(self):
        self.store.upsert_documents([self.chunk(f"c{i}", idx=i) for i in (1, 2, 3)])
        self.cache.store.clear()

        docs = self.store.get_documents_by_ids(["c3", "missing", "c1"])
        self.assertEqual(["c3", "c1"], [doc["chunk_id"] for doc in docs])

    def test_a_warm_read_is_served_from_the_cache(self):
        self.store.upsert_documents([self.chunk("c1")])
        self.cache.store.clear()
        self.store.get_documents_by_ids(["c1"])

        warm = ParentChunkStore(unit_of_work=unavailable, cache=self.cache)
        self.assertEqual(["c1"], [doc["chunk_id"] for doc in warm.get_documents_by_ids(["c1"])])

    def test_deleting_a_document_removes_its_chunks_and_their_cache_entries(self):
        self.store.upsert_documents([
            self.chunk("c1"), self.chunk("c2", idx=1), self.chunk("c3", filename="bus.pdf"),
        ])

        self.assertEqual(2, self.store.delete_by_filename("fees.pdf"))
        self.assertNotIn("parent_chunk:c1", self.cache.store)
        self.cache.store.clear()
        self.assertEqual(["c3"], [d["chunk_id"] for d in self.store.get_documents_by_ids(["c1", "c2", "c3"])])

    def test_sections_are_one_level_in_file_and_position_order(self):
        self.store.upsert_documents([
            self.chunk("b1", filename="b.pdf", idx=1),
            self.chunk("a2", filename="a.pdf", idx=2),
            self.chunk("a1", filename="a.pdf", idx=1),
            self.chunk("a0", filename="a.pdf", idx=0, level=2),
        ])
        self.assertEqual(["a1", "a2", "b1"], [chunk.chunk_id for chunk in self.store.sections(1)])

    def test_a_large_document_is_written_across_batches(self):
        self.assertEqual(1201, self.store.upsert_documents([self.chunk(f"c{i}", idx=i) for i in range(1201)]))
        self.cache.store.clear()
        ids = [f"c{i}" for i in range(1201)]
        self.assertEqual(1201, len(self.store.get_documents_by_ids(ids)))


if __name__ == "__main__":
    unittest.main()
