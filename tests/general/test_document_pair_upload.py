"""The bilingual upload job: what it writes, and — mostly — what it refuses to write.

The ordering property is the point of these tests. A pair is two files ingested as one
unit, so everything that can REJECT the upload has to happen before anything that
writes. Get that backwards and a file dropped in the wrong column leaves the corpus in
a state the form cannot express: one side indexed, the entry unpaired, and the admin
with no obvious way back.
"""
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.chat.language import ARABIC, ENGLISH
from backend.db.models import DocumentPair
from backend.indexing import pair_store

ARABIC_BODY = "الرسوم الدراسية للصف الرابع الابتدائي تشمل الكتب والأنشطة والنقل المدرسي بالكامل"
ENGLISH_BODY = "Tuition fees for grade four include books, activities and school transport in full"


def _chunks(text):
    """One parent and one leaf, the shape load_document returns."""
    return [
        {"text": text, "chunk_level": 1, "chunk_id": "c1"},
        {"text": text, "chunk_level": 3, "chunk_id": "c3"},
    ]


class PairUploadJobTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        DocumentPair.__table__.create(engine)
        patcher = patch.object(
            pair_store, "SessionLocal",
            sessionmaker(bind=engine, autoflush=False, expire_on_commit=False),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        import backend.api.routes.documents as documents

        self.documents = documents
        self.loader = MagicMock()
        self.writer = MagicMock()
        self.parents = MagicMock()
        self.cleanup = MagicMock(return_value=0)
        self.jobs = MagicMock()

        for name, double in (
            ("loader", self.loader),
            ("milvus_writer", self.writer),
            ("parent_chunk_store", self.parents),
            ("delete_document_transactionally", self.cleanup),
            ("upload_job_manager", self.jobs),
        ):
            p = patch.object(documents, name, double)
            p.start()
            self.addCleanup(p.stop)

    def _run(self, sides, pair_id="", title="Fees"):
        self.documents._process_pair_upload_job("job1", pair_id, title, sides)

    def _failure(self):
        self.assertTrue(self.jobs.fail_job.called, "the job was expected to fail")
        return self.jobs.fail_job.call_args[0][2]

    def test_both_sides_are_indexed_and_recorded_as_one_entry(self):
        self.loader.load_document.side_effect = [_chunks(ARABIC_BODY), _chunks(ENGLISH_BODY)]

        self._run([
            (ARABIC, "/tmp/fees_ar.docx", "fees_ar.docx"),
            (ENGLISH, "/tmp/fees_en.docx", "fees_en.docx"),
        ])

        self.jobs.fail_job.assert_not_called()
        self.assertEqual(2, self.writer.write_documents.call_count)

        rows = pair_store.list_pairs()
        self.assertEqual(1, len(rows), "the two files should be ONE entry")
        self.assertEqual("fees_ar.docx", rows[0]["filename_ar"])
        self.assertEqual("fees_en.docx", rows[0]["filename_en"])
        self.assertTrue(rows[0]["paired"])
        self.assertEqual("Fees", rows[0]["title"])

    def test_one_side_alone_is_a_complete_entry(self):
        self.loader.load_document.side_effect = [_chunks(ENGLISH_BODY)]

        self._run([(ENGLISH, "/tmp/bus_en.docx", "bus_en.docx")])

        self.jobs.fail_job.assert_not_called()
        rows = pair_store.list_pairs()
        self.assertEqual("bus_en.docx", rows[0]["filename_en"])
        self.assertEqual("", rows[0]["filename_ar"])
        self.assertFalse(rows[0]["paired"], "one side is not a pair")

    def test_a_file_in_the_wrong_column_is_rejected(self):
        self.loader.load_document.side_effect = [_chunks(ENGLISH_BODY)]

        self._run([(ARABIC, "/tmp/fees.docx", "fees.docx")])

        message = self._failure()
        self.assertIn("fees.docx", message)
        self.assertIn("Arabic", message)
        self.assertIn("English column", message, "the message should say where it belongs")

    def test_a_mismatch_on_the_SECOND_file_indexes_neither(self):
        """The ordering invariant.

        The Arabic half is valid and comes first. The English slot holds another Arabic
        document. Nothing may be written — not the vectors, not the parent chunks, and
        not the pair row — or the admin is left with half an entry.
        """
        self.loader.load_document.side_effect = [_chunks(ARABIC_BODY), _chunks(ARABIC_BODY)]

        self._run([
            (ARABIC, "/tmp/fees_ar.docx", "fees_ar.docx"),
            (ENGLISH, "/tmp/fees_en.docx", "fees_en.docx"),
        ])

        self._failure()
        self.writer.write_documents.assert_not_called()
        self.parents.upsert_documents.assert_not_called()
        self.cleanup.assert_not_called()
        self.assertEqual([], pair_store.list_pairs(), "a rejected upload left a row behind")

    def test_the_second_language_joins_the_existing_entry(self):
        """Uploading the Arabic half months later must fill the SAME row, which is the
        reason pairs are a table rather than a value on a chunk."""
        self.loader.load_document.side_effect = [_chunks(ENGLISH_BODY)]
        self._run([(ENGLISH, "/tmp/fees_en.docx", "fees_en.docx")], title="Fees policy")
        pair_id = pair_store.list_pairs()[0]["pair_id"]

        self.loader.load_document.side_effect = [_chunks(ARABIC_BODY)]
        self._run([(ARABIC, "/tmp/fees_ar.docx", "fees_ar.docx")], pair_id=pair_id, title="Fees policy")

        rows = pair_store.list_pairs()
        self.assertEqual(1, len(rows), "a second entry was created instead of filling the first")
        self.assertTrue(rows[0]["paired"])
        self.assertEqual(["fees_en.docx"], pair_store.superseded_filenames(ARABIC))

    def test_a_document_that_yields_no_leaf_chunks_is_rejected(self):
        self.loader.load_document.side_effect = [
            [{"text": ARABIC_BODY, "chunk_level": 1, "chunk_id": "c1"}]
        ]

        self._run([(ARABIC, "/tmp/fees_ar.docx", "fees_ar.docx")])

        self.assertIn("leaf chunks", self._failure())
        self.assertEqual([], pair_store.list_pairs())

    def test_an_unreadable_file_is_rejected_by_name(self):
        self.loader.load_document.side_effect = [[]]

        self._run([(ARABIC, "/tmp/broken.docx", "broken.docx")])

        self.assertIn("broken.docx", self._failure())
        self.assertEqual([], pair_store.list_pairs())


if __name__ == "__main__":
    unittest.main()
