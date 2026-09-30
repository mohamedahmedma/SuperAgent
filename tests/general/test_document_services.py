"""The document services: the paths no test covered before they left the routes.

The single-file upload job, the delete job, the synchronous upload and delete, saving an
upload, the job lookups, the document list and the pair list had no tests while they lived in
api/routes/documents.py. Moving them behind a service boundary is what makes them cheap to test:
plain fakes, no HTTP, no database. Every message asserted here is the one the admin UI renders,
copied from the route these came out of, so these tests are what shows the move changed nothing.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.application.services.documents import (
    DocumentCatalogue,
    DocumentIngestion,
    DocumentRemoval,
)
from backend.domain.errors import NotFound, OperationFailed


class _Jobs:
    """A job tracker that records every call in order."""

    def __init__(self, jobs=None):
        self.calls = []
        self.jobs = jobs or {}

    def create_job(self, filename, **options):
        self.calls.append(("create", filename, options))
        return {"job_id": "job-1", "filename": filename}

    def update_step(self, job_id, step, percent, status="running", message="", **details):
        self.calls.append(("update", step, percent, status, message))

    def complete_step(self, job_id, step, message=""):
        self.calls.append(("complete_step", step, message))

    def complete_job(self, job_id, message=""):
        self.calls.append(("complete_job", message))

    def fail_job(self, job_id, step, error):
        self.calls.append(("fail_job", step, error))

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def list_jobs(self):
        return list(self.jobs.values())

    def of(self, kind):
        return [call for call in self.calls if call[0] == kind]


class _Loader:
    def __init__(self, docs=(), error=None):
        self.docs = list(docs)
        self.error = error
        self.calls = []

    def load_document(self, file_path, filename, progress=None):
        self.calls.append((file_path, filename))
        if self.error:
            raise self.error
        return list(self.docs)


class _Writer:
    def __init__(self):
        self.written = []

    def write_documents(self, documents, *, progress_callback=None):
        self.written.append(list(documents))
        if progress_callback:
            progress_callback(len(documents), len(documents))


class _Parents:
    def __init__(self):
        self.upserted = []

    def upsert_documents(self, docs):
        self.upserted.append(list(docs))
        return len(docs)

    def documents_by_filename(self, filename):
        return []


class _Remover:
    def __init__(self, count=0, error=None, log=None):
        self.count = count
        self.error = error
        self.calls = []
        self.log = log

    def remove(self, filename, job_manager=None, job_id=None, include_assets=True):
        self.calls.append(filename)
        if self.log is not None:
            self.log.append(("remove", filename))
        if self.error:
            raise self.error
        return self.count


class _Pairs:
    def __init__(self, pairs=(), known=(), detach_error=None):
        self.pairs = list(pairs)
        self.known = set(known)
        self.detached = []
        self.detach_error = detach_error

    def get_pair(self, pair_id):
        return object() if pair_id in self.known else None

    def detach(self, filename):
        if self.detach_error:
            raise self.detach_error
        self.detached.append(filename)

    def list_pairs(self):
        return list(self.pairs)


def _docs(parents=1, leaves=2, figures=0):
    return (
        [{"chunk_level": 1, "text": "p"} for _ in range(parents)]
        + [{"chunk_level": 3, "text": "leaf"} for _ in range(leaves)]
        + [{"chunk_level": 3, "text": "fig", "modality": "figure"} for _ in range(figures)]
    )


class _IngestionCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.upload_dir = Path(directory.name) / "documents"
        self.jobs = _Jobs()
        self.loader = _Loader(_docs())
        self.writer = _Writer()
        self.parents = _Parents()
        self.remover = _Remover()
        self.pairs = _Pairs(known={"pair-1"})
        self.forgotten = []

    def ingestion(self):
        return DocumentIngestion(
            loader=self.loader,
            parent_chunks=self.parents,
            vector_writer=self.writer,
            remover=self.remover,
            pairs=self.pairs,
            jobs=self.jobs,
            upload_dir=self.upload_dir,
            forget_corpus_languages=lambda: self.forgotten.append(True),
        )


class SingleFileJobTests(_IngestionCase):
    def test_a_document_goes_through_every_step_in_order(self):
        self.ingestion().ingest("job-1", "/tmp/fees.pdf", "fees.pdf")

        self.assertEqual(["fees.pdf"], self.remover.calls)
        self.assertEqual([[{"chunk_level": 1, "text": "p"}]], self.parents.upserted)
        self.assertEqual(2, len(self.writer.written[0]))
        steps = [call[1] for call in self.jobs.of("complete_step")]
        self.assertEqual(["upload", "cleanup", "parse", "parent_store", "vector_store"], steps)
        self.assertEqual(
            ("complete_step", "parse", "Parsing complete: 1 parent chunks, 2 leaf chunks"),
            self.jobs.of("complete_step")[2],
        )
        self.assertEqual(
            [("complete_job", "Successfully uploaded and processed fees.pdf")],
            self.jobs.of("complete_job"),
        )
        self.assertEqual([], self.jobs.of("fail_job"))

    def test_figure_leaves_are_named_in_the_parse_message(self):
        self.loader.docs = _docs(parents=1, leaves=1, figures=2)
        self.ingestion().ingest("job-1", "/tmp/fees.pdf", "fees.pdf")
        self.assertEqual(
            "Parsing complete: 1 parent chunks, 3 leaf chunks, 2 from figures",
            self.jobs.of("complete_step")[2][2],
        )

    def test_a_document_with_no_content_fails_the_parse_step(self):
        self.loader.docs = []
        self.ingestion().ingest("job-1", "/tmp/x.pdf", "x.pdf")
        self.assertEqual(
            [("fail_job", "parse", "Document processing failed: could not extract content")],
            self.jobs.of("fail_job"),
        )
        self.assertEqual([], self.writer.written)

    def test_a_document_with_no_leaves_fails_the_parse_step(self):
        self.loader.docs = _docs(parents=2, leaves=0)
        self.ingestion().ingest("job-1", "/tmp/x.pdf", "x.pdf")
        self.assertEqual(
            "Document processing failed: no retrievable leaf chunks were generated",
            self.jobs.of("fail_job")[0][2],
        )

    def test_a_cleanup_failure_is_reported_against_cleanup(self):
        self.remover.error = RuntimeError("milvus down")
        self.ingestion().ingest("job-1", "/tmp/x.pdf", "x.pdf")
        self.assertEqual([("fail_job", "cleanup", "milvus down")], self.jobs.of("fail_job"))
        self.assertEqual([], self.loader.calls)


class AcceptTests(_IngestionCase):
    def test_the_file_is_saved_where_ingestion_will_read_it(self):
        saved = []

        async def save(path):
            saved.append(path)

        accepted = asyncio.run(self.ingestion().accept_upload("fees.pdf", save))

        self.assertEqual([self.upload_dir / "fees.pdf"], saved)
        self.assertTrue(self.upload_dir.is_dir())
        self.assertEqual(("job-1", "fees.pdf"), (accepted.job_id, accepted.filename))
        self.assertEqual(
            ("complete_step", "upload", "File uploaded, waiting for background processing"),
            self.jobs.of("complete_step")[0],
        )

    def test_a_failed_save_fails_the_job_and_the_request(self):
        async def save(path):
            raise OSError("disk full")

        with self.assertRaises(OperationFailed) as caught:
            asyncio.run(self.ingestion().accept_upload("fees.pdf", save))

        self.assertEqual("Failed to save file: disk full", caught.exception.message)
        self.assertEqual(
            [("fail_job", "upload", "Failed to save file: disk full")], self.jobs.of("fail_job")
        )

    def test_a_pair_saves_both_files_under_one_job(self):
        saved = []

        def saver(name):
            async def save(path):
                saved.append(path.name)

            return save

        accepted = asyncio.run(
            self.ingestion().accept_pair(
                [("ar", "fees_ar.pdf", saver("ar")), ("en", "fees_en.pdf", saver("en"))]
            )
        )

        self.assertEqual(["fees_ar.pdf", "fees_en.pdf"], saved)
        self.assertEqual("fees_ar.pdf, fees_en.pdf", accepted.names)
        self.assertEqual("fees_ar.pdf, fees_en.pdf", self.jobs.of("create")[0][1])
        self.assertEqual(["ar", "en"], [side[0] for side in accepted.sides])
        self.assertEqual(
            "2 file(s) uploaded, waiting for background processing",
            self.jobs.of("complete_step")[0][2],
        )

    def test_an_unknown_pair_is_not_found_and_an_empty_one_is_a_new_row(self):
        ingestion = self.ingestion()
        with self.assertRaises(NotFound) as caught:
            ingestion.require_pair("missing")
        self.assertEqual("No document pair missing", caught.exception.message)
        ingestion.require_pair("")
        ingestion.require_pair("pair-1")


class IngestNowTests(_IngestionCase):
    def test_the_old_document_is_removed_before_the_new_one_is_parsed(self):
        log = []
        self.remover.log = log
        original = self.loader.load_document

        def load(file_path, filename, progress=None):
            log.append(("load", filename))
            return original(file_path, filename, progress)

        self.loader.load_document = load

        result = self.ingestion().ingest_now("fees.pdf", b"%PDF")

        self.assertEqual([("remove", "fees.pdf"), ("load", "fees.pdf")], log)
        self.assertEqual(b"%PDF", (self.upload_dir / "fees.pdf").read_bytes())
        self.assertEqual((2, 1), (result.leaf_chunks, result.parent_chunks))

    def test_each_failure_is_reported_in_the_routes_own_words(self):
        cases = [
            (_Loader(error=ValueError("bad zip")), "Document processing failed: bad zip"),
            (_Loader([]), "Document processing failed: could not extract content"),
            (
                _Loader(_docs(leaves=0)),
                "Document processing failed: no retrievable leaf chunks were generated",
            ),
        ]
        for loader, message in cases:
            with self.subTest(message=message):
                self.loader = loader
                with self.assertRaises(OperationFailed) as caught:
                    self.ingestion().ingest_now("x.pdf", b"x")
                self.assertEqual(message, caught.exception.message)

    def test_anything_else_is_a_failed_upload(self):
        self.remover.error = RuntimeError("milvus down")
        with self.assertRaises(OperationFailed) as caught:
            self.ingestion().ingest_now("x.pdf", b"x")
        self.assertEqual("Document upload failed: milvus down", caught.exception.message)


class UploadJobReadTests(_IngestionCase):
    def test_an_expired_job_is_not_found(self):
        with self.assertRaises(NotFound) as caught:
            self.ingestion().upload_job("gone")
        self.assertEqual("Upload job does not exist or has expired", caught.exception.message)

    def test_jobs_are_listed_newest_first(self):
        self.jobs.jobs = {
            "a": {"job_id": "a", "created_at": "2026-09-01"},
            "b": {"job_id": "b", "created_at": "2026-09-03"},
            "c": {"job_id": "c", "created_at": "2026-09-02"},
        }
        self.assertEqual(["b", "c", "a"], [job["job_id"] for job in self.ingestion().upload_jobs()])


class RemovalTests(unittest.TestCase):
    def setUp(self):
        self.jobs = _Jobs()
        self.remover = _Remover(count=7)
        self.pairs = _Pairs()
        self.forgotten = []

    def removal(self):
        return DocumentRemoval(
            remover=self.remover,
            pairs=self.pairs,
            jobs=self.jobs,
            delete_steps=[("prepare", "Preparing deletion")],
            forget_corpus_languages=lambda: self.forgotten.append(True),
        )

    def test_a_delete_job_starts_at_prepare(self):
        job = self.removal().start("fees.pdf")
        self.assertEqual("job-1", job["job_id"])
        _, filename, options = self.jobs.of("create")[0]
        self.assertEqual("fees.pdf", filename)
        self.assertEqual("prepare", options["current_step"])
        self.assertEqual("parent_store", options["completion_step"])
        self.assertEqual(
            ("update", "prepare", 1, "running", "Delete job submitted"), self.jobs.calls[1]
        )

    def test_a_delete_detaches_the_file_and_forgets_the_corpus_languages(self):
        self.removal().remove("job-1", "fees.pdf")
        self.assertEqual(["fees.pdf"], self.pairs.detached)
        self.assertEqual([True], self.forgotten)
        self.assertEqual(
            [("complete_job", "Deleted fees.pdf, 7 vector records removed")],
            self.jobs.of("complete_job"),
        )

    def test_a_failed_delete_is_reported_against_the_step_it_reached(self):
        self.remover.error = RuntimeError("milvus down")
        self.jobs.jobs = {"job-1": {"current_step": "bm25"}}
        self.removal().remove("job-1", "fees.pdf")
        self.assertEqual([("fail_job", "bm25", "milvus down")], self.jobs.of("fail_job"))

    def test_a_failed_detach_does_not_fail_a_completed_delete(self):
        self.pairs.detach_error = RuntimeError("row locked")
        with self.assertLogs("backend.application.services.documents", "ERROR"):
            self.removal().remove("job-1", "fees.pdf")
        self.assertEqual(1, len(self.jobs.of("complete_job")))
        self.assertEqual([], self.jobs.of("fail_job"))

    def test_a_delete_while_waiting_returns_the_count(self):
        self.assertEqual(7, self.removal().remove_now("fees.pdf"))
        self.assertEqual(["fees.pdf"], self.pairs.detached)

    def test_a_failed_delete_while_waiting_says_so(self):
        self.remover.error = RuntimeError("milvus down")
        with self.assertRaises(OperationFailed) as caught:
            self.removal().remove_now("fees.pdf")
        self.assertEqual("Failed to delete document: milvus down", caught.exception.message)

    def test_an_expired_delete_job_is_not_found(self):
        with self.assertRaises(NotFound) as caught:
            self.removal().delete_job("gone")
        self.assertEqual("Delete job does not exist or has expired", caught.exception.message)


class _Index:
    def __init__(self, rows=(), error=None):
        self.rows = list(rows)
        self.error = error

    def init_collection(self):
        if self.error:
            raise self.error

    def query(self, filter_expr="", output_fields=None, limit=10000):
        return list(self.rows)

    def query_all(self, filter_expr="", output_fields=None):
        return list(self.rows)


class CatalogueTests(unittest.TestCase):
    def catalogue(self, index, pairs=None):
        return DocumentCatalogue(
            vectors=index, parent_chunks=_Parents(), pairs=pairs or _Pairs(), assets=None
        )

    def test_documents_are_counted_by_file(self):
        index = _Index(
            [
                {"filename": "a.pdf", "file_type": "pdf"},
                {"filename": "a.pdf", "file_type": "pdf"},
                {"filename": "b.docx", "file_type": "docx"},
            ]
        )
        self.assertEqual(
            [
                {"filename": "a.pdf", "file_type": "pdf", "chunk_count": 2},
                {"filename": "b.docx", "file_type": "docx", "chunk_count": 1},
            ],
            self.catalogue(index).list_documents(),
        )

    def test_an_unreachable_index_is_reported_as_the_list_failing(self):
        with self.assertRaises(OperationFailed) as caught:
            self.catalogue(_Index(error=RuntimeError("milvus down"))).list_documents()
        self.assertEqual("Failed to retrieve document list: milvus down", caught.exception.message)

    def test_pairs_carry_their_counts_and_unpaired_files_are_listed_too(self):
        pair = SimpleNamespace(
            pair_id="p1",
            title="Fees",
            filename_ar="fees_ar.pdf",
            filename_en="fees_en.pdf",
            paired=True,
        )
        index = _Index(
            [{"filename": "fees_ar.pdf"}, {"filename": "fees_en.pdf"}, {"filename": "old.pdf"}]
        )
        rows = self.catalogue(index, _Pairs(pairs=[pair])).list_pairs()

        self.assertEqual(
            ("p1", 1, 1), (rows[0]["pair_id"], rows[0]["chunk_count_ar"], rows[0]["chunk_count_en"])
        )
        self.assertEqual(
            {
                "pair_id": "",
                "title": "old.pdf",
                "filename_ar": "",
                "filename_en": "old.pdf",
                "paired": False,
                "chunk_count_ar": 0,
                "chunk_count_en": 1,
                "unassigned": True,
            },
            rows[1],
        )

    def test_a_pair_list_failure_says_so(self):
        with self.assertRaises(OperationFailed) as caught:
            self.catalogue(_Index(error=RuntimeError("milvus down"))).list_pairs()
        self.assertEqual("Failed to retrieve document pairs: milvus down", caught.exception.message)


if __name__ == "__main__":
    unittest.main()
