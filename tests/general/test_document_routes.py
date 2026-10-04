"""The document routes: what only the HTTP layer decides, and what the admin UI is told.

The services have their own tests (test_document_services.py). These pin the rest: the order
a request's checks run in, the body a domain error becomes in the application `create_app()`
really builds, that a store which cannot be built does not take down the routes that never
use it, and the one thing the synchronous upload route owns - reading the upload before
anything is replaced.
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api import resources
from backend.api.errors import status_for
from backend.api.routes.documents import _pair_uploads, mark_asset_reviewed, upload_document
from backend.api.validation import require_supported_document
from backend.app import create_app
from backend.application.services import DocumentCatalogue, DocumentIngestion
from backend.composition import Services, set_default_services
from backend.domain.errors import FeatureDisabled, InvalidInput, NotFound, OperationFailed
from backend.infra.auth import require_admin
from backend.profiles import get_profile


def _never(store):
    def provide():
        raise AssertionError(f"this request must not build {store}")

    return provide


class _Jobs:
    """A job tracker that keeps its jobs in memory and records how each one failed."""

    def __init__(self, jobs=None):
        self.jobs = jobs or {}
        self.failures = []

    def create_job(self, filename, **options):
        self.jobs["job-1"] = {"job_id": "job-1", "filename": filename}
        return self.jobs["job-1"]

    def update_step(self, job_id, step, percent, status="running", message="", **details):
        pass

    def complete_step(self, job_id, step, message=""):
        pass

    def complete_job(self, job_id, message=""):
        pass

    def fail_job(self, job_id, step, error):
        self.failures.append((step, error))

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def list_jobs(self):
        return list(self.jobs.values())


class _Upload:
    """An `UploadFile` as the route reads it: a filename and an awaitable body."""

    def __init__(self, filename, content=b"", error=None):
        self.filename = filename
        self._content = content
        self._error = error

    async def read(self):
        if self._error:
            raise self._error
        return self._content


def _ingestion(upload_dir=Path("."), **given):
    stores = {
        name: _never(f"the {name} store")
        for name in ("loader", "parent_chunks", "vector_writer", "remover", "pairs", "jobs")
    }
    stores.update(given)
    return DocumentIngestion(**stores, upload_dir=upload_dir, forget_corpus_languages=lambda: None)


class ThroughTheApplicationTests(unittest.TestCase):
    """The real application, over a container whose vector index cannot be built.

    `MILVUS_TIMEOUT=30s` is an operator's typo the process boots with: nothing at startup
    builds the index, so requests reach the routes. Each route must answer as it did when it
    built its own stores - the job polls and request validation untouched, and the document
    list failing as JSON in its own words rather than as a plain-text 500.
    """

    def setUp(self):
        environment = patch.dict(os.environ, {"MILVUS_TIMEOUT": "30s"})
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(set_default_services, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.upload_dir = Path(directory.name)

        self.upload_jobs = _Jobs()
        services = Services(upload_jobs=self.upload_jobs, delete_jobs=_Jobs())
        app = create_app(services)
        app.dependency_overrides[require_admin] = lambda: SimpleNamespace(username="admin")
        self.client = TestClient(app)

    def test_a_domain_error_is_answered_as_json_by_the_real_application(self):
        """And not logged: the access log already has every 4xx, and the routes this
        translation replaced never wrote one."""
        with self.assertNoLogs("backend.api.errors"):
            response = self.client.get("/documents/upload/jobs/gone")
        self.assertEqual(404, response.status_code)
        self.assertEqual("application/json", response.headers["content-type"])
        self.assertEqual({"detail": "Upload job does not exist or has expired"}, response.json())

    def test_the_job_polls_answer_while_the_index_cannot_be_built(self):
        self.assertEqual([], self.client.get("/documents/upload/jobs").json())
        response = self.client.get("/documents/delete/jobs/gone")
        self.assertEqual(404, response.status_code)
        self.assertEqual({"detail": "Delete job does not exist or has expired"}, response.json())

    def test_an_unsupported_upload_is_refused_before_any_store_is_built(self):
        with self.assertNoLogs("backend.api.errors"):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("virus.exe", b"MZ", "application/octet-stream")},
            )
        self.assertEqual(400, response.status_code)
        self.assertEqual({"detail": get_profile().user_copy.unsupported_file_type}, response.json())

    def test_an_async_upload_is_accepted_and_its_job_fails_at_cleanup_in_its_own_words(self):
        """As the route always answered: the file saved and the job created in the request,
        then the background ingest failing at `cleanup` - the first step that needs the
        index - with the error the admin UI shows on the job."""
        with patch.object(resources, "UPLOAD_DIR", self.upload_dir):
            response = self.client.post(
                "/documents/upload/async",
                files={"file": ("fees.pdf", b"%PDF", "application/pdf")},
            )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("job-1", response.json()["job_id"])
        self.assertEqual(b"%PDF", (self.upload_dir / "fees.pdf").read_bytes())
        self.assertEqual(
            [("cleanup", "could not convert string to float: '30s'")], self.upload_jobs.failures
        )

    def test_an_index_that_cannot_be_built_fails_the_list_in_its_own_words(self):
        with self.assertLogs("backend.api.errors", "ERROR"):
            response = self.client.get("/documents")
        self.assertEqual(500, response.status_code)
        self.assertEqual("application/json", response.headers["content-type"])
        self.assertEqual(
            "Failed to retrieve document list: could not convert string to float: '30s'",
            response.json()["detail"],
        )


class SynchronousUploadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.upload_dir = Path(directory.name)

    def test_an_unreadable_body_replaces_nothing(self):
        """The body is read before the old document is removed. It used to be read after,
        so a request whose body could not be read deleted the document it meant to replace.
        Every store here refuses to be built: none may be touched."""
        upload = _Upload("fees.pdf", error=OSError("connection reset"))
        with self.assertRaises(OperationFailed) as caught:
            asyncio.run(upload_document(file=upload, _=None, ingestion=_ingestion()))
        self.assertEqual("Document upload failed: connection reset", caught.exception.message)
        self.assertEqual(500, status_for(caught.exception))

    def test_a_processed_upload_says_what_it_stored(self):
        removed, parents, written = [], [], []
        docs = [{"chunk_level": 1}, {"chunk_level": 3}, {"chunk_level": 3}]
        ingestion = _ingestion(
            self.upload_dir,
            remover=lambda: SimpleNamespace(remove=removed.append),
            loader=lambda: SimpleNamespace(load_document=lambda path, name: list(docs)),
            parent_chunks=lambda: SimpleNamespace(upsert_documents=parents.append),
            vector_writer=lambda: SimpleNamespace(write_documents=written.append),
        )
        response = asyncio.run(
            upload_document(file=_Upload("fees.pdf", b"%PDF"), _=None, ingestion=ingestion)
        )
        self.assertEqual(["fees.pdf"], removed)
        self.assertEqual([[{"chunk_level": 1}]], parents)
        self.assertEqual([[{"chunk_level": 3}, {"chunk_level": 3}]], written)
        self.assertEqual(2, response.chunks_processed)
        self.assertEqual(
            "Successfully uploaded and processed fees.pdf: 2 leaf chunks, "
            "1 parent chunks (stored in PostgreSQL)",
            response.message,
        )


class PairRequestOrderTests(unittest.TestCase):
    """A pair request wrong in two ways is told the first, in the order the route always
    checked: no file, an unknown row, an unsupported type, the same file twice."""

    def check(self, file_ar, file_en, pair_id=""):
        ingestion = _ingestion(pairs=lambda: SimpleNamespace(get_pair={"pair-1": True}.get))
        return _pair_uploads(file_ar, file_en, pair_id, ingestion)

    def refused(self, error, message, *request):
        with self.assertRaises(error) as caught:
            self.check(*request)
        self.assertEqual(message, caught.exception.message)

    def test_no_file_is_told_before_an_unknown_row(self):
        self.refused(
            InvalidInput,
            "Upload an Arabic file, an English file, or both",
            None,
            _Upload(""),
            "ghost",
        )

    def test_an_unknown_row_is_told_before_an_unsupported_file(self):
        self.refused(NotFound, "No document pair ghost", _Upload("a.exe"), None, "ghost")

    def test_an_unsupported_file_is_told_before_the_same_file_twice(self):
        self.refused(
            InvalidInput,
            get_profile().user_copy.unsupported_file_type,
            _Upload("a.exe"),
            _Upload("a.exe"),
        )

    def test_the_same_file_cannot_be_both_halves(self):
        self.refused(
            InvalidInput,
            "The Arabic and English files must be different documents",
            _Upload("fees.pdf"),
            _Upload("fees.pdf"),
            "pair-1",
        )

    def test_a_valid_request_names_its_sides_in_order(self):
        arabic, english = _Upload("fees_ar.pdf"), _Upload("fees_en.pdf")
        self.assertEqual([("ar", arabic), ("en", english)], self.check(arabic, english, "pair-1"))
        self.assertEqual([("en", english)], self.check(_Upload(""), english))


class RequestValidationTests(unittest.TestCase):
    def test_a_document_name_is_checked_against_the_profile(self):
        with self.assertRaises(InvalidInput) as caught:
            require_supported_document("")
        self.assertEqual("Filename cannot be empty", caught.exception.message)
        with self.assertRaises(InvalidInput) as caught:
            require_supported_document("virus.exe")
        self.assertEqual(get_profile().user_copy.unsupported_file_type, caught.exception.message)
        self.assertEqual("fees.pdf", require_supported_document("fees.pdf"))

    def test_reviewing_an_asset_with_assets_off_is_a_404_that_touches_no_store(self):
        from backend.profiles.registry import load_profile, set_profile

        profile = load_profile("base").model_copy(deep=True)
        profile.assets.enabled = False
        set_profile(profile)
        self.addCleanup(set_profile, None)

        catalogue = DocumentCatalogue(
            vectors=_never("the index"),
            parent_chunks=_never("the parent chunks"),
            pairs=_never("the pair store"),
            assets=_never("the asset store"),
        )
        with self.assertRaises(FeatureDisabled) as caught:
            asyncio.run(mark_asset_reviewed("a1", _=None, catalogue=catalogue))
        self.assertEqual(404, status_for(caught.exception))
        self.assertEqual("Asset support is disabled for this deployment.", caught.exception.message)


if __name__ == "__main__":
    unittest.main()
