"""Ports for the collaborators a service is handed, as opposed to the repositories under it.

`repositories.py` describes persistence. This file describes the next layer up: the stores and
engines the backend already has - conversation storage, voice notes, the vector index, the
document loader, the job trackers - as seen by the services that use them. Each Protocol names
only the methods a service actually calls, so a test's fake is as small as the use it stands
in for.

They exist so a service never imports the concrete class. Those classes reach, through their
own imports, the database, Redis, Milvus and the graph framework, and `.importlinter` refuses
the application layer every one of those - directly and through any chain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from backend.application.ports.repositories import AttachmentRecord, DocumentPairRecord

if TYPE_CHECKING:
    from backend.assets.dossier import AssetDossier

#: A collaborator handed over unbuilt, for the service to build where it first uses it.
#:
#: A service is built while FastAPI resolves a route's dependencies: before the route body
#: runs, and outside every `operation(...)`. A collaborator built there turns a failure to
#: build it - a malformed `MILVUS_TIMEOUT`, a tokenizer that has to be downloaded - into a
#: plain-text 500 from every route of that service, including the ones that never touch it:
#: a job poll, an upload refused for its file type. Provided instead, each one is built
#: inside the operation that uses it, and fails as that operation, with its message - which
#: is where the routes built it when they did the work themselves.
type Provider[T] = Callable[[], T]

# -- conversations ------------------------------------------------------------------


class ConversationStore(Protocol):
    """A user's stored conversations. Implemented by `agent.chat.storage.ConversationStorage`."""

    def list_session_infos(self, user_id: str) -> list[dict]: ...

    def get_session_page(
        self,
        user_id: str,
        session_id: str,
        limit: int | None = None,
        before_id: int | None = None,
    ) -> dict: ...

    def delete_session(self, user_id: str, session_id: str) -> bool: ...


class AttachmentLookup(Protocol):
    """A user's voice notes, looked up by id. Implemented by `agent.chat.attachments.ChatAttachments`."""

    def get_many(
        self, username: str, attachment_ids: Sequence[str]
    ) -> dict[str, AttachmentRecord]: ...


#: Stored messages in, the same messages with their asset ids resolved to displayable
#: references out. `agent.chat.assets_bridge.restore_session_assets`.
AssetRestorer = Callable[[list[dict]], list[dict]]


# -- the corpus ---------------------------------------------------------------------


class VectorIndex(Protocol):
    """The leaf chunks as retrieval holds them. `indexing.milvus_client.MilvusStore`."""

    def init_collection(self) -> None: ...

    def query(
        self, filter_expr: str = "", output_fields: list[str] | None = None, limit: int = 10000
    ) -> list: ...

    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list: ...


class VectorWriter(Protocol):
    """Embeds leaf chunks and writes them to the index. `indexing.milvus_writer.MilvusWriter`."""

    def write_documents(
        self,
        documents: list[dict],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Any: ...


class ParentChunks(Protocol):
    """The level-1 and level-2 chunks, kept in Postgres. `indexing.parent_chunk_store.ParentChunkStore`."""

    def upsert_documents(self, docs: list[dict]) -> int: ...

    def documents_by_filename(self, filename: str) -> list[dict]: ...


class DocumentParser(Protocol):
    """A file on disk in, its three-level chunks out. `indexing.document_loader.DocumentLoader`."""

    def load_document(self, file_path: str, filename: str, progress: Any = None) -> list[dict]: ...


class DocumentEraser(Protocol):
    """Takes a document out of every store it was written to. `indexing.removal.DocumentRemover`."""

    def remove(
        self,
        filename: str,
        job_manager: Any = None,
        job_id: Any = None,
        include_assets: bool = True,
    ) -> int: ...


class DocumentPairs(Protocol):
    """The bilingual rows of the corpus. `indexing.pair_store.DocumentPairService`."""

    def get_pair(self, pair_id: str) -> DocumentPairRecord | None: ...

    def attach(
        self, pair_id: str, language: str, filename: str, title: str = ""
    ) -> DocumentPairRecord: ...

    def detach(self, filename: str) -> Any: ...

    def list_pairs(self) -> list[DocumentPairRecord]: ...


class AssetReview(Protocol):
    """Extracted images and their review state. `assets.store.AssetStore`."""

    def list_by_filename(self, filename: str) -> list[AssetDossier]: ...

    def mark_reviewed(self, asset_id: str) -> AssetDossier | None: ...


class JobTracker(Protocol):
    """Progress of an upload or a delete, polled by the admin UI. `jobs.upload_jobs.IngestJobTracker`."""

    def create_job(
        self,
        filename: str,
        *,
        steps: list[tuple[str, str]] | None = None,
        current_step: str = "upload",
        message: str = "Waiting to upload",
        completion_step: str = "vector_store",
    ) -> dict: ...

    def update_step(
        self,
        job_id: str,
        step_key: str,
        percent: int,
        status: str = "running",
        message: str = "",
        **details: Any,
    ) -> Any: ...

    def complete_step(self, job_id: str, step_key: str, message: str = "") -> Any: ...

    def complete_job(self, job_id: str, message: str = "") -> Any: ...

    def fail_job(self, job_id: str, step_key: str, error: str) -> Any: ...

    def get_job(self, job_id: str) -> dict | None: ...

    def list_jobs(self) -> list[dict]: ...


__all__ = [
    "AssetRestorer",
    "AssetReview",
    "AttachmentLookup",
    "ConversationStore",
    "DocumentEraser",
    "DocumentPairs",
    "DocumentParser",
    "JobTracker",
    "ParentChunks",
    "Provider",
    "VectorIndex",
    "VectorWriter",
]
