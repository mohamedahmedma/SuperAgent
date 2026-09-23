"""Persistence interfaces, declared by the code that uses them.

The arrangement follows `sis/application/ports/repositories.py`, for the reasons written
out there: the services own the shape of the storage they need; a Protocol lets a test
hand a service an in-memory fake that inherits nothing; and nothing here mentions a
session, a query or a transaction. Repositories stage work. Only the unit of work
commits (`backend/application/ports/unit_of_work.py`), so a service that writes several
things either lands all of them or none.

Repositories return the frozen records below, never ORM rows. A row carried out of a
repository is still attached to its session: reading a lazy attribute after the unit of
work has closed raises, and assigning to one is a change nobody will ever commit. A
record is only data.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Named only in annotations. Importing from `backend.indexing` at runtime runs its
    # package `__init__`, which loads the document loader, Milvus and the embedder.
    from backend.indexing.section_summary import SectionRecord


# -- conversations --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredSession:
    """A conversation: the id its messages hang off, and the metadata stored with it."""

    id: int
    session_id: str
    metadata: dict


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: int
    message_type: str
    content: str
    timestamp: datetime
    rag_trace: dict | None
    #: The recording this message was spoken as, when it was one. See `ChatAttachment`.
    attachment_id: str | None = None


@dataclass(frozen=True, slots=True)
class NewMessage:
    message_type: str
    content: str
    timestamp: datetime
    rag_trace: dict | None
    attachment_id: str | None = None


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    metadata: dict
    updated_at: datetime
    message_count: int


class ConversationRepository(Protocol):
    """A user's conversations and the messages in them, addressed by username."""

    def find_session(self, username: str, session_id: str) -> StoredSession | None:
        """The conversation, or None when the user or the conversation does not exist."""
        ...

    def open_session(self, username: str, session_id: str, metadata: dict) -> StoredSession | None:
        """The conversation, created with `metadata` if absent; None only for an unknown user."""
        ...

    def patch_session(
        self, session: StoredSession, *, metadata: dict | None, updated_at: datetime
    ) -> None:
        """Merge `metadata` into the stored metadata, key by key, and move `updated_at`.

        A merge in the database rather than a replacement from memory: two writers holding
        the metadata as it was when their turn began — two overlapping turns on one
        conversation — each land only the keys they changed, and neither puts the other's
        back the way it found them. A key mapped to None is stored as null, which
        is how a pending question is cleared.
        """
        ...

    def add_messages(self, session: StoredSession, messages: Sequence[NewMessage]) -> Sequence[int]:
        """Stage the messages in order and return the ids they were given, in that order."""
        ...

    def messages(self, session: StoredSession) -> Sequence[StoredMessage]:
        """The whole conversation, oldest first."""
        ...

    def latest_messages(
        self, session: StoredSession, *, limit: int, before_id: int | None
    ) -> Sequence[StoredMessage]:
        """Up to `limit` messages older than `before_id` (or the newest), newest first."""
        ...

    def summaries(self, username: str) -> Sequence[SessionSummary]:
        """Every conversation the user has, most recently updated first, in one query."""
        ...

    def delete_session(self, username: str, session_id: str) -> bool:
        """Remove the conversation and its messages. False when there was nothing to remove."""
        ...


# -- attachments ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NewAttachment:
    """A recording to store: everything but the owner, who is named at the call."""

    id: str
    kind: str
    sha256: str
    storage_uri: str
    content_type: str
    byte_size: int
    duration_ms: int
    transcript: str | None
    transcript_status: str


@dataclass(frozen=True, slots=True)
class AttachmentRecord:
    id: str
    kind: str
    sha256: str
    storage_uri: str
    content_type: str
    byte_size: int
    duration_ms: int
    transcript: str | None
    transcript_status: str
    created_at: datetime


class ChatAttachmentRepository(Protocol):
    """Recordings parents sent, addressed by owner and id.

    Every read names the owner, so a URL that carries a note's id resolves only for the
    account that sent it. There is no read by id alone.
    """

    def add(self, username: str, attachment: NewAttachment) -> AttachmentRecord | None:
        """Stage the attachment for `username`; None only for an unknown user."""
        ...

    def get(self, username: str, attachment_id: str) -> AttachmentRecord | None:
        ...

    def get_many(self, username: str, attachment_ids: Sequence[str]) -> Sequence[AttachmentRecord]:
        """The owner's attachments among `attachment_ids`, in no particular order."""
        ...


# -- document pairs -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DocumentPairRecord:
    """One knowledge-base entry and the file on each language's side ("" when empty)."""

    pair_id: str
    title: str
    filename_ar: str = ""
    filename_en: str = ""

    @property
    def paired(self) -> bool:
        # Derived rather than stored: a stored flag is one more thing that can disagree
        # with the two fields beside it.
        return bool(self.filename_ar and self.filename_en)

    @property
    def empty(self) -> bool:
        return not (self.filename_ar or self.filename_en)


class DocumentPairRepository(Protocol):
    def list_all(self) -> Sequence[DocumentPairRecord]:
        """Every entry, newest first."""
        ...

    def get(self, pair_id: str) -> DocumentPairRecord | None:
        ...

    def holding(self, filename: str) -> Sequence[DocumentPairRecord]:
        """The entries naming `filename` on either side, oldest first."""
        ...

    def paired_filenames(self) -> Sequence[tuple[str, str]]:
        """(filename_ar, filename_en) of every entry with a file on both sides, oldest first."""
        ...

    def save(self, pair: DocumentPairRecord) -> None:
        """Insert or update the entry, visibly to later reads in the same transaction."""
        ...

    def delete_empty(self) -> None:
        """Remove every entry with neither side filled."""
        ...


# -- parent chunks --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParentChunkRecord:
    """A level-1 or level-2 chunk: the text a retrieved leaf is merged up into."""

    chunk_id: str
    text: str
    filename: str
    file_type: str = ""
    file_path: str = ""
    page_number: int = 0
    parent_chunk_id: str = ""
    root_chunk_id: str = ""
    chunk_level: int = 0
    chunk_idx: int = 0
    modality: str = "text"
    asset_ids: tuple[str, ...] = ()


class ParentChunkRepository(Protocol):
    def upsert_many(self, chunks: Sequence[ParentChunkRecord]) -> None:
        """Insert each chunk, or replace the stored chunk with the same id."""
        ...

    def get_many(self, chunk_ids: Sequence[str]) -> Sequence[ParentChunkRecord]:
        """The stored chunks among `chunk_ids`, in no particular order."""
        ...

    def delete_by_filename(self, filename: str) -> Sequence[str]:
        """Remove a document's chunks and return the ids that were removed."""
        ...

    def sections(self, level: int) -> Sequence[ParentChunkRecord]:
        """Every chunk at `level`, in file and position order."""
        ...

    def by_filename(self, filename: str) -> Sequence[ParentChunkRecord]:
        """Every parent chunk of one document, in level and position order."""
        ...


# -- the scope catalogue --------------------------------------------------------------


@dataclass
class DigestRecord:
    """The corpus-level description and the floor derived from the same corpus."""

    paragraph: str = ""
    sections_sha256: str = ""
    section_count: int = 0
    floor: float = 0.0
    floor_sha256: str = ""
    question_count: int = 0
    model_used: str = ""


class SectionSummaryRepository(Protocol):
    def for_profile(self, profile: str) -> Sequence[SectionRecord]:
        ...

    def hashes(self, profile: str) -> dict[str, str]:
        """chunk_id -> content hash, for deciding what needs re-summarising."""
        ...

    def upsert_many(self, profile: str, records: Sequence[SectionRecord]) -> int:
        """Insert or update each record under `profile`; returns how many were written."""
        ...

    def delete_except(self, profile: str, live_chunk_ids: Collection[str]) -> int:
        """Remove the profile's entries for sections not in `live_chunk_ids`; returns how many."""
        ...


class CorpusDigestRepository(Protocol):
    def get(self, profile: str) -> DigestRecord | None:
        ...

    def save(self, profile: str, digest: DigestRecord) -> None:
        ...


# -- assets ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredAssetRef:
    """What removing an occurrence leaves the caller to clean up."""

    asset_id: str
    sha256: str
    storage_uri: str


@dataclass(frozen=True, slots=True)
class StoredExtraction:
    """A cached extraction as stored. The payload stays raw JSON: whether an unreadable
    one is skipped or fatal is the asset store's decision, not the repository's."""

    sha256: str
    payload: dict


class DocumentAssetRepository(Protocol):
    """Occurrences of images in documents, one dossier per row."""

    def upsert_many(self, dossiers: Sequence[AssetDossier]) -> None:
        """Insert each dossier, or replace the stored one with the same asset id."""
        ...

    def get(self, asset_id: str) -> AssetDossier | None:
        ...

    def get_many(self, asset_ids: Sequence[str]) -> Sequence[AssetDossier]:
        """The stored dossiers among `asset_ids`, in no particular order."""
        ...

    def list_by_filename(self, filename: str, *, extracted_only: bool) -> Sequence[AssetDossier]:
        """A document's occurrences by page, optionally only those already extracted."""
        ...

    def displayable_hashes(self, filenames: Sequence[str]) -> Sequence[tuple[str, str]]:
        """(filename, sha256) for every occurrence among `filenames` whose bytes are stored."""
        ...

    def delete_by_filename(self, filename: str) -> Sequence[StoredAssetRef]:
        """Remove a document's occurrences and describe what was removed."""
        ...

    def referenced_digests(self, digests: Collection[str]) -> set[str]:
        """The digests among `digests` that some remaining occurrence still uses."""
        ...

    def older_than(self, dossier_version: int, *, after_asset_id: str, limit: int) -> Sequence[AssetDossier]:
        """The next page of occurrences below `dossier_version`, keyset-paginated by id."""
        ...

    def status_counts(self) -> dict[str, int]:
        ...

    def occurrence_count(self) -> int:
        ...

    def distinct_image_count(self) -> int:
        ...


class AssetExtractionRepository(Protocol):
    """The content-addressed extraction cache."""

    def find_many(
        self, digests: Collection[str], profile: str, dossier_version: int
    ) -> Sequence[StoredExtraction]:
        ...

    def save(
        self,
        sha256: str,
        profile: str,
        dossier_version: int,
        payload: dict,
        *,
        model_used: str,
        confidence: float,
        needs_review: bool,
    ) -> None:
        """Insert the extraction, or replace the stored one for the same key."""
        ...

    def count(self) -> int:
        ...


@dataclass(frozen=True, slots=True)
class EntityAttributeRecord:
    """One value of one attribute of one entity asset; a multi-valued attribute has several."""

    asset_id: str
    profile: str
    name: str
    value_key: str
    value_text: str | None = None
    value_number: float | None = None
    value_bool: bool | None = None


class EntityAttributeRepository(Protocol):
    def replace_for_asset(self, asset_id: str, rows: Sequence[EntityAttributeRecord]) -> None:
        """Make `rows` the asset's entire attribute set."""
        ...

    def delete_for_assets(self, asset_ids: Sequence[str]) -> int:
        ...

    def matching_asset_ids(
        self,
        name: str,
        *,
        profile: str | None,
        restrict_to: Collection[str] | None,
        minimum: float | None = None,
        maximum: float | None = None,
        boolean: bool | None = None,
        keys: Collection[str] | None = None,
    ) -> set[str]:
        """Assets with a value of `name` satisfying every criterion given."""
        ...

    def facet_counts(
        self,
        name: str,
        *,
        kind: str,
        profile: str | None,
        restrict_to: Collection[str] | None,
        limit: int,
    ) -> list[tuple[object, int]]:
        """(value, how many assets) for one attribute, most common first.

        `kind` is "number", "boolean" or "text" — which typed column holds the values.
        """
        ...

    def stats(self) -> dict:
        ...


# -- ingest jobs ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobStep:
    key: str
    label: str
    percent: int = 0
    status: str = "pending"
    message: str = ""

    #: A sub-stage of this step, drawn as a nested bar beneath it. Figure extraction is
    #: the first: it is minutes of work INSIDE `parse`, so it cannot be a step of its own
    #: without claiming to finish while its parent is still running, and a percentage
    #: buried in `message` is a number the UI cannot draw. `sub_total` of 0 means there
    #: is no sub-stage and nothing is drawn.
    sub_label: str = ""
    sub_done: int = 0
    sub_total: int = 0


@dataclass(frozen=True, slots=True)
class IngestJobRecord:
    """One upload or delete job and the steps its progress is reported in."""

    job_id: str
    kind: str
    filename: str
    status: str
    current_step: str
    completion_step: str
    message: str
    steps: tuple[JobStep, ...]
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    total_chunks: int = 0
    processed_chunks: int = 0


class IngestJobRepository(Protocol):
    def add(self, job: IngestJobRecord) -> None:
        ...

    def get(self, kind: str, job_id: str, *, for_update: bool = False) -> IngestJobRecord | None:
        """The job, optionally row-locked until the transaction ends."""
        ...

    def save(self, job: IngestJobRecord) -> None:
        """Overwrite the stored job's state with `job`'s."""
        ...

    def recent(self, kind: str, limit: int) -> Sequence[IngestJobRecord]:
        """The newest jobs of `kind`, newest first."""
        ...

    def delete_older_than(self, kind: str, cutoff: datetime) -> int:
        ...
