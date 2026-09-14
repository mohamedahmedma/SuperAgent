"""The composition root: the one place that knows how this backend is assembled.

Steps 1-3 put persistence behind repositories and a unit of work, but every service was
still reached through a module-level instance built at import — `storage`,
`document_pairs`, `section_catalogue`, `upload_job_manager`, `get_asset_store()`, the
objects in `api/resources.py`. That has three costs, and they are the reason this file
exists:

  * **Import order becomes behaviour.** An instance built at import captures whatever
    configuration existed at the moment something first imported its module. `identity`
    and `sis` both hit this and both solved it the same way (see
    `identity/infrastructure/db/session.py`): build on first use, never at import.
  * **A caller cannot be given a different one.** Swapping the store under a test, or
    pointing one at another schema, meant reaching in and rebinding a module global —
    which is shared process-wide, so it leaks between tests.
  * **Nothing states what the application is made of.** The dependency graph was spread
    over a dozen modules' import lines. It is now this file, read top to bottom.

## What this is, and what it is not

`Services` is a container of long-lived collaborators. It holds no request state, so one
instance serves every request in the process; `create_app()` builds it and stores it on
`app.state.services`, and routes receive it through `backend/api/deps.py`.

Everything is built **lazily and once**. Laziness is not a micro-optimisation here: the
asset store, the Milvus client and the embedder each open a connection or load a model on
construction, and most processes that import this package (the migration CLI, a test, the
scope-index builder) never touch them. Building the container must stay free.

`default_services()` exists for entry points that have no injection seam yet — the
background CLI jobs, and a chat turn served without one. It is a lazily built process
default, NOT the import-time singleton it replaces: nothing is constructed until
something asks, and a caller with its own `Services` never consults it.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from backend.application.ports import UnitOfWorkFactory
    from backend.assets.blobs import BlobStore
    from backend.assets.delivery import AssetPresenter
    from backend.assets.entity_store import EntityAttributeIndex
    from backend.assets.pipeline import FigurePipeline
    from backend.assets.store import AssetStore
    from backend.chat.background import BackgroundJobs
    from backend.chat.storage import ConversationStorage
    from backend.indexing.document_loader import DocumentLoader
    from backend.indexing.embedding import EmbeddingService
    from backend.indexing.milvus_client import MilvusStore
    from backend.indexing.milvus_writer import MilvusWriter
    from backend.indexing.pair_store import DocumentPairService
    from backend.indexing.parent_chunk_store import ParentChunkStore
    from backend.indexing.removal import DocumentRemover
    from backend.indexing.summary_store import SectionCatalogueStore
    from backend.infra.cache import RedisCache
    from backend.jobs.upload_jobs import IngestJobTracker
    from backend.llm_models import ChatModelFactory
    from backend.rag.entity_retrieval import EntityRetriever

T = TypeVar("T")


class Services:
    """The backend's long-lived services, built on first use and shared thereafter.

    Each property names one collaborator and how it is built. Constructor arguments
    override the defaults, which is what lets a test hand the whole application a unit of
    work over its own schema instead of rebinding module globals.
    """

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWorkFactory | None = None,
        cache: RedisCache | None = None,
        conversations: ConversationStorage | None = None,
        background_jobs: BackgroundJobs | None = None,
        document_pairs: DocumentPairService | None = None,
        parent_chunks: ParentChunkStore | None = None,
        section_catalogue: SectionCatalogueStore | None = None,
        upload_jobs: IngestJobTracker | None = None,
        delete_jobs: IngestJobTracker | None = None,
        milvus: MilvusStore | None = None,
        embedder: EmbeddingService | None = None,
        document_loader: DocumentLoader | None = None,
        milvus_writer: MilvusWriter | None = None,
        document_remover: DocumentRemover | None = None,
        asset_store: AssetStore | None = None,
        blob_store: BlobStore | None = None,
        asset_presenter: AssetPresenter | None = None,
        entity_index: EntityAttributeIndex | None = None,
        figure_pipeline: FigurePipeline | None = None,
        entity_retriever: EntityRetriever | None = None,
        models: ChatModelFactory | None = None,
    ) -> None:
        """Every service is nameable here, and anything named is used as given.

        Two kinds of argument, both of which replace what the property would otherwise
        build. `unit_of_work` and `cache` are the INFRASTRUCTURE ones: name a unit of work
        over a throwaway schema and every service built below it writes there, which is
        how a test exercises the real services against a real database. The rest replace
        one service outright, for a test that wants a stand-in for that collaborator and
        the genuine article everywhere else.

        A service that is passed in is never built, so passing one costs nothing it would
        otherwise have opened.
        """
        self._overrides: dict[str, Any] = {
            name: value
            for name, value in (
                ("unit_of_work", unit_of_work),
                ("cache", cache),
                ("conversations", conversations),
                ("background_jobs", background_jobs),
                ("document_pairs", document_pairs),
                ("parent_chunks", parent_chunks),
                ("section_catalogue", section_catalogue),
                ("upload_jobs", upload_jobs),
                ("delete_jobs", delete_jobs),
                ("milvus", milvus),
                ("embedder", embedder),
                ("document_loader", document_loader),
                ("milvus_writer", milvus_writer),
                ("document_remover", document_remover),
                ("asset_store", asset_store),
                ("blob_store", blob_store),
                ("asset_presenter", asset_presenter),
                ("entity_index", entity_index),
                ("figure_pipeline", figure_pipeline),
                ("entity_retriever", entity_retriever),
                ("models", models),
            )
            if value is not None
        }
        self._built: dict[str, Any] = {}
        # Reentrant: building one service builds the services it depends on, on this
        # same thread, and a plain Lock would deadlock on the second acquire.
        self._lock = threading.RLock()

    def _singleton(self, name: str, build: Callable[[], T]) -> T:
        """Call `build()` once per name, then hand back the same instance forever.

        Read outside the lock first. Serving every request through one lock to fetch an
        object that was built at boot would serialise the request path on a dictionary
        lookup; the lock is only there to stop two threads racing the first build —
        which, for the embedder, would mean loading bge-m3 twice.
        """
        if name in self._overrides:
            return self._overrides[name]
        try:
            return self._built[name]
        except KeyError:
            pass
        with self._lock:
            if name not in self._built:
                self._built[name] = build()
            return self._built[name]

    # -- infrastructure ---------------------------------------------------------

    @property
    def unit_of_work(self) -> UnitOfWorkFactory:
        """The factory every service opens its transactions with.

        The class itself, not an instance: a unit of work IS one transaction, so services
        call this per operation and each gets its own session.
        """

        def build() -> UnitOfWorkFactory:
            from backend.infra.unit_of_work import SqlAlchemyUnitOfWork

            return SqlAlchemyUnitOfWork

        return self._singleton("unit_of_work", build)

    @property
    def cache(self) -> RedisCache:
        def build() -> RedisCache:
            from backend.infra.cache import RedisCache

            return RedisCache()

        return self._singleton("cache", build)

    # -- conversations ----------------------------------------------------------

    @property
    def conversations(self) -> ConversationStorage:
        def build() -> ConversationStorage:
            from backend.chat.storage import ConversationStorage

            return ConversationStorage(unit_of_work=self.unit_of_work, cache=self.cache)

        return self._singleton("conversations", build)

    @property
    def background_jobs(self) -> BackgroundJobs:
        """The threads a turn hands its save and its note update to.

        One per process: the ordering it promises — a conversation's writes land in the
        order they were queued — holds only among jobs that share an instance. Drained by
        `create_app()`'s lifespan on the way down, so a stop cannot lose a queued save.
        """

        def build() -> BackgroundJobs:
            from backend.chat.background import BackgroundJobs

            return BackgroundJobs()

        return self._singleton("background_jobs", build)

    # -- the corpus -------------------------------------------------------------

    @property
    def document_pairs(self) -> DocumentPairService:
        def build() -> DocumentPairService:
            from backend.indexing.pair_store import DocumentPairService

            return DocumentPairService(unit_of_work=self.unit_of_work)

        return self._singleton("document_pairs", build)

    @property
    def parent_chunks(self) -> ParentChunkStore:
        def build() -> ParentChunkStore:
            from backend.indexing.parent_chunk_store import ParentChunkStore

            return ParentChunkStore(unit_of_work=self.unit_of_work, cache=self.cache)

        return self._singleton("parent_chunks", build)

    @property
    def section_catalogue(self) -> SectionCatalogueStore:
        def build() -> SectionCatalogueStore:
            from backend.indexing.summary_store import SectionCatalogueStore

            return SectionCatalogueStore(unit_of_work=self.unit_of_work)

        return self._singleton("section_catalogue", build)

    # -- the vector side of the corpus ------------------------------------------

    @property
    def milvus(self) -> MilvusStore:
        """The vector collection, and the only client for it in this process.

        `rag/utils.py` used to open one at import — so importing the retrieval helpers,
        in any process, for any reason, connected to Milvus.
        """

        def build() -> MilvusStore:
            from backend.indexing.milvus_client import MilvusStore

            return MilvusStore()

        return self._singleton("milvus", build)

    @property
    def embedder(self) -> EmbeddingService:
        """The embedding model, shared by ingest and retrieval.

        One instance per process, deliberately: bge-m3 is hundreds of megabytes and takes
        roughly 110 seconds to load, so a second one is not a duplicate object but a
        duplicate model. Construction is cheap — the model loads on `warm_up()` or on
        first use — which is what lets this be built lazily like everything else here.
        """

        def build() -> EmbeddingService:
            from backend.indexing.embedding import EmbeddingService

            return EmbeddingService()

        return self._singleton("embedder", build)

    @property
    def document_loader(self) -> DocumentLoader:
        def build() -> DocumentLoader:
            from backend.indexing.document_loader import DocumentLoader

            return DocumentLoader()

        return self._singleton("document_loader", build)

    @property
    def milvus_writer(self) -> MilvusWriter:
        """Embeds leaf chunks and writes them.

        Both collaborators come from this container, so the writer embeds with the same
        model the retrieval path queries with — which is what keeps their vectors
        comparable.
        """

        def build() -> MilvusWriter:
            from backend.indexing.milvus_writer import MilvusWriter

            return MilvusWriter(
                embedding_service=self.embedder,
                milvus_manager=self.milvus,
            )

        return self._singleton("milvus_writer", build)

    # -- assets -----------------------------------------------------------------

    @property
    def blob_store(self) -> BlobStore:
        """Where image bytes live: a local tree or S3, as the profile selects."""

        def build() -> BlobStore:
            from backend.assets.blobs import build_blob_store
            from backend.profiles import get_profile

            return build_blob_store(get_profile().assets)

        return self._singleton("blob_store", build)

    @property
    def asset_store(self) -> AssetStore:
        """Asset occurrences and the content-addressed extraction cache."""

        def build() -> AssetStore:
            from backend.assets.store import AssetStore

            return AssetStore(
                unit_of_work=self.unit_of_work,
                blob_store=self.blob_store,
                cache=self.cache,
            )

        return self._singleton("asset_store", build)

    @property
    def asset_presenter(self) -> AssetPresenter:
        """Turns stored assets into what a given client can actually display."""

        def build() -> AssetPresenter:
            from backend.assets.delivery import AssetPresenter

            return AssetPresenter()

        return self._singleton("asset_presenter", build)

    @property
    def entity_index(self) -> EntityAttributeIndex:
        def build() -> EntityAttributeIndex:
            from backend.assets.entity_store import EntityAttributeIndex

            return EntityAttributeIndex(unit_of_work=self.unit_of_work)

        return self._singleton("entity_index", build)

    @property
    def figure_pipeline(self) -> FigurePipeline:
        """Extraction for the images found during ingest."""

        def build() -> FigurePipeline:
            from backend.assets.pipeline import FigurePipeline

            return FigurePipeline(
                store=self.asset_store,
                blob_store=self.blob_store,
                entity_index=self.entity_index,
            )

        return self._singleton("figure_pipeline", build)

    @property
    def entity_retriever(self) -> EntityRetriever:
        """Retrieval over entity assets by their indexed attributes."""

        def build() -> EntityRetriever:
            from backend.rag.entity_retrieval import EntityRetriever

            return EntityRetriever(
                asset_store=self.asset_store,
                entity_index=self.entity_index,
            )

        return self._singleton("entity_retriever", build)

    @property
    def document_remover(self) -> DocumentRemover:
        """Deletes a document from every store that holds part of it."""

        def build() -> DocumentRemover:
            from backend.indexing.removal import DocumentRemover

            return DocumentRemover(
                milvus=self.milvus,
                parent_chunks=self.parent_chunks,
                # Deferred: a profile with images disabled must not open a blob backend
                # merely because a document was deleted.
                asset_store=lambda: self.asset_store,
            )

        return self._singleton("document_remover", build)

    # -- models -------------------------------------------------------------------

    @property
    def models(self) -> ChatModelFactory:
        """The chat models the retrieval path calls, by role.

        One factory rather than three module globals, so which model id and whose
        credentials each role reaches for is stated once.
        """

        def build() -> ChatModelFactory:
            from backend.llm_models import ChatModelFactory

            return ChatModelFactory()

        return self._singleton("models", build)

    # -- ingest jobs ------------------------------------------------------------
    # Two trackers over one table, distinguished by the kind of job they own. Separate
    # instances rather than one with a `kind` argument on every call, because each is
    # handed to a different part of the ingest path and neither should be able to read
    # or finish the other's jobs.

    @property
    def upload_jobs(self) -> IngestJobTracker:
        def build() -> IngestJobTracker:
            from backend.jobs.upload_jobs import IngestJobTracker

            return IngestJobTracker("upload", unit_of_work=self.unit_of_work)

        return self._singleton("upload_jobs", build)

    @property
    def delete_jobs(self) -> IngestJobTracker:
        def build() -> IngestJobTracker:
            from backend.jobs.upload_jobs import IngestJobTracker

            return IngestJobTracker("delete", unit_of_work=self.unit_of_work)

        return self._singleton("delete_jobs", build)


_default: Services | None = None
_default_lock = threading.Lock()


def default_services() -> Services:
    """The process-wide container, for entry points with nowhere to inject from.

    The CLI jobs (`backend/assets/backfill.py`, `backend/indexing/build_scope_index.py`)
    and a chat turn served without an explicit container use this. An HTTP request never
    does: `create_app()` builds its own and the routes receive that one.
    """
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Services()
    return _default


def set_default_services(services: Services | None) -> None:
    """Replace the process default, or drop it so the next caller builds a fresh one.

    For tests, and for CLI entry points that configure the environment before doing any
    work. Passing `None` is the reset.
    """
    global _default
    with _default_lock:
        _default = services


__all__ = ["Services", "default_services", "set_default_services"]
