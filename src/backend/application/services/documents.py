"""The document corpus: what is in it, putting documents in, and taking them out.

Three services, one per kind of use, so each is built with only the collaborators it touches:

  `DocumentCatalogue`   reads - the documents, their chunks, their images, the bilingual rows -
                        and accepting an image's extraction after review.
  `DocumentIngestion`   uploads: a single file, or a bilingual pair, saved and then ingested in
                        the background; and a single file ingested while the request waits.
  `DocumentRemoval`     deletes, in the background or while the request waits.

What each does is what `api/routes/documents.py` did inline, moved without change: every job
step, percentage, message and error string below is the one the route used, because the admin
UI renders them and the tests pin them. The route now keeps only what is HTTP's - reading the
upload, validating the request, scheduling the background task, shaping the response.

Every collaborator is a `Provider`, called inside the operation that first uses it and never
when the service is built: a job poll builds the job tracker and nothing else, and a loader
that cannot be built fails the job at `parse`. Every store whose construction can fail is built
where the route built it, and fails with the route's message. The two whose constructors only
keep a unit of work - the job tracker and the pair store - are now built after the request is
validated rather than before, and the pair list builds its store inside its operation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import backend.indexing.language_check as language_check
from backend.application.ports.collaborators import (
    AssetReview,
    DocumentEraser,
    DocumentPairs,
    DocumentParser,
    JobTracker,
    ParentChunks,
    Provider,
    VectorIndex,
    VectorWriter,
)
from backend.domain.errors import NotFound, OperationFailed, operation
from backend.text_matching import fold

logger = logging.getLogger(__name__)

#: One chunk listing may not carry more than this many chunks. A document is normally a few
#: hundred; the ceiling is here so a pathological one cannot build a response large enough to
#: matter, and `total` still reports the truth so the UI can say it was cut.
CHUNK_PAGE_LIMIT = 2000

#: What the inspector reads. `dense_embedding` and `sparse_embedding` are deliberately absent -
#: they are most of a chunk's bytes and none of its meaning.
CHUNK_FIELDS = [
    "chunk_id",
    "parent_chunk_id",
    "root_chunk_id",
    "chunk_level",
    "chunk_idx",
    "page_number",
    "modality",
    "text",
    "asset_ids",
]

#: Saves one uploaded file to the path it is given. The route supplies it, because reading
#: the request body is HTTP's business and nothing here may import the framework.
SaveUpload = Callable[[Path], Awaitable[None]]


def _level(doc: dict) -> int:
    return int(doc.get("chunk_level", 0) or 0)


def _split(docs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Parents are levels 1 and 2, kept in Postgres; leaves are level 3, vectorised."""
    return (
        [doc for doc in docs if _level(doc) in (1, 2)],
        [doc for doc in docs if _level(doc) == 3],
    )


def _forget_corpus_languages(callback: Callable[[], None]) -> None:
    """Make the next question re-ask which languages the corpus is published in.

    Pairing is what tells retrieval which languages the corpus speaks, and that answer is
    memoized for a few minutes. Clearing it means an admin who has just uploaded the Arabic
    half sees translations stop immediately. It is a cache hint, so it never fails the job.
    """
    try:
        callback()
    except Exception:  # pragma: no cover - a cache hint must never fail an upload
        logger.exception("could not clear the corpus-language memo")


def _detach_from_pair(pairs: DocumentPairs, filename: str, forget: Callable[[], None]) -> None:
    """Take a deleted file off whatever pair row holds it.

    Never raises: the document is already gone from the index by the time this runs, so a
    failure here must not turn a completed delete into a failed job. The cost of it failing is
    a row naming a file that no longer exists, which shows as a zero chunk count in the pair
    list rather than as a wrong answer - routing only ever excludes a twin, so a stale row can
    hide a document but never invent one.
    """
    try:
        pairs.detach(filename)
        _forget_corpus_languages(forget)
    except Exception:  # pragma: no cover - a bookkeeping failure must not fail a delete
        logger.exception("could not detach %s from its document pair", filename)


# -- reads --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkListing:
    """A document's chunks as the inspector draws them. `chunks` are dicts in `ChunkInfo`'s
    shape, ordered by level then index, at most `CHUNK_PAGE_LIMIT` of them."""

    filename: str
    total: int
    returned: int
    match_count: int
    query: str
    truncated: bool
    chunks: list[dict]


@dataclass(frozen=True)
class AssetListing:
    """A document's images, ordered by page, and the chunks each one produced."""

    dossiers: list[Any]
    chunk_ids_by_asset: dict[str, list[str]]


def _chunk_asset_ids(raw: Any) -> list[str]:
    """`asset_ids` is stored as a JSON array in a VARCHAR. A malformed one is not worth
    failing a whole inspection over, so it reads as no assets."""
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        parsed = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _chunk(row: dict) -> dict:
    text = str(row.get("text") or "")
    return {
        "chunk_id": str(row.get("chunk_id") or ""),
        "parent_chunk_id": str(row.get("parent_chunk_id") or ""),
        "root_chunk_id": str(row.get("root_chunk_id") or ""),
        "chunk_level": int(row.get("chunk_level") or 0),
        "chunk_idx": int(row.get("chunk_idx") or 0),
        "page_number": int(row.get("page_number") or 0),
        "modality": str(row.get("modality") or "text"),
        "text": text,
        "char_count": len(text),
        "asset_ids": _chunk_asset_ids(row.get("asset_ids")),
        "matched": False,
    }


def _filename_filter(filename: str) -> str:
    return f"filename == {json.dumps(filename, ensure_ascii=False)}"


class DocumentCatalogue:
    """What the corpus holds, read for the admin views."""

    def __init__(
        self,
        *,
        vectors: Provider[VectorIndex],
        parent_chunks: Provider[ParentChunks],
        pairs: Provider[DocumentPairs],
        assets: Provider[AssetReview],
    ) -> None:
        self._vectors = vectors
        self._parent_chunks = parent_chunks
        self._pairs = pairs
        self._assets = assets

    def list_documents(self) -> list[dict]:
        """Each indexed file with its type and how many leaf chunks it has in the index."""
        with operation("Failed to retrieve document list"):
            vectors = self._vectors()
            vectors.init_collection()
            results = vectors.query(output_fields=["filename", "file_type"], limit=10000)
            stats: dict[str, dict] = {}
            for item in results:
                filename = item.get("filename", "")
                if filename not in stats:
                    stats[filename] = {
                        "filename": filename,
                        "file_type": item.get("file_type", ""),
                        "chunk_count": 0,
                    }
                stats[filename]["chunk_count"] += 1
            return list(stats.values())

    def list_chunks(self, filename: str, query: str = "") -> ChunkListing:
        """Every chunk a document was indexed into, with `query` MARKED rather than filtered.

        BOTH stores are read, and that is not an optimisation. Only leaf chunks are
        vectorised: levels 1 and 2 are in the parent-chunk store and level 3 is in the index.
        Reading the index alone returns every leaf with a `parent_chunk_id` naming a chunk
        that is not in the response - measured on the shipped corpus, 175 of 175 - so the tree
        would be 175 orphans.

        `query` marks each chunk `matched` and the document stays whole, because the bordered
        document view needs the whole structure to draw. Matching is on FOLDED text, the same
        folding the sparse retrieval lane keys on, so an admin typing مدرسة finds مدرسه and
        typing without diacritics finds text with them. Folding here rather than in the
        browser keeps one implementation of it.
        """
        with operation("Failed to read chunks"):
            vectors = self._vectors()
            vectors.init_collection()
            rows = vectors.query_all(
                filter_expr=_filename_filter(filename), output_fields=CHUNK_FIELDS
            )
            rows = list(rows) + self._parent_chunks().documents_by_filename(filename)

        chunks = sorted(
            (_chunk(row) for row in rows),
            key=lambda chunk: (chunk["chunk_level"], chunk["chunk_idx"], chunk["chunk_id"]),
        )
        total = len(chunks)
        needle = fold(query)
        if needle:
            for chunk in chunks:
                chunk["matched"] = needle in fold(chunk["text"])

        shown = chunks[:CHUNK_PAGE_LIMIT]
        return ChunkListing(
            filename=filename,
            total=total,
            returned=len(shown),
            match_count=sum(1 for chunk in shown if chunk["matched"]),
            query=query,
            truncated=total > CHUNK_PAGE_LIMIT,
            chunks=shown,
        )

    def list_assets(self, filename: str) -> AssetListing:
        """A document's images with what extraction made of them, for manual review.

        Each image carries the ids of the chunks it produced. The link is only ever written
        the other way - a chunk names its `asset_ids` - so the reverse is built here by
        inverting it. Chunks are read best-effort: a document's images are still worth
        reviewing when the vector store cannot be reached, so an unreadable index costs the
        chunk ids and nothing else.
        """
        with operation("Failed to read assets"):
            dossiers = self._assets().list_by_filename(filename)

        chunk_ids_by_asset: dict[str, list[str]] = {}
        try:
            vectors = self._vectors()
            vectors.init_collection()
            rows = vectors.query_all(
                filter_expr=_filename_filter(filename), output_fields=CHUNK_FIELDS
            )
            for row in sorted(rows, key=lambda r: (r.get("chunk_level", 0), r.get("chunk_idx", 0))):
                for asset_id in _chunk_asset_ids(row.get("asset_ids")):
                    chunk_ids_by_asset.setdefault(asset_id, []).append(row.get("chunk_id", ""))
        except Exception:
            logger.exception("Could not read chunks for %s; listing assets without them", filename)

        ordered = sorted(
            dossiers, key=lambda dossier: (dossier.source.page_number, dossier.asset_id)
        )
        return AssetListing(dossiers=ordered, chunk_ids_by_asset=chunk_ids_by_asset)

    def mark_reviewed(self, asset_id: str) -> Any:
        """Record that an admin accepted this image's extraction; the image as it now reads.

        Clears `needs_review` and nothing else: no re-extraction, no edited text, no change to
        whether the image is indexed. Keyed by DIGEST, because the extraction is, so a
        letterhead accepted on page 1 does not ask again on page 40.
        """
        with operation("Failed to record the review"):
            dossier = self._assets().mark_reviewed(asset_id)
        if dossier is None:
            raise NotFound("Asset not found")
        return dossier

    def list_pairs(self) -> list[dict]:
        """The bilingual view of the corpus: one row per entry, up to two files on it.

        Chunk counts come from the index, the only place that knows how much of a document was
        actually indexed. A file on a row but absent from the index shows zero, which means an
        ingest failed after the row was written - worth surfacing, not hiding. Files that
        belong to no row (indexed before pairing existed, or through the single-file upload)
        are listed too, because they still answer questions.
        """
        with operation("Failed to retrieve document pairs"):
            vectors = self._vectors()
            vectors.init_collection()
            counts: dict[str, int] = {}
            for item in vectors.query(output_fields=["filename"], limit=10000):
                name = item.get("filename", "")
                counts[name] = counts.get(name, 0) + 1

            pairs = self._pairs().list_pairs()
            rows = [
                {
                    "pair_id": pair.pair_id,
                    "title": pair.title,
                    "filename_ar": pair.filename_ar,
                    "filename_en": pair.filename_en,
                    "paired": pair.paired,
                    "chunk_count_ar": counts.get(pair.filename_ar, 0) if pair.filename_ar else 0,
                    "chunk_count_en": counts.get(pair.filename_en, 0) if pair.filename_en else 0,
                }
                for pair in pairs
            ]
            claimed = {pair.filename_ar for pair in pairs} | {pair.filename_en for pair in pairs}
            unpaired = [
                {
                    "pair_id": "",
                    "title": name,
                    "filename_ar": "",
                    "filename_en": name,
                    "paired": False,
                    "chunk_count_ar": 0,
                    "chunk_count_en": count,
                    "unassigned": True,
                }
                for name, count in sorted(counts.items())
                if name and name not in claimed
            ]
            return rows + unpaired


# -- uploads ------------------------------------------------------------------------


class FigureProgress:
    """Figure extraction, reported into the upload job the admin UI polls.

    Implements `indexing.ingest_progress.IngestProgress` structurally - the loader only calls
    its two methods - rather than by subclassing it, because that module names a type from the
    asset pipeline and would pull the whole ingestion stack into this layer.

    Without it the `parse` step reports 5% once and then nothing until every image in the
    document has been through a vision model: minutes of a frozen bar on an illustrated
    document, which the job tracker's own docstring names as the reason it cannot tell a slow
    ingest from a dead one.
    """

    #: `parse` is at 5% when extraction starts and needs room afterwards for chunking, so
    #: extraction owns the span between. Both ends are reported: 5% the moment the image count
    #: is known, which is itself news.
    START, END = 5, 85

    def __init__(self, jobs: JobTracker, job_id: str) -> None:
        self._jobs = jobs
        self._job_id = job_id
        self._report: Any = None

    def figures_progress(self, done: int, total: int) -> None:
        percent = self.START + int((self.END - self.START) * done / total) if total else self.START
        self._jobs.update_step(
            self._job_id,
            "parse",
            percent,
            "running",
            f"Extracting images: {done} of {total}",
            sub_label="Extracting images",
            sub_done=done,
            sub_total=total,
        )

    def figures_finished(self, report: Any) -> None:
        self._report = report

    def summary(self) -> str:
        """What extraction produced, for the step's closing message.

        Empty when the document had no images, so a text-only document reads exactly as it
        did before. `failed` is named even at zero once anything was extracted: "0 failed"
        is the sentence an admin needs to see to stop wondering.
        """
        report = self._report
        if report is None or not report.total:
            return ""
        parts = [f"{report.extracted} extracted"]
        if report.cached:
            parts.append(f"{report.cached} already known")
        if report.dropped:
            parts.append(f"{report.dropped} skipped")
        parts.append(f"{report.failed} failed")
        return f". Images: {report.total} ({', '.join(parts)})"


@dataclass(frozen=True)
class AcceptedUpload:
    """A file saved and its job created; ingestion is the caller's to schedule."""

    job_id: str
    filename: str
    path: Path


@dataclass(frozen=True)
class AcceptedPair:
    """One or two files saved under one job. `sides` is (language, file path, filename)."""

    job_id: str
    names: str
    sides: list[tuple[str, str, str]]


@dataclass(frozen=True)
class IngestedDocument:
    """What a document ingested while the request waited produced."""

    filename: str
    leaf_chunks: int
    parent_chunks: int


class DocumentIngestion:
    """Documents into the corpus.

    The asynchronous paths are two halves on purpose. `accept_*` runs inside the request: it
    saves the bytes and creates the job, so a failure to save is still the request's to report.
    `ingest*` runs afterwards on a background task and reports only through the job, because
    there is no request left to answer.
    """

    def __init__(
        self,
        *,
        loader: Provider[DocumentParser],
        parent_chunks: Provider[ParentChunks],
        vector_writer: Provider[VectorWriter],
        remover: Provider[DocumentEraser],
        pairs: Provider[DocumentPairs],
        jobs: Provider[JobTracker],
        upload_dir: Path,
        forget_corpus_languages: Callable[[], None],
    ) -> None:
        self._loader = loader
        self._parent_chunks = parent_chunks
        self._vector_writer = vector_writer
        self._remover = remover
        self._pairs = pairs
        self._jobs = jobs
        self._upload_dir = upload_dir
        self._forget_corpus_languages = forget_corpus_languages

    # -- the request half --

    async def accept_upload(self, filename: str, save: SaveUpload) -> AcceptedUpload:
        """Save one file and create its job. The filename is already validated."""
        jobs = self._jobs()
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        job = jobs.create_job(filename)
        path = self._upload_dir / filename
        try:
            jobs.update_step(job["job_id"], "upload", 1, "running", "Saving file to server")
            await save(path)
            jobs.complete_step(
                job["job_id"], "upload", "File uploaded, waiting for background processing"
            )
        except Exception as exc:
            jobs.fail_job(job["job_id"], "upload", f"Failed to save file: {exc}")
            raise OperationFailed("Failed to save file", exc) from exc
        return AcceptedUpload(job_id=job["job_id"], filename=filename, path=path)

    def require_pair(self, pair_id: str) -> None:
        """Refuse a `pair_id` that names no row. An empty one means a new row."""
        if pair_id and not self._pairs().get_pair(pair_id):
            raise NotFound(f"No document pair {pair_id}")

    async def accept_pair(self, uploads: Sequence[tuple[str, str, SaveUpload]]) -> AcceptedPair:
        """Save the one or two files of a bilingual entry under one job.

        `uploads` is (language, filename, save). The request has already been validated:
        at least one file, each of a supported type, and not the same file twice.
        """
        names = [filename for _language, filename, _save in uploads]
        jobs = self._jobs()
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        job = jobs.create_job(", ".join(names))
        saved: list[tuple[str, str, str]] = []
        try:
            jobs.update_step(job["job_id"], "upload", 1, "running", "Saving files to server")
            for language, filename, save in uploads:
                path = self._upload_dir / filename
                await save(path)
                saved.append((language, str(path), filename))
            jobs.complete_step(
                job["job_id"],
                "upload",
                f"{len(saved)} file(s) uploaded, waiting for background processing",
            )
        except Exception as exc:
            jobs.fail_job(job["job_id"], "upload", f"Failed to save file: {exc}")
            raise OperationFailed("Failed to save file", exc) from exc
        return AcceptedPair(job_id=job["job_id"], names=", ".join(names), sides=saved)

    # -- the background half --

    def ingest(self, job_id: str, file_path: str, filename: str) -> None:
        """Parse, chunk and index one saved file, replacing any document of the same name."""
        jobs = self._jobs()
        failed_step = "cleanup"
        try:
            jobs.complete_step(job_id, "upload", "File saved to server")

            failed_step = "cleanup"
            jobs.update_step(
                job_id, "cleanup", 10, "running", "Cleaning up old document with the same name"
            )
            self._remover().remove(filename)
            jobs.complete_step(job_id, "cleanup", "Old version cleanup complete")

            failed_step = "parse"
            jobs.update_step(
                job_id,
                "parse",
                5,
                "running",
                "Parsing document and performing three-level chunking",
            )
            # Extraction reports INTO `parse` rather than as a step of its own. It is part of
            # parsing, and a new step key would have to be added to the frontend store's
            # `createUploadSteps()` as well - `updateUploadStep` drops an unknown key with
            # `if (idx === -1) return`, so half the change would show nothing and log nothing.
            progress = FigureProgress(jobs, job_id)
            new_docs = self._loader().load_document(file_path, filename, progress=progress)
            if not new_docs:
                raise ValueError("Document processing failed: could not extract content")

            parent_docs, leaf_docs = _split(new_docs)
            if not leaf_docs:
                raise ValueError(
                    "Document processing failed: no retrievable leaf chunks were generated"
                )
            # Figure enrichment happens inside load_document, so its outcome is reported here
            # rather than as a separate job step - that keeps the step lists, and the progress
            # UI built on them, unchanged.
            figure_chunks = sum(1 for doc in leaf_docs if doc.get("modality") == "figure")
            figure_note = f", {figure_chunks} from figures" if figure_chunks else ""
            jobs.complete_step(
                job_id,
                "parse",
                f"Parsing complete: {len(parent_docs)} parent chunks, "
                f"{len(leaf_docs)} leaf chunks{figure_note}{progress.summary()}",
            )

            failed_step = "parent_store"
            jobs.update_step(job_id, "parent_store", 20, "running", "Writing parent chunks")
            self._parent_chunks().upsert_documents(parent_docs)
            jobs.complete_step(job_id, "parent_store", f"Parent chunks stored: {len(parent_docs)}")

            failed_step = "vector_store"
            total_leaf = len(leaf_docs)
            jobs.update_step(
                job_id,
                "vector_store",
                0,
                "running",
                f"Vectorizing and storing: 0 / {total_leaf}",
                total_chunks=total_leaf,
                processed_chunks=0,
            )

            def on_vector_progress(processed: int, total: int) -> None:
                percent = round(processed * 100 / total) if total else 100
                jobs.update_step(
                    job_id,
                    "vector_store",
                    percent,
                    "running",
                    f"Vectorizing and storing: {processed} / {total}",
                    total_chunks=total,
                    processed_chunks=processed,
                )

            self._vector_writer().write_documents(leaf_docs, progress_callback=on_vector_progress)
            jobs.complete_step(
                job_id,
                "vector_store",
                f"Vectorization and storage complete: {total_leaf} leaf chunks",
            )
            jobs.complete_job(job_id, f"Successfully uploaded and processed {filename}")
        except Exception as exc:
            jobs.fail_job(job_id, failed_step, str(exc))

    def _parse_side(self, language: str, file_path: str, filename: str) -> list[dict]:
        """Parse one half of a pair and check it against the column it arrived in.

        Raises rather than returning a status, so `ingest_pair` can do this for BOTH sides
        before writing either. A pair half-ingested because the second file was in the wrong
        column would leave the corpus in a state the form cannot express - one side indexed,
        the row unpaired - and the admin with no obvious way back.
        """
        docs = self._loader().load_document(file_path, filename)
        if not docs:
            raise ValueError(f"{filename}: could not extract content")

        verdict = language_check.verify(" ".join(d.get("text") or "" for d in docs[:40]), language)
        if not verdict.agrees:
            raise ValueError(language_check.describe_mismatch(filename, verdict))
        return docs

    def ingest_pair(
        self, job_id: str, pair_id: str, title: str, sides: Sequence[tuple[str, str, str]]
    ) -> None:
        """Ingest one bilingual entry: up to two files, one per language.

        `sides` is (language, file path, filename). One entry is a single-language row, which
        is a normal and permanent state - a document that exists only in English still answers
        Arabic questions.

        Ordered so that everything which can REJECT the upload happens before anything that
        writes: both files are parsed and language-checked first, and only then is either
        indexed. That is what keeps a rejected pair from leaving half a document behind.
        """
        jobs = self._jobs()
        failed_step = "parse"
        try:
            jobs.complete_step(job_id, "upload", "Files saved to server")

            failed_step = "parse"
            parsed = []
            for index, (language, file_path, filename) in enumerate(sides, start=1):
                jobs.update_step(
                    job_id,
                    "parse",
                    round(index * 100 / (len(sides) + 1)),
                    "running",
                    f"Parsing {filename} ({language})",
                )
                parsed.append((language, filename, self._parse_side(language, file_path, filename)))

            leaf_total = sum(1 for _, _, docs in parsed for d in docs if _level(d) == 3)
            if not leaf_total:
                raise ValueError("no retrievable leaf chunks were generated")
            jobs.complete_step(
                job_id, "parse", f"Parsed {len(parsed)} file(s), {leaf_total} leaf chunks"
            )

            # Only now does anything get written. Replacing a same-named document is part of
            # writing, not of validation, so it happens after both files are known good.
            #
            # `include_assets=False` because "parsing" above was not read-only: figure
            # enrichment runs inside load_document and has already committed each filename's
            # document_assets rows. Deleting them here - keyed on the same filename - wiped
            # exactly what the parse had just written, so every paired upload finished with an
            # empty document_assets while the extraction cache still reported every image as
            # "from cache". Nothing could be DISPLAYED. The single-file path is unaffected: it
            # cleans up BEFORE it parses.
            failed_step = "cleanup"
            jobs.update_step(job_id, "cleanup", 10, "running", "Cleaning up old versions")
            for _, filename, _ in parsed:
                self._remover().remove(filename, include_assets=False)
            jobs.complete_step(job_id, "cleanup", "Old version cleanup complete")

            failed_step = "parent_store"
            jobs.update_step(job_id, "parent_store", 20, "running", "Writing parent chunks")
            parent_written = 0
            for _, _, docs in parsed:
                parents = [d for d in docs if _level(d) in (1, 2)]
                self._parent_chunks().upsert_documents(parents)
                parent_written += len(parents)
            jobs.complete_step(job_id, "parent_store", f"Parent chunks stored: {parent_written}")

            failed_step = "vector_store"
            written = 0
            jobs.update_step(
                job_id,
                "vector_store",
                0,
                "running",
                f"Vectorizing and storing: 0 / {leaf_total}",
                total_chunks=leaf_total,
                processed_chunks=0,
            )
            for _, _, docs in parsed:
                leaves = [d for d in docs if _level(d) == 3]

                def on_progress(processed: int, _total: int, base: int = written) -> None:
                    done = base + processed
                    jobs.update_step(
                        job_id,
                        "vector_store",
                        round(done * 100 / leaf_total),
                        "running",
                        f"Vectorizing and storing: {done} / {leaf_total}",
                        total_chunks=leaf_total,
                        processed_chunks=done,
                    )

                self._vector_writer().write_documents(leaves, progress_callback=on_progress)
                written += len(leaves)
            jobs.complete_step(
                job_id, "vector_store", f"Vectorization and storage complete: {written} leaf chunks"
            )

            # The row is written LAST. A file that failed to index must not be recorded as
            # this entry's Arabic or English half, or routing would exclude the twin in favour
            # of a document that is not in the corpus.
            for language, filename, _ in parsed:
                pair_id = self._pairs().attach(pair_id, language, filename, title=title).pair_id

            _forget_corpus_languages(self._forget_corpus_languages)

            names = ", ".join(filename for _, filename, _ in parsed)
            jobs.complete_job(job_id, f"Successfully uploaded and processed {names}")
        except Exception as exc:
            jobs.fail_job(job_id, failed_step, str(exc))

    # -- while the request waits --

    def ingest_now(self, filename: str, content: bytes) -> IngestedDocument:
        """Replace any same-named document with `content`, parsed and indexed before returning.

        The filename is already validated. Each failure is reported under the operation that
        failed, in the words the synchronous upload route always used.
        """
        with operation("Document upload failed"):
            self._upload_dir.mkdir(parents=True, exist_ok=True)
            # Clean up an existing document with the same name, to keep the stores consistent.
            self._remover().remove(filename)
            path = self._upload_dir / filename
            path.write_bytes(content)

            try:
                new_docs = self._loader().load_document(str(path), filename)
            except Exception as exc:
                raise OperationFailed("Document processing failed", exc) from exc
            if not new_docs:
                raise OperationFailed("Document processing failed: could not extract content")

            parent_docs, leaf_docs = _split(new_docs)
            if not leaf_docs:
                raise OperationFailed(
                    "Document processing failed: no retrievable leaf chunks were generated"
                )

            self._parent_chunks().upsert_documents(parent_docs)
            self._vector_writer().write_documents(leaf_docs)
            return IngestedDocument(
                filename=filename, leaf_chunks=len(leaf_docs), parent_chunks=len(parent_docs)
            )

    # -- the upload jobs --

    def upload_job(self, job_id: str) -> dict:
        job = self._jobs().get_job(job_id)
        if not job:
            raise NotFound("Upload job does not exist or has expired")
        return job

    def upload_jobs(self) -> list[dict]:
        """Every upload job, newest first."""
        jobs = self._jobs().list_jobs()
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return jobs


# -- deletes ------------------------------------------------------------------------


class DocumentRemoval:
    """Documents out of the corpus: the index, the parent chunks, the images, and the pair row."""

    def __init__(
        self,
        *,
        remover: Provider[DocumentEraser],
        pairs: Provider[DocumentPairs],
        jobs: Provider[JobTracker],
        delete_steps: list[tuple[str, str]],
        forget_corpus_languages: Callable[[], None],
    ) -> None:
        self._remover = remover
        self._pairs = pairs
        self._jobs = jobs
        self._delete_steps = delete_steps
        self._forget_corpus_languages = forget_corpus_languages

    def start(self, filename: str) -> dict:
        """Create the delete job the admin UI will poll. The work is the caller's to schedule."""
        jobs = self._jobs()
        job = jobs.create_job(
            filename,
            steps=self._delete_steps,
            current_step="prepare",
            message="Waiting to delete",
            completion_step="parent_store",
        )
        jobs.update_step(job["job_id"], "prepare", 1, "running", "Delete job submitted")
        return job

    def remove(self, job_id: str, filename: str) -> None:
        """Delete a document in the background, reporting through its job."""
        jobs = self._jobs()
        try:
            chunks_deleted = self._remover().remove(filename, jobs, job_id)
            # Clear the file off its pair row. Done HERE and not in the remover, which upload
            # also calls to replace a same-named document - detaching there would silently
            # unpair a document every time somebody re-uploaded one side of it.
            _detach_from_pair(self._pairs(), filename, self._forget_corpus_languages)
            jobs.complete_job(
                job_id, f"Deleted {filename}, {chunks_deleted} vector records removed"
            )
        except Exception as exc:
            job = jobs.get_job(job_id)
            current_step = job.get("current_step", "prepare") if job else "prepare"
            jobs.fail_job(job_id, current_step, str(exc))

    def remove_now(self, filename: str) -> int:
        """Delete a document while the request waits; how many vector records went."""
        with operation("Failed to delete document"):
            chunks_deleted = self._remover().remove(filename)
            _detach_from_pair(self._pairs(), filename, self._forget_corpus_languages)
            return chunks_deleted

    def delete_job(self, job_id: str) -> dict:
        job = self._jobs().get_job(job_id)
        if not job:
            raise NotFound("Delete job does not exist or has expired")
        return job


__all__ = [
    "CHUNK_FIELDS",
    "CHUNK_PAGE_LIMIT",
    "AcceptedPair",
    "AcceptedUpload",
    "AssetListing",
    "ChunkListing",
    "DocumentCatalogue",
    "DocumentIngestion",
    "DocumentRemoval",
    "FigureProgress",
    "IngestedDocument",
    "SaveUpload",
]
