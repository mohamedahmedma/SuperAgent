import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from backend.api.deps import get_services
from backend.api.resources import (
    UPLOAD_DIR,
    ensure_upload_dir,
    is_supported_document,
    save_upload_file,
)
from backend.chat.language import ARABIC, ENGLISH
from backend.composition import Services
from backend.db.models import User
import backend.indexing.language_check as language_check
from backend.infra.auth import require_admin
from backend.profiles import get_profile
from backend.text_matching import fold
from backend.jobs import DELETE_STEPS
from backend.schemas import (
    ChunkInfo,
    DocumentChunkListResponse,
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentPairInfo,
    DocumentPairListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


def _process_upload_job(services: Services, job_id: str, file_path: str, filename: str) -> None:
    jobs = services.upload_jobs
    failed_step = "cleanup"
    try:
        jobs.complete_step(job_id, "upload", "File saved to server")

        failed_step = "cleanup"
        jobs.update_step(job_id, "cleanup", 10, "running", "Cleaning up old document with the same name")
        services.document_remover.remove(filename)
        jobs.complete_step(job_id, "cleanup", "Old version cleanup complete")

        failed_step = "parse"
        jobs.update_step(job_id, "parse", 5, "running", "Parsing document and performing three-level chunking")
        new_docs = services.document_loader.load_document(file_path, filename)
        if not new_docs:
            raise ValueError("Document processing failed: could not extract content")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise ValueError("Document processing failed: no retrievable leaf chunks were generated")
        # Figure enrichment happens inside load_document, so its outcome is reported
        # here rather than as a separate job step — that keeps DELETE_STEPS/DEFAULT_STEPS
        # (and the progress UI built on them) unchanged.
        figure_chunks = sum(1 for doc in leaf_docs if doc.get("modality") == "figure")
        figure_note = f", {figure_chunks} from figures" if figure_chunks else ""
        jobs.complete_step(
            job_id,
            "parse",
            f"Parsing complete: {len(parent_docs)} parent chunks, {len(leaf_docs)} leaf chunks{figure_note}",
        )

        failed_step = "parent_store"
        jobs.update_step(job_id, "parent_store", 20, "running", "Writing parent chunks")
        services.parent_chunks.upsert_documents(parent_docs)
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

        def _on_vector_progress(processed: int, total: int) -> None:
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

        services.milvus_writer.write_documents(leaf_docs, progress_callback=_on_vector_progress)
        jobs.complete_step(job_id, "vector_store", f"Vectorization and storage complete: {total_leaf} leaf chunks")
        jobs.complete_job(job_id, f"Successfully uploaded and processed {filename}")
    except Exception as e:
        jobs.fail_job(job_id, failed_step, str(e))


def _parse_side(loader, language: str, file_path: str, filename: str) -> list:
    """Parse one half of a pair and check it against the column it arrived in.

    Raises rather than returning a status, so `_process_pair_upload_job` can do this for
    BOTH sides before writing either. A pair half-ingested because the second file was
    in the wrong column would leave the corpus in a state the form cannot express — one
    side indexed, the row unpaired — and the admin with no obvious way back.
    """
    docs = loader.load_document(file_path, filename)
    if not docs:
        raise ValueError(f"{filename}: could not extract content")

    verdict = language_check.verify(" ".join(d.get("text") or "" for d in docs[:40]), language)
    if not verdict.agrees:
        raise ValueError(language_check.describe_mismatch(filename, verdict))
    return docs


def _process_pair_upload_job(
    services: Services, job_id: str, pair_id: str, title: str, sides: list
) -> None:
    """Ingest one bilingual entry: up to two files, one per language.

    `sides` is a list of (language, file_path, filename). One entry is a single-language
    row, which is a normal and permanent state — a document that exists only in English
    still answers Arabic questions (see document_pairs.superseded_filenames).

    Ordered so that everything which can REJECT the upload happens before anything that
    writes: both files are parsed and language-checked first, and only then is either
    indexed. That is what keeps a rejected pair from leaving half a document behind.
    """
    jobs = services.upload_jobs
    failed_step = "parse"
    try:
        jobs.complete_step(job_id, "upload", "Files saved to server")

        failed_step = "parse"
        parsed = []
        for index, (language, file_path, filename) in enumerate(sides, start=1):
            jobs.update_step(
                job_id, "parse", round(index * 100 / (len(sides) + 1)), "running",
                f"Parsing {filename} ({language})",
            )
            parsed.append(
                (
                    language,
                    filename,
                    _parse_side(services.document_loader, language, file_path, filename),
                )
            )

        leaf_total = sum(
            1 for _, _, docs in parsed for d in docs if int(d.get("chunk_level", 0) or 0) == 3
        )
        if not leaf_total:
            raise ValueError("no retrievable leaf chunks were generated")
        jobs.complete_step(
            job_id, "parse",
            f"Parsed {len(parsed)} file(s), {leaf_total} leaf chunks",
        )

        # Only now does anything get written. Replacing a same-named document is part of
        # writing, not of validation, so it happens after both files are known good.
        #
        # `include_assets=False` because "parsing" above was not read-only: figure
        # enrichment runs inside load_document and has already committed each
        # filename's document_assets rows. Deleting them here — keyed on the same
        # filename — wiped exactly what the parse had just written, so every paired
        # upload finished with an empty document_assets while the extraction cache
        # (which delete_by_filename keeps on purpose) still reported every image as
        # "from cache". Retrieval was unaffected, since the vision surrogates live in
        # the chunks, but nothing could be DISPLAYED: _displayable_hashes went empty
        # and no figure could reach the user. The single-file path is unaffected — it
        # cleans up BEFORE it parses.
        failed_step = "cleanup"
        jobs.update_step(job_id, "cleanup", 10, "running", "Cleaning up old versions")
        for _, filename, _ in parsed:
            services.document_remover.remove(filename, include_assets=False)
        jobs.complete_step(job_id, "cleanup", "Old version cleanup complete")

        failed_step = "parent_store"
        jobs.update_step(job_id, "parent_store", 20, "running", "Writing parent chunks")
        parent_written = 0
        for _, _, docs in parsed:
            parents = [d for d in docs if int(d.get("chunk_level", 0) or 0) in (1, 2)]
            services.parent_chunks.upsert_documents(parents)
            parent_written += len(parents)
        jobs.complete_step(job_id, "parent_store", f"Parent chunks stored: {parent_written}")

        failed_step = "vector_store"
        written = 0
        jobs.update_step(
            job_id, "vector_store", 0, "running", f"Vectorizing and storing: 0 / {leaf_total}",
            total_chunks=leaf_total, processed_chunks=0,
        )
        for _, _, docs in parsed:
            leaves = [d for d in docs if int(d.get("chunk_level", 0) or 0) == 3]

            def _on_progress(processed: int, _total: int, base: int = written) -> None:
                done = base + processed
                jobs.update_step(
                    job_id, "vector_store", round(done * 100 / leaf_total), "running",
                    f"Vectorizing and storing: {done} / {leaf_total}",
                    total_chunks=leaf_total, processed_chunks=done,
                )

            services.milvus_writer.write_documents(leaves, progress_callback=_on_progress)
            written += len(leaves)
        jobs.complete_step(
            job_id, "vector_store", f"Vectorization and storage complete: {written} leaf chunks"
        )

        # The row is written LAST. A file that failed to index must not be recorded as
        # this entry's Arabic or English half, or routing would exclude the twin in
        # favour of a document that is not in the corpus.
        for language, filename, _ in parsed:
            pair_id = services.document_pairs.attach(
                pair_id, language, filename, title=title
            ).pair_id

        # Pairing is what tells retrieval which languages the corpus speaks, and that
        # answer is memoized for a few minutes so it is not re-read on every question.
        # Clearing it here means an admin who has just uploaded the Arabic half sees
        # translations stop immediately, rather than wondering for five minutes whether
        # the upload worked. Invalidated from the ROUTE because the store must not import
        # retrieval — indexing does not depend on rag anywhere else either.
        _forget_corpus_languages()

        names = ", ".join(filename for _, filename, _ in parsed)
        jobs.complete_job(job_id, f"Successfully uploaded and processed {names}")
    except Exception as e:
        jobs.fail_job(job_id, failed_step, str(e))


def _detach_from_pair(document_pairs, filename: str) -> None:
    """Take a deleted file off whatever pair row holds it.

    Never raises: the document is already gone from the index by the time this runs, so
    a failure here must not turn a completed delete into a failed job. The cost of it
    failing is a row naming a file that no longer exists, which shows as a zero chunk
    count in the pair list rather than as a wrong answer — routing only ever excludes a
    twin, so a stale row can hide a document but never invent one.
    """
    try:
        document_pairs.detach(filename)
        _forget_corpus_languages()
    except Exception:  # pragma: no cover - a bookkeeping failure must not fail a delete
        logger.exception("could not detach %s from its document pair", filename)


def _forget_corpus_languages() -> None:
    """Make the next question re-ask which languages the corpus is published in."""
    try:
        from backend.rag.query_translation import reset_coverage

        reset_coverage()
    except Exception:  # pragma: no cover - a cache hint must never fail an upload
        logger.exception("could not clear the corpus-language memo")


def _process_delete_job(services: Services, job_id: str, filename: str) -> None:
    jobs = services.delete_jobs
    failed_step = "prepare"
    try:
        chunks_deleted = services.document_remover.remove(filename, jobs, job_id)
        # Clear the file off its pair row. Done HERE and not in
        # DocumentRemover.remove, which upload also calls to replace a
        # same-named document — detaching there would silently unpair a document every
        # time somebody re-uploaded one side of it.
        _detach_from_pair(services.document_pairs, filename)
        jobs.complete_job(job_id, f"Deleted {filename}, {chunks_deleted} vector records removed")
    except Exception as e:
        job = jobs.get_job(job_id)
        current_step = job.get("current_step", "prepare") if job else "prepare"
        jobs.fail_job(job_id, current_step, str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    try:
        services.milvus.init_collection()
        results = services.milvus.query(
            output_fields=["filename", "file_type"],
            limit=10000,
        )

        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_count": 0,
                }
            file_stats[filename]["chunk_count"] += 1

        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document list: {str(e)}")


#: One response may not carry more than this many chunks. A document is normally a few
#: hundred; the ceiling is here so a pathological one cannot build a response large
#: enough to matter, and `total` still reports the truth so the UI can say it was cut.
_CHUNK_PAGE_LIMIT = 2000

#: What the inspector reads. `dense_embedding` and `sparse_embedding` are deliberately
#: absent — they are most of a chunk's bytes and none of its meaning.
_CHUNK_FIELDS = [
    "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level", "chunk_idx",
    "page_number", "modality", "text", "asset_ids",
]


def _chunk_asset_ids(raw) -> list:
    """`asset_ids` is stored as a JSON array in a VARCHAR. A malformed one is not worth
    failing a whole inspection over, so it reads as no assets."""
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        parsed = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _chunk_info(row: dict) -> ChunkInfo:
    text = str(row.get("text") or "")
    return ChunkInfo(
        chunk_id=str(row.get("chunk_id") or ""),
        parent_chunk_id=str(row.get("parent_chunk_id") or ""),
        root_chunk_id=str(row.get("root_chunk_id") or ""),
        chunk_level=int(row.get("chunk_level") or 0),
        chunk_idx=int(row.get("chunk_idx") or 0),
        page_number=int(row.get("page_number") or 0),
        modality=str(row.get("modality") or "text"),
        text=text,
        char_count=len(text),
        asset_ids=_chunk_asset_ids(row.get("asset_ids")),
    )


@router.get("/documents/{filename}/chunks", response_model=DocumentChunkListResponse)
async def list_document_chunks(
    filename: str,
    q: str = "",
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    """Every chunk a document was indexed into, for the admin inspector.

    The corpus as retrieval sees it, which is the one thing no other view shows. The
    document list gives a count; this gives the chunks themselves, their hierarchy
    (`chunk_level` with `parent_chunk_id`/`root_chunk_id`) and the metadata a developer
    needs to explain why a question did or did not find something.

    `q` MARKS rather than removes. Each chunk comes back with `matched`, and the document
    stays whole: the view that draws the corpus as a bordered document needs the whole
    structure to draw, and a response filtered down to the hits would have it rendering
    fragments of a document and calling them the document. The list view narrows itself
    on the flag.

    Matching is on FOLDED text — `text_matching.fold`, the same folding the sparse
    retrieval lane keys on. That is what makes it usable on this corpus: an admin typing
    مدرسة finds مدرسه, typing without diacritics finds text with them, and case never
    matters. Folding on the server rather than in the browser keeps one implementation of
    it; a second one in TypeScript would drift from this one the first time either was
    edited, and the filter would quietly stop agreeing with retrieval.

    BOTH stores are read, and that is not an optimisation. Only leaf chunks are
    vectorised: `_process_upload_job` writes levels 1 and 2 to `parent_chunks` and level 3
    to Milvus. Reading Milvus alone returns every leaf with a `parent_chunk_id` naming a
    chunk that is not in the response — measured on the shipped corpus, 175 of 175 — so
    the tree would be 175 orphans and the view would be lying about the structure it
    claims to show.

    Ordered by level then index, so the flat list reads the way the document does and the
    tree can be built from it without a second pass.
    """
    try:
        services.milvus.init_collection()
        rows = services.milvus.query_all(
            filter_expr=f"filename == {json.dumps(filename, ensure_ascii=False)}",
            output_fields=_CHUNK_FIELDS,
        )
        rows = list(rows) + services.parent_chunks.documents_by_filename(filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read chunks: {exc}")

    chunks = sorted(
        (_chunk_info(row) for row in rows),
        key=lambda chunk: (chunk.chunk_level, chunk.chunk_idx, chunk.chunk_id),
    )
    total = len(chunks)

    needle = fold(q)
    if needle:
        for chunk in chunks:
            chunk.matched = needle in fold(chunk.text)

    truncated = total > _CHUNK_PAGE_LIMIT
    shown = chunks[:_CHUNK_PAGE_LIMIT]
    return DocumentChunkListResponse(
        filename=filename,
        total=total,
        returned=len(shown),
        match_count=sum(1 for chunk in shown if chunk.matched),
        query=q,
        truncated=truncated,
        chunks=shown,
    )


@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    jobs = services.upload_jobs
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")
    if not is_supported_document(filename):
        raise HTTPException(status_code=400, detail=get_profile().user_copy.unsupported_file_type)

    ensure_upload_dir()
    job = jobs.create_job(filename)
    file_path = UPLOAD_DIR / filename

    try:
        jobs.update_step(job["job_id"], "upload", 1, "running", "Saving file to server")
        await save_upload_file(file, file_path)
        jobs.complete_step(job["job_id"], "upload", "File uploaded, waiting for background processing")
    except Exception as e:
        jobs.fail_job(job["job_id"], "upload", f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    background_tasks.add_task(_process_upload_job, services, job["job_id"], str(file_path), filename)
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message="File uploaded, parsing and vectorization in progress in the background",
    )


@router.post("/documents/upload/pair", response_model=DocumentUploadStartResponse)
async def upload_document_pair(
    background_tasks: BackgroundTasks,
    file_ar: UploadFile | None = File(None),
    file_en: UploadFile | None = File(None),
    title: str = Form(""),
    pair_id: str = Form(""),
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    """Upload one bilingual entry: an Arabic file, an English file, or one of the two.

    `pair_id` names an EXISTING row to fill in, which is how the second language gets
    added months after the first without re-uploading it. Omitted, a new row is created.
    """
    jobs = services.upload_jobs
    document_pairs = services.document_pairs
    sides = [(ARABIC, file_ar), (ENGLISH, file_en)]
    provided = [(language, upload) for language, upload in sides if upload and upload.filename]
    if not provided:
        raise HTTPException(status_code=400, detail="Upload an Arabic file, an English file, or both")

    if pair_id and not document_pairs.get_pair(pair_id):
        raise HTTPException(status_code=404, detail=f"No document pair {pair_id}")

    for _language, upload in provided:
        if not is_supported_document(upload.filename or ""):
            raise HTTPException(status_code=400, detail=get_profile().user_copy.unsupported_file_type)

    # Both halves cannot be the same file: the row would claim one document as its own
    # translation, and pair_store would then detach it from one side as it attached the
    # other, leaving a row that silently lost a language.
    names = [upload.filename for _language, upload in provided]
    if len(names) == 2 and names[0] == names[1]:
        raise HTTPException(
            status_code=400,
            detail="The Arabic and English files must be different documents",
        )

    ensure_upload_dir()
    job = jobs.create_job(", ".join(names))
    saved = []
    try:
        jobs.update_step(job["job_id"], "upload", 1, "running", "Saving files to server")
        for language, upload in provided:
            path = UPLOAD_DIR / upload.filename
            await save_upload_file(upload, path)
            saved.append((language, str(path), upload.filename))
        jobs.complete_step(
            job["job_id"], "upload", f"{len(saved)} file(s) uploaded, waiting for background processing"
        )
    except Exception as e:
        jobs.fail_job(job["job_id"], "upload", f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    background_tasks.add_task(
        _process_pair_upload_job, services, job["job_id"], pair_id, title.strip(), saved
    )
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=", ".join(names),
        message="Files uploaded, parsing and vectorization in progress in the background",
    )


@router.get("/documents/pairs", response_model=DocumentPairListResponse)
async def list_document_pairs(
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    """The bilingual view of the corpus: one row per entry, up to two files on it.

    Chunk counts come from Milvus, which remains the only place that knows how much of a
    document was actually indexed. A file present on a row but absent from Milvus shows
    zero — worth surfacing rather than hiding, because it means an ingest failed after
    the row was written.
    """
    document_pairs = services.document_pairs
    try:
        services.milvus.init_collection()
        counts: dict = {}
        for item in services.milvus.query(output_fields=["filename"], limit=10000):
            name = item.get("filename", "")
            counts[name] = counts.get(name, 0) + 1

        pairs = document_pairs.list_pairs()
        rows = [
            DocumentPairInfo(
                pair_id=pair.pair_id,
                title=pair.title,
                filename_ar=pair.filename_ar,
                filename_en=pair.filename_en,
                paired=pair.paired,
                chunk_count_ar=counts.get(pair.filename_ar, 0) if pair.filename_ar else 0,
                chunk_count_en=counts.get(pair.filename_en, 0) if pair.filename_en else 0,
            )
            for pair in pairs
        ]
        # Files indexed before this feature existed, or uploaded through the
        # single-file route, belong to no row. They are still part of the corpus and
        # still answer questions, so the list has to show them rather than pretend the
        # corpus is only what has been paired.
        claimed = {pair.filename_ar for pair in pairs} | {pair.filename_en for pair in pairs}
        unpaired = [
            DocumentPairInfo(
                pair_id="", title=name, filename_ar="", filename_en=name,
                paired=False, chunk_count_ar=0, chunk_count_en=count, unassigned=True,
            )
            for name, count in sorted(counts.items())
            if name and name not in claimed
        ]
        return DocumentPairListResponse(pairs=rows + unpaired)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document pairs: {str(e)}")


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(
    job_id: str,
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    job = services.upload_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Upload job does not exist or has expired")
    return DocumentUploadJobResponse(**job)


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    jobs = services.upload_jobs.list_jobs()
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    jobs = services.delete_jobs
    job = jobs.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="Waiting to delete",
        completion_step="parent_store",
    )
    jobs.update_step(job["job_id"], "prepare", 1, "running", "Delete job submitted")
    background_tasks.add_task(_process_delete_job, services, job["job_id"], filename)
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"Deleting {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(
    job_id: str,
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    job = services.delete_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Delete job does not exist or has expired")
    return DocumentDeleteJobResponse(**job)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    try:
        filename = file.filename or ""
        if not filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty")
        if not is_supported_document(filename):
            raise HTTPException(status_code=400, detail=get_profile().user_copy.unsupported_file_type)

        ensure_upload_dir()

        # Clean up existing document with the same name to preserve consistency
        services.document_remover.remove(filename)

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            new_docs = services.document_loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"Document processing failed: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="Document processing failed: could not extract content")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="Document processing failed: no retrievable leaf chunks were generated")

        services.parent_chunks.upsert_documents(parent_docs)
        services.milvus_writer.write_documents(leaf_docs)

        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=(
                f"Successfully uploaded and processed {filename}: {len(leaf_docs)} leaf chunks, "
                f"{len(parent_docs)} parent chunks (stored in PostgreSQL)"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(
    filename: str,
    _: User = Depends(require_admin),
    services: Services = Depends(get_services),
):
    try:
        chunks_deleted = services.document_remover.remove(filename)
        _detach_from_pair(services.document_pairs, filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=chunks_deleted,
            message=f"Successfully deleted vector data for document {filename} (local file retained)",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
