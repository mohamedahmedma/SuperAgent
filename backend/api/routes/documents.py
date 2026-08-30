import logging
import os

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from backend.api.resources import (
    UPLOAD_DIR,
    ensure_upload_dir,
    is_supported_document,
    loader,
    milvus_manager,
    milvus_writer,
    parent_chunk_store,
    delete_document_transactionally,
    save_upload_file,
)
from backend.chat.language import ARABIC, ENGLISH
from backend.db.models import User
import backend.indexing.language_check as language_check
import backend.indexing.pair_store as pair_store
from backend.infra.auth import require_admin
from backend.profiles import get_profile
from backend.jobs import DELETE_STEPS, delete_job_manager, upload_job_manager
from backend.schemas import (
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


def _process_upload_job(job_id: str, file_path: str, filename: str) -> None:
    failed_step = "cleanup"
    try:
        upload_job_manager.complete_step(job_id, "upload", "File saved to server")

        failed_step = "cleanup"
        upload_job_manager.update_step(job_id, "cleanup", 10, "running", "Cleaning up old document with the same name")
        delete_document_transactionally(filename)
        upload_job_manager.complete_step(job_id, "cleanup", "Old version cleanup complete")

        failed_step = "parse"
        upload_job_manager.update_step(job_id, "parse", 5, "running", "Parsing document and performing three-level chunking")
        new_docs = loader.load_document(file_path, filename)
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
        upload_job_manager.complete_step(
            job_id,
            "parse",
            f"Parsing complete: {len(parent_docs)} parent chunks, {len(leaf_docs)} leaf chunks{figure_note}",
        )

        failed_step = "parent_store"
        upload_job_manager.update_step(job_id, "parent_store", 20, "running", "Writing parent chunks")
        parent_chunk_store.upsert_documents(parent_docs)
        upload_job_manager.complete_step(job_id, "parent_store", f"Parent chunks stored: {len(parent_docs)}")

        failed_step = "vector_store"
        total_leaf = len(leaf_docs)
        upload_job_manager.update_step(
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
            upload_job_manager.update_step(
                job_id,
                "vector_store",
                percent,
                "running",
                f"Vectorizing and storing: {processed} / {total}",
                total_chunks=total,
                processed_chunks=processed,
            )

        milvus_writer.write_documents(leaf_docs, progress_callback=_on_vector_progress)
        upload_job_manager.complete_step(job_id, "vector_store", f"Vectorization and storage complete: {total_leaf} leaf chunks")
        upload_job_manager.complete_job(job_id, f"Successfully uploaded and processed {filename}")
    except Exception as e:
        upload_job_manager.fail_job(job_id, failed_step, str(e))


def _parse_side(language: str, file_path: str, filename: str) -> list:
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


def _process_pair_upload_job(job_id: str, pair_id: str, title: str, sides: list) -> None:
    """Ingest one bilingual entry: up to two files, one per language.

    `sides` is a list of (language, file_path, filename). One entry is a single-language
    row, which is a normal and permanent state — a document that exists only in English
    still answers Arabic questions (see pair_store.superseded_filenames).

    Ordered so that everything which can REJECT the upload happens before anything that
    writes: both files are parsed and language-checked first, and only then is either
    indexed. That is what keeps a rejected pair from leaving half a document behind.
    """
    failed_step = "parse"
    try:
        upload_job_manager.complete_step(job_id, "upload", "Files saved to server")

        failed_step = "parse"
        parsed = []
        for index, (language, file_path, filename) in enumerate(sides, start=1):
            upload_job_manager.update_step(
                job_id, "parse", round(index * 100 / (len(sides) + 1)), "running",
                f"Parsing {filename} ({language})",
            )
            parsed.append((language, filename, _parse_side(language, file_path, filename)))

        leaf_total = sum(
            1 for _, _, docs in parsed for d in docs if int(d.get("chunk_level", 0) or 0) == 3
        )
        if not leaf_total:
            raise ValueError("no retrievable leaf chunks were generated")
        upload_job_manager.complete_step(
            job_id, "parse",
            f"Parsed {len(parsed)} file(s), {leaf_total} leaf chunks",
        )

        # Only now does anything get written. Replacing a same-named document is part of
        # writing, not of validation, so it happens after both files are known good.
        failed_step = "cleanup"
        upload_job_manager.update_step(job_id, "cleanup", 10, "running", "Cleaning up old versions")
        for _, filename, _ in parsed:
            delete_document_transactionally(filename)
        upload_job_manager.complete_step(job_id, "cleanup", "Old version cleanup complete")

        failed_step = "parent_store"
        upload_job_manager.update_step(job_id, "parent_store", 20, "running", "Writing parent chunks")
        parent_written = 0
        for _, _, docs in parsed:
            parents = [d for d in docs if int(d.get("chunk_level", 0) or 0) in (1, 2)]
            parent_chunk_store.upsert_documents(parents)
            parent_written += len(parents)
        upload_job_manager.complete_step(job_id, "parent_store", f"Parent chunks stored: {parent_written}")

        failed_step = "vector_store"
        written = 0
        upload_job_manager.update_step(
            job_id, "vector_store", 0, "running", f"Vectorizing and storing: 0 / {leaf_total}",
            total_chunks=leaf_total, processed_chunks=0,
        )
        for _, _, docs in parsed:
            leaves = [d for d in docs if int(d.get("chunk_level", 0) or 0) == 3]

            def _on_progress(processed: int, _total: int, base: int = written) -> None:
                done = base + processed
                upload_job_manager.update_step(
                    job_id, "vector_store", round(done * 100 / leaf_total), "running",
                    f"Vectorizing and storing: {done} / {leaf_total}",
                    total_chunks=leaf_total, processed_chunks=done,
                )

            milvus_writer.write_documents(leaves, progress_callback=_on_progress)
            written += len(leaves)
        upload_job_manager.complete_step(
            job_id, "vector_store", f"Vectorization and storage complete: {written} leaf chunks"
        )

        # The row is written LAST. A file that failed to index must not be recorded as
        # this entry's Arabic or English half, or routing would exclude the twin in
        # favour of a document that is not in the corpus.
        for language, filename, _ in parsed:
            pair_id = pair_store.attach(pair_id, language, filename, title=title)["pair_id"]

        names = ", ".join(filename for _, filename, _ in parsed)
        upload_job_manager.complete_job(job_id, f"Successfully uploaded and processed {names}")
    except Exception as e:
        upload_job_manager.fail_job(job_id, failed_step, str(e))


def _detach_from_pair(filename: str) -> None:
    """Take a deleted file off whatever pair row holds it.

    Never raises: the document is already gone from the index by the time this runs, so
    a failure here must not turn a completed delete into a failed job. The cost of it
    failing is a row naming a file that no longer exists, which shows as a zero chunk
    count in the pair list rather than as a wrong answer — routing only ever excludes a
    twin, so a stale row can hide a document but never invent one.
    """
    try:
        pair_store.detach(filename)
    except Exception:  # pragma: no cover - a bookkeeping failure must not fail a delete
        logger.exception("could not detach %s from its document pair", filename)


def _process_delete_job(job_id: str, filename: str) -> None:
    failed_step = "prepare"
    try:
        chunks_deleted = delete_document_transactionally(filename, delete_job_manager, job_id)
        # Clear the file off its pair row. Done HERE and not in
        # delete_document_transactionally, which upload also calls to replace a
        # same-named document — detaching there would silently unpair a document every
        # time somebody re-uploaded one side of it.
        _detach_from_pair(filename)
        delete_job_manager.complete_job(job_id, f"Deleted {filename}, {chunks_deleted} vector records removed")
    except Exception as e:
        job = delete_job_manager.get_job(job_id)
        current_step = job.get("current_step", "prepare") if job else "prepare"
        delete_job_manager.fail_job(job_id, current_step, str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    try:
        milvus_manager.init_collection()
        results = milvus_manager.query(
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


@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
):
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")
    if not is_supported_document(filename):
        raise HTTPException(status_code=400, detail=get_profile().user_copy.unsupported_file_type)

    ensure_upload_dir()
    job = upload_job_manager.create_job(filename)
    file_path = UPLOAD_DIR / filename

    try:
        upload_job_manager.update_step(job["job_id"], "upload", 1, "running", "Saving file to server")
        await save_upload_file(file, file_path)
        upload_job_manager.complete_step(job["job_id"], "upload", "File uploaded, waiting for background processing")
    except Exception as e:
        upload_job_manager.fail_job(job["job_id"], "upload", f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    background_tasks.add_task(_process_upload_job, job["job_id"], str(file_path), filename)
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
):
    """Upload one bilingual entry: an Arabic file, an English file, or one of the two.

    `pair_id` names an EXISTING row to fill in, which is how the second language gets
    added months after the first without re-uploading it. Omitted, a new row is created.
    """
    sides = [(ARABIC, file_ar), (ENGLISH, file_en)]
    provided = [(language, upload) for language, upload in sides if upload and upload.filename]
    if not provided:
        raise HTTPException(status_code=400, detail="Upload an Arabic file, an English file, or both")

    if pair_id and not pair_store.get_pair(pair_id):
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
    job = upload_job_manager.create_job(", ".join(names))
    saved = []
    try:
        upload_job_manager.update_step(job["job_id"], "upload", 1, "running", "Saving files to server")
        for language, upload in provided:
            path = UPLOAD_DIR / upload.filename
            await save_upload_file(upload, path)
            saved.append((language, str(path), upload.filename))
        upload_job_manager.complete_step(
            job["job_id"], "upload", f"{len(saved)} file(s) uploaded, waiting for background processing"
        )
    except Exception as e:
        upload_job_manager.fail_job(job["job_id"], "upload", f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    background_tasks.add_task(
        _process_pair_upload_job, job["job_id"], pair_id, title.strip(), saved
    )
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=", ".join(names),
        message="Files uploaded, parsing and vectorization in progress in the background",
    )


@router.get("/documents/pairs", response_model=DocumentPairListResponse)
async def list_document_pairs(_: User = Depends(require_admin)):
    """The bilingual view of the corpus: one row per entry, up to two files on it.

    Chunk counts come from Milvus, which remains the only place that knows how much of a
    document was actually indexed. A file present on a row but absent from Milvus shows
    zero — worth surfacing rather than hiding, because it means an ingest failed after
    the row was written.
    """
    try:
        milvus_manager.init_collection()
        counts: dict = {}
        for item in milvus_manager.query(output_fields=["filename"], limit=10000):
            name = item.get("filename", "")
            counts[name] = counts.get(name, 0) + 1

        rows = [
            DocumentPairInfo(
                **row,
                chunk_count_ar=counts.get(row["filename_ar"], 0) if row["filename_ar"] else 0,
                chunk_count_en=counts.get(row["filename_en"], 0) if row["filename_en"] else 0,
            )
            for row in pair_store.list_pairs()
        ]
        # Files indexed before this feature existed, or uploaded through the
        # single-file route, belong to no row. They are still part of the corpus and
        # still answer questions, so the list has to show them rather than pretend the
        # corpus is only what has been paired.
        claimed = {row["filename_ar"] for row in pair_store.list_pairs()} | {
            row["filename_en"] for row in pair_store.list_pairs()
        }
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
async def get_upload_job(job_id: str, _: User = Depends(require_admin)):
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Upload job does not exist or has expired")
    return DocumentUploadJobResponse(**job)


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_: User = Depends(require_admin)):
    jobs = upload_job_manager.list_jobs()
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    _: User = Depends(require_admin),
):
    job = delete_job_manager.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="Waiting to delete",
        completion_step="parent_store",
    )
    delete_job_manager.update_step(job["job_id"], "prepare", 1, "running", "Delete job submitted")
    background_tasks.add_task(_process_delete_job, job["job_id"], filename)
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"Deleting {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id: str, _: User = Depends(require_admin)):
    job = delete_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Delete job does not exist or has expired")
    return DocumentDeleteJobResponse(**job)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    try:
        filename = file.filename or ""
        if not filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty")
        if not is_supported_document(filename):
            raise HTTPException(status_code=400, detail=get_profile().user_copy.unsupported_file_type)

        ensure_upload_dir()

        # Clean up existing document with the same name to preserve consistency
        delete_document_transactionally(filename)

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            new_docs = loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"Document processing failed: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="Document processing failed: could not extract content")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="Document processing failed: no retrievable leaf chunks were generated")

        parent_chunk_store.upsert_documents(parent_docs)
        milvus_writer.write_documents(leaf_docs)

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
async def delete_document(filename: str, _: User = Depends(require_admin)):
    try:
        chunks_deleted = delete_document_transactionally(filename)
        _detach_from_pair(filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=chunks_deleted,
            message=f"Successfully deleted vector data for document {filename} (local file retained)",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
