import os

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

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
from backend.db.models import User
from backend.infra.auth import require_admin
from backend.profiles import get_profile
from backend.jobs import DELETE_STEPS, delete_job_manager, upload_job_manager
from backend.schemas import (
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
)

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


def _process_delete_job(job_id: str, filename: str) -> None:
    failed_step = "prepare"
    try:
        chunks_deleted = delete_document_transactionally(filename, delete_job_manager, job_id)
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

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=chunks_deleted,
            message=f"Successfully deleted vector data for document {filename} (local file retained)",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
