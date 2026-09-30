"""The document corpus over HTTP, for administrators.

Each route validates the request, calls one of the three document services, and shapes the
response. What a document upload, a pair, a delete or an inspection DOES is in
`backend/application/services/documents.py`; which status a failure becomes is in
`backend/api/errors.py`. Routes are registered in the order they always were, because FastAPI
matches paths in registration order.
"""

from functools import partial

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from backend.agent.chat.language import ARABIC, ENGLISH
from backend.api.deps import document_catalogue, document_ingestion, document_removal
from backend.api.resources import save_upload_file
from backend.api.schemas.documents import (
    AssetInfo,
    ChunkInfo,
    DocumentAssetListResponse,
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
from backend.api.validation import require_assets_enabled, require_supported_document
from backend.application.services import DocumentCatalogue, DocumentIngestion, DocumentRemoval
from backend.assets.delivery import asset_url_path
from backend.db.models import User
from backend.domain.errors import InvalidInput, operation
from backend.infra.auth import require_admin

router = APIRouter(tags=["documents"])


def _asset_info(dossier, chunk_ids_by_asset) -> AssetInfo:
    """One dossier, flattened for review.

    The nesting is real - text surface, provenance, blob, source are separate objects - but a
    reviewer reads one image at a time, and a shape that mirrored the storage would make the UI
    walk four objects to render one row.
    """
    extraction = dossier.extraction
    text = extraction.text if extraction else None
    provenance = extraction.provenance if extraction else None
    return AssetInfo(
        asset_id=dossier.asset_id,
        sha256=dossier.sha256,
        page_number=dossier.source.page_number,
        status=dossier.status.value,
        role=dossier.role.value,
        tier=dossier.tier.value,
        indexable=dossier.is_indexable,
        caption=(text.caption if text else ""),
        description=(text.description if text else ""),
        transcription=(text.transcription if text else ""),
        tags=list(text.tags) if text else [],
        model_used=(provenance.model_used if provenance else ""),
        confidence=(provenance.confidence if provenance else 0.0),
        needs_review=bool(provenance.needs_review) if provenance else False,
        error=(provenance.error if provenance else ""),
        width=dossier.blob.width,
        height=dossier.blob.height,
        byte_size=dossier.blob.byte_size,
        content_type=dossier.blob.content_type,
        url=asset_url_path(dossier.asset_id),
        chunk_ids=chunk_ids_by_asset.get(dossier.asset_id, []),
    )


def _pair_uploads(
    file_ar: UploadFile | None,
    file_en: UploadFile | None,
    pair_id: str,
    ingestion: DocumentIngestion,
) -> list[tuple[str, UploadFile]]:
    """The files of a pair request, checked in the order the route has always checked them.

    The order is part of the contract: a request that is wrong in two ways is told the first.
    """
    sides = [(ARABIC, file_ar), (ENGLISH, file_en)]
    provided = [(language, upload) for language, upload in sides if upload and upload.filename]
    if not provided:
        raise InvalidInput("Upload an Arabic file, an English file, or both")

    ingestion.require_pair(pair_id)

    for _language, upload in provided:
        require_supported_document(upload.filename or "")

    # Both halves cannot be the same file: the row would claim one document as its own
    # translation, and pair_store would then detach it from one side as it attached the other,
    # leaving a row that silently lost a language.
    names = [upload.filename for _language, upload in provided]
    if len(names) == 2 and names[0] == names[1]:
        raise InvalidInput("The Arabic and English files must be different documents")
    return provided


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    _: User = Depends(require_admin),
    catalogue: DocumentCatalogue = Depends(document_catalogue),
):
    return DocumentListResponse(
        documents=[DocumentInfo(**stats) for stats in catalogue.list_documents()]
    )


@router.get("/documents/{filename}/chunks", response_model=DocumentChunkListResponse)
async def list_document_chunks(
    filename: str,
    q: str = "",
    _: User = Depends(require_admin),
    catalogue: DocumentCatalogue = Depends(document_catalogue),
):
    """Every chunk a document was indexed into, for the admin inspector, with `q` marked.

    See `DocumentCatalogue.list_chunks` for why both stores are read and why matching is on
    folded text.
    """
    listing = catalogue.list_chunks(filename, q)
    return DocumentChunkListResponse(
        filename=listing.filename,
        total=listing.total,
        returned=listing.returned,
        match_count=listing.match_count,
        query=listing.query,
        truncated=listing.truncated,
        chunks=[ChunkInfo(**chunk) for chunk in listing.chunks],
    )


@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
    ingestion: DocumentIngestion = Depends(document_ingestion),
):
    filename = require_supported_document(file.filename or "")
    accepted = await ingestion.accept_upload(filename, partial(save_upload_file, file))
    background_tasks.add_task(
        ingestion.ingest, accepted.job_id, str(accepted.path), accepted.filename
    )
    return DocumentUploadStartResponse(
        job_id=accepted.job_id,
        filename=accepted.filename,
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
    ingestion: DocumentIngestion = Depends(document_ingestion),
):
    """Upload one bilingual entry: an Arabic file, an English file, or one of the two.

    `pair_id` names an EXISTING row to fill in, which is how the second language gets added
    months after the first without re-uploading it. Omitted, a new row is created.
    """
    provided = _pair_uploads(file_ar, file_en, pair_id, ingestion)
    accepted = await ingestion.accept_pair(
        [
            (language, upload.filename, partial(save_upload_file, upload))
            for language, upload in provided
        ]
    )
    background_tasks.add_task(
        ingestion.ingest_pair, accepted.job_id, pair_id, title.strip(), accepted.sides
    )
    return DocumentUploadStartResponse(
        job_id=accepted.job_id,
        filename=accepted.names,
        message="Files uploaded, parsing and vectorization in progress in the background",
    )


@router.get("/documents/{filename}/assets", response_model=DocumentAssetListResponse)
async def list_document_assets(
    filename: str,
    _: User = Depends(require_admin),
    catalogue: DocumentCatalogue = Depends(document_catalogue),
):
    """Every image of a document with what extraction made of it, for manual review.

    The counterpart to `/documents/{filename}/chunks`: that view shows the corpus as retrieval
    holds it, this one shows where a figure chunk's text CAME FROM - the description, the
    transcription, the model, its confidence, the error that explains a failure. The public
    asset routes answer with `AssetReference`, which carries none of it and should not.
    `needs_review` is what the extractor stored, not something computed here.
    """
    listing = catalogue.list_assets(filename)
    assets = [_asset_info(dossier, listing.chunk_ids_by_asset) for dossier in listing.dossiers]
    return DocumentAssetListResponse(
        filename=filename,
        assets=assets,
        total=len(assets),
        needs_review_count=sum(1 for asset in assets if asset.needs_review),
    )


@router.post("/documents/assets/{asset_id:path}/reviewed", response_model=AssetInfo)
async def mark_asset_reviewed(
    asset_id: str,
    _: User = Depends(require_admin),
    catalogue: DocumentCatalogue = Depends(document_catalogue),
):
    """Record that an admin looked at this extraction and accepted it.

    `needs_review` was raised by the extractor and until this existed nothing could lower it -
    a flag that only goes up is a permanent label, not a queue. The response is the asset as it
    now reads, so the caller updates from what was stored rather than assuming the write did
    what it asked.
    """
    require_assets_enabled()
    return _asset_info(catalogue.mark_reviewed(asset_id), {})


@router.get("/documents/pairs", response_model=DocumentPairListResponse)
async def list_document_pairs(
    _: User = Depends(require_admin),
    catalogue: DocumentCatalogue = Depends(document_catalogue),
):
    """The bilingual view of the corpus: one row per entry, up to two files on it."""
    return DocumentPairListResponse(
        pairs=[DocumentPairInfo(**row) for row in catalogue.list_pairs()]
    )


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(
    job_id: str,
    _: User = Depends(require_admin),
    ingestion: DocumentIngestion = Depends(document_ingestion),
):
    return DocumentUploadJobResponse(**ingestion.upload_job(job_id))


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(
    _: User = Depends(require_admin),
    ingestion: DocumentIngestion = Depends(document_ingestion),
):
    return [DocumentUploadJobResponse(**job) for job in ingestion.upload_jobs()]


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    _: User = Depends(require_admin),
    removal: DocumentRemoval = Depends(document_removal),
):
    job = removal.start(filename)
    background_tasks.add_task(removal.remove, job["job_id"], filename)
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"Deleting {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(
    job_id: str,
    _: User = Depends(require_admin),
    removal: DocumentRemoval = Depends(document_removal),
):
    return DocumentDeleteJobResponse(**removal.delete_job(job_id))


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
    ingestion: DocumentIngestion = Depends(document_ingestion),
):
    filename = require_supported_document(file.filename or "")
    # Read before anything is replaced: a request whose body cannot be read must not have
    # deleted the document it was meant to replace.
    with operation("Document upload failed"):
        content = await file.read()
    ingested = ingestion.ingest_now(filename, content)
    return DocumentUploadResponse(
        filename=ingested.filename,
        chunks_processed=ingested.leaf_chunks,
        message=(
            f"Successfully uploaded and processed {ingested.filename}: "
            f"{ingested.leaf_chunks} leaf chunks, "
            f"{ingested.parent_chunks} parent chunks (stored in PostgreSQL)"
        ),
    )


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(
    filename: str,
    _: User = Depends(require_admin),
    removal: DocumentRemoval = Depends(document_removal),
):
    chunks_deleted = removal.remove_now(filename)
    return DocumentDeleteResponse(
        filename=filename,
        chunks_deleted=chunks_deleted,
        message=f"Successfully deleted vector data for document {filename} (local file retained)",
    )
