from typing import List, Optional

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class DocumentPairInfo(BaseModel):
    """One knowledge-base entry as the admin UI shows it: a row with two language slots.

    `paired` is what retrieval actually keys on — only a row with BOTH sides filled
    causes one half to be excluded from an answer. A row with one side is a complete,
    normal entry that answers questions in either language.
    """

    pair_id: str
    title: str
    filename_ar: str = ""
    filename_en: str = ""
    paired: bool = False
    chunk_count_ar: int = 0
    chunk_count_en: int = 0
    #: A file indexed outside the pairing system — uploaded through the single-file
    #: route, or before it existed. Shown so the list describes the whole corpus, and
    #: flagged so the UI can offer to file it into a row.
    unassigned: bool = False


class DocumentPairListResponse(BaseModel):
    pairs: List[DocumentPairInfo]


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_processed: int
    message: str


class DocumentUploadStartResponse(BaseModel):
    job_id: str
    filename: str
    message: str


class UploadStepInfo(BaseModel):
    key: str
    label: str
    percent: int
    status: str
    message: str = ""


class DocumentUploadJobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    current_step: str
    message: str
    total_chunks: int = 0
    processed_chunks: int = 0
    error: Optional[str] = None
    created_at: str
    updated_at: str
    steps: List[UploadStepInfo]


class DocumentDeleteStartResponse(BaseModel):
    job_id: str
    filename: str
    message: str


class DocumentDeleteJobResponse(DocumentUploadJobResponse):
    pass


class DocumentDeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    message: str
