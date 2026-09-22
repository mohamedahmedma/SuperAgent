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


class ChunkInfo(BaseModel):
    """One indexed chunk, as the admin inspector shows it.

    Everything here is read straight out of Milvus rather than recomputed, because the
    point of the inspector is to show what retrieval will actually see. A field that
    disagreed with the index would be worse than not showing it.
    """

    chunk_id: str
    #: The hierarchy. A leaf names its parent and its root; an L1 chunk has neither, and
    #: that absence is what the tree uses to find its roots.
    parent_chunk_id: str = ""
    root_chunk_id: str = ""
    chunk_level: int = 0
    chunk_idx: int = 0
    page_number: int = 0
    #: "text" | "figure" | "table" — what the chunk was made from.
    modality: str = "text"
    text: str = ""
    #: Counted server side so the UI never has to agree with Python about what a
    #: character is: the size bound this corpus is chunked against is in characters.
    char_count: int = 0
    asset_ids: List[str] = []
    #: Whether the filter matched this chunk. The filter MARKS rather than removes, so
    #: the document view can show the whole document with the hits lit up inside it —
    #: filtering the response down to the hits would leave that view rendering fragments
    #: of a structure and calling it the document.
    matched: bool = False


class DocumentChunkListResponse(BaseModel):
    filename: str
    #: Chunks the document has, and how many are in this response — they differ only when
    #: the document is larger than one response may carry.
    total: int = 0
    returned: int = 0
    #: How many chunks the filter matched, so the UI can say "12 of 229 match" without
    #: counting flags itself.
    match_count: int = 0
    #: The filter as asked, echoed back. The server folds it before matching and the UI
    #: shows what the user typed, so the two must not be confused for each other.
    query: str = ""
    #: True when the document has more chunks than one response may carry. The count
    #: above is still the real total, so the UI can say so rather than quietly show less.
    truncated: bool = False
    chunks: List[ChunkInfo] = []


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
