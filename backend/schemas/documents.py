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

    #: A nested bar inside this step. Figure extraction reports here: it runs INSIDE
    #: `parse`, so it cannot be a step of its own without appearing to finish while its
    #: parent is still going. `sub_total` of 0 means draw nothing.
    sub_label: str = ""
    sub_done: int = 0
    sub_total: int = 0


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


class AssetInfo(BaseModel):
    """One image of a document, as the admin review view shows it.

    This is deliberately NOT `AssetReference`. That one is the public asset contract a
    chat client consumes — caption, alt text, tags — and widening it to carry a model's
    confidence and its error text would put an operational detail on a surface parents
    read. This is the other half of the dossier, for the admin who has to decide whether
    an extraction is good enough to leave in the index.
    """

    asset_id: str
    sha256: str = ""
    page_number: int = 0
    #: "extracted" | "failed" | "skipped" | "pending" | "stale".
    status: str = ""
    #: "figure" | "entity" | "decorative".
    role: str = ""
    #: What extraction was ASKED for: "simple" | "complex" | "layout" | "drop".
    tier: str = ""
    #: Whether this asset produces a retrievable chunk at all. A decorative image or a
    #: failed extraction is stored and visible here, but is not in the index.
    indexable: bool = False

    # -- the retrieval surface, which is the thing actually being reviewed -------
    caption: str = ""
    description: str = ""
    transcription: str = ""
    tags: List[str] = []

    # -- how it was produced, which is how an admin judges it --------------------
    model_used: str = ""
    confidence: float = 0.0
    #: Already set on every extraction, by the extractor: a vision confidence below the
    #: profile's `escalate_below_confidence`, or a heuristic run that recovered no text.
    needs_review: bool = False
    #: Why an extraction failed, when it did. Empty otherwise.
    error: str = ""

    width: int = 0
    height: int = 0
    byte_size: int = 0
    content_type: str = ""
    #: Authenticated GET returns the image itself, so the reviewer can compare the
    #: transcription against the picture it claims to describe.
    url: str = ""

    #: The chunks this image produced, filled by the caller. Empty when the document's
    #: chunks could not be read — an asset list is still worth showing without them.
    chunk_ids: List[str] = []


class DocumentAssetListResponse(BaseModel):
    filename: str
    assets: List[AssetInfo]
    total: int = 0
    #: How many carry `needs_review`, so the UI can lead with the number that matters
    #: without counting a list it may have truncated.
    needs_review_count: int = 0
