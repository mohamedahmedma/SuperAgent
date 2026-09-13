"""What indexing stores beside the vector index: parent chunks, pairs and the scope catalogue."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.models._columns import timestamp_column
from backend.infra.database import Base


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    chunk_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_type: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parent_chunk_id: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    root_chunk_id: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    chunk_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_idx: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Carried up from the leaf units so that auto-merging a figure hit to its parent
    # keeps the reference to the image, instead of returning text about a picture the
    # caller can no longer show.
    modality: Mapped[str] = mapped_column(String(20), default="text", nullable=False)
    asset_ids: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    updated_at: Mapped[datetime] = timestamp_column(updates=True)


class DocumentPair(Base):
    """One knowledge-base entry, in up to two languages.

    The first thing about documents that this database stores at all. Until now the
    corpus was described entirely by Milvus: `list_documents` queries the collection
    and groups chunks by `filename`, so a "document" was whatever chunks happened to
    carry the same name. That works for a bag of files and cannot express what this
    table is for — that `fees_ar.docx` and `fees_en.docx` are ONE thing said twice.

    A row is a lasting entry, not an upload event. Both sides may be empty at
    different times: an admin uploads the English half in September and drops the
    Arabic half into the same row in January, without re-uploading the first. That is
    the whole reason this is a table rather than a shared id written onto chunks — a
    pair id on a chunk can only be set when the chunk is written, so late pairing
    would mean silently re-indexing a document nobody touched.

    ## What retrieval reads

    Only `filename_ar` and `filename_en`, and only to answer "does this document have a
    twin in the language being asked in" — see `DocumentPairService.superseded_filenames`. A row
    with one side filled is UNPAIRED and stays eligible for every question whatever its
    language, which is what keeps an English-only document answerable in Arabic.
    """

    __tablename__ = "document_pairs"
    __table_args__ = (
        # Both sides are looked up by filename on every upload and delete, to find the
        # row a file belongs to. Not unique: a side is "" when empty, so every half-filled
        # row shares the same value there. `pair_store` enforces one row per filename.
        Index("ix_document_pairs_ar", "filename_ar"),
        Index("ix_document_pairs_en", "filename_en"),
    )

    pair_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    #: What an admin calls this entry, shown in the document list. Defaults to the stem
    #: of whichever file arrived first; it names the ROW, so it survives either side
    #: being replaced.
    title: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    #: "" rather than NULL for an empty side. Every read is a truthiness test, and
    #: three-valued logic in a filter expression is how a half-filled row quietly drops
    #: out of a `!=` comparison.
    filename_ar: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    filename_en: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime] = timestamp_column()
    updated_at: Mapped[datetime] = timestamp_column(updates=True)


class CorpusDigest(Base):
    """What the WHOLE corpus is about, in prose, plus the scope floor derived from it.

    One row per profile. It exists because the scope prompt previously described the
    corpus by joining up to 24 topic labels with commas — "admissions, fees, transport"
    — which tells a model the shelf headings and nothing about what is on them. The
    decision it supports is "is this question this corpus's subject", and labels are
    thin evidence for it. A paragraph is not.

    **Built at ingest, from every section.** Not assembled per request from whatever
    rows happen to be loaded, which is what let the old description drift: it silently
    described the sections it could see rather than the corpus. Regenerated whenever
    `sections_sha256` stops matching, so it cannot describe a corpus that no longer
    exists.

    `floor` is cached here for a different reason. Deriving it is leave-one-out over
    every catalogued question — quadratic in their number, and measured at 64 seconds
    for 40,000 questions even after the blocked rewrite. It is a pure function of the
    question vectors, so recomputing it on every process start bills that to whichever
    user arrives first after a deploy. `floor_sha256` covers the vectors, the embedding
    model and the percentile, so any change to the inputs invalidates it.
    """

    __tablename__ = "corpus_digests"

    profile: Mapped[str] = mapped_column(String(64), primary_key=True)

    # The paragraph shown to the scope model. Prose, not labels.
    paragraph: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Identity of the section set the paragraph was written from, so staleness is a
    # fact rather than a guess about how long ago the corpus changed.
    sections_sha256: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    section_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    floor: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # Covers the question vectors, the embedding model and the percentile — every input
    # to the floor. A mismatch means recompute, never "use it anyway".
    floor_sha256: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    question_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    model_used: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    updated_at: Mapped[datetime] = timestamp_column(updates=True)


class SectionSummary(Base):
    """What one corpus section is ABOUT, and which questions it can answer.

    Built once per section at ingest, not per query. Its purpose is a single decision —
    is an incoming question this corpus's subject at all — which retrieval cannot make
    cheaply, because answering it with retrieval means running the search you were
    trying to avoid.

    Keyed by (chunk_id, profile): the same section summarised under a different domain
    profile is a genuinely different summary, since the profile supplies the topic
    vocabulary and the persona's idea of what counts as in scope.

    `content_sha256` is what makes re-indexing cheap and behaviour constant. A section
    whose text has not changed reuses its summary verbatim, so an unchanged corpus
    re-indexes for free and the same input provably yields the same output — no model
    re-run, no drift in the scope boundary between deployments.

    `answers` is the field that gets embedded, one vector per question, not the summary
    prose. Queries arrive as questions; matching a question against a description of a
    topic throws away most of the signal, and averaging several questions into one
    vector puts the centroid somewhere near none of them.
    """

    __tablename__ = "section_summaries"
    __table_args__ = (
        Index("ix_section_summaries_profile", "profile"),
        Index("ix_section_summaries_filename", "filename"),
    )

    chunk_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    profile: Mapped[str] = mapped_column(String(64), primary_key=True)

    content_sha256: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    chunk_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # One or two sentences. Feeds the scope model's prompt and the corpus catalogue;
    # deliberately NOT the embedded field.
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # The questions this section can answer. Embedded one vector each.
    answers: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    # Those questions' vectors, positionally parallel to `answers`.
    #
    # Persisted rather than embedded at boot for a measured reason: re-embedding 222
    # questions on CPU took 22.4 seconds, and the index builds lazily on first use — so
    # the first user after every deploy would have waited it out. The vectors are a pure
    # function of the questions and the embedding model, so storing them costs a few
    # megabytes and removes the wait entirely. A model change invalidates them, which is
    # what `embedding_model` records.
    question_vectors: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    # Chosen from the profile's frozen vocabulary rather than invented, so the corpus
    # catalogue cannot drift between re-indexes.
    topics: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)

    model_used: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    created_at: Mapped[datetime] = timestamp_column()
    updated_at: Mapped[datetime] = timestamp_column(updates=True)


__all__ = ["CorpusDigest", "DocumentPair", "ParentChunk", "SectionSummary"]
