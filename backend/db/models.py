from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.infra.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_user_session"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_ref_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    rag_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    session = relationship("ChatSession", back_populates="messages")


class AssetExtraction(Base):
    """Content-derived extraction, cached globally by image content.

    Primary key is (sha256, profile, dossier_version): the same bytes extracted under
    a different domain profile or a newer schema is a genuinely different result, but
    the same bytes seen in a thousand documents is one row. This table is where the
    ingest cost saving actually lives.

    `payload` holds the full ExtractionPayload as JSON rather than normalised columns,
    because the extraction schema is versioned and expected to grow; the columns
    beside it exist only to make the backfill job's scans indexable.
    """

    __tablename__ = "asset_extractions"
    __table_args__ = (
        Index("ix_asset_extractions_version", "dossier_version"),
    )

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile: Mapped[str] = mapped_column(String(64), primary_key=True)
    dossier_version: Mapped[int] = mapped_column(Integer, primary_key=True)

    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    model_used: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentAsset(Base):
    """One OCCURRENCE of an image in a document.

    Many rows here can share one `AssetExtraction` via sha256. The dossier JSON is the
    source of truth; the scalar columns duplicate the fields the system filters and
    joins on (delete-by-filename, backfill-by-version, indexable-by-status), because
    those queries must not degrade into JSON scans as the corpus grows.
    """

    __tablename__ = "document_assets"
    __table_args__ = (
        Index("ix_document_assets_filename_page", "filename", "page_number"),
        Index("ix_document_assets_status_version", "status", "dossier_version"),
    )

    asset_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    profile: Mapped[str] = mapped_column(String(64), default="base", nullable=False)
    dossier_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    filename: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    role: Mapped[str] = mapped_column(String(20), default="figure", nullable=False)
    tier: Mapped[str] = mapped_column(String(20), default="simple", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)

    storage_uri: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    content_type: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    height: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    dossier: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class EntityAttribute(Base):
    """Attribute index for entity assets, in entity-attribute-value form.

    EAV rather than a column per attribute, because the vocabulary is declared per
    domain profile: a column-per-attribute design would need a schema migration every
    time a domain added a facet, which is exactly the coupling the profile system
    exists to remove.

    Values are split across three typed columns so range queries on numbers stay
    index-friendly instead of degrading into casts over text.
    """

    __tablename__ = "entity_attributes"
    __table_args__ = (
        # The two access patterns: facet lookup ("all reds") and per-asset fetch.
        Index("ix_entity_attributes_name_text", "name", "value_text"),
        Index("ix_entity_attributes_name_number", "name", "value_number"),
        UniqueConstraint("asset_id", "name", "value_key", name="uq_entity_attribute_value"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String(512), index=True, nullable=False)
    profile: Mapped[str] = mapped_column(String(64), default="base", nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)

    # Multi-valued attributes produce one row per value; value_key makes the
    # (asset, name, value) triple unique so re-ingest is idempotent.
    value_key: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    value_text: Mapped[str | None] = mapped_column(String(255), nullable=True)
    value_number: Mapped[float | None] = mapped_column(Float, nullable=True)
    value_bool: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


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
    asset_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
