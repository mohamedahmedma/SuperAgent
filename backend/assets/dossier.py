"""The AssetDossier: everything the system will ever know about one image, resolved
at ingest so that query time only selects and decides.

Two models, deliberately separated:

* `ExtractionPayload` is a pure function of (image bytes, profile, dossier version).
  It holds the expensive VLM/OCR output and is therefore cached globally by sha256 —
  a letterhead appearing in 400 documents is extracted exactly once, ever.
* `AssetDossier` is one OCCURRENCE of an image: an extraction plus where it appeared
  and what it relates to. Two documents sharing a logo produce two dossiers backed by
  one extraction.

Collapsing these into a single record is the mistake that makes image ingest
expensive, because it ties the cost of extraction to the number of occurrences
instead of the number of distinct images.
"""
from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# Bumped whenever the MEANING of an extraction changes — a new required field, a
# reworked prompt, a different transcription format. Rows below this version are
# stale and are picked up by backend/assets/backfill.py.
DOSSIER_VERSION = 1


class AssetRole(str, Enum):
    """What the image IS, which decides which index it belongs in."""

    FIGURE = "figure"          # evidence inside a document → text index
    ENTITY = "entity"          # the image IS the record (product) → entity indexes
    PAGE = "page"              # full-page render (scanned / layout-critical)
    DECORATIVE = "decorative"  # logo, rule, spacer → dropped, never indexed


class AssetTier(str, Enum):
    """How much extraction the asset warrants. Set by triage, upgradeable later."""

    DROP = "drop"        # T0 — no extraction at all
    SIMPLE = "simple"    # T1 — caption + OCR from a small model
    COMPLEX = "complex"  # T2 — full structured extraction, chart→table
    LAYOUT = "layout"    # T3 — T2 plus page render and visual embedding


class ExtractionStatus(str, Enum):
    PENDING = "pending"      # recorded, not yet extracted
    EXTRACTED = "extracted"  # complete at `dossier_version`
    SKIPPED = "skipped"      # triaged as DROP; intentionally has no extraction
    FAILED = "failed"        # extraction attempted and errored
    STALE = "stale"          # extracted at an older version, needs re-extraction


class RelationKind(str, Enum):
    PART_OF = "part_of"            # → document/chunk that contains this asset
    REFERENCED_BY = "referenced_by"  # → chunk whose text points at this figure
    DEPICTS = "depicts"            # → product/entity id
    VARIANT_OF = "variant_of"      # → sibling asset (same product, other angle)
    SUPERSEDES = "supersedes"      # → older asset this replaces


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRef(_Model):
    """Where this occurrence came from. `bbox` is (x0, top, x1, bottom) in the source
    document's coordinate space; None for standalone assets such as product images."""

    filename: str = ""
    file_path: str = ""
    page_number: int = 0
    bbox: Optional[List[float]] = None
    # The text chunk this asset sits inside, so a figure hit can be merged back into
    # its surrounding prose exactly like a leaf chunk merges to its parent.
    doc_chunk_id: str = ""


class TextSurface(_Model):
    """What gets embedded and BM25-indexed. This is the entire retrieval surface for
    a figure — if it is not in here, the image cannot be found by a text query."""

    caption: str = ""
    description: str = ""
    # OCR verbatim plus chart→table transcription. Kept separate from `description`
    # because it is literal source text: it carries the rare tokens (labels, units,
    # figures) that BM25 scores well and a paraphrase would destroy.
    transcription: str = ""
    tags: List[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not any((self.caption.strip(), self.description.strip(), self.transcription.strip()))


class Measure(_Model):
    """One numeric series lifted out of a chart, so a value can be answered without
    ever re-reading the pixels."""

    label: str = ""
    value: Union[float, str, None] = None
    unit: str = ""
    series: str = ""


# A single attribute value, or several. Multi-valued attributes are the norm rather
# than the exception — a product has several colours and materials — and each value
# becomes its own row in the filter index so every one of them is queryable.
AttributeValue = Union[str, int, float, bool]


class StructuredSurface(_Model):
    """Filterable, LLM-free query surface. Values stay scalar (or lists of scalars) so
    they can become database filter expressions rather than another model call."""

    attributes: Dict[str, Union[AttributeValue, List[AttributeValue]]] = Field(default_factory=dict)
    measures: List[Measure] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)


class Relation(_Model):
    kind: RelationKind
    target: str
    note: str = ""


class Provenance(_Model):
    """How this extraction was produced. `confidence` and `needs_review` drive the
    escalation-to-a-bigger-model decision and the human review queue."""

    tier: AssetTier = AssetTier.SIMPLE
    pipeline: str = ""
    model_used: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = False
    error: str = ""
    extracted_at: Optional[datetime] = None


class ExtractionPayload(_Model):
    """The cacheable, content-derived half. Keyed by (sha256, profile, version)."""

    text: TextSurface = Field(default_factory=TextSurface)
    structured: StructuredSurface = Field(default_factory=StructuredSurface)
    # Synthetic queries this asset answers — HyDE computed at ingest. Retrieval that
    # would otherwise need a runtime rewrite cycle can hit these on the first pass.
    answerable_questions: List[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class BlobRef(_Model):
    """Pointer to the stored bytes. Content-addressed, so it is stable across
    re-uploads and shared by every occurrence of the same image."""

    uri: str = ""
    content_type: str = ""
    byte_size: int = 0
    width: int = 0
    height: int = 0


class AssetDossier(_Model):
    """One occurrence of one image, with everything known about it."""

    asset_id: str
    sha256: str
    profile: str = "base"
    dossier_version: int = DOSSIER_VERSION

    role: AssetRole = AssetRole.FIGURE
    tier: AssetTier = AssetTier.SIMPLE
    status: ExtractionStatus = ExtractionStatus.PENDING

    source: SourceRef = Field(default_factory=SourceRef)
    blob: BlobRef = Field(default_factory=BlobRef)
    extraction: Optional[ExtractionPayload] = None
    relations: List[Relation] = Field(default_factory=list)

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # -- derived ----------------------------------------------------------------

    @property
    def is_indexable(self) -> bool:
        """Whether this asset should produce a retrievable chunk. Decorative assets and
        failed/pending extractions must not reach the index — a chunk with an empty
        text surface is unfindable noise that still costs an embedding."""
        if self.role is AssetRole.DECORATIVE or self.tier is AssetTier.DROP:
            return False
        if self.status is not ExtractionStatus.EXTRACTED or self.extraction is None:
            return False
        return not self.extraction.text.is_empty()

    def is_stale(self, current_version: int = DOSSIER_VERSION) -> bool:
        return self.dossier_version < current_version

    def render_surrogate(self, section_path: Optional[List[str]] = None) -> str:
        """The text that represents this image in the retrieval index.

        Ordered most- to least-specific so that a truncation at any downstream byte cap
        drops the least valuable content first: caption identifies, description
        explains, transcription supplies the literal tokens, tags round out recall.
        """
        if not self.is_indexable:
            return ""

        text = self.extraction.text
        parts: List[str] = []
        if section_path:
            parts.append(" > ".join(section_path[-3:]))

        header = f"[Figure] {text.caption}".strip() if text.caption else "[Figure]"
        parts.append(header)

        for value in (text.description, text.transcription):
            if value and value.strip():
                parts.append(value.strip())

        if text.tags:
            parts.append("Tags: " + ", ".join(text.tags))

        questions = self.extraction.answerable_questions
        if questions:
            parts.append("Answers: " + " ".join(questions))

        return "\n".join(parts).strip()

    def touch(self) -> "AssetDossier":
        self.updated_at = datetime.now(UTC)
        if self.created_at is None:
            self.created_at = self.updated_at
        return self


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

_ASSET_ID_UNSAFE = re.compile(r"\s+")


def build_asset_id(filename: str, page_number: int, index: int) -> str:
    """Deterministic occurrence id, mirroring the chunk-id convention in
    document_loader (`{filename}::p{page}::l{level}::{index}`) so both id families
    read the same way in traces and citations."""
    clean = _ASSET_ID_UNSAFE.sub(" ", (filename or "").strip())
    return f"{clean}::p{int(page_number)}::img{int(index)}"


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

class Migration:
    """One dossier-version upgrade.

    `requires_reextraction` is the field that matters operationally. A migration that
    only reshapes stored data is free and runs over the whole corpus in seconds. A
    migration that needs new model output cannot be applied offline — the backfill job
    marks those rows STALE and leaves them for the extraction pipeline, so a schema
    change never silently fabricates data it does not have.
    """

    def __init__(
        self,
        from_version: int,
        description: str,
        transform: Optional[Callable[[dict], dict]] = None,
        requires_reextraction: bool = False,
    ):
        self.from_version = from_version
        self.to_version = from_version + 1
        self.description = description
        self.transform = transform
        self.requires_reextraction = requires_reextraction

    def apply(self, payload: dict) -> dict:
        result = dict(payload)
        if self.transform is not None:
            result = self.transform(result)
        result["dossier_version"] = self.to_version
        return result


# from_version -> Migration. Register one entry per DOSSIER_VERSION bump; the chain
# must be contiguous, which `validate_migration_chain` enforces at import time.
MIGRATIONS: Dict[int, Migration] = {}


def validate_migration_chain() -> None:
    for version in range(1, DOSSIER_VERSION):
        if version not in MIGRATIONS:
            raise RuntimeError(
                f"Missing dossier migration from version {version} to {version + 1}. "
                f"Every DOSSIER_VERSION bump needs a Migration registered in MIGRATIONS."
            )


def migrate_payload(payload: dict, target_version: int = DOSSIER_VERSION) -> tuple[dict, bool]:
    """Walk a stored dossier payload up to `target_version`.

    Returns (payload, needs_reextraction). The flag is sticky: if ANY migration in the
    chain needed fresh model output, the result is not trustworthy as-is and the caller
    must mark the row STALE rather than EXTRACTED.
    """
    current = dict(payload)
    version = int(current.get("dossier_version", 1))
    needs_reextraction = False

    while version < target_version:
        migration = MIGRATIONS.get(version)
        if migration is None:
            raise RuntimeError(f"No dossier migration registered from version {version}")
        current = migration.apply(current)
        needs_reextraction = needs_reextraction or migration.requires_reextraction
        version = migration.to_version

    return current, needs_reextraction


validate_migration_chain()
