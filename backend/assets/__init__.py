"""Asset (image) subsystem: dossiers, content-addressed blobs, and the extraction cache.

Phase 2 provides the durable substrate only — the schema, the stores, and the version
migration machinery. The extraction pipelines that populate a dossier's text and
structured surfaces arrive in Phase 3 (figures) and Phase 5 (entities); they write
through `AssetStore` and never touch these tables directly.
"""
from backend.assets.blobs import (
    BlobStore,
    LocalBlobStore,
    S3BlobStore,
    build_blob_store,
    get_blob_store,
    set_blob_store,
)
from backend.assets.dossier import (
    DOSSIER_VERSION,
    MIGRATIONS,
    AssetDossier,
    AssetRole,
    AssetTier,
    BlobRef,
    ExtractionPayload,
    ExtractionStatus,
    Measure,
    Migration,
    Provenance,
    Relation,
    RelationKind,
    SourceRef,
    StructuredSurface,
    TextSurface,
    build_asset_id,
    compute_sha256,
    migrate_payload,
)
from backend.assets.store import (
    AssetStore,
    BackfillReport,
    DeleteResult,
    get_asset_store,
    set_asset_store,
)

__all__ = [
    "DOSSIER_VERSION",
    "MIGRATIONS",
    "AssetDossier",
    "AssetRole",
    "AssetStore",
    "AssetTier",
    "BackfillReport",
    "BlobRef",
    "BlobStore",
    "DeleteResult",
    "ExtractionPayload",
    "ExtractionStatus",
    "LocalBlobStore",
    "Measure",
    "Migration",
    "Provenance",
    "Relation",
    "RelationKind",
    "S3BlobStore",
    "SourceRef",
    "StructuredSurface",
    "TextSurface",
    "build_asset_id",
    "build_blob_store",
    "compute_sha256",
    "get_asset_store",
    "get_blob_store",
    "migrate_payload",
    "set_asset_store",
    "set_blob_store",
]
