"""FigurePipeline: document images in, stored dossiers out.

The order of operations is the cost model:

    sha256 → blob (idempotent) → triage → CACHE LOOKUP → extract → store

Triage rejects the majority of a corpus's images for free. The cache lookup then
catches every repeat of a surviving image, across documents and across re-uploads.
Only what is left reaches an extractor. A logo on 400 pages costs one hash per
occurrence and nothing else.

Extraction failures never fail an upload: the pipeline falls back to the heuristic
extractor and, failing that, records the asset as FAILED so it is visible to the
backfill queue rather than silently absent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.assets.blobs import get_blob_store
from backend.assets.dossier import (
    DOSSIER_VERSION,
    AssetDossier,
    AssetRole,
    AssetTier,
    BlobRef,
    ExtractionPayload,
    ExtractionStatus,
    Provenance,
    Relation,
    RelationKind,
    SourceRef,
    build_asset_id,
    compute_sha256,
)
from backend.assets.attributes import AttributeSchema, build_attribute_schema
from backend.assets.entity_extractor import build_entity_extractor, resolve_role
from backend.assets.extractors import (
    ExtractionRequest,
    FigureExtractor,
    HeuristicExtractor,
    build_extractor,
)
from backend.assets.store import AssetStore, get_asset_store
from backend.assets.triage import ImageFacts, count_digest_pages, probe_dimensions, triage_image

logger = logging.getLogger(__name__)


@dataclass
class ImageInput:
    """One image found in a document, with its document-order context."""

    data: bytes
    content_type: str = "image/png"
    page_number: int = 0
    index: int = 0
    bbox: Optional[List[float]] = None
    alt_text: str = ""
    text_before: str = ""
    text_after: str = ""
    section_path: List[str] = field(default_factory=list)
    declared_decorative: bool = False
    doc_chunk_id: str = ""
    # Set by a caller that already knows what this image is (a direct product upload).
    # Overrides the profile's role_strategy.
    declared_role: Optional[AssetRole] = None


@dataclass
class FigureReport:
    total: int = 0
    dropped: int = 0
    cached: int = 0
    extracted: int = 0
    failed: int = 0
    indexable: int = 0
    skipped_over_limit: int = 0
    entities: int = 0
    attributes_indexed: int = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "dropped": self.dropped,
            "cached": self.cached,
            "extracted": self.extracted,
            "failed": self.failed,
            "indexable": self.indexable,
            "skipped_over_limit": self.skipped_over_limit,
            "entities": self.entities,
            "attributes_indexed": self.attributes_indexed,
        }

    def summary(self) -> str:
        return (
            f"{self.indexable} indexable of {self.total} images "
            f"({self.dropped} triaged out, {self.cached} from cache, "
            f"{self.extracted} extracted, {self.failed} failed)"
        )


class FigurePipeline:
    def __init__(
        self,
        profile=None,
        store: Optional[AssetStore] = None,
        blob_store=None,
        extractor: Optional[FigureExtractor] = None,
        fallback_extractor: Optional[FigureExtractor] = None,
        entity_extractor: Optional[FigureExtractor] = None,
        entity_index=None,
    ):
        self._profile = profile
        self._store = store
        self._blob_store = blob_store
        self._extractor = extractor
        self._fallback = fallback_extractor
        self._entity_extractor = entity_extractor
        self._entity_index = entity_index
        self._attribute_schema: Optional[AttributeSchema] = None

    # -- lazily resolved collaborators -----------------------------------------

    @property
    def profile(self):
        if self._profile is None:
            from backend.profiles import get_profile

            self._profile = get_profile()
        return self._profile

    @property
    def store(self) -> AssetStore:
        if self._store is None:
            self._store = get_asset_store()
        return self._store

    @property
    def blob_store(self):
        if self._blob_store is None:
            self._blob_store = get_blob_store()
        return self._blob_store

    @property
    def extractor(self) -> FigureExtractor:
        if self._extractor is None:
            self._extractor = build_extractor(self.profile.assets.figures)
        return self._extractor

    @property
    def attribute_schema(self) -> AttributeSchema:
        if self._attribute_schema is None:
            self._attribute_schema = build_attribute_schema(self.profile.assets.entities)
        return self._attribute_schema

    @property
    def entity_extractor(self) -> FigureExtractor:
        if self._entity_extractor is None:
            self._entity_extractor = build_entity_extractor(
                self.profile.assets.entities,
                self.attribute_schema,
                self.profile.assets.figures,
            )
        return self._entity_extractor

    @property
    def entity_index(self):
        if self._entity_index is None:
            from backend.assets.entity_store import get_entity_index

            self._entity_index = get_entity_index()
        return self._entity_index

    def _extractor_for(self, role: AssetRole) -> FigureExtractor:
        """Role decides the extractor: a figure needs prose and a transcription, an
        entity needs typed attributes."""
        return self.entity_extractor if role is AssetRole.ENTITY else self.extractor

    # Extractors that read no pixels. A cached result from one of these is superseded
    # the moment a vision-capable extractor becomes available.
    _MODEL_FREE = {"", "heuristic", "entity_heuristic"}

    def _cache_is_weaker(self, cached: ExtractionPayload, role: AssetRole) -> bool:
        """Whether a cached extraction was produced by a weaker extractor than the one
        configured now — the condition under which the cache must be bypassed."""
        current = self._extractor_for(role)
        if current.name in self._MODEL_FREE:
            return False
        return (cached.provenance.model_used or "") in self._MODEL_FREE

    @property
    def fallback(self) -> FigureExtractor:
        if self._fallback is None:
            self._fallback = HeuristicExtractor(self.profile.assets.figures)
        return self._fallback

    # -- main entry point -------------------------------------------------------

    def process(
        self,
        images: List[ImageInput],
        filename: str,
        file_path: str = "",
    ) -> tuple[List[AssetDossier], FigureReport]:
        report = FigureReport(total=len(images))
        if not images:
            return [], report

        assets_config = self.profile.assets
        profile_name = self.profile.name

        limit = assets_config.max_images_per_document
        if len(images) > limit:
            # A pathological document must not be able to spend an unbounded amount of
            # extraction budget. Truncation is reported, never silent.
            report.skipped_over_limit = len(images) - limit
            logger.warning(
                "%s contains %d images; processing the first %d (assets.max_images_per_document)",
                filename, len(images), limit,
            )
            images = images[:limit]

        digests = [compute_sha256(image.data) for image in images]
        pages_by_digest, total_pages = count_digest_pages(
            [{"sha256": digest, "page_number": image.page_number}
             for digest, image in zip(digests, images)]
        )

        dossiers: List[AssetDossier] = []
        for image, digest in zip(images, digests):
            dossier = self._process_one(
                image, digest, filename, file_path, profile_name,
                assets_config, pages_by_digest, total_pages, report,
            )
            dossiers.append(dossier)

        self.store.record_many(dossiers)
        report.indexable = sum(1 for dossier in dossiers if dossier.is_indexable)
        return dossiers, report

    # -- per-image --------------------------------------------------------------

    def _process_one(
        self,
        image: ImageInput,
        digest: str,
        filename: str,
        file_path: str,
        profile_name: str,
        assets_config,
        pages_by_digest: Dict[str, int],
        total_pages: int,
        report: FigureReport,
    ) -> AssetDossier:
        asset_id = build_asset_id(filename, image.page_number, image.index)

        dimensions = probe_dimensions(image.data) or (0, 0)
        facts = ImageFacts(
            sha256=digest,
            byte_size=len(image.data),
            width=dimensions[0],
            height=dimensions[1],
            pages_with_digest=pages_by_digest.get(digest, 1),
            total_pages=total_pages,
            declared_decorative=image.declared_decorative,
            has_alt_text=bool(image.alt_text.strip()),
        )
        verdict = triage_image(facts, assets_config.triage)

        # Triage decides whether an image survives; the profile's role_strategy decides
        # what a survivor IS. A dropped image stays DECORATIVE regardless — declaring a
        # catalogue import must not resurrect its spacers and logos.
        role = verdict.role
        if not verdict.is_dropped:
            role = resolve_role(assets_config.entities.role_strategy, image.declared_role)

        dossier = AssetDossier(
            asset_id=asset_id,
            sha256=digest,
            profile=profile_name,
            dossier_version=DOSSIER_VERSION,
            role=role,
            tier=verdict.tier,
            status=ExtractionStatus.PENDING,
            source=SourceRef(
                filename=filename,
                file_path=file_path,
                page_number=image.page_number,
                bbox=image.bbox,
                doc_chunk_id=image.doc_chunk_id,
            ),
            blob=BlobRef(
                content_type=image.content_type,
                byte_size=len(image.data),
                width=dimensions[0],
                height=dimensions[1],
            ),
            relations=[Relation(kind=RelationKind.PART_OF, target=filename)],
        )

        if verdict.is_dropped:
            # Dropped assets are still recorded. Knowing an image existed and why it
            # was rejected is what makes a triage threshold tunable after the fact.
            dossier.status = ExtractionStatus.SKIPPED
            dossier.extraction = ExtractionPayload(
                provenance=Provenance(
                    tier=verdict.tier, pipeline="triage", model_used="", confidence=1.0,
                    error=verdict.reason,
                )
            )
            report.dropped += 1
            return dossier

        # Blobs are stored only for assets that survive triage; a rejected logo does
        # not earn disk space.
        try:
            dossier.blob.uri = self.blob_store.put(digest, image.data, image.content_type)
        except Exception:
            logger.exception("Failed to store blob for %s", asset_id)

        cached = self.store.find_extraction(digest, profile_name, DOSSIER_VERSION)
        if cached is not None and self._cache_is_weaker(cached, role):
            # Turning vision on must actually take effect. A heuristic extraction is a
            # legitimate cache entry while heuristic is all that is available, but once
            # a vision extractor is configured it is a strictly worse answer for the
            # same bytes, and serving it would make the upgrade a silent no-op.
            logger.info(
                "Re-extracting %s: cached result came from %r, now running %r",
                asset_id, cached.provenance.model_used, self._extractor_for(role).name,
            )
            cached = None
        if cached is not None:
            dossier.extraction = cached
            dossier.status = ExtractionStatus.EXTRACTED
            report.cached += 1
            self._index_entity(dossier, profile_name, report)
            return dossier

        payload = self._extract(image, verdict.tier, filename, role)
        if payload is None:
            dossier.status = ExtractionStatus.FAILED
            report.failed += 1
            return dossier

        dossier.extraction = payload
        dossier.status = ExtractionStatus.EXTRACTED
        report.extracted += 1
        try:
            self.store.save_extraction(digest, profile_name, payload, DOSSIER_VERSION)
        except Exception:
            # A cache write failure costs money next time, not correctness now.
            logger.exception("Failed to cache extraction for %s", asset_id)

        self._index_entity(dossier, profile_name, report)
        return dossier

    def _index_entity(self, dossier: AssetDossier, profile_name: str, report: FigureReport) -> None:
        """Write an entity's attributes to the filter index.

        Runs for cache hits too: the extraction is shared by digest, but the INDEX is
        per occurrence, and a second document's copy of the same product still has to
        be findable by attribute.
        """
        if dossier.role is not AssetRole.ENTITY or dossier.extraction is None:
            return
        report.entities += 1
        attributes = dossier.extraction.structured.attributes
        if not attributes:
            return
        try:
            report.attributes_indexed += self.entity_index.index_asset(
                dossier.asset_id, profile_name, attributes, self.attribute_schema
            )
        except Exception:
            # The entity is still semantically retrievable; it just will not answer an
            # attribute filter until a re-index. Not worth failing an upload over.
            logger.exception("Failed to index attributes for %s", dossier.asset_id)

    def _extract(
        self,
        image: ImageInput,
        tier: AssetTier,
        filename: str,
        role: AssetRole = AssetRole.FIGURE,
    ) -> Optional[ExtractionPayload]:
        request = ExtractionRequest(
            data=image.data,
            content_type=image.content_type,
            tier=tier,
            alt_text=image.alt_text,
            text_before=image.text_before,
            text_after=image.text_after,
            section_path=list(image.section_path),
            filename=filename,
            page_number=image.page_number,
        )

        extractor = self._extractor_for(role)
        try:
            return extractor.extract(request)
        except Exception:
            logger.exception("Asset extraction failed for %s page %d", filename, image.page_number)

        if self.fallback is extractor:
            return None
        try:
            # Degrade to alt-text/caption extraction rather than losing the asset:
            # a weak text surface still beats an image that can never be retrieved.
            return self.fallback.extract(request)
        except Exception:
            logger.exception("Fallback extraction also failed for %s", filename)
            return None


_pipeline: Optional[FigurePipeline] = None


def get_figure_pipeline() -> FigurePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FigurePipeline()
    return _pipeline


def set_figure_pipeline(pipeline: Optional[FigurePipeline]) -> None:
    global _pipeline
    _pipeline = pipeline
