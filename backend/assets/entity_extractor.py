"""Entity extraction: turning a product image into filterable attributes.

Differs from figure extraction in what it is trying to produce. A figure needs prose
and a transcription so a text model can answer questions about it later. An entity
needs *typed attributes*, because the query path that matters — "red running shoes
under 100" — is a scalar filter, not a semantic match.

The structured-output schema is generated from the profile's attribute vocabulary at
runtime, so this module contains no knowledge of shoes, colours, or prices. Point it
at a different `attributes:` block and it extracts a different domain.
"""
from __future__ import annotations

import base64
import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.assets.attributes import AttributeSchema
from backend.assets.dossier import (
    AssetRole,
    AssetTier,
    ExtractionPayload,
    Provenance,
    StructuredSurface,
    TextSurface,
)
from backend.assets.extractors import ExtractionRequest, FigureExtractor, downscale_image

logger = logging.getLogger(__name__)

class _EntityCore(BaseModel):
    """The non-attribute half of an entity extraction; the attribute half is
    generated per profile and merged into this at model-build time."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(default="", description="Short product name")
    summary: str = Field(default="", description="One or two sentence listing description")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def build_entity_model(schema: AttributeSchema):
    """Compose the core fields with the profile's generated attribute model."""
    from pydantic import create_model

    attribute_model = schema.build_extraction_model("ExtractedAttributes")
    return create_model(
        "EntityExtraction",
        __base__=_EntityCore,
        attributes=(attribute_model, Field(default_factory=attribute_model)),
    )


class HeuristicEntityExtractor(FigureExtractor):
    """Model-free entity extraction.

    Produces a title from the alt text or caption and no attributes at all. That is
    honest rather than useless: the entity is still findable semantically, and an
    empty attribute set simply means it never matches an attribute filter — which is
    correct, since nothing established those attributes.
    """

    name = "entity_heuristic"

    def __init__(self, entities_config, schema: AttributeSchema):
        self._config = entities_config
        self._schema = schema

    def extract(self, request: ExtractionRequest) -> ExtractionPayload:
        title = request.caption_candidate()
        return ExtractionPayload(
            text=TextSurface(caption=title[:300], tags=list(request.section_path[-2:])),
            structured=StructuredSurface(attributes={}),
            provenance=Provenance(
                tier=request.tier,
                pipeline="entity",
                model_used=self.name,
                confidence=0.2 if title else 0.0,
                needs_review=True,
                extracted_at=datetime.now(UTC),
            ),
        )


class VisionEntityExtractor(FigureExtractor):
    """Vision extraction against a profile-generated attribute schema."""

    name = "entity_vision"

    def __init__(self, entities_config, schema: AttributeSchema, model_id: str,
                 api_key: str, base_url: str, figures_config=None):
        self._config = entities_config
        self._schema = schema
        self._model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._figures = figures_config
        self._model = None
        self._output_model = build_entity_model(schema)

    def _get_model(self):
        if self._model is None:
            from backend.assets.vision import VisionCredentials, build_vision_model

            self._model = build_vision_model(
                0.0,
                VisionCredentials(
                    model_id=self._model_id, api_key=self._api_key, base_url=self._base_url
                ),
                max_tokens=getattr(self._figures, "vision_max_output_tokens", 4096),
                extra_params=getattr(self._figures, "vision_extra_params", None),
            )
        return self._model

    def _context(self, request: ExtractionRequest) -> str:
        parts = []
        if request.section_path:
            parts.append("Section: " + " > ".join(request.section_path[-3:]))
        if request.alt_text:
            parts.append(f"Alt text: {request.alt_text}")
        if request.text_before:
            parts.append(f"Text before: {request.text_before[-400:]}")
        if request.text_after:
            parts.append(f"Text after: {request.text_after[:400]}")
        return "\n".join(parts) or "(no surrounding text)"

    def extract(self, request: ExtractionRequest) -> ExtractionPayload:
        from backend.prompts import resolve as resolve_prompt

        prompt = resolve_prompt(
            self._config.extraction_prompt,
            "assets/entity_extraction.j2",
            attributes=self._schema.describe(),
            context=self._context(request),
        )

        max_edge = getattr(self._figures, "vision_max_edge", 1024)
        encoded = base64.b64encode(downscale_image(request.data, max_edge)).decode("ascii")
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{request.content_type or 'image/png'};base64,{encoded}"}},
            ],
        }
        from backend.assets.vision import invoke_structured

        result = invoke_structured(
            self._get_model(), self._output_model, [message], config=self._figures
        )

        raw: Dict[str, Any] = {}
        attributes = getattr(result, "attributes", None)
        if attributes is not None:
            raw = attributes.model_dump() if hasattr(attributes, "model_dump") else dict(attributes)
        # Coercion and closed-vocabulary validation happen here, not in the prompt: a
        # model told to pick from a list will still occasionally invent a value.
        normalized = self._schema.normalize(raw)

        title = (getattr(result, "title", "") or "").strip()
        summary = (getattr(result, "summary", "") or "").strip()
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)

        return ExtractionPayload(
            text=TextSurface(
                caption=title,
                description=summary,
                # Attributes join the text surface too, so semantic recall can find
                # "red shoes" before any filter runs — the filter narrows, it does not
                # do the finding.
                transcription=_attributes_as_text(normalized),
                tags=_attribute_tags(normalized),
            ),
            structured=StructuredSurface(attributes=normalized),
            answerable_questions=[],
            provenance=Provenance(
                tier=request.tier,
                pipeline="entity",
                model_used=self._model_id,
                confidence=confidence,
                needs_review=confidence < 0.4 or not normalized,
                extracted_at=datetime.now(UTC),
            ),
        )


def _attributes_as_text(attributes: Dict[str, Any]) -> str:
    lines = []
    for name, value in attributes.items():
        rendered = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
        lines.append(f"{name}: {rendered}")
    return "\n".join(lines)


def _attribute_tags(attributes: Dict[str, Any], limit: int = 10) -> List[str]:
    tags: List[str] = []
    for value in attributes.values():
        for item in (value if isinstance(value, list) else [value]):
            text = str(item).strip()
            if text and text not in tags and not text.replace(".", "", 1).isdigit():
                tags.append(text)
    return tags[:limit]


def build_entity_extractor(entities_config, schema: AttributeSchema, figures_config=None) -> FigureExtractor:
    """Vision extractor when configured and the vocabulary is non-empty, else the
    heuristic one. An empty vocabulary means there is nothing for vision to fill in."""
    if not entities_config.vision_enabled or not schema:
        return HeuristicEntityExtractor(entities_config, schema)

    from backend.assets.vision import resolve_vision_credentials

    credentials = resolve_vision_credentials()
    if not credentials.available:
        logger.warning(
            "Profile enables entity vision extraction but credentials are incomplete "
            "(%s) — falling back to heuristic extraction.",
            credentials.describe(),
        )
        return HeuristicEntityExtractor(entities_config, schema)

    return VisionEntityExtractor(
        entities_config, schema,
        model_id=credentials.model_id, api_key=credentials.api_key,
        base_url=credentials.base_url,
        figures_config=figures_config,
    )


def resolve_role(strategy: str, declared: Optional[AssetRole] = None) -> AssetRole:
    """Role for an image under a profile's strategy. An explicit declaration from the
    caller (a direct product upload) always wins."""
    if declared is not None:
        return declared
    if strategy == "entity":
        return AssetRole.ENTITY
    return AssetRole.FIGURE


__all__ = [
    "HeuristicEntityExtractor",
    "VisionEntityExtractor",
    "build_entity_extractor",
    "build_entity_model",
    "resolve_role",
]
