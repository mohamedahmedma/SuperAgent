"""Figure extractors: image bytes and surrounding context in, text surface out.

Two implementations behind one port:

* `HeuristicExtractor` needs no model at all. It builds a text surface from the
  signals the document already carries — alt text, the caption line under the figure,
  the section path. Recall is worse than a vision model's, but it is free, instant,
  and it means image support works on a deployment that has no vision endpoint.
* `VisionExtractor` sends the image to a vision-capable model and asks for a
  structured surface, including the chart→table transcription that lets a text model
  answer questions about a figure forever afterwards.

The pipeline runs the configured extractor and falls back to the heuristic one on any
failure, so a vision outage degrades recall instead of failing an upload.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from backend.assets.dossier import (
    AssetTier,
    ExtractionPayload,
    Provenance,
    StructuredSurface,
    TextSurface,
)

logger = logging.getLogger(__name__)

# "Figure 3: Tuition by grade" / "الشكل ٢: ..." / "Table 4 — ..."
CAPTION_PREFIX_RE = re.compile(
    r"^\s*(figure|fig\.?|table|chart|diagram|exhibit|image|شكل|الشكل|جدول|الجدول)\s*"
    r"[\d٠-٩IVXivx]*\s*[:.–—-]?\s*",
    re.IGNORECASE,
)


@dataclass
class ExtractionRequest:
    """One image plus everything known about where it sits."""

    data: bytes
    content_type: str = "image/png"
    tier: AssetTier = AssetTier.SIMPLE
    alt_text: str = ""
    # Text immediately before/after the image in document order. The line after a
    # figure is very often its caption, which is the single most valuable signal.
    text_before: str = ""
    text_after: str = ""
    section_path: List[str] = field(default_factory=list)
    filename: str = ""
    page_number: int = 0

    def caption_candidate(self) -> str:
        """Best available human-authored name for the figure, cheapest source first."""
        for candidate in (self.alt_text, self.text_after, self.text_before):
            text = " ".join((candidate or "").split())
            if not text:
                continue
            if CAPTION_PREFIX_RE.match(text):
                return text
        # No explicit caption marker: fall back to alt text, then a short following line.
        alt = " ".join((self.alt_text or "").split())
        if alt:
            return alt
        following = " ".join((self.text_after or "").split())
        return following[:160] if 0 < len(following) <= 200 else ""


class FigureExtraction(BaseModel):
    """Structured-output contract for a vision model."""

    caption: str = Field(default="", description="One line naming what this image is")
    description: str = Field(default="", description="What the image shows and how to read it")
    transcription: str = Field(
        default="",
        description="All text visible in the image, verbatim. For a chart or table, a markdown table of the underlying values.",
    )
    tags: List[str] = Field(default_factory=list)
    image_type: Literal[
        "chart", "diagram", "table", "photo", "screenshot", "map", "logo", "other"
    ] = "other"
    answerable_questions: List[str] = Field(
        default_factory=list,
        description="Questions this image can answer, phrased as a user would ask them",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FigureExtractor(ABC):
    name: str = "extractor"

    @abstractmethod
    def extract(self, request: ExtractionRequest) -> ExtractionPayload:
        ...


class HeuristicExtractor(FigureExtractor):
    """Model-free extraction from signals the document already carries."""

    name = "heuristic"

    def __init__(self, config=None):
        self._config = config

    def extract(self, request: ExtractionRequest) -> ExtractionPayload:
        caption = request.caption_candidate()
        # Strip the "Figure 3:" marker from the caption body but keep the words —
        # "Figure 3" alone is not a retrieval signal, the rest of the line is.
        stripped = CAPTION_PREFIX_RE.sub("", caption).strip() or caption

        tags = [part for part in request.section_path[-2:] if part]
        max_tags = getattr(self._config, "max_tags", 8)

        text = TextSurface(
            caption=stripped[:300],
            # No description is invented: an empty surface is honest and is correctly
            # treated as non-indexable, whereas a fabricated one would pollute recall.
            description="",
            transcription="",
            tags=tags[:max_tags],
        )
        return ExtractionPayload(
            text=text,
            structured=StructuredSurface(),
            answerable_questions=[],
            provenance=Provenance(
                tier=request.tier,
                pipeline="figure",
                model_used=self.name,
                # Deliberately low: these assets are the first candidates for
                # re-extraction once a vision model is configured.
                confidence=0.25 if stripped else 0.0,
                needs_review=not stripped,
                extracted_at=datetime.now(UTC),
            ),
        )


class VisionExtractor(FigureExtractor):
    """Vision-model extraction via an OpenAI-compatible multimodal endpoint."""

    name = "vision"

    def __init__(self, config, model_id: str, api_key: str, base_url: str):
        self._config = config
        self._model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._model = None

    def _get_model(self):
        if self._model is None:
            from backend.assets.vision import VisionCredentials, build_vision_model

            self._model = build_vision_model(
                self._config.vision_temperature,
                VisionCredentials(
                    model_id=self._model_id, api_key=self._api_key, base_url=self._base_url
                ),
                max_tokens=self._config.vision_max_output_tokens,
                extra_params=self._config.vision_extra_params,
            )
        return self._model

    def _context_block(self, request: ExtractionRequest) -> str:
        limit = self._config.context_chars
        parts = []
        if request.section_path:
            parts.append("Section: " + " > ".join(request.section_path[-3:]))
        if request.alt_text:
            parts.append(f"Alt text: {request.alt_text}")
        if request.text_before:
            parts.append(f"Text before: {request.text_before[-limit:]}")
        if request.text_after:
            parts.append(f"Text after: {request.text_after[:limit]}")
        parts.append(f"Source: {request.filename} page {request.page_number}")
        return "\n".join(parts)

    def _image_payload(self, request: ExtractionRequest) -> str:
        data = downscale_image(request.data, self._config.vision_max_edge)
        encoded = base64.b64encode(data).decode("ascii")
        content_type = request.content_type or "image/png"
        return f"data:{content_type};base64,{encoded}"

    def extract(self, request: ExtractionRequest) -> ExtractionPayload:
        prompt = (self._config.extraction_prompt or "").replace(
            "{context}", self._context_block(request)
        )
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self._image_payload(request)}},
            ],
        }
        from backend.assets.vision import invoke_structured

        result = invoke_structured(
            self._get_model(), FigureExtraction, [message], config=self._config
        )

        max_tags = self._config.max_tags
        max_questions = self._config.max_answerable_questions
        tags = [tag.strip() for tag in (result.tags or []) if tag and tag.strip()][:max_tags]
        if result.image_type and result.image_type not in tags:
            tags.append(result.image_type)

        return ExtractionPayload(
            text=TextSurface(
                caption=(result.caption or "").strip(),
                description=(result.description or "").strip(),
                transcription=(result.transcription or "").strip(),
                tags=tags,
            ),
            structured=StructuredSurface(attributes={"image_type": result.image_type}),
            answerable_questions=[
                question.strip()
                for question in (result.answerable_questions or [])
                if question and question.strip()
            ][:max_questions],
            provenance=Provenance(
                tier=request.tier,
                pipeline="figure",
                model_used=self._model_id,
                confidence=float(result.confidence or 0.0),
                needs_review=float(result.confidence or 0.0) < self._config.escalate_below_confidence,
                extracted_at=datetime.now(UTC),
            ),
        )


def downscale_image(data: bytes, max_edge: int) -> bytes:
    """Shrink so the longest edge is at most `max_edge`.

    Image tokens scale with area, so this is the single largest lever on vision cost.
    Returns the input unchanged when Pillow is unavailable or the image is already
    small enough — never fails an extraction over a resize.
    """
    if max_edge <= 0:
        return data
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - depends on the environment
        return data
    try:
        with Image.open(io.BytesIO(data)) as image:
            if max(image.width, image.height) <= max_edge:
                return data
            scale = max_edge / max(image.width, image.height)
            resized = image.convert("RGB").resize(
                (max(int(image.width * scale), 1), max(int(image.height * scale), 1)),
                Image.LANCZOS,
            )
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
    except Exception:
        logger.debug("Downscale failed; sending the original bytes", exc_info=True)
        return data


def build_extractor(figures_config) -> FigureExtractor:
    """The configured extractor, or the heuristic one when vision is off or unusable.

    Credentials are resolved by backend/assets/vision.py, the single place the
    VISION_* -> main-model fallback chain is defined.
    """
    if not figures_config.vision_enabled:
        return HeuristicExtractor(figures_config)

    from backend.assets.vision import resolve_vision_credentials

    credentials = resolve_vision_credentials()
    if not credentials.available:
        logger.warning(
            "Profile enables figure vision extraction but credentials are incomplete "
            "(%s) — falling back to heuristic extraction.",
            credentials.describe(),
        )
        return HeuristicExtractor(figures_config)

    return VisionExtractor(
        figures_config,
        model_id=credentials.model_id,
        api_key=credentials.api_key,
        base_url=credentials.base_url,
    )
