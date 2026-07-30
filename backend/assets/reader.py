"""On-demand pixel reading: answering a question about one image.

Distinct from `extractors.py` on purpose. Extraction is exhaustive, runs at ingest,
and produces a dossier. Reading is targeted, runs at query time, answers one question,
and its result is recorded as a session observation so the same pixels are never sent
twice for the same conversation.

Availability is a first-class answer here. `FigureReader.available` lets the caller
decide between escalating and serving text, instead of discovering mid-call that no
vision model is configured.
"""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from backend.assets.extractors import downscale_image
from backend.assets.vision import strip_reasoning

logger = logging.getLogger(__name__)

DEFAULT_READ_PROMPT = (
    "Answer the question using ONLY what is visible in this image.\n"
    "Be specific: quote labels, values, and units exactly as they appear.\n"
    "If the image does not contain the answer, say so plainly rather than guessing.\n"
    "Keep the answer under 120 words.\n\n"
    "{context}\n\n"
    "Question: {question}"
)


@dataclass
class ReadRequest:
    data: bytes
    content_type: str = "image/png"
    question: str = ""
    caption: str = ""
    source: str = ""


class FigureReader:
    """Vision-model reader over an OpenAI-compatible multimodal endpoint."""

    def __init__(self, figures_config, model_id: str = "", api_key: str = "", base_url: str = ""):
        self._config = figures_config
        self._model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._model = None

    @property
    def available(self) -> bool:
        return bool(self._model_id and self._api_key)

    @property
    def model_id(self) -> str:
        return self._model_id

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

    def read(self, request: ReadRequest) -> Optional[str]:
        """The model's answer, or None when unavailable or the call failed.

        None rather than an exception: the caller's fallback (serve the dossier text)
        is a perfectly good answer, and a vision outage must not fail the turn.
        """
        if not self.available:
            return None

        context_parts: List[str] = []
        if request.caption:
            context_parts.append(f"This image is captioned: {request.caption}")
        if request.source:
            context_parts.append(f"It appears in {request.source}.")
        prompt = DEFAULT_READ_PROMPT.format(
            context="\n".join(context_parts),
            question=request.question or "Describe what this image shows.",
        )

        data = downscale_image(request.data, self._config.vision_max_edge)
        encoded = base64.b64encode(data).decode("ascii")
        content_type = request.content_type or "image/png"

        from backend.assets.vision import call_with_rate_limit_retry

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{content_type};base64,{encoded}"}},
            ],
        }
        try:
            result = call_with_rate_limit_retry(
                lambda: self._get_model().invoke([message]), self._config, "figure read"
            )
        except Exception:
            logger.exception("Figure read failed")
            return None

        content = getattr(result, "content", "")
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return strip_reasoning(content) or None


def build_figure_reader(figures_config) -> FigureReader:
    """Reader for the active profile; credentials come from backend/assets/vision.py."""
    if not figures_config.vision_enabled:
        return FigureReader(figures_config)

    from backend.assets.vision import resolve_vision_credentials

    credentials = resolve_vision_credentials()
    return FigureReader(
        figures_config,
        model_id=credentials.model_id,
        api_key=credentials.api_key,
        base_url=credentials.base_url,
    )


_reader: Optional[FigureReader] = None


def get_figure_reader() -> FigureReader:
    global _reader
    if _reader is None:
        from backend.profiles import get_profile

        _reader = build_figure_reader(get_profile().assets.figures)
    return _reader


def set_figure_reader(reader: Optional[FigureReader]) -> None:
    global _reader
    _reader = reader
