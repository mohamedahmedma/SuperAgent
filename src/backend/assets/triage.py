"""Deterministic image triage: role and tier, decided before any model runs.

This is the largest cost lever in image ingest. A document corpus is mostly logos,
rules, spacers, and repeated letterheads, and every one of them rejected here is a
VLM call never made. Nothing in this module calls a model, opens a network
connection, or costs more than a few microseconds per image.

Decisions are explained, not just made: every result carries a `reason` string that
lands in the dossier's provenance, so "why was this figure never indexed?" has an
answer without re-running ingest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from backend.assets.dossier import AssetRole, AssetTier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageFacts:
    """Everything triage needs. Deliberately plain data so the rules stay testable
    without decoding a real image."""

    sha256: str
    byte_size: int
    width: int = 0
    height: int = 0
    # How many distinct pages of THIS document contain this digest, and how many
    # pages the document has. Together they identify page furniture.
    pages_with_digest: int = 1
    total_pages: int = 1
    # True when the source format told us this is decorative (e.g. an empty alt="").
    declared_decorative: bool = False
    has_alt_text: bool = False


@dataclass(frozen=True)
class TriageResult:
    role: AssetRole
    tier: AssetTier
    reason: str

    @property
    def is_dropped(self) -> bool:
        return self.tier is AssetTier.DROP


def _aspect_ratio(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 1.0
    return max(width, height) / min(width, height)


def triage_image(facts: ImageFacts, config) -> TriageResult:
    """Classify one image against a profile's TriageConfig.

    Order matters: the cheapest and most certain rejections come first, and the
    page-furniture rule runs before any size rule so that a large repeated
    letterhead is still recognised as furniture rather than promoted for extraction.
    """
    drop = lambda reason: TriageResult(AssetRole.DECORATIVE, AssetTier.DROP, reason)

    if facts.declared_decorative:
        return drop("declared_decorative")

    if facts.byte_size < config.min_byte_size:
        return drop(f"byte_size<{config.min_byte_size}")

    # Page furniture: the same bytes on most pages of a multi-page document is a
    # letterhead or watermark, whatever its size. Single-page documents cannot
    # establish repetition, so the rule does not apply to them.
    if facts.total_pages > 1 and facts.pages_with_digest > 1:
        fraction = facts.pages_with_digest / facts.total_pages
        if fraction >= config.repeat_page_fraction:
            return drop(f"page_furniture({facts.pages_with_digest}/{facts.total_pages})")

    # Dimensions may be unknown (some formats do not expose them cheaply). An unknown
    # size must not be treated as zero — that would silently drop every image from
    # such a format — so the size rules only apply when both dimensions are known.
    if facts.width > 0 and facts.height > 0:
        if facts.width < config.min_width or facts.height < config.min_height:
            return drop(f"smaller_than_{config.min_width}x{config.min_height}")
        if facts.width * facts.height < config.min_area:
            return drop(f"area<{config.min_area}")
        if _aspect_ratio(facts.width, facts.height) > config.max_aspect_ratio:
            return drop(f"aspect_ratio>{config.max_aspect_ratio}")

    area = facts.width * facts.height
    if area >= config.complex_min_area:
        # Big enough to carry structure worth transcribing (chart, diagram, table
        # screenshot) — worth a capable model.
        return TriageResult(AssetRole.FIGURE, AssetTier.COMPLEX, f"area>={config.complex_min_area}")

    return TriageResult(AssetRole.FIGURE, AssetTier.SIMPLE, "default_simple")


def count_digest_pages(images: list) -> Tuple[Dict[str, int], int]:
    """Per-digest page counts and the document's page count.

    Computed once per document and shared by every triage call, so the furniture rule
    costs one pass rather than a scan per image.
    """
    pages_by_digest: Dict[str, set] = {}
    pages: set = set()
    for item in images:
        digest = item.get("sha256") or ""
        page = int(item.get("page_number", 0) or 0)
        pages.add(page)
        if digest:
            pages_by_digest.setdefault(digest, set()).add(page)
    return ({digest: len(seen) for digest, seen in pages_by_digest.items()}, max(len(pages), 1))


def probe_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """(width, height) via Pillow, or None when it is unavailable or the bytes are
    not a decodable image. Pillow arrives transitively with pdfplumber; treating it
    as optional keeps triage working (on metadata alone) if that ever changes."""
    try:
        import io

        from PIL import Image
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None
