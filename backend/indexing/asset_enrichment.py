"""Enrichment stage: image blocks in, retrievable text blocks out.

Slots into the shared layout pipeline as one more pure transformation:

    <format>_blocks parser → [enrich_image_blocks] → stitch → units → hierarchy

An image block leaves this stage as an ordinary text block whose content is the
asset's text surrogate, so everything downstream — cross-page stitching, section
tagging, the L1/L2/L3 hierarchy, BM25, auto-merge — treats a figure exactly like a
paragraph and needed no changes to support images.

This module is the only place that knows both vocabularies. `backend/assets` never
hears the word "block"; the layout parsers never hear the word "dossier".
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.assets.pipeline import FigureReport, ImageInput

logger = logging.getLogger(__name__)

# How much neighbouring prose is handed to the extractor as context. The caption line
# is usually short and sits immediately after the image; more than this is noise.
NEIGHBOUR_CONTEXT_CHARS = 600
# Section-stack depth cap, matching the one in document_loader so a misclassified run
# of headings cannot grow an absurd path.
MAX_SECTION_DEPTH = 4


def _neighbour_text(blocks: List[Dict[str, Any]], index: int, step: int) -> str:
    """Nearest text/heading content in one direction, skipping other images."""
    cursor = index + step
    while 0 <= cursor < len(blocks):
        block = blocks[cursor]
        block_type = block.get("type")
        if block_type in ("text", "heading"):
            return " ".join((block.get("content") or "").split())[:NEIGHBOUR_CONTEXT_CHARS]
        if block_type == "table":
            return ""
        cursor += step
    return ""


def _section_paths(blocks: List[Dict[str, Any]]) -> List[List[str]]:
    """Section path in effect at each block index.

    A local heading stack rather than a shared one: enrichment runs before the units
    stage, and coupling the two would make `backend.assets` depend on chunking.
    """
    paths: List[List[str]] = []
    stack: List[Tuple[str, Optional[int]]] = []
    for block in blocks:
        if block.get("type") == "heading":
            title = (block.get("content") or "").strip()
            level = block.get("level")
            if title:
                if level is not None:
                    while stack and (stack[-1][1] is None or stack[-1][1] >= level):
                        stack.pop()
                elif stack and stack[-1][1] is None:
                    stack.pop()
                while len(stack) >= MAX_SECTION_DEPTH:
                    stack.pop()
                stack.append((title, level))
        paths.append([title for title, _ in stack])
    return paths


def _to_image_input(
    block: Dict[str, Any],
    index_in_document: int,
    section_path: List[str],
    text_before: str,
    text_after: str,
) -> Optional[ImageInput]:
    data = block.get("data")
    if not data:
        return None
    return ImageInput(
        data=data,
        content_type=block.get("content_type") or "image/png",
        page_number=int(block.get("page_number", 0) or 0),
        index=index_in_document,
        bbox=list(block["bbox"]) if block.get("bbox") else None,
        alt_text=(block.get("alt_text") or "").strip(),
        text_before=text_before,
        text_after=text_after,
        section_path=section_path,
        declared_decorative=bool(block.get("declared_decorative")),
    )


def enrich_image_blocks(
    blocks: List[Dict[str, Any]],
    filename: str,
    file_path: str = "",
    pipeline=None,
) -> Tuple[List[Dict[str, Any]], FigureReport]:
    """Replace image blocks with their text surrogates.

    Images that triage rejected, or whose extraction produced nothing usable, are
    dropped from the stream entirely: a chunk with an empty text surface is
    unretrievable noise that would still cost an embedding.
    """
    image_positions = [i for i, block in enumerate(blocks) if block.get("type") == "image"]
    if not image_positions:
        return blocks, FigureReport()

    section_paths = _section_paths(blocks)
    inputs: List[ImageInput] = []
    kept_positions: List[int] = []
    for order, position in enumerate(image_positions):
        image_input = _to_image_input(
            blocks[position],
            order,
            section_paths[position],
            _neighbour_text(blocks, position, -1),
            _neighbour_text(blocks, position, +1),
        )
        if image_input is not None:
            inputs.append(image_input)
            kept_positions.append(position)

    if not inputs:
        return [block for block in blocks if block.get("type") != "image"], FigureReport()

    if pipeline is None:
        from backend.assets.pipeline import get_figure_pipeline

        pipeline = get_figure_pipeline()

    try:
        dossiers, report = pipeline.process(inputs, filename=filename, file_path=file_path)
    except Exception:
        # Asset enrichment is additive. If it fails wholesale, the document must still
        # index its text — losing figures is far better than losing the document.
        logger.exception("Figure enrichment failed for %s; indexing text only", filename)
        return [block for block in blocks if block.get("type") != "image"], FigureReport()

    surrogate_by_position = {}
    for position, dossier in zip(kept_positions, dossiers):
        surrogate = dossier.render_surrogate()
        if surrogate:
            surrogate_by_position[position] = (surrogate, dossier)

    enriched: List[Dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.get("type") != "image":
            enriched.append(block)
            continue
        entry = surrogate_by_position.get(index)
        if entry is None:
            continue
        surrogate, dossier = entry
        enriched.append({
            "type": "text",
            "content": surrogate,
            "page_number": int(block.get("page_number", 0) or 0),
            "top": block.get("top", 0.0),
            # Carried through units into chunks so retrieval can surface the image
            # itself alongside the text that made it findable.
            "asset_ids": [dossier.asset_id],
        })

    return enriched, report
