"""HTML layout parsing to the shared block contract (heading/text/table).

Builds on html_processor's proven readers (encoding detection, noise stripping,
main/article root selection) but maps structure to typed blocks instead of a
markdown-flavored string: h1–h6 become heading blocks feeding the section stack,
<table> becomes a rows block (validated with the same real-table heuristic as PDFs,
since HTML tables are often used for page layout, not data), and p/li/pre become
text blocks. This replaces the flat path where table cells were smeared into
prose lines and headings never reached the section stack.

HTML has no pages: page_number is always 0 and `top` is a document-order surrogate.
"""
from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import backend.indexing.html_processor as html_processor
from backend.indexing.pdf_layout import (
    flatten_table_to_text,
    format_table_rows,
    looks_like_real_table,
    normalize_table_rows,
)

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_BLOCK_TAGS = list(_HEADING_TAGS) + ["p", "li", "pre", "table", "img", "figure"]

_DATA_URI_RE = re.compile(r"^data:(?P<mime>image/[\w.+-]+)?;?(?P<encoding>base64)?,(?P<payload>.*)$", re.S)
_LOCAL_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
}
# Ingest never dereferences a remote src. Fetching URLs out of an uploaded document
# would turn document upload into a server-side request forge, and would make ingest
# depend on the network. Remote images contribute their alt text and nothing else.
MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024


def _decode_data_uri(src: str) -> Optional[Tuple[bytes, str]]:
    match = _DATA_URI_RE.match(src.strip())
    if not match or not match.group("encoding"):
        return None
    try:
        data = base64.b64decode(match.group("payload"), validate=False)
    except (ValueError, binascii.Error):
        return None
    if not data or len(data) > MAX_INLINE_IMAGE_BYTES:
        return None
    return data, match.group("mime") or "image/png"


def _read_local_image(src: str, base_dir: Path) -> Optional[Tuple[bytes, str]]:
    """Load an image referenced by a relative path, confined to the document's own
    directory tree so a crafted `src` cannot read arbitrary files."""
    candidate = (base_dir / unquote(src.split("?", 1)[0].split("#", 1)[0])).resolve()
    try:
        candidate.relative_to(base_dir.resolve())
    except ValueError:
        return None
    content_type = _LOCAL_IMAGE_TYPES.get(candidate.suffix.lower())
    if not content_type or not candidate.is_file():
        return None
    try:
        if candidate.stat().st_size > MAX_INLINE_IMAGE_BYTES:
            return None
        return candidate.read_bytes(), content_type
    except OSError:
        return None


def _image_payload(src: str, base_dir: Path) -> Optional[Tuple[bytes, str]]:
    src = (src or "").strip()
    if not src:
        return None
    if src.startswith("data:"):
        return _decode_data_uri(src)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", src) or src.startswith("//"):
        return None  # remote — see MAX_INLINE_IMAGE_BYTES comment above
    return _read_local_image(src, base_dir)


def _table_rows(table_el) -> List[List[str]]:
    """Extract this table's OWN rows. find_all is recursive, so rows and cells of a
    nested table must be filtered out by parent identity — otherwise inner rows get
    double-counted as extra top-level rows (the inner content still reaches the
    output flattened inside its outer cell's text)."""
    rows = []
    for tr in table_el.find_all("tr"):
        if tr.find_parent("table") is not table_el:
            continue
        cells = [
            cell.get_text(" ", strip=True)
            for cell in tr.find_all(["td", "th"])
            if cell.find_parent("tr") is tr
        ]
        rows.append(cells)
    return normalize_table_rows(rows)


def parse_html_blocks(file_path: str) -> List[Dict[str, Any]]:
    """Parse an HTML file into ordered heading/text/table blocks. Raises on
    unreadable files — the caller decides whether to fall back to the flat path."""
    from bs4 import BeautifulSoup

    source_path = Path(file_path)
    base_dir = source_path.parent
    html = html_processor._read_html_text(source_path)
    soup = BeautifulSoup(html, "html.parser")
    html_processor._strip_noise(soup)
    root = html_processor._pick_root(soup)

    blocks: List[Dict[str, Any]] = []
    order = 0.0

    title = html_processor._doc_title(soup)
    if title:
        blocks.append({
            "type": "heading",
            "content": title,
            "level": 1,
            "page_number": 0,
            "top": order,
        })
        order += 1.0

    for el in root.find_all(_BLOCK_TAGS, limit=8000):
        # Table content is emitted once, by the <table> element itself.
        if el.name != "table" and el.find_parent("table"):
            continue
        # Nested tables are covered by their outermost table.
        if el.name == "table" and el.find_parent("table"):
            continue
        # A <p> inside a list item would duplicate the <li> text.
        if el.name == "p" and el.find_parent("li"):
            continue

        if el.name == "figure":
            # <figure> is a wrapper; its <img> and <figcaption> are visited on their
            # own. Skipping it here avoids emitting the caption text twice.
            continue

        if el.name == "img":
            alt = (el.get("alt") or "").strip()
            payload = _image_payload(el.get("src") or "", base_dir)
            if payload is None:
                # No decodable bytes. Alt text is still real, human-authored signal,
                # so it is kept as ordinary text rather than discarded.
                if alt:
                    blocks.append({
                        "type": "text",
                        "content": alt,
                        "page_number": 0,
                        "top": order,
                    })
                    order += 1.0
                continue
            data, content_type = payload
            blocks.append({
                "type": "image",
                "content": "",
                "data": data,
                "content_type": content_type,
                "alt_text": alt,
                # An explicitly empty alt is the HTML author declaring the image
                # decorative; triage honours that rather than second-guessing it.
                "declared_decorative": el.has_attr("alt") and not alt,
                "page_number": 0,
                "top": order,
            })
            order += 1.0
            continue

        if el.name == "table":
            rows = _table_rows(el)
            if not rows:
                continue
            if looks_like_real_table(rows):
                blocks.append({
                    "type": "table",
                    "content": format_table_rows(rows),
                    "rows": rows,
                    "page_number": 0,
                    "top": order,
                })
            else:
                blocks.append({
                    "type": "text",
                    "content": flatten_table_to_text(rows),
                    "page_number": 0,
                    "top": order,
                })
            order += 1.0
            continue

        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in _HEADING_TAGS:
            blocks.append({
                "type": "heading",
                "content": text,
                "level": int(el.name[1]),
                "page_number": 0,
                "top": order,
            })
        else:
            blocks.append({
                "type": "text",
                "content": text,
                "page_number": 0,
                "top": order,
            })
        order += 1.0

    return blocks
