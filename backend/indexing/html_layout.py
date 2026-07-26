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

from pathlib import Path
from typing import Any, Dict, List

import backend.indexing.html_processor as html_processor
from backend.indexing.pdf_layout import (
    flatten_table_to_text,
    format_table_rows,
    looks_like_real_table,
    normalize_table_rows,
)

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_BLOCK_TAGS = list(_HEADING_TAGS) + ["p", "li", "pre", "table"]


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

    html = html_processor._read_html_text(Path(file_path))
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
