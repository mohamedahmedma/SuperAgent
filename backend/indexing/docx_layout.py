"""DOCX layout parsing to the shared block contract (heading/text/table).

Word documents carry explicit structure — real heading styles ("Heading 1"…"Heading 9",
"Title") and real table objects — so unlike the PDF parser no font heuristics are
needed: styles map directly to heading levels and tables are taken as authored (no
real-table demotion heuristic). Blocks flow through the same downstream pipeline as
PDFs (stitch → section-tag → hierarchy), which is what finally stops Word tables
from being smeared into prose by the flat Docx2txtLoader path.

DOCX has no page geometry at parse time, so page_number is always 0 and `top` is a
monotonically increasing document-order surrogate (the contract's sort key).
python-docx is imported lazily so importing this module stays cheap for non-DOCX flows.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.indexing.pdf_layout import format_table_rows, normalize_table_rows

_HEADING_STYLE_RE = re.compile(r"^heading\s*(\d+)$", re.IGNORECASE)


def _heading_level_from_style(style_name: str) -> Optional[int]:
    """"Heading N" -> N, "Title" -> 1, anything else -> None (not a heading)."""
    name = (style_name or "").strip()
    if not name:
        return None
    if name.lower() == "title":
        return 1
    match = _HEADING_STYLE_RE.match(name)
    if match:
        return int(match.group(1))
    return None


def parse_docx_blocks(file_path: str) -> List[Dict[str, Any]]:
    """Parse a .docx into ordered heading/text/table blocks. Raises on unreadable
    files — the caller decides whether to fall back to flat text extraction."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(file_path)
    blocks: List[Dict[str, Any]] = []
    order = 0.0

    for element in document.element.body.iterchildren():
        tag = element.tag
        if tag.endswith("}p"):
            paragraph = Paragraph(element, document)
            text = (paragraph.text or "").strip()
            if not text:
                continue
            style_name = paragraph.style.name if paragraph.style is not None else ""
            level = _heading_level_from_style(style_name)
            if level is not None:
                blocks.append({
                    "type": "heading",
                    "content": text,
                    "level": level,
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
        elif tag.endswith("}tbl"):
            table = Table(element, document)
            rows = normalize_table_rows(
                [[cell.text for cell in row.cells] for row in table.rows]
            )
            if rows:
                blocks.append({
                    "type": "table",
                    "content": format_table_rows(rows),
                    "rows": rows,
                    "page_number": 0,
                    "top": order,
                })
        else:
            continue
        order += 1.0

    return blocks
