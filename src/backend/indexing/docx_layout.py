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

#: Word's style for a bulleted or numbered item.
_LIST_STYLE = "List Paragraph"

# OOXML namespaces needed to find an inline image and its relationship id.
_DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_RELATIONSHIP_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# Word stores images in their authored format; these are the ones a downstream
# extractor can actually decode. WMF/EMF vector blobs are skipped rather than
# shipped as bytes nothing in the pipeline can read.
_SUPPORTED_IMAGE_TYPES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/gif": "image/gif",
    "image/bmp": "image/bmp",
    "image/tiff": "image/tiff",
    "image/webp": "image/webp",
}


def _paragraph_image_blocks(element, document, order: float) -> List[Dict[str, Any]]:
    """Image blocks for the inline shapes in one paragraph.

    Resolved through the package relationships (r:embed -> related part) because that
    is the only mapping that survives Word's habit of reusing one image part across
    many anchors.
    """
    blocks: List[Dict[str, Any]] = []
    for blip in element.iter(f"{_DRAWING_NS}blip"):
        rel_id = blip.get(f"{_RELATIONSHIP_NS}embed")
        if not rel_id:
            continue
        try:
            part = document.part.related_parts[rel_id]
            data = part.blob
            content_type = _SUPPORTED_IMAGE_TYPES.get((part.content_type or "").lower())
        except (KeyError, AttributeError):
            continue
        if not data or not content_type:
            continue
        blocks.append({
            "type": "image",
            "content": "",
            "data": data,
            "content_type": content_type,
            "page_number": 0,
            "top": order,
        })
    return blocks


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


#: A lead-in line ends with none of these. A heading names a topic; a sentence stops.
_LEAD_IN_ENDINGS = (".", ":", "!", "?", ",", ";", "،", "؛", "؟", "۔")

#: How long a line may be and still be introducing something rather than saying it.
_LEAD_IN_MAX_CHARS = 70
_LEAD_IN_MAX_WORDS = 10

#: The level a promoted lead-in is given. Deeper than any real heading in this corpus
#: (the deepest authored is Heading 4), so promoting one never pops a real heading off
#: the stack — it only ever adds a level beneath the one already there.
_LEAD_IN_LEVEL = 5


def _is_lead_in(text: str, style_name: str, next_style: str) -> bool:
    """Whether an unstyled paragraph is really the heading of the list beneath it.

    Word documents are written by people, not by a schema, and this corpus's authors
    used `Normal` for lines that plainly head a section: "Transfer required documents",
    "Additional documents for Egyptian students", "School Direct communication Channels".
    The parser trusted styles alone, so the section stack kept the last REAL heading and
    every chunk below inherited it — "Admission Assessment" ended up labelling 23
    paragraphs, among them the transfer-document lists and the tuition fees.

    That is worse than a missing label. A prefix that names the wrong topic is a wrong
    answer written into the index, and it is what made the section path measurably hurt
    retrieval rather than merely cost space.

    Deliberately narrow: a short, punctuation-free `Normal` line IMMEDIATELY followed by
    list items. A line carrying an internal colon is a labelled value ("Preferred Method:
    Online bank transfer"), not a heading, and is left alone.
    """
    if style_name != "Normal" or next_style != "List Paragraph":
        return False
    if not text or len(text) > _LEAD_IN_MAX_CHARS or len(text.split()) > _LEAD_IN_MAX_WORDS:
        return False
    if text.endswith(_LEAD_IN_ENDINGS) or ": " in text:
        return False
    return True


def parse_docx_blocks(file_path: str) -> List[Dict[str, Any]]:
    """Parse a .docx into ordered heading/text/table/image blocks. Raises on unreadable
    files — the caller decides whether to fall back to flat text extraction."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(file_path)
    blocks: List[Dict[str, Any]] = []
    order = 0.0

    # Materialised rather than streamed, because recognising a lead-in needs to see what
    # FOLLOWS it: the line is only a heading if list items come next.
    children = list(document.element.body.iterchildren())

    # Which list a paragraph belongs to, so the chunker can decline to cut through one.
    # Word states this outright in the paragraph style and the parser used to drop it:
    # every `List Paragraph` became an ordinary `text` block, so seven bus districts
    # became seven independent units and a window boundary could land between "Maadi"
    # and "Mokattam". Measured before this: 22 of the corpus's 36 lists were split.
    list_group = 0
    in_list = False

    def next_paragraph_style(start: int) -> str:
        """The style of the next paragraph that has any text, or ""."""
        for element in children[start + 1:]:
            if not element.tag.endswith("}p"):
                return ""
            following = Paragraph(element, document)
            if (following.text or "").strip():
                return following.style.name if following.style is not None else ""
        return ""

    for index, element in enumerate(children):
        tag = element.tag
        if tag.endswith("}p"):
            paragraph = Paragraph(element, document)
            # Images are emitted before the paragraph's own text, so the caption line
            # that usually follows a figure is the next block in document order.
            for image_block in _paragraph_image_blocks(element, document, order):
                blocks.append(image_block)
                order += 1.0

            text = (paragraph.text or "").strip()
            if not text:
                continue
            style_name = paragraph.style.name if paragraph.style is not None else ""
            is_list_item = style_name == _LIST_STYLE
            if is_list_item and not in_list:
                list_group += 1
                in_list = True
                # The line that introduces a list belongs to it. "Currently, covered
                # districts:" is not a heading — it ends in a colon, which is exactly
                # what `_is_lead_in` refuses — but cutting between it and its items
                # leaves a list of place names with nothing saying what they are.
                # A HEADING counts too, and leaving it out was a bug with teeth: "The
                # school features:" is a real Heading 3, so the list below it was moved
                # to a fresh window without it and became five bare items — "Qualified
                # British Staff, Exceptional Facilities …" with nothing saying what they
                # are. That chunk then matched nothing and the list vanished from the
                # results entirely, which is the failure keeping the list together was
                # supposed to prevent.
                if blocks and blocks[-1].get("type") in ("text", "heading") and not blocks[-1].get("list_group"):
                    if len(blocks[-1].get("content") or "") <= _LEAD_IN_MAX_CHARS:
                        blocks[-1]["list_group"] = list_group
            elif not is_list_item:
                in_list = False

            level = _heading_level_from_style(style_name)
            if level is None and _is_lead_in(text, style_name, next_paragraph_style(index)):
                level = _LEAD_IN_LEVEL
            if level is not None:
                in_list = False
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
                    "list_group": list_group if is_list_item else 0,
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
