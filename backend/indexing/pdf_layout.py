"""Layout-aware PDF parsing: typed heading/text/table blocks instead of flat page text.

Produces an ordered list of blocks per document:
    {"type": "heading", "content": str, "level": int | None, "page_number": int, "top": float}
    {"type": "text",    "content": str, "page_number": int, "top": float}
    {"type": "table",   "content": str, "rows": list[list[str]], "page_number": int, "top": float}

Tables are detected with pdfplumber's table finder and validated with a structural
heuristic (real data grids have short cells in consistently multi-column rows) so
prose that merely *looks* grid-aligned is demoted back to a text block. Text lines
that fall inside a detected table's bbox are excluded from paragraph extraction to
avoid duplicating table content.

Headings are classified from font signals (size vs. document body median, boldness)
plus the numbered-prefix and short-line heuristics from the DataProcessing reference
pipeline (pdf_chunk_titling.py) — which computed titles but never wired them into
chunking; here they feed the section stack in document_loader for real.

page_number is 0-based, matching the PyPDFLoader metadata the legacy path emits.
pdfplumber is imported lazily so importing this module stays cheap for non-PDF flows.
"""
from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional

# Max vertical gap (pt) between consecutive lines that still belong to one paragraph.
PARA_LINE_GAP = 15.0
# Vertical tolerance (pt) when grouping words into a line.
LINE_GROUP_TOLERANCE = 3.0
# Structural table test: a genuine grid has mostly multi-cell rows with short cells.
TABLE_MIN_MULTI_CELL_FRAC = 0.8
TABLE_MAX_AVG_CELL_LEN = 20
# Heading heuristics (short_heading_words=8 in the DataProcessing reference; we allow
# slightly longer since font signals gate the decision too).
HEADING_MAX_WORDS = 10
HEADING_MAX_CHARS = 80
HEADING_SIZE_RATIO = 1.15
# Numbered heading prefix: "1", "1.2", "2.3.4" followed by space/:/./)
NUM_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)(\s+|:|\.|\))")
# Punctuation that disqualifies a line from being a heading (headings do not end
# in sentence/clause punctuation). Includes Arabic-script marks: ، U+060C comma,
# ؛ U+061B semicolon, ؟ U+061F question mark, ۔ U+06D4 full stop.
_SENTENCE_ENDINGS = (
    ".", ",", ";", ":", "。", "，", "；", "：",
    "،", "؛", "؟", "۔",
)
# Page furniture (repeated headers/footers): a short line whose digit-masked text
# repeats on enough pages, inside the top/bottom edge bands when height is known.
FURNITURE_MIN_PAGES = 3
FURNITURE_MIN_PAGE_FRAC = 0.6
FURNITURE_EDGE_BAND_FRAC = 0.15
FURNITURE_MAX_CHARS = 80
_DIGIT_RE = re.compile(r"\d+")


def _normalize_cell(cell: Any) -> str:
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def normalize_table_rows(rows: List[List[Any]]) -> List[List[str]]:
    """Normalize extracted cells and drop rows that are entirely empty."""
    normalized: List[List[str]] = []
    for row in rows or []:
        cells = [_normalize_cell(cell) for cell in (row or [])]
        if any(cells):
            normalized.append(cells)
    return normalized


def looks_like_real_table(
    rows: List[List[str]],
    max_avg_cell_len: int = TABLE_MAX_AVG_CELL_LEN,
    min_multi_cell_frac: float = TABLE_MIN_MULTI_CELL_FRAC,
) -> bool:
    """Structural (content-agnostic) test: short cells aligned in multi-column rows.

    Table finders often mis-detect prose or Q&A layouts as tables; those produce long
    sentence cells and/or mostly single-cell rows and must stay text.
    """
    cells = [cell for row in rows for cell in row if cell]
    if not cells or len(rows) < 2:
        return False
    multi_cell_counts = [sum(1 for cell in row if cell) for row in rows]
    multi_cell_frac = sum(1 for count in multi_cell_counts if count >= 2) / max(len(rows), 1)
    avg_cell_len = sum(len(cell) for cell in cells) / len(cells)
    return multi_cell_frac >= min_multi_cell_frac and avg_cell_len <= max_avg_cell_len


def format_table_rows(rows: List[List[str]]) -> str:
    return "\n".join(" | ".join(row) for row in rows)


def flatten_table_to_text(rows: List[List[str]]) -> str:
    """Rejected-table fallback: rejoin cells as plain prose lines."""
    return "\n".join(" ".join(cell for cell in row if cell) for row in rows)


def merge_lines_to_paragraphs(
    lines: List[Dict[str, Any]],
    gap_threshold: float = PARA_LINE_GAP,
) -> List[Dict[str, Any]]:
    """Merge consecutive text lines into paragraphs by vertical proximity.

    Input lines need "text" and "top"; output blocks are {"type": "text", "content", "top"}.
    """
    kept = [line for line in lines if (line.get("text") or "").strip()]
    if not kept:
        return []

    kept.sort(key=lambda line: line["top"])
    paragraphs: List[Dict[str, Any]] = []
    buffer = [kept[0]]
    last_top = kept[0]["top"]

    for line in kept[1:]:
        if abs(line["top"] - last_top) <= gap_threshold:
            buffer.append(line)
        else:
            paragraphs.append({
                "type": "text",
                "content": "\n".join(item["text"].strip() for item in buffer),
                "top": buffer[0]["top"],
            })
            buffer = [line]
        last_top = line["top"]

    paragraphs.append({
        "type": "text",
        "content": "\n".join(item["text"].strip() for item in buffer),
        "top": buffer[0]["top"],
    })
    return paragraphs


# ---------------------------------------------------------------------------
# Heading classification
# ---------------------------------------------------------------------------

def _line_font_size(chars: List[dict]) -> Optional[float]:
    sizes = [c.get("size") for c in chars or [] if isinstance(c.get("size"), (int, float))]
    if not sizes:
        return None
    return float(statistics.median(sizes))


def _line_is_bold(chars: List[dict]) -> bool:
    names = [str(c.get("fontname", "")).lower() for c in chars or []]
    if not names:
        return False
    bold = sum(1 for name in names if "bold" in name)
    return bold >= max(len(names) // 2, 1)


def body_font_size(pages: List[Dict[str, Any]]) -> Optional[float]:
    """Document-wide body font size: the median line size, weighted toward prose by
    only counting lines with at least three words (headings are usually shorter)."""
    sizes: List[float] = []
    for page in pages:
        for line in page.get("lines", []):
            text = (line.get("text") or "").strip()
            size = line.get("size")
            if size is None or len(text.split()) < 3:
                continue
            sizes.append(float(size))
    if not sizes:
        # Fall back to all lines (short-line-only documents).
        for page in pages:
            for line in page.get("lines", []):
                if line.get("size") is not None:
                    sizes.append(float(line["size"]))
    return float(statistics.median(sizes)) if sizes else None


def is_heading_line(line: Dict[str, Any], body_size: Optional[float]) -> bool:
    """Short line + at least one strong signal (larger font, bold, numbered prefix,
    all-caps). Sentence-ending punctuation disqualifies (headings don't end in '.')."""
    text = (line.get("text") or "").strip()
    if not text or len(text) > HEADING_MAX_CHARS:
        return False
    if len(text.split()) > HEADING_MAX_WORDS:
        return False
    if text.endswith(_SENTENCE_ENDINGS):
        return False

    size = line.get("size")
    if body_size and size and float(size) >= body_size * HEADING_SIZE_RATIO:
        return True
    if line.get("bold"):
        return True
    if NUM_HEADING_RE.match(text):
        return True
    letters = [ch for ch in text if ch.isalpha()]
    cased = [ch for ch in letters if ch.isupper() or ch.islower()]
    # The all-caps signal only means something in a bicameral script. `isupper()`
    # is True for caseless text containing a single Latin acronym, so an ordinary
    # Arabic sentence like "يجب تقديم شهادة IELTS للتسجيل" would otherwise be
    # promoted to a heading — very common in this corpus (IGCSE, IB, SAT, KG).
    # Require the line to be predominantly cased before trusting the signal.
    if len(letters) >= 4 and cased and len(cased) >= 0.8 * len(letters) and text.isupper():
        return True
    return False


def heading_level(text: str, size: Optional[float], size_rank: Dict[float, int]) -> Optional[int]:
    """Numbered prefixes give an explicit level ("2.3 Fees" -> 2); otherwise rank the
    heading's font size against all heading sizes in the document (largest = 1).
    Returns None when neither signal resolves a level."""
    match = NUM_HEADING_RE.match(text.strip())
    if match:
        return match.group(1).count(".") + 1
    if size is not None:
        rank = size_rank.get(round(float(size), 1))
        if rank is not None:
            return rank
    return None


def _furniture_key(text: str) -> str:
    """Digit-masked normalization so 'Page 1 of 9' and 'Page 2 of 9' count as the
    same repeated furniture line."""
    return _DIGIT_RE.sub("#", " ".join(text.lower().split()))


def _in_edge_band(line: Dict[str, Any], page_height: Optional[float]) -> bool:
    """Whether the line sits in the header/footer band. Unknown height means the
    band test can't apply, so any position qualifies (text-repeat rule still gates)."""
    if not page_height:
        return True
    center = (line.get("top", 0.0) + line.get("bottom", line.get("top", 0.0))) / 2
    band = page_height * FURNITURE_EDGE_BAND_FRAC
    return center <= band or center >= page_height - band


def remove_page_furniture(
    pages: List[Dict[str, Any]],
    min_pages: int = FURNITURE_MIN_PAGES,
    min_page_frac: float = FURNITURE_MIN_PAGE_FRAC,
) -> List[Dict[str, Any]]:
    """Drop repeated header/footer lines before block building, so boilerplate is
    never chunked or indexed N times. A line is furniture when its digit-masked
    text appears in the edge band of at least min_pages pages AND at least
    min_page_frac of all pages. Short documents are left untouched — repetition
    can't be established reliably below min_pages."""
    if len(pages) < min_pages:
        return pages

    pages_seen: Dict[str, set] = {}
    for page in pages:
        height = page.get("height")
        for line in page.get("lines", []):
            text = (line.get("text") or "").strip()
            if not text or len(text) > FURNITURE_MAX_CHARS:
                continue
            if not _in_edge_band(line, height):
                continue
            pages_seen.setdefault(_furniture_key(text), set()).add(page.get("page_number"))

    required = max(min_pages, int(min_page_frac * len(pages) + 0.999))
    furniture_keys = {key for key, seen in pages_seen.items() if len(seen) >= required}
    if not furniture_keys:
        return pages

    cleaned: List[Dict[str, Any]] = []
    for page in pages:
        height = page.get("height")
        kept_lines = [
            line for line in page.get("lines", [])
            if not (
                _in_edge_band(line, height)
                and _furniture_key((line.get("text") or "").strip()) in furniture_keys
            )
        ]
        cleaned.append({**page, "lines": kept_lines})
    return cleaned


def build_blocks_from_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pure block builder over pre-collected page data (unit-testable without I/O).

    Each page dict: {"page_number": int, "height": float | None,
                     "tables": [{"bbox": (x0, top, x1, bottom), "rows": [[cell, ...]]}],
                     "lines": [{"text", "top", "bottom", "x0", "x1", "size", "bold"}]}
    """
    pages = remove_page_furniture(pages)
    body_size = body_font_size(pages)

    # First pass: find heading candidates so font sizes can be ranked document-wide.
    heading_sizes: set = set()
    page_partitions: List[Dict[str, Any]] = []
    for page in pages:
        table_bboxes = [table["bbox"] for table in page.get("tables", [])]
        heading_lines: List[Dict[str, Any]] = []
        body_lines: List[Dict[str, Any]] = []
        for line in page.get("lines", []):
            cx = (line.get("x0", 0.0) + line.get("x1", 0.0)) / 2
            cy = (line.get("top", 0.0) + line.get("bottom", line.get("top", 0.0))) / 2
            if any(_point_in_bbox(cx, cy, bbox) for bbox in table_bboxes):
                continue
            if is_heading_line(line, body_size):
                heading_lines.append(line)
                if line.get("size") is not None:
                    heading_sizes.add(round(float(line["size"]), 1))
            else:
                body_lines.append(line)
        page_partitions.append({
            "page": page,
            "heading_lines": heading_lines,
            "body_lines": body_lines,
        })

    size_rank = {size: rank + 1 for rank, size in enumerate(sorted(heading_sizes, reverse=True))}

    blocks: List[Dict[str, Any]] = []
    for partition in page_partitions:
        page = partition["page"]
        page_number = int(page.get("page_number", 0))
        page_blocks: List[Dict[str, Any]] = []

        for table in page.get("tables", []):
            rows = normalize_table_rows(table.get("rows") or [])
            if not rows:
                continue
            top = float(table["bbox"][1])
            if looks_like_real_table(rows):
                page_blocks.append({
                    "type": "table",
                    "content": format_table_rows(rows),
                    "rows": rows,
                    "page_number": page_number,
                    "top": top,
                })
            else:
                page_blocks.append({
                    "type": "text",
                    "content": flatten_table_to_text(rows),
                    "page_number": page_number,
                    "top": top,
                })

        for line in partition["heading_lines"]:
            text = line["text"].strip()
            page_blocks.append({
                "type": "heading",
                "content": text,
                "level": heading_level(text, line.get("size"), size_rank),
                "page_number": page_number,
                "top": float(line["top"]),
            })

        for paragraph in merge_lines_to_paragraphs(partition["body_lines"]):
            paragraph["page_number"] = page_number
            page_blocks.append(paragraph)

        page_blocks.sort(key=lambda block: block["top"])
        blocks.extend(page_blocks)

    return blocks


def _point_in_bbox(cx: float, cy: float, bbox) -> bool:
    x0, top, x1, bottom = bbox
    return x0 <= cx <= x1 and top <= cy <= bottom


def _extract_page_lines(page) -> List[Dict[str, Any]]:
    """Extract positioned text lines with font metadata, preferring pdfplumber's
    native line extraction (its char lists carry size/fontname)."""
    if hasattr(page, "extract_text_lines"):
        lines = page.extract_text_lines()
        return [
            {
                "text": line.get("text", ""),
                "top": line.get("top", 0.0),
                "bottom": line.get("bottom", line.get("top", 0.0)),
                "x0": line.get("x0", 0.0),
                "x1": line.get("x1", 0.0),
                "size": _line_font_size(line.get("chars") or []),
                "bold": _line_is_bold(line.get("chars") or []),
            }
            for line in lines
        ]

    # Older pdfplumber: rebuild lines from words grouped by vertical position.
    words = page.extract_words(extra_attrs=["size", "fontname"])
    if not words:
        return []
    words.sort(key=lambda w: (w["top"], w["x0"]))
    lines: List[Dict[str, Any]] = []
    current: List[dict] = [words[0]]
    for word in words[1:]:
        if abs(word["top"] - current[0]["top"]) <= LINE_GROUP_TOLERANCE:
            current.append(word)
        else:
            lines.append(_words_to_line(current))
            current = [word]
    lines.append(_words_to_line(current))
    return lines


def _words_to_line(words: List[dict]) -> Dict[str, Any]:
    words = sorted(words, key=lambda w: w["x0"])
    sizes = [w.get("size") for w in words if isinstance(w.get("size"), (int, float))]
    names = [str(w.get("fontname", "")).lower() for w in words]
    return {
        "text": " ".join(w["text"] for w in words),
        "top": min(w["top"] for w in words),
        "bottom": max(w.get("bottom", w["top"]) for w in words),
        "x0": min(w["x0"] for w in words),
        "x1": max(w.get("x1", w["x0"]) for w in words),
        "size": float(statistics.median(sizes)) if sizes else None,
        "bold": bool(names) and sum(1 for n in names if "bold" in n) >= max(len(names) // 2, 1),
    }


def parse_pdf_blocks(file_path: str) -> List[Dict[str, Any]]:
    """Parse a PDF into ordered heading/text/table blocks. Raises on unreadable
    files — the caller decides whether to fall back to flat text extraction."""
    import pdfplumber

    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            found_tables = page.find_tables()
            pages.append({
                "page_number": page_number,
                "height": float(page.height) if getattr(page, "height", None) else None,
                "tables": [
                    {"bbox": table.bbox, "rows": table.extract()}
                    for table in found_tables
                ],
                "lines": _extract_page_lines(page),
            })
    return build_blocks_from_pages(pages)
