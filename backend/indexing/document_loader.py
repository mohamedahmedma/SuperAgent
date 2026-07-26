"""Document loading and chunking service"""
import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.indexing.docx_layout import parse_docx_blocks
from backend.indexing.html_layout import parse_html_blocks
from backend.indexing.pdf_layout import parse_pdf_blocks
from backend.indexing.xlsx_layout import parse_xlsx_blocks

logger = logging.getLogger(__name__)


def _read_positive_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 1)
    except ValueError:
        logger.warning("Invalid %s=%r — using default %d", name, raw, default)
        return default


# Layout-aware parsing (typed heading/text/table blocks) for all structured formats.
# LAYOUT_PARSER_ENABLED is the master switch; PDF_LAYOUT_PARSER_ENABLED is kept as a
# PDF-specific override for backward compatibility. Any layout-parser failure falls
# back to the legacy flat loader per file.
LAYOUT_PARSER_ENABLED = os.getenv("LAYOUT_PARSER_ENABLED", "true").lower() != "false"
PDF_LAYOUT_PARSER_ENABLED = (
    os.getenv("PDF_LAYOUT_PARSER_ENABLED", "true" if LAYOUT_PARSER_ENABLED else "false").lower() != "false"
)

# Leaf chunks merge small neighbors up to ~level_3_size / divisor (a topic-sized
# retrieval granule), while level_3_size stays only a hard cap — mirrors
# MERGE_TARGET_DIVISOR in the DataProcessing reference (chunk_merger.py).
MERGE_TARGET_DIVISOR = 4

# Chunking configuration (tune without code changes). CHUNK_SIZE / CHUNK_OVERLAP are
# the leaf (L3) budgets in CHARACTERS for every strategy — the token strategy
# converts internally (~4 chars/token) so there is exactly one unit across the
# config, avoiding the chars-vs-tokens ambiguity the DataProcessing reference had.
# L1/L2 derive from the leaf size unless CHUNK_L1_SIZE / CHUNK_L2_SIZE override them.
CHUNK_STRATEGIES = ("recursive", "sentence", "token")
_TOKEN_CHARS_ESTIMATE = 4


class SentenceSplitter:
    """Sentence-boundary splitter (port of the DataProcessing reference
    SentenceBasedChunker) with two fixes: budgets are in characters, matching every
    other strategy here, and overlap is expressed in SENTENCES — the reference
    passed its character overlap count into overlap_sentences, ballooning chunks
    into near-duplicates of each other."""

    # Terminators across Latin, CJK, and Arabic script (؟ U+061F question mark,
    # ۔ U+06D4 Urdu full stop). Arabic ، (comma) is deliberately absent — it is a
    # separator, not a terminator, and splitting on it shreds Arabic sentences.
    _SENTENCE_RE = re.compile(r"(?<=[.!?。！？؟۔])\s+")

    def __init__(self, max_chars: int, overlap_sentences: int = 1):
        self.max_chars = max(int(max_chars), 1)
        self.overlap_sentences = max(int(overlap_sentences), 0)

    def split_text(self, text: str) -> List[str]:
        sentences = [s.strip() for s in self._SENTENCE_RE.split((text or "").strip()) if s.strip()]
        if not sentences:
            return []
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) + 1 > self.max_chars:
                chunks.append(" ".join(current))
                overlap = current[-self.overlap_sentences:] if self.overlap_sentences else []
                current = list(overlap)
                current_len = sum(len(s) + 1 for s in current)
            current.append(sentence)
            current_len += len(sentence) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

# Text normalization is shared with the retrieval layer so documents and queries
# are normalized identically (see backend/text_normalization.py). Re-exported here
# because callers have always imported sanitize_text from this module.
from backend.text_normalization import fit_utf8_bytes, sanitize_text  # noqa: E402


class DocumentLoader:
    """Document loading and chunking service"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        chunk_size = chunk_size or _read_positive_int_env("CHUNK_SIZE", 800)
        chunk_overlap = chunk_overlap if chunk_overlap is not None else _read_positive_int_env("CHUNK_OVERLAP", 100)
        self._strategy = self._resolve_strategy()

        level_1_size = _read_positive_int_env("CHUNK_L1_SIZE", max(2000, chunk_size * 3))
        level_1_overlap = max(400, chunk_overlap * 3)
        level_2_size = _read_positive_int_env("CHUNK_L2_SIZE", max(1000, chunk_size * 2))
        level_2_overlap = max(200, chunk_overlap * 2)
        level_3_size = max(600, chunk_size)
        level_3_overlap = max(100, chunk_overlap)

        self._splitter_level_1 = self._make_splitter(level_1_size, level_1_overlap)
        self._splitter_level_2 = self._make_splitter(level_2_size, level_2_overlap)
        self._splitter_level_3 = self._make_splitter(level_3_size, level_3_overlap)
        self._level_1_size = level_1_size
        self._level_2_size = level_2_size
        self._level_3_size = level_3_size

    @staticmethod
    def _resolve_strategy() -> str:
        strategy = (os.getenv("CHUNK_STRATEGY") or "recursive").strip().lower()
        if strategy not in CHUNK_STRATEGIES:
            logger.warning(
                "Unknown CHUNK_STRATEGY=%r (choices: %s) — using 'recursive'",
                strategy, ", ".join(CHUNK_STRATEGIES),
            )
            return "recursive"
        if strategy == "token":
            try:
                import tiktoken  # noqa: F401 — availability check only
            except ImportError:
                logger.warning("CHUNK_STRATEGY=token but tiktoken is not installed — using 'recursive'")
                return "recursive"
        return strategy

    def _make_splitter(self, size: int, overlap: int):
        """Splitter factory: any object with split_text(str) -> list[str] plugs in,
        so new strategies are additions here, not changes elsewhere."""
        if self._strategy == "sentence":
            return SentenceSplitter(
                max_chars=size,
                overlap_sentences=_read_positive_int_env("CHUNK_SENTENCE_OVERLAP", 1),
            )
        if self._strategy == "token":
            from langchain_text_splitters import TokenTextSplitter

            return TokenTextSplitter(
                encoding_name="cl100k_base",
                chunk_size=max(size // _TOKEN_CHARS_ESTIMATE, 1),
                chunk_overlap=max(overlap // _TOKEN_CHARS_ESTIMATE, 0),
            )
        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            add_start_index=True,
            separators=["\n\n", "。", "！", "？", "\n", "，", "、", " ", ""],
        )

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
        counters: Optional[Dict[str, int]] = None,
    ) -> List[Dict]:
        """`counters` carries per-level ID counters across multiple text runs on the
        same page (layout parsing splits a page at table boundaries); without it a
        second run would restart at 0 and collide on chunk IDs."""
        if not text:
            return []
        if counters is None:
            counters = {"l1": 0, "l2": 0, "l3": 0}

        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(filename, page_number, 1, counters["l1"])
            counters["l1"] += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(filename, page_number, 2, counters["l2"])
                counters["l2"] += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue
                    level_3_id = self._build_chunk_id(filename, page_number, 3, counters["l3"])
                    counters["l3"] += 1
                    root_chunks.append({
                        **base_doc,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks

    # Milvus VARCHAR limits are in BYTES (see milvus_client.ensure_collection).
    # Chunk texts are therefore capped on UTF-8 length, not character count.
    _MILVUS_TEXT_CAP_BYTES = 60000
    _MILVUS_FILENAME_CAP_BYTES = 255
    _MILVUS_FILE_PATH_CAP_BYTES = 1024

    @staticmethod
    def _render_rows(rows: List[List[str]]) -> str:
        return "\n".join(" | ".join(row) for row in rows if row)

    def _split_table_row_groups(self, rows: List[List[str]], budget: int) -> List[List[List[str]]]:
        """Group table rows into budget-sized groups with the header row repeated per
        group, so every group stays self-describing. A single row larger than the
        budget still ships (with its header) rather than being cut mid-row."""
        rows = [row for row in rows if row]
        if not rows:
            return []
        if len(self._render_rows(rows)) <= budget:
            return [rows]

        header = rows[0]
        header_len = len(" | ".join(header))
        groups: List[List[List[str]]] = []
        current: List[List[str]] = [header]
        current_len = header_len
        for row in rows[1:]:
            line_len = len(" | ".join(row)) + 1
            if len(current) > 1 and current_len + line_len > budget:
                groups.append(current)
                current = [header]
                current_len = header_len
            current.append(row)
            current_len += line_len
        # `not groups` covers a single-row table larger than the budget: without it
        # the header-only group is discarded and the table silently vanishes from
        # the index (a data-loss bug the DataProcessing reference merger shared).
        if len(current) > 1 or not groups:
            groups.append(current)
        return groups

    def _table_units(
        self,
        rows: List[List[str]],
        budget: int,
        sections: tuple = (),
        page: int = 0,
    ) -> List[Dict]:
        return [
            {
                "kind": "table",
                "rows": group,
                "text": self._render_rows(group),
                "sections": sections,
                "page": page,
            }
            for group in self._split_table_row_groups(rows, budget)
        ]

    @staticmethod
    def _text_units(
        text: str,
        splitter,
        sections: tuple = (),
        page: int = 0,
    ) -> List[Dict]:
        units: List[Dict] = []
        for piece in splitter.split_text(text):
            piece = (piece or "").strip()
            if piece:
                units.append({"kind": "text", "text": piece, "sections": sections, "page": page})
        return units

    def _refine_units(
        self,
        units: List[Dict],
        splitter,
        table_budget: int,
    ) -> List[Dict]:
        """Re-split a window's units at the next (finer) level: text through the
        level's splitter, tables re-grouped by rows against the level's budget.
        Refined pieces inherit their source unit's section path and page."""
        refined: List[Dict] = []
        for unit in units:
            sections = unit.get("sections", ())
            page = unit.get("page", 0)
            if unit["kind"] == "table":
                refined.extend(self._table_units(unit["rows"], table_budget, sections, page))
            else:
                refined.extend(self._text_units(unit["text"], splitter, sections, page))
        return refined

    @staticmethod
    def _pack_units(
        units: List[Dict],
        budget: int,
        target: Optional[int] = None,
        isolate_tables: bool = False,
        split_on_section_change: bool = False,
    ) -> List[List[Dict]]:
        """Greedily pack consecutive units into windows, preserving document order.

        `budget` is the hard cap; `target` (when set, at leaf level) stops packing
        once a window is topic-sized, so small fragments merge without gluing
        distinct passages into one embedding — the merge_small_chunks semantics of
        the DataProcessing reference. With isolate_tables, a table unit always gets
        its own window so leaf table chunks stay pure — their section context lives
        in the L1/L2 parents, which do mix tables with surrounding text. With
        split_on_section_change, a window never spans two different section paths,
        keeping leaves citable under one topic.
        """
        effective_target = target or budget
        windows: List[List[Dict]] = []
        current: List[Dict] = []
        current_len = 0

        def close():
            nonlocal current, current_len
            if current:
                windows.append(current)
                current = []
                current_len = 0

        for unit in units:
            unit_len = len(unit["text"])
            if isolate_tables and unit["kind"] == "table":
                close()
                windows.append([unit])
                continue
            if split_on_section_change and current and unit.get("sections", ()) != current[0].get("sections", ()):
                close()
            if current and (current_len >= effective_target or current_len + 2 + unit_len > budget):
                close()
            current.append(unit)
            current_len += unit_len + (2 if current_len else 0)
        close()
        return windows

    @staticmethod
    def _window_text(window: List[Dict]) -> str:
        return "\n\n".join(unit["text"] for unit in window)

    @staticmethod
    def _window_sections(window: List[Dict]) -> List[str]:
        return list(window[0].get("sections", ())) if window else []

    @staticmethod
    def _apply_section_prefix(text: str, sections: List[str]) -> str:
        """Prepend the section path ("Admissions > Fees") so the chunk carries its
        topic into the embedding, BM25 index, and citations. Skipped when the
        chunk already contains its own heading near the top."""
        if not text or not sections:
            return text
        if sections[-1] in text[:200]:
            return text
        prefix = " > ".join(sections[-3:])[:150].strip()
        return f"{prefix}\n{text}" if prefix else text

    # Sentence-terminal punctuation: a text block ending with one of these is a
    # finished paragraph, not a page-break continuation candidate. Covers Latin,
    # CJK, and Arabic-script terminators (؟ U+061F, ؛ U+061B, ۔ U+06D4) — Arabic
    # sentences end with these, never with the Latin set alone.
    _TERMINAL_PUNCTUATION = (
        ".", "!", "?", "。", "！", "？", ":", "：", ";", "；", '"', ")", "）",
        "؟", "؛", "۔",
    )

    @staticmethod
    def _starts_like_continuation(text: str) -> bool:
        """Whether the first alphabetic character looks like a mid-sentence
        continuation. Only an explicit UPPERCASE letter rules it out: unicameral
        scripts (Arabic, Hebrew, CJK, Thai, Devanagari…) have no case at all, so
        a case-positive test would reject every one of their fragments and no
        Arabic paragraph split by a page break would ever be rejoined."""
        for ch in text:
            if ch.isalpha():
                return not ch.isupper()
            if not ch.isspace():
                return False
        return False

    @classmethod
    def _is_paragraph_continuation(cls, prev: Dict, block: Dict) -> bool:
        """Adjacent text blocks across a page break where the first ends mid-sentence
        and the second reads like its continuation."""
        if prev.get("type") != "text" or block.get("type") != "text":
            return False
        if block.get("page_number") != prev.get("_last_page", prev.get("page_number", 0)) + 1:
            return False
        prev_text = (prev.get("content") or "").rstrip()
        next_text = (block.get("content") or "").lstrip()
        if not prev_text or not next_text:
            return False
        if prev_text.endswith(cls._TERMINAL_PUNCTUATION):
            return False
        return cls._starts_like_continuation(next_text)

    @staticmethod
    def _is_table_continuation(prev: Dict, block: Dict) -> bool:
        """Adjacent table blocks across a page break with the same column count are
        one table split by the break (conservative structural heuristic)."""
        if prev.get("type") != "table" or block.get("type") != "table":
            return False
        if block.get("page_number") != prev.get("_last_page", prev.get("page_number", 0)) + 1:
            return False
        prev_rows = prev.get("rows") or []
        next_rows = block.get("rows") or []
        if not prev_rows or not next_rows:
            return False
        return len(prev_rows[0]) == len(next_rows[0])

    def _stitch_cross_page_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Stitching stage: rejoin paragraphs and tables that the page break split.

        Adjacency in the block stream is the position signal (the previous block was
        the last on its page, the candidate the first on the next). A stitched block
        keeps its START page for citations and tracks its last page internally so
        tables spanning three or more pages chain correctly. A repeated header row
        on the continuation table is dropped.
        """
        stitched: List[Dict] = []
        for block in blocks:
            prev = stitched[-1] if stitched else None
            if prev is not None and self._is_paragraph_continuation(prev, block):
                prev["content"] = prev["content"].rstrip() + " " + (block.get("content") or "").lstrip()
                prev["_last_page"] = block.get("page_number", 0)
                continue
            if prev is not None and self._is_table_continuation(prev, block):
                next_rows = list(block.get("rows") or [])
                if next_rows and next_rows[0] == (prev.get("rows") or [[]])[0]:
                    next_rows = next_rows[1:]
                prev["rows"] = list(prev.get("rows") or []) + next_rows
                prev["content"] = self._render_rows(prev["rows"])
                prev["_last_page"] = block.get("page_number", 0)
                continue
            stitched.append(dict(block))
        return stitched

    @staticmethod
    def _update_section_stack(stack: List[Dict], title: str, level: Optional[int]) -> None:
        """Maintain the heading hierarchy stack (DataProcessing pdf_chunk_titling's
        heading-stack idea): a known level pops everything at its level or deeper;
        an unknown level replaces the current deepest heading. Depth-capped so a
        misclassified run can't grow an absurd path."""
        if level is not None:
            while stack and (stack[-1]["level"] is None or stack[-1]["level"] >= level):
                stack.pop()
        elif stack and stack[-1]["level"] is None:
            stack.pop()
        while len(stack) >= 4:
            stack.pop()
        stack.append({"title": title, "level": level})

    def _blocks_to_units(self, blocks: List[Dict]) -> List[Dict]:
        """Section-tagging stage: heading blocks maintain the section stack — which
        persists ACROSS pages, so a section's topic carries onto its continuation
        pages — and ride along as text units so parents contain them verbatim.
        Text/table blocks become units tagged with their section path and start page."""
        units: List[Dict] = []
        section_stack: List[Dict] = []
        for block in blocks:
            page_number = int(block.get("page_number", 0))
            block_type = block.get("type")
            if block_type == "heading":
                title = sanitize_text(block.get("content") or "").strip()
                if title:
                    self._update_section_stack(section_stack, title, block.get("level"))
                    units.append({
                        "kind": "text",
                        "text": title,
                        "sections": tuple(entry["title"] for entry in section_stack),
                        "page": page_number,
                    })
                continue

            sections = tuple(entry["title"] for entry in section_stack)
            if block_type == "table":
                units.extend(
                    self._table_units(block.get("rows") or [], self._level_1_size, sections, page_number)
                )
            else:
                content = (block.get("content") or "").strip()
                if content:
                    units.extend(
                        self._text_units(content, self._splitter_level_1, sections, page_number)
                    )
        return units

    def _hierarchy_chunks(self, units: List[Dict], doc_info: Dict) -> List[Dict]:
        """Hierarchy stage: build the standard L1→L2→L3 chunks over the whole
        document's unit stream — windows may span page breaks, so a section that
        continues onto the next page chunks as one context instead of resetting.

        Tables are normal citizens of the hierarchy: they pack into the same L1/L2
        section windows as their surrounding paragraphs, so a table leaf's parents
        contain the table plus its section context, and auto-merging a table hit
        yields table-with-topic rather than a floating grid. Only at leaf level is
        a table isolated into its own chunk (never split mid-row, never diluted
        with unrelated prose in one embedding).

        Leaf packing stops at a topic-sized merge target (level_3_size /
        MERGE_TARGET_DIVISOR) and never crosses a section boundary; every chunk is
        prefixed with its section path when its own heading isn't already inside.
        Chunk IDs use document-level per-level counters; each chunk's page_number
        is the page its window STARTS on (the citation anchor).
        """
        filename = doc_info["filename"]
        leaf_merge_target = max(self._level_3_size // MERGE_TARGET_DIVISOR, 1)
        counters: Dict[str, int] = {"l1": 0, "l2": 0, "l3": 0}
        chunks: List[Dict] = []
        chunk_idx = 0

        def window_page(window: List[Dict]) -> int:
            return int(window[0].get("page", 0)) if window else 0

        for window_1 in self._pack_units(units, self._level_1_size):
            level_1_text = sanitize_text(self._window_text(window_1)).strip()
            level_1_text = self._apply_section_prefix(level_1_text, self._window_sections(window_1))
            if not level_1_text:
                continue
            level_1_page = window_page(window_1)
            level_1_id = self._build_chunk_id(filename, level_1_page, 1, counters["l1"])
            counters["l1"] += 1
            chunks.append({
                **doc_info,
                "page_number": level_1_page,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": chunk_idx,
            })
            chunk_idx += 1

            units_2 = self._refine_units(window_1, self._splitter_level_2, self._level_2_size)
            for window_2 in self._pack_units(units_2, self._level_2_size):
                level_2_text = sanitize_text(self._window_text(window_2)).strip()
                level_2_text = self._apply_section_prefix(level_2_text, self._window_sections(window_2))
                if not level_2_text:
                    continue
                level_2_page = window_page(window_2)
                level_2_id = self._build_chunk_id(filename, level_2_page, 2, counters["l2"])
                counters["l2"] += 1
                chunks.append({
                    **doc_info,
                    "page_number": level_2_page,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": chunk_idx,
                })
                chunk_idx += 1

                units_3 = self._refine_units(window_2, self._splitter_level_3, self._level_3_size)
                for window_3 in self._pack_units(
                    units_3,
                    self._level_3_size,
                    target=leaf_merge_target,
                    isolate_tables=True,
                    split_on_section_change=True,
                ):
                    level_3_text = sanitize_text(self._window_text(window_3)).strip()
                    level_3_text = self._apply_section_prefix(level_3_text, self._window_sections(window_3))
                    level_3_text = fit_utf8_bytes(level_3_text, self._MILVUS_TEXT_CAP_BYTES).strip()
                    if not level_3_text:
                        continue
                    level_3_page = window_page(window_3)
                    level_3_id = self._build_chunk_id(filename, level_3_page, 3, counters["l3"])
                    counters["l3"] += 1
                    chunks.append({
                        **doc_info,
                        "page_number": level_3_page,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": chunk_idx,
                    })
                    chunk_idx += 1

        return chunks

    def _load_blocks_with_layout(
        self,
        blocks: List[Dict],
        file_path: str,
        filename: str,
        doc_type: str,
    ) -> list[dict]:
        """Shared layout pipeline (pipes-and-filters) behind every format parser:

            <format>_blocks parser → typed heading/text/table blocks (I/O boundary)
            _stitch_cross_page_blocks → rejoin paragraphs/tables split by page breaks
            _blocks_to_units      → section-tagged, page-tagged units
            _hierarchy_chunks     → L1/L2/L3 chunks over the whole document

        Each stage is a pure transformation of its input, so supporting a new file
        format only means writing a block parser — the rest of the pipeline (and
        every future stage added to it) is shared.
        """
        blocks = self._stitch_cross_page_blocks(blocks)
        units = self._blocks_to_units(blocks)
        doc_info = {
            "filename": sanitize_text(filename),
            "file_path": sanitize_text(file_path),
            "file_type": doc_type,
        }
        return self._hierarchy_chunks(units, doc_info)

    def _try_layout_path(self, parse_blocks, file_path: str, filename: str, doc_type: str):
        """Run a format's block parser through the shared pipeline; None means the
        caller should fall back to its legacy flat loader (failure or empty result)."""
        try:
            documents = self._load_blocks_with_layout(parse_blocks(file_path), file_path, filename, doc_type)
            if documents:
                return documents
            logger.warning(
                "Layout parser produced no chunks for %s (%s); falling back to flat extraction",
                filename, doc_type,
            )
        except Exception:
            logger.exception(
                "Layout-aware parsing failed for %s (%s); falling back to flat extraction",
                filename, doc_type,
            )
        return None

    def _load_from_langchain_docs(
        self,
        raw_docs: list,
        file_path: str,
        filename: str,
        doc_type: str,
    ) -> list[dict]:
        documents: list[dict] = []
        page_global_chunk_idx = 0
        for doc in raw_docs:
            meta = getattr(doc, "metadata", None) or {}
            page_num = meta.get("page", 0)
            if page_num is None:
                page_num = 0
            try:
                page_num = int(page_num)
            except (TypeError, ValueError):
                page_num = 0
            base_doc = {
                "filename": sanitize_text(filename),
                "file_path": sanitize_text(file_path),
                "file_type": sanitize_text(doc_type),
                "page_number": page_num,
            }
            page_chunks = self._split_page_to_three_levels(
                text=sanitize_text((doc.page_content or "").strip()),
                base_doc=base_doc,
                page_global_chunk_idx=page_global_chunk_idx,
            )
            page_global_chunk_idx += len(page_chunks)
            documents.extend(page_chunks)
        return documents

    def _validate_identifiers(self, file_path: str, filename: str) -> None:
        """Fail fast, with an actionable message, when identifiers cannot fit the
        Milvus VARCHAR columns.

        These are BYTE limits, so a perfectly ordinary Arabic or Chinese filename
        (2-3 bytes per character) can exceed 255 bytes at well under 255
        characters. Truncating instead would silently break delete-by-filename
        and citations, orphaning vectors; failing here costs the user a rename
        but never corrupts the index, and it happens BEFORE any embedding spend
        rather than as an opaque Milvus error mid-batch.
        """
        checks = (
            ("filename", filename, self._MILVUS_FILENAME_CAP_BYTES),
            ("file path", file_path, self._MILVUS_FILE_PATH_CAP_BYTES),
        )
        for label, value, limit in checks:
            size = len((value or "").encode("utf-8"))
            if size > limit:
                raise ValueError(
                    f"{label} is too long for the index: {size} bytes (limit {limit}). "
                    f"Note the limit is in BYTES — non-Latin characters cost 2-3 bytes each. "
                    f"Please shorten it and upload again."
                )

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        self._validate_identifiers(file_path, filename)
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            if PDF_LAYOUT_PARSER_ENABLED:
                documents = self._try_layout_path(parse_pdf_blocks, file_path, filename, doc_type)
                if documents:
                    return documents
            loader = PyPDFLoader(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            # python-docx reads only .docx; legacy .doc always uses the flat loader.
            if LAYOUT_PARSER_ENABLED and file_lower.endswith(".docx"):
                documents = self._try_layout_path(parse_docx_blocks, file_path, filename, doc_type)
                if documents:
                    return documents
            loader = Docx2txtLoader(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            # openpyxl reads only .xlsx; legacy .xls always uses the flat loader.
            if LAYOUT_PARSER_ENABLED and file_lower.endswith(".xlsx"):
                documents = self._try_layout_path(parse_xlsx_blocks, file_path, filename, doc_type)
                if documents:
                    return documents
            loader = UnstructuredExcelLoader(file_path)
        elif file_lower.endswith((".html", ".htm")):
            doc_type = "HTML"
            if LAYOUT_PARSER_ENABLED:
                documents = self._try_layout_path(parse_html_blocks, file_path, filename, doc_type)
                if documents:
                    return documents
            from backend.indexing.html_processor import load_html_for_document_loader

            raw_docs = load_html_for_document_loader(file_path, filename)
            return self._load_from_langchain_docs(raw_docs, file_path, filename, doc_type)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        try:
            raw_docs = loader.load()
            return self._load_from_langchain_docs(raw_docs, file_path, filename, doc_type)
        except Exception as e:
            raise Exception(f"Failed to process document: {str(e)}") from e

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        all_documents = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (
                file_lower.endswith(".pdf")
                or file_lower.endswith((".docx", ".doc"))
                or file_lower.endswith((".xlsx", ".xls"))
                or file_lower.endswith((".html", ".htm"))
            ):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue

        return all_documents
