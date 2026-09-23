"""Document loading and chunking service"""
import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.env import env_bool
from backend.indexing.docx_layout import parse_docx_blocks
from backend.indexing.ingest_progress import IngestProgress
from backend.indexing.html_layout import parse_html_blocks
from backend.indexing.pdf_layout import parse_pdf_blocks
from backend.indexing.xlsx_layout import parse_xlsx_blocks
from backend.profiles import get_profile

logger = logging.getLogger(__name__)

# Chunking defaults come from the active domain profile; the CHUNK_* environment
# variables below still override them, and an explicit DocumentLoader(...) argument
# overrides both. Effective order: constructor arg > env > profile > schema default.
_CHUNKING = get_profile().chunking


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
LAYOUT_PARSER_ENABLED = env_bool("LAYOUT_PARSER_ENABLED", _CHUNKING.layout_parser_enabled)
PDF_LAYOUT_PARSER_ENABLED = env_bool("PDF_LAYOUT_PARSER_ENABLED", LAYOUT_PARSER_ENABLED)

# Leaf chunks merge small neighbors up to ~level_3_size / divisor (a topic-sized
# retrieval granule), while level_3_size stays only a hard cap — mirrors
# MERGE_TARGET_DIVISOR in the DataProcessing reference (chunk_merger.py).
MERGE_TARGET_DIVISOR = _CHUNKING.merge_target_divisor

# Chunking configuration (tune without code changes). CHUNK_SIZE / CHUNK_OVERLAP are
# the leaf (L3) budgets in CHARACTERS for every strategy — the token strategy
# converts internally (~4 chars/token) so there is exactly one unit across the
# config, avoiding the chars-vs-tokens ambiguity the DataProcessing reference had.
# L1/L2 derive from the leaf size unless CHUNK_L1_SIZE / CHUNK_L2_SIZE override them.
CHUNK_STRATEGIES = ("recursive", "sentence", "token")
_TOKEN_CHARS_ESTIMATE = 4


def cut_to_budget(text: str, budget: int) -> List[str]:
    """One string as one or more pieces, none of them longer than `budget`.

    The last resort in every splitter here, and deliberately the only one. A limit with
    an exception is not a limit: a figure cap that cut at line boundaries let a first
    line of any length through whole, a table row longer than its budget shipped whole,
    and a sentence longer than `max_chars` did too — each of them a path by which one
    chunk could reach the 60 KB storage cap, and a handful of those is a prompt whose
    size the corpus decides rather than the retrieval.

    Cutting inside a line loses a phrase across the boundary, which is a real cost and a
    far smaller one than the alternative, and it is paid only by input that is already
    pathological.
    """
    if len(text) <= budget:
        return [text]
    return [text[index:index + budget] for index in range(0, len(text), budget)]


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
        # A sentence longer than the whole budget is cut, not passed through: without
        # this the strategy has no upper bound at all, and the legacy loader it is
        # reachable from applies no byte cap either. Measured before this line, one
        # 5,001-character sentence produced one 5,001-character chunk at max_chars 800.
        sentences = [
            piece
            for raw in self._SENTENCE_RE.split((text or "").strip())
            if raw.strip()
            for piece in cut_to_budget(raw.strip(), self.max_chars)
        ]
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
        chunk_size = chunk_size or _read_positive_int_env("CHUNK_SIZE", _CHUNKING.chunk_size)
        chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else _read_positive_int_env("CHUNK_OVERLAP", _CHUNKING.chunk_overlap)
        )
        self._strategy = self._resolve_strategy()

        # A null l1_size/l2_size in the profile keeps the original derivation from the
        # leaf size, so tuning CHUNK_SIZE alone still scales the whole hierarchy.
        level_1_size = _read_positive_int_env(
            "CHUNK_L1_SIZE", _CHUNKING.l1_size or max(2000, chunk_size * 3)
        )
        level_1_overlap = max(400, chunk_overlap * 3)
        level_2_size = _read_positive_int_env(
            "CHUNK_L2_SIZE", _CHUNKING.l2_size or max(1000, chunk_size * 2)
        )
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
        strategy = (os.getenv("CHUNK_STRATEGY") or _CHUNKING.strategy).strip().lower()
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
                overlap_sentences=_read_positive_int_env(
                    "CHUNK_SENTENCE_OVERLAP", _CHUNKING.sentence_overlap
                ),
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
        # Cells are coerced rather than assumed. Every parser routes its table through
        # `normalize_table_rows`, so a None or a number does not reach here today — but
        # this is the one place a stray cell would raise, and raising here fails the
        # whole document's ingest, not the row. `_header_key` beside it already coerces;
        # the two disagreeing about their own contract is the part worth fixing.
        return "\n".join(" | ".join(str(cell or "") for cell in row) for row in rows if row)

    def _fit_row(self, row: List[str], budget: int) -> List[List[str]]:
        """One row as one or more rows, none of them rendering longer than `budget`.

        A row larger than the budget used to ship whole, on the reasoning that cutting
        mid-row is worse than a large chunk. It is not, and the asymmetry is the whole
        lesson of the incident this work exists to fix: an over-long row made "budget"
        advisory, so one cell holding a paragraph — a policy note written inside a grid,
        a calendar row listing every event of a week — produced a leaf bounded by
        nothing but `_MILVUS_TEXT_CAP_BYTES`, sixty thousand bytes. A handful of those
        is a grading prompt whose size the CORPUS decides, which is exactly what commit
        1794762 stopped happening by a different means.

        Cut into continuation rows instead. That one row loses its columns, which is a
        real loss and a smaller one than the row being unretrievable at a usable size —
        and every other row in the table keeps its grid.
        """
        rendered = " | ".join(row)
        if len(rendered) <= budget:
            return [row]
        return [[piece] for piece in self._cut_line_to_budget(rendered, budget)]

    def _split_table_row_groups(self, rows: List[List[str]], budget: int) -> List[List[List[str]]]:
        """Group table rows into budget-sized groups with the header row repeated per
        group, so every group stays self-describing — and so that no group exceeds the
        budget, which is a property the rest of the pipeline now depends on."""
        rows = [row for row in rows if row]
        if not rows:
            return []
        if len(self._render_rows(rows)) <= budget:
            return [rows]

        header = rows[0]
        header_len = len(" | ".join(header))
        if header_len + 1 >= budget:
            # The header alone fills the budget, so there is no room to repeat it and
            # therefore no grid left to preserve. Bounded lines are what remains, and
            # bounded is the property that has to hold.
            return [
                [[piece]] for piece in self._line_passages(self._render_rows(rows), budget)
            ]

        # Every body row made to fit BESIDE a repeated header, so a group is the header
        # plus at least one row and still within budget.
        rows = [header] + [
            fitted
            for row in rows[1:]
            for fitted in self._fit_row(row, budget - header_len - 1)
        ]

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
        asset_ids: tuple = (),
        kind: str = "text",
        list_group: int = 0,
    ) -> List[Dict]:
        """`kind="figure"` marks a unit built from an image's text surrogate.

        Figures reach here once, from `_blocks_to_units`, and are never re-split
        afterwards — `_refine_units` hands them through untouched. The asset reference is
        still copied onto every piece, because the level-1 pass can divide a surrogate
        longer than a whole section window and a figure that HAS been divided must still
        point at the image it describes.
        """
        units: List[Dict] = []
        for piece in splitter.split_text(text):
            piece = (piece or "").strip()
            if piece:
                units.append({
                    "kind": kind,
                    "text": piece,
                    "sections": sections,
                    "page": page,
                    "asset_ids": tuple(asset_ids),
                    "list_group": list_group,
                })
        return units

    #: When a transcription is a grid rather than prose. Three is the fewest lines that
    #: can be a header plus two rows, which is also the fewest worth repeating a header
    #: for. Shorter runs of pipes are packed as ordinary lines.
    _FIGURE_TABLE_MIN_ROWS = 3

    #: A markdown table's rule row (`---`, `:--`, `--:`). Formatting, not evidence, and
    #: `_render_rows` re-renders the grid in this loader's own row form anyway.
    _TABLE_RULE_CELL = re.compile(r"^:?-{3,}:?$")

    #: One line, bounded. The hole the review found in the figure cap this replaces: it
    #: cut at a LINE boundary and so let a first line of ANY length through whole — a
    #: synthetic 10,000-character line passed unchanged, which makes the bound neither
    #: lossless nor actually a bound. A transcription rendered as one enormous row is
    #: exactly that shape, and it is how one image became one 60 KB chunk. Everything
    #: downstream — the leaf budget, the merge window, the size of a prompt — is derived
    #: from this holding for every line.
    _cut_line_to_budget = staticmethod(cut_to_budget)

    def _line_passages(self, text: str, budget: int) -> List[str]:
        """`text` as whole lines packed into passages of at most `budget` characters."""
        passages: List[str] = []
        current: List[str] = []
        used = 0
        for raw_line in (text or "").splitlines():
            for line in self._cut_line_to_budget(raw_line, budget):
                if not current and not line.strip():
                    continue
                if current and used + len(line) + 1 > budget:
                    passages.append("\n".join(current))
                    current, used = [], 0
                current.append(line)
                used += len(line) + (1 if used else 0)
        if current:
            passages.append("\n".join(current))
        return [passage for passage in passages if passage.strip()]

    @classmethod
    def _table_row(cls, line: str) -> List[str]:
        """A pipe-delimited line as cells, without the empty ends markdown leaves."""
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    @classmethod
    def _is_rule_row(cls, row: List[str]) -> bool:
        return bool(row) and all(cls._TABLE_RULE_CELL.match(cell or "") for cell in row)

    def _transcription_passages(self, transcription: str, budget: int) -> List[str]:
        """A transcription as passages, as ROW GROUPS wherever it is a table.

        The extraction prompt asks for "EVERY piece of text visible in the image ... also
        render the underlying values as a markdown table", so a transcription is often
        part prose and part grid. Cut a grid at an arbitrary line and the header goes
        with the first piece only, and a fee row without its column names answers
        nothing — which is the same reason `_split_table_row_groups` exists for real
        tables, so it is reused here rather than restated.

        Runs rather than a whole-text verdict, so a transcription that is a paragraph
        followed by a grid keeps both: nothing is dropped for not matching the shape of
        its neighbours.
        """
        passages: List[str] = []
        for is_table, lines in self._transcription_runs(transcription):
            if is_table:
                rows = [self._table_row(line) for line in lines]
                rows = [row for row in rows if row and not self._is_rule_row(row)]
                if rows:
                    passages.extend(
                        self._render_rows(group)
                        for group in self._split_table_row_groups(rows, budget)
                    )
                continue
            passages.extend(self._line_passages("\n".join(lines), budget))
        return [passage for passage in passages if passage.strip()]

    @classmethod
    def _transcription_runs(cls, transcription: str) -> List[tuple]:
        """Consecutive lines grouped into (is_table, lines) runs."""
        runs: List[tuple] = []
        for line in (transcription or "").splitlines():
            if not line.strip():
                continue
            piped = "|" in line
            if runs and runs[-1][0] == piped:
                runs[-1][1].append(line)
            else:
                runs.append((piped, [line]))
        # A short run of pipes is a sentence containing one, not a grid to regroup.
        return [
            (piped and len(lines) >= cls._FIGURE_TABLE_MIN_ROWS, lines)
            for piped, lines in runs
        ]

    def _discovery_passages(self, description: str, summary: str, budget: int) -> List[str]:
        """What the picture IS and what it can answer, with those two never separated.

        `summary` is the `Tags:` and `Answers:` lines, and `Answers:` is made of question
        phrasings — which is precisely what made the old orphan tail out-retrieve the
        figure it came from. Packing description and summary as one block is not enough
        to prevent that: a description longer than a leaf divides, and the summary then
        lands in a passage of its own carrying nothing else. So the summary is ATTACHED
        to the first piece of the description rather than flowed after it.

        The one case that cannot be fixed here is a summary longer than a whole leaf on
        its own, which leaves it nothing to be attached to. It is then packed like any
        other text — still headed, so still identified, and no worse than the figure it
        describes having no description at all.
        """
        if not (description or summary):
            return []
        reserved = len(summary) + 1
        if summary and description and reserved < budget:
            pieces = self._line_passages(description, budget - reserved)
            if pieces:
                return [f"{pieces[0]}\n{summary}", *pieces[1:]]
        joined = "\n".join(part for part in (description, summary) if part)
        return self._line_passages(joined, budget)

    def _figure_passages(self, figure: Dict, content: str, budget: int) -> List[str]:
        """One image's text as passages that each stand on their own.

        A figure used to be one indivisible unit, capped at four leaves' worth of text.
        Both halves of that were wrong in the same way: the cap threw evidence away at
        INDEXING, where nothing can get it back, and being indivisible is what made the
        cap necessary — a single unit larger than a level's budget is packed alone and
        unshortened, which is how one transcribed calendar became one 60 KB chunk.

        Divided instead, and divided so that the reason it was made indivisible cannot
        return. Commit 8b4e367 found that a naively split surrogate leaves a tail of
        `Tags:` and `Answers:` with no picture, no caption and no description, and that
        the tail out-retrieves the figure it came from, because it is made of question
        phrasings and carries no other subject to dilute the match. So:

        - the header is repeated on EVERY passage, so no piece is ever unidentified;
        - `Tags:` and `Answers:` stay attached to the description, in the passage whose
          job is already to be matched against a question. There is exactly one of them,
          and it is the whole discovery surface rather than an orphaned fragment of it;
        - every passage keeps the same `asset_ids`, so the picture is cited and shown
          once however many passages of it were retrieved.

        Nothing is dropped: the transcription is divided, not cut, and the full
        extraction is in the asset store regardless.
        """
        header = (figure.get("header") or "").strip()
        description = (figure.get("description") or "").strip()
        transcription = (figure.get("transcription") or "").strip()
        summary = (figure.get("summary") or "").strip()
        if not figure:
            # A block written without the structured parts — an older caller, or a test
            # constructing one by hand. The first line is the header by construction
            # (`render_surrogate` writes it first when no section path is given) and the
            # rest is treated as one block, which is bounded and identified even though
            # it cannot tell description from transcription.
            lines = (content or "").strip().splitlines()
            header, rest = (lines[0].strip(), "\n".join(lines[1:])) if lines else ("", "")
            description, transcription, summary = rest, "", ""

        # Room for the header, which is repeated, plus the newline joining it on.
        #
        # The header is cut first, because it is the one input to this bound that is not
        # itself bounded: a caption comes from a vision model as whatever string it
        # returned, and only the heuristic extractor caps it. A caption as long as the
        # leaf budget would otherwise leave a body budget of one character — every
        # passage over the bound, and one image exploding into hundreds of chunks that
        # all share an asset_id and compete for the final slots. Capped at a third, the
        # same shape of rule `_apply_section_prefix` already applies to a section path.
        header = header[:max(budget // 3, 1)].strip()
        body_budget = max(budget - len(header) - 1, 1)

        bodies = self._discovery_passages(description, summary, body_budget)
        bodies.extend(self._transcription_passages(transcription, body_budget))

        if not bodies:
            return [header] if header else []
        return [f"{header}\n{body}" if header else body for body in bodies]

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
            elif unit["kind"] == "figure":
                # A figure passage is refined by NOBODY, and this is still the rule that
                # commit 8b4e367 wrote — only its reason has moved upstream.
                #
                # Put through a splitter, a surrogate came apart at whatever character
                # the budget fell on, and because `render_surrogate` writes the caption
                # first and `Tags:`/`Answers:` last, the tail was a chunk with no
                # picture, no caption and no description. Those tails retrieved BETTER
                # than the figures they came from — they are made of question phrasings,
                # so a question matches them closely, and having lost the description
                # they carry no other subject to dilute the match. One uniform question
                # came back with two of five snippets being tails whose own figures never
                # reached the same result.
                #
                # `_figure_passages` has now divided the surrogate by its FIELDS, under
                # the leaf budget, repeating the header on each piece. Every passage
                # arriving here is therefore already small enough and already identified,
                # and re-splitting one could only undo both.
                refined.append(dict(unit))
            else:
                refined.extend(
                    self._text_units(
                        unit["text"],
                        splitter,
                        sections,
                        page,
                        asset_ids=unit.get("asset_ids", ()),
                        kind=unit.get("kind", "text"),
                        list_group=unit.get("list_group", 0),
                    )
                )
        return refined

    # Unit kinds that must never share a leaf chunk with anything else: a table stays
    # a pure grid, and a figure passage stays attached to exactly its own image
    # rather than diluting an unrelated paragraph's embedding.
    #
    # Note what this does NOT bound. An atomic unit is given a window of its own
    # regardless of size, so isolation was never a size limit — for as long as one image
    # produced one unit, `_FIGURE_LEAF_SIZE_MULTIPLIER` had to exist to stop a
    # transcribed calendar reaching 60 KB, and it bought that bound by destroying
    # evidence at indexing. `_figure_passages` bounds the unit instead, so the cap is
    # gone and the isolation is back to meaning only what it says.
    _ATOMIC_LEAF_KINDS = ("table", "figure")

    @staticmethod
    def _pack_units(
        units: List[Dict],
        budget: int,
        target: Optional[int] = None,
        isolate_atomic: bool = False,
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

        # How big each list is, so one can be moved WHOLE rather than broken.
        # Keeping a list together is not enough on its own: a window that has already
        # filled with preceding prose has no room left, the hard budget wins, and the
        # list breaks exactly where it would have broken anyway. Measured on the bus
        # districts, that is precisely what happened — the list stayed intact through
        # the level-1 pass and then split at level 2 with five items in one window and
        # two in the next.
        # Switchable so the two builds can be compared on the same corpus rather than
        # argued about. `1` (the default) keeps lists whole.
        cohesion = (os.getenv("CHUNK_LIST_COHESION") or "1").strip() not in ("0", "false", "no")
        list_sizes: Dict[int, int] = {}
        for item in units if cohesion else ():
            group = item.get("list_group")
            if group:
                list_sizes[group] = list_sizes.get(group, 0) + len(item["text"]) + 2

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
            if isolate_atomic and unit["kind"] in DocumentLoader._ATOMIC_LEAF_KINDS:
                close()
                windows.append([unit])
                continue
            if split_on_section_change and current and unit.get("sections", ()) != current[0].get("sections", ()):
                close()
            # A list is one thing. Word says so in the paragraph style, and the parser
            # now carries it here. Closing a window between "Maadi" and "Mokattam"
            # leaves a run of bare place names in one chunk with nothing saying what
            # they are, and the previous chunk promising a list it does not contain.
            # Measured before this rule: 22 of the corpus's 36 lists were cut this way.
            #
            # The TARGET yields to a list; the hard budget never does. A list longer
            # than a whole window still has to break somewhere, and breaking it is
            # better than an unbounded chunk — the bound is what every other rule here
            # exists to keep.
            group = unit.get("list_group") if cohesion else 0
            continues_list = bool(current and group and group == current[-1].get("list_group"))
            fits = current_len + 2 + unit_len <= budget

            # A list that is ABOUT to start and cannot fit in what is left of this
            # window begins a new one, so it arrives whole instead of straddling the
            # boundary. Only when it would actually fit somewhere: a list longer than a
            # whole window has to break, and moving it would just break it later.
            starts_list = bool(group and not continues_list)
            if (
                starts_list
                and current
                and list_sizes.get(group, 0) <= budget
                and current_len + list_sizes.get(group, 0) > budget
            ):
                close()
            elif current and (current_len >= effective_target or not fits):
                if not (continues_list and fits):
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
    def _window_modality(window: List[Dict]) -> str:
        """A window's dominant kind. "figure" wins whenever an image contributed to
        it, because that is the fact a caller filters on ("show me chunks backed by
        a picture"); a window of only tables reports "table"; everything else "text"."""
        kinds = {unit.get("kind", "text") for unit in window}
        if "figure" in kinds:
            return "figure"
        if kinds == {"table"}:
            return "table"
        return "text"

    @staticmethod
    def _window_asset_ids(window: List[Dict]) -> List[str]:
        """Union of the asset references in a window, in first-seen order, so a parent
        chunk that swallowed several figures still points at all of them."""
        seen: List[str] = []
        for unit in window:
            for asset_id in unit.get("asset_ids", ()):
                if asset_id and asset_id not in seen:
                    seen.append(asset_id)
        return seen

    @staticmethod
    def _apply_section_prefix(text: str, sections: List[str], modality: str = "text") -> str:
        """The section path a chunk carries into its embedding, and when it earns it.

        Measured over 349 questions, three ways, each a full reindex (arm A = the old
        behaviour, B = title dropped, C = no prefix at all):

            stage       A full    B specific    C none
            recalled      337         340         341
            ranked        320         319         322

        So the path as it was written cost recall rather than adding it — but the loss is
        not spread evenly. Broken out by what the chunk IS:

            text    274 -> 278   (+4 without the prefix)
            table    40 ->  38   (-2 without it)

        Both halves of that make sense, and together they are the rule below. A prose
        chunk already says what it is about, so a path repeated across 118 of 175 leaves
        from a vocabulary of only 34 distinct strings is 18% of the text saying nothing
        new — and it pulls the corpus together in embedding space, mean pairwise cosine
        0.4518 -> 0.5156, which is the same complaint `_apply_bm25_section_prefix` makes
        about the sparse lane. A TABLE has no such prose. "Grade | Fee | 105,000" carries
        no topic at all, and the path is the only thing that says which table it is.

        So the path goes on tables and figures and nowhere else, and it never carries the
        document's own title, which by definition repeats on every chunk in the document.

        Measured again on ENGLISH queries — the language retrieval actually runs in,
        since an Arabic question is translated before it reaches the index — the path
        earns nothing there either. So `none` is the default, and `full`, `specific` and
        `structured` stay reachable through `CHUNK_SECTION_PREFIX` for re-measurement.
        """
        if not text or not sections:
            return text
        mode = (os.getenv("CHUNK_SECTION_PREFIX") or "none").strip().lower()
        if mode == "none":
            return text
        if mode == "structured" and modality not in ("table", "figure"):
            return text
        # The title repeats on every chunk of the document, so it is signal in none of
        # them. Only `full` — the measured-worse arm — still carries it.
        #
        # Dropped only when something is left underneath. `sections[0]` is the document's
        # own title in THIS corpus, where one Heading 1 wraps everything; it is not a
        # title in a document whose top level is topical, and stripping it there throws
        # away the only context a table has. `test_loader_end_to_end_word_table_is_row_
        # safe_and_topic_prefixed` is that document, and it caught this.
        levels = list(sections) if mode == "full" or len(sections) < 2 else list(sections[1:])
        if not levels:
            return text
        if levels[-1] in text[:200]:
            return text
        keep = 3 if mode == "full" else 2
        prefix = " > ".join(levels[-keep:])[:150].strip()
        return f"{prefix}\n{text}" if prefix else text

    @staticmethod
    def _apply_bm25_section_prefix(text: str, sections: List[str]) -> str:
        """Section prefix for the BM25 field, with the document-root heading dropped.

        sections[0] is the document's own title, so it repeats on effectively every
        chunk. Indexing it gives every chunk the same high-frequency terms, which
        flattens BM25 scoring across the corpus — the sparse half then contributes
        near-uniform noise to RRF and outvotes strong dense matches. The specific
        levels ("Services > One Trace") are genuine per-chunk signal and stay: they
        are what makes a keyword query for a section name match its body chunks.
        """
        if not text or not sections:
            return text
        specific = list(sections[1:]) if len(sections) > 1 else []
        if not specific or specific[-1] in text[:200]:
            return text
        prefix = " > ".join(specific[-2:])[:150].strip()
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
        # A figure's surrogate is not a paragraph, and it arrives here as a text block
        # only so that stitching, section tagging and the hierarchy never had to learn
        # about images. Stitching is the one stage that WRITES `content`, and a figure
        # block is now indexed from the structured `figure` parts beside it — so text
        # appended to its `content` would be indexed nowhere at all. It is also not text
        # anyone wants joined: a transcription's last row is not a sentence that broke
        # across a page.
        if prev.get("asset_ids") or block.get("asset_ids"):
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

    #: How near a page edge a table must sit for the break to look like the reason it
    #: stopped. A fraction of the page, so it does not depend on the paper size.
    #:
    #: Wide, and deliberately. A table cut off by a page break stops above whatever the
    #: page keeps below it — the bottom margin, and the footer, whose band pdf_layout
    #: already reckons at 15% of the page. At 12% an ordinary 1.5-inch margin put a
    #: genuine continuation outside the band and split it. Being too wide costs a join
    #: between two tables that really do meet at the page edges with nothing between
    #: them; being too narrow costs a split down the middle of one table, which is the
    #: failure that cannot be recovered downstream.
    _PAGE_EDGE_FRACTION = 0.22

    @staticmethod
    def _header_key(row) -> tuple:
        """A header row reduced to what it says, not how it was typeset."""
        return tuple(
            re.sub(r"\W+", "", str(cell or ""), flags=re.UNICODE).casefold() for cell in row
        )

    #: How much of the wider span the two tables must share to be one table. A
    #: continuation is laid out with the columns it is continuing, so it occupies the
    #: same span; a different grid beside it on the page usually does not. Tolerant,
    #: because a borderless grid's detected edge moves a little with its content — it
    #: takes a third of the width to disagree before this refuses anything.
    _HORIZONTAL_OVERLAP_MIN = 0.7

    @staticmethod
    def _column_count(rows) -> int:
        """How many columns a grid has: the width most of its rows share.

        Not the first row's width. A header with a merged cell is narrower than the body
        beneath it, and comparing that against a continuation's data row — which has the
        body's width — refuses a join over a difference that exists only in the heading.
        Ties go to the wider count, since a row is more often short than long.
        """
        widths = [len(row) for row in rows if row]
        if not widths:
            return 0
        return max(set(widths), key=lambda width: (widths.count(width), width))

    @classmethod
    def _spans_the_same_columns(cls, prev: Dict, block: Dict) -> bool:
        """Whether the two tables occupy the same horizontal band of the page.

        Not reported means not asked: only the PDF parser measures this, and a parser
        that cannot is not held to it.
        """
        left = prev.get("_last_x0", prev.get("x0"))
        right = prev.get("_last_x1", prev.get("x1"))
        if left is None or right is None or block.get("x0") is None or block.get("x1") is None:
            return True
        left, right = float(left), float(right)
        other_left, other_right = float(block["x0"]), float(block["x1"])
        widest = max(right, other_right) - min(left, other_left)
        if widest <= 0:
            return True
        shared = min(right, other_right) - max(left, other_left)
        return shared / widest >= cls._HORIZONTAL_OVERLAP_MIN

    @staticmethod
    def _page_edges(block: Dict, trailing: bool = False) -> Optional[tuple]:
        """The top and bottom of the page a block sits on, in the block's own units.

        `page_height` is accepted as a fallback for a parser that reports only an extent,
        and read as a page running from 0 to that height.
        """
        prefix = "_last_" if trailing and "_last_bottom" in block else ""
        bottom = block.get(prefix + "page_bottom")
        top = block.get(prefix + "page_top")
        if bottom is None:
            height = block.get(prefix + "page_height") if prefix else block.get("page_height")
            if not height:
                return None
            top, bottom = 0.0, float(height)
        return float(top or 0.0), float(bottom)

    @classmethod
    def _ends_at_the_page_foot(cls, block: Dict) -> Optional[bool]:
        """Whether a block runs to the foot of the page it ENDS on.

        The page it ends on, not the one it started on: a run already stitched across two
        pages is judged on where it actually stopped, so a three-page chain asks whether
        page two ran to its foot rather than re-asking about page one. Without that a
        table ending mid-way down page two still looked cut off, and a different table at
        the head of page three was pulled into it.
        """
        edges = cls._page_edges(block, trailing=True)
        bottom = block.get("_last_bottom", block.get("bottom"))
        if edges is None or bottom is None:
            return None
        page_top, page_bottom = edges
        if page_bottom <= page_top:
            return None
        return float(bottom) >= page_bottom - (page_bottom - page_top) * cls._PAGE_EDGE_FRACTION

    @classmethod
    def _starts_at_the_page_head(cls, block: Dict) -> Optional[bool]:
        edges = cls._page_edges(block)
        top = block.get("top")
        if edges is None or top is None:
            return None
        page_top, page_bottom = edges
        if page_bottom <= page_top:
            return None
        return float(top) <= page_top + (page_bottom - page_top) * cls._PAGE_EDGE_FRACTION

    @classmethod
    def _breaks_at_a_page_edge(cls, prev: Dict, block: Dict) -> bool:
        """Whether the first table runs to the foot of its page and the second starts at
        the head of the next — which is what a page break actually does to one table.

        Unknown geometry passes. DOCX and XLSX report none at all (and DOCX never reaches
        here anyway, its pages all being numbered 0), so the unknown case is a caller that
        cannot be asked the question rather than one that answered no.
        """
        foot = cls._ends_at_the_page_foot(prev)
        head = cls._starts_at_the_page_head(block)
        return (foot is None or foot) and (head is None or head)

    @classmethod
    def _is_table_continuation(cls, prev: Dict, block: Dict) -> bool:
        """Whether two adjacent table blocks are ONE table that a page break split.

        This used to ask only whether both were tables on consecutive pages with the same
        number of columns, which is true of any two unrelated three-column tables that
        happen to land either side of a break. Joining them makes one grid out of two,
        and a row from the second then answers a question asked about the first — the
        same shape of failure as reading the wrong line of the right table, except built
        in at indexing time where nothing downstream can see it.

        What is added is LAYOUT, and only layout. Every test below asks where the ink is,
        never what the cells say:

          * the break falls at the page edges. One table split by a break runs to the
            foot of its page and resumes at the head of the next; a table that merely
            happened to be last on its page, and stopped half way down it, was not cut
            off by anything. This is the load-bearing one.
          * the two occupy the same horizontal band. A continuation is laid out with the
            columns it is continuing, so it spans what they span.
          * they have the same number of columns — counted as the width most rows share,
            not the first row's, so a merged heading cell does not refuse a join over a
            difference that exists only in the heading.

        Each is measured from what the parser reports and skipped where it reports
        nothing, so a format carrying no geometry is never refused for failing a question
        it was not asked.

        The other two signals the review asked for need no test of their own. "The same
        section" is already carried by adjacency: a heading between the two tables
        becomes `prev` and fails the very first check, so a join can never cross one. And
        a matching header is used where it is safe — to drop the duplicate copy — never
        to refuse, for the reason below.

        NO CONTENT TEST MAY REFUSE A JOIN. Two were tried and both were wrong. Requiring
        a matching header refuses the ordinary continuation that simply carries on with
        its rows, which `test_table_continuation_without_repeated_header_keeps_all_rows`
        exists to keep. Reading a digit in the first row as proof it is data — and its
        absence as proof it is a new header — was measured against the shipped corpus
        afterwards: 19 of its 30 table rows contain no digit at all, the whole of both
        curriculum tables being prose, so that rule would have split every continuation
        of them. Table content varies more than any such rule survives, and the cost of
        being wrong falls at indexing time where nothing downstream can see it.
        """
        if prev.get("type") != "table" or block.get("type") != "table":
            return False
        if block.get("page_number") != prev.get("_last_page", prev.get("page_number", 0)) + 1:
            return False
        prev_rows = prev.get("rows") or []
        next_rows = block.get("rows") or []
        if not prev_rows or not next_rows:
            return False
        if cls._column_count(prev_rows) != cls._column_count(next_rows):
            return False
        if not cls._spans_the_same_columns(prev, block):
            return False
        return cls._breaks_at_a_page_edge(prev, block)

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
                # Normalised, so a header the break re-typeset — a wrapped cell, a
                # different space — is still recognised as the copy it is.
                if next_rows and self._header_key(next_rows[0]) == self._header_key(
                    (prev.get("rows") or [[]])[0]
                ):
                    next_rows = next_rows[1:]
                prev["rows"] = list(prev.get("rows") or []) + next_rows
                prev["content"] = self._render_rows(prev["rows"])
                prev["_last_page"] = block.get("page_number", 0)
                # Where the run now ENDS, alongside the page it ends on. A chain is
                # judged page by page: without this the next candidate was still being
                # measured against the first page's geometry, so a table that finished
                # half way down page two still read as cut off and pulled in whatever
                # started page three.
                prev["_last_bottom"] = block.get("bottom")
                prev["_last_page_top"] = block.get("page_top")
                prev["_last_page_bottom"] = block.get("page_bottom")
                prev["_last_page_height"] = block.get("page_height")
                prev["_last_x0"] = block.get("x0")
                prev["_last_x1"] = block.get("x1")
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
                        # Carried so a heading is never separated from the list it heads.
                        "list_group": block.get("list_group", 0),
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
                    # A block carrying asset references came from the enrichment stage
                    # (an image's text surrogate); it packs as an atomic leaf.
                    asset_ids = tuple(block.get("asset_ids") or ())
                    if asset_ids:
                        # Divided HERE, at the leaf budget, and then carried through
                        # every level untouched. One image is several chunks and one
                        # asset: each passage names the picture and points at it, so a
                        # citation still resolves to one image and the answer still
                        # shows it once.
                        #
                        # This is the step that bounds everything downstream. `_pack_units`
                        # closes a window when the NEXT unit would overflow it, but a
                        # single unit larger than the budget is packed alone and whole —
                        # so for as long as one image was one unit, one image could be one
                        # chunk of any size, and the size of the grading prompt was a
                        # property of the corpus. Every unit under the leaf budget means
                        # every chunk under its level's budget.
                        for passage in self._figure_passages(
                            block.get("figure") or {}, content, self._level_3_size
                        ):
                            units.append({
                                "kind": "figure",
                                "text": passage,
                                "sections": sections,
                                "page": page_number,
                                "asset_ids": asset_ids,
                            })
                    else:
                        units.extend(
                            self._text_units(
                                content,
                                self._splitter_level_1,
                                sections,
                                page_number,
                                list_group=block.get("list_group", 0),
                            )
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
            level_1_body = sanitize_text(self._window_text(window_1)).strip()
            level_1_sections = self._window_sections(window_1)
            level_1_text = self._apply_section_prefix(
                    level_1_body, level_1_sections, self._window_modality(window_1)
                )
            level_1_bm25 = self._apply_bm25_section_prefix(level_1_body, level_1_sections)
            if not level_1_text:
                continue
            level_1_page = window_page(window_1)
            level_1_id = self._build_chunk_id(filename, level_1_page, 1, counters["l1"])
            counters["l1"] += 1
            chunks.append({
                **doc_info,
                "page_number": level_1_page,
                "text": level_1_text,
                "bm25_text": level_1_bm25,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": chunk_idx,
                "asset_ids": self._window_asset_ids(window_1),
                "modality": self._window_modality(window_1),
            })
            chunk_idx += 1

            units_2 = self._refine_units(window_1, self._splitter_level_2, self._level_2_size)
            for window_2 in self._pack_units(units_2, self._level_2_size):
                level_2_body = sanitize_text(self._window_text(window_2)).strip()
                level_2_sections = self._window_sections(window_2)
                level_2_text = self._apply_section_prefix(
                    level_2_body, level_2_sections, self._window_modality(window_2)
                )
                level_2_bm25 = self._apply_bm25_section_prefix(level_2_body, level_2_sections)
                if not level_2_text:
                    continue
                level_2_page = window_page(window_2)
                level_2_id = self._build_chunk_id(filename, level_2_page, 2, counters["l2"])
                counters["l2"] += 1
                chunks.append({
                    **doc_info,
                    "page_number": level_2_page,
                    "text": level_2_text,
                    "bm25_text": level_2_bm25,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": chunk_idx,
                    "asset_ids": self._window_asset_ids(window_2),
                    "modality": self._window_modality(window_2),
                })
                chunk_idx += 1

                units_3 = self._refine_units(window_2, self._splitter_level_3, self._level_3_size)
                for window_3 in self._pack_units(
                    units_3,
                    self._level_3_size,
                    target=leaf_merge_target,
                    isolate_atomic=True,
                    split_on_section_change=True,
                ):
                    level_3_body = sanitize_text(self._window_text(window_3)).strip()
                    level_3_sections = self._window_sections(window_3)
                    level_3_text = self._apply_section_prefix(
                    level_3_body, level_3_sections, self._window_modality(window_3)
                )
                    level_3_text = fit_utf8_bytes(level_3_text, self._MILVUS_TEXT_CAP_BYTES).strip()
                    level_3_bm25 = fit_utf8_bytes(
                        self._apply_bm25_section_prefix(level_3_body, level_3_sections),
                        self._MILVUS_TEXT_CAP_BYTES,
                    ).strip()
                    if not level_3_text:
                        continue
                    level_3_page = window_page(window_3)
                    level_3_id = self._build_chunk_id(filename, level_3_page, 3, counters["l3"])
                    counters["l3"] += 1
                    chunks.append({
                        **doc_info,
                        "page_number": level_3_page,
                        "text": level_3_text,
                        "bm25_text": level_3_bm25,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": chunk_idx,
                        "asset_ids": self._window_asset_ids(window_3),
                        "modality": self._window_modality(window_3),
                    })
                    chunk_idx += 1

        return chunks

    def _load_blocks_with_layout(
        self,
        blocks: List[Dict],
        file_path: str,
        filename: str,
        doc_type: str,
        progress: Optional[IngestProgress] = None,
    ) -> list[dict]:
        """Shared layout pipeline (pipes-and-filters) behind every format parser:

            <format>_blocks parser → typed heading/text/table/image blocks (I/O boundary)
            enrich_image_blocks   → images become text surrogates (or are dropped)
            _stitch_cross_page_blocks → rejoin paragraphs/tables split by page breaks
            _blocks_to_units      → section-tagged, page-tagged units
            _hierarchy_chunks     → L1/L2/L3 chunks over the whole document

        Each stage is a pure transformation of its input, so supporting a new file
        format only means writing a block parser — the rest of the pipeline (and
        every future stage added to it) is shared.

        Enrichment runs FIRST so that everything downstream sees only text and table
        blocks. That is what let image support land without touching stitching,
        section tagging, or the hierarchy builder.
        """
        blocks = self._enrich_assets(blocks, filename, file_path, progress)
        blocks = self._stitch_cross_page_blocks(blocks)
        units = self._blocks_to_units(blocks)
        doc_info = {
            "filename": sanitize_text(filename),
            "file_path": sanitize_text(file_path),
            "file_type": doc_type,
        }
        return self._hierarchy_chunks(units, doc_info)

    @staticmethod
    def _enrich_assets(
        blocks: List[Dict], filename: str, file_path: str,
        progress: Optional[IngestProgress] = None,
    ) -> List[Dict]:
        """Turn image blocks into retrievable text, or drop them.

        Disabled by profile (or absent of any images) this is a no-op, and a failure
        inside it never costs the document its text — see asset_enrichment.
        """
        profile = get_profile()
        if not (profile.assets.enabled and profile.assets.figures.enabled):
            return [block for block in blocks if block.get("type") != "image"]

        from backend.indexing.asset_enrichment import enrich_image_blocks

        enriched, report = enrich_image_blocks(
            blocks, filename=filename, file_path=file_path, progress=progress
        )
        if report.total:
            logger.info("Figure enrichment for %s: %s", filename, report.summary())
        return enriched

    def _try_layout_path(
        self, parse_blocks, file_path: str, filename: str, doc_type: str,
        progress: Optional[IngestProgress] = None,
    ):
        """Run a format's block parser through the shared pipeline; None means the
        caller should fall back to its legacy flat loader (failure or empty result)."""
        try:
            documents = self._load_blocks_with_layout(
                parse_blocks(file_path), file_path, filename, doc_type, progress
            )
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

    def load_document(
        self, file_path: str, filename: str, progress: Optional[IngestProgress] = None
    ) -> list[dict]:
        self._validate_identifiers(file_path, filename)
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            if PDF_LAYOUT_PARSER_ENABLED:
                documents = self._try_layout_path(parse_pdf_blocks, file_path, filename, doc_type, progress)
                if documents:
                    return documents
            loader = PyPDFLoader(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            # python-docx reads only .docx; legacy .doc always uses the flat loader.
            if LAYOUT_PARSER_ENABLED and file_lower.endswith(".docx"):
                documents = self._try_layout_path(parse_docx_blocks, file_path, filename, doc_type, progress)
                if documents:
                    return documents
            loader = Docx2txtLoader(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            # openpyxl reads only .xlsx; legacy .xls always uses the flat loader.
            if LAYOUT_PARSER_ENABLED and file_lower.endswith(".xlsx"):
                documents = self._try_layout_path(parse_xlsx_blocks, file_path, filename, doc_type, progress)
                if documents:
                    return documents
            loader = UnstructuredExcelLoader(file_path)
        elif file_lower.endswith((".html", ".htm")):
            doc_type = "HTML"
            if LAYOUT_PARSER_ENABLED:
                documents = self._try_layout_path(parse_html_blocks, file_path, filename, doc_type, progress)
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
