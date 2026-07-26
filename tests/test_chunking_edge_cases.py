"""Systematic edge-case suite for the chunking pipeline (ISTQB test-design pass).

Techniques applied per class:
- Boundary Value Analysis (BVA): budgets, thresholds, word/char limits at N-1/N/N+1.
- Equivalence Partitioning (EP): valid/invalid partitions for config and heuristics.
- State Transition Testing: the section-stack heading automaton.
- Decision Table Testing: cross-page stitching condition combinations.
- Error Guessing / Robustness (ISO 25010 reliability): empty inputs, Unicode
  (zero-width, Arabic RTL, CJK), oversized inputs, data-loss corners the
  DataProcessing reference never covered.
"""
import os
import unittest
from unittest.mock import patch

import backend.indexing.document_loader as document_loader_module
from backend.indexing.document_loader import DocumentLoader, SentenceSplitter
from backend.indexing.pdf_layout import (
    body_font_size,
    is_heading_line,
    looks_like_real_table,
    merge_lines_to_paragraphs,
    remove_page_furniture,
)
from tests.test_milvus_writer import FakeMilvusStore, load_milvus_writer_module


def _line(text, top=10.0, size=10.0, bold=False):
    return {"text": text, "top": top, "bottom": top + size, "x0": 0.0, "x1": 200.0,
            "size": size, "bold": bold}


def _text_block(content, page, top):
    return {"type": "text", "content": content, "page_number": page, "top": top}


def _table_block(rows, page, top):
    return {"type": "table", "content": "", "rows": rows, "page_number": page, "top": top}


class TableSplitBoundaryTests(unittest.TestCase):
    """BVA on _split_table_row_groups budgets — including the data-loss corner the
    DataProcessing reference silently shared (its merger also required 2+ rows)."""

    def setUp(self):
        self.loader = DocumentLoader()

    def test_render_exactly_at_budget_stays_one_group(self):
        rows = [["ab", "cd"], ["ef", "gh"]]
        budget = len(self.loader._render_rows(rows))
        self.assertEqual([rows], self.loader._split_table_row_groups(rows, budget))

    def test_render_one_char_over_budget_splits(self):
        rows = [["ab", "cd"], ["ef", "gh"], ["ij", "kl"]]
        budget = len(self.loader._render_rows(rows)) - 1
        groups = self.loader._split_table_row_groups(rows, budget)
        self.assertGreater(len(groups), 1)

    def test_oversized_header_only_table_is_not_lost(self):
        # 1-row table whose render exceeds the budget: must still produce a group,
        # never silently drop the data.
        rows = [["A" * 50, "B" * 50]]
        groups = self.loader._split_table_row_groups(rows, budget=60)
        self.assertEqual([rows], groups)

    def test_single_body_row_larger_than_budget_ships_with_header(self):
        rows = [["Grade", "Fee"], ["X" * 100, "Y" * 100]]
        groups = self.loader._split_table_row_groups(rows, budget=50)
        self.assertEqual(1, len(groups))
        self.assertEqual(rows, groups[0])

    def test_empty_and_all_empty_rows_yield_no_groups(self):
        self.assertEqual([], self.loader._split_table_row_groups([], budget=100))


class PackUnitsBoundaryTests(unittest.TestCase):
    """BVA on _pack_units target/budget closing conditions."""

    @staticmethod
    def _unit(length, kind="text", sections=()):
        return {"kind": kind, "text": "x" * length, "sections": sections, "page": 0}

    def test_window_closes_exactly_at_target(self):
        units = [self._unit(50), self._unit(10)]
        windows = DocumentLoader._pack_units(units, budget=100, target=50)
        self.assertEqual(2, len(windows))

    def test_window_stays_open_one_below_target(self):
        units = [self._unit(49), self._unit(10)]
        windows = DocumentLoader._pack_units(units, budget=100, target=50)
        self.assertEqual(1, len(windows))

    def test_budget_hard_cap_beats_target(self):
        units = [self._unit(30), self._unit(80)]
        windows = DocumentLoader._pack_units(units, budget=100, target=200)
        self.assertEqual(2, len(windows))

    def test_empty_units_yield_no_windows(self):
        self.assertEqual([], DocumentLoader._pack_units([], budget=100))

    def test_isolated_table_between_texts_produces_three_windows(self):
        units = [self._unit(10), self._unit(10, kind="table"), self._unit(10)]
        windows = DocumentLoader._pack_units(units, budget=100, isolate_tables=True)
        self.assertEqual(3, len(windows))
        self.assertEqual("table", windows[1][0]["kind"])


class SentenceSplitterBoundaryTests(unittest.TestCase):
    """BVA/EP for SentenceSplitter, incl. the overlap partitions where the
    DataProcessing reference misbehaved (overlap larger than a chunk)."""

    def test_empty_and_whitespace_input(self):
        splitter = SentenceSplitter(max_chars=50)
        self.assertEqual([], splitter.split_text(""))
        self.assertEqual([], splitter.split_text("   \n  "))

    def test_sentence_exactly_at_budget_is_one_chunk(self):
        sentence = "A" * 24 + "."
        splitter = SentenceSplitter(max_chars=25, overlap_sentences=0)
        self.assertEqual([sentence], splitter.split_text(sentence))

    def test_overlap_larger_than_chunk_terminates_and_keeps_all_content(self):
        splitter = SentenceSplitter(max_chars=40, overlap_sentences=10)
        text = "Alpha one done. Beta two done. Gamma three done. Delta four done."
        chunks = splitter.split_text(text)
        joined = " ".join(chunks)
        for word in ("Alpha", "Beta", "Gamma", "Delta"):
            self.assertIn(word, joined)
        self.assertLess(len(chunks), 20)

    def test_cjk_sentence_endings_split(self):
        splitter = SentenceSplitter(max_chars=12, overlap_sentences=0)
        chunks = splitter.split_text("学费包括教材。 校车另外收费。")
        self.assertEqual(["学费包括教材。", "校车另外收费。"], chunks)


class RealTableHeuristicBoundaryTests(unittest.TestCase):
    """BVA on looks_like_real_table's two thresholds."""

    def test_multi_cell_fraction_exactly_at_threshold_passes(self):
        rows = [["a", "b"]] * 4 + [["only", ""]]  # 4/5 = 0.8
        self.assertTrue(looks_like_real_table(rows))

    def test_multi_cell_fraction_below_threshold_fails(self):
        rows = [["a", "b"]] * 3 + [["x", ""], ["y", ""]]  # 3/5 = 0.6
        self.assertFalse(looks_like_real_table(rows))

    def test_avg_cell_len_exactly_twenty_passes_twenty_one_fails(self):
        self.assertTrue(looks_like_real_table([["c" * 20, "d" * 20]] * 2))
        self.assertFalse(looks_like_real_table([["c" * 21, "d" * 21]] * 2))


class HeadingBoundaryTests(unittest.TestCase):
    """BVA/EP for heading classification limits."""

    def test_word_count_boundary_ten_passes_eleven_fails(self):
        ten = _line("One Two Three Four Five Six Seven Eight Nine Ten", bold=True)
        eleven = _line("One Two Three Four Five Six Seven Eight Nine Ten Eleven", bold=True)
        self.assertTrue(is_heading_line(ten, body_size=10.0))
        self.assertFalse(is_heading_line(eleven, body_size=10.0))

    def test_size_ratio_boundary(self):
        at_ratio = _line("Mixed Case Line Here", size=11.5)
        below_ratio = _line("Mixed Case Line Here", size=11.49)
        self.assertTrue(is_heading_line(at_ratio, body_size=10.0))
        self.assertFalse(is_heading_line(below_ratio, body_size=10.0))

    def test_all_caps_needs_at_least_four_letters(self):
        self.assertTrue(is_heading_line(_line("FEES INFO"), body_size=10.0))
        self.assertFalse(is_heading_line(_line("FEE"), body_size=10.0))

    def test_missing_font_sizes_disable_size_signal_but_not_bold(self):
        pages = [{"page_number": 0, "tables": [], "lines": [
            {"text": "Some body line with words", "top": 10.0, "bottom": 20.0,
             "x0": 0.0, "x1": 100.0, "size": None, "bold": False},
        ]}]
        self.assertIsNone(body_font_size(pages))
        bold_line = {"text": "Fees", "top": 5.0, "bottom": 15.0, "x0": 0.0,
                     "x1": 100.0, "size": None, "bold": True}
        self.assertTrue(is_heading_line(bold_line, body_size=None))


class ParagraphMergeBoundaryTests(unittest.TestCase):
    def test_gap_exactly_at_threshold_merges(self):
        lines = [{"text": "a line", "top": 10.0}, {"text": "b line", "top": 25.0}]
        self.assertEqual(1, len(merge_lines_to_paragraphs(lines, gap_threshold=15.0)))

    def test_gap_one_over_threshold_splits(self):
        lines = [{"text": "a line", "top": 10.0}, {"text": "b line", "top": 25.1}]
        self.assertEqual(2, len(merge_lines_to_paragraphs(lines, gap_threshold=15.0)))


class FurnitureBoundaryTests(unittest.TestCase):
    """BVA on remove_page_furniture page-count thresholds."""

    @staticmethod
    def _page(page_number, lines):
        return {"page_number": page_number, "height": 800.0, "tables": [], "lines": lines}

    def _pages_with_footer_on(self, footer_pages, total=5):
        pages = []
        for i in range(total):
            lines = [_line(f"Body paragraph {i} with several distinct words.", top=400.0)]
            if i in footer_pages:
                lines.append(_line(f"Page {i + 1} of {total}", top=780.0))
            pages.append(self._page(i, lines))
        return pages

    def test_footer_on_exactly_sixty_percent_of_pages_is_removed(self):
        cleaned = remove_page_furniture(self._pages_with_footer_on({0, 2, 4}))
        for page in cleaned:
            for line in page["lines"]:
                self.assertNotIn("Page", line["text"])

    def test_footer_below_threshold_is_kept(self):
        cleaned = remove_page_furniture(self._pages_with_footer_on({0, 4}))
        kept = [l["text"] for p in cleaned for l in p["lines"]]
        self.assertTrue(any("Page" in text for text in kept))

    def test_exactly_three_pages_is_the_minimum_for_removal(self):
        three = remove_page_furniture(self._pages_with_footer_on({0, 1, 2}, total=3))
        for page in three:
            self.assertEqual(1, len(page["lines"]))


class SectionStackStateTransitionTests(unittest.TestCase):
    """State-transition testing for the heading automaton."""

    @staticmethod
    def _titles(stack):
        return [entry["title"] for entry in stack]

    def test_known_level_nesting_and_pop(self):
        stack = []
        for title, level in (("A", 1), ("B", 2), ("C", 3)):
            DocumentLoader._update_section_stack(stack, title, level)
        self.assertEqual(["A", "B", "C"], self._titles(stack))
        DocumentLoader._update_section_stack(stack, "D", 2)
        self.assertEqual(["A", "D"], self._titles(stack))

    def test_unknown_level_appends_then_replaces_its_sibling(self):
        stack = []
        DocumentLoader._update_section_stack(stack, "A", 1)
        DocumentLoader._update_section_stack(stack, "E", None)
        self.assertEqual(["A", "E"], self._titles(stack))
        DocumentLoader._update_section_stack(stack, "F", None)
        self.assertEqual(["A", "F"], self._titles(stack))

    def test_known_level_pops_through_unknowns(self):
        stack = []
        DocumentLoader._update_section_stack(stack, "A", 1)
        DocumentLoader._update_section_stack(stack, "B", 2)
        DocumentLoader._update_section_stack(stack, "F", None)
        DocumentLoader._update_section_stack(stack, "G", 1)
        self.assertEqual(["G"], self._titles(stack))

    def test_stack_depth_is_capped(self):
        stack = []
        for level, title in enumerate(("A", "B", "C", "D", "E"), start=1):
            DocumentLoader._update_section_stack(stack, title, level)
        self.assertLessEqual(len(stack), 4)
        self.assertEqual("E", stack[-1]["title"])


class StitchingDecisionTableTests(unittest.TestCase):
    """Decision-table testing: type × page-adjacency × punctuation × continuation."""

    def setUp(self):
        self.loader = DocumentLoader()

    def test_same_page_adjacent_texts_never_stitch(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("An unfinished thought about", 0, 100.0),
            _text_block("something on the same page.", 0, 300.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_page_gap_of_two_never_stitches(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("An unfinished thought about", 0, 700.0),
            _text_block("something two pages later.", 2, 30.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_cjk_continuation_stitches(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("本学期的学费包括", 0, 700.0),
            _text_block("所有教材费用。", 1, 30.0),
        ])
        self.assertEqual(1, len(blocks))
        self.assertIn("学费包括 所有教材", blocks[0]["content"])

    def test_uppercase_start_blocks_stitching(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("The fee schedule includes", 0, 700.0),
            _text_block("Transportation is billed separately.", 1, 30.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_table_continuation_without_repeated_header_keeps_all_rows(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _table_block([["Grade", "Fee"], ["1", "100"]], 0, 700.0),
            _table_block([["2", "200"], ["3", "300"]], 1, 30.0),
        ])
        self.assertEqual(1, len(blocks))
        self.assertEqual(4, len(blocks[0]["rows"]))


class RobustnessTests(unittest.TestCase):
    """Error guessing / ISO 25010 reliability: Unicode, RTL, oversized, empty."""

    def setUp(self):
        self.loader = DocumentLoader()

    def _load(self, blocks):
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            return self.loader.load_document("edge.pdf", "edge.pdf")

    def test_zero_width_characters_are_sanitized_out_of_chunks(self):
        blocks = [_text_block("Fees​ are due﻿ in September or October each year.", 0, 10.0)]
        docs = self._load(blocks)
        for doc in docs:
            self.assertNotIn("​", doc["text"])
            self.assertNotIn("﻿", doc["text"])
        self.assertTrue(any("Fees are due" in d["text"] for d in docs))

    def test_arabic_heading_prefixes_arabic_content(self):
        blocks = [
            {"type": "heading", "content": "الرسوم الدراسية", "level": 1, "page_number": 0, "top": 5.0},
            _text_block("تشمل الرسوم جميع الكتب المدرسية للفصل الدراسي الأول.", 0, 20.0),
        ]
        docs = self._load(blocks)
        leaves = [d for d in docs if d["chunk_level"] == 3]
        self.assertTrue(leaves)
        self.assertTrue(leaves[0]["text"].startswith("الرسوم الدراسية"))

    def test_giant_single_table_row_respects_milvus_cap(self):
        blocks = [_table_block([["K" * 70000, "V"]], 0, 10.0)]
        docs = self._load(blocks)
        leaves = [d for d in docs if d["chunk_level"] == 3]
        self.assertTrue(leaves)
        for leaf in leaves:
            self.assertLessEqual(len(leaf["text"]), 60000)

    def test_empty_block_stream_falls_back_to_flat_loader(self):
        class _StubPage:
            page_content = "flat fallback text"
            metadata = {"page": 0}

        class _StubPyPDFLoader:
            def __init__(self, file_path):
                pass

            def load(self):
                return [_StubPage()]

        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=[]), patch.object(
            document_loader_module, "PyPDFLoader", _StubPyPDFLoader
        ):
            docs = self.loader.load_document("empty.pdf", "empty.pdf")
        self.assertTrue(any("flat fallback" in d["text"] for d in docs))

    def test_heading_only_document_still_produces_chunks(self):
        blocks = [{"type": "heading", "content": "Lonely Heading", "level": 1,
                   "page_number": 0, "top": 5.0}]
        docs = self._load(blocks)
        self.assertTrue(docs)
        self.assertTrue(all("Lonely Heading" in d["text"] for d in docs))


class ConfigNegativeTests(unittest.TestCase):
    """EP invalid partitions for env configuration (must never crash ingestion)."""

    def test_zero_and_negative_chunk_size_clamp_to_positive(self):
        with patch.dict(os.environ, {"CHUNK_SIZE": "0"}):
            self.assertGreaterEqual(DocumentLoader()._level_3_size, 1)
        with patch.dict(os.environ, {"CHUNK_SIZE": "-50"}):
            self.assertGreaterEqual(DocumentLoader()._level_3_size, 1)

    def test_garbage_chunk_size_uses_default(self):
        with patch.dict(os.environ, {"CHUNK_SIZE": "eight hundred"}):
            self.assertEqual(800, DocumentLoader()._level_3_size)

    def test_strategy_is_case_insensitive(self):
        with patch.dict(os.environ, {"CHUNK_STRATEGY": "SENTENCE"}):
            self.assertEqual("sentence", DocumentLoader()._strategy)


class WriterDedupBoundaryTests(unittest.TestCase):
    """BVA on the semantic-dedup cosine threshold + robustness of the writer."""

    def _writer(self, module, embeddings, events, env=None):
        class _Service:
            def get_embeddings(self, texts):
                return [embeddings[text] for text in texts]

        with patch.dict(os.environ, env or {}):
            return module.MilvusWriter(
                embedding_service=_Service(),
                milvus_manager=FakeMilvusStore(events),
            )

    @staticmethod
    def _doc(idx, text):
        return {"text": text, "filename": "d.pdf", "file_type": "PDF", "chunk_id": f"c{idx}"}

    def test_cosine_exactly_at_threshold_is_skipped_and_below_is_kept(self):
        module = load_milvus_writer_module()
        cases = [
            (0.97, ["c0"]),          # == threshold -> skipped
            (0.9699, ["c0", "c1"]),  # just below -> kept
        ]
        for cos, expected in cases:
            with self.subTest(cos=cos):
                events = []
                embeddings = {
                    "first text": [1.0, 0.0],
                    "second text": [cos, (1 - cos * cos) ** 0.5],
                }
                writer = self._writer(
                    module, embeddings, events,
                    env={"SEMANTIC_DEDUP_ENABLED": "true", "SEMANTIC_DEDUP_THRESHOLD": "0.97"},
                )
                writer.write_documents([self._doc(0, "first text"), self._doc(1, "second text")])
                inserts = [e for e in events if e[0] == "insert"]
                self.assertEqual([("insert", expected)], inserts)

    def test_invalid_threshold_env_falls_back(self):
        module = load_milvus_writer_module()
        with patch.dict(os.environ, {"SEMANTIC_DEDUP_THRESHOLD": "not-a-number"}):
            writer = module.MilvusWriter(embedding_service=object(), milvus_manager=object())
        self.assertEqual(0.97, writer.semantic_dedup_threshold)

    def test_empty_document_list_makes_no_calls(self):
        module = load_milvus_writer_module()
        events = []
        writer = self._writer(module, {}, events)
        writer.write_documents([])
        self.assertEqual([], events)

    def test_zero_vector_embedding_does_not_crash_semantic_dedup(self):
        module = load_milvus_writer_module()
        events = []
        embeddings = {"a text": [0.0, 0.0], "b text": [1.0, 0.0]}
        writer = self._writer(
            module, embeddings, events, env={"SEMANTIC_DEDUP_ENABLED": "true"}
        )
        writer.write_documents([self._doc(0, "a text"), self._doc(1, "b text")])
        inserts = [e for e in events if e[0] == "insert"]
        self.assertEqual([("insert", ["c0", "c1"])], inserts)


class SectionPrefixBoundaryTests(unittest.TestCase):
    """BVA for _apply_section_prefix: the 200-char heading-lookback window,
    the 150-char prefix cap, and the depth-3 path truncation."""

    def test_heading_inside_lookback_window_skips_prefix(self):
        text = "Fees\n" + "body " * 20
        result = DocumentLoader._apply_section_prefix(text, ["Fees"])
        self.assertEqual(text, result)

    def test_heading_beyond_lookback_window_gets_prefix(self):
        text = "x" * 201 + " Fees mentioned late"
        result = DocumentLoader._apply_section_prefix(text, ["Fees"])
        self.assertTrue(result.startswith("Fees\n"))

    def test_prefix_is_capped_at_150_chars(self):
        long_title = "T" * 400
        result = DocumentLoader._apply_section_prefix("body text", [long_title])
        prefix_line = result.splitlines()[0]
        self.assertLessEqual(len(prefix_line), 150)

    def test_only_last_three_sections_are_used(self):
        result = DocumentLoader._apply_section_prefix("body", ["A", "B", "C", "D"])
        prefix_line = result.splitlines()[0]
        self.assertEqual("B > C > D", prefix_line)
        self.assertNotIn("A", prefix_line)

    def test_empty_sections_or_text_are_untouched(self):
        self.assertEqual("body", DocumentLoader._apply_section_prefix("body", []))
        self.assertEqual("", DocumentLoader._apply_section_prefix("", ["Fees"]))


class HierarchyInvariantTests(unittest.TestCase):
    """Structural (property-style) testing: referential integrity of the produced
    hierarchy over a rich synthetic document — the invariants every downstream
    consumer (auto-merge, ParentChunkStore, citations) silently relies on."""

    RICH_BLOCKS = [
        {"type": "heading", "content": "Admissions", "level": 1, "page_number": 0, "top": 5.0},
        _text_block("The admissions office reviews every application in order of arrival. " * 12, 0, 20.0),
        _table_block([["Step", "Owner"], ["Form", "Parent"], ["Review", "Office"]], 0, 40.0),
        {"type": "heading", "content": "Fees", "level": 1, "page_number": 1, "top": 5.0},
        _text_block("Tuition is payable in three installments across the academic year. " * 12, 1, 20.0),
        _table_block([["Grade", "Fee"], ["1", "100"], ["2", "200"]], 1, 60.0),
        _text_block("Late payments accrue a small administrative surcharge.", 1, 90.0),
    ]

    @classmethod
    def setUpClass(cls):
        loader = DocumentLoader()
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=cls.RICH_BLOCKS):
            cls.docs = loader.load_document("rich.pdf", "rich.pdf")

    def test_only_levels_one_two_three_exist(self):
        self.assertEqual({1, 2, 3}, {d["chunk_level"] for d in self.docs})

    def test_every_parent_and_root_reference_resolves(self):
        by_id = {d["chunk_id"]: d for d in self.docs}
        for doc in self.docs:
            if doc["chunk_level"] == 1:
                self.assertEqual("", doc["parent_chunk_id"])
                self.assertEqual(doc["chunk_id"], doc["root_chunk_id"])
                continue
            parent = by_id.get(doc["parent_chunk_id"])
            root = by_id.get(doc["root_chunk_id"])
            self.assertIsNotNone(parent, f"dangling parent for {doc['chunk_id']}")
            self.assertIsNotNone(root, f"dangling root for {doc['chunk_id']}")
            self.assertEqual(doc["chunk_level"] - 1, parent["chunk_level"])
            self.assertEqual(1, root["chunk_level"])

    def test_chunk_idx_is_dense_from_zero(self):
        idxs = sorted(d["chunk_idx"] for d in self.docs)
        self.assertEqual(list(range(len(self.docs))), idxs)

    def test_leaf_body_is_contained_in_its_parent(self):
        by_id = {d["chunk_id"]: d for d in self.docs}
        for leaf in (d for d in self.docs if d["chunk_level"] == 3):
            parent = by_id[leaf["parent_chunk_id"]]
            body = leaf["text"].split("\n", 1)[-1][:80]
            self.assertIn(body[:40], parent["text"])

    def test_later_window_anchors_to_its_own_start_page(self):
        loader = DocumentLoader()
        blocks = [
            _text_block("T" * 2300, 0, 10.0),
            _text_block("Second page paragraph with enough characters to overflow the window budget entirely. " * 3, 1, 10.0),
        ]
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            docs = loader.load_document("pages.pdf", "pages.pdf")
        level_1_pages = {d["page_number"] for d in docs if d["chunk_level"] == 1}
        self.assertEqual({0, 1}, level_1_pages)
        self.assertTrue(any("::p1::l1::" in d["chunk_id"] for d in docs))


class RefineUnitsTests(unittest.TestCase):
    """EP: level refinement must preserve section/page tags and re-split tables
    against the finer budget with the header repeated."""

    def setUp(self):
        self.loader = DocumentLoader()

    def test_table_unit_resplits_at_finer_budget_with_header(self):
        rows = [["H1", "H2"]] + [[f"a{i}" * 4, f"b{i}" * 4] for i in range(20)]
        unit = {"kind": "table", "rows": rows, "text": self.loader._render_rows(rows),
                "sections": ("Fees",), "page": 2}
        refined = self.loader._refine_units([unit], self.loader._splitter_level_3, table_budget=120)
        self.assertGreater(len(refined), 1)
        for piece in refined:
            self.assertEqual("table", piece["kind"])
            self.assertEqual(["H1", "H2"], piece["rows"][0])
            self.assertEqual(("Fees",), piece["sections"])
            self.assertEqual(2, piece["page"])

    def test_text_unit_pieces_inherit_sections_and_page(self):
        unit = {"kind": "text", "text": "word " * 400, "sections": ("Fees",), "page": 3}
        refined = self.loader._refine_units([unit], self.loader._splitter_level_3, table_budget=800)
        self.assertGreater(len(refined), 1)
        for piece in refined:
            self.assertEqual(("Fees",), piece["sections"])
            self.assertEqual(3, piece["page"])


class StitchingDecisionTableCompletionTests(unittest.TestCase):
    """Remaining decision-table rows: mixed types, digits, empty rows, chains."""

    def setUp(self):
        self.loader = DocumentLoader()

    def test_text_then_table_never_stitches(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("A sentence about the fee", 0, 700.0),
            _table_block([["Grade", "Fee"], ["1", "100"]], 1, 30.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_digit_start_blocks_paragraph_stitching(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("The academic year", 0, 700.0),
            _text_block("2026 begins in September.", 1, 30.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_empty_row_tables_never_stitch(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _table_block([], 0, 700.0),
            _table_block([["a", "b"]], 1, 30.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_three_page_paragraph_chain(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("Enrollment continues with", 0, 700.0),
            _text_block("a document check and", 1, 700.0),
            _text_block("an interview to finish.", 2, 30.0),
        ])
        self.assertEqual(1, len(blocks))
        self.assertIn("continues with a document check and an interview", blocks[0]["content"])

    def test_heading_between_paragraphs_blocks_stitching(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("An unfinished sentence about", 0, 700.0),
            {"type": "heading", "content": "New Topic", "level": 1, "page_number": 1, "top": 10.0},
            _text_block("something entirely different now.", 1, 30.0),
        ])
        self.assertEqual(3, len(blocks))


class PdfBlockBuilderTests(unittest.TestCase):
    """EP + ordering on build_blocks_from_pages: fake-table demotion, in-page
    ordering, heading level ranking by font size."""

    def test_prose_table_is_demoted_and_ordering_preserved(self):
        from backend.indexing.pdf_layout import build_blocks_from_pages

        pages = [{
            "page_number": 0,
            "height": 800.0,
            "tables": [{
                "bbox": (0.0, 50.0, 200.0, 70.0),
                "rows": [
                    ["A long prose sentence pretending to be inside a table cell."],
                    ["Another long prose sentence continuing the paragraph flow."],
                ],
            }],
            "lines": [
                _line("Intro paragraph line with several plain words.", top=10.0),
                _line("Closing paragraph line with several plain words.", top=90.0),
            ],
        }]
        blocks = build_blocks_from_pages(pages)
        self.assertEqual(["text", "text", "text"], [b["type"] for b in blocks])
        self.assertEqual([10.0, 50.0, 90.0], [b["top"] for b in blocks])

    def test_two_heading_font_sizes_rank_into_levels(self):
        from backend.indexing.pdf_layout import build_blocks_from_pages

        pages = [{
            "page_number": 0,
            "height": 800.0,
            "tables": [],
            "lines": [
                _line("Main Title Here", top=10.0, size=18.0, bold=True),
                _line("Sub Section Here", top=40.0, size=14.0, bold=True),
                _line("Ordinary body sentence with plenty of everyday words in it", top=70.0),
                _line("Another ordinary body sentence with plenty of everyday words", top=85.0),
            ],
        }]
        blocks = build_blocks_from_pages(pages)
        headings = [b for b in blocks if b["type"] == "heading"]
        self.assertEqual([1, 2], [h["level"] for h in headings])


class FurnitureInteractionTests(unittest.TestCase):
    """Error guessing: repeated MID-PAGE text (e.g., a recurring section heading)
    must never be treated as furniture — only edge-band repetition qualifies."""

    def test_repeated_bold_mid_page_heading_survives_on_every_page(self):
        from backend.indexing.pdf_layout import build_blocks_from_pages

        pages = []
        for i in range(4):
            pages.append({
                "page_number": i,
                "height": 800.0,
                "tables": [],
                "lines": [
                    _line("Fees", top=400.0, size=18.0, bold=True),
                    _line(f"Page specific body content number {i} with words.", top=430.0),
                ],
            })
        blocks = build_blocks_from_pages(pages)
        headings = [b for b in blocks if b["type"] == "heading"]
        self.assertEqual(4, len(headings))


class LegacyPathRegressionTests(unittest.TestCase):
    """Compatibility: the legacy flat path (used by .doc/.xls fallbacks) must keep
    its per-page three-level hierarchy and unique IDs after all refactors."""

    class _StubDoc:
        def __init__(self, content, page):
            self.page_content = content
            self.metadata = {"page": page}

    def test_flat_path_hierarchy_integrity(self):
        loader = DocumentLoader()
        raw_docs = [
            self._StubDoc("First page paragraph content with enough words to chunk.", 0),
            self._StubDoc("Second page paragraph content with enough words to chunk.", 1),
        ]
        docs = loader._load_from_langchain_docs(raw_docs, "f.doc", "f.doc", "Word")
        ids = [d["chunk_id"] for d in docs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({1, 2, 3}, {d["chunk_level"] for d in docs})
        self.assertEqual({0, 1}, {d["page_number"] for d in docs})


class WriterDedupFlowTests(unittest.TestCase):
    """Flow semantics: cross-batch exact dedup, progress accounting, and the
    kept-list chain rule for semantic dedup."""

    @staticmethod
    def _doc(idx, text):
        return {"text": text, "filename": "d.pdf", "file_type": "PDF", "chunk_id": f"c{idx}"}

    def test_duplicate_in_later_batch_is_skipped_without_embedding(self):
        module = load_milvus_writer_module()
        events = []

        class _Service:
            def get_embeddings(self, texts):
                events.append(("embed", list(texts)))
                return [[1.0] for _ in texts]

        writer = module.MilvusWriter(embedding_service=_Service(), milvus_manager=FakeMilvusStore(events))
        progress = []
        writer.write_documents(
            [self._doc(0, "alpha"), self._doc(1, "beta"), self._doc(2, "ALPHA")],
            batch_size=2,
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        embeds = [e for e in events if e[0] == "embed"]
        self.assertEqual([("embed", ["alpha", "beta"])], embeds)
        # Progress still reports over the INPUT count so job bars complete.
        self.assertEqual([(2, 3), (3, 3)], progress)

    def test_semantic_chain_compares_against_kept_only(self):
        import os as _os

        module = load_milvus_writer_module()
        events = []
        vectors = {
            "first": [1.0, 0.0],
            "near first": [0.995, (1 - 0.995 ** 2) ** 0.5],  # dropped vs "first"
            "far away": [0.5, 3 ** 0.5 / 2],                  # cos 0.5 vs "first" -> kept
        }

        class _Service:
            def get_embeddings(self, texts):
                return [vectors[t] for t in texts]

        with patch.dict(_os.environ, {"SEMANTIC_DEDUP_ENABLED": "true"}):
            writer = module.MilvusWriter(embedding_service=_Service(), milvus_manager=FakeMilvusStore(events))
        writer.write_documents([self._doc(0, "first"), self._doc(1, "near first"), self._doc(2, "far away")])
        inserts = [e for e in events if e[0] == "insert"]
        self.assertEqual([("insert", ["c0", "c2"])], inserts)


if __name__ == "__main__":
    unittest.main()
