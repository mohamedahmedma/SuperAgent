"""A1/A2: layout-aware PDF parsing and table-aware chunking.

Covers the pure logic (table heuristic, paragraph merging, header-repeated table
splitting) and the loader integration (atomic table leaf chunks interleaved with
three-level text hierarchy, unique IDs, legacy fallback) without real PDFs.
"""
import unittest
from unittest.mock import Mock, patch

import backend.indexing.document_loader as document_loader_module
from backend.indexing.document_loader import DocumentLoader
from backend.indexing.pdf_layout import (
    build_blocks_from_pages,
    looks_like_real_table,
    merge_lines_to_paragraphs,
    normalize_table_rows,
    remove_page_furniture,
)


class TableHeuristicTests(unittest.TestCase):
    def test_short_cells_in_multi_column_rows_is_a_table(self):
        rows = [
            ["Grade", "Fee", "Currency"],
            ["1", "100", "USD"],
            ["2", "200", "USD"],
        ]
        self.assertTrue(looks_like_real_table(rows))

    def test_long_prose_cells_are_not_a_table(self):
        rows = [
            ["This is a long sentence that clearly belongs to a paragraph of prose text."],
            ["Another long sentence continuing the same paragraph with more detail here."],
        ]
        self.assertFalse(looks_like_real_table(rows))

    def test_mostly_single_cell_rows_are_not_a_table(self):
        rows = [
            ["Heading", ""],
            ["only one filled cell", ""],
            ["another lonely cell", ""],
        ]
        self.assertFalse(looks_like_real_table(rows))

    def test_single_row_is_not_a_table(self):
        self.assertFalse(looks_like_real_table([["a", "b", "c"]]))


class NormalizeRowsTests(unittest.TestCase):
    def test_none_cells_and_empty_rows_are_cleaned(self):
        rows = [
            ["  Grade ", None, "Fee\nSAR"],
            [None, None],
            ["1", "", "100"],
        ]
        self.assertEqual(
            [["Grade", "", "Fee SAR"], ["1", "", "100"]],
            normalize_table_rows(rows),
        )


class ParagraphMergeTests(unittest.TestCase):
    def test_close_lines_merge_and_far_lines_split(self):
        lines = [
            {"text": "line one", "top": 10.0},
            {"text": "line two", "top": 20.0},
            {"text": "new paragraph", "top": 80.0},
        ]
        paragraphs = merge_lines_to_paragraphs(lines, gap_threshold=15.0)
        self.assertEqual(2, len(paragraphs))
        self.assertEqual("line one\nline two", paragraphs[0]["content"])
        self.assertEqual("new paragraph", paragraphs[1]["content"])

    def test_blank_lines_are_dropped(self):
        self.assertEqual([], merge_lines_to_paragraphs([{"text": "   ", "top": 1.0}]))


class TableSplitTests(unittest.TestCase):
    def setUp(self):
        self.loader = DocumentLoader()

    def test_small_table_stays_one_group(self):
        rows = [["Grade", "Fee"], ["1", "100"], ["2", "200"]]
        groups = self.loader._split_table_row_groups(rows, budget=500)
        self.assertEqual([rows], groups)
        self.assertEqual("Grade | Fee\n1 | 100\n2 | 200", self.loader._render_rows(rows))

    def test_oversized_table_splits_with_header_repeated(self):
        header = ["Grade", "Tuition Fee", "Registration"]
        body = [[f"G{i}", "x" * 12, "y" * 12] for i in range(150)]
        budget = 600
        groups = self.loader._split_table_row_groups([header] + body, budget=budget)

        self.assertGreater(len(groups), 1)
        for group in groups:
            self.assertEqual(header, group[0])
            self.assertLessEqual(len(self.loader._render_rows(group)), budget + 40)
        body_rows = sum(len(group) - 1 for group in groups)
        self.assertEqual(len(body), body_rows)


class HeadingDetectionTests(unittest.TestCase):
    @staticmethod
    def _page(lines, tables=None, page_number=0):
        return {"page_number": page_number, "tables": tables or [], "lines": lines}

    @staticmethod
    def _line(text, top, size=10.0, bold=False):
        return {
            "text": text,
            "top": top,
            "bottom": top + size,
            "x0": 0.0,
            "x1": 200.0,
            "size": size,
            "bold": bold,
        }

    def test_large_bold_line_becomes_heading_block(self):
        lines = [
            self._line("Admission Fees", 10, size=18.0, bold=True),
            self._line("Fees are reviewed annually by the board of trustees.", 30),
            self._line("Payments are due at the start of each term this year.", 45),
        ]
        blocks = build_blocks_from_pages([self._page(lines)])
        self.assertEqual("heading", blocks[0]["type"])
        self.assertEqual("Admission Fees", blocks[0]["content"])
        self.assertEqual(1, blocks[0]["level"])
        self.assertEqual("text", blocks[1]["type"])
        self.assertIn("reviewed annually", blocks[1]["content"])

    def test_numbered_prefix_gives_explicit_level(self):
        lines = [
            self._line("2.1 Tuition Details", 10),
            self._line("Tuition covers all core academic subjects and materials.", 30),
        ]
        blocks = build_blocks_from_pages([self._page(lines)])
        self.assertEqual("heading", blocks[0]["type"])
        self.assertEqual(2, blocks[0]["level"])

    def test_sentence_punctuation_disqualifies_heading(self):
        lines = [
            self._line("This is a short sentence.", 10, size=18.0, bold=True),
            self._line("Body text follows here with several ordinary words in it.", 30),
        ]
        blocks = build_blocks_from_pages([self._page(lines)])
        self.assertTrue(all(block["type"] == "text" for block in blocks))


class PageFurnitureTests(unittest.TestCase):
    @staticmethod
    def _line(text, top, page_height=800.0):
        return {"text": text, "top": top, "bottom": top + 10.0, "x0": 0.0, "x1": 200.0,
                "size": 10.0, "bold": False}

    def _page(self, page_number, body_text, footer_text):
        return {
            "page_number": page_number,
            "height": 800.0,
            "tables": [],
            "lines": [
                self._line("BHC School Handbook", 20.0),
                self._line(body_text, 400.0),
                self._line(footer_text, 780.0),
            ],
        }

    def test_repeated_headers_and_numbered_footers_are_dropped(self):
        pages = [
            self._page(i, f"Unique body paragraph number {i} with plenty of words.", f"Page {i + 1} of 4")
            for i in range(4)
        ]
        cleaned = remove_page_furniture(pages)
        for page in cleaned:
            texts = [line["text"] for line in page["lines"]]
            self.assertEqual(1, len(texts))
            self.assertIn("Unique body paragraph", texts[0])

    def test_body_line_repeated_mid_page_is_kept(self):
        # Same text on every page but OUTSIDE the edge bands -> not furniture.
        pages = []
        for i in range(4):
            pages.append({
                "page_number": i,
                "height": 800.0,
                "tables": [],
                "lines": [
                    self._line("All fees include registration.", 400.0),
                    self._line(f"Body {i} with several ordinary words in it.", 420.0),
                ],
            })
        cleaned = remove_page_furniture(pages)
        for page in cleaned:
            self.assertEqual(2, len(page["lines"]))

    def test_short_documents_are_left_untouched(self):
        pages = [self._page(i, f"Body {i}", "Page N") for i in range(2)]
        self.assertEqual(pages, remove_page_furniture(pages))


def _text_block(content, page, top):
    return {"type": "text", "content": content, "page_number": page, "top": top}


def _table_block(rows, page, top):
    return {
        "type": "table",
        "content": "\n".join(" | ".join(row) for row in rows),
        "rows": rows,
        "page_number": page,
        "top": top,
    }


SAMPLE_BLOCKS = [
    _text_block("Intro paragraph about the fee schedule.", 0, 10.0),
    _table_block([["Grade", "Fee"], ["1", "100"], ["2", "200"]], 0, 50.0),
    _text_block("Closing note under the table.", 0, 90.0),
    _text_block("Second page opening paragraph.", 1, 12.0),
]


class LayoutLoaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.loader = DocumentLoader()

    def _load(self, blocks):
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            return self.loader.load_document("fees.pdf", "fees.pdf")

    TABLE_TEXT = "Grade | Fee\n1 | 100\n2 | 200"

    def test_table_leaf_is_pure_and_row_safe(self):
        docs = self._load(SAMPLE_BLOCKS)
        table_leaves = [
            d for d in docs if d["chunk_level"] == 3 and d["text"] == self.TABLE_TEXT
        ]
        self.assertEqual(1, len(table_leaves))
        table = table_leaves[0]
        self.assertEqual(0, table["page_number"])
        # Pure table leaf: no surrounding prose diluted into the embedding text.
        self.assertNotIn("Intro paragraph", table["text"])

    def test_table_leaf_has_section_parents_containing_topic_context(self):
        docs = self._load(SAMPLE_BLOCKS)
        by_id = {d["chunk_id"]: d for d in docs}
        table = next(
            d for d in docs if d["chunk_level"] == 3 and d["text"] == self.TABLE_TEXT
        )

        # Normal hierarchy citizenship: real L2 parent and L1 root, like any leaf.
        self.assertTrue(table["parent_chunk_id"])
        self.assertTrue(table["root_chunk_id"])
        self.assertNotEqual(table["chunk_id"], table["root_chunk_id"])

        parent = by_id[table["parent_chunk_id"]]
        root = by_id[table["root_chunk_id"]]
        self.assertEqual(2, parent["chunk_level"])
        self.assertEqual(1, root["chunk_level"])
        # The parents carry the table WITH its section context, so auto-merging a
        # table hit yields table-plus-topic rather than a floating grid.
        for ancestor in (parent, root):
            self.assertIn(self.TABLE_TEXT, ancestor["text"])
            self.assertIn("Intro paragraph", ancestor["text"])
            self.assertIn("Closing note", ancestor["text"])

    def test_text_and_table_share_three_level_hierarchy_with_unique_ids(self):
        docs = self._load(SAMPLE_BLOCKS)
        chunk_ids = [d["chunk_id"] for d in docs]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertEqual({1, 2, 3}, {d["chunk_level"] for d in docs})

        for leaf in (d for d in docs if d["chunk_level"] == 3):
            self.assertTrue(leaf["parent_chunk_id"])
            self.assertTrue(leaf["root_chunk_id"])

        # Text leaves never absorb table rows (leaf isolation).
        text_leaves = [
            d for d in docs if d["chunk_level"] == 3 and d["text"] != self.TABLE_TEXT
        ]
        self.assertTrue(text_leaves)
        for leaf in text_leaves:
            self.assertNotIn("Grade | Fee", leaf["text"])

    def test_chunk_idx_is_strictly_increasing_and_page_is_window_start(self):
        docs = self._load(SAMPLE_BLOCKS)
        idxs = [d["chunk_idx"] for d in docs]
        self.assertEqual(idxs, sorted(idxs))
        self.assertEqual(len(idxs), len(set(idxs)))
        # Windows span pages now; each chunk anchors to the page its window starts
        # on. All sample windows start on page 0.
        self.assertEqual({0}, {d["page_number"] for d in docs})
        # Page-1 content still lands in the index (inside a window starting on p0).
        self.assertTrue(any("Second page opening" in d["text"] for d in docs if d["chunk_level"] == 3))

    HEADED_BLOCKS = [
        {"type": "heading", "content": "Admission Fees", "level": 1, "page_number": 0, "top": 5.0},
        _text_block("Fees are reviewed annually by the board.", 0, 20.0),
        _table_block([["Grade", "Fee"], ["1", "100"], ["2", "200"]], 0, 40.0),
        {"type": "heading", "content": "Transportation", "level": 1, "page_number": 0, "top": 80.0},
        _text_block("Bus service is optional for all grades.", 0, 90.0),
    ]

    def test_table_leaf_carries_its_section_topic(self):
        docs = self._load(self.HEADED_BLOCKS)
        table_leaf = next(
            d for d in docs if d["chunk_level"] == 3 and "Grade | Fee" in d["text"]
        )
        # The section-path prefix answers "what topic is this table under".
        self.assertTrue(table_leaf["text"].startswith("Admission Fees\n"))
        self.assertIn("2 | 200", table_leaf["text"])
        self.assertNotIn("Bus service", table_leaf["text"])

    def test_leaf_windows_do_not_cross_section_boundaries(self):
        docs = self._load(self.HEADED_BLOCKS)
        leaves = [d for d in docs if d["chunk_level"] == 3]
        fees_leaf = next(d for d in leaves if "reviewed annually" in d["text"])
        self.assertNotIn("Bus service", fees_leaf["text"])
        transport_leaf = next(d for d in leaves if "Bus service" in d["text"])
        # Section heading rides inline (no duplicate prefix added on top).
        self.assertTrue(transport_leaf["text"].startswith("Transportation"))
        self.assertEqual(1, transport_leaf["text"].count("Transportation"))

    def test_small_fragments_merge_into_topic_sized_leaves(self):
        blocks = [
            {"type": "heading", "content": "Uniform Policy", "level": 1, "page_number": 0, "top": 5.0},
            _text_block("Shirts must be white.", 0, 20.0),
            _text_block("Trousers must be grey.", 0, 30.0),
            _text_block("Shoes must be black.", 0, 40.0),
        ]
        docs = self._load(blocks)
        leaves = [d for d in docs if d["chunk_level"] == 3]
        self.assertEqual(1, len(leaves))
        for fragment in ("Shirts", "Trousers", "Shoes"):
            self.assertIn(fragment, leaves[0]["text"])
        self.assertTrue(leaves[0]["text"].startswith("Uniform Policy"))

    def test_paragraph_split_by_page_break_is_stitched(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("The admission process requires families to", 0, 700.0),
            _text_block("submit all documents before the deadline.", 1, 30.0),
        ])
        self.assertEqual(1, len(blocks))
        self.assertEqual(
            "The admission process requires families to submit all documents before the deadline.",
            blocks[0]["content"],
        )
        self.assertEqual(0, blocks[0]["page_number"])

    def test_completed_paragraph_is_not_stitched(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("All documents were received on time.", 0, 700.0),
            _text_block("the next section covers transportation.", 1, 30.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_table_split_by_page_break_is_stitched_with_repeated_header_dropped(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _table_block([["Grade", "Fee"], ["1", "100"]], 0, 700.0),
            _table_block([["Grade", "Fee"], ["2", "200"]], 1, 30.0),
            _table_block([["Grade", "Fee"], ["3", "300"]], 2, 30.0),
        ])
        self.assertEqual(1, len(blocks))
        self.assertEqual(
            [["Grade", "Fee"], ["1", "100"], ["2", "200"], ["3", "300"]],
            blocks[0]["rows"],
        )
        self.assertEqual(0, blocks[0]["page_number"])

    def test_table_with_different_column_count_is_not_stitched(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _table_block([["Grade", "Fee"], ["1", "100"]], 0, 700.0),
            _table_block([["Day", "Start", "End"], ["Mon", "8", "3"]], 1, 30.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_cross_page_section_chunks_as_one_context(self):
        blocks = [
            {"type": "heading", "content": "Admission Process", "level": 1, "page_number": 0, "top": 5.0},
            _text_block("Families must first complete the online form and", 0, 700.0),
            _text_block("then book an assessment appointment for the child.", 1, 30.0),
        ]
        docs = self._load(blocks)
        leaves = [d for d in docs if d["chunk_level"] == 3]
        self.assertEqual(1, len(leaves))
        leaf = leaves[0]
        self.assertIn("online form and then book an assessment", leaf["text"])
        self.assertTrue(leaf["text"].startswith("Admission Process"))

    def test_layout_failure_falls_back_to_flat_text_extraction(self):
        class _StubPage:
            def __init__(self, content, page):
                self.page_content = content
                self.metadata = {"page": page}

        class _StubPyPDFLoader:
            def __init__(self, file_path):
                self.file_path = file_path

            def load(self):
                return [_StubPage("legacy flat text content", 0)]

        with patch.object(
            document_loader_module, "parse_pdf_blocks", side_effect=RuntimeError("boom")
        ), patch.object(document_loader_module, "PyPDFLoader", _StubPyPDFLoader):
            docs = self.loader.load_document("broken.pdf", "broken.pdf")

        self.assertTrue(docs)
        self.assertTrue(all("legacy flat text content" in d["text"] for d in docs))

    def test_flag_disabled_skips_layout_parser_entirely(self):
        class _StubPage:
            def __init__(self, content, page):
                self.page_content = content
                self.metadata = {"page": page}

        class _StubPyPDFLoader:
            def __init__(self, file_path):
                self.file_path = file_path

            def load(self):
                return [_StubPage("legacy path", 0)]

        layout_mock = Mock()
        with patch.object(document_loader_module, "PDF_LAYOUT_PARSER_ENABLED", False), patch.object(
            document_loader_module, "parse_pdf_blocks", layout_mock
        ), patch.object(document_loader_module, "PyPDFLoader", _StubPyPDFLoader):
            docs = self.loader.load_document("any.pdf", "any.pdf")

        layout_mock.assert_not_called()
        self.assertTrue(docs)


if __name__ == "__main__":
    unittest.main()
