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
from tests.general.test_milvus_writer import FakeMilvusStore, load_milvus_writer_module


def _line(text, top=10.0, size=10.0, bold=False):
    return {"text": text, "top": top, "bottom": top + size, "x0": 0.0, "x1": 200.0,
            "size": size, "bold": bold}


def _text_block(content, page, top):
    return {"type": "text", "content": content, "page_number": page, "top": top}


def _table_block(rows, page, top, bottom=None, page_height=None,
                 page_top=None, page_bottom=None, x0=None, x1=None):
    """A table block. The geometry is what pdf_layout carries so the stitcher can ask
    whether a break fell at the page edges; omitting it is the older shape, and every
    caller that omits it is asserting the no-geometry path."""
    block = {"type": "table", "content": "", "rows": rows, "page_number": page, "top": top}
    for key, value in (("bottom", bottom), ("page_height", page_height),
                       ("page_top", page_top), ("page_bottom", page_bottom),
                       ("x0", x0), ("x1", x1)):
        if value is not None:
            block[key] = value
    return block


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
        # 1-row table whose render exceeds the budget: must still produce groups, never
        # silently drop the data. It used to come back as the single whole row; it is now
        # cut to the budget, because a header that fills the budget leaves no room to
        # repeat it and so no grid to preserve.
        rows = [["A" * 50, "B" * 50]]
        groups = self.loader._split_table_row_groups(rows, budget=60)

        self.assertTrue(groups)
        rendered = "".join(self.loader._render_rows(group) for group in groups)
        self.assertEqual(50, rendered.count("A"))
        self.assertEqual(50, rendered.count("B"))
        for group in groups:
            self.assertLessEqual(len(self.loader._render_rows(group)), 60)

    def test_a_body_row_larger_than_the_budget_is_cut_rather_than_shipped_whole(self):
        """This used to ship whole, with its header, on the reasoning that cutting
        mid-row is worse than a large chunk.

        It is not, and that asymmetry is what this phase exists to correct. An over-long
        row made the budget advisory: one cell holding a paragraph produced a leaf
        bounded only by `_MILVUS_TEXT_CAP_BYTES`, sixty thousand bytes, and a handful of
        those is a grading prompt sized by the corpus — the failure commit 1794762
        stopped by truncating the grader's view instead. Measured end to end before this
        change, a 40-row fee table with one paragraph-sized cell produced a 3,652-character
        leaf; after it, 800.

        The row loses its columns. Every other row keeps them, and nothing is lost."""
        rows = [["Grade", "Fee"], ["X" * 100, "Y" * 100]]
        groups = self.loader._split_table_row_groups(rows, budget=50)

        for group in groups:
            self.assertLessEqual(len(self.loader._render_rows(group)), 50)
            self.assertEqual(["Grade", "Fee"], group[0], "every group keeps its header")
        rendered = "".join(self.loader._render_rows(group) for group in groups)
        self.assertEqual(100, rendered.count("X"))
        self.assertEqual(100, rendered.count("Y"))

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
        windows = DocumentLoader._pack_units(units, budget=100, isolate_atomic=True)
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


#: A4 at 72dpi, the page these cases are laid out on. The edge band is 22% of it, so a
#: table is at the foot past 656.8 and at the head before 185.2.
A4 = 842.0
FOOT = 780.0          # a table running to the bottom text margin
HEAD = 70.0           # a table starting at the top text margin
MIDDLE = 400.0        # neither


#: A4 at 72dpi, the page these cases are laid out on. The edge band is 22% of it, so a
#: table is at the foot past 656.8 and at the head before 185.2.
A4 = 842.0
FOOT = 780.0          # a table running to the bottom text margin
HEAD = 70.0           # a table starting at the top text margin
MIDDLE = 400.0        # neither
LEFT, RIGHT = 72.0, 523.0     # the text column, edge to edge


class CrossPageTableJoinTests(unittest.TestCase):
    """Which two tables are ONE table, and which are two.

    The rule used to be "same column count, next page", which is true of any two
    unrelated three-column tables that land either side of a break. Joining them makes
    one grid out of two, and a row from the second then answers a question asked about
    the first — the wrong-line failure, built in at indexing time where nothing
    downstream can see it, because by then there is only one table. Item 6.

    Both directions are enumerated, because both are bugs and they are not equally cheap.
    A wrong JOIN corrupts a grid. A wrong SPLIT strands a continuation from its header —
    still indexed, still retrievable, but answering with a column it can no longer name.
    Neither is acceptable and the second is by far the easier to cause, which is why
    every refusal here is structural. Nothing about what the cells SAY may refuse a join;
    two content rules were tried and both split real tables (see `_is_table_continuation`).
    """

    def setUp(self):
        self.loader = DocumentLoader()

    def _stitch(self, *blocks):
        return self.loader._stitch_cross_page_blocks(list(blocks))

    def _rows(self, blocks):
        return blocks[0]["rows"]

    def _cut(self, rows, page, **kw):
        """A table running to the foot of its page — the shape of one about to be cut."""
        return _table_block(rows, page, kw.pop("top", 500.0), kw.pop("bottom", FOOT),
                            A4, x0=kw.pop("x0", LEFT), x1=kw.pop("x1", RIGHT), **kw)

    def _resumed(self, rows, page, **kw):
        """A table starting at the head of its page — the shape of one resuming."""
        return _table_block(rows, page, kw.pop("top", HEAD), kw.pop("bottom", MIDDLE),
                            A4, x0=kw.pop("x0", LEFT), x1=kw.pop("x1", RIGHT), **kw)

    # =================================================================================
    # ONE table, split by the break. Every one of these must JOIN.
    # =================================================================================

    def test_numeric_rows_with_the_header_repeated(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "95,000"]], 0),
            self._resumed([["Grade", "Fee"], ["Y03", "105,000"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual([["Grade", "Fee"], ["Y01", "95,000"], ["Y03", "105,000"]],
                         self._rows(blocks), "the repeated header is dropped exactly once")

    def test_numeric_rows_carrying_straight_on(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "95,000"]], 0),
            self._resumed([["Y03", "105,000"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)))

    def test_prose_rows_carrying_straight_on(self):
        """The corpus's curriculum tables, verbatim. 19 of its 30 table rows carry no
        digit at all, so a rule reading a digit as proof of a data row split every one of
        these. Measured only after that rule shipped, which is why no content test may
        refuse a join now."""
        blocks = self._stitch(
            self._cut([["Subject Group", "Subjects Taught"],
                       ["Mathematics", "Counting, Place Value, Time & Money"]], 0),
            self._resumed([["Humanities", "History, Geography and Egyptian Social Studies"],
                           ["Languages", "Arabic Language, Islamic Education, French"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(4, len(self._rows(blocks)))

    def test_arabic_prose_rows(self):
        blocks = self._stitch(
            self._cut([["المادة", "المحتوى"], ["الرياضيات", "العد والقياس"]], 0),
            self._resumed([["العلوم", "التجارب والمواد"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)))

    def test_a_table_mixing_arabic_and_latin_cells(self):
        blocks = self._stitch(
            self._cut([["الصف", "Fee"], ["الحضانة", "75,000 EGP"]], 0),
            self._resumed([["Year 3", "105,000 جنيه"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)))

    def test_a_cjk_table(self):
        blocks = self._stitch(
            self._cut([["年级", "学费"], ["一年级", "95,000"]], 0),
            self._resumed([["三年级", "105,000"]], 1),
        )
        self.assertEqual(1, len(blocks))

    def test_a_header_the_break_respaced_and_recased(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee (EGP)"], ["Y01", "95,000"]], 0),
            self._resumed([[" grade ", "FEE  (egp)"], ["Y03", "105,000"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)), "the copy goes, not a data row")

    def test_a_header_differing_only_in_punctuation(self):
        blocks = self._stitch(
            self._cut([["Grade:", "Fee -"], ["Y01", "1"]], 0),
            self._resumed([["Grade", "Fee"], ["Y03", "3"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)))

    def test_a_header_repeated_on_every_page_of_a_chain(self):
        """Three pages, three copies of the header, one header in the result."""
        header = ["Grade", "Fee"]
        blocks = self._stitch(
            self._cut([header, ["Y01", "1"]], 0),
            self._cut([header, ["Y03", "3"]], 1, top=HEAD),
            self._resumed([header, ["Y11", "11"]], 2),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual([header, ["Y01", "1"], ["Y03", "3"], ["Y11", "11"]],
                         self._rows(blocks))

    def test_a_table_spanning_three_pages(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "95,000"]], 0),
            self._cut([["Y03", "105,000"]], 1, top=HEAD),
            self._resumed([["Y11", "150,000"]], 2),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(4, len(self._rows(blocks)))

    def test_a_table_spanning_ten_pages(self):
        blocks = self._stitch(
            self._cut([["n", "v"], ["0", "0"]], 0),
            *[self._cut([[str(page), str(page)]], page, top=HEAD) for page in range(1, 9)],
            self._resumed([["9", "9"]], 9),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(11, len(self._rows(blocks)))
        self.assertEqual(9, blocks[0]["_last_page"])

    def test_the_last_page_of_a_chain_may_end_anywhere(self):
        """Only the page being LEFT has to be full. The page arriving does not."""
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            self._resumed([["Y03", "3"]], 1, bottom=120.0),
        )
        self.assertEqual(1, len(blocks))

    def test_a_generous_bottom_margin(self):
        """1.5 inches of margin. At the old 12% band this genuine continuation split."""
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0, bottom=A4 - 108),
            self._resumed([["Y03", "3"]], 1, top=108.0),
        )
        self.assertEqual(1, len(blocks))

    def test_a_two_inch_margin_with_a_footer_below_the_table(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0, bottom=A4 - 144),
            self._resumed([["Y03", "3"]], 1, top=144.0),
        )
        self.assertEqual(1, len(blocks))

    def test_exactly_on_the_edge_band(self):
        """BVA: the last position that still counts as cut off, and as resumed."""
        band = A4 * 0.22
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0, bottom=A4 - band),
            self._resumed([["Y03", "3"]], 1, top=band),
        )
        self.assertEqual(1, len(blocks))

    def test_a_cropped_page_whose_coordinates_do_not_start_at_zero(self):
        """page_height is an extent; a cropped page's ink is measured from bbox[1]. Both
        edges travel with the block so the comparison cannot drift."""
        offset = 200.0
        blocks = self._stitch(
            _table_block([["Grade", "Fee"], ["Y01", "1"]], 0, offset + 300, offset + FOOT,
                         page_top=offset, page_bottom=offset + A4, x0=LEFT, x1=RIGHT),
            _table_block([["Y03", "3"]], 1, offset + HEAD, offset + MIDDLE,
                         page_top=offset, page_bottom=offset + A4, x0=LEFT, x1=RIGHT),
        )
        self.assertEqual(1, len(blocks))

    def test_a_landscape_page_after_a_portrait_one(self):
        """Each side is judged against its own page, so a rotated page still works."""
        blocks = self._stitch(
            _table_block([["Grade", "Fee"], ["Y01", "1"]], 0, 500.0, FOOT, A4,
                         x0=LEFT, x1=RIGHT),
            _table_block([["Y03", "3"]], 1, 40.0, 300.0, 595.0, x0=LEFT, x1=RIGHT),
        )
        self.assertEqual(1, len(blocks))

    def test_a_parser_reporting_no_geometry_at_all(self):
        """DOCX and XLSX report none, and DOCX never reaches here anyway — its pages are
        all numbered 0 and the rule needs page + 1. The question cannot be asked, so it is
        not held against them."""
        blocks = self._stitch(
            _table_block([["Grade", "Fee"], ["Y01", "1"]], 0, 700.0),
            _table_block([["Y03", "3"]], 1, 30.0),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)))

    def test_vertical_geometry_on_only_one_side(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            _table_block([["Y03", "3"]], 1, 30.0),
        )
        self.assertEqual(1, len(blocks))

    def test_a_horizontal_span_reported_on_only_one_side(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            _table_block([["Y03", "3"]], 1, HEAD, MIDDLE, A4),
        )
        self.assertEqual(1, len(blocks))

    def test_a_span_that_shifted_slightly_between_pages(self):
        """A borderless grid's detected edge moves with its content. Well inside the
        tolerance, and a continuation must not be split over a few points."""
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0, x0=LEFT, x1=RIGHT),
            self._resumed([["Y03", "3"]], 1, x0=LEFT + 12, x1=RIGHT - 9),
        )
        self.assertEqual(1, len(blocks))

    def test_a_span_exactly_at_the_overlap_threshold(self):
        """BVA on the horizontal rule: 70% of the wider span shared."""
        width = RIGHT - LEFT
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0, x0=LEFT, x1=RIGHT),
            self._resumed([["Y03", "3"]], 1, x0=LEFT + width * 0.3, x1=RIGHT),
        )
        self.assertEqual(1, len(blocks))

    def test_a_single_column_table(self):
        blocks = self._stitch(
            self._cut([["Policy"], ["No phones"]], 0),
            self._resumed([["No jewellery"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)))

    def test_a_ten_column_table(self):
        header = [f"c{i}" for i in range(10)]
        blocks = self._stitch(
            self._cut([header, ["x"] * 10], 0),
            self._resumed([["y"] * 10], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)))

    def test_a_header_with_a_merged_cell_over_a_wider_body(self):
        """The header spans two columns and the body has three. Counting columns from
        the first row alone refused this join over a difference only in the heading."""
        blocks = self._stitch(
            self._cut([["Fees", "EGP"], ["Y01", "95,000", "Egyptian"],
                       ["Y03", "105,000", "Egyptian"]], 0),
            self._resumed([["Y11", "150,000", "Egyptian"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(4, len(self._rows(blocks)))

    def test_a_continuation_of_one_row(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            self._resumed([["Y03", "3"]], 1),
        )
        self.assertEqual(1, len(blocks))

    def test_a_continuation_of_many_rows(self):
        blocks = self._stitch(
            self._cut([["n", "v"]] + [[str(i), str(i)] for i in range(40)], 0),
            self._resumed([[str(i), str(i)] for i in range(40, 90)], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(91, len(self._rows(blocks)))

    def test_a_continuation_whose_first_row_has_empty_cells(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            self._resumed([["", "3"], ["Y11", "11"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(4, len(self._rows(blocks)))

    def test_a_continuation_repeating_a_DATA_row_keeps_it(self):
        """Only the header is a duplicate worth dropping. A data row that happens to
        equal one already seen is data, and losing it loses a fact."""
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "95,000"]], 0),
            self._resumed([["Y01", "95,000"], ["Y03", "105,000"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(4, len(self._rows(blocks)))

    def test_a_totals_row_opening_the_continuation(self):
        """"Total | 500,000" reads like a heading and is not one."""
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "95,000"]], 0),
            self._resumed([["Total", "500,000"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)))

    def test_a_table_of_only_numbers(self):
        blocks = self._stitch(
            self._cut([["1", "2"], ["3", "4"]], 0),
            self._resumed([["5", "6"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)))

    # =================================================================================
    # Two tables. Every one of these must stay SEPARATE.
    # =================================================================================

    def test_a_table_that_stopped_mid_page_was_not_cut_off(self):
        """The load-bearing refusal. Nothing cut this table short, so what follows on the
        next page is a different table."""
        blocks = self._stitch(
            _table_block([["Grade", "Fee"], ["Y01", "1"]], 0, 200.0, MIDDLE, A4,
                         x0=LEFT, x1=RIGHT),
            self._resumed([["Programme", "Cost"], ["Half-Day", "2,500"]], 1),
        )
        self.assertEqual(2, len(blocks))

    def test_a_table_starting_mid_page_had_something_above_it(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            _table_block([["Programme", "Cost"], ["Half-Day", "2,500"]], 1, MIDDLE, 700.0,
                         A4, x0=LEFT, x1=RIGHT),
        )
        self.assertEqual(2, len(blocks))

    def test_neither_one_at_an_edge(self):
        blocks = self._stitch(
            _table_block([["Grade", "Fee"], ["Y01", "1"]], 0, 200.0, MIDDLE, A4,
                         x0=LEFT, x1=RIGHT),
            _table_block([["Programme", "Cost"], ["Half-Day", "2"]], 1, MIDDLE, 600.0, A4,
                         x0=LEFT, x1=RIGHT),
        )
        self.assertEqual(2, len(blocks))

    def test_one_point_past_the_edge_band(self):
        """BVA: the first position that no longer counts as cut off."""
        band = A4 * 0.22
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0, bottom=A4 - band - 1),
            self._resumed([["Y03", "3"]], 1),
        )
        self.assertEqual(2, len(blocks))

    def test_one_point_past_the_head_band(self):
        band = A4 * 0.22
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            self._resumed([["Y03", "3"]], 1, top=band + 1),
        )
        self.assertEqual(2, len(blocks))

    def test_a_narrow_table_below_a_wide_one(self):
        """Both meet at the page edges and both have two columns, so only the horizontal
        span tells them apart — a sidebar grid under a full-width one."""
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0, x0=LEFT, x1=RIGHT),
            self._resumed([["Note", "See above"]], 1, x0=LEFT, x1=LEFT + 120),
        )
        self.assertEqual(2, len(blocks))

    def test_a_table_in_the_other_column_of_the_page(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0, x0=LEFT, x1=LEFT + 200),
            self._resumed([["Programme", "Cost"]], 1, x0=RIGHT - 200, x1=RIGHT),
        )
        self.assertEqual(2, len(blocks))

    def test_a_different_column_count(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            self._resumed([["Y03", "3", "extra"]], 1),
        )
        self.assertEqual(2, len(blocks))

    def test_a_body_whose_width_differs_from_the_previous_body(self):
        """Counting columns by the width most rows share, not the first row's."""
        blocks = self._stitch(
            self._cut([["Fees", "EGP"], ["Y01", "1", "Egyptian"], ["Y03", "3", "Egyptian"]], 0),
            self._resumed([["Half-Day", "2,500"], ["Full-Day", "4,000"]], 1),
        )
        self.assertEqual(2, len(blocks))

    def test_a_page_gap_of_two(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            self._resumed([["Y03", "3"]], 2),
        )
        self.assertEqual(2, len(blocks))

    def test_a_page_gap_of_ten(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            self._resumed([["Y03", "3"]], 10),
        )
        self.assertEqual(2, len(blocks))

    def test_two_tables_on_the_same_page(self):
        blocks = self._stitch(
            _table_block([["Grade", "Fee"], ["Y01", "1"]], 0, 100.0, MIDDLE, A4,
                         x0=LEFT, x1=RIGHT),
            self._cut([["Y03", "3"]], 0),
        )
        self.assertEqual(2, len(blocks))

    def test_a_table_on_an_earlier_page(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 1),
            self._resumed([["Y03", "3"]], 0),
        )
        self.assertEqual(2, len(blocks))

    def test_a_heading_between_them(self):
        """"The same section" needs no test of its own: the heading becomes `prev`."""
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            {"type": "heading", "content": "Summer Camp", "level": 1,
             "page_number": 1, "top": 40.0},
            self._resumed([["Programme", "Cost"], ["Half-Day", "2,500"]], 1, top=120.0),
        )
        self.assertEqual(3, len(blocks))

    def test_a_paragraph_between_them(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            _text_block("The camp is priced separately.", 1, 40.0),
            self._resumed([["Programme", "Cost"], ["Half-Day", "2,500"]], 1, top=120.0),
        )
        self.assertEqual(3, len(blocks))

    def test_a_chain_does_not_swallow_a_new_table_after_it_ends(self):
        """Page two finishes the table half way down; page three starts a different one.
        Judged against page ONE's geometry the run still looked cut off, so all three
        merged into a single grid — the worse of the two failures, and invisible after."""
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            self._resumed([["Y03", "3"]], 1, bottom=200.0),
            self._resumed([["Programme", "Cost"], ["Half-Day", "2,500"]], 2),
        )
        self.assertEqual(2, len(blocks))
        self.assertEqual(3, len(blocks[0]["rows"]), "pages one and two are one table")
        self.assertEqual(2, len(blocks[1]["rows"]))

    def test_a_chain_does_not_swallow_a_headerless_new_table_either(self):
        """The same stale-geometry bug with nothing in the content to mask it: page three
        opens with a plain data row, so only where the ink sits can refuse it."""
        blocks = self._stitch(
            self._cut([["n", "v"], ["1", "100"]], 0),
            self._resumed([["2", "200"]], 1, bottom=200.0),
            self._resumed([["3", "300"]], 2),
        )
        self.assertEqual(2, len(blocks))
        self.assertEqual(3, len(blocks[0]["rows"]))

    def test_a_chain_does_not_swallow_a_table_of_a_different_width(self):
        blocks = self._stitch(
            self._cut([["n", "v"], ["1", "100"]], 0),
            self._cut([["2", "200"]], 1, top=HEAD),
            self._resumed([["3", "300"]], 2, x0=LEFT, x1=LEFT + 100),
        )
        self.assertEqual(2, len(blocks))

    def test_an_empty_table_before(self):
        blocks = self._stitch(
            self._cut([], 0),
            self._resumed([["Y03", "3"]], 1),
        )
        self.assertEqual(2, len(blocks))

    def test_an_empty_table_after(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"]], 0),
            self._resumed([], 1),
        )
        self.assertEqual(2, len(blocks))

    def test_a_text_block_before_a_table(self):
        blocks = self._stitch(
            _text_block("Fees are reviewed annually", 0, 700.0),
            self._resumed([["Grade", "Fee"]], 1),
        )
        self.assertEqual(2, len(blocks))

    def test_a_table_before_a_text_block(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            _text_block("Fees are reviewed annually.", 1, 40.0),
        )
        self.assertEqual(2, len(blocks))

    # =================================================================================
    # What the merged block looks like afterwards.
    # =================================================================================

    def test_the_merged_block_cites_the_page_it_started_on(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 3),
            self._resumed([["Y03", "3"]], 4),
        )
        self.assertEqual(3, blocks[0]["page_number"])
        self.assertEqual(4, blocks[0]["_last_page"])

    def test_the_merged_content_is_rendered_from_the_merged_rows(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "95,000"]], 0),
            self._resumed([["Y03", "105,000"]], 1),
        )
        self.assertIn("Y01 | 95,000", blocks[0]["content"])
        self.assertIn("Y03 | 105,000", blocks[0]["content"])

    def test_row_order_survives_the_join(self):
        blocks = self._stitch(
            self._cut([["n"], ["1"], ["2"]], 0),
            self._resumed([["3"], ["4"]], 1),
        )
        self.assertEqual([["n"], ["1"], ["2"], ["3"], ["4"]], self._rows(blocks))

    def test_the_merged_block_carries_the_geometry_it_ended_on(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0),
            self._resumed([["Y03", "3"]], 1, bottom=321.0),
        )
        self.assertEqual(321.0, blocks[0]["_last_bottom"])

    def test_the_input_blocks_are_not_mutated(self):
        """The caller's list is parsed data, not scratch space."""
        first = self._cut([["Grade", "Fee"], ["Y01", "1"]], 0)
        second = self._resumed([["Y03", "3"]], 1)
        self._stitch(first, second)
        self.assertEqual([["Grade", "Fee"], ["Y01", "1"]], first["rows"])
        self.assertNotIn("_last_page", first)

    def test_blocks_that_never_join_pass_through_unchanged(self):
        heading = {"type": "heading", "content": "Fees", "level": 1,
                   "page_number": 0, "top": 10.0}
        blocks = self._stitch(heading, self._cut([["Grade", "Fee"], ["Y01", "1"]], 0))
        self.assertEqual(heading, blocks[0])

    def test_an_empty_block_stream(self):
        self.assertEqual([], self._stitch())

    def test_a_single_block(self):
        blocks = self._stitch(self._cut([["Grade", "Fee"]], 0))
        self.assertEqual(1, len(blocks))

    # =================================================================================
    # Malformed input. None of it may raise.
    # =================================================================================

    def test_rows_holding_none_cells(self):
        """Depth, not a live bug: every parser routes its table through
        `normalize_table_rows` first, so a None cell does not reach the stitcher today.
        It is asserted because rendering is where one WOULD raise, and raising there
        fails the whole document's ingest rather than the row."""
        blocks = self._stitch(
            self._cut([["Grade", None], ["Y01", "1"]], 0),
            self._resumed([["Grade", None], ["Y03", "3"]], 1),
        )
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(self._rows(blocks)), "the header still matches through None")

    def test_rows_holding_numbers_rather_than_strings(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", 95000]], 0),
            self._resumed([["Y03", 105000]], 1),
        )
        self.assertEqual(1, len(blocks))

    def test_a_block_with_no_rows_key(self):
        first = self._cut([["Grade", "Fee"]], 0)
        second = self._resumed([["Y03", "3"]], 1)
        del second["rows"]
        self.assertEqual(2, len(self._stitch(first, second)))

    def test_a_block_with_no_page_number(self):
        second = self._resumed([["Y03", "3"]], 1)
        del second["page_number"]
        self.assertEqual(2, len(self._stitch(self._cut([["Grade", "Fee"]], 0), second)))

    def test_a_page_of_zero_height(self):
        """Degenerate geometry is unanswerable, not a refusal."""
        blocks = self._stitch(
            _table_block([["Grade", "Fee"], ["Y01", "1"]], 0, 0.0, 0.0,
                         page_top=0.0, page_bottom=0.0),
            _table_block([["Y03", "3"]], 1, 0.0, 0.0, page_top=0.0, page_bottom=0.0),
        )
        self.assertEqual(1, len(blocks))

    def test_negative_page_coordinates(self):
        blocks = self._stitch(
            _table_block([["Grade", "Fee"], ["Y01", "1"]], 0, -500.0, -70.0,
                         page_top=-842.0, page_bottom=0.0, x0=LEFT, x1=RIGHT),
            _table_block([["Y03", "3"]], 1, -800.0, -600.0,
                         page_top=-842.0, page_bottom=0.0, x0=LEFT, x1=RIGHT),
        )
        self.assertEqual(1, len(blocks))

    def test_a_zero_width_horizontal_span(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"]], 0, x0=100.0, x1=100.0),
            self._resumed([["Y03", "3"]], 1, x0=100.0, x1=100.0),
        )
        self.assertEqual(1, len(blocks))

    def test_rows_of_uneven_width_on_both_sides(self):
        blocks = self._stitch(
            self._cut([["Grade", "Fee"], ["Y01", "1"], ["note"]], 0),
            self._resumed([["Y03", "3"], ["Y11", "11"], ["note"]], 1),
        )
        self.assertEqual(1, len(blocks))


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
    the 150-char prefix cap, and the depth-3 path truncation.

    Run under `full` EXPLICITLY. The shipped default is now `none` — five reindexed arms
    measured the path costing recall rather than adding it, in Arabic and in English — but
    the other modes remain selectable and their mechanics are still worth pinning. What
    ships is asserted by `TheShippedPrefixDefaultTests` below.
    """

    def setUp(self):
        patcher = patch.dict(os.environ, {"CHUNK_SECTION_PREFIX": "full"})
        patcher.start()
        self.addCleanup(patcher.stop)

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


class TheShippedPrefixDefaultTests(unittest.TestCase):
    """What a chunk carries when nobody sets anything.

    The section path was measured five ways, each a full reindex over 349 questions:
    with the document title and three levels (the old default) scored 320 ranked, without
    the title 319, and with no path at all 322. Re-measured on English queries — the
    language retrieval actually runs in, since an Arabic question is translated before it
    reaches the index — the path earned nothing there either. Only 34 distinct strings
    covered 118 of 175 leaves, 18% of their text, and embedding them raised mean pairwise
    cosine from 0.4518 to 0.5156: every chunk looking more like every other one.
    """

    def test_a_chunk_carries_no_section_path_by_default(self):
        self.assertEqual(
            "body", DocumentLoader._apply_section_prefix("body", ["Doc", "Section", "Sub"])
        )

    def test_not_even_a_table(self):
        """`structured` — the path on tables and figures only — was measured too, and
        came last of the five at 317."""
        self.assertEqual(
            "body",
            DocumentLoader._apply_section_prefix("body", ["Doc", "Section"], "table"),
        )

    def test_the_other_modes_are_still_reachable(self):
        with patch.dict(os.environ, {"CHUNK_SECTION_PREFIX": "full"}):
            self.assertTrue(
                DocumentLoader._apply_section_prefix("body", ["Doc"]).startswith("Doc\n")
            )


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


class BM25SectionPrefixTests(unittest.TestCase):
    """The BM25 field drops the document-root heading. Indexing it put the same
    terms on every chunk of a document, flattening BM25 scores so the sparse half
    contributed noise to RRF and outvoted strong dense matches."""

    @staticmethod
    def _prefix(text, sections):
        from backend.indexing.document_loader import DocumentLoader

        return DocumentLoader._apply_bm25_section_prefix(text, sections)

    def test_document_root_heading_is_dropped(self):
        out = self._prefix("Pricing: 250,000 EGP.", ["GS1 Egypt Guide", "Services", "One Trace"])
        self.assertNotIn("GS1 Egypt Guide", out)
        self.assertTrue(out.startswith("Services > One Trace\n"))

    def test_specific_section_names_are_kept(self):
        """They are per-chunk signal: a keyword query for a section name must
        still match that section's body chunks."""
        out = self._prefix("Annual renewal 20,000 EGP.", ["Doc", "Renewal", "Annual Renewal"])
        self.assertIn("Renewal > Annual Renewal", out)

    def test_single_root_section_yields_no_prefix(self):
        self.assertEqual("Body text.", self._prefix("Body text.", ["Doc Title Only"]))

    def test_prefix_skipped_when_heading_already_in_body(self):
        out = self._prefix("One Trace\n\nA traceability system.", ["Doc", "Services", "One Trace"])
        self.assertEqual("One Trace\n\nA traceability system.", out)

    def test_empty_inputs_pass_through(self):
        self.assertEqual("", self._prefix("", ["Doc", "Services"]))
        self.assertEqual("Body.", self._prefix("Body.", []))

    def test_writer_falls_back_to_text_when_chunker_supplies_no_bm25_text(self):
        """The flat fallback splitter carries no section path."""
        import backend.indexing.milvus_writer as module

        events = []

        class _Service:
            def get_embeddings(self, texts):
                return [[0.1, 0.2] for _ in texts]

        captured = {}

        class _Store:
            def init_collection(self, dim):
                pass

            def insert(self, data):
                captured["rows"] = data

        writer = module.MilvusWriter(embedding_service=_Service(), milvus_manager=_Store())
        writer.write_documents([
            {"text": "no sections here", "filename": "f", "file_type": "Word", "chunk_id": "c0"},
        ])
        self.assertEqual("no sections here", captured["rows"][0]["bm25_text"])


if __name__ == "__main__":
    unittest.main()
