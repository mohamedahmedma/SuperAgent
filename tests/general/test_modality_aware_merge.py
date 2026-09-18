"""Auto-merge is modality aware, and bounded by where the match actually is.

Merging gives a text hit its surrounding context, which is worth having. For figures
it costs something specific: two figure leaves collapse into one parent, so an answer
citing `[1]` points at several images and the reader has to work out which was meant.
Text keeps merging; figures stay individually citable.

It also costs SIZE, which is what the second half of this file is about. Merging is the
one thing that can hand back a chunk bigger than the chunking budgets produced, and
nothing between it and the model trims anything — so once the grader reads chunks whole,
the whole parent is the difference between a prompt sized by the retrieval and a prompt
sized by the corpus. It now returns the text around the match instead.
"""
import unittest

from backend.rag.utils import (
    EVIDENCE_WINDOW_CHARS,
    _is_figure_chunk,
    _merge_to_parent_level,
    _match_spans,
    _parent_window,
)


def chunk(chunk_id, parent, modality="text", assets=None, level=3):
    return {
        "chunk_id": chunk_id,
        "parent_chunk_id": parent,
        "chunk_level": level,
        "modality": modality,
        "asset_ids": assets or [],
        "text": f"body of {chunk_id}",
        "score": 0.5,
    }


class FigureDetectionTests(unittest.TestCase):
    def test_modality_marks_a_figure(self):
        self.assertTrue(_is_figure_chunk(chunk("c1", "p1", modality="figure")))

    def test_asset_ids_alone_mark_a_figure(self):
        """Chunks written before the modality field existed still carry asset_ids."""
        self.assertTrue(_is_figure_chunk({"asset_ids": ["a1"]}))

    def test_plain_text_is_not_a_figure(self):
        self.assertFalse(_is_figure_chunk(chunk("c1", "p1")))
        self.assertFalse(_is_figure_chunk({}))
        self.assertFalse(_is_figure_chunk({"modality": "table", "asset_ids": []}))


class MergeBehaviourTests(unittest.TestCase):
    """_merge_to_parent_level only merges groups whose parent it can fetch; with no
    parent store available nothing is replaced, so these assert on which groups were
    SELECTED by counting the merges reported."""

    def _merge(self, docs, **kwargs):
        from unittest.mock import MagicMock

        from backend.composition import Services, set_default_services

        # Stand in for the Postgres parent lookup: every requested parent exists.
        def fake_parents(ids):
            return [{"chunk_id": pid, "text": f"parent {pid}", "chunk_level": 2,
                     "modality": "text", "asset_ids": []} for pid in ids]

        store = MagicMock()
        store.get_documents_by_ids.side_effect = fake_parents
        # The merge resolves its parent store from the process container, so that is
        # where the stand-in goes.
        set_default_services(Services(parent_chunks=store))
        self.addCleanup(set_default_services, None)
        return _merge_to_parent_level(docs, **kwargs)

    def test_text_siblings_still_merge_at_the_normal_threshold(self):
        docs = [chunk("c1", "p1"), chunk("c2", "p1")]
        merged, count = self._merge(docs, threshold=2)
        self.assertEqual(2, count)
        self.assertEqual(1, len(merged))
        self.assertEqual("p1", merged[0]["chunk_id"])

    def test_a_figure_group_is_left_unmerged_by_default(self):
        """The behaviour that makes one citation point at one image."""
        docs = [
            chunk("c1", "p1", modality="figure", assets=["img5"]),
            chunk("c2", "p1", modality="figure", assets=["img6"]),
        ]
        merged, count = self._merge(docs, threshold=2, figure_threshold=None)
        self.assertEqual(0, count)
        self.assertEqual(["c1", "c2"], [d["chunk_id"] for d in merged])
        self.assertEqual([["img5"], ["img6"]], [d["asset_ids"] for d in merged])

    def test_one_figure_among_text_siblings_protects_the_whole_group(self):
        """Merging would swallow the figure into a parent carrying both images."""
        docs = [chunk("c1", "p1"), chunk("c2", "p1", modality="figure", assets=["img1"])]
        merged, count = self._merge(docs, threshold=2, figure_threshold=None)
        self.assertEqual(0, count)
        self.assertEqual(2, len(merged))

    def test_text_and_figure_groups_are_decided_independently(self):
        docs = [
            chunk("t1", "p_text"), chunk("t2", "p_text"),
            chunk("f1", "p_fig", modality="figure", assets=["img1"]),
            chunk("f2", "p_fig", modality="figure", assets=["img2"]),
        ]
        merged, count = self._merge(docs, threshold=2, figure_threshold=None)
        self.assertEqual(2, count)  # only the text group
        ids = [d["chunk_id"] for d in merged]
        self.assertIn("p_text", ids)
        self.assertIn("f1", ids)
        self.assertIn("f2", ids)

    def test_a_figure_threshold_lets_a_domain_opt_back_in(self):
        """A section with many figures can still be collapsed if a domain prefers it."""
        docs = [chunk(f"c{i}", "p1", modality="figure", assets=[f"img{i}"]) for i in range(3)]
        merged, count = self._merge(docs, threshold=2, figure_threshold=3)
        self.assertEqual(3, count)
        self.assertEqual(["p1"], [d["chunk_id"] for d in merged])

    def test_a_figure_group_below_its_threshold_stays_split(self):
        docs = [chunk(f"c{i}", "p1", modality="figure", assets=[f"img{i}"]) for i in range(2)]
        merged, count = self._merge(docs, threshold=2, figure_threshold=3)
        self.assertEqual(0, count)
        self.assertEqual(2, len(merged))

    def test_a_lone_child_never_merges(self):
        merged, count = self._merge([chunk("c1", "p1")], threshold=2)
        self.assertEqual(0, count)

    def test_chunks_without_a_parent_are_passed_through(self):
        merged, count = self._merge([chunk("c1", "")], threshold=2)
        self.assertEqual(0, count)
        self.assertEqual(["c1"], [d["chunk_id"] for d in merged])


class LocatingTheMatchInsideItsParentTests(unittest.TestCase):
    """A leaf is NOT reliably a substring of its own parent, which is why locating one
    takes more than a `find`.

    `_apply_section_prefix` prepends a synthesised path ("Fees > Payment") that the
    parent normally does not contain, because the parent holds those headings as
    separate block lines. Measured over the whole school corpus, 211 child/parent pairs:
    the block locates 51.7% exactly, the block minus its first line a further 47.9%, the
    line strategy the last 0.5%, and nothing was left unlocated. The middle one carries
    nearly half the corpus, so it is not a fallback that never runs."""

    def test_an_exact_child_is_located(self):
        parent = "Fees are payable each term.\nYear 3 costs 105,000 EGP.\nPay by the 1st."
        start, end = _match_spans(parent, "Year 3 costs 105,000 EGP.", 600)[0]
        self.assertEqual("Year 3 costs 105,000 EGP.", parent[start:end])

    def test_a_child_carrying_a_section_prefix_is_still_located(self):
        """The 47% case: the leaf begins with a path the parent never contains."""
        parent = "Admissions\n\nFees are payable each term.\nYear 3 costs 105,000 EGP."
        child = "Aurexis > 1. ADMISSION > Fees\nYear 3 costs 105,000 EGP."
        span = _match_spans(parent, child, 600)[0]
        self.assertIsNotNone(span)
        self.assertEqual("Year 3 costs 105,000 EGP.", parent[span[0]:span[1]])

    def test_a_child_that_shares_only_its_longest_line_is_located(self):
        parent = "Heading\nYear 3 costs 105,000 EGP for Egyptian pupils.\nMore prose."
        child = "Prefix\nYear 3 costs 105,000 EGP for Egyptian pupils.\nTrailing text"
        span = _match_spans(parent, child, 600)[0]
        self.assertIsNotNone(span)
        self.assertIn("105,000 EGP", parent[span[0]:span[1]])

    def test_a_child_from_a_different_document_is_not_located(self):
        self.assertEqual([], _match_spans("Fees are payable each term.", "Uniform is navy.", 600))

    def test_a_repeated_header_is_never_the_anchor(self):
        """The failure this rule exists for, and it is silent rather than loud.

        `_split_table_row_groups` re-emits the header row in every group and
        `_figure_passages` repeats the caption on every passage — so the lines the
        locator falls back to are exactly the ones that appear many times, and they are
        usually LONGER than the data rows beneath them. Taking the first occurrence
        anchored a child holding rows 35-40 of a fee table at offset 0, and the window
        came back holding rows 1-34: the parent's text, none of the child's, and a fee
        for the wrong year group. Verified end to end before the uniqueness rule."""
        header = "Year Group | Tuition Egyptian | Tuition International | Bus | Books"
        rows = [f"Y{i:02d} | {90 + i},000 EGP | {100 + i},000 EGP | 8,000 | 2,000"
                for i in range(60)]
        parent = "\n".join([header] + rows)
        child = "\n".join(["Fees > Schedule", header] + rows[35:40])

        start, end = _match_spans(parent, child, 600)[0]
        self.assertIn("Y35", parent[start:end] + parent[end:end + 200])
        self.assertNotEqual(0, start, "anchored on the header repeated in every group")

        window = _parent_window(parent, [{"text": child}], 600)
        self.assertIn("Y35 |", window)
        self.assertIn("Y37 |", window)

    def test_an_ambiguous_child_is_unlocatable_rather_than_guessed(self):
        """Ambiguity means unlocatable, not "pick one": the merge is then skipped and the
        children that actually matched are what the turn keeps."""
        parent = "same line\n" + "\n".join(["repeated row"] * 40)
        self.assertEqual([], _match_spans(parent, "repeated row", 600))


class TheMergeWindowTests(unittest.TestCase):
    def test_a_parent_within_the_budget_comes_back_untouched(self):
        """The common case, and byte-identical to promoting the whole parent."""
        parent = "\n".join(f"line {i}" for i in range(10))
        self.assertEqual(parent, _parent_window(parent, [{"text": "line 5"}], 1000))

    def test_an_oversized_parent_is_windowed_around_the_match(self):
        parent = "\n".join(f"line {i:03d} " + "x" * 90 for i in range(100))
        window = _parent_window(parent, [{"text": "line 050 " + "x" * 90}], 400)

        self.assertLessEqual(len(window), 400)
        self.assertIn("line 050", window, "the match itself must survive its own window")
        self.assertNotIn("line 000", window)
        self.assertNotIn("line 099", window)

    def test_the_window_grows_outward_so_the_match_keeps_its_context(self):
        """What merging is FOR. The match alone is the leaf the retrieval already had."""
        parent = "\n".join(f"line {i:03d}" for i in range(200))
        window = _parent_window(parent, [{"text": "line 100"}], 200)

        self.assertIn("line 099", window)
        self.assertIn("line 101", window)

    def test_two_matches_are_both_inside_the_window(self):
        parent = "\n".join(f"line {i:03d}" for i in range(200))
        window = _parent_window(
            parent, [{"text": "line 100"}, {"text": "line 104"}], 300
        )
        self.assertIn("line 100", window)
        self.assertIn("line 104", window)

    def test_a_single_line_wider_than_the_budget_is_cut(self):
        """A window of whole lines is not a bound on its own: `_render_rows` makes one
        table row one line, so one row can be wider than the whole window."""
        parent = "short\n" + "y" * 5000 + "\nshort"
        window = _parent_window(parent, [{"text": "y" * 5000}], 400)
        self.assertLessEqual(len(window), 400)

    def test_an_unlocatable_match_refuses_to_merge(self):
        """Not "promote the whole parent": a merge that cannot find its own match cannot
        claim to be sending the text around it, and the children it would replace are
        the passages that matched and are already leaf-sized."""
        parent = "\n".join(f"line {i:03d}" for i in range(400))
        self.assertIsNone(_parent_window(parent, [{"text": "nothing like this"}], 400))


class MergeKeepsItsPromisesTests(unittest.TestCase):
    """The doc-level consequences of windowing, through `_merge_to_parent_level`."""

    def _merge(self, docs, parent_text, **kwargs):
        from unittest.mock import MagicMock

        from backend.composition import Services, set_default_services

        def fake_parents(ids):
            return [{"chunk_id": pid, "text": parent_text, "chunk_level": 2,
                     "modality": "figure", "asset_ids": ["img1"]} for pid in ids]

        store = MagicMock()
        store.get_documents_by_ids.side_effect = fake_parents
        set_default_services(Services(parent_chunks=store))
        self.addCleanup(set_default_services, None)
        return _merge_to_parent_level(docs, **kwargs)

    def test_a_windowed_parent_that_lost_its_figure_stops_claiming_one(self):
        """The model is shown `[FIGURE n]` for each asset a chunk carries. A parent whose
        figure text was windowed out would otherwise attach a picture to an answer
        written from two paragraphs that never mention it — and splitting images makes
        this common, because one image now spans several parents."""
        parent_text = (
            "[Figure] Uniform guide\n" + "\n".join(f"figure line {i}" for i in range(80))
            + "\n" + "\n".join(f"prose line {i:03d} " + "z" * 60 for i in range(60))
        )
        docs = [chunk("c1", "p1"), chunk("c2", "p1")]
        docs[0]["text"] = "prose line 030 " + "z" * 60
        docs[1]["text"] = "prose line 031 " + "z" * 60
        merged, count = self._merge(docs, parent_text, threshold=2, window_chars=400)

        self.assertEqual(1, len(merged))
        self.assertNotIn("[Figure]", merged[0]["text"])
        self.assertEqual([], merged[0]["asset_ids"])
        self.assertEqual("text", merged[0]["modality"])

    def test_a_windowed_parent_that_kept_its_figure_keeps_the_asset(self):
        parent_text = (
            "[Figure] Uniform guide\n" + "\n".join(f"figure line {i}" for i in range(80))
            + "\n" + "\n".join(f"prose line {i:03d} " + "z" * 60 for i in range(60))
        )
        docs = [chunk("c1", "p1"), chunk("c2", "p1")]
        docs[0]["text"] = "figure line 2"
        docs[1]["text"] = "figure line 3"
        merged, _ = self._merge(docs, parent_text, threshold=2, window_chars=400)

        self.assertIn("[Figure]", merged[0]["text"])
        self.assertEqual(["img1"], merged[0]["asset_ids"])

    def test_an_unlocatable_group_keeps_its_children_instead(self):
        parent_text = "\n".join(f"unrelated line {i:03d} " + "q" * 60 for i in range(80))
        docs = [chunk("c1", "p1"), chunk("c2", "p1")]
        merged, count = self._merge(docs, parent_text, threshold=2, window_chars=400)

        self.assertEqual(0, count)
        self.assertEqual(["c1", "c2"], [d["chunk_id"] for d in merged])

    def test_the_window_sits_above_what_correct_chunking_can_produce(self):
        """A ceiling, not a target, and the drift guard the schema comment points at.

        Trimming merged parents below the chunking budget was measured and reversed: at
        1,600 the retrieval eval lost 13 of 176 questions at the RECALL stage, because
        the fact a question needed was in the part of the parent the window cut. A parent
        is bounded by `l1_size` anyway, for as long as every unit is — so this is set just
        above that, fires only on an index built before figures were split, and is a
        no-op on a healthy one.

        Derived from the profile rather than written twice: widen `chunking.l1_size` and
        this must widen with it, or merging starts trimming parents that were never too
        big."""
        from backend.profiles.registry import load_profile

        profile = load_profile("base")
        chunking = profile.chunking
        l1_size = chunking.l1_size or max(2000, chunking.chunk_size * 3)
        # `_apply_section_prefix` runs AFTER packing, so a chunk is its budget plus the
        # prefix: 150 characters and the newline joining it on.
        self.assertGreaterEqual(profile.retrieval.evidence_window_chars, l1_size + 151)
        self.assertGreaterEqual(EVIDENCE_WINDOW_CHARS, l1_size + 151)


class ProfileWiringTests(unittest.TestCase):
    def test_the_default_keeps_figures_separate(self):
        from backend.profiles.registry import load_profile

        self.assertIsNone(load_profile("base").retrieval.auto_merge_figure_threshold)
        self.assertEqual(2, load_profile("base").retrieval.auto_merge_threshold)

    def test_the_threshold_is_reported_in_the_retrieval_trace(self):
        """So a diagnostic can explain why a figure group did or did not merge."""
        from backend.rag.utils import RETRIEVAL_TRACE_FIELDS

        self.assertIn("auto_merge_figure_threshold", RETRIEVAL_TRACE_FIELDS)


if __name__ == "__main__":
    unittest.main()
