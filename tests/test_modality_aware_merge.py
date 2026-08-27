"""Auto-merge is modality aware.

Merging gives a text hit its surrounding context, which is worth having. For figures
it costs something specific: two figure leaves collapse into one parent, so an answer
citing `[1]` points at several images and the reader has to work out which was meant.
Text keeps merging; figures stay individually citable.
"""
import unittest

from backend.rag.utils import _is_figure_chunk, _merge_to_parent_level


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
        from unittest.mock import patch

        # Stand in for the Postgres parent lookup: every requested parent exists.
        def fake_parents(ids):
            return [{"chunk_id": pid, "text": f"parent {pid}", "chunk_level": 2,
                     "modality": "text", "asset_ids": []} for pid in ids]

        with patch("backend.rag.utils._parent_chunk_store") as store:
            store.get_documents_by_ids.side_effect = fake_parents
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
