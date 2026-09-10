"""A picture goes WHERE the answer put it, and an invented marker costs nothing.

The model is shown `[FIGURE 1]` on a chunk header and asked to write the same marker at
the point in its answer where the picture belongs. Finalize turns the markers it wrote
into anchors, and the frontend renders each image at its anchor — so the figure lands
inside the sentence that describes it instead of as a card underneath the whole answer.

Three properties, one class each:

  * THE NUMBER IS THE HANDLE. The model is given a turn-local number and never an
    asset_id, because an id in the prompt is an id in the answer.
  * AN INVENTED MARKER COSTS NOTHING. It is deleted, silently. This is the whole lesson
    of the grounding layer this feature replaced: it withheld answers whose citations it
    could not verify, and a correct answer withdrawn over a marker is a far worse outcome
    than a missing picture.
  * THE ANCHOR SELECTS THE PICTURE. It outranks the `[n]` citation path, which stays
    intact underneath it, and it never leaves the reader's message for the model's
    history.
"""
import unittest


class _Ctx:
    """The two attributes finalize reads. `figure_numbers` is a plain dict here for the
    same reason `answer_blocks` is a list in test_answer_blocks: the property on the real
    context is a copy under a lock, and neither the lock nor the copy is what is on
    trial."""

    def __init__(self, figure_numbers=None, answer_blocks=()):
        self.figure_numbers = dict(figure_numbers or {})
        self.answer_blocks = list(answer_blocks)


class TheNumberIsTheHandle(unittest.TestCase):
    """The model is handed a number. It is never handed an id."""

    def test_each_asset_gets_its_own_number_in_retrieval_order(self):
        from backend.tools.knowledge import _figure_markers

        per_chunk, mapping = _figure_markers([
            {"asset_ids": ["a.pdf::p1::img0"]},
            {"text": "no picture here"},
            {"asset_ids": ["b.pdf::p3::img1"]},
        ])
        self.assertEqual([[1], [], [2]], per_chunk)
        self.assertEqual({1: "a.pdf::p1::img0", 2: "b.pdf::p3::img1"}, mapping)

    def test_a_chunk_with_two_pictures_can_be_pointed_at_individually(self):
        """`auto_merge_figure_threshold: null` keeps figure groups unmerged so this is
        rare, but a chunk that does carry two images must not force the answer to name
        both or neither."""
        from backend.tools.knowledge import _figure_markers

        per_chunk, mapping = _figure_markers([{"asset_ids": ["k::p1::img0", "k::p1::img1"]}])
        self.assertEqual([[1, 2]], per_chunk)
        self.assertEqual("k::p1::img1", mapping[2])

    def test_an_empty_asset_id_is_not_given_a_number(self):
        """A number that resolves to nothing would render as a deleted marker — the
        model told there is a picture, and no picture."""
        from backend.tools.knowledge import _figure_markers

        per_chunk, mapping = _figure_markers([{"asset_ids": ["", None, "real::p1::img0"]}])
        self.assertEqual([[1]], per_chunk)
        self.assertEqual({1: "real::p1::img0"}, mapping)

    def test_the_numbering_never_depends_on_the_asset_id(self):
        """Turn-local, so it does not inherit `build_asset_id`'s positional instability:
        one image added to page 2 shifts every later id, and a document-global figure
        number would have shifted with it."""
        from backend.tools.knowledge import _figure_markers

        _, before = _figure_markers([{"asset_ids": ["kb.pdf::p9::img4"]}])
        _, after = _figure_markers([{"asset_ids": ["kb.pdf::p9::img5"]}])
        self.assertEqual([1], list(before))
        self.assertEqual(list(before), list(after))


class AnInventedMarkerCostsNothing(unittest.TestCase):
    """The rule that makes this feature safe to ship at all."""

    def test_a_known_marker_becomes_an_anchor(self):
        from backend.chat.service import _resolve_figure_markers

        out = _resolve_figure_markers(
            "الزي الصيفي كالتالي [FIGURE 1] وبيتغير في الشتاء.",
            _Ctx({1: "kb.pdf::p2::img0"}),
        )
        self.assertIn("<!--figure:kb.pdf::p2::img0-->", out)
        self.assertNotIn("[FIGURE 1]", out)

    def test_an_unknown_number_is_deleted_and_the_answer_survives(self):
        """The load-bearing test. A model that writes [FIGURE 9] on a turn that
        retrieved one figure must not cost the parent their answer."""
        from backend.chat.service import _resolve_figure_markers

        out = _resolve_figure_markers(
            "المصروفات 12,000 جنيه [FIGURE 9] للعام الدراسي.",
            _Ctx({1: "kb.pdf::p2::img0"}),
        )
        self.assertNotIn("FIGURE", out)
        self.assertNotIn("figure:", out)
        self.assertIn("المصروفات 12,000 جنيه", out)
        self.assertIn("للعام الدراسي.", out)

    def test_deleting_a_marker_does_not_leave_a_double_space(self):
        from backend.chat.service import _resolve_figure_markers

        out = _resolve_figure_markers("قبل [FIGURE 4] بعد", _Ctx({1: "a::p1::img0"}))
        self.assertEqual("قبل بعد", out)

    def test_arabic_indic_digits_name_the_same_figure(self):
        """The corpus is Arabic and so is the prose. A parser that only knew ASCII would
        have dropped most real markers and shown no picture at all."""
        from backend.chat.service import _resolve_figure_markers

        for marker in ("[FIGURE ٢]", "[الشكل ٢]", "[شكل 2]", "[figure 2]"):
            with self.subTest(marker=marker):
                out = _resolve_figure_markers(
                    f"انظر {marker} هنا", _Ctx({2: "kb.pdf::p5::img1"})
                )
                self.assertIn("<!--figure:kb.pdf::p5::img1-->", out)

    def test_an_answer_with_no_marker_is_untouched(self):
        from backend.chat.service import _resolve_figure_markers

        answer = "المصروفات 12,000 جنيه للعام الدراسي [1]."
        self.assertEqual(answer, _resolve_figure_markers(answer, _Ctx({1: "a::p1::img0"})))

    def test_a_turn_that_retrieved_no_figure_still_drops_the_marker(self):
        from backend.chat.service import _resolve_figure_markers

        out = _resolve_figure_markers("انظر [FIGURE 1] هنا", _Ctx({}))
        self.assertEqual("انظر هنا", out)


class TheAnchorSelectsThePicture(unittest.TestCase):
    """Ahead of the citation path, which is left exactly as it was underneath."""

    CONFIG = type("Delivery", (), {"attach_only_cited": True, "attach_to_response": True})()

    class _TurnCtx:
        def __init__(self, surfaced):
            self._surfaced = list(surfaced)

        def surfaced_asset_ids(self):
            return list(self._surfaced)

    def test_the_anchored_asset_is_the_one_attached(self):
        from backend.chat.assets_bridge import asset_ids_for_answer

        ids = asset_ids_for_answer(
            "الزي الصيفي <!--figure:kb.pdf::p2::img1--> كالتالي",
            self._TurnCtx(["kb.pdf::p2::img0", "kb.pdf::p2::img1"]),
            {"retrieved_chunks": [{"asset_ids": ["kb.pdf::p2::img0"]}]},
            self.CONFIG,
        )
        self.assertEqual(["kb.pdf::p2::img1"], ids)

    def test_an_anchor_the_turn_did_not_surface_is_ignored(self):
        """An anchor is text in a message and a message can be replayed. Without the
        intersection, an old answer could name an asset this turn never retrieved and the
        turn would go and fetch it."""
        from backend.chat.assets_bridge import asset_ids_for_answer

        ids = asset_ids_for_answer(
            "<!--figure:other.pdf::p1::img0-->",
            self._TurnCtx(["kb.pdf::p2::img0"]),
            {"retrieved_chunks": [{"asset_ids": ["kb.pdf::p2::img0"]}]},
            self.CONFIG,
        )
        self.assertEqual(["kb.pdf::p2::img0"], ids)

    def test_with_no_anchor_the_citation_path_still_decides(self):
        from backend.chat.assets_bridge import asset_ids_for_answer

        ids = asset_ids_for_answer(
            "كما في [2]",
            self._TurnCtx(["a::p1::img0", "b::p1::img0"]),
            {"retrieved_chunks": [
                {"asset_ids": ["a::p1::img0"]},
                {"asset_ids": ["b::p1::img0"]},
            ]},
            self.CONFIG,
        )
        self.assertEqual(["b::p1::img0"], ids)

    def test_anchors_are_read_in_the_order_they_appear(self):
        from backend.chat.assets_bridge import anchored_asset_ids

        self.assertEqual(
            ["second::p1::img0", "first::p1::img0"],
            anchored_asset_ids(
                "أ <!--figure:second::p1::img0--> ب <!--figure:first::p1::img0-->"
            ),
        )

    def test_an_anchor_is_invisible_to_a_reader(self):
        """An HTML comment, and only because the frontend drops raw HTML outright
        (`renderer.html = () => ''`). Same reason BLOCK_MARKER is one, and the same
        payoff: a frontend that predates this renders clean prose."""
        from backend.chat.service import _resolve_figure_markers

        out = _resolve_figure_markers("انظر [FIGURE 1]", _Ctx({1: "a::p1::img0"}))
        anchor = out[out.index("<!--"):]
        self.assertTrue(anchor.startswith("<!--"))
        self.assertTrue(anchor.endswith("-->"))

    def test_the_anchor_stays_out_of_the_models_history(self):
        """The anchor carries an asset_id, and an id in the model's history is an id in
        its next answer — shown one, a small model writes it back as an image link that
        cannot load. The reader keeps the picture; the model reads the sentence."""
        from backend.chat.service import _resolve_figure_markers, strip_answer_blocks

        stored = _resolve_figure_markers(
            "الزي الصيفي [FIGURE 1] كالتالي", _Ctx({1: "kb.pdf::p2::img0"})
        )
        seen = strip_answer_blocks(stored)
        self.assertNotIn("figure:", seen)
        self.assertNotIn("kb.pdf::p2::img0", seen)
        self.assertIn("الزي الصيفي", seen)
        self.assertIn("كالتالي", seen)


if __name__ == "__main__":
    unittest.main()
