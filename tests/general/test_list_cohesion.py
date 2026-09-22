# -*- coding: utf-8 -*-
"""A list is one thing, and a chunk boundary may not run through it.

The failure this exists for, from the shipped corpus: asked which districts the school
bus covers, the answer named five. The other two, Mokattam and Madinaty, were in the next
chunk. Nothing was broken — every district was indexed, and the retrieval eval scored the
question as a hit, because it asks "did the gold span appear in a retrieved chunk" and one
of them had. A metric that counts spans cannot see a list arriving in pieces, and a user
reading five of seven districts has no way to know two are missing.

Measured on the corpus before these rules: 22 of its 36 lists were cut this way, and at
top_k=4 only 27 of 29 came back complete. After them, 29 of 29 at top_k 4 and 8 alike.

## Why the rules are shaped this way

Three of them, and each exists because the previous one was not enough:

  1. carry the list through the parser. Word states membership in the paragraph style and
     the parser discarded it, so seven districts became seven independent units;
  2. do not CLOSE a window mid-list. Not enough on its own — a window already full of
     prose has no room, the hard budget wins, and the list breaks where it would have
     anyway. Traced on the districts: intact through level 1, split at level 2, five
     items and two;
  3. MOVE a list that will not fit to a window of its own. This is what finally fixed the
     districts — and on its own it broke something else, which rule 4 is for;
  4. a list travels with its heading. "The school features:" is a real Heading 3, so rule
     3 moved the list away from it and left five bare items — "Qualified British Staff,
     Exceptional Facilities…" — with nothing saying what they are. That chunk then matched
     nothing and the list vanished from the results entirely: the precise failure keeping
     the list together was supposed to prevent.

The hard budget still wins over all of them. A list longer than a whole window has to
break somewhere, and an unbounded chunk is the thing every other rule in the chunker
exists to prevent.
"""
import unittest

from backend.indexing.document_loader import DocumentLoader


def unit(text, list_group=0, kind="text", sections=("Doc",)):
    return {"kind": kind, "text": text, "sections": sections, "page": 0,
            "asset_ids": (), "list_group": list_group}


def texts(windows):
    return [[u["text"] for u in window] for window in windows]


class AListIsNotCutInHalfTests(unittest.TestCase):
    def test_a_list_stays_in_one_window(self):
        units = [unit("lead in:", 1)] + [unit(f"item {i}", 1) for i in range(6)]
        windows = DocumentLoader._pack_units(units, budget=400, target=20)

        self.assertEqual(1, len(windows), "a target that small would cut every item apart")
        self.assertEqual(7, len(windows[0]))

    def test_prose_around_a_list_still_packs_by_size(self):
        """The rule is about lists, not about abandoning the budget everywhere."""
        units = [unit("a" * 30), unit("b" * 30), unit("c" * 30)]
        windows = DocumentLoader._pack_units(units, budget=400, target=20)

        self.assertEqual(3, len(windows))

    def test_a_list_that_will_not_fit_moves_whole_to_the_next_window(self):
        """The districts, exactly. Keeping a list together is not enough when the window
        it lands in is already full — it has to start a new one."""
        units = [unit("x" * 180)] + [unit("lead in:", 1)] + [unit(f"item {i}", 1) for i in range(6)]
        windows = DocumentLoader._pack_units(units, budget=200, target=200)

        self.assertEqual(2, len(windows))
        self.assertEqual(["x" * 180], texts(windows)[0])
        self.assertEqual(7, len(windows[1]), "the whole list moved, not part of it")

    def test_a_list_travels_with_its_heading(self):
        """"The school features:" is a Heading 3. Moving the list without it left five
        bare items that matched nothing and disappeared from the results."""
        units = [unit("x" * 180)] + [unit("The school features:", 1)] + [
            unit(f"feature {i}", 1) for i in range(5)
        ]
        windows = DocumentLoader._pack_units(units, budget=200, target=200)

        with_list = next(w for w in windows if any(u["text"] == "feature 0" for u in w))
        self.assertEqual("The school features:", with_list[0]["text"])

    def test_two_small_lists_may_share_a_window_but_neither_is_broken(self):
        """The guarantee is that one list is never cut, NOT that every list gets a
        window to itself. Two short lists that both fit pack together, exactly as two
        short paragraphs would — the packer has never isolated prose either, and making
        lists special in that direction is behaviour nothing has measured.
        """
        units = [unit(f"a{i}", 1) for i in range(3)] + [unit(f"b{i}", 2) for i in range(3)]
        windows = DocumentLoader._pack_units(units, budget=400, target=20)

        for group, members in ((1, ["a0", "a1", "a2"]), (2, ["b0", "b1", "b2"])):
            holding = [w for w in windows if any(u["list_group"] == group for u in w)]
            self.assertEqual(1, len(holding), f"list {group} was split across windows")
            self.assertEqual(members, [u["text"] for u in holding[0] if u["list_group"] == group])

    def test_a_list_longer_than_a_window_still_breaks(self):
        """The hard budget is not negotiable. An unbounded chunk is what every other rule
        in the chunker exists to prevent, and a list does not get an exemption."""
        units = [unit("y" * 90, 1) for _ in range(6)]
        windows = DocumentLoader._pack_units(units, budget=200, target=200)

        self.assertGreater(len(windows), 1)
        for window in windows:
            self.assertLessEqual(sum(len(u["text"]) for u in window), 200)

    def test_units_carrying_no_list_are_unaffected(self):
        units = [unit("a" * 30), unit("b" * 30)]
        self.assertEqual(2, len(DocumentLoader._pack_units(units, budget=400, target=20)))

    def test_cohesion_can_be_switched_off(self):
        """Kept switchable because that is how the two builds were compared at all."""
        from unittest.mock import patch

        units = [unit("lead in:", 1)] + [unit(f"item {i}", 1) for i in range(6)]
        with patch.dict("os.environ", {"CHUNK_LIST_COHESION": "0"}):
            windows = DocumentLoader._pack_units(units, budget=400, target=20)
        self.assertGreater(len(windows), 1, "off means the old size-only packing")


class TheCorpusItselfTests(unittest.TestCase):
    """Against the shipped document rather than a fixture, because the rules were
    written from what its authors actually did with Word."""

    CORPUS = "data/documents/Aurexis_Knowledge_Base_Mock_Egypt.docx"

    @classmethod
    def setUpClass(cls):
        import os

        if not os.path.exists(cls.CORPUS):
            raise unittest.SkipTest("the shipped corpus is not present")
        from backend.indexing.docx_layout import parse_docx_blocks

        cls.blocks = parse_docx_blocks(cls.CORPUS)

    def test_word_list_items_arrive_carrying_their_list(self):
        districts = [b for b in self.blocks if (b.get("content") or "") in
                     ("Fifth Settlement", "New Cairo", "Mokattam", "Madinaty")]
        self.assertEqual(4, len(districts))
        groups = {b.get("list_group") for b in districts}
        self.assertEqual(1, len(groups), "the districts are one list")
        self.assertTrue(all(groups), "and it is not group zero")

    def test_a_lead_in_is_pulled_into_the_list_it_introduces(self):
        lead = next(b for b in self.blocks
                    if (b.get("content") or "") == "Currently, covered districts:")
        district = next(b for b in self.blocks
                        if (b.get("content") or "") == "Fifth Settlement")
        self.assertEqual(district["list_group"], lead.get("list_group"))

    def test_a_heading_is_pulled_in_too(self):
        heading = next(b for b in self.blocks
                       if (b.get("content") or "") == "The school features:")
        item = next(b for b in self.blocks
                    if (b.get("content") or "") == "Qualified British Staff")
        self.assertEqual("heading", heading["type"], "it is a real Heading 3")
        self.assertEqual(item["list_group"], heading.get("list_group"))

    def test_every_district_ends_up_in_one_chunk(self):
        loader = DocumentLoader()
        docs = loader.load_document(self.CORPUS, "kb.docx")
        leaves = [d for d in docs if d["chunk_level"] == 3]
        districts = ["Fifth Settlement", "New Cairo", "Nasr City", "Heliopolis",
                     "Maadi", "Mokattam", "Madinaty"]
        homes = {
            next(i for i, leaf in enumerate(leaves) if name in leaf["text"])
            for name in districts
            if any(name in leaf["text"] for leaf in leaves)
        }
        self.assertEqual(1, len(homes), "the answer to 'which districts' is one chunk")


if __name__ == "__main__":
    unittest.main()
