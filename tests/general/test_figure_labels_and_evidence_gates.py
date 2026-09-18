# -*- coding: utf-8 -*-
"""Four gates one question had to pass, and failed at every one.

A parent asked what their daughter's school uniform is. The child is in Year 12, the
corpus holds uniform figures for Reception, for Years 1-6 and for Secondary, and the
answer was the canned "the school's material does not have this". Every stage was
separately wrong, and each would have hidden the next:

  * the figures for Secondary were captioned as the primary ones, because the document
    prints each label UNDER its picture and the extractor was told to prefer the
    document's wording over the title printed in the image;
  * the caption rule could not be corrected on its own, because extractions are cached
    on (sha256, profile, dossier_version) and the images had not changed;
  * the figure surrogates were being split, and the tail — `Tags:` plus `Answers:` and
    nothing else — retrieved BETTER than the figure, because it is made of question
    phrasings;
  * the grader then read year-labelled material that did not name Year 12 as being about
    a different subject, which is the only verdict that ends a turn in a denial;
  * and had it not, the answer prompt's narrowing branch told the model to answer for
    Year 12 alone, out of material that has no Year 12 — an empty set, and a denial from
    the one stage that had the chunks in front of it.

The last class covers a different bug with the same cause as the first: `name_key` folds
ى onto ي, which is what makes «ليلي» find «ليلى» and also what makes the preposition على
indistinguishable from the name علي.
"""
import re
import unittest
from pathlib import Path

from backend.assets.dossier import DOSSIER_VERSION, MIGRATIONS
from backend.chat.signals import _names_the_child
from backend.indexing.document_loader import DocumentLoader
from backend.prompts import render as render_prompt

TEMPLATES = Path("backend/prompts/templates")


def _template(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


class AFigureIsDividedButNeverOrphaned(unittest.TestCase):
    """A figure's text is divided into leaf-sized passages, and no passage is anonymous.

    This class used to assert the opposite — that a surrogate is ONE indivisible unit —
    and the reason it gave is still true and still enforced here. `render_surrogate`
    writes caption, description and transcription first and `Tags:`/`Answers:` last, so
    cutting it at whatever character a budget falls on leaves a tail with no picture, no
    caption and no description. That tail out-retrieves the figure it came from, because
    a question matches four question phrasings more closely than it matches an
    infographic's contents.

    Indivisible was one way to prevent that, and it cost the other half: a unit larger
    than a level's budget is packed alone and unshortened, so one image could be one
    chunk of any size, and the cap that held it back destroyed evidence at indexing.
    `_figure_passages` divides by FIELD instead of by character — the header on every
    passage, `Tags:`/`Answers:` kept with the description they belong to — so every piece
    is identified and every piece is bounded.
    """

    def setUp(self):
        self.loader = DocumentLoader()

    def _figure_unit(self, text: str) -> dict:
        return {"kind": "figure", "text": text, "sections": ("Handbook", "Uniform"),
                "page": 0, "asset_ids": ("kb.docx::p0::imgabc123456789",)}

    def _parts(self, transcription: str = "") -> dict:
        return {
            "header": "[Figure] Day Wear: Secondary School - Girls",
            "description": "An infographic illustrating the school uniform. " * 30,
            "transcription": transcription or (
                "White Shirt (long sleeves) Navy Blue Blazer Navy Blue Pleated Skirt " * 10
            ),
            "summary": (
                "Tags: School Uniform, Girls, Secondary, Day Wear, Blazer\n"
                "Answers: What is the uniform for secondary girls? What colour is the blazer?"
            ),
        }

    def _blocks(self, parts: dict, asset_id: str = "kb.docx::p0::imgabc123456789") -> list:
        return [
            {"type": "heading", "content": "School uniform", "level": 3, "page_number": 0},
            {"type": "text", "content": "the joined surrogate", "page_number": 0,
             "asset_ids": [asset_id], "figure": parts},
        ]

    def _figure_units(self, parts: dict) -> list:
        return [unit for unit in self.loader._blocks_to_units(self._blocks(parts))
                if unit["kind"] == "figure"]

    def test_every_passage_names_the_picture(self):
        """The orphan tail, stated as the rule that prevents it."""
        for unit in self._figure_units(self._parts()):
            self.assertTrue(
                unit["text"].startswith("[Figure] Day Wear: Secondary School - Girls"),
                f"a passage with no caption was produced: {unit['text'][:80]!r}",
            )

    def test_the_question_phrasings_never_stand_alone(self):
        """`Answers:` is what made the tail win: it is made of question phrasings, so a
        question matches it closely and nothing else in the piece dilutes the match. It
        stays in the passage that also carries the description — the one whose job is
        already to be matched against a question."""
        answering = [unit["text"] for unit in self._figure_units(self._parts())
                     if "Answers:" in unit["text"]]
        self.assertEqual(1, len(answering))
        self.assertIn("An infographic illustrating", answering[0])

    def test_a_passage_is_never_re_split_by_refinement(self):
        """`_refine_units` re-splits what a coarser level produced. A figure passage is
        already leaf-sized and already headed, so re-splitting could only undo both."""
        passage = self._figure_units(self._parts())[0]["text"]
        refined = self.loader._refine_units(
            [self._figure_unit(passage)], self.loader._splitter_level_3, 600
        )
        self.assertEqual(1, len(refined))
        self.assertEqual(passage, refined[0]["text"])

    def test_every_passage_keeps_the_asset_reference(self):
        """One image, several passages, one picture to cite and show."""
        units = self._figure_units(self._parts())
        self.assertGreater(len(units), 1, "this fixture is meant to divide")
        for unit in units:
            self.assertEqual(("kb.docx::p0::imgabc123456789",), unit["asset_ids"])
            self.assertEqual("figure", unit["kind"])

    def test_a_figure_is_bounded_by_the_leaf_budget(self):
        """What `_FIGURE_LEAF_SIZE_MULTIPLIER` used to buy by throwing evidence away.

        Removing the splitter had left `_MILVUS_TEXT_CAP_BYTES` — 60 KB, seventy-five
        times the leaf budget — as the only bound, and the grader is handed every
        retrieved chunk in full: a handful of figures that size is a prompt the grading
        model cannot answer inside its output window, and the call returns
        `finish_reason: length`, a priced call turned into a parse failure."""
        calendar = self._parts("\n".join(
            f"Row {i} | White Shirt | Navy Blue Blazer | Navy Blue Pleated Skirt"
            for i in range(400)
        ))
        units = self._figure_units(calendar)

        self.assertGreater(len(units), 1)
        for unit in units:
            self.assertLessEqual(len(unit["text"]), self.loader._level_3_size)

    def test_nothing_is_dropped_when_a_figure_is_divided(self):
        """The half the cap got wrong. A transcription is divided, not truncated: the
        last row of a 400-row calendar is as indexed as the first."""
        calendar = self._parts("\n".join(
            f"Row {i} | White Shirt | Navy Blue Blazer" for i in range(400)
        ))
        indexed = "\n".join(unit["text"] for unit in self._figure_units(calendar))
        self.assertIn("Row 0 ", indexed)
        self.assertIn("Row 399 ", indexed)

    def test_a_transcribed_table_repeats_its_header_in_every_group(self):
        """A fee row without its column names answers nothing, which is why real tables
        already group this way — `_split_table_row_groups`, reused rather than restated."""
        fees = self._parts("\n".join(
            ["| Year | Egyptian | International |", "|---|---|---|"]
            + [f"| Y{i:02d} | {90 + i},000 EGP | {100 + i},000 EGP |" for i in range(60)]
        ))
        groups = [unit["text"] for unit in self._figure_units(fees)
                  if "EGP" in unit["text"]]
        self.assertGreater(len(groups), 1, "this fixture is meant to divide")
        for group in groups:
            self.assertIn("Year | Egyptian | International", group)

    def test_a_single_enormous_line_is_cut_rather_than_passed_through(self):
        """The hole the review found in the cap this replaces: it cut at a LINE boundary,
        so a first line of any length went through whole — a synthetic 10,000-character
        line passed unchanged, which makes the bound neither lossless nor a bound."""
        # A character that appears nowhere in the header, description or summary, so
        # counting it counts the transcription and not the repeated caption.
        one_line = self._parts("q" * 10_000)
        units = self._figure_units(one_line)

        for unit in units:
            self.assertLessEqual(len(unit["text"]), self.loader._level_3_size)
        self.assertEqual(
            10_000,
            sum(unit["text"].count("q") for unit in units),
            "cutting the line must bound it, not lose it",
        )

    def test_an_enormous_caption_cannot_starve_the_passage_budget(self):
        """The one input to this bound that is not itself bounded: a caption is whatever
        string the vision model returned, and only the heuristic extractor caps it.
        Uncut, it leaves a body budget of one character — every passage over the bound,
        and one image exploding into hundreds of chunks that all share an asset_id and
        compete for the same final slots."""
        shouty = self._parts()
        shouty["header"] = "[Figure] " + ("A very long caption indeed. " * 80)
        units = self._figure_units(shouty)

        self.assertLess(len(units), 20, "the passage count must not explode")
        for unit in units:
            self.assertLessEqual(len(unit["text"]), self.loader._level_3_size)

    def test_a_block_without_the_structured_parts_is_still_bounded_and_headed(self):
        """An older caller, or a block built by hand, carries only the joined surrogate.
        Its fields cannot be told apart — but the first line is the header by
        construction, and both properties this class exists for still hold."""
        flat = "[Figure] Fee schedule\n" + "\n".join(
            f"Grade {i} | 88,000 EGP" for i in range(300)
        )
        blocks = [{"type": "text", "content": flat, "page_number": 0,
                   "asset_ids": ["kb.docx::p0::imgdeadbeef1234"]}]
        units = [unit for unit in self.loader._blocks_to_units(blocks)
                 if unit["kind"] == "figure"]

        self.assertGreater(len(units), 1)
        for unit in units:
            self.assertTrue(unit["text"].startswith("[Figure] Fee schedule"))
            self.assertLessEqual(len(unit["text"]), self.loader._level_3_size)

    def test_the_whole_hierarchy_stays_within_its_level_budgets(self):
        """What bounding the UNIT buys, and the reason step 3 of this phase is safe.

        `_pack_units` closes a window when the NEXT unit would overflow it, but a single
        unit larger than the budget is packed alone and whole. So the level budgets bound
        a chunk only for as long as every unit is smaller than one — which is exactly
        what a figure was not."""
        calendar = self._parts("\n".join(
            f"Row {i} | White Shirt | Navy Blue Blazer" for i in range(400)
        ))
        chunks = self.loader._hierarchy_chunks(
            self.loader._blocks_to_units(self._blocks(calendar)),
            {"filename": "kb.docx", "file_path": "kb.docx", "file_type": "Word"},
        )
        budgets = {1: self.loader._level_1_size, 2: self.loader._level_2_size,
                   3: self.loader._level_3_size}
        # The section prefix is prepended after packing, so it is allowed on top of the
        # budget; it is itself capped at 150 characters plus its newline.
        for chunk in chunks:
            self.assertLessEqual(
                len(chunk["text"]), budgets[chunk["chunk_level"]] + 151,
                f"L{chunk['chunk_level']} chunk over budget: {len(chunk['text'])}",
            )

    def test_ordinary_prose_is_still_split(self):
        """The exemption is for figures, not a quiet end to leaf chunking."""
        prose = {"kind": "text", "text": "طلاب المدرسة " * 400, "sections": (), "page": 0}
        refined = self.loader._refine_units([prose], self.loader._splitter_level_3, 600)
        self.assertGreater(len(refined), 1)

    def test_a_table_is_still_regrouped_by_rows(self):
        table = {"kind": "table", "rows": [["Grade", "Fee"], ["Y1", "88,000"]],
                 "sections": (), "page": 0}
        refined = self.loader._refine_units([table], self.loader._splitter_level_3, 600)
        self.assertTrue(refined)
        self.assertTrue(all(unit["kind"] == "table" for unit in refined))


class TheCaptionComesFromTheImage(unittest.TestCase):
    def test_the_prompt_prefers_the_title_printed_in_the_image(self):
        """The old rule — "prefer the document's own caption wording" — is what put the
        Secondary figures under the Until-Grade-6 label: the document labels each picture
        underneath, so the nearest preceding text belongs to the picture before."""
        prompt = _template("assets/figure_extraction.j2")
        self.assertIn("its own title or heading, that IS the caption", prompt)
        self.assertNotIn("Prefer the document's own caption wording", prompt)

    def test_the_extractor_is_warned_the_context_may_be_a_neighbours(self):
        self.assertIn("NEIGHBOURING image", _template("assets/figure_extraction.j2"))

    def test_the_caption_change_forces_re_extraction(self):
        """The prompt fix is inert without this. Extractions are cached on (sha256,
        profile, dossier_version) and the images did not change, so a corrected prompt
        would keep serving captions the old one produced — which is exactly what a
        re-upload of the corrected document did."""
        self.assertGreaterEqual(DOSSIER_VERSION, 2)
        migration = MIGRATIONS[DOSSIER_VERSION - 1]
        self.assertTrue(migration.requires_reextraction)
        self.assertIn("caption", migration.description.lower())

    def test_every_version_below_the_current_one_has_a_migration(self):
        for version in range(1, DOSSIER_VERSION):
            self.assertIn(version, MIGRATIONS)


def _knowledge_result(**kwargs) -> str:
    base = dict(outcome="chunks", chunks="[1] Day Wear: Until Grade 6", constraints=[],
                child_year="", discriminate="unknown", rewritten=False, partial=False,
                figures=False)
    base.update(kwargs)
    return render_prompt("tools/knowledge_result.j2", **base)


class NarrowingIsNeverAReasonToRefuse(unittest.TestCase):
    """`discriminate == "yes"` was the only one of the three branches that did not forbid
    a refusal, and the only one that can narrow to an empty set."""

    def test_a_year_the_material_does_not_cover_is_reported_not_denied(self):
        rendered = _knowledge_result(child_year="Year 12", discriminate="yes")
        self.assertIn("Year 12", rendered)
        self.assertIn("does not cover", rendered)
        self.assertIn("Do not reply as though nothing was found", rendered)

    def test_a_carried_condition_the_material_does_not_cover_is_reported_too(self):
        rendered = _knowledge_result(constraints=["girls only"], discriminate="yes")
        self.assertIn("do not refuse", rendered)

    def test_the_other_two_branches_still_forbid_withholding(self):
        """Regression guard: these were correct before and must stay so."""
        for verdict in ("no", "unknown"):
            with self.subTest(discriminate=verdict):
                rendered = _knowledge_result(child_year="Year 12", discriminate=verdict)
                self.assertRegex(rendered, r"Do NOT withhold it|[Nn]ever refuse")

    def test_a_turn_about_no_child_renders_none_of_it(self):
        self.assertNotIn("THE YEAR TO ANSWER FOR", _knowledge_result())


class MaterialThatExcludesTheirCaseIsStillOnSubject(unittest.TestCase):
    """`relevance: none` is the only judgement that can end a turn in a denial
    (`rag/policy.py`). The prompt guarded the case where material is SILENT about a
    condition; it said nothing about material that names groups and leaves theirs out,
    which is what a uniform table per year group does."""

    def test_the_grader_is_told_that_naming_other_groups_is_not_a_different_subject(self):
        prompt = _template("rag/evidence_grade.j2")
        self.assertIn("NAMES groups and leaves the user's out", prompt)
        self.assertIn("`none` is only ever for snippets about something else entirely", prompt)

    def test_the_original_silence_rule_is_still_there(self):
        prompt = _template("rag/evidence_grade.j2")
        self.assertIn("Never grade `none` because a condition went unmentioned", prompt)


class ANameThatIsAlsoAnOrdinaryWord(unittest.TestCase):
    """`_names_the_child` is the only thing standing between a classifier that invents a
    name and a child being selected by it — and it compared both sides folded, where على
    and علي are one string."""

    def test_a_preposition_no_longer_confirms_an_invented_name(self):
        """Measured behaviour, not hypothetical: asked to classify a message after a turn
        about علي, the classifier returns `named` with «علي». Folded, the preposition
        satisfied the check, and a name beats a pin — so the parent was shown the wrong
        child's records."""
        self.assertFalse(_names_the_child("فيه خصم على الاخ التاني؟", "علي"))

    def test_the_name_spelled_as_the_name_still_counts(self):
        self.assertTrue(_names_the_child("ايه مصاريف علي؟", "علي"))

    def test_a_relationship_word_does_not_matter_here(self):
        """This check asks only whether the words appeared. Weighing the evidence for a
        homograph is `child_names.strip_child_names`' job, on a different question."""
        self.assertTrue(_names_the_child("ابني علي عنده امتحان", "علي"))

    def test_an_unambiguous_name_keeps_the_forgiving_comparison(self):
        """The whole reason folding is here: the SIS spells it with a maksura, the parent
        types a yeh, and they are one child."""
        self.assertTrue(_names_the_child("ايه مواعيد باص ليلي؟", "ليلى"))
        self.assertTrue(_names_the_child("مصاريف احمد", "أحمد"))

    def test_a_question_word_does_not_confirm_a_child_called_aya(self):
        """«آية» folds onto «إيه» — the Egyptian "what", which opens half of these
        messages."""
        self.assertFalse(_names_the_child("ايه مصاريف المدرسه؟", "آية"))
        self.assertTrue(_names_the_child("ايه مصاريف آية؟", "آية"))

    def test_an_age_does_not_confirm_a_child_called_omar(self):
        self.assertFalse(_names_the_child("ما هو العمر المطلوب للتقديم؟", "عمر"))

    def test_the_conjunction_attached_to_the_name_still_confirms_it(self):
        """Arabic writes "and" onto the front of the next word, so «وعمر عامل ايه؟» — how
        a parent actually moves the conversation to another child — holds no token «عمر».
        Rejecting it left the turn on the sibling they had stopped asking about."""
        self.assertTrue(_names_the_child("وعمر عامل ايه؟", "عمر"))
        self.assertTrue(_names_the_child("وعلي عنده امتحان؟", "علي"))

    def test_the_other_clitics_are_not_looked_past(self):
        """ف, ب, ل and ك build real words out of these names — «فعلي» is "actual". A name
        behind one goes unconfirmed, which falls back to the pin, not to a wrong child."""
        self.assertFalse(_names_the_child("ده اجراء فعلي في المدرسة", "علي"))

    def test_a_name_the_message_does_not_contain_at_all_is_still_refused(self):
        self.assertFalse(_names_the_child("ايه مصاريف المدرسه؟", "ليلى"))
        self.assertFalse(_names_the_child("ايه مصاريف المدرسه؟", ""))


if __name__ == "__main__":
    unittest.main()
