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


class AFigureSurrogateIsNeverCutInHalf(unittest.TestCase):
    """`_refine_units` re-splits what a coarser level produced. A figure must not be in
    it: `AssetDossier.render_surrogate` writes caption, description and transcription
    first and `Tags:`/`Answers:` last, so a split leaves a tail with no picture, no
    caption and no description — and that tail out-retrieves the figure it came from,
    because a question matches four question phrasings more closely than it matches an
    infographic's contents."""

    def setUp(self):
        self.loader = DocumentLoader()

    def _figure_unit(self, text: str) -> dict:
        return {"kind": "figure", "text": text, "sections": ("Handbook", "Uniform"),
                "page": 0, "asset_ids": ("kb.docx::p0::imgabc123456789",)}

    def _surrogate(self) -> str:
        return "\n".join([
            "[Figure] Day Wear: Secondary School - Girls",
            "An infographic illustrating the school uniform. " * 30,
            "White Shirt (long sleeves) Navy Blue Blazer Navy Blue Pleated Skirt " * 10,
            "Tags: School Uniform, Girls, Secondary, Day Wear, Blazer",
            "Answers: What is the uniform for secondary girls? What colour is the blazer?",
        ])

    def test_a_long_surrogate_survives_refinement_as_one_unit(self):
        surrogate = self._surrogate()
        refined = self.loader._refine_units(
            [self._figure_unit(surrogate)], self.loader._splitter_level_3, 600
        )
        self.assertEqual(1, len(refined))
        self.assertEqual(surrogate, refined[0]["text"])

    def test_the_tail_never_becomes_a_unit_of_its_own(self):
        refined = self.loader._refine_units(
            [self._figure_unit(self._surrogate())], self.loader._splitter_level_3, 600
        )
        for unit in refined:
            self.assertIn("[Figure]", unit["text"], "a piece with no caption was produced")

    def test_the_asset_reference_survives(self):
        refined = self.loader._refine_units(
            [self._figure_unit(self._surrogate())], self.loader._splitter_level_3, 600
        )
        self.assertEqual(("kb.docx::p0::imgabc123456789",), refined[0]["asset_ids"])
        self.assertEqual("figure", refined[0]["kind"])

    def test_a_surrogate_enters_the_unit_stream_whole(self):
        """The other half of the invariant. `_refine_units` carries a figure through
        untouched, which is worth nothing if the unit it was handed had already been cut
        — and the level-1 splitter would cut one whose transcription is long enough.

        Sized under the figure cap, because that is the case this asserts: a surrogate
        over it is trimmed rather than divided, which the cap tests below cover."""
        long_surrogate = self._surrogate() + ("\nNavy Blue Trousers " * 40)
        blocks = [
            {"type": "heading", "content": "School uniform", "level": 3, "page_number": 0},
            {"type": "text", "content": long_surrogate, "page_number": 0,
             "asset_ids": ["kb.docx::p0::imgabc123456789"]},
        ]
        figures = [unit for unit in self.loader._blocks_to_units(blocks)
                   if unit["kind"] == "figure"]
        self.assertEqual(1, len(figures))
        self.assertEqual(long_surrogate.strip(), figures[0]["text"])

    def test_a_figure_is_bounded_even_though_it_is_never_split(self):
        """Atomic is not the same as unlimited, and conflating them cost a production
        turn. Removing the splitter left `_MILVUS_TEXT_CAP_BYTES` — 60 KB, seventy-five
        times the leaf budget — as the only bound. `pipeline._format_docs` hands the
        grader every retrieved chunk in full, so a handful of figures that size is a
        prompt the grading model cannot answer inside its output window, and the call
        returns `finish_reason: length`: a priced call turned into a parse failure."""
        cap = self.loader._level_3_size * self.loader._FIGURE_LEAF_SIZE_MULTIPLIER
        huge = "[Figure] Day Wear: Secondary School - Girls\n" + "\n".join(
            f"Row {i} | White Shirt | Navy Blue Blazer | Navy Blue Pleated Skirt"
            for i in range(400)
        )
        blocks = [
            {"type": "heading", "content": "School uniform", "level": 3, "page_number": 0},
            {"type": "text", "content": huge, "page_number": 0,
             "asset_ids": ["kb.docx::p0::imgdeadbeef1234"]},
        ]
        figures = [unit for unit in self.loader._blocks_to_units(blocks)
                   if unit["kind"] == "figure"]

        self.assertEqual(1, len(figures), "still one chunk per image")
        self.assertLessEqual(len(figures[0]["text"]), cap)
        self.assertLess(cap, self.loader._MILVUS_TEXT_CAP_BYTES)

    def test_the_cap_keeps_the_caption_and_cuts_whole_lines(self):
        """`render_surrogate` orders its fields so a truncation drops the least valuable
        content first. Cutting mid-line would end a transcribed table mid-row."""
        huge = "[Figure] Day Wear: Secondary School - Girls\n" + "\n".join(
            f"Row {i} | White Shirt | Navy Blue Blazer" for i in range(400)
        )
        capped = self.loader._cap_figure_text(huge)

        self.assertTrue(capped.startswith("[Figure] Day Wear: Secondary School - Girls"))
        self.assertIn(capped.splitlines()[-1], huge.splitlines())

    def test_a_figure_within_the_cap_is_untouched(self):
        surrogate = self._surrogate()
        self.assertEqual(surrogate, self.loader._cap_figure_text(surrogate))

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
