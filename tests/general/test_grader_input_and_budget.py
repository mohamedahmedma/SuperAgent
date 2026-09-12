# -*- coding: utf-8 -*-
"""The grading call has to fit, and it was sized by the corpus rather than the retrieval.

Production failed with `openai.LengthFinishReasonError: Could not parse response content
as the length limit was reached` — a grading call that ended at `finish_reason: length`
having emitted no JSON at all. Three things were true at once and each is fixed here:

  * `_format_docs` hands the grader EVERY retrieved chunk in full, so the size of the
    grading prompt was whatever the corpus happened to hold. A figure whose transcription
    is a school calendar rendered as a year of table rows is one chunk;
  * `grade_max_tokens` was 0, which does not mean "no limit" — it means the provider's
    own default, and GRADE_MODEL here is a reasoning model whose scratchpad is billed
    against the same completion budget as the JSON and produced first;
  * and a truncated grade raised, which costs the whole turn: grading is the HIGH rung,
    so the ladder reaches nothing and the turn routes to `retrieval_error` — the user is
    told the knowledge base is broken while its answer sits in the retrieved chunks.
"""
import unittest
from unittest.mock import patch

from backend.profiles import get_profile
from backend.rag.evidence import AssessmentContext, Certainty
from backend.rag.pipeline import (
    _GRADER_CHUNK_CHARS,
    _GRADER_FIGURE_BODY_CHARS,
    _GRADE_RETRY_MAX_TOKENS,
    _grading_view,
    EvidenceGrade,
    LLMGraderAssessor,
    _format_docs,
    _format_docs_for_grading,
    _head,
)


def _figure_doc(rows: int = 300) -> dict:
    return {
        "filename": "kb.docx",
        "page_number": 0,
        "text": "[Figure] School calendar 2025-2026\nA year planner for every term.\n"
        + "\n".join(f"Week {i} | Mon | Tue | Wed | Thu | Fri" for i in range(rows)),
    }


class TheGraderSeesLessThanTheAnswerDoes(unittest.TestCase):
    def test_each_chunk_is_capped(self):
        rendered = _format_docs_for_grading([_figure_doc() for _ in range(8)])
        # Eight chunks, each capped, plus the per-chunk header and separator.
        self.assertLess(len(rendered), 8 * (_GRADER_CHUNK_CHARS + 200))

    def test_the_answer_path_is_untouched(self):
        """A fee table read out of an image is only useful entire, and that is what the
        answering model is given. Only the grade is taken on the head of the chunk."""
        docs = [_figure_doc()]
        self.assertGreater(
            len(_format_docs(docs)), 5 * len(_format_docs_for_grading(docs))
        )

    def test_the_caption_and_description_survive(self):
        """What a grade is actually made of. `render_surrogate` writes caption first and
        description second, so a cap taken from the top keeps exactly the part that says
        what the figure IS and drops the literal rows."""
        rendered = _format_docs_for_grading([_figure_doc()])
        self.assertIn("[Figure] School calendar 2025-2026", rendered)
        self.assertIn("A year planner for every term.", rendered)
        self.assertNotIn("Week 299", rendered)

    def test_an_ordinary_chunk_is_not_trimmed_at_all(self):
        """The cap sits above a normal leaf, so it only ever touches an outlier."""
        doc = {"filename": "kb.docx", "page_number": 0, "text": "الرسوم الدراسية للصف الثالث"}
        self.assertIn("الرسوم الدراسية للصف الثالث", _format_docs_for_grading([doc]))

    def test_chunk_numbering_still_matches_the_answer_path(self):
        """`supporting_chunks` is 1-based over this list and `select_context_indices`
        applies it to the full docs, so the two renderings must number alike."""
        docs = [_figure_doc(), _figure_doc(), _figure_doc()]
        for marker in ("[1]", "[2]", "[3]"):
            self.assertIn(marker, _format_docs_for_grading(docs))
            self.assertIn(marker, _format_docs(docs))

    def test_a_figure_keeps_its_title_tags_and_questions(self):
        """What the grade is actually made of. The transcription is the ANSWER's
        evidence; for "are these snippets about school uniform" it is noise that can
        outweigh the rest of the prompt."""
        view = _grading_view(
            "[Figure] Day Wear: Secondary School - Girls\n"
            "An infographic showing the secondary uniform.\n"
            + "\n".join(f"Row {i} | White Shirt | Navy Blazer" for i in range(300))
            + "\nTags: School Uniform, Girls, Secondary\n"
              "Answers: What is the uniform for secondary girls?"
        )
        self.assertIn("[Figure] Day Wear: Secondary School - Girls", view)
        self.assertIn("An infographic showing the secondary uniform.", view)
        self.assertIn("Tags: School Uniform, Girls, Secondary", view)
        self.assertIn("Answers: What is the uniform for secondary girls?", view)
        self.assertNotIn("Row 299", view)

    def test_the_summary_lines_survive_a_transcription_that_buries_them(self):
        """Truncating from the top would keep the first rows of the table and lose the
        two lines underneath that say what the picture is about."""
        view = _grading_view(
            "[Figure] Fee schedule\n"
            + "\n".join(f"Grade {i} | 88,000 EGP" for i in range(500))
            + "\nTags: fees, tuition"
        )
        self.assertIn("Tags: fees, tuition", view)
        self.assertLess(len(view), _GRADER_FIGURE_BODY_CHARS + 200)

    def test_prose_is_capped_but_never_summarised(self):
        """Only a figure has a transcription to drop. Ordinary text is left alone below
        the cap, because a leaf is smaller than it."""
        prose = "الرسوم الدراسية للصف الثالث الابتدائي هي ٨٨٬٠٠٠ جنيه"
        self.assertEqual(prose, _grading_view(prose))

    def test_the_retry_ceiling_is_a_doubling_not_a_leap(self):
        self.assertLessEqual(_GRADE_RETRY_MAX_TOKENS, 2 * get_profile().models.grade_max_tokens)

    def test_the_head_helper_cuts_whole_lines(self):
        text = "\n".join(f"line {i}" for i in range(200))
        cut = _head(text, 100)
        self.assertLessEqual(len(cut), 100)
        self.assertIn(cut.splitlines()[-1], text.splitlines())

    def test_a_single_line_longer_than_the_budget_still_yields(self):
        """Half of a transcription rendered as one enormous row beats none of it."""
        self.assertEqual(40, len(_head("x" * 500, 40)))


class TheGradeCallIsGivenRoomToFinish(unittest.TestCase):
    def test_the_profile_floors_the_grade_output_budget(self):
        """0 does not mean "no limit" — it means the provider's default, which a
        reasoning model can spend entirely on its scratchpad before any JSON appears.

        A floor, not a licence: the grade is under 200 tokens and `low` effort reasons in
        the hundreds, so this clears both without room for a model to think for a page."""
        budget = get_profile().models.grade_max_tokens
        self.assertGreaterEqual(budget, 1024)
        self.assertLessEqual(budget, 2048)

    def test_grading_still_asks_for_the_least_reasoning_the_parameter_offers(self):
        """`low`, and deliberately NOT none/off: `_validate_effort` maps those onto "",
        which OMITS the field, and a reasoning model then applies its own higher default.
        Turning it "off" here would buy more reasoning and less room."""
        self.assertEqual("low", get_profile().models.grade_reasoning_effort)


class _Truncated(Exception):
    """Stands in for `openai.LengthFinishReasonError`, which cannot be constructed
    without a real completion object."""


class _Grader:
    def __init__(self, raises=None, grade=None):
        self._raises = raises
        self._grade = grade
        self.calls = 0

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._grade


_GRADE = EvidenceGrade(
    relevance="strong", answerability="sufficient", ambiguity="none",
    constraints_discriminate="no", route="answer", confidence=0.9,
)


class ATruncatedGradeIsRetriedNotSurrendered(unittest.TestCase):
    def _assess(self, first, second):
        def _model(*, headroom: bool = False):
            return second if headroom else first

        with patch("backend.rag.pipeline._get_grader_model", _model), \
             patch("backend.rag.pipeline._TRUNCATED_RESPONSE", (_Truncated,)):
            return LLMGraderAssessor().assess(
                AssessmentContext(
                    question="ايه لبس المدرسة؟",
                    docs=[_figure_doc(rows=5)],
                    retrieval_meta={},
                    config=get_profile().rag,
                    constraints=[],
                )
            )

    def test_the_second_attempt_produces_the_grade(self):
        first = _Grader(raises=_Truncated("length"))
        second = _Grader(grade=_GRADE)
        report = self._assess(first, second)

        self.assertEqual(1, first.calls)
        self.assertEqual(1, second.calls)
        self.assertEqual("strong", report.relevance)
        self.assertEqual(Certainty.HIGH, report.certainty)

    def test_a_grade_that_fits_never_builds_the_second_model(self):
        first = _Grader(grade=_GRADE)
        second = _Grader(grade=_GRADE)
        self._assess(first, second)

        self.assertEqual(1, first.calls)
        self.assertEqual(0, second.calls, "the headroom model is only for a retry")

    def test_a_second_truncation_is_allowed_to_fail(self):
        """Retried once, with room rather than with different instructions. A second
        failure means the honest "try again" is the right answer."""
        first = _Grader(raises=_Truncated("length"))
        second = _Grader(raises=_Truncated("length again"))
        with self.assertRaises(_Truncated):
            self._assess(first, second)


if __name__ == "__main__":
    unittest.main()
