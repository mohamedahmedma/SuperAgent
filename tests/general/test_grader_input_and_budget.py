# -*- coding: utf-8 -*-
"""The grading call has to fit, and it was sized by the corpus rather than the retrieval.

Production failed with `openai.LengthFinishReasonError: Could not parse response content
as the length limit was reached` — a grading call that ended at `finish_reason: length`
having emitted no JSON at all. Three things were true at once:

  * `format_docs` hands the grader EVERY retrieved chunk in full, so the size of the
    grading prompt was whatever the corpus happened to hold. A figure whose transcription
    is a school calendar rendered as a year of table rows is one chunk;
  * `grade_max_tokens` was 0, which does not mean "no limit" — it means the provider's
    own default, and GRADE_MODEL here is a reasoning model whose scratchpad is billed
    against the same completion budget as the JSON and produced first;
  * and a truncated grade raised, which costs the whole turn: grading is the HIGH rung,
    so the ladder reaches nothing and the turn routes to `retrieval_error` — the user is
    told the knowledge base is broken while its answer sits in the retrieved chunks.

The last two are unchanged and their guards are the last two classes here.

The FIRST was fixed by showing the grader less than the answer model: prose cut to 1,200
characters, a figure summarised to 500. That bounded the prompt and it hid the evidence —
measured on the school corpus, 28% of questions had their answer in the part of a chunk
the grader never saw, and 58% of retrieved chunks were over the cap. The truncation is
gone and the bound moved to where the size is MADE, which is what the first class here
now pins: no chunk larger than the evidence window, therefore no grading prompt larger
than `top_k` times it, whatever the corpus contains.
"""
import unittest
from unittest.mock import patch

from backend.indexing.document_loader import DocumentLoader
from backend.llm_models import GRADE_RETRY_MAX_TOKENS
from backend.profiles import get_profile
from backend.rag.evidence import AssessmentContext, Certainty
from backend.rag.evidence_view import format_docs
from backend.rag.pipeline import EvidenceGrade, LLMGraderAssessor
from backend.rag.utils import EVIDENCE_WINDOW_CHARS, _parent_window


def _figure_doc(rows: int = 300) -> dict:
    return {
        "filename": "kb.docx",
        "page_number": 0,
        "text": "[Figure] School calendar 2025-2026\nA year planner for every term.\n"
        + "\n".join(f"Week {i} | Mon | Tue | Wed | Thu | Fri" for i in range(rows)),
    }


def _calendar_parts(rows: int = 400) -> dict:
    return {
        "header": "[Figure] School calendar 2025-2026",
        "description": "A year planner for every term.",
        "transcription": "\n".join(
            f"| Week {i} | Mon | Tue | Wed | Thu | Fri |" for i in range(rows)
        ),
        "summary": "Tags: calendar, terms",
    }


class TheGradingPromptIsSizedByTheRetrieval(unittest.TestCase):
    """The invariant commit 1794762 protected, kept by bounding instead of truncating.

    The grader now reads whole chunks, so the only thing standing between the corpus and
    the size of the prompt is the size of a chunk. These assert that bound at the two
    places a chunk can get big — indexing, and the merge — and then assert the arithmetic
    that follows from it."""

    def setUp(self):
        self.loader = DocumentLoader()

    def test_the_grader_now_reads_what_the_answer_reads(self):
        """The 28%. A fee table read out of an image is only useful entire, and the
        grader was deciding whether the snippets settle the question without being shown
        the part that settles it."""
        docs = [_figure_doc()]
        self.assertEqual(format_docs(docs), format_docs(docs))
        self.assertIn("Week 299", format_docs(docs))

    def test_one_image_can_no_longer_be_one_enormous_chunk(self):
        """The chunk that killed the grading call. Its transcription is now indexed as
        passages, none of them larger than a leaf."""
        units = [
            unit for unit in self.loader._blocks_to_units([
                {"type": "text", "content": "joined", "page_number": 0,
                 "asset_ids": ["kb.docx::p0::imgabc"], "figure": _calendar_parts()},
            ]) if unit["kind"] == "figure"
        ]

        self.assertGreater(len(units), 1)
        for unit in units:
            self.assertLessEqual(len(unit["text"]), self.loader._level_3_size)

    def test_merging_cannot_exceed_the_evidence_window(self):
        """The other way a chunk gets big: a two-child match promotes a whole parent."""
        parent = "\n".join(f"line {i:04d} " + "x" * 80 for i in range(400))
        window = _parent_window(parent, [{"text": "line 0200 " + "x" * 80}], 1200)

        self.assertLessEqual(len(window), 1200)
        self.assertIn("line 0200", window)

    def test_the_prompt_is_top_k_times_the_window_and_nothing_else(self):
        """The arithmetic the invariant actually is. Eight chunks at the ceiling is a
        number this file can state; eight chunks of "whatever the corpus holds" is not,
        and that difference is the whole finding."""
        docs = [
            {"filename": "kb.docx", "page_number": 0, "text": "x" * EVIDENCE_WINDOW_CHARS}
            for _ in range(8)
        ]
        rendered = format_docs(docs)

        per_chunk_overhead = 60  # "[n] kb.docx (Page 0):" plus the separator
        self.assertLess(len(rendered), 8 * (EVIDENCE_WINDOW_CHARS + per_chunk_overhead))

    def test_chunk_numbering_is_the_contract_between_grade_and_answer(self):
        """`supporting_chunks` is 1-based over this list and `select_context_indices`
        applies it to the same docs, so the numbering is what keeps a grade about chunk 3
        pointing at chunk 3."""
        docs = [_figure_doc(rows=2), _figure_doc(rows=2), _figure_doc(rows=2)]
        rendered = format_docs(docs)
        for marker in ("[1]", "[2]", "[3]"):
            self.assertIn(marker, rendered)

    def test_an_ordinary_chunk_is_passed_through_untouched(self):
        doc = {"filename": "kb.docx", "page_number": 0, "text": "الرسوم الدراسية للصف الثالث"}
        self.assertIn("الرسوم الدراسية للصف الثالث", format_docs([doc]))

    def test_no_documents_renders_nothing(self):
        self.assertEqual("", format_docs([]))


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
