"""Conditional evidence grading.

The grader is one synchronous LLM call per retrieval — and one per sub-agent on a
decomposed question — so it usually dominates a turn's latency while often only
confirming what retrieval already made obvious.

The bias is deliberately asymmetric. Skipping wrongly means answering from weak
evidence, which is the failure the pipeline exists to prevent. Grading unnecessarily
only costs time. So every signal must agree before the grader is skipped, and these
tests are weighted toward proving it is NOT skipped when it matters.
"""
import unittest

from backend.profiles.registry import load_profile
from backend.rag.confidence import (
    ConfidenceVerdict,
    assess,
    content_tokens,
    has_rerank_scores,
    should_grade,
    term_coverage,
)


def rag_config(**overrides):
    config = load_profile("base").rag
    return config.model_copy(update=overrides) if overrides else config


def doc(text, score=0.5, rerank=None):
    item = {"text": text, "score": score, "filename": "kb.docx"}
    if rerank is not None:
        item["rerank_score"] = rerank
    return item


class TokenisationTests(unittest.TestCase):
    def test_stop_words_and_single_characters_are_dropped(self):
        tokens = content_tokens("what are the partners of the school")
        self.assertIn("partners", tokens)
        self.assertIn("school", tokens)
        for noise in ("what", "are", "the", "of"):
            self.assertNotIn(noise, tokens)

    def test_arabic_function_words_are_dropped(self):
        tokens = content_tokens("ما هي الرسوم في المدرسة")
        self.assertNotIn("ما", tokens)
        self.assertNotIn("في", tokens)
        self.assertTrue(tokens)

    def test_an_empty_question_yields_no_tokens(self):
        self.assertEqual([], content_tokens(""))
        self.assertEqual([], content_tokens("the of a an"))


class TermCoverageTests(unittest.TestCase):
    def test_full_coverage_when_the_chunk_answers_the_question(self):
        docs = [doc("Our partner organisations include three universities and the school board.")]
        self.assertEqual(1.0, term_coverage("what are the partners", docs))

    def test_inflections_still_count(self):
        """Prefix matching, so 'partner' covers 'partners' and Arabic stems match."""
        self.assertEqual(1.0, term_coverage("partner", [doc("our partners are listed below")]))

    def test_no_coverage_for_an_unrelated_corpus(self):
        self.assertEqual(0.0, term_coverage("quantum entanglement", [doc("school uniform policy")]))

    def test_a_question_of_only_stop_words_scores_zero(self):
        self.assertEqual(0.0, term_coverage("what is the", [doc("anything at all")]))

    def test_empty_documents_score_zero(self):
        self.assertEqual(0.0, term_coverage("partners", []))


class SkipDecisionTests(unittest.TestCase):
    """Every condition must hold; any doubt routes to the grader."""

    def setUp(self):
        self.config = rag_config()
        self.strong = [doc("The school partners are the university and the council.")] * 4

    def test_strong_retrieval_skips_the_grader(self):
        verdict = assess("what are the school partners", self.strong, {}, self.config)
        self.assertTrue(verdict.confident)
        self.assertEqual(1.0, verdict.term_coverage)

    def test_too_few_chunks_always_grades(self):
        """Two matching chunks is not the same evidence as several."""
        verdict = assess("what are the school partners", self.strong[:2], {}, self.config)
        self.assertFalse(verdict.confident)
        self.assertIn("chunk", verdict.reasons[0])

    def test_low_term_coverage_always_grades(self):
        docs = [doc("The uniform is navy blue with gold accents.")] * 4
        verdict = assess("what are the scholarship deadlines", docs, {}, self.config)
        self.assertFalse(verdict.confident)
        self.assertIn("coverage", verdict.reasons[0])

    def test_no_documents_is_never_confidence(self):
        verdict = assess("anything", [], {}, self.config)
        self.assertFalse(verdict.confident)
        self.assertEqual(["no_documents"], verdict.reasons)

    def test_a_weak_rerank_score_always_grades(self):
        docs = [doc("The school partners are the university.", rerank=0.1)] * 4
        verdict = assess("what are the school partners", docs, {}, self.config)
        self.assertFalse(verdict.confident)
        self.assertIn("rerank", verdict.reasons[0])

    def test_a_strong_rerank_score_is_accepted(self):
        docs = [doc("The school partners are the university.", rerank=0.9)] * 4
        self.assertTrue(assess("what are the school partners", docs, {}, self.config).confident)

    def test_rrf_scores_are_not_measured_against_the_rerank_threshold(self):
        """The bug this guards: `rerank_applied` is set BEFORE the HTTP call and stays
        true when it fails, so a raw RRF score (~0.03) was being compared against a
        threshold meant for a calibrated 0-1 relevance score, rejecting everything."""
        docs = [doc("The school partners are the university.", score=0.03)] * 4
        meta = {"rerank_applied": True, "rerank_error": "connection failed"}
        self.assertFalse(has_rerank_scores(docs))
        self.assertTrue(assess("what are the school partners", docs, meta, self.config).confident)

    def test_thresholds_are_profile_driven(self):
        strict = rag_config(skip_grading_term_coverage=0.99, skip_grading_min_chunks=10)
        self.assertFalse(assess("what are the school partners", self.strong, {}, strict).confident)


class GradingModeTests(unittest.TestCase):
    def setUp(self):
        self.strong = [doc("The school partners are the university and the council.")] * 4

    def test_always_grades_regardless_of_confidence(self):
        need, verdict = should_grade("what are the school partners", self.strong, {},
                                     rag_config(grading_mode="always"))
        self.assertTrue(need)
        self.assertIn("grading_mode=always", verdict.reasons)

    def test_never_skips_regardless_of_confidence(self):
        need, verdict = should_grade("quantum entanglement", [doc("uniform policy")], {},
                                     rag_config(grading_mode="never"))
        self.assertFalse(need)
        self.assertIn("grading_mode=never", verdict.reasons)

    def test_uncertain_only_decides_per_query(self):
        config = rag_config(grading_mode="uncertain_only")
        self.assertFalse(should_grade("what are the school partners", self.strong, {}, config)[0])
        self.assertTrue(should_grade("scholarship deadlines abroad",
                                     [doc("uniform policy")] * 4, {}, config)[0])

    def test_the_shipped_default_is_always(self):
        """Held at `always` until a semantic assessor replaces lexical coverage. The
        conditional path stays tested and one config line away."""
        self.assertEqual("always", load_profile("base").rag.grading_mode)


class SkippedGradeShapeTests(unittest.TestCase):
    """What the pipeline substitutes when it skips the call."""

    def test_a_cheap_assessor_can_never_route_to_a_human(self):
        """clarify and scope_select need a model to have read the question. This used to
        be enforced by a helper that hand-built a fake grade; it is now structural — the
        report's ambiguity defaults to the inert value and only an LLM assessor sets it,
        so no cheap rung can invent an ambiguity it never assessed."""
        from backend.rag.evidence import LexicalAssessor, AssessmentContext
        from backend.rag.policy import decide_route

        config = rag_config()
        ctx = AssessmentContext(
            question="what are the school partners",
            docs=[doc("The school partners are the university and the council.")] * 4,
            config=config,
        )
        report = LexicalAssessor().assess(ctx)
        self.assertEqual("none", report.ambiguity)
        self.assertIsNone(report.preferred_route)

        # And a LOW-certainty report cannot claim sufficiency either: with the profile
        # requiring `high`, acting on it is refused rather than guessed.
        route, reason = decide_route(report, has_docs=True, rewrite_count=0,
                                     is_sub_agent=False, config=config)
        self.assertNotIn(route, ("clarify", "scope_select"))
        self.assertEqual("retrieval_error", route)
        self.assertIn("requires high", reason)

    def test_the_decision_is_recorded_in_the_trace(self):
        """A skipped grade must be visible, not invisible."""
        verdict = ConfidenceVerdict(confident=True, term_coverage=0.9, chunk_count=4,
                                    reasons=["4 chunks"])
        trace = verdict.as_trace()
        self.assertTrue(trace["grading_confident"])
        self.assertEqual(0.9, trace["grading_term_coverage"])
        self.assertEqual(4, trace["grading_chunk_count"])
        self.assertIn("4 chunks", trace["grading_reason"])

    def test_the_trace_schema_carries_the_new_fields(self):
        from backend.schemas.chat import normalize_rag_trace

        trace = normalize_rag_trace({
            "grading_skipped": True, "grading_confident": True,
            "grading_term_coverage": 0.9, "grading_chunk_count": 4,
            "grading_reason": "4 chunks",
        })
        self.assertTrue(trace["grading_skipped"])
        self.assertEqual(0.9, trace["grading_term_coverage"])


if __name__ == "__main__":
    unittest.main()
