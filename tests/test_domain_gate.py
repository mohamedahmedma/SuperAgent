"""The domain gate and the cross-encoder rung.

The gate's saving is real but secondary. What these tests mostly prove is that it fails
open: a false rejection is an unrecoverable silent refusal of a valid question, while a
false acceptance costs one search the grader would have caught anyway.
"""
import unittest
from unittest.mock import patch

from backend.profiles.registry import load_profile
from backend.rag.domain_gate import (
    DomainReference,
    DomainReferenceStore,
    classify,
    should_run,
)
from backend.rag.evidence import AssessmentContext, Certainty
from backend.rag.rerank_assessor import CrossEncoderAssessor, _to_unit, reset_model_cache


def rag_config(**overrides):
    settings = {"domain_gate_enabled": True, **overrides}
    return load_profile("base").rag.model_copy(update=settings)


# Orthogonal unit vectors, so a query aligned with one scores 1.0 and 0.0 against others.
UNIFORM = [1.0, 0.0, 0.0]
FEES = [0.0, 1.0, 0.0]
TERMS = [0.0, 0.0, 1.0]

REFERENCE = DomainReference(vectors=[("uniform", UNIFORM), ("fees", FEES), ("terms", TERMS)])


class EligibilityTests(unittest.TestCase):
    def test_disabled_by_default(self):
        run, why = should_run("what is the uniform policy", has_history=False,
                              config=load_profile("base").rag)
        self.assertFalse(run)
        self.assertIn("domain_gate_enabled=false", why)

    def test_a_turn_with_history_is_exempt(self):
        """"What about grade 6?" carries its topic in the previous turn. Scoring its own
        words measures nothing, and a low score would refuse a valid follow-up."""
        run, why = should_run("what about grade 6", has_history=True, config=rag_config())
        self.assertFalse(run)
        self.assertIn("follow-up", why)

    def test_history_exemption_can_be_turned_off(self):
        run, _ = should_run("what is the uniform policy for grade 6", has_history=True,
                            config=rag_config(domain_gate_skip_with_history=False))
        self.assertTrue(run)

    def test_a_very_short_question_is_exempt(self):
        run, why = should_run("fees?", has_history=False, config=rag_config())
        self.assertFalse(run)
        self.assertIn("shorter than", why)

    def test_an_empty_question_is_exempt(self):
        self.assertFalse(should_run("   ", has_history=False, config=rag_config())[0])

    def test_a_normal_first_turn_question_is_eligible(self):
        run, _ = should_run("what is the school uniform policy", has_history=False,
                            config=rag_config())
        self.assertTrue(run)


class ClassificationTests(unittest.TestCase):
    def test_an_aligned_question_is_in_domain_with_its_topic(self):
        verdict = classify(UNIFORM, REFERENCE, rag_config())
        self.assertTrue(verdict.in_domain)
        self.assertFalse(verdict.abstained)
        self.assertEqual(["uniform"], verdict.topics)
        self.assertAlmostEqual(1.0, verdict.score)

    def test_an_unrelated_question_is_rejected(self):
        verdict = classify([0.0, 0.0, 0.0], REFERENCE, rag_config())
        self.assertFalse(verdict.in_domain)
        self.assertFalse(verdict.abstained)

    def test_several_topics_are_carried_when_several_match(self):
        """One guessed topic that is wrong destroys recall, so the gate keeps a shortlist."""
        blended = [0.7, 0.7, 0.0]
        verdict = classify(blended, REFERENCE, rag_config())
        self.assertTrue(verdict.in_domain)
        self.assertEqual({"uniform", "fees"}, set(verdict.topics))

    def test_the_topic_shortlist_is_capped_by_config(self):
        verdict = classify([0.6, 0.6, 0.6], REFERENCE, rag_config(domain_gate_topic_count=2))
        self.assertLessEqual(len(verdict.topics), 2)

    def test_the_threshold_is_profile_driven(self):
        near = [0.4, 0.0, 0.0]
        self.assertTrue(classify(near, REFERENCE, rag_config()).in_domain)
        self.assertFalse(
            classify(near, REFERENCE, rag_config(domain_gate_min_similarity=0.9)).in_domain)


class FailOpenTests(unittest.TestCase):
    """Every failure must let the question through."""

    def test_no_query_vector_abstains(self):
        verdict = classify(None, REFERENCE, rag_config())
        self.assertTrue(verdict.in_domain)
        self.assertTrue(verdict.abstained)
        self.assertEqual("no_query_vector", verdict.reason)

    def test_an_empty_reference_set_abstains(self):
        verdict = classify(UNIFORM, DomainReference(), rag_config())
        self.assertTrue(verdict.in_domain)
        self.assertTrue(verdict.abstained)

    def test_a_dimension_mismatch_abstains_rather_than_rejecting(self):
        """A re-index with a different embedding model must not look like every question
        suddenly being off-topic."""
        verdict = classify([1.0, 0.0], REFERENCE, rag_config())
        self.assertTrue(verdict.in_domain)
        self.assertTrue(verdict.abstained)

    def test_a_broken_reference_provider_leaves_the_store_empty(self):
        def explode():
            raise RuntimeError("milvus down")

        store = DomainReferenceStore(provider=explode)
        self.assertFalse(store.get().ready)
        self.assertTrue(classify(UNIFORM, store.get(), rag_config()).abstained)

    def test_a_broken_provider_is_not_retried_on_every_request(self):
        calls = []

        def explode():
            calls.append(1)
            raise RuntimeError("milvus down")

        store = DomainReferenceStore(provider=explode)
        store.get()
        store.get()
        store.get()
        self.assertEqual(1, len(calls))

    def test_invalidate_allows_a_rebuild_after_reindexing(self):
        calls = []

        def provider():
            calls.append(1)
            return REFERENCE

        store = DomainReferenceStore(provider=provider)
        store.get()
        store.get()
        self.assertEqual(1, len(calls))
        store.invalidate()
        store.get()
        self.assertEqual(2, len(calls))


class SharedEmbeddingTests(unittest.TestCase):
    def test_the_query_vector_is_computed_once_per_text(self):
        """The gate classifies with the vector retrieval is about to search with. A second
        bge-m3 forward pass would cost more than the gate saves."""
        import backend.indexing.embedding as embedding

        calls = []

        class Counting:
            def get_embeddings(self, texts):
                calls.append(texts[0])
                return [[0.1, 0.2, 0.3]]

        embedding.reset_query_vector_cache()
        try:
            with patch.object(embedding, "embedding_service", Counting()):
                first = embedding.embed_query("what is the uniform policy")
                second = embedding.embed_query("what is the uniform policy")
            self.assertEqual(1, len(calls))
            self.assertEqual(first, second)
        finally:
            embedding.reset_query_vector_cache()

    def test_a_caller_mutating_the_vector_cannot_corrupt_the_memo(self):
        import backend.indexing.embedding as embedding

        class Fixed:
            def get_embeddings(self, texts):
                return [[0.1, 0.2, 0.3]]

        embedding.reset_query_vector_cache()
        try:
            with patch.object(embedding, "embedding_service", Fixed()):
                first = embedding.embed_query("q")
                first.append(99.0)
                self.assertEqual([0.1, 0.2, 0.3], embedding.embed_query("q"))
        finally:
            embedding.reset_query_vector_cache()


class CrossEncoderAssessorTests(unittest.TestCase):
    """Rung 2: the semantic signal whose absence forced the rollback."""

    def setUp(self):
        reset_model_cache()
        self.docs = [{"text": f"chunk {i}"} for i in range(1, 5)]

    def _assess(self, scores, **overrides):
        config = load_profile("base").rag.model_copy(
            update={"rerank_cross_encoder_enabled": True, **overrides})
        ctx = AssessmentContext(question="what is the uniform", docs=self.docs, config=config)
        with patch("backend.rag.rerank_assessor.score_pairs", lambda *a, **k: scores):
            return CrossEncoderAssessor().assess(ctx)

    def test_disabled_by_default(self):
        ctx = AssessmentContext(question="q", docs=self.docs, config=load_profile("base").rag)
        self.assertIsNone(CrossEncoderAssessor().assess(ctx))

    def test_an_unavailable_model_abstains_so_the_ladder_climbs(self):
        config = load_profile("base").rag.model_copy(update={"rerank_cross_encoder_enabled": True})
        ctx = AssessmentContext(question="q", docs=self.docs, config=config)
        with patch("backend.rag.rerank_assessor.score_pairs", lambda *a, **k: None):
            self.assertIsNone(CrossEncoderAssessor().assess(ctx))

    def test_high_scores_conclude_sufficient_at_medium_certainty(self):
        report = self._assess([0.91, 0.72, 0.11, 0.04])
        self.assertEqual(Certainty.MEDIUM, report.certainty)
        self.assertEqual("sufficient", report.sufficiency)
        self.assertEqual("answer", report.preferred_route)

    def test_it_names_which_chunks_carried_the_evidence(self):
        """This is what makes trimming a consequence of the judgement rather than a guess."""
        report = self._assess([0.91, 0.72, 0.11, 0.04])
        self.assertEqual([1, 2], report.supported_indices())

    def test_a_uniformly_irrelevant_pool_concludes_no_knowledge(self):
        report = self._assess([0.02, 0.01, 0.004, 0.0])
        self.assertEqual(Certainty.MEDIUM, report.certainty)
        self.assertEqual("none", report.relevance)
        self.assertEqual("no_knowledge", report.preferred_route)

    def test_the_middle_band_abstains_and_leaves_it_to_the_grader(self):
        """Relevant enough not to dismiss, not clearly sufficient — exactly where an LLM
        is worth its cost. The report stays LOW so no policy acts on it."""
        report = self._assess([0.45, 0.33, 0.2, 0.1])
        self.assertEqual(Certainty.LOW, report.certainty)
        self.assertEqual("unknown", report.sufficiency)
        self.assertIsNone(report.preferred_route)

    def test_it_can_never_invent_an_ambiguity(self):
        report = self._assess([0.91, 0.72, 0.11, 0.04])
        self.assertEqual("none", report.ambiguity)

    def test_thresholds_are_profile_driven(self):
        report = self._assess([0.5, 0.4, 0.1, 0.0], rerank_sufficient_score=0.45)
        self.assertEqual("sufficient", report.sufficiency)
        self.assertEqual([1], report.supported_indices())

    def test_requiring_more_supporting_chunks_defers_to_the_grader(self):
        report = self._assess([0.91, 0.2, 0.1, 0.0], rerank_min_supporting_chunks=2)
        self.assertEqual(Certainty.LOW, report.certainty)

    def test_scores_are_recorded_per_chunk_for_the_trace(self):
        report = self._assess([0.91, 0.72, 0.11, 0.04])
        self.assertEqual(0.91, report.chunks[0].score)
        self.assertEqual(0.91, report.chunks[0].signals["cross_encoder"])

    def test_no_documents_abstains(self):
        config = load_profile("base").rag.model_copy(update={"rerank_cross_encoder_enabled": True})
        ctx = AssessmentContext(question="q", docs=[], config=config)
        self.assertIsNone(CrossEncoderAssessor().assess(ctx))


class ScoreNormalisationTests(unittest.TestCase):
    def test_probabilities_pass_through(self):
        self.assertEqual(0.73, _to_unit(0.73))

    def test_out_of_range_logits_are_squashed(self):
        """Only values outside [0, 1] can be identified as logits. A logit of 0.4 is
        indistinguishable from a probability of 0.4, so in-range values always pass
        through — which is correct for sentence-transformers, whose CrossEncoder already
        applies a sigmoid for single-label models. This is a safety net for checkpoints
        that do not."""
        self.assertGreater(_to_unit(4.0), 0.9)
        self.assertLess(_to_unit(-4.0), 0.1)
        self.assertEqual(0.4, _to_unit(0.4))

    def test_garbage_scores_to_zero_rather_than_raising(self):
        self.assertEqual(0.0, _to_unit(None))
        self.assertEqual(0.0, _to_unit("banana"))


if __name__ == "__main__":
    unittest.main()
