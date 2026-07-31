"""The scope catalogue: summarisation, the derived floor, and the two scope rungs.

Two properties carry the safety of this subsystem:

  * the catalogue rung may ADMIT but never REFUSE — only a model that read the question
    can end a turn;
  * the cut point is DERIVED from the corpus, so there is no number to tune and nothing
    to go stale on the next re-index.

Most of these tests exist to pin those.
"""
import unittest
from unittest.mock import patch

from backend.chat.signals import RequestSignals, Scope, SignalContext
from backend.indexing.section_summary import (
    SectionRecord,
    content_hash,
    corpus_catalogue,
    plan_sections,
    summarise_section,
)
from backend.profiles.registry import load_profile
from backend.rag.evidence import Certainty
from backend.rag.scope_detector import (
    CatalogueScopeDetector,
    ScopeIndexStore,
    ScopeModelDetector,
)
from backend.rag.scope_index import ScopeIndex, build_index, derive_floor, percentile

VOCABULARY = ["admissions", "fees", "uniform", "transport"]


def config(**overrides):
    settings = {"scope_index_enabled": True, "scope_topic_vocabulary": VOCABULARY, **overrides}
    return load_profile("base").rag.model_copy(update=settings)


def ctx(question="what is the sports uniform", history=(), **overrides):
    return SignalContext(question=question, history=history, config=config(**overrides))


# Orthogonal-ish unit vectors so similarity is exact and readable.
def unit(index, dimension=4):
    return [1.0 if position == index else 0.0 for position in range(dimension)]


def record(chunk_id, answers, topics=("uniform",)):
    return SectionRecord(chunk_id=chunk_id, answers=list(answers), topics=list(topics))


class ContentHashTests(unittest.TestCase):
    def test_identical_text_reuses_its_summary(self):
        self.assertEqual(content_hash("Hello world"), content_hash("Hello world"))

    def test_reflowed_whitespace_is_the_same_section(self):
        """Re-chunking can reflow a section without changing a word, and re-summarising
        identical prose would spend a call to produce the same answer."""
        self.assertEqual(content_hash("a b   c"), content_hash("a\n b\tc"))

    def test_edited_text_is_a_different_section(self):
        self.assertNotEqual(content_hash("fees are 100"), content_hash("fees are 200"))


class PlanTests(unittest.TestCase):
    def test_unchanged_sections_are_reused(self):
        sections = [{"chunk_id": "s1", "text": "uniform policy"}]
        plan = plan_sections(sections, {"s1": content_hash("uniform policy")})
        self.assertEqual(1, len(plan["reuse"]))
        self.assertEqual(0, len(plan["summarise"]))

    def test_edited_and_new_sections_are_summarised(self):
        sections = [{"chunk_id": "s1", "text": "new text"}, {"chunk_id": "s2", "text": "fresh"}]
        plan = plan_sections(sections, {"s1": "stale-hash"})
        self.assertEqual(2, len(plan["summarise"]))


class SummariseTests(unittest.TestCase):
    def _summarise(self, payload, **kwargs):
        return summarise_section(
            "Uniform policy text", invoke=lambda *a: payload, vocabulary=VOCABULARY, **kwargs
        )

    def test_a_usable_entry_is_returned(self):
        result = self._summarise({
            "summary": "About the uniform.",
            "answers": ["what is the uniform?", "where do I buy it?"],
            "topics": ["uniform"],
        })
        self.assertEqual(2, len(result["answers"]))
        self.assertEqual(["uniform"], result["topics"])

    def test_invented_topics_are_discarded(self):
        """The vocabulary is frozen so the catalogue cannot drift between re-indexes."""
        result = self._summarise({
            "summary": "s", "answers": ["what is it?"], "topics": ["uniform", "quantum_physics"],
        })
        self.assertEqual(["uniform"], result["topics"])

    def test_topic_matching_ignores_case_and_spacing(self):
        result = self._summarise({"summary": "s", "answers": ["what is it?"], "topics": [" Uniform "]})
        self.assertEqual(["uniform"], result["topics"])

    def test_near_duplicate_questions_collapse(self):
        """Each question costs a vector and near-duplicates add no coverage."""
        result = self._summarise({
            "summary": "s",
            "answers": ["What is the uniform?", "what is the uniform", "Where to buy?"],
        })
        self.assertEqual(2, len(result["answers"]))

    def test_a_section_with_no_questions_is_skipped_not_guessed(self):
        """An invented question would teach the gate to accept a subject the corpus
        cannot help with."""
        self.assertIsNone(self._summarise({"summary": "s", "answers": []}))

    def test_a_failed_call_returns_none(self):
        def boom(*a):
            raise RuntimeError("model down")

        self.assertIsNone(
            summarise_section("text", invoke=boom, vocabulary=VOCABULARY)
        )

    def test_the_question_count_is_bounded(self):
        result = self._summarise(
            {"summary": "s", "answers": [f"question {n}?" for n in range(50)]},
            max_questions=5,
        )
        self.assertEqual(5, len(result["answers"]))


class DerivedFloorTests(unittest.TestCase):
    """The cut point is a statistic of the corpus, not a number anyone chose."""

    def test_percentile_interpolates(self):
        self.assertEqual(0.0, percentile([0.0, 1.0], 0))
        self.assertEqual(1.0, percentile([0.0, 1.0], 100))
        self.assertAlmostEqual(0.5, percentile([0.0, 1.0], 50))

    def test_an_empty_or_single_question_corpus_yields_no_floor(self):
        """Zero means "escalate nothing on score alone" — the model stays in charge
        rather than a boundary being invented from one data point."""
        self.assertEqual(0.0, derive_floor([], []))
        self.assertEqual(0.0, derive_floor([unit(0)], ["s1"]))

    def test_questions_are_scored_against_OTHER_sections_only(self):
        """Scoring a question against its own siblings measures how repetitive one
        section is, which says nothing about scope — and would put the floor near 1.0,
        escalating almost every real query."""
        vectors = [unit(0), unit(0), unit(1)]
        chunk_ids = ["s1", "s1", "s2"]
        self.assertEqual(0.0, derive_floor(vectors, chunk_ids, point=50))

    def test_a_cohesive_corpus_yields_a_higher_floor(self):
        """Sections that resemble each other mean an in-scope question scores high, so
        the bar for looking in-scope rises with it."""
        near = [1.0, 0.1, 0.0, 0.0]
        cohesive = derive_floor([unit(0), near], ["s1", "s2"], point=50)
        distinct = derive_floor([unit(0), unit(1)], ["s1", "s2"], point=50)
        self.assertGreater(cohesive, distinct)

    def test_a_single_section_corpus_never_refuses_on_score(self):
        vectors = [unit(0), unit(1), unit(2)]
        self.assertEqual(0.0, derive_floor(vectors, ["s1", "s1", "s1"]))


class BuildIndexTests(unittest.TestCase):
    def _build(self, records, **kwargs):
        lookup = {"what is the uniform?": unit(0), "where do I buy it?": unit(1),
                  "what are the fees?": unit(2), "when is term?": unit(3)}
        return build_index(records, embed=lambda qs: [lookup[q] for q in qs], **kwargs)

    def test_one_vector_per_question_not_per_section(self):
        """Averaging a section's questions into one vector puts the centroid near none
        of them."""
        index = self._build([
            record("s1", ["what is the uniform?", "where do I buy it?"]),
            record("s2", ["what are the fees?"], topics=["fees"]),
        ])
        self.assertEqual(3, len(index.vectors))
        self.assertEqual(2, len(set(index.chunk_ids)))

    def test_matching_is_max_over_questions(self):
        index = self._build([record("s1", ["what is the uniform?", "where do I buy it?"])])
        matches = index.best_matches(unit(1), limit=2)
        self.assertEqual("where do I buy it?", matches[0].question)
        self.assertAlmostEqual(1.0, matches[0].score)

    def test_an_empty_catalogue_is_not_ready(self):
        self.assertFalse(build_index([], embed=lambda qs: []).ready)

    def test_a_mismatched_embedder_abstains_rather_than_misaligning(self):
        """Vectors and questions are positionally paired; a length mismatch would map
        every score to the wrong question."""
        index = build_index([record("s1", ["what is it?", "how much?"])],
                            embed=lambda qs: [unit(0)])
        self.assertFalse(index.ready)

    def test_dimension_mismatches_are_skipped_not_scored(self):
        """A re-index under a different embedding model must not look like every
        question suddenly being off-topic."""
        index = self._build([record("s1", ["what is the uniform?"])])
        self.assertEqual([], index.best_matches([1.0, 0.0], limit=3))


class CatalogueRungTests(unittest.TestCase):
    """Rung 1: may admit, may never refuse."""

    def _detector(self, index):
        store = ScopeIndexStore(builder=lambda: index)
        return CatalogueScopeDetector(store=store)

    def _index(self, floor=0.5):
        return ScopeIndex(
            questions=["what is the uniform?", "what are the fees?"],
            vectors=[unit(0), unit(1)],
            chunk_ids=["s1", "s2"],
            topics=[["uniform"], ["fees"]],
            floor=floor,
        )

    def _detect(self, index, vector, question="q", history=()):
        signals = RequestSignals(question=question)
        with patch("backend.indexing.embedding.embed_query", lambda _t: vector):
            return self._detector(index).detect(ctx(question, history), signals)

    def test_a_strong_match_admits_at_medium(self):
        signals = self._detect(self._index(), unit(0))
        self.assertIs(Scope.IN_DOMAIN, signals.scope)
        self.assertEqual(Certainty.MEDIUM, signals.scope_certainty)

    def test_a_weak_match_rejects_only_at_low(self):
        """LOW cannot end the climb, so the model still gets to look.

        unit(2) is orthogonal to every catalogued question, so it scores 0.0 — the
        shape of a genuinely unrelated query.
        """
        signals = self._detect(self._index(floor=0.5), unit(2))
        self.assertIs(Scope.OUT_OF_DOMAIN, signals.scope)
        self.assertEqual(Certainty.LOW, signals.scope_certainty)

    def test_the_matched_questions_are_carried_for_the_model(self):
        signals = self._detect(self._index(), unit(0))
        self.assertEqual("what is the uniform?", signals.scope_matches[0].question)
        self.assertEqual(["s1", "s2"], signals.candidate_sections)

    def test_it_abstains_when_disabled(self):
        signals = RequestSignals(question="q")
        detector = self._detector(self._index())
        self.assertIsNone(detector.detect(ctx(scope_index_enabled=False), signals))

    def test_it_abstains_without_a_catalogue(self):
        signals = RequestSignals(question="q")
        self.assertIsNone(self._detector(ScopeIndex()).detect(ctx(), signals))

    def test_an_embedding_failure_abstains(self):
        signals = RequestSignals(question="q")
        with patch("backend.indexing.embedding.embed_query", side_effect=RuntimeError("boom")):
            self.assertIsNone(self._detector(self._index()).detect(ctx(), signals))

    def test_a_follow_up_is_scored_with_the_turn_before_it(self):
        captured = {}

        def fake_embed(text):
            captured["text"] = text
            return unit(0)

        signals = RequestSignals(question="what about grade 6")
        history = [{"role": "user", "content": "what are the fees for grade 5"}]
        with patch("backend.indexing.embedding.embed_query", fake_embed):
            self._detector(self._index()).detect(ctx("what about grade 6", history), signals)
        self.assertIn("grade 5", captured["text"])


class ScopeModelRungTests(unittest.TestCase):
    """Rung 2: the only rung that may end a turn."""

    def _detect(self, payload, signals=None):
        detector = ScopeModelDetector(invoke=lambda *a: payload)
        return detector.detect(
            ctx(request_envelope_enabled=True),
            signals or RequestSignals(question="q"),
        )

    def test_it_confirms_out_of_domain_at_high(self):
        signals = self._detect({"scope": "out_of_domain", "reason": "asks about weather"})
        self.assertIs(Scope.OUT_OF_DOMAIN, signals.scope)
        self.assertEqual(Certainty.HIGH, signals.scope_certainty)

    def test_it_rescues_a_tentative_rejection(self):
        prior = RequestSignals(question="q", scope=Scope.OUT_OF_DOMAIN,
                               scope_certainty=Certainty.LOW)
        signals = self._detect({"scope": "in_domain"}, prior)
        self.assertIs(Scope.IN_DOMAIN, signals.scope)

    def test_disclosed_details_override_a_rejection(self):
        """Enforced in code, not asked for in the prompt: someone answering "he is 9"
        is continuing a conversation."""
        signals = self._detect({"scope": "out_of_domain", "personal_data": ["child_age"]})
        self.assertIs(Scope.IN_DOMAIN, signals.scope)

    def test_it_abstains_when_disabled(self):
        detector = ScopeModelDetector(invoke=lambda *a: {"scope": "out_of_domain"})
        self.assertIsNone(detector.detect(ctx(), RequestSignals(question="q")))

    def test_an_unusable_verdict_leaves_scope_alone(self):
        signals = self._detect({"scope": "maybe"})
        self.assertIs(Scope.UNKNOWN, signals.scope)


class LadderCompositionTests(unittest.TestCase):
    def test_the_catalogue_supersedes_the_chunk_gate(self):
        from backend.chat.signals import build_ladder

        names = [d.name for d in build_ladder(
            config(scope_index_enabled=True, domain_gate_enabled=True)
        )._detectors]
        self.assertIn("scope_catalogue", names)
        self.assertNotIn("corpus_similarity", names)

    def test_the_chunk_gate_remains_for_deployments_without_a_catalogue(self):
        from backend.chat.signals import build_ladder

        names = [d.name for d in build_ladder(
            config(scope_index_enabled=False, domain_gate_enabled=True)
        )._detectors]
        self.assertIn("corpus_similarity", names)

    def test_the_scope_model_needs_both_switches(self):
        from backend.chat.signals import build_ladder

        names = [d.name for d in build_ladder(
            config(scope_index_enabled=True, request_envelope_enabled=False)
        )._detectors]
        self.assertNotIn("scope_model", names)


class StoreTests(unittest.TestCase):
    def test_a_failed_build_is_not_retried_every_request(self):
        calls = []

        def explode():
            calls.append(1)
            raise RuntimeError("db down")

        store = ScopeIndexStore(builder=explode)
        store.get(); store.get(); store.get()
        self.assertEqual(1, len(calls))
        self.assertFalse(store.get().ready)

    def test_invalidate_rebuilds_after_a_reindex(self):
        calls = []

        def builder():
            calls.append(1)
            return ScopeIndex(questions=["q"], vectors=[unit(0)], chunk_ids=["s1"], topics=[[]])

        store = ScopeIndexStore(builder=builder)
        store.get(); store.get()
        self.assertEqual(1, len(calls))
        store.invalidate()
        store.get()
        self.assertEqual(2, len(calls))


class CatalogueTests(unittest.TestCase):
    def test_topics_are_deduped_in_order(self):
        records = [record("s1", ["what is it?"], ["uniform"]),
                   record("s2", ["how much is it?"], ["fees", "uniform"])]
        self.assertEqual("uniform, fees", corpus_catalogue(records))

    def test_a_record_without_questions_is_unusable(self):
        self.assertFalse(SectionRecord(chunk_id="s1").usable)
        self.assertTrue(record("s1", ["what is it?"]).usable)


if __name__ == "__main__":
    unittest.main()


class CompletenessTests(unittest.TestCase):
    """A partial catalogue fails silently by nature.

    A section that did not summarise leaves no error behind once the run ends. The gate
    simply has a hole: questions about that part of the corpus score low and escalate
    forever, which looks like the model being cautious rather than like a missing row.
    So completeness is asserted explicitly rather than assumed.
    """

    def _verify(self, records, expected):
        from backend.indexing.build_scope_index import verify

        return verify(records, expected)

    def _complete(self, chunk_id="s1"):
        return SectionRecord(
            chunk_id=chunk_id,
            answers=["what is it?", "how much?"],
            question_vectors=[unit(0), unit(1)],
        )

    def test_a_full_catalogue_is_complete(self):
        report = self._verify([self._complete("s1"), self._complete("s2")], expected=2)
        self.assertTrue(report["complete"])
        self.assertEqual(4, report["questions"])

    def test_a_missing_section_is_reported(self):
        report = self._verify([self._complete("s1")], expected=3)
        self.assertFalse(report["complete"])
        self.assertIn("2 section(s) have no catalogue entry", report["problems"][0])

    def test_a_section_without_questions_is_reported(self):
        report = self._verify([self._complete("s1"), SectionRecord(chunk_id="s2")], expected=2)
        self.assertFalse(report["complete"])
        self.assertTrue(any("no questions: s2" in p for p in report["problems"]))

    def test_questions_without_stored_vectors_are_reported(self):
        """The case that actually happened: a transient failure left the previous
        summary in place, so the content hash said "nothing to do" while the vectors
        were never written."""
        stale = SectionRecord(chunk_id="s2", answers=["what is it?"], question_vectors=[])
        report = self._verify([self._complete("s1"), stale], expected=2)
        self.assertFalse(report["complete"])
        self.assertTrue(any("no stored vectors: s2" in p for p in report["problems"]))

    def test_a_partial_vector_list_is_not_treated_as_present(self):
        """Positional pairing means a short list would misalign questions with other
        questions' vectors."""
        partial = SectionRecord(
            chunk_id="s2", answers=["a?", "b?"], question_vectors=[unit(0)],
        )
        self.assertFalse(self._verify([partial], expected=1)["complete"])


class VectorReuseTests(unittest.TestCase):
    def test_stored_vectors_are_used_without_re_embedding(self):
        def never(questions):
            raise AssertionError(f"re-embedded {len(questions)} question(s)")

        records = [SectionRecord(chunk_id="s1", answers=["a?", "b?"],
                                 question_vectors=[unit(0), unit(1)])]
        index = build_index(records, embed=never)
        self.assertEqual(2, len(index.vectors))

    def test_only_the_records_missing_vectors_are_embedded(self):
        seen = []

        def embed(questions):
            seen.extend(questions)
            return [unit(2) for _ in questions]

        records = [
            SectionRecord(chunk_id="s1", answers=["stored?"], question_vectors=[unit(0)]),
            SectionRecord(chunk_id="s2", answers=["fresh?"]),
        ]
        build_index(records, embed=embed)
        self.assertEqual(["fresh?"], seen)

    def test_a_stale_vector_list_is_discarded_whole(self):
        """Rather than pairing the first N questions with vectors and leaving the rest
        unmatched, which would silently score a question against another's vector."""
        seen = []

        def embed(questions):
            seen.extend(questions)
            return [unit(3) for _ in questions]

        records = [SectionRecord(chunk_id="s1", answers=["a?", "b?"],
                                 question_vectors=[unit(0)])]
        build_index(records, embed=embed)
        self.assertEqual(["a?", "b?"], seen)
