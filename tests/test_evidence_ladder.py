"""The evidence ladder, and the policies that read its report.

The architectural claim under test: every decision about retrieved evidence reads ONE
report, and the report is honest about how it was produced. So these tests care less
about individual signals than about two invariants:

  * a cheap rung can never establish something only a model could know;
  * a policy below its certainty floor degrades rather than guessing.
"""
import unittest
from unittest.mock import patch

from backend.profiles.registry import load_profile
from backend.rag.evidence import (
    AssessmentContext,
    AssessmentLadder,
    Certainty,
    ChunkAssessment,
    EvidenceReport,
    LexicalAssessor,
    StructuralAssessor,
    build_ladder,
    parse_certainty,
)
from backend.rag.policy import decide_route, select_context_indices


def rag_config(**overrides):
    return load_profile("base").rag.model_copy(update=overrides) if overrides else load_profile("base").rag


def doc(text="The school partners are Cairo University and the British Council."):
    return {"text": text, "filename": "kb.docx", "page_number": 1}


def ctx(question="what are the school partners", docs=None, config=None):
    return AssessmentContext(
        question=question,
        docs=docs if docs is not None else [doc()] * 4,
        config=config or rag_config(),
    )


class StubAssessor:
    """A rung with a fixed verdict, for testing the ladder rather than any signal."""

    def __init__(self, name, certainty, report=None, boom=False):
        self.name = name
        self.certainty = certainty
        self._report = report
        self._boom = boom
        self.calls = 0

    def assess(self, context):
        self.calls += 1
        if self._boom:
            raise RuntimeError("assessor exploded")
        return self._report


def report_at(certainty, **kwargs):
    fields = {
        "certainty": certainty,
        "relevance": "strong",
        "sufficiency": "sufficient",
        "preferred_route": "answer",
        "chunks": [ChunkAssessment(index=1)],
    }
    fields.update(kwargs)
    return EvidenceReport(**fields)


class CertaintyTests(unittest.TestCase):
    def test_levels_are_ordered_so_policies_can_state_a_floor(self):
        self.assertLess(Certainty.NONE, Certainty.LOW)
        self.assertLess(Certainty.LOW, Certainty.MEDIUM)
        self.assertLess(Certainty.MEDIUM, Certainty.HIGH)

    def test_parsed_from_profile_strings(self):
        self.assertEqual(Certainty.MEDIUM, parse_certainty("medium"))
        self.assertEqual(Certainty.HIGH, parse_certainty("HIGH"))
        self.assertEqual(Certainty.HIGH, parse_certainty("nonsense"))
        self.assertEqual(Certainty.LOW, parse_certainty(None, default=Certainty.LOW))


class StructuralAssessorTests(unittest.TestCase):
    def test_no_documents_is_conclusive_without_a_model(self):
        report = StructuralAssessor().assess(ctx(docs=[]))
        self.assertEqual(Certainty.HIGH, report.certainty)
        self.assertEqual("none", report.relevance)
        self.assertEqual("no_knowledge", report.preferred_route)

    def test_it_abstains_when_documents_exist(self):
        self.assertIsNone(StructuralAssessor().assess(ctx()))


class LexicalAssessorTests(unittest.TestCase):
    """The rung whose over-reach caused the rollback. It must not conclude."""

    def test_it_reports_numbers_but_claims_nothing(self):
        report = LexicalAssessor().assess(ctx())
        self.assertEqual(Certainty.LOW, report.certainty)
        self.assertEqual("unknown", report.relevance)
        self.assertEqual("unknown", report.sufficiency)
        self.assertIsNone(report.preferred_route)

    def test_perfect_overlap_is_still_only_low_certainty(self):
        """Word overlap is not meaning, however complete it looks."""
        report = LexicalAssessor().assess(ctx(question="school partners"))
        self.assertEqual(1.0, report.confidence)
        self.assertEqual(Certainty.LOW, report.certainty)

    def test_it_can_never_invent_an_ambiguity(self):
        report = LexicalAssessor().assess(ctx())
        self.assertEqual("none", report.ambiguity)
        self.assertEqual([], report.hitl_options)

    def test_per_chunk_signals_are_recorded_for_the_trace(self):
        report = LexicalAssessor().assess(ctx())
        self.assertEqual(4, len(report.chunks))
        self.assertIn("lexical_coverage", report.chunks[0].signals)

    def test_no_chunk_is_marked_supported(self):
        """Trimming keys off `supported`; a lexical rung must leave it unknown."""
        report = LexicalAssessor().assess(ctx())
        self.assertEqual([], report.supported_indices())


class LadderTests(unittest.TestCase):
    def test_it_stops_as_soon_as_the_requirement_is_met(self):
        cheap = StubAssessor("cheap", Certainty.MEDIUM, report_at(Certainty.MEDIUM))
        expensive = StubAssessor("expensive", Certainty.HIGH, report_at(Certainty.HIGH))
        ladder = AssessmentLadder([expensive, cheap], required=Certainty.MEDIUM)

        report = ladder.run(ctx())
        self.assertEqual(Certainty.MEDIUM, report.certainty)
        self.assertEqual(1, cheap.calls)
        self.assertEqual(0, expensive.calls, "the expensive rung must not run")

    def test_it_climbs_when_the_cheap_rung_cannot_conclude(self):
        cheap = StubAssessor("cheap", Certainty.MEDIUM, None)
        expensive = StubAssessor("expensive", Certainty.HIGH, report_at(Certainty.HIGH))
        report = AssessmentLadder([cheap, expensive], required=Certainty.HIGH).run(ctx())

        self.assertEqual(Certainty.HIGH, report.certainty)
        self.assertEqual(1, expensive.calls)

    def test_rungs_run_cheapest_first_regardless_of_registration_order(self):
        order = []

        class Recorder(StubAssessor):
            def assess(self, context):
                order.append(self.name)
                return None

        ladder = AssessmentLadder(
            [Recorder("high", Certainty.HIGH), Recorder("low", Certainty.LOW),
             Recorder("mid", Certainty.MEDIUM)],
            required=Certainty.HIGH,
        )
        ladder.run(ctx())
        self.assertEqual(["low", "mid", "high"], order)

    def test_a_failing_rung_degrades_certainty_and_never_the_request(self):
        broken = StubAssessor("broken", Certainty.MEDIUM, boom=True)
        report = AssessmentLadder([broken], required=Certainty.MEDIUM).run(ctx())
        self.assertEqual(Certainty.NONE, report.certainty)
        self.assertNotIn("broken", report.assessed_by)

    def test_a_failing_rung_does_not_stop_the_climb(self):
        broken = StubAssessor("broken", Certainty.MEDIUM, boom=True)
        grader = StubAssessor("grader", Certainty.HIGH, report_at(Certainty.HIGH))
        report = AssessmentLadder([broken, grader], required=Certainty.HIGH).run(ctx())
        self.assertEqual(Certainty.HIGH, report.certainty)
        self.assertEqual(["grader"], report.assessed_by)

    def test_provenance_names_every_rung_that_contributed(self):
        low = StubAssessor("lex", Certainty.LOW, EvidenceReport(
            certainty=Certainty.LOW, confidence=0.4,
            chunks=[ChunkAssessment(index=1, signals={"lexical_coverage": 0.4})],
            assessed_by=["lex"], reasons=["lexical coverage 0.40"]))
        high = StubAssessor("grader", Certainty.HIGH, report_at(
            Certainty.HIGH, assessed_by=["grader"], reasons=["grader routed to answer"]))

        report = AssessmentLadder([low, high], required=Certainty.HIGH).run(ctx())
        self.assertEqual(["lex", "grader"], report.assessed_by)
        # The whole climb is visible, not just the final step.
        self.assertIn("lexical coverage 0.40", report.reasons)
        self.assertIn("grader routed to answer", report.reasons)

    def test_cheap_per_chunk_signals_survive_the_merge(self):
        low = StubAssessor("lex", Certainty.LOW, EvidenceReport(
            certainty=Certainty.LOW,
            chunks=[ChunkAssessment(index=1, signals={"lexical_coverage": 0.9})],
            assessed_by=["lex"]))
        high = StubAssessor("grader", Certainty.HIGH, report_at(
            Certainty.HIGH, chunks=[ChunkAssessment(index=1, supported=True)],
            assessed_by=["grader"]))

        report = AssessmentLadder([low, high], required=Certainty.HIGH).run(ctx())
        chunk = report.chunks[0]
        self.assertTrue(chunk.supported)
        self.assertEqual(0.9, chunk.signals["lexical_coverage"])

    def test_the_built_ladder_honours_profile_config(self):
        ladder = build_ladder(rag_config(evidence_lexical_enabled=False))
        self.assertEqual(["structural"], [a.name for a in ladder._assessors])

        ladder = build_ladder(rag_config(evidence_lexical_enabled=True))
        self.assertIn("lexical", [a.name for a in ladder._assessors])


class RoutePolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = rag_config()
        self.kwargs = {"has_docs": True, "rewrite_count": 0, "is_sub_agent": False,
                       "config": self.config}

    def test_sufficient_evidence_answers(self):
        route, _ = decide_route(report_at(Certainty.HIGH), **self.kwargs)
        self.assertEqual("answer", route)

    def test_no_documents_is_no_knowledge(self):
        route, _ = decide_route(report_at(Certainty.HIGH), **{**self.kwargs, "has_docs": False})
        self.assertEqual("no_knowledge", route)

    def test_a_report_below_the_required_certainty_is_never_acted_on(self):
        """Retrieval worked but nothing could judge it. Answering would be unfounded and
        denying would be wrong, so the honest outcome is "try again"."""
        route, reason = decide_route(report_at(Certainty.LOW), **self.kwargs)
        self.assertEqual("retrieval_error", route)
        self.assertIn("low", reason)

    def test_a_lower_requirement_lets_a_cheaper_report_through(self):
        config = rag_config(evidence_required_certainty="medium")
        route, _ = decide_route(report_at(Certainty.MEDIUM), **{**self.kwargs, "config": config})
        self.assertEqual("answer", route)

    def test_ambiguity_routes_to_a_human(self):
        route, _ = decide_route(report_at(Certainty.HIGH, ambiguity="missing_slot"), **self.kwargs)
        self.assertEqual("clarify", route)
        route, _ = decide_route(
            report_at(Certainty.HIGH, ambiguity="multiple_candidates"), **self.kwargs)
        self.assertEqual("scope_select", route)

    def test_a_sub_agent_keeps_partial_evidence_for_synthesis(self):
        report = report_at(Certainty.HIGH, relevance="weak", sufficiency="partial",
                           preferred_route="rewrite")
        route, _ = decide_route(report, **{**self.kwargs, "is_sub_agent": True})
        self.assertEqual("answer", route)

    def test_a_sub_agent_with_nothing_usable_stops(self):
        report = report_at(Certainty.HIGH, relevance="weak", sufficiency="none",
                           preferred_route="rewrite")
        route, _ = decide_route(report, **{**self.kwargs, "is_sub_agent": True})
        self.assertEqual("no_knowledge", route)

    def test_relevance_none_overrides_a_route_the_grader_asked_for(self):
        report = report_at(Certainty.HIGH, relevance="none", preferred_route="answer")
        route, _ = decide_route(report, **self.kwargs)
        self.assertEqual("no_knowledge", route)


class ContextPolicyTests(unittest.TestCase):
    def setUp(self):
        self.docs = [doc()] * 4

    def _report(self, certainty, supported=()):
        chunks = [ChunkAssessment(index=i, supported=(i in supported) if supported else None)
                  for i in range(1, 5)]
        return report_at(certainty, chunks=chunks)

    def test_the_shipped_default_trims_to_the_named_chunks(self):
        """rag_config() here is the profile UNMODIFIED, so this pins what a deployment
        actually does — the tests below force adaptive and cover the mechanism."""
        keep, reason = select_context_indices(
            self._report(Certainty.HIGH, supported=(1,)), self.docs, rag_config())
        self.assertEqual([1], keep)
        self.assertIn("1 of 4", reason)

    def test_a_low_certainty_report_can_never_trim(self):
        """This is the rollback encoded as a rule: lexical signals do not license
        dropping evidence."""
        config = rag_config(context_selection_mode="adaptive")
        keep, reason = select_context_indices(self._report(Certainty.LOW), self.docs, config)
        self.assertIsNone(keep)
        self.assertIn("below", reason)

    def test_trims_to_the_chunks_an_assessment_named(self):
        config = rag_config(context_selection_mode="adaptive")
        keep, reason = select_context_indices(
            self._report(Certainty.HIGH, supported=(1, 3)), self.docs, config)
        self.assertEqual([1, 3], keep)
        self.assertIn("2 of 4", reason)

    def test_no_per_chunk_judgement_means_unknown_not_none(self):
        """An empty supported list must never be read as "no chunk mattered"."""
        config = rag_config(context_selection_mode="adaptive")
        keep, reason = select_context_indices(self._report(Certainty.HIGH), self.docs, config)
        self.assertIsNone(keep)
        self.assertIn("no per-chunk judgement", reason)

    def test_every_chunk_supported_sends_everything(self):
        config = rag_config(context_selection_mode="adaptive")
        keep, _ = select_context_indices(
            self._report(Certainty.HIGH, supported=(1, 2, 3, 4)), self.docs, config)
        self.assertIsNone(keep)

    def test_the_floor_pads_from_the_ranking(self):
        config = rag_config(context_selection_mode="adaptive", context_min_chunks=3)
        keep, _ = select_context_indices(
            self._report(Certainty.HIGH, supported=(2,)), self.docs, config)
        self.assertEqual(3, len(keep))
        self.assertIn(2, keep)


class ReportTraceTests(unittest.TestCase):
    def test_existing_trace_field_names_are_preserved(self):
        """The frontend and any integrating client read these names; the report is an
        internal refactor and must not be a breaking API change."""
        trace = report_at(Certainty.HIGH, chunks=[ChunkAssessment(index=1, supported=True)],
                          confidence=0.9, reasons=["grader said so"],
                          assessed_by=["llm_grader"]).as_trace()
        for key in ("evidence_relevance", "evidence_answerability", "evidence_ambiguity",
                    "evidence_confidence", "evidence_reason", "missing_slots"):
            self.assertIn(key, trace)
        self.assertEqual("sufficient", trace["evidence_answerability"])

    def test_provenance_is_additive(self):
        trace = report_at(Certainty.MEDIUM, assessed_by=["lexical", "cross_encoder"]).as_trace()
        self.assertEqual("medium", trace["evidence_certainty"])
        self.assertEqual(["lexical", "cross_encoder"], trace["evidence_assessed_by"])

    def test_supported_chunks_reach_the_trace(self):
        report = report_at(Certainty.HIGH, chunks=[
            ChunkAssessment(index=1, supported=True),
            ChunkAssessment(index=2, supported=False),
        ])
        self.assertEqual([1], report.as_trace()["evidence_supported_chunks"])


class GraderAssessorTests(unittest.TestCase):
    """The one rung that can name which chunks carried the answer."""

    def _assessor_with(self, grade):
        import backend.rag.pipeline as pipeline

        class FakeStructured:
            def invoke(self, _messages):
                return grade

        class FakeGrader:
            def with_structured_output(self, _schema):
                return FakeStructured()

        return pipeline, FakeGrader()

    def test_supporting_chunks_become_per_chunk_judgements(self):
        import backend.rag.pipeline as pipeline

        grade = pipeline.EvidenceGrade(
            relevance="strong", answerability="sufficient", route="answer",
            confidence=0.9, supporting_chunks=[1, 3],
        )
        module, grader = self._assessor_with(grade)
        with patch.object(module, "_get_grader_model", lambda: grader):
            report = module.LLMGraderAssessor().assess(ctx())

        self.assertEqual(Certainty.HIGH, report.certainty)
        self.assertEqual([1, 3], report.supported_indices())

    def test_an_empty_supporting_list_leaves_every_chunk_unjudged(self):
        """"It did not tell us" and "it excluded this chunk" are different facts, and
        conflating them would silently drop evidence."""
        import backend.rag.pipeline as pipeline

        grade = pipeline.EvidenceGrade(
            relevance="strong", answerability="sufficient", route="answer",
            supporting_chunks=[],
        )
        module, grader = self._assessor_with(grade)
        with patch.object(module, "_get_grader_model", lambda: grader):
            report = module.LLMGraderAssessor().assess(ctx())

        self.assertEqual([], report.supported_indices())
        self.assertTrue(all(chunk.supported is None for chunk in report.chunks))

    def test_out_of_range_chunk_numbers_are_ignored(self):
        import backend.rag.pipeline as pipeline

        grade = pipeline.EvidenceGrade(
            relevance="strong", answerability="sufficient", route="answer",
            supporting_chunks=[2, 99, 0, -1],
        )
        module, grader = self._assessor_with(grade)
        with patch.object(module, "_get_grader_model", lambda: grader):
            report = module.LLMGraderAssessor().assess(ctx())

        self.assertEqual([2], report.supported_indices())


if __name__ == "__main__":
    unittest.main()
