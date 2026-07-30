"""Adaptive context sizing.

Trimming too little costs tokens; trimming too much costs correctness, and only one of
those is recoverable. These tests are weighted toward proving the set is NOT trimmed
whenever there is any doubt.
"""
import unittest

from backend.profiles.registry import load_profile
from backend.rag.context_selection import (
    ContextSelection,
    select_context,
    wants_exhaustive_answer,
)


def rag_config(**overrides):
    """Adaptive mode explicitly, since the shipped default is off. These tests cover the
    mechanism; `test_the_shipped_default_is_off` covers what is actually enabled."""
    settings = {"context_selection_mode": "adaptive", **overrides}
    return load_profile("base").rag.model_copy(update=settings)


def doc(text, score=0.5):
    return {"text": text, "score": score, "filename": "kb.docx", "page_number": 1}


# Long enough to clear context_min_chunk_chars, and it actually answers the question.
ANSWER = doc(
    "The school partners are Cairo University, the British Council and the Ministry "
    "of Education, each supporting a different part of the academic programme."
)
FILLER = doc(
    "Partner organisations are reviewed annually by the school board, which meets "
    "each September to consider renewals and new applications from institutions."
)


class ExhaustiveIntentTests(unittest.TestCase):
    def test_breadth_questions_are_recognised(self):
        for question in ("list all the partners", "compare grade 5 and grade 6 fees",
                         "how many teachers are there", "what is the difference between them"):
            self.assertTrue(wants_exhaustive_answer(question), question)

    def test_arabic_breadth_questions_are_recognised(self):
        self.assertTrue(wants_exhaustive_answer("اذكر جميع الشركاء"))
        self.assertTrue(wants_exhaustive_answer("قارن بين الصف الخامس والسادس"))

    def test_single_fact_questions_are_not(self):
        for question in ("what is the school sports uniform", "who is the principal"):
            self.assertFalse(wants_exhaustive_answer(question), question)


class TrimmingTests(unittest.TestCase):
    def setUp(self):
        self.config = rag_config()

    def test_one_chunk_is_enough_when_it_covers_the_question(self):
        docs = [ANSWER, FILLER, FILLER, FILLER]
        kept, selection = select_context("what are the school partners", docs, self.config)
        self.assertEqual(1, len(kept))
        self.assertTrue(selection.trimmed)
        self.assertEqual(1.0, selection.coverage)
        self.assertEqual(4, selection.available)

    def test_order_is_preserved_so_citation_markers_stay_aligned(self):
        docs = [ANSWER, FILLER, FILLER, FILLER]
        kept, _ = select_context("what are the school partners", docs, self.config)
        self.assertEqual(docs[0]["text"], kept[0]["text"])

    def test_more_chunks_are_kept_until_the_question_is_covered(self):
        docs = [
            doc("The school partners include Cairo University and several local institutions "
                "that support the academic programme throughout the year."),
            doc("Uniform requirements for sports are navy shorts and a white polo shirt, "
                "which must be worn for every physical education lesson."),
        ] * 2
        kept, selection = select_context("school partners sports uniform", docs, self.config)
        self.assertGreaterEqual(len(kept), 2)
        self.assertEqual(1.0, selection.coverage)


class NoTrimmingGuardTests(unittest.TestCase):
    """Every case where the full set must survive."""

    def setUp(self):
        self.config = rag_config()

    def test_breadth_questions_keep_everything(self):
        docs = [ANSWER, FILLER, FILLER, FILLER]
        kept, selection = select_context("list all the school partners", docs, self.config)
        self.assertEqual(4, len(kept))
        self.assertFalse(selection.trimmed)
        self.assertIn("complete set", selection.reasons[0])

    def test_a_heading_can_never_end_selection(self):
        """A heading matches its question's vocabulary perfectly and answers nothing.
        Roughly a fifth of a real corpus is these short heading-only leaves."""
        docs = [doc("School Partners"), ANSWER, FILLER, FILLER]
        kept, _ = select_context("what are the school partners", docs, self.config)
        self.assertGreater(len(kept), 1)
        self.assertIn("Cairo University", kept[1]["text"])

    def test_weak_retrieval_keeps_everything(self):
        docs = [doc("The uniform is navy blue with gold accents and must be worn daily.")] * 4
        kept, selection = select_context("scholarship application deadlines abroad", docs, self.config)
        self.assertEqual(4, len(kept))
        self.assertFalse(selection.trimmed)
        self.assertIn("not confidently on target", selection.reasons[0])

    def test_too_few_chunks_to_trim(self):
        kept, selection = select_context("what are the school partners", [ANSWER], self.config)
        self.assertEqual(1, len(kept))
        self.assertFalse(selection.trimmed)

    def test_no_documents_is_handled(self):
        kept, selection = select_context("anything", [], self.config)
        self.assertEqual([], kept)
        self.assertEqual(0, selection.available)

    def test_a_question_of_only_stop_words_keeps_everything(self):
        docs = [ANSWER, FILLER, FILLER, FILLER]
        kept, selection = select_context("what is it", docs, self.config)
        self.assertEqual(4, len(kept))
        self.assertFalse(selection.trimmed)

    def test_off_mode_keeps_everything(self):
        docs = [ANSWER, FILLER, FILLER, FILLER]
        kept, selection = select_context("what are the school partners", docs,
                                         rag_config(context_selection_mode="off"))
        self.assertEqual(4, len(kept))
        self.assertFalse(selection.trimmed)
        self.assertIn("context_selection_mode=off", selection.reasons)


class ConfigurationTests(unittest.TestCase):
    def test_the_floor_is_respected(self):
        docs = [ANSWER, FILLER, FILLER, FILLER]
        kept, _ = select_context("what are the school partners", docs,
                                 rag_config(context_min_chunks=3))
        self.assertEqual(3, len(kept))

    def test_a_stricter_coverage_target_keeps_more(self):
        """Nothing can exceed 100% coverage, so an unreachable target keeps the set."""
        docs = [ANSWER, FILLER, FILLER, FILLER]
        kept, _ = select_context("what are the school partners", docs,
                                 rag_config(context_target_coverage=1.01))
        self.assertEqual(4, len(kept))

    def test_the_shipped_default_is_off(self):
        """Trimming after the grader ruled on the full set answers from narrower
        evidence than was judged sufficient. Off until the grader reports which chunks
        carried its judgement."""
        config = load_profile("base").rag
        self.assertEqual("off", config.context_selection_mode)
        self.assertEqual(1, config.context_min_chunks)
        self.assertEqual(1.0, config.context_target_coverage)
        self.assertEqual(120, config.context_min_chunk_chars)


class TraceTests(unittest.TestCase):
    def test_the_decision_is_recorded(self):
        trace = ContextSelection(kept=1, available=4, coverage=1.0, trimmed=True,
                                 reasons=["1 of 4 chunks"]).as_trace()
        self.assertEqual(1, trace["context_chunks_kept"])
        self.assertEqual(4, trace["context_chunks_available"])
        self.assertTrue(trace["context_trimmed"])
        self.assertEqual(1.0, trace["context_coverage"])

    def test_the_trace_schema_carries_the_new_fields(self):
        from backend.schemas.chat import normalize_rag_trace

        trace = normalize_rag_trace({
            "context_chunks_kept": 1, "context_chunks_available": 4,
            "context_trimmed": True, "context_coverage": 1.0,
            "context_selection_reason": "1 of 4 chunks",
        })
        self.assertEqual(1, trace["context_chunks_kept"])
        self.assertTrue(trace["context_trimmed"])


class PipelineIntegrationTests(unittest.TestCase):
    def test_trimming_rewrites_context_and_the_cited_chunk_list(self):
        """retrieved_chunks must match what the answer saw: citation markers and asset
        attribution both index into it, so a stale full list mis-attributes images."""
        from unittest.mock import patch

        import backend.rag.pipeline as pipeline

        class SilentContext:
            def emit_rag_step(self, *args, **kwargs):
                pass

        docs = [ANSWER, FILLER, FILLER, FILLER]
        state = {
            "question": "what are the school partners",
            "docs": docs,
            "context": pipeline._format_docs(docs),
            "rag_trace": {"retrieved_chunks": docs, "initial_retrieved_chunks": docs},
            "request_context": SilentContext(),
        }
        # grading_mode=never keeps the node offline; the trimming path is what is under
        # test here, not the grader.
        config = rag_config(grading_mode="never")
        with patch.object(pipeline, "_RAG", config):
            update = pipeline.grade_documents_node(state)

        self.assertEqual("answer", update["route"])
        self.assertEqual(1, len(update["docs"]))
        self.assertEqual(1, len(update["rag_trace"]["retrieved_chunks"]))
        self.assertNotIn("[2]", update["context"])
        # The full set stays available for the trace panel.
        self.assertEqual(4, len(update["rag_trace"]["initial_retrieved_chunks"]))
        self.assertTrue(update["rag_trace"]["context_trimmed"])


if __name__ == "__main__":
    unittest.main()
