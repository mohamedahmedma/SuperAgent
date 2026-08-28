"""B1: a retrieval backend outage must surface as retrieval_error (static, no LLM),
never as no_knowledge — an unreachable Milvus says nothing about corpus coverage."""
import sys
import types
import unittest
from unittest.mock import patch

from backend.chat.request_context import ChatRequestContext
from tests.general.test_rag_short_circuit import (
    FakeStructuredModel,
    enable_complexity_planning,
    load_pipeline,
    _doc,
    _meta,
)


def _failed_meta(reason="retrieve_failed"):
    return {
        "retrieval_mode": "failed",
        "retrieval_pipeline": "recall_merge_rerank",
        "retrieval_error": reason,
        "recall_count": 0,
        "retrieval_empty": True,
    }


def _forbidden_grader(schema, prompt):
    raise AssertionError("grader LLM must not be called during a retrieval outage")


class RetrievalOutageTests(unittest.TestCase):
    def _ctx(self):
        return ChatRequestContext.for_sync(user_id="u", session_id="s")

    def test_simple_outage_short_circuits_without_grader_or_rewrite(self):
        calls = {"retrieve": 0, "rewrite": 0}

        def retrieve(query, top_k=5):
            calls["retrieve"] += 1
            return {"docs": [], "meta": _failed_meta()}

        def rewrite(query):
            calls["rewrite"] += 1
            return {}

        pipeline = load_pipeline(retrieve_documents=retrieve, rewrite_query_once=rewrite)
        pipeline._get_complexity_model = lambda: FakeStructuredModel(
            lambda schema, prompt: {"complexity": "simple", "reason": "unit"}
        )
        pipeline._get_grader_model = lambda: FakeStructuredModel(_forbidden_grader)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("question during backend outage", ctx)
        finally:
            ctx.close()

        self.assertEqual("retrieval_error", result.get("retrieval_status"))
        self.assertEqual("retrieval_error", result.get("route"))
        self.assertEqual([], result.get("docs"))
        self.assertEqual(1, calls["retrieve"])
        self.assertEqual(0, calls["rewrite"])
        trace = result.get("rag_trace", {})
        self.assertEqual("retrieval_error", trace.get("retrieval_status"))
        self.assertEqual("knowledge_base_unreachable", trace.get("evidence_reason"))
        self.assertNotIn("no_knowledge", (trace.get("retrieval_status"), result.get("route")))
        self.assertIsNone(result.get("hitl_resume_state"))

    def test_outage_during_rewritten_retrieval_short_circuits_second_grade(self):
        calls = {"retrieve": [], "rewrite": 0, "grade": 0}

        def retrieve(query, top_k=5):
            calls["retrieve"].append(query)
            if query.startswith("rewritten"):
                return {"docs": [], "meta": _failed_meta()}
            return {"docs": [_doc("weak evidence")], "meta": _meta(1)}

        def rewrite(query):
            calls["rewrite"] += 1
            return {
                "rewrite_method": "step_back",
                "step_back_question": "broader question",
                "hyde_document": "",
                "rewritten_query": f"rewritten {query}",
            }

        def grade(schema, prompt):
            calls["grade"] += 1
            return {
                "relevance": "weak",
                "answerability": "partial",
                "ambiguity": "none",
                "route": "rewrite",
                "confidence": 0.4,
            }

        pipeline = load_pipeline(retrieve_documents=retrieve, rewrite_query_once=rewrite)
        pipeline._get_complexity_model = lambda: FakeStructuredModel(
            lambda schema, prompt: {"complexity": "simple", "reason": "unit"}
        )
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("question with flaky backend", ctx)
        finally:
            ctx.close()

        self.assertEqual(2, len(calls["retrieve"]))
        self.assertEqual(1, calls["rewrite"])
        self.assertEqual(1, calls["grade"])
        self.assertEqual("retrieval_error", result.get("retrieval_status"))
        self.assertEqual([], result.get("docs"))

    def test_complex_all_subs_outage_synthesizes_retrieval_error(self):
        def retrieve(query, top_k=5):
            return {"docs": [], "meta": _failed_meta()}

        def complexity(schema, prompt):
            return {
                "complexity": "complex",
                "reason": "unit",
                "sub_questions": ["first sub", "second sub"],
            }

        pipeline = enable_complexity_planning(load_pipeline(retrieve_documents=retrieve))
        pipeline._get_complexity_model = lambda: FakeStructuredModel(complexity)
        pipeline._get_grader_model = lambda: FakeStructuredModel(_forbidden_grader)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("complex question during outage", ctx)
        finally:
            ctx.close()

        self.assertEqual("retrieval_error", result.get("retrieval_status"))
        self.assertEqual("retrieval_error", result.get("route"))
        self.assertEqual([], result.get("docs"))

    def test_complex_partial_outage_still_answers_from_healthy_sub(self):
        def retrieve(query, top_k=5):
            if query == "healthy sub":
                return {"docs": [_doc("solid evidence", "healthy")], "meta": _meta(1)}
            return {"docs": [], "meta": _failed_meta()}

        def complexity(schema, prompt):
            return {
                "complexity": "complex",
                "reason": "unit",
                "sub_questions": ["healthy sub", "outage sub"],
            }

        def grade(schema, prompt):
            return {
                "relevance": "strong",
                "answerability": "sufficient",
                "ambiguity": "none",
                "route": "answer",
                "confidence": 0.9,
            }

        pipeline = enable_complexity_planning(load_pipeline(retrieve_documents=retrieve))
        pipeline._get_complexity_model = lambda: FakeStructuredModel(complexity)
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("complex question with one failing sub", ctx)
        finally:
            ctx.close()

        self.assertEqual(1, len(result.get("docs", [])))
        self.assertEqual("answerable", result.get("retrieval_status"))
        self.assertEqual("answer", result.get("route"))

    def test_hitl_resume_outage_returns_retrieval_error(self):
        def retrieve(query, top_k=5):
            return {"docs": [], "meta": _failed_meta("embedding_failed")}

        pipeline = load_pipeline(retrieve_documents=retrieve)
        pipeline._get_grader_model = lambda: FakeStructuredModel(_forbidden_grader)
        resume_state = {
            "question": "What is this character's element?",
            "route": "clarify",
            "retrieval_status": "needs_clarification",
        }

        ctx = self._ctx()
        try:
            result = pipeline.resume_rag_from_hitl(resume_state, "Danjin", ctx)
        finally:
            ctx.close()

        self.assertEqual("retrieval_error", result.get("retrieval_status"))
        self.assertEqual([], result.get("docs"))
        self.assertNotIn("hitl_resume_state", result)

    def test_trace_normalization_preserves_outage_fields(self):
        from backend.schemas.chat import normalize_rag_trace

        trace = {
            "tool_used": True,
            "retrieval_status": "retrieval_error",
            "route": "retrieval_error",
            "retrieval_error": "retrieve_failed",
            "retrieval_mode": "failed",
        }
        normalized = normalize_rag_trace(trace)
        self.assertEqual("retrieval_error", normalized.get("retrieval_status"))
        self.assertEqual("retrieval_error", normalized.get("route"))
        # The failure reason must survive into stored traces/frontend, not be
        # silently dropped by schema filtering.
        self.assertEqual("retrieve_failed", normalized.get("retrieval_error"))

    def test_knowledge_tool_returns_static_retrieval_error_message(self):
        from backend.tools.knowledge import make_search_knowledge_base

        fake_pipeline = types.ModuleType("backend.rag.pipeline")
        fake_pipeline.run_rag_graph = lambda query, ctx: {
            "docs": [],
            "rag_trace": {
                "retrieval_status": "retrieval_error",
                "route": "retrieval_error",
            },
        }

        ctx = self._ctx()
        try:
            tool = make_search_knowledge_base(ctx)
            with patch.dict(sys.modules, {"backend.rag.pipeline": fake_pipeline}):
                message = tool.invoke({"query": "anything"})
        finally:
            ctx.close()

        self.assertTrue(message.startswith("RETRIEVAL_ERROR:"))
        self.assertNotIn("NO_KNOWLEDGE", message)
        self.assertIn("temporary technical issue", message)


if __name__ == "__main__":
    unittest.main()
