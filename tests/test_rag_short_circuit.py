import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Pre-import pipeline.py's heavy dependencies so they are already in sys.modules when
# load_pipeline's patch.dict snapshots it. Otherwise the restore on exit evicts them and
# the next exec re-initializes native modules (uuid_utils via langchain_core), which
# PyO3 forbids twice per process.
from langchain.chat_models import init_chat_model  # noqa: F401
from langgraph.graph import StateGraph  # noqa: F401
from langgraph.types import Send  # noqa: F401

from backend.chat.request_context import ChatRequestContext
from backend.schemas.chat import HitlResumeState  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeStructuredInvoker:
    def __init__(self, schema, handler):
        self.schema = schema
        self.handler = handler

    def invoke(self, messages):
        content = messages[0]["content"] if messages and isinstance(messages[0], dict) else str(messages)
        payload = self.handler(self.schema, content)
        return self.schema(**payload)


class FakeStructuredModel:
    def __init__(self, handler):
        self.handler = handler

    def with_structured_output(self, schema):
        return FakeStructuredInvoker(schema, self.handler)


def _dedupe_documents(docs):
    seen = set()
    out = []
    for doc in docs:
        key = doc.get("chunk_id") or doc.get("text")
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def load_pipeline(
    *,
    retrieve_documents,
    rewrite_query_once=None,
):
    fake_rag = types.ModuleType("backend.rag")
    fake_rag.__path__ = []

    fake_utils = types.ModuleType("backend.rag.utils")
    fake_utils.RETRIEVAL_TOP_K = 5
    fake_utils.retrieve_documents = retrieve_documents
    fake_utils.rewrite_query_once = rewrite_query_once or (lambda query: {
        "rewrite_method": "step_back",
        "step_back_question": "broader question",
        "hyde_document": "",
        "rewritten_query": f"rewritten {query}",
    })
    fake_utils.dedupe_documents = _dedupe_documents
    fake_utils.retrieval_trace_fields = lambda meta: dict(meta)

    module_name = f"rag_pipeline_under_test_{id(retrieve_documents)}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "backend" / "rag" / "pipeline.py",
    )
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, {"backend.rag": fake_rag, "backend.rag.utils": fake_utils}):
        spec.loader.exec_module(module)

    return module


def enable_complexity_planning(pipeline):
    """Recompile the loaded pipeline with the planning branch registered.

    The profile ships with `complexity_planning_enabled: false`, so a freshly loaded
    module has no classify_complexity node. Tests covering the planner have to ask for
    it explicitly. The module global is set as well as the graph rebuilt, because
    `_initial_state` reads it — leaving the two disagreeing would seed a state that
    claims planning is off into a graph that plans.
    """
    pipeline.COMPLEXITY_PLANNING_ENABLED = True
    pipeline.rag_graph = pipeline.build_rag_graph(complexity_planning_enabled=True)
    return pipeline


def _doc(text, chunk_id="chunk-1", filename="doc.md"):
    return {
        "filename": filename,
        "page_number": 1,
        "text": text,
        "chunk_id": chunk_id,
    }


def _meta(count):
    return {
        "retrieval_mode": "hybrid",
        "retrieval_pipeline": "recall_merge_rerank",
        "candidate_k": count,
        "retrieval_top_k": 5,
        "recall_count": count,
        "retrieval_empty": count == 0,
    }


class RagShortCircuitTests(unittest.TestCase):
    def _ctx(self):
        return ChatRequestContext.for_sync(user_id="u", session_id="s")

    def test_grader_uses_only_grade_model(self):
        pipeline = load_pipeline(
            retrieve_documents=lambda query, top_k=5: {"docs": [], "meta": _meta(0)}
        )
        initialized = Mock()
        grader = object()
        initialized.return_value = grader
        pipeline.API_KEY = "test-key"
        pipeline.BASE_URL = "https://example.test/v1"
        pipeline.FAST_MODEL = "fast-model"
        pipeline.GRADE_MODEL = "grade-model"
        pipeline._grader_model = None
        pipeline.init_chat_model = initialized

        self.assertIs(grader, pipeline._get_grader_model())
        initialized.assert_called_once_with(
            model="grade-model",
            model_provider="openai",
            api_key="test-key",
            base_url="https://example.test/v1",
            temperature=0,
            stream_usage=True,
        )

    def test_grader_does_not_use_other_models_when_grade_model_is_missing(self):
        pipeline = load_pipeline(
            retrieve_documents=lambda query, top_k=5: {"docs": [], "meta": _meta(0)}
        )
        pipeline.API_KEY = "test-key"
        pipeline.FAST_MODEL = "fast-model"
        pipeline.GRADE_MODEL = None
        pipeline._grader_model = None
        pipeline.init_chat_model = Mock()

        self.assertIsNone(pipeline._get_grader_model())
        pipeline.init_chat_model.assert_not_called()

    def test_simple_no_retrieval_short_circuits_without_rewrite(self):
        calls = {"retrieve": 0, "step_back": 0}

        def retrieve(query, top_k=5):
            calls["retrieve"] += 1
            return {"docs": [], "meta": _meta(0)}

        def step_back(query):
            calls["step_back"] += 1
            return {
                "rewrite_method": "step_back",
                "step_back_question": "broader question",
                "hyde_document": "",
                "rewritten_query": f"rewritten {query}",
            }

        pipeline = load_pipeline(retrieve_documents=retrieve, rewrite_query_once=step_back)
        pipeline._get_complexity_model = lambda: FakeStructuredModel(
            lambda schema, prompt: {"complexity": "simple", "reason": "unit"}
        )
        pipeline._get_grader_model = lambda: FakeStructuredModel(lambda schema, prompt: {})

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("uncovered question", ctx)
        finally:
            ctx.close()

        self.assertEqual([], result.get("docs"))
        self.assertEqual("no_knowledge", result.get("retrieval_status"))
        self.assertEqual("no_knowledge", result.get("rag_trace", {}).get("retrieval_status"))
        self.assertEqual(1, calls["retrieve"])
        self.assertEqual(0, calls["step_back"])

    def test_obvious_simple_question_skips_complexity_model(self):
        def retrieve(query, top_k=5):
            return {"docs": [_doc("Danjin is the Imaginary element")], "meta": _meta(1)}

        def grade(schema, prompt):
            return {
                "relevance": "strong",
                "answerability": "sufficient",
                "ambiguity": "none",
                "route": "answer",
                "confidence": 0.95,
            }

        pipeline = enable_complexity_planning(load_pipeline(retrieve_documents=retrieve))
        complexity_model = Mock(return_value=FakeStructuredModel(
            lambda schema, prompt: {"complexity": "simple", "reason": "model"}
        ))
        pipeline._get_complexity_model = complexity_model
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("What element is Danjin?", ctx)
        finally:
            ctx.close()

        complexity_model.assert_not_called()
        self.assertEqual("simple", result.get("complexity"))
        self.assertIn("fast_path", result.get("complexity_reason", ""))

    def test_multi_dimension_keyword_query_still_uses_complexity_model(self):
        def retrieve(query, top_k=5):
            return {"docs": [_doc("comparison evidence")], "meta": _meta(1)}

        def complexity(schema, prompt):
            return {
                "complexity": "complex",
                "reason": "multiple entities and dimensions",
                "sub_questions": ["Danjin's element and weapon", "Kakaro's element and weapon"],
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
        complexity_model_calls = {"count": 0}

        def get_complexity_model():
            complexity_model_calls["count"] += 1
            return FakeStructuredModel(complexity)

        pipeline._get_complexity_model = get_complexity_model
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("Danjin Kakaro element weapon type combat role", ctx)
        finally:
            ctx.close()

        self.assertGreaterEqual(complexity_model_calls["count"], 1)
        self.assertEqual("complex", result.get("complexity"))
        self.assertEqual(2, result.get("rag_trace", {}).get("sub_agent_count"))

    def test_complexity_plan_includes_child_queries(self):
        model_schemas = []

        def retrieve(query, top_k=5):
            return {"docs": [_doc(f"evidence for {query}", query)], "meta": _meta(1)}

        def plan(schema, prompt):
            model_schemas.append(schema.__name__)
            return {
                "complexity": "complex",
                "reason": "comparison",
                "sub_questions": ["Danjin's role", "Kakaro's role"],
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
        pipeline._get_complexity_model = lambda: FakeStructuredModel(plan)
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("Compare the combat roles of Danjin and Kakaro", ctx)
        finally:
            ctx.close()

        self.assertEqual(["ComplexityResult"], model_schemas)
        self.assertEqual(2, result.get("rag_trace", {}).get("sub_agent_count"))

    def test_planning_disabled_removes_the_node_and_the_call(self):
        """With planning off the planner is not skipped — it is absent.

        Asserting only "the model was not called" would also pass if the node ran and
        took its fast path, which is a different (and still billable) system. So the
        graph's own node list is checked too."""
        def retrieve(query, top_k=5):
            return {"docs": [_doc("direct answer evidence")], "meta": _meta(1)}

        def grade(schema, prompt):
            return {
                "relevance": "strong",
                "answerability": "sufficient",
                "ambiguity": "none",
                "route": "answer",
                "confidence": 0.9,
            }

        pipeline = load_pipeline(retrieve_documents=retrieve)
        pipeline.COMPLEXITY_PLANNING_ENABLED = False
        pipeline.rag_graph = pipeline.build_rag_graph(complexity_planning_enabled=False)
        complexity_model = Mock(side_effect=AssertionError("planner must not be reached"))
        pipeline._get_complexity_model = complexity_model
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        nodes = set(pipeline.rag_graph.get_graph().nodes)
        for absent in ("classify_complexity", "prepare_sub_questions", "rag_sub_agent", "synthesis"):
            self.assertNotIn(absent, nodes)
        self.assertIn("retrieve_initial", nodes)

        ctx = self._ctx()
        try:
            # A question the fast path would NOT classify: long, comparative, and
            # multi-dimension, so it is the planner's own case. It still must not run.
            result = pipeline.run_rag_graph(
                "Compare the combat roles of Danjin and Kakaro across weapon type and element", ctx
            )
        finally:
            ctx.close()

        complexity_model.assert_not_called()
        self.assertIsNone(result.get("complexity"))
        self.assertEqual("complexity_planning_disabled", result.get("complexity_reason"))
        self.assertEqual("answerable", result.get("retrieval_status"))
        self.assertEqual(1, len(result.get("docs", [])))

    def test_strong_evidence_returns_after_initial_grade(self):
        calls = {"retrieve": 0, "step_back": 0}

        def retrieve(query, top_k=5):
            calls["retrieve"] += 1
            return {"docs": [_doc("direct answer evidence")], "meta": _meta(1)}

        def grade(schema, prompt):
            return {
                "relevance": "strong",
                "answerability": "sufficient",
                "ambiguity": "none",
                "route": "answer",
                "confidence": 0.93,
            }

        pipeline = load_pipeline(
            retrieve_documents=retrieve,
            rewrite_query_once=lambda query: calls.__setitem__("step_back", calls["step_back"] + 1) or {},
        )
        pipeline._get_complexity_model = lambda: FakeStructuredModel(
            lambda schema, prompt: {"complexity": "simple", "reason": "unit"}
        )
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("covered question", ctx)
        finally:
            ctx.close()

        self.assertEqual(1, len(result.get("docs", [])))
        self.assertEqual("answerable", result.get("retrieval_status"))
        self.assertEqual(1, calls["retrieve"])
        self.assertEqual(0, calls["step_back"])

    def test_weak_evidence_rewrites_once_then_clarifies(self):
        calls = {"retrieve": [], "step_back": 0}

        def retrieve(query, top_k=5):
            calls["retrieve"].append(query)
            if query.startswith("rewritten"):
                return {"docs": [_doc("still partial evidence", "chunk-2")], "meta": _meta(1)}
            return {"docs": [_doc("weak evidence", "chunk-1")], "meta": _meta(1)}

        def grade(schema, prompt):
            return {
                "relevance": "weak",
                "answerability": "partial",
                "ambiguity": "none",
                "route": "rewrite",
                "confidence": 0.44,
            }

        def step_back(query):
            calls["step_back"] += 1
            return {
                "rewrite_method": "step_back",
                "step_back_question": "general?",
                "hyde_document": "",
                "rewritten_query": f"rewritten {query}",
            }

        pipeline = load_pipeline(retrieve_documents=retrieve, rewrite_query_once=step_back)
        pipeline._get_complexity_model = lambda: FakeStructuredModel(
            lambda schema, prompt: {"complexity": "simple", "reason": "unit"}
        )
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("weak question", ctx)
        finally:
            ctx.close()

        self.assertEqual(["weak question", "rewritten weak question"], calls["retrieve"])
        self.assertEqual(1, calls["step_back"])
        self.assertEqual("needs_clarification", result.get("retrieval_status"))
        self.assertEqual([], result.get("docs"))

    def test_hyde_rewrite_runs_only_selected_retrieval(self):
        calls = {"retrieve": [], "rewrite": 0, "grade": 0}

        def retrieve(query, top_k=5):
            calls["retrieve"].append(query)
            return {"docs": [_doc(f"evidence for {query}")], "meta": _meta(1)}

        def grade(schema, prompt):
            calls["grade"] += 1
            if calls["grade"] == 1:
                return {
                    "relevance": "weak",
                    "answerability": "partial",
                    "ambiguity": "none",
                    "route": "rewrite",
                    "confidence": 0.5,
                }
            return {
                "relevance": "strong",
                "answerability": "sufficient",
                "ambiguity": "none",
                "route": "answer",
                "confidence": 0.9,
            }

        def rewrite(query):
            calls["rewrite"] += 1
            return {
                "rewrite_method": "hyde",
                "step_back_question": "",
                "hyde_document": "A hypothetical answer used to recall real evidence",
                "rewritten_query": "HyDE rewritten query",
            }

        pipeline = load_pipeline(retrieve_documents=retrieve, rewrite_query_once=rewrite)
        pipeline._get_complexity_model = lambda: FakeStructuredModel(
            lambda schema, prompt: {"complexity": "simple", "reason": "unit"}
        )
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("vague conceptual question", ctx)
        finally:
            ctx.close()

        self.assertEqual(["vague conceptual question", "HyDE rewritten query"], calls["retrieve"])
        self.assertEqual(1, calls["rewrite"])
        self.assertEqual(2, calls["grade"])
        self.assertEqual("hyde", result.get("rag_trace", {}).get("rewrite_method"))
        self.assertIn("hypothetical answer", result.get("rag_trace", {}).get("hyde_document", ""))
        self.assertNotIn("step_back_question", result.get("rag_trace", {}))

    def test_missing_slot_and_scope_select_do_not_rewrite(self):
        cases = [
            ("missing_slot", "clarify", "needs_clarification"),
            ("multiple_candidates", "scope_select", "needs_scope_selection"),
        ]
        for ambiguity, route, status in cases:
            with self.subTest(ambiguity=ambiguity):
                calls = {"retrieve": 0, "step_back": 0}

                def retrieve(query, top_k=5):
                    calls["retrieve"] += 1
                    return {"docs": [_doc("related but ambiguous")], "meta": _meta(1)}

                def grade(schema, prompt):
                    return {
                        "relevance": "strong",
                        "answerability": "partial",
                        "ambiguity": ambiguity,
                        "route": route,
                        "confidence": 0.61,
                        "missing_slots": ["version"] if ambiguity == "missing_slot" else [],
                        "hitl_prompt": "Please provide the version" if ambiguity == "missing_slot" else "Please choose a direction",
                        "hitl_options": ["A", "B"] if ambiguity == "multiple_candidates" else [],
                    }

                pipeline = load_pipeline(
                    retrieve_documents=retrieve,
                    rewrite_query_once=lambda query: calls.__setitem__("step_back", calls["step_back"] + 1) or {},
                )
                pipeline._get_complexity_model = lambda: FakeStructuredModel(
                    lambda schema, prompt: {"complexity": "simple", "reason": "unit"}
                )
                pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

                ctx = self._ctx()
                try:
                    result = pipeline.run_rag_graph("ambiguous question", ctx)
                finally:
                    ctx.close()

                self.assertEqual(status, result.get("retrieval_status"))
                self.assertEqual([], result.get("docs"))
                self.assertEqual(1, calls["retrieve"])
                self.assertEqual(0, calls["step_back"])

    def test_hitl_result_includes_only_current_resume_state(self):
        def retrieve(query, top_k=5):
            return {"docs": [_doc("Both Danjin and Dan Heng could be relevant", "candidate")], "meta": _meta(1)}

        def grade(schema, prompt):
            return {
                "relevance": "strong",
                "answerability": "partial",
                "ambiguity": "missing_slot",
                "route": "clarify",
                "confidence": 0.7,
                "missing_slots": ["character name"],
                "hitl_prompt": "Please specify the character name",
                "hitl_options": ["Danjin", "Dan Heng"],
            }

        pipeline = load_pipeline(retrieve_documents=retrieve)
        pipeline._get_complexity_model = lambda: FakeStructuredModel(
            lambda schema, prompt: {"complexity": "simple", "reason": "unit"}
        )
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("What is this character's element?", ctx)
        finally:
            ctx.close()

        resume_state = result.get("hitl_resume_state")
        self.assertIsInstance(resume_state, dict)
        self.assertEqual("What is this character's element?", resume_state.get("question"))
        self.assertEqual("needs_clarification", resume_state.get("retrieval_status"))
        self.assertEqual({
            "question",
            "route",
            "retrieval_status",
            "rewrite_count",
            "complexity",
            "complexity_reason",
            "sub_questions",
        }, set(resume_state))

    def test_resume_goes_directly_to_targeted_retrieval_after_hitl_answer(self):
        calls = {"retrieve": []}

        def retrieve(query, top_k=5):
            calls["retrieve"].append(query)
            return {"docs": [_doc("Danjin is the Imaginary element", "retrieved")], "meta": _meta(1)}

        def grade(schema, prompt):
            return {
                "relevance": "strong",
                "answerability": "sufficient",
                "ambiguity": "none",
                "route": "answer",
                "confidence": 0.9,
            }

        pipeline = load_pipeline(retrieve_documents=retrieve)
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)
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

        self.assertEqual(["Danjin: What is this character's element?"], calls["retrieve"])
        self.assertEqual("answerable", result.get("retrieval_status"))
        self.assertEqual(1, len(result.get("docs", [])))
        self.assertTrue(result.get("rag_trace", {}).get("hitl_resumed"))
        self.assertEqual("targeted_retrieval", result.get("rag_trace", {}).get("hitl_resume_strategy"))
        self.assertEqual("hitl_targeted_retrieval", result.get("rag_trace", {}).get("retrieval_stage"))

    def test_complex_sub_agents_keep_partial_docs_without_rewrite(self):
        calls = {"retrieve": [], "step_back": 0}

        def retrieve(query, top_k=5):
            calls["retrieve"].append(query)
            if query == "known sub":
                return {"docs": [_doc("partial sub evidence", "known")], "meta": _meta(1)}
            return {"docs": [], "meta": _meta(0)}

        def complexity(schema, prompt):
            return {
                "complexity": "complex",
                "reason": "unit",
                "sub_questions": ["known sub", "unknown sub"],
            }

        def grade(schema, prompt):
            return {
                "relevance": "weak",
                "answerability": "partial",
                "ambiguity": "none",
                "route": "rewrite",
                "confidence": 0.5,
            }

        pipeline = enable_complexity_planning(load_pipeline(
            retrieve_documents=retrieve,
            rewrite_query_once=lambda query: calls.__setitem__("step_back", calls["step_back"] + 1) or {},
        ))
        pipeline._get_complexity_model = lambda: FakeStructuredModel(complexity)
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("complex question", ctx)
        finally:
            ctx.close()

        self.assertCountEqual(["known sub", "unknown sub"], calls["retrieve"])
        self.assertEqual(0, calls["step_back"])
        self.assertEqual(1, len(result.get("docs", [])))
        self.assertEqual("partial", result.get("retrieval_status"))

    def test_complex_all_no_knowledge_synthesizes_no_knowledge(self):
        calls = {"retrieve": 0}

        def retrieve(query, top_k=5):
            calls["retrieve"] += 1
            return {"docs": [], "meta": _meta(0)}

        def complexity(schema, prompt):
            return {
                "complexity": "complex",
                "reason": "unit",
                "sub_questions": ["missing one", "missing two"],
            }

        pipeline = enable_complexity_planning(load_pipeline(retrieve_documents=retrieve))
        pipeline._get_complexity_model = lambda: FakeStructuredModel(complexity)
        pipeline._get_grader_model = lambda: FakeStructuredModel(lambda schema, prompt: {})

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("complex uncovered", ctx)
        finally:
            ctx.close()

        self.assertEqual(2, calls["retrieve"])
        self.assertEqual([], result.get("docs"))
        self.assertEqual("no_knowledge", result.get("retrieval_status"))

    def test_complex_preserves_sub_agent_hitl_when_no_docs_can_be_synthesized(self):
        def retrieve(query, top_k=5):
            return {"docs": [_doc("ambiguous related evidence", query)], "meta": _meta(1)}

        def complexity(schema, prompt):
            return {
                "complexity": "complex",
                "reason": "unit",
                "sub_questions": ["feature of it", "genesis of it"],
            }

        def grade(schema, prompt):
            return {
                "relevance": "weak",
                "answerability": "none",
                "ambiguity": "missing_slot",
                "route": "clarify",
                "confidence": 0.4,
                "missing_slots": ["referent"],
                "hitl_prompt": "Please clarify exactly what it refers to.",
            }

        pipeline = enable_complexity_planning(load_pipeline(retrieve_documents=retrieve))
        pipeline._get_complexity_model = lambda: FakeStructuredModel(complexity)
        pipeline._get_grader_model = lambda: FakeStructuredModel(grade)

        ctx = self._ctx()
        try:
            result = pipeline.run_rag_graph("What are its main characteristics and causes?", ctx)
        finally:
            ctx.close()

        self.assertEqual([], result.get("docs"))
        self.assertEqual("needs_clarification", result.get("retrieval_status"))
        self.assertEqual("clarify", result.get("route"))
        self.assertIn("what it refers to", result.get("hitl_prompt", ""))


if __name__ == "__main__":
    unittest.main()
