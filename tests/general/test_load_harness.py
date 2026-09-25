"""The load-test harness (RAG_FIX_PLAN item 45) has to be right before its numbers are.

A stub that answers a call shape wrongly does not fail loudly: the backend takes an
error path, the turn ends fast, and the report reads as a quick backend. So the stub's
answers are pinned here — valid against the schema asked for, routed the way an ordinary
knowledge-base turn goes — and so is the load report's refusal to count a turn that
ended in an apology.
"""
import json
import math
import unittest

from fastapi.testclient import TestClient

from tests.load import stub_provider as stub
from tests.load.chat_load import DEGRADED_ROUTES, Level, backend_env, pct


def _no_wait():
    stub.Latency.structured_ms = stub.Latency.toolcall_ms = 0
    stub.Latency.ttft_ms = stub.Latency.token_ms = stub.Latency.embed_ms = 0
    stub.Latency.tokens = 5


class SchemaInstanceTests(unittest.TestCase):
    def test_every_required_field_is_present_and_typed(self):
        schema = {"type": "object", "properties": {
            "scope": {"enum": ["in_domain", "out_of_domain"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"}, "count": {"type": "integer"},
            "flag": {"type": "boolean"}, "why": {"type": "string"},
        }}
        value = stub.instance(schema, question="q")
        self.assertEqual({"scope": "in_domain", "tags": [], "confidence": 0.0, "count": 0,
                          "flag": False, "why": "q"}, value)

    def test_refs_and_optional_unions_resolve(self):
        schema = {"type": "object", "$defs": {"Kind": {"enum": ["a", "b"]}},
                  "properties": {"kind": {"$ref": "#/$defs/Kind"},
                                 "maybe": {"anyOf": [{"type": "null"}, {"type": "integer"}]}}}
        self.assertEqual({"kind": "a", "maybe": 0}, stub.instance(schema))

    def test_the_route_deciding_fields_send_a_turn_down_the_knowledge_path(self):
        fmt = {"type": "json_schema", "json_schema": {"name": "RequestEnvelope", "schema": {
            "type": "object", "properties": {
                "scope": {"enum": ["out_of_domain", "in_domain"]},
                "needed_tools": {"type": "array", "items": {"type": "string"}},
            }}}}
        value = stub.structured_answer(fmt, "fees?")
        self.assertEqual("in_domain", value["scope"])
        self.assertEqual([stub.KNOWLEDGE_TOOL], value["needed_tools"])

    def test_a_pinned_field_the_schema_lacks_is_not_invented(self):
        fmt = {"type": "json_schema", "json_schema": {"name": "EvidenceGrade", "schema": {
            "type": "object", "properties": {"route": {"enum": ["rewrite", "answer"]}}}}}
        self.assertEqual({"route": "answer"}, stub.structured_answer(fmt, "q"))

    def test_the_resolver_hands_back_the_question_it_was_given(self):
        fmt = {"type": "json_schema", "json_schema": {"name": "ResolvedQuery", "schema": {
            "type": "object", "properties": {"question": {"type": "string"},
                                             "search_text": {"type": "string"}}}}}
        value = stub.structured_answer(fmt, "When does term start?")
        self.assertEqual("When does term start?", value["question"])
        self.assertEqual("When does term start?", value["search_text"])


class StubEndpointTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(stub.Latency, k) for k in vars(stub.Latency) if not k.startswith("_")}
        _no_wait()
        self.client = TestClient(stub.app)

    def tearDown(self):
        for key, value in self.saved.items():
            setattr(stub.Latency, key, value)

    def _tools(self):
        return [{"type": "function", "function": {"name": stub.KNOWLEDGE_TOOL, "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}}}}]

    def test_offered_tools_and_none_run_yet_is_a_call_to_the_knowledge_tool(self):
        body = {"model": "m", "tools": self._tools(),
                "messages": [{"role": "user", "content": "What are the fees?"}]}
        message = self.client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
        call = message["tool_calls"][0]["function"]
        self.assertEqual(stub.KNOWLEDGE_TOOL, call["name"])
        self.assertEqual({"query": "What are the fees?"}, json.loads(call["arguments"]))

    def test_once_a_tool_has_run_the_reply_is_text(self):
        body = {"model": "m", "tools": self._tools(), "messages": [
            {"role": "user", "content": "fees?"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "c1", "content": "evidence"}]}
        message = self.client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
        self.assertTrue(message["content"])
        self.assertNotIn("tool_calls", message)

    def test_a_result_folded_into_the_transcript_counts_as_a_tool_that_ran(self):
        """The backend folds tool results into text for its provider (provider_compat).
        Missed, the stub asked for the tool again after every planned dispatch, and every
        load-test knowledge turn paid a model call, a search and a grade production does not."""
        from backend.provider_compat import fold_messages

        folded = fold_messages([
            {"role": "user", "content": "What are the fees?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": stub.KNOWLEDGE_TOOL, "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Fees are reviewed each year."}])
        self.assertFalse(any(message.get("role") == "tool" for message in folded))
        message = self.client.post("/v1/chat/completions", json={
            "model": "m", "tools": self._tools(), "messages": folded}).json()["choices"][0]["message"]
        self.assertTrue(message["content"])
        self.assertNotIn("tool_calls", message)
        self.assertEqual("What are the fees?", stub.last_user_text(folded))

    def test_a_streamed_answer_arrives_in_chunks_and_ends(self):
        body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        with self.client.stream("POST", "/v1/chat/completions", json=body) as response:
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]
        self.assertEqual("data: [DONE]", lines[-1])
        words = [json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                 for line in lines[:-1]]
        self.assertGreaterEqual(sum(1 for w in words if w.strip()), stub.Latency.tokens)

    def test_embeddings_are_unit_length_deterministic_and_one_per_input(self):
        body = {"model": "e", "input": ["a", "b", "a"]}
        data = self.client.post("/v1/embeddings", json=body).json()["data"]
        self.assertEqual(3, len(data))
        self.assertEqual(data[0]["embedding"], data[2]["embedding"])
        self.assertNotEqual(data[0]["embedding"], data[1]["embedding"])
        self.assertAlmostEqual(1.0, math.sqrt(sum(v * v for v in data[0]["embedding"])), places=6)
        self.assertEqual(stub.Latency.embed_dim, len(data[0]["embedding"]))


class LoadReportTests(unittest.TestCase):
    def test_a_turn_that_ended_in_an_apology_is_a_failure_not_a_fast_turn(self):
        self.assertIn("retrieval_error", DEGRADED_ROUTES)

    def test_percentiles_of_nothing_are_not_zero(self):
        self.assertTrue(math.isnan(pct([], 95)))

    def test_the_report_keeps_failures_out_of_the_latency_figures(self):
        level = Level(parents=2)
        level.turn.append(1000.0)
        level.fail("route_retrieval_error")
        row = level.row()
        self.assertEqual(1, row["completed"])
        self.assertEqual({"route_retrieval_error": 1}, row["errors"])
        self.assertIsNone(row["ttft_p50_ms"])  # no first word was timed

    def test_a_level_where_everything_failed_still_reports(self):
        level = Level(parents=3)
        for _ in range(3):
            level.fail("http_500")
        row = level.row()
        self.assertEqual(0, row["completed"])
        self.assertIsNone(row["turn_p95_ms"])

    def test_the_backend_under_test_leaves_nothing_pointed_at_a_real_provider(self):
        env = backend_env("http://stub/v1")
        self.assertEqual("", env["LLM_PROVIDER"])
        for key in ("BASE_URL", "EMBEDDING_BASE_URL"):
            self.assertEqual("http://stub/v1", env[key])
        for key in ("MODEL", "FAST_MODEL", "GRADE_MODEL", "EMBEDDING_MODEL"):
            self.assertTrue(env[key].startswith("stub"))


if __name__ == "__main__":
    unittest.main()
