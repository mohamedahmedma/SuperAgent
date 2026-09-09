"""One search per distinct query, however many times the model asks for it.

`openai/gpt-oss-20b` emits the same `search_knowledge_base` call two to five times in a
single assistant message — measured on 4 of 6 turns against one question. Two layers
answer it, and they cover different halves:

  * middleware collapses the copies inside ONE message, before the graph dispatches them
  * the graph's memo answers an identical query asked again LATER in the same turn,
    which middleware cannot see because it only ever holds one message

Both are tested here so the pair is visible in one place.
"""
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

import backend.chat.runtime as runtime
from backend.chat.request_context import ChatRequestContext
from backend.chat.runtime import dedupe_tool_calls


def _call(query, call_id, name="search_knowledge_base"):
    return {"name": name, "args": {"query": query}, "id": call_id}


class DedupeRuleTests(unittest.TestCase):
    def test_identical_calls_collapse_to_the_first(self):
        calls = [_call("مصاريف ابني كام", f"c{i}") for i in range(5)]
        kept = dedupe_tool_calls(calls)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["id"], "c0")

    def test_a_different_query_is_a_different_search(self):
        """Measured: one turn emitted four copies of one query plus a fifth, narrower
        one. The narrow one is a real second search and must survive."""
        calls = [_call("مصاريف ابني كام", f"c{i}") for i in range(4)]
        calls.append(_call("مصاريف ابني", "c4"))
        self.assertEqual(len(dedupe_tool_calls(calls)), 2)

    def test_the_same_query_to_a_different_tool_is_not_a_duplicate(self):
        calls = [
            _call("مصاريف", "c0"),
            _call("مصاريف", "c1", name="search_products"),
        ]
        self.assertEqual(len(dedupe_tool_calls(calls)), 2)

    def test_argument_key_order_does_not_make_two_calls_different(self):
        calls = [
            {"name": "t", "args": {"a": 1, "b": 2}, "id": "c0"},
            {"name": "t", "args": {"b": 2, "a": 1}, "id": "c1"},
        ]
        self.assertEqual(len(dedupe_tool_calls(calls)), 1)

    def test_an_empty_or_single_list_is_returned_as_is(self):
        self.assertEqual(dedupe_tool_calls([]), [])
        self.assertEqual(len(dedupe_tool_calls([_call("x", "c0")])), 1)


class MiddlewareTests(unittest.TestCase):
    def _apply(self, calls):
        ctx = ChatRequestContext(user_id="u", session_id="s")
        middleware = runtime._collapse_duplicate_tool_calls(ctx)
        message = AIMessage(content="", tool_calls=calls, id="m1")
        update = middleware.after_model({"messages": [message]}, None)
        return update, ctx

    def test_five_copies_become_one_call(self):
        update, ctx = self._apply([_call("مصاريف ابني كام", f"c{i}") for i in range(5)])
        self.assertEqual(len(update["messages"][0].tool_calls), 1)
        self.assertEqual(ctx.duplicate_tool_calls, 4)

    def test_the_rewritten_message_keeps_its_id_so_the_reducer_replaces_it(self):
        """A new id would append a second assistant message instead of replacing the
        first, leaving both the duplicates and the collapse in the transcript."""
        update, _ = self._apply([_call("q", f"c{i}") for i in range(3)])
        self.assertEqual(update["messages"][0].id, "m1")

    def test_a_message_with_nothing_repeated_is_left_alone(self):
        update, ctx = self._apply([_call("a", "c0"), _call("b", "c1")])
        self.assertIsNone(update)
        self.assertEqual(ctx.duplicate_tool_calls, 0)

    def test_a_message_with_no_tool_calls_is_left_alone(self):
        ctx = ChatRequestContext(user_id="u", session_id="s")
        middleware = runtime._collapse_duplicate_tool_calls(ctx)
        state = {"messages": [AIMessage(content="مرحبا", id="m1")]}
        self.assertIsNone(middleware.after_model(state, None))


class AgentLoopTests(unittest.TestCase):
    """The whole point: the duplicates never become tool results."""

    def test_the_tool_runs_once_and_leaves_one_result(self):
        executed = []

        @tool("search_knowledge_base")
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base."""
            executed.append(query)
            return "Retrieved Chunks:\n[1] fees.pdf (Page 3):\nرسوم الصف الأول 30,000 جنيه."

        from langchain.agents import create_agent
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.outputs import ChatGeneration, ChatResult

        scripted = iter([
            AIMessage(
                content="",
                tool_calls=[_call("مصاريف ابني كام", f"c{i}") for i in range(5)],
                id="m1",
            ),
            AIMessage(content="رسوم الصف الأول 30,000 جنيه [1]", id="m2"),
        ])

        class Scripted(GenericFakeChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kw):
                return ChatResult(generations=[ChatGeneration(message=next(scripted))])

            def bind_tools(self, *args, **kwargs):
                return self

        ctx = ChatRequestContext(user_id="u", session_id="s")
        agent = create_agent(
            model=Scripted(messages=iter([])),
            tools=[search_knowledge_base],
            middleware=[runtime._collapse_duplicate_tool_calls(ctx)],
        )
        out = agent.invoke({"messages": [HumanMessage(content="مصاريف ابني كام")]})

        self.assertEqual(len(executed), 1)
        # Every surviving tool_call_id must have exactly one result, or the next request
        # to the provider is rejected for an unanswered call.
        results = [m for m in out["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(results), 1)
        self.assertEqual(ctx.duplicate_tool_calls, 4)


class RetrievalMemoTests(unittest.TestCase):
    """The graph's half: an identical query asked again in the same turn is free."""

    def _run(self, questions):
        from backend.rag import pipeline

        ctx = ChatRequestContext(user_id="u", session_id="s")
        invocations = []

        def _fake_invoke(state):
            invocations.append(state["question"])
            return {"docs": [{"text": "رسوم الصف الأول 30,000"}], "rag_trace": {}}

        with patch.object(pipeline.rag_graph, "invoke", _fake_invoke):
            results = [pipeline.run_rag_graph(q, ctx) for q in questions]
        return invocations, results, ctx

    def test_the_same_question_twice_runs_the_graph_once(self):
        invocations, results, _ = self._run(["مصاريف ابني كام"] * 3)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(results[0]["docs"], results[2]["docs"])

    def test_a_different_question_still_runs(self):
        invocations, _, _ = self._run(["مصاريف ابني كام", "مواعيد المدرسة"])
        self.assertEqual(len(invocations), 2)

    def test_whitespace_and_presentation_forms_are_the_same_question(self):
        invocations, _, _ = self._run(["مصاريف ابني كام", "  مصاريف ابني كام  "])
        self.assertEqual(len(invocations), 1)

    def test_a_caller_stamping_its_result_cannot_write_into_the_memo(self):
        """`run_rag_graph` itself sets `hitl_resume_state` on what it returns."""
        invocations, results, _ = self._run(["مصاريف ابني كام"] * 2)
        results[0]["hitl_resume_state"] = {"leaked": True}
        self.assertNotIn("hitl_resume_state", results[1])

    def test_the_memo_does_not_outlive_the_turn(self):
        """Two contexts are two turns, and the corpus may have changed between them."""
        from backend.rag import pipeline

        invocations = []

        def _fake_invoke(state):
            invocations.append(state["question"])
            return {"docs": [], "rag_trace": {}}

        with patch.object(pipeline.rag_graph, "invoke", _fake_invoke):
            for _ in range(2):
                pipeline.run_rag_graph(
                    "مصاريف ابني كام", ChatRequestContext(user_id="u", session_id="s")
                )
        self.assertEqual(len(invocations), 2)


if __name__ == "__main__":
    unittest.main()
