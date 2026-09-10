"""The knowledge tool reports what it came back with, the way every record tool does.

Regression for the production failure of 2026-09-10. Every knowledge-base question the
classifier routed to `search_knowledge_base` alone was answered with the could-not-verify
copy — «معلش، مقدرتش أتأكد من الأرقام دي من مستندات المدرسة نفسها…» — including «مين
الشركاء بتوع المدرسة», a question with no figure in it. Retrieval ran, the model
answered, and the answer was replaced afterwards.

The mechanism: the planner REQUIRES the one tool it narrowed the turn to, and
`service._enforce_forced_tool_ran` then reads `ctx.tool_outcomes` to check that the
required tool ran. Only the record tools reported there (`records._reporter`). The
knowledge tool never had, so to that check it had never run — on every turn it was
required on, which is every knowledge-base turn.

The existing tests for that check used a fabricated `("search_knowledge_base", "ok")`
outcome, so they passed against a tool that never produced one. Everything here invokes
the real tool.
"""
import json
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from backend.chat.request_context import ChatRequestContext
from backend.tools import KNOWLEDGE_TOOL
from backend.tools.knowledge import make_search_knowledge_base

CHUNKS = [
    {
        "filename": "partners.pdf",
        "page_number": 1,
        "text": "شركاء المدرسة: جامعة القاهرة، ومؤسسة مصر الخير.",
    },
]

ANSWER = "شركاء المدرسة هم جامعة القاهرة ومؤسسة مصر الخير."


def _rag(status, docs=CHUNKS, route="answer", **trace):
    return {
        "docs": list(docs),
        "rag_trace": {"retrieval_status": status, "route": route, **trace},
    }


def _pipeline(result):
    """Retrieval replaced at the module boundary the tool imports lazily."""
    module = types.ModuleType("backend.rag.pipeline")
    module.run_rag_graph = lambda query, ctx: result
    return patch.dict(sys.modules, {"backend.rag.pipeline": module})


def _pipeline_raises(exc):
    """Same module boundary as `_pipeline`, but retrieval blows up instead of returning."""
    def _raise(query, ctx):
        raise exc

    module = types.ModuleType("backend.rag.pipeline")
    module.run_rag_graph = _raise
    return patch.dict(sys.modules, {"backend.rag.pipeline": module})


def _ctx():
    ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
    ctx.reset_knowledge_tool_budget()
    return ctx


def _search(ctx, result, query="مين الشركاء بتوع المدرسة"):
    with _pipeline(result):
        return make_search_knowledge_base(ctx).invoke({"query": query})


# One case per branch of the tool. The outcome names are the ones
# tools/knowledge_result.j2 renders, and the same string reaches `ctx.tool_outcomes`.
OUTCOMES = [
    ("chunks", _rag("answerable")),
    ("chunks", _rag("partial")),
    ("empty", _rag("answerable", docs=[])),
    ("no_knowledge", _rag("no_knowledge", docs=[], route="no_knowledge")),
    ("retrieval_error", _rag("retrieval_error", docs=[], route="retrieval_error")),
    (
        "needs_clarification",
        _rag("needs_clarification", docs=[], route="clarify", hitl_prompt="أنهي سنة؟"),
    ),
    (
        "needs_scope_selection",
        _rag(
            "needs_scope_selection",
            docs=[],
            route="scope_select",
            hitl_prompt="أنهي قسم؟",
            hitl_options=["المصاريف", "الزي"],
        ),
    ),
]


@pytest.mark.parametrize(
    "outcome, result",
    OUTCOMES,
    ids=[
        f"{outcome}-{result['rag_trace']['retrieval_status']}"
        for outcome, result in OUTCOMES
    ],
)
def test_the_knowledge_tool_reports_its_own_outcome(outcome, result):
    """Same contract as `test_the_timetable_tool_reports_its_own_outcome`."""
    ctx = _ctx()
    try:
        _search(ctx, result)
        assert ctx.tool_outcomes == [(KNOWLEDGE_TOOL, outcome)]
    finally:
        ctx.close()


def test_a_refused_repeat_call_is_still_reported():
    """`call_limit` is this tool having been reached, and the first call already reported."""
    ctx = _ctx()
    try:
        tool = make_search_knowledge_base(ctx)
        with _pipeline(_rag("answerable")):
            for _ in range(10):
                tool.invoke({"query": "المصاريف"})
                if ctx.tool_outcomes[-1] == (KNOWLEDGE_TOOL, "call_limit"):
                    break
        outcomes = ctx.tool_outcomes
        assert outcomes[-1] == (KNOWLEDGE_TOOL, "call_limit")
        assert set(outcomes[:-1]) == {(KNOWLEDGE_TOOL, "chunks")}
        assert len(outcomes) >= 2
    finally:
        ctx.close()


def test_a_raise_from_run_rag_graph_is_reported_as_retrieval_error():
    """A node with no guard of its own (`classify_complexity`, today; any future node,
    tomorrow) can raise straight out of `run_rag_graph`. The tool's own try/except is
    the backstop: same outcome, same copy, as the graceful `retrieval_error` route
    below — so the model never has to learn a second way retrieval can fail, and a
    forced-tool turn still sees the search as having run.
    """
    ctx = _ctx()
    try:
        with _pipeline_raises(RuntimeError("Milvus unreachable")):
            # (a) does not raise out to the caller.
            shown = make_search_knowledge_base(ctx).invoke(
                {"query": "مين الشركاء بتوع المدرسة"}
            )
        # (b) recorded exactly like the graceful retrieval_error branch.
        assert ctx.tool_outcomes == [(KNOWLEDGE_TOOL, "retrieval_error")]
    finally:
        ctx.close()

    ctx2 = _ctx()
    try:
        graceful = _search(ctx2, _rag("retrieval_error", docs=[], route="retrieval_error"))
        assert ctx2.tool_outcomes == [(KNOWLEDGE_TOOL, "retrieval_error")]
    finally:
        ctx2.close()

    # (c) the exception path and the graceful path converge on identical model-facing text.
    assert shown == graceful


class TheForcedToolCheckSeesARealSearch(unittest.TestCase):
    """The check that replaced every answer, fed the real tool instead of a fake outcome."""

    class _Plan:
        short_circuit = False

    class _Finalizer:
        answer = ANSWER

    def _verdict(self, ctx):
        from backend.chat import service

        return service._enforce_forced_tool_ran(self._Finalizer(), ctx, self._Plan())

    def test_a_required_search_that_ran_is_left_alone(self):
        ctx = _ctx()
        try:
            ctx.note_turn_plan([], [], forced_tool=KNOWLEDGE_TOOL)
            _search(ctx, _rag("answerable"))
            self.assertEqual("", self._verdict(ctx))
        finally:
            ctx.close()

    def test_a_search_that_found_nothing_still_counts_as_having_run(self):
        """The requirement is that the tool RAN, not that it found something. A turn
        that searched and found nothing gets the no-knowledge copy by its own route."""
        ctx = _ctx()
        try:
            ctx.note_turn_plan([], [], forced_tool=KNOWLEDGE_TOOL)
            _search(ctx, _rag("no_knowledge", docs=[], route="no_knowledge"))
            self.assertEqual("", self._verdict(ctx))
        finally:
            ctx.close()

    def test_a_required_search_that_never_ran_is_still_replaced(self):
        from backend.chat import service

        ctx = _ctx()
        try:
            ctx.note_turn_plan([], [], forced_tool=KNOWLEDGE_TOOL)
            self.assertEqual(service._COPY.unverified_answer, self._verdict(ctx))
        finally:
            ctx.close()


class _ObedientModel(GenericFakeChatModel):
    """Calls the tool it is required to call, then answers.

    See `tests.evals.planner_execution_eval._StubModel` for why obeying `tool_choice`
    matters: it is how a narrowed turn reaches its tool, and a stub that ignored it
    would report a property of the stub as "no tool ran".
    """

    def __init__(self, answer, **kwargs):
        super().__init__(messages=iter([]), **kwargs)
        object.__setattr__(self, "_answer", answer)
        object.__setattr__(self, "_forced", None)

    def bind_tools(self, tools=(), **kwargs):
        choice = kwargs.get("tool_choice")
        if isinstance(choice, dict):
            choice = (choice.get("function") or {}).get("name")
        if choice in (None, "auto", "any", "none", "required"):
            choice = None
        object.__setattr__(self, "_forced", choice)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        already_ran = any(isinstance(m, ToolMessage) for m in messages)
        if self._forced and not already_ran:
            call = {"name": self._forced, "args": {"query": "شركاء المدرسة"}, "id": "forced-1"}
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))]
            )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._answer))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # The base fake streams text and `additional_kwargs` only — a tool call in the
        # generated message never reaches a streaming agent, which would make this stub
        # report "no tool ran" on the streamed path for a reason of its own. The real
        # provider streams the call as `tool_call_chunks`, so this does too.
        message = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = message.generations[0].message
        if message.tool_calls:
            chunks = [
                {"name": call["name"], "args": json.dumps(call["args"]),
                 "id": call["id"], "index": index}
                for index, call in enumerate(message.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        yield ChatGenerationChunk(message=AIMessageChunk(content=message.content))


class AKnowledgeBaseTurnKeepsItsAnswer(unittest.IsolatedAsyncioTestCase):
    """The production failure, end to end through the real stream.

    Real: `create_agent_for_request`, the middleware stack, the knowledge tool, the
    finalizer and both post-answer checks. Stubbed: the planner's decision (a plan that
    requires the search, which is what the classifier produces for a knowledge-base
    question), retrieval, and the model.
    """

    async def _shown(self, plan):
        import backend.chat.runtime as runtime
        from backend.chat import service
        from backend.chat.orchestrator import _hand_to_graph
        from backend.chat.signals import RequestSignals
        from tests.general.test_chat_hitl_resume import FakeStorage
        from tests.general.test_parent_turn_scenarios import _parse_sse, _text_shown

        def fake_plan_turn(question, history=None, ctx=None, **kwargs):
            # The hand-off is real: it is what puts `forced_tool` and `planned_calls`
            # on the context, where both the middleware and the check read them.
            _hand_to_graph(ctx, plan)
            return plan, RequestSignals()

        chunks = []
        with (
            _pipeline(_rag("answerable")),
            patch.object(runtime, "model", _ObedientModel(ANSWER)),
            patch.object(service, "plan_turn", fake_plan_turn),
            patch.object(service, "storage", FakeStorage()),
            patch.object(service, "generate_session_title", Mock(return_value="t")),
            patch.object(service, "update_persistent_note", AsyncMock(return_value="")),
        ):
            async for chunk in service.chat_with_agent_stream(
                "مين الشركاء بتوع المدرسة", "parent-1", "session-1"
            ):
                chunks.append(chunk)
        return _text_shown(_parse_sse(chunks))

    def _assert_answered(self, shown):
        from backend.chat import service

        self.assertNotEqual(service._COPY.unverified_answer, shown)
        self.assertIn("جامعة القاهرة", shown)

    async def test_a_turn_that_required_the_search_keeps_the_answer(self):
        from backend.chat.turn_policy import TurnPlan

        plan = TurnPlan(exposed_tools=[KNOWLEDGE_TOOL], forced_tool=KNOWLEDGE_TOOL)
        self._assert_answered(await self._shown(plan))

    async def test_a_turn_whose_search_the_planner_dispatched_keeps_the_answer(self):
        """The other path a knowledge-base turn takes: the classifier NAMED the tool, so
        the planner wrote the call itself and the model only saw the result."""
        from backend.chat.turn_policy import TurnPlan

        plan = TurnPlan(
            exposed_tools=[KNOWLEDGE_TOOL],
            forced_tool=KNOWLEDGE_TOOL,
            planned_calls=[{"name": KNOWLEDGE_TOOL, "args": {"query": "شركاء المدرسة"}}],
        )
        self._assert_answered(await self._shown(plan))
