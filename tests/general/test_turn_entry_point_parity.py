"""The two chat entry points must answer the same turn the same way.

`chat_with_agent` (sync, `/chat`) and `chat_with_agent_stream` (`/chat/stream`, the one
the web app calls) each re-implemented the turn by hand, and they drifted: a rule added
to one was not always carried to the other. Each class here drives BOTH through the same
turn and asserts what the parent is shown and what is stored, so a divergence is a
failing test rather than a support ticket.

These are also the specification for folding the two paths into one turn pipeline: once
every case passes on both, the unification has nothing left to reconcile.
"""
from datetime import datetime, timezone
import asyncio
import importlib
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.chat.signals import RequestSignals
from backend.chat.turn_policy import TurnPlan
from backend.composition import Services
from backend.tools import KNOWLEDGE_TOOL
from tests.general.test_chat_hitl_resume import FakeStorage, _parse_sse_events
from backend.chat.answer_checks import terminal_reply
from backend.chat.clarification import PENDING_HITL_KEY

# A clarification asked a moment ago. Pending questions expire after a day
# (agent.clarification_ttl_minutes), so a fixture modelling a LIVE one is dated now.
_ASKED_JUST_NOW = datetime.now(timezone.utc).isoformat()

service = importlib.import_module("backend.chat.service")


def _stream_shown(*args, **kwargs) -> str:
    """What the parent ends up reading from a streamed turn.

    A `content_replace` ASSIGNS on the client, so the last one wins; without one, the
    reply is the `content` deltas in order.
    """

    async def collect():
        return [chunk async for chunk in service.chat_with_agent_stream(*args, **kwargs)]

    events = _parse_sse_events(asyncio.run(collect()))
    replaced = [event["content"] for event in events if event.get("type") == "content_replace"]
    if replaced:
        return replaced[-1]
    return "".join(event.get("content", "") for event in events if event.get("type") == "content")


class _ModelThatMustNotAnswer:
    """The direct-answer model, which neither case below may reach."""

    def invoke(self, messages):
        raise AssertionError("no model call is warranted on this turn")

    async def astream(self, messages):
        raise AssertionError("no model call is warranted on this turn")
        yield  # pragma: no cover - makes this an async generator


class AResumedClarificationAfterAnOutage(unittest.TestCase):
    """B1, at the chat layer: an outage is `retrieval_error`, never `no_knowledge`.

    `resume_rag_from_hitl` already returns `retrieval_error` when the knowledge base is
    unreachable — test_retrieval_outage pins that. What no test checked is what the
    PARENT is then told. The sync path maps the status to the retry copy. The streamed
    path looked only at whether any documents came back, found none, and told the parent
    the school has no information on it — the reading B1 exists to forbid, on the path
    the web app actually uses.
    """

    PENDING = {
        "id": "hitl-1",
        "original_question": "What is this character's element?",
        "prompt": "Please specify the character name",
        "options": ["Danjin", "Dan Heng"],
        "route": "clarify",
        "retrieval_status": "needs_clarification",
        "answers": [],
        "created_at": _ASKED_JUST_NOW,
        "resume_state": {
            "question": "What is this character's element?",
            "route": "clarify",
            "retrieval_status": "needs_clarification",
            "rewrite_count": 0,
            "complexity": "simple",
            "complexity_reason": "unit",
            "sub_questions": [],
        },
    }
    OUTAGE = {
        "docs": [],
        "retrieval_status": "retrieval_error",
        "route": "retrieval_error",
        "rag_trace": {"retrieval_status": "retrieval_error", "route": "retrieval_error"},
    }

    def setUp(self):
        # The assertions below are only meaningful while the two copies differ.
        self.assertNotEqual(service._COPY.retrieval_error, service._COPY.no_knowledge)

    def _storage(self):
        return FakeStorage(
            messages=[
                HumanMessage(content="What is this character's element?"),
                AIMessage(content="Please specify the character name"),
            ],
            metadata={PENDING_HITL_KEY: dict(self.PENDING)},
        )

    def _patched(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(
            service, "create_agent_for_request",
            Mock(side_effect=AssertionError("a resume must not build the agent")),
        ))
        stack.enter_context(patch.object(
            service, "_resume_rag_from_hitl_sync", Mock(return_value=dict(self.OUTAGE)),
        ))
        stack.enter_context(patch.object(service, "model", _ModelThatMustNotAnswer()))
        stack.enter_context(patch.object(service, "_update_persistent_note_sync", Mock(return_value="")))
        return stack

    def test_the_streamed_path_tells_the_parent_to_try_again(self):
        storage = self._storage()
        with self._patched():
            shown = _stream_shown("Danjin", "u", "s", services=Services(conversations=storage))

        self.assertEqual(service._COPY.retrieval_error, shown)
        self.assertEqual(service._COPY.retrieval_error, storage.messages[-1].content)

    def test_the_sync_path_tells_the_parent_to_try_again(self):
        storage = self._storage()
        with self._patched():
            response = service.chat_with_agent(
                "Danjin", "u", "s", services=Services(conversations=storage)
            )

        self.assertEqual(service._COPY.retrieval_error, response["response"])
        self.assertEqual(service._COPY.retrieval_error, storage.messages[-1].content)


class _SyncAgentEndingOnATerminalResult:
    """What the real graph leaves behind after `_end_turn_on_terminal_retrieval` jumps
    to the end: the model's tool call, the tool's result, and no answer after them."""

    def __init__(self, ctx, status):
        self.ctx, self.status = ctx, status

    def invoke(self, payload, config=None):
        self.ctx.store_rag_trace({"retrieval_status": self.status, "route": self.status}, None)
        self.ctx.note_short_circuit(self.status)
        call = AIMessage(content="", tool_calls=[{
            "name": KNOWLEDGE_TOOL, "args": {"query": "q"}, "id": "c1", "type": "tool_call",
        }])
        result = ToolMessage(
            content="NO_KNOWLEDGE: nothing in the school's documents covers this.",
            tool_call_id="c1",
            name=KNOWLEDGE_TOOL,
        )
        return {"messages": [*payload["messages"], call, result]}


class _StreamAgentEndingOnATerminalResult(_SyncAgentEndingOnATerminalResult):
    async def astream(self, payload, stream_mode=None, config=None):
        self.ctx.store_rag_trace({"retrieval_status": self.status, "route": self.status}, None)
        self.ctx.note_short_circuit(self.status)
        yield ToolMessage(
            content="NO_KNOWLEDGE: nothing in the school's documents covers this.",
            tool_call_id="c1",
            name=KNOWLEDGE_TOOL,
        ), {}


class AKnowledgeSearchThatEndsTheTurn(unittest.TestCase):
    """When retrieval concludes, the graph ends itself and the profile's copy is served.

    `_end_turn_on_terminal_retrieval` stops the agent before a second model call would
    reword profile copy, and records why on the context. The streamed path reads that and
    serves the profile's own reply. The sync path never did — the commit that added it
    touched only the streamed entry point — so its answer became the graph's last
    message, which is the TOOL result: the evidence cut strips it, and the parent was
    handed the could-not-verify copy instead.
    """

    STATUSES = ("no_knowledge", "retrieval_error")

    def setUp(self):
        for status in self.STATUSES:
            self.assertNotEqual(
                terminal_reply(status, "en"), service._COPY.unverified_answer
            )

    def _patched(self, agent_type, status) -> ExitStack:
        plan = TurnPlan(exposed_tools=[KNOWLEDGE_TOOL], language="en", reasons=["unit"])
        stack = ExitStack()
        stack.enter_context(patch.object(
            service, "plan_turn", lambda *a, **k: (plan, RequestSignals()),
        ))
        stack.enter_context(patch.object(
            service, "create_agent_for_request",
            lambda ctx, *a, **k: agent_type(ctx, status),
        ))
        stack.enter_context(patch.object(service, "_update_persistent_note_sync", Mock(return_value="")))
        return stack

    def test_the_streamed_path_serves_the_profile_reply(self):
        for status in self.STATUSES:
            with self.subTest(status=status):
                storage = FakeStorage()
                with self._patched(_StreamAgentEndingOnATerminalResult, status):
                    shown = _stream_shown(
                        "what is partner", "u", "s", services=Services(conversations=storage)
                    )
                expected = terminal_reply(status, "en")
                self.assertEqual(expected, shown)
                self.assertEqual(expected, storage.messages[-1].content)

    def test_the_sync_path_serves_the_profile_reply(self):
        for status in self.STATUSES:
            with self.subTest(status=status):
                storage = FakeStorage()
                with self._patched(_SyncAgentEndingOnATerminalResult, status):
                    response = service.chat_with_agent(
                        "what is partner", "u", "s", services=Services(conversations=storage)
                    )
                expected = terminal_reply(status, "en")
                self.assertEqual(expected, response["response"])
                self.assertEqual(expected, storage.messages[-1].content)


class AShortCircuitedTurn(unittest.TestCase):
    """A turn no agent is built for spends no model call on either path, and records the same trace.

    The streamed path has always left the persistent note alone here: a canned reply the
    corpus never saw contributes nothing worth summarising, and updating the note would be
    the one model call on a path whose point is making none. The sync path never learned
    that, and folded canned replies into a long conversation's memory at the price of a
    call. Its trace also dropped the request signals the streamed trace records.

    The note runs as a background job now, so each path is driven through a job runner
    that is drained before the assertion — "not called" has to mean "not queued either".
    """

    def _history(self):
        # Longer than the context window, so the note would otherwise be due.
        window = service._PROFILE.agent.context_window_messages
        return [
            HumanMessage(content=f"question {index}") if index % 2 == 0 else AIMessage(content=f"answer {index}")
            for index in range(window + 2)
        ]

    def _patched(self, note) -> ExitStack:
        plan = TurnPlan(static_reply="That is outside what I can help with.", exposed_tools=[], reasons=["unit"])
        stack = ExitStack()
        stack.enter_context(patch.object(service, "plan_turn", lambda *a, **k: (plan, RequestSignals())))
        stack.enter_context(patch.object(
            service, "create_agent_for_request",
            Mock(side_effect=AssertionError("a short-circuited turn builds no agent")),
        ))
        stack.enter_context(patch.object(service, "_update_persistent_note_sync", note))
        return stack

    def _services(self):
        from backend.chat.background import InlineJobs

        return Services(conversations=FakeStorage(self._history()), background_jobs=InlineJobs())

    def test_the_sync_path_spends_no_note_update(self):
        note = Mock(return_value="note")
        with self._patched(note):
            service.chat_with_agent("what is the weather", "u", "s", services=self._services())
        note.assert_not_called()

    def test_the_streamed_path_spends_no_note_update(self):
        note = Mock(return_value="note")
        with self._patched(note):
            _stream_shown("what is the weather", "u", "s", services=self._services())
        note.assert_not_called()

    def test_both_paths_store_the_request_signals(self):
        for path in ("sync", "stream"):
            with self.subTest(path=path):
                storage = FakeStorage(self._history())
                with self._patched(Mock(return_value="")):
                    if path == "sync":
                        service.chat_with_agent("what is the weather", "u", "s", services=Services(conversations=storage))
                    else:
                        _stream_shown("what is the weather", "u", "s", services=Services(conversations=storage))
                stored = storage.appends[-1]["messages"][-1].rag_trace
                self.assertIn("request_scope", stored)


if __name__ == "__main__":
    unittest.main()
