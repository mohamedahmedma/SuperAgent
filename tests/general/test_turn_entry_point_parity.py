"""The two chat entry points must answer the same turn the same way.

`chat_with_agent` (sync, `/chat`) and `chat_with_agent_stream` (`/chat/stream`, the one
the web app calls) each re-implemented the turn by hand, and they drifted: a rule added
to one was not always carried to the other. Each class here drives BOTH through the same
turn and asserts what the parent is shown and what is stored, so a divergence is a
failing test rather than a support ticket.

These are also the specification for folding the two paths into one turn pipeline: once
every case passes on both, the unification has nothing left to reconcile.
"""
import asyncio
import importlib
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from backend.composition import Services
from tests.general.test_chat_hitl_resume import FakeStorage, _parse_sse_events

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
        "created_at": "2026-07-11T00:00:00+00:00",
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
            metadata={service.PENDING_HITL_KEY: dict(self.PENDING)},
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
        stack.enter_context(patch.object(service, "update_persistent_note", AsyncMock(return_value="")))
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


if __name__ == "__main__":
    unittest.main()
