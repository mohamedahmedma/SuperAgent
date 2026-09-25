from datetime import datetime, timezone
import importlib
import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from backend.chat.background import InlineJobs
from backend.composition import Services
from backend.chat.clarification import PENDING_HITL_KEY

# A clarification asked a moment ago. Pending questions expire after a day
# (agent.clarification_ttl_minutes), so a fixture modelling a LIVE one is dated now.
_ASKED_JUST_NOW = datetime.now(timezone.utc).isoformat()

service = importlib.import_module("backend.chat.service")


class FakeStorage:
    """`ConversationStorage` in memory: what a turn appends, and the metadata merged with it.

    `messages` are the stored conversation as langchain messages, so a test reads the
    last answer as `storage.messages[-1].content`; `appends` keeps each append as it was
    handed over, trace included, for the tests that check what a turn stored beside it.
    """

    def __init__(self, messages=None, metadata=None):
        self.messages = list(messages or [])
        self.metadata = dict(metadata or {})
        self.appends = []

    def load_for_turn(self, user_id, session_id, *, window):
        return list(self.messages)[-window:], dict(self.metadata)

    def append(self, user_id, session_id, messages, *, metadata=None):
        for message in messages:
            self.messages.append(
                HumanMessage(content=message.content)
                if message.message_type == "human"
                else AIMessage(content=message.content)
            )
        if metadata:
            # The same merge Postgres does with `||`: right-hand keys win, None stays.
            self.metadata.update(metadata)
        self.appends.append({"messages": list(messages), "metadata": metadata})
        return list(range(len(self.messages) - len(messages) + 1, len(self.messages) + 1))


class FakeStreamAgent:
    def __init__(self, ctx, trace=None, chunks=None, captured_prompts=None, resume_state=None):
        self.ctx = ctx
        self.trace = trace
        self.chunks = chunks or []
        self.captured_prompts = captured_prompts
        self.resume_state = resume_state

    async def astream(self, payload, stream_mode=None, config=None):
        if self.captured_prompts is not None:
            self.captured_prompts.append(payload["messages"][-1].content)
        if self.trace:
            self.ctx.store_rag_trace(self.trace, self.resume_state)
        for chunk in self.chunks:
            yield AIMessageChunk(content=chunk), {}


class FakeDirectModel:
    def __init__(self, chunks):
        self.chunks = chunks
        self.messages = []

    async def astream(self, messages):
        self.messages.append(messages)
        for chunk in self.chunks:
            yield AIMessageChunk(content=chunk)


def _parse_sse_events(chunks):
    events = []
    for chunk in chunks:
        payload = chunk.strip()
        if not payload.startswith("data: "):
            continue
        data = payload[len("data: "):]
        if data == "[DONE]":
            events.append({"type": "DONE"})
        else:
            events.append(json.loads(data))
    return events


async def _collect_stream(*args, **kwargs):
    chunks = []
    async for chunk in service.chat_with_agent_stream(*args, **kwargs):
        chunks.append(chunk)
    return chunks


class ChatHitlResumeTests(unittest.IsolatedAsyncioTestCase):
    """HITL streaming, isolated from the turn planner.

    The planner runs before the agent and can end a turn outright, so a deployment that
    enables scope detection would otherwise short-circuit these fixtures and make a
    test of the HITL protocol fail for reasons that have nothing to do with it — and
    reach a live model to do it.
    """

    def setUp(self):
        from backend.chat.turn_policy import TurnPlan
        from backend.chat.signals import RequestSignals

        self._planner = patch.object(
            service, "plan_turn", lambda *a, **k: (TurnPlan(), RequestSignals()),
        )
        self._planner.start()
        self.addCleanup(self._planner.stop)

    async def test_stream_immediately_reports_progress(self):
        fake_storage = FakeStorage()

        def make_agent(ctx, tool_names=None, language=None):
            return FakeStreamAgent(ctx, chunks=["direct answer"])

        with (
            patch.object(service, "create_agent_for_request", make_agent),
            patch.object(service, "generate_session_title", Mock(return_value="short question")),
        ):
            chunks = await _collect_stream(
                "Hello", "u", "s", services=Services(conversations=fake_storage, background_jobs=InlineJobs())
            )

        events = _parse_sse_events(chunks)
        self.assertEqual("rag_step", events[0].get("type"))
        # NOTE: this label is a hardcoded string in backend/chat/service.py and must
        # stay in sync with that file.
        self.assertEqual("Request received, preparing response", events[0]["step"]["label"])

    async def test_stream_hitl_request_persists_pending_state_without_content(self):
        trace = {
            "retrieval_status": "needs_clarification",
            "route": "clarify",
            "hitl_prompt": "Please specify the character name",
            "hitl_options": ["Danjin", "Dan Heng"],
        }
        resume_state = {
            "question": "What is this character's element?",
            "route": "clarify",
            "retrieval_status": "needs_clarification",
            "rewrite_count": 0,
            # Carried across the resume boundary so "ask once per question" survives a
            # graph that starts fresh on every resume.
            "hitl_rounds": 0,
            "complexity": "simple",
            "complexity_reason": "unit",
            "sub_questions": [],
            # Conditions set before the clarification, carried across the resume
            # boundary for the same reason `hitl_rounds` is.
            "carried_constraints": [],
            # The planner's hints the question ran with, carried so the resumed search
            # runs under them too. Empty here: this fake agent planned nothing.
            "language": "",
            "child_year": "",
            "retrieval_sections": [],
            "child_names": [],
        }
        fake_storage = FakeStorage()

        def make_agent(ctx, tool_names=None, language=None):
            return FakeStreamAgent(
                ctx,
                trace=trace,
                chunks=["Please specify the character name"],
                resume_state=resume_state,
            )

        with (
            patch.object(service, "create_agent_for_request", make_agent),
            patch.object(service, "generate_session_title", Mock(return_value="character question")),
        ):
            chunks = await _collect_stream(
                "What is this character's element?",
                "u",
                "s",
                services=Services(conversations=fake_storage, background_jobs=InlineJobs()),
            )

        events = _parse_sse_events(chunks)
        self.assertFalse([event for event in events if event.get("type") == "content"])
        hitl_events = [event for event in events if event.get("type") == "hitl_request"]
        self.assertEqual(1, len(hitl_events))
        self.assertEqual("Please specify the character name", hitl_events[0]["hitl"]["prompt"])
        self.assertEqual(["Danjin", "Dan Heng"], hitl_events[0]["hitl"]["options"])

        pending_hitl = fake_storage.metadata.get(PENDING_HITL_KEY)
        self.assertIsInstance(pending_hitl, dict)
        self.assertEqual("What is this character's element?", pending_hitl["original_question"])
        self.assertEqual("Please specify the character name", pending_hitl["prompt"])
        self.assertEqual(resume_state, pending_hitl["resume_state"])
        self.assertEqual(
            "Please specify the character name\n\nAvailable options:\n- Danjin\n- Dan Heng",
            fake_storage.messages[-1].content,
        )

    async def test_stream_resume_uses_saved_rag_state_without_reentering_agent(self):
        pending_hitl = {
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
        fake_storage = FakeStorage(
            messages=[
                HumanMessage(content="What is this character's element?"),
                AIMessage(content="Please specify the character name"),
            ],
            metadata={PENDING_HITL_KEY: pending_hitl},
        )
        fake_model = FakeDirectModel(["Danjin is the Imaginary element.[1]"])
        resume_mock = Mock(return_value={
            "docs": [{"filename": "chars.pdf", "page_number": 1, "text": "Danjin is the Imaginary element."}],
            "retrieval_status": "answerable",
            "route": "answer",
            "rag_trace": {"retrieval_status": "answerable", "route": "answer"},
        })
        create_agent_mock = Mock(side_effect=AssertionError("agent should not be created on HITL resume"))

        with (
            patch.object(service, "create_agent_for_request", create_agent_mock),
            patch.object(service, "_resume_rag_from_hitl_sync", resume_mock),
            patch.object(service, "model", fake_model),
        ):
            chunks = await _collect_stream(
                "Danjin", "u", "s", services=Services(conversations=fake_storage, background_jobs=InlineJobs())
            )

        events = _parse_sse_events(chunks)
        self.assertEqual(["Danjin is the Imaginary element.[1]"], [
            event["content"] for event in events if event.get("type") == "content"
        ])
        self.assertFalse([event for event in events if event.get("type") == "hitl_request"])
        self.assertIsNone(fake_storage.metadata.get(PENDING_HITL_KEY))
        self.assertEqual("Danjin", fake_storage.messages[-2].content)
        self.assertEqual("Danjin is the Imaginary element.[1]", fake_storage.messages[-1].content)
        resume_mock.assert_called_once()
        create_agent_mock.assert_not_called()
        self.assertIn("Original question:\nWhat is this character's element?", fake_model.messages[-1][-1].content)
        self.assertIn("User's answer:\nDanjin", fake_model.messages[-1][-1].content)


if __name__ == "__main__":
    unittest.main()
