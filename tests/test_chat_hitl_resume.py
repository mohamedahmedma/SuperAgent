import importlib
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

service = importlib.import_module("backend.chat.service")


class FakeStorage:
    def __init__(self, messages=None, metadata=None):
        self.messages = list(messages or [])
        self.metadata = dict(metadata or {})
        self.saves = []

    def load_with_meta(self, user_id, session_id):
        return list(self.messages), dict(self.metadata)

    def save(self, user_id, session_id, messages, metadata=None, extra_message_data=None):
        self.messages = list(messages)
        if metadata is not None:
            self.metadata = {**self.metadata, **metadata}
        self.saves.append({
            "messages": list(messages),
            "metadata": metadata,
            "extra_message_data": extra_message_data,
        })


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
    def test_first_persistent_note_bootstraps_trimmed_history(self):
        fake_model = Mock()
        fake_model.invoke.return_value = Mock(content="summary")
        history = [
            HumanMessage(content="first round question"),
            AIMessage(content="first round answer"),
        ]

        with patch.object(service, "fast_model", fake_model):
            note = service._update_persistent_note_sync(
                "",
                "latest question",
                "latest answer",
                history_messages=history,
            )

        prompt = fake_model.invoke.call_args.args[0][0].content
        self.assertEqual("summary", note)
        self.assertIn("User: first round question", prompt)
        self.assertIn("AI: first round answer", prompt)

    async def test_stream_immediately_reports_progress_and_skips_note_for_short_chat(self):
        fake_storage = FakeStorage()
        update_note = AsyncMock(return_value="updated note")

        def make_agent(ctx):
            return FakeStreamAgent(ctx, chunks=["direct answer"])

        with (
            patch.object(service, "storage", fake_storage),
            patch.object(service, "create_agent_for_request", make_agent),
            patch.object(service, "generate_session_title", Mock(return_value="short question")),
            patch.object(service, "update_persistent_note", update_note),
        ):
            chunks = await _collect_stream("Hello", "u", "s")

        events = _parse_sse_events(chunks)
        self.assertEqual("rag_step", events[0].get("type"))
        # NOTE: this label is a hardcoded string in backend/chat/service.py and must
        # stay in sync with that file.
        self.assertEqual("Request received, preparing response", events[0]["step"]["label"])
        update_note.assert_not_called()

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
            "complexity": "simple",
            "complexity_reason": "unit",
            "sub_questions": [],
        }
        fake_storage = FakeStorage()
        update_note = AsyncMock(return_value="updated note")

        def make_agent(ctx):
            return FakeStreamAgent(
                ctx,
                trace=trace,
                chunks=["Please specify the character name"],
                resume_state=resume_state,
            )

        with (
            patch.object(service, "storage", fake_storage),
            patch.object(service, "create_agent_for_request", make_agent),
            patch.object(service, "generate_session_title", Mock(return_value="character question")),
            patch.object(service, "update_persistent_note", update_note),
        ):
            chunks = await _collect_stream("What is this character's element?", "u", "s")

        events = _parse_sse_events(chunks)
        self.assertFalse([event for event in events if event.get("type") == "content"])
        hitl_events = [event for event in events if event.get("type") == "hitl_request"]
        self.assertEqual(1, len(hitl_events))
        self.assertEqual("Please specify the character name", hitl_events[0]["hitl"]["prompt"])
        self.assertEqual(["Danjin", "Dan Heng"], hitl_events[0]["hitl"]["options"])

        pending_hitl = fake_storage.metadata.get(service.PENDING_HITL_KEY)
        self.assertIsInstance(pending_hitl, dict)
        self.assertEqual("What is this character's element?", pending_hitl["original_question"])
        self.assertEqual("Please specify the character name", pending_hitl["prompt"])
        self.assertEqual(resume_state, pending_hitl["resume_state"])
        self.assertEqual(
            "Please specify the character name\n\nAvailable options:\n- Danjin\n- Dan Heng",
            fake_storage.messages[-1].content,
        )
        update_note.assert_not_called()

    async def test_stream_resume_uses_saved_rag_state_without_reentering_agent(self):
        pending_hitl = {
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
        fake_storage = FakeStorage(
            messages=[
                HumanMessage(content="What is this character's element?"),
                AIMessage(content="Please specify the character name"),
            ],
            metadata={service.PENDING_HITL_KEY: pending_hitl},
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
            patch.object(service, "storage", fake_storage),
            patch.object(service, "create_agent_for_request", create_agent_mock),
            patch.object(service, "_resume_rag_from_hitl_sync", resume_mock),
            patch.object(service, "model", fake_model),
            patch.object(service, "update_persistent_note", AsyncMock(return_value="updated note")),
        ):
            chunks = await _collect_stream("Danjin", "u", "s")

        events = _parse_sse_events(chunks)
        self.assertEqual(["Danjin is the Imaginary element.[1]"], [
            event["content"] for event in events if event.get("type") == "content"
        ])
        self.assertFalse([event for event in events if event.get("type") == "hitl_request"])
        self.assertIsNone(fake_storage.metadata.get(service.PENDING_HITL_KEY))
        self.assertEqual("Danjin", fake_storage.messages[-2].content)
        self.assertEqual("Danjin is the Imaginary element.[1]", fake_storage.messages[-1].content)
        resume_mock.assert_called_once()
        create_agent_mock.assert_not_called()
        self.assertIn("Original question:\nWhat is this character's element?", fake_model.messages[-1][-1].content)
        self.assertIn("User's answer:\nDanjin", fake_model.messages[-1][-1].content)


if __name__ == "__main__":
    unittest.main()
