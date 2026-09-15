"""A conversation keeps everything a parent has seen in it.

Three ways it did not, each found in production on 2026-09-14 and each pinned here:

* **The save rewrote the conversation.** It took the caller's copy of the message list
  and made the database match, deleting and re-inserting when the two disagreed — which
  two overlapping turns, or one Redis write that failed silently, made them do. The
  rewrite dropped the other turn's rows and the trace (the images, the tables) of every
  message the caller had not re-supplied. Writes only append now.
* **The answer was saved after `[DONE]`, on the request.** A browser that left once the
  answer was on screen cancelled the request, and with it the save. The save is queued on
  a background thread before `[DONE]` goes out, and a cut-off stream stores what had
  reached the parent.
* **Metadata was written back whole.** A turn's copy of the session metadata, taken when
  it began, overwrote whatever a concurrent writer had changed since. Turns patch the keys
  they changed, and Postgres merges them.

The background runner has its own contract — order within a key, independence between
keys, a barrier the next turn can wait on — and that is tested first, without a database.
"""
import asyncio
import importlib
import json
import threading
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from backend.chat.background import MODELS, WRITES, BackgroundJobs, InlineJobs
from backend.chat.signals import RequestSignals
from backend.chat.storage import ConversationStorage, MessageToStore
from backend.chat.turn_policy import TurnPlan
from backend.composition import Services
from tests.general.postgres_support import postgres_schema
from tests.general.test_asset_delivery import DictCache
from tests.general.test_chat_hitl_resume import FakeStorage, FakeStreamAgent

service = importlib.import_module("backend.chat.service")


class BackgroundJobsTests(unittest.TestCase):
    def setUp(self):
        self.jobs = self._runner(lanes={WRITES: 3, MODELS: 1})

    def _runner(self, **options):
        jobs = BackgroundJobs(name="test-jobs", **options)
        self.addCleanup(jobs.shutdown, 5.0)
        return jobs

    def test_a_job_waiting_its_turn_holds_no_thread(self):
        """Two threads: one held by a blocked job. The job queued behind it under the same
        key must not take the other, or every other conversation's save waits for both."""
        jobs, gate, ran = self._runner(lanes={WRITES: 2}), threading.Event(), []
        jobs.submit("conversation:u:a", lambda: gate.wait(5))
        jobs.submit("conversation:u:a", lambda: ran.append("a2"))
        jobs.submit("conversation:u:b", lambda: ran.append("b"))

        self.assertTrue(jobs.flush("conversation:u:b", timeout=2))
        self.assertEqual(["b"], ran)
        gate.set()
        self.assertTrue(jobs.flush("conversation:u:a", timeout=5))
        self.assertEqual(["b", "a2"], ran)

    def test_a_slow_model_call_does_not_hold_up_a_save(self):
        """The note is a model call, seconds long. A save is milliseconds and must not
        queue behind it — the lanes are separate pools."""
        jobs, gate, ran = self._runner(lanes={WRITES: 1, MODELS: 1}), threading.Event(), []
        jobs.submit("note:u:s", lambda: gate.wait(5), lane=MODELS)
        jobs.submit("conversation:u:s", lambda: ran.append("saved"))

        self.assertTrue(jobs.flush("conversation:u:s", timeout=2))
        self.assertEqual(["saved"], ran)
        gate.set()

    def test_an_unknown_lane_is_refused(self):
        with self.assertRaises(ValueError):
            self.jobs.submit("k", lambda: None, lane="express")

    def test_jobs_under_one_key_run_in_the_order_they_were_queued(self):
        """A question must be stored before its answer, whatever the pool does."""
        order, gate = [], threading.Event()

        def first():
            gate.wait(2)
            order.append("first")

        self.jobs.submit("conversation:u:s", first)
        self.jobs.submit("conversation:u:s", lambda: order.append("second"))
        gate.set()

        self.assertTrue(self.jobs.flush("conversation:u:s", timeout=5))
        self.assertEqual(["first", "second"], order)

    def test_jobs_under_different_keys_do_not_wait_for_each_other(self):
        """One conversation's slow save must not hold another parent's."""
        gate, ran = threading.Event(), []
        self.jobs.submit("conversation:u:slow", lambda: gate.wait(5))
        self.jobs.submit("conversation:u:quick", lambda: ran.append("quick"))

        self.assertTrue(self.jobs.flush("conversation:u:quick", timeout=2))
        self.assertEqual(["quick"], ran)
        gate.set()

    def test_a_failed_job_is_logged_by_name_and_the_next_still_runs(self):
        ran = []

        def failing():
            raise RuntimeError("database down")

        with self.assertLogs("backend.chat.background", level="ERROR") as logs:
            self.jobs.submit("k", failing, describe="store the answer (u/s)")
            self.jobs.submit("k", lambda: ran.append("next"))
            self.assertTrue(self.jobs.flush("k", timeout=5))

        self.assertEqual(["next"], ran)
        self.assertIn("store the answer (u/s)", "\n".join(logs.output))

    def test_flush_reports_a_timeout_rather_than_hanging(self):
        gate = threading.Event()
        self.jobs.submit("k", lambda: gate.wait(5))
        self.assertFalse(self.jobs.flush("k", timeout=0.05))
        gate.set()

    def test_flushing_a_key_nothing_was_queued_under_returns_at_once(self):
        self.assertTrue(self.jobs.flush("conversation:nobody:nothing", timeout=0.01))

    def test_shutdown_finishes_queued_work_and_refuses_new_work(self):
        ran = []
        self.jobs.submit("k", lambda: ran.append("queued before the stop"))
        self.jobs.shutdown(timeout=5)

        self.assertEqual(["queued before the stop"], ran)
        with self.assertRaises(RuntimeError):
            self.jobs.submit("k", lambda: None)

    def test_inline_jobs_run_at_once_and_raise_where_they_are_called(self):
        ran = []
        InlineJobs().submit("k", lambda: ran.append(1))
        self.assertEqual([1], ran)

        def failing():
            raise ValueError("visible to the caller")

        with self.assertRaises(ValueError):
            InlineJobs().submit("k", failing)


class _CacheThatCanFail(DictCache):
    """Redis as `RedisCache` wraps it: a write that fails is swallowed, not raised."""

    def __init__(self):
        super().__init__()
        self.writes_fail = False

    def set_json(self, key, value, ttl=None):
        if not self.writes_fail:
            super().set_json(key, value, ttl)

    def delete(self, key):
        if not self.writes_fail:
            super().delete(key)


class AppendOnlyStorageTests(unittest.TestCase):
    """What `ConversationStorage` promises about rows once they are stored: nothing
    removes or rewrites them, whatever the caller believed the conversation was."""

    def setUp(self):
        from backend.db.models import ChatMessage, ChatSession, User

        schema = postgres_schema(self, User, ChatSession, ChatMessage)
        factory = schema.sessionmaker()
        db = factory()
        db.add(User(username="parent", password_hash="x"))
        db.commit()
        db.close()

        self.factory = factory
        self.cache = _CacheThatCanFail()
        self.storage = ConversationStorage(unit_of_work=schema.unit_of_work, cache=self.cache)

    def _rows(self):
        from backend.db.models import ChatMessage

        db = self.factory()
        try:
            return db.query(ChatMessage).order_by(ChatMessage.id.asc()).all()
        finally:
            db.close()

    def _append(self, kind, text, trace=None, metadata=None):
        return self.storage.append(
            "parent", "s", [MessageToStore(kind, text, rag_trace=trace)], metadata=metadata
        )

    def test_two_overlapping_turns_both_land_and_touch_nothing_older(self):
        """The replay of the production loss. Turn 2 loaded the conversation before turn
        1 had stored its answer; under the old save, the later of the two rewrote the
        conversation and the earlier answer — and the first answer's image — was gone."""
        self._append("human", "Show me the PE uniform")
        self._append("ai", "Here is the PE kit", trace={"tool_used": True, "asset_ids": ["uniforms::img1"]},
                     metadata={"title": "PE uniform"})
        self._append("human", "What are the bus fees?")          # turn 1
        self._append("human", "And tomorrow's timetable?")       # turn 2, overlapping
        self._append("ai", "The documents do not cover bus fees.", metadata={"pending_hitl": None})
        self._append("ai", "Tomorrow: Arabic, Maths, PE.")

        rows = self._rows()
        self.assertEqual(6, len(rows))
        self.assertEqual(
            ["Show me the PE uniform", "Here is the PE kit", "What are the bus fees?",
             "And tomorrow's timetable?", "The documents do not cover bus fees.", "Tomorrow: Arabic, Maths, PE."],
            [row.content for row in rows],
        )
        self.assertEqual(["uniforms::img1"], rows[1].rag_trace["asset_ids"], "the older answer kept its image")
        metadata = self.storage.session_metadata("parent", "s")
        self.assertEqual("PE uniform", metadata["title"])
        self.assertIsNone(metadata["pending_hitl"])

    def test_a_failed_cache_write_costs_nothing_that_was_stored(self):
        """One Redis write failing used to leave the next turn loading a stale copy, and
        the save then deleted the rows the copy did not know about."""
        self._append("human", "Show me the PE uniform")
        self._append("ai", "Here is the PE kit", trace={"tool_used": True, "asset_ids": ["uniforms::img1"]})
        self.storage.load_with_meta("parent", "s")  # warms the cache with two messages

        self.cache.writes_fail = True
        self._append("human", "What are the bus fees?")
        self._append("ai", "The documents do not cover bus fees.")
        self.cache.writes_fail = False

        messages, _metadata = self.storage.load_with_meta("parent", "s")
        self.assertEqual(4, len(messages), "a turn reads what is stored, not what the cache held")
        self._append("human", "OK, what about the canteen menu?")

        rows = self._rows()
        self.assertEqual(5, len(rows))
        self.assertEqual(["uniforms::img1"], rows[1].rag_trace["asset_ids"])

    def test_metadata_is_patched_key_by_key(self):
        """A turn writes the keys it changed. The note, written behind it by another job,
        survives — and a key set to None is stored as null, which is how a pending
        question is cleared."""
        self._append("human", "hello", metadata={"title": "Hello", "pending_hitl": {"id": "q1"}})
        self.storage.patch_metadata("parent", "s", {"persistent_note": "asked about fees"})
        self._append("ai", "hi", metadata={"pending_hitl": None})

        self.assertEqual(
            {"title": "Hello", "pending_hitl": None, "persistent_note": "asked about fees"},
            self.storage.session_metadata("parent", "s"),
        )

    def test_a_write_invalidates_the_cached_conversation(self):
        self._append("human", "first")
        self.storage.get_session_messages("parent", "s")  # cached
        self._append("ai", "second")

        page = self.storage.get_session_page("parent", "s", limit=10)
        self.assertEqual(["first", "second"], [message["content"] for message in page["messages"]])
        self.assertEqual([], self.storage.list_session_infos("parent")[0:0])  # no error; list rebuilt
        self.assertEqual(2, self.storage.list_session_infos("parent")[0]["message_count"])

    def test_append_returns_the_row_ids_in_order(self):
        ids = self.storage.append(
            "parent", "s", [MessageToStore("human", "a"), MessageToStore("ai", "b")]
        )
        self.assertEqual(2, len(ids))
        self.assertEqual(sorted(ids), ids)
        self.assertEqual(ids, [row.id for row in self._rows()])

    def test_an_unknown_user_stores_nothing(self):
        self.assertEqual([], self.storage.append("stranger", "s", [MessageToStore("human", "a")]))
        self.assertEqual([], self._rows())


class _AgentThatStalls:
    """Streams one chunk, then never finishes — the shape of a turn cut off mid-answer."""

    def __init__(self, ctx, *args, **kwargs):
        self.ctx = ctx

    async def astream(self, payload, stream_mode=None, config=None):
        yield AIMessageChunk(content="The fees for Year 1 are "), {}
        await asyncio.sleep(30)


class StreamedTurnStorageTests(unittest.TestCase):
    """The streamed entry point over the real background runner.

    Driven the way the production bugs happened: a consumer that stops reading at
    `[DONE]`, a consumer cancelled mid-answer, and a note update slower than the turn.
    """

    def setUp(self):
        self.jobs = BackgroundJobs(lanes={WRITES: 2, MODELS: 1}, name="test-turn-jobs")
        self.addCleanup(self.jobs.shutdown, 5.0)
        self.note = Mock(return_value="")
        self._patches = [
            patch.object(service, "plan_turn", lambda *a, **k: (TurnPlan(), RequestSignals())),
            patch.object(service, "generate_session_title", Mock(return_value="title")),
            patch.object(service, "_update_persistent_note_sync", self.note),
        ]
        for item in self._patches:
            item.start()
            self.addCleanup(item.stop)

    def _services(self, storage):
        return Services(conversations=storage, background_jobs=self.jobs)

    def _agent(self, factory):
        patcher = patch.object(service, "create_agent_for_request", factory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_answer_is_queued_for_storage_before_done_is_sent(self):
        """The browser that leaves at `[DONE]`. The generator is closed the moment the
        sentinel arrives, before the server has said anything more."""
        storage = FakeStorage()
        self._agent(lambda ctx, *a, **k: FakeStreamAgent(ctx, chunks=["The bus leaves at 07:30."]))

        async def leave_at_done():
            stream = service.chat_with_agent_stream("bus?", "u", "s", services=self._services(storage))
            async for chunk in stream:
                if chunk == service._DONE:
                    break
            await stream.aclose()

        asyncio.run(leave_at_done())
        self.assertTrue(self.jobs.drain(timeout=5))

        self.assertEqual(["bus?", "The bus leaves at 07:30."], [m.content for m in storage.messages])
        self.assertEqual("title", storage.metadata["title"])

    def test_the_stream_reports_the_row_ids_once_the_turn_is_stored(self):
        """After `[DONE]`, a `stored` event names the rows the question and the answer now
        have. The client keys its copies on them when it reopens the conversation."""
        storage = FakeStorage()
        self._agent(lambda ctx, *a, **k: FakeStreamAgent(ctx, chunks=["07:30."]))

        async def whole_stream():
            return [chunk async for chunk in service.chat_with_agent_stream("bus?", "u", "s", services=self._services(storage))]

        chunks = asyncio.run(whole_stream())

        done_at = chunks.index(service._DONE)
        stored = [json.loads(c[len("data: "):]) for c in chunks[done_at + 1:] if c.startswith("data: {")]
        self.assertEqual([{"type": "stored", "message_ids": [1, 2]}], stored)

    def test_a_stream_cut_off_mid_answer_stores_what_the_parent_saw(self):
        """Stop pressed, or the connection dropped, while the model was still talking.
        The question was already stored; what had reached the parent is stored as its
        answer and marked interrupted, so the reopened chat does not show a question the
        assistant apparently never answered."""
        storage = FakeStorage()
        self._agent(_AgentThatStalls)

        async def cut_off_after_first_words():
            stream = service.chat_with_agent_stream("fees?", "u", "s", services=self._services(storage))

            async def consume():
                async for chunk in stream:
                    if '"type": "content"' in chunk:
                        return  # the browser has the first words on screen; now it goes away

            consumer = asyncio.create_task(consume())
            await consumer
            await stream.aclose()

        asyncio.run(cut_off_after_first_words())
        self.assertTrue(self.jobs.drain(timeout=5))

        self.assertEqual("The fees for Year 1 are ", storage.messages[-1].content)
        stored = storage.appends[-1]["messages"][-1]
        self.assertEqual("ai", stored.message_type)
        self.assertTrue(stored.rag_trace["turn_interrupted"])

    def test_the_note_runs_behind_the_turn_on_the_note_as_it_is_then(self):
        """The turn is due a note update (a note exists). The stream must finish without
        waiting for the model call, and the job must build on the note as stored when it
        RUNS — not on the copy the turn loaded — because the note job ahead of it in the
        queue may have rewritten it."""
        from backend.chat.turn_pipeline import TurnPipeline

        storage = FakeStorage(metadata={"persistent_note": "asked about the uniform"})
        self._agent(lambda ctx, *a, **k: FakeStreamAgent(ctx, chunks=["07:30."]))
        seen = {}

        def note(current_note, user_text, ai_response, *, history_messages=None):
            seen["current"] = current_note
            return f"{current_note}; the bus leaves at 07:30"

        self.note.side_effect = note
        # The previous turn's note job, still running: this turn's queues behind it.
        release = threading.Event()
        self.jobs.submit(TurnPipeline.note_key("u", "s"), lambda: release.wait(5), lane=MODELS)

        async def whole_stream():
            return [chunk async for chunk in service.chat_with_agent_stream("bus?", "u", "s", services=self._services(storage))]

        chunks = asyncio.run(whole_stream())
        self.assertIn(service._DONE, chunks)
        self.assertEqual("07:30.", storage.messages[-1].content, "the answer is stored when the stream ends")
        self.assertEqual([], storage.patches, "the note has not been written: the stream did not wait for it")

        # The previous turn's job finishes by folding itself into the note.
        storage.metadata["persistent_note"] = "asked about the uniform; asked about fees"
        release.set()
        self.assertTrue(self.jobs.drain(timeout=5))

        self.assertEqual("asked about the uniform; asked about fees", seen["current"])
        self.assertEqual(
            [{
                "persistent_note": "asked about the uniform; asked about fees; the bus leaves at 07:30",
                # Stamped with the guardian it was written for; this turn has none.
                "persistent_note_guardian": None,
            }],
            storage.patches,
        )

    def test_the_sync_entry_point_returns_with_the_answer_stored(self):
        storage = FakeStorage()

        class _Agent:
            def __init__(self, ctx, *a, **k):
                pass

            def invoke(self, payload, config=None):
                return {"messages": [*payload["messages"], AIMessage(content="07:30.")]}

        self._agent(_Agent)
        response = service.chat_with_agent("bus?", "u", "s", services=self._services(storage))

        self.assertEqual("07:30.", response["response"])
        self.assertEqual(["bus?", "07:30."], [m.content for m in storage.messages])

    def test_the_next_turn_waits_for_the_previous_turns_save(self):
        """Two messages in quick succession: the second turn opens while the first turn's
        save is still queued, and must still see that turn in its history — here, by not
        mistaking itself for the conversation's first message and naming the session."""
        from backend.chat.turn_pipeline import TurnPipeline

        storage = FakeStorage()
        gate = threading.Event()

        def the_previous_turns_save():
            gate.wait(5)
            storage.append("u", "s", [MessageToStore("human", "bus?"), MessageToStore("ai", "07:30.")])

        self.jobs.submit(TurnPipeline.conversation_key("u", "s"), the_previous_turns_save)
        self._agent(lambda ctx, *a, **k: FakeStreamAgent(ctx, chunks=["Arabic, Maths, PE."]))
        threading.Timer(0.2, gate.set).start()

        async def next_turn():
            return [chunk async for chunk in service.chat_with_agent_stream(
                "and tomorrow's timetable?", "u", "s", services=self._services(storage)
            )]

        asyncio.run(next_turn())

        self.assertEqual(
            ["bus?", "07:30.", "and tomorrow's timetable?", "Arabic, Maths, PE."],
            [m.content for m in storage.messages],
        )
        self.assertNotIn("title", storage.metadata, "the turn saw the earlier messages, so it was not the first")


if __name__ == "__main__":
    unittest.main()
