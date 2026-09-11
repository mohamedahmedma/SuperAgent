"""End to end, as a parent experiences it: what actually arrives on the wire.

Every other test in this batch checks one component. This one drives the real
`chat_with_agent_stream` — turn planner, agent loop, middleware, tools, finalizer,
grounding check, SSE encoding — and asserts on the events a browser would receive. The
model is scripted, so the scenarios are free, deterministic, and reproduce the exact
outputs measured from `openai/gpt-oss-20b` on Together rather than invented ones.

The four defects being held down, all of which reached a real parent on one turn:

    1  the model's private reasoning, and a pre-tool answer, streamed as the answer
    2  a fee figure and a citation invented outright
    3  the same search issued two to five times
    4  a Year 4 fee quoted to the parent of a Year 1 child

They interacted, which is why they are exercised together here and separately
elsewhere: the duplicate calls (3) disengaged the guard that would have stopped the
fabrication (2), and the missing year (4) is what made a wrong row plausible.

Scenario names read as the parent's situation, not as the code path, so a failure says
what a person would have seen.
"""
import importlib
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

import backend.chat.runtime as runtime

service = importlib.import_module("backend.chat.service")

# ---------------------------------------------------------------------------
# The school's actual material, as retrieval would hand it over.
# ---------------------------------------------------------------------------

YEAR_1 = "الصف الأول الابتدائي"

FEE_CHUNKS = [
    {"filename": "fees_2026.pdf", "page_number": "3",
     "text": "رسوم الصف الأول الابتدائي للعام 2026: 30,000 جنيه على ثلاث دفعات."},
    {"filename": "fees_2026.pdf", "page_number": "4",
     "text": "رسوم الصف الثاني الابتدائي: 35,000 جنيه على ثلاث دفعات."},
    {"filename": "fees_2026.pdf", "page_number": "5",
     "text": "رسوم الصف الرابع الابتدائي: 45,000 جنيه على ثلاث دفعات."},
]

# A document written once for everybody. The case where narrowing to a year would be
# the WRONG answer, and the reason the grader's `discriminate` verdict exists.
GENERAL_CHUNKS = [
    {"filename": "transfer.pdf", "page_number": "1",
     "text": "أوراق التحويل المطلوبة لكل الصفوف: شهادة الميلاد، آخر شهادة درجات، وصورة البطاقة."},
]


def _trace(chunks, status="answerable", discriminate="yes"):
    return {
        "retrieval_status": status,
        "retrieved_chunks": list(chunks),
        "evidence_constraints_discriminate": discriminate,
    }


# ---------------------------------------------------------------------------
# A model that says exactly what the real one was measured saying.
# ---------------------------------------------------------------------------


class ScriptedAgent:
    """Streams a scripted sequence of assistant messages through the real service.

    Each entry is `(message_id, [chunk, ...], tool_calls)`. Chunks are streamed one at a
    time so the finalizer's incremental path is what runs — the same code that has to
    cope with a Harmony token split across two deltas.
    """

    def __init__(self, ctx, script, trace=None, tool_results=0, on_run=None):
        self.ctx = ctx
        self.script = script
        self.trace = trace
        self.tool_results = tool_results
        # What a tool would have done to the context during the run — noting an answer
        # block, say — for scenarios whose subject is what the turn does with it.
        self.on_run = on_run

    async def astream(self, payload, stream_mode=None, config=None):
        if self.on_run is not None:
            self.on_run(self.ctx)
        if self.trace is not None:
            self.ctx.store_rag_trace(self.trace, None)
            # The real middleware ends the turn on a terminal verdict and records it on
            # the context, and `_agent_worker`'s `finally` puts the profile copy on the
            # wire from that record. A scripted agent that skipped this would make the
            # no-knowledge scenarios assert against an empty bubble and pass for the
            # wrong reason. See `backend/chat/runtime.py`.
            status = self.trace.get("retrieval_status")
            if status in runtime.TERMINAL_STATUSES:
                self.ctx.note_short_circuit(status)
                return
        for message_id, chunks, tool_calls in self.script:
            if tool_calls:
                yield AIMessageChunk(
                    content="", id=message_id, tool_call_chunks=tool_calls
                ), {}
            for chunk in chunks:
                yield AIMessageChunk(content=chunk, id=message_id), {}


def _tool_chunk(index=0):
    return [{"name": "search_knowledge_base", "args": '{"query":"x"}',
             "id": f"c{index}", "index": index}]


def _split(text, size=7):
    return [text[i:i + size] for i in range(0, len(text), size)]


def _parse_sse(chunks):
    events = []
    for chunk in chunks:
        payload = chunk.strip()
        if not payload.startswith("data: "):
            continue
        data = payload[len("data: "):]
        events.append({"type": "DONE"} if data == "[DONE]" else json.loads(data))
    return events


def _text_shown(events):
    """What the message bubble ends up holding, replaces applied as the browser does."""
    shown = ""
    for event in events:
        if event.get("type") == "content":
            shown += event.get("content", "")
        elif event.get("type") == "content_replace":
            shown = event.get("content", "")
    return shown


class ParentTurnScenario(unittest.IsolatedAsyncioTestCase):
    """Base: run one turn through the real stream and read what arrived."""

    def setUp(self):
        from backend.chat.orchestrator import _hand_to_graph
        from backend.chat.signals import RequestSignals
        from backend.chat.turn_policy import TurnPlan

        self.plan = TurnPlan()
        # What the parent typed. Overridden per scenario where the wording matters.
        self.question = "مصاريف ابني كام"

        def fake_plan_turn(question, history=None, ctx=None, **kwargs):
            # The planner is stubbed so these scenarios never reach a live classifier,
            # but the HANDOFF is not: `_hand_to_graph` is what puts the plan's child
            # year and conditions on the context, and stubbing it out too would let a
            # test assert that the year arrived when nothing had carried it.
            _hand_to_graph(ctx, self.plan)
            return self.plan, RequestSignals()

        self._planner = patch.object(service, "plan_turn", fake_plan_turn)
        self._planner.start()
        self.addCleanup(self._planner.stop)

    def new_storage(self):
        """A conversation store that survives between turns of one test.

        Passed to `run_turn` when a scenario needs more than one turn: the second turn
        has to see the first one's history, which is where a follow-up gets its meaning.
        """
        from tests.general.test_chat_hitl_resume import FakeStorage

        return FakeStorage([])

    async def run_turn(self, script, trace=None, storage_messages=None, storage=None,
                       question=None, on_run=None):
        from tests.general.test_chat_hitl_resume import FakeStorage

        captured = {}

        def make_agent(ctx, tool_names=None, language=None):
            captured["ctx"] = ctx
            return ScriptedAgent(ctx, script, trace, on_run=on_run)

        if question is not None:
            self.question = question

        chunks = []
        with (
            patch.object(service, "storage", storage or FakeStorage(storage_messages or [])),
            patch.object(service, "create_agent_for_request", make_agent),
            patch.object(service, "generate_session_title", Mock(return_value="سؤال")),
            patch.object(service, "update_persistent_note", AsyncMock(return_value="")),
        ):
            async for chunk in service.chat_with_agent_stream(
                self.question, "parent-1", "session-1"
            ):
                chunks.append(chunk)

        events = _parse_sse(chunks)
        return events, _text_shown(events), captured.get("ctx")


class ReasoningNeverReachesTheParent(ParentTurnScenario):
    """Bug 1. Four shapes, all measured from the live provider."""

    async def test_the_full_transcript_envelope_is_stripped(self):
        envelope = (
            "<|channel|>analysis<|message|>We have three chunks. The user asked about "
            "fees. Provide both options.<|end|><|start|>assistant<|channel|>final"
            "<|message|>رسوم الصف الأول الابتدائي 30,000 جنيه على ثلاث دفعات. [1]"
        )
        _, shown, _ = await self.run_turn(
            [("m1", [], _tool_chunk()), ("m2", _split(envelope), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertEqual(shown.strip(), "رسوم الصف الأول الابتدائي 30,000 جنيه على ثلاث دفعات. [1]")
        self.assertNotIn("<|", shown)
        self.assertNotIn("We have three chunks", shown)

    async def test_an_answer_written_before_the_tool_ran_is_never_shown(self):
        """The measured failure that showed two contradictory answers in one bubble: the
        model writes a confident refusal alongside its tool call, then answers properly
        from the chunks a moment later."""
        premature = "عذرًا، لا أستطيع العثور على معلومات حول مصاريف ابنك."
        real = "رسوم الصف الأول الابتدائي 30,000 جنيه على ثلاث دفعات. [1]"
        _, shown, _ = await self.run_turn(
            [("m1", _split(premature), _tool_chunk()), ("m2", _split(real), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertEqual(shown.strip(), real)
        self.assertNotIn("لا أستطيع", shown)

    async def test_bare_reasoning_prose_with_no_marker_is_dropped(self):
        """The shape no text rule can find — it is caught by WHICH message carries it."""
        leak = "We need to see the result. Let's assume the knowledge base returns something."
        real = "رسوم الصف الأول الابتدائي 30,000 جنيه. [1]"
        _, shown, _ = await self.run_turn(
            [("m1", _split(leak), _tool_chunk()), ("m2", _split(real), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertNotIn("We need to see", shown)
        self.assertEqual(shown.strip(), real)

    async def test_a_bare_channel_header_is_stripped_from_the_answer(self):
        header = ("commentary to=functions.search_knowledge_base "
                  "رسوم الصف الأول الابتدائي 30,000 جنيه. [1]")
        _, shown, _ = await self.run_turn(
            [("m1", [], _tool_chunk()), ("m2", _split(header), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertNotIn("to=functions", shown)
        self.assertIn("30,000", shown)

    async def test_a_clean_answer_arrives_byte_for_byte(self):
        """The turn that was never broken must not become the turn this broke."""
        real = "رسوم الصف الأول الابتدائي 30,000 جنيه على ثلاث دفعات. [1]"
        _, shown, _ = await self.run_turn(
            [("m1", [], _tool_chunk()), ("m2", _split(real, 3), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertEqual(shown, real)

    async def test_the_trace_reports_what_was_withheld(self):
        events, _, _ = await self.run_turn(
            [("m1", _split("invented pre-tool answer"), _tool_chunk()),
             ("m2", _split("رسوم الصف الأول 30,000 جنيه. [1]"), None)],
            trace=_trace(FEE_CHUNKS),
        )
        trace = next(e["rag_trace"] for e in events if e.get("type") == "trace")
        self.assertEqual(trace["finalize_dropped_tool_call_messages"], 1)
        self.assertGreater(trace["finalize_dropped_chars"], 0)


class TheRightYearIsAnswered(ParentTurnScenario):
    """Bug 4, through the whole stack rather than at the template."""

    async def test_the_year_reaches_the_graph_beside_the_question(self):
        self.plan.child_hint = "علي"
        self.plan.child_year = YEAR_1
        _, _, ctx = await self.run_turn(
            [("m1", [], _tool_chunk()),
             ("m2", _split("رسوم الصف الأول الابتدائي 30,000 جنيه. [1]"), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertEqual(ctx.child_year, YEAR_1)

    async def test_the_grounding_check_does_not_catch_a_wrong_year(self):
        """A limit, asserted so nobody mistakes the two fixes for one.

        The Year 4 row really is in the corpus, so every figure in this answer is
        grounded and the check serves it — correctly, by its own rule. Nothing here is
        broken: the grounding check verifies that a number came from the evidence, and
        it cannot know which year group the parent's child is in.

        What stops this answer being written is the year binding in the prompt, which
        is asserted in `test_child_year_binding.py`. If someone later makes the
        grounding check year-aware and this test starts failing, that is a real
        improvement — delete it then, deliberately, rather than being surprised by it.
        """
        self.plan.child_hint = "علي"
        self.plan.child_year = YEAR_1
        _, shown, _ = await self.run_turn(
            [("m1", [], _tool_chunk()),
             ("m2", _split("رسوم الصف الرابع الابتدائي 45,000 جنيه. [3]"), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertIn("45,000", shown)


class OneSearchPerQuestion(ParentTurnScenario):
    """Bug 3, at the level a parent feels it.

    The collapse MECHANISM is tested against the real agent loop in
    `test_duplicate_tool_calls.py`; this scripted agent bypasses the middleware, so what
    is checked here is only the parent-visible outcome — that a turn which asked several
    times still produces one coherent answer with no tool plumbing in it.
    """

    async def test_repeated_calls_do_not_put_a_tool_error_in_front_of_the_parent(self):
        """Copies beyond the budget returned TOOL_CALL_LIMIT_REACHED, which the model
        narrates around — text the parent then read."""
        script = [("m1", [], _tool_chunk(i)) for i in range(3)]
        script.append(("m2", _split("رسوم الصف الأول 30,000 جنيه. [1]"), None))
        _, shown, _ = await self.run_turn(script, trace=_trace(FEE_CHUNKS))
        self.assertNotIn("TOOL_CALL_LIMIT", shown)
        self.assertNotIn("tool call failed", shown.lower())
        self.assertEqual(shown.strip(), "رسوم الصف الأول 30,000 جنيه. [1]")


class TheCorpusSaidNothing(ParentTurnScenario):
    """Bugs 2 and 3 together: the interaction that produced the shipped transcript."""

    async def test_no_knowledge_is_answered_with_the_school_s_own_words(self):
        _, shown, _ = await self.run_turn(
            [("m1", [], _tool_chunk())],
            trace=_trace([], status="no_knowledge"),
        )
        self.assertEqual(shown.strip(), service._COPY.no_knowledge.strip())


if __name__ == "__main__":
    unittest.main()


class AConversationThatBuilds(ParentTurnScenario):
    """Two turns, where the second only makes sense against the first.

    Almost every real fee conversation is this shape: the parent asks what the fees
    are, hears a total, and immediately asks what one instalment costs. The follow-up
    states a figure no document contains, so it is the turn where a verifier that
    cannot follow arithmetic starts blocking correct answers — and a verifier that
    blocks correct answers gets switched off.
    """

    async def test_a_follow_up_instalment_is_served_not_blocked(self):
        storage = self.new_storage()
        _, first, _ = await self.run_turn(
            [("m1", [], _tool_chunk()),
             ("m2", _split("رسوم الصف الأول 30,000 جنيه على ثلاث دفعات. [1]"), None)],
            trace=_trace(FEE_CHUNKS), storage=storage,
        )
        self.assertIn("30,000", first)

        _, second, _ = await self.run_turn(
            [("m3", [], _tool_chunk()),
             ("m4", _split("كل دفعة 10,000 جنيه. [1]"), None)],
            trace=_trace(FEE_CHUNKS), storage=storage,
            question="يعني الدفعة الواحدة كام؟",
        )
        self.assertEqual(second.strip(), "كل دفعة 10,000 جنيه. [1]")
        self.assertNotEqual(second, service._COPY.unverified_answer)

    async def test_the_second_turn_sees_the_first_one_s_history(self):
        storage = self.new_storage()
        await self.run_turn(
            [("m1", [], _tool_chunk()),
             ("m2", _split("رسوم الصف الأول 30,000 جنيه. [1]"), None)],
            trace=_trace(FEE_CHUNKS), storage=storage,
        )
        await self.run_turn(
            [("m3", _split("تمام."), None)],
            trace=_trace(FEE_CHUNKS), storage=storage, question="شكرا",
        )
        texts = [getattr(m, "content", "") for m in storage.messages]
        self.assertIn("مصاريف ابني كام", texts)
        self.assertIn("شكرا", texts)


class ALongHelpfulAnswer(ParentTurnScenario):
    """Not a one-line reply — the multi-paragraph, bulleted answer a parent gets when
    they ask a broad question. Long answers are where a transcript leak hides in the
    middle rather than at the front, and where markdown gives the chunker more ways to
    split a Harmony token."""

    ANSWER = (
        "رسوم الصف الأول الابتدائي للعام 2026 هي **30,000 جنيه** [1]، وتُدفع كالتالي:\n\n"
        "- الدفعة الأولى: 10,000 جنيه\n"
        "- الدفعة الثانية: 10,000 جنيه\n"
        "- الدفعة الثالثة: 10,000 جنيه\n\n"
        "لو حضرتك عايز تفاصيل عن مواعيد السداد أقدر أساعدك."
    )

    async def test_a_leak_in_the_middle_of_a_long_answer_is_removed(self):
        leaked = (
            "<|channel|>analysis<|message|>The user wants fees. Chunk 1 has 30,000. "
            "I should break it into instalments.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>" + self.ANSWER
        )
        _, shown, _ = await self.run_turn(
            [("m1", [], _tool_chunk()), ("m2", _split(leaked, 5), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertEqual(shown.strip(), self.ANSWER)
        self.assertNotIn("<|", shown)
        self.assertNotIn("I should break it", shown)

    async def test_the_markdown_structure_survives_intact(self):
        _, shown, _ = await self.run_turn(
            [("m1", [], _tool_chunk()), ("m2", _split(self.ANSWER, 4), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertEqual(shown, self.ANSWER)
        self.assertEqual(shown.count("- الدفعة"), 3)

    async def test_every_instalment_figure_is_verified_not_just_the_total(self):
        """Three 10,000 lines and one 30,000 total — all derived from one chunk."""
        _, shown, _ = await self.run_turn(
            [("m1", [], _tool_chunk()), ("m2", _split(self.ANSWER, 11), None)],
            trace=_trace(FEE_CHUNKS),
        )
        self.assertNotEqual(shown, service._COPY.unverified_answer)
