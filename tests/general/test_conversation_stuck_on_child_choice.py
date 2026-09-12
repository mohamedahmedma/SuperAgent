# -*- coding: utf-8 -*-
"""The conversation that got stuck on «which child?», defect by defect.

On 2026-09-11 a father was asked which of his children he meant, was offered the same
name twice, tapped one — and every message he sent afterwards was answered with the
subjects list that question had been about. Sunday's lessons, Monday's lessons, the term
dates: the same answer, with the tool's own evidence header pasted into it.

`test_parent_conversation_replay.py` replays that conversation whole. This file holds the
unit tests for each thing that was wrong, grouped by the component it lived in, so a
regression says WHICH rule broke rather than that a fifteen-message replay did.

  * The answered "which child?" question was never cleared — `_TurnEntry`, both save sites.
  * A short-circuit turn dropped the child pin — `_stream_static_reply`.
  * The evidence cut ran only on turns with a table — `_settle_answer_blocks`.
  * One child listed twice, and two children under one name, made the question
    unanswerable — `child_roster`, `child_resolution`, `_pin_the_child_the_parent_named`.
  * A spelling of Monday the school's vocabulary did not carry — `school_week`.
"""
import importlib
import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk

from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_context import SessionChild
from backend.chat.child_resolution import resolve_child
from backend.chat.child_roster import ChildOption, _as_options
from backend.chat.request_context import ChatRequestContext
from backend.chat.signals import RequestSignals
from backend.chat.turn_policy import TurnPlan
from backend.school_week import day_phrase
from tests.general.test_chat_hitl_resume import FakeStorage

service = importlib.import_module("backend.chat.service")

GUARDIAN = "G-stuck"
TOKEN = "signed.identity.token"
USER = "parent-stuck"
SESSION = "session-stuck"

FATMA = "فاطمه محمد ابوالحسن"
DAUGHTER = {"student_id": "S-1", "full_name_ar": FATMA, "full_name_en": "Fatma", "gender": "female", "year_level": "Year 11"}
NAMESAKE = {"student_id": "S-2", "full_name_ar": FATMA, "gender": "male", "year_level": "Year 9"}
BROTHER = {"student_id": "S-3", "full_name_ar": "عمر محمد ابوالحسن", "gender": "male", "year_level": "Year 6"}

ORIGINAL = "طيب بتاخد مواد اي"
SUNDAY = "طيب هي يوم الحد عندها حصص اي"
WHICH = "حضرتك تقصد أنهي واحد فيهم؟"


# ---------------------------------------------------------------------------
# Driving the real service with a scripted planner and model
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _roster(rows):
    def fake_get(url, headers=None, params=None, timeout=None):
        return _Response(200, {"guardian_id": GUARDIAN, "students": list(rows)})

    return fake_get


def _asks(options):
    """A plan that ends the turn on "which child?"."""
    plan = TurnPlan()
    plan.child_options = list(options)
    plan.static_reply = WHICH
    plan.exposed_tools = []
    return plan


def _social(reply="العفو"):
    plan = TurnPlan()
    plan.static_reply = reply
    plan.exposed_tools = []
    return plan


class _StreamAgent:
    def __init__(self, ctx, reply, *, fail=False, on_run=None):
        self.ctx, self.reply, self.fail, self.on_run = ctx, reply, fail, on_run

    async def astream(self, payload, stream_mode=None, config=None):
        if self.on_run is not None:
            self.on_run(self.ctx)
        if self.fail:
            raise RuntimeError("provider timed out")
        for start in range(0, len(self.reply), 9):
            yield AIMessageChunk(content=self.reply[start:start + 9], id="answer"), {}


class _SyncAgent:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, payload, config=None):
        return {"messages": [AIMessage(content=self.reply)]}


def _events(chunks):
    out = []
    for chunk in chunks:
        body = chunk.strip()
        if body.startswith("data: "):
            data = body[len("data: "):]
            out.append({"type": "DONE"} if data == "[DONE]" else json.loads(data))
    return out


def _shown(events):
    text = ""
    for event in events:
        if event.get("type") == "content":
            text += event.get("content", "")
        elif event.get("type") == "content_replace":
            text = event.get("content", "")
    return text


class _Session(unittest.IsolatedAsyncioTestCase):
    """One stored conversation, driven message by message through the real stream.

    `plans` maps the text the PLANNER receives to the plan it returns; `replies` maps it
    to what the model says. Both are keyed by planner input rather than by what the parent
    typed, which is exactly the distinction the bug was about.
    """

    roster = [DAUGHTER, NAMESAKE]

    def setUp(self):
        self.storage = FakeStorage([])
        self.plans = {}
        self.replies = {}
        self.planned = []
        self.fail_agent = False
        self.on_run = None
        env = patch.dict(os.environ, {"CHILD_ROSTER_TTL_SECONDS": "0"})
        env.start()
        self.addCleanup(env.stop)

    def _plan(self, question, history=None, ctx=None, **kwargs):
        self.planned.append(question)
        return self.plans.get(question, TurnPlan()), RequestSignals(question=question)

    def _agent(self, ctx, tool_names=None, language=None):
        asked = self.planned[-1] if self.planned else ""
        return _StreamAgent(ctx, self.replies.get(asked, ""), fail=self.fail_agent, on_run=self.on_run)

    async def say(self, text):
        caller = CallerIdentity(user_id=USER, guardian_id=GUARDIAN, guardian_token=TOKEN)
        chunks = []
        with (
            patch("backend.chat.child_roster.requests.get", _roster(self.roster)),
            patch.object(service, "storage", self.storage),
            patch.object(service, "plan_turn", self._plan),
            patch.object(service, "create_agent_for_request", self._agent),
            patch.object(service, "generate_session_title", Mock(return_value="س")),
            patch.object(service, "update_persistent_note", AsyncMock(return_value="")),
        ):
            async for chunk in service.chat_with_agent_stream(text, USER, SESSION, caller=caller):
                chunks.append(chunk)
        return _events(chunks)

    @property
    def pending(self):
        return self.storage.metadata.get("pending_hitl")

    @property
    def pin(self):
        return dict(self.storage.metadata.get("child_context") or {})


# ---------------------------------------------------------------------------
# A. The answered question is spent
# ---------------------------------------------------------------------------


class TheRuleForSpendingAPendingQuestion(unittest.TestCase):
    """`_TurnEntry.spends_the_pending_question`, the one place every save site asks."""

    def test_a_child_choice_spends_it(self):
        self.assertTrue(service._TurnEntry(child_choice=FATMA).spends_the_pending_question())

    def test_a_child_choice_spends_it_even_when_the_agent_then_fails(self):
        """The pin was written before the agent ran. Keeping the question would read
        the parent's next message as another name."""
        entry = service._TurnEntry(child_choice=FATMA)
        self.assertTrue(entry.spends_the_pending_question(agent_error=True))

    def test_a_resumed_clarification_spends_it(self):
        self.assertTrue(service._TurnEntry(is_hitl_resume=True).spends_the_pending_question())

    def test_a_superseded_clarification_spends_it(self):
        self.assertTrue(service._TurnEntry(superseded=True).spends_the_pending_question())

    def test_a_retrieval_clarification_survives_an_agent_error(self):
        """Unchanged behaviour, stated: the answer was never used, so the parent can
        retry it."""
        entry = service._TurnEntry(is_hitl_resume=True)
        self.assertFalse(entry.spends_the_pending_question(agent_error=True))

    def test_an_ordinary_turn_spends_nothing(self):
        self.assertFalse(service._TurnEntry().spends_the_pending_question())


class TheAnsweredChildQuestionIsSpent(_Session):
    """The defect that reached production, on the streaming path parents actually use."""

    def setUp(self):
        super().setUp()
        self.plans[ORIGINAL] = _asks([f"{FATMA} — Year 11", f"{FATMA} — Year 9"])
        self.replies[ORIGINAL] = "بتاخد العربي والإنجليزي."
        self.replies[SUNDAY] = "يوم الأحد عندها كيمياء."

    async def test_the_pending_state_is_cleared_once_the_child_is_chosen(self):
        await self.say(ORIGINAL)
        self.assertEqual(self.pending["route"], "child_select")

        self.plans[ORIGINAL] = TurnPlan()  # the re-plan settles her and answers
        await self.say(f"{FATMA} — Year 11")

        self.assertIsNone(self.pending)

    async def test_the_next_message_is_planned_as_the_parent_typed_it(self):
        """Fifteen messages into the real conversation, this was false for six of them."""
        await self.say(ORIGINAL)
        self.plans[ORIGINAL] = TurnPlan()
        await self.say(f"{FATMA} — Year 11")

        events = await self.say(SUNDAY)

        self.assertEqual(self.planned[-1], SUNDAY)
        self.assertIn("كيمياء", _shown(events))
        self.assertNotIn("العربي والإنجليزي", _shown(events))

    async def test_the_choice_is_spent_even_when_the_agent_fails(self):
        await self.say(ORIGINAL)
        self.plans[ORIGINAL] = TurnPlan()
        self.fail_agent = True
        await self.say(f"{FATMA} — Year 11")

        self.assertIsNone(self.pending)
        self.assertEqual(self.pin.get("student_id"), "S-1")

    async def test_choosing_badly_asks_again_and_stores_the_new_question(self):
        """The one case the old code got right must stay right: a reply matching nobody
        re-plans, the re-plan asks again, and THAT question is what is stored."""
        await self.say(ORIGINAL)
        first_id = self.pending["id"]

        await self.say("الكبيرة")

        self.assertEqual(self.pending["route"], "child_select")
        self.assertNotEqual(self.pending["id"], first_id)
        self.assertEqual(self.pending["original_question"], ORIGINAL)

    async def test_a_retrieval_clarification_still_outlives_an_agent_error(self):
        """The sibling route keeps its existing contract."""
        self.storage.metadata["pending_hitl"] = service._build_pending_hitl(
            {"retrieval_status": "needs_clarification", "hitl_prompt": "أي سنة؟"},
            "مصاريف كام",
        )
        resolution = Mock(supersedes_pending_question=False, question="مصاريف كام", constraints=[])
        self.fail_agent = True
        with patch.object(service, "resolve_turn_question", Mock(return_value=resolution)):
            await self.say("الصف الرابع")

        self.assertIsNotNone(self.pending)
        self.assertEqual(self.pending["route"], "clarify")


class TheSyncPathSpendsItToo(unittest.TestCase):
    """`chat_with_agent` has its own save site, and it had the same hole."""

    def setUp(self):
        env = patch.dict(os.environ, {"CHILD_ROSTER_TTL_SECONDS": "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_the_pending_state_is_cleared_once_the_child_is_chosen(self):
        storage = FakeStorage([])
        storage.metadata["pending_hitl"] = service._child_choice_pending(
            _asks([f"{FATMA} — Year 11", f"{FATMA} — Year 9"]), ORIGINAL
        )
        planned = []

        def plan(question, history=None, ctx=None, **kwargs):
            planned.append(question)
            return TurnPlan(), RequestSignals(question=question)

        caller = CallerIdentity(user_id=USER, guardian_id=GUARDIAN, guardian_token=TOKEN)
        with (
            patch("backend.chat.child_roster.requests.get", _roster([DAUGHTER, NAMESAKE])),
            patch.object(service, "storage", storage),
            patch.object(service, "plan_turn", plan),
            patch.object(service, "create_agent_for_request", lambda ctx, *a, **k: _SyncAgent("بتاخد العربي.")),
            patch.object(service, "generate_session_title", Mock(return_value="س")),
            patch.object(service, "_update_persistent_note_sync", Mock(return_value="")),
        ):
            service.chat_with_agent(f"{FATMA} — Year 11", USER, SESSION, caller=caller)

        self.assertEqual(planned, [ORIGINAL])
        self.assertIsNone(storage.metadata.get("pending_hitl"))
        self.assertEqual(storage.metadata["child_context"]["student_id"], "S-1")


# ---------------------------------------------------------------------------
# D. A short-circuit turn keeps the pin
# ---------------------------------------------------------------------------


class TheStaticReplyKeepsThePin(_Session):
    """`_stream_static_reply` never saved the child state; the two agent paths did."""

    async def test_a_choice_followed_by_static_copy_is_not_forgotten(self):
        """Answer "which child?", and the re-planned turn ends on the profile's own copy.
        The pin written a moment earlier has to reach storage anyway."""
        self.plans[ORIGINAL] = _asks([f"{FATMA} — Year 11", f"{FATMA} — Year 9"])
        await self.say(ORIGINAL)
        self.assertEqual(self.pin.get("student_id", ""), "")

        self.plans[ORIGINAL] = _social()
        await self.say(f"{FATMA} — Year 11")

        self.assertEqual(self.pin.get("student_id"), "S-1")
        self.assertTrue(self.pin.get("chosen_by_parent"))
        self.assertIsNone(self.pending)

    async def test_a_static_turn_with_nothing_to_pin_leaves_the_pin_alone(self):
        self.storage.metadata["child_context"] = {
            "student_id": "S-3", "label": "عمر", "gender": "male", "guardian_id": GUARDIAN,
        }
        self.plans["شكرا"] = _social()
        await self.say("شكرا")
        self.assertEqual(self.pin.get("student_id"), "S-3")


# ---------------------------------------------------------------------------
# C. The evidence cut runs on every answer
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, blocks=()):
        self.answer_blocks = list(blocks)
        self.language = "ar"


LEAKED = (
    f"حضرتك، {FATMA} بتاخد المواد التالية:\n"
    f"SUBJECTS for {FATMA} — الصف الثاني الثانوي: اللغة العربية، الفيزياء. [1]"
)


class TheEvidenceCutRunsWithoutATable(unittest.TestCase):
    def test_a_subjects_answer_loses_the_header_it_pasted(self):
        """Verbatim from production. `subjects` renders no block, and the cut used to
        live behind the block check."""
        settled, blocks = service._settle_answer_blocks(LEAKED, _Ctx())
        self.assertEqual(settled, f"حضرتك، {FATMA} بتاخد المواد التالية:")
        self.assertEqual(blocks, [])

    def test_the_sync_text_only_path_cuts_too(self):
        self.assertNotIn("SUBJECTS", service._append_answer_blocks(LEAKED, _Ctx()))

    def test_every_records_header_is_cut_with_or_without_a_block(self):
        for header in ("CLASS", "TEACHERS", "SUBJECT_TEACHER", "ATTENDANCE", "NO_RECORDS", "NOT_AUTHORIZED"):
            with self.subTest(header=header):
                text = f"الجواب:\n{header} for X: something the model should not have pasted"
                self.assertEqual(service._append_answer_blocks(text, _Ctx()), "الجواب:")

    def test_the_knowledge_tools_headers_are_cut_as_well(self):
        for header in ("NO_KNOWLEDGE", "PARTIAL_EVIDENCE", "RETRIEVAL_ERROR", "NEEDS_CLARIFICATION", "NEEDS_SCOPE_SELECTION"):
            with self.subTest(header=header):
                text = f"عذرًا.\n{header}: instructions to the model"
                self.assertEqual(service._append_answer_blocks(text, _Ctx()), "عذرًا.")

    def test_an_answer_that_was_only_evidence_becomes_the_unverified_copy(self):
        settled, _ = service._settle_answer_blocks(f"SUBJECTS for {FATMA}: العربي", _Ctx())
        self.assertEqual(settled, service._COPY.unverified_answer)

    def test_an_answer_that_said_nothing_stays_silent(self):
        self.assertEqual(service._settle_answer_blocks("", _Ctx()), ("", []))
        self.assertEqual(service._settle_answer_blocks("   ", _Ctx())[0], "")

    def test_a_clean_answer_is_untouched(self):
        clean = "حضرتك، فاطمه في الصف الثاني الثانوي."
        self.assertEqual(service._settle_answer_blocks(clean, _Ctx()), (clean, []))

    def test_english_prose_a_parent_may_legitimately_read_survives(self):
        """The fee and uniform answers from the same conversation."""
        for line in ("Sports Wear: All Grades - Unisex", "Pre-K: 34,000", "Y11-Y12: 72,000 SAR (VAT)"):
            with self.subTest(line=line):
                self.assertEqual(service._append_answer_blocks(line, _Ctx()), line)


class TheLeakIsCutOnTheWire(_Session):
    """Through the real stream: the parent sees the cut text, and so does storage."""

    async def test_a_subjects_turn_that_pasted_evidence_is_replaced_and_stored_cut(self):
        self.replies[ORIGINAL] = LEAKED
        events = await self.say(ORIGINAL)

        shown = _shown(events)
        self.assertNotIn("SUBJECTS", shown)
        self.assertIn("بتاخد المواد التالية", shown)
        self.assertTrue([e for e in events if e.get("type") == "content_replace"])
        self.assertEqual(self.storage.messages[-1].content.strip(), shown.strip())


# ---------------------------------------------------------------------------
# B. Every child once, and every option answerable
# ---------------------------------------------------------------------------


class TheRosterListsEachChildOnce(unittest.TestCase):
    def test_the_same_student_served_twice_is_one_child(self):
        options = _as_options([DAUGHTER, dict(DAUGHTER)])
        self.assertEqual([c.student_id for c in options], ["S-1"])
        self.assertEqual(options[0].label, FATMA)

    def test_the_first_row_wins(self):
        options = _as_options([DAUGHTER, {**DAUGHTER, "year_level": "Year 3"}])
        self.assertEqual(options[0].year_level, "Year 11")

    def test_distinct_children_are_all_kept(self):
        self.assertEqual(len(_as_options([DAUGHTER, NAMESAKE, BROTHER])), 3)


class TwoChildrenUnderOneNameAreToldApart(unittest.TestCase):
    def test_a_shared_name_carries_the_year(self):
        labels = [c.label for c in _as_options([DAUGHTER, NAMESAKE])]
        self.assertEqual(labels, [f"{FATMA} — Year 11", f"{FATMA} — Year 9"])

    def test_every_member_of_the_group_is_suffixed(self):
        """No bare name may survive, or a reply equal to it matches the group."""
        for child in _as_options([DAUGHTER, NAMESAKE]):
            self.assertNotEqual(child.label, FATMA)

    def test_a_child_with_a_unique_name_is_untouched(self):
        options = _as_options([DAUGHTER, NAMESAKE, BROTHER])
        self.assertEqual(options[2].label, "عمر محمد ابوالحسن")

    def test_the_latin_spelling_is_next_when_the_year_is_missing(self):
        rows = [{**DAUGHTER, "year_level": ""}, {**NAMESAKE, "year_level": "", "full_name_en": "Fatma M."}]
        labels = [c.label for c in _as_options(rows)]
        self.assertEqual(labels, [f"{FATMA} — Fatma", f"{FATMA} — Fatma M."])

    def test_a_counter_is_the_last_resort(self):
        rows = [{**DAUGHTER, "year_level": "", "full_name_en": ""}, {**NAMESAKE, "year_level": ""}]
        labels = [c.label for c in _as_options(rows)]
        self.assertEqual(labels, [f"{FATMA} — 1", f"{FATMA} — 2"])

    def test_a_shared_year_falls_through_to_the_next_detail(self):
        rows = [DAUGHTER, {**NAMESAKE, "year_level": "Year 11", "full_name_en": "Fatma B."}]
        labels = [c.label for c in _as_options(rows)]
        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(labels[1], f"{FATMA} — Fatma B.")

    def test_the_options_offered_are_therefore_distinct(self):
        options = _as_options([DAUGHTER, NAMESAKE])
        asked = resolve_child(reference="named", child_name=FATMA, roster=options)
        self.assertTrue(asked.ask)
        self.assertEqual(len(set(asked.option_labels)), 2)

    def test_the_student_number_never_appears_in_a_label(self):
        for rows in ([DAUGHTER, NAMESAKE], [{**DAUGHTER, "year_level": "", "full_name_en": ""}, {**NAMESAKE, "year_level": ""}]):
            for child in _as_options(rows):
                self.assertNotIn(child.student_id, child.label)


class TheParentsOwnChoiceSettlesASharedName(unittest.TestCase):
    """`resolve_child`, route 1, with a name that matches two children."""

    def setUp(self):
        self.roster = _as_options([DAUGHTER, NAMESAKE])

    def test_with_no_pin_it_asks(self):
        out = resolve_child(reference="named", child_name=FATMA, roster=self.roster)
        self.assertTrue(out.ask)

    def test_a_pin_the_records_tool_merely_inferred_still_asks(self):
        """Inference is not a statement about THIS ambiguity. Answering about the
        pinned one could show a sibling's marks to a parent who typed a name."""
        pin = SessionChild(student_id="S-1")
        out = resolve_child(reference="named", child_name=FATMA, roster=self.roster, pin=pin)
        self.assertTrue(out.ask)
        self.assertFalse(out.resolved)

    def test_the_child_the_parent_chose_is_resolved(self):
        pin = SessionChild(student_id="S-1", chosen_by_parent=True)
        out = resolve_child(reference="named", child_name=FATMA, roster=self.roster, pin=pin)
        self.assertTrue(out.resolved)
        self.assertEqual(out.student_id, "S-1")
        self.assertEqual(out.source, "pin")

    def test_a_chosen_child_outside_the_matches_does_not_rescue_the_turn(self):
        pin = SessionChild(student_id="S-3", chosen_by_parent=True)
        roster = _as_options([DAUGHTER, NAMESAKE, BROTHER])
        out = resolve_child(reference="named", child_name=FATMA, roster=roster, pin=pin)
        self.assertTrue(out.ask)

    def test_a_unique_name_still_beats_any_pin(self):
        pin = SessionChild(student_id="S-1", chosen_by_parent=True)
        roster = _as_options([DAUGHTER, NAMESAKE, BROTHER])
        out = resolve_child(reference="named", child_name="عمر", roster=roster, pin=pin)
        self.assertEqual(out.student_id, "S-3")


class TheChoiceIsRememberedAsAChoice(unittest.TestCase):
    """`SessionChild.chosen_by_parent`, and its journey through metadata."""

    def test_a_pin_written_by_a_tool_is_not_a_choice(self):
        child = SessionChild()
        child.pin(student_id="S-1", label=FATMA)
        self.assertFalse(child.chosen_by_parent)

    def test_a_choice_is_kept_when_the_same_child_is_pinned_again_by_a_tool(self):
        child = SessionChild()
        child.pin(student_id="S-1", chosen_by_parent=True)
        child.pin(student_id="S-1", label=FATMA)
        self.assertTrue(child.chosen_by_parent)

    def test_a_different_child_resets_it(self):
        child = SessionChild()
        child.pin(student_id="S-1", chosen_by_parent=True)
        child.pin(student_id="S-3")
        self.assertFalse(child.chosen_by_parent)

    def test_it_round_trips_through_metadata(self):
        child = SessionChild(guardian_id=GUARDIAN)
        child.pin(student_id="S-1", chosen_by_parent=True)
        loaded = SessionChild.from_metadata({"child_context": child.to_metadata()}, guardian_id=GUARDIAN)
        self.assertTrue(loaded.chosen_by_parent)

    def test_a_pin_stored_before_the_field_existed_still_loads(self):
        stored = {"student_id": "S-1", "label": FATMA, "gender": "female", "guardian_id": GUARDIAN}
        loaded = SessionChild.from_metadata({"child_context": stored}, guardian_id=GUARDIAN)
        self.assertEqual(loaded.student_id, "S-1")
        self.assertFalse(loaded.chosen_by_parent)

    def test_the_context_passes_the_flag_through(self):
        ctx = ChatRequestContext(user_id=USER, session_id=SESSION)
        try:
            ctx.remember_child("S-1", label=FATMA, chosen_by_parent=True)
            self.assertTrue(ctx.child.chosen_by_parent)
        finally:
            ctx.close()


class TappingAnOfferedOptionPinsThatChild(unittest.TestCase):
    """`_pin_the_child_the_parent_named`, where two offered names contain each other."""

    def setUp(self):
        env = patch.dict(os.environ, {"CHILD_ROSTER_TTL_SECONDS": "0"})
        env.start()
        self.addCleanup(env.stop)
        self.ctx = ChatRequestContext(
            user_id=USER, session_id=SESSION,
            caller=CallerIdentity(user_id=USER, guardian_id=GUARDIAN, guardian_token=TOKEN),
        )
        self.addCleanup(self.ctx.close)

    def _pin(self, chosen, rows=(DAUGHTER, NAMESAKE)):
        with patch("backend.chat.child_roster.requests.get", _roster(list(rows))):
            return service._pin_the_child_the_parent_named(self.ctx, chosen)

    def test_the_tapped_option_pins_exactly_that_child(self):
        self.assertTrue(self._pin(f"{FATMA} — Year 9"))
        self.assertEqual(self.ctx.child.student_id, "S-2")
        self.assertEqual(self.ctx.child.label, f"{FATMA} — Year 9")

    def test_the_pin_records_that_the_parent_chose(self):
        self._pin(f"{FATMA} — Year 11")
        self.assertTrue(self.ctx.child.chosen_by_parent)

    def test_the_bare_shared_name_still_pins_nobody(self):
        """Typed rather than tapped, and matching both: one more question is right."""
        self.assertFalse(self._pin(FATMA))
        self.assertFalse(self.ctx.child.is_set)

    def test_a_typed_partial_name_still_resolves_where_it_is_unique(self):
        self.assertTrue(self._pin("عمر", rows=[DAUGHTER, NAMESAKE, BROTHER]))
        self.assertEqual(self.ctx.child.student_id, "S-3")

    def test_the_same_child_served_twice_needs_no_choice_at_all(self):
        options = _as_options([DAUGHTER, dict(DAUGHTER)])
        out = resolve_child(reference="named", child_name=FATMA, roster=options)
        self.assertTrue(out.resolved)
        self.assertEqual(out.student_id, "S-1")


# ---------------------------------------------------------------------------
# G. The parent's spelling of Monday
# ---------------------------------------------------------------------------


class TheDayVocabularyKnowsTheParentsSpelling(unittest.TestCase):
    def test_the_message_from_production_names_monday(self):
        self.assertEqual(day_phrase("الاتنيني ابعتلي الحصص بالترتيب وكل حاجه"), "monday")

    def test_the_forms_already_known_still_work(self):
        self.assertEqual(day_phrase("طيب هي يوم الحد عندها حصص اي"), "sunday")
        self.assertEqual(day_phrase("الاتنين ابعتلي الحصص"), "monday")

    def test_no_tolerance_was_introduced(self):
        """A trailing-letter rule would read the eleventh as Sunday."""
        self.assertEqual(day_phrase("الحادي عشر"), "")
        self.assertEqual(day_phrase("عندها اربع حصص"), "")


if __name__ == "__main__":
    unittest.main()
