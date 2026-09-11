# -*- coding: utf-8 -*-
"""A parent's real conversation, replayed end to end, message by message.

On 2026-09-11 a father asked the school assistant fifteen questions about his daughter.
The first eight were answered well. At the ninth he was asked which child he meant and
offered two identical names; he tapped one, got his answer — and from then on every
message he sent was answered with that same subjects list. Sunday's lessons, Monday's
lessons, when the second term starts: the subjects list, every time, with the tool's own
evidence header pasted into it, a line of the model's private reasoning in front of it,
and a placeholder where a table was meant to go. «Thanks» finally got the answer to the
question before it.

Every one of those defects lived in a DIFFERENT component, and every component's own
tests passed. The clarification was saved correctly, read back correctly, re-planned
correctly — and never cleared, which is the one step no test took. So this file does not
test a component. It drives the real `chat_with_agent_stream` through the whole
conversation, in order, over one stored session, and asserts on what the parent saw.

What is real: `_enter_turn` and the pending-question lifecycle, the planner (`plan_turn`,
the ladder, child resolution against the roster, tool narrowing, planned calls and the
`$day` argument), every records tool and the knowledge tool with their templates and
outcome reporting, the finalizer, the evidence cut, answer blocks, and the save.

What is scripted, and only this: the MODEL's words — the envelope classifier's verdict
and the answers, reproducing what the live model actually produced — and the network,
which serves one school's facade payloads. Retrieval is stubbed at `run_rag_graph`, so the
knowledge tool itself still runs and reports its own outcome.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import types
import unittest
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessageChunk, ToolMessage

import backend.chat.runtime as runtime
from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_roster import _as_options
from backend.chat.orchestrator import plan_turn as _real_plan_turn
from backend.chat.resolution import FOLLOWUP, ResolvedQuestion, unresolved
from backend.chat.runtime import planned_tool_calls
from backend.profiles.registry import load_profile, set_profile
from backend.tools import build_tools
from tests.general.test_chat_hitl_resume import FakeStorage

service = importlib.import_module("backend.chat.service")

GUARDIAN = "G-replay"
TOKEN = "signed.identity.token"
USER = "parent-replay"
SESSION = "session-replay"

FATMA = "فاطمه محمد ابوالحسن"
CLASS_AR = "الصف الثاني الثانوي علمي بنين 1"
TERM_AR = "الفصل الدراسي الثاني"

# --- the family ----------------------------------------------------------------------
#
# Two rows under ONE Arabic name — production offered «فاطمه محمد ابوالحسن» twice. The
# sexes differ, which is what let every turn before the ninth settle on her without
# asking: «بنتي» leaves one candidate. The ninth named her instead, and a name matches
# both rows.

DAUGHTER_ROW = {
    "student_id": "S-1",
    "full_name_ar": FATMA,
    "full_name_en": "Fatma Mohamed Abolhassan",
    "gender": "female",
    "year_level": "Year 11",
}
NAMESAKE_ROW = {
    "student_id": "S-2",
    "full_name_ar": FATMA,
    "full_name_en": "",
    "gender": "male",
    "year_level": "Year 9",
}
PRODUCTION_ROSTER = [DAUGHTER_ROW, NAMESAKE_ROW]
#: The other way production could have come to offer one name twice: the same child
#: served twice. Nothing distinguishes the rows, so the only correct question is none.
SAME_CHILD_TWICE = [DAUGHTER_ROW, dict(DAUGHTER_ROW)]

# --- the school's records, as the facade serves them ------------------------------------

_SUBJECTS = [
    ("ARAB", "اللغة العربية", "Arabic"),
    ("ENG", "اللغة الإنجليزية", "English"),
    ("BIO", "الأحياء", "Biology"),
    ("L2", "اللغة الثانية", "Second Language"),
    ("PHY", "الفيزياء", "Physics"),
    ("CHEM", "الكيمياء", "Chemistry"),
    ("MATH1", "رياضة 1", "Mathematics 1"),
    ("MATH2", "رياضة 2", "Mathematics 2"),
]
_BY_NAME = {ar: (code, ar, en) for code, ar, en in _SUBJECTS}

GRADES = {
    "term": {"term_id": "2026-T2", "name_ar": TERM_AR},
    "courses": [
        {
            "course_id": str(9000 + i),
            "subject_name_ar": ar,
            "subject_name_en": en,
            "computed_percentage": pct,
            "letter_grade": letter,
            "excused_count": 0,
            "missing_count": 0,
            "is_complete": True,
        }
        for i, ((code, ar, en), pct, letter) in enumerate(
            zip(
                _SUBJECTS,
                [84.0, 78.0, 89.0, 57.0, 62.0, 80.0, 97.0, 65.0],
                ["B", "C", "B", "F", "D", "B", "A", "D"],
            )
        )
    ],
}

#: The week as production drew it — Sunday to Thursday, a break numbered as period 5.
_WEEK_GRID = {
    "sunday": ["الكيمياء", "اللغة الإنجليزية", "اللغة العربية", "الفيزياء", "رياضة 1"],
    "monday": ["الأحياء", "الكيمياء", "اللغة الإنجليزية", "اللغة العربية", "الفيزياء"],
    "tuesday": ["رياضة 1", "الأحياء", "الكيمياء", "اللغة الإنجليزية", "اللغة العربية"],
    "wednesday": ["اللغة العربية", "رياضة 1", "الأحياء", "الكيمياء", "اللغة الإنجليزية"],
    "thursday": ["اللغة الإنجليزية", "اللغة العربية", "رياضة 1", "الأحياء", "الكيمياء"],
}
_TEACHING_PERIODS = [1, 2, 3, 4, 6]


def _period(number, name_ar, name_en, starts, ends, teaching=True):
    return {
        "period_number": number,
        "name_ar": name_ar,
        "name_en": name_en,
        "starts_at": starts,
        "ends_at": ends,
        "is_teaching": teaching,
    }


WEEK = {
    "student": {"student_id": "S-1"},
    "term": {"term_id": "2026-T2", "name_ar": TERM_AR},
    "status": "ok",
    "class_code": "11S1",
    "class_name_ar": CLASS_AR,
    "class_name_en": "Year 11 Science Boys 1",
    "days": list(_WEEK_GRID),
    "periods": [
        _period(1, "حصة 1", "Period 1", "07:45", "08:30"),
        _period(2, "حصة 2", "Period 2", "08:30", "09:15"),
        _period(3, "حصة 3", "Period 3", "09:15", "10:00"),
        _period(4, "حصة 4", "Period 4", "10:00", "10:45"),
        _period(5, "الفسحة", "Break", "10:45", "11:15", teaching=False),
        _period(6, "حصة 6", "Period 6", "11:15", "12:00"),
    ],
    "lessons": [
        {
            "day_of_week": day,
            "period_number": period,
            "subject_code": _BY_NAME[subject][0],
            "subject_name_ar": subject,
            "subject_name_en": _BY_NAME[subject][2],
        }
        for day, subjects in _WEEK_GRID.items()
        for period, subject in zip(_TEACHING_PERIODS, subjects)
    ],
}

ROOM = {
    "student": {"student_id": "S-1"},
    "term": {"term_id": "2026-T2"},
    "status": "ok",
    "class_code": "11S1",
    "class_name_ar": CLASS_AR,
    "class_name_en": "Year 11 Science Boys 1",
    "year_level_name_ar": "الصف الثاني الثانوي",
    "subjects": [{"code": code, "name_ar": ar, "name_en": en} for code, ar, en in _SUBJECTS],
    "teachers": [],
}

# --- the school's published material, as retrieval would hand it over ---------------------

UNIFORM_CHUNK = {
    "filename": "BHCR Knowledge Base-Jan 2026 (4).docx",
    "page_number": "12",
    "text": "Sports Wear: All Grades - Unisex. زي رياضي موحد باللونين الأزرق الداكن والذهبي.",
}
FEES_CHUNK = {
    "filename": "fees_2026_27.pdf",
    "page_number": "1",
    "text": "رسوم العام الدراسي 26/27 بالريال: Pre-K 34,000 ... Y11-Y12 72,000 (غير شاملة VAT للطلاب السعوديين).",
}
CALENDAR_CHUNK = {
    "filename": "calendar_2025_26.pdf",
    "page_number": "2",
    "text": "يبدأ الفصل الدراسي الثاني يوم الأحد 4 يناير 2026.",
}


def _retrieval(query: str, ctx) -> dict:
    """What `run_rag_graph` returns for each subject this conversation asks about."""
    text = query or ""
    if "الالعاب" in text or "الألعاب" in text:
        chunks = [UNIFORM_CHUNK]
    elif "مصاريف" in text:
        chunks = [FEES_CHUNK]
    elif "الترم" in text or "الفصل" in text:
        chunks = [CALENDAR_CHUNK]
    else:
        # The school uniform itself was not in the corpus (the fourth message).
        return {"docs": [], "rag_trace": {"retrieval_status": "no_knowledge"}}
    return {
        "docs": chunks,
        "rag_trace": {"retrieval_status": "answerable", "retrieved_chunks": chunks},
    }


# --- the network -----------------------------------------------------------------------


class _Response:
    def __init__(self, status_code: int, payload: Optional[dict] = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class _Facade:
    """The records facade, by route. Records which child every record read named."""

    def __init__(self, roster: list):
        self.roster = roster
        self.reads: List[tuple] = []

    def __call__(self, url, headers=None, params=None, timeout=None):
        if url.endswith("/students"):
            return _Response(200, {"guardian_id": GUARDIAN, "students": list(self.roster)})
        for suffix, payload in (
            ("/grades", GRADES),
            ("/timetable", WEEK),
            ("/class", ROOM),
            ("/subjects", ROOM),
            ("/teachers", ROOM),
        ):
            if url.endswith(suffix):
                student = url.rstrip("/").split("/")[-2]
                self.reads.append((suffix.strip("/"), student))
                return _Response(200, payload)
        return _Response(404)


# --- the conversation ----------------------------------------------------------------------


def _envelope(*, about_child=True, reference="context", name="", kind="records", tools=()):
    return {
        "scope": "in_domain",
        "about_child": about_child,
        "child_reference": reference,
        "child_name": name,
        "child_question_kind": kind,
        "needed_tools": list(tools),
    }


def _first_line_starting(texts: List[str], header: str) -> str:
    for text in texts:
        for line in (text or "").splitlines():
            if line.strip().startswith(header):
                return line.strip()
    return ""


@dataclass
class Turn:
    """One message, and what the live model said when it was the question being answered.

    `envelopes` are the classifier's verdicts, one per time this question is planned (the
    last repeats) — the live classifier does not give the same answer twice for a message
    whose history has changed. `answer` receives what the tools returned, so a turn that
    is answered as the WRONG question — the production failure — produces exactly the
    wrong answer production did.
    """

    says: str
    envelopes: List[dict]
    answer: Callable[[List[str]], str] = lambda _texts: ""
    #: The standalone question the RESOLVER made of this message, when it made one. Most
    #: messages here need none; the ninth does, because that is how production came to
    #: see a NAMED reference in a message that names nobody: the resolver read «هي» as
    #: her full name, and the classifier was shown the resolved text.
    resolved: str = ""


def _turns(*, which_child_replan_names_her: bool = False) -> List[Turn]:
    named = _envelope(reference="named", name=FATMA, tools=["get_student_subjects"])
    # What production evidently got on the re-plan: it answered, which a NAMED reference
    # over two rows sharing the name could not have done.
    as_she = _envelope(reference="daughter", tools=["get_student_subjects"])
    return [
        Turn(
            "كنت عايز اعرف جدول بنتي",
            [_envelope(reference="daughter", tools=["get_student_timetable"])],
            lambda _t: f"ده جدول {FATMA} في {CLASS_AR} لل{TERM_AR}:",
        ),
        Turn(
            "طيب هي جابت كام في العربي",
            [_envelope(reference="daughter", tools=["get_student_grades"])],
            lambda _t: f"حضرتك، {FATMA} حصلت على 84.0% في اللغة العربية.",
        ),
        Turn(
            "طب وباقي المواد",
            [_envelope(tools=["get_student_grades"])],
            lambda _t: f"حضرتك، دي درجات {FATMA} في {TERM_AR}:",
        ),
        Turn(
            "طيب لبس المدرسه اي",
            [_envelope(about_child=False, kind="school_matter", tools=["search_knowledge_base"])],
        ),
        Turn(
            "طب لبس الالعاب اي",
            [_envelope(about_child=False, kind="school_matter", tools=["search_knowledge_base"])],
            lambda _t: (
                "لبس الألعاب في المدرسة زي رياضي موحد للكل، باللونين الأزرق الداكن والذهبي."
                "\n\nSports Wear: All Grades - Unisex [1]"
            ),
        ),
        Turn(
            "طب كام مصاريف المدرسه",
            [_envelope(about_child=False, kind="school_matter", tools=["search_knowledge_base"])],
            lambda _t: "حضرتك، رسوم المدرسة للعام الدراسي 26/27 بالريال:\n\n- Pre-K: 34,000\n- Y11-Y12: 72,000 [1]",
        ),
        Turn(
            "طب مصاريف بنتي كام كده",
            [_envelope(reference="daughter", kind="school_matter", tools=["search_knowledge_base"])],
            lambda _t: "حضرتك، مصاريف بنتك في Y11-Y12 هي 72,000 ريال للمواطنين السعوديين. [1]",
        ),
        Turn(
            "طب هي في فصل كام",
            [_envelope(reference="daughter", tools=["get_student_class"])],
            lambda _t: f"{FATMA} في {CLASS_AR}.",
        ),
        # The ninth: the resolver read «she» as her full name, so the classifier saw a
        # NAMED reference — and a name matches both rows.
        Turn(
            "طيب بتاخد مواد اي\\",
            [named, named if which_child_replan_names_her else as_she],
            # Verbatim shape of what production showed: a framing sentence, then the
            # tool's own model-facing line pasted underneath it, citation and all.
            lambda texts: (
                f"حضرتك، {FATMA} في {CLASS_AR} بتاخد المواد التالية:\n"
                + _first_line_starting(texts, "SUBJECTS")
                + " [1]"
            ),
            resolved=f"إيه المواد اللي بتاخدها {FATMA}؟",
        ),
        # The tenth is the parent tapping the child — see `_Replay._reply_to_which_child`.
        Turn(
            "طيب هي يوم الحد عندها حصص اي",
            [_envelope(reference="daughter", tools=["get_student_timetable"])],
            # The live model's habit when told the table is appended: a slot for it.
            lambda _t: f"دي حصص {FATMA} يوم الأحد بالترتيب:\n[جدول الحصص]",
        ),
        Turn(
            "طب ممكن تبعتلي النتيجه بتاعت المدرسه",
            [_envelope(reference="daughter", tools=["get_student_grades"])],
            # Unmarked reasoning run straight into the answer — measured on the live model.
            lambda _t: (
                "We need to get the timetable.We need to see the result."
                f"حضرتك، دي نتيجة {FATMA} في {TERM_AR}:"
            ),
        ),
        Turn(
            "الاتنيني ابعتلي الحصص بالترتيب وكل حاجه",
            [_envelope(reference="daughter", tools=["get_student_timetable"])],
            lambda _t: f"دي حصص {FATMA} يوم الاتنين بالترتيب:",
        ),
        Turn(
            "طيب الترم التاني هيبداء امتي",
            [_envelope(about_child=False, kind="school_matter", tools=["search_knowledge_base"])],
            lambda _t: "حضرتك، الفصل الدراسي الثاني بيبدأ يوم الأحد 4 يناير 2026. [1]",
        ),
        Turn("Thanks", [_envelope(about_child=False, kind="school_matter")]),
    ]


TURNS = _turns()
#: Zero-based index, in TURNS, of the message the parent was asked "which child?" on.
WHICH_CHILD_TURN = 8


@dataclass
class Observed:
    """One message as the system handled it, and as the parent experienced it."""

    says: str
    planned_as: List[str] = field(default_factory=list)
    plan: object = None
    calls: List[dict] = field(default_factory=list)
    agent_ran: bool = False
    events: List[dict] = field(default_factory=list)
    shown: str = ""
    stored: str = ""
    pending: Optional[dict] = None
    pin: Dict = field(default_factory=dict)

    def of_type(self, kind: str) -> List[dict]:
        return [event for event in self.events if event.get("type") == kind]

    @property
    def options(self) -> List[str]:
        asked = self.of_type("hitl_request")
        return list(asked[-1]["hitl"]["options"]) if asked else []

    @property
    def blocks(self) -> List[dict]:
        found = self.of_type("answer_blocks")
        return list(found[-1]["answer_blocks"]) if found else []

    def called(self, name: str) -> List[dict]:
        return [call for call in self.calls if call["name"] == name]


class _ScriptedAgent:
    """The graph, minus the model: planned calls dispatched through the REAL tools, then
    the scripted answer streamed through the real finalizer."""

    def __init__(self, replay: "_Replay", ctx):
        self.replay = replay
        self.ctx = ctx

    async def astream(self, payload, stream_mode=None, config=None):
        observed = self.replay.current
        observed.agent_ran = True
        asked = observed.planned_as[-1] if observed.planned_as else observed.says
        turn = self.replay.by_text.get(asked)
        # Exactly what `_dispatch_planned_tools` hands the graph's tool node.
        calls = planned_tool_calls(list(getattr(self.ctx, "planned_calls", None) or []), {})
        texts = []
        for call in calls:
            tool = build_tools([call["name"]], self.ctx)[0]
            text = tool.invoke(call["args"])
            observed.calls.append({"name": call["name"], "args": dict(call["args"])})
            texts.append(text)
            yield ToolMessage(content=text, tool_call_id=call["id"]), {}

        stored = self.ctx.peek_rag_trace() or {}
        status = (stored.get("rag_trace") or {}).get("retrieval_status")
        if status in runtime.TERMINAL_STATUSES:
            # What `_end_turn_on_terminal_retrieval` does: end on the verdict, no answer.
            self.ctx.note_short_circuit(status)
            return

        answer = turn.answer(texts) if turn else ""
        for start in range(0, len(answer), 7):
            yield AIMessageChunk(content=answer[start:start + 7], id="answer"), {}


class _Replay:
    """The whole conversation over one stored session.

    `which_at` is the index in `turns` of the message that is answered with "which child?"
    — the parent's tap is sent straight after it, as its own message.
    """

    def __init__(self, roster: list, turns: List[Turn], *, which_at: Optional[int] = WHICH_CHILD_TURN):
        self.roster = roster
        self.turns = turns
        self.which_at = which_at
        self.by_text: Dict[str, Turn] = {turn.says: turn for turn in turns}
        self.storage = FakeStorage([])
        self.facade = _Facade(roster)
        self.observed: List[Observed] = []
        self.current: Optional[Observed] = None
        self.profile = load_profile("school")
        self._classified: Counter = Counter()

    def _plan(self, question, history=None, ctx=None, **kwargs):
        self.current.planned_as.append(question)
        turn = self.by_text.get(question)
        if turn is None:
            envelope = _envelope(about_child=False)
        else:
            seen = self._classified[question]
            self._classified[question] += 1
            envelope = dict(turn.envelopes[min(seen, len(turn.envelopes) - 1)])
        if kwargs.get("resolution") is None:
            kwargs["resolution"] = (
                ResolvedQuestion(
                    question=turn.resolved, intent=FOLLOWUP, resolved=True, reason="replay"
                )
                if turn is not None and turn.resolved
                else unresolved(question, "replay")
            )
        plan, signals = _real_plan_turn(
            question, history, ctx, envelope_invoke=lambda *_: dict(envelope), **kwargs
        )
        self.current.plan = plan
        return plan, signals

    def _agent(self, ctx, tool_names=None, language=None):
        return _ScriptedAgent(self, ctx)

    def _reply_to_which_child(self) -> str:
        """What the father tapped: the option naming his daughter, however it is spelled."""
        asked = self.observed[-1].options if self.observed else []
        daughter = next(
            (child.label for child in _as_options(self.roster) if child.student_id == "S-1"),
            FATMA,
        )
        if daughter in asked:
            return daughter
        return asked[0] if asked else FATMA

    async def _one(self, says: str) -> Observed:
        self.current = Observed(says=says)
        caller = CallerIdentity(user_id=USER, guardian_id=GUARDIAN, guardian_token=TOKEN)
        chunks = []
        async for chunk in service.chat_with_agent_stream(says, USER, SESSION, caller=caller):
            chunks.append(chunk)
        observed = self.current
        for chunk in chunks:
            body = chunk.strip()
            if not body.startswith("data: "):
                continue
            data = body[len("data: "):]
            observed.events.append({"type": "DONE"} if data == "[DONE]" else json.loads(data))
        shown = ""
        for event in observed.events:
            if event.get("type") == "content":
                shown += event.get("content", "")
            elif event.get("type") == "content_replace":
                shown = event.get("content", "")
        observed.shown = shown
        last = self.storage.messages[-1] if self.storage.messages else None
        observed.stored = getattr(last, "content", "") if last is not None else ""
        observed.pending = self.storage.metadata.get("pending_hitl")
        observed.pin = dict(self.storage.metadata.get("child_context") or {})
        return observed

    async def _run(self) -> None:
        for index, turn in enumerate(self.turns):
            self.observed.append(await self._one(turn.says))
            if index == self.which_at:
                self.observed.append(await self._one(self._reply_to_which_child()))

    def run(self) -> "_Replay":
        fake_rag = types.ModuleType("backend.rag.pipeline")
        fake_rag.run_rag_graph = _retrieval
        with (
            patch.dict(os.environ, {"CHILD_ROSTER_TTL_SECONDS": "0"}),
            patch.dict(sys.modules, {"backend.rag.pipeline": fake_rag}),
            patch("requests.get", self.facade),
            patch.object(service, "_PROFILE", self.profile),
            patch.object(service, "_COPY", self.profile.user_copy),
            patch.object(service, "storage", self.storage),
            patch.object(service, "plan_turn", self._plan),
            patch.object(service, "create_agent_for_request", self._agent),
            patch.object(service, "generate_session_title", Mock(return_value="جدول بنتي")),
            patch.object(service, "update_persistent_note", AsyncMock(return_value="")),
        ):
            set_profile(self.profile)
            try:
                asyncio.run(self._run())
            finally:
                set_profile(None)
        return self


_CACHE: Dict[str, _Replay] = {}


def _replayed(key: str, build: Callable[[], _Replay]) -> _Replay:
    """Each replay runs once per test process and is shared by its class's assertions."""
    if key not in _CACHE:
        _CACHE[key] = build().run()
    return _CACHE[key]


# Positions in a full replay. The parent's tap is its own message straight after the
# ninth, so every message from «يوم الحد» onwards sits one place after its index in TURNS.
T_WEEK, T_ARABIC, T_ALL_MARKS, T_UNIFORM, T_SPORTS, T_FEES, T_HER_FEES, T_CLASS = range(8)
T_WHICH, T_TAP, T_SUNDAY, T_RESULT, T_MONDAY, T_TERM, T_THANKS = range(8, 15)


class TheConversationFromProduction(unittest.TestCase):
    """Fifteen messages, as the father sent them, through the real service."""

    @classmethod
    def setUpClass(cls):
        cls.replay = _replayed("production", lambda: _Replay(PRODUCTION_ROSTER, TURNS))
        cls.turns = cls.replay.observed

    # -- the conversation as a whole ------------------------------------------------------

    def test_every_message_was_answered(self):
        self.assertEqual(len(self.turns), 15)
        for turn in self.turns:
            self.assertTrue(turn.shown.strip(), f"nothing shown for {turn.says!r}")

    def test_every_later_message_is_planned_as_the_parent_typed_it(self):
        """The production failure itself: from the tenth message on, the planner was
        handed «طيب بتاخد مواد اي» every time, whatever the father had typed."""
        for position in (T_SUNDAY, T_RESULT, T_MONDAY, T_TERM, T_THANKS):
            turn = self.turns[position]
            self.assertEqual(turn.planned_as, [turn.says], f"message {position + 1}")

    def test_the_which_child_question_is_spent_once_answered(self):
        for turn in self.turns[T_TAP:]:
            self.assertIsNone(turn.pending, f"still pending after {turn.says!r}")

    def test_what_is_stored_is_what_the_parent_saw(self):
        for turn in self.turns:
            self.assertEqual(turn.stored.strip(), turn.shown.strip(), turn.says)

    def test_the_daughter_stays_settled_to_the_end(self):
        for turn in self.turns[T_TAP:]:
            self.assertEqual(turn.pin.get("student_id"), "S-1", turn.says)

    def test_every_record_read_was_about_the_daughter(self):
        self.assertTrue(self.replay.facade.reads)
        for route, student in self.replay.facade.reads:
            self.assertEqual(student, "S-1", route)

    # -- the ninth and tenth: asking which child -----------------------------------------

    def test_asking_which_child_offers_choices_a_parent_can_tell_apart(self):
        options = self.turns[T_WHICH].options
        self.assertGreaterEqual(len(options), 2)
        self.assertEqual(len(options), len(set(options)), options)

    def test_choosing_her_answers_the_question_that_was_asked(self):
        tap = self.turns[T_TAP]
        self.assertEqual(tap.planned_as, [TURNS[WHICH_CHILD_TURN].says])
        self.assertTrue(tap.called("get_student_subjects"))
        self.assertIn("بتاخد المواد التالية", tap.shown)
        self.assertEqual(tap.of_type("hitl_request"), [])
        self.assertEqual(tap.pin.get("student_id"), "S-1")

    # -- what the model must never put in front of a parent --------------------------------

    def test_no_tool_evidence_header_reaches_the_parent(self):
        for turn in self.turns:
            self.assertIsNone(service._EVIDENCE_MARKERS.search(turn.shown), turn.says)
            self.assertNotIn("SUBJECTS for", turn.shown, turn.says)

    def test_the_sentence_around_the_evidence_survives_the_cut(self):
        tap = self.turns[T_TAP]
        self.assertIn("بتاخد المواد التالية", tap.shown)

    @unittest.expectedFailure
    def test_no_private_reasoning_reaches_the_parent(self):
        """KNOWN OPEN. Unmarked reasoning glued to the answer ("We need to see the
        result.حضرتك…") carries no Harmony token, so no text rule in `model_output.py`
        can find the seam, and it arrived on an ANSWERING message, so the finalizer's
        tool-call rule did not apply either. Deliberately left failing rather than
        deleted: the day this passes, remove the decorator."""
        shown = self.turns[T_RESULT].shown
        self.assertNotIn("We need", shown)
        self.assertTrue(shown.lstrip().startswith("حضرتك"), shown[:60])

    @unittest.expectedFailure
    def test_no_placeholder_for_the_table_reaches_the_parent(self):
        """KNOWN OPEN. Told the table is appended underneath, the live model writes a slot
        for it — «[جدول الحصص]» — and nothing in the prompt forbids that or removes it.
        Left failing on purpose; remove the decorator once it is handled."""
        for turn in self.turns:
            self.assertNotIn("[جدول الحصص]", turn.shown, turn.says)

    # -- the questions that were swallowed -------------------------------------------------

    def test_sunday_is_answered_with_sundays_lessons(self):
        sunday = self.turns[T_SUNDAY]
        self.assertEqual(
            [call["args"].get("day") for call in sunday.called("get_student_timetable")],
            ["sunday"],
        )
        self.assertFalse(sunday.called("get_student_subjects"))
        drawn = [block for block in sunday.blocks if block["kind"] == "timetable"]
        self.assertEqual(len(drawn), 1)
        self.assertEqual([day["day"] for day in drawn[0]["data"]["days"]], ["sunday"])

    def test_the_parents_spelling_of_monday_is_understood(self):
        monday = self.turns[T_MONDAY]
        self.assertEqual(
            [call["args"].get("day") for call in monday.called("get_student_timetable")],
            ["monday"],
        )
        drawn = [block for block in monday.blocks if block["kind"] == "timetable"]
        self.assertEqual(len(drawn), 1)
        self.assertEqual([day["day"] for day in drawn[0]["data"]["days"]], ["monday"])

    def test_the_result_question_reads_her_marks(self):
        result = self.turns[T_RESULT]
        self.assertTrue(result.called("get_student_grades"))
        self.assertFalse(result.called("get_student_subjects"))

    def test_the_term_question_goes_to_the_school_material(self):
        term = self.turns[T_TERM]
        self.assertTrue(term.called("search_knowledge_base"))
        self.assertFalse([call for call in term.calls if call["name"].startswith("get_student")])
        self.assertIn("يناير", term.shown)

    def test_thanks_is_answered_as_thanks(self):
        thanks = self.turns[T_THANKS]
        self.assertFalse(thanks.agent_ran)
        self.assertEqual(thanks.calls, [])
        self.assertNotIn("الترم", thanks.shown)
        self.assertNotIn("يناير", thanks.shown)

    # -- the eight that already worked must keep working -----------------------------------

    def test_the_first_eight_messages_are_unchanged(self):
        week, arabic, marks = self.turns[T_WEEK], self.turns[T_ARABIC], self.turns[T_ALL_MARKS]
        self.assertEqual(
            [day["day"] for day in week.blocks[0]["data"]["days"]],
            ["sunday", "monday", "tuesday", "wednesday", "thursday"],
        )
        self.assertIn("84.0%", arabic.shown)
        self.assertEqual(len(marks.blocks[0]["data"]["courses"]), 8)
        self.assertTrue(self.turns[T_SPORTS].called("search_knowledge_base"))
        self.assertIn("72,000", self.turns[T_HER_FEES].shown)
        self.assertIn(CLASS_AR, self.turns[T_CLASS].shown)
        for turn in self.turns[:T_WHICH]:
            self.assertEqual(turn.of_type("hitl_request"), [], turn.says)


class TheReplanStillNamesHer(unittest.TestCase):
    """The classifier is a model and may name her again when the original question is
    planned after the parent chose. Her choice has to settle it anyway: a named reference
    that matches two rows must not ask a question the parent has just answered."""

    @classmethod
    def setUpClass(cls):
        turns = _turns(which_child_replan_names_her=True)
        cls.replay = _replayed(
            "replan-names-her",
            lambda: _Replay(
                PRODUCTION_ROSTER,
                [turns[WHICH_CHILD_TURN], turns[WHICH_CHILD_TURN + 1]],
                which_at=0,
            ),
        )
        cls.turns = cls.replay.observed

    def test_the_choice_answers_rather_than_asking_again(self):
        tap = self.turns[1]
        self.assertEqual(tap.of_type("hitl_request"), [])
        self.assertTrue(tap.called("get_student_subjects"))
        self.assertIsNone(tap.pending)

    def test_the_chosen_child_is_the_one_read(self):
        self.assertEqual({student for _, student in self.replay.facade.reads}, {"S-1"})

    def test_the_following_message_is_its_own_question(self):
        sunday = self.turns[2]
        self.assertEqual(sunday.planned_as, [sunday.says])
        self.assertEqual(
            [call["args"].get("day") for call in sunday.called("get_student_timetable")],
            ["sunday"],
        )


class TheSameChildServedTwice(unittest.TestCase):
    """The other shape the duplicate could have had: one child, listed twice. Nothing
    tells the rows apart, so the right number of questions to ask is zero."""

    @classmethod
    def setUpClass(cls):
        cls.replay = _replayed(
            "same-child-twice",
            lambda: _Replay(
                SAME_CHILD_TWICE,
                [TURNS[WHICH_CHILD_TURN], TURNS[WHICH_CHILD_TURN + 1]],
                which_at=None,
            ),
        )
        cls.turns = cls.replay.observed

    def test_naming_her_asks_nothing_and_answers(self):
        named = self.turns[0]
        self.assertEqual(named.of_type("hitl_request"), [])
        self.assertTrue(named.called("get_student_subjects"))
        self.assertIsNone(named.pending)

    def test_the_next_message_is_its_own_question(self):
        sunday = self.turns[1]
        self.assertEqual(sunday.planned_as, [sunday.says])
        self.assertEqual(
            [call["args"].get("day") for call in sunday.called("get_student_timetable")],
            ["sunday"],
        )


if __name__ == "__main__":
    unittest.main()
