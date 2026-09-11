"""A record reaches the reader from the TOOL; the model writes only the sentence around it.

This replaced answer grounding for records rather than improving it. Grounding read the
model's PROSE about a table the model did not author, and every failure came from the
prose: «10:00» before a physics lesson became 10,000 because the multiplier lookahead had
truncated «الفيزياء» to «الف», and a correct timetable was withdrawn from a parent. The
answer is not to check the paraphrase harder — it is to stop paraphrasing the data.

So the tool renders the grid once, that render is appended untouched, and nothing the
parent reads as data passes through the model at all.

Three properties, one class each:

  * THE BLOCK IS THE TOOL'S. Appended verbatim, after every replacement decision, and
    never under a refusal. Evidence the model relayed by mistake is cut first.
  * IT IS NARROWED TO WHAT WAS ASKED. The tool decides what is true, the model's own
    sentence decides what is relevant, and nothing is rewritten either way.
  * IT STAYS OUT OF THE MODEL'S HISTORY. The reader keeps the table; the model gets the
    sentence, because a stored table crowds out the question on the next turn.

And one the rest rely on: IT ALSO TRAVELS AS DATA, so a client can draw the record rather
than print it — see the second half of this file.
"""
import unittest

from tests.general import test_parent_turn_scenarios as scenarios


class TheRecordIsRenderedNotRetyped(unittest.TestCase):
    """Appended by the service, marked, and never mixed with the model's own figures."""

    BLOCK = "**الأحد**\n1) اللغة العربية · 07:45–08:30"

    class _Ctx:
        def __init__(self, blocks):
            self.answer_blocks = list(blocks)

    def test_the_block_goes_under_the_prose(self):
        from backend.chat.service import BLOCK_MARKER, _append_answer_blocks

        out = _append_answer_blocks("جدول بنتك:", self._Ctx([self.BLOCK]))
        self.assertEqual(f"جدول بنتك:\n\n{BLOCK_MARKER}\n{self.BLOCK}", out)

    def test_no_block_leaves_the_answer_untouched(self):
        from backend.chat.service import _append_answer_blocks

        self.assertEqual("أهلاً", _append_answer_blocks("أهلاً", self._Ctx([])))

    def test_a_block_with_no_prose_stands_alone(self):
        from backend.chat.service import BLOCK_MARKER, _append_answer_blocks

        self.assertEqual(
            f"{BLOCK_MARKER}\nTABLE", _append_answer_blocks("   ", self._Ctx(["TABLE"]))
        )

    def test_the_marker_is_invisible_to_a_reader(self):
        """An HTML comment, and only because the frontend drops raw HTML outright
        (`renderer.html = () => ''` in utils/markdown.ts). That is the whole reason a
        marker can live inside a message a parent reads."""
        from backend.chat.service import BLOCK_MARKER

        self.assertTrue(BLOCK_MARKER.startswith("<!--"))
        self.assertTrue(BLOCK_MARKER.endswith("-->"))

    def test_relayed_evidence_is_cut_before_the_block_is_added(self):
        """Told not to reformat the grid, the live model pasted the MODEL-FACING render
        instead — outcome header, raw `07:45:00`, English day keys. First try."""
        from backend.chat.service import _append_answer_blocks

        leaked = (
            "حضرتك، الجدول كالتالي:\n\n"
            "TIMETABLE for فاطمة — 2/1:\n"
            "- sunday: 1) عربي 07:45:00-08:30:00"
        )
        out = _append_answer_blocks(leaked, self._Ctx([self.BLOCK]))
        self.assertNotIn("TIMETABLE for", out)
        self.assertNotIn("sunday", out)
        self.assertTrue(out.startswith("حضرتك، الجدول كالتالي:"))
        self.assertIn("**الأحد**", out)

    def test_every_outcome_header_is_recognised(self):
        """A header this misses is a header a parent can be shown."""
        import io
        import re

        from backend.chat.service import _EVIDENCE_MARKERS

        rendered = io.open(
            "backend/prompts/templates/tools/records_result.j2", encoding="utf-8"
        ).read()
        headers = set(re.findall(r"^([A-Z][A-Z_]{3,})(?: for|:)", rendered, re.MULTILINE))
        self.assertTrue(headers, "no headers found — did the template change shape?")
        for header in sorted(headers):
            with self.subTest(header=header):
                self.assertTrue(_EVIDENCE_MARKERS.search(header + " for x"), header)


class TheBlockIsNarrowedToWhatWasAsked(unittest.TestCase):
    """The tool decides what is TRUE; the model's sentence decides what is RELEVANT.

    A parent asking «هي جابت كام في العربي» was given the Arabic mark in the sentence and
    then every other subject underneath it — answering a question nobody asked and burying
    the one they did.

    Nothing is rewritten: a row either survives or it does not, so every figure the parent
    reads is still the tool's own.
    """

    GRADES = (
        "اللغة العربية: 84.0% (B)\n"
        "اللغة الإنجليزية: 78.0% (C)\n"
        "الأحياء: 89.0% (B)"
    )
    WEEK = (
        "**الأحد**\n1) اللغة العربية · 07:45–08:30\n"
        "**الاثنين**\n1) الأحياء · 07:45–08:30"
    )

    class _Ctx:
        def __init__(self, blocks):
            self.answer_blocks = list(blocks)

    def _shown(self, answer, kind, block):
        from backend.chat.service import BLOCK_MARKER, _append_answer_blocks

        out = _append_answer_blocks(answer, self._Ctx([{"kind": kind, "text": block}]))
        return out.split(BLOCK_MARKER, 1)[1].strip()

    def test_one_named_subject_keeps_only_that_row(self):
        shown = self._shown("حصلت على 84.0% في اللغة العربية.", "grades", self.GRADES)
        self.assertEqual("اللغة العربية: 84.0% (B)", shown)

    def test_naming_no_subject_keeps_the_whole_record(self):
        """"Show me her grades" must still show all of them."""
        self.assertEqual(self.GRADES, self._shown("دي درجاتها:", "grades", self.GRADES))

    def test_one_named_day_keeps_only_that_day(self):
        shown = self._shown("جدولها يوم الأحد:", "timetable", self.WEEK)
        self.assertIn("**الأحد**", shown)
        self.assertNotIn("**الاثنين**", shown)

    def test_naming_no_day_keeps_the_whole_week(self):
        self.assertEqual(self.WEEK, self._shown("دي جدولها:", "timetable", self.WEEK))

    def test_an_unknown_kind_is_left_alone(self):
        """A block a future tool adds is passed through rather than silently emptied."""
        self.assertEqual("ROWS", self._shown("أي جملة", "something_new", "ROWS"))


class TheBlockStaysOutOfTheModelsHistory(unittest.TestCase):
    """The reader keeps the table; the model gets the sentence.

    The regression this fixes, measured in production: a stored timetable answer is 95%
    table (1,240 characters against 52 of prose). `conversation_text` clips each message
    to 600, so the next turn's context was lesson rows and almost no sentence — and
    «ومين بيديها في الفصل» resolved against that wall, was classified as a timetable
    question, and was answered with the timetable again.
    """

    class _Ctx:
        def __init__(self, blocks):
            self.answer_blocks = list(blocks)

    def test_the_stored_answer_keeps_the_block(self):
        from backend.chat.service import _append_answer_blocks

        out = _append_answer_blocks("جدولها:", self._Ctx(["**الأحد**\n1) عربي"]))
        self.assertIn("**الأحد**", out)

    def test_the_model_reads_back_only_the_prose(self):
        from backend.chat.service import _append_answer_blocks, strip_answer_blocks

        out = _append_answer_blocks("جدولها:", self._Ctx(["**الأحد**\n1) عربي"]))
        self.assertEqual("جدولها:", strip_answer_blocks(out))

    def test_an_answer_with_no_block_survives_intact(self):
        from backend.chat.service import strip_answer_blocks

        self.assertEqual("أهلاً بحضرتك", strip_answer_blocks("أهلاً بحضرتك"))

    def test_the_resolver_sees_the_subject_not_the_rows(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from backend.chat.resolution import conversation_text
        from backend.chat.service import _append_answer_blocks

        stored = _append_answer_blocks(
            "حضرتك، جدول فاطمة للفصل الدراسي الثاني:",
            self._Ctx(["**الأحد**\n1) الكيمياء · 07:45–08:30\n2) عربي · 08:30–09:15"]),
        )
        seen = conversation_text(
            [HumanMessage(content="جدول بنتي"), AIMessage(content=stored)]
        )
        self.assertIn("جدول فاطمة للفصل الدراسي الثاني", seen)
        self.assertNotIn("07:45", seen)
        self.assertNotIn("الكيمياء", seen)


# --- the same record, as DATA ------------------------------------------------------------
#
# Each block also leaves the service as a structure a client can draw — a phone shows the
# week one day at a time, a wide screen as a grid — while the text above stays exactly
# what it was, for storage, for the model's history, and for every client that knows
# nothing else. The contract is `AnswerBlock` in backend/schemas/chat.py.

WEEK_TEXT = (
    "**الأحد**\n1) اللغة العربية · 07:45–08:30\n3) — · 08:50–09:35\n"
    "**الاثنين**\n1) الأحياء · 07:45–08:30"
)
WEEK_DATA = {
    "class_label": "الثالث أ",
    "term_label": "الفصل الأول",
    "periods": [
        {"number": 1, "starts_at": "07:45", "ends_at": "08:30"},
        {"number": 2, "label": "فسحة", "starts_at": "08:30", "ends_at": "08:50",
         "is_teaching": False},
        {"number": 3, "starts_at": "08:50", "ends_at": "09:35"},
    ],
    "days": [
        {"day": "sunday", "label": "الأحد", "slots": [
            {"period": 1, "subject": "اللغة العربية"},
            {"period": 3, "subject": "", "is_free": True},
        ]},
        {"day": "monday", "label": "الاثنين", "slots": [{"period": 1, "subject": "الأحياء"}]},
    ],
}
GRADES_TEXT = (
    "اللغة العربية: 84.0% (B)\n"
    "اللغة الإنجليزية: 78.0% (C)\n"
    "الأحياء: — · لسه جاري"
)
GRADES_DATA = {
    "term_label": "الفصل الأول",
    "courses": [
        {"subject": "اللغة العربية", "percentage": 84.0, "letter": "B"},
        {"subject": "اللغة الإنجليزية", "percentage": 78.0, "letter": "C"},
        {"subject": "الأحياء", "percentage": None, "in_progress": True},
    ],
}
TIMETABLE_BLOCK = {"kind": "timetable", "index": 0, "language": "ar", "data": WEEK_DATA}


class _TurnCtx:
    """What `_settle_answer_blocks` reads off the real context, and nothing else."""

    def __init__(self, blocks, language=""):
        self.answer_blocks = list(blocks)
        self.language = language


def _settle(answer, blocks, language="ar"):
    from backend.chat.service import _settle_answer_blocks

    return _settle_answer_blocks(answer, _TurnCtx(blocks, language))


class TheRecordAlsoTravelsAsData(unittest.TestCase):
    """One decision, two copies: the text every client can show, the data one can draw."""

    def test_the_data_rides_beside_an_unchanged_text(self):
        from backend.chat.service import BLOCK_MARKER

        text, blocks = _settle(
            "دي جدولها:", [{"kind": "timetable", "text": WEEK_TEXT, "data": WEEK_DATA}]
        )
        self.assertEqual(f"دي جدولها:\n\n{BLOCK_MARKER}\n{WEEK_TEXT}", text)
        self.assertEqual([("timetable", 0, "ar")],
                         [(b["kind"], b["index"], b["language"]) for b in blocks])
        self.assertEqual(["sunday", "monday"], [d["day"] for d in blocks[0]["data"]["days"]])

    def test_a_break_keeps_its_place_in_the_day(self):
        """The facade is explicit that a client drawing the day must show it."""
        _, blocks = _settle(
            "دي جدولها:", [{"kind": "timetable", "text": WEEK_TEXT, "data": WEEK_DATA}]
        )
        periods = blocks[0]["data"]["periods"]
        self.assertEqual([1, 2, 3], [p["number"] for p in periods])
        self.assertEqual([True, False, True], [p["is_teaching"] for p in periods])

    def test_the_index_names_its_own_marker_after_a_text_only_block(self):
        """Positional matching would draw the week in the grades' place."""
        from backend.chat.service import BLOCK_MARKER

        text, blocks = _settle("درجاتها وجدولها:", [
            {"kind": "grades", "text": GRADES_TEXT, "data": None},
            {"kind": "timetable", "text": WEEK_TEXT, "data": WEEK_DATA},
        ])
        self.assertEqual(2, text.count(BLOCK_MARKER))
        self.assertEqual([("timetable", 1)], [(b["kind"], b["index"]) for b in blocks])

    def test_data_that_breaks_the_contract_costs_the_drawing_and_not_the_record(self):
        text, blocks = _settle(
            "دي جدولها:", [{"kind": "timetable", "text": WEEK_TEXT, "data": {"days": 3}}]
        )
        self.assertIn(WEEK_TEXT, text)
        self.assertEqual([], blocks)

    def test_a_kind_no_client_draws_goes_out_as_text_alone(self):
        text, blocks = _settle(
            "أي جملة", [{"kind": "something_new", "text": "ROWS", "data": {"rows": [1]}}]
        )
        self.assertIn("ROWS", text)
        self.assertEqual([], blocks)

    def test_no_language_is_invented_for_a_turn_that_set_none(self):
        _, blocks = _settle(
            "دي جدولها:", [{"kind": "timetable", "text": WEEK_TEXT, "data": WEEK_DATA}],
            language="",
        )
        self.assertEqual("", blocks[0]["language"])


class BothCopiesAreNarrowedAlike(unittest.TestCase):
    """A phone drawing Sunday alone while the stored text keeps the whole week would be
    two answers to one question. The rules are separate code; these hold them in step."""

    def _both(self, answer, kind, text, data):
        from backend.chat.service import BLOCK_MARKER

        settled, blocks = _settle(answer, [{"kind": kind, "text": text, "data": data}])
        return settled.split(BLOCK_MARKER, 1)[1].strip(), blocks[0]["data"]

    def test_the_same_days_survive_in_both(self):
        for answer in ("دي جدولها:", "جدولها يوم الأحد:", "الأحد والاثنين:", "Her week:", ""):
            with self.subTest(answer=answer):
                shown, data = self._both(answer, "timetable", WEEK_TEXT, WEEK_DATA)
                headings = [ln.strip("*") for ln in shown.splitlines() if ln.startswith("**")]
                self.assertEqual(headings, [day["label"] for day in data["days"]])

    def test_one_named_day_is_all_either_copy_keeps(self):
        shown, data = self._both("جدولها يوم الأحد:", "timetable", WEEK_TEXT, WEEK_DATA)
        self.assertNotIn("**الاثنين**", shown)
        self.assertEqual(["sunday"], [day["day"] for day in data["days"]])

    def test_the_same_subjects_survive_in_both(self):
        for answer in ("دي درجاتها:", "حصلت على 84.0% في اللغة العربية.", "العربية والأحياء"):
            with self.subTest(answer=answer):
                shown, data = self._both(answer, "grades", GRADES_TEXT, GRADES_DATA)
                subjects = [ln.split(":")[0] for ln in shown.splitlines()]
                self.assertEqual(subjects, [c["subject"] for c in data["courses"]])


class TheBlocksSurviveTheTrace(unittest.TestCase):
    """The trace is where a stored message keeps its blocks, so a reload can draw them."""

    def test_a_valid_block_is_kept(self):
        from backend.schemas.chat import normalize_rag_trace

        trace = normalize_rag_trace({"answer_blocks": [TIMETABLE_BLOCK]})
        self.assertEqual(
            ["sunday", "monday"],
            [day["day"] for day in trace["answer_blocks"][0]["data"]["days"]],
        )

    def test_one_bad_block_is_dropped_and_the_trace_around_it_survives(self):
        """This runs on every save and every history load. Raising here would cost a
        turn its save, or a whole conversation its reload, over one table."""
        from backend.schemas.chat import normalize_rag_trace

        bad = {"kind": "timetable", "index": 1, "data": {"days": "not a list"}}
        trace = normalize_rag_trace({"route": "agent", "answer_blocks": [bad, TIMETABLE_BLOCK]})
        self.assertEqual("agent", trace["route"])
        self.assertEqual([0], [block["index"] for block in trace["answer_blocks"]])

    def test_a_trace_left_with_no_valid_block_loses_the_key(self):
        from backend.schemas.chat import normalize_rag_trace

        trace = normalize_rag_trace({"route": "agent", "answer_blocks": [{"kind": "nope"}]})
        self.assertNotIn("answer_blocks", trace)

    def test_a_blank_grade_is_never_stored_as_zero(self):
        from backend.schemas.chat import normalize_rag_trace

        block = {"kind": "grades", "index": 0, "data": GRADES_DATA}
        biology = normalize_rag_trace({"answer_blocks": [block]})["answer_blocks"][0][
            "data"]["courses"][2]
        self.assertIsNone(biology.get("percentage"))
        self.assertTrue(biology["in_progress"])

    def test_a_rejected_block_is_logged_without_the_child_s_record(self):
        """A validation error quotes its input, and the input is a child's marks."""
        from backend.schemas.chat import normalize_answer_blocks

        bad = {"kind": "grades", "index": 0,
               "data": {"courses": [{"subject": "اللغة العربية", "percentage": "ممتاز"}]}}
        with self.assertLogs("backend.schemas.chat", level="WARNING") as logs:
            self.assertEqual([], normalize_answer_blocks([bad]))
        logged = "\n".join(logs.output)
        self.assertNotIn("ممتاز", logged)
        self.assertNotIn("اللغة العربية", logged)

    def test_a_turn_with_no_trace_gets_one_rather_than_losing_its_blocks(self):
        from backend.chat.service import _attach_answer_blocks

        self.assertEqual({"answer_blocks": [TIMETABLE_BLOCK]},
                         _attach_answer_blocks(None, [TIMETABLE_BLOCK]))
        self.assertIsNone(_attach_answer_blocks(None, []))

    def test_storage_keeps_the_blocks_while_it_trims_the_assets(self):
        from backend.chat.assets_bridge import trace_for_storage

        stored = trace_for_storage(
            {"answer_blocks": [TIMETABLE_BLOCK], "assets": [{"asset_id": "a::p0::img0"}]}
        )
        self.assertEqual([TIMETABLE_BLOCK], stored["answer_blocks"])

    def test_the_sync_response_carries_them_beside_the_assets(self):
        from backend.schemas.chat import ChatResponse

        response = ChatResponse(response="x", answer_blocks=[TIMETABLE_BLOCK])
        self.assertEqual("timetable", response.answer_blocks[0].kind)


class TheStreamSendsTheDataAheadOfItsText(scenarios.ParentTurnScenario):
    """Through the real stream, as a browser receives it."""

    async def _timetable_turn(self, storage=None):
        def the_timetable_tool_ran(ctx):
            ctx.note_answer_block(WEEK_TEXT, kind="timetable", data=WEEK_DATA)

        return await self.run_turn(
            [("m1", [], scenarios._tool_chunk()),
             ("m2", scenarios._split("دي جدول ليلى:"), None)],
            question="جدول بنتي",
            on_run=the_timetable_tool_ran,
            storage=storage,
        )

    async def test_the_data_arrives_before_the_text_that_places_it(self):
        """The other way round, the reader sees the markdown for one event, then a swap."""
        from backend.chat.service import BLOCK_MARKER

        events, shown, _ = await self._timetable_turn()
        kinds = [event["type"] for event in events]
        last_replace = len(kinds) - 1 - kinds[::-1].index("content_replace")
        self.assertLess(kinds.index("answer_blocks"), last_replace)
        blocks = next(e["answer_blocks"] for e in events if e["type"] == "answer_blocks")
        self.assertEqual([("timetable", 0)], [(b["kind"], b["index"]) for b in blocks])
        # And the text still carries the record for every client that draws nothing.
        self.assertIn(BLOCK_MARKER, shown)
        self.assertIn("**الأحد**", shown)

    async def test_the_trace_and_the_stored_answer_keep_them_for_a_reload(self):
        storage = self.new_storage()
        events, _, _ = await self._timetable_turn(storage)
        trace = next(e["rag_trace"] for e in events if e["type"] == "trace")
        self.assertEqual("timetable", trace["answer_blocks"][0]["kind"])
        stored = storage.saves[-1]["extra_message_data"][-1]["rag_trace"]
        self.assertEqual("timetable", stored["answer_blocks"][0]["kind"])


if __name__ == "__main__":
    unittest.main()
