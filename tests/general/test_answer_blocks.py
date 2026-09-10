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
"""
import unittest


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


if __name__ == "__main__":
    unittest.main()
