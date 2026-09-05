"""Validating the ANSWER on a records turn, at the edges the happy-path suites skip.

A records turn is the one place in this deployment where the assistant states numbers
that nobody else can see. `get_student_records` writes nothing into the RAG trace, so the
tool's own reply is the only record of what the turn read, and two different checks have
to hold against it:

  * the numeric one — every figure in the answer has to exist in, or be derivable from,
    what the turn actually retrieved (`backend/chat/grounding.py`), with the records text
    supplied as `extra_evidence` so it can ground a figure WITHOUT ever counting as a
    citable chunk; and
  * the denial one — an answer that tells a parent nothing was found, on a turn whose
    tool returned marks, states no figure at all and is therefore invisible to the first
    check (`_denies_the_records` in `backend/chat/service.py`).

`test_planner_tool_selection.py` pins the two headline cases of each. This file covers
what sits around them, because that is where each check can quietly stop being a check:

  * digits arriving in Arabic-Indic (٠-٩) or Eastern Arabic (۰-۹) script — a verifier
    defeated by writing 45000 as ٤٥٠٠٠ would pass every fabrication that changed keyboard;
  * `extra_evidence` staying out of `evidence_count`, in the presence of real chunks, so
    a child's unnumbered marks can never make a `[n]` marker valid;
  * the floor — 87.5 is BELOW the default 100 and is not checked at all, which is the
    entire reason the denial check has to exist as a separate mechanism;
  * `Finalizer` as the thing that carries the records text from the stream to the check,
    including that its trace COUNTS tool results and quotes none of them (they hold a real
    child's name and their marks);
  * `_tool_messages_in`, the synchronous path's replacement for watching the stream, on
    every result shape `invoke` can hand back;
  * `_grounding_expected` across plan shapes, since the planner's narrowing is what
    decides which tools are bound and therefore whether the check runs at all; and
  * the three `records_denial_mode` settings, whose whole point is that `observe` detects
    without acting.

Everything here is offline: no model, no roster fetch, no records call. The profile is
process-global, so every test that changes `agent` restores it in a `finally` — leaking a
stub agent config out of this file breaks suites that never mention it.
"""
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.chat import service
from backend.chat.finalize import Finalizer
from backend.chat.grounding import DEFAULT_FLOOR, verify

#: One real-looking record line, in the shape `get_student_records` renders. Its name and
#: its marks are what `as_trace` must never repeat.
CHILD_NAME = "ليلى أحمد"
RECORDS_TEXT = f"{CHILD_NAME}: الرياضيات 87.5% — العلوم 91.0%"

RECORDS_TOOL = "get_student_records"
KNOWLEDGE_TOOL = "search_knowledge_base"


class _Plan:
    """The two fields the validation helpers read off a `TurnPlan`."""

    def __init__(self, exposed_tools=None, short_circuit=False):
        self.exposed_tools = exposed_tools
        self.short_circuit = short_circuit


class _Ctx:
    """A request context as the denial check sees it: a list of (tool, outcome)."""

    def __init__(self, outcomes=()):
        self.tool_outcomes = list(outcomes)


class _AgentCfg:
    """A stand-in for `profile.agent`, one knob at a time.

    Spelled out rather than loaded so these stay unit tests of the checks instead of
    assertions about whichever profile happens to be ambient.
    """

    def __init__(self, *, tools=(), denial_mode="off", denial_phrases=(),
                 grounding_mode="observe", grounding_floor=DEFAULT_FLOOR):
        self.tools = list(tools)
        self.records_denial_mode = denial_mode
        self.records_denial_phrases = list(denial_phrases)
        self.answer_grounding_mode = grounding_mode
        self.answer_grounding_number_floor = grounding_floor


class _ProfileAgentPatch:
    """Swap `service._PROFILE.agent` for the length of one test and put it back.

    The profile object is created once at import and shared by every suite in the run, so
    a test that replaced `agent` and raised before restoring it would fail files that
    never touch this feature — and it would fail them somewhere far away.
    """

    def __init__(self, agent):
        self._agent = agent
        self._original = None

    def __enter__(self):
        self._original = service._PROFILE.agent
        service._PROFILE.__dict__["agent"] = self._agent
        return self._agent

    def __exit__(self, *exc):
        service._PROFILE.__dict__["agent"] = self._original
        return False


def _finalizer_answering(answer, *, tool_texts=()):
    """A finalizer holding a finished answer and whatever the tools returned."""
    finalizer = Finalizer()
    for text in tool_texts:
        finalizer.note_tool_result(ToolMessage(content=text, tool_call_id="c1"))
    finalizer.replace_answer(answer)
    return finalizer


class RecordsTextGroundsFiguresWithoutBecomingEvidence(unittest.TestCase):
    """`extra_evidence` answers one question and must stay out of the other.

    Numbers and provenance are separate: a child's marks are real material the turn read,
    so they may ground a figure; they carry no `[n]`, so they must never enlarge the set a
    citation marker is checked against.
    """

    def test_extra_evidence_never_enlarges_the_citation_denominator(self):
        """The regression this guards: one retrieved chunk plus two records strings
        makes `[2]` look valid if the two lists are simply concatenated."""
        report = verify(
            "الرسوم 45000 جنيه [2]",
            ["الرسوم 45000 جنيه"],
            extra_evidence=[RECORDS_TEXT, "سجل آخر"],
        )
        self.assertEqual(report.evidence_count, 1)
        self.assertEqual(report.invalid_citations, (2,))
        self.assertFalse(report.ok)

    def test_a_citation_within_the_real_chunks_still_passes(self):
        report = verify(
            "الرسوم 45000 جنيه [1]",
            ["الرسوم 45000 جنيه"],
            extra_evidence=[RECORDS_TEXT],
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.evidence_count, 1)

    def test_chunks_and_records_ground_figures_side_by_side(self):
        """Both sources feed the same number set; neither shadows the other."""
        report = verify(
            "الرسوم 45000 والغرامة 1200 [1]",
            ["الرسوم 45000 جنيه"],
            extra_evidence=["غرامة تأخير 1200 جنيه"],
        )
        self.assertTrue(report.ok, report.reason)

    def test_a_figure_in_neither_source_is_still_caught(self):
        report = verify(
            "الرسوم 45000 والغرامة 9999",
            ["الرسوم 45000 جنيه"],
            extra_evidence=["غرامة تأخير 1200 جنيه"],
        )
        self.assertFalse(report.ok)
        self.assertIn(9999.0, report.ungrounded)
        self.assertNotIn(45000.0, report.ungrounded)

    def test_empty_and_missing_records_strings_are_ignored_not_counted(self):
        """A tool that returned nothing contributes nothing — and must not raise, and
        must not be counted as a chunk either."""
        report = verify(
            "الرسوم 45000 جنيه",
            [],
            extra_evidence=["", None, "الرسوم 45000 جنيه", ""],
        )
        self.assertTrue(report.ok, report.reason)
        self.assertEqual(report.evidence_count, 0)

    def test_records_that_ground_nothing_leave_the_answer_ungrounded(self):
        report = verify("الرسوم 45000 جنيه", [], extra_evidence=[None, ""])
        self.assertFalse(report.ok)
        self.assertEqual(report.ungrounded, (45000.0,))

    def test_empty_evidence_chunks_do_not_count_toward_the_denominator(self):
        """A retrieved chunk with no text cannot be cited, so `[1]` against it is the
        cite-with-nothing-retrieved case, not a valid citation."""
        report = verify("لا رقم هنا [1]", ["", None], extra_evidence=[RECORDS_TEXT])
        self.assertEqual(report.evidence_count, 0)
        self.assertTrue(report.cited_without_evidence)


class TheCheckReadsEveryDigitScript(unittest.TestCase):
    """A check that could be defeated by changing keyboard would not be a check.

    The corpus, the records facade and the model all emit digits in three scripts, and
    the same figure written in two of them is one claim.
    """

    def test_records_written_in_another_script_ground_an_ascii_answer(self):
        for label, evidence in (
            ("arabic-indic", "الرسوم ٤٥٠٠٠ جنيه"),
            ("eastern-arabic", "الرسوم ۴۵۰۰۰ جنيه"),
        ):
            with self.subTest(script=label):
                report = verify("The fee is 45000 EGP", [], extra_evidence=[evidence])
                self.assertTrue(report.ok, report.reason)

    def test_an_answer_written_in_another_script_is_grounded_by_ascii_records(self):
        for label, answer in (
            ("arabic-indic", "الرسوم ٤٥٠٠٠ جنيه"),
            ("eastern-arabic", "الرسوم ۴۵۰۰۰ جنيه"),
        ):
            with self.subTest(script=label):
                report = verify(answer, [], extra_evidence=["fee 45000 EGP"])
                self.assertTrue(report.ok, report.reason)

    def test_a_fabrication_in_another_script_is_still_caught(self):
        """The folding must not become a way of passing anything: a different NUMBER in
        Arabic-Indic digits is still ungrounded."""
        report = verify("الرسوم ٣٠٠٠٠ جنيه", [], extra_evidence=["fee 45000 EGP"])
        self.assertFalse(report.ok)
        self.assertEqual(report.ungrounded, (30000.0,))

    def test_a_multiplier_word_scales_a_non_ascii_figure(self):
        """«٤٥ ألف» is 45000, the same claim as "45000" — the scale word and the digit
        folding have to compose, or one of them undoes the other."""
        report = verify("الرسوم ٤٥ ألف جنيه", [], extra_evidence=["fee 45000 EGP"])
        self.assertTrue(report.ok, report.reason)


class TheFloorIsWhyTheDenialCheckExists(unittest.TestCase):
    """Marks live below the floor, so grounding never looks at them.

    `answer_grounding_number_floor` defaults to 100 because "3 instalments" against a
    corpus that spells «ثلاث» in words is a formatting difference, not a fabrication. The
    consequence is that 87.5% and 91.0% — the most figure-dense thing this assistant ever
    says — are outside the numeric check entirely. Anyone lowering that floor should have
    to come here and decide that on purpose.
    """

    def test_the_default_floor_is_above_any_percentage_mark(self):
        self.assertGreater(DEFAULT_FLOOR, 100 - 1)
        self.assertGreaterEqual(DEFAULT_FLOOR, 100)

    def test_a_mark_the_records_never_reported_is_not_even_checked(self):
        """The answer invents 99.9% against records saying 87.5% and it still passes —
        not a bug, the floor's deliberate scope. The denial check covers this class."""
        report = verify("ليلى حاصلة على 99.9% في الرياضيات", [], extra_evidence=[RECORDS_TEXT])
        self.assertTrue(report.ok)
        self.assertEqual(report.checked, 0)

    def test_lowering_the_floor_brings_marks_back_into_scope(self):
        """Proves the previous test measures the floor rather than a broken extractor."""
        report = verify(
            "ليلى حاصلة على 99.9% في الرياضيات", [], extra_evidence=[RECORDS_TEXT], floor=1
        )
        self.assertFalse(report.ok)
        self.assertIn(99.9, report.ungrounded)

    def test_a_reported_mark_passes_once_the_floor_is_lowered(self):
        report = verify(
            "ليلى حاصلة على 87.5% في الرياضيات", [], extra_evidence=[RECORDS_TEXT], floor=1
        )
        self.assertTrue(report.ok, report.reason)


class DerivationWorksFromRecordsToo(unittest.TestCase):
    """`extra_evidence` feeds the derivation search, not just the membership test.

    If records text only supported exact matches, an answer that divided a grounded total
    into instalments would be scored a fabrication on precisely the turns the planner
    narrowed to the records tool.
    """

    def test_a_grounded_total_split_into_instalments_is_grounded(self):
        report = verify(
            "كل قسط 15000 جنيه", [], extra_evidence=["إجمالي 45000 على ثلاث دفعات"]
        )
        self.assertTrue(report.ok, report.reason)

    def test_a_sum_of_two_records_figures_is_grounded(self):
        report = verify("الإجمالي 32000", [], extra_evidence=["رسوم 20000", "أنشطة 12000"])
        self.assertTrue(report.ok, report.reason)

    def test_a_percentage_in_the_records_applies_to_a_grounded_figure(self):
        report = verify(
            "بعد الخصم 27000 جنيه", [], extra_evidence=["الرسوم 30000 وخصم 10% للأخ التاني"]
        )
        self.assertTrue(report.ok, report.reason)

    def test_derivation_still_rejects_an_unrelated_figure(self):
        """The allowance keeps its teeth: 30000 is not any split, sum or share of 45000."""
        report = verify("الرسوم 33333 جنيه", [], extra_evidence=["إجمالي 45000 على ثلاث دفعات"])
        self.assertFalse(report.ok)
        self.assertEqual(report.ungrounded, (33333.0,))


class TheFinalizerCarriesWhatTheToolsReturned(unittest.TestCase):
    """The records text has to survive from the stream to the check, and no further.

    It is held in memory for one turn because it is the only record of what the turn
    read. It is never persisted and never traced, because it is a real child's name and
    their marks.
    """

    def test_tool_result_texts_come_back_in_the_order_they_arrived(self):
        finalizer = Finalizer()
        finalizer.note_tool_result(ToolMessage(content="first", tool_call_id="a"))
        finalizer.note_tool_result(ToolMessage(content="second", tool_call_id="b"))
        self.assertEqual(finalizer.tool_result_texts, ["first", "second"])

    def test_a_result_with_no_message_counts_but_contributes_no_text(self):
        """The streamed path calls this bare in one place; it must still register that a
        tool ran, or an answer citing `[1]` on a tool-less turn stops being detectable."""
        finalizer = Finalizer()
        finalizer.note_tool_result()
        finalizer.note_tool_result(ToolMessage(content="", tool_call_id="a"))
        self.assertEqual(finalizer.tool_results, 2)
        self.assertEqual(finalizer.tool_result_texts, [])

    def test_tool_result_texts_returns_a_copy_the_caller_cannot_corrupt(self):
        finalizer = Finalizer()
        finalizer.note_tool_result(ToolMessage(content="first", tool_call_id="a"))
        finalizer.tool_result_texts.append("forged")
        self.assertEqual(finalizer.tool_result_texts, ["first"])

    def test_verify_passes_the_retained_texts_through_without_being_asked(self):
        """The bug this closes: both entry points would otherwise have to remember to
        supply the records text, and the one that forgot would silently stop checking a
        whole class of answer."""
        finalizer = _finalizer_answering("الرسوم 45000 جنيه", tool_texts=["الرسوم 45000 جنيه"])
        report = finalizer.verify([])
        self.assertTrue(report.ok, report.reason)
        self.assertEqual(report.evidence_count, 0)

    def test_without_a_tool_result_the_same_answer_fails(self):
        """Confirms the previous test measured the pass-through and not a lenient check."""
        finalizer = _finalizer_answering("الرسوم 45000 جنيه")
        report = finalizer.verify([])
        self.assertFalse(report.ok)
        self.assertEqual(report.ungrounded, (45000.0,))

    def test_the_verdict_is_retained_on_the_finalizer(self):
        finalizer = _finalizer_answering("الرسوم 45000 جنيه")
        self.assertIsNone(finalizer.grounding)
        report = finalizer.verify([])
        self.assertIs(finalizer.grounding, report)

    def test_the_trace_counts_tool_results_and_quotes_none_of_them(self):
        """The whole reason the count and the text are separate: the trace is persisted
        and shipped to LangSmith, and these strings are a child's name and their marks."""
        finalizer = _finalizer_answering("ok", tool_texts=[RECORDS_TEXT])
        finalizer.verify([])
        trace = finalizer.as_trace()
        self.assertEqual(trace["finalize_tool_results"], 1)
        rendered = json.dumps(trace, ensure_ascii=False)
        self.assertNotIn(CHILD_NAME, rendered)
        self.assertNotIn("87.5", rendered)
        self.assertNotIn(RECORDS_TEXT, rendered)

    def test_the_trace_carries_the_grounding_verdict_once_verify_has_run(self):
        finalizer = _finalizer_answering("الرسوم 45000 جنيه")
        finalizer.verify([])
        trace = finalizer.as_trace()
        self.assertFalse(trace["grounding_ok"])
        self.assertEqual(trace["grounding_ungrounded_numbers"], ["45000"])

    def test_a_finalizer_never_verified_reports_no_verdict_at_all(self):
        """"the check passed" and "the check never ran" have to look different in a log,
        or a deployment cannot tell a working check from a disengaged one."""
        trace = _finalizer_answering("ok").as_trace()
        self.assertNotIn("grounding_ok", trace)
        self.assertEqual(trace["finalize_tool_results"], 0)


class ReadingToolResultsOffAFinishedRun(unittest.TestCase):
    """`_tool_messages_in` is the synchronous path's substitute for watching the stream.

    It runs inside the answer-validation step, so anything it raises takes the answer
    down with it. Every shape `invoke` can plausibly return has to come back as a list.
    """

    def test_tool_messages_are_pulled_out_in_order(self):
        result = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content=""),
                ToolMessage(content="first", tool_call_id="a"),
                AIMessage(content="answer"),
                ToolMessage(content="second", tool_call_id="b"),
            ]
        }
        found = service._tool_messages_in(result)
        self.assertEqual([m.content for m in found], ["first", "second"])

    def test_every_other_result_shape_yields_an_empty_list(self):
        for label, result in (
            ("none", None),
            ("string", "an answer"),
            ("list", [ToolMessage(content="x", tool_call_id="a")]),
            ("dict without messages", {"output": "an answer"}),
            ("dict with null messages", {"messages": None}),
            ("dict with empty messages", {"messages": []}),
            ("dict with no tool messages", {"messages": [AIMessage(content="hi")]}),
        ):
            with self.subTest(shape=label):
                self.assertEqual(service._tool_messages_in(result), [])


class WhichTurnsGetChecked(unittest.TestCase):
    """`_grounding_expected` reads the tools the PLAN bound, not the profile's list.

    The narrowing and the check are one change: once the planner started binding
    `[get_student_records]` alone, reading the citation set here would have made the
    narrowing itself the thing that switched the check off.
    """

    def _expected(self, exposed, *, profile_tools=(KNOWLEDGE_TOOL,)):
        with _ProfileAgentPatch(_AgentCfg(tools=profile_tools)):
            return service._grounding_expected(_Plan(exposed_tools=exposed))

    def test_each_plan_shape_decides_the_check_for_itself(self):
        for label, exposed, expected in (
            ("records only", [RECORDS_TOOL], True),
            ("knowledge only", [KNOWLEDGE_TOOL], True),
            ("both", [KNOWLEDGE_TOOL, RECORDS_TOOL], True),
            ("nothing bound", [], False),
            ("an unchecked tool", ["search_products_unknown"], False),
        ):
            with self.subTest(plan=label):
                self.assertEqual(self._expected(exposed), expected)

    def test_no_narrowing_falls_back_to_the_profiles_own_tool_list(self):
        """`exposed_tools is None` means "bind everything", so the profile's list is the
        turn's list — and a profile that binds no checked tool is not checked."""
        self.assertTrue(self._expected(None, profile_tools=(KNOWLEDGE_TOOL,)))
        self.assertTrue(self._expected(None, profile_tools=(RECORDS_TOOL,)))
        self.assertFalse(self._expected(None, profile_tools=()))

    def test_an_empty_exposed_list_is_not_read_as_no_narrowing(self):
        """`[]` and `None` are different plans and the falsy check that conflates them
        would check a turn that bound nothing at all."""
        with _ProfileAgentPatch(_AgentCfg(tools=(KNOWLEDGE_TOOL,))):
            self.assertFalse(service._grounding_expected(_Plan(exposed_tools=[])))


class ACitationHabitMustNotSinkARecordsAnswer(unittest.TestCase):
    """The false positive found by running the real model end to end.

    The Arabic style pack (`packs/school/arabic_style.j2`) shows two worked examples
    of keeping a figure exact, and both happen to end in `[1]`/`[2]` — it is a STYLE
    example, shown on every Arabic turn regardless of which tools are bound, and it
    taught the model to imitate the marker even on a turn where nothing was ever asked
    to cite anything. Measured live: a records-only turn answered a child's real marks
    correctly, added a stray `[1]` out of habit, and `_enforce_grounding` — seeing zero
    RAG chunks retrieved, because none were — flagged it as "cited evidence on a turn
    that retrieved none" and replaced a fully correct answer with the could-not-verify
    copy, under `answer_grounding_mode: enforce`.

    `_grounding_expected` (CHECKED_TOOLS) and `_citations_expected` (GROUNDED_TOOLS)
    have to answer separately for exactly this reason: the NUMBER in that answer is
    rightly checked and was fine; the STRAY MARKER should never have been asked about
    at all, because the prompt never gave this turn a citation contract to keep.
    """

    def test_the_exact_reported_failure_no_longer_replaces_a_correct_answer(self):
        finalizer = _finalizer_answering(
            "الرياضيات: 88.0 % (A)\nالعربي: 91.5 % (A)\n\n[1]",
            tool_texts=["STUDENT_GRADES: الرياضيات 88.0%, العربي 91.5%"],
        )
        with _ProfileAgentPatch(_AgentCfg(grounding_mode="enforce")):
            replacement = service._enforce_grounding(
                finalizer, {}, _Plan(exposed_tools=[RECORDS_TOOL])
            )
        self.assertEqual(replacement, "")
        self.assertTrue(finalizer.grounding.ok)

    def test_citations_are_not_expected_on_a_records_only_turn(self):
        self.assertFalse(service._citations_expected(_Plan(exposed_tools=[RECORDS_TOOL])))

    def test_citations_are_still_expected_on_a_knowledge_base_turn(self):
        """The fix narrows WHO is exempt, not whether the check exists at all — a
        knowledge-base answer citing `[1]` with nothing retrieved must still be caught."""
        self.assertTrue(service._citations_expected(_Plan(exposed_tools=[KNOWLEDGE_TOOL])))
        finalizer = _finalizer_answering("الرسوم زي ما هو مكتوب. [1]")
        with _ProfileAgentPatch(_AgentCfg(grounding_mode="enforce")):
            replacement = service._enforce_grounding(
                finalizer, {}, _Plan(exposed_tools=[KNOWLEDGE_TOOL])
            )
        self.assertNotEqual(replacement, "")

    def test_a_mixed_turn_binding_both_tools_still_expects_citations(self):
        """Records joins the checked set without weakening the citation contract on a
        turn that ALSO bound the knowledge tool — the exemption is per-answer scope,
        not a blanket switch that a records tool anywhere in the plan flips off."""
        self.assertTrue(
            service._citations_expected(_Plan(exposed_tools=[KNOWLEDGE_TOOL, RECORDS_TOOL]))
        )

    def test_a_genuinely_fabricated_figure_is_still_caught_on_a_records_turn(self):
        """The fix removes the CITATION false positive; it must not also blunt the
        NUMBER check that was the entire point of adding records to CHECKED_TOOLS."""
        finalizer = _finalizer_answering(
            "حضوره كان 500 يوم من أصل 600",
            tool_texts=["ATTENDANCE: present 55, total 58 sessions"],
        )
        with _ProfileAgentPatch(_AgentCfg(grounding_mode="enforce")):
            replacement = service._enforce_grounding(
                finalizer, {}, _Plan(exposed_tools=[RECORDS_TOOL])
            )
        self.assertNotEqual(replacement, "")
        self.assertFalse(finalizer.grounding.ok)


class DenyingARecordTheTurnActuallyRead(unittest.TestCase):
    """Both halves are required, and they come from opposite ends.

    What the tool returned is fact, reported by the tool. What the answer claims is a
    phrase list, which is a guess — so an outcome that legitimately found nothing must
    never fire, however the answer is worded.
    """

    def _denies(self, outcomes, answer, phrases=("couldn't find", "لم اجد")):
        with _ProfileAgentPatch(_AgentCfg(denial_phrases=phrases)):
            return service._denies_the_records(_Ctx(outcomes), answer)

    def test_a_denial_after_each_successful_outcome_is_caught(self):
        for outcome in sorted(service.RECORDS_RETRIEVED):
            with self.subTest(outcome=outcome):
                self.assertTrue(self._denies(
                    [(RECORDS_TOOL, outcome)], "Sorry, I couldn't find any records."
                ))

    def test_an_outcome_that_found_nothing_makes_the_denial_correct(self):
        """These are turns where saying so is the RIGHT answer; flagging them would train
        operators to ignore the check."""
        for outcome in (
            "no_records",
            "no_students",
            "unavailable",
            "not_authorized",
            "which_student",
            "call_limit",
            "not_a_parent",
        ):
            with self.subTest(outcome=outcome):
                self.assertFalse(self._denies(
                    [(RECORDS_TOOL, outcome)], "Sorry, I couldn't find any records."
                ))

    def test_a_successful_outcome_anywhere_in_the_turn_is_enough(self):
        """A turn that called the tool twice — the measured failure did exactly that —
        must be judged on the call that succeeded."""
        self.assertTrue(self._denies(
            [(RECORDS_TOOL, "which_student"), (RECORDS_TOOL, "grades")],
            "I couldn't find any records.",
        ))

    def test_a_turn_that_called_no_tool_at_all_never_fires(self):
        self.assertFalse(self._denies([], "I couldn't find any records."))

    def test_the_match_folds_through_name_key_so_an_alif_variant_still_hits(self):
        """The phrase list is written by hand into yaml; a hamza the copywriter typed and
        the model did not must not be the difference between checked and unchecked."""
        self.assertTrue(self._denies(
            [(RECORDS_TOOL, "grades")],
            "للأسف لم أجد أي سجلات لليلى",
            phrases=("لم اجد",),
        ))
        self.assertTrue(self._denies(
            [(RECORDS_TOOL, "grades")],
            "للأسف لم اجد أي سجلات لليلى",
            phrases=("لم أجد",),
        ))

    def test_the_match_ignores_case_the_model_chose(self):
        self.assertTrue(self._denies(
            [(RECORDS_TOOL, "grades")], "I COULDN'T FIND ANY RECORDS."
        ))

    def test_an_answer_reporting_the_marks_is_not_flagged(self):
        self.assertFalse(self._denies(
            [(RECORDS_TOOL, "grades")], "ليلى حاصلة على 87.5% في الرياضيات و 91.0% في العلوم"
        ))

    def test_an_empty_answer_is_not_a_denial(self):
        self.assertFalse(self._denies([(RECORDS_TOOL, "grades")], ""))
        self.assertFalse(self._denies([(RECORDS_TOOL, "grades")], "   "))

    def test_no_phrases_configured_means_do_not_guess(self):
        self.assertFalse(self._denies(
            [(RECORDS_TOOL, "grades")], "I couldn't find any records.", phrases=()
        ))

    def test_an_empty_phrase_in_the_list_does_not_match_everything(self):
        """An empty string is a substring of every answer, so the check has to skip it —
        otherwise one stray entry turns the rule into "replace every records answer"."""
        self.assertFalse(self._denies(
            [(RECORDS_TOOL, "grades")],
            "ليلى حاصلة على 87.5% في الرياضيات",
            phrases=("",),
        ))

    def test_a_whitespace_only_phrase_does_not_match_everything(self):
        """A config typo must not become a total outage of the feature it guards.

        It was red when written. The guard skipped falsy phrases with `if phrase`, which
        drops "" but keeps "   " — and `name_key("   ")` folds to the empty string, a
        substring of every answer. One blank line in a deployment's
        `records_denial_phrases` made EVERY records answer read as a denial, and under
        `records_denial_mode: enforce` replaced every one with the could-not-verify copy.
        The emptiness test has to happen AFTER folding."""
        self.assertFalse(self._denies(
            [(RECORDS_TOOL, "grades")],
            "ليلى حاصلة على 87.5% في الرياضيات",
            phrases=("   ",),
        ))


class TheThreeModesOfTheDenialCheck(unittest.TestCase):
    """`observe` has to DETECT without acting, or it cannot be measured before enforcing.

    A deployment turns this on at `observe`, reads its own logs, and only then moves to
    `enforce`. That is only meaningful if the two modes differ solely in the return value.
    """

    ANSWER = "Sorry, I couldn't find any records for her."
    PHRASES = ("couldn't find",)

    #: `plan=None` is a meaningful argument to the helper under test, so the "caller said
    #: nothing" default cannot be None as well.
    _DEFAULT = object()

    def _enforce(self, mode, *, plan=_DEFAULT, answer=None, outcomes=None):
        finalizer = _finalizer_answering(
            self.ANSWER if answer is None else answer, tool_texts=[RECORDS_TEXT]
        )
        ctx = _Ctx(outcomes if outcomes is not None else [(RECORDS_TOOL, "grades")])
        if plan is self._DEFAULT:
            plan = _Plan(exposed_tools=[RECORDS_TOOL])
        with _ProfileAgentPatch(_AgentCfg(denial_mode=mode, denial_phrases=self.PHRASES)):
            return service._enforce_records_agreement(finalizer, ctx, plan)

    def test_off_leaves_a_denying_answer_alone(self):
        self.assertEqual(self._enforce("off"), "")

    def test_observe_leaves_the_answer_alone_but_the_situation_is_still_detected(self):
        self.assertEqual(self._enforce("observe"), "")
        with _ProfileAgentPatch(_AgentCfg(denial_phrases=self.PHRASES)):
            self.assertTrue(
                service._denies_the_records(_Ctx([(RECORDS_TOOL, "grades")]), self.ANSWER)
            )

    def test_enforce_serves_the_unverified_answer_copy(self):
        self.assertEqual(self._enforce("enforce"), service._COPY.unverified_answer)

    def test_enforce_leaves_an_answer_that_reports_the_marks_alone(self):
        self.assertEqual(
            self._enforce("enforce", answer="ليلى حاصلة على 87.5% في الرياضيات"), ""
        )

    def test_enforce_leaves_a_denial_alone_when_the_lookup_really_failed(self):
        self.assertEqual(
            self._enforce("enforce", outcomes=[(RECORDS_TOOL, "no_records")]), ""
        )

    def test_a_short_circuited_plan_is_never_checked(self):
        """The which-child reply is the profile's own copy, served without a model. It is
        not the model's answer and must not be judged as one."""
        self.assertEqual(
            self._enforce("enforce", plan=_Plan(exposed_tools=[], short_circuit=True)), ""
        )

    def test_a_turn_with_no_plan_at_all_is_never_checked(self):
        self.assertEqual(self._enforce("enforce", plan=None), "")

    def test_the_profile_agent_is_restored_after_every_patch(self):
        """A guard on this file's own machinery: the profile is process-global, and a
        stub leaking out of here fails suites that never mention records."""
        original = service._PROFILE.agent
        try:
            with _ProfileAgentPatch(_AgentCfg()):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIs(service._PROFILE.agent, original)


class TheTwoChecksCoverDifferentFailures(unittest.TestCase):
    """Why there are two mechanisms and not one.

    The measured failure — records returned 87.5% and 91.0% and the assistant said it
    found nothing — is invisible to the numeric check twice over: the answer states no
    figure, and the figures it would have stated are below the floor. Pinning both facts
    together is what stops someone folding the denial check into the grounding report.
    """

    def test_the_denial_answer_passes_grounding_cleanly(self):
        finalizer = _finalizer_answering(
            "I couldn't find any records for her.", tool_texts=[RECORDS_TEXT]
        )
        report = finalizer.verify([])
        self.assertTrue(report.ok)
        self.assertEqual(report.checked, 0)

    def test_the_denial_check_catches_what_grounding_passed(self):
        with _ProfileAgentPatch(_AgentCfg(denial_phrases=("couldn't find",))):
            self.assertTrue(service._denies_the_records(
                _Ctx([(RECORDS_TOOL, "grades")]), "I couldn't find any records for her."
            ))


if __name__ == "__main__":
    unittest.main()
