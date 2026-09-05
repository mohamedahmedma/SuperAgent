"""Which tool a turn gets, decided before the model is asked.

Three questions were measured failing against `openai/gpt-oss-20b` with both tools bound:

    «درجات ليلى أحمد كام؟»                    searched the fee corpus, then reported that
                                              no information about the child existed
    «تفاصيل درجة ليلى في الرياضيات»            searched the knowledge base, not the record
    "My daughter is Fatma Mohamed, ...?"      called the records tool twice and still
                                              answered "I couldn't find any records",
                                              having been handed 87.5% and 91.0%

Every one is a selection failure, and the planner already held the answer: it had
resolved which child the turn was about, from the school's own roster, before the agent
was built. These tests cover the five things that had to become true for that answer to
reach the turn — the classifier separating a records question from a school matter, the
plan binding one tool, the agent being made to call it, the tool reading the roster's
child rather than the model's spelling of a name, and the answer being checked against
what the tool actually returned.

The assertions lean on the DEGRADED cases as hard as the narrowing ones. Every rung here
is allowed to fail, and each has to fail toward binding everything, because that is the
behaviour that shipped before any of this existed.
"""
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.chat import runtime
from backend.chat.child_resolution import no_child, resolve_child
from backend.chat.child_roster import ChildOption
from backend.chat.grounding import verify
from backend.chat.signals import RequestSignals
from backend.chat.turn_policy import KNOWLEDGE_TOOL, RECORDS_TOOL, resolve_turn

LAYLA = ChildOption(student_id="S-1", label="ليلى أحمد", gender="female", year_level="Year 4")
OMAR = ChildOption(student_id="S-2", label="عمر أحمد", gender="male")


class _Agent:
    """The school profile's shape, without loading the school profile.

    Spelled out rather than loaded so a test can turn one knob at a time — and so these
    stay unit tests of the policy rather than assertions about a deployment's yaml, which
    `test_school_profile.py` is for.
    """

    tools = [KNOWLEDGE_TOOL, RECORDS_TOOL]
    social_phrases = []
    social_reply_mode = "model"
    narrow_tools_to_the_turn = True
    year_reference_markers = ()


class _Copy:
    social = None
    out_of_domain = None
    which_child = "Which child do you mean?"


def _plan(child=None, *, agent=None, **signal_kwargs):
    signals = RequestSignals(question="q", **signal_kwargs)
    return resolve_turn(
        signals, agent_config=agent or _Agent(), copy_config=_Copy(), child=child
    )


def _settled(roster=(LAYLA,)):
    """A turn whose child the roster resolved on its own."""
    return resolve_child(reference="context", roster=list(roster))


class ThePlanPicksTheTool(unittest.TestCase):
    def test_a_question_about_a_childs_own_record_binds_only_the_records_tool(self):
        plan = _plan(_settled(), about_child=True, child_question_kind="records")
        self.assertEqual(plan.exposed_tools, [RECORDS_TOOL])

    def test_a_school_matter_asked_about_a_child_binds_only_the_knowledge_tool(self):
        """«مصاريف ابني» is about_child AND a knowledge question.

        The distinction the whole feature rests on. Reading `about_child` as "records"
        would send one of the commonest messages this deployment gets to the tool that
        holds no fee schedule.
        """
        plan = _plan(_settled(), about_child=True, child_question_kind="school_matter")
        self.assertEqual(plan.exposed_tools, [KNOWLEDGE_TOOL])

    def test_a_question_that_asks_for_each_binds_everything(self):
        plan = _plan(_settled(), about_child=True, child_question_kind="both")
        self.assertIsNone(plan.exposed_tools)

    def test_the_tools_keep_the_profiles_own_order(self):
        class _Both(_Agent):
            tools = [RECORDS_TOOL, KNOWLEDGE_TOOL]

        plan = _plan(_settled(), agent=_Both(), about_child=True,
                     child_question_kind="records")
        self.assertEqual(plan.exposed_tools, [RECORDS_TOOL])


class EveryFailureBindsEverything(unittest.TestCase):
    """The narrowing is an optimisation, so it may only ever act on a positive answer."""

    def test_a_turn_that_resolved_no_child_is_left_alone(self):
        plan = _plan(no_child("not about a child"), child_question_kind="records")
        self.assertIsNone(plan.exposed_tools)

    def test_a_turn_that_cannot_say_which_child_binds_nothing_and_asks(self):
        """Not "bind everything and let the agent sort it out". Which child is meant is
        a fact about a roster, so the turn ends here with the question — there is no tool
        worth binding for a lookup nobody can perform yet."""
        plan = _plan(resolve_child(reference="child", roster=[LAYLA, OMAR]),
                     about_child=True, child_question_kind="records")
        self.assertEqual(plan.child_options, ["ليلى أحمد", "عمر أحمد"])
        self.assertTrue(plan.short_circuit)
        self.assertEqual(plan.exposed_tools, [])
        self.assertEqual(plan.forced_tool, "")

    def test_a_kind_outside_the_closed_set_is_left_alone(self):
        """What a classifier that abstained, rate-limited or invented a value produces."""
        for kind in ("", "unknown", "RECORDS!", "records "):
            with self.subTest(kind=kind):
                plan = _plan(_settled(), about_child=True, child_question_kind=kind)
                self.assertIsNone(plan.exposed_tools)

    def test_the_default_signal_is_left_alone(self):
        """A `RequestSignals` nobody classified must plan exactly as it did before."""
        self.assertIsNone(_plan(_settled()).exposed_tools)

    def test_a_profile_that_has_not_asked_for_it_is_left_alone(self):
        class _Off(_Agent):
            narrow_tools_to_the_turn = False

        plan = _plan(_settled(), agent=_Off(), about_child=True,
                     child_question_kind="records")
        self.assertIsNone(plan.exposed_tools)

    def test_a_profile_binding_no_such_tool_narrows_nothing(self):
        """A plan naming a tool the profile refuses is a startup error in `build_tools`,
        and an empty list would read as "bind no tools at all"."""
        class _KnowledgeOnly(_Agent):
            tools = [KNOWLEDGE_TOOL]

        plan = _plan(_settled(), agent=_KnowledgeOnly(), about_child=True,
                     child_question_kind="records")
        self.assertIsNone(plan.exposed_tools)
        self.assertIn("binds no such tool", "; ".join(plan.reasons))

    def test_a_social_turn_still_binds_nothing(self):
        """The rung above this one still wins: narrowing must not resurrect a tool."""
        plan = _plan(_settled(), is_social=True, about_child=True,
                     child_question_kind="records")
        self.assertEqual(plan.exposed_tools, [])


class TheDecisionIsVisible(unittest.TestCase):
    def test_the_trace_names_the_bound_tools_and_the_required_one(self):
        trace = _plan(_settled(), about_child=True,
                      child_question_kind="records").as_trace()
        self.assertEqual(trace["turn_exposed_tools"], [RECORDS_TOOL])
        self.assertEqual(trace["turn_forced_tool"], RECORDS_TOOL)

    def test_the_trace_names_no_child(self):
        """A child is resolved here and the trace is persisted and streamed. It records
        THAT one was settled, never which — the rule `as_trace` already follows."""
        trace = _plan(_settled(), about_child=True,
                      child_question_kind="records").as_trace()
        self.assertTrue(trace["turn_child_resolved"])
        self.assertNotIn("ليلى", str(trace))

    def test_the_plan_never_offers_a_tool_without_a_reason(self):
        plan = _plan(_settled(), about_child=True, child_question_kind="records")
        self.assertIn("records question about one child", "; ".join(plan.reasons))


class TheForcedTool(unittest.TestCase):
    """`tool_choice`, and only on the turn's first model call.

    Measured against this endpoint: gpt-oss ignores `tool_choice` once a tool result is
    in the history, so forcing later passes buys nothing and risks a turn that cannot
    end. The first call is where selection is decided.
    """

    def test_narrowing_to_one_tool_requires_it(self):
        plan = _plan(_settled(), about_child=True, child_question_kind="records")
        self.assertEqual(plan.forced_tool, RECORDS_TOOL)

    def test_a_turn_that_narrowed_nothing_requires_nothing(self):
        self.assertEqual(_plan(_settled()).forced_tool, "")


class _Request:
    """The parts of `ModelRequest` the middleware touches."""

    def __init__(self, tools, state, tool_choice=None):
        self.tools = tools
        self.state = state
        self.tool_choice = tool_choice

    def override(self, **overrides):
        return _Request(
            overrides.get("tools", self.tools),
            self.state,
            overrides.get("tool_choice", self.tool_choice),
        )


class _Tool:
    def __init__(self, name):
        self.name = name


class _Ctx:
    def __init__(self, forced_tool=""):
        self.forced_tool = forced_tool


def _chosen(state, *, forced=RECORDS_TOOL, tools=(KNOWLEDGE_TOOL, RECORDS_TOOL),
            tool_choice=None):
    """The `tool_choice` the middleware lets through, given the turn so far."""
    middleware = runtime._force_the_planned_tool(_Ctx(forced))
    seen = {}

    def handler(request):
        seen["tool_choice"] = request.tool_choice
        return AIMessage(content="")

    middleware.wrap_model_call(
        _Request([_Tool(name) for name in tools], state, tool_choice), handler
    )
    return seen["tool_choice"]


class ForcingTheCall(unittest.TestCase):
    def test_the_first_call_of_the_turn_is_required_to_use_the_tool(self):
        state = {"messages": [HumanMessage(content="درجات ليلى كام؟")]}
        self.assertEqual(_chosen(state), RECORDS_TOOL)

    def test_a_turn_with_a_tool_result_behind_it_is_not_forced(self):
        """The measured provider limit, and the reason it costs nothing: by the time a
        result exists the selection has already happened."""
        state = {"messages": [
            HumanMessage(content="درجات ليلى كام؟"),
            AIMessage(content="", tool_calls=[
                {"name": RECORDS_TOOL, "args": {}, "id": "call-1"}
            ]),
            ToolMessage(content="الرياضيات ٨٧.٥٪", tool_call_id="call-1"),
        ]}
        self.assertIsNone(_chosen(state))

    def test_a_turn_the_budget_has_already_counted_is_not_forced(self):
        """`tool_calls_made` catches the one case the messages cannot — a call that was
        requested and produced no result message."""
        state = {"messages": [HumanMessage(content="q")],
                 "tool_calls_made": {RECORDS_TOOL: 1}}
        self.assertIsNone(_chosen(state))

    def test_a_tool_the_budget_withheld_is_never_required(self):
        """The one shape the provider rejects outright is a required call with nothing to
        call. The budget composes OUTSIDE this middleware, so a withheld tool has already
        left `request.tools` by the time it is asked for."""
        state = {"messages": [HumanMessage(content="q")]}
        self.assertIsNone(_chosen(state, tools=(KNOWLEDGE_TOOL,)))

    def test_a_choice_somebody_else_made_is_left_alone(self):
        state = {"messages": [HumanMessage(content="q")]}
        self.assertEqual(_chosen(state, tool_choice="any"), "any")

    def test_a_turn_that_planned_no_tool_is_left_alone(self):
        state = {"messages": [HumanMessage(content="q")]}
        self.assertIsNone(_chosen(state, forced=""))

    def test_the_name_is_sent_bare(self):
        """`ChatOpenAI.bind_tools` turns a name it recognises among the bound tools into
        the provider's `{"type": "function", ...}` shape itself. Sending the dict here
        would be building it twice, in two places that could disagree."""
        state = {"messages": [HumanMessage(content="q")]}
        self.assertIsInstance(_chosen(state), str)


class RecordsAreCheckedToo(unittest.TestCase):
    """The narrowing and the grounding check are one change, not two.

    `_grounding_expected` asks whether any BOUND tool is checked. Under the old single
    set, narrowing a records turn to `[get_student_records]` removed the last checked
    tool from the turn — so the narrowing itself would have switched the check off,
    precisely on the turns that most needed it.
    """

    def test_a_records_only_turn_is_still_checked(self):
        from backend.chat import service

        self.assertTrue(service._grounding_expected(
            _plan(_settled(), about_child=True, child_question_kind="records")
        ))

    def test_a_turn_that_bound_nothing_is_not_checked(self):
        from backend.chat import service

        self.assertFalse(service._grounding_expected(
            _plan(_settled(), is_social=True)
        ))

    def test_the_citation_set_stays_a_subset_of_the_checked_set(self):
        from backend.tools import CHECKED_TOOLS, GROUNDED_TOOLS, TOOL_BUILDERS

        self.assertTrue(GROUNDED_TOOLS <= CHECKED_TOOLS)
        self.assertEqual(set(), CHECKED_TOOLS - set(TOOL_BUILDERS))

    def test_records_text_grounds_a_figure_no_chunk_holds(self):
        report = verify("الرسوم 45000 جنيه", [], extra_evidence=["الرسوم 45000 جنيه"])
        self.assertTrue(report.ok)

    def test_a_figure_in_neither_is_still_caught(self):
        report = verify("الرسوم 45000 جنيه", [], extra_evidence=["الرياضيات 87.5%"])
        self.assertFalse(report.ok)
        self.assertEqual(report.ungrounded, (45000.0,))

    def test_records_text_never_makes_a_citation_valid(self):
        """A child's marks carry no `[n]`. Counting them as chunks would make `[1]` valid
        on a turn that retrieved nothing to cite — which is the fault the citation check
        exists to catch."""
        report = verify("لا يوجد رقم هنا [1]", [], extra_evidence=["الرياضيات 87.5%"])
        self.assertFalse(report.ok)
        self.assertTrue(report.cited_without_evidence)
        self.assertEqual(report.evidence_count, 0)


class DenyingWhatTheToolReturned(unittest.TestCase):
    """The failure the numeric check cannot see.

    A denial states no figure, so there is nothing for grounding to verify — and a mark
    is under `answer_grounding_number_floor` in any case. The contradiction is only
    visible by holding what the tool RETURNED against what the answer CLAIMED.
    """

    class _Outcomes:
        def __init__(self, outcomes):
            self.tool_outcomes = list(outcomes)

    def _denies(self, outcomes, answer, phrases=("couldn't find", "ما لقيتش")):
        from backend.chat import service

        class _AgentCfg:
            records_denial_phrases = list(phrases)

        original = service._PROFILE.agent
        try:
            service._PROFILE.__dict__["agent"] = _AgentCfg()
            return service._denies_the_records(self._Outcomes(outcomes), answer)
        finally:
            service._PROFILE.__dict__["agent"] = original

    def test_a_denial_after_a_successful_lookup_is_caught(self):
        self.assertTrue(self._denies(
            [("get_student_records", "grades")], "I couldn't find any records for her."
        ))

    def test_the_arabic_wording_is_caught_through_folding(self):
        self.assertTrue(self._denies(
            [("get_student_records", "grades")], "ما لقيتش أي معلومات عن درجات ليلى أحمد"
        ))

    def test_a_denial_after_a_failed_lookup_is_the_correct_answer(self):
        """`no_records`, an outage and a refusal all SHOULD produce a reply saying so."""
        for outcome in ("no_records", "unavailable", "not_authorized", "which_student"):
            with self.subTest(outcome=outcome):
                self.assertFalse(self._denies(
                    [("get_student_records", outcome)], "I couldn't find any records."
                ))

    def test_an_answer_that_reports_the_marks_is_not_flagged(self):
        self.assertFalse(self._denies(
            [("get_student_records", "grades")], "ليلى حاصلة على 87.5% في الرياضيات"
        ))

    def test_a_deployment_with_no_phrases_configured_never_fires(self):
        """The phrase list is the guessing half of this check, so an empty one has to
        mean 'do not guess' rather than 'match everything'."""
        self.assertFalse(self._denies(
            [("get_student_records", "grades")], "I couldn't find any records.",
            phrases=(),
        ))


if __name__ == "__main__":
    unittest.main()


SARA = ChildOption(student_id="S-3", label="سارة أحمد", gender="female")
ALI = ChildOption(student_id="S-4", label="علي أحمد", gender="male")
AHMED = ChildOption(student_id="S-5", label="أحمد أحمد", gender="male")


class TheGenderNarrowsTheRoster(unittest.TestCase):
    """"give me the results of my son", against a roster the school already gave us.

    No model reads the roster and no model picks the child. The classifier's only job is
    to report that the message said "son"; everything after that is a filter over a list
    of real children and a length check. The two cases below are the whole feature:
    """

    def test_one_son_among_two_children_is_answered_without_asking(self):
        """A son and a daughter. "my son" leaves exactly one candidate, so the parent is
        never asked a question whose answer is already on file."""
        plan = _plan(resolve_child(reference="son", roster=[ALI, SARA]),
                     about_child=True, child_question_kind="records")

        self.assertEqual(plan.child_hint, "علي أحمد")
        self.assertEqual(plan.child_id, "S-4")
        self.assertEqual(plan.child_options, [])
        self.assertFalse(plan.short_circuit)
        self.assertEqual(plan.exposed_tools, [RECORDS_TOOL])

    def test_two_sons_are_offered_as_a_choice_and_the_daughter_is_not(self):
        """Asking "Ali, Ahmed or Sara?" after the parent said "my son" ignores what they
        just told us. The options are the candidates the filter left, never the family."""
        plan = _plan(resolve_child(reference="son", roster=[ALI, AHMED, SARA]),
                     about_child=True, child_question_kind="records")

        self.assertEqual(plan.child_options, ["علي أحمد", "أحمد أحمد"])
        self.assertNotIn("سارة أحمد", plan.child_options)
        self.assertTrue(plan.short_circuit)

    def test_a_daughter_among_sons_is_answered_without_asking(self):
        plan = _plan(resolve_child(reference="daughter", roster=[ALI, AHMED, SARA]),
                     about_child=True, child_question_kind="records")
        self.assertEqual(plan.child_hint, "سارة أحمد")

    def test_an_only_child_is_never_asked_about_at_all(self):
        plan = _plan(resolve_child(reference="context", roster=[ALI]),
                     about_child=True, child_question_kind="records")
        self.assertEqual(plan.child_hint, "علي أحمد")

    def test_a_child_whose_sex_the_school_never_recorded_stays_a_candidate(self):
        """`unknown` matches both, so a half-filled column can never select a child by
        virtue of a blank cell — the state every child is in until a registrar fills it."""
        blank = ChildOption(student_id="S-6", label="نور أحمد")
        plan = _plan(resolve_child(reference="son", roster=[ALI, blank]),
                     about_child=True, child_question_kind="records")
        self.assertEqual(plan.child_options, ["علي أحمد", "نور أحمد"])

    def test_saying_son_when_no_child_could_be_one_asks_rather_than_refusing(self):
        """The parent's wording is better evidence than the column, so this asks instead
        of declaring them wrong about their own family."""
        nour = ChildOption(student_id="S-7", label="نور أحمد", gender="female")
        plan = _plan(resolve_child(reference="son", roster=[SARA, nour]),
                     about_child=True, child_question_kind="records")
        self.assertEqual(plan.child_options, ["سارة أحمد", "نور أحمد"])

    def test_an_only_child_is_answered_even_when_the_wording_slips(self):
        """One child on file and the parent says "my son" about a daughter. The roster
        route that fires first is deliberate: with nothing to disambiguate, asking would
        be pedantry about a family the parent knows better than the SIS column does."""
        plan = _plan(resolve_child(reference="son", roster=[SARA]),
                     about_child=True, child_question_kind="records")
        self.assertEqual(plan.child_hint, "سارة أحمد")
        self.assertEqual(plan.child_options, [])

    def test_the_question_is_the_profiles_own_copy_in_the_turns_language(self):
        plan = _plan(resolve_child(reference="son", roster=[ALI, AHMED]),
                     about_child=True, child_question_kind="records")
        self.assertEqual(plan.static_reply, "Which child do you mean?")

    def test_no_copy_configured_falls_through_to_the_agent(self):
        """Refusing with an empty string is worse than answering — the rule
        `_plan_out_of_domain` already follows."""
        class _NoCopy:
            social = None
            out_of_domain = None
            which_child = None

        signals = RequestSignals(question="q", about_child=True,
                                 child_question_kind="records")
        plan = resolve_turn(signals, agent_config=_Agent(), copy_config=_NoCopy(),
                            child=resolve_child(reference="son", roster=[ALI, AHMED]))
        self.assertFalse(plan.short_circuit)
        self.assertIsNone(plan.exposed_tools)


class TheChoiceComesBackAsAnAnswer(unittest.TestCase):
    """The second half: what the parent taps becomes a pinned child, not a search term."""

    def _pending(self):
        from backend.chat.service import _child_choice_pending

        plan = _plan(resolve_child(reference="son", roster=[ALI, AHMED]),
                     about_child=True, child_question_kind="records")
        return _child_choice_pending(plan, "نتيجة ابني ايه؟")

    def test_the_question_is_stored_as_a_real_clarification(self):
        pending = self._pending()
        self.assertEqual(pending["route"], "child_select")
        self.assertEqual(pending["retrieval_status"], "needs_child_choice")
        self.assertEqual(pending["options"], ["علي أحمد", "أحمد أحمد"])

    def test_it_carries_the_original_question_and_no_resume_state(self):
        """The next message is a name and nothing else, so the question it answers has to
        travel with it. There is no half-finished search to resume."""
        pending = self._pending()
        self.assertEqual(pending["original_question"], "نتيجة ابني ايه؟")
        self.assertIsNone(pending["resume_state"])

    def test_a_plan_that_settled_the_child_asks_nothing(self):
        from backend.chat.service import _child_choice_pending

        plan = _plan(resolve_child(reference="son", roster=[ALI, SARA]),
                     about_child=True, child_question_kind="records")
        self.assertIsNone(_child_choice_pending(plan, "q"))

    def test_the_reply_reopens_the_original_question_rather_than_searching_for_a_name(self):
        """Folding "علي" into the query the way the retrieval clarifications do would
        search for a child's name instead of for what the parent actually asked."""
        from backend.chat.service import _enter_turn

        entry = _enter_turn("علي", [], {"pending_hitl": self._pending()})

        self.assertEqual(entry.child_choice, "علي")
        self.assertEqual(entry.effective_user_text, "نتيجة ابني ايه؟")
        self.assertFalse(entry.is_hitl_resume)

    def test_the_reply_costs_no_model_call(self):
        """Matching a name to a roster row is not a judgement, so nothing here may reach
        a model — a resolver call on this path would be paying to re-derive a fact."""
        from unittest.mock import patch

        import backend.chat.service as service

        with patch.object(service, "resolve_turn_question") as resolver:
            service._enter_turn("علي", [], {"pending_hitl": self._pending()})
        resolver.assert_not_called()
