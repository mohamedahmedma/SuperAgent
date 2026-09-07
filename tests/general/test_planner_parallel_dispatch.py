"""The planner decides the tool SET; the graph runs it concurrently.

`test_planner_tool_selection.py` covers the one-tool case: the plan picks a tool and
`tool_choice` requires it. This file is about the case that one deliberately leaves
alone — a question that needs several tools — and it is a different mechanism, because
`tool_choice` names one function and cannot express "each of these". The only way to
require a set is to write the calls and dispatch them.

What is asserted here, and why each is a way the architecture quietly stops working:

  * THE PLAN IS A SET OF CALLS, not a set of names. A name alone still leaves the model
    to compose the call and the loop to discover it one at a time, which is the cost
    being removed. Arguments come from a closed placeholder table, so a typo in a
    profile has to fail into the old behaviour rather than into a tool called with the
    literal text `$child_labell`.

  * SELECTION GENERALISES PAST TWO TOOLS. `child_question_kind` is an enum over exactly
    two; `agent.tool_selection` plus the classifier's `needed_tools` is the same decision
    for ten. Both have to land on "bind everything" when they fail.

  * THE TOOLS ACTUALLY OVERLAP. Proved with a barrier rather than a stopwatch: each tool
    waits for the other to arrive, so a sequential dispatch deadlocks and fails the test
    instead of passing slowly on a fast machine. Asserted on BOTH the sync and the async
    path, because `backend/chat/service.py` streams every real turn through `astream`
    and the two go through different `ToolNode` internals.

  * THE FOUR THINGS THAT BITE, each with its own class below: the planner is not given
    the last word, the seeded calls are counted against their budgets, the terminal
    guard still reaches the right verdict, and the two hooks that can jump have a
    defined precedence.
"""
import asyncio
import threading
import unittest

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from backend.chat import runtime
from backend.chat.child_resolution import no_child, resolve_child
from backend.chat.child_roster import ChildOption
from backend.chat.request_context import ChatRequestContext
from backend.chat.signals import EnvelopeDetector, RequestSignals
from backend.chat.turn_policy import (
    ATTENDANCE_TOOL,
    GRADES_TOOL,
    KNOWLEDGE_TOOL,
    RECORDS_TOOLS,
    resolve_turn,
)

#: The record tool most of these tests use. One per record now, so a test that needs
#: "a records tool" names one rather than pretending there is only one.
RECORDS_TOOL = GRADES_TOOL

LAYLA = ChildOption(student_id="S-1", label="ليلى أحمد", gender="female", year_level="Year 4")
OMAR = ChildOption(student_id="S-2", label="عمر أحمد", gender="male")


# --------------------------------------------------------------------------------------
# Fixtures. Spelled out rather than loaded from yaml, so a test can turn one knob at a
# time; `test_school_profile.py` is where the deployment's own settings are asserted.
# --------------------------------------------------------------------------------------


class _Agent:
    """A profile that plans and dispatches, in the school profile's shape."""

    tools = [KNOWLEDGE_TOOL, GRADES_TOOL, ATTENDANCE_TOOL]
    social_phrases = []
    social_reply_mode = "model"
    narrow_tools_to_the_turn = True
    parallel_tool_calls = True
    year_reference_markers = ()
    tool_selection = {}
    planned_tool_arguments = {
        KNOWLEDGE_TOOL: {"query": "$resolved_question"},
        GRADES_TOOL: {"student_name": "$child_label"},
        ATTENDANCE_TOOL: {"student_name": "$child_label"},
    }


class _Copy:
    social = None
    out_of_domain = None
    which_child = "Which child do you mean?"


def _plan(child=None, *, agent=None, question="q", **signal_kwargs):
    signals = RequestSignals(question=question, **signal_kwargs)
    return resolve_turn(
        signals, agent_config=agent or _Agent(), copy_config=_Copy(), child=child
    )


def _settled(roster=(LAYLA,)):
    return resolve_child(reference="context", roster=list(roster))


def _both(**kwargs):
    """The turn this feature exists for: one child, and a classifier naming each tool.

    Named tools, not `child_question_kind="both"`. Only a source that names tools ONE BY
    ONE may be dispatched — see `_needed_tools`. The enum names a family, which is a
    statement about where to look rather than about what to call.
    """
    kwargs.setdefault("needed_tools", [KNOWLEDGE_TOOL, GRADES_TOOL])
    return _plan(_settled(), about_child=True, **kwargs)


# --------------------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------------------


class ThePlanIsASetOfCalls(unittest.TestCase):
    def test_a_question_needing_each_tool_plans_a_call_to_each(self):
        plan = _both()
        self.assertEqual(plan.exposed_tools, [KNOWLEDGE_TOOL, RECORDS_TOOL])
        self.assertEqual(
            [call["name"] for call in plan.planned_calls],
            [KNOWLEDGE_TOOL, RECORDS_TOOL],
        )

    def test_the_calls_carry_arguments_resolved_from_the_plan(self):
        """A name alone would still leave the model to compose the call."""
        plan = _both(resolved_question="درجات ليلى كام والمصاريف كام؟")
        by_name = {call["name"]: call["args"] for call in plan.planned_calls}
        self.assertEqual(by_name[KNOWLEDGE_TOOL], {"query": "درجات ليلى كام والمصاريف كام؟"})
        # The label the ROSTER matched, not a name a model transcribed.
        self.assertEqual(
            by_name[RECORDS_TOOL], {"student_name": "ليلى أحمد"}
        )

    def test_nothing_is_forced_when_a_set_is_dispatched(self):
        """`tool_choice` names one function. Setting it beside a dispatched set would
        require one of the tools whose answer is already in hand."""
        self.assertEqual(_both().forced_tool, "")

    def test_one_tool_is_still_forced_rather_than_dispatched(self):
        plan = _plan(_settled(), about_child=True, needed_tools=[RECORDS_TOOL])
        self.assertEqual(plan.exposed_tools, [RECORDS_TOOL])
        self.assertEqual(plan.forced_tool, RECORDS_TOOL)
        self.assertEqual(plan.planned_calls, [])

    def test_a_planned_call_never_names_a_tool_the_profile_does_not_bind(self):
        """A plan naming an unbound tool reaches the graph as a call for a tool that does
        not exist, which the provider rejects for the whole turn."""
        class _KnowledgeOnly(_Agent):
            tools = [KNOWLEDGE_TOOL]

        plan = _both(agent=_KnowledgeOnly())  # the plan names a tool it does not bind
        self.assertEqual(plan.exposed_tools, [KNOWLEDGE_TOOL])
        # One surviving call is not worth a dispatch — see `_plan_parallel_calls`.
        self.assertEqual(plan.planned_calls, [])

    def test_the_trace_reports_the_names_and_never_the_arguments(self):
        """This trace is persisted per message and streamed to the browser, and a planned
        call's arguments carry the child's name and the question itself."""
        trace = _both(resolved_question="درجات ليلى").as_trace()
        self.assertEqual(trace["turn_planned_calls"], [KNOWLEDGE_TOOL, RECORDS_TOOL])
        self.assertNotIn("ليلى", repr(trace["turn_planned_calls"]))


class EveryFailureFallsBackToTheOrdinaryLoop(unittest.TestCase):
    """Dispatching is an optimisation, so it may only ever act on a positive answer."""

    def test_the_switch_off_reproduces_todays_behaviour_exactly(self):
        """The tools are still narrowed to what the turn needs — that is selection, and
        it predates this feature. What the switch controls is whether they are CALLED."""
        class _Off(_Agent):
            parallel_tool_calls = False

        plan = _both(agent=_Off())
        self.assertEqual(plan.exposed_tools, [KNOWLEDGE_TOOL, GRADES_TOOL])
        self.assertEqual(plan.planned_calls, [])

    def test_a_profile_declaring_no_arguments_plans_nothing(self):
        class _NoArgs(_Agent):
            planned_tool_arguments = {}

        plan = _both(agent=_NoArgs())
        self.assertEqual(plan.planned_calls, [])

    def test_a_tool_with_no_argument_template_is_left_to_the_model(self):
        """What makes this adoptable one tool at a time: an undeclared tool stays bound
        and behaves exactly as it did before."""
        class _KnowledgeOnlyArgs(_Agent):
            planned_tool_arguments = {KNOWLEDGE_TOOL: {"query": "$resolved_question"}}

        plan = _both(agent=_KnowledgeOnlyArgs())
        self.assertEqual(plan.exposed_tools, [KNOWLEDGE_TOOL, GRADES_TOOL])
        self.assertEqual(plan.planned_calls, [])

    def test_an_unknown_placeholder_drops_its_argument_rather_than_sending_the_text(self):
        """A typo in a profile must not become a child's name."""
        class _Typo(_Agent):
            planned_tool_arguments = {
                KNOWLEDGE_TOOL: {"query": "$resolved_question"},
                RECORDS_TOOL: {"student_name": "$child_labell", "record_type": "grades"},
            }

        plan = _both(agent=_Typo(), resolved_question="q")
        by_name = {call["name"]: call["args"] for call in plan.planned_calls}
        self.assertEqual(by_name[RECORDS_TOOL], {"record_type": "grades"})

    def test_a_placeholder_resolving_to_nothing_drops_its_argument(self):
        """An empty student name means "the parent did not say which", which the records
        tool handles better from its own default than from an explicit blank."""
        # `needed_tools` reaches the plan without a resolved child, so `$child_label` is
        # empty and the records call has nothing left in it.
        plan = _plan(no_child("n/a"), needed_tools=[KNOWLEDGE_TOOL, RECORDS_TOOL],
                     resolved_question="q")
        self.assertEqual(plan.planned_calls, [])

    def test_a_lone_surviving_call_is_not_dispatched(self):
        """One call ahead of the model buys no concurrency and still spends the planner's
        credibility on the guess."""
        plan = _plan(no_child("n/a"), needed_tools=[KNOWLEDGE_TOOL, RECORDS_TOOL],
                     resolved_question="q")
        self.assertEqual(plan.planned_calls, [])

    def test_an_unsettled_child_is_asked_about_rather_than_dispatched_for(self):
        plan = _plan(resolve_child(reference="child", roster=[LAYLA, OMAR]),
                     about_child=True, needed_tools=[KNOWLEDGE_TOOL, GRADES_TOOL])
        self.assertEqual(plan.planned_calls, [])
        self.assertTrue(plan.short_circuit)

    def test_the_question_kind_enum_narrows_but_never_dispatches(self):
        """The line the architecture turns on. `records` names a FAMILY — it cannot tell
        marks from absences — so it binds those tools and lets the model choose. Treating
        it as a plan would read a child's attendance because they asked about maths."""
        plan = _plan(_settled(), about_child=True, child_question_kind="records")
        self.assertEqual(plan.exposed_tools, [GRADES_TOOL, ATTENDANCE_TOOL])
        self.assertEqual(plan.planned_calls, [])
        self.assertEqual(plan.forced_tool, "")

    def test_both_narrows_nothing_at_all(self):
        """`both` is absent from the map on purpose: binding everything is not a
        decision, and recording it as one would only make the trace lie."""
        plan = _plan(_settled(), about_child=True, child_question_kind="both")
        self.assertIsNone(plan.exposed_tools)
        self.assertEqual(plan.planned_calls, [])

    def test_a_literal_argument_is_never_read_as_a_placeholder(self):
        class _Literal(_Agent):
            planned_tool_arguments = {
                KNOWLEDGE_TOOL: {"query": "$resolved_question"},
                RECORDS_TOOL: {"student_name": "everyone"},
            }

        plan = _both(agent=_Literal(), resolved_question="q")
        by_name = {call["name"]: call["args"] for call in plan.planned_calls}
        self.assertEqual(by_name[RECORDS_TOOL], {"student_name": "everyone"})


class SelectionGeneralisesPastTwoTools(unittest.TestCase):
    """`child_question_kind` is an enum over two tools. Ten needs a list."""

    def test_the_records_family_is_every_record_tool(self):
        """One name per record now, so the family has to be spelled out somewhere; this
        pins the policy layer's copy against the registry's."""
        from backend.tools import RECORDS_TOOLS as REGISTERED

        self.assertEqual(set(RECORDS_TOOLS), set(REGISTERED))

    def test_the_classifier_list_chooses_three_of_ten(self):
        class _Ten(_Agent):
            tools = [f"tool_{i}" for i in range(10)]
            planned_tool_arguments = {
                f"tool_{i}": {"query": "$resolved_question"} for i in range(10)
            }

        plan = _plan(no_child("n/a"), agent=_Ten(), resolved_question="q",
                     needed_tools=["tool_1", "tool_4", "tool_7"])
        self.assertEqual(plan.exposed_tools, ["tool_1", "tool_4", "tool_7"])
        self.assertEqual(
            [call["name"] for call in plan.planned_calls],
            ["tool_1", "tool_4", "tool_7"],
        )

    def test_the_classifier_list_needs_no_resolved_child(self):
        """It is a fact about the MESSAGE, so it is just as meaningful on a deployment
        that has no children at all — unlike `child_question_kind`, which is only
        meaningful once a real child has been resolved against a roster."""
        plan = _plan(no_child("n/a"), needed_tools=[KNOWLEDGE_TOOL, RECORDS_TOOL],
                     resolved_question="q")
        self.assertEqual(plan.exposed_tools, [KNOWLEDGE_TOOL, RECORDS_TOOL])

    def test_the_classifier_list_refines_the_question_kind_enum(self):
        """Where the two agree, the finer answer wins: `records` says which family, the
        list says which member, and the member is what gets dispatched."""
        plan = _plan(_settled(), about_child=True, child_question_kind="records",
                     needed_tools=[ATTENDANCE_TOOL], resolved_question="q")
        self.assertEqual(plan.exposed_tools, [ATTENDANCE_TOOL])
        self.assertEqual(plan.forced_tool, ATTENDANCE_TOOL)

    def test_a_list_contradicting_the_enum_is_not_a_decision(self):
        """The two fields come from one call and are independent readings of one message,
        so they can disagree — measured at about one turn in four on a dialect phrasing,
        where `records` was right every time and the list named the corpus twice in eight.

        That is the failure the whole mechanism exists to stop: a named child's marks sent
        to a fee corpus, answered with "no information about your daughter". So a
        contradiction falls back to the family, which is the coarser, measured and safer
        of the two readings.
        """
        plan = _plan(_settled(), about_child=True, child_question_kind="records",
                     needed_tools=[KNOWLEDGE_TOOL], resolved_question="q")
        # The family, intersected with what this profile actually binds — `_Agent` ships
        # no subject tool, and a plan may never name a tool `build_tools` would reject.
        self.assertEqual(
            plan.exposed_tools, [t for t in _Agent.tools if t in RECORDS_TOOLS]
        )
        self.assertNotIn(KNOWLEDGE_TOOL, plan.exposed_tools)
        self.assertEqual(plan.planned_calls, [])

    def test_both_spans_the_families_so_nothing_contradicts_it(self):
        """`both` names no family, so a list spanning each side is not caught by the
        contradiction rule — which is the case the parallel dispatch exists for."""
        plan = _plan(_settled(), about_child=True, child_question_kind="both",
                     needed_tools=[KNOWLEDGE_TOOL, GRADES_TOOL], resolved_question="q")
        self.assertEqual(plan.exposed_tools, [KNOWLEDGE_TOOL, GRADES_TOOL])
        self.assertEqual(len(plan.planned_calls), 2)

    def test_a_name_the_profile_does_not_bind_is_dropped(self):
        plan = _plan(no_child("n/a"), needed_tools=[KNOWLEDGE_TOOL, "invented_tool"],
                     resolved_question="q")
        self.assertEqual(plan.exposed_tools, [KNOWLEDGE_TOOL])

    def test_selection_naming_nothing_bindable_leaves_everything_bound(self):
        plan = _plan(no_child("n/a"), needed_tools=["invented_tool"], resolved_question="q")
        self.assertIsNone(plan.exposed_tools)


class TheClassifierReadsItsOwnCatalogue(unittest.TestCase):
    """`EnvelopeDetector._read_needed_tools`: what survives from the model's answer."""

    class _Config:
        tool_selection = {KNOWLEDGE_TOOL: "school material", GRADES_TOOL: "one child"}

    def _read(self, named, config=None):
        signals = RequestSignals(question="q")
        EnvelopeDetector._read_needed_tools(
            {"needed_tools": named}, signals, config or self._Config()
        )
        return signals.needed_tools

    def test_a_deployment_with_no_catalogue_never_selects(self):
        class _NoCatalogue:
            tool_selection = {}

        self.assertEqual(self._read([KNOWLEDGE_TOOL], _NoCatalogue()), [])

    def test_names_outside_the_catalogue_are_dropped(self):
        """Filtered against what the node was SHOWN, which is a tighter guard than what
        the profile happens to bind: a name outside it was never on offer."""
        self.assertEqual(self._read([KNOWLEDGE_TOOL, "delete_everything"]), [KNOWLEDGE_TOOL])

    def test_the_order_is_the_catalogues_and_not_the_models(self):
        """Two plans naming the same tools must be the same plan, or an identical
        question produces a different `exposed_tools` on every turn."""
        self.assertEqual(
            self._read([GRADES_TOOL, KNOWLEDGE_TOOL]), [KNOWLEDGE_TOOL, GRADES_TOOL]
        )

    def test_anything_that_is_not_a_list_is_an_abstention(self):
        for answer in (None, "", "search_knowledge_base", 7, {}):
            with self.subTest(answer=answer):
                self.assertEqual(self._read(answer), [])

    def test_the_signal_reaches_the_trace(self):
        signals = RequestSignals(question="q")
        EnvelopeDetector._read_needed_tools(
            {"needed_tools": [KNOWLEDGE_TOOL]}, signals, self._Config()
        )
        self.assertEqual(signals.as_trace()["request_needed_tools"], [KNOWLEDGE_TOOL])


# --------------------------------------------------------------------------------------
# The dispatch arithmetic
# --------------------------------------------------------------------------------------


class TheDispatchArithmetic(unittest.TestCase):
    """`planned_tool_calls`, without a graph."""

    def _calls(self, planned, made=None):
        return runtime.planned_tool_calls(planned, made or {})

    def test_every_call_gets_an_id_because_results_are_matched_by_it(self):
        calls = self._calls([
            {"name": KNOWLEDGE_TOOL, "args": {"query": "q"}},
            {"name": RECORDS_TOOL, "args": {"student_name": "ليلى"}},
        ])
        ids = [call["id"] for call in calls]
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(all(ids))

    def test_the_same_plan_produces_the_same_ids_twice(self):
        planned = [{"name": KNOWLEDGE_TOOL, "args": {"query": "q"}},
                   {"name": RECORDS_TOOL, "args": {"student_name": "ليلى"}}]
        self.assertEqual(
            [c["id"] for c in self._calls(planned)],
            [c["id"] for c in self._calls(planned)],
        )

    def test_a_repeated_call_is_collapsed_by_the_rule_that_already_exists(self):
        calls = self._calls([
            {"name": KNOWLEDGE_TOOL, "args": {"query": "q"}},
            {"name": KNOWLEDGE_TOOL, "args": {"query": "q"}},
        ])
        self.assertEqual(len(calls), 1)

    def test_a_call_past_its_budget_is_dropped(self):
        """The planner is not exempt from the deployment's own ceiling — a planner that
        could overspend it would be a second, invisible budget."""
        spent = {KNOWLEDGE_TOOL: runtime.budget_for(KNOWLEDGE_TOOL)}
        calls = self._calls(
            [{"name": KNOWLEDGE_TOOL, "args": {"query": "q"}},
             {"name": RECORDS_TOOL, "args": {"student_name": "ليلى"}}],
            made=spent,
        )
        self.assertEqual([call["name"] for call in calls], [RECORDS_TOOL])

    def test_a_malformed_plan_entry_is_skipped_rather_than_dispatched(self):
        calls = self._calls([
            {"name": "", "args": {"query": "q"}},
            {"name": KNOWLEDGE_TOOL, "args": "not a dict"},
            {"name": KNOWLEDGE_TOOL, "args": {"query": "q"}},
        ])
        self.assertEqual([call["name"] for call in calls], [KNOWLEDGE_TOOL])


# --------------------------------------------------------------------------------------
# The middleware, in a real graph
# --------------------------------------------------------------------------------------


def _scripted(*messages):
    """A model that says exactly what it is told to, and counts how often it was asked."""
    calls = {"n": 0}
    script = iter(messages)

    class Scripted(GenericFakeChatModel):
        def _generate(self, msgs, stop=None, run_manager=None, **kw):
            calls["n"] += 1
            return ChatResult(generations=[ChatGeneration(message=next(script))])

        def bind_tools(self, *args, **kwargs):
            return self

    return Scripted(messages=iter([])), calls


class _Meeting:
    """Two tools that each refuse to finish until the other has started.

    A barrier rather than a stopwatch. If the dispatch is sequential the first tool waits
    for a partner that cannot arrive until it returns, and the test fails on the timeout
    instead of passing slowly on a fast machine or flaking on a loaded one.
    """

    def __init__(self, timeout=5.0):
        self.barrier = threading.Barrier(2, timeout=timeout)
        self.overlapped = False

    def tools(self):
        @tool(KNOWLEDGE_TOOL)
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base."""
            self._meet()
            return "Retrieved Chunks:\n[1] fees.pdf (Page 3):\n30,000 جنيه."

        @tool(RECORDS_TOOL)
        def get_student_grades(student_name: str = "") -> str:
            """Read one child's marks."""
            self._meet()
            return "الرياضيات 87.5%"

        return [search_knowledge_base, get_student_grades]

    def _meet(self):
        try:
            self.barrier.wait()
            self.overlapped = True
        except threading.BrokenBarrierError:
            pass


class _Ctx(ChatRequestContext):
    """A real context, so the middleware's accounting is the real accounting."""


class _BoundTool:
    """A bound tool as `_tool_name` sees it: something with a name."""

    def __init__(self, name):
        self.name = name


class _Request:
    """A request-shaped object for asserting on what the budget middleware offers."""

    def __init__(self, state, tools=(KNOWLEDGE_TOOL, RECORDS_TOOL)):
        self.state = state
        self.tools = [_BoundTool(name) for name in tools]
        self.tool_choice = None

    def override(self, **overrides):
        replaced = _Request(self.state, ())
        replaced.tools = overrides.get("tools", self.tools)
        replaced.tool_choice = overrides.get("tool_choice", self.tool_choice)
        return replaced


def _ctx(planned):
    ctx = _Ctx(user_id="u", session_id="s")
    ctx.planned_calls = planned
    return ctx


PLANNED = [
    {"name": KNOWLEDGE_TOOL, "args": {"query": "المصاريف"}},
    {"name": RECORDS_TOOL, "args": {"student_name": "ليلى أحمد"}},
]


def _agent(ctx, tools, model):
    return create_agent(
        model=model,
        tools=tools,
        middleware=[
            runtime._dispatch_planned_tools(ctx),
            runtime._collapse_duplicate_tool_calls(ctx),
            runtime._spend_tool_budgets(ctx),
            runtime._force_the_planned_tool(ctx),
        ],
    )


class TheToolsActuallyOverlap(unittest.TestCase):
    """The claim the whole feature rests on."""

    def test_the_planned_tools_run_at_the_same_time(self):
        meeting = _Meeting()
        ctx = _ctx(PLANNED)
        model, calls = _scripted(AIMessage(content="30,000 جنيه [1]، والرياضيات 87.5%"))
        out = _agent(ctx, meeting.tools(), model).invoke(
            {"messages": [HumanMessage(content="درجات ليلى كام والمصاريف كام؟")]}
        )

        self.assertTrue(meeting.overlapped, "the two tools did not run concurrently")
        results = [m for m in out["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(results), 2)

    def test_the_streamed_path_overlaps_too(self):
        """`backend/chat/service.py` streams every real turn through `astream`, which
        composes different `ToolNode` internals from the sync path."""
        meeting = _Meeting()
        ctx = _ctx(PLANNED)
        model, calls = _scripted(AIMessage(content="30,000 جنيه [1]"))
        agent = _agent(ctx, meeting.tools(), model)
        out = asyncio.run(
            agent.ainvoke({"messages": [HumanMessage(content="درجات ليلى والمصاريف")]})
        )

        self.assertTrue(meeting.overlapped, "the two tools did not run concurrently")
        self.assertEqual(
            len([m for m in out["messages"] if isinstance(m, ToolMessage)]), 2
        )

    def test_the_whole_turn_costs_one_model_call(self):
        """The larger saving. Discovered one at a time these two tools cost three calls —
        one to ask for each, one to answer."""
        meeting = _Meeting()
        ctx = _ctx(PLANNED)
        model, calls = _scripted(AIMessage(content="done"))
        _agent(ctx, meeting.tools(), model).invoke(
            {"messages": [HumanMessage(content="q")]}
        )
        self.assertEqual(calls["n"], 1)

    def test_the_planned_arguments_are_what_the_tools_receive(self):
        seen = {}

        @tool(KNOWLEDGE_TOOL)
        def search_knowledge_base(query: str) -> str:
            """Search."""
            seen["query"] = query
            return "[1] ok"

        @tool(RECORDS_TOOL)
        def get_student_grades(student_name: str = "") -> str:
            """Read."""
            seen["student_name"] = student_name
            return "ok"

        ctx = _ctx(PLANNED)
        model, _ = _scripted(AIMessage(content="done"))
        _agent(ctx, [search_knowledge_base, get_student_grades], model).invoke(
            {"messages": [HumanMessage(content="q")]}
        )
        self.assertEqual(seen["query"], "المصاريف")
        self.assertEqual(seen["student_name"], "ليلى أحمد")

    def test_a_turn_with_no_plan_runs_the_ordinary_loop(self):
        """The default, and what every failure upstream produces."""
        ran = []

        @tool(KNOWLEDGE_TOOL)
        def search_knowledge_base(query: str) -> str:
            """Search."""
            ran.append(query)
            return "[1] ok"

        ctx = _ctx([])
        model, calls = _scripted(
            AIMessage(content="", tool_calls=[
                {"name": KNOWLEDGE_TOOL, "args": {"query": "asked for"}, "id": "a"}
            ]),
            AIMessage(content="done"),
        )
        _agent(ctx, [search_knowledge_base], model).invoke(
            {"messages": [HumanMessage(content="q")]}
        )
        self.assertEqual(ran, ["asked for"])
        self.assertEqual(calls["n"], 2)


# --------------------------------------------------------------------------------------
# The four things that bite
# --------------------------------------------------------------------------------------


class ThePlannerIsNotGivenTheLastWord(unittest.TestCase):
    """Problem 1. A wrong plan must cost a round-trip, never the turn's tool use."""

    def test_the_model_can_still_call_a_tool_the_planner_already_dispatched(self):
        """How a planned query that came back empty gets corrected: the records tool's
        budget is 3, the plan spent one, so two remain."""
        queried = []

        @tool(KNOWLEDGE_TOOL)
        def search_knowledge_base(query: str) -> str:
            """Search."""
            return "[1] nothing useful"

        @tool(ATTENDANCE_TOOL)
        def get_student_attendance(student_name: str = "") -> str:
            """Read."""
            queried.append(student_name)
            return "ok"

        ctx = _ctx(PLANNED)
        model, _ = _scripted(
            # The plan read marks; the parent also wanted absences.
            AIMessage(content="", tool_calls=[{
                "name": ATTENDANCE_TOOL,
                "args": {"student_name": "ليلى أحمد"},
                "id": "fix",
            }]),
            AIMessage(content="غابت يومين"),
        )
        _agent(ctx, [search_knowledge_base, get_student_attendance], model).invoke(
            {"messages": [HumanMessage(content="q")]}
        )
        self.assertEqual(queried, ["ليلى أحمد"])

    def test_the_chosen_tools_stay_bound_after_the_dispatch(self):
        plan = _both()
        self.assertEqual(plan.exposed_tools, [c["name"] for c in plan.planned_calls])

    def test_forcing_stands_down_once_the_plan_has_answered(self):
        """A model told it MUST call a tool, on a turn whose results are already in hand,
        has nothing useful left to do with the requirement."""
        state = {"messages": [], "tool_calls_made": {KNOWLEDGE_TOOL: 1}}
        self.assertFalse(runtime._nothing_has_run_yet(state))


class TheSeededCallsAreCountedAgainstTheirBudgets(unittest.TestCase):
    """Problem 2. `_ToolBudget` counts in `after_model`, which the jump skips."""

    def test_the_dispatch_writes_the_counter_itself(self):
        ctx = _ctx(PLANNED)
        hook = runtime._dispatch_planned_tools(ctx)
        update = hook.before_model({"messages": [HumanMessage(content="q")]}, None)

        self.assertEqual(update["jump_to"], "tools")
        self.assertEqual(update["tool_calls_made"], {KNOWLEDGE_TOOL: 1, RECORDS_TOOL: 1})

    def test_a_seeded_tool_is_then_withheld_from_the_model(self):
        """The counter's whole purpose. Asserted on the request the budget middleware
        builds rather than through a scripted model, because a fake model returns its
        script whatever it was offered — so an agent-level assertion here would pass on a
        middleware that had stopped withholding anything at all.
        """
        ctx = _ctx(PLANNED)
        seeded = runtime._dispatch_planned_tools(ctx).before_model(
            {"messages": [HumanMessage(content="q")]}, None
        )
        spent = seeded["tool_calls_made"]
        # One knowledge call is the base profile's whole budget for a turn.
        self.assertGreaterEqual(spent[KNOWLEDGE_TOOL], runtime.budget_for(KNOWLEDGE_TOOL))

        budget = runtime._spend_tool_budgets(ctx)
        offered = []
        budget.wrap_model_call(
            _Request(state={"tool_calls_made": spent}), lambda r: offered.extend(r.tools)
        )
        self.assertNotIn(KNOWLEDGE_TOOL, [t.name for t in offered])

    def test_the_dispatch_is_recorded_for_the_deployment_to_watch(self):
        ctx = _ctx(PLANNED)
        runtime._dispatch_planned_tools(ctx).before_model(
            {"messages": [HumanMessage(content="q")]}, None
        )
        self.assertEqual(ctx.planned_dispatches, 2)


class TheTerminalGuardStillReachesTheRightVerdict(unittest.TestCase):
    """Problem 3. Both hooks are `before_model`, and the plan changes what the guard sees."""

    def test_an_empty_corpus_no_longer_hides_the_records_that_did_arrive(self):
        """Sequentially the search ran first and ended the turn before the records tool
        was reached, so a parent asking for each was told nothing at all — including
        about the marks, which were sitting there unread."""
        messages = [
            ToolMessage(content="no knowledge", name=KNOWLEDGE_TOOL, tool_call_id="a"),
            ToolMessage(content="الرياضيات 87.5%", name=RECORDS_TOOL, tool_call_id="b"),
        ]
        self.assertTrue(runtime._other_tools_ran(messages))

    def test_a_lone_empty_search_still_ends_the_turn(self):
        messages = [ToolMessage(content="no knowledge", name=KNOWLEDGE_TOOL, tool_call_id="a")]
        self.assertFalse(runtime._other_tools_ran(messages))


class TheTwoJumpingHooksHaveADefinedPrecedence(unittest.TestCase):
    """Problem 4. `before_model` hooks chain in list order and a jump skips the rest."""

    def test_the_dispatch_is_listed_before_the_terminal_guard(self):
        """List order is what decides precedence, so it is asserted on the list the agent
        is actually built from rather than restated from the source."""
        captured = {}

        def _spy(**kwargs):
            captured["middleware"] = kwargs["middleware"]
            return object()

        real = runtime.create_agent
        runtime.create_agent = _spy
        try:
            runtime.create_agent_for_request(_ctx(PLANNED), [KNOWLEDGE_TOOL])
        finally:
            runtime.create_agent = real

        hooks = [
            type(m).__name__ for m in captured["middleware"]
            if hasattr(type(m), "before_model")
            and type(m).before_model is not AgentMiddleware.before_model
        ]
        self.assertEqual(hooks[0], "_run_the_planned_calls_together")
        self.assertIn("_stop_after_a_terminal_tool_result", hooks)
        self.assertLess(
            hooks.index("_run_the_planned_calls_together"),
            hooks.index("_stop_after_a_terminal_tool_result"),
        )

    def test_the_dispatch_is_inert_once_a_tool_has_returned(self):
        """Which is what makes "they can never both fire" true rather than lucky."""
        ctx = _ctx(PLANNED)
        hook = runtime._dispatch_planned_tools(ctx)
        state = {
            "messages": [
                HumanMessage(content="q"),
                ToolMessage(content="ok", name=KNOWLEDGE_TOOL, tool_call_id="a"),
            ]
        }
        self.assertIsNone(hook.before_model(state, None))

    def test_the_dispatch_never_seeds_twice(self):
        ctx = _ctx(PLANNED)
        hook = runtime._dispatch_planned_tools(ctx)
        first = hook.before_model({"messages": [HumanMessage(content="q")]}, None)
        again = hook.before_model(
            {"messages": [HumanMessage(content="q")], "tool_calls_made": first["tool_calls_made"]},
            None,
        )
        self.assertIsNone(again)

    def test_a_plan_of_one_call_does_not_jump(self):
        ctx = _ctx([PLANNED[0]])
        hook = runtime._dispatch_planned_tools(ctx)
        self.assertIsNone(hook.before_model({"messages": [HumanMessage(content="q")]}, None))


class TheContextCarriesThePlanSafely(unittest.TestCase):
    def test_the_plan_is_copied_rather_than_aliased(self):
        """The middleware reads this on another thread; sharing the object would let a
        later planner edit change what a running turn is about to dispatch."""
        ctx = ChatRequestContext(user_id="u", session_id="s")
        # Built here rather than copied from `PLANNED`: this test mutates what it hands
        # over, and a shallow copy of that constant would share the nested `args` — which
        # is the very bug being asserted against, arriving as pollution of every test
        # after this one instead of as a failure here.
        planned = [{"name": KNOWLEDGE_TOOL, "args": {"query": "المصاريف"}}]
        ctx.note_turn_plan([], [], planned_calls=planned)
        planned[0]["args"]["query"] = "changed"
        self.assertEqual(ctx.planned_calls[0]["args"]["query"], "المصاريف")

    def test_a_context_written_against_the_older_signature_still_works(self):
        """`note_turn_plan` promises an integrating deployment's context keeps working;
        the orchestrator drops the newest hint first when one rejects the call."""
        ctx = ChatRequestContext(user_id="u", session_id="s")
        ctx.note_turn_plan([], [])
        self.assertEqual(ctx.planned_calls, [])


if __name__ == "__main__":
    unittest.main()
