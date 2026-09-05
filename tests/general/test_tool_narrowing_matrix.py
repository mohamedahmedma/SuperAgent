"""The tool-narrowing decision, as a closed matrix rather than a handful of examples.

`_plan_tools` is four independent conditions stacked on top of each other — the profile's
opt-in, whether a child is settled, what kind of question it is, and which tools the
profile actually binds — and each of them is allowed to say "change nothing". A test per
happy path (which `test_planner_tool_selection.py` already has) proves the cells someone
thought of. It does not prove that the cells nobody thought of still fall the safe way,
and every one of those cells is a turn where a parent either loses a tool they needed or
gains one the profile refused.

So this file enumerates the product of all four dimensions and asserts `exposed_tools`
and `forced_tool` for every cell against an expectation derived from the RULE:

    narrow only when the profile asked for it, AND a child is already settled against
    the school's own roster, AND the classifier gave a positive answer inside the closed
    set, AND the profile itself binds the tool that answer names — otherwise leave the
    plan alone (`exposed_tools is None`, "bind everything").

Alongside the matrix it pins the two invariants that make a narrowed plan safe to hand
downstream:

  * the narrowed list is always a SUBSET of the profile's own tools, in the profile's
    own order — a plan naming a tool the profile refuses is not a wrong answer, it is a
    `build_tools` startup crash, so the matrix's every output is fed to `build_tools` for
    real;
  * `forced_tool` is set only where exactly one tool is bound, and is always a member of
    `exposed_tools` — `_ForcePlannedTool` sends it to the provider as `tool_choice`, and
    a required call with nothing to call is the one shape the endpoint rejects outright.

Finally it covers the decision's observability: `as_trace()` and the orchestrator's
progress step have to describe the narrowing that happened and never one that did not,
because narrowing is invisible in the answer text and the trace is the only place a wrong
one can be seen.
"""
import unittest

from backend.chat import orchestrator
from backend.chat.child_resolution import ResolvedChild, no_child
from backend.chat.child_roster import ChildOption
from backend.chat.signals import RequestSignals, Scope
from backend.chat.turn_policy import (
    KNOWLEDGE_TOOL,
    RECORDS_TOOL,
    TurnPlan,
    _tools_for,
    resolve_turn,
)
from backend.rag.evidence import Certainty
from backend.tools import build_tools

#: A third registered tool this deployment happens to bind. It exists in this file only
#: to prove narrowing never drags in a tool the question did not ask for.
PRODUCTS_TOOL = "search_products"

LAYLA = ChildOption(student_id="S-1", label="ليلى أحمد", gender="female", year_level="Year 4")
OMAR = ChildOption(student_id="S-2", label="عمر أحمد", gender="male")


class _Agent:
    """The knobs `resolve_turn` reads, spelled out so one can be turned at a time."""

    tools = [KNOWLEDGE_TOOL, RECORDS_TOOL]
    social_phrases = []
    social_reply_mode = "model"
    narrow_tools_to_the_turn = True
    year_reference_markers = ()


class _Copy:
    social = None
    out_of_domain = "That is not something I can help with."
    which_child = "Which child do you mean?"


class _FakeCtx:
    """Enough of a `ChatRequestContext` for a tool BUILDER to close over.

    The builders registered in `backend/tools/__init__.py` only capture the context;
    nothing is called until a tool runs, and no tool runs here.
    """

    planned_child_id = ""
    planned_child_label = ""
    forced_tool = ""

    def note_tool_outcome(self, *_args, **_kwargs):  # pragma: no cover - never called
        return None


def _agent(tools, narrow=True):
    class _Configured(_Agent):
        pass

    _Configured.tools = list(tools)
    _Configured.narrow_tools_to_the_turn = narrow
    return _Configured()


def _resolved_child():
    """A child the roster settled on its own — the state narrowing requires."""
    return ResolvedChild(
        student_id=LAYLA.student_id,
        label=LAYLA.label,
        year_level=LAYLA.year_level,
        resolved=True,
        source="only_child",
        reason="one child on file",
    )


def _asking_child():
    """Two candidates and no way to choose — the state that ENDS the turn."""
    return ResolvedChild(options=(LAYLA, OMAR), ask=True, reason="two match")


CHILD_STATES = {
    "resolved": _resolved_child,
    "asking": _asking_child,
    "none": lambda: no_child("not about a child"),
}

#: Every value `child_question_kind` can carry: the two that narrow, the default that
#: means "narrow nothing", the empty string a classifier that never ran leaves, and a
#: value from outside the closed set.
KINDS = ("records", "school_matter", "both", "", "junk")

PROFILE_TOOLS = {
    "both": [KNOWLEDGE_TOOL, RECORDS_TOOL],
    "knowledge_only": [KNOWLEDGE_TOOL],
    "records_only": [RECORDS_TOOL],
    "none": [],
}

_TOOL_THE_KIND_NEEDS = {"records": RECORDS_TOOL, "school_matter": KNOWLEDGE_TOOL}


def _expected(kind, state, tools, narrow):
    """What the RULE says this cell must produce, written independently of the code.

    Returns `(exposed_tools, forced_tool)`. `None` for the first means "bind everything
    the profile allows", which is the pre-feature behaviour and the answer to every
    uncertainty.
    """
    if state == "asking":
        # The turn is over before tools are considered at all: the parent is asked which
        # child, deterministically, and nothing is bound for a lookup nobody can perform.
        return [], ""
    if state == "none" or not narrow:
        return None, ""
    wanted = _TOOL_THE_KIND_NEEDS.get(kind)
    if wanted is None:
        return None, ""
    narrowed = [name for name in tools if name == wanted]
    if not narrowed:
        # The profile does not bind the tool this kind of question needs. "Bind nothing"
        # would be a silent capability loss, so the plan is left alone.
        return None, ""
    return narrowed, narrowed[0] if len(narrowed) == 1 else ""


def _plan(child=None, *, agent=None, **signal_kwargs):
    signals = RequestSignals(question="q", **signal_kwargs)
    return resolve_turn(
        signals, agent_config=agent or _Agent(), copy_config=_Copy(), child=child
    )


def _cells():
    """Every (kind, child state, profile tools, opt-in) combination."""
    for kind in KINDS:
        for state in CHILD_STATES:
            for tools_name, tools in PROFILE_TOOLS.items():
                for narrow in (True, False):
                    yield kind, state, tools_name, tools, narrow


class TheNarrowingMatrix(unittest.TestCase):
    """Every cell of the decision, expectation derived from the rule."""

    def test_every_combination_binds_what_the_rule_says_it_binds(self):
        for kind, state, tools_name, tools, narrow in _cells():
            with self.subTest(kind=kind, child=state, profile=tools_name, narrow=narrow):
                plan = _plan(
                    CHILD_STATES[state](),
                    agent=_agent(tools, narrow),
                    about_child=state != "none",
                    child_question_kind=kind,
                )
                exposed, forced = _expected(kind, state, tools, narrow)
                self.assertEqual(plan.exposed_tools, exposed)
                self.assertEqual(plan.forced_tool, forced)

    def test_a_profile_that_never_opted_in_plans_exactly_as_it_did_before(self):
        """The whole feature is off unless a deployment measured it and asked for it.

        Catches an edit that makes narrowing the default: with the opt-in false, every
        cell of the matrix must be indistinguishable from the pre-feature planner.
        """
        for kind, state, tools_name, tools, narrow in _cells():
            if narrow:
                continue
            with self.subTest(kind=kind, child=state, profile=tools_name):
                plan = _plan(
                    CHILD_STATES[state](),
                    agent=_agent(tools, False),
                    about_child=state != "none",
                    child_question_kind=kind,
                )
                if state == "asking":
                    # Asking which child is a different rung and is not opt-in gated.
                    self.assertEqual(plan.exposed_tools, [])
                else:
                    self.assertIsNone(plan.exposed_tools)

    def test_only_a_positive_answer_inside_the_closed_set_narrows_anything(self):
        """Narrowing is never the residue of a missing classification.

        A classifier that abstained, was rate-limited, or answered outside the closed set
        leaves a kind that must bind everything — the failure mode the module's docstring
        promises.
        """
        for kind in ("both", "", "junk", "Records", "records ", "school-matter", None):
            with self.subTest(kind=kind):
                plan = _plan(
                    _resolved_child(),
                    about_child=True,
                    **({} if kind is None else {"child_question_kind": kind}),
                )
                self.assertIsNone(plan.exposed_tools)
                self.assertEqual(plan.forced_tool, "")


class TheNarrowedListIsAlwaysASubsetOfTheProfile(unittest.TestCase):
    """A plan naming a tool the profile refuses is a startup crash, not a wrong answer."""

    def test_no_cell_of_the_matrix_names_a_tool_the_profile_does_not_bind(self):
        for kind, state, tools_name, tools, narrow in _cells():
            with self.subTest(kind=kind, child=state, profile=tools_name, narrow=narrow):
                plan = _plan(
                    CHILD_STATES[state](),
                    agent=_agent(tools, narrow),
                    about_child=state != "none",
                    child_question_kind=kind,
                )
                if plan.exposed_tools is None:
                    continue
                self.assertTrue(set(plan.exposed_tools).issubset(set(tools)))

    def test_build_tools_accepts_every_list_the_matrix_produces(self):
        """The invariant above stated as the thing that actually breaks.

        `create_agent_for_request` hands the planned list straight to `build_tools`,
        which raises `UnknownToolError` on a name it does not know. A narrowing that
        invented a name would surface as a 500 on a real parent's turn, so every list the
        matrix can produce is built for real against a stand-in context.
        """
        ctx = _FakeCtx()
        for kind, state, tools_name, tools, narrow in _cells():
            plan = _plan(
                CHILD_STATES[state](),
                agent=_agent(tools, narrow),
                about_child=state != "none",
                child_question_kind=kind,
            )
            planned = plan.exposed_tools if plan.exposed_tools is not None else tools
            with self.subTest(kind=kind, child=state, profile=tools_name, narrow=narrow):
                built = build_tools(list(planned), ctx)
                self.assertEqual([tool.name for tool in built], list(planned))

    def test_an_unrelated_tool_the_profile_binds_is_never_dragged_in(self):
        """A deployment binding a catalogue tool as well must not have narrowing hand a
        records question a product search it did not ask for."""
        agent = _agent([KNOWLEDGE_TOOL, PRODUCTS_TOOL, RECORDS_TOOL])
        for kind, expected in (("records", [RECORDS_TOOL]),
                               ("school_matter", [KNOWLEDGE_TOOL])):
            with self.subTest(kind=kind):
                plan = _plan(_resolved_child(), agent=agent, about_child=True,
                             child_question_kind=kind)
                self.assertEqual(plan.exposed_tools, expected)
                self.assertNotIn(PRODUCTS_TOOL, plan.exposed_tools)

    def test_the_kept_tools_come_back_in_the_profiles_declaration_order(self):
        """`_tools_for` intersects; it must never reorder to match what was wanted.

        The tool list is also the order the system prompt describes the tools in, so two
        orderings for one list is one that can disagree. Exercised on the helper because
        no single `child_question_kind` ever wants two tools at once — the ordering rule
        would otherwise be unobservable until the day a kind does.
        """
        forwards = _agent([KNOWLEDGE_TOOL, RECORDS_TOOL])
        backwards = _agent([RECORDS_TOOL, KNOWLEDGE_TOOL])
        wanted = (KNOWLEDGE_TOOL, RECORDS_TOOL)
        self.assertEqual(_tools_for(forwards, keep=wanted), [KNOWLEDGE_TOOL, RECORDS_TOOL])
        self.assertEqual(_tools_for(backwards, keep=wanted), [RECORDS_TOOL, KNOWLEDGE_TOOL])

    def test_a_profile_with_an_unrelated_tool_only_keeps_what_was_wanted(self):
        agent = _agent([PRODUCTS_TOOL, RECORDS_TOOL, KNOWLEDGE_TOOL])
        self.assertEqual(
            _tools_for(agent, keep=(KNOWLEDGE_TOOL, RECORDS_TOOL)),
            [RECORDS_TOOL, KNOWLEDGE_TOOL],
        )

    def test_a_profile_that_declares_no_tools_at_all_narrows_to_nothing(self):
        self.assertEqual(_tools_for(_agent([]), keep=(RECORDS_TOOL,)), [])


class TheForcedToolFollowsTheBoundList(unittest.TestCase):
    """`forced_tool` becomes `tool_choice`. It may only ever name a bound tool."""

    def test_the_forced_tool_is_always_a_member_of_the_bound_list(self):
        for kind, state, tools_name, tools, narrow in _cells():
            with self.subTest(kind=kind, child=state, profile=tools_name, narrow=narrow):
                plan = _plan(
                    CHILD_STATES[state](),
                    agent=_agent(tools, narrow),
                    about_child=state != "none",
                    child_question_kind=kind,
                )
                if not plan.forced_tool:
                    continue
                self.assertIsNotNone(plan.exposed_tools)
                self.assertIn(plan.forced_tool, plan.exposed_tools)

    def test_a_tool_is_required_exactly_when_one_tool_is_bound(self):
        for kind, state, tools_name, tools, narrow in _cells():
            with self.subTest(kind=kind, child=state, profile=tools_name, narrow=narrow):
                plan = _plan(
                    CHILD_STATES[state](),
                    agent=_agent(tools, narrow),
                    about_child=state != "none",
                    child_question_kind=kind,
                )
                bound_exactly_one = (
                    plan.exposed_tools is not None and len(plan.exposed_tools) == 1
                )
                self.assertEqual(bool(plan.forced_tool), bound_exactly_one)

    def test_a_turn_that_bound_everything_requires_nothing(self):
        """`exposed_tools is None` means the profile's whole list. Forcing one of them
        would be a selection decision nobody made."""
        plan = _plan(_resolved_child(), about_child=True, child_question_kind="both")
        self.assertIsNone(plan.exposed_tools)
        self.assertEqual(plan.forced_tool, "")

    def test_a_degenerate_profile_that_names_a_tool_twice_requires_nothing(self):
        """Not a real deployment, but the rule is "exactly one", not "at least one".

        Catches an edit that reads `narrowed[0]` unconditionally and so sends
        `tool_choice` on a turn where the plan did not settle on a single tool.
        """
        plan = _plan(
            _resolved_child(),
            agent=_agent([RECORDS_TOOL, RECORDS_TOOL]),
            about_child=True,
            child_question_kind="records",
        )
        self.assertEqual(set(plan.exposed_tools), {RECORDS_TOOL})
        self.assertEqual(plan.forced_tool, "")


class NarrowingNeverResurrectsATool(unittest.TestCase):
    """The rungs above narrowing already unbound everything. It must not undo that."""

    def test_a_social_turn_binds_nothing_even_with_a_settled_child(self):
        plan = _plan(_resolved_child(), is_social=True, about_child=True,
                     child_question_kind="records")
        self.assertEqual(plan.exposed_tools, [])
        self.assertEqual(plan.forced_tool, "")

    def test_a_confirmed_out_of_domain_turn_binds_nothing_even_with_a_settled_child(self):
        """Confirmed by something that READ the question, so the turn ends here. A
        narrowing that ran afterwards would rebuild an agent for a turn already answered.
        """
        plan = _plan(
            _resolved_child(),
            about_child=True,
            child_question_kind="records",
            scope=Scope.OUT_OF_DOMAIN,
            scope_certainty=Certainty.HIGH,
        )
        self.assertTrue(plan.short_circuit)
        self.assertEqual(plan.exposed_tools, [])
        self.assertEqual(plan.forced_tool, "")

    def test_a_merely_suspected_out_of_domain_turn_still_narrows_normally(self):
        """The other side of the same rung: a weak scope guess may not end a turn, so the
        narrowing decision is unaffected by it."""
        plan = _plan(
            _resolved_child(),
            about_child=True,
            child_question_kind="records",
            scope=Scope.OUT_OF_DOMAIN,
            scope_certainty=Certainty.LOW,
        )
        self.assertFalse(plan.short_circuit)
        self.assertEqual(plan.exposed_tools, [RECORDS_TOOL])

    def test_asking_which_child_wins_over_every_narrowing_input(self):
        for kind in KINDS:
            for tools_name, tools in PROFILE_TOOLS.items():
                with self.subTest(kind=kind, profile=tools_name):
                    plan = _plan(_asking_child(), agent=_agent(tools), about_child=True,
                                 child_question_kind=kind)
                    self.assertTrue(plan.short_circuit)
                    self.assertEqual(plan.exposed_tools, [])
                    self.assertEqual(plan.forced_tool, "")
                    self.assertEqual(plan.child_options, [LAYLA.label, OMAR.label])


class TheReasonsMatchWhatHappened(unittest.TestCase):
    """`plan.reasons` is the only human-readable account of a decision nobody can see in
    the answer text. It must never claim a narrowing that did not happen, and never stay
    silent about one that did."""

    def test_every_narrowed_plan_says_which_tools_it_bound_and_no_other_plan_does(self):
        for kind, state, tools_name, tools, narrow in _cells():
            with self.subTest(kind=kind, child=state, profile=tools_name, narrow=narrow):
                plan = _plan(
                    CHILD_STATES[state](),
                    agent=_agent(tools, narrow),
                    about_child=state != "none",
                    child_question_kind=kind,
                )
                narrowed = bool(plan.exposed_tools) and not plan.short_circuit
                claims = [line for line in plan.reasons if "bound" in line]
                self.assertEqual(bool(claims), narrowed)
                if narrowed:
                    for name in plan.exposed_tools:
                        self.assertIn(name, claims[0])

    def test_a_profile_missing_the_tool_says_so_rather_than_saying_nothing(self):
        """The one abstention that is worth reading in a trace: the classifier answered,
        the child was settled, and the deployment simply does not bind that tool."""
        plan = _plan(
            _resolved_child(),
            agent=_agent([KNOWLEDGE_TOOL]),
            about_child=True,
            child_question_kind="records",
        )
        self.assertIsNone(plan.exposed_tools)
        self.assertTrue(any("no such tool" in line for line in plan.reasons))

    def test_a_plan_always_carries_at_least_one_reason(self):
        for kind, state, tools_name, tools, narrow in _cells():
            with self.subTest(kind=kind, child=state, profile=tools_name, narrow=narrow):
                plan = _plan(
                    CHILD_STATES[state](),
                    agent=_agent(tools, narrow),
                    about_child=state != "none",
                    child_question_kind=kind,
                )
                self.assertTrue(plan.reasons)


class TheTraceDescribesTheNarrowing(unittest.TestCase):
    def test_the_trace_reports_the_bound_list_and_the_required_tool_for_every_shape(self):
        for kind, state, tools_name, tools, narrow in _cells():
            with self.subTest(kind=kind, child=state, profile=tools_name, narrow=narrow):
                plan = _plan(
                    CHILD_STATES[state](),
                    agent=_agent(tools, narrow),
                    about_child=state != "none",
                    child_question_kind=kind,
                )
                trace = plan.as_trace()
                self.assertEqual(trace["turn_exposed_tools"], plan.exposed_tools)
                self.assertEqual(trace["turn_forced_tool"], plan.forced_tool or None)
                self.assertEqual(trace["turn_short_circuit"], plan.short_circuit)

    def test_binding_everything_is_reported_as_null_not_as_an_empty_list(self):
        """"Bind everything" and "bind nothing" are opposite decisions. A trace that
        rendered both as `[]` would make a lost tool unreadable after the fact."""
        everything = _plan(_resolved_child(), about_child=True,
                           child_question_kind="both").as_trace()
        nothing = _plan(_resolved_child(), is_social=True).as_trace()
        self.assertIsNone(everything["turn_exposed_tools"])
        self.assertEqual(nothing["turn_exposed_tools"], [])

    def test_the_trace_copies_the_list_rather_than_aliasing_it(self):
        """A trace is persisted and streamed; a later mutation of the plan must not
        rewrite a record of what was decided."""
        plan = _plan(_resolved_child(), about_child=True, child_question_kind="records")
        trace = plan.as_trace()
        plan.exposed_tools.append(KNOWLEDGE_TOOL)
        self.assertEqual(trace["turn_exposed_tools"], [RECORDS_TOOL])

    def test_the_trace_never_carries_the_childs_name(self):
        """Narrowing only happens on turns that settled on a real child, so the trace it
        produces is exactly the trace most likely to leak one."""
        plan = _plan(_resolved_child(), about_child=True, child_question_kind="records")
        self.assertTrue(plan.as_trace()["turn_child_resolved"])
        self.assertNotIn(LAYLA.label, str(plan.as_trace()))


class _RecordingCtx:
    """A progress sink with the CURRENT `emit_rag_step` signature.

    `_emit` swallows everything it raises, so a double whose signature lags the real one
    records nothing at all and every assertion about steps quietly passes — see
    `test_a_double_whose_signature_lags_records_nothing` below, which is why this one is
    kept in step with `ChatRequestContext.emit_rag_step`.
    """

    def __init__(self):
        self.steps = []

    def emit_rag_step(self, icon, label, detail="", *, group=None, group_label=None):
        self.steps.append({"icon": icon, "label": label, "detail": detail})


class _LaggingCtx:
    def __init__(self):
        self.steps = []

    def emit_rag_step(self, icon, label):  # deliberately missing `detail`
        self.steps.append((icon, label))


def _steps(plan, signals=None):
    ctx = _RecordingCtx()
    orchestrator._emit(ctx, signals or RequestSignals(question="q"), plan)
    return ctx.steps


class TheProgressStepNamesTheNarrowing(unittest.TestCase):
    def test_a_narrowed_turn_reports_how_many_tools_survived(self):
        plan = _plan(_resolved_child(), about_child=True, child_question_kind="records")
        steps = _steps(plan)
        narrowing = [step for step in steps if step["icon"] == "🎯"]
        self.assertEqual(len(narrowing), 1)
        self.assertIn(str(len(plan.exposed_tools)), narrowing[0]["label"])
        self.assertTrue(narrowing[0]["detail"])

    def test_a_turn_that_bound_everything_reports_no_narrowing_step(self):
        """A plan that changed nothing is the common case; saying so every turn would
        bury the steps that matter."""
        plan = _plan(_resolved_child(), about_child=True, child_question_kind="both")
        self.assertFalse([step for step in _steps(plan) if step["icon"] == "🎯"])

    def test_a_settled_child_is_announced_before_the_narrowing(self):
        """In a streamed turn this pair is the only thing between the opening step and
        the agent, so the order is what a waiting parent reads."""
        plan = _plan(_resolved_child(), about_child=True, child_question_kind="records")
        icons = [step["icon"] for step in _steps(plan)]
        self.assertEqual(icons, ["👤", "🎯"])

    def test_asking_which_child_reports_the_question_and_not_a_narrowing(self):
        plan = _plan(_asking_child(), about_child=True, child_question_kind="records")
        icons = [step["icon"] for step in _steps(plan)]
        self.assertEqual(icons, ["👥", "🚪"])

    def test_a_turn_with_no_child_and_no_narrowing_reports_nothing_at_all(self):
        plan = _plan(no_child("not about a child"), child_question_kind="records")
        self.assertEqual(_steps(plan), [])

    def test_every_matrix_cell_emits_at_most_one_narrowing_step(self):
        for kind, state, tools_name, tools, narrow in _cells():
            with self.subTest(kind=kind, child=state, profile=tools_name, narrow=narrow):
                plan = _plan(
                    CHILD_STATES[state](),
                    agent=_agent(tools, narrow),
                    about_child=state != "none",
                    child_question_kind=kind,
                )
                narrowing = [step for step in _steps(plan) if step["icon"] == "🎯"]
                expected = plan.exposed_tools is not None and not plan.short_circuit
                self.assertEqual(len(narrowing), 1 if expected else 0)

    def test_emitting_never_breaks_a_turn_and_never_needs_a_context(self):
        plan = _plan(_resolved_child(), about_child=True, child_question_kind="records")
        orchestrator._emit(None, RequestSignals(question="q"), plan)

    def test_a_double_whose_signature_lags_records_nothing(self):
        """Documents the trap the recording double above avoids.

        `_emit` reports progress and must never break a turn, so it swallows everything —
        including a `TypeError` from a stale sink. A test written against such a sink
        would assert on an empty list and pass forever.
        """
        ctx = _LaggingCtx()
        plan = _plan(_resolved_child(), about_child=True, child_question_kind="records")
        orchestrator._emit(ctx, RequestSignals(question="q"), plan)
        self.assertEqual(ctx.steps, [])


class TheEmptyPlanIsTheSafeDefault(unittest.TestCase):
    def test_a_plan_nobody_touched_binds_everything_and_requires_nothing(self):
        """The planner's own failure path returns a bare `TurnPlan`. It has to mean
        "run the turn exactly as it ran before this module existed"."""
        plan = TurnPlan()
        self.assertIsNone(plan.exposed_tools)
        self.assertEqual(plan.forced_tool, "")
        self.assertFalse(plan.short_circuit)
        self.assertIsNone(plan.as_trace()["turn_exposed_tools"])


if __name__ == "__main__":
    unittest.main()
