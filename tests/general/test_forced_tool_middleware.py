"""Requiring the tool the planner chose — the middleware, not the plan.

`tests/general/test_planner_tool_selection.py` proves the PLAN picks a tool. This file is
about the last hop, `_ForcePlannedTool.wrap_model_call`: the moment a chosen tool name
becomes a `tool_choice` on the request that actually reaches the provider. Narrowing the
bound list removes the wrong choice but not the choice of making none, and the model has
been measured answering «درجات ليلى أحمد كام؟» from memory with `get_student_records` as
its only tool. `tool_choice` is what closes that, so the conditions under which it is and
is not set are the whole guarantee.

What is asserted here, and why each one is a way the guarantee can quietly die:

  * FIRST-CALL DETECTION over a matrix of state shapes. The turn is only constrained on
    its first model call, and "first" is inferred, not counted — from the absence of a
    `ToolMessage` and of `tool_calls_made`. A conversation loaded from storage can carry
    an assistant message that once made tool calls with no result message beside it; if
    that were read as "the turn is under way" the forcing would never fire on a resumed
    thread, and nothing downstream would say so.

  * THE COMPOSITION WITH THE BUDGET, in the order `create_agent_for_request` lists them.
    The one request shape the provider rejects outright is "you must call a tool" with no
    tools offered, so the combined behaviour — not either middleware alone — is what has
    to be checked.

  * THE VALUE IS A BARE NAME. `ChatOpenAI.bind_tools` builds the provider's dict itself.
    Somebody "helpfully" building it here would double-wrap it.

  * THE ASYNC PATH, which nothing else covers and which is how this backend actually
    runs: `backend/chat/service.py` streams every turn through `request_agent.astream`.
    See `TheStreamedPath` below — the finding there is the reason this file exists at
    this length.
"""
import asyncio
import os
import unittest

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

from backend.chat import runtime
from backend.profiles import load_profile, registry, set_profile

RECORDS_TOOL = "get_student_records"
KNOWLEDGE_TOOL = "search_knowledge_base"


# --------------------------------------------------------------------------------------
# Fixtures. Self-contained on purpose: these tests are about a middleware's decision, so
# nothing here should be able to change because a roster or a profile helper moved.
# --------------------------------------------------------------------------------------


class _Tool:
    """A bound tool as `_tool_name` sees it: something with a name."""

    def __init__(self, name):
        self.name = name


class _Stub:
    """A request-shaped object for the degenerate cases a real `ModelRequest` will not
    hold: no tools at all, or no state. Built by hand rather than by mutating a
    `ModelRequest`, which deprecates attribute assignment."""

    def __init__(self, tools=(), state=None, tool_choice=None):
        self.tools = tools
        self.state = state
        self.tool_choice = tool_choice

    def override(self, **overrides):
        return _Stub(overrides.get("tools", self.tools),
                     self.state,
                     overrides.get("tool_choice", self.tool_choice))


class _Ctx:
    """The only thing `_force_the_planned_tool` reads off the request context."""

    def __init__(self, forced_tool=""):
        self.forced_tool = forced_tool


def _request(state=None, tools=(KNOWLEDGE_TOOL, RECORDS_TOOL), tool_choice=None):
    """A real `ModelRequest`, so `override` behaves the way the graph's does."""
    return ModelRequest(
        model=None,
        messages=[],
        tools=[t if isinstance(t, dict) else _Tool(t) for t in tools],
        tool_choice=tool_choice,
        state=state if state is not None else {},
        runtime=None,
    )


def _seen(middlewares, request):
    """What the innermost handler is finally handed, after `middlewares` compose.

    First in the list is the outermost layer — the same convention `create_agent` uses,
    so a list written in `create_agent_for_request` order composes here the way it does
    in the graph.
    """
    record = {}

    def handler(req):
        record["tool_choice"] = req.tool_choice
        record["tools"] = [runtime._tool_name(t) for t in (req.tools or [])]
        return "response"

    call = handler
    for middleware in reversed(middlewares):
        call = (lambda mw, nxt: lambda req: mw.wrap_model_call(req, nxt))(middleware, call)
    call(request)
    return record


def _chosen(state=None, *, forced=RECORDS_TOOL, tools=(KNOWLEDGE_TOOL, RECORDS_TOOL),
            tool_choice=None):
    """The `tool_choice` the forcing middleware alone lets through."""
    middleware = runtime._force_the_planned_tool(_Ctx(forced))
    return _seen([middleware], _request(state, tools, tool_choice))["tool_choice"]


class ProfileScopedTest(unittest.TestCase):
    """Profiles are process-global, and the budget reads one; restore it either way."""

    def setUp(self):
        self._saved = os.environ.get(registry.PROFILE_ENV_VAR)
        os.environ[registry.PROFILE_ENV_VAR] = "school"
        set_profile(None)
        load_profile("school")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(registry.PROFILE_ENV_VAR, None)
        else:
            os.environ[registry.PROFILE_ENV_VAR] = self._saved
        set_profile(None)


# --------------------------------------------------------------------------------------
# First-call detection
# --------------------------------------------------------------------------------------


class WhatCountsAsTheFirstCall(unittest.TestCase):
    """Only the first model call of a turn is constrained.

    Measured against this endpoint, gpt-oss ignores `tool_choice` once a tool result is
    in the history, so forcing later passes buys nothing — and costs a turn that cannot
    end, because a model told it must call a tool whose budget is spent has no legal
    move. The inference therefore has to be right in both directions.
    """

    def _states(self):
        an_ai_message_that_called_a_tool = AIMessage(
            content="",
            tool_calls=[{"name": RECORDS_TOOL, "args": {}, "id": "c1"}],
        )
        a_result = ToolMessage(content="{}", tool_call_id="c1", name=RECORDS_TOOL)
        return [
            # (label, state, is the turn still at its first model call?)
            ("state missing every key", {}, True),
            ("no messages yet", {"messages": []}, True),
            ("only the parent's question", {"messages": [HumanMessage("درجات ابني كام؟")]}, True),
            # The shape that matters: a thread loaded from storage keeps the assistant
            # text of past turns and, on some stores, the tool_calls attached to it —
            # but never the ToolMessages. Reading that as "a tool already ran" would
            # switch the forcing off for every resumed conversation.
            ("history with a tool call but no result",
             {"messages": [HumanMessage("hi"), an_ai_message_that_called_a_tool]}, True),
            ("tool_calls_made present but empty",
             {"messages": [], "tool_calls_made": {}}, True),
            ("a tool result is in this turn",
             {"messages": [an_ai_message_that_called_a_tool, a_result]}, False),
            # The one case the messages cannot show: a call the model made that produced
            # no result message at all. The budget's counter is what catches it.
            ("the budget has counted a call",
             {"messages": [], "tool_calls_made": {RECORDS_TOOL: 1}}, False),
            ("both a result and a count",
             {"messages": [an_ai_message_that_called_a_tool, a_result],
              "tool_calls_made": {RECORDS_TOOL: 1}}, False),
            ("a count for some other tool",
             {"messages": [], "tool_calls_made": {KNOWLEDGE_TOOL: 1}}, False),
        ]

    def test_the_state_shapes_a_turn_can_arrive_in_are_read_correctly(self):
        for label, state, first in self._states():
            with self.subTest(label):
                self.assertIs(runtime._first_model_call(_request(state)), first)

    def test_only_a_first_call_is_forced(self):
        """The same matrix through the middleware, because `_first_model_call` being
        right is worth nothing if the caller consults it wrongly."""
        for label, state, first in self._states():
            with self.subTest(label):
                self.assertEqual(_chosen(state), RECORDS_TOOL if first else None)

    def test_a_request_carrying_no_state_at_all_is_still_a_first_call(self):
        """`state` is None on a request built outside the graph — a middleware that
        raised here would take the whole turn down rather than declining to force."""
        self.assertIs(runtime._first_model_call(_Stub(state=None)), True)


# --------------------------------------------------------------------------------------
# Forcing only what is actually on offer
# --------------------------------------------------------------------------------------


class OnlyAToolThatIsActuallyBound(unittest.TestCase):
    """A `tool_choice` naming something that is not bound is the shape that breaks the
    call: `bind_tools` cannot resolve it, and the provider is asked to require a tool it
    was never given."""

    def test_a_tool_missing_from_the_request_is_not_forced(self):
        self.assertIsNone(_chosen(forced=RECORDS_TOOL, tools=(KNOWLEDGE_TOOL,)))

    def test_a_name_nobody_bound_is_not_forced(self):
        self.assertIsNone(_chosen(forced="a_tool_nobody_has_written_yet"))

    def test_an_empty_tool_list_is_not_forced(self):
        self.assertIsNone(_chosen(tools=()))

    def test_a_request_whose_tools_are_none_is_not_forced(self):
        middleware = runtime._force_the_planned_tool(_Ctx(RECORDS_TOOL))
        self.assertIsNone(_seen([middleware], _Stub(tools=None))["tool_choice"])

    def test_a_tool_carried_in_provider_dict_shape_still_counts_as_bound(self):
        """Server-side tools arrive as dicts rather than objects. Reading only `.name`
        would leave a genuinely bound tool looking absent, and silently unforced."""
        as_dict = {"type": "function", "function": {"name": RECORDS_TOOL}}
        self.assertEqual(_chosen(tools=(KNOWLEDGE_TOOL, as_dict)), RECORDS_TOOL)


class NothingToForce(unittest.TestCase):
    """No plan, no forcing. The planner leaves `forced_tool` empty whenever it is not
    certain, and uncertainty must never turn into a required call."""

    def test_an_absent_plan_never_forces(self):
        for label, forced in [("empty string", ""), ("spaces", "   "), ("a tab", "\t"),
                              ("None", None)]:
            with self.subTest(label):
                self.assertIsNone(_chosen(forced=forced))

    def test_a_name_with_stray_whitespace_is_still_the_tool(self):
        self.assertEqual(_chosen(forced="  %s " % RECORDS_TOOL), RECORDS_TOOL)

    def test_a_context_with_no_forced_tool_attribute_at_all_is_harmless(self):
        """`_force_the_planned_tool` reads the attribute off the context defensively;
        an older context object must degrade to "force nothing", not to an AttributeError
        that ends the turn."""
        class _Bare:
            pass

        middleware = runtime._force_the_planned_tool(_Bare())
        self.assertIsNone(_seen([middleware], _request())["tool_choice"])


class SomebodyElseAlreadyDecided(unittest.TestCase):
    """An existing `tool_choice` has a better claim: structured output binds `any`, and
    the budget clears it to None when it has withheld everything. Overruling either is
    how a turn ends up requiring a call it cannot make."""

    def test_an_existing_choice_is_never_overruled(self):
        for existing in ["any", "required", "none", "auto", KNOWLEDGE_TOOL]:
            with self.subTest(existing):
                self.assertEqual(_chosen(tool_choice=existing), existing)

    def test_a_dict_shaped_choice_is_left_exactly_as_it_arrived(self):
        existing = {"type": "function", "function": {"name": KNOWLEDGE_TOOL}}
        self.assertEqual(_chosen(tool_choice=existing), existing)

    def test_a_false_but_not_none_choice_is_still_somebody_elses_decision(self):
        """`False` is a real instruction to OpenAI-shaped providers, and it is falsy —
        a truthiness check here would silently overrule it."""
        self.assertIs(_chosen(tool_choice=False), False)


class TheRequestIsNeverMutated(unittest.TestCase):
    """The middleware overrides a copy. Mutating the request in place would leak the
    forced choice into a retry, or into a sibling middleware that had already read it."""

    def test_the_incoming_request_still_has_its_own_tool_choice_afterwards(self):
        request = _request()
        middleware = runtime._force_the_planned_tool(_Ctx(RECORDS_TOOL))
        seen = _seen([middleware], request)
        self.assertEqual(seen["tool_choice"], RECORDS_TOOL)
        self.assertIsNone(request.tool_choice)

    def test_the_handler_is_handed_a_different_object(self):
        request = _request()
        middleware = runtime._force_the_planned_tool(_Ctx(RECORDS_TOOL))
        handed = {}

        def handler(req):
            handed["request"] = req
            return "response"

        middleware.wrap_model_call(request, handler)
        self.assertIsNot(handed["request"], request)
        self.assertEqual([runtime._tool_name(t) for t in handed["request"].tools],
                         [KNOWLEDGE_TOOL, RECORDS_TOOL])

    def test_declining_to_force_passes_the_original_request_straight_through(self):
        request = _request({"tool_calls_made": {RECORDS_TOOL: 1}})
        middleware = runtime._force_the_planned_tool(_Ctx(RECORDS_TOOL))
        handed = {}
        middleware.wrap_model_call(request, lambda req: handed.setdefault("request", req))
        self.assertIs(handed["request"], request)


# --------------------------------------------------------------------------------------
# The value that reaches the provider
# --------------------------------------------------------------------------------------


class TheValueSentIsABareName(unittest.TestCase):
    """Verified against the installed `langchain_openai`, `chat_models/base.py` in
    `bind_tools`:

        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice in tool_names:
                    tool_choice = {"type": "function", "function": {"name": tool_choice}}

    So the provider dict is built there, from a bare name. Building it in the middleware
    instead would hand `bind_tools` a dict, which it passes through untouched — and a
    dict wrapped around a dict is not a shape the endpoint accepts. This test exists to
    stop that "helpful" change.
    """

    def test_the_choice_is_the_tool_name_and_not_a_structure(self):
        choice = _chosen()
        self.assertIsInstance(choice, str)
        self.assertEqual(choice, RECORDS_TOOL)

    def test_binding_that_bare_name_is_what_produces_the_provider_shape(self):
        """Run through the real `ChatOpenAI.bind_tools` — no network, binding is local.
        If this ever stops producing the dict, the middleware's contract changed under it
        and the bare name would reach the wire unconverted."""
        from langchain_openai import ChatOpenAI

        @tool
        def get_student_records(student_name: str) -> str:
            """The child's record."""
            return "{}"

        bound = ChatOpenAI(model="gpt-4o-mini", api_key="not-a-real-key").bind_tools(
            [get_student_records], tool_choice=_chosen(tools=(RECORDS_TOOL,))
        )
        self.assertEqual(
            bound.kwargs["tool_choice"],
            {"type": "function", "function": {"name": RECORDS_TOOL}},
        )


# --------------------------------------------------------------------------------------
# Composed with the budget, in the order the agent composes them
# --------------------------------------------------------------------------------------


class ComposedWithTheBudget(ProfileScopedTest):
    """`create_agent_for_request` puts the budget OUTSIDE the forcing, so the request
    reaching the forcing has already had spent tools removed. The pair is what has to be
    correct: the one shape the provider rejects outright is a required tool call with no
    tools on offer.
    """

    def _both(self, state, forced=RECORDS_TOOL, tools=(KNOWLEDGE_TOOL, RECORDS_TOOL)):
        return _seen(
            [runtime._spend_tool_budgets(_Ctx()), runtime._force_the_planned_tool(_Ctx(forced))],
            _request(state, tools),
        )

    def test_a_fresh_turn_is_offered_everything_and_forced_to_the_planned_tool(self):
        seen = self._both({})
        self.assertEqual(seen["tools"], [KNOWLEDGE_TOOL, RECORDS_TOOL])
        self.assertEqual(seen["tool_choice"], RECORDS_TOOL)

    def test_the_budget_withholding_the_planned_tool_never_leaves_it_forced(self):
        """`get_student_records` has a budget of three in the school profile; spend it
        and the tool is withheld. Whatever the reason the forcing stands down — spent
        budget, or simply not being the first call any more — the outcome that matters
        is that nothing requires a tool the request no longer offers."""
        seen = self._both({"tool_calls_made": {RECORDS_TOOL: 9}})
        self.assertNotIn(RECORDS_TOOL, seen["tools"])
        self.assertIsNone(seen["tool_choice"])

    def test_the_budget_withholding_everything_leaves_no_choice_at_all(self):
        """The rejected shape: required call, nothing to call. `tool_choice` must be
        None here and never a tool name."""
        seen = self._both({"tool_calls_made": {KNOWLEDGE_TOOL: 9, RECORDS_TOOL: 9}})
        self.assertEqual(seen["tools"], [])
        self.assertIsNone(seen["tool_choice"])

    def test_a_spent_sibling_does_not_stop_the_planned_tool_being_offered(self):
        seen = self._both({"tool_calls_made": {KNOWLEDGE_TOOL: 9}})
        self.assertEqual(seen["tools"], [RECORDS_TOOL])
        self.assertIsNone(seen["tool_choice"])

    def test_forcing_and_withholding_can_never_be_in_conflict_on_a_first_call(self):
        """Structural, and the reason the guard in the forcing is sufficient: the budget
        only withholds once `tool_calls_made` is non-empty, and a non-empty
        `tool_calls_made` is exactly what makes the call not the first one. So on the
        only call that is ever forced, nothing has been withheld."""
        for spent in [{}, {"tool_calls_made": {}}]:
            with self.subTest(spent):
                self.assertTrue(runtime._first_model_call(_request(spent)))
                self.assertEqual(self._both(spent)["tools"],
                                 [KNOWLEDGE_TOOL, RECORDS_TOOL])
        self.assertFalse(runtime._first_model_call(
            _request({"tool_calls_made": {KNOWLEDGE_TOOL: 1}})))


# --------------------------------------------------------------------------------------
# The path production actually takes
# --------------------------------------------------------------------------------------


@tool
def get_student_records(student_name: str) -> str:
    """Read a child's record."""
    return "{}"


class _RecordingModel(GenericFakeChatModel):
    """Answers immediately and remembers how it was bound.

    `create_agent` binds through `request.model.bind_tools(tools, tool_choice=...)`, so
    recording that keyword is exactly recording what the provider would have been told.
    """

    bindings: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bindings.append(kwargs)
        return self


def _agent_with(middleware, model):
    return create_agent(model=model, tools=[get_student_records], middleware=[middleware])


def _one_question():
    return {"messages": [HumanMessage("درجات ليلى كام؟")]}


class TheStreamedPath(unittest.TestCase):
    """The middleware defines only the SYNCHRONOUS `wrap_model_call`, and this backend
    runs every turn through `request_agent.astream` (backend/chat/service.py).

    Read in the installed langchain (`langchain/agents/factory.py`, v1.3.18): the async
    composition collects middleware into `middleware_w_awrap_model_call` when the class
    overrides EITHER hook —

        if m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
        or m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call

    — and then composes `m.awrap_model_call`. A sync-only middleware therefore ends up in
    the async stack with the BASE class's `awrap_model_call`, which raises
    NotImplementedError ("you defined only the sync version ... and invoked your agent in
    an asynchronous context"). There is no sync-to-async adaptation anywhere on that path.

    So on the streamed path the forcing does not merely fail to apply: the model node
    raises, and every turn built by `create_agent_for_request` is affected — `_ToolBudget`
    has the same shape, and it is installed on every turn regardless of the plan.

    The synchronous test below is the control: it proves the harness really does observe
    a forced `tool_choice`, so the async failures are the production code's and not the
    fixture's.
    """

    def test_a_synchronous_turn_forces_the_planned_tool_through_a_real_agent(self):
        model = _RecordingModel(messages=iter([AIMessage(content="ليلى حصلت على 95.")]))
        agent = _agent_with(runtime._ForcePlannedTool(RECORDS_TOOL), model)
        agent.invoke(_one_question())
        self.assertEqual([b.get("tool_choice") for b in model.bindings], [RECORDS_TOOL])

    def test_a_synchronous_turn_with_no_plan_leaves_the_choice_open(self):
        model = _RecordingModel(messages=iter([AIMessage(content="أهلاً.")]))
        agent = _agent_with(runtime._ForcePlannedTool(""), model)
        agent.invoke(_one_question())
        self.assertEqual([b.get("tool_choice") for b in model.bindings], [None])

    def test_a_streamed_turn_forces_the_planned_tool_too(self):
        """Production streams every turn, so this is THE path that matters.

        It was red when written: the middleware had only the sync `wrap_model_call`, and
        the base class's `awrap_model_call` raises. Forcing therefore did nothing on the
        one path every parent uses, and the unit tests all passed because they drive the
        sync hook directly."""
        model = _RecordingModel(messages=iter([AIMessage(content="ليلى حصلت على 95.")]))
        agent = _agent_with(runtime._ForcePlannedTool(RECORDS_TOOL), model)
        asyncio.run(agent.ainvoke(_one_question()))
        self.assertEqual([b.get("tool_choice") for b in model.bindings], [RECORDS_TOOL])

    def test_a_streamed_turn_survives_the_budget_middleware(self):
        """Same fault, wider blast radius: `_ToolBudget` is bound on EVERY turn, planned
        or not, so the missing async hook took the streamed chat down before any model
        call happened at all."""
        model = _RecordingModel(messages=iter([AIMessage(content="أهلاً.")]))
        agent = _agent_with(runtime._ToolBudget(), model)
        asyncio.run(agent.ainvoke(_one_question()))
        self.assertEqual([b.get("tool_choice") for b in model.bindings], [None])

    def test_both_middlewares_answer_on_the_async_side_too(self):
        """Pins the shape of the fix rather than the symptom.

        `awrap_model_call` raises on the base class and `aafter_model` is a silent no-op,
        so a middleware overriding only the sync halves either crashes the streamed turn
        or quietly stops counting. Anything added to this stack has to implement both.
        """
        for middleware in [runtime._ForcePlannedTool(RECORDS_TOOL), runtime._ToolBudget()]:
            with self.subTest(type(middleware).__name__):
                kind = type(middleware)
                self.assertIsNot(kind.awrap_model_call, AgentMiddleware.awrap_model_call)
                model = _RecordingModel(messages=iter([AIMessage(content="أهلاً.")]))
                agent = _agent_with(middleware, model)
                asyncio.run(agent.ainvoke(_one_question()))
                self.assertTrue(model.bindings, "the async path never reached the model")

    def test_the_budget_counts_a_streamed_tool_call(self):
        """`after_model` alone is not enough: the async graph calls `aafter_model`, whose
        default returns nothing, so a streamed turn counted no tool calls and the budget
        read zero forever — invisible, because it looks exactly like a spent budget."""
        counted = runtime._ToolBudget()
        state = {"messages": [
            HumanMessage(content="درجات ليلى كام؟"),
            AIMessage(content="", tool_calls=[
                {"name": RECORDS_TOOL, "args": {}, "id": "call-1"}
            ]),
        ]}
        self.assertEqual(
            asyncio.run(counted.aafter_model(state, None)),
            {"tool_calls_made": {RECORDS_TOOL: 1}},
        )


if __name__ == "__main__":
    unittest.main()
