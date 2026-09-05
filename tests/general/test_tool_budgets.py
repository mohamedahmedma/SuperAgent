"""What the agent is allowed to ask for, and what happens when it has asked enough.

The tools have always refused a call past their own ceiling, and the refusal is what
does the damage: it comes back as a tool result, the model reads it as a failure, and it
asks again in different words. Measured against `openai/gpt-oss-20b`, one question whose
search returned nothing produced 29 consecutive `search_knowledge_base` calls, and an
earlier arrangement produced 85 before the recursion limit ended the turn — every one a
full retrieval for an answer the model already had.

So the budget moved into the graph, and it withholds instead of refusing. A tool that is
not in the request is not a failure to reason about; it is simply not on offer, and the
turn ends in an answer. These tests assert on what the model is OFFERED, because that is
the mechanism — asserting on the count alone would pass just as well for the refusing
version that caused the problem.
"""
import os
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from backend.chat import runtime
from backend.profiles import load_profile, registry, set_profile


class _Request:
    """The parts of `ModelRequest` this middleware touches."""

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


def _offered(state, tools=("search_knowledge_base", "get_student_records")):
    """What `wrap_model_call` lets through, given what the turn has already spent."""
    middleware = runtime._spend_tool_budgets(ctx=None)
    seen = {}

    def handler(request):
        seen["tools"] = [runtime._tool_name(t) for t in request.tools]
        seen["tool_choice"] = request.tool_choice
        return "response"

    middleware.wrap_model_call(_Request([_Tool(n) for n in tools], state, "auto"), handler)
    return seen


class ProfileScopedTest(unittest.TestCase):
    """Profiles are process-global; restore the cache either way."""

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


class TheBudgetComesFromOnePlacePerTool(ProfileScopedTest):
    def test_retrieval_reuses_the_knowledge_budget_rather_than_a_second_setting(self):
        """Two numbers for one budget is two numbers that can disagree — and when they
        disagree the lower one refuses, which is the failure this exists to prevent."""
        self.assertEqual(runtime.budget_for("search_knowledge_base"), 2)

    def test_records_keeps_the_higher_ceiling_it_already_had(self):
        """Two children is a legitimate three-call sequence: list them, then read each.
        Three and not the four the tool itself allows — the fourth pass pushed a turn
        past the step limit on a question that otherwise answered fine."""
        self.assertEqual(runtime.budget_for("get_student_records"), 3)

    def test_anything_unlisted_may_be_called_once(self):
        self.assertEqual(runtime.budget_for("search_products"), 1)
        self.assertEqual(runtime.budget_for("a_tool_nobody_has_written_yet"), 1)


class TheCountLivesInGraphState(ProfileScopedTest):
    def _count(self, state, message):
        middleware = runtime._spend_tool_budgets(ctx=None)
        return middleware.after_model({**state, "messages": [message]}, None)

    def test_a_call_is_counted_against_the_tool_that_made_it(self):
        update = self._count({}, AIMessage(content="", tool_calls=[
            {"name": "search_knowledge_base", "args": {}, "id": "c1"}]))
        self.assertEqual(update, {"tool_calls_made": {"search_knowledge_base": 1}})

    def test_counts_accumulate_across_model_steps(self):
        update = self._count(
            {"tool_calls_made": {"search_knowledge_base": 1}},
            AIMessage(content="", tool_calls=[
                {"name": "search_knowledge_base", "args": {}, "id": "c2"},
                {"name": "get_student_records", "args": {}, "id": "c3"}]))
        self.assertEqual(update["tool_calls_made"],
                         {"search_knowledge_base": 2, "get_student_records": 1})

    def test_a_message_with_no_tool_call_changes_nothing(self):
        self.assertIsNone(self._count({}, AIMessage(content="رسوم الصف الأول 30,000 جنيه.")))


class ASpentToolIsNotOffered(ProfileScopedTest):
    def test_everything_is_offered_before_anything_is_spent(self):
        self.assertEqual(_offered({})["tools"],
                         ["search_knowledge_base", "get_student_records"])

    def test_the_tool_survives_up_to_its_budget(self):
        offered = _offered({"tool_calls_made": {"search_knowledge_base": 1}})["tools"]
        self.assertIn("search_knowledge_base", offered)

    def test_the_tool_disappears_once_its_budget_is_spent(self):
        offered = _offered({"tool_calls_made": {"search_knowledge_base": 2}})["tools"]
        self.assertNotIn("search_knowledge_base", offered)
        self.assertIn("get_student_records", offered)

    def test_a_spent_tool_does_not_take_the_others_with_it(self):
        """The turn should still be able to do the half it has not done yet."""
        offered = _offered({"tool_calls_made": {"search_knowledge_base": 9}})["tools"]
        self.assertEqual(offered, ["get_student_records"])

    def test_records_survives_where_a_default_budget_tool_would_not(self):
        state = {"tool_calls_made": {"get_student_records": 2}}
        self.assertIn("get_student_records", _offered(state)["tools"])

    def test_forcing_a_tool_call_is_dropped_when_none_are_left(self):
        """A request that requires a tool call and offers none is rejected by the
        provider before the model ever sees it."""
        seen = _offered({"tool_calls_made":
                         {"search_knowledge_base": 2, "get_student_records": 4}})
        self.assertEqual(seen["tools"], [])
        self.assertIsNone(seen["tool_choice"])

    def test_an_untouched_request_keeps_its_tool_choice(self):
        self.assertEqual(_offered({})["tool_choice"], "auto")


class TheBudgetsMustFitTheStepLimit(ProfileScopedTest):
    """The two settings are independent and they interact, which is how they drifted.

    Every tool call costs a whole pass of the agent loop, so budgets totalling more
    passes than `recursion_limit` allows produce a turn that dies at the limit HOLDING A
    FINISHED ANSWER. That is not hypothetical: `get_student_records: 4` beside two
    knowledge calls needed about thirty steps against a limit of sixteen, and one
    question — "درجات يوسف ابراهيم إيه؟" — failed three times out of three with an empty
    reply while every other question in the suite passed.
    """

    def test_every_shipped_profile_can_afford_its_own_budgets(self):
        from backend.profiles import available_profiles

        for name in available_profiles():
            with self.subTest(profile=name):
                agent = load_profile(name).agent
                rounds = sum(agent.budget_for_tool(t) for t in agent.tools) + 1
                self.assertGreaterEqual(agent.recursion_limit, rounds * 5)

    def test_a_budget_the_graph_cannot_spend_is_refused_at_load(self):
        """Caught when the profile loads rather than on the one question that needed the
        last call — which is where it was found the first time."""
        from backend.profiles.schema import AgentConfig

        with self.assertRaises(Exception) as raised:
            AgentConfig(tools=["search_knowledge_base", "get_student_records"],
                        recursion_limit=8,
                        tool_call_budgets={"get_student_records": 4},
                        max_knowledge_calls_per_turn=2)
        self.assertIn("recursion_limit", str(raised.exception))

    def test_the_resolver_on_the_config_agrees_with_the_one_in_runtime(self):
        """Two implementations of one rule is one implementation and one bug waiting."""
        agent = load_profile("school").agent
        for name in ("search_knowledge_base", "get_student_records", "search_products"):
            with self.subTest(tool=name):
                self.assertEqual(agent.budget_for_tool(name), runtime.budget_for(name))


class TheMiddlewareIsWiredIn(ProfileScopedTest):
    def test_the_agent_carries_the_budget_and_counts_after_the_duplicate_collapse(self):
        """Ordering matters: duplicates are dropped as the model produces them, so the
        budget counts calls that will actually run rather than the copies."""
        import inspect

        source = inspect.getsource(runtime.create_agent_for_request)
        self.assertIn("_spend_tool_budgets(ctx)", source)
        self.assertLess(source.index("_collapse_duplicate_tool_calls(ctx)"),
                        source.index("_spend_tool_budgets(ctx)"))


if __name__ == "__main__":
    unittest.main()
