"""The child's year group reaches the answer, without reaching the search query.

The failure: a parent whose son is in Year 1 asked «مصاريف ابني كام» and was told the
Year 4 figure. The roster knew the year, `ResolvedChild` carried it, and it reached the
prompt as decoration — "who is in {{ child_year }}" — with nothing saying it bound the
answer. Retrieval ranked the whole fee table and the wrong row won.

The obvious fix is the one that must NOT be made. Appending conditions to the retrieval
query was tried and reverted here after costing three of twenty turns
(`backend/rag/pipeline.py:_search_query`): a year group appears in no passage the corpus
wrote once for everybody, so every such query term is dilution. The year therefore
travels beside the question to the two stages that can act on it without costing recall
— the grader, and the answer prompt.
"""
import unittest

from backend.chat.child_resolution import ResolvedChild
from backend.chat.request_context import ChatRequestContext
from backend.chat.turn_policy import _plan_child, TurnPlan, question_names_a_year
from backend.profiles import get_profile
from backend.prompts import render as render_prompt, resolve as resolve_prompt
from backend.rag import pipeline

MARKERS = get_profile().agent.year_reference_markers
YEAR = "الصف الأول الابتدائي"


def _resolved(year=YEAR):
    return ResolvedChild(
        student_id="s1", label="علي", year_level=year, resolved=True, source="only_child"
    )


class YearGateTests(unittest.TestCase):
    """Whether the parent scoped the question to a year themselves."""

    def test_a_question_that_names_no_year_lets_the_roster_year_apply(self):
        self.assertFalse(question_names_a_year("مصاريف ابني كام", MARKERS))

    def test_a_question_naming_a_year_is_recognised(self):
        for question in (
            "مصاريف ابني في الصف الرابع كام",
            "what are the fees for grade 4",
            "fees for primary 3",
        ):
            with self.subTest(question=question):
                self.assertTrue(question_names_a_year(question, MARKERS))

    def test_orthography_does_not_defeat_the_gate(self):
        """Folded like the roster matcher, so a different alif form is the same word."""
        self.assertTrue(question_names_a_year("مصاريف ابني في الصف الرابع", MARKERS))

    def test_an_empty_marker_list_never_blocks(self):
        self.assertFalse(question_names_a_year("مصاريف الصف الرابع", []))


class PlanTests(unittest.TestCase):
    def test_the_year_is_applied_when_the_question_names_none(self):
        plan = TurnPlan()
        _plan_child(plan, _resolved(), "مصاريف ابني كام", MARKERS)
        self.assertEqual(plan.child_hint, "علي")
        self.assertEqual(plan.child_year, YEAR)
        self.assertTrue(plan.as_trace()["turn_child_year_applied"])

    def test_the_name_survives_when_the_year_is_withheld(self):
        """«مصاريف ابني في الصف الرابع» is still about this child — but the year to
        answer for is the one the parent said, not the one on file."""
        plan = TurnPlan()
        _plan_child(plan, _resolved(), "مصاريف ابني في الصف الرابع كام", MARKERS)
        self.assertEqual(plan.child_hint, "علي")
        self.assertEqual(plan.child_year, "")
        self.assertFalse(plan.as_trace()["turn_child_year_applied"])
        self.assertIn("question names its own year", "; ".join(plan.reasons))

    def test_a_child_with_no_year_on_file_applies_nothing(self):
        plan = TurnPlan()
        _plan_child(plan, _resolved(year=""), "مصاريف ابني كام", MARKERS)
        self.assertEqual(plan.child_hint, "علي")
        self.assertEqual(plan.child_year, "")

    def test_an_unresolved_child_still_only_asks(self):
        plan = TurnPlan()
        _plan_child(plan, ResolvedChild(ask=True), "مصاريف ابني كام", MARKERS)
        self.assertEqual(plan.child_year, "")
        self.assertEqual(plan.child_hint, "")


class ItReachesTheGraphTests(unittest.TestCase):
    def test_note_turn_plan_carries_the_year_into_the_graph_state(self):
        ctx = ChatRequestContext(user_id="u", session_id="s")
        ctx.note_turn_plan([], [], child_year=YEAR)
        state = pipeline._initial_state("مصاريف ابني كام", ctx)
        self.assertEqual(state["child_year"], YEAR)

    def test_the_search_query_is_still_only_the_question(self):
        """The measured regression this whole design avoids. If this ever fails, recall
        has been traded away — read `_search_query`'s docstring before changing it."""
        ctx = ChatRequestContext(user_id="u", session_id="s")
        ctx.note_turn_plan([], [], child_year=YEAR)
        state = pipeline._initial_state("مصاريف ابني كام", ctx)
        self.assertEqual(pipeline._search_query(state), "مصاريف ابني كام")
        self.assertNotIn(YEAR, pipeline._search_query(state))

    def test_the_grader_is_told_the_year_as_a_condition(self):
        state = {"carried_constraints": [], "child_year": YEAR}
        conditions = pipeline.grading_conditions(state)
        self.assertEqual(len(conditions), 1)
        self.assertIn(YEAR, conditions[0])

    def test_the_condition_says_the_records_are_its_source(self):
        """A condition that misreports where it came from is the fabricated-provenance
        pattern `backend/rag/evidence.py` exists to prevent — the user did not say this."""
        conditions = pipeline.grading_conditions({"child_year": YEAR})
        self.assertIn("school's records", conditions[0])

    def test_user_conditions_and_the_year_travel_together(self):
        state = {"carried_constraints": ["up to Year 6"], "child_year": YEAR}
        self.assertEqual(len(pipeline.grading_conditions(state)), 2)

    def test_no_year_adds_no_condition(self):
        self.assertEqual(pipeline.grading_conditions({"carried_constraints": []}), [])


class AnswerPromptTests(unittest.TestCase):
    """What the model is actually told, per the grader's verdict."""

    def _render(self, discriminate):
        return render_prompt(
            "tools/knowledge_result.j2",
            outcome="chunks",
            chunks="[1] fees.pdf (Page 3): ...",
            constraints=[],
            discriminate=discriminate,
            rewritten=False,
            partial=False,
            figures=False,
            child_year=YEAR,
        )

    def test_material_that_varies_by_year_narrows_the_answer(self):
        rendered = self._render("yes")
        self.assertIn(YEAR, rendered)
        self.assertIn("leave the other year groups out", rendered)

    def test_material_that_does_not_vary_is_given_in_full(self):
        """The catastrophic case the carried-condition block already documents: a
        document list written once for everybody must not be withheld from a Year 1
        parent because it does not name Year 1."""
        rendered = self._render("no")
        self.assertIn("does not vary by year group", rendered)
        self.assertIn("Do NOT withhold it", rendered)

    def test_an_unknown_verdict_still_forbids_another_year_s_figure(self):
        rendered = self._render("unknown")
        self.assertIn("Never state a figure for a different year group", rendered)

    def test_a_turn_with_no_child_year_renders_no_year_block(self):
        rendered = render_prompt(
            "tools/knowledge_result.j2",
            outcome="chunks",
            chunks="[1] fees.pdf: ...",
            constraints=[],
            discriminate="unknown",
            rewritten=False,
            partial=False,
            figures=False,
        )
        self.assertNotIn("THE YEAR TO ANSWER FOR", rendered)

    def test_the_turn_context_states_the_year_as_binding(self):
        rendered = resolve_prompt(
            "", "agent/turn_context.j2",
            resolved_question="", constraints=[],
            child_hint="علي", child_year=YEAR, child_options=[],
        )
        self.assertIn("binds the answer", rendered)
        self.assertIn("must not be given as theirs", rendered)


if __name__ == "__main__":
    unittest.main()


class OlderContextCompatibilityTests(unittest.TestCase):
    """`note_turn_plan` promises a context written against an earlier signature keeps
    working. Adding `child_year` broke that in the one direction nobody checked: the
    call raised TypeError, the blanket handler swallowed it, and the turn silently lost
    its sections, conditions AND language as well."""

    class OldCtx:
        """No `child_year` parameter — the signature before this change."""

        def __init__(self):
            self.retrieval_sections = []
            self.carried_constraints = []
            self.language = ""

        def note_turn_plan(self, retrieval_sections, scope_options, *,
                           carried_constraints=(), is_followup=False, language=""):
            self.retrieval_sections = list(retrieval_sections or [])
            self.carried_constraints = list(carried_constraints or [])
            self.language = language

    def test_an_older_context_still_receives_every_hint_it_understands(self):
        from backend.chat.orchestrator import _hand_to_graph

        plan = TurnPlan(
            retrieval_sections=["fees"],
            carried_constraints=["up to Year 6"],
            language="ar",
            child_year=YEAR,
        )
        ctx = self.OldCtx()
        _hand_to_graph(ctx, plan)

        self.assertEqual(ctx.retrieval_sections, ["fees"])
        self.assertEqual(ctx.carried_constraints, ["up to Year 6"])
        self.assertEqual(ctx.language, "ar")

    def test_a_context_taking_only_the_two_positional_arguments_still_works(self):
        from backend.chat.orchestrator import _hand_to_graph

        class Minimal:
            def __init__(self):
                self.sections = None

            def note_turn_plan(self, retrieval_sections, scope_options):
                self.sections = list(retrieval_sections or [])

        ctx = Minimal()
        _hand_to_graph(ctx, TurnPlan(retrieval_sections=["fees"], child_year=YEAR))
        self.assertEqual(ctx.sections, ["fees"])
