"""The agent system prompt is composed per turn, from the tools that turn bound.

What these tests protect, in order of how expensive the failure is:

* **Prompt and binding stay one decision.** A prompt that describes an unbound tool
  invites a call for a tool that does not exist; a bound tool whose instructions were
  dropped gets used wrongly. Both are silent at runtime, so they are pinned here.
* **StrictUndefined stays on.** Jinja's default renders an unknown variable as an
  empty string, which would ship a prompt missing its citation contract with no error
  at all — and asset attachment parses [n] out of the answer downstream.
* **Profile text stays data.** A persona is supplied by a deployment, so it must never
  be evaluated as template source.
* **The prompt stays small.** It is paid on every turn, so growth is a real cost and
  wants to be a deliberate, visible edit rather than a drift.
"""
import unittest

from jinja2 import UndefinedError

from backend.profiles.registry import load_profile
from backend.prompts import render, resolve, template_names
from backend.tools import GROUNDED_TOOLS, TOOL_BUILDERS

# The prompt is paid on every turn. ~4 chars/token for English, so this is roughly a
# 150-token ceiling — well above the ~96 it renders at today, but low enough that
# re-adding one of the paragraphs deleted in the cleanup trips it.
SYSTEM_PROMPT_CHAR_BUDGET = 600


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile("supermew")

    def test_a_grounded_turn_gets_the_citation_contract(self):
        prompt = self.profile.render_system_prompt(["search_knowledge_base"])
        self.assertIn("[1], or [2][3]", prompt)
        self.assertIn("retrieved chunks", prompt)

    def test_a_turn_with_nothing_to_cite_does_not_pay_for_citations(self):
        """The whole point of composing: a weather-only turn has no chunks and no
        citations, so every word about them would be waste on every one of its turns."""
        prompt = self.profile.render_system_prompt(["get_current_weather"])
        self.assertNotIn("[1]", prompt)
        self.assertNotIn("Grounding rules", prompt)
        self.assertLess(len(prompt), len(self.profile.render_system_prompt(["search_knowledge_base"])))

    def test_every_grounded_tool_triggers_the_contract_on_its_own(self):
        for name in sorted(GROUNDED_TOOLS):
            with self.subTest(tool=name):
                self.assertIn("[1], or [2][3]", self.profile.render_system_prompt([name]))

    def test_the_persona_always_opens_the_prompt(self):
        for tools in (None, ["search_knowledge_base"], ["get_current_weather"], []):
            with self.subTest(tools=tools):
                self.assertTrue(
                    self.profile.render_system_prompt(tools).startswith(
                        self.profile.identity.persona
                    )
                )

    def test_style_rules_survive_every_turn_shape(self):
        """Language and output shape are not tool-conditional — an ungrounded turn
        still has to answer in the user's language."""
        for tools in (None, ["search_knowledge_base"], ["get_current_weather"], []):
            with self.subTest(tools=tools):
                prompt = self.profile.render_system_prompt(tools)
                self.assertIn("language the user wrote in", prompt)
                self.assertIn("Markdown is supported", prompt)

    def test_none_means_everything_the_profile_allows(self):
        """Narrowing is an optimisation, so absence of a decision must never be the
        reason a capability goes unmentioned while its tool is bound."""
        self.assertEqual(
            self.profile.render_system_prompt(),
            self.profile.render_system_prompt(self.profile.agent.tools),
        )

    def test_the_prompt_stays_within_its_per_turn_budget(self):
        for name in ("base", "supermew", "document_kb", "ecommerce"):
            with self.subTest(profile=name):
                prompt = load_profile(name).render_system_prompt()
                self.assertLess(len(prompt), SYSTEM_PROMPT_CHAR_BUDGET)


class OverrideTests(unittest.TestCase):
    def test_a_profile_supplied_prompt_wins_verbatim(self):
        profile = load_profile("base").model_copy(deep=True)
        profile.agent.system_prompt = "{persona} Only this."
        self.assertEqual(
            f"{profile.identity.persona} Only this.",
            profile.render_system_prompt(["search_knowledge_base"]),
        )

    def test_an_override_containing_literal_braces_survives(self):
        """render uses replace(), not format(): prompts carry JSON examples and [n]
        citation markers that format() would raise on."""
        profile = load_profile("base").model_copy(deep=True)
        profile.agent.system_prompt = '{persona} Return {"a": 1} and cite [1].'
        rendered = profile.render_system_prompt()
        self.assertIn('{"a": 1}', rendered)
        self.assertIn("cite [1]", rendered)

    def test_the_shipped_profiles_leave_it_to_the_template(self):
        """A profile that sets system_prompt opts out of per-turn narrowing, so none
        of the shipped ones do."""
        for name in ("base", "supermew", "document_kb", "ecommerce"):
            with self.subTest(profile=name):
                self.assertEqual("", load_profile(name).agent.system_prompt)


class TemplateEnvironmentTests(unittest.TestCase):
    def test_a_missing_variable_raises_instead_of_rendering_empty(self):
        """StrictUndefined is the guard. Without it a typo'd variable silently yields
        an empty string, and a prompt missing its citation rule ships unnoticed."""
        with self.assertRaises(UndefinedError):
            render("agent/system.j2", grounded=True)  # no persona

    def test_profile_text_is_data_not_template_source(self):
        """A persona comes from a deployment's config. Jinja does not recursively
        expand a variable's contents, and this pins that: the braces stay literal."""
        profile = load_profile("base").model_copy(deep=True)
        profile.identity.persona = "You are {{ 7 * 6 }} and {% raw %}x{% endraw %}."
        rendered = profile.render_system_prompt()
        self.assertIn("{{ 7 * 6 }}", rendered)
        self.assertNotIn("42", rendered)

    def test_every_shipped_template_is_syntactically_valid(self):
        from backend.prompts import _environment

        names = template_names()
        self.assertTrue(names, "no templates were discovered")
        for name in names:
            with self.subTest(template=name):
                _environment().get_template(name)  # compiles, or raises


class ResolveTests(unittest.TestCase):
    """`resolve` is what keeps the migration out of profile YAML reversible: a
    deployment that already tuned a prompt keeps its text, everything else gets the
    shipped template."""

    def test_an_override_wins_over_the_template(self):
        out = resolve("Grade {question} against {context}", "rag/evidence_grade.j2",
                      question="Q", context="C")
        self.assertEqual("Grade Q against C", out)

    def test_no_override_falls_through_to_the_template(self):
        out = resolve("", "rag/evidence_grade.j2", question="Q", context="C")
        self.assertIn("RAG evidence grader", out)
        self.assertIn("Q", out)

    def test_an_override_containing_literal_braces_does_not_raise(self):
        """Three call sites used str.format before, which raises on any literal brace —
        a JSON example in an override was a runtime failure. replace() cannot."""
        out = resolve('Return {"a": 1} for {question}', "rag/complexity.j2", question="Q")
        self.assertEqual('Return {"a": 1} for Q', out)

    def test_an_unused_placeholder_is_left_alone(self):
        self.assertEqual("only {question}", resolve("only {question}", "rag/complexity.j2"))


class MigratedTemplateTests(unittest.TestCase):
    """Every prompt that moved out of profile YAML still substitutes its payload."""

    CASES = [
        ("rag/evidence_grade.j2", {"question": "Q", "context": "C"}),
        ("rag/complexity.j2", {"question": "Q"}),
        ("rag/rewrite.j2", {"query": "Q"}),
        ("agent/persistent_note.j2", {"max_chars": 500}),
        ("agent/resume_answer.j2", {}),
        ("assets/figure_extraction.j2", {"context": "C"}),
        ("assets/entity_extraction.j2", {"attributes": "A", "context": "C"}),
        ("assets/figure_read.j2", {"context": "C", "question": "Q"}),
    ]

    def test_each_renders_with_its_documented_context(self):
        for name, context in self.CASES:
            with self.subTest(template=name):
                out = render(name, **context)
                self.assertTrue(out.strip(), f"{name} rendered empty")
                for value in context.values():
                    self.assertIn(str(value), out)

    def test_none_leaks_an_unrendered_placeholder(self):
        """A `{name}` surviving into output means a template kept the old str.format
        syntax and is silently shipping a literal placeholder to the model."""
        import re

        for name, context in self.CASES:
            with self.subTest(template=name):
                leftover = re.findall(r"\{[a-z_]+\}", render(name, **context))
                self.assertEqual([], leftover)


class LanguageDirectiveTests(unittest.TestCase):
    """Naming the language is only safe where the detector positively established it.

    `detect_language` is a two-way Arabic-script test: `en` is what it returns when it
    did NOT find Arabic, so Chinese and French both come back `en`. The base profile
    ships Chinese fast-path markers, so that is a live case — and telling a Chinese
    user to "reply in English" would be worse than saying nothing.
    """

    def setUp(self):
        self.profile = load_profile("supermew")

    def test_arabic_is_named_because_it_is_positively_detected(self):
        prompt = self.profile.render_system_prompt(None, "ar")
        self.assertIn("Reply in Arabic.", prompt)
        self.assertNotIn("the language the user wrote in", prompt)

    def test_english_keeps_the_generic_instruction(self):
        """`en` is the detector's fallback, not a detection, so it must not be named."""
        prompt = self.profile.render_system_prompt(None, "en")
        self.assertIn("Reply in the language the user wrote in.", prompt)
        self.assertNotIn("Reply in English", prompt)

    def test_an_unknown_or_absent_language_keeps_the_generic_instruction(self):
        for code in (None, "", "zh", "fr"):
            with self.subTest(language=code):
                prompt = self.profile.render_system_prompt(None, code)
                self.assertIn("Reply in the language the user wrote in.", prompt)

    def test_the_plan_carries_the_language_to_the_agent(self):
        """The prompt can only name a language if the plan brought it this far."""
        from backend.chat.signals import RequestSignals
        from backend.chat.turn_policy import resolve_turn

        profile = load_profile("base")
        plan = resolve_turn(
            RequestSignals(question="ما هي الرسوم؟", language="ar"),
            agent_config=profile.agent,
            copy_config=profile.user_copy,
            rag_config=profile.rag,
        )
        self.assertEqual("ar", plan.language)


class ToolResultEnvelopeTests(unittest.TestCase):
    """Class B routing: what the tool says back depends on what retrieval returned,
    which the system prompt cannot know because it is built before the tool runs."""

    TEMPLATE = "tools/knowledge_result.j2"

    def test_each_outcome_leads_with_its_sentinel(self):
        expected = {
            "call_limit": "TOOL_CALL_LIMIT_REACHED:",
            "needs_clarification": "NEEDS_CLARIFICATION:",
            "needs_scope_selection": "NEEDS_SCOPE_SELECTION:",
            "retrieval_error": "RETRIEVAL_ERROR:",
            "no_knowledge": "NO_KNOWLEDGE:",
        }
        for outcome, sentinel in expected.items():
            with self.subTest(outcome=outcome):
                out = render(self.TEMPLATE, outcome=outcome, prompt="p", options=[])
                self.assertTrue(out.startswith(sentinel), out[:60])

    def test_sentences_are_not_hard_wrapped(self):
        """A sentence broken across a real newline costs tokens and silently defeats
        any downstream match on a phrase that now straddles the break."""
        out = render(self.TEMPLATE, outcome="retrieval_error")
        self.assertIn("temporary technical issue", out)
        self.assertIn("Do NOT claim the knowledge base lacks this information", out)

    def test_the_rewrite_caveat_is_paid_only_when_a_rewrite_happened(self):
        without = render(self.TEMPLATE, outcome="chunks", chunks="[1] a", rewritten=False)
        with_caveat = render(self.TEMPLATE, outcome="chunks", chunks="[1] a", rewritten=True)
        self.assertNotIn("retrieval aids", without)
        self.assertIn("retrieval aids, not evidence", with_caveat)
        self.assertLess(len(without), len(with_caveat))

    def test_scope_options_appear_only_when_there_are_any(self):
        with_options = render(self.TEMPLATE, outcome="needs_scope_selection",
                              prompt="Which?", options=["Primary", "Secondary"])
        self.assertIn("Options: Primary; Secondary", with_options)
        self.assertNotIn(
            "Options:",
            render(self.TEMPLATE, outcome="needs_scope_selection", prompt="Which?", options=[]),
        )

    def test_a_figure_bearing_chunk_names_its_asset_id(self):
        """The model can only pass back an id it has actually seen, so this line is
        what makes view_figure callable at all."""
        from backend.tools.knowledge import _format_chunk

        entry = _format_chunk(1, {"filename": "kb.pdf", "page_number": 2,
                                  "text": "t", "asset_ids": ["kb.pdf::p2::img0"]})
        self.assertIn("view_figure asset_id: kb.pdf::p2::img0", entry)
        self.assertNotIn(
            "view_figure",
            _format_chunk(1, {"filename": "kb.pdf", "page_number": 2, "text": "t"}),
        )


class RegistryDriftTests(unittest.TestCase):
    def test_grounded_tools_are_all_real_tools(self):
        """GROUNDED_TOOLS sits beside TOOL_BUILDERS so a fifth tool has to be
        classified. A name here that no longer exists would silently stop grounding."""
        self.assertEqual(set(), GROUNDED_TOOLS - set(TOOL_BUILDERS))


if __name__ == "__main__":
    unittest.main()
