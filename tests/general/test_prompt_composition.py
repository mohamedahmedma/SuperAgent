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
#
# It was briefly raised to 1200 to admit an expanded grounding contract. That rewrite
# was rolled back for making answers slower and more caveat-heavy, and the budget came
# back with it. Raising this is the signal that a per-turn cost is growing; if you need
# to raise it, say why here.
SYSTEM_PROMPT_CHAR_BUDGET = 600


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile("supermew")

    def test_a_grounded_turn_gets_the_citation_contract(self):
        prompt = self.profile.render_system_prompt(["search_knowledge_base"])
        self.assertIn("[1], or [2][3]", prompt)
        self.assertIn("retrieved chunks", prompt)

    def test_a_turn_with_nothing_to_cite_does_not_pay_for_citations(self):
        """The whole point of composing: a turn bound only to an ungrounded tool has no
        chunks and no citations, so every word about them would be waste on every one of
        its turns."""
        prompt = self.profile.render_system_prompt(["get_student_records"])
        self.assertNotIn("[1]", prompt)
        self.assertNotIn("Grounding rules", prompt)
        self.assertLess(len(prompt), len(self.profile.render_system_prompt(["search_knowledge_base"])))

    def test_every_grounded_tool_triggers_the_contract_on_its_own(self):
        for name in sorted(GROUNDED_TOOLS):
            with self.subTest(tool=name):
                self.assertIn("[1], or [2][3]", self.profile.render_system_prompt([name]))

    def test_the_persona_always_opens_the_prompt(self):
        for tools in (None, ["search_knowledge_base"], ["get_student_records"], []):
            with self.subTest(tools=tools):
                self.assertTrue(
                    self.profile.render_system_prompt(tools).startswith(
                        self.profile.identity.persona
                    )
                )

    def test_style_rules_survive_every_turn_shape(self):
        """Language and output shape are not tool-conditional — an ungrounded turn
        still has to answer in the user's language."""
        for tools in (None, ["search_knowledge_base"], ["get_student_records"], []):
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
        """A profile that sets system_prompt opts out of per-turn narrowing, so these
        leave it alone.

        `school` is the deliberate exception — it needs the whole agent prompt to carry
        an Egyptian-Arabic register that must not leak into the classifier prompts via
        persona. What that costs it is pinned by SchoolOverrideKeepsTheContractTests.
        """
        for name in ("base", "supermew", "document_kb", "ecommerce"):
            with self.subTest(profile=name):
                self.assertEqual("", load_profile(name).agent.system_prompt)


class SchoolOverrideKeepsTheContractTests(unittest.TestCase):
    """school.yaml sets `agent.system_prompt`, which bypasses agent/system.j2 entirely.

    That is a real trade, taken so the Arabic register reaches the agent WITHOUT going
    through `identity.persona` — persona is also rendered into request_envelope,
    resolve_question, scope_check, section_summary and corpus_digest, where style
    examples would be billed on every classifier call.

    The cost is that agent/_grounding.j2 no longer composes in, so its rules are copied
    into the YAML. These tests exist because that copy can go stale silently, and the
    symptom is not a missing sentence: asset delivery parses [n] out of the answer, so
    losing the citation line stops figures being attached at all.
    """

    def setUp(self):
        self.profile = load_profile("school")
        # language="ar": the Egyptian register is appended only on an Arabic turn, so
        # rendering without a language would test a prompt that deliberately omits it.
        self.prompt = self.profile.render_system_prompt(
            ["search_knowledge_base"], language="ar"
        )

    def test_every_grounding_rule_survives_the_override(self):
        """Compared against the fragment itself, so editing _grounding.j2 without
        updating school.yaml fails here rather than in production."""
        from backend.prompts import render

        fragment = render("agent/_grounding.j2")
        rules = [line.strip() for line in fragment.splitlines() if line.strip().startswith("-")]
        self.assertTrue(rules, "the grounding fragment has no rules to check")
        for rule in rules:
            with self.subTest(rule=rule):
                self.assertIn(rule, self.prompt)

    def test_the_persona_is_still_substituted(self):
        self.assertNotIn("{persona}", self.prompt)
        self.assertIn(self.profile.identity.persona[:40], self.prompt)

    def test_the_register_reaches_the_agent(self):
        """حضرتك is the marker that makes the register semi-formal rather than either
        stiff MSA or over-familiar slang."""
        self.assertIn("حضرتك", self.prompt)

    def test_the_register_does_not_reach_the_classifier_prompts(self):
        """The reason this lives in system_prompt rather than persona."""
        self.assertNotIn("حضرتك", self.profile.identity.persona)


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
        out = resolve("", "rag/evidence_grade.j2", question="Q", context="C", constraints=[])
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
        ("rag/evidence_grade.j2",
         {"question": "Q", "context": "C", "constraints": ["CONDITION"]}),
        ("rag/complexity.j2", {"question": "Q"}),
        ("rag/rewrite.j2", {"query": "Q"}),
        ("agent/persistent_note.j2", {"max_chars": 500}),
        ("agent/resume_answer.j2", {}),
        ("assets/figure_extraction.j2", {"context": "C"}),
        ("assets/entity_extraction.j2", {"attributes": "A", "context": "C"}),
    ]

    def test_each_renders_with_its_documented_context(self):
        for name, context in self.CASES:
            with self.subTest(template=name):
                out = render(name, **context)
                self.assertTrue(out.strip(), f"{name} rendered empty")
                for value in context.values():
                    # A list payload is joined into the prompt, never repr'd, so it is
                    # its ELEMENTS that have to survive substitution.
                    for expected in (value if isinstance(value, list) else [value]):
                        self.assertIn(str(expected), out)

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

    def _chunks(self, **flags):
        fields = {
            "outcome": "chunks",
            "chunks": "[1] a",
            "rewritten": False,
            "partial": False,
            "constraints": [],
            "discriminate": "unknown",
            "figures": False,
        }
        fields.update(flags)
        return render(self.TEMPLATE, **fields)

    def test_carried_conditions_are_stated_only_when_the_turn_has_any(self):
        """Retrieval widens the query with a condition but cannot enforce it — a search
        for fees "up to Year 6" still returns the whole fee table. So the narrowing has
        to reach the model that writes the answer, and only on the turns that carry one."""
        without = self._chunks(constraints=[])
        with_scope = self._chunks(constraints=["grades up to Year 6"], discriminate="yes")
        self.assertNotIn("SCOPE OF THE ANSWER", without)
        self.assertIn("SCOPE OF THE ANSWER", with_scope)
        self.assertIn("grades up to Year 6", with_scope)
        self.assertIn("leave the other cases out", with_scope)

    def test_material_that_does_not_vary_is_answered_in_full(self):
        """The failure this branch exists for: a single admissions document list does
        not vary by year group, and telling the model to answer "only for what the
        conditions cover" made it refuse an answer sitting in chunk 1."""
        out = self._chunks(constraints=["grades up to Year 6"], discriminate="no")
        self.assertIn("does not vary by them", out)
        self.assertIn("Give it in full", out)
        self.assertIn("Do NOT withhold it", out)
        self.assertNotIn("leave the other cases out", out)

    def test_an_undecided_verdict_still_forbids_refusing(self):
        out = self._chunks(constraints=["grades up to Year 6"], discriminate="unknown")
        self.assertIn("Answer either way", out)
        self.assertIn("never refuse", out)

    def test_every_verdict_names_the_conditions(self):
        for verdict in ("yes", "no", "unknown"):
            with self.subTest(discriminate=verdict):
                out = self._chunks(constraints=["girls only"], discriminate=verdict)
                self.assertIn("girls only", out)

    def test_the_figure_rule_is_paid_only_when_a_figure_was_retrieved(self):
        """Rung 3 of the ladder in backend/prompts/__init__.py: an instruction that is
        only true when retrieval returned a figure is billed only on those turns."""
        without = self._chunks(figures=False)
        with_rule = self._chunks(figures=True)
        self.assertNotIn("[FIGURE]", without)
        self.assertIn("[FIGURE]", with_rule)
        self.assertLess(len(without), len(with_rule))

    def test_the_figure_rule_makes_the_citation_the_selector(self):
        """This is the whole image feature at query time: no tool to look at a picture
        and no second call to choose one, just the `[n]` the agent already emits. That
        only works if the model is told the marker SELECTS rather than labels — without
        it, it cites the prose beside a figure as readily as the figure itself, and the
        answer describes an image nobody attached."""
        out = self._chunks(figures=True)
        self.assertIn("whenever you cite that chunk", out)
        self.assertIn("Cite the [FIGURE] chunk you actually described", out)

    def test_the_figure_rule_still_forbids_writing_the_picture(self):
        """The model has no id to write any more, but it can still invent a URL."""
        out = self._chunks(figures=True)
        for forbidden in ("an image", "a link", "a file path"):
            with self.subTest(forbidden=forbidden):
                self.assertIn(forbidden, out)

    def test_the_figure_rule_is_not_hard_wrapped(self):
        out = self._chunks(figures=True)
        self.assertIn("the image itself is shown to the user whenever you cite", out)

    def test_the_figure_rule_reaches_no_other_outcome(self):
        """A refusal or a clarification shows the model no chunk headers at all, so it
        has no asset_id to write and no rule to pay for."""
        for outcome in ("no_knowledge", "retrieval_error", "empty", "call_limit"):
            with self.subTest(outcome=outcome):
                out = render(self.TEMPLATE, outcome=outcome, prompt="p", options=[])
                self.assertNotIn("already shown to the user", out)

    def test_the_rewrite_caveat_is_paid_only_when_a_rewrite_happened(self):
        without = self._chunks(rewritten=False)
        with_caveat = self._chunks(rewritten=True)
        self.assertNotIn("retrieval aids", without)
        self.assertIn("retrieval aids, not evidence", with_caveat)
        self.assertLess(len(without), len(with_caveat))

    def test_partial_evidence_is_told_to_answer_rather_than_refuse(self):
        """The model cannot see the grade, and its default reading of "this doesn't
        fully answer the question" is to refuse. Paid only when the grader said
        partial, which is why it is here and not in the system prompt."""
        without = self._chunks(partial=False)
        with_guidance = self._chunks(partial=True)
        self.assertNotIn("PARTIAL_EVIDENCE", without)
        self.assertIn("PARTIAL_EVIDENCE", with_guidance)
        self.assertIn("Answer anyway, from what they do establish", with_guidance)
        self.assertIn("Refusing here is the wrong outcome", with_guidance)

    def test_scope_options_appear_only_when_there_are_any(self):
        with_options = render(self.TEMPLATE, outcome="needs_scope_selection",
                              prompt="Which?", options=["Primary", "Secondary"])
        self.assertIn("Options: Primary; Secondary", with_options)
        self.assertNotIn(
            "Options:",
            render(self.TEMPLATE, outcome="needs_scope_selection", prompt="Which?", options=[]),
        )

    def test_a_figure_bearing_chunk_is_marked_but_never_identified(self):
        """WHETHER, not which. The marker is what lets the model choose a picture with
        its citation; the id would only give it something to write into the answer."""
        from backend.tools.knowledge import _format_chunk

        entry = _format_chunk(1, {"filename": "kb.pdf", "page_number": 2,
                                  "text": "t", "asset_ids": ["kb.pdf::p2::img0"]})
        self.assertIn("[FIGURE]", entry)
        self.assertNotIn("kb.pdf::p2::img0", entry)
        self.assertNotIn(
            "[FIGURE]",
            _format_chunk(1, {"filename": "kb.pdf", "page_number": 2, "text": "t"}),
        )


class CachingContractTests(unittest.TestCase):
    """Static text before variable text, in every prompt that has both.

    Providers cache on a shared prefix. A variable placed above the instructions
    invalidates everything after it on every call, which silently turns the whole
    instruction block into tokens paid in full every time.
    """

    # template -> the variables it substitutes.
    #
    # complexity.j2 and figure_extraction.j2 are deliberately absent. Both interleave
    # their payload with instructions, so neither holds this property today. Fixing
    # that is a pure reordering, but reordering changes the prompt the model receives,
    # and prompt wording is currently frozen pending an eval set — so they are excluded
    # rather than quietly rewritten. Add them here when their wording is next revisited.
    PAYLOADS = {
        "rag/evidence_grade.j2": {"question": "Q", "context": "C", "constraints": []},
        "rag/rewrite.j2": {"query": "Q"},
        "assets/entity_extraction.j2": {"attributes": "A", "context": "C"},
    }

    # Distinctive on purpose. Single letters collide with the instruction text itself —
    # "Q" matches the "QUESTION" in the grader's procedure — which makes a naive
    # .index() report a variable far earlier than it really appears.
    SENTINEL = "ZZPAYLOADZZ"

    def _filled(self, context, suffix=""):
        """Sentinel-fill the string payloads, pass anything else through.

        A non-string value here selects a BRANCH rather than carrying payload —
        `constraints` decides whether the grader is told about carried conditions at
        all. Filling it with a sentinel string would both take the wrong branch and
        make the list render as its own characters.
        """
        return {
            key: (f"{self.SENTINEL}{key}{suffix}" if isinstance(value, str) else value)
            for key, value in context.items()
        }

    def test_nothing_static_follows_the_variable_payload(self):
        """Rendered twice with different payloads, the shared prefix must cover
        everything up to the first substitution."""
        for name, context in self.PAYLOADS.items():
            with self.subTest(template=name):
                short = render(name, **self._filled(context))
                long = render(name, **self._filled(context, "y" * 40))
                prefix = 0
                while prefix < min(len(short), len(long)) and short[prefix] == long[prefix]:
                    prefix += 1
                tail = short[prefix:]
                # Whatever follows the first difference is payload plus its labels; no
                # instruction paragraph should be stranded down there.
                self.assertLess(
                    len(tail), 150,
                    f"{name}: {len(tail)} chars sit after the first variable",
                )

    def test_the_instruction_block_is_the_bulk_of_the_prefix(self):
        for name, context in self.PAYLOADS.items():
            with self.subTest(template=name):
                out = render(name, **self._filled(context))
                self.assertGreater(out.index(self.SENTINEL), len(out) // 2)

    def test_the_grader_keeps_the_contract_when_conditions_are_carried(self):
        """The conditions branch adds ~1.5K of instruction. Placed below the question —
        where it started — every one of those characters sits under the cache boundary
        and is paid in full on each constrained turn. The instruction belongs in the
        prefix; only the condition VALUES belong with the payload."""
        payload = {"question": "Q", "context": "C", "constraints": ["grades up to Year 6"]}
        short = render("rag/evidence_grade.j2", **self._filled(payload))
        long = render("rag/evidence_grade.j2", **self._filled(payload, "y" * 40))
        prefix = 0
        while prefix < min(len(short), len(long)) and short[prefix] == long[prefix]:
            prefix += 1
        self.assertLess(len(short[prefix:]), 150, short[prefix:])
        self.assertIn("constraints_discriminate", short[:prefix],
                      "the conditions instruction must sit inside the cacheable prefix")


class GroundingContractTests(unittest.TestCase):
    """One faithfulness contract, two consumers.

    The agent path and the HITL-resume path answer the same kind of question from the
    same kind of evidence. When each carried its own wording they were two contracts
    that could drift, and nothing downstream could tell which produced an answer.
    """

    RULES = [
        "Base every factual claim on the retrieved chunks",
        "Cite the chunk behind each claim",
        "Say plainly when the retrieved context does not answer",
    ]

    def test_both_paths_carry_the_identical_contract(self):
        fragment = render("agent/_grounding.j2")
        agent = load_profile("base").render_system_prompt(["search_knowledge_base"])
        resume = render("agent/resume_answer.j2")
        for surface, label in ((agent, "system.j2"), (resume, "resume_answer.j2")):
            with self.subTest(prompt=label):
                self.assertIn(fragment, surface)

    def test_every_rule_survives_rendering(self):
        fragment = render("agent/_grounding.j2")
        for rule in self.RULES:
            with self.subTest(rule=rule):
                self.assertIn(rule, fragment)

    def test_an_ungrounded_turn_carries_no_contract(self):
        """Nothing to cite, so the rules would be noise on every one of its turns."""
        prompt = load_profile("base").render_system_prompt(["get_student_records"])
        self.assertNotIn("Cite the chunk", prompt)


class RegistryDriftTests(unittest.TestCase):
    def test_grounded_tools_are_all_real_tools(self):
        """GROUNDED_TOOLS sits beside TOOL_BUILDERS so a fifth tool has to be
        classified. A name here that no longer exists would silently stop grounding."""
        self.assertEqual(set(), GROUNDED_TOOLS - set(TOOL_BUILDERS))


if __name__ == "__main__":
    unittest.main()
