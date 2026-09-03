"""What the agent's system prompt costs, and what it must still contain at that price.

The school profile overrides `agent.system_prompt` to carry an Egyptian-Arabic register.
That text is paid on EVERY turn, and Arabic costs roughly three tokens a word on this
tokenizer — the first draft measured 588 tokens against a 500 budget while looking, in
the file, like a short block. Character counts are not a usable proxy, so the ceiling is
checked with the real tokenizer or not at all.

A budget on its own is only half the test. Trimming to a number is easy if nothing
checks WHAT was trimmed, and the two things most likely to be cut are the two that
matter most: the citation contract (its loss breaks asset delivery silently, because
attribution parses [n] out of the answer) and the worked examples (a small model cannot
instantiate a register from adjectives). Both are asserted below alongside the count.
"""
import unittest

import tiktoken

from backend.profiles.registry import load_profile

#: The gpt-oss / modern OpenAI-family encoding. The deployment's MODEL is
#: openai/gpt-oss-20b; this is the closest public tokenizer to what it bills on, and it
#: is the same one either way for the comparison that matters — Arabic vs English cost.
ENCODING = "o200k_base"

#: Rendered ceiling for the school profile, in tokens. Every turn pays it.
#:
#: Raised from 500 when the worked examples were labelled. A model that cannot tell a
#: style example from real context reuses the example's CONTENT: this profile shipped a
#: worked answer carrying a grade and a fee, and the model reproduced both verbatim —
#: as its answer and as its search query — for accounts with no child on file. The
#: labels («أمثلة أسلوب، مش أسئلة ولا معلومات», and «س»/«ج» on each pair) are what say
#: "illustration" rather than "context", and they cost about 28 tokens a turn. That is
#: the trade: a fixed, small, per-turn cost against a fabricated fee reaching a parent.
#:
#: 505 is deliberately close to what the prompt actually renders (500). The labels are
#: not to be trimmed to buy headroom — shortening them is what made the examples
#: mistakable for context in the first place. Anything that needs more room here should
#: come out of the prose, or move the ceiling on purpose.
SCHOOL_PROMPT_BUDGET = 505


def count(text: str) -> int:
    return len(tiktoken.get_encoding(ENCODING).encode(text))


class SchoolPromptBudgetTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile("school")
        self.prompt = self.profile.render_system_prompt(
            ["search_knowledge_base", "view_figure", "get_student_records"], language="ar"
        )

    def test_the_rendered_prompt_fits_the_budget(self):
        tokens = count(self.prompt)
        self.assertLessEqual(
            tokens,
            SCHOOL_PROMPT_BUDGET,
            f"the system prompt is {tokens} tokens, over the {SCHOOL_PROMPT_BUDGET} budget "
            f"by {tokens - SCHOOL_PROMPT_BUDGET}. It is paid on every turn.",
        )

    def test_the_budget_is_not_silently_slack(self):
        """A ceiling nobody is near stops being a ceiling. If this fails the prompt has
        shrunk a lot — either something load-bearing was dropped, or the budget should
        come down to match."""
        self.assertGreater(count(self.prompt), SCHOOL_PROMPT_BUDGET // 2)

    def test_an_english_turn_does_not_pay_for_the_arabic_register(self):
        """The register only works written in Arabic, which makes it dead weight on an
        English turn — and it is more than half the prompt."""
        english = self.profile.render_system_prompt(
            ["search_knowledge_base"], language="en"
        )
        self.assertLess(count(english), count(self.prompt) // 2)
        self.assertNotIn("حضرتك", english, "the Arabic register leaked into an English turn")

    def test_an_undetected_language_is_treated_as_not_arabic(self):
        """`detect_language` returns "en" for anything it cannot place as Arabic, and a
        turn nobody classified must not be handed Egyptian style rules."""
        for language in ("", None, "fr"):
            with self.subTest(language=language):
                rendered = self.profile.render_system_prompt(
                    ["search_knowledge_base"], language=language
                )
                self.assertNotIn("حضرتك", rendered)

    def test_the_language_directive_names_the_language_outright(self):
        """The override bypasses the shipped template, so this is the only thing telling
        the model which language THIS turn is.

        Both languages are NAMED on this profile. Measured on gpt-oss-20b, the generic
        "Reply in the language the user wrote in" left 3 of 4 English questions answered
        in Arabic — the persona mentions Arabic and everything around the turn is
        Arabic-first, so a directive that does not say "English" loses to the context.
        Naming it took that to 4 of 4 correct.
        """
        self.assertIn("Reply in Arabic.", self.prompt)
        self.assertIn(
            "Reply in English.",
            self.profile.render_system_prompt(["search_knowledge_base"], language="en"),
        )

    def test_naming_english_is_scoped_to_this_profile(self):
        """`en` is what the detector returns for anything it could not place as Arabic,
        so naming English is only safe where the set of languages is closed. A profile
        that has not said so keeps the generic directive."""
        generic = load_profile("supermew").render_system_prompt(
            ["search_knowledge_base"], language="en"
        )
        self.assertIn("Reply in the language the user wrote in.", generic)
        self.assertNotIn("Reply in English.", generic)

    def test_the_grounding_contract_holds_in_both_languages(self):
        """Splitting the prompt must not have taken the citation rule with it."""
        for language in ("ar", "en"):
            with self.subTest(language=language):
                rendered = self.profile.render_system_prompt(
                    ["search_knowledge_base"], language=language
                )
                self.assertIn("[1], or [2][3]", rendered)

    def test_the_cost_does_not_depend_on_the_bound_tools(self):
        """`system_prompt` is used verbatim, so per-turn tool narrowing does not apply.
        Worth pinning: it is the trade this override makes, and language is now the only
        thing that changes the prompt."""
        sizes = {
            count(self.profile.render_system_prompt(tools, language="ar"))
            for tools in (["search_knowledge_base"], ["view_figure"], None)
        }
        self.assertEqual(1, len(sizes), f"the prompt varies by bound tools: {sizes}")

    def test_the_citation_contract_survived_the_trim(self):
        """The first thing a token trim deletes, and the one with a silent failure mode:
        asset delivery parses [n] out of the answer, so losing this line stops figures
        being attached at all rather than merely losing provenance."""
        self.assertIn("[1], or [2][3]", self.prompt)
        self.assertIn("retrieved chunks", self.prompt)

    def test_the_register_instruction_is_in_arabic(self):
        """The cheapest thing in the prompt that actually works. An English sentence
        saying "reply in Egyptian Arabic" barely moves a 20B English-centric model;
        the same instruction in Arabic anchors the output distribution."""
        self.assertIn("الرد يبقى مصري مهذّب", self.prompt)

    def test_the_lexical_anchors_survived(self):
        """Concrete words, not adjectives. Small models act on «استخدم حضرتك» and do
        very little with "be warm but respectful"."""
        for marker in ["حضرتك", "عايز", "إمتى", "كام", "دلوقتي"]:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.prompt)

    def test_the_register_is_bounded_from_both_sides(self):
        """Small models overshoot in one direction. Showing only the target leaves which
        direction unsaid, so the too-formal and too-casual poles both have to stay."""
        self.assertIn("يُرجى", self.prompt, "the too-formal pole is missing")
        self.assertIn("يا باشا", self.prompt, "the too-casual pole is missing")
        self.assertIn("المطلوب", self.prompt, "the target is missing")

    def test_at_least_two_worked_examples_survived(self):
        """Register is imitative. Examples are the most expensive thing here and the
        first candidate for a trim, which is exactly why the floor is asserted."""
        question_marks = self.prompt.count("؟")
        self.assertGreaterEqual(question_marks, 2, "fewer than two worked examples remain")

    def test_facts_are_exempted_from_the_dialect(self):
        """Asking a small model for dialect and not saying this produces loosely
        restated amounts and dates."""
        self.assertIn("الأرقام والتواريخ", self.prompt)


class ArabicCostsMoreThanEnglishTests(unittest.TestCase):
    """Why the budget is measured rather than eyeballed."""

    def test_the_same_sentence_costs_more_in_arabic(self):
        english = "The fees for grade four are 45,000 pounds paid over three instalments."
        arabic = "مصاريف الصف الرابع 45 ألف جنيه على تلات دفعات."
        self.assertGreater(
            count(arabic) / max(len(arabic.split()), 1),
            count(english) / max(len(english.split()), 1),
            "Arabic is expected to cost more tokens per word on this encoding",
        )


class OtherProfilesStayOnTheTemplateTests(unittest.TestCase):
    """Only school pays this. The others compose per turn and cost far less."""

    def test_the_template_profiles_are_cheaper_than_the_override(self):
        template = load_profile("supermew").render_system_prompt(["search_knowledge_base"])
        school = load_profile("school").render_system_prompt(["search_knowledge_base"])
        self.assertLess(count(template), count(school))


if __name__ == "__main__":
    unittest.main()
