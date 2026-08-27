"""Complexity fast path — the gate in front of sub-question decomposition.

Getting this wrong is expensive in a way that is invisible from the outside: a
question that misses the fast path reaches the planner, which may split it into up to
four sub-questions, and EACH sub-question pays its own retrieval plus its own grader
call. A plain lookup can quietly cost five model calls instead of one.
"""
import unittest

import backend.rag.pipeline as pipeline


class SingularAndPluralTests(unittest.TestCase):
    """Plural interrogatives were missing entirely, so "what IS the partner" was
    fast-pathed while "what ARE the partners" went to the planner and got decomposed."""

    def _simple(self, question):
        return pipeline._simple_question_fast_path_reason(question) is not None

    def test_singular_and_plural_are_treated_alike(self):
        pairs = [
            ("what is the partner with the school", "what are the partners with the school"),
            ("who is the school partner", "who are the school partners"),
            ("where is the campus", "where are the campuses"),
            ("when is the term", "when are the terms"),
        ]
        for singular, plural in pairs:
            with self.subTest(pair=(singular, plural)):
                self.assertTrue(self._simple(singular))
                self.assertTrue(self._simple(plural))

    def test_a_list_answer_does_not_make_a_question_complex(self):
        """One passage answers all of these; decomposing them buys nothing."""
        for question in (
            "what are the partner organisations",
            "which subjects are taught",
            "what are the school facilities",
        ):
            with self.subTest(question=question):
                self.assertTrue(self._simple(question))

    def test_a_wh_question_opening_on_a_copula_is_simple(self):
        """Caught by the pattern rather than a literal marker — these phrasings are
        open-ended, so enumerating every one of them in the vocabulary is hopeless."""
        for question in ("what were the results", "when does the term start", "who does the transport"):
            with self.subTest(question=question):
                self.assertEqual(
                    "obvious_simple_fast_path:wh_attribute_question",
                    pipeline._simple_question_fast_path_reason(question),
                )


class OverrideMarkerTests(unittest.TestCase):
    """Some single-fact phrasings contain a substring that also opens a complex
    question. The override list must beat the complex list, not lose to it."""

    def _reason(self, question):
        return pipeline._simple_question_fast_path_reason(question)

    def test_how_many_beats_the_how_complex_marker(self):
        self.assertIn("how ", pipeline._COMPLEX_QUERY_MARKERS)
        self.assertEqual(
            "obvious_simple_fast_path:single_fact_override",
            self._reason("how many students are there"),
        )

    def test_how_much_and_how_long_are_lookups(self):
        for question in ("how much is the fee", "how long is the term", "how old must a child be"):
            with self.subTest(question=question):
                self.assertIsNotNone(self._reason(question))

    def test_a_genuinely_analytical_how_still_reaches_the_planner(self):
        for question in ("how do i apply for a place", "how does the transport work"):
            with self.subTest(question=question):
                self.assertIsNone(self._reason(question))


class ComplexQuestionsStillDecomposeTests(unittest.TestCase):
    """Widening the fast path must not disable decomposition for questions that
    genuinely need it — that was the point of the feature."""

    def _reason(self, question):
        return pipeline._simple_question_fast_path_reason(question)

    def test_comparisons_reach_the_planner(self):
        for question in (
            "compare fees",
            "what is the difference between grade 5 and 6",
            "pros and cons of the transport",
        ):
            with self.subTest(question=question):
                self.assertIsNone(self._reason(question))

    def test_causal_and_procedural_questions_reach_the_planner(self):
        for question in ("why are the fees increasing", "how do i register my child"):
            with self.subTest(question=question):
                self.assertIsNone(self._reason(question))

    def test_multi_intent_questions_reach_the_planner(self):
        self.assertIsNone(self._reason("what are the fees and the term dates"))

    def test_long_questions_reach_the_planner(self):
        self.assertIsNone(self._reason("x" * (pipeline._RAG.fast_path_max_chars + 1)))


class ArabicBehaviourTests(unittest.TestCase):
    """The corpus is Arabic-first; widening the English rules must not regress it."""

    def _reason(self, question):
        return pipeline._simple_question_fast_path_reason(question)

    def test_arabic_single_fact_questions_fast_path(self):
        for question in ("ما هي الرسوم؟", "متى التسجيل؟", "أين المدرسة؟", "كم عدد الطلاب"):
            with self.subTest(question=question):
                self.assertIsNotNone(self._reason(question))

    def test_arabic_analytical_questions_reach_the_planner(self):
        for question in ("قارن الرسوم", "ما الفرق بين الصفين؟", "لماذا الرسوم؟", "اشرح الشروط"):
            with self.subTest(question=question):
                self.assertIsNone(self._reason(question))


class ProfileWiringTests(unittest.TestCase):
    def test_the_override_list_is_profile_data(self):
        from backend.profiles.registry import load_profile

        markers = load_profile("base").rag.simple_override_markers
        self.assertIn("how many", markers)
        self.assertIn("how much", markers)

    def test_the_planner_prompt_tells_the_model_to_prefer_simple(self):
        """Decomposition costs a retrieval and a grader call per sub-question, so the
        prompt has to say that rather than leaving it to taste."""
        from backend.prompts import render

        prompt = render("rag/complexity.j2", question="q").lower()
        self.assertIn("default to simple", prompt)
        self.assertIn("still simple when its answer happens to be a list", prompt)


if __name__ == "__main__":
    unittest.main()
