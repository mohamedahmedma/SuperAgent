"""The school eval dataset is data, and data rots quietly.

`tests/evals/school_retrieval_eval.py` is not run by pytest — it needs Milvus and the
corpus indexed. The dataset it reads is a plain module, though, and a mistake in it does
not fail: a case with no gold silently passes every stage, a duplicate id silently
overwrites another case's result, and a gold span carrying Word's en dash silently never
matches. Each of those makes the score better, which is the direction a measurement must
never be allowed to rot in.

So the oracle gets the same treatment as the code it judges.
"""
import unittest

from tests.evals.school_dataset import CASES, DATASET_VERSION, cases, missing, satisfied

#: Characters Word writes that the indexed text may not carry verbatim. A gold span
#: holding one of these matches nothing, and the case reads as a retrieval failure.
TYPOGRAPHIC = "–—‘’“” "

ARABIC = range(0x0600, 0x0700)


def _spans(case):
    for item in case.required:
        yield from (item if isinstance(item, tuple) else (item,))


class TheDatasetIsWellFormed(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [case.id for case in CASES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_scored_case_states_its_gold(self):
        """Only a figure case (its text is vision output) or an unanswerable one may omit it."""
        for case in CASES:
            if case.modality == "figure" or case.kind == "unanswerable":
                continue
            with self.subTest(case=case.id):
                self.assertTrue(case.required, "a case with no gold passes every stage")

    def test_unanswerable_cases_claim_no_evidence(self):
        for case in CASES:
            if case.kind == "unanswerable":
                with self.subTest(case=case.id):
                    self.assertEqual((), tuple(case.required))
                    self.assertTrue(case.note, "say why the corpus cannot answer it")

    def test_follow_ups_carry_the_turn_before_them(self):
        """A follow-up without its context is just a vague question with a strict oracle."""
        for case in CASES:
            if case.kind == "followup":
                with self.subTest(case=case.id):
                    self.assertTrue(case.context)

    def test_questions_are_in_arabic(self):
        """The corpus is English and the parents are not. That gap is the thing being
        measured, so a question that slipped through in English would flatter the score."""
        for case in CASES:
            with self.subTest(case=case.id):
                self.assertTrue(
                    any(ord(char) in ARABIC for char in case.question),
                    f"{case.id} is not an Arabic question",
                )

    def test_gold_spans_avoid_typography_that_will_not_match(self):
        for case in CASES:
            for span in _spans(case):
                with self.subTest(case=case.id, span=span):
                    self.assertFalse(
                        [ch for ch in span if ch in TYPOGRAPHIC],
                        "use plain ASCII punctuation in gold spans",
                    )

    def test_gold_spans_are_specific_enough_to_mean_something(self):
        for case in CASES:
            for span in _spans(case):
                with self.subTest(case=case.id, span=span):
                    self.assertGreaterEqual(len(span.strip()), 3)

    def test_the_holdout_is_a_real_minority(self):
        held = len(cases("holdout"))
        self.assertEqual(len(CASES), held + len(cases("dev")))
        self.assertTrue(0.1 <= held / len(CASES) <= 0.35, f"holdout is {held}/{len(CASES)}")

    def test_the_version_is_stated(self):
        self.assertRegex(DATASET_VERSION, r"^\d{4}-\d{2}-\d{2}\.\d+$")


class TheScorerMeansAllOfTheEvidence(unittest.TestCase):
    """`missing` is what separates this harness from a hit-rate one — see the module."""

    def test_a_tuple_is_satisfied_by_any_member(self):
        self.assertTrue(satisfied(("alpha", "beta"), "... beta ..."))
        self.assertFalse(satisfied(("alpha", "beta"), "gamma"))

    def test_a_string_must_appear(self):
        self.assertTrue(satisfied("105,000 EGP", "the fee is 105,000 EGP a year"))
        self.assertFalse(satisfied("105,000 EGP", "the fee is 115,000 EGP a year"))

    def test_matching_ignores_case(self):
        self.assertTrue(satisfied("White Rose", "we use the white rose scheme"))

    def test_a_partly_answered_case_is_still_missing_evidence(self):
        case = next(c for c in CASES if len(c.required) > 1 and c.kind == "multi_part")
        first = next(iter(_spans(case)))
        self.assertTrue(missing(case, first), "half the evidence must not count as a pass")


if __name__ == "__main__":
    unittest.main()
