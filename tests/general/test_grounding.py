"""Deterministic verification of an answer's figures against retrieved evidence.

Built around the transcript that motivated it: a parent asked what their son's fees
were, the corpus returned nothing, and the assistant answered «مصاريف الصف الرابع 45 ألف
جنيه على تلات دفعات. [1]». Every claim in that sentence was invented, including the
citation. Each test below names which part of it the check catches.

The runtime guard that lets a `no_knowledge` verdict end the turn before the model can
answer around it is the other half of this fix, and it is tested next to the rest of the
middleware in `test_turn_orchestration.py`.
"""
import unittest

from backend.chat.grounding import citation_indices, numeric_claims, verify

FABRICATED = "مصاريف الصف الرابع 45 ألف جنيه على تلات دفعات. [1]"
EVIDENCE_GRADE_1 = ["رسوم الصف الأول الابتدائي للعام 2026: 30,000 جنيه على ثلاث دفعات."]
EVIDENCE_GRADE_4 = ["رسوم الصف الرابع الابتدائي: 45,000 جنيه على ثلاث دفعات."]


class NumberReadingTests(unittest.TestCase):
    """A check defeated by writing the same figure differently would not be a check."""

    def test_the_same_figure_in_every_form_reads_as_one_number(self):
        for text in ("45000", "45,000", "45 ألف", "45 thousand", "٤٥٠٠٠", "45k"):
            with self.subTest(written=text):
                self.assertEqual(numeric_claims(text), {45000.0})

    def test_a_multiplier_only_scales_the_number_it_follows(self):
        self.assertEqual(numeric_claims("45 طالب"), {45.0})

    def test_citation_markers_are_not_figures(self):
        self.assertEqual(numeric_claims("الرسوم مذكورة [3]", floor=0), set())
        self.assertEqual(citation_indices("انظر [1] و [3]"), [1, 3])

    def test_arabic_decimal_and_thousands_separators(self):
        self.assertEqual(numeric_claims("1٬500٫5"), {1500.5})


class GroundingVerdictTests(unittest.TestCase):
    def test_the_fabricated_answer_is_rejected_when_nothing_was_retrieved(self):
        report = verify(FABRICATED, [])
        self.assertFalse(report.ok)
        self.assertTrue(report.cited_without_evidence)
        self.assertIn(45000.0, report.ungrounded)

    def test_the_fabricated_answer_is_rejected_against_the_wrong_grade(self):
        """The exact failure: chunks existed, but not for the figure that was stated."""
        report = verify(FABRICATED, EVIDENCE_GRADE_1)
        self.assertFalse(report.ok)
        self.assertEqual(report.ungrounded, (45000.0,))

    def test_the_same_answer_passes_when_the_evidence_really_says_it(self):
        self.assertTrue(verify(FABRICATED, EVIDENCE_GRADE_4).ok)

    def test_a_citation_beyond_the_retrieved_chunks_is_rejected(self):
        report = verify("الرسوم 30,000 جنيه [4]", EVIDENCE_GRADE_1)
        self.assertFalse(report.ok)
        self.assertEqual(report.invalid_citations, (4,))

    def test_an_answer_with_no_figures_and_no_citations_passes(self):
        self.assertTrue(verify("أهلاً بحضرتك، تحت أمرك.", []).ok)

    def test_counts_below_the_floor_are_not_claims(self):
        """Evidence spells «ثلاث دفعات» in words; an answer writing 3 is not inventing."""
        self.assertTrue(verify("على 3 دفعات [1]", EVIDENCE_GRADE_4).ok)


class DerivationTests(unittest.TestCase):
    """Arithmetic the evidence supports is the model working, not fabricating."""

    def test_a_total_split_into_instalments_is_grounded(self):
        self.assertTrue(verify("كل دفعة 15,000 جنيه [1]", EVIDENCE_GRADE_4).ok)

    def test_half_a_grounded_total_is_grounded(self):
        self.assertTrue(verify("نصف المبلغ 22,500 [1]", EVIDENCE_GRADE_4).ok)

    def test_the_allowance_still_rejects_an_invented_figure(self):
        report = verify("المصاريف 30,000 جنيه [1]", EVIDENCE_GRADE_4)
        self.assertFalse(report.ok)
        self.assertEqual(report.ungrounded, (30000.0,))

    def test_a_sum_of_two_grounded_figures_is_grounded(self):
        evidence = ["الرسوم 30,000 والأنشطة 5,000"]
        self.assertTrue(verify("الإجمالي 35,000 [1]", evidence).ok)


if __name__ == "__main__":
    unittest.main()
