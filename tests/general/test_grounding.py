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
from backend.chat.service import _COPY

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



class ACitationOnAToolAnsweredTurn(unittest.TestCase):
    """A `[n]` means two different things, and the remedy differs — but the RULE does not.

    `verify` keeps calling this `cited_without_evidence`: tool text is not a citable
    chunk and must never make `[1]` valid. What it also reports now is whether a tool ran
    at all, so the caller can tell "invented a source" from "cited on a turn whose
    evidence carries no `[n]`". Acting on that is policy, and it lives in
    `service._enforce_grounding` beside `answer_grounding_mode`.
    """

    TOOL = "TIMETABLE for ليلى — 4A, Term 2 (ARABIC-2025-2026):\n- Sunday: 1) Arabic"

    def test_the_citation_rule_itself_is_unchanged_by_tool_evidence(self):
        report = verify("Sunday is 1) Arabic. [1]", [], extra_evidence=[self.TOOL])
        self.assertFalse(report.ok)
        self.assertTrue(report.cited_without_evidence)
        self.assertEqual(report.evidence_count, 0)

    def test_the_report_says_whether_a_tool_supplied_the_evidence(self):
        self.assertTrue(verify("x [1]", [], extra_evidence=[self.TOOL]).tool_evidence)
        self.assertFalse(verify("x [1]", [], extra_evidence=[]).tool_evidence)
        self.assertFalse(verify("x [1]", [], extra_evidence=[None, ""]).tool_evidence)

    def test_tool_evidence_does_not_excuse_an_invented_figure(self):
        report = verify("The fee is 45,000. [1]", [], extra_evidence=[self.TOOL])
        self.assertFalse(report.ok)
        self.assertIn(45000.0, report.ungrounded)

    def test_the_fact_reaches_the_trace(self):
        trace = verify("x [1]", [], extra_evidence=[self.TOOL]).as_trace()
        self.assertTrue(trace["grounding_tool_evidence"])


class TheCallerStripsRatherThanRefuses(unittest.TestCase):
    """`service._enforce_grounding` is where the marker is forgiven, and only there.

    The failure it answers to: a correct timetable, grounded in the tool's own text, was
    replaced with "I could not verify these figures" because the model had appended `[1]`.
    """

    TOOL = "TIMETABLE for ليلى — 4A (ARABIC-2025-2026):\n- Sunday: 1) Arabic 07:45-08:30"

    class _Finalizer:
        def __init__(self, answer, report):
            self.answer = answer
            self._report = report

        def verify(self, evidence, *, floor, check_citations):
            return self._report

    def _verdict(self, answer, evidence=(), extra=()):
        from unittest import mock

        from backend.chat import service

        report = verify(answer, list(evidence), floor=100,
                        check_citations=True, extra_evidence=list(extra))
        plan = type("P", (), {"exposed_tools": ["get_student_timetable"],
                              "short_circuit": False})()
        # `enforce` explicitly: the base profile this module loads observes rather than
        # enforces, and observing never replaces anything — which would make every
        # assertion below pass on an empty string and prove nothing. The school profile
        # is the one that enforces, and it is the deployment this bug was found on.
        with mock.patch.object(service._PROFILE.agent, "answer_grounding_mode", "enforce"):
            return service._enforce_grounding(
                self._Finalizer(answer, report), None, plan
            )

    def test_a_marker_on_a_tool_answer_is_stripped_and_the_answer_kept(self):
        out = self._verdict("الأحد: 1) اللغة العربية 07:45-08:30. [1]", extra=[self.TOOL])
        self.assertEqual("الأحد: 1) اللغة العربية 07:45-08:30.", out)

    def test_a_marker_with_no_tool_behind_it_is_still_refused(self):
        """The protection this must not weaken: nothing was read, so nothing was cited."""
        out = self._verdict("الرسوم 45000 جنيه. [1]", extra=[])
        self.assertEqual(_COPY.unverified_answer, out)

    def test_an_invented_figure_is_refused_even_with_tool_evidence(self):
        out = self._verdict("الرسوم 45000 جنيه. [1]", extra=[self.TOOL])
        self.assertEqual(_COPY.unverified_answer, out)

    def test_a_clean_tool_answer_is_left_exactly_alone(self):
        self.assertEqual("", self._verdict("الأحد: 1) اللغة العربية.", extra=[self.TOOL]))


if __name__ == "__main__":
    unittest.main()
