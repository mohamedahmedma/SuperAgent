"""Ordinary questions that are hard to answer — not exotic inputs.

The distinction this file is built on: an EDGE case is rare and strange (an empty
roster, a malformed token, a name that matches nobody). A COMPLEX case is common and
complicated — two children, a sibling discount, a follow-up that only makes sense
against the previous turn. Edge cases are covered in the per-bug files. These are the
turns a school assistant actually gets all day, and they are where the four fixes have
to hold TOGETHER rather than one at a time.

Each scenario below is one a parent of a real child would recognise:

    two children, and the parent moves between them mid-conversation
    a sibling discount, where the right answer is a figure no document states
    "so how much is each instalment?" — arithmetic the parent asked for
    "and what about next year?" — a year the child is not in yet
    a compound question whose answer is spread over two chunks
    a late-payment surcharge, which is the discount case with the sign flipped

The sibling-discount scenario is the one that found a real defect: the grounding check
rejected 27,000 against evidence of «30,000» and «خصم 10%», because 27,000 is not a
sum, product, quotient or n-way split of {30000, 10}. Percentages are now read as
operators rather than amounts (`backend/chat/grounding.py:percentages_in`). A discount
is not an edge case; it is most of the fee questions this school gets.
"""
import unittest

from backend.chat.child_resolution import resolve_child
from backend.chat.child_roster import ChildOption
from backend.chat.child_context import SessionChild
from backend.chat.grounding import percentages_in, verify
from backend.chat.turn_policy import TurnPlan, _plan_child
from backend.profiles import get_profile

MARKERS = get_profile().agent.year_reference_markers

YEAR_1 = "الصف الأول الابتدائي"
YEAR_2 = "الصف الثاني الابتدائي"
YEAR_4 = "الصف الرابع الابتدائي"

# ---------------------------------------------------------------------------
# The corpus, written the way a school actually writes it: figures in prose,
# instalment counts spelled as words, discounts as percentages.
# ---------------------------------------------------------------------------

FEES_WITH_DISCOUNT = [
    "رسوم الصف الأول الابتدائي للعام 2026: 30,000 جنيه على ثلاث دفعات.",
    "خصم 10% للأخ التاني، وخصم 15% للأخ التالت.",
]

FEES_WITH_SURCHARGE = [
    "رسوم الصف الرابع الابتدائي: 45,000 جنيه على ثلاث دفعات.",
    "غرامة تأخير 5% على الدفعة اللي تتأخر عن ميعادها.",
]

FEES_AND_DEADLINE = [
    "رسوم الصف الأول الابتدائي للعام 2026: 30,000 جنيه على ثلاث دفعات.",
    "آخر ميعاد لسداد الدفعة الأولى هو 15 سبتمبر 2026.",
]

FULL_FEE_TABLE = [
    "رسوم الصف الأول الابتدائي للعام 2026: 30,000 جنيه على ثلاث دفعات.",
    "رسوم الصف الثاني الابتدائي: 35,000 جنيه على ثلاث دفعات.",
    "رسوم الصف الرابع الابتدائي: 45,000 جنيه على ثلاث دفعات.",
]


def _roster():
    """A family with a son in Year 1 and a daughter in Year 4. The common case."""
    return [
        ChildOption(student_id="s1", label="علي أحمد", gender="male", year_level=YEAR_1),
        ChildOption(student_id="s2", label="ليلى أحمد", gender="female", year_level=YEAR_4),
    ]


class ASiblingDiscount(unittest.TestCase):
    """The right answer is a figure that appears in no document.

    A school that offers 10% off the second child forces every correct answer to be
    computed. If the verifier cannot follow that arithmetic it blocks good answers, and
    a check that blocks good answers gets switched off — which is how a deployment ends
    up with no verification at all.
    """

    def test_a_percentage_is_read_as_an_operator_not_as_an_amount(self):
        self.assertEqual(percentages_in("خصم 10% للأخ التاني"), {10.0})
        self.assertEqual(percentages_in("غرامة تأخير 5%"), {5.0})

    def test_the_price_after_a_sibling_discount_is_grounded(self):
        """30,000 less 10% is 27,000, and no chunk says 27,000."""
        report = verify("رسوم الأخ التاني 27,000 جنيه بعد الخصم [1][2]", FEES_WITH_DISCOUNT)
        self.assertTrue(report.ok, report.reason)

    def test_the_discount_amount_itself_is_grounded(self):
        report = verify("الخصم قيمته 3,000 جنيه [2]", FEES_WITH_DISCOUNT)
        self.assertTrue(report.ok, report.reason)

    def test_the_third_child_rate_is_grounded_too(self):
        """15% off 30,000 is 25,500 — a second percentage in the same chunk."""
        report = verify("الأخ التالت 25,500 جنيه [1][2]", FEES_WITH_DISCOUNT)
        self.assertTrue(report.ok, report.reason)

    def test_a_discount_that_was_never_offered_is_caught(self):
        """The allowance must not become a licence to state any figure. 40% off 30,000
        is 18,000, no chunk offers 40%, and 18,000 is not a whole-number split of
        anything here either."""
        report = verify("مع خصم 40% تدفع 18,000 جنيه [1]", FEES_WITH_DISCOUNT)
        self.assertFalse(report.ok)
        self.assertIn(18000.0, report.ungrounded)

    def test_a_late_surcharge_is_the_same_rule_with_the_sign_flipped(self):
        """45,000 plus 5% is 47,250."""
        report = verify("لو اتأخرت هتدفع 47,250 جنيه [1][2]", FEES_WITH_SURCHARGE)
        self.assertTrue(report.ok, report.reason)


class WhatTheNumberCheckCannotDo(unittest.TestCase):
    """Two limits, asserted so they are decisions rather than surprises.

    Both were found by writing the scenarios above and being wrong about what would
    happen. Written down here because a limit nobody recorded gets rediscovered as a
    bug report six months later.
    """

    def test_an_invented_half_price_is_indistinguishable_from_two_instalments(self):
        """A 50% discount nobody offered lands on 15,000 — which is also 30,000 split
        two ways, and splitting a total is something the check must permit. Arithmetic
        alone cannot tell the two apart; only the corpus can, and it is not asked.

        Accepted rather than fixed: narrowing the split allowance to forbid halves
        would reject «الدفعة الأولى 15,000» on a school that bills twice a year, which
        is a real answer to a real question."""
        report = verify("مع خصم 50% تدفع 15,000 جنيه [1]", FEES_WITH_DISCOUNT)
        self.assertTrue(report.ok)

    def test_derivation_is_one_level_deep_and_stops_there(self):
        """«خصم 10% وبعدين على تلات دفعات» is 9,000 — a discount and then a split, two
        steps from anything written down. The check blocks it, and that is the right
        trade even though the answer is correct.

        Composing the two steps would admit 8,500 as well, because 25,500 (the third
        child's rate) divided by three is exactly 8,500 — see
        `AnAnswerThatIsMostlyRight`, where 8,500 is an invented bus fare that a parent
        must not be shown. A blocked correct answer costs the "couldn't verify" copy; an
        admitted invented one costs a parent the wrong number."""
        blocked = verify("كل دفعة 9,000 جنيه بعد الخصم [1][2]", FEES_WITH_DISCOUNT)
        self.assertFalse(blocked.ok)
        self.assertIn(9000.0, blocked.ungrounded)

        # The coincidence that makes composing unsafe, asserted so it stays visible.
        still_caught = verify("مصاريف الأتوبيس 8,500 جنيه", FEES_WITH_DISCOUNT)
        self.assertFalse(still_caught.ok)


class ArithmeticTheParentAskedFor(unittest.TestCase):
    """«يعني الدفعة الواحدة كام؟» — the most common follow-up there is."""

    def test_an_instalment_of_a_grounded_total(self):
        report = verify("الدفعة الواحدة 10,000 جنيه [1]", FULL_FEE_TABLE)
        self.assertTrue(report.ok, report.reason)

    def test_two_children_added_together(self):
        """A parent with a Year 1 and a Year 4 child asking the household total."""
        report = verify("إجمالي مصاريف الاتنين 75,000 جنيه [1][3]", FULL_FEE_TABLE)
        self.assertTrue(report.ok, report.reason)

    def test_the_difference_between_two_years(self):
        report = verify("الفرق بين الصفين 15,000 جنيه [1][3]", FULL_FEE_TABLE)
        self.assertTrue(report.ok, report.reason)

    def test_a_total_that_matches_no_combination_is_caught(self):
        report = verify("إجمالي مصاريف الاتنين 88,000 جنيه [1][3]", FULL_FEE_TABLE)
        self.assertFalse(report.ok)
        self.assertIn(88000.0, report.ungrounded)


class ACompoundQuestion(unittest.TestCase):
    """«مصاريف ابني كام وامتى آخر ميعاد للدفع» — two facts, two chunks, one answer."""

    def test_both_facts_and_both_citations_survive(self):
        answer = ("رسوم الصف الأول 30,000 جنيه على ثلاث دفعات [1]، "
                  "وآخر ميعاد للدفعة الأولى 15 سبتمبر 2026 [2].")
        report = verify(answer, FEES_AND_DEADLINE)
        self.assertTrue(report.ok, report.reason)

    def test_a_date_invented_alongside_a_correct_fee_is_still_caught(self):
        """The dangerous shape: one true fact lends credibility to a false one, and a
        parent who trusts the fee has no reason to doubt the deadline beside it."""
        answer = ("رسوم الصف الأول 30,000 جنيه [1]، "
                  "وآخر ميعاد للدفع 30 نوفمبر 2027 [2].")
        report = verify(answer, FEES_AND_DEADLINE)
        self.assertFalse(report.ok)
        self.assertIn(2027.0, report.ungrounded)

    def test_a_citation_beyond_the_two_chunks_is_caught(self):
        answer = "رسوم الصف الأول 30,000 جنيه [1] وآخر ميعاد 15 سبتمبر 2026 [3]."
        report = verify(answer, FEES_AND_DEADLINE)
        self.assertFalse(report.ok)
        self.assertEqual(report.invalid_citations, (3,))


class TwoChildrenInOneConversation(unittest.TestCase):
    """A son and a daughter, and the parent moves between them.

    Every step below is a message a parent actually sends. What makes it complex is
    that the answer to each depends on the one before, and the year group has to move
    with the child — a pin that outlives its subject quotes one child's fee under the
    other's name.
    """

    def _settle(self, reference, name="", pin=None, question=""):
        child = resolve_child(
            reference=reference, child_name=name, roster=_roster(), pin=pin or SessionChild()
        )
        plan = TurnPlan()
        _plan_child(plan, child, question, MARKERS)
        return child, plan

    def test_a_vague_first_question_asks_which_child(self):
        """«مصاريف ولادي كام» with two on file. Answering about either would be a
        guess, and a guess here quotes the wrong fee."""
        child, plan = self._settle("context", question="مصاريف ابني كام")
        self.assertTrue(child.ask)
        self.assertEqual(sorted(plan.child_options), sorted(["علي أحمد", "ليلى أحمد"]))
        self.assertEqual(plan.child_year, "")

    def test_naming_the_son_settles_the_year_with_him(self):
        child, plan = self._settle("named", "علي", question="مصاريف علي كام")
        self.assertTrue(child.resolved)
        self.assertEqual(plan.child_hint, "علي أحمد")
        self.assertEqual(plan.child_year, YEAR_1)

    def test_the_next_question_moves_to_the_daughter_and_the_year_moves_with_her(self):
        """«وبنتي؟» after a turn about Ali. The pin says Ali; the word says daughter,
        and the word wins — otherwise Layla's fees are quoted at Ali's year."""
        pin = SessionChild(student_id="s1", label="علي أحمد", gender="male")
        child, plan = self._settle("daughter", pin=pin, question="وبنتي؟")
        self.assertEqual(plan.child_hint, "ليلى أحمد")
        self.assertEqual(plan.child_year, YEAR_4)
        self.assertNotEqual(plan.child_year, YEAR_1)

    def test_a_follow_up_with_no_new_subject_stays_on_the_pinned_child(self):
        """«وامتى الدفع؟» — the parent has not changed subject, so neither do we."""
        pin = SessionChild(student_id="s2", label="ليلى أحمد", gender="female")
        child, plan = self._settle("context", pin=pin, question="وامتى الدفع؟")
        self.assertEqual(plan.child_hint, "ليلى أحمد")
        self.assertEqual(plan.child_year, YEAR_4)

    def test_asking_about_both_children_narrows_to_neither(self):
        """«مصاريف ولادي» is about two children. Collapsing it to one is worse than not
        narrowing at all, and the year of one of them must not bind the answer."""
        child, plan = self._settle("plural", question="مصاريف ولادي كام")
        self.assertFalse(child.resolved)
        self.assertEqual(plan.child_year, "")
        self.assertEqual(plan.child_options, [])


class PlanningAheadForNextYear(unittest.TestCase):
    """«هيبقى كام في الصف التاني؟» — a parent budgeting for September.

    Common, and the case where the roster's year is exactly the wrong answer: the child
    is in Year 1 today and the question is about Year 2.
    """

    def test_the_roster_year_stands_down_when_the_parent_names_one(self):
        child = resolve_child(reference="son", roster=_roster(), pin=SessionChild())
        plan = TurnPlan()
        _plan_child(plan, child, "هيبقى كام في الصف الثاني؟", MARKERS)
        self.assertEqual(plan.child_hint, "علي أحمد")
        self.assertEqual(plan.child_year, "")

    def test_the_child_is_still_named_so_the_answer_stays_about_him(self):
        """Withholding the YEAR must not withhold the CHILD — the turn is still his."""
        child = resolve_child(reference="son", roster=_roster(), pin=SessionChild())
        plan = TurnPlan()
        _plan_child(plan, child, "هيبقى كام في الصف الثاني؟", MARKERS)
        self.assertTrue(plan.as_trace()["turn_child_resolved"])
        self.assertFalse(plan.as_trace()["turn_child_year_applied"])

    def test_next_year_s_figure_is_grounded_against_the_same_table(self):
        report = verify("رسوم الصف الثاني 35,000 جنيه [2]", FULL_FEE_TABLE)
        self.assertTrue(report.ok, report.reason)


class AnAnswerThatIsMostlyRight(unittest.TestCase):
    """The realistic failure is not a wholly invented answer — it is a long, helpful,
    correct-sounding one with a single wrong figure in the middle of it."""

    def test_one_wrong_figure_in_an_otherwise_grounded_answer_is_caught(self):
        answer = (
            "رسوم الصف الأول الابتدائي للعام 2026 هي 30,000 جنيه [1]، "
            "تتقسم على ثلاث دفعات كل واحدة 10,000 جنيه. "
            "وفيه خصم 10% للأخ التاني يعني 27,000 جنيه. "
            "ومصاريف الأتوبيس 8,500 جنيه في السنة."
        )
        report = verify(answer, FEES_WITH_DISCOUNT)
        self.assertFalse(report.ok)
        # Only the bus fare is unsupported; everything else follows from the chunks.
        self.assertEqual(report.ungrounded, (8500.0,))

    def test_the_same_answer_without_the_invented_line_passes(self):
        answer = (
            "رسوم الصف الأول الابتدائي للعام 2026 هي 30,000 جنيه [1]، "
            "تتقسم على ثلاث دفعات كل واحدة 10,000 جنيه. "
            "وفيه خصم 10% للأخ التاني يعني 27,000 جنيه [2]."
        )
        report = verify(answer, FEES_WITH_DISCOUNT)
        self.assertTrue(report.ok, report.reason)


if __name__ == "__main__":
    unittest.main()
