"""A parent moving between two children in one conversation.

Every step below is a message a parent actually sends, and what makes the sequence hard
is that each answer depends on the one before: the year group has to move with the child,
because a pin that outlives its subject quotes one child's fee under the other's name.

This file used to also hold the numeric grounding scenarios — a sibling discount, a late
surcharge, arithmetic the parent asked for. Those went with the check itself: a record now
reaches the reader as a block the tool rendered, so there is no model-written figure left
to verify. See tests/general/test_answer_blocks.py for what replaced it.
"""
import unittest

from backend.chat.child_context import SessionChild
from backend.chat.child_resolution import resolve_child
from backend.chat.child_roster import ChildOption
from backend.chat.turn_policy import TurnPlan, _plan_child
from backend.profiles import get_profile

MARKERS = get_profile().agent.year_reference_markers

YEAR_1 = "الصف الأول الابتدائي"
YEAR_4 = "الصف الرابع الابتدائي"


def _roster():
    """A family with a son in Year 1 and a daughter in Year 4. The common case."""
    return [
        ChildOption(student_id="s1", label="علي أحمد", gender="male", year_level=YEAR_1),
        ChildOption(student_id="s2", label="ليلى أحمد", gender="female", year_level=YEAR_4),
    ]


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
