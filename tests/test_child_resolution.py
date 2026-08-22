"""Which child a turn is about.

The rules are ordered by how firmly the parent stated it, and most of these cases exist
to pin down the ORDER rather than any single rule — because the order is where this goes
wrong. A gender word that outranks the pin re-asks a parent who already answered; a pin
that outranks a name answers about the sister of the child they just asked about.
"""
import unittest

from backend.chat.child_context import SessionChild
from backend.chat.child_resolution import resolve_child
from backend.chat.child_roster import ChildOption

ALI = ChildOption(student_id="S-1", label="علي حسن", gender="male", year_level="Year 4")
AHMED = ChildOption(student_id="S-2", label="أحمد حسن", gender="male")
LAYLA = ChildOption(student_id="S-3", label="ليلى حسن", gender="female")
UNKNOWN = ChildOption(student_id="S-4", label="سيد حسن")

TWO_SONS = [ALI, AHMED]
ONE_SON_TWO_DAUGHTERS = [ALI, LAYLA, ChildOption(student_id="S-5", label="سارة", gender="female")]


def _pin(student_id="", gender="unknown"):
    return SessionChild(student_id=student_id, gender=gender)


class NobodyToChooseFrom(unittest.TestCase):
    def test_an_empty_roster_says_nothing_rather_than_asking(self):
        """"This parent has no readable children" is a fact the records tool has careful
        wording for, and only it should say it."""
        out = resolve_child(reference="son", roster=[])
        self.assertFalse(out.resolved)
        self.assertFalse(out.ask)

    def test_an_only_child_is_never_asked_about(self):
        out = resolve_child(reference="context", roster=[ALI])
        self.assertTrue(out.resolved)
        self.assertEqual(out.source, "only_child")
        self.assertEqual(out.label, "علي حسن")
        self.assertEqual(out.year_level, "Year 4")


class TheParentNamedSomebody(unittest.TestCase):
    def test_a_unique_name_resolves(self):
        out = resolve_child(reference="named", child_name="علي", roster=TWO_SONS)
        self.assertEqual(out.student_id, "S-1")
        self.assertEqual(out.source, "named")

    def test_a_partial_name_matches_the_full_one_on_file(self):
        """A parent writing "Ali" means the child stored as "Ali Hassan"; this school
        stores a full patronymic."""
        out = resolve_child(reference="named", child_name="Ali", roster=[
            ChildOption(student_id="S-1", label="Ali Osman", gender="male"), LAYLA,
        ])
        self.assertEqual(out.student_id, "S-1")

    def test_a_name_inside_two_children_asks_between_those_two(self):
        """«أحمد» is inside both «علي أحمد حسن» and «أحمد حسن», and picking the first
        shows one child's marks while the parent asked about the other."""
        ali_ahmed = ChildOption(student_id="S-9", label="علي أحمد حسن", gender="male")
        out = resolve_child(reference="named", child_name="أحمد", roster=[ali_ahmed, AHMED, LAYLA])

        self.assertTrue(out.ask)
        self.assertEqual(sorted(out.option_labels), sorted(["علي أحمد حسن", "أحمد حسن"]))

    def test_a_name_beats_the_pin(self):
        """"And how is Omar?" must move the conversation on even when the previous
        question was about his sister."""
        out = resolve_child(
            reference="named", child_name="أحمد", roster=TWO_SONS, pin=_pin("S-1"),
        )
        self.assertEqual(out.student_id, "S-2")

    def test_a_name_matching_nobody_asks_rather_than_falling_back_to_the_pin(self):
        """The parent named somebody. Answering about a different child while they watch
        is worse than one more question."""
        out = resolve_child(
            reference="named", child_name="خالد", roster=TWO_SONS, pin=_pin("S-1"),
        )
        self.assertTrue(out.ask)
        self.assertFalse(out.resolved)


class TheParentSaidSonOrDaughter(unittest.TestCase):
    def test_the_only_son_among_daughters_resolves_with_no_question(self):
        out = resolve_child(reference="son", roster=ONE_SON_TWO_DAUGHTERS)
        self.assertEqual(out.student_id, "S-1")
        self.assertEqual(out.source, "gender")

    def test_the_only_daughter_among_sons_resolves_with_no_question(self):
        out = resolve_child(reference="daughter", roster=[ALI, AHMED, LAYLA])
        self.assertEqual(out.student_id, "S-3")

    def test_two_sons_are_asked_between(self):
        out = resolve_child(reference="son", roster=TWO_SONS)
        self.assertTrue(out.ask)
        self.assertEqual(out.option_labels, ["علي حسن", "أحمد حسن"])

    def test_the_daughters_are_not_offered_after_the_parent_said_son(self):
        """Asking "Ali, Ahmed or Layla?" after "my son" ignores what they just said."""
        out = resolve_child(reference="son", roster=[ALI, AHMED, LAYLA])
        self.assertTrue(out.ask)
        self.assertNotIn("ليلى حسن", out.option_labels)

    def test_a_child_with_no_gender_on_file_could_be_either(self):
        """Every child is in this state until a registrar uploads it, and a blank cell
        must never be able to select one."""
        out = resolve_child(reference="son", roster=[LAYLA, UNKNOWN])
        self.assertEqual(out.student_id, "S-4")
        out = resolve_child(reference="daughter", roster=[ALI, UNKNOWN])
        self.assertEqual(out.student_id, "S-4")

    def test_an_unknown_child_alongside_a_real_son_is_still_ambiguous(self):
        out = resolve_child(reference="son", roster=[ALI, UNKNOWN])
        self.assertTrue(out.ask)
        self.assertEqual(len(out.options), 2)

    def test_nobody_could_be_the_son_asks_rather_than_correcting_the_parent(self):
        """Their wording is better evidence than the column."""
        out = resolve_child(reference="son", roster=[LAYLA, ChildOption("S-6", "سارة", "female")])
        self.assertTrue(out.ask)
        self.assertEqual(len(out.options), 2)


class TheConversationAlreadySettledIt(unittest.TestCase):
    def test_the_pin_answers_a_follow_up(self):
        """The whole point of requirement 4: a parent who answered once is not asked
        again on their next question."""
        out = resolve_child(reference="context", roster=TWO_SONS, pin=_pin("S-1"))
        self.assertEqual(out.student_id, "S-1")
        self.assertEqual(out.source, "pin")

    def test_a_pin_survives_a_gender_word_that_agrees_with_it(self):
        """A parent settled on Ali who says "my son" is continuing, not restarting — and
        re-asking here is the exact bug this feature exists to remove."""
        out = resolve_child(reference="son", roster=TWO_SONS, pin=_pin("S-1"))
        self.assertEqual(out.student_id, "S-1")
        self.assertEqual(out.source, "pin")

    def test_a_gender_word_that_contradicts_the_pin_evicts_it(self):
        """Settled on a daughter, then "my son": that is a change of subject."""
        out = resolve_child(reference="son", roster=[ALI, AHMED, LAYLA], pin=_pin("S-3"))
        self.assertTrue(out.ask)
        self.assertNotIn("ليلى حسن", out.option_labels)

    def test_a_pin_naming_a_child_no_longer_on_the_roster_is_ignored(self):
        """Access was revoked between turns. Falling through re-asks; using it would
        name a child this parent may no longer be told about."""
        out = resolve_child(reference="context", roster=TWO_SONS, pin=_pin("S-99"))
        self.assertTrue(out.ask)
        self.assertFalse(out.resolved)

    def test_no_pin_and_nothing_in_the_message_asks(self):
        out = resolve_child(reference="context", roster=TWO_SONS)
        self.assertTrue(out.ask)
        self.assertEqual(len(out.options), 2)


class MoreThanOneChild(unittest.TestCase):
    def test_a_plural_reference_never_narrows_and_never_asks(self):
        """Collapsing "all of them" to one child is worse than not helping, and the tool
        can still read the whole roster."""
        out = resolve_child(reference="plural", roster=TWO_SONS, pin=_pin("S-1"))
        self.assertFalse(out.resolved)
        self.assertFalse(out.ask)

    def test_a_plural_reference_with_one_child_still_resolves(self):
        out = resolve_child(reference="plural", roster=[ALI])
        self.assertTrue(out.resolved)


class ResolvedAndAskAreExclusive(unittest.TestCase):
    def test_no_outcome_ever_sets_both(self):
        rosters = [[], [ALI], TWO_SONS, ONE_SON_TWO_DAUGHTERS, [ALI, UNKNOWN, LAYLA]]
        references = ["none", "son", "daughter", "child", "plural", "named", "context"]
        pins = [_pin(), _pin("S-1"), _pin("S-99")]

        for roster in rosters:
            for reference in references:
                for pin in pins:
                    with self.subTest(reference=reference, size=len(roster), pin=pin.student_id):
                        out = resolve_child(
                            reference=reference, child_name="علي", roster=roster, pin=pin,
                        )
                        self.assertFalse(out.resolved and out.ask)
                        if out.ask:
                            self.assertTrue(out.options)
                        if out.resolved:
                            self.assertTrue(out.student_id)
                            self.assertTrue(out.label)


if __name__ == "__main__":
    unittest.main()
