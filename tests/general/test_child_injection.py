"""The settled child, from the plan to the prompt.

The resolver is tested next door; this covers the wiring around it, which is where a
feature like this actually dies — a plan field nothing renders, a render gate that
returns None before the new value is looked at, a roster read that never starts because
the caller is not a parent.
"""
import os
import unittest
from unittest.mock import patch

from backend.chat.child_context import SessionChild
from backend.chat.child_resolution import ResolvedChild, no_child, resolve_child
from backend.chat.child_roster import ChildOption
from backend.chat.service import _turn_context_message
from backend.chat.signals import RequestSignals
from backend.chat.turn_policy import resolve_turn

ALI = ChildOption(student_id="S-1", label="علي حسن", gender="male", year_level="Year 4")
AHMED = ChildOption(student_id="S-2", label="أحمد حسن", gender="male")


class _Agent:
    tools = ["search_knowledge_base", "get_student_records"]
    social_phrases = []
    social_reply_mode = "model"


class _Copy:
    social = None
    out_of_domain = None


def _plan(child=None, **signal_kwargs):
    signals = RequestSignals(question="q", **signal_kwargs)
    return resolve_turn(signals, agent_config=_Agent(), copy_config=_Copy(), child=child)


class ThePlanCarriesIt(unittest.TestCase):
    def test_a_settled_child_becomes_a_hint(self):
        plan = _plan(resolve_child(reference="context", roster=[ALI]))
        self.assertEqual(plan.child_hint, "علي حسن")
        self.assertEqual(plan.child_year, "Year 4")
        self.assertEqual(plan.child_options, [])

    def test_an_open_question_becomes_options(self):
        plan = _plan(resolve_child(reference="son", roster=[ALI, AHMED]))
        self.assertEqual(plan.child_hint, "")
        self.assertEqual(plan.child_options, ["علي حسن", "أحمد حسن"])

    def test_a_turn_about_nobody_carries_neither(self):
        plan = _plan(no_child("not a turn about a child"))
        self.assertEqual(plan.child_hint, "")
        self.assertEqual(plan.child_options, [])

    def test_a_caller_that_passes_no_child_still_plans_a_turn(self):
        """Every existing caller — a test double, an integrating deployment — keeps
        working and simply plans a turn about nobody in particular."""
        plan = resolve_turn(RequestSignals(question="q"), agent_config=_Agent(), copy_config=_Copy())
        self.assertEqual(plan.child_hint, "")

    def test_the_two_are_never_both_set(self):
        for child in (
            ResolvedChild(student_id="S-1", label="علي", resolved=True, options=(ALI, AHMED)),
            ResolvedChild(options=(ALI, AHMED), ask=True),
        ):
            with self.subTest(resolved=child.resolved):
                plan = _plan(child)
                self.assertFalse(plan.child_hint and plan.child_options)

    def test_the_trace_reports_the_decision_but_not_the_name(self):
        trace = _plan(resolve_child(reference="context", roster=[ALI])).as_trace()
        self.assertTrue(trace["turn_child_resolved"])
        self.assertFalse(trace["turn_child_asked"])
        self.assertNotIn("علي", repr(trace))


class ThePromptRendersIt(unittest.TestCase):
    def test_a_settled_child_reaches_the_model_with_their_year(self):
        message = _turn_context_message(_plan(resolve_child(reference="context", roster=[ALI])))

        self.assertIsNotNone(message)
        self.assertIn("علي حسن", message.content)
        self.assertIn("Year 4", message.content)

    def test_the_authorisation_caveat_travels_with_the_name(self):
        """A name in a prompt is a hint. Saying so where the model reads it is what
        stops it treating the name as permission."""
        message = _turn_context_message(_plan(resolve_child(reference="context", roster=[ALI])))
        self.assertIn("authorises nothing", message.content)

    def test_an_open_question_offers_exactly_the_candidates(self):
        message = _turn_context_message(_plan(resolve_child(reference="son", roster=[ALI, AHMED])))

        self.assertIn("علي حسن", message.content)
        self.assertIn("أحمد حسن", message.content)
        self.assertIn("ask", message.content.lower())

    def test_a_child_with_no_year_on_file_renders_without_one(self):
        """The state every child is in until SIS carries a year group."""
        yearless = ChildOption(student_id="S-9", label="سارة")
        message = _turn_context_message(_plan(resolve_child(reference="context", roster=[yearless])))

        self.assertIn("سارة", message.content)
        self.assertNotIn("who is in", message.content)

    def test_a_turn_about_no_child_renders_nothing_at_all(self):
        """Most turns. The feature has to cost them nothing."""
        self.assertIsNone(_turn_context_message(_plan(no_child("general question"))))

    def test_the_child_survives_a_turn_that_carries_nothing_else(self):
        """The single point of failure: the render gate used to return None whenever
        there was no resolved question and no constraints, which is most child turns."""
        plan = _plan(resolve_child(reference="context", roster=[ALI]))
        self.assertEqual(plan.resolved_question, "")
        self.assertEqual(plan.carried_constraints, [])

        self.assertIsNotNone(_turn_context_message(plan))


class TheRosterReadStarts(unittest.TestCase):
    def setUp(self):
        # The roster sits behind a cache shared with whatever Redis is running, and the
        # guardian ids in a test suite collide across files. Without this, another
        # file having cached G-1 makes the deliberately-failing fetch below succeed.
        self._env = patch.dict(os.environ, {"CHILD_ROSTER_TTL_SECONDS": "0"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_no_thread_is_started_for_someone_who_is_not_a_parent(self):
        import backend.chat.child_roster as child_roster

        class _NotAParent:
            guardian_id = ""
            guardian_token = ""

        self.assertIsNone(child_roster.prefetch(_NotAParent()))

    def test_a_prefetch_returns_what_the_fetch_produced(self):
        import backend.chat.child_roster as child_roster

        class _Parent:
            guardian_id = "G-1"
            guardian_token = "t"
            session_id = "s"

        rows = [{"student_id": "S-1", "full_name_ar": "علي"}]
        ahead = child_roster.prefetch(
            _Parent(), fetch=lambda *a: (child_roster.OK, rows)
        )
        outcome, children = ahead.result(timeout=5)

        self.assertEqual(outcome, child_roster.OK)
        self.assertEqual([c.student_id for c in children], ["S-1"])

    def test_a_failing_fetch_degrades_to_unavailable_rather_than_raising(self):
        import backend.chat.child_roster as child_roster

        class _Parent:
            guardian_id = "G-1"
            guardian_token = "t"
            session_id = "s"

        def boom(*_a):
            raise RuntimeError("facade on fire")

        ahead = child_roster.prefetch(_Parent(), fetch=boom)
        outcome, children = ahead.result(timeout=5)

        self.assertEqual(outcome, child_roster.UNAVAILABLE)
        self.assertEqual(children, [])


class ThePinIsWrittenBack(unittest.TestCase):
    def test_the_resolver_and_the_tool_agree_on_the_same_child(self):
        """One route table. Two matchers with different rules drift, and drift here
        means the prompt naming one child while the tool answers about another."""
        from backend.tools.records import _match_student

        class _Ctx:
            child = SessionChild(student_id="S-2")

        planner = resolve_child(reference="context", roster=[ALI, AHMED], pin=_Ctx.child)
        tool = _match_student([ALI, AHMED], "", ctx=_Ctx())

        self.assertEqual(planner.student_id, "S-2")
        self.assertEqual(tool.student_id, "S-2")


if __name__ == "__main__":
    unittest.main()
