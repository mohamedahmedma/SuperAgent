"""The child a conversation remembers, and the thread it remembers it against.

Two things are under test here, and they are the two halves of the same bug. The pin
used to live on a per-request object that nothing loaded and nothing saved, so it died
at the turn boundary — meaning a parent who answered "Layla" once was asked again on
their next question, forever. And a thread had no transport-level identity at all, so a
client that forgot to send one wrote every conversation into the same bucket.

The tests that matter most are the ones about NOT remembering: a pin that outlives the
guardian binding it was resolved under would inject one family's child into another
family's conversation, and that is the only failure here that is worse than the bug.
"""
import unittest

from backend.api.routes.chat import DEFAULT_THREAD, _thread_id
from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_context import (
    SESSION_CHILD_KEY,
    SessionChild,
    load_child_state,
    save_child_state,
)
from backend.chat.request_context import ChatRequestContext


class SessionChildTests(unittest.TestCase):
    def test_a_fresh_conversation_has_no_child(self):
        self.assertFalse(load_child_state({}).is_set)
        self.assertFalse(load_child_state(None).is_set)

    def test_a_pin_round_trips_through_metadata(self):
        child = load_child_state({}, guardian_id="gdn_7f3a")
        child.pin(student_id="S1001", label="ليلى أحمد", gender="female")

        save_meta = {}
        save_child_state(save_meta, child)
        # Threaded by reference: the object the turn mutated is the object that is saved,
        # so nothing has to remember to copy the pin back out.
        reloaded = load_child_state(save_meta, guardian_id="gdn_7f3a")

        self.assertEqual(reloaded.student_id, "S1001")
        self.assertEqual(reloaded.label, "ليلى أحمد")
        self.assertEqual(reloaded.gender, "female")

    def test_a_pin_from_another_guardian_is_not_inherited(self):
        """The one failure worse than being asked twice.

        `chat_sessions` is keyed by username; the right to read a child is keyed by the
        guardian handle. An administrator can rebind one to the other
        (identity/routes.py:342), so without this check a custody transfer leaves the
        conversation pinned to the previous family's child.
        """
        stored = {SESSION_CHILD_KEY: SessionChild(
            student_id="S1001", label="ليلى", guardian_id="gdn_OLD"
        ).to_metadata()}

        inherited = load_child_state(stored, guardian_id="gdn_NEW")

        self.assertFalse(inherited.is_set)
        self.assertEqual(inherited.guardian_id, "gdn_NEW")

    def test_a_pin_written_before_the_guardian_was_stamped_is_adopted(self):
        """Degrading the other way would re-ask every parent once on the day this ships,
        and buy nothing: the read re-checks the pin regardless."""
        stored = {SESSION_CHILD_KEY: {"student_id": "S1001", "label": "ليلى"}}

        adopted = load_child_state(stored, guardian_id="gdn_7f3a")

        self.assertEqual(adopted.student_id, "S1001")
        self.assertEqual(adopted.guardian_id, "gdn_7f3a")

    def test_unreadable_stored_state_is_a_fresh_conversation_not_an_error(self):
        for junk in ("not a dict", [], {"student_id": {"nested": True}}, {"bogus": 1}):
            with self.subTest(junk=junk):
                self.assertFalse(load_child_state({SESSION_CHILD_KEY: junk}).is_set)

    def test_pinning_a_different_child_drops_the_previous_one_wholesale(self):
        """A stale label attached to a new student id is worse than no label: it is the
        shape that names one child while answering about another."""
        child = SessionChild()
        child.pin(student_id="S1001", label="ليلى", gender="female")
        child.pin(student_id="S1002", label="", gender="")

        self.assertEqual(child.student_id, "S1002")
        self.assertEqual(child.label, "")
        self.assertEqual(child.gender, "unknown")

    def test_repinning_the_same_child_from_a_thinner_source_keeps_what_was_known(self):
        child = SessionChild()
        child.pin(student_id="S1001", label="ليلى", gender="female")
        child.pin(student_id="S1001")

        self.assertEqual(child.label, "ليلى")
        self.assertEqual(child.gender, "female")

    def test_pinning_nothing_is_a_no_op(self):
        child = SessionChild()
        child.pin(student_id="")
        self.assertFalse(child.is_set)


class RequestContextPinTests(unittest.TestCase):
    def _ctx(self, child=None):
        return ChatRequestContext.for_sync(
            user_id="guardian:gdn_7f3a",
            session_id="thread-1",
            caller=CallerIdentity(
                user_id="guardian:gdn_7f3a", guardian_id="gdn_7f3a", guardian_token="t"
            ),
            child=child,
        )

    def test_the_context_writes_through_to_the_object_that_gets_saved(self):
        child = SessionChild(guardian_id="gdn_7f3a")
        ctx = self._ctx(child)

        ctx.remember_child("S1001", label="ليلى")

        # Same object, not a copy — this is what makes the write-back free.
        self.assertEqual(child.student_id, "S1001")
        self.assertEqual(ctx.remembered_child, "S1001")
        self.assertEqual(ctx.remembered_child_label, "ليلى")

    def test_a_context_built_without_a_pin_still_works(self):
        """A staff session, a background job, a test. Not a parent, remembers nothing."""
        ctx = ChatRequestContext.for_sync(user_id="staff", session_id="s")
        self.assertEqual(ctx.remembered_child, "")
        ctx.remember_child("S1001")
        self.assertEqual(ctx.remembered_child, "S1001")

    def test_forgetting_clears_the_pin(self):
        child = SessionChild(guardian_id="gdn_7f3a")
        ctx = self._ctx(child)
        ctx.remember_child("S1001", label="ليلى")

        ctx.forget_child()

        self.assertEqual(ctx.remembered_child, "")
        self.assertEqual(child.student_id, "")


class ThreadIdTests(unittest.TestCase):
    def test_the_header_names_the_conversation(self):
        self.assertEqual(_thread_id("thread-abc", None), "thread-abc")

    def test_the_body_still_works_for_a_client_that_predates_the_header(self):
        self.assertEqual(_thread_id(None, "session_123"), "session_123")

    def test_the_header_wins_when_the_two_disagree(self):
        """A body default silently overriding a deliberately-set header is how a client
        ends up writing into `default_session` while believing otherwise."""
        self.assertEqual(_thread_id("thread-abc", "default_session"), "thread-abc")

    def test_naming_no_thread_still_produces_a_working_conversation(self):
        self.assertEqual(_thread_id(None, None), DEFAULT_THREAD)
        self.assertEqual(_thread_id("", "   "), DEFAULT_THREAD)

    def test_a_hostile_thread_id_is_sanitised_rather_than_refused(self):
        """It becomes part of a cache key and a database column. A client sending
        something odd should get a working conversation, not a 422 it cannot act on."""
        cleaned = _thread_id("a b/../*:\n x", None)

        self.assertNotIn("/", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("\n", cleaned)
        self.assertTrue(cleaned)

    def test_a_very_long_thread_id_cannot_overrun_the_storage_column(self):
        self.assertLessEqual(len(_thread_id("x" * 5000, None)), 120)

    def test_an_id_of_only_unsafe_characters_falls_back_rather_than_becoming_dashes(self):
        # Sanitising "///" to "---" would be a legal-looking key nobody chose.
        self.assertTrue(_thread_id("  ", None))


if __name__ == "__main__":
    unittest.main()
