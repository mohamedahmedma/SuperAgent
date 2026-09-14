"""What a conversation remembers between turns, and for how long.

The second batch of fixes from the 2026-09-14 review. Each class here is one rule about
session state — the pending question, the child pin, the persistent note, the hints a
resumed search runs with — stated small enough that a regression names the rule.
"""
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.chat.background import InlineJobs
from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_context import SESSION_CHILD_KEY
from backend.chat.clarification import build_pending_hitl, enter_turn
from backend.chat.request_context import ChatRequestContext
from backend.chat.turn_pipeline import TurnCollaborators, TurnPipeline
from backend.profiles import get_profile
from tests.general.test_chat_hitl_resume import FakeStorage

ASKED_AT = datetime(2026, 9, 14, 8, 0, tzinfo=timezone.utc)

PARENT = CallerIdentity(user_id="parent", guardian_id="G-1", guardian_token="token")


def _pipeline(storage, **collaborators) -> TurnPipeline:
    """A turn pipeline over a stored conversation, with only the collaborators a test names.

    The rules here are decided before the planner or the agent is reached, so those stay
    None: a test that wandered into them would fail loudly rather than reach a model.
    """
    defaults = dict(
        conversations=storage,
        background=InlineJobs(),
        profile=get_profile(),
        plan=None,
        resolve_question=Mock(side_effect=AssertionError("nothing here needs the resolver")),
        create_agent=None,
        resume_retrieval=None,
        answer_model=None,
        session_title=None,
        update_note=None,
        context_type=ChatRequestContext,
    )
    return TurnPipeline(TurnCollaborators(**{**defaults, **collaborators}))


def _turn(pipeline: TurnPipeline, text="what are the fees?", caller=PARENT):
    """A turn opened, read, and given its context — up to the point a rule is decided."""
    turn = pipeline.open(text, caller.user_id, "s", caller)
    pipeline.enter(turn)
    pipeline.sync_context(turn)
    return turn


class TheChildPinIsWrittenOnlyWhenItChanged(unittest.TestCase):
    """`TurnPipeline.save_metadata`, on the pin.

    Every turn used to write its copy of the pin back, and the copy was a snapshot from
    the turn's start. Two messages in flight on one conversation — the second sent before
    the first was answered — and the later save put back the pin the earlier one had
    just replaced. A pin the turn did not change is left out of the patch, and the merge
    in the database then keeps whatever the other turn wrote.
    """

    def test_a_turn_that_pinned_nobody_writes_no_pin(self):
        pipeline = _pipeline(FakeStorage())
        turn = _turn(pipeline)

        self.assertNotIn(SESSION_CHILD_KEY, pipeline.save_metadata(turn))

    def test_a_turn_that_pinned_a_child_writes_the_pin(self):
        pipeline = _pipeline(FakeStorage())
        turn = _turn(pipeline)
        turn.ctx.remember_child("S-1", label="ليلى", gender="female")

        pin = pipeline.save_metadata(turn)[SESSION_CHILD_KEY]
        self.assertEqual(("S-1", "ليلى", "G-1"), (pin["student_id"], pin["label"], pin["guardian_id"]))

    def test_a_pin_another_turn_wrote_meanwhile_survives_this_turns_save(self):
        storage = FakeStorage()
        pipeline = _pipeline(storage)
        first = _turn(pipeline, "what are the fees?")
        second = _turn(pipeline, "and Layla's marks?")

        second.ctx.remember_child("S-1", label="ليلى")
        second.answer = "Layla's marks are …"
        pipeline.commit(second, pipeline.save_metadata(second))
        first.answer = "The fees are …"
        pipeline.commit(first, pipeline.save_metadata(first))

        self.assertEqual("S-1", storage.metadata[SESSION_CHILD_KEY]["student_id"])

    def test_a_pin_stored_before_the_guardian_stamp_is_stamped_once(self):
        """Adopting the pin is a change worth writing: from then on a rebind is caught.
        The turn after it finds the stamped pin and writes nothing."""
        storage = FakeStorage(metadata={SESSION_CHILD_KEY: {"student_id": "S-1", "label": "ليلى"}})
        pipeline = _pipeline(storage)

        stamped = pipeline.save_metadata(_turn(pipeline))[SESSION_CHILD_KEY]
        self.assertEqual("G-1", stamped["guardian_id"])

        storage.metadata[SESSION_CHILD_KEY] = stamped
        self.assertNotIn(SESSION_CHILD_KEY, pipeline.save_metadata(_turn(pipeline)))

    def test_a_pin_another_guardian_left_is_dropped(self):
        """The custody-transfer path: the account was rebound, and the previous family's
        pin is not this caller's to inherit — nor to leave lying in the metadata."""
        storage = FakeStorage(metadata={SESSION_CHILD_KEY: {"student_id": "S-9", "label": "عمر", "guardian_id": "G-old"}})
        pipeline = _pipeline(storage)

        pin = pipeline.save_metadata(_turn(pipeline))[SESSION_CHILD_KEY]
        self.assertEqual(("", "G-1"), (pin["student_id"], pin["guardian_id"]))


def _pending_asked_at(moment: datetime) -> dict:
    pending = build_pending_hitl(
        {"retrieval_status": "needs_clarification", "hitl_prompt": "Which year group?"}, "what are the fees?"
    )
    pending["created_at"] = moment.isoformat()
    return pending


class APendingQuestionExpires(unittest.TestCase):
    """`enter_turn` against a clarification asked some time ago.

    Read against the profile's `clarification_ttl_minutes` (a day). Past it the message
    is a fresh one and the stale question is marked for clearing; a question with no
    readable timestamp is kept, and a TTL of zero keeps every question forever.
    """

    def _entry(self, asked: datetime, now: datetime, **overrides):
        resolver = Mock(side_effect=AssertionError("an expired question must not reach the resolver"))
        return enter_turn(
            "when does school start?", [], {"pending_hitl": _pending_asked_at(asked)}, resolve=resolver, now=now, **overrides
        )

    def test_a_question_asked_yesterday_still_waits(self):
        entry = enter_turn(
            "Year 4", [], {"pending_hitl": _pending_asked_at(ASKED_AT)},
            resolve=Mock(return_value=SimpleNamespace(supersedes_pending_question=False, question="", constraints=[], resolved=False)),
            now=ASKED_AT + timedelta(hours=23),
        )
        self.assertIsNotNone(entry.pending_hitl)
        self.assertTrue(entry.is_hitl_resume)

    def test_a_question_asked_two_days_ago_is_gone_and_gets_cleared(self):
        entry = self._entry(ASKED_AT, ASKED_AT + timedelta(days=2))

        self.assertIsNone(entry.pending_hitl)
        self.assertTrue(entry.invalid_pending_hitl, "cleared at the save, so the next message is not read against it either")
        self.assertFalse(entry.is_hitl_resume)
        self.assertEqual("when does school start?", entry.effective_user_text)

    def test_a_question_with_no_readable_timestamp_is_kept(self):
        pending = _pending_asked_at(ASKED_AT)
        pending["created_at"] = "sometime"
        resolution = SimpleNamespace(supersedes_pending_question=False, question="", constraints=[], resolved=False)
        entry = enter_turn("Year 4", [], {"pending_hitl": pending}, resolve=Mock(return_value=resolution), now=ASKED_AT + timedelta(days=30))
        self.assertIsNotNone(entry.pending_hitl)

    def test_a_ttl_of_zero_never_expires(self):
        profile = SimpleNamespace(agent=SimpleNamespace(clarification_ttl_minutes=0))
        resolution = SimpleNamespace(supersedes_pending_question=False, question="", constraints=[], resolved=False)
        with patch("backend.chat.clarification.get_profile", return_value=profile):
            entry = enter_turn(
                "Year 4", [], {"pending_hitl": _pending_asked_at(ASKED_AT)},
                resolve=Mock(return_value=resolution), now=ASKED_AT + timedelta(days=365),
            )
        self.assertIsNotNone(entry.pending_hitl)


if __name__ == "__main__":
    unittest.main()
