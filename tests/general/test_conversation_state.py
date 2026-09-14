"""What a conversation remembers between turns, and for how long.

The second batch of fixes from the 2026-09-14 review. Each class here is one rule about
session state — the pending question, the child pin, the persistent note, the hints a
resumed search runs with — stated small enough that a regression names the rule.
"""
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.chat.clarification import build_pending_hitl, enter_turn

ASKED_AT = datetime(2026, 9, 14, 8, 0, tzinfo=timezone.utc)


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
