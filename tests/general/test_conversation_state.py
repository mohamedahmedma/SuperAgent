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
from backend.chat.persistent_note import NOTE_GUARDIAN_KEY, NOTE_KEY, usable_note
from backend.chat.request_context import ChatRequestContext
from backend.chat.resolution import unresolved
from backend.chat.turn_pipeline import TurnCollaborators, TurnPipeline
from backend.profiles import get_profile
from backend.rag.hitl_resume import build_hitl_resume_state
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


OLD_NOTE = "The parent asks about Omar (Year 6); prefers Arabic."


class TheNoteIsReadForTheGuardianItWasWrittenFor(unittest.TestCase):
    """`usable_note`, and the turn around it.

    The child pin already refuses to outlive the guardian it was resolved under. The
    persistent note — the assistant's own summary, which names the children — did not:
    after an administrator rebound the account to a different guardian, the previous
    family's note kept steering answers for the next. The note is stamped with the
    guardian it was written for, read only for the same one, and a refused note is
    cleared rather than left to be consulted again.
    """

    def test_the_guardian_it_was_written_for_reads_it(self):
        self.assertEqual((OLD_NOTE, False), usable_note({NOTE_KEY: OLD_NOTE, NOTE_GUARDIAN_KEY: "G-1"}, "G-1"))

    def test_another_guardian_reads_nothing_and_the_note_is_marked_for_clearing(self):
        self.assertEqual(("", True), usable_note({NOTE_KEY: OLD_NOTE, NOTE_GUARDIAN_KEY: "G-old"}, "G-1"))

    def test_a_note_written_before_the_stamp_is_adopted(self):
        self.assertEqual((OLD_NOTE, False), usable_note({NOTE_KEY: OLD_NOTE}, "G-1"))

    def test_a_session_with_no_guardian_reads_the_note_as_stored(self):
        self.assertEqual((OLD_NOTE, False), usable_note({NOTE_KEY: OLD_NOTE, NOTE_GUARDIAN_KEY: "G-1"}, ""))

    def test_a_turn_for_the_new_guardian_neither_reads_nor_keeps_the_old_note(self):
        storage = FakeStorage(metadata={NOTE_KEY: OLD_NOTE, NOTE_GUARDIAN_KEY: "G-old"})
        pipeline = _pipeline(storage)
        turn = _turn(pipeline)

        self.assertEqual("", turn.persistent_note, "the model is never shown another family's summary")
        patch = pipeline.save_metadata(turn)
        self.assertEqual(("", None), (patch[NOTE_KEY], patch[NOTE_GUARDIAN_KEY]))

    def test_the_note_update_starts_from_nothing_and_is_stamped_for_the_new_guardian(self):
        from langchain_core.messages import AIMessage, HumanMessage

        window = get_profile().agent.context_window_messages
        history = [
            HumanMessage(content=f"question {i}") if i % 2 == 0 else AIMessage(content=f"answer {i}")
            for i in range(window + 2)
        ]
        storage = FakeStorage(history, metadata={NOTE_KEY: OLD_NOTE, NOTE_GUARDIAN_KEY: "G-old"})
        update_note = Mock(return_value="The parent asks about Layla (Year 2).")
        pipeline = _pipeline(storage, update_note=update_note)
        turn = _turn(pipeline)
        pipeline.record_question(turn)
        turn.answer = "Layla is in Year 2."

        self.assertTrue(pipeline.schedule_note(turn))

        self.assertEqual("", update_note.call_args.args[0], "the old note is not offered as a starting point")
        self.assertEqual(
            [{NOTE_KEY: "The parent asks about Layla (Year 2).", NOTE_GUARDIAN_KEY: "G-1"}], storage.patches
        )
        # The job's write replaces the old note; the save must not clear it afterwards.
        self.assertNotIn(NOTE_KEY, pipeline.save_metadata(turn))

    def test_the_same_guardians_note_is_read_and_built_on(self):
        storage = FakeStorage(metadata={NOTE_KEY: OLD_NOTE, NOTE_GUARDIAN_KEY: "G-1"})
        update_note = Mock(return_value=OLD_NOTE + " Asked about the bus.")
        pipeline = _pipeline(storage, update_note=update_note)
        turn = _turn(pipeline)
        pipeline.record_question(turn)
        turn.answer = "The bus leaves at 07:30."

        self.assertEqual(OLD_NOTE, turn.persistent_note)
        pipeline.schedule_note(turn)
        self.assertEqual(OLD_NOTE, update_note.call_args.args[0])
        self.assertEqual("G-1", storage.metadata[NOTE_GUARDIAN_KEY])
        self.assertNotIn(NOTE_KEY, pipeline.save_metadata(turn))


class AResumedSearchRunsUnderTheHintsItsQuestionWasAskedWith(unittest.TestCase):
    """The planner runs on the fresh-question path only. A search paused for a
    clarification was resumed on a context nothing had planned into: no language, so both
    halves of a bilingual document competed; no year group; and the child's name back in
    the query. The hints travel in the resume state and reach the context before the
    search resumes — the same way the conditions and the round count already did.
    """

    ASKED = {"retrieval_status": "needs_clarification", "route": "clarify", "hitl_prompt": "Which term?"}

    def _graph_result(self) -> dict:
        """What the graph leaves behind when it stops to ask, on a planned turn."""
        ctx = ChatRequestContext.for_sync(user_id="parent", session_id="s")
        ctx.note_turn_plan(["fees"], [], language="ar", child_year="Year 6", child_names=["عمر"])
        return {
            **self.ASKED,
            "question": "مصاريف عمر",
            "language": "ar",
            "child_year": "Year 6",
            "retrieval_sections": ["fees"],
            "request_context": ctx,
        }

    def test_the_hints_are_carried_when_the_question_is_asked(self):
        carried = build_hitl_resume_state(self._graph_result())

        self.assertEqual(
            ("ar", "Year 6", ["fees"], ["عمر"]),
            (carried["language"], carried["child_year"], carried["retrieval_sections"], carried["child_names"]),
        )

    def _resume(self, resume_state: dict) -> dict:
        """The hints on the context at the moment retrieval resumes."""
        seen = {}

        def resume_retrieval(pending, user_text, ctx, resolution):
            seen.update(
                language=ctx.language,
                child_year=ctx.child_year,
                sections=list(ctx.retrieval_sections),
                names=list(ctx.child_names),
            )
            return {}

        pending = build_pending_hitl(self.ASKED, "مصاريف عمر", resume_state=resume_state)
        pipeline = _pipeline(
            FakeStorage(metadata={"pending_hitl": pending}),
            resolve_question=lambda question, history, **kwargs: unresolved(question, "unit"),
            resume_retrieval=resume_retrieval,
        )
        turn = _turn(pipeline, "الترم الأول")
        self.assertTrue(pipeline.resumes_a_search(turn))
        pipeline.run_resumed_search(turn)
        return seen

    def test_the_resumed_search_gets_them_back(self):
        seen = self._resume(build_hitl_resume_state(self._graph_result()))

        self.assertEqual({"language": "ar", "child_year": "Year 6", "sections": ["fees"], "names": ["عمر"]}, seen)

    def test_a_question_paused_before_the_hints_were_carried_resumes_as_an_unplanned_turn(self):
        seen = self._resume({"question": "مصاريف عمر", "route": "clarify", "retrieval_status": "needs_clarification"})

        self.assertEqual({"language": "", "child_year": "", "sections": [], "names": []}, seen)


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
