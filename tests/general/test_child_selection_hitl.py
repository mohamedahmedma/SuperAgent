"""The "which of your children?" clarification, as a full round trip.

The planner ending a turn with a question is only half of the feature. The other half is
the message AFTER it: a bare name, arriving with no model call, which has to be matched
against the school's own roster and pinned to the session before the ORIGINAL question is
planned again. That second half is where this route differs from every other
clarification in the system, and every difference is somewhere it can break:

  * it is the only pending state with `resume_state: None`, so it only survives a
    persistence round trip because the schema was widened to allow that.
    `_current_pending_hitl` re-validates whatever was stored on every later message, so a
    field the schema rejects does not degrade the route — it deletes it;
  * it is the only route `_enter_turn` answers without a resolver call, so a regression
    there costs a model call per reply and, worse, folds a child's NAME into the query
    the clarify/scope_select routes build — searching the corpus for a child rather than
    for what the parent actually asked;
  * it is the only route whose answer is settled against a live roster read, so a reply
    matching nobody, matching several children, or arriving while the facade is down must
    all pin NOTHING and let the next plan ask again. Pinning on a doubtful match is the
    one failure this feature exists to prevent: it answers confidently about a child the
    parent never chose.

The existing suite covers the happy path. Everything below is a way the parent's reply,
or the world around it, can go wrong — plus the guarantee that the two older HITL routes
did not change when this branch was added in front of them.
"""
import os
import unittest
from unittest.mock import patch

from backend.chat import service
from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_resolution import resolve_child
from backend.chat.child_roster import ChildOption
from backend.chat.request_context import ChatRequestContext
from backend.chat.resolution import CORRECTION, STANDALONE, ResolvedQuestion
from backend.chat.service import (
    _child_choice_pending,
    _current_pending_hitl,
    _enter_turn,
    _pin_the_child_the_parent_named,
)
from backend.chat.signals import RequestSignals
from backend.chat.turn_policy import resolve_turn
from backend.schemas.chat import PendingHitlState

# One family, spelled the way a school's SIS actually spells it: a bare alif on one row,
# a hamza on another. Both are the same word to a parent typing it, and neither is
# necessarily the way they will type it.
ALI = ChildOption(student_id="S-1", label="علي احمد", gender="male", year_level="Year 4")
AHMED = ChildOption(student_id="S-2", label="أحمد احمد", gender="male")
SARA = ChildOption(
    student_id="S-3", label="سارة أحمد", gender="female", label_en="Sara Ahmed"
)

ROSTER_ROWS = [
    {"student_id": "S-1", "full_name_ar": "علي احمد", "gender": "male",
     "year_level": "Year 4"},
    {"student_id": "S-3", "full_name_ar": "سارة أحمد", "full_name_en": "Sara Ahmed",
     "gender": "female"},
]
# Two rows sharing a first name — the reason `_named` only trusts a UNIQUE match.
SAME_NAME_ROWS = [
    {"student_id": "S-1", "full_name_ar": "أحمد علي حسن", "gender": "male"},
    {"student_id": "S-2", "full_name_ar": "أحمد عمر حسن", "gender": "male"},
]
TWO_SONS_ROWS = [
    {"student_id": "S-1", "full_name_ar": "علي احمد", "gender": "male"},
    {"student_id": "S-2", "full_name_ar": "أحمد احمد", "gender": "male"},
]

QUESTION = "نتيجة ابني ايه؟"


class _Agent:
    """The school deployment's shape, spelled out rather than loaded.

    These are unit tests of the round trip, not assertions about a yaml file — that is
    what `test_school_profile.py` is for.
    """

    tools = ["search_knowledge_base", "get_student_records"]
    social_phrases = []
    social_reply_mode = "model"
    narrow_tools_to_the_turn = True
    year_reference_markers = ()


class _Copy:
    social = None
    out_of_domain = None
    which_child = "Which of your children do you mean?"


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _serves(rows):
    """A `requests.get` that answers the roster route with these rows."""

    def fake_get(url, headers=None, params=None, timeout=None):
        return _Response(200, {"guardian_id": "G-1", "students": rows})

    return fake_get


def _refuses(status=500):
    def fake_get(url, headers=None, params=None, timeout=None):
        return _Response(status)

    return fake_get


def _ctx(guardian_id="G-1", token="signed.identity.token"):
    return ChatRequestContext(
        user_id="user-1",
        session_id="session-1",
        caller=CallerIdentity(
            user_id="user-1", guardian_id=guardian_id, guardian_token=token
        ),
    )


def _plan(child, **signal_kwargs):
    signals = RequestSignals(question=QUESTION, **signal_kwargs)
    return resolve_turn(signals, agent_config=_Agent(), copy_config=_Copy(), child=child)


def _asking_plan(roster=(ALI, AHMED)):
    """A plan that ended the turn on "which child?" — two sons, and "my son"."""
    return _plan(
        resolve_child(reference="son", roster=list(roster)),
        about_child=True,
        child_question_kind="records",
    )


def _pending(question=QUESTION, roster=(ALI, AHMED)):
    return _child_choice_pending(_asking_plan(roster), question)


class _NoRosterCache(unittest.TestCase):
    """Base for every case that reads a roster.

    The roster sits behind a cache that may be a real Redis on the machine running this,
    and a shared cache carries one case's children into the next — which is how a suite
    starts reporting the wrong family for a test about an outage. Zero turns it off, for
    reading and for writing, which is the supported way to say "read it fresh".
    """

    def setUp(self):
        patcher = patch.dict(os.environ, {"CHILD_ROSTER_TTL_SECONDS": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)


class ThePendingQuestionSurvivesStorage(unittest.TestCase):
    """What `_child_choice_pending` builds is written to session metadata and read back
    on the next message through `PendingHitlState.model_validate(...)`. If that round
    trip loses or rejects anything, the reply arrives as an ordinary new question and the
    parent is asked which child forever."""

    def test_the_stored_state_reads_back_unchanged(self):
        stored = _pending()

        revived = PendingHitlState.model_validate(stored).model_dump()

        self.assertEqual(revived, stored)

    def test_a_null_resume_state_is_accepted_rather_than_rejected(self):
        """The schema change that made this route possible. A required `resume_state`
        would make a state carrying None fail validation on the way back in, and
        `_current_pending_hitl` would return None — silently turning the parent's reply
        into a fresh question about a child's name."""
        stored = _pending()
        self.assertIsNone(stored["resume_state"])

        self.assertIsNotNone(_current_pending_hitl(stored))

    def test_the_question_the_reply_answers_travels_with_it(self):
        """The next message is a name and nothing else. If the original question does not
        survive storage there is nothing left to answer once the child is known."""
        revived = _current_pending_hitl(_pending())

        self.assertEqual(revived["original_question"], QUESTION)
        self.assertEqual(revived["route"], "child_select")
        self.assertEqual(revived["retrieval_status"], "needs_child_choice")

    def test_the_options_offered_are_the_ones_read_back(self):
        revived = _current_pending_hitl(_pending())
        self.assertEqual(revived["options"], _asking_plan().child_options)

    def test_a_state_that_cannot_be_validated_is_dropped_rather_than_trusted(self):
        """Half-written, hand-edited, or written by an older build. Each of these would
        raise somewhere later if it were passed through; dropping it costs one repeated
        question, which the system already handles on every first message."""
        broken = {
            "an unknown route": {**_pending(), "route": "child_selection"},
            "an unknown status": {**_pending(), "retrieval_status": "needs_child"},
            "an empty original question": {**_pending(), "original_question": ""},
            "a resume state of the wrong shape": {
                **_pending(), "resume_state": {"question": ""}
            },
            "a missing id": {k: v for k, v in _pending().items() if k != "id"},
            "not a dict at all": ["a", "b"],
        }
        for label, value in broken.items():
            with self.subTest(label):
                self.assertIsNone(_current_pending_hitl(value))


class ThePlannerOnlyAsksWhenThereIsSomethingToAsk(unittest.TestCase):
    def test_a_plan_that_settled_the_child_stores_nothing(self):
        plan = _plan(
            resolve_child(reference="son", roster=[ALI, SARA]),
            about_child=True,
            child_question_kind="records",
        )
        self.assertTrue(plan.child_hint)
        self.assertIsNone(_child_choice_pending(plan, QUESTION))

    def test_a_plan_with_no_static_reply_stores_nothing(self):
        """Options with no copy to ask them with is not a question anybody was asked, so
        storing a pending state would leave the session waiting for a reply to a message
        that was never sent."""
        class _NoCopy(_Copy):
            which_child = None

        signals = RequestSignals(
            question=QUESTION, about_child=True, child_question_kind="records"
        )
        plan = resolve_turn(
            signals,
            agent_config=_Agent(),
            copy_config=_NoCopy(),
            child=resolve_child(reference="son", roster=[ALI, AHMED]),
        )
        self.assertIsNone(plan.static_reply)
        self.assertIsNone(_child_choice_pending(plan, QUESTION))

    def test_a_plan_with_no_options_stores_nothing(self):
        class _Bare:
            child_options = []
            static_reply = "Which of your children do you mean?"

        self.assertIsNone(_child_choice_pending(_Bare(), QUESTION))

    def test_a_plan_object_missing_the_field_entirely_stores_nothing(self):
        """An integrating deployment's own plan object, or an older one. Reading the
        field defensively is what keeps this from raising on a turn it has no business
        touching."""
        class _Foreign:
            static_reply = "..."

        self.assertIsNone(_child_choice_pending(_Foreign(), QUESTION))


class TheReplyReopensTheOriginalQuestion(unittest.TestCase):
    """`_enter_turn` on the message after the question. No roster is read at this stage —
    it only decides what the turn is about."""

    def test_choosing_an_offered_option_replays_the_question_not_the_name(self):
        entry = _enter_turn("أحمد احمد", [], {"pending_hitl": _pending()})

        self.assertEqual(entry.child_choice, "أحمد احمد")
        self.assertEqual(entry.effective_user_text, QUESTION)
        self.assertEqual(entry.original_question, QUESTION)

    def test_the_reply_is_never_folded_into_the_query_the_way_a_clarify_reply_is(self):
        """The clarify and scope_select routes concatenate the reply onto the original
        question, because there the reply NARROWS a search. Here the reply names a person,
        and folding it would retrieve for a child's name instead of for the fees, the
        timetable, or whatever was actually asked."""
        entry = _enter_turn("علي", [], {"pending_hitl": _pending()})

        self.assertNotIn("علي", entry.effective_user_text)
        self.assertNotIn("continuation", entry.effective_user_text.lower())
        self.assertEqual(entry.effective_user_text, QUESTION)

    def test_answering_which_child_is_not_a_hitl_resume(self):
        """There is no half-finished search to pick up. Treating it as a resume would
        send the turn down the resume path, which restores graph state this route never
        saved and reads a `resume_state` that is None."""
        entry = _enter_turn("علي", [], {"pending_hitl": _pending()})

        self.assertFalse(entry.is_hitl_resume)
        self.assertFalse(entry.superseded)
        self.assertIsNone(entry.resume_state)

    def test_the_reply_costs_no_resolver_call(self):
        """Matching a name to a roster row is not a judgement. A resolver call here pays
        a model to re-derive a fact the roster already holds — on every reply."""
        with patch.object(service, "resolve_turn_question") as resolver:
            service._enter_turn("علي", [], {"pending_hitl": _pending()})

        resolver.assert_not_called()

    def test_a_reply_matching_nobody_still_reopens_the_original_question(self):
        """"the older one" is not a name, and this stage cannot know that — the roster is
        read later. What matters is that the turn is still about the question the parent
        asked, so the next plan can ask again rather than answer about a phrase."""
        entry = _enter_turn("الكبير", [], {"pending_hitl": _pending()})

        self.assertEqual(entry.effective_user_text, QUESTION)
        self.assertEqual(entry.child_choice, "الكبير")

    def test_a_corrupt_pending_state_starts_a_fresh_turn(self):
        """Nothing is resumed, nothing is chosen, and the flag is raised so the turn can
        say the previous question was dropped."""
        metadata = {"pending_hitl": {"route": "child_select", "options": ["علي"]}}

        with patch.object(service, "resolve_turn_question") as resolver:
            entry = service._enter_turn("علي", [], metadata)

        self.assertTrue(entry.invalid_pending_hitl)
        self.assertEqual(entry.child_choice, "")
        self.assertEqual(entry.effective_user_text, "علي")
        resolver.assert_not_called()

    def test_no_pending_state_at_all_is_an_ordinary_turn(self):
        entry = _enter_turn("متى تبدأ الدراسة؟", [], {})

        self.assertFalse(entry.invalid_pending_hitl)
        self.assertEqual(entry.child_choice, "")
        self.assertEqual(entry.effective_user_text, "متى تبدأ الدراسة؟")


class TheOlderHitlRoutesAreUnaffected(unittest.TestCase):
    """The child branch was added in FRONT of the resolver call in `_enter_turn`. If it
    ever matched more than its own route, every retrieval clarification in the product
    would stop resuming and start being read as a child's name."""

    def _pending_clarify(self, route="clarify"):
        status = "needs_clarification" if route == "clarify" else "needs_scope_selection"
        return PendingHitlState(
            id="pending-1",
            original_question="ما هي المصاريف؟",
            prompt="Which year group?",
            options=["Year 4", "Year 5"],
            route=route,
            retrieval_status=status,
            answers=[],
            resume_state={
                "question": "ما هي المصاريف؟",
                "route": route,
                "retrieval_status": status,
            },
            created_at="2026-01-01T00:00:00+00:00",
        ).model_dump()

    def test_a_clarify_reply_still_takes_the_resolver_path(self):
        for route in ("clarify", "scope_select"):
            with self.subTest(route=route):
                resolved = ResolvedQuestion(
                    question="ما هي المصاريف؟", intent=STANDALONE, resolved=True
                )
                with patch.object(
                    service, "resolve_turn_question", return_value=resolved
                ) as resolver:
                    entry = service._enter_turn(
                        "Year 4", [], {"pending_hitl": self._pending_clarify(route)}
                    )

                resolver.assert_called_once()
                self.assertTrue(entry.is_hitl_resume)
                self.assertEqual(entry.child_choice, "")
                # Folded, which is the behaviour these routes have always had.
                self.assertIn("Year 4", entry.effective_user_text)
                self.assertIn("ما هي المصاريف؟", entry.effective_user_text)
                self.assertIsNotNone(entry.resume_state)

    def test_a_correction_still_supersedes_the_clarification(self):
        resolved = ResolvedQuestion(
            question="ما هي مواعيد الحافلة؟", intent=CORRECTION, resolved=True
        )
        with patch.object(service, "resolve_turn_question", return_value=resolved):
            entry = service._enter_turn(
                "no, I meant the bus", [], {"pending_hitl": self._pending_clarify()}
            )

        self.assertTrue(entry.superseded)
        self.assertFalse(entry.is_hitl_resume)
        self.assertEqual(entry.original_question, "ما هي مواعيد الحافلة؟")
        self.assertEqual(entry.child_choice, "")


class TheChosenChildIsSettledAgainstTheRoster(_NoRosterCache):
    """`_pin_the_child_the_parent_named`, driven through a canned facade.

    Pinning is what makes the answer stick for the rest of the conversation, so the bar
    for pinning is a UNIQUE roster match and nothing less.
    """

    def test_a_name_spelled_differently_from_the_option_still_pins(self):
        """The SIS holds the name with a hamza; a parent types a bare alif — or the other
        way round. Matching on raw strings fails this, which in practice means the parent
        is asked the same question again immediately after answering it correctly."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(ROSTER_ROWS)):
            pinned = _pin_the_child_the_parent_named(ctx, "سارة احمد")

        self.assertTrue(pinned)
        self.assertEqual(ctx.child.student_id, "S-3")
        self.assertEqual(ctx.child.gender, "female")

    def test_a_latin_spelling_of_an_arabic_row_pins(self):
        """A parent on an English keyboard answering a question rendered in Arabic."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(ROSTER_ROWS)):
            self.assertTrue(_pin_the_child_the_parent_named(ctx, "Sara"))

        self.assertEqual(ctx.child.student_id, "S-3")

    def test_the_label_pinned_is_the_rosters_own_name_not_what_was_typed(self):
        """One display name per child, chosen where the roster was read. A pin holding the
        parent's spelling would let the same child be named two ways in one turn."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(ROSTER_ROWS)):
            _pin_the_child_the_parent_named(ctx, "سارة احمد")

        self.assertEqual(ctx.child.label, "سارة أحمد")

    def test_a_reply_matching_nobody_pins_nothing(self):
        """"the older one" is a perfectly reasonable thing for a parent to type, and it is
        not an answer this code can act on. Guessing here would answer about a child
        nobody chose."""
        for reply in ("الكبير", "asdf", "the older one", "1"):
            with self.subTest(reply=reply):
                ctx = _ctx()
                with patch(
                    "backend.chat.child_roster.requests.get", _serves(ROSTER_ROWS)
                ):
                    pinned = _pin_the_child_the_parent_named(ctx, reply)

                self.assertFalse(pinned)
                self.assertEqual(ctx.child.student_id, "")

    def test_a_reply_matching_several_children_pins_nothing(self):
        """A shared first name is inside two rows. Picking the first would show one
        brother's marks while the parent watched, asking about the other."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(SAME_NAME_ROWS)):
            pinned = _pin_the_child_the_parent_named(ctx, "أحمد")

        self.assertFalse(pinned)
        self.assertEqual(ctx.child.student_id, "")

    def test_a_fuller_name_disambiguates_where_the_shared_one_does_not(self):
        """The other half of the rule above: the parent CAN settle it, by saying more."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(SAME_NAME_ROWS)):
            self.assertTrue(_pin_the_child_the_parent_named(ctx, "أحمد عمر"))

        self.assertEqual(ctx.child.student_id, "S-2")

    def test_an_empty_reply_pins_nothing_and_does_not_read_the_roster(self):
        ctx = _ctx()
        with patch(
            "backend.chat.child_roster.requests.get", side_effect=_refuses(500)
        ) as spy:
            self.assertFalse(_pin_the_child_the_parent_named(ctx, ""))

        spy.assert_not_called()

    def test_the_roster_being_unavailable_pins_nothing_and_does_not_raise(self):
        """A blip between the question and the answer. The parent typed a real name and it
        cannot be checked — which is one repeated question, never a guess."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _refuses(500)):
            pinned = _pin_the_child_the_parent_named(ctx, "سارة")

        self.assertFalse(pinned)
        self.assertEqual(ctx.child.student_id, "")

    def test_a_facade_that_raises_pins_nothing_and_does_not_raise(self):
        import requests as requests_module

        def explode(url, headers=None, params=None, timeout=None):
            raise requests_module.ConnectionError("no route to host")

        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", explode):
            self.assertFalse(_pin_the_child_the_parent_named(ctx, "سارة"))

        self.assertEqual(ctx.child.student_id, "")

    def test_a_rejected_identity_pins_nothing(self):
        """An expired sign-in between the question and the reply."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _refuses(401)):
            self.assertFalse(_pin_the_child_the_parent_named(ctx, "سارة"))

        self.assertEqual(ctx.child.student_id, "")

    def test_a_session_with_no_guardian_pins_nothing(self):
        """Staff, a background job, a test. Nobody to ask a roster about."""
        ctx = ChatRequestContext(user_id="staff-1", session_id="s")
        with patch(
            "backend.chat.child_roster.requests.get", side_effect=_serves(ROSTER_ROWS)
        ) as spy:
            self.assertFalse(_pin_the_child_the_parent_named(ctx, "سارة"))

        spy.assert_not_called()

    def test_no_context_at_all_pins_nothing(self):
        self.assertFalse(_pin_the_child_the_parent_named(None, "سارة"))


class TheNextPlanUsesWhateverWasPinned(_NoRosterCache):
    """The point of the round trip: what the following turn does with the pin, or with
    its absence. `resolve_child` is pure, so these run the real rules over the real pin
    object the previous stage wrote."""

    def _replan(self, ctx, roster=(ALI, AHMED)):
        """The original question, planned again now that the reply has been handled."""
        return _plan(
            resolve_child(reference="son", roster=list(roster), pin=ctx.child),
            about_child=True,
            child_question_kind="records",
        )

    def test_a_settled_reply_answers_the_original_question_without_asking_again(self):
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(TWO_SONS_ROWS)):
            _pin_the_child_the_parent_named(ctx, "علي")

        plan = self._replan(ctx)

        self.assertEqual(plan.child_hint, "علي احمد")
        self.assertEqual(plan.child_id, "S-1")
        self.assertFalse(plan.short_circuit)
        self.assertEqual(plan.child_options, [])

    def test_a_reply_that_pinned_nothing_asks_again_rather_than_guessing(self):
        """The failure this whole path is shaped to avoid: an unmatched reply must not
        leave a stale or arbitrary child pinned, and the turn has to end on the question
        again rather than on an answer about somebody."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(TWO_SONS_ROWS)):
            _pin_the_child_the_parent_named(ctx, "الكبير")

        plan = self._replan(ctx)

        self.assertTrue(plan.short_circuit)
        self.assertEqual(plan.child_options, ["علي احمد", "أحمد احمد"])
        self.assertEqual(plan.exposed_tools, [])
        self.assertEqual(plan.child_hint, "")

    def test_asking_again_produces_a_pending_state_again(self):
        """A parent may answer badly twice. The second question has to be storable too, or
        the third message is read as a fresh one and the loop is broken in the worst
        possible place."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(TWO_SONS_ROWS)):
            _pin_the_child_the_parent_named(ctx, "asdf")

        pending = _child_choice_pending(self._replan(ctx), QUESTION)

        self.assertIsNotNone(pending)
        self.assertEqual(pending["original_question"], QUESTION)
        self.assertIsNotNone(_current_pending_hitl(pending))

    def test_a_pin_the_parent_contradicts_does_not_survive_the_contradiction(self):
        """Pinned on a son, the parent then asks about their daughter. The pin is only
        consulted among the candidates the stated sex allows, so this moves on rather than
        answering about the brother."""
        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(ROSTER_ROWS)):
            _pin_the_child_the_parent_named(ctx, "علي")

        plan = _plan(
            resolve_child(reference="daughter", roster=[ALI, SARA], pin=ctx.child),
            about_child=True,
            child_question_kind="records",
        )

        self.assertEqual(plan.child_hint, "سارة أحمد")


class TheWholeRoundTrip(_NoRosterCache):
    """Three messages end to end, with no model and no network: the question that could
    not be settled, the reply, and the re-plan."""

    def test_a_question_a_reply_and_an_answer_about_the_child_that_was_chosen(self):
        asked = _asking_plan()
        self.assertTrue(asked.short_circuit)
        self.assertEqual(asked.exposed_tools, [])
        stored = {"pending_hitl": _child_choice_pending(asked, QUESTION)}

        # Message two: the parent taps the second option.
        with patch.object(service, "resolve_turn_question") as resolver:
            entry = service._enter_turn("أحمد احمد", [], stored)
        resolver.assert_not_called()

        ctx = _ctx()
        with patch("backend.chat.child_roster.requests.get", _serves(TWO_SONS_ROWS)):
            self.assertTrue(_pin_the_child_the_parent_named(ctx, entry.child_choice))

        replanned = _plan(
            resolve_child(reference="son", roster=[ALI, AHMED], pin=ctx.child),
            about_child=True,
            child_question_kind="records",
        )

        self.assertEqual(entry.effective_user_text, QUESTION)
        self.assertEqual(replanned.child_hint, "أحمد احمد")
        self.assertEqual(replanned.child_id, "S-2")
        self.assertFalse(replanned.short_circuit)
        self.assertEqual(replanned.forced_tool, "get_student_records")


if __name__ == "__main__":
    unittest.main()
