"""Which child, over a SEQUENCE of messages rather than one.

Every other file in this feature asks the same question once: given this message, this
roster and this pin, which child is it about. That is the unit, and it is well covered.
What nobody covers is the thing a parent actually experiences — a conversation, where
one message settles a child and the next four inherit that decision without saying so.

The pin is what carries it, and a pin is the most dangerous piece of state in the
feature: it is the only input to `resolve_child` that the CURRENT message did not
supply. Everything it can do wrong is a wrong child's marks shown to a parent who
watched the assistant get their previous question right:

  * it must carry a settled child through "طيب وغيابها؟" — a message with no subject
    at all — so a parent is asked at most once;
  * it must LOSE to anything the new message actually states: a sibling's name, a sex
    that contradicts it, "my kids";
  * it must never rescue a turn it does not belong to. Pinned on a son, "my daughter"
    with two daughters on file is still a question, and a pin that answered it would be
    answering with a child the parent had just excluded;
  * it must fall through when it names a child the roster no longer carries — withdrawn,
    unlinked, or resolved under a guardian this account is no longer bound to;
  * and a turn that resolves nobody must leave it exactly as it was, because "this
    message was not about a child" is not evidence that the previous one wasn't either.

The narrowing has a sequence dimension too: two consecutive turns about the SAME settled
child can need opposite tools ("her marks", then "the fees for her year"), and a plan
that narrowed once and remembered it would answer the second from the wrong place.

The last class drives whole conversations through the real `plan_turn` with the envelope,
the resolver and the roster read all stubbed, so the ladder underneath these rules is
exercised end to end without a model or a socket.
"""
import os
import unittest
from unittest.mock import patch

from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_context import SessionChild
from backend.chat.child_resolution import resolve_child
from backend.chat.child_roster import ChildOption
from backend.chat.orchestrator import plan_turn
from backend.chat.request_context import ChatRequestContext
from backend.chat.resolution import unresolved
from backend.chat.signals import RequestSignals
from backend.chat.turn_policy import resolve_turn
from backend.profiles.registry import load_profile, set_profile

KNOWLEDGE_TOOL = "search_knowledge_base"
RECORDS_TOOL = "get_student_records"

GUARDIAN = "G-conv"
PARENT_TOKEN = "signed.identity.token"
SESSION = "conversation-1"

# --- fixtures, deliberately local -------------------------------------------------
#
# Nothing is imported from a sibling suite. These rows describe the family this file's
# conversations are about — two daughters, a son, and one child whose gender column the
# registrar has not filled — and sharing them would tie two files' idea of a roster
# together.

LAYLA = ChildOption(student_id="S-1", label="ليلى أحمد", gender="female", year_level="Year 4")
OMAR = ChildOption(student_id="S-2", label="عمر أحمد", gender="male", year_level="Year 6")
SARA = ChildOption(student_id="S-3", label="سارة أحمد", gender="female")
#: The state every child is in until a registrar uploads the column. Matches BOTH sexes,
#: so it is a candidate for "my son" and for "my daughter" alike.
HANI = ChildOption(student_id="S-4", label="هاني أحمد", gender="unknown")

FAMILY = [LAYLA, OMAR, SARA]


def _pin(child: ChildOption | None = None, *, student_id: str = "", guardian_id: str = GUARDIAN) -> SessionChild:
    """The pin a previous turn would have left behind.

    Written the way production writes it — `ChatRequestContext.remember_child`, which is
    what the records tool and the child-choice resume both call — rather than by hand,
    so a change to what a settled turn stores is visible here.
    """
    pinned = SessionChild(guardian_id=guardian_id)
    if child is not None:
        pinned.pin(student_id=child.student_id, label=child.label, gender=child.gender)
    elif student_id:
        pinned.pin(student_id=student_id, label="somebody", gender="female")
    return pinned


class _Agent:
    """The school deployment's shape, spelled out so one knob can move at a time."""

    tools = [KNOWLEDGE_TOOL, RECORDS_TOOL]
    social_phrases = []
    social_reply_mode = "model"
    narrow_tools_to_the_turn = True
    year_reference_markers = ()


class _Copy:
    social = None
    out_of_domain = None
    which_child = "Which child do you mean?"


def _plan(child, *, question="q", **signal_kwargs):
    signals = RequestSignals(question=question, **signal_kwargs)
    return resolve_turn(
        signals, agent_config=_Agent(), copy_config=_Copy(), child=child
    )


def _turn(reference, *, pin=None, name="", roster=FAMILY):
    """One conversational turn's child decision, given what the last one left behind."""
    return resolve_child(
        reference=reference, child_name=name, roster=list(roster), pin=pin
    )


class TheSettledChildCarriesToTheNextMessage(unittest.TestCase):
    """"طيب وغيابها؟" — a message whose subject is entirely in the previous one."""

    def test_a_pronoun_followup_resolves_to_the_pinned_child(self):
        """The failure this catches is being asked "which child?" twice in a row.

        A message like «طيب وغيابها؟» states nothing at all, so every rung above the pin
        abstains and the pin is the only thing that can answer it.
        """
        settled = _turn("context", pin=_pin(LAYLA))

        self.assertTrue(settled.resolved)
        self.assertEqual(settled.student_id, LAYLA.student_id)
        self.assertEqual(settled.source, "pin")
        self.assertFalse(settled.ask)

    def test_the_followup_turn_asks_nothing_and_still_runs_the_agent(self):
        """A resolved follow-up is an ordinary turn: a hint on the plan, no options, and
        no static reply standing in for an answer."""
        plan = _plan(_turn("context", pin=_pin(LAYLA)), about_child=True)

        self.assertEqual(plan.child_hint, LAYLA.label)
        self.assertEqual(plan.child_options, [])
        self.assertFalse(plan.short_circuit)

    def test_the_pinned_childs_year_travels_with_them_on_every_later_turn(self):
        """The pin stores an id and a label only. The year has to come back off the
        roster row each turn, or a follow-up about a school matter is answered for the
        wrong year group while still naming the right child."""
        plan = _plan(_turn("context", pin=_pin(LAYLA)), about_child=True)

        self.assertEqual(plan.child_year, LAYLA.year_level)

    def test_the_same_pin_answers_turn_after_turn(self):
        """Nothing about resolving consumes the pin: five vague messages in a row are
        five resolutions to the same child, not one and then four questions."""
        pinned = _pin(OMAR)
        for turn in range(1, 6):
            with self.subTest(turn=turn):
                self.assertEqual(_turn("context", pin=pinned).student_id, OMAR.student_id)

    def test_the_first_vague_message_of_a_conversation_still_asks(self):
        """The control for the case above: without a pin there is nothing to carry, so
        the parent is asked once — which is the one question this feature permits."""
        opening = _turn("context")

        self.assertTrue(opening.ask)
        self.assertEqual(opening.option_labels, [c.label for c in FAMILY])


class ANameMovesTheConversationOn(unittest.TestCase):
    """A sibling switch. The named rung is above the pin for exactly this."""

    def test_naming_a_sibling_beats_the_pin(self):
        switched = _turn("named", name="عمر", pin=_pin(LAYLA))

        self.assertEqual(switched.student_id, OMAR.student_id)
        self.assertEqual(switched.source, "named")

    def test_a_name_matching_nobody_asks_rather_than_falling_back_on_the_pin(self):
        """The parent named somebody. Answering about the previously settled child while
        they watch is worse than one more question — and it is the failure a pin
        consulted after a failed name lookup would produce every time."""
        stranger = _turn("named", name="خالد", pin=_pin(LAYLA))

        self.assertFalse(stranger.resolved)
        self.assertTrue(stranger.ask)
        self.assertEqual(stranger.option_labels, [c.label for c in FAMILY])

    def test_an_ambiguous_name_asks_among_the_matches_and_not_the_pin(self):
        """«أحمد» is inside all three of these children's names. The pin is one of the
        matches, and picking it would be the resolver choosing for the parent on the
        strength of the last question rather than this one."""
        several = _turn("named", name="أحمد", pin=_pin(LAYLA))

        self.assertTrue(several.ask)
        self.assertEqual(several.option_labels, [c.label for c in FAMILY])

    def test_naming_the_pinned_child_again_is_still_credited_to_the_name(self):
        """Same answer, different provenance — and the provenance is what the trace and
        the reasons report, so a conversation that keeps re-stating the name should not
        read as one that has been coasting on a pin."""
        again = _turn("named", name="ليلى", pin=_pin(LAYLA))

        self.assertEqual(again.student_id, LAYLA.student_id)
        self.assertEqual(again.source, "named")


class AStatedSexOverridesAContradictingPin(unittest.TestCase):
    """"my son" after a conversation about a daughter is a change of subject."""

    def test_a_son_after_settling_on_a_daughter_switches_to_the_son(self):
        switched = _turn("son", pin=_pin(LAYLA))

        self.assertEqual(switched.student_id, OMAR.student_id)
        self.assertEqual(switched.source, "gender")

    def test_the_pin_still_breaks_a_tie_among_children_the_sex_allows(self):
        """The pin is not ignored when a sex is stated — it is consulted AFTER the
        filter. Pinned on Layla, "my daughter" with two daughters on file is the
        follow-up it was created to answer."""
        stayed = _turn("daughter", pin=_pin(LAYLA))

        self.assertEqual(stayed.student_id, LAYLA.student_id)
        self.assertEqual(stayed.source, "pin")

    def test_a_pin_outside_the_stated_sex_does_not_rescue_an_ambiguous_turn(self):
        """Pinned on the son, "my daughter", two daughters on file. The pin is not a
        candidate, so nothing can settle it and the parent is asked — between the
        DAUGHTERS, never the whole family."""
        asked = _turn("daughter", pin=_pin(OMAR))

        self.assertFalse(asked.resolved)
        self.assertTrue(asked.ask)
        self.assertEqual(asked.option_labels, [LAYLA.label, SARA.label])

    def test_an_unfilled_gender_column_can_keep_a_turn_open_against_a_pin(self):
        """`unknown` is a candidate for both sexes, so a child the registrar has not
        classified sits alongside the son. Pinned on a daughter, "my son" therefore has
        two candidates and neither is the pin: still a question."""
        asked = _turn("son", pin=_pin(LAYLA), roster=[LAYLA, OMAR, HANI])

        self.assertTrue(asked.ask)
        self.assertEqual(asked.option_labels, [OMAR.label, HANI.label])

    def test_a_sex_nobody_on_file_can_be_asks_the_whole_roster_despite_the_pin(self):
        """The parent's wording is better evidence than a half-filled column, so the
        offer widens back to everyone rather than quietly continuing with the pin."""
        asked = _turn("son", pin=_pin(LAYLA), roster=[LAYLA, SARA])

        self.assertTrue(asked.ask)
        self.assertEqual(asked.option_labels, [LAYLA.label, SARA.label])

    def test_the_switch_ends_the_turn_with_the_question_rather_than_a_guess(self):
        """The plan side of the case above: no hint, no tools, and the profile's own
        copy as the reply. An agent built here could only guess."""
        plan = _plan(_turn("daughter", pin=_pin(OMAR)), about_child=True,
                     child_question_kind="records")

        self.assertEqual(plan.child_hint, "")
        self.assertEqual(plan.child_options, [LAYLA.label, SARA.label])
        self.assertTrue(plan.short_circuit)
        self.assertEqual(plan.exposed_tools, [])
        self.assertEqual(plan.forced_tool, "")


class APinTheRosterNoLongerCarries(unittest.TestCase):
    """A conversation outliving the access it was pinned under."""

    def test_a_child_withdrawn_mid_conversation_falls_through_to_asking(self):
        """The pin names a real id that is simply not on this roster read any more —
        withdrawn, unlinked, or a custody change applied between two messages. Matching
        it against the candidates by id is what stops it resolving to nobody: an
        id-carrying `ResolvedChild` for a child the school will refuse is a refusal the
        parent cannot make sense of."""
        gone = _turn("context", pin=_pin(student_id="S-withdrawn"))

        self.assertFalse(gone.resolved)
        self.assertEqual(gone.student_id, "")
        self.assertTrue(gone.ask)
        self.assertEqual(gone.option_labels, [c.label for c in FAMILY])

    def test_a_stale_pin_never_survives_as_a_label_only(self):
        """A pin whose label is still readable but whose id is gone must contribute
        nothing at all — not the name, and above all not the id a records read would be
        issued for."""
        stale = _pin(student_id="S-withdrawn")
        stale.label = LAYLA.label
        plan = _plan(_turn("context", pin=stale), about_child=True)

        self.assertEqual(plan.child_hint, "")
        self.assertEqual(plan.child_id, "")

    def test_an_only_child_is_still_resolved_when_the_pin_has_gone_stale(self):
        """The only-child rung fires above the pin, so a family that cannot be ambiguous
        is never asked because of state that went bad."""
        settled = _turn("context", pin=_pin(student_id="S-withdrawn"), roster=[LAYLA])

        self.assertTrue(settled.resolved)
        self.assertEqual(settled.source, "only_child")

    def test_a_pin_from_another_guardian_is_not_inherited_by_this_conversation(self):
        """The custody-transfer path: an administrator rebinds the account between two
        messages. `chat_sessions` is keyed by username and the right to read a child is
        keyed by the guardian, so without the stamp the conversation would carry on
        naming the previous family's child."""
        stored = _pin(LAYLA, guardian_id="G-somebody-else").to_metadata()
        inherited = SessionChild.from_metadata({"child_context": stored}, guardian_id=GUARDIAN)

        self.assertFalse(inherited.is_set)
        self.assertTrue(_turn("context", pin=inherited).ask)


class PluralMidConversation(unittest.TestCase):
    """"اولادي" — the one reference that must neither narrow nor ask."""

    def test_a_plural_message_does_not_collapse_onto_the_pin(self):
        """The failure is silent and the worst kind: a parent asks about all of their
        children and is answered about one of them, correctly, so nothing looks wrong."""
        both = _turn("plural", pin=_pin(LAYLA))

        self.assertFalse(both.resolved)
        self.assertEqual(both.student_id, "")

    def test_a_plural_message_does_not_ask_which_child_either(self):
        """Asking "which one?" after "all of them" is worse than not helping. The tool
        can still read the whole roster."""
        both = _turn("plural", pin=_pin(LAYLA))

        self.assertFalse(both.ask)
        self.assertEqual(both.options, ())

    def test_a_plural_turn_keeps_every_tool_bound(self):
        """Nothing was settled, so the narrowing must not fire off the previous turn's
        child. Both tools stay bound and nothing is forced."""
        plan = _plan(_turn("plural", pin=_pin(LAYLA)), about_child=True,
                     child_question_kind="records")

        self.assertIsNone(plan.exposed_tools)
        self.assertEqual(plan.forced_tool, "")
        self.assertEqual(plan.child_hint, "")

    def test_a_plural_message_in_a_one_child_family_is_still_that_child(self):
        """Only-child sits above plural on purpose: "how are my kids doing" from a
        parent with one child on file is about that child, and asking or abstaining
        would both be pedantry."""
        settled = _turn("plural", pin=_pin(LAYLA), roster=[LAYLA])

        self.assertTrue(settled.resolved)
        self.assertEqual(settled.source, "only_child")

    def test_the_pin_survives_a_plural_turn_for_the_message_after_it(self):
        """A plural turn is not a reset. The next vague message is still about the child
        the conversation settled on before it."""
        pinned = _pin(LAYLA)
        _turn("plural", pin=pinned)

        self.assertEqual(_turn("context", pin=pinned).student_id, LAYLA.student_id)


class ATurnThatResolvesNobodyLeavesThePinAlone(unittest.TestCase):
    """Resolution is pure. Nothing in this chain may write to the pin."""

    def test_an_ambiguous_turn_does_not_clear_the_pin(self):
        """Pinned on the son, asked "my daughter", the parent is asked which one. That
        question is not evidence the son was wrong — if the parent walks away and comes
        back with «وهو؟», the pin still has to be there."""
        pinned = _pin(OMAR)
        _turn("daughter", pin=pinned)

        self.assertEqual(pinned.student_id, OMAR.student_id)
        self.assertEqual(pinned.label, OMAR.label)

    def test_a_failed_name_lookup_does_not_clear_the_pin(self):
        pinned = _pin(LAYLA)
        _turn("named", name="خالد", pin=pinned)

        self.assertEqual(pinned.student_id, LAYLA.student_id)

    def test_a_turn_about_no_child_at_all_leaves_the_pin_untouched(self):
        """Most turns in a real conversation are this one — "what time does the bus
        leave" between two questions about a child."""
        pinned = _pin(LAYLA)
        resolve_child(reference="none", roster=[], pin=pinned)

        self.assertEqual(pinned.student_id, LAYLA.student_id)

    def test_resolving_a_different_child_does_not_rewrite_the_pin_by_itself(self):
        """`resolve_child` decides; only `remember_child` writes. Keeping those apart is
        what lets a turn resolve a child, fail at the facade, and leave the previous,
        working pin in place."""
        pinned = _pin(LAYLA)
        switched = _turn("named", name="عمر", pin=pinned)

        self.assertEqual(switched.student_id, OMAR.student_id)
        self.assertEqual(pinned.student_id, LAYLA.student_id)

    def test_pinning_a_new_child_drops_the_previous_ones_details(self):
        """The write side, for the moment the caller does settle the switch: a pin that
        kept the old label would name Layla while reading Omar's record."""
        pinned = _pin(LAYLA)
        pinned.pin(student_id=OMAR.student_id, label=OMAR.label, gender=OMAR.gender)

        self.assertEqual(pinned.label, OMAR.label)
        self.assertEqual(pinned.gender, OMAR.gender)


class TheNarrowingIsDecidedAfreshEveryTurn(unittest.TestCase):
    """Two turns about the same child can need opposite tools."""

    def test_a_records_turn_then_a_school_matter_narrows_the_other_way(self):
        """«درجاتها؟» then «ومصاريف سنتها؟» — same child, same pin, opposite tools. A
        narrowing cached with the child would answer the second from the record."""
        pinned = _pin(LAYLA)

        records = _plan(_turn("context", pin=pinned), about_child=True,
                        child_question_kind="records")
        school_matter = _plan(_turn("context", pin=pinned), about_child=True,
                              child_question_kind="school_matter")

        self.assertEqual(records.exposed_tools, [RECORDS_TOOL])
        self.assertEqual(records.forced_tool, RECORDS_TOOL)
        self.assertEqual(school_matter.exposed_tools, [KNOWLEDGE_TOOL])
        self.assertEqual(school_matter.forced_tool, KNOWLEDGE_TOOL)

    def test_a_turn_that_needs_both_rebinds_everything_after_a_narrowed_one(self):
        """A conversation cannot get stuck narrow. `both` is the classifier's abstention
        as well as a real verdict, and either way it must restore every tool."""
        pinned = _pin(LAYLA)
        _plan(_turn("context", pin=pinned), about_child=True, child_question_kind="records")
        after = _plan(_turn("context", pin=pinned), about_child=True,
                      child_question_kind="both")

        self.assertIsNone(after.exposed_tools)
        self.assertEqual(after.forced_tool, "")

    def test_the_turn_after_an_unanswered_question_binds_everything_again(self):
        """The "which child?" turn binds nothing, because there is no lookup to make.
        The turn after it, once the parent has chosen, is an ordinary narrowed turn —
        the empty list must not persist."""
        asked = _plan(_turn("context"), about_child=True, child_question_kind="records")
        chosen = _plan(_turn("context", pin=_pin(LAYLA)), about_child=True,
                       child_question_kind="records")

        self.assertEqual(asked.exposed_tools, [])
        self.assertEqual(chosen.exposed_tools, [RECORDS_TOOL])

    def test_switching_child_mid_conversation_re_narrows_for_the_new_child(self):
        """The narrowing follows whichever child the CURRENT turn settled on, so the
        forced records read is issued against the sibling's id, not the pin's."""
        plan = _plan(_turn("named", name="عمر", pin=_pin(LAYLA)), about_child=True,
                     child_question_kind="records")

        self.assertEqual(plan.child_id, OMAR.student_id)
        self.assertEqual(plan.forced_tool, RECORDS_TOOL)


# --- the whole ladder, offline ----------------------------------------------------


class _Conversation:
    """A sequence of planned turns over one context, with no model and no socket.

    The envelope, the resolver and the roster read are all injected, which is the
    arrangement `plan_turn` was given them for. What is NOT simulated is anything that
    writes the pin: production writes it from the records tool and from the child-choice
    resume, and `settle` below is the only place this harness does it, so a test can say
    exactly which turn settled the conversation.
    """

    def __init__(self, roster_rows):
        self.ctx = ChatRequestContext(
            user_id="user-conv",
            session_id=SESSION,
            caller=CallerIdentity(
                user_id="user-conv", guardian_id=GUARDIAN, guardian_token=PARENT_TOKEN
            ),
        )
        self._rows = list(roster_rows)
        self.reads = 0

    def _fetch(self, guardian_id, token, request_id):
        self.reads += 1
        return "ok", list(self._rows)

    def ask(self, question, *, reference="context", kind="records", name=""):
        def envelope(_question, _history, _config):
            return {
                "scope": "in_domain",
                "about_child": True,
                "child_reference": reference,
                "child_name": name,
                "child_question_kind": kind,
            }

        plan, signals = plan_turn(
            question,
            [],
            self.ctx,
            envelope_invoke=envelope,
            resolution=unresolved(question, "test"),
            roster_fetch=self._fetch,
        )
        return plan

    def settle(self, plan):
        """What the records tool does once it has actually read a child."""
        self.ctx.remember_child(plan.child_id, label=plan.child_hint)

    def close(self):
        self.ctx.close()


class AWholeConversationThroughThePlanner(unittest.TestCase):
    """The rules above, but reached through the real ladder rather than called directly.

    The pure cases cannot see what `plan_turn` adds on the way out — the roster read, the
    envelope's distrust of its own fields, and `note_turn_plan` copying the settled child
    onto the context for the records tool to prefer over the model's spelling.
    """

    def setUp(self):
        env = patch.dict(os.environ, {"CHILD_ROSTER_TTL_SECONDS": "0"})
        env.start()
        self.addCleanup(env.stop)
        set_profile(load_profile("school"))
        self.addCleanup(set_profile, None)
        self.chat = _Conversation(
            [
                {"student_id": "S-1", "full_name_ar": "ليلى أحمد", "full_name_en": "Layla Ahmed",
                 "gender": "female", "year_level": "Year 4"},
                {"student_id": "S-2", "full_name_ar": "عمر أحمد", "full_name_en": "Omar Ahmed",
                 "gender": "male", "year_level": "Year 6"},
                {"student_id": "S-3", "full_name_ar": "سارة أحمد", "full_name_en": "Sara Ahmed",
                 "gender": "female"},
            ]
        )
        self.addCleanup(self.chat.close)

    def test_a_named_turn_settles_the_child_and_the_next_pronoun_turn_inherits_them(self):
        """The whole feature in two messages: name her once, then ask about "her"."""
        first = self.chat.ask("درجات ليلى كام؟", reference="named", name="ليلى")
        self.assertEqual(first.child_id, "S-1")
        self.chat.settle(first)

        second = self.chat.ask("طيب وغيابها؟", reference="context", kind="records")

        self.assertEqual(second.child_hint, "ليلى أحمد")
        self.assertEqual(second.child_options, [])
        self.assertFalse(second.short_circuit)

    def test_the_settled_child_reaches_the_records_tool_as_an_id(self):
        """`planned_child_id` is what makes the tool read the roster's child rather than
        the model's spelling of a name. A follow-up that resolved only through the pin
        has to put it there just as a named turn does."""
        self.chat.settle(self.chat.ask("درجات ليلى؟", reference="named", name="ليلى"))
        self.chat.ask("طيب وغيابها؟")

        self.assertEqual(self.chat.ctx.planned_child_id, "S-1")
        self.assertEqual(self.chat.ctx.planned_child_label, "ليلى أحمد")

    def test_naming_a_sibling_on_the_next_message_moves_the_planned_child(self):
        self.chat.settle(self.chat.ask("درجات ليلى؟", reference="named", name="ليلى"))
        self.chat.ask("وعمر عامل ايه؟", reference="named", name="عمر")

        self.assertEqual(self.chat.ctx.planned_child_id, "S-2")

    def test_a_stated_sex_overrides_the_pin_through_the_whole_ladder(self):
        """Settled on a daughter, then «ابني» — one son on file, so the turn moves to
        him without a question."""
        self.chat.settle(self.chat.ask("درجات ليلى؟", reference="named", name="ليلى"))
        plan = self.chat.ask("وابني عامل ايه؟", reference="son")

        self.assertEqual(plan.child_id, "S-2")
        self.assertEqual(plan.child_hint, "عمر أحمد")

    def test_a_vague_opening_message_ends_the_turn_with_the_question(self):
        """No pin yet, three children, nothing in the message: the deterministic ask,
        with the profile's own copy and no agent built."""
        plan = self.chat.ask("كيف حاله في الدراسة؟")

        self.assertTrue(plan.short_circuit)
        self.assertEqual(len(plan.child_options), 3)
        self.assertEqual(plan.exposed_tools, [])

    def test_asking_which_child_pins_nothing_onto_the_conversation(self):
        """A question is not an answer. Pinning the first candidate here would make the
        next vague message resolve to a child nobody chose."""
        self.chat.ask("كيف حاله في الدراسة؟")

        self.assertEqual(self.chat.ctx.planned_child_id, "")
        self.assertFalse(self.chat.ctx.child.is_set)

    def test_a_plural_message_after_a_settled_child_narrows_nothing(self):
        """«اولادي» mid-conversation: no hint, no question, and every tool bound so the
        tool can read the whole roster."""
        self.chat.settle(self.chat.ask("درجات ليلى؟", reference="named", name="ليلى"))
        plan = self.chat.ask("عايز اعرف مستوى اولادي", reference="plural")

        self.assertEqual(plan.child_hint, "")
        self.assertEqual(plan.child_options, [])
        self.assertIsNone(plan.exposed_tools)

    def test_the_pin_still_answers_the_message_after_a_plural_one(self):
        self.chat.settle(self.chat.ask("درجات ليلى؟", reference="named", name="ليلى"))
        self.chat.ask("عايز اعرف مستوى اولادي", reference="plural")
        plan = self.chat.ask("طيب وغيابها؟")

        self.assertEqual(plan.child_id, "S-1")

    def test_two_consecutive_turns_about_one_child_narrow_to_opposite_tools(self):
        self.chat.settle(self.chat.ask("درجات ليلى؟", reference="named", name="ليلى"))
        records = self.chat.ask("طيب وغيابها؟", kind="records")
        school_matter = self.chat.ask("ومصاريف سنتها كام؟", kind="school_matter")

        self.assertEqual(records.exposed_tools, ["get_student_records"])
        self.assertEqual(school_matter.exposed_tools, ["search_knowledge_base"])

    def test_the_forced_tool_is_re_decided_rather_than_carried_over(self):
        """`forced_tool` reaches the middleware through the context, which lives for one
        turn — but the plan is what sets it, and a school-matter turn following a records
        one must force the knowledge tool, not the record."""
        self.chat.settle(self.chat.ask("درجات ليلى؟", reference="named", name="ليلى"))
        self.chat.ask("طيب وغيابها؟", kind="records")
        plan = self.chat.ask("ومصاريف سنتها كام؟", kind="school_matter")

        self.assertEqual(plan.forced_tool, "search_knowledge_base")

    def test_a_child_withdrawn_between_two_messages_reopens_the_question(self):
        """The pin outlives the roster row. The next vague message must ask rather than
        plan a read for a child the school will refuse."""
        self.chat.settle(self.chat.ask("درجات ليلى؟", reference="named", name="ليلى"))
        self.chat._rows = [row for row in self.chat._rows if row["student_id"] != "S-1"]

        plan = self.chat.ask("طيب وغيابها؟")

        self.assertEqual(plan.child_hint, "")
        self.assertTrue(plan.short_circuit)
        self.assertEqual(len(plan.child_options), 2)

    def test_the_trace_of_a_followup_reports_the_decision_and_never_the_name(self):
        """Persisted per message and streamed to the browser. A follow-up resolved off a
        pin is exactly the turn where a name would leak without the parent having typed
        one."""
        self.chat.settle(self.chat.ask("درجات ليلى؟", reference="named", name="ليلى"))
        trace = self.chat.ask("طيب وغيابها؟").as_trace()

        self.assertTrue(trace["turn_child_resolved"])
        self.assertFalse(trace["turn_child_asked"])
        self.assertNotIn("ليلى", str(trace))
        self.assertNotIn("S-1", str(trace))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
