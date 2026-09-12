# -*- coding: utf-8 -*-
"""The child's name reaches the answer and not the search box.

Two halves, and the second is the one that will actually break.

The first is the decision itself — `backend/chat/child_names.py` — which is pure and can
be asserted case by case. Most of it is about ONE hazard: `name_key` folds ى onto ي, so
the preposition على and the given name علي are the same string by the time anything
compares them, and this deployment's roster really does carry a child called علي. Cutting
every folded match would take the preposition out of «الخصم على الصف الثالث»; cutting
none leaves a rare, high-IDF term in a query against a corpus that holds no pupil's name.
So the rules below are about how much evidence each cut needs.

The second half is the wiring, which is where a feature like this dies: a plan field
nothing reads, a hint an older context rejects, a strip applied to the text the model
answers from instead of the text retrieval searches for. The last of those would be a
regression in the other direction — the parent asked about their child by name and the
answer stops saying which child it is about.
"""
import unittest

from backend.chat.child_names import name_surfaces, strip_child_names
from backend.chat.child_resolution import ResolvedChild, no_child, resolve_child
from backend.chat.child_roster import ChildOption
from backend.chat.request_context import ChatRequestContext
from backend.chat.service import _turn_context_message
from backend.chat.signals import RequestSignals
from backend.chat.turn_policy import resolve_turn
from backend.rag.pipeline import _search_query

#: The homograph, as this deployment actually carries it.
ALI = ChildOption(student_id="S-1", label="علي حسن", gender="male", year_level="Year 4")
#: A name no Arabic sentence contains by accident, and one the SIS spells with a maksura
#: while a parent types a yeh.
LAYLA = ChildOption(
    student_id="S-2", label="ليلى أحمد", label_en="Layla Ahmed", gender="female"
)


def _surfaces(child, name, reference="named"):
    return name_surfaces(reference=reference, child_name=name, label=child.label)


def _cut(text, child, name, reference="named"):
    return strip_child_names(text, _surfaces(child, name, reference))


class ANameTheMessageActuallyUsed(unittest.TestCase):
    """The plain cases: a name the corpus cannot hold, taken out of the search."""

    def test_a_named_child_leaves_the_query(self):
        text, cuts = _cut("ايه مصاريف علي؟", ALI, "علي")
        self.assertEqual("ايه مصاريف؟", text)
        self.assertEqual(1, cuts)

    def test_the_full_name_goes_as_one_name(self):
        text, _ = _cut("مصاريف علي حسن كام؟", ALI, "علي حسن")
        self.assertEqual("مصاريف كام؟", text)

    def test_a_spelling_the_roster_does_not_share_still_goes(self):
        """The parent types a yeh, the registrar typed a maksura. That is the whole
        reason `name_key` folds, and an unambiguous name is safe to fold on."""
        text, _ = _cut("ايه مواعيد باص ليلي؟", LAYLA, "ليلي")
        self.assertEqual("ايه مواعيد باص؟", text)

    def test_an_unambiguous_name_goes_everywhere_it_appears(self):
        text, cuts = _cut("مصاريف ليلى وباص ليلى كام؟", LAYLA, "ليلى")
        self.assertNotIn("ليلى", text)
        self.assertEqual(2, cuts)

    def test_an_english_name_goes_too(self):
        text, _ = _cut("what is the bus fee for Layla?", LAYLA, "Layla")
        self.assertEqual("what is the bus fee for?", text)

    def test_the_rest_of_the_question_is_left_exactly_as_written(self):
        """Cut from the original string, never from a folded copy of it: what comes back
        is read by the embedder and by the rewrite model, and `backend/text_matching.py`
        says what folded Arabic does to both."""
        text, _ = _cut("إمتى امتحانات ليلى النصف الأول؟", LAYLA, "ليلى")
        self.assertEqual("إمتى امتحانات النصف الأول؟", text)


class TheNameThatIsAlsoAWord(unittest.TestCase):
    """علي and على are one string after folding, and only one of them is a child.

    Every case here is the same question asked at a different evidence level. The
    asymmetry they encode: leaving a name in costs precision, cutting a preposition out
    changes what was asked and nothing downstream can see that it happened.
    """

    def test_a_turn_that_names_nobody_cuts_nothing(self):
        """The commonest shape by far, and the one that must never regress: a general
        question whose child was settled by a pin or by being an only child. The message
        contains no name, so there is nothing to look for."""
        for reference in ("context", "son", "child", "plural", "none"):
            with self.subTest(reference=reference):
                text, cuts = _cut(
                    "هل فيه خصم على الصف الثالث؟", ALI, "", reference=reference
                )
                self.assertEqual("هل فيه خصم على الصف الثالث؟", text)
                self.assertEqual(0, cuts)

    def test_the_name_goes_and_the_preposition_stays(self):
        """Both tokens fold to the same key. The parent spelled one of them the way the
        roster spells the child, and that is the discriminator."""
        text, cuts = _cut("مصاريف علي وهل فيه خصم على الاخ التاني؟", ALI, "علي")
        self.assertEqual("مصاريف وهل فيه خصم على الاخ التاني؟", text)
        self.assertEqual(1, cuts)

    def test_a_relationship_word_is_enough_to_cut_the_other_spelling(self):
        """A parent writing «على» for their son is ordinary. «ابني» in front of it is
        what makes the reading unambiguous."""
        text, _ = _cut("ابني على عنده امتحانات امتى؟", ALI, "علي")
        self.assertEqual("ابني عنده امتحانات امتى؟", text)

    def test_a_folded_only_match_with_nothing_behind_it_is_left_alone(self):
        """The abstention this module exists for. The classifier said the message names
        the child — it can say that on a folded match alone — and there is still nothing
        here that distinguishes the preposition from the name."""
        text, cuts = _cut("الخصم على الاخ التاني كام؟", ALI, "علي")
        self.assertEqual("الخصم على الاخ التاني كام؟", text)
        self.assertEqual(0, cuts)

    def test_at_most_one_occurrence_of_a_homograph_is_ever_cut(self):
        text, cuts = _cut("علي عنده خصم على المصاريف ولا على الباص؟", ALI, "علي")
        self.assertEqual(1, cuts)
        self.assertIn("على المصاريف", text)
        self.assertIn("على الباص", text)

    def test_a_name_that_folds_onto_a_question_word(self):
        """«آية» folds onto «إيه» — the Egyptian "what", which opens half the messages
        this deployment receives. Exact spelling is what tells them apart."""
        aya = ChildOption(student_id="S-3", label="آية محمود", gender="female")
        text, cuts = strip_child_names(
            "ايه مصاريف آية؟", name_surfaces(reference="named", child_name="آية", label=aya.label)
        )
        self.assertEqual("ايه مصاريف؟", text)
        self.assertEqual(1, cuts)

    def test_a_name_that_is_also_the_word_for_an_age(self):
        """«عمر» is a name and it is what a fee table's eligibility rule is written in."""
        omar = ChildOption(student_id="S-4", label="عمر خالد", gender="male")
        surfaces = name_surfaces(reference="named", child_name="عمر", label=omar.label)
        text, cuts = strip_child_names("ما هو العمر المطلوب للتقديم؟", surfaces)
        self.assertEqual("ما هو العمر المطلوب للتقديم؟", text)
        self.assertEqual(0, cuts)

    def test_a_patronymic_is_never_cut_on_its_own(self):
        """«حسن» is the father's name, every sibling carries it, and on its own it is an
        ordinary adjective. It goes as part of «علي حسن» and never by itself."""
        text, cuts = _cut("هل مستوى ابني حسن كويس؟", ALI, "علي")
        self.assertEqual("هل مستوى ابني حسن كويس؟", text)
        self.assertEqual(0, cuts)

    def test_a_question_that_is_only_a_name_is_left_whole(self):
        """An empty query is a wasted retrieval, which is the same reason
        `turn_policy._resolve_argument` drops an empty planned argument."""
        text, cuts = _cut("علي؟", ALI, "علي")
        self.assertEqual("علي؟", text)
        self.assertEqual(0, cuts)


class _Agent:
    tools = ["search_knowledge_base", "get_student_grades"]
    social_phrases = []
    social_reply_mode = "model"
    year_reference_markers = ()


class _Copy:
    social = None
    out_of_domain = None
    which_child = "Which child do you mean?"


def _plan(child=None, **signal_kwargs):
    signals = RequestSignals(question=signal_kwargs.pop("question", "q"), **signal_kwargs)
    return resolve_turn(signals, agent_config=_Agent(), copy_config=_Copy(), child=child)


class ThePlanCarriesItAndTheTraceDoesNot(unittest.TestCase):
    def test_a_named_child_puts_its_spellings_on_the_plan(self):
        plan = _plan(
            resolve_child(reference="named", child_name="علي", roster=[ALI]),
            question="مصاريف علي كام؟",
            about_child=True,
            child_reference="named",
            child_name="علي",
        )
        self.assertIn("علي", plan.child_names)
        self.assertIn("علي حسن", plan.child_names)

    def test_a_child_settled_without_being_named_puts_nothing_there(self):
        plan = _plan(
            resolve_child(reference="context", roster=[ALI]),
            question="هل فيه خصم على الصف الثالث؟",
            about_child=True,
            child_reference="context",
        )
        self.assertTrue(plan.child_hint)
        self.assertEqual([], plan.child_names)

    def test_a_turn_about_nobody_puts_nothing_there(self):
        self.assertEqual([], _plan(no_child("not about a child")).child_names)

    def test_the_trace_still_names_no_child(self):
        """`test_child_selection_identity.TheTraceNamesNoChild` states the rule: this
        trace is persisted per message and streamed to a browser, so it reports that a
        decision was made and never what it was. A list of a real child's name spellings
        is the same disclosure as the name."""
        plan = _plan(
            resolve_child(reference="named", child_name="ليلى", roster=[LAYLA]),
            question="مصاريف ليلى كام؟",
            about_child=True,
            child_reference="named",
            child_name="ليلى",
        )
        rendered = repr(plan.as_trace())
        for secret in ("ليلى", "أحمد", "Layla", "S-2"):
            self.assertNotIn(secret, rendered)


class _Ctx:
    """The one attribute the graph reads. A stand-in for every caller that has none."""

    def __init__(self, names=()):
        self.child_names = list(names)


class TheGraphSearchesWithoutItAndAnswersWithIt(unittest.TestCase):
    def test_the_search_text_loses_the_name(self):
        state = {"question": "مصاريف علي كام؟", "request_context": _Ctx(["علي حسن", "علي"])}
        self.assertEqual("مصاريف كام؟", _search_query(state))

    def test_the_question_itself_is_untouched(self):
        """The distinction the whole feature rests on. `state["question"]` is what the
        grader, the HITL prompts and `service._resume_answer` read, and the last of those
        hands it to a model that has to write the parent a sentence about their child."""
        state = {"question": "مصاريف علي كام؟", "request_context": _Ctx(["علي حسن", "علي"])}
        _search_query(state)
        self.assertEqual("مصاريف علي كام؟", state["question"])

    def test_a_state_with_no_context_behind_it_searches_as_it_always_did(self):
        for ctx in (None, _Ctx(), object()):
            with self.subTest(ctx=type(ctx).__name__):
                state = {"question": "مصاريف علي كام؟", "request_context": ctx}
                self.assertEqual("مصاريف علي كام؟", _search_query(state))

    def test_the_context_takes_the_hint(self):
        ctx = ChatRequestContext.for_sync(user_id="u-1", session_id="s-1")
        ctx.note_turn_plan([], [], child_names=["علي حسن", "علي"])
        self.assertEqual(["علي حسن", "علي"], ctx.child_names)

    def test_a_context_that_never_heard_of_the_hint_still_gets_the_others(self):
        """`orchestrator._hand_to_graph` drops unknown hints newest-first. A context
        written before this feature must still receive the sections and the language."""
        from backend.chat.orchestrator import _hand_to_graph

        class _OldContext:
            def __init__(self):
                self.seen = None

            def note_turn_plan(self, sections, options, **hints):
                if "child_names" in hints:
                    raise TypeError("unexpected keyword argument 'child_names'")
                self.seen = hints

        plan = _plan(
            resolve_child(reference="named", child_name="علي", roster=[ALI]),
            question="مصاريف علي كام؟",
            about_child=True,
            child_reference="named",
            child_name="علي",
        )
        ctx = _OldContext()
        _hand_to_graph(ctx, plan)
        self.assertIsNotNone(ctx.seen)
        self.assertIn("carried_constraints", ctx.seen)
        self.assertNotIn("child_names", ctx.seen)

    def test_the_model_is_still_told_which_child(self):
        """The regression in the other direction. A parent who asked about their child by
        name must not get an answer that has forgotten which child it is about."""
        plan = _plan(
            resolve_child(reference="named", child_name="علي", roster=[ALI]),
            question="مصاريف علي كام؟",
            about_child=True,
            child_reference="named",
            child_name="علي",
        )
        message = _turn_context_message(plan)
        self.assertIsNotNone(message)
        self.assertIn("علي حسن", str(message.content))


if __name__ == "__main__":
    unittest.main()
