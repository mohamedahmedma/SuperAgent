"""Answering the question that was asked, not the words that were typed.

Every test here traces back to one production failure, and it is worth stating in full
because the design is shaped around it:

    user: "what is the clothes for children under year 6"
    bot:  <answers>
    user: "and what is the fees for this years"
    bot:  "I found several possibly relevant directions. Which one are you asking about?
             - What subjects are taught in Years 3 to 6?
             - What is the day wear uniform for girls until Grade 6?
             - What subjects are taught for Years 3 to 6?"
    user: "no i mean what is the school fees for this years"
    bot:  <every fee for every year group, including years above 6>

Four separate defects produced that, and each has tests below:

  1. the follow-up was embedded as `previous + current`, so it scored nearest the
     PREVIOUS turn's subject and the offered directions were about uniforms
  2. two of the three options were paraphrases of one another, and having two or more
     options is itself what authorises interrupting the user
  3. a follow-up's subject was already settled, so offering a choice of subjects could
     not narrow anything and only cost the user a round trip
  4. the resume path never saw the conversation, and built its query by concatenating
     the correction onto the reading it was correcting — so it retrieved both
"""
import unittest
from unittest.mock import patch

from backend.chat.resolution import (
    CORRECTION,
    FOLLOWUP,
    NEW_TOPIC,
    STANDALONE,
    ResolvedQuestion,
    conversation_text,
    needs_resolution,
    resolve_question,
)
from backend.chat.signals import RequestSignals, SignalContext
from backend.chat.turn_policy import resolve_turn
from backend.profiles.registry import load_profile, set_profile
from backend.rag.evidence import Certainty, EvidenceReport
from backend.rag.policy import can_ask_human, decide_route, offerable_directions
from backend.rag.scope_index import ScopeMatch


def _config(**overrides):
    """The base profile's agent+rag sections, flattened the way detectors see them."""
    profile = load_profile("base")
    agent = profile.agent.model_copy(update=overrides)

    class _Flat:
        def __getattr__(self, name):
            if hasattr(agent, name):
                return getattr(agent, name)
            return getattr(profile.rag, name)

    return _Flat()


def _history(*pairs):
    """`("user", text)` / `("assistant", text)` as plain dicts, which both the
    resolver and the signal ladder accept alongside LangChain messages."""
    return [{"role": role, "content": text} for role, text in pairs]


UNIFORM_TURN = _history(
    ("user", "what is the clothes for children under year 6"),
    ("assistant", "Girls up to Grade 6 wear the navy day-wear set..."),
)


class TheGateTests(unittest.TestCase):
    """Resolution is a model call, so most turns must settle without one."""

    def test_a_first_message_has_nothing_to_inherit(self):
        wanted, reason = needs_resolution("what are the school fees", [], _config())
        self.assertFalse(wanted)
        self.assertIn("first message", reason)

    def test_a_message_that_points_back_is_resolved(self):
        wanted, reason = needs_resolution(
            "and what is the fees for this years", UNIFORM_TURN, _config()
        )
        self.assertTrue(wanted)
        self.assertTrue(reason)

    def test_an_arabic_follow_up_is_resolved(self):
        """The Arabic conjunction is written joined to the next word, so it can only be
        caught as a prefix — a token match would miss every "وما..." follow-up."""
        wanted, _ = needs_resolution("وما هي الرسوم؟", UNIFORM_TURN, _config())
        self.assertTrue(wanted)

    def test_a_long_self_contained_question_is_left_alone(self):
        wanted, reason = needs_resolution(
            "what documents does a new student need to submit before enrolment at the school",
            UNIFORM_TURN,
            _config(),
        )
        self.assertFalse(wanted)
        self.assertIn("carries its own subject", reason)

    def test_a_short_reply_is_resolved_even_with_no_marker(self):
        """"grade 5" answers a question and names no referent. It is exactly the case a
        marker list cannot catch, and exactly the case that needs resolving."""
        wanted, _ = needs_resolution("grade 5", UNIFORM_TURN, _config())
        self.assertTrue(wanted)

    def test_a_marker_does_not_fire_inside_a_longer_word(self):
        """"it" must not match "admission". Single-word markers are word-bounded."""
        wanted, _ = needs_resolution(
            "please describe the admission requirements for international applicants here",
            UNIFORM_TURN,
            _config(),
        )
        self.assertFalse(wanted)

    def test_switching_it_off_costs_nothing(self):
        wanted, reason = needs_resolution(
            "and what about those?", UNIFORM_TURN, _config(query_resolution_enabled=False)
        )
        self.assertFalse(wanted)
        self.assertIn("disabled", reason)


class ResolutionTests(unittest.TestCase):
    def _resolve(self, question, history, payload, **overrides):
        return resolve_question(
            question, history, _config(**overrides), invoke=lambda *a: payload
        )

    def test_a_follow_up_inherits_its_subject_and_its_conditions(self):
        resolved = self._resolve(
            "and what is the fees for this years",
            UNIFORM_TURN,
            {
                "question": "what are the school fees for children in the years up to Year 6",
                "constraints": ["grades up to Year 6"],
                "intent": "followup",
            },
        )
        self.assertTrue(resolved.resolved)
        self.assertIn("fees", resolved.question)
        self.assertEqual(["grades up to Year 6"], resolved.constraints)
        self.assertTrue(resolved.is_followup)
        self.assertFalse(resolved.supersedes_pending_question)

    def test_a_correction_replaces_the_pending_reading(self):
        resolved = self._resolve(
            "no i mean what is the school fees for this years",
            UNIFORM_TURN,
            {
                "question": "what are the school fees for the years up to Year 6",
                "constraints": ["grades up to Year 6"],
                "intent": "correction",
            },
        )
        self.assertEqual(CORRECTION, resolved.intent)
        self.assertTrue(resolved.supersedes_pending_question)
        self.assertTrue(resolved.is_followup)

    def test_a_new_topic_inherits_nothing(self):
        """Enforced in code, not asked for in the prompt: a condition surviving a
        subject change is this mechanism's own failure mode, pointed the other way."""
        resolved = self._resolve(
            "who is the principal",
            UNIFORM_TURN,
            {
                "question": "who is the principal",
                "constraints": ["grades up to Year 6"],
                "intent": "new_topic",
            },
        )
        self.assertEqual(NEW_TOPIC, resolved.intent)
        self.assertEqual([], resolved.constraints)
        self.assertTrue(resolved.supersedes_pending_question)

    def test_constraints_are_capped(self):
        resolved = self._resolve(
            "and those?",
            UNIFORM_TURN,
            {
                "question": "q",
                "constraints": [f"c{i}" for i in range(10)],
                "intent": "followup",
            },
            carried_constraint_limit=3,
        )
        self.assertEqual(3, len(resolved.constraints))

    def test_duplicate_constraints_collapse(self):
        resolved = self._resolve(
            "and those?",
            UNIFORM_TURN,
            {
                "question": "q",
                "constraints": ["Grades up to Year 6", "grades up to year 6 ", "girls"],
                "intent": "followup",
            },
        )
        self.assertEqual(2, len(resolved.constraints))

    def test_an_empty_question_is_an_abstention(self):
        """Substituting an empty question would search for nothing and deny the turn."""
        resolved = self._resolve(
            "and those?", UNIFORM_TURN, {"question": "  ", "intent": "followup"}
        )
        self.assertFalse(resolved.resolved)
        self.assertEqual("and those?", resolved.question)

    def test_a_resolver_failure_never_costs_the_turn(self):
        def boom(*_args):
            raise RuntimeError("provider down")

        resolved = resolve_question("and those?", UNIFORM_TURN, _config(), invoke=boom)
        self.assertFalse(resolved.resolved)
        self.assertEqual("and those?", resolved.question)
        self.assertEqual(STANDALONE, resolved.intent)
        self.assertEqual([], resolved.constraints)

    def test_an_unusable_verdict_falls_back_to_a_sane_intent(self):
        resolved = self._resolve(
            "and those?", UNIFORM_TURN, {"question": "the fees for Year 6", "intent": "banana"}
        )
        self.assertEqual(FOLLOWUP, resolved.intent)


class ConversationTextTests(unittest.TestCase):
    def test_both_sides_are_rendered(self):
        """A follow-up often points at something only the assistant said."""
        rendered = conversation_text(UNIFORM_TURN)
        self.assertIn("User: what is the clothes", rendered)
        self.assertIn("Assistant: Girls up to Grade 6", rendered)

    def test_a_long_turn_is_clipped(self):
        rendered = conversation_text(_history(("assistant", "x" * 5000)), max_chars=100)
        self.assertLess(len(rendered), 200)
        self.assertTrue(rendered.endswith("…"))

    def test_langchain_messages_work_too(self):
        from langchain_core.messages import AIMessage, HumanMessage

        rendered = conversation_text([HumanMessage(content="hi"), AIMessage(content="hello")])
        self.assertEqual("User: hi\nAssistant: hello", rendered)


class TextToScoreTests(unittest.TestCase):
    """What the scope rungs actually measure.

    This is defect 1. Concatenating the previous turn onto the current one produces a
    vector averaging two subjects, and the older turn — longer, more content-bearing —
    wins. That is why a fees question came back matching uniform questions.
    """

    def test_a_resolved_question_is_scored_on_its_own(self):
        ctx = SignalContext(
            question="and what is the fees for this years",
            history=UNIFORM_TURN,
            resolved_question="what are the school fees for the years up to Year 6",
        )
        self.assertEqual("what are the school fees for the years up to Year 6", ctx.text_to_score)
        self.assertNotIn("clothes", ctx.text_to_score)

    def test_without_a_resolution_the_old_concatenation_still_applies(self):
        """The fallback is deliberately unchanged: blunt, but better than scoring a
        bare follow-up alone, which measures nothing and looks like being off-topic."""
        ctx = SignalContext(
            question="and what is the fees for this years", history=UNIFORM_TURN
        )
        self.assertIn("clothes", ctx.text_to_score)
        self.assertIn("fees", ctx.text_to_score)

    def test_a_first_message_scores_itself(self):
        ctx = SignalContext(question="what are the fees", history=[])
        self.assertEqual("what are the fees", ctx.text_to_score)


class DirectionGatingTests(unittest.TestCase):
    """Defect 2: matching is recall, offering is a question put to a person."""

    def _match(self, question, score, vector):
        return ScopeMatch(question=question, score=score, chunk_id="c", vector=vector)

    def test_paraphrases_collapse_into_one_direction(self):
        from backend.rag.scope_detector import distinct_directions

        options = distinct_directions(
            [
                self._match("What subjects are taught in Years 3 to 6?", 0.61, [1.0, 0.0]),
                self._match("What subjects are taught for Years 3 to 6?", 0.60, [0.999, 0.045]),
            ],
            floor=0.5,
        )
        self.assertEqual(1, len(options), options)

    def test_genuinely_different_questions_both_survive(self):
        """The case the section-based rule got wrong: two different questions living in
        one chunk. Similarity between the questions says the right thing here."""
        from backend.rag.scope_detector import distinct_directions

        options = distinct_directions(
            [
                self._match("Which partner organisations are listed?", 0.586, [1.0, 0.0]),
                self._match("Who is the contact for partnership enquiries?", 0.585, [0.3, 0.95]),
            ],
            floor=0.5,
        )
        self.assertEqual(2, len(options))

    def test_a_clear_leader_is_not_an_ambiguity(self):
        from backend.rag.scope_detector import distinct_directions

        options = distinct_directions(
            [
                self._match("What are the school fees?", 0.72, [1.0, 0.0]),
                self._match("What is the uniform policy?", 0.55, [0.0, 1.0]),
            ],
            floor=0.5,
            max_score_gap=0.06,
        )
        self.assertEqual(["What are the school fees?"], options)

    def test_below_the_floor_is_never_offered(self):
        from backend.rag.scope_detector import distinct_directions

        options = distinct_directions(
            [
                self._match("A", 0.61, [1.0, 0.0]),
                self._match("B", 0.30, [0.0, 1.0]),
            ],
            floor=0.5,
        )
        self.assertEqual(["A"], options)

    def test_a_match_with_no_vector_is_never_treated_as_a_duplicate(self):
        """An index built before vectors were carried offers the list it always did,
        rather than silently collapsing options it cannot compare."""
        from backend.rag.scope_detector import distinct_directions

        options = distinct_directions(
            [self._match("A", 0.61, None), self._match("B", 0.60, None)], floor=0.5
        )
        self.assertEqual(["A", "B"], options)


class FollowUpRoutingTests(unittest.TestCase):
    """Defect 3: a subject settled one message ago is not a choice to put to the user."""

    def _report(self, **overrides):
        fields = {
            "question": "what are the fees",
            "certainty": Certainty.HIGH,
            "relevance": "weak",
            "sufficiency": "partial",
            "ambiguity": "multiple_candidates",
            "preferred_route": "clarify",
        }
        fields.update(overrides)
        return EvidenceReport(**fields)

    def _rag(self, **overrides):
        return load_profile("base").rag.model_copy(update=overrides)

    DIRECTIONS = [
        "What subjects are taught in Years 3 to 6?",
        "What is the day wear uniform for girls until Grade 6?",
    ]

    def test_a_fresh_question_still_gets_the_choice(self):
        route, reason = decide_route(
            self._report(),
            has_docs=True,
            rewrite_count=0,
            is_sub_agent=False,
            config=self._rag(),
            scope_options=self.DIRECTIONS,
            is_followup=False,
        )
        self.assertEqual("scope_select", route)
        self.assertTrue(reason)

    def test_a_follow_up_is_answered_instead_of_interrogated(self):
        """The exact turn from the transcript: a fees question asked after a uniform
        question must not come back offering a choice between uniform directions."""
        route, _ = decide_route(
            self._report(),
            has_docs=True,
            rewrite_count=0,
            is_sub_agent=False,
            config=self._rag(),
            scope_options=self.DIRECTIONS,
            is_followup=True,
        )
        self.assertNotEqual("scope_select", route)
        self.assertIn(route, ("rewrite", "answer"))

    def test_the_graders_own_candidates_still_authorise_an_ask(self):
        """Those come from a model that read the chunks and found several answers
        sitting in them. That is ambiguity in the evidence, not a guess about the
        query, and a follow-up does not make it go away."""
        allowed, reason = can_ask_human(
            self._report(hitl_options=["Day fees", "Boarding fees"]),
            hitl_rounds=0,
            config=self._rag(),
            scope_options=[],
            is_followup=True,
        )
        self.assertTrue(allowed, reason)

    def test_catalogued_directions_alone_cannot_interrupt_a_follow_up(self):
        allowed, reason = can_ask_human(
            self._report(),
            hitl_rounds=0,
            config=self._rag(),
            scope_options=self.DIRECTIONS,
            is_followup=True,
        )
        self.assertFalse(allowed)
        self.assertIn("inherits its subject", reason)

    def test_offerable_directions_returns_nothing_for_a_follow_up(self):
        self.assertEqual([], offerable_directions(self.DIRECTIONS, is_followup=True))
        self.assertEqual(self.DIRECTIONS, offerable_directions(self.DIRECTIONS))

    def test_one_direction_is_never_a_choice(self):
        self.assertEqual([], offerable_directions(["only one"]))


class ConstraintTests(unittest.TestCase):
    """Conditions never touch a query.

    They were appended to the retrieval query once. Measured over a twenty-turn
    conversation that denied three turns outright — a condition is by definition absent
    from every passage the corpus wrote once for everybody, so appending it is pure
    dilution exactly where recall matters most. See
    tests/test_conversation_sequence.py.
    """

    def test_the_search_query_is_the_question_alone(self):
        from backend.rag.pipeline import _search_query

        state = {
            "question": "what time does the school day start",
            "carried_constraints": ["the child is 5 years old"],
        }
        self.assertEqual(state["question"], _search_query(state))

    def test_nothing_appends_conditions_to_a_query_any_more(self):
        """A guard against reintroducing it. The condition still reaches the grader and
        the answer prompt — those can act on it without costing a document its rank."""
        import backend.chat.resolution as resolution

        self.assertFalse(hasattr(resolution, "apply_constraints"))


class TurnPlanTests(unittest.TestCase):
    def setUp(self):
        set_profile(load_profile("base"))
        self.addCleanup(set_profile, None)

    def _plan(self, **signal_overrides):
        profile = load_profile("base")
        signals = RequestSignals(question="and what about the fees", **signal_overrides)
        return resolve_turn(
            signals,
            agent_config=profile.agent,
            copy_config=profile.user_copy,
            rag_config=profile.rag,
        )

    def test_the_plan_carries_the_resolution_forward(self):
        plan = self._plan(
            resolved_question="what are the school fees for Year 6",
            carried_constraints=["grades up to Year 6"],
            followup_intent=FOLLOWUP,
        )
        self.assertEqual("what are the school fees for Year 6", plan.resolved_question)
        self.assertEqual(["grades up to Year 6"], plan.carried_constraints)
        self.assertTrue(plan.is_followup)

    def test_a_standalone_turn_carries_nothing(self):
        plan = self._plan()
        self.assertEqual("", plan.resolved_question)
        self.assertEqual([], plan.carried_constraints)
        self.assertFalse(plan.is_followup)

    def test_a_short_circuited_turn_still_records_what_it_thought_was_asked(self):
        """An out-of-domain refusal has to be arguable from the trace."""
        from backend.chat.signals import Scope

        plan = self._plan(
            scope=Scope.OUT_OF_DOMAIN,
            scope_certainty=Certainty.HIGH,
            resolved_question="who won the football last night",
            followup_intent=NEW_TOPIC,
        )
        self.assertTrue(plan.short_circuit)
        self.assertEqual("who won the football last night", plan.resolved_question)


class PlanTurnWiringTests(unittest.TestCase):
    """The resolver runs before the ladder, and its output reaches the graph."""

    def setUp(self):
        profile = load_profile("base")
        set_profile(
            profile.model_copy(
                update={
                    "agent": profile.agent.model_copy(update={"request_envelope_enabled": False}),
                    "rag": profile.rag.model_copy(
                        update={"scope_index_enabled": False, "domain_gate_enabled": False}
                    ),
                }
            )
        )
        self.addCleanup(set_profile, None)

    class Ctx:
        def __init__(self):
            self.retrieval_sections = []
            self.scope_options = []
            self.carried_constraints = []
            self.is_followup = False
            self.steps = []

        def emit_rag_step(self, icon, label, detail="", **kwargs):
            self.steps.append((icon, label, detail))

        def note_turn_plan(self, retrieval_sections, scope_options, *,
                           carried_constraints=(), is_followup=False, language=""):
            self.retrieval_sections = list(retrieval_sections or [])
            self.scope_options = list(scope_options or [])
            self.carried_constraints = list(carried_constraints or [])
            self.is_followup = bool(is_followup)
            self.language = language

    def test_the_resolution_reaches_the_rag_graph(self):
        from backend.chat.orchestrator import plan_turn

        ctx = self.Ctx()
        plan, signals = plan_turn(
            "and what is the fees for this years",
            UNIFORM_TURN,
            ctx,
            resolve_invoke=lambda *a: {
                "question": "what are the school fees for the years up to Year 6",
                "constraints": ["grades up to Year 6"],
                "intent": "followup",
            },
        )
        self.assertEqual(["grades up to Year 6"], ctx.carried_constraints)
        self.assertTrue(ctx.is_followup)
        self.assertEqual(
            "what are the school fees for the years up to Year 6", signals.resolved_question
        )

    def test_a_precomputed_resolution_is_not_recomputed(self):
        """The HITL path resolves before it can decide which branch to take. Resolving
        again in the planner would pay for the same call twice."""
        from backend.chat.orchestrator import plan_turn

        calls = []

        def spy(*args):
            calls.append(args)
            return {"question": "x", "intent": "followup"}

        plan, _ = plan_turn(
            "and those?",
            UNIFORM_TURN,
            self.Ctx(),
            resolve_invoke=spy,
            resolution=ResolvedQuestion(
                question="already resolved", intent=FOLLOWUP, resolved=True
            ),
        )
        self.assertEqual([], calls)
        self.assertEqual("already resolved", plan.resolved_question)

    def test_a_resolver_failure_leaves_the_turn_running(self):
        from backend.chat.orchestrator import plan_turn

        def boom(*_args):
            raise RuntimeError("down")

        plan, signals = plan_turn("and those?", UNIFORM_TURN, self.Ctx(), resolve_invoke=boom)
        self.assertFalse(plan.short_circuit)
        self.assertEqual("", plan.resolved_question)
        self.assertIsNone(plan.exposed_tools)


class ResumeQuestionTests(unittest.TestCase):
    """Defect 4: the resume path could not express replacement, only concatenation."""

    RESUME_STATE = {
        "question": "school fees",  # what the AGENT wrote for the tool
        "route": "scope_select",
        "retrieval_status": "needs_scope_selection",
        "rewrite_count": 0,
        "hitl_rounds": 0,
        "sub_questions": [],
        "carried_constraints": [],
    }

    def test_a_correction_replaces_rather_than_concatenates(self):
        from backend.rag.pipeline import _refined_question_for_hitl

        refined = _refined_question_for_hitl(
            self.RESUME_STATE,
            "no i mean what is the school fees for this years",
            ResolvedQuestion(
                question="what are the school fees for the years up to Year 6",
                constraints=["grades up to Year 6"],
                intent=CORRECTION,
                resolved=True,
            ),
            "what is the clothes for children under year 6",
        )
        self.assertEqual("what are the school fees for the years up to Year 6", refined)
        self.assertNotIn("no i mean", refined)

    def test_without_a_resolution_it_anchors_on_the_users_question(self):
        """`resume_state["question"]` holds the query the AGENT wrote, so a condition
        the user set and the agent dropped was already gone before this ran."""
        from backend.rag.pipeline import _refined_question_for_hitl

        refined = _refined_question_for_hitl(
            self.RESUME_STATE,
            "Primary",
            None,
            "what is the clothes for children under year 6",
        )
        self.assertIn("children under year 6", refined)
        self.assertIn("Primary", refined)

    def test_an_abstaining_resolver_falls_back_rather_than_blanking(self):
        from backend.rag.pipeline import _refined_question_for_hitl

        refined = _refined_question_for_hitl(
            self.RESUME_STATE,
            "Primary",
            ResolvedQuestion(question="Primary", intent=STANDALONE, resolved=False),
            "the original question",
        )
        self.assertIn("the original question", refined)


class ResumeStateTests(unittest.TestCase):
    def test_conditions_survive_the_resume_boundary(self):
        """The graph starts fresh on a resume, and the turn that established
        "up to Year 6" is several messages back by the time the user answers."""
        from backend.rag.pipeline import _state_from_resume

        class Ctx:
            retrieval_sections = []
            scope_options = []
            carried_constraints = []
            is_followup = False

        state = _state_from_resume(
            {
                "question": "school fees",
                "route": "scope_select",
                "retrieval_status": "needs_scope_selection",
                "rewrite_count": 0,
                "hitl_rounds": 0,
                "sub_questions": [],
                "carried_constraints": ["grades up to Year 6"],
            },
            "no i mean the fees",
            Ctx(),
            resolved=ResolvedQuestion(
                question="what are the school fees up to Year 6",
                constraints=["girls only"],
                intent=CORRECTION,
                resolved=True,
            ),
            original_question="what is the clothes for children under year 6",
        )
        # Union, not replacement: a reply that narrows further does not withdraw a
        # condition set two turns ago.
        self.assertEqual(["grades up to Year 6", "girls only"], state["carried_constraints"])
        self.assertTrue(state["is_followup"], "a resumed turn is a continuation by construction")
        self.assertEqual(1, state["hitl_rounds"])


class TurnEntryTests(unittest.TestCase):
    """Whether a reply answers the pending clarification or replaces it."""

    def setUp(self):
        set_profile(load_profile("base"))
        self.addCleanup(set_profile, None)

    PENDING = {
        "id": "h1",
        "original_question": "what is the clothes for children under year 6",
        "prompt": "Which one are you asking about?",
        "options": ["Subjects in Years 3 to 6", "Day wear uniform until Grade 6"],
        "route": "scope_select",
        "retrieval_status": "needs_scope_selection",
        "answers": [],
        "created_at": "2026-08-07T00:00:00+00:00",
        "resume_state": {
            "question": "school fees",
            "route": "scope_select",
            "retrieval_status": "needs_scope_selection",
            "rewrite_count": 0,
            "hitl_rounds": 0,
            "sub_questions": [],
            "carried_constraints": [],
        },
    }

    def _enter(self, user_text, resolution):
        import backend.chat.service as service

        with patch.object(service, "resolve_turn_question", lambda *a, **k: resolution):
            return service._enter_turn(
                user_text, list(UNIFORM_TURN), {service.PENDING_HITL_KEY: self.PENDING}
            )

    def test_a_correction_abandons_the_clarification(self):
        entry = self._enter(
            "no i mean what is the school fees for this years",
            ResolvedQuestion(
                question="what are the school fees for the years up to Year 6",
                intent=CORRECTION,
                resolved=True,
            ),
        )
        self.assertTrue(entry.superseded)
        self.assertFalse(entry.is_hitl_resume)
        self.assertEqual(
            "what are the school fees for the years up to Year 6", entry.original_question
        )

    def test_a_selection_resumes_it(self):
        entry = self._enter(
            "the uniform one",
            ResolvedQuestion(
                question="what is the day wear uniform until Grade 6",
                intent=FOLLOWUP,
                resolved=True,
            ),
        )
        self.assertFalse(entry.superseded)
        self.assertTrue(entry.is_hitl_resume)
        self.assertEqual(self.PENDING["original_question"], entry.original_question)

    def test_a_new_topic_abandons_it_too(self):
        entry = self._enter(
            "who is the principal",
            ResolvedQuestion(question="who is the principal", intent=NEW_TOPIC, resolved=True),
        )
        self.assertTrue(entry.superseded)
        self.assertFalse(entry.is_hitl_resume)

    def test_no_pending_clarification_means_no_resolver_call_here(self):
        import backend.chat.service as service

        calls = []
        with patch.object(
            service, "resolve_turn_question", lambda *a, **k: calls.append(a) or None
        ):
            entry = service._enter_turn("what are the fees", [], {})
        self.assertEqual([], calls, "the planner resolves an ordinary turn, not this")
        self.assertFalse(entry.is_hitl_resume)
        self.assertIsNone(entry.resolution)


class TurnContextMessageTests(unittest.TestCase):
    """The per-turn fragment, which must cost nothing on a turn that carries nothing."""

    def setUp(self):
        set_profile(load_profile("base"))
        self.addCleanup(set_profile, None)

    def test_nothing_to_say_produces_no_message(self):
        import backend.chat.service as service
        from backend.chat.turn_policy import TurnPlan

        self.assertIsNone(service._turn_context_message(TurnPlan()))
        self.assertIsNone(service._turn_context_message(None))

    def test_the_resolved_question_and_conditions_are_both_stated(self):
        import backend.chat.service as service
        from backend.chat.turn_policy import TurnPlan

        message = service._turn_context_message(
            TurnPlan(
                resolved_question="what are the school fees up to Year 6",
                carried_constraints=["grades up to Year 6"],
            )
        )
        self.assertIsNotNone(message)
        self.assertIn("what are the school fees up to Year 6", message.content)
        self.assertIn("grades up to Year 6", message.content)
        self.assertIn("bind the answer", message.content)

    def test_it_sits_between_the_history_and_the_message(self):
        import backend.chat.service as service
        from backend.chat.turn_policy import TurnPlan
        from langchain_core.messages import HumanMessage

        built = service._build_context_messages(
            [HumanMessage(content="earlier")],
            "",
            "and the fees?",
            TurnPlan(resolved_question="what are the school fees up to Year 6"),
        )
        self.assertEqual("earlier", built[0].content)
        self.assertIn("what are the school fees up to Year 6", built[1].content)
        self.assertEqual("and the fees?", built[2].content)


if __name__ == "__main__":
    unittest.main()
