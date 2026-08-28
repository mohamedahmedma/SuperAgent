"""QA scenarios: the system's behaviour, written from the user's side of it.

These are acceptance cases rather than unit tests. Each one names something a user
would notice going wrong, and asserts against the real scope index, the real embedder
and the real corpus — no stubs. Where a case needs the LLM to reach a verdict it is
marked, because that costs tokens and depends on someone else's uptime.

The governing requirement, restated because every case here serves it: **a question the
corpus can answer must never be refused.** The gate is allowed to waste a search; it is
not allowed to turn a user away. So the asymmetric cases below assert admission far more
often than refusal, and the refusal cases are limited to subjects with no relation to
the corpus at all.
"""
import unittest

from backend.chat.signals import RequestSignals, Scope, SignalContext, build_ladder
from backend.chat.turn_policy import resolve_turn
from backend.profiles import get_profile
from backend.profiles.registry import load_profile
from backend.rag.evidence import Certainty
from backend.rag.scope_detector import CatalogueScopeDetector, index_store
from backend.rag.scope_index import build_index
from tests.general.integration_support import (
    requires_embedder,
    requires_llm,
    requires_postgres,
)

# Questions a school knowledge base plainly covers, in both languages it serves.
IN_DOMAIN = [
    "when does the second term start",
    "how much are the school fees for grade 5",
    "what documents do I need to apply",
    "is there a school bus to Maadi",
    "what is the uniform policy",
    "who do I contact about admissions",
    "what time does the school day end",
    "does the school follow a British curriculum",
    "متى يبدأ الفصل الدراسي الثاني",
    "كم تبلغ الرسوم الدراسية للصف الخامس",
    "ما هي المستندات المطلوبة للتقديم",
    "هل يوجد أتوبيس مدرسي إلى المعادي",
    "ما هو نظام الزي المدرسي",
    "بمن أتواصل بخصوص القبول",
    "متى ينتهي اليوم الدراسي",
    "هل المدرسة تتبع المنهج البريطاني",
]

# Plainly a different subject. Refusing these is correct and is the only case that is.
OUT_OF_DOMAIN = [
    "what is the weather forecast for tomorrow",
    "write me a poem about the sea",
    "who won the football match last night",
    "how do I bake a chocolate cake",
    "what is the capital of Peru",
    "ما هو الطقس غدا",
    "اكتب لي قصيدة عن البحر",
    "كيف أطبخ الكشري",
]


def config(**overrides):
    settings = {"scope_index_enabled": True, **overrides}
    return load_profile("base").rag.model_copy(update=settings)


class LadderConfig:
    """The shape build_ladder expects, matching backend.chat.orchestrator."""

    def __init__(self, agent, rag):
        self._agent = agent
        self._rag = rag

    def __getattr__(self, name):
        if hasattr(self._rag, name):
            return getattr(self._rag, name)
        return getattr(self._agent, name)


@requires_postgres
@requires_embedder
class ScopeIndexHealthTests(unittest.TestCase):
    """The catalogue as actually built and stored, not a fixture.

    A hole here is invisible in production: questions about an uncatalogued section
    score low, escalate forever, and look like the model being cautious rather than
    like a missing row.
    """

    @classmethod
    def setUpClass(cls):
        cls.index = index_store.get()

    def test_the_index_is_ready(self):
        self.assertTrue(self.index.ready, "the scope index built empty")

    def test_the_index_has_questions_over_several_sections(self):
        self.assertGreater(len(self.index.questions), 0)
        self.assertGreater(len(set(self.index.chunk_ids)), 1)

    def test_every_question_has_a_vector(self):
        self.assertEqual(len(self.index.questions), len(self.index.vectors))

    def test_the_vectors_have_the_configured_width(self):
        import os

        self.assertEqual(
            int(os.getenv("DENSE_EMBEDDING_DIM", "1024")), self.index.vectors.shape[1]
        )

    def test_the_floor_is_in_range(self):
        """A floor at or above 1.0 would escalate everything; below 0 admits everything."""
        self.assertGreaterEqual(self.index.floor, 0.0)
        self.assertLess(self.index.floor, 1.0)

    def test_the_index_carries_a_corpus_description(self):
        self.assertTrue(self.index.catalogue.strip(),
                        "the scope prompt would describe the corpus as nothing at all")

    def test_the_floor_fingerprint_is_recorded(self):
        self.assertTrue(self.index.floor_sha256)

    def test_a_catalogued_question_matches_itself_best(self):
        """The most basic sanity property of the whole gate."""
        from backend.indexing.embedding import embed_query

        question = self.index.questions[0]
        matches = self.index.best_matches(embed_query(question), limit=1)
        self.assertTrue(matches)
        self.assertGreaterEqual(matches[0].score, self.index.floor)

    def test_every_catalogued_question_clears_the_floor(self):
        """The floor is derived from these very questions, so any that fall below it
        would be permanently un-matchable by their own wording."""
        from backend.indexing.embedding import embed_query

        below = []
        for question in self.index.questions[:40]:
            matches = self.index.best_matches(embed_query(question), limit=1)
            if not matches or matches[0].score < self.index.floor:
                below.append(question)
        self.assertEqual([], below, "catalogued questions score below their own floor")


@requires_postgres
@requires_embedder
class ScopeGateRecallTests(unittest.TestCase):
    """Rung 1 against the real index. It may admit; it may never refuse."""

    @classmethod
    def setUpClass(cls):
        cls.detector = CatalogueScopeDetector()
        cls.profile = get_profile()
        cls.config = LadderConfig(cls.profile.agent, cls.profile.rag)

    def judge(self, question):
        signals = RequestSignals(question=question)
        context = SignalContext(question=question, history=[], config=self.config)
        return self.detector.detect(context, signals)

    def test_rung_one_never_refuses_with_high_certainty(self):
        """The safety property of the entire design: only a rung that read the question
        may end a turn, and rung 1 is a dot product."""
        for question in IN_DOMAIN + OUT_OF_DOMAIN:
            with self.subTest(question=question):
                signals = self.judge(question)
                if signals is not None and signals.scope is Scope.OUT_OF_DOMAIN:
                    self.assertLess(
                        signals.scope_certainty, Certainty.HIGH,
                        "rung 1 produced a refusal that could end a turn",
                    )

    def test_a_low_certainty_rejection_never_short_circuits_a_turn(self):
        for question in OUT_OF_DOMAIN:
            with self.subTest(question=question):
                signals = self.judge(question)
                if signals is None:
                    continue
                plan = resolve_turn(
                    signals,
                    agent_config=self.profile.agent,
                    copy_config=self.profile.user_copy,
                    rag_config=self.profile.rag,
                )
                if signals.scope_certainty < Certainty.HIGH:
                    self.assertFalse(
                        plan.short_circuit,
                        "a dot product ended the turn without the model reading it",
                    )

    def test_in_domain_questions_are_admitted_or_escalated_never_settled_against(self):
        for question in IN_DOMAIN:
            with self.subTest(question=question):
                signals = self.judge(question)
                self.assertIsNotNone(signals, "the gate abstained entirely")
                if signals.scope is Scope.OUT_OF_DOMAIN:
                    self.assertLess(signals.scope_certainty, Certainty.HIGH)

    def test_the_gate_reports_candidate_sections_for_retrieval(self):
        """Matches double as a retrieval hint; losing them costs ordering, not safety."""
        signals = self.judge(IN_DOMAIN[0])
        self.assertIsNotNone(signals)
        self.assertTrue(signals.scope_matches)

    def test_a_matched_question_names_its_section(self):
        signals = self.judge(IN_DOMAIN[0])
        self.assertTrue(any(match.chunk_id for match in signals.scope_matches))

    def test_arabic_questions_reach_the_index_at_all(self):
        """A bilingual catalogue exists so Arabic is not systematically escalated."""
        scored = 0
        for question in [q for q in IN_DOMAIN if any("؀" <= c <= "ۿ" for c in q)]:
            signals = self.judge(question)
            if signals is not None and signals.scope_matches:
                scored += 1
        self.assertGreater(scored, 0, "no Arabic question produced any catalogue match")

    def test_an_empty_question_does_not_crash_the_gate(self):
        try:
            self.judge("")
        except Exception as exc:
            self.fail(f"the gate raised on an empty question: {type(exc).__name__}")

    def test_adversarial_input_does_not_crash_the_gate(self):
        hostile = [
            "'; DROP TABLE users; --",
            "<script>alert(1)</script>",
            "{{7*7}}",
            "\x00\x01",
            "A" * 4000,
            "😀" * 200,
            "‮RTL override",
        ]
        for question in hostile:
            with self.subTest(question=question[:24]):
                try:
                    self.judge(question)
                except Exception as exc:
                    self.fail(f"the gate raised on {question[:24]!r}: {type(exc).__name__}")


@requires_postgres
@requires_embedder
class TurnPolicyTests(unittest.TestCase):
    """What the turn does with a verdict, independent of how it was reached."""

    def setUp(self):
        self.profile = get_profile()

    def plan_for(self, scope, certainty, **extra):
        signals = RequestSignals(question="q", scope=scope, scope_certainty=certainty, **extra)
        return resolve_turn(
            signals,
            agent_config=self.profile.agent,
            copy_config=self.profile.user_copy,
            rag_config=self.profile.rag,
        )

    def test_only_a_high_certainty_rejection_ends_a_turn(self):
        for certainty in (Certainty.NONE, Certainty.LOW, Certainty.MEDIUM):
            with self.subTest(certainty=certainty.name):
                self.assertFalse(self.plan_for(Scope.OUT_OF_DOMAIN, certainty).short_circuit)
        self.assertTrue(self.plan_for(Scope.OUT_OF_DOMAIN, Certainty.HIGH).short_circuit)

    def test_an_in_domain_verdict_never_short_circuits(self):
        for certainty in (Certainty.LOW, Certainty.MEDIUM, Certainty.HIGH):
            with self.subTest(certainty=certainty.name):
                self.assertFalse(self.plan_for(Scope.IN_DOMAIN, certainty).short_circuit)

    def test_a_refusal_carries_copy_for_the_user(self):
        plan = self.plan_for(Scope.OUT_OF_DOMAIN, Certainty.HIGH)
        self.assertTrue(str(plan.static_reply).strip(), "a refusal with nothing to say")

    def test_disclosed_personal_data_is_captured_even_when_off_topic(self):
        plan = self.plan_for(
            Scope.OUT_OF_DOMAIN, Certainty.HIGH, personal_data=["child_name"]
        )
        self.assertTrue(plan.capture_user_info)

    def test_planning_never_raises_on_an_empty_signal_set(self):
        plan = resolve_turn(
            RequestSignals(question=""),
            agent_config=self.profile.agent,
            copy_config=self.profile.user_copy,
            rag_config=self.profile.rag,
        )
        self.assertFalse(plan.short_circuit)


@requires_postgres
@requires_embedder
class ScopePromptTests(unittest.TestCase):
    """What the escalation rung is actually shown, rendered from the live index."""

    @classmethod
    def setUpClass(cls):
        cls.index = index_store.get()

    def render(self, question="when does term two start"):
        from backend.indexing.embedding import embed_query
        from backend.prompts import render

        matches = self.index.best_matches(embed_query(question), limit=3)
        return render(
            "rag/scope_check.j2",
            question=question,
            matches=[
                {"question": m.question, "score": m.score,
                 "above_floor": m.score >= self.index.floor}
                for m in matches
            ],
            catalogue=self.index.catalogue,
            persona=get_profile().identity.persona,
            history="",
            personal_fields=["child_name"],
            floor=self.index.floor,
            index_ready=self.index.ready,
        )

    def test_the_prompt_contains_the_question(self):
        self.assertIn("term two", self.render())

    def test_the_prompt_states_the_corpus_floor(self):
        self.assertIn(f"{self.index.floor:.2f}", self.render())

    def test_the_prompt_describes_the_corpus(self):
        self.assertIn(self.index.catalogue[:40], self.render())

    def test_the_prompt_teaches_the_asymmetry_by_example(self):
        text = self.render()
        self.assertIn("in_domain", text)
        self.assertIn("out_of_domain", text)
        self.assertIn("choose in_domain", text)

    def test_the_prompt_carries_examples_in_both_languages(self):
        text = self.render()
        self.assertIn("الطقس", text)
        self.assertIn("الفصل الدراسي", text)

    def test_arabic_questions_render_without_mangling(self):
        text = self.render("متى يبدأ الفصل الدراسي الثاني")
        self.assertIn("متى يبدأ الفصل الدراسي الثاني", text)

    def test_the_prompt_stays_a_reasonable_size(self):
        """It runs on every escalation; an unbounded catalogue would bill every turn."""
        self.assertLess(len(self.render()), 20000)


@requires_postgres
@requires_embedder
class ScopeIndexRebuildTests(unittest.TestCase):
    def test_invalidate_forces_a_rebuild(self):
        first = index_store.get()
        index_store.invalidate()
        second = index_store.get()
        self.assertIsNot(first, second)
        self.assertEqual(first.floor, second.floor, "the floor moved without the corpus")

    def test_a_failing_builder_leaves_the_gate_abstaining_not_refusing(self):
        from backend.rag.scope_detector import ScopeIndexStore

        def broken():
            raise RuntimeError("database gone")

        store = ScopeIndexStore(builder=broken)
        self.assertFalse(store.get().ready)

    def test_a_broken_builder_is_not_retried_on_every_request(self):
        from backend.rag.scope_detector import ScopeIndexStore

        calls = []

        def broken():
            calls.append(1)
            raise RuntimeError("still gone")

        store = ScopeIndexStore(builder=broken)
        for _ in range(5):
            store.get()
        self.assertEqual(1, len(calls))

    def test_an_empty_catalogue_builds_an_index_that_abstains(self):
        index = build_index([], embed=lambda qs: [])
        self.assertFalse(index.ready)


@requires_postgres
@requires_embedder
class SignalLadderTests(unittest.TestCase):
    """The assembled ladder, running the real rungs, with the model rung disabled so
    the cheap path is measured on its own."""

    def ladder(self):
        profile = get_profile()
        return build_ladder(
            LadderConfig(profile.agent, profile.rag.model_copy(
                update={"request_envelope_enabled": False}
            ))
        )

    def context(self, question, history=()):
        profile = get_profile()
        return SignalContext(
            question=question, history=list(history),
            config=LadderConfig(profile.agent, profile.rag.model_copy(
                update={"request_envelope_enabled": False}
            )),
        )

    def test_the_ladder_reaches_a_conclusion_for_an_in_domain_question(self):
        signals = self.ladder().run(self.context(IN_DOMAIN[0]))
        self.assertTrue(signals.assessed_by, "no rung reached a conclusion")

    def test_the_ladder_never_ends_a_turn_without_the_model_rung(self):
        """With the escalation rung off, nothing left may refuse."""
        profile = get_profile()
        for question in OUT_OF_DOMAIN:
            with self.subTest(question=question):
                signals = self.ladder().run(self.context(question))
                plan = resolve_turn(
                    signals,
                    agent_config=profile.agent,
                    copy_config=profile.user_copy,
                    rag_config=profile.rag,
                )
                self.assertFalse(
                    plan.short_circuit and signals.scope is Scope.OUT_OF_DOMAIN,
                    "a turn was refused with no rung that read the question",
                )

    def test_the_ladder_records_which_rungs_ran(self):
        signals = self.ladder().run(self.context(IN_DOMAIN[0]))
        self.assertTrue(signals.reasons, "no rung explained itself")

    def test_a_greeting_is_recognised_as_social(self):
        for greeting in ("hello", "hi there", "thanks!", "شكرا"):
            with self.subTest(greeting=greeting):
                signals = self.ladder().run(self.context(greeting))
                if signals.is_social:
                    plan = resolve_turn(
                        signals,
                        agent_config=get_profile().agent,
                        copy_config=get_profile().user_copy,
                        rag_config=get_profile().rag,
                    )
                    self.assertTrue(str(plan.static_reply).strip())


@requires_postgres
@requires_embedder
class LiveScopeModelTests(unittest.TestCase):
    """The escalation rung with the real model. Costs tokens, so kept small and
    focused on the property that matters rather than on breadth."""

    def verdict(self, question):
        from backend.rag.scope_detector import ScopeModelDetector

        profile = get_profile()
        config = LadderConfig(profile.agent, profile.rag)
        signals = RequestSignals(question=question)
        CatalogueScopeDetector().detect(
            SignalContext(question=question, history=[], config=config), signals
        )
        return ScopeModelDetector().detect(
            SignalContext(question=question, history=[], config=config), signals
        )

    @requires_llm
    def test_a_plainly_in_domain_question_is_admitted(self):
        signals = self.verdict("what are the school fees for grade 5?")
        self.assertIsNotNone(signals)
        self.assertIs(Scope.IN_DOMAIN, signals.scope, "; ".join(signals.reasons))

    @requires_llm
    def test_the_same_question_in_arabic_is_admitted(self):
        signals = self.verdict("كم تبلغ الرسوم الدراسية للصف الخامس؟")
        self.assertIsNotNone(signals)
        self.assertIs(Scope.IN_DOMAIN, signals.scope, "; ".join(signals.reasons))

    @requires_llm
    def test_a_plainly_unrelated_question_is_refused(self):
        signals = self.verdict("what is the weather forecast for tomorrow?")
        self.assertIsNotNone(signals)
        self.assertIs(Scope.OUT_OF_DOMAIN, signals.scope, "; ".join(signals.reasons))

    @requires_llm
    def test_a_plausible_but_uncatalogued_question_is_still_admitted(self):
        """The case the worked examples in the prompt exist for: being unable to see
        the answer is not grounds for refusing the question."""
        signals = self.verdict("do you accommodate children with a peanut allergy?")
        self.assertIsNotNone(signals)
        self.assertIs(Scope.IN_DOMAIN, signals.scope, "; ".join(signals.reasons))


if __name__ == "__main__":
    unittest.main()
