"""The request-signal ladder and the turn policy that reads it.

Two invariants carry the safety of this layer, and most of these tests exist to pin
them:

  * A cheap rung may ADMIT a question to the corpus but never REJECT it. Only a
    detector that read the question can end a turn.
  * There is no state in which the knowledge tool is reachable but crippled. Either
    the turn ends before the agent runs, or the tool works.
"""
import unittest
from unittest.mock import patch

from backend.chat.language import ARABIC, ENGLISH, arabic_ratio, detect_language
from backend.chat.signals import (
    CorpusSimilarityDetector,
    EnvelopeDetector,
    RequestSignals,
    Scope,
    SignalContext,
    SignalLadder,
    SocialDetector,
    build_ladder,
)
from backend.chat.turn_policy import TurnPlan, localized, resolve_turn
from backend.profiles.registry import load_profile
from backend.rag.evidence import Certainty


def agent_config(**overrides):
    return load_profile("base").agent.model_copy(update=overrides) if overrides else load_profile("base").agent


def copy_config(**overrides):
    return load_profile("base").user_copy.model_copy(update=overrides) if overrides else load_profile("base").user_copy


def ctx(question, history=(), config=None):
    return SignalContext(question=question, history=history, config=config or agent_config())


def user_message(text):
    return {"role": "user", "content": text}


class StubDetector:
    def __init__(self, name, certainty, mutate=None):
        self.name = name
        self.certainty = certainty
        self._mutate = mutate
        self.calls = 0

    def detect(self, context, signals):
        self.calls += 1
        if self._mutate is None:
            return None
        self._mutate(signals)
        return signals


def _sets(scope, certainty):
    def mutate(signals):
        signals.scope = scope
        signals.scope_certainty = certainty
    return mutate


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

class LanguageTests(unittest.TestCase):
    def test_arabic_and_english_questions(self):
        self.assertEqual(ARABIC, detect_language("ما هو الزي الرياضي للمدرسة"))
        self.assertEqual(ENGLISH, detect_language("what is the school sports uniform"))

    def test_arabic_with_latin_proper_nouns_stays_arabic(self):
        """Arabic questions routinely carry Latin-script names. Requiring a majority
        would answer them in English."""
        self.assertEqual(ARABIC, detect_language("ما هو الـ IB Diploma في المدرسة"))

    def test_digits_and_punctuation_do_not_vote(self):
        self.assertEqual(ARABIC, detect_language("رقم ٩٦٦٥٥٨٩٨٩٦٥٣ ؟ الرسوم 2026"))

    def test_empty_input_uses_the_default(self):
        self.assertEqual(ENGLISH, detect_language("   "))
        self.assertEqual(ARABIC, detect_language("", default=ARABIC))

    def test_ratio_ignores_non_letters(self):
        self.assertEqual(0.0, arabic_ratio("123 !!! ???"))
        self.assertEqual(1.0, arabic_ratio("شكرا"))


# ---------------------------------------------------------------------------
# Social lookup
# ---------------------------------------------------------------------------

class SocialDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = SocialDetector()

    def _detect(self, text):
        return self.detector.detect(ctx(text), RequestSignals(question=text))

    def test_a_bare_pleasantry_matches(self):
        for text in ("thanks", "Thank you!", "hello", "  ok  ", "شكرا", "مرحبا"):
            self.assertIsNotNone(self._detect(text), text)

    def test_a_question_wearing_a_greeting_does_not(self):
        """The failure this list must never cause: answering a real question with a
        pleasantry."""
        for text in (
            "thanks, and what are the fees?",
            "hi, do you know the term dates",
            "sure, tell me about the uniform",
            "ok and what about grade 6",
            "شكرا، ما هي الرسوم؟",
        ):
            self.assertIsNone(self._detect(text), text)

    def test_punctuation_and_case_do_not_defeat_the_match(self):
        self.assertIsNotNone(self._detect("Thanks!!!"))
        self.assertIsNotNone(self._detect("THANK YOU."))

    def test_a_social_turn_is_not_out_of_domain(self):
        """Routing a greeting to a refusal is the wrong reply to "thank you"."""
        signals = self._detect("thanks")
        self.assertTrue(signals.is_social)
        self.assertIs(Scope.UNKNOWN, signals.scope)

    def test_an_empty_phrase_list_disables_it(self):
        config = agent_config(social_phrases=[])
        self.assertIsNone(
            SocialDetector().detect(ctx("thanks", config=config), RequestSignals(question="thanks"))
        )


# ---------------------------------------------------------------------------
# Corpus similarity
# ---------------------------------------------------------------------------

class Verdict:
    def __init__(self, in_domain, score=0.9, topics=(), abstained=False, reason="r"):
        self.in_domain = in_domain
        self.score = score
        self.topics = list(topics)
        self.abstained = abstained
        self.reason = reason


class CorpusSimilarityTests(unittest.TestCase):
    def _detect(self, verdict, question="what is the uniform policy", history=()):
        signals = RequestSignals(question=question)
        with patch("backend.indexing.embedding.embed_query", lambda _t: [0.1, 0.2]), \
             patch("backend.rag.domain_gate.classify", lambda *a, **k: verdict), \
             patch("backend.rag.domain_gate.reference_store"):
            return CorpusSimilarityDetector().detect(ctx(question, history), signals)

    def test_a_match_admits_at_medium_certainty(self):
        signals = self._detect(Verdict(True, score=0.82, topics=["uniform"]))
        self.assertIs(Scope.IN_DOMAIN, signals.scope)
        self.assertEqual(Certainty.MEDIUM, signals.scope_certainty)
        self.assertEqual(["uniform"], signals.candidate_sections)

    def test_a_miss_rejects_only_at_low_certainty(self):
        """A weak score can mean off-topic, or a different language, or different
        vocabulary. It is not a verdict."""
        signals = self._detect(Verdict(False, score=0.04))
        self.assertIs(Scope.OUT_OF_DOMAIN, signals.scope)
        self.assertEqual(Certainty.LOW, signals.scope_certainty)

    def test_abstention_leaves_scope_untouched(self):
        self.assertIsNone(self._detect(Verdict(True, abstained=True)))

    def test_an_embedding_failure_abstains(self):
        signals = RequestSignals(question="q")
        with patch("backend.indexing.embedding.embed_query", side_effect=RuntimeError("boom")):
            self.assertIsNone(CorpusSimilarityDetector().detect(ctx("q"), signals))

    def test_a_follow_up_is_scored_with_the_turn_before_it(self):
        """"what about grade 6?" carries its subject in the previous turn."""
        captured = {}

        def fake_embed(text):
            captured["text"] = text
            return [0.1, 0.2]

        signals = RequestSignals(question="what about grade 6")
        history = [user_message("what are the fees for grade 5")]
        with patch("backend.indexing.embedding.embed_query", fake_embed), \
             patch("backend.rag.domain_gate.classify", lambda *a, **k: Verdict(True)), \
             patch("backend.rag.domain_gate.reference_store"):
            CorpusSimilarityDetector().detect(ctx("what about grade 6", history), signals)

        self.assertIn("grade 5", captured["text"])
        self.assertIn("grade 6", captured["text"])

    def test_a_first_turn_is_scored_alone(self):
        captured = {}

        def fake_embed(text):
            captured["text"] = text
            return [0.1, 0.2]

        with patch("backend.indexing.embedding.embed_query", fake_embed), \
             patch("backend.rag.domain_gate.classify", lambda *a, **k: Verdict(True)), \
             patch("backend.rag.domain_gate.reference_store"):
            CorpusSimilarityDetector().detect(ctx("what are the fees"), RequestSignals())

        self.assertEqual("what are the fees", captured["text"].strip())


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

class EnvelopeTests(unittest.TestCase):
    def _detect(self, payload, signals=None):
        detector = EnvelopeDetector(invoke=lambda *a: payload)
        return detector.detect(ctx("q"), signals or RequestSignals(question="q"))

    def test_it_confirms_out_of_domain_at_high_certainty(self):
        signals = self._detect({"scope": "out_of_domain", "reason": "asks about football"})
        self.assertIs(Scope.OUT_OF_DOMAIN, signals.scope)
        self.assertEqual(Certainty.HIGH, signals.scope_certainty)

    def test_it_can_rescue_a_false_rejection(self):
        """The override that lets the cheap rung be imperfect without being dangerous."""
        prior = RequestSignals(question="q", scope=Scope.OUT_OF_DOMAIN,
                               scope_certainty=Certainty.LOW)
        signals = self._detect({"scope": "in_domain"}, prior)
        self.assertIs(Scope.IN_DOMAIN, signals.scope)
        self.assertEqual(Certainty.HIGH, signals.scope_certainty)

    def test_scope_and_personal_data_arrive_together(self):
        signals = self._detect({"scope": "in_domain", "personal_data": ["phone", "child_age"]})
        self.assertEqual(["phone", "child_age"], signals.personal_data)

    def test_disclosing_profile_data_overrides_an_out_of_domain_verdict(self):
        """Someone answering "he is 9" is continuing a conversation, not changing
        the subject."""
        signals = self._detect({"scope": "out_of_domain", "personal_data": ["child_age"]})
        self.assertIs(Scope.IN_DOMAIN, signals.scope)

    def test_an_unusable_verdict_leaves_scope_alone(self):
        signals = self._detect({"scope": "banana"})
        self.assertIs(Scope.UNKNOWN, signals.scope)

    def test_a_non_dict_result_abstains(self):
        self.assertIsNone(self._detect("nonsense"))


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

class LadderTests(unittest.TestCase):
    def test_a_confident_admission_stops_the_climb(self):
        cheap = StubDetector("cheap", Certainty.MEDIUM, _sets(Scope.IN_DOMAIN, Certainty.MEDIUM))
        expensive = StubDetector("envelope", Certainty.HIGH, _sets(Scope.IN_DOMAIN, Certainty.HIGH))
        SignalLadder([cheap, expensive], required=Certainty.MEDIUM).run(ctx("q"))
        self.assertEqual(1, cheap.calls)
        self.assertEqual(0, expensive.calls, "the in-corpus majority must not reach the model")

    def test_a_tentative_rejection_always_climbs(self):
        """The core rule: a cheap rung may admit, never reject. Even at a MEDIUM floor,
        a LOW out-of-domain must escalate."""
        cheap = StubDetector("cheap", Certainty.MEDIUM, _sets(Scope.OUT_OF_DOMAIN, Certainty.LOW))
        expensive = StubDetector("envelope", Certainty.HIGH, _sets(Scope.IN_DOMAIN, Certainty.HIGH))
        signals = SignalLadder([cheap, expensive], required=Certainty.MEDIUM).run(ctx("q"))
        self.assertEqual(1, expensive.calls)
        self.assertIs(Scope.IN_DOMAIN, signals.scope)

    def test_a_social_match_stops_immediately(self):
        def mark_social(signals):
            signals.is_social = True
            signals.scope_certainty = Certainty.HIGH

        social = StubDetector("social", Certainty.HIGH, mark_social)
        expensive = StubDetector("envelope", Certainty.HIGH, _sets(Scope.IN_DOMAIN, Certainty.HIGH))
        SignalLadder([social, expensive], required=Certainty.MEDIUM).run(ctx("thanks"))
        self.assertEqual(0, expensive.calls)

    def test_a_failing_detector_never_breaks_the_turn(self):
        class Broken(StubDetector):
            def detect(self, context, signals):
                raise RuntimeError("boom")

        broken = Broken("broken", Certainty.MEDIUM)
        good = StubDetector("good", Certainty.HIGH, _sets(Scope.IN_DOMAIN, Certainty.HIGH))
        signals = SignalLadder([broken, good], required=Certainty.MEDIUM).run(ctx("q"))
        self.assertIs(Scope.IN_DOMAIN, signals.scope)
        self.assertEqual(["good"], signals.assessed_by)

    def test_provenance_is_stamped_by_the_ladder(self):
        detector = StubDetector("cheap", Certainty.MEDIUM, _sets(Scope.IN_DOMAIN, Certainty.MEDIUM))
        signals = SignalLadder([detector], required=Certainty.MEDIUM).run(ctx("q"))
        self.assertEqual(["cheap"], signals.assessed_by)
        self.assertEqual("cheap", signals.as_trace()["request_assessed_by"][0])

    def test_the_language_is_always_resolved(self):
        signals = SignalLadder([], required=Certainty.MEDIUM).run(ctx("ما هي الرسوم"))
        self.assertEqual(ARABIC, signals.language)

    def test_the_ladder_is_assembled_from_profile_config(self):
        names = [d.name for d in build_ladder(agent_config())._detectors]
        self.assertEqual(["social"], names, "gate and envelope are off by default")

        config = agent_config(request_envelope_enabled=True)
        config = config.model_copy(update={"domain_gate_enabled": True})
        self.assertIn("envelope", [d.name for d in build_ladder(config)._detectors])


# ---------------------------------------------------------------------------
# Turn policy
# ---------------------------------------------------------------------------

class TurnPolicyTests(unittest.TestCase):
    def _resolve(self, signals, **overrides):
        return resolve_turn(
            signals,
            agent_config=overrides.pop("agent", agent_config()),
            copy_config=overrides.pop("copy", copy_config()),
        )

    def test_an_unknown_scope_changes_nothing(self):
        plan = self._resolve(RequestSignals(question="q"))
        self.assertFalse(plan.short_circuit)
        self.assertIsNone(plan.exposed_tools, "None means bind everything the profile allows")

    def test_a_confirmed_out_of_domain_turn_ends_before_the_agent(self):
        signals = RequestSignals(question="what is the weather", language=ENGLISH,
                                 scope=Scope.OUT_OF_DOMAIN, scope_certainty=Certainty.HIGH)
        plan = self._resolve(signals)
        self.assertTrue(plan.short_circuit)
        self.assertEqual([], plan.exposed_tools)
        self.assertIn("outside what I can help with", plan.static_reply)

    def test_the_static_reply_follows_the_question_language(self):
        signals = RequestSignals(question="ما هو الطقس اليوم", language=ARABIC,
                                 scope=Scope.OUT_OF_DOMAIN, scope_certainty=Certainty.HIGH)
        self.assertIn("خارج نطاق", self._resolve(signals).static_reply)

    def test_a_tentative_rejection_never_ends_a_turn(self):
        """The whole safety argument: only something that read the question may refuse."""
        signals = RequestSignals(question="q", scope=Scope.OUT_OF_DOMAIN,
                                 scope_certainty=Certainty.LOW)
        plan = self._resolve(signals)
        self.assertFalse(plan.short_circuit)
        self.assertIsNone(plan.exposed_tools, "the knowledge tool stays bound AND working")
        self.assertIn("stays available", "; ".join(plan.reasons))

    def test_a_medium_rejection_also_never_ends_a_turn(self):
        signals = RequestSignals(question="q", scope=Scope.OUT_OF_DOMAIN,
                                 scope_certainty=Certainty.MEDIUM)
        self.assertFalse(self._resolve(signals).short_circuit)

    def test_sections_are_a_hint_and_never_a_restriction(self):
        """There is no field here that can remove a document from reach."""
        signals = RequestSignals(question="q", scope=Scope.IN_DOMAIN,
                                 scope_certainty=Certainty.MEDIUM,
                                 candidate_sections=["uniform", "fees"])
        plan = self._resolve(signals)
        self.assertEqual(["uniform", "fees"], plan.retrieval_sections)
        self.assertIsNone(plan.exposed_tools)

    def test_missing_copy_falls_through_to_the_agent(self):
        """Refusing with an empty string is worse than answering."""
        from backend.profiles.schema import LocalizedText

        signals = RequestSignals(question="q", scope=Scope.OUT_OF_DOMAIN,
                                 scope_certainty=Certainty.HIGH)
        plan = self._resolve(signals, copy=copy_config(out_of_domain=LocalizedText()))
        self.assertFalse(plan.short_circuit)
        self.assertIsNone(plan.exposed_tools)

    def test_a_social_turn_unbinds_the_tools_but_still_answers(self):
        plan = self._resolve(RequestSignals(question="thanks", is_social=True))
        self.assertFalse(plan.short_circuit)
        self.assertEqual([], plan.exposed_tools)

    def test_static_social_replies_are_opt_in(self):
        plan = self._resolve(RequestSignals(question="thanks", is_social=True, language=ENGLISH),
                             agent=agent_config(social_reply_mode="static"))
        self.assertTrue(plan.short_circuit)
        self.assertIn("Happy to help", plan.static_reply)

    def test_profile_capture_is_independent_of_scope(self):
        """Someone can give their phone number inside an off-topic message."""
        signals = RequestSignals(question="q", personal_data=["phone"],
                                 scope=Scope.OUT_OF_DOMAIN, scope_certainty=Certainty.HIGH)
        self.assertTrue(self._resolve(signals).capture_user_info)

    def test_the_plan_is_traceable(self):
        trace = TurnPlan(static_reply="no", exposed_tools=[], reasons=["out of domain"]).as_trace()
        self.assertTrue(trace["turn_short_circuit"])
        self.assertEqual([], trace["turn_exposed_tools"])


class LocalizedTextTests(unittest.TestCase):
    def test_it_picks_the_requested_language(self):
        from backend.profiles.schema import LocalizedText

        text = LocalizedText(en="hello", ar="مرحبا")
        self.assertEqual("hello", localized(text, ENGLISH))
        self.assertEqual("مرحبا", localized(text, ARABIC))

    def test_a_missing_translation_falls_back_rather_than_blanking(self):
        from backend.profiles.schema import LocalizedText

        self.assertEqual("hello", localized(LocalizedText(en="hello"), ARABIC))
        self.assertEqual("مرحبا", localized(LocalizedText(ar="مرحبا"), ENGLISH))

    def test_empty_and_none_are_handled(self):
        from backend.profiles.schema import LocalizedText

        self.assertEqual("", localized(LocalizedText(), ENGLISH))
        self.assertEqual("", localized(None, ENGLISH))

    def test_a_plain_string_still_works(self):
        self.assertEqual("legacy", localized("legacy", ARABIC))


if __name__ == "__main__":
    unittest.main()
