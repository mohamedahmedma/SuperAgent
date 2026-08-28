"""The classification node: one call, three decisions.

Two kinds of case here. The first is that the node reads a well-formed verdict
correctly. The second, and the one that earns its place, is that it survives a model
returning something it was not asked for — because the child half of this verdict
selects a real child, and a classifier that invents a name selects the wrong one.

The prompt itself is asserted too. It is 2,400 tokens paid on every turn, and the two
properties that make that affordable — everything fixed first, the message last — are
invisible to every other test in the suite.
"""
import unittest

from backend.chat.signals import (
    CHILD_REFERENCES,
    EnvelopeDetector,
    RequestSignals,
    Scope,
    SignalContext,
)
from backend.prompts import render
from backend.profiles import load_profile
from backend.rag.evidence import Certainty


def _config(**overrides):
    class _C:
        request_envelope_enabled = True
        child_context_enabled = True
        personal_data_fields = ["child_name", "year_group"]
        coverage = "Term dates, fees, uniform."
        query_resolution_history_messages = 6

    for key, value in overrides.items():
        setattr(_C, key, value)
    return _C()


def _run(payload, question="anything", history=(), **config_overrides):
    signals = RequestSignals(question=question)
    ctx = SignalContext(question=question, history=history, config=_config(**config_overrides))
    return EnvelopeDetector(invoke=lambda *a: payload).detect(ctx, signals)


class ScopeTests(unittest.TestCase):
    def test_an_out_of_domain_verdict_is_believed_at_high_certainty(self):
        signals = _run({"scope": "out_of_domain", "reason": "weather"})
        self.assertIs(signals.scope, Scope.OUT_OF_DOMAIN)
        self.assertIs(signals.scope_certainty, Certainty.HIGH)

    def test_an_in_domain_verdict_rescues_a_rejection_from_the_rung_below(self):
        signals = _run({"scope": "in_domain"})
        self.assertIs(signals.scope, Scope.IN_DOMAIN)

    def test_an_unusable_verdict_leaves_scope_unsettled_rather_than_guessing(self):
        signals = _run({"scope": "perhaps"})
        self.assertIs(signals.scope, Scope.UNKNOWN)

    def test_a_non_dict_result_abstains(self):
        self.assertIsNone(_run("not a dict"))
        self.assertIsNone(_run(None))

    def test_disclosing_a_detail_overrides_a_rejection(self):
        signals = _run({"scope": "out_of_domain", "personal_data": ["year_group"]})
        self.assertIs(signals.scope, Scope.IN_DOMAIN)

    def test_a_question_about_the_callers_child_is_never_out_of_domain(self):
        """The single worst answer this deployment can give is refusing "how is my son
        doing?" as off-topic. A prompt is a request; this has to hold every time."""
        signals = _run({"scope": "out_of_domain", "about_child": True, "child_reference": "son"})

        self.assertIs(signals.scope, Scope.IN_DOMAIN)
        self.assertTrue(signals.about_child)


class ChildVerdictTests(unittest.TestCase):
    def test_a_general_question_is_not_about_a_child(self):
        signals = _run({"scope": "in_domain", "about_child": False})
        self.assertFalse(signals.about_child)
        self.assertEqual(signals.child_reference, "none")
        self.assertEqual(signals.child_name, "")

    def test_every_declared_reference_survives(self):
        for reference in CHILD_REFERENCES:
            if reference == "none":
                continue
            with self.subTest(reference=reference):
                payload = {"scope": "in_domain", "about_child": True, "child_reference": reference}
                question = "anything"
                if reference == "named":
                    payload["child_name"] = "علي"
                    # The message has to actually name him, or the guard below rightly
                    # refuses a name the model could only have carried over.
                    question = "علي عامل ايه؟"
                self.assertEqual(_run(payload, question=question).child_reference, reference)

    def test_a_name_is_kept_verbatim_and_whitespace_normalised(self):
        signals = _run(
            {"scope": "in_domain", "about_child": True,
             "child_reference": "named", "child_name": "  علي   حسن "},
            question="درجات علي حسن ايه؟",
        )
        self.assertEqual(signals.child_name, "علي حسن")

    def test_a_reference_outside_the_closed_set_degrades_to_asking(self):
        """Not to guessing. An unrecognised reference must never narrow the roster."""
        signals = _run({"scope": "in_domain", "about_child": True, "child_reference": "eldest"})
        self.assertEqual(signals.child_reference, "context")

    def test_about_a_child_referred_to_in_no_way_is_read_as_context(self):
        signals = _run({"scope": "in_domain", "about_child": True, "child_reference": "none"})
        self.assertEqual(signals.child_reference, "context")

    def test_a_name_is_dropped_when_the_message_did_not_contain_one(self):
        """A name on any other reference kind is the model carrying one over from the
        conversation — which is a guess, and it resolves against real children."""
        signals = _run({
            "scope": "in_domain", "about_child": True,
            "child_reference": "son", "child_name": "علي",
        })
        self.assertEqual(signals.child_name, "")

    def test_named_with_no_name_falls_back_rather_than_selecting_nobody(self):
        signals = _run({
            "scope": "in_domain", "about_child": True,
            "child_reference": "named", "child_name": "   ",
        })
        self.assertEqual(signals.child_reference, "context")
        self.assertEqual(signals.child_name, "")

    def test_a_name_the_message_does_not_contain_is_refused(self):
        """Measured against the live model, not hypothetical.

        Asked to classify "طيب وجدوله؟" after a turn about علي, gpt-oss-20b returns
        reference "named" with "علي" — resolving the reference itself, which the
        prompt forbids
        because this node is not what holds the school's list of children. It guesses
        right while one child has been discussed and wrong the moment two have, and a
        wrong name here OVERRIDES a correct pin.
        """
        signals = _run(
            {"scope": "in_domain", "about_child": True,
             "child_reference": "named", "child_name": "علي"},
            question="طيب وجدوله؟",
        )

        self.assertEqual(signals.child_reference, "context")
        self.assertEqual(signals.child_name, "")

    def test_a_name_the_resolver_supplied_is_evidence_and_is_kept(self):
        """A resolver naming the child is the designed path — a separate call that read
        the whole conversation. Its name is not a guess."""
        ctx = SignalContext(
            question="طيب وجدوله؟",
            history=[{"role": "user", "content": "درجات علي"}],
            config=_config(),
            resolved_question="ما هو جدول علي الدراسي؟",
        )
        signals = EnvelopeDetector(
            invoke=lambda *a: {"scope": "in_domain", "about_child": True,
                               "child_reference": "named", "child_name": "علي"}
        ).detect(ctx, RequestSignals(question="طيب وجدوله؟"))

        self.assertEqual(signals.child_reference, "named")
        self.assertEqual(signals.child_name, "علي")

    def test_the_trace_reports_the_signal_but_never_the_name(self):
        """This trace is persisted per message and streamed to the browser, and a turn
        may resolve a child silently without ever showing the name."""
        signals = _run(
            {"scope": "in_domain", "about_child": True,
             "child_reference": "named", "child_name": "ليلى"},
            question="ليلى عاملة ايه؟",
        )
        trace = signals.as_trace()

        self.assertTrue(trace["request_about_child"])
        self.assertEqual(trace["request_child_reference"], "named")
        self.assertNotIn("ليلى", repr(trace))


class ItReadsTheResolvedQuestionTests(unittest.TestCase):
    def test_a_follow_up_is_classified_by_its_resolved_wording(self):
        """A follow-up's own words do not name its subject. Judging "and what about
        those?" alone measures nothing and looks identical to being off-topic."""
        seen = {}

        def spy(text, history, config):
            seen["text"] = text
            return {"scope": "in_domain"}

        ctx = SignalContext(
            question="وجدوله؟",
            history=[{"role": "user", "content": "درجات علي"}],
            config=_config(),
            resolved_question="ما هو جدول علي الدراسي؟",
        )
        EnvelopeDetector(invoke=spy).detect(ctx, RequestSignals(question="وجدوله؟"))

        self.assertEqual(seen["text"], "ما هو جدول علي الدراسي؟")


class PromptShapeTests(unittest.TestCase):
    def _render(self, **overrides):
        profile = load_profile("school")
        context = dict(
            question="طيب وجدوله؟",
            persona=profile.identity.persona,
            coverage=profile.agent.coverage,
            history="User: درجات ابني علي\nAssistant: ...",
            personal_fields=profile.agent.personal_data_fields,
            child_context=True,
        )
        context.update(overrides)
        return render("chat/request_envelope.j2", **context)

    def test_the_message_comes_last_so_the_rules_above_it_can_be_cached(self):
        """~2,400 tokens of rules are paid on every turn. Providers cache on a shared
        prefix, so a message placed above them invalidates all of it every time."""
        rendered = self._render()
        heading = rendered.index("# THE MESSAGE")
        self.assertLess(rendered.index("Decision 1"), heading)
        self.assertLess(rendered.index("# The conversation so far"), heading)
        self.assertTrue(rendered.rstrip().endswith("طيب وجدوله؟"))

    def test_the_child_section_is_not_paid_for_by_deployments_without_children(self):
        without = self._render(child_context=False)
        self.assertNotIn("Decision 3", without)
        self.assertIn("Decision 3", self._render())
        self.assertLess(len(without), len(self._render()))

    def test_the_coverage_description_actually_reaches_the_model(self):
        """Without a catalogue this is the only evidence the node has. An empty string
        here is how an assistant starts refusing valid questions."""
        self.assertIn("term dates", self._render().lower())

    def test_both_languages_are_shown_in_the_examples(self):
        rendered = self._render()
        self.assertIn("ابني", rendered)
        self.assertIn("my son", rendered)

    def test_the_enclitic_possessive_case_is_taught_explicitly(self):
        """The form requirement 4 takes in Arabic, and the one no word list can reach."""
        self.assertIn("طيب وجدوله؟", self._render())

    def test_a_persona_containing_jinja_is_data_not_template(self):
        rendered = self._render(persona="We are {{ evil }} school")
        self.assertIn("{{ evil }}", rendered)

    def test_it_renders_with_nothing_optional_supplied(self):
        rendered = self._render(history="", coverage="", personal_fields=[], child_context=False)
        self.assertIn("THE MESSAGE", rendered)
        self.assertNotIn("Decision 2", rendered)


if __name__ == "__main__":
    unittest.main()
