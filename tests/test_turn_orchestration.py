"""The turn planner, and its wiring into the chat service.

The planner's whole value is work NOT done, so the assertions are mostly about
absence: the agent was never built, the tool list was shorter, no model was called.

Its whole risk is the same thing, so the rest pin that it degrades to today's
behaviour whenever anything is uncertain or broken.
"""
import asyncio
import json
import unittest
from unittest.mock import patch

from backend.chat.orchestrator import plan_turn
from backend.chat.signals import RequestSignals, Scope
from backend.chat.turn_policy import TurnPlan
from backend.profiles import get_profile
from backend.profiles.registry import load_profile, set_profile
from backend.rag.evidence import Certainty


class RecordingContext:
    """Captures progress steps instead of streaming them."""

    def __init__(self):
        self.steps = []

    def emit_rag_step(self, icon, label, detail="", **kwargs):
        self.steps.append((icon, label, detail))


def temp_profile(**agent_overrides):
    """Activate the base profile with the agent section adjusted."""
    profile = load_profile("base")
    return profile.model_copy(
        update={"agent": profile.agent.model_copy(update=agent_overrides)}
    )


class ActiveProfile:
    def __init__(self, profile):
        self._profile = profile

    def __enter__(self):
        set_profile(self._profile)
        return self._profile

    def __exit__(self, *exc):
        set_profile(None)


class PlanTurnTests(unittest.TestCase):
    def test_an_ordinary_question_changes_nothing(self):
        plan, signals = plan_turn("what are the school partners", [], None)
        self.assertFalse(plan.short_circuit)
        self.assertIsNone(plan.exposed_tools, "None means the profile's full tool list")

    def test_a_social_turn_unbinds_every_tool(self):
        plan, _ = plan_turn("thanks", [], None)
        self.assertEqual([], plan.exposed_tools)
        self.assertFalse(plan.short_circuit, "a greeting still gets a real reply")

    def test_a_social_turn_can_be_answered_without_a_model(self):
        with ActiveProfile(temp_profile(social_reply_mode="static")):
            plan, _ = plan_turn("thanks", [], None)
        self.assertTrue(plan.short_circuit)
        self.assertIn("Happy to help", plan.static_reply)

    def test_an_arabic_social_turn_gets_arabic_copy(self):
        with ActiveProfile(temp_profile(social_reply_mode="static")):
            plan, signals = plan_turn("شكرا", [], None)
        self.assertEqual("ar", signals.language)
        self.assertIn("سعيد بمساعدتك", plan.static_reply)

    def test_a_question_that_opens_with_a_greeting_is_not_social(self):
        plan, _ = plan_turn("hi, what are the school fees?", [], None)
        self.assertIsNone(plan.exposed_tools)
        self.assertFalse(plan.short_circuit)

    def test_a_planner_failure_never_costs_the_turn(self):
        with patch("backend.chat.orchestrator.build_ladder", side_effect=RuntimeError("boom")):
            plan, _ = plan_turn("what are the fees", [], None)
        self.assertFalse(plan.short_circuit)
        self.assertIsNone(plan.exposed_tools)
        self.assertIn("planner error", "; ".join(plan.reasons))

    def test_the_decision_is_reported_as_a_progress_step(self):
        ctx = RecordingContext()
        plan_turn("thanks", [], ctx)
        self.assertTrue(any("Narrowed" in label for _, label, _ in ctx.steps))

    def test_an_unremarkable_turn_stays_quiet(self):
        """A plan that changed nothing must not add noise to every turn's steps."""
        ctx = RecordingContext()
        plan_turn("what are the school partners", [], ctx)
        self.assertEqual([], ctx.steps)

    def test_detectors_see_both_profile_sections(self):
        """Social phrases live under `agent`, gate thresholds under `rag`. A detector
        should not have to know which."""
        captured = {}

        class Probe:
            name = "probe"
            certainty = Certainty.LOW

            def detect(self, ctx, signals):
                captured["social"] = getattr(ctx.config, "social_phrases", None)
                captured["gate"] = getattr(ctx.config, "domain_gate_min_similarity", None)
                return None

        with patch("backend.chat.orchestrator.build_ladder") as build:
            from backend.chat.signals import SignalLadder

            build.return_value = SignalLadder([Probe()], required=Certainty.MEDIUM)
            plan_turn("anything", [], None)

        self.assertTrue(captured["social"])
        self.assertIsNotNone(captured["gate"])


class ServiceWiringTests(unittest.IsolatedAsyncioTestCase):
    """The saving only exists if the agent is genuinely never constructed."""

    async def _stream(self, plan, signals=None):
        import backend.chat.service as service

        built = []

        def spy_create_agent(ctx, tool_names=None, language=None):
            built.append(tool_names)
            raise AssertionError("the agent must not be built on a short-circuited turn")

        chunks = []
        with patch.object(service, "plan_turn", lambda *a, **k: (plan, signals or RequestSignals())), \
             patch.object(service, "create_agent_for_request", spy_create_agent), \
             patch.object(service.storage, "load_with_meta", lambda *a: ([], {})), \
             patch.object(service.storage, "save", lambda *a, **k: None), \
             patch.object(service, "generate_session_title", lambda _t: "T"):
            async for chunk in service.chat_with_agent_stream("what is the weather", "u", "s"):
                chunks.append(chunk)
        return chunks, built

    def _events(self, chunks):
        events = []
        for chunk in chunks:
            payload = chunk.removeprefix("data: ").strip()
            if payload and payload != "[DONE]":
                events.append(json.loads(payload))
        return events

    async def test_a_short_circuited_turn_never_builds_the_agent(self):
        plan = TurnPlan(static_reply="Out of scope, sorry.", exposed_tools=[],
                        reasons=["out of domain"])
        chunks, built = await self._stream(plan)
        self.assertEqual([], built)
        self.assertIn("data: [DONE]\n\n", chunks)

    async def test_the_static_reply_reaches_the_client_as_content(self):
        plan = TurnPlan(static_reply="Out of scope, sorry.", exposed_tools=[])
        chunks, _ = await self._stream(plan)
        contents = [e["content"] for e in self._events(chunks) if e.get("type") == "content"]
        self.assertEqual(["Out of scope, sorry."], contents)

    async def test_a_short_circuited_turn_still_emits_a_trace(self):
        """A client must not have to know which path produced its answer."""
        plan = TurnPlan(static_reply="No.", exposed_tools=[], reasons=["out of domain"])
        signals = RequestSignals(question="q", scope=Scope.OUT_OF_DOMAIN,
                                 scope_certainty=Certainty.HIGH)
        chunks, _ = await self._stream(plan, signals)
        traces = [e for e in self._events(chunks) if e.get("type") == "trace"]
        self.assertEqual(1, len(traces))
        self.assertTrue(traces[0]["rag_trace"]["turn_short_circuit"])

    async def test_the_first_turn_still_gets_a_session_title(self):
        plan = TurnPlan(static_reply="No.", exposed_tools=[])
        chunks, _ = await self._stream(plan)
        titles = [e for e in self._events(chunks) if e.get("type") == "session_title"]
        self.assertEqual(1, len(titles))

    async def test_a_normal_turn_passes_the_narrowed_list_to_the_factory(self):
        import backend.chat.service as service

        captured = {}

        class FakeAgent:
            async def astream(self, *a, **k):
                if False:
                    yield None

        def spy_create_agent(ctx, tool_names=None, language=None):
            captured["tools"] = tool_names
            return FakeAgent()

        plan = TurnPlan(exposed_tools=["search_knowledge_base"])
        with patch.object(service, "plan_turn", lambda *a, **k: (plan, RequestSignals())), \
             patch.object(service, "create_agent_for_request", spy_create_agent), \
             patch.object(service.storage, "load_with_meta", lambda *a: ([], {})), \
             patch.object(service.storage, "save", lambda *a, **k: None), \
             patch.object(service, "generate_session_title", lambda _t: "T"):
            async for _ in service.chat_with_agent_stream("q", "u", "s"):
                pass

        self.assertEqual(["search_knowledge_base"], captured["tools"])


class AgentFactoryTests(unittest.TestCase):
    def test_none_means_the_profile_list(self):
        import backend.chat.runtime as runtime

        captured = {}
        with patch.object(runtime, "build_tools", lambda names, ctx: captured.setdefault("n", names) or []), \
             patch.object(runtime, "create_agent", lambda **kw: kw):
            runtime.create_agent_for_request(object(), None)
        self.assertEqual(get_profile().agent.tools, captured["n"])

    def test_an_explicit_list_narrows_what_is_bound(self):
        import backend.chat.runtime as runtime

        captured = {}
        with patch.object(runtime, "build_tools", lambda names, ctx: captured.setdefault("n", names) or []), \
             patch.object(runtime, "create_agent", lambda **kw: kw):
            runtime.create_agent_for_request(object(), [])
        self.assertEqual([], captured["n"])


class RagGraphTests(unittest.TestCase):
    def test_the_rag_graph_no_longer_gates_domain_scope(self):
        """Scope moved one layer up. A gate here could no longer save the agent call it
        was meant to prevent, because the agent has already chosen to search."""
        import backend.rag.pipeline as pipeline

        self.assertFalse(hasattr(pipeline, "domain_gate_node"))
        self.assertFalse(hasattr(pipeline, "_route_after_domain_gate"))

    def test_the_evidence_ladder_is_untouched(self):
        import backend.rag.pipeline as pipeline

        self.assertTrue(hasattr(pipeline, "_assess_evidence"))
        self.assertTrue(hasattr(pipeline, "LLMGraderAssessor"))


if __name__ == "__main__":
    unittest.main()
