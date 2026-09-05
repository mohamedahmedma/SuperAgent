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
from backend.tools import KNOWLEDGE_TOOL


class RecordingContext:
    """Captures progress steps instead of streaming them."""

    def __init__(self):
        self.steps = []
        self.retrieval_sections = []
        self.scope_options = []

    def emit_rag_step(self, icon, label, detail="", **kwargs):
        self.steps.append((icon, label, detail))

    # Mirrors ChatRequestContext.note_turn_plan. It has to: the orchestrator swallows
    # any exception from this call ("a hint must never break a turn"), so a double whose
    # signature lags the real one does not fail — it silently records nothing.
    def note_turn_plan(self, retrieval_sections, scope_options, *,
                       carried_constraints=(), is_followup=False, language=""):
        self.retrieval_sections = list(retrieval_sections or [])
        self.scope_options = list(scope_options or [])
        self.carried_constraints = list(carried_constraints or [])
        self.is_followup = bool(is_followup)
        self.language = language


def temp_profile(**agent_overrides):
    """The base profile with the agent section adjusted.

    Scope detection is force-disabled unless a test asks for it: these tests describe
    the planner, and a deployment profile that enables the catalogue rung would
    otherwise pull them onto the network and make them depend on a corpus.
    """
    profile = load_profile("base")
    agent = {"request_envelope_enabled": False, **agent_overrides}
    return profile.model_copy(update={
        "agent": profile.agent.model_copy(update=agent),
        "rag": profile.rag.model_copy(update={"scope_index_enabled": False,
                                              "domain_gate_enabled": False}),
    })


class ActiveProfile:
    def __init__(self, profile):
        self._profile = profile

    def __enter__(self):
        set_profile(self._profile)
        return self._profile

    def __exit__(self, *exc):
        set_profile(None)


class PlanTurnTests(unittest.TestCase):
    def setUp(self):
        self._profile = ActiveProfile(temp_profile())
        self._profile.__enter__()

    def tearDown(self):
        self._profile.__exit__(None, None, None)

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


class TerminalToolResultTests(unittest.IsolatedAsyncioTestCase):
    """Once retrieval concludes there is nothing to say, the model is not asked to say it.

    The reply is profile copy either way. Letting the agent loop continue spends a whole
    generation call — with the conversation and the tool result as input — to paraphrase
    a string already in hand.
    """

    def _tool_message(self, name, content="NO_KNOWLEDGE: nothing found"):
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=content, tool_call_id="c1", name=name)

    async def _run(self, stream_items, trace, plan=None):
        import backend.chat.service as service
        from backend.chat.turn_policy import TurnPlan

        generated = []

        class FakeAgent:
            async def astream(self, *a, **k):
                for item in stream_items:
                    if isinstance(item, str):
                        chunk = AIMessageChunk(content=item)
                        generated.append(item)
                        yield chunk, {}
                    else:
                        yield item, {}

        class Ctx:
            def peek_rag_trace(self):
                return {"rag_trace": trace}

        chunks = []
        with patch.object(service, "plan_turn",
                          lambda *a, **k: (plan or TurnPlan(), RequestSignals())), \
             patch.object(service, "create_agent_for_request", lambda ctx, tools=None: FakeAgent()), \
             patch.object(service, "ChatRequestContext", Ctx), \
             patch.object(service.storage, "load_with_meta", lambda *a: ([], {})), \
             patch.object(service.storage, "save", lambda *a, **k: None), \
             patch.object(service, "generate_session_title", lambda _t: "T"):
            async for chunk in service.chat_with_agent_stream("what is partner", "u", "s"):
                chunks.append(chunk)
        return chunks, generated

    def test_terminal_statuses_are_named(self):
        import backend.chat.runtime as runtime

        self.assertEqual({"no_knowledge", "retrieval_error"}, runtime.TERMINAL_STATUSES)

    def test_the_reply_comes_from_profile_copy_in_the_right_language(self):
        import backend.chat.service as service

        english = service._terminal_reply("no_knowledge", "en")
        arabic = service._terminal_reply("no_knowledge", "ar")
        self.assertTrue(english)
        self.assertTrue(arabic)
        self.assertEqual(english, service._terminal_reply("no_knowledge", "de"),
                         "an unknown language falls back rather than blanking")

    def test_retrieval_error_and_no_knowledge_read_differently(self):
        import backend.chat.service as service

        self.assertNotEqual(
            service._terminal_reply("no_knowledge", "en"),
            service._terminal_reply("retrieval_error", "en"),
        )

    def test_a_non_terminal_status_is_not_short_circuited(self):
        import backend.chat.runtime as runtime

        class Ctx:
            def peek_rag_trace(self):
                return {"rag_trace": {"retrieval_status": "answerable"}}

        self.assertIsNone(runtime.terminal_status(Ctx()))

    def test_a_terminal_status_is_recognised(self):
        import backend.chat.runtime as runtime

        class Ctx:
            def peek_rag_trace(self):
                return {"rag_trace": {"retrieval_status": "no_knowledge"}}

        self.assertEqual("no_knowledge", runtime.terminal_status(Ctx()))

    def test_a_missing_trace_is_not_terminal(self):
        import backend.chat.runtime as runtime

        class Ctx:
            def peek_rag_trace(self):
                return None

        self.assertIsNone(runtime.terminal_status(Ctx()))


class TerminalShortCircuitMiddlewareTests(unittest.TestCase):
    """The loop is cut inside the graph, not by abandoning its stream.

    Breaking out of `astream` from the consumer side left the generator suspended for
    the garbage collector to close, which threw GeneratorExit into LangGraph at its
    `yield` and reported every short-circuited turn as a failed run.
    """

    class Ctx:
        def __init__(self, status):
            self._status = status
            self.noted = None

        def peek_rag_trace(self):
            return {"rag_trace": {"retrieval_status": self._status}}

        def note_short_circuit(self, status):
            self.noted = status

    def _state(self, tool_names):
        """Tool results named explicitly, because WHICH tool ran is the whole rule.

        The names used to be `tool_0`, `tool_1` placeholders: the guard counted results
        and never read them. It now counts tools, so a fixture that names them
        arbitrarily is describing a different turn than it means to.
        """
        from langchain_core.messages import HumanMessage, ToolMessage

        messages = [HumanMessage(content="what is partner")]
        messages += [
            ToolMessage(content="NO_KNOWLEDGE", tool_call_id=f"c{i}", name=name)
            for i, name in enumerate(tool_names)
        ]
        return {"messages": messages}

    def _decide(self, status, tool_names):
        import backend.chat.runtime as runtime

        ctx = self.Ctx(status)
        middleware = runtime._end_turn_on_terminal_retrieval(ctx)
        return middleware.before_model(self._state(tool_names), None), ctx

    def test_a_terminal_result_ends_the_graph_before_the_second_model_call(self):
        verdict, ctx = self._decide("no_knowledge", [KNOWLEDGE_TOOL])
        self.assertEqual({"jump_to": "end"}, verdict)
        self.assertEqual("no_knowledge", ctx.noted)

    def test_duplicate_knowledge_calls_still_end_the_graph(self):
        """The regression that let an invented fee reach a parent.

        This model emits the same search_knowledge_base call two to five times in one
        message (measured on 4 of 6 turns). The guard counted RESULTS, so a turn whose
        corpus said `no_knowledge` arrived with two or three of them, `!= 1` was true,
        and the rail stood down — leaving the model free to answer a fees question from
        its own knowledge. Repeat calls to one tool are one tool having run.
        """
        for count in (2, 3, 5):
            with self.subTest(duplicate_calls=count):
                verdict, ctx = self._decide("no_knowledge", [KNOWLEDGE_TOOL] * count)
                self.assertEqual({"jump_to": "end"}, verdict)
                self.assertEqual("no_knowledge", ctx.noted)

    def test_the_first_model_call_is_never_cut(self):
        """Nothing has retrieved yet, so there is no trace and nothing to short-circuit."""
        verdict, ctx = self._decide(None, [])
        self.assertIsNone(verdict)
        self.assertIsNone(ctx.noted)

    def test_a_usable_result_leaves_the_loop_alone(self):
        verdict, ctx = self._decide("answerable", [KNOWLEDGE_TOOL])
        self.assertIsNone(verdict)
        self.assertIsNone(ctx.noted)

    def test_a_second_tool_keeps_the_model_call(self):
        """A turn that also called, say, the records tool still has material to answer
        from, and cutting it there would throw that away."""
        verdict, ctx = self._decide("no_knowledge", [KNOWLEDGE_TOOL, "get_student_records"])
        self.assertIsNone(verdict)
        self.assertIsNone(ctx.noted)

    def test_the_jump_target_is_declared_to_the_graph(self):
        """`can_jump_to` is what builds the conditional edge. Without it the returned
        `jump_to` is silently ignored and the model call happens anyway."""
        import backend.chat.runtime as runtime

        middleware = runtime._end_turn_on_terminal_retrieval(self.Ctx("no_knowledge"))
        self.assertEqual(["end"], getattr(middleware.before_model, "__can_jump_to__", None))

    def test_the_agent_is_built_with_both_guards_bound(self):
        """Named rather than counted: an unbound middleware is inert, and a bare count
        says nothing about WHICH of the two went missing."""
        import backend.chat.runtime as runtime

        captured = {}
        with patch.object(runtime, "build_tools", lambda names, ctx: []), \
             patch.object(runtime, "create_agent", lambda **kw: captured.update(kw)):
            runtime.create_agent_for_request(self.Ctx("no_knowledge"), [])

        bound = " ".join(
            f"{type(item).__name__} {getattr(item, 'name', '')}"
            for item in captured["middleware"]
        )
        self.assertIn("_drop_repeated_calls", bound)
        self.assertIn("_stop_after_a_terminal_tool_result", bound)


class ShortCircuitedStreamTests(unittest.IsolatedAsyncioTestCase):
    """The whole point, on a real graph: the loop is cut without abandoning the stream.

    Asserted on a real `create_agent` rather than a stand-in, because what broke was the
    interaction with LangGraph's own generator. The stream is consumed to exhaustion
    here — no `break` — so a saved model call can only mean the graph ended itself.
    Without the middleware the same exhausted stream costs two model calls, which is
    what makes this assertion worth making.
    """

    async def test_the_stream_completes_without_a_second_model_call(self):
        from langchain.agents import create_agent
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import tool

        from backend.chat.runtime import _end_turn_on_terminal_retrieval

        calls = {"model": 0}

        class FakeToolCaller(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake-tool-caller"

            def bind_tools(self, tools, **kw):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kw):
                calls["model"] += 1
                message = (
                    AIMessage(content="", tool_calls=[{
                        "name": "search_knowledge_base",
                        "args": {"query": "partner"},
                        "id": "c1",
                        "type": "tool_call",
                    }])
                    if calls["model"] == 1
                    else AIMessage(content="a reworded fallback nobody asked for")
                )
                return ChatResult(generations=[ChatGeneration(message=message)])

        class Ctx:
            trace = None
            short = None

            def peek_rag_trace(self):
                return {"rag_trace": self.trace} if self.trace else None

            def note_short_circuit(self, status):
                self.short = status

        ctx = Ctx()

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base."""
            ctx.trace = {"retrieval_status": "retrieval_error"}
            return "RETRIEVAL_ERROR: the knowledge base could not be searched right now."

        agent = create_agent(
            model=FakeToolCaller(),
            tools=[search_knowledge_base],
            middleware=[_end_turn_on_terminal_retrieval(ctx)],
        )

        # Consumed to exhaustion, with no `break`. Nothing is left suspended for the
        # garbage collector to close, which is what threw GeneratorExit into LangGraph.
        async for _msg, _meta in agent.astream(
            {"messages": [HumanMessage(content="what is partner")]},
            stream_mode="messages",
        ):
            pass

        self.assertEqual(1, calls["model"], "the composing call was supposed to be skipped")
        self.assertEqual("retrieval_error", ctx.short)



class TheContextReceivesThePlan(unittest.TestCase):
    """`_hand_to_graph` is the only route from the plan to the tools.

    It is wrapped in a blanket `except`, deliberately — a hint must never break a turn —
    which is exactly what makes it worth testing directly: a field that never arrives
    fails silently and looks like a feature that was never switched on.
    """

    def _handed(self, **plan_kwargs):
        from backend.chat.caller_identity import CallerIdentity
        from backend.chat.orchestrator import _hand_to_graph
        from backend.chat.request_context import ChatRequestContext

        ctx = ChatRequestContext(
            user_id="u-1",
            session_id="s-1",
            caller=CallerIdentity(user_id="u-1", guardian_id="G-1", guardian_token="t"),
        )
        _hand_to_graph(ctx, TurnPlan(**plan_kwargs))
        return ctx

    def test_the_resolved_child_reaches_the_tools(self):
        ctx = self._handed(child_hint="ليلى أحمد", child_id="S-1")
        self.assertEqual(ctx.planned_child_id, "S-1")
        self.assertEqual(ctx.planned_child_label, "ليلى أحمد")

    def test_the_required_tool_reaches_the_middleware(self):
        ctx = self._handed(forced_tool="get_student_records")
        self.assertEqual(ctx.forced_tool, "get_student_records")

    def test_a_plan_that_settled_nothing_leaves_nothing_behind(self):
        ctx = self._handed()
        self.assertEqual(ctx.planned_child_id, "")
        self.assertEqual(ctx.forced_tool, "")

    def test_an_older_context_still_gets_the_hints_it_understands(self):
        """The version ladder. A context predating these arguments must keep working and
        simply carry fewer hints — the promise `note_turn_plan` makes in as many words.
        Anything added to the hints must go to the FRONT of the drop list, or the retry
        re-sends the argument that raised and the ladder hands over nothing at all."""
        from backend.chat.orchestrator import _hand_to_graph

        class _OldContext:
            def note_turn_plan(self, retrieval_sections, scope_options, *,
                               carried_constraints=(), is_followup=False, language=""):
                self.sections = list(retrieval_sections)
                self.language = language

        ctx = _OldContext()
        _hand_to_graph(ctx, TurnPlan(
            retrieval_sections=["fees"], language="ar",
            child_hint="ليلى", child_id="S-1", forced_tool="get_student_records",
        ))
        self.assertEqual(ctx.sections, ["fees"])
        self.assertEqual(ctx.language, "ar")
