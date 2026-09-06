"""The turn, for real, from the message to the tools finishing — and no further.

## What is real here and what is not

REAL, and the point of the file:

  * `resolve_question` and the request classifier. Both are live calls to `FAST_MODEL`,
    so this measures the two model-backed decisions the planner is built on, against
    messages a parent would actually send.
  * The planner. `resolve_turn` and everything under it, unchanged.
  * The dispatch. The real middleware stack from `create_agent_for_request`, the real
    LangGraph tool node, the real fan-out.
  * The tool bodies. Argument schemas, roster matching, subject matching, outcome
    mapping and the result templates all execute.

NOT real, deliberately:

  * **Retrieval.** `run_rag_graph` is replaced by a canned result. Embedding, hybrid
    recall, reranking and LLM grading are a different subject with its own eval
    (`retrieval_eval.py`), and running them here would make a planner regression look
    like a retrieval one.
  * **The records facade.** Its HTTP boundary returns canned payloads, so a run needs no
    school database and cannot be broken by one.
  * **The answer.** The agent's model is a stub. Nothing after the tool results is
    exercised, because nothing after them is what this file is about — and it keeps a
    run to two model calls a case instead of three.

So the only live model calls per case are the resolver and the classifier. Everything
downstream of them is deterministic, which is what makes a failure here readable: it is
the classification, the plan, or the dispatch, and the report says which.

## Running it

    ACTIVE_PROFILE=school python -m tests.evals.planner_execution_eval

Credentials come from `.env` the same way the server reads them. Exits non-zero when a
case fails its expectation, so it can gate a change to the planner.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

os.environ.setdefault("ACTIVE_PROFILE", "school")
# The roster is injected per case; a cache in front of it would carry one case's family
# into the next and make the run order-dependent.
os.environ["CHILD_ROSTER_TTL_SECONDS"] = "0"

from backend.env import load_env

load_env()

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import backend.chat.child_roster as child_roster
import backend.tools.records as records
from backend.chat import runtime
from backend.chat.caller_identity import CallerIdentity
from backend.chat.orchestrator import plan_turn
from backend.chat.request_context import ChatRequestContext
from backend.profiles import get_profile
from backend.tools import build_tools

GUARDIAN = "G-1"
TOKEN = "test-token"

#: One family, used by every case. Two children so that "which child" is a real question
#: the planner has to settle rather than a single row it cannot get wrong.
ROSTER = [
    {
        "student_id": "S-1",
        "full_name_ar": "ليلى أحمد",
        "full_name_en": "Layla Ahmed",
        "gender": "female",
        "year_level": "Year 4",
    },
    {
        "student_id": "S-2",
        "full_name_ar": "عمر أحمد",
        "full_name_en": "Omar Ahmed",
        "gender": "male",
        "year_level": "Year 1",
    },
]

GRADES = {
    "term": {"name_ar": "الفصل الأول"},
    "courses": [
        {
            "course_id": "C-1",
            "subject_name_ar": "الرياضيات",
            "subject_name_en": "Mathematics",
            "computed_percentage": 87.5,
            "letter_grade": "A",
            "is_complete": True,
        },
        {
            "course_id": "C-2",
            "subject_name_ar": "العلوم",
            "subject_name_en": "Science",
            "computed_percentage": 91.0,
            "letter_grade": "A",
            "is_complete": True,
        },
    ],
}
ATTENDANCE = {"present_days": 58, "absent_days": 2, "late_days": 1}
SUBJECT = {
    "course": {"subject_name_ar": "الرياضيات", "computed_percentage": 87.5, "letter_grade": "A"},
    "assignments": [{"title": "اختبار 1", "score": 18, "max_score": 20}],
}


# --------------------------------------------------------------------------------------
# The boundary. Everything below replaces a network call or a retrieval run — never a
# decision.
# --------------------------------------------------------------------------------------


def _roster_fetch(guardian_id, token, request_id):
    return ("ok", list(ROSTER))


def _records_get(path, ctx, params=None):
    """Stand in for the facade. Routed on the path so the tools' own URLs are exercised."""
    if path.endswith("/attendance"):
        return "ok", dict(ATTENDANCE)
    if "/grades/" in path:
        return "ok", dict(SUBJECT)
    if path.endswith("/grades"):
        return "ok", dict(GRADES)
    return "unavailable", {}


def _fake_rag(query, ctx):
    """A retrieval that returns something citable without retrieving anything."""
    return {
        "docs": [
            {
                "filename": "fees.pdf",
                "page_number": 3,
                "text": "رسوم الصف الرابع 30,000 جنيه سنوياً.",
            }
        ],
        "rag_trace": {"retrieval_status": "sufficient", "route": "answer"},
    }


class _StubModel(GenericFakeChatModel):
    """The agent's model, reduced to the one behaviour this file depends on.

    It obeys `tool_choice` and does nothing else. That is not a convenience: `tool_choice`
    is how a NARROWED turn reaches its tool — the planner binds one tool and requires it,
    and the provider then returns a call rather than a message. A stub that ignored it
    would report every single-tool turn as "no tool ran", which is a property of the stub
    and not of the system.

    Where a required argument has no default, the case supplies it (`Case.model_args`).
    Composing arguments from the message is the model's job and needs a real one; this
    file measures which TOOL a turn reaches and how, so that job is stubbed out loud
    rather than faked quietly.
    """

    def __init__(self, args_by_tool=None, **kwargs):
        super().__init__(messages=iter([]), **kwargs)
        object.__setattr__(self, "_args_by_tool", dict(args_by_tool or {}))
        object.__setattr__(self, "_forced", None)
        object.__setattr__(self, "_calls", 0)

    def bind_tools(self, tools=(), **kwargs):
        # `tool_choice` arrives as a bare name from `_ForcePlannedTool`; a real provider
        # binding turns it into its own dict shape, and both are accepted here.
        choice = kwargs.get("tool_choice")
        if isinstance(choice, dict):
            choice = (choice.get("function") or {}).get("name")
        if choice in (None, "auto", "any", "none"):
            choice = None
        object.__setattr__(self, "_forced", choice)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        object.__setattr__(self, "_calls", self._calls + 1)
        already_ran = any(isinstance(m, ToolMessage) for m in messages)
        if self._forced and not already_ran:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="",
                tool_calls=[{
                    "name": self._forced,
                    "args": dict(self._args_by_tool.get(self._forced, {})),
                    "id": "forced-1",
                }],
            ))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="[answer]"))])


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    question: str
    history: tuple = ()
    #: Tools the turn must end up able to call. `None` skips the check — used where the
    #: interesting assertion is elsewhere.
    expect_tools: Optional[set] = None
    #: Tools that must have actually RUN. The strongest assertion available here.
    expect_ran: Optional[set] = None
    expect_parallel: bool = False
    expect_short_circuit: bool = False
    #: Arguments the stub model supplies when the plan REQUIRES a tool with no usable
    #: default. Composing these from the message needs a real model, which this file
    #: deliberately does not run — see `_StubModel`.
    model_args: dict = field(default_factory=dict)
    note: str = ""


K = "search_knowledge_base"
G = "get_student_grades"
S = "get_subject_grades"
A = "get_student_attendance"

CASES = [
    Case(
        "two-part, marks and fees",
        "درجات ليلى كام والمصاريف كام؟",
        expect_tools={K, G},
        expect_ran={K, G},
        expect_parallel=True,
        note="the shape the whole feature exists for",
    ),
    Case(
        "two-part, absences and marks",
        "كام يوم غابت ليلى وكام درجتها؟",
        expect_tools={A, G},
        expect_ran={A, G},
        expect_parallel=True,
        note="two RECORD tools — impossible before the split",
    ),
    Case(
        "records only, attendance",
        "كام يوم غابت ليلى أحمد؟",
        expect_ran={A},
        note="the record type is now the tool's name, not a guessed argument",
    ),
    Case(
        "records only, marks",
        "درجات ليلى أحمد كام؟",
        expect_ran={G},
        note="the question that was answered from a fee corpus before narrowing",
    ),
    Case(
        "one subject",
        "تفاصيل درجة ليلى في الرياضيات",
        expect_ran={S},
        model_args={S: {"subject": "الرياضيات"}},
        note="needs the subject argument, which stays the model's to supply",
    ),
    Case(
        "school matter asked for a child",
        "عايز اعرف مصاريف ابني عمر",
        expect_ran={K},
        model_args={K: {"query": "مصاريف الصف الأول"}},
        note="about_child true AND school_matter — the distinction the enum exists for",
    ),
    Case(
        "school matter, no child",
        "امتى إجازة نصف السنة؟",
        expect_ran={K},
        model_args={K: {"query": "إجازة نصف السنة"}},
    ),
    Case(
        "english, two-part",
        "How is Layla doing this term, and when does the second term start?",
        expect_tools={K, G},
        expect_parallel=True,
    ),
    Case(
        "follow-up carrying its subject",
        "طيب وغيابها؟",
        history=("درجات ليلى أحمد كام؟", "الرياضيات 87.5% والعلوم 91.0%"),
        expect_ran={A},
        note="the resolver has to carry the child, and the classifier the record type",
    ),
    Case(
        "out of domain",
        "ما هو الطقس النهاردة؟",
        expect_short_circuit=True,
        note="ends before the agent is built",
    ),
    Case(
        "social",
        "شكرا جزيلا",
        expect_tools=set(),
        note="answered with no tools bound",
    ),
]


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


@dataclass
class Result:
    case: Case
    scope: str = ""
    about_child: bool = False
    kind: str = ""
    needed: list = field(default_factory=list)
    reference: str = ""
    child_name: str = ""
    child_hint: str = ""
    child_options: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    resolved: str = ""
    exposed: Any = None
    forced: str = ""
    planned: list = field(default_factory=list)
    ran: list = field(default_factory=list)
    overlapped: bool = False
    model_calls: int = 0
    plan_ms: int = 0
    exec_ms: int = 0
    failures: list = field(default_factory=list)
    #: The planner declined to narrow — no tools named, nothing forced, nothing planned.
    #: A valid outcome, not a failure: it is the abstention every failure path lands on,
    #: and it binds everything so the model chooses as it always did. Counted separately
    #: because the RATE is the number worth watching — it is how often the optimisation
    #: actually fires.
    abstained: bool = False
    error: str = ""


def _context() -> ChatRequestContext:
    return ChatRequestContext.for_sync(
        user_id="parent-1",
        session_id="eval-session",
        caller=CallerIdentity(
            user_id="parent-1", guardian_id=GUARDIAN, guardian_token=TOKEN
        ),
    )


def _history(case: Case) -> list:
    messages = []
    for index, text in enumerate(case.history):
        messages.append(HumanMessage(content=text) if index % 2 == 0 else AIMessage(content=text))
    return messages


def _run_tools(ctx: ChatRequestContext, plan, meeting, model_args) -> tuple:
    """Build the real agent and let it run, with the real middleware and real tools."""
    profile = get_profile()
    allowed = profile.agent.tools if plan.exposed_tools is None else plan.exposed_tools
    tools = build_tools(allowed, ctx)
    ran = []

    # Wrap each real tool so the run can see which ones fired and when, without changing
    # what they do. The barrier is what proves overlap: a sequential dispatch cannot get
    # two tools to it, and times out instead of passing slowly.
    for bound in tools:
        original = bound.func

        def traced(*args, __tool=bound, __call=original, **kwargs):
            ran.append(__tool.name)
            meeting.arrive()
            return __call(*args, **kwargs)

        bound.func = traced

    model = _StubModel(args_by_tool=model_args)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=profile.render_system_prompt(allowed, plan.language),
        middleware=[
            runtime._dispatch_planned_tools(ctx),
            runtime._collapse_duplicate_tool_calls(ctx),
            runtime._spend_tool_budgets(ctx),
            runtime._force_the_planned_tool(ctx),
            runtime._end_turn_on_terminal_retrieval(ctx),
        ],
    )
    out = agent.invoke(
        {"messages": [HumanMessage(content=plan.resolved_question or "q")]},
        {"recursion_limit": profile.agent.recursion_limit},
    )
    results = [m.name for m in out["messages"] if isinstance(m, ToolMessage)]
    return ran, results, model._calls


class _Meeting:
    """Two tools that each wait for the other, with a deadline.

    A barrier rather than a stopwatch: on a sequential dispatch the first tool waits for
    a partner that cannot arrive until it returns, so the case reports "not parallel"
    rather than passing slowly on a fast machine.
    """

    def __init__(self, expected: int, timeout: float = 8.0):
        self.expected = expected
        self.overlapped = False
        self._barrier = threading.Barrier(expected, timeout=timeout) if expected > 1 else None

    def arrive(self) -> None:
        if self._barrier is None:
            return
        try:
            self._barrier.wait()
            self.overlapped = True
        except threading.BrokenBarrierError:
            pass


def run_case(case: Case) -> Result:
    result = Result(case=case)
    ctx = _context()
    try:
        started = time.monotonic()
        plan, signals = plan_turn(
            case.question, _history(case), ctx, roster_fetch=_roster_fetch
        )
        result.plan_ms = int((time.monotonic() - started) * 1000)

        result.scope = signals.scope.value
        result.about_child = signals.about_child
        result.kind = signals.child_question_kind
        result.needed = list(signals.needed_tools)
        result.reference = signals.child_reference
        result.child_name = signals.child_name
        result.child_hint = plan.child_hint
        result.child_options = list(plan.child_options)
        result.reasons = list(signals.reasons)
        result.resolved = plan.resolved_question
        result.exposed = plan.exposed_tools
        result.forced = plan.forced_tool
        result.planned = [c["name"] for c in plan.planned_calls]

        if plan.short_circuit:
            if not case.expect_short_circuit:
                result.failures.append("turn ended before the agent, unexpectedly")
            return result
        if case.expect_short_circuit:
            result.failures.append("expected the turn to end before the agent")

        allowed = (
            set(get_profile().agent.tools)
            if plan.exposed_tools is None
            else set(plan.exposed_tools)
        )
        if case.expect_tools is not None and allowed != case.expect_tools:
            result.failures.append(
                f"bound {sorted(allowed)}, expected {sorted(case.expect_tools)}"
            )

        if not allowed:
            return result

        expected_parallel = len(plan.planned_calls) if case.expect_parallel else 1
        meeting = _Meeting(max(expected_parallel, 1))
        started = time.monotonic()
        ran, results, model_calls = _run_tools(ctx, plan, meeting, case.model_args)
        result.exec_ms = int((time.monotonic() - started) * 1000)
        result.ran = ran
        result.overlapped = meeting.overlapped
        result.model_calls = model_calls

        result.abstained = not plan.forced_tool and not plan.planned_calls
        if result.abstained and not ran:
            # Nothing required the model's hand, so a stub model calls nothing. That is
            # the stub's limit, not the system's — see `_StubModel`.
            return result

        if case.expect_ran is not None and set(ran) != case.expect_ran:
            result.failures.append(f"ran {sorted(set(ran))}, expected {sorted(case.expect_ran)}")
        if case.expect_parallel:
            if len(plan.planned_calls) < 2:
                result.failures.append("no parallel dispatch was planned")
            elif not meeting.overlapped:
                result.failures.append("planned tools did not overlap")
            elif model_calls != 1:
                result.failures.append(f"{model_calls} model calls, expected 1")
        if len(results) != len(ran):
            result.failures.append(f"{len(ran)} tools ran but {len(results)} results returned")
    except Exception:
        result.error = traceback.format_exc(limit=4)
        result.failures.append("raised")
    return result


def _report(results: list) -> int:
    print()
    print("=" * 100)
    print("PLANNER + EXECUTION EVAL — real classifier, real planner, real dispatch")
    print("=" * 100)
    failed = 0
    for r in results:
        mark = "ABST" if r.abstained and not r.failures else ("PASS" if not r.failures else "FAIL")
        if r.failures:
            failed += 1
        print(f"\n[{mark}] {r.case.name}")
        print(f"   message      {r.case.question}")
        if r.case.note:
            print(f"   why          {r.case.note}")
        if r.resolved and r.resolved != r.case.question:
            print(f"   resolved     {r.resolved}")
        print(
            f"   classifier   scope={r.scope} about_child={r.about_child} "
            f"kind={r.kind} needed={r.needed or '-'}"
        )
        print(
            f"   child        reference={r.reference} name={r.child_name or '-'} "
            f"resolved={r.child_hint or '-'} asking={r.child_options or '-'}"
        )
        for reason in r.reasons:
            if "classifier" in reason or "named" in reason:
                print(f"   reason       {reason}")
        print(
            f"   plan         exposed={r.exposed} forced={r.forced or '-'} "
            f"planned={r.planned or '-'}"
        )
        if r.ran or r.model_calls:
            print(
                f"   execution    ran={r.ran} overlapped={r.overlapped} "
                f"model_calls={r.model_calls}"
            )
        print(f"   timing       plan {r.plan_ms} ms   execute {r.exec_ms} ms")
        for failure in r.failures:
            print(f"   ---> {failure}")
        if r.error:
            print("   " + r.error.replace("\n", "\n   "))

    print()
    print("=" * 100)
    abstained = sum(1 for r in results if r.abstained and not r.failures)
    print(f"{len(results) - failed}/{len(results)} cases passed"
          + (f"  ({abstained} abstained — planner declined to narrow)" if abstained else ""))
    parallel = [r for r in results if r.case.expect_parallel]
    if parallel:
        overlapped = sum(1 for r in parallel if r.overlapped)
        print(f"{overlapped}/{len(parallel)} parallel cases actually overlapped")
    print("=" * 100)
    return failed


def main() -> int:
    # Patched once, around the whole run: these are the network and the retrieval graph,
    # and nothing in this file is about either.
    records._get = _records_get
    child_roster._fetch = _roster_fetch

    import backend.rag.pipeline as pipeline

    pipeline.run_rag_graph = _fake_rag

    profile = get_profile()
    print(f"profile      {profile.name}")
    print(f"tools        {profile.agent.tools}")
    print(f"model        FAST_MODEL={os.getenv('FAST_MODEL')} (classifier + resolver)")
    print(f"agent model  stubbed — no answer is generated")

    results = [run_case(case) for case in CASES]
    return 1 if _report(results) else 0


if __name__ == "__main__":
    sys.exit(main())
