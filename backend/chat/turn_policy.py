"""What to do with a turn, given what we know about it.

Pure functions over `RequestSignals` and profile config — no model calls, no I/O, no
graph state. The same rule as `backend/rag/policy.py`: when a new decision is needed
it becomes a function here reading the existing signals, rather than a component that
derives its own view and drifts.

## The retrieval question

There are three ways to act on "this question probably isn't in the corpus", and two
of them are the same mistake:

  * **Unbind the knowledge tool.** If the classifier is wrong, the agent cannot search
    and answers from nothing.
  * **Bind it but short-circuit it internally.** If the classifier is wrong, the agent
    searches and gets a canned refusal. Same wrong answer, having also paid for the
    tool schema and a round trip.

Both make a cheap, fallible signal into an unappealable verdict. So neither is used.

What this module does instead is scale the action to the confidence behind it:

  * **HIGH-certainty out-of-domain** — a model read the question and said so. End the
    turn before the agent runs: static reply, no tool schemas, no search.
  * **Anything less** — the knowledge tool stays bound AND fully functional. Weak
    signals become a retrieval *hint* (`candidate_sections`), which reorders what is
    searched first and can never remove a document from reach.

There is deliberately no state in which the agent can call the knowledge tool and be
silently denied an answer. The cheap rung's failures cost latency; only the expensive
rung's failures can cost a refusal, and that is the one that read the question.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.chat.child_resolution import ResolvedChild
from backend.text_matching import name_key
from backend.chat.language import ARABIC, ENGLISH
from backend.chat.signals import RequestSignals, Scope
from backend.rag.evidence import Certainty

logger = logging.getLogger(__name__)

# Only a detector that read the question may end a turn. This floor is the whole
# safety argument of this module; lowering it hands refusals to a dot product.
SHORT_CIRCUIT_MIN_CERTAINTY = Certainty.HIGH


@dataclass
class TurnPlan:
    """The decision for one turn."""

    # When set, the turn is over: reply with this text and run no model.
    static_reply: Optional[str] = None
    # Tool names to bind for this turn. None means "bind everything the profile allows"
    # — the safe default, and what any uncertainty produces.
    exposed_tools: Optional[List[str]] = None
    # A tool this turn must actually CALL, not merely be offered. Set only where the
    # plan narrowed to exactly one tool, and applied only to the turn's first model
    # call — see `_ForcePlannedTool` in backend/chat/runtime.py. Empty is the normal
    # case and means "offer, do not require".
    forced_tool: str = ""
    # Tool calls the planner made ITSELF, to be dispatched before the model's first
    # call rather than waited for. Each is `{"name": str, "args": dict}`.
    #
    # This is the difference between deciding what a turn needs and doing it. `forced_tool`
    # above still leaves the model to compose the call and the graph to run it alone;
    # these are complete calls, handed to the tool node together, which runs them
    # concurrently. Empty is the normal case and means the agent loop behaves exactly as
    # it always has.
    #
    # Never set unless `exposed_tools` names every tool in it — see `_plan_parallel_calls`.
    # A planned call for an unbound tool would reach the graph as a call for a tool that
    # does not exist.
    planned_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Sections to search first. A hint; never a filter.
    retrieval_sections: List[str] = field(default_factory=list)
    # The turn's subject in words that stand on their own, when a resolver produced
    # them. Empty means the message was already standalone or nothing resolved it —
    # every reader falls back to the message itself, which is what happened before.
    resolved_question: str = ""
    # Conditions inherited from earlier turns that still bind this answer. Reaches the
    # retrieval query, the answer prompt, and the resume path, because narrowing the
    # search is only half of it: the right chunks still produce an answer about every
    # year group unless the answer prompt is told which years were asked about.
    carried_constraints: List[str] = field(default_factory=list)
    # Whether this turn's subject came from the conversation rather than its own words.
    # Read by routing: a subject settled one message ago must not be handed back to the
    # user as a choice between subjects.
    is_followup: bool = False
    # Catalogued questions the query resembled, one per section. Routing may offer these
    # as a scope_select when the evidence comes back short — see `decide_route`. Carried
    # on the plan rather than re-derived in the graph because the embedding and the
    # match were already paid for here.
    scope_options: List[str] = field(default_factory=list)
    # Language to answer in, as a detector code. Carried on the plan rather than read
    # from the signals downstream because the agent is built from the plan, and the
    # prompt has to be able to say it.
    language: str = ENGLISH
    # The child this turn is about, as a display label. Empty when the turn is not
    # about a child, or when which child is still an open question.
    child_hint: str = ""
    # The same child as the school's own record number. Travels with the label and never
    # without it, and never reaches a prompt or a trace — it exists so the records tool
    # can read the child the ROSTER matched rather than the name the model typed. See
    # `ChatRequestContext.planned_child_id`.
    child_id: str = ""
    # The child's year group, when the roster reported one. Rendered beside the name so
    # an answer about a general school matter can be given for the right year.
    child_year: str = ""
    # Children to offer when the parent has to be asked which one. Mutually exclusive
    # with `child_hint`, and enforced as such in `_plan_child` rather than in Jinja —
    # a template deciding between them would be policy no test could see.
    child_options: List[str] = field(default_factory=list)

    # Whether this turn is worth a post-turn profile extraction.
    capture_user_info: bool = False
    reasons: List[str] = field(default_factory=list)

    @property
    def short_circuit(self) -> bool:
        return self.static_reply is not None

    def as_trace(self) -> Dict[str, Any]:
        return {
            "turn_short_circuit": self.short_circuit,
            "turn_exposed_tools": list(self.exposed_tools) if self.exposed_tools is not None else None,
            "turn_forced_tool": self.forced_tool or None,
            # The NAMES, never the arguments. This trace is persisted per message and
            # streamed to the browser, and a planned call's arguments carry the child's
            # name and the question itself.
            "turn_planned_calls": [str(call.get("name") or "") for call in self.planned_calls],
            "turn_retrieval_sections": list(self.retrieval_sections),
            "turn_scope_options": list(self.scope_options),
            "turn_resolved_question": self.resolved_question or None,
            "turn_carried_constraints": list(self.carried_constraints),
            "turn_is_followup": self.is_followup,
            "turn_language": self.language,
            "turn_capture_user_info": self.capture_user_info,
            # The resolution, never the name. This trace is persisted per message and
            # streamed to the browser, and a turn may settle on a child silently.
            "turn_child_resolved": bool(self.child_hint),
            "turn_child_asked": bool(self.child_options),
            # Whether the roster's year was applied, never which year it is. This trace
            # is persisted per message and streamed to the browser, and a year group
            # plus a name narrows a child to a handful of real people.
            "turn_child_year_applied": bool(self.child_year),
            "turn_reason": "; ".join(self.reasons) or "n/a",
        }


def localized(copy, language: str) -> str:
    """Pick the copy for a language, falling back rather than failing.

    A missing translation must degrade to the other language, never to an empty
    message — a blank reply is worse than one in the wrong language.
    """
    if copy is None:
        return ""
    if isinstance(copy, str):
        return copy
    preferred = getattr(copy, language, None) or getattr(copy, "get", lambda _k, _d=None: None)(language)
    if preferred:
        return preferred
    for fallback in (ENGLISH, ARABIC):
        alternative = getattr(copy, fallback, None) or getattr(copy, "get", lambda _k, _d=None: None)(fallback)
        if alternative:
            return alternative
    return ""


#: The tools that read one child's own record, and the one that reads the school's
#: material. Named here rather than imported from `backend.tools`, which reaches back
#: into the request context: this module is pure by design and a policy layer that
#: could not be imported without the tool layer would be a circular import waiting for
#: the first person to unit-test a plan. `AgentConfig` spells `search_knowledge_base`
#: out for the same reason, and says so.
#:
#: These must stay in step with `backend.tools.RECORDS_TOOLS`, which is asserted in
#: tests/general/test_planner_parallel_dispatch.py — two spellings of one list is the
#: price of the import boundary, and the test is what stops them drifting.
GRADES_TOOL = "get_student_grades"
SUBJECT_TOOL = "get_subject_grades"
ATTENDANCE_TOOL = "get_student_attendance"
TIMETABLE_TOOL = "get_student_timetable"
CLASS_TOOL = "get_student_class"
SUBJECTS_TOOL = "get_student_subjects"
TEACHERS_TOOL = "get_student_teachers"
SUBJECT_TEACHER_TOOL = "get_subject_teacher"
RECORDS_TOOLS = (
    GRADES_TOOL,
    SUBJECT_TOOL,
    ATTENDANCE_TOOL,
    TIMETABLE_TOOL,
    CLASS_TOOL,
    SUBJECTS_TOOL,
    TEACHERS_TOOL,
    SUBJECT_TEACHER_TOOL,
)
KNOWLEDGE_TOOL = "search_knowledge_base"

#: Which FAMILY of tools each kind of child question needs. `both` is absent on purpose:
#: it means "narrow nothing", which is `exposed_tools = None`, not a list.
#:
#: A family, not a tool. `records` names every record tool because the classifier's
#: enum cannot tell marks from absences from a timetable — that resolution belongs to
#: `needed_tools`. So
#: this map NARROWS (bind these, let the model pick among them) and never PLANS, which
#: is the line `_plan_tools` draws below.
_TOOLS_FOR_KIND = {
    "records": RECORDS_TOOLS,
    "school_matter": (KNOWLEDGE_TOOL,),
}

#: What a `$placeholder` in `agent.planned_tool_arguments` may name, and where its value
#: comes from on the plan.
#:
#: A closed set, and it has to be: that setting is data a domain owner edits, so the
#: alternative is an expression language embedded in YAML — code that no test can see and
#: no schema can validate. Everything here is something the planner has already worked
#: out and is willing to stand behind; nothing reaches into the request context, so a
#: planned call can never carry an identity the model was not supposed to see.
PLAN_PLACEHOLDERS = {
    "$question": lambda plan, question: question,
    "$resolved_question": lambda plan, question: plan.resolved_question or question,
    "$child_label": lambda plan, question: plan.child_hint,
    "$child_id": lambda plan, question: plan.child_id,
    "$child_year": lambda plan, question: plan.child_year,
    "$language": lambda plan, question: plan.language,
}


def _tools_for(config, *, keep: tuple = ()) -> List[str]:
    """The profile's own tools, narrowed to `keep`, in the profile's declaration order.

    Intersected rather than substituted, and that is the whole function. A plan naming a
    tool the profile does not bind would hand `create_agent_for_request` a name
    `build_tools` raises on — so a deployment that ships only the knowledge tool, or a
    profile that has not opted into records at all, gets a plan that narrows to nothing
    and is treated as "no narrowing" by the caller rather than a startup failure.

    Order comes from the profile because the tool list is also the order the system
    prompt describes them in, and two orderings for one list is one that can disagree.
    """
    allowed = list(getattr(config, "tools", None) or [])
    return [name for name in allowed if name in keep]


def resolve_turn(
    signals: RequestSignals,
    *,
    agent_config,
    copy_config,
    rag_config=None,
    child: Optional[ResolvedChild] = None,
) -> TurnPlan:
    """Decide how this turn should run.

    `child` is keyword-only and defaulted, so every existing caller — a test double, an
    integrating deployment — keeps working and simply plans a turn that is about nobody
    in particular.
    """
    plan = TurnPlan()
    # Independent of scope: someone can disclose their phone number in the middle of
    # an off-topic message, and that is still worth keeping.
    plan.capture_user_info = bool(signals.personal_data)
    plan.language = signals.language
    # Carried onto every plan, including the short-circuited ones. A turn that ends
    # without the agent still has to be able to say what it thought was being asked,
    # or an out-of-domain refusal becomes impossible to argue with from the trace.
    plan.resolved_question = signals.resolved_question
    plan.carried_constraints = list(signals.carried_constraints)
    plan.is_followup = signals.followup_intent in ("followup", "correction")

    if signals.is_social:
        return _plan_social(signals, plan, agent_config, copy_config)

    if signals.scope is Scope.OUT_OF_DOMAIN and signals.scope_certainty >= SHORT_CIRCUIT_MIN_CERTAINTY:
        return _plan_out_of_domain(signals, plan, copy_config)

    if signals.scope is Scope.OUT_OF_DOMAIN:
        # Believed, but not by anything that read the question. The tool stays bound
        # and working; only its ordering is nudged.
        plan.reasons.append(
            f"out-of-domain at {signals.scope_certainty.name.lower()} certainty — "
            "knowledge tool stays available"
        )

    _plan_child(
        plan,
        child,
        signals.question,
        getattr(agent_config, "year_reference_markers", ()),
    )
    if plan.child_options:
        return _plan_child_choice(plan, signals, copy_config)

    _plan_tools(plan, signals, agent_config)
    plan.retrieval_sections = list(signals.candidate_sections)
    plan.scope_options = list(signals.scope_options)
    if plan.resolved_question and plan.resolved_question != signals.question:
        plan.reasons.append(f"resolved as {signals.followup_intent}")
    if plan.carried_constraints:
        plan.reasons.append(f"carrying {len(plan.carried_constraints)} condition(s) forward")
    if plan.child_hint:
        plan.reasons.append(f"about one child ({child.source})")
    if plan.child_options:
        plan.reasons.append(f"asking which of {len(plan.child_options)} children")
    if plan.retrieval_sections:
        plan.reasons.append(f"search {len(plan.retrieval_sections)} section(s) first")
    if len(plan.scope_options) > 1:
        plan.reasons.append(f"{len(plan.scope_options)} corpus direction(s) available to offer")
    if not plan.reasons:
        plan.reasons.append("no signal changed the default plan")
    return plan


def question_names_a_year(question: str, markers) -> bool:
    """Whether the question scoped itself to a year group.

    Pure and literal: a folded substring test against a configured vocabulary, no model
    and no inference. It gates one thing — whether the roster's year for this child is
    applied as a condition — and it is deliberately the permissive side of that gate. A
    marker it fails to recognise leaves the year applied, which is the behaviour that
    already exists; a marker it recognises only withholds a narrowing. Neither direction
    can produce an answer scoped to a year nobody asked about.

    Folded through `name_key` for the same reason the roster matcher is: a parent typing
    «الصف» with a different alif form means the same word, and a gate defeated by
    orthography would be no gate at all.
    """
    folded = name_key(question or "")
    if not folded:
        return False
    return any(name_key(marker) in folded for marker in (markers or []) if marker)


def _plan_child(
    plan: TurnPlan,
    child: Optional[ResolvedChild],
    question: str = "",
    year_markers=(),
) -> None:
    """Put the resolved child on the plan — as a name, or as a question, never both.

    The mutual exclusion lives here because it is a decision. Expressed in the template
    it would become `{% if hint and not options %}`, which is policy that no test covers
    and no type checker sees — the thing `backend/prompts/__init__.py` forbids in as
    many words.

    A turn that is not about a child sets neither, and `_turn_context_message` then
    renders nothing at all, so most turns pay nothing for this feature.

    The year is withheld — while the NAME is still set — when the question named a year
    of its own. Those two travel together everywhere else, and separating them here is
    the point: «مصاريف ابني في الصف الرابع» is still about this child, so the answer
    should be about them, but the year to answer for is the one the parent said and not
    the one on file.
    """
    if child is None:
        return
    if child.resolved:
        plan.child_hint = child.label
        plan.child_id = child.student_id
        if not question_names_a_year(question, year_markers):
            plan.child_year = child.year_level
        elif child.year_level:
            plan.reasons.append("question names its own year — roster year not applied")
        return
    if child.ask:
        plan.child_options = list(child.option_labels)


def _plan_child_choice(plan: TurnPlan, signals, copy_config) -> TurnPlan:
    """End the turn with the question, rather than asking a model to ask it.

    ## Why this is not a prompt

    Everything needed to ask it is already known, and none of it came from a model. The
    roster is the school's own list of this parent's children, fetched under their token.
    Narrowing it by the sex the message stated is a filter over that list. Whether one
    child is left or several is a length check. There is no judgement anywhere in that
    chain, so routing it through a 20B model can only subtract: the previous arrangement
    handed the model the candidate names and four sentences of instruction — ask by name,
    offer exactly these, do not look anything up, do not guess, do not show figures — and
    every one of those is a rule that holds only as often as the model obeys it.

    Asked as a real clarification rather than as prose, so the answer comes back as a
    selected option the next turn resolves against the roster by id, and the client can
    render the children as buttons instead of asking a parent to re-type an Arabic name
    the assistant already knows how to spell.

    Falls through to the agent when a deployment has configured no copy — the same rule
    `_plan_out_of_domain` follows, and for the same reason: refusing with an empty string
    is worse than answering.
    """
    plan.static_reply = localized(getattr(copy_config, "which_child", None), signals.language)
    plan.exposed_tools = []
    if not plan.static_reply:
        plan.static_reply = None
        plan.exposed_tools = None
        plan.reasons.append("more than one child matches, but no copy configured")
        return plan
    plan.reasons.append(f"asking which of {len(plan.child_options)} children, deterministically")
    return plan


def _plan_tools(plan: TurnPlan, signals: RequestSignals, agent_config) -> None:
    """Bind the tools the turn needs, when something already knows which those are.

    Three outcomes, and which one applies is decided by how many tools are needed rather
    than by a separate switch:

      * **None known** — bind everything. The default, and where every failure lands.
      * **Exactly one** — bind it and require it. Binding removes the wrong choice;
        requiring removes the remaining one, which is answering from memory.
      * **More than one** — bind them and, where the profile has said what their calls
        look like, WRITE those calls so the graph can run them together. `tool_choice`
        names one function and cannot express "each of these", so a set can only be
        required by dispatching it.

    ## Why this is not the mistake the module docstring forbids

    That docstring rules out unbinding the knowledge tool on a scope guess, because a
    wrong guess then produces a refusal the user cannot appeal. This narrowing is a
    different transaction and it is worth being explicit about how:

      * It runs only when a child has ALREADY been resolved against the school's own
        roster — a pure function over a list of real children, not a similarity score.
      * The direction is chosen by a field whose every failure mode is `both`: a
        classifier that abstained, rate-limited, or answered outside the closed set
        leaves `child_question_kind` at its default and nothing here fires. Narrowing
        needs a positive answer; it is never the residue of a missing one.
      * The cost of being wrong is bounded and visible. A records turn that should have
        searched gets the records tool's own careful "no record of that" wording, and the
        parent's next message re-plans from scratch with everything bound. Nothing is
        cached, nothing is pinned, and no refusal is manufactured here.

    What it buys is the failure that was actually measured: with both tools bound, a 20B
    model asked «درجات ليلى أحمد كام؟» searched the knowledge base, found no marks in a
    fee corpus, and told a parent no information about their daughter existed. A tool
    that is not bound cannot be chosen by mistake.

    Off unless the profile asks for it, so every deployment that has not measured this
    keeps today's behaviour exactly.
    """
    wanted, why, plannable = _needed_tools(plan, signals, agent_config)
    if not wanted:
        return
    narrowed = _tools_for(agent_config, keep=wanted)
    if not narrowed:
        # The profile does not bind the tool this kind of question needs. Say nothing
        # and change nothing: a plan naming a tool the profile refuses is a startup
        # error in `build_tools`, and an empty list would read as "bind no tools at all".
        plan.reasons.append(f"{why}, but the profile binds no such tool")
        return
    plan.exposed_tools = narrowed
    if len(narrowed) == 1:
        # One tool bound and one tool required are the same decision seen twice. Binding
        # removes the wrong choice; requiring removes the remaining one — answering from
        # memory without calling anything, which is the other half of the measured
        # failure and the half narrowing alone does not touch.
        plan.forced_tool = narrowed[0]
    elif plannable:
        # Several tools, and something named each of them: the turn needs them all.
        # `tool_choice` cannot express that — it names one function — so the only way to
        # require a SET is to write the calls and dispatch them, which is what this does.
        _plan_parallel_calls(plan, narrowed, agent_config, signals.question)
    plan.reasons.append(f"{why} — bound {', '.join(narrowed)} only")


def _needed_tools(plan: TurnPlan, signals: RequestSignals, agent_config) -> tuple:
    """Which tools this turn needs. Returns `(tools, why, plannable)`; `()` is everything.

    ## Narrowing and planning are different claims

    `plannable` is the whole reason this returns three things. Both sources below say
    which tools a turn needs, but they do not say it with the same confidence, and the
    difference decides whether the planner may CALL them or only BIND them:

      1. **The classifier's own list** (`needed_tools`, from `agent.tool_selection`).
         Tool by tool, by name, having read the message against a description of each.
         That is specific enough to act on, so it is `plannable` — every tool in it is
         one the turn is asserted to need, and dispatching them together is exactly
         what the assertion means.

      2. **`child_question_kind`.** A three-valued enum over two FAMILIES. `records`
         names all three record tools because the enum cannot tell marks from absences
         — so it is a statement about where to look, not about what to call. Binding
         those three and letting the model choose is right; dispatching all three would
         read a child's attendance because they asked about maths.

    Empty means bind everything, and every failure lands there: no catalogue, an
    abstaining classifier, an unresolved child, a kind outside the closed set. Narrowing
    is an optimisation, and an optimisation may never be the reason a capability is out
    of reach.
    """
    # 1. The classifier read the message against this deployment's own tool catalogue.
    # Trusted without the child gate below, and deliberately: that gate exists because
    # `child_question_kind` is only meaningful once a real child has been resolved, while
    # this list is about the MESSAGE and is just as meaningful on a deployment that has
    # no children at all.
    if signals.needed_tools:
        named = tuple(signals.needed_tools)
        # ## When the classifier's two answers contradict each other
        #
        # The same call reports a FAMILY (`child_question_kind`) and a tool LIST
        # (`needed_tools`). They are independent readings of one message, so they can
        # disagree — and measured on gpt-oss-20b they do, on about one turn in four for a
        # question phrased in dialect: «ليلى أحمد جابت كام؟» is `records` every time and
        # still named `search_knowledge_base` twice in eight runs.
        #
        # That is the exact failure this whole mechanism exists to stop: a question about
        # a named child's marks, sent to a fee corpus, answered with "no information about
        # your daughter". Taking the tool list on its own would reintroduce it, and taking
        # the intersection would leave nothing.
        #
        # So a contradiction is treated as what it is — the classifier not having settled —
        # and this falls through to the family below, which is the coarser, measured, and
        # safer of the two readings. Only a genuine contradiction: `both` names no family,
        # so a message spanning both sides is not caught by this.
        family = _TOOLS_FOR_KIND.get(signals.child_question_kind)
        if family and not set(named) & set(family):
            logger.info(
                "classifier disagreed with itself (%s vs %s); using the family",
                signals.child_question_kind, ", ".join(named),
            )
        else:
            return named, f"needs {', '.join(named)}", True

    if not getattr(agent_config, "narrow_tools_to_the_turn", False):
        return (), "", False
    # An open question about WHICH child is not a settled turn. The agent has to be able
    # to ask, and if the parent's next message answers with a name the turn may still go
    # either way — so it keeps everything.
    if plan.child_options or not plan.child_hint:
        return (), "", False

    # 2. One family, chosen by a field whose every failure mode is `both` — which is
    # absent from the map, so it narrows nothing.
    kind = signals.child_question_kind
    wanted = _TOOLS_FOR_KIND.get(kind)
    if not wanted:
        return (), "", False
    return wanted, f"{kind} question about one child", False


def _plan_parallel_calls(
    plan: TurnPlan, tools: List[str], agent_config, question: str
) -> None:
    """Write the calls for `tools`, so they can be dispatched together instead of found.

    Nothing here is a guess about what the tools will RETURN — it is only a statement of
    what to ask them, built from things the planner already settled and the profile
    already declared. The whole judgement was made upstream, in deciding that this turn
    needs these tools.

    A tool is dropped from the plan, rather than called blank, when the profile declares
    no arguments for it or when every argument it declared resolved to nothing. Both mean
    the same thing: this deployment has not said what a planned call to that tool looks
    like, so the model should make it the ordinary way. That is what lets a deployment
    adopt this one tool at a time — an undeclared tool stays bound and behaves exactly as
    it did before.

    Silent no-op unless the profile asked for it, and a no-op again if fewer than two
    calls survive: dispatching a single call ahead of the model buys none of the
    concurrency this exists for, while still spending the planner's credibility on it.
    """
    if not getattr(agent_config, "parallel_tool_calls", False):
        return
    templates = dict(getattr(agent_config, "planned_tool_arguments", None) or {})
    if not templates:
        return

    calls = []
    for name in tools:
        declared = templates.get(name)
        if not declared:
            continue
        args = {}
        for key, value in declared.items():
            resolved = _resolve_argument(value, plan, question)
            # An empty argument is dropped rather than sent. A search for "" is a wasted
            # retrieval, and an empty student name means "the parent did not say which",
            # which the records tool already handles better from its own default than
            # from an explicit blank.
            if resolved:
                args[key] = resolved
        if args:
            calls.append({"name": name, "args": args})

    if len(calls) < 2:
        return
    plan.planned_calls = calls
    plan.reasons.append(f"dispatching {len(calls)} tools together")


def _resolve_argument(value: str, plan: TurnPlan, question: str) -> str:
    """One declared argument, with a `$placeholder` swapped for what the planner knows.

    Whole-value only — `"$resolved_question"` is a placeholder, `"fees for $child_label"`
    is the literal string it looks like. Substring interpolation would make this a
    template language, and the argument against that is in `PLAN_PLACEHOLDERS`.

    An unrecognised `$name` resolves to nothing, which drops the argument. That is the
    right way for a typo in a profile to fail: quietly, into the behaviour that existed
    before, rather than by sending the literal text `$child_labell` to a tool as a
    child's name.
    """
    text = str(value or "")
    if not text.startswith("$"):
        return text
    reader = PLAN_PLACEHOLDERS.get(text)
    if reader is None:
        logger.warning("unknown placeholder %r in planned_tool_arguments", text)
        return ""
    return str(reader(plan, question) or "")


def _plan_social(signals, plan: TurnPlan, agent_config, copy_config) -> TurnPlan:
    """A pleasantry. Cheap either way, but not a refusal.

    Default is still a model call — with every tool unbound, so it costs the lean
    prompt and nothing else. That keeps replies varied; an identical canned string for
    every greeting reads as a script, and greetings are the first thing a user sees.
    Deployments that would rather pay nothing can switch to static.
    """
    plan.exposed_tools = []
    if getattr(agent_config, "social_reply_mode", "model") == "static":
        plan.static_reply = localized(getattr(copy_config, "social", None), signals.language)
        if plan.static_reply:
            plan.reasons.append("social phrase, static reply")
            return plan
        plan.reasons.append("social phrase, static reply configured but no copy — using the model")
    else:
        plan.reasons.append("social phrase, model reply with no tools bound")
    return plan


def _plan_out_of_domain(signals, plan: TurnPlan, copy_config) -> TurnPlan:
    plan.static_reply = localized(getattr(copy_config, "out_of_domain", None), signals.language)
    plan.exposed_tools = []
    if not plan.static_reply:
        # Refusing with an empty string is worse than answering. Fall through to the
        # agent and let it explain itself.
        plan.static_reply = None
        plan.exposed_tools = None
        plan.reasons.append("out of domain but no copy configured — falling through to the agent")
        return plan
    plan.reasons.append("out of domain, confirmed by a classifier that read the question")
    return plan
