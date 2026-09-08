import logging
import os
from typing import Any, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    after_model,
    before_model,
)
from langchain_core.messages import AIMessage
from typing_extensions import NotRequired
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

from backend.chat.request_context import ChatRequestContext
from backend.llm import sampling
from backend.provider_compat import fold_tool_results_into_text
from backend.profiles import get_profile
from backend.tools import KNOWLEDGE_TOOL, build_tools

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
FAST_MODEL = os.getenv("FAST_MODEL")
BASE_URL = os.getenv("BASE_URL")

logger = logging.getLogger(__name__)

# The agent's model is the one that replays tool results, and this endpoint stops
# parsing its own transcript format the moment it sees one. See backend/provider_compat.py.
model = fold_tool_results_into_text(init_chat_model(
    model=MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    stream_usage=True,
    **sampling("answer"),
))

fast_model = init_chat_model(
    model=FAST_MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    stream_usage=True,
    **sampling("fast"),
)


# Tool results whose outcome is already the final answer. The model adds nothing to
# these but a rewording of profile copy, and it is charged for the whole conversation
# plus the tool result to do it.
TERMINAL_STATUSES = {"no_knowledge", "retrieval_error"}


def terminal_status(ctx) -> Optional[str]:
    stored = ctx.peek_rag_trace()
    trace = (stored or {}).get("rag_trace") or {}
    status = trace.get("retrieval_status")
    return status if status in TERMINAL_STATUSES else None


def _other_tools_ran(messages) -> bool:
    """Whether any tool BUT the knowledge tool put material in front of the model.

    A result carrying no name is read as the knowledge tool's. That is the conservative
    reading here rather than the permissive one: this guard's job is to stop the model
    answering when the corpus came back empty, so an unidentifiable result must not be
    what excuses it from stopping. It also preserves the behaviour of every existing
    caller, which is what a middleware fix should cost.
    """
    return any(
        isinstance(message, ToolMessage)
        and (getattr(message, "name", "") or KNOWLEDGE_TOOL) != KNOWLEDGE_TOOL
        for message in messages
    )


def dedupe_tool_calls(calls) -> list:
    """The same list with repeated calls removed, first occurrence kept.

    Two calls are the same when they name the same tool with the same arguments. Only
    exact equality: "مصاريف ابني كام" and "مصاريف ابني" are different searches and might
    retrieve different chunks, so deciding they mean the same thing would need a model,
    and a model is not something this can afford to be — the call budget is what handles
    the near-duplicates.

    Arguments are compared through sorted items rather than `==` so that two dicts built
    in a different key order still compare equal, which is how they arrive when the
    provider streams the same call several times.
    """
    seen = set()
    kept = []
    for call in calls or []:
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        try:
            fingerprint = (name, repr(sorted((args or {}).items())))
        except TypeError:  # unorderable keys — treat as unique rather than guessing
            kept.append(call)
            continue
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(call)
    return kept


class ToolBudgetState(AgentState):
    """The agent's own record of what it has already called this turn.

    In graph state rather than on the request context because it is a fact about the
    conversation: it is checkpointed with the messages it describes, it survives a
    resume, and `wrap_model_call` can read it at the moment it decides what to offer.
    The context's own counters stay where they are — they guard the tools themselves,
    which is a different job from deciding what the model is allowed to ask for.
    """

    tool_calls_made: NotRequired[dict]


def _tool_name(tool) -> str:
    """The name of a bound tool, whichever shape the request carries it in."""
    name = getattr(tool, "name", None)
    if name:
        return str(name)
    if isinstance(tool, dict):
        return str((tool.get("function") or {}).get("name") or tool.get("name") or "")
    return ""


def budget_for(name: str) -> int:
    """How many times `name` may be called in one turn.

    Retrieval falls back to `max_knowledge_calls_per_turn` rather than carrying its own
    number, because two settings for one budget is two settings that can disagree — and
    when they disagree the lower one refuses, which is the failure this exists to stop.
    """
    agent = get_profile().agent
    budgets = agent.tool_call_budgets or {}
    if name in budgets:
        return int(budgets[name])
    if name == KNOWLEDGE_TOOL:
        return int(agent.max_knowledge_calls_per_turn)
    return int(agent.default_tool_call_budget)


class _ToolBudget(AgentMiddleware[ToolBudgetState, Any, Any]):
    """Stop offering a tool once this turn has used it up.

    The tools already refuse a call past their ceiling, and the refusal is what does the
    damage: it arrives as a tool result, the model reads it as a failure, and it tries
    again in different words. Measured on one question whose search returned nothing,
    that produced 29 consecutive `search_knowledge_base` requests, and an earlier
    arrangement produced 85 before the recursion limit ended the turn. Requests, not
    executions: the duplicate-collapsing middleware above and the retrieval memo absorb
    many of them, so the true cost is lower than those numbers and the wasted model
    round-trips are not.

    Withholding is quieter than refusing. A tool that is not in the request is not a
    failure the model has to reason about; it is simply not an option, so the turn ends
    in an answer instead of another attempt. Nothing is said to the model about limits,
    which is deliberate: `TOOL_CALL_LIMIT_REACHED` is exactly the kind of sentence it
    narrates around, and parents have seen it do so.

    Counting happens in `after_model`, against the message the model just produced, so a
    call that the duplicate-collapsing middleware removes is never counted — the two are
    ordered together in `create_agent_for_request` for that reason.
    """

    # Declared BOTH ways on purpose. The class attribute is what `create_agent` reads to
    # widen the compiled state schema, and the generic parameter is what types the
    # `state` handed to the hooks. Setting only the attribute compiles a graph whose
    # state has no `tool_calls_made` key, so every update to it is silently discarded and
    # the budget reads zero forever — which looks exactly like a middleware that is not
    # running at all.
    state_schema = ToolBudgetState  # type: ignore[assignment]

    # BOTH the sync and the async hook, on every one of these middlewares, and the
    # decision lives in a plain method that neither of them duplicates.
    #
    # This is not defensive symmetry, it is the production path. Every streamed turn goes
    # through `astream`, which composes the ASYNC hooks: `awrap_model_call` on the base
    # class raises `NotImplementedError` outright, and `aafter_model` is a silent no-op.
    # A middleware that implemented only the sync side therefore took the whole streamed
    # chat down on its first model call, and — had it survived that — would have counted
    # no tool calls at all, so the budget below would have read zero forever. The sync
    # path (`invoke`) is the one the unit tests exercise and very nearly the only place
    # it worked.
    def _count(self, state) -> dict | None:
        messages = state.get("messages") or []
        last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last is None or not getattr(last, "tool_calls", None):
            return None
        made = dict(state.get("tool_calls_made") or {})
        for call in last.tool_calls:
            name = call.get("name")
            if name:
                made[name] = made.get(name, 0) + 1
        return {"tool_calls_made": made}

    def _affordable(self, request):
        """The request with any tool this turn has used up removed from it."""
        made = (getattr(request, "state", None) or {}).get("tool_calls_made") or {}
        if not made or not request.tools:
            return request
        affordable = [
            tool for tool in request.tools
            if made.get(_tool_name(tool), 0) < budget_for(_tool_name(tool))
        ]
        if len(affordable) == len(request.tools):
            return request
        withheld = sorted({_tool_name(t) for t in request.tools}
                          - {_tool_name(t) for t in affordable})
        logger.info("tool budget spent, not offering: %s", ", ".join(withheld))
        overrides = {"tools": affordable}
        if not affordable:
            # A request that forces a tool call and offers none is rejected by the
            # provider before the model ever sees it.
            overrides["tool_choice"] = None
        return request.override(**overrides)

    def after_model(self, state, runtime):
        return self._count(state)

    async def aafter_model(self, state, runtime):
        return self._count(state)

    def wrap_model_call(self, request, handler):
        return handler(self._affordable(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._affordable(request))


def _spend_tool_budgets(ctx: ChatRequestContext) -> _ToolBudget:
    """One budget middleware for this turn. `ctx` is unused today and kept for symmetry
    with the other middleware factories, which all take it."""
    return _ToolBudget()


def _nothing_has_run_yet(state) -> bool:
    """Whether no tool has run yet this turn, given the graph state itself.

    A `ToolMessage` exists only inside the agent loop — the conversation loaded from
    storage holds the text of past turns, never their tool traffic — so its absence is
    exactly "no tool has returned yet in THIS turn". `tool_calls_made` is consulted as
    well where the budget middleware put it there, which catches the two cases the
    messages cannot: a tool that was requested and produced no result message, and the
    planner's own dispatch, which writes that counter as it seeds.
    """
    state = state or {}
    if state.get("tool_calls_made"):
        return False
    return not any(isinstance(m, ToolMessage) for m in (state.get("messages") or []))


def _first_model_call(request) -> bool:
    """`_nothing_has_run_yet` for a caller holding a model request rather than a state.

    Two entry points to one predicate, and one definition: the forcing middleware reads a
    request and the dispatch middleware reads a state, and they must agree about what
    "first" means or the planner's seeded results would leave `tool_choice` still set on
    a call the model has already been given the answers for.
    """
    return _nothing_has_run_yet(getattr(request, "state", None) or {})


class _ForcePlannedTool(AgentMiddleware):
    """Call the tool the planner chose, rather than offering it and hoping.

    Narrowing the tool list removes the wrong choice; it does not remove the choice of
    making none. Measured on the same three questions: handed only `get_student_records`,
    `openai/gpt-oss-20b` still answered «درجات ليلى أحمد كام؟» from memory on some runs —
    a fluent paragraph about a child whose record it had not read. `tool_choice` closes
    that, because the provider will not return a message without the call.

    ## Only the first call, and that is not a compromise

    Measured against this endpoint: gpt-oss ignores `tool_choice` once a tool result is in
    the history. Forcing every pass would therefore buy nothing on later passes and cost a
    turn that cannot end — a model told it must call a tool, on a request whose tool it
    has already spent, has no legal move. The first call is where the tool selection is
    actually decided, so constraining it is the whole of the win.

    ## Composed INSIDE the budget

    `create_agent_for_request` lists the budget first, and `wrap_model_call` composes
    first-in-list as the OUTERMOST layer, so the request reaching this middleware has
    already had spent tools removed from it. That ordering is what makes the guard below
    — force only a tool still on the request — sufficient to prevent the one shape the
    provider rejects outright: a request that requires a tool call and offers no tools.
    """

    def __init__(self, tool_name: str) -> None:
        super().__init__()
        self._tool = (tool_name or "").strip()

    def _required(self, request):
        """The request with this turn's tool required, when requiring it is safe."""
        if not self._tool:
            return request
        # Somebody with a better claim already decided — structured output binds `any`,
        # and the budget clears it to None when it has withheld everything. Neither is a
        # decision to overrule.
        if getattr(request, "tool_choice", None) is not None:
            return request
        if not _first_model_call(request):
            return request
        if self._tool not in {_tool_name(tool) for tool in (request.tools or [])}:
            return request
        logger.info("planner requires %s on this turn's first call", self._tool)
        # The bare name. `ChatOpenAI.bind_tools` turns a name it recognises among the
        # bound tools into the provider's `{"type": "function", ...}` shape itself, and
        # raises on one it does not — which the guard above has already made impossible.
        return request.override(tool_choice=self._tool)

    # Sync and async both, for the reason spelled out on `_ToolBudget`: the streamed path
    # composes the async hooks, and the base class's `awrap_model_call` raises. A forcing
    # middleware that existed only on the sync side would have forced nothing on the one
    # path every parent actually uses.
    def wrap_model_call(self, request, handler):
        return handler(self._required(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._required(request))


def _force_the_planned_tool(ctx: ChatRequestContext) -> _ForcePlannedTool:
    """Require this turn's chosen tool, if the planner chose one.

    Read off the context rather than passed down through `create_agent_for_request`,
    which already takes a tool LIST and would then be taking two overlapping answers to
    the same question. The planner writes it in `_hand_to_graph` before the agent is
    built, alongside the retrieval hints it has always written there.
    """
    return _ForcePlannedTool(getattr(ctx, "forced_tool", "") or "")


def planned_tool_calls(planned, already_made: dict) -> list:
    """The planner's calls, in the shape the graph dispatches, with the unaffordable cut.

    Pure and separate from the middleware so the arithmetic can be tested without a
    graph. Three things happen here, in this order, and the order matters:

      1. **Shape.** A plan carries `{"name", "args"}`; a tool call needs an `id` as well,
         because the result message is matched back to its call by that id. Malformed
         entries are dropped HERE, before anything else touches them:
         `dedupe_tool_calls` fingerprints a call by sorting its arguments, so an entry
         whose `args` is not a dict raises there rather than being skipped.
      2. **Deduplicate.** Through the same function the model's own calls go through, so
         a profile that names one tool twice is collapsed by the rule that already exists
         rather than by a second one that could disagree with it.
      3. **Afford.** A call is dropped once its tool's budget for the turn is spent. The
         budget is the deployment's statement of how many times a tool may run, and the
         planner is not exempt from it — a planner that could overspend it would be a
         second, invisible budget.

    Ids are positional rather than random so the same plan produces the same transcript
    twice. They only have to be unique within the message they travel in.
    """
    well_formed = [
        {"name": str((call or {}).get("name") or ""), "args": (call or {}).get("args")}
        for call in (planned or [])
    ]
    well_formed = [
        call for call in well_formed
        if call["name"] and isinstance(call["args"], dict)
    ]

    calls = []
    made = dict(already_made or {})
    for index, call in enumerate(dedupe_tool_calls(well_formed)):
        name = call["name"]
        if made.get(name, 0) >= budget_for(name):
            logger.info("planned call to %s is past its budget for the turn", name)
            continue
        made[name] = made.get(name, 0) + 1
        calls.append({"name": name, "args": dict(call["args"]), "id": f"plan_{index}_{name}"})
    return calls


def _dispatch_planned_tools(ctx: ChatRequestContext):
    """Run the tools the planner chose, together, before the model is asked anything.

    ## What this replaces

    Left to itself the agent discovers its tools one at a time: it asks for one, waits
    for the result, reads it, asks for the next. A question needing three tools costs
    four model round-trips, and the tools run strictly in sequence because a model emits
    its calls one message at a time. On a turn where the planner already knows all three
    are needed, every one of those waits is bought with nothing.

    Seeding the calls collapses that to one model call. The assistant message the model
    would eventually have written is written by the planner instead, and the graph's own
    tool node dispatches it — one `Send` per call, so the tools genuinely overlap. The
    model wakes up with every result already in front of it and writes the answer.

    ## Why the graph runs them and not this function

    This hook has no async half, so LangGraph runs it in a worker thread; calling the
    tools here would run them one after another in that single thread and deliver none of
    the concurrency this exists for. Handing the calls back and jumping to `tools` puts
    them where the fan-out already lives. That is also why the seeded message is a real
    `AIMessage` rather than anything bespoke: everything downstream — the budget counter,
    the duplicate collapser, the transcript fold in `provider_compat`, the checkpointer —
    already knows how to read one.

    ## The planner is not given the last word

    A wrong plan is worth more here than a wrong tool choice is in the ordinary loop, so
    three things deliberately limit it:

      * **The tools stay bound.** The model can still call anything the planner chose,
        because `exposed_tools` is the same list.
      * **Their budgets are only partly spent.** What the planner dispatched is counted
        (see `planned_tool_calls`), so a tool with budget left can be called again with
        different arguments — which is how the model corrects a planned query that came
        back empty.
      * **No surviving call means no dispatch at all.** A lone surviving call IS
        dispatched, which it did not used to be. The old rule weighed concurrency only —
        one call overlaps with nothing — but the round-trip it removes is the same
        round-trip either way, and the single-tool turn is this deployment's common
        case. What the planner spends its credibility on is the ARGUMENTS, and those are
        written from what it already settled, however many calls there are.

    ## Ordering against the other hook that jumps

    Listed FIRST in `create_agent_for_request`, ahead of `_end_turn_on_terminal_retrieval`,
    and `before_model` hooks are chained in list order with a jump short-circuiting the
    rest. The two can never contend for the same pass: this one fires only while nothing
    has run, and that one only once retrieval has returned a verdict. The ordering is
    stated anyway, because "they cannot both fire" is a property of today's conditions
    and list order is what would decide it if that ever stopped being true.
    """

    @before_model(state_schema=ToolBudgetState, can_jump_to=["tools"])
    def _run_the_planned_calls_together(state, runtime):
        planned = list(getattr(ctx, "planned_calls", None) or [])
        if not planned:
            return None
        # Once anything has run, the model is mid-conversation with its own tools and
        # the plan is spent. This is the same predicate the forcing middleware uses, so
        # the seeded results also stand `tool_choice` down — the model must be free to
        # answer from what it has rather than be required to call something again.
        if not _nothing_has_run_yet(state):
            return None

        made = dict(state.get("tool_calls_made") or {})
        calls = planned_tool_calls(planned, made)
        if not calls:
            return None

        for call in calls:
            made[call["name"]] = made.get(call["name"], 0) + 1
        logger.info(
            "planner dispatching %d tools together: %s",
            len(calls),
            ", ".join(call["name"] for call in calls),
        )
        try:
            ctx.note_planned_dispatch(len(calls))
        except Exception:  # pragma: no cover - accounting must never break a turn
            logger.debug("could not record the planned dispatch", exc_info=True)
        return {
            # Empty content, deliberately. Anything written here is text the model did
            # not write, sitting in its own voice in its own transcript — and on the
            # harmony path `provider_compat` replaces this message with its own
            # description of the calls anyway.
            "messages": [AIMessage(content="", tool_calls=calls)],
            # Counted HERE and not in `after_model`, which is where the budget middleware
            # counts and which this jump skips entirely. Without this the planner's calls
            # would be free, and the model could then spend every tool's full budget over
            # again on results it already has.
            "tool_calls_made": made,
            "jump_to": "tools",
        }

    return _run_the_planned_calls_together


def _collapse_duplicate_tool_calls(ctx: ChatRequestContext):
    """Run each distinct tool call once, however many times the model asked for it.

    `openai/gpt-oss-20b` emits the same `search_knowledge_base` call two to five times in
    a single assistant message — measured on 4 of 6 turns against the same question. Each
    copy is a full retrieval: embedding, hybrid recall, reranking and an LLM grading
    call, all to produce the answer already in hand. Nothing downstream wanted them.

    Left alone they did three kinds of damage, in rising order of how much they cost:

      * They spent the turn's retrieval budget. `max_knowledge_calls_per_turn` is meant
        to buy two DIFFERENT searches; duplicates spent it on one, so a genuine
        follow-up query was refused.
      * They returned `TOOL_CALL_LIMIT_REACHED` to the model, which reads it as a
        failure and narrates around it — "the tool call failed, we need to call
        correctly" — which is text a parent then saw.
      * They disengaged the terminal-retrieval guard above, which is how a question the
        corpus could not answer was answered from the model's own knowledge instead.

    Collapsed here rather than absorbed in the tool because the duplicates should not
    become tool results at all: rewriting the assistant message drops them before the
    graph dispatches, so there is one call, one result, and no orphaned `tool_call_id`
    for the provider to reject on the next request.
    """

    @after_model
    def _drop_repeated_calls(state, runtime):
        messages = state.get("messages") or []
        if not messages:
            return None
        message = messages[-1]
        calls = list(getattr(message, "tool_calls", None) or [])
        if len(calls) < 2:
            return None
        kept = dedupe_tool_calls(calls)
        if len(kept) == len(calls):
            return None
        try:
            ctx.note_duplicate_tool_calls(len(calls) - len(kept))
        except Exception:  # pragma: no cover - accounting must never break a turn
            logger.debug("could not record duplicate tool calls", exc_info=True)
        logger.info(
            "collapsed %d duplicate tool call(s) into %d", len(calls) - len(kept), len(kept)
        )
        # Same id, so LangGraph's reducer replaces the message rather than appending a
        # second copy of it.
        return {"messages": [message.model_copy(update={"tool_calls": kept})]}

    return _drop_repeated_calls


def _end_turn_on_terminal_retrieval(ctx: ChatRequestContext):
    """Stop the loop before the model is asked to paraphrase profile copy.

    The saving is the second model call: once retrieval has concluded there is no
    knowledge, the reply is a string already in hand, and sending the conversation plus
    the tool result to have it reworded buys nothing.

    It has to be the graph's own decision. Breaking out of `astream` from the consumer
    side leaves the generator suspended — Python does not finalize an async generator
    at `break`, so `aclose()` lands later at GC and throws GeneratorExit into LangGraph
    at its `yield`, where `except BaseException` reports the run as failed. Ending here
    lets the stream finish on its own, so the turn is traced as the success it is.
    """

    @before_model(can_jump_to=["end"])
    def _stop_after_a_terminal_tool_result(state, runtime):
        status = terminal_status(ctx)
        if not status:
            return None
        # Only when the knowledge tool is the sole tool that ran. On a turn that also
        # called, say, a tool with nothing to cite, the model still has material to answer from
        # and cutting it here would throw that away.
        #
        # Counted by TOOL, not by message. Counting messages made the guard fail exactly
        # when it was needed most: this model emits the same search_knowledge_base call
        # two to five times in one message (measured on 4 of 6 turns), so a turn whose
        # corpus said `no_knowledge` arrived here with two or three results, `!= 1` was
        # true, and the guard stood down — leaving the model to answer a fees question
        # from its own knowledge. That is how an invented figure reached a parent.
        # Duplicate calls to the SAME tool are one tool having run, and its verdict is
        # already in `status`.
        #
        # A PLANNED turn reaches here with both results at once, which changes when this
        # fires and does so correctly: the guard exists to stop the model answering a
        # question the corpus could not answer, and on a planned turn the records tool has
        # already put real material in front of it. Sequentially the search ran first and
        # ended the turn before the records tool was ever reached, so a parent asking
        # «درجات ليلى كام والمصاريف كام؟» against a corpus with no fee table was told
        # nothing at all — including about the marks, which were sitting there unread.
        if _other_tools_ran(state["messages"]):
            return None
        ctx.note_short_circuit(status)
        return {"jump_to": "end"}

    return _stop_after_a_terminal_tool_result


def create_agent_for_request(
    ctx: ChatRequestContext,
    tool_names: list[str] | None = None,
    language: str | None = None,
):
    """Build the agent for one turn.

    `tool_names` narrows what is bound for this turn — the turn planner passes a
    shorter list when signals say a capability cannot apply, and every tool schema
    omitted is tokens saved on a call that is paid on every turn.

    None means "whatever the profile allows", which is what any uncertainty produces:
    narrowing is an optimisation, so it must never be the reason a capability is out
    of reach.
    """
    profile = get_profile()
    allowed = profile.agent.tools if tool_names is None else tool_names
    return create_agent(
        model=model,
        tools=build_tools(allowed, ctx),
        # Same list drives both, so the prompt can never describe a capability this
        # turn did not bind, or stay silent about one it did.
        system_prompt=profile.render_system_prompt(allowed, language),
        middleware=[
            # ORDER IS LOAD-BEARING. `before_model` hooks are chained FIRST-IN-LIST-FIRST
            # and a jump short-circuits the rest, while `wrap_model_call` composes
            # first-in-list as the OUTERMOST layer:
            #
            #   the planner's dispatch runs before every other before_model hook, so a
            #   turn whose tools were decided in advance never reaches the model without
            #   them — and because it jumps to the tool node, nothing after it on that
            #   pass runs at all. It is inert from the second pass onward, so the hooks
            #   below see an ordinary loop;
            #
            #   duplicates are dropped as the model produces them, so the budget counts
            #   the calls that will actually run rather than the copies, and the terminal
            #   guard still sees a single result;
            #
            #   the budget sits OUTSIDE the forcing, so a request that has had its spent
            #   tools withheld reaches the forcing already narrowed — which is what lets
            #   that middleware settle for "force only a tool still on the request" and
            #   never produce the one shape the provider rejects outright, a required
            #   call with nothing to call.
            _dispatch_planned_tools(ctx),
            _collapse_duplicate_tool_calls(ctx),
            _spend_tool_budgets(ctx),
            _force_the_planned_tool(ctx),
            _end_turn_on_terminal_retrieval(ctx),
        ],
    )
