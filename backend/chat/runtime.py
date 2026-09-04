import logging
import os
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import after_model, before_model
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

from backend.chat.request_context import ChatRequestContext
from backend.llm import sampling
from backend.profiles import get_profile
from backend.tools import KNOWLEDGE_TOOL, build_tools

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
FAST_MODEL = os.getenv("FAST_MODEL")
BASE_URL = os.getenv("BASE_URL")

logger = logging.getLogger(__name__)

model = init_chat_model(
    model=MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    stream_usage=True,
    **sampling("answer"),
)

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
            # Ordered: duplicates are dropped as the model produces them, so the guard
            # below counts the tools that actually ran rather than the copies.
            _collapse_duplicate_tool_calls(ctx),
            _end_turn_on_terminal_retrieval(ctx),
        ],
    )
