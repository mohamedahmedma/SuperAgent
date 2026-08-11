import os
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

from backend.chat.request_context import ChatRequestContext
from backend.llm import sampling
from backend.profiles import get_profile
from backend.tools import build_tools

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
FAST_MODEL = os.getenv("FAST_MODEL")
BASE_URL = os.getenv("BASE_URL")

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
        # called, say, the weather tool, the model still has material to answer from
        # and cutting it here would throw that away.
        tool_results = sum(1 for message in state["messages"] if isinstance(message, ToolMessage))
        if tool_results != 1:
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
        middleware=[_end_turn_on_terminal_retrieval(ctx)],
    )
