import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from backend.chat.request_context import ChatRequestContext
from backend.profiles import get_profile
from backend.tools import build_tools

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
FAST_MODEL = os.getenv("FAST_MODEL")
BASE_URL = os.getenv("BASE_URL")

_profile = get_profile()

model = init_chat_model(
    model=MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=_profile.models.answer_temperature,
    stream_usage=True,
)

fast_model = init_chat_model(
    model=FAST_MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=_profile.models.fast_temperature,
    stream_usage=True,
)


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
    )
