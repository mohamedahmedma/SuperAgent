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

# Persona, tool contract, and citation rules all come from the active profile.
# See backend/profiles/definitions/*.yaml.
SYSTEM_PROMPT = _profile.render_system_prompt()


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


def create_agent_for_request(ctx: ChatRequestContext):
    profile = get_profile()
    return create_agent(
        model=model,
        tools=build_tools(profile.agent.tools, ctx),
        system_prompt=profile.render_system_prompt(),
    )
