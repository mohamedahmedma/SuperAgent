"""Tools callable by the LangChain Agent (functions decorated with @tool).

`TOOL_BUILDERS` is the registry that turns the tool NAMES declared in a domain
profile into bound tool objects. Every builder takes the request context and returns
a tool, so request-scoped tools (which need the context for budgets and step
emission) and stateless ones share one signature.
"""
from typing import Callable, Dict, List

from backend.chat.request_context import ChatRequestContext
from backend.tools.figures import make_view_figure
from backend.tools.knowledge import make_search_knowledge_base
from backend.tools.products import make_search_products
from backend.tools.weather import get_current_weather_tool as get_current_weather

# name -> builder(ctx) -> tool
TOOL_BUILDERS: Dict[str, Callable[[ChatRequestContext], object]] = {
    "get_current_weather": lambda _ctx: get_current_weather,
    "search_knowledge_base": make_search_knowledge_base,
    "view_figure": make_view_figure,
    "search_products": make_search_products,
}


class UnknownToolError(ValueError):
    """A profile named a tool that is not registered."""


def build_tools(names: List[str], ctx: ChatRequestContext) -> list:
    """Resolve profile tool names to bound tools, preserving declaration order.

    Unknown names raise rather than being skipped: a profile that silently loses a
    tool produces an agent that quietly cannot do its job, which is far harder to
    diagnose than a startup failure naming the offending tool.
    """
    unknown = [name for name in names if name not in TOOL_BUILDERS]
    if unknown:
        raise UnknownToolError(
            f"Profile declares unregistered tool(s): {', '.join(unknown)}. "
            f"Registered tools: {', '.join(sorted(TOOL_BUILDERS))}"
        )
    return [TOOL_BUILDERS[name](ctx) for name in names]


__all__ = [
    "TOOL_BUILDERS",
    "UnknownToolError",
    "build_tools",
    "get_current_weather",
    "make_search_knowledge_base",
    "make_view_figure",
    "make_search_products",
]
