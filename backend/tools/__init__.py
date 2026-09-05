"""Tools callable by the LangChain Agent (functions decorated with @tool).

`TOOL_BUILDERS` is the registry that turns the tool NAMES declared in a domain
profile into bound tool objects. Every builder takes the request context and returns
a tool, so request-scoped tools (which need the context for budgets and step
emission) and stateless ones share one signature.
"""
from typing import Callable, Dict, List

from backend.chat.request_context import ChatRequestContext
from backend.tools.knowledge import make_search_knowledge_base
from backend.tools.products import make_search_products
from backend.tools.records import make_get_student_records

#: The corpus tool, named once. It is the tool whose verdict `retrieval_status` reports,
#: so the runtime's terminal-retrieval guard has to be able to recognise its results
#: among a turn's other tool results — see `backend/chat/runtime.py`.
KNOWLEDGE_TOOL = "search_knowledge_base"

# name -> builder(ctx) -> tool
TOOL_BUILDERS: Dict[str, Callable[[ChatRequestContext], object]] = {
    KNOWLEDGE_TOOL: make_search_knowledge_base,
    "search_products": make_search_products,
    # Reads a student's academic record from the records facade. Registered but not
    # bound by any profile yet — a deployment opts in by naming it, which keeps every
    # existing profile's behaviour unchanged.
    "get_student_records": make_get_student_records,
}


# Tools whose results are numbered evidence the answer is expected to cite. This is a
# property of the TOOL, so it lives next to the registry: whoever adds another tool
# sees both structures together and has to decide which side it falls on.
#
# It drives whether the agent's system prompt includes its grounding-and-citation
# block at all (backend/prompts/templates/agent/system.j2). A deployment binding only
# get_student_records has nothing to cite, and paying for citation rules on every one
# of its turns would be pure waste.
#
# The citation block is load-bearing for images as well as provenance: which figure a
# turn attaches is read out of the `[n]` markers (backend/chat/assets_bridge.py), so a
# grounded tool that drops them silently stops showing pictures.
GROUNDED_TOOLS: frozenset = frozenset({KNOWLEDGE_TOOL, "search_products"})

# Tools whose results make the answer's FIGURES checkable. A superset of the citation
# set above, and the two are separate because they answer different questions:
#
#   GROUNDED_TOOLS — "must this answer cite its sources?"  Decides prompt text.
#   CHECKED_TOOLS  — "may this answer state a number?"     Decides whether the assembled
#                    answer is verified against what the turn actually retrieved.
#
# They were one set, and that conflation was a hole. `get_student_records` is correctly
# outside the citation set — it returns one child's marks, there is no chunk to number
# and no picture to attach — but its results are the most figure-dense thing this
# assistant ever says, and «الرياضيات ٨٧.٥٪» is exactly the class of claim a parent acts
# on. Under one set, a turn that read a child's record was checked against nothing at
# all.
#
# The hole widened the moment the planner started narrowing tools. `_grounding_expected`
# in backend/chat/service.py asks whether any BOUND tool is in this set, so a records
# turn narrowed to `["get_student_records"]` switched the check off entirely — the
# narrowing would have removed the last checked tool from a turn precisely because that
# turn was about records. Splitting the sets is what makes the narrowing safe to ship.
CHECKED_TOOLS: frozenset = GROUNDED_TOOLS | frozenset({"get_student_records"})


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
    "KNOWLEDGE_TOOL",
    "TOOL_BUILDERS",
    "GROUNDED_TOOLS",
    "CHECKED_TOOLS",
    "UnknownToolError",
    "build_tools",
    "make_search_knowledge_base",
    "make_search_products",
    "make_get_student_records",
]
