"""The search_products tool.

The design point: the tool's argument schema is **generated from the profile's
attribute vocabulary**, so the agent fills in `filters` as part of the call it was
already going to make. Attribute filtering therefore costs zero additional LLM calls,
and the agent cannot name a facet the index does not have — the schema it is given
only contains real ones.

Change the `attributes:` block in a profile and this tool's signature changes with it.
Nothing in this module knows about shoes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import ConfigDict, Field, create_model

from backend.chat.request_context import ChatRequestContext

logger = logging.getLogger(__name__)

NO_ENTITY_SUPPORT = (
    "PRODUCT_SEARCH_UNAVAILABLE: this deployment has no product catalogue configured. "
    "Use search_knowledge_base instead."
)
NO_RESULTS = (
    "NO_PRODUCTS_FOUND: nothing in the catalogue matched. Tell the user plainly and "
    "suggest relaxing a constraint; do not invent products."
)


def _describe(result) -> str:
    lines = [f"Found {len(result.hits)} product(s):"]
    for index, hit in enumerate(result.hits, 1):
        parts = [f"[{index}] {hit.caption or '(untitled)'}"]
        if hit.summary:
            parts.append(hit.summary)
        if hit.attributes:
            rendered = ", ".join(
                f"{name}={', '.join(str(item) for item in value) if isinstance(value, list) else value}"
                for name, value in hit.attributes.items()
            )
            parts.append(rendered)
        # A product hit IS its image, so the id is the record's identity here rather
        # than a handle for looking at pixels — unlike a knowledge chunk, whose header
        # carries a bare [FIGURE] marker and no id at all (backend/tools/knowledge.py).
        parts.append(f"asset_id: {hit.asset_id}")
        lines.append("\n    ".join(parts))

    if result.rejected_filters:
        lines.append("Ignored filters: " + "; ".join(result.rejected_filters))
    return "\n".join(lines)


def make_search_products(ctx: ChatRequestContext):
    from backend.assets.attributes import build_attribute_schema
    from backend.profiles import get_profile

    profile = get_profile()
    schema = build_attribute_schema(profile.assets.entities)
    filter_model = schema.build_filter_model("ProductFilters")

    # Argument schema built here, per profile, rather than declared statically — that
    # is what carries the real vocabulary (and its closed value lists) into the
    # function-calling schema the model sees.
    args_model = create_model(
        "SearchProductsArgs",
        __config__=ConfigDict(extra="forbid"),
        query=(str, Field(description="What the customer is looking for, in their own words")),
        filters=(Optional[filter_model], Field(
            default=None,
            description=(
                "Hard constraints the customer stated explicitly. Leave a field unset "
                "when they did not state it — an invented constraint hides products "
                "they would have wanted."
            ),
        )),
    )

    def _search(query: str, filters: Optional[Any] = None) -> str:
        entities_config = profile.assets.entities
        if not (entities_config.enabled and schema):
            return NO_ENTITY_SUPPORT

        if not ctx.acquire_knowledge_tool_slot():
            return (
                "TOOL_CALL_LIMIT_REACHED: the catalogue has already been searched this "
                "turn. Answer from the results you have."
            )

        payload: Dict[str, Any] = {}
        if filters is not None:
            payload = filters.model_dump(exclude_none=True) if hasattr(filters, "model_dump") else dict(filters)

        from backend.rag.entity_retrieval import get_entity_retriever

        try:
            result = get_entity_retriever().search(query, payload)
        except Exception as exc:
            logger.exception("Product search failed")
            return f"PRODUCT_SEARCH_ERROR: {exc}"

        ctx.note_surfaced_assets([hit.asset_id for hit in result.hits])
        ctx.emit_rag_step(
            "🛍️", f"Catalogue search: {len(result.hits)} match(es)",
            f"recalled {result.recalled}, after filters {result.after_filter}",
        )

        if not result.hits:
            if result.filtered_to_empty:
                # Worth distinguishing: the query found things, the constraints removed
                # them. The agent should offer to relax a filter, not claim ignorance.
                return (
                    f"NO_PRODUCTS_FOUND: {result.recalled} candidate(s) matched the description "
                    f"but none satisfied the filters {result.filters_applied}. Suggest relaxing one."
                )
            return NO_RESULTS
        return _describe(result)

    return StructuredTool.from_function(
        func=_search,
        name="search_products",
        description=(
            "Search the product catalogue. Put what the customer described in `query`, "
            "and only the constraints they actually stated in `filters`."
        ),
        args_schema=args_model,
    )
