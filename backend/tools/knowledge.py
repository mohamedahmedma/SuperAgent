"""The search_knowledge_base tool.

What this tool SAYS back to the model lives in
backend/prompts/templates/tools/knowledge_result.j2, not in this file. That text is
prompt — it instructs the model — and it is the Class B half of prompt routing: every
branch of it depends on what retrieval actually returned, which the system prompt
cannot know because it is built before the tool runs. Keeping it in a template means
each instruction is paid only on the turns where its condition is real.

This module decides WHICH outcome occurred. The template renders it.
"""
from langchain_core.tools import tool

from backend.chat.request_context import ChatRequestContext
from backend.prompts import render as render_prompt


def _format_chunk(index: int, doc: dict) -> str:
    """One retrieved chunk, as the model sees it.

    Serialization rather than instruction, which is why it is here and not in the
    template: there is nothing to instruct, and Jinja loop whitespace control would
    make it harder to read for no gain.
    """
    entry = (
        f"[{index}] {doc.get('filename', 'Unknown')} "
        f"(Page {doc.get('page_number', 'N/A')}):\n{doc.get('text', '')}"
    )
    # Naming the asset_id inline is what makes view_figure callable at all — the model
    # can only pass back an id it has actually seen.
    asset_ids = [item for item in (doc.get("asset_ids") or []) if item]
    if asset_ids:
        entry += "\n(figure available — view_figure asset_id: " + ", ".join(asset_ids) + ")"
    return entry


def make_search_knowledge_base(ctx: ChatRequestContext):
    @tool("search_knowledge_base")
    def search_knowledge_base(query: str) -> str:
        """Search the knowledge base for documents that answer the user's question.

        Use this whenever the user asks something the corpus would know. Call it once
        per turn and answer from what it returns; a second call in the same turn is
        refused. How the search runs is not your concern — pass the user's information
        need in your own words.
        """
        if not ctx.acquire_knowledge_tool_slot():
            return render_prompt("tools/knowledge_result.j2", outcome="call_limit")

        # Delayed import keeps tests and lightweight imports away from RAG/embedding startup.
        from backend.rag.pipeline import run_rag_graph

        rag_result = run_rag_graph(query, ctx)

        docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
        rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
        hitl_resume_state = (
            rag_result.get("hitl_resume_state")
            if isinstance(rag_result, dict)
            else None
        )
        ctx.store_rag_trace(rag_trace, hitl_resume_state)

        status = rag_trace.get("retrieval_status") if isinstance(rag_trace, dict) else None
        route = rag_trace.get("route") if isinstance(rag_trace, dict) else None
        if status == "needs_clarification" or route == "clarify":
            return render_prompt(
                "tools/knowledge_result.j2",
                outcome="needs_clarification",
                prompt=rag_trace.get("hitl_prompt")
                or "I found related knowledge, but need one more detail before answering.",
            )

        if status == "needs_scope_selection" or route == "scope_select":
            return render_prompt(
                "tools/knowledge_result.j2",
                outcome="needs_scope_selection",
                prompt=rag_trace.get("hitl_prompt")
                or "I found multiple related knowledge-base directions. Ask the user to choose one.",
                options=[str(item) for item in (rag_trace.get("hitl_options") or [])],
            )

        if status == "retrieval_error" or route == "retrieval_error":
            return render_prompt("tools/knowledge_result.j2", outcome="retrieval_error")

        if status == "no_knowledge" or route == "no_knowledge":
            return render_prompt("tools/knowledge_result.j2", outcome="no_knowledge")

        if not docs:
            return render_prompt("tools/knowledge_result.j2", outcome="empty")

        # Pin every asset retrieval surfaced, whether or not the model goes on to look
        # at one. This is what lets a client attach the picture to a response that was
        # answered entirely from the caption.
        from backend.assets.delivery import collect_asset_ids

        ctx.note_surfaced_assets(collect_asset_ids(docs))

        return render_prompt(
            "tools/knowledge_result.j2",
            outcome="chunks",
            chunks="\n\n---\n\n".join(
                _format_chunk(i, doc) for i, doc in enumerate(docs, 1)
            ),
            # Only a rewritten retrieval needs the caveat, and only the trace knows
            # whether one happened — which the system prompt could not, being fixed
            # before any of this ran. Paid on the minority of turns that rewrote.
            rewritten=bool(rag_trace.get("rewrite_method")),
        )

    return search_knowledge_base
