"""The view_figure tool: on-demand access to a retrieved image.

Retrieval hands the model a figure's caption and transcription as text. Most questions
are answered by that alone, so pixels are never pre-attached to a request — image
tokens are the most expensive thing in a multimodal call and paying them speculatively
on every retrieval that happens to touch a figure is almost entirely waste.

Instead the model calls this tool when the text is not enough. The escalation ladder,
cheapest rung first:

    1. prior observations in this session  -> free, and the common case from turn 2 on
    2. the dossier's own text surface      -> free
    3. an actual pixel read                -> budgeted, recorded as an observation

Step 3 is what makes steps 1 and 2 work later: reading an image once converts it into
text for the rest of the conversation.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from backend.chat.request_context import ChatRequestContext

logger = logging.getLogger(__name__)

TOOL_BUDGET_EXHAUSTED = (
    "FIGURE_BUDGET_REACHED: the figure-viewing budget for this turn is used up. "
    "Answer from the retrieved text you already have."
)
UNKNOWN_ASSET = (
    "FIGURE_NOT_FOUND: no such figure. Use an asset_id exactly as it appeared in the "
    "retrieved context, and do not invent one."
)


def _describe(dossier) -> str:
    """The dossier's text surface, formatted for the model."""
    extraction = dossier.extraction
    if extraction is None:
        return "(no extracted description available for this figure)"

    text = extraction.text
    parts = []
    if text.caption:
        parts.append(f"Caption: {text.caption}")
    if text.description:
        parts.append(f"Description: {text.description}")
    if text.transcription:
        parts.append(f"Text visible in the image:\n{text.transcription}")
    if text.tags:
        parts.append("Tags: " + ", ".join(text.tags))
    return "\n".join(parts) or "(no extracted description available for this figure)"


def make_view_figure(ctx: ChatRequestContext):
    @tool("view_figure")
    def view_figure(asset_id: str, question: str = "") -> str:
        """Look at a figure that appeared in the retrieved context.

        Use this only when the figure's caption and transcription in the retrieved
        context do not answer the question — for example to read a value off a chart
        or check a detail of a diagram. Pass the asset_id exactly as shown, and the
        specific question you need answered.
        """
        asset_id = (asset_id or "").strip()
        if not asset_id:
            return UNKNOWN_ASSET

        if not ctx.acquire_figure_tool_slot():
            return TOOL_BUDGET_EXHAUSTED

        # Delayed imports keep tests and lightweight flows away from store/model setup.
        from backend.assets.store import get_asset_store
        from backend.profiles import get_profile

        dossier = get_asset_store().get(asset_id)
        if dossier is None:
            return UNKNOWN_ASSET

        profile = get_profile()
        delivery = profile.assets.delivery
        state = ctx.asset_state
        state.pin(asset_id)

        source = f"{dossier.source.filename} page {dossier.source.page_number}"
        described = _describe(dossier)

        # Rung 1 — what a previous read in this session already established.
        pinned = state.get(asset_id)
        prior = pinned.observation_text() if pinned else ""

        # Rung 3 — pixels, only when vision exists AND this asset has budget left.
        if state.can_read_pixels(asset_id, delivery.max_pixel_reads_per_asset):
            observation = _read_pixels(dossier, question, source)
            if observation:
                state.record_observation(
                    asset_id,
                    observation,
                    question=question,
                    max_observations=delivery.max_observations_per_asset,
                )
                return (
                    f"Figure {asset_id} ({source})\n{described}\n\n"
                    f"Looked at the image:\n{observation}"
                )

        # Rung 2 — dossier text, plus anything earlier reads established.
        answer = f"Figure {asset_id} ({source})\n{described}"
        if prior:
            answer += f"\n\nAlready observed in this conversation:\n{prior}"
        else:
            answer += (
                "\n\n(Image not re-read: either no vision model is configured or this "
                "figure's re-read budget is used up. Answer from the text above, and "
                "say plainly if it is insufficient.)"
            )
        return answer

    return view_figure


def _read_pixels(dossier, question: str, source: str) -> str:
    """One pixel read, or "" when unavailable. Never raises into the agent loop."""
    from backend.assets.blobs import get_blob_store
    from backend.assets.reader import ReadRequest, get_figure_reader

    reader = get_figure_reader()
    if not reader.available or not dossier.blob.uri:
        return ""

    try:
        data = get_blob_store().get(dossier.blob.uri)
    except Exception:
        logger.exception("Could not load bytes for %s", dossier.asset_id)
        return ""

    caption = dossier.extraction.text.caption if dossier.extraction else ""
    try:
        # The stock reader already swallows its own errors, but the port allows any
        # implementation. A raising reader must degrade to dossier text, not abort
        # the agent's turn.
        observation = reader.read(ReadRequest(
            data=data,
            content_type=dossier.blob.content_type,
            question=question,
            caption=caption,
            source=source,
        ))
    except Exception:
        logger.exception("Figure reader raised for %s", dossier.asset_id)
        return ""
    return observation or ""
