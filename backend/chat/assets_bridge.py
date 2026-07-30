"""Bridge between a chat turn and the asset layer.

Kept out of `service.py` so the chat flow stays about conversation. Everything here is
about one question: which images did this turn surface, and how should this particular
client receive them.

Nothing in this module knows what the client is. It takes declared capabilities and
returns structured references; a browser, a bot, and a downstream service all go
through the same function.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from backend.assets.delivery import AssetReference, ClientCapabilities, collect_asset_ids
from backend.chat.asset_context import SESSION_ASSET_KEY, SessionAssetState

logger = logging.getLogger(__name__)


def load_asset_state(metadata: Optional[dict]) -> SessionAssetState:
    return SessionAssetState.from_metadata(metadata).begin_turn()


def save_asset_state(save_meta: dict, state: SessionAssetState) -> dict:
    save_meta[SESSION_ASSET_KEY] = state.to_metadata()
    return save_meta


def effective_capabilities(
    requested: Optional[ClientCapabilities],
    delivery_config,
) -> ClientCapabilities:
    """Client wishes, clamped by deployment policy.

    A client may ask for less than the profile allows, never more: an integrator that
    requests 40 MB of inlined images or fifty attachments would otherwise be able to
    set the server's egress budget for it.
    """
    capabilities = (requested or ClientCapabilities()).model_copy(deep=True)
    capabilities.max_inline_bytes = min(capabilities.max_inline_bytes, delivery_config.max_inline_bytes)
    capabilities.max_assets = min(capabilities.max_assets, delivery_config.max_assets_per_response)
    return capabilities


# "[1]", "[2][3]", "[1, 2]" — the citation markers the agent is instructed to emit.
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_chunk_indices(answer: str) -> List[int]:
    """1-based chunk numbers the answer actually cited, in the order first cited.

    The agent is already required to cite `[n]` inline, and `search_knowledge_base`
    numbers the chunks it returns from 1. That numbering is therefore a free, exact
    record of which evidence the answer leaned on — no extra model call needed to
    recover it.
    """
    seen: List[int] = []
    for group in _CITATION_RE.findall(answer or ""):
        for part in group.split(","):
            try:
                index = int(part.strip())
            except ValueError:
                continue
            if index > 0 and index not in seen:
                seen.append(index)
    return seen


def asset_ids_for_answer(
    answer: str,
    ctx,
    rag_trace: Optional[dict],
    delivery_config,
) -> List[str]:
    """Assets to attach to THIS answer.

    Retrieval routinely surfaces several figures — a uniform question hits every
    uniform image in the document — but the answer usually rests on one. Attaching all
    of them makes the user do the filtering the citation already did.

    So when the answer cites its sources, only the cited chunks' assets are attached.
    An answer that cites nothing falls back to everything surfaced: it may still have
    used a figure, and showing too much beats showing nothing.
    """
    surfaced = asset_ids_for_turn(ctx, rag_trace)
    if not surfaced or not getattr(delivery_config, "attach_only_cited", True):
        return surfaced

    chunks = (rag_trace or {}).get("retrieved_chunks") or []
    cited = cited_chunk_indices(answer)
    if not cited or not chunks:
        return surfaced

    ids: List[str] = []
    for index in cited:
        if 1 <= index <= len(chunks):
            for asset_id in chunks[index - 1].get("asset_ids") or []:
                if asset_id and asset_id not in ids:
                    ids.append(asset_id)

    # A citation that pointed only at text chunks yields nothing. That is a correct
    # answer to "which images did this use?", so it is respected rather than widened.
    return ids if ids else []


def asset_ids_for_turn(ctx, rag_trace: Optional[dict]) -> List[str]:
    """Assets this turn surfaced, most relevant first.

    The context is the primary source because the knowledge tool pins ids as it
    formats results. The trace is a fallback for paths that bypass the tool — the HITL
    resume flow answers directly from `docs` without ever calling it.
    """
    ids = list(ctx.surfaced_asset_ids()) if ctx is not None else []
    if ids:
        return ids
    if isinstance(rag_trace, dict):
        return collect_asset_ids(rag_trace.get("retrieved_chunks") or [])
    return []


def build_asset_references(
    asset_ids: List[str],
    capabilities: ClientCapabilities,
    delivery_config,
) -> List[AssetReference]:
    """Resolve ids to renditions. Never raises into a chat turn: a failure here costs
    the user their pictures, and must not cost them their answer."""
    if not delivery_config.attach_to_response or not asset_ids:
        return []
    try:
        from backend.assets.delivery import get_asset_presenter
        from backend.assets.store import get_asset_store

        dossiers = get_asset_store().get_many(asset_ids[: capabilities.max_assets])
        return get_asset_presenter().present_many(dossiers, capabilities)
    except Exception:
        logger.exception("Failed to build asset references for %d asset(s)", len(asset_ids))
        return []


def attach_assets_to_trace(rag_trace: Optional[dict], references: List[AssetReference]) -> Optional[dict]:
    if not rag_trace or not references:
        return rag_trace
    enriched = dict(rag_trace)
    enriched["assets"] = [reference.model_dump(mode="json", exclude_none=True) for reference in references]
    return enriched
