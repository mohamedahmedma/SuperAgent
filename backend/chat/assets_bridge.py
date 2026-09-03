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

from backend.assets.delivery import (
    AssetReference,
    ClientCapabilities,
    collect_asset_ids,
    references_by_id,
)

logger = logging.getLogger(__name__)


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

    Retrieval routinely surfaces several figures — a question about the sports kit hits
    every uniform image in the document — but the answer rests on one. Attaching all of
    them makes the reader do the filtering, and hands a parent asking about PE clothes
    two pictures of day wear they did not ask for.

    The answer already says which chunk it used: the `[n]` markers the agent is required
    to emit. The chunk header tells the model which chunks ARE figures, so citing one is
    a deliberate choice rather than a coincidence, and no second model call is needed to
    recover it — the selection rides along with a citation the turn was paying for anyway.

    When the citations cannot select — none emitted, none landing on a figure, or no
    chunk list to map them against — the BEST-RANKED figure is shown, and only that one.
    Retrieval ranked it first for this question, which is a weaker signal than a citation
    but a much better one than showing everything: the turn that asks "where is the
    picture?" must not come back with none, and the turn that asks about the sports kit
    must not come back with three.
    """
    surfaced = asset_ids_for_turn(ctx, rag_trace)
    if not surfaced or not getattr(delivery_config, "attach_only_cited", True):
        return surfaced

    chunks = (rag_trace or {}).get("retrieved_chunks") or []
    cited = cited_chunk_indices(answer)

    ids: List[str] = []
    for index in cited:
        if 1 <= index <= len(chunks):
            for asset_id in chunks[index - 1].get("asset_ids") or []:
                if asset_id and asset_id not in ids:
                    ids.append(asset_id)

    return ids or surfaced[:1]


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
    """Record the turn's renditions on the trace, which is what persists them.

    A trace is created when there is none but there are references, because the trace
    is the only place a stored message keeps its images: dropping them here would show
    the pictures live and lose them on the next reload. That gap is narrow — the
    knowledge tool stores a trace before it pins anything — but it is the difference
    between an image that survives a restart and one that does not.
    """
    if not references:
        return rag_trace
    enriched = dict(rag_trace or {})
    enriched["assets"] = [reference.model_dump(mode="json", exclude_none=True) for reference in references]
    return enriched


def trace_for_storage(rag_trace: Optional[dict]) -> Optional[dict]:
    """The trace as it should be PERSISTED: assets by id, not by value.

    A stored rendition is a second copy of something the asset store already holds — and
    under inline delivery that copy is the image itself, base64'd into the conversation
    row, once per message that showed it. A conversation would then grow with the
    pictures in it rather than with the words, and the copies would go stale the moment
    a document was re-ingested.

    So only the ids are kept. `asset_id` is the asset table's primary key, which makes
    restoring a keyed lookup rather than a search, and makes an image stored once serve
    every message that ever showed it.
    """
    if not rag_trace or "assets" not in rag_trace:
        return rag_trace
    stored = {key: value for key, value in rag_trace.items() if key != "assets"}
    asset_ids = [
        asset["asset_id"]
        for asset in rag_trace.get("assets") or []
        if isinstance(asset, dict) and asset.get("asset_id")
    ]
    if asset_ids:
        stored["asset_ids"] = asset_ids
    return stored


def restore_session_assets(
    records: List[dict],
    capabilities: Optional[ClientCapabilities] = None,
    delivery_config=None,
) -> List[dict]:
    """Rebuild renditions for a loaded conversation, in ONE lookup for the whole session.

    Stored traces carry ids; a client needs renditions. Resolving them message by message
    would be a query per message, so every id in the session is collected, looked up
    once, and handed back to the messages that referenced it — a reloaded conversation
    costs one indexed `IN` query no matter how many images it showed.

    Messages saved before ids were stored still carry their renditions inline; those are
    left exactly as they are. An id that no longer resolves — its document deleted or
    re-ingested — simply yields no image, rather than a card pointing at nothing.
    """
    wanted: List[str] = []
    for record in records:
        for asset_id in (record.get("rag_trace") or {}).get("asset_ids") or []:
            if asset_id and asset_id not in wanted:
                wanted.append(asset_id)
    if not wanted:
        return records

    if delivery_config is None:
        from backend.profiles import get_profile

        delivery_config = get_profile().assets.delivery
    capabilities = effective_capabilities(capabilities, delivery_config)
    # The session's whole set is resolved in one call, so the per-response cap must not
    # truncate the lookup; it is applied per message below, where it means something.
    lookup = capabilities.model_copy(update={"max_assets": len(wanted)})
    # No early return when nothing resolves: a message that referenced assets gets an
    # `assets` list either way, so "none of these are available any more" is a fact a
    # client can read rather than a missing key it has to interpret.
    by_id = references_by_id(build_asset_references(wanted, lookup, delivery_config))

    restored = []
    for record in records:
        trace = record.get("rag_trace")
        asset_ids = (trace or {}).get("asset_ids") or []
        if not trace or not asset_ids:
            restored.append(record)
            continue
        references = [by_id[asset_id] for asset_id in asset_ids if asset_id in by_id]
        current = dict(record)
        current["rag_trace"] = {
            **trace,
            "assets": [
                reference.model_dump(mode="json", exclude_none=True)
                for reference in references[: capabilities.max_assets]
            ],
        }
        restored.append(current)
    return restored
