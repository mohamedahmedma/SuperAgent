"""Asset HTTP surface.

Three endpoints, deliberately generic:

    GET  /media/{asset_id}           bytes
    GET  /media/{asset_id}/metadata  one AssetReference
    POST /media/resolve              many AssetReferences, capability-aware

The prefix is "/media" rather than "/assets" because Vite builds the frontend into an
`assets/` directory; an API mounted there would shadow the SPA's own bundle.

The resolve endpoint is what makes a non-browser integration practical. A Slack bot or
a downstream service posts the asset ids it saw in a chat response along with what it
can handle, and gets back either URLs or inlined bytes accordingly — without the
backend knowing anything about that client.

`asset_id` is a lookup key, never a path. Bytes are located by reading `storage_uri`
from the database row, so a crafted id cannot escape the blob root no matter what it
contains.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from backend.assets.delivery import AssetReference, ClientCapabilities, get_asset_presenter
from backend.db.models import User
from backend.infra.auth import get_current_user
from backend.profiles import get_profile

logger = logging.getLogger(__name__)

# NOTE: the /metadata route is registered before the bytes route on purpose. Both use
# a `:path` converter, which matches greedily, so the more specific one must come
# first or `/assets/x/metadata` would resolve as an asset literally named "x/metadata".
router = APIRouter(tags=["assets"])


class AssetResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_ids: List[str] = Field(default_factory=list, max_length=64)
    capabilities: Optional[ClientCapabilities] = None


class AssetResolveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: List[AssetReference] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)


def _require_assets_enabled() -> None:
    if not get_profile().assets.enabled:
        raise HTTPException(status_code=404, detail="Asset support is disabled for this deployment.")


def _load(asset_id: str):
    from backend.assets.store import get_asset_store

    dossier = get_asset_store().get(asset_id)
    if dossier is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return dossier


@router.get("/media/{asset_id:path}/metadata", response_model=AssetReference)
async def get_asset_metadata(asset_id: str, _: User = Depends(get_current_user)) -> AssetReference:
    """Caption, dimensions, and source for one asset — no bytes."""
    _require_assets_enabled()
    return get_asset_presenter().present(_load(asset_id), ClientCapabilities())


@router.post("/media/resolve", response_model=AssetResolveResponse)
async def resolve_assets(
    request: AssetResolveRequest,
    _: User = Depends(get_current_user),
) -> AssetResolveResponse:
    """Batch-resolve asset ids into renditions the calling client can actually use."""
    _require_assets_enabled()
    from backend.assets.store import get_asset_store

    ids = [item.strip() for item in request.asset_ids if item and item.strip()]
    if not ids:
        return AssetResolveResponse()

    dossiers = get_asset_store().get_many(ids)
    found = {dossier.asset_id for dossier in dossiers}
    capabilities = request.capabilities or ClientCapabilities()
    references = get_asset_presenter().present_many(dossiers, capabilities)
    return AssetResolveResponse(
        assets=references,
        missing=[item for item in ids if item not in found],
    )


@router.get("/media/{asset_id:path}")
async def get_asset_bytes(
    asset_id: str,
    request: Request,
    _: User = Depends(get_current_user),
) -> Response:
    """The asset's bytes.

    Content is addressed by sha256 and therefore immutable, so the digest doubles as a
    strong ETag and the response can be cached indefinitely. A client that already has
    the bytes gets a 304 and transfers nothing.
    """
    _require_assets_enabled()
    dossier = _load(asset_id)
    if not dossier.blob.uri:
        raise HTTPException(status_code=404, detail="Asset has no stored bytes")

    etag = f'"{dossier.sha256}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})

    from backend.assets.blobs import get_blob_store

    try:
        data = get_blob_store().get(dossier.blob.uri)
    except FileNotFoundError:
        # The row survived but the blob did not — a real inconsistency worth logging
        # loudly, and a 404 rather than a 500 because the resource genuinely is gone.
        logger.error("Blob missing for asset %s (%s)", asset_id, dossier.blob.uri)
        raise HTTPException(status_code=404, detail="Asset bytes are no longer available")
    except Exception as exc:
        logger.exception("Failed to read blob for asset %s", asset_id)
        raise HTTPException(status_code=500, detail=f"Failed to read asset: {exc}")

    max_age = get_profile().assets.delivery.cache_max_age_seconds
    return Response(
        content=data,
        media_type=dossier.blob.content_type or "application/octet-stream",
        headers={
            "ETag": etag,
            "Cache-Control": f"private, max-age={max_age}, immutable",
            # Never render an uploaded document's image as a document in the browser.
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "inline",
        },
    )
