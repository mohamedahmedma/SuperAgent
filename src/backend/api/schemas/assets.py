"""What POST /media/resolve accepts and answers.

`extra="forbid"` on both, as they always had: a client sending a field this API does not know
is told so with a 422, rather than having it silently ignored.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.assets.delivery import AssetReference, ClientCapabilities


class AssetResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_ids: List[str] = Field(default_factory=list, max_length=64)
    capabilities: Optional[ClientCapabilities] = None


class AssetResolveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: List[AssetReference] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)


__all__ = ["AssetResolveRequest", "AssetResolveResponse"]
