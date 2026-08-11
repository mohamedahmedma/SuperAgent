"""Asset delivery: turning stored dossiers into something a client can display.

The governing rule is that **the backend emits structured references, never
presentation**. No HTML, no markdown image syntax, no field named after a UI
component. A Vue app, a React rewrite, a Slack bot, a CLI, and a downstream service
all consume the same `AssetReference` JSON and decide for themselves what to do with
it. Replacing the frontend entirely requires no backend change.

What varies between consumers is capability, not format, so capability is something
the client *declares* rather than something the server guesses:

    browser      accepts_images=True                      -> url reference
    Slack / bot  prefers_inline=True                      -> base64 data URI
    CLI / logs   accepts_images=False                     -> metadata only

`AssetPresenter` is the single place that maps (dossier, capabilities) to a
rendition. Adding a new channel means declaring its capabilities, not adding a code
path.
"""
from __future__ import annotations

import base64
import logging
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from backend.assets.dossier import AssetDossier, AssetRole

logger = logging.getLogger(__name__)


class AssetRenditionMode(str, Enum):
    """How the bytes reach the consumer."""

    REFERENCE = "reference"  # a URL the client fetches itself
    INLINE = "inline"        # base64 data URI embedded in the response
    METADATA = "metadata"    # no bytes at all — caption and dimensions only


class ClientCapabilities(BaseModel):
    """What the CONSUMER can handle. Declared per request; never inferred from a
    User-Agent or hardcoded per endpoint.

    Defaults describe an ordinary browser client, so an integrator that sends nothing
    gets sensible behaviour and an integrator with unusual constraints says so
    explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    accepts_images: bool = True
    # Set by consumers that cannot make a second authenticated request for the bytes
    # (chat bots, webhooks, anything without a cookie jar).
    prefers_inline: bool = False
    max_inline_bytes: int = Field(default=512_000, ge=0)
    accepted_content_types: List[str] = Field(
        default_factory=lambda: ["image/png", "image/jpeg", "image/webp", "image/gif"]
    )
    # Hard cap on how many assets are attached to one response.
    max_assets: int = Field(default=8, ge=0)

    def resolve_mode(self, content_type: str, byte_size: int) -> AssetRenditionMode:
        if not self.accepts_images:
            return AssetRenditionMode.METADATA
        if self.accepted_content_types and content_type not in self.accepted_content_types:
            return AssetRenditionMode.METADATA
        if self.prefers_inline and 0 < byte_size <= self.max_inline_bytes:
            return AssetRenditionMode.INLINE
        return AssetRenditionMode.REFERENCE


class AssetSource(BaseModel):
    """Where the asset came from, for citation."""

    model_config = ConfigDict(extra="forbid")

    filename: str = ""
    page_number: int = 0
    bbox: Optional[List[float]] = None


class AssetReference(BaseModel):
    """The public asset contract.

    Treat this as an API: **additive changes only**. Consumers outside this repository
    may depend on it, and a removed field is a breaking change for every one of them.
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    sha256: str = ""
    mode: AssetRenditionMode = AssetRenditionMode.REFERENCE

    # Populated for REFERENCE mode: an authenticated GET returns the bytes.
    url: Optional[str] = None
    # Populated for INLINE mode: a complete `data:` URI, directly usable as an <img> src.
    inline_data: Optional[str] = None

    content_type: str = ""
    byte_size: int = 0
    width: int = 0
    height: int = 0

    # Always populated, in every mode. A text-only channel still gets something
    # meaningful to say, and an image client gets its accessibility text for free.
    caption: str = ""
    alt_text: str = ""
    tags: List[str] = Field(default_factory=list)

    role: str = AssetRole.FIGURE.value
    source: AssetSource = Field(default_factory=AssetSource)

    def describe(self) -> str:
        """One-line text rendering, for channels that cannot show an image at all."""
        where = f"{self.source.filename} p{self.source.page_number}".strip()
        label = self.caption or self.alt_text or "figure"
        return f"[image: {label}]" + (f" ({where})" if where.strip() else "")


def asset_url_path(asset_id: str, base_path: str = "/media") -> str:
    """URL for an asset's bytes.

    The id is fully percent-encoded (`safe=""`), because it embeds a filename that may
    contain slashes, spaces, or non-Latin characters. The server resolves it through
    the store, so the id is a lookup key and never a filesystem path.
    """
    return f"{base_path.rstrip('/')}/{quote(asset_id, safe='')}"


class AssetPresenter:
    """Maps dossiers plus client capabilities to renditions."""

    def __init__(self, delivery_config=None, blob_store=None):
        self._config = delivery_config
        self._blob_store = blob_store

    @property
    def config(self):
        if self._config is None:
            from backend.profiles import get_profile

            self._config = get_profile().assets.delivery
        return self._config

    @property
    def blob_store(self):
        if self._blob_store is None:
            from backend.assets.blobs import get_blob_store

            self._blob_store = get_blob_store()
        return self._blob_store

    def present(
        self,
        dossier: AssetDossier,
        capabilities: Optional[ClientCapabilities] = None,
    ) -> AssetReference:
        capabilities = capabilities or ClientCapabilities()
        text = dossier.extraction.text if dossier.extraction else None

        reference = AssetReference(
            asset_id=dossier.asset_id,
            sha256=dossier.sha256,
            content_type=dossier.blob.content_type,
            byte_size=dossier.blob.byte_size,
            width=dossier.blob.width,
            height=dossier.blob.height,
            caption=(text.caption if text else ""),
            alt_text=(text.caption or text.description[:200] if text else ""),
            tags=list(text.tags) if text else [],
            role=dossier.role.value,
            source=AssetSource(
                filename=dossier.source.filename,
                page_number=dossier.source.page_number,
                bbox=dossier.source.bbox,
            ),
        )

        # An asset with no stored bytes (triaged out, or a failed blob write) can only
        # ever be metadata, whatever the client asked for.
        if not dossier.blob.uri:
            reference.mode = AssetRenditionMode.METADATA
            return reference

        mode = capabilities.resolve_mode(dossier.blob.content_type, dossier.blob.byte_size)
        if mode is AssetRenditionMode.INLINE:
            inline = self._inline(dossier)
            if inline is None:
                mode = AssetRenditionMode.REFERENCE
            else:
                reference.inline_data = inline

        if mode is AssetRenditionMode.REFERENCE:
            reference.url = asset_url_path(dossier.asset_id, self.config.base_path)

        reference.mode = mode
        return reference

    def present_many(
        self,
        dossiers: Iterable[AssetDossier],
        capabilities: Optional[ClientCapabilities] = None,
    ) -> List[AssetReference]:
        capabilities = capabilities or ClientCapabilities()
        references: List[AssetReference] = []
        for dossier in dossiers:
            if len(references) >= capabilities.max_assets:
                break
            references.append(self.present(dossier, capabilities))
        return references

    def _inline(self, dossier: AssetDossier) -> Optional[str]:
        """Base64 data URI, or None when the bytes cannot be read — in which case the
        caller falls back to a URL rather than dropping the asset."""
        try:
            data = self.blob_store.get(dossier.blob.uri)
        except Exception:
            logger.warning("Could not inline asset %s; falling back to a URL", dossier.asset_id)
            return None
        content_type = dossier.blob.content_type or "image/png"
        return f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"


_presenter: Optional[AssetPresenter] = None


def get_asset_presenter() -> AssetPresenter:
    global _presenter
    if _presenter is None:
        _presenter = AssetPresenter()
    return _presenter


def set_asset_presenter(presenter: Optional[AssetPresenter]) -> None:
    global _presenter
    _presenter = presenter


def collect_asset_ids(chunks: Sequence[dict], limit: int = 32) -> List[str]:
    """Asset ids referenced by retrieved chunks, in retrieval order, deduplicated.

    Order matters: the highest-ranked chunk's figure should be the first image a user
    sees, so this preserves rank rather than sorting or grouping.
    """
    seen: List[str] = []
    for chunk in chunks:
        for asset_id in chunk.get("asset_ids") or []:
            if asset_id and asset_id not in seen:
                seen.append(asset_id)
                if len(seen) >= limit:
                    return seen
    return seen


def references_by_id(references: Iterable[AssetReference]) -> Dict[str, AssetReference]:
    return {reference.asset_id: reference for reference in references}
