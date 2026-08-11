"""Session asset state: turning an image into text, once, per conversation.

The problem this solves is that re-sending the same picture on turns 2, 3, and 4
multiplies image tokens for no new information. Image tokens dominate a request, so a
five-turn conversation about one chart can cost more than the entire rest of the
session.

The fix is to treat the first pixel read as a **conversion**. What the model saw is
recorded as an observation on the session, and every later question about that asset
is answered from the dossier plus those observations — as text, for free. Pixels are
re-read only when a question demands detail that neither contains, and only while the
per-asset budget allows.

State lives in the session metadata alongside `pending_hitl`, so it survives across
requests and processes without a new store.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

SESSION_ASSET_KEY = "asset_context"


class AssetObservation(BaseModel):
    """Something a model reported after actually looking at the pixels."""

    model_config = ConfigDict(extra="forbid")

    question: str = ""
    observation: str
    at: str = ""


class PinnedAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    first_seen_turn: int = 0
    pixel_reads: int = 0
    observations: List[AssetObservation] = Field(default_factory=list)

    def observation_text(self) -> str:
        if not self.observations:
            return ""
        return "\n".join(f"- {item.observation}" for item in self.observations)


class SessionAssetState(BaseModel):
    """Assets surfaced in this conversation, and what has been learned about them."""

    model_config = ConfigDict(extra="forbid")

    assets: Dict[str, PinnedAsset] = Field(default_factory=dict)
    turn: int = 0

    # -- lifecycle --------------------------------------------------------------

    @classmethod
    def from_metadata(cls, metadata: Optional[dict]) -> "SessionAssetState":
        """Rebuild from session metadata. A payload written by an older schema is
        discarded rather than raised on — losing the observation log costs one extra
        pixel read, whereas failing the request costs the user their turn."""
        raw = (metadata or {}).get(SESSION_ASSET_KEY)
        if not isinstance(raw, dict):
            return cls()
        try:
            return cls.model_validate(raw)
        except Exception:
            return cls()

    def to_metadata(self) -> dict:
        return self.model_dump(mode="json")

    def begin_turn(self) -> "SessionAssetState":
        self.turn += 1
        return self

    # -- pinning ----------------------------------------------------------------

    def pin(self, asset_id: str) -> PinnedAsset:
        pinned = self.assets.get(asset_id)
        if pinned is None:
            pinned = PinnedAsset(asset_id=asset_id, first_seen_turn=self.turn)
            self.assets[asset_id] = pinned
        return pinned

    def pin_many(self, asset_ids: List[str]) -> None:
        for asset_id in asset_ids:
            if asset_id:
                self.pin(asset_id)

    def is_pinned(self, asset_id: str) -> bool:
        return asset_id in self.assets

    def get(self, asset_id: str) -> Optional[PinnedAsset]:
        return self.assets.get(asset_id)

    # -- pixel budget -----------------------------------------------------------

    def can_read_pixels(self, asset_id: str, max_reads: int) -> bool:
        pinned = self.assets.get(asset_id)
        return (pinned.pixel_reads if pinned else 0) < max_reads

    def record_observation(
        self,
        asset_id: str,
        observation: str,
        question: str = "",
        max_observations: int = 6,
    ) -> PinnedAsset:
        """Record what a pixel read revealed. This is the step that makes every
        subsequent question about the asset a text question."""
        pinned = self.pin(asset_id)
        pinned.pixel_reads += 1
        text = (observation or "").strip()
        if text:
            pinned.observations.append(
                AssetObservation(
                    question=(question or "").strip(),
                    observation=text,
                    at=datetime.now(UTC).isoformat(),
                )
            )
            # Keep the most recent: later observations answer later questions, and an
            # unbounded log would grow the prompt without bound.
            if len(pinned.observations) > max_observations:
                pinned.observations = pinned.observations[-max_observations:]
        return pinned

    # -- reporting --------------------------------------------------------------

    def pinned_ids(self) -> List[str]:
        return list(self.assets)

    def total_pixel_reads(self) -> int:
        return sum(item.pixel_reads for item in self.assets.values())

    def summary(self) -> dict:
        return {
            "pinned": len(self.assets),
            "pixel_reads": self.total_pixel_reads(),
            "with_observations": sum(1 for item in self.assets.values() if item.observations),
        }
