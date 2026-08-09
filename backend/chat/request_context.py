from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.chat.asset_context import SessionAssetState
from backend.chat.caller_identity import CallerIdentity
from backend.schemas.chat import HitlResumeState, normalize_rag_trace

logger = logging.getLogger(__name__)


@dataclass
class ChatRequestContext:
    """Request-owned state shared explicitly across agent tools and RAG nodes."""

    user_id: str
    session_id: str
    output_queue: Optional[asyncio.Queue] = None
    loop: Optional[asyncio.AbstractEventLoop] = None

    # Verified identity for this turn, set by the HTTP layer from the caller's bearer
    # token and never by anything downstream. OPAQUE to the agent: no prompt renders
    # it, no tool takes it as an argument, and the model has no way to read or
    # influence it. The records tool relays the token; the records facade checks its
    # signature and decides what it authorises.
    #
    # One object rather than loose fields, so guardian id and token cannot drift apart
    # — a context holding one without the other is not a state any code has to handle.
    # Defaults to a bare identity derived from `user_id`, which is the correct and safe
    # shape for a staff session, a background job or a test: not a parent, reads
    # nothing.
    caller: Optional[CallerIdentity] = None

    # Assets surfaced and read during this conversation. Carried on the context (not
    # a module global) so concurrent requests cannot see each other's state.
    asset_state: SessionAssetState = field(default_factory=SessionAssetState)

    _lock: threading.RLock = field(default_factory=threading.RLock)
    _active: bool = True
    _rag_trace: Optional[dict] = None
    _knowledge_tool_slots_used: int = 0
    _figure_tool_slots_used: int = 0
    _records_tool_slots_used: int = 0
    _surfaced_asset_ids: list = field(default_factory=list)
    _started_at: float = field(default_factory=time.monotonic)
    _last_step_at: Optional[float] = None

    def __post_init__(self) -> None:
        """Settle the caller, and refuse a context whose identity contradicts itself.

        `user_id` remains the storage key, so a `caller` naming someone else would
        write one user's conversation under another's name while reading a third
        party's records. That is a silent, serious bug, and it is cheap to make
        impossible here rather than plausible everywhere downstream.
        """
        if self.caller is None:
            self.caller = CallerIdentity.for_user(self.user_id)
        elif self.caller.user_id != self.user_id:
            raise ValueError(
                f"caller.user_id {self.caller.user_id!r} does not match "
                f"user_id {self.user_id!r}"
            )

    # Read-only views onto the caller. Properties rather than fields so there is
    # exactly one place the identity lives; a tool cannot assign to these and change
    # whose records the turn may read.
    @property
    def guardian_id(self) -> str:
        return self.caller.guardian_id if self.caller else ""

    @property
    def guardian_token(self) -> str:
        return self.caller.guardian_token if self.caller else ""

    @property
    def is_parent(self) -> bool:
        return bool(self.caller and self.caller.is_parent)

    @classmethod
    def for_stream(
        cls,
        *,
        user_id: str,
        session_id: str,
        output_queue: asyncio.Queue,
        asset_state: Optional[SessionAssetState] = None,
        caller: Optional[CallerIdentity] = None,
    ) -> ChatRequestContext:
        return cls(
            user_id=user_id,
            session_id=session_id,
            output_queue=output_queue,
            loop=asyncio.get_running_loop(),
            asset_state=asset_state or SessionAssetState(),
            caller=caller,
        )

    @classmethod
    def for_sync(
        cls,
        *,
        user_id: str,
        session_id: str,
        asset_state: Optional[SessionAssetState] = None,
        caller: Optional[CallerIdentity] = None,
    ) -> ChatRequestContext:
        return cls(
            user_id=user_id,
            session_id=session_id,
            asset_state=asset_state or SessionAssetState(),
            caller=caller,
        )

    def emit_rag_step(
        self,
        icon: str,
        label: str,
        detail: str = "",
        *,
        group: Optional[str] = None,
        group_label: Optional[str] = None,
    ) -> None:
        with self._lock:
            if not self._active:
                return
            if self.output_queue is None or self.loop is None:
                return
            now = time.monotonic()
            last_step_at = self._last_step_at or self._started_at
            elapsed_ms = max(int((now - self._started_at) * 1000), 0)
            stage_elapsed_ms = max(int((now - last_step_at) * 1000), 0)
            self._last_step_at = now
            queue = self.output_queue
            loop = self.loop

        step = {
            "icon": icon,
            "label": label,
            "detail": detail,
            "elapsed_ms": elapsed_ms,
            "stage_elapsed_ms": stage_elapsed_ms,
        }
        if group:
            step["group"] = group
        if group_label:
            step["group_label"] = group_label

        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "rag_step", "step": step},
                )
        except Exception:
            logger.exception("Failed to emit RAG step")

    def store_rag_trace(self, rag_trace: dict, hitl_resume_state: Optional[dict] = None) -> None:
        current_trace = normalize_rag_trace(rag_trace)
        if not current_trace:
            return
        with self._lock:
            if self._active:
                self._rag_trace = {"rag_trace": current_trace}
                if hitl_resume_state:
                    self._rag_trace["hitl_resume_state"] = HitlResumeState.model_validate(
                        hitl_resume_state
                    ).model_dump()

    def take_rag_trace(self) -> Optional[dict]:
        with self._lock:
            context = self._rag_trace
            self._rag_trace = None
            return context

    def peek_rag_trace(self) -> Optional[dict]:
        with self._lock:
            return self._rag_trace

    def reset_knowledge_tool_budget(self) -> None:
        with self._lock:
            self._knowledge_tool_slots_used = 0
            self._figure_tool_slots_used = 0

    def acquire_records_tool_slot(self) -> bool:
        """Budget for get_student_records, separate from the other tool budgets.

        A parent asking about two children in one turn is a legitimate three-call
        sequence — list the children, then read each one's grades — so the ceiling is
        higher than retrieval's. It exists at all because every call is a network
        round trip to another service and an audited read of a minor's records; a
        model that loops here is expensive in a way that shows up in a compliance
        report, not just a bill.
        """
        limit = int(os.getenv("RECORDS_MAX_CALLS_PER_TURN") or 4)
        with self._lock:
            if self._records_tool_slots_used >= limit:
                return False
            self._records_tool_slots_used += 1
            return True

    def acquire_figure_tool_slot(self) -> bool:
        """Budget for view_figure, separate from the knowledge-tool budget: looking at
        a figure must not consume the turn's single retrieval."""
        from backend.profiles import get_profile

        limit = get_profile().agent.max_figure_calls_per_turn
        with self._lock:
            if self._figure_tool_slots_used >= limit:
                return False
            self._figure_tool_slots_used += 1
            return True

    def note_surfaced_assets(self, asset_ids) -> None:
        """Record assets that retrieval put in front of the model.

        Pinning here (rather than when a figure is read) is what lets a client attach
        images to a response the model never explicitly looked at — the common case,
        since a caption usually answers the question on its own.
        """
        with self._lock:
            if not self._active:
                return
            for asset_id in asset_ids or []:
                if asset_id and asset_id not in self._surfaced_asset_ids:
                    self._surfaced_asset_ids.append(asset_id)
            self.asset_state.pin_many(list(self._surfaced_asset_ids))

    def surfaced_asset_ids(self) -> list:
        with self._lock:
            return list(self._surfaced_asset_ids)

    def acquire_knowledge_tool_slot(self) -> bool:
        # Budget comes from the profile: a catalogue deployment may legitimately allow
        # more knowledge calls per turn than a single-shot document assistant.
        from backend.profiles import get_profile

        limit = get_profile().agent.max_knowledge_calls_per_turn
        with self._lock:
            if self._knowledge_tool_slots_used >= limit:
                return False
            self._knowledge_tool_slots_used += 1
            return True

    def close(self) -> None:
        with self._lock:
            self._active = False
            self.output_queue = None
            self.loop = None
            # Drop the bearer token when the turn ends. The context can outlive the
            # request that authorised it — held by a queue, a traceback, a retained
            # reference in a test — and a live credential should not outlive the reason
            # it was handed over. Who the caller was stays readable.
            if self.caller is not None:
                self.caller = self.caller.without_credentials()
