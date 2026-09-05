from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_context import SessionChild
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

    # The child this conversation settled on. Loaded from session metadata at turn
    # start and threaded BY REFERENCE, so a turn that resolves a child has already
    # written it back by the time the metadata is saved. A default-constructed pin is
    # the correct shape for a staff
    # session, a background job or a test: empty, and never consulted, because nothing
    # asks about a child without a guardian to ask on behalf of.
    child: SessionChild = field(default_factory=SessionChild)

    _lock: threading.RLock = field(default_factory=threading.RLock)
    _active: bool = True
    _rag_trace: Optional[dict] = None
    _knowledge_tool_slots_used: int = 0
    _records_tool_slots_used: int = 0
    _short_circuit_status: Optional[str] = None
    _surfaced_asset_ids: list = field(default_factory=list)
    _duplicate_tool_calls: int = 0
    # `(tool, outcome)` per call this turn. See `note_tool_outcome`.
    _tool_outcomes: list = field(default_factory=list)
    # Retrieval results already produced this turn, keyed by normalised query. See
    # `remember_retrieval` for why a request-scoped memo is the right lifetime.
    _retrieval_memo: dict = field(default_factory=dict)
    _started_at: float = field(default_factory=time.monotonic)
    _last_step_at: Optional[float] = None

    # What the turn planner worked out before the agent ran, for the RAG graph to read.
    # Public because the graph reads them positionally through `getattr` and there is
    # nothing to guard: both are hints, and an empty list is the correct default for a
    # turn that was never planned (a sync call, a test).
    retrieval_sections: list = field(default_factory=list)
    scope_options: list = field(default_factory=list)
    # Conditions carried over from earlier turns ("grades up to Year 6"). The graph
    # appends them to the retrieval query and states them in the answer prompt.
    carried_constraints: list = field(default_factory=list)
    # Whether this turn's subject came from the conversation. Routing reads it to
    # decide whether offering the user a choice of subjects could possibly help.
    is_followup: bool = False
    # The year group the school's records put this turn's child in, when the turn is
    # about one child and the question did not name a year itself. Travels BESIDE the
    # question, never inside it — see `backend/rag/pipeline.py:_search_query` for the
    # measurement that settled that.
    child_year: str = ""
    # WHICH child this turn is about, as the planner resolved it against the school's
    # own roster, before the agent ran. Empty when the turn is not about one child, or
    # when which child is still an open question.
    #
    # Per TURN, and deliberately not the same thing as `child` above: that is the pin the
    # conversation settled on and outlives this message; this is one turn's answer, and a
    # turn that resolves nobody must leave nothing behind. The records tool reads it in
    # preference to the name the MODEL supplied, which is the point — a 20B model
    # transcribing «ليلى أحمد» out of a message it half-read is the least reliable link
    # in a chain whose other end is the school's own roster.
    planned_child_id: str = ""
    planned_child_label: str = ""
    # A tool this turn must call rather than merely be offered, when the planner narrowed
    # to exactly one. Read by the middleware that sets `tool_choice` on the turn's first
    # model call; empty on every turn that did not narrow, which is most of them.
    forced_tool: str = ""

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
        caller: Optional[CallerIdentity] = None,
        child: Optional[SessionChild] = None,
    ) -> ChatRequestContext:
        return cls(
            user_id=user_id,
            session_id=session_id,
            output_queue=output_queue,
            loop=asyncio.get_running_loop(),
            caller=caller,
            child=child if child is not None else SessionChild(),
        )

    @classmethod
    def for_sync(
        cls,
        *,
        user_id: str,
        session_id: str,
        caller: Optional[CallerIdentity] = None,
        child: Optional[SessionChild] = None,
    ) -> ChatRequestContext:
        return cls(
            user_id=user_id,
            session_id=session_id,
            caller=caller,
            child=child if child is not None else SessionChild(),
        )

    def note_turn_plan(
        self,
        retrieval_sections,
        scope_options,
        *,
        carried_constraints=(),
        is_followup: bool = False,
        language: str = "",
        child_year: str = "",
        child_id: str = "",
        child_label: str = "",
        forced_tool: str = "",
    ) -> None:
        """Hand the planner's findings to the RAG graph.

        Plain values rather than the `TurnPlan` itself, so this module keeps knowing
        nothing about turn policy. Without this call the planner computed
        `retrieval_sections` for nobody: `_initial_state` read it off the context, the
        context never had it, and the hint had been inert since it was written.

        The later arguments are keyword-only and defaulted so that a caller written
        against the two-argument form — a test double, an integrating deployment —
        keeps working and simply carries nothing forward.

        `language` is the turn's detected language, carried for document-pair routing:
        where a document exists in both Arabic and English, retrieval answers from the
        half matching the question. Empty means "not established", and searches
        everything — which is the correct behaviour for a turn nobody classified, not a
        degraded one.
        """
        with self._lock:
            if not self._active:
                return
            self.retrieval_sections = list(retrieval_sections or [])
            self.scope_options = list(scope_options or [])
            self.carried_constraints = list(carried_constraints or [])
            self.is_followup = bool(is_followup)
            self.language = (language or "").strip()
            self.child_year = (child_year or "").strip()
            # Both or neither. A label with no id names a child nothing can read, and an
            # id with no label leaves the tool nothing to call them — either half alone
            # is a state every reader downstream would have to handle, so it is not one.
            child_id = (child_id or "").strip()
            child_label = (child_label or "").strip()
            self.planned_child_id = child_id if child_id and child_label else ""
            self.planned_child_label = child_label if child_id and child_label else ""
            self.forced_tool = (forced_tool or "").strip()

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

    def note_short_circuit(self, status: str) -> None:
        """Record that the graph ended itself rather than calling the model again.

        Written by the agent middleware that makes the call, read by the streamer that
        has to put a reply on the wire. One decision point: the two cannot disagree
        about whether the model was skipped, which is what makes it safe for the
        streamer to treat an empty response as intentional rather than as a failure.
        """
        with self._lock:
            if self._active:
                self._short_circuit_status = status

    def short_circuit_status(self) -> Optional[str]:
        with self._lock:
            return self._short_circuit_status

    def reset_knowledge_tool_budget(self) -> None:
        with self._lock:
            self._knowledge_tool_slots_used = 0

    @property
    def remembered_child(self) -> str:
        """The child already under discussion, or `""` on a fresh conversation.

        A hint and never an authority. It selects *which* of this parent's own children a
        vague question is about; it can never widen who may be read, because every records
        call is re-checked against the guardian link on the server that answers it.
        """
        with self._lock:
            return self.child.student_id

    @property
    def remembered_child_label(self) -> str:
        """What to call the pinned child. Empty when nothing is pinned."""
        with self._lock:
            return self.child.label

    def remember_child(
        self, student_id: str, *, label: str = "", gender: str = ""
    ) -> None:
        """Pin this conversation to a child.

        Called only once a child has actually been resolved, so a turn that failed to
        identify one does not leave the wrong child pinned for every question after it.

        Writes through to the `SessionChild` this context was built with, which is the
        same object the turn will serialise into session metadata — so the pin is
        durable without this method knowing anything about storage.
        """
        if not student_id:
            return
        with self._lock:
            self.child.pin(student_id=student_id, label=label, gender=gender)

    def forget_child(self) -> None:
        """Drop the pin.

        For the one case that is real evidence it is wrong: the records facade refusing
        this guardian. A transient outage must NOT come here — a hint the reader
        re-checks anyway is not worth discarding over a timeout, and doing so would
        re-ask the parent for no reason they could see.
        """
        with self._lock:
            self.child.clear()

    def note_tool_outcome(self, tool: str, outcome: str) -> None:
        """Record what a tool call actually came back with.

        The counterpart to the graph's `tool_calls_made`, which knows how MANY times a
        tool ran and nothing about what happened. Both halves are needed for the same
        reason: a turn that called the records tool once and a turn that called it once
        and got a child's marks are different facts, and only the second one contradicts
        an answer saying no record exists.

        Reported BY the tool, with the same string it hands its own template, rather than
        read back out of the rendered text. Parsing the render would put the outcome in
        two places — a branch here and a sentence in
        `backend/prompts/templates/tools/records_result.j2` — that could be edited apart,
        and the wording is translated copy that is expected to change.
        """
        if not tool or not outcome:
            return
        with self._lock:
            if not self._active:
                return
            self._tool_outcomes.append((str(tool), str(outcome)))

    @property
    def tool_outcomes(self) -> list:
        """Every `(tool, outcome)` this turn produced, in the order they happened."""
        with self._lock:
            return list(self._tool_outcomes)

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

    def note_surfaced_assets(self, asset_ids) -> None:
        """Record assets that retrieval put in front of the model.

        This is the whole of the query-time image feature: the figure was read into
        text at ingest, so the model answers from the chunk and never looks at pixels.
        These ids are what a citation is resolved against when the turn decides which
        picture to attach (backend/chat/assets_bridge.py).
        """
        with self._lock:
            if not self._active:
                return
            for asset_id in asset_ids or []:
                if asset_id and asset_id not in self._surfaced_asset_ids:
                    self._surfaced_asset_ids.append(asset_id)

    def surfaced_asset_ids(self) -> list:
        with self._lock:
            return list(self._surfaced_asset_ids)

    def note_duplicate_tool_calls(self, count: int) -> None:
        """Record calls the model asked for more than once. Diagnostic only."""
        if count <= 0:
            return
        with self._lock:
            self._duplicate_tool_calls += int(count)

    @property
    def duplicate_tool_calls(self) -> int:
        return self._duplicate_tool_calls

    def remembered_retrieval(self, key: str):
        """A retrieval already run for `key` this turn, or None."""
        if not key:
            return None
        with self._lock:
            return self._retrieval_memo.get(key)

    def remember_retrieval(self, key: str, result) -> None:
        """Keep a retrieval so an identical query this turn does not run it again.

        Request-scoped rather than cached in Redis, and that is the whole design: two
        identical searches within ONE turn are the same question asked twice and must
        give the same answer, while the same question next week must be answered from
        whatever the corpus says then. A turn is exactly the window where reuse is
        certainly correct, so it is the window this holds.
        """
        if not key:
            return
        with self._lock:
            if self._active:
                self._retrieval_memo[key] = result

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
