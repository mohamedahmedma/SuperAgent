"""One chat turn, from loading the conversation to saving the answer.

The two entry points — `chat_with_agent`, which returns a dict, and
`chat_with_agent_stream`, which streams events to the web app — each used to carry the
whole turn, written out twice. They drifted: a rule added to one did not always reach the
other, and two of those gaps reached people (see `test_turn_entry_point_parity.py`).

`TurnPipeline` makes every decision a turn involves, once. What the entry points still own
is only how a turn is DELIVERED: whether the agent is invoked or streamed, which calls
leave the event loop, and when each event goes on the wire relative to the save. Neither
can decide something differently from the other, because neither decides anything.

The collaborators a turn reaches for — the planner, the agent factory, the models, the
note writer — arrive as `TurnCollaborators` rather than being imported here, so a turn can
be assembled from stand-ins without patching a module.

## What the parent waits for, and what they do not

Storing the turn and updating the persistent note change nothing the parent is shown, so
neither runs on the request. Both are queued on `TurnCollaborators.background`
(`backend/chat/background.py`) — the save before the stream's last event is sent, the
note behind it — and the request is free to end. The next turn on the same conversation
waits for that queue before it reads the conversation, which is what keeps "queued" and
"stored" indistinguishable from where the parent sits.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.chat.answer_blocks import attach_answer_blocks, resolve_figure_markers, settle_answer_blocks
from backend.chat.answer_checks import (
    enforce_forced_tool_ran,
    enforce_records_agreement,
    nothing_usable_reply,
    resumed_static_reply,
    terminal_reply,
)
from backend.chat.assets_bridge import (
    asset_ids_for_answer,
    attach_assets_to_trace,
    build_asset_references,
    effective_capabilities,
    trace_for_storage,
)
from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_context import load_child_state, save_child_state
from backend.chat.clarification import (
    PENDING_HITL_KEY,
    TurnEntry,
    build_pending_hitl,
    child_choice_pending,
    enter_turn,
    format_hitl_message,
    is_hitl_trace,
    pin_the_child_the_parent_named,
)
from backend.chat.context_messages import build_context_messages, build_resume_answer_messages
from backend.chat.background import MODELS, JobRunner
from backend.chat.finalize import Finalizer, finalize_text, message_text
from backend.chat.storage import MessageToStore
from backend.schemas.chat import normalize_rag_trace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnCollaborators:
    """Everything a turn reaches outside itself for.

    Each of these was a module global in `service.py`, and each is a seam some test
    substitutes. `service.py` still assembles the real ones from its own globals at call
    time, which is why a test that patches one there keeps reaching the turn it drives.
    """

    conversations: Any
    #: Where the save and the note update run: `BackgroundJobs`, or `InlineJobs` for a
    #: caller that wants both to have happened when the turn returns.
    background: JobRunner
    profile: Any
    plan: Callable[..., tuple]
    resolve_question: Callable[..., Any]
    create_agent: Callable[..., Any]
    resume_retrieval: Callable[..., dict]
    answer_model: Any
    session_title: Callable[[str], str]
    update_note: Callable[..., str]
    context_type: Any


@dataclass
class Turn:
    """One turn's state, from the moment its conversation is loaded to the save."""

    user_text: str
    user_id: str
    session_id: str
    caller: CallerIdentity
    messages: list
    metadata: dict
    child_state: Any
    persistent_note: str
    is_first_message: bool
    #: The conversation before this message, for the direct answer a resumed search gets.
    history: list
    entry: TurnEntry | None = None
    ctx: Any = None
    plan: Any = None
    signals: Any = None
    title: str = ""
    rag_trace: dict | None = None
    next_pending: dict | None = None
    answer: str = ""
    answer_blocks: list = field(default_factory=list)
    asset_references: list = field(default_factory=list)
    agent_error: str | None = None
    #: Whether the answer has been handed to storage. An interrupted turn stores what had
    #: reached the parent — unless the turn had already stored its answer.
    committed: bool = False

    def asset_payload(self) -> list:
        return [
            reference.model_dump(mode="json", exclude_none=True)
            for reference in self.asset_references
        ]


@dataclass(frozen=True)
class StreamedAnswer:
    """An agent run whose chunks were read as they arrived."""

    text: str
    finalizer: Finalizer

    def nothing_usable(self, turn: Turn, copy) -> str:
        return nothing_usable_reply(self.finalizer, turn.plan)


@dataclass(frozen=True)
class InvokedAnswer:
    """An agent run that returned all at once.

    Its finalizer never saw the chunks, so it has no count of what it withheld. Whether
    the model produced nothing but transcript is measured instead by comparing the answer
    before and after the strip — the same question `nothing_usable_reply` asks, read from
    the only evidence this path has.
    """

    text: str
    finalizer: Finalizer
    withheld_everything: bool

    def nothing_usable(self, turn: Turn, copy) -> str:
        if not self.withheld_everything:
            return ""
        logger.warning("the model produced no answer channel this turn; serving the retry copy")
        return copy.retrieval_error


@dataclass(frozen=True)
class AnswerSettlement:
    """What settling an agent's answer changed, for an entry point that must show it.

    The streamed path has already put the model's words on the wire, so it needs to know
    what changed: `replacement` when the answer was withdrawn, `settled` when records were
    placed under it. The sync path reads the final answer off the turn and ignores this.
    """

    asking: bool = False
    replacement: str = ""
    settled: str | None = None
    blocks: list = field(default_factory=list)


class TurnPipeline:
    """The decisions of one turn, in the order a turn makes them."""

    #: How long a turn waits at its start for the previous turn's save to land. Past it
    #: the turn goes on with what is stored — the alternative is a parent whose message
    #: hangs behind a database that has stopped answering — and says so in the log.
    SAVE_WAIT_SECONDS = 10.0

    def __init__(self, collaborators: TurnCollaborators) -> None:
        self._c = collaborators

    @property
    def collaborators(self) -> TurnCollaborators:
        return self._c

    @staticmethod
    def conversation_key(user_id: str, session_id: str) -> str:
        """The background queue a conversation's writes share, so they land in order."""
        return f"conversation:{user_id}:{session_id}"

    @staticmethod
    def note_key(user_id: str, session_id: str) -> str:
        """A queue of its own for the note: waiting for a save must never mean waiting
        for the model call behind it."""
        return f"note:{user_id}:{session_id}"

    # -- opening -----------------------------------------------------------------------

    def open(self, user_text: str, user_id: str, session_id: str, caller: CallerIdentity) -> Turn:
        """Load the conversation this message belongs to.

        After the previous turn's writes: its save was queued rather than waited for, and a
        parent's next message can arrive before a queued save has run.
        """
        if not self._c.background.flush(self.conversation_key(user_id, session_id), timeout=self.SAVE_WAIT_SECONDS):
            logger.warning(
                "the previous turn's save for %s/%s is still running after %.0fs; "
                "continuing with what is stored",
                user_id, session_id, self.SAVE_WAIT_SECONDS,
            )
        messages, metadata = self._c.conversations.load_with_meta(user_id, session_id)
        return Turn(
            user_text=user_text,
            user_id=user_id,
            session_id=session_id,
            caller=caller,
            messages=messages,
            metadata=metadata,
            # The pin travels by reference into the turn's context and back out into the
            # saved metadata, so a turn that resolves a child has already recorded it.
            child_state=load_child_state(metadata, guardian_id=caller.guardian_id if caller else ""),
            persistent_note=metadata.get("persistent_note", ""),
            is_first_message=len(messages) == 0,
            history=list(messages),
        )

    def enter(self, turn: Turn) -> None:
        """Read this message against the question the assistant is still waiting on.

        May cost a resolver call, which is why the streamed entry point runs it on a
        worker thread.
        """
        turn.entry = enter_turn(
            turn.user_text, list(turn.messages), turn.metadata, resolve=self._c.resolve_question
        )

    def sync_context(self, turn: Turn):
        return self._attach(turn, self._c.context_type.for_sync(
            user_id=turn.user_id,
            session_id=turn.session_id,
            caller=turn.caller,
            child=turn.child_state,
        ))

    def stream_context(self, turn: Turn, output_queue):
        return self._attach(turn, self._c.context_type.for_stream(
            user_id=turn.user_id,
            session_id=turn.session_id,
            output_queue=output_queue,
            caller=turn.caller,
            child=turn.child_state,
        ))

    def _attach(self, turn: Turn, ctx):
        turn.ctx = ctx
        ctx.reset_knowledge_tool_budget()
        # Settled before anything plans this turn, so the pin is in place by the time the
        # planner resolves a child.
        if turn.entry.child_choice:
            pin_the_child_the_parent_named(ctx, turn.entry.child_choice)
        return ctx

    def record_question(self, turn: Turn) -> None:
        turn.messages.append(HumanMessage(content=turn.user_text))
        self._store(turn, [MessageToStore("human", turn.user_text)], describe="store the question")

    def _store(self, turn: Turn, messages: list, *, metadata: dict | None = None, describe: str) -> None:
        """Queue an append to this conversation, behind whatever it already has queued."""
        conversations = self._c.conversations
        user_id, session_id = turn.user_id, turn.session_id

        def work():
            conversations.append(user_id, session_id, messages, metadata=metadata)

        self._c.background.submit(
            self.conversation_key(user_id, session_id), work, describe=f"{describe} ({user_id}/{session_id})"
        )

    # -- a resumed search --------------------------------------------------------------

    @staticmethod
    def resumes_a_search(turn: Turn) -> bool:
        return bool(turn.entry.is_hitl_resume and turn.entry.resume_state)

    def run_resumed_search(self, turn: Turn) -> dict:
        return self._c.resume_retrieval(
            turn.entry.pending_hitl, turn.user_text, turn.ctx, turn.entry.resolution
        )

    def settle_resumed_search(self, turn: Turn, rag_result) -> list | None:
        """Decide what a resumed search leaves the turn to say.

        Returns the prompt for a direct model answer from the retrieved documents, or None
        when the answer is already settled on `turn`: another question, or copy for a
        search with nothing to answer from.
        """
        turn.rag_trace = normalize_rag_trace(
            rag_result.get("rag_trace") if isinstance(rag_result, dict) else None
        )
        if is_hitl_trace(turn.rag_trace):
            self._ask(turn, rag_result.get("hitl_resume_state"))
            return None
        static_reply = resumed_static_reply(rag_result)
        if static_reply is not None:
            turn.answer = static_reply
            return None
        return build_resume_answer_messages(
            turn.entry.pending_hitl,
            turn.user_text,
            rag_result.get("docs") or [],
            resolved_question=rag_result.get("question") or "",
            constraints=_resume_constraints(rag_result),
            history=turn.history,
        )

    def _ask(self, turn: Turn, resume_state) -> None:
        turn.next_pending = build_pending_hitl(
            turn.rag_trace,
            turn.entry.original_question or turn.user_text,
            previous_answers=turn.entry.hitl_answers,
            resume_state=resume_state,
        )
        turn.answer = format_hitl_message(turn.next_pending["prompt"], turn.next_pending["options"])

    # -- planning ------------------------------------------------------------------------

    def plan(self, turn: Turn) -> None:
        turn.plan, turn.signals = self._c.plan(
            turn.entry.effective_user_text,
            turn.messages[:-1],
            turn.ctx,
            resolution=turn.entry.resolution,
        )

    def name_the_session(self, turn: Turn) -> None:
        if turn.is_first_message:
            turn.title = self._c.session_title(turn.user_text)

    def settle_short_circuit(self, turn: Turn) -> None:
        """The planned reply for a turn no agent is built for.

        A confirmed out-of-domain question, a social turn a profile answers statically, or
        a parent asked which of their children they meant — in which case the reply is a
        question, carried as the pending clarification the next message answers.
        """
        plan = turn.plan
        turn.next_pending = child_choice_pending(plan, turn.entry.original_question or turn.user_text)
        reply = plan.static_reply or ""
        turn.answer = (
            format_hitl_message(reply, turn.next_pending["options"]) if turn.next_pending else reply
        )
        turn.rag_trace = normalize_rag_trace({**plan.as_trace(), **turn.signals.as_trace()})
        self.name_the_session(turn)

    # -- the agent -----------------------------------------------------------------------

    def agent_call(self, turn: Turn) -> tuple:
        """The agent, the conversation it is shown, and the config it runs under."""
        agent = self._c.create_agent(turn.ctx, turn.plan.exposed_tools, turn.plan.language)
        messages = build_context_messages(
            turn.messages[:-1], turn.persistent_note, turn.entry.effective_user_text, turn.plan
        )
        config = {"recursion_limit": self._c.profile.agent.recursion_limit}
        return agent, messages, config

    def terminal_reply(self, turn: Turn) -> str | None:
        """The profile's reply when the graph ended itself on a terminal retrieval result.

        `_end_turn_on_terminal_retrieval` stops the loop before a second model call would
        reword profile copy, so nothing the model said stands as the answer.
        """
        status = turn.ctx.short_circuit_status()
        return terminal_reply(status, turn.plan.language) if status else None

    @staticmethod
    def is_asking_a_question(turn: Turn) -> bool:
        """Whether retrieval has decided this turn ends in a question, not an answer.

        Read from the live trace, because the knowledge tool decides it part-way through
        the turn — after the stream has already started.
        """
        stored = turn.ctx.peek_rag_trace()
        return is_hitl_trace(normalize_rag_trace(stored.get("rag_trace") if stored else None))

    def read_invoked_answer(self, turn: Turn, result) -> InvokedAnswer:
        """An `invoke` result, read into what settling the answer needs."""
        text = _text_of(result)
        reply = self.terminal_reply(turn)
        if reply is not None:
            text = reply
        # The agent loop has ended, so the last message answered rather than called a
        # tool — but it may still be wearing its transcript. See `backend/chat/finalize.py`.
        raw = text
        text = finalize_text(text)
        finalizer = Finalizer()
        # The streamed path sees each tool result go past; here the whole conversation is
        # in hand at once, so they are replayed off it. Without them the records check had
        # no record of a child's marks and passed every answer about them.
        for tool_message in _tool_messages_in(result):
            finalizer.note_tool_result(tool_message)
        finalizer.replace_answer(text)
        return InvokedAnswer(
            text=text,
            finalizer=finalizer,
            withheld_everything=bool(raw.strip()) and not text.strip(),
        )

    def settle_agent_answer(self, turn: Turn, answer: StreamedAnswer | InvokedAnswer) -> AnswerSettlement:
        """Settle what the agent said: a question, a replacement, or the answer with its records.

        The evidence is in hand for the first time here, so this is where the answer can be
        checked against it. A failed check replaces the answer rather than appending to it —
        the reader of a streamed turn has already seen the words being withdrawn.
        """
        finalizer = answer.finalizer
        stored = turn.ctx.take_rag_trace()
        turn.rag_trace = normalize_rag_trace(stored.get("rag_trace") if stored else None)
        if is_hitl_trace(turn.rag_trace):
            self._ask(turn, stored.get("hitl_resume_state") if stored else None)
            return AnswerSettlement(asking=True)

        replacement = (
            enforce_records_agreement(finalizer, turn.ctx, turn.plan)
            or enforce_forced_tool_ran(finalizer, turn.ctx, turn.plan)
        )
        if not replacement and not answer.text.strip():
            replacement = answer.nothing_usable(turn, self._c.profile.user_copy)
        if replacement:
            turn.answer = finalizer.replace_answer(replacement)
            finalizer.log_summary()
            return AnswerSettlement(replacement=replacement)

        # Nothing was withheld, so the record goes under the sentence — as the TOOL rendered
        # it, never as the model retyped it — after the figure markers resolve, and never
        # under a refusal, which is why this comes after the checks above.
        settled, blocks = settle_answer_blocks(resolve_figure_markers(answer.text, turn.ctx), turn.ctx)
        changed = settled != answer.text
        turn.answer = finalizer.replace_answer(settled) if changed else answer.text
        turn.answer_blocks = blocks
        turn.rag_trace = attach_answer_blocks(turn.rag_trace, blocks)
        finalizer.log_summary()
        return AnswerSettlement(settled=settled if changed else None, blocks=blocks)

    def record_finalize_stage(self, turn: Turn, answer: StreamedAnswer | InvokedAnswer) -> None:
        """What the finalize stage withheld, on every turn it ran.

        Not only the failing ones: "nothing was dropped" and "the stage never ran" are
        different facts, and a provider switch is exactly when telling them apart matters.
        """
        if turn.rag_trace:
            turn.rag_trace.update(answer.finalizer.as_trace())

    # -- shared by every path ------------------------------------------------------------

    def attach_assets(self, turn: Turn, client_capabilities, *, answer: str | None = None) -> None:
        """The images this turn surfaced, rendered for what the client can display.

        `answer` is the text the references are read from when it is not the turn's final
        answer — on a streamed turn that ended in a question, what the model had said before
        the question replaced it.
        """
        delivery = self._c.profile.assets.delivery
        turn.asset_references = build_asset_references(
            asset_ids_for_answer(turn.answer if answer is None else answer, turn.ctx, turn.rag_trace, delivery),
            effective_capabilities(client_capabilities, delivery),
            delivery,
        )
        turn.rag_trace = attach_assets_to_trace(turn.rag_trace, turn.asset_references)

    def save_metadata(self, turn: Turn, *, interrupted: bool = False) -> dict:
        """The session metadata this turn changed — a patch, never the whole record.

        Only the keys the turn decided are written, and they are merged in the database
        (`ConversationStorage.append`). A copy of the metadata as it stood when the turn
        began used to be written back whole, which put every key a concurrent writer had
        changed in between — the note, another turn's pending question — back the way this
        turn had found it.
        """
        patch: dict = {}
        save_child_state(patch, turn.child_state)
        if turn.entry.invalid_pending_hitl:
            patch[PENDING_HITL_KEY] = None
        if turn.title:
            patch["title"] = turn.title
        if turn.next_pending:
            patch[PENDING_HITL_KEY] = turn.next_pending
        elif turn.entry.spends_the_pending_question(agent_error=bool(turn.agent_error) or interrupted):
            # Answered, replaced, or settled by naming a child — every way a clarification
            # ends is decided in one place. See `TurnEntry`.
            patch[PENDING_HITL_KEY] = None
        return patch

    def note_is_due(self, turn: Turn) -> bool:
        """Whether this turn pays for persistent-note maintenance.

        Not when the turn ends in a question, whose answer is still to come. Not on a
        short-circuited turn either: a reply the corpus never saw contributes nothing worth
        summarising, and updating the note would spend a model call on the one path whose
        point is not making one. Otherwise, once the short-term context starts trimming.
        """
        if turn.next_pending or (turn.plan is not None and turn.plan.short_circuit):
            return False
        window = self._c.profile.agent.context_window_messages
        return bool(turn.persistent_note) or len(turn.messages) > window

    def schedule_note(self, turn: Turn) -> bool:
        """Queue the persistent-note update, when this turn is due one.

        A model call, so it runs behind the turn rather than in it — the parent has their
        answer, and the note is for the turns after this one. Queued before `commit` so
        the history it summarises is the conversation up to this message, which is what
        the note writer has always been shown.

        The note it builds on is read when the job RUNS, not copied from when the turn
        began: another turn may have finished in between and folded itself in, and a
        summary written over its work would drop it. Jobs for one conversation's note run
        in order, so two turns cannot race each other to the write either.
        """
        if not self.note_is_due(turn):
            return False
        conversations, update_note = self._c.conversations, self._c.update_note
        user_id, session_id = turn.user_id, turn.session_id
        user_text, answer = turn.entry.effective_user_text, turn.answer
        # The whole conversation only when there is no note yet to build on — a note is
        # bootstrapped from the history the window has already trimmed.
        history = list(turn.messages[:-1]) if not turn.persistent_note else None

        def work():
            current = str(conversations.session_metadata(user_id, session_id).get("persistent_note") or "")
            note = update_note(current, user_text, answer, history_messages=history)
            if note and note != current:
                conversations.patch_metadata(user_id, session_id, {"persistent_note": note})

        self._c.background.submit(
            self.note_key(user_id, session_id),
            work,
            # A model call: on the lane for them, so a run of note updates cannot hold the
            # threads the saves need.
            lane=MODELS,
            describe=f"update the persistent note ({user_id}/{session_id})",
        )
        return True

    def commit(self, turn: Turn, save_meta: dict) -> None:
        """Queue the answer and this turn's metadata for storage.

        The stored copy keeps its assets as ids rather than renditions — the renditions on
        the wire were built for the client that asked, and rebuilding them on load is a
        keyed lookup (`trace_for_storage`).
        """
        turn.messages.append(AIMessage(content=turn.answer))
        turn.committed = True
        self._store(
            turn,
            [MessageToStore("ai", turn.answer, rag_trace=trace_for_storage(turn.rag_trace))],
            metadata=save_meta,
            describe="store the answer",
        )

    def commit_interrupted(self, turn: Turn, shown: str) -> None:
        """Store what the parent saw of an answer whose stream was cut off.

        Stop was pressed, or the connection dropped. The question was already stored, and
        a conversation that ends on a question with no answer reads as one the assistant
        never answered — so what had reached the parent is stored as the answer, marked as
        interrupted on its trace. A turn that had already stored its answer is left alone.
        The pending clarification, if there was one, is kept: the answer never got used,
        and the parent should be able to try it again.
        """
        if turn.committed:
            return
        stored = turn.ctx.peek_rag_trace() if turn.ctx is not None else None
        trace = dict((stored or {}).get("rag_trace") or {})
        trace["turn_interrupted"] = True
        turn.answer = shown
        turn.rag_trace = normalize_rag_trace(trace)
        self.commit(turn, self.save_metadata(turn, interrupted=True))

    def wait_for_save(self, turn: Turn, timeout: float | None = None) -> bool:
        """Block until this conversation's queued writes have run. False on timeout."""
        return self._c.background.flush(
            self.conversation_key(turn.user_id, turn.session_id),
            timeout=self.SAVE_WAIT_SECONDS if timeout is None else timeout,
        )

    @staticmethod
    def response(turn: Turn) -> dict:
        return {
            "response": turn.answer,
            "rag_trace": turn.rag_trace,
            "assets": turn.asset_payload(),
            "answer_blocks": turn.answer_blocks,
        }


def resolve_caller(caller: CallerIdentity | None, user_id: str) -> tuple[CallerIdentity, str]:
    """Settle identity once, at the top of a turn, for both entry points.

    Returns `(caller, user_id)` guaranteed to agree. A provided `caller` wins, because it
    came from a verified token while the positional `user_id` is only a convenience default;
    deriving one from the other here means no code further down has to wonder which is
    authoritative.
    """
    if caller is not None:
        return caller, caller.user_id
    return CallerIdentity.for_user(user_id), user_id


def _text_of(result) -> str:
    """The answer text of an agent run that returned all at once, whatever shape `invoke` gave."""
    if isinstance(result, dict):
        if "output" in result:
            return str(result["output"])
        if "messages" in result and result["messages"]:
            return message_text(result["messages"][-1])
        return str(result)
    if hasattr(result, "content"):
        return message_text(result)
    return str(result)


def _tool_messages_in(result) -> list:
    """Every tool result in a finished agent run, for the path that has no stream.

    Tolerant of every shape `invoke` can return, because a grounding check that raised on an
    unexpected result would take the answer down with it.
    """
    if not isinstance(result, dict):
        return []
    return [m for m in (result.get("messages") or []) if isinstance(m, ToolMessage)]


def _resume_constraints(rag_result: dict) -> list[str]:
    trace = rag_result.get("rag_trace") or {}
    return [str(item) for item in (trace.get("turn_carried_constraints") or [])]


__all__ = [
    "AnswerSettlement",
    "InvokedAnswer",
    "StreamedAnswer",
    "Turn",
    "TurnCollaborators",
    "TurnPipeline",
    "resolve_caller",
]
