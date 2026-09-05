"""The last thing between what the model emitted and what a parent reads.

Every answer crosses this stage, on both paths — the streamed one and the synchronous
one — and it is the only place allowed to decide that something the model produced does
not reach the user. Before it, that decision was made inline at the stream's edge by a
single line in `service.py`:

    if getattr(msg, "tool_call_chunks", None):
        continue

which was the right intent at the wrong granularity. It skipped the chunks that CARRIED
a tool-call delta, but the model's prose arrives in different chunks of the same
message — measured at 2 tool-call chunks against 136 content chunks on one call — so
every one of those 136 was forwarded. That is how a parent was shown an answer the model
wrote before its tool had run.

## Why a stage and not a graph node

`backend/chat/orchestrator.py` states the reason at length and it holds here: the agent
is LangChain's prebuilt `create_agent`, and wrapping it in a hand-built StateGraph to
gain one post-step would mean re-implementing tool execution, streaming and recursion
limits. So this is a stage the two callers drive rather than a node the graph runs. What
matters is that it is SINGLE — one object, holding the whole rule, that both paths must
pass through — not which scheduler turns the handle.

## What it decides

Two rules today, and they cover two failures that look identical in a transcript and are
not:

  **A message that is calling a tool has not answered anything.** Its prose was written
  before the evidence existed; it is the model guessing what the tool will return. Every
  unmarked reasoning leak measured arrived this way, and so did a complete, confident,
  entirely invented Arabic answer that a parent saw. Content from such a message is
  dropped whole — no text rule needed, and none would have worked, because that shape
  carries no marker to match.

  **A message that is answering may still be wearing its transcript.** The provider's
  Harmony parser leaks the analysis channel into `content` (4 of 5 calls, measured), and
  on an answering message the content cannot simply be dropped — it holds the answer. So
  it is parsed instead, and only the final channel survives. See
  `backend/chat/model_output.py`.

Dropping tool-call content costs no streaming latency. The tool-call delta arrives
FIRST — at chunk 41, 45 and 72 across three runs, with zero characters of content ahead
of it — so a message identifies itself before it says anything, and nothing has to be
buffered to find out.

## Where the rest of the answer-shaping belongs

Here. This module exists so that the checks that follow have somewhere to live that is
not the middle of a streaming loop: whether an answer cites chunks on a turn where no
tool ran, whether citation markers point at retrieved evidence, whether an answer about
a child contradicts the year the roster reported. Each is a rule about the finished
answer rather than about the model's format, and each gets a method here and a line in
`as_trace()`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import backend.chat.grounding as grounding
from backend.chat.grounding import DEFAULT_FLOOR, GroundingReport
from backend.chat.model_output import HarmonyFilter, strip_harmony

logger = logging.getLogger(__name__)


def message_text(msg: Any) -> str:
    """The text of a message, whatever shape its content arrived in.

    Providers return either a string or a list of typed blocks, and the two have to be
    read the same way everywhere or a list-shaped response silently becomes `str(...)`
    of a Python list in front of a user.
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, str):
                text += block
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        return text
    return str(content or "")


class Finalizer:
    """One turn's answer, assembled from the model's stream under the rules above.

    Stateful and single-use: it tracks which message is being streamed, whether that
    message declared itself a tool call, and how much of the Harmony transcript has been
    consumed. Construct one per turn.
    """

    __slots__ = (
        "_filters",
        "_tool_calling",
        "_current",
        "_answer",
        "_dropped_messages",
        "_dropped_chars",
        "_harmony_messages",
        "_tool_results",
        "_tool_texts",
        "_grounding",
    )

    def __init__(self) -> None:
        self._filters: Dict[str, HarmonyFilter] = {}
        self._tool_calling: Dict[str, bool] = {}
        self._current: Optional[str] = None
        self._answer = ""
        self._dropped_messages = 0
        self._dropped_chars = 0
        self._harmony_messages = 0
        self._tool_results = 0
        self._tool_texts: list = []
        self._grounding: Optional[GroundingReport] = None

    # -- streaming ---------------------------------------------------------------

    def consider(self, msg: Any) -> str:
        """Take one `AIMessageChunk`; return the text a user may see, often empty.

        The returned text may include the held-back tail of the PREVIOUS message, since
        a message is only known to be over when the next one starts.
        """
        key = self._key(msg)
        released = ""
        if key != self._current:
            released = self._close_current()
            self._current = key

        if getattr(msg, "tool_call_chunks", None):
            # The message has identified itself. Everything it says, before or after
            # this chunk, is pre-tool narration.
            self._tool_calling[key] = True

        text = message_text(msg)
        if not text:
            return released

        if self._tool_calling.get(key):
            self._dropped_chars += len(text)
            return released

        emitted = self._filter_for(key).feed(text)
        self._answer += emitted
        return released + emitted

    def finish(self) -> str:
        """Close the stream. Returns whatever the last message was still holding."""
        released = self._close_current()
        self._current = None
        return released

    def _close_current(self) -> str:
        key = self._current
        if key is None:
            return ""
        if self._tool_calling.get(key):
            self._dropped_messages += 1
            self._filters.pop(key, None)
            return ""
        harmony = self._filters.pop(key, None)
        if harmony is None:
            return ""
        if harmony.saw_markup:
            self._harmony_messages += 1
        tail = harmony.flush()
        self._answer += tail
        return tail

    def _filter_for(self, key: str) -> HarmonyFilter:
        harmony = self._filters.get(key)
        if harmony is None:
            harmony = HarmonyFilter()
            self._filters[key] = harmony
        return harmony

    @staticmethod
    def _key(msg: Any) -> str:
        # Chunks of one model call share an id, and the agent's two calls do not — which
        # is what lets a tool-calling message be told apart from the answer that follows
        # it. A provider that omits ids collapses every chunk into one message, which is
        # the old behaviour and still correct, just less precise.
        return str(getattr(msg, "id", "") or "")

    # -- the whole-response path -------------------------------------------------

    def note_tool_result(self, msg: Any = None) -> None:
        """Record that a tool actually returned something this turn, and what it said.

        The count is what tells an answer citing `[1]` on a turn where no tool ran from
        one that has something to point at — the case that put an invented fee in front
        of a parent.

        The TEXT is kept for the check that came after it. A tool result is the only
        record of what this turn actually read, and for `get_student_records` it is the
        only one there will ever be: it writes nothing into the RAG trace, so a turn that
        fetched «الرياضيات ٨٧.٥٪» and then said something else about it could be compared
        against nothing. Held in memory for the length of one turn and never persisted —
        see `as_trace`, which counts these and quotes none of them, because the string
        contains a real child's name and their marks.
        """
        self._tool_results += 1
        text = message_text(msg) if msg is not None else ""
        if text:
            self._tool_texts.append(text)

    @property
    def tool_result_texts(self) -> list:
        """What each tool returned this turn, in the order they returned it."""
        return list(self._tool_texts)

    @property
    def answer(self) -> str:
        """Everything this turn emitted, after filtering."""
        return self._answer

    @property
    def tool_results(self) -> int:
        return self._tool_results

    @property
    def grounding(self) -> Optional[GroundingReport]:
        """The verdict, once `verify` has run. None means it was never asked for."""
        return self._grounding

    def verify(
        self,
        evidence,
        *,
        floor: int = DEFAULT_FLOOR,
        check_citations: bool = True,
    ) -> GroundingReport:
        """Check the assembled answer against the evidence the turn retrieved.

        Runs on the finished answer rather than per chunk, because the unit being
        checked is a claim and a claim is not complete until its sentence is. What the
        caller does with a failing verdict is the profile's decision, not this object's
        — see `agent.answer_grounding_mode`.

        What the tools returned is added by this object rather than asked of the caller.
        Both entry points into the check would otherwise have to remember to pass it, and
        the one that forgot would silently stop checking a whole class of answer — which
        is the exact shape of the bug this parameter exists to close.
        """
        report = grounding.verify(
            self._answer,
            evidence,
            floor=floor,
            check_citations=check_citations,
            extra_evidence=self._tool_texts,
        )
        self._grounding = report
        return report

    def replace_answer(self, replacement: str) -> str:
        """Discard what was assembled and stand `replacement` in its place.

        The turn's stored message has to match what the reader was left with, or the
        next turn's history carries an answer nobody ever saw.
        """
        self._answer = replacement
        return replacement

    def as_trace(self) -> Dict[str, Any]:
        """What this stage did, for the trace and for LangSmith.

        Worth carrying even when it did nothing: "the model emitted no transcript
        markup this turn" and "this stage was never consulted" look the same in a log
        that only records action, and they are the two states a provider switch moves
        between.
        """
        trace = {
            "finalize_dropped_tool_call_messages": self._dropped_messages,
            "finalize_dropped_chars": self._dropped_chars,
            "finalize_harmony_messages": self._harmony_messages,
            "finalize_tool_results": self._tool_results,
        }
        if self._grounding is not None:
            trace.update(self._grounding.as_trace())
        return trace

    def log_summary(self) -> None:
        if self._dropped_messages or self._harmony_messages:
            logger.info(
                "finalize: dropped %d tool-call message(s) (%d chars), "
                "stripped transcript markup from %d message(s)",
                self._dropped_messages,
                self._dropped_chars,
                self._harmony_messages,
            )
        if self._grounding is not None and not self._grounding.ok:
            # Warning, not info: every one of these is either an answer that was about
            # to state an invented figure, or a corpus this check reads badly. Both need
            # somebody to look.
            logger.warning("finalize: answer failed grounding — %s", self._grounding.reason)


def finalize_text(text: str, *, has_tool_calls: bool = False) -> str:
    """The same rules, for a response that arrived complete rather than streamed.

    The synchronous path in `chat_with_agent` receives a finished message and never sees
    chunks, so it cannot use `Finalizer` — but it must not therefore be the path where a
    transcript reaches a user. Same two rules, applied once.
    """
    if has_tool_calls:
        return ""
    return strip_harmony(text or "")


__all__ = ["Finalizer", "finalize_text", "message_text"]
