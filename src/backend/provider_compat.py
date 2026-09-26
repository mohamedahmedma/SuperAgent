"""One workaround for one measured bug in the provider's own protocol handling.

## What is wrong

`openai/gpt-oss-*` speaks the OpenAI **Harmony** format, which carries three channels in
one response: `analysis` for the model's private reasoning, `commentary` for tool calls,
and `final` for the answer. The serving layer is supposed to parse those and hand back
the final channel in `content` with the reasoning in a separate `reasoning` field —
Together documents exactly that, and it is what the endpoint does:

    delta fields ['content', 'reasoning', 'role']    reasoning PRESENT    content clean

Until the conversation contains a single `role: "tool"` message. Then, on the very next
call, the `reasoning` field disappears from the response entirely and the raw transcript
comes through inside `content`:

    delta fields ['content', 'role']                 reasoning ABSENT     content:
        <|channel|>analysis<|message|>We have a chunk: "رسوم الصف الأول…"

That is the whole bug, and it is the provider's parser failing rather than the model
misbehaving. It was isolated by elimination against the live endpoint; every one of these
was held constant while the tool result was present, and none of them changed the outcome:

    the system prompt moved to a `developer` message   still broken
    the vendor's own temperature=1.0, top_p=1.0        still broken
    a three-line system prompt instead of 1,685 chars  still broken
    no system prompt at all                            still broken
    tools removed from the request entirely            still broken
    openai/gpt-oss-120b instead of 20b                 still broken

and the control — the documented single-turn example — reproduces cleanly on both models.
So the trigger is the message role, nothing else. Both models leak; 120b tends to leak
with its delimiters eaten (`analysisThe user asks:`) and 20b with them intact
(`<|channel|>analysis<|message|>`), which is the same parser giving up at two different
points and is why one endpoint produced several apparent "shapes" of the same fault.

## What this does about it

Rewrites the outgoing request so no `role: "tool"` message is ever sent. Each tool result
becomes ordinary text in a user message, and the assistant turn that made the call is
restated as text — what it DID is kept, what it THOUGHT is discarded. Measured on the same
question, same retrieved chunk, two rounds of tools:

    role:"tool" messages   reasoning ABSENT    content 495 ch, leaking
    folded to text         reasoning PRESENT   content 113 ch, clean, both facts kept

Discarding the original content is not incidental: it is the narration the model wrote
*before* its tool ran, and the gpt-oss model card is explicit that reasoning from previous
turns must not be replayed. Replaying it teaches the model that reasoning belongs in
`content`, which makes the leak worse the longer a conversation runs — so the same edit
that fixes the parser also stops feeding the failure.

Keeping the turn is equally deliberate, and was learned the expensive way. An earlier
version dropped it outright; the model then had no record of its own call, and against an
empty result it called the same tool 85 times in one turn until the recursion limit ended
it. `_describe` is what restores that record without restoring the reasoning.

## Why it is applied to the payload rather than to the graph

The agent decides what to do next by looking at its own state — an `AIMessage` carrying
tool calls is how it knows a tool ran. Rewriting that state would be rewriting the agent's
memory of its own turn, and the tool loop, the checkpointer and the stored transcript all
read it. So the fold happens at the last possible moment, on the dictionary that is about
to become JSON: LangGraph keeps a correct conversation, and only the wire changes.

This is a workaround for somebody else's bug and it is written to be deleted. When the
endpoint parses its own format on this path, `test_provider_compat.py` starts failing on
the case that pins the broken behaviour, and that is the signal to remove this file.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

#: How a tool result is introduced once it is no longer a `role: "tool"` message. Named
#: after the tool that produced it, because the model otherwise has no way to tell which
#: of several results answers which part of the question.
CONTEXT_HEADING = "Result from {name}:"

#: The role the folded result is sent as. `user` and `system` both restore the parser;
#: `user` is used because a retrieved chunk is material for this turn's answer rather
#: than a standing instruction, and because consecutive user messages are ordinary in
#: this format while a second system message is not.
FOLDED_ROLE = "user"


def fold_tool_results_into_text(model: Any) -> Any:
    """Send tool results as text, so the provider keeps parsing its own format.

    Wraps the model's request-payload builder — the one point that sees every message on
    every call, including the ones LangGraph replays out of state, which is where the
    offending message comes from.

    Returns the same object, patched in place, so provider selection stays where it
    belongs in `backend/llm_provider.py` and this composes with `init_chat_model`.
    """
    # Asked of the CLASS, not the instance: `init_chat_model` returns a lazy configurable
    # model when no model id is set, and that object answers any unknown attribute by
    # building the real model — so probing the instance turns "no model configured" from
    # a no-op into an import-time error in every module that imports the agent.
    if not hasattr(type(model), "_get_request_payload"):
        logger.debug("provider_compat: %s exposes no payload hook", type(model).__name__)
        return model

    builder = model._get_request_payload

    def _patched(input_: Any, *, stop: Any = None, **kwargs: Any) -> dict:
        payload = builder(input_, stop=stop, **kwargs)
        messages = payload.get("messages")
        if messages:
            payload["messages"] = fold_messages(messages)
        return payload

    object.__setattr__(model, "_get_request_payload", _patched)
    return model


def fold_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One conversation, with every tool result turned into text.

    Pure: takes the outgoing message dictionaries and returns new ones. A conversation
    with no tool result passes through unchanged and is returned as the same objects, so
    the common case — every turn before the first tool runs — pays nothing.
    """
    if not any(_is_tool_result(m) or _is_tool_call(m) for m in messages):
        return messages

    # The tool's NAME lives on the assistant turn that requested it, and that turn is
    # about to be dropped, so it is collected first.
    names: Dict[str, str] = {}
    for message in messages:
        for call in _tool_calls(message):
            call_id = call.get("id")
            name = (call.get("function") or {}).get("name")
            if call_id and name:
                names[call_id] = name

    folded: List[Dict[str, Any]] = []
    last_folded = -1
    for message in messages:
        if _is_tool_call(message):
            # The turn is KEPT, as plain text, and its original content is thrown away.
            #
            # Both halves matter. The content is what the model said BEFORE its tool ran
            # — a guess at the answer, and the thing the leak is made of — and the model
            # card is explicit that reasoning from earlier turns must not be replayed.
            # But dropping the turn outright removes the model's only record that it
            # already called the tool, and a model that cannot see its own call makes it
            # again: measured at 85 consecutive `search_knowledge_base` calls in one turn
            # against an empty result, until the recursion limit ended it. So what it did
            # is restated, and only what it thought is discarded.
            folded.append({"role": "assistant", "content": _describe(_tool_calls(message))})
            continue

        if _is_tool_result(message):
            name = names.get(message.get("tool_call_id")) or "the tool"
            block = CONTEXT_HEADING.format(name=name) + "\n" + str(message.get("content") or "")
            if last_folded == len(folded) - 1 and last_folded >= 0:
                # Results that arrived together stay together, rather than becoming a run
                # of separate user turns that reads like the parent said all of them.
                folded[last_folded]["content"] += "\n\n" + block
            else:
                folded.append({"role": FOLDED_ROLE, "content": block})
                last_folded = len(folded) - 1
            continue

        folded.append(message)

    return folded


def _describe(calls: List[Dict[str, Any]]) -> str:
    """What the assistant turn said it was doing, without what it thought about it.

    The arguments are included because "I searched the knowledge base" and "I searched
    the knowledge base for the fees" lead to different next moves when the result comes
    back empty — the second tells the model to widen the query rather than repeat it.
    """
    described = []
    for call in calls:
        function = call.get("function") or {}
        name = function.get("name") or "a tool"
        arguments = (function.get("arguments") or "").strip()
        described.append(f"{name}({arguments})" if arguments else name)
    return "Called " + ", ".join(described) + "."


def _tool_calls(message: Any) -> List[Dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    return message.get("tool_calls") or []


def _is_tool_call(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "assistant" and bool(_tool_calls(message))


def _is_tool_result(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "tool"


__all__ = ["CONTEXT_HEADING", "FOLDED_ROLE", "fold_messages", "fold_tool_results_into_text"]
