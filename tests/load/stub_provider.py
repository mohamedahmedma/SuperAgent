"""A stand-in model provider with fixed, known latency — for load-testing OUR infrastructure.

Measuring the backend against a real provider measures the provider too: a slow afternoon
at Groq reads as a regression here, and a fast one as a fix. This serves the two
OpenAI-compatible endpoints the backend calls — `/v1/chat/completions` and
`/v1/embeddings` — with latencies you set, so whatever changes between two runs is ours.

It answers every call shape a turn makes, well enough that a turn runs END TO END —
planner, scope check, resolver, agent, knowledge tool, grader, streamed answer, save —
because a stub the backend rejects would measure an error path instead:

  * structured output (`response_format: json_schema`) — an instance of the requested
    schema, built from the schema itself, with the handful of fields that decide a turn's
    route pinned so every turn takes the ordinary knowledge-base path (in domain, simple,
    evidence sufficient). Any schema not listed still gets a valid instance.
  * tool calling — when tools are offered and none has run yet, a call to the knowledge
    tool (or the forced one) with the parent's question as the query; once a tool result
    is in the conversation, a streamed text answer.
  * embeddings — deterministic unit vectors derived from the text, so the same question
    always lands in the same place and the dense lane stays exercised.

Everything waits with `asyncio.sleep`, so the stub itself is never the bottleneck: a
thousand concurrent calls cost it nothing but memory.

    python -m tests.load.stub_provider --port 8900 \\
        --structured-ms 500 --ttft-ms 400 --token-ms 15 --tokens 60 --embed-ms 450

Latency defaults approximate a hosted provider seen from the production server: ~0.5 s
for a structured call, ~0.4 s to first token, and ~450 ms per embedding (Novita measured
from Contabo, RAG_FIX_PLAN item 39).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

KNOWLEDGE_TOOL = "search_knowledge_base"

# The fields that decide which way a turn goes, pinned so every turn takes the ordinary
# knowledge-base path. Keyed by the schema NAME the backend sends — the Pydantic class
# name. A schema not listed is still answered, from its own definition.
ROUTE = {
    "RequestEnvelope": {"scope": "in_domain", "needed_tools": [KNOWLEDGE_TOOL],
                        "about_child": False, "child_reference": "none"},
    "ScopeVerdict": {"scope": "in_domain"},
    "ResolvedQuery": {"intent": "standalone"},
    "ComplexityResult": {"complexity": "simple"},
    "EvidenceGrade": {"relevance": "strong", "answerability": "sufficient",
                      "ambiguity": "none", "route": "answer", "confidence": 0.9,
                      "supporting_chunks": [1, 2]},
}

ANSWER = ("According to the school's published material, the answer is set out in the "
          "handbook section on this topic. Fees are reviewed each year and the office can "
          "confirm the figure for the current term. ")


class Latency:
    structured_ms = 500.0
    toolcall_ms = 400.0
    ttft_ms = 400.0
    token_ms = 15.0
    tokens = 60
    embed_ms = 450.0
    embed_dim = 1024
    jitter = 0.10  # +/- fraction, so the stub is not unrealistically metronomic


def _wait(ms: float) -> float:
    spread = ms * Latency.jitter
    return max(0.0, (ms + random.uniform(-spread, spread)) / 1000.0)


# -- schema instances ----------------------------------------------------------------

def _resolve_ref(schema: dict, root: dict) -> dict:
    ref = schema.get("$ref")
    if not ref:
        return schema
    node: Any = root
    for part in ref.lstrip("#/").split("/"):
        node = node.get(part, {})
    return node


def instance(schema: dict, root: dict | None = None, *, question: str = "") -> Any:
    """A value valid against `schema`: first enum member, empty collections, zero numbers."""
    root = root or schema
    schema = _resolve_ref(schema, root)
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            options = [o for o in schema[key] if o.get("type") != "null"] or schema[key]
            return instance(options[0], root, question=question)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object" or "properties" in schema:
        return {name: instance(sub, root, question=question)
                for name, sub in (schema.get("properties") or {}).items()}
    if kind == "array":
        return []
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    if kind == "string":
        return question or "stub"
    return None


def structured_answer(response_format: dict, question: str) -> dict:
    spec = response_format.get("json_schema") or {}
    schema = spec.get("schema") or {}
    name = spec.get("name") or schema.get("title") or ""
    value = instance(schema, question=question) if schema else {}
    if isinstance(value, dict):
        value.update({k: v for k, v in ROUTE.get(name, {}).items()
                      if k in (schema.get("properties") or {})})
        if name == "ResolvedQuery":
            value["question"] = question
            value["search_text"] = question
    return value


# -- request reading -----------------------------------------------------------------

def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def last_user_text(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _text(message.get("content"))[-300:]
    return ""


def forced_tool(tool_choice: Any) -> str:
    if isinstance(tool_choice, dict):
        return (tool_choice.get("function") or {}).get("name", "")
    return ""


# -- app -----------------------------------------------------------------------------

app = FastAPI(title="stub provider")
_calls: dict[str, int] = {}


def _count(kind: str) -> None:
    _calls[kind] = _calls.get(kind, 0) + 1


@app.get("/stats")
async def stats() -> dict:
    return dict(_calls)


def _completion(model: str, message: dict, finish: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def _chunk(model: str, delta: dict, finish: str | None = None) -> str:
    body = {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(body)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "stub")
    messages = body.get("messages") or []
    question = last_user_text(messages)
    tools = body.get("tools") or []
    stream = bool(body.get("stream"))
    tool_ran = any(m.get("role") == "tool" for m in messages)

    # 1. structured output
    fmt = body.get("response_format") or {}
    if fmt.get("type") == "json_schema":
        _count("structured")
        await asyncio.sleep(_wait(Latency.structured_ms))
        content = json.dumps(structured_answer(fmt, question))
        if stream:
            async def one():
                yield _chunk(model, {"role": "assistant", "content": content})
                yield _chunk(model, {}, "stop")
                yield "data: [DONE]\n\n"
            return StreamingResponse(one(), media_type="text/event-stream")
        return JSONResponse(_completion(model, {"role": "assistant", "content": content}, "stop"))

    # 2. a tool call, when tools are offered and none has run yet
    names = [(t.get("function") or {}).get("name", "") for t in tools]
    wanted = forced_tool(body.get("tool_choice"))
    if tools and not tool_ran and body.get("tool_choice") != "none":
        name = wanted or (KNOWLEDGE_TOOL if KNOWLEDGE_TOOL in names else names[0])
        spec = next((t["function"] for t in tools if (t.get("function") or {}).get("name") == name), {})
        args = instance(spec.get("parameters") or {"type": "object"}, question=question)
        if isinstance(args, dict) and "query" in args:
            args["query"] = question
        call = {"id": f"call_{uuid.uuid4().hex[:10]}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}
        _count(f"tool:{name}")
        await asyncio.sleep(_wait(Latency.toolcall_ms))
        if stream:
            async def tool_stream():
                yield _chunk(model, {"role": "assistant", "content": None,
                                     "tool_calls": [{"index": 0, **call}]})
                yield _chunk(model, {}, "tool_calls")
                yield "data: [DONE]\n\n"
            return StreamingResponse(tool_stream(), media_type="text/event-stream")
        return JSONResponse(_completion(
            model, {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"))

    # 3. text — the answer
    _count("answer")
    words = (ANSWER * 8).split(" ")[: Latency.tokens]
    if stream:
        async def text_stream():
            await asyncio.sleep(_wait(Latency.ttft_ms))
            yield _chunk(model, {"role": "assistant", "content": ""})
            for word in words:
                yield _chunk(model, {"content": word + " "})
                await asyncio.sleep(_wait(Latency.token_ms))
            yield _chunk(model, {}, "stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(text_stream(), media_type="text/event-stream")
    await asyncio.sleep(_wait(Latency.ttft_ms + Latency.token_ms * len(words)))
    return JSONResponse(_completion(model, {"role": "assistant", "content": " ".join(words)}, "stop"))


def unit_vector(text: str, dim: int) -> list[float]:
    """Deterministic in the text, uniform on the sphere, unit length."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    vector = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    inputs = body.get("input")
    texts = [inputs] if isinstance(inputs, str) else list(inputs or [])
    _count("embeddings")
    await asyncio.sleep(_wait(Latency.embed_ms))
    return JSONResponse({
        "object": "list", "model": body.get("model", "stub"),
        "data": [{"object": "embedding", "index": i, "embedding": unit_vector(t, Latency.embed_dim)}
                 for i, t in enumerate(texts)],
        "usage": {"prompt_tokens": 10 * len(texts), "total_tokens": 10 * len(texts)},
    })


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    for name in ("structured_ms", "toolcall_ms", "ttft_ms", "token_ms", "embed_ms", "jitter"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(Latency, name))
    parser.add_argument("--tokens", type=int, default=Latency.tokens)
    parser.add_argument("--embed-dim", type=int, default=Latency.embed_dim)
    parser.add_argument("--keep-alive", type=float, default=5.0,
                        help="seconds an idle connection is kept open (uvicorn's default is 5); "
                             "set low to make the client's stale-socket handling earn its keep")
    args = parser.parse_args()
    for name in ("structured_ms", "toolcall_ms", "ttft_ms", "token_ms", "embed_ms", "jitter",
                 "tokens", "embed_dim"):
        setattr(Latency, name, getattr(args, name))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False,
                timeout_keep_alive=args.keep_alive)


if __name__ == "__main__":
    main()
