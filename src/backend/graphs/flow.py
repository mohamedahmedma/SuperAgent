"""Draw how one chat turn runs through this backend, end to end, as a single PNG.

    python -m backend.graphs                    # $GRAPHS_OUTPUT_DIR/project_flow.png
    python -m backend.graphs --profile base     # another profile's flow
    python -m backend.graphs --parts            # also each LangGraph graph alone
    python -m backend.graphs --check            # only verify the map, draw nothing

A turn is only partly a LangGraph graph. The HTTP door, the turn pipeline, the planner and
the answer checks are plain Python (`backend/agent/chat/turn_pipeline.py` makes every decision a
turn involves, in order), and the agent and the RAG graph are compiled graphs inside it.
So the picture is built from two kinds of part, and neither is allowed to drift:

  COMPILED   The agent (`create_agent_for_request`), the live RAG graph (`rag_graph`, in
             whichever shape the profile compiles), the signal-ladder rungs
             (`build_ladder`) and the bound tools are read from the code on every run —
             a middleware added or a rung switched off shows up without touching this file.
  DECLARED   The plain-Python steps are listed below, each naming the function that does
             it. Every name is imported before anything is drawn, and a name that no
             longer resolves stops the drawing (`--check`, and
             `tests/general/test_turn_flow_map.py`, which runs the same check in CI).

Colour key: amber = a model call, blue = another service or store, green = what the
parent is sent, red = a refusal at the door.

Rendering goes through mermaid.ink, LangGraph's own default: the Mermaid text — function
names and edges, nothing else — is sent to that service. PNG is the only output: one
picture per chart, at the size `render()` asks for, and nothing else written beside it.

This lives under `backend/` rather than `scripts/` because the image carries `backend/`
and nothing else, so `python -m backend.graphs` is runnable in the container the same way
`alembic upgrade head` is.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

MERMAID_INK = "https://mermaid.ink"

# Where the pictures land. `GRAPHS_OUTPUT_DIR` is read from the environment (and so from
# `.env`, and from the value docker-compose sets on the container) rather than being a
# path baked in here: in the image the repository root is not writable by the app user,
# and the directory that is, is a mounted volume.
DEFAULT_OUTPUT_DIR = "graphs"


def output_dir() -> Path:
    """The configured output directory, as an absolute path."""
    configured = (os.getenv("GRAPHS_OUTPUT_DIR") or "").strip() or DEFAULT_OUTPUT_DIR
    return Path(configured).expanduser().resolve()


# -- the declared half ------------------------------------------------------------------
#
# (id, section, label, note, kind, ref). `label` is what the box is called in the code;
# `ref` is "module:qualname" and must import. `kind` picks the colour.

SECTIONS = {
    "http": "HTTP door · backend/api/routes/chat.py",
    "entry": "Turn entry · TurnPipeline",
    "resume": "Resumed clarification · a paused search picks up",
    "planner": "Planner · plan_turn (backend/agent/chat/orchestrator.py)",
    "agent": "Agent · create_agent_for_request (compiled LangGraph)",
    "tools": "Tools bound by the profile",
    "rag": "RAG graph · rag_graph (compiled LangGraph, live shape)",
    "retrieval": "retrieve_documents · backend/agent/rag/utils.py",
    "settle": "Answer settlement · TurnPipeline.settle_agent_answer",
    "save": "Save · off the request, in the background",
}

STEPS = [
    # HTTP door
    ("parent", "http", "Parent message", "POST /chat/stream", "start", ""),
    (
        "auth",
        "http",
        "get_current_user",
        "RS256 token, checked against identity's JWKS",
        "step",
        "backend.infra.auth:get_current_user",
    ),
    ("r401", "http", "401", "no valid token", "refuse", ""),
    (
        "attach",
        "http",
        "_attachment_id",
        "a voice note must be the caller's own",
        "step",
        "backend.api.routes.chat:_attachment_id",
    ),
    (
        "admit",
        "http",
        "TurnAdmission.admit",
        "provider cooldown → per-user GCRA rate → per-user concurrent lease · Redis Lua, fails open",
        "step",
        "backend.agent.chat.admission:TurnAdmission.admit",
    ),
    ("r429", "http", "429 + Retry-After", "refused at the door", "refuse", ""),
    (
        "stream",
        "http",
        "chat_with_agent_stream",
        "SSE: rag_step · content · trace · done",
        "step",
        "backend.agent.chat.service:chat_with_agent_stream",
    ),
    # Turn entry
    (
        "open",
        "entry",
        "TurnPipeline.open",
        "waits ≤10s for the previous save · loads window + child pin",
        "step",
        "backend.agent.chat.turn_pipeline:TurnPipeline.open",
    ),
    (
        "enter",
        "entry",
        "enter_turn",
        "reads the message against a pending clarification (TTL)",
        "step",
        "backend.agent.chat.clarification:enter_turn",
    ),
    (
        "resolve_reply",
        "entry",
        "resolve_turn_question",
        "FAST_MODEL · answer, correction or new question?",
        "llm",
        "backend.agent.chat.orchestrator:resolve_turn_question",
    ),
    (
        "child_choice",
        "entry",
        "TurnPipeline.settle_child_choice",
        "a reply to 'which child?' pins the child",
        "step",
        "backend.agent.chat.turn_pipeline:TurnPipeline.settle_child_choice",
    ),
    (
        "record",
        "entry",
        "TurnPipeline.record_question",
        "queued, not awaited",
        "step",
        "backend.agent.chat.turn_pipeline:TurnPipeline.record_question",
    ),
    (
        "resumes",
        "entry",
        "TurnPipeline.resumes_a_search",
        "is this the answer to a paused search?",
        "decision",
        "backend.agent.chat.turn_pipeline:TurnPipeline.resumes_a_search",
    ),
    # Resumed clarification
    (
        "run_resume",
        "resume",
        "TurnPipeline.run_resumed_search",
        "carries the paused turn's language, year, names",
        "step",
        "backend.agent.chat.turn_pipeline:TurnPipeline.run_resumed_search",
    ),
    (
        "resume_rag",
        "resume",
        "resume_rag_from_hitl",
        "refines the question with the reply",
        "step",
        "backend.agent.rag.pipeline:resume_rag_from_hitl",
    ),
    (
        "resume_retrieval",
        "resume",
        "ResumeRetrieval",
        "targeted search, graded like any other",
        "step",
        "backend.agent.rag.graph_nodes:ResumeRetrieval",
    ),
    (
        "settle_resume",
        "resume",
        "TurnPipeline.settle_resumed_search",
        "",
        "decision",
        "backend.agent.chat.turn_pipeline:TurnPipeline.settle_resumed_search",
    ),
    (
        "resume_answer",
        "resume",
        "build_resume_answer_messages",
        "MODEL answers from the retrieved documents",
        "llm",
        "backend.agent.chat.context_messages:build_resume_answer_messages",
    ),
    (
        "resume_static",
        "resume",
        "resumed_static_reply",
        "profile copy: nothing to answer from",
        "reply",
        "backend.agent.chat.answer_checks:resumed_static_reply",
    ),
    # Planner
    (
        "plan",
        "planner",
        "TurnPipeline.plan → plan_turn",
        "never raises: a failed plan runs the turn unplanned",
        "step",
        "backend.agent.chat.orchestrator:plan_turn",
    ),
    (
        "roster",
        "planner",
        "_start_roster",
        "records roster prefetch, in parallel · cached per guardian",
        "io",
        "backend.agent.chat.orchestrator:_start_roster",
    ),
    (
        "resolve",
        "planner",
        "resolve_question",
        "FAST_MODEL · only for a follow-up (needs_resolution)",
        "llm",
        "backend.agent.chat.resolution:resolve_question",
    ),
    (
        "ladder",
        "planner",
        "SignalLadder.run",
        "cheapest rung first, stops once scope is settled",
        "step",
        "backend.agent.chat.signals:SignalLadder.run",
    ),
    (
        "settle_child",
        "planner",
        "_settle_child → resolve_child",
        "which child this turn is about",
        "step",
        "backend.agent.chat.orchestrator:_settle_child",
    ),
    (
        "resolve_turn",
        "planner",
        "resolve_turn",
        "the plan: what runs, with which tools",
        "decision",
        "backend.agent.chat.turn_policy:resolve_turn",
    ),
    (
        "plan_social",
        "planner",
        "_plan_social",
        "greeting / thanks",
        "step",
        "backend.agent.chat.turn_policy:_plan_social",
    ),
    (
        "plan_ood",
        "planner",
        "_plan_out_of_domain",
        "out of scope, with certainty",
        "step",
        "backend.agent.chat.turn_policy:_plan_out_of_domain",
    ),
    (
        "plan_child_choice",
        "planner",
        "_plan_child_choice",
        "several children, none named",
        "step",
        "backend.agent.chat.turn_policy:_plan_child_choice",
    ),
    (
        "plan_tools",
        "planner",
        "_plan_tools",
        "narrow tools · planned parallel calls · forced tool",
        "step",
        "backend.agent.chat.turn_policy:_plan_tools",
    ),
    (
        "short_circuit",
        "planner",
        "TurnPipeline.settle_short_circuit",
        "static reply, refusal, or 'which child?' — no agent is built",
        "reply",
        "backend.agent.chat.turn_pipeline:TurnPipeline.settle_short_circuit",
    ),
    # Agent (its graph is compiled in; these are the doors in and out)
    (
        "agent_call",
        "agent",
        "TurnPipeline.agent_call",
        "recursion_limit from the profile",
        "step",
        "backend.agent.chat.turn_pipeline:TurnPipeline.agent_call",
    ),
    # Knowledge tool internals (the tool boxes themselves are read from the profile)
    (
        "run_rag",
        "tools",
        "run_rag_graph",
        "turn memo: an identical query is answered from the first run",
        "step",
        "backend.agent.rag.pipeline:run_rag_graph",
    ),
    (
        "kb_outcome",
        "tools",
        "knowledge outcome",
        "chunks · no_knowledge · needs_clarification · needs_scope_selection · retrieval_error · empty",
        "decision",
        "backend.agent.tools.knowledge:make_search_knowledge_base",
    ),
    (
        "records_service",
        "tools",
        "records service :8100",
        "guardian ↔ student check · access audit",
        "io",
        "",
    ),
    # Retrieval internals
    (
        "translate",
        "retrieval",
        "translate_for_search",
        "FAST_MODEL · only when not in the corpus's language",
        "llm",
        "backend.agent.rag.query_translation:translate_for_search",
    ),
    (
        "cache_lookup",
        "retrieval",
        "RetrievalCache.lookup",
        "Redis · keyed on corpus_version",
        "io",
        "backend.agent.rag.retrieval_cache:RetrievalCache.lookup",
    ),
    (
        "embed",
        "retrieval",
        "embed_query",
        "bge-m3 · in-process memo → Redis",
        "io",
        "backend.indexing.embedding:embed_query",
    ),
    (
        "hybrid",
        "retrieval",
        "MilvusStore.hybrid_retrieve",
        "dense + BM25 · RRF k=60 · leaf chunks (L3)",
        "io",
        "backend.indexing.milvus_client:MilvusStore.hybrid_retrieve",
    ),
    (
        "merge",
        "retrieval",
        "_auto_merge_candidates",
        "leaves → L2/L1 parents from Postgres",
        "io",
        "backend.agent.rag.utils:_auto_merge_candidates",
    ),
    (
        "rerank",
        "retrieval",
        "_rerank_documents",
        "Jina rerank, optional · min-score filter",
        "step",
        "backend.agent.rag.utils:_rerank_documents",
    ),
    (
        "cache_store",
        "retrieval",
        "RetrievalCache.store",
        "",
        "io",
        "backend.agent.rag.retrieval_cache:RetrievalCache.store",
    ),
    # Answer settlement
    (
        "settle",
        "settle",
        "TurnPipeline.settle_agent_answer",
        "evidence in hand for the first time",
        "decision",
        "backend.agent.chat.turn_pipeline:TurnPipeline.settle_agent_answer",
    ),
    (
        "ask",
        "settle",
        "build_pending_hitl",
        "retrieval asked a question · the next message answers it",
        "reply",
        "backend.agent.chat.clarification:build_pending_hitl",
    ),
    (
        "check_records",
        "settle",
        "enforce_records_agreement",
        "figures must match the records",
        "step",
        "backend.agent.chat.answer_checks:enforce_records_agreement",
    ),
    (
        "check_forced",
        "settle",
        "enforce_forced_tool_ran",
        "a required tool must have run",
        "step",
        "backend.agent.chat.answer_checks:enforce_forced_tool_ran",
    ),
    (
        "check_figures",
        "settle",
        "enforce_answer_figures",
        "numbers must appear in the retrieved chunks",
        "step",
        "backend.agent.chat.answer_checks:enforce_answer_figures",
    ),
    (
        "check_empty",
        "settle",
        "nothing_usable_reply",
        "the model said nothing usable",
        "step",
        "backend.agent.chat.answer_checks:nothing_usable_reply",
    ),
    ("replace", "settle", "content_replace", "answer withdrawn and replaced", "reply", ""),
    (
        "blocks",
        "settle",
        "settle_answer_blocks",
        "record tables as typed blocks · figure markers resolved",
        "reply",
        "backend.agent.chat.answer_blocks:settle_answer_blocks",
    ),
    (
        "assets",
        "settle",
        "TurnPipeline.attach_assets",
        "figures, rendered for this client",
        "step",
        "backend.agent.chat.turn_pipeline:TurnPipeline.attach_assets",
    ),
    # Save
    (
        "save_meta",
        "save",
        "TurnPipeline.save_metadata",
        "a patch: child pin, pending question, title",
        "step",
        "backend.agent.chat.turn_pipeline:TurnPipeline.save_metadata",
    ),
    (
        "commit",
        "save",
        "TurnPipeline.commit",
        "BackgroundJobs lane per conversation · retry · idempotency keys",
        "step",
        "backend.agent.chat.turn_pipeline:TurnPipeline.commit",
    ),
    (
        "interrupted",
        "save",
        "TurnPipeline.commit_interrupted",
        "stop pressed / connection dropped",
        "step",
        "backend.agent.chat.turn_pipeline:TurnPipeline.commit_interrupted",
    ),
    ("postgres", "save", "Postgres · conversations", "append-only messages", "io", ""),
    (
        "hold",
        "save",
        "_hold_until_stored",
        "'stored' event with the row ids",
        "step",
        "backend.agent.chat.service:_hold_until_stored",
    ),
    ("done", "save", "Reply delivered", "SSE done", "finish", ""),
]

# (from, to, label, dotted). Dotted = a call made on the side, or a path most turns skip.
EDGES = [
    ("parent", "auth", "", False),
    ("auth", "r401", "invalid", True),
    ("auth", "attach", "", False),
    ("attach", "admit", "", False),
    ("admit", "r429", "refused", True),
    ("admit", "stream", "lease", False),
    ("stream", "open", "", False),
    ("open", "enter", "", False),
    ("enter", "resolve_reply", "clarification pending", True),
    ("resolve_reply", "child_choice", "", True),
    ("enter", "child_choice", "", False),
    ("child_choice", "record", "", False),
    ("record", "postgres", "queued", True),
    ("record", "resumes", "", False),
    ("resumes", "run_resume", "yes", False),
    ("resumes", "plan", "no", False),
    ("run_resume", "resume_rag", "", False),
    ("resume_rag", "resume_retrieval", "", False),
    ("resume_retrieval", "cache_lookup", "search", True),
    ("resume_retrieval", "settle_resume", "", False),
    ("settle_resume", "resume_answer", "documents", False),
    ("settle_resume", "resume_static", "nothing to answer from", False),
    ("settle_resume", "ask", "asks again", False),
    ("resume_answer", "assets", "", False),
    ("resume_static", "assets", "", False),
    ("plan", "roster", "starts first", True),
    ("plan", "resolve", "", False),
    # resolve → ladder rungs → settle_child is wired from the compiled ladder.
    ("roster", "settle_child", "roster", True),
    ("settle_child", "resolve_turn", "", False),
    ("resolve_turn", "plan_social", "social", False),
    ("resolve_turn", "plan_ood", "out of domain", False),
    ("resolve_turn", "plan_child_choice", "which child?", False),
    ("resolve_turn", "plan_tools", "otherwise", False),
    ("plan_social", "short_circuit", "", False),
    ("plan_ood", "short_circuit", "", False),
    ("plan_child_choice", "short_circuit", "", False),
    ("plan_tools", "agent_call", "", False),
    ("short_circuit", "save_meta", "", False),
    # agent_call → agent graph → settle is wired from the compiled agent.
    ("run_rag", "kb_outcome", "memo hit", True),
    ("translate", "cache_lookup", "", False),
    ("cache_lookup", "embed", "miss", False),
    ("embed", "hybrid", "", False),
    ("hybrid", "merge", "", False),
    ("merge", "rerank", "", False),
    ("rerank", "cache_store", "", False),
    ("settle", "ask", "retrieval asked", False),
    ("settle", "check_records", "answer", False),
    ("check_records", "check_forced", "ok", False),
    ("check_forced", "check_figures", "ok", False),
    ("check_figures", "check_empty", "ok", False),
    ("check_empty", "blocks", "ok", False),
    ("check_records", "replace", "fails", True),
    ("check_forced", "replace", "fails", True),
    ("check_figures", "replace", "fails", True),
    ("check_empty", "replace", "empty", True),
    ("ask", "assets", "", False),
    ("replace", "assets", "", False),
    ("blocks", "assets", "", False),
    ("assets", "save_meta", "", False),
    ("save_meta", "commit", "", False),
    ("commit", "postgres", "background", True),
    ("interrupted", "commit", "", True),
    ("commit", "hold", "", False),
    ("hold", "done", "", False),
]

#: What each compiled node does, by the name the graph gives it. Optional — a node with no
#: note still draws — but a note whose node has gone is stale, and the check says so.
AGENT_NOTES = {
    "_run_the_planned_calls_together.before_model": "the planner's calls, dispatched without asking the model",
    "_stop_after_a_terminal_tool_result.before_model": "ends the turn on a terminal retrieval result",
    "model": "MODEL · wrapped by the tool budget and the forced-tool rule",
    "_ToolBudget.after_model": "withholds a tool once its budget is spent",
    "_drop_repeated_calls.after_model": "identical tool calls collapsed",
    "tools": "runs the tool calls, in parallel",
}
AGENT_MODEL_NODES = {"model"}

RAG_NOTES = {
    "classify_complexity": "FAST_MODEL, after a local rule",
    "prepare_sub_questions": "2-4 sub-questions",
    "rag_sub_agent": "retrieve + grade, one per sub-question (Send)",
    "synthesis": "merges and ranks the sub-agents' evidence",
    "retrieve_initial": "child's name removed from the query",
    "grade_documents": "GRADE_MODEL · answer / rewrite / clarify / scope_select / no_knowledge",
    "rewrite_question": "FAST_MODEL · one Step-back or HyDE rewrite",
    "retrieve_rewritten": "second search with the rewrite",
}
RAG_MODEL_NODES = {"classify_complexity", "grade_documents", "rewrite_question"}
#: Where each RAG node reaches into retrieval, and whether it translates first.
RAG_RETRIEVAL_ENTRY = {
    "retrieve_initial": "translate",
    "rag_sub_agent": "translate",
    "retrieve_rewritten": "cache_lookup",
}

LADDER_NOTES = {
    "backend.agent.chat.signals:SocialDetector": ("social phrases · no model", "step"),
    "backend.agent.rag.scope_detector:CatalogueScopeDetector": (
        "scope catalogue · vector match",
        "step",
    ),
    "backend.agent.rag.scope_detector:ScopeModelDetector": (
        "FAST_MODEL · second opinion on scope",
        "llm",
    ),
    "backend.agent.chat.signals:CorpusSimilarityDetector": (
        "domain gate · corpus similarity",
        "step",
    ),
    "backend.agent.chat.signals:EnvelopeDetector": ("FAST_MODEL · request envelope", "llm"),
}


# -- the target flow --------------------------------------------------------------------
#
# A PROPOSAL, not the code. Everything above is held to what runs; this is the shape the
# turn is meant to reach, drawn so the two can be put side by side. It names no functions
# on purpose — inventing plausible ones would make a design read as an implementation, and
# `--check` deliberately does not validate it.
#
# READ THE REJECTED LIST BEFORE CHANGING THIS ONE. A first draft of this target proposed
# four "obvious" wins that this repo had already considered and settled — in code, in
# comments and in commits. They are recorded below so the next reader does not re-derive
# them; each would have undone a fix that had reached a parent.
#
# What the target changes, each answering a finding that survived checking:
#   T1  the two writes still on the event loop — `record_question` and `commit` are called
#       straight from the streaming generator while every other blocking call around them
#       was deliberately moved off it. They reach a synchronous Redis pipeline, so a
#       stalled Redis freezes every concurrent stream in the worker, twice a turn.
#   T2  the quota check narrowed to the models THIS turn needs — today one model's
#       cooldown refuses every turn in the replica, including turns that never call it.
#   T3  the lease held until the work actually ends — cancelling the agent task does not
#       stop the sync tool already running on an executor thread.
#   T4  the HITL verdict memoised on the trace's version, NOT hoisted out of the loop.
#       It cannot be hoisted: the knowledge tool decides it after the stream has started.
#   T5  a degraded retrieval says so — a fault inside result assembly silently re-runs
#       dense-only, and unlike the inner failure that path logs nothing.
#   T6  an event journal behind the stream, so `Last-Event-ID` resumes a dropped
#       connection. The largest gap, and the one with no counter-argument.
#   T7  the prompt prefix as a deliberate TRADE-OFF, not a free win. Ordering for caching
#       is already done; what breaks the prefix is the planner narrowing tools per turn,
#       which is itself a token saving. One or the other — measure, then choose.
#   T8  the grader off the critical path when retrieval is already decisive. PARKED:
#       RAG_FIX_PLAN items 9/14 say not to start this. Drawn dashed, as a lever, not as
#       agreed work.
#
# REJECTED — considered, and wrong. Do not re-propose without reading the reason:
#   R1  merge the resolver and the scope classifier into one call. The classifier is FED
#       the resolver's output (`SignalContext.text_to_score`); merging makes it score the
#       raw message, which is the "blunt instrument" the resolver exists to replace. Both
#       are independently gated, so one merged call costs MORE calls, not fewer.
#   R2  run the attachment check and admission concurrently. Admission is asked only once
#       the request is known valid; run together, a bad attachment still takes a lease,
#       and a gather that raises drops the other result, so the lease leaks until it
#       expires — a five-minute self-lockout.
#   R3  merge the two admission Redis scripts. The order is load-bearing: the rate script
#       WRITES on admit, so a turn refused for concurrency would have spent a rate token,
#       and the refusal reason that picks the parent's copy could no longer tell the two
#       apart.
#   R4  skip the second grader call after a rewrite. The rewrite ADDS to the first pass;
#       the second grade is what routes the turn, splits answerable from partial, and
#       feeds the trimming that keeps the answer prompt bounded. A change here was tried
#       and reverted once already.

TARGET_SECTIONS = {
    "t_door": "Door · unchanged order, work that outlives the client accounted for  [T2, T3]",
    "t_graph": "Turn · one place that decides, resumable  [T6]",
    "t_classify": "Understand · two gated stages, unchanged  [R1]",
    "t_retrieve": "Retrieve · unchanged, degradation made visible  [T5, R4]",
    "t_answer": "Answer · prefix caching as a measured trade-off  [T7, T8]",
    "t_deliver": "Deliver · journal first, stream from it  [T1, T4, T6]",
}

TARGET_STEPS = [
    ("t_msg", "t_door", "Parent message", "POST /chat/stream", "start"),
    (
        "t_valid",
        "t_door",
        "validate the request",
        "the attachment must be the caller's own · still before admission  [R2]",
        "step",
    ),
    (
        "t_admit",
        "t_door",
        "admit",
        "rate, then lease · still two scripts, on purpose  [R3] · quota checked only for the "
        "models this turn will call  [T2]",
        "step",
    ),
    (
        "t_429",
        "t_door",
        "401 / 429 + Retry-After",
        "one model's cooldown no longer refuses every turn  [T2]",
        "refuse",
    ),
    (
        "t_lease",
        "t_door",
        "lease held until the work ends",
        "a cancelled stream does not free a place while its tool thread still runs  [T3]",
        "io",
    ),
    (
        "t_resume",
        "t_graph",
        "resume or start",
        "a reconnect replays from the journal · nothing is replanned",
        "decision",
    ),
    (
        "t_open",
        "t_graph",
        "load the thread",
        "window + child pin · the pending question merged as a PATCH, never written whole",
        "step",
    ),
    (
        "t_resolve",
        "t_classify",
        "resolve the question",
        "FAST_MODEL · gated: a follow-up or a translation, not every turn",
        "llm",
    ),
    (
        "t_ladder",
        "t_classify",
        "scope ladder",
        "cheapest rung first · fed the RESOLVED question · stops once settled  [R1]",
        "step",
    ),
    ("t_roster", "t_classify", "roster prefetch", "started first, joined here · unchanged", "io"),
    (
        "t_route",
        "t_classify",
        "route",
        "social · out of domain · which child · records · knowledge",
        "decision",
    ),
    ("t_static", "t_classify", "profile copy", "no model call at all", "reply"),
    (
        "t_search",
        "t_retrieve",
        "retrieve",
        "cache → embed → hybrid + BM25 → merge → rerank · unchanged",
        "io",
    ),
    (
        "t_degraded",
        "t_retrieve",
        "say when it degraded",
        "a dense-only fallback is logged and marked on the trace, never silent  [T5]",
        "step",
    ),
    (
        "t_grade",
        "t_retrieve",
        "grade",
        "GRADE_MODEL · memoised per query · skipped only when decisive  [T8]",
        "llm",
    ),
    (
        "t_rewrite",
        "t_retrieve",
        "rewrite once",
        "adds to the first pass, and IS graded again  [R4]",
        "llm",
    ),
    (
        "t_ask",
        "t_retrieve",
        "ask the parent",
        "clarification · kept if the stream is cut off",
        "reply",
    ),
    (
        "t_answer",
        "t_answer",
        "answer",
        "MODEL · narrow tools OR a cached prefix — measure which is cheaper  [T7]",
        "llm",
    ),
    (
        "t_checks",
        "t_answer",
        "grounding checks",
        "records agreement · forced tool · figures · unchanged",
        "step",
    ),
    (
        "t_replace",
        "t_answer",
        "withdraw and replace",
        "a failed check never reaches the parent as an answer",
        "reply",
    ),
    (
        "t_journal",
        "t_deliver",
        "append to the event journal",
        "numbered and durable BEFORE it is sent · the append is off the event loop  [T1, T6]",
        "io",
    ),
    (
        "t_stream",
        "t_deliver",
        "stream from the journal",
        "Last-Event-ID resumes a dropped connection  [T6] · the HITL verdict is memoised on "
        "the trace version, not recomputed per token  [T4]",
        "step",
    ),
    ("t_save", "t_deliver", "save", "background lane, idempotent · off the loop  [T1]", "step"),
    ("t_done", "t_deliver", "Reply delivered", "resumable", "finish"),
]

TARGET_EDGES = [
    ("t_msg", "t_valid", "", False),
    ("t_valid", "t_admit", "valid", False),
    ("t_admit", "t_429", "refused", True),
    ("t_admit", "t_lease", "admitted", False),
    ("t_lease", "t_resume", "", False),
    ("t_resume", "t_stream", "a reconnect replays what it missed", True),
    ("t_resume", "t_open", "a new turn", False),
    ("t_open", "t_resolve", "", False),
    ("t_open", "t_roster", "starts first", True),
    ("t_resolve", "t_ladder", "", False),
    ("t_roster", "t_route", "", True),
    ("t_ladder", "t_route", "", False),
    ("t_route", "t_static", "social · out of domain · which child", False),
    ("t_route", "t_search", "knowledge", False),
    ("t_route", "t_answer", "records only · no search", False),
    ("t_static", "t_journal", "", False),
    ("t_search", "t_degraded", "", False),
    ("t_degraded", "t_grade", "", False),
    ("t_degraded", "t_answer", "decisive evidence, grader skipped", True),
    ("t_grade", "t_answer", "sufficient", False),
    ("t_grade", "t_rewrite", "insufficient", False),
    ("t_grade", "t_ask", "needs the parent", False),
    ("t_rewrite", "t_search", "once", False),
    ("t_ask", "t_journal", "", False),
    ("t_answer", "t_checks", "", False),
    ("t_checks", "t_replace", "a check failed", True),
    ("t_checks", "t_journal", "ok", False),
    ("t_replace", "t_journal", "", False),
    ("t_journal", "t_stream", "", False),
    ("t_stream", "t_save", "", False),
    ("t_save", "t_done", "", False),
]


def build_target() -> FlowChart:
    """The proposed flow. Declared outright: it describes code that does not exist yet."""
    chart = FlowChart(sections=TARGET_SECTIONS)
    for node_id, section, label, note, kind in TARGET_STEPS:
        chart.node(node_id, section, label, note, kind)
    for edge in TARGET_EDGES:
        chart.edge(*edge)
    return chart


# -- checking the map against the code --------------------------------------------------


def resolve_ref(ref: str):
    module, _, qualname = ref.partition(":")
    target = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def stale_references() -> list[str]:
    """Every declared name that no longer imports. Empty means the map matches the code."""
    refs = [step[5] for step in STEPS if step[5]] + list(LADDER_NOTES)
    stale = []
    for ref in refs:
        try:
            resolve_ref(ref)
        except (ImportError, AttributeError) as exc:
            stale.append(f"{ref}  ({type(exc).__name__}: {exc})")
    return stale


# -- building the chart -----------------------------------------------------------------


def _safe_id(prefix: str, name: str) -> str:
    return prefix + re.sub(r"[^A-Za-z0-9_]", "_", name)


def _text(value: str) -> str:
    return value.replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")


@dataclass
class FlowChart:
    nodes: dict = field(default_factory=dict)  # id -> (label, note, kind)
    members: dict = field(default_factory=dict)  # section -> [id]
    edges: list = field(default_factory=list)  # (from, to, label, dotted)
    #: The lanes this chart draws, in order. Defaulted to the live turn's, so the target
    #: chart is the only caller that has to say which it means.
    sections: dict = field(default_factory=lambda: SECTIONS)

    def node(
        self, node_id: str, section: str, label: str, note: str = "", kind: str = "step"
    ) -> None:
        self.nodes[node_id] = (label, note, kind)
        self.members.setdefault(section, []).append(node_id)

    def edge(self, source: str, target: str, label: str = "", dotted: bool = False) -> None:
        self.edges.append((source, target, label, dotted))

    def render(self) -> str:
        lines = ["flowchart TD"]
        for section, title in self.sections.items():
            ids = self.members.get(section)
            if not ids:
                continue
            # Prefixed: a subgraph sharing a node's id (`settle`) is read as its own parent.
            lines.append(f'  subgraph sg_{section}["{_text(title)}"]')
            for node_id in ids:
                lines.append("    " + self._render_node(node_id))
            lines.append("  end")
        for source, target, label, dotted in self.edges:
            arrow = "-.->" if dotted else "-->"
            text = f'|"{_text(label)}"|' if label else ""
            lines.append(f"  {source} {arrow}{text} {target}")
        lines += [
            "  classDef step fill:#f2f0ff,stroke:#9370db,color:#222",
            "  classDef decision fill:#ede7ff,stroke:#6a4fc9,color:#222",
            "  classDef llm fill:#fff3d1,stroke:#d49a00,color:#222",
            "  classDef io fill:#e3f1ff,stroke:#3b82c4,color:#222",
            "  classDef reply fill:#e2f6e5,stroke:#2e9447,color:#222",
            "  classDef refuse fill:#fde4e4,stroke:#c94040,color:#222",
            "  classDef start fill:#ffffff,stroke:#6a4fc9,color:#222",
            "  classDef finish fill:#bfb0ff,stroke:#6a4fc9,color:#222",
        ]
        by_kind: dict = {}
        for node_id, (_, _, kind) in self.nodes.items():
            by_kind.setdefault(kind, []).append(node_id)
        for kind, ids in by_kind.items():
            lines.append(f"  class {','.join(ids)} {kind}")
        for section in self.sections:
            if self.members.get(section):
                lines.append(f"  style sg_{section} fill:#fafaff,stroke:#c9c3e6,color:#444")
        return "\n".join(lines) + "\n"

    def _render_node(self, node_id: str) -> str:
        label, note, kind = self.nodes[node_id]
        text = f"<b>{_text(label)}</b>" + (f"<br/>{_text(note)}" if note else "")
        if kind in ("start", "finish"):
            return f'{node_id}(["{text}"])'
        if kind == "decision":
            return f'{node_id}{{{{"{text}"}}}}'
        if kind == "io":
            return f'{node_id}[("{text}")]'
        return f'{node_id}["{text}"]'


def _embed_graph(
    chart: FlowChart,
    graph,
    *,
    prefix: str,
    section: str,
    notes: dict,
    model_nodes: set,
    enter_from: str,
    exit_to: str,
) -> set:
    """Copy a compiled graph into the chart, its start and end joined to the steps around it."""
    drawable = graph.get_graph(xray=True)
    named = set()
    for node_id, node in drawable.nodes.items():
        if node_id in ("__start__", "__end__"):
            continue
        named.add(node.name)
        kind = "llm" if node.name in model_nodes else "step"
        chart.node(_safe_id(prefix, node_id), section, node.name, notes.get(node.name, ""), kind)

    def endpoint(node_id: str) -> str:
        if node_id == "__start__":
            return enter_from
        if node_id == "__end__":
            return exit_to
        return _safe_id(prefix, node_id)

    for edge in drawable.edges:
        label = edge.data if edge.data and edge.data != edge.target else ""
        chart.edge(
            endpoint(edge.source), endpoint(edge.target), str(label or ""), bool(edge.conditional)
        )
    return named


@dataclass
class Compiled:
    """What the code says the flow is, for the profile this process loaded."""

    profile_name: str
    agent: object
    rag: object
    ladder: tuple
    tools: list
    knowledge_tool: str
    records_tools: frozenset


def compile_parts() -> Compiled:
    from backend.agent.chat.orchestrator import _LadderConfig
    from backend.agent.chat.request_context import ChatRequestContext
    from backend.agent.chat.runtime import create_agent_for_request
    from backend.agent.chat.signals import build_ladder
    from backend.agent.profiles.registry import get_profile
    from backend.agent.rag.pipeline import rag_graph
    from backend.agent.tools import KNOWLEDGE_TOOL, RECORDS_TOOLS, build_tools

    profile = get_profile()
    # A context no turn will ever use: nothing here invokes a graph, only compiles it.
    ctx = ChatRequestContext.for_sync(user_id="draw-graphs", session_id="draw-graphs")
    return Compiled(
        profile_name=profile.name,
        agent=create_agent_for_request(ctx),
        rag=rag_graph,
        # The same config the planner builds its ladder from, so the rungs drawn are the
        # rungs a turn on this profile climbs.
        ladder=build_ladder(_LadderConfig(profile.agent, profile.rag)).detectors,
        tools=[tool.name for tool in build_tools(profile.agent.tools, ctx)],
        knowledge_tool=KNOWLEDGE_TOOL,
        records_tools=frozenset(RECORDS_TOOLS),
    )


def build_flow(parts: Compiled) -> tuple[FlowChart, dict]:
    """The whole turn as one chart, plus what was compiled into it (for the checks)."""
    chart = FlowChart()
    for node_id, section, label, note, kind, _ in STEPS:
        chart.node(node_id, section, label, note, kind)
    for edge in EDGES:
        chart.edge(*edge)

    # The ladder, rung by rung, between the resolver and the child.
    chart.edge("resolve", "ladder")
    previous = "ladder"
    for index, detector in enumerate(parts.ladder):
        ref = f"{type(detector).__module__}:{type(detector).__qualname__}"
        note, kind = LADDER_NOTES.get(ref, ("", "step"))
        rung = f"rung{index}"
        chart.node(rung, "planner", type(detector).__qualname__, note, kind)
        chart.edge(previous, rung, "" if index == 0 else "not settled")
        previous = rung
    chart.edge(previous, "settle_child", "scope settled" if parts.ladder else "")

    # The agent, joined to the planner and to settlement.
    agent_nodes = _embed_graph(
        chart,
        parts.agent,
        prefix="ag_",
        section="agent",
        notes=AGENT_NOTES,
        model_nodes=AGENT_MODEL_NODES,
        enter_from="agent_call",
        exit_to="settle",
    )
    chart.edge("ag_model", "interrupted", "stop / disconnect", True)

    # Each bound tool, hung off the agent's tool node.
    for name in parts.tools:
        tool_id = _safe_id("tool_", name)
        if name == parts.knowledge_tool:
            chart.node(tool_id, "tools", name, "one call per turn", "step")
            chart.edge("ag_tools", tool_id, "", True)
            chart.edge(tool_id, "run_rag", "", False)
        else:
            chart.node(tool_id, "tools", name, "", "step")
            chart.edge("ag_tools", tool_id, "", True)
            if name in parts.records_tools:
                chart.edge(tool_id, "records_service", "", True)
    chart.edge("kb_outcome", "ag_tools", "tool result", True)

    # The RAG graph, entered from the knowledge tool and ending in its outcome.
    rag_nodes = _embed_graph(
        chart,
        parts.rag,
        prefix="rag_",
        section="rag",
        notes=RAG_NOTES,
        model_nodes=RAG_MODEL_NODES,
        enter_from="run_rag",
        exit_to="kb_outcome",
    )
    for node_name, entry in RAG_RETRIEVAL_ENTRY.items():
        if node_name in rag_nodes:
            chart.edge(_safe_id("rag_", node_name), entry, "search", True)

    # A tool box only exists when a tool is bound; drop what this profile never reaches.
    if not any(name in parts.records_tools for name in parts.tools):
        _drop(chart, "records_service")
    if parts.knowledge_tool not in parts.tools:
        for node_id in ("run_rag", "kb_outcome"):
            _drop(chart, node_id)
    return chart, {"agent": agent_nodes, "rag": rag_nodes}


def _drop(chart: FlowChart, node_id: str) -> None:
    chart.nodes.pop(node_id, None)
    for ids in chart.members.values():
        if node_id in ids:
            ids.remove(node_id)
    chart.edges = [e for e in chart.edges if node_id not in (e[0], e[1])]


# -- rendering --------------------------------------------------------------------------


def _pako(mermaid: str) -> str:
    """mermaid.ink's compressed form: short enough for a large chart to fit in a URL."""
    payload = json.dumps({"code": mermaid, "mermaid": {"theme": "default"}}).encode("utf-8")
    return "pako:" + base64.urlsafe_b64encode(zlib.compress(payload, 9)).decode("ascii")


def render(mermaid: str, path_stem: Path) -> str:
    """Write the chart as a PNG. Returns the file name written.

    PNG is the only format drawn. An SVG and the Mermaid source used to be written beside
    it; one picture per chart is what anybody opened, and the other two went stale in the
    output directory without anyone noticing they had.
    """
    import httpx

    url = f"{MERMAID_INK}/img/{_pako(mermaid)}?type=png&bgColor=FFFFFF&width=2400&scale=2"
    target = path_stem.with_suffix(".png")
    target.write_bytes(_fetch(httpx, url).content)
    return target.name


def _fetch(httpx, url: str, attempts: int = 4):
    """One rendering, retried through the service being briefly unavailable.

    A free public renderer answers 502/503/429 often enough that a single attempt makes
    the command look broken when nothing is. A 400 is the opposite — it is the Mermaid
    parser rejecting THIS chart, and it will reject it again, so it is raised at once
    with the parser's own message rather than waited on four times.
    """
    for attempt in range(attempts):
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        if response.status_code == 200:
            return response
        if response.status_code < 500 and response.status_code != 429:
            raise RuntimeError(f"mermaid.ink {response.status_code}: {response.text[:300]}")
        if attempt == attempts - 1:
            raise RuntimeError(f"mermaid.ink {response.status_code} after {attempts} attempts")
        time.sleep(2**attempt)


def write_chart(name: str, mermaid: str, out: Path) -> bool:
    """Draw one chart. Returns whether it was written.

    A rendering failure is reported and swallowed: this runs on the way into the server,
    and a public renderer being briefly down is not a reason to refuse to start.
    """
    try:
        print(f"  {render(mermaid, out / name)}")
        return True
    except Exception as exc:  # the service is down, or this machine is offline
        print(f"  {name}.png FAILED: {exc}")
        return False
