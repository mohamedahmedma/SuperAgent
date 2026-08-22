from typing import List, Literal, Optional  # noqa: F401

from pydantic import BaseModel, ConfigDict, Field

from backend.assets.delivery import AssetReference, ClientCapabilities


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(StrictSchema):
    message: str
    session_id: Optional[str] = "default_session"
    # What this client can render. Omitted means "an ordinary browser"; a bot or a
    # downstream service declares its own limits instead of the server guessing.
    client_capabilities: Optional[ClientCapabilities] = None


class RetrievedChunk(StrictSchema):
    filename: str
    page_number: Optional[str | int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    chunk_id: Optional[str] = None
    # "text" | "table" | "figure" — lets a client style or filter a chunk without
    # parsing its content.
    modality: Optional[str] = None
    # Images this chunk was derived from; resolve them via /media/resolve.
    asset_ids: Optional[List[str]] = None


class RagTraceFields(StrictSchema):
    tool_used: Optional[bool] = None
    tool_name: Optional[str] = None
    query: Optional[str] = None
    rewrite_method: Optional[Literal["step_back", "hyde"]] = None
    rewritten_query: Optional[str] = None
    step_back_question: Optional[str] = None
    hyde_document: Optional[str] = None
    retrieval_stage: Optional[str] = None
    route: Optional[str] = None
    retrieval_status: Optional[str] = None
    retrieval_error: Optional[str] = None
    evidence_relevance: Optional[str] = None
    evidence_answerability: Optional[str] = None
    evidence_ambiguity: Optional[str] = None
    evidence_confidence: Optional[float] = None
    evidence_reason: Optional[str] = None
    evidence_constraints_discriminate: Optional[str] = None
    # Why `route` was chosen. Written by grade_documents_node since routing began, but
    # never declared here — so `normalize_rag_trace` dropped it and the single most
    # useful field for diagnosing an unexpected denial never reached a client.
    route_reason: Optional[str] = None
    grading_skipped: Optional[bool] = None
    grading_confident: Optional[bool] = None
    grading_term_coverage: Optional[float] = None
    grading_chunk_count: Optional[int] = None
    grading_reason: Optional[str] = None
    context_chunks_kept: Optional[int] = None
    context_chunks_available: Optional[int] = None
    context_trimmed: Optional[bool] = None
    context_coverage: Optional[float] = None
    context_selection_reason: Optional[str] = None
    # Set by the turn planner, before the agent runs. Present on every turn it
    # touched — including the ones it ended without a model, where these are the only
    # trace fields there are.
    turn_short_circuit: Optional[bool] = None
    turn_exposed_tools: Optional[List[str]] = None
    turn_retrieval_sections: Optional[List[str]] = None
    # Declared here or dropped: `normalize_rag_trace` keeps only the keys this model
    # names, so a field the planner emits but this does not never reaches a client.
    # These three were emitted and discarded, which is why a turn that ended in
    # "which one are you asking about?" carried no record of where the options came
    # from — the one question a trace of that turn has to be able to answer.
    turn_scope_options: Optional[List[str]] = None
    turn_language: Optional[str] = None
    turn_capture_user_info: Optional[bool] = None
    # Whether the turn settled on one child, and whether it had to ask. The NAME is
    # deliberately absent for the same reason it is absent from the request half:
    # this trace is persisted per message and streamed to the browser.
    turn_child_resolved: Optional[bool] = None
    turn_child_asked: Optional[bool] = None
    turn_reason: Optional[str] = None
    # What the turn was taken to be about, once references were resolved against the
    # conversation, and the conditions inherited with it.
    turn_resolved_question: Optional[str] = None
    turn_carried_constraints: Optional[List[str]] = None
    turn_is_followup: Optional[bool] = None
    request_scope: Optional[str] = None
    request_scope_certainty: Optional[str] = None
    request_language: Optional[str] = None
    request_is_social: Optional[bool] = None
    request_personal_data: Optional[List[str]] = None
    # Whether the classifier read this message as being about one of the caller's own
    # children, and how it was referred to. Anything not declared in this model is
    # silently dropped, so a signal that is not here cannot be debugged from a trace.
    #
    # The child's NAME is deliberately absent. This trace is persisted per message and
    # streamed to the browser, and a turn may resolve a child silently without ever
    # showing that name — putting it here would disclose it anyway. The boolean and the
    # reference kind answer every question a name would.
    request_about_child: Optional[bool] = None
    request_child_reference: Optional[str] = None
    request_candidate_sections: Optional[List[str]] = None
    request_scope_options: Optional[List[str]] = None
    request_top_match: Optional[dict] = None
    request_resolved_question: Optional[str] = None
    request_carried_constraints: Optional[List[str]] = None
    request_followup_intent: Optional[str] = None
    request_assessed_by: Optional[List[str]] = None
    request_reason: Optional[str] = None
    missing_slots: Optional[List[str]] = None
    hitl_prompt: Optional[str] = None
    hitl_options: Optional[List[str]] = None
    hitl_resumed: Optional[bool] = None
    hitl_answer: Optional[str] = None
    hitl_resume_strategy: Optional[str] = None
    hitl_resume_from_status: Optional[str] = None
    hitl_resume_from_route: Optional[str] = None
    hitl_targeted_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    rerank_enabled: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    rerank_error: Optional[str] = None
    rerank_timeout_seconds: Optional[float] = None
    rerank_min_score: Optional[float] = None
    post_rerank_count: Optional[int] = None
    post_threshold_count: Optional[int] = None
    retrieval_empty: Optional[bool] = None
    retrieval_mode: Optional[str] = None
    retrieval_pipeline: Optional[str] = None
    candidate_k: Optional[int] = None
    candidate_k_source: Optional[str] = None
    candidate_k_config_error: Optional[str] = None
    retrieval_candidate_multiplier: Optional[int] = None
    retrieval_top_k: Optional[int] = None
    recall_count: Optional[int] = None
    post_merge_candidate_count: Optional[int] = None
    candidate_count: Optional[int] = None
    leaf_retrieve_level: Optional[int] = None
    auto_merge_enabled: Optional[bool] = None
    auto_merge_applied: Optional[bool] = None
    auto_merge_threshold: Optional[int] = None
    auto_merge_figure_threshold: Optional[int] = None
    auto_merge_replaced_chunks: Optional[int] = None
    auto_merge_steps: Optional[int] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    rewrite_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    # Fields added for complexity-based routing
    complexity: Optional[str] = None
    complexity_reason: Optional[str] = None
    sub_questions: Optional[List[str]] = None
    sub_agent_count: Optional[int] = None
    synthesis_merged_count: Optional[int] = None
    # Renditions for every asset the retrieved chunks referenced, already adapted to
    # the calling client's declared capabilities. What goes on the wire.
    assets: Optional[List[AssetReference]] = None
    # The same assets as ids alone — what a STORED trace keeps instead. A rendition is a
    # copy of what the asset store already holds, and in inline mode that copy is the
    # image itself, base64'd into a conversation row once per message that showed it.
    # Ids cost a few dozen bytes, cannot go stale, and resolve by primary key on load.
    asset_ids: Optional[List[str]] = None


class RagSubTrace(RagTraceFields):
    pass


class RagTrace(RagTraceFields):
    sub_traces: Optional[List[RagSubTrace]] = None


class HitlResumeState(StrictSchema):
    question: str = Field(min_length=1)
    route: Literal["clarify", "scope_select"]
    retrieval_status: Literal["needs_clarification", "needs_scope_selection"]
    rewrite_count: int = Field(default=0, ge=0)
    # How many times this question has already been handed back. Carried across the
    # resume boundary because the graph starts fresh there: without it, "ask once" would
    # mean "ask once per graph run", which is every run.
    hitl_rounds: int = Field(default=0, ge=0)
    complexity: Optional[Literal["simple", "complex"]] = None
    complexity_reason: Optional[str] = None
    sub_questions: List[str] = Field(default_factory=list, max_length=4)
    # Conditions set before the clarification was asked ("grades up to Year 6").
    # Carried across the resume boundary for the same reason as `hitl_rounds`: the graph
    # starts fresh there, and the turn that established them is several messages back by
    # the time the user answers.
    carried_constraints: List[str] = Field(default_factory=list, max_length=8)


class PendingHitlState(StrictSchema):
    id: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    options: List[str] = Field(default_factory=list)
    route: Literal["clarify", "scope_select"]
    retrieval_status: Literal["needs_clarification", "needs_scope_selection"]
    answers: List[str] = Field(default_factory=list)
    resume_state: HitlResumeState
    created_at: str


def _normalize_chunks(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    fields = RetrievedChunk.model_fields
    return [
        RetrievedChunk.model_validate({key: item[key] for key in fields if key in item}).model_dump(
            exclude_none=True
        )
        for item in value
        if isinstance(item, dict) and item.get("filename")
    ]


def _normalize_trace_fields(trace: dict, fields: dict) -> dict:
    normalized = {key: trace[key] for key in fields if key in trace}
    for key in (
        "retrieved_chunks",
        "initial_retrieved_chunks",
        "rewrite_retrieved_chunks",
        "hitl_targeted_retrieved_chunks",
    ):
        if key in normalized:
            normalized[key] = _normalize_chunks(normalized[key])
    return normalized


def normalize_rag_sub_trace(trace: dict | None) -> Optional[dict]:
    if not isinstance(trace, dict) or not trace:
        return None
    normalized = _normalize_trace_fields(trace, RagSubTrace.model_fields)
    return RagSubTrace.model_validate(normalized).model_dump(exclude_none=True)


def normalize_rag_trace(trace: dict | None) -> Optional[dict]:
    if not isinstance(trace, dict) or not trace:
        return None
    normalized = _normalize_trace_fields(trace, RagTrace.model_fields)
    if "sub_traces" in normalized:
        sub_traces = normalized["sub_traces"] if isinstance(normalized["sub_traces"], list) else []
        normalized["sub_traces"] = [
            item
            for item in (
                normalize_rag_sub_trace(sub_trace)
                for sub_trace in sub_traces
                if isinstance(sub_trace, dict)
            )
            if item is not None
        ]
    return RagTrace.model_validate(normalized).model_dump(exclude_none=True)


class ChatResponse(StrictSchema):
    response: str
    rag_trace: Optional[RagTrace] = None
    # Duplicated out of the trace so a client can show images without depending on
    # the trace's shape, which is diagnostic and may change.
    assets: List[AssetReference] = Field(default_factory=list)


class MessageInfo(StrictSchema):
    # The row id, and the cursor a client scrolls back from. Optional because a message
    # cached before ids were stored has none; such a page simply cannot be paged from.
    id: Optional[int] = None
    type: str
    content: str
    timestamp: str
    rag_trace: Optional[RagTrace] = None


class SessionMessagesResponse(StrictSchema):
    # Oldest-first: reading order, whichever batch this is.
    messages: List[MessageInfo]
    # Whether anything older than `messages[0]` exists. A client scrolling back stops
    # asking when this is false, rather than probing until it gets an empty page.
    has_more: bool = False


class SessionInfo(StrictSchema):
    session_id: str
    title: Optional[str] = None
    updated_at: str
    message_count: int


class SessionListResponse(StrictSchema):
    sessions: List[SessionInfo]


class SessionDeleteResponse(StrictSchema):
    session_id: str
    message: str
