from typing import Annotated, Any, Literal, TypedDict, List, Optional
import logging
import operator
import os
import re
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from pydantic import BaseModel, Field

from backend.chat.request_context import ChatRequestContext
from backend.llm import sampling
from backend.rag.evidence import (
    AssessmentContext,
    Certainty,
    ChunkAssessment,
    EvidenceReport,
    build_ladder,
)
from backend.prompts import resolve as resolve_prompt
from backend.rag.policy import decide_route, offerable_directions, select_context_indices
from backend.rag.rerank_assessor import CrossEncoderAssessor
from backend.assets.vision import call_with_rate_limit_retry
from backend.profiles import get_profile
from backend.schemas.chat import HitlResumeState, normalize_rag_sub_trace
from backend.rag.utils import (
    RETRIEVAL_TOP_K,
    retrieve_documents,
    rewrite_query_once,
    dedupe_documents,
    retrieval_trace_fields,
)

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
BASE_URL = os.getenv("BASE_URL")
FAST_MODEL = os.getenv("FAST_MODEL")
GRADE_MODEL = os.getenv("GRADE_MODEL")

# Prompts, routing budgets, and the fast-path vocabulary are all profile data.
_PROFILE = get_profile()
_RAG = _PROFILE.rag
_COPY = _PROFILE.user_copy

_grader_model = None
_complexity_model = None


class _RetryBudget:
    """Adapts the rag section to the retry helper's attribute names, which were coined
    for the vision path. One helper, two callers, no duplicated backoff logic."""

    vision_retry_attempts = _RAG.model_retry_attempts
    vision_retry_base_seconds = _RAG.model_retry_base_seconds
    vision_retry_max_seconds = _RAG.model_retry_max_seconds


_RETRY = _RetryBudget()


def _get_grader_model():
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            stream_usage=True,
            **sampling("grade"),
        )
    return _grader_model


def _get_complexity_model():
    """FAST_MODEL is used for question-complexity classification and sub-question decomposition."""
    global _complexity_model
    if not API_KEY or not FAST_MODEL:
        return None
    if _complexity_model is None:
        _complexity_model = init_chat_model(
            model=FAST_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            stream_usage=True,
            **sampling("planner"),
        )
    return _complexity_model


EVIDENCE_GRADE_PROMPT = _RAG.evidence_grade_prompt

# Whether the complexity planner and the decomposition branch exist at all this
# process. Read once here rather than at each call site so the graph shape and the
# state it starts from can never disagree about it.
COMPLEXITY_PLANNING_ENABLED = _RAG.complexity_planning_enabled


class EvidenceGrade(BaseModel):
    """Structured evidence grade: judges relevance, answerability, and the next routing step together.

    These descriptions are the ONLY definition of what each value means. They are
    serialized into the JSON schema and sent on the same call as the prompt, so stating
    the semantics in `evidence_grade_prompt` as well was paying for them twice. The
    prompt now defines only the route decision, which reads across these fields.

    Field order was briefly changed to put `reason` first, on the theory that a model
    emitting JSON in declaration order should reason before it commits. That is sound
    for a non-reasoning model and wrong here: this deployment runs reasoning models,
    which already think before emitting, so an explicit reason field made them reason
    twice and put 2-3 sentences of prose on the critical path of every grading call.
    Latency went up, quality did not. Leave `reason` where it is.
    """

    relevance: Literal["none", "weak", "strong"] = Field(
        description=(
            "Topical relevance of the snippets to the question. none: the topic is "
            "unrelated. weak: the topic is close but the evidence is thin. strong: "
            "clearly relevant."
        )
    )
    answerability: Literal["none", "partial", "sufficient"] = Field(
        description=(
            "Whether the snippets can answer the question. none: they cannot. partial: "
            "some clues, but not enough to be definitive. sufficient: they support an "
            "answer, directly or jointly."
        )
    )
    ambiguity: Literal["none", "missing_slot", "multiple_candidates"] = Field(
        default="none",
        description=(
            "Whether the question under-specifies what to look for. missing_slot: a key "
            "condition is absent (role name, version, file type, module name, product "
            "line). multiple_candidates: several directions could all be relevant. "
            "none: no clear ambiguity."
        )
    )
    constraints_discriminate: Literal["yes", "no", "unknown"] = Field(
        default="unknown",
        description=(
            "Only when conditions carried over from earlier turns were listed. yes: the "
            "snippets give different answers depending on those conditions, so the answer "
            "must be narrowed to them. no: the snippets give one answer that applies "
            "regardless of them. unknown: not enough to tell, or none were listed."
        ),
    )
    route: Literal["answer", "rewrite", "clarify", "scope_select", "no_knowledge"] = Field(
        description="The next routing step"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_slots: List[str] = Field(default_factory=list)
    hitl_prompt: str = ""
    hitl_options: List[str] = Field(default_factory=list)
    reason: str = ""
    supporting_chunks: List[int] = Field(
        default_factory=list,
        description=(
            "1-based numbers of the retrieved snippets that actually carry the evidence, "
            "as shown in the [n] markers. Leave empty if every snippet contributes."
        ),
    )


class ComplexityResult(BaseModel):
    """Question complexity classification result."""

    complexity: Literal["simple", "complex"] = Field(
        description="Question complexity: 'simple' for simple questions, 'complex' for complex questions"
    )
    reason: str = Field(default="", description="Classification reason")
    sub_questions: List[str] = Field(
        default_factory=list,
        description="2-4 independently retrievable sub-questions for a complex question; left empty for a simple question",
        max_length=_RAG.max_sub_questions,
    )


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    retrieval_status: Optional[str]
    retrieval_failed: Optional[bool]
    evidence_relevance: Optional[str]
    evidence_answerability: Optional[str]
    evidence_ambiguity: Optional[str]
    evidence_confidence: Optional[float]
    missing_slots: Optional[List[str]]
    hitl_prompt: Optional[str]
    hitl_options: Optional[List[str]]
    rewrite_count: int
    # HITL turns already spent on this question, surviving the resume boundary.
    hitl_rounds: int
    rewrite_method: Optional[str]
    rewritten_query: Optional[str]
    step_back_question: Optional[str]
    hyde_document: Optional[str]
    rag_trace: Optional[dict]
    # Fields added for complexity routing
    complexity: Optional[str]
    complexity_reason: Optional[str]
    sub_questions: Optional[List[str]]
    is_sub_agent: bool
    sub_results: Annotated[List[dict], operator.add]
    request_context: ChatRequestContext
    rag_step_group: Optional[str]
    rag_step_group_label: Optional[str]
    # Corpus sections the turn planner matched, forwarded as a retrieval hint. None
    # means no planner ran or it abstained. Never a filter — see chat/turn_policy.
    retrieval_sections: Optional[List[str]]
    # Catalogued questions the planner matched, one per section. Routing offers these
    # as a scope_select rather than guessing with a rewrite — see policy.decide_route.
    scope_options: Optional[List[str]]
    # Conditions carried over from earlier turns ("grades up to Year 6"). Appended to
    # the retrieval query and stated to the answering model, because narrowing the
    # search is only half of it — the right fee tables still yield an answer covering
    # every year group unless something tells the model which years were asked about.
    carried_constraints: Optional[List[str]]
    # Whether this turn inherited its subject from the conversation. Routing reads it to
    # decide whether a choice of corpus directions could possibly narrow anything.
    is_followup: Optional[bool]


def _format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)


def _copy_jsonable_doc(doc: dict) -> dict:
    """Keep resume snapshots small and JSON-safe."""
    allowed = {
        "filename",
        "page_number",
        "text",
        "score",
        "rrf_rank",
        "rerank_score",
        "chunk_id",
        "doc_id",
        "asset_ids",
        "modality",
    }
    return {key: value for key, value in doc.items() if key in allowed}


def _copy_jsonable_docs(docs: List[dict] | None) -> List[dict]:
    return [_copy_jsonable_doc(doc) for doc in (docs or []) if isinstance(doc, dict)]


def _is_hitl_result(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    trace = result.get("rag_trace") or {}
    status = result.get("retrieval_status") or trace.get("retrieval_status")
    route = result.get("route") or trace.get("route")
    return status in ("needs_clarification", "needs_scope_selection") or route in ("clarify", "scope_select")


def _build_hitl_resume_state(result: dict) -> dict:
    trace = result.get("rag_trace") or {}
    return HitlResumeState(
        question=result.get("question") or trace.get("query") or "",
        route=result.get("route") or trace.get("route"),
        retrieval_status=result.get("retrieval_status") or trace.get("retrieval_status"),
        rewrite_count=int(result.get("rewrite_count") or 0),
        hitl_rounds=int(result.get("hitl_rounds") or 0),
        complexity=result.get("complexity") or trace.get("complexity"),
        complexity_reason=result.get("complexity_reason") or trace.get("complexity_reason"),
        sub_questions=result.get("sub_questions") or trace.get("sub_questions") or [],
        # Conditions the user set before the clarification. They survive the resume
        # boundary or they are lost: the graph starts fresh there, and the turn that
        # established "up to Year 6" is several messages back by the time the user
        # answers.
        carried_constraints=list(result.get("carried_constraints") or []),
    ).model_dump()


def _refined_question_for_hitl(
    resume_state: dict,
    user_answer: str,
    resolved=None,
    original_question: str = "",
) -> str:
    """The question a resumed turn should actually search for.

    The resolved form when a resolver produced one, and this is the whole reason the
    resolver is reachable from here. What it replaced was `f"{answer}: {question}"`,
    and string formatting cannot express the thing a clarification reply most often
    does: replace. "no i mean the school fees" concatenated onto the reading it was
    correcting produces a query containing both readings, retrieves both, and answers
    from the union — which is exactly what the user was trying to stop.

    The fallback keeps the old shape but anchors on the USER's question rather than
    `resume_state["question"]`. That field holds the query the AGENT wrote for the tool,
    so any condition the user set and the agent did not repeat was already gone before
    this function ever saw it.
    """
    if resolved is not None and getattr(resolved, "resolved", False):
        refined = (getattr(resolved, "question", "") or "").strip()
        if refined:
            return refined

    question = (original_question or resume_state.get("question") or "").strip()
    answer = user_answer.strip()
    if not question:
        return answer
    if answer and answer in question:
        return question
    return f"{answer}: {question}" if answer else question


def _emit(state: RAGState, icon: str, label: str, detail: str = "") -> None:
    ctx = state["request_context"]
    ctx.emit_rag_step(
        icon,
        label,
        detail,
        group=state.get("rag_step_group"),
        group_label=state.get("rag_step_group_label"),
    )


def _initial_state(
    question: str,
    ctx: ChatRequestContext,
    *,
    is_sub_agent: bool = False,
    rag_step_group: Optional[str] = None,
    rag_step_group_label: Optional[str] = None,
) -> dict:
    return {
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "retrieval_status": None,
        "retrieval_failed": False,
        "evidence_relevance": None,
        "evidence_answerability": None,
        "evidence_ambiguity": None,
        "evidence_confidence": None,
        "missing_slots": [],
        "hitl_prompt": "",
        "hitl_options": [],
        "rewrite_count": 0,
        "hitl_rounds": 0,
        "rewrite_method": None,
        "rewritten_query": None,
        "step_back_question": None,
        "hyde_document": None,
        "rag_trace": None,
        "complexity": None,
        # Left unset when planning is off, because nothing classified this question and
        # writing "simple" would be the fabricated-grade pattern evidence.py exists to
        # prevent. The reason still says so, so a trace shows absence, not silence.
        "complexity_reason": None if COMPLEXITY_PLANNING_ENABLED else "complexity_planning_disabled",
        "sub_questions": None,
        "is_sub_agent": is_sub_agent,
        "sub_results": [],
        "request_context": ctx,
        "rag_step_group": rag_step_group,
        "rag_step_group_label": rag_step_group_label,
        "retrieval_sections": list(getattr(ctx, "retrieval_sections", None) or []),
        "scope_options": list(getattr(ctx, "scope_options", None) or []),
        "carried_constraints": list(getattr(ctx, "carried_constraints", None) or []),
        "is_followup": bool(getattr(ctx, "is_followup", False)),
    }


def _search_query(state: RAGState) -> str:
    """The text this state searches for: the question, and nothing appended to it.

    An earlier revision appended the turn's carried conditions here, on the reasoning
    that the agent's tool query might omit a condition the user set two turns ago.
    Measured over a twenty-turn conversation (tests/test_conversation_sequence.py) that
    cost three of twenty turns outright: "what time does the school day start (the child
    is 5 years old)" ranks the term-dates passage below anything sharing the words
    "child" or "years", because those terms appear nowhere in a passage about opening
    hours. Every query term absent from the target passage is dilution, and a condition
    is absent from every passage the corpus wrote once for everybody.

    The condition still reaches retrieval when it belongs there — the RESOLVED question
    states it in natural language ("the fees for the years up to Year 6"), which is
    vocabulary the fee table actually contains. What it must not do is arrive twice, or
    arrive as bare keywords stapled to a question about something else.

    Constraints now travel to the two stages that can act on them without costing
    recall: the grader, which reports whether the material varies by them, and the
    answer prompt, which narrows or does not on that verdict.
    """
    return state["question"]


def retrieve_initial(state: RAGState) -> RAGState:
    query = _search_query(state)
    _emit(state, "🔍", "Searching the knowledge base...", "Initial retrieval")
    if state.get("carried_constraints"):
        _emit(
            state, "🧷", "Conditions carried from earlier turns",
            "Applied when the answer is written, not to the search — "
            + "; ".join(state["carried_constraints"]),
        )
    retrieved = retrieve_documents(query, top_k=RETRIEVAL_TOP_K)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    retrieval_failed = retrieve_meta.get("retrieval_mode") == "failed"
    context = _format_docs(results)
    if retrieval_failed:
        _emit(
            state,
            "🚧",
            "Knowledge base temporarily unreachable",
            retrieve_meta.get("retrieval_error") or "retrieval backend error",
        )
    else:
        _emit(
            state,
            "🧱",
            "Three-tier chunk retrieval",
            (
                f"Leaf level L{retrieve_meta.get('leaf_retrieve_level', 3)} recall, "
                f"candidates {retrieve_meta.get('candidate_k', 0)}"
            ),
        )
        _emit(
            state,
            "🧩",
            "Auto-merging",
            (
                f"Enabled: {bool(retrieve_meta.get('auto_merge_enabled'))}, "
                f"Applied: {bool(retrieve_meta.get('auto_merge_applied'))}, "
                f"Replaced chunks: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        _emit(state, "✅", f"Retrieval complete, found {len(results)} snippets", f"Mode: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
        if not results:
            _emit(state, "⚠️", "No snippets available, proceeding to the evidence-grading short-circuit check")
    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "retrieved_chunks": results,
        "initial_retrieved_chunks": results,
        "retrieval_stage": "initial",
        "complexity": state.get("complexity"),
        "complexity_reason": state.get("complexity_reason"),
        **retrieval_trace_fields(retrieve_meta),
    }
    return {
        "query": query,
        "docs": results,
        "context": context,
        "retrieval_failed": retrieval_failed,
        "rag_trace": rag_trace,
    }


def _route_after_initial(state: RAGState) -> Literal["grade_documents"]:
    return "grade_documents"


def _route_after_grade(state: RAGState) -> Literal["rewrite_question", "end"]:
    if state.get("route") == "rewrite":
        return "rewrite_question"
    return "end"


def _route_after_rewrite(state: RAGState) -> Literal["retrieve_rewritten", "end"]:
    """Only re-retrieve when a rewrite was actually planned.

    This edge used to be unconditional, so a node that gave up on planning one fell
    into `retrieve_rewritten`, which requires `rewrite_method` and raises `ValueError`
    without it. A planner that returned nothing — an unconfigured FAST_MODEL, a
    provider error, a response that failed schema validation — therefore failed the
    whole turn, and did it on the path meant to degrade gracefully.
    """
    return "retrieve_rewritten" if state.get("rewrite_method") else "end"


def _retrieval_status_for_route(route: str, report: EvidenceReport) -> str:
    if route == "answer":
        # Only `sufficient` earns "answerable". Everything else routed to answer rests
        # on less than the question asked for — including `none`, which reaches this
        # route now that on-subject evidence is never denied. That distinction is what
        # the knowledge tool reads to tell the model to answer from what it has and name
        # the gap, so collapsing it into "answerable" would silently drop the guidance
        # on exactly the turns that need it.
        return "answerable" if report.sufficiency == "sufficient" else "partial"
    if route == "rewrite":
        return "needs_rewrite"
    if route == "clarify":
        return "needs_clarification"
    if route == "scope_select":
        return "needs_scope_selection"
    if route == "retrieval_error":
        return "retrieval_error"
    return "no_knowledge"


def _default_hitl_prompt(route: str, report: EvidenceReport) -> str:
    if report.hitl_prompt:
        return report.hitl_prompt
    if route == "scope_select":
        return _COPY.hitl_scope_default
    if report.missing_slots:
        return _COPY.hitl_clarify_missing_slots + ", ".join(report.missing_slots)
    return _COPY.hitl_clarify_default


class LLMGraderAssessor:
    """The top rung: a model reads the question and the chunks together.

    The only rung that can judge meaning, spot an under-specified question, or name
    which snippets actually carry the answer — and the only one that costs a call. It
    lives here rather than in evidence.py because the model factory and the prompt do.

    It is also asked to do double duty. Naming its supporting chunks costs a handful of
    output tokens on a call already being made, and it turns context trimming from a
    lexical guess into a consequence of the same judgement that approved the answer.
    """

    name = "llm_grader"
    certainty = Certainty.HIGH

    def assess(self, ctx: AssessmentContext) -> Optional[EvidenceReport]:
        grader = _get_grader_model()
        if not grader:
            raise RuntimeError("GRADE_MODEL is required for evidence grading")

        prompt = resolve_prompt(
            EVIDENCE_GRADE_PROMPT,
            "rag/evidence_grade.j2",
            question=ctx.question,
            context=_format_docs(ctx.docs),
            # Alongside the question, never folded into it. See AssessmentContext.
            constraints=list(ctx.constraints),
        )
        grade = grader.with_structured_output(EvidenceGrade).invoke(
            [{"role": "user", "content": prompt}]
        )

        total = len(ctx.docs)
        cited = {n for n in (grade.supporting_chunks or []) if 1 <= n <= total}
        chunks = []
        for index in range(1, total + 1):
            # supported stays None when the grader named nothing, because "it did not
            # tell us" and "it excluded this chunk" are different facts.
            chunks.append(ChunkAssessment(index=index, supported=(index in cited) if cited else None))

        return EvidenceReport(
            question=ctx.question,
            chunks=chunks,
            certainty=Certainty.HIGH,
            relevance=grade.relevance,
            sufficiency=grade.answerability,
            ambiguity=grade.ambiguity,
            constraints_discriminate=grade.constraints_discriminate,
            confidence=grade.confidence,
            preferred_route=grade.route,
            missing_slots=list(grade.missing_slots),
            hitl_prompt=grade.hitl_prompt,
            hitl_options=list(grade.hitl_options),
            assessed_by=[self.name],
            reasons=[grade.reason or f"grader routed to {grade.route}"],
        )


def _assess_evidence(state: RAGState) -> EvidenceReport:
    """Climb the ladder for this retrieval, once.

    Rungs are ordered by the certainty each can establish, which is also their cost, so
    the grader is reached only when the cheaper rungs cannot conclude.
    """
    docs = state.get("docs") or []
    ladder = build_ladder(_RAG, extra=[CrossEncoderAssessor(), LLMGraderAssessor()])
    return ladder.run(
        AssessmentContext(
            # The QUESTION, not the constrained query. Folding the conditions in was a
            # mistake with one visible failure mode: a grader asked whether a general
            # admissions document list answers "what documents are required (grades up
            # to Year 6)" sees no year mentioned anywhere, honestly returns
            # `relevance: none`, and `decide_route` denies — which is the one outcome
            # that policy exists to make impossible while material is in hand.
            #
            # The conditions travel beside the question instead, and the grader reports
            # on them in `constraints_discriminate` rather than scoring against them.
            question=state["question"],
            docs=docs,
            retrieval_meta=state.get("rag_trace") or {},
            config=_RAG,
            constraints=list(state.get("carried_constraints") or []),
        )
    )


def _report_update(
    report: EvidenceReport,
    route: str,
    scope_options: List[str] | None = None,
    *,
    is_followup: bool = False,
) -> dict:
    """Flatten a report plus its route into the trace fields the rest of the system
    already reads. Names are unchanged so the frontend and any integrating client keep
    working; `evidence_certainty` and `evidence_assessed_by` are additive."""
    hitl_prompt = _default_hitl_prompt(route, report) if route in ("clarify", "scope_select") else ""
    # The grader's own list wins when it wrote one. The catalogued directions are the
    # fallback, and on a scope_select routed BY those directions they are the only list
    # there is — a scope_select with no options is never asked, so without this the
    # route decided in policy would be silently dropped here.
    #
    # `offerable_directions` is applied rather than the raw list, so this fallback obeys
    # the same rule routing does. Without it a follow-up could still be handed the
    # catalogue neighbours that routing had just refused to ask about, arriving through
    # a grader-initiated scope_select instead.
    options = list(report.hitl_options) or offerable_directions(
        scope_options or [], is_followup=is_followup
    )
    return {
        **report.as_trace(),
        "retrieval_status": _retrieval_status_for_route(route, report),
        "hitl_prompt": hitl_prompt,
        "hitl_options": options if route == "scope_select" else list(report.hitl_options),
        "route": route,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    """Assess the retrieved evidence once, then apply the policies that read it.

    This node orchestrates and nothing more: it holds no thresholds, computes no
    signals, and does not know which assessors exist. Assessment produces one report;
    routing and context sizing are pure functions over it.
    """
    if state.get("retrieval_failed"):
        # A backend outage says nothing about what the KB contains, so this must not
        # reach an assessor or surface as no_knowledge. Static short-circuit only.
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update({
            "retrieval_status": "retrieval_error",
            "route": "retrieval_error",
            "evidence_reason": "knowledge_base_unreachable",
        })
        _emit(state, "\U0001f6a7", "Retrieval failed, returning a static retry notice", "Not treated as missing knowledge")
        return {
            "route": "retrieval_error",
            "retrieval_status": "retrieval_error",
            "docs": [],
            "context": "",
            "rag_trace": rag_trace,
        }

    docs = state.get("docs") or []
    if docs:
        _emit(state, "\U0001f4ca", "Evaluating evidence quality...")

    report = _assess_evidence(state)
    scope_options = list(state.get("scope_options") or [])
    route, route_reason = decide_route(
        report,
        has_docs=bool(docs),
        rewrite_count=int(state.get("rewrite_count") or 0),
        is_sub_agent=bool(state.get("is_sub_agent")),
        config=_RAG,
        hitl_rounds=int(state.get("hitl_rounds") or 0),
        scope_options=scope_options,
        is_followup=bool(state.get("is_followup")),
    )

    report_update = _report_update(
        report, route, scope_options, is_followup=bool(state.get("is_followup"))
    )
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(report_update)
    rag_trace["route_reason"] = route_reason

    if route == "answer":
        if report.sufficiency == "partial":
            _emit(state, "\U0001f7e1", "Keeping partially relevant evidence", f"Confidence: {report.confidence:.2f}")
        else:
            _emit(state, "\u2705", "Evidence sufficient, returning retrieved snippets", f"Confidence: {report.confidence:.2f}")
    elif route == "rewrite":
        _emit(state, "\u26a0\ufe0f", "Evidence insufficient, will rewrite the query once", f"Confidence: {report.confidence:.2f}")
    elif route in ("clarify", "scope_select"):
        _emit(state, "\u2753", "Needs more information from the user", report_update["hitl_prompt"])
    elif route == "retrieval_error":
        # Retrieval worked but assessment could not reach the required standard, so
        # neither answering nor denying would be honest. Say "try again" instead.
        _emit(state, "\U0001f6a7", "Evidence could not be assessed", route_reason)
    else:
        _emit(state, "\u26d4", "No usable evidence found in the knowledge base", route_reason)

    update = {
        "route": route,
        "retrieval_status": report_update["retrieval_status"],
        "evidence_relevance": report.relevance,
        "evidence_answerability": report.sufficiency,
        "evidence_ambiguity": report.ambiguity,
        "evidence_confidence": report.confidence,
        "missing_slots": list(report.missing_slots),
        "hitl_prompt": report_update["hitl_prompt"],
        "hitl_options": list(report_update["hitl_options"]),
        "rag_trace": rag_trace,
    }

    if route in ("no_knowledge", "clarify", "scope_select", "retrieval_error"):
        # Dropping the chunks from the TRACE too, not just from the state. Asset
        # attachment falls back to `rag_trace.retrieved_chunks` whenever the knowledge
        # tool pinned nothing (backend/chat/assets_bridge.py), and on these routes it
        # pins nothing by design — so leaving them here attached the figures from
        # chunks this turn just decided not to answer from. "The knowledge base has no
        # reliable information on this" arriving with the picture that answers the
        # question is the most confusing thing the system can do. `initial_retrieved_chunks`
        # still holds the full set for the trace panel.
        if docs:
            rag_trace["retrieved_chunks"] = []
        update.update({"docs": [], "context": ""})
        return update

    if route == "answer" and docs:
        keep, reason = select_context_indices(
            report, docs, _RAG, answer_ceiling=RETRIEVAL_TOP_K,
        )
        rag_trace["context_selection_reason"] = reason
        rag_trace["context_chunks_available"] = len(docs)
        if keep:
            kept = [docs[i - 1] for i in keep]
            _emit(
                state, "\u2702\ufe0f", f"Sending {len(kept)} of {len(docs)} chunks to the model", reason,
            )
            update.update({"docs": kept, "context": _format_docs(kept)})
            # retrieved_chunks has to match what the answer was built from: citation
            # markers and asset attribution both index into it. The full set stays in
            # initial_retrieved_chunks for the trace panel.
            rag_trace["retrieved_chunks"] = kept
            rag_trace["context_trimmed"] = True
            rag_trace["context_chunks_kept"] = len(kept)
        else:
            rag_trace["context_trimmed"] = False
            rag_trace["context_chunks_kept"] = len(docs)

    return update


def _stop_rewriting(state: RAGState, reason: str, label: str, detail: str) -> RAGState:
    """End the rewrite branch without re-retrieving, keeping the first pass.

    The rewrite is a bonus second attempt, so failing to plan one must cost the turn
    nothing it already had. Both callers used to return `no_knowledge` with `docs`
    cleared — which denied knowledge the first pass had retrieved and graded on-subject,
    the exact outcome `decide_route` exists to prevent — while their own step message
    said "answering from the first pass only". The first pass is now what the turn
    answers from, and only a genuinely empty one still denies.
    """
    docs = state.get("docs") or []
    status = "partial" if docs else "no_knowledge"
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "retrieval_status": status,
        "route": "answer" if docs else "no_knowledge",
        "evidence_reason": reason,
    })
    if not docs:
        rag_trace["retrieved_chunks"] = []
    _emit(state, "⚠️" if docs else "⛔", label, detail)
    return {
        "route": "answer" if docs else "no_knowledge",
        "retrieval_status": status,
        "docs": docs,
        "context": state.get("context") or "",
        "rag_trace": rag_trace,
    }


def rewrite_question_node(state: RAGState) -> RAGState:
    question = _search_query(state)
    _emit(state, "✏️", "Rewriting the query...")

    rewrite_count = int(state.get("rewrite_count") or 0)
    if rewrite_count >= _RAG.max_rewrites:
        return _stop_rewriting(
            state,
            "rewrite_budget_exhausted",
            "Rewrite budget exhausted, stopping retrieval",
            "Answering from what the earlier passes found",
        )

    _emit(state, "🧠", "Choosing between Step-back / HyDE rewrite")
    rewrite = rewrite_query_once(question)
    if not rewrite:
        return _stop_rewriting(
            state,
            "rewrite_unavailable",
            "Could not plan a rewrite, stopping retrieval",
            "Answering from the first pass only",
        )

    rewrite_method = (rewrite.get("rewrite_method") or "").strip()
    step_back_question = (rewrite.get("step_back_question") or "").strip()
    hyde_document = (rewrite.get("hyde_document") or "").strip()
    rewritten_query = (rewrite.get("rewritten_query") or "").strip()

    method_label = "Step-back" if rewrite_method == "step_back" else "HyDE"
    _emit(state, "✅", f"Selected {method_label} rewrite", "Only this rewrite method will run this round")

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_method": rewrite_method,
        "rewritten_query": rewritten_query,
        "rewrite_count": rewrite_count + 1,
    })
    if step_back_question:
        rag_trace["step_back_question"] = step_back_question
    if hyde_document:
        rag_trace["hyde_document"] = hyde_document

    return {
        "rewrite_method": rewrite_method,
        "rewritten_query": rewritten_query,
        "step_back_question": step_back_question,
        "hyde_document": hyde_document,
        "rewrite_count": rewrite_count + 1,
        "rag_trace": rag_trace,
    }


def retrieve_rewritten(state: RAGState) -> RAGState:
    rewrite_method = (state.get("rewrite_method") or "").strip()
    if rewrite_method not in ("step_back", "hyde"):
        raise ValueError("rewrite_method is required for rewritten retrieval")
    rewritten_query = (state.get("rewritten_query") or "").strip()
    if not rewritten_query:
        raise ValueError("rewritten_query is required for rewritten retrieval")
    method_label = "Step-back" if rewrite_method == "step_back" else "HyDE"
    _emit(state, "🔄", f"Re-retrieving with the {method_label} query...")
    retrieved = retrieve_documents(rewritten_query, top_k=RETRIEVAL_TOP_K)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    retrieval_failed = retrieve_meta.get("retrieval_mode") == "failed"

    # The rewrite ADDS to the first pass, it does not replace it. A step-back or HyDE
    # query is a different question by construction, so its results can easily miss a
    # chunk the literal question found — and that chunk was already graded on-subject,
    # or this branch would not be running. Replacing the set meant a rewrite could
    # leave the turn with less evidence than it started with, and the second grading
    # pass would then deny knowledge the first pass had in hand.
    #
    # Rewritten results lead because they are the targeted retry; the first pass keeps
    # whatever they did not rediscover. The graded set can reach 2 x top_k on the
    # minority of turns that rewrite at all, and adaptive context selection trims it
    # back to the chunks the grader cites before any of it reaches the answer prompt.
    first_pass = [] if retrieval_failed else list(state.get("docs") or [])
    merged = dedupe_documents(results + first_pass)
    for index, item in enumerate(merged, 1):
        item["rrf_rank"] = index
    context = _format_docs(merged)
    if retrieval_failed:
        _emit(
            state,
            "🚧",
            "Knowledge base temporarily unreachable during rewritten retrieval",
            retrieve_meta.get("retrieval_error") or "retrieval backend error",
        )
    else:
        _emit(
            state,
            "🧱",
            f"{method_label} three-tier retrieval",
            (
                f"L{retrieve_meta.get('leaf_retrieve_level', 3)} recall, "
                f"candidates {retrieve_meta.get('candidate_k', 0)}, "
                f"merge-replaced {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        _emit(
            state,
            "✅",
            f"Rewritten retrieval complete, {len(merged)} snippets total",
            f"{len(results)} from the {method_label} query, "
            f"{len(merged) - len(results)} kept from the first pass",
        )
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_method": rewrite_method,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": merged,
        # The rewritten pass ALONE, so a trace still shows what the rewrite itself
        # found rather than the union it was folded into.
        "rewrite_retrieved_chunks": results,
        "retrieval_stage": "rewritten",
        **retrieval_trace_fields(retrieve_meta),
    })
    if state.get("step_back_question"):
        rag_trace["step_back_question"] = state["step_back_question"]
    if state.get("hyde_document"):
        rag_trace["hyde_document"] = state["hyde_document"]
    return {"docs": merged, "context": context, "retrieval_failed": retrieval_failed, "rag_trace": rag_trace}


# ---------------------------------------------------------------------------
# Complexity classification & sub-question decomposition
# ---------------------------------------------------------------------------

COMPLEXITY_PROMPT = _RAG.complexity_prompt

# Fast-path classification vocabulary. These are language- AND domain-specific, so
# they are profile data: an e-commerce catalogue needs product-attribute markers where
# a school corpus needs admissions vocabulary. Tuples keep the original membership-test
# semantics at the call sites below.
_SIMPLE_OVERRIDE_MARKERS = tuple(_RAG.simple_override_markers)
_SIMPLE_QUERY_MARKERS = tuple(_RAG.simple_query_markers)
_COMPLEX_QUERY_MARKERS = tuple(_RAG.complex_query_markers)
_QUERY_DIMENSION_MARKERS = tuple(_RAG.query_dimension_markers)



def _simple_question_fast_path_reason(question: str) -> Optional[str]:
    """Return a reason only when a local rule can confidently classify a simple query."""
    normalized = re.sub(r"\s+", " ", (question or "").strip()).lower()
    if not normalized or len(normalized) > _RAG.fast_path_max_chars:
        return None
    # Overrides run FIRST. "how many students" is a single-fact lookup, but it contains
    # "how ", which also opens genuinely analytical questions. Without this the complex
    # marker wins and a plain lookup is sent to the planner — which may decompose it
    # into several sub-questions, each paying its own retrieval and grader call.
    if any(marker in normalized for marker in _SIMPLE_OVERRIDE_MARKERS):
        return "obvious_simple_fast_path:single_fact_override"
    if any(marker in normalized for marker in _COMPLEX_QUERY_MARKERS):
        return None
    if "、" in normalized:
        return None
    if re.search(r"[一-鿿]", normalized) and normalized.count(" ") >= 2:
        return None
    if sum(marker in normalized for marker in _QUERY_DIMENSION_MARKERS) >= 2:
        return None
    if sum(normalized.count(mark) for mark in ("?", "？", ";", "；")) > 1:
        return None
    if any(marker in normalized for marker in _SIMPLE_QUERY_MARKERS):
        return "obvious_simple_fast_path:single_fact_marker"
    # A wh-question opening directly onto a copula ("what are the partners") or with an
    # attribute in between ("what element is X"). Both are single-fact lookups that the
    # literal markers miss, and the first form is common enough that missing it sent
    # ordinary plural questions down the decomposition path.
    if re.match(
        r"^(what|which|who|where|when)\s+(?:\w+\s+)?(is|are|was|were|does|do)\b",
        normalized,
    ):
        return "obvious_simple_fast_path:wh_attribute_question"
    # Terminators include Arabic ؟ and ۔ — otherwise an Arabic question keeps its
    # mark and measures one character longer against the length rule below.
    if len(normalized.rstrip("?？。.!！؟۔،")) <= _RAG.fast_path_short_intent_chars:
        return "obvious_simple_fast_path:short_single_intent"
    return None


def classify_complexity(state: RAGState) -> RAGState:
    """Uses FAST_MODEL to determine question complexity."""
    question = state["question"]
    _emit(state, "🧭", "Analyzing question complexity...")

    fast_path_reason = _simple_question_fast_path_reason(question)
    if fast_path_reason:
        _emit(state, "⚡", "Fast-classified as a simple question → using the standard RAG flow")
        return {"complexity": "simple", "complexity_reason": fast_path_reason}

    model = _get_complexity_model()
    if not model:
        raise RuntimeError("FAST_MODEL is required for complexity planning")

    prompt = resolve_prompt(COMPLEXITY_PROMPT, "rag/complexity.j2", question=question)
    result = model.with_structured_output(ComplexityResult).invoke(
        [{"role": "user", "content": prompt}]
    )
    complexity = (result.complexity or "simple").strip().lower()
    reason = (result.reason or "").strip()
    sub_questions = [
        item.strip()
        for item in (result.sub_questions or [])
        if item and item.strip()
    ][: _RAG.max_sub_questions]
    if complexity not in ("simple", "complex"):
        raise ValueError(f"Unsupported complexity result: {complexity}")
    if complexity == "complex" and not sub_questions:
        raise ValueError("Complexity planner returned no sub-questions")

    if complexity == "simple":
        _emit(state, "✅", "Simple question → using the standard RAG flow", f"Reason: {reason[:60]}")
    else:
        _emit(state, "🔀", "Complex question → decomposing into sub-questions for parallel retrieval", f"Reason: {reason[:60]}")

    return {
        "complexity": complexity,
        "complexity_reason": reason,
        "sub_questions": sub_questions if complexity == "complex" else [],
    }


def prepare_sub_questions(state: RAGState) -> RAGState:
    """Emit the sub-questions produced by the complexity planner."""
    planned_sub_questions = [
        item.strip()
        for item in (state.get("sub_questions") or [])
        if item and item.strip()
    ]
    for i, sq in enumerate(planned_sub_questions, 1):
        _emit(state, "📌", f"Sub-question {i}", f"{sq[:80]} added to parallel retrieval")
    return {"sub_questions": planned_sub_questions}


def _route_after_complexity(state: RAGState):
    """Simple questions go straight to retrieval; complex questions retrieve the planned sub-questions in parallel."""
    if state.get("complexity") == "complex":
        return "prepare_sub_questions"
    return "retrieve_initial"


def _fanout_sub_questions(state: RAGState):
    """Dispatches the planned sub-questions to rag_sub_agent in parallel via the Send API."""
    sub_qs = state.get("sub_questions") or []
    ctx = state["request_context"]
    return [
        Send(
            "rag_sub_agent",
            _initial_state(
                sq,
                ctx,
                is_sub_agent=True,
                rag_step_group=f"Sub-question {i}",
                rag_step_group_label=sq,
            ),
        )
        for i, sq in enumerate(sub_qs, 1)
    ]


def _synthesis_report(
    docs: List[dict],
    retrieval_status: str,
    sub_results: List[dict],
) -> EvidenceReport:
    """The merged evidence report for a decomposed question.

    Synthesis does not assess anything itself — every document here was already graded by
    the sub-agent that retrieved it. So the report inherits HIGH certainty and records
    that provenance, rather than restating conclusions in its own words. Before this, this
    function hand-wrote `relevance="strong" if has_docs else "none"`, which is the same
    fabricated-grade pattern the ladder exists to remove.
    """
    assessed_by = ["synthesis"]
    for result in sub_results:
        for name in (result.get("rag_trace") or {}).get("evidence_assessed_by") or []:
            if name not in assessed_by:
                assessed_by.append(name)

    if not docs:
        return EvidenceReport(
            certainty=Certainty.HIGH,
            relevance="none",
            sufficiency="none",
            assessed_by=assessed_by,
            reasons=[f"no sub-question produced usable evidence ({retrieval_status})"],
        )

    return EvidenceReport(
        chunks=[ChunkAssessment(index=i) for i, _ in enumerate(docs, 1)],
        certainty=Certainty.HIGH,
        relevance="strong",
        sufficiency="partial" if retrieval_status == "partial" else "sufficient",
        preferred_route="answer",
        assessed_by=assessed_by,
        reasons=[f"merged {len(docs)} graded chunk(s) from {len(sub_results)} sub-question(s)"],
    )


def synthesis(state: RAGState) -> RAGState:
    """Merges all documents retrieved by the sub-agents, dedupes and ranks them, and outputs the final context."""
    sub_results = state.get("sub_results", [])
    _emit(state, "🔬", f"Synthesizing retrieval results from {len(sub_results)} sub-questions...")

    all_docs: List[dict] = []
    for result in sub_results:
        status = result.get("retrieval_status")
        if status not in ("answerable", "partial"):
            continue
        docs = result.get("docs", [])
        all_docs.extend(docs)

    deduped = dedupe_documents(all_docs)
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    # An outage on any sub-question with zero merged docs means the empty result reflects
    # backend availability, not corpus coverage — surface retry, not no_knowledge.
    retrieval_outage = (not deduped) and any(
        result.get("retrieval_status") == "retrieval_error" for result in sub_results
    )

    context = _format_docs(deduped)
    if deduped:
        _emit(state, "✅", f"Synthesis complete, {len(deduped)} deduplicated snippets total")
    elif retrieval_outage:
        _emit(state, "🚧", "Knowledge base temporarily unreachable for the sub-questions", "Returning a static retry notice")
    else:
        _emit(state, "⛔", "None of the sub-questions had usable evidence")

    # Merge rag_trace from all sub-agents
    sub_traces = []
    for result in sub_results:
        trace = result.get("rag_trace")
        if trace:
            normalized_trace = normalize_rag_sub_trace(trace)
            if normalized_trace:
                sub_traces.append(normalized_trace)

    original_trace = state.get("rag_trace") or {}
    has_docs = bool(deduped)
    retrieval_status = "answerable" if has_docs else "no_knowledge"
    if has_docs and any(result.get("retrieval_status") == "partial" for result in sub_results):
        retrieval_status = "partial"
    if retrieval_outage:
        retrieval_status = "retrieval_error"
    hitl_traces = [
        trace for trace in sub_traces
        if trace.get("retrieval_status") in ("needs_clarification", "needs_scope_selection")
    ]
    hitl_route = None
    hitl_prompt = ""
    hitl_options: List[str] = []
    # Outage outranks HITL: asking the user to clarify can't fix an unreachable backend.
    if not has_docs and not retrieval_outage and hitl_traces:
        scope_trace = next(
            (trace for trace in hitl_traces if trace.get("retrieval_status") == "needs_scope_selection"),
            None,
        )
        chosen_trace = scope_trace or hitl_traces[0]
        retrieval_status = chosen_trace.get("retrieval_status") or "needs_clarification"
        hitl_route = "scope_select" if retrieval_status == "needs_scope_selection" else "clarify"
        prompts = [
            trace.get("hitl_prompt")
            for trace in hitl_traces
            if trace.get("hitl_prompt")
        ]
        hitl_prompt = "; ".join(dict.fromkeys(prompts))
        for trace in hitl_traces:
            for option in trace.get("hitl_options") or []:
                if option not in hitl_options:
                    hitl_options.append(option)

    fallback_route = "retrieval_error" if retrieval_outage else "no_knowledge"
    route = "answer" if has_docs else (hitl_route or fallback_route)
    rag_trace = {
        **original_trace,
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": state["question"],
        "retrieved_chunks": deduped,
        "retrieval_stage": "synthesis",
        "complexity": "complex",
        "complexity_reason": state.get("complexity_reason", ""),
        "sub_questions": state.get("sub_questions", []),
        "sub_agent_count": len(sub_results),
        "synthesis_merged_count": len(all_docs),
        "sub_traces": sub_traces,
        "retrieval_status": retrieval_status,
        "route": route,
        "hitl_prompt": hitl_prompt,
        "hitl_options": hitl_options,
        **_synthesis_report(deduped, retrieval_status, sub_results).as_trace(),
    }

    return {
        "docs": deduped,
        "context": context,
        "route": route,
        "retrieval_status": retrieval_status,
        "hitl_prompt": hitl_prompt,
        "hitl_options": hitl_options,
        "rag_trace": rag_trace,
    }


def rag_sub_agent(state: RAGState) -> RAGState:
    """Run the only reachable sub-agent path directly: retrieve → grade."""
    question = state.get("question", "")
    result = dict(state)
    result.update(retrieve_initial(result))
    result.update(grade_documents_node(result))
    trace = result.get("rag_trace") or {}
    return {
        "sub_results": [{
            "question": question,
            "docs": result.get("docs", []),
            "retrieval_status": result.get("retrieval_status") or trace.get("retrieval_status"),
            "route": result.get("route") or trace.get("route"),
            "rag_trace": trace,
        }],
    }


# ---------------------------------------------------------------------------
# Main RAG graph
# ---------------------------------------------------------------------------

def build_rag_graph(complexity_planning_enabled: Optional[bool] = None):
    """Compile the retrieval graph for this process.

    The planning branch is registered or not, rather than registered and skipped. A
    disabled node that still exists costs a trace span, an edge to reason about, and a
    reader's attention every time they follow the graph; a node that was never added
    costs none of those. The argument exists so a caller can compile the other shape
    without reloading the module — production passes nothing and takes the profile's.
    """
    planning = (
        COMPLEXITY_PLANNING_ENABLED
        if complexity_planning_enabled is None
        else complexity_planning_enabled
    )
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_rewritten", retrieve_rewritten)

    if planning:
        graph.add_node("classify_complexity", classify_complexity)
        graph.add_node("prepare_sub_questions", prepare_sub_questions)
        graph.add_node("rag_sub_agent", rag_sub_agent)
        graph.add_node("synthesis", synthesis)

        # Entry point: complexity classification. Domain scope is decided one layer up,
        # in backend/chat/orchestrator.py — by the time this graph runs, the agent has
        # already chosen to search, so a gate here could no longer save the call it was
        # meant to.
        graph.set_entry_point("classify_complexity")

        # Simple questions go straight to retrieval; complex questions use the sub-questions the planner produced in one pass.
        graph.add_conditional_edges(
            "classify_complexity",
            _route_after_complexity,
            {
                "retrieve_initial": "retrieve_initial",
                "prepare_sub_questions": "prepare_sub_questions",
            },
        )

        graph.add_conditional_edges("prepare_sub_questions", _fanout_sub_questions)

        # Parallel sub-agents → synthesis
        graph.add_edge("rag_sub_agent", "synthesis")
        graph.add_edge("synthesis", END)
    else:
        # No planner, so nothing to classify and nothing to decompose: every question
        # takes what used to be the simple path, which is also the path the fast path
        # was already sending most of them down for free.
        graph.set_entry_point("retrieve_initial")

    # Simple-question path
    graph.add_edge("retrieve_initial", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grade,
        {
            "rewrite_question": "rewrite_question",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "rewrite_question",
        _route_after_rewrite,
        {
            "retrieve_rewritten": "retrieve_rewritten",
            "end": END,
        },
    )
    graph.add_edge("retrieve_rewritten", "grade_documents")

    return graph.compile()


rag_graph = build_rag_graph()


def _state_from_resume(
    resume_state: dict,
    user_answer: str,
    ctx: ChatRequestContext,
    *,
    resolved=None,
    original_question: str = "",
) -> dict:
    current_resume_state = HitlResumeState.model_validate(resume_state).model_dump()
    refined_question = _refined_question_for_hitl(
        current_resume_state, user_answer, resolved, original_question
    )
    # The conditions in force before the clarification, plus anything the reply itself
    # established. Union rather than replacement: answering "grade 5" narrows further,
    # it does not withdraw the year the user set two turns ago.
    constraints = list(current_resume_state.get("carried_constraints") or [])
    for item in list(getattr(resolved, "constraints", None) or []):
        if item and item.lower() not in {existing.lower() for existing in constraints}:
            constraints.append(item)

    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": refined_question,
        "hitl_resumed": True,
        "hitl_answer": user_answer,
        "hitl_resume_from_status": current_resume_state["retrieval_status"],
        "hitl_resume_from_route": current_resume_state["route"],
        "turn_resolved_question": refined_question,
        "turn_carried_constraints": constraints,
    }
    if current_resume_state.get("complexity"):
        rag_trace["complexity"] = current_resume_state["complexity"]
    if current_resume_state.get("complexity_reason"):
        rag_trace["complexity_reason"] = current_resume_state["complexity_reason"]
    if current_resume_state.get("sub_questions"):
        rag_trace["sub_questions"] = current_resume_state["sub_questions"]
    state = _initial_state(refined_question, ctx)
    state.update({
        "query": refined_question,
        "rewrite_count": current_resume_state["rewrite_count"],
        # The turn being resumed IS a HITL round. Counting it here is what makes the
        # limit mean "per question" rather than "per graph run".
        "hitl_rounds": int(current_resume_state.get("hitl_rounds") or 0) + 1,
        "complexity": current_resume_state.get("complexity"),
        "complexity_reason": current_resume_state.get("complexity_reason"),
        "sub_questions": current_resume_state.get("sub_questions") or [],
        "carried_constraints": constraints,
        # A resumed turn is a continuation by construction — the user is answering a
        # question about a subject already on the table. Offering them a fresh choice of
        # corpus directions here would be the second interruption in a row.
        "is_followup": True,
        "rag_trace": rag_trace,
    })
    return state


def _retrieve_resume_query(state: dict) -> dict:
    _emit(state, "🔎", "Running targeted retrieval using the HITL follow-up", "Skipping complexity classification and sub-question decomposition")
    query = _search_query(state)
    retrieved = retrieve_documents(query, top_k=RETRIEVAL_TOP_K)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    retrieval_failed = retrieve_meta.get("retrieval_mode") == "failed"
    context = _format_docs(results)
    if retrieval_failed:
        _emit(
            state,
            "🚧",
            "Knowledge base temporarily unreachable during HITL retrieval",
            retrieve_meta.get("retrieval_error") or "retrieval backend error",
        )
    else:
        _emit(
            state,
            "🧱",
            "HITL three-tier chunk retrieval",
            (
                f"Leaf level L{retrieve_meta.get('leaf_retrieve_level', 3)} recall, "
                f"candidates {retrieve_meta.get('candidate_k', 0)}"
            ),
        )
        _emit(
            state,
            "🧩",
            "Auto-merging",
            (
                f"Enabled: {bool(retrieve_meta.get('auto_merge_enabled'))}, "
                f"Applied: {bool(retrieve_meta.get('auto_merge_applied'))}, "
                f"Replaced chunks: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        _emit(state, "✅", f"HITL targeted retrieval complete, found {len(results)} snippets", f"Mode: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
    rag_trace = state.get("rag_trace") or {}
    rag_trace.update({
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "retrieved_chunks": results,
        "hitl_targeted_retrieved_chunks": results,
        "hitl_resumed": True,
        "hitl_resume_strategy": "targeted_retrieval",
        "retrieval_stage": "hitl_targeted_retrieval",
        **retrieval_trace_fields(retrieve_meta),
    })
    state.update({
        "query": query,
        "docs": results,
        "context": context,
        "retrieval_failed": retrieval_failed,
        "rag_trace": rag_trace,
    })
    state.update(grade_documents_node(state))
    return state


def resume_rag_from_hitl(
    resume_state: dict,
    user_answer: str,
    ctx: ChatRequestContext,
    *,
    resolved=None,
    original_question: str = "",
) -> dict:
    """Resume a paused RAG run from the HITL breakpoint without re-entering the main graph.

    `resolved` is the user's reply read against the conversation, and `original_question`
    is what THEY asked rather than the query the agent wrote for the tool. Both are
    supplied by the caller because this path used to see neither: it searched for the
    reply concatenated onto the agent's paraphrase, with no access to the conversation
    at all, so a condition set before the clarification could not survive it.
    """
    state = _state_from_resume(
        resume_state, user_answer, ctx, resolved=resolved, original_question=original_question
    )
    _emit(state, "▶️", "Received HITL follow-up, continuing the original RAG flow", user_answer)
    if state["question"] != user_answer:
        _emit(state, "🧭", "Read in context, the question is", state["question"][:90])

    state = _retrieve_resume_query(state)
    if _is_hitl_result(state):
        state["hitl_resume_state"] = _build_hitl_resume_state(state)
    return state


def run_rag_graph(question: str, ctx: ChatRequestContext) -> dict:
    result = rag_graph.invoke(_initial_state(question, ctx))
    if _is_hitl_result(result):
        result["hitl_resume_state"] = _build_hitl_resume_state(result)
    return result
