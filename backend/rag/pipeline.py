from typing import Literal, List, Optional
import logging
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from backend.chat.request_context import ChatRequestContext
from backend.rag.evidence import (
    AssessmentContext,
    Certainty,
    ChunkAssessment,
    EvidenceReport,
)
from backend.prompts import resolve as resolve_prompt
from backend.rag.evidence_view import format_docs
from backend.rag.graph_nodes import (
    ClassifyComplexity,
    FanOutSubQuestions,
    GradeDocuments,
    PrepareSubQuestions,
    RAGState,
    RagSubAgent,
    ResumeRetrieval,
    RetrieveInitial,
    RetrieveRewritten,
    RewriteQuestion,
    Synthesis,
    emit,
    route_after_complexity,
    route_after_grade,
    route_after_rewrite,
)
from backend.rag.hitl_resume import build_hitl_resume_state, is_hitl_result, refined_question_for_hitl
from backend.rag.rerank_assessor import CrossEncoderAssessor
from backend.profiles import get_profile
from backend.schemas.chat import HitlResumeState
from backend.text_normalization import normalize_query
from backend.rag.utils import (
    EVIDENCE_WINDOW_CHARS,
    RETRIEVAL_TOP_K,
    retrieve_documents,
    rewrite_query_once,
    dedupe_documents,
    retrieval_trace_fields,
)

logger = logging.getLogger(__name__)


# Prompts, routing budgets, and the fast-path vocabulary are all profile data.
_PROFILE = get_profile()
_RAG = _PROFILE.rag
_COPY = _PROFILE.user_copy


#: What a response cut off at the output ceiling raises. Imported defensively: the
#: concrete class lives in the OpenAI client, which is a transitive dependency here, and
#: a grading path that fails to IMPORT would be a worse outage than the one it handles.
try:  # pragma: no cover - depends on the installed client
    from openai import LengthFinishReasonError as _OpenAILengthError

    _TRUNCATED_RESPONSE: tuple = (_OpenAILengthError,)
except Exception:  # pragma: no cover - older or absent client
    _TRUNCATED_RESPONSE = ()


def _get_grader_model(*, headroom: bool = False):
    """The grading model, or the same model with room to finish.

    Two instances rather than one, because sampling is fixed when the model is built and
    the retry has to change it. `headroom` is only ever reached from the truncation
    handler in `LLMGraderAssessor.assess`.

    A function rather than a direct container call at each site: it is the seam the
    grading tests substitute, and it keeps "which model grades" one name in this module.
    """
    from backend.composition import default_services

    return default_services().models.grader(headroom=headroom)


def _get_complexity_model():
    """FAST_MODEL is used for question-complexity classification and sub-question decomposition."""
    from backend.composition import default_services

    return default_services().models.planner()


EVIDENCE_GRADE_PROMPT = _RAG.evidence_grade_prompt
COMPLEXITY_PROMPT = _RAG.complexity_prompt

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
        "language": str(getattr(ctx, "language", "") or ""),
        "child_year": str(getattr(ctx, "child_year", "") or ""),
    }


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
            context=format_docs(ctx.docs),
            # Alongside the question, never folded into it. See AssessmentContext.
            constraints=list(ctx.constraints),
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            grade = grader.with_structured_output(EvidenceGrade).invoke(messages)
        except _TRUNCATED_RESPONSE as exc:
            # The call ended at `finish_reason: length`, so there is no JSON to parse and
            # nothing partial worth keeping. Raising here costs the whole turn: grading is
            # the HIGH rung, `evidence_required_certainty` is high, and a ladder that
            # reaches no rung routes to `retrieval_error` — the user is told the knowledge
            # base is having a technical problem while its answer sits in the chunks that
            # were already retrieved.
            #
            # Retried ONCE, with room rather than with different instructions: the prompt
            # and the evidence are unchanged, so a grade that arrives now is the grade the
            # first call was in the middle of making. A second failure falls through, and
            # the honest "try again" is then correct.
            logger.warning(
                "grading was cut off at the output ceiling; retrying with headroom (%s)", exc
            )
            spacious = _get_grader_model(headroom=True)
            if spacious is None:
                raise
            grade = spacious.with_structured_output(EvidenceGrade).invoke(messages)

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


# ---------------------------------------------------------------------------
# The graph's steps, bound to this module
# ---------------------------------------------------------------------------

class _ModuleDependencies:
    """`RagDependencies` read from this module's globals at the moment a step asks.

    Resolved per call rather than captured, so the seams the tests rely on — the stubs a
    re-executed module is loaded with, and `patch.object` on this module afterwards —
    still reach a step inside a graph compiled before either of them happened.
    """

    @property
    def config(self):
        return _RAG

    @property
    def copy(self):
        return _COPY

    @property
    def top_k(self) -> int:
        return RETRIEVAL_TOP_K

    @property
    def evidence_window_chars(self) -> int:
        return EVIDENCE_WINDOW_CHARS

    @property
    def complexity_prompt(self) -> str:
        return COMPLEXITY_PROMPT

    @property
    def complexity_schema(self) -> type:
        return ComplexityResult

    def retrieve_documents(self, query: str, *, top_k: int, language: str) -> dict:
        return retrieve_documents(query, top_k=top_k, language=language)

    def rewrite_query_once(self, question: str) -> dict:
        return rewrite_query_once(question)

    def dedupe_documents(self, docs: List[dict]) -> List[dict]:
        return dedupe_documents(docs)

    def retrieval_trace_fields(self, meta: dict) -> dict:
        return retrieval_trace_fields(meta)

    def model_assessors(self) -> list:
        return [CrossEncoderAssessor(), LLMGraderAssessor()]

    def complexity_model(self):
        return _get_complexity_model()

    def initial_state(self, question: str, ctx: ChatRequestContext, **kwargs) -> dict:
        return _initial_state(question, ctx, **kwargs)


_DEPENDENCIES = _ModuleDependencies()

retrieve_initial = RetrieveInitial(_DEPENDENCIES)
grade_documents_node = GradeDocuments(_DEPENDENCIES)
rewrite_question_node = RewriteQuestion(_DEPENDENCIES)
retrieve_rewritten = RetrieveRewritten(_DEPENDENCIES)
classify_complexity = ClassifyComplexity(_DEPENDENCIES)
prepare_sub_questions = PrepareSubQuestions()
fanout_sub_questions = FanOutSubQuestions(_DEPENDENCIES)
rag_sub_agent = RagSubAgent(retrieve_initial, grade_documents_node)
synthesis = Synthesis(_DEPENDENCIES)
resume_retrieval = ResumeRetrieval(_DEPENDENCIES, grade_documents_node)


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
            route_after_complexity,
            {
                "retrieve_initial": "retrieve_initial",
                "prepare_sub_questions": "prepare_sub_questions",
            },
        )

        graph.add_conditional_edges("prepare_sub_questions", fanout_sub_questions)

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
        route_after_grade,
        {
            "rewrite_question": "rewrite_question",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "rewrite_question",
        route_after_rewrite,
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
    refined_question = refined_question_for_hitl(
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
    emit(state, "▶️", "Received HITL follow-up, continuing the original RAG flow", user_answer)
    if state["question"] != user_answer:
        emit(state, "🧭", "Read in context, the question is", state["question"][:90])

    state = resume_retrieval(state)
    if is_hitl_result(state):
        state["hitl_resume_state"] = build_hitl_resume_state(state)
    return state


def run_rag_graph(question: str, ctx: ChatRequestContext) -> dict:
    """Run the graph for `question`, or hand back this turn's answer for it.

    The memo is the graph's own half of the duplicate-call fix. Middleware collapses the
    copies the model emits in one message (`backend/chat/runtime.py`), which is where
    almost all of them come from — but it can only see one message at a time, and the
    model also re-asks the identical question on a LATER step of the same turn, after
    the first result is already in its context. That one arrives here.

    Re-running would not just be slow. Every pass pays an embedding, a hybrid recall, a
    rerank and an LLM grading call, and because grading is a model call it is not
    deterministic: the same query graded twice can come back `partial` once and
    `no_knowledge` the next, so the turn's behaviour would depend on which copy landed
    last. Answering the second ask from the first result makes a repeated question
    boring rather than expensive, which is what it should be.

    Scoped to the turn — see `ChatRequestContext.remember_retrieval`. Sub-questions do
    not come through here (they are dispatched to `rag_sub_agent` by `Send`), so a
    decomposed question still searches each of its parts properly.
    """
    key = normalize_query(question or "")
    cached = ctx.remembered_retrieval(key) if ctx is not None else None
    if cached is not None:
        logger.info("retrieval memo hit; not re-running the graph for an identical query")
        # Shallow copy so a caller stamping a key onto its result — `hitl_resume_state`
        # below does exactly that — cannot write into what the next caller reads.
        return dict(cached)

    result = rag_graph.invoke(_initial_state(question, ctx))
    if is_hitl_result(result):
        result["hitl_resume_state"] = build_hitl_resume_state(result)
    if ctx is not None:
        ctx.remember_retrieval(key, result)
    return result
