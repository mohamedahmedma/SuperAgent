from typing import Annotated, Any, Literal, TypedDict, List, Optional
import operator
import os
import re
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from pydantic import BaseModel, Field

from backend.chat.request_context import ChatRequestContext
from backend.schemas.chat import HitlResumeState, normalize_rag_sub_trace
from backend.rag.utils import (
    RETRIEVAL_TOP_K,
    retrieve_documents,
    rewrite_query_once,
    dedupe_documents,
    retrieval_trace_fields,
)

API_KEY = os.getenv("ARK_API_KEY")
BASE_URL = os.getenv("BASE_URL")
FAST_MODEL = os.getenv("FAST_MODEL")
GRADE_MODEL = os.getenv("GRADE_MODEL")

_grader_model = None
_complexity_model = None


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
            temperature=0,
            stream_usage=True,
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
            temperature=0,
            stream_usage=True,
        )
    return _complexity_model


EVIDENCE_GRADE_PROMPT = (
    "You are a RAG evidence grader. Based only on the retrieved snippets, judge whether "
    "they are sufficient to answer the user's question. Do not add information that is not in the snippets.\n\n"
    "User question:\n{question}\n\n"
    "Retrieved snippets:\n{context}\n\n"
    "Give a structured result following these rules:\n"
    "- relevance: none means the topic is unrelated; weak means the topic is close but the evidence is weak; strong means the topic is clearly relevant.\n"
    "- answerability: none means it cannot be answered; partial means there are some clues but not enough for a definitive answer; "
    "sufficient means the snippets can directly or jointly support an answer.\n"
    "- ambiguity: missing_slot means a key condition is missing (e.g. role name, version, file type, module name, product line); "
    "multiple_candidates means several candidate directions could all be relevant; none means there is no clear ambiguity.\n"
    "- route may only be one of: answer, rewrite, clarify, scope_select, no_knowledge.\n"
    "  answer: relevance=strong and answerability=sufficient.\n"
    "  rewrite: there is a relevant signal, but the evidence is insufficient, likely due to phrasing, aliasing, or generalization level.\n"
    "  clarify: a key condition is missing and the user needs to provide it.\n"
    "  scope_select: multiple candidate directions are relevant and the user needs to choose one.\n"
    "  no_knowledge: no recall, or the topic is unrelated.\n"
    "- If route is clarify or scope_select, provide hitl_prompt; if options can be listed, provide hitl_options."
)


class EvidenceGrade(BaseModel):
    """Structured evidence grade: judges relevance, answerability, and the next routing step together."""

    relevance: Literal["none", "weak", "strong"] = Field(
        description="Topical relevance between the retrieved snippets and the question"
    )
    answerability: Literal["none", "partial", "sufficient"] = Field(
        description="Whether the retrieved snippets are sufficient to answer the question"
    )
    ambiguity: Literal["none", "missing_slot", "multiple_candidates"] = Field(
        default="none",
        description="Whether the question is missing a condition or has multiple candidate directions"
    )
    route: Literal["answer", "rewrite", "clarify", "scope_select", "no_knowledge"] = Field(
        description="The next routing step"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_slots: List[str] = Field(default_factory=list)
    hitl_prompt: str = ""
    hitl_options: List[str] = Field(default_factory=list)
    reason: str = ""


class ComplexityResult(BaseModel):
    """Question complexity classification result."""

    complexity: Literal["simple", "complex"] = Field(
        description="Question complexity: 'simple' for simple questions, 'complex' for complex questions"
    )
    reason: str = Field(default="", description="Classification reason")
    sub_questions: List[str] = Field(
        default_factory=list,
        description="2-4 independently retrievable sub-questions for a complex question; left empty for a simple question",
        max_length=4,
    )


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    retrieval_status: Optional[str]
    evidence_relevance: Optional[str]
    evidence_answerability: Optional[str]
    evidence_ambiguity: Optional[str]
    evidence_confidence: Optional[float]
    missing_slots: Optional[List[str]]
    hitl_prompt: Optional[str]
    hitl_options: Optional[List[str]]
    rewrite_count: int
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
        complexity=result.get("complexity") or trace.get("complexity"),
        complexity_reason=result.get("complexity_reason") or trace.get("complexity_reason"),
        sub_questions=result.get("sub_questions") or trace.get("sub_questions") or [],
    ).model_dump()


def _refined_question_for_hitl(resume_state: dict, user_answer: str) -> str:
    question = resume_state.get("question") or ""
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
        "evidence_relevance": None,
        "evidence_answerability": None,
        "evidence_ambiguity": None,
        "evidence_confidence": None,
        "missing_slots": [],
        "hitl_prompt": "",
        "hitl_options": [],
        "rewrite_count": 0,
        "rewrite_method": None,
        "rewritten_query": None,
        "step_back_question": None,
        "hyde_document": None,
        "rag_trace": None,
        "complexity": None,
        "complexity_reason": None,
        "sub_questions": None,
        "is_sub_agent": is_sub_agent,
        "sub_results": [],
        "request_context": ctx,
        "rag_step_group": rag_step_group,
        "rag_step_group_label": rag_step_group_label,
    }


def retrieve_initial(state: RAGState) -> RAGState:
    query = state["question"]
    _emit(state, "🔍", "Searching the knowledge base...", "Initial retrieval")
    retrieved = retrieve_documents(query, top_k=RETRIEVAL_TOP_K)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(results)
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
        "rag_trace": rag_trace,
    }


def _route_after_initial(state: RAGState) -> Literal["grade_documents"]:
    return "grade_documents"


def _route_after_grade(state: RAGState) -> Literal["rewrite_question", "end"]:
    if state.get("route") == "rewrite":
        return "rewrite_question"
    return "end"


def _retrieval_status_for_route(route: str, grade: EvidenceGrade) -> str:
    if route == "answer":
        if grade.answerability == "partial":
            return "partial"
        return "answerable"
    if route == "rewrite":
        return "needs_rewrite"
    if route == "clarify":
        return "needs_clarification"
    if route == "scope_select":
        return "needs_scope_selection"
    return "no_knowledge"


def _default_hitl_prompt(route: str, grade: EvidenceGrade) -> str:
    if grade.hitl_prompt:
        return grade.hitl_prompt
    if route == "scope_select":
        return "I found several possibly relevant directions in the knowledge base. Which one are you asking about?"
    if grade.missing_slots:
        return "I found relevant content, but key information is still missing: " + ", ".join(grade.missing_slots)
    return "I found relevant content, but the evidence isn't enough to determine an answer. Please provide more detail about what you're asking."


def _grade_for_no_docs() -> EvidenceGrade:
    return EvidenceGrade(
        relevance="none",
        answerability="none",
        ambiguity="none",
        route="no_knowledge",
        confidence=1.0,
        reason="no_retrieved_documents",
    )


def _resolve_route(grade: EvidenceGrade, state: RAGState) -> str:
    docs = state.get("docs") or []
    rewrite_count = int(state.get("rewrite_count") or 0)
    is_sub_agent = bool(state.get("is_sub_agent"))
    route = grade.route

    if not docs or grade.relevance == "none":
        return "no_knowledge"

    if grade.ambiguity == "missing_slot":
        return "clarify"
    if grade.ambiguity == "multiple_candidates":
        return "scope_select"

    answer_is_supported = grade.relevance == "strong" and grade.answerability == "sufficient"
    if route == "answer" and answer_is_supported:
        return "answer"

    # Sub-questions don't get a second correction pass. Partial evidence is left for synthesis to merge; fully unanswerable stops here.
    if is_sub_agent:
        if grade.answerability in ("partial", "sufficient"):
            return "answer"
        return "no_knowledge"

    if route == "rewrite" and rewrite_count < 1:
        return "rewrite"

    if route == "rewrite" and rewrite_count >= 1:
        if grade.answerability == "partial":
            return "clarify"
        return "no_knowledge"

    if grade.answerability == "partial":
        if rewrite_count < 1:
            return "rewrite"
        return "clarify"

    if answer_is_supported:
        return "answer"

    return "no_knowledge"


def _grade_update(grade: EvidenceGrade, route: str) -> dict:
    status = _retrieval_status_for_route(route, grade)
    hitl_prompt = _default_hitl_prompt(route, grade) if route in ("clarify", "scope_select") else ""
    return {
        "retrieval_status": status,
        "evidence_relevance": grade.relevance,
        "evidence_answerability": grade.answerability,
        "evidence_ambiguity": grade.ambiguity,
        "evidence_confidence": grade.confidence,
        "evidence_reason": grade.reason,
        "missing_slots": grade.missing_slots,
        "hitl_prompt": hitl_prompt,
        "hitl_options": grade.hitl_options,
        "route": route,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    _emit(state, "📊", "Evaluating evidence quality...")
    docs = state.get("docs") or []
    if not docs:
        grade = _grade_for_no_docs()
    else:
        grader = _get_grader_model()
        if not grader:
            raise RuntimeError("GRADE_MODEL is required for evidence grading")
        question = state["question"]
        context = state.get("context", "")
        prompt = EVIDENCE_GRADE_PROMPT.format(question=question, context=context)
        grade = grader.with_structured_output(EvidenceGrade).invoke(
            [{"role": "user", "content": prompt}]
        )

    route = _resolve_route(grade, state)
    grade_update = _grade_update(grade, route)
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(grade_update)

    if route == "answer":
        if grade.answerability == "partial":
            _emit(state, "🟡", "Keeping partially relevant evidence", f"Confidence: {grade.confidence:.2f}")
        else:
            _emit(state, "✅", "Evidence sufficient, returning retrieved snippets", f"Confidence: {grade.confidence:.2f}")
    elif route == "rewrite":
        _emit(state, "⚠️", "Evidence insufficient, will rewrite the query once", f"Confidence: {grade.confidence:.2f}")
    elif route in ("clarify", "scope_select"):
        _emit(state, "❓", "Needs more information from the user", grade_update["hitl_prompt"])
    else:
        _emit(state, "⛔", "No usable evidence found in the knowledge base", grade.reason or "no_knowledge")

    update = {
        "route": route,
        "retrieval_status": grade_update["retrieval_status"],
        "evidence_relevance": grade.relevance,
        "evidence_answerability": grade.answerability,
        "evidence_ambiguity": grade.ambiguity,
        "evidence_confidence": grade.confidence,
        "missing_slots": grade.missing_slots,
        "hitl_prompt": grade_update["hitl_prompt"],
        "hitl_options": grade.hitl_options,
        "rag_trace": rag_trace,
    }

    if route in ("no_knowledge", "clarify", "scope_select"):
        if route in ("clarify", "scope_select") and docs:
            rag_trace["retrieved_chunks"] = []
        update.update({"docs": [], "context": ""})

    return update


def rewrite_question_node(state: RAGState) -> RAGState:
    question = state["question"]
    _emit(state, "✏️", "Rewriting the query...")

    rewrite_count = int(state.get("rewrite_count") or 0)
    if rewrite_count >= 1:
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update({
            "retrieval_status": "no_knowledge",
            "route": "no_knowledge",
            "evidence_reason": "rewrite_budget_exhausted",
        })
        _emit(state, "⛔", "Rewrite budget exhausted, stopping retrieval")
        return {
            "route": "no_knowledge",
            "retrieval_status": "no_knowledge",
            "docs": [],
            "context": "",
            "rag_trace": rag_trace,
        }

    _emit(state, "🧠", "Choosing between Step-back / HyDE rewrite")
    rewrite = rewrite_query_once(question)
    rewrite_method = (rewrite.get("rewrite_method") or "").strip()
    step_back_question = (rewrite.get("step_back_question") or "").strip()
    hyde_document = (rewrite.get("hyde_document") or "").strip()
    rewritten_query = (rewrite.get("rewritten_query") or "").strip()
    if rewrite_method not in ("step_back", "hyde") or not rewritten_query:
        raise ValueError("Query rewriting returned an incomplete result")
    if rewrite_method == "step_back" and (not step_back_question or hyde_document):
        raise ValueError("Step-back rewriting returned an invalid result")
    if rewrite_method == "hyde" and (not hyde_document or step_back_question):
        raise ValueError("HyDE rewriting returned an invalid result")

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
    context = _format_docs(results)
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
    _emit(state, "✅", f"Rewritten retrieval complete, {len(results)} snippets total")
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_method": rewrite_method,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": results,
        "rewrite_retrieved_chunks": results,
        "retrieval_stage": "rewritten",
        **retrieval_trace_fields(retrieve_meta),
    })
    if state.get("step_back_question"):
        rag_trace["step_back_question"] = state["step_back_question"]
    if state.get("hyde_document"):
        rag_trace["hyde_document"] = state["hyde_document"]
    return {"docs": results, "context": context, "rag_trace": rag_trace}


# ---------------------------------------------------------------------------
# Complexity classification & sub-question decomposition
# ---------------------------------------------------------------------------

COMPLEXITY_PROMPT = (
    "You are a question complexity planner. Determine the complexity of the user's question.\n\n"
    "[Simple question]: factual lookups, definition lookups, single-information-point queries, clear "
    "either/or questions, or queries about a specific attribute/parameter/spec.\n"
    "[Complex question]: questions that need cross-document synthesis, multi-angle analysis, comparisons, "
    "multi-step reasoning, or that require multiple information sources to be fully answered.\n\n"
    "User question: {question}\n\n"
    "If it's a complex question, also provide 2-4 non-overlapping, independently retrievable sub-questions; "
    "if it's a simple question, leave sub_questions empty."
)

_SIMPLE_QUERY_MARKERS = (
    "是什么",
    "是谁",
    "哪里",
    "何时",
    "多少",
    "是否",
    "哪个",
    "哪种",
    "属性",
    "参数",
    "规格",
    "定义",
    "含义",
    "what is",
    "who is",
    "where is",
    "when is",
    "how many",
    "which",
)

_COMPLEX_QUERY_MARKERS = (
    "比较",
    "对比",
    "区别",
    "差异",
    "优缺点",
    "优势",
    "劣势",
    "分析",
    "总结",
    "综合",
    "原因",
    "成因",
    "影响",
    "方案",
    "步骤",
    "如何",
    "为什么",
    "以及",
    "同时",
    "并且",
    "和",
    "与",
    "谁更",
    "compare",
    "versus",
    "difference",
    "different",
    "analyze",
    "summarize",
    "trade-off",
    "pros and cons",
    "why ",
    "how ",
    "complex",
)

_QUERY_DIMENSION_MARKERS = (
    "属性",
    "武器",
    "定位",
    "技能",
    "机制",
    "参数",
    "规格",
    "性能",
    "价格",
    "优点",
    "缺点",
    "作用",
)


def _simple_question_fast_path_reason(question: str) -> Optional[str]:
    """Return a reason only when a local rule can confidently classify a simple query."""
    normalized = re.sub(r"\s+", " ", (question or "").strip()).lower()
    if not normalized or len(normalized) > 48:
        return None
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
    # Wh-word + attribute + copula ("what element is X", "which weapon does X use")
    # is a single-fact lookup, but the contiguous "what is" marker cannot match it
    # and such questions are usually longer than the short-intent rule allows.
    # Checked AFTER the complex markers, so comparisons ("what is the difference
    # between ...") are already excluded before reaching here.
    if re.match(r"^(what|which|who|where|when)\s+\w+\s+(is|are|was|were|does|do)\b", normalized):
        return "obvious_simple_fast_path:wh_attribute_question"
    if len(normalized.rstrip("?？。.!！")) <= 18:
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

    prompt = COMPLEXITY_PROMPT.format(question=question)
    result = model.with_structured_output(ComplexityResult).invoke(
        [{"role": "user", "content": prompt}]
    )
    complexity = (result.complexity or "simple").strip().lower()
    reason = (result.reason or "").strip()
    sub_questions = [
        item.strip()
        for item in (result.sub_questions or [])
        if item and item.strip()
    ][:4]
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

    context = _format_docs(deduped)
    if deduped:
        _emit(state, "✅", f"Synthesis complete, {len(deduped)} deduplicated snippets total")
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
    hitl_traces = [
        trace for trace in sub_traces
        if trace.get("retrieval_status") in ("needs_clarification", "needs_scope_selection")
    ]
    hitl_route = None
    hitl_prompt = ""
    hitl_options: List[str] = []
    if not has_docs and hitl_traces:
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
        "evidence_relevance": "strong" if has_docs else "none",
        "evidence_answerability": "partial" if retrieval_status == "partial" else ("sufficient" if has_docs else "none"),
        "evidence_confidence": None,
        "route": "answer" if has_docs else (hitl_route or "no_knowledge"),
        "hitl_prompt": hitl_prompt,
        "hitl_options": hitl_options,
    }

    return {
        "docs": deduped,
        "context": context,
        "route": "answer" if has_docs else (hitl_route or "no_knowledge"),
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

def build_rag_graph():
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("classify_complexity", classify_complexity)
    graph.add_node("prepare_sub_questions", prepare_sub_questions)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_rewritten", retrieve_rewritten)
    graph.add_node("rag_sub_agent", rag_sub_agent)
    graph.add_node("synthesis", synthesis)

    # Entry point: complexity classification
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
    graph.add_edge("rewrite_question", "retrieve_rewritten")
    graph.add_edge("retrieve_rewritten", "grade_documents")

    # Parallel sub-agents → synthesis
    graph.add_edge("rag_sub_agent", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


rag_graph = build_rag_graph()


def _state_from_resume(
    resume_state: dict,
    user_answer: str,
    ctx: ChatRequestContext,
) -> dict:
    current_resume_state = HitlResumeState.model_validate(resume_state).model_dump()
    refined_question = _refined_question_for_hitl(current_resume_state, user_answer)
    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": refined_question,
        "hitl_resumed": True,
        "hitl_answer": user_answer,
        "hitl_resume_from_status": current_resume_state["retrieval_status"],
        "hitl_resume_from_route": current_resume_state["route"],
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
        "complexity": current_resume_state.get("complexity"),
        "complexity_reason": current_resume_state.get("complexity_reason"),
        "sub_questions": current_resume_state.get("sub_questions") or [],
        "rag_trace": rag_trace,
    })
    return state


def _retrieve_resume_query(state: dict) -> dict:
    _emit(state, "🔎", "Running targeted retrieval using the HITL follow-up", "Skipping complexity classification and sub-question decomposition")
    query = state["question"]
    retrieved = retrieve_documents(query, top_k=RETRIEVAL_TOP_K)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(results)
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
        "rag_trace": rag_trace,
    })
    state.update(grade_documents_node(state))
    return state


def resume_rag_from_hitl(
    resume_state: dict,
    user_answer: str,
    ctx: ChatRequestContext,
) -> dict:
    """Resume a paused RAG run from the HITL breakpoint without re-entering the main graph."""
    state = _state_from_resume(resume_state, user_answer, ctx)
    _emit(state, "▶️", "Received HITL follow-up, continuing the original RAG flow", user_answer)

    state = _retrieve_resume_query(state)
    if _is_hitl_result(state):
        state["hitl_resume_state"] = _build_hitl_resume_state(state)
    return state


def run_rag_graph(question: str, ctx: ChatRequestContext) -> dict:
    result = rag_graph.invoke(_initial_state(question, ctx))
    if _is_hitl_result(result):
        result["hitl_resume_state"] = _build_hitl_resume_state(result)
    return result
