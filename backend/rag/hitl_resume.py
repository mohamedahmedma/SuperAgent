"""What a retrieval that ended in a clarification carries across to the next message.

When the graph stops to ask the user something, the turn ends there and the next message
resumes it. These functions decide whether a result is such a stop, snapshot what the
resume needs, and work out the question the resumed search should actually run.

Moved out of `pipeline.py` with its behaviour unchanged. Pure: nothing here searches or
calls a model.
"""
from backend.schemas.chat import HitlResumeState


def is_hitl_result(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    trace = result.get("rag_trace") or {}
    status = result.get("retrieval_status") or trace.get("retrieval_status")
    route = result.get("route") or trace.get("route")
    return status in ("needs_clarification", "needs_scope_selection") or route in ("clarify", "scope_select")


def build_hitl_resume_state(result: dict) -> dict:
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


def refined_question_for_hitl(
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
